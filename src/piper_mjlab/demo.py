"""Scripted state and RGB-D pick-and-place demos."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.native import NativeMujocoViewer
from mjlab.viewer.viser import ViserPlayViewer

from piper_mjlab.env import (
  CAMERA_FOVY,
  CAMERA_HEIGHT,
  CAMERA_POSITION,
  CAMERA_WIDTH,
  CUBE_HALF_SIZE,
  GOAL_POSITION,
  ObservationMode,
  make_env_cfg,
)

ViewerName = Literal["auto", "native", "viser", "none"]


@dataclass(frozen=True)
class PinholeCamera:
  position: tuple[float, float, float] = CAMERA_POSITION
  fovy: float = CAMERA_FOVY
  width: int = CAMERA_WIDTH
  height: int = CAMERA_HEIGHT

  @property
  def focal_length(self) -> float:
    return 0.5 * self.height / math.tan(math.radians(self.fovy) / 2.0)


def estimate_red_cube(
  rgb: torch.Tensor,
  depth: torch.Tensor,
  camera: PinholeCamera = PinholeCamera(),
) -> tuple[torch.Tensor, torch.Tensor]:
  """Estimate the red cube center from fixed overhead RGB-D.

  Args:
    rgb: Normalized image ``[B, 3, H, W]``.
    depth: Planar metric depth ``[B, 1, H, W]``.

  Returns:
    Estimated world-frame cube centers ``[B, 3]`` and a validity mask ``[B]``.
  """
  if rgb.ndim != 4 or rgb.shape[1] != 3:
    raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(rgb.shape)}")
  if depth.shape != (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]):
    raise ValueError("Depth must have shape [B,1,H,W] matching RGB")

  red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
  planar_depth = depth[:, 0]
  mask = (
    (red > 0.55)
    & (red > 1.8 * green)
    & (red > 1.8 * blue)
    # Keep the tabletop object and reject red robot pixels above it.
    & (planar_depth > 0.65)
    & (planar_depth < 0.80)
  )

  weights = mask.float()
  count = weights.sum(dim=(1, 2))
  valid = count >= 4
  safe_count = count.clamp_min(1.0)

  rows = torch.arange(rgb.shape[2], device=rgb.device, dtype=depth.dtype)
  cols = torch.arange(rgb.shape[3], device=rgb.device, dtype=depth.dtype)
  v = (weights * rows[None, :, None]).sum(dim=(1, 2)) / safe_count + 0.5
  u = (weights * cols[None, None, :]).sum(dim=(1, 2)) / safe_count + 0.5
  d = (weights * planar_depth).sum(dim=(1, 2)) / safe_count

  focal = camera.focal_length
  x = camera.position[0] + (u - camera.width / 2.0) * d / focal
  y = camera.position[1] + (camera.height / 2.0 - v) * d / focal
  z = camera.position[2] - d - CUBE_HALF_SIZE
  estimate = torch.stack((x, y, z), dim=-1)
  estimate[~valid] = torch.nan
  return estimate, valid


class PickPlacePolicy:
  """A deliberately small Cartesian waypoint state machine."""

  _PHASES = (
    "approach",
    "descend",
    "close",
    "lift",
    "transfer",
    "place",
    "open",
    "retreat",
    "done",
  )
  _MOTION_PHASES = {"approach", "descend", "lift", "transfer", "place", "retreat"}

  def __init__(self, env: ManagerBasedRlEnv, observation_mode: ObservationMode):
    self.env = env
    self.observation_mode = observation_mode
    self.camera = PinholeCamera()
    self.reset()

  def reset(self) -> None:
    self.phase_index = 0
    self.phase_steps = 0
    self.stable_steps = 0
    self.pick_position: torch.Tensor | None = None
    self.command_position: torch.Tensor | None = None
    self._reported_result = False
    print(f"[demo] observation={self.observation_mode}; phase=approach")

  @property
  def phase(self) -> str:
    return self._PHASES[self.phase_index]

  def _observe_cube(self, obs: dict[str, Any]) -> torch.Tensor | None:
    if self.observation_mode == "state":
      return obs["object"]["cube_position"][0]
    estimate, valid = estimate_red_cube(
      obs["vision"]["rgb"], obs["vision"]["depth"], self.camera
    )
    return estimate[0] if bool(valid[0]) else None

  def _target(self) -> torch.Tensor:
    assert self.pick_position is not None
    pick = self.pick_position
    goal = torch.tensor(GOAL_POSITION, device=pick.device, dtype=pick.dtype)
    if self.phase == "approach":
      return torch.stack((pick[0], pick[1], pick[2] + 0.16))
    if self.phase in ("descend", "close"):
      # The grasp site is centered along the finger pads. A small positive
      # clearance keeps the pads off the table under gravity/PD tracking error.
      # Low poses also need a small outward compensation for static PD sag.
      xy = pick[:2] * (1.0 + 0.012 / torch.linalg.vector_norm(pick[:2]))
      return torch.stack((xy[0], xy[1], pick[2] + 0.015))
    if self.phase == "lift":
      xy = pick[:2] * (1.0 + 0.012 / torch.linalg.vector_norm(pick[:2]))
      return torch.stack((xy[0], xy[1], pick.new_tensor(0.18)))
    if self.phase == "transfer":
      return torch.stack((goal[0], goal[1], goal.new_tensor(0.18)))
    if self.phase in ("place", "open"):
      xy = goal[:2] * (1.0 + 0.012 / torch.linalg.vector_norm(goal[:2]))
      return torch.stack((xy[0], xy[1], goal[2]))
    return torch.stack((goal[0], goal[1], goal.new_tensor(0.18)))

  @staticmethod
  def _top_down_quaternion(position: torch.Tensor) -> torch.Tensor:
    """World-frame tool orientation with local +Z down and radial yaw."""
    yaw = torch.atan2(position[1], position[0])
    half_yaw = yaw / 2.0
    zero = half_yaw.new_zeros(())
    return torch.stack((zero, -torch.sin(half_yaw), torch.cos(half_yaw), zero))

  def _advance(self) -> None:
    if self.phase_index >= len(self._PHASES) - 1:
      return
    self.phase_index += 1
    self.phase_steps = 0
    self.stable_steps = 0
    print(f"[demo] phase={self.phase}")

  def _update_phase(self, ee_position: torch.Tensor, target: torch.Tensor) -> None:
    self.phase_steps += 1
    if self.phase in self._MOTION_PHASES:
      error = torch.linalg.vector_norm(ee_position - target).item()
      self.stable_steps = self.stable_steps + 1 if error < 0.018 else 0
      if self.stable_steps >= 12 or self.phase_steps >= 350:
        self._advance()
    elif self.phase in ("close", "open") and self.phase_steps >= 75:
      self._advance()

  def _slew_limit(self, target: torch.Tensor, max_step: float = 0.002) -> torch.Tensor:
    """Limit Cartesian target motion to avoid impulsive contact forces."""
    if self.command_position is None:
      self.command_position = target.clone()
      return self.command_position
    delta = target - self.command_position
    distance = torch.linalg.vector_norm(delta)
    scale = torch.clamp(target.new_tensor(max_step) / distance.clamp_min(1e-9), max=1.0)
    self.command_position = self.command_position + scale * delta
    return self.command_position

  def _report_result(self) -> None:
    if self.phase != "done" or self._reported_result:
      return
    # Ground truth is read here only for the final metric, never for vision control.
    cube = self.env.scene["cube"].data.root_link_pos_w[0]
    goal = torch.tensor(GOAL_POSITION, device=cube.device, dtype=cube.dtype)
    error = torch.linalg.vector_norm(cube - goal).item()
    result = "SUCCESS" if error < 0.05 else "FAILED"
    print(f"[demo] {result}: final cube error={error:.3f} m")
    self._reported_result = True

  @torch.no_grad()
  def __call__(self, obs: dict[str, Any]) -> torch.Tensor:
    # Vision is used only to locate the cube before closure. Thereafter the
    # stored pick point and known place goal are sufficient.
    if self.phase in ("approach", "descend"):
      measured = self._observe_cube(obs)
      if measured is not None:
        measured = measured.clone()
        measured[2] = CUBE_HALF_SIZE
        self.pick_position = (
          measured
          if self.pick_position is None
          else 0.8 * self.pick_position + 0.2 * measured
        )

    if self.pick_position is None:
      # Wait with the gripper open until a valid image/state sample arrives.
      ee = obs["proprio"]["grasp_position"][0]
      quat = self._top_down_quaternion(ee)
      return torch.cat((ee, quat, ee.new_tensor([0.05]))).unsqueeze(0)

    ee = obs["proprio"]["grasp_position"][0]
    target = self._target()
    self._update_phase(ee, target)
    if self.command_position is None:
      self.command_position = ee.clone()
    target = self._slew_limit(self._target())
    quat = self._top_down_quaternion(target)
    gripper = 0.0 if self.phase in ("close", "lift", "transfer", "place") else 0.05
    self._report_result()
    return torch.cat((target, quat, target.new_tensor([gripper]))).unsqueeze(0)


def run_demo(
  observation_mode: ObservationMode,
  viewer: ViewerName = "auto",
  device: str | None = None,
  steps: int | None = None,
) -> None:
  if device is None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(make_env_cfg(observation_mode), device=device)
  env.reset(seed=0)
  policy = PickPlacePolicy(env, observation_mode)

  if viewer == "auto":
    viewer = "native" if os.environ.get("DISPLAY") else "viser"

  try:
    if viewer == "none":
      for _ in range(steps or 2_200):
        obs = env.get_observations()
        env.step(policy(obs))
        if policy.phase == "done":
          break
      policy._report_result()
    elif viewer == "native":
      NativeMujocoViewer(env, policy).run(num_steps=steps)
    elif viewer == "viser":
      ViserPlayViewer(env, policy).run(num_steps=steps)
    else:
      raise ValueError(f"Unsupported viewer: {viewer}")
  finally:
    env.close()


def _parse_args(argv: Sequence[str] | None, default_mode: ObservationMode | None):
  parser = argparse.ArgumentParser(description=__doc__)
  if default_mode is None:
    parser.add_argument("--observation", choices=("state", "vision"), default="state")
  parser.add_argument(
    "--viewer", choices=("auto", "native", "viser", "none"), default="auto"
  )
  parser.add_argument("--device", default=None, help="MJLab device, e.g. cuda:0")
  parser.add_argument("--steps", type=int, default=None, help="Stop after N control steps")
  args = parser.parse_args(argv)
  args.observation = default_mode or args.observation
  return args


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv, None)
  run_demo(args.observation, args.viewer, args.device, args.steps)


def state_main() -> None:
  args = _parse_args(None, "state")
  run_demo("state", args.viewer, args.device, args.steps)


def vision_main() -> None:
  args = _parse_args(None, "vision")
  run_demo("vision", args.viewer, args.device, args.steps)


if __name__ == "__main__":
  main()
