"""A small, deterministic PiPER-X pick-and-place environment."""

from typing import Literal

import mujoco
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import (
  DifferentialIKActionCfg,
  JointPositionActionCfg,
)
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from piper_mjlab.robots.piper_x_constants import get_piper_x_robot_cfg

ObservationMode = Literal["state", "vision"]

CUBE_HALF_SIZE = 0.02
CUBE_START = (0.32, 0.10, CUBE_HALF_SIZE)
GOAL_POSITION = (0.40, -0.10, CUBE_HALF_SIZE + 0.002)

CAMERA_NAME = "overhead_camera"
CAMERA_POSITION = (0.35, 0.0, 0.80)
CAMERA_FOVY = 50.0
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120


def cube_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(CUBE_HALF_SIZE,) * 3,
    mass=0.05,
    rgba=(0.9, 0.05, 0.05, 1.0),
  )
  return spec


def _goal_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal")
  body.add_geom(
    name="goal_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.04, 0.04, 0.001),
    rgba=(0.1, 0.8, 0.2, 0.7),
    contype=0,
    conaffinity=0,
  )
  return spec


def _joint_position(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.joint_pos


def _joint_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.joint_vel


def _grasp_position(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  site_ids, _ = robot.find_sites("grasp_site")
  return robot.data.site_pos_w[:, site_ids].squeeze(1)


def _cube_position(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["cube"].data.root_link_pos_w


def _camera_rgb(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor: CameraSensor = env.scene[CAMERA_NAME]
  assert sensor.data.rgb is not None
  return sensor.data.rgb.permute(0, 3, 1, 2).float() / 255.0


def _camera_depth(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor: CameraSensor = env.scene[CAMERA_NAME]
  assert sensor.data.depth is not None
  return sensor.data.depth.permute(0, 3, 1, 2)


def make_env_cfg(observation_mode: ObservationMode = "state") -> ManagerBasedRlEnvCfg:
  """Build the one-environment demo config.

  The arm action is a Cartesian grasp-site pose plus one gripper opening. State
  mode exposes the cube pose; vision mode replaces it with overhead RGB-D.
  """
  if observation_mode not in ("state", "vision"):
    raise ValueError(f"Unsupported observation mode: {observation_mode}")

  robot = get_piper_x_robot_cfg()
  robot.init_state = EntityCfg.InitialStateCfg(
    # A feasible, nearly top-down pose beside the cube, leaving it visible to
    # the overhead camera at reset.
    joint_pos={
      "joint1": -0.303,
      "joint2": 1.652,
      "joint3": -1.550,
      "joint4": 1.553,
      "joint5": 0.0,
      "joint6": 0.0,
      "gripper_joint1": 0.04,
      "gripper_joint2": -0.04,
    },
    joint_vel={".*": 0.0},
  )

  observations = {
    "proprio": ObservationGroupCfg(
      terms={
        "joint_position": ObservationTermCfg(func=_joint_position),
        "joint_velocity": ObservationTermCfg(func=_joint_velocity),
        "grasp_position": ObservationTermCfg(func=_grasp_position),
      },
      enable_corruption=False,
      concatenate_terms=False,
    )
  }
  sensors: tuple[CameraSensorCfg, ...] = ()
  if observation_mode == "state":
    observations["object"] = ObservationGroupCfg(
      terms={"cube_position": ObservationTermCfg(func=_cube_position)},
      enable_corruption=False,
      concatenate_terms=False,
    )
  else:
    sensors = (
      CameraSensorCfg(
        name=CAMERA_NAME,
        pos=CAMERA_POSITION,
        quat=(1.0, 0.0, 0.0, 0.0),  # MuJoCo cameras look along local -Z.
        fovy=CAMERA_FOVY,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        data_types=("rgb", "depth"),
        enabled_geom_groups=(0, 2),
        use_shadows=False,
        use_textures=True,
      ),
    )
    observations["vision"] = ObservationGroupCfg(
      terms={
        "rgb": ObservationTermCfg(func=_camera_rgb),
        "depth": ObservationTermCfg(func=_camera_depth),
      },
      enable_corruption=False,
      concatenate_terms=False,
    )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.0,
      entities={
        "robot": robot,
        "cube": EntityCfg(
          spec_fn=cube_spec,
          init_state=EntityCfg.InitialStateCfg(pos=CUBE_START),
        ),
        "goal": EntityCfg(
          spec_fn=_goal_spec,
          init_state=EntityCfg.InitialStateCfg(
            pos=(GOAL_POSITION[0], GOAL_POSITION[1], 0.001)
          ),
        ),
      },
      sensors=sensors,
    ),
    observations=observations,
    actions={
      "ee_pose": DifferentialIKActionCfg(
        entity_name="robot",
        actuator_names=("joint[1-6]",),
        frame_type="site",
        frame_name="grasp_site",
        use_relative_mode=False,
        orientation_weight=0.5,
        damping=0.05,
        # The target offset must be large enough for the position PD to create
        # gravity-support torque; 0.05 rad caps J2/J3 at only about 4 Nm.
        max_dq=0.50,
        posture_weight=0.01,
      ),
      "gripper": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=("gripper_joint1",),
        scale=1.0,
        use_default_offset=False,
        clip={"gripper_joint1": (0.0, 0.05)},
      ),
    },
    sim=SimulationCfg(
      nconmax=64,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.3,
      elevation=-25.0,
      azimuth=135.0,
    ),
    decimation=4,
    episode_length_s=60.0,
    auto_reset=False,
  )
