"""Minimal PPO pick-cube tasks with state or raw RGB-D observations."""

from __future__ import annotations

import sys
from typing import Callable, Literal

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from piper_mjlab.env import CAMERA_FOVY, CAMERA_POSITION, cube_spec
from piper_mjlab.robots import get_piper_x_robot_cfg
from piper_mjlab.robots.piper_x_constants import get_spec

ObservationMode = Literal["state", "vision"]

TASK_IDS = {
  "state": "PiperX-Pick-Cube-State",
  "vision": "PiperX-Pick-Cube-Vision",
}

_ACTION_SCALE = {
  "joint[1-3]": 0.3,
  "joint[4-6]": 0.5,
  "gripper_joint1": 0.03,
}
_TARGET_CLIP = {
  "joint1": (-2.618, 2.618),
  "joint2": (0.0, 3.142),
  "joint3": (-2.967, 0.0),
  "joint[4-5]": (-1.553, 1.553),
  "joint6": (-2.8715, 2.8715),
  "gripper_joint1": (0.0, 0.05),
}
_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_CNN_MODEL = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


def _vision_robot_spec() -> mujoco.MjSpec:
  """Attach one overhead camera to each replicated robot base."""
  spec = get_spec()
  spec.body("base_link").add_camera(
    name="rl_camera",
    pos=CAMERA_POSITION,
    quat=(1.0, 0.0, 0.0, 0.0),
    fovy=CAMERA_FOVY,
  )
  return spec


def make_pick_cube_env_cfg(
  observation_mode: ObservationMode = "state", play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create the shared PiPER-X lift task.

  The state actor sees object-relative vectors. The vision actor instead sees
  raw overhead RGB-D; its critic retains privileged state during training.
  """
  if observation_mode not in TASK_IDS:
    raise ValueError(f"Unsupported observation mode: {observation_mode}")

  cfg = make_lift_cube_env_cfg()
  cfg.scene.num_envs = 1 if play else (1024 if observation_mode == "state" else 64)
  robot = get_piper_x_robot_cfg()
  if observation_mode == "vision":
    robot.spec_fn = _vision_robot_spec
  cfg.scene.entities = {
    "robot": robot,
    "cube": EntityCfg(spec_fn=cube_spec),
  }

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = _ACTION_SCALE
  action.clip = _TARGET_CLIP

  ee_cfg = SceneEntityCfg("robot", site_names=("grasp_site",))
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"] = ee_cfg
  cfg.rewards["lift"].params["asset_cfg"] = ee_cfg

  for name in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events[name].params["asset_cfg"].geom_names = r"[lr]f_pad"

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if isinstance(sensor, ContactSensorCfg) and sensor.name == "ee_ground_collision":
      sensor.primary.pattern = "link6"

  command = cfg.commands["lift_height"]
  assert isinstance(command, LiftingCommandCfg)
  command.difficulty = "dynamic"
  command.resampling_time_range = (20.0, 20.0)
  command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(0.35, 0.35), y=(0.0, 0.0), z=(0.25, 0.25)
  )
  command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    # Keep the simple demo spawn clear of the arm's reset-pose occlusion.
    x=(0.30, 0.35), y=(0.08, 0.12), z=(0.02, 0.02), yaw=(-3.14, 3.14)
  )
  command.debug_vis = play

  cfg.viewer.body_name = "base_link"
  if observation_mode == "vision":
    camera = CameraSensorCfg(
      name="rl_camera",
      camera_name="robot/rl_camera",
      width=64,
      height=64,
      data_types=("rgb", "depth"),
      enabled_geom_groups=(0, 2),
      use_shadows=False,
      use_textures=True,
    )
    cfg.scene.sensors += (camera,)
    cfg.observations["camera"] = ObservationGroupCfg(
      terms={
        "rgb": ObservationTermCfg(
          func=manipulation_mdp.camera_rgb,
          params={"sensor_name": camera.name},
        ),
        "depth": ObservationTermCfg(
          func=manipulation_mdp.camera_depth,
          params={"sensor_name": camera.name, "cutoff_distance": 1.0},
        ),
      },
      concatenate_dim=0,
    )
    actor = cfg.observations["actor"]
    actor.terms.pop("ee_to_cube")
    actor.terms.pop("cube_to_goal")

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


def make_ppo_cfg(observation_mode: ObservationMode = "state") -> RslRlOnPolicyRunnerCfg:
  """Create the PPO config; only the visual actor needs a CNN."""
  if observation_mode not in TASK_IDS:
    raise ValueError(f"Unsupported observation mode: {observation_mode}")
  vision = observation_mode == "vision"
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name=_CNN_MODEL if vision else "MLPModel",
      cnn_cfg=_CNN_CFG if vision else None,
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      entropy_coef=0.005,
    ),
    obs_groups={
      "actor": ("actor", "camera") if vision else ("actor",),
      "critic": ("critic",),
    },
    experiment_name=f"piperx_pick_cube_{observation_mode}",
    logger="tensorboard",
    upload_model=False,
    clip_actions=1.0,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3000,
  )


for _mode, _task_id in TASK_IDS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=make_pick_cube_env_cfg(_mode),
    play_env_cfg=make_pick_cube_env_cfg(_mode, play=True),
    rl_cfg=make_ppo_cfg(_mode),
    runner_cls=ManipulationOnPolicyRunner,
  )


def _dispatch(main: Callable[[], None]) -> None:
  if len(sys.argv) < 2 or sys.argv[1] not in TASK_IDS:
    raise SystemExit(f"usage: {sys.argv[0]} {{state,vision}} [MJLab options]")
  sys.argv[1] = TASK_IDS[sys.argv[1]]
  main()


def train_main() -> None:
  from mjlab.scripts.train import main

  _dispatch(main)


def play_main() -> None:
  from mjlab.scripts.play import main

  _dispatch(main)
