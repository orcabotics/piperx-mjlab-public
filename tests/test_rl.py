from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import list_tasks

from piper_mjlab.rl import TASK_IDS, make_pick_cube_env_cfg, make_ppo_cfg


def test_pick_cube_tasks_are_registered() -> None:
  assert set(TASK_IDS.values()) <= set(list_tasks())


def test_state_task_uses_object_state() -> None:
  cfg = make_pick_cube_env_cfg("state")
  assert cfg.scene.num_envs == 1024
  assert "camera" not in cfg.observations
  assert "ee_to_cube" in cfg.observations["actor"].terms
  assert "cube_to_goal" in cfg.observations["actor"].terms
  assert make_ppo_cfg("state").actor.class_name == "MLPModel"


def test_vision_actor_uses_raw_rgbd_not_object_state() -> None:
  cfg = make_pick_cube_env_cfg("vision")
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  cameras = [
    sensor for sensor in cfg.scene.sensors or () if sensor.name == "rl_camera"
  ]

  assert cfg.scene.num_envs == 64
  assert "ee_to_cube" not in actor_terms and "cube_to_goal" not in actor_terms
  assert "ee_to_cube" in critic_terms and "cube_to_goal" in critic_terms
  assert len(cameras) == 1 and isinstance(cameras[0], CameraSensorCfg)
  assert cameras[0].camera_name == "robot/rl_camera"
  assert cameras[0].data_types == ("rgb", "depth")

  agent = make_ppo_cfg("vision")
  assert agent.obs_groups["actor"] == ("actor", "camera")
  assert agent.obs_groups["critic"] == ("critic",)
  assert "SpatialSoftmaxCNNModel" in agent.actor.class_name
