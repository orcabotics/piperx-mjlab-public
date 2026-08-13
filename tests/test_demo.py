import math

import torch

from piper_mjlab.demo import PinholeCamera, estimate_red_cube
from piper_mjlab.env import make_env_cfg


def test_rgbd_cube_projection() -> None:
  camera = PinholeCamera(position=(0.0, 0.0, 0.8), fovy=60, width=8, height=6)
  rgb = torch.zeros((1, 3, 6, 8))
  depth = torch.full((1, 1, 6, 8), 0.9)
  rgb[:, 0, 2:4, 4:6] = 1.0
  depth[:, :, 2:4, 4:6] = 0.70

  estimate, valid = estimate_red_cube(rgb, depth, camera)
  focal = 0.5 * camera.height / math.tan(math.radians(camera.fovy) / 2)
  expected_x = (5.0 - camera.width / 2) * 0.70 / focal

  assert valid.tolist() == [True]
  assert torch.allclose(
    estimate[0], torch.tensor([expected_x, 0.0, 0.08]), atol=1e-6
  )


def test_vision_config_hides_privileged_object_position() -> None:
  state_cfg = make_env_cfg("state")
  vision_cfg = make_env_cfg("vision")

  assert "object" in state_cfg.observations
  assert "vision" not in state_cfg.observations
  assert "vision" in vision_cfg.observations
  assert "object" not in vision_cfg.observations
  assert tuple(vision_cfg.actions) == ("ee_pose", "gripper")
