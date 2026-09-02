"""AgileX PiPER-X constants.

Built from the official URDF (agilexrobotics/agx_arm_urdf, piper_x/), converted
to a MuJoCo-loadable file by scripts/convert_piper_x_urdf.py. Unlike the
menagerie PiPER, the URDF ships no actuators, so we add Builtin position
actuators here. The URDF's virtual "gripper" drive joint was dropped at
conversion; the finger coupling (gripper_joint2 = -gripper_joint1) becomes a
MuJoCo joint equality added in get_spec().

Measured gripper geometry (numerically, from the compiled model; gripper_base
frame, fingers half-open):
- fingers close along +/-y, full stroke 0.05 m per finger (0.10 m opening)
- finger contact faces: x +/-0.0148, z 0.115-0.138, fingertip at z = 0.138
- pad center/half-size are identical in both finger body frames (mirrored
  mounts): pos (0, -0.0115, 0.003), size (0.0148, 0.0115, 0.003)
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PIPER_X_URDF: Path = (
  Path(__file__).resolve().parent.parent / "assets" / "agilex_piper_x" / "piper_x.urdf"
)
assert PIPER_X_URDF.exists()

# Grasp point on gripper_base, at the finger-pad center height. Keeps the
# site-to-fingertip margin small (1.15 cm) so a top-down grasp with the site on
# the cube center leaves the fingertips clear of the ground.
GRASP_SITE_POS = (0.0, 0.0, 0.1265)

_PAD_POS = (0.0, -0.0115, 0.003)
_PAD_SIZE = (0.0148, 0.0115, 0.003)


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PIPER_X_URDF))

  # Name geoms and move them to menagerie-style groups (visual 2, collision 3).
  # The URDF importer leaves everything unnamed with collision in group 0.
  for body in spec.bodies:
    if not body.name or body.name == "world":
      continue
    for g in body.geoms:
      if g.contype == 0 and g.conaffinity == 0:
        g.name = f"{body.name}_visual"
        g.group = 2
      else:
        g.name = f"{body.name}_collision"
        g.group = 3

  # Box pads on the finger contact faces (the mesh collisions get disabled by
  # CollisionCfg below; convex hulls of the whole finger grasp poorly).
  for body_name, pad_name in (("gripper_link1", "lf_pad"), ("gripper_link2", "rf_pad")):
    spec.body(body_name).add_geom(
      name=pad_name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=_PAD_POS,
      size=_PAD_SIZE,
      group=3,
    )

  spec.body("gripper_base").add_site(
    name="grasp_site",
    pos=GRASP_SITE_POS,
    size=[0.005, 0.005, 0.005],
    group=5,
  )

  # Finger coupling: gripper_joint2 = -gripper_joint1 (URDF mimic tags are
  # dropped by the importer). Stiff solref: the position actuator can slam the
  # fingers with the full 10 N forcerange, and the default equality lets the
  # coupling get torn centimeters apart, crossing the fingers.
  eq = spec.add_equality()
  eq.type = mujoco.mjtEq.mjEQ_JOINT
  eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
  eq.name1 = "gripper_joint2"
  eq.name2 = "gripper_joint1"
  eq.data[:5] = [0.0, -1.0, 0.0, 0.0, 0.0]
  eq.solref = [0.005, 1.0]

  # Fingers can momentarily overlap when slammed shut; if their pads collide
  # they wedge against each other in the crossed state and the gripper locks
  # up permanently. Excluding finger-finger contact removes that failure mode
  # (cube and ground contacts are unaffected).
  ex = spec.add_exclude()
  ex.bodyname1 = "gripper_link1"
  ex.bodyname2 = "gripper_link2"

  # Joint passive properties (URDF carries none; menagerie-piper-like values).
  for j in spec.joints:
    if j.name.startswith("joint"):
      j.frictionloss = 0.3
      j.armature = 0.005
    elif j.name.startswith("gripper_joint"):
      # Zero-armature fingers overshoot their limits under full actuator
      # force; armature + damping + stiff limit constraints keep them inside
      # the physical stroke.
      j.armature = 0.005
      j.damping[:] = 1.0
      j.solref_limit = [0.005, 1.0]
  return spec


##
# Actuator config.
##

# Position actuators added by MJLab (the URDF has none). The gains are stable
# simulation gains, not hardware-identified motor parameters. Effort limits
# come from the URDF. Only gripper_joint1 is actuated; joint2 follows through
# the equality.
PIPER_X_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("joint[1-3]",), stiffness=80, damping=5, effort_limit=100
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("joint4",), stiffness=40, damping=5, effort_limit=100
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("joint[5-6]",), stiffness=10, damping=1.5, effort_limit=100
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("gripper_joint1",), stiffness=40, damping=5, effort_limit=10
  ),
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=PIPER_X_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint2": 1.57,
    "joint3": -1.35,
    "gripper_joint1": 0.025,
    "gripper_joint2": -0.025,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Gripper-only collisions: wrist/gripper housing + pads. Finger mesh hulls and
# all arm links are disabled.
GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", "[lr]f_pad"),
  contype={
    "(link6|gripper_base)_collision": 1,
    "[lr]f_pad": 1,
    ".*_collision": 0,
  },
  conaffinity={
    "(link6|gripper_base)_collision": 1,
    "[lr]f_pad": 1,
    ".*_collision": 0,
  },
  condim={
    "[lr]f_pad": 6,
    ".*_collision": 3,
  },
  friction={
    "[lr]f_pad": (1, 5e-3, 5e-4),
    ".*_collision": (0.6,),
  },
  solref={
    "[lr]f_pad": (0.01, 1),
  },
  priority={
    "[lr]f_pad": 1,
    ".*_collision": 0,
  },
)

##
# Final config.
##


def get_piper_x_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )
