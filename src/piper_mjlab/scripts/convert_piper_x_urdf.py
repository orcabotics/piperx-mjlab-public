"""Convert the official AgileX piper_x URDF (+ gripper xacro) to an MJCF asset.

Source: https://github.com/agilexrobotics/agx_arm_urdf (piper_x/). The gripper
xacro is a plain include of the arm URDF plus flange/gripper elements, so we
expand it textually instead of requiring ROS xacro:

- merge arm URDF + gripper additions
- drop the virtual "gripper" drive joint and its massless "gripper_link"
  (MuJoCo rejects massless jointed bodies; the finger coupling becomes an
  equality constraint added later in piper_x_constants.get_spec())
- strip <mimic> tags (ignored by the MuJoCo URDF importer anyway)
- rewrite package:// mesh paths to local files, pointing visuals at the STL
  meshes (MuJoCo cannot load .dae)
- replace upstream inertia tensors which exceed the mass/COM/STL envelope's
  mathematical upper bound with homogeneous closed-STL mass properties
- prepend the <mujoco> extension block for the importer

Usage: python -m piper_mjlab.scripts.convert_piper_x_urdf <path-to-agx_arm_urdf/piper_x>
Writes assets/agilex_piper_x/{piper_x.urdf,meshes/*.stl}.
"""

import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "agilex_piper_x"

# These upstream tensors are physically impossible for their declared mass,
# COM, and complete mesh envelope.  For example, link6 declares a 0.466 m
# radius of gyration for a 0.038 m diameter, 7 g part.  Keep the upstream mass
# and COM (which determine gravity torque), but use the official closed STL as
# a reproducible inertia-shape prior.  Other upstream tensors are retained
# because they pass the same envelope check and may encode non-uniform motors,
# transmissions, and electronics better than a homogeneous mesh can.
MESH_INERTIA_LINKS = frozenset({
  "link2",
  "link3",
  "link4",
  "link6",
  "gripper_link1",
  "gripper_link2",
})

MUJOCO_EXTENSION = """\
    <mujoco>
        <compiler meshdir="meshes" balanceinertia="true" discardvisual="false" fusestatic="false"/>
    </mujoco>
"""


def _homogeneous_stl_inertia_per_kg(path: Path) -> np.ndarray:
  """Return a binary closed STL's inertia tensor about its volume centroid."""
  raw = path.read_bytes()
  if len(raw) < 84:
    raise ValueError(f"STL is truncated: {path}")
  triangle_count = struct.unpack_from("<I", raw, 80)[0]
  expected_size = 84 + triangle_count * 50
  if len(raw) != expected_size:
    raise ValueError(f"Expected a binary STL with {expected_size} bytes: {path}")
  record = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
  ])
  vertices = np.frombuffer(
    raw,
    dtype=record,
    count=triangle_count,
    offset=84,
  )["vertices"].astype(float)
  a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
  tetrahedron_volume = np.einsum(
    "ij,ij->i", a, np.cross(b, c)
  ) / 6.0
  volume = float(np.sum(tetrahedron_volume))
  if not np.isfinite(volume) or volume <= 0.0:
    raise ValueError(f"STL is not a consistently oriented closed solid: {path}")
  centroid = np.sum(
    tetrahedron_volume[:, None] * (a + b + c) / 4.0,
    axis=0,
  ) / volume

  second_moment = np.zeros((3, 3))
  for row in range(3):
    second_moment[row, row] = np.sum(
      tetrahedron_volume / 10.0 * (
        a[:, row] ** 2
        + b[:, row] ** 2
        + c[:, row] ** 2
        + a[:, row] * b[:, row]
        + a[:, row] * c[:, row]
        + b[:, row] * c[:, row]
      )
    )
    for column in range(row):
      second_moment[row, column] = second_moment[column, row] = np.sum(
        tetrahedron_volume / 20.0 * (
          2.0 * (
            a[:, row] * a[:, column]
            + b[:, row] * b[:, column]
            + c[:, row] * c[:, column]
          )
          + a[:, row] * b[:, column]
          + b[:, row] * a[:, column]
          + a[:, row] * c[:, column]
          + c[:, row] * a[:, column]
          + b[:, row] * c[:, column]
          + c[:, row] * b[:, column]
        )
      )
  inertia_at_origin = (
    np.trace(second_moment) * np.eye(3) - second_moment
  )
  centroid_shift = volume * (
    float(centroid @ centroid) * np.eye(3)
    - np.outer(centroid, centroid)
  )
  return (inertia_at_origin - centroid_shift) / volume


def _replace_nonphysical_inertias(urdf: str) -> str:
  for link_name in MESH_INERTIA_LINKS:
    block_pattern = re.compile(
      rf'(<link name="{re.escape(link_name)}">.*?'
      rf'<mass value="([^"]+)"\s*/>.*?)(<inertia [^>]*/>)',
      re.DOTALL,
    )
    match = block_pattern.search(urdf)
    if match is None:
      raise ValueError(f"Missing inertial block for {link_name}")
    mass_kg = float(match.group(2))
    tensor = mass_kg * _homogeneous_stl_inertia_per_kg(
      ASSET_DIR / "meshes" / f"{link_name}.stl"
    )
    inertia = (
      f'<inertia ixx="{tensor[0, 0]:.12g}" '
      f'ixy="{tensor[0, 1]:.12g}" ixz="{tensor[0, 2]:.12g}" '
      f'iyy="{tensor[1, 1]:.12g}" iyz="{tensor[1, 2]:.12g}" '
      f'izz="{tensor[2, 2]:.12g}"/>'
    )
    urdf = urdf[:match.start(3)] + inertia + urdf[match.end(3):]
  return urdf


def main(src_dir: Path) -> None:
  arm = (src_dir / "urdf" / "piper_x_description.urdf").read_text()
  grip = (src_dir / "urdf" / "piper_x_with_gripper_description.xacro").read_text()

  # Gripper additions = everything between the xacro include and </robot>.
  m = re.search(r"<xacro:include[^>]*/>\n(.*)</robot>", grip, re.DOTALL)
  assert m, "unexpected xacro layout"
  additions = m.group(1)

  # Drop the virtual drive joint and its massless link.
  additions = additions.replace('<link name="gripper_link"/>', "")
  additions = re.sub(
    r'<joint name="gripper" type="prismatic">.*?</joint>', "", additions, flags=re.DOTALL
  )
  additions = re.sub(r"<mimic[^>]*/>", "", additions)

  merged = arm.replace("</robot>", additions + "</robot>")
  merged = merged.replace("<robot", "<robot", 1)
  # Insert the mujoco extension right after the opening <robot ...> tag.
  merged = re.sub(r"(<robot[^>]*>)", r"\1\n" + MUJOCO_EXTENSION, merged, count=1)

  # package://.../meshes/dae/foo.dae -> foo.stl ; package://.../meshes/foo.stl -> foo.stl
  merged = re.sub(r'filename="package://[^"]*/meshes/dae/([^"/]+)\.dae"', r'filename="\1.stl"', merged)
  merged = re.sub(r'filename="package://[^"]*/meshes/([^"/]+)"', r'filename="\1"', merged)

  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  (ASSET_DIR / "meshes").mkdir(exist_ok=True)
  for stl in (src_dir / "meshes").glob("*.stl"):
    shutil.copy(stl, ASSET_DIR / "meshes" / stl.name)
  merged = _replace_nonphysical_inertias(merged)
  out = ASSET_DIR / "piper_x.urdf"
  out.write_text(merged)
  print(f"wrote {out}")

  # Sanity: importable by MuJoCo.
  import mujoco

  spec = mujoco.MjSpec.from_file(str(out))
  model = spec.compile()
  print(f"compiled OK: nq={model.nq} nbody={model.nbody} ngeom={model.ngeom}")
  print("joints:", [model.joint(i).name for i in range(model.njnt)])


if __name__ == "__main__":
  main(Path(sys.argv[1]))
