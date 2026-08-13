from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from piper_mjlab.scripts.convert_piper_x_urdf import (
  MESH_INERTIA_LINKS,
  _homogeneous_stl_inertia_per_kg,
)


ASSET_DIRECTORY = (
  Path(__file__).resolve().parents[1]
  / "src"
  / "piper_mjlab"
  / "assets"
  / "agilex_piper_x"
)
URDF_PATH = ASSET_DIRECTORY / "piper_x.urdf"


def _links() -> dict[str, ET.Element]:
  root = ET.parse(URDF_PATH).getroot()
  return {link.get("name"): link for link in root.findall("link")}


def _inertia_tensor(link: ET.Element) -> np.ndarray:
  values = {
    name: float(value)
    for name, value in link.find("inertial/inertia").attrib.items()
  }
  return np.asarray((
    (values["ixx"], values["ixy"], values["ixz"]),
    (values["ixy"], values["iyy"], values["iyz"]),
    (values["ixz"], values["iyz"], values["izz"]),
  ))


def _binary_stl_vertices(path: Path) -> np.ndarray:
  raw = path.read_bytes()
  triangle_count = struct.unpack_from("<I", raw, 80)[0]
  assert len(raw) == 84 + 50 * triangle_count
  record = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
  ])
  return np.frombuffer(
    raw,
    dtype=record,
    count=triangle_count,
    offset=84,
  )["vertices"].reshape(-1, 3)


def test_mass_matches_catalog_at_the_assembly_level() -> None:
  links = _links()

  def total_mass(names: tuple[str, ...]) -> float:
    return sum(
      float(links[name].find("inertial/mass").get("value"))
      for name in names
    )

  arm_mass_kg = total_mass((
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "flange_link",
  ))
  gripper_mass_kg = total_mass((
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
  ))

  # AgileX specifies 4.5 kg for PiPER-X and 0.5 kg for its gripper.  The
  # link-level CAD export is 3.4% below the rounded arm catalog value and the
  # gripper decomposition agrees exactly; neither supports mass rescaling.
  assert arm_mass_kg == pytest.approx(4.5, rel=0.04)
  assert gripper_mass_kg == pytest.approx(0.5)


def test_every_inertia_is_positive_and_fits_its_mesh_envelope() -> None:
  for name, link in _links().items():
    inertial = link.find("inertial")
    visual_mesh = link.find("visual/geometry/mesh")
    if inertial is None or visual_mesh is None:
      continue
    mass_kg = float(inertial.find("mass").get("value"))
    center_of_mass = np.fromstring(
      inertial.find("origin").get("xyz"),
      sep=" ",
    )
    inertia = _inertia_tensor(link)
    eigenvalues = np.linalg.eigvalsh(inertia)
    assert np.all(eigenvalues > 0.0), name
    assert eigenvalues[-1] <= eigenvalues[0] + eigenvalues[1] + 1e-12, name

    vertices = _binary_stl_vertices(
      ASSET_DIRECTORY / "meshes" / visual_mesh.get("filename")
    )
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    assert np.all(center_of_mass >= lower), name
    assert np.all(center_of_mass <= upper), name

    # For X in [a,b] with known mean mu, Var(X) <= (mu-a)(b-mu).
    # Applying that bound to both coordinates perpendicular to each axis
    # gives a hard upper bound on each diagonal moment of inertia.
    maximum_variance = (center_of_mass - lower) * (
      upper - center_of_mass
    )
    maximum_diagonal = mass_kg * np.asarray((
      maximum_variance[1] + maximum_variance[2],
      maximum_variance[0] + maximum_variance[2],
      maximum_variance[0] + maximum_variance[1],
    ))
    assert np.all(np.diag(inertia) <= 1.05 * maximum_diagonal), name


def test_mesh_fallbacks_match_the_conversion_script() -> None:
  links = _links()
  for name in MESH_INERTIA_LINKS:
    link = links[name]
    mass_kg = float(link.find("inertial/mass").get("value"))
    expected = mass_kg * _homogeneous_stl_inertia_per_kg(
      ASSET_DIRECTORY / "meshes" / f"{name}.stl"
    )
    np.testing.assert_allclose(
      _inertia_tensor(link),
      expected,
      rtol=5e-12,
      atol=1e-15,
    )
