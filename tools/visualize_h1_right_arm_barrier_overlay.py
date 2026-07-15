from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

REPO = Path(__file__).resolve().parents[3]
EXTERNAL = REPO / "external"
ROBOBASE = EXTERNAL / "robobase"
OSCBF = EXTERNAL / "oscbf"

for path in (ROBOBASE, OSCBF):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from oscbf.core.treemanipulator import TreeManipulator  # noqa: E402
from robobase.safetyfilter.oscbf.oscbf_eehumancapsule_velocity_config import (  # noqa: E402
    OSCBFEEHumanCapsuleVelocityConfig,
)

DEFAULT_BARRIER_URDF = OSCBF / "oscbf/assets/h1/h1.urdf"
DEFAULT_SOURCE_URDF = OSCBF / "oscbf/assets/h1/h1_with_hand.urdf"
DEFAULT_OUT_DIR = ROBOBASE / "eval_safety/h1_right_arm_barrier_overlay"

OSCBF_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

SOURCE_LINKS_WITH_HAND = (
    "torso_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_hand_link",
    "R_hand_base_link",
    "R_thumb_proximal_base",
    "R_thumb_proximal",
    "R_thumb_intermediate",
    "R_thumb_distal",
    "R_index_proximal",
    "R_index_intermediate",
    "R_middle_proximal",
    "R_middle_intermediate",
    "R_ring_proximal",
    "R_ring_intermediate",
    "R_pinky_proximal",
    "R_pinky_intermediate",
)

SOURCE_LINKS_ARM_ONLY = SOURCE_LINKS_WITH_HAND[:5]

# Apply a fixed scene correction so the right-view orientation matches the
# robot-specific reference (yaw, pitch, x-rotation): 0°, 180°, 180°.
# Tuple order is (roll, pitch, yaw) in degrees.
SCENE_CORRECTION_RPY_DEG = (0.0, 180.0, 180.0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a GLB and HTML viewer for the H1 right-arm OSCBF "
            "capsule/sphere overlay."
        )
    )
    parser.add_argument("--barrier-urdf", type=Path, default=DEFAULT_BARRIER_URDF)
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_SOURCE_URDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--q-urdf-arm",
        type=float,
        nargs=8,
        metavar=("LSP", "LSR", "LSY", "LE", "RSP", "RSR", "RSY", "RE"),
        default=[0.0] * 8,
        help=(
            "URDF arm joint values in OSCBF order. Defaults to the neutral "
            "fixed-torso surrogate pose."
        ),
    )
    parser.add_argument(
        "--right-hand-angle",
        type=float,
        default=0.0,
        help="Only affects optional hand source meshes, not OSCBF barrier geometry.",
    )
    parser.add_argument("--arm-only", action="store_true", help="Skip hand meshes.")
    parser.add_argument("--mesh-format", choices=("stl", "dae"), default="stl")
    parser.add_argument("--source-color", default="#b9c0c9")
    parser.add_argument("--source-alpha", type=float, default=1.0)
    parser.add_argument("--overlay-color", default="#ff8a00")
    parser.add_argument("--overlay-alpha", type=float, default=0.42)
    parser.add_argument("--capsule-sections", type=int, default=32)
    parser.add_argument("--sphere-subdivisions", type=int, default=4)
    parser.add_argument(
        "--hide-axis",
        action="store_true",
        help="Hide the XYZ axis gizmo (shown by default).",
    )
    return parser.parse_args()

def hex_to_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    color = hex_color.strip().lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {hex_color!r}")
    rgb = tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (*rgb, float(np.clip(alpha, 0.0, 1.0)))

def rgba_u8(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    rgba = hex_to_rgba(hex_color, alpha)
    return tuple(int(round(v * 255)) for v in rgba)

def make_material(
    name: str,
    color: str,
    alpha: float,
    roughness: float = 0.72,
    metallic: float = 0.0,
) -> PBRMaterial:
    return PBRMaterial(
        name=name,
        baseColorFactor=hex_to_rgba(color, alpha),
        metallicFactor=metallic,
        roughnessFactor=roughness,
        alphaMode="BLEND" if alpha < 0.999 else "OPAQUE",
        doubleSided=True,
    )

def rpy_to_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx

def xyz_rpy_to_matrix(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = np.asarray(tuple(xyz), dtype=np.float64)
    return transform

def axis_angle_to_matrix(axis: Iterable[float], angle: float) -> np.ndarray:
    axis_arr = np.asarray(tuple(axis), dtype=np.float64)
    norm = float(np.linalg.norm(axis_arr))
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)

    x, y, z = axis_arr / norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    rot = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    return transform

def prismatic_to_matrix(axis: Iterable[float], value: float) -> np.ndarray:
    axis_arr = np.asarray(tuple(axis), dtype=np.float64)
    norm = float(np.linalg.norm(axis_arr))
    transform = np.eye(4, dtype=np.float64)
    if norm >= 1e-12:
        transform[:3, 3] = axis_arr / norm * float(value)
    return transform

def parse_vector(raw: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if raw is None:
        return default
    return tuple(float(part) for part in raw.split())

def parse_urdf(urdf_path: Path) -> tuple[list[str], list[dict], dict[str, dict]]:
    root = ET.parse(urdf_path).getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    link_visuals: dict[str, dict] = {}

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        visual = link.find("visual")
        if visual is None:
            continue

        mesh_node = visual.find("./geometry/mesh")
        if mesh_node is None:
            continue

        origin = visual.find("origin")
        xyz = parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        scale = parse_vector(mesh_node.attrib.get("scale"), (1.0, 1.0, 1.0))
        link_visuals[link_name] = {
            "filename": mesh_node.attrib["filename"],
            "origin": xyz_rpy_to_matrix(xyz, rpy),
            "scale": scale,
        }

    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue

        xyz = parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent.attrib["link"],
                "child": child.attrib["link"],
                "origin": xyz_rpy_to_matrix(xyz, rpy),
                "axis": parse_vector(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
            }
        )

    return links, joints, link_visuals

def apply_transform_to_point(point: Iterable[float], transform: np.ndarray) -> np.ndarray:
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(tuple(point), dtype=np.float64)
    return (transform @ point_h)[:3]


def resolve_mesh_path(urdf_path: Path, filename: str, mesh_format: str) -> Path:
    if filename.startswith("package://h1_description/"):
        relative = filename.removeprefix("package://h1_description/")
        path = urdf_path.parent / "h1_description" / relative
    else:
        path = (urdf_path.parent / filename).resolve()

    if mesh_format == "stl":
        stl_path = path.with_suffix(".STL")
        if stl_path.is_file():
            return stl_path
        stl_path = path.with_suffix(".stl")
        if stl_path.is_file():
            return stl_path

    return path

def forward_kinematics(
    links: list[str],
    joints: list[dict],
    q_by_joint: dict[str, float],
) -> dict[str, np.ndarray]:
    child_links = {joint["child"] for joint in joints}
    root_links = [link for link in links if link not in child_links]
    if not root_links:
        raise ValueError("Could not find a URDF root link.")

    children: dict[str, list[dict]] = {}
    for joint in joints:
        children.setdefault(joint["parent"], []).append(joint)

    transforms: dict[str, np.ndarray] = {}

    def walk(link_name: str, world_from_link: np.ndarray) -> None:
        transforms[link_name] = world_from_link
        for joint in children.get(link_name, []):
            joint_type = joint["type"]
            value = float(q_by_joint.get(joint["name"], 0.0))
            if joint_type in ("revolute", "continuous"):
                motion = axis_angle_to_matrix(joint["axis"], value)
            elif joint_type == "prismatic":
                motion = prismatic_to_matrix(joint["axis"], value)
            else:
                motion = np.eye(4, dtype=np.float64)
            walk(joint["child"], world_from_link @ joint["origin"] @ motion)

    for root_link in root_links:
        walk(root_link, np.eye(4, dtype=np.float64))

    return transforms

def load_visual_mesh(
    urdf_path: Path,
    visual: dict,
    link_transform: np.ndarray,
    mesh_format: str,
    face_color: tuple[int, int, int, int],
    material: PBRMaterial,
) -> trimesh.Trimesh:
    mesh_path = resolve_mesh_path(urdf_path, visual["filename"], mesh_format)
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a mesh from {mesh_path}, got {type(mesh).__name__}")

    mesh = mesh.copy()
    mesh.apply_scale(visual["scale"])
    mesh.apply_transform(link_transform @ visual["origin"])
    mesh.visual.face_colors = face_color
    mesh.visual = TextureVisuals(material=material)
    return mesh

def make_capsule_mesh(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    sections: int,
    material: PBRMaterial,
    face_color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-8:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        mesh.apply_translation(start)
    else:
        transform = trimesh.geometry.align_vectors([0, 0, 1], axis / length)
        transform[:3, 3] = 0.5 * (start + end)
        mesh = trimesh.creation.capsule(
            height=length,
            radius=radius,
            count=[max(12, sections), max(6, sections // 2)],
            transform=transform,
        )
    mesh.visual.face_colors = face_color
    mesh.visual = TextureVisuals(material=material)
    return mesh

def make_sphere_mesh(
    center: np.ndarray,
    radius: float,
    subdivisions: int,
    material: PBRMaterial,
    face_color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(np.asarray(center, dtype=np.float64))
    mesh.visual.face_colors = face_color
    mesh.visual = TextureVisuals(material=material)
    return mesh

def build_barrier_geometry(
    barrier_urdf: Path,
    q_urdf_arm: np.ndarray,
) -> tuple[list[dict], dict]:
    robot = TreeManipulator.from_urdf(
        urdf_filename=str(barrier_urdf),
        ee_joint_idx=0,
        controlled_joint_indices=None,
    )
    controlled_indices = tuple(robot.joint_index(name) for name in OSCBF_ARM_JOINT_NAMES)
    robot.set_controlled_joints(controlled_indices)
    robot.ee_joint_idx = robot.joint_index("right_elbow_joint")

    q = np.zeros(robot.num_joints, dtype=np.float32)
    q[np.asarray(controlled_indices, dtype=np.int64)] = q_urdf_arm.astype(np.float32)

    config = OSCBFEEHumanCapsuleVelocityConfig(
        robot=robot,
        capsule_a_init=np.zeros((2, 3), dtype=np.float32),
        capsule_b_init=np.zeros((2, 3), dtype=np.float32),
        capsule_radii_init=np.ones((2,), dtype=np.float32) * 0.1,
    )

    robot_a, robot_b, robot_radii = config._right_arm_capsules(jnp.asarray(q, dtype=jnp.float32))
    sphere_center, sphere_radius = config._right_gripper_sphere(jnp.asarray(q, dtype=jnp.float32))

    capsules = []
    for name, a, b, radius in zip(
        ("right_shoulder_to_elbow_capsule", "right_elbow_to_ee_capsule"),
        np.asarray(robot_a, dtype=np.float32),
        np.asarray(robot_b, dtype=np.float32),
        np.asarray(robot_radii, dtype=np.float32),
    ):
        capsules.append(
            {
                "name": name,
                "a": a.tolist(),
                "b": b.tolist(),
                "radius": float(radius),
            }
        )

    sphere = {
        "name": "right_gripper_sphere",
        "center": np.asarray(sphere_center, dtype=np.float32).tolist(),
        "radius": float(np.asarray(sphere_radius, dtype=np.float32)),
        "offset_in_right_elbow_frame": np.asarray(
            config.right_gripper_sphere_offset,
            dtype=np.float32,
        ).tolist(),
    }

    constants = {
        "right_arm_contact_margin": float(config.right_arm_contact_margin),
        "right_arm_capsule_radii": np.asarray(config.right_arm_capsule_radii, dtype=np.float32).tolist(),
        "right_gripper_sphere_radius": float(np.asarray(config.right_gripper_sphere_radius, dtype=np.float32)),
    }
    return capsules, {"sphere": sphere, "constants": constants}

def scene_bounds(scene: trimesh.Scene) -> dict:
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    radius = max(float(np.linalg.norm(extents) * 0.5), 0.25)
    return {
        "min": bounds[0].tolist(),
        "max": bounds[1].tolist(),
        "center": center.tolist(),
        "radius": radius,
    }

def build_scene(args: argparse.Namespace) -> tuple[trimesh.Scene, dict]:
    source_urdf = args.source_urdf.resolve()
    barrier_urdf = args.barrier_urdf.resolve()
    q_urdf_arm = np.asarray(args.q_urdf_arm, dtype=np.float32)

    source_material = make_material(
        "h1_source_material",
        args.source_color,
        args.source_alpha,
    )
    overlay_material = make_material(
        "barrier_overlay_material",
        args.overlay_color,
        args.overlay_alpha,
        roughness=0.55,
    )
    source_color = rgba_u8(args.source_color, args.source_alpha)
    overlay_color = rgba_u8(args.overlay_color, args.overlay_alpha)

    # Axis gizmo materials (drawn as simple capsules, not affected by alpha controls).
    axis_colors = {
        "x": rgba_u8("#ef4444", 1.0),
        "y": rgba_u8("#22c55e", 1.0),
        "z": rgba_u8("#3b82f6", 1.0),
    }
    axis_hex_colors = {
        "x": "#ef4444",
        "y": "#22c55e",
        "z": "#3b82f6",
    }
    axis_materials = {
        "x": make_material("axis_x_material", "#ef4444", 1.0, roughness=0.2),
        "y": make_material("axis_y_material", "#22c55e", 1.0, roughness=0.2),
        "z": make_material("axis_z_material", "#3b82f6", 1.0, roughness=0.2),
    }
    axis_length = 0.35
    axis_radius = 0.004
    axis_label_offset = 0.42

    links, joints, visuals = parse_urdf(source_urdf)
    q_by_joint = {name: value for name, value in zip(OSCBF_ARM_JOINT_NAMES, q_urdf_arm)}
    q_by_joint["right_hand_joint"] = float(args.right_hand_angle)
    source_fk = forward_kinematics(links, joints, q_by_joint)

    source_links = SOURCE_LINKS_ARM_ONLY if args.arm_only else SOURCE_LINKS_WITH_HAND
    scene_transform = xyz_rpy_to_matrix(
        (0.0, 0.0, 0.0),
        tuple(math.radians(v) for v in SCENE_CORRECTION_RPY_DEG),
    )
    scene = trimesh.Scene()
    loaded_links = []
    skipped_links = []

    for link_name in source_links:
        if link_name not in visuals or link_name not in source_fk:
            skipped_links.append(link_name)
            continue

        mesh = load_visual_mesh(
            source_urdf,
            visuals[link_name],
            source_fk[link_name],
            args.mesh_format,
            source_color,
            source_material,
        )
        scene.add_geometry(mesh, geom_name=f"h1_{link_name}", node_name=f"h1_{link_name}")
        loaded_links.append(link_name)

    capsules, barrier_payload = build_barrier_geometry(barrier_urdf, q_urdf_arm)
    for capsule in capsules:
        mesh = make_capsule_mesh(
            np.asarray(capsule["a"], dtype=np.float64),
            np.asarray(capsule["b"], dtype=np.float64),
            float(capsule["radius"]),
            args.capsule_sections,
            overlay_material,
            overlay_color,
        )
        scene.add_geometry(mesh, geom_name=capsule["name"], node_name=capsule["name"])

    sphere = barrier_payload["sphere"]
    sphere_mesh = make_sphere_mesh(
        np.asarray(sphere["center"], dtype=np.float64),
        float(sphere["radius"]),
        args.sphere_subdivisions,
        overlay_material,
        overlay_color,
    )
    scene.add_geometry(sphere_mesh, geom_name=sphere["name"], node_name=sphere["name"])

    if not args.hide_axis:
        axis_origin = np.zeros(3, dtype=np.float64)
        for name, axis_end, axis_mat, axis_face_color in (
            ("world_x_axis", np.array([axis_length, 0.0, 0.0]), axis_materials["x"], axis_colors["x"]),
            ("world_y_axis", np.array([0.0, axis_length, 0.0]), axis_materials["y"], axis_colors["y"]),
            ("world_z_axis", np.array([0.0, 0.0, axis_length]), axis_materials["z"], axis_colors["z"]),
        ):
            scene.add_geometry(
                make_capsule_mesh(
                    axis_origin,
                    axis_end,
                    axis_radius,
                    max(12, args.capsule_sections),
                    axis_mat,
                    axis_face_color,
                ),
                geom_name=name,
                node_name=name,
            )

    scene.apply_transform(scene_transform)
    for capsule in capsules:
        capsule["a"] = apply_transform_to_point(capsule["a"], scene_transform).tolist()
        capsule["b"] = apply_transform_to_point(capsule["b"], scene_transform).tolist()

    sphere["center"] = apply_transform_to_point(sphere["center"], scene_transform).tolist()

    axis_labels = []
    if not args.hide_axis:
        axis_labels = [
            {"name": "X", "axis": "x", "position": apply_transform_to_point(
                np.array([axis_label_offset, 0.0, 0.0], dtype=np.float64), scene_transform
            ).tolist(), "color": axis_hex_colors["x"]},
            {"name": "Y", "axis": "y", "position": apply_transform_to_point(
                np.array([0.0, axis_label_offset, 0.0], dtype=np.float64), scene_transform
            ).tolist(), "color": axis_hex_colors["y"]},
            {"name": "Z", "axis": "z", "position": apply_transform_to_point(
                np.array([0.0, 0.0, axis_label_offset], dtype=np.float64), scene_transform
            ).tolist(), "color": axis_hex_colors["z"]},
        ]

    bounds = scene_bounds(scene)
    metadata = {
        "source_urdf": str(source_urdf),
        "barrier_urdf": str(barrier_urdf),
        "mesh_format": args.mesh_format,
        "q_urdf_arm_order": OSCBF_ARM_JOINT_NAMES,
        "q_urdf_arm": q_urdf_arm.tolist(),
        "right_hand_angle": float(args.right_hand_angle),
        "source_links_loaded": loaded_links,
        "source_links_skipped": skipped_links,
        "barrier_capsules": capsules,
        "barrier_sphere": sphere,
        "barrier_constants": barrier_payload["constants"],
        "axis_labels": axis_labels,
        "scene_bounds": bounds,
        "note": (
            "The barrier overlay is computed from "
            "OSCBFEEHumanCapsuleVelocityConfig._right_arm_capsules and "
            "_right_gripper_sphere. Source meshes are visual context only."
        ),
    }
    return scene, metadata

def html_template(glb_data_uri: str, metadata: dict, glb_filename: str) -> str:
    center = metadata["scene_bounds"]["center"]
    radius = float(metadata["scene_bounds"]["radius"])
    target = f"{center[0]:.5f}m {center[1]:.5f}m {center[2]:.5f}m"
    distance = max(0.8, radius * 2.4)
    # Default camera is the right (+y) view.
    orbit_default = f"0deg 0deg {distance:.5f}m"
    # Axis-aligned presets: right(+y), front(-x), top(-z) view directions.
    # Additional user-requested adjustments:
    #  - front: +90° clockwise around world Y.
    #  - top: +90° clockwise around world Z (applied as azimuth offset in this viewer API).
    orbit_front = f"90deg 90deg {distance:.5f}m"
    orbit_top = f"180deg 90deg {distance:.5f}m"
    axis_hotspot_markup = []
    for idx, item in enumerate(metadata.get("axis_labels", [])):
        px, py, pz = item.get("position", (0.0, 0.0, 0.0))
        axis = item.get("axis", "")
        label = item.get("name", "")
        axis_class = f"axis-hotspot axis-{axis}" if axis else "axis-hotspot"
        axis_hotspot_markup.append(
            f"    <button class=\"{axis_class}\" slot=\"hotspot-{idx}\""
            f" data-position=\"{px:.5f}m {py:.5f}m {pz:.5f}m\""
            f" data-normal=\"0 1 0\">{label}</button>"
        )
    axis_hotspots = "\n".join(axis_hotspot_markup)

    metadata_json = json.dumps(metadata, indent=2)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>H1 Right Arm Barrier Overlay</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --panel: rgba(248, 250, 252, 0.92);
      --ink: #172033;
      --muted: #5d687a;
      --line: rgba(28, 39, 57, 0.16);
      --accent: #ff8a00;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #e8edf3;
      color: var(--ink);
      overflow: hidden;
    }}
    model-viewer {{
      width: 100vw;
      height: 100vh;
      background: radial-gradient(circle at 50% 40%, #ffffff 0%, #eef2f6 52%, #dbe3ec 100%);
    }}
    .panel {{
      position: fixed;
      top: 16px;
      right: 16px;
      width: min(360px, calc(100vw - 32px));
      max-height: calc(100vh - 32px);
      overflow: auto;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 18px 40px rgba(31, 42, 57, 0.18);
      backdrop-filter: blur(14px);
    }}
    .title {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    h1 {{
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .badge {{
      flex: 0 0 auto;
      font-size: 12px;
      color: var(--muted);
    }}
    .control {{
      display: grid;
      grid-template-columns: 128px 1fr;
      align-items: center;
      gap: 10px;
      margin: 10px 0;
      font-size: 13px;
    }}
    .control label {{
      color: var(--muted);
    }}
    .axis-hotspot {{
      --min-hotspot-opacity: 0;
      --max-hotspot-opacity: 1;
      border: 1px solid rgba(0, 0, 0, 0.2);
      border-radius: 999px;
      width: 18px;
      height: 18px;
      padding: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: #ffffff;
      font-size: 11px;
      font-weight: 700;
      line-height: 1;
      text-shadow: 0 0 2px rgba(0, 0, 0, 0.45);
      pointer-events: none;
    }}
    .axis-hotspot.axis-x {{
      background: #ef4444;
    }}
    .axis-hotspot.axis-y {{
      background: #22c55e;
    }}
    .axis-hotspot.axis-z {{
      background: #3b82f6;
    }}
    .axis-legend {{
      margin: 10px 0;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.8);
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    .axis-title {{
      font-size: 12px;
      font-weight: 600;
      color: var(--ink);
      margin-bottom: 2px;
    }}
    .axis-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .axis-dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(0, 0, 0, 0.12);
      display: inline-block;
    }}
    .axis-x {{
      background: #ef4444;
    }}
    .axis-y {{
      background: #22c55e;
    }}
    .axis-z {{
      background: #3b82f6;
    }}
    input[type="color"] {{
      width: 100%;
      height: 36px;
      padding: 2px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    input[type="text"], input[type="number"] {{
      width: 100%;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      padding: 7px 8px;
      font: inherit;
      font-size: 12px;
    }}
    .row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-top: 14px;
    }}
    button {{
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }}
    button.primary {{
      grid-column: span 2;
      background: #172033;
      color: white;
    }}
    pre {{
      display: none;
    }}
    .status {{
      min-height: 18px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <model-viewer
    id="viewer"
    src="{glb_data_uri}"
    camera-controls
    touch-action="pan-y"
    interaction-prompt="none"
    camera-target="{target}"
    camera-orbit="{orbit_default}"
    field-of-view="24deg"
    exposure="0.95"
    shadow-intensity="0.32">
{axis_hotspots}
  </model-viewer>

  <section class="panel" aria-label="viewer controls">
    <div class="title">
      <h1>H1 Right Arm Barrier</h1>
      <span class="badge">{glb_filename}</span>
    </div>
    <div class="control">
      <label for="overlayColor">Overlay colour</label>
      <input id="overlayColor" type="color" value="#ff8a00">
    </div>
    <div class="control">
      <label for="overlayAlpha">Overlay alpha</label>
      <input id="overlayAlpha" type="range" min="0" max="1" step="0.01" value="0.42">
    </div>
    <div class="control">
      <label for="sourceAlpha">H1 arm alpha</label>
      <input id="sourceAlpha" type="range" min="0.05" max="1" step="0.01" value="1">
    </div>
    <div class="control">
      <label for="hideAxis">Hide axis</label>
      <input id="hideAxis" type="checkbox">
    </div>
    <div id="axisLegend" class="axis-legend" aria-label="Axis legend">
      <div class="axis-title">Axis</div>
      <div class="axis-item"><span class="axis-dot axis-x"></span> X</div>
      <div class="axis-item"><span class="axis-dot axis-y"></span> Y</div>
      <div class="axis-item"><span class="axis-dot axis-z"></span> Z</div>
    </div>

    <div class="row">
      <button id="capture" class="primary">Capture PNG</button>
      <button id="reset">Reset</button>
    </div>
    <div id="status" class="status"></div>
  </section>

  <pre id="metadata">{metadata_json}</pre>

  <script>
    const viewer = document.querySelector("#viewer");
    const overlayColor = document.querySelector("#overlayColor");
    const overlayAlpha = document.querySelector("#overlayAlpha");
    const sourceAlpha = document.querySelector("#sourceAlpha");
    const status = document.querySelector("#status");
    const hideAxis = document.querySelector("#hideAxis");
    const target = "{target}";
    const defaultOrbit = "{orbit_default}";

    function toRadians(deg) {{
      return Number(deg) * Math.PI / 180;
    }}

    function toDegrees(rad) {{
      return Number(rad) * 180 / Math.PI;
    }}

    function clamp(v, min, max) {{
      return Math.max(min, Math.min(max, v));
    }}

    function parseOrbitField(value) {{
      if (typeof value === "number") {{
        if (Math.abs(value) <= 2 * Math.PI + 1e-6) {{
          return toDegrees(value);
        }}
        return value;
      }}

      if (typeof value !== "string") {{
        return NaN;
      }}

      const num = Number.parseFloat(value);
      if (!Number.isFinite(num)) {{
        return NaN;
      }}

      const isRad = /rad/i.test(value);
      if (isRad || Math.abs(num) <= 2 * Math.PI + 1e-6) {{
        return toDegrees(num);
      }}
      return num;
    }}

    function parseDistanceField(value) {{
      if (typeof value === "number") {{
        return value;
      }}

      if (value && typeof value === "object") {{
        const num = Number.parseFloat(String(value.value ?? value["value"] ?? value.toString()));
        return Number.isFinite(num) ? num : NaN;
      }}

      if (typeof value !== "string") {{
        return NaN;
      }}

      const num = Number.parseFloat(value);
      return Number.isFinite(num) ? num : NaN;
    }}

    function parseVector3(value) {{
      if (!value) {{
        return null;
      }}

      if (typeof value === "object") {{
        const x = Number.parseFloat(String(value.x ?? value[0] ?? value.toString()));
        const y = Number.parseFloat(String(value.y ?? value[1] ?? value.toString()));
        const z = Number.parseFloat(String(value.z ?? value[2] ?? value.toString()));
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {{
          return [x, y, z];
        }}
      }}

      if (typeof value === "string") {{
        const nums = value
          .trim()
          .replace(/m$/i, "")
          .split(/\s+/)
          .map((item) => Number.parseFloat(item.replace(/m$/i, "")));
        if (nums.length === 3 && nums.every(Number.isFinite)) {{
          return nums;
        }}
      }}

      return null;
    }}

    function formatVector3(v) {{
      return `${{v[0].toFixed(5)}}m ${{v[1].toFixed(5)}}m ${{v[2].toFixed(5)}}m`;
    }}

    const AXIS_MATERIAL_FACTORS = {{
      axis_x_material: [239 / 255, 68 / 255, 68 / 255, 1],
      axis_y_material: [34 / 255, 197 / 255, 94 / 255, 1],
      axis_z_material: [59 / 255, 130 / 255, 246 / 255, 1],
    }};

    function hexToFactor(hex, alpha) {{
      const value = hex.replace("#", "");
      const r = parseInt(value.slice(0, 2), 16) / 255;
      const g = parseInt(value.slice(2, 4), 16) / 255;
      const b = parseInt(value.slice(4, 6), 16) / 255;
      return [r, g, b, Number(alpha)];
    }}

    function setMaterial(material, factor, alpha) {{
      material.pbrMetallicRoughness.setBaseColorFactor(factor);
      material.setAlphaMode(alpha < 0.999 ? "BLEND" : "OPAQUE");
      material.setDoubleSided(true);
    }}

    function setAxisVisibility() {{
      const visible = !hideAxis.checked;
      document.querySelectorAll(".axis-hotspot").forEach((el) => {{
        el.style.display = visible ? "inline-flex" : "none";
      }});
      const axisLegend = document.querySelector("#axisLegend");
      if (axisLegend) {{
        axisLegend.style.display = visible ? "grid" : "none";
      }}

      if (!viewer.model) {{
        return;
      }}
      for (const material of viewer.model.materials) {{
        const name = material.name || "";
        if (!name.startsWith("axis_")) {{
          continue;
        }}
        const base = AXIS_MATERIAL_FACTORS[name];
        if (!base) {{
          continue;
        }}
        if (visible) {{
          setMaterial(material, base, 1);
        }} else {{
          setMaterial(material, [base[0], base[1], base[2], 0], 0);
        }}
      }}
    }}

    function applyMaterials() {{
      if (!viewer.model) return;
      const overlayFactor = hexToFactor(overlayColor.value, overlayAlpha.value);
      const sourceFactor = hexToFactor("#b9c0c9", sourceAlpha.value);
      for (const material of viewer.model.materials) {{
        const name = material.name || "";
        if (name.startsWith("barrier_")) {{
          setMaterial(material, overlayFactor, Number(overlayAlpha.value));
        }} else if (name.startsWith("h1_")) {{
          setMaterial(material, sourceFactor, Number(sourceAlpha.value));
        }}
      }}
      document.documentElement.style.setProperty("--accent", overlayColor.value);
      setAxisVisibility();
    }}

    viewer.addEventListener("load", () => {{
      applyMaterials();
      status.textContent = "Drag to orbit. Use Capture PNG when the angle is set.";
    }});

    [overlayColor, overlayAlpha, sourceAlpha].forEach((input) => {{
      input.addEventListener("input", applyMaterials);
    }});
    hideAxis.addEventListener("change", () => {{
      setAxisVisibility();
      applyMaterials();
    }});

    document.querySelectorAll("[data-orbit]").forEach((button) => {{
      button.addEventListener("click", () => {{
        viewer.cameraTarget = target;
        viewer.cameraOrbit = button.dataset.orbit;
      }});
    }});

    document.querySelector("#reset").addEventListener("click", () => {{
      viewer.cameraTarget = target;
      viewer.cameraOrbit = defaultOrbit;
      overlayColor.value = "#ff8a00";
      overlayAlpha.value = "0.42";
      sourceAlpha.value = "1";
      hideAxis.checked = false;
      applyMaterials();
      setAxisVisibility();
    }});

    document.querySelector("#capture").addEventListener("click", async () => {{
      await viewer.updateComplete;
      const link = document.createElement("a");
      link.download = "h1_right_arm_barrier_overlay.png";
      link.href = viewer.toDataURL("image/png");
      link.click();
      status.textContent = "PNG captured.";
    }});
  </script>
</body>
</html>
"""

def write_outputs(scene: trimesh.Scene, metadata: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "h1_right_arm_barrier_overlay.glb"
    html_path = out_dir / "h1_right_arm_barrier_overlay_viewer.html"
    metadata_path = out_dir / "h1_right_arm_barrier_overlay_metadata.json"

    glb_path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    glb_data_uri = "data:model/gltf-binary;base64," + base64.b64encode(glb_path.read_bytes()).decode("ascii")
    html_path.write_text(html_template(glb_data_uri, metadata, glb_path.name))
    return {"glb": glb_path, "html": html_path, "metadata": metadata_path}

def main() -> None:
    args = parse_args()
    scene, metadata = build_scene(args)
    outputs = write_outputs(scene, metadata, args.out_dir.resolve())
    print("saved_glb:", outputs["glb"])
    print("saved_html:", outputs["html"])
    print("saved_metadata:", outputs["metadata"])
    print("barrier_capsules:", len(metadata["barrier_capsules"]))
    print("barrier_sphere_radius:", metadata["barrier_sphere"]["radius"])

if __name__ == "__main__":
    main()
