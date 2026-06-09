from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from string import Template

import mujoco
import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


EXTERNAL = Path(__file__).resolve().parents[2]
ROBOBASE = EXTERNAL / "robobase"
BIGYM = EXTERNAL / "bigym"
for path in (BIGYM, ROBOBASE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bigym.action_modes import JointPositionActionMode, PelvisDof  # noqa: E402
from bigym.envs.cupboards_with_human_arm import HumanArmDrawerTopOpen  # noqa: E402
from bigym.utils.observation_config import ObservationConfig  # noqa: E402


DEFAULT_OUT_DIR = ROBOBASE / "eval_safety/h1_right_arm_barrier_overlay"

HUMAN_JOINT_NAMES = [
    "cylinder_arm/arm_tx",
    "cylinder_arm/arm_ty",
    "cylinder_arm/arm_shoulder_base",
    "cylinder_arm/arm_shoulder_yaw",
    "cylinder_arm/arm_shoulder_pitch",
    "cylinder_arm/arm_elbow",
]
Q_OUTSIDE = [0.06, -0.10, 0.0, 0.0, -0.35, 1.20]
Q_BLOCK = [0.06, 0.25, 0.0, 1.57, -0.45, 1.00]

# MuJoCo is Z-up. glTF/model-viewer is Y-up. This preserves MuJoCo X and
# displays MuJoCo XY as the viewer's horizontal XZ plane.
MUJOCO_TO_VIEWER = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


HTML_TEMPLATE = Template(
    r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuJoCo Human Arm Viewer</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --panel: rgba(248, 250, 252, 0.94);
      --ink: #182234;
      --muted: #5c6778;
      --line: rgba(30, 42, 60, 0.16);
      --accent: $human_color;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #e9edf2;
      color: var(--ink);
    }
    model-viewer {
      width: 100vw;
      height: 100vh;
      background: radial-gradient(circle at 50% 43%, #ffffff 0%, #edf1f5 57%, #d8e0e9 100%);
    }
    .panel {
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
    }
    .title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    h1 {
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      font-weight: 700;
      letter-spacing: 0;
    }
    .badge {
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .control {
      display: grid;
      grid-template-columns: 112px 1fr;
      align-items: center;
      gap: 10px;
      margin: 10px 0;
      font-size: 13px;
    }
    .control label { color: var(--muted); }
    input[type="color"] {
      width: 100%;
      height: 34px;
      padding: 2px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-top: 14px;
    }
    button {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }
    button.primary {
      grid-column: span 2;
      background: #182234;
      color: white;
    }
    .status {
      min-height: 18px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
  </style>
</head>
<body>
  <model-viewer
    id="viewer"
    src="$glb_data_uri"
    camera-controls
    touch-action="none"
    interaction-prompt="none"
    camera-target="$target"
    camera-orbit="$iso_orbit"
    field-of-view="22deg"
    exposure="0.95"
    shadow-intensity="0.35">
  </model-viewer>

  <section class="panel" aria-label="viewer controls">
    <div class="title">
      <h1>MuJoCo Human Arm</h1>
      <span class="badge">$pose_label pose</span>
    </div>
    <div class="control">
      <label for="armColor">Arm colour</label>
      <input id="armColor" type="color" value="$human_color">
    </div>
    <div class="control">
      <label for="armAlpha">Arm alpha</label>
      <input id="armAlpha" type="range" min="0.05" max="1" step="0.01" value="$human_alpha">
    </div>
    <div class="row">
      <button data-orbit="$top_orbit">Top</button>
      <button data-orbit="$iso_orbit">Iso</button>
      <button data-orbit="$front_orbit">Front</button>
      <button data-orbit="$side_orbit">Side</button>
      <button id="copyCamera">Copy View</button>
      <button id="capture" class="primary">Capture PNG</button>
      <button id="reset">Reset</button>
    </div>
    <div id="status" class="status"></div>
  </section>

  <script>
    const viewer = document.querySelector("#viewer");
    const status = document.querySelector("#status");
    const armColor = document.querySelector("#armColor");
    const armAlpha = document.querySelector("#armAlpha");
    const defaults = {
      target: "$target",
      orbit: "$iso_orbit",
      color: "$human_color",
      alpha: "$human_alpha",
    };

    function hexToFactor(hex, alpha) {
      const value = hex.replace("#", "");
      const r = parseInt(value.slice(0, 2), 16) / 255;
      const g = parseInt(value.slice(2, 4), 16) / 255;
      const b = parseInt(value.slice(4, 6), 16) / 255;
      return [r, g, b, Number(alpha)];
    }

    function applyMaterial() {
      if (!viewer.model) return;
      const alpha = Number(armAlpha.value);
      const factor = hexToFactor(armColor.value, alpha);
      for (const material of viewer.model.materials) {
        material.pbrMetallicRoughness.setBaseColorFactor(factor);
        material.setAlphaMode(alpha < 0.999 ? "BLEND" : "OPAQUE");
        material.setDoubleSided(true);
      }
      document.documentElement.style.setProperty("--accent", armColor.value);
    }

    function fmt(value) {
      return Number(value).toFixed(5);
    }

    viewer.addEventListener("load", () => {
      applyMaterial();
      status.textContent = "Human arm loaded. Drag to orbit, then Capture PNG.";
    });

    armColor.addEventListener("input", applyMaterial);
    armAlpha.addEventListener("input", applyMaterial);

    document.querySelectorAll("[data-orbit]").forEach((button) => {
      button.addEventListener("click", () => {
        viewer.cameraTarget = defaults.target;
        viewer.cameraOrbit = button.dataset.orbit;
      });
    });

    document.querySelector("#reset").addEventListener("click", () => {
      viewer.cameraTarget = defaults.target;
      viewer.cameraOrbit = defaults.orbit;
      armColor.value = defaults.color;
      armAlpha.value = defaults.alpha;
      applyMaterial();
    });

    document.querySelector("#copyCamera").addEventListener("click", async () => {
      const target = viewer.getCameraTarget();
      const orbit = viewer.getCameraOrbit();
      const text =
        'camera-target="' + fmt(target.x) + 'm ' + fmt(target.y) + 'm ' + fmt(target.z) + 'm" ' +
        'camera-orbit="' + fmt(orbit.theta) + 'rad ' + fmt(orbit.phi) + 'rad ' + fmt(orbit.radius) + 'm"';
      try {
        await navigator.clipboard.writeText(text);
        status.textContent = "Camera view copied.";
      } catch (_) {
        status.textContent = text;
      }
    });

    document.querySelector("#capture").addEventListener("click", async () => {
      await viewer.updateComplete;
      const link = document.createElement("a");
      link.download = "mujoco_human_arm_alone.png";
      link.href = viewer.toDataURL("image/png");
      link.click();
      status.textContent = "PNG captured.";
    });
  </script>
</body>
</html>
'''
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a draggable GLB/HTML viewer of the real MuJoCo human arm model only."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", default="mujoco_human_arm_alone_viewer")
    parser.add_argument("--human-pose", choices=("outside", "block"), default="block")
    parser.add_argument("--human-color", default="#d14b3f")
    parser.add_argument("--human-alpha", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def hex_to_rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    value = color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB, got {color!r}")
    rgb = tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (*rgb, float(np.clip(alpha, 0.0, 1.0)))


def rgba_u8(color: str, alpha: float) -> tuple[int, int, int, int]:
    return tuple(int(round(v * 255)) for v in hex_to_rgba(color, alpha))


def make_material(name: str, color: str, alpha: float) -> PBRMaterial:
    return PBRMaterial(
        name=name,
        baseColorFactor=hex_to_rgba(color, alpha),
        metallicFactor=0.0,
        roughnessFactor=0.72,
        alphaMode="BLEND" if alpha < 0.999 else "OPAQUE",
        doubleSided=True,
    )


def make_env():
    return HumanArmDrawerTopOpen(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        observation_config=ObservationConfig(
            cameras=[],
            proprioception=True,
            privileged_information=True,
        ),
        render_mode=None,
        arm_action_mode="scripted",
        control_frequency=50,
        enable_temporary_human_blocker=True,
        trigger_dist=1.20,
        enter_duration=1.2,
        hold_duration=1.2,
        exit_duration=1.2,
        natural_motion_scale=1.0,
        human_joint_names=HUMAN_JOINT_NAMES,
        q_outside=Q_OUTSIDE,
        q_block=Q_BLOCK,
        q_exit=None,
        ee_site_name="right_end_effector",
        handle_site_name="drawer_small_4",
    )


def resolve_mujoco_name(model, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id >= 0:
        return int(obj_id)

    if obj_type == mujoco.mjtObj.mjOBJ_JOINT:
        count = int(model.njnt)
    elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
        count = int(model.ngeom)
    else:
        count = 0

    matches = []
    for i in range(count):
        candidate = mujoco.mj_id2name(model, obj_type, i) or ""
        if candidate == name or candidate.endswith(f"/{name}"):
            matches.append(i)
    if len(matches) == 1:
        return int(matches[0])
    if not matches:
        raise ValueError(f"MuJoCo name not found: {name}")
    raise ValueError(f"MuJoCo name {name!r} is ambiguous: {matches}")


def set_human_pose(model, data, pose: str) -> list[float]:
    q_values = Q_BLOCK if pose == "block" else Q_OUTSIDE
    for joint_name, value in zip(HUMAN_JOINT_NAMES, q_values):
        joint_id = resolve_mujoco_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = float(value)
        data.qvel[qvel_adr] = 0.0
    mujoco.mj_forward(model, data)
    return [float(v) for v in q_values]


def geom_transform(data, geom_id: int) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    return transform


def mesh_from_geom(model, data, geom_id: int):
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    transform = geom_transform(data, geom_id)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(size[0]))
        mesh.apply_transform(transform)
        return mesh
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh = trimesh.creation.capsule(
            radius=float(size[0]),
            height=max(float(size[1]) * 2.0, 1e-6),
            count=[24, 12],
        )
        mesh.apply_transform(transform)
        return mesh
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        mesh = trimesh.creation.cylinder(
            radius=float(size[0]),
            height=max(float(size[1]) * 2.0, 1e-6),
            sections=32,
        )
        mesh.apply_transform(transform)
        return mesh
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return trimesh.creation.box(
            extents=np.maximum(size[:3] * 2.0, 1e-6),
            transform=transform,
        )
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(np.maximum(size[:3], 1e-6))
        mesh.apply_transform(transform)
        return mesh
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        face_adr = int(model.mesh_faceadr[mesh_id])
        face_num = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[vert_adr:vert_adr + vert_num], dtype=np.float64).copy()
        faces = np.asarray(model.mesh_face[face_adr:face_adr + face_num], dtype=np.int64).copy()
        if len(vertices) == 0 or len(faces) == 0:
            return None
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.apply_transform(transform)
        return mesh
    return None


def is_human_arm_geom(name: str) -> bool:
    if not name.startswith("cylinder_arm/"):
        return False
    return name.rsplit("/", 1)[-1] not in {"axis_x", "axis_y", "axis_z"}


def add_geom_to_scene(scene, model, data, geom_id: int, material, face_color):
    mesh = mesh_from_geom(model, data, geom_id)
    if mesh is None:
        return False
    mesh.apply_transform(MUJOCO_TO_VIEWER)
    mesh.visual.face_colors = face_color
    mesh.visual = TextureVisuals(material=material)
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    safe_name = name.replace("/", "_").replace(" ", "_")
    scene.add_geometry(
        mesh,
        geom_name=f"human_{geom_id}_{safe_name}",
        node_name=f"human_{geom_id}_{safe_name}",
    )
    return True


def scene_bounds(scene: trimesh.Scene) -> dict:
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    radius = max(float(np.linalg.norm(extents) * 0.5), 0.20)
    return {
        "min": bounds[0].tolist(),
        "max": bounds[1].tolist(),
        "center": center.tolist(),
        "radius": radius,
    }


def build_scene(args):
    material = make_material("human_material", args.human_color, args.human_alpha)
    face_color = rgba_u8(args.human_color, args.human_alpha)
    env = make_env()
    try:
        env.reset(seed=args.seed)
        model = env.mojo.physics.model.ptr
        data = env.mojo.physics.data.ptr
        pose_qpos = set_human_pose(model, data, args.human_pose)

        scene = trimesh.Scene()
        included_geoms = []
        excluded_axis_geoms = []
        for geom_id in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("cylinder_arm/") and name.rsplit("/", 1)[-1] in {"axis_x", "axis_y", "axis_z"}:
                excluded_axis_geoms.append(name)
                continue
            if not is_human_arm_geom(name):
                continue
            if add_geom_to_scene(scene, model, data, geom_id, material, face_color):
                included_geoms.append(name)

        metadata = {
            "env": "HumanArmDrawerTopOpen",
            "seed": int(args.seed),
            "human_pose": args.human_pose,
            "human_joint_names": HUMAN_JOINT_NAMES,
            "human_pose_qpos": pose_qpos,
            "geom_count": len(included_geoms),
            "included_geoms": included_geoms,
            "excluded_axis_geoms": excluded_axis_geoms,
            "scene_bounds": scene_bounds(scene),
            "axis_transform": "MuJoCo (x,y,z) -> viewer/glTF (x,z,-y)",
            "note": "Only cylinder_arm/* MuJoCo geoms are exported. axis_x/axis_y/axis_z visualization geoms are intentionally excluded.",
        }
        return scene, metadata
    finally:
        env.close()


def write_outputs(scene, metadata, args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = args.out_dir / f"{args.basename}.glb"
    html_path = args.out_dir / f"{args.basename}.html"
    metadata_path = args.out_dir / f"{args.basename}.json"

    glb_path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    center = metadata["scene_bounds"]["center"]
    radius = float(metadata["scene_bounds"]["radius"])
    distance = max(0.55, radius * 2.8)
    target = f"{center[0]:.5f}m {center[1]:.5f}m {center[2]:.5f}m"
    top_orbit = f"0deg 0deg {distance:.5f}m"
    iso_orbit = f"42deg 62deg {distance:.5f}m"
    front_orbit = f"0deg 82deg {distance:.5f}m"
    side_orbit = f"90deg 82deg {distance:.5f}m"
    glb_data_uri = "data:model/gltf-binary;base64," + base64.b64encode(glb_path.read_bytes()).decode("ascii")

    html = HTML_TEMPLATE.substitute(
        glb_data_uri=glb_data_uri,
        target=target,
        top_orbit=top_orbit,
        iso_orbit=iso_orbit,
        front_orbit=front_orbit,
        side_orbit=side_orbit,
        pose_label=args.human_pose,
        human_color=args.human_color,
        human_alpha=f"{args.human_alpha:.2f}",
    )
    html_path.write_text(html)
    return {"glb": glb_path, "html": html_path, "metadata": metadata_path}


def main():
    args = parse_args()
    scene, metadata = build_scene(args)
    outputs = write_outputs(scene, metadata, args)
    print("saved_glb:", outputs["glb"])
    print("saved_html:", outputs["html"])
    print("saved_metadata:", outputs["metadata"])
    print("human_pose:", metadata["human_pose"])
    print("geom_count:", metadata["geom_count"])


if __name__ == "__main__":
    main()
