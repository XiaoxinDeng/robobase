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
from demonstrations.demo import Demo  # noqa: E402


DEFAULT_OUT_DIR = ROBOBASE / "eval_safety/h1_right_arm_barrier_overlay"
DEFAULT_DEMO_MANIFEST = Path(
    "~/.bigym/demonstrations/0.9.0/DrawerTopOpen/"
    "JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/"
    "lightweight/manifest.json"
).expanduser()

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

# MuJoCo is Z-up. glTF/model-viewer is Y-up. This maps MuJoCo XY ground
# into the glTF XZ ground plane, so model-viewer's top camera is a real
# MuJoCo top-down view.
MUJOCO_TO_VIEWER = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


HTML_TEMPLATE = Template(r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuJoCo H1 Human Arm Top-Down Viewer</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --panel: rgba(248, 250, 252, 0.94);
      --ink: #172033;
      --muted: #5d687a;
      --line: rgba(28, 39, 57, 0.16);
      --accent: #1769c2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #e8edf3;
      color: var(--ink);
    }
    model-viewer {
      width: 100vw;
      height: 100vh;
      background: radial-gradient(circle at 50% 42%, #ffffff 0%, #eef2f6 54%, #dbe3ec 100%);
    }
    .panel {
      position: fixed;
      top: 16px;
      right: 16px;
      width: min(390px, calc(100vw - 32px));
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
    }
    .control {
      display: grid;
      grid-template-columns: 132px 1fr;
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
      background: #172033;
      color: white;
    }
    .status {
      min-height: 18px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    pre { display: none; }
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
    camera-orbit="$top_orbit"
    field-of-view="18deg"
    exposure="0.9"
    shadow-intensity="0.35">
  </model-viewer>

  <section class="panel" aria-label="viewer controls">
    <div class="title">
      <h1>MuJoCo Top-Down Scene</h1>
      <span class="badge">$glb_filename</span>
    </div>
    <div class="control">
      <label for="robotColor">Robot colour</label>
      <input id="robotColor" type="color" value="$robot_color">
    </div>
    <div class="control">
      <label for="robotAlpha">Robot alpha</label>
      <input id="robotAlpha" type="range" min="0.05" max="1" step="0.01" value="$robot_alpha">
    </div>
    <div class="control">
      <label for="humanColor">Human arm colour</label>
      <input id="humanColor" type="color" value="$human_color">
    </div>
    <div class="control">
      <label for="humanAlpha">Human arm alpha</label>
      <input id="humanAlpha" type="range" min="0.05" max="1" step="0.01" value="$human_alpha">
    </div>
    <div class="control">
      <label for="humanVisible">Show human arm</label>
      <input id="humanVisible" type="checkbox" checked>
    </div>
    <div class="control">
      <label for="trajectoryVisible">Show arm trajectory</label>
      <input id="trajectoryVisible" type="checkbox" checked>
    </div>
    <div class="control">
      <label for="eeTrajectoryVisible">Show EE trajectory</label>
      <input id="eeTrajectoryVisible" type="checkbox" checked>
    </div>
    <div class="control">
      <label for="eeTrajectoryColor">EE trajectory colour</label>
      <input id="eeTrajectoryColor" type="color" value="$ee_trajectory_color">
    </div>
    <div class="control">
      <label for="eeTrajectoryAlpha">EE trajectory alpha</label>
      <input id="eeTrajectoryAlpha" type="range" min="0.05" max="1" step="0.01" value="$ee_trajectory_alpha">
    </div>
    <div class="control">
      <label for="cabinetAlpha">Drawer/cabinet alpha</label>
      <input id="cabinetAlpha" type="range" min="0.05" max="1" step="0.01" value="$cabinet_alpha">
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

  <pre id="metadata">$metadata_json</pre>

  <script>
    const viewer = document.querySelector("#viewer");
    const status = document.querySelector("#status");
    const controls = {
      robotColor: document.querySelector("#robotColor"),
      robotAlpha: document.querySelector("#robotAlpha"),
      humanColor: document.querySelector("#humanColor"),
      humanAlpha: document.querySelector("#humanAlpha"),
      humanVisible: document.querySelector("#humanVisible"),
      trajectoryVisible: document.querySelector("#trajectoryVisible"),
      eeTrajectoryVisible: document.querySelector("#eeTrajectoryVisible"),
      eeTrajectoryColor: document.querySelector("#eeTrajectoryColor"),
      eeTrajectoryAlpha: document.querySelector("#eeTrajectoryAlpha"),
      cabinetAlpha: document.querySelector("#cabinetAlpha"),
    };
    const defaults = {
      target: "$target",
      orbit: "$top_orbit",
      robotColor: "$robot_color",
      robotAlpha: "$robot_alpha",
      humanColor: "$human_color",
      humanAlpha: "$human_alpha",
      humanVisible: true,
      trajectoryVisible: true,
      eeTrajectoryVisible: true,
      eeTrajectoryColor: "$ee_trajectory_color",
      eeTrajectoryAlpha: "$ee_trajectory_alpha",
      cabinetAlpha: "$cabinet_alpha",
    };

    function hexToFactor(hex, alpha) {
      const value = hex.replace("#", "");
      const r = parseInt(value.slice(0, 2), 16) / 255;
      const g = parseInt(value.slice(2, 4), 16) / 255;
      const b = parseInt(value.slice(4, 6), 16) / 255;
      return [r, g, b, Number(alpha)];
    }

    function setMaterial(material, factor, alpha) {
      material.pbrMetallicRoughness.setBaseColorFactor(factor);
      material.setAlphaMode(alpha < 0.999 ? "BLEND" : "OPAQUE");
      material.setDoubleSided(true);
    }

    function applyMaterials() {
      if (!viewer.model) return;
      const robotFactor = hexToFactor(controls.robotColor.value, controls.robotAlpha.value);
      const humanAlpha = controls.humanVisible.checked ? Number(controls.humanAlpha.value) : 0;
      const trajectoryAlpha = controls.trajectoryVisible.checked ? 0.9 : 0;
      const eeTrajectoryAlpha = controls.eeTrajectoryVisible.checked ? Number(controls.eeTrajectoryAlpha.value) : 0;
      const humanFactor = hexToFactor(controls.humanColor.value, humanAlpha);
      const trajectoryFactor = hexToFactor("#dd7a22", trajectoryAlpha);
      const eeTrajectoryFactor = hexToFactor(controls.eeTrajectoryColor.value, eeTrajectoryAlpha);
      const cabinetFactor = hexToFactor("#b6aa98", controls.cabinetAlpha.value);
      for (const material of viewer.model.materials) {
        const name = material.name || "";
        if (name.startsWith("robot_")) {
          setMaterial(material, robotFactor, Number(controls.robotAlpha.value));
        } else if (name.startsWith("human_")) {
          setMaterial(material, humanFactor, humanAlpha);
        } else if (name.startsWith("ee_trajectory_")) {
          setMaterial(material, eeTrajectoryFactor, eeTrajectoryAlpha);
        } else if (name.startsWith("trajectory_")) {
          setMaterial(material, trajectoryFactor, trajectoryAlpha);
        } else if (name.startsWith("cabinet_")) {
          setMaterial(material, cabinetFactor, Number(controls.cabinetAlpha.value));
        }
      }
      document.documentElement.style.setProperty("--accent", controls.robotColor.value);
    }

    function cameraTargetString(target) {
      return `$${target.x}m $${target.y}m $${target.z}m`;
    }

    function cameraOrbitString(orbit) {
      return `$${orbit.theta}rad $${orbit.phi}rad $${orbit.radius}m`;
    }

    viewer.addEventListener("load", () => {
      applyMaterials();
      status.textContent = "Top-down MuJoCo scene loaded. Drag to orbit, then Capture PNG.";
    });

    Object.values(controls).forEach((input) => {
      input.addEventListener("input", applyMaterials);
      input.addEventListener("change", applyMaterials);
    });

    document.querySelectorAll("[data-orbit]").forEach((button) => {
      button.addEventListener("click", () => {
        viewer.cameraTarget = defaults.target;
        viewer.cameraOrbit = button.dataset.orbit;
      });
    });

    document.querySelector("#reset").addEventListener("click", () => {
      viewer.cameraTarget = defaults.target;
      viewer.cameraOrbit = defaults.orbit;
      controls.robotColor.value = defaults.robotColor;
      controls.robotAlpha.value = defaults.robotAlpha;
      controls.humanColor.value = defaults.humanColor;
      controls.humanAlpha.value = defaults.humanAlpha;
      controls.humanVisible.checked = defaults.humanVisible;
      controls.trajectoryVisible.checked = defaults.trajectoryVisible;
      controls.eeTrajectoryVisible.checked = defaults.eeTrajectoryVisible;
      controls.eeTrajectoryColor.value = defaults.eeTrajectoryColor;
      controls.eeTrajectoryAlpha.value = defaults.eeTrajectoryAlpha;
      controls.cabinetAlpha.value = defaults.cabinetAlpha;
      applyMaterials();
    });

    document.querySelector("#copyCamera").addEventListener("click", async () => {
      const text = `camera-target="$${cameraTargetString(viewer.getCameraTarget())}" camera-orbit="$${cameraOrbitString(viewer.getCameraOrbit())}"`;
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
      link.download = "mujoco_h1_human_arm_topdown.png";
      link.href = viewer.toDataURL("image/png");
      link.click();
      status.textContent = "PNG captured.";
    });
  </script>
</body>
</html>
''')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a draggable top-down GLB/HTML viewer from the real MuJoCo H1 + human arm scene."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", default="mujoco_h1_human_arm_topdown_viewer")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DEMO_MANIFEST)
    parser.add_argument("--demo-path", type=Path, default=None)
    parser.add_argument("--human-pose", choices=("outside", "block"), default="block")
    parser.add_argument("--robot-color", default="#5b8fc9")
    parser.add_argument("--robot-alpha", type=float, default=0.55)
    parser.add_argument("--human-color", default="#d14b3f")
    parser.add_argument("--human-alpha", type=float, default=0.78)
    parser.add_argument("--cabinet-alpha", type=float, default=0.36)
    parser.add_argument("--trajectory-radius", type=float, default=0.012)
    parser.add_argument("--trajectory-sample-every", type=int, default=1)
    parser.add_argument("--trajectory-extra-time", type=float, default=0.30)
    parser.add_argument("--robot-ee-trajectory-color", default="#1769c2")
    parser.add_argument("--robot-ee-trajectory-alpha", type=float, default=0.92)
    parser.add_argument("--robot-ee-trajectory-radius", type=float, default=0.010)
    parser.add_argument("--robot-ee-trajectory-sample-every", type=int, default=1)
    parser.add_argument("--robot-ee-trajectory-max-steps", type=int, default=None)
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


def make_env(args):
    return HumanArmDrawerTopOpen(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        observation_config=ObservationConfig(cameras=[], proprioception=True, privileged_information=True),
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
    elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
        count = int(model.nsite)
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


def set_human_pose(model, data, pose: str):
    q_values = Q_BLOCK if pose == "block" else Q_OUTSIDE
    for joint_name, value in zip(HUMAN_JOINT_NAMES, q_values):
        joint_id = resolve_mujoco_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = float(value)
        data.qvel[qvel_adr] = 0.0
    mujoco.mj_forward(model, data)


def capsule_axis_from_geom(model, data, geom_id: int, fallback_half_length: float):
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    axis = rot[:, 2]
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    half_length = float(size[1]) if len(size) > 1 and float(size[1]) > 1e-8 else fallback_half_length
    radius = float(size[0]) if len(size) else 0.0
    return center - half_length * axis, center + half_length * axis, center, radius


def sample_temporal_blocker_state(model, data, ids, qpos_adrs, step: int, time_s: float, info: dict):
    upper_a, upper_b, upper_center, _upper_radius = capsule_axis_from_geom(
        model,
        data,
        ids["upperarm_geom"],
        0.34 / 2.0,
    )
    fore_a, fore_b, fore_center, _fore_radius = capsule_axis_from_geom(
        model,
        data,
        ids["forearm_geom"],
        0.30 / 2.0,
    )
    handle = np.asarray(data.site_xpos[ids["handle_site"]], dtype=np.float64)
    fore_endpoints = np.stack([fore_a, fore_b], axis=0)
    hand = fore_endpoints[np.argmin(np.linalg.norm(fore_endpoints - handle, axis=1))]
    return {
        "step": int(step),
        "time": float(time_s),
        "human_phase": str(info.get("human_phase", "")),
        "human_blocker_triggered": bool(info.get("human_blocker_triggered", False)),
        "ee_to_handle_dist": float(info.get("ee_to_handle_dist", np.nan)),
        "min_robot_human_distance": (
            None
            if info.get("min_robot_human_distance") is None
            else float(info.get("min_robot_human_distance"))
        ),
        "qpos": [float(data.qpos[adr]) for adr in qpos_adrs],
        "upperarm_center_mujoco": upper_center.tolist(),
        "forearm_center_mujoco": fore_center.tolist(),
        "hand_mujoco": np.asarray(hand, dtype=np.float64).tolist(),
        "drawer_handle_mujoco": handle.tolist(),
    }


def sample_temporal_blocker_trajectory(env, args):
    blocker = getattr(env, "_temporary_human_blocker", None)
    if blocker is None:
        return []

    model = env.mojo.physics.model.ptr
    data = env.mojo.physics.data.ptr
    ids = {
        "upperarm_geom": resolve_mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_arm/upperarm_geom"),
        "forearm_geom": resolve_mujoco_name(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_arm/forearm_geom"),
        "handle_site": resolve_mujoco_name(model, mujoco.mjtObj.mjOBJ_SITE, "drawer_small_4"),
    }
    qpos_adrs = [
        int(model.jnt_qposadr[resolve_mujoco_name(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in HUMAN_JOINT_NAMES
    ]

    dt = float(env.get_dt())
    total_duration = 1.2 + 1.2 + 1.2 + max(0.0, float(args.trajectory_extra_time))
    total_steps = max(1, int(np.ceil(total_duration / max(dt, 1e-9))))
    sample_every = max(1, int(args.trajectory_sample_every))

    info = dict(blocker.last_info)
    samples = [sample_temporal_blocker_state(model, data, ids, qpos_adrs, 0, 0.0, info)]
    last_phase = samples[-1]["human_phase"]
    for step in range(1, total_steps + 1):
        info = blocker.update(dt)
        phase = str(info.get("human_phase", ""))
        should_sample = step % sample_every == 0 or phase != last_phase or step == total_steps
        if should_sample:
            samples.append(sample_temporal_blocker_state(model, data, ids, qpos_adrs, step, step * dt, info))
        last_phase = phase
    return samples



def successful_demo_path(manifest_path: Path) -> Path:
    records = json.loads(manifest_path.read_text())
    successful = [
        record
        for record in records
        if int(record.get("success", 0)) == 1 and record.get("target_path")
    ]
    if not successful:
        raise RuntimeError(f"No successful demo found in {manifest_path}")
    successful.sort(
        key=lambda record: (
            int(record.get("num_timesteps", 10**9)),
            str(record.get("target_path")),
        )
    )
    return Path(successful[0]["target_path"])


def clamp_action(env, action):
    action = np.asarray(action, dtype=np.float32)
    return np.clip(action, env.action_space.low, env.action_space.high)


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def sample_robot_ee_trajectory(args):
    demo_path = args.demo_path or successful_demo_path(args.manifest.expanduser())
    demo = Demo.from_safetensors(demo_path)
    if demo is None:
        raise RuntimeError(f"Could not load demo: {demo_path}")

    env = make_env(args)
    try:
        _obs, info = env.reset(seed=args.seed)
        model = env.mojo.physics.model.ptr
        data = env.mojo.physics.data.ptr
        ee_site = resolve_mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "right_end_effector",
        )
        sample_every = max(1, int(args.robot_ee_trajectory_sample_every))
        max_steps = len(demo.timesteps)
        if args.robot_ee_trajectory_max_steps is not None:
            max_steps = min(max_steps, int(args.robot_ee_trajectory_max_steps))

        samples = [
            {
                "step": 0,
                "time": float(data.time),
                "ee_mujoco": np.asarray(data.site_xpos[ee_site], dtype=np.float64).tolist(),
                "human_phase": str(info.get("human_phase", "")),
            }
        ]
        for step_i in range(max_steps):
            action = clamp_action(env, demo.timesteps[step_i].executed_action)
            _obs, _reward, terminated, truncated, info = env.step(action)
            should_sample = (
                (step_i + 1) % sample_every == 0
                or step_i + 1 == max_steps
                or terminated
                or truncated
            )
            if should_sample:
                samples.append(
                    {
                        "step": int(step_i + 1),
                        "time": float(data.time),
                        "ee_mujoco": np.asarray(data.site_xpos[ee_site], dtype=np.float64).tolist(),
                        "human_phase": str(info.get("human_phase", "")),
                    }
                )
            if terminated or truncated:
                break
    finally:
        env.close()

    points = np.asarray([sample["ee_mujoco"] for sample in samples], dtype=np.float64)
    point_min = points.min(axis=0).tolist() if len(points) else None
    point_max = points.max(axis=0).tolist() if len(points) else None
    summary = {
        "demo_path": str(demo_path),
        "num_samples": len(samples),
        "num_recorded_steps": int(samples[-1]["step"]) if samples else 0,
        "path_length_m": path_length(points),
        "xyz_min_mujoco": point_min,
        "xyz_max_mujoco": point_max,
        "trajectory_radius_m": float(args.robot_ee_trajectory_radius),
        "sample_every_steps": sample_every,
    }
    return samples, summary


def make_capsule_between(start, end, radius: float, material, face_color):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-7:
        return None
    transform = trimesh.geometry.align_vectors([0, 0, 1], axis / length)
    transform[:3, 3] = 0.5 * (start + end)
    mesh = trimesh.creation.capsule(
        radius=float(radius),
        height=length,
        count=[16, 8],
        transform=transform,
    )
    mesh.visual.face_colors = face_color
    mesh.visual = TextureVisuals(material=material)
    return mesh



def add_points_trajectory_to_scene(scene, points, radius: float, material, face_color, prefix: str):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return 0

    count = 0
    for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
        mesh = make_capsule_between(start, end, radius, material, face_color)
        if mesh is None:
            continue
        mesh.apply_transform(MUJOCO_TO_VIEWER)
        scene.add_geometry(
            mesh,
            geom_name=f"{prefix}_{i:04d}",
            node_name=f"{prefix}_{i:04d}",
        )
        count += 1

    marker_points = [("start", points[0])]
    if len(points) > 1 and float(np.linalg.norm(points[-1] - points[0])) > radius:
        marker_points.append(("end", points[-1]))
    for label, point in marker_points:
        marker = trimesh.creation.icosphere(subdivisions=2, radius=radius * 2.4)
        marker.apply_translation(point)
        marker.apply_transform(MUJOCO_TO_VIEWER)
        marker.visual.face_colors = face_color
        marker.visual = TextureVisuals(material=material)
        scene.add_geometry(
            marker,
            geom_name=f"{prefix}_{label}",
            node_name=f"{prefix}_{label}",
        )
        count += 1
    return count


def add_temporal_trajectory_to_scene(scene, trajectory_samples, radius: float, material, face_color):
    points = np.asarray(
        [sample["hand_mujoco"] for sample in trajectory_samples],
        dtype=np.float64,
    )
    if len(points) < 2:
        return 0

    count = 0
    for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
        mesh = make_capsule_between(start, end, radius, material, face_color)
        if mesh is None:
            continue
        mesh.apply_transform(MUJOCO_TO_VIEWER)
        scene.add_geometry(
            mesh,
            geom_name=f"trajectory_temporal_blocker_hand_{i:03d}",
            node_name=f"trajectory_temporal_blocker_hand_{i:03d}",
        )
        count += 1

    for label, point in (("start", points[0]), ("end", points[-1])):
        marker = trimesh.creation.icosphere(subdivisions=2, radius=radius * 2.2)
        marker.apply_translation(point)
        marker.apply_transform(MUJOCO_TO_VIEWER)
        marker.visual.face_colors = face_color
        marker.visual = TextureVisuals(material=material)
        scene.add_geometry(
            marker,
            geom_name=f"trajectory_temporal_blocker_hand_{label}",
            node_name=f"trajectory_temporal_blocker_hand_{label}",
        )
        count += 1
    return count


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
        height = max(float(size[1]) * 2.0, 1e-6)
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=height, count=[24, 12])
        mesh.apply_transform(transform)
        return mesh
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        height = max(float(size[1]) * 2.0, 1e-6)
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=height, sections=32)
        mesh.apply_transform(transform)
        return mesh
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        mesh = trimesh.creation.box(extents=np.maximum(size[:3] * 2.0, 1e-6), transform=transform)
        return mesh
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


def geom_category(name: str):
    if name.startswith("h1/"):
        return "robot"
    if name.startswith("cylinder_arm/"):
        if name.rsplit("/", 1)[-1] in {"axis_x", "axis_y", "axis_z"}:
            return None
        return "human"
    if name.startswith("base_cabinet_600/"):
        return "cabinet"
    return None


def add_geom_to_scene(scene, model, data, geom_id: int, category: str, material, face_color):
    mesh = mesh_from_geom(model, data, geom_id)
    if mesh is None:
        return False
    mesh.apply_transform(MUJOCO_TO_VIEWER)
    mesh.visual.face_colors = face_color
    mesh.visual = TextureVisuals(material=material)
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    safe_name = name.replace("/", "_").replace(" ", "_")
    scene.add_geometry(mesh, geom_name=f"{category}_{geom_id}_{safe_name}", node_name=f"{category}_{geom_id}_{safe_name}")
    return True


def scene_bounds(scene: trimesh.Scene) -> dict:
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    radius = max(float(np.linalg.norm(extents) * 0.5), 0.5)
    return {"min": bounds[0].tolist(), "max": bounds[1].tolist(), "center": center.tolist(), "radius": radius}


def build_scene(args):
    robot_material = make_material("robot_material", args.robot_color, args.robot_alpha)
    human_material = make_material("human_material", args.human_color, args.human_alpha)
    cabinet_material = make_material("cabinet_material", "#b6aa98", args.cabinet_alpha)
    trajectory_material = make_material("trajectory_material", "#dd7a22", 0.90)
    trajectory_color = rgba_u8("#dd7a22", 0.90)
    ee_trajectory_material = make_material(
        "ee_trajectory_material",
        args.robot_ee_trajectory_color,
        args.robot_ee_trajectory_alpha,
    )
    ee_trajectory_color = rgba_u8(
        args.robot_ee_trajectory_color,
        args.robot_ee_trajectory_alpha,
    )
    materials = {
        "robot": (robot_material, rgba_u8(args.robot_color, args.robot_alpha)),
        "human": (human_material, rgba_u8(args.human_color, args.human_alpha)),
        "cabinet": (cabinet_material, rgba_u8("#b6aa98", args.cabinet_alpha)),
    }

    robot_ee_trajectory, robot_ee_trajectory_summary = sample_robot_ee_trajectory(args)
    robot_ee_points = np.asarray(
        [sample["ee_mujoco"] for sample in robot_ee_trajectory],
        dtype=np.float64,
    )

    env = make_env(args)
    try:
        env.reset(seed=args.seed)
        model = env.mojo.physics.model.ptr
        data = env.mojo.physics.data.ptr
        temporal_blocker_trajectory = sample_temporal_blocker_trajectory(env, args)
        set_human_pose(model, data, args.human_pose)

        scene = trimesh.Scene()
        counts = {"robot": 0, "human": 0, "cabinet": 0, "trajectory": 0, "robot_ee_trajectory": 0}
        for geom_id in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            category = geom_category(name)
            if category is None:
                continue
            material, face_color = materials[category]
            if add_geom_to_scene(scene, model, data, geom_id, category, material, face_color):
                counts[category] += 1

        counts["trajectory"] = add_temporal_trajectory_to_scene(
            scene,
            temporal_blocker_trajectory,
            args.trajectory_radius,
            trajectory_material,
            trajectory_color,
        )
        counts["robot_ee_trajectory"] = add_points_trajectory_to_scene(
            scene,
            robot_ee_points,
            args.robot_ee_trajectory_radius,
            ee_trajectory_material,
            ee_trajectory_color,
            "ee_trajectory_right_end_effector",
        )

        trajectory_points = np.asarray(
            [sample["hand_mujoco"] for sample in temporal_blocker_trajectory],
            dtype=np.float64,
        )
        if len(trajectory_points) > 1:
            trajectory_path_length = float(
                np.linalg.norm(trajectory_points[1:] - trajectory_points[:-1], axis=1).sum()
            )
        else:
            trajectory_path_length = 0.0
        trajectory_phase_counts = {
            phase: int(sum(sample["human_phase"] == phase for sample in temporal_blocker_trajectory))
            for phase in sorted({sample["human_phase"] for sample in temporal_blocker_trajectory})
        }

        bounds = scene_bounds(scene)
        ee_site = resolve_mujoco_name(model, mujoco.mjtObj.mjOBJ_SITE, "right_end_effector")
        handle_site = resolve_mujoco_name(model, mujoco.mjtObj.mjOBJ_SITE, "drawer_small_4")
        metadata = {
            "env": "HumanArmDrawerTopOpen",
            "seed": args.seed,
            "human_pose": args.human_pose,
            "geom_counts": counts,
            "temporal_blocker_trajectory_summary": {
                "num_samples": len(temporal_blocker_trajectory),
                "path_length_m": trajectory_path_length,
                "phase_counts": trajectory_phase_counts,
                "trajectory_radius_m": float(args.trajectory_radius),
                "sample_every_steps": int(args.trajectory_sample_every),
            },
            "temporal_blocker_trajectory": temporal_blocker_trajectory,
            "robot_ee_trajectory_summary": robot_ee_trajectory_summary,
            "robot_ee_trajectory": robot_ee_trajectory,
            "human_joint_names": HUMAN_JOINT_NAMES,
            "scene_bounds": bounds,
            "axis_transform": "MuJoCo (x,y,z) -> viewer/glTF (x,z,-y)",
            "right_end_effector_mujoco": np.asarray(data.site_xpos[ee_site], dtype=np.float64).tolist(),
            "drawer_handle_mujoco": np.asarray(data.site_xpos[handle_site], dtype=np.float64).tolist(),
            "note": "GLB geometry is exported from the actual compiled MuJoCo geoms for h1/, cylinder_arm/, and base_cabinet_600/. The temporal blocker trajectory is logged from TemporaryDrawerArmBlocker.update(dt), and the robot right end-effector trajectory is sampled while replaying the selected DrawerTopOpen demo.",
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
    distance = max(1.2, radius * 2.2)
    target = f"{center[0]:.5f}m {center[1]:.5f}m {center[2]:.5f}m"
    top_orbit = f"0deg 0deg {distance:.5f}m"
    iso_orbit = f"45deg 58deg {distance:.5f}m"
    front_orbit = f"0deg 82deg {distance:.5f}m"
    side_orbit = f"90deg 82deg {distance:.5f}m"
    glb_data_uri = "data:model/gltf-binary;base64," + base64.b64encode(glb_path.read_bytes()).decode("ascii")

    html = HTML_TEMPLATE.substitute(
        glb_data_uri=glb_data_uri,
        glb_filename=glb_path.name,
        metadata_json=json.dumps(metadata, indent=2),
        target=target,
        top_orbit=top_orbit,
        iso_orbit=iso_orbit,
        front_orbit=front_orbit,
        side_orbit=side_orbit,
        robot_color=args.robot_color,
        robot_alpha=f"{args.robot_alpha:.2f}",
        human_color=args.human_color,
        human_alpha=f"{args.human_alpha:.2f}",
        ee_trajectory_color=args.robot_ee_trajectory_color,
        ee_trajectory_alpha=f"{args.robot_ee_trajectory_alpha:.2f}",
        cabinet_alpha=f"{args.cabinet_alpha:.2f}",
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
    print("geom_counts:", metadata["geom_counts"])
    print("human_pose:", metadata["human_pose"])


if __name__ == "__main__":
    main()
