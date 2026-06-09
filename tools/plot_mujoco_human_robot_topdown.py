from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
BIGYM_ROOT = REPO / "bigym"
ROBOBASE_ROOT = REPO / "robobase"
for path in (BIGYM_ROOT, ROBOBASE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bigym.action_modes import JointPositionActionMode, PelvisDof  # noqa: E402
from bigym.envs.cupboards_with_human_arm import HumanArmDrawerTopOpen  # noqa: E402
from bigym.utils.observation_config import ObservationConfig  # noqa: E402
from demonstrations.demo import Demo  # noqa: E402


DEFAULT_DEMO_MANIFEST = Path(
    "~/.bigym/demonstrations/0.9.0/DrawerTopOpen/"
    "JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/"
    "lightweight/manifest.json"
).expanduser()

DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "eval_safety/h1_right_arm_barrier_overlay"
)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a real MuJoCo BiGym drawer demo and draw top-down robot/human "
            "arm trajectories with the temporary human blocker enabled."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DEMO_MANIFEST)
    parser.add_argument("--demo-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--trigger-dist", type=float, default=1.20)
    parser.add_argument("--enter-duration", type=float, default=1.2)
    parser.add_argument("--hold-duration", type=float, default=1.2)
    parser.add_argument("--exit-duration", type=float, default=1.2)
    parser.add_argument("--natural-motion-scale", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--basename",
        default="real_mujoco_topdown_human_robot_trajectory",
    )
    return parser.parse_args()


def resolve_mujoco_name(model, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id >= 0:
        return int(obj_id)

    if obj_type == mujoco.mjtObj.mjOBJ_SITE:
        count = model.nsite
    elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
        count = model.ngeom
    elif obj_type == mujoco.mjtObj.mjOBJ_BODY:
        count = model.nbody
    else:
        count = 0

    matches = []
    for candidate_id in range(int(count)):
        candidate = mujoco.mj_id2name(model, obj_type, candidate_id) or ""
        if candidate == name or candidate.endswith(f"/{name}"):
            matches.append(candidate_id)
    if len(matches) == 1:
        return int(matches[0])
    if not matches:
        raise ValueError(f"MuJoCo name {name!r} not found")
    raise ValueError(f"MuJoCo name {name!r} is ambiguous: {matches}")


def successful_demo_path(manifest_path: Path) -> Path:
    records = json.loads(manifest_path.read_text())
    successful = [
        record
        for record in records
        if int(record.get("success", 0)) == 1
        and record.get("target_path")
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


def make_env(args):
    return HumanArmDrawerTopOpen(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[
                PelvisDof.X,
                PelvisDof.Y,
                PelvisDof.Z,
                PelvisDof.RZ,
            ],
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
        trigger_dist=args.trigger_dist,
        enter_duration=args.enter_duration,
        hold_duration=args.hold_duration,
        exit_duration=args.exit_duration,
        natural_motion_scale=args.natural_motion_scale,
        human_joint_names=HUMAN_JOINT_NAMES,
        q_outside=Q_OUTSIDE,
        q_block=Q_BLOCK,
        q_exit=None,
        ee_site_name="right_end_effector",
        handle_site_name="drawer_small_4",
    )


def clamp_action(env, action):
    action = np.asarray(action, dtype=np.float32)
    return np.clip(action, env.action_space.low, env.action_space.high)


def capsule_from_geom(model, data, geom_id: int, fallback_half_length: float):
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    axis = rot[:, 2]
    radius = float(model.geom_size[geom_id, 0])
    half_length = fallback_half_length
    return {
        "a": center - half_length * axis,
        "b": center + half_length * axis,
        "center": center,
        "radius": radius,
    }


def sample_state(model, data, ids, step: int, info: dict):
    upper = capsule_from_geom(model, data, ids["upperarm_geom"], 0.34 / 2.0)
    fore = capsule_from_geom(model, data, ids["forearm_geom"], 0.30 / 2.0)
    # In top-down view the most useful hand proxy is the forearm endpoint
    # closest to the drawer handle.
    handle = np.asarray(data.site_xpos[ids["handle_site"]], dtype=np.float64)
    fore_endpoints = np.stack([fore["a"], fore["b"]], axis=0)
    hand = fore_endpoints[np.argmin(np.linalg.norm(fore_endpoints - handle, axis=1))]
    return {
        "step": int(step),
        "time": float(data.time),
        "robot_ee": np.asarray(data.site_xpos[ids["ee_site"]], dtype=np.float64),
        "drawer_handle": handle,
        "human_hand": hand,
        "upper_capsule": upper,
        "fore_capsule": fore,
        "human_phase": str(info.get("human_phase", "")),
        "human_blocker_triggered": bool(info.get("human_blocker_triggered", False)),
        "ee_to_handle_dist": float(info.get("ee_to_handle_dist", np.nan)),
        "min_robot_human_distance": (
            None
            if info.get("min_robot_human_distance") is None
            else float(info.get("min_robot_human_distance"))
        ),
        "drawer_open_distance": (
            None
            if info.get("drawer_open_distance") is None
            else float(info.get("drawer_open_distance"))
        ),
    }


def topdown_box_vertices(model, data, geom_id: int):
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    sx = float(size[0])
    sy = float(size[1])
    corners = np.asarray(
        [[-sx, -sy, 0.0], [sx, -sy, 0.0], [sx, sy, 0.0], [-sx, sy, 0.0]],
        dtype=np.float64,
    )
    world = center + corners @ rot.T
    return world[:, :2]


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 3:
        return points
    points = np.unique(points[:, :2], axis=0)
    if len(points) <= 3:
        return points

    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def capsule_polygon_xy(a, b, radius: float, segments: int = 12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)[:2]
    b = np.asarray(b, dtype=np.float64)[:2]
    delta = b - a
    length = float(np.linalg.norm(delta))
    radius = float(radius)
    if length < 1e-8:
        theta = np.linspace(0.0, 2.0 * np.pi, 2 * segments, endpoint=False)
        return np.column_stack([a[0] + radius * np.cos(theta), a[1] + radius * np.sin(theta)])

    direction = delta / length
    angle = np.arctan2(direction[1], direction[0])
    cap_a = angle + np.linspace(np.pi / 2.0, 3.0 * np.pi / 2.0, segments)
    cap_b = angle + np.linspace(-np.pi / 2.0, np.pi / 2.0, segments)
    points_a = np.column_stack([a[0] + radius * np.cos(cap_a), a[1] + radius * np.sin(cap_a)])
    points_b = np.column_stack([b[0] + radius * np.cos(cap_b), b[1] + radius * np.sin(cap_b)])
    return np.vstack([points_a, points_b])


def mesh_topdown_hull(model, data, geom_id: int):
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    if vert_num <= 0:
        return None

    local = np.asarray(model.mesh_vert[vert_adr:vert_adr + vert_num], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    world = center + local @ rot.T
    return convex_hull_xy(world[:, :2])


def robot_topdown_geometry(model, data):
    shapes = []
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("h1/"):
            continue
        geom_type = int(model.geom_type[geom_id])
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            hull = mesh_topdown_hull(model, data, geom_id)
            if hull is not None and len(hull) >= 3:
                shapes.append({"kind": "polygon", "points": hull, "name": name})
                continue
            radius = float(model.geom_rbound[geom_id])
            shapes.append({"kind": "circle", "center": center[:2], "radius": radius, "name": name})
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            shapes.append({"kind": "polygon", "points": topdown_box_vertices(model, data, geom_id), "name": name})
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            axis = rot[:, 2]
            radius = float(size[0])
            half_length = float(size[1])
            a = center - half_length * axis
            b = center + half_length * axis
            shapes.append({"kind": "polygon", "points": capsule_polygon_xy(a, b, radius), "name": name})
        elif geom_type in (
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        ):
            radius = float(max(size[0], size[1] if len(size) > 1 else size[0]))
            shapes.append({"kind": "circle", "center": center[:2], "radius": radius, "name": name})
        else:
            radius = float(model.geom_rbound[geom_id])
            shapes.append({"kind": "circle", "center": center[:2], "radius": radius, "name": name})
    return shapes


def draw_robot_topdown_snapshot(ax, shapes, label="H1 spawn pose top-down geometry"):
    first = True
    for shape in shapes:
        patch_label = label if first else None
        first = False
        if shape["kind"] == "polygon":
            ax.add_patch(
                Polygon(
                    np.asarray(shape["points"], dtype=np.float64),
                    closed=True,
                    facecolor="#5b8fc9",
                    edgecolor="#194f88",
                    linewidth=0.45,
                    alpha=0.24,
                    zorder=3,
                    label=patch_label,
                )
            )
        elif shape["kind"] == "circle":
            ax.add_patch(
                Circle(
                    np.asarray(shape["center"], dtype=np.float64),
                    float(shape["radius"]),
                    facecolor="#5b8fc9",
                    edgecolor="#194f88",
                    linewidth=0.45,
                    alpha=0.24,
                    zorder=3,
                    label=patch_label,
                )
            )


def topdown_shape_points(shapes):
    points = []
    for shape in shapes:
        if shape["kind"] == "polygon":
            points.append(np.asarray(shape["points"], dtype=np.float64))
        elif shape["kind"] == "circle":
            center = np.asarray(shape["center"], dtype=np.float64)
            radius = float(shape["radius"])
            points.append(
                np.asarray(
                    [
                        center + [radius, 0.0],
                        center + [-radius, 0.0],
                        center + [0.0, radius],
                        center + [0.0, -radius],
                    ],
                    dtype=np.float64,
                )
            )
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(points)


def drawer_geometry(model, data):
    body_id = resolve_mujoco_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "drawer_small_4",
    )
    cabinet_prefix = "base_cabinet_600"
    cabinet_boxes = []
    drawer_boxes = []
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if name.startswith(cabinet_prefix):
            cabinet_boxes.append(topdown_box_vertices(model, data, geom_id))
        if int(model.geom_bodyid[geom_id]) == body_id:
            drawer_boxes.append(topdown_box_vertices(model, data, geom_id))
    return cabinet_boxes, drawer_boxes


def path_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.stack([points[:-1], points[1:]], axis=1)


def add_gradient_path(ax, points, color, label, linewidth=3.0, alpha=0.95):
    segments = path_segments(np.asarray(points, dtype=np.float64))
    if len(segments) == 0:
        return
    fade = np.linspace(0.25, alpha, len(segments))
    colors = []
    base = matplotlib.colors.to_rgba(color)
    for value in fade:
        colors.append((base[0], base[1], base[2], float(value)))
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=linewidth,
        capstyle="round",
        joinstyle="round",
        label=label,
        zorder=5,
    )
    ax.add_collection(collection)


def add_small_arrows(ax, points, color, count=6):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return
    candidate = np.linspace(1, len(points) - 1, min(count, len(points) - 1), dtype=int)
    for idx in candidate:
        start = points[idx - 1]
        end = points[idx]
        delta = end - start
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            continue
        delta = delta / norm * min(0.035, norm)
        ax.annotate(
            "",
            xy=(start[0] + delta[0], start[1] + delta[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.1,
                "mutation_scale": 9,
                "shrinkA": 0,
                "shrinkB": 0,
                "alpha": 0.9,
            },
            zorder=7,
        )


def draw_capsule_snapshot(ax, sample, color, alpha=0.35, label=None):
    for key in ("upper_capsule", "fore_capsule"):
        cap = sample[key]
        a = np.asarray(cap["a"], dtype=np.float64)[:2]
        b = np.asarray(cap["b"], dtype=np.float64)[:2]
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=color,
            linewidth=8,
            alpha=alpha,
            solid_capstyle="round",
            zorder=4,
            label=label if key == "upper_capsule" else None,
        )


def first_index_for_phase(samples: list[dict], phases: Iterable[str]):
    phases = set(phases)
    for i, sample in enumerate(samples):
        if sample["human_phase"] in phases:
            return i
    return None


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def plot(samples: list[dict], model, data, args, demo_path: Path, robot_shapes):
    robot_xy = np.asarray([sample["robot_ee"][:2] for sample in samples])
    human_xy = np.asarray([sample["human_hand"][:2] for sample in samples])
    handle_xy = np.asarray(samples[0]["drawer_handle"][:2])
    cabinet_boxes, drawer_boxes = drawer_geometry(model, data)

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    fig.patch.set_facecolor("#f7f4ef")
    ax.set_facecolor("#fbfaf7")

    for box in cabinet_boxes:
        ax.add_patch(
            Polygon(
                box,
                closed=True,
                facecolor="#d9d1c3",
                edgecolor="#a69c8f",
                linewidth=0.8,
                alpha=0.38,
                zorder=1,
            )
        )
    for i, box in enumerate(drawer_boxes):
        ax.add_patch(
            Polygon(
                box,
                closed=True,
                facecolor="#cfe7d0",
                edgecolor="#3f7f55",
                linewidth=1.4,
                alpha=0.62,
                zorder=2,
                label="Drawer body" if i == 0 else None,
            )
        )

    draw_robot_topdown_snapshot(ax, robot_shapes)

    enter_i = first_index_for_phase(samples, {"enter"})
    hold_i = first_index_for_phase(samples, {"hold"})
    snapshot_indices = [
        ("Human blocking pose", hold_i if hold_i is not None else enter_i, "#d14b3f", 0.42),
    ]
    used_snapshot_labels = set()
    for label, idx, color, alpha in snapshot_indices:
        if idx is None:
            continue
        draw_capsule_snapshot(
            ax,
            samples[int(idx)],
            color=color,
            alpha=alpha,
            label=None if label in used_snapshot_labels else label,
        )
        used_snapshot_labels.add(label)

    add_gradient_path(ax, robot_xy, "#1769c2", "Robot right end-effector path", linewidth=3.5)
    add_gradient_path(ax, human_xy, "#dd7a22", "Human forearm/hand path", linewidth=3.2)
    add_small_arrows(ax, robot_xy[:: max(1, len(robot_xy) // 120)], "#1769c2", count=5)
    add_small_arrows(ax, human_xy[:: max(1, len(human_xy) // 120)], "#dd7a22", count=5)

    ax.scatter(
        [robot_xy[-1, 0]],
        [robot_xy[-1, 1]],
        s=68,
        marker="x",
        color="#0d4f99",
        linewidth=1.8,
        zorder=9,
        label="Right end-effector site",
    )
    ax.scatter(
        [human_xy[0, 0]],
        [human_xy[0, 1]],
        s=42,
        marker="o",
        color="#dd7a22",
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
        label="Human start",
    )
    ax.scatter(
        [human_xy[-1, 0]],
        [human_xy[-1, 1]],
        s=58,
        marker="s",
        color="#dd7a22",
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
        label="Human end",
    )
    ax.scatter(
        [handle_xy[0]],
        [handle_xy[1]],
        marker="*",
        s=170,
        color="#2f7d4e",
        edgecolor="white",
        linewidth=0.8,
        zorder=9,
        label="Drawer handle site",
    )

    ax.annotate(
        "drawer location\nhandle site: drawer_small_4",
        xy=(handle_xy[0], handle_xy[1]),
        xytext=(handle_xy[0] + 0.10, handle_xy[1] + 0.09),
        ha="left",
        va="center",
        fontsize=10,
        color="#244432",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#fbfaf7", "ec": "#91b99a", "alpha": 0.92},
        arrowprops={
            "arrowstyle": "->",
            "color": "#2f7d4e",
            "lw": 1.0,
            "mutation_scale": 8,
            "shrinkA": 3,
            "shrinkB": 4,
        },
        zorder=10,
    )

    all_xy = np.vstack([robot_xy, human_xy, handle_xy.reshape(1, 2)])
    robot_points = topdown_shape_points(robot_shapes)
    if len(robot_points):
        all_xy = np.vstack([all_xy, robot_points])
    for box in cabinet_boxes + drawer_boxes:
        all_xy = np.vstack([all_xy, np.asarray(box)])
    pad = 0.18
    xy_min = all_xy.min(axis=0) - pad
    xy_max = all_xy.max(axis=0) + pad
    span = xy_max - xy_min
    if span[0] > span[1]:
        extra = (span[0] - span[1]) * 0.5
        xy_min[1] -= extra
        xy_max[1] += extra
    else:
        extra = (span[1] - span[0]) * 0.5
        xy_min[0] -= extra
        xy_max[0] += extra

    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d8d5cf", linewidth=0.6, alpha=0.75)
    ax.set_xlabel("MuJoCo world x (m)")
    ax.set_ylabel("MuJoCo world y (m)")
    ax.set_title(
        "Top-down MuJoCo trajectory: H1 right arm vs temporary human arm blocker",
        fontsize=14,
        pad=12,
    )
    ax.text(
        0.012,
        0.012,
        f"Demo replay: {demo_path.name} | samples: {len(samples)}",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#5f5b53",
        ha="left",
        va="bottom",
    )
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        facecolor="#fbfaf7",
        edgecolor="#c9c4ba",
        fontsize=8.5,
        borderpad=0.7,
        labelspacing=0.45,
    )
    legend.set_zorder(20)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"{args.basename}.png"
    svg_path = args.out_dir / f"{args.basename}.svg"
    json_path = args.out_dir / f"{args.basename}.json"
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)

    triggered = [sample for sample in samples if sample["human_blocker_triggered"]]
    min_clearances = [
        sample["min_robot_human_distance"]
        for sample in samples
        if sample["min_robot_human_distance"] is not None
    ]
    metadata = {
        "demo_path": str(demo_path),
        "seed": args.seed,
        "sample_every": args.sample_every,
        "num_samples": len(samples),
        "num_recorded_steps": int(samples[-1]["step"]),
        "trigger_dist": args.trigger_dist,
        "phase_counts": {
            phase: int(sum(sample["human_phase"] == phase for sample in samples))
            for phase in sorted({sample["human_phase"] for sample in samples})
        },
        "human_blocker_triggered": bool(triggered),
        "first_triggered_step": None if not triggered else int(triggered[0]["step"]),
        "min_robot_human_distance": None if not min_clearances else float(np.min(min_clearances)),
        "final_drawer_open_distance": samples[-1]["drawer_open_distance"],
        "site_names": {
            "robot_ee": "h1/right_end_effector",
            "drawer_handle": "base_cabinet_600/drawer_small_4",
        },
        "note": (
            "Trajectories were sampled from MuJoCo site/geom world positions while "
            "replaying a cached successful DrawerTopOpen action demo in "
            "HumanArmDrawerTopOpen with enable_temporary_human_blocker=True."
        ),
        "samples": [
            {
                "step": sample["step"],
                "time": sample["time"],
                "robot_ee": sample["robot_ee"],
                "human_hand": sample["human_hand"],
                "drawer_handle": sample["drawer_handle"],
                "human_phase": sample["human_phase"],
                "human_blocker_triggered": sample["human_blocker_triggered"],
                "ee_to_handle_dist": sample["ee_to_handle_dist"],
                "min_robot_human_distance": sample["min_robot_human_distance"],
                "drawer_open_distance": sample["drawer_open_distance"],
            }
            for sample in samples
        ],
    }
    json_path.write_text(json.dumps(to_jsonable(metadata), indent=2))
    return png_path, svg_path, json_path, metadata


def main():
    args = parse_args()
    demo_path = args.demo_path or successful_demo_path(args.manifest.expanduser())
    demo = Demo.from_safetensors(demo_path)
    if demo is None:
        raise RuntimeError(f"Could not load demo: {demo_path}")

    env = make_env(args)
    try:
        _obs, info = env.reset(seed=args.seed)
        model = env.mojo.physics.model.ptr
        data = env.mojo.physics.data.ptr
        ids = {
            "ee_site": resolve_mujoco_name(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "right_end_effector",
            ),
            "handle_site": resolve_mujoco_name(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "drawer_small_4",
            ),
            "upperarm_geom": resolve_mujoco_name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "cylinder_arm/upperarm_geom",
            ),
            "forearm_geom": resolve_mujoco_name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "cylinder_arm/forearm_geom",
            ),
        }

        samples = [sample_state(model, data, ids, 0, info)]
        robot_spawn_shapes = robot_topdown_geometry(model, data)
        max_steps = len(demo.timesteps) if args.max_steps is None else args.max_steps
        max_steps = min(int(max_steps), len(demo.timesteps))
        for step_i in range(max_steps):
            action = clamp_action(env, demo.timesteps[step_i].executed_action)
            _obs, _reward, terminated, truncated, info = env.step(action)
            should_sample = (
                (step_i + 1) % max(1, args.sample_every) == 0
                or step_i + 1 == max_steps
                or terminated
                or truncated
            )
            if should_sample:
                samples.append(sample_state(model, data, ids, step_i + 1, info))
            if terminated or truncated:
                break

        png_path, svg_path, json_path, metadata = plot(
            samples,
            model,
            data,
            args,
            demo_path,
            robot_spawn_shapes,
        )
        print("saved_png:", png_path)
        print("saved_svg:", svg_path)
        print("saved_json:", json_path)
        print("samples:", metadata["num_samples"])
        print("phase_counts:", metadata["phase_counts"])
        print("first_triggered_step:", metadata["first_triggered_step"])
        print("min_robot_human_distance:", metadata["min_robot_human_distance"])
        print("final_drawer_open_distance:", metadata["final_drawer_open_distance"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
