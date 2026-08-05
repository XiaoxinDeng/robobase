from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from robobase.safetyfilter.eval_utils.eval_utils import make_cfg, make_eval_env
from robobase.safetyfilter.h1_state_bridge import extract_h1_state, get_bigym_task
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter


REPO = Path("/home/xd1125/Workspace/safe_bigym_hoi")
H1_URDF = REPO / "external/oscbf/oscbf/assets/h1/h1.urdf"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="bigym/human_arm_drawer_top_open")
    parser.add_argument("--demos", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="eval_safety/oscbf_geometry_debug")
    parser.add_argument("--human-margin", type=float, default=0.08)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _axis_equal_3d(ax, points):
    points = np.asarray(points, dtype=np.float32)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_capsule_axis(ax, a, b, radius, label, color, linewidth=3.0):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        [a[2], b[2]],
        color=color,
        linewidth=linewidth,
        label=label,
    )
    ax.scatter([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, s=20)
    mid = 0.5 * (a + b)
    ax.text(mid[0], mid[1], mid[2], f"r={radius:.3f}", color=color)


def _human_geom_capsules(task):
    model = task._mojo.model
    data = task._mojo.data
    specs = [
        ("upperarm_geom", 0.035, 0.34 / 2.0),
        ("forearm_geom", 0.032, 0.30 / 2.0),
    ]
    capsules = []
    for human in task.humanarms:
        for geom_name, nominal_radius, half_length in specs:
            full_name = human._pref(geom_name)
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, full_name)
            if gid < 0:
                continue
            center = np.asarray(data.geom_xpos[gid], dtype=np.float32)
            rot = np.asarray(data.geom_xmat[gid], dtype=np.float32).reshape(3, 3)
            local_z_world = rot[:, 2].astype(np.float32)
            size = np.asarray(model.geom_size[gid], dtype=np.float32).copy()
            radius = float(size[0]) if size.size else float(nominal_radius)
            capsules.append(
                {
                    "name": full_name,
                    "a": center - half_length * local_z_world,
                    "b": center + half_length * local_z_world,
                    "radius": radius,
                    "model_geom_size": size.tolist(),
                }
            )
    return capsules


def main():
    args = parse_args()
    cfg_args = SimpleNamespace(
        env=args.env,
        demos=args.demos,
        episodes=1,
        override=args.override,
    )
    cfg = make_cfg(cfg_args)
    env = make_eval_env(cfg)
    try:
        obs, _ = env.reset(seed=args.seed)
        task = get_bigym_task(env)

        oscbf = OSCBFFilter(
            urdf_path=str(H1_URDF),
            debug=False,
            human_margin=args.human_margin,
            control_type="absolute",
        )
        h1 = extract_h1_state(env)
        q_urdf, _, _, _ = oscbf._build_urdf_surrogate_state_from_bigym(
            h1.q_full,
            h1.qd_full,
        )
        t_world_urdf = oscbf._get_world_T_urdf_from_bigym_state(h1.q_full)
        ee_urdf = np.asarray(
            oscbf.robot_model.ee_position(jnp.asarray(q_urdf, dtype=jnp.float32)),
            dtype=np.float32,
        )
        ee_world = (t_world_urdf @ np.r_[ee_urdf, 1.0])[:3].astype(np.float32)

        human = oscbf._extract_human_obstacles(env, obs)
        h_capsules = [
            {"a": a, "b": b, "radius": float(r)}
            for a, b, r in zip(
                human["capsule_a"],
                human["capsule_b"],
                human["capsule_radii"],
            )
        ]
        mj_capsules = _human_geom_capsules(task)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        all_points = [ee_world]
        for i, cap in enumerate(mj_capsules):
            _plot_capsule_axis(
                ax,
                cap["a"],
                cap["b"],
                cap["radius"],
                "MuJoCo human geom" if i == 0 else None,
                "tab:blue",
                linewidth=2.0,
            )
            all_points.extend([cap["a"], cap["b"]])
        for i, cap in enumerate(h_capsules):
            _plot_capsule_axis(
                ax,
                cap["a"],
                cap["b"],
                cap["radius"],
                "OSCBF inflated human capsule" if i == 0 else None,
                "tab:orange",
                linewidth=5.0,
            )
            all_points.extend([cap["a"], cap["b"]])

        ax.scatter(
            [ee_world[0]],
            [ee_world[1]],
            [ee_world[2]],
            color="tab:red",
            s=80,
            label="EE point used by h",
        )
        ax.set_title("OSCBF h geometry vs MuJoCo human geoms")
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
        ax.set_zlabel("world z")
        ax.legend(loc="upper left")
        _axis_equal_3d(ax, np.asarray(all_points))

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        png_path = out / "oscbf_geometry.png"
        json_path = out / "oscbf_geometry.json"
        fig.tight_layout()
        fig.savefig(png_path, dpi=180)

        payload = {
            "env": args.env,
            "seed": args.seed,
            "human_margin": args.human_margin,
            "ee_world": ee_world.tolist(),
            "ee_urdf": ee_urdf.tolist(),
            "mujoco_human_capsules": [
                {
                    **cap,
                    "a": np.asarray(cap["a"]).tolist(),
                    "b": np.asarray(cap["b"]).tolist(),
                }
                for cap in mj_capsules
            ],
            "oscbf_human_capsules": [
                {
                    "a": np.asarray(cap["a"]).tolist(),
                    "b": np.asarray(cap["b"]).tolist(),
                    "radius": cap["radius"],
                }
                for cap in h_capsules
            ],
            "note": (
                "Current h_1 uses only the URDF surrogate end-effector point "
                "against these inflated human capsules, not full robot MuJoCo "
                "collision geometry."
            ),
        }
        json_path.write_text(json.dumps(payload, indent=2))

        print("saved_png:", png_path)
        print("saved_json:", json_path)
        print("ee_world:", ee_world.tolist())
        for i, cap in enumerate(mj_capsules):
            print(f"mujoco_capsule[{i}] {cap['name']} radius={cap['radius']}")
        for i, cap in enumerate(h_capsules):
            print(f"oscbf_capsule[{i}] radius={cap['radius']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
