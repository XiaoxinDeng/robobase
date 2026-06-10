from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import imageio
import jax.numpy as jnp
import numpy as np
import torch

from robobase import utils
from robobase.envs.bigym import BiGymEnvFactory
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter
from robobase.workspace import Workspace
from hydra import compose, initialize_config_dir

REPO = Path("/home/xd1125/Workspace/safe_bigym_hoi")
ROBOBASE_CFG = REPO / "external/robobase/robobase/cfgs"


def _render_single_env_if_vector(env: gym.vector.VectorEnv):
    if getattr(env, "is_vector_env", False):
        if getattr(env, "parent_pipes", False):
            old_parent_pipes = env.parent_pipes
            env.parent_pipes = old_parent_pipes[:1]
            img = env.call("render")[0]
            env.parent_pipes = old_parent_pipes
        elif getattr(env, "envs", False):
            old_envs = env.envs
            env.envs = old_envs[:1]
            img = env.call("render")[0]
            env.envs = old_envs
        else:
            raise ValueError("Unrecognized vector env.")
    else:
        img = env.render()
    return img


def _mujoco_data_and_forward(env):
    try:
        from robobase.safetyfilter.h1_state_bridge import get_bigym_task

        task = get_bigym_task(env)
    except Exception:  # noqa: BLE001
        return None, None

    mojo = getattr(task, "_mojo", None) or getattr(task, "mojo", None)
    if mojo is None:
        return None, None

    physics = getattr(mojo, "physics", None)
    data = getattr(physics, "data", None) if physics is not None else None
    if data is None:
        data = getattr(mojo, "data", None)
    if data is None:
        return None, None

    forward = getattr(physics, "forward", None) if physics is not None else None
    if forward is None:
        forward = getattr(mojo, "forward", None)
    return data, forward


def _snapshot_mujoco_state(env):
    data, _forward = _mujoco_data_and_forward(env)
    if data is None:
        return None

    state = {"time": float(getattr(data, "time", 0.0))}
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
        value = getattr(data, name, None)
        if value is not None:
            state[name] = np.asarray(value).copy()
    return state


def _restore_mujoco_state(env, state) -> bool:
    if state is None:
        return False

    data, forward = _mujoco_data_and_forward(env)
    if data is None:
        return False

    if "time" in state and hasattr(data, "time"):
        data.time = float(state["time"])
    for name, value in state.items():
        if name == "time":
            continue
        target = getattr(data, name, None)
        if target is not None and np.shape(target) == np.shape(value):
            target[...] = value

    if callable(forward):
        forward()
    return True


class WallClockVideoRecorder:
    def __init__(self, save_dir: Path, render_size=256, fps=20, time_base="sim"):
        self.save_dir = save_dir
        if save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if time_base not in {"sim", "wall"}:
            raise ValueError(f"time_base must be sim or wall, got {time_base!r}")
        self.render_size = render_size
        self.fps = fps
        self.time_base = time_base
        self.frames = []
        self._states = []
        self.timestamps = []
        self._deferred = True
        self._env = None

    def init(self, env, enabled=True):
        self.frames = []
        self._states = []
        self.timestamps = []
        self.enabled = self.save_dir is not None and enabled
        self._deferred = True
        self._env = env
        self.record(env)

    def record(self, env):
        if not self.enabled:
            return

        self._env = env
        timestamp = time.perf_counter() if self.time_base == "wall" else None
        state = _snapshot_mujoco_state(env)
        if state is not None and self._deferred:
            self._states.append(state)
            if self.time_base == "wall":
                self.timestamps.append(timestamp)
            else:
                self.timestamps.append((len(self._states) - 1) / float(self.fps))
            return

        self._deferred = False
        frame = _render_single_env_if_vector(env)
        if frame is None:
            return
        self.frames.append(frame)
        if self.time_base == "wall":
            self.timestamps.append(timestamp)
        else:
            self.timestamps.append((len(self.frames) - 1) / float(self.fps))

    def save(self, file_name):
        if not self.enabled:
            return

        frames = self.frames
        if self._deferred and self._states and self._env is not None:
            current_state = _snapshot_mujoco_state(self._env)
            rendered_frames = []
            try:
                for state in self._states:
                    if _restore_mujoco_state(self._env, state):
                        frame = _render_single_env_if_vector(self._env)
                        if frame is not None:
                            rendered_frames.append(frame)
            finally:
                _restore_mujoco_state(self._env, current_state)
            frames = rendered_frames

        if len(frames) == 0:
            return

        path = self.save_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = self.fps
        if self.time_base == "wall" and len(self.timestamps) > 1:
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration > 0:
                fps = len(frames) / duration
        imageio.mimsave(str(path), np.array(frames), fps=fps)


def infer_env_action_shape(env, fallback=(16, 16)) -> tuple[int, ...]:
    if hasattr(env, "action_space"):
        shape = getattr(env.action_space, "shape", None)
        if shape is not None:
            return tuple(int(x) for x in shape)
    return tuple(int(x) for x in fallback)


def extract_first_action(env_action: np.ndarray) -> np.ndarray:
    env_action = np.asarray(env_action, dtype=np.float32)

    if env_action.ndim == 1:
        return env_action.copy()

    if env_action.ndim == 2:
        return env_action[0].copy()

    raise ValueError(f"Unsupported env_action shape: {env_action.shape}")


def replace_first_action(
    env_action: np.ndarray,
    safe_first_action: np.ndarray,
) -> np.ndarray:
    env_action = np.asarray(env_action, dtype=np.float32).copy()
    safe_first_action = np.asarray(safe_first_action, dtype=np.float32).reshape(-1)

    if env_action.ndim == 1:
        if env_action.shape != safe_first_action.shape:
            raise ValueError(
                f"env_action shape {env_action.shape} does not match "
                f"safe_first_action shape {safe_first_action.shape}"
            )
        return safe_first_action.astype(np.float32)

    if env_action.ndim == 2:
        if env_action.shape[1] != safe_first_action.shape[0]:
            raise ValueError(
                f"env_action second dim {env_action.shape[1]} does not match "
                f"safe_first_action dim {safe_first_action.shape[0]}"
            )
        env_action[0] = safe_first_action
        return env_action.astype(np.float32)

    raise ValueError(f"Unsupported env_action shape: {env_action.shape}")


def get_non_arm_indices(action_dim: int, arm_indices: Sequence[int]) -> np.ndarray:
    arm_set = set(int(i) for i in arm_indices)
    return np.asarray(
        [i for i in range(action_dim) if i not in arm_set],
        dtype=np.int64,
    )


def to_numpy_action(action: Any) -> np.ndarray:
    if isinstance(action, tuple):
        action = action[0]

    if isinstance(action, dict):
        for key in ["action", "actions", "act"]:
            if key in action:
                action = action[key]
                break

    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()

    return np.asarray(action, dtype=np.float32)


def normalise_env_action_shape(
    action: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)

    if action.shape == expected_shape:
        return action

    if action.ndim == len(expected_shape) + 1 and action.shape[0] == 1:
        action = action[0]
        if action.shape == expected_shape:
            return action

    raise ValueError(
        f"Policy action shape {action.shape} does not match expected env action "
        f"shape {expected_shape}. You may need to edit policy_action()."
    )


def make_output_paths(args):
    output_root = Path(args.output_dir) if args.output_dir is not None else Path(args.out).with_suffix("")
    step_jsonl_path = output_root / Path(args.out).name
    episode_summary_path = output_root / (Path(args.out).stem + "_episodes.json")
    final_summary_path = output_root / (Path(args.out).stem + "_summary.json")
    return output_root, step_jsonl_path, episode_summary_path, final_summary_path


def make_cfg(args):
    overrides = [
        f"env={args.env}",
        "method=act",
        "launch=act_pixel_bigym",
        f"demos={args.demos}",
        f"num_eval_episodes={args.episodes}",
        "num_train_envs=0",
        "log_eval_video=false",
        "wandb.use=false",
        "replay.num_workers=0",
        # Do not let Workspace-level safety_filter interfere with this script.
        # This script applies/monitors OSCBF explicitly.
        "safety_filter.enabled=false",
    ]

    overrides.extend(args.override)

    with initialize_config_dir(config_dir=str(ROBOBASE_CFG), version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=overrides,
        )

    return cfg


def make_eval_env(cfg):
    env_factory = BiGymEnvFactory()

    if cfg.demos != 0:
        env_factory.collect_or_fetch_demos(cfg, cfg.demos)

    env = env_factory.make_eval_env(cfg)

    if cfg.demos != 0:
        env_factory.post_collect_or_fetch_demos(cfg)

    return env


def make_workspace_and_load_snapshot(cfg, snapshot_path: Path) -> Workspace:
    run_dir = snapshot_path.parent.parent
    ws = Workspace(cfg, work_dir=run_dir)
    try:
        ws.load_snapshot(snapshot_path)
    except RuntimeError as exc:
        message = str(exc)
        if "size mismatch" in message:
            frame_stack = cfg.get("frame_stack", "<unset>")
            frame_stack_on_channel = cfg.get(
                "frame_stack_on_channel", "<unset>"
            )
            raise RuntimeError(
                "Snapshot architecture does not match the eval config. "
                f"Current eval frame_stack={frame_stack}, "
                f"frame_stack_on_channel={frame_stack_on_channel}. "
                "For frame-stacked ACT checkpoints, pass the same Hydra "
                "architecture overrides used during training, e.g. "
                "--override frame_stack=4. Also make sure commented-out "
                "override lines are not placed inside a backslash-continued "
                "shell command, because bash can drop later overrides."
            ) from exc
        raise
    return ws


def policy_action(ws: Workspace, obs: Any, step: int) -> np.ndarray:
    agent = getattr(ws, "agent", None)
    if agent is None:
        agent = getattr(ws, "_agent", None)

    if agent is None:
        raise AttributeError("Could not find ws.agent or ws._agent.")

    device = getattr(ws, "device", torch.device("cpu"))

    with torch.no_grad(), utils.eval_mode(agent):
        torch_observations = {
            k: torch.from_numpy(v).to(device)
            for k, v in obs.items()
        }

        torch_observations = {
            k: v.unsqueeze(0)
            for k, v in torch_observations.items()
        }

        action = agent.act(
            torch_observations,
            step,
            eval_mode=True,
        )

        if isinstance(action, tuple):
            action, _act_info = action

        action = action.cpu().detach().numpy()

        if action.ndim != 3:
            raise ValueError(
                "Expected action from agent.act to have shape "
                f"(Batch, Timesteps, Action Dim), got {action.shape}"
            )

        action = action[0]

    return np.asarray(action, dtype=np.float32)


def compute_oscbf_h_monitor(
    filt: OSCBFFilter,
    env,
    obs,
    q_full: np.ndarray,
    qd_full: np.ndarray,
):
    if filt.oscbf_config is None:
        return None, None, None

    try:
        q_urdf, _, _, _ = filt._build_urdf_surrogate_state_from_bigym(q_full, qd_full)

        human_obstacles = filt._extract_human_obstacles(env, obs)

        capsule_a_world = human_obstacles["capsule_a"]
        capsule_b_world = human_obstacles["capsule_b"]
        capsule_radii = human_obstacles["capsule_radii"]

        T_world_urdf = filt._get_world_T_urdf_from_bigym_state(q_full)
        T_urdf_world = np.linalg.inv(T_world_urdf)

        capsule_a_urdf = filt._transform_points(T_urdf_world, capsule_a_world)
        capsule_b_urdf = filt._transform_points(T_urdf_world, capsule_b_world)

        if hasattr(filt, "compute_live_h_values"):
            h_values = filt.compute_live_h_values(
                q_urdf,
                capsule_a_urdf,
                capsule_b_urdf,
                capsule_radii,
            )
        else:
            filt._validate_capsules(capsule_a_urdf, capsule_b_urdf, capsule_radii)
            filt.oscbf_config.set_human_capsules(
                capsule_a_urdf,
                capsule_b_urdf,
                capsule_radii,
            )
            h_values = np.asarray(
                filt.oscbf_config.h_1(jnp.asarray(q_urdf, dtype=jnp.float32)),
                dtype=np.float32,
            ).reshape(-1)

        if h_values is None:
            return None, None, None

        min_h = float(np.min(h_values))
        return min_h, h_values.tolist(), bool(min_h < 0.0)

    except AttributeError:
        return None, None, None


def _segment_segment_distance_sq(p1, q1, p2, q2) -> float:
    p1 = np.asarray(p1, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-8

    if a <= eps and e <= eps:
        return float(np.dot(p1 - p2, p1 - p2))
    if a <= eps:
        s = 0.0
        t = np.clip(f / max(e, eps), 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    c1 = p1 + s * d1
    c2 = p2 + t * d2
    return float(np.dot(c1 - c2, c1 - c2))


def _robot_arm_capsules_urdf(filt: OSCBFFilter, q_urdf: np.ndarray):
    robot = filt.robot_model
    if robot is None:
        return [], [], []

    transforms = np.asarray(
        robot.joint_to_world_transforms(jnp.asarray(q_urdf, dtype=jnp.float32)),
        dtype=np.float32,
    )
    pos_by_name = {name: transforms[i, :3, 3] for i, name in enumerate(robot.joint_names)}
    segments = [
        ("left_shoulder_upper", "left_shoulder_pitch_joint", "left_shoulder_roll_joint"),
        ("left_upperarm", "left_shoulder_roll_joint", "left_shoulder_yaw_joint"),
        ("left_forearm", "left_shoulder_yaw_joint", "left_elbow_joint"),
        ("right_shoulder_upper", "right_shoulder_pitch_joint", "right_shoulder_roll_joint"),
        ("right_upperarm", "right_shoulder_roll_joint", "right_shoulder_yaw_joint"),
        ("right_forearm", "right_shoulder_yaw_joint", "right_elbow_joint"),
    ]

    capsule_a = []
    capsule_b = []
    names = []
    for label, a_name, b_name in segments:
        if a_name in pos_by_name and b_name in pos_by_name:
            capsule_a.append(pos_by_name[a_name])
            capsule_b.append(pos_by_name[b_name])
            names.append(label)
    return capsule_a, capsule_b, names


def compute_oscbf_full_arm_h_monitor(
    filt: OSCBFFilter,
    env,
    obs,
    q_full: np.ndarray,
    qd_full: np.ndarray,
    robot_radius: float = 0.06,
):
    if filt.oscbf_config is None or filt.robot_model is None:
        return None, None, None, None

    try:
        q_urdf, _, _, _ = filt._build_urdf_surrogate_state_from_bigym(q_full, qd_full)
        human_obstacles = filt._extract_human_obstacles(env, obs)

        t_world_urdf = filt._get_world_T_urdf_from_bigym_state(q_full)
        t_urdf_world = np.linalg.inv(t_world_urdf)

        human_a = filt._transform_points(t_urdf_world, human_obstacles["capsule_a"])
        human_b = filt._transform_points(t_urdf_world, human_obstacles["capsule_b"])
        human_r = np.asarray(human_obstacles["capsule_radii"], dtype=np.float32)

        robot_a, robot_b, robot_names = _robot_arm_capsules_urdf(filt, q_urdf)
        h_values = []
        labels = []
        for r_idx, (ra, rb) in enumerate(zip(robot_a, robot_b)):
            for h_idx, (ha, hb) in enumerate(zip(human_a, human_b)):
                combined_radius = float(robot_radius + human_r[h_idx])
                dist_sq = _segment_segment_distance_sq(ra, rb, ha, hb)
                h_values.append(dist_sq - combined_radius * combined_radius)
                labels.append(f"{robot_names[r_idx]}:human_capsule_{h_idx}")

        if not h_values:
            return None, None, None, None
        h_arr = np.asarray(h_values, dtype=np.float32)
        min_idx = int(np.argmin(h_arr))
        min_h = float(h_arr[min_idx])
        return min_h, h_arr.tolist(), bool(min_h < 0.0), labels[min_idx]

    except AttributeError:
        return None, None, None, None


def robot_human_contact_pairs(env) -> Optional[list[str]]:
    try:
        from robobase.safetyfilter.h1_state_bridge import get_bigym_task

        task = get_bigym_task(env)
        model = task._mojo.model
        data = task._mojo.data

        pairs = []
        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)

            name1 = model.geom(g1).name or ""
            name2 = model.geom(g2).name or ""

            n1 = name1.lower()
            n2 = name2.lower()

            is_human_1 = "human" in n1 or "upperarm" in n1 or "forearm" in n1
            is_human_2 = "human" in n2 or "upperarm" in n2 or "forearm" in n2

            robot_tokens = ("h1", "shoulder", "elbow", "wrist", "hand", "gripper", "finger", "robotiq")
            is_robot_1 = any(token in n1 for token in robot_tokens)
            is_robot_2 = any(token in n2 for token in robot_tokens)

            if (is_human_1 and is_robot_2) or (is_human_2 and is_robot_1):
                pairs.append(f"{name1}<->{name2}")

        return pairs

    except Exception:  # noqa: BLE001
        return None


def count_robot_human_contacts(env) -> Optional[int]:
    pairs = robot_human_contact_pairs(env)
    return None if pairs is None else len(pairs)


def extract_success(info: Any, reward: float, terminated: bool) -> bool:
    if isinstance(info, dict):
        for key in [
            "success",
            "is_success",
            "task_success",
            "is_successful",
            "demo_success",
        ]:
            if key in info:
                value = info[key]
                if isinstance(value, (list, tuple, np.ndarray)):
                    return bool(np.asarray(value).any())
                return bool(value)

    return False


def assert_action_properties(
    nominal_action: np.ndarray,
    safe_action: np.ndarray,
    arm_indices: np.ndarray,
):
    assert safe_action.shape == nominal_action.shape
    assert np.isfinite(safe_action).all()

    non_arm_idx = get_non_arm_indices(nominal_action.shape[0], arm_indices)

    if not np.allclose(
        nominal_action[non_arm_idx],
        safe_action[non_arm_idx],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError(
            "Non-arm dimensions changed.\n"
            f"non_arm_idx={non_arm_idx}\n"
            f"nominal={nominal_action[non_arm_idx]}\n"
            f"safe={safe_action[non_arm_idx]}"
        )


def summarise_episode(metrics: list[StepMetrics]) -> dict:
    if len(metrics) == 0:
        return {}

    rewards = np.asarray([m.reward for m in metrics], dtype=np.float32)
    arm_delta = np.asarray([m.arm_delta for m in metrics], dtype=np.float32)
    non_arm_delta = np.asarray([m.non_arm_delta for m in metrics], dtype=np.float32)
    interventions = np.asarray([m.intervention_active for m in metrics], dtype=np.float32)
    filter_times = np.asarray([m.filter_time_ms for m in metrics], dtype=np.float32)

    valid_h = [m.min_h for m in metrics if m.min_h is not None]
    valid_violations = [m.h_violation for m in metrics if m.h_violation is not None]
    valid_contacts = [m.contact_count for m in metrics if m.contact_count is not None]
    valid_contact_steps = [count > 0 for count in valid_contacts]

    return {
        "condition": metrics[0].condition,
        "episode": metrics[0].episode,
        "episode_return": float(np.sum(rewards)),
        "episode_length": int(len(metrics)),
        "success": bool(any(m.success for m in metrics)),

        "episode_min_h": float(np.min(valid_h)) if valid_h else None,
        "mean_min_h": float(np.mean(valid_h)) if valid_h else None,
        "h_violation_count": int(np.sum(valid_violations)) if valid_violations else None,
        "h_violation_rate": float(np.mean(valid_violations)) if valid_violations else None,

        "contact_count_total": int(np.sum(valid_contact_steps)) if valid_contact_steps else None,
        "contact_step_count": int(np.sum(valid_contact_steps)) if valid_contact_steps else None,
        "contact_step_rate": float(np.mean(valid_contact_steps)) if valid_contact_steps else None,
        "contact_episode": bool(np.sum(valid_contact_steps) > 0) if valid_contact_steps else None,

        "mean_arm_delta": float(np.mean(arm_delta)),
        "max_arm_delta": float(np.max(arm_delta)),
        "mean_non_arm_delta": float(np.mean(non_arm_delta)),
        "max_non_arm_delta": float(np.max(non_arm_delta)),
        "intervention_frequency": float(np.mean(interventions)),

        "mean_filter_time_ms": float(np.mean(filter_times)),
        "max_filter_time_ms": float(np.max(filter_times)),

        "terminated": bool(metrics[-1].terminated),
        "truncated": bool(metrics[-1].truncated),
    }


def summarise_all_episodes(episode_summaries: list[dict]) -> dict:
    if len(episode_summaries) == 0:
        return {}

    def mean_of(key, summaries=None):
        summaries = episode_summaries if summaries is None else summaries
        vals = [s[key] for s in summaries if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def sum_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.sum(vals)) if vals else None

    successful_summaries = [s for s in episode_summaries if bool(s.get("success"))]
    failed_summaries = [s for s in episode_summaries if not bool(s.get("success"))]
    mean_success_episode_length = mean_of("episode_length", successful_summaries)

    return {
        "condition": episode_summaries[0]["condition"],
        "num_episodes": len(episode_summaries),

        "success_rate": mean_of("success"),
        "mean_return": mean_of("episode_return"),
        "mean_episode_length": mean_success_episode_length,
        "mean_steps": mean_success_episode_length,
        "mean_success_episode_length": mean_success_episode_length,
        "mean_all_episode_length": mean_of("episode_length"),
        "mean_failed_episode_length": mean_of("episode_length", failed_summaries),

        "mean_episode_min_h": mean_of("episode_min_h"),
        "mean_h_violation_rate": mean_of("h_violation_rate"),
        "total_h_violations": sum_of("h_violation_count"),

        "collision_episode_rate": mean_of("contact_episode"),
        "total_contacts": sum_of("contact_count_total"),
        "total_contact_steps": sum_of("contact_step_count"),
        "mean_contact_step_rate": mean_of("contact_step_rate"),

        "mean_arm_delta": mean_of("mean_arm_delta"),
        "max_arm_delta_over_episodes": max(
            [s["max_arm_delta"] for s in episode_summaries],
            default=None,
        ),
        "mean_intervention_frequency": mean_of("intervention_frequency"),

        "mean_filter_time_ms": mean_of("mean_filter_time_ms"),
        "max_filter_time_ms_over_episodes": max(
            [s["max_filter_time_ms"] for s in episode_summaries],
            default=None,
        ),
    }
