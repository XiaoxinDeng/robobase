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


class WallClockVideoRecorder:
    def __init__(self, save_dir: Path, render_size=256, fps=20):
        self.save_dir = save_dir
        if save_dir is not None:
            self.save_dir.mkdir(exist_ok=True)
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.timestamps = []

    def init(self, env, enabled=True):
        self.frames = []
        self.timestamps = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = _render_single_env_if_vector(env)
            if frame is not None:
                self.frames.append(frame)
                self.timestamps.append(time.perf_counter())

    def save(self, file_name):
        if self.enabled and len(self.frames) > 0:
            path = self.save_dir / file_name
            frames = np.array(self.frames)
            fps = self.fps
            if len(self.timestamps) > 1:
                duration = self.timestamps[-1] - self.timestamps[0]
                if duration > 0:
                    fps = len(self.frames) / duration
            imageio.mimsave(str(path), frames, fps=fps)


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
    ws.load_snapshot(snapshot_path)
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

        min_h = float(np.min(h_values))
        return min_h, h_values.tolist(), bool(min_h < 0.0)

    except AttributeError:
        return None, None, None


def count_robot_human_contacts(env) -> Optional[int]:
    try:
        from robobase.safetyfilter.h1_state_bridge import get_bigym_task

        task = get_bigym_task(env)
        model = task._mojo.model
        data = task._mojo.data

        count = 0
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

            is_robot_1 = "h1" in n1 or "shoulder" in n1 or "elbow" in n1 or "wrist" in n1
            is_robot_2 = "h1" in n2 or "shoulder" in n2 or "elbow" in n2 or "wrist" in n2

            if (is_human_1 and is_robot_2) or (is_human_2 and is_robot_1):
                count += 1

        return count

    except Exception:  # noqa: BLE001
        return None


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

        "contact_count_total": int(np.sum(valid_contacts)) if valid_contacts else None,
        "contact_episode": bool(np.sum(valid_contacts) > 0) if valid_contacts else None,

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

    def mean_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def sum_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.sum(vals)) if vals else None

    return {
        "condition": episode_summaries[0]["condition"],
        "num_episodes": len(episode_summaries),

        "success_rate": mean_of("success"),
        "mean_return": mean_of("episode_return"),
        "mean_episode_length": mean_of("episode_length"),

        "mean_episode_min_h": mean_of("episode_min_h"),
        "mean_h_violation_rate": mean_of("h_violation_rate"),
        "total_h_violations": sum_of("h_violation_count"),

        "collision_episode_rate": mean_of("contact_episode"),
        "total_contacts": sum_of("contact_count_total"),

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
