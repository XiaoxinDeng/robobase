#!/usr/bin/env python3
"""ACT-only ablation that teleports robot arm q to a recorded recovery state.

Question answered:
    If recovery/MPC had magically placed the arm at the recorded recovered q
    state (or its last 4-frame q window), can ACT continue the task in a clean
    Bigym environment with no human blocker and no safety filter?

This isolates ACT resume robustness from recovery execution dynamics.  It does
not run SafeChunk, OSCBF, MPC, or a human arm blocker.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_THIS_FILE = Path(__file__).resolve()
_ROBOBASE_ROOT = _THIS_FILE.parents[1]
if str(_ROBOBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOBASE_ROOT))

from tools.act_tube_error_ablation import (  # noqa: E402
    REPO,
    ROBOBASE_ROOT,
    _adapt_policy_obs_to_space,
    _current_obs,
    _diagnostic_task_state,
    _finite_float,
    _infer_clean_env,
    _jsonable_trace_value,
    _load_eval_args,
    _load_jsonl,
    _make_runtime_namespace,
    _mujoco_forward,
    _perturb_indices,
    _set_h1_q,
    _task_progress_from_state,
)
from robobase.safetyfilter.eval_utils.eval_utils import (  # noqa: E402
    extract_success,
    infer_env_action_shape,
    make_cfg,
    make_workspace_and_load_snapshot,
    normalise_env_action_shape,
    policy_action,
)
from robobase.safetyfilter.eval_utils.eval_video import (  # noqa: E402
    _make_eval_env_with_normalization,
    _resolve_normalization_cfg,
)
from robobase.safetyfilter.h1_state_bridge import extract_h1_state  # noqa: E402

DEFAULT_SOURCE_PATTERN = "eval_safety/temporal_human_blocker_chunk_deform_*"
RESUME_EVENT_FIELDS = (
    "committed_released_for_act_resume",
    "resume_from_committed_rejoin",
    "recovery_action_history_reset",
    "post_recovery_act_bridge_started",
    "request_action_history_reset_after_recovery",
    "mpc_handoff_accepted",
)
SOURCE_Q_FIELDS = (
    "bigym_actual_post_action_q",
    "actual_q_at_replay",
    "actual_pre_action_q",
    "bigym_actual_pre_action_q",
    "safety_filter_actual_pre_action_q",
    "committed_abort_robot_q",
    "planned_post_action_q",
    "planned_pre_action_q",
    "q_full",
    "q",
    "qpos",
)


@dataclass
class ForceResumeResult:
    episode: int
    trial: int
    variant: str
    seed: int
    force_step: int
    source_episode: int
    source_step: int
    force_dims: str
    history_mode: str
    window_len: int
    forced_q_l2_from_clean: float
    forced_arm_l2_from_clean: float
    forced_base_l2_from_clean: float
    success: bool
    terminated: bool
    truncated: bool
    episode_length: int
    final_reward: float
    final_task_progress: Optional[float]
    max_task_progress: Optional[float]
    final_drawer_open_distance: Optional[float]
    final_drawer_open_fraction: Optional[float]
    progress_after_force: Optional[float]
    max_progress_after_force: Optional[float]
    q_before_force: Optional[list[float]]
    q_after_force: Optional[list[float]]
    source_q_terminal: Optional[list[float]]
    output_dir: str


def _latest_run(pattern: str) -> Optional[Path]:
    candidates = sorted(
        ROBOBASE_ROOT.glob(pattern),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "executed_policy_trajectory.jsonl").is_file():
            return candidate
    return None


def _resolve_run(source_run: Optional[str]) -> Path:
    if source_run:
        path = Path(source_run).expanduser()
        if not path.is_absolute():
            path = ROBOBASE_ROOT / path
        if not path.is_dir():
            raise FileNotFoundError(f"source run is not a directory: {path}")
        return path
    latest = _latest_run(DEFAULT_SOURCE_PATTERN)
    if latest is None:
        raise FileNotFoundError(
            f"could not find latest source run matching {DEFAULT_SOURCE_PATTERN}"
        )
    return latest


def _event_value_active(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(float(value) > 0.0 and math.isfinite(float(value)))
    return False


def _load_source_steps(metrics_path: Path, explicit_step: Optional[int]) -> dict[int, int]:
    if explicit_step is not None:
        return {}
    rows = _load_jsonl(metrics_path)
    steps: dict[int, int] = {}
    for row in rows:
        ep = row.get("episode")
        step = row.get("step")
        if not isinstance(ep, int) or not isinstance(step, int):
            continue
        if int(ep) in steps:
            continue
        if any(_event_value_active(row.get(field)) for field in RESUME_EVENT_FIELDS):
            steps[int(ep)] = int(step)
    return steps


def _load_q_trajectory(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows = _load_jsonl(path)
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        ep = row.get("episode")
        step = row.get("step")
        q = None
        for field in SOURCE_Q_FIELDS:
            candidate = row.get(field)
            if candidate is not None:
                q = candidate
                break
        if not isinstance(ep, int) or not isinstance(step, int):
            continue
        try:
            q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            continue
        if q_arr.size == 0 or not np.all(np.isfinite(q_arr)):
            continue
        by_episode.setdefault(int(ep), []).append(
            {"episode": int(ep), "step": int(step), "q_full": q_arr.astype(float)}
        )
    for ep in by_episode:
        by_episode[ep].sort(key=lambda item: int(item["step"]))
    return by_episode


def _select_q_window(
    trajectory: dict[int, list[dict[str, Any]]],
    source_episode: int,
    source_step: int,
    window_len: int,
) -> tuple[list[np.ndarray], int]:
    rows = trajectory.get(int(source_episode)) or []
    if not rows:
        raise ValueError(f"no q trajectory rows for source episode {source_episode}")
    before = [row for row in rows if int(row["step"]) <= int(source_step)]
    if not before:
        before = [rows[0]]
    selected = before[-max(1, int(window_len)) :]
    if len(selected) < max(1, int(window_len)):
        selected = [selected[0]] * (max(1, int(window_len)) - len(selected)) + selected
    return [np.asarray(row["q_full"], dtype=np.float64).copy() for row in selected], int(selected[-1]["step"])


def _seed_policy_stack_from_frames(policy_obs: Any, frames: list[Any]) -> Any:
    if not isinstance(policy_obs, dict) or not frames:
        return policy_obs
    merged = copy.deepcopy(policy_obs)
    for key, target_value in list(merged.items()):
        try:
            target = np.asarray(target_value)
        except Exception:  # noqa: BLE001
            continue
        frame_values = []
        for frame in frames:
            if not isinstance(frame, dict) or key not in frame:
                continue
            try:
                frame_values.append(np.asarray(frame[key]))
            except Exception:  # noqa: BLE001
                pass
        if not frame_values:
            continue
        last = frame_values[-1]
        if target.shape == last.shape:
            merged[key] = last.astype(target.dtype, copy=False)
            continue
        if target.ndim >= 2 and last.shape == target.shape[1:]:
            updated = target.copy()
            usable = frame_values[-target.shape[0] :]
            if len(usable) < target.shape[0]:
                usable = [usable[0]] * (target.shape[0] - len(usable)) + usable
            for i, value in enumerate(usable[-target.shape[0] :]):
                updated[i] = value.astype(target.dtype, copy=False)
            merged[key] = updated
    return merged


def _reset_wrapper_histories(env: Any) -> None:
    for method_name in ("reset_obs_history", "reset_observation_history", "reset_frame_stack"):
        owner = env
        seen = set()
        while owner is not None and id(owner) not in seen:
            seen.add(id(owner))
            method = getattr(owner, method_name, None)
            if callable(method):
                try:
                    method()
                except TypeError:
                    try:
                        method(None)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            owner = getattr(owner, "env", None)


def _apply_forced_q_window(
    env: Any,
    policy_obs: Any,
    source_q_window: list[np.ndarray],
    *,
    force_dims: str,
    history_mode: str,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    q_before = np.asarray(extract_h1_state(env).q_full, dtype=np.float64).reshape(-1)
    indices = _perturb_indices(force_dims)
    frames: list[Any] = []
    q_target = q_before.copy()
    for source_q in source_q_window:
        source_q = np.asarray(source_q, dtype=np.float64).reshape(-1)
        q_target = np.asarray(extract_h1_state(env).q_full, dtype=np.float64).reshape(-1)
        valid = indices[(indices >= 0) & (indices < min(q_target.size, source_q.size))]
        q_target[valid] = source_q[valid]
        _set_h1_q(env, q_target, zero_velocity=True)
        if history_mode == "source_window":
            frames.append(_current_obs(env, policy_obs))
    q_after = np.asarray(extract_h1_state(env).q_full, dtype=np.float64).reshape(-1)

    if history_mode == "source_window":
        policy_obs = _seed_policy_stack_from_frames(policy_obs, frames)
    elif history_mode == "seed_current":
        current = _current_obs(env, policy_obs)
        policy_obs = _seed_policy_stack_from_frames(policy_obs, [current])
    elif history_mode == "last_current":
        current = _current_obs(env, policy_obs)
        from tools.act_tube_error_ablation import _merge_current_obs_into_policy_stack

        policy_obs = _merge_current_obs_into_policy_stack(policy_obs, current)
    elif history_mode == "reset_wrappers":
        _reset_wrapper_histories(env)
        current = _current_obs(env, policy_obs)
        policy_obs = _seed_policy_stack_from_frames(policy_obs, [current])
    elif history_mode == "keep":
        pass
    else:
        raise ValueError(
            "--history-mode must be one of source_window, seed_current, last_current, reset_wrappers, keep"
        )
    return policy_obs, q_before, q_after, q_after - q_before


def _policy_obs_space(ws: Any, env: Any) -> Any:
    for owner in (getattr(ws, "eval_env", None), getattr(ws, "train_env", None), env):
        space = getattr(owner, "observation_space", None)
        if space is not None:
            return space
    return getattr(env, "observation_space", None)


def _run_trial(
    *,
    env: Any,
    ws: Any,
    policy_observation_space: Any,
    env_action_shape: tuple[int, ...],
    episode: int,
    trial: int,
    seed: int,
    steps: int,
    force_step: int,
    source_episode: int,
    source_step: int,
    source_q_window: list[np.ndarray],
    force_dims: str,
    history_mode: str,
    output_dir: Path,
    trajectory_handle: Optional[Any],
) -> ForceResumeResult:
    obs, _info = env.reset(seed=int(seed))
    policy_obs = obs
    policy_step = 0
    final_reward = 0.0
    success = False
    terminated = False
    truncated = False
    final_task_state: dict[str, Any] = {}
    progress_values: list[float] = []
    progress_after_values: list[float] = []
    progress_after_force = None
    q_before_force = None
    q_after_force = None
    q_delta = None
    forced = False
    step_count = 0

    for step in range(int(steps)):
        if step == int(force_step):
            policy_obs, q_before, q_after, q_delta = _apply_forced_q_window(
                env,
                policy_obs,
                source_q_window,
                force_dims=force_dims,
                history_mode=history_mode,
            )
            q_before_force = q_before.astype(float).tolist()
            q_after_force = q_after.astype(float).tolist()
            forced = True
            task_state = _diagnostic_task_state(env)
            progress_after_force = _task_progress_from_state(task_state)

        task_state_before = _diagnostic_task_state(env)
        progress_before = _task_progress_from_state(task_state_before)
        if progress_before is not None:
            progress_values.append(float(progress_before))
            if forced:
                progress_after_values.append(float(progress_before))

        policy_obs_for_action = _adapt_policy_obs_to_space(policy_obs, policy_observation_space)
        env_action = policy_action(ws, policy_obs_for_action, step=policy_step)
        env_action = normalise_env_action_shape(env_action, env_action_shape)
        obs, reward, terminated, truncated, info = env.step(env_action)
        final_reward = float(reward)
        policy_obs = obs
        policy_step += 1
        step_count = step + 1
        final_task_state = _diagnostic_task_state(env)
        final_progress = _task_progress_from_state(final_task_state)
        if final_progress is not None:
            progress_values.append(float(final_progress))
            if forced:
                progress_after_values.append(float(final_progress))
        success = bool(extract_success(info, float(reward), bool(terminated)))

        if trajectory_handle is not None:
            try:
                q_now = extract_h1_state(env).q_full.astype(float).tolist()
            except Exception:  # noqa: BLE001
                q_now = None
            trajectory_handle.write(
                json.dumps(
                    _jsonable_trace_value(
                        {
                            "episode": int(episode),
                            "trial": int(trial),
                            "variant": "force_arm_resume",
                            "step": int(step),
                            "force_step": int(force_step),
                            "source_episode": int(source_episode),
                            "source_step": int(source_step),
                            "forced": bool(forced),
                            "success": bool(success),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "reward": float(reward),
                            "task_state": final_task_state,
                            "q_full": q_now,
                            "action_first": np.asarray(
                                env_action[0] if env_action.ndim == 2 else env_action,
                                dtype=float,
                            ).tolist(),
                        }
                    )
                )
                + "\n"
            )

        if terminated or truncated or success:
            break

    final_progress = _task_progress_from_state(final_task_state)
    max_progress = max(progress_values) if progress_values else None
    max_progress_after = max(progress_after_values) if progress_after_values else None
    if q_delta is None:
        q_delta = np.zeros_like(np.asarray(source_q_window[-1], dtype=np.float64))
    return ForceResumeResult(
        episode=int(episode),
        trial=int(trial),
        variant="force_arm_resume",
        seed=int(seed),
        force_step=int(force_step),
        source_episode=int(source_episode),
        source_step=int(source_step),
        force_dims=str(force_dims),
        history_mode=str(history_mode),
        window_len=int(len(source_q_window)),
        forced_q_l2_from_clean=float(np.linalg.norm(q_delta)),
        forced_arm_l2_from_clean=float(np.linalg.norm(q_delta[min(4, q_delta.size) :])),
        forced_base_l2_from_clean=float(np.linalg.norm(q_delta[: min(4, q_delta.size)])),
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        episode_length=int(step_count),
        final_reward=float(final_reward),
        final_task_progress=None if final_progress is None else float(final_progress),
        max_task_progress=None if max_progress is None else float(max_progress),
        final_drawer_open_distance=_finite_float(final_task_state.get("drawer_open_distance")),
        final_drawer_open_fraction=_finite_float(final_task_state.get("drawer_open_fraction")),
        progress_after_force=None if progress_after_force is None else float(progress_after_force),
        max_progress_after_force=None if max_progress_after is None else float(max_progress_after),
        q_before_force=q_before_force,
        q_after_force=q_after_force,
        source_q_terminal=np.asarray(source_q_window[-1], dtype=float).tolist(),
        output_dir=str(output_dir),
    )


def _summarize(results: list[ForceResumeResult], args: argparse.Namespace, source_run: Path) -> dict[str, Any]:
    def mean_field(field: str) -> Optional[float]:
        vals = [getattr(r, field) for r in results]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        return None if not vals else float(np.mean(vals))

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_config": args.eval_config,
        "source_run": str(source_run),
        "source_trajectory": str(args.source_trajectory) if args.source_trajectory else None,
        "episodes": int(args.episodes),
        "steps": int(args.steps),
        "force_dims": args.force_dims,
        "history_mode": args.history_mode,
        "window_len": int(args.window_len),
        "success_rate": float(np.mean([1.0 if r.success else 0.0 for r in results])) if results else None,
        "mean_episode_length": mean_field("episode_length"),
        "mean_forced_q_l2_from_clean": mean_field("forced_q_l2_from_clean"),
        "mean_forced_arm_l2_from_clean": mean_field("forced_arm_l2_from_clean"),
        "mean_final_task_progress": mean_field("final_task_progress"),
        "mean_max_task_progress": mean_field("max_task_progress"),
        "mean_final_drawer_open_distance": mean_field("final_drawer_open_distance"),
        "results": [asdict(r) for r in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-config",
        default="robobase/cfgs/eval_scenarios/experiments/temporal_human_blocker_chunk_deform.yaml",
        help="Eval config used only to find ACT checkpoint/normalization; env is made clean/no-human.",
    )
    parser.add_argument("--source-run", default=None, help="SafeChunk eval run containing executed_policy_trajectory.jsonl")
    parser.add_argument("--source-trajectory", default=None, help="Optional explicit executed_policy_trajectory.jsonl")
    parser.add_argument("--source-metrics", default=None, help="Optional explicit metrics.jsonl for resume event step")
    parser.add_argument("--clean-env", default="auto", help="Clean Bigym env, or auto from eval snapshot")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force-step", type=int, default=None, help="Clean-env step to teleport arm q; defaults to source resume step")
    parser.add_argument("--source-step", type=int, default=None, help="Recorded source q step; defaults to first resume/reset/handoff event")
    parser.add_argument("--source-episode", type=int, default=None, help="Use one source episode for all trials; default matches clean episode index")
    parser.add_argument(
        "--force-dims",
        default="arm",
        choices=("controlled", "base", "arm", "right_arm", "left_arm"),
        help="Which H1 q dims to overwrite from source q; default arm keeps clean base pose.",
    )
    parser.add_argument(
        "--history-mode",
        default="source_window",
        choices=("source_window", "seed_current", "last_current", "reset_wrappers", "keep"),
        help="How to make ACT observation stack consistent after teleport.",
    )
    parser.add_argument("--window-len", type=int, default=4, help="Number of source q frames to render into ACT stack")
    parser.add_argument("--record-trajectories", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run = _resolve_run(args.source_run)
    trajectory_path = Path(args.source_trajectory).expanduser() if args.source_trajectory else source_run / "executed_policy_trajectory.jsonl"
    metrics_path = Path(args.source_metrics).expanduser() if args.source_metrics else source_run / "metrics.jsonl"
    if not trajectory_path.is_absolute():
        trajectory_path = ROBOBASE_ROOT / trajectory_path
    if not metrics_path.is_absolute():
        metrics_path = ROBOBASE_ROOT / metrics_path
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"missing source trajectory: {trajectory_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing source metrics: {metrics_path}")

    base_args, _container, _resolved = _load_eval_args(args.eval_config)
    snapshot_path = Path(base_args.snapshot).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = ROBOBASE_ROOT / snapshot_path
    clean_env = _infer_clean_env(str(getattr(base_args, "env", "")), snapshot_path, args.clean_env)
    runtime_args = _make_runtime_namespace(
        base_args,
        env=clean_env,
        safety_filter=None,
        record_video=False,
        save_frame_images=False,
        plot_chunk_trajectories_3d=False,
        log_chunk_trajectories=False,
        log_mpc_replay_diagnostics=False,
        log_nominal_rollout_diagnostics=False,
    )
    cfg = make_cfg(runtime_args)
    ws = make_workspace_and_load_snapshot(cfg, snapshot_path)
    normalization_result = _resolve_normalization_cfg(runtime_args, cfg, snapshot_path)
    normalization_cfg = normalization_result
    if isinstance(normalization_result, tuple):
        # eval_video._resolve_normalization_cfg returns (source_label, cfg).
        # For clean ACT ablations cfg may legitimately be None.
        normalization_cfg = normalization_result[1] if len(normalization_result) > 1 else None
    env = _make_eval_env_with_normalization(cfg, normalization_cfg)
    policy_observation_space = _policy_obs_space(ws, env)
    env_action_shape = infer_env_action_shape(env)

    source_steps = _load_source_steps(metrics_path, args.source_step)
    trajectory = _load_q_trajectory(trajectory_path)
    if not trajectory:
        replay_path = source_run / "mpc_replay_diagnostics.jsonl"
        if replay_path.is_file():
            trajectory = _load_q_trajectory(replay_path)
            trajectory_path = replay_path
    if not trajectory:
        trajectory = _load_q_trajectory(metrics_path)
        trajectory_path = metrics_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else ROBOBASE_ROOT / "eval_safety" / f"act_force_arm_resume_ablation_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = ROBOBASE_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "force_arm_resume_results.jsonl"
    summary_path = output_dir / "force_arm_resume_summary.json"
    traj_path = output_dir / "force_arm_resume_trajectory.jsonl"

    results: list[ForceResumeResult] = []
    base_seed = int(args.seed if args.seed is not None else getattr(base_args, "seed", 1))
    trajectory_handle = traj_path.open("w", encoding="utf-8") if args.record_trajectories else None
    try:
        with results_path.open("w", encoding="utf-8") as out:
            for ep in range(int(args.episodes)):
                source_episode = int(args.source_episode) if args.source_episode is not None else int(ep)
                source_step = int(args.source_step) if args.source_step is not None else int(source_steps.get(source_episode, args.force_step if args.force_step is not None else 0))
                force_step = int(args.force_step) if args.force_step is not None else int(source_step)
                q_window, actual_source_step = _select_q_window(
                    trajectory,
                    source_episode,
                    source_step,
                    int(args.window_len),
                )
                for trial in range(int(args.trials)):
                    seed = base_seed + ep * 1000 + trial
                    result = _run_trial(
                        env=env,
                        ws=ws,
                        policy_observation_space=policy_observation_space,
                        env_action_shape=env_action_shape,
                        episode=ep,
                        trial=trial,
                        seed=seed,
                        steps=int(args.steps),
                        force_step=force_step,
                        source_episode=source_episode,
                        source_step=actual_source_step,
                        source_q_window=q_window,
                        force_dims=args.force_dims,
                        history_mode=args.history_mode,
                        output_dir=output_dir,
                        trajectory_handle=trajectory_handle,
                    )
                    results.append(result)
                    out.write(json.dumps(_jsonable_trace_value(asdict(result))) + "\n")
                    out.flush()
    finally:
        if trajectory_handle is not None:
            trajectory_handle.close()
        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary = _summarize(results, args, source_run)
    summary_path.write_text(json.dumps(_jsonable_trace_value(summary), indent=2) + "\n", encoding="utf-8")
    print("Saved:")
    print(f"  results: {results_path}")
    print(f"  summary: {summary_path}")
    if args.record_trajectories:
        print(f"  trajectory: {traj_path}")
    print("Summary:")
    for key in (
        "success_rate",
        "mean_forced_q_l2_from_clean",
        "mean_forced_arm_l2_from_clean",
        "mean_final_task_progress",
        "mean_max_task_progress",
        "mean_final_drawer_open_distance",
    ):
        print(f"  {key}: {summary.get(key)}")


if __name__ == "__main__":
    main()
