#!/usr/bin/env python3
"""ACT-only robustness ablation for recovery-to-tube state error.

This script answers a narrow question:

    If recovery brings the robot back close to the nominal ACT tube, but with the
    same q-space residual observed in the full SafeChunk run, can ACT alone still
    finish the drawer task when there is no safety filter and no human blocker?

It intentionally does NOT instantiate SafeChunk/OSCBF/recovery.  It only uses a
previous eval run to estimate the perturbation magnitude, then runs clean ACT
rollouts and injects that perturbation into the live MuJoCo H1 q state.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import numpy as np
from omegaconf import OmegaConf

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from robobase.safetyfilter.eval_utils.eval_config import (  # noqa: E402
    DEFAULT_EVAL_ARGS,
    _load_eval_config_container,
)
from robobase.safetyfilter.eval_utils.eval_environment import (  # noqa: E402
    _adapt_policy_obs_to_space,
    _diagnostic_task_state,
    _finite_task_progress,
    _find_wrapped_env_with_attr,
    _jsonable_trace_value,
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
    _load_snapshot_normalization_cfg,
    _make_eval_env_with_normalization,
    _resolve_normalization_cfg,
)
from robobase.safetyfilter.h1_state_bridge import (  # noqa: E402
    TREE_JOINT_NAMES,
    build_tree_to_mujoco_index_map,
    extract_h1_state,
    get_bigym_mojo,
)

REPO = Path("/home/xd1125/Workspace/safe_bigym_hoi")
ROBOBASE_ROOT = REPO / "external" / "robobase"
DEFAULT_SOURCE_PATTERN = "eval_safety/temporal_human_blocker_chunk_deform_no_affordance_noreset_*"

TUBE_ERROR_SCALAR_FIELDS = (
    "committed_rejoin_resume_tube_terminal_dist",
    "mpc_recovery_target_tube_terminal_dist",
    "recover_resume_tube_terminal_dist",
    "resume_tube_terminal_dist",
)
TUBE_ERROR_VECTOR_FIELDS = (
    "committed_rejoin_resume_tube_terminal_delta",
    "committed_rejoin_resume_tube_q_error",
    "mpc_recovery_target_tube_terminal_delta",
    "mpc_recovery_target_tube_q_error",
    "recover_resume_tube_terminal_delta",
    "resume_tube_terminal_delta",
)
REJOIN_STEP_FIELDS = (
    "resume_from_committed_rejoin",
    "post_recovery_act_bridge_started",
    "request_action_history_reset_after_recovery",
)


@dataclass
class TrialResult:
    episode: int
    trial: int
    variant: str
    seed: int
    perturb_step: int
    perturb_dim_mode: str
    perturb_l2_target: float
    perturb_l2_applied: float
    perturb_arm_l2_applied: float
    perturb_base_l2_applied: float
    source_error_quantile: float
    source_error_l2: float
    success: bool
    terminated: bool
    truncated: bool
    episode_length: int
    final_reward: float
    final_task_progress: Optional[float]
    max_task_progress: Optional[float]
    final_drawer_open_distance: Optional[float]
    final_drawer_open_fraction: Optional[float]
    progress_after_perturb: Optional[float]
    max_progress_after_perturb: Optional[float]
    q_at_perturb: Optional[list[float]]
    q_after_perturb: Optional[list[float]]
    perturb_vector: Optional[list[float]]
    output_dir: str


def _as_plain_dict(container: Any) -> dict[str, Any]:
    if container is None:
        return {}
    if isinstance(container, dict):
        return dict(container)
    return dict(OmegaConf.to_container(container, resolve=True) or {})


def _load_eval_args(eval_config: str) -> tuple[SimpleNamespace, dict[str, Any], Path]:
    container, resolved_path = _load_eval_config_container(eval_config)
    eval_args = copy.deepcopy(DEFAULT_EVAL_ARGS)
    eval_args.update(_as_plain_dict(container.get("eval_args")))
    eval_args["eval_config"] = str(resolved_path)
    eval_args.setdefault("override", [])
    if eval_args.get("override") is None:
        eval_args["override"] = []
    return SimpleNamespace(**eval_args), container, resolved_path


def _latest_matching_run(pattern: str) -> Optional[Path]:
    candidates = sorted(
        (ROBOBASE_ROOT.glob(pattern) if not Path(pattern).is_absolute() else Path("/").glob(str(Path(pattern).relative_to("/")))),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "metrics.jsonl").is_file():
            return candidate
        if candidate.is_file() and candidate.name.endswith(".jsonl"):
            return candidate.parent
    return None


def _resolve_source_metrics(source_run: Optional[str], source_metrics: Optional[str]) -> Optional[Path]:
    if source_metrics:
        path = Path(source_metrics).expanduser()
        if not path.is_absolute():
            path = ROBOBASE_ROOT / path
        return path
    if source_run:
        path = Path(source_run).expanduser()
        if not path.is_absolute():
            path = ROBOBASE_ROOT / path
        if path.is_file():
            return path
        return path / "metrics.jsonl"
    latest = _latest_matching_run(DEFAULT_SOURCE_PATTERN)
    if latest is None:
        return None
    return latest / "metrics.jsonl"


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
    return None


def _finite_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is None or not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _source_tube_error_stats(metrics_path: Optional[Path], fallback_error_l2: float) -> dict[str, Any]:
    rows = _load_jsonl(metrics_path) if metrics_path is not None else []
    scalars: list[float] = []
    vectors: list[list[float]] = []
    vector_l2s: list[float] = []
    rejoin_steps_by_episode: dict[int, list[int]] = {}
    source_fields: dict[str, int] = {}

    for row in rows:
        for field in TUBE_ERROR_SCALAR_FIELDS:
            value = _finite_float(row.get(field))
            if value is not None:
                scalars.append(value)
                source_fields[field] = source_fields.get(field, 0) + 1
                break
        for field in TUBE_ERROR_VECTOR_FIELDS:
            vec = _finite_vector(row.get(field))
            if vec is not None:
                vectors.append(vec.astype(float).tolist())
                vector_l2s.append(float(np.linalg.norm(vec)))
                source_fields[field] = source_fields.get(field, 0) + 1
                break
        if any(bool(row.get(field)) for field in REJOIN_STEP_FIELDS):
            ep = row.get("episode")
            step = row.get("step")
            if isinstance(ep, int) and isinstance(step, int):
                rejoin_steps_by_episode.setdefault(int(ep), []).append(int(step))

    if vector_l2s:
        scalars = vector_l2s
        source_fields["__vector_l2_as_error_scalar__"] = len(vector_l2s)
    elif not scalars and vectors:
        scalars = [float(np.linalg.norm(np.asarray(vec, dtype=np.float64))) for vec in vectors]
    if not scalars:
        scalars = [float(fallback_error_l2)]

    arr = np.asarray(scalars, dtype=np.float64)
    return {
        "metrics_path": None if metrics_path is None else str(metrics_path),
        "num_rows": int(len(rows)),
        "num_error_scalars": int(len(scalars)),
        "num_error_vectors": int(len(vectors)),
        "source_fields": source_fields,
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "values": arr.astype(float).tolist(),
        "vector_l2_values": [float(v) for v in vector_l2s],
        "vectors": vectors,
        "rejoin_steps_by_episode": {str(k): v for k, v in rejoin_steps_by_episode.items()},
    }


def _select_source_error(stats: dict[str, Any], quantile: float) -> float:
    values = np.asarray(stats.get("values", []), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, float(np.clip(quantile, 0.0, 1.0))))


def _infer_clean_env(eval_env: str, snapshot_path: Path, clean_env: str) -> str:
    if clean_env and clean_env != "auto":
        return clean_env
    try:
        snapshot_cfg = _load_snapshot_normalization_cfg(snapshot_path)
        task_name = str(snapshot_cfg.env.task_name)
        if task_name:
            clean_task = task_name.replace("human_arm_", "", 1) if "human_arm_" in task_name else task_name
            return clean_task if clean_task.startswith("bigym/") else f"bigym/{clean_task}"
    except Exception:  # noqa: BLE001
        pass
    env = str(eval_env)
    if "human_arm_" in env:
        return env.replace("human_arm_", "", 1)
    return env


def _make_runtime_namespace(base_args: SimpleNamespace, **updates: Any) -> SimpleNamespace:
    data = vars(copy.deepcopy(base_args))
    data.update(updates)
    return SimpleNamespace(**data)


def _mujoco_forward(mojo: Any) -> None:
    physics = getattr(mojo, "physics", None)
    forward = getattr(physics, "forward", None) if physics is not None else None
    if forward is None:
        forward = getattr(mojo, "forward", None)
    if callable(forward):
        forward()


def _set_h1_q(env: Any, q_full: np.ndarray, *, zero_velocity: bool = True) -> bool:
    q_full = np.asarray(q_full, dtype=np.float64).reshape(-1)
    if q_full.size < len(TREE_JOINT_NAMES):
        return False
    mojo = get_bigym_mojo(env)
    model = mojo.model
    data = mojo.data
    tree_to_mj = build_tree_to_mujoco_index_map(env, TREE_JOINT_NAMES)
    for i, joint_name in enumerate(TREE_JOINT_NAMES):
        joint_id = tree_to_mj[joint_name]
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = float(q_full[i])
        if zero_velocity:
            data.qvel[dof_adr] = 0.0
    _mujoco_forward(mojo)
    return True


def _current_obs(env: Any, fallback_obs: Any) -> Any:
    getter_owner = _find_wrapped_env_with_attr(env, "get_observation")
    getter = getattr(getter_owner, "get_observation", None) if getter_owner is not None else None
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001
            return fallback_obs
    unwrapped = getattr(env, "unwrapped", None)
    getter = getattr(unwrapped, "get_observation", None) if unwrapped is not None else None
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001
            return fallback_obs
    return fallback_obs


def _merge_current_obs_into_policy_stack(policy_obs: Any, current_obs: Any) -> Any:
    if not isinstance(policy_obs, dict) or not isinstance(current_obs, dict):
        return policy_obs
    merged = copy.deepcopy(policy_obs)
    for key, current_value in current_obs.items():
        if key not in merged:
            continue
        try:
            target = np.asarray(merged[key])
            current = np.asarray(current_value)
        except Exception:  # noqa: BLE001
            continue
        if target.shape == current.shape:
            merged[key] = current.astype(target.dtype, copy=False)
            continue
        if target.ndim >= 2 and current.shape == target.shape[1:]:
            updated = target.copy()
            updated[:-1] = updated[1:]
            updated[-1] = current.astype(target.dtype, copy=False)
            merged[key] = updated
    return merged


def _perturb_indices(mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "controlled":
        return np.arange(len(TREE_JOINT_NAMES), dtype=np.int64)
    if mode == "base":
        return np.arange(0, 4, dtype=np.int64)
    if mode == "arm":
        return np.arange(4, len(TREE_JOINT_NAMES), dtype=np.int64)
    if mode == "right_arm":
        return np.arange(9, len(TREE_JOINT_NAMES), dtype=np.int64)
    if mode == "left_arm":
        return np.arange(4, 9, dtype=np.int64)
    raise ValueError(
        "--perturb-dims must be one of: controlled, base, arm, right_arm, left_arm"
    )


def _make_perturb_vector(
    q_dim: int,
    indices: np.ndarray,
    l2_target: float,
    rng: np.random.Generator,
    source_vectors: Iterable[Any] = (),
    trial: int = 0,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    vec = np.zeros(q_dim, dtype=np.float64)
    source_list = list(source_vectors or [])
    if source_list:
        source = _finite_vector(source_list[int(trial) % len(source_list)])
        if source is not None and source.size >= q_dim:
            candidate = np.zeros(q_dim, dtype=np.float64)
            candidate[indices] = source[:q_dim][indices]
            norm = float(np.linalg.norm(candidate[indices]))
            if norm > 1e-12:
                vec[indices] = candidate[indices] / norm * float(l2_target)
                return vec
    direction = rng.normal(size=indices.size)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.ones(indices.size, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
    vec[indices] = direction / norm * float(l2_target)
    return vec


def _episode_rejoin_step(stats: dict[str, Any], episode: int, fallback_step: int) -> int:
    by_episode = stats.get("rejoin_steps_by_episode", {}) or {}
    values = by_episode.get(str(int(episode))) or by_episode.get(int(episode)) or []
    if values:
        return int(values[0])
    all_steps = []
    for steps in by_episode.values():
        if isinstance(steps, list):
            all_steps.extend(int(s) for s in steps if isinstance(s, int))
    if all_steps:
        return int(np.median(np.asarray(all_steps, dtype=np.int64)))
    return int(fallback_step)


def _task_progress_from_state(task_state: dict[str, Any]) -> Optional[float]:
    return _finite_task_progress(task_state)


def _run_single_trial(
    *,
    env: Any,
    ws: Any,
    policy_observation_space: Any,
    env_action_shape: tuple[int, ...],
    episode: int,
    trial: int,
    seed: int,
    steps: int,
    perturb_step: int,
    perturb_l2: float,
    perturb_dims: str,
    source_error_l2: float,
    source_error_quantile: float,
    source_vectors: list[Any],
    output_dir: Path,
    trajectory_handle: Optional[Any],
) -> TrialResult:
    obs, _info = env.reset(seed=int(seed))
    policy_obs = obs
    policy_step = 0
    rng_seed = (int(seed) + 1) * 1009 + (int(trial) + 2) * 9176 + (int(episode) + 1) * 101
    rng = np.random.default_rng(int(rng_seed))
    variant = "baseline" if trial < 0 or perturb_l2 <= 0.0 else "perturbed"

    q_at_perturb = None
    q_after_perturb = None
    perturb_vector = None
    perturb_l2_applied = 0.0
    perturb_arm_l2 = 0.0
    perturb_base_l2 = 0.0
    progress_after_perturb = None
    progress_values: list[float] = []
    progress_values_after_perturb: list[float] = []
    final_reward = 0.0
    success = False
    terminated = False
    truncated = False
    final_task_state: dict[str, Any] = {}
    step_count = 0

    for step in range(int(steps)):
        if step == int(perturb_step) and variant == "perturbed":
            h1_state = extract_h1_state(env)
            q_before = np.asarray(h1_state.q_full, dtype=np.float64).reshape(-1)
            indices = _perturb_indices(perturb_dims)
            delta = _make_perturb_vector(
                q_before.size,
                indices,
                float(perturb_l2),
                rng,
                source_vectors=source_vectors,
                trial=max(0, trial),
            )
            q_target = q_before + delta
            _set_h1_q(env, q_target, zero_velocity=True)
            q_after = np.asarray(extract_h1_state(env).q_full, dtype=np.float64).reshape(-1)
            applied = q_after - q_before
            q_at_perturb = q_before.astype(float).tolist()
            q_after_perturb = q_after.astype(float).tolist()
            perturb_vector = applied.astype(float).tolist()
            perturb_l2_applied = float(np.linalg.norm(applied))
            perturb_base_l2 = float(np.linalg.norm(applied[: min(4, applied.size)]))
            perturb_arm_l2 = float(np.linalg.norm(applied[min(4, applied.size) :]))
            policy_obs = _merge_current_obs_into_policy_stack(policy_obs, _current_obs(env, policy_obs))
            task_state = _diagnostic_task_state(env)
            progress_after_perturb = _task_progress_from_state(task_state)

        task_state_before = _diagnostic_task_state(env)
        progress_before = _task_progress_from_state(task_state_before)
        if progress_before is not None:
            progress_values.append(float(progress_before))
            if step >= int(perturb_step):
                progress_values_after_perturb.append(float(progress_before))

        policy_obs_for_action = _adapt_policy_obs_to_space(
            policy_obs,
            policy_observation_space,
        )
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
            if step >= int(perturb_step):
                progress_values_after_perturb.append(float(final_progress))

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
                            "variant": variant,
                            "step": int(step),
                            "perturb_step": int(perturb_step),
                            "success": bool(success),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "reward": float(reward),
                            "task_state": final_task_state,
                            "q_full": q_now,
                            "action_shape": list(env_action.shape),
                            "action_first": np.asarray(env_action[0] if env_action.ndim == 2 else env_action, dtype=float).tolist(),
                        }
                    )
                )
                + "\n"
            )

        if terminated or truncated or success:
            break

    final_progress = _task_progress_from_state(final_task_state)
    max_progress = max(progress_values) if progress_values else None
    max_progress_after = max(progress_values_after_perturb) if progress_values_after_perturb else None

    return TrialResult(
        episode=int(episode),
        trial=int(trial),
        variant=variant,
        seed=int(seed),
        perturb_step=int(perturb_step),
        perturb_dim_mode=str(perturb_dims),
        perturb_l2_target=float(perturb_l2),
        perturb_l2_applied=float(perturb_l2_applied),
        perturb_arm_l2_applied=float(perturb_arm_l2),
        perturb_base_l2_applied=float(perturb_base_l2),
        source_error_quantile=float(source_error_quantile),
        source_error_l2=float(source_error_l2),
        success=bool(success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        episode_length=int(step_count),
        final_reward=float(final_reward),
        final_task_progress=None if final_progress is None else float(final_progress),
        max_task_progress=None if max_progress is None else float(max_progress),
        final_drawer_open_distance=_finite_float(final_task_state.get("drawer_open_distance")),
        final_drawer_open_fraction=_finite_float(final_task_state.get("drawer_open_fraction")),
        progress_after_perturb=None if progress_after_perturb is None else float(progress_after_perturb),
        max_progress_after_perturb=None if max_progress_after is None else float(max_progress_after),
        q_at_perturb=q_at_perturb,
        q_after_perturb=q_after_perturb,
        perturb_vector=perturb_vector,
        output_dir=str(output_dir),
    )


def _summarize_trials(results: list[TrialResult], source_stats: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    def group(variant: str) -> list[TrialResult]:
        return [r for r in results if r.variant == variant]

    def rate(items: list[TrialResult]) -> Optional[float]:
        if not items:
            return None
        return float(np.mean([1.0 if r.success else 0.0 for r in items]))

    def mean_field(items: list[TrialResult], field: str) -> Optional[float]:
        vals = [getattr(r, field) for r in items]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if not vals:
            return None
        return float(np.mean(vals))

    baseline = group("baseline")
    perturbed = group("perturbed")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_config": args.eval_config,
        "clean_env": args.clean_env,
        "source_run": args.source_run,
        "source_metrics": args.source_metrics,
        "episodes": int(args.episodes),
        "num_perturbations": int(args.num_perturbations),
        "steps": int(args.steps),
        "perturb_dims": args.perturb_dims,
        "perturb_step": args.perturb_step,
        "error_quantile": float(args.error_quantile),
        "error_multiplier": float(args.error_multiplier),
        "source_tube_error_stats": source_stats,
        "num_trials": int(len(results)),
        "baseline_trials": int(len(baseline)),
        "perturbed_trials": int(len(perturbed)),
        "baseline_success_rate": rate(baseline),
        "perturbed_success_rate": rate(perturbed),
        "baseline_mean_final_task_progress": mean_field(baseline, "final_task_progress"),
        "perturbed_mean_final_task_progress": mean_field(perturbed, "final_task_progress"),
        "baseline_mean_max_task_progress": mean_field(baseline, "max_task_progress"),
        "perturbed_mean_max_task_progress": mean_field(perturbed, "max_task_progress"),
        "perturbed_mean_applied_l2": mean_field(perturbed, "perturb_l2_applied"),
        "perturbed_mean_applied_arm_l2": mean_field(perturbed, "perturb_arm_l2_applied"),
        "perturbed_mean_applied_base_l2": mean_field(perturbed, "perturb_base_l2_applied"),
    }
    base_rate = summary["baseline_success_rate"]
    pert_rate = summary["perturbed_success_rate"]
    if base_rate is not None and pert_rate is not None:
        summary["success_rate_drop"] = float(base_rate - pert_rate)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACT-only ablation: inject recovery-to-tube q error in clean Bigym env."
    )
    parser.add_argument(
        "--eval-config",
        default="robobase/cfgs/eval_scenarios/experiments/temporal_human_blocker_chunk_deform_no_affordance_noreset.yaml",
        help="Existing eval config used only for ACT snapshot/architecture overrides.",
    )
    parser.add_argument("--snapshot", default=None, help="Override ACT snapshot path.")
    parser.add_argument(
        "--clean-env",
        default="auto",
        help="Clean ACT runtime env. 'auto' uses the checkpoint task, avoiding the human-arm env.",
    )
    parser.add_argument("--source-run", default=None, help="Previous eval run dir containing metrics.jsonl.")
    parser.add_argument("--source-metrics", default=None, help="Previous metrics.jsonl path.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to eval_safety/act_tube_error_ablation_<timestamp>.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-perturbations", type=int, default=4)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--perturb-step",
        type=int,
        default=-1,
        help="Step to inject perturbation. -1 uses committed rejoin step from source metrics, else 54.",
    )
    parser.add_argument(
        "--perturb-dims",
        default="controlled",
        choices=("controlled", "base", "arm", "right_arm", "left_arm"),
    )
    parser.add_argument("--error-quantile", type=float, default=0.95)
    parser.add_argument("--error-multiplier", type=float, default=1.0)
    parser.add_argument("--fallback-error-l2", type=float, default=0.04)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--normalization-source", default=None, choices=(None, "auto", "eval", "snapshot"))
    parser.add_argument("--demos", type=int, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override for ACT/env construction. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    eval_args, _container, resolved_eval_config = _load_eval_args(cli_args.eval_config)
    snapshot_path = Path(cli_args.snapshot or eval_args.snapshot).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = ROBOBASE_ROOT / snapshot_path
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    clean_env = _infer_clean_env(str(eval_args.env), snapshot_path, cli_args.clean_env)
    cli_args.clean_env = clean_env
    source_metrics = _resolve_source_metrics(cli_args.source_run, cli_args.source_metrics)
    source_stats = _source_tube_error_stats(source_metrics, cli_args.fallback_error_l2)
    source_error_l2 = _select_source_error(source_stats, cli_args.error_quantile)
    perturb_l2 = float(source_error_l2) * float(cli_args.error_multiplier)

    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else ROBOBASE_ROOT / "eval_safety" / f"act_tube_error_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not output_dir.is_absolute():
        output_dir = ROBOBASE_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_override = list(eval_args.override or []) + list(cli_args.override or [])
    runtime_args = _make_runtime_namespace(
        eval_args,
        env=clean_env,
        episodes=int(cli_args.episodes),
        steps=int(cli_args.steps),
        seed=int(cli_args.seed),
        demos=int(cli_args.demos if cli_args.demos is not None else eval_args.demos),
        override=runtime_override,
        normalization_source=cli_args.normalization_source or eval_args.normalization_source,
    )
    cfg = make_cfg(runtime_args)
    normalization_source, normalization_cfg = _resolve_normalization_cfg(runtime_args, cfg, snapshot_path)

    print("=== ACT tube-error ablation ===")
    print("eval_config:", resolved_eval_config)
    print("snapshot:", snapshot_path)
    print("clean_env:", clean_env)
    print("source_metrics:", source_stats.get("metrics_path"))
    print("source_error_l2:", source_error_l2)
    print("perturb_l2:", perturb_l2)
    print("perturb_dims:", cli_args.perturb_dims)
    print("normalization_source:", normalization_source)
    print("output_dir:", output_dir)

    print("\n=== Creating ACT workspace ===")
    ws = make_workspace_and_load_snapshot(cfg, snapshot_path)
    policy_observation_space = getattr(ws.eval_env, "observation_space", None)

    print("\n=== Creating clean ACT env ===")
    env = _make_eval_env_with_normalization(cfg, normalization_cfg)
    env_action_shape = infer_env_action_shape(env)
    print("env_action_shape:", env_action_shape)

    results_path = output_dir / "act_tube_error_ablation_results.jsonl"
    summary_path = output_dir / "act_tube_error_ablation_summary.json"
    source_path = output_dir / "source_tube_error_stats.json"
    trajectory_path = output_dir / "act_tube_error_ablation_trajectories.jsonl"

    results: list[TrialResult] = []
    source_vectors = list(source_stats.get("vectors", []) or [])
    trajectory_handle = trajectory_path.open("w", encoding="utf-8") if cli_args.save_trajectories else None
    try:
        for episode in range(int(cli_args.episodes)):
            base_seed = int(cli_args.seed) + int(episode)
            perturb_step = (
                _episode_rejoin_step(source_stats, episode, 54)
                if int(cli_args.perturb_step) < 0
                else int(cli_args.perturb_step)
            )
            trial_ids: list[int] = [] if cli_args.no_baseline else [-1]
            trial_ids.extend(range(int(cli_args.num_perturbations)))
            for trial in trial_ids:
                trial_seed = base_seed
                trial_perturb_l2 = 0.0 if trial < 0 else perturb_l2
                result = _run_single_trial(
                    env=env,
                    ws=ws,
                    policy_observation_space=policy_observation_space,
                    env_action_shape=env_action_shape,
                    episode=episode,
                    trial=trial,
                    seed=trial_seed,
                    steps=int(cli_args.steps),
                    perturb_step=perturb_step,
                    perturb_l2=trial_perturb_l2,
                    perturb_dims=cli_args.perturb_dims,
                    source_error_l2=source_error_l2,
                    source_error_quantile=float(cli_args.error_quantile),
                    source_vectors=source_vectors,
                    output_dir=output_dir,
                    trajectory_handle=trajectory_handle,
                )
                results.append(result)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_jsonable_trace_value(asdict(result))) + "\n")
                print(
                    f"episode={episode} trial={trial} {result.variant} "
                    f"success={result.success} len={result.episode_length} "
                    f"final_progress={result.final_task_progress} "
                    f"max_progress={result.max_task_progress} "
                    f"applied_l2={result.perturb_l2_applied:.5f}"
                )
    finally:
        if trajectory_handle is not None:
            trajectory_handle.close()
        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary = _summarize_trials(results, source_stats, cli_args)
    summary["snapshot"] = str(snapshot_path)
    summary["resolved_eval_config"] = str(resolved_eval_config)
    summary["normalization_source"] = str(normalization_source)
    summary["results_path"] = str(results_path)
    summary["summary_path"] = str(summary_path)
    summary["trajectory_path"] = str(trajectory_path) if cli_args.save_trajectories else None

    source_path.write_text(json.dumps(_jsonable_trace_value(source_stats), indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_jsonable_trace_value(summary), indent=2) + "\n", encoding="utf-8")

    print("\n=== Summary ===")
    for key in (
        "baseline_success_rate",
        "perturbed_success_rate",
        "success_rate_drop",
        "baseline_mean_final_task_progress",
        "perturbed_mean_final_task_progress",
        "baseline_mean_max_task_progress",
        "perturbed_mean_max_task_progress",
        "perturbed_mean_applied_l2",
        "perturbed_mean_applied_arm_l2",
        "perturbed_mean_applied_base_l2",
    ):
        print(f"{key}: {summary.get(key)}")
    print("Saved results:", results_path)
    print("Saved summary:", summary_path)
    print("Saved source stats:", source_path)
    if cli_args.save_trajectories:
        print("Saved trajectories:", trajectory_path)


if __name__ == "__main__":
    main()
