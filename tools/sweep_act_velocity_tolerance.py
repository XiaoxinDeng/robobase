#!/usr/bin/env python3
"""Sweep ACT tolerance to local velocity perturbations.

This is a standalone diagnostic tool. It intentionally does not add another mode
inside eval_act_oscbf_safety_metrics.py.

Experiment idea:
  1. Run ACT normally until a perturbation step.
  2. Query ACT for its predicted action horizon.
  3. Select a short window of consecutive nominal waypoints inside that horizon.
  4. Scale the local waypoint increments to emulate speed-up, slow-down,
     acceleration, or deceleration.
  5. Execute the perturbed waypoint(s), then hand control back to ACT and record
     whether the task still succeeds.

The default perturbation only touches the base+arm action dimensions used by the
safety filter and preserves gripper/pass-through dimensions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

ROBOBASE_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOBASE_ROOT))

from robobase.safetyfilter.eval_utils.eval_utils import (  # noqa: E402
    extract_first_action,
    extract_success,
    infer_env_action_shape,
    make_cfg,
    make_workspace_and_load_snapshot,
    normalise_env_action_shape,
    policy_action,
)
from robobase.safetyfilter.eval_utils.eval_config import (  # noqa: E402
    DEFAULT_EVAL_ARGS,
    _load_eval_config_defaults,
)
from robobase.safetyfilter.eval_utils.eval_environment import (  # noqa: E402
    _apply_final_human_arm_clearance,
    _apply_robot_spawn_offset_xy,
    _configure_human_arm_challenge,
    _disable_human_arm_collisions,
    _enable_human_arm_collisions,
    _freeze_human_arm,
    _make_policy_env_cfg,
    _policy_obs_with_hidden_human_arm,
    _reset_action_sequence_history,
    _update_temporary_human_blocker_if_present,
)
from robobase.safetyfilter.h1_state_bridge import extract_h1_state  # noqa: E402
from robobase.safetyfilter.eval_utils.eval_video import (  # noqa: E402
    _load_snapshot_normalization_cfg,
    _make_eval_env_with_normalization,
    _resolve_normalization_cfg,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    tqdm = None


DEFAULT_CONTROL_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12)


def _make_progress_bar(total: int, enabled: bool):
    if not enabled:
        return None
    if tqdm is None:
        return None
    return tqdm(
        total=int(total),
        desc="velocity tolerance trials",
        dynamic_ncols=True,
        leave=True,
    )


def _progress_update(progress, completed: int, total: int, result: dict[str, Any], enabled: bool) -> None:
    if not enabled:
        return
    if progress is not None:
        progress.update(1)
        progress.set_postfix(
            variant=result.get("variant_id"),
            success=bool(result.get("success")),
            scale=result.get("scale"),
            profile=result.get("profile"),
            refresh=False,
        )
        return
    if completed == 1 or completed == total or completed % max(1, min(10, total // 10 or 1)) == 0:
        print(
            f"velocity tolerance trials: {completed}/{total} "
            f"variant={result.get('variant_id')} "
            f"success={bool(result.get('success'))} "
            f"profile={result.get('profile')} scale={result.get('scale')}",
            flush=True,
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        arr = np.asarray(value)
        if arr.ndim > 0:
            return arr.tolist()
        return _jsonable(arr.item())
    except Exception:  # noqa: BLE001
        return str(value)


def _parse_csv_values(text: str | None, cast, *, default: Iterable[Any]) -> list[Any]:
    if text is None or str(text).strip() == "":
        return list(default)
    values: list[Any] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def _parse_window_starts(text: str, horizon: int, window_len: int) -> list[int]:
    if str(text).strip().lower() == "all":
        return list(range(0, max(0, horizon - window_len + 1)))
    starts = _parse_csv_values(text, int, default=(0,))
    return [s for s in starts if 0 <= s <= horizon - window_len]


def _make_args(eval_config: str, cli_args: argparse.Namespace) -> SimpleNamespace:
    defaults = copy.deepcopy(DEFAULT_EVAL_ARGS)
    config_defaults, resolved_paths = _load_eval_config_defaults(eval_config)
    defaults.update(config_defaults)

    for key in ("episodes", "steps", "demos", "seed", "snapshot", "env"):
        value = getattr(cli_args, key, None)
        if value is not None:
            defaults[key] = value

    if cli_args.output_dir is not None:
        defaults["output_dir"] = cli_args.output_dir
    elif defaults.get("output_dir") in (None, "eval_safety"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        defaults["output_dir"] = f"eval_safety/act_velocity_tolerance_{stamp}"

    defaults.setdefault("override", [])
    defaults["eval_config"] = eval_config
    defaults["resolved_eval_config_paths"] = [str(path) for path in resolved_paths]
    defaults["condition"] = "act_velocity_tolerance"
    defaults["no_record_video"] = True
    defaults["hide_human_arm_policy_obs"] = bool(cli_args.hide_human_arm_policy_obs)
    return SimpleNamespace(**defaults)


def _reset_env(env, seed: int | None):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _step_env(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        return out
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), False, info
    raise ValueError(f"Unsupported env.step output: {type(out)} length={len(out) if isinstance(out, tuple) else 'n/a'}")


def _adapt_policy_obs_to_space(obs: Any, observation_space: Any) -> Any:
    if not isinstance(obs, dict):
        return obs
    spaces = getattr(observation_space, "spaces", None)
    if not isinstance(spaces, dict):
        return obs
    adapted = {key: np.asarray(obs[key]) for key in spaces if key in obs}
    return adapted if adapted else obs


def _act_chunk(ws, obs, step: int, expected_shape: tuple[int, ...], observation_space: Any) -> np.ndarray:
    policy_obs = _adapt_policy_obs_to_space(obs, observation_space)
    action = policy_action(ws, policy_obs, step=step)
    return normalise_env_action_shape(action, expected_shape)


def _as_chunk(action: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 2:
        return action.copy()
    if action.ndim == 1 and len(expected_shape) == 2:
        return np.repeat(action.reshape(1, -1), expected_shape[0], axis=0).astype(np.float32)
    if action.ndim == 1:
        return action.reshape(1, -1).astype(np.float32)
    raise ValueError(f"Unsupported action shape for chunk conversion: {action.shape}")


def _control_indices(text: str, action_dim: int) -> np.ndarray:
    if str(text).strip().lower() == "all":
        raw = list(range(action_dim))
    elif str(text).strip().lower() == "default":
        raw = list(DEFAULT_CONTROL_INDICES)
    else:
        raw = _parse_csv_values(text, int, default=DEFAULT_CONTROL_INDICES)
    idx = np.asarray(raw, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < action_dim)]
    return np.unique(idx)


def _direction_axes(text: str, action_dim: int, control_idx: np.ndarray) -> np.ndarray:
    text = str(text).strip().lower()
    if text in ("", "auto", "default"):
        if control_idx.size >= 2:
            return np.unique(control_idx[:2])
        return np.unique(control_idx)
    raw = _parse_csv_values(text, int, default=(0, 1))
    idx = np.asarray(raw, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < action_dim)]
    if idx.size == 0 and control_idx.size >= 2:
        return np.unique(control_idx[:2])
    return np.unique(idx)


def _profile_multipliers(profile: str, scale: float, window_len: int) -> np.ndarray:
    profile = str(profile).lower()
    if window_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if profile == "constant":
        return np.full((window_len,), float(scale), dtype=np.float32)
    if window_len == 1:
        return np.asarray([float(scale)], dtype=np.float32)
    if profile == "accelerate":
        return np.linspace(1.0, float(scale), window_len, dtype=np.float32)
    if profile == "decelerate":
        return np.linspace(float(scale), 1.0, window_len, dtype=np.float32)
    if profile == "pulse":
        out = np.ones((window_len,), dtype=np.float32)
        out[0] = float(scale)
        return out
    raise ValueError(f"Unknown profile: {profile}")


def _perturb_chunk_velocity(
    nominal_chunk: np.ndarray,
    *,
    window_start: int,
    window_len: int,
    scale: float,
    profile: str,
    control_idx: np.ndarray,
    perturbation_mode: str,
    direction_axes: np.ndarray,
    direction_angle_scale: float,
    previous_first_action: np.ndarray | None,
    clip_action: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    perturbation_mode = str(perturbation_mode).lower()
    if perturbation_mode not in ("magnitude", "direction"):
        perturbation_mode = "magnitude"

    nominal = np.asarray(nominal_chunk, dtype=np.float32)
    perturbed = nominal.copy()
    horizon, action_dim = perturbed.shape
    end = min(horizon, int(window_start) + int(window_len))
    if window_start < 0 or window_start >= horizon or end <= window_start:
        return perturbed, {"perturb_skipped": True, "perturb_skip_reason": "invalid_window"}

    idx = np.asarray(control_idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < action_dim)]
    if idx.size == 0:
        return perturbed, {"perturb_skipped": True, "perturb_skip_reason": "empty_control_indices"}

    direction_idx = np.asarray(direction_axes, dtype=np.int64)
    direction_idx = direction_idx[(direction_idx >= 0) & (direction_idx < action_dim)]
    direction_pos = np.array(
        [int(np.where(idx == axis)[0][0]) for axis in direction_idx if np.any(idx == axis)],
        dtype=np.int64,
    )
    if perturbation_mode == "direction":
        if direction_pos.size < 2:
            return perturbed, {
                "perturb_skipped": True,
                "perturb_skip_reason": "invalid_direction_axes",
            }
        direction_pos = np.unique(direction_pos)[:2]
        first_axis = int(direction_pos[0])
        second_axis = int(direction_pos[1])

    multipliers = _profile_multipliers(profile, scale, end - window_start)
    if window_start > 0:
        nominal_prev = nominal[window_start - 1, idx].copy()
        perturbed_prev = perturbed[window_start - 1, idx].copy()
    elif previous_first_action is not None:
        previous = np.asarray(previous_first_action, dtype=np.float32).reshape(-1)
        nominal_prev = previous[idx].copy()
        perturbed_prev = previous[idx].copy()
    else:
        nominal_prev = nominal[window_start, idx].copy()
        perturbed_prev = nominal_prev.copy()

    direction_angles: list[float] = []
    original = nominal.copy()
    for local_i, row in enumerate(range(window_start, end)):
        nominal_delta = nominal[row, idx] - nominal_prev
        nominal_delta = nominal_delta.astype(np.float32)
        perturbed_delta = nominal_delta.copy()
        if perturbation_mode == "direction":
            angle = (float(multipliers[local_i]) - 1.0) * float(direction_angle_scale)
            direction_angles.append(float(angle))
            c = float(math.cos(angle))
            s = float(math.sin(angle))
            v0 = float(perturbed_delta[first_axis])
            v1 = float(perturbed_delta[second_axis])
            perturbed_delta[first_axis] = v0 * c - v1 * s
            perturbed_delta[second_axis] = v0 * s + v1 * c
        else:
            direction_angles.append(0.0)
            perturbed_delta = perturbed_delta * float(multipliers[local_i])

        perturbed[row, idx] = perturbed_prev + perturbed_delta
        nominal_prev = nominal[row, idx].copy()
        perturbed_prev = perturbed[row, idx].copy()

    clip_count = 0
    clip_max_abs_excess = 0.0
    if clip_action:
        before_clip = perturbed.copy()
        perturbed = np.clip(perturbed, -1.0, 1.0)
        clipped_delta = np.abs(before_clip - perturbed)
        clip_count = int(np.count_nonzero(clipped_delta > 1e-7))
        clip_max_abs_excess = float(np.max(clipped_delta)) if clipped_delta.size else 0.0

    delta = perturbed - original
    window_delta = delta[window_start:end, :]
    control_delta = window_delta[:, idx]
    return perturbed.astype(np.float32), {
        "perturb_skipped": False,
        "profile": profile,
        "scale": float(scale),
        "window_start": int(window_start),
        "window_len": int(end - window_start),
        "control_indices": idx.astype(int).tolist(),
        "multipliers": multipliers.astype(float).tolist(),
        "perturbation_mode": perturbation_mode,
        "perturb_l2": float(np.linalg.norm(control_delta)),
        "perturb_linf": float(np.max(np.abs(control_delta))) if control_delta.size else 0.0,
        "clip_count": clip_count,
        "clip_max_abs_excess": clip_max_abs_excess,
        "direction_axes": idx[direction_pos].astype(int).tolist() if perturbation_mode == "direction" and direction_pos.size > 1 else [],
        "direction_angle_scale": float(direction_angle_scale) if perturbation_mode == "direction" else 0.0,
        "direction_angles_rad": direction_angles,
        "direction_angle_deg_max_abs": float(np.max(np.abs(direction_angles))) if direction_angles else 0.0,
    }


def _single_waypoint_action(
    desired_first_action: np.ndarray,
    template_chunk: np.ndarray,
    expected_shape: tuple[int, ...],
    tail: str,
) -> np.ndarray:
    desired = np.asarray(desired_first_action, dtype=np.float32).reshape(-1)
    if len(expected_shape) == 1:
        return desired.astype(np.float32)
    horizon = int(expected_shape[0])
    if tail == "nominal":
        out = np.asarray(template_chunk, dtype=np.float32).copy()
        out[0] = desired
        return out
    if tail == "repeat":
        return np.repeat(desired.reshape(1, -1), horizon, axis=0).astype(np.float32)
    raise ValueError(f"Unknown waypoint tail mode: {tail}")


def _configure_env_runtime(env, args) -> None:
    _configure_human_arm_challenge(env, args)
    if bool(getattr(args, "enable_human_arm_collisions", False)):
        _enable_human_arm_collisions(env)
    else:
        _disable_human_arm_collisions(env)
    if bool(getattr(args, "freeze_human_arm", False)):
        _freeze_human_arm(env)


def _h1_snapshot(env, prefix: str) -> dict[str, Any]:
    try:
        state = extract_h1_state(env)
    except Exception:  # noqa: BLE001
        return {}
    q = np.asarray(state.q_full, dtype=np.float64).reshape(-1)
    qd = np.asarray(state.qd_full, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_q": q.tolist(),
        f"{prefix}_qd": qd.tolist(),
        f"{prefix}_q_norm": float(np.linalg.norm(q)),
        f"{prefix}_qd_norm": float(np.linalg.norm(qd)),
    }


def _task_success_from_info(info: Any, reward: float, terminated: bool) -> bool:
    try:
        return bool(extract_success(info, reward, terminated))
    except Exception:  # noqa: BLE001
        return False


def _run_one_trial(
    *,
    env,
    ws,
    args,
    expected_shape: tuple[int, ...],
    observation_space: Any,
    episode: int,
    variant: dict[str, Any],
    step_writer,
) -> dict[str, Any]:
    seed = int(args.seed) + int(episode)
    obs, reset_info = _reset_env(env, seed)
    _configure_env_runtime(env, args)
    policy_obs = (
        _policy_obs_with_hidden_human_arm(env, obs)
        if bool(getattr(args, "hide_human_arm_policy_obs", True))
        else obs
    )

    total_reward = 0.0
    success = False
    success_step = None
    terminated = False
    truncated = False
    previous_first_action = None
    perturb_done = False
    perturb_info: dict[str, Any] = {}
    before_state: dict[str, Any] = {}
    after_state: dict[str, Any] = {}
    action_dim = expected_shape[-1]
    control_idx = _control_indices(str(variant["control_indices"]), action_dim)
    perturbation_mode = str(variant["perturbation_mode"])
    direction_axes = (
        _direction_axes(str(variant.get("direction_axes", "auto")), action_dim, control_idx)
        if perturbation_mode == "direction"
        else np.asarray([], dtype=np.int64)
    )
    direction_angle_scale = float(variant.get("direction_angle_scale", 0.0))

    step = 0
    trial_start = time.perf_counter()
    while step < int(args.steps):
        _update_temporary_human_blocker_if_present(env)
        _apply_final_human_arm_clearance(env, args, step)

        if step == int(variant["perturb_step"]) and not perturb_done:
            nominal_action = _act_chunk(ws, policy_obs, step, expected_shape, observation_space)
            nominal_chunk = _as_chunk(nominal_action, expected_shape)
            before_state = _h1_snapshot(env, "before_perturb")
            perturbed_chunk, perturb_info = _perturb_chunk_velocity(
                nominal_chunk,
                window_start=int(variant["window_start"]),
                window_len=int(variant["window_len"]),
                scale=float(variant["scale"]),
                profile=str(variant["profile"]),
                control_idx=control_idx,
                perturbation_mode=perturbation_mode,
                direction_axes=direction_axes,
                direction_angle_scale=direction_angle_scale,
                previous_first_action=previous_first_action,
                clip_action=bool(variant["clip_action"]),
            )
            perturb_done = True

            if str(variant["execution_mode"]) == "chunk":
                actions_to_execute = [perturbed_chunk]
                action_sources = ["perturbed_chunk"]
            else:
                start = int(perturb_info.get("window_start", variant["window_start"]))
                length = int(perturb_info.get("window_len", variant["window_len"]))
                rows = range(start, min(start + length, perturbed_chunk.shape[0]))
                actions_to_execute = [
                    _single_waypoint_action(
                        perturbed_chunk[row],
                        nominal_chunk,
                        expected_shape,
                        str(variant["waypoint_tail"]),
                    )
                    for row in rows
                ]
                action_sources = [f"perturbed_waypoint_{row}" for row in rows]

            for local_i, action_to_execute in enumerate(actions_to_execute):
                if step >= int(args.steps):
                    break
                prev_policy_obs = policy_obs
                obs, reward, terminated, truncated, info = _step_env(env, action_to_execute)
                policy_obs = (
                    _policy_obs_with_hidden_human_arm(
                        env,
                        obs,
                        prev_policy_obs=prev_policy_obs,
                    )
                    if bool(getattr(args, "hide_human_arm_policy_obs", True))
                    else obs
                )
                reward_float = float(np.asarray(reward).reshape(-1)[0])
                total_reward += reward_float
                step_success = _task_success_from_info(info, reward_float, terminated)
                success = bool(success or step_success)
                if step_success and success_step is None:
                    success_step = int(step)
                previous_first_action = extract_first_action(action_to_execute)
                if step_writer is not None:
                    step_writer.write(json.dumps(_jsonable({
                        "episode": episode,
                        "variant_id": variant["variant_id"],
                        "step": step,
                        "phase": "perturb",
                        "local_perturb_step": local_i,
                        "action_source": action_sources[local_i],
                        "reward": reward_float,
                        "success": step_success,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    })) + "\n")
                step += 1
                if terminated or truncated or (success and not bool(args.continue_after_success)):
                    break

            after_state = _h1_snapshot(env, "after_perturb")
            if bool(variant["reset_action_history_after_perturb"]):
                reset_count = _reset_action_sequence_history(env)
                perturb_info["reset_action_history_after_perturb_count"] = int(reset_count)
            if terminated or truncated or (success and not bool(args.continue_after_success)):
                break
            continue

        action = _act_chunk(ws, policy_obs, step, expected_shape, observation_space)
        prev_policy_obs = policy_obs
        obs, reward, terminated, truncated, info = _step_env(env, action)
        policy_obs = (
            _policy_obs_with_hidden_human_arm(
                env,
                obs,
                prev_policy_obs=prev_policy_obs,
            )
            if bool(getattr(args, "hide_human_arm_policy_obs", True))
            else obs
        )
        reward_float = float(np.asarray(reward).reshape(-1)[0])
        total_reward += reward_float
        step_success = _task_success_from_info(info, reward_float, terminated)
        success = bool(success or step_success)
        if step_success and success_step is None:
            success_step = int(step)
        previous_first_action = extract_first_action(action)
        if step_writer is not None:
            step_writer.write(json.dumps(_jsonable({
                "episode": episode,
                "variant_id": variant["variant_id"],
                "step": step,
                "phase": "act" if not perturb_done else "resume_act",
                "reward": reward_float,
                "success": step_success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })) + "\n")
        step += 1
        if terminated or truncated or (success and not bool(args.continue_after_success)):
            break

    qd_delta_norm = None
    if before_state and after_state:
        before_qd = np.asarray(before_state.get("before_perturb_qd", []), dtype=np.float64)
        after_qd = np.asarray(after_state.get("after_perturb_qd", []), dtype=np.float64)
        if before_qd.shape == after_qd.shape and before_qd.size:
            qd_delta_norm = float(np.linalg.norm(after_qd - before_qd))

    result = {
        "episode": int(episode),
        "seed": int(seed),
        "variant_id": variant["variant_id"],
        "success": bool(success),
        "success_step": success_step,
        "episode_return": float(total_reward),
        "steps_executed": int(step),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "perturb_done": bool(perturb_done),
        "wall_time_s": float(time.perf_counter() - trial_start),
        "qd_delta_norm_after_perturb": qd_delta_norm,
        "reset_info": reset_info,
        **variant,
        **perturb_info,
        **before_state,
        **after_state,
    }
    return result


def _build_variants(args: argparse.Namespace, horizon: int) -> list[dict[str, Any]]:
    scales = _parse_csv_values(args.scales, float, default=(1.0, 0.5, 0.25, 1.5, 2.0, 3.0))
    if 1.0 not in scales:
        scales = [1.0] + scales

    perturb_modes = []
    for mode in _parse_csv_values(args.perturb_modes, str, default=("magnitude",)):
        mode = str(mode).strip().lower()
        if mode not in ("magnitude", "direction"):
            raise ValueError(f"Unknown perturbation mode: {mode}")
        if mode not in perturb_modes:
            perturb_modes.append(mode)

    profiles = _parse_csv_values(args.profiles, str, default=("constant", "accelerate", "decelerate"))
    perturb_steps = _parse_csv_values(args.perturb_steps, int, default=(40, 80, 120))
    window_starts = _parse_window_starts(args.window_starts, horizon, int(args.window_len))

    direction_angle_scale = float(math.radians(float(args.direction_max_angle_deg)))
    variants: list[dict[str, Any]] = []
    for perturb_step in perturb_steps:
        for window_start in window_starts:
            for profile in profiles:
                for scale in scales:
                    for perturb_mode in perturb_modes:
                        variant_id = len(variants)
                        variants.append(
                            {
                                "variant_id": int(variant_id),
                                "perturb_step": int(perturb_step),
                                "window_start": int(window_start),
                                "window_len": int(args.window_len),
                                "profile": str(profile),
                                "scale": float(scale),
                                "perturbation_mode": str(perturb_mode),
                                "direction_axes": str(args.direction_axes),
                                "direction_angle_scale": direction_angle_scale if perturb_mode == "direction" else 0.0,
                                "execution_mode": str(args.execution_mode),
                                "waypoint_tail": str(args.waypoint_tail),
                                "control_indices": str(args.control_indices),
                                "clip_action": bool(args.clip_action),
                                "reset_action_history_after_perturb": bool(args.reset_action_history_after_perturb),
                            }
                        )
    return variants


def _summarise_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values):
        vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
        return float(np.mean(vals)) if vals else None

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            result.get("perturb_step"),
            result.get("window_start"),
            result.get("profile"),
            result.get("scale"),
            result.get("perturbation_mode"),
            result.get("execution_mode"),
        )
        grouped.setdefault(key, []).append(result)

    groups = []
    for key, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        successes = [1.0 if item.get("success") else 0.0 for item in items]
        groups.append(
            {
                "perturb_step": key[0],
                "window_start": key[1],
                "profile": key[2],
                "scale": key[3],
                "perturbation_mode": key[4],
                "execution_mode": key[5],
                "trials": len(items),
                "success_rate": mean(successes),
                "mean_return": mean(item.get("episode_return") for item in items),
                "mean_steps_executed": mean(item.get("steps_executed") for item in items),
                "mean_qd_delta_norm_after_perturb": mean(
                    item.get("qd_delta_norm_after_perturb") for item in items
                ),
                "mean_perturb_l2": mean(item.get("perturb_l2") for item in items),
                "mean_perturb_linf": mean(item.get("perturb_linf") for item in items),
                "total_clip_count": int(sum(int(item.get("clip_count") or 0) for item in items)),
            }
        )

    return {
        "num_trials": len(results),
        "success_rate": mean(1.0 if result.get("success") else 0.0 for result in results),
        "mean_return": mean(result.get("episode_return") for result in results),
        "mean_qd_delta_norm_after_perturb": mean(
            result.get("qd_delta_norm_after_perturb") for result in results
        ),
        "groups": groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", required=True, help="Eval scenario YAML to load.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to eval_safety/act_velocity_tolerance_TIMESTAMP.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes from config.")
    parser.add_argument("--steps", type=int, default=None, help="Override max steps from config.")
    parser.add_argument("--demos", type=int, default=None, help="Override demos from config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    parser.add_argument("--snapshot", default=None, help="Override ACT snapshot from config.")
    parser.add_argument("--env", default=None, help="Override env from config.")
    parser.add_argument("--perturb-steps", default="40,80,120", help="Comma-separated env steps where perturbation is injected.")
    parser.add_argument("--window-starts", default="0,4,8,12", help="Comma-separated horizon window starts, or 'all'.")
    parser.add_argument("--window-len", type=int, default=4, help="Number of consecutive nominal waypoints to perturb.")
    parser.add_argument("--scales", default="1.0,0.5,0.25,1.5,2.0,3.0", help="Comma-separated velocity/increment scales.")
    parser.add_argument(
        "--perturb-modes",
        default="magnitude",
        help="Comma-separated perturbation modes: magnitude,direction.",
    )
    parser.add_argument("--profiles", default="constant,accelerate,decelerate", help="Comma-separated profiles: constant,accelerate,decelerate,pulse.")
    parser.add_argument(
        "--direction-axes",
        default="auto",
        help="For direction perturbations, axes defining the XY rotation plane (e.g. '0,1'). Set to 'auto' to use first two control indices.",
    )
    parser.add_argument(
        "--direction-max-angle-deg",
        type=float,
        default=45.0,
        help="Maximum direction perturbation angle at scale=2.0. Angle = (scale-1.0)*max.",
    )
    parser.add_argument("--execution-mode", choices=("waypoint", "chunk"), default="waypoint", help="Execute selected waypoints over several env steps, or pass one perturbed chunk once.")
    parser.add_argument("--waypoint-tail", choices=("repeat", "nominal"), default="repeat", help="How to fill unused rows when executing a single perturbed waypoint.")
    parser.add_argument("--control-indices", default="default", help="Action dims to perturb: default, all, or comma-separated indices.")
    clip_group = parser.add_mutually_exclusive_group()
    clip_group.add_argument("--clip-action", dest="clip_action", action="store_true", help="Clip perturbed normalized actions to [-1, 1].")
    clip_group.add_argument("--no-clip-action", dest="clip_action", action="store_false", help="Do not clip perturbed normalized actions. This is the default.")
    parser.set_defaults(clip_action=False)
    policy_obs_group = parser.add_mutually_exclusive_group()
    policy_obs_group.add_argument(
        "--hide-human-arm-policy-obs",
        dest="hide_human_arm_policy_obs",
        action="store_true",
        default=True,
        help="Render clean policy observations with the human arm hidden. This is the default.",
    )
    policy_obs_group.add_argument(
        "--show-human-arm-policy-obs",
        dest="hide_human_arm_policy_obs",
        action="store_false",
        help="Let ACT see the human arm in observations. Useful only for ablations.",
    )
    parser.add_argument("--reset-action-history-after-perturb", action="store_true", help="Reset action-sequence wrapper history before handing back to ACT.")
    parser.add_argument("--max-variants", type=int, default=None, help="Optional cap for quick partial sweeps.")
    parser.add_argument("--log-steps", action="store_true", help="Write compact per-step phase logs.")
    parser.add_argument("--no-progress", action="store_true", help="Disable trial-level progress bar/prints.")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    args = _make_args(cli_args.eval_config, cli_args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.snapshot is None:
        raise ValueError("ACT snapshot is required; set eval_args.snapshot or pass --snapshot.")
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    cfg = make_cfg(args)
    runtime_cfg = cfg
    workspace_cfg = cfg
    direct_human_runtime = False
    snapshot_cfg = None
    try:
        snapshot_cfg = _load_snapshot_normalization_cfg(snapshot_path)
    except FileNotFoundError:
        snapshot_cfg = None
    if snapshot_cfg is not None:
        runtime_task = str(cfg.env.task_name)
        snapshot_task = str(snapshot_cfg.env.task_name)
        if runtime_task != snapshot_task and runtime_task.startswith("human_arm_"):
            workspace_cfg = _make_policy_env_cfg(cfg, f"bigym/{snapshot_task}")
            for key in ("manifest", "privileged_information", "require_mode_label"):
                if key in snapshot_cfg.env:
                    workspace_cfg.env[key] = snapshot_cfg.env[key]
                elif key == "manifest" and key in workspace_cfg.env:
                    workspace_cfg.env[key] = None
            direct_human_runtime = True
            args.hide_human_arm_policy_obs = True

    normalization_source, normalization_cfg = _resolve_normalization_cfg(
        args,
        workspace_cfg if direct_human_runtime else cfg,
        snapshot_path,
    )
    if direct_human_runtime and normalization_cfg is None:
        normalization_cfg = workspace_cfg
        normalization_source = f"{normalization_source}+policy_workspace"

    ws = make_workspace_and_load_snapshot(workspace_cfg, snapshot_path)
    policy_observation_space = getattr(ws.eval_env, "observation_space", None)

    env = _make_eval_env_with_normalization(runtime_cfg, normalization_cfg)
    _apply_robot_spawn_offset_xy(env, getattr(args, "robot_spawn_offset_xy", None))
    expected_shape = infer_env_action_shape(env, fallback=(int(getattr(args, "horizon", 16)), 16))
    horizon = int(expected_shape[0]) if len(expected_shape) == 2 else 1

    variants = _build_variants(cli_args, horizon)
    if cli_args.max_variants is not None:
        variants = variants[: max(0, int(cli_args.max_variants))]

    trial_path = output_dir / "act_velocity_tolerance_trials.jsonl"
    step_path = output_dir / "act_velocity_tolerance_steps.jsonl"
    summary_path = output_dir / "act_velocity_tolerance_summary.json"
    config_path = output_dir / "act_velocity_tolerance_config.json"

    config_record = {
        "eval_config": cli_args.eval_config,
        "resolved_eval_config_paths": getattr(args, "resolved_eval_config_paths", []),
        "output_dir": str(output_dir),
        "episodes": int(args.episodes),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "snapshot": str(args.snapshot),
        "env": str(args.env),
        "expected_action_shape": expected_shape,
        "runtime_task": str(runtime_cfg.env.task_name),
        "workspace_task": str(workspace_cfg.env.task_name),
        "direct_human_runtime": bool(direct_human_runtime),
        "normalization_source": str(normalization_source),
        "clip_action_default": bool(cli_args.clip_action),
        "hide_human_arm_policy_obs": bool(args.hide_human_arm_policy_obs),
        "perturb_modes": str(cli_args.perturb_modes),
        "direction_axes": str(cli_args.direction_axes),
        "direction_max_angle_deg": float(cli_args.direction_max_angle_deg),
        "variants": variants,
    }
    config_path.write_text(json.dumps(_jsonable(config_record), indent=2) + "\n")

    results: list[dict[str, Any]] = []
    with trial_path.open("w") as trial_f:
        step_f = step_path.open("w") if cli_args.log_steps else None
        try:
            for episode in range(int(args.episodes)):
                for variant in variants:
                    result = _run_one_trial(
                        env=env,
                        ws=ws,
                        args=args,
                        expected_shape=expected_shape,
                        observation_space=policy_observation_space,
                        episode=episode,
                        variant=variant,
                        step_writer=step_f,
                    )
                    results.append(result)
                    trial_f.write(json.dumps(_jsonable(result)) + "\n")
                    trial_f.flush()
        finally:
            if step_f is not None:
                step_f.close()

    summary = _summarise_results(results)
    summary.update(
        {
            "output_dir": str(output_dir),
            "trial_path": str(trial_path),
            "step_path": str(step_path) if cli_args.log_steps else None,
            "config_path": str(config_path),
        }
    )
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    try:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
