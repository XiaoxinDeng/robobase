from __future__ import annotations
from robobase.safetyfilter.safechunkdeform.stepmetrics import StepMetrics
import copy
import imageio
import json
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf
import torch
import jax.numpy as jnp

try:
    from tqdm import tqdm
except ImportError:  # tqdm may not be installed in every environment
    tqdm = None

from robobase.safetyfilter.eval_utils.eval_utils import (
    WallClockVideoRecorder,
    _render_single_env_if_vector,
    infer_env_action_shape,
    extract_first_action,
    replace_first_action,
    get_non_arm_indices,
    normalise_env_action_shape,
    make_output_paths,
    make_cfg,
    make_eval_env,
    make_workspace_and_load_snapshot,
    policy_action,
    compute_oscbf_h_monitor,
    compute_oscbf_full_arm_h_monitor,
    count_robot_human_contacts,
    robot_human_contact_pairs,
    extract_success,
    assert_action_properties,
)

from robobase.safetyfilter.eval_utils.eval_environment import (
    assert_action_properties,
    _adapt_policy_obs_to_space,
    _apply_robot_spawn_offset_xy,
    _configure_human_arm_challenge,
    _diagnostic_progress_delta,
    _diagnostic_task_state,
    _disable_human_arm_collisions,
    _enable_human_arm_collisions,
    _find_wrapped_attr,
    _find_wrapped_env_with_attr,
    _finite_task_progress,
    _freeze_human_arm,
    _hard_hold_action_from_live_robot,
    _human_arm_trajectory_sample,
    _make_policy_env_cfg,
    _normalize_h_robot_part,
    _phase_reanchor_action,
    _phase_reanchor_state,
    _policy_obs_with_hidden_human_arm,
    _post_recovery_task_guard_ready,
    _post_recovery_task_guard_reanchor_allowed,
    _raw_scaled_first_action,
    _reset_action_sequence_history,
    _reset_policy_visual_history_after_recovery,
    _seed_policy_visual_history_after_recovery,
    _restore_action_sequence_temporal_ensemble,
    _robot_ee_trajectory_sample,
    _robot_ee_world_xy,
    _robot_gripper_geom_world_xy,
    _set_action_sequence_temporal_ensemble,
    _set_robot_freeze_next_step,
    _should_start_phase_reanchor,
    _sync_animated_legs,
    _sync_named_mujoco_state,
    _sync_robot_low_level_hold_state,
    _update_scripted_human_arm_pose,
    _update_temporary_human_blocker_if_present,
)

from robobase.safetyfilter.eval_utils.eval_metrics import (
    _chunk_filter_advantage_metrics,
    _chunk_horizon_h_monitor_fallback,
    _horizon_risk_gap,
    _optional_bool,
    _optional_float,
    _optional_int,
    _optional_str,
    _path_deviation_metrics,
    summarise_all_chunk_episodes,
    summarise_chunk_episode,
)

from robobase.safetyfilter.eval_utils.eval_visualization import (
    _apply_robot_part_color_overrides,
    _clearance_sequence_payload,
    _jsonable_trace_value,
    _plot_episode_metrics,
    _restore_robot_part_color_overrides,
    _save_chunk_trajectory_viewer,
)

from robobase.safetyfilter.eval_utils.eval_config import (
    DEFAULT_EVAL_ARGS,
    _args_safety_filter,
    _flatten_eval_config_paths,
    _path_consistent_brake_eval_config,
    _path_consistent_brake_kwargs_from_config,
    _safety_filter_debug,
    _safety_filter_section,
    parse_args,
)

from robobase.safetyfilter.eval_utils.eval_video import (
    _make_eval_env_with_normalization,
    _normalize_rgb_frame,
    _policy_obs_rgb_frame,
    _print_normalization_source,
    _resolve_normalization_cfg,
    _resolve_video_stop_steps,
    _save_policy_obs_video,
    _video_duration_seconds,
    _video_recorded_steps,
    _load_snapshot_normalization_cfg
)

from robobase.safetyfilter.eval_utils.eval_runtime import (
    HorizonOSCBFOperator,
    _h_argmin_metadata,
    _warmup_oscbf_cbf_paths,
)

from robobase.safetyfilter.h1_state_bridge import (
    TREE_JOINT_NAMES,
    build_tree_to_mujoco_index_map,
    extract_h1_state,
    get_bigym_mojo,
)
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter
from robobase.safetyfilter.safechunkdeform.safechunk_deform_filter import SafeChunkDeformFilter
from robobase.safetyfilter.pacs.path_consistent_brake_filter import PathConsistentBrakeFilter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _IgnoreBigymVersionMismatchFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Installed version of bigym" not in message

logging.getLogger().addFilter(_IgnoreBigymVersionMismatchFilter())


REPO = Path("/home/xd1125/Workspace/safe_bigym_hoi")
H1_URDF = REPO / "external/oscbf/oscbf/assets/h1/h1.urdf"
_H_HUMAN_CAPSULE_PARTS = ("human_upper_arm", "human_forearm")


def _is_brake_or_fallback_execution(safety_info) -> bool:
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    if bool(_safe_info_get(safety_info, "fallback_used")):
        return True
    return mode in {
        "horizon_brake",
        "horizon_brake_intended_step",
        "path_consistent_brake",
        "path_consistent_brake_intended_step",
        "pause_on_unsafe",
        "pause_and_restart",
        "stop",
        "verified_failsafe",
        "unverified_emergency_failsafe",
    } or source in {
        "horizon_brake",
        "path_consistent_brake",
        "path_consistent_brake_slowdown",
    }

def _is_safety_intervention_mode(safety_info) -> bool:
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    if mode in {
        "horizon_brake",
        "path_consistent_brake",
        "path_consistent_brake_intended_step",
        "verified_failsafe",
        "unverified_emergency_failsafe",
        "pause_on_unsafe",
        "stop",
        "horizon_deform",
        "sequential_oscbf",
        "single_step_oscbf",
        "phase_reanchor",
    }:
        return True
    return source in {"chunk_deform", "sequential_oscbf", "sequential_oscbf_fallback"}

def _should_hold_policy_step(safety_info, first_action, safe_first_action, arm_idx, eps) -> bool:
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    if mode not in {"horizon_brake", "path_consistent_brake", "pause_on_unsafe", "stop"}:
        return False
    if bool(_safe_info_get(safety_info, "brake_hold_current")):
        return True
    first_action = np.asarray(first_action, dtype=np.float32).reshape(-1)
    safe_first_action = np.asarray(safe_first_action, dtype=np.float32).reshape(-1)
    arm_idx = np.asarray(arm_idx, dtype=np.int64)
    valid = arm_idx < min(first_action.shape[0], safe_first_action.shape[0])
    if not np.any(valid):
        return False
    delta = np.linalg.norm(safe_first_action[arm_idx[valid]] - first_action[arm_idx[valid]])
    return bool(delta > float(eps))

def _make_progress_bar(*args, **kwargs):
    if tqdm is None:
        return None
    return tqdm(
        *args,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        **kwargs,
    )

def make_oscbf_filter(args) -> OSCBFFilter:
    eager_cbf_conditions = {
        "oscbf",
        "sequential",
        "sequential_oscbf",
        "chunk_deform",
        "path_consistent_brake",
    }
    sf = _args_safety_filter(args)
    oscbf_cfg = _safety_filter_section(args, "oscbf_operator") or sf
    urdf_path = oscbf_cfg.get("urdf_path") or H1_URDF
    return OSCBFFilter(
        urdf_path=str(urdf_path),
        debug=bool(sf.get("debug", oscbf_cfg.get("debug", getattr(args, "debug", False)))),
        use_dummy_filter=bool(sf.get("use_dummy_filter", oscbf_cfg.get("use_dummy_filter", False))),
        dummy_scale=float(sf.get("dummy_scale", oscbf_cfg.get("dummy_scale", 0.5))),
        control_type=str(sf.get("control_type", oscbf_cfg.get("control_type", "absolute"))),
        max_action_delta=sf.get("max_action_delta", oscbf_cfg.get("max_action_delta")),
        human_margin=float(sf.get("human_margin", oscbf_cfg.get("human_margin", 0.08))),
        alpha_gain=float(sf.get("alpha_gain", oscbf_cfg.get("alpha_gain", 10.0))),
        pelvis_velocity_limits=sf.get(
            "pelvis_velocity_limits",
            oscbf_cfg.get(
                "pelvis_velocity_limits",
                DEFAULT_EVAL_ARGS.get("oscbf_pelvis_velocity_limits"),
            ),
        ),
        pelvis_cbf_weight=float(
            sf.get(
                "pelvis_cbf_weight",
                oscbf_cfg.get(
                    "pelvis_cbf_weight",
                    DEFAULT_EVAL_ARGS.get("oscbf_pelvis_weight", 1.0),
                ),
            )
        ),
        arm_cbf_weight=float(
            sf.get(
                "arm_cbf_weight",
                oscbf_cfg.get(
                    "arm_cbf_weight",
                    DEFAULT_EVAL_ARGS.get("oscbf_arm_weight", 1.0),
                ),
            )
        ),
        build_cbf_eagerly=args.condition in eager_cbf_conditions,
    )

def _nested_safechunk_cfg_from_eval(
    args,
    operator: HorizonOSCBFOperator,
    oscbf: Optional[OSCBFFilter] = None,
) -> dict[str, Any]:
    """Map eval configs/overrides into SafeChunkDeformFilter's nested cfg."""
    sf = _args_safety_filter(args)
    nested = copy.deepcopy(sf.get("cfg", {})) if isinstance(sf.get("cfg"), dict) else {}
    filter_cfg = nested.setdefault("safety_filter", {})
    intervention_cfg = nested.setdefault("intervention", {})
    brake_cfg = intervention_cfg.setdefault("brake", {})
    deform_cfg = intervention_cfg.setdefault("deform", {})
    recovery_cfg = intervention_cfg.setdefault("recovery", {})

    def move_flat(keys, section):
        for key in keys:
            if key not in sf:
                continue
            value = copy.deepcopy(sf[key])
            if isinstance(value, dict) and isinstance(section.get(key), dict):
                section[key] = OmegaConf.to_container(
                    OmegaConf.merge(
                        OmegaConf.create(section[key]),
                        OmegaConf.create(value),
                    ),
                    resolve=True,
                )
            else:
                section[key] = value

    move_flat(
        (
            "horizon",
            "dt",
            "action_dim",
            "expected_motion_dim",
            "control_type",
            "controlled_action_indices",
            "controlled_state_indices",
            "min_clearance",
            "diagnostics",
            "rollout_model",
            "rollout_mismatch",
            "debug",
            "enabled",
        ),
        filter_cfg,
    )
    move_flat(
        (
            "brake_progress_threshold",
            "deadlock_window",
            "temporary_blocker",
            "safechunk_active_safety",
        ),
        brake_cfg,
    )
    move_flat(
        (
            "deformation_enabled",
            "mode",
            "chunk_deformation_scales",
            "chunk_deformation_smoothing",
            "sequential_oscbf_fallback",
            "deform_after_deadlock_window",
            "unsafe_deformation_fallback",
            "optimized_fallback",
            "detach_passthrough_dims",
            "opt_iters",
            "opt_lr",
            "opt_population",
            "opt_elite_frac",
            "opt_seed",
            "lambda_safety",
            "lambda_action",
            "lambda_path",
            "lambda_smooth",
            "optimized_deform",
            "safechunk_acceptance",
            "debug_safety_feasibility",
            "action_low",
            "action_high",
            "max_action_delta",
        ),
        deform_cfg,
    )
    move_flat(
        (
            "recoverable_deform",
            "explicit_recovery",
            "safechunk_replan",
            "safechunk_recover",
            "safechunk_recovery_corridor",
            "lambda_rejoin",
            "rejoin_threshold",
            "min_rejoin_offset",
            "use_ee_pose_rejoin",
            "use_object_state_rejoin",
            "brake_if_unrecoverable",
        ),
        recovery_cfg,
    )

    controlled_action_indices = filter_cfg.get("controlled_action_indices")
    controlled_state_indices = filter_cfg.get("controlled_state_indices")
    if oscbf is not None and controlled_action_indices is None:
        controlled_action_indices = getattr(
            oscbf, "bigym_action_safety_indices", controlled_action_indices
        )
    if oscbf is not None and controlled_state_indices is None:
        controlled_state_indices = getattr(
            oscbf, "bigym_state_safety_indices", controlled_state_indices
        )
    filter_cfg["oscbf_operator"] = operator
    filter_cfg["controlled_action_indices"] = controlled_action_indices
    filter_cfg["controlled_state_indices"] = controlled_state_indices
    filter_cfg["debug"] = _safety_filter_debug(args)
    safechunk_recover_cfg = recovery_cfg.setdefault("safechunk_recover", {})
    if isinstance(safechunk_recover_cfg, dict):
        frame_stack = getattr(args, "frame_stack", None)
        if frame_stack is not None:
            safechunk_recover_cfg["act_frame_stack"] = max(1, int(frame_stack))

    deform_cfg["deformation_enabled"] = bool(deform_cfg.get("deformation_enabled", True))
    deform_cfg["sequential_oscbf_fallback"] = bool(
        deform_cfg.get("sequential_oscbf_fallback", False)
    )

    optimized_deform = copy.deepcopy(deform_cfg.get("optimized_deform", {}) or {})
    if optimized_deform.get("gradient_eps") is None:
        optimized_deform["gradient_eps"] = max(
            1e-4,
            float(deform_cfg.get("opt_lr", filter_cfg.get("opt_lr", 0.03))) * 0.5,
        )
    deform_cfg["optimized_deform"] = optimized_deform
    return nested


def make_safechunk_filter(
    args,
    operator: HorizonOSCBFOperator,
    oscbf: Optional[OSCBFFilter] = None,
) -> SafeChunkDeformFilter:
    sf = _args_safety_filter(args)
    if args.condition == "path_consistent_brake":
        controlled_action_indices = sf.get("controlled_action_indices")
        controlled_state_indices = sf.get("controlled_state_indices")
        if oscbf is not None and controlled_action_indices is None:
            controlled_action_indices = getattr(
                oscbf, "bigym_action_safety_indices", controlled_action_indices
            )
        if oscbf is not None and controlled_state_indices is None:
            controlled_state_indices = getattr(
                oscbf, "bigym_state_safety_indices", controlled_state_indices
            )
        path_consistent_brake_kwargs = _path_consistent_brake_kwargs_from_config(args, sf)
        return PathConsistentBrakeFilter(
            oscbf_operator=operator,
            horizon=int(sf.get("horizon", 16)),
            dt=float(sf.get("dt", 0.05)),
            action_dim=int(sf.get("action_dim", 16)),
            expected_motion_dim=int(sf.get("expected_motion_dim", 14)),
            control_type=str(sf.get("control_type", "absolute")),
            controlled_action_indices=controlled_action_indices,
            controlled_state_indices=controlled_state_indices,
            min_clearance=float(sf.get("min_clearance", 0.12)),
            brake_progress_threshold=float(sf.get("brake_progress_threshold", 0.05)),
            deadlock_window=int(sf.get("deadlock_window", 5)),
            deformation_enabled=False,
            mode=str(sf.get("mode", "candidate")),
            chunk_deformation_scales=sf.get("chunk_deformation_scales", [0.0, 0.25, 0.5, 0.75]),
            chunk_deformation_smoothing=int(sf.get("chunk_deformation_smoothing", 1)),
            sequential_oscbf_fallback=False,
            deform_after_deadlock_window=bool(sf.get("deform_after_deadlock_window", True)),
            unsafe_deformation_fallback=str(sf.get("unsafe_deformation_fallback", "brake")),
            opt_iters=int(sf.get("opt_iters", 20)),
            opt_lr=float(sf.get("opt_lr", 0.03)),
            lambda_safety=float(sf.get("lambda_safety", 500.0)),
            lambda_action=float(sf.get("lambda_action", 0.1)),
            lambda_path=float(sf.get("lambda_path", 0.2)),
            lambda_smooth=float(sf.get("lambda_smooth", 0.1)),
            optimized_fallback=str(sf.get("optimized_fallback", "brake")),
            detach_passthrough_dims=bool(sf.get("detach_passthrough_dims", True)),
            recoverable_deform=_safety_filter_section(args, "recoverable_deform"),
            optimized_deform=_safety_filter_section(args, "optimized_deform"),
            deform_envelope=_safety_filter_section(args, "deform_envelope"),
            explicit_recovery=_safety_filter_section(args, "explicit_recovery"),
            temporary_blocker=_safety_filter_section(args, "temporary_blocker"),
            safechunk_replan=_safety_filter_section(args, "safechunk_replan"),
            safechunk_acceptance=_safety_filter_section(args, "safechunk_acceptance"),
            safechunk_recover=_safety_filter_section(args, "safechunk_recover"),
            safechunk_active_safety=_safety_filter_section(args, "safechunk_active_safety"),
            safechunk_recovery_corridor=_safety_filter_section(args, "safechunk_recovery_corridor"),
            diagnostics=_safety_filter_section(args, "diagnostics"),
            opt_population=int(sf.get("opt_population", 32)),
            opt_elite_frac=float(sf.get("opt_elite_frac", 0.25)),
            opt_seed=sf.get("opt_seed", 0),
            max_action_delta=sf.get("max_action_delta"),
            **path_consistent_brake_kwargs,
            debug=_safety_filter_debug(args),
        )

    return SafeChunkDeformFilter(cfg=_nested_safechunk_cfg_from_eval(args, operator, oscbf=oscbf))

def _parse_h_pair_label(label: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    if not isinstance(label, str):
        return None, None, None, None
    if ":" not in label:
        return None, None, None, None
    robot_raw, human_raw = label.split(":", 1)
    robot_raw = robot_raw.strip()
    human_raw = human_raw.strip()

    if not human_raw.startswith("human_capsule_"):
        return _normalize_h_robot_part(robot_raw), None, None, None

    robot_part = _normalize_h_robot_part(robot_raw)
    if not human_raw.replace("human_capsule_", "", 1).isdigit():
        return robot_part, None, None, None
    human_capsule_index = int(human_raw.replace("human_capsule_", "", 1))
    if human_capsule_index < 0 or len(_H_HUMAN_CAPSULE_PARTS) <= 0:
        return robot_part, None, human_capsule_index, None
    human_part = _H_HUMAN_CAPSULE_PARTS[human_capsule_index % len(_H_HUMAN_CAPSULE_PARTS)]
    human_arm_index = human_capsule_index // len(_H_HUMAN_CAPSULE_PARTS)
    return robot_part, human_part, int(human_capsule_index), int(human_arm_index)

def _h_pair_label_metadata(label: Optional[str]) -> dict[str, Any]:
    robot_part, human_part, human_capsule_index, human_arm_index = _parse_h_pair_label(label)
    return {
        "h_argmin_robot_part": robot_part,
        "h_argmin_human_part": human_part,
        "h_argmin_human_capsule_index": human_capsule_index,
        "h_argmin_human_arm_index": human_arm_index,
    }


def _first_rollout_state(q_seq: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(q_seq, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(-1).copy()
    return arr.reshape(arr.shape[0], -1)[0].copy()


def _rollout_projection_for_prediction(
    q_full: Any,
    pred_q: Any,
    state_indices: Any,
) -> tuple[np.ndarray, np.ndarray, str]:
    pred = np.asarray(pred_q, dtype=np.float32).reshape(-1)
    full = np.asarray(q_full, dtype=np.float32).reshape(-1)
    idx = np.asarray(state_indices, dtype=np.int64).reshape(-1)
    if pred.size == full.size:
        return pred, full.astype(np.float32), "full_q"
    if idx.size >= pred.size and pred.size > 0:
        selected = idx[: pred.size]
        if selected.size and int(np.max(selected)) < full.size and int(np.min(selected)) >= 0:
            return pred, full[selected].astype(np.float32), "controlled_state_indices"
    n = min(pred.size, full.size)
    return pred[:n], full[:n].astype(np.float32), "prefix_fallback"


def _rollout_error_payload(
    prefix: str,
    pred_q: Any,
    actual_q_full: Any,
    state_indices: Any,
) -> dict[str, Any]:
    if pred_q is None or actual_q_full is None:
        return {
            f"{prefix}_prediction_available": False,
        }
    pred, actual, projection = _rollout_projection_for_prediction(
        actual_q_full,
        pred_q,
        state_indices,
    )
    if pred.size == 0 or actual.size == 0:
        return {
            f"{prefix}_prediction_available": False,
            f"{prefix}_projection": projection,
        }
    err = actual - pred
    base_n = min(4, err.size)
    arm_start = min(4, err.size)
    arm_err = err[arm_start:]
    out = {
        f"{prefix}_prediction_available": True,
        f"{prefix}_projection": projection,
        f"{prefix}_q_dim": int(err.size),
        f"{prefix}_pred_q_next": pred.astype(float).tolist(),
        f"{prefix}_actual_q_next": actual.astype(float).tolist(),
        f"{prefix}_q_error": err.astype(float).tolist(),
        f"{prefix}_q_l2": float(np.linalg.norm(err)),
        f"{prefix}_q_max_abs": float(np.max(np.abs(err))),
        f"{prefix}_base_l2": float(np.linalg.norm(err[:base_n])) if base_n > 0 else None,
        f"{prefix}_arm_l2": float(np.linalg.norm(arm_err)) if arm_err.size > 0 else None,
    }
    return out


def _rollout_residual_feedback(
    pred_q: Any,
    actual_q_full: Any,
    state_indices: Any,
) -> dict[str, Any] | None:
    if pred_q is None or actual_q_full is None:
        return None
    try:
        pred, actual, projection = _rollout_projection_for_prediction(
            actual_q_full,
            pred_q,
            state_indices,
        )
    except Exception:  # noqa: BLE001
        return None
    if pred.size == 0 or actual.size == 0:
        return None
    residual = actual - pred
    if residual.size == 0:
        return None
    base_n = min(4, residual.size)
    arm_residual = residual[4:] if residual.size > 4 else np.asarray([], dtype=np.float32)
    residual_l2 = float(np.linalg.norm(residual))
    residual_max_abs = float(np.max(np.abs(residual)))
    return {
        "rollout_residual_projection": projection,
        "rollout_residual_state": residual.astype(float).tolist(),
        "rollout_residual_l2": residual_l2,
        "rollout_residual_max_abs": residual_max_abs,
        "rollout_residual_base_l2": float(np.linalg.norm(residual[:base_n])) if base_n > 0 else None,
        "rollout_residual_arm_l2": float(np.linalg.norm(arm_residual)) if arm_residual.size > 0 else None,
        "rollout_prediction_untrusted": bool(residual_l2 >= 0.75 or residual_max_abs >= 0.35),
    }


def _nominal_rollout_error_summary(records: list[dict]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        out = []
        for record in records:
            value = record.get(key)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                out.append(float(value))
        return out

    summary: dict[str, Any] = {
        "nominal_rollout_diagnostic_events": int(len(records)),
    }
    for prefix in ("nominal", "safe"):
        for metric in ("q_l2", "q_max_abs", "base_l2", "arm_l2"):
            vals = values(f"{prefix}_{metric}")
            if vals:
                summary[f"nominal_rollout_diagnostic_mean_{prefix}_{metric}"] = float(np.mean(vals))
                summary[f"nominal_rollout_diagnostic_max_{prefix}_{metric}"] = float(np.max(vals))
    pass_through_vals = [
        float(record["nominal_q_l2"])
        for record in records
        if bool(record.get("act_step"))
        and isinstance(record.get("nominal_q_l2"), (int, float))
        and np.isfinite(float(record.get("nominal_q_l2")))
    ]
    if pass_through_vals:
        summary["nominal_rollout_diagnostic_mean_pass_through_nominal_q_l2"] = float(np.mean(pass_through_vals))
        summary["nominal_rollout_diagnostic_max_pass_through_nominal_q_l2"] = float(np.max(pass_through_vals))
    return summary

def _full_arm_h_pair_labels(oscbf, human_capsule_count: int) -> list[str]:
    cfg = getattr(oscbf, "oscbf_config", None)
    if cfg is None:
        return []
    if int(human_capsule_count) <= 0:
        return []

    if hasattr(cfg, "_right_arm_capsules"):
        robot_names = ("right_shoulder_upper", "right_upperarm", "right_forearm")
        has_gripper = hasattr(cfg, "_right_gripper_sphere")
    elif hasattr(cfg, "h_1"):
        robot_names = ("right_shoulder_upper", "right_upperarm", "right_forearm")
        has_gripper = False
    else:
        return []

    labels: list[str] = []
    for robot_name in robot_names:
        for h_idx in range(int(human_capsule_count)):
            labels.append(f"{robot_name}:human_capsule_{int(h_idx)}")
    if has_gripper:
        for h_idx in range(int(human_capsule_count)):
            labels.append(f"right_wrist:human_capsule_{int(h_idx)}")
    return labels

def _compute_full_arm_horizon_h_values(
    horizon_operator,
    obs,
    q_seq: np.ndarray,
) -> Optional[tuple[np.ndarray, list[str]]]:
    try:
        oscbf = horizon_operator.oscbf
        if oscbf is None or oscbf.oscbf_config is None or oscbf.robot_model is None:
            return None

        q_seq = np.asarray(q_seq, dtype=np.float32)
        if q_seq.ndim != 2 or q_seq.shape[0] == 0:
            return None
        qd_zero = np.zeros_like(q_seq[0])

        (
            capsule_a_world_seq,
            capsule_b_world_seq,
            capsule_radii_eval,
            _,
        ) = horizon_operator._human_capsule_rollout_cached(obs, q_seq.shape[0])

        human_capsule_count = int(np.asarray(capsule_radii_eval).shape[0])
        pair_labels = _full_arm_h_pair_labels(oscbf, human_capsule_count)
        if not pair_labels:
            return None

        q_urdf_seq = []
        capsule_a_urdf_seq = []
        capsule_b_urdf_seq = []

        for step_idx, q_bigym in enumerate(q_seq):
            qd_bigym = qd_zero
            q_urdf, _, _, _ = oscbf._build_urdf_surrogate_state_from_bigym(
                q_bigym,
                qd_bigym,
            )
            t_world_urdf = oscbf._get_world_T_urdf_from_bigym_state(q_bigym)
            t_urdf_world = np.linalg.inv(t_world_urdf)
            capsule_a_urdf = oscbf._transform_points(
                t_urdf_world,
                capsule_a_world_seq[step_idx],
            )
            capsule_b_urdf = oscbf._transform_points(
                t_urdf_world,
                capsule_b_world_seq[step_idx],
            )
            oscbf._validate_capsules(
                capsule_a_urdf,
                capsule_b_urdf,
                capsule_radii_eval,
            )
            q_urdf_seq.append(q_urdf)
            capsule_a_urdf_seq.append(capsule_a_urdf)
            capsule_b_urdf_seq.append(capsule_b_urdf)

        if not q_urdf_seq:
            return None

        h_values = np.asarray(
            horizon_operator._batched_h_fn(
                jnp.asarray(q_urdf_seq, dtype=jnp.float32),
                jnp.asarray(capsule_a_urdf_seq, dtype=jnp.float32),
                jnp.asarray(capsule_b_urdf_seq, dtype=jnp.float32),
                jnp.asarray(capsule_radii_eval, dtype=jnp.float32),
            ),
            dtype=np.float32,
        )
        return h_values, pair_labels
    except Exception:  # noqa: BLE001
        return None

def _parts_from_horizon_h_values(
    h_values: np.ndarray,
    h_pair_labels: Sequence[str],
) -> set[str]:
    if h_values is None:
        return set()
    values = np.asarray(h_values, dtype=np.float32)
    if values.size == 0 or values.ndim != 2:
        return set()
    if len(h_pair_labels) != values.shape[1]:
        return set()

    violating_indices = np.flatnonzero(np.any(values < 0.0, axis=0))
    parts: set[str] = set()
    for idx in violating_indices:
        robot_part, _, _, _ = _parse_h_pair_label(h_pair_labels[int(idx)])
        if robot_part is not None:
            parts.add(robot_part)
    return parts

def _controlled_pause_anchor(q_full, action_indices, state_indices, dtype=np.float32):
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    action_idx = np.asarray(action_indices, dtype=np.int64)
    state_idx = np.asarray(state_indices, dtype=np.int64)
    valid = state_idx < q.shape[0]
    action_idx = action_idx[valid]
    state_idx = state_idx[valid]
    anchor = np.zeros(action_idx.shape, dtype=dtype)
    absolute = state_idx >= 4
    if np.any(absolute):
        anchor[absolute] = q[state_idx[absolute]].astype(dtype, copy=False)
    return action_idx, anchor

def _pause_arm_at_current_q(action, q_full, action_indices, state_indices):
    safe = np.asarray(action, dtype=np.float32).copy()
    action_idx, anchor = _controlled_pause_anchor(
        q_full, action_indices, state_indices, dtype=safe.dtype
    )
    if safe.ndim == 1:
        safe[action_idx] = anchor
    elif safe.ndim == 2:
        safe[:, action_idx] = anchor[None, :]
    else:
        raise ValueError(f"Unsupported action shape for pause fallback: {safe.shape}")
    return safe

def _should_pause_for_safety(args, min_h, safety_info):
    if not args.pause_on_unsafe:
        return False, None
    threshold = float(args.pause_clearance_threshold)
    if min_h is not None and np.isfinite(float(min_h)) and float(min_h) < threshold:
        return True, "current_clearance"
    chunk_min = _safe_info_get(safety_info, "min_clearance")
    if chunk_min is not None and np.isfinite(float(chunk_min)) and float(chunk_min) < threshold:
        return True, "horizon_clearance"
    deform_safe = _safe_info_get(safety_info, "deform_safe")
    deform_min = _safe_info_get(safety_info, "deform_min_clearance")
    if (
        deform_safe is False
        and deform_min is not None
        and np.isfinite(float(deform_min))
        and float(deform_min) < threshold
    ):
        return True, "deform_clearance"
    return False, None


def _action_bridge_gripper_index(index: Any, action_dim: int) -> int | None:
    try:
        idx = int(index)
    except Exception:  # noqa: BLE001
        return None
    if idx < 0:
        idx += int(action_dim)
    if idx < 0 or idx >= int(action_dim):
        return None
    return idx


def _action_bridge_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom <= 1e-8:
        return None
    return float(np.dot(a, b) / denom)



def _ablation_action_rows(action: Any) -> np.ndarray | None:
    if action is None:
        return None
    try:
        arr = np.asarray(action, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(-1, arr.shape[-1])


def _ablation_nominal_env_action_for_sequence_step(
    target_action: Any,
    sequence_index: int,
    template_env_action: Any,
) -> np.ndarray | None:
    """Build the env action chunk that should accompany a forced nominal q frame.

    The forced-q ablation is only meaningful if the action-sequence wrapper sees
    the same temporal action context that ACT would have produced near this
    nominal window. For frame k, feed a shifted nominal action chunk starting
    from target_action[k], instead of a live hold action.
    """

    rows = _ablation_action_rows(target_action)
    if rows is None:
        return None
    try:
        template = np.asarray(template_env_action, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    if template.size == 0:
        return None
    action_dim = int(template.shape[-1])
    if rows.shape[-1] != action_dim:
        return None

    start = max(0, int(sequence_index))
    if template.ndim == 1:
        return rows[min(start, rows.shape[0] - 1)].astype(template.dtype, copy=True)

    out = template.copy()
    flat = out.reshape(-1, action_dim)
    for i in range(flat.shape[0]):
        flat[i] = rows[min(start + i, rows.shape[0] - 1)]
    return out.astype(template.dtype, copy=False)


def _seed_action_sequence_history_with_nominal_actions(
    env: Any,
    target_action: Any,
    *,
    history_window_len: int | None = None,
) -> int:
    """Seed ActionSequence temporal ensemble history with nominal ACT actions."""

    wrapper = _find_wrapped_env_with_attr(env, "_action_history")
    if wrapper is None:
        return 0
    rows = _ablation_action_rows(target_action)
    if rows is None:
        return 0
    try:
        history = getattr(wrapper, "_action_history")
        history_arr = np.asarray(history)
        cur_step = int(getattr(wrapper, "_cur_step"))
        sequence_length = int(getattr(wrapper, "_sequence_length"))
    except Exception:  # noqa: BLE001
        return 0
    if history_arr.ndim != 3 or cur_step < 0:
        return 0
    action_dim = int(history_arr.shape[-1])
    if rows.shape[-1] != action_dim:
        return 0
    slot_count = int(history_window_len) if history_window_len is not None else int(rows.shape[0])
    slot_count = max(1, min(slot_count, int(rows.shape[0]), int(history_arr.shape[0])))
    end_col = min(cur_step + sequence_length, history_arr.shape[1])
    if end_col <= cur_step:
        return 0

    # At bridge time the wrapper's _cur_step may have been reset or held.  The
    # temporal ensemble reads all nonzero rows at column _cur_step, so seed that
    # actual read column directly instead of limiting valid slots by cur_step.
    # This turns the selected nominal resume action window into the ensemble
    # context ACT will really execute on the next step.
    try:
        history_arr[:, cur_step:end_col] = 0
    except Exception:  # noqa: BLE001
        pass
    first_row = max(0, min(cur_step - slot_count + 1, history_arr.shape[0] - slot_count))
    filled = 0
    for slot in range(slot_count):
        row = first_row + slot
        if row < 0 or row >= history_arr.shape[0]:
            continue
        for col in range(cur_step, end_col):
            nominal_idx = min(col - cur_step, rows.shape[0] - 1)
            history_arr[row, col] = rows[nominal_idx].astype(history_arr.dtype, copy=False)
        filled += 1
    try:
        if not np.shares_memory(history_arr, history):
            setattr(wrapper, "_action_history", history_arr.astype(getattr(history, "dtype", history_arr.dtype)))
    except Exception:  # noqa: BLE001
        pass
    if hasattr(wrapper, "_last_smoothed_action"):
        try:
            last_idx = min(slot_count - 1, rows.shape[0] - 1)
            setattr(wrapper, "_last_smoothed_action", rows[last_idx].astype(history_arr.dtype, copy=True))
        except Exception:  # noqa: BLE001
            pass
    return int(filled)

def _seed_action_sequence_history_with_prior_act_chunk(
    env: Any,
    target_action: Any,
) -> int:
    """Seed one real, progress-validated prior ACT prediction before a fresh query."""

    wrapper = _find_wrapped_env_with_attr(env, "_action_history")
    if wrapper is None:
        return 0
    rows = _ablation_action_rows(target_action)
    if rows is None:
        return 0
    try:
        history = np.asarray(getattr(wrapper, "_action_history"))
        sequence_length = int(getattr(wrapper, "_sequence_length"))
    except Exception:  # noqa: BLE001
        return 0
    if history.ndim != 3 or history.shape[0] < 2:
        return 0
    if rows.shape[-1] != history.shape[-1]:
        return 0

    # Row 0 represents the last progress-producing ACT query. The wrapper will
    # write the current fresh ACT query to row 1, so temporal ensembling sees
    # exactly one verified prior contributor plus the live prediction.
    history[...] = 0
    cur_step = 1
    end_col = min(cur_step + sequence_length, history.shape[1])
    for col in range(cur_step, end_col):
        action_idx = min(col - cur_step, rows.shape[0] - 1)
        history[0, col] = rows[action_idx].astype(history.dtype, copy=False)
    setattr(wrapper, "_cur_step", cur_step)
    if hasattr(wrapper, "_last_smoothed_action"):
        setattr(wrapper, "_last_smoothed_action", None)
    return 1


def _temporal_action_history_stats(env: Any, resume_first_action: np.ndarray) -> dict[str, Any]:
    wrapper = _find_wrapped_env_with_attr(env, "_action_history")
    if wrapper is None:
        return {
            "action_bridge_temporal_history_slot_count": None,
            "action_bridge_temporal_history_vs_resume_l2": None,
        }
    try:
        history = np.asarray(getattr(wrapper, "_action_history"))
        cur_step = int(getattr(wrapper, "_cur_step"))
    except Exception:  # noqa: BLE001
        return {
            "action_bridge_temporal_history_slot_count": None,
            "action_bridge_temporal_history_vs_resume_l2": None,
        }
    if history.ndim != 3 or cur_step < 0 or cur_step >= history.shape[1]:
        return {
            "action_bridge_temporal_history_slot_count": 0,
            "action_bridge_temporal_history_vs_resume_l2": None,
        }
    cur_actions = np.asarray(history[:, cur_step], dtype=np.float32)
    if cur_actions.ndim != 2 or cur_actions.shape[-1] != resume_first_action.shape[-1]:
        return {
            "action_bridge_temporal_history_slot_count": 0,
            "action_bridge_temporal_history_vs_resume_l2": None,
        }
    valid = np.all(cur_actions != 0, axis=1)
    cur_actions = cur_actions[valid]
    slot_count = int(cur_actions.shape[0])
    if slot_count <= 0:
        return {
            "action_bridge_temporal_history_slot_count": 0,
            "action_bridge_temporal_history_vs_resume_l2": None,
        }
    gain = float(getattr(wrapper, "_gain", 0.0))
    weights = np.exp(-gain * np.arange(slot_count, dtype=np.float32))
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        weights = np.ones(slot_count, dtype=np.float32) / float(slot_count)
    else:
        weights = weights / weight_sum
    ensemble = np.sum(cur_actions * weights[:, None], axis=0)
    return {
        "action_bridge_temporal_history_slot_count": slot_count,
        "action_bridge_temporal_history_vs_resume_l2": float(
            np.linalg.norm(ensemble - resume_first_action)
        ),
    }


def _action_bridge_diagnostics(
    *,
    env: Any,
    last_recovery_first_action: Any,
    resume_first_action: Any,
    arm_indices: Any,
    gripper_index: Any,
) -> dict[str, Any]:
    resume = np.asarray(resume_first_action, dtype=np.float32).reshape(-1)
    out: dict[str, Any] = {
        "action_bridge_resume_first_action_norm": float(np.linalg.norm(resume)),
    }
    out.update(_temporal_action_history_stats(env, resume))
    if last_recovery_first_action is None:
        out.update(
            {
                "action_bridge_last_recovery_vs_resume_l2": None,
                "action_bridge_last_recovery_vs_resume_cosine": None,
                "action_bridge_last_recovery_arm_l2": None,
                "action_bridge_last_recovery_gripper_delta": None,
                "action_bridge_last_recovery_action_norm": None,
            }
        )
        return out
    last = np.asarray(last_recovery_first_action, dtype=np.float32).reshape(-1)
    if last.shape != resume.shape:
        out.update(
            {
                "action_bridge_last_recovery_vs_resume_l2": None,
                "action_bridge_last_recovery_vs_resume_cosine": None,
                "action_bridge_last_recovery_arm_l2": None,
                "action_bridge_last_recovery_gripper_delta": None,
                "action_bridge_last_recovery_action_norm": float(np.linalg.norm(last)),
            }
        )
        return out
    delta = resume - last
    arm_idx = np.asarray(arm_indices, dtype=np.int64).reshape(-1)
    arm_idx = arm_idx[(arm_idx >= 0) & (arm_idx < resume.shape[0])]
    grip_idx = _action_bridge_gripper_index(gripper_index, resume.shape[0])
    out.update(
        {
            "action_bridge_last_recovery_vs_resume_l2": float(np.linalg.norm(delta)),
            "action_bridge_last_recovery_vs_resume_cosine": _action_bridge_cosine(last, resume),
            "action_bridge_last_recovery_arm_l2": (
                None if arm_idx.size == 0 else float(np.linalg.norm(delta[arm_idx]))
            ),
            "action_bridge_last_recovery_gripper_delta": (
                None if grip_idx is None else float(resume[grip_idx] - last[grip_idx])
            ),
            "action_bridge_last_recovery_action_norm": float(np.linalg.norm(last)),
        }
    )
    return out


def _diagnostic_mode_flags(safety_info: dict, arm_delta: float, eps: float) -> dict[str, Any]:
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    recovery_phase = _safe_info_get(safety_info, "recovery_phase")
    fallback_step = bool(_safe_info_get(safety_info, "fallback_used"))
    committed_active = bool(_safe_info_get(safety_info, "committed_chunk_active"))
    committed_mode = _safe_info_get(safety_info, "committed_chunk_mode")
    horizon_brake_intended_step = mode == "horizon_brake_intended_step"
    path_consistent_brake_intended_step = mode == "path_consistent_brake_intended_step"
    horizon_brake_failsafe_step = mode in {
        "horizon_brake",
        "path_consistent_brake",
        "verified_failsafe",
        "unverified_emergency_failsafe",
    }
    brake_step = bool(
        horizon_brake_failsafe_step
        or mode in {"pause_on_unsafe", "pause_and_restart", "stop"}
        or (source == "horizon_brake" and not horizon_brake_intended_step)
        or (source == "path_consistent_brake" and not path_consistent_brake_intended_step)
        or _safe_info_get(safety_info, "pause_reason") is not None
    )
    optimized_attempt_step = _safe_info_get(safety_info, "optimized_accepted") is not None
    optimized_accepted_step = bool(_safe_info_get(safety_info, "optimized_accepted"))
    deform_step = bool(
        not fallback_step
        and not brake_step
        and (
            mode == "emergency_deform_away"
            or committed_mode == "horizon_deform"
            or recovery_phase == "horizon_deform"
            or (source in {"explicit_recover_deform", "explicit_return_deform"} and optimized_accepted_step)
        )
    )
    recover_step = bool(
        not fallback_step
        and not brake_step
        and (
            committed_mode == "recover"
            or recovery_phase == "recover"
        )
    )
    act_step = bool(
        horizon_brake_intended_step
        or path_consistent_brake_intended_step
        or (
            not deform_step
            and not recover_step
            and not brake_step
            and not fallback_step
            and arm_delta <= float(eps)
        )
    )
    if fallback_step:
        step_mode = "fallback"
    elif brake_step:
        step_mode = "brake"
    elif recover_step:
        step_mode = "recover"
    elif deform_step:
        step_mode = "horizon_deform"
    else:
        step_mode = "act"
    return {
        "diagnostic_step_mode": step_mode,
        "act_step": act_step,
        "deform_step": deform_step,
        "recover_step": recover_step,
        "brake_step": brake_step,
        "fallback_step": fallback_step,
        "optimized_attempt_step": bool(optimized_attempt_step),
        "optimized_accepted_step": bool(optimized_accepted_step),
        "committed_active": committed_active,
    }

def _as_chunk(action) -> tuple[np.ndarray, bool]:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        return action.reshape(1, -1), True
    if action.ndim == 2:
        return action, False
    raise ValueError(f"Unsupported action shape: {action.shape}")


def _restore_action_shape(chunk: np.ndarray, was_single: bool) -> np.ndarray:
    return chunk[0].copy() if was_single else chunk.copy()


def _safe_info_get(info: dict, key: str, default=None):
    value = info.get(key, default)
    if isinstance(value, np.generic):
        return value.item()
    return value



def _action_sequence_history_snapshot(env: Any) -> dict[str, Any]:
    wrapper = _find_wrapped_env_with_attr(env, "_action_history")
    if wrapper is None:
        return {"available": False}
    try:
        history = np.asarray(getattr(wrapper, "_action_history"), dtype=np.float32)
        cur_step = int(getattr(wrapper, "_cur_step"))
        if history.ndim != 3 or not 0 <= cur_step < history.shape[1]:
            return {"available": False}
        column = history[:, cur_step]
        smoothed = getattr(wrapper, "_last_smoothed_action", None)
        smoothed_arr = np.asarray(smoothed, dtype=np.float32).reshape(-1) if smoothed is not None else np.asarray([], dtype=np.float32)
        return {"available": True, "cur_step": cur_step, "column_sum": float(np.sum(column)), "column_l2": float(np.linalg.norm(column)), "column_nonzero_count": int(np.count_nonzero(column)), "last_smoothed_l2": float(np.linalg.norm(smoothed_arr))}
    except Exception:  # noqa: BLE001
        return {"available": False}


def _action_sequence_history_snapshot_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not (bool(before.get("available")) and bool(after.get("available"))):
        return {"mpc_handoff_scoring_history_snapshot_available": False, "mpc_handoff_scoring_history_mutated": None}
    delta = max(abs(float(after.get("column_sum", 0.0)) - float(before.get("column_sum", 0.0))), abs(float(after.get("column_l2", 0.0)) - float(before.get("column_l2", 0.0))), abs(float(after.get("last_smoothed_l2", 0.0)) - float(before.get("last_smoothed_l2", 0.0))), abs(int(after.get("column_nonzero_count", 0)) - int(before.get("column_nonzero_count", 0))), abs(int(after.get("cur_step", 0)) - int(before.get("cur_step", 0))))
    return {"mpc_handoff_scoring_history_snapshot_available": True, "mpc_handoff_scoring_history_mutated": bool(delta > 1e-6), "mpc_handoff_scoring_history_mutation_delta": float(delta), "mpc_handoff_scoring_history_before_cur_step": int(before.get("cur_step", 0)), "mpc_handoff_scoring_history_after_cur_step": int(after.get("cur_step", 0))}


def _resolve_nominal_window_source_path(source):
    if source is None:
        return None
    path = Path(str(source)).expanduser()
    if path.is_dir():
        return path / "nominal_rollout_diagnostics.jsonl"
    return path


def _load_phase_reanchor_nominal_windows(args):
    if not bool(getattr(args, "phase_reanchor_nominal_window_enabled", False)):
        return []
    source_path = _resolve_nominal_window_source_path(
        getattr(args, "phase_reanchor_nominal_window_source", None)
    )
    if source_path is None or not source_path.exists():
        return []

    run_dir = source_path.parent
    success_eps = None
    ep_summary = run_dir / "metrics_episodes.json"
    if ep_summary.exists():
        try:
            episodes = json.loads(ep_summary.read_text())
            success_eps = {
                int(e.get("episode")) for e in episodes if bool(e.get("success"))
            }
        except Exception:  # noqa: BLE001
            success_eps = None

    metric_by_ep_step = {}
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ep = int(row.get("episode", -1))
                    step = int(row.get("step", -1))
                    metric_by_ep_step[(ep, step)] = row
        except Exception:  # noqa: BLE001
            metric_by_ep_step = {}

    diag_by_ep = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ep = int(row.get("episode", -1))
            if success_eps is not None and ep not in success_eps:
                continue
            step = int(row.get("step", -1))
            q = row.get("q_before")
            act = row.get("nominal_first_action") or row.get("safe_first_action")
            if q is None:
                continue
            diag_by_ep.setdefault(ep, {})[step] = {"q": q, "action": act}

    window_len = max(1, int(getattr(args, "phase_reanchor_nominal_window_len", 4)))
    lead = max(0, int(getattr(args, "phase_reanchor_nominal_window_lead_steps", 3)))
    windows = []
    seen = set()
    for ep, rows in sorted(diag_by_ep.items()):
        if not rows:
            continue
        manipulation_steps = sorted(
            step for (m_ep, step), m in metric_by_ep_step.items()
            if m_ep == ep and m.get("interaction_context") == "manipulation_progress"
        )
        candidate_windows = []
        if manipulation_steps:
            raw_lags = getattr(args, "phase_reanchor_nominal_window_pregrasp_lags", [])
            if raw_lags is None:
                raw_lags = []
            if isinstance(raw_lags, str):
                raw_lags = [x.strip() for x in raw_lags.split(",") if x.strip()]
            pregrasp_lags = []
            for lag in raw_lags:
                try:
                    lag_i = int(lag)
                except Exception:  # noqa: BLE001
                    continue
                if lag_i >= 0:
                    pregrasp_lags.append(lag_i)
            first_manipulation_step = int(manipulation_steps[0])
            for lag_i in sorted(set(pregrasp_lags), reverse=True):
                target_end = max(0, first_manipulation_step - lag_i)
                candidate_windows.append((target_end, f"pregrasp_lag_{lag_i}_window", "pregrasp"))
            # Use multiple windows around the successful pregrasp-to-manipulation
            # transition, not only the first one.  This gives retargeting a more
            # task-useful option when q is close but EE/handle geometry is still off.
            for step in manipulation_steps[:24]:
                candidate_windows.append((max(0, int(step) - 1 + lead), "manipulation_progress_sliding_window", "manipulation"))
        else:
            scored = []
            for step in rows:
                m = metric_by_ep_step.get((ep, step), {})
                d = m.get("resume_affordance_target_distance") or m.get("ee_to_handle_dist")
                try:
                    d = float(d)
                except Exception:  # noqa: BLE001
                    d = float("inf")
                scored.append((d, step))
            if scored:
                candidate_windows.append((min(scored)[1], "min_distance_window", "unknown"))
        for target_end, selection_label, resume_stage in candidate_windows:
            steps = [step for step in range(target_end - window_len + 1, target_end + 1) if step in rows]
            if len(steps) < window_len:
                available = [step for step in sorted(rows) if step <= target_end]
                steps = available[-window_len:]
            if len(steps) < window_len:
                continue
            key = (ep, tuple(steps))
            if key in seen:
                continue
            seen.add(key)
            try:
                q_window = np.asarray([rows[step]["q"] for step in steps], dtype=np.float32)
            except Exception:  # noqa: BLE001
                continue
            action_rows = [rows[step].get("action") for step in steps]
            action_window = None
            if all(a is not None for a in action_rows):
                try:
                    action_window = np.asarray(action_rows, dtype=np.float32)
                except Exception:  # noqa: BLE001
                    action_window = None
            ref_metrics = [metric_by_ep_step.get((ep, step), {}) for step in steps]
            distances = []
            progress_steps = 0
            for m in ref_metrics:
                if m.get("interaction_context") == "manipulation_progress":
                    progress_steps += 1
                d = m.get("resume_affordance_target_distance") or m.get("ee_to_handle_dist")
                try:
                    distances.append(float(d))
                except Exception:  # noqa: BLE001
                    pass
            windows.append(
                {
                    "source": str(source_path),
                    "episode": int(ep),
                    "steps": [int(x) for x in steps],
                    "start_step": int(steps[0]),
                    "end_step": int(steps[-1]),
                    "q_window": q_window,
                    "action_window": action_window,
                    "mean_target_distance": float(np.mean(distances)) if distances else None,
                    "min_target_distance": float(np.min(distances)) if distances else None,
                    "manipulation_progress_steps": int(progress_steps),
                    "selection": str(selection_label),
                    "resume_stage": str(resume_stage),
                }
            )
    return windows


def _select_phase_reanchor_nominal_window(windows, q_full, oscbf, phase_state=None, args=None):
    if not windows:
        return None
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    state_idx = np.asarray(getattr(oscbf, "bigym_state_safety_indices", []), dtype=np.int64)
    if state_idx.size == 0:
        base = np.asarray(getattr(oscbf, "bigym_state_base_indices", []), dtype=np.int64)
        arm = np.asarray(getattr(oscbf, "bigym_state_arm_indices", []), dtype=np.int64)
        state_idx = np.concatenate([base, arm]) if base.size or arm.size else arm
    live_distance = None
    if phase_state is not None:
        try:
            live_distance = float(phase_state.get("ee_to_handle_dist"))
        except Exception:  # noqa: BLE001
            live_distance = None
    q_ok_threshold = 0.24
    distance_margin = 0.04
    selector_mode = str(getattr(args, "phase_reanchor_nominal_window_selector", "taskspace_aware"))
    requested_stage = str(getattr(args, "phase_reanchor_nominal_window_stage", "auto"))
    candidate_windows = list(windows)
    if requested_stage != "auto":
        staged = [w for w in candidate_windows if str(w.get("resume_stage", "unknown")) == requested_stage]
        if staged:
            candidate_windows = staged
    best = None
    for window in candidate_windows:
        q_window = np.asarray(window.get("q_window"), dtype=np.float32)
        if q_window.ndim != 2 or q_window.shape[0] == 0:
            continue
        valid = state_idx < min(q.shape[0], q_window.shape[1])
        if not np.any(valid):
            continue
        idx = state_idx[valid]
        err = q_window[:, idx] - q[idx].reshape(1, -1)
        l2 = float(np.min(np.linalg.norm(err, axis=1)))
        ref_distance = window.get("mean_target_distance")
        ref_distance = float(ref_distance) if isinstance(ref_distance, (int, float)) else 0.5
        progress_bonus = 0.02 * float(window.get("manipulation_progress_steps", 0) or 0)
        taskspace_gap = None
        if live_distance is not None and np.isfinite(live_distance):
            taskspace_gap = float(live_distance - ref_distance)
        retarget = bool(taskspace_gap is not None and taskspace_gap > distance_margin)
        if selector_mode == "distribution_first":
            # Distribution-first reanchor treats live task-space geometry as a
            # feasibility signal, not as the objective. ACT should regain a
            # plausible pregrasp/resume history and then perform approach/grasp
            # itself instead of being driven to the handle by the safety filter.
            stage = str(window.get("resume_stage", "unknown"))
            score = l2 + 0.05 * ref_distance
            reason = f"distribution_first_{stage}"
        elif retarget:
            # Nominal q windows are priors, not absolute targets.  When the
            # live end effector is visibly farther from the task target than a
            # candidate successful window, prefer task-space-compatible windows
            # and keep q distance as a regularizer instead of a hard precondition.
            q_penalty = 0.10 * l2 if l2 <= q_ok_threshold else 0.35 * l2
            score = ref_distance + q_penalty - progress_bonus
            reason = "taskspace_retarget"
        else:
            score = l2 + 0.25 * ref_distance - progress_bonus
            reason = "nearest_full_q"
        if best is None or score < best[0]:
            best = (score, l2, reason, window)
    if best is None:
        return None
    _, l2, reason, window = best
    selected = dict(window)
    selected["current_q_l2"] = float(l2)
    selected["selection_reason"] = reason
    selected["selector_mode"] = selector_mode
    if live_distance is not None and np.isfinite(live_distance):
        selected["live_target_distance"] = float(live_distance)
    return selected

def _phase_reanchor_live_release_status(phase_state, args):
    if not bool(getattr(args, 'phase_reanchor_release_requires_live_taskspace', True)):
        return True, 'disabled', None, None
    if not isinstance(phase_state, dict):
        return False, 'missing_phase_state', None, None
    phase = str(phase_state.get('phase', 'unknown'))
    if phase == 'done':
        return True, 'task_done', 0.0, 0.0
    try:
        handle_pos = np.asarray(phase_state.get('handle_pos'), dtype=np.float64).reshape(-1)
        ee_pos = np.asarray(phase_state.get('ee_pos'), dtype=np.float64).reshape(-1)
    except Exception:
        handle_pos = np.asarray([], dtype=np.float64)
        ee_pos = np.asarray([], dtype=np.float64)
    target_error = None
    try:
        target_error = float(phase_state.get('ee_to_target_dist'))
        if not np.isfinite(target_error):
            target_error = None
    except (TypeError, ValueError):
        target_error = None
    if target_error is None and handle_pos.size >= 2 and ee_pos.size >= 2:
        if phase == 'pull':
            offset_xy = getattr(args, 'phase_reanchor_pull_offset_xy', [0.0, -0.1])
        elif phase == 'grasp':
            offset_xy = getattr(args, 'phase_reanchor_grasp_offset_xy', [-0.03, 0.0])
        else:
            offset_xy = getattr(args, 'phase_reanchor_pregrasp_offset_xy', [-0.12, -0.06])
        offset_xy = np.asarray(offset_xy, dtype=np.float64).reshape(2)
        target_xy = handle_pos[:2] + offset_xy
        target_error = float(np.linalg.norm(target_xy - ee_pos[:2]))
    handle_dist = None
    try:
        handle_dist = float(phase_state.get('ee_to_handle_dist'))
        if not np.isfinite(handle_dist):
            handle_dist = None
    except (TypeError, ValueError):
        handle_dist = None
    if bool(phase_state.get('task_point_geometry_untrusted', False)):
        return False, 'live_taskspace_geometry_untrusted', target_error, handle_dist
    target_limit = float(getattr(args, 'phase_reanchor_live_release_target_error', 0.16))
    handle_limit = float(getattr(args, 'phase_reanchor_live_release_handle_dist', 0.24))
    target_ok = target_error is not None and np.isfinite(target_error) and target_error <= target_limit
    handle_ok = handle_dist is not None and np.isfinite(handle_dist) and handle_dist <= handle_limit
    if bool(getattr(args, "phase_reanchor_live_release_require_both", False)):
        if target_ok and handle_ok:
            return True, 'live_taskspace_ready_both', target_error, handle_dist
        if target_error is None and handle_dist is None:
            return False, 'live_taskspace_unavailable', target_error, handle_dist
        return False, 'live_taskspace_not_ready_both', target_error, handle_dist
    if target_ok or handle_ok:
        return True, 'live_taskspace_ready', target_error, handle_dist
    if target_error is None and handle_dist is None:
        return False, 'live_taskspace_unavailable', target_error, handle_dist
    return False, 'live_taskspace_not_ready', target_error, handle_dist


def _phase_reanchor_bridge_contact_status(phase_state, args):
    requires_contact = bool(getattr(args, "phase_reanchor_bridge_requires_handle_proximity", False))
    try:
        limit = float(
            getattr(
                args,
                "phase_reanchor_bridge_handle_dist",
                getattr(args, "phase_reanchor_live_release_handle_dist", 0.24),
            )
        )
    except (TypeError, ValueError):
        limit = float(getattr(args, "phase_reanchor_live_release_handle_dist", 0.24))
    handle_dist = None
    if isinstance(phase_state, dict):
        for key in ("gripper_to_handle_dist", "ee_to_handle_dist"):
            try:
                value = float(phase_state.get(key))
                if np.isfinite(value):
                    handle_dist = value
                    break
            except (TypeError, ValueError):
                pass
        if bool(phase_state.get("task_point_geometry_untrusted", False)):
            return False, "bridge_contact_geometry_untrusted", handle_dist, limit
    elif requires_contact:
        return False, "bridge_contact_missing_phase_state", None, limit
    if not requires_contact:
        return True, "bridge_contact_gate_disabled", handle_dist, limit
    if handle_dist is None:
        return False, "bridge_contact_handle_unavailable", handle_dist, limit
    if handle_dist <= limit:
        return True, "bridge_contact_ready", handle_dist, limit
    return False, "bridge_contact_not_ready", handle_dist, limit


def _act_resumable_score_terms(safety_info, args):
    if not isinstance(safety_info, dict):
        return {}

    def _finite_float(key, default=None):
        try:
            value = safety_info.get(key, default)
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def _distance_score(distance, good, scale):
        if distance is None:
            return None
        return float(np.clip(1.0 - max(0.0, distance - good) / max(scale, 1e-6), 0.0, 1.0))

    nominal_score = _finite_float("resume_affordance_component_score")
    if nominal_score is None:
        nominal_score = _finite_float("resume_affordance_score")
    nominal_min = _finite_float("resume_affordance_min_component_score", 0.25)
    if nominal_min is None:
        nominal_min = 0.25

    target_limit = float(getattr(args, "phase_reanchor_live_release_target_error", 0.16))
    handle_limit = float(getattr(args, "phase_reanchor_live_release_handle_dist", 0.24))
    target_dist = _finite_float("phase_reanchor_ee_to_target_dist")
    if target_dist is None:
        target_dist = _finite_float("phase_reanchor_live_release_target_error")
    handle_dist = _finite_float("phase_reanchor_ee_to_handle_dist")
    if handle_dist is None:
        handle_dist = _finite_float("phase_reanchor_live_release_handle_dist")

    bridge_requires_handle = bool(getattr(args, "phase_reanchor_bridge_requires_handle_proximity", False))
    bridge_handle_limit = float(getattr(args, "phase_reanchor_bridge_handle_dist", handle_limit))
    target_score = _distance_score(target_dist, target_limit, 0.45)
    handle_score = _distance_score(handle_dist, bridge_handle_limit if bridge_requires_handle else handle_limit, 0.45)
    affordance_available = bool(safety_info.get("resume_affordance_available", False))
    affordance_task_relevant = bool(safety_info.get("resume_affordance_task_relevant", False))
    affordance_ok = bool(safety_info.get("resume_affordance_ok", False))
    live_scores = [score for score in (target_score, handle_score) if score is not None]
    if bridge_requires_handle and target_score is not None and handle_score is not None:
        live_score = min(target_score, handle_score)
    else:
        live_score = max(live_scores) if live_scores else _finite_float("resume_affordance_target_distance_score")
    geometry_untrusted = bool(safety_info.get("phase_reanchor_task_point_geometry_untrusted", False))
    if geometry_untrusted and live_score is not None:
        live_score = 0.0

    nominal_ok = bool(nominal_score is not None and nominal_score >= nominal_min)
    target_ok = target_dist is not None and target_dist <= target_limit
    handle_ok = handle_dist is not None and handle_dist <= (bridge_handle_limit if bridge_requires_handle else handle_limit)
    if bridge_requires_handle:
        live_ok = bool(not geometry_untrusted and target_ok and handle_ok)
    elif target_dist is None and handle_dist is None:
        # Bigym can expose a task-specific live affordance without exposing a
        # phase-reanchor distance. Do not turn valid live geometry into OOD just
        # because the optional reanchor telemetry is unavailable.
        live_ok = bool(
            not geometry_untrusted
            and affordance_available
            and affordance_task_relevant
            and affordance_ok
        )
    else:
        live_ok = bool(not geometry_untrusted and (target_ok or handle_ok))
    score = None
    if nominal_score is not None and live_score is not None:
        score = float(min(nominal_score, live_score))

    return {
        "act_resumable_score": score,
        "act_resumable_nominal_score": nominal_score,
        "act_resumable_live_score": live_score,
        "act_resumable_nominal_ok": nominal_ok if nominal_score is not None else None,
        "act_resumable_live_ok": live_ok if live_score is not None else None,
        "act_resumable_ok": bool(nominal_ok and live_ok) if score is not None else None,
        "act_resumable_live_target_distance": target_dist,
        "act_resumable_live_handle_distance": handle_dist,
        "act_resumable_live_requires_handle_proximity": bridge_requires_handle,
        "act_resumable_live_handle_limit": bridge_handle_limit if bridge_requires_handle else handle_limit,
        "act_resumable_geometry_untrusted": geometry_untrusted,
    }

def _resume_affordance_context_from_task_state(
    task_state: Optional[dict[str, Any]],
    phase_state: Optional[dict[str, Any]],
    *,
    gripper_latched: bool = False,
    args=None,
) -> dict[str, Any]:
    """Translate eval/task diagnostics into generic SafeChunk resume features."""
    context: dict[str, Any] = {
        "resume_adapter": "bigym_task_diagnostics",
        "resume_context_source": "diagnostic_task_state",
        "resume_target_label": "interaction_target",
    }
    task_state = task_state or {}
    object_state = task_state.get("object_state")
    if isinstance(object_state, dict):
        context["resume_object_state_available"] = True
    progress = task_state.get("task_progress")
    if progress is not None:
        try:
            progress_f = float(progress)
            if np.isfinite(progress_f):
                context["resume_task_progress"] = progress_f
        except (TypeError, ValueError):
            pass
    distance = task_state.get("ee_object_distance")
    if distance is not None:
        try:
            distance_f = float(distance)
            if np.isfinite(distance_f):
                context["resume_target_distance"] = distance_f
                context["resume_target_distance_source"] = "diagnostic_task_state.ee_object_distance"
        except (TypeError, ValueError):
            pass

    if isinstance(phase_state, dict):
        context["resume_adapter"] = "bigym_phase_reanchor_adapter"
        context["resume_context_source"] = "phase_reanchor_state"
        phase = str(phase_state.get("phase", "unknown"))
        if phase == "pre_grasp":
            context["interaction_context"] = "pre_contact"
        elif phase == "grasp":
            context["interaction_context"] = "contact_rich"
        elif phase == "pull":
            context["interaction_context"] = "manipulation_progress"
        elif phase == "done":
            context["interaction_context"] = "done"
        else:
            context["interaction_context"] = phase
        distance = phase_state.get("ee_to_handle_dist")
        if distance is not None:
            try:
                distance_f = float(distance)
                if np.isfinite(distance_f):
                    context["resume_target_distance"] = distance_f
                    context["resume_target_distance_source"] = "phase_reanchor_state.ee_to_handle_dist"
            except (TypeError, ValueError):
                pass
        for source_key, context_key in (
            ("handle_pos", "resume_target_position"),
            ("ee_pos", "resume_current_ee_position"),
            ("ee_to_handle_xy", "resume_target_vector"),
        ):
            vector = phase_state.get(source_key)
            if vector is None:
                continue
            try:
                vector_arr = np.asarray(vector, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if vector_arr.size > 0 and bool(np.all(np.isfinite(vector_arr))):
                context[context_key] = [float(v) for v in vector_arr.tolist()]
                context[f"{context_key}_source"] = f"phase_reanchor_state.{source_key}"
        drawer_fraction = phase_state.get("drawer_open_fraction")
        if drawer_fraction is not None:
            try:
                progress_f = float(drawer_fraction)
                if np.isfinite(progress_f):
                    context["resume_task_progress"] = progress_f
            except (TypeError, ValueError):
                pass
        gripper_closed = bool(phase_state.get("gripper_closed", False))
        context["resume_gripper_closed"] = gripper_closed
        context["resume_target_contact"] = bool(gripper_latched or gripper_closed)

    if "interaction_context" not in context:
        progress_f = context.get("resume_task_progress")
        distance_f = context.get("resume_target_distance")
        grasp_dist = float(getattr(args, "phase_reanchor_grasp_dist", 0.12)) if args is not None else 0.12
        min_progress = float(getattr(args, "post_recovery_task_guard_min_progress", 1e-6)) if args is not None else 1e-6
        if progress_f is not None and progress_f > min_progress:
            context["interaction_context"] = "manipulation_progress"
        elif distance_f is not None and distance_f <= grasp_dist:
            context["interaction_context"] = "pre_contact"
        elif distance_f is not None:
            context["interaction_context"] = "free_motion"
        else:
            context["interaction_context"] = "unknown"
    if "resume_target_contact" not in context and gripper_latched:
        context["resume_target_contact"] = True
    return context


def _unmodelled_robot_contact_reason(contact_pairs):
    if not contact_pairs:
        return None
    unmodelled_tokens = ("head", "helmet")
    for pair in contact_pairs:
        lower = str(pair).lower()
        if any(token in lower for token in unmodelled_tokens):
            return f"unmodelled_robot_contact:{pair}"
    return None


def _chunk_obs_with_q(obs, q_full: np.ndarray):
    if isinstance(obs, dict):
        chunk_obs = dict(obs)
    else:
        chunk_obs = {"obs": obs}
    chunk_obs["q"] = np.asarray(q_full, dtype=np.float32).reshape(-1)
    return chunk_obs


def _assert_chunk_properties(nominal_chunk, safe_chunk, arm_indices):
    nominal_chunk, _ = _as_chunk(nominal_chunk)
    safe_chunk, _ = _as_chunk(safe_chunk)
    if safe_chunk.shape != nominal_chunk.shape:
        raise AssertionError(
            f"Safe chunk shape {safe_chunk.shape} != nominal chunk shape {nominal_chunk.shape}"
        )
    if not np.isfinite(safe_chunk).all():
        raise AssertionError("Safe chunk contains non-finite values")
    for k in range(nominal_chunk.shape[0]):
        assert_action_properties(nominal_chunk[k], safe_chunk[k], arm_indices)


def _sliced_safety_eval(safety_eval, start: int, end: int):
    if not isinstance(safety_eval, dict):
        return None
    sliced = dict(safety_eval)
    clearances = safety_eval.get("min_clearances")
    if clearances is None:
        return sliced
    try:
        arr = np.asarray(clearances, dtype=np.float32).reshape(-1)
        arr = arr[max(0, int(start)) : max(0, int(end))]
        sliced["min_clearances"] = arr
        if arr.size:
            sliced["min_clearance"] = float(np.min(arr))
    except Exception:  # noqa: BLE001
        pass
    return sliced


def _ee_xyz_from_q_seq(horizon_operator, q_seq):
    if horizon_operator is None or q_seq is None:
        return None
    try:
        q_seq = np.asarray(q_seq, dtype=np.float32)
        if q_seq.size == 0:
            return []
        ee_seq = horizon_operator.ee_pose_sequence(q_seq)
        if ee_seq is None:
            return None
        ee_seq = np.asarray(ee_seq, dtype=np.float32)
        if ee_seq.ndim == 3 and ee_seq.shape[-1] == 3:
            xyz = ee_seq[:, 0, :]
        elif ee_seq.ndim == 2 and ee_seq.shape[1] >= 3:
            xyz = ee_seq[:, :3]
        elif ee_seq.ndim == 1 and ee_seq.size >= 3:
            xyz = ee_seq.reshape(1, -1)[:, :3]
        else:
            return None
        return xyz.astype(float).tolist()
    except Exception as exc:  # noqa: BLE001
        logger.debug("EE trajectory extraction failed: %s", exc)
        return None


def _state_trace_payload(
    name: str,
    q_seq,
    action_chunk,
    horizon_operator,
    include_q_states: bool,
    safety_eval=None,
):
    q_arr = np.asarray(q_seq, dtype=np.float32)
    action_arr = None if action_chunk is None else np.asarray(action_chunk, dtype=np.float32)
    payload = {
        "name": name,
        "horizon": int(q_arr.shape[0]) if q_arr.ndim >= 1 else 0,
        "state_shape": list(q_arr.shape),
        "action_shape": None if action_arr is None else list(action_arr.shape),
        "ee_xyz": _ee_xyz_from_q_seq(horizon_operator, q_arr),
        "frame": "safety_model_fk_world_estimate",
        "min_clearance": (
            None
            if not isinstance(safety_eval, dict) or safety_eval.get("min_clearance") is None
            else float(safety_eval.get("min_clearance"))
        ),
        "min_clearances": _clearance_sequence_payload(
            safety_eval,
            int(q_arr.shape[0]) if q_arr.ndim >= 1 else 0,
        ),
    }
    robot_h_geometry = None
    if horizon_operator is not None:
        try:
            robot_h_geometry = horizon_operator.robot_safety_geometry_sequence(
                q_arr,
                trace_indices=(0,),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Robot h-compute trace geometry failed: %s", exc)
    if robot_h_geometry is not None:
        payload["robot_h_compute_geometry"] = _jsonable_trace_value(robot_h_geometry)
    if action_arr is not None:
        payload["action_chunk"] = action_arr.astype(float).tolist()
    if include_q_states:
        payload["q_seq"] = q_arr.astype(float).tolist()
    return payload


def _horizon_human_capsule_trace(horizon_operator, obs, horizon: int):
    if horizon_operator is None or horizon <= 0:
        return None
    try:
        human_obstacles = horizon_operator.oscbf._extract_human_obstacles(
            horizon_operator.env,
            obs,
        )
        capsule_a = np.asarray(human_obstacles["capsule_a"], dtype=np.float32)
        capsule_b = np.asarray(human_obstacles["capsule_b"], dtype=np.float32)
        capsule_radii = np.asarray(human_obstacles["capsule_radii"], dtype=np.float32)
        a_seq, b_seq, radii_eval, prediction_info = horizon_operator._human_capsule_rollout(
            capsule_a,
            capsule_b,
            capsule_radii,
            int(horizon),
        )
        centers = 0.5 * (np.asarray(a_seq, dtype=np.float32) + np.asarray(b_seq, dtype=np.float32))
        return {
            "capsule_a_world": np.asarray(a_seq, dtype=np.float32).astype(float).tolist(),
            "capsule_b_world": np.asarray(b_seq, dtype=np.float32).astype(float).tolist(),
            "capsule_centers_world": centers.astype(float).tolist(),
            "capsule_radii": np.asarray(radii_eval, dtype=np.float32).astype(float).tolist(),
            "prediction_info": _jsonable_trace_value(prediction_info),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Human capsule trajectory extraction failed: %s", exc)
        return None


def _should_log_chunk_trajectory_trace(
    args,
    safety_info: dict,
    nominal_action,
    safe_action,
    eps: float,
) -> bool:
    if not getattr(args, "log_chunk_trajectories", False):
        return False
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    committed_active = bool(_safe_info_get(safety_info, "committed_chunk_active"))
    optimized_known = _safe_info_get(safety_info, "optimized_accepted") is not None
    try:
        nominal, _ = _as_chunk(nominal_action)
        safe, _ = _as_chunk(safe_action)
        chunk_delta = float(np.linalg.norm(safe - nominal)) if safe.shape == nominal.shape else float("inf")
    except Exception:  # noqa: BLE001
        chunk_delta = 0.0
    return bool(
        committed_active
        or optimized_known
        or chunk_delta > float(eps)
        or mode in {
            "horizon_brake",
            "path_consistent_brake",
            "path_consistent_brake_slowdown",
            "path_consistent_brake_intended_step",
            "unverified_emergency_failsafe",
            "pause_on_unsafe",
            "pause_and_restart",
            "horizon_deform",
            "phase_reanchor",
            "committed_explicit_recovery",
            "recover",
            "recover_safe_prefix",
            "deform_safe_prefix",
        }
        or source in {
            "chunk_deform",
            "explicit_recover_deform",
            "explicit_return_deform",
            "committed_explicit_recovery",
            "horizon_brake",
            "path_consistent_brake",
            "path_consistent_brake_slowdown",
            "unverified_emergency_failsafe",
            "sequential_oscbf_fallback",
        }
    )


def _segment_len_from_info(safety_info, *keys, default=0):
    for key in keys:
        value = _safe_info_get(safety_info, key)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except Exception:  # noqa: BLE001
            continue
    return int(default)


def _to_flat_float_array(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    return arr


def _float_list(value) -> Optional[list[float]]:
    arr = _to_flat_float_array(value)
    if arr is None:
        return None
    return arr.astype(float).tolist()


def _diff_list(actual, expected) -> Optional[list[float]]:
    actual_arr = _to_flat_float_array(actual)
    expected_arr = _to_flat_float_array(expected)
    if actual_arr is None or expected_arr is None:
        return None
    n = min(actual_arr.size, expected_arr.size)
    if n <= 0:
        return []
    return (actual_arr[:n] - expected_arr[:n]).astype(float).tolist()


def _vector_stats(values) -> dict[str, Optional[float]]:
    arr = _to_flat_float_array(values)
    if arr is None or arr.size == 0:
        return {"l2": None, "max_abs": None, "mean_abs": None}
    abs_arr = np.abs(arr)
    return {
        "l2": float(np.linalg.norm(arr)),
        "max_abs": float(np.max(abs_arr)),
        "mean_abs": float(np.mean(abs_arr)),
    }


def _h1_q_error_group_stats(error_values) -> dict[str, dict[str, Optional[float]]]:
    arr = _to_flat_float_array(error_values)
    if arr is None:
        return {
            "base": _vector_stats(None),
            "arm": _vector_stats(None),
            "all": _vector_stats(None),
        }
    return {
        "base": _vector_stats(arr[:4]),
        "arm": _vector_stats(arr[4:14]),
        "all": _vector_stats(arr),
    }


def _named_h1_q_error_rows(actual, expected) -> list[dict[str, Any]]:
    actual_arr = _to_flat_float_array(actual)
    expected_arr = _to_flat_float_array(expected)
    if actual_arr is None or expected_arr is None:
        return []
    n = min(actual_arr.size, expected_arr.size)
    rows = []
    names = list(TREE_JOINT_NAMES)
    for i in range(n):
        error = float(actual_arr[i] - expected_arr[i])
        rows.append(
            {
                "index": int(i),
                "name": names[i] if i < len(names) else f"dim_{i}",
                "actual": float(actual_arr[i]),
                "expected": float(expected_arr[i]),
                "error": error,
                "abs_error": abs(error),
            }
        )
    return rows


def _named_action_error_rows(actual, expected) -> list[dict[str, Any]]:
    actual_arr = _to_flat_float_array(actual)
    expected_arr = _to_flat_float_array(expected)
    if actual_arr is None or expected_arr is None:
        return []
    n = min(actual_arr.size, expected_arr.size)
    rows = []
    for i in range(n):
        error = float(actual_arr[i] - expected_arr[i])
        rows.append(
            {
                "index": int(i),
                "actual": float(actual_arr[i]),
                "expected": float(expected_arr[i]),
                "error": error,
                "abs_error": abs(error),
            }
        )
    return rows


def _mujoco_state_snapshot(env) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        mojo = get_bigym_mojo(env)
        data = mojo.data
        snapshot.update(
            {
                "time": float(data.time),
                "qpos": np.asarray(data.qpos, dtype=np.float64).copy().astype(float).tolist(),
                "qvel": np.asarray(data.qvel, dtype=np.float64).copy().astype(float).tolist(),
                "ctrl": np.asarray(data.ctrl, dtype=np.float64).copy().astype(float).tolist(),
            }
        )
        try:
            snapshot["act"] = np.asarray(data.act, dtype=np.float64).copy().astype(float).tolist()
        except Exception:  # noqa: BLE001
            snapshot["act"] = None
    except Exception as exc:  # noqa: BLE001
        snapshot["mujoco_snapshot_error"] = str(exc)

    try:
        h1_state = extract_h1_state(env)
        snapshot.update(
            {
                "h1_joint_names": list(TREE_JOINT_NAMES),
                "h1_q_full": h1_state.q_full.astype(float).tolist(),
                "h1_qd_full": h1_state.qd_full.astype(float).tolist(),
                "h1_q_ctrl": h1_state.q_ctrl.astype(float).tolist(),
                "h1_qd_ctrl": h1_state.qd_ctrl.astype(float).tolist(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        snapshot["h1_snapshot_error"] = str(exc)
    return _jsonable_trace_value(snapshot)



def _mujoco_visual_pose_snapshot(env: Any) -> dict[str, Any]:
    """Snapshot world poses for visual-context bodies/sites.

    We intentionally keep this tiny: only body/site names containing wrist/head/camera/cam.
    This tells us whether ACT's wrist/head camera geometry is actually the same at
    recovery resume, without logging full RGB frames.
    """

    out: dict[str, Any] = {}
    try:
        mojo = get_bigym_mojo(env)
        model = mojo.model
        data = mojo.data
    except Exception as exc:  # noqa: BLE001
        return {"snapshot_error": str(exc)}

    try:
        import mujoco  # type: ignore
    except Exception:  # noqa: BLE001
        mujoco = None

    tokens = ("wrist", "head", "camera", "cam")

    def _name(kind: str, idx: int) -> str | None:
        if mujoco is not None:
            try:
                obj = mujoco.mjtObj.mjOBJ_BODY if kind == "body" else mujoco.mjtObj.mjOBJ_SITE
                name = mujoco.mj_id2name(model, obj, int(idx))
                if name:
                    return str(name)
            except Exception:  # noqa: BLE001
                pass
        attr = "body_names" if kind == "body" else "site_names"
        try:
            names = getattr(model, attr)
            name = names[int(idx)]
            if isinstance(name, bytes):
                return name.decode("utf-8", errors="ignore")
            return str(name)
        except Exception:  # noqa: BLE001
            return None

    def _include(name: str | None) -> bool:
        if not name:
            return False
        lower = str(name).lower()
        return any(token in lower for token in tokens)

    def _capture(kind: str, count_attr: str, pos_attr: str, mat_attr: str) -> None:
        try:
            count = int(getattr(model, count_attr))
            pos_arr = np.asarray(getattr(data, pos_attr), dtype=np.float64)
            mat_arr = np.asarray(getattr(data, mat_attr), dtype=np.float64)
        except Exception:  # noqa: BLE001
            return
        for idx in range(count):
            name = _name(kind, idx)
            if not _include(name):
                continue
            try:
                pos = pos_arr[idx].reshape(-1)[:3].astype(float).tolist()
            except Exception:  # noqa: BLE001
                pos = None
            try:
                xmat = mat_arr[idx].reshape(-1)[:9].astype(float).tolist()
            except Exception:  # noqa: BLE001
                xmat = None
            out[f"{kind}:{name}"] = {"pos": pos, "xmat": xmat}

    _capture("body", "nbody", "xpos", "xmat")
    _capture("site", "nsite", "site_xpos", "site_xmat")
    return _jsonable_trace_value(out)


def _mujoco_visual_pose_compare_metrics(
    current: Any,
    expected: Any,
    prefix: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"{prefix}_key_count": None,
        f"{prefix}_common_key_count": None,
        f"{prefix}_missing_key_count": None,
        f"{prefix}_pos_l2": None,
        f"{prefix}_pos_max_abs": None,
        f"{prefix}_pos_worst_key": None,
        f"{prefix}_pos_worst_l2": None,
        f"{prefix}_wrist_pos_l2": None,
        f"{prefix}_head_pos_l2": None,
        f"{prefix}_camera_pos_l2": None,
        f"{prefix}_xmat_l2": None,
        f"{prefix}_xmat_max_abs": None,
        f"{prefix}_xmat_worst_key": None,
        f"{prefix}_xmat_worst_l2": None,
    }
    if not isinstance(current, dict) or not isinstance(expected, dict):
        return out
    current_keys = {k for k, v in current.items() if isinstance(v, dict)}
    expected_keys = {k for k, v in expected.items() if isinstance(v, dict)}
    common = sorted(current_keys & expected_keys)
    out[f"{prefix}_key_count"] = int(len(current_keys))
    out[f"{prefix}_common_key_count"] = int(len(common))
    out[f"{prefix}_missing_key_count"] = int(len(current_keys ^ expected_keys))

    pos_sum_sq = 0.0
    pos_max_abs = 0.0
    pos_worst_key = None
    pos_worst_l2 = None
    xmat_sum_sq = 0.0
    xmat_max_abs = 0.0
    xmat_worst_key = None
    xmat_worst_l2 = None
    group_pos_sum_sq = {"wrist": 0.0, "head": 0.0, "camera": 0.0}

    for key in common:
        cur = current.get(key) or {}
        exp = expected.get(key) or {}
        for field, total_name in (("pos", "pos"), ("xmat", "xmat")):
            cur_arr = _to_flat_float_array(cur.get(field))
            exp_arr = _to_flat_float_array(exp.get(field))
            if cur_arr is None or exp_arr is None:
                continue
            n = min(cur_arr.size, exp_arr.size)
            if n <= 0:
                continue
            diff = cur_arr[:n] - exp_arr[:n]
            l2 = float(np.linalg.norm(diff))
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            if total_name == "pos":
                pos_sum_sq += float(np.dot(diff, diff))
                pos_max_abs = max(pos_max_abs, max_abs)
                if pos_worst_l2 is None or l2 > pos_worst_l2:
                    pos_worst_l2 = l2
                    pos_worst_key = str(key)
                lower = str(key).lower()
                if "wrist" in lower:
                    group_pos_sum_sq["wrist"] += float(np.dot(diff, diff))
                if "head" in lower:
                    group_pos_sum_sq["head"] += float(np.dot(diff, diff))
                if "camera" in lower or "cam" in lower:
                    group_pos_sum_sq["camera"] += float(np.dot(diff, diff))
            else:
                xmat_sum_sq += float(np.dot(diff, diff))
                xmat_max_abs = max(xmat_max_abs, max_abs)
                if xmat_worst_l2 is None or l2 > xmat_worst_l2:
                    xmat_worst_l2 = l2
                    xmat_worst_key = str(key)

    out[f"{prefix}_pos_l2"] = float(np.sqrt(pos_sum_sq)) if common else None
    out[f"{prefix}_pos_max_abs"] = float(pos_max_abs) if common else None
    out[f"{prefix}_pos_worst_key"] = pos_worst_key
    out[f"{prefix}_pos_worst_l2"] = pos_worst_l2
    out[f"{prefix}_wrist_pos_l2"] = float(np.sqrt(group_pos_sum_sq["wrist"]))
    out[f"{prefix}_head_pos_l2"] = float(np.sqrt(group_pos_sum_sq["head"]))
    out[f"{prefix}_camera_pos_l2"] = float(np.sqrt(group_pos_sum_sq["camera"]))
    out[f"{prefix}_xmat_l2"] = float(np.sqrt(xmat_sum_sq)) if common else None
    out[f"{prefix}_xmat_max_abs"] = float(xmat_max_abs) if common else None
    out[f"{prefix}_xmat_worst_key"] = xmat_worst_key
    out[f"{prefix}_xmat_worst_l2"] = xmat_worst_l2
    return out

_ABLATION_PLANNED_RECOVERY_Q_SOURCES = (
    # Eval-only ablation should test the planner's recovery/rejoin terminal
    # state, not the one-step predicted post-action state. If none of these
    # fields is present, skip the teleport so the ablation stays interpretable.
    "mpc_recovery_target_tube_terminal_q",
    "committed_rejoin_resume_tube_terminal_q",
    "recover_resume_tube_terminal_q",
)

_ABLATION_PLANNED_RECOVERY_Q_WINDOW_SOURCES = (
    "mpc_recovery_target_tube_window_q",
    "mpc_recovery_target_tube_window_q_seq",
    "mpc_recovery_target_tube_q_window",
    "committed_rejoin_resume_tube_window_q",
    "recover_resume_tube_window_q",
    "recover_resume_window_q",
)

_ABLATION_PLANNED_RECOVERY_Q_TARGET_WINDOW_SOURCES = (
    # Original nominal 4-frame window selected by the recovery resume-window
    # objective.  In source_mode=original_nominal_window, the ablation teleports
    # to the last q in this window and seeds ACT with the whole window.
    "recover_resume_window_target_q",
)

_ABLATION_PLANNED_RECOVERY_ACCEPT_KEYS = (
    "mpc_recovery_accepted",
    "committed_suffix_replan_accepted",
    "recover_accepted",
    "explicit_recovery_accepted",
    "optimized_accepted",
    "committed_released_for_act_resume",
    "resume_from_committed_rejoin",
)


def _mujoco_forward_for_ablation(mojo) -> None:
    try:
        mojo.forward()
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        import mujoco  # type: ignore

        mujoco.mj_forward(mojo.model, mojo.data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not call MuJoCo forward after ablation q set: %s", exc)


def _ablation_as_1d_q(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return None
    if arr.ndim >= 2:
        arr = arr.reshape((-1, arr.shape[-1]))[-1]
    else:
        arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _ablation_select_planned_recovery_q(
    safety_info: dict,
    source_mode: str = "planned_terminal",
) -> tuple[np.ndarray | None, str | None]:
    if not isinstance(safety_info, dict):
        return None, None
    mode = str(source_mode or "planned_terminal").strip().lower()
    source_groups = []
    if mode in {
        "original_nominal_window",
        "nominal_window",
        "target_window",
        "closest_nominal_window",
        "recover_resume_window_target_q",
    }:
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_TARGET_WINDOW_SOURCES)
    if mode in {"auto", "target_then_planned", "nominal_then_planned"}:
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_TARGET_WINDOW_SOURCES)
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_SOURCES)
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_WINDOW_SOURCES)
    if not source_groups:
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_SOURCES)
        # Planned recovery runs often expose the 4-frame resume window but not a
        # separate terminal-q field.  Use the last q in that window as the
        # terminal force target so the replay-sequence ablation can test the
        # actual optimizer output instead of silently skipping.
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_WINDOW_SOURCES)
    for sources in source_groups:
        for key in sources:
            q = _ablation_as_1d_q(_safe_info_get(safety_info, key))
            if q is not None:
                return q, key
    return None, None


def _ablation_should_force_planned_recovery_q(safety_info: dict, trigger: str) -> bool:
    trigger = str(trigger or "accepted").strip().lower()
    if trigger in {"always", "any", "any_planned", "planned"}:
        return True
    if trigger in {"accepted", "accepted_recovery", "recovery_accepted"}:
        return any(bool(_safe_info_get(safety_info, key)) for key in _ABLATION_PLANNED_RECOVERY_ACCEPT_KEYS)
    if trigger.startswith("key:"):
        return bool(_safe_info_get(safety_info, trigger.split(":", 1)[1]))
    return any(bool(_safe_info_get(safety_info, key)) for key in _ABLATION_PLANNED_RECOVERY_ACCEPT_KEYS)


def _ablation_force_q_indices(mode: str, q_dim: int) -> np.ndarray:
    mode = str(mode or "controlled").strip().lower()
    if q_dim <= 0:
        return np.zeros((0,), dtype=np.int64)
    if mode in {"controlled", "all", "full", "robot", "q", "whole_robot"}:
        start, end = 0, q_dim
    elif mode in {"base", "floating_base"}:
        start, end = 0, min(4, q_dim)
    elif mode in {"arm", "arms", "upper_body"}:
        start, end = min(4, q_dim), q_dim
    elif mode in {"left_arm", "left"}:
        start, end = min(4, q_dim), min(9, q_dim)
    elif mode in {"right_arm", "right"}:
        start, end = min(9, q_dim), q_dim
    else:
        start, end = 0, q_dim
    return np.arange(start, end, dtype=np.int64)


def _copy_policy_obs_for_ablation(obs):
    if not isinstance(obs, dict):
        return obs
    copied = {}
    for key, value in obs.items():
        try:
            copied[key] = np.asarray(value).copy()
        except Exception:  # noqa: BLE001
            copied[key] = copy.deepcopy(value)
    return copied


def _observation_snapshot_for_ablation(env, observation_space=None):
    candidates = []
    if env is not None:
        candidates.append(env)
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None and id(unwrapped) != id(env):
            candidates.append(unwrapped)
        try:
            provider = _find_wrapped_env_with_attr(env, "get_observation")
            if provider is not None and all(id(provider) != id(c) for c in candidates):
                candidates.append(provider)
        except Exception:  # noqa: BLE001
            pass

    for candidate in candidates:
        for name in ("get_observation", "_get_obs", "get_obs"):
            getter = getattr(candidate, name, None)
            if not callable(getter):
                continue
            try:
                obs = getter()
                obs = _copy_policy_obs_for_ablation(obs)
                try:
                    obs = _adapt_policy_obs_to_space(obs, observation_space)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not adapt ablation policy obs snapshot: %s", exc)
                return obs
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not collect ablation observation via %s: %s", name, exc)
                continue
    return None


def _ablation_select_planned_recovery_q_window(
    safety_info: dict,
    live_q: np.ndarray,
    target_q: np.ndarray,
    q_dim: int,
    stack_len: int,
    *,
    allow_interpolation: bool,
    source_mode: str = "planned_terminal",
) -> tuple[np.ndarray | None, str | None, bool]:
    stack_len = max(1, int(stack_len))
    mode = str(source_mode or "planned_terminal").strip().lower()
    source_groups = []
    if mode in {
        "original_nominal_window",
        "nominal_window",
        "target_window",
        "closest_nominal_window",
        "recover_resume_window_target_q",
    }:
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_TARGET_WINDOW_SOURCES)
    if mode in {"auto", "target_then_planned", "nominal_then_planned"}:
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_TARGET_WINDOW_SOURCES)
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_WINDOW_SOURCES)
    if not source_groups:
        source_groups.append(_ABLATION_PLANNED_RECOVERY_Q_WINDOW_SOURCES)
    for sources in source_groups:
        for key in sources:
            value = _safe_info_get(safety_info, key)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=np.float64)
            except Exception:  # noqa: BLE001
                continue
            if arr.ndim < 2 or arr.shape[-1] <= 0:
                continue
            arr = arr.reshape((-1, arr.shape[-1]))
            if arr.shape[0] == 0:
                continue
            full = np.repeat(target_q.reshape(1, -1), arr.shape[0], axis=0)
            cols = min(q_dim, arr.shape[-1], full.shape[-1])
            full[:, :cols] = arr[:, :cols]
            if full.shape[0] >= stack_len:
                full = full[-stack_len:]
            else:
                pad = np.repeat(full[:1], stack_len - full.shape[0], axis=0)
                full = np.concatenate([pad, full], axis=0)
            return full, key, False

    if not allow_interpolation:
        return None, None, False

    alphas = np.linspace(1.0 / float(stack_len), 1.0, stack_len, dtype=np.float64)
    window = live_q.reshape(1, -1) + alphas.reshape(-1, 1) * (target_q - live_q).reshape(1, -1)
    return window, "interpolated_live_to_terminal", True


def _collect_ablation_policy_obs_window(
    *,
    env,
    policy_env,
    safety_runtime_env,
    q_window: np.ndarray,
    indices: np.ndarray,
    zero_velocity: bool,
    qvel_window: np.ndarray | None = None,
) -> list[dict]:
    observation_space = None
    for candidate in (policy_env, env, safety_runtime_env):
        observation_space = getattr(candidate, "observation_space", None)
        if observation_space is not None:
            break

    history = []
    seen_envs: set[int] = set()
    force_envs = []
    for candidate in (env, policy_env, safety_runtime_env):
        if candidate is None or id(candidate) in seen_envs:
            continue
        seen_envs.add(id(candidate))
        force_envs.append(candidate)

    obs_env = policy_env if policy_env is not None else env
    q_arr = np.asarray(q_window, dtype=np.float64).reshape((-1, q_window.shape[-1]))
    qvel_arr = None
    if qvel_window is not None:
        qvel_arr = np.asarray(qvel_window, dtype=np.float64).reshape((-1, q_window.shape[-1]))
    for frame_idx, q_frame in enumerate(q_arr):
        frame_qvel = qvel_arr[frame_idx] if qvel_arr is not None and frame_idx < qvel_arr.shape[0] else None
        for candidate in force_envs:
            _set_h1_q_for_ablation(
                candidate, q_frame, indices, zero_velocity=zero_velocity, target_qvel=frame_qvel
            )
        obs_snapshot = _observation_snapshot_for_ablation(obs_env, observation_space)
        if isinstance(obs_snapshot, dict):
            history.append(obs_snapshot)
    return history


def _snapshot_mujoco_states_for_obs_seed(*envs) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in envs:
        if candidate is None:
            continue
        try:
            mojo = get_bigym_mojo(candidate)
            data = mojo.data
        except Exception:  # noqa: BLE001
            continue
        data_id = id(data)
        if data_id in seen:
            continue
        seen.add(data_id)
        state: dict[str, Any] = {
            "mojo": mojo,
            "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
        }
        ctrl = getattr(data, "ctrl", None)
        if ctrl is not None:
            state["ctrl"] = np.asarray(ctrl, dtype=np.float64).copy()
        snapshots.append(state)
    return snapshots


def _restore_mujoco_states_for_obs_seed(snapshots: list[dict[str, Any]]) -> int:
    restored = 0
    for state in reversed(snapshots):
        try:
            mojo = state["mojo"]
            data = mojo.data
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            if "ctrl" in state and getattr(data, "ctrl", None) is not None:
                data.ctrl[:] = state["ctrl"]
            _mujoco_forward_for_ablation(mojo)
            restored += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not restore MuJoCo state after nominal obs seed collection: %s", exc)
    return int(restored)


def _collect_policy_obs_window_preserving_state(
    *,
    env,
    policy_env,
    safety_runtime_env,
    q_window: np.ndarray,
    indices: np.ndarray,
    zero_velocity: bool,
    qvel_window: np.ndarray | None = None,
) -> tuple[list[dict], int]:
    snapshots = _snapshot_mujoco_states_for_obs_seed(env, policy_env, safety_runtime_env)
    try:
        history = _collect_ablation_policy_obs_window(
            env=env,
            policy_env=policy_env,
            safety_runtime_env=safety_runtime_env,
            q_window=q_window,
            indices=indices,
            zero_velocity=zero_velocity,
            qvel_window=qvel_window,
        )
    finally:
        restored = _restore_mujoco_states_for_obs_seed(snapshots)
    return history, restored


def _ablation_env_dt(env) -> float:
    try:
        task = _find_wrapped_env_with_attr(env, "get_dt")
        if task is not None:
            return float(task.get_dt())
    except Exception:  # noqa: BLE001
        pass
    return 0.05


def _ablation_window_qvel(q_window: np.ndarray, dt: float) -> np.ndarray | None:
    if q_window is None:
        return None
    arr = np.asarray(q_window, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    dt = float(dt) if dt else 0.05
    if dt <= 0:
        dt = 0.05
    n = arr.shape[0]
    if n == 1:
        return np.zeros_like(arr)
    qvel = np.zeros_like(arr)
    qvel[0] = (arr[1] - arr[0]) / dt
    qvel[-1] = (arr[-1] - arr[-2]) / dt
    if n > 2:
        qvel[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    return qvel


def _set_h1_q_for_ablation(
    env,
    target_q: np.ndarray,
    indices: np.ndarray,
    *,
    zero_velocity: bool,
    target_qvel: np.ndarray | None = None,
) -> bool:
    if env is None or target_q is None or indices.size == 0:
        return False
    try:
        mojo = get_bigym_mojo(env)
        model = mojo.model
        data = mojo.data
        tree_to_mj = build_tree_to_mujoco_index_map(env, TREE_JOINT_NAMES)
        updated = 0
        for raw_i in indices:
            i = int(raw_i)
            if i < 0 or i >= len(TREE_JOINT_NAMES) or i >= target_q.size:
                continue
            joint_name = TREE_JOINT_NAMES[i]
            joint_id = tree_to_mj.get(joint_name) if hasattr(tree_to_mj, "get") else tree_to_mj[joint_name]
            qpos_adr = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos_adr] = float(target_q[i])
            if target_qvel is not None and i < target_qvel.size:
                dof_adr = int(model.jnt_dofadr[joint_id])
                data.qvel[dof_adr] = float(target_qvel[i])
            elif zero_velocity:
                dof_adr = int(model.jnt_dofadr[joint_id])
                data.qvel[dof_adr] = 0.0
            updated += 1
        if updated <= 0:
            return False
        _mujoco_forward_for_ablation(mojo)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not force planned recovery q for ablation: %s", exc)
        return False


def _maybe_force_planned_recovery_q_ablation(
    *,
    args,
    env,
    policy_env,
    safety_runtime_env,
    safechunk,
    safety_info,
    safe_env_action,
) -> tuple[dict, np.ndarray, bool, list[dict] | None]:
    safety_info = dict(safety_info or {})
    if not bool(getattr(args, "ablation_force_planned_recovery_q", False)):
        return safety_info, safe_env_action, False, None

    safety_info["ablation_force_planned_recovery_q_enabled"] = True
    source_mode = str(getattr(args, "ablation_force_planned_recovery_q_source_mode", "planned_terminal"))
    window_mode = str(getattr(args, "ablation_force_planned_recovery_q_window_mode", "default"))
    safety_info["ablation_force_planned_recovery_q_source_mode"] = source_mode
    safety_info["ablation_force_planned_recovery_q_window_mode"] = window_mode
    trigger = str(getattr(args, "ablation_force_planned_recovery_q_trigger", "accepted"))
    if not _ablation_should_force_planned_recovery_q(safety_info, trigger):
        safety_info.update(
            {
                "ablation_force_planned_recovery_q_applied": False,
                "ablation_force_planned_recovery_q_skip_reason": "trigger_not_matched",
                "ablation_force_planned_recovery_q_trigger": trigger,
                "ablation_force_planned_recovery_q_source_mode": source_mode,
            }
        )
        return safety_info, safe_env_action, False, None

    planned_q, source = _ablation_select_planned_recovery_q(safety_info, source_mode)
    if planned_q is None:
        safety_info.update(
            {
                "ablation_force_planned_recovery_q_applied": False,
                "ablation_force_planned_recovery_q_skip_reason": "planned_q_missing",
                "ablation_force_planned_recovery_q_trigger": trigger,
                "ablation_force_planned_recovery_q_source_mode": source_mode,
            }
        )
        return safety_info, safe_env_action, False, None

    try:
        live_q = np.asarray(extract_h1_state(env).q_full, dtype=np.float64).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        safety_info.update(
            {
                "ablation_force_planned_recovery_q_applied": False,
                "ablation_force_planned_recovery_q_skip_reason": "live_q_unavailable",
                "ablation_force_planned_recovery_q_error": str(exc),
            }
        )
        return safety_info, safe_env_action, False, None

    q_dim = int(min(live_q.size, planned_q.size, len(TREE_JOINT_NAMES)))
    mode = str(getattr(args, "ablation_force_planned_recovery_q_mode", "controlled"))
    indices = _ablation_force_q_indices(mode, q_dim)
    if indices.size == 0:
        safety_info.update(
            {
                "ablation_force_planned_recovery_q_applied": False,
                "ablation_force_planned_recovery_q_skip_reason": "empty_force_indices",
                "ablation_force_planned_recovery_q_source": source,
                "ablation_force_planned_recovery_q_mode": mode,
            }
        )
        return safety_info, safe_env_action, False, None

    terminal_target_q = live_q.copy()
    terminal_target_q[indices] = planned_q[indices]
    replay_sequence = bool(getattr(args, "ablation_force_planned_recovery_q_replay_sequence", False))
    if window_mode == "set_state_only":
        replay_sequence = False
    should_prepare_window = bool(getattr(args, "ablation_force_planned_recovery_q_seed_policy_window", False)) or replay_sequence
    q_window = None
    qvel_window = None
    window_source = None
    window_interpolated = False
    mean_step_l2 = None
    max_step_l2 = None
    qvel_dt = None
    mean_qvel_l2 = None
    max_qvel_l2 = None
    stack_len = max(1, int(getattr(args, "ablation_force_planned_recovery_q_window_len", 4)))
    if should_prepare_window:
        q_window, window_source, window_interpolated = _ablation_select_planned_recovery_q_window(
            safety_info,
            live_q,
            terminal_target_q,
            q_dim,
            stack_len,
            allow_interpolation=bool(
                getattr(args, "ablation_force_planned_recovery_q_window_interpolate", True)
            ),
            source_mode=source_mode,
        )
        if q_window is not None and q_window.shape[0] > 1:
            step_delta = np.diff(q_window[:, indices], axis=0)
            step_l2 = np.linalg.norm(step_delta, axis=1)
            mean_step_l2 = float(np.mean(step_l2))
            max_step_l2 = float(np.max(step_l2))
        if q_window is not None and bool(
            getattr(args, "ablation_force_planned_recovery_q_seed_window_velocity", True)
        ):
            qvel_dt = _ablation_env_dt(env)
            qvel_window = _ablation_window_qvel(q_window, qvel_dt)
            if qvel_window is not None:
                try:
                    qvel_l2 = np.linalg.norm(np.asarray(qvel_window, dtype=np.float64)[:, indices], axis=1)
                    mean_qvel_l2 = float(np.mean(qvel_l2))
                    max_qvel_l2 = float(np.max(qvel_l2))
                except Exception:
                    mean_qvel_l2 = None
                    max_qvel_l2 = None

    target_q = terminal_target_q.copy()
    target_qvel = None
    if replay_sequence and q_window is not None and q_window.shape[0] > 0:
        target_q = live_q.copy()
        cols = min(target_q.shape[0], q_window.shape[-1])
        valid_indices = indices[indices < cols]
        target_q[valid_indices] = np.asarray(q_window[0], dtype=np.float64)[valid_indices]
        if qvel_window is not None:
            target_qvel = np.zeros_like(target_q)
            target_qvel[valid_indices] = np.asarray(qvel_window[0], dtype=np.float64)[valid_indices]
    delta = target_q - live_q
    forced_env_count = 0
    seen_envs: set[int] = set()
    force_envs = []
    for candidate in (env, policy_env, safety_runtime_env):
        if candidate is None or id(candidate) in seen_envs:
            continue
        seen_envs.add(id(candidate))
        force_envs.append(candidate)
    zero_velocity = bool(getattr(args, "ablation_force_planned_recovery_q_zero_velocity", True))
    for candidate in force_envs:
        if _set_h1_q_for_ablation(
            candidate, target_q, indices, zero_velocity=zero_velocity, target_qvel=target_qvel
        ):
            forced_env_count += 1

    if forced_env_count <= 0:
        safety_info.update(
            {
                "ablation_force_planned_recovery_q_applied": False,
                "ablation_force_planned_recovery_q_skip_reason": "mujoco_set_failed",
                "ablation_force_planned_recovery_q_source": source,
                "ablation_force_planned_recovery_q_mode": mode,
            }
        )
        return safety_info, safe_env_action, False, None

    ablation_policy_obs_history = None
    if should_prepare_window:
        window_obs_count = 0
        if (
            q_window is not None
            and bool(getattr(args, "ablation_force_planned_recovery_q_seed_policy_window", False))
            and not replay_sequence
        ):
            ablation_policy_obs_history = _collect_ablation_policy_obs_window(
                env=env,
                policy_env=policy_env,
                safety_runtime_env=safety_runtime_env,
                q_window=q_window,
                indices=indices,
                zero_velocity=zero_velocity,
                qvel_window=qvel_window,
            )
            window_obs_count = len(ablation_policy_obs_history)
        safety_info.update(
            {
                "ablation_force_planned_recovery_q_seed_window_enabled": bool(
                    getattr(args, "ablation_force_planned_recovery_q_seed_policy_window", False)
                ),
                "ablation_force_planned_recovery_q_window_mode": window_mode,
                "ablation_force_planned_recovery_q_window_source": window_source,
                "ablation_force_planned_recovery_q_window_interpolated": bool(window_interpolated),
                "ablation_force_planned_recovery_q_window_len": int(q_window.shape[0]) if q_window is not None else 0,
                "ablation_force_planned_recovery_q_window_obs_count": int(window_obs_count),
                "ablation_force_planned_recovery_q_window_step_l2_mean": mean_step_l2,
                "ablation_force_planned_recovery_q_window_step_l2_max": max_step_l2,
                "ablation_force_planned_recovery_q_window_qvel_enabled": qvel_window is not None,
                "ablation_force_planned_recovery_q_window_qvel_dt": qvel_dt,
                "ablation_force_planned_recovery_q_window_qvel_l2_mean": mean_qvel_l2,
                "ablation_force_planned_recovery_q_window_qvel_l2_max": max_qvel_l2,
                "ablation_force_planned_recovery_q_replay_sequence_enabled": bool(replay_sequence),
                "ablation_force_planned_recovery_q_replay_sequence_source": window_source,
                "ablation_force_planned_recovery_q_replay_sequence_len": int(q_window.shape[0]) if q_window is not None else 0,
                "ablation_force_planned_recovery_q_replay_sequence_q": (
                    np.asarray(q_window, dtype=np.float64).astype(float).tolist()
                    if replay_sequence and q_window is not None
                    else None
                ),
                "ablation_force_planned_recovery_q_replay_sequence_qvel": (
                    np.asarray(qvel_window, dtype=np.float64).astype(float).tolist()
                    if replay_sequence and qvel_window is not None
                    else None
                ),
            }
        )

    reset_history_count = 0
    if bool(getattr(args, "ablation_force_planned_recovery_q_reset_history", False)):
        reset_seen: set[int] = set()
        for candidate in (env, policy_env):
            if candidate is None or id(candidate) in reset_seen:
                continue
            reset_seen.add(id(candidate))
            try:
                reset_history_count += int(_reset_action_sequence_history(candidate))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not reset action-sequence history for ablation: %s", exc)

    filter_reset = False
    if bool(getattr(args, "ablation_force_planned_recovery_q_reset_filter", True)) and hasattr(safechunk, "reset"):
        try:
            safechunk.reset()
            filter_reset = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not reset SafeChunk after ablation q force: %s", exc)

    sync_low_level_state_count = 0
    if bool(getattr(args, "ablation_force_planned_recovery_q_sync_low_level_state", False)):
        sync_seen: set[int] = set()
        for candidate in (env, policy_env, safety_runtime_env):
            if candidate is None or id(candidate) in sync_seen:
                continue
            sync_seen.add(id(candidate))
            try:
                sync_low_level_state_count += int(_sync_robot_low_level_hold_state(candidate))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not sync low-level state for ablation teleport: %s", exc)

    hold_delta = None
    hold_indices = None
    sequence_action_source = None
    sequence_action_index = None
    sequence_nominal_action_used = False
    if bool(getattr(args, "ablation_force_planned_recovery_q_hold_current_step", True)):
        target_action = _safe_info_get(safety_info, "recover_resume_window_target_action")
        nominal_env_action = _ablation_nominal_env_action_for_sequence_step(
            target_action,
            0,
            safe_env_action,
        )
        if nominal_env_action is not None:
            safe_env_action = nominal_env_action
            sequence_action_source = "recover_resume_window_target_action"
            sequence_action_index = 0
            sequence_nominal_action_used = True
        else:
            try:
                safe_env_action, hold_indices, hold_delta = _hard_hold_action_from_live_robot(env, safe_env_action)
                sequence_action_source = "live_hold"
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not switch current ablation step to live hold action: %s", exc)

    arm_start = min(4, delta.size)
    base_end = min(4, delta.size)
    safety_info.update(
        {
            "ablation_force_planned_recovery_q_applied": True,
            "ablation_force_planned_recovery_q_skip_reason": None,
            "ablation_force_planned_recovery_q_trigger": trigger,
            "ablation_force_planned_recovery_q_source": source,
            "ablation_force_planned_recovery_q_source_mode": source_mode,
            "ablation_force_planned_recovery_q_window_mode": window_mode,
            "ablation_force_planned_recovery_q_mode": mode,
            "ablation_force_planned_recovery_q_indices": indices.astype(int).tolist(),
            "ablation_force_planned_recovery_q_sequence_nominal_action_used": bool(sequence_nominal_action_used),
            "ablation_force_planned_recovery_q_sequence_action_source": sequence_action_source,
            "ablation_force_planned_recovery_q_sequence_action_index": sequence_action_index,
            "ablation_force_planned_recovery_q_dim": int(q_dim),
            "ablation_force_planned_recovery_q_l2_from_pre": float(np.linalg.norm(delta[indices])),
            "ablation_force_planned_recovery_q_max_abs_from_pre": float(np.max(np.abs(delta[indices]))) if indices.size else 0.0,
            "ablation_force_planned_recovery_q_arm_l2_from_pre": float(np.linalg.norm(delta[arm_start:])),
            "ablation_force_planned_recovery_q_base_l2_from_pre": float(np.linalg.norm(delta[:base_end])),
            "ablation_force_planned_recovery_q_forced_env_count": int(forced_env_count),
            "ablation_force_planned_recovery_q_reset_history_count": int(reset_history_count),
            "ablation_force_planned_recovery_q_reset_filter": bool(filter_reset),
            "ablation_force_planned_recovery_q_sync_low_level_state_count": int(sync_low_level_state_count),
            "ablation_force_planned_recovery_q_zero_velocity": bool(zero_velocity),
            "ablation_force_planned_recovery_q_hold_current_step": bool(getattr(args, "ablation_force_planned_recovery_q_hold_current_step", True)),
            "ablation_force_planned_recovery_q_hold_delta": float(hold_delta) if hold_delta is not None else None,
            "ablation_force_planned_recovery_q_hold_indices": hold_indices,
        }
    )
    return safety_info, safe_env_action, True, ablation_policy_obs_history



def _vector_compare_metrics(lhs, rhs, prefix: str) -> dict[str, Any]:
    try:
        lhs_arr = np.asarray(lhs, dtype=np.float64).reshape(-1)
        rhs_arr = np.asarray(rhs, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return {}
    if lhs_arr.size == 0 or rhs_arr.size == 0:
        return {}
    n = min(lhs_arr.size, rhs_arr.size)
    lhs_arr = lhs_arr[:n]
    rhs_arr = rhs_arr[:n]
    diff = lhs_arr - rhs_arr
    lhs_norm = float(np.linalg.norm(lhs_arr))
    rhs_norm = float(np.linalg.norm(rhs_arr))
    denom = lhs_norm * rhs_norm
    cosine = None if denom <= 1e-12 else float(np.dot(lhs_arr, rhs_arr) / denom)
    return {
        f"{prefix}_l2": float(np.linalg.norm(diff)),
        f"{prefix}_max_abs": float(np.max(np.abs(diff))),
        f"{prefix}_cosine": cosine,
        f"{prefix}_dim": int(n),
    }




def _action_agreement_metrics(
    lhs,
    rhs,
    prefix: str,
    *,
    arm_indices=None,
    gripper_index=None,
) -> dict[str, Any]:
    out = _vector_compare_metrics(lhs, rhs, prefix)
    try:
        lhs_arr = np.asarray(lhs, dtype=np.float64).reshape(-1)
        rhs_arr = np.asarray(rhs, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return out
    if lhs_arr.size == 0 or rhs_arr.size == 0:
        return out
    n = min(lhs_arr.size, rhs_arr.size)
    lhs_arr = lhs_arr[:n]
    rhs_arr = rhs_arr[:n]
    if arm_indices is not None:
        try:
            arm_idx = np.asarray(arm_indices, dtype=np.int64).reshape(-1)
            arm_idx = np.where(arm_idx < 0, arm_idx + n, arm_idx)
            arm_idx = arm_idx[(arm_idx >= 0) & (arm_idx < n)]
            if arm_idx.size:
                arm_delta = lhs_arr[arm_idx] - rhs_arr[arm_idx]
                out.update(
                    {
                        f"{prefix}_arm_l2": float(np.linalg.norm(arm_delta)),
                        f"{prefix}_arm_max_abs": float(np.max(np.abs(arm_delta))),
                        f"{prefix}_arm_dim": int(arm_idx.size),
                    }
                )
        except Exception:  # noqa: BLE001
            pass
    if gripper_index is not None:
        try:
            grip_idx = int(gripper_index)
            if grip_idx < 0:
                grip_idx += n
            if 0 <= grip_idx < n:
                delta = float(lhs_arr[grip_idx] - rhs_arr[grip_idx])
                out.update(
                    {
                        f"{prefix}_gripper_delta": delta,
                        f"{prefix}_gripper_abs_delta": abs(delta),
                    }
                )
        except Exception:  # noqa: BLE001
            pass
    return out




def _numeric_array_or_none(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return None
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
        return None
    try:
        return arr.astype(np.float64, copy=False)
    except Exception:  # noqa: BLE001
        return None


def _looks_like_image_obs_key(key: Any, arr: np.ndarray) -> bool:
    key_s = str(key).lower()
    if any(token in key_s for token in ("rgb", "image", "camera", "pixels", "front", "wrist", "head")):
        return True
    return bool(arr.ndim >= 3)


def _policy_obs_snapshot_compare_metrics(current: Any, expected: Any, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"{prefix}_key_count": None,
        f"{prefix}_common_key_count": None,
        f"{prefix}_missing_key_count": None,
        f"{prefix}_shape_mismatch_key_count": None,
        f"{prefix}_numeric_key_count": None,
        f"{prefix}_numeric_mismatch_key_count": None,
        f"{prefix}_numeric_l2": None,
        f"{prefix}_numeric_mean_abs": None,
        f"{prefix}_numeric_max_abs": None,
        f"{prefix}_numeric_worst_key": None,
        f"{prefix}_numeric_worst_l2": None,
        f"{prefix}_image_key_count": None,
        f"{prefix}_image_mismatch_key_count": None,
        f"{prefix}_image_l2": None,
        f"{prefix}_image_mean_abs": None,
        f"{prefix}_image_max_abs": None,
        f"{prefix}_image_worst_key": None,
        f"{prefix}_image_worst_l2": None,
    }
    if not isinstance(current, dict) or not isinstance(expected, dict):
        return out

    current_keys = set(current.keys())
    expected_keys = set(expected.keys())
    common_keys = sorted(current_keys & expected_keys, key=str)
    out[f"{prefix}_key_count"] = int(len(current_keys))
    out[f"{prefix}_common_key_count"] = int(len(common_keys))
    out[f"{prefix}_missing_key_count"] = int(len((current_keys ^ expected_keys)))

    numeric_sum_sq = 0.0
    numeric_sum_abs = 0.0
    numeric_count = 0
    numeric_keys = 0
    numeric_mismatch_keys = 0
    numeric_max_abs = 0.0
    numeric_worst_key = None
    numeric_worst_l2 = None

    image_sum_sq = 0.0
    image_sum_abs = 0.0
    image_count = 0
    image_keys = 0
    image_mismatch_keys = 0
    image_max_abs = 0.0
    image_worst_key = None
    image_worst_l2 = None
    shape_mismatch = 0

    for key in common_keys:
        cur = _numeric_array_or_none(current.get(key))
        exp = _numeric_array_or_none(expected.get(key))
        if cur is None or exp is None:
            continue
        numeric_keys += 1
        if cur.shape != exp.shape:
            shape_mismatch += 1
        n = min(cur.size, exp.size)
        if n <= 0:
            continue
        cur_flat = cur.reshape(-1)[:n]
        exp_flat = exp.reshape(-1)[:n]
        diff = cur_flat - exp_flat
        key_l2 = float(np.linalg.norm(diff))
        key_abs_sum = float(np.sum(np.abs(diff)))
        key_max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        if key_l2 > 1e-9 or key_max_abs > 1e-9 or cur.shape != exp.shape:
            numeric_mismatch_keys += 1
        numeric_sum_sq += float(np.dot(diff, diff))
        numeric_sum_abs += key_abs_sum
        numeric_count += int(n)
        numeric_max_abs = max(numeric_max_abs, key_max_abs)
        if numeric_worst_l2 is None or key_l2 > numeric_worst_l2:
            numeric_worst_l2 = key_l2
            numeric_worst_key = str(key)

        if _looks_like_image_obs_key(key, cur) or _looks_like_image_obs_key(key, exp):
            image_keys += 1
            if key_l2 > 1e-9 or key_max_abs > 1e-9 or cur.shape != exp.shape:
                image_mismatch_keys += 1
            image_sum_sq += float(np.dot(diff, diff))
            image_sum_abs += key_abs_sum
            image_count += int(n)
            image_max_abs = max(image_max_abs, key_max_abs)
            if image_worst_l2 is None or key_l2 > image_worst_l2:
                image_worst_l2 = key_l2
                image_worst_key = str(key)

    out[f"{prefix}_shape_mismatch_key_count"] = int(shape_mismatch)
    out[f"{prefix}_numeric_key_count"] = int(numeric_keys)
    out[f"{prefix}_numeric_mismatch_key_count"] = int(numeric_mismatch_keys)
    if numeric_count > 0:
        out[f"{prefix}_numeric_l2"] = float(np.sqrt(numeric_sum_sq))
        out[f"{prefix}_numeric_mean_abs"] = float(numeric_sum_abs / float(numeric_count))
        out[f"{prefix}_numeric_max_abs"] = float(numeric_max_abs)
        out[f"{prefix}_numeric_worst_key"] = numeric_worst_key
        out[f"{prefix}_numeric_worst_l2"] = numeric_worst_l2
    out[f"{prefix}_image_key_count"] = int(image_keys)
    out[f"{prefix}_image_mismatch_key_count"] = int(image_mismatch_keys)
    if image_count > 0:
        out[f"{prefix}_image_l2"] = float(np.sqrt(image_sum_sq))
        out[f"{prefix}_image_mean_abs"] = float(image_sum_abs / float(image_count))
        out[f"{prefix}_image_max_abs"] = float(image_max_abs)
        out[f"{prefix}_image_worst_key"] = image_worst_key
        out[f"{prefix}_image_worst_l2"] = image_worst_l2
    return out

def _target_action_window_diagnostics(
    target_action: Any,
    predicted_first_action: Any,
) -> dict[str, Any]:
    rows = _ablation_action_rows(target_action)
    if rows is None:
        return {
            "act_resume_diag_target_action_rows": None,
            "act_resume_diag_target_action_dim": None,
            "act_resume_diag_target_first_action_norm": None,
            "act_resume_diag_predicted_first_action_norm": None,
            "act_resume_diag_target_window_best_index": None,
            "act_resume_diag_target_window_best_l2": None,
            "act_resume_diag_target_window_l2_0": None,
            "act_resume_diag_target_window_l2_1": None,
            "act_resume_diag_target_window_l2_2": None,
            "act_resume_diag_target_window_l2_3": None,
        }
    try:
        pred = np.asarray(predicted_first_action, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        pred = np.empty((0,), dtype=np.float64)
    action_dim = int(rows.shape[-1]) if rows.ndim == 2 else 0
    out: dict[str, Any] = {
        "act_resume_diag_target_action_rows": int(rows.shape[0]),
        "act_resume_diag_target_action_dim": int(action_dim),
        "act_resume_diag_target_first_action_norm": float(np.linalg.norm(rows[0])) if rows.shape[0] else None,
        "act_resume_diag_predicted_first_action_norm": float(np.linalg.norm(pred)) if pred.size else None,
        "act_resume_diag_target_window_best_index": None,
        "act_resume_diag_target_window_best_l2": None,
        "act_resume_diag_target_window_l2_0": None,
        "act_resume_diag_target_window_l2_1": None,
        "act_resume_diag_target_window_l2_2": None,
        "act_resume_diag_target_window_l2_3": None,
    }
    if pred.size == 0 or action_dim <= 0:
        return out
    n = min(int(pred.size), action_dim)
    pred_n = pred[:n]
    best_idx = None
    best_l2 = None
    for idx, row in enumerate(rows):
        row_n = np.asarray(row, dtype=np.float64).reshape(-1)[:n]
        l2 = float(np.linalg.norm(pred_n - row_n))
        if idx < 4:
            out[f"act_resume_diag_target_window_l2_{idx}"] = l2
        if best_l2 is None or l2 < best_l2:
            best_l2 = l2
            best_idx = idx
    out["act_resume_diag_target_window_best_index"] = int(best_idx) if best_idx is not None else None
    out["act_resume_diag_target_window_best_l2"] = best_l2
    return out

def _first_action_or_none(action) -> np.ndarray | None:
    if action is None:
        return None
    try:
        return np.asarray(extract_first_action(action), dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        try:
            chunk, _ = _as_chunk(action)
            if len(chunk) == 0:
                return None
            return np.asarray(chunk[0], dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            return None

def _should_log_mpc_replay_diagnostic(safety_info: dict) -> bool:
    if not isinstance(safety_info, dict):
        return False
    diagnostic_keys = (
        "planned_pre_action_q",
        "planned_post_action_q",
        "planned_action_at_index",
        "actual_pre_action_q",
        "replay_predicted_post_action_q",
        "committed_action",
    )
    if any(_safe_info_get(safety_info, key) is not None for key in diagnostic_keys):
        return True
    return bool(
        _safe_info_get(safety_info, "mpc_recovery_replan_attempted")
        or _safe_info_get(safety_info, "mpc_recovery_accepted")
        or _safe_info_get(safety_info, "committed_chunk_active")
        or _safe_info_get(safety_info, "committed_suffix_replan_attempted")
        or _safe_info_get(safety_info, "resume_from_committed_rejoin")
    )


def _collect_mpc_replay_diagnostic(
    *,
    episode: int,
    step: int,
    safety_info: dict,
    safe_env_action,
    pre_mujoco_snapshot: Optional[dict[str, Any]],
    post_mujoco_snapshot: Optional[dict[str, Any]],
    reward: Optional[float],
    terminated: Optional[bool],
    truncated: Optional[bool],
) -> Optional[dict[str, Any]]:
    if not _should_log_mpc_replay_diagnostic(safety_info):
        return None

    planned_pre_q = _safe_info_get(safety_info, "planned_pre_action_q")
    planned_post_q = _safe_info_get(safety_info, "planned_post_action_q")
    replay_pre_q = _safe_info_get(safety_info, "actual_pre_action_q")
    replay_post_q = _safe_info_get(safety_info, "replay_predicted_post_action_q")
    planned_action = _safe_info_get(safety_info, "planned_action_at_index")
    committed_action = _safe_info_get(safety_info, "committed_action")
    executed_first_action = extract_first_action(safe_env_action)

    actual_pre_q = None
    if isinstance(pre_mujoco_snapshot, dict):
        actual_pre_q = pre_mujoco_snapshot.get("h1_q_full")
    if actual_pre_q is None:
        actual_pre_q = replay_pre_q

    actual_post_q = None
    if isinstance(post_mujoco_snapshot, dict):
        actual_post_q = post_mujoco_snapshot.get("h1_q_full")

    actual_pre_minus_planned_pre = _diff_list(actual_pre_q, planned_pre_q)
    replay_pre_minus_planned_pre = _diff_list(replay_pre_q, planned_pre_q)
    actual_post_minus_planned_post = _diff_list(actual_post_q, planned_post_q)
    replay_post_minus_planned_post = _diff_list(replay_post_q, planned_post_q)
    actual_post_minus_replay_post = _diff_list(actual_post_q, replay_post_q)
    committed_minus_planned_action = _diff_list(committed_action, planned_action)
    executed_minus_planned_action = _diff_list(executed_first_action, planned_action)

    return _jsonable_trace_value(
        {
            "episode": int(episode),
            "step": int(step),
            "mode": _safe_info_get(safety_info, "mode"),
            "safety_mode": _safe_info_get(safety_info, "safety_mode"),
            "deformation_source": _safe_info_get(safety_info, "deformation_source"),
            "committed_chunk_active": _safe_info_get(safety_info, "committed_chunk_active"),
            "committed_chunk_mode": _safe_info_get(safety_info, "committed_chunk_mode"),
            "committed_chunk_index": _safe_info_get(safety_info, "committed_chunk_index"),
            "committed_chunk_length": _safe_info_get(safety_info, "committed_chunk_length"),
            "mpc_recovery_replan_attempted": _safe_info_get(safety_info, "mpc_recovery_replan_attempted"),
            "mpc_recovery_accepted": _safe_info_get(safety_info, "mpc_recovery_accepted"),
            "mpc_recovery_reject_reason": _safe_info_get(safety_info, "mpc_recovery_reject_reason"),
            "committed_suffix_replan_attempted": _safe_info_get(safety_info, "committed_suffix_replan_attempted"),
            "committed_suffix_replan_accepted": _safe_info_get(safety_info, "committed_suffix_replan_accepted"),
            "committed_suffix_replan_rejected": _safe_info_get(safety_info, "committed_suffix_replan_rejected"),
            "committed_suffix_replan_reject_reason": _safe_info_get(safety_info, "committed_suffix_replan_reject_reason"),
            "control_type": _safe_info_get(safety_info, "control_type"),
            "dt": _safe_info_get(safety_info, "dt"),
            "controlled_state_indices": _safe_info_get(safety_info, "controlled_state_indices"),
            "controlled_action_indices": _safe_info_get(safety_info, "controlled_action_indices"),
            "action_conversion_mode": _safe_info_get(safety_info, "action_conversion_mode"),
            "reward": None if reward is None else float(reward),
            "terminated": None if terminated is None else bool(terminated),
            "truncated": None if truncated is None else bool(truncated),
            "h1_joint_names": list(TREE_JOINT_NAMES),
            "planned_pre_action_q": _float_list(planned_pre_q),
            "planned_post_action_q": _float_list(planned_post_q),
            "safety_filter_actual_pre_action_q": _float_list(replay_pre_q),
            "safety_filter_replay_predicted_post_action_q": _float_list(replay_post_q),
            "bigym_actual_pre_action_q": _float_list(actual_pre_q),
            "bigym_actual_post_action_q": _float_list(actual_post_q),
            "planned_action_at_index": _float_list(planned_action),
            "committed_action": _float_list(committed_action),
            "executed_first_action": _float_list(executed_first_action),
            "actual_pre_minus_planned_pre_q": actual_pre_minus_planned_pre,
            "replay_pre_minus_planned_pre_q": replay_pre_minus_planned_pre,
            "actual_post_minus_planned_post_q": actual_post_minus_planned_post,
            "replay_post_minus_planned_post_q": replay_post_minus_planned_post,
            "actual_post_minus_replay_predicted_post_q": actual_post_minus_replay_post,
            "committed_minus_planned_action": committed_minus_planned_action,
            "executed_minus_planned_action": executed_minus_planned_action,
            "actual_pre_minus_planned_pre_q_stats": _h1_q_error_group_stats(actual_pre_minus_planned_pre),
            "actual_post_minus_planned_post_q_stats": _h1_q_error_group_stats(actual_post_minus_planned_post),
            "replay_post_minus_planned_post_q_stats": _h1_q_error_group_stats(replay_post_minus_planned_post),
            "actual_post_minus_replay_predicted_post_q_stats": _h1_q_error_group_stats(actual_post_minus_replay_post),
            "committed_minus_planned_action_stats": _vector_stats(committed_minus_planned_action),
            "executed_minus_planned_action_stats": _vector_stats(executed_minus_planned_action),
            "actual_post_vs_planned_post_by_joint": _named_h1_q_error_rows(actual_post_q, planned_post_q),
            "actual_post_vs_replay_predicted_post_by_joint": _named_h1_q_error_rows(actual_post_q, replay_post_q),
            "actual_pre_vs_planned_pre_by_joint": _named_h1_q_error_rows(actual_pre_q, planned_pre_q),
            "executed_vs_planned_action_by_dim": _named_action_error_rows(executed_first_action, planned_action),
            "planned_clearance_pre": _safe_info_get(safety_info, "planned_clearance_pre"),
            "planned_clearance_post": _safe_info_get(safety_info, "planned_clearance_post"),
            "replay_clearance_pre": _safe_info_get(safety_info, "replay_clearance_pre"),
            "replay_clearance_post": _safe_info_get(safety_info, "replay_clearance_post"),
            "actual_vs_planned_pre_q_error": _safe_info_get(safety_info, "actual_vs_planned_pre_q_error"),
            "actual_vs_planned_post_q_error": _safe_info_get(safety_info, "actual_vs_planned_post_q_error"),
            "planning_vs_replay_clearance_pre_error": _safe_info_get(safety_info, "planning_vs_replay_clearance_pre_error"),
            "planning_vs_replay_clearance_post_error": _safe_info_get(safety_info, "planning_vs_replay_clearance_post_error"),
            "planning_human_state_snapshot": _safe_info_get(safety_info, "planning_human_state_snapshot"),
            "replay_human_state": _safe_info_get(safety_info, "replay_human_state"),
            "pre_mujoco_state": pre_mujoco_snapshot,
            "post_mujoco_state": post_mujoco_snapshot,
        }
    )


def _mpc_replay_error_summary(records: Sequence[dict]) -> dict[str, Optional[float]]:
    post_l2 = []
    post_max_abs = []
    model_l2 = []
    model_max_abs = []
    base_l2 = []
    arm_l2 = []
    for record in records:
        post_stats = (record.get("actual_post_minus_planned_post_q_stats") or {}).get("all") or {}
        model_stats = (record.get("actual_post_minus_replay_predicted_post_q_stats") or {}).get("all") or {}
        base_stats = (record.get("actual_post_minus_planned_post_q_stats") or {}).get("base") or {}
        arm_stats = (record.get("actual_post_minus_planned_post_q_stats") or {}).get("arm") or {}
        if post_stats.get("l2") is not None:
            post_l2.append(float(post_stats["l2"]))
        if post_stats.get("max_abs") is not None:
            post_max_abs.append(float(post_stats["max_abs"]))
        if model_stats.get("l2") is not None:
            model_l2.append(float(model_stats["l2"]))
        if model_stats.get("max_abs") is not None:
            model_max_abs.append(float(model_stats["max_abs"]))
        if base_stats.get("l2") is not None:
            base_l2.append(float(base_stats["l2"]))
        if arm_stats.get("l2") is not None:
            arm_l2.append(float(arm_stats["l2"]))

    def _mean(values):
        return float(np.mean(values)) if values else None

    def _max(values):
        return float(np.max(values)) if values else None

    return {
        "mpc_replay_diagnostic_mean_actual_post_vs_planned_post_l2": _mean(post_l2),
        "mpc_replay_diagnostic_max_actual_post_vs_planned_post_l2": _max(post_l2),
        "mpc_replay_diagnostic_mean_actual_post_vs_planned_post_max_abs": _mean(post_max_abs),
        "mpc_replay_diagnostic_max_actual_post_vs_planned_post_max_abs": _max(post_max_abs),
        "mpc_replay_diagnostic_mean_actual_post_vs_replay_post_l2": _mean(model_l2),
        "mpc_replay_diagnostic_max_actual_post_vs_replay_post_l2": _max(model_l2),
        "mpc_replay_diagnostic_mean_actual_post_vs_replay_post_max_abs": _mean(model_max_abs),
        "mpc_replay_diagnostic_max_actual_post_vs_replay_post_max_abs": _max(model_max_abs),
        "mpc_replay_diagnostic_mean_base_l2": _mean(base_l2),
        "mpc_replay_diagnostic_mean_arm_l2": _mean(arm_l2),
    }


def _collect_chunk_trajectory_trace(
    *,
    args,
    episode: int,
    step: int,
    safechunk,
    horizon_operator,
    obs,
    nominal_chunk,
    generated_chunk,
    safety_info: dict,
    human_sample=None,
    policy_anchor_sample=None,
):
    try:
        nominal, _ = _as_chunk(nominal_chunk)
        generated, _ = _as_chunk(generated_chunk)
        nominal_q = safechunk.intervention.rollout_nominal_chunk(obs, nominal)
        nominal_eval = safechunk.intervention.evaluate_horizon_safety(obs, nominal_q)
        generated_q = safechunk.intervention.rollout_nominal_chunk(obs, generated)
        generated_eval = safechunk.intervention.evaluate_horizon_safety(obs, generated_q)

        # The controller executes only the first row of each ACT chunk before
        # replanning. Keep trajectory trace payloads aligned with that
        # receding-horizon execution contract; full-horizon rollouts are used
        # only internally for safety evaluation above.
        nominal_exec = nominal[:1].copy()
        generated_exec = generated[:1].copy()
        nominal_exec_q = safechunk.intervention.rollout_nominal_chunk(obs, nominal_exec)
        generated_exec_q = safechunk.intervention.rollout_nominal_chunk(obs, generated_exec)
        nominal_exec_eval = _sliced_safety_eval(nominal_eval, 0, 1)
        generated_exec_eval = _sliced_safety_eval(generated_eval, 0, 1)

        retiming_source = (
            _safe_info_get(safety_info, "retiming_source")
            or _safe_info_get(safety_info, "deformation_source")
            or _safe_info_get(safety_info, "safety_mode")
            or _safe_info_get(safety_info, "mode")
        )
        if retiming_source in {
            "path_consistent_brake",
            "path_consistent_brake_slowdown",
            "unverified_emergency_failsafe",
        }:
            braked_chunk = generated.copy()
            braked_q = generated_q
            braked_exec = generated_exec.copy()
            braked_exec_q = generated_exec_q
            brake_eval = generated_eval
            brake_exec_eval = generated_exec_eval
            brake_info = {
                "brake_trace_source": "generated_safe_chunk",
                "brake_trace_reason": str(retiming_source),
                "brake_safe": bool(generated_eval.get("horizon_safe", False))
                if isinstance(generated_eval, dict)
                else None,
                "brake_min_clearance": generated_eval.get("min_clearance")
                if isinstance(generated_eval, dict)
                else None,
            }
        else:
            braked_chunk, brake_info = safechunk.brake.horizon_brake(
                obs,
                nominal,
                nominal_eval,
            )
            braked_q = safechunk.intervention.rollout_nominal_chunk(obs, braked_chunk)
            brake_eval = safechunk.intervention.evaluate_horizon_safety(obs, braked_q)
            braked_exec = braked_chunk[:1].copy()
            braked_exec_q = safechunk.intervention.rollout_nominal_chunk(obs, braked_exec)
            brake_exec_eval = _sliced_safety_eval(brake_eval, 0, 1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Chunk trajectory trace collection failed: %s", exc)
        return None

    include_q = bool(getattr(args, "chunk_trajectory_include_q_states", True))
    traces = {
        "nominal": _state_trace_payload(
            "nominal",
            nominal_exec_q,
            nominal_exec,
            horizon_operator,
            include_q,
            nominal_exec_eval,
        ),
        "braking": _state_trace_payload(
            "braking",
            braked_exec_q,
            braked_exec,
            horizon_operator,
            include_q,
            brake_exec_eval,
        ),
        "generated": _state_trace_payload(
            "generated",
            generated_exec_q,
            generated_exec,
            horizon_operator,
            include_q,
            generated_exec_eval,
        ),
    }

    executed_policy_sample = None
    generated_payload = traces.get("generated", {})
    generated_ee = generated_payload.get("ee_xyz") if isinstance(generated_payload, dict) else None
    if generated_ee:
        executed_policy_sample = {
            "episode": int(episode),
            "step": int(step + 1),
            "planning_step": int(step),
            "horizon_index": 0,
            "source": "transformed_safe_action_sequence",
            "ee_pos": _jsonable_trace_value(generated_ee[0]),
            "action": _jsonable_trace_value(generated_exec[0]) if generated_exec.shape[0] else None,
        }
        if include_q and isinstance(generated_payload, dict):
            q_seq_payload = generated_payload.get("q_seq")
            if q_seq_payload:
                executed_policy_sample["q"] = _jsonable_trace_value(q_seq_payload[0])

    planned_total = int(generated.shape[0]) if generated.ndim == 2 else 0
    total = int(generated_exec.shape[0]) if generated_exec.ndim == 2 else 0
    deform_len = _segment_len_from_info(
        safety_info,
        "deform_chunk_length",
        default=0,
    )
    recover_len = _segment_len_from_info(
        safety_info,
        "recover_chunk_length",
        "return_chunk_length",
        default=0,
    )
    committed_mode = _safe_info_get(safety_info, "committed_chunk_mode")
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    if deform_len <= 0 and total > 0 and committed_mode == "horizon_deform":
        deform_len = total
    elif deform_len <= 0 and recover_len <= 0 and total > 0 and (
        mode in {"horizon_deform", "deform_safe_prefix"}
        or source in {"chunk_deform", "explicit_recover_deform", "explicit_return_deform"}
    ):
        deform_len = total
    if recover_len <= 0 and total > 0 and committed_mode == "recover":
        recover_len = total

    deform_len = max(0, min(deform_len, total))
    recover_start = deform_len
    if recover_start >= total and committed_mode == "recover":
        recover_start = 0
    recover_len = max(0, min(recover_len, total - recover_start))

    if deform_len > 0:
        traces["deformed"] = _state_trace_payload(
            "deformed",
            np.asarray(generated_exec_q, dtype=np.float32)[:deform_len],
            generated_exec[:deform_len],
            horizon_operator,
            include_q,
            _sliced_safety_eval(generated_exec_eval, 0, deform_len),
        )
    if recover_len > 0:
        end = recover_start + recover_len
        traces["recovery"] = _state_trace_payload(
            "recovery",
            np.asarray(generated_exec_q, dtype=np.float32)[recover_start:end],
            generated_exec[recover_start:end],
            horizon_operator,
            include_q,
            _sliced_safety_eval(generated_exec_eval, recover_start, end),
        )

    horizon = max(
        int(np.asarray(nominal_exec_q).shape[0]),
        int(np.asarray(generated_exec_q).shape[0]),
    )
    record = {
        "episode": int(episode),
        "step": int(step),
        "condition": getattr(args, "condition", None),
        "safety_mode": mode,
        "deform_mode": _safe_info_get(safety_info, "deform_mode"),
        "recovery_phase": _safe_info_get(safety_info, "recovery_phase"),
        "deformation_source": source,
        "accepted_path_name": _safe_info_get(
                safety_info,
                "accepted_path_name",
                _safe_info_get(safety_info, "accepted_candidate_name"),
            ),
        "accepted_path_type": _safe_info_get(
                safety_info,
                "accepted_path_type",
                _safe_info_get(safety_info, "accepted_candidate_type"),
            ),
        "optimized_accepted": _safe_info_get(safety_info, "optimized_accepted"),
        "fallback_used": _safe_info_get(safety_info, "fallback_used"),
        "first_violation": _safe_info_get(safety_info, "first_violation"),
        "brake_info": _jsonable_trace_value(brake_info),
        "segment_lengths": {
            "deform": int(deform_len),
            "recover": int(recover_len),
            "recover_start": int(recover_start),
            "total": int(total),
            "planned_total": int(planned_total),
        },
        "executed_action_horizon": int(total),
        "planned_action_horizon": int(planned_total),
        "controlled_state_indices": _jsonable_trace_value(
            getattr(safechunk, "controlled_state_indices", None)
        ),
        "controlled_action_indices": _jsonable_trace_value(
            getattr(safechunk, "controlled_action_indices", None)
        ),
        "traces": traces,
        "human_arm_sample": _jsonable_trace_value(human_sample),
        "policy_anchor_sample": _jsonable_trace_value(policy_anchor_sample),
        "executed_policy_sample": _jsonable_trace_value(executed_policy_sample),
        "human_capsule_prediction": _horizon_human_capsule_trace(
            horizon_operator,
            obs,
            horizon,
        ),
    }
    return _jsonable_trace_value(record)


def _execution_mode_from_safety_info(safety_info: dict) -> str:
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    diagnostic_mode = _safe_info_get(safety_info, "diagnostic_step_mode")
    recovery_phase = _safe_info_get(safety_info, "recovery_phase")
    committed_mode = _safe_info_get(safety_info, "committed_chunk_mode")
    if diagnostic_mode == "recover" or bool(_safe_info_get(safety_info, "recover_step")):
        return "recover"
    if diagnostic_mode in {"brake", "fallback"} or bool(_safe_info_get(safety_info, "brake_step")):
        return "braking"
    if diagnostic_mode == "horizon_deform" or bool(_safe_info_get(safety_info, "deform_step")):
        return "deform"
    if diagnostic_mode == "act" or bool(_safe_info_get(safety_info, "act_step")):
        return "policy"
    if recovery_phase == "recover" or committed_mode == "recover":
        return "recover"
    if recovery_phase == "horizon_deform" or committed_mode == "horizon_deform":
        return "deform"
    if mode in {None, "pass_through", "path_consistent_brake_intended_step"}:
        return "policy"
    if mode in {"horizon_brake", "unverified_emergency_failsafe", "pause_on_unsafe", "pause_and_restart", "stop"}:
        return "braking"
    if source in {"horizon_brake", "path_consistent_brake", "path_consistent_brake_slowdown"}:
        return "braking"
    if mode in {"recover", "recover_safe_prefix", "committed_explicit_recovery"}:
        return "recover"
    if mode in {"horizon_deform", "chunk_deform", "deform_safe_prefix", "emergency_deform_away"} or source in {"chunk_deform", "explicit_recover_deform", "explicit_return_deform"}:
        return "deform"
    if mode in {"phase_reanchor", "sequential_oscbf"}:
        return str(mode)
    return "intervention"


def _execution_group_from_mode(mode: str) -> str:
    return "policy" if str(mode) in {"policy", "pass_through", "initial"} else "intervention"


def _annotate_executed_trajectory_sample(sample: dict, safety_info: dict | None, *, initial: bool = False):
    if sample is None:
        return None
    info = safety_info if isinstance(safety_info, dict) else {}
    mode = "policy" if initial else _execution_mode_from_safety_info(info)
    sample.update(
        {
            "execution_mode": mode,
            "execution_group": _execution_group_from_mode(mode),
            "safety_mode": "initial" if initial else _safe_info_get(info, "safety_mode"),
            "deformation_source": None if initial else _safe_info_get(info, "deformation_source"),
            "diagnostic_step_mode": None if initial else _safe_info_get(info, "diagnostic_step_mode"),
            "recovery_phase": None if initial else _safe_info_get(info, "recovery_phase"),
            "committed_chunk_mode": None if initial else _safe_info_get(info, "committed_chunk_mode"),
            "deform_mode": None if initial else _safe_info_get(info, "deform_mode"),
            "intervention_active": False if initial else _is_safety_intervention_mode(info),
            "brake_step": False if initial else mode == "braking",
            "deform_step": False if initial else mode == "deform",
            "recover_step": False if initial else mode == "recover",
        }
    )
    return sample


def main():
    args = parse_args()
    if args.eval_config:
        print("eval_config:", _flatten_eval_config_paths(args.eval_config))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        if (
            args.policy_env is None
            and args.safety_env is None
            and runtime_task != snapshot_task
            and runtime_task.startswith("human_arm_")
        ):
            workspace_cfg = _make_policy_env_cfg(cfg, f"bigym/{snapshot_task}")
            for key in ("manifest", "privileged_information", "require_mode_label"):
                if key in snapshot_cfg.env:
                    workspace_cfg.env[key] = snapshot_cfg.env[key]
                elif key == "manifest" and key in workspace_cfg.env:
                    workspace_cfg.env[key] = None
            direct_human_runtime = True
            if not args.hide_human_arm_policy_obs:
                args.hide_human_arm_policy_obs = True
                print("direct_human_runtime: hiding human arm from policy observations")
            print("direct_human_runtime: using checkpoint task for policy workspace")
            print("policy_workspace_task:", workspace_cfg.env.task_name)
            print("runtime_task:", runtime_cfg.env.task_name)

    normalization_source, normalization_cfg = _resolve_normalization_cfg(
        args,
        workspace_cfg if direct_human_runtime else cfg,
        snapshot_path,
    )
    if direct_human_runtime and normalization_cfg is None:
        normalization_cfg = workspace_cfg
        normalization_source = f"{normalization_source}+policy_workspace"

    print("\n=== Normalization stats source ===")
    _print_normalization_source(
        normalization_source,
        normalization_cfg if normalization_cfg is not None else cfg,
    )

    print("\n=== Eval control config ===")
    print("action_sequence:", workspace_cfg.get("action_sequence", None))
    print("execution_length:", workspace_cfg.get("execution_length", None))
    print("temporal_ensemble:", workspace_cfg.get("temporal_ensemble", None))

    print("\n=== Creating Workspace and loading ACT snapshot ===")
    ws = make_workspace_and_load_snapshot(workspace_cfg, snapshot_path)
    policy_observation_space = getattr(ws.eval_env, "observation_space", None)

    print("\n=== Creating evaluation env ===")
    env = _make_eval_env_with_normalization(runtime_cfg, normalization_cfg)
    robot_spawn_info = _apply_robot_spawn_offset_xy(env, args.robot_spawn_offset_xy)
    if robot_spawn_info is not None:
        print("robot_spawn:", robot_spawn_info)
    if args.freeze_human_arm:
        print("freeze_human_arm: will freeze after each reset")
    if (
        args.human_arm_aggression != 1.0
        or args.human_arm_substeps != 1
        or args.human_arm_zero_dwell
        or args.human_arm_walk_radius is not None
        or args.human_arm_goal_xy is not None
        or args.human_arm_keepout_min_clear is not None
        or args.human_arm_disable_keepout
        or args.human_arm_force_carrier_xy is not None
        or args.human_arm_force_carrier_amp_xy is not None
        or args.human_arm_drawer_obstruction
        or args.human_arm_natural_contact_motion
        or args.human_arm_final_clear_after_steps >= 0
    ):
        print(
            "human_arm_challenge:",
            f"aggression={args.human_arm_aggression}",
            f"substeps={args.human_arm_substeps}",
            f"zero_dwell={args.human_arm_zero_dwell}",
            f"walk_radius={args.human_arm_walk_radius}",
            f"goal_xy={args.human_arm_goal_xy}",
            f"keepout_min_clear={args.human_arm_keepout_min_clear}",
            f"disable_keepout={args.human_arm_disable_keepout}",
            f"force_carrier_xy={args.human_arm_force_carrier_xy}",
            f"force_carrier_amp_xy={args.human_arm_force_carrier_amp_xy}",
            f"force_carrier_frequency={args.human_arm_force_carrier_frequency}",
            f"drawer_obstruction={args.human_arm_drawer_obstruction}",
            f"drawer_obstruction_xy={args.human_arm_drawer_obstruction_xy}",
            f"drawer_obstruction_amp_xy={args.human_arm_drawer_obstruction_amp_xy}",
            f"yaw_offset_deg={args.human_arm_yaw_offset_deg}",
            f"natural_contact_motion={args.human_arm_natural_contact_motion}",
            f"natural_motion_frequency={args.human_arm_natural_motion_frequency}",
            f"natural_lateral_scale={args.human_arm_natural_lateral_scale}",
            f"natural_return_curl_scale={args.human_arm_natural_return_curl_scale}",
            f"final_clear_after_steps={args.human_arm_final_clear_after_steps}",
            f"final_clear_duration_steps={args.human_arm_final_clear_duration_steps}",
            f"final_clear_trigger={args.human_arm_final_clear_trigger}",
            f"final_clear_max_carrier_speed={args.human_arm_final_clear_max_carrier_speed}",
            f"final_clear_max_joint_speed={args.human_arm_final_clear_max_joint_speed}",
            f"final_clear_carrier_xy={args.human_arm_final_clear_carrier_xy}",
        )
    if args.visual_only_human_arm:
        disabled = _disable_human_arm_collisions(env)
        print(f"visual_only_human_arm: disabled_physical_contact_geoms={disabled}")
        print(
            "visual_only_human_arm: actual contact response is disabled; "
            "use h_violation/min_robot_human_distance as the collision check."
        )
    elif args.enable_human_arm_collisions:
        enabled = _enable_human_arm_collisions(env)
        print(f"enable_human_arm_collisions: enabled_physical_contact_geoms={enabled}")
    env_action_shape = infer_env_action_shape(env, fallback=(16, 16))
    print("env_action_shape:", env_action_shape)

    if args.hide_human_arm_policy_obs and args.policy_env is not None:
        raise ValueError(
            "Use either --hide-human-arm-policy-obs or --policy-env, not both."
        )

    policy_env = None
    policy_env_action_shape = None
    policy_robot_spawn_info = None
    if args.policy_env is not None:
        print("\n=== Creating clean policy observation env ===")
        policy_cfg = _make_policy_env_cfg(cfg, args.policy_env)
        policy_env = _make_eval_env_with_normalization(
            policy_cfg,
            normalization_cfg,
        )
        policy_env_action_shape = infer_env_action_shape(
            policy_env,
            fallback=env_action_shape,
        )
        if policy_env_action_shape != env_action_shape:
            raise ValueError(
                "Policy env action shape does not match eval env action shape: "
                f"policy={policy_env_action_shape}, eval={env_action_shape}."
            )
        policy_robot_spawn_info = _apply_robot_spawn_offset_xy(
            policy_env, args.robot_spawn_offset_xy
        )
        print("policy_env:", args.policy_env)
        print("policy_env_action_shape:", policy_env_action_shape)
        if policy_robot_spawn_info is not None:
            print("policy_robot_spawn:", policy_robot_spawn_info)

    safety_env = None
    safety_robot_spawn_info = None
    if args.safety_env is not None:
        print("\n=== Creating mirrored safety env ===")
        safety_cfg = _make_policy_env_cfg(runtime_cfg, args.safety_env)
        safety_env = _make_eval_env_with_normalization(
            safety_cfg,
            normalization_cfg if normalization_cfg is not None else cfg,
        )
        safety_env_action_shape = infer_env_action_shape(
            safety_env,
            fallback=env_action_shape,
        )
        if safety_env_action_shape != env_action_shape:
            raise ValueError(
                "Safety env action shape does not match task env action shape: "
                f"safety={safety_env_action_shape}, task={env_action_shape}."
            )
        if args.visual_only_human_arm:
            disabled = _disable_human_arm_collisions(safety_env)
            print(
                "safety_env visual_only_human_arm: "
                f"disabled_physical_contact_geoms={disabled}"
            )
        elif args.enable_human_arm_collisions:
            enabled = _enable_human_arm_collisions(safety_env)
            print(
                "safety_env enable_human_arm_collisions: "
                f"enabled_physical_contact_geoms={enabled}"
            )
        safety_robot_spawn_info = _apply_robot_spawn_offset_xy(
            safety_env, args.robot_spawn_offset_xy
        )
        print("safety_env:", args.safety_env)
        print("safety_env_action_shape:", safety_env_action_shape)
        if safety_robot_spawn_info is not None:
            print("safety_robot_spawn:", safety_robot_spawn_info)

    output_root, step_jsonl_path, episode_summary_path, final_summary_path = make_output_paths(args)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.video_dir is not None:
        video_dir = Path(args.video_dir)
    else:
        video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_recorder = WallClockVideoRecorder(
        video_dir if args.record_video else None,
        fps=20,
        time_base=args.video_time_base,
    )
    video_stop_steps = _resolve_video_stop_steps(args, video_recorder)
    if args.frame_image_dir is not None:
        frame_image_dir = Path(args.frame_image_dir)
    else:
        frame_image_dir = output_root / "frames"
    if args.save_frame_images:
        frame_image_dir.mkdir(parents=True, exist_ok=True)
    saved_frame_image_paths = []

    trajectory_logging_enabled = bool(
        args.log_chunk_trajectories
        and args.condition in {"sequential", "sequential_oscbf", "chunk_deform", "path_consistent_brake"}
    )
    chunk_trajectory_jsonl_path = output_root / "chunk_trajectory_traces.jsonl"
    mpc_replay_diagnostics_jsonl_path = output_root / "mpc_replay_diagnostics.jsonl"
    nominal_rollout_diagnostics_jsonl_path = output_root / "nominal_rollout_diagnostics.jsonl"
    human_arm_trajectory_jsonl_path = output_root / "human_arm_trajectory.jsonl"
    executed_policy_trajectory_jsonl_path = output_root / "executed_policy_trajectory.jsonl"
    trajectory_plot_dir = output_root / "trajectory_viewers"
    mpc_replay_diagnostic_logging_enabled = bool(
        getattr(args, "log_mpc_replay_diagnostics", False)
        and args.condition in {"chunk_deform", "sequential_oscbf", "path_consistent_brake"}
    )
    nominal_rollout_diagnostic_logging_enabled = bool(
        mpc_replay_diagnostic_logging_enabled
        and args.condition in {"chunk_deform", "sequential_oscbf", "path_consistent_brake"}
    )
    nominal_rollout_diagnostics_max_events = int(
        getattr(
            args,
            "nominal_rollout_diagnostics_max_events",
            max(1000, int(getattr(args, "mpc_replay_diagnostics_max_events", 300))),
        )
    )
    if trajectory_logging_enabled and args.plot_chunk_trajectories_3d:
        trajectory_plot_dir.mkdir(parents=True, exist_ok=True)

    print("record_video:", args.record_video)
    print("video_dir:", video_dir)
    print("save_frame_images:", args.save_frame_images)
    if args.save_frame_images:
        print("frame_image_dir:", frame_image_dir)
        print("frame_image_every:", args.frame_image_every)
    print("stop_video_at_seconds:", args.stop_video_at_seconds)
    print("video_time_base:", args.video_time_base)
    print("stop_video_at_steps:", video_stop_steps)
    print("log_chunk_trajectories:", trajectory_logging_enabled)
    print("log_mpc_replay_diagnostics:", mpc_replay_diagnostic_logging_enabled)
    if mpc_replay_diagnostic_logging_enabled:
        print("mpc_replay_diagnostics_jsonl:", mpc_replay_diagnostics_jsonl_path)
        print("mpc_replay_diagnostics_max_events:", args.mpc_replay_diagnostics_max_events)
    print("log_nominal_rollout_diagnostics:", nominal_rollout_diagnostic_logging_enabled)
    if nominal_rollout_diagnostic_logging_enabled:
        print("nominal_rollout_diagnostics_jsonl:", nominal_rollout_diagnostics_jsonl_path)
        print("nominal_rollout_diagnostics_max_events:", nominal_rollout_diagnostics_max_events)
    if trajectory_logging_enabled:
        print("chunk_trajectory_jsonl:", chunk_trajectory_jsonl_path)
        print("human_arm_trajectory_jsonl:", human_arm_trajectory_jsonl_path)
        print("executed_policy_trajectory_jsonl:", executed_policy_trajectory_jsonl_path)
        print("plot_chunk_trajectories_3d:", args.plot_chunk_trajectories_3d)
        print("trajectory_viewer_dir:", trajectory_plot_dir)

    replay_actions = None
    if args.replay_actions is not None:
        replay_path = Path(args.replay_actions)
        if not replay_path.is_file():
            raise FileNotFoundError(f"Replay actions not found: {replay_path}")
        replay_npz = np.load(replay_path)
        replay_actions = np.asarray(replay_npz["actions"], dtype=np.float32)
        if replay_actions.ndim == len(env_action_shape) + 1:
            replay_actions = replay_actions[None]
        if replay_actions.ndim != len(env_action_shape) + 2:
            raise ValueError(
                f"Expected replay actions with shape (episodes, steps, {env_action_shape}), "
                f"got {replay_actions.shape}."
            )
        if tuple(replay_actions.shape[2:]) != env_action_shape:
            raise ValueError(
                f"Replay action chunk shape {replay_actions.shape[2:]} does not match "
                f"env_action_shape {env_action_shape}."
            )
        print("replay_actions:", replay_path)
        print("replay_actions_shape:", replay_actions.shape)

    path_consistent_brake_eval_config = _path_consistent_brake_eval_config(args)
    path_consistent_background_check_only = bool(
        path_consistent_brake_eval_config.get("background_check_only", False)
    )
    safety_filter_cfg = _args_safety_filter(args)
    barrier_operator_min_clearance = (
        path_consistent_brake_eval_config.get(
            "min_clearance", safety_filter_cfg.get("min_clearance", 0.12)
        )
        if args.condition == "path_consistent_brake"
        else safety_filter_cfg.get("min_clearance", 0.12)
    )

    print("\n=== Creating barrier safety operator/filter ===")
    oscbf = make_oscbf_filter(args)
    horizon_operator = HorizonOSCBFOperator(
        oscbf,
        min_clearance=barrier_operator_min_clearance,
        dt=float(safety_filter_cfg.get("dt", 0.05)),
        predict_human_motion=bool(
            safety_filter_cfg.get("horizon_predict_human_motion", True)
        ),
        human_prediction_max_time=safety_filter_cfg.get(
            "human_motion_prediction_max_time", 0.25
        ),
        human_prediction_max_speed=safety_filter_cfg.get(
            "human_motion_prediction_max_speed", 3.0
        ),
    )
    safechunk = make_safechunk_filter(args, horizon_operator, oscbf=oscbf)
    print("condition:", args.condition)
    if args.condition == "path_consistent_brake":
        print("path_consistent_background_check_only:", path_consistent_background_check_only)
        print("path_consistent_config_min_clearance:", barrier_operator_min_clearance)
    print("arm indices:", oscbf.bigym_action_arm_indices.tolist())
    if args.condition in {"sequential", "sequential_oscbf", "chunk_deform", "path_consistent_brake"}:
        print("chunk controlled indices:", safechunk.controlled_action_indices.tolist())
        print("chunk_deform_mode:", safechunk.deform.mode)
        print("chunk_deformation_enabled:", safechunk.deform.deformation_enabled)
        print("chunk_deformation_scales:", safechunk.deform.chunk_deformation_scales)
        print("recoverable_deform_enabled:", safechunk.recovery.recoverable_deform_enabled)
        print("recoverable_inner_rejoin_metric:", safechunk.recovery.inner_rejoin_metric)
        print("recoverable_final_rejoin_metric:", safechunk.recovery.final_rejoin_metric)
        print("recoverable_q_rejoin_threshold:", safechunk.recovery.q_rejoin_threshold)
        print("recoverable_ee_rejoin_threshold:", safechunk.recovery.ee_rejoin_threshold)
        print("recoverable_cache_nominal_ee:", safechunk.recovery.cache_nominal_ee)
        print("recoverable_ee_rejoin_in_inner_loop:", safechunk.recovery.ee_rejoin_in_inner_loop)
        print("recoverable_explicit_recovery:", safechunk.recovery.explicit_return)
        print("explicit_recovery_commit_accepted_chunks:", safechunk.recovery.commit_accepted_chunks)
        print("explicit_recovery_committed_chunk_safety_check:", safechunk.recovery.committed_chunk_safety_check)
        print("explicit_recovery_committed_min_clearance_for_abort:", safechunk.recovery.committed_min_clearance_for_abort)
        print(
            "explicit_recovery_committed_deform_min_clearance_for_abort:",
            safechunk.recovery.committed_deform_min_clearance_for_abort,
        )
        print("explicit_recovery_repair_committed_action:", safechunk.recovery.repair_committed_action)
        print("explicit_recovery_monotonic_committed_repair:", safechunk.recovery.monotonic_committed_repair)
        print("explicit_recovery_committed_execution_margin:", safechunk.recovery.committed_execution_margin)
        print("explicit_recovery_committed_state_error_threshold:", safechunk.recovery.committed_state_error_threshold)
        print("explicit_recovery_committed_state_error_action:", safechunk.recovery.committed_state_error_action)
        print(
            "explicit_recovery_committed_state_mismatch_abort_requires_unsafe:",
            safechunk.recovery.committed_state_mismatch_abort_requires_unsafe,
        )
        print("explicit_recovery_mpc_recovery_enabled:", safechunk.recovery.mpc_recovery_enabled)
        print("explicit_recovery_mpc_recovery_horizon:", safechunk.recovery.mpc_recovery_horizon)
        print("explicit_recovery_mpc_recovery_prefix_len:", safechunk.recovery.mpc_recovery_prefix_len)
        print("recoverable_deform_horizon:", safechunk.recovery.deform_horizon)
        print("recoverable_recover_horizon:", safechunk.recovery.return_horizon)
        print("recoverable_use_ee_final_check:", safechunk.recovery.use_ee_final_check)
        print("optimized_debug_safety_feasibility:", safechunk.deform.debug_safety_feasibility)
        print("chunk_horizon_predict_human_motion:", horizon_operator.predict_human_motion)
        print("chunk_human_motion_prediction_max_time:", horizon_operator.human_prediction_max_time)
        print("chunk_human_motion_prediction_max_speed:", horizon_operator.human_prediction_max_speed)
        print("diagnostics_enabled:", args.diagnostics_enabled)
        print("diagnostics_large_arm_delta_threshold:", args.diagnostics_large_arm_delta_threshold)
        print("diagnostics_large_base_delta_threshold:", args.diagnostics_large_base_delta_threshold)
        print("diagnostics_low_act_ratio_threshold:", args.diagnostics_low_act_ratio_threshold)
        print("diagnostics_high_fallback_ratio_threshold:", args.diagnostics_high_fallback_ratio_threshold)
        print("recoverable_brake_if_unrecoverable:", safechunk.recovery.brake_if_unrecoverable)
        print("sequential_oscbf_fallback:", safechunk.deform.sequential_oscbf_fallback)


    phase_reanchor_nominal_windows = _load_phase_reanchor_nominal_windows(args)
    if phase_reanchor_nominal_windows:
        print(
            "phase_reanchor_nominal_windows:",
            f"count={len(phase_reanchor_nominal_windows)}",
            f"source={phase_reanchor_nominal_windows[0].get('source')}",
        )
    elif bool(getattr(args, "phase_reanchor_nominal_window_enabled", False)):
        print("phase_reanchor_nominal_windows: unavailable")

    all_step_metrics: list[StepMetrics] = []
    saved_action_episodes = []
    all_episode_summaries: list[dict] = []
    all_chunk_trajectory_records: list[dict] = []
    all_mpc_replay_diagnostic_records: list[dict] = []
    all_nominal_rollout_diagnostic_records: list[dict] = []
    all_human_arm_trajectory_samples: list[dict] = []
    all_executed_policy_trajectory_samples: list[dict] = []
    trajectory_plot_paths: list[str] = []
    show_progress = tqdm is not None and not args.no_progress
    episode_bar = None

    if show_progress:
        episode_bar = _make_progress_bar(
            total=args.episodes,
            desc="episodes",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
        episode_bar.set_postfix(episodes_left=args.episodes)

    try:
        episode_index_offset = int(getattr(args, "episode_index_offset", 0))
        for episode_local in range(args.episodes):
            episode = episode_index_offset + episode_local
            print(f"\n========== Episode {episode} ==========")

            reset_seed = args.seed + episode

            def _reset_episode_envs_for_eval(*, announce: bool):
                if policy_env is None:
                    reset_obs, reset_info = env.reset(seed=reset_seed)
                    if args.freeze_human_arm:
                        frozen = _freeze_human_arm(env)
                        if announce:
                            print(f"freeze_human_arm: episode={episode} frozen={frozen}")
                    else:
                        challenged = _configure_human_arm_challenge(env, args)
                        if challenged and episode == 0 and announce:
                            print(f"human_arm_challenge: episode={episode} configured={challenged}")
                    if args.hide_human_arm_policy_obs:
                        reset_policy_obs = _policy_obs_with_hidden_human_arm(env, reset_obs)
                    else:
                        reset_policy_obs = reset_obs
                else:
                    reset_obs, reset_info = env.reset(seed=reset_seed)
                    if args.freeze_human_arm:
                        frozen = _freeze_human_arm(env)
                        if announce:
                            print(f"freeze_human_arm: episode={episode} frozen={frozen}")
                    else:
                        challenged = _configure_human_arm_challenge(env, args)
                        if challenged and episode == 0 and announce:
                            print(f"human_arm_challenge: episode={episode} configured={challenged}")
                    reset_policy_obs, _policy_info = policy_env.reset(seed=reset_seed)

                reset_safety_runtime_env = safety_env if safety_env is not None else env
                if safety_env is not None:
                    safety_env.reset(seed=reset_seed)
                    sync_counts = _sync_named_mujoco_state(env, safety_env)
                    legs_synced = _sync_animated_legs(safety_env, is_moving=False)
                    if args.freeze_human_arm:
                        frozen = _freeze_human_arm(safety_env)
                        if announce:
                            print(f"safety_env freeze_human_arm: episode={episode} frozen={frozen}")
                    else:
                        challenged = _configure_human_arm_challenge(safety_env, args)
                        if challenged and episode == 0 and announce:
                            print(f"safety_env human_arm_challenge: episode={episode} configured={challenged}")
                    if episode == 0 and announce:
                        print(
                            "safety_env mirrored_state: "
                            f"joints={sync_counts['joints']} actuators={sync_counts['actuators']} "
                            f"animated_legs={legs_synced}"
                        )
                return reset_obs, reset_info, reset_policy_obs, reset_safety_runtime_env

            obs, info, policy_obs, safety_runtime_env = _reset_episode_envs_for_eval(
                announce=True,
            )
            video_recorder.init(safety_runtime_env, enabled=args.record_video)
            episode_metrics: list[StepMetrics] = []
            ablation_force_planned_recovery_q_done = False
            pending_ablation_policy_obs_history = None
            pending_ablation_resume_diag = None
            pending_ablation_force_q_sequence = None
            pending_ablation_force_q_sequence_qvel = None
            pending_ablation_force_q_sequence_index = 0
            pending_ablation_force_q_sequence_history = None
            pending_ablation_force_q_sequence_indices = None
            pending_ablation_force_q_sequence_target_action = None
            pending_ablation_force_q_sequence_target_action_source = None
            ablation_pure_act_resume_steps_left = 0
            ablation_pure_act_resume_total_steps = max(
                0,
                int(getattr(args, "ablation_force_planned_recovery_q_pure_act_resume_steps", 0)),
            )
            episode_chunk_trajectory_records: list[dict] = []
            episode_mpc_replay_diagnostic_records: list[dict] = []
            episode_nominal_rollout_diagnostic_records: list[dict] = []
            episode_human_arm_trajectory_samples: list[dict] = []
            episode_executed_policy_trajectory_samples: list[dict] = []
            if trajectory_logging_enabled:
                initial_executed_sample = _annotate_executed_trajectory_sample(
                    _robot_ee_trajectory_sample(env, episode, 0),
                    None,
                    initial=True,
                )
                if initial_executed_sample is not None:
                    episode_executed_policy_trajectory_samples.append(initial_executed_sample)
                    all_executed_policy_trajectory_samples.append(initial_executed_sample)
            saved_episode_actions = []
            episode_stop_reason = None
            policy_video_frames = []
            policy_video_timestamps = []
            gripper_latched = False
            policy_step = 0
            human_done_clear_steps = 0
            action_history_reset_after_exit = False
            visual_history_reset_after_exit = False
            human_exit_resume_started = False
            human_exit_resume_start_step = None
            human_exit_resume_action_reset_count = 0
            human_exit_resume_visual_reset_count = 0
            pause_restart_reset_after_exit = False
            initial_pause_restart_reset = False
            last_safety_intervention_active = False
            last_diagnostic_step_mode = None
            last_rollout_residual_state = None
            last_rollout_residual_l2 = None
            last_rollout_residual_max_abs = None
            last_rollout_residual_base_l2 = None
            last_rollout_residual_arm_l2 = None
            last_rollout_prediction_untrusted = False
            phase_reanchor_steps_left = 0
            phase_reanchor_cooldown_left = 0
            phase_reanchor_drawer_history = []
            phase_reanchor_reset_after_step = False
            phase_reanchor_best_target_distance = None
            phase_reanchor_best_target_signature = None
            phase_reanchor_best_control_distance = None
            phase_reanchor_best_control_signature = None
            phase_reanchor_taskspace_worsen_count = 0
            phase_reanchor_suppress_q_servo_steps_left = 0
            phase_reanchor_live_extension_count = 0
            phase_reanchor_bridge_preload_start_progress = None
            phase_reanchor_bridge_preload_count = 0
            phase_reanchor_bridge_preload_validated_latched = False
            phase_reanchor_bridge_preload_reason_last = None
            phase_reanchor_bridge_preload_progress_delta_last = None
            phase_reanchor_bridge_preload_handle_dist_last = None
            phase_reanchor_early_release_grace_left = 0
            post_recovery_task_guard_steps_left = 0
            post_recovery_task_guard_reason = None
            post_recovery_task_guard_best_progress = None
            post_recovery_progress_regression = None
            post_recovery_reanchor_started = False
            post_recovery_no_progress_count = 0
            post_recovery_mid_progress_no_progress_count = 0
            post_recovery_mid_progress_best_progress = None
            post_recovery_mid_progress_best_distance = None
            post_recovery_mid_progress_distance_regression = None
            post_recovery_mid_progress_reseed_count = 0
            post_recovery_mid_progress_reseed_triggered = False
            post_recovery_mid_progress_reseed_reset_count = 0
            post_recovery_mid_progress_reseed_reason = None
            post_recovery_mid_progress_prior_action_seed_count = 0
            post_recovery_mid_progress_prior_action_seed_step = None
            post_recovery_mid_progress_prior_action_seed_age = None
            last_progressing_act_chunk = None
            last_progressing_act_chunk_step = None
            post_recovery_no_progress_triggered = False
            post_recovery_no_progress_target_distance = None
            recovery_policy_obs_history = []
            post_recovery_act_bridge_steps_left = 0
            post_recovery_act_bridge_progress_best = None
            post_recovery_act_bridge_no_progress_count = 0
            post_recovery_act_bridge_total_steps = max(
                0,
                int(getattr(args, "post_recovery_act_bridge_steps", 0)),
            )
            post_recovery_seed_fresh_act_history_pending = False
            last_recovery_first_action_for_bridge = None
            last_recovery_action_step_for_bridge = None
            handoff_release_first_action_for_bridge = None
            handoff_release_action_step_for_bridge = None
            handoff_release_executed_action_for_bridge = None
            if hasattr(safechunk, "reset"):
                safechunk.reset()
            # Human-motion prediction carries previous capsule velocity. Clear it
            # at every episode boundary so sequence mode matches isolated mode.
            if hasattr(horizon_operator, "reset_human_motion_prediction"):
                horizon_operator.reset_human_motion_prediction()
            precomputed_step0_filter = None
            if episode == 0:
                warm_h1state = extract_h1_state(env)
                warm_q = np.asarray(warm_h1state.q_full, dtype=np.float32).reshape(-1)
                warm_qd = np.asarray(warm_h1state.qd_full, dtype=np.float32).reshape(-1)
                warm_obs = _chunk_obs_with_q(obs, warm_q)
                warm_task_state = (
                    _diagnostic_task_state(env)
                    if args.diagnostics_enabled
                    else {"task_progress": None}
                )
                warm_task_progress = warm_task_state.get("task_progress")
                horizon_operator.set_context(safety_runtime_env, warm_obs, warm_q, warm_qd)
                if args.condition in {
                    "oscbf",
                    "sequential",
                    "sequential_oscbf",
                    "chunk_deform",
                    "path_consistent_brake",
                }:
                    cbf_warmup_t0 = time.perf_counter()
                    cbf_warmup_info = _warmup_oscbf_cbf_paths(
                        oscbf,
                        safety_runtime_env,
                        warm_obs,
                        warm_q,
                        warm_qd,
                    )
                    cbf_warmup_info["wall_time_ms"] = float(
                        1000.0 * (time.perf_counter() - cbf_warmup_t0)
                    )
                    print("oscbf_cbf_warmup_info:", cbf_warmup_info)
                if args.live_h_monitor:
                    warm_monitor_t0 = time.perf_counter()
                    compute_oscbf_h_monitor(
                        filt=oscbf,
                        env=safety_runtime_env,
                        obs=obs,
                        q_full=warm_q,
                        qd_full=warm_qd,
                    )
                    print(
                        "live_h_warmup_time_ms:",
                        f"{1000.0 * (time.perf_counter() - warm_monitor_t0):.3f}",
                    )
                if (
                    args.condition == "chunk_deform"
                    and getattr(safechunk, "mode", None) == "optimized"
                ):
                    warmup_info = safechunk.deform.warmup_optimizer(warm_obs)
                    print(
                        "safechunk_optimizer_warmup_time_ms:",
                        f"{warmup_info.get('optimizer_warmup_time_ms', 0.0):.3f}",
                    )
                    print(
                        "safechunk_optimizer_warmup_results:",
                        warmup_info.get("optimizer_warmup_results"),
                    )
                    print(
                        "safechunk_optimizer_warmup_live_path_result:",
                        warmup_info.get("optimizer_warmup_live_path_result"),
                    )
                if args.condition in {
                    "sequential",
                    "sequential_oscbf",
                    "chunk_deform",
                    "path_consistent_brake",
                }:
                    try:
                        if replay_actions is None:
                            warm_policy_obs = _adapt_policy_obs_to_space(
                                policy_obs,
                                policy_observation_space,
                            )
                            warm_env_action = policy_action(ws, warm_policy_obs, step=policy_step)
                            warm_env_action = normalise_env_action_shape(
                                warm_env_action,
                                env_action_shape,
                            )
                        elif episode < replay_actions.shape[0] and replay_actions.shape[1] > 0:
                            warm_env_action = replay_actions[episode, 0].copy()
                        else:
                            warm_env_action = None
                        if warm_env_action is not None:
                            warm_nominal_chunk, _ = _as_chunk(warm_env_action)
                            if (
                                args.condition in {"sequential", "sequential_oscbf"}
                                or (
                                    args.condition == "chunk_deform"
                                    and bool(getattr(safechunk, "sequential_oscbf_fallback", False))
                                )
                            ):
                                sequential_warmup_t0 = time.perf_counter()
                                _seq_warm_chunk, sequential_warmup_info = horizon_operator.filter_chunk(
                                    warm_nominal_chunk,
                                    obs=warm_obs,
                                    env=safety_runtime_env,
                                    q_full=warm_q,
                                    qd_full=warm_qd,
                                )
                                print(
                                    "sequential_oscbf_jax_warmup_time_ms:",
                                    f"{1000.0 * (time.perf_counter() - sequential_warmup_t0):.3f}",
                                )
                                print(
                                    "sequential_oscbf_jax_warmup_info:",
                                    {
                                        "used": sequential_warmup_info.get("jax_sequential_oscbf_used"),
                                        "pelvis": sequential_warmup_info.get("jax_sequential_oscbf_use_pelvis_cbf"),
                                        "filter_time_ms": sequential_warmup_info.get("jax_sequential_oscbf_time_ms"),
                                    },
                                )
                                if getattr(horizon_operator, "predict_human_motion", False):
                                    try:
                                        human_obstacles, _ = horizon_operator._current_human_obstacles(warm_obs)
                                        horizon_operator._capsule_a_velocity_world = np.zeros_like(
                                            human_obstacles["capsule_a"],
                                            dtype=np.float32,
                                        )
                                        horizon_operator._capsule_b_velocity_world = np.zeros_like(
                                            human_obstacles["capsule_b"],
                                            dtype=np.float32,
                                        )
                                        horizon_operator._human_motion_prediction_available = True
                                        horizon_operator._human_motion_prediction_speed = 0.0
                                        horizon_operator._human_rollout_cache = {}
                                        predicted_warmup_t0 = time.perf_counter()
                                        _seq_pred_warm_chunk, sequential_pred_warmup_info = (
                                            horizon_operator.filter_chunk(
                                                warm_nominal_chunk,
                                                obs=warm_obs,
                                                env=safety_runtime_env,
                                                q_full=warm_q,
                                                qd_full=warm_qd,
                                            )
                                        )
                                        print(
                                            "sequential_oscbf_jax_predicted_warmup_time_ms:",
                                            f"{1000.0 * (time.perf_counter() - predicted_warmup_t0):.3f}",
                                        )
                                        print(
                                            "sequential_oscbf_jax_predicted_warmup_info:",
                                            {
                                                "used": sequential_pred_warmup_info.get("jax_sequential_oscbf_used"),
                                                "pelvis": sequential_pred_warmup_info.get("jax_sequential_oscbf_use_pelvis_cbf"),
                                                "prediction_available": sequential_pred_warmup_info.get("human_motion_prediction_available"),
                                                "filter_time_ms": sequential_pred_warmup_info.get("jax_sequential_oscbf_time_ms"),
                                            },
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        print(f"sequential_oscbf_jax_predicted_warmup_failed: {exc}")
                                    finally:
                                        horizon_operator._human_rollout_cache = {}
                            live_warmup_t0 = time.perf_counter()
                            if (
                                args.condition == "chunk_deform"
                                and getattr(safechunk, "mode", None) == "optimized"
                            ):
                                policy_warmup_info = safechunk.deform.warmup_optimizer(
                                    warm_obs,
                                    nominal_chunk=warm_nominal_chunk,
                                )
                                print(
                                    "safechunk_policy_chunk_warmup_time_ms:",
                                    f"{1000.0 * (time.perf_counter() - live_warmup_t0):.3f}",
                                )
                                print(
                                    "safechunk_policy_chunk_warmup_live_path_result:",
                                    policy_warmup_info.get(
                                        "optimizer_warmup_live_path_result"
                                    ),
                                )
                                live_warmup_t0 = time.perf_counter()
                            if args.condition in {"sequential", "sequential_oscbf"}:
                                safechunk.deform.deform_chunk_with_oscbf(
                                    warm_obs,
                                    warm_nominal_chunk,
                                    env=safety_runtime_env,
                                    q_full=warm_q,
                                    qd_full=warm_qd,
                                )
                            else:
                                live_warm_chunk, live_warm_info = safechunk.filter_chunk(
                                    warm_obs,
                                    warm_nominal_chunk,
                                    env=safety_runtime_env,
                                    q_full=warm_q,
                                    qd_full=warm_qd,
                                    task_progress=warm_task_progress,
                                    live_monitor_min_h=None,
                                    live_monitor_h_violation=False,
                                )
                                if (
                                    args.condition == "chunk_deform"
                                    and getattr(safechunk, "mode", None) == "optimized"
                                    and replay_actions is None
                                    and args.initial_pause_restart_steps <= 0
                                ):
                                    precomputed_step0_filter = {
                                        "env_action": np.asarray(warm_env_action).copy(),
                                        "nominal_chunk": np.asarray(warm_nominal_chunk).copy(),
                                        "safe_chunk": np.asarray(live_warm_chunk).copy(),
                                        "safety_info": dict(live_warm_info or {}),
                                    }
                            if hasattr(safechunk, "synchronize_accelerator"):
                                safechunk.synchronize_accelerator()
                            print(
                                "chunk_filter_live_warmup_time_ms:",
                                f"{1000.0 * (time.perf_counter() - live_warmup_t0):.3f}",
                            )
                            if (
                                args.condition == "chunk_deform"
                                and getattr(safechunk, "mode", None) == "optimized"
                                and replay_actions is None
                                and args.initial_pause_restart_steps <= 0
                            ):
                                post_step_warmup_info = {"attempted": True}
                                try:
                                    if hasattr(safechunk, "reset"):
                                        safechunk.reset()
                                    post_step_t0 = time.perf_counter()

                                    dummy_h1state = extract_h1_state(env)
                                    dummy_q = np.asarray(dummy_h1state.q_full, dtype=np.float32).reshape(-1)
                                    dummy_qd = np.asarray(dummy_h1state.qd_full, dtype=np.float32).reshape(-1)
                                    dummy_obs = _chunk_obs_with_q(obs, dummy_q)
                                    dummy_task_state = (
                                        _diagnostic_task_state(env)
                                        if args.diagnostics_enabled
                                        else {"task_progress": None}
                                    )
                                    dummy_policy_obs_for_action = _adapt_policy_obs_to_space(
                                        policy_obs,
                                        policy_observation_space,
                                    )
                                    dummy_env_action = policy_action(
                                        ws,
                                        dummy_policy_obs_for_action,
                                        step=0,
                                    )
                                    dummy_env_action = normalise_env_action_shape(
                                        dummy_env_action,
                                        env_action_shape,
                                    )
                                    dummy_nominal_chunk, dummy_was_single = _as_chunk(dummy_env_action)
                                    horizon_operator.set_context(
                                        safety_runtime_env,
                                        dummy_obs,
                                        dummy_q,
                                        dummy_qd,
                                    )
                                    first_filter_t0 = time.perf_counter()
                                    dummy_safe_chunk, dummy_first_info = safechunk.filter_chunk(
                                        dummy_obs,
                                        dummy_nominal_chunk,
                                        env=safety_runtime_env,
                                        q_full=dummy_q,
                                        qd_full=dummy_qd,
                                        task_progress=dummy_task_state.get("task_progress"),
                                        live_monitor_min_h=None,
                                        live_monitor_h_violation=False,
                                    )
                                    first_filter_time_ms = 1000.0 * (
                                        time.perf_counter() - first_filter_t0
                                    )
                                    dummy_safe_env_action = _restore_action_shape(
                                        np.asarray(dummy_safe_chunk, dtype=np.float32),
                                        dummy_was_single,
                                    )
                                    obs_after_dummy, _dummy_reward, _dummy_terminated, _dummy_truncated, _dummy_info = env.step(
                                        dummy_safe_env_action
                                    )
                                    if policy_env is None:
                                        if args.hide_human_arm_policy_obs:
                                            policy_obs_after_dummy = _policy_obs_with_hidden_human_arm(
                                                env,
                                                obs_after_dummy,
                                                prev_policy_obs=policy_obs,
                                            )
                                        else:
                                            policy_obs_after_dummy = obs_after_dummy
                                    else:
                                        policy_obs_after_dummy, _policy_reward, _policy_terminated, _policy_truncated, _policy_info = policy_env.step(
                                            dummy_safe_env_action
                                        )
                                    if safety_env is not None:
                                        _sync_named_mujoco_state(env, safety_env)
                                        _sync_animated_legs(safety_env, is_moving=True)

                                    dummy_h1state_after = extract_h1_state(env)
                                    dummy_q_after = np.asarray(dummy_h1state_after.q_full, dtype=np.float32).reshape(-1)
                                    dummy_qd_after = np.asarray(dummy_h1state_after.qd_full, dtype=np.float32).reshape(-1)
                                    dummy_obs_after = _chunk_obs_with_q(obs_after_dummy, dummy_q_after)
                                    dummy_task_state_after = (
                                        _diagnostic_task_state(env)
                                        if args.diagnostics_enabled
                                        else {"task_progress": None}
                                    )
                                    dummy_policy_obs_after_for_action = _adapt_policy_obs_to_space(
                                        policy_obs_after_dummy,
                                        policy_observation_space,
                                    )
                                    dummy_env_action_after = policy_action(
                                        ws,
                                        dummy_policy_obs_after_for_action,
                                        step=1,
                                    )
                                    dummy_env_action_after = normalise_env_action_shape(
                                        dummy_env_action_after,
                                        env_action_shape,
                                    )
                                    dummy_nominal_chunk_after, _dummy_was_single_after = _as_chunk(
                                        dummy_env_action_after
                                    )
                                    horizon_operator.set_context(
                                        safety_runtime_env,
                                        dummy_obs_after,
                                        dummy_q_after,
                                        dummy_qd_after,
                                    )
                                    second_filter_t0 = time.perf_counter()
                                    dummy_second_chunk, dummy_second_info = safechunk.filter_chunk(
                                        dummy_obs_after,
                                        dummy_nominal_chunk_after,
                                        env=safety_runtime_env,
                                        q_full=dummy_q_after,
                                        qd_full=dummy_qd_after,
                                        task_progress=dummy_task_state_after.get("task_progress"),
                                        live_monitor_min_h=None,
                                        live_monitor_h_violation=False,
                                    )
                                    second_filter_time_ms = 1000.0 * (
                                        time.perf_counter() - second_filter_t0
                                    )
                                    if hasattr(safechunk, "synchronize_accelerator"):
                                        safechunk.synchronize_accelerator()
                                    post_step_warmup_info.update(
                                        {
                                            "compiled": True,
                                            "time_ms": float(1000.0 * (time.perf_counter() - post_step_t0)),
                                            "first_filter_time_ms": float(first_filter_time_ms),
                                            "second_filter_time_ms": float(second_filter_time_ms),
                                            "first_mode": dummy_first_info.get("mode"),
                                            "second_mode": dummy_second_info.get("mode"),
                                            "first_explicit_optimizer_time_ms": dummy_first_info.get("explicit_optimizer_time_ms"),
                                            "second_explicit_optimizer_time_ms": dummy_second_info.get("explicit_optimizer_time_ms"),
                                            "second_deform_gradient_initial_batch_cost_time_ms": dummy_second_info.get("deform_gradient_initial_batch_cost_time_ms"),
                                            "second_return_gradient_initial_batch_cost_time_ms": dummy_second_info.get("return_gradient_initial_batch_cost_time_ms"),
                                        }
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    post_step_warmup_info.update(
                                        {
                                            "compiled": False,
                                            "reason": f"{type(exc).__name__}: {exc}",
                                        }
                                    )
                                finally:
                                    obs, info, policy_obs, safety_runtime_env = _reset_episode_envs_for_eval(
                                        announce=False,
                                    )
                                    video_recorder.init(
                                        safety_runtime_env,
                                        enabled=args.record_video,
                                    )
                                    policy_step = 0
                                    precomputed_step0_filter = None
                                    if hasattr(safechunk, "reset"):
                                        safechunk.reset()
                                    try:
                                        rewarm_t0 = time.perf_counter()
                                        rewarm_h1state = extract_h1_state(env)
                                        rewarm_q = np.asarray(rewarm_h1state.q_full, dtype=np.float32).reshape(-1)
                                        rewarm_qd = np.asarray(rewarm_h1state.qd_full, dtype=np.float32).reshape(-1)
                                        rewarm_obs = _chunk_obs_with_q(obs, rewarm_q)
                                        rewarm_task_state = (
                                            _diagnostic_task_state(env)
                                            if args.diagnostics_enabled
                                            else {"task_progress": None}
                                        )
                                        rewarm_policy_obs = _adapt_policy_obs_to_space(
                                            policy_obs,
                                            policy_observation_space,
                                        )
                                        rewarm_env_action = policy_action(
                                            ws,
                                            rewarm_policy_obs,
                                            step=0,
                                        )
                                        rewarm_env_action = normalise_env_action_shape(
                                            rewarm_env_action,
                                            env_action_shape,
                                        )
                                        rewarm_nominal_chunk, _rewarm_was_single = _as_chunk(
                                            rewarm_env_action
                                        )
                                        horizon_operator.set_context(
                                            safety_runtime_env,
                                            rewarm_obs,
                                            rewarm_q,
                                            rewarm_qd,
                                        )
                                        rewarm_safe_chunk, rewarm_info = safechunk.filter_chunk(
                                            rewarm_obs,
                                            rewarm_nominal_chunk,
                                            env=safety_runtime_env,
                                            q_full=rewarm_q,
                                            qd_full=rewarm_qd,
                                            task_progress=rewarm_task_state.get("task_progress"),
                                            live_monitor_min_h=None,
                                            live_monitor_h_violation=False,
                                        )
                                        if hasattr(safechunk, "synchronize_accelerator"):
                                            safechunk.synchronize_accelerator()
                                        precomputed_step0_filter = {
                                            "env_action": np.asarray(rewarm_env_action).copy(),
                                            "nominal_chunk": np.asarray(rewarm_nominal_chunk).copy(),
                                            "safe_chunk": np.asarray(rewarm_safe_chunk).copy(),
                                            "safety_info": dict(rewarm_info or {}),
                                        }
                                        post_step_warmup_info["step0_recache_time_ms"] = float(
                                            1000.0 * (time.perf_counter() - rewarm_t0)
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        post_step_warmup_info["step0_recache_failed"] = (
                                            f"{type(exc).__name__}: {exc}"
                                        )
                                        if hasattr(safechunk, "reset"):
                                            safechunk.reset()
                                print(
                                    "safechunk_post_step_live_warmup_info:",
                                    post_step_warmup_info,
                                )
                    except Exception as exc:  # noqa: BLE001
                        print(f"chunk_filter_live_warmup_failed: {exc}")
                    finally:
                        if hasattr(safechunk, "reset") and precomputed_step0_filter is None:
                            safechunk.reset()
            episode_wall_t0 = time.perf_counter()
            last_step_wall_t = episode_wall_t0
            if show_progress:
                progress_bar = _make_progress_bar(
                    total=args.steps,
                    desc=f"ep={episode:03d} steps",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                )
                progress_bar.set_postfix(steps_left=args.steps)
            else:
                progress_bar = None

            for step in range(args.steps):
                blocker_info = {}
                if safety_env is not None:
                    _sync_named_mujoco_state(env, safety_env)

                if not args.freeze_human_arm:
                    maybe_blocker_info = _update_temporary_human_blocker_if_present(safety_runtime_env)
                    if maybe_blocker_info is None:
                        human_anchor_xy = None
                        if args.human_arm_ee_obstruction and not args.human_arm_drawer_obstruction:
                            human_anchor_xy = _robot_gripper_geom_world_xy(
                                safety_runtime_env,
                                offset_xy=args.human_arm_ee_offset_xy,
                            )
                            if human_anchor_xy is None:
                                ee_state = extract_h1_state(env)
                                human_anchor_xy = _robot_ee_world_xy(
                                    oscbf,
                                    np.asarray(ee_state.q_full, dtype=np.float32).reshape(-1),
                                    np.asarray(ee_state.qd_full, dtype=np.float32).reshape(-1),
                                    offset_xy=args.human_arm_ee_offset_xy,
                                )
                        _update_scripted_human_arm_pose(
                            safety_runtime_env,
                            args,
                            step=step,
                            anchor_xy=human_anchor_xy,
                        )
                    else:
                        blocker_info = maybe_blocker_info
                elif safety_env is not None:
                    _sync_animated_legs(safety_env, is_moving=False)

                human_arm_trace_sample = None
                human_arm_stride = max(1, int(args.human_arm_trajectory_stride))
                if trajectory_logging_enabled and step % human_arm_stride == 0:
                    human_arm_trace_sample = _human_arm_trajectory_sample(
                        safety_runtime_env,
                        episode,
                        step,
                    )
                    if human_arm_trace_sample is not None:
                        episode_human_arm_trajectory_samples.append(human_arm_trace_sample)
                        all_human_arm_trajectory_samples.append(human_arm_trace_sample)

                h1state = extract_h1_state(
                    env,
                    print_diagnostics=(episode == 0 and step == 0),
                )

                q_full = np.asarray(h1state.q_full, dtype=np.float32).reshape(-1)
                qd_full = np.asarray(h1state.qd_full, dtype=np.float32).reshape(-1)

                if q_full.shape != (14,):
                    raise ValueError(f"Expected q_full shape (14,), got {q_full.shape}")
                if qd_full.shape != (14,):
                    raise ValueError(f"Expected qd_full shape (14,), got {qd_full.shape}")

                task_state_before = (
                    _diagnostic_task_state(env)
                    if args.diagnostics_enabled
                    else {
                        "drawer_open_distance": None,
                        "drawer_open_fraction": None,
                        "drawer_joint_position": None,
                        "task_progress": None,
                        "ee_object_distance": None,
                        "object_state": None,
                    }
                )

                phase_reanchor_state = None
                phase_reanchor_drawer_progress = None
                phase_reanchor_reset_after_step = False
                phase_context_control_enabled = bool(
                    args.phase_reanchor
                    or args.post_recovery_task_guard
                )
                if phase_context_control_enabled:
                    if phase_reanchor_cooldown_left > 0:
                        phase_reanchor_cooldown_left -= 1
                    if phase_reanchor_early_release_grace_left > 0:
                        phase_reanchor_early_release_grace_left -= 1
                phase_reanchor_state = _phase_reanchor_state(env, args)
                if phase_reanchor_state is not None and phase_context_control_enabled:
                    drawer_fraction = phase_reanchor_state.get("drawer_open_fraction")
                    if drawer_fraction is not None and np.isfinite(float(drawer_fraction)):
                        phase_reanchor_drawer_history.append(float(drawer_fraction))

                post_recovery_progress_regression = None
                post_recovery_reanchor_started = False
                post_recovery_no_progress_triggered = False
                post_recovery_mid_progress_reseed_triggered = False
                post_recovery_mid_progress_reseed_reset_count = 0
                post_recovery_mid_progress_reseed_reason = None
                post_recovery_mid_progress_prior_action_seed_count = 0
                post_recovery_mid_progress_prior_action_seed_step = None
                post_recovery_mid_progress_prior_action_seed_age = None
                post_recovery_no_progress_target_distance = None
                post_recovery_no_progress_distance_source = None
                post_recovery_task_guard_ready = False
                post_recovery_task_guard_phase_reason = None
                if args.post_recovery_task_guard:
                    (
                        post_recovery_task_guard_ready,
                        post_recovery_task_guard_phase_reason,
                    ) = _post_recovery_task_guard_ready(
                        task_state_before,
                        phase_reanchor_state,
                        args,
                    )
                    progress_before = _finite_task_progress(task_state_before)
                    if progress_before is not None:
                        if (
                            post_recovery_task_guard_best_progress is None
                            or progress_before > post_recovery_task_guard_best_progress
                        ):
                            post_recovery_task_guard_best_progress = progress_before
                        regression = post_recovery_task_guard_best_progress - progress_before
                        if (
                            regression > args.post_recovery_progress_tolerance
                            and post_recovery_task_guard_ready
                        ):
                            post_recovery_progress_regression = float(regression)
                            post_recovery_task_guard_steps_left = max(
                                post_recovery_task_guard_steps_left,
                                int(args.post_recovery_task_guard_steps),
                            )
                            post_recovery_task_guard_reason = "progress_regression:" + str(
                                post_recovery_task_guard_phase_reason
                            )
                    no_progress_enabled = bool(
                        getattr(args, "post_recovery_no_progress_reanchor", False)
                    )
                    no_progress_start_step = int(
                        getattr(args, "post_recovery_no_progress_start_step", 90)
                    )
                    no_progress_patience = max(
                        1,
                        int(getattr(args, "post_recovery_no_progress_patience", 8)),
                    )
                    no_progress_max_progress = float(
                        getattr(args, "post_recovery_no_progress_max_progress", 0.04)
                    )
                    no_progress_min_target_distance = float(
                        getattr(args, "post_recovery_no_progress_min_target_distance", 0.62)
                    )
                    if phase_reanchor_state is not None:
                        distance_source_request = str(
                            getattr(args, "post_recovery_no_progress_distance_source", "measurement")
                        )
                        distance_keys = {
                            "measurement": ("ee_to_handle_dist", "measurement"),
                            "control": ("control_ee_to_handle_dist", "control"),
                            "site": ("site_ee_to_handle_dist", "site"),
                            "gripper": ("gripper_to_handle_dist", "gripper"),
                        }
                        distance_key, distance_source = distance_keys.get(
                            distance_source_request,
                            ("ee_to_handle_dist", "measurement"),
                        )
                        try:
                            post_recovery_no_progress_target_distance = float(
                                phase_reanchor_state.get(distance_key)
                            )
                            if not np.isfinite(post_recovery_no_progress_target_distance):
                                post_recovery_no_progress_target_distance = None
                            else:
                                post_recovery_no_progress_distance_source = distance_source
                        except (TypeError, ValueError):
                            post_recovery_no_progress_target_distance = None
                        if post_recovery_no_progress_target_distance is None and distance_key != "ee_to_handle_dist":
                            try:
                                post_recovery_no_progress_target_distance = float(
                                    phase_reanchor_state.get("ee_to_handle_dist")
                                )
                                if not np.isfinite(post_recovery_no_progress_target_distance):
                                    post_recovery_no_progress_target_distance = None
                                else:
                                    post_recovery_no_progress_distance_source = "measurement_fallback"
                            except (TypeError, ValueError):
                                post_recovery_no_progress_target_distance = None
                    if post_recovery_act_bridge_steps_left > 0:
                        bridge_progress_value = None
                        try:
                            bridge_progress_value = float(task_state_before.get("drawer_open_distance"))
                            if not np.isfinite(bridge_progress_value):
                                bridge_progress_value = None
                        except (TypeError, ValueError):
                            bridge_progress_value = None
                        if bridge_progress_value is None:
                            bridge_progress_value = progress_before
                        if bridge_progress_value is not None and np.isfinite(float(bridge_progress_value)):
                            bridge_progress_value = float(bridge_progress_value)
                            bridge_epsilon = float(
                                getattr(args, "post_recovery_act_bridge_no_progress_epsilon", 1e-5)
                            )
                            if (
                                post_recovery_act_bridge_progress_best is None
                                or bridge_progress_value > post_recovery_act_bridge_progress_best + bridge_epsilon
                            ):
                                post_recovery_act_bridge_progress_best = bridge_progress_value
                                post_recovery_act_bridge_no_progress_count = 0
                            else:
                                post_recovery_act_bridge_no_progress_count += 1
                            bridge_elapsed = int(
                                max(
                                    0,
                                    int(post_recovery_act_bridge_total_steps)
                                    - int(post_recovery_act_bridge_steps_left),
                                )
                            )
                            if (
                                bridge_elapsed >= int(
                                    getattr(args, "post_recovery_act_bridge_no_progress_min_steps", 8)
                                )
                                and post_recovery_act_bridge_no_progress_count >= int(
                                    getattr(args, "post_recovery_act_bridge_no_progress_patience", 8)
                                )
                                and phase_reanchor_steps_left <= 0
                            ):
                                post_recovery_no_progress_triggered = True
                                post_recovery_task_guard_steps_left = max(
                                    post_recovery_task_guard_steps_left,
                                    int(args.post_recovery_task_guard_steps),
                                )
                                post_recovery_task_guard_reason = "post_bridge_no_drawer_progress"
                                post_recovery_no_progress_count = 0
                                phase_reanchor_cooldown_left = 0
                                post_recovery_act_bridge_steps_left = 0

                    no_progress_ready = bool(
                        no_progress_enabled
                        and post_recovery_task_guard_ready
                        and phase_reanchor_cooldown_left <= 0
                        and phase_reanchor_early_release_grace_left <= 1
                        and step >= no_progress_start_step
                        and progress_before is not None
                        and progress_before <= no_progress_max_progress
                        and post_recovery_no_progress_target_distance is not None
                        and np.isfinite(post_recovery_no_progress_target_distance)
                        and post_recovery_no_progress_target_distance
                        >= no_progress_min_target_distance
                    )
                    if no_progress_ready:
                        post_recovery_no_progress_count += 1
                    else:
                        post_recovery_no_progress_count = 0
                    if post_recovery_no_progress_count >= no_progress_patience:
                        post_recovery_no_progress_triggered = True
                        post_recovery_task_guard_steps_left = max(
                            post_recovery_task_guard_steps_left,
                            int(args.post_recovery_task_guard_steps),
                        )
                        post_recovery_task_guard_reason = "no_progress:" + str(
                            post_recovery_task_guard_phase_reason
                        )
                        post_recovery_no_progress_count = 0

                    mid_progress_enabled = bool(
                        getattr(args, "post_recovery_mid_progress_no_progress_reanchor", False)
                    )
                    mid_progress_min_progress = float(
                        getattr(args, "post_recovery_mid_progress_min_progress", 0.35)
                    )
                    mid_progress_patience = max(
                        1,
                        int(getattr(args, "post_recovery_mid_progress_patience", 8)),
                    )
                    mid_progress_epsilon = float(
                        getattr(args, "post_recovery_mid_progress_epsilon", 0.001)
                    )
                    mid_progress_distance_regression_threshold = float(
                        getattr(args, "post_recovery_mid_progress_distance_regression", 0.06)
                    )
                    mid_progress_min_target_distance = float(
                        getattr(args, "post_recovery_mid_progress_min_target_distance", 0.42)
                    )
                    reseed_budget_available = bool(
                        post_recovery_mid_progress_reseed_count
                        < max(
                            0,
                            int(
                                getattr(
                                    args,
                                    "post_recovery_mid_progress_reseed_max_count",
                                    1,
                                )
                            ),
                        )
                    )
                    post_recovery_mid_progress_distance_regression = None
                    mid_progress_ready = bool(
                        mid_progress_enabled
                        and post_recovery_task_guard_ready
                        and phase_reanchor_cooldown_left <= 0
                        and phase_reanchor_early_release_grace_left <= 1
                        and phase_reanchor_steps_left <= 0
                        and post_recovery_act_bridge_steps_left <= 0
                        and not last_safety_intervention_active
                        and step >= no_progress_start_step
                        and progress_before is not None
                        and progress_before >= mid_progress_min_progress
                        and post_recovery_no_progress_target_distance is not None
                        and np.isfinite(post_recovery_no_progress_target_distance)
                    )
                    if mid_progress_ready:
                        progress_improved = (
                            post_recovery_mid_progress_best_progress is None
                            or progress_before
                            > post_recovery_mid_progress_best_progress + mid_progress_epsilon
                        )
                        if progress_improved:
                            post_recovery_mid_progress_best_progress = progress_before
                            post_recovery_mid_progress_best_distance = (
                                post_recovery_no_progress_target_distance
                            )
                            post_recovery_mid_progress_no_progress_count = 0
                        else:
                            if post_recovery_mid_progress_best_distance is None:
                                post_recovery_mid_progress_best_distance = (
                                    post_recovery_no_progress_target_distance
                                )
                            post_recovery_mid_progress_distance_regression = float(
                                post_recovery_no_progress_target_distance
                                - post_recovery_mid_progress_best_distance
                            )
                            if (
                                post_recovery_mid_progress_distance_regression
                                >= mid_progress_distance_regression_threshold
                                and post_recovery_no_progress_target_distance
                                >= mid_progress_min_target_distance
                            ):
                                post_recovery_mid_progress_no_progress_count += 1
                            else:
                                post_recovery_mid_progress_no_progress_count = 0
                    else:
                        post_recovery_mid_progress_no_progress_count = 0
                        if progress_before is not None and progress_before < mid_progress_min_progress:
                            post_recovery_mid_progress_best_progress = None
                            post_recovery_mid_progress_best_distance = None

                    if (
                        post_recovery_mid_progress_no_progress_count
                        >= mid_progress_patience
                    ):
                        post_recovery_no_progress_triggered = True
                        mid_progress_phase = (
                            str(phase_reanchor_state.get("phase"))
                            if isinstance(phase_reanchor_state, dict)
                            and phase_reanchor_state.get("phase") is not None
                            else None
                        )
                        reseed_phases = {
                            str(phase)
                            for phase in getattr(
                                args,
                                "post_recovery_mid_progress_reseed_phases",
                                ["pull"],
                            )
                        }
                        reseed_enabled = bool(
                            getattr(
                                args,
                                "post_recovery_mid_progress_reseed_action_history",
                                False,
                            )
                            and mid_progress_phase in reseed_phases
                            and reseed_budget_available
                            and replay_actions is None
                        )
                        reseed_reset_count = 0
                        if reseed_enabled:
                            reseed_reset_count = _reset_action_sequence_history(env)
                            if policy_env is not None:
                                reseed_reset_count += _reset_action_sequence_history(
                                    policy_env
                                )
                        if reseed_enabled and reseed_reset_count > 0:
                            prior_action_seed_count = 0
                            prior_action_seed_age = None
                            prior_action_seed_step = None
                            prior_action_enabled = bool(
                                getattr(
                                    args,
                                    "post_recovery_mid_progress_reseed_prior_progress_action",
                                    False,
                                )
                            )
                            if (
                                prior_action_enabled
                                and last_progressing_act_chunk is not None
                                and last_progressing_act_chunk_step is not None
                            ):
                                prior_action_seed_age = int(
                                    step - last_progressing_act_chunk_step
                                )
                                if prior_action_seed_age <= max(
                                    0,
                                    int(
                                        getattr(
                                            args,
                                            "post_recovery_mid_progress_reseed_prior_max_age",
                                            8,
                                        )
                                    ),
                                ):
                                    seed_seen: set[int] = set()
                                    for candidate in (env, policy_env):
                                        if (
                                            candidate is None
                                            or id(candidate) in seed_seen
                                        ):
                                            continue
                                        seed_seen.add(id(candidate))
                                        prior_action_seed_count += int(
                                            _seed_action_sequence_history_with_prior_act_chunk(
                                                candidate,
                                                last_progressing_act_chunk,
                                            )
                                        )
                                    if prior_action_seed_count > 0:
                                        prior_action_seed_step = int(
                                            last_progressing_act_chunk_step
                                        )
                            bridge_steps = max(
                                1,
                                int(getattr(args, "post_recovery_act_bridge_steps", 4)),
                            )
                            post_recovery_act_bridge_total_steps = max(
                                post_recovery_act_bridge_total_steps,
                                bridge_steps,
                            )
                            post_recovery_act_bridge_steps_left = max(
                                post_recovery_act_bridge_steps_left,
                                bridge_steps,
                            )
                            post_recovery_act_bridge_progress_best = None
                            post_recovery_act_bridge_no_progress_count = 0
                            post_recovery_seed_fresh_act_history_pending = bool(
                                prior_action_seed_count <= 0
                                and getattr(
                                    args,
                                    "handoff_seed_action_history_from_fresh_act",
                                    True,
                                )
                            )
                            phase_reanchor_cooldown_left = max(
                                phase_reanchor_cooldown_left,
                                bridge_steps,
                            )
                            phase_reanchor_steps_left = 0
                            post_recovery_task_guard_steps_left = 0
                            recovery_policy_obs_history.clear()
                            post_recovery_task_guard_reason = (
                                "mid_progress_reseed:"
                                + str(post_recovery_task_guard_phase_reason)
                            )
                            post_recovery_mid_progress_reseed_count += 1
                            post_recovery_mid_progress_reseed_triggered = True
                            post_recovery_mid_progress_reseed_reset_count = int(
                                reseed_reset_count
                            )
                            post_recovery_mid_progress_reseed_reason = (
                                "recent_progress_act_prior:"
                                f"{mid_progress_phase}"
                                if prior_action_seed_count > 0
                                else f"fresh_act_reseed:{mid_progress_phase}"
                            )
                            post_recovery_mid_progress_prior_action_seed_count = int(
                                prior_action_seed_count
                            )
                            post_recovery_mid_progress_prior_action_seed_step = (
                                prior_action_seed_step
                            )
                            post_recovery_mid_progress_prior_action_seed_age = (
                                prior_action_seed_age
                            )
                            post_recovery_mid_progress_best_progress = progress_before
                            post_recovery_mid_progress_best_distance = (
                                post_recovery_no_progress_target_distance
                            )
                        else:
                            post_recovery_task_guard_steps_left = max(
                                post_recovery_task_guard_steps_left,
                                int(args.post_recovery_task_guard_steps),
                            )
                            post_recovery_task_guard_reason = (
                                "mid_progress_no_progress:"
                                + str(post_recovery_task_guard_phase_reason)
                            )
                        post_recovery_mid_progress_no_progress_count = 0

                    if phase_reanchor_early_release_grace_left > 1:
                        post_recovery_task_guard_steps_left = 0
                        post_recovery_no_progress_count = 0
                        post_recovery_mid_progress_no_progress_count = 0
                        post_recovery_task_guard_reason = (
                            f"suppressed:phase_reanchor_early_release_act_grace:{phase_reanchor_early_release_grace_left}"
                        )
                    if post_recovery_task_guard_steps_left > 0:
                        if args.post_recovery_task_guard_force_gripper:
                            gripper_latched = True
                        reanchor_allowed, _guard_phase = _post_recovery_task_guard_reanchor_allowed(
                            phase_reanchor_state,
                            args,
                        )
                        if reanchor_allowed and phase_reanchor_steps_left <= 0:
                            phase_reanchor_steps_left = max(
                                1,
                                int(post_recovery_task_guard_steps_left),
                            )
                            phase_reanchor_cooldown_left = 0
                            phase_reanchor_best_target_distance = None
                            phase_reanchor_best_control_distance = None
                            phase_reanchor_taskspace_worsen_count = 0
                            phase_reanchor_suppress_q_servo_steps_left = 0
                            post_recovery_mid_progress_best_progress = None
                            post_recovery_mid_progress_best_distance = None
                            phase_reanchor_bridge_preload_start_progress = None
                            phase_reanchor_bridge_preload_count = 0
                            phase_reanchor_bridge_preload_validated_latched = False
                            phase_reanchor_bridge_preload_reason_last = None
                            phase_reanchor_bridge_preload_progress_delta_last = None
                            phase_reanchor_bridge_preload_handle_dist_last = None
                            post_recovery_reanchor_started = True
                        post_recovery_task_guard_steps_left = max(
                            0,
                            post_recovery_task_guard_steps_left - 1,
                        )

                resume_affordance_context = _resume_affordance_context_from_task_state(
                    task_state_before,
                    phase_reanchor_state,
                    gripper_latched=bool(gripper_latched),
                    args=args,
                )

                if (
                    args.initial_pause_restart_steps > 0
                    and not initial_pause_restart_reset
                    and step >= args.initial_pause_restart_steps
                    and replay_actions is None
                ):
                    reset_count = _reset_action_sequence_history(env)
                    if policy_env is not None:
                        reset_count += _reset_action_sequence_history(policy_env)
                    if hasattr(safechunk, "reset"):
                        safechunk.reset()
                    policy_step = 0
                    initial_pause_restart_reset = True
                    if episode == 0 or args.debug:
                        print(
                            "initial_pause_restart: reset_action_history "
                            f"step={step} reset_wrappers={reset_count}"
                        )

                if args.record_policy_video and step % args.policy_video_every == 0:
                    policy_frame = _policy_obs_rgb_frame(policy_obs)
                    if policy_frame is not None:
                        policy_video_frames.append(policy_frame)
                        policy_video_timestamps.append(time.perf_counter())

                policy_obs_adapt_time_ms = 0.0
                policy_action_time_ms = 0.0
                policy_obs_for_action = None
                act_resume_diag_info = {}
                act_resume_diag_target_first_action = None
                use_precomputed_step0_filter = bool(
                    step == 0
                    and precomputed_step0_filter is not None
                    and replay_actions is None
                )
                if replay_actions is None:
                    if use_precomputed_step0_filter:
                        env_action = np.asarray(
                            precomputed_step0_filter["env_action"]
                        ).copy()
                    else:
                        policy_obs_adapt_t0 = time.perf_counter()
                        policy_obs_for_action = _adapt_policy_obs_to_space(
                            policy_obs,
                            policy_observation_space,
                        )
                        policy_obs_adapt_time_ms = 1000.0 * (time.perf_counter() - policy_obs_adapt_t0)
                        policy_action_t0 = time.perf_counter()
                        env_action = policy_action(ws, policy_obs_for_action, step=policy_step)
                        env_action = normalise_env_action_shape(env_action, env_action_shape)
                        policy_action_time_ms = 1000.0 * (time.perf_counter() - policy_action_t0)
                else:
                    if episode >= replay_actions.shape[0] or step >= replay_actions.shape[1]:
                        print(
                            f"Stopping episode {episode}: replay actions ended at "
                            f"shape {replay_actions.shape}."
                        )
                        break
                    env_action = replay_actions[episode, step].copy()

                if pending_ablation_resume_diag is not None and replay_actions is None:
                    diag_seed_step = pending_ablation_resume_diag.get("seed_step")
                    try:
                        diag_target_age_steps = int(step) - int(diag_seed_step)
                    except Exception:  # noqa: BLE001
                        diag_target_age_steps = None
                    act_resume_diag_info = {
                        "act_resume_diag_active": True,
                        "act_resume_diag_seed_step": diag_seed_step,
                        "act_resume_diag_query_step": int(step),
                        "act_resume_diag_target_age_steps": diag_target_age_steps,
                        "act_resume_diag_target_action_source": pending_ablation_resume_diag.get("target_action_source"),
                    }
                    expected_low_dim = pending_ablation_resume_diag.get("seeded_low_dim_state")
                    expected_policy_obs = pending_ablation_resume_diag.get("seeded_policy_obs_for_action")
                    if isinstance(policy_obs_for_action, dict):
                        if expected_low_dim is not None:
                            current_low_dim = policy_obs_for_action.get("low_dim_state")
                            act_resume_diag_info.update(
                                _vector_compare_metrics(
                                    current_low_dim,
                                    expected_low_dim,
                                    "act_resume_diag_policy_low_dim_vs_seed",
                                )
                            )
                        if expected_policy_obs is not None:
                            act_resume_diag_info.update(
                                _policy_obs_snapshot_compare_metrics(
                                    policy_obs_for_action,
                                    expected_policy_obs,
                                    "act_resume_diag_policy_obs_vs_seed",
                                )
                            )
                    expected_visual_pose = pending_ablation_resume_diag.get("seeded_visual_pose_snapshot")
                    if expected_visual_pose is not None:
                        act_resume_diag_info.update(
                            _mujoco_visual_pose_compare_metrics(
                                _mujoco_visual_pose_snapshot(env),
                                expected_visual_pose,
                                "act_resume_diag_visual_pose_vs_seed",
                            )
                        )
                    target_action = pending_ablation_resume_diag.get("target_action")
                    act_resume_diag_target_first_action = _first_action_or_none(target_action)
                    predicted_first_action = _first_action_or_none(env_action)
                    act_resume_diag_info.update(
                        _target_action_window_diagnostics(target_action, predicted_first_action)
                    )
                    if act_resume_diag_target_first_action is not None and predicted_first_action is not None:
                        act_resume_diag_info.update(
                            _vector_compare_metrics(
                                predicted_first_action,
                                act_resume_diag_target_first_action,
                                "act_resume_diag_first_action_vs_target",
                            )
                        )
                    pending_ablation_resume_diag = None

                first_action = extract_first_action(env_action)
                chunk_filter_mode = args.condition in {"sequential", "sequential_oscbf", "chunk_deform", "path_consistent_brake"}
                pelvis_cbf_mode = (
                    args.condition == "oscbf"
                    and getattr(oscbf, "enable_pelvis_cbf", False)
                    and getattr(oscbf, "pelvis_oscbf_config", None) is not None
                )
                arm_idx = (
                    safechunk.controlled_action_indices
                    if chunk_filter_mode
                    else (
                        oscbf.bigym_action_safety_indices
                        if pelvis_cbf_mode
                        else oscbf.bigym_action_arm_indices
                    )
                )
                state_idx = (
                    safechunk.controlled_state_indices
                    if chunk_filter_mode
                    else (
                        oscbf.bigym_state_safety_indices
                        if pelvis_cbf_mode
                        else oscbf.bigym_state_arm_indices
                    )
                )
                base_idx = getattr(oscbf, "bigym_action_base_indices", np.asarray([], dtype=np.int64))
                valid_base_idx = base_idx[base_idx < first_action.shape[0]]
                non_arm_idx = get_non_arm_indices(first_action.shape[0], arm_idx)

                post_recovery_act_bridge_active = bool(
                    replay_actions is None
                    and chunk_filter_mode
                    and post_recovery_act_bridge_steps_left > 0
                )
                post_recovery_act_bridge_step_index = (
                    None
                    if not post_recovery_act_bridge_active
                    else int(
                        max(
                            0,
                            post_recovery_act_bridge_total_steps
                            - post_recovery_act_bridge_steps_left,
                        )
                    )
                )
                action_bridge_info = {}
                if post_recovery_act_bridge_active:
                    fresh_action_seed_count = 0
                    fresh_action_seed_reset_count = 0
                    fresh_action_seed_source = None
                    if (
                        bool(getattr(args, "handoff_seed_action_history_from_fresh_act", True))
                        and post_recovery_seed_fresh_act_history_pending
                    ):
                        seed_seen: set[int] = set()
                        for candidate in (env, policy_env):
                            if candidate is None or id(candidate) in seed_seen:
                                continue
                            seed_seen.add(id(candidate))
                            try:
                                fresh_action_seed_count += int(
                                    _seed_action_sequence_history_with_nominal_actions(
                                        candidate,
                                        env_action,
                                        history_window_len=max(
                                            1,
                                            int(getattr(args, "frame_stack", 4) or 4),
                                        ),
                                    )
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "Could not seed post-recovery bridge action history "
                                    "from fresh ACT chunk: %s",
                                    exc,
                                )
                        if fresh_action_seed_count > 0:
                            fresh_action_seed_source = (
                                "fresh_act_env_action_single_contributor"
                            )
                            post_recovery_seed_fresh_act_history_pending = False
                        else:
                            fresh_action_seed_source = "fresh_act_seed_failed"
                    action_bridge_info = _action_bridge_diagnostics(
                        env=env,
                        last_recovery_first_action=last_recovery_first_action_for_bridge,
                        resume_first_action=first_action,
                        arm_indices=arm_idx,
                        gripper_index=args.gripper_latch_dim,
                    )
                    action_bridge_info.update(
                        {
                            "post_recovery_act_bridge_fresh_action_seed_pending": bool(
                                post_recovery_seed_fresh_act_history_pending
                            ),
                            "post_recovery_act_bridge_fresh_action_seed_count": int(
                                fresh_action_seed_count
                            ),
                            "post_recovery_act_bridge_fresh_action_seed_reset_count": int(
                                fresh_action_seed_reset_count
                            ),
                            "post_recovery_act_bridge_fresh_action_seed_source": fresh_action_seed_source,
                        }
                    )

                force_live_h_monitor_for_contact_rich = bool(args.condition == "chunk_deform")
                live_h_monitor_skipped = not bool(args.live_h_monitor or force_live_h_monitor_for_contact_rich)
                h_attribution_info = {}
                current_h_violation_parts: set[str] = set()
                horizon_violation_parts: set[str] = set()
                live_h_monitor_info = {}
                live_min_clearance = None
                if live_h_monitor_skipped:
                    min_h = None
                    h_values = None
                    h_violation = False
                    h_pair_label = None
                    monitor_time_ms = 0.0
                else:
                    monitor_t0 = time.perf_counter()
                    min_h, h_values, h_violation, h_pair_label, live_h_monitor_info = compute_oscbf_full_arm_h_monitor(
                        filt=oscbf,
                        env=safety_runtime_env,
                        obs=obs,
                        q_full=q_full,
                        qd_full=qd_full,
                        clearance_threshold=0.0,
                        return_details=True,
                    )
                    live_min_clearance = _safe_info_get(live_h_monitor_info, "live_min_clearance")
                    monitor_time_ms = 1000.0 * (time.perf_counter() - monitor_t0)
                    if h_values is not None:
                        h_attribution_info = _h_argmin_metadata(h_values)
                        if h_pair_label is not None:
                            h_attribution_info.update(_h_pair_label_metadata(h_pair_label))
                        robot_part, _, _, _ = _parse_h_pair_label(h_pair_label)
                        if h_violation and robot_part is not None:
                            current_h_violation_parts = {robot_part}

                filter_t0 = time.perf_counter()
                safety_info = {}
                chunk_trace_context = None
                nominal_rollout_diagnostic_context = None
                nominal_rollout_post_step_state = None
                nominal_pred_q_next_for_feedback = None
                nominal_rollout_shape_for_feedback = None
                nominal_rollout_feedback_error = None
                rollout_feedback = None
                pacs_background_safety_info = None
                pacs_background_chunk_for_metrics = None

                if chunk_filter_mode:
                    chunk_obs = _chunk_obs_with_q(obs, q_full)
                    horizon_operator.set_context(safety_runtime_env, chunk_obs, q_full, qd_full)
                    nominal_chunk, was_single_chunk = _as_chunk(env_action)
                    chunk_trace_context = {
                        "obs": chunk_obs,
                        "nominal_chunk": np.asarray(nominal_chunk, dtype=np.float32).copy(),
                    }
                    if (
                        nominal_rollout_diagnostic_logging_enabled
                        and len(all_nominal_rollout_diagnostic_records) < max(0, nominal_rollout_diagnostics_max_events)
                    ):
                        try:
                            nominal_q_seq_feedback = np.asarray(
                                safechunk.deform.rollout_nominal_chunk(chunk_obs, nominal_chunk),
                                dtype=np.float32,
                            )
                            nominal_rollout_shape_for_feedback = list(nominal_q_seq_feedback.shape)
                            nominal_pred_q_next_for_feedback = _first_rollout_state(nominal_q_seq_feedback)
                        except Exception as exc:  # noqa: BLE001
                            nominal_rollout_feedback_error = repr(exc)
                        nominal_rollout_diagnostic_context = {
                            "episode": int(episode),
                            "step": int(step),
                            "q_before": np.asarray(q_full, dtype=np.float32).reshape(-1).copy(),
                            "qd_before": np.asarray(qd_full, dtype=np.float32).reshape(-1).copy(),
                            "nominal_first_action": np.asarray(nominal_chunk[0], dtype=np.float32).reshape(-1).copy(),
                            "nominal_action_shape": list(np.asarray(nominal_chunk).shape),
                            "nominal_rollout_shape": nominal_rollout_shape_for_feedback,
                            "nominal_pred_q_next": nominal_pred_q_next_for_feedback,
                        }
                        if nominal_rollout_feedback_error is not None:
                            nominal_rollout_diagnostic_context["nominal_rollout_error"] = nominal_rollout_feedback_error
                    if ablation_pure_act_resume_steps_left > 0:
                        pure_steps_left_before = int(ablation_pure_act_resume_steps_left)
                        pure_step_index = int(
                            max(0, ablation_pure_act_resume_total_steps - pure_steps_left_before)
                        )
                        safe_chunk = np.asarray(nominal_chunk, dtype=np.float32).copy()
                        safety_info = {
                            "safety_mode": "ablation_pure_act_resume",
                            "mode": "ablation_pure_act_resume",
                            "deformation_source": "ablation_pure_act_resume",
                            "deformation_norm": 0.0,
                            "retiming_source": None,
                            "retiming_norm": 0.0,
                            "suppress_outer_pause": True,
                            "ablation_pure_act_resume_enabled": bool(ablation_pure_act_resume_total_steps > 0),
                            "ablation_pure_act_resume_active": True,
                            "ablation_pure_act_resume_step_index": pure_step_index,
                            "ablation_pure_act_resume_steps_left": pure_steps_left_before,
                            "ablation_pure_act_resume_total_steps": int(ablation_pure_act_resume_total_steps),
                        }
                        ablation_pure_act_resume_steps_left = max(0, pure_steps_left_before - 1)
                    elif use_precomputed_step0_filter and args.condition == "chunk_deform":
                        safe_chunk = np.asarray(
                            precomputed_step0_filter["safe_chunk"],
                            dtype=np.float32,
                        ).copy()
                        safety_info = dict(precomputed_step0_filter["safety_info"])
                        safety_info.update(
                            {
                                "precomputed_step0_filter_used": True,
                                "precomputed_step0_filter_time_ms": 0.0,
                            }
                        )
                        precomputed_step0_filter = None
                    elif args.condition in {"sequential", "sequential_oscbf"}:
                        safe_chunk, safety_info = safechunk.deform.deform_chunk_with_oscbf(
                            chunk_obs,
                            nominal_chunk,
                            env=safety_runtime_env,
                            q_full=q_full,
                            qd_full=qd_full,
                        )
                        safety_info = dict(safety_info)
                        safety_info.update({
                            "safety_mode": "sequential_oscbf",
                            "mode": "sequential_oscbf",
                            "deformation_source": "sequential_oscbf",
                        })
                    else:
                        handoff_history_snapshot_before = _action_sequence_history_snapshot(env)
                        pre_filter_contact_pairs = robot_human_contact_pairs(safety_runtime_env)
                        pre_filter_contact_count = (
                            None
                            if pre_filter_contact_pairs is None
                            else len(pre_filter_contact_pairs)
                        )
                        safe_chunk, safety_info = safechunk.filter_chunk(
                            chunk_obs,
                            nominal_chunk,
                            env=safety_runtime_env,
                            q_full=q_full,
                            qd_full=qd_full,
                            task_progress=task_state_before.get("task_progress"),
                            resume_affordance_context=resume_affordance_context,
                            live_monitor_min_h=min_h,
                            live_monitor_min_clearance=live_min_clearance,
                            live_monitor_h_violation=h_violation,
                            live_monitor_available=bool(live_min_clearance is not None),
                            robot_human_contact_count=pre_filter_contact_count,
                            human_phase=_safe_info_get(blocker_info, "human_phase"),
                            goal_region_human_distance=_safe_info_get(blocker_info, "goal_region_human_distance"),
                            goal_region_blocked=_safe_info_get(blocker_info, "goal_region_blocked"),
                        )
                        handoff_history_snapshot_after = _action_sequence_history_snapshot(env)
                        if _safe_info_get(safety_info, "mpc_handoff_attempted"):
                            safety_info = dict(safety_info)
                            safety_info.update(_action_sequence_history_snapshot_delta(handoff_history_snapshot_before, handoff_history_snapshot_after))
                    if (
                        args.condition == "path_consistent_brake"
                        and path_consistent_background_check_only
                    ):
                        pacs_background_safety_info = dict(safety_info)
                        pacs_background_chunk_for_metrics = np.asarray(
                            safe_chunk,
                            dtype=np.float32,
                        ).copy()
                        pacs_background_mode = (
                            _safe_info_get(pacs_background_safety_info, "safety_mode")
                            or _safe_info_get(pacs_background_safety_info, "mode")
                        )
                        pacs_background_source = _safe_info_get(
                            pacs_background_safety_info,
                            "deformation_source",
                        )
                        pacs_background_retiming_source = _safe_info_get(
                            pacs_background_safety_info,
                            "retiming_source",
                        )
                        if pacs_background_retiming_source is None:
                            pacs_background_retiming_source = pacs_background_source
                        safety_info = dict(safety_info)
                        safety_info.update(
                            {
                                "pacs_background_check_only": True,
                                "pacs_background_safety_mode": pacs_background_mode,
                                "pacs_background_deformation_source": pacs_background_source,
                                "pacs_background_retiming_source": pacs_background_retiming_source,
                                "pacs_background_min_clearance": _safe_info_get(
                                    pacs_background_safety_info,
                                    "min_clearance",
                                ),
                                "pacs_background_first_violation": _safe_info_get(
                                    pacs_background_safety_info,
                                    "first_violation",
                                ),
                                "pacs_background_unsafe_count": _safe_info_get(
                                    pacs_background_safety_info,
                                    "unsafe_count",
                                ),
                                "safety_mode": "pass_through",
                                "mode": "pass_through",
                                "deformation_source": None,
                                "deformation_norm": 0.0,
                                "retiming_source": None,
                                "retiming_norm": 0.0,
                                "suppress_outer_pause": True,
                            }
                        )
                        safe_chunk = np.asarray(nominal_chunk, dtype=np.float32).copy()

                    if live_h_monitor_skipped:
                        (
                            fallback_min_h,
                            fallback_h_values,
                            fallback_h_violation,
                            fallback_live_min_clearance,
                            fallback_h_source,
                        ) = _chunk_horizon_h_monitor_fallback(
                            safety_info,
                            0.0,
                        )
                        if fallback_min_h is not None or fallback_live_min_clearance is not None:
                            min_h = fallback_min_h
                            h_values = fallback_h_values
                            h_violation = bool(fallback_h_violation)
                            live_min_clearance = fallback_live_min_clearance
                            safety_info["h_monitor_source"] = (
                                "chunk_horizon_signed_clearance"
                                if fallback_live_min_clearance is not None
                                else "chunk_horizon_raw_h_debug_only"
                            )
                            safety_info["live_h_violation_source"] = f"fallback_{fallback_h_source}"
                            safety_info["live_h_violation_threshold"] = 0.0
                            if fallback_live_min_clearance is not None:
                                safety_info["live_min_clearance"] = fallback_live_min_clearance

                    if live_h_monitor_info:
                        safety_info.update({
                            key: value
                            for key, value in live_h_monitor_info.items()
                            if value is not None
                        })

                    should_compute_horizon_parts = False
                    if not bool(_safe_info_get(safety_info, "precomputed_step0_filter_used")):
                        min_clearances = _safe_info_get(safety_info, "min_clearances")
                        try:
                            min_clearances_array = np.asarray(min_clearances, dtype=np.float32)
                            if min_clearances_array.size > 0 and np.isfinite(min_clearances_array).any() and np.any(
                                min_clearances_array < 0.0
                            ):
                                should_compute_horizon_parts = True
                        except Exception:  # noqa: BLE001
                            if h_violation:
                                should_compute_horizon_parts = True

                    if should_compute_horizon_parts:
                        try:
                            planned_q_seq = np.asarray(
                                safechunk.intervention.rollout_nominal_chunk(chunk_obs, nominal_chunk),
                                dtype=np.float32,
                            )
                            horizon_h_result = _compute_full_arm_horizon_h_values(
                                horizon_operator,
                                chunk_obs,
                                planned_q_seq,
                            )
                            if horizon_h_result is not None:
                                horizon_h_values, horizon_h_labels = horizon_h_result
                                horizon_violation_parts = _parts_from_horizon_h_values(
                                    horizon_h_values,
                                    horizon_h_labels,
                                )
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("Could not compute horizon robot-part h violations: %s", exc)

                    safe_env_action = _restore_action_shape(
                        np.asarray(safe_chunk, dtype=np.float32),
                        was_single_chunk,
                    )
                    safe_first_action = extract_first_action(safe_env_action)
                elif args.condition == "oscbf":
                    safe_first_action = oscbf(
                        action=first_action,
                        env=safety_runtime_env,
                        observations=obs,
                        q_full=q_full,
                        qd_full=qd_full,
                    )
                    safe_env_action = replace_first_action(
                        env_action=env_action,
                        safe_first_action=safe_first_action,
                    )
                else:
                    safe_first_action = first_action.copy()
                    safe_env_action = env_action.copy()

                if chunk_filter_mode:
                    pass_through_idx = np.asarray(non_arm_idx, dtype=np.int64)
                    pass_through_idx = pass_through_idx[
                        (pass_through_idx >= 0)
                        & (pass_through_idx < extract_first_action(safe_env_action).shape[0])
                    ]
                    chunk_passthrough_restore_delta = 0.0
                    if pass_through_idx.size:
                        restored_action = np.asarray(safe_env_action, dtype=np.float32).copy()
                        reference_action = np.asarray(env_action, dtype=np.float32)
                        reference_first_action = extract_first_action(reference_action)
                        if restored_action.ndim == 1:
                            before = restored_action[pass_through_idx].copy()
                            restored_action[pass_through_idx] = reference_first_action[pass_through_idx]
                            after = restored_action[pass_through_idx]
                        else:
                            before = restored_action[:, pass_through_idx].copy()
                            if reference_action.ndim == 2 and reference_action.shape[0] == restored_action.shape[0]:
                                restored_action[:, pass_through_idx] = reference_action[:, pass_through_idx]
                            else:
                                restored_action[:, pass_through_idx] = reference_first_action[pass_through_idx][None, :]
                            after = restored_action[:, pass_through_idx]
                        chunk_passthrough_restore_delta = float(np.linalg.norm(after - before))
                        safe_env_action = restored_action
                        safe_first_action = extract_first_action(safe_env_action)
                    safety_info = dict(safety_info)
                    safety_info.update(
                        {
                            "chunk_passthrough_restored_indices": pass_through_idx.tolist(),
                            "chunk_passthrough_restore_delta_norm": chunk_passthrough_restore_delta,
                        }
                    )

                if replay_actions is None and _is_brake_or_fallback_execution(safety_info):
                    pre_live_hold_action = np.asarray(safe_env_action, dtype=np.float32).copy()
                    safe_env_action, brake_live_hold_indices, brake_live_hold_delta = (
                        _hard_hold_action_from_live_robot(
                            env,
                            safe_env_action,
                        )
                    )
                    pass_through_idx = np.asarray(non_arm_idx, dtype=np.int64)
                    pass_through_idx = pass_through_idx[
                        (pass_through_idx >= 0)
                        & (pass_through_idx < extract_first_action(safe_env_action).shape[0])
                    ]
                    pass_through_restore_delta = 0.0
                    if pass_through_idx.size:
                        restored_action = np.asarray(safe_env_action, dtype=np.float32).copy()
                        if restored_action.ndim == 1:
                            before = restored_action[pass_through_idx].copy()
                            restored_action[pass_through_idx] = pre_live_hold_action[pass_through_idx]
                            after = restored_action[pass_through_idx]
                        else:
                            before = restored_action[:, pass_through_idx].copy()
                            restored_action[:, pass_through_idx] = pre_live_hold_action[:, pass_through_idx]
                            after = restored_action[:, pass_through_idx]
                        pass_through_restore_delta = float(np.linalg.norm(after - before))
                        safe_env_action = restored_action
                    safe_first_action = extract_first_action(safe_env_action)
                    safety_info = dict(safety_info)
                    safety_info.update(
                        {
                            "brake_live_robot_hold_current": bool(brake_live_hold_indices),
                            "brake_live_robot_hold_action_indices": brake_live_hold_indices,
                            "brake_live_robot_hold_delta_norm": float(brake_live_hold_delta),
                            "brake_passthrough_restored_indices": pass_through_idx.tolist(),
                            "brake_passthrough_restore_delta_norm": pass_through_restore_delta,
                        }
                    )

                policy_hold_active = False
                if args.pause_policy_step_on_brake and replay_actions is None:
                    policy_hold_active = _should_hold_policy_step(
                        safety_info,
                        first_action,
                        safe_first_action,
                        arm_idx,
                        args.intervention_eps,
                    )

                phase_for_pause_restart = _safe_info_get(blocker_info, "human_phase")
                pause_restart_active = (
                    args.pause_and_restart_on_human_blocker
                    and phase_for_pause_restart in {"enter", "hold", "exit"}
                )
                initial_pause_restart_active = (
                    args.initial_pause_restart_steps > 0
                    and step < args.initial_pause_restart_steps
                )
                current_pause_clearance = live_min_clearance if live_min_clearance is not None else min_h
                pause_active, pause_reason = _should_pause_for_safety(args, current_pause_clearance, safety_info)
                if bool(_safe_info_get(safety_info, "suppress_outer_pause")):
                    pause_active = False
                    pause_reason = None
                if pause_restart_active or initial_pause_restart_active:
                    pause_active = True
                    pause_reason = (
                        "initial_pause_restart"
                        if initial_pause_restart_active
                        else "human_blocker_pause_restart"
                    )
                if pause_active:
                    pause_action_idx = arm_idx
                    pause_state_idx = state_idx
                    if pause_restart_active or initial_pause_restart_active:
                        pause_action_idx = getattr(
                            oscbf,
                            "bigym_action_safety_indices",
                            arm_idx,
                        )
                        pause_state_idx = getattr(
                            oscbf,
                            "bigym_state_safety_indices",
                            state_idx,
                        )
                    safe_env_action = _scale_controlled_motion_from_current_q(
                        safe_env_action,
                        q_full,
                        pause_action_idx,
                        pause_state_idx,
                        args.pause_motion_scale,
                    )
                    safe_first_action = extract_first_action(safe_env_action)
                    safety_info = dict(safety_info)
                    safety_info.update(
                        {
                            "safety_mode": (
                                "pause_and_restart"
                                if pause_restart_active
                                else "pause_on_unsafe"
                            ),
                            "mode": (
                                "pause_and_restart"
                                if pause_restart_active
                                else "pause_on_unsafe"
                            ),
                            "pause_reason": pause_reason,
                        }
                    )
                    if args.pause_policy_step_on_brake and replay_actions is None:
                        policy_hold_active = True

                filter_time_ms = 1000.0 * (time.perf_counter() - filter_t0)

                assertion_idx = arm_idx
                brake_hold_action_indices = _safe_info_get(safety_info, "brake_hold_action_indices")
                brake_gripper_hold_indices = _safe_info_get(
                    safety_info,
                    "brake_gripper_hold_action_indices",
                )
                brake_live_hold_indices = _safe_info_get(
                    safety_info,
                    "brake_live_robot_hold_action_indices",
                )
                brake_assertion_indices = []
                if brake_hold_action_indices is not None:
                    brake_assertion_indices.extend(brake_hold_action_indices)
                if brake_gripper_hold_indices is not None:
                    brake_assertion_indices.extend(brake_gripper_hold_indices)
                if brake_live_hold_indices is not None:
                    brake_assertion_indices.extend(brake_live_hold_indices)
                if brake_assertion_indices:
                    assertion_idx = np.unique(
                        np.asarray(brake_assertion_indices, dtype=np.int64)
                    )
                if pause_active and (pause_restart_active or initial_pause_restart_active):
                    assertion_idx = pause_action_idx
                if chunk_filter_mode:
                    _assert_chunk_properties(
                        env_action,
                        safe_env_action,
                        assertion_idx,
                    )
                else:
                    assert_action_properties(
                        nominal_action=first_action,
                        safe_action=safe_first_action,
                        arm_indices=assertion_idx,
                    )

                if (args.phase_reanchor or args.post_recovery_task_guard) and replay_actions is None:
                    if phase_reanchor_steps_left <= 0 and args.phase_reanchor:
                        should_start_reanchor, phase_reanchor_drawer_progress = _should_start_phase_reanchor(
                            args,
                            step,
                            phase_reanchor_state,
                            phase_reanchor_drawer_history,
                            phase_reanchor_cooldown_left,
                        )
                        if should_start_reanchor:
                            phase_reanchor_steps_left = int(args.phase_reanchor_steps)
                            phase_reanchor_best_target_distance = None
                            phase_reanchor_best_target_signature = None
                            phase_reanchor_best_control_distance = None
                            phase_reanchor_best_control_signature = None
                            phase_reanchor_taskspace_worsen_count = 0
                            phase_reanchor_suppress_q_servo_steps_left = 0
                            phase_reanchor_live_extension_count = 0
                            phase_reanchor_bridge_preload_start_progress = None
                            phase_reanchor_bridge_preload_count = 0
                            phase_reanchor_bridge_preload_validated_latched = False
                            phase_reanchor_bridge_preload_reason_last = None
                            phase_reanchor_bridge_preload_progress_delta_last = None
                            phase_reanchor_bridge_preload_handle_dist_last = None
                            reset_count = _reset_action_sequence_history(env)
                            if policy_env is not None:
                                reset_count += _reset_action_sequence_history(policy_env)
                            if hasattr(safechunk, "reset"):
                                safechunk.reset()
                            policy_step = 0
                            if episode == 0 or args.debug:
                                print(
                                    "phase_reanchor: start "
                                    f"episode={episode} step={step} "
                                    f"phase={phase_reanchor_state.get('phase')} "
                                    f"drawer_fraction={phase_reanchor_state.get('drawer_open_fraction'):.3f} "
                                    f"window_progress={phase_reanchor_drawer_progress:.3f} "
                                    f"reset_wrappers={reset_count}"
                                )

                    if phase_reanchor_steps_left > 0:
                        if (
                            phase_reanchor_state is None
                            or phase_reanchor_state.get("phase") == "done"
                        ):
                            phase_reanchor_steps_left = 0
                            phase_reanchor_cooldown_left = int(args.phase_reanchor_cooldown_steps)
                            phase_reanchor_reset_after_step = True
                        else:
                            phase_reanchor_action_state = phase_reanchor_state
                            live_taskspace_distance = None
                            live_taskspace_guard_active = bool(
                                getattr(args, "phase_reanchor_live_taskspace_guard", True)
                            )
                            live_taskspace_suppress_q_servo = False
                            live_taskspace_suppress_q_servo_reason = None
                            live_taskspace_stop_requested = False
                            live_taskspace_stop_reason = None
                            phase_reanchor_budget_hint = max(
                                1,
                                int(getattr(args, "phase_reanchor_steps", 1)),
                                int(getattr(args, "post_recovery_task_guard_steps", 1)),
                            )
                            live_taskspace_elapsed_steps = max(
                                0,
                                phase_reanchor_budget_hint - int(phase_reanchor_steps_left),
                            )
                            live_taskspace_distance_source = None
                            if phase_reanchor_state is not None:
                                for _distance_key, _distance_source in (
                                    ("ee_to_target_dist", "phase_target"),
                                    ("ee_to_handle_dist", "handle"),
                                ):
                                    try:
                                        live_taskspace_distance = float(
                                            phase_reanchor_state.get(_distance_key)
                                        )
                                        if np.isfinite(live_taskspace_distance):
                                            live_taskspace_distance_source = _distance_source
                                            break
                                    except (TypeError, ValueError):
                                        live_taskspace_distance = None
                            if (
                                live_taskspace_guard_active
                                and live_taskspace_distance is not None
                                and np.isfinite(live_taskspace_distance)
                            ):
                                worsen_tolerance = float(
                                    getattr(
                                        args,
                                        "phase_reanchor_live_taskspace_worsen_tolerance",
                                        0.005,
                                    )
                                )
                                worsen_patience = max(
                                    1,
                                    int(
                                        getattr(
                                            args,
                                            "phase_reanchor_live_taskspace_worsen_patience",
                                            1,
                                        )
                                    ),
                                )
                                guard_disable_steps = max(
                                    1,
                                    int(
                                        getattr(
                                            args,
                                            "phase_reanchor_live_taskspace_guard_disable_steps",
                                            6,
                                        )
                                    ),
                                )
                                taskspace_ready_dist = float(
                                    getattr(args, "phase_reanchor_grasp_dist", 0.12)
                                )
                                control_taskspace_distance = None
                                control_taskspace_distance_source = None
                                for _control_key, _control_source in (
                                    ("control_ee_to_target_dist", "control_phase_target"),
                                    ("control_ee_to_handle_dist", "control_handle"),
                                ):
                                    try:
                                        control_taskspace_distance = float(
                                            phase_reanchor_state.get(_control_key)
                                        )
                                        if np.isfinite(control_taskspace_distance):
                                            control_taskspace_distance_source = _control_source
                                            break
                                    except (TypeError, ValueError):
                                        control_taskspace_distance = None
                                control_distance_worsening = None
                                phase_signature = str(phase_reanchor_state.get("phase", "unknown"))
                                target_signature = (phase_signature, live_taskspace_distance_source)
                                if phase_reanchor_best_target_signature != target_signature:
                                    phase_reanchor_best_target_distance = None
                                    phase_reanchor_taskspace_worsen_count = 0
                                    phase_reanchor_best_target_signature = target_signature
                                control_signature = (phase_signature, control_taskspace_distance_source)
                                if phase_reanchor_best_control_signature != control_signature:
                                    phase_reanchor_best_control_distance = None
                                    phase_reanchor_best_control_signature = control_signature
                                if control_taskspace_distance is not None:
                                    if (
                                        phase_reanchor_best_control_distance is None
                                        or control_taskspace_distance
                                        < phase_reanchor_best_control_distance - 0.5 * worsen_tolerance
                                    ):
                                        phase_reanchor_best_control_distance = float(control_taskspace_distance)
                                        control_distance_worsening = False
                                    else:
                                        control_distance_worsening = bool(
                                            phase_reanchor_best_control_distance is not None
                                            and control_taskspace_distance
                                            > phase_reanchor_best_control_distance + worsen_tolerance
                                        )
                                if (
                                    phase_reanchor_best_target_distance is None
                                    or live_taskspace_distance
                                    < phase_reanchor_best_target_distance - 0.5 * worsen_tolerance
                                ):
                                    phase_reanchor_best_target_distance = float(live_taskspace_distance)
                                    phase_reanchor_taskspace_worsen_count = 0
                                elif (
                                    phase_reanchor_best_target_distance is not None
                                    and live_taskspace_distance
                                    > phase_reanchor_best_target_distance + worsen_tolerance
                                    and live_taskspace_distance > taskspace_ready_dist
                                ):
                                    if control_distance_worsening is False:
                                        phase_reanchor_taskspace_worsen_count = 0
                                    else:
                                        phase_reanchor_taskspace_worsen_count += 1
                                else:
                                    phase_reanchor_taskspace_worsen_count = 0
                                if phase_reanchor_taskspace_worsen_count >= worsen_patience:
                                    release_target_limit = float(
                                        getattr(args, "phase_reanchor_live_release_target_error", 0.16)
                                    )
                                    release_handle_limit = float(
                                        getattr(args, "phase_reanchor_live_release_handle_dist", 0.24)
                                    )
                                    if live_taskspace_distance_source == "handle":
                                        stop_ready_dist = max(taskspace_ready_dist, release_handle_limit + 0.12)
                                    else:
                                        stop_ready_dist = max(taskspace_ready_dist, release_target_limit + 0.12)
                                    stop_near_release_band = bool(
                                        phase_reanchor_best_target_distance is not None
                                        and phase_reanchor_best_target_distance <= stop_ready_dist
                                    )
                                    if (
                                        bool(getattr(args, "phase_reanchor_live_taskspace_stop_on_worsening", False))
                                        and live_taskspace_elapsed_steps >= int(getattr(args, "phase_reanchor_live_taskspace_stop_min_steps", 12))
                                        and stop_near_release_band
                                    ):
                                        live_taskspace_stop_requested = True
                                        live_taskspace_stop_reason = "live_taskspace_worsening_near_release"
                                        phase_reanchor_steps_left = min(phase_reanchor_steps_left, 1)
                                    elif bool(getattr(args, "phase_reanchor_live_taskspace_stop_on_worsening", False)):
                                        live_taskspace_stop_reason = "live_taskspace_worsening_far_from_release"
                                    phase_reanchor_suppress_q_servo_steps_left = max(
                                        phase_reanchor_suppress_q_servo_steps_left,
                                        guard_disable_steps,
                                    )
                                    phase_reanchor_taskspace_worsen_count = 0
                            if (
                                bool(getattr(args, "phase_reanchor_suppress_q_servo_far_target", False))
                                and live_taskspace_distance is not None
                                and np.isfinite(live_taskspace_distance)
                                and live_taskspace_distance
                                > float(getattr(args, "phase_reanchor_q_servo_enable_target_dist", 0.24))
                            ):
                                phase_reanchor_suppress_q_servo_steps_left = max(
                                    phase_reanchor_suppress_q_servo_steps_left,
                                    1,
                                )
                                live_taskspace_suppress_q_servo_reason = "live_target_far"
                            if phase_reanchor_suppress_q_servo_steps_left > 0:
                                live_taskspace_suppress_q_servo = True
                                if live_taskspace_suppress_q_servo_reason is None:
                                    live_taskspace_suppress_q_servo_reason = "live_taskspace_worsening"
                                phase_reanchor_suppress_q_servo_steps_left -= 1
                            selected_nominal_window = _select_phase_reanchor_nominal_window(
                                phase_reanchor_nominal_windows,
                                q_full,
                                oscbf,
                                phase_reanchor_state,
                                args,
                            )
                            if selected_nominal_window is not None:
                                phase_reanchor_action_state = dict(phase_reanchor_state)
                                phase_reanchor_action_state.update(
                                    {
                                        "nominal_reentry_q_window": selected_nominal_window.get("q_window"),
                                        "nominal_reentry_action_window": selected_nominal_window.get("action_window"),
                                        "nominal_reentry_source": selected_nominal_window.get("source"),
                                        "nominal_reentry_episode": selected_nominal_window.get("episode"),
                                        "nominal_reentry_start_step": selected_nominal_window.get("start_step"),
                                        "nominal_reentry_end_step": selected_nominal_window.get("end_step"),
                                        "nominal_reentry_window_steps": selected_nominal_window.get("steps"),
                                        "nominal_reentry_current_q_l2": selected_nominal_window.get("current_q_l2"),
                                        "nominal_reentry_selection": selected_nominal_window.get("selection"),
                                        "nominal_reentry_selection_reason": selected_nominal_window.get("selection_reason"),
                                        "nominal_reentry_live_target_distance": selected_nominal_window.get("live_target_distance"),
                                        "nominal_reentry_suppress_q_servo": live_taskspace_suppress_q_servo,
                                        "nominal_reentry_suppress_q_servo_reason": live_taskspace_suppress_q_servo_reason,
                                        "live_taskspace_guard_distance": live_taskspace_distance,
                                        "live_taskspace_guard_best_distance": phase_reanchor_best_target_distance,
                                    }
                                )
                            elif isinstance(phase_reanchor_action_state, dict):
                                phase_reanchor_action_state = dict(phase_reanchor_action_state)
                                phase_reanchor_action_state.update(
                                    {
                                        "nominal_reentry_suppress_q_servo": live_taskspace_suppress_q_servo,
                                        "nominal_reentry_suppress_q_servo_reason": live_taskspace_suppress_q_servo_reason,
                                        "live_taskspace_guard_distance": live_taskspace_distance,
                                        "live_taskspace_guard_best_distance": phase_reanchor_best_target_distance,
                                    }
                                )
                            reanchor_action, reanchor_info = _phase_reanchor_action(
                                env,
                                safe_env_action,
                                q_full,
                                oscbf,
                                args,
                                phase_reanchor_action_state,
                            )
                            if reanchor_action is None:
                                phase_reanchor_steps_left = 0
                                phase_reanchor_cooldown_left = int(args.phase_reanchor_cooldown_steps)
                                if episode == 0 or args.debug:
                                    print(
                                        "phase_reanchor: unavailable "
                                        f"episode={episode} step={step}"
                                    )
                            else:
                                reanchor_to_execute = reanchor_action
                                reanchor_acceptance = None
                                reanchor_accepted = True
                                if (
                                    args.post_recovery_task_guard_check_safety
                                    and chunk_filter_mode
                                    and hasattr(safechunk, "evaluate_candidate_acceptance")
                                ):
                                    try:
                                        reanchor_acceptance = safechunk.intervention.evaluate_candidate_acceptance(
                                            _chunk_obs_with_q(obs, q_full),
                                            reanchor_action,
                                            "deform",
                                        )
                                        reanchor_accepted = bool(
                                            reanchor_acceptance.get("accepted")
                                        )
                                        if (
                                            reanchor_accepted
                                            and reanchor_acceptance.get("safe_prefix_execution")
                                            and hasattr(safechunk, "_truncate_chunk_to_safe_prefix")
                                        ):
                                            reanchor_to_execute = safechunk.intervention._truncate_chunk_to_safe_prefix(
                                                reanchor_action,
                                                reanchor_acceptance,
                                            )
                                    except Exception as exc:  # noqa: BLE001
                                        reanchor_accepted = False
                                        reanchor_acceptance = {
                                            "rejection_reason": f"acceptance_error:{type(exc).__name__}",
                                        }

                                if not reanchor_accepted:
                                    phase_reanchor_steps_left = 0
                                    phase_reanchor_cooldown_left = int(args.phase_reanchor_cooldown_steps)
                                    phase_reanchor_reset_after_step = True
                                    safety_info = dict(safety_info)
                                    safety_info.update(
                                        {
                                            "phase_reanchor_rejected": True,
                                            "phase_reanchor_reject_reason": (
                                                reanchor_acceptance or {}
                                            ).get("rejection_reason"),
                                            "phase_reanchor_acceptance_type": (
                                                reanchor_acceptance or {}
                                            ).get("acceptance_type"),
                                            "phase_reanchor_immediate_clearance": (
                                                reanchor_acceptance or {}
                                            ).get("immediate_clearance"),
                                            "phase_reanchor_horizon_min_clearance": (
                                                reanchor_acceptance or {}
                                            ).get("horizon_min_clearance"),
                                        }
                                    )
                                else:
                                    safe_env_action = reanchor_to_execute
                                    safe_first_action = extract_first_action(safe_env_action)
                                    phase_reanchor_steps_left -= 1
                                    phase_reanchor_live_release_ready = None
                                    phase_reanchor_live_release_reason = None
                                    phase_reanchor_live_release_target_error = None
                                    phase_reanchor_live_release_handle_dist = None
                                    phase_reanchor_live_extension_started = False
                                    phase_reanchor_live_extension_budget_exhausted = False
                                    phase_reanchor_early_release_triggered = False
                                    phase_reanchor_early_release_reason = None
                                    phase_reanchor_early_release_arm_q_error = None
                                    phase_reanchor_bridge_contact_ready = None
                                    phase_reanchor_bridge_contact_reason = None
                                    phase_reanchor_bridge_contact_handle_dist = None
                                    phase_reanchor_bridge_contact_handle_limit = None
                                    phase_reanchor_bridge_preload_validated = None
                                    phase_reanchor_bridge_preload_reason = None
                                    phase_reanchor_bridge_preload_steps = None
                                    phase_reanchor_bridge_preload_progress_delta = None
                                    phase_reanchor_bridge_preload_handle_dist = None
                                    phase_reanchor_bridge_preload_handle_limit = None
                                    phase_reanchor_bridge_preload_progress_ok = None
                                    phase_reanchor_bridge_preload_handle_ok = None
                                    phase_reanchor_bridge_preload_validation_source = None
                                    phase_reanchor_bridge_preload_progress_abs = None
                                    if bool(getattr(args, "phase_reanchor_early_release_on_resumable_window", False)):
                                        try:
                                            phase_reanchor_early_release_arm_q_error = reanchor_info.get(
                                                "arm_servo_target_window_score"
                                            )
                                            if phase_reanchor_early_release_arm_q_error is None:
                                                phase_reanchor_early_release_arm_q_error = reanchor_info.get(
                                                    "arm_servo_error_norm"
                                                )
                                            if phase_reanchor_early_release_arm_q_error is not None:
                                                phase_reanchor_early_release_arm_q_error = float(
                                                    phase_reanchor_early_release_arm_q_error
                                                )
                                                if not np.isfinite(phase_reanchor_early_release_arm_q_error):
                                                    phase_reanchor_early_release_arm_q_error = None
                                        except (TypeError, ValueError):
                                            phase_reanchor_early_release_arm_q_error = None
                                        early_release_arm_limit = float(
                                            getattr(args, "phase_reanchor_early_release_arm_q_error", 0.12)
                                        )
                                        early_release_min_steps = int(
                                            getattr(args, "phase_reanchor_early_release_min_steps", 8)
                                        )
                                        early_release_elapsed = int(live_taskspace_elapsed_steps)
                                        (
                                            early_release_live_ready,
                                            early_release_live_reason,
                                            _early_release_target_error,
                                            _early_release_handle_dist,
                                        ) = _phase_reanchor_live_release_status(
                                            phase_reanchor_action_state,
                                            args,
                                        )
                                        early_release_resume_ready = bool(
                                            _safe_info_get(safety_info, "resume_affordance_ok")
                                        )
                                        if (
                                            phase_reanchor_early_release_arm_q_error is None
                                            and isinstance(phase_reanchor_action_state, dict)
                                        ):
                                            try:
                                                q_window = phase_reanchor_action_state.get("nominal_reentry_q_window")
                                                q_rows = np.asarray(q_window, dtype=np.float64)
                                                if q_rows.ndim == 1:
                                                    q_rows = q_rows.reshape(1, -1)
                                                elif q_rows.ndim > 2:
                                                    q_rows = q_rows.reshape((-1, q_rows.shape[-1]))
                                                live_q = np.asarray(q_full, dtype=np.float64).reshape(-1)
                                                dim = min(live_q.size, q_rows.shape[-1])
                                                if dim > 0 and np.isfinite(q_rows[:, :dim]).all() and np.isfinite(live_q[:dim]).all():
                                                    compare_idx = np.arange(dim, dtype=np.int64)
                                                    if not bool(getattr(args, "phase_reanchor_nominal_window_track_base", True)):
                                                        arm_state_idx = np.asarray(
                                                            getattr(oscbf, "bigym_state_arm_indices", []),
                                                            dtype=np.int64,
                                                        )
                                                        arm_state_idx = arm_state_idx[
                                                            (arm_state_idx >= 0) & (arm_state_idx < dim)
                                                        ]
                                                        if arm_state_idx.size > 0:
                                                            compare_idx = arm_state_idx
                                                    adapted_q_dists = np.linalg.norm(
                                                        q_rows[:, compare_idx] - live_q[compare_idx][None, :],
                                                        axis=1,
                                                    )
                                                    phase_reanchor_early_release_arm_q_error = float(adapted_q_dists[-1])
                                            except Exception as exc:  # noqa: BLE001
                                                logger.debug("Could not compute early-release nominal-q distance: %s", exc)
                                        early_release_arm_ready = bool(
                                            phase_reanchor_early_release_arm_q_error is not None
                                            and phase_reanchor_early_release_arm_q_error <= early_release_arm_limit
                                        )
                                        (
                                            phase_reanchor_bridge_contact_ready,
                                            phase_reanchor_bridge_contact_reason,
                                            phase_reanchor_bridge_contact_handle_dist,
                                            phase_reanchor_bridge_contact_handle_limit,
                                        ) = _phase_reanchor_bridge_contact_status(
                                            phase_reanchor_action_state,
                                            args,
                                        )
                                        preload_enabled = bool(
                                            getattr(args, "phase_reanchor_bridge_preload_validation", False)
                                        )
                                        phase_reanchor_bridge_preload_validated = not preload_enabled
                                        phase_reanchor_bridge_preload_reason = (
                                            "disabled" if not preload_enabled else "waiting_for_readiness"
                                        )
                                        phase_reanchor_bridge_preload_steps = int(
                                            phase_reanchor_bridge_preload_count
                                        )
                                        phase_reanchor_bridge_preload_progress_delta = (
                                            phase_reanchor_bridge_preload_progress_delta_last
                                        )
                                        phase_reanchor_bridge_preload_handle_dist = (
                                            phase_reanchor_bridge_contact_handle_dist
                                        )
                                        phase_reanchor_bridge_preload_handle_limit = float(
                                            getattr(args, "phase_reanchor_bridge_preload_handle_dist", 0.245)
                                        )
                                        preload_base_ready = bool(phase_reanchor_bridge_contact_ready)
                                        if preload_enabled and preload_base_ready:
                                            current_progress = None
                                            try:
                                                current_progress = float(task_state_before.get("drawer_open_distance"))
                                                if not np.isfinite(current_progress):
                                                    current_progress = None
                                            except (TypeError, ValueError):
                                                current_progress = None
                                            if current_progress is None and progress_before is not None:
                                                try:
                                                    current_progress = float(progress_before)
                                                except (TypeError, ValueError):
                                                    current_progress = None
                                            if (
                                                phase_reanchor_bridge_preload_start_progress is None
                                                and current_progress is not None
                                                and np.isfinite(current_progress)
                                            ):
                                                phase_reanchor_bridge_preload_start_progress = float(current_progress)
                                                phase_reanchor_bridge_preload_count = 0
                                            phase_reanchor_bridge_preload_count += 1
                                            phase_reanchor_bridge_preload_steps = int(
                                                phase_reanchor_bridge_preload_count
                                            )
                                            if (
                                                current_progress is not None
                                                and phase_reanchor_bridge_preload_start_progress is not None
                                            ):
                                                phase_reanchor_bridge_preload_progress_delta = float(
                                                    current_progress - phase_reanchor_bridge_preload_start_progress
                                                )
                                                phase_reanchor_bridge_preload_progress_delta_last = (
                                                    phase_reanchor_bridge_preload_progress_delta
                                                )
                                            phase_reanchor_bridge_preload_handle_dist_last = (
                                                phase_reanchor_bridge_preload_handle_dist
                                            )
                                            preload_min_steps = int(
                                                getattr(args, "phase_reanchor_bridge_preload_steps", 8)
                                            )
                                            preload_warmed = bool(phase_reanchor_bridge_preload_steps >= preload_min_steps)
                                            phase_reanchor_bridge_preload_progress_abs = current_progress
                                            preload_progress_abs_ok = bool(
                                                current_progress is not None
                                                and current_progress >= float(
                                                    getattr(
                                                        args,
                                                        "phase_reanchor_bridge_preload_progress_min_abs",
                                                        0.0,
                                                    )
                                                )
                                            )
                                            preload_progress_ok = bool(
                                                preload_warmed
                                                and preload_progress_abs_ok
                                                and phase_reanchor_bridge_preload_progress_delta is not None
                                                and phase_reanchor_bridge_preload_progress_delta
                                                >= float(
                                                    getattr(
                                                        args,
                                                        "phase_reanchor_bridge_preload_progress_delta",
                                                        0.0002,
                                                    )
                                                )
                                            )
                                            preload_handle_ok = bool(
                                                phase_reanchor_bridge_preload_handle_dist is not None
                                                and phase_reanchor_bridge_preload_handle_dist
                                                <= phase_reanchor_bridge_preload_handle_limit
                                            )
                                            phase_reanchor_bridge_preload_progress_ok = bool(preload_progress_ok)
                                            phase_reanchor_bridge_preload_handle_ok = bool(preload_handle_ok)
                                            allow_handle_only = bool(
                                                getattr(args, "phase_reanchor_bridge_preload_allow_handle_only", False)
                                            )
                                            phase_reanchor_bridge_preload_validated = bool(
                                                phase_reanchor_bridge_preload_validated_latched
                                                or preload_progress_ok
                                                or (allow_handle_only and preload_handle_ok)
                                            )
                                            if phase_reanchor_bridge_preload_validated:
                                                phase_reanchor_bridge_preload_validated_latched = True
                                                phase_reanchor_bridge_preload_validation_source = (
                                                    "drawer_progress"
                                                    if preload_progress_ok
                                                    else "strict_handle"
                                                )
                                                phase_reanchor_bridge_preload_reason = (
                                                    "validated:drawer_progress"
                                                    if preload_progress_ok
                                                    else "validated:strict_handle"
                                                )
                                            elif phase_reanchor_bridge_preload_steps < preload_min_steps:
                                                phase_reanchor_bridge_preload_reason = (
                                                    f"warming:{phase_reanchor_bridge_preload_steps}/{preload_min_steps}"
                                                )
                                            elif preload_handle_ok and not bool(
                                                getattr(args, "phase_reanchor_bridge_preload_allow_handle_only", False)
                                            ):
                                                phase_reanchor_bridge_preload_reason = (
                                                    "waiting:strict_handle_without_progress"
                                                )
                                            else:
                                                phase_reanchor_bridge_preload_reason = (
                                                    "waiting:contact_preload_not_validated"
                                                )
                                            phase_reanchor_bridge_preload_reason_last = (
                                                phase_reanchor_bridge_preload_reason
                                            )
                                        elif preload_enabled:
                                            phase_reanchor_bridge_preload_start_progress = None
                                            phase_reanchor_bridge_preload_count = 0
                                            phase_reanchor_bridge_preload_validated_latched = False
                                            phase_reanchor_bridge_preload_progress_delta_last = None
                                            phase_reanchor_bridge_preload_handle_dist_last = (
                                                phase_reanchor_bridge_preload_handle_dist
                                            )
                                            phase_reanchor_bridge_preload_steps = 0
                                            phase_reanchor_bridge_preload_progress_delta = None
                                            phase_reanchor_bridge_preload_progress_abs = None
                                            phase_reanchor_bridge_preload_progress_ok = False
                                            phase_reanchor_bridge_preload_handle_ok = False
                                            phase_reanchor_bridge_preload_validation_source = None
                                            phase_reanchor_bridge_preload_reason_last = (
                                                phase_reanchor_bridge_preload_reason
                                            )
                                        if (
                                            phase_reanchor_steps_left > 0
                                            and early_release_elapsed >= early_release_min_steps
                                            and bool(early_release_live_ready)
                                            and bool(phase_reanchor_bridge_contact_ready)
                                            and bool(phase_reanchor_bridge_preload_validated)
                                            and early_release_resume_ready
                                            and early_release_arm_ready
                                        ):
                                            phase_reanchor_steps_left = 0
                                            phase_reanchor_early_release_triggered = True
                                            phase_reanchor_early_release_reason = (
                                                "live_resume_arm_window_bridge_contact_ready"
                                            )
                                            phase_reanchor_early_release_grace_left = max(
                                                phase_reanchor_early_release_grace_left,
                                                int(
                                                    getattr(
                                                        args,
                                                        "phase_reanchor_early_release_act_grace_steps",
                                                        16,
                                                    )
                                                ),
                                            )
                                            phase_bridge_monitor_steps = max(
                                                int(
                                                    getattr(
                                                        args,
                                                        "post_recovery_act_bridge_no_progress_monitor_steps",
                                                        0,
                                                    )
                                                ),
                                                int(
                                                    getattr(
                                                        args,
                                                        "phase_reanchor_early_release_act_grace_steps",
                                                        16,
                                                    )
                                                ),
                                            )
                                            if phase_bridge_monitor_steps > 0:
                                                post_recovery_act_bridge_steps_left = max(
                                                    post_recovery_act_bridge_steps_left,
                                                    phase_bridge_monitor_steps,
                                                )
                                                post_recovery_act_bridge_total_steps = max(
                                                    post_recovery_act_bridge_total_steps,
                                                    phase_bridge_monitor_steps,
                                                )
                                                post_recovery_act_bridge_progress_best = None
                                                post_recovery_act_bridge_no_progress_count = 0
                                            post_recovery_task_guard_steps_left = 0
                                            post_recovery_no_progress_count = 0
                                        elif phase_reanchor_early_release_arm_q_error is not None:
                                            phase_reanchor_early_release_reason = (
                                                f"waiting:live={bool(early_release_live_ready)};"
                                                f"bridge_contact={bool(phase_reanchor_bridge_contact_ready)};"
                                                f"bridge_contact_reason={phase_reanchor_bridge_contact_reason};"
                                                f"preload={bool(phase_reanchor_bridge_preload_validated)};"
                                                f"preload_reason={phase_reanchor_bridge_preload_reason};"
                                                f"resume={early_release_resume_ready};"
                                                f"arm={early_release_arm_ready};"
                                                f"elapsed={early_release_elapsed}"
                                            )
                                        else:
                                            phase_reanchor_early_release_reason = (
                                                f"waiting:live={bool(early_release_live_ready)};"
                                                f"bridge_contact={bool(phase_reanchor_bridge_contact_ready)};"
                                                f"bridge_contact_reason={phase_reanchor_bridge_contact_reason};"
                                                f"resume={early_release_resume_ready};"
                                                f"arm_unavailable;elapsed={early_release_elapsed}"
                                            )
                                    if live_taskspace_stop_requested:
                                        post_recovery_task_guard_steps_left = 0
                                        phase_reanchor_steps_left = 0
                                    if phase_reanchor_steps_left <= 0:
                                        (
                                            phase_reanchor_live_release_ready,
                                            phase_reanchor_live_release_reason,
                                            phase_reanchor_live_release_target_error,
                                            phase_reanchor_live_release_handle_dist,
                                        ) = _phase_reanchor_live_release_status(
                                            phase_reanchor_action_state,
                                            args,
                                        )
                                        can_extend_live = (
                                            bool(getattr(args, "phase_reanchor_live_extend_on_not_ready", True))
                                            and not bool(phase_reanchor_live_release_ready)
                                            and phase_reanchor_live_extension_count
                                            < int(getattr(args, "phase_reanchor_live_max_extensions", 3))
                                        )
                                        if can_extend_live:
                                            extend_steps = max(
                                                1,
                                                int(getattr(args, "phase_reanchor_live_extend_steps", 16)),
                                            )
                                            phase_reanchor_live_extension_count += 1
                                            phase_reanchor_steps_left = extend_steps
                                            phase_reanchor_cooldown_left = 0
                                            phase_reanchor_reset_after_step = False
                                            phase_reanchor_suppress_q_servo_steps_left = max(
                                                phase_reanchor_suppress_q_servo_steps_left,
                                                extend_steps,
                                            )
                                            phase_reanchor_taskspace_worsen_count = 0
                                            post_recovery_task_guard_steps_left = max(
                                                post_recovery_task_guard_steps_left,
                                                extend_steps,
                                            )
                                            phase_reanchor_live_extension_started = True
                                            live_taskspace_suppress_q_servo = True
                                        else:
                                            phase_reanchor_live_extension_budget_exhausted = (
                                                not bool(phase_reanchor_live_release_ready)
                                            )
                                            phase_reanchor_cooldown_left = int(args.phase_reanchor_cooldown_steps)
                                            phase_reanchor_reset_after_step = True
                                    safety_info = dict(safety_info)
                                    safety_info.update(
                                        {
                                            "safety_mode": "phase_reanchor",
                                            "mode": "phase_reanchor",
                                            "pause_reason": (
                                                "phase_reanchor:"
                                                f"{reanchor_info.get('phase', 'unknown')}"
                                            ),
                                            "deformation_source": "phase_reanchor",
                                            "phase_reanchor_steps_left": int(phase_reanchor_steps_left),
                                            "phase_reanchor_phase": reanchor_info.get("phase"),
                                            "phase_reanchor_base_cmd_xy": reanchor_info.get("base_cmd_xy"),
                                            "phase_reanchor_base_cmd_normalized_xy": reanchor_info.get("base_cmd_normalized_xy"),
                                            "phase_reanchor_base_cmd_effective_raw_xy": reanchor_info.get("base_cmd_effective_raw_xy"),
                                            "phase_reanchor_base_cmd_clip_delta_norm": reanchor_info.get("base_cmd_clip_delta_norm"),
                                            "phase_reanchor_ee_error_xy": reanchor_info.get("ee_error_xy"),
                                            "phase_reanchor_drawer_fraction": reanchor_info.get("drawer_open_fraction"),
                                            "phase_reanchor_ee_to_handle_dist": reanchor_info.get("ee_to_handle_dist"),
                                            "phase_reanchor_ee_to_target_dist": reanchor_info.get("ee_to_target_dist"),
                                            "phase_reanchor_task_point_source": reanchor_info.get("task_point_source"),
                                            "phase_reanchor_task_point_requested_source": reanchor_info.get("task_point_requested_source"),
                                            "phase_reanchor_task_point_fallback_reason": reanchor_info.get("task_point_fallback_reason"),
                                            "phase_reanchor_control_task_point_source": reanchor_info.get("control_task_point_source"),
                                            "phase_reanchor_control_task_point_requested_source": reanchor_info.get("control_task_point_requested_source"),
                                            "phase_reanchor_control_task_point_fallback_reason": reanchor_info.get("control_task_point_fallback_reason"),
                                            "phase_reanchor_control_ee_to_handle_dist": reanchor_info.get("control_ee_to_handle_dist"),
                                            "phase_reanchor_control_ee_to_target_dist": reanchor_info.get("control_ee_to_target_dist"),
                                            "phase_reanchor_control_error_source": reanchor_info.get("control_error_source"),
                                            "phase_reanchor_site_ee_to_handle_dist": reanchor_info.get("site_ee_to_handle_dist"),
                                            "phase_reanchor_site_ee_to_target_dist": reanchor_info.get("site_ee_to_target_dist"),
                                            "phase_reanchor_gripper_to_handle_dist": reanchor_info.get("gripper_to_handle_dist"),
                                            "phase_reanchor_gripper_to_target_dist": reanchor_info.get("gripper_to_target_dist"),
                                            "phase_reanchor_gripper_site_xy_error": reanchor_info.get("gripper_site_xy_error"),
                                            "phase_reanchor_task_point_geometry_untrusted": reanchor_info.get("task_point_geometry_untrusted"),
                                            "phase_reanchor_handle_assist_enabled": reanchor_info.get("handle_assist_enabled"),
                                            "phase_reanchor_handle_assist_reason": reanchor_info.get("handle_assist_reason"),
                                            "phase_reanchor_handle_assist_error_norm": reanchor_info.get("handle_assist_error_norm"),
                                            "phase_reanchor_handle_assist_base_cmd_xy": reanchor_info.get("handle_assist_base_cmd_xy"),
                                            "phase_reanchor_preload_gripper_forced": reanchor_info.get("preload_gripper_forced"),
                                            "phase_reanchor_preload_gripper_limit": reanchor_info.get("preload_gripper_limit"),
                                            "phase_reanchor_preload_target_grasp": reanchor_info.get("preload_target_grasp"),
                                            "phase_reanchor_preload_grasp_limit": reanchor_info.get("preload_grasp_limit"),
                                            "phase_reanchor_arm_hold_enabled": reanchor_info.get("arm_hold_enabled"),
                                            "phase_reanchor_arm_hold_reason": reanchor_info.get("arm_hold_reason"),
                                            "phase_reanchor_arm_servo_enabled": reanchor_info.get("arm_servo_enabled"),
                                            "phase_reanchor_arm_servo_reason": reanchor_info.get("arm_servo_reason"),
                                            "phase_reanchor_arm_servo_rank": reanchor_info.get("arm_servo_rank"),
                                            "phase_reanchor_arm_servo_error_norm": reanchor_info.get("arm_servo_error_norm"),
                                            "phase_reanchor_arm_servo_delta_norm": reanchor_info.get("arm_servo_delta_norm"),
                                            "phase_reanchor_arm_servo_command_norm": reanchor_info.get("arm_servo_command_norm"),
                                            "phase_reanchor_arm_servo_action_delta_norm": reanchor_info.get("arm_servo_action_delta_norm"),
                                            "phase_reanchor_arm_servo_target_source": reanchor_info.get("arm_servo_target_source"),
                                            "phase_reanchor_arm_servo_target_episode": reanchor_info.get("arm_servo_target_episode"),
                                            "phase_reanchor_arm_servo_target_start_step": reanchor_info.get("arm_servo_target_start_step"),
                                            "phase_reanchor_arm_servo_target_step": reanchor_info.get("arm_servo_target_step"),
                                            "phase_reanchor_arm_servo_target_window_index": reanchor_info.get("arm_servo_target_window_index"),
                                            "phase_reanchor_arm_servo_target_window_score": reanchor_info.get("arm_servo_target_window_score"),
                                            "phase_reanchor_nominal_reentry_selection_reason": phase_reanchor_action_state.get("nominal_reentry_selection_reason") if isinstance(phase_reanchor_action_state, dict) else None,
                                            "phase_reanchor_nominal_reentry_live_target_distance": phase_reanchor_action_state.get("nominal_reentry_live_target_distance") if isinstance(phase_reanchor_action_state, dict) else None,
                                            "phase_reanchor_live_taskspace_guard_active": live_taskspace_guard_active,
                                            "phase_reanchor_live_taskspace_suppress_q_servo": live_taskspace_suppress_q_servo,
                                            "phase_reanchor_live_taskspace_suppress_q_servo_reason": live_taskspace_suppress_q_servo_reason,
                                            "phase_reanchor_live_taskspace_distance": live_taskspace_distance,
                                            "phase_reanchor_live_taskspace_distance_source": live_taskspace_distance_source,
                                            "phase_reanchor_live_taskspace_best_distance": phase_reanchor_best_target_distance,
                                            "phase_reanchor_live_taskspace_worsen_count": int(phase_reanchor_taskspace_worsen_count),
                                            "phase_reanchor_live_taskspace_stop_requested": live_taskspace_stop_requested,
                                            "phase_reanchor_live_taskspace_stop_reason": live_taskspace_stop_reason,
                                            "phase_reanchor_live_taskspace_elapsed_steps": int(live_taskspace_elapsed_steps),
                                            "phase_reanchor_live_release_ready": phase_reanchor_live_release_ready,
                                            "phase_reanchor_live_release_reason": phase_reanchor_live_release_reason,
                                            "phase_reanchor_live_release_target_error": phase_reanchor_live_release_target_error,
                                            "phase_reanchor_live_release_handle_dist": phase_reanchor_live_release_handle_dist,
                                            "phase_reanchor_live_extension_started": phase_reanchor_live_extension_started,
                                            "phase_reanchor_live_extension_count": int(phase_reanchor_live_extension_count),
                                            "phase_reanchor_live_extension_budget_exhausted": phase_reanchor_live_extension_budget_exhausted,
                                            "phase_reanchor_early_release_triggered": phase_reanchor_early_release_triggered,
                                            "phase_reanchor_early_release_reason": phase_reanchor_early_release_reason,
                                            "phase_reanchor_early_release_arm_q_error": phase_reanchor_early_release_arm_q_error,
                                            "phase_reanchor_bridge_contact_ready": phase_reanchor_bridge_contact_ready,
                                            "phase_reanchor_bridge_contact_reason": phase_reanchor_bridge_contact_reason,
                                            "phase_reanchor_bridge_contact_handle_dist": phase_reanchor_bridge_contact_handle_dist,
                                            "phase_reanchor_bridge_contact_handle_limit": phase_reanchor_bridge_contact_handle_limit,
                                            "phase_reanchor_bridge_preload_validated": phase_reanchor_bridge_preload_validated,
                                            "phase_reanchor_bridge_preload_reason": phase_reanchor_bridge_preload_reason,
                                            "phase_reanchor_bridge_preload_steps": phase_reanchor_bridge_preload_steps,
                                            "phase_reanchor_bridge_preload_progress_delta": phase_reanchor_bridge_preload_progress_delta,
                                            "phase_reanchor_bridge_preload_progress_abs": phase_reanchor_bridge_preload_progress_abs,
                                            "phase_reanchor_bridge_preload_handle_dist": phase_reanchor_bridge_preload_handle_dist,
                                            "phase_reanchor_bridge_preload_handle_limit": phase_reanchor_bridge_preload_handle_limit,
                                            "phase_reanchor_bridge_preload_progress_ok": phase_reanchor_bridge_preload_progress_ok,
                                            "phase_reanchor_bridge_preload_handle_ok": phase_reanchor_bridge_preload_handle_ok,
                                            "phase_reanchor_bridge_preload_validation_source": phase_reanchor_bridge_preload_validation_source,
                                            "phase_reanchor_preload_pull_probe_enabled": reanchor_info.get("preload_pull_probe_enabled"),
                                            "phase_reanchor_preload_pull_probe_reason": reanchor_info.get("preload_pull_probe_reason"),
                                            "phase_reanchor_preload_pull_probe_axis_xy": reanchor_info.get("preload_pull_probe_axis_xy"),
                                            "phase_reanchor_preload_pull_probe_step": reanchor_info.get("preload_pull_probe_step"),
                                            "phase_reanchor_preload_pull_probe_delta_norm": reanchor_info.get("preload_pull_probe_delta_norm"),
                                            "phase_reanchor_early_release_act_grace": bool(phase_reanchor_early_release_grace_left > 0),
                                            "phase_reanchor_early_release_act_grace_steps": int(phase_reanchor_early_release_grace_left),
                                            "phase_reanchor_live_ee_servo_enabled": reanchor_info.get("live_ee_servo_enabled"),
                                            "phase_reanchor_live_ee_servo_reason": reanchor_info.get("live_ee_servo_reason"),
                                            "phase_reanchor_live_ee_servo_error_norm": reanchor_info.get("live_ee_servo_error_norm"),
                                            "phase_reanchor_live_ee_servo_delta_norm": reanchor_info.get("live_ee_servo_delta_norm"),
                                            "phase_reanchor_live_ee_servo_command_norm": reanchor_info.get("live_ee_servo_command_norm"),
                                            "phase_reanchor_live_ee_servo_jacobian_rank": reanchor_info.get("live_ee_servo_jacobian_rank"),
                                            "phase_reanchor_live_ee_servo_nominal_reg": reanchor_info.get("live_ee_servo_nominal_reg"),
                                            "phase_reanchor_live_ee_servo_fk_site_xy_error": reanchor_info.get("live_ee_servo_fk_site_xy_error"),
                                            "phase_reanchor_live_ee_servo_fk_gripper_xy_error": reanchor_info.get("live_ee_servo_fk_gripper_xy_error"),
                                            "phase_reanchor_live_ee_servo_predicted_error_before": reanchor_info.get("live_ee_servo_predicted_error_before"),
                                            "phase_reanchor_live_ee_servo_predicted_error_after": reanchor_info.get("live_ee_servo_predicted_error_after"),
                                            "phase_reanchor_live_ee_servo_geometry_untrusted": reanchor_info.get("live_ee_servo_geometry_untrusted"),
                                            "phase_reanchor_suppressed_q_servo_arm_hold": reanchor_info.get("suppressed_q_servo_arm_hold"),
                                            "phase_reanchor_rejected": False,
                                            "phase_reanchor_acceptance_type": (
                                                reanchor_acceptance or {}
                                            ).get("acceptance_type"),
                                            "phase_reanchor_immediate_clearance": (
                                                reanchor_acceptance or {}
                                            ).get("immediate_clearance"),
                                            "phase_reanchor_horizon_min_clearance": (
                                                reanchor_acceptance or {}
                                            ).get("horizon_min_clearance"),
                                        }
                                    )
                                    policy_hold_active = True

                latch_dim = int(args.gripper_latch_dim)
                latch_requested = bool(
                    args.gripper_latch
                    or (
                        args.post_recovery_task_guard
                        and args.post_recovery_task_guard_force_gripper
                        and gripper_latched
                    )
                )
                if latch_requested:
                    if not -safe_first_action.shape[0] <= latch_dim < safe_first_action.shape[0]:
                        raise ValueError(
                            f"--gripper-latch-dim {latch_dim} is out of range "
                            f"for action shape {safe_first_action.shape}."
                        )
                    if (
                        args.gripper_latch
                        and not gripper_latched
                        and step >= args.gripper_latch_start_step
                        and safe_first_action[latch_dim] >= args.gripper_latch_trigger
                    ):
                        gripper_latched = True
                        if args.debug:
                            print(
                                f"gripper latch activated at episode={episode} "
                                f"step={step} dim={latch_dim} "
                                f"value={safe_first_action[latch_dim]:.3f}"
                            )
                    if gripper_latched:
                        safe_first_action[latch_dim] = args.gripper_latch_value

                raw_first_action = _raw_scaled_first_action(env, safe_first_action)
                if raw_first_action is None:
                    raw_action_norm = None
                    raw_arm_min = None
                    raw_arm_max = None
                else:
                    raw_arm = raw_first_action[arm_idx]
                    raw_action_norm = float(np.linalg.norm(raw_first_action))
                    raw_arm_min = float(np.min(raw_arm))
                    raw_arm_max = float(np.max(raw_arm))

                if latch_requested and gripper_latched:
                    if safe_env_action.ndim == 1:
                        safe_env_action[latch_dim] = args.gripper_latch_value
                    else:
                        safe_env_action[:, latch_dim] = args.gripper_latch_value

                safe_gripper_action = (
                    float(safe_first_action[latch_dim])
                    if -safe_first_action.shape[0] <= latch_dim < safe_first_action.shape[0]
                    else None
                )
                raw_gripper_action = (
                    float(raw_first_action[latch_dim])
                    if raw_first_action is not None
                    and -raw_first_action.shape[0] <= latch_dim < raw_first_action.shape[0]
                    else None
                )

                post_recovery_task_guard_active = bool(
                    args.post_recovery_task_guard
                    and (
                        post_recovery_task_guard_steps_left > 0
                        or post_recovery_reanchor_started
                        or post_recovery_progress_regression is not None
                        or post_recovery_no_progress_triggered
                        or (
                            post_recovery_task_guard_reason is not None
                            and phase_reanchor_steps_left > 0
                        )
                    )
                )
                safety_info = dict(safety_info)
                for _resume_key, _resume_value in resume_affordance_context.items():
                    safety_info.setdefault(_resume_key, _resume_value)
                act_reentry_diag = {}
                act_reentry_diag_active = bool(
                    post_recovery_task_guard_active
                    or _safe_info_get(safety_info, "deformation_source") == "phase_reanchor"
                    or _safe_info_get(safety_info, "safety_mode") == "phase_reanchor"
                )
                if act_reentry_diag_active:
                    act_reentry_diag["act_reentry_diag_active"] = True
                    try:
                        act_first_arr = np.asarray(first_action, dtype=np.float32).reshape(-1)
                        safe_first_arr = np.asarray(safe_first_action, dtype=np.float32).reshape(-1)
                        dim = min(act_first_arr.size, safe_first_arr.size)
                        if dim > 0:
                            a = act_first_arr[:dim]
                            b = safe_first_arr[:dim]
                            delta = b - a
                            denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                            act_reentry_diag.update(
                                {
                                    "act_reentry_diag_act_first_vs_safe_l2": float(np.linalg.norm(delta)),
                                    "act_reentry_diag_act_first_vs_safe_max_abs": float(np.max(np.abs(delta))),
                                    "act_reentry_diag_act_first_vs_safe_cosine": float(np.dot(a, b) / denom),
                                    "act_reentry_diag_act_first_vs_safe_dim": int(dim),
                                    "act_resume_diag_active": True,
                                    "act_resume_diag_query_step": int(step),
                                    "act_resume_diag_predicted_first_action_norm": float(np.linalg.norm(a)),
                                    "act_resume_diag_target_first_action_norm": float(np.linalg.norm(b)),
                                    "act_resume_diag_executed_vs_act_first_l2": float(np.linalg.norm(delta)),
                                    "act_resume_diag_executed_vs_act_first_max_abs": float(np.max(np.abs(delta))),
                                    "act_resume_diag_executed_vs_act_first_cosine": float(np.dot(a, b) / denom),
                                    "act_resume_diag_executed_vs_act_first_dim": int(dim),
                                }
                            )
                    except Exception as exc:  # noqa: BLE001
                        act_reentry_diag["act_reentry_diag_error"] = repr(exc)
                    if isinstance(policy_obs, dict):
                        low_dim = policy_obs.get("low_dim_state")
                        if low_dim is not None:
                            try:
                                low_arr = np.asarray(low_dim, dtype=np.float32).reshape(-1)
                                low_arr = low_arr[np.isfinite(low_arr)]
                                if low_arr.size:
                                    act_reentry_diag.update(
                                        {
                                            "act_reentry_diag_policy_low_dim_dim": int(low_arr.size),
                                            "act_reentry_diag_policy_low_dim_norm": float(np.linalg.norm(low_arr)),
                                            "act_reentry_diag_policy_low_dim_mean": float(np.mean(low_arr)),
                                            "act_reentry_diag_policy_low_dim_std": float(np.std(low_arr)),
                                            "act_reentry_diag_policy_low_dim_max_abs": float(np.max(np.abs(low_arr))),
                                        }
                                    )
                            except Exception as exc:  # noqa: BLE001
                                act_reentry_diag["act_reentry_diag_low_dim_error"] = repr(exc)
                        image_means = []
                        image_stds = []
                        image_max_abs = []
                        for obs_key, obs_value in policy_obs.items():
                            obs_key_str = str(obs_key)
                            if not (
                                obs_key_str.startswith("rgb_")
                                or "camera" in obs_key_str
                                or "image" in obs_key_str
                            ):
                                continue
                            try:
                                image_arr = np.asarray(obs_value, dtype=np.float32)
                                if image_arr.size == 0:
                                    continue
                                if np.nanmax(np.abs(image_arr)) > 2.0:
                                    image_arr = image_arr / 255.0
                                finite = image_arr[np.isfinite(image_arr)]
                                if finite.size == 0:
                                    continue
                                image_means.append(float(np.mean(finite)))
                                image_stds.append(float(np.std(finite)))
                                image_max_abs.append(float(np.max(np.abs(finite))))
                            except Exception:  # noqa: BLE001
                                continue
                        if image_means:
                            act_reentry_diag.update(
                                {
                                    "act_reentry_diag_image_key_count": int(len(image_means)),
                                    "act_reentry_diag_image_mean_mean": float(np.mean(image_means)),
                                    "act_reentry_diag_image_std_mean": float(np.mean(image_stds)),
                                    "act_reentry_diag_image_max_abs_mean": float(np.mean(image_max_abs)),
                                }
                            )

                action_agreement_info = {
                    "act_action_agreement_logged": False,
                    "act_action_agreement_pair_count": 0,
                    "act_action_agreement_post_recovery_or_reentry": bool(act_reentry_diag_active),
                }
                action_agreement_context = (
                    _safe_info_get(safety_info, "safety_mode")
                    or _safe_info_get(safety_info, "mode")
                    or _safe_info_get(safety_info, "deformation_source")
                    or "pass_through"
                )
                if post_recovery_act_bridge_active:
                    action_agreement_context = "act_bridge"
                elif act_reentry_diag_active:
                    action_agreement_context = "post_recovery_or_reentry"
                action_agreement_info["act_action_agreement_context"] = str(action_agreement_context)
                act_first_for_agreement = _first_action_or_none(env_action)
                safe_first_for_agreement = _first_action_or_none(safe_env_action)
                nominal_first_for_agreement = _first_action_or_none(env_action)
                target_first_for_agreement = act_resume_diag_target_first_action
                last_recovery_first_for_agreement = _first_action_or_none(
                    last_recovery_first_action_for_bridge
                )
                handoff_release_first_for_agreement = _first_action_or_none(
                    handoff_release_first_action_for_bridge
                )
                handoff_release_executed_for_agreement = _first_action_or_none(
                    handoff_release_executed_action_for_bridge
                )
                action_agreement_pairs = (
                    (act_first_for_agreement, safe_first_for_agreement, "act_action_agreement_act_vs_safe"),
                    (act_first_for_agreement, nominal_first_for_agreement, "act_action_agreement_act_vs_nominal"),
                    (safe_first_for_agreement, nominal_first_for_agreement, "act_action_agreement_safe_vs_nominal"),
                    (act_first_for_agreement, target_first_for_agreement, "act_action_agreement_act_vs_target"),
                    (safe_first_for_agreement, target_first_for_agreement, "act_action_agreement_safe_vs_target"),
                    (act_first_for_agreement, last_recovery_first_for_agreement, "act_action_agreement_act_vs_last_recovery"),
                    (safe_first_for_agreement, last_recovery_first_for_agreement, "act_action_agreement_safe_vs_last_recovery"),
                    (act_first_for_agreement, handoff_release_first_for_agreement, "act_action_agreement_act_vs_handoff_release"),
                    (safe_first_for_agreement, handoff_release_first_for_agreement, "act_action_agreement_safe_vs_handoff_release"),
                    (act_first_for_agreement, handoff_release_executed_for_agreement, "act_action_agreement_act_vs_executed_handoff_release"),
                    (safe_first_for_agreement, handoff_release_executed_for_agreement, "act_action_agreement_safe_vs_executed_handoff_release"),
                )
                action_agreement_pair_count = 0
                for _lhs_action, _rhs_action, _agreement_prefix in action_agreement_pairs:
                    _agreement_metrics = _action_agreement_metrics(
                        _lhs_action,
                        _rhs_action,
                        _agreement_prefix,
                        arm_indices=arm_idx,
                        gripper_index=latch_dim,
                    )
                    if _agreement_metrics:
                        action_agreement_info.update(_agreement_metrics)
                        action_agreement_pair_count += 1
                action_agreement_info["act_action_agreement_pair_count"] = int(action_agreement_pair_count)
                action_agreement_info["act_action_agreement_logged"] = bool(
                    action_agreement_pair_count > 0
                )

                safety_info.update(
                    {
                        "gripper_latched": bool(gripper_latched),
                        "gripper_latch_dim": int(latch_dim),
                        "safe_gripper_action": safe_gripper_action,
                        "raw_gripper_action": raw_gripper_action,
                        "post_recovery_task_guard_active": post_recovery_task_guard_active,
                        "post_recovery_task_guard_steps_left": int(post_recovery_task_guard_steps_left),
                        "post_recovery_task_guard_reason": post_recovery_task_guard_reason,
                        "post_recovery_task_guard_best_progress": post_recovery_task_guard_best_progress,
                        "post_recovery_progress_regression": post_recovery_progress_regression,
                        "post_recovery_reanchor_started": bool(post_recovery_reanchor_started),
                        "post_recovery_no_progress_count": int(post_recovery_no_progress_count),
                        "post_recovery_mid_progress_no_progress_count": int(
                            post_recovery_mid_progress_no_progress_count
                        ),
                        "post_recovery_mid_progress_best_progress": post_recovery_mid_progress_best_progress,
                        "post_recovery_mid_progress_best_distance": post_recovery_mid_progress_best_distance,
                        "post_recovery_mid_progress_distance_regression": post_recovery_mid_progress_distance_regression,
                        "post_recovery_mid_progress_reseed_triggered": bool(
                            post_recovery_mid_progress_reseed_triggered
                        ),
                        "post_recovery_mid_progress_reseed_reset_count": int(
                            post_recovery_mid_progress_reseed_reset_count
                        ),
                        "post_recovery_mid_progress_reseed_reason": post_recovery_mid_progress_reseed_reason,
                        "post_recovery_mid_progress_prior_action_seed_count": int(
                            post_recovery_mid_progress_prior_action_seed_count
                        ),
                        "post_recovery_mid_progress_prior_action_seed_step": post_recovery_mid_progress_prior_action_seed_step,
                        "post_recovery_mid_progress_prior_action_seed_age": post_recovery_mid_progress_prior_action_seed_age,
                        "post_recovery_no_progress_triggered": bool(post_recovery_no_progress_triggered),
                        "post_recovery_no_progress_target_distance": post_recovery_no_progress_target_distance,
                        "post_recovery_no_progress_distance_source": post_recovery_no_progress_distance_source,
                        "post_recovery_act_bridge_active": bool(post_recovery_act_bridge_active),
                        "post_recovery_act_bridge_steps_left": int(post_recovery_act_bridge_steps_left),
                        "post_recovery_act_bridge_total_steps": int(post_recovery_act_bridge_total_steps),
                        "post_recovery_act_bridge_step_index": post_recovery_act_bridge_step_index,
                        "post_recovery_act_bridge_last_recovery_step": last_recovery_action_step_for_bridge,
                        **act_reentry_diag,
                        **action_agreement_info,
                    }
                )
                if post_recovery_act_bridge_active:
                    safety_info.update(action_bridge_info)

                if (
                    trajectory_logging_enabled
                    and chunk_filter_mode
                    and chunk_trace_context is not None
                    and len(all_chunk_trajectory_records) < max(0, int(args.chunk_trajectory_max_events))
                    and _should_log_chunk_trajectory_trace(
                        args,
                        safety_info,
                        env_action,
                        safe_env_action,
                        args.intervention_eps,
                    )
                ):
                    trace_record = _collect_chunk_trajectory_trace(
                        args=args,
                        episode=episode,
                        step=step,
                        safechunk=safechunk,
                        horizon_operator=horizon_operator,
                        obs=chunk_trace_context["obs"],
                        nominal_chunk=chunk_trace_context["nominal_chunk"],
                        generated_chunk=safe_env_action,
                        safety_info=safety_info,
                        human_sample=human_arm_trace_sample,
                        policy_anchor_sample=_robot_ee_trajectory_sample(
                            env,
                            episode,
                            step,
                            task_state_before,
                        ),
                    )
                    if trace_record is not None:
                        episode_chunk_trajectory_records.append(trace_record)
                        all_chunk_trajectory_records.append(trace_record)

                if nominal_rollout_diagnostic_context is not None and chunk_trace_context is not None:
                    try:
                        safe_diag_chunk, _ = _as_chunk(safe_env_action)
                        safe_q_seq_diag = np.asarray(
                            safechunk.deform.rollout_nominal_chunk(
                                chunk_trace_context["obs"],
                                safe_diag_chunk,
                            ),
                            dtype=np.float32,
                        )
                        nominal_rollout_diagnostic_context["safe_rollout_shape"] = list(safe_q_seq_diag.shape)
                        nominal_rollout_diagnostic_context["safe_pred_q_next"] = _first_rollout_state(safe_q_seq_diag)
                        nominal_rollout_diagnostic_context["safe_first_action"] = np.asarray(
                            safe_diag_chunk[0],
                            dtype=np.float32,
                        ).reshape(-1).copy()
                    except Exception as exc:  # noqa: BLE001
                        nominal_rollout_diagnostic_context["safe_rollout_error"] = repr(exc)

                saved_episode_actions.append(
                    np.asarray(safe_env_action, dtype=np.float32).copy()
                )

                safety_intervention_active = _is_safety_intervention_mode(safety_info)
                reset_policy_after_intervention = (
                    replay_actions is None
                    and chunk_filter_mode
                    and safety_intervention_active
                    and not last_safety_intervention_active
                )

                brake_execution_active = (
                    replay_actions is None
                    and _is_brake_or_fallback_execution(safety_info)
                )
                brake_action_history_reset_count = 0
                brake_low_level_hold_sync_count = 0
                brake_robot_freeze_count = 0
                brake_temporal_ensemble_records = []
                phase_reanchor_temporal_ensemble_bypass = False
                phase_reanchor_temporal_ensemble_bypass_count = 0
                if brake_execution_active:
                    brake_action_history_reset_count = _reset_action_sequence_history(env)
                    brake_low_level_hold_sync_count = _sync_robot_low_level_hold_state(env)
                    brake_robot_freeze_count = _set_robot_freeze_next_step(env)
                    brake_temporal_ensemble_records.extend(
                        _set_action_sequence_temporal_ensemble(env, False)
                    )
                    if policy_env is not None:
                        brake_action_history_reset_count += _reset_action_sequence_history(policy_env)
                        brake_low_level_hold_sync_count += _sync_robot_low_level_hold_state(policy_env)
                        brake_robot_freeze_count += _set_robot_freeze_next_step(policy_env)
                        brake_temporal_ensemble_records.extend(
                            _set_action_sequence_temporal_ensemble(policy_env, False)
                        )
                    safety_info = dict(safety_info)
                    safety_info.update(
                        {
                            "brake_temporal_ensemble_bypass": True,
                            "brake_action_history_reset_count": int(brake_action_history_reset_count),
                            "brake_low_level_hold_sync_count": int(brake_low_level_hold_sync_count),
                            "brake_robot_freeze_count": int(brake_robot_freeze_count),
                        }
                    )

                phase_reanchor_execution_active = (
                    replay_actions is None
                    and bool(getattr(args, "phase_reanchor_bypass_temporal_ensemble", False))
                    and (
                        _safe_info_get(safety_info, "deformation_source") == "phase_reanchor"
                        or _safe_info_get(safety_info, "safety_mode") == "phase_reanchor"
                        or _safe_info_get(safety_info, "mode") == "phase_reanchor"
                    )
                )
                if phase_reanchor_execution_active:
                    phase_records = _set_action_sequence_temporal_ensemble(env, False)
                    phase_reanchor_temporal_ensemble_bypass_count += len(phase_records)
                    brake_temporal_ensemble_records.extend(phase_records)
                    if policy_env is not None:
                        phase_records = _set_action_sequence_temporal_ensemble(policy_env, False)
                        phase_reanchor_temporal_ensemble_bypass_count += len(phase_records)
                        brake_temporal_ensemble_records.extend(phase_records)
                    phase_reanchor_temporal_ensemble_bypass = bool(
                        phase_reanchor_temporal_ensemble_bypass_count > 0
                    )
                    safety_info = dict(safety_info)
                    safety_info.update(
                        {
                            "phase_reanchor_temporal_ensemble_bypass": phase_reanchor_temporal_ensemble_bypass,
                            "phase_reanchor_temporal_ensemble_bypass_count": int(
                                phase_reanchor_temporal_ensemble_bypass_count
                            ),
                        }
                    )

                ablation_force_sequence_applied_this_step = False
                ablation_force_sequence_this_step_q = None
                ablation_force_sequence_this_step_qvel = None
                ablation_force_sequence_this_step_indices = None
                if (
                    pending_ablation_force_q_sequence is not None
                    and replay_actions is None
                    and chunk_filter_mode
                ):
                    try:
                        q_seq_arr = np.asarray(pending_ablation_force_q_sequence, dtype=np.float64)
                    except Exception:
                        q_seq_arr = np.empty((0, 0), dtype=np.float64)
                    qvel_seq_arr = (
                        np.asarray(pending_ablation_force_q_sequence_qvel, dtype=np.float64)
                        if pending_ablation_force_q_sequence_qvel is not None
                        else None
                    )
                    seq_idx = int(pending_ablation_force_q_sequence_index)
                    if q_seq_arr.ndim == 2 and 0 <= seq_idx < q_seq_arr.shape[0]:
                        indices_arr = np.asarray(pending_ablation_force_q_sequence_indices, dtype=np.int64)
                        frame_qvel = (
                            qvel_seq_arr[seq_idx]
                            if qvel_seq_arr is not None and seq_idx < qvel_seq_arr.shape[0]
                            else None
                        )
                        forced_env_count = 0
                        seen_envs = set()
                        for candidate in (env, policy_env, safety_runtime_env):
                            if candidate is None or id(candidate) in seen_envs:
                                continue
                            seen_envs.add(id(candidate))
                            if _set_h1_q_for_ablation(
                                candidate,
                                q_seq_arr[seq_idx],
                                indices_arr,
                                zero_velocity=bool(
                                    getattr(args, "ablation_force_planned_recovery_q_zero_velocity", True)
                                ),
                                target_qvel=frame_qvel,
                            ):
                                forced_env_count += 1
                        sequence_action_source = None
                        sequence_action_index = None
                        sequence_nominal_action_used = False
                        nominal_env_action = _ablation_nominal_env_action_for_sequence_step(
                            pending_ablation_force_q_sequence_target_action,
                            seq_idx,
                            safe_env_action,
                        )
                        if nominal_env_action is not None:
                            safe_env_action = nominal_env_action
                            hold_indices = None
                            hold_delta = None
                            sequence_action_source = (
                                pending_ablation_force_q_sequence_target_action_source
                                or "recover_resume_window_target_action"
                            )
                            sequence_action_index = int(seq_idx)
                            sequence_nominal_action_used = True
                        else:
                            try:
                                safe_env_action, hold_indices, hold_delta = _hard_hold_action_from_live_robot(
                                    env, safe_env_action
                                )
                                sequence_action_source = "live_hold"
                            except Exception as exc:  # noqa: BLE001
                                hold_indices = None
                                hold_delta = None
                                logger.debug("Could not hold during planned recovery q sequence ablation: %s", exc)
                        safety_info = dict(safety_info)
                        safety_info.update(
                            {
                                "safety_mode": "ablation_force_recovery_q_sequence",
                                "mode": "ablation_force_recovery_q_sequence",
                                "deformation_source": "ablation_force_recovery_q_sequence",
                                "suppress_outer_pause": True,
                                "ablation_force_planned_recovery_q_sequence_active": True,
                                "ablation_force_planned_recovery_q_sequence_index": int(seq_idx),
                                "ablation_force_planned_recovery_q_sequence_len": int(q_seq_arr.shape[0]),
                                "ablation_force_planned_recovery_q_sequence_forced_env_count": int(forced_env_count),
                                "ablation_force_planned_recovery_q_sequence_hold_delta": (
                                    float(hold_delta) if hold_delta is not None else None
                                ),
                                "ablation_force_planned_recovery_q_sequence_hold_indices": hold_indices,
                                "ablation_force_planned_recovery_q_sequence_qvel_seeded": bool(
                                    frame_qvel is not None
                                ),
                                "ablation_force_planned_recovery_q_sequence_nominal_action_used": bool(
                                    sequence_nominal_action_used
                                ),
                                "ablation_force_planned_recovery_q_sequence_action_source": sequence_action_source,
                                "ablation_force_planned_recovery_q_sequence_action_index": sequence_action_index,
                            }
                        )
                        pending_ablation_force_q_sequence_index = seq_idx + 1
                        ablation_force_sequence_applied_this_step = True
                        ablation_force_sequence_this_step_q = q_seq_arr[seq_idx].copy()
                        ablation_force_sequence_this_step_qvel = (
                            frame_qvel.copy() if frame_qvel is not None else None
                        )
                        ablation_force_sequence_this_step_indices = indices_arr.copy()
                        safe_first_action = extract_first_action(safe_env_action)
                        policy_hold_active = True
                    else:
                        pending_ablation_force_q_sequence = None

                if (
                    bool(getattr(args, "ablation_force_planned_recovery_q", False))
                    and replay_actions is None
                    and chunk_filter_mode
                    and pending_ablation_force_q_sequence is None
                    and not (
                        bool(getattr(args, "ablation_force_planned_recovery_q_once_per_episode", True))
                        and ablation_force_planned_recovery_q_done
                    )
                ):
                    (
                        safety_info,
                        safe_env_action,
                        ablation_force_applied,
                        pending_ablation_policy_obs_history,
                    ) = _maybe_force_planned_recovery_q_ablation(
                        args=args,
                        env=env,
                        policy_env=policy_env,
                        safety_runtime_env=safety_runtime_env,
                        safechunk=safechunk,
                        safety_info=safety_info,
                        safe_env_action=safe_env_action,
                    )
                    if ablation_force_applied:
                        ablation_force_planned_recovery_q_done = True
                        q_sequence_payload = _safe_info_get(
                            safety_info,
                            "ablation_force_planned_recovery_q_replay_sequence_q",
                        )
                        sequence_enabled = bool(
                            _safe_info_get(
                                safety_info,
                                "ablation_force_planned_recovery_q_replay_sequence_enabled",
                            )
                            and q_sequence_payload is not None
                        )
                        if sequence_enabled:
                            pending_ablation_force_q_sequence = np.asarray(
                                q_sequence_payload, dtype=np.float64
                            )
                            qvel_sequence_payload = _safe_info_get(
                                safety_info,
                                "ablation_force_planned_recovery_q_replay_sequence_qvel",
                            )
                            pending_ablation_force_q_sequence_qvel = (
                                np.asarray(qvel_sequence_payload, dtype=np.float64)
                                if qvel_sequence_payload is not None
                                else None
                            )
                            pending_ablation_force_q_sequence_index = min(
                                1, int(pending_ablation_force_q_sequence.shape[0])
                            )
                            pending_ablation_force_q_sequence_history = []
                            pending_ablation_force_q_sequence_indices = _safe_info_get(
                                safety_info,
                                "ablation_force_planned_recovery_q_indices",
                            )
                            pending_ablation_force_q_sequence_target_action = _safe_info_get(
                                safety_info,
                                "recover_resume_window_target_action",
                            )
                            pending_ablation_force_q_sequence_target_action_source = (
                                "recover_resume_window_target_action"
                                if pending_ablation_force_q_sequence_target_action is not None
                                else None
                            )
                            safety_info = dict(safety_info)
                            safety_info.update(
                                {
                                    "ablation_force_planned_recovery_q_sequence_active": True,
                                    "ablation_force_planned_recovery_q_sequence_index": 0,
                                    "ablation_force_planned_recovery_q_sequence_len": int(
                                        pending_ablation_force_q_sequence.shape[0]
                                    ),
                                }
                            )
                            ablation_force_sequence_applied_this_step = True
                            ablation_force_sequence_this_step_q = pending_ablation_force_q_sequence[0].copy()
                            ablation_force_sequence_this_step_qvel = (
                                pending_ablation_force_q_sequence_qvel[0].copy()
                                if pending_ablation_force_q_sequence_qvel is not None
                                else None
                            )
                            ablation_force_sequence_this_step_indices = (
                                np.asarray(pending_ablation_force_q_sequence_indices, dtype=np.int64)
                                if pending_ablation_force_q_sequence_indices is not None
                                else None
                            )
                        elif ablation_pure_act_resume_total_steps > 0:
                            ablation_pure_act_resume_steps_left = max(
                                int(ablation_pure_act_resume_steps_left),
                                int(ablation_pure_act_resume_total_steps),
                            )
                            safety_info = dict(safety_info)
                            safety_info.update(
                                {
                                    "ablation_pure_act_resume_enabled": True,
                                    "ablation_pure_act_resume_scheduled_steps": int(
                                        ablation_pure_act_resume_steps_left
                                    ),
                                    "ablation_pure_act_resume_total_steps": int(
                                        ablation_pure_act_resume_total_steps
                                    ),
                                }
                            )
                        safe_first_action = extract_first_action(safe_env_action)
                        policy_hold_active = True

                if act_resume_diag_info:
                    executed_first_action = _first_action_or_none(safe_env_action)
                    act_first_action = _first_action_or_none(env_action)
                    if executed_first_action is not None and act_first_action is not None:
                        act_resume_diag_info.update(
                            _vector_compare_metrics(
                                executed_first_action,
                                act_first_action,
                                "act_resume_diag_executed_vs_act_first",
                            )
                        )
                    if executed_first_action is not None and act_resume_diag_target_first_action is not None:
                        act_resume_diag_info.update(
                            _vector_compare_metrics(
                                executed_first_action,
                                act_resume_diag_target_first_action,
                                "act_resume_diag_executed_vs_target",
                            )
                        )
                    safety_info = dict(safety_info)
                    safety_info.update(act_resume_diag_info)

                mpc_replay_pre_mujoco_snapshot = None
                if (
                    mpc_replay_diagnostic_logging_enabled
                    and len(all_mpc_replay_diagnostic_records) < max(0, int(args.mpc_replay_diagnostics_max_events))
                    and _should_log_mpc_replay_diagnostic(safety_info)
                ):
                    mpc_replay_pre_mujoco_snapshot = _mujoco_state_snapshot(env)

                try:
                    env_step_t0 = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(safe_env_action)
                    env_step_time_ms = 1000.0 * (time.perf_counter() - env_step_t0)
                    rhc_executed_action = info.get("rhc_executed_action") if isinstance(info, dict) else None
                    rhc_requested_action = info.get("rhc_requested_action") if isinstance(info, dict) else None
                    rhc_execution_index = info.get("rhc_execution_index") if isinstance(info, dict) else None
                    if rhc_executed_action is None:
                        for candidate in (env, policy_env):
                            wrapper = _find_wrapped_env_with_attr(candidate, "_last_executed_action")
                            if wrapper is None:
                                continue
                            rhc_executed_action = getattr(wrapper, "_last_executed_action", None)
                            rhc_requested_action = getattr(wrapper, "_last_requested_action", None)
                            rhc_execution_index = getattr(wrapper, "_last_execution_index", None)
                            if rhc_executed_action is not None:
                                break
                    if rhc_executed_action is not None:
                        safety_info = dict(safety_info)
                        safety_info["rhc_executed_action_available"] = True
                        safety_info["rhc_execution_index"] = rhc_execution_index
                        safety_info.update(
                            _vector_compare_metrics(
                                rhc_requested_action,
                                rhc_executed_action,
                                "rhc_requested_vs_executed",
                            )
                        )
                        if bool(_safe_info_get(safety_info, "mpc_handoff_accepted")):
                            handoff_release_executed_action_for_bridge = np.asarray(
                                rhc_executed_action, dtype=np.float32
                            ).reshape(-1).copy()
                            safety_info["mpc_handoff_executed_release_provenance_source"] = "rhc_post_ensemble_smoothing"
                            safety_info["mpc_handoff_executed_release_age_steps"] = 0
                    policy_obs_update_time_ms = 0.0
                    if policy_env is None:
                        policy_obs_update_t0 = time.perf_counter()
                        if args.hide_human_arm_policy_obs:
                            policy_obs = _policy_obs_with_hidden_human_arm(
                                env,
                                obs,
                                prev_policy_obs=policy_obs,
                            )
                        else:
                            policy_obs = obs
                        policy_obs_update_time_ms = 1000.0 * (time.perf_counter() - policy_obs_update_t0)
                    else:
                        policy_env_step_t0 = time.perf_counter()
                        policy_obs, _policy_reward, policy_terminated, policy_truncated, _policy_info = policy_env.step(
                            safe_env_action
                        )
                        env_step_time_ms += 1000.0 * (time.perf_counter() - policy_env_step_t0)
                        if (policy_terminated or policy_truncated) and not (
                            terminated or truncated or extract_success(info, float(reward), bool(terminated))
                        ):
                            print(
                                "Warning: clean policy env ended before eval env; "
                                "policy/eval states may have diverged."
                            )
                finally:
                    _restore_action_sequence_temporal_ensemble(brake_temporal_ensemble_records)

                if (
                    ablation_force_sequence_applied_this_step
                    and ablation_force_sequence_this_step_q is not None
                    and ablation_force_sequence_this_step_indices is not None
                ):
                    # The hold-action env.step above already integrated one real
                    # physics step from the qpos/qvel we injected before stepping, and
                    # its resulting policy_obs went through the normal wrapper stack
                    # (FrameStack/ConcatDim), so it is correctly shaped. We only
                    # re-affirm the live MuJoCo q/qvel here for downstream consumers
                    # that read sim state directly (e.g. the next forced frame's
                    # `live_q`) — deliberately NOT re-deriving policy_obs from a raw
                    # env snapshot, since bypassing the wrapper stack drops the
                    # FrameStack dimension and corrupts obs shapes fed to ACT.
                    reaffirm_seen = set()
                    for candidate in (env, policy_env, safety_runtime_env):
                        if candidate is None or id(candidate) in reaffirm_seen:
                            continue
                        reaffirm_seen.add(id(candidate))
                        _set_h1_q_for_ablation(
                            candidate,
                            ablation_force_sequence_this_step_q,
                            ablation_force_sequence_this_step_indices,
                            zero_velocity=bool(
                                getattr(args, "ablation_force_planned_recovery_q_zero_velocity", True)
                            ),
                            target_qvel=ablation_force_sequence_this_step_qvel,
                        )

                if (
                    ablation_force_sequence_applied_this_step
                    and pending_ablation_force_q_sequence_history is not None
                    and isinstance(policy_obs, dict)
                ):
                    pending_ablation_force_q_sequence_history.append(copy.deepcopy(policy_obs))
                    try:
                        q_seq_len = int(np.asarray(pending_ablation_force_q_sequence).shape[0])
                    except Exception:
                        q_seq_len = 0
                    if int(pending_ablation_force_q_sequence_index) >= q_seq_len:
                        pending_ablation_policy_obs_history = list(
                            pending_ablation_force_q_sequence_history
                        )
                        safety_info = dict(safety_info)
                        if pending_ablation_force_q_sequence_target_action is not None:
                            safety_info["recover_resume_window_target_action"] = (
                                pending_ablation_force_q_sequence_target_action
                            )
                        safety_info.update(
                            {
                                "ablation_force_planned_recovery_q_sequence_completed": True,
                                "ablation_force_planned_recovery_q_sequence_obs_count": int(
                                    len(pending_ablation_policy_obs_history)
                                ),
                            }
                        )
                        pending_ablation_force_q_sequence = None
                        pending_ablation_force_q_sequence_history = None
                        pending_ablation_force_q_sequence_indices = None

                if pending_ablation_policy_obs_history:
                    (
                        policy_obs,
                        ablation_window_seed_count,
                        ablation_frame_stack_seed_count,
                    ) = _seed_policy_visual_history_after_recovery(
                        policy_obs,
                        pending_ablation_policy_obs_history,
                        env=env,
                        policy_env=policy_env,
                    )
                    target_action = _safe_info_get(safety_info, "recover_resume_window_target_action")
                    target_action_source = "recover_resume_window_target_action" if target_action is not None else None
                    if target_action is None and pending_ablation_force_q_sequence_target_action is not None:
                        target_action = pending_ablation_force_q_sequence_target_action
                        target_action_source = pending_ablation_force_q_sequence_target_action_source
                    action_history_seed_count = 0
                    action_history_seed_source = None
                    if target_action is not None:
                        seed_seen: set[int] = set()
                        for candidate in (env, policy_env):
                            if candidate is None or id(candidate) in seed_seen:
                                continue
                            seed_seen.add(id(candidate))
                            try:
                                action_history_seed_count += _seed_action_sequence_history_with_nominal_actions(
                                    candidate,
                                    target_action,
                                    history_window_len=len(pending_ablation_policy_obs_history),
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("Could not seed action-sequence history for ablation: %s", exc)
                        if action_history_seed_count > 0:
                            action_history_seed_source = target_action_source
                    seeded_policy_obs_for_action = None
                    try:
                        seeded_policy_obs_for_action = copy.deepcopy(
                            _adapt_policy_obs_to_space(policy_obs, policy_observation_space)
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Could not snapshot seeded policy obs for ablation diagnostics: %s", exc)
                    pending_ablation_resume_diag = {
                        "seed_step": int(step),
                        "seeded_low_dim_state": (
                            np.asarray(policy_obs.get("low_dim_state"), dtype=np.float32).copy()
                            if isinstance(policy_obs, dict) and policy_obs.get("low_dim_state") is not None
                            else None
                        ),
                        "seeded_policy_obs_for_action": seeded_policy_obs_for_action,
                        "seeded_visual_pose_snapshot": _mujoco_visual_pose_snapshot(env),
                        "target_action": (
                            np.asarray(target_action, dtype=np.float32).copy()
                            if target_action is not None
                            else None
                        ),
                        "target_action_source": target_action_source,
                    }
                    safety_info = dict(safety_info)
                    safety_info.update(
                        {
                            "ablation_force_planned_recovery_q_window_seed_count": int(ablation_window_seed_count),
                            "ablation_force_planned_recovery_q_frame_stack_seed_count": int(
                                ablation_frame_stack_seed_count
                            ),
                            "ablation_force_planned_recovery_q_window_seed_source_count": int(
                                len(pending_ablation_policy_obs_history)
                            ),
                            "ablation_force_planned_recovery_q_action_history_seed_count": int(
                                action_history_seed_count
                            ),
                            "ablation_force_planned_recovery_q_action_history_seed_source": action_history_seed_source,
                        }
                    )
                    if (
                        bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_completed"))
                        and ablation_pure_act_resume_total_steps > 0
                    ):
                        ablation_pure_act_resume_steps_left = max(
                            int(ablation_pure_act_resume_steps_left),
                            int(ablation_pure_act_resume_total_steps),
                        )
                        safety_info.update(
                            {
                                "ablation_pure_act_resume_enabled": True,
                                "ablation_pure_act_resume_scheduled_steps": int(
                                    ablation_pure_act_resume_steps_left
                                ),
                                "ablation_pure_act_resume_total_steps": int(
                                    ablation_pure_act_resume_total_steps
                                ),
                            }
                        )
                        pending_ablation_force_q_sequence_target_action = None
                        pending_ablation_force_q_sequence_target_action_source = None
                    pending_ablation_policy_obs_history = None

                if (
                    mpc_replay_diagnostic_logging_enabled
                    and len(all_mpc_replay_diagnostic_records) < max(0, int(args.mpc_replay_diagnostics_max_events))
                    and _should_log_mpc_replay_diagnostic(safety_info)
                ):
                    diagnostic_record = _collect_mpc_replay_diagnostic(
                        episode=episode,
                        step=step,
                        safety_info=safety_info,
                        safe_env_action=safe_env_action,
                        pre_mujoco_snapshot=mpc_replay_pre_mujoco_snapshot,
                        post_mujoco_snapshot=_mujoco_state_snapshot(env),
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                    )
                    if diagnostic_record is not None:
                        episode_mpc_replay_diagnostic_records.append(diagnostic_record)
                        all_mpc_replay_diagnostic_records.append(diagnostic_record)

                if (
                    nominal_rollout_diagnostic_context is not None
                    or (chunk_filter_mode and nominal_pred_q_next_for_feedback is not None)
                ):
                    try:
                        post_h1state = extract_h1_state(env)
                        post_q_full = np.asarray(post_h1state.q_full, dtype=np.float32).reshape(-1)
                        post_qd_full = np.asarray(post_h1state.qd_full, dtype=np.float32).reshape(-1)
                        (
                            post_min_h,
                            _post_h_values,
                            post_h_violation,
                            post_h_pair_label,
                            post_live_info,
                        ) = compute_oscbf_full_arm_h_monitor(
                            filt=oscbf,
                            env=safety_runtime_env,
                            obs=obs,
                            q_full=post_q_full,
                            qd_full=post_qd_full,
                            clearance_threshold=0.0,
                            return_details=True,
                        )
                        nominal_rollout_post_step_state = {
                            "q_after": post_q_full,
                            "qd_after": post_qd_full,
                            "post_live_min_h": post_min_h,
                            "post_live_h_violation": bool(post_h_violation),
                            "post_live_h_pair_label": post_h_pair_label,
                            "post_live_min_clearance": _safe_info_get(post_live_info, "live_min_clearance"),
                        }
                    except Exception as exc:  # noqa: BLE001
                        nominal_rollout_post_step_state = {"post_step_state_error": repr(exc)}

                task_state_after = (
                    _diagnostic_task_state(env)
                    if args.diagnostics_enabled
                    else {
                        "drawer_open_distance": None,
                        "drawer_open_fraction": None,
                        "drawer_joint_position": None,
                        "task_progress": None,
                        "ee_object_distance": None,
                        "object_state": None,
                    }
                )
                if trajectory_logging_enabled:
                    executed_sample = _annotate_executed_trajectory_sample(
                        _robot_ee_trajectory_sample(env, episode, step + 1, task_state_after),
                        safety_info,
                    )
                    if executed_sample is not None:
                        episode_executed_policy_trajectory_samples.append(executed_sample)
                        all_executed_policy_trajectory_samples.append(executed_sample)

                task_progress_delta = _diagnostic_progress_delta(
                    task_state_before,
                    task_state_after,
                )
                progress_cache_epsilon = float(
                    getattr(args, "post_recovery_mid_progress_epsilon", 0.001)
                )
                current_mode_for_progress_cache = (
                    _safe_info_get(safety_info, "safety_mode")
                    or _safe_info_get(safety_info, "mode")
                )
                if (
                    replay_actions is None
                    and chunk_filter_mode
                    and task_progress_delta is not None
                    and float(task_progress_delta) > progress_cache_epsilon
                    and current_mode_for_progress_cache == "pass_through"
                ):
                    last_progressing_act_chunk = np.asarray(
                        env_action,
                        dtype=np.float32,
                    ).copy()
                    last_progressing_act_chunk_step = int(step)

                if chunk_filter_mode:
                    actual_q_after_for_feedback = None
                    if isinstance(nominal_rollout_post_step_state, dict):
                        actual_q_after_for_feedback = nominal_rollout_post_step_state.get("q_after")
                    rollout_feedback = _rollout_residual_feedback(
                        nominal_pred_q_next_for_feedback,
                        actual_q_after_for_feedback,
                        state_idx,
                    )
                    if rollout_feedback is not None:
                        last_rollout_residual_state = rollout_feedback.get("rollout_residual_state")
                        last_rollout_residual_l2 = rollout_feedback.get("rollout_residual_l2")
                        last_rollout_residual_max_abs = rollout_feedback.get("rollout_residual_max_abs")
                        last_rollout_residual_base_l2 = rollout_feedback.get("rollout_residual_base_l2")
                        last_rollout_residual_arm_l2 = rollout_feedback.get("rollout_residual_arm_l2")
                        last_rollout_prediction_untrusted = bool(
                            rollout_feedback.get("rollout_prediction_untrusted", False)
                        )
                    else:
                        last_rollout_residual_state = None
                        last_rollout_residual_l2 = None
                        last_rollout_residual_max_abs = None
                        last_rollout_residual_base_l2 = None
                        last_rollout_residual_arm_l2 = None
                        last_rollout_prediction_untrusted = False

                if replay_actions is None and chunk_filter_mode:
                    recovery_history_mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
                    recovery_history_source = _safe_info_get(safety_info, "deformation_source")
                    recovery_history_phase = _safe_info_get(safety_info, "recovery_phase")
                    recovery_history_committed_mode = _safe_info_get(safety_info, "committed_chunk_mode")
                    explicit_recovery_history_reset_request = bool(
                        _safe_info_get(safety_info, "request_action_history_reset_after_recovery")
                    )
                    implicit_recovery_handoff_reset_request = bool(
                        getattr(
                            args,
                            "reset_action_history_after_intervention_boundary",
                            False,
                        )
                        and args.reset_action_history_after_recovery
                        and last_safety_intervention_active
                        and not safety_intervention_active
                        and len(recovery_policy_obs_history) > 0
                    )
                    recovery_handoff_reset_request = bool(
                        explicit_recovery_history_reset_request
                        or implicit_recovery_handoff_reset_request
                    )
                    record_recovery_history = bool(
                        _is_safety_intervention_mode(safety_info)
                        or explicit_recovery_history_reset_request
                        or recovery_history_mode in {
                            "recover",
                            "committed_explicit_recovery",
                            "horizon_deform",
                            "horizon_brake",
                            "phase_reanchor",
                        }
                        or recovery_history_source in {
                            "committed_explicit_recovery",
                            "explicit_recover_deform",
                            "chunk_deform",
                            "horizon_deform",
                            "horizon_brake",
                            "phase_reanchor",
                        }
                        or recovery_history_phase in {
                            "recover",
                            "horizon_deform",
                            "resume_act",
                        }
                        or recovery_history_committed_mode in {
                            "recover",
                            "horizon_deform",
                        }
                    )
                    if record_recovery_history:
                        recovery_policy_obs_history.append(copy.deepcopy(policy_obs))
                        if len(recovery_policy_obs_history) > 8:
                            recovery_policy_obs_history = recovery_policy_obs_history[-8:]
                    elif not recovery_handoff_reset_request:
                        recovery_policy_obs_history.clear()
                else:
                    explicit_recovery_history_reset_request = False
                    implicit_recovery_handoff_reset_request = False
                    recovery_handoff_reset_request = False

                if replay_actions is None:
                    if reset_policy_after_intervention:
                        reset_count = _reset_action_sequence_history(env)
                        if policy_env is not None:
                            reset_count += _reset_action_sequence_history(policy_env)
                        if episode == 0 and args.debug and reset_count > 0:
                            mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
                            print(
                                "intervention_requery: reset_action_history "
                                f"step={step} mode={mode} reset_wrappers={reset_count}"
                            )
                    if (
                        chunk_filter_mode
                        and recovery_handoff_reset_request
                    ):
                        reset_count = 0
                        low_level_sync_count = 0
                        visual_reset_count = 0
                        visual_seed_count = 0
                        visual_seed_source_count = len(recovery_policy_obs_history)
                        handoff_fresh_act_seed_enabled = bool(
                            getattr(args, "handoff_seed_action_history_from_fresh_act", True)
                        )
                        effective_action_history_reset = bool(
                            args.reset_action_history_after_recovery
                            or handoff_fresh_act_seed_enabled
                        )
                        if effective_action_history_reset:
                            reset_count = _reset_action_sequence_history(env)
                            if policy_env is not None:
                                reset_count += _reset_action_sequence_history(policy_env)
                            policy_step = 0
                            policy_hold_active = True
                        if bool(getattr(args, "handoff_sanitize_controller_state", True)):
                            sync_seen: set[int] = set()
                            for candidate in (env, policy_env, safety_runtime_env):
                                if candidate is None or id(candidate) in sync_seen:
                                    continue
                                sync_seen.add(id(candidate))
                                try:
                                    low_level_sync_count += int(
                                        _sync_robot_low_level_hold_state(candidate)
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    logger.debug(
                                        "Could not sanitize low-level controller state "
                                        "after recovery handoff: %s",
                                        exc,
                                    )
                        if args.reset_visual_history_after_recovery:
                            policy_obs, visual_reset_count = _reset_policy_visual_history_after_recovery(
                                env,
                                policy_env,
                                policy_obs,
                            )
                        else:
                            (
                                policy_obs,
                                visual_seed_count,
                                visual_frame_stack_seed_count,
                            ) = _seed_policy_visual_history_after_recovery(
                                policy_obs,
                                recovery_policy_obs_history,
                                env=env,
                                policy_env=policy_env,
                            )
                        recovery_policy_obs_history.clear()
                        safety_info = dict(safety_info)
                        safety_info.update(
                            {
                                "recovery_action_history_reset": bool(effective_action_history_reset),
                                "recovery_action_history_reset_count": int(reset_count),
                                "recovery_action_history_reset_config": bool(
                                    args.reset_action_history_after_recovery
                                ),
                                "recovery_action_history_reset_reason": (
                                    "config"
                                    if explicit_recovery_history_reset_request
                                    and bool(args.reset_action_history_after_recovery)
                                    else (
                                        "intervention_boundary"
                                        if implicit_recovery_handoff_reset_request
                                        and bool(args.reset_action_history_after_recovery)
                                        else (
                                            "fresh_act_seed"
                                            if handoff_fresh_act_seed_enabled
                                            else None
                                        )
                                    )
                                ),
                                "request_action_history_reset_after_recovery": bool(
                                    explicit_recovery_history_reset_request
                                ),
                                "implicit_action_history_reset_after_intervention": bool(
                                    implicit_recovery_handoff_reset_request
                                ),
                                "recovery_action_history_reset_request_reason": (
                                    "explicit_filter_request"
                                    if explicit_recovery_history_reset_request
                                    else (
                                        "intervention_boundary"
                                        if implicit_recovery_handoff_reset_request
                                        else None
                                    )
                                ),
                                "recovery_low_level_hold_sync": bool(
                                    getattr(args, "handoff_sanitize_controller_state", True)
                                ),
                                "recovery_low_level_hold_sync_count": int(low_level_sync_count),
                                "recovery_visual_history_reset": bool(args.reset_visual_history_after_recovery),
                                "recovery_visual_frame_stack_seed_count": int(
                                    locals().get("visual_frame_stack_seed_count", 0)
                                ),
                                "recovery_visual_history_reset_count": int(visual_reset_count),
                                "recovery_visual_history_seed": bool(
                                    (not args.reset_visual_history_after_recovery)
                                    and visual_seed_count > 0
                                ),
                                "recovery_visual_history_seed_count": int(visual_seed_count),
                                "recovery_visual_history_seed_source_count": int(visual_seed_source_count),
                                "recovery_policy_obs_history_seed": bool(
                                    (not args.reset_visual_history_after_recovery)
                                    and visual_seed_count > 0
                                ),
                                "recovery_policy_obs_history_seed_count": int(visual_seed_count),
                                "recovery_policy_obs_history_seed_source_count": int(visual_seed_source_count),
                                "post_recovery_act_bridge_started": bool(
                                    post_recovery_act_bridge_total_steps > 0
                                ),
                            }
                        )
                        if post_recovery_act_bridge_total_steps > 0:
                            post_recovery_act_bridge_steps_left = max(
                                int(post_recovery_act_bridge_steps_left),
                                int(post_recovery_act_bridge_total_steps),
                            )
                            post_recovery_seed_fresh_act_history_pending = bool(
                                handoff_fresh_act_seed_enabled
                            )
                            safety_info.update(
                                {
                                    "post_recovery_act_bridge_steps_left": int(
                                        post_recovery_act_bridge_steps_left
                                    ),
                                    "post_recovery_act_bridge_total_steps": int(
                                        post_recovery_act_bridge_total_steps
                                    ),
                                    "post_recovery_act_bridge_fresh_action_seed_pending": bool(
                                        post_recovery_seed_fresh_act_history_pending
                                    ),
                                }
                            )
                        phase_reanchor_early_release_grace = bool(
                            _safe_info_get(safety_info, "phase_reanchor_early_release_triggered")
                            and (
                                (_safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode"))
                                == "phase_reanchor"
                            )
                        )
                        phase_reanchor_early_release_grace_steps = 0
                        if phase_reanchor_early_release_grace:
                            phase_reanchor_early_release_grace_steps = max(
                                1,
                                int(getattr(args, "phase_reanchor_early_release_act_grace_steps", 16)),
                            )
                            phase_bridge_monitor_steps = max(
                                phase_reanchor_early_release_grace_steps,
                                int(getattr(args, "post_recovery_act_bridge_no_progress_monitor_steps", 0)),
                            )
                            post_recovery_act_bridge_steps_left = max(
                                post_recovery_act_bridge_steps_left,
                                phase_bridge_monitor_steps,
                            )
                            post_recovery_act_bridge_total_steps = max(
                                post_recovery_act_bridge_total_steps,
                                phase_bridge_monitor_steps,
                            )
                            post_recovery_act_bridge_progress_best = None
                            post_recovery_act_bridge_no_progress_count = 0
                            post_recovery_task_guard_steps_left = 0
                            phase_reanchor_cooldown_left = max(
                                phase_reanchor_cooldown_left,
                                phase_reanchor_early_release_grace_steps,
                            )
                            policy_hold_active = False
                            safety_info = dict(safety_info)
                            safety_info.update(
                                {
                                    "phase_reanchor_early_release_act_grace": True,
                                    "phase_reanchor_early_release_act_grace_steps": int(
                                        phase_reanchor_early_release_grace_steps
                                    ),
                                }
                            )
                        if args.post_recovery_task_guard:
                            progress_after_recovery = _finite_task_progress(task_state_after)
                            if progress_after_recovery is not None:
                                if (
                                    post_recovery_task_guard_best_progress is None
                                    or progress_after_recovery > post_recovery_task_guard_best_progress
                                ):
                                    post_recovery_task_guard_best_progress = progress_after_recovery
                            guard_ready, guard_phase_reason = _post_recovery_task_guard_ready(
                                task_state_after,
                                phase_reanchor_state,
                                args,
                            )
                            if phase_reanchor_early_release_grace:
                                guard_ready = False
                                guard_phase_reason = (
                                    f"phase_reanchor_early_release_act_grace:{phase_reanchor_early_release_grace_steps}"
                                )
                            if guard_ready:
                                post_recovery_task_guard_steps_left = max(
                                    post_recovery_task_guard_steps_left,
                                    int(args.post_recovery_task_guard_steps),
                                )
                                post_recovery_task_guard_reason = "recovery_completed:" + str(
                                    guard_phase_reason
                                )
                                if args.post_recovery_task_guard_force_gripper:
                                    gripper_latched = True
                                reanchor_allowed, _guard_phase = _post_recovery_task_guard_reanchor_allowed(
                                    phase_reanchor_state,
                                    args,
                                )
                                if reanchor_allowed:
                                    phase_reanchor_steps_left = max(
                                        phase_reanchor_steps_left,
                                        int(args.post_recovery_task_guard_steps),
                                    )
                                    phase_reanchor_cooldown_left = 0
                                    phase_reanchor_bridge_preload_start_progress = None
                                    phase_reanchor_bridge_preload_count = 0
                                    phase_reanchor_bridge_preload_validated_latched = False
                                    phase_reanchor_bridge_preload_reason_last = None
                                    phase_reanchor_bridge_preload_progress_delta_last = None
                                    phase_reanchor_bridge_preload_handle_dist_last = None
                                post_recovery_guard_active_after_reset = True
                                post_recovery_guard_reanchor_after_reset = bool(reanchor_allowed)
                            else:
                                post_recovery_task_guard_reason = "suppressed:" + str(
                                    guard_phase_reason
                                )
                                post_recovery_guard_active_after_reset = False
                                post_recovery_guard_reanchor_after_reset = False
                            safety_info.update(
                                {
                                    "gripper_latched": bool(gripper_latched),
                                    "post_recovery_task_guard_active": bool(post_recovery_guard_active_after_reset),
                                    "post_recovery_task_guard_steps_left": int(post_recovery_task_guard_steps_left),
                                    "post_recovery_task_guard_reason": post_recovery_task_guard_reason,
                                    "post_recovery_task_guard_best_progress": post_recovery_task_guard_best_progress,
                                    "post_recovery_progress_regression": post_recovery_progress_regression,
                                    "post_recovery_reanchor_started": bool(post_recovery_guard_reanchor_after_reset),
                                    "post_recovery_no_progress_count": int(post_recovery_no_progress_count),
                                    "post_recovery_mid_progress_no_progress_count": int(
                                        post_recovery_mid_progress_no_progress_count
                                    ),
                                    "post_recovery_mid_progress_best_progress": post_recovery_mid_progress_best_progress,
                                    "post_recovery_mid_progress_best_distance": post_recovery_mid_progress_best_distance,
                                    "post_recovery_mid_progress_distance_regression": post_recovery_mid_progress_distance_regression,
                                    "post_recovery_mid_progress_reseed_triggered": bool(
                                        post_recovery_mid_progress_reseed_triggered
                                    ),
                                    "post_recovery_mid_progress_reseed_reset_count": int(
                                        post_recovery_mid_progress_reseed_reset_count
                                    ),
                                    "post_recovery_mid_progress_reseed_reason": post_recovery_mid_progress_reseed_reason,
                                    "post_recovery_mid_progress_prior_action_seed_count": int(
                                        post_recovery_mid_progress_prior_action_seed_count
                                    ),
                                    "post_recovery_mid_progress_prior_action_seed_step": post_recovery_mid_progress_prior_action_seed_step,
                                    "post_recovery_mid_progress_prior_action_seed_age": post_recovery_mid_progress_prior_action_seed_age,
                                    "post_recovery_no_progress_triggered": bool(post_recovery_no_progress_triggered),
                                    "post_recovery_no_progress_target_distance": post_recovery_no_progress_target_distance,
                                    "post_recovery_no_progress_distance_source": post_recovery_no_progress_distance_source,
                                }
                            )
                        if episode == 0 or args.debug:
                            mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
                            print(
                                "recovery_requery: reset_histories "
                                f"step={step} mode={mode} "
                                f"action_reset_wrappers={reset_count} "
                                f"visual_reset_entries={visual_reset_count} "
                                f"visual_seed_entries={visual_seed_count} "
                                f"visual_seed_source_obs={visual_seed_source_count}"
                            )

                    if phase_reanchor_reset_after_step:
                        phase_reanchor_best_target_distance = None
                        phase_reanchor_best_target_signature = None
                        phase_reanchor_best_control_distance = None
                        phase_reanchor_best_control_signature = None
                        phase_reanchor_taskspace_worsen_count = 0
                        phase_reanchor_suppress_q_servo_steps_left = 0
                        phase_bridge_resume_ok = bool(_safe_info_get(safety_info, "resume_affordance_ok"))
                        phase_bridge_live_ready = _safe_info_get(safety_info, "phase_reanchor_live_release_ready")
                        phase_bridge_live_reason = _safe_info_get(safety_info, "phase_reanchor_live_release_reason")
                        if phase_bridge_live_ready is None:
                            (
                                phase_bridge_live_ready,
                                phase_bridge_live_reason,
                                _phase_bridge_live_target_error,
                                _phase_bridge_live_handle_dist,
                            ) = _phase_reanchor_live_release_status(
                                phase_reanchor_action_state,
                                args,
                            )
                        (
                            phase_bridge_contact_ready,
                            phase_bridge_contact_reason,
                            phase_bridge_contact_handle_dist,
                            phase_bridge_contact_handle_limit,
                        ) = _phase_reanchor_bridge_contact_status(
                            phase_reanchor_action_state,
                            args,
                        )
                        phase_bridge_preload_enabled = bool(
                            getattr(args, "phase_reanchor_bridge_preload_validation", False)
                        )
                        phase_bridge_preload_validated = bool(
                            (not phase_bridge_preload_enabled)
                            or phase_reanchor_bridge_preload_validated_latched
                        )
                        phase_bridge_preload_reason = (
                            "disabled"
                            if not phase_bridge_preload_enabled
                            else (
                                phase_reanchor_bridge_preload_reason_last
                                or "not_validated_before_bridge_seed"
                            )
                        )
                        phase_bridge_preload_steps = int(phase_reanchor_bridge_preload_count)
                        phase_bridge_preload_progress_delta = (
                            phase_reanchor_bridge_preload_progress_delta_last
                        )
                        phase_bridge_preload_handle_dist = (
                            phase_reanchor_bridge_preload_handle_dist_last
                            if phase_reanchor_bridge_preload_handle_dist_last is not None
                            else phase_bridge_contact_handle_dist
                        )
                        phase_bridge_preload_handle_limit = float(
                            getattr(args, "phase_reanchor_bridge_preload_handle_dist", 0.245)
                        )
                        phase_bridge_seed_ok = bool(
                            phase_bridge_resume_ok
                            and phase_bridge_live_ready
                            and phase_bridge_contact_ready
                            and phase_bridge_preload_validated
                        )
                        phase_bridge_nominal_q_l2 = None
                        phase_bridge_nominal_q_window_l2_mean = None
                        phase_bridge_nominal_q_window_l2_max = None
                        phase_bridge_nominal_q_window_len = None
                        phase_bridge_nominal_q_ok = None
                        phase_bridge_nominal_q_adapted_l2 = None
                        phase_bridge_nominal_q_adapted_window_l2_mean = None
                        phase_bridge_nominal_q_adapted_window_l2_max = None
                        phase_bridge_nominal_q_adapted_dims = None
                        phase_bridge_nominal_q_base_l2 = None
                        phase_bridge_nominal_q_arm_l2 = None
                        phase_bridge_nominal_q_track_base = bool(
                            getattr(args, "phase_reanchor_nominal_window_track_base", True)
                        )
                        phase_bridge_nominal_q_rows = None
                        phase_bridge_nominal_q_live_dim = None
                        phase_bridge_nominal_q_base_state_idx = np.asarray([], dtype=np.int64)
                        if isinstance(phase_reanchor_action_state, dict):
                            try:
                                q_window = phase_reanchor_action_state.get("nominal_reentry_q_window")
                                if q_window is not None:
                                    q_rows = np.asarray(q_window, dtype=np.float64)
                                    if q_rows.ndim == 1:
                                        q_rows = q_rows.reshape(1, -1)
                                    elif q_rows.ndim > 2:
                                        q_rows = q_rows.reshape((-1, q_rows.shape[-1]))
                                    live_q = np.asarray(q_full, dtype=np.float64).reshape(-1)
                                    dim = min(live_q.size, q_rows.shape[-1])
                                    if dim > 0 and np.isfinite(q_rows[:, :dim]).all() and np.isfinite(live_q[:dim]).all():
                                        phase_bridge_nominal_q_rows = np.asarray(q_rows, dtype=np.float64).copy()
                                        phase_bridge_nominal_q_live_dim = int(dim)
                                        q_dists = np.linalg.norm(q_rows[:, :dim] - live_q[:dim][None, :], axis=1)
                                        phase_bridge_nominal_q_l2 = float(q_dists[-1])
                                        phase_bridge_nominal_q_window_l2_mean = float(np.mean(q_dists))
                                        phase_bridge_nominal_q_window_l2_max = float(np.max(q_dists))
                                        phase_bridge_nominal_q_window_len = int(q_rows.shape[0])
                                        compare_idx = np.arange(dim, dtype=np.int64)
                                        base_state_idx = np.asarray(
                                            getattr(oscbf, "bigym_state_base_indices", []),
                                            dtype=np.int64,
                                        )
                                        phase_bridge_nominal_q_base_state_idx = base_state_idx.copy()
                                        arm_state_idx = np.asarray(
                                            getattr(oscbf, "bigym_state_arm_indices", []),
                                            dtype=np.int64,
                                        )
                                        base_state_idx = base_state_idx[
                                            (base_state_idx >= 0) & (base_state_idx < dim)
                                        ]
                                        arm_state_idx = arm_state_idx[
                                            (arm_state_idx >= 0) & (arm_state_idx < dim)
                                        ]
                                        if base_state_idx.size > 0:
                                            base_dists = np.linalg.norm(
                                                q_rows[:, base_state_idx] - live_q[base_state_idx][None, :],
                                                axis=1,
                                            )
                                            phase_bridge_nominal_q_base_l2 = float(base_dists[-1])
                                        if arm_state_idx.size > 0:
                                            arm_dists = np.linalg.norm(
                                                q_rows[:, arm_state_idx] - live_q[arm_state_idx][None, :],
                                                axis=1,
                                            )
                                            phase_bridge_nominal_q_arm_l2 = float(arm_dists[-1])
                                        if not phase_bridge_nominal_q_track_base and arm_state_idx.size > 0:
                                            compare_idx = arm_state_idx
                                            phase_bridge_nominal_q_adapted_dims = "arm_only_base_live_taskspace"
                                        else:
                                            phase_bridge_nominal_q_adapted_dims = "full_controlled"
                                        adapted_q_dists = np.linalg.norm(
                                            q_rows[:, compare_idx] - live_q[compare_idx][None, :],
                                            axis=1,
                                        )
                                        phase_bridge_nominal_q_adapted_l2 = float(adapted_q_dists[-1])
                                        phase_bridge_nominal_q_adapted_window_l2_mean = float(
                                            np.mean(adapted_q_dists)
                                        )
                                        phase_bridge_nominal_q_adapted_window_l2_max = float(
                                            np.max(adapted_q_dists)
                                        )
                                        phase_bridge_nominal_q_ok = bool(
                                            phase_bridge_nominal_q_adapted_l2 <= 0.35
                                        )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("Could not compute phase-reanchor nominal-q bridge readiness: %s", exc)
                        phase_bridge_seed_source_count = int(len(recovery_policy_obs_history))
                        phase_bridge_obs_seed_source = "recovery_policy_obs_history"
                        phase_bridge_obs_seed_restore_count = 0
                        phase_bridge_obs_seed_window_count = 0
                        phase_bridge_visual_seed_count = 0
                        phase_bridge_frame_stack_seed_count = 0
                        phase_bridge_action_seed_count = 0
                        phase_bridge_action_seed_source = None
                        phase_bridge_action_window_len = None
                        phase_bridge_act_vs_nominal_l2 = None
                        phase_bridge_act_vs_nominal_cosine = None
                        phase_bridge_target_action = None
                        phase_bridge_action_base_adapted = False
                        phase_bridge_action_base_adapted_dims = 0
                        if isinstance(phase_reanchor_action_state, dict):
                            phase_bridge_target_action = phase_reanchor_action_state.get(
                                "nominal_reentry_action_window"
                            )
                        if phase_bridge_target_action is not None:
                            phase_bridge_action_seed_source = "nominal_reentry_action_window"
                        else:
                            phase_bridge_target_action = safe_env_action
                            phase_bridge_action_seed_source = "phase_reanchor_safe_action"
                        if (
                            phase_bridge_target_action is not None
                            and not phase_bridge_nominal_q_track_base
                        ):
                            try:
                                target_rows = _ablation_action_rows(phase_bridge_target_action)
                                safe_rows = _ablation_action_rows(safe_env_action)
                                base_action_idx = np.asarray(
                                    getattr(oscbf, "bigym_action_base_indices", []),
                                    dtype=np.int64,
                                )
                                if (
                                    target_rows is not None
                                    and safe_rows is not None
                                    and safe_rows.shape[0] > 0
                                    and base_action_idx.size > 0
                                ):
                                    adapted_rows = np.asarray(target_rows, dtype=np.float32).copy()
                                    base_action_idx = base_action_idx[
                                        (base_action_idx >= 0) & (base_action_idx < adapted_rows.shape[1])
                                    ]
                                    safe_first = np.asarray(safe_rows[0], dtype=np.float32).reshape(-1)
                                    base_action_idx = base_action_idx[base_action_idx < safe_first.size]
                                    if base_action_idx.size > 0:
                                        adapted_rows[:, base_action_idx] = safe_first[base_action_idx]
                                        phase_bridge_target_action = adapted_rows
                                        phase_bridge_action_base_adapted = True
                                        phase_bridge_action_base_adapted_dims = int(base_action_idx.size)
                                        phase_bridge_action_seed_source = (
                                            f"{phase_bridge_action_seed_source}_base_live_adapted"
                                        )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("Could not adapt phase-reanchor bridge base action history: %s", exc)
                        phase_bridge_target_rows = _ablation_action_rows(phase_bridge_target_action)
                        phase_bridge_action_agreement_ok = None
                        if phase_bridge_target_rows is not None:
                            phase_bridge_action_window_len = int(phase_bridge_target_rows.shape[0])
                            try:
                                act_first_arr = np.asarray(first_action, dtype=np.float32).reshape(-1)
                                nominal_first_arr = np.asarray(phase_bridge_target_rows[0], dtype=np.float32).reshape(-1)
                                dim = min(act_first_arr.size, nominal_first_arr.size)
                                if dim > 0:
                                    a = act_first_arr[:dim]
                                    b = nominal_first_arr[:dim]
                                    phase_bridge_act_vs_nominal_l2 = float(np.linalg.norm(a - b))
                                    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                                    phase_bridge_act_vs_nominal_cosine = float(np.dot(a, b) / denom)
                                    action_l2_limit = float(
                                        getattr(args, "phase_reanchor_bridge_action_agreement_l2", 1.0)
                                    )
                                    action_cosine_limit = float(
                                        getattr(args, "phase_reanchor_bridge_action_agreement_cosine", 0.8)
                                    )
                                    action_agreement_mode = str(
                                        getattr(args, "phase_reanchor_bridge_action_agreement_mode", "and")
                                    )
                                    if action_agreement_mode == "or":
                                        phase_bridge_action_agreement_ok = bool(
                                            phase_bridge_act_vs_nominal_l2 <= action_l2_limit
                                            or phase_bridge_act_vs_nominal_cosine >= action_cosine_limit
                                        )
                                    else:
                                        phase_bridge_action_agreement_ok = bool(
                                            phase_bridge_act_vs_nominal_l2 <= action_l2_limit
                                            and phase_bridge_act_vs_nominal_cosine >= action_cosine_limit
                                        )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("Could not compute phase-reanchor ACT-vs-nominal action diagnostic: %s", exc)
                        phase_bridge_live_taskspace_ok = bool(
                            phase_bridge_live_ready
                            and phase_bridge_contact_ready
                            and phase_bridge_preload_validated
                        )
                        phase_bridge_nominal_action_window_ok = bool(
                            isinstance(phase_bridge_action_seed_source, str)
                            and phase_bridge_action_seed_source.startswith(
                                "nominal_reentry_action_window"
                            )
                            and phase_bridge_action_window_len is not None
                            and phase_bridge_action_window_len >= int(
                                getattr(args, "phase_reanchor_nominal_window_len", 4)
                            )
                        )
                        phase_bridge_requires_resume_affordance = bool(
                            getattr(args, "phase_reanchor_bridge_requires_resume_affordance", True)
                        )
                        phase_bridge_nominal_history_ok = bool(
                            (phase_bridge_resume_ok or not phase_bridge_requires_resume_affordance)
                            and phase_bridge_nominal_q_ok is True
                            and phase_bridge_action_agreement_ok is True
                            and phase_bridge_nominal_action_window_ok
                        )
                        phase_bridge_seed_mode = str(
                            getattr(args, "phase_reanchor_bridge_seed_mode", "live_taskspace")
                        )
                        phase_bridge_live_veto = bool(
                            phase_bridge_live_reason == "live_taskspace_geometry_untrusted"
                            or phase_bridge_contact_reason == "bridge_contact_geometry_untrusted"
                        )
                        if phase_bridge_seed_mode == "nominal_history":
                            phase_bridge_readiness_ok = bool(phase_bridge_nominal_history_ok)
                        elif phase_bridge_seed_mode == "nominal_history_with_live_veto":
                            phase_bridge_readiness_ok = bool(
                                phase_bridge_nominal_history_ok and not phase_bridge_live_veto
                            )
                        else:
                            phase_bridge_seed_mode = "live_taskspace"
                            phase_bridge_readiness_ok = bool(
                                phase_bridge_nominal_history_ok and phase_bridge_live_taskspace_ok
                            )
                        phase_bridge_seed_blockers: list[str] = []
                        if phase_bridge_requires_resume_affordance and not phase_bridge_resume_ok:
                            phase_bridge_seed_blockers.append("resume_affordance")
                        if phase_bridge_nominal_q_ok is not True:
                            phase_bridge_seed_blockers.append("nominal_q")
                        if phase_bridge_action_agreement_ok is not True:
                            phase_bridge_seed_blockers.append("action_agreement")
                        if not phase_bridge_nominal_action_window_ok:
                            phase_bridge_seed_blockers.append("nominal_action_window")
                        if phase_bridge_seed_mode == "live_taskspace" and not phase_bridge_live_ready:
                            phase_bridge_seed_blockers.append(f"live_release:{phase_bridge_live_reason}")
                        if phase_bridge_seed_mode == "live_taskspace" and not phase_bridge_contact_ready:
                            phase_bridge_seed_blockers.append(f"contact:{phase_bridge_contact_reason}")
                        if phase_bridge_seed_mode == "live_taskspace" and not phase_bridge_preload_validated:
                            phase_bridge_seed_blockers.append(f"preload:{phase_bridge_preload_reason}")
                        if phase_bridge_seed_mode == "nominal_history_with_live_veto" and phase_bridge_live_veto:
                            phase_bridge_seed_blockers.append("live_geometry_untrusted")
                        phase_bridge_seed_ok = bool(phase_bridge_readiness_ok)
                        reset_count = 0
                        reset_phase_action_history = not (
                            phase_bridge_seed_ok
                            and phase_bridge_action_seed_source == "nominal_reentry_action_window"
                            and phase_bridge_target_rows is not None
                        )
                        if reset_phase_action_history:
                            reset_count = _reset_action_sequence_history(env)
                            if policy_env is not None:
                                reset_count += _reset_action_sequence_history(policy_env)
                        phase_bridge_policy_obs_history = recovery_policy_obs_history
                        if (
                            phase_bridge_seed_ok
                            and str(getattr(args, "phase_reanchor_bridge_seed_obs_source", "recovery"))
                            == "nominal_q_window"
                            and phase_bridge_nominal_q_rows is not None
                            and phase_bridge_nominal_q_live_dim is not None
                        ):
                            try:
                                obs_q_rows = np.asarray(phase_bridge_nominal_q_rows, dtype=np.float64).copy()
                                live_q_for_obs = np.asarray(q_full, dtype=np.float64).reshape(-1)
                                obs_dim = min(
                                    int(phase_bridge_nominal_q_live_dim),
                                    obs_q_rows.shape[-1],
                                    live_q_for_obs.size,
                                )
                                if (
                                    not phase_bridge_nominal_q_track_base
                                    and phase_bridge_nominal_q_base_state_idx.size > 0
                                ):
                                    base_idx = phase_bridge_nominal_q_base_state_idx[
                                        (phase_bridge_nominal_q_base_state_idx >= 0)
                                        & (phase_bridge_nominal_q_base_state_idx < obs_dim)
                                    ]
                                    if base_idx.size > 0:
                                        obs_q_rows[:, base_idx] = live_q_for_obs[base_idx][None, :]
                                obs_indices = np.arange(obs_dim, dtype=np.int64)
                                obs_qvel_window = _ablation_window_qvel(
                                    obs_q_rows[:, :obs_dim],
                                    _ablation_env_dt(env),
                                )
                                (
                                    nominal_obs_history,
                                    phase_bridge_obs_seed_restore_count,
                                ) = _collect_policy_obs_window_preserving_state(
                                    env=env,
                                    policy_env=policy_env,
                                    safety_runtime_env=safety_env,
                                    q_window=obs_q_rows[:, :obs_dim],
                                    indices=obs_indices,
                                    zero_velocity=False,
                                    qvel_window=obs_qvel_window,
                                )
                                if nominal_obs_history:
                                    phase_bridge_policy_obs_history = nominal_obs_history
                                    phase_bridge_obs_seed_source = "nominal_q_window"
                                    phase_bridge_obs_seed_window_count = int(len(nominal_obs_history))
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "Could not collect nominal-q policy obs bridge window: %s",
                                    exc,
                                )
                        phase_bridge_seed_source_count = int(len(phase_bridge_policy_obs_history))
                        if phase_bridge_seed_ok and phase_bridge_policy_obs_history:
                            (
                                policy_obs,
                                phase_bridge_visual_seed_count,
                                phase_bridge_frame_stack_seed_count,
                            ) = _seed_policy_visual_history_after_recovery(
                                policy_obs,
                                phase_bridge_policy_obs_history,
                                env=env,
                                policy_env=policy_env,
                            )
                            seed_seen: set[int] = set()
                            for candidate in (env, policy_env):
                                if candidate is None or id(candidate) in seed_seen:
                                    continue
                                seed_seen.add(id(candidate))
                                try:
                                    phase_bridge_action_seed_count += _seed_action_sequence_history_with_nominal_actions(
                                        candidate,
                                        phase_bridge_target_action,
                                        history_window_len=(
                                            phase_bridge_action_window_len
                                            if phase_bridge_action_window_len is not None
                                            else phase_bridge_seed_source_count
                                        ),
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    logger.debug("Could not seed phase-reanchor action history: %s", exc)
                            if phase_bridge_action_seed_count <= 0 and not reset_phase_action_history:
                                reset_count = _reset_action_sequence_history(env)
                                if policy_env is not None:
                                    reset_count += _reset_action_sequence_history(policy_env)
                                phase_bridge_action_seed_source = "nominal_reentry_action_window_seed_failed"
                        phase_bridge_temporal_stats = {
                            "action_bridge_temporal_history_slot_count": None,
                            "action_bridge_temporal_history_vs_resume_l2": None,
                        }
                        if phase_bridge_target_rows is not None and phase_bridge_target_rows.shape[0] > 0:
                            try:
                                phase_bridge_temporal_stats = _temporal_action_history_stats(
                                    env,
                                    np.asarray(phase_bridge_target_rows[0], dtype=np.float32).reshape(-1),
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "Could not compute phase-reanchor bridge temporal action stats: %s",
                                    exc,
                                )
                        phase_bridge_policy_step_before = int(policy_step)
                        phase_bridge_policy_step_after = 0
                        phase_bridge_policy_step_source = "reset_zero"
                        if (
                            phase_bridge_seed_ok
                            and str(getattr(args, "phase_reanchor_bridge_policy_step_source", "reset_zero"))
                            == "nominal_window"
                            and isinstance(phase_reanchor_action_state, dict)
                        ):
                            candidate_policy_step = None
                            target_steps = phase_reanchor_action_state.get(
                                "nominal_reentry_window_steps"
                            )
                            try:
                                if target_steps is not None:
                                    target_steps_list = list(target_steps)
                                    target_index = _safe_info_get(
                                        safety_info,
                                        "phase_reanchor_arm_servo_target_window_index",
                                    )
                                    if target_index is not None:
                                        target_index = int(target_index)
                                        if 0 <= target_index < len(target_steps_list):
                                            candidate_policy_step = int(target_steps_list[target_index])
                                    if candidate_policy_step is None and target_steps_list:
                                        candidate_policy_step = int(target_steps_list[0])
                            except Exception:  # noqa: BLE001
                                candidate_policy_step = None
                            if candidate_policy_step is None:
                                try:
                                    candidate_policy_step = int(
                                        phase_reanchor_action_state.get(
                                            "nominal_reentry_start_step"
                                        )
                                    )
                                except Exception:  # noqa: BLE001
                                    candidate_policy_step = None
                            if candidate_policy_step is not None and candidate_policy_step >= 0:
                                phase_bridge_policy_step_after = int(candidate_policy_step)
                                phase_bridge_policy_step_source = "nominal_window"
                        if hasattr(safechunk, "reset"):
                            safechunk.reset()
                        if phase_bridge_seed_ok:
                            recovery_policy_obs_history.clear()
                        policy_step = int(phase_bridge_policy_step_after)
                        phase_bridge_post_seed_act_vs_nominal_l2 = None
                        phase_bridge_post_seed_act_vs_nominal_cosine = None
                        phase_bridge_post_seed_action_agreement_ok = None
                        phase_bridge_post_seed_action_seed_count = 0
                        phase_bridge_temporal_stats_source = phase_bridge_action_seed_source
                        phase_bridge_requires_post_seed_action_agreement = bool(
                            getattr(args, "phase_reanchor_bridge_requires_post_seed_action_agreement", False)
                        )
                        post_seed_rows = None
                        if phase_bridge_seed_ok:
                            try:
                                post_seed_env_action = policy_action(ws, policy_obs, step=policy_step)
                                post_seed_rows = _ablation_action_rows(post_seed_env_action)
                                if (
                                    post_seed_rows is not None
                                    and post_seed_rows.shape[0] > 0
                                    and phase_bridge_target_rows is not None
                                    and phase_bridge_target_rows.shape[0] > 0
                                ):
                                    a = np.asarray(post_seed_rows[0], dtype=np.float32).reshape(-1)
                                    b = np.asarray(phase_bridge_target_rows[0], dtype=np.float32).reshape(-1)
                                    dim = min(a.size, b.size)
                                    if dim > 0:
                                        phase_bridge_post_seed_act_vs_nominal_l2 = float(
                                            np.linalg.norm(a[:dim] - b[:dim])
                                        )
                                        denom = float(np.linalg.norm(a[:dim]) * np.linalg.norm(b[:dim]) + 1e-8)
                                        phase_bridge_post_seed_act_vs_nominal_cosine = float(
                                            np.dot(a[:dim], b[:dim]) / denom
                                        )
                                        action_l2_limit = float(
                                            getattr(args, "phase_reanchor_bridge_action_agreement_l2", 1.0)
                                        )
                                        action_cosine_limit = float(
                                            getattr(args, "phase_reanchor_bridge_action_agreement_cosine", 0.8)
                                        )
                                        action_agreement_mode = str(
                                            getattr(args, "phase_reanchor_bridge_action_agreement_mode", "and")
                                        )
                                        if action_agreement_mode == "or":
                                            phase_bridge_post_seed_action_agreement_ok = bool(
                                                phase_bridge_post_seed_act_vs_nominal_l2 <= action_l2_limit
                                                or phase_bridge_post_seed_act_vs_nominal_cosine >= action_cosine_limit
                                            )
                                        else:
                                            phase_bridge_post_seed_action_agreement_ok = bool(
                                                phase_bridge_post_seed_act_vs_nominal_l2 <= action_l2_limit
                                                and phase_bridge_post_seed_act_vs_nominal_cosine >= action_cosine_limit
                                            )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "Could not compute phase-reanchor post-seed ACT action diagnostic: %s",
                                    exc,
                                )
                        if (
                            phase_bridge_seed_ok
                            and phase_bridge_requires_post_seed_action_agreement
                            and phase_bridge_post_seed_action_agreement_ok is not True
                        ):
                            phase_bridge_seed_blockers.append("post_seed_action_agreement")
                            phase_bridge_seed_ok = False
                            phase_bridge_readiness_ok = False
                            try:
                                reset_count += _reset_action_sequence_history(env)
                                if policy_env is not None:
                                    reset_count += _reset_action_sequence_history(policy_env)
                                policy_obs, reset_visual_count = _reset_policy_visual_history_after_recovery(
                                    env,
                                    policy_env,
                                    policy_obs,
                                )
                                reset_count += int(reset_visual_count)
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "Could not reset bridge history after post-seed ACT mismatch: %s",
                                    exc,
                                )
                        if (
                            phase_bridge_seed_ok
                            and bool(
                                getattr(
                                    args,
                                    "phase_reanchor_bridge_reseed_action_history_with_post_seed_act",
                                    False,
                                )
                            )
                            and post_seed_rows is not None
                            and post_seed_rows.shape[0] > 0
                        ):
                            try:
                                seed_seen: set[int] = set()
                                for candidate in (env, policy_env):
                                    if candidate is None or id(candidate) in seed_seen:
                                        continue
                                    seed_seen.add(id(candidate))
                                    phase_bridge_post_seed_action_seed_count += (
                                        _seed_action_sequence_history_with_nominal_actions(
                                            candidate,
                                            post_seed_rows,
                                            history_window_len=(
                                                phase_bridge_action_window_len
                                                if phase_bridge_action_window_len is not None
                                                else post_seed_rows.shape[0]
                                            ),
                                        )
                                    )
                                if phase_bridge_post_seed_action_seed_count > 0:
                                    phase_bridge_action_seed_count = int(phase_bridge_post_seed_action_seed_count)
                                    phase_bridge_action_seed_source = "post_seed_act_policy_action"
                                    phase_bridge_temporal_stats_source = phase_bridge_action_seed_source
                                    phase_bridge_temporal_stats = _temporal_action_history_stats(
                                        env,
                                        np.asarray(post_seed_rows[0], dtype=np.float32).reshape(-1),
                                    )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "Could not reseed phase-reanchor bridge action history with post-seed ACT: %s",
                                    exc,
                                )
                        policy_hold_active = bool(phase_bridge_seed_ok)
                        safety_info = dict(safety_info)
                        safety_info.update(
                            {
                                "phase_reanchor_bridge_history_seed": bool(
                                    phase_bridge_seed_ok
                                    and (phase_bridge_visual_seed_count > 0 or phase_bridge_action_seed_count > 0)
                                ),
                                "phase_reanchor_bridge_seed_mode": phase_bridge_seed_mode,
                                "phase_reanchor_bridge_seed_reason": (
                                    f"{phase_bridge_seed_mode}_ok"
                                    if phase_bridge_seed_ok
                                    else "blocked:" + ",".join(phase_bridge_seed_blockers)
                                ),
                                "phase_reanchor_bridge_seed_blockers": phase_bridge_seed_blockers,
                                "phase_reanchor_bridge_nominal_history_ok": phase_bridge_nominal_history_ok,
                                "phase_reanchor_bridge_nominal_action_window_ok": phase_bridge_nominal_action_window_ok,
                                "phase_reanchor_bridge_live_taskspace_ok": phase_bridge_live_taskspace_ok,
                                "phase_reanchor_bridge_policy_step_before": phase_bridge_policy_step_before,
                                "phase_reanchor_bridge_policy_step_after": phase_bridge_policy_step_after,
                                "phase_reanchor_bridge_policy_step_source": phase_bridge_policy_step_source,
                                "phase_reanchor_bridge_post_seed_act_vs_nominal_l2": phase_bridge_post_seed_act_vs_nominal_l2,
                                "phase_reanchor_bridge_post_seed_act_vs_nominal_cosine": phase_bridge_post_seed_act_vs_nominal_cosine,
                                "phase_reanchor_bridge_post_seed_action_seed_count": int(phase_bridge_post_seed_action_seed_count),
                                "phase_reanchor_bridge_post_seed_action_agreement_ok": phase_bridge_post_seed_action_agreement_ok,
                                "phase_reanchor_bridge_requires_post_seed_action_agreement": phase_bridge_requires_post_seed_action_agreement,
                                "phase_reanchor_bridge_temporal_stats_source": phase_bridge_temporal_stats_source,
                                "action_bridge_temporal_history_slot_count": phase_bridge_temporal_stats.get(
                                    "action_bridge_temporal_history_slot_count"
                                ),
                                "action_bridge_temporal_history_vs_resume_l2": phase_bridge_temporal_stats.get(
                                    "action_bridge_temporal_history_vs_resume_l2"
                                ),
                                "phase_reanchor_bridge_visual_seed_count": int(phase_bridge_visual_seed_count),
                                "phase_reanchor_bridge_visual_seed_source_count": int(phase_bridge_seed_source_count),
                                "phase_reanchor_bridge_frame_stack_seed_count": int(phase_bridge_frame_stack_seed_count),
                                "phase_reanchor_bridge_obs_seed_source": phase_bridge_obs_seed_source,
                                "phase_reanchor_bridge_obs_seed_window_count": int(phase_bridge_obs_seed_window_count),
                                "phase_reanchor_bridge_obs_seed_restore_count": int(phase_bridge_obs_seed_restore_count),
                                "phase_reanchor_bridge_action_seed_count": int(phase_bridge_action_seed_count),
                                "phase_reanchor_bridge_action_seed_source": phase_bridge_action_seed_source,
                                "phase_reanchor_bridge_action_window_len": phase_bridge_action_window_len,
                                "phase_reanchor_bridge_act_vs_nominal_l2": phase_bridge_act_vs_nominal_l2,
                                "phase_reanchor_bridge_act_vs_nominal_cosine": phase_bridge_act_vs_nominal_cosine,
                                "phase_reanchor_bridge_action_agreement_ok": phase_bridge_action_agreement_ok,
                                "phase_reanchor_bridge_action_base_adapted": phase_bridge_action_base_adapted,
                                "phase_reanchor_bridge_action_base_adapted_dims": phase_bridge_action_base_adapted_dims,
                                "phase_reanchor_bridge_contact_ready": phase_bridge_contact_ready,
                                "phase_reanchor_bridge_contact_reason": phase_bridge_contact_reason,
                                "phase_reanchor_bridge_contact_handle_dist": phase_bridge_contact_handle_dist,
                                "phase_reanchor_bridge_contact_handle_limit": phase_bridge_contact_handle_limit,
                                "phase_reanchor_bridge_preload_validated": phase_bridge_preload_validated,
                                "phase_reanchor_bridge_preload_reason": phase_bridge_preload_reason,
                                "phase_reanchor_bridge_preload_steps": phase_bridge_preload_steps,
                                "phase_reanchor_bridge_preload_progress_delta": phase_bridge_preload_progress_delta,
                                "phase_reanchor_bridge_preload_handle_dist": phase_bridge_preload_handle_dist,
                                "phase_reanchor_bridge_preload_handle_limit": phase_bridge_preload_handle_limit,
                                "phase_reanchor_bridge_nominal_q_l2": phase_bridge_nominal_q_l2,
                                "phase_reanchor_bridge_nominal_q_window_l2_mean": phase_bridge_nominal_q_window_l2_mean,
                                "phase_reanchor_bridge_nominal_q_window_l2_max": phase_bridge_nominal_q_window_l2_max,
                                "phase_reanchor_bridge_nominal_q_window_len": phase_bridge_nominal_q_window_len,
                                "phase_reanchor_bridge_nominal_q_adapted_l2": phase_bridge_nominal_q_adapted_l2,
                                "phase_reanchor_bridge_nominal_q_adapted_window_l2_mean": phase_bridge_nominal_q_adapted_window_l2_mean,
                                "phase_reanchor_bridge_nominal_q_adapted_window_l2_max": phase_bridge_nominal_q_adapted_window_l2_max,
                                "phase_reanchor_bridge_nominal_q_adapted_dims": phase_bridge_nominal_q_adapted_dims,
                                "phase_reanchor_bridge_nominal_q_base_l2": phase_bridge_nominal_q_base_l2,
                                "phase_reanchor_bridge_nominal_q_arm_l2": phase_bridge_nominal_q_arm_l2,
                                "phase_reanchor_bridge_nominal_q_track_base": phase_bridge_nominal_q_track_base,
                                "phase_reanchor_bridge_nominal_q_ok": phase_bridge_nominal_q_ok,
                                "phase_reanchor_bridge_readiness_ok": phase_bridge_readiness_ok,
                                "recovery_visual_history_seed": bool(phase_bridge_visual_seed_count > 0),
                                "recovery_visual_history_seed_count": int(phase_bridge_visual_seed_count),
                                "recovery_visual_history_seed_source_count": int(phase_bridge_seed_source_count),
                                "recovery_policy_obs_history_seed": bool(phase_bridge_visual_seed_count > 0),
                                "recovery_policy_obs_history_seed_count": int(phase_bridge_visual_seed_count),
                                "recovery_policy_obs_history_seed_source_count": int(phase_bridge_seed_source_count),
                            }
                        )
                        if episode == 0 or args.debug:
                            print(
                                "phase_reanchor: finish "
                                f"episode={episode} step={step} reset_wrappers={reset_count} "
                                f"bridge_seed={phase_bridge_seed_ok} "
                                f"visual_seed={phase_bridge_visual_seed_count} "
                                f"action_seed={phase_bridge_action_seed_count} "
                                f"action_source={phase_bridge_action_seed_source} "
                                f"cooldown={phase_reanchor_cooldown_left}"
                            )
                    if not policy_hold_active:
                        policy_step += 1

                bridge_history_mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
                bridge_history_source = _safe_info_get(safety_info, "deformation_source")
                bridge_history_phase = _safe_info_get(safety_info, "recovery_phase")
                bridge_history_committed_mode = _safe_info_get(safety_info, "committed_chunk_mode")
                bridge_record_last_action = bool(
                    post_recovery_act_bridge_active
                    or _is_safety_intervention_mode(safety_info)
                    or _is_brake_or_fallback_execution(safety_info)
                    or bool(_safe_info_get(safety_info, "request_action_history_reset_after_recovery"))
                    or bridge_history_mode in {
                        "recover",
                        "committed_explicit_recovery",
                        "horizon_deform",
                        "horizon_brake",
                        "phase_reanchor",
                    }
                    or bridge_history_source in {
                        "committed_explicit_recovery",
                        "explicit_recover_deform",
                        "chunk_deform",
                        "horizon_deform",
                        "horizon_brake",
                        "phase_reanchor",
                    }
                    or bridge_history_phase in {
                        "recover",
                        "horizon_deform",
                        "resume_act",
                    }
                    or bridge_history_committed_mode in {
                        "recover",
                        "horizon_deform",
                    }
                )
                if bridge_record_last_action:
                    last_recovery_first_action_for_bridge = np.asarray(
                        safe_first_action,
                        dtype=np.float32,
                    ).reshape(-1).copy()
                    last_recovery_action_step_for_bridge = int(step)
                if bool(_safe_info_get(safety_info, "mpc_handoff_accepted")):
                    handoff_release_first_action_for_bridge = np.asarray(
                        safe_first_action,
                        dtype=np.float32,
                    ).reshape(-1).copy()
                    handoff_release_action_step_for_bridge = int(step)
                if post_recovery_act_bridge_active:
                    post_recovery_act_bridge_steps_left = max(
                        0,
                        int(post_recovery_act_bridge_steps_left) - 1,
                    )

                last_safety_intervention_active = bool(safety_intervention_active)

                if safety_env is not None:
                    # Mirror the robot/task state after the action, but keep the human
                    # pose fixed to the one the filter just evaluated. The next human
                    # update happens at the start of the next control step, before
                    # monitor/filter, so contacts are measured against a visible pose.
                    _sync_named_mujoco_state(env, safety_env)
                    _sync_animated_legs(safety_env, is_moving=True)

                phase_for_resume = _safe_info_get(blocker_info, "human_phase")
                dist_for_resume = _safe_info_get(blocker_info, "min_robot_human_distance")
                should_reset_after_human_exit = (
                    args.reset_action_history_after_human_exit
                    and not action_history_reset_after_exit
                )
                should_restart_after_blocker_pause = (
                    args.pause_and_restart_on_human_blocker
                    and not pause_restart_reset_after_exit
                )
                if (
                    (should_reset_after_human_exit or should_restart_after_blocker_pause)
                    and phase_for_resume == "done"
                ):
                    if (
                        dist_for_resume is None
                        or float(dist_for_resume) >= args.resume_clearance_threshold
                    ):
                        human_done_clear_steps += 1
                    else:
                        human_done_clear_steps = 0
                    if human_done_clear_steps >= args.resume_clear_steps:
                        reset_count = _reset_action_sequence_history(env)
                        if policy_env is not None:
                            reset_count += _reset_action_sequence_history(policy_env)
                        visual_reset_count = 0
                        if bool(getattr(args, "reset_visual_history_after_human_exit", False)):
                            policy_obs, visual_reset_count = _reset_policy_visual_history_after_recovery(
                                env,
                                policy_env,
                                policy_obs,
                            )
                        if hasattr(safechunk, "reset"):
                            safechunk.reset()
                        policy_step = 0
                        human_exit_resume_started = True
                        human_exit_resume_start_step = int(step)
                        human_exit_resume_action_reset_count = int(reset_count)
                        human_exit_resume_visual_reset_count = int(visual_reset_count)
                        if should_reset_after_human_exit:
                            action_history_reset_after_exit = True
                            visual_history_reset_after_exit = bool(visual_reset_count > 0)
                        if should_restart_after_blocker_pause:
                            pause_restart_reset_after_exit = True
                        if episode == 0 or args.debug:
                            reason = (
                                "pause_and_restart_after_human_blocker"
                                if should_restart_after_blocker_pause
                                else "reset_action_history_after_human_exit"
                            )
                            print(
                                f"resume_supervisor: {reason} "
                                f"step={step} reset_wrappers={reset_count}"
                            )
                elif phase_for_resume != "done":
                    human_done_clear_steps = 0

                robot_violation_color_overrides = _apply_robot_part_color_overrides(
                    safety_runtime_env,
                    red_parts=current_h_violation_parts,
                    blue_parts=(horizon_violation_parts - current_h_violation_parts),
                )
                try:
                    video_recorder.record(safety_runtime_env)
                    if args.save_frame_images and step % args.frame_image_every == 0:
                        frame = _render_single_env_if_vector(safety_runtime_env)
                        if frame is not None:
                            frame_path = frame_image_dir / (
                                f"{args.condition}_episode_{episode:03d}_step_{step:06d}.png"
                            )
                            imageio.imwrite(str(frame_path), np.asarray(frame))
                            saved_frame_image_paths.append(str(frame_path))
                finally:
                    _restore_robot_part_color_overrides(safety_runtime_env, robot_violation_color_overrides)
                step_wall_t = time.perf_counter()
                elapsed_wall_time_s = step_wall_t - episode_wall_t0
                step_wall_time_s = step_wall_t - last_step_wall_t
                last_step_wall_t = step_wall_t

                nominal_arm = first_action[arm_idx]
                safe_arm = safe_first_action[arm_idx]

                arm_delta = float(np.linalg.norm(safe_arm - nominal_arm))
                base_delta = float(
                    np.linalg.norm(
                        safe_first_action[valid_base_idx] - first_action[valid_base_idx]
                    )
                ) if valid_base_idx.size else 0.0
                non_arm_delta = float(
                    np.linalg.norm(
                        safe_first_action[non_arm_idx] - first_action[non_arm_idx]
                    )
                )
                full_delta = float(np.linalg.norm(safe_first_action - first_action))

                nominal_chunk_for_metrics, _ = _as_chunk(env_action)
                safe_chunk_for_metrics, _ = _as_chunk(safe_env_action)
                chunk_arm_delta = float(
                    np.linalg.norm(
                        safe_chunk_for_metrics[:, arm_idx]
                        - nominal_chunk_for_metrics[:, arm_idx]
                    )
                )
                chunk_base_delta = float(
                    np.linalg.norm(
                        safe_chunk_for_metrics[:, valid_base_idx]
                        - nominal_chunk_for_metrics[:, valid_base_idx]
                    )
                ) if valid_base_idx.size else 0.0
                chunk_non_arm_delta = float(
                    np.linalg.norm(
                        safe_chunk_for_metrics[:, non_arm_idx]
                        - nominal_chunk_for_metrics[:, non_arm_idx]
                    )
                )
                chunk_full_delta = float(
                    np.linalg.norm(safe_chunk_for_metrics - nominal_chunk_for_metrics)
                )
                chunk_advantage_metrics = _chunk_filter_advantage_metrics(
                    nominal_chunk_for_metrics,
                    safe_chunk_for_metrics,
                    arm_idx,
                    args.intervention_eps,
                )
                path_deviation_metrics = (
                    _path_deviation_metrics(
                        safechunk,
                        _chunk_obs_with_q(obs, q_full),
                        nominal_chunk_for_metrics,
                        safe_chunk_for_metrics,
                    )
                    if chunk_filter_mode
                    else {
                        "path_mean_deviation": None,
                        "path_max_deviation": None,
                        "path_final_deviation": None,
                    }
                )
                if pacs_background_chunk_for_metrics is not None:
                    pacs_background_chunk = np.asarray(
                        pacs_background_chunk_for_metrics,
                        dtype=np.float32,
                    )
                    pacs_background_first_action = pacs_background_chunk[0]
                    pacs_background_arm_delta = float(
                        np.linalg.norm(pacs_background_first_action[arm_idx] - nominal_arm)
                    )
                    pacs_background_chunk_arm_delta = float(
                        np.linalg.norm(
                            pacs_background_chunk[:, arm_idx]
                            - nominal_chunk_for_metrics[:, arm_idx]
                        )
                    )
                    pacs_background_advantage_metrics = _chunk_filter_advantage_metrics(
                        nominal_chunk_for_metrics,
                        pacs_background_chunk,
                        arm_idx,
                        args.intervention_eps,
                    )
                    pacs_background_flags = _diagnostic_mode_flags(
                        pacs_background_safety_info or {},
                        arm_delta=pacs_background_arm_delta,
                        eps=args.intervention_eps,
                    )
                    safety_info = dict(safety_info)
                    safety_info.update(
                        {
                            "pacs_background_brake_step": pacs_background_flags["brake_step"],
                            "pacs_background_act_step": pacs_background_flags["act_step"],
                            "pacs_background_arm_delta": pacs_background_arm_delta,
                            "pacs_background_chunk_arm_delta": pacs_background_chunk_arm_delta,
                            "pacs_background_chunk_modified_fraction": pacs_background_advantage_metrics[
                                "chunk_modified_fraction"
                            ],
                            "pacs_background_retiming_arm_delta": pacs_background_arm_delta,
                            "pacs_background_retiming_chunk_arm_delta": pacs_background_chunk_arm_delta,
                            "pacs_background_retiming_changed_fraction": pacs_background_advantage_metrics[
                                "chunk_modified_fraction"
                            ],
                        }
                    )
                if args.diagnostics_enabled:
                    diagnostic_flags = _diagnostic_mode_flags(
                        safety_info,
                        arm_delta=arm_delta,
                        eps=args.intervention_eps,
                    )
                    current_diagnostic_mode = diagnostic_flags["diagnostic_step_mode"]
                    mode_transition = (
                        f"{last_diagnostic_step_mode}->{current_diagnostic_mode}"
                        if last_diagnostic_step_mode is not None
                        and current_diagnostic_mode != last_diagnostic_step_mode
                        else None
                    )
                    last_diagnostic_step_mode = current_diagnostic_mode
                else:
                    diagnostic_flags = {
                        "diagnostic_step_mode": None,
                        "act_step": None,
                        "deform_step": None,
                        "recover_step": None,
                        "brake_step": None,
                        "fallback_step": None,
                        "optimized_attempt_step": None,
                        "optimized_accepted_step": None,
                    }
                    mode_transition = None
                (
                    horizon_risk_gap,
                    horizon_risk_gap_active,
                    horizon_clearance_drop,
                ) = _horizon_risk_gap(
                    min_h,
                    _safe_info_get(safety_info, "min_clearance"),
                )

                success = extract_success(info, float(reward), bool(terminated))
                contact_pairs = robot_human_contact_pairs(safety_runtime_env)
                contact_count = None if contact_pairs is None else len(contact_pairs)
                contact_now = bool(contact_count is not None and contact_count > 0)
                if nominal_rollout_diagnostic_context is not None:
                    record = dict(nominal_rollout_diagnostic_context)
                    record.update(
                        {
                            "condition": args.condition,
                            "safety_mode": _safe_info_get(safety_info, "safety_mode"),
                            "mode": _safe_info_get(safety_info, "mode"),
                            "deform_mode": _safe_info_get(safety_info, "deform_mode"),
                            "deformation_source": _safe_info_get(safety_info, "deformation_source"),
                            "fallback_reason": _safe_info_get(safety_info, "fallback_reason"),
                            "optimized_fallback": _safe_info_get(safety_info, "optimized_fallback"),
                            "act_step": diagnostic_flags.get("act_step"),
                            "deform_step": diagnostic_flags.get("deform_step"),
                            "recover_step": diagnostic_flags.get("recover_step"),
                            "brake_step": diagnostic_flags.get("brake_step"),
                            "fallback_step": diagnostic_flags.get("fallback_step"),
                            "task_progress_before": task_state_before.get("task_progress"),
                            "task_progress_after": task_state_after.get("task_progress"),
                            "drawer_open_distance_after": task_state_after.get("drawer_open_distance"),
                            "nominal_horizon_min_clearance": _safe_info_get(safety_info, "min_clearance"),
                            "nominal_horizon_first_violation": _safe_info_get(safety_info, "first_violation"),
                            "nominal_horizon_unsafe_count": _safe_info_get(safety_info, "unsafe_count"),
                            "rollout_residual_correction_applied": _safe_info_get(safety_info, "rollout_residual_correction_applied"),
                            "rollout_mismatch_prediction_untrusted": _safe_info_get(safety_info, "rollout_mismatch_prediction_untrusted"),
                            "rollout_mismatch_live_clear_to_continue": _safe_info_get(safety_info, "rollout_mismatch_live_clear_to_continue"),
                            "rollout_mismatch_pass_through": _safe_info_get(safety_info, "rollout_mismatch_pass_through"),
                            "rollout_mismatch_escape_reason": _safe_info_get(safety_info, "rollout_mismatch_escape_reason"),
                            "horizon_unsafe_ignored_due_to_rollout_mismatch": _safe_info_get(safety_info, "horizon_unsafe_ignored_due_to_rollout_mismatch"),
                            "hard_executable_prefix_safe": _safe_info_get(safety_info, "hard_executable_prefix_safe"),
                            "full_horizon_soft_pass_through": _safe_info_get(safety_info, "full_horizon_soft_pass_through"),
                            "horizon_unsafe_ignored_due_to_executable_prefix_safe": _safe_info_get(safety_info, "horizon_unsafe_ignored_due_to_executable_prefix_safe"),
                            "committed_soft_handoff_release_to_main_filter": _safe_info_get(safety_info, "committed_soft_handoff_release_to_main_filter"),
                            "committed_soft_handoff_release_reason": _safe_info_get(safety_info, "committed_soft_handoff_release_reason"),
                            "live_min_clearance_before": live_min_clearance,
                            "live_h_violation_before": h_violation,
                            "contact_count_after": contact_count,
                            "contact_pairs_after": contact_pairs,
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    if nominal_rollout_post_step_state is not None:
                        record.update(nominal_rollout_post_step_state)
                    actual_q_after = record.get("q_after")
                    record.update(
                        _rollout_error_payload(
                            "nominal",
                            record.get("nominal_pred_q_next"),
                            actual_q_after,
                            state_idx,
                        )
                    )
                    record.update(
                        _rollout_error_payload(
                            "safe",
                            record.get("safe_pred_q_next"),
                            actual_q_after,
                            state_idx,
                        )
                    )
                    if rollout_feedback is not None:
                        record.update(rollout_feedback)
                    nominal_first = record.get("nominal_first_action")
                    safe_first = record.get("safe_first_action")
                    if nominal_first is not None and safe_first is not None:
                        try:
                            action_delta = np.asarray(safe_first, dtype=np.float32) - np.asarray(nominal_first, dtype=np.float32)
                            record["safe_vs_nominal_first_action_l2"] = float(np.linalg.norm(action_delta))
                            record["safe_vs_nominal_first_action_max_abs"] = float(np.max(np.abs(action_delta)))
                        except Exception:  # noqa: BLE001
                            pass
                    episode_nominal_rollout_diagnostic_records.append(record)
                    all_nominal_rollout_diagnostic_records.append(record)
                safety_info["contact_during_hold"] = bool(
                    contact_now
                    and diagnostic_flags.get("brake_step")
                    and _safe_info_get(safety_info, "brake_hold_current")
                )
                safety_info["contact_during_brake"] = bool(
                    contact_now and diagnostic_flags.get("brake_step")
                )
                safety_info["contact_during_deform"] = bool(
                    contact_now and diagnostic_flags.get("deform_step")
                )
                safety_info["contact_during_recover"] = bool(
                    contact_now and diagnostic_flags.get("recover_step")
                )
                safety_info.update(_act_resumable_score_terms(safety_info, args))
                safety_info.update(
                    {
                        "human_done_clear_steps": int(human_done_clear_steps),
                        "action_history_reset_after_exit": bool(action_history_reset_after_exit),
                        "visual_history_reset_after_exit": bool(visual_history_reset_after_exit),
                        "human_exit_resume_started": bool(human_exit_resume_started),
                        "human_exit_resume_start_step": human_exit_resume_start_step,
                        "human_exit_resume_action_reset_count": int(human_exit_resume_action_reset_count),
                        "human_exit_resume_visual_reset_count": int(human_exit_resume_visual_reset_count),
                        "human_exit_resume_act_resumable_ok": (
                            _safe_info_get(safety_info, "act_resumable_ok")
                            if human_exit_resume_started
                            else None
                        ),
                    }
                )

                unmodelled_contact_reason = _unmodelled_robot_contact_reason(contact_pairs)
                info_dict = info if isinstance(info, dict) else {}
                if blocker_info:
                    info_dict = {**info_dict, **blocker_info}

                step_metrics = StepMetrics(
                    condition=args.condition,
                    episode=episode,
                    step=step,

                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    success=success,

                    human_phase=_optional_str(_safe_info_get(info_dict, "human_phase")),
                    ee_to_handle_dist=_optional_float(_safe_info_get(info_dict, "ee_to_handle_dist")),
                    human_blocker_triggered=_optional_bool(_safe_info_get(info_dict, "human_blocker_triggered")),
                    human_time_in_phase=_optional_float(_safe_info_get(info_dict, "human_time_in_phase")),
                    min_robot_human_distance=_optional_float(
                        _safe_info_get(info_dict, "min_robot_human_distance")
                        if _safe_info_get(info_dict, "min_robot_human_distance") is not None
                        else _safe_info_get(info_dict, "human_min_robot_distance")
                    ),
                    goal_region_human_distance=_optional_float(_safe_info_get(info_dict, "goal_region_human_distance")),
                    goal_region_blocked=_optional_bool(_safe_info_get(info_dict, "goal_region_blocked")),
                    drawer_open_distance=_optional_float(
                        task_state_after.get("drawer_open_distance")
                        if task_state_after.get("drawer_open_distance") is not None
                        else _safe_info_get(info_dict, "drawer_open_distance")
                    ),
                    drawer_open_fraction=_optional_float(task_state_after.get("drawer_open_fraction")),
                    drawer_joint_position=_optional_float(task_state_after.get("drawer_joint_position")),
                    task_progress_before=_optional_float(task_state_before.get("task_progress")),
                    task_progress_after=_optional_float(task_state_after.get("task_progress")),
                    task_progress_delta=_optional_float(task_progress_delta),
                    ee_object_distance=_optional_float(task_state_after.get("ee_object_distance")),
                    object_state=task_state_after.get("object_state"),
                    interaction_context=_optional_str(_safe_info_get(safety_info, "interaction_context")),
                    resume_adapter=_optional_str(_safe_info_get(safety_info, "resume_adapter")),
                    resume_context_source=_optional_str(_safe_info_get(safety_info, "resume_context_source")),
                    resume_target_label=_optional_str(_safe_info_get(safety_info, "resume_target_label")),
                    resume_affordance_enabled=_optional_bool(_safe_info_get(safety_info, "resume_affordance_enabled")),
                    resume_affordance_available=_optional_bool(_safe_info_get(safety_info, "resume_affordance_available")),
                    resume_affordance_task_relevant=_optional_bool(_safe_info_get(safety_info, "resume_affordance_task_relevant")),
                    resume_affordance_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_score")),
                    resume_affordance_ok=_optional_bool(_safe_info_get(safety_info, "resume_affordance_ok")),
                    resume_affordance_min_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_min_score")),
                    resume_affordance_component_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_component_score")),
                    resume_affordance_min_component_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_min_component_score")),
                    resume_affordance_target_distance=_optional_float(_safe_info_get(safety_info, "resume_affordance_target_distance", _safe_info_get(safety_info, "resume_target_distance"))),
                    resume_affordance_target_distance_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_target_distance_score")),
                    resume_affordance_target_distance_good=_optional_float(_safe_info_get(safety_info, "resume_affordance_target_distance_good")),
                    resume_affordance_target_distance_scale=_optional_float(_safe_info_get(safety_info, "resume_affordance_target_distance_scale")),
                    resume_affordance_contact_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_contact_score")),
                    resume_affordance_contact_available=_optional_bool(_safe_info_get(safety_info, "resume_affordance_contact_available")),
                    resume_affordance_progress=_optional_float(_safe_info_get(safety_info, "resume_affordance_progress", _safe_info_get(safety_info, "resume_task_progress"))),
                    resume_affordance_progress_delta=_optional_float(_safe_info_get(safety_info, "resume_affordance_progress_delta", _safe_info_get(safety_info, "resume_task_progress_delta"))),
                    resume_affordance_progress_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_progress_score")),
                    resume_affordance_alignment_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_alignment_score")),
                    resume_affordance_continuity_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_continuity_score")),
                    resume_affordance_safety_score=_optional_float(_safe_info_get(safety_info, "resume_affordance_safety_score")),
                    resume_affordance_prefix_min_clearance=_optional_float(_safe_info_get(safety_info, "resume_affordance_prefix_min_clearance")),
                    resume_affordance_required_clearance=_optional_float(_safe_info_get(safety_info, "resume_affordance_required_clearance")),
                    act_resumable_score=_optional_float(_safe_info_get(safety_info, "act_resumable_score")),
                    act_resumable_nominal_score=_optional_float(_safe_info_get(safety_info, "act_resumable_nominal_score")),
                    act_resumable_live_score=_optional_float(_safe_info_get(safety_info, "act_resumable_live_score")),
                    act_resumable_nominal_ok=_optional_bool(_safe_info_get(safety_info, "act_resumable_nominal_ok")),
                    act_resumable_live_ok=_optional_bool(_safe_info_get(safety_info, "act_resumable_live_ok")),
                    act_resumable_ok=_optional_bool(_safe_info_get(safety_info, "act_resumable_ok")),
                    act_resumable_live_target_distance=_optional_float(_safe_info_get(safety_info, "act_resumable_live_target_distance")),
                    act_resumable_live_handle_distance=_optional_float(_safe_info_get(safety_info, "act_resumable_live_handle_distance")),
                    act_resumable_live_requires_handle_proximity=_optional_bool(_safe_info_get(safety_info, "act_resumable_live_requires_handle_proximity")),
                    act_resumable_live_handle_limit=_optional_float(_safe_info_get(safety_info, "act_resumable_live_handle_limit")),
                    act_resumable_geometry_untrusted=_optional_bool(_safe_info_get(safety_info, "act_resumable_geometry_untrusted")),
                    human_done_clear_steps=_optional_int(_safe_info_get(safety_info, "human_done_clear_steps")),
                    action_history_reset_after_exit=_optional_bool(_safe_info_get(safety_info, "action_history_reset_after_exit")),
                    visual_history_reset_after_exit=_optional_bool(_safe_info_get(safety_info, "visual_history_reset_after_exit")),
                    human_exit_resume_started=_optional_bool(_safe_info_get(safety_info, "human_exit_resume_started")),
                    human_exit_resume_start_step=_optional_int(_safe_info_get(safety_info, "human_exit_resume_start_step")),
                    human_exit_resume_action_reset_count=_optional_int(_safe_info_get(safety_info, "human_exit_resume_action_reset_count")),
                    human_exit_resume_visual_reset_count=_optional_int(_safe_info_get(safety_info, "human_exit_resume_visual_reset_count")),
                    human_exit_resume_act_resumable_ok=_optional_bool(_safe_info_get(safety_info, "human_exit_resume_act_resumable_ok")),
                    ablation_force_planned_recovery_q_enabled=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_enabled")),
                    ablation_force_planned_recovery_q_applied=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_applied")),
                    ablation_force_planned_recovery_q_skip_reason=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_skip_reason")),
                    ablation_force_planned_recovery_q_trigger=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_trigger")),
                    ablation_force_planned_recovery_q_source=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_source")),
                    ablation_force_planned_recovery_q_source_mode=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_source_mode")),
                    ablation_force_planned_recovery_q_window_mode=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_mode")),
                    ablation_force_planned_recovery_q_mode=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_mode")),
                    ablation_force_planned_recovery_q_indices=_safe_info_get(safety_info, "ablation_force_planned_recovery_q_indices"),
                    ablation_force_planned_recovery_q_dim=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_dim")),
                    ablation_force_planned_recovery_q_l2_from_pre=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_l2_from_pre")),
                    ablation_force_planned_recovery_q_max_abs_from_pre=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_max_abs_from_pre")),
                    ablation_force_planned_recovery_q_arm_l2_from_pre=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_arm_l2_from_pre")),
                    ablation_force_planned_recovery_q_base_l2_from_pre=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_base_l2_from_pre")),
                    ablation_force_planned_recovery_q_forced_env_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_forced_env_count")),
                    ablation_force_planned_recovery_q_reset_history_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_reset_history_count")),
                    ablation_force_planned_recovery_q_reset_filter=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_reset_filter")),
                    ablation_force_planned_recovery_q_sync_low_level_state_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sync_low_level_state_count")),
                    ablation_force_planned_recovery_q_zero_velocity=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_zero_velocity")),
                    ablation_force_planned_recovery_q_hold_current_step=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_hold_current_step")),
                    ablation_force_planned_recovery_q_hold_delta=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_hold_delta")),
                    ablation_force_planned_recovery_q_hold_indices=_safe_info_get(safety_info, "ablation_force_planned_recovery_q_hold_indices"),
                    ablation_force_planned_recovery_q_seed_window_enabled=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_seed_window_enabled")),
                    ablation_force_planned_recovery_q_window_source=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_source")),
                    ablation_force_planned_recovery_q_window_interpolated=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_interpolated")),
                    ablation_force_planned_recovery_q_window_len=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_len")),
                    ablation_force_planned_recovery_q_window_obs_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_obs_count")),
                    ablation_force_planned_recovery_q_window_seed_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_seed_count")),
                    ablation_force_planned_recovery_q_frame_stack_seed_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_frame_stack_seed_count")),
                    ablation_force_planned_recovery_q_window_seed_source_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_seed_source_count")),
                    ablation_force_planned_recovery_q_window_step_l2_mean=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_step_l2_mean")),
                    ablation_force_planned_recovery_q_window_step_l2_max=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_step_l2_max")),
                    ablation_force_planned_recovery_q_window_qvel_enabled=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_qvel_enabled")),
                    ablation_force_planned_recovery_q_window_qvel_dt=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_qvel_dt")),
                    ablation_force_planned_recovery_q_window_qvel_l2_mean=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_qvel_l2_mean")),
                    ablation_force_planned_recovery_q_window_qvel_l2_max=_optional_float(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_window_qvel_l2_max")),
                    ablation_force_planned_recovery_q_action_history_seed_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_action_history_seed_count")),
                    ablation_force_planned_recovery_q_action_history_seed_source=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_action_history_seed_source")),
                    ablation_force_planned_recovery_q_replay_sequence_enabled=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_replay_sequence_enabled")),
                    ablation_force_planned_recovery_q_replay_sequence_source=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_replay_sequence_source")),
                    ablation_force_planned_recovery_q_replay_sequence_len=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_replay_sequence_len")),
                    ablation_force_planned_recovery_q_sequence_active=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_active")),
                    ablation_force_planned_recovery_q_sequence_index=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_index")),
                    ablation_force_planned_recovery_q_sequence_len=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_len")),
                    ablation_force_planned_recovery_q_sequence_nominal_action_used=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_nominal_action_used")),
                    ablation_force_planned_recovery_q_sequence_action_source=_optional_str(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_action_source")),
                    ablation_force_planned_recovery_q_sequence_action_index=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_action_index")),
                    ablation_force_planned_recovery_q_sequence_forced_env_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_forced_env_count")),
                    ablation_force_planned_recovery_q_sequence_completed=_optional_bool(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_completed")),
                    ablation_force_planned_recovery_q_sequence_obs_count=_optional_int(_safe_info_get(safety_info, "ablation_force_planned_recovery_q_sequence_obs_count")),
                    ablation_pure_act_resume_enabled=_optional_bool(_safe_info_get(safety_info, "ablation_pure_act_resume_enabled")),
                    ablation_pure_act_resume_active=_optional_bool(_safe_info_get(safety_info, "ablation_pure_act_resume_active")),
                    ablation_pure_act_resume_step_index=_optional_int(_safe_info_get(safety_info, "ablation_pure_act_resume_step_index")),
                    ablation_pure_act_resume_steps_left=_optional_int(_safe_info_get(safety_info, "ablation_pure_act_resume_steps_left")),
                    ablation_pure_act_resume_scheduled_steps=_optional_int(_safe_info_get(safety_info, "ablation_pure_act_resume_scheduled_steps")),
                    ablation_pure_act_resume_total_steps=_optional_int(_safe_info_get(safety_info, "ablation_pure_act_resume_total_steps")),
                    act_resume_diag_active=_optional_bool(_safe_info_get(safety_info, "act_resume_diag_active")),
                    act_resume_diag_seed_step=_optional_int(_safe_info_get(safety_info, "act_resume_diag_seed_step")),
                    act_resume_diag_query_step=_optional_int(_safe_info_get(safety_info, "act_resume_diag_query_step")),
                    act_resume_diag_target_age_steps=_optional_int(_safe_info_get(safety_info, "act_resume_diag_target_age_steps")),
                    act_resume_diag_target_action_source=_optional_str(_safe_info_get(safety_info, "act_resume_diag_target_action_source")),
                    act_resume_diag_target_action_rows=_optional_int(_safe_info_get(safety_info, "act_resume_diag_target_action_rows")),
                    act_resume_diag_target_action_dim=_optional_int(_safe_info_get(safety_info, "act_resume_diag_target_action_dim")),
                    act_resume_diag_predicted_first_action_norm=_optional_float(_safe_info_get(safety_info, "act_resume_diag_predicted_first_action_norm")),
                    act_resume_diag_target_first_action_norm=_optional_float(_safe_info_get(safety_info, "act_resume_diag_target_first_action_norm")),
                    act_resume_diag_target_window_best_index=_optional_int(_safe_info_get(safety_info, "act_resume_diag_target_window_best_index")),
                    act_resume_diag_target_window_best_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_target_window_best_l2")),
                    act_resume_diag_target_window_l2_0=_optional_float(_safe_info_get(safety_info, "act_resume_diag_target_window_l2_0")),
                    act_resume_diag_target_window_l2_1=_optional_float(_safe_info_get(safety_info, "act_resume_diag_target_window_l2_1")),
                    act_resume_diag_target_window_l2_2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_target_window_l2_2")),
                    act_resume_diag_target_window_l2_3=_optional_float(_safe_info_get(safety_info, "act_resume_diag_target_window_l2_3")),
                    act_resume_diag_policy_obs_vs_seed_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_key_count")),
                    act_resume_diag_policy_obs_vs_seed_common_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_common_key_count")),
                    act_resume_diag_policy_obs_vs_seed_missing_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_missing_key_count")),
                    act_resume_diag_policy_obs_vs_seed_shape_mismatch_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_shape_mismatch_key_count")),
                    act_resume_diag_policy_obs_vs_seed_numeric_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_numeric_key_count")),
                    act_resume_diag_policy_obs_vs_seed_numeric_mismatch_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_numeric_mismatch_key_count")),
                    act_resume_diag_policy_obs_vs_seed_numeric_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_numeric_l2")),
                    act_resume_diag_policy_obs_vs_seed_numeric_mean_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_numeric_mean_abs")),
                    act_resume_diag_policy_obs_vs_seed_numeric_max_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_numeric_max_abs")),
                    act_resume_diag_policy_obs_vs_seed_numeric_worst_key=_optional_str(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_numeric_worst_key")),
                    act_resume_diag_policy_obs_vs_seed_numeric_worst_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_numeric_worst_l2")),
                    act_resume_diag_policy_obs_vs_seed_image_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_image_key_count")),
                    act_resume_diag_policy_obs_vs_seed_image_mismatch_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_image_mismatch_key_count")),
                    act_resume_diag_policy_obs_vs_seed_image_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_image_l2")),
                    act_resume_diag_policy_obs_vs_seed_image_mean_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_image_mean_abs")),
                    act_resume_diag_policy_obs_vs_seed_image_max_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_image_max_abs")),
                    act_resume_diag_policy_obs_vs_seed_image_worst_key=_optional_str(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_image_worst_key")),
                    act_resume_diag_policy_obs_vs_seed_image_worst_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_obs_vs_seed_image_worst_l2")),
                    act_resume_diag_visual_pose_vs_seed_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_key_count")),
                    act_resume_diag_visual_pose_vs_seed_common_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_common_key_count")),
                    act_resume_diag_visual_pose_vs_seed_missing_key_count=_optional_int(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_missing_key_count")),
                    act_resume_diag_visual_pose_vs_seed_pos_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_pos_l2")),
                    act_resume_diag_visual_pose_vs_seed_pos_max_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_pos_max_abs")),
                    act_resume_diag_visual_pose_vs_seed_pos_worst_key=_optional_str(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_pos_worst_key")),
                    act_resume_diag_visual_pose_vs_seed_pos_worst_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_pos_worst_l2")),
                    act_resume_diag_visual_pose_vs_seed_wrist_pos_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_wrist_pos_l2")),
                    act_resume_diag_visual_pose_vs_seed_head_pos_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_head_pos_l2")),
                    act_resume_diag_visual_pose_vs_seed_camera_pos_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_camera_pos_l2")),
                    act_resume_diag_visual_pose_vs_seed_xmat_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_xmat_l2")),
                    act_resume_diag_visual_pose_vs_seed_xmat_max_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_xmat_max_abs")),
                    act_resume_diag_visual_pose_vs_seed_xmat_worst_key=_optional_str(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_xmat_worst_key")),
                    act_resume_diag_visual_pose_vs_seed_xmat_worst_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_visual_pose_vs_seed_xmat_worst_l2")),
                    act_resume_diag_policy_low_dim_vs_seed_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_low_dim_vs_seed_l2")),
                    act_resume_diag_policy_low_dim_vs_seed_max_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_low_dim_vs_seed_max_abs")),
                    act_resume_diag_policy_low_dim_vs_seed_cosine=_optional_float(_safe_info_get(safety_info, "act_resume_diag_policy_low_dim_vs_seed_cosine")),
                    act_resume_diag_policy_low_dim_vs_seed_dim=_optional_int(_safe_info_get(safety_info, "act_resume_diag_policy_low_dim_vs_seed_dim")),
                    act_resume_diag_first_action_vs_target_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_first_action_vs_target_l2")),
                    act_resume_diag_first_action_vs_target_max_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_first_action_vs_target_max_abs")),
                    act_resume_diag_first_action_vs_target_cosine=_optional_float(_safe_info_get(safety_info, "act_resume_diag_first_action_vs_target_cosine")),
                    act_resume_diag_first_action_vs_target_dim=_optional_int(_safe_info_get(safety_info, "act_resume_diag_first_action_vs_target_dim")),
                    act_resume_diag_executed_vs_act_first_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_executed_vs_act_first_l2")),
                    act_resume_diag_executed_vs_act_first_max_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_executed_vs_act_first_max_abs")),
                    act_resume_diag_executed_vs_act_first_cosine=_optional_float(_safe_info_get(safety_info, "act_resume_diag_executed_vs_act_first_cosine")),
                    act_resume_diag_executed_vs_act_first_dim=_optional_int(_safe_info_get(safety_info, "act_resume_diag_executed_vs_act_first_dim")),
                    act_resume_diag_executed_vs_target_l2=_optional_float(_safe_info_get(safety_info, "act_resume_diag_executed_vs_target_l2")),
                    act_resume_diag_executed_vs_target_max_abs=_optional_float(_safe_info_get(safety_info, "act_resume_diag_executed_vs_target_max_abs")),
                    act_resume_diag_executed_vs_target_cosine=_optional_float(_safe_info_get(safety_info, "act_resume_diag_executed_vs_target_cosine")),
                    act_resume_diag_executed_vs_target_dim=_optional_int(_safe_info_get(safety_info, "act_resume_diag_executed_vs_target_dim")),
                    act_reentry_diag_active=_optional_bool(_safe_info_get(safety_info, "act_reentry_diag_active")),
                    act_reentry_diag_policy_low_dim_dim=_optional_int(_safe_info_get(safety_info, "act_reentry_diag_policy_low_dim_dim")),
                    act_reentry_diag_policy_low_dim_norm=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_policy_low_dim_norm")),
                    act_reentry_diag_policy_low_dim_mean=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_policy_low_dim_mean")),
                    act_reentry_diag_policy_low_dim_std=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_policy_low_dim_std")),
                    act_reentry_diag_policy_low_dim_max_abs=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_policy_low_dim_max_abs")),
                    act_reentry_diag_image_key_count=_optional_int(_safe_info_get(safety_info, "act_reentry_diag_image_key_count")),
                    act_reentry_diag_image_mean_mean=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_image_mean_mean")),
                    act_reentry_diag_image_std_mean=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_image_std_mean")),
                    act_reentry_diag_image_max_abs_mean=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_image_max_abs_mean")),
                    rhc_executed_action_available=_safe_info_get(safety_info, "rhc_executed_action_available"),
                    rhc_execution_index=_optional_int(_safe_info_get(safety_info, "rhc_execution_index")),
                    rhc_requested_vs_executed_l2=_optional_float(_safe_info_get(safety_info, "rhc_requested_vs_executed_l2")),
                    rhc_requested_vs_executed_max_abs=_optional_float(_safe_info_get(safety_info, "rhc_requested_vs_executed_max_abs")),
                    rhc_requested_vs_executed_cosine=_optional_float(_safe_info_get(safety_info, "rhc_requested_vs_executed_cosine")),
                    act_reentry_diag_act_first_vs_safe_l2=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_act_first_vs_safe_l2")),
                    act_reentry_diag_act_first_vs_safe_max_abs=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_act_first_vs_safe_max_abs")),
                    act_reentry_diag_act_first_vs_safe_cosine=_optional_float(_safe_info_get(safety_info, "act_reentry_diag_act_first_vs_safe_cosine")),
                    act_reentry_diag_act_first_vs_safe_dim=_optional_int(_safe_info_get(safety_info, "act_reentry_diag_act_first_vs_safe_dim")),
                    act_action_agreement_logged=_optional_bool(_safe_info_get(safety_info, "act_action_agreement_logged")),
                    act_action_agreement_context=_optional_str(_safe_info_get(safety_info, "act_action_agreement_context")),
                    act_action_agreement_pair_count=_optional_int(_safe_info_get(safety_info, "act_action_agreement_pair_count")),
                    act_action_agreement_post_recovery_or_reentry=_optional_bool(_safe_info_get(safety_info, "act_action_agreement_post_recovery_or_reentry")),
                    act_action_agreement_act_vs_safe_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_l2")),
                    act_action_agreement_act_vs_safe_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_max_abs")),
                    act_action_agreement_act_vs_safe_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_cosine")),
                    act_action_agreement_act_vs_safe_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_dim")),
                    act_action_agreement_act_vs_safe_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_arm_l2")),
                    act_action_agreement_act_vs_safe_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_arm_max_abs")),
                    act_action_agreement_act_vs_safe_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_arm_dim")),
                    act_action_agreement_act_vs_safe_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_gripper_delta")),
                    act_action_agreement_act_vs_safe_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_safe_gripper_abs_delta")),
                    act_action_agreement_act_vs_nominal_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_l2")),
                    act_action_agreement_act_vs_nominal_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_max_abs")),
                    act_action_agreement_act_vs_nominal_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_cosine")),
                    act_action_agreement_act_vs_nominal_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_dim")),
                    act_action_agreement_act_vs_nominal_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_arm_l2")),
                    act_action_agreement_act_vs_nominal_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_arm_max_abs")),
                    act_action_agreement_act_vs_nominal_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_arm_dim")),
                    act_action_agreement_act_vs_nominal_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_gripper_delta")),
                    act_action_agreement_act_vs_nominal_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_nominal_gripper_abs_delta")),
                    act_action_agreement_safe_vs_nominal_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_l2")),
                    act_action_agreement_safe_vs_nominal_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_max_abs")),
                    act_action_agreement_safe_vs_nominal_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_cosine")),
                    act_action_agreement_safe_vs_nominal_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_dim")),
                    act_action_agreement_safe_vs_nominal_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_arm_l2")),
                    act_action_agreement_safe_vs_nominal_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_arm_max_abs")),
                    act_action_agreement_safe_vs_nominal_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_arm_dim")),
                    act_action_agreement_safe_vs_nominal_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_gripper_delta")),
                    act_action_agreement_safe_vs_nominal_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_nominal_gripper_abs_delta")),
                    act_action_agreement_act_vs_target_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_l2")),
                    act_action_agreement_act_vs_target_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_max_abs")),
                    act_action_agreement_act_vs_target_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_cosine")),
                    act_action_agreement_act_vs_target_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_dim")),
                    act_action_agreement_act_vs_target_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_arm_l2")),
                    act_action_agreement_act_vs_target_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_arm_max_abs")),
                    act_action_agreement_act_vs_target_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_arm_dim")),
                    act_action_agreement_act_vs_target_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_gripper_delta")),
                    act_action_agreement_act_vs_target_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_target_gripper_abs_delta")),
                    act_action_agreement_safe_vs_target_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_l2")),
                    act_action_agreement_safe_vs_target_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_max_abs")),
                    act_action_agreement_safe_vs_target_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_cosine")),
                    act_action_agreement_safe_vs_target_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_dim")),
                    act_action_agreement_safe_vs_target_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_arm_l2")),
                    act_action_agreement_safe_vs_target_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_arm_max_abs")),
                    act_action_agreement_safe_vs_target_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_arm_dim")),
                    act_action_agreement_safe_vs_target_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_gripper_delta")),
                    act_action_agreement_safe_vs_target_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_target_gripper_abs_delta")),
                    act_action_agreement_act_vs_last_recovery_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_l2")),
                    act_action_agreement_act_vs_last_recovery_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_max_abs")),
                    act_action_agreement_act_vs_last_recovery_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_cosine")),
                    act_action_agreement_act_vs_last_recovery_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_dim")),
                    act_action_agreement_act_vs_last_recovery_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_arm_l2")),
                    act_action_agreement_act_vs_last_recovery_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_arm_max_abs")),
                    act_action_agreement_act_vs_last_recovery_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_arm_dim")),
                    act_action_agreement_act_vs_last_recovery_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_gripper_delta")),
                    act_action_agreement_act_vs_last_recovery_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_last_recovery_gripper_abs_delta")),
                    act_action_agreement_safe_vs_last_recovery_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_l2")),
                    act_action_agreement_safe_vs_last_recovery_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_max_abs")),
                    act_action_agreement_safe_vs_last_recovery_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_cosine")),
                    act_action_agreement_safe_vs_last_recovery_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_dim")),
                    act_action_agreement_safe_vs_last_recovery_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_arm_l2")),
                    act_action_agreement_safe_vs_last_recovery_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_arm_max_abs")),
                    act_action_agreement_safe_vs_last_recovery_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_arm_dim")),
                    act_action_agreement_safe_vs_last_recovery_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_gripper_delta")),
                    act_action_agreement_safe_vs_last_recovery_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_last_recovery_gripper_abs_delta")),
                    act_action_agreement_act_vs_handoff_release_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_l2")),
                    act_action_agreement_act_vs_handoff_release_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_max_abs")),
                    act_action_agreement_act_vs_handoff_release_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_cosine")),
                    act_action_agreement_act_vs_handoff_release_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_dim")),
                    act_action_agreement_act_vs_handoff_release_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_arm_l2")),
                    act_action_agreement_act_vs_handoff_release_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_arm_max_abs")),
                    act_action_agreement_act_vs_handoff_release_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_arm_dim")),
                    act_action_agreement_act_vs_handoff_release_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_gripper_delta")),
                    act_action_agreement_act_vs_handoff_release_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_handoff_release_gripper_abs_delta")),
                    act_action_agreement_safe_vs_handoff_release_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_l2")),
                    act_action_agreement_safe_vs_handoff_release_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_max_abs")),
                    act_action_agreement_safe_vs_handoff_release_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_cosine")),
                    act_action_agreement_safe_vs_handoff_release_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_dim")),
                    act_action_agreement_safe_vs_handoff_release_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_arm_l2")),
                    act_action_agreement_safe_vs_handoff_release_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_arm_max_abs")),
                    act_action_agreement_safe_vs_handoff_release_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_arm_dim")),
                    act_action_agreement_safe_vs_handoff_release_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_gripper_delta")),
                    act_action_agreement_safe_vs_handoff_release_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_handoff_release_gripper_abs_delta")),
                    act_action_agreement_act_vs_executed_handoff_release_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_l2")),
                    act_action_agreement_act_vs_executed_handoff_release_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_max_abs")),
                    act_action_agreement_act_vs_executed_handoff_release_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_cosine")),
                    act_action_agreement_act_vs_executed_handoff_release_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_dim")),
                    act_action_agreement_act_vs_executed_handoff_release_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_arm_l2")),
                    act_action_agreement_act_vs_executed_handoff_release_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_arm_max_abs")),
                    act_action_agreement_act_vs_executed_handoff_release_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_arm_dim")),
                    act_action_agreement_act_vs_executed_handoff_release_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_gripper_delta")),
                    act_action_agreement_act_vs_executed_handoff_release_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_act_vs_executed_handoff_release_gripper_abs_delta")),
                    act_action_agreement_safe_vs_executed_handoff_release_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_l2")),
                    act_action_agreement_safe_vs_executed_handoff_release_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_max_abs")),
                    act_action_agreement_safe_vs_executed_handoff_release_cosine=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_cosine")),
                    act_action_agreement_safe_vs_executed_handoff_release_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_dim")),
                    act_action_agreement_safe_vs_executed_handoff_release_arm_l2=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_arm_l2")),
                    act_action_agreement_safe_vs_executed_handoff_release_arm_max_abs=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_arm_max_abs")),
                    act_action_agreement_safe_vs_executed_handoff_release_arm_dim=_optional_int(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_arm_dim")),
                    act_action_agreement_safe_vs_executed_handoff_release_gripper_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_gripper_delta")),
                    act_action_agreement_safe_vs_executed_handoff_release_gripper_abs_delta=_optional_float(_safe_info_get(safety_info, "act_action_agreement_safe_vs_executed_handoff_release_gripper_abs_delta")),
                    min_h=min_h,
                    live_min_clearance=_optional_float(live_min_clearance),
                    live_h_violation_threshold=_optional_float(_safe_info_get(safety_info, "live_h_violation_threshold")),
                    live_h_violation_source=_optional_str(_safe_info_get(safety_info, "live_h_violation_source")),
                    live_h_argmin_pair_label=_optional_str(_safe_info_get(safety_info, "live_h_argmin_pair_label")),
                    live_h_argmin_pair_index=_optional_int(_safe_info_get(safety_info, "live_h_argmin_pair_index")),
                    live_h_argmin_dist_sq=_optional_float(_safe_info_get(safety_info, "live_h_argmin_dist_sq")),
                    live_h_argmin_combined_radius=_optional_float(_safe_info_get(safety_info, "live_h_argmin_combined_radius")),
                    live_h_argmin_robot_radius=_optional_float(_safe_info_get(safety_info, "live_h_argmin_robot_radius")),
                    live_h_argmin_human_radius=_optional_float(_safe_info_get(safety_info, "live_h_argmin_human_radius")),
                    live_h_clearance_values=_jsonable_trace_value(_safe_info_get(safety_info, "live_h_clearance_values")),
                    live_raw_min_h=_optional_float(_safe_info_get(safety_info, "live_raw_min_h")),
                    live_raw_min_h_pair_label=_optional_str(_safe_info_get(safety_info, "live_raw_min_h_pair_label")),
                    h_values=h_values,
                    h_argmin_horizon_index=_optional_int(_safe_info_get(
                        safety_info,
                        "h_argmin_horizon_index",
                        h_attribution_info.get("h_argmin_horizon_index"),
                    )),
                    h_argmin_pair_index=_optional_int(_safe_info_get(
                        safety_info,
                        "h_argmin_pair_index",
                        h_attribution_info.get("h_argmin_pair_index"),
                    )),
                    h_argmin_robot_part=_optional_str(_safe_info_get(
                        safety_info,
                        "h_argmin_robot_part",
                        h_attribution_info.get("h_argmin_robot_part"),
                    )),
                    h_argmin_human_part=_optional_str(_safe_info_get(
                        safety_info,
                        "h_argmin_human_part",
                        h_attribution_info.get("h_argmin_human_part"),
                    )),
                    h_argmin_human_capsule_index=_optional_int(_safe_info_get(
                        safety_info,
                        "h_argmin_human_capsule_index",
                        h_attribution_info.get("h_argmin_human_capsule_index"),
                    )),
                    h_argmin_human_arm_index=_optional_int(_safe_info_get(
                        safety_info,
                        "h_argmin_human_arm_index",
                        h_attribution_info.get("h_argmin_human_arm_index"),
                    )),
                    h_pair_values_at_argmin=_jsonable_trace_value(_safe_info_get(
                        safety_info,
                        "h_pair_values_at_argmin",
                        h_attribution_info.get("h_pair_values_at_argmin"),
                    )),
                    h_violation=h_violation,
                    live_h_monitor_skipped=bool(live_h_monitor_skipped),
                    chunk_min_clearance=_safe_info_get(safety_info, "min_clearance"),
                    chunk_first_violation=_safe_info_get(safety_info, "first_violation"),
                    chunk_unsafe_count=_safe_info_get(safety_info, "unsafe_count"),
                    nominal_path_blocked=_optional_bool(_safe_info_get(safety_info, "nominal_path_blocked")),
                    nominal_path_blockage_check_available=_optional_bool(_safe_info_get(safety_info, "nominal_path_blockage_check_available")),
                    nominal_path_blockage_source=_optional_str(_safe_info_get(safety_info, "nominal_path_blockage_source")),
                    path_block_pause_sufficient=_optional_bool(_safe_info_get(safety_info, "path_block_pause_sufficient")),
                    path_block_pause_sufficiency_available=_optional_bool(_safe_info_get(safety_info, "path_block_pause_sufficiency_available")),
                    path_block_pause_sufficiency_source=_optional_str(_safe_info_get(safety_info, "path_block_pause_sufficiency_source")),
                    path_block_requires_bypass=_optional_bool(_safe_info_get(safety_info, "path_block_requires_bypass")),
                    nominal_goal_blocked=_optional_bool(_safe_info_get(safety_info, "nominal_goal_blocked")),
                    nominal_goal_distance=_optional_float(_safe_info_get(safety_info, "nominal_goal_distance")),
                    nominal_goal_blockage_check_available=_optional_bool(_safe_info_get(safety_info, "nominal_goal_blockage_check_available")),
                    nominal_goal_blockage_source=_optional_str(_safe_info_get(safety_info, "nominal_goal_blockage_source")),
                    nominal_blockage_route=_optional_str(_safe_info_get(safety_info, "nominal_blockage_route")),
                    goal_block_release_wait_active=_optional_bool(_safe_info_get(safety_info, "goal_block_release_wait_active")),
                    goal_block_release_wait_completed=_optional_bool(_safe_info_get(safety_info, "goal_block_release_wait_completed")),
                    goal_block_release_wait_expired=_optional_bool(_safe_info_get(safety_info, "goal_block_release_wait_expired")),
                    goal_block_release_wait_count=_optional_int(_safe_info_get(safety_info, "goal_block_release_wait_count")),
                    nominal_goal_blocked_deform_suppressed=_optional_bool(_safe_info_get(safety_info, "nominal_goal_blocked_deform_suppressed")),
                    nominal_goal_blocked_deform_suppression_reason=_optional_str(_safe_info_get(safety_info, "nominal_goal_blocked_deform_suppression_reason")),
                    nominal_collision_attribution_enabled=_optional_bool(_safe_info_get(safety_info, "nominal_collision_attribution_enabled")),
                    nominal_collision_source=_optional_str(_safe_info_get(safety_info, "nominal_collision_source")),
                    nominal_current_human_unsafe=_optional_bool(_safe_info_get(safety_info, "nominal_current_human_unsafe")),
                    nominal_predicted_human_unsafe=_optional_bool(_safe_info_get(safety_info, "nominal_predicted_human_unsafe")),
                    nominal_predicted_only_collision=_optional_bool(_safe_info_get(safety_info, "nominal_predicted_only_collision")),
                    nominal_attribution_pause_recommended=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_pause_recommended")),
                    nominal_attribution_deform_recommended=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_deform_recommended")),
                    nominal_attribution_pause_applied=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_pause_applied")),
                    nominal_attribution_pause_budget_open=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_pause_budget_open")),
                    nominal_attribution_pause_budget_exhausted=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_pause_budget_exhausted")),
                    nominal_attribution_pause_trigger_reason=_optional_str(_safe_info_get(safety_info, "nominal_attribution_pause_trigger_reason")),
                    nominal_attribution_deform_after_pause_budget=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_deform_after_pause_budget")),
                    nominal_attribution_pause_budget_brake_streak=_optional_int(_safe_info_get(safety_info, "nominal_attribution_pause_budget_brake_streak")),
                    nominal_attribution_pause_budget_unsafe_streak=_optional_int(_safe_info_get(safety_info, "nominal_attribution_pause_budget_unsafe_streak")),
                    nominal_attribution_pause_budget_max_brake_steps=_optional_int(_safe_info_get(safety_info, "nominal_attribution_pause_budget_max_brake_steps")),
                    nominal_attribution_pause_budget_min_unsafe_steps=_optional_int(_safe_info_get(safety_info, "nominal_attribution_pause_budget_min_unsafe_steps")),
                    nominal_attribution_pause_forced_deform=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_pause_forced_deform")),
                    nominal_attribution_pause_force_deform_reason=_optional_str(_safe_info_get(safety_info, "nominal_attribution_pause_force_deform_reason")),
                    nominal_attribution_pause_forced_deform_suppressed=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_pause_forced_deform_suppressed")),
                    nominal_attribution_pause_force_deform_suppression_reason=_optional_str(_safe_info_get(safety_info, "nominal_attribution_pause_force_deform_suppression_reason")),
                    nominal_attribution_stop_counterfactual_safe=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_stop_counterfactual_safe")),
                    nominal_attribution_stop_counterfactual_min_clearance=_optional_float(_safe_info_get(safety_info, "nominal_attribution_stop_counterfactual_min_clearance")),
                    nominal_attribution_stop_counterfactual_first_violation=_optional_int(_safe_info_get(safety_info, "nominal_attribution_stop_counterfactual_first_violation")),
                    nominal_attribution_goal_blocked=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_goal_blocked")),
                    nominal_attribution_goal_distance=_optional_float(_safe_info_get(safety_info, "nominal_attribution_goal_distance")),
                    nominal_attribution_goal_check_available=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_goal_check_available")),
                    nominal_attribution_goal_check_source=_optional_str(_safe_info_get(safety_info, "nominal_attribution_goal_check_source")),
                    nominal_attribution_goal_allows_suppression=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_goal_allows_suppression")),
                    nominal_attribution_human_motion_speed=_optional_float(_safe_info_get(safety_info, "nominal_attribution_human_motion_speed")),
                    nominal_attribution_human_motion_static=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_human_motion_static")),
                    nominal_attribution_human_motion_static_threshold=_optional_float(_safe_info_get(safety_info, "nominal_attribution_human_motion_static_threshold")),
                    nominal_attribution_early_deform_suppressed=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_early_deform_suppressed")),
                    nominal_attribution_early_deform_suppression_reason=_optional_str(_safe_info_get(safety_info, "nominal_attribution_early_deform_suppression_reason")),
                    nominal_attribution_early_deform_candidate_accepted_before_suppression=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_early_deform_candidate_accepted_before_suppression")),
                    nominal_attribution_early_deform_stop_counterfactual_safe=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_early_deform_stop_counterfactual_safe")),
                    nominal_attribution_early_deform_stop_counterfactual_min_clearance=_optional_float(_safe_info_get(safety_info, "nominal_attribution_early_deform_stop_counterfactual_min_clearance")),
                    nominal_attribution_early_deform_human_motion_speed=_optional_float(_safe_info_get(safety_info, "nominal_attribution_early_deform_human_motion_speed")),
                    nominal_attribution_early_deform_human_motion_displacement=_optional_float(_safe_info_get(safety_info, "nominal_attribution_early_deform_human_motion_displacement")),
                    nominal_attribution_early_deform_human_motion_static=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_early_deform_human_motion_static")),
                    nominal_attribution_early_deform_goal_blocked=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_early_deform_goal_blocked")),
                    nominal_attribution_early_deform_goal_distance=_optional_float(_safe_info_get(safety_info, "nominal_attribution_early_deform_goal_distance")),
                    nominal_attribution_early_deform_goal_check_available=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_early_deform_goal_check_available")),
                    nominal_attribution_early_deform_goal_check_source=_optional_str(_safe_info_get(safety_info, "nominal_attribution_early_deform_goal_check_source")),
                    stationary_human_local_escape_enabled=_optional_bool(_safe_info_get(safety_info, "stationary_human_local_escape_enabled")),
                    stationary_human_local_escape_applied=_optional_bool(_safe_info_get(safety_info, "stationary_human_local_escape_applied")),
                    stationary_human_local_escape_skip_reason=_optional_str(_safe_info_get(safety_info, "stationary_human_local_escape_skip_reason")),
                    stationary_human_local_escape_counter=_optional_int(_safe_info_get(safety_info, "stationary_human_local_escape_counter")),
                    stationary_human_local_escape_max_steps=_optional_int(_safe_info_get(safety_info, "stationary_human_local_escape_max_steps")),
                    stationary_human_local_escape_stop_counterfactual_safe=_optional_bool(_safe_info_get(safety_info, "stationary_human_local_escape_stop_counterfactual_safe")),
                    stationary_human_local_escape_stop_counterfactual_min_clearance=_optional_float(_safe_info_get(safety_info, "stationary_human_local_escape_stop_counterfactual_min_clearance")),
                    stationary_human_local_escape_human_motion_speed=_optional_float(_safe_info_get(safety_info, "stationary_human_local_escape_human_motion_speed")),
                    stationary_human_local_escape_human_motion_displacement=_optional_float(_safe_info_get(safety_info, "stationary_human_local_escape_human_motion_displacement")),
                    stationary_human_local_escape_human_motion_static=_optional_bool(_safe_info_get(safety_info, "stationary_human_local_escape_human_motion_static")),
                    stationary_human_local_escape_goal_blocked=_optional_bool(_safe_info_get(safety_info, "stationary_human_local_escape_goal_blocked")),
                    stationary_human_local_escape_goal_distance=_optional_float(_safe_info_get(safety_info, "stationary_human_local_escape_goal_distance")),
                    stationary_human_local_escape_goal_check_available=_optional_bool(_safe_info_get(safety_info, "stationary_human_local_escape_goal_check_available")),
                    stationary_human_local_escape_goal_check_source=_optional_str(_safe_info_get(safety_info, "stationary_human_local_escape_goal_check_source")),
                    stationary_human_local_escape_current_human_wait=_optional_bool(_safe_info_get(safety_info, "stationary_human_local_escape_current_human_wait")),
                    stationary_human_local_escape_current_human_wait_clearance=_optional_float(_safe_info_get(safety_info, "stationary_human_local_escape_current_human_wait_clearance")),
                    stationary_human_local_escape_nominal_source=_optional_str(_safe_info_get(safety_info, "stationary_human_local_escape_nominal_source")),
                    stationary_human_local_escape_candidate_accepted=_optional_bool(_safe_info_get(safety_info, "stationary_human_local_escape_candidate_accepted")),
                    stationary_human_local_escape_candidate_path=_optional_str(_safe_info_get(safety_info, "stationary_human_local_escape_candidate_path")),
                    stationary_human_local_escape_candidate_hold_clearance=_optional_float(_safe_info_get(safety_info, "stationary_human_local_escape_candidate_hold_clearance")),
                    stationary_human_local_escape_trigger_reason=_optional_str(_safe_info_get(safety_info, "stationary_human_local_escape_trigger_reason")),
                    optimized_candidate_suppressed_by_local_escape=_optional_bool(_safe_info_get(safety_info, "optimized_candidate_suppressed_by_local_escape")),
                    optimized_accepted_before_local_escape=_optional_bool(_safe_info_get(safety_info, "optimized_accepted_before_local_escape")),
                    recover_accepted_before_local_escape=_optional_bool(_safe_info_get(safety_info, "recover_accepted_before_local_escape")),
                    return_accepted_before_local_escape=_optional_bool(_safe_info_get(safety_info, "return_accepted_before_local_escape")),
                    nominal_attribution_pause_current_human_unsafe=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_pause_current_human_unsafe")),
                    nominal_attribution_pause_current_min_clearance=_optional_float(_safe_info_get(safety_info, "nominal_attribution_pause_current_min_clearance")),
                    nominal_attribution_pause_min_clearance_threshold=_optional_float(_safe_info_get(safety_info, "nominal_attribution_pause_min_clearance_threshold")),
                    nominal_attribution_bypass_soft_pass_through=_optional_bool(_safe_info_get(safety_info, "nominal_attribution_bypass_soft_pass_through")),
                    policy_collision_slowdown_enabled=_optional_bool(_safe_info_get(safety_info, "policy_collision_slowdown_enabled")),
                    policy_collision_slowdown_applied=_optional_bool(_safe_info_get(safety_info, "policy_collision_slowdown_applied")),
                    policy_collision_slowdown_skip_reason=_optional_str(_safe_info_get(safety_info, "policy_collision_slowdown_skip_reason")),
                    policy_collision_slowdown_trigger_reason=_optional_str(_safe_info_get(safety_info, "policy_collision_slowdown_trigger_reason")),
                    policy_collision_slowdown_counter=_optional_int(_safe_info_get(safety_info, "policy_collision_slowdown_counter")),
                    policy_collision_slowdown_max_steps=_optional_int(_safe_info_get(safety_info, "policy_collision_slowdown_max_steps")),
                    policy_collision_slowdown_min_first_violation=_optional_int(_safe_info_get(safety_info, "policy_collision_slowdown_min_first_violation")),
                    policy_collision_slowdown_first_violation=_optional_int(_safe_info_get(safety_info, "policy_collision_slowdown_first_violation")),
                    policy_collision_slowdown_to_pause=_optional_bool(_safe_info_get(safety_info, "policy_collision_slowdown_to_pause")),
                    policy_collision_pause_after_slowdown=_optional_bool(_safe_info_get(safety_info, "policy_collision_pause_after_slowdown")),
                    slowdown_safe=_optional_bool(_safe_info_get(safety_info, "slowdown_safe")),
                    slowdown_applied=_optional_bool(_safe_info_get(safety_info, "slowdown_applied")),
                    slowdown_factor=_optional_float(_safe_info_get(safety_info, "slowdown_factor")),
                    slowdown_min_clearance=_optional_float(_safe_info_get(safety_info, "slowdown_min_clearance")),
                    slowdown_first_violation=_optional_int(_safe_info_get(safety_info, "slowdown_first_violation")),
                    slowdown_unsafe_count=_optional_int(_safe_info_get(safety_info, "slowdown_unsafe_count")),
                    slowdown_horizon_safe=_optional_bool(_safe_info_get(safety_info, "slowdown_horizon_safe")),
                    slowdown_progress_scale=_optional_float(_safe_info_get(safety_info, "slowdown_progress_scale")),
                    slowdown_skip_reason=_optional_str(_safe_info_get(safety_info, "slowdown_skip_reason")),
                    nominal_current_human_horizon_safe=_optional_bool(_safe_info_get(safety_info, "nominal_current_human_horizon_safe")),
                    nominal_current_human_min_clearance=_optional_float(_safe_info_get(safety_info, "nominal_current_human_min_clearance")),
                    nominal_current_human_first_violation=_optional_int(_safe_info_get(safety_info, "nominal_current_human_first_violation")),
                    nominal_current_human_unsafe_count=_optional_int(_safe_info_get(safety_info, "nominal_current_human_unsafe_count")),
                    nominal_predicted_human_horizon_safe=_optional_bool(_safe_info_get(safety_info, "nominal_predicted_human_horizon_safe")),
                    nominal_predicted_human_min_clearance=_optional_float(_safe_info_get(safety_info, "nominal_predicted_human_min_clearance")),
                    nominal_predicted_human_first_violation=_optional_int(_safe_info_get(safety_info, "nominal_predicted_human_first_violation")),
                    nominal_predicted_human_unsafe_count=_optional_int(_safe_info_get(safety_info, "nominal_predicted_human_unsafe_count")),
                    pause_exit_smoothing_enabled=_optional_bool(_safe_info_get(safety_info, "pause_exit_smoothing_enabled")),
                    pause_exit_smoothing_armed=_optional_bool(_safe_info_get(safety_info, "pause_exit_smoothing_armed")),
                    pause_exit_smoothing_applied=_optional_bool(_safe_info_get(safety_info, "pause_exit_smoothing_applied")),
                    pause_exit_smoothing_context=_optional_str(_safe_info_get(safety_info, "pause_exit_smoothing_context")),
                    pause_exit_smoothing_arm_reason=_optional_str(_safe_info_get(safety_info, "pause_exit_smoothing_arm_reason")),
                    pause_exit_smoothing_skip_reason=_optional_str(_safe_info_get(safety_info, "pause_exit_smoothing_skip_reason")),
                    pause_exit_smoothing_alpha=_optional_float(_safe_info_get(safety_info, "pause_exit_smoothing_alpha")),
                    pause_exit_smoothing_step_index=_optional_int(_safe_info_get(safety_info, "pause_exit_smoothing_step_index")),
                    pause_exit_smoothing_remaining=_optional_int(_safe_info_get(safety_info, "pause_exit_smoothing_remaining")),
                    pause_exit_smoothing_action_l2_before=_optional_float(_safe_info_get(safety_info, "pause_exit_smoothing_action_l2_before")),
                    pause_exit_smoothing_action_l2_after=_optional_float(_safe_info_get(safety_info, "pause_exit_smoothing_action_l2_after")),
                    pause_exit_smoothing_original_min_clearance=_optional_float(_safe_info_get(safety_info, "pause_exit_smoothing_original_min_clearance")),
                    pause_exit_smoothing_smoothed_min_clearance=_optional_float(_safe_info_get(safety_info, "pause_exit_smoothing_smoothed_min_clearance")),
                    intervention_policy=_optional_str(_safe_info_get(safety_info, "intervention_policy")),
                    intervention_previous_mode=_optional_str(_safe_info_get(safety_info, "intervention_previous_mode")),
                    intervention_mode=_optional_str(_safe_info_get(safety_info, "intervention_mode")),
                    intervention_transition_reason=_optional_str(_safe_info_get(safety_info, "intervention_transition_reason")),
                    intervention_unsafe_reason=_optional_str(_safe_info_get(safety_info, "intervention_unsafe_reason")),
                    intervention_pause_counter=_optional_int(_safe_info_get(safety_info, "intervention_pause_counter")),
                    intervention_slowdown_counter=_optional_int(_safe_info_get(safety_info, "intervention_slowdown_counter")),
                    intervention_deform_valid_counter=_optional_int(_safe_info_get(safety_info, "intervention_deform_valid_counter")),
                    intervention_nominal_clear_counter=_optional_int(_safe_info_get(safety_info, "intervention_nominal_clear_counter")),
                    intervention_deform_commit_counter=_optional_int(_safe_info_get(safety_info, "intervention_deform_commit_counter")),
                    intervention_deform_stall_counter=_optional_int(_safe_info_get(safety_info, "intervention_deform_stall_counter")),
                    intervention_deform_failure_latched=_optional_bool(_safe_info_get(safety_info, "intervention_deform_failure_latched")),
                    intervention_deformation_admissible=_optional_bool(_safe_info_get(safety_info, "intervention_deformation_admissible")),
                    intervention_deform_horizon_safe=_optional_bool(_safe_info_get(safety_info, "intervention_deform_horizon_safe")),
                    intervention_deform_has_progress=_optional_bool(_safe_info_get(safety_info, "intervention_deform_has_progress")),
                    intervention_deform_executable=_optional_bool(_safe_info_get(safety_info, "intervention_deform_executable")),
                    intervention_deform_min_distance=_optional_float(_safe_info_get(safety_info, "intervention_deform_min_distance")),
                    intervention_deform_progress=_optional_float(_safe_info_get(safety_info, "intervention_deform_progress")),
                    intervention_deform_min_velocity_ratio=_optional_float(_safe_info_get(safety_info, "intervention_deform_min_velocity_ratio")),
                    intervention_deform_failure_reason=_optional_str(_safe_info_get(safety_info, "intervention_deform_failure_reason")),
                    intervention_deform_commit_blocked=_optional_bool(_safe_info_get(safety_info, "intervention_deform_commit_blocked")),
                    intervention_deform_commit_block_reason=_optional_str(_safe_info_get(safety_info, "intervention_deform_commit_block_reason")),
                    intervention_human_motion_since_failure=_optional_float(_safe_info_get(safety_info, "intervention_human_motion_since_failure")),
                    intervention_nominal_change_since_failure=_optional_float(_safe_info_get(safety_info, "intervention_nominal_change_since_failure")),
                    intervention_latch_release_reason=_optional_str(_safe_info_get(safety_info, "intervention_latch_release_reason")),
                    intervention_resume_block_reason=_optional_str(_safe_info_get(safety_info, "intervention_resume_block_reason")),
                    intervention_resume_blend_active=_optional_bool(_safe_info_get(safety_info, "intervention_resume_blend_active")),
                    intervention_resume_blend_alpha=_optional_float(_safe_info_get(safety_info, "intervention_resume_blend_alpha")),
                    intervention_resume_blend_step=_optional_int(_safe_info_get(safety_info, "intervention_resume_blend_step")),
                    intervention_resume_blend_steps=_optional_int(_safe_info_get(safety_info, "intervention_resume_blend_steps")),
                    intervention_resume_blend_min_clearance=_optional_float(_safe_info_get(safety_info, "intervention_resume_blend_min_clearance")),
                    intervention_fsm_handoff_min_commit_ok=_optional_bool(_safe_info_get(safety_info, "intervention_fsm_handoff_min_commit_ok")),
                    intervention_fsm_handoff_block_reason=_optional_str(_safe_info_get(safety_info, "intervention_fsm_handoff_block_reason")),
                    conflict_aware_policy_enabled=_optional_bool(_safe_info_get(safety_info, "conflict_aware_policy_enabled")),
                    conflict_aware_previous_mode=_optional_str(_safe_info_get(safety_info, "conflict_aware_previous_mode")),
                    conflict_aware_selected_mode=_optional_str(_safe_info_get(safety_info, "conflict_aware_selected_mode")),
                    conflict_aware_decision_reason=_optional_str(_safe_info_get(safety_info, "conflict_aware_decision_reason")),
                    conflict_aware_nominal_min_distance=_optional_float(_safe_info_get(safety_info, "conflict_aware_nominal_min_distance")),
                    conflict_aware_stop_min_distance=_optional_float(_safe_info_get(safety_info, "conflict_aware_stop_min_distance")),
                    conflict_aware_stopping_sufficient=_optional_bool(_safe_info_get(safety_info, "conflict_aware_stopping_sufficient")),
                    conflict_aware_goal_blocked=_optional_bool(_safe_info_get(safety_info, "conflict_aware_goal_blocked")),
                    conflict_aware_deformation_safe=_optional_bool(_safe_info_get(safety_info, "conflict_aware_deformation_safe")),
                    conflict_aware_deformation_admissible=_optional_bool(_safe_info_get(safety_info, "conflict_aware_deformation_admissible")),
                    conflict_aware_task_progress_value=_optional_float(_safe_info_get(safety_info, "conflict_aware_task_progress_value")),
                    conflict_aware_terminal_deviation=_optional_float(_safe_info_get(safety_info, "conflict_aware_terminal_deviation")),
                    conflict_aware_pause_budget_used=_optional_int(_safe_info_get(safety_info, "conflict_aware_pause_budget_used")),
                    conflict_aware_commit_steps_remaining=_optional_int(_safe_info_get(safety_info, "conflict_aware_commit_steps_remaining")),
                    conflict_aware_resume_hysteresis_count=_optional_int(_safe_info_get(safety_info, "conflict_aware_resume_hysteresis_count")),
                    conflict_aware_goal_distance=_optional_float(_safe_info_get(safety_info, "conflict_aware_goal_distance")),
                    conflict_aware_goal_check_available=_optional_bool(_safe_info_get(safety_info, "conflict_aware_goal_check_available")),
                    horizon_risk_gap=horizon_risk_gap,
                    horizon_risk_gap_active=horizon_risk_gap_active,
                    horizon_clearance_drop=horizon_clearance_drop,
                    pacs_background_check_only=_safe_info_get(safety_info, "pacs_background_check_only"),
                    pacs_background_safety_mode=_safe_info_get(safety_info, "pacs_background_safety_mode"),
                    pacs_background_deformation_source=_safe_info_get(safety_info, "pacs_background_deformation_source"),
                    pacs_background_retiming_source=_safe_info_get(safety_info, "pacs_background_retiming_source"),
                    pacs_background_brake_step=_safe_info_get(safety_info, "pacs_background_brake_step"),
                    pacs_background_act_step=_safe_info_get(safety_info, "pacs_background_act_step"),
                    pacs_background_min_clearance=_safe_info_get(safety_info, "pacs_background_min_clearance"),
                    pacs_background_first_violation=_safe_info_get(safety_info, "pacs_background_first_violation"),
                    pacs_background_unsafe_count=_safe_info_get(safety_info, "pacs_background_unsafe_count"),
                    pacs_background_arm_delta=_safe_info_get(safety_info, "pacs_background_arm_delta"),
                    pacs_background_chunk_arm_delta=_safe_info_get(safety_info, "pacs_background_chunk_arm_delta"),
                    pacs_background_chunk_modified_fraction=_safe_info_get(safety_info, "pacs_background_chunk_modified_fraction"),
                    pacs_background_retiming_arm_delta=_safe_info_get(safety_info, "pacs_background_retiming_arm_delta"),
                    pacs_background_retiming_chunk_arm_delta=_safe_info_get(safety_info, "pacs_background_retiming_chunk_arm_delta"),
                    pacs_background_retiming_changed_fraction=_safe_info_get(safety_info, "pacs_background_retiming_changed_fraction"),

                    contact_count=contact_count,
                    contact_pairs=contact_pairs,

                    arm_delta=arm_delta,
                    base_delta=base_delta,
                    non_arm_delta=non_arm_delta,
                    full_delta=full_delta,
                    chunk_arm_delta=chunk_arm_delta,
                    chunk_base_delta=chunk_base_delta,
                    chunk_non_arm_delta=chunk_non_arm_delta,
                    chunk_full_delta=chunk_full_delta,
                    chunk_modified_fraction=chunk_advantage_metrics["chunk_modified_fraction"],
                    chunk_modified_steps=chunk_advantage_metrics["chunk_modified_steps"],
                    chunk_first_modified_step=chunk_advantage_metrics["chunk_first_modified_step"],
                    chunk_last_modified_step=chunk_advantage_metrics["chunk_last_modified_step"],
                    chunk_mean_step_arm_delta=chunk_advantage_metrics["chunk_mean_step_arm_delta"],
                    chunk_max_step_arm_delta=chunk_advantage_metrics["chunk_max_step_arm_delta"],
                    chunk_future_arm_delta=chunk_advantage_metrics["chunk_future_arm_delta"],
                    chunk_future_edit_fraction=chunk_advantage_metrics["chunk_future_edit_fraction"],
                    chunk_first_edit_fraction=chunk_advantage_metrics["chunk_first_edit_fraction"],
                    chunk_safe_arm_variation=chunk_advantage_metrics["chunk_safe_arm_variation"],
                    chunk_nominal_arm_variation=chunk_advantage_metrics["chunk_nominal_arm_variation"],
                    chunk_arm_variation_delta=chunk_advantage_metrics["chunk_arm_variation_delta"],
                    chunk_edit_variation=chunk_advantage_metrics["chunk_edit_variation"],
                    path_mean_deviation=path_deviation_metrics["path_mean_deviation"],
                    path_max_deviation=path_deviation_metrics["path_max_deviation"],
                    path_final_deviation=path_deviation_metrics["path_final_deviation"],
                    chunk_preemptive_intervention=chunk_advantage_metrics["chunk_preemptive_intervention"],
                    intervention_active=bool(chunk_arm_delta > args.intervention_eps),

                    nominal_arm_min=float(np.min(nominal_arm)),
                    nominal_arm_max=float(np.max(nominal_arm)),
                    safe_arm_min=float(np.min(safe_arm)),
                    safe_arm_max=float(np.max(safe_arm)),

                    action_norm=float(np.linalg.norm(first_action)),
                    safe_action_norm=float(np.linalg.norm(safe_first_action)),
                    raw_action_norm=raw_action_norm,
                    raw_arm_min=raw_arm_min,
                    raw_arm_max=raw_arm_max,
                    chunk_action_norm=float(np.linalg.norm(nominal_chunk_for_metrics)),
                    safe_chunk_action_norm=float(np.linalg.norm(safe_chunk_for_metrics)),
                    safety_mode=(
                        _safe_info_get(safety_info, "safety_mode")
                        or _safe_info_get(safety_info, "mode")
                    ),
                    pause_reason=_safe_info_get(safety_info, "pause_reason"),
                    deformation_source=_safe_info_get(safety_info, "deformation_source"),
                    deformation_norm=_safe_info_get(safety_info, "deformation_norm"),
                    retiming_source=_safe_info_get(safety_info, "retiming_source"),
                    retiming_norm=_safe_info_get(safety_info, "retiming_norm"),
                    deform_safe=_safe_info_get(safety_info, "deform_safe"),
                    deform_min_clearance=_safe_info_get(safety_info, "deform_min_clearance"),
                    chunk_deform_scale=_safe_info_get(safety_info, "chunk_deform_scale"),
                    chunk_deform_attempts=_safe_info_get(safety_info, "chunk_deform_attempts"),
                    deform_mode=_safe_info_get(safety_info, "deform_mode"),
                    optimized_accepted=_safe_info_get(safety_info, "optimized_accepted"),
                    optimized_fallback=_safe_info_get(safety_info, "optimized_fallback"),
                    optimized_reject_reason=_safe_info_get(safety_info, "optimized_reject_reason"),
                    debug_safety_feasibility=_safe_info_get(
                        safety_info, "debug_safety_feasibility"
                    ),
                    safety_rejected=_safe_info_get(safety_info, "safety_rejected"),
                    recovery_rejected=_safe_info_get(safety_info, "recovery_rejected"),
                    rejection_cause=_safe_info_get(safety_info, "rejection_cause"),
                    best_min_clearance=_safe_info_get(safety_info, "best_min_clearance"),
                    required_min_clearance=_safe_info_get(
                        safety_info, "required_min_clearance"
                    ),
                    clearance_gap=_safe_info_get(safety_info, "clearance_gap"),
                    recovery_mode=_safe_info_get(safety_info, "recovery_mode"),
                    recovery_phase=_safe_info_get(safety_info, "recovery_phase"),
                    cached_motion_active=_safe_info_get(
                        safety_info, "cached_motion_active"
                    ),
                    deform_stage_min_clearance=_safe_info_get(
                        safety_info, "deform_stage_min_clearance"
                    ),
                    deform_stage_accepted=_safe_info_get(safety_info, "deform_stage_accepted"),
                    recover_min_clearance=_safe_info_get(
                        safety_info, "recover_min_clearance"
                    ),
                    recover_rejoin_loss=_safe_info_get(
                        safety_info, "recover_rejoin_loss"
                    ),
                    recover_target_index=_safe_info_get(
                        safety_info, "recover_target_index"
                    ),
                    recover_accepted=_safe_info_get(safety_info, "recover_accepted"),
                    recover_corridor_accepted=_safe_info_get(
                        safety_info,
                        "recover_corridor_accepted",
                        _safe_info_get(safety_info, "return_accepted"),
                    ),
                    recover_required=_safe_info_get(safety_info, "recover_required"),
                    recovery_candidate_class=_safe_info_get(safety_info, "recovery_candidate_class"),
                    recover_reject_reason=_safe_info_get(safety_info, "recover_reject_reason"),
                    recover_path_min_clearance=_safe_info_get(safety_info, "recover_path_min_clearance"),
                    recover_immediate_clearance=_safe_info_get(safety_info, "recover_immediate_clearance"),
                    recover_prefix_min_clearance=_safe_info_get(safety_info, "recover_prefix_min_clearance"),
                    recover_path_safe=_safe_info_get(safety_info, "recover_path_safe"),
                    recover_immediate_safe=_safe_info_get(safety_info, "recover_immediate_safe"),
                    recover_prefix_safe=_safe_info_get(safety_info, "recover_prefix_safe"),
                    recover_safe_prefix_len=_safe_info_get(safety_info, "recover_safe_prefix_len"),
                    recover_target_key=_safe_info_get(safety_info, "recover_target_key"),
                    recovery_path_failure_streak=_safe_info_get(safety_info, "recovery_path_failure_streak"),
                    direct_rejoin_attempted=_safe_info_get(safety_info, "direct_rejoin_attempted"),
                    direct_rejoin_rejected=_safe_info_get(safety_info, "direct_rejoin_rejected"),
                    detour_rejoin_attempted=_safe_info_get(safety_info, "detour_rejoin_attempted"),
                    detour_rejoin_accepted=_safe_info_get(safety_info, "detour_rejoin_accepted"),
                    delayed_rejoin_active=_safe_info_get(safety_info, "delayed_rejoin_active"),
                    delayed_rejoin_steps=_safe_info_get(safety_info, "delayed_rejoin_steps"),
                    repeated_unsafe_target=_safe_info_get(safety_info, "repeated_unsafe_target"),
                    post_recovery_act_window_active=_safe_info_get(safety_info, "post_recovery_act_window_active"),
                    post_recovery_act_steps_remaining=_safe_info_get(safety_info, "post_recovery_act_steps_remaining"),
                    post_recovery_act_window_interrupted=_safe_info_get(safety_info, "post_recovery_act_window_interrupted"),
                    resumed_from_cached_index=_safe_info_get(
                        safety_info, "resumed_from_cached_index"
                    ),
                    is_recoverable=_safe_info_get(safety_info, "is_recoverable"),
                    rejoin_index=_safe_info_get(safety_info, "rejoin_index"),
                    rejoin_cost=_safe_info_get(safety_info, "rejoin_cost"),
                    safety_loss=_safe_info_get(safety_info, "safety_loss"),
                    action_deviation_loss=_safe_info_get(safety_info, "action_deviation_loss"),
                    path_loss=_safe_info_get(safety_info, "path_loss"),
                    rejoin_loss=_safe_info_get(safety_info, "rejoin_loss"),
                    q_rejoin_loss=_safe_info_get(safety_info, "q_rejoin_loss"),
                    q_rejoin_dist=_safe_info_get(safety_info, "q_rejoin_dist"),
                    q_rejoin_threshold=_safe_info_get(
                        safety_info, "q_rejoin_threshold"
                    ),
                    q_rejoin_index=_safe_info_get(safety_info, "q_rejoin_index"),
                    qd_rejoin_loss=_safe_info_get(safety_info, "qd_rejoin_loss"),
                    qd_rejoin_dist=_safe_info_get(safety_info, "qd_rejoin_dist"),
                    qd_rejoin_threshold=_safe_info_get(
                        safety_info, "qd_rejoin_threshold"
                    ),
                    qd_rejoin_index=_safe_info_get(safety_info, "qd_rejoin_index"),
                    deform_rejoin_available=_safe_info_get(safety_info, "deform_rejoin_available"),
                    deform_rejoin_window_loss=_safe_info_get(safety_info, "deform_rejoin_window_loss"),
                    deform_rejoin_q_loss=_safe_info_get(safety_info, "deform_rejoin_q_loss"),
                    deform_rejoin_qd_loss=_safe_info_get(safety_info, "deform_rejoin_qd_loss"),
                    deform_rejoin_action_loss=_safe_info_get(safety_info, "deform_rejoin_action_loss"),
                    deform_rejoin_heading_loss=_safe_info_get(safety_info, "deform_rejoin_heading_loss"),
                    deform_rejoin_q_dist=_safe_info_get(safety_info, "deform_rejoin_q_dist"),
                    deform_rejoin_qd_dist=_safe_info_get(safety_info, "deform_rejoin_qd_dist"),
                    deform_rejoin_action_dist=_safe_info_get(safety_info, "deform_rejoin_action_dist"),
                    deform_rejoin_heading_cosine=_safe_info_get(safety_info, "deform_rejoin_heading_cosine"),
                    deform_rejoin_best_window_offset=_safe_info_get(safety_info, "deform_rejoin_best_window_offset"),
                    ee_rejoin_loss=_safe_info_get(safety_info, "ee_rejoin_loss"),
                    ee_rejoin_dist=_safe_info_get(safety_info, "ee_rejoin_dist"),
                    ee_rejoin_threshold=_safe_info_get(
                        safety_info, "ee_rejoin_threshold"
                    ),
                    ee_rejoin_index=_safe_info_get(safety_info, "ee_rejoin_index"),
                    ee_final_check_available=_safe_info_get(
                        safety_info, "ee_final_check_available"
                    ),
                    inner_rejoin_metric=_safe_info_get(
                        safety_info, "inner_rejoin_metric"
                    ),
                    final_rejoin_metric=_safe_info_get(
                        safety_info, "final_rejoin_metric"
                    ),
                    rejoin_q_eval_time_ms=_safe_info_get(
                        safety_info, "rejoin_q_eval_time_ms"
                    ),
                    rejoin_qd_eval_time_ms=_safe_info_get(
                        safety_info, "rejoin_qd_eval_time_ms"
                    ),
                    ee_nom_cache_time_ms=_safe_info_get(
                        safety_info, "ee_nom_cache_time_ms"
                    ),
                    ee_final_check_time_ms=_safe_info_get(
                        safety_info, "ee_final_check_time_ms"
                    ),
                    existing_optimization_loss=_safe_info_get(
                        safety_info, "existing_optimization_loss"
                    ),
                    smoothness_loss=_safe_info_get(safety_info, "smoothness_loss"),
                    deform_envelope_loss=_safe_info_get(safety_info, "deform_envelope_loss"),
                    deform_envelope_first_delta=_safe_info_get(safety_info, "deform_envelope_first_delta"),
                    deform_envelope_first_violation=_safe_info_get(safety_info, "deform_envelope_first_violation"),
                    deform_envelope_avoid_rate_loss=_safe_info_get(safety_info, "deform_envelope_avoid_rate_loss"),
                    deform_envelope_return_rate_loss=_safe_info_get(safety_info, "deform_envelope_return_rate_loss"),
                    deform_envelope_max_rate=_safe_info_get(safety_info, "deform_envelope_max_rate"),
                    deform_envelope_terminal_delta=_safe_info_get(safety_info, "deform_envelope_terminal_delta"),
                    deform_envelope_terminal_violation=_safe_info_get(safety_info, "deform_envelope_terminal_violation"),
                    deform_envelope_terminal_loss=_safe_info_get(safety_info, "deform_envelope_terminal_loss"),
                    deform_envelope_acceleration_loss=_safe_info_get(safety_info, "deform_envelope_acceleration_loss"),
                    total_loss=_safe_info_get(safety_info, "total_loss"),
                    fallback_used=_safe_info_get(safety_info, "fallback_used"),
                    act_resume_index=_safe_info_get(safety_info, "act_resume_index"),
                    act_resume_supported=_safe_info_get(
                        safety_info, "act_resume_supported"
                    ),
                    committed_chunk_active=_safe_info_get(safety_info, "committed_chunk_active"),
                    committed_chunk_mode=_safe_info_get(safety_info, "committed_chunk_mode"),
                    committed_chunk_index=_safe_info_get(safety_info, "committed_chunk_index"),
                    committed_chunk_length=_safe_info_get(safety_info, "committed_chunk_length"),
                    committed_rejoin_index=_safe_info_get(safety_info, "committed_rejoin_index"),
                    committed_chunk_started=_safe_info_get(safety_info, "committed_chunk_started"),
                    committed_chunk_completed=_safe_info_get(safety_info, "committed_chunk_completed"),
                    committed_receding_horizon_replan=_safe_info_get(safety_info, "committed_receding_horizon_replan"),
                    committed_receding_horizon_reason=_safe_info_get(safety_info, "committed_receding_horizon_reason"),
                    committed_receding_horizon_prefix_steps=_safe_info_get(safety_info, "committed_receding_horizon_prefix_steps"),
                    committed_receding_horizon_remaining_steps=_safe_info_get(safety_info, "committed_receding_horizon_remaining_steps"),
                    committed_receding_recover_steps=_safe_info_get(safety_info, "committed_receding_recover_steps"),
                    committed_nominal_tube_tracking_applied=_safe_info_get(safety_info, "committed_nominal_tube_tracking_applied"),
                    committed_nominal_tube_tracking_source=_safe_info_get(safety_info, "committed_nominal_tube_tracking_source"),
                    committed_nominal_tube_tracking_target_index=_safe_info_get(safety_info, "committed_nominal_tube_tracking_target_index"),
                    committed_nominal_tube_tracking_sequential_targeting=_safe_info_get(safety_info, "committed_nominal_tube_tracking_sequential_targeting"),
                    committed_nominal_tube_tracking_sequence_step=_safe_info_get(safety_info, "committed_nominal_tube_tracking_sequence_step"),
                    committed_nominal_tube_tracking_desired_target_index=_safe_info_get(safety_info, "committed_nominal_tube_tracking_desired_target_index"),
                    committed_nominal_tube_tracking_sequence_reset=_safe_info_get(safety_info, "committed_nominal_tube_tracking_sequence_reset"),
                    committed_nominal_tube_tracking_sequence_hold_due_to_actual_error=_safe_info_get(safety_info, "committed_nominal_tube_tracking_sequence_hold_due_to_actual_error"),
                    committed_nominal_tube_tracking_q_l2_before=_safe_info_get(safety_info, "committed_nominal_tube_tracking_q_l2_before"),
                    committed_nominal_tube_tracking_q_l2_after=_safe_info_get(safety_info, "committed_nominal_tube_tracking_q_l2_after"),
                    committed_nominal_tube_tracking_q_window_l2_before=_safe_info_get(safety_info, "committed_nominal_tube_tracking_q_window_l2_before"),
                    committed_nominal_tube_tracking_q_window_l2_after=_safe_info_get(safety_info, "committed_nominal_tube_tracking_q_window_l2_after"),
                    committed_nominal_tube_tracking_live_window_min_l2=_safe_info_get(safety_info, "committed_nominal_tube_tracking_live_window_min_l2"),
                    committed_nominal_tube_tracking_live_window_l2_mean=_safe_info_get(safety_info, "committed_nominal_tube_tracking_live_window_l2_mean"),
                    committed_nominal_tube_tracking_live_window_l2_max=_safe_info_get(safety_info, "committed_nominal_tube_tracking_live_window_l2_max"),
                    committed_nominal_tube_tracking_live_window_slot_count=_safe_info_get(safety_info, "committed_nominal_tube_tracking_live_window_slot_count"),
                    committed_nominal_tube_tracking_q_max_abs_before=_safe_info_get(safety_info, "committed_nominal_tube_tracking_q_max_abs_before"),
                    committed_nominal_tube_tracking_selected_resume_window_index=_safe_info_get(safety_info, "committed_nominal_tube_tracking_selected_resume_window_index"),
                    committed_nominal_tube_tracking_selected_resume_score=_safe_info_get(safety_info, "committed_nominal_tube_tracking_selected_resume_score"),
                    committed_nominal_tube_tracking_resume_score_before=_safe_info_get(safety_info, "committed_nominal_tube_tracking_resume_score_before"),
                    committed_nominal_tube_tracking_resume_score_after=_safe_info_get(safety_info, "committed_nominal_tube_tracking_resume_score_after"),
                    committed_nominal_tube_tracking_heading_error_before=_safe_info_get(safety_info, "committed_nominal_tube_tracking_heading_error_before"),
                    committed_nominal_tube_tracking_heading_error_after=_safe_info_get(safety_info, "committed_nominal_tube_tracking_heading_error_after"),
                    committed_nominal_tube_tracking_rollout_solver=_safe_info_get(safety_info, "committed_nominal_tube_tracking_rollout_solver"),
                    committed_nominal_tube_tracking_servo_mode=_safe_info_get(safety_info, "committed_nominal_tube_tracking_servo_mode"),
                    committed_nominal_tube_tracking_servo_scale=_safe_info_get(safety_info, "committed_nominal_tube_tracking_servo_scale"),
                    committed_nominal_tube_tracking_servo_boost_active=_safe_info_get(safety_info, "committed_nominal_tube_tracking_servo_boost_active"),
                    committed_nominal_tube_tracking_servo_adaptive_slowdown_active=_safe_info_get(safety_info, "committed_nominal_tube_tracking_servo_adaptive_slowdown_active"),
                    committed_nominal_tube_tracking_servo_previous_negative_streak=_safe_info_get(safety_info, "committed_nominal_tube_tracking_servo_previous_negative_streak"),
                    committed_nominal_tube_tracking_adaptive_retargeted=_safe_info_get(safety_info, "committed_nominal_tube_tracking_adaptive_retargeted"),
                    committed_nominal_tube_tracking_servo_boost_scale=_safe_info_get(safety_info, "committed_nominal_tube_tracking_servo_boost_scale"),
                    committed_nominal_tube_tracking_response_gain_source=_safe_info_get(safety_info, "committed_nominal_tube_tracking_response_gain_source"),
                    committed_nominal_tube_tracking_response_gain_mean=_safe_info_get(safety_info, "committed_nominal_tube_tracking_response_gain_mean"),
                    committed_nominal_tube_tracking_response_gain_min=_safe_info_get(safety_info, "committed_nominal_tube_tracking_response_gain_min"),
                    committed_nominal_tube_tracking_response_gain_max=_safe_info_get(safety_info, "committed_nominal_tube_tracking_response_gain_max"),
                    committed_nominal_tube_tracking_solver_candidate_count=_safe_info_get(safety_info, "committed_nominal_tube_tracking_solver_candidate_count"),
                    committed_nominal_tube_tracking_best_candidate=_safe_info_get(safety_info, "committed_nominal_tube_tracking_best_candidate"),
                    committed_nominal_tube_tracking_solver_loss=_safe_info_get(safety_info, "committed_nominal_tube_tracking_solver_loss"),
                    committed_nominal_tube_tracking_rollout_predicted_improvement=_safe_info_get(safety_info, "committed_nominal_tube_tracking_rollout_predicted_improvement"),
                    committed_nominal_tube_tracking_actual_improvement=_safe_info_get(safety_info, "committed_nominal_tube_tracking_actual_improvement"),
                    committed_nominal_tube_tracking_negative_actual_improvement_streak=_safe_info_get(safety_info, "committed_nominal_tube_tracking_negative_actual_improvement_streak"),
                    committed_nominal_tube_tracking_max_negative_actual_steps=_safe_info_get(safety_info, "committed_nominal_tube_tracking_max_negative_actual_steps"),
                    committed_nominal_tube_tracking_failed=_safe_info_get(safety_info, "committed_nominal_tube_tracking_failed"),
                    committed_nominal_tube_tracking_min_predicted_improvement=_safe_info_get(safety_info, "committed_nominal_tube_tracking_min_predicted_improvement"),
                    committed_nominal_tube_tracking_retargeted=_safe_info_get(safety_info, "committed_nominal_tube_tracking_retargeted"),
                    committed_nominal_tube_tracking_retarget_count=_safe_info_get(safety_info, "committed_nominal_tube_tracking_retarget_count"),
                    committed_nominal_tube_tracking_action_delta_l2=_safe_info_get(safety_info, "committed_nominal_tube_tracking_action_delta_l2"),
                    committed_nominal_tube_tracking_done_threshold=_safe_info_get(safety_info, "committed_nominal_tube_tracking_done_threshold"),
                    committed_nominal_tube_tracking_ready=_safe_info_get(safety_info, "committed_nominal_tube_tracking_ready"),
                    committed_nominal_tube_tracking_max_recover_steps=_safe_info_get(safety_info, "committed_nominal_tube_tracking_max_recover_steps"),
                    committed_receding_horizon_deferred=_safe_info_get(safety_info, "committed_receding_horizon_deferred"),
                    committed_receding_horizon_defer_reason=_safe_info_get(safety_info, "committed_receding_horizon_defer_reason"),
                    committed_receding_horizon_step_cap_reached=_safe_info_get(safety_info, "committed_receding_horizon_step_cap_reached"),
                    committed_aborted_due_to_safety=_safe_info_get(safety_info, "committed_aborted_due_to_safety"),
                    committed_repaired_step=_safe_info_get(safety_info, "committed_repaired_step"),
                    committed_repair_min_clearance=_safe_info_get(safety_info, "committed_repair_min_clearance"),
                    committed_repair_clearance_gain=_safe_info_get(safety_info, "committed_repair_clearance_gain"),
                    committed_repair_time_ms=_safe_info_get(safety_info, "committed_repair_time_ms"),
                    committed_repair_safety_time_ms=_safe_info_get(safety_info, "committed_repair_safety_time_ms"),
                    committed_action_safety_time_ms=_safe_info_get(safety_info, "committed_action_safety_time_ms"),
                    committed_abort_brake_time_ms=_safe_info_get(safety_info, "committed_abort_brake_time_ms"),
                    recover_steps_executed=_safe_info_get(safety_info, "recover_steps_executed", _safe_info_get(safety_info, "return_steps_executed")),
                    deform_steps_executed=_safe_info_get(safety_info, "deform_steps_executed", _safe_info_get(safety_info, "yield_steps_executed")),
                    resume_from_committed_rejoin=_safe_info_get(safety_info, "resume_from_committed_rejoin"),
                    request_action_history_reset_after_recovery=_safe_info_get(safety_info, "request_action_history_reset_after_recovery"),
                    implicit_action_history_reset_after_intervention=_safe_info_get(safety_info, "implicit_action_history_reset_after_intervention"),
                    recovery_action_history_reset_request_reason=_safe_info_get(safety_info, "recovery_action_history_reset_request_reason"),
                    recovery_action_history_reset=_safe_info_get(safety_info, "recovery_action_history_reset"),
                    recovery_action_history_reset_count=_safe_info_get(safety_info, "recovery_action_history_reset_count"),
                    recovery_action_history_reset_config=_safe_info_get(safety_info, "recovery_action_history_reset_config"),
                    recovery_action_history_reset_reason=_safe_info_get(safety_info, "recovery_action_history_reset_reason"),
                    recovery_low_level_hold_sync=_safe_info_get(safety_info, "recovery_low_level_hold_sync"),
                    recovery_low_level_hold_sync_count=_safe_info_get(safety_info, "recovery_low_level_hold_sync_count"),
                    recovery_visual_history_reset=_safe_info_get(safety_info, "recovery_visual_history_reset"),
                    recovery_visual_history_reset_count=_safe_info_get(safety_info, "recovery_visual_history_reset_count"),
                    recovery_visual_history_seed=_safe_info_get(safety_info, "recovery_visual_history_seed"),
                    recovery_visual_history_seed_count=_safe_info_get(safety_info, "recovery_visual_history_seed_count"),
                    recovery_visual_history_seed_source_count=_safe_info_get(safety_info, "recovery_visual_history_seed_source_count"),
                    recovery_policy_obs_history_seed=_safe_info_get(safety_info, "recovery_policy_obs_history_seed"),
                    recovery_policy_obs_history_seed_count=_safe_info_get(safety_info, "recovery_policy_obs_history_seed_count"),
                    recovery_policy_obs_history_seed_source_count=_safe_info_get(safety_info, "recovery_policy_obs_history_seed_source_count"),
                    post_recovery_act_bridge_started=_safe_info_get(safety_info, "post_recovery_act_bridge_started"),
                    post_recovery_act_bridge_active=_safe_info_get(safety_info, "post_recovery_act_bridge_active"),
                    post_recovery_act_bridge_steps_left=_safe_info_get(safety_info, "post_recovery_act_bridge_steps_left"),
                    post_recovery_act_bridge_total_steps=_safe_info_get(safety_info, "post_recovery_act_bridge_total_steps"),
                    post_recovery_act_bridge_step_index=_safe_info_get(safety_info, "post_recovery_act_bridge_step_index"),
                    post_recovery_act_bridge_last_recovery_step=_safe_info_get(safety_info, "post_recovery_act_bridge_last_recovery_step"),
                    post_recovery_act_bridge_fresh_action_seed_pending=_safe_info_get(safety_info, "post_recovery_act_bridge_fresh_action_seed_pending"),
                    post_recovery_act_bridge_fresh_action_seed_count=_safe_info_get(safety_info, "post_recovery_act_bridge_fresh_action_seed_count"),
                    post_recovery_act_bridge_fresh_action_seed_reset_count=_safe_info_get(safety_info, "post_recovery_act_bridge_fresh_action_seed_reset_count"),
                    post_recovery_act_bridge_fresh_action_seed_source=_safe_info_get(safety_info, "post_recovery_act_bridge_fresh_action_seed_source"),
                    action_bridge_last_recovery_vs_resume_l2=_safe_info_get(safety_info, "action_bridge_last_recovery_vs_resume_l2"),
                    action_bridge_last_recovery_vs_resume_cosine=_safe_info_get(safety_info, "action_bridge_last_recovery_vs_resume_cosine"),
                    action_bridge_last_recovery_arm_l2=_safe_info_get(safety_info, "action_bridge_last_recovery_arm_l2"),
                    action_bridge_last_recovery_gripper_delta=_safe_info_get(safety_info, "action_bridge_last_recovery_gripper_delta"),
                    action_bridge_temporal_history_slot_count=_safe_info_get(safety_info, "action_bridge_temporal_history_slot_count"),
                    action_bridge_temporal_history_vs_resume_l2=_safe_info_get(safety_info, "action_bridge_temporal_history_vs_resume_l2"),
                    action_bridge_resume_first_action_norm=_safe_info_get(safety_info, "action_bridge_resume_first_action_norm"),
                    action_bridge_last_recovery_action_norm=_safe_info_get(safety_info, "action_bridge_last_recovery_action_norm"),
                    committed_abort_step=_safe_info_get(safety_info, "committed_abort_step"),
                    committed_abort_mode=_safe_info_get(safety_info, "committed_abort_mode"),
                    committed_abort_index=_safe_info_get(safety_info, "committed_abort_index"),
                    committed_abort_chunk_length=_safe_info_get(safety_info, "committed_abort_chunk_length"),
                    committed_abort_action=_safe_info_get(safety_info, "committed_abort_action"),
                    committed_abort_min_clearance=_safe_info_get(safety_info, "committed_abort_min_clearance"),
                    committed_abort_required_clearance=_safe_info_get(safety_info, "committed_abort_required_clearance"),
                    committed_abort_clearance_gap=_safe_info_get(safety_info, "committed_abort_clearance_gap"),
                    committed_abort_human_state=_safe_info_get(safety_info, "committed_abort_human_state"),
                    committed_abort_robot_q=_safe_info_get(safety_info, "committed_abort_robot_q"),
                    committed_abort_robot_qd=_safe_info_get(safety_info, "committed_abort_robot_qd"),
                    committed_abort_reason=_safe_info_get(safety_info, "committed_abort_reason"),
                    planned_min_clearance_at_index=_safe_info_get(safety_info, "planned_min_clearance_at_index"),
                    planned_h_at_index=_safe_info_get(safety_info, "planned_h_at_index"),
                    planned_q_at_index=_safe_info_get(safety_info, "planned_q_at_index"),
                    planned_action_at_index=_safe_info_get(safety_info, "planned_action_at_index"),
                    planned_vs_actual_q_error=_safe_info_get(safety_info, "planned_vs_actual_q_error"),
                    planned_vs_actual_action_error=_safe_info_get(safety_info, "planned_vs_actual_action_error"),
                    actual_one_step_clearance=_safe_info_get(safety_info, "actual_one_step_clearance"),
                    planned_clearance_for_this_index=_safe_info_get(safety_info, "planned_clearance_for_this_index"),
                    clearance_prediction_error=_safe_info_get(safety_info, "clearance_prediction_error"),
                    planned_pre_action_q=_safe_info_get(safety_info, "planned_pre_action_q"),
                    planned_post_action_q=_safe_info_get(safety_info, "planned_post_action_q"),
                    predicted_post_action_q=_safe_info_get(safety_info, "predicted_post_action_q"),
                    actual_pre_action_q=_safe_info_get(safety_info, "actual_pre_action_q"),
                    replay_predicted_post_action_q=_safe_info_get(safety_info, "replay_predicted_post_action_q"),
                    committed_action=_safe_info_get(safety_info, "committed_action"),
                    planned_clearance_pre=_safe_info_get(safety_info, "planned_clearance_pre"),
                    planned_clearance_post=_safe_info_get(safety_info, "planned_clearance_post"),
                    replay_clearance_pre=_safe_info_get(safety_info, "replay_clearance_pre"),
                    replay_clearance_post=_safe_info_get(safety_info, "replay_clearance_post"),
                    actual_vs_planned_pre_q_error=_safe_info_get(safety_info, "actual_vs_planned_pre_q_error"),
                    actual_vs_planned_post_q_error=_safe_info_get(safety_info, "actual_vs_planned_post_q_error"),
                    planning_vs_replay_human_error=_safe_info_get(safety_info, "planning_vs_replay_human_error"),
                    planning_vs_replay_clearance_pre_error=_safe_info_get(safety_info, "planning_vs_replay_clearance_pre_error"),
                    planning_vs_replay_clearance_post_error=_safe_info_get(safety_info, "planning_vs_replay_clearance_post_error"),
                    planning_human_state_snapshot=_safe_info_get(safety_info, "planning_human_state_snapshot"),
                    replay_human_state=_safe_info_get(safety_info, "replay_human_state"),
                    control_type=_safe_info_get(safety_info, "control_type"),
                    dt=_safe_info_get(safety_info, "dt"),
                    controlled_state_indices=_safe_info_get(safety_info, "controlled_state_indices"),
                    controlled_action_indices=_safe_info_get(safety_info, "controlled_action_indices"),
                    action_conversion_mode=_safe_info_get(safety_info, "action_conversion_mode"),
                    human_motion_since_plan=_safe_info_get(safety_info, "human_motion_since_plan"),
                    accepted_min_clearance=_safe_info_get(safety_info, "accepted_min_clearance"),
                    accepted_clearance_margin=_safe_info_get(safety_info, "accepted_clearance_margin"),
                    committed_abort_due_to_human_motion=_safe_info_get(safety_info, "committed_abort_due_to_human_motion"),
                    committed_abort_due_to_prediction_error=_safe_info_get(safety_info, "committed_abort_due_to_prediction_error"),
                    committed_abort_due_to_safety_semantics_mismatch=_safe_info_get(safety_info, "committed_abort_due_to_safety_semantics_mismatch"),
                    committed_state_error=_safe_info_get(safety_info, "committed_state_error"),
                    committed_state_error_threshold=_safe_info_get(safety_info, "committed_state_error_threshold"),
                    committed_aborted_due_to_state_mismatch=_safe_info_get(safety_info, "committed_aborted_due_to_state_mismatch"),
                    committed_replan_due_to_state_mismatch=_safe_info_get(safety_info, "committed_replan_due_to_state_mismatch"),
                    committed_rejected_missing_planned_q=_safe_info_get(safety_info, "committed_rejected_missing_planned_q"),
                    committed_state_mismatch_detected=_safe_info_get(safety_info, "committed_state_mismatch_detected"),
                    committed_state_mismatch_recovered=_safe_info_get(safety_info, "committed_state_mismatch_recovered"),
                    committed_suffix_replan_attempted=_safe_info_get(safety_info, "committed_suffix_replan_attempted"),
                    committed_suffix_replan_accepted=_safe_info_get(safety_info, "committed_suffix_replan_accepted"),
                    committed_suffix_replan_rejected=_safe_info_get(safety_info, "committed_suffix_replan_rejected"),
                    committed_suffix_replan_reject_reason=_safe_info_get(safety_info, "committed_suffix_replan_reject_reason"),
                    committed_suffix_replan_from_index=_safe_info_get(safety_info, "committed_suffix_replan_from_index"),
                    committed_suffix_replan_old_length=_safe_info_get(safety_info, "committed_suffix_replan_old_length"),
                    committed_suffix_replan_new_length=_safe_info_get(safety_info, "committed_suffix_replan_new_length"),
                    committed_suffix_replan_target_index=_safe_info_get(safety_info, "committed_suffix_replan_target_index"),
                    committed_suffix_replan_seed_start_index=_safe_info_get(safety_info, "committed_suffix_replan_seed_start_index"),
                    committed_suffix_replan_min_clearance=_safe_info_get(safety_info, "committed_suffix_replan_min_clearance"),
                    committed_suffix_replan_required_clearance=_safe_info_get(safety_info, "committed_suffix_replan_required_clearance"),
                    committed_opportunistic_resume=_safe_info_get(safety_info, "committed_opportunistic_resume"),
                    committed_released_for_act_resume=_safe_info_get(safety_info, "committed_released_for_act_resume"),
                    committed_rejoin_resume_tube_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_score"),
                    committed_rejoin_resume_tube_ok=_safe_info_get(safety_info, "committed_rejoin_resume_tube_ok"),
                    committed_rejoin_resume_tube_min_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_min_score"),
                    committed_rejoin_resume_tube_component_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_component_score"),
                    committed_rejoin_resume_tube_min_component_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_min_component_score"),
                    committed_rejoin_resume_tube_component_ok=_safe_info_get(safety_info, "committed_rejoin_resume_tube_component_ok"),
                    committed_rejoin_resume_tube_terminal_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_terminal_score"),
                    committed_rejoin_resume_tube_path_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_path_score"),
                    committed_rejoin_resume_tube_progress_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_progress_score"),
                    committed_rejoin_resume_tube_heading_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_heading_score"),
                    committed_rejoin_resume_tube_clearance_score=_safe_info_get(safety_info, "committed_rejoin_resume_tube_clearance_score"),
                    committed_rejoin_resume_tube_terminal_dist=_safe_info_get(safety_info, "committed_rejoin_resume_tube_terminal_dist"),
                    committed_rejoin_resume_tube_terminal_delta=_safe_info_get(safety_info, "committed_rejoin_resume_tube_terminal_delta"),
                    committed_rejoin_resume_tube_q_error=_safe_info_get(safety_info, "committed_rejoin_resume_tube_q_error"),
                    committed_rejoin_resume_tube_terminal_threshold=_safe_info_get(safety_info, "committed_rejoin_resume_tube_terminal_threshold"),
                    committed_rejoin_resume_tube_ordered_loss=_safe_info_get(safety_info, "committed_rejoin_resume_tube_ordered_loss"),
                    committed_rejoin_resume_tube_prefix_min_clearance=_safe_info_get(safety_info, "committed_rejoin_resume_tube_prefix_min_clearance"),
                    committed_rejoin_resume_tube_required_clearance=_safe_info_get(safety_info, "committed_rejoin_resume_tube_required_clearance"),
                    committed_rejoin_resume_tube_prefix_safe=_safe_info_get(safety_info, "committed_rejoin_resume_tube_prefix_safe"),
                    committed_rejoin_resume_tube_terminal_ok=_safe_info_get(safety_info, "committed_rejoin_resume_tube_terminal_ok"),
                    committed_rejoin_resume_allowed=_safe_info_get(safety_info, "committed_rejoin_resume_allowed"),
                    committed_rejoin_resume_blocked=_safe_info_get(safety_info, "committed_rejoin_resume_blocked"),
                    committed_rejoin_resume_block_reason=_safe_info_get(safety_info, "committed_rejoin_resume_block_reason"),
                    committed_soft_handoff_release_to_main_filter=_safe_info_get(safety_info, "committed_soft_handoff_release_to_main_filter"),
                    committed_soft_handoff_release_reason=_safe_info_get(safety_info, "committed_soft_handoff_release_reason"),
                    committed_soft_handoff_prefix_min_clearance=_safe_info_get(safety_info, "committed_soft_handoff_prefix_min_clearance"),
                    committed_soft_handoff_required_clearance=_safe_info_get(safety_info, "committed_soft_handoff_required_clearance"),
                    committed_soft_handoff_live_min_clearance=_safe_info_get(safety_info, "committed_soft_handoff_live_min_clearance"),
                    committed_soft_handoff_live_prefix_safe=_safe_info_get(safety_info, "committed_soft_handoff_live_prefix_safe"),
                    committed_soft_handoff_prefix_safe=_safe_info_get(safety_info, "committed_soft_handoff_prefix_safe"),
                    committed_soft_handoff_prefix_ok=_safe_info_get(safety_info, "committed_soft_handoff_prefix_ok"),
                    committed_soft_handoff_resume_tube_score=_safe_info_get(safety_info, "committed_soft_handoff_resume_tube_score"),
                    committed_soft_handoff_resume_tube_ok=_safe_info_get(safety_info, "committed_soft_handoff_resume_tube_ok"),
                    committed_soft_handoff_resume_tube_component_score=_safe_info_get(safety_info, "committed_soft_handoff_resume_tube_component_score"),
                    committed_recovery_budget_exit=_safe_info_get(safety_info, "committed_recovery_budget_exit"),
                    committed_replan_due_to_recovery_budget=_safe_info_get(safety_info, "committed_replan_due_to_recovery_budget"),
                    committed_opportunistic_resume_available=_safe_info_get(safety_info, "committed_opportunistic_resume_available"),
                    committed_opportunistic_resume_reason=_safe_info_get(safety_info, "committed_opportunistic_resume_reason"),
                    committed_opportunistic_resume_q_dist=_safe_info_get(safety_info, "committed_opportunistic_resume_q_dist"),
                    committed_opportunistic_resume_q_threshold=_safe_info_get(safety_info, "committed_opportunistic_resume_q_threshold"),
                    committed_opportunistic_resume_min_clearance=_safe_info_get(safety_info, "committed_opportunistic_resume_min_clearance"),
                    committed_opportunistic_resume_required_clearance=_safe_info_get(safety_info, "committed_opportunistic_resume_required_clearance"),
                    committed_opportunistic_resume_rejoin_index=_safe_info_get(safety_info, "committed_opportunistic_resume_rejoin_index"),
                    committed_opportunistic_resume_tube_ok=_safe_info_get(safety_info, "committed_opportunistic_resume_tube_ok"),
                    committed_opportunistic_resume_tube_score=_safe_info_get(safety_info, "committed_opportunistic_resume_tube_score"),
                    committed_opportunistic_resume_tube_component_score=_safe_info_get(safety_info, "committed_opportunistic_resume_tube_component_score"),
                    committed_opportunistic_resume_affordance_ok=_safe_info_get(safety_info, "committed_opportunistic_resume_affordance_ok"),
                    committed_opportunistic_resume_affordance_score=_safe_info_get(safety_info, "committed_opportunistic_resume_affordance_score"),
                    committed_opportunistic_resume_affordance_component_score=_safe_info_get(safety_info, "committed_opportunistic_resume_affordance_component_score"),
                    committed_recover_steps_since_act=_safe_info_get(safety_info, "committed_recover_steps_since_act"),
                    max_recover_steps_before_act_resume=_safe_info_get(safety_info, "max_recover_steps_before_act_resume"),
                    max_recover_steps_with_progress=_safe_info_get(safety_info, "max_recover_steps_with_progress"),
                    extend_recovery_budget_on_progress=_safe_info_get(safety_info, "extend_recovery_budget_on_progress"),
                    recovery_budget_extended=_safe_info_get(safety_info, "recovery_budget_extended"),
                    recovery_budget_extended_count=_safe_info_get(safety_info, "recovery_budget_extended_count"),
                    recovery_budget_live_q_dist=_safe_info_get(safety_info, "recovery_budget_live_q_dist"),
                    recovery_budget_best_q_dist=_safe_info_get(safety_info, "recovery_budget_best_q_dist"),
                    recovery_budget_progress_delta=_safe_info_get(safety_info, "recovery_budget_progress_delta"),
                    recovery_budget_progress_ok=_safe_info_get(safety_info, "recovery_budget_progress_ok"),
                    recovery_budget_no_progress_count=_safe_info_get(safety_info, "recovery_budget_no_progress_count"),
                    recovery_budget_no_progress_limit=_safe_info_get(safety_info, "recovery_budget_no_progress_limit"),
                    recovery_budget_q_rejoin_index=_safe_info_get(safety_info, "recovery_budget_q_rejoin_index"),
                    staged_recovery_enabled=_safe_info_get(safety_info, "staged_recovery_enabled"),
                    staged_recovery_ordered_path_softened=_safe_info_get(safety_info, "staged_recovery_ordered_path_softened"),
                    staged_recovery_progress_accepted=_safe_info_get(safety_info, "staged_recovery_progress_accepted"),
                    staged_recovery_accept_reason=_safe_info_get(safety_info, "staged_recovery_accept_reason"),
                    staged_recovery_safety_ok=_safe_info_get(safety_info, "staged_recovery_safety_ok"),
                    staged_recovery_progress_ok=_safe_info_get(safety_info, "staged_recovery_progress_ok"),
                    staged_recovery_reject_reason_before_progress=_safe_info_get(safety_info, "staged_recovery_reject_reason_before_progress"),
                    staged_recovery_min_progress_delta=_safe_info_get(safety_info, "staged_recovery_min_progress_delta"),
                    recover_handover_ready=_safe_info_get(safety_info, "recover_handover_ready"),
                    recover_progress_only=_safe_info_get(safety_info, "recover_progress_only"),
                    recovery_handover_pending=_safe_info_get(safety_info, "recovery_handover_pending"),
                    committed_suffix_replans_in_current_recovery=_safe_info_get(safety_info, "committed_suffix_replans_in_current_recovery"),
                    max_suffix_replans_per_recovery=_safe_info_get(safety_info, "max_suffix_replans_per_recovery"),
                    mpc_recovery_enabled=_safe_info_get(safety_info, "mpc_recovery_enabled"),
                    mpc_recovery_active=_safe_info_get(safety_info, "mpc_recovery_active"),
                    mpc_recovery_replan_attempted=_safe_info_get(safety_info, "mpc_recovery_replan_attempted"),
                    mpc_recovery_replan_accepted=_safe_info_get(safety_info, "mpc_recovery_replan_accepted"),
                    mpc_recovery_replan_rejected=_safe_info_get(safety_info, "mpc_recovery_replan_rejected"),
                    mpc_recovery_replan_reject_reason=_safe_info_get(safety_info, "mpc_recovery_replan_reject_reason"),
                    mpc_recovery_reference_index=_safe_info_get(safety_info, "mpc_recovery_reference_index"),
                    mpc_recovery_horizon=_safe_info_get(safety_info, "mpc_recovery_horizon"),
                    mpc_recovery_prefix_len=_safe_info_get(safety_info, "mpc_recovery_prefix_len"),
                    mpc_recovery_replan_count=_safe_info_get(safety_info, "mpc_recovery_replan_count"),
                    mpc_recovery_accepted_count=_safe_info_get(safety_info, "mpc_recovery_accepted_count"),
                    mpc_recovery_rejected_count=_safe_info_get(safety_info, "mpc_recovery_rejected_count"),
                    mpc_recovery_replans_in_current_recovery=_safe_info_get(safety_info, "mpc_recovery_replans_in_current_recovery"),
                    mpc_recovery_max_replans_per_recovery=_safe_info_get(safety_info, "mpc_recovery_max_replans_per_recovery"),
                    mpc_recovery_require_live_progress=_safe_info_get(safety_info, "mpc_recovery_require_live_progress"),
                    mpc_recovery_min_progress_delta=_safe_info_get(safety_info, "mpc_recovery_min_progress_delta"),
                    mpc_recovery_live_q_dist_before=_safe_info_get(safety_info, "mpc_recovery_live_q_dist_before"),
                    mpc_recovery_live_q_dist_after=_safe_info_get(safety_info, "mpc_recovery_live_q_dist_after"),
                    mpc_recovery_live_q_progress_delta=_safe_info_get(safety_info, "mpc_recovery_live_q_progress_delta"),
                    mpc_recovery_live_rejoin_index_before=_safe_info_get(safety_info, "mpc_recovery_live_rejoin_index_before"),
                    mpc_recovery_live_rejoin_index_after=_safe_info_get(safety_info, "mpc_recovery_live_rejoin_index_after"),
                    mpc_recovery_live_progress_ok=_safe_info_get(safety_info, "mpc_recovery_live_progress_ok"),
                    mpc_recovery_prefix_replay_step=_safe_info_get(safety_info, "mpc_recovery_prefix_replay_step"),
                    mpc_recovery_recover_local_index=_safe_info_get(safety_info, "mpc_recovery_recover_local_index"),
                    committed_state_mismatch_ignored_for_mpc_prefix=_safe_info_get(safety_info, "committed_state_mismatch_ignored_for_mpc_prefix"),
                    mpc_recovery_no_progress_count=_safe_info_get(safety_info, "mpc_recovery_no_progress_count"),
                    mpc_recovery_no_progress_limit=_safe_info_get(safety_info, "mpc_recovery_no_progress_limit"),
                    mpc_recovery_no_progress_reject_count=_safe_info_get(safety_info, "mpc_recovery_no_progress_reject_count"),
                    mpc_recovery_budget_escape=_safe_info_get(safety_info, "mpc_recovery_budget_escape"),
                    mpc_recovery_budget_escape_count=_safe_info_get(safety_info, "mpc_recovery_budget_escape_count"),
                    mpc_recovery_target_tube_available=_safe_info_get(safety_info, "mpc_recovery_target_tube_available"),
                    mpc_recovery_target_tube_ok=_safe_info_get(safety_info, "mpc_recovery_target_tube_ok"),
                    mpc_recovery_target_tube_progress_ok=_safe_info_get(safety_info, "mpc_recovery_target_tube_progress_ok"),
                    mpc_recovery_target_tube_loss=_safe_info_get(safety_info, "mpc_recovery_target_tube_loss"),
                    mpc_recovery_target_tube_terminal_loss=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_loss"),
                    mpc_recovery_target_tube_terminal_dist=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_dist"),
                    mpc_recovery_target_tube_min_path_loss=_safe_info_get(safety_info, "mpc_recovery_target_tube_min_path_loss"),
                    mpc_recovery_target_tube_min_path_dist=_safe_info_get(safety_info, "mpc_recovery_target_tube_min_path_dist"),
                    mpc_recovery_target_tube_loss_threshold=_safe_info_get(safety_info, "mpc_recovery_target_tube_loss_threshold"),
                    mpc_recovery_target_tube_dist_threshold=_safe_info_get(safety_info, "mpc_recovery_target_tube_dist_threshold"),
                    mpc_recovery_target_tube_current_local_index=_safe_info_get(safety_info, "mpc_recovery_target_tube_current_local_index"),
                    mpc_recovery_target_tube_terminal_local_index=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_local_index"),
                    mpc_recovery_target_tube_local_index_progress=_safe_info_get(safety_info, "mpc_recovery_target_tube_local_index_progress"),
                    mpc_recovery_target_tube_target_index=_safe_info_get(safety_info, "mpc_recovery_target_tube_target_index"),
                    mpc_recovery_target_tube_heading_cosine=_safe_info_get(safety_info, "mpc_recovery_target_tube_heading_cosine"),
                    mpc_recovery_target_tube_progress_projection=_safe_info_get(safety_info, "mpc_recovery_target_tube_progress_projection"),
                    mpc_recovery_target_tube_target_tangent_norm=_safe_info_get(safety_info, "mpc_recovery_target_tube_target_tangent_norm"),
                    mpc_recovery_target_tube_terminal_delta_norm=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_delta_norm"),
                    mpc_recovery_target_tube_terminal_error_l2=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_error_l2"),
                    mpc_recovery_target_tube_window_len=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_len"),
                    mpc_recovery_target_tube_requested_window_len=_safe_info_get(safety_info, "mpc_recovery_target_tube_requested_window_len"),
                    mpc_recovery_target_tube_window_loss=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_loss"),
                    mpc_recovery_target_tube_window_total_loss=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_total_loss"),
                    mpc_recovery_target_tube_window_dist=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dist"),
                    mpc_recovery_target_tube_window_error_l2=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_error_l2"),
                    mpc_recovery_target_tube_window_dq_loss=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_loss"),
                    mpc_recovery_target_tube_window_dq_dist=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_dist"),
                    mpc_recovery_target_tube_window_action_loss=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_action_loss"),
                    mpc_recovery_target_tube_window_action_dist=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_action_dist"),
                    mpc_recovery_target_tube_window_q_frame_l2=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_q_frame_l2"),
                    mpc_recovery_target_tube_window_q_frame_l2_mean=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_q_frame_l2_mean"),
                    mpc_recovery_target_tube_window_q_frame_l2_max=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_q_frame_l2_max"),
                    mpc_recovery_target_tube_window_wrist_l2=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_wrist_l2"),
                    mpc_recovery_target_tube_window_wrist_l2_mean=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_wrist_l2_mean"),
                    mpc_recovery_target_tube_window_wrist_l2_max=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_wrist_l2_max"),
                    mpc_recovery_target_tube_window_left_wrist_abs=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_left_wrist_abs"),
                    mpc_recovery_target_tube_window_left_wrist_abs_mean=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_left_wrist_abs_mean"),
                    mpc_recovery_target_tube_window_left_wrist_abs_max=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_left_wrist_abs_max"),
                    mpc_recovery_target_tube_window_right_wrist_abs=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_right_wrist_abs"),
                    mpc_recovery_target_tube_window_right_wrist_abs_mean=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_right_wrist_abs_mean"),
                    mpc_recovery_target_tube_window_right_wrist_abs_max=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_right_wrist_abs_max"),
                    mpc_recovery_target_tube_window_recovery_step_l2=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_recovery_step_l2"),
                    mpc_recovery_target_tube_window_target_step_l2=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_target_step_l2"),
                    mpc_recovery_target_tube_window_step_l2_error=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_step_l2_error"),
                    mpc_recovery_target_tube_window_step_l2_error_mean=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_step_l2_error_mean"),
                    mpc_recovery_target_tube_window_step_l2_error_max=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_step_l2_error_max"),
                    mpc_recovery_target_tube_window_dq_error_l2=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_error_l2"),
                    mpc_recovery_target_tube_window_dq_cosine=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_cosine"),
                    mpc_recovery_target_tube_window_dq_cosine_mean=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_cosine_mean"),
                    mpc_recovery_target_tube_window_dq_cosine_min=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_cosine_min"),
                    mpc_recovery_target_tube_window_dq_norm_ratio=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_norm_ratio"),
                    mpc_recovery_target_tube_window_dq_norm_ratio_mean=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_norm_ratio_mean"),
                    mpc_recovery_target_tube_window_dq_norm_ratio_min=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_norm_ratio_min"),
                    mpc_recovery_target_tube_window_start_local_index=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_start_local_index"),
                    mpc_recovery_target_tube_window_end_local_index=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_end_local_index"),
                    mpc_recovery_target_tube_window_weight=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_weight"),
                    mpc_recovery_target_tube_window_dq_weight=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_dq_weight"),
                    mpc_recovery_target_tube_window_action_weight=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_action_weight"),
                    mpc_recovery_target_tube_terminal_delta=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_delta"),
                    mpc_recovery_target_tube_q_error=_safe_info_get(safety_info, "mpc_recovery_target_tube_q_error"),
                    mpc_recovery_target_tube_terminal_q=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_q"),
                    mpc_recovery_target_tube_target_q=_safe_info_get(safety_info, "mpc_recovery_target_tube_target_q"),
                    mpc_recovery_planned_q_seq=_safe_info_get(safety_info, "mpc_recovery_planned_q_seq"),
                    mpc_recovery_planned_action_seq=_safe_info_get(safety_info, "mpc_recovery_planned_action_seq"),
                    mpc_recovery_target_tube_window_q=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_q"),
                    mpc_recovery_target_tube_target_window_q=_safe_info_get(safety_info, "mpc_recovery_target_tube_target_window_q"),
                    mpc_recovery_target_tube_window_action=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_action"),
                    mpc_recovery_target_tube_target_window_action=_safe_info_get(safety_info, "mpc_recovery_target_tube_target_window_action"),
                    mpc_recovery_target_tube_terminal_valid_delta=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_valid_delta"),
                    mpc_recovery_target_tube_terminal_weighted_delta=_safe_info_get(safety_info, "mpc_recovery_target_tube_terminal_weighted_delta"),
                    mpc_recovery_target_tube_state_indices=_safe_info_get(safety_info, "mpc_recovery_target_tube_state_indices"),
                    mpc_recovery_target_tube_state_weights=_safe_info_get(safety_info, "mpc_recovery_target_tube_state_weights"),
                    mpc_recovery_target_tube_target_source=_safe_info_get(safety_info, "mpc_recovery_target_tube_target_source"),
                    mpc_recovery_target_tube_window_start=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_start"),
                    mpc_recovery_target_tube_window_end=_safe_info_get(safety_info, "mpc_recovery_target_tube_window_end"),
                    mpc_recovery_target_tube_live_prefix_safe=_safe_info_get(safety_info, "mpc_recovery_target_tube_live_prefix_safe"),
                    mpc_recovery_target_tube_live_prefix_best_min_clearance=_safe_info_get(safety_info, "mpc_recovery_target_tube_live_prefix_best_min_clearance"),
                    mpc_recovery_q_rejoin_overridden_by_tube=_safe_info_get(safety_info, "mpc_recovery_q_rejoin_overridden_by_tube"),
                    mpc_recovery_original_reject_reason=_safe_info_get(safety_info, "mpc_recovery_original_reject_reason"),
                    mpc_handoff_attempted=_safe_info_get(safety_info, "mpc_handoff_attempted"),
                    mpc_handoff_accepted=_safe_info_get(safety_info, "mpc_handoff_accepted"),
                    mpc_handoff_rejected=_safe_info_get(safety_info, "mpc_handoff_rejected"),
                    mpc_handoff_reject_reason=_safe_info_get(safety_info, "mpc_handoff_reject_reason"),
                    mpc_handoff_reason=_safe_info_get(safety_info, "mpc_handoff_reason"),
                    mpc_handoff_attempt_count=_safe_info_get(safety_info, "mpc_handoff_attempt_count"),
                    mpc_handoff_accept_count=_safe_info_get(safety_info, "mpc_handoff_accept_count"),
                    mpc_handoff_reject_count=_safe_info_get(safety_info, "mpc_handoff_reject_count"),
                    mpc_handoff_scoring_history_snapshot_available=_safe_info_get(safety_info, "mpc_handoff_scoring_history_snapshot_available"),
                    mpc_handoff_scoring_history_mutated=_safe_info_get(safety_info, "mpc_handoff_scoring_history_mutated"),
                    mpc_handoff_scoring_history_mutation_delta=_optional_float(_safe_info_get(safety_info, "mpc_handoff_scoring_history_mutation_delta")),
                    mpc_handoff_scoring_history_before_cur_step=_optional_int(_safe_info_get(safety_info, "mpc_handoff_scoring_history_before_cur_step")),
                    mpc_handoff_scoring_history_after_cur_step=_optional_int(_safe_info_get(safety_info, "mpc_handoff_scoring_history_after_cur_step")),
                    mpc_handoff_act_window_available=_safe_info_get(safety_info, "mpc_handoff_act_window_available"),
                    mpc_handoff_pose_dist=_safe_info_get(safety_info, "mpc_handoff_pose_dist"),
                    mpc_handoff_pose_tube_dist_threshold=_safe_info_get(safety_info, "mpc_handoff_pose_tube_dist_threshold"),
                    mpc_handoff_pose_tube_ok=_safe_info_get(safety_info, "mpc_handoff_pose_tube_ok"),
                    mpc_handoff_actual_direction_available=_safe_info_get(safety_info, "mpc_handoff_actual_direction_available"),
                    mpc_handoff_actual_direction_source=_safe_info_get(safety_info, "mpc_handoff_actual_direction_source"),
                    mpc_handoff_previous_q_available=_safe_info_get(safety_info, "mpc_handoff_previous_q_available"),
                    mpc_handoff_previous_q_adjacent=_safe_info_get(safety_info, "mpc_handoff_previous_q_adjacent"),
                    mpc_handoff_target_source=_safe_info_get(safety_info, "mpc_handoff_target_source"),
                    mpc_handoff_heading_cosine=_safe_info_get(safety_info, "mpc_handoff_heading_cosine"),
                    mpc_handoff_heading_cosine_threshold=_safe_info_get(safety_info, "mpc_handoff_heading_cosine_threshold"),
                    mpc_handoff_heading_ok=_safe_info_get(safety_info, "mpc_handoff_heading_ok"),
                    mpc_handoff_progress_projection=_safe_info_get(safety_info, "mpc_handoff_progress_projection"),
                    mpc_handoff_progress_ok=_safe_info_get(safety_info, "mpc_handoff_progress_ok"),
                    mpc_handoff_release_action_safe=_safe_info_get(safety_info, "mpc_handoff_release_action_safe"),
                    mpc_handoff_bridge_action_source=_safe_info_get(safety_info, "mpc_handoff_bridge_action_source"),
                    mpc_handoff_bridge_target_action_safe=_safe_info_get(safety_info, "mpc_handoff_bridge_target_action_safe"),
                    mpc_handoff_bridge_ramp_enabled=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_enabled"),
                    mpc_handoff_bridge_ramp_max_steps=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_max_steps"),
                    mpc_handoff_bridge_ramp_steps_since_act=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_steps_since_act"),
                    mpc_handoff_bridge_ramp_budget_ok=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_budget_ok"),
                    mpc_handoff_bridge_ramp_reason_ok=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_reason_ok"),
                    mpc_handoff_bridge_ramp_allowed=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_allowed"),
                    mpc_handoff_bridge_ramp_executed=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_executed"),
                    mpc_handoff_bridge_ramp_block_reason=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_block_reason"),
                    mpc_handoff_bridge_ramp_release_to_act=_safe_info_get(safety_info, "mpc_handoff_bridge_ramp_release_to_act"),
                    mpc_handoff_deferred_release_reason=_safe_info_get(safety_info, "mpc_handoff_deferred_release_reason"),
                    mpc_handoff_action_agreement_override_enabled=_safe_info_get(safety_info, "mpc_handoff_action_agreement_override_enabled"),
                    mpc_handoff_action_agreement_source=_safe_info_get(safety_info, "mpc_handoff_action_agreement_source"),
                    mpc_handoff_resume_readiness_required=_safe_info_get(safety_info, "mpc_handoff_resume_readiness_required"),
                    mpc_handoff_resume_allowed=_safe_info_get(safety_info, "mpc_handoff_resume_allowed"),
                    mpc_handoff_resume_block_reason=_safe_info_get(safety_info, "mpc_handoff_resume_block_reason"),
                    mpc_handoff_action_agreement_l2_threshold=_safe_info_get(safety_info, "mpc_handoff_action_agreement_l2_threshold"),
                    mpc_handoff_action_agreement_cosine_threshold=_safe_info_get(safety_info, "mpc_handoff_action_agreement_cosine_threshold"),
                    mpc_handoff_action_agreement_arm_l2_threshold=_safe_info_get(safety_info, "mpc_handoff_action_agreement_arm_l2_threshold"),
                    mpc_handoff_action_agreement_l2_ok=_safe_info_get(safety_info, "mpc_handoff_action_agreement_l2_ok"),
                    mpc_handoff_action_agreement_cosine_ok=_safe_info_get(safety_info, "mpc_handoff_action_agreement_cosine_ok"),
                    mpc_handoff_action_agreement_arm_l2_ok=_safe_info_get(safety_info, "mpc_handoff_action_agreement_arm_l2_ok"),
                    mpc_handoff_action_agreement_ok=_safe_info_get(safety_info, "mpc_handoff_action_agreement_ok"),
                    mpc_handoff_action_agreement_live_ok=_safe_info_get(safety_info, "mpc_handoff_action_agreement_live_ok"),
                    mpc_handoff_action_agreement_override_allowed=_safe_info_get(safety_info, "mpc_handoff_action_agreement_override_allowed"),
                    mpc_handoff_action_agreement_override_reason=_safe_info_get(safety_info, "mpc_handoff_action_agreement_override_reason"),
                    mpc_handoff_heading_ok_raw=_safe_info_get(safety_info, "mpc_handoff_heading_ok_raw"),
                    mpc_handoff_progress_ok_raw=_safe_info_get(safety_info, "mpc_handoff_progress_ok_raw"),
                    mpc_handoff_heading_ok_effective=_safe_info_get(safety_info, "mpc_handoff_heading_ok_effective"),
                    mpc_handoff_progress_ok_effective=_safe_info_get(safety_info, "mpc_handoff_progress_ok_effective"),
                    mpc_handoff_heading_overridden_by_action_agreement=_safe_info_get(safety_info, "mpc_handoff_heading_overridden_by_action_agreement"),
                    mpc_handoff_progress_overridden_by_action_agreement=_safe_info_get(safety_info, "mpc_handoff_progress_overridden_by_action_agreement"),
                    mpc_handoff_act_vs_release_action_l2=_safe_info_get(safety_info, "mpc_handoff_act_vs_release_action_l2"),
                    mpc_handoff_act_vs_release_action_max_abs=_safe_info_get(safety_info, "mpc_handoff_act_vs_release_action_max_abs"),
                    mpc_handoff_act_vs_release_action_cosine=_safe_info_get(safety_info, "mpc_handoff_act_vs_release_action_cosine"),
                    mpc_handoff_act_vs_release_action_dim=_safe_info_get(safety_info, "mpc_handoff_act_vs_release_action_dim"),
                    mpc_handoff_act_vs_release_action_arm_l2=_safe_info_get(safety_info, "mpc_handoff_act_vs_release_action_arm_l2"),
                    mpc_handoff_act_vs_release_action_arm_max_abs=_safe_info_get(safety_info, "mpc_handoff_act_vs_release_action_arm_max_abs"),
                    mpc_handoff_act_vs_release_action_arm_dim=_safe_info_get(safety_info, "mpc_handoff_act_vs_release_action_arm_dim"),
                    mpc_handoff_act_vs_target_action_l2=_safe_info_get(safety_info, "mpc_handoff_act_vs_target_action_l2"),
                    mpc_handoff_act_vs_target_action_max_abs=_safe_info_get(safety_info, "mpc_handoff_act_vs_target_action_max_abs"),
                    mpc_handoff_act_vs_target_action_cosine=_safe_info_get(safety_info, "mpc_handoff_act_vs_target_action_cosine"),
                    mpc_handoff_act_vs_target_action_dim=_safe_info_get(safety_info, "mpc_handoff_act_vs_target_action_dim"),
                    mpc_handoff_act_vs_target_action_arm_l2=_safe_info_get(safety_info, "mpc_handoff_act_vs_target_action_arm_l2"),
                    mpc_handoff_act_vs_target_action_arm_max_abs=_safe_info_get(safety_info, "mpc_handoff_act_vs_target_action_arm_max_abs"),
                    mpc_handoff_act_vs_target_action_arm_dim=_safe_info_get(safety_info, "mpc_handoff_act_vs_target_action_arm_dim"),
                    mpc_handoff_act_prefix_safe=_safe_info_get(safety_info, "mpc_handoff_act_prefix_safe"),
                    mpc_handoff_act_prefix_min_clearance=_optional_float(_safe_info_get(safety_info, "mpc_handoff_act_prefix_min_clearance")),
                    mpc_handoff_shadow_prefix_available=_safe_info_get(safety_info, "mpc_handoff_shadow_prefix_available"),
                    mpc_handoff_shadow_prefix_safe=_safe_info_get(safety_info, "mpc_handoff_shadow_prefix_safe"),
                    mpc_handoff_shadow_prefix_reason=_optional_str(_safe_info_get(safety_info, "mpc_handoff_shadow_prefix_reason")),
                    mpc_handoff_shadow_prefix_len=_optional_int(_safe_info_get(safety_info, "mpc_handoff_shadow_prefix_len")),
                    mpc_handoff_shadow_prefix_target_start=_optional_int(_safe_info_get(safety_info, "mpc_handoff_shadow_prefix_target_start")),
                    mpc_handoff_shadow_prefix_min_clearance=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_prefix_min_clearance")),
                    mpc_handoff_shadow_prefix_required_clearance=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_prefix_required_clearance")),
                    mpc_handoff_shadow_prefix_clearance_margin=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_prefix_clearance_margin")),
                    mpc_handoff_shadow_act_prefix_available=_safe_info_get(safety_info, "mpc_handoff_shadow_act_prefix_available"),
                    mpc_handoff_shadow_act_prefix_safe=_safe_info_get(safety_info, "mpc_handoff_shadow_act_prefix_safe"),
                    mpc_handoff_shadow_act_prefix_min_clearance=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_act_prefix_min_clearance")),
                    mpc_handoff_shadow_act_prefix_clearance_margin=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_act_prefix_clearance_margin")),
                    mpc_handoff_shadow_target_prefix_available=_safe_info_get(safety_info, "mpc_handoff_shadow_target_prefix_available"),
                    mpc_handoff_shadow_target_prefix_safe=_safe_info_get(safety_info, "mpc_handoff_shadow_target_prefix_safe"),
                    mpc_handoff_shadow_target_prefix_min_clearance=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_target_prefix_min_clearance")),
                    mpc_handoff_shadow_target_prefix_clearance_margin=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_target_prefix_clearance_margin")),
                    mpc_handoff_shadow_release_act_prefix_available=_safe_info_get(safety_info, "mpc_handoff_shadow_release_act_prefix_available"),
                    mpc_handoff_shadow_release_act_prefix_safe=_safe_info_get(safety_info, "mpc_handoff_shadow_release_act_prefix_safe"),
                    mpc_handoff_shadow_release_act_prefix_min_clearance=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_release_act_prefix_min_clearance")),
                    mpc_handoff_shadow_release_act_prefix_clearance_margin=_optional_float(_safe_info_get(safety_info, "mpc_handoff_shadow_release_act_prefix_clearance_margin")),
                    mpc_handoff_live_prefix_required=_safe_info_get(safety_info, "mpc_handoff_live_prefix_required"),
                    mpc_handoff_live_prefix_safe=_safe_info_get(safety_info, "mpc_handoff_live_prefix_safe"),
                    mpc_handoff_live_prefix_safe_count=_safe_info_get(safety_info, "mpc_handoff_live_prefix_safe_count"),
                    mpc_handoff_live_prefix_eval_count=_safe_info_get(safety_info, "mpc_handoff_live_prefix_eval_count"),
                    mpc_handoff_live_prefix_best_min_clearance=_safe_info_get(safety_info, "mpc_handoff_live_prefix_best_min_clearance"),
                    mpc_handoff_live_prefix_required_clearance=_safe_info_get(safety_info, "mpc_handoff_live_prefix_required_clearance"),
                    mpc_handoff_live_prefix_best_start=_safe_info_get(safety_info, "mpc_handoff_live_prefix_best_start"),
                    mpc_handoff_rejoin_index=_safe_info_get(safety_info, "mpc_handoff_rejoin_index"),
                    mpc_handoff_resume_tube_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_score"),
                    mpc_handoff_resume_tube_ok=_safe_info_get(safety_info, "mpc_handoff_resume_tube_ok"),
                    mpc_handoff_resume_tube_min_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_min_score"),
                    mpc_handoff_resume_tube_component_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_component_score"),
                    mpc_handoff_resume_tube_min_component_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_min_component_score"),
                    mpc_handoff_resume_tube_component_ok=_safe_info_get(safety_info, "mpc_handoff_resume_tube_component_ok"),
                    mpc_handoff_resume_tube_terminal_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_terminal_score"),
                    mpc_handoff_resume_tube_path_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_path_score"),
                    mpc_handoff_resume_tube_progress_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_progress_score"),
                    mpc_handoff_resume_tube_heading_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_heading_score"),
                    mpc_handoff_resume_tube_clearance_score=_safe_info_get(safety_info, "mpc_handoff_resume_tube_clearance_score"),
                    mpc_handoff_resume_tube_terminal_dist=_safe_info_get(safety_info, "mpc_handoff_resume_tube_terminal_dist"),
                    mpc_handoff_resume_tube_terminal_threshold=_safe_info_get(safety_info, "mpc_handoff_resume_tube_terminal_threshold"),
                    mpc_handoff_resume_tube_ordered_loss=_safe_info_get(safety_info, "mpc_handoff_resume_tube_ordered_loss"),
                    mpc_handoff_resume_tube_prefix_min_clearance=_safe_info_get(safety_info, "mpc_handoff_resume_tube_prefix_min_clearance"),
                    mpc_handoff_resume_tube_required_clearance=_safe_info_get(safety_info, "mpc_handoff_resume_tube_required_clearance"),
                    mpc_handoff_resume_tube_prefix_safe=_safe_info_get(safety_info, "mpc_handoff_resume_tube_prefix_safe"),
                    mpc_handoff_resume_tube_terminal_ok=_safe_info_get(safety_info, "mpc_handoff_resume_tube_terminal_ok"),
                    mpc_bridge_direction_available=_safe_info_get(safety_info, "mpc_bridge_direction_available"),
                    mpc_bridge_direction_ok=_safe_info_get(safety_info, "mpc_bridge_direction_ok"),
                    mpc_bridge_heading_ok=_safe_info_get(safety_info, "mpc_bridge_heading_ok"),
                    mpc_bridge_progress_ok=_safe_info_get(safety_info, "mpc_bridge_progress_ok"),
                    mpc_bridge_heading_cosine=_safe_info_get(safety_info, "mpc_bridge_heading_cosine"),
                    mpc_bridge_progress_projection=_safe_info_get(safety_info, "mpc_bridge_progress_projection"),
                    mpc_bridge_direction_loss=_safe_info_get(safety_info, "mpc_bridge_direction_loss"),
                    mpc_bridge_direction_weight=_safe_info_get(safety_info, "mpc_bridge_direction_weight"),
                    mpc_bridge_weighted_direction_loss=_safe_info_get(safety_info, "mpc_bridge_weighted_direction_loss"),
                    mpc_bridge_live_prefix_available=_safe_info_get(safety_info, "mpc_bridge_live_prefix_available"),
                    mpc_bridge_live_prefix_clearance_ok=_safe_info_get(safety_info, "mpc_bridge_live_prefix_clearance_ok"),
                    mpc_bridge_live_prefix_min_clearance=_safe_info_get(safety_info, "mpc_bridge_live_prefix_min_clearance"),
                    mpc_bridge_live_prefix_required_clearance=_safe_info_get(safety_info, "mpc_bridge_live_prefix_required_clearance"),
                    mpc_bridge_live_prefix_clearance_loss=_safe_info_get(safety_info, "mpc_bridge_live_prefix_clearance_loss"),
                    mpc_bridge_live_prefix_clearance_weight=_safe_info_get(safety_info, "mpc_bridge_live_prefix_clearance_weight"),
                    mpc_bridge_target_source=_safe_info_get(safety_info, "mpc_bridge_target_source"),
                    mpc_bridge_target_live_prefix_safe=_safe_info_get(safety_info, "mpc_bridge_target_live_prefix_safe"),
                    mpc_bridge_target_live_prefix_fallback_selected=_safe_info_get(safety_info, "mpc_bridge_target_live_prefix_fallback_selected"),
                    mpc_bridge_target_live_prefix_min_clearance=_safe_info_get(safety_info, "mpc_bridge_target_live_prefix_min_clearance"),
                    mpc_bridge_target_live_prefix_best_min_clearance=_safe_info_get(safety_info, "mpc_bridge_target_live_prefix_best_min_clearance"),
                    mpc_bridge_target_live_prefix_required_clearance=_safe_info_get(safety_info, "mpc_bridge_target_live_prefix_required_clearance"),
                    mpc_bridge_target_window_start=_safe_info_get(safety_info, "mpc_bridge_target_window_start"),
                    mpc_bridge_replans_in_current_recovery=_safe_info_get(safety_info, "mpc_bridge_replans_in_current_recovery"),
                    mpc_bridge_max_replans_per_recovery=_safe_info_get(safety_info, "mpc_bridge_max_replans_per_recovery"),
                    mpc_bridge_replan_cooldown_remaining=_safe_info_get(safety_info, "mpc_bridge_replan_cooldown_remaining"),
                    mpc_bridge_replan_cooldown_active=_safe_info_get(safety_info, "mpc_bridge_replan_cooldown_active"),
                    mpc_bridge_replan_metric_improved=_safe_info_get(safety_info, "mpc_bridge_replan_metric_improved"),
                    mpc_bridge_replan_under_cap=_safe_info_get(safety_info, "mpc_bridge_replan_under_cap"),
                    mpc_bridge_replan_suppressed_reason=_safe_info_get(safety_info, "mpc_bridge_replan_suppressed_reason"),
                    mpc_bridge_replan_heading_improved=_safe_info_get(safety_info, "mpc_bridge_replan_heading_improved"),
                    mpc_bridge_replan_progress_improved=_safe_info_get(safety_info, "mpc_bridge_replan_progress_improved"),
                    mpc_bridge_replan_clearance_improved=_safe_info_get(safety_info, "mpc_bridge_replan_clearance_improved"),
                    planned_q_at_index_before_suffix_replan=_safe_info_get(safety_info, "planned_q_at_index_before_suffix_replan"),
                    actual_q_at_suffix_replan=_safe_info_get(safety_info, "actual_q_at_suffix_replan"),
                    actual_q_at_replay=_safe_info_get(safety_info, "actual_q_at_replay"),
                    diagnostic_step_mode=diagnostic_flags["diagnostic_step_mode"],
                    mode_transition=mode_transition,
                    act_step=diagnostic_flags["act_step"],
                    deform_step=diagnostic_flags["deform_step"],
                    recover_step=diagnostic_flags["recover_step"],
                    brake_step=diagnostic_flags["brake_step"],
                    fallback_step=diagnostic_flags["fallback_step"],
                    brake_temporal_ensemble_bypass=_safe_info_get(
                        safety_info,
                        "brake_temporal_ensemble_bypass",
                    ),
                    brake_action_history_reset_count=_safe_info_get(
                        safety_info,
                        "brake_action_history_reset_count",
                    ),
                    brake_low_level_hold_sync_count=_safe_info_get(
                        safety_info,
                        "brake_low_level_hold_sync_count",
                    ),
                    brake_robot_freeze_count=_safe_info_get(
                        safety_info,
                        "brake_robot_freeze_count",
                    ),
                    optimized_attempt_step=diagnostic_flags["optimized_attempt_step"],
                    optimized_accepted_step=diagnostic_flags["optimized_accepted_step"],
                    unsafe_streak=_safe_info_get(safety_info, "unsafe_streak"),
                    brake_streak=_safe_info_get(safety_info, "brake_streak"),
                    recovery_failure_streak=_safe_info_get(safety_info, "recovery_failure_streak"),
                    recovery_failure_streak_max=_safe_info_get(safety_info, "recovery_failure_streak_max"),
                    recovery_optimizer_cooldown_remaining=_safe_info_get(safety_info, "recovery_optimizer_cooldown_remaining"),
                    recovery_retry_cooldown_steps=_safe_info_get(safety_info, "recovery_retry_cooldown_steps"),
                    recovery_attempts_in_unsafe_streak=_safe_info_get(safety_info, "recovery_attempts_in_unsafe_streak"),
                    recovery_max_attempts_per_unsafe_streak=_safe_info_get(safety_info, "recovery_max_attempts_per_unsafe_streak"),
                    recovery_optimization_skipped=_safe_info_get(safety_info, "recovery_optimization_skipped"),
                    recovery_optimization_skip_reason=_safe_info_get(safety_info, "recovery_optimization_skip_reason"),
                    recovery_optimization_skipped_count=_safe_info_get(safety_info, "recovery_optimization_skipped_count"),
                    recovery_attempt_reset_after_brake_timeout=_safe_info_get(safety_info, "recovery_attempt_reset_after_brake_timeout"),
                    recovery_attempt_reset_count=_safe_info_get(safety_info, "recovery_attempt_reset_count"),
                    recovery_attempt_reset_reason=_safe_info_get(safety_info, "recovery_attempt_reset_reason"),
                    recovery_attempt_reset_brake_streak=_safe_info_get(safety_info, "recovery_attempt_reset_brake_streak"),
                    recovery_attempt_reset_steps_since_previous=_safe_info_get(safety_info, "recovery_attempt_reset_steps_since_previous"),
                    recovery_attempt_reset_previous_attempts=_safe_info_get(safety_info, "recovery_attempt_reset_previous_attempts"),
                    recovery_attempt_reset_previous_cooldown=_safe_info_get(safety_info, "recovery_attempt_reset_previous_cooldown"),
                    recovery_attempt_reset_hold_clearance=_safe_info_get(safety_info, "recovery_attempt_reset_hold_clearance"),
                    recovery_attempt_reset_min_hold_clearance=_safe_info_get(safety_info, "recovery_attempt_reset_min_hold_clearance"),
                    temporary_blocker_waiting=_safe_info_get(safety_info, "temporary_blocker_waiting"),
                    deform_trigger_reason=_safe_info_get(safety_info, "deform_trigger_reason"),
                    nominal_became_safe_after_brake=_safe_info_get(safety_info, "nominal_became_safe_after_brake"),
                    resume_act_after_wait=_safe_info_get(safety_info, "resume_act_after_wait"),
                    temporary_wait_step=_safe_info_get(safety_info, "temporary_wait_step"),
                    deform_suppressed_by_temporary_wait=_safe_info_get(safety_info, "deform_suppressed_by_temporary_wait"),
                    deform_after_persistent_block=_safe_info_get(safety_info, "deform_after_persistent_block"),
                    deform_replan_count=_safe_info_get(safety_info, "deform_replan_count"),
                    recovery_replan_count=_safe_info_get(safety_info, "recovery_replan_count"),
                    recovery_target_feasible=_safe_info_get(safety_info, "recovery_target_feasible"),
                    stale_recovery_attempted=_safe_info_get(safety_info, "stale_recovery_attempted"),
                    stale_recovery_suppressed_count=_safe_info_get(safety_info, "stale_recovery_suppressed_count"),
                    recovery_target_infeasible_count=_safe_info_get(safety_info, "recovery_target_infeasible_count"),
                    recover_to_task_progress=_safe_info_get(safety_info, "recover_to_task_progress"),
                    recover_anchor_is_current=_safe_info_get(safety_info, "recover_anchor_is_current"),
                    deform_anchor_is_current=_safe_info_get(safety_info, "deform_anchor_is_current"),
                    emergency_brake_steps=_safe_info_get(safety_info, "emergency_brake_steps"),
                    emergency_brake_immediate_unsafe=_safe_info_get(safety_info, "emergency_brake_immediate_unsafe"),
                    optimized_path_count=_safe_info_get(
                        safety_info,
                        "optimized_path_count",
                        _safe_info_get(safety_info, "optimized_candidate_count"),
                    ),
                    optimized_solution_count=_safe_info_get(safety_info, "optimized_solution_count"),
                    fallback_path_count=_safe_info_get(
                        safety_info,
                        "fallback_path_count",
                        _safe_info_get(safety_info, "fallback_candidate_count"),
                    ),
                    fallback_path_accepted_count=_safe_info_get(
                        safety_info,
                        "fallback_path_accepted_count",
                        _safe_info_get(safety_info, "fallback_candidate_accepted_count"),
                    ),
                    path_fallback_enabled=_safe_info_get(
                        safety_info,
                        "path_fallback_enabled",
                        _safe_info_get(safety_info, "candidate_fallback_enabled"),
                    ),
                    optimized_rejected_count=_safe_info_get(safety_info, "optimized_rejected_count"),
                    deform_path_count=_safe_info_get(
                        safety_info,
                        "deform_path_count",
                        _safe_info_get(safety_info, "deform_candidate_count"),
                    ),
                    deform_accepted_count=_safe_info_get(safety_info, "deform_accepted_count"),
                    deform_rejected_count=_safe_info_get(safety_info, "deform_rejected_count"),
                    recover_path_count=_safe_info_get(
                        safety_info,
                        "recover_path_count",
                        _safe_info_get(safety_info, "recover_candidate_count"),
                    ),
                    recover_accepted_count=_safe_info_get(safety_info, "recover_accepted_count"),
                    recover_rejected_count=_safe_info_get(safety_info, "recover_rejected_count"),
                    safe_corridor_recovery_count=_safe_info_get(safety_info, "safe_corridor_recovery_count"),
                    direct_rejoin_attempt_count=_safe_info_get(safety_info, "direct_rejoin_attempt_count"),
                    direct_rejoin_reject_count=_safe_info_get(safety_info, "direct_rejoin_reject_count"),
                    detour_rejoin_attempt_count=_safe_info_get(safety_info, "detour_rejoin_attempt_count"),
                    detour_rejoin_accept_count=_safe_info_get(safety_info, "detour_rejoin_accept_count"),
                    delayed_rejoin_count=_safe_info_get(safety_info, "delayed_rejoin_count"),
                    recover_path_unsafe_count=_safe_info_get(safety_info, "recover_path_unsafe_count"),
                    recovery_path_failure_streak_max=_safe_info_get(safety_info, "recovery_path_failure_streak_max"),
                    repeated_unsafe_target_count=_safe_info_get(safety_info, "repeated_unsafe_target_count"),
                    post_recovery_act_window_count=_safe_info_get(safety_info, "post_recovery_act_window_count"),
                    post_recovery_act_window_interrupted_count=_safe_info_get(safety_info, "post_recovery_act_window_interrupted_count"),
                    mean_recover_path_min_clearance=_safe_info_get(safety_info, "mean_recover_path_min_clearance"),
                    min_recover_path_min_clearance=_safe_info_get(safety_info, "min_recover_path_min_clearance"),
                    safe_prefix_accepted_count=_safe_info_get(safety_info, "safe_prefix_accepted_count"),
                    first_action_only_accepted_count=_safe_info_get(safety_info, "first_action_only_accepted_count"),
                    immediate_hard_reject_count=_safe_info_get(safety_info, "immediate_hard_reject_count"),
                    no_safe_prefix_reject_count=_safe_info_get(safety_info, "no_safe_prefix_reject_count"),
                    horizon_margin_reject_count=_safe_info_get(safety_info, "horizon_margin_reject_count"),
                    accepted_deform_steps=_safe_info_get(safety_info, "accepted_deform_steps"),
                    accepted_recover_steps=_safe_info_get(safety_info, "accepted_recover_steps"),
                    fallback_brake_after_reject_count=_safe_info_get(safety_info, "fallback_brake_after_reject_count"),
                    accepted_path_type=_safe_info_get(
                        safety_info,
                        "accepted_path_type",
                        _safe_info_get(safety_info, "accepted_candidate_type"),
                    ),
                    accepted_path_name=_safe_info_get(
                        safety_info,
                        "accepted_path_name",
                        _safe_info_get(safety_info, "accepted_candidate_name"),
                    ),
                    acceptance_type=_safe_info_get(safety_info, "acceptance_type"),
                    safe_prefix_len=_safe_info_get(safety_info, "safe_prefix_len"),
                    immediate_clearance=_safe_info_get(safety_info, "immediate_clearance"),
                    prefix_min_clearance=_safe_info_get(safety_info, "prefix_min_clearance"),
                    horizon_min_clearance=_safe_info_get(safety_info, "horizon_min_clearance"),
                    full_horizon_required=_safe_info_get(safety_info, "full_horizon_required"),
                    rolling_replan_on_prefix=_safe_info_get(safety_info, "rolling_replan_on_prefix"),
                    safe_prefix_execution=_safe_info_get(safety_info, "safe_prefix_execution"),
                    recover_projection_on_nominal=_safe_info_get(safety_info, "recover_projection_on_nominal"),
                    recover_cosine_to_nominal=_safe_info_get(safety_info, "recover_cosine_to_nominal"),
                    recover_direction_cosine=_safe_info_get(safety_info, "recover_direction_cosine"),
                    recover_direction_cosine_threshold=_safe_info_get(safety_info, "recover_direction_cosine_threshold"),
                    recover_direction_loss=_safe_info_get(safety_info, "recover_direction_loss"),
                    recover_direction_ok=_safe_info_get(safety_info, "recover_direction_ok"),
                    recover_direction_alignment_available=_safe_info_get(safety_info, "recover_direction_alignment_available"),
                    recover_direction_alignment_weight=_safe_info_get(safety_info, "recover_direction_alignment_weight"),
                    recover_act_direction_available=_safe_info_get(safety_info, "recover_act_direction_available"),
                    recover_act_progress_loss=_safe_info_get(safety_info, "recover_act_progress_loss"),
                    recover_act_heading_loss=_safe_info_get(safety_info, "recover_act_heading_loss"),
                    recover_act_direction_loss=_safe_info_get(safety_info, "recover_act_direction_loss"),
                    recover_act_progress_projection=_safe_info_get(safety_info, "recover_act_progress_projection"),
                    recover_act_target_progress=_safe_info_get(safety_info, "recover_act_target_progress"),
                    recover_act_heading_cosine=_safe_info_get(safety_info, "recover_act_heading_cosine"),
                    recover_act_heading_cosine_min=_safe_info_get(safety_info, "recover_act_heading_cosine_min"),
                    recover_act_progress_ok=_safe_info_get(safety_info, "recover_act_progress_ok"),
                    recover_act_heading_ok=_safe_info_get(safety_info, "recover_act_heading_ok"),
                    recover_act_progress_weight=_safe_info_get(safety_info, "recover_act_progress_weight"),
                    recover_act_heading_weight=_safe_info_get(safety_info, "recover_act_heading_weight"),
                    recover_min_act_heading_cosine=_safe_info_get(safety_info, "recover_min_act_heading_cosine"),
                    recover_ordered_path_available=_safe_info_get(safety_info, "recover_ordered_path_available"),
                    recover_ordered_target_index=_safe_info_get(safety_info, "recover_ordered_target_index"),
                    recover_ordered_horizon=_safe_info_get(safety_info, "recover_ordered_horizon"),
                    recover_ordered_pose_loss=_safe_info_get(safety_info, "recover_ordered_pose_loss"),
                    recover_ordered_delta_loss=_safe_info_get(safety_info, "recover_ordered_delta_loss"),
                    recover_ordered_waypoint_pose_loss=_safe_info_get(safety_info, "recover_ordered_waypoint_pose_loss"),
                    recover_ordered_waypoint_rmse=_safe_info_get(safety_info, "recover_ordered_waypoint_rmse"),
                    recover_ordered_heading_loss=_safe_info_get(safety_info, "recover_ordered_heading_loss"),
                    recover_ordered_heading_cosine=_safe_info_get(safety_info, "recover_ordered_heading_cosine"),
                    recover_ordered_heading_cosine_min=_safe_info_get(safety_info, "recover_ordered_heading_cosine_min"),
                    recover_ordered_heading_cosine_threshold=_safe_info_get(safety_info, "recover_ordered_heading_cosine_threshold"),
                    recover_ordered_backtrack_count=_safe_info_get(safety_info, "recover_ordered_backtrack_count"),
                    recover_ordered_monotonic_ok=_safe_info_get(safety_info, "recover_ordered_monotonic_ok"),
                    recover_ordered_pose_tube_threshold=_safe_info_get(safety_info, "recover_ordered_pose_tube_threshold"),
                    recover_ordered_pose_tube_ok=_safe_info_get(safety_info, "recover_ordered_pose_tube_ok"),
                    recover_ordered_waypoint_tube_ok=_safe_info_get(safety_info, "recover_ordered_waypoint_tube_ok"),
                    recover_ordered_strict_ok=_safe_info_get(safety_info, "recover_ordered_strict_ok"),
                    recover_ordered_waypoint_index_start=_safe_info_get(safety_info, "recover_ordered_waypoint_index_start"),
                    recover_ordered_waypoint_index_end=_safe_info_get(safety_info, "recover_ordered_waypoint_index_end"),
                    recover_ordered_loss=_safe_info_get(safety_info, "recover_ordered_loss"),
                    recover_ordered_pose_weight=_safe_info_get(safety_info, "recover_ordered_pose_weight"),
                    recover_ordered_delta_weight=_safe_info_get(safety_info, "recover_ordered_delta_weight"),
                    recover_ordered_pose_threshold=_safe_info_get(safety_info, "recover_ordered_pose_threshold"),
                    recover_ordered_delta_threshold=_safe_info_get(safety_info, "recover_ordered_delta_threshold"),
                    recover_ordered_ok=_safe_info_get(safety_info, "recover_ordered_ok"),
                    nominal_rejoin_score=_safe_info_get(safety_info, "nominal_rejoin_score"),
                    nominal_rejoin_available=_safe_info_get(safety_info, "nominal_rejoin_available"),
                    nominal_rejoin_suppressed_reason=_safe_info_get(safety_info, "nominal_rejoin_suppressed_reason"),
                    nominal_rejoin_clearance=_safe_info_get(safety_info, "nominal_rejoin_clearance"),
                    nominal_rejoin_safe_prefix_len=_safe_info_get(safety_info, "nominal_rejoin_safe_prefix_len"),
                    nominal_rejoin_window_start=_safe_info_get(safety_info, "nominal_rejoin_window_start"),
                    nominal_rejoin_window_end=_safe_info_get(safety_info, "nominal_rejoin_window_end"),
                    nominal_rejoin_window_len=_safe_info_get(safety_info, "nominal_rejoin_window_len"),
                    nominal_rejoin_window_type=_safe_info_get(safety_info, "nominal_rejoin_window_type"),
                    safe_rejoin_window_found=_safe_info_get(safety_info, "safe_rejoin_window_found"),
                    short_staging_window_found=_safe_info_get(safety_info, "short_staging_window_found"),
                    recover_task_progress_score=_safe_info_get(safety_info, "recover_task_progress_score"),
                    recover_score_total=_safe_info_get(safety_info, "recover_score_total"),
                    recover_rejoin_weight_effective=_safe_info_get(safety_info, "recover_rejoin_weight_effective"),
                    recover_step_since_deform=_safe_info_get(safety_info, "recover_step_since_deform"),
                    recover_resume_tube_weight=_safe_info_get(safety_info, "recover_resume_tube_weight"),
                    recover_resume_tube_score=_safe_info_get(safety_info, "recover_resume_tube_score"),
                    recover_resume_tube_ok=_safe_info_get(safety_info, "recover_resume_tube_ok"),
                    recover_resume_tube_min_score=_safe_info_get(safety_info, "recover_resume_tube_min_score"),
                    recover_resume_tube_component_score=_safe_info_get(safety_info, "recover_resume_tube_component_score"),
                    recover_resume_tube_min_component_score=_safe_info_get(safety_info, "recover_resume_tube_min_component_score"),
                    recover_resume_tube_component_ok=_safe_info_get(safety_info, "recover_resume_tube_component_ok"),
                    recover_resume_tube_terminal_score=_safe_info_get(safety_info, "recover_resume_tube_terminal_score"),
                    recover_resume_tube_path_score=_safe_info_get(safety_info, "recover_resume_tube_path_score"),
                    recover_resume_tube_progress_score=_safe_info_get(safety_info, "recover_resume_tube_progress_score"),
                    recover_resume_tube_heading_score=_safe_info_get(safety_info, "recover_resume_tube_heading_score"),
                    recover_resume_tube_clearance_score=_safe_info_get(safety_info, "recover_resume_tube_clearance_score"),
                    recover_resume_tube_terminal_dist=_safe_info_get(safety_info, "recover_resume_tube_terminal_dist"),
                    recover_resume_tube_terminal_delta=_safe_info_get(safety_info, "recover_resume_tube_terminal_delta"),
                    recover_resume_tube_q_error=_safe_info_get(safety_info, "recover_resume_tube_q_error"),
                    recover_resume_tube_terminal_threshold=_safe_info_get(safety_info, "recover_resume_tube_terminal_threshold"),
                    recover_resume_tube_ordered_loss=_safe_info_get(safety_info, "recover_resume_tube_ordered_loss"),
                    recover_resume_tube_prefix_min_clearance=_safe_info_get(safety_info, "recover_resume_tube_prefix_min_clearance"),
                    recover_resume_tube_required_clearance=_safe_info_get(safety_info, "recover_resume_tube_required_clearance"),
                    recover_resume_tube_prefix_safe=_safe_info_get(safety_info, "recover_resume_tube_prefix_safe"),
                    recover_resume_tube_terminal_ok=_safe_info_get(safety_info, "recover_resume_tube_terminal_ok"),
                    recover_resume_window_available=_safe_info_get(safety_info, "recover_resume_window_available"),
                    recover_resume_window_len=_safe_info_get(safety_info, "recover_resume_window_len"),
                    recover_resume_window_requested_len=_safe_info_get(safety_info, "recover_resume_window_requested_len"),
                    recover_resume_window_loss=_safe_info_get(safety_info, "recover_resume_window_loss"),
                    recover_resume_window_total_loss=_safe_info_get(safety_info, "recover_resume_window_total_loss"),
                    recover_resume_window_dist=_safe_info_get(safety_info, "recover_resume_window_dist"),
                    recover_resume_window_error_l2=_safe_info_get(safety_info, "recover_resume_window_error_l2"),
                    recover_resume_window_dq_loss=_safe_info_get(safety_info, "recover_resume_window_dq_loss"),
                    recover_resume_window_dq_dist=_safe_info_get(safety_info, "recover_resume_window_dq_dist"),
                    recover_resume_window_action_loss=_safe_info_get(safety_info, "recover_resume_window_action_loss"),
                    recover_resume_window_action_dist=_safe_info_get(safety_info, "recover_resume_window_action_dist"),
                    recover_resume_window_q_frame_l2=_safe_info_get(safety_info, "recover_resume_window_q_frame_l2"),
                    recover_resume_window_q_frame_l2_mean=_safe_info_get(safety_info, "recover_resume_window_q_frame_l2_mean"),
                    recover_resume_window_q_frame_l2_max=_safe_info_get(safety_info, "recover_resume_window_q_frame_l2_max"),
                    recover_resume_window_wrist_l2=_safe_info_get(safety_info, "recover_resume_window_wrist_l2"),
                    recover_resume_window_wrist_l2_mean=_safe_info_get(safety_info, "recover_resume_window_wrist_l2_mean"),
                    recover_resume_window_wrist_l2_max=_safe_info_get(safety_info, "recover_resume_window_wrist_l2_max"),
                    recover_resume_window_left_wrist_abs=_safe_info_get(safety_info, "recover_resume_window_left_wrist_abs"),
                    recover_resume_window_left_wrist_abs_mean=_safe_info_get(safety_info, "recover_resume_window_left_wrist_abs_mean"),
                    recover_resume_window_left_wrist_abs_max=_safe_info_get(safety_info, "recover_resume_window_left_wrist_abs_max"),
                    recover_resume_window_right_wrist_abs=_safe_info_get(safety_info, "recover_resume_window_right_wrist_abs"),
                    recover_resume_window_right_wrist_abs_mean=_safe_info_get(safety_info, "recover_resume_window_right_wrist_abs_mean"),
                    recover_resume_window_right_wrist_abs_max=_safe_info_get(safety_info, "recover_resume_window_right_wrist_abs_max"),
                    recover_resume_window_recovery_step_l2=_safe_info_get(safety_info, "recover_resume_window_recovery_step_l2"),
                    recover_resume_window_target_step_l2=_safe_info_get(safety_info, "recover_resume_window_target_step_l2"),
                    recover_resume_window_step_l2_error=_safe_info_get(safety_info, "recover_resume_window_step_l2_error"),
                    recover_resume_window_step_l2_error_mean=_safe_info_get(safety_info, "recover_resume_window_step_l2_error_mean"),
                    recover_resume_window_step_l2_error_max=_safe_info_get(safety_info, "recover_resume_window_step_l2_error_max"),
                    recover_resume_window_dq_error_l2=_safe_info_get(safety_info, "recover_resume_window_dq_error_l2"),
                    recover_resume_window_dq_cosine=_safe_info_get(safety_info, "recover_resume_window_dq_cosine"),
                    recover_resume_window_dq_cosine_mean=_safe_info_get(safety_info, "recover_resume_window_dq_cosine_mean"),
                    recover_resume_window_dq_cosine_min=_safe_info_get(safety_info, "recover_resume_window_dq_cosine_min"),
                    recover_resume_window_dq_norm_ratio=_safe_info_get(safety_info, "recover_resume_window_dq_norm_ratio"),
                    recover_resume_window_dq_norm_ratio_mean=_safe_info_get(safety_info, "recover_resume_window_dq_norm_ratio_mean"),
                    recover_resume_window_dq_norm_ratio_min=_safe_info_get(safety_info, "recover_resume_window_dq_norm_ratio_min"),
                    recover_resume_window_start_local_index=_safe_info_get(safety_info, "recover_resume_window_start_local_index"),
                    recover_resume_window_end_local_index=_safe_info_get(safety_info, "recover_resume_window_end_local_index"),
                    recover_resume_window_weight=_safe_info_get(safety_info, "recover_resume_window_weight"),
                    recover_resume_window_dq_weight=_safe_info_get(safety_info, "recover_resume_window_dq_weight"),
                    recover_resume_window_action_weight=_safe_info_get(safety_info, "recover_resume_window_action_weight"),
                    recover_resume_window_q=_safe_info_get(safety_info, "recover_resume_window_q"),
                    recover_resume_window_target_q=_safe_info_get(safety_info, "recover_resume_window_target_q"),
                    recover_resume_window_action=_safe_info_get(safety_info, "recover_resume_window_action"),
                    recover_resume_window_target_action=_safe_info_get(safety_info, "recover_resume_window_target_action"),
                    recover_resume_affordance_weight=_safe_info_get(safety_info, "recover_resume_affordance_weight"),
                    recover_resume_affordance_bonus=_safe_info_get(safety_info, "recover_resume_affordance_bonus"),
                    recover_resume_affordance_loss=_safe_info_get(safety_info, "recover_resume_affordance_loss"),
                    recover_resume_affordance_score_gap=_safe_info_get(safety_info, "recover_resume_affordance_score_gap"),
                    recover_resume_affordance_component_gap=_safe_info_get(safety_info, "recover_resume_affordance_component_gap"),
                    recover_resume_affordance_enabled=_safe_info_get(safety_info, "recover_resume_affordance_enabled"),
                    recover_resume_affordance_available=_safe_info_get(safety_info, "recover_resume_affordance_available"),
                    recover_resume_affordance_task_relevant=_safe_info_get(safety_info, "recover_resume_affordance_task_relevant"),
                    recover_resume_affordance_score=_safe_info_get(safety_info, "recover_resume_affordance_score"),
                    recover_resume_affordance_ok=_safe_info_get(safety_info, "recover_resume_affordance_ok"),
                    recover_resume_affordance_min_score=_safe_info_get(safety_info, "recover_resume_affordance_min_score"),
                    recover_resume_affordance_component_score=_safe_info_get(safety_info, "recover_resume_affordance_component_score"),
                    recover_resume_affordance_min_component_score=_safe_info_get(safety_info, "recover_resume_affordance_min_component_score"),
                    recover_resume_affordance_target_distance=_safe_info_get(safety_info, "recover_resume_affordance_target_distance"),
                    recover_resume_affordance_target_distance_score=_safe_info_get(safety_info, "recover_resume_affordance_target_distance_score"),
                    recover_resume_affordance_contact_score=_safe_info_get(safety_info, "recover_resume_affordance_contact_score"),
                    recover_resume_affordance_progress_score=_safe_info_get(safety_info, "recover_resume_affordance_progress_score"),
                    recover_resume_affordance_alignment_score=_safe_info_get(safety_info, "recover_resume_affordance_alignment_score"),
                    recover_resume_affordance_continuity_score=_safe_info_get(safety_info, "recover_resume_affordance_continuity_score"),
                    recover_resume_affordance_safety_score=_safe_info_get(safety_info, "recover_resume_affordance_safety_score"),
                    recover_resume_affordance_interaction_context=_safe_info_get(safety_info, "recover_resume_affordance_interaction_context"),
                    recover_final_resume_gate_checked=_safe_info_get(safety_info, "recover_final_resume_gate_checked"),
                    recover_final_resume_gate_allowed=_safe_info_get(safety_info, "recover_final_resume_gate_allowed"),
                    recover_final_resume_gate_rejected=_safe_info_get(safety_info, "recover_final_resume_gate_rejected"),
                    recover_final_resume_gate_reject_reason=_safe_info_get(safety_info, "recover_final_resume_gate_reject_reason"),
                    recover_final_resume_gate_affordance_available=_safe_info_get(safety_info, "recover_final_resume_gate_affordance_available"),
                    recover_final_resume_gate_affordance_task_relevant=_safe_info_get(safety_info, "recover_final_resume_gate_affordance_task_relevant"),
                    recover_final_resume_gate_affordance_ok=_safe_info_get(safety_info, "recover_final_resume_gate_affordance_ok"),
                    recover_final_resume_gate_affordance_score=_safe_info_get(safety_info, "recover_final_resume_gate_affordance_score"),
                    recover_final_resume_gate_affordance_component_score=_safe_info_get(safety_info, "recover_final_resume_gate_affordance_component_score"),
                    recover_final_resume_gate_affordance_component_threshold=_safe_info_get(safety_info, "recover_final_resume_gate_affordance_component_threshold"),
                    recover_final_resume_gate_affordance_target_distance=_safe_info_get(safety_info, "recover_final_resume_gate_affordance_target_distance"),
                    recover_final_resume_gate_window_q_frame_l2_mean=_safe_info_get(safety_info, "recover_final_resume_gate_window_q_frame_l2_mean"),
                    recover_final_resume_gate_window_q_frame_l2_mean_threshold=_safe_info_get(safety_info, "recover_final_resume_gate_window_q_frame_l2_mean_threshold"),
                    recover_final_resume_gate_window_q_frame_l2_max=_safe_info_get(safety_info, "recover_final_resume_gate_window_q_frame_l2_max"),
                    recover_final_resume_gate_window_q_frame_l2_max_threshold=_safe_info_get(safety_info, "recover_final_resume_gate_window_q_frame_l2_max_threshold"),
                    recover_final_resume_gate_window_dq_cosine_min=_safe_info_get(safety_info, "recover_final_resume_gate_window_dq_cosine_min"),
                    recover_final_resume_gate_window_dq_cosine_threshold=_safe_info_get(safety_info, "recover_final_resume_gate_window_dq_cosine_threshold"),
                    recover_final_resume_gate_window_step_l2_error_max=_safe_info_get(safety_info, "recover_final_resume_gate_window_step_l2_error_max"),
                    recover_final_resume_gate_window_step_l2_error_threshold=_safe_info_get(safety_info, "recover_final_resume_gate_window_step_l2_error_threshold"),
                    mpc_handoff_resume_affordance_score=_safe_info_get(safety_info, "mpc_handoff_resume_affordance_score"),
                    mpc_handoff_resume_affordance_ok=_safe_info_get(safety_info, "mpc_handoff_resume_affordance_ok"),
                    mpc_handoff_resume_affordance_component_score=_safe_info_get(safety_info, "mpc_handoff_resume_affordance_component_score"),
                    mpc_handoff_resume_affordance_target_distance=_safe_info_get(safety_info, "mpc_handoff_resume_affordance_target_distance"),
                    mpc_handoff_resume_affordance_contact_score=_safe_info_get(safety_info, "mpc_handoff_resume_affordance_contact_score"),
                    mpc_handoff_resume_affordance_interaction_context=_safe_info_get(safety_info, "mpc_handoff_interaction_context"),
                    committed_rejoin_resume_affordance_score=_safe_info_get(safety_info, "committed_rejoin_resume_affordance_score"),
                    committed_rejoin_resume_affordance_ok=_safe_info_get(safety_info, "committed_rejoin_resume_affordance_ok"),
                    committed_rejoin_resume_affordance_component_score=_safe_info_get(safety_info, "committed_rejoin_resume_affordance_component_score"),
                    committed_rejoin_resume_affordance_target_distance=_safe_info_get(safety_info, "committed_rejoin_resume_affordance_target_distance"),
                    committed_rejoin_resume_affordance_contact_score=_safe_info_get(safety_info, "committed_rejoin_resume_affordance_contact_score"),
                    committed_rejoin_resume_affordance_interaction_context=_safe_info_get(safety_info, "committed_rejoin_interaction_context"),
                    committed_soft_handoff_resume_affordance_score=_safe_info_get(safety_info, "committed_soft_handoff_resume_affordance_score"),
                    committed_soft_handoff_resume_affordance_ok=_safe_info_get(safety_info, "committed_soft_handoff_resume_affordance_ok"),
                    committed_soft_handoff_resume_affordance_component_score=_safe_info_get(safety_info, "committed_soft_handoff_resume_affordance_component_score"),
                    recover_batch_fixed_cost_time_ms=_safe_info_get(safety_info, "recover_batch_fixed_cost_time_ms"),
                    recover_batch_target_rollout_time_ms=_safe_info_get(safety_info, "recover_batch_target_rollout_time_ms"),
                    recover_batch_progress_score_time_ms=_safe_info_get(safety_info, "recover_batch_progress_score_time_ms"),
                    recover_batch_rejoin_score_time_ms=_safe_info_get(safety_info, "recover_batch_rejoin_score_time_ms"),
                    recover_batch_ordered_terms_time_ms=_safe_info_get(safety_info, "recover_batch_ordered_terms_time_ms"),
                    recover_batch_loss_build_time_ms=_safe_info_get(safety_info, "recover_batch_loss_build_time_ms"),
                    recover_batch_total_time_ms=_safe_info_get(safety_info, "recover_batch_total_time_ms"),
                    optimizer_method=_safe_info_get(safety_info, "optimizer_method"),
                    gradient_iterations_run=_safe_info_get(safety_info, "gradient_iterations_run"),
                    gradient_early_stopped=_safe_info_get(safety_info, "gradient_early_stopped"),
                    gradient_max_iters=_safe_info_get(safety_info, "gradient_max_iters"),
                    gradient_samples=_safe_info_get(safety_info, "gradient_samples"),
                    gradient_eps=_safe_info_get(safety_info, "gradient_eps"),
                    gradient_path_early_stopped=_safe_info_get(
                        safety_info,
                        "gradient_path_early_stopped",
                        _safe_info_get(safety_info, "gradient_candidate_early_stopped"),
                    ),
                    gradient_batched_line_search=_safe_info_get(safety_info, "gradient_batched_line_search"),
                    gradient_line_search_batch_evaluations=_safe_info_get(safety_info, "gradient_line_search_batch_evaluations"),
                    gradient_line_search_batch_size=_safe_info_get(safety_info, "gradient_line_search_batch_size"),
                    gradient_jax_scan_used=_safe_info_get(safety_info, "gradient_jax_scan_used"),
                    gradient_jax_scan_used_count=_safe_info_get(safety_info, "gradient_jax_scan_used_count"),
                    gradient_full_jax_scan_used=_safe_info_get(safety_info, "gradient_full_jax_scan_used"),
                    gradient_full_jax_scan_time_ms=_safe_info_get(safety_info, "gradient_full_jax_scan_time_ms"),
                    fixed_shape_jax_optimizer_loop=_safe_info_get(safety_info, "fixed_shape_jax_optimizer_loop"),
                    jax_scan_cost_kind=_safe_info_get(safety_info, "jax_scan_cost_kind"),
                    optimizer_evaluations=_safe_info_get(safety_info, "optimizer_evaluations"),
                    gradient_initial_records_time_ms=_safe_info_get(safety_info, "gradient_initial_records_time_ms"),
                    deform_gradient_initial_records_time_ms=_safe_info_get(safety_info, "deform_gradient_initial_records_time_ms"),
                    return_gradient_initial_records_time_ms=_safe_info_get(safety_info, "return_gradient_initial_records_time_ms"),
                    deform_gradient_initial_batch_cost_time_ms=_safe_info_get(safety_info, "deform_gradient_initial_batch_cost_time_ms"),
                    return_gradient_initial_batch_cost_time_ms=_safe_info_get(safety_info, "return_gradient_initial_batch_cost_time_ms"),
                    gradient_initial_project_time_ms=_safe_info_get(safety_info, "gradient_initial_project_time_ms"),
                    gradient_initial_batch_cost_time_ms=_safe_info_get(safety_info, "gradient_initial_batch_cost_time_ms"),
                    gradient_initial_record_build_time_ms=_safe_info_get(safety_info, "gradient_initial_record_build_time_ms"),
                    gradient_initial_sort_time_ms=_safe_info_get(safety_info, "gradient_initial_sort_time_ms"),
                    gradient_direction_sample_time_ms=_safe_info_get(safety_info, "gradient_direction_sample_time_ms"),
                    gradient_perturb_control_time_ms=_safe_info_get(safety_info, "gradient_perturb_control_time_ms"),
                    gradient_perturb_project_time_ms=_safe_info_get(safety_info, "gradient_perturb_project_time_ms"),
                    gradient_perturb_records_time_ms=_safe_info_get(safety_info, "gradient_perturb_records_time_ms"),
                    gradient_line_control_time_ms=_safe_info_get(safety_info, "gradient_line_control_time_ms"),
                    gradient_line_project_time_ms=_safe_info_get(safety_info, "gradient_line_project_time_ms"),
                    gradient_line_records_time_ms=_safe_info_get(safety_info, "gradient_line_records_time_ms"),
                    deform_optimizer_time_ms=_safe_info_get(safety_info, "deform_optimizer_time_ms"),
                    return_optimizer_time_ms=_safe_info_get(safety_info, "return_optimizer_time_ms"),
                    explicit_optimizer_time_ms=_safe_info_get(safety_info, "explicit_optimizer_time_ms"),
                    precomputed_step0_filter_used=_safe_info_get(safety_info, "precomputed_step0_filter_used"),
                    precomputed_step0_filter_time_ms=_safe_info_get(safety_info, "precomputed_step0_filter_time_ms"),
                    committed_suffix_optimizer_time_ms=_safe_info_get(safety_info, "committed_suffix_optimizer_time_ms"),
                    committed_plan_rollout_time_ms=_safe_info_get(safety_info, "committed_plan_rollout_time_ms"),
                    committed_plan_safety_time_ms=_safe_info_get(safety_info, "committed_plan_safety_time_ms"),
                    committed_plan_diagnostics_time_ms=_safe_info_get(safety_info, "committed_plan_diagnostics_time_ms"),
                    fixed_shape_jax_cost=_safe_info_get(safety_info, "fixed_shape_jax_cost"),
                    fixed_shape_jax_safety=_safe_info_get(safety_info, "fixed_shape_jax_safety"),
                    fixed_shape_safety_method=_safe_info_get(safety_info, "fixed_shape_safety_method"),
                    fixed_shape_cost_batch_size=_safe_info_get(safety_info, "fixed_shape_cost_batch_size"),
                    fixed_shape_cost_original_batch_size=_safe_info_get(safety_info, "fixed_shape_cost_original_batch_size"),
                    fixed_shape_cost_padding=_safe_info_get(safety_info, "fixed_shape_cost_padding"),
                    fixed_shape_rollout_time_ms=_safe_info_get(safety_info, "fixed_shape_rollout_time_ms"),
                    fixed_shape_safety_time_ms=_safe_info_get(safety_info, "fixed_shape_safety_time_ms"),
                    fixed_shape_jax_reduce_time_ms=_safe_info_get(safety_info, "fixed_shape_jax_reduce_time_ms"),
                    cem_iterations_run=_safe_info_get(safety_info, "cem_iterations_run"),
                    cem_early_stopped=_safe_info_get(safety_info, "cem_early_stopped"),
                    cem_max_iters=_safe_info_get(safety_info, "cem_max_iters"),
                    cem_population=_safe_info_get(safety_info, "cem_population"),
                    deform_cem_iterations_run=_safe_info_get(safety_info, "deform_cem_iterations_run"),
                    deform_cem_early_stopped=_safe_info_get(safety_info, "deform_cem_early_stopped"),
                    return_cem_iterations_run=_safe_info_get(safety_info, "return_cem_iterations_run"),
                    return_cem_early_stopped=_safe_info_get(safety_info, "return_cem_early_stopped"),
                    nominal_rejoin_available_count=_safe_info_get(safety_info, "nominal_rejoin_available_count"),
                    nominal_rejoin_suppressed_count=_safe_info_get(safety_info, "nominal_rejoin_suppressed_count"),
                    stale_nominal_rejoin_suppressed_count=_safe_info_get(safety_info, "stale_nominal_rejoin_suppressed_count"),
                    nominal_prefix_unsafe_suppressed_count=_safe_info_get(safety_info, "nominal_prefix_unsafe_suppressed_count"),
                    recover_positive_projection_count=_safe_info_get(safety_info, "recover_positive_projection_count"),
                    recover_nonpositive_projection_count=_safe_info_get(safety_info, "recover_nonpositive_projection_count"),
                    mean_recover_projection_on_nominal=_safe_info_get(safety_info, "mean_recover_projection_on_nominal"),
                    mean_recover_cosine_to_nominal=_safe_info_get(safety_info, "mean_recover_cosine_to_nominal"),
                    mean_recover_direction_cosine=_safe_info_get(safety_info, "mean_recover_direction_cosine"),
                    mean_recover_direction_loss=_safe_info_get(safety_info, "mean_recover_direction_loss"),
                    mean_recover_act_progress_loss=_safe_info_get(safety_info, "mean_recover_act_progress_loss"),
                    mean_recover_act_heading_loss=_safe_info_get(safety_info, "mean_recover_act_heading_loss"),
                    mean_recover_act_direction_loss=_safe_info_get(safety_info, "mean_recover_act_direction_loss"),
                    mean_recover_act_progress_projection=_safe_info_get(safety_info, "mean_recover_act_progress_projection"),
                    mean_recover_act_target_progress=_safe_info_get(safety_info, "mean_recover_act_target_progress"),
                    mean_recover_act_heading_cosine=_safe_info_get(safety_info, "mean_recover_act_heading_cosine"),
                    min_recover_act_heading_cosine=_safe_info_get(safety_info, "min_recover_act_heading_cosine"),
                    recover_act_progress_ok_count=_safe_info_get(safety_info, "recover_act_progress_ok_count"),
                    recover_act_heading_ok_count=_safe_info_get(safety_info, "recover_act_heading_ok_count"),
                    mean_recover_task_progress_score=_safe_info_get(safety_info, "mean_recover_task_progress_score"),
                    mean_recover_ordered_pose_loss=_safe_info_get(safety_info, "mean_recover_ordered_pose_loss"),
                    mean_recover_ordered_delta_loss=_safe_info_get(safety_info, "mean_recover_ordered_delta_loss"),
                    mean_recover_ordered_loss=_safe_info_get(safety_info, "mean_recover_ordered_loss"),
                    contact_during_hold=_safe_info_get(safety_info, "contact_during_hold"),
                    contact_during_brake=_safe_info_get(safety_info, "contact_during_brake"),
                    contact_during_deform=_safe_info_get(safety_info, "contact_during_deform"),
                    contact_during_recover=_safe_info_get(safety_info, "contact_during_recover"),
                    chosen_action_norm=_safe_info_get(safety_info, "chosen_action_norm"),
                    controlled_action_delta_norm=_safe_info_get(safety_info, "controlled_action_delta_norm"),
                    arm_delta_norm=_safe_info_get(safety_info, "arm_delta_norm"),
                    gripper_latched=_safe_info_get(safety_info, "gripper_latched"),

                    contact_rich_state=_safe_info_get(safety_info, "contact_rich_state"),
                    contact_rich_pause_enabled=_safe_info_get(safety_info, "contact_rich_pause_enabled"),
                    contact_rich_pause_active=_safe_info_get(safety_info, "contact_rich_pause_active"),
                    contact_rich_pause_signal=_safe_info_get(safety_info, "contact_rich_pause_signal"),
                    contact_rich_pause_raw_signal=_safe_info_get(safety_info, "contact_rich_pause_raw_signal"),
                    contact_rich_pause_reason=_safe_info_get(safety_info, "contact_rich_pause_reason"),
                    contact_rich_pause_metadata_signal=_safe_info_get(safety_info, "contact_rich_pause_metadata_signal"),
                    contact_rich_pause_gripper_signal=_safe_info_get(safety_info, "contact_rich_pause_gripper_signal"),
                    contact_rich_pause_gripper_value=_safe_info_get(safety_info, "contact_rich_pause_gripper_value"),
                    contact_rich_pause_progress_allowed=_safe_info_get(safety_info, "contact_rich_pause_progress_allowed"),
                    contact_rich_pause_hold_remaining=_safe_info_get(safety_info, "contact_rich_pause_hold_remaining"),
                    contact_rich_pause_clear_steps=_safe_info_get(safety_info, "contact_rich_pause_clear_steps"),
                    contact_rich_pause_only=_safe_info_get(safety_info, "contact_rich_pause_only"),
                    deformation_blocked_by_contact_rich_state=_safe_info_get(safety_info, "deformation_blocked_by_contact_rich_state"),
                    gripper_latch_dim=_safe_info_get(safety_info, "gripper_latch_dim"),
                    safe_gripper_action=_safe_info_get(safety_info, "safe_gripper_action"),
                    raw_gripper_action=_safe_info_get(safety_info, "raw_gripper_action"),
                    phase_reanchor_steps_left=_safe_info_get(safety_info, "phase_reanchor_steps_left"),
                    phase_reanchor_phase=_safe_info_get(safety_info, "phase_reanchor_phase"),
                    phase_reanchor_base_cmd_xy=_safe_info_get(safety_info, "phase_reanchor_base_cmd_xy"),
                    phase_reanchor_base_cmd_normalized_xy=_safe_info_get(safety_info, "phase_reanchor_base_cmd_normalized_xy"),
                    phase_reanchor_base_cmd_effective_raw_xy=_safe_info_get(safety_info, "phase_reanchor_base_cmd_effective_raw_xy"),
                    phase_reanchor_base_cmd_clip_delta_norm=_safe_info_get(safety_info, "phase_reanchor_base_cmd_clip_delta_norm"),
                    phase_reanchor_temporal_ensemble_bypass=_safe_info_get(safety_info, "phase_reanchor_temporal_ensemble_bypass"),
                    phase_reanchor_temporal_ensemble_bypass_count=_safe_info_get(safety_info, "phase_reanchor_temporal_ensemble_bypass_count"),
                    phase_reanchor_ee_error_xy=_safe_info_get(safety_info, "phase_reanchor_ee_error_xy"),
                    phase_reanchor_drawer_fraction=_safe_info_get(safety_info, "phase_reanchor_drawer_fraction"),
                    phase_reanchor_ee_to_handle_dist=_safe_info_get(safety_info, "phase_reanchor_ee_to_handle_dist"),
                    phase_reanchor_ee_to_target_dist=_safe_info_get(safety_info, "phase_reanchor_ee_to_target_dist"),
                    phase_reanchor_arm_hold_enabled=_safe_info_get(safety_info, "phase_reanchor_arm_hold_enabled"),
                    phase_reanchor_arm_hold_reason=_safe_info_get(safety_info, "phase_reanchor_arm_hold_reason"),
                    phase_reanchor_arm_servo_enabled=_safe_info_get(safety_info, "phase_reanchor_arm_servo_enabled"),
                    phase_reanchor_arm_servo_reason=_safe_info_get(safety_info, "phase_reanchor_arm_servo_reason"),
                    phase_reanchor_arm_servo_rank=_safe_info_get(safety_info, "phase_reanchor_arm_servo_rank"),
                    phase_reanchor_arm_servo_error_norm=_safe_info_get(safety_info, "phase_reanchor_arm_servo_error_norm"),
                    phase_reanchor_arm_servo_delta_norm=_safe_info_get(safety_info, "phase_reanchor_arm_servo_delta_norm"),
                    phase_reanchor_arm_servo_command_norm=_safe_info_get(safety_info, "phase_reanchor_arm_servo_command_norm"),
                    phase_reanchor_arm_servo_action_delta_norm=_safe_info_get(safety_info, "phase_reanchor_arm_servo_action_delta_norm"),
                    phase_reanchor_arm_servo_target_source=_safe_info_get(safety_info, "phase_reanchor_arm_servo_target_source"),
                    phase_reanchor_arm_servo_target_episode=_safe_info_get(safety_info, "phase_reanchor_arm_servo_target_episode"),
                    phase_reanchor_arm_servo_target_start_step=_safe_info_get(safety_info, "phase_reanchor_arm_servo_target_start_step"),
                    phase_reanchor_arm_servo_target_step=_safe_info_get(safety_info, "phase_reanchor_arm_servo_target_step"),
                    phase_reanchor_arm_servo_target_window_index=_safe_info_get(safety_info, "phase_reanchor_arm_servo_target_window_index"),
                    phase_reanchor_arm_servo_target_window_score=_safe_info_get(safety_info, "phase_reanchor_arm_servo_target_window_score"),
                    phase_reanchor_nominal_reentry_selection_reason=_safe_info_get(safety_info, "phase_reanchor_nominal_reentry_selection_reason"),
                    phase_reanchor_nominal_reentry_live_target_distance=_safe_info_get(safety_info, "phase_reanchor_nominal_reentry_live_target_distance"),
                    phase_reanchor_live_taskspace_guard_active=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_guard_active"),
                    phase_reanchor_live_taskspace_suppress_q_servo=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_suppress_q_servo"),
                    phase_reanchor_live_taskspace_suppress_q_servo_reason=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_suppress_q_servo_reason"),
                    phase_reanchor_live_taskspace_distance=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_distance"),
                    phase_reanchor_live_taskspace_distance_source=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_distance_source"),
                    phase_reanchor_live_taskspace_best_distance=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_best_distance"),
                    phase_reanchor_live_taskspace_worsen_count=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_worsen_count"),
                    phase_reanchor_live_taskspace_stop_requested=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_stop_requested"),
                    phase_reanchor_live_taskspace_stop_reason=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_stop_reason"),
                    phase_reanchor_live_taskspace_elapsed_steps=_safe_info_get(safety_info, "phase_reanchor_live_taskspace_elapsed_steps"),
                    phase_reanchor_live_release_ready=_safe_info_get(safety_info, "phase_reanchor_live_release_ready"),
                    phase_reanchor_live_release_reason=_safe_info_get(safety_info, "phase_reanchor_live_release_reason"),
                    phase_reanchor_live_release_target_error=_safe_info_get(safety_info, "phase_reanchor_live_release_target_error"),
                    phase_reanchor_live_release_handle_dist=_safe_info_get(safety_info, "phase_reanchor_live_release_handle_dist"),
                    phase_reanchor_task_point_source=_safe_info_get(safety_info, "phase_reanchor_task_point_source"),
                    phase_reanchor_task_point_requested_source=_safe_info_get(safety_info, "phase_reanchor_task_point_requested_source"),
                    phase_reanchor_task_point_fallback_reason=_safe_info_get(safety_info, "phase_reanchor_task_point_fallback_reason"),
                    phase_reanchor_control_task_point_source=_safe_info_get(safety_info, "phase_reanchor_control_task_point_source"),
                    phase_reanchor_control_task_point_requested_source=_safe_info_get(safety_info, "phase_reanchor_control_task_point_requested_source"),
                    phase_reanchor_control_task_point_fallback_reason=_safe_info_get(safety_info, "phase_reanchor_control_task_point_fallback_reason"),
                    phase_reanchor_control_ee_to_handle_dist=_safe_info_get(safety_info, "phase_reanchor_control_ee_to_handle_dist"),
                    phase_reanchor_control_ee_to_target_dist=_safe_info_get(safety_info, "phase_reanchor_control_ee_to_target_dist"),
                    phase_reanchor_control_error_source=_safe_info_get(safety_info, "phase_reanchor_control_error_source"),
                    phase_reanchor_handle_assist_enabled=_safe_info_get(safety_info, "phase_reanchor_handle_assist_enabled"),
                    phase_reanchor_handle_assist_reason=_safe_info_get(safety_info, "phase_reanchor_handle_assist_reason"),
                    phase_reanchor_handle_assist_error_norm=_safe_info_get(safety_info, "phase_reanchor_handle_assist_error_norm"),
                    phase_reanchor_handle_assist_base_cmd_xy=_safe_info_get(safety_info, "phase_reanchor_handle_assist_base_cmd_xy"),
                    phase_reanchor_site_ee_to_handle_dist=_safe_info_get(safety_info, "phase_reanchor_site_ee_to_handle_dist"),
                    phase_reanchor_site_ee_to_target_dist=_safe_info_get(safety_info, "phase_reanchor_site_ee_to_target_dist"),
                    phase_reanchor_gripper_to_handle_dist=_safe_info_get(safety_info, "phase_reanchor_gripper_to_handle_dist"),
                    phase_reanchor_gripper_to_target_dist=_safe_info_get(safety_info, "phase_reanchor_gripper_to_target_dist"),
                    phase_reanchor_gripper_site_xy_error=_safe_info_get(safety_info, "phase_reanchor_gripper_site_xy_error"),
                    phase_reanchor_task_point_geometry_untrusted=_safe_info_get(safety_info, "phase_reanchor_task_point_geometry_untrusted"),
                    phase_reanchor_live_extension_started=_safe_info_get(safety_info, "phase_reanchor_live_extension_started"),
                    phase_reanchor_live_extension_count=_safe_info_get(safety_info, "phase_reanchor_live_extension_count"),
                    phase_reanchor_live_extension_budget_exhausted=_safe_info_get(safety_info, "phase_reanchor_live_extension_budget_exhausted"),
                    phase_reanchor_early_release_triggered=_safe_info_get(safety_info, "phase_reanchor_early_release_triggered"),
                    phase_reanchor_early_release_reason=_safe_info_get(safety_info, "phase_reanchor_early_release_reason"),
                    phase_reanchor_early_release_arm_q_error=_safe_info_get(safety_info, "phase_reanchor_early_release_arm_q_error"),
                    phase_reanchor_bridge_contact_ready=_safe_info_get(safety_info, "phase_reanchor_bridge_contact_ready"),
                    phase_reanchor_bridge_contact_reason=_safe_info_get(safety_info, "phase_reanchor_bridge_contact_reason"),
                    phase_reanchor_bridge_contact_handle_dist=_safe_info_get(safety_info, "phase_reanchor_bridge_contact_handle_dist"),
                    phase_reanchor_bridge_contact_handle_limit=_safe_info_get(safety_info, "phase_reanchor_bridge_contact_handle_limit"),
                    phase_reanchor_bridge_preload_validated=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_validated"),
                    phase_reanchor_bridge_preload_reason=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_reason"),
                    phase_reanchor_bridge_preload_steps=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_steps"),
                    phase_reanchor_bridge_preload_progress_delta=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_progress_delta"),
                    phase_reanchor_bridge_preload_progress_abs=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_progress_abs"),
                    phase_reanchor_bridge_preload_handle_dist=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_handle_dist"),
                    phase_reanchor_bridge_preload_handle_limit=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_handle_limit"),
                    phase_reanchor_bridge_preload_progress_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_progress_ok"),
                    phase_reanchor_bridge_preload_handle_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_handle_ok"),
                    phase_reanchor_bridge_preload_validation_source=_safe_info_get(safety_info, "phase_reanchor_bridge_preload_validation_source"),
                    phase_reanchor_preload_gripper_forced=_safe_info_get(safety_info, "phase_reanchor_preload_gripper_forced"),
                    phase_reanchor_preload_gripper_limit=_safe_info_get(safety_info, "phase_reanchor_preload_gripper_limit"),
                    phase_reanchor_preload_target_grasp=_safe_info_get(safety_info, "phase_reanchor_preload_target_grasp"),
                    phase_reanchor_preload_grasp_limit=_safe_info_get(safety_info, "phase_reanchor_preload_grasp_limit"),
                    phase_reanchor_preload_pull_probe_enabled=_safe_info_get(safety_info, "phase_reanchor_preload_pull_probe_enabled"),
                    phase_reanchor_preload_pull_probe_reason=_safe_info_get(safety_info, "phase_reanchor_preload_pull_probe_reason"),
                    phase_reanchor_preload_pull_probe_axis_xy=_safe_info_get(safety_info, "phase_reanchor_preload_pull_probe_axis_xy"),
                    phase_reanchor_preload_pull_probe_step=_safe_info_get(safety_info, "phase_reanchor_preload_pull_probe_step"),
                    phase_reanchor_preload_pull_probe_delta_norm=_safe_info_get(safety_info, "phase_reanchor_preload_pull_probe_delta_norm"),
                    phase_reanchor_early_release_act_grace=_safe_info_get(safety_info, "phase_reanchor_early_release_act_grace"),
                    phase_reanchor_early_release_act_grace_steps=_safe_info_get(safety_info, "phase_reanchor_early_release_act_grace_steps"),
                    phase_reanchor_live_ee_servo_enabled=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_enabled"),
                    phase_reanchor_live_ee_servo_reason=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_reason"),
                    phase_reanchor_live_ee_servo_error_norm=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_error_norm"),
                    phase_reanchor_live_ee_servo_delta_norm=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_delta_norm"),
                    phase_reanchor_live_ee_servo_command_norm=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_command_norm"),
                    phase_reanchor_live_ee_servo_jacobian_rank=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_jacobian_rank"),
                    phase_reanchor_live_ee_servo_nominal_reg=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_nominal_reg"),
                    phase_reanchor_live_ee_servo_fk_site_xy_error=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_fk_site_xy_error"),
                    phase_reanchor_live_ee_servo_fk_gripper_xy_error=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_fk_gripper_xy_error"),
                    phase_reanchor_live_ee_servo_predicted_error_before=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_predicted_error_before"),
                    phase_reanchor_live_ee_servo_predicted_error_after=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_predicted_error_after"),
                    phase_reanchor_live_ee_servo_geometry_untrusted=_safe_info_get(safety_info, "phase_reanchor_live_ee_servo_geometry_untrusted"),
                    phase_reanchor_suppressed_q_servo_arm_hold=_safe_info_get(safety_info, "phase_reanchor_suppressed_q_servo_arm_hold"),
                    phase_reanchor_bridge_history_seed=_safe_info_get(safety_info, "phase_reanchor_bridge_history_seed"),
                    phase_reanchor_bridge_seed_mode=_safe_info_get(safety_info, "phase_reanchor_bridge_seed_mode"),
                    phase_reanchor_bridge_seed_reason=_safe_info_get(safety_info, "phase_reanchor_bridge_seed_reason"),
                    phase_reanchor_bridge_seed_blockers=_safe_info_get(safety_info, "phase_reanchor_bridge_seed_blockers"),
                    phase_reanchor_bridge_nominal_history_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_history_ok"),
                    phase_reanchor_bridge_nominal_action_window_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_action_window_ok"),
                    phase_reanchor_bridge_live_taskspace_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_live_taskspace_ok"),
                    phase_reanchor_bridge_policy_step_before=_safe_info_get(safety_info, "phase_reanchor_bridge_policy_step_before"),
                    phase_reanchor_bridge_policy_step_after=_safe_info_get(safety_info, "phase_reanchor_bridge_policy_step_after"),
                    phase_reanchor_bridge_policy_step_source=_safe_info_get(safety_info, "phase_reanchor_bridge_policy_step_source"),
                    phase_reanchor_bridge_post_seed_act_vs_nominal_l2=_safe_info_get(safety_info, "phase_reanchor_bridge_post_seed_act_vs_nominal_l2"),
                    phase_reanchor_bridge_post_seed_act_vs_nominal_cosine=_safe_info_get(safety_info, "phase_reanchor_bridge_post_seed_act_vs_nominal_cosine"),
                    phase_reanchor_bridge_post_seed_action_seed_count=_safe_info_get(safety_info, "phase_reanchor_bridge_post_seed_action_seed_count"),
                    phase_reanchor_bridge_post_seed_action_agreement_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_post_seed_action_agreement_ok"),
                    phase_reanchor_bridge_requires_post_seed_action_agreement=_safe_info_get(safety_info, "phase_reanchor_bridge_requires_post_seed_action_agreement"),
                    phase_reanchor_bridge_temporal_stats_source=_safe_info_get(safety_info, "phase_reanchor_bridge_temporal_stats_source"),
                    phase_reanchor_bridge_visual_seed_count=_safe_info_get(safety_info, "phase_reanchor_bridge_visual_seed_count"),
                    phase_reanchor_bridge_visual_seed_source_count=_safe_info_get(safety_info, "phase_reanchor_bridge_visual_seed_source_count"),
                    phase_reanchor_bridge_frame_stack_seed_count=_safe_info_get(safety_info, "phase_reanchor_bridge_frame_stack_seed_count"),
                    phase_reanchor_bridge_obs_seed_source=_safe_info_get(safety_info, "phase_reanchor_bridge_obs_seed_source"),
                    phase_reanchor_bridge_obs_seed_window_count=_safe_info_get(safety_info, "phase_reanchor_bridge_obs_seed_window_count"),
                    phase_reanchor_bridge_obs_seed_restore_count=_safe_info_get(safety_info, "phase_reanchor_bridge_obs_seed_restore_count"),
                    phase_reanchor_bridge_action_seed_count=_safe_info_get(safety_info, "phase_reanchor_bridge_action_seed_count"),
                    phase_reanchor_bridge_action_seed_source=_safe_info_get(safety_info, "phase_reanchor_bridge_action_seed_source"),
                    phase_reanchor_bridge_action_window_len=_safe_info_get(safety_info, "phase_reanchor_bridge_action_window_len"),
                    phase_reanchor_bridge_act_vs_nominal_l2=_safe_info_get(safety_info, "phase_reanchor_bridge_act_vs_nominal_l2"),
                    phase_reanchor_bridge_act_vs_nominal_cosine=_safe_info_get(safety_info, "phase_reanchor_bridge_act_vs_nominal_cosine"),
                    phase_reanchor_bridge_action_agreement_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_action_agreement_ok"),
                    phase_reanchor_bridge_action_base_adapted=_safe_info_get(safety_info, "phase_reanchor_bridge_action_base_adapted"),
                    phase_reanchor_bridge_action_base_adapted_dims=_safe_info_get(safety_info, "phase_reanchor_bridge_action_base_adapted_dims"),
                    phase_reanchor_bridge_nominal_q_l2=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_l2"),
                    phase_reanchor_bridge_nominal_q_window_l2_mean=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_window_l2_mean"),
                    phase_reanchor_bridge_nominal_q_window_l2_max=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_window_l2_max"),
                    phase_reanchor_bridge_nominal_q_window_len=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_window_len"),
                    phase_reanchor_bridge_nominal_q_adapted_l2=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_adapted_l2"),
                    phase_reanchor_bridge_nominal_q_adapted_window_l2_mean=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_adapted_window_l2_mean"),
                    phase_reanchor_bridge_nominal_q_adapted_window_l2_max=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_adapted_window_l2_max"),
                    phase_reanchor_bridge_nominal_q_adapted_dims=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_adapted_dims"),
                    phase_reanchor_bridge_nominal_q_base_l2=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_base_l2"),
                    phase_reanchor_bridge_nominal_q_arm_l2=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_arm_l2"),
                    phase_reanchor_bridge_nominal_q_track_base=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_track_base"),
                    phase_reanchor_bridge_nominal_q_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_nominal_q_ok"),
                    phase_reanchor_bridge_readiness_ok=_safe_info_get(safety_info, "phase_reanchor_bridge_readiness_ok"),
                    post_recovery_task_guard_active=_safe_info_get(safety_info, "post_recovery_task_guard_active"),
                    post_recovery_task_guard_steps_left=_safe_info_get(safety_info, "post_recovery_task_guard_steps_left"),
                    post_recovery_task_guard_reason=_safe_info_get(safety_info, "post_recovery_task_guard_reason"),
                    post_recovery_task_guard_best_progress=_safe_info_get(safety_info, "post_recovery_task_guard_best_progress"),
                    post_recovery_progress_regression=_safe_info_get(safety_info, "post_recovery_progress_regression"),
                    post_recovery_reanchor_started=_safe_info_get(safety_info, "post_recovery_reanchor_started"),
                    post_recovery_no_progress_count=_safe_info_get(safety_info, "post_recovery_no_progress_count"),
                    post_recovery_mid_progress_no_progress_count=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_no_progress_count",
                    ),
                    post_recovery_mid_progress_best_progress=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_best_progress",
                    ),
                    post_recovery_mid_progress_best_distance=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_best_distance",
                    ),
                    post_recovery_mid_progress_distance_regression=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_distance_regression",
                    ),
                    post_recovery_mid_progress_reseed_triggered=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_reseed_triggered",
                    ),
                    post_recovery_mid_progress_reseed_reset_count=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_reseed_reset_count",
                    ),
                    post_recovery_mid_progress_reseed_reason=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_reseed_reason",
                    ),
                    post_recovery_mid_progress_prior_action_seed_count=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_prior_action_seed_count",
                    ),
                    post_recovery_mid_progress_prior_action_seed_step=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_prior_action_seed_step",
                    ),
                    post_recovery_mid_progress_prior_action_seed_age=_safe_info_get(
                        safety_info,
                        "post_recovery_mid_progress_prior_action_seed_age",
                    ),
                    post_recovery_no_progress_triggered=_safe_info_get(safety_info, "post_recovery_no_progress_triggered"),
                    post_recovery_no_progress_target_distance=_safe_info_get(safety_info, "post_recovery_no_progress_target_distance"),
                    post_recovery_no_progress_distance_source=_safe_info_get(safety_info, "post_recovery_no_progress_distance_source"),
                    hold_immediate_clearance=_safe_info_get(safety_info, "hold_immediate_clearance"),
                    hold_horizon_min_clearance=_safe_info_get(safety_info, "hold_horizon_min_clearance"),
                    hold_acceptance_type=_safe_info_get(safety_info, "hold_acceptance_type"),
                    hold_rejected_reason=_safe_info_get(safety_info, "hold_rejected_reason"),
                    hold_predicted_contact=_safe_info_get(safety_info, "hold_predicted_contact"),
                    human_prediction_available=_safe_info_get(safety_info, "human_prediction_available"),
                    human_velocity_toward_robot=_safe_info_get(safety_info, "human_velocity_toward_robot"),
                    human_motion_prediction_enabled=_safe_info_get(safety_info, "human_motion_prediction_enabled"),
                    human_motion_prediction_available=_safe_info_get(safety_info, "human_motion_prediction_available"),
                    human_motion_prediction_speed=_safe_info_get(safety_info, "human_motion_prediction_speed"),
                    human_motion_prediction_max_displacement=_safe_info_get(safety_info, "human_motion_prediction_max_displacement"),
                    emergency_deform_away=_safe_info_get(safety_info, "emergency_deform_away"),
                    emergency_deform_away_steps=_safe_info_get(safety_info, "emergency_deform_away_steps"),
                    emergency_deform_away_count=_safe_info_get(safety_info, "emergency_deform_away_count"),
                    hold_unsafe_count=_safe_info_get(safety_info, "hold_unsafe_count"),
                    hold_predicted_contact_count=_safe_info_get(safety_info, "hold_predicted_contact_count"),
                    contact_during_hold_count=_safe_info_get(safety_info, "contact_during_hold_count"),
                    contact_during_brake_count=_safe_info_get(safety_info, "contact_during_brake_count"),
                    contact_during_deform_count=_safe_info_get(safety_info, "contact_during_deform_count"),
                    contact_during_recover_count=_safe_info_get(safety_info, "contact_during_recover_count"),
                    mean_hold_horizon_min_clearance=_safe_info_get(safety_info, "mean_hold_horizon_min_clearance"),
                    min_hold_horizon_min_clearance=_safe_info_get(safety_info, "min_hold_horizon_min_clearance"),

                    elapsed_wall_time_s=float(elapsed_wall_time_s),
                    step_wall_time_s=float(step_wall_time_s),

                    filter_time_ms=float(filter_time_ms),
                    monitor_time_ms=float(monitor_time_ms),
                    env_step_time_ms=float(env_step_time_ms),
                    policy_obs_adapt_time_ms=float(policy_obs_adapt_time_ms),
                    policy_action_time_ms=float(policy_action_time_ms),
                    policy_obs_update_time_ms=float(policy_obs_update_time_ms),
                )

                episode_metrics.append(step_metrics)
                all_step_metrics.append(step_metrics)

                video_duration_s = _video_duration_seconds(video_recorder)
                video_recorded_steps = _video_recorded_steps(video_recorder)
                video_left_s = None
                video_left_steps = None
                if args.record_video and args.video_time_base == "sim" and video_stop_steps is not None:
                    video_left_steps = max(0, video_stop_steps - video_recorded_steps)
                elif args.record_video and args.stop_video_at_seconds is not None:
                    video_left_s = max(0.0, args.stop_video_at_seconds - video_duration_s)

                if progress_bar is not None:
                    progress_bar.update(1)
                    postfix = {"steps_left": args.steps - progress_bar.n}
                    if video_left_steps is not None:
                        postfix["video_steps_left"] = video_left_steps
                    elif video_left_s is not None:
                        postfix["video_left"] = f"{video_left_s:.1f}s"
                    progress_bar.set_postfix(postfix)
                elif args.debug:
                    print(
                        f"ep={episode:03d} step={step:04d} "
                        f"reward={float(reward):.3f} "
                        f"min_h={min_h} "
                        f"arm_delta={arm_delta:.5f} "
                        f"non_arm_delta={non_arm_delta:.5f} "
                        f"gripper_latched={gripper_latched} "
                        f"contact_count={contact_count} "
                        f"filter_ms={filter_time_ms:.2f}"
                    )

                if args.video_time_base == "sim":
                    reached_video_limit = (
                        args.record_video
                        and video_stop_steps is not None
                        and video_recorded_steps >= video_stop_steps
                    )
                else:
                    reached_video_limit = (
                        args.record_video
                        and args.stop_video_at_seconds is not None
                        and video_duration_s >= args.stop_video_at_seconds
                    )
                if reached_video_limit:
                    if args.video_time_base == "sim":
                        episode_stop_reason = f"video_step_limit:{video_recorded_steps}"
                        print(
                            f"Stopping episode {episode} at "
                            f"{video_recorded_steps} recorded env steps "
                            f"(target {video_stop_steps})."
                        )
                    else:
                        episode_stop_reason = f"video_wall_limit:{video_duration_s:.3f}"
                        print(
                            f"Stopping episode {episode} at "
                            f"{video_duration_s:.1f}s of recorded video "
                            f"(target {args.stop_video_at_seconds:.1f}s)."
                        )
                    break
                if unmodelled_contact_reason is not None:
                    episode_stop_reason = unmodelled_contact_reason
                    if episode == 0 or args.debug:
                        print(
                            f"Stopping episode {episode} at step {step}: "
                            f"{unmodelled_contact_reason}"
                        )
                    break
                if terminated or truncated or (success and not args.continue_after_success):
                    if terminated:
                        episode_stop_reason = "terminated"
                    elif truncated:
                        episode_stop_reason = "truncated"
                    elif success and not args.continue_after_success:
                        episode_stop_reason = "success"
                    break

            saved_action_episodes.append(
                np.asarray(saved_episode_actions, dtype=np.float32)
            )

            episode_summary = summarise_chunk_episode(
                episode_metrics,
                diagnostics_cfg={
                    **_safety_filter_section(args, "diagnostics"),
                    "success_threshold": args.phase_reanchor_done_threshold,
                },
            )
            if episode_metrics:
                wall_time_s = episode_metrics[-1].elapsed_wall_time_s
                step_wall_times = np.asarray(
                    [m.step_wall_time_s for m in episode_metrics],
                    dtype=np.float32,
                )
                episode_summary["wall_time_s"] = float(wall_time_s)
                episode_summary["mean_step_wall_time_s"] = float(
                    np.mean(step_wall_times)
                )
                episode_summary["steps_per_wall_second"] = float(
                    len(episode_metrics) / max(wall_time_s, 1e-9)
                )
                video_recorded_duration_s = float(_video_duration_seconds(video_recorder))
                episode_summary["video_recorded_duration_s"] = video_recorded_duration_s
                episode_summary["video_recorded_wall_time_s"] = video_recorded_duration_s
                episode_summary["video_recorded_steps"] = int(
                    _video_recorded_steps(video_recorder)
                )
            episode_summary["stop_reason"] = episode_stop_reason
            episode_summary["video_stop_steps"] = video_stop_steps
            episode_summary["normalization_source"] = normalization_source
            episode_summary["robot_spawn"] = robot_spawn_info
            episode_summary["policy_robot_spawn"] = policy_robot_spawn_info
            episode_summary["safety_robot_spawn"] = safety_robot_spawn_info
            episode_summary["execution_length"] = workspace_cfg.get("execution_length", None)
            episode_summary["action_sequence"] = workspace_cfg.get("action_sequence", None)
            episode_summary["video_time_base"] = args.video_time_base
            if args.save_frame_images:
                episode_frame_prefix = f"{args.condition}_episode_{episode:03d}_"
                episode_frames = [
                    path for path in saved_frame_image_paths
                    if Path(path).name.startswith(episode_frame_prefix)
                ]
                episode_summary["frame_image_dir"] = str(frame_image_dir)
                episode_summary["frame_image_every"] = int(args.frame_image_every)
                episode_summary["frame_image_count"] = int(len(episode_frames))
                episode_summary["frame_image_paths"] = episode_frames
            if mpc_replay_diagnostic_logging_enabled:
                episode_summary["mpc_replay_diagnostic_events"] = int(
                    len(episode_mpc_replay_diagnostic_records)
                )
            if nominal_rollout_diagnostic_logging_enabled:
                episode_summary["nominal_rollout_diagnostic_events"] = int(
                    len(episode_nominal_rollout_diagnostic_records)
                )
            if trajectory_logging_enabled:
                trajectory_cutoff_step = None
                episode_summary["chunk_trajectory_trace_events"] = int(
                    len(episode_chunk_trajectory_records)
                )
                episode_summary["human_arm_trajectory_samples"] = int(
                    len(episode_human_arm_trajectory_samples)
                )
                episode_summary["executed_policy_trajectory_samples"] = int(
                    len(episode_executed_policy_trajectory_samples)
                )
                episode_summary["trajectory_cutoff_step"] = trajectory_cutoff_step
                if args.plot_chunk_trajectories_3d and (
                    episode_chunk_trajectory_records
                    or episode_human_arm_trajectory_samples
                    or episode_executed_policy_trajectory_samples
                ):
                    plot_path = trajectory_plot_dir / (
                        f"{args.condition}_episode_{episode:03d}_trajectories_3d.html"
                    )
                    saved_plot = _save_chunk_trajectory_viewer(
                        plot_path,
                        f"Episode {episode:03d} SafeChunk 3D trajectories",
                        episode_chunk_trajectory_records,
                        episode_human_arm_trajectory_samples,
                        episode_executed_policy_trajectory_samples,
                        args.chunk_trajectory_plot_max_events,
                    )
                    if saved_plot is not None:
                        episode_summary["chunk_trajectory_viewer_3d"] = saved_plot
                        trajectory_plot_paths.append(saved_plot)
            all_episode_summaries.append(episode_summary)

            if progress_bar is not None:
                progress_bar.close()

            print("\nEpisode summary:")
            for key, value in episode_summary.items():
                print(f"  {key}: {value}")

            if args.plot_terminal:
                _plot_episode_metrics(episode, episode_metrics)

            video_recorder.save(f"{args.condition}_episode_{episode:03d}.mp4")
            if args.record_policy_video:
                policy_video_path = video_dir / f"{args.condition}_policy_obs_episode_{episode:03d}.mp4"
                _save_policy_obs_video(
                    policy_video_frames,
                    policy_video_timestamps,
                    policy_video_path,
                )
                print("  policy obs video:", policy_video_path)

            if episode_bar is not None:
                episode_bar.update(1)
                episode_bar.set_postfix(episodes_left=args.episodes - episode_bar.n)

    finally:
        if episode_bar is not None:
            episode_bar.close()
        if policy_env is not None:
            policy_env.close()
        if safety_env is not None:
            safety_env.close()
        env.close()

    final_summary = summarise_all_chunk_episodes(all_episode_summaries)
    final_summary["normalization_source"] = normalization_source
    final_summary["robot_spawn"] = robot_spawn_info
    final_summary["policy_robot_spawn"] = policy_robot_spawn_info
    final_summary["safety_robot_spawn"] = safety_robot_spawn_info
    final_summary["execution_length"] = workspace_cfg.get("execution_length", None)
    final_summary["action_sequence"] = workspace_cfg.get("action_sequence", None)
    final_summary["video_time_base"] = args.video_time_base
    final_summary["video_stop_steps"] = video_stop_steps
    if mpc_replay_diagnostic_logging_enabled:
        final_summary["mpc_replay_diagnostic_events"] = int(
            len(all_mpc_replay_diagnostic_records)
        )
        final_summary["mpc_replay_diagnostics_jsonl"] = str(mpc_replay_diagnostics_jsonl_path)
        final_summary.update(_mpc_replay_error_summary(all_mpc_replay_diagnostic_records))
    if nominal_rollout_diagnostic_logging_enabled:
        final_summary["nominal_rollout_diagnostic_events"] = int(
            len(all_nominal_rollout_diagnostic_records)
        )
        final_summary["nominal_rollout_diagnostics_jsonl"] = str(nominal_rollout_diagnostics_jsonl_path)
        final_summary.update(_nominal_rollout_error_summary(all_nominal_rollout_diagnostic_records))

    if trajectory_logging_enabled:
        final_summary["chunk_trajectory_trace_events"] = int(
            len(all_chunk_trajectory_records)
        )
        final_summary["human_arm_trajectory_samples"] = int(
            len(all_human_arm_trajectory_samples)
        )
        final_summary["executed_policy_trajectory_samples"] = int(
            len(all_executed_policy_trajectory_samples)
        )
        final_summary["chunk_trajectory_trace_jsonl"] = str(chunk_trajectory_jsonl_path)
        final_summary["human_arm_trajectory_jsonl"] = str(human_arm_trajectory_jsonl_path)
        final_summary["executed_policy_trajectory_jsonl"] = str(executed_policy_trajectory_jsonl_path)
        final_summary["chunk_trajectory_viewer_count"] = int(len(trajectory_plot_paths))
        final_summary["chunk_trajectory_viewers"] = list(trajectory_plot_paths)
        final_summary["chunk_trajectory_include_q_states"] = bool(
            args.chunk_trajectory_include_q_states
        )

    if args.save_actions is not None:
        save_actions_path = Path(args.save_actions)
        if not save_actions_path.is_absolute():
            save_actions_path = output_root / save_actions_path
        save_actions_path.parent.mkdir(parents=True, exist_ok=True)

        if saved_action_episodes:
            min_steps = min(actions.shape[0] for actions in saved_action_episodes)
            actions_to_save = np.stack(
                [actions[:min_steps] for actions in saved_action_episodes],
                axis=0,
            ).astype(np.float32)
        else:
            actions_to_save = np.empty((0, 0) + env_action_shape, dtype=np.float32)

        np.savez_compressed(
            save_actions_path,
            actions=actions_to_save,
            env_action_shape=np.asarray(env_action_shape, dtype=np.int64),
            normalization_source=np.asarray(str(normalization_source)),
        )
        final_summary["saved_actions"] = str(save_actions_path)

    if mpc_replay_diagnostic_logging_enabled:
        with mpc_replay_diagnostics_jsonl_path.open("w") as f:
            for record in all_mpc_replay_diagnostic_records:
                f.write(json.dumps(_jsonable_trace_value(record)) + "\n")
    if nominal_rollout_diagnostic_logging_enabled:
        with nominal_rollout_diagnostics_jsonl_path.open("w") as f:
            for record in all_nominal_rollout_diagnostic_records:
                f.write(json.dumps(_jsonable_trace_value(record)) + "\n")

    if trajectory_logging_enabled:
        with chunk_trajectory_jsonl_path.open("w") as f:
            for record in all_chunk_trajectory_records:
                f.write(json.dumps(_jsonable_trace_value(record)) + "\n")
        with human_arm_trajectory_jsonl_path.open("w") as f:
            for sample in all_human_arm_trajectory_samples:
                f.write(json.dumps(_jsonable_trace_value(sample)) + "\n")
        with executed_policy_trajectory_jsonl_path.open("w") as f:
            for sample in all_executed_policy_trajectory_samples:
                f.write(json.dumps(_jsonable_trace_value(sample)) + "\n")

    with step_jsonl_path.open("w") as f:
        for metric in all_step_metrics:
            f.write(json.dumps(_jsonable_trace_value(asdict(metric))) + "\n")

    with episode_summary_path.open("w") as f:
        json.dump(all_episode_summaries, f, indent=2)

    with final_summary_path.open("w") as f:
        json.dump(final_summary, f, indent=2)

    if final_summary.get("diagnostic_warning"):
        print(f"WARNING: {final_summary['diagnostic_warning']}")

    print("\n========== Final summary ==========")
    for key, value in final_summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print("  step metrics:", step_jsonl_path)
    print("  episode summaries:", episode_summary_path)
    print("  final summary:", final_summary_path)
    if mpc_replay_diagnostic_logging_enabled:
        print("  mpc replay diagnostics:", mpc_replay_diagnostics_jsonl_path)
    if nominal_rollout_diagnostic_logging_enabled:
        print("  nominal rollout diagnostics:", nominal_rollout_diagnostics_jsonl_path)
    if trajectory_logging_enabled:
        print("  chunk trajectory traces:", chunk_trajectory_jsonl_path)
        print("  human arm trajectory:", human_arm_trajectory_jsonl_path)
        print("  executed policy trajectory:", executed_policy_trajectory_jsonl_path)
        if trajectory_plot_paths:
            print("  trajectory viewers:", trajectory_plot_dir)
    if args.save_actions is not None:
        print("  saved actions:", final_summary["saved_actions"])


if __name__ == "__main__":
    main()
