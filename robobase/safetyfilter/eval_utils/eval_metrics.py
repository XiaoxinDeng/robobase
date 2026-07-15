from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from robobase.safetyfilter.eval_utils.eval_utils import summarise_all_episodes, summarise_episode
from robobase.safetyfilter.safechunkdeform.stepmetrics import StepMetrics

logger = logging.getLogger(__name__)


def _as_chunk(action) -> tuple[np.ndarray, bool]:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False


def _safe_info_get(info: dict, key: str, default=None):
    if isinstance(info, dict):
        return info.get(key, default)
    return default


def _finite_float(value):
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return None
    return out if np.isfinite(out) else None

def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return None
        value = arr[0]
    return float(value)


def _optional_int(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return None
        value = arr[0]
    return int(value)


def _optional_bool(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.asarray(value).any())
    return bool(value)


def _optional_str(value):
    if value is None:
        return None
    return str(value)


def _chunk_filter_advantage_metrics(
    nominal_chunk,
    safe_chunk,
    arm_indices,
    intervention_eps: float,
) -> dict[str, Optional[float]]:
    nominal_chunk, _ = _as_chunk(nominal_chunk)
    safe_chunk, _ = _as_chunk(safe_chunk)
    arm_indices = np.asarray(arm_indices, dtype=np.int64)

    nominal_arm = nominal_chunk[:, arm_indices]
    safe_arm = safe_chunk[:, arm_indices]
    delta_arm = safe_arm - nominal_arm
    step_arm_delta = np.linalg.norm(delta_arm, axis=1)
    edited = step_arm_delta > float(intervention_eps)
    edited_steps = np.flatnonzero(edited)
    chunk_arm_delta = float(np.linalg.norm(delta_arm))
    first_delta = float(step_arm_delta[0]) if step_arm_delta.size else 0.0
    future_delta = (
        float(np.linalg.norm(delta_arm[1:])) if delta_arm.shape[0] > 1 else 0.0
    )

    if nominal_arm.shape[0] > 1:
        nominal_variation = float(np.mean(np.linalg.norm(np.diff(nominal_arm, axis=0), axis=1)))
        safe_variation = float(np.mean(np.linalg.norm(np.diff(safe_arm, axis=0), axis=1)))
        edit_variation = float(np.mean(np.linalg.norm(np.diff(delta_arm, axis=0), axis=1)))
    else:
        nominal_variation = 0.0
        safe_variation = 0.0
        edit_variation = 0.0

    denom = max(chunk_arm_delta, 1e-12)
    return {
        "chunk_modified_fraction": float(np.mean(edited)) if edited.size else 0.0,
        "chunk_modified_steps": int(edited_steps.size),
        "chunk_first_modified_step": int(edited_steps[0]) if edited_steps.size else None,
        "chunk_last_modified_step": int(edited_steps[-1]) if edited_steps.size else None,
        "chunk_mean_step_arm_delta": float(np.mean(step_arm_delta)) if step_arm_delta.size else 0.0,
        "chunk_max_step_arm_delta": float(np.max(step_arm_delta)) if step_arm_delta.size else 0.0,
        "chunk_future_arm_delta": future_delta,
        "chunk_future_edit_fraction": float(future_delta / denom),
        "chunk_first_edit_fraction": float(first_delta / denom),
        "chunk_safe_arm_variation": safe_variation,
        "chunk_nominal_arm_variation": nominal_variation,
        "chunk_arm_variation_delta": float(safe_variation - nominal_variation),
        "chunk_edit_variation": edit_variation,
        "chunk_preemptive_intervention": bool(first_delta <= float(intervention_eps) and future_delta > float(intervention_eps)),
    }


def _path_deviation_metrics(safechunk, obs, nominal_chunk, safe_chunk) -> dict[str, Optional[float]]:
    try:
        nominal_q = np.asarray(
            safechunk.rollout_nominal_chunk(obs, nominal_chunk),
            dtype=np.float32,
        )
        safe_q = np.asarray(
            safechunk.rollout_nominal_chunk(obs, safe_chunk),
            dtype=np.float32,
        )
        if nominal_q.shape != safe_q.shape or nominal_q.ndim != 2:
            return {
                "path_mean_deviation": None,
                "path_max_deviation": None,
                "path_final_deviation": None,
            }
        state_idx = np.asarray(safechunk.controlled_state_indices, dtype=np.int64)
        valid = state_idx < nominal_q.shape[1]
        state_idx = state_idx[valid]
        if state_idx.size == 0:
            return {
                "path_mean_deviation": None,
                "path_max_deviation": None,
                "path_final_deviation": None,
            }
        step_deviation = np.linalg.norm(safe_q[:, state_idx] - nominal_q[:, state_idx], axis=1)
        return {
            "path_mean_deviation": float(np.mean(step_deviation)) if step_deviation.size else 0.0,
            "path_max_deviation": float(np.max(step_deviation)) if step_deviation.size else 0.0,
            "path_final_deviation": float(step_deviation[-1]) if step_deviation.size else 0.0,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Path deviation metric failed: %s", exc)
        return {
            "path_mean_deviation": None,
            "path_max_deviation": None,
            "path_final_deviation": None,
        }


def _horizon_risk_gap(current_h, horizon_min_clearance, eps: float = 1e-9):
    if current_h is None or horizon_min_clearance is None:
        return None, None, None
    current = float(current_h)
    horizon = float(horizon_min_clearance)
    if not np.isfinite(current) or not np.isfinite(horizon):
        return None, None, None
    clearance_drop = current - horizon
    risk_gap = max(0.0, clearance_drop)
    return float(risk_gap), bool(risk_gap > eps), float(clearance_drop)


def _chunk_horizon_h_monitor_fallback(safety_info: dict, violation_threshold: float):
    """Map chunk/PACS horizon safety onto legacy h-monitor fields.

    This fallback runs when the live geometric h monitor is disabled.  Older
    code treated chunk ``min_clearance``/``unsafe_count`` as if they were a
    live signed distance. In the SafeChunk path those fields can be legacy raw
    barrier values, so they are useful diagnostics but should not create a
    current h_violation by themselves.

    Returns:
        raw/debug min_h, raw/debug values, h_violation, signed min clearance,
        and the field used as the signed-clearance source.
    """
    if not isinstance(safety_info, dict):
        safety_info = {}

    def _finite_array(value):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:  # noqa: BLE001
            return None
        arr = arr[np.isfinite(arr)]
        return arr if arr.size > 0 else None

    raw_h = _finite_array(_safe_info_get(safety_info, "h_values"))
    raw_min_h = None
    if raw_h is not None:
        raw_min_h = float(np.min(raw_h))
    else:
        for key in ("min_h", "raw_min_h", "min_clearance"):
            value = _safe_info_get(safety_info, key)
            try:
                if value is not None and np.isfinite(float(value)):
                    raw_min_h = float(value)
                    raw_h = np.asarray([raw_min_h], dtype=np.float32)
                    break
            except (TypeError, ValueError):
                continue

    signed_clearances = None
    signed_source = None
    for key in (
        "hold_horizon_min_clearance",
        "horizon_min_clearance",
        "prefix_min_clearance",
        "recover_path_min_clearance",
        "recover_prefix_min_clearance",
    ):
        value = _safe_info_get(safety_info, key)
        try:
            if value is not None and np.isfinite(float(value)):
                signed_clearances = np.asarray([float(value)], dtype=np.float32)
                signed_source = key
                break
        except (TypeError, ValueError):
            continue

    clearance_units = _safe_info_get(safety_info, "clearance_units")
    if signed_clearances is None and clearance_units == "m":
        for key in (
            "min_clearances",
            "horizon_min_clearances",
            "hold_horizon_min_clearances",
            "prefix_min_clearances",
            "recover_path_min_clearances",
        ):
            arr = _finite_array(_safe_info_get(safety_info, key))
            if arr is not None:
                signed_clearances = arr
                signed_source = key
                break
        if signed_clearances is None:
            value = _safe_info_get(safety_info, "min_clearance")
            try:
                if value is not None and np.isfinite(float(value)):
                    signed_clearances = np.asarray([float(value)], dtype=np.float32)
                    signed_source = "min_clearance"
            except (TypeError, ValueError):
                pass

    if signed_clearances is not None:
        min_signed_clearance = float(np.min(signed_clearances))
        threshold = float(violation_threshold)
        h_values = raw_h.astype(float).tolist() if raw_h is not None else signed_clearances.astype(float).tolist()
        min_h = raw_min_h if raw_min_h is not None else min_signed_clearance
        return min_h, h_values, bool(min_signed_clearance < threshold), min_signed_clearance, signed_source

    if raw_min_h is None:
        return None, None, False, None, "raw_h_debug_only"

    return raw_min_h, raw_h.astype(float).tolist(), False, None, "raw_h_debug_only"

def _metric_safety_violation(metric: StepMetrics) -> bool:
    if metric.h_violation is not None:
        return bool(metric.h_violation)
    return False


def _metric_is_brake_step(metric: StepMetrics) -> bool:
    mode = metric.safety_mode
    source = metric.deformation_source
    if mode in {
        "horizon_brake",
        "verified_failsafe",
        "unverified_emergency_failsafe",
        "pause_on_unsafe",
        "pause_and_restart",
        "stop",
    }:
        return True
    if source == "horizon_brake" and mode != "horizon_brake_intended_step":
        return True
    if source == "path_consistent_brake" and mode != "path_consistent_brake_intended_step":
        return True
    return metric.pause_reason is not None


def _metric_is_deformation_step(metric: StepMetrics) -> bool:
    mode = metric.safety_mode
    source = metric.deformation_source
    if mode in {"horizon_deform", "chunk_deform", "emergency_deform_away"}:
        return True
    return source == "chunk_deform"


def _resume_latency_after_human_exit(metrics: list[StepMetrics], default_dt: float = 0.05):
    if not metrics:
        return None
    phases = [m.human_phase for m in metrics]
    if "done" not in phases or not any(phase in {"enter", "hold", "exit"} for phase in phases):
        return None

    first_done = next(i for i, phase in enumerate(phases) if phase == "done")
    for resume_idx in range(first_done, len(metrics)):
        if not _metric_is_brake_step(metrics[resume_idx]) and not _metric_is_deformation_step(metrics[resume_idx]):
            return float((resume_idx - first_done) * default_dt)
    return None

def summarise_chunk_episode(metrics: list[StepMetrics], diagnostics_cfg: Optional[dict[str, float]] = None) -> dict:
    summary = summarise_episode(metrics)
    if len(metrics) == 0:
        return summary

    diagnostics_cfg = dict(diagnostics_cfg or {})
    large_arm_delta_threshold = float(diagnostics_cfg.get("large_arm_delta_threshold", 3.0))
    large_base_delta_threshold = float(diagnostics_cfg.get("large_base_delta_threshold", 0.5))
    low_act_ratio_threshold = float(diagnostics_cfg.get("low_act_ratio_threshold", 0.3))
    high_fallback_ratio_threshold = float(diagnostics_cfg.get("high_fallback_ratio_threshold", 0.5))
    success_threshold = float(diagnostics_cfg.get("success_threshold", 0.9))

    chunk_arm_delta = np.asarray([m.chunk_arm_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_non_arm_delta = np.asarray([m.chunk_non_arm_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_full_delta = np.asarray([m.chunk_full_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_interventions = np.asarray([m.intervention_active for m in metrics], dtype=np.float32)
    chunk_modified_fraction = np.asarray([m.chunk_modified_fraction or 0.0 for m in metrics], dtype=np.float32)
    chunk_modified_steps = np.asarray([m.chunk_modified_steps or 0 for m in metrics], dtype=np.float32)
    chunk_mean_step_arm_delta = np.asarray([m.chunk_mean_step_arm_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_max_step_arm_delta = np.asarray([m.chunk_max_step_arm_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_future_edit_fraction = np.asarray([m.chunk_future_edit_fraction or 0.0 for m in metrics], dtype=np.float32)
    chunk_first_edit_fraction = np.asarray([m.chunk_first_edit_fraction or 0.0 for m in metrics], dtype=np.float32)
    chunk_safe_arm_variation = np.asarray([m.chunk_safe_arm_variation or 0.0 for m in metrics], dtype=np.float32)
    chunk_nominal_arm_variation = np.asarray([m.chunk_nominal_arm_variation or 0.0 for m in metrics], dtype=np.float32)
    chunk_arm_variation_delta = np.asarray([m.chunk_arm_variation_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_edit_variation = np.asarray([m.chunk_edit_variation or 0.0 for m in metrics], dtype=np.float32)
    path_mean_deviation = np.asarray([m.path_mean_deviation for m in metrics if m.path_mean_deviation is not None], dtype=np.float32)
    path_max_deviation = np.asarray([m.path_max_deviation for m in metrics if m.path_max_deviation is not None], dtype=np.float32)
    path_final_deviation = np.asarray([m.path_final_deviation for m in metrics if m.path_final_deviation is not None], dtype=np.float32)
    chunk_preemptive_interventions = np.asarray([bool(m.chunk_preemptive_intervention) for m in metrics], dtype=np.float32)
    horizon_risk_gaps = [m.horizon_risk_gap for m in metrics if m.horizon_risk_gap is not None]
    horizon_clearance_drops = [m.horizon_clearance_drop for m in metrics if m.horizon_clearance_drop is not None]
    horizon_risk_gap_active = [m.horizon_risk_gap_active for m in metrics if m.horizon_risk_gap_active is not None]
    horizon_only_risk = [
        bool(m.horizon_risk_gap_active) and not bool(m.h_violation)
        for m in metrics
        if m.horizon_risk_gap_active is not None and m.h_violation is not None
    ]
    first_modified_steps = [m.chunk_first_modified_step for m in metrics if m.chunk_first_modified_step is not None]
    deform_norms = [m.deformation_norm for m in metrics if m.deformation_norm is not None]
    deform_envelope_loss = np.asarray([m.deform_envelope_loss for m in metrics if m.deform_envelope_loss is not None], dtype=np.float32)
    deform_envelope_first_delta = np.asarray([m.deform_envelope_first_delta for m in metrics if m.deform_envelope_first_delta is not None], dtype=np.float32)
    deform_envelope_first_violation = np.asarray([m.deform_envelope_first_violation for m in metrics if m.deform_envelope_first_violation is not None], dtype=np.float32)
    deform_envelope_avoid_rate_loss = np.asarray([m.deform_envelope_avoid_rate_loss for m in metrics if m.deform_envelope_avoid_rate_loss is not None], dtype=np.float32)
    deform_envelope_return_rate_loss = np.asarray([m.deform_envelope_return_rate_loss for m in metrics if m.deform_envelope_return_rate_loss is not None], dtype=np.float32)
    deform_envelope_max_rate = np.asarray([m.deform_envelope_max_rate for m in metrics if m.deform_envelope_max_rate is not None], dtype=np.float32)
    deform_envelope_terminal_delta = np.asarray([m.deform_envelope_terminal_delta for m in metrics if m.deform_envelope_terminal_delta is not None], dtype=np.float32)
    deform_envelope_terminal_violation = np.asarray([m.deform_envelope_terminal_violation for m in metrics if m.deform_envelope_terminal_violation is not None], dtype=np.float32)
    deform_envelope_terminal_loss = np.asarray([m.deform_envelope_terminal_loss for m in metrics if m.deform_envelope_terminal_loss is not None], dtype=np.float32)
    deform_envelope_acceleration_loss = np.asarray([m.deform_envelope_acceleration_loss for m in metrics if m.deform_envelope_acceleration_loss is not None], dtype=np.float32)
    deform_safe = [m.deform_safe for m in metrics if m.deform_safe is not None]
    optimized_records = [m for m in metrics if m.optimized_accepted is not None]
    optimized_attempts = [m.optimized_accepted for m in optimized_records]
    optimized_safe = [bool(m.deform_safe) for m in optimized_records]
    recoverable_checks = [
        bool(m.is_recoverable) for m in optimized_records if m.is_recoverable is not None
    ]
    fallback_steps = [m.fallback_used for m in metrics if m.fallback_used is not None]
    rejection_causes = [m.rejection_cause for m in optimized_records]
    deform_stage_checks = [m.deform_stage_accepted for m in metrics if m.deform_stage_accepted is not None]
    recover_checks = [m.recover_accepted for m in metrics if m.recover_accepted is not None]
    recover_reject_reasons = [
        m.recover_reject_reason
        for m in metrics
        if m.recover_reject_reason is not None
    ]

    def _count_strings(values):
        items = [str(value) for value in values if value is not None]
        return {item: items.count(item) for item in sorted(set(items))}
    direct_rejoin_attempted_steps = int(
        np.sum([bool(m.direct_rejoin_attempted) for m in metrics])
    )
    direct_rejoin_rejected_steps = int(
        np.sum([bool(m.direct_rejoin_rejected) for m in metrics])
    )
    detour_rejoin_attempted_steps = int(
        np.sum([bool(m.detour_rejoin_attempted) for m in metrics])
    )
    detour_rejoin_accepted_steps = int(
        np.sum([bool(m.detour_rejoin_accepted) for m in metrics])
    )
    delayed_rejoin_active_steps = int(
        np.sum([bool(m.delayed_rejoin_active) for m in metrics])
    )
    repeated_unsafe_target_steps = int(
        np.sum([bool(m.repeated_unsafe_target) for m in metrics])
    )
    post_recovery_act_window_steps = int(
        np.sum([bool(m.post_recovery_act_window_active) for m in metrics])
    )
    post_recovery_act_window_interrupted_steps = int(
        np.sum([bool(m.post_recovery_act_window_interrupted) for m in metrics])
    )
    cached_motion = [m.cached_motion_active for m in metrics if m.cached_motion_active is not None]
    resumed_indices = [
        m.resumed_from_cached_index
        for m in metrics
        if m.resumed_from_cached_index is not None
    ]

    def finite_metric(name):
        vals = [getattr(m, name) for m in metrics if getattr(m, name) is not None]
        vals = [float(v) for v in vals if np.isfinite(float(v))]
        return np.asarray(vals, dtype=np.float32)

    q_rejoin_dist = finite_metric("q_rejoin_dist")
    qd_rejoin_dist = finite_metric("qd_rejoin_dist")
    ee_rejoin_dist = finite_metric("ee_rejoin_dist")
    rejoin_q_eval_time_ms = finite_metric("rejoin_q_eval_time_ms")
    rejoin_qd_eval_time_ms = finite_metric("rejoin_qd_eval_time_ms")
    ee_nom_cache_time_ms = finite_metric("ee_nom_cache_time_ms")
    ee_final_check_time_ms = finite_metric("ee_final_check_time_ms")
    deform_stage_min_clearance = finite_metric("deform_stage_min_clearance")
    recover_min_clearance = finite_metric("recover_min_clearance")
    recover_rejoin_loss = finite_metric("recover_rejoin_loss")
    recover_path_min_clearance = finite_metric("recover_path_min_clearance")
    recover_immediate_clearance = finite_metric("recover_immediate_clearance")
    recover_prefix_min_clearance = finite_metric("recover_prefix_min_clearance")
    committed_clearance_prediction_error = finite_metric("clearance_prediction_error")
    committed_planned_vs_actual_q_error = finite_metric("planned_vs_actual_q_error")
    committed_human_motion_since_plan = finite_metric("human_motion_since_plan")
    committed_accepted_clearance_margin = finite_metric("accepted_clearance_margin")
    committed_state_error = finite_metric("committed_state_error")
    committed_repair_time_ms = finite_metric("committed_repair_time_ms")
    committed_repair_safety_time_ms = finite_metric("committed_repair_safety_time_ms")
    committed_action_safety_time_ms = finite_metric("committed_action_safety_time_ms")
    committed_abort_brake_time_ms = finite_metric("committed_abort_brake_time_ms")
    planning_vs_replay_clearance_post_error = finite_metric(
        "planning_vs_replay_clearance_post_error"
    )
    planning_vs_replay_human_error = finite_metric("planning_vs_replay_human_error")
    actual_vs_planned_post_q_error = finite_metric("actual_vs_planned_post_q_error")
    recover_projection_on_nominal = finite_metric("recover_projection_on_nominal")
    recover_cosine_to_nominal = finite_metric("recover_cosine_to_nominal")
    recover_direction_cosine = finite_metric("recover_direction_cosine")
    recover_direction_loss = finite_metric("recover_direction_loss")
    recover_act_progress_loss = finite_metric("recover_act_progress_loss")
    recover_act_heading_loss = finite_metric("recover_act_heading_loss")
    recover_act_direction_loss = finite_metric("recover_act_direction_loss")
    recover_act_progress_projection = finite_metric("recover_act_progress_projection")
    recover_act_target_progress = finite_metric("recover_act_target_progress")
    recover_act_heading_cosine = finite_metric("recover_act_heading_cosine")
    recover_ordered_pose_loss = finite_metric("recover_ordered_pose_loss")
    recover_ordered_delta_loss = finite_metric("recover_ordered_delta_loss")
    recover_ordered_loss = finite_metric("recover_ordered_loss")
    recover_task_progress_score = finite_metric("recover_task_progress_score")
    recover_resume_tube_score = finite_metric("recover_resume_tube_score")
    recover_resume_tube_component_score = finite_metric("recover_resume_tube_component_score")
    recover_resume_tube_terminal_score = finite_metric("recover_resume_tube_terminal_score")
    recover_resume_tube_path_score = finite_metric("recover_resume_tube_path_score")
    recover_resume_tube_progress_score = finite_metric("recover_resume_tube_progress_score")
    recover_resume_tube_heading_score = finite_metric("recover_resume_tube_heading_score")
    recover_resume_tube_clearance_score = finite_metric("recover_resume_tube_clearance_score")
    recover_resume_tube_terminal_dist = finite_metric("recover_resume_tube_terminal_dist")
    recover_resume_tube_ordered_loss = finite_metric("recover_resume_tube_ordered_loss")
    recover_resume_tube_prefix_min_clearance = finite_metric("recover_resume_tube_prefix_min_clearance")
    recover_resume_window_wrist_l2_mean = finite_metric("recover_resume_window_wrist_l2_mean")
    recover_resume_window_wrist_l2_max = finite_metric("recover_resume_window_wrist_l2_max")
    recover_resume_window_left_wrist_abs_mean = finite_metric("recover_resume_window_left_wrist_abs_mean")
    recover_resume_window_left_wrist_abs_max = finite_metric("recover_resume_window_left_wrist_abs_max")
    recover_resume_window_right_wrist_abs_mean = finite_metric("recover_resume_window_right_wrist_abs_mean")
    recover_resume_window_right_wrist_abs_max = finite_metric("recover_resume_window_right_wrist_abs_max")
    resume_affordance_score = finite_metric("resume_affordance_score")
    resume_affordance_component_score = finite_metric("resume_affordance_component_score")
    resume_affordance_target_distance = finite_metric("resume_affordance_target_distance")
    resume_affordance_contact_score = finite_metric("resume_affordance_contact_score")
    act_resumable_score = finite_metric("act_resumable_score")
    act_resumable_nominal_score = finite_metric("act_resumable_nominal_score")
    act_resumable_live_score = finite_metric("act_resumable_live_score")
    act_action_agreement_act_vs_safe_l2 = finite_metric("act_action_agreement_act_vs_safe_l2")
    act_action_agreement_act_vs_safe_cosine = finite_metric("act_action_agreement_act_vs_safe_cosine")
    act_action_agreement_act_vs_safe_arm_l2 = finite_metric("act_action_agreement_act_vs_safe_arm_l2")
    act_action_agreement_act_vs_safe_gripper_abs_delta = finite_metric("act_action_agreement_act_vs_safe_gripper_abs_delta")
    act_action_agreement_act_vs_nominal_l2 = finite_metric("act_action_agreement_act_vs_nominal_l2")
    act_action_agreement_act_vs_nominal_cosine = finite_metric("act_action_agreement_act_vs_nominal_cosine")
    act_action_agreement_act_vs_nominal_arm_l2 = finite_metric("act_action_agreement_act_vs_nominal_arm_l2")
    act_action_agreement_act_vs_nominal_gripper_abs_delta = finite_metric("act_action_agreement_act_vs_nominal_gripper_abs_delta")
    act_action_agreement_safe_vs_nominal_l2 = finite_metric("act_action_agreement_safe_vs_nominal_l2")
    act_action_agreement_safe_vs_nominal_cosine = finite_metric("act_action_agreement_safe_vs_nominal_cosine")
    act_action_agreement_safe_vs_nominal_arm_l2 = finite_metric("act_action_agreement_safe_vs_nominal_arm_l2")
    act_action_agreement_safe_vs_nominal_gripper_abs_delta = finite_metric("act_action_agreement_safe_vs_nominal_gripper_abs_delta")
    act_action_agreement_act_vs_target_l2 = finite_metric("act_action_agreement_act_vs_target_l2")
    act_action_agreement_act_vs_target_cosine = finite_metric("act_action_agreement_act_vs_target_cosine")
    act_action_agreement_act_vs_target_arm_l2 = finite_metric("act_action_agreement_act_vs_target_arm_l2")
    act_action_agreement_act_vs_target_gripper_abs_delta = finite_metric("act_action_agreement_act_vs_target_gripper_abs_delta")
    act_action_agreement_safe_vs_target_l2 = finite_metric("act_action_agreement_safe_vs_target_l2")
    act_action_agreement_safe_vs_target_cosine = finite_metric("act_action_agreement_safe_vs_target_cosine")
    act_action_agreement_safe_vs_target_arm_l2 = finite_metric("act_action_agreement_safe_vs_target_arm_l2")
    act_action_agreement_safe_vs_target_gripper_abs_delta = finite_metric("act_action_agreement_safe_vs_target_gripper_abs_delta")
    act_action_agreement_act_vs_last_recovery_l2 = finite_metric("act_action_agreement_act_vs_last_recovery_l2")
    act_action_agreement_act_vs_last_recovery_cosine = finite_metric("act_action_agreement_act_vs_last_recovery_cosine")
    act_action_agreement_act_vs_last_recovery_arm_l2 = finite_metric("act_action_agreement_act_vs_last_recovery_arm_l2")
    act_action_agreement_act_vs_last_recovery_gripper_abs_delta = finite_metric("act_action_agreement_act_vs_last_recovery_gripper_abs_delta")
    act_action_agreement_safe_vs_last_recovery_l2 = finite_metric("act_action_agreement_safe_vs_last_recovery_l2")
    act_action_agreement_safe_vs_last_recovery_cosine = finite_metric("act_action_agreement_safe_vs_last_recovery_cosine")
    act_action_agreement_safe_vs_last_recovery_arm_l2 = finite_metric("act_action_agreement_safe_vs_last_recovery_arm_l2")
    act_action_agreement_safe_vs_last_recovery_gripper_abs_delta = finite_metric("act_action_agreement_safe_vs_last_recovery_gripper_abs_delta")
    recover_resume_affordance_score = finite_metric("recover_resume_affordance_score")
    recover_resume_affordance_component_score = finite_metric("recover_resume_affordance_component_score")
    recover_resume_affordance_target_distance = finite_metric("recover_resume_affordance_target_distance")
    recover_resume_affordance_contact_score = finite_metric("recover_resume_affordance_contact_score")
    recover_resume_affordance_bonus = finite_metric("recover_resume_affordance_bonus")
    mpc_handoff_resume_affordance_score = finite_metric("mpc_handoff_resume_affordance_score")
    mpc_handoff_resume_affordance_component_score = finite_metric("mpc_handoff_resume_affordance_component_score")
    committed_rejoin_resume_affordance_score = finite_metric("committed_rejoin_resume_affordance_score")
    committed_rejoin_resume_affordance_component_score = finite_metric("committed_rejoin_resume_affordance_component_score")
    committed_soft_handoff_resume_affordance_score = finite_metric("committed_soft_handoff_resume_affordance_score")
    mpc_handoff_act_vs_release_action_l2 = finite_metric("mpc_handoff_act_vs_release_action_l2")
    mpc_handoff_act_vs_release_action_cosine = finite_metric("mpc_handoff_act_vs_release_action_cosine")
    mpc_handoff_act_vs_release_action_arm_l2 = finite_metric("mpc_handoff_act_vs_release_action_arm_l2")
    mpc_handoff_act_vs_target_action_l2 = finite_metric("mpc_handoff_act_vs_target_action_l2")
    mpc_handoff_act_vs_target_action_cosine = finite_metric("mpc_handoff_act_vs_target_action_cosine")
    mpc_handoff_act_vs_target_action_arm_l2 = finite_metric("mpc_handoff_act_vs_target_action_arm_l2")
    mpc_handoff_resume_tube_score = finite_metric("mpc_handoff_resume_tube_score")
    mpc_handoff_resume_tube_component_score = finite_metric("mpc_handoff_resume_tube_component_score")
    mpc_handoff_resume_tube_terminal_score = finite_metric("mpc_handoff_resume_tube_terminal_score")
    mpc_handoff_resume_tube_path_score = finite_metric("mpc_handoff_resume_tube_path_score")
    mpc_handoff_resume_tube_progress_score = finite_metric("mpc_handoff_resume_tube_progress_score")
    mpc_handoff_resume_tube_heading_score = finite_metric("mpc_handoff_resume_tube_heading_score")
    mpc_handoff_resume_tube_clearance_score = finite_metric("mpc_handoff_resume_tube_clearance_score")
    mpc_handoff_resume_tube_terminal_dist = finite_metric("mpc_handoff_resume_tube_terminal_dist")
    mpc_handoff_resume_tube_ordered_loss = finite_metric("mpc_handoff_resume_tube_ordered_loss")
    mpc_handoff_resume_tube_prefix_min_clearance = finite_metric("mpc_handoff_resume_tube_prefix_min_clearance")
    mpc_recovery_target_tube_window_wrist_l2_mean = finite_metric("mpc_recovery_target_tube_window_wrist_l2_mean")
    mpc_recovery_target_tube_window_wrist_l2_max = finite_metric("mpc_recovery_target_tube_window_wrist_l2_max")
    mpc_recovery_target_tube_window_left_wrist_abs_mean = finite_metric("mpc_recovery_target_tube_window_left_wrist_abs_mean")
    mpc_recovery_target_tube_window_left_wrist_abs_max = finite_metric("mpc_recovery_target_tube_window_left_wrist_abs_max")
    mpc_recovery_target_tube_window_right_wrist_abs_mean = finite_metric("mpc_recovery_target_tube_window_right_wrist_abs_mean")
    mpc_recovery_target_tube_window_right_wrist_abs_max = finite_metric("mpc_recovery_target_tube_window_right_wrist_abs_max")
    committed_soft_handoff_resume_tube_score = finite_metric("committed_soft_handoff_resume_tube_score")
    committed_soft_handoff_resume_tube_component_score = finite_metric("committed_soft_handoff_resume_tube_component_score")
    committed_rejoin_resume_tube_score = finite_metric("committed_rejoin_resume_tube_score")
    committed_rejoin_resume_tube_component_score = finite_metric("committed_rejoin_resume_tube_component_score")
    committed_rejoin_resume_tube_terminal_score = finite_metric("committed_rejoin_resume_tube_terminal_score")
    committed_rejoin_resume_tube_path_score = finite_metric("committed_rejoin_resume_tube_path_score")
    committed_rejoin_resume_tube_progress_score = finite_metric("committed_rejoin_resume_tube_progress_score")
    committed_rejoin_resume_tube_heading_score = finite_metric("committed_rejoin_resume_tube_heading_score")
    committed_rejoin_resume_tube_clearance_score = finite_metric("committed_rejoin_resume_tube_clearance_score")
    committed_rejoin_resume_tube_terminal_dist = finite_metric("committed_rejoin_resume_tube_terminal_dist")
    committed_rejoin_resume_tube_ordered_loss = finite_metric("committed_rejoin_resume_tube_ordered_loss")
    committed_rejoin_resume_tube_prefix_min_clearance = finite_metric("committed_rejoin_resume_tube_prefix_min_clearance")
    gradient_iterations_run = finite_metric("gradient_iterations_run")
    gradient_line_search_batch_evaluations = finite_metric("gradient_line_search_batch_evaluations")
    gradient_jax_scan_used_count = finite_metric("gradient_jax_scan_used_count")
    gradient_full_jax_scan_time_ms = finite_metric("gradient_full_jax_scan_time_ms")
    optimizer_evaluations = finite_metric("optimizer_evaluations")
    deform_optimizer_time_ms = finite_metric("deform_optimizer_time_ms")
    return_optimizer_time_ms = finite_metric("return_optimizer_time_ms")
    explicit_optimizer_time_ms = finite_metric("explicit_optimizer_time_ms")
    committed_suffix_optimizer_time_ms = finite_metric("committed_suffix_optimizer_time_ms")
    committed_plan_rollout_time_ms = finite_metric("committed_plan_rollout_time_ms")
    committed_plan_safety_time_ms = finite_metric("committed_plan_safety_time_ms")
    committed_plan_diagnostics_time_ms = finite_metric("committed_plan_diagnostics_time_ms")
    cem_iterations_run = finite_metric("cem_iterations_run")
    deform_cem_iterations_run = finite_metric("deform_cem_iterations_run")
    return_cem_iterations_run = finite_metric("return_cem_iterations_run")
    hold_horizon_min_clearance = finite_metric("hold_horizon_min_clearance")
    pacs_background_min_clearance = finite_metric("pacs_background_min_clearance")
    pacs_background_arm_delta = finite_metric("pacs_background_arm_delta")
    pacs_background_chunk_arm_delta = finite_metric("pacs_background_chunk_arm_delta")
    pacs_background_chunk_modified_fraction = finite_metric("pacs_background_chunk_modified_fraction")
    pacs_background_retiming_arm_delta = finite_metric("pacs_background_retiming_arm_delta")
    pacs_background_retiming_chunk_arm_delta = finite_metric("pacs_background_retiming_chunk_arm_delta")
    pacs_background_retiming_changed_fraction = finite_metric("pacs_background_retiming_changed_fraction")
    retiming_norm = finite_metric("retiming_norm")
    task_progress = finite_metric("task_progress_after")
    task_progress_delta = finite_metric("task_progress_delta")
    total_steps = len(metrics)
    act_step_flags = np.asarray([bool(m.act_step) for m in metrics], dtype=np.bool_)
    deform_step_flags = np.asarray([bool(m.deform_step) for m in metrics], dtype=np.bool_)
    recover_step_flags = np.asarray([bool(m.recover_step) for m in metrics], dtype=np.bool_)
    brake_step_flags = np.asarray([bool(m.brake_step) for m in metrics], dtype=np.bool_)
    fallback_step_flags = np.asarray([bool(m.fallback_step) for m in metrics], dtype=np.bool_)
    optimized_attempt_step_flags = np.asarray([bool(m.optimized_attempt_step) for m in metrics], dtype=np.bool_)
    optimized_accepted_step_flags = np.asarray([bool(m.optimized_accepted_step) for m in metrics], dtype=np.bool_)
    temporary_wait_steps = int(np.sum([bool(m.temporary_wait_step) for m in metrics]))
    resume_after_wait_count = int(np.sum([bool(m.resume_act_after_wait) for m in metrics]))
    deform_after_persistent_block_count = int(np.sum([bool(m.deform_after_persistent_block) for m in metrics]))
    deform_suppressed_by_temporary_wait_count = int(
        np.sum([bool(m.deform_suppressed_by_temporary_wait) for m in metrics])
    )
    recovery_optimization_skipped_steps = int(
        np.sum([bool(m.recovery_optimization_skipped) for m in metrics])
    )
    recovery_failure_streak_vals = [
        int(m.recovery_failure_streak_max)
        for m in metrics
        if m.recovery_failure_streak_max is not None
    ]
    recovery_failure_streak_max = (
        int(np.max(recovery_failure_streak_vals))
        if recovery_failure_streak_vals
        else 0
    )

    def max_int_metric(name):
        vals = [
            int(getattr(m, name))
            for m in metrics
            if getattr(m, name) is not None
        ]
        return int(np.max(vals)) if vals else 0

    deform_replan_count = max_int_metric("deform_replan_count")
    recovery_replan_count = max_int_metric("recovery_replan_count")
    recovery_optimization_skipped_count = max_int_metric(
        "recovery_optimization_skipped_count"
    )
    stale_recovery_suppressed_count = max_int_metric("stale_recovery_suppressed_count")
    recovery_target_infeasible_count = max_int_metric("recovery_target_infeasible_count")
    emergency_brake_steps = max_int_metric("emergency_brake_steps")
    optimized_attempt_count = max_int_metric("optimized_attempt_count")
    optimized_solution_count = max_int_metric("optimized_solution_count")
    fallback_attempt_count = max_int_metric("fallback_attempt_count")
    fallback_attempt_accepted_count = max_int_metric("fallback_attempt_accepted_count")
    optimized_rejected_count = max_int_metric("optimized_rejected_count")
    deform_option_attempt_count = max_int_metric("deform_option_attempt_count")
    deform_accepted_count = max_int_metric("deform_accepted_count")
    deform_rejected_count = max_int_metric("deform_rejected_count")
    recover_option_attempt_count = max_int_metric("recover_option_attempt_count")
    recover_accepted_count = max_int_metric("recover_accepted_count")
    recover_rejected_count = max_int_metric("recover_rejected_count")
    safe_corridor_recovery_count = max_int_metric("safe_corridor_recovery_count")
    direct_rejoin_attempt_count = max_int_metric("direct_rejoin_attempt_count")
    direct_rejoin_reject_count = max_int_metric("direct_rejoin_reject_count")
    detour_rejoin_attempt_count = max_int_metric("detour_rejoin_attempt_count")
    detour_rejoin_accept_count = max_int_metric("detour_rejoin_accept_count")
    delayed_rejoin_count = max_int_metric("delayed_rejoin_count")
    recover_path_unsafe_count = max_int_metric("recover_path_unsafe_count")
    recovery_path_failure_streak_max = max_int_metric("recovery_path_failure_streak_max")
    repeated_unsafe_target_count = max_int_metric("repeated_unsafe_target_count")
    post_recovery_act_window_count = max_int_metric("post_recovery_act_window_count")
    post_recovery_act_window_interrupted_count = max_int_metric("post_recovery_act_window_interrupted_count")
    safe_prefix_accepted_count = max_int_metric("safe_prefix_accepted_count")
    first_action_only_accepted_count = max_int_metric("first_action_only_accepted_count")
    immediate_hard_reject_count = max_int_metric("immediate_hard_reject_count")
    no_safe_prefix_reject_count = max_int_metric("no_safe_prefix_reject_count")
    horizon_margin_reject_count = max_int_metric("horizon_margin_reject_count")
    accepted_deform_steps = max_int_metric("accepted_deform_steps")
    accepted_recover_steps = max_int_metric("accepted_recover_steps")
    fallback_brake_after_reject_count = max_int_metric("fallback_brake_after_reject_count")
    nominal_rejoin_available_count = max_int_metric("nominal_rejoin_available_count")
    nominal_rejoin_suppressed_count = max_int_metric("nominal_rejoin_suppressed_count")
    stale_nominal_rejoin_suppressed_count = max_int_metric("stale_nominal_rejoin_suppressed_count")
    nominal_prefix_unsafe_suppressed_count = max_int_metric("nominal_prefix_unsafe_suppressed_count")
    recover_positive_projection_count = max_int_metric("recover_positive_projection_count")
    recover_act_progress_ok_count = sum(
        1 for metric in metrics if metric.recover_act_progress_ok is True
    )
    recover_act_heading_ok_count = sum(
        1 for metric in metrics if metric.recover_act_heading_ok is True
    )
    recover_resume_tube_ok_count = sum(
        1 for metric in metrics if metric.recover_resume_tube_ok is True
    )
    recover_resume_tube_reject_count = sum(
        1 for metric in metrics if metric.recover_resume_tube_ok is False
    )
    resume_affordance_available_count = sum(
        1 for metric in metrics if metric.resume_affordance_available is True
    )
    resume_affordance_ok_count = sum(
        1 for metric in metrics if metric.resume_affordance_ok is True
    )
    act_resumable_ok_count = sum(
        1 for metric in metrics if getattr(metric, "act_resumable_ok", None) is True
    )
    act_resumable_live_ok_count = sum(
        1 for metric in metrics if getattr(metric, "act_resumable_live_ok", None) is True
    )
    act_resumable_nominal_ok_count = sum(
        1 for metric in metrics if getattr(metric, "act_resumable_nominal_ok", None) is True
    )
    act_action_agreement_logged_count = sum(
        1 for metric in metrics if getattr(metric, "act_action_agreement_logged", None) is True
    )
    act_action_agreement_post_recovery_or_reentry_count = sum(
        1
        for metric in metrics
        if getattr(metric, "act_action_agreement_post_recovery_or_reentry", None) is True
    )
    recover_resume_affordance_ok_count = sum(
        1 for metric in metrics if metric.recover_resume_affordance_ok is True
    )
    recover_resume_affordance_reject_count = sum(
        1 for metric in metrics if metric.recover_resume_affordance_ok is False
    )
    mpc_handoff_resume_affordance_ok_count = sum(
        1 for metric in metrics if metric.mpc_handoff_resume_affordance_ok is True
    )
    committed_rejoin_resume_affordance_ok_count = sum(
        1 for metric in metrics if metric.committed_rejoin_resume_affordance_ok is True
    )
    mpc_handoff_resume_tube_ok_count = sum(
        1 for metric in metrics if metric.mpc_handoff_resume_tube_ok is True
    )
    mpc_handoff_resume_tube_reject_count = sum(
        1 for metric in metrics if metric.mpc_handoff_resume_tube_ok is False
    )
    committed_soft_handoff_resume_tube_ok_count = sum(
        1 for metric in metrics if metric.committed_soft_handoff_resume_tube_ok is True
    )
    committed_rejoin_resume_tube_ok_count = sum(
        1 for metric in metrics if metric.committed_rejoin_resume_tube_ok is True
    )
    committed_rejoin_resume_tube_reject_count = sum(
        1 for metric in metrics if metric.committed_rejoin_resume_tube_ok is False
    )
    recover_nonpositive_projection_count = max_int_metric("recover_nonpositive_projection_count")
    emergency_deform_away_steps = max_int_metric("emergency_deform_away_steps")
    emergency_deform_away_count = max_int_metric("emergency_deform_away_count")
    hold_unsafe_count = max_int_metric("hold_unsafe_count")
    hold_predicted_contact_count = max_int_metric("hold_predicted_contact_count")
    contact_during_hold_count = max_int_metric("contact_during_hold_count")
    contact_during_brake_count = max_int_metric("contact_during_brake_count")
    contact_during_deform_count = max_int_metric("contact_during_deform_count")
    contact_during_recover_count = max_int_metric("contact_during_recover_count")

    def mean_progress_for(flag_name):
        vals = [
            m.task_progress_delta
            for m in metrics
            if bool(getattr(m, flag_name))
            and m.task_progress_delta is not None
            and np.isfinite(float(m.task_progress_delta))
        ]
        return float(np.mean(vals)) if vals else None

    act_steps = int(np.sum(act_step_flags))
    deform_steps = int(np.sum(deform_step_flags))
    recover_steps = int(np.sum(recover_step_flags))
    brake_step_count = int(np.sum(brake_step_flags))
    fallback_step_count = int(np.sum(fallback_step_flags))
    pacs_background_check_only_steps = int(
        np.sum([bool(m.pacs_background_check_only) for m in metrics])
    )
    pacs_background_brake_steps = int(
        np.sum([bool(m.pacs_background_brake_step) for m in metrics])
    )
    pacs_background_act_steps = int(
        np.sum([bool(m.pacs_background_act_step) for m in metrics])
    )
    optimized_attempt_steps = int(np.sum(optimized_attempt_step_flags))
    optimized_accepted_steps = int(np.sum(optimized_accepted_step_flags))
    committed_chunk_started_count = int(
        np.sum([bool(m.committed_chunk_started) for m in metrics])
    )
    committed_chunk_completed_count = int(
        np.sum([bool(m.committed_chunk_completed) for m in metrics])
    )
    committed_state_mismatch_abort_count = int(
        np.sum([bool(m.committed_aborted_due_to_state_mismatch) for m in metrics])
    )
    committed_state_mismatch_recovered_count = int(
        np.sum([bool(m.committed_state_mismatch_recovered) for m in metrics])
    )
    committed_suffix_replan_attempt_count = int(
        np.sum([bool(m.committed_suffix_replan_attempted) for m in metrics])
    )
    committed_suffix_replan_accepted_count = int(
        np.sum([bool(m.committed_suffix_replan_accepted) for m in metrics])
    )
    committed_suffix_replan_rejected_count = int(
        np.sum([bool(m.committed_suffix_replan_rejected) for m in metrics])
    )
    committed_suffix_replan_reject_reason_counts = _count_strings(
        [m.committed_suffix_replan_reject_reason for m in metrics]
    )
    committed_opportunistic_resume_count = int(
        np.sum([bool(m.committed_opportunistic_resume) for m in metrics])
    )
    mpc_recovery_replan_attempt_count = int(
        np.sum([bool(m.mpc_recovery_replan_attempted) for m in metrics])
    )
    mpc_recovery_replan_accepted_count = int(
        np.sum([bool(m.mpc_recovery_replan_accepted) for m in metrics])
    )
    mpc_recovery_replan_rejected_count = int(
        np.sum([bool(m.mpc_recovery_replan_rejected) for m in metrics])
    )
    mpc_recovery_active_steps = int(
        np.sum([bool(m.mpc_recovery_active) for m in metrics])
    )
    mpc_recovery_replan_reject_reason_counts = _count_strings(
        [m.mpc_recovery_replan_reject_reason for m in metrics]
    )
    mpc_handoff_attempt_count = int(
        np.sum([bool(m.mpc_handoff_attempted) for m in metrics])
    )
    mpc_handoff_accepted_count = int(
        np.sum([bool(m.mpc_handoff_accepted) for m in metrics])
    )
    mpc_handoff_rejected_count = int(
        np.sum([bool(m.mpc_handoff_rejected) for m in metrics])
    )
    mpc_handoff_reject_reason_counts = _count_strings(
        [m.mpc_handoff_reject_reason for m in metrics]
    )
    mpc_handoff_action_agreement_ok_count = int(
        np.sum([bool(getattr(m, "mpc_handoff_action_agreement_ok", False)) for m in metrics])
    )
    mpc_handoff_action_agreement_override_allowed_count = int(
        np.sum([bool(getattr(m, "mpc_handoff_action_agreement_override_allowed", False)) for m in metrics])
    )
    mpc_handoff_heading_overridden_by_action_agreement_count = int(
        np.sum([bool(getattr(m, "mpc_handoff_heading_overridden_by_action_agreement", False)) for m in metrics])
    )
    mpc_handoff_progress_overridden_by_action_agreement_count = int(
        np.sum([bool(getattr(m, "mpc_handoff_progress_overridden_by_action_agreement", False)) for m in metrics])
    )
    mpc_handoff_action_agreement_override_reason_counts = _count_strings(
        [getattr(m, "mpc_handoff_action_agreement_override_reason", None) for m in metrics]
    )
    mpc_progress_delta = np.asarray(
        [
            m.mpc_recovery_live_q_progress_delta
            for m in metrics
            if m.mpc_recovery_live_q_progress_delta is not None
        ],
        dtype=np.float32,
    )
    mpc_live_before = np.asarray(
        [
            m.mpc_recovery_live_q_dist_before
            for m in metrics
            if m.mpc_recovery_live_q_dist_before is not None
        ],
        dtype=np.float32,
    )
    mpc_live_after = np.asarray(
        [
            m.mpc_recovery_live_q_dist_after
            for m in metrics
            if m.mpc_recovery_live_q_dist_after is not None
        ],
        dtype=np.float32,
    )
    mpc_live_progress_flags = [
        bool(m.mpc_recovery_live_progress_ok)
        for m in metrics
        if m.mpc_recovery_live_progress_ok is not None
    ]
    recovery_budget_progress_delta = np.asarray(
        [
            m.recovery_budget_progress_delta
            for m in metrics
            if m.recovery_budget_progress_delta is not None
            and np.isfinite(float(m.recovery_budget_progress_delta))
        ],
        dtype=np.float32,
    )
    recovery_budget_live_q_dist = np.asarray(
        [
            m.recovery_budget_live_q_dist
            for m in metrics
            if m.recovery_budget_live_q_dist is not None
        ],
        dtype=np.float32,
    )
    recovery_budget_progress_flags = [
        bool(m.recovery_budget_progress_ok)
        for m in metrics
        if m.recovery_budget_progress_ok is not None
    ]
    recovery_budget_extended_steps = int(
        np.sum([bool(m.recovery_budget_extended) for m in metrics])
    )
    staged_recovery_ordered_path_softened_steps = int(
        np.sum([bool(m.staged_recovery_ordered_path_softened) for m in metrics])
    )
    staged_recovery_progress_accepted_steps = int(
        np.sum([bool(m.staged_recovery_progress_accepted) for m in metrics])
    )
    recover_handover_ready_steps = int(
        np.sum([bool(m.recover_handover_ready) for m in metrics])
    )
    recover_progress_only_steps = int(
        np.sum([bool(m.recover_progress_only) for m in metrics])
    )
    recovery_handover_pending_steps = int(
        np.sum([bool(m.recovery_handover_pending) for m in metrics])
    )
    mpc_recovery_budget_escape_steps = int(
        np.sum([bool(m.mpc_recovery_budget_escape) for m in metrics])
    )
    mpc_recovery_prefix_replay_steps = int(
        np.sum([bool(m.mpc_recovery_prefix_replay_step) for m in metrics])
    )
    committed_state_mismatch_ignored_for_mpc_prefix_count = int(
        np.sum([bool(m.committed_state_mismatch_ignored_for_mpc_prefix) for m in metrics])
    )
    committed_released_for_act_resume_count = int(
        np.sum([bool(m.committed_released_for_act_resume) for m in metrics])
    )
    committed_recovery_budget_exit_count = int(
        np.sum([bool(m.committed_recovery_budget_exit) for m in metrics])
    )
    committed_replan_due_to_recovery_budget_count = int(
        np.sum([bool(m.committed_replan_due_to_recovery_budget) for m in metrics])
    )
    committed_chunk_abort_count = int(
        np.sum(
            [
                bool(m.committed_aborted_due_to_safety)
                or bool(m.committed_aborted_due_to_state_mismatch)
                for m in metrics
            ]
        )
    )
    committed_repaired_step_count = int(
        np.sum([bool(m.committed_repaired_step) for m in metrics])
    )
    committed_abort_due_to_human_motion_count = int(
        np.sum([bool(m.committed_abort_due_to_human_motion) for m in metrics])
    )
    committed_abort_due_to_prediction_error_count = int(
        np.sum([bool(m.committed_abort_due_to_prediction_error) for m in metrics])
    )
    committed_abort_due_to_safety_semantics_mismatch_count = int(
        np.sum(
            [
                bool(m.committed_abort_due_to_safety_semantics_mismatch)
                for m in metrics
            ]
        )
    )
    committed_deform_steps_executed = int(
        np.sum([m.deform_steps_executed or 0 for m in metrics])
    )
    committed_recover_steps_executed = int(
        np.sum([m.recover_steps_executed or 0 for m in metrics])
    )
    resume_from_committed_rejoin_count = int(
        np.sum([bool(m.resume_from_committed_rejoin) for m in metrics])
    )
    recovery_action_history_reset_count = int(
        np.sum([bool(m.recovery_action_history_reset) for m in metrics])
    )
    recovery_visual_history_reset_count = int(
        np.sum([bool(m.recovery_visual_history_reset) for m in metrics])
    )
    recovery_visual_history_reset_entries = int(
        np.sum([m.recovery_visual_history_reset_count or 0 for m in metrics])
    )
    contact_during_hold_count = int(np.sum([bool(m.contact_during_hold) for m in metrics]))
    contact_during_brake_count = int(np.sum([bool(m.contact_during_brake) for m in metrics]))
    contact_during_deform_count = int(np.sum([bool(m.contact_during_deform) for m in metrics]))
    contact_during_recover_count = int(np.sum([bool(m.contact_during_recover) for m in metrics]))
    act_ratio = float(act_steps / total_steps) if total_steps else None
    safety_mode_ratio = (
        float((deform_steps + recover_steps + brake_step_count + fallback_step_count) / total_steps)
        if total_steps
        else None
    )
    fallback_ratio = float(fallback_step_count / total_steps) if total_steps else None
    final_task_progress = float(task_progress[-1]) if task_progress.size else None
    max_task_progress = float(np.max(task_progress)) if task_progress.size else None
    mean_chunk_arm_delta_for_failure = float(np.mean(chunk_arm_delta)) if chunk_arm_delta.size else 0.0
    accepted_recover_chunks_not_executed = bool(
        recover_checks and np.sum(recover_checks) > 0 and recover_steps == 0
    )
    diagnostic_warning = (
        "accepted_recover_chunks_not_executed"
        if accepted_recover_chunks_not_executed
        else None
    )

    if act_ratio is not None and act_ratio < low_act_ratio_threshold:
        likely_failure_cause = "low_act_utilization"
    elif (
        max_task_progress is not None
        and final_task_progress is not None
        and max_task_progress > success_threshold * 0.7
        and final_task_progress < max_task_progress * 0.5
    ):
        likely_failure_cause = "progress_lost_after_intervention"
    elif mean_chunk_arm_delta_for_failure > large_arm_delta_threshold:
        likely_failure_cause = (
            "large_retiming_delta_ood"
            if metrics[0].condition == "path_consistent_brake"
            else "large_deformation_ood"
        )
    elif fallback_ratio is not None and fallback_ratio > high_fallback_ratio_threshold:
        likely_failure_cause = "fallback_braking_timeout"
    else:
        likely_failure_cause = "unknown"

    modes = [m.safety_mode for m in metrics]
    pacs_background_modes = [m.pacs_background_safety_mode for m in metrics]
    pause_reasons = [m.pause_reason for m in metrics]
    sources = [m.deformation_source for m in metrics]
    robot_human_distances = [m.min_robot_human_distance for m in metrics if m.min_robot_human_distance is not None]
    drawer_open_distances = [m.drawer_open_distance for m in metrics if m.drawer_open_distance is not None]
    safety_violations = [_metric_safety_violation(m) for m in metrics]
    brake_steps = [_metric_is_brake_step(m) for m in metrics]
    deformation_steps = [_metric_is_deformation_step(m) for m in metrics]
    phase_reanchor_steps = [m.safety_mode == "phase_reanchor" for m in metrics]
    gripper_latched_steps = int(np.sum([bool(m.gripper_latched) for m in metrics]))
    post_recovery_task_guard_steps = int(
        np.sum([bool(m.post_recovery_task_guard_active) for m in metrics])
    )
    post_recovery_reanchor_started_count = int(
        np.sum([bool(m.post_recovery_reanchor_started) for m in metrics])
    )
    post_recovery_progress_regression_count = int(
        np.sum(
            [
                (m.post_recovery_progress_regression or 0.0) > 0.0
                for m in metrics
            ]
        )
    )
    post_recovery_mid_progress_no_progress_count = int(
        np.sum(
            [
                (m.post_recovery_mid_progress_no_progress_count or 0) > 0
                for m in metrics
            ]
        )
    )
    post_recovery_mid_progress_triggered_count = int(
        np.sum(
            [
                bool(m.post_recovery_no_progress_triggered)
                and isinstance(m.post_recovery_task_guard_reason, str)
                and m.post_recovery_task_guard_reason.startswith("mid_progress_no_progress")
                for m in metrics
            ]
        )
    )
    post_recovery_mid_progress_reseed_count = int(
        np.sum(
            [
                bool(m.post_recovery_mid_progress_reseed_triggered)
                for m in metrics
            ]
        )
    )

    def rate(values, target):
        return float(np.mean([v == target for v in values])) if values else None

    env_step_time_ms = np.asarray([m.env_step_time_ms for m in metrics], dtype=np.float32)
    policy_obs_adapt_time_ms = np.asarray([m.policy_obs_adapt_time_ms for m in metrics], dtype=np.float32)
    policy_action_time_ms = np.asarray([m.policy_action_time_ms for m in metrics], dtype=np.float32)
    policy_obs_update_time_ms = np.asarray([m.policy_obs_update_time_ms for m in metrics], dtype=np.float32)
    policy_total_time_ms = policy_obs_adapt_time_ms + policy_action_time_ms + policy_obs_update_time_ms
    step_wall_time_ms = np.asarray([1000.0 * m.step_wall_time_s for m in metrics], dtype=np.float32)
    filter_time_ms = np.asarray([m.filter_time_ms for m in metrics], dtype=np.float32)
    monitor_time_ms = np.asarray([m.monitor_time_ms for m in metrics], dtype=np.float32)
    residual_time_ms = np.maximum(
        0.0,
        step_wall_time_ms
        - filter_time_ms
        - monitor_time_ms
        - env_step_time_ms
        - policy_total_time_ms,
    )

    summary.update(
        {
            "mean_env_step_time_ms": float(np.mean(env_step_time_ms)),
            "p50_env_step_time_ms": float(np.percentile(env_step_time_ms, 50)),
            "p95_env_step_time_ms": float(np.percentile(env_step_time_ms, 95)),
            "max_env_step_time_ms": float(np.max(env_step_time_ms)),
            "mean_policy_obs_adapt_time_ms": float(np.mean(policy_obs_adapt_time_ms)),
            "p50_policy_obs_adapt_time_ms": float(np.percentile(policy_obs_adapt_time_ms, 50)),
            "p95_policy_obs_adapt_time_ms": float(np.percentile(policy_obs_adapt_time_ms, 95)),
            "max_policy_obs_adapt_time_ms": float(np.max(policy_obs_adapt_time_ms)),
            "mean_policy_action_time_ms": float(np.mean(policy_action_time_ms)),
            "p50_policy_action_time_ms": float(np.percentile(policy_action_time_ms, 50)),
            "p95_policy_action_time_ms": float(np.percentile(policy_action_time_ms, 95)),
            "max_policy_action_time_ms": float(np.max(policy_action_time_ms)),
            "mean_policy_obs_update_time_ms": float(np.mean(policy_obs_update_time_ms)),
            "p50_policy_obs_update_time_ms": float(np.percentile(policy_obs_update_time_ms, 50)),
            "p95_policy_obs_update_time_ms": float(np.percentile(policy_obs_update_time_ms, 95)),
            "max_policy_obs_update_time_ms": float(np.max(policy_obs_update_time_ms)),
            "mean_policy_total_time_ms": float(np.mean(policy_total_time_ms)),
            "p50_policy_total_time_ms": float(np.percentile(policy_total_time_ms, 50)),
            "p95_policy_total_time_ms": float(np.percentile(policy_total_time_ms, 95)),
            "max_policy_total_time_ms": float(np.max(policy_total_time_ms)),
            "mean_latency_residual_ms": float(np.mean(residual_time_ms)),
            "p50_latency_residual_ms": float(np.percentile(residual_time_ms, 50)),
            "p95_latency_residual_ms": float(np.percentile(residual_time_ms, 95)),
            "max_latency_residual_ms": float(np.max(residual_time_ms)),
            "mean_chunk_arm_delta": float(np.mean(chunk_arm_delta)),
            "max_chunk_arm_delta": float(np.max(chunk_arm_delta)),
            "mean_chunk_non_arm_delta": float(np.mean(chunk_non_arm_delta)),
            "max_chunk_non_arm_delta": float(np.max(chunk_non_arm_delta)),
            "mean_chunk_full_delta": float(np.mean(chunk_full_delta)),
            "max_chunk_full_delta": float(np.max(chunk_full_delta)),
            "chunk_intervention_frequency": float(np.mean(chunk_interventions)),
            "mean_chunk_modified_fraction": float(np.mean(chunk_modified_fraction)),
            "mean_chunk_modified_steps": float(np.mean(chunk_modified_steps)),
            "mean_chunk_first_modified_step": float(np.mean(first_modified_steps)) if first_modified_steps else None,
            "mean_chunk_mean_step_arm_delta": float(np.mean(chunk_mean_step_arm_delta)),
            "max_chunk_step_arm_delta": float(np.max(chunk_max_step_arm_delta)),
            "mean_chunk_future_edit_fraction": float(np.mean(chunk_future_edit_fraction)),
            "mean_chunk_first_edit_fraction": float(np.mean(chunk_first_edit_fraction)),
            "mean_chunk_safe_arm_variation": float(np.mean(chunk_safe_arm_variation)),
            "mean_chunk_nominal_arm_variation": float(np.mean(chunk_nominal_arm_variation)),
            "mean_chunk_arm_variation_delta": float(np.mean(chunk_arm_variation_delta)),
            "mean_chunk_edit_variation": float(np.mean(chunk_edit_variation)),
            "mean_path_deviation": float(np.mean(path_mean_deviation)) if path_mean_deviation.size else None,
            "max_path_deviation": float(np.max(path_max_deviation)) if path_max_deviation.size else None,
            "mean_final_path_deviation": float(np.mean(path_final_deviation)) if path_final_deviation.size else None,
            "chunk_preemptive_intervention_frequency": float(np.mean(chunk_preemptive_interventions)),
            "mean_horizon_risk_gap": float(np.mean(horizon_risk_gaps)) if horizon_risk_gaps else None,
            "max_horizon_risk_gap": float(np.max(horizon_risk_gaps)) if horizon_risk_gaps else None,
            "mean_horizon_clearance_drop": float(np.mean(horizon_clearance_drops)) if horizon_clearance_drops else None,
            "horizon_risk_gap_rate": float(np.mean(horizon_risk_gap_active)) if horizon_risk_gap_active else None,
            "horizon_only_risk_rate": float(np.mean(horizon_only_risk)) if horizon_only_risk else None,
            "h_argmin_robot_part_counts": _count_strings(m.h_argmin_robot_part for m in metrics),
            "h_argmin_human_part_counts": _count_strings(m.h_argmin_human_part for m in metrics),
            "h_violation_argmin_robot_part_counts": _count_strings(
                m.h_argmin_robot_part for m in metrics if bool(m.h_violation)
            ),
            "h_violation_argmin_human_part_counts": _count_strings(
                m.h_argmin_human_part for m in metrics if bool(m.h_violation)
            ),
            "mean_deformation_norm": float(np.mean(deform_norms)) if deform_norms else None,
            "mean_deform_envelope_loss": float(np.mean(deform_envelope_loss)) if deform_envelope_loss.size else None,
            "max_deform_envelope_loss": float(np.max(deform_envelope_loss)) if deform_envelope_loss.size else None,
            "mean_deform_envelope_first_delta": float(np.mean(deform_envelope_first_delta)) if deform_envelope_first_delta.size else None,
            "max_deform_envelope_first_delta": float(np.max(deform_envelope_first_delta)) if deform_envelope_first_delta.size else None,
            "mean_deform_envelope_first_violation": float(np.mean(deform_envelope_first_violation)) if deform_envelope_first_violation.size else None,
            "max_deform_envelope_first_violation": float(np.max(deform_envelope_first_violation)) if deform_envelope_first_violation.size else None,
            "mean_deform_envelope_avoid_rate_loss": float(np.mean(deform_envelope_avoid_rate_loss)) if deform_envelope_avoid_rate_loss.size else None,
            "mean_deform_envelope_return_rate_loss": float(np.mean(deform_envelope_return_rate_loss)) if deform_envelope_return_rate_loss.size else None,
            "mean_deform_envelope_max_rate": float(np.mean(deform_envelope_max_rate)) if deform_envelope_max_rate.size else None,
            "max_deform_envelope_max_rate": float(np.max(deform_envelope_max_rate)) if deform_envelope_max_rate.size else None,
            "mean_deform_envelope_terminal_delta": float(np.mean(deform_envelope_terminal_delta)) if deform_envelope_terminal_delta.size else None,
            "max_deform_envelope_terminal_delta": float(np.max(deform_envelope_terminal_delta)) if deform_envelope_terminal_delta.size else None,
            "mean_deform_envelope_terminal_violation": float(np.mean(deform_envelope_terminal_violation)) if deform_envelope_terminal_violation.size else None,
            "max_deform_envelope_terminal_violation": float(np.max(deform_envelope_terminal_violation)) if deform_envelope_terminal_violation.size else None,
            "mean_deform_envelope_terminal_loss": float(np.mean(deform_envelope_terminal_loss)) if deform_envelope_terminal_loss.size else None,
            "mean_deform_envelope_acceleration_loss": float(np.mean(deform_envelope_acceleration_loss)) if deform_envelope_acceleration_loss.size else None,
            "mean_retiming_norm": float(np.mean(retiming_norm)) if retiming_norm.size else None,
            "deform_safe_rate": float(np.mean(deform_safe)) if deform_safe else None,
            "optimized_attempts": int(len(optimized_records)),
            "optimized_safe_count": int(np.sum(optimized_safe)) if optimized_records else 0,
            "optimized_recoverable_count": int(np.sum(recoverable_checks)) if recoverable_checks else 0,
            "optimized_accepted_count": int(np.sum(optimized_attempts)) if optimized_attempts else 0,
            "rejected_unsafe_count": int(np.sum([c == "unsafe" for c in rejection_causes])),
            "rejected_unrecoverable_count": int(np.sum([c == "unrecoverable" for c in rejection_causes])),
            "rejected_both_count": int(np.sum([c == "unsafe_and_unrecoverable" for c in rejection_causes])),
            "fallback_used_count": int(np.sum(fallback_steps)) if fallback_steps else 0,
            "deform_stage_accepted_count": int(np.sum(deform_stage_checks)) if deform_stage_checks else 0,
            "recover_accepted_count": int(np.sum(recover_checks)) if recover_checks else 0,
            "cached_motion_active_count": int(np.sum(cached_motion)) if cached_motion else 0,
            "resumed_from_cached_count": int(len(resumed_indices)),
            "mean_deform_stage_min_clearance": float(np.mean(deform_stage_min_clearance)) if deform_stage_min_clearance.size else None,
            "mean_recover_min_clearance": float(np.mean(recover_min_clearance)) if recover_min_clearance.size else None,
            "mean_recover_rejoin_loss": float(np.mean(recover_rejoin_loss)) if recover_rejoin_loss.size else None,
            "mean_recover_path_min_clearance": float(np.mean(recover_path_min_clearance)) if recover_path_min_clearance.size else None,
            "min_recover_path_min_clearance": float(np.min(recover_path_min_clearance)) if recover_path_min_clearance.size else None,
            "mean_recover_immediate_clearance": float(np.mean(recover_immediate_clearance)) if recover_immediate_clearance.size else None,
            "mean_recover_prefix_min_clearance": float(np.mean(recover_prefix_min_clearance)) if recover_prefix_min_clearance.size else None,
            "optimized_attempt_count": int(len(optimized_attempts)),
            "optimized_accept_rate": float(np.mean(optimized_attempts)) if optimized_attempts else None,
            "recoverable_rate": float(np.mean(recoverable_checks)) if recoverable_checks else None,
            "fallback_used_rate": float(np.mean(fallback_steps)) if fallback_steps else None,
            "mean_q_rejoin_dist": float(np.mean(q_rejoin_dist)) if q_rejoin_dist.size else None,
            "max_q_rejoin_dist": float(np.max(q_rejoin_dist)) if q_rejoin_dist.size else None,
            "mean_qd_rejoin_dist": float(np.mean(qd_rejoin_dist)) if qd_rejoin_dist.size else None,
            "max_qd_rejoin_dist": float(np.max(qd_rejoin_dist)) if qd_rejoin_dist.size else None,
            "mean_ee_rejoin_dist": float(np.mean(ee_rejoin_dist)) if ee_rejoin_dist.size else None,
            "max_ee_rejoin_dist": float(np.max(ee_rejoin_dist)) if ee_rejoin_dist.size else None,
            "mean_rejoin_q_eval_time_ms": float(np.mean(rejoin_q_eval_time_ms)) if rejoin_q_eval_time_ms.size else None,
            "mean_rejoin_qd_eval_time_ms": float(np.mean(rejoin_qd_eval_time_ms)) if rejoin_qd_eval_time_ms.size else None,
            "mean_ee_nom_cache_time_ms": float(np.mean(ee_nom_cache_time_ms)) if ee_nom_cache_time_ms.size else None,
            "mean_ee_final_check_time_ms": float(np.mean(ee_final_check_time_ms)) if ee_final_check_time_ms.size else None,
            "act_steps": act_steps,
            "deform_steps": deform_steps,
            "recover_steps": recover_steps,
            "brake_steps": brake_step_count,
            "fallback_steps": fallback_step_count,
            "pacs_background_check_only_steps": pacs_background_check_only_steps,
            "pacs_background_brake_steps": pacs_background_brake_steps,
            "pacs_background_act_steps": pacs_background_act_steps,
            "pacs_background_check_only_rate": (
                float(pacs_background_check_only_steps / total_steps) if total_steps else None
            ),
            "pacs_background_brake_rate": (
                float(pacs_background_brake_steps / total_steps) if total_steps else None
            ),
            "pacs_background_act_rate": (
                float(pacs_background_act_steps / total_steps) if total_steps else None
            ),
            "mean_pacs_background_min_clearance": (
                float(np.mean(pacs_background_min_clearance))
                if pacs_background_min_clearance.size
                else None
            ),
            "min_pacs_background_min_clearance": (
                float(np.min(pacs_background_min_clearance))
                if pacs_background_min_clearance.size
                else None
            ),
            "mean_pacs_background_arm_delta": (
                float(np.mean(pacs_background_arm_delta))
                if pacs_background_arm_delta.size
                else None
            ),
            "mean_pacs_background_chunk_arm_delta": (
                float(np.mean(pacs_background_chunk_arm_delta))
                if pacs_background_chunk_arm_delta.size
                else None
            ),
            "mean_pacs_background_chunk_modified_fraction": (
                float(np.mean(pacs_background_chunk_modified_fraction))
                if pacs_background_chunk_modified_fraction.size
                else None
            ),
            "mean_pacs_background_retiming_arm_delta": (
                float(np.mean(pacs_background_retiming_arm_delta))
                if pacs_background_retiming_arm_delta.size
                else None
            ),
            "mean_pacs_background_retiming_chunk_arm_delta": (
                float(np.mean(pacs_background_retiming_chunk_arm_delta))
                if pacs_background_retiming_chunk_arm_delta.size
                else None
            ),
            "mean_pacs_background_retiming_changed_fraction": (
                float(np.mean(pacs_background_retiming_changed_fraction))
                if pacs_background_retiming_changed_fraction.size
                else None
            ),
            "pacs_background_path_consistent_brake_intended_rate": rate(
                pacs_background_modes,
                "path_consistent_brake_intended_step",
            ),
            "pacs_background_verified_failsafe_rate": rate(
                pacs_background_modes,
                "verified_failsafe",
            ),
            "pacs_background_unverified_emergency_failsafe_rate": rate(
                pacs_background_modes,
                "unverified_emergency_failsafe",
            ),
            "optimized_attempt_steps": optimized_attempt_steps,
            "optimized_accepted_steps": optimized_accepted_steps,
            "committed_chunk_started_count": committed_chunk_started_count,
            "committed_chunk_completed_count": committed_chunk_completed_count,
            "committed_chunk_abort_count": committed_chunk_abort_count,
            "committed_repaired_step_count": committed_repaired_step_count,
            "committed_abort_due_to_human_motion_count": committed_abort_due_to_human_motion_count,
            "committed_abort_due_to_prediction_error_count": committed_abort_due_to_prediction_error_count,
            "committed_abort_due_to_safety_semantics_mismatch_count": committed_abort_due_to_safety_semantics_mismatch_count,
            "committed_state_mismatch_abort_count": committed_state_mismatch_abort_count,
            "committed_state_mismatch_recovered_count": committed_state_mismatch_recovered_count,
            "committed_suffix_replan_attempt_count": committed_suffix_replan_attempt_count,
            "committed_suffix_replan_accepted_count": committed_suffix_replan_accepted_count,
            "committed_suffix_replan_rejected_count": committed_suffix_replan_rejected_count,
            "committed_suffix_replan_reject_reason_counts": committed_suffix_replan_reject_reason_counts,
            "committed_opportunistic_resume_count": committed_opportunistic_resume_count,
            "mpc_recovery_active_steps": mpc_recovery_active_steps,
            "mpc_recovery_replan_attempt_count": mpc_recovery_replan_attempt_count,
            "mpc_recovery_replan_accepted_count": mpc_recovery_replan_accepted_count,
            "mpc_recovery_replan_rejected_count": mpc_recovery_replan_rejected_count,
            "mpc_recovery_replan_reject_reason_counts": mpc_recovery_replan_reject_reason_counts,
            "mpc_handoff_attempt_count": mpc_handoff_attempt_count,
            "mpc_handoff_accepted_count": mpc_handoff_accepted_count,
            "mpc_handoff_rejected_count": mpc_handoff_rejected_count,
            "mpc_handoff_reject_reason_counts": mpc_handoff_reject_reason_counts,
            "mpc_handoff_action_agreement_ok_count": mpc_handoff_action_agreement_ok_count,
            "mpc_handoff_action_agreement_override_allowed_count": mpc_handoff_action_agreement_override_allowed_count,
            "mpc_handoff_heading_overridden_by_action_agreement_count": mpc_handoff_heading_overridden_by_action_agreement_count,
            "mpc_handoff_progress_overridden_by_action_agreement_count": mpc_handoff_progress_overridden_by_action_agreement_count,
            "mpc_handoff_action_agreement_override_reason_counts": mpc_handoff_action_agreement_override_reason_counts,
            "mean_mpc_handoff_act_vs_release_action_l2": float(np.mean(mpc_handoff_act_vs_release_action_l2)) if mpc_handoff_act_vs_release_action_l2.size else None,
            "mean_mpc_handoff_act_vs_release_action_cosine": float(np.mean(mpc_handoff_act_vs_release_action_cosine)) if mpc_handoff_act_vs_release_action_cosine.size else None,
            "mean_mpc_handoff_act_vs_release_action_arm_l2": float(np.mean(mpc_handoff_act_vs_release_action_arm_l2)) if mpc_handoff_act_vs_release_action_arm_l2.size else None,
            "mean_mpc_handoff_act_vs_target_action_l2": float(np.mean(mpc_handoff_act_vs_target_action_l2)) if mpc_handoff_act_vs_target_action_l2.size else None,
            "mean_mpc_handoff_act_vs_target_action_cosine": float(np.mean(mpc_handoff_act_vs_target_action_cosine)) if mpc_handoff_act_vs_target_action_cosine.size else None,
            "mean_mpc_handoff_act_vs_target_action_arm_l2": float(np.mean(mpc_handoff_act_vs_target_action_arm_l2)) if mpc_handoff_act_vs_target_action_arm_l2.size else None,
            "mean_mpc_handoff_resume_tube_score": float(np.mean(mpc_handoff_resume_tube_score)) if mpc_handoff_resume_tube_score.size else None,
            "min_mpc_handoff_resume_tube_score": float(np.min(mpc_handoff_resume_tube_score)) if mpc_handoff_resume_tube_score.size else None,
            "mean_mpc_handoff_resume_tube_component_score": float(np.mean(mpc_handoff_resume_tube_component_score)) if mpc_handoff_resume_tube_component_score.size else None,
            "mean_mpc_handoff_resume_tube_terminal_score": float(np.mean(mpc_handoff_resume_tube_terminal_score)) if mpc_handoff_resume_tube_terminal_score.size else None,
            "mean_mpc_handoff_resume_tube_path_score": float(np.mean(mpc_handoff_resume_tube_path_score)) if mpc_handoff_resume_tube_path_score.size else None,
            "mean_mpc_handoff_resume_tube_progress_score": float(np.mean(mpc_handoff_resume_tube_progress_score)) if mpc_handoff_resume_tube_progress_score.size else None,
            "mean_mpc_handoff_resume_tube_heading_score": float(np.mean(mpc_handoff_resume_tube_heading_score)) if mpc_handoff_resume_tube_heading_score.size else None,
            "mean_mpc_handoff_resume_tube_clearance_score": float(np.mean(mpc_handoff_resume_tube_clearance_score)) if mpc_handoff_resume_tube_clearance_score.size else None,
            "mean_mpc_handoff_resume_tube_terminal_dist": float(np.mean(mpc_handoff_resume_tube_terminal_dist)) if mpc_handoff_resume_tube_terminal_dist.size else None,
            "mean_mpc_handoff_resume_tube_ordered_loss": float(np.mean(mpc_handoff_resume_tube_ordered_loss)) if mpc_handoff_resume_tube_ordered_loss.size else None,
            "mean_mpc_handoff_resume_tube_prefix_min_clearance": float(np.mean(mpc_handoff_resume_tube_prefix_min_clearance)) if mpc_handoff_resume_tube_prefix_min_clearance.size else None,
            "mpc_handoff_resume_tube_ok_count": mpc_handoff_resume_tube_ok_count,
            "mpc_handoff_resume_tube_reject_count": mpc_handoff_resume_tube_reject_count,
            "mpc_handoff_count": max_int_metric("mpc_handoff_attempt_count"),
            "mpc_handoff_accept_count": max_int_metric("mpc_handoff_accept_count"),
            "mpc_handoff_reject_count": max_int_metric("mpc_handoff_reject_count"),
            "mpc_recovery_replan_count": max_int_metric("mpc_recovery_replan_count"),
            "mpc_recovery_accepted_count": max_int_metric("mpc_recovery_accepted_count"),
            "mpc_recovery_rejected_count": max_int_metric("mpc_recovery_rejected_count"),
            "mpc_recovery_no_progress_reject_count": max_int_metric("mpc_recovery_no_progress_reject_count"),
            "mpc_recovery_budget_escape_count": max_int_metric("mpc_recovery_budget_escape_count"),
            "mpc_recovery_budget_escape_steps": mpc_recovery_budget_escape_steps,
            "mean_mpc_recovery_target_tube_window_wrist_l2_mean": float(np.mean(mpc_recovery_target_tube_window_wrist_l2_mean)) if mpc_recovery_target_tube_window_wrist_l2_mean.size else None,
            "max_mpc_recovery_target_tube_window_wrist_l2_max": float(np.max(mpc_recovery_target_tube_window_wrist_l2_max)) if mpc_recovery_target_tube_window_wrist_l2_max.size else None,
            "mean_mpc_recovery_target_tube_window_left_wrist_abs_mean": float(np.mean(mpc_recovery_target_tube_window_left_wrist_abs_mean)) if mpc_recovery_target_tube_window_left_wrist_abs_mean.size else None,
            "max_mpc_recovery_target_tube_window_left_wrist_abs_max": float(np.max(mpc_recovery_target_tube_window_left_wrist_abs_max)) if mpc_recovery_target_tube_window_left_wrist_abs_max.size else None,
            "mean_mpc_recovery_target_tube_window_right_wrist_abs_mean": float(np.mean(mpc_recovery_target_tube_window_right_wrist_abs_mean)) if mpc_recovery_target_tube_window_right_wrist_abs_mean.size else None,
            "max_mpc_recovery_target_tube_window_right_wrist_abs_max": float(np.max(mpc_recovery_target_tube_window_right_wrist_abs_max)) if mpc_recovery_target_tube_window_right_wrist_abs_max.size else None,
            "recovery_budget_extended_count": max_int_metric("recovery_budget_extended_count"),
            "recovery_budget_extended_steps": recovery_budget_extended_steps,
            "recovery_budget_progress_ok_rate": float(np.mean(recovery_budget_progress_flags)) if recovery_budget_progress_flags else None,
            "mean_recovery_budget_progress_delta": float(np.mean(recovery_budget_progress_delta)) if recovery_budget_progress_delta.size else None,
            "mean_recovery_budget_live_q_dist": float(np.mean(recovery_budget_live_q_dist)) if recovery_budget_live_q_dist.size else None,
            "staged_recovery_ordered_path_softened_steps": staged_recovery_ordered_path_softened_steps,
            "staged_recovery_progress_accepted_steps": staged_recovery_progress_accepted_steps,
            "recover_handover_ready_steps": recover_handover_ready_steps,
            "recover_progress_only_steps": recover_progress_only_steps,
            "recovery_handover_pending_steps": recovery_handover_pending_steps,
            "mpc_recovery_prefix_replay_steps": mpc_recovery_prefix_replay_steps,
            "committed_state_mismatch_ignored_for_mpc_prefix_count": committed_state_mismatch_ignored_for_mpc_prefix_count,
            "mean_mpc_recovery_live_q_dist_before": float(np.mean(mpc_live_before)) if mpc_live_before.size else None,
            "mean_mpc_recovery_live_q_dist_after": float(np.mean(mpc_live_after)) if mpc_live_after.size else None,
            "mean_mpc_recovery_live_q_progress_delta": float(np.mean(mpc_progress_delta)) if mpc_progress_delta.size else None,
            "min_mpc_recovery_live_q_progress_delta": float(np.min(mpc_progress_delta)) if mpc_progress_delta.size else None,
            "mpc_recovery_live_progress_ok_rate": float(np.mean(mpc_live_progress_flags)) if mpc_live_progress_flags else None,
            "committed_released_for_act_resume_count": committed_released_for_act_resume_count,
            "mean_committed_soft_handoff_resume_tube_score": float(np.mean(committed_soft_handoff_resume_tube_score)) if committed_soft_handoff_resume_tube_score.size else None,
            "mean_committed_soft_handoff_resume_tube_component_score": float(np.mean(committed_soft_handoff_resume_tube_component_score)) if committed_soft_handoff_resume_tube_component_score.size else None,
            "committed_soft_handoff_resume_tube_ok_count": committed_soft_handoff_resume_tube_ok_count,
            "mean_committed_rejoin_resume_tube_score": float(np.mean(committed_rejoin_resume_tube_score)) if committed_rejoin_resume_tube_score.size else None,
            "min_committed_rejoin_resume_tube_score": float(np.min(committed_rejoin_resume_tube_score)) if committed_rejoin_resume_tube_score.size else None,
            "mean_committed_rejoin_resume_tube_component_score": float(np.mean(committed_rejoin_resume_tube_component_score)) if committed_rejoin_resume_tube_component_score.size else None,
            "mean_committed_rejoin_resume_tube_terminal_score": float(np.mean(committed_rejoin_resume_tube_terminal_score)) if committed_rejoin_resume_tube_terminal_score.size else None,
            "mean_committed_rejoin_resume_tube_path_score": float(np.mean(committed_rejoin_resume_tube_path_score)) if committed_rejoin_resume_tube_path_score.size else None,
            "mean_committed_rejoin_resume_tube_progress_score": float(np.mean(committed_rejoin_resume_tube_progress_score)) if committed_rejoin_resume_tube_progress_score.size else None,
            "mean_committed_rejoin_resume_tube_heading_score": float(np.mean(committed_rejoin_resume_tube_heading_score)) if committed_rejoin_resume_tube_heading_score.size else None,
            "mean_committed_rejoin_resume_tube_clearance_score": float(np.mean(committed_rejoin_resume_tube_clearance_score)) if committed_rejoin_resume_tube_clearance_score.size else None,
            "mean_committed_rejoin_resume_tube_terminal_dist": float(np.mean(committed_rejoin_resume_tube_terminal_dist)) if committed_rejoin_resume_tube_terminal_dist.size else None,
            "mean_committed_rejoin_resume_tube_ordered_loss": float(np.mean(committed_rejoin_resume_tube_ordered_loss)) if committed_rejoin_resume_tube_ordered_loss.size else None,
            "mean_committed_rejoin_resume_tube_prefix_min_clearance": float(np.mean(committed_rejoin_resume_tube_prefix_min_clearance)) if committed_rejoin_resume_tube_prefix_min_clearance.size else None,
            "committed_rejoin_resume_tube_ok_count": committed_rejoin_resume_tube_ok_count,
            "committed_rejoin_resume_tube_reject_count": committed_rejoin_resume_tube_reject_count,
            "committed_recovery_budget_exit_count": committed_recovery_budget_exit_count,
            "committed_replan_due_to_recovery_budget_count": committed_replan_due_to_recovery_budget_count,
            "mean_committed_repair_time_ms": (
                float(np.mean(committed_repair_time_ms))
                if committed_repair_time_ms.size
                else None
            ),
            "max_committed_repair_time_ms": (
                float(np.max(committed_repair_time_ms))
                if committed_repair_time_ms.size
                else None
            ),
            "mean_committed_repair_safety_time_ms": (
                float(np.mean(committed_repair_safety_time_ms))
                if committed_repair_safety_time_ms.size
                else None
            ),
            "max_committed_repair_safety_time_ms": (
                float(np.max(committed_repair_safety_time_ms))
                if committed_repair_safety_time_ms.size
                else None
            ),
            "mean_committed_action_safety_time_ms": (
                float(np.mean(committed_action_safety_time_ms))
                if committed_action_safety_time_ms.size
                else None
            ),
            "max_committed_action_safety_time_ms": (
                float(np.max(committed_action_safety_time_ms))
                if committed_action_safety_time_ms.size
                else None
            ),
            "mean_committed_abort_brake_time_ms": (
                float(np.mean(committed_abort_brake_time_ms))
                if committed_abort_brake_time_ms.size
                else None
            ),
            "max_committed_abort_brake_time_ms": (
                float(np.max(committed_abort_brake_time_ms))
                if committed_abort_brake_time_ms.size
                else None
            ),
            "mean_planning_vs_replay_clearance_post_error": (
                float(np.mean(planning_vs_replay_clearance_post_error))
                if planning_vs_replay_clearance_post_error.size
                else None
            ),
            "mean_planning_vs_replay_human_error": (
                float(np.mean(planning_vs_replay_human_error))
                if planning_vs_replay_human_error.size
                else None
            ),
            "mean_actual_vs_planned_post_q_error": (
                float(np.mean(actual_vs_planned_post_q_error))
                if actual_vs_planned_post_q_error.size
                else None
            ),
            "mean_committed_state_error": (
                float(np.mean(committed_state_error))
                if committed_state_error.size
                else None
            ),
            "max_committed_state_error": (
                float(np.max(committed_state_error))
                if committed_state_error.size
                else None
            ),
            "mean_committed_clearance_prediction_error": (
                float(np.mean(committed_clearance_prediction_error))
                if committed_clearance_prediction_error.size
                else None
            ),
            "mean_committed_planned_vs_actual_q_error": (
                float(np.mean(committed_planned_vs_actual_q_error))
                if committed_planned_vs_actual_q_error.size
                else None
            ),
            "mean_committed_human_motion_since_plan": (
                float(np.mean(committed_human_motion_since_plan))
                if committed_human_motion_since_plan.size
                else None
            ),
            "mean_committed_accepted_clearance_margin": (
                float(np.mean(committed_accepted_clearance_margin))
                if committed_accepted_clearance_margin.size
                else None
            ),
            "committed_deform_steps_executed": committed_deform_steps_executed,
            "committed_recover_steps_executed": committed_recover_steps_executed,
            "resume_from_committed_rejoin_count": resume_from_committed_rejoin_count,
            "recovery_action_history_reset_count": recovery_action_history_reset_count,
            "recovery_visual_history_reset_count": recovery_visual_history_reset_count,
            "recovery_visual_history_reset_entries": recovery_visual_history_reset_entries,
            "accepted_recover_chunks_not_executed": accepted_recover_chunks_not_executed,
            "diagnostic_warning": diagnostic_warning,
            "temporary_wait_steps": temporary_wait_steps,
            "resume_after_wait_count": resume_after_wait_count,
            "deform_after_persistent_block_count": deform_after_persistent_block_count,
            "deform_suppressed_by_temporary_wait_count": deform_suppressed_by_temporary_wait_count,
            "recovery_failure_streak_max": recovery_failure_streak_max,
            "deform_replan_count": deform_replan_count,
            "recovery_replan_count": recovery_replan_count,
            "recovery_optimization_skipped_count": recovery_optimization_skipped_count,
            "recovery_optimization_skipped_steps": recovery_optimization_skipped_steps,
            "stale_recovery_suppressed_count": stale_recovery_suppressed_count,
            "recovery_target_infeasible_count": recovery_target_infeasible_count,
            "emergency_brake_steps": emergency_brake_steps,
            "optimized_attempt_count": optimized_attempt_count,
            "optimized_solution_count": optimized_solution_count,
            "fallback_attempt_count": fallback_attempt_count,
            "fallback_attempt_accepted_count": fallback_attempt_accepted_count,
            "optimized_rejected_count": optimized_rejected_count,
            "deform_option_attempt_count": deform_option_attempt_count,
            "deform_accepted_count": deform_accepted_count,
            "deform_rejected_count": deform_rejected_count,
            "recover_option_attempt_count": recover_option_attempt_count,
            "recover_accepted_count": recover_accepted_count,
            "recover_rejected_count": recover_rejected_count,
            "safe_corridor_recovery_count": safe_corridor_recovery_count,
            "direct_rejoin_attempt_count": direct_rejoin_attempt_count,
            "direct_rejoin_reject_count": direct_rejoin_reject_count,
            "detour_rejoin_attempt_count": detour_rejoin_attempt_count,
            "detour_rejoin_accept_count": detour_rejoin_accept_count,
            "delayed_rejoin_count": delayed_rejoin_count,
            "recover_path_unsafe_count": recover_path_unsafe_count,
            "recovery_path_failure_streak_max": recovery_path_failure_streak_max,
            "repeated_unsafe_target_count": repeated_unsafe_target_count,
            "post_recovery_act_window_count": post_recovery_act_window_count,
            "post_recovery_act_window_interrupted_count": post_recovery_act_window_interrupted_count,
            "direct_rejoin_attempted_steps": direct_rejoin_attempted_steps,
            "direct_rejoin_rejected_steps": direct_rejoin_rejected_steps,
            "detour_rejoin_attempted_steps": detour_rejoin_attempted_steps,
            "detour_rejoin_accepted_steps": detour_rejoin_accepted_steps,
            "delayed_rejoin_active_steps": delayed_rejoin_active_steps,
            "repeated_unsafe_target_steps": repeated_unsafe_target_steps,
            "post_recovery_act_window_steps": post_recovery_act_window_steps,
            "post_recovery_act_window_interrupted_steps": post_recovery_act_window_interrupted_steps,
            "recover_reject_reason_counts": {
                reason: recover_reject_reasons.count(reason)
                for reason in sorted(set(recover_reject_reasons))
            },
            "safe_prefix_accepted_count": safe_prefix_accepted_count,
            "first_action_only_accepted_count": first_action_only_accepted_count,
            "immediate_hard_reject_count": immediate_hard_reject_count,
            "no_safe_prefix_reject_count": no_safe_prefix_reject_count,
            "horizon_margin_reject_count": horizon_margin_reject_count,
            "accepted_deform_steps": accepted_deform_steps,
            "accepted_recover_steps": accepted_recover_steps,
            "fallback_brake_after_reject_count": fallback_brake_after_reject_count,
            "nominal_rejoin_available_count": nominal_rejoin_available_count,
            "nominal_rejoin_suppressed_count": nominal_rejoin_suppressed_count,
            "stale_nominal_rejoin_suppressed_count": stale_nominal_rejoin_suppressed_count,
            "nominal_prefix_unsafe_suppressed_count": nominal_prefix_unsafe_suppressed_count,
            "recover_positive_projection_count": recover_positive_projection_count,
            "recover_nonpositive_projection_count": recover_nonpositive_projection_count,
            "mean_recover_projection_on_nominal": (
                float(np.mean(recover_projection_on_nominal))
                if recover_projection_on_nominal.size
                else (
                    float(np.mean(finite_metric("mean_recover_projection_on_nominal")))
                    if finite_metric("mean_recover_projection_on_nominal").size
                    else None
                )
            ),
            "mean_recover_cosine_to_nominal": (
                float(np.mean(recover_cosine_to_nominal))
                if recover_cosine_to_nominal.size
                else (
                    float(np.mean(finite_metric("mean_recover_cosine_to_nominal")))
                    if finite_metric("mean_recover_cosine_to_nominal").size
                    else None
                )
            ),
            "mean_recover_direction_cosine": (
                float(np.mean(recover_direction_cosine))
                if recover_direction_cosine.size
                else (
                    float(np.mean(finite_metric("mean_recover_direction_cosine")))
                    if finite_metric("mean_recover_direction_cosine").size
                    else None
                )
            ),
            "mean_recover_direction_loss": (
                float(np.mean(recover_direction_loss))
                if recover_direction_loss.size
                else (
                    float(np.mean(finite_metric("mean_recover_direction_loss")))
                    if finite_metric("mean_recover_direction_loss").size
                    else None
                )
            ),
            "mean_recover_act_progress_loss": (
                float(np.mean(recover_act_progress_loss))
                if recover_act_progress_loss.size
                else None
            ),
            "mean_recover_act_heading_loss": (
                float(np.mean(recover_act_heading_loss))
                if recover_act_heading_loss.size
                else None
            ),
            "mean_recover_act_direction_loss": (
                float(np.mean(recover_act_direction_loss))
                if recover_act_direction_loss.size
                else None
            ),
            "mean_recover_act_progress_projection": (
                float(np.mean(recover_act_progress_projection))
                if recover_act_progress_projection.size
                else None
            ),
            "mean_recover_act_target_progress": (
                float(np.mean(recover_act_target_progress))
                if recover_act_target_progress.size
                else None
            ),
            "mean_recover_act_heading_cosine": (
                float(np.mean(recover_act_heading_cosine))
                if recover_act_heading_cosine.size
                else None
            ),
            "min_recover_act_heading_cosine": (
                float(np.min(recover_act_heading_cosine))
                if recover_act_heading_cosine.size
                else None
            ),
            "recover_act_progress_ok_count": recover_act_progress_ok_count,
            "recover_act_heading_ok_count": recover_act_heading_ok_count,
            "mean_recover_task_progress_score": (
                float(np.mean(recover_task_progress_score))
                if recover_task_progress_score.size
                else (
                    float(np.mean(finite_metric("mean_recover_task_progress_score")))
                    if finite_metric("mean_recover_task_progress_score").size
                    else None
                )
            ),
            "mean_recover_resume_tube_score": float(np.mean(recover_resume_tube_score)) if recover_resume_tube_score.size else None,
            "min_recover_resume_tube_score": float(np.min(recover_resume_tube_score)) if recover_resume_tube_score.size else None,
            "mean_recover_resume_tube_component_score": float(np.mean(recover_resume_tube_component_score)) if recover_resume_tube_component_score.size else None,
            "mean_recover_resume_tube_terminal_score": float(np.mean(recover_resume_tube_terminal_score)) if recover_resume_tube_terminal_score.size else None,
            "mean_recover_resume_tube_path_score": float(np.mean(recover_resume_tube_path_score)) if recover_resume_tube_path_score.size else None,
            "mean_recover_resume_tube_progress_score": float(np.mean(recover_resume_tube_progress_score)) if recover_resume_tube_progress_score.size else None,
            "mean_recover_resume_tube_heading_score": float(np.mean(recover_resume_tube_heading_score)) if recover_resume_tube_heading_score.size else None,
            "mean_recover_resume_tube_clearance_score": float(np.mean(recover_resume_tube_clearance_score)) if recover_resume_tube_clearance_score.size else None,
            "mean_recover_resume_tube_terminal_dist": float(np.mean(recover_resume_tube_terminal_dist)) if recover_resume_tube_terminal_dist.size else None,
            "mean_recover_resume_tube_ordered_loss": float(np.mean(recover_resume_tube_ordered_loss)) if recover_resume_tube_ordered_loss.size else None,
            "mean_recover_resume_tube_prefix_min_clearance": float(np.mean(recover_resume_tube_prefix_min_clearance)) if recover_resume_tube_prefix_min_clearance.size else None,
            "mean_recover_resume_window_wrist_l2_mean": float(np.mean(recover_resume_window_wrist_l2_mean)) if recover_resume_window_wrist_l2_mean.size else None,
            "max_recover_resume_window_wrist_l2_max": float(np.max(recover_resume_window_wrist_l2_max)) if recover_resume_window_wrist_l2_max.size else None,
            "mean_recover_resume_window_left_wrist_abs_mean": float(np.mean(recover_resume_window_left_wrist_abs_mean)) if recover_resume_window_left_wrist_abs_mean.size else None,
            "max_recover_resume_window_left_wrist_abs_max": float(np.max(recover_resume_window_left_wrist_abs_max)) if recover_resume_window_left_wrist_abs_max.size else None,
            "mean_recover_resume_window_right_wrist_abs_mean": float(np.mean(recover_resume_window_right_wrist_abs_mean)) if recover_resume_window_right_wrist_abs_mean.size else None,
            "max_recover_resume_window_right_wrist_abs_max": float(np.max(recover_resume_window_right_wrist_abs_max)) if recover_resume_window_right_wrist_abs_max.size else None,
            "recover_resume_tube_ok_count": recover_resume_tube_ok_count,
            "recover_resume_tube_reject_count": recover_resume_tube_reject_count,
            "interaction_context_counts": _count_strings([m.interaction_context for m in metrics]),
            "resume_adapter_counts": _count_strings([m.resume_adapter for m in metrics]),
            "resume_affordance_available_count": resume_affordance_available_count,
            "resume_affordance_ok_count": resume_affordance_ok_count,
            "mean_resume_affordance_score": float(np.mean(resume_affordance_score)) if resume_affordance_score.size else None,
            "min_resume_affordance_score": float(np.min(resume_affordance_score)) if resume_affordance_score.size else None,
            "mean_resume_affordance_component_score": float(np.mean(resume_affordance_component_score)) if resume_affordance_component_score.size else None,
            "mean_resume_affordance_target_distance": float(np.mean(resume_affordance_target_distance)) if resume_affordance_target_distance.size else None,
            "mean_resume_affordance_contact_score": float(np.mean(resume_affordance_contact_score)) if resume_affordance_contact_score.size else None,
            "act_resumable_ok_count": act_resumable_ok_count,
            "act_resumable_live_ok_count": act_resumable_live_ok_count,
            "act_resumable_nominal_ok_count": act_resumable_nominal_ok_count,
            "mean_act_resumable_score": float(np.mean(act_resumable_score)) if act_resumable_score.size else None,
            "min_act_resumable_score": float(np.min(act_resumable_score)) if act_resumable_score.size else None,
            "mean_act_resumable_nominal_score": float(np.mean(act_resumable_nominal_score)) if act_resumable_nominal_score.size else None,
            "mean_act_resumable_live_score": float(np.mean(act_resumable_live_score)) if act_resumable_live_score.size else None,
            "act_action_agreement_logged_count": act_action_agreement_logged_count,
            "act_action_agreement_post_recovery_or_reentry_count": act_action_agreement_post_recovery_or_reentry_count,
            "act_action_agreement_context_counts": _count_strings([m.act_action_agreement_context for m in metrics]),
            "mean_act_action_agreement_act_vs_safe_l2": float(np.mean(act_action_agreement_act_vs_safe_l2)) if act_action_agreement_act_vs_safe_l2.size else None,
            "mean_act_action_agreement_act_vs_safe_cosine": float(np.mean(act_action_agreement_act_vs_safe_cosine)) if act_action_agreement_act_vs_safe_cosine.size else None,
            "mean_act_action_agreement_act_vs_safe_arm_l2": float(np.mean(act_action_agreement_act_vs_safe_arm_l2)) if act_action_agreement_act_vs_safe_arm_l2.size else None,
            "mean_act_action_agreement_act_vs_safe_gripper_abs_delta": float(np.mean(act_action_agreement_act_vs_safe_gripper_abs_delta)) if act_action_agreement_act_vs_safe_gripper_abs_delta.size else None,
            "mean_act_action_agreement_act_vs_nominal_l2": float(np.mean(act_action_agreement_act_vs_nominal_l2)) if act_action_agreement_act_vs_nominal_l2.size else None,
            "mean_act_action_agreement_act_vs_nominal_cosine": float(np.mean(act_action_agreement_act_vs_nominal_cosine)) if act_action_agreement_act_vs_nominal_cosine.size else None,
            "mean_act_action_agreement_act_vs_nominal_arm_l2": float(np.mean(act_action_agreement_act_vs_nominal_arm_l2)) if act_action_agreement_act_vs_nominal_arm_l2.size else None,
            "mean_act_action_agreement_act_vs_nominal_gripper_abs_delta": float(np.mean(act_action_agreement_act_vs_nominal_gripper_abs_delta)) if act_action_agreement_act_vs_nominal_gripper_abs_delta.size else None,
            "mean_act_action_agreement_safe_vs_nominal_l2": float(np.mean(act_action_agreement_safe_vs_nominal_l2)) if act_action_agreement_safe_vs_nominal_l2.size else None,
            "mean_act_action_agreement_safe_vs_nominal_cosine": float(np.mean(act_action_agreement_safe_vs_nominal_cosine)) if act_action_agreement_safe_vs_nominal_cosine.size else None,
            "mean_act_action_agreement_safe_vs_nominal_arm_l2": float(np.mean(act_action_agreement_safe_vs_nominal_arm_l2)) if act_action_agreement_safe_vs_nominal_arm_l2.size else None,
            "mean_act_action_agreement_safe_vs_nominal_gripper_abs_delta": float(np.mean(act_action_agreement_safe_vs_nominal_gripper_abs_delta)) if act_action_agreement_safe_vs_nominal_gripper_abs_delta.size else None,
            "mean_act_action_agreement_act_vs_target_l2": float(np.mean(act_action_agreement_act_vs_target_l2)) if act_action_agreement_act_vs_target_l2.size else None,
            "mean_act_action_agreement_act_vs_target_cosine": float(np.mean(act_action_agreement_act_vs_target_cosine)) if act_action_agreement_act_vs_target_cosine.size else None,
            "mean_act_action_agreement_act_vs_target_arm_l2": float(np.mean(act_action_agreement_act_vs_target_arm_l2)) if act_action_agreement_act_vs_target_arm_l2.size else None,
            "mean_act_action_agreement_act_vs_target_gripper_abs_delta": float(np.mean(act_action_agreement_act_vs_target_gripper_abs_delta)) if act_action_agreement_act_vs_target_gripper_abs_delta.size else None,
            "mean_act_action_agreement_safe_vs_target_l2": float(np.mean(act_action_agreement_safe_vs_target_l2)) if act_action_agreement_safe_vs_target_l2.size else None,
            "mean_act_action_agreement_safe_vs_target_cosine": float(np.mean(act_action_agreement_safe_vs_target_cosine)) if act_action_agreement_safe_vs_target_cosine.size else None,
            "mean_act_action_agreement_safe_vs_target_arm_l2": float(np.mean(act_action_agreement_safe_vs_target_arm_l2)) if act_action_agreement_safe_vs_target_arm_l2.size else None,
            "mean_act_action_agreement_safe_vs_target_gripper_abs_delta": float(np.mean(act_action_agreement_safe_vs_target_gripper_abs_delta)) if act_action_agreement_safe_vs_target_gripper_abs_delta.size else None,
            "mean_act_action_agreement_act_vs_last_recovery_l2": float(np.mean(act_action_agreement_act_vs_last_recovery_l2)) if act_action_agreement_act_vs_last_recovery_l2.size else None,
            "mean_act_action_agreement_act_vs_last_recovery_cosine": float(np.mean(act_action_agreement_act_vs_last_recovery_cosine)) if act_action_agreement_act_vs_last_recovery_cosine.size else None,
            "mean_act_action_agreement_act_vs_last_recovery_arm_l2": float(np.mean(act_action_agreement_act_vs_last_recovery_arm_l2)) if act_action_agreement_act_vs_last_recovery_arm_l2.size else None,
            "mean_act_action_agreement_act_vs_last_recovery_gripper_abs_delta": float(np.mean(act_action_agreement_act_vs_last_recovery_gripper_abs_delta)) if act_action_agreement_act_vs_last_recovery_gripper_abs_delta.size else None,
            "mean_act_action_agreement_safe_vs_last_recovery_l2": float(np.mean(act_action_agreement_safe_vs_last_recovery_l2)) if act_action_agreement_safe_vs_last_recovery_l2.size else None,
            "mean_act_action_agreement_safe_vs_last_recovery_cosine": float(np.mean(act_action_agreement_safe_vs_last_recovery_cosine)) if act_action_agreement_safe_vs_last_recovery_cosine.size else None,
            "mean_act_action_agreement_safe_vs_last_recovery_arm_l2": float(np.mean(act_action_agreement_safe_vs_last_recovery_arm_l2)) if act_action_agreement_safe_vs_last_recovery_arm_l2.size else None,
            "mean_act_action_agreement_safe_vs_last_recovery_gripper_abs_delta": float(np.mean(act_action_agreement_safe_vs_last_recovery_gripper_abs_delta)) if act_action_agreement_safe_vs_last_recovery_gripper_abs_delta.size else None,
            "mean_recover_resume_affordance_score": float(np.mean(recover_resume_affordance_score)) if recover_resume_affordance_score.size else None,
            "min_recover_resume_affordance_score": float(np.min(recover_resume_affordance_score)) if recover_resume_affordance_score.size else None,
            "mean_recover_resume_affordance_component_score": float(np.mean(recover_resume_affordance_component_score)) if recover_resume_affordance_component_score.size else None,
            "mean_recover_resume_affordance_target_distance": float(np.mean(recover_resume_affordance_target_distance)) if recover_resume_affordance_target_distance.size else None,
            "mean_recover_resume_affordance_contact_score": float(np.mean(recover_resume_affordance_contact_score)) if recover_resume_affordance_contact_score.size else None,
            "mean_recover_resume_affordance_bonus": float(np.mean(recover_resume_affordance_bonus)) if recover_resume_affordance_bonus.size else None,
            "recover_resume_affordance_ok_count": recover_resume_affordance_ok_count,
            "recover_resume_affordance_reject_count": recover_resume_affordance_reject_count,
            "mean_mpc_handoff_resume_affordance_score": float(np.mean(mpc_handoff_resume_affordance_score)) if mpc_handoff_resume_affordance_score.size else None,
            "mean_mpc_handoff_resume_affordance_component_score": float(np.mean(mpc_handoff_resume_affordance_component_score)) if mpc_handoff_resume_affordance_component_score.size else None,
            "mpc_handoff_resume_affordance_ok_count": mpc_handoff_resume_affordance_ok_count,
            "mean_committed_rejoin_resume_affordance_score": float(np.mean(committed_rejoin_resume_affordance_score)) if committed_rejoin_resume_affordance_score.size else None,
            "mean_committed_rejoin_resume_affordance_component_score": float(np.mean(committed_rejoin_resume_affordance_component_score)) if committed_rejoin_resume_affordance_component_score.size else None,
            "committed_rejoin_resume_affordance_ok_count": committed_rejoin_resume_affordance_ok_count,
            "mean_committed_soft_handoff_resume_affordance_score": float(np.mean(committed_soft_handoff_resume_affordance_score)) if committed_soft_handoff_resume_affordance_score.size else None,
            "mean_recover_ordered_pose_loss": (
                float(np.mean(recover_ordered_pose_loss))
                if recover_ordered_pose_loss.size
                else (
                    float(np.mean(finite_metric("mean_recover_ordered_pose_loss")))
                    if finite_metric("mean_recover_ordered_pose_loss").size
                    else None
                )
            ),
            "mean_recover_ordered_delta_loss": (
                float(np.mean(recover_ordered_delta_loss))
                if recover_ordered_delta_loss.size
                else (
                    float(np.mean(finite_metric("mean_recover_ordered_delta_loss")))
                    if finite_metric("mean_recover_ordered_delta_loss").size
                    else None
                )
            ),
            "mean_recover_ordered_loss": (
                float(np.mean(recover_ordered_loss))
                if recover_ordered_loss.size
                else (
                    float(np.mean(finite_metric("mean_recover_ordered_loss")))
                    if finite_metric("mean_recover_ordered_loss").size
                    else None
                )
            ),
            "optimizer_method_counts": {
                method: sum(
                    1 for metric in metrics if metric.optimizer_method == method
                )
                for method in sorted(
                    {
                        str(metric.optimizer_method)
                        for metric in metrics
                        if metric.optimizer_method is not None
                    }
                )
            },
            "mean_gradient_iterations_run": (
                float(np.mean(gradient_iterations_run))
                if gradient_iterations_run.size
                else None
            ),
            "max_gradient_iterations_run": (
                int(np.max(gradient_iterations_run))
                if gradient_iterations_run.size
                else None
            ),
            "gradient_early_stopped_count": int(
                np.sum([bool(m.gradient_early_stopped) for m in metrics])
            ),
            "gradient_candidate_early_stopped_count": int(
                np.sum([bool(m.gradient_candidate_early_stopped) for m in metrics])
            ),
            "gradient_batched_line_search_count": int(
                np.sum([bool(m.gradient_batched_line_search) for m in metrics])
            ),
            "gradient_jax_scan_used_count": int(
                np.sum([bool(m.gradient_jax_scan_used) for m in metrics])
            ),
            "gradient_full_jax_scan_used_count": int(
                np.sum([bool(m.gradient_full_jax_scan_used) for m in metrics])
            ),
            "fixed_shape_jax_optimizer_loop_count": int(
                np.sum([bool(m.fixed_shape_jax_optimizer_loop) for m in metrics])
            ),
            "jax_scan_cost_kind_counts": {
                kind: sum(
                    1 for metric in metrics if metric.jax_scan_cost_kind == kind
                )
                for kind in sorted(
                    {
                        str(metric.jax_scan_cost_kind)
                        for metric in metrics
                        if metric.jax_scan_cost_kind is not None
                    }
                )
            },
            "mean_gradient_full_jax_scan_time_ms": (
                float(np.mean(gradient_full_jax_scan_time_ms))
                if gradient_full_jax_scan_time_ms.size
                else None
            ),
            "mean_gradient_line_search_batch_evaluations": (
                float(np.mean(gradient_line_search_batch_evaluations))
                if gradient_line_search_batch_evaluations.size
                else None
            ),
            "mean_gradient_jax_scan_used_count": (
                float(np.mean(gradient_jax_scan_used_count))
                if gradient_jax_scan_used_count.size
                else None
            ),
            "mean_optimizer_evaluations": (
                float(np.mean(optimizer_evaluations))
                if optimizer_evaluations.size
                else None
            ),
            "mean_deform_optimizer_time_ms": (
                float(np.mean(deform_optimizer_time_ms))
                if deform_optimizer_time_ms.size
                else None
            ),
            "mean_return_optimizer_time_ms": (
                float(np.mean(return_optimizer_time_ms))
                if return_optimizer_time_ms.size
                else None
            ),
            "mean_explicit_optimizer_time_ms": (
                float(np.mean(explicit_optimizer_time_ms))
                if explicit_optimizer_time_ms.size
                else None
            ),
            "mean_committed_suffix_optimizer_time_ms": (
                float(np.mean(committed_suffix_optimizer_time_ms))
                if committed_suffix_optimizer_time_ms.size
                else None
            ),
            "mean_committed_plan_safety_time_ms": (
                float(np.mean(committed_plan_safety_time_ms))
                if committed_plan_safety_time_ms.size
                else None
            ),
            "mean_committed_plan_diagnostics_time_ms": (
                float(np.mean(committed_plan_diagnostics_time_ms))
                if committed_plan_diagnostics_time_ms.size
                else None
            ),
            "max_committed_plan_diagnostics_time_ms": (
                float(np.max(committed_plan_diagnostics_time_ms))
                if committed_plan_diagnostics_time_ms.size
                else None
            ),
            "mean_cem_iterations_run": (
                float(np.mean(cem_iterations_run)) if cem_iterations_run.size else None
            ),
            "max_cem_iterations_run": (
                int(np.max(cem_iterations_run)) if cem_iterations_run.size else None
            ),
            "cem_early_stopped_count": int(
                np.sum([bool(m.cem_early_stopped) for m in metrics])
            ),
            "mean_deform_cem_iterations_run": (
                float(np.mean(deform_cem_iterations_run))
                if deform_cem_iterations_run.size
                else None
            ),
            "mean_return_cem_iterations_run": (
                float(np.mean(return_cem_iterations_run))
                if return_cem_iterations_run.size
                else None
            ),
            "deform_cem_early_stopped_count": int(
                np.sum([bool(m.deform_cem_early_stopped) for m in metrics])
            ),
            "return_cem_early_stopped_count": int(
                np.sum([bool(m.return_cem_early_stopped) for m in metrics])
            ),
            "hold_unsafe_count": hold_unsafe_count,
            "hold_predicted_contact_count": hold_predicted_contact_count,
            "emergency_deform_away_steps": emergency_deform_away_steps,
            "emergency_deform_away_count": emergency_deform_away_count,
            "contact_during_hold_count": contact_during_hold_count,
            "contact_during_brake_count": contact_during_brake_count,
            "contact_during_deform_count": contact_during_deform_count,
            "contact_during_recover_count": contact_during_recover_count,
            "mean_hold_horizon_min_clearance": (
                float(np.mean(hold_horizon_min_clearance))
                if hold_horizon_min_clearance.size
                else None
            ),
            "min_hold_horizon_min_clearance": (
                float(np.min(hold_horizon_min_clearance))
                if hold_horizon_min_clearance.size
                else None
            ),
            "act_ratio": act_ratio,
            "safety_mode_ratio": safety_mode_ratio,
            "fallback_ratio": fallback_ratio,
            "mean_task_progress": float(np.mean(task_progress)) if task_progress.size else None,
            "max_task_progress": max_task_progress,
            "final_task_progress": final_task_progress,
            "mean_task_progress_delta": float(np.mean(task_progress_delta)) if task_progress_delta.size else None,
            "mean_progress_during_act": mean_progress_for("act_step"),
            "mean_progress_during_deform": mean_progress_for("deform_step"),
            "mean_progress_during_recover": mean_progress_for("recover_step"),
            "mean_progress_during_brake": mean_progress_for("brake_step"),
            "mean_progress_during_fallback": mean_progress_for("fallback_step"),
            "num_progress_regressions": int(np.sum(task_progress_delta < -1e-6)) if task_progress_delta.size else 0,
            "num_large_arm_delta_events": int(
                np.sum([m.arm_delta > large_arm_delta_threshold for m in metrics])
            ),
            "num_large_base_delta_events": int(
                np.sum([m.base_delta > large_base_delta_threshold for m in metrics])
            ),
            "large_arm_delta_threshold": large_arm_delta_threshold,
            "large_base_delta_threshold": large_base_delta_threshold,
            "low_act_ratio_threshold": low_act_ratio_threshold,
            "high_fallback_ratio_threshold": high_fallback_ratio_threshold,
            "task_success_threshold": success_threshold,
            "likely_failure_cause": likely_failure_cause,
            "pass_through_rate": rate(modes, "pass_through"),
            "horizon_brake_rate": rate(modes, "horizon_brake"),
            "path_consistent_brake_rate": rate(modes, "path_consistent_brake"),
            "path_consistent_brake_intended_rate": rate(modes, "path_consistent_brake_intended_step"),
            "horizon_brake_intended_rate": rate(modes, "horizon_brake_intended_step"),
            "verified_failsafe_rate": rate(modes, "verified_failsafe"),
            "unverified_emergency_failsafe_rate": rate(modes, "unverified_emergency_failsafe"),
            "horizon_deform_rate": rate(modes, "horizon_deform"),
            "sequential_oscbf_rate": rate(modes, "sequential_oscbf"),
            "pause_on_unsafe_rate": rate(modes, "pause_on_unsafe"),
            "pause_and_restart_rate": rate(modes, "pause_and_restart"),
            "phase_reanchor_rate": rate(modes, "phase_reanchor"),
            "phase_reanchor_source_rate": rate(sources, "phase_reanchor"),
            "phase_reanchor_steps": int(np.sum(phase_reanchor_steps)),
            "gripper_latched_steps": gripper_latched_steps,
            "post_recovery_task_guard_steps": post_recovery_task_guard_steps,
            "post_recovery_reanchor_started_count": post_recovery_reanchor_started_count,
            "post_recovery_progress_regression_count": post_recovery_progress_regression_count,
            "post_recovery_mid_progress_no_progress_count": post_recovery_mid_progress_no_progress_count,
            "post_recovery_mid_progress_triggered_count": post_recovery_mid_progress_triggered_count,
            "post_recovery_mid_progress_reseed_count": post_recovery_mid_progress_reseed_count,
            "pause_current_clearance_rate": rate(pause_reasons, "current_clearance"),
            "pause_horizon_clearance_rate": rate(pause_reasons, "horizon_clearance"),
            "pause_deform_clearance_rate": rate(pause_reasons, "deform_clearance"),
            "chunk_deform_source_rate": rate(sources, "chunk_deform"),
            "sequential_oscbf_source_rate": rate(sources, "sequential_oscbf"),
            "min_robot_human_distance": float(np.min(robot_human_distances)) if robot_human_distances else None,
            "num_safety_violations": int(np.sum(safety_violations)),
            "num_filter_activations": int(np.sum(chunk_interventions)),
            "total_brake_steps": int(np.sum(brake_steps)),
            "total_deformation_steps": int(np.sum(deformation_steps)),
            "task_success": bool(any(m.success for m in metrics)),
            "drawer_open_distance": float(drawer_open_distances[-1]) if drawer_open_distances else None,
            "resume_latency_after_human_exit": _resume_latency_after_human_exit(metrics),
        }
    )
    return summary


def summarise_all_chunk_episodes(episode_summaries: list[dict]) -> dict:
    summary = summarise_all_episodes(episode_summaries)

    def mean_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def max_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.max(vals)) if vals else None

    def min_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.min(vals)) if vals else None

    def sum_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return int(np.sum(vals)) if vals else None

    def _merge_count_dicts(summaries, key):
        merged = {}
        for item in summaries:
            counts = item.get(key)
            if not isinstance(counts, dict):
                continue
            for name, value in counts.items():
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                merged[str(name)] = merged.get(str(name), 0) + value
        return merged

    summary.update(
        {
            "mean_chunk_arm_delta": mean_of("mean_chunk_arm_delta"),
            "max_chunk_arm_delta_over_episodes": max_of("max_chunk_arm_delta"),
            "mean_chunk_intervention_frequency": mean_of("chunk_intervention_frequency"),
            "mean_chunk_modified_fraction": mean_of("mean_chunk_modified_fraction"),
            "mean_chunk_modified_steps": mean_of("mean_chunk_modified_steps"),
            "mean_chunk_first_modified_step": mean_of("mean_chunk_first_modified_step"),
            "mean_chunk_mean_step_arm_delta": mean_of("mean_chunk_mean_step_arm_delta"),
            "max_chunk_step_arm_delta_over_episodes": max_of("max_chunk_step_arm_delta"),
            "mean_chunk_future_edit_fraction": mean_of("mean_chunk_future_edit_fraction"),
            "mean_chunk_first_edit_fraction": mean_of("mean_chunk_first_edit_fraction"),
            "mean_chunk_safe_arm_variation": mean_of("mean_chunk_safe_arm_variation"),
            "mean_chunk_nominal_arm_variation": mean_of("mean_chunk_nominal_arm_variation"),
            "mean_chunk_arm_variation_delta": mean_of("mean_chunk_arm_variation_delta"),
            "mean_chunk_edit_variation": mean_of("mean_chunk_edit_variation"),
            "mean_path_deviation": mean_of("mean_path_deviation"),
            "max_path_deviation_over_episodes": max_of("max_path_deviation"),
            "mean_final_path_deviation": mean_of("mean_final_path_deviation"),
            "mean_chunk_preemptive_intervention_frequency": mean_of("chunk_preemptive_intervention_frequency"),
            "mean_horizon_risk_gap": mean_of("mean_horizon_risk_gap"),
            "max_horizon_risk_gap_over_episodes": max_of("max_horizon_risk_gap"),
            "mean_horizon_clearance_drop": mean_of("mean_horizon_clearance_drop"),
            "mean_horizon_risk_gap_rate": mean_of("horizon_risk_gap_rate"),
            "mean_horizon_only_risk_rate": mean_of("horizon_only_risk_rate"),
            "mean_deformation_norm": mean_of("mean_deformation_norm"),
            "mean_deform_envelope_loss": mean_of("mean_deform_envelope_loss"),
            "max_deform_envelope_loss_over_episodes": max_of("max_deform_envelope_loss"),
            "mean_deform_envelope_first_delta": mean_of("mean_deform_envelope_first_delta"),
            "max_deform_envelope_first_delta_over_episodes": max_of("max_deform_envelope_first_delta"),
            "mean_deform_envelope_first_violation": mean_of("mean_deform_envelope_first_violation"),
            "max_deform_envelope_first_violation_over_episodes": max_of("max_deform_envelope_first_violation"),
            "mean_deform_envelope_avoid_rate_loss": mean_of("mean_deform_envelope_avoid_rate_loss"),
            "mean_deform_envelope_return_rate_loss": mean_of("mean_deform_envelope_return_rate_loss"),
            "mean_deform_envelope_max_rate": mean_of("mean_deform_envelope_max_rate"),
            "max_deform_envelope_max_rate_over_episodes": max_of("max_deform_envelope_max_rate"),
            "mean_deform_envelope_terminal_delta": mean_of("mean_deform_envelope_terminal_delta"),
            "max_deform_envelope_terminal_delta_over_episodes": max_of("max_deform_envelope_terminal_delta"),
            "mean_deform_envelope_terminal_violation": mean_of("mean_deform_envelope_terminal_violation"),
            "max_deform_envelope_terminal_violation_over_episodes": max_of("max_deform_envelope_terminal_violation"),
            "mean_deform_envelope_terminal_loss": mean_of("mean_deform_envelope_terminal_loss"),
            "mean_deform_envelope_acceleration_loss": mean_of("mean_deform_envelope_acceleration_loss"),
            "mean_deform_safe_rate": mean_of("deform_safe_rate"),
            "optimized_attempts": sum_of("optimized_attempts"),
            "optimized_safe_count": sum_of("optimized_safe_count"),
            "optimized_recoverable_count": sum_of("optimized_recoverable_count"),
            "optimized_accepted_count": sum_of("optimized_accepted_count"),
            "rejected_unsafe_count": sum_of("rejected_unsafe_count"),
            "rejected_unrecoverable_count": sum_of("rejected_unrecoverable_count"),
            "rejected_both_count": sum_of("rejected_both_count"),
            "fallback_used_count": sum_of("fallback_used_count"),
            "deform_stage_accepted_count": sum_of("deform_stage_accepted_count"),
            "recover_accepted_count": sum_of("recover_accepted_count"),
            "recovery_optimization_skipped_count": sum_of(
                "recovery_optimization_skipped_count"
            ),
            "recovery_optimization_skipped_steps": sum_of(
                "recovery_optimization_skipped_steps"
            ),
            "cached_motion_active_count": sum_of("cached_motion_active_count"),
            "resumed_from_cached_count": sum_of("resumed_from_cached_count"),
            "mean_deform_stage_min_clearance": mean_of("mean_deform_stage_min_clearance"),
            "mean_recover_min_clearance": mean_of("mean_recover_min_clearance"),
            "mean_recover_rejoin_loss": mean_of("mean_recover_rejoin_loss"),
            "mean_recover_path_min_clearance": mean_of("mean_recover_path_min_clearance"),
            "min_recover_path_min_clearance": min_of("min_recover_path_min_clearance"),
            "mean_recover_immediate_clearance": mean_of("mean_recover_immediate_clearance"),
            "mean_recover_prefix_min_clearance": mean_of("mean_recover_prefix_min_clearance"),
            "total_optimized_attempt_count": sum_of("optimized_attempt_count"),
            "mean_optimized_accept_rate": mean_of("optimized_accept_rate"),
            "mean_recoverable_rate": mean_of("recoverable_rate"),
            "mean_fallback_used_rate": mean_of("fallback_used_rate"),
            "mean_q_rejoin_dist": mean_of("mean_q_rejoin_dist"),
            "max_q_rejoin_dist_over_episodes": max_of("max_q_rejoin_dist"),
            "mean_qd_rejoin_dist": mean_of("mean_qd_rejoin_dist"),
            "max_qd_rejoin_dist_over_episodes": max_of("max_qd_rejoin_dist"),
            "mean_ee_rejoin_dist": mean_of("mean_ee_rejoin_dist"),
            "max_ee_rejoin_dist_over_episodes": max_of("max_ee_rejoin_dist"),
            "mean_rejoin_q_eval_time_ms": mean_of("mean_rejoin_q_eval_time_ms"),
            "mean_rejoin_qd_eval_time_ms": mean_of("mean_rejoin_qd_eval_time_ms"),
            "mean_env_step_time_ms": mean_of("mean_env_step_time_ms"),
            "p50_env_step_time_ms": mean_of("p50_env_step_time_ms"),
            "p95_env_step_time_ms": mean_of("p95_env_step_time_ms"),
            "max_env_step_time_ms_over_episodes": max_of("max_env_step_time_ms"),
            "mean_policy_obs_adapt_time_ms": mean_of("mean_policy_obs_adapt_time_ms"),
            "p50_policy_obs_adapt_time_ms": mean_of("p50_policy_obs_adapt_time_ms"),
            "p95_policy_obs_adapt_time_ms": mean_of("p95_policy_obs_adapt_time_ms"),
            "max_policy_obs_adapt_time_ms_over_episodes": max_of("max_policy_obs_adapt_time_ms"),
            "mean_policy_action_time_ms": mean_of("mean_policy_action_time_ms"),
            "p50_policy_action_time_ms": mean_of("p50_policy_action_time_ms"),
            "p95_policy_action_time_ms": mean_of("p95_policy_action_time_ms"),
            "max_policy_action_time_ms_over_episodes": max_of("max_policy_action_time_ms"),
            "mean_policy_obs_update_time_ms": mean_of("mean_policy_obs_update_time_ms"),
            "p50_policy_obs_update_time_ms": mean_of("p50_policy_obs_update_time_ms"),
            "p95_policy_obs_update_time_ms": mean_of("p95_policy_obs_update_time_ms"),
            "max_policy_obs_update_time_ms_over_episodes": max_of("max_policy_obs_update_time_ms"),
            "mean_policy_total_time_ms": mean_of("mean_policy_total_time_ms"),
            "p50_policy_total_time_ms": mean_of("p50_policy_total_time_ms"),
            "p95_policy_total_time_ms": mean_of("p95_policy_total_time_ms"),
            "max_policy_total_time_ms_over_episodes": max_of("max_policy_total_time_ms"),
            "mean_latency_residual_ms": mean_of("mean_latency_residual_ms"),
            "p50_latency_residual_ms": mean_of("p50_latency_residual_ms"),
            "p95_latency_residual_ms": mean_of("p95_latency_residual_ms"),
            "max_latency_residual_ms_over_episodes": max_of("max_latency_residual_ms"),
            "mean_ee_nom_cache_time_ms": mean_of("mean_ee_nom_cache_time_ms"),
            "mean_ee_final_check_time_ms": mean_of("mean_ee_final_check_time_ms"),
            "act_steps": sum_of("act_steps"),
            "deform_steps": sum_of("deform_steps"),
            "recover_steps": sum_of("recover_steps"),
            "brake_steps": sum_of("brake_steps"),
            "fallback_steps": sum_of("fallback_steps"),
            "pacs_background_check_only_steps": sum_of("pacs_background_check_only_steps"),
            "pacs_background_brake_steps": sum_of("pacs_background_brake_steps"),
            "pacs_background_act_steps": sum_of("pacs_background_act_steps"),
            "mean_pacs_background_check_only_rate": mean_of("pacs_background_check_only_rate"),
            "mean_pacs_background_brake_rate": mean_of("pacs_background_brake_rate"),
            "mean_pacs_background_act_rate": mean_of("pacs_background_act_rate"),
            "mean_pacs_background_min_clearance": mean_of("mean_pacs_background_min_clearance"),
            "min_pacs_background_min_clearance": min_of("min_pacs_background_min_clearance"),
            "mean_pacs_background_arm_delta": mean_of("mean_pacs_background_arm_delta"),
            "mean_pacs_background_chunk_arm_delta": mean_of("mean_pacs_background_chunk_arm_delta"),
            "mean_pacs_background_chunk_modified_fraction": mean_of("mean_pacs_background_chunk_modified_fraction"),
            "mean_pacs_background_retiming_arm_delta": mean_of("mean_pacs_background_retiming_arm_delta"),
            "mean_pacs_background_retiming_chunk_arm_delta": mean_of("mean_pacs_background_retiming_chunk_arm_delta"),
            "mean_pacs_background_retiming_changed_fraction": mean_of("mean_pacs_background_retiming_changed_fraction"),
            "mean_pacs_background_path_consistent_brake_intended_rate": mean_of("pacs_background_path_consistent_brake_intended_rate"),
            "mean_pacs_background_verified_failsafe_rate": mean_of("pacs_background_verified_failsafe_rate"),
            "mean_pacs_background_unverified_emergency_failsafe_rate": mean_of("pacs_background_unverified_emergency_failsafe_rate"),
            "optimized_attempt_steps": sum_of("optimized_attempt_steps"),
            "optimized_accepted_steps": sum_of("optimized_accepted_steps"),
            "committed_chunk_started_count": sum_of("committed_chunk_started_count"),
            "committed_chunk_completed_count": sum_of("committed_chunk_completed_count"),
            "committed_chunk_abort_count": sum_of("committed_chunk_abort_count"),
            "committed_repaired_step_count": sum_of("committed_repaired_step_count"),
            "committed_abort_due_to_human_motion_count": sum_of("committed_abort_due_to_human_motion_count"),
            "committed_abort_due_to_prediction_error_count": sum_of("committed_abort_due_to_prediction_error_count"),
            "committed_abort_due_to_safety_semantics_mismatch_count": sum_of("committed_abort_due_to_safety_semantics_mismatch_count"),
            "committed_state_mismatch_abort_count": sum_of("committed_state_mismatch_abort_count"),
            "committed_state_mismatch_recovered_count": sum_of("committed_state_mismatch_recovered_count"),
            "committed_suffix_replan_attempt_count": sum_of("committed_suffix_replan_attempt_count"),
            "committed_suffix_replan_accepted_count": sum_of("committed_suffix_replan_accepted_count"),
            "committed_suffix_replan_rejected_count": sum_of("committed_suffix_replan_rejected_count"),
            "committed_opportunistic_resume_count": sum_of("committed_opportunistic_resume_count"),
            "mpc_recovery_active_steps": sum_of("mpc_recovery_active_steps"),
            "mpc_recovery_replan_attempt_count": sum_of("mpc_recovery_replan_attempt_count"),
            "mpc_recovery_replan_accepted_count": sum_of("mpc_recovery_replan_accepted_count"),
            "mpc_recovery_replan_rejected_count": sum_of("mpc_recovery_replan_rejected_count"),
            "mpc_recovery_replan_count": sum_of("mpc_recovery_replan_count"),
            "mpc_recovery_accepted_count": sum_of("mpc_recovery_accepted_count"),
            "mpc_recovery_rejected_count": sum_of("mpc_recovery_rejected_count"),
            "mpc_recovery_no_progress_reject_count": sum_of("mpc_recovery_no_progress_reject_count"),
            "mpc_recovery_budget_escape_count": sum_of("mpc_recovery_budget_escape_count"),
            "mpc_recovery_budget_escape_steps": sum_of("mpc_recovery_budget_escape_steps"),
            "mean_mpc_recovery_target_tube_window_wrist_l2_mean": mean_of("mean_mpc_recovery_target_tube_window_wrist_l2_mean"),
            "max_mpc_recovery_target_tube_window_wrist_l2_max": max_of("max_mpc_recovery_target_tube_window_wrist_l2_max"),
            "mean_mpc_recovery_target_tube_window_left_wrist_abs_mean": mean_of("mean_mpc_recovery_target_tube_window_left_wrist_abs_mean"),
            "max_mpc_recovery_target_tube_window_left_wrist_abs_max": max_of("max_mpc_recovery_target_tube_window_left_wrist_abs_max"),
            "mean_mpc_recovery_target_tube_window_right_wrist_abs_mean": mean_of("mean_mpc_recovery_target_tube_window_right_wrist_abs_mean"),
            "max_mpc_recovery_target_tube_window_right_wrist_abs_max": max_of("max_mpc_recovery_target_tube_window_right_wrist_abs_max"),
            "recovery_budget_extended_count": sum_of("recovery_budget_extended_count"),
            "recovery_budget_extended_steps": sum_of("recovery_budget_extended_steps"),
            "recovery_budget_progress_ok_rate": mean_of("recovery_budget_progress_ok_rate"),
            "mean_recovery_budget_progress_delta": mean_of("mean_recovery_budget_progress_delta"),
            "mean_recovery_budget_live_q_dist": mean_of("mean_recovery_budget_live_q_dist"),
            "staged_recovery_ordered_path_softened_steps": sum_of("staged_recovery_ordered_path_softened_steps"),
            "staged_recovery_progress_accepted_steps": sum_of("staged_recovery_progress_accepted_steps"),
            "recover_handover_ready_steps": sum_of("recover_handover_ready_steps"),
            "recover_progress_only_steps": sum_of("recover_progress_only_steps"),
            "recovery_handover_pending_steps": sum_of("recovery_handover_pending_steps"),
            "mpc_recovery_prefix_replay_steps": sum_of("mpc_recovery_prefix_replay_steps"),
            "committed_state_mismatch_ignored_for_mpc_prefix_count": sum_of("committed_state_mismatch_ignored_for_mpc_prefix_count"),
            "mean_mpc_recovery_live_q_dist_before": mean_of("mean_mpc_recovery_live_q_dist_before"),
            "mean_mpc_recovery_live_q_dist_after": mean_of("mean_mpc_recovery_live_q_dist_after"),
            "mean_mpc_recovery_live_q_progress_delta": mean_of("mean_mpc_recovery_live_q_progress_delta"),
            "min_mpc_recovery_live_q_progress_delta": min_of("min_mpc_recovery_live_q_progress_delta"),
            "mpc_recovery_live_progress_ok_rate": mean_of("mpc_recovery_live_progress_ok_rate"),
            "mpc_recovery_replan_reject_reason_counts": _merge_count_dicts(
                episode_summaries,
                "mpc_recovery_replan_reject_reason_counts",
            ),
            "mpc_handoff_attempt_count": sum_of("mpc_handoff_attempt_count"),
            "mpc_handoff_accepted_count": sum_of("mpc_handoff_accepted_count"),
            "mpc_handoff_rejected_count": sum_of("mpc_handoff_rejected_count"),
            "mpc_handoff_reject_reason_counts": _merge_count_dicts(
                episode_summaries,
                "mpc_handoff_reject_reason_counts",
            ),
            "mpc_handoff_action_agreement_ok_count": sum_of("mpc_handoff_action_agreement_ok_count"),
            "mpc_handoff_action_agreement_override_allowed_count": sum_of("mpc_handoff_action_agreement_override_allowed_count"),
            "mpc_handoff_heading_overridden_by_action_agreement_count": sum_of("mpc_handoff_heading_overridden_by_action_agreement_count"),
            "mpc_handoff_progress_overridden_by_action_agreement_count": sum_of("mpc_handoff_progress_overridden_by_action_agreement_count"),
            "mpc_handoff_action_agreement_override_reason_counts": _merge_count_dicts(
                episode_summaries,
                "mpc_handoff_action_agreement_override_reason_counts",
            ),
            "mean_mpc_handoff_act_vs_release_action_l2": mean_of("mean_mpc_handoff_act_vs_release_action_l2"),
            "mean_mpc_handoff_act_vs_release_action_cosine": mean_of("mean_mpc_handoff_act_vs_release_action_cosine"),
            "mean_mpc_handoff_act_vs_release_action_arm_l2": mean_of("mean_mpc_handoff_act_vs_release_action_arm_l2"),
            "mean_mpc_handoff_act_vs_target_action_l2": mean_of("mean_mpc_handoff_act_vs_target_action_l2"),
            "mean_mpc_handoff_act_vs_target_action_cosine": mean_of("mean_mpc_handoff_act_vs_target_action_cosine"),
            "mean_mpc_handoff_act_vs_target_action_arm_l2": mean_of("mean_mpc_handoff_act_vs_target_action_arm_l2"),
            "mean_mpc_handoff_resume_tube_score": mean_of("mean_mpc_handoff_resume_tube_score"),
            "min_mpc_handoff_resume_tube_score": min_of("min_mpc_handoff_resume_tube_score"),
            "mean_mpc_handoff_resume_tube_component_score": mean_of("mean_mpc_handoff_resume_tube_component_score"),
            "mean_mpc_handoff_resume_tube_terminal_score": mean_of("mean_mpc_handoff_resume_tube_terminal_score"),
            "mean_mpc_handoff_resume_tube_path_score": mean_of("mean_mpc_handoff_resume_tube_path_score"),
            "mean_mpc_handoff_resume_tube_progress_score": mean_of("mean_mpc_handoff_resume_tube_progress_score"),
            "mean_mpc_handoff_resume_tube_heading_score": mean_of("mean_mpc_handoff_resume_tube_heading_score"),
            "mean_mpc_handoff_resume_tube_clearance_score": mean_of("mean_mpc_handoff_resume_tube_clearance_score"),
            "mean_mpc_handoff_resume_tube_terminal_dist": mean_of("mean_mpc_handoff_resume_tube_terminal_dist"),
            "mean_mpc_handoff_resume_tube_ordered_loss": mean_of("mean_mpc_handoff_resume_tube_ordered_loss"),
            "mean_mpc_handoff_resume_tube_prefix_min_clearance": mean_of("mean_mpc_handoff_resume_tube_prefix_min_clearance"),
            "mpc_handoff_resume_tube_ok_count": sum_of("mpc_handoff_resume_tube_ok_count"),
            "mpc_handoff_resume_tube_reject_count": sum_of("mpc_handoff_resume_tube_reject_count"),
            "committed_released_for_act_resume_count": sum_of("committed_released_for_act_resume_count"),
            "mean_committed_soft_handoff_resume_tube_score": mean_of("mean_committed_soft_handoff_resume_tube_score"),
            "mean_committed_soft_handoff_resume_tube_component_score": mean_of("mean_committed_soft_handoff_resume_tube_component_score"),
            "committed_soft_handoff_resume_tube_ok_count": sum_of("committed_soft_handoff_resume_tube_ok_count"),
            "mean_committed_rejoin_resume_tube_score": mean_of("mean_committed_rejoin_resume_tube_score"),
            "min_committed_rejoin_resume_tube_score": min_of("min_committed_rejoin_resume_tube_score"),
            "mean_committed_rejoin_resume_tube_component_score": mean_of("mean_committed_rejoin_resume_tube_component_score"),
            "mean_committed_rejoin_resume_tube_terminal_score": mean_of("mean_committed_rejoin_resume_tube_terminal_score"),
            "mean_committed_rejoin_resume_tube_path_score": mean_of("mean_committed_rejoin_resume_tube_path_score"),
            "mean_committed_rejoin_resume_tube_progress_score": mean_of("mean_committed_rejoin_resume_tube_progress_score"),
            "mean_committed_rejoin_resume_tube_heading_score": mean_of("mean_committed_rejoin_resume_tube_heading_score"),
            "mean_committed_rejoin_resume_tube_clearance_score": mean_of("mean_committed_rejoin_resume_tube_clearance_score"),
            "mean_committed_rejoin_resume_tube_terminal_dist": mean_of("mean_committed_rejoin_resume_tube_terminal_dist"),
            "mean_committed_rejoin_resume_tube_ordered_loss": mean_of("mean_committed_rejoin_resume_tube_ordered_loss"),
            "mean_committed_rejoin_resume_tube_prefix_min_clearance": mean_of("mean_committed_rejoin_resume_tube_prefix_min_clearance"),
            "committed_rejoin_resume_tube_ok_count": sum_of("committed_rejoin_resume_tube_ok_count"),
            "committed_rejoin_resume_tube_reject_count": sum_of("committed_rejoin_resume_tube_reject_count"),
            "committed_recovery_budget_exit_count": sum_of("committed_recovery_budget_exit_count"),
            "committed_replan_due_to_recovery_budget_count": sum_of("committed_replan_due_to_recovery_budget_count"),
            "committed_suffix_replan_reject_reason_counts": _merge_count_dicts(
                episode_summaries,
                "committed_suffix_replan_reject_reason_counts",
            ),
            "mean_committed_repair_time_ms": mean_of("mean_committed_repair_time_ms"),
            "max_committed_repair_time_ms": max_of("max_committed_repair_time_ms"),
            "mean_committed_repair_safety_time_ms": mean_of("mean_committed_repair_safety_time_ms"),
            "max_committed_repair_safety_time_ms": max_of("max_committed_repair_safety_time_ms"),
            "mean_committed_action_safety_time_ms": mean_of("mean_committed_action_safety_time_ms"),
            "max_committed_action_safety_time_ms": max_of("max_committed_action_safety_time_ms"),
            "mean_committed_abort_brake_time_ms": mean_of("mean_committed_abort_brake_time_ms"),
            "max_committed_abort_brake_time_ms": max_of("max_committed_abort_brake_time_ms"),
            "mean_planning_vs_replay_clearance_post_error": mean_of("mean_planning_vs_replay_clearance_post_error"),
            "mean_planning_vs_replay_human_error": mean_of("mean_planning_vs_replay_human_error"),
            "mean_actual_vs_planned_post_q_error": mean_of("mean_actual_vs_planned_post_q_error"),
            "mean_committed_state_error": mean_of("mean_committed_state_error"),
            "max_committed_state_error": max_of("max_committed_state_error"),
            "mean_committed_clearance_prediction_error": mean_of("mean_committed_clearance_prediction_error"),
            "mean_committed_planned_vs_actual_q_error": mean_of("mean_committed_planned_vs_actual_q_error"),
            "mean_committed_human_motion_since_plan": mean_of("mean_committed_human_motion_since_plan"),
            "mean_committed_accepted_clearance_margin": mean_of("mean_committed_accepted_clearance_margin"),
            "committed_deform_steps_executed": sum_of("committed_deform_steps_executed"),
            "committed_recover_steps_executed": sum_of("committed_recover_steps_executed"),
            "resume_from_committed_rejoin_count": sum_of("resume_from_committed_rejoin_count"),
            "recovery_action_history_reset_count": sum_of("recovery_action_history_reset_count"),
            "post_recovery_mid_progress_reseed_count": sum_of(
                "post_recovery_mid_progress_reseed_count"
            ),
            "recovery_visual_history_reset_count": sum_of("recovery_visual_history_reset_count"),
            "recovery_visual_history_reset_entries": sum_of("recovery_visual_history_reset_entries"),
            "accepted_recover_chunks_not_executed": any(
                bool(s.get("accepted_recover_chunks_not_executed"))
                for s in episode_summaries
            ),
            "diagnostic_warning": (
                "accepted_recover_chunks_not_executed"
                if any(bool(s.get("accepted_recover_chunks_not_executed")) for s in episode_summaries)
                else None
            ),
            "temporary_wait_steps": sum_of("temporary_wait_steps"),
            "resume_after_wait_count": sum_of("resume_after_wait_count"),
            "deform_after_persistent_block_count": sum_of("deform_after_persistent_block_count"),
            "deform_suppressed_by_temporary_wait_count": sum_of("deform_suppressed_by_temporary_wait_count"),
            "recovery_failure_streak_max": max_of("recovery_failure_streak_max"),
            "deform_replan_count": max_of("deform_replan_count"),
            "recovery_replan_count": max_of("recovery_replan_count"),
            "stale_recovery_suppressed_count": max_of("stale_recovery_suppressed_count"),
            "recovery_target_infeasible_count": max_of("recovery_target_infeasible_count"),
            "emergency_brake_steps": max_of("emergency_brake_steps"),
            "optimized_attempt_count": max_of("optimized_attempt_count"),
            "optimized_solution_count": max_of("optimized_solution_count"),
            "fallback_attempt_count": max_of("fallback_attempt_count"),
            "fallback_attempt_accepted_count": max_of("fallback_attempt_accepted_count"),
            "optimized_rejected_count": max_of("optimized_rejected_count"),
            "deform_option_attempt_count": max_of("deform_option_attempt_count"),
            "deform_accepted_count": max_of("deform_accepted_count"),
            "deform_rejected_count": max_of("deform_rejected_count"),
            "recover_option_attempt_count": max_of("recover_option_attempt_count"),
            "recover_accepted_count": max_of("recover_accepted_count"),
            "recover_rejected_count": max_of("recover_rejected_count"),
            "safe_corridor_recovery_count": max_of("safe_corridor_recovery_count"),
            "direct_rejoin_attempt_count": max_of("direct_rejoin_attempt_count"),
            "direct_rejoin_reject_count": max_of("direct_rejoin_reject_count"),
            "detour_rejoin_attempt_count": max_of("detour_rejoin_attempt_count"),
            "detour_rejoin_accept_count": max_of("detour_rejoin_accept_count"),
            "delayed_rejoin_count": max_of("delayed_rejoin_count"),
            "recover_path_unsafe_count": max_of("recover_path_unsafe_count"),
            "recovery_path_failure_streak_max": max_of("recovery_path_failure_streak_max"),
            "repeated_unsafe_target_count": max_of("repeated_unsafe_target_count"),
            "post_recovery_act_window_count": max_of("post_recovery_act_window_count"),
            "post_recovery_act_window_interrupted_count": max_of("post_recovery_act_window_interrupted_count"),
            "direct_rejoin_attempted_steps": sum_of("direct_rejoin_attempted_steps"),
            "direct_rejoin_rejected_steps": sum_of("direct_rejoin_rejected_steps"),
            "detour_rejoin_attempted_steps": sum_of("detour_rejoin_attempted_steps"),
            "detour_rejoin_accepted_steps": sum_of("detour_rejoin_accepted_steps"),
            "delayed_rejoin_active_steps": sum_of("delayed_rejoin_active_steps"),
            "repeated_unsafe_target_steps": sum_of("repeated_unsafe_target_steps"),
            "post_recovery_act_window_steps": sum_of("post_recovery_act_window_steps"),
            "post_recovery_act_window_interrupted_steps": sum_of("post_recovery_act_window_interrupted_steps"),
            "safe_prefix_accepted_count": max_of("safe_prefix_accepted_count"),
            "first_action_only_accepted_count": max_of("first_action_only_accepted_count"),
            "immediate_hard_reject_count": max_of("immediate_hard_reject_count"),
            "no_safe_prefix_reject_count": max_of("no_safe_prefix_reject_count"),
            "horizon_margin_reject_count": max_of("horizon_margin_reject_count"),
            "accepted_deform_steps": max_of("accepted_deform_steps"),
            "accepted_recover_steps": max_of("accepted_recover_steps"),
            "fallback_brake_after_reject_count": max_of("fallback_brake_after_reject_count"),
            "nominal_rejoin_available_count": max_of("nominal_rejoin_available_count"),
            "nominal_rejoin_suppressed_count": max_of("nominal_rejoin_suppressed_count"),
            "stale_nominal_rejoin_suppressed_count": max_of("stale_nominal_rejoin_suppressed_count"),
            "nominal_prefix_unsafe_suppressed_count": max_of("nominal_prefix_unsafe_suppressed_count"),
            "recover_positive_projection_count": max_of("recover_positive_projection_count"),
            "recover_nonpositive_projection_count": max_of("recover_nonpositive_projection_count"),
            "mean_recover_projection_on_nominal": mean_of("mean_recover_projection_on_nominal"),
            "mean_recover_cosine_to_nominal": mean_of("mean_recover_cosine_to_nominal"),
            "mean_recover_direction_cosine": mean_of("mean_recover_direction_cosine"),
            "mean_recover_direction_loss": mean_of("mean_recover_direction_loss"),
            "mean_recover_act_progress_loss": mean_of("mean_recover_act_progress_loss"),
            "mean_recover_act_heading_loss": mean_of("mean_recover_act_heading_loss"),
            "mean_recover_act_direction_loss": mean_of("mean_recover_act_direction_loss"),
            "mean_recover_act_progress_projection": mean_of("mean_recover_act_progress_projection"),
            "mean_recover_act_target_progress": mean_of("mean_recover_act_target_progress"),
            "mean_recover_act_heading_cosine": mean_of("mean_recover_act_heading_cosine"),
            "min_recover_act_heading_cosine": min_of("min_recover_act_heading_cosine"),
            "recover_act_progress_ok_count": max_of("recover_act_progress_ok_count"),
            "recover_act_heading_ok_count": max_of("recover_act_heading_ok_count"),
            "mean_recover_task_progress_score": mean_of("mean_recover_task_progress_score"),
            "mean_recover_resume_tube_score": mean_of("mean_recover_resume_tube_score"),
            "min_recover_resume_tube_score": min_of("min_recover_resume_tube_score"),
            "mean_recover_resume_tube_component_score": mean_of("mean_recover_resume_tube_component_score"),
            "mean_recover_resume_tube_terminal_score": mean_of("mean_recover_resume_tube_terminal_score"),
            "mean_recover_resume_tube_path_score": mean_of("mean_recover_resume_tube_path_score"),
            "mean_recover_resume_tube_progress_score": mean_of("mean_recover_resume_tube_progress_score"),
            "mean_recover_resume_tube_heading_score": mean_of("mean_recover_resume_tube_heading_score"),
            "mean_recover_resume_tube_clearance_score": mean_of("mean_recover_resume_tube_clearance_score"),
            "mean_recover_resume_tube_terminal_dist": mean_of("mean_recover_resume_tube_terminal_dist"),
            "mean_recover_resume_tube_ordered_loss": mean_of("mean_recover_resume_tube_ordered_loss"),
            "mean_recover_resume_tube_prefix_min_clearance": mean_of("mean_recover_resume_tube_prefix_min_clearance"),
            "mean_recover_resume_window_wrist_l2_mean": mean_of("mean_recover_resume_window_wrist_l2_mean"),
            "max_recover_resume_window_wrist_l2_max": max_of("max_recover_resume_window_wrist_l2_max"),
            "mean_recover_resume_window_left_wrist_abs_mean": mean_of("mean_recover_resume_window_left_wrist_abs_mean"),
            "max_recover_resume_window_left_wrist_abs_max": max_of("max_recover_resume_window_left_wrist_abs_max"),
            "mean_recover_resume_window_right_wrist_abs_mean": mean_of("mean_recover_resume_window_right_wrist_abs_mean"),
            "max_recover_resume_window_right_wrist_abs_max": max_of("max_recover_resume_window_right_wrist_abs_max"),
            "recover_resume_tube_ok_count": sum_of("recover_resume_tube_ok_count"),
            "recover_resume_tube_reject_count": sum_of("recover_resume_tube_reject_count"),
            "resume_affordance_available_count": sum_of("resume_affordance_available_count"),
            "resume_affordance_ok_count": sum_of("resume_affordance_ok_count"),
            "mean_resume_affordance_score": mean_of("mean_resume_affordance_score"),
            "min_resume_affordance_score": min_of("min_resume_affordance_score"),
            "mean_resume_affordance_component_score": mean_of("mean_resume_affordance_component_score"),
            "mean_resume_affordance_target_distance": mean_of("mean_resume_affordance_target_distance"),
            "mean_resume_affordance_contact_score": mean_of("mean_resume_affordance_contact_score"),
            "act_resumable_ok_count": sum_of("act_resumable_ok_count"),
            "act_resumable_live_ok_count": sum_of("act_resumable_live_ok_count"),
            "act_resumable_nominal_ok_count": sum_of("act_resumable_nominal_ok_count"),
            "mean_act_resumable_score": mean_of("mean_act_resumable_score"),
            "min_act_resumable_score": min_of("min_act_resumable_score"),
            "mean_act_resumable_nominal_score": mean_of("mean_act_resumable_nominal_score"),
            "mean_act_resumable_live_score": mean_of("mean_act_resumable_live_score"),
            "act_action_agreement_logged_count": sum_of("act_action_agreement_logged_count"),
            "act_action_agreement_post_recovery_or_reentry_count": sum_of("act_action_agreement_post_recovery_or_reentry_count"),
            "mean_act_action_agreement_act_vs_safe_l2": mean_of("mean_act_action_agreement_act_vs_safe_l2"),
            "mean_act_action_agreement_act_vs_safe_cosine": mean_of("mean_act_action_agreement_act_vs_safe_cosine"),
            "mean_act_action_agreement_act_vs_safe_arm_l2": mean_of("mean_act_action_agreement_act_vs_safe_arm_l2"),
            "mean_act_action_agreement_act_vs_safe_gripper_abs_delta": mean_of("mean_act_action_agreement_act_vs_safe_gripper_abs_delta"),
            "mean_act_action_agreement_act_vs_nominal_l2": mean_of("mean_act_action_agreement_act_vs_nominal_l2"),
            "mean_act_action_agreement_act_vs_nominal_cosine": mean_of("mean_act_action_agreement_act_vs_nominal_cosine"),
            "mean_act_action_agreement_act_vs_nominal_arm_l2": mean_of("mean_act_action_agreement_act_vs_nominal_arm_l2"),
            "mean_act_action_agreement_act_vs_nominal_gripper_abs_delta": mean_of("mean_act_action_agreement_act_vs_nominal_gripper_abs_delta"),
            "mean_act_action_agreement_safe_vs_nominal_l2": mean_of("mean_act_action_agreement_safe_vs_nominal_l2"),
            "mean_act_action_agreement_safe_vs_nominal_cosine": mean_of("mean_act_action_agreement_safe_vs_nominal_cosine"),
            "mean_act_action_agreement_safe_vs_nominal_arm_l2": mean_of("mean_act_action_agreement_safe_vs_nominal_arm_l2"),
            "mean_act_action_agreement_safe_vs_nominal_gripper_abs_delta": mean_of("mean_act_action_agreement_safe_vs_nominal_gripper_abs_delta"),
            "mean_act_action_agreement_act_vs_target_l2": mean_of("mean_act_action_agreement_act_vs_target_l2"),
            "mean_act_action_agreement_act_vs_target_cosine": mean_of("mean_act_action_agreement_act_vs_target_cosine"),
            "mean_act_action_agreement_act_vs_target_arm_l2": mean_of("mean_act_action_agreement_act_vs_target_arm_l2"),
            "mean_act_action_agreement_act_vs_target_gripper_abs_delta": mean_of("mean_act_action_agreement_act_vs_target_gripper_abs_delta"),
            "mean_act_action_agreement_safe_vs_target_l2": mean_of("mean_act_action_agreement_safe_vs_target_l2"),
            "mean_act_action_agreement_safe_vs_target_cosine": mean_of("mean_act_action_agreement_safe_vs_target_cosine"),
            "mean_act_action_agreement_safe_vs_target_arm_l2": mean_of("mean_act_action_agreement_safe_vs_target_arm_l2"),
            "mean_act_action_agreement_safe_vs_target_gripper_abs_delta": mean_of("mean_act_action_agreement_safe_vs_target_gripper_abs_delta"),
            "mean_act_action_agreement_act_vs_last_recovery_l2": mean_of("mean_act_action_agreement_act_vs_last_recovery_l2"),
            "mean_act_action_agreement_act_vs_last_recovery_cosine": mean_of("mean_act_action_agreement_act_vs_last_recovery_cosine"),
            "mean_act_action_agreement_act_vs_last_recovery_arm_l2": mean_of("mean_act_action_agreement_act_vs_last_recovery_arm_l2"),
            "mean_act_action_agreement_act_vs_last_recovery_gripper_abs_delta": mean_of("mean_act_action_agreement_act_vs_last_recovery_gripper_abs_delta"),
            "mean_act_action_agreement_safe_vs_last_recovery_l2": mean_of("mean_act_action_agreement_safe_vs_last_recovery_l2"),
            "mean_act_action_agreement_safe_vs_last_recovery_cosine": mean_of("mean_act_action_agreement_safe_vs_last_recovery_cosine"),
            "mean_act_action_agreement_safe_vs_last_recovery_arm_l2": mean_of("mean_act_action_agreement_safe_vs_last_recovery_arm_l2"),
            "mean_act_action_agreement_safe_vs_last_recovery_gripper_abs_delta": mean_of("mean_act_action_agreement_safe_vs_last_recovery_gripper_abs_delta"),
            "mean_recover_resume_affordance_score": mean_of("mean_recover_resume_affordance_score"),
            "min_recover_resume_affordance_score": min_of("min_recover_resume_affordance_score"),
            "mean_recover_resume_affordance_component_score": mean_of("mean_recover_resume_affordance_component_score"),
            "mean_recover_resume_affordance_target_distance": mean_of("mean_recover_resume_affordance_target_distance"),
            "mean_recover_resume_affordance_contact_score": mean_of("mean_recover_resume_affordance_contact_score"),
            "mean_recover_resume_affordance_bonus": mean_of("mean_recover_resume_affordance_bonus"),
            "recover_resume_affordance_ok_count": sum_of("recover_resume_affordance_ok_count"),
            "recover_resume_affordance_reject_count": sum_of("recover_resume_affordance_reject_count"),
            "mean_mpc_handoff_resume_affordance_score": mean_of("mean_mpc_handoff_resume_affordance_score"),
            "mean_mpc_handoff_resume_affordance_component_score": mean_of("mean_mpc_handoff_resume_affordance_component_score"),
            "mpc_handoff_resume_affordance_ok_count": sum_of("mpc_handoff_resume_affordance_ok_count"),
            "mean_committed_rejoin_resume_affordance_score": mean_of("mean_committed_rejoin_resume_affordance_score"),
            "mean_committed_rejoin_resume_affordance_component_score": mean_of("mean_committed_rejoin_resume_affordance_component_score"),
            "committed_rejoin_resume_affordance_ok_count": sum_of("committed_rejoin_resume_affordance_ok_count"),
            "mean_committed_soft_handoff_resume_affordance_score": mean_of("mean_committed_soft_handoff_resume_affordance_score"),
            "mean_recover_ordered_pose_loss": mean_of("mean_recover_ordered_pose_loss"),
            "mean_recover_ordered_delta_loss": mean_of("mean_recover_ordered_delta_loss"),
            "mean_recover_ordered_loss": mean_of("mean_recover_ordered_loss"),
            "mean_gradient_iterations_run": mean_of("mean_gradient_iterations_run"),
            "max_gradient_iterations_run": max_of("max_gradient_iterations_run"),
            "gradient_early_stopped_count": sum_of("gradient_early_stopped_count"),
            "gradient_candidate_early_stopped_count": sum_of("gradient_candidate_early_stopped_count"),
            "gradient_batched_line_search_count": sum_of("gradient_batched_line_search_count"),
            "gradient_jax_scan_used_count": sum_of("gradient_jax_scan_used_count"),
            "gradient_full_jax_scan_used_count": sum_of("gradient_full_jax_scan_used_count"),
            "fixed_shape_jax_optimizer_loop_count": sum_of("fixed_shape_jax_optimizer_loop_count"),
            "mean_gradient_full_jax_scan_time_ms": mean_of("mean_gradient_full_jax_scan_time_ms"),
            "mean_gradient_line_search_batch_evaluations": mean_of("mean_gradient_line_search_batch_evaluations"),
            "mean_gradient_jax_scan_used_count": mean_of("mean_gradient_jax_scan_used_count"),
            "mean_optimizer_evaluations": mean_of("mean_optimizer_evaluations"),
            "mean_deform_optimizer_time_ms": mean_of("mean_deform_optimizer_time_ms"),
            "mean_return_optimizer_time_ms": mean_of("mean_return_optimizer_time_ms"),
            "mean_explicit_optimizer_time_ms": mean_of("mean_explicit_optimizer_time_ms"),
            "mean_committed_suffix_optimizer_time_ms": mean_of("mean_committed_suffix_optimizer_time_ms"),
            "mean_committed_plan_safety_time_ms": mean_of("mean_committed_plan_safety_time_ms"),
            "mean_committed_plan_diagnostics_time_ms": mean_of("mean_committed_plan_diagnostics_time_ms"),
            "max_committed_plan_diagnostics_time_ms": max_of("max_committed_plan_diagnostics_time_ms"),
            "mean_cem_iterations_run": mean_of("mean_cem_iterations_run"),
            "max_cem_iterations_run": max_of("max_cem_iterations_run"),
            "cem_early_stopped_count": sum_of("cem_early_stopped_count"),
            "mean_deform_cem_iterations_run": mean_of("mean_deform_cem_iterations_run"),
            "mean_return_cem_iterations_run": mean_of("mean_return_cem_iterations_run"),
            "deform_cem_early_stopped_count": sum_of("deform_cem_early_stopped_count"),
            "return_cem_early_stopped_count": sum_of("return_cem_early_stopped_count"),
            "hold_unsafe_count": max_of("hold_unsafe_count"),
            "hold_predicted_contact_count": max_of("hold_predicted_contact_count"),
            "emergency_deform_away_steps": max_of("emergency_deform_away_steps"),
            "emergency_deform_away_count": max_of("emergency_deform_away_count"),
            "contact_during_hold_count": max_of("contact_during_hold_count"),
            "contact_during_brake_count": max_of("contact_during_brake_count"),
            "contact_during_deform_count": max_of("contact_during_deform_count"),
            "contact_during_recover_count": max_of("contact_during_recover_count"),
            "mean_hold_horizon_min_clearance": mean_of("mean_hold_horizon_min_clearance"),
            "min_hold_horizon_min_clearance": min(
                [s.get("min_hold_horizon_min_clearance") for s in episode_summaries if s.get("min_hold_horizon_min_clearance") is not None],
                default=None,
            ),
            "mean_act_ratio": mean_of("act_ratio"),
            "mean_safety_mode_ratio": mean_of("safety_mode_ratio"),
            "mean_fallback_ratio": mean_of("fallback_ratio"),
            "mean_task_progress": mean_of("mean_task_progress"),
            "max_task_progress": max_of("max_task_progress"),
            "final_task_progress": mean_of("final_task_progress"),
            "mean_task_progress_delta": mean_of("mean_task_progress_delta"),
            "mean_progress_during_act": mean_of("mean_progress_during_act"),
            "mean_progress_during_deform": mean_of("mean_progress_during_deform"),
            "mean_progress_during_recover": mean_of("mean_progress_during_recover"),
            "mean_progress_during_brake": mean_of("mean_progress_during_brake"),
            "mean_progress_during_fallback": mean_of("mean_progress_during_fallback"),
            "num_progress_regressions": sum_of("num_progress_regressions"),
            "num_large_arm_delta_events": sum_of("num_large_arm_delta_events"),
            "num_large_base_delta_events": sum_of("num_large_base_delta_events"),
            "likely_failure_cause": (
                episode_summaries[-1].get("likely_failure_cause")
                if len(episode_summaries) == 1
                else None
            ),
            "likely_failure_cause_counts": {
                cause: sum(1 for s in episode_summaries if s.get("likely_failure_cause") == cause)
                for cause in sorted({s.get("likely_failure_cause") for s in episode_summaries if s.get("likely_failure_cause") is not None})
            },
            "mean_pass_through_rate": mean_of("pass_through_rate"),
            "mean_horizon_brake_rate": mean_of("horizon_brake_rate"),
            "mean_path_consistent_brake_rate": mean_of("path_consistent_brake_rate"),
            "mean_path_consistent_brake_intended_rate": mean_of("path_consistent_brake_intended_rate"),
            "mean_horizon_brake_intended_rate": mean_of("horizon_brake_intended_rate"),
            "mean_verified_failsafe_rate": mean_of("verified_failsafe_rate"),
            "mean_unverified_emergency_failsafe_rate": mean_of("unverified_emergency_failsafe_rate"),
            "mean_horizon_deform_rate": mean_of("horizon_deform_rate"),
            "mean_sequential_oscbf_rate": mean_of("sequential_oscbf_rate"),
            "mean_pause_on_unsafe_rate": mean_of("pause_on_unsafe_rate"),
            "mean_pause_and_restart_rate": mean_of("pause_and_restart_rate"),
            "mean_phase_reanchor_rate": mean_of("phase_reanchor_rate"),
            "mean_phase_reanchor_source_rate": mean_of("phase_reanchor_source_rate"),
            "total_phase_reanchor_steps": sum_of("phase_reanchor_steps"),
            "total_gripper_latched_steps": sum_of("gripper_latched_steps"),
            "total_post_recovery_task_guard_steps": sum_of("post_recovery_task_guard_steps"),
            "total_post_recovery_reanchor_started_count": sum_of("post_recovery_reanchor_started_count"),
            "total_post_recovery_progress_regression_count": sum_of("post_recovery_progress_regression_count"),
            "mean_chunk_deform_source_rate": mean_of("chunk_deform_source_rate"),
            "mean_sequential_oscbf_source_rate": mean_of("sequential_oscbf_source_rate"),
            "min_robot_human_distance": min_of("min_robot_human_distance"),
            "mean_min_robot_human_distance": mean_of("min_robot_human_distance"),
            "total_safety_violations": sum_of("num_safety_violations"),
            "total_filter_activations": sum_of("num_filter_activations"),
            "total_brake_steps": sum_of("total_brake_steps"),
            "total_deformation_steps": sum_of("total_deformation_steps"),
            "task_success_rate": mean_of("task_success"),
            "mean_drawer_open_distance": mean_of("drawer_open_distance"),
            "mean_resume_latency_after_human_exit": mean_of("resume_latency_after_human_exit"),
        }
    )
    return summary
