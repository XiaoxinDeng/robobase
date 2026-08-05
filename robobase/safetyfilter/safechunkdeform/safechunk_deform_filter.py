from __future__ import annotations

import logging
import time
from typing import Any, Optional, Mapping

import numpy as np

from .safechunk_brake import Brake
from .safechunk_deform import Deform, DeformConfig
from .safechunk_recovery import Recovery, RecoveryContext
from .safechunk_intervention_factory import InterventionExecutionFactory
from .intervention_state import (
    DeformationAdmissibilityResult,
    DeformationEvaluation,
    InterventionFSMConfig,
    InterventionMode,
    InterventionPolicy,
    InterventionStateMachine,
    StopCounterfactualResult,
    UnsafeReason,
)


logger = logging.getLogger(__name__)


class SafeChunkDeformFilter:
    """High-level SafeChunk-Deform controller that routes chunks through executors."""

    _RECOVERY_COMMIT_DIAGNOSTIC_KEYS: tuple[str, ...] = (
        "optimized_accepted",
        "deform_stage_accepted",
        "recover_accepted",
        "recover_target_index",
        "resumed_from_recover_index",
        "return_accepted",
        "return_target_index",
        "resumed_from_cached_index",
        "is_safe",
        "is_recoverable",
        "rejoin_index",
        "q_rejoin_dist",
        "recover_rejoin_loss",
        "recover_projection_on_nominal",
        "recover_cosine_to_nominal",
        "recover_direction_cosine",
        "recover_direction_cosine_threshold",
        "recover_direction_loss",
        "recover_direction_ok",
        "recover_direction_alignment_available",
        "recover_direction_alignment_weight",
        "recover_ordered_path_available",
        "recover_ordered_target_index",
        "recover_ordered_horizon",
        "recover_ordered_pose_loss",
        "recover_ordered_delta_loss",
        "recover_ordered_loss",
        "recover_ordered_pose_weight",
        "recover_ordered_delta_weight",
        "recover_ordered_pose_threshold",
        "recover_ordered_delta_threshold",
        "recover_ordered_ok",
        "nominal_delta_norm",
        "path_delta_norm",
        "nominal_rejoin_score",
        "nominal_rejoin_available",
        "nominal_rejoin_suppressed_reason",
        "nominal_rejoin_clearance",
        "nominal_rejoin_safe_prefix_len",
        "deform_min_clearance_stage",
        "recover_min_clearance",
        "return_rejoin_loss",
        "return_min_clearance",
        "deformation_norm",
        "deform_min_clearance",
        "best_min_clearance",
        "required_min_clearance",
        "clearance_gap",
        "explicit_recovery",
        "deform_chunk_length",
        "recover_chunk_length",
        "explicit_return",
        "return_chunk_length",
        "committed_chunk_total_length",
        "nominal_attribution_early_deform_suppressed",
        "nominal_attribution_early_deform_suppression_reason",
        "nominal_attribution_early_deform_candidate_accepted_before_suppression",
        "nominal_attribution_early_deform_stop_counterfactual_safe",
        "nominal_attribution_early_deform_stop_counterfactual_min_clearance",
        "nominal_attribution_early_deform_human_motion_speed",
        "nominal_attribution_early_deform_human_motion_displacement",
        "nominal_attribution_early_deform_human_motion_static",
        "nominal_attribution_early_deform_goal_blocked",
        "nominal_attribution_early_deform_goal_distance",
        "nominal_attribution_early_deform_goal_check_available",
        "nominal_attribution_early_deform_goal_check_source",
        "path_block_pause_sufficient",
        "path_block_pause_sufficiency_available",
        "path_block_pause_sufficiency_source",
        "path_block_requires_bypass",
        "stationary_human_local_escape_enabled",
        "stationary_human_local_escape_applied",
        "stationary_human_local_escape_skip_reason",
        "stationary_human_local_escape_counter",
        "stationary_human_local_escape_max_steps",
        "stationary_human_local_escape_stop_counterfactual_safe",
        "stationary_human_local_escape_stop_counterfactual_min_clearance",
        "stationary_human_local_escape_human_motion_speed",
        "stationary_human_local_escape_human_motion_displacement",
        "stationary_human_local_escape_human_motion_static",
        "stationary_human_local_escape_goal_blocked",
        "stationary_human_local_escape_goal_distance",
        "stationary_human_local_escape_goal_check_available",
        "stationary_human_local_escape_goal_check_source",
        "stationary_human_local_escape_candidate_accepted",
        "stationary_human_local_escape_candidate_path",
        "stationary_human_local_escape_candidate_hold_clearance",
        "stationary_human_local_escape_trigger_reason",
        "optimized_candidate_suppressed_by_local_escape",
        "optimized_accepted_before_local_escape",
        "recover_accepted_before_local_escape",
        "return_accepted_before_local_escape",
    )

    def __init__(self, cfg: Any | None = None) -> None:
        """Construct filter wiring for brake/deform/recovery executors and shared state."""
        cfg = self._cfg_to_dict(cfg)
        # Split nested configuration into filter-owned setup and executor-owned sections.
        filter_cfg: dict[str, Any] = self._cfg_section(cfg, "safety_filter")
        if not filter_cfg:
            filter_cfg = self._cfg_section(cfg, "filter")

        legacy_brake_cfg: dict[str, Any] = self._cfg_section(cfg, "brake")
        legacy_deform_cfg: dict[str, Any] = self._cfg_section(cfg, "deform")
        legacy_recovery_cfg: dict[str, Any] = self._cfg_section(cfg, "recovery")
        intervention_cfg: dict[str, Any] = self._cfg_section(cfg, "intervention")
        legacy_recovery_intervention_cfg: dict[str, Any] = self._cfg_section(
            legacy_recovery_cfg,
            "intervention",
        )
        if legacy_recovery_intervention_cfg:
            intervention_cfg = {
                **legacy_recovery_intervention_cfg,
                **intervention_cfg,
            }

        intervention_brake_cfg: dict[str, Any] = self._cfg_section(intervention_cfg, "brake")
        intervention_deform_cfg: dict[str, Any] = self._cfg_section(intervention_cfg, "deform")
        intervention_recovery_cfg: dict[str, Any] = self._cfg_section(intervention_cfg, "recovery")
        brake_cfg: dict[str, Any] = {**legacy_brake_cfg, **intervention_brake_cfg}
        deform_cfg: dict[str, Any] = {**legacy_deform_cfg, **intervention_deform_cfg}
        recovery_cfg: dict[str, Any] = {**legacy_recovery_cfg, **intervention_recovery_cfg}
        intervention_cfg = dict(intervention_cfg)
        intervention_cfg["brake"] = brake_cfg
        intervention_cfg["deform"] = deform_cfg
        intervention_cfg["recovery"] = recovery_cfg

        # Filter-owned runtime settings are parsed before executor construction.
        oscbf_operator = self._cfg_value(filter_cfg, cfg, "oscbf_operator", None)
        horizon = self._cfg_value(filter_cfg, cfg, "horizon", 16)
        dt = self._cfg_value(filter_cfg, cfg, "dt", 1.0 / 20.0)
        action_dim = self._cfg_value(filter_cfg, cfg, "action_dim", 16)
        expected_motion_dim = self._cfg_value(filter_cfg, cfg, "expected_motion_dim", 14)
        control_type = self._cfg_value(filter_cfg, cfg, "control_type", "absolute")
        controlled_action_indices = self._cfg_value(
            filter_cfg,
            cfg,
            "controlled_action_indices",
            None,
        )
        controlled_state_indices = self._cfg_value(
            filter_cfg,
            cfg,
            "controlled_state_indices",
            None,
        )
        min_clearance = self._cfg_value(filter_cfg, cfg, "min_clearance", 0.08)
        diagnostics = self._cfg_value(filter_cfg, cfg, "diagnostics", None)
        contact_rich_pause_cfg: dict[str, Any] = self._cfg_section(
            filter_cfg,
            "contact_rich_pause",
        )
        if not contact_rich_pause_cfg:
            contact_rich_pause_cfg = self._cfg_section(cfg, "contact_rich_pause")
        rollout_mismatch_cfg: dict[str, Any] = self._cfg_section(
            filter_cfg,
            "rollout_mismatch",
        )
        if not rollout_mismatch_cfg:
            rollout_mismatch_cfg = self._cfg_section(cfg, "rollout_mismatch")
        rollout_model_cfg: dict[str, Any] = self._cfg_section(
            filter_cfg,
            "rollout_model",
        )
        if not rollout_model_cfg:
            rollout_model_cfg = self._cfg_section(cfg, "rollout_model")
        debug = self._cfg_value(filter_cfg, cfg, "debug", True)
        enabled = self._cfg_value(filter_cfg, cfg, "enabled", True)

        # Brake executor settings define the first intervention rung.
        brake_progress_threshold = self._cfg_value(
            brake_cfg,
            cfg,
            "brake_progress_threshold",
            0.05,
        )
        deadlock_window = self._cfg_value(brake_cfg, cfg, "deadlock_window", 5)
        temporary_blocker = self._cfg_value(brake_cfg, cfg, "temporary_blocker", None)
        safechunk_active_safety = self._cfg_value(
            brake_cfg,
            cfg,
            "safechunk_active_safety",
            None,
        )

        # Deform executor settings define candidate generation and optimization.
        deformation_enabled = self._cfg_value(
            deform_cfg,
            cfg,
            "deformation_enabled",
            True,
        )
        mode = self._cfg_value(deform_cfg, cfg, "mode", "optimized")
        chunk_deformation_scales = self._cfg_value(
            deform_cfg,
            cfg,
            "chunk_deformation_scales",
            None,
        )
        chunk_deformation_smoothing = self._cfg_value(
            deform_cfg,
            cfg,
            "chunk_deformation_smoothing",
            1,
        )
        sequential_oscbf_fallback = self._cfg_value(
            deform_cfg,
            cfg,
            "sequential_oscbf_fallback",
            False,
        )
        deform_after_deadlock_window = self._cfg_value(
            deform_cfg,
            cfg,
            "deform_after_deadlock_window",
            True,
        )
        unsafe_deformation_fallback = self._cfg_value(
            deform_cfg,
            cfg,
            "unsafe_deformation_fallback",
            "brake",
        )
        optimized_fallback = self._cfg_value(
            deform_cfg,
            cfg,
            "optimized_fallback",
            "brake",
        )
        detach_passthrough_dims = self._cfg_value(
            deform_cfg,
            cfg,
            "detach_passthrough_dims",
            True,
        )
        opt_iters = self._cfg_value(deform_cfg, cfg, "opt_iters", 20)
        opt_lr = self._cfg_value(deform_cfg, cfg, "opt_lr", 0.03)
        opt_population = self._cfg_value(deform_cfg, cfg, "opt_population", 32)
        opt_elite_frac = self._cfg_value(deform_cfg, cfg, "opt_elite_frac", 0.25)
        opt_seed = self._cfg_value(deform_cfg, cfg, "opt_seed", 0)
        lambda_safety = self._cfg_value(deform_cfg, cfg, "lambda_safety", 100.0)
        lambda_action = self._cfg_value(deform_cfg, cfg, "lambda_action", 1.0)
        lambda_path = self._cfg_value(deform_cfg, cfg, "lambda_path", 1.0)
        lambda_smooth = self._cfg_value(deform_cfg, cfg, "lambda_smooth", 0.1)
        optimized_deform = self._cfg_value(deform_cfg, cfg, "optimized_deform", None)
        safechunk_acceptance = self._cfg_value(deform_cfg,cfg,"safechunk_acceptance",None,)
        debug_safety_feasibility = self._cfg_value(deform_cfg,cfg,"debug_safety_feasibility",False,)
        action_low = self._cfg_value(deform_cfg, cfg, "action_low", None)
        action_high = self._cfg_value(deform_cfg, cfg, "action_high", None)
        max_action_delta = self._cfg_value(deform_cfg, cfg, "max_action_delta", None)
        pause_exit_smoothing_cfg: dict[str, Any] = self._cfg_section(
            deform_cfg,
            "pause_exit_smoothing",
        )
        intervention_state_cfg: dict[str, Any] = self._cfg_section(
            deform_cfg,
            "intervention_state",
        )
        raw_intervention_policy = intervention_state_cfg.get(
            "intervention_policy",
            deform_cfg.get("intervention_policy", InterventionPolicy.LEGACY.value),
        )
        try:
            intervention_policy = InterventionPolicy(str(raw_intervention_policy))
        except ValueError:
            intervention_policy = InterventionPolicy.LEGACY

        # Recovery executor settings define rejoin and committed-recovery behavior.
        recoverable_deform = self._cfg_value(recovery_cfg,cfg,"recoverable_deform",None,)
        explicit_recovery = self._cfg_value(recovery_cfg,cfg,"explicit_recovery",None,)
        safechunk_replan = self._cfg_value(recovery_cfg,cfg,"safechunk_replan",None,)
        safechunk_recover = self._cfg_value(recovery_cfg,cfg,"safechunk_recover",None,)
        safechunk_recovery_corridor = self._cfg_value(recovery_cfg,cfg,"safechunk_recovery_corridor",None,)
        lambda_rejoin = self._cfg_value(recovery_cfg, cfg, "lambda_rejoin", 5.0)
        rejoin_threshold = self._cfg_value(recovery_cfg, cfg, "rejoin_threshold", 0.03)
        min_rejoin_offset = self._cfg_value(recovery_cfg, cfg, "min_rejoin_offset", 2)
        use_ee_pose_rejoin = self._cfg_value(recovery_cfg,cfg,"use_ee_pose_rejoin",False,)
        use_object_state_rejoin = self._cfg_value(recovery_cfg,cfg,"use_object_state_rejoin",False,)
        brake_if_unrecoverable = self._cfg_value(recovery_cfg,cfg,"brake_if_unrecoverable",True,)

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        if control_type not in {"absolute", "delta", "velocity"}:
            raise ValueError(
                "control_type must be one of ['absolute', 'delta', 'velocity'], "
                f"got {control_type}"
            )
        if unsafe_deformation_fallback not in {"brake", "best"}:
            raise ValueError(
                "unsafe_deformation_fallback must be one of ['brake', 'best'], "
                f"got {unsafe_deformation_fallback}"
            )
        mode = str(mode).lower()
        if mode != "optimized":
            raise ValueError(
                "mode must be optimized; candidate mode has been removed; "
                f"got {mode}"
            )
        optimized_fallback = str(optimized_fallback).lower()
        if optimized_fallback != "brake":
            raise ValueError(
                "optimized_fallback must be 'brake'; candidate fallback has been removed. "
                f"got {optimized_fallback}"
            )

        # Filter-owned fields are explicitly annotated so ownership is visible.
        self.oscbf_operator: Any = oscbf_operator
        self._operator_instantiation_failed: bool = False
        self.horizon: int = int(horizon)
        self.dt: float = float(dt)
        self.action_dim: int = int(action_dim)
        self.expected_motion_dim: int = int(expected_motion_dim)
        self.control_type: str = control_type
        self.controlled_action_indices: np.ndarray = np.asarray(
            controlled_action_indices
            if controlled_action_indices is not None
            else [4, 5, 6, 7, 9, 10, 11, 12],
            dtype=np.int64,
        )
        self.controlled_state_indices: np.ndarray = np.asarray(
            controlled_state_indices
            if controlled_state_indices is not None
            else [4, 5, 6, 7, 9, 10, 11, 12],
            dtype=np.int64,
        )
        self.min_clearance: float = float(min_clearance)
        self.diagnostics: dict[str, Any] = dict(diagnostics or {})
        self.rollout_model_config: dict[str, Any] = dict(rollout_model_cfg or {})
        self.contact_rich_pause_config: dict[str, Any] = dict(contact_rich_pause_cfg)
        self.contact_rich_pause_enabled: bool = bool(
            contact_rich_pause_cfg.get("enabled", False)
        )
        self.contact_rich_pause_min_hold_steps: int = max(
            0,
            int(contact_rich_pause_cfg.get("min_hold_steps", 4)),
        )
        self.contact_rich_pause_exit_safe_steps: int = max(
            0,
            int(contact_rich_pause_cfg.get("exit_safe_steps", 2)),
        )
        self.contact_rich_pause_max_steps: int = max(
            0,
            int(contact_rich_pause_cfg.get("max_pause_steps", 8)),
        )
        self.contact_rich_pause_activate_on_metadata: bool = bool(
            contact_rich_pause_cfg.get("activate_on_metadata", True)
        )
        self.contact_rich_pause_activate_on_gripper_command: bool = bool(
            contact_rich_pause_cfg.get("activate_on_gripper_command", True)
        )
        default_gripper_indices = (
            [self.expected_motion_dim, self.expected_motion_dim + 1]
            if self.action_dim > self.expected_motion_dim
            else [-1]
        )
        self.contact_rich_pause_gripper_action_indices: tuple[int, ...] = tuple(
            int(idx)
            for idx in contact_rich_pause_cfg.get(
                "gripper_action_indices",
                default_gripper_indices,
            )
        )
        self.contact_rich_pause_gripper_closed_is_high: bool = bool(
            contact_rich_pause_cfg.get("gripper_closed_is_high", True)
        )
        self.contact_rich_pause_gripper_close_threshold: float = float(
            contact_rich_pause_cfg.get("gripper_close_threshold", 0.5)
        )
        self.contact_rich_pause_gripper_lookahead: int = max(
            1,
            int(contact_rich_pause_cfg.get("gripper_lookahead", 4)),
        )
        self.contact_rich_gripper_only_continue_when_clear: bool = bool(
            contact_rich_pause_cfg.get("gripper_only_continue_when_clear", True)
        )
        self.contact_rich_gripper_only_min_clearance: float = float(
            contact_rich_pause_cfg.get("gripper_only_min_clearance", 0.02)
        )
        self.contact_rich_pause_min_task_progress: float | None = self._optional_float(
            contact_rich_pause_cfg.get("min_task_progress", None)
        )
        self.contact_rich_pause_max_task_progress: float | None = self._optional_float(
            contact_rich_pause_cfg.get("max_task_progress", None)
        )
        self.rollout_mismatch_mitigation_enabled: bool = bool(
            rollout_mismatch_cfg.get("enabled", True)
        )
        self.rollout_residual_correction_enabled: bool = bool(
            rollout_mismatch_cfg.get("residual_correction_enabled", False)
        )
        self.rollout_residual_correction_scale: float = float(
            rollout_mismatch_cfg.get("residual_correction_scale", 0.5)
        )
        self.rollout_residual_correction_decay: float = float(
            rollout_mismatch_cfg.get("residual_correction_decay", 0.75)
        )
        self.rollout_residual_correction_max_abs: float = float(
            rollout_mismatch_cfg.get("residual_correction_max_abs", 0.25)
        )
        self.rollout_mismatch_l2_threshold: float = float(
            rollout_mismatch_cfg.get("l2_threshold", 0.75)
        )
        self.rollout_mismatch_max_abs_threshold: float = float(
            rollout_mismatch_cfg.get("max_abs_threshold", 0.35)
        )
        self.rollout_mismatch_live_min_clearance: float = float(
            rollout_mismatch_cfg.get("live_min_clearance", 0.02)
        )
        self.rollout_mismatch_escape_trigger_steps: int = max(
            1,
            int(rollout_mismatch_cfg.get("escape_trigger_steps", 4)),
        )
        self.rollout_mismatch_escape_pass_through_steps: int = max(
            1,
            int(rollout_mismatch_cfg.get("escape_pass_through_steps", 3)),
        )
        self.rollout_mismatch_progress_eps: float = float(
            rollout_mismatch_cfg.get("progress_eps", 0.003)
        )
        self.hard_safety_prefix_len: int = max(
            1,
            int(rollout_mismatch_cfg.get("hard_safety_prefix_len", 1)),
        )
        self.full_horizon_soft_when_prefix_safe: bool = bool(
            rollout_mismatch_cfg.get("full_horizon_soft_when_prefix_safe", True)
        )
        self.contact_rich_pause_signal_keys: tuple[str, ...] = tuple(
            str(key)
            for key in contact_rich_pause_cfg.get(
                "signal_keys",
                (
                    "contact_rich",
                    "contact_rich_state",
                    "in_contact",
                    "gripper_latched",
                    "handle_grasped",
                    "drawer_handle_grasped",
                    "object_grasped",
                    "grasped",
                    "is_grasped",
                    "is_holding",
                    "robot_object_contact",
                    "robot_drawer_contact",
                    "robot_handle_contact",
                    "ee_object_contact",
                    "gripper_contact",
                ),
            )
        )
        self.contact_rich_pause_count_keys: tuple[str, ...] = tuple(
            str(key)
            for key in contact_rich_pause_cfg.get(
                "count_keys",
                (
                    "contact_count",
                    "contacts",
                    "robot_object_contact_count",
                    "robot_drawer_contact_count",
                    "robot_handle_contact_count",
                    "robot_human_contact_count",
                ),
            )
        )
        self.debug: bool = bool(debug)
        self.enabled: bool = bool(enabled)
        self.last_info: dict[str, Any] = {}
        self._rollout_context: dict[str, Any] = {}
        self._rollout_context_step: int = 0
        self._trigger_count: int = 0
        self._warned_no_safety_eval: bool = False
        self.pause_exit_smoothing_enabled: bool = bool(
            pause_exit_smoothing_cfg.get("enabled", False)
        )
        self.pause_exit_smoothing_steps: int = max(
            0,
            int(pause_exit_smoothing_cfg.get("steps", 3)),
        )
        self.pause_exit_smoothing_first_alpha: float = float(
            pause_exit_smoothing_cfg.get("first_alpha", 0.35)
        )
        self.pause_exit_smoothing_alpha_increment: float = float(
            pause_exit_smoothing_cfg.get("alpha_increment", 0.3)
        )
        self.pause_exit_smoothing_clearance_tolerance: float = max(
            0.0,
            float(pause_exit_smoothing_cfg.get("clearance_tolerance", 0.005)),
        )
        self._pause_exit_smoothing_remaining: int = 0
        self._pause_exit_smoothing_total_steps: int = 0
        self._pause_exit_smoothing_anchor_action: np.ndarray | None = None
        self._pause_exit_smoothing_rearm_blocked: bool = False
        # A goal hold can end while the human is still leaving the scene. Keep
        # braking until the nominal horizon is actually safe instead of
        # immediately treating the exit transient as a path-deform request.
        self.goal_block_release_wait_steps: int = max(
            0,
            int(intervention_state_cfg.get("goal_block_release_wait_steps", 20)),
        )
        self._goal_block_hold_active: bool = False
        self._goal_block_release_wait_count: int = 0
        self.intervention_fsm = InterventionStateMachine(
            InterventionFSMConfig(
                enabled=bool(intervention_state_cfg.get("enabled", True)),
                intervention_policy=intervention_policy.value,
                deform_valid_required_steps=max(1, int(intervention_state_cfg.get("deform_valid_required_steps", 1))),
                nominal_clear_required_steps=max(1, int(intervention_state_cfg.get("nominal_clear_required_steps", 3))),
                deform_commit_min_steps=max(0, int(intervention_state_cfg.get("deform_commit_min_steps", 0))),
                deform_stall_required_steps=max(1, int(intervention_state_cfg.get("deform_stall_required_steps", 3))),
                min_deform_progress=max(0.0, float(intervention_state_cfg.get("min_deform_progress", 0.0))),
                min_step_progress=max(0.0, float(intervention_state_cfg.get("min_step_progress", 0.0))),
                min_deform_velocity_ratio=max(0.0, float(intervention_state_cfg.get("min_deform_velocity_ratio", 0.0))),
                stop_action_threshold=max(0.0, float(intervention_state_cfg.get("stop_action_threshold", 0.0))),
                human_reconsider_distance=max(0.0, float(intervention_state_cfg.get("human_reconsider_distance", 0.03))),
                nominal_change_threshold=max(0.0, float(intervention_state_cfg.get("nominal_change_threshold", 0.05))),
                resume_blend_steps=max(1, int(intervention_state_cfg.get("resume_blend_steps", 5))),
                pause_deform_on_current_human_unsafe=bool(intervention_state_cfg.get("pause_deform_on_current_human_unsafe", True)),
                pause_deform_min_clearance_threshold=float(intervention_state_cfg.get("pause_deform_min_clearance_threshold", 0.03)),
                pause_deform_suppress_when_stop_sufficient=bool(intervention_state_cfg.get("pause_deform_suppress_when_stop_sufficient", True)),
                pause_deform_static_human_speed_threshold=max(0.0, float(intervention_state_cfg.get("pause_deform_static_human_speed_threshold", 0.03))),
                pause_deform_suppress_requires_goal_check=bool(intervention_state_cfg.get("pause_deform_suppress_requires_goal_check", False)),
                early_deform_suppress_when_stop_sufficient=bool(intervention_state_cfg.get("early_deform_suppress_when_stop_sufficient", True)),
                early_deform_static_human_displacement_threshold=max(0.0, float(intervention_state_cfg.get("early_deform_static_human_displacement_threshold", 0.02))),
                stationary_human_local_escape_enabled=bool(intervention_state_cfg.get("stationary_human_local_escape_enabled", True)),
                stationary_human_local_escape_max_steps=max(0, int(intervention_state_cfg.get("stationary_human_local_escape_max_steps", 6))),
                policy_collision_slowdown_enabled=bool(intervention_state_cfg.get("policy_collision_slowdown_enabled", False)),
                policy_collision_slowdown_max_steps=max(1, int(intervention_state_cfg.get("policy_collision_slowdown_max_steps", 1))),
                policy_collision_slowdown_min_first_violation=max(0, int(intervention_state_cfg.get("policy_collision_slowdown_min_first_violation", 6))),
                pause_guard_slowdown_enabled=bool(intervention_state_cfg.get("pause_guard_slowdown_enabled", True)),
                pause_guard_slowdown_max_steps=max(0, int(intervention_state_cfg.get("pause_guard_slowdown_max_steps", 2))),
                stop_counterfactual_enabled=bool(intervention_state_cfg.get("stop_counterfactual_enabled", True)),
                goal_block_check_enabled=bool(intervention_state_cfg.get("goal_block_check_enabled", True)),
                deformation_admissibility_enabled=bool(intervention_state_cfg.get("deformation_admissibility_enabled", True)),
                pause_budget_steps=max(0, int(intervention_state_cfg.get("pause_budget_steps", 0))),
                resume_hysteresis_steps=max(1, int(intervention_state_cfg.get("resume_hysteresis_steps", intervention_state_cfg.get("nominal_clear_required_steps", 3)))),
                deform_commit_steps=max(0, int(intervention_state_cfg.get("deform_commit_steps", intervention_state_cfg.get("deform_commit_min_steps", 0)))),
                goal_block_radius=max(0.0, float(intervention_state_cfg.get("goal_block_radius", 0.10))),
                max_terminal_deviation=max(0.0, float(intervention_state_cfg.get("max_terminal_deviation", 0.50))),
                min_task_progress=max(0.0, float(intervention_state_cfg.get("min_task_progress", intervention_state_cfg.get("min_deform_progress", 0.0)))),
            )
        )
        raw_slowdown_factors = intervention_state_cfg.get(
            "policy_collision_slowdown_factors",
            (0.95,),
        )
        try:
            self.policy_collision_slowdown_factors: tuple[float, ...] = tuple(
                float(v) for v in raw_slowdown_factors
            )
        except Exception:  # noqa: BLE001
            self.policy_collision_slowdown_factors = (0.95,)
        raw_pause_guard_slowdown_factors = intervention_state_cfg.get(
            "pause_guard_slowdown_factors",
            (0.65, 0.45, 0.25, 0.10, 0.05),
        )
        try:
            self.pause_guard_slowdown_factors: tuple[float, ...] = tuple(
                float(v) for v in raw_pause_guard_slowdown_factors
            )
        except Exception:  # noqa: BLE001
            self.pause_guard_slowdown_factors = (0.65, 0.45, 0.25, 0.10, 0.05)
        self._stationary_human_local_escape_counter: int = 0
        self._reset_contact_rich_pause_state()
        self._reset_rollout_mismatch_state()
        self.safety_filter_config: dict[str, Any] = dict(filter_cfg)
        self.intervention_config: dict[str, Any] = dict(intervention_cfg)
        self.intervention: dict[str, Any] = self.intervention_config
        self.intervention_factory: InterventionExecutionFactory = InterventionExecutionFactory(
            self,
            intervention=self.intervention_config,
        )

        # Recovery is constructed first because brake/deform consume shared recovery policy.
        self.recovery: Recovery = Recovery(
            self,
            deformation_enabled=deformation_enabled,
            temporary_blocker=temporary_blocker,
            optimized_deform=optimized_deform,
            recoverable_deform=recoverable_deform,
            explicit_recovery=explicit_recovery,
            safechunk_replan=safechunk_replan,
            safechunk_recover=safechunk_recover,
            safechunk_recovery_corridor=safechunk_recovery_corridor,
            intervention=self.intervention_factory.intervention_recovery_config,
            intervention_factory=self.intervention_factory,
            lambda_rejoin=lambda_rejoin,
            lambda_smooth=lambda_smooth,
            rejoin_threshold=rejoin_threshold,
            min_rejoin_offset=min_rejoin_offset,
            use_ee_pose_rejoin=use_ee_pose_rejoin,
            use_object_state_rejoin=use_object_state_rejoin,
            brake_if_unrecoverable=brake_if_unrecoverable,
        )
        self.brake: Brake = Brake(
            self,
            brake_progress_threshold=brake_progress_threshold,
            deadlock_window=deadlock_window,
            deformation_enabled=deformation_enabled,
            unsafe_deformation_fallback=unsafe_deformation_fallback,
            recover_retry_cooldown_steps=self.recovery.recover_retry_cooldown_steps,
            recover_max_attempts_per_unsafe_streak=(
                self.recovery.recover_max_attempts_per_unsafe_streak
            ),
            explicit_return=self.recovery.explicit_return,
            commit_accepted_chunks=self.recovery.commit_accepted_chunks,
            temporary_blocker=temporary_blocker,
            safechunk_active_safety=safechunk_active_safety,
            intervention=self.intervention_factory.intervention_brake_config,
            intervention_factory=self.intervention_factory,
        )

        optimized_cfg = Deform.optimized_deform_config(
            optimized_deform,
            debug_safety_feasibility=debug_safety_feasibility,
            opt_iters=opt_iters,
            opt_lr=opt_lr,
            opt_population=opt_population,
            opt_elite_frac=opt_elite_frac,
            opt_seed=opt_seed,
            jax_batched_optimizer=True,
            jax_batched_optimizer_fallback=True,
        )
        if (
            bool(optimized_cfg["debug_safety_feasibility"])
            and self.recovery.final_rejoin_metric == "ee_pose"
        ):
            self.recovery.final_rejoin_metric = "none"
        acceptance_cfg = Deform.safechunk_acceptance_config(safechunk_acceptance)
        self.deform: Deform = Deform(
            self,
            config=DeformConfig(
                mode=mode,
                deformation_enabled=deformation_enabled,
                recoverable_deform_enabled=self.recovery.recoverable_deform_enabled,
                explicit_return=self.recovery.explicit_return,
                safechunk_recover_enabled=self.recovery.safechunk_recover_enabled,
                recover_retry_cooldown_steps=self.recovery.recover_retry_cooldown_steps,
                recover_max_attempts_per_unsafe_streak=(
                    self.recovery.recover_max_attempts_per_unsafe_streak
                ),
                unsafe_deformation_fallback=unsafe_deformation_fallback,
                commit_accepted_chunks=self.recovery.commit_accepted_chunks,
                safechunk_acceptance_enabled=bool(acceptance_cfg["enabled"]),
                allow_candidate_fallback=bool(
                    acceptance_cfg.get("allow_fallback_path", acceptance_cfg.get("allow_candidate_fallback", False))
                ),
                candidate_fallback_only_if_no_optimized_result=bool(
                    acceptance_cfg.get("fallback_only_if_no_optimized_result", acceptance_cfg.get("candidate_fallback_only_if_no_optimized_result", True))
                ),
                optimized_fallback=optimized_fallback,
                chunk_deformation_scales=tuple(
                    float(x)
                    for x in (
                        chunk_deformation_scales
                        if chunk_deformation_scales is not None
                        else [0.0, 0.25, 0.5, 0.75]
                    )
                ),
                chunk_deformation_smoothing=chunk_deformation_smoothing,
                sequential_oscbf_fallback=sequential_oscbf_fallback,
                deform_after_deadlock_window=deform_after_deadlock_window,
                opt_iters=optimized_cfg["opt_iters"],
                opt_lr=optimized_cfg["opt_lr"],
                opt_population=optimized_cfg["opt_population"],
                opt_elite_frac=optimized_cfg.get(
                    "opt_elite_frac",
                    opt_elite_frac,
                ),
                opt_seed=optimized_cfg["opt_seed"],
                optimizer_method=optimized_cfg.get(
                    "optimizer_method",
                    "cem",
                ),
                gradient_samples=optimized_cfg["gradient_samples"],
                gradient_eps=optimized_cfg["gradient_eps"],
                gradient_adam_beta1=optimized_cfg["gradient_adam_beta1"],
                gradient_adam_beta2=optimized_cfg["gradient_adam_beta2"],
                gradient_min_improvement=optimized_cfg["gradient_min_improvement"],
                gradient_line_search_scales=tuple(
                    float(v) for v in optimized_cfg["gradient_line_search_scales"]
                ),
                gradient_batched_line_search=optimized_cfg[
                    "gradient_batched_line_search"
                ],
                gradient_early_stop_on_candidate=optimized_cfg.get("gradient_early_stop_on_path", optimized_cfg.get("gradient_early_stop_on_candidate", True)),
                lambda_safety=lambda_safety,
                lambda_action=lambda_action,
                lambda_path=lambda_path,
                lambda_rejoin=self.recovery.lambda_rejoin,
                lambda_smooth=lambda_smooth,
                lambda_deform_safety=self.intervention_factory.lambda_deform_safety,
                lambda_deform_action=self.intervention_factory.lambda_deform_action,
                lambda_deform_smooth=self.intervention_factory.lambda_deform_smooth,
                lambda_retreat=self.intervention_factory.lambda_retreat,
                jax_batched_optimizer=bool(optimized_cfg["jax_batched_optimizer"]),
                jax_batched_optimizer_fallback=bool(
                    optimized_cfg["jax_batched_optimizer_fallback"]
                ),
                action_low=action_low,
                action_high=action_high,
                max_action_delta=max_action_delta,
                brake_if_unrecoverable=self.recovery.brake_if_unrecoverable,
                inner_rejoin_metric=self.recovery.inner_rejoin_metric,
                final_rejoin_metric=self.recovery.final_rejoin_metric,
                cache_nominal_ee=self.recovery.cache_nominal_ee,
                ee_rejoin_in_inner_loop=self.recovery.ee_rejoin_in_inner_loop,
                debug_safety_feasibility=bool(optimized_cfg["debug_safety_feasibility"]),
                min_rejoin_offset=self.recovery.min_rejoin_offset,
                q_rejoin_threshold=self.recovery.q_rejoin_threshold,
                qd_rejoin_threshold=self.recovery.qd_rejoin_threshold,
                qd_rejoin_hard_threshold=self.recovery.qd_rejoin_hard_threshold,
                require_qd_rejoin=self.recovery.require_qd_rejoin,
                ee_rejoin_threshold=self.recovery.ee_rejoin_threshold,
                q_rejoin_weights=self.recovery.q_rejoin_weights,
                use_ee_final_check=self.recovery.use_ee_final_check,
                deform_horizon=self.recovery.deform_horizon,
                return_horizon=self.recovery.return_horizon,
                committed_execution_margin=self.recovery.committed_execution_margin,
                acceptance_clearance_tol=self.intervention_factory.acceptance_clearance_tol,
                acceptance_hard_min_clearance=float(acceptance_cfg["hard_min_clearance"]),
                acceptance_desired_min_clearance=float(
                    acceptance_cfg["desired_min_clearance"]
                ),
                allow_safe_prefix_execution=bool(
                    acceptance_cfg["allow_safe_prefix_execution"]
                ),
                min_safe_prefix_len=int(acceptance_cfg["min_safe_prefix_len"]),
                prefix_min_clearance=float(acceptance_cfg["prefix_min_clearance"]),
                rolling_replan_on_prefix=bool(acceptance_cfg["rolling_replan_on_prefix"]),
                full_horizon_required_for_recover=bool(
                    acceptance_cfg["full_horizon_required_for_recover"]
                ),
                full_horizon_required_for_deform=bool(
                    acceptance_cfg["full_horizon_required_for_deform"]
                ),
                emergency_brake_if_immediate_below_hard_margin=bool(
                    acceptance_cfg["emergency_brake_if_immediate_below_hard_margin"]
                ),
                recover_task_progress_weight=self.recovery.recover_task_progress_weight,
            ),
            intervention=self.intervention_factory.intervention_deform_config,
            intervention_factory=self.intervention_factory,
            rng=np.random.default_rng(opt_seed),
        )
        # After all executors exist, wire explicit sibling references everywhere.
        self.intervention_factory.attach_executors(
            brake=self.brake,
            deform=self.deform,
            recovery=self.recovery,
        )
        self.brake.attach_executors(
            brake=self.brake,
            deform=self.deform,
            recovery=self.recovery,
        )
        self.deform.attach_executors(
            brake=self.brake,
            deform=self.deform,
            recovery=self.recovery,
        )
        self.recovery.attach_executors(
            brake=self.brake,
            deform=self.deform,
            recovery=self.recovery,
        )

    def reset(self) -> None:
        """Reset transient filter and executor state at episode boundaries."""
        self.last_info = {}
        self.brake.reset_execution_state()
        self.deform.reset_execution_state()
        self.recovery.reset_execution_state()
        self._trigger_count = 0
        self.unsafe_streak = 0
        self.recovery._clear_committed_chunk()
        self.recovery.set_resume_affordance_context({})
        if hasattr(self, "intervention_fsm"):
            self.intervention_fsm.reset()
        self._pause_exit_smoothing_remaining = 0
        self._pause_exit_smoothing_total_steps = 0
        self._pause_exit_smoothing_anchor_action = None
        self._pause_exit_smoothing_rearm_blocked = False
        self._goal_block_hold_active = False
        self._goal_block_release_wait_count = 0
        self._reset_contact_rich_pause_state()
        self._reset_rollout_mismatch_state()

    @staticmethod
    def _cfg_to_dict(config: Any | None) -> dict[str, Any]:
        """Normalize filter config payload into a plain mapping."""
        if config is None:
            return {}
        if hasattr(config, "items"):
            return {str(k): v for k, v in config.items()}
        return dict(config)

    @classmethod
    def _cfg_section(cls, cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
        """Read a nested section from config with no mutation."""
        section = cfg.get(name, {})
        return cls._cfg_to_dict(section)

    @staticmethod
    def _cfg_value(
        section: Mapping[str, Any],
        legacy_cfg: Mapping[str, Any],
        key: str,
        default: Any,
    ) -> Any:
        """Read modern config key or fallback to legacy flat entry."""
        if key in section:
            return section[key]
        if key in legacy_cfg:
            logger.warning(
                "Deprecated flat SafeChunk-Deform cfg key '%s'; move it into the nested task section.",
                key,
            )
            return legacy_cfg[key]
        return default

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        """Parse optional numeric config values without forcing null to zero."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _reset_contact_rich_pause_state(self) -> None:
        """Reset the lightweight contact-rich pause state machine."""
        self.contact_rich_state: str = "free"
        self._contact_rich_pause_active: bool = False
        self._contact_rich_pause_hold_remaining: int = 0
        self._contact_rich_pause_clear_steps: int = 0
        self._contact_rich_pause_last_reason: str | None = None
        self._contact_rich_pause_last_signal: bool = False
        self._contact_rich_pause_last_raw_signal: bool = False
        self._contact_rich_pause_last_progress_allowed: bool = True
        self._contact_rich_pause_last_gripper_value: float | None = None
        self._contact_rich_pause_steps: int = 0
        self._contact_rich_pause_timed_out: bool = False
        self._contact_rich_pause_signal_consumed: bool = False

    def _reset_rollout_mismatch_state(self) -> None:
        """Reset counters used when Bigym execution disagrees with q-space rollout."""
        self._rollout_mismatch_live_safe_steps: int = 0
        self._rollout_mismatch_no_progress_steps: int = 0
        self._rollout_mismatch_best_progress: float | None = None
        self._rollout_mismatch_escape_steps_remaining: int = 0

    @classmethod
    def _metadata_lookup(
        cls,
        source: Any,
        key: str,
        *,
        depth: int = 2,
    ) -> tuple[bool, Any]:
        """Find a shallow metadata key in dict-like observations/kwargs."""
        if source is None or depth < 0 or not hasattr(source, "items"):
            return False, None
        try:
            items = list(source.items())
        except Exception:
            return False, None
        for raw_key, value in items:
            if str(raw_key) == key:
                return True, value
        for _raw_key, value in items:
            if hasattr(value, "items"):
                found, found_value = cls._metadata_lookup(value, key, depth=depth - 1)
                if found:
                    return True, found_value
        return False, None

    @staticmethod
    def _signal_value_active(value: Any) -> bool:
        """Interpret common contact/grasp metadata values as active/inactive."""
        if value is None:
            return False
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"", "0", "false", "none", "free", "open", "released"}:
                return False
            return lowered in {
                "1",
                "true",
                "contact",
                "contact_rich",
                "contact-rich",
                "grasp",
                "grasped",
                "gripping",
                "holding",
                "latched",
                "latching",
                "pull",
                "contact_pull",
                "contact-hold",
            }
        if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
            return False
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return False
            if arr.dtype.kind in {"b"}:
                return bool(np.any(arr))
            if arr.dtype.kind in {"i", "u", "f"}:
                finite = np.asarray(arr, dtype=np.float64)
                finite = finite[np.isfinite(finite)]
                if finite.size == 0:
                    return False
                return bool(np.nanmax(np.abs(finite)) > 0.0)
        except Exception:
            pass
        return bool(value)

    def _task_progress_allows_contact_rich_pause(self, kwargs: Mapping[str, Any]) -> bool:
        """Apply optional progress gates for contact-rich pause activation."""
        progress = kwargs.get("task_progress")
        try:
            progress_value = None if progress is None else float(progress)
        except (TypeError, ValueError):
            progress_value = None
        if progress_value is None or not np.isfinite(progress_value):
            return True
        if (
            self.contact_rich_pause_min_task_progress is not None
            and progress_value < self.contact_rich_pause_min_task_progress
        ):
            return False
        if (
            self.contact_rich_pause_max_task_progress is not None
            and progress_value > self.contact_rich_pause_max_task_progress
        ):
            return False
        return True

    def _contact_rich_metadata_signal(
        self,
        obs: Any,
        info: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> tuple[bool, str | None]:
        """Read explicit contact/grasp signals from kwargs, obs, or current info."""
        if not self.contact_rich_pause_activate_on_metadata:
            return False, None
        sources = (kwargs, obs, info)
        for key in self.contact_rich_pause_signal_keys:
            for source in sources:
                found, value = self._metadata_lookup(source, key)
                if found and self._signal_value_active(value):
                    return True, f"metadata:{key}"
        for key in self.contact_rich_pause_count_keys:
            for source in sources:
                found, value = self._metadata_lookup(source, key)
                if found and self._signal_value_active(value):
                    return True, f"contact_count:{key}"
        return False, None

    def _contact_rich_gripper_signal(self, chunk: np.ndarray) -> tuple[bool, str | None, float | None]:
        """Use upcoming gripper command as a temporary contact-rich proxy."""
        if not self.contact_rich_pause_activate_on_gripper_command:
            return False, None, None
        chunk_arr = np.asarray(chunk, dtype=np.float32)
        if chunk_arr.ndim != 2 or chunk_arr.shape[1] == 0:
            return False, None, None
        indices: list[int] = []
        for raw_idx in self.contact_rich_pause_gripper_action_indices:
            idx = int(raw_idx)
            if idx < 0:
                idx = chunk_arr.shape[1] + idx
            if 0 <= idx < chunk_arr.shape[1] and idx not in indices:
                indices.append(idx)
        if not indices:
            return False, None, None
        horizon = min(int(self.contact_rich_pause_gripper_lookahead), chunk_arr.shape[0])
        values = chunk_arr[:horizon, indices]
        if values.size == 0:
            return False, None, None
        if self.contact_rich_pause_gripper_closed_is_high:
            signal_value = float(np.nanmax(values))
            active = bool(signal_value >= self.contact_rich_pause_gripper_close_threshold)
        else:
            signal_value = float(np.nanmin(values))
            active = bool(signal_value <= self.contact_rich_pause_gripper_close_threshold)
        if not np.isfinite(signal_value):
            return False, None, None
        return active, "gripper_command", signal_value

    def _update_contact_rich_pause_state(
        self,
        obs: Any,
        chunk: np.ndarray,
        safety_info: Mapping[str, Any],
        info: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Advance contact-rich state; unsafe contact-rich chunks pause instead of deform."""
        if not self.contact_rich_pause_enabled:
            return {}
        metadata_active, metadata_reason = self._contact_rich_metadata_signal(obs, info, kwargs)
        gripper_active, gripper_reason, gripper_value = self._contact_rich_gripper_signal(chunk)
        progress_allowed = self._task_progress_allows_contact_rich_pause(kwargs)
        raw_signal = bool(metadata_active or gripper_active)
        if not raw_signal:
            self._contact_rich_pause_signal_consumed = False
        effective_signal = bool(raw_signal and not self._contact_rich_pause_signal_consumed)
        timed_out_before_signal = bool(
            self.contact_rich_pause_max_steps > 0
            and self._contact_rich_pause_active
            and self._contact_rich_pause_steps >= self.contact_rich_pause_max_steps
        )
        signal = bool(effective_signal and progress_allowed and not timed_out_before_signal)
        reason = metadata_reason if metadata_active else gripper_reason
        if raw_signal and not progress_allowed:
            reason = "task_progress_gate"
        elif timed_out_before_signal:
            reason = "max_pause_timeout"
        horizon_safe = bool(safety_info.get("horizon_safe", False))
        self._contact_rich_pause_timed_out = False

        if signal:
            self._contact_rich_pause_signal_consumed = True
            self._contact_rich_pause_active = True
            self.contact_rich_state = "contact_rich_pause"
            self._contact_rich_pause_hold_remaining = max(
                int(self._contact_rich_pause_hold_remaining),
                int(self.contact_rich_pause_min_hold_steps),
            )
            self._contact_rich_pause_clear_steps = 0
            self._contact_rich_pause_last_reason = reason
        elif self._contact_rich_pause_active:
            timed_out = bool(
                self.contact_rich_pause_max_steps > 0
                and self._contact_rich_pause_steps >= self.contact_rich_pause_max_steps
            )
            if timed_out:
                self._contact_rich_pause_active = False
                self.contact_rich_state = "free"
                self._contact_rich_pause_hold_remaining = 0
                self._contact_rich_pause_clear_steps = 0
                self._contact_rich_pause_last_reason = "max_pause_timeout"
                self._contact_rich_pause_timed_out = True
            else:
                if horizon_safe:
                    self._contact_rich_pause_clear_steps += 1
                else:
                    self._contact_rich_pause_clear_steps = 0
                if self._contact_rich_pause_hold_remaining > 0:
                    self._contact_rich_pause_hold_remaining -= 1
                if (
                    self._contact_rich_pause_hold_remaining <= 0
                    and self._contact_rich_pause_clear_steps >= self.contact_rich_pause_exit_safe_steps
                ):
                    self._contact_rich_pause_active = False
                    self.contact_rich_state = "free"
                    self._contact_rich_pause_last_reason = "cleared"
        else:
            self.contact_rich_state = "free"
            self._contact_rich_pause_clear_steps = 0
            self._contact_rich_pause_hold_remaining = 0
            self._contact_rich_pause_last_reason = None

        if self._contact_rich_pause_active:
            self._contact_rich_pause_steps += 1
        else:
            self._contact_rich_pause_steps = 0

        self._contact_rich_pause_last_signal = signal
        self._contact_rich_pause_last_raw_signal = raw_signal
        self._contact_rich_pause_last_progress_allowed = progress_allowed
        self._contact_rich_pause_last_gripper_value = gripper_value
        return {
            "contact_rich_state": self.contact_rich_state,
            "contact_rich_pause_enabled": bool(self.contact_rich_pause_enabled),
            "contact_rich_pause_active": bool(self._contact_rich_pause_active),
            "contact_rich_pause_signal": bool(signal),
            "contact_rich_pause_raw_signal": bool(raw_signal),
            "contact_rich_deform_block_active": bool(self._contact_rich_deform_block_is_active()),
            "contact_rich_pause_reason": self._contact_rich_pause_last_reason,
            "contact_rich_pause_metadata_signal": bool(metadata_active),
            "contact_rich_pause_gripper_signal": bool(gripper_active),
            "contact_rich_pause_gripper_value": gripper_value,
            "contact_rich_pause_progress_allowed": bool(progress_allowed),
            "contact_rich_pause_hold_remaining": int(self._contact_rich_pause_hold_remaining),
            "contact_rich_pause_clear_steps": int(self._contact_rich_pause_clear_steps),
            "contact_rich_pause_steps": int(self._contact_rich_pause_steps),
            "contact_rich_pause_max_steps": int(self.contact_rich_pause_max_steps),
            "contact_rich_pause_timed_out": bool(self._contact_rich_pause_timed_out),
            "contact_rich_pause_signal_consumed": bool(self._contact_rich_pause_signal_consumed),
        }

    def _contact_rich_pause_is_active(self) -> bool:
        """Return whether the current unsafe chunk must pause instead of deform."""
        return bool(self.contact_rich_pause_enabled and self._contact_rich_pause_active)

    def _contact_rich_deform_block_is_active(self) -> bool:
        """Block spatial recovery/deform while contact-rich evidence is still present."""
        return bool(
            self.contact_rich_pause_enabled
            and self._contact_rich_pause_last_progress_allowed
            and (
                self._contact_rich_pause_active
                or self._contact_rich_pause_last_raw_signal
                or self._contact_rich_pause_signal_consumed
            )
        )

    def _contact_rich_gripper_only_clear_to_continue(
        self,
        info: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> bool:
        """Allow ACT to continue for gripper-only contact-rich signals when human clearance is safe."""
        if not (self.contact_rich_pause_enabled and self.contact_rich_gripper_only_continue_when_clear):
            return False
        if not bool(info.get("contact_rich_pause_raw_signal", self._contact_rich_pause_last_raw_signal)):
            return False
        if not bool(info.get("contact_rich_pause_gripper_signal", False)):
            return False
        if bool(info.get("contact_rich_pause_metadata_signal", False)):
            return False
        if not self._contact_rich_pause_last_progress_allowed:
            return False
        if bool(kwargs.get("live_monitor_h_violation", False)):
            return False

        contact_count = kwargs.get("robot_human_contact_count")
        if contact_count is None:
            contact_count = kwargs.get("current_robot_human_contact_count")
        if contact_count is None:
            contact_count = kwargs.get("contact_count")
        try:
            if contact_count is not None and int(contact_count) > 0:
                return False
        except (TypeError, ValueError):
            return False

        clearance = kwargs.get("live_monitor_min_clearance")
        if clearance is None:
            clearance = info.get("live_min_clearance")
        try:
            clearance_f = float(clearance)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(clearance_f):
            return False
        return bool(clearance_f >= self.contact_rich_gripper_only_min_clearance)

    def _perform_contact_rich_clear_pass_through(
        self,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        info: dict[str, Any],
        *,
        reason: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Keep executing ACT when contact-rich evidence is only a safe gripper command."""
        info.update(
            {
                "safety_mode": "pass_through",
                "mode": "pass_through",
                "deform_mode": None,
                "deformation_source": None,
                "deformation_norm": 0.0,
                "retiming_source": None,
                "retiming_norm": 0.0,
                "optimized_fallback": None,
                "fallback_reason": None,
                "contact_rich_pause_only": False,
                "contact_rich_deform_block_active": False,
                "deformation_blocked_by_contact_rich_state": False,
                "deformation_deferred": False,
                "contact_rich_clearance_pass_through": True,
                "contact_rich_clearance_pass_through_reason": reason,
                "horizon_raw_unsafe_ignored_due_to_contact_rich_clearance": True,
            }
        )
        self.last_info = info
        return np.asarray(chunk).reshape(original_shape), info

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return out

    def _normalize_resume_affordance_context(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize task-adapter resume features from filter kwargs.

        The filter only understands generic names.  Task-specific code should
        translate environment semantics into these fields before calling
        ``filter_chunk``.
        """
        context: dict[str, Any] = {}
        raw_context = kwargs.get("resume_affordance_context")
        if raw_context is not None:
            if hasattr(raw_context, "items"):
                context.update(dict(raw_context.items()))
            else:
                try:
                    context.update(dict(raw_context))
                except Exception:
                    context["resume_context_parse_error"] = type(raw_context).__name__
        for key in (
            "interaction_context",
            "resume_adapter",
            "resume_context_source",
            "resume_target_label",
            "resume_target_distance",
            "resume_target_contact",
            "resume_task_progress",
            "resume_task_progress_delta",
            "resume_alignment_score",
            "resume_alignment_cosine",
            "resume_control_continuity_score",
            "resume_gripper_closed",
        ):
            if key in kwargs and kwargs.get(key) is not None:
                context[key] = kwargs.get(key)
        if "resume_task_progress" not in context and kwargs.get("task_progress") is not None:
            context["resume_task_progress"] = kwargs.get("task_progress")
            context.setdefault("resume_context_source", "filter_kwargs.task_progress")
        if "interaction_context" not in context:
            contact_state = kwargs.get("contact_rich_state", getattr(self, "contact_rich_state", None))
            if contact_state is not None:
                context["interaction_context"] = contact_state
        return context

    def _rollout_prediction_untrusted(self, kwargs: Mapping[str, Any]) -> bool:
        """Return whether recent Bigym execution made q-space rollout low-confidence."""
        if not self.rollout_mismatch_mitigation_enabled:
            return False
        if bool(kwargs.get("rollout_prediction_untrusted", False)):
            return True
        residual_l2 = self._finite_float(kwargs.get("rollout_residual_l2"))
        residual_max_abs = self._finite_float(kwargs.get("rollout_residual_max_abs"))
        return bool(
            (residual_l2 is not None and residual_l2 >= self.rollout_mismatch_l2_threshold)
            or (
                residual_max_abs is not None
                and residual_max_abs >= self.rollout_mismatch_max_abs_threshold
            )
        )

    def _live_clear_to_continue(self, kwargs: Mapping[str, Any]) -> bool:
        """Gate rollout-mismatch escape on live signed clearance and actual contacts."""
        if bool(kwargs.get("live_monitor_h_violation", False)):
            return False
        contact_count = kwargs.get("robot_human_contact_count")
        if contact_count is None:
            contact_count = kwargs.get("current_robot_human_contact_count")
        if contact_count is None:
            contact_count = kwargs.get("contact_count")
        try:
            if contact_count is not None and int(contact_count) > 0:
                return False
        except (TypeError, ValueError):
            return False
        clearance = self._finite_float(kwargs.get("live_monitor_min_clearance"))
        if clearance is None:
            return False
        return bool(clearance >= self.rollout_mismatch_live_min_clearance)

    def _hard_executable_prefix_safe(self, safety_info: Mapping[str, Any]) -> bool:
        """Treat only the actually executed receding-horizon prefix as hard safety."""
        if bool(safety_info.get("horizon_safe", False)):
            return True
        prefix_len = max(1, int(self.hard_safety_prefix_len))
        first_violation = safety_info.get("first_violation")
        if first_violation is not None:
            try:
                return bool(int(first_violation) >= prefix_len)
            except (TypeError, ValueError):
                pass
        safe_prefix_len = safety_info.get("safe_prefix_len")
        if safe_prefix_len is not None:
            try:
                return bool(int(safe_prefix_len) >= prefix_len)
            except (TypeError, ValueError):
                pass
        clearances = safety_info.get("min_clearances")
        if clearances is not None:
            try:
                arr = np.asarray(clearances, dtype=np.float32).reshape(-1)
                if arr.size >= prefix_len:
                    threshold = float(getattr(self, "min_clearance", 0.0))
                    return bool(np.min(arr[:prefix_len]) >= threshold)
            except Exception:  # noqa: BLE001
                pass
        return False

    @staticmethod
    def _flatten_numeric(value: Any) -> np.ndarray | None:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:  # noqa: BLE001
            return None
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return arr.copy()

    def _human_signature(self, obs: Any, kwargs: Mapping[str, Any]) -> np.ndarray | None:
        keys = (
            "human_qpos",
            "human_q",
            "human_state",
            "human_pose",
            "human_position",
            "human_capsules",
            "human_body_pos",
            "human_arm_qpos",
        )
        for source in (kwargs, obs if isinstance(obs, Mapping) else {}):
            for key in keys:
                if key in source:
                    arr = self._flatten_numeric(source[key])
                    if arr is not None:
                        return arr
        return None

    def _nominal_signature(
        self,
        chunk: np.ndarray,
        q_seq: np.ndarray | None = None,
    ) -> np.ndarray | None:
        pieces: list[np.ndarray] = []
        try:
            action_arr = np.asarray(chunk, dtype=np.float32)
            if action_arr.ndim == 2 and action_arr.shape[0] > 0:
                horizon = min(4, action_arr.shape[0])
                valid = self.controlled_action_indices[
                    self.controlled_action_indices < action_arr.shape[1]
                ]
                if valid.size:
                    pieces.append(action_arr[:horizon, valid].reshape(-1))
                else:
                    pieces.append(action_arr[:horizon].reshape(-1))
        except Exception:  # noqa: BLE001
            pass
        try:
            q_arr = np.asarray(q_seq, dtype=np.float32)
            if q_arr.ndim == 2 and q_arr.shape[0] > 0:
                pieces.append(q_arr[-1].reshape(-1))
        except Exception:  # noqa: BLE001
            pass
        if not pieces:
            return None
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def _robot_signature(self, q_seq: np.ndarray | None) -> np.ndarray | None:
        try:
            q_arr = np.asarray(q_seq, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return None
        if q_arr.ndim != 2 or q_arr.shape[0] == 0:
            return None
        return q_arr[0].reshape(-1).copy()

    def _update_intervention_info(
        self,
        info: dict[str, Any],
        *,
        evaluation: DeformationEvaluation | None = None,
        unsafe_reason: UnsafeReason | str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if evaluation is not None:
            self.intervention_fsm.last_evaluation = evaluation
        if unsafe_reason is not None:
            self.intervention_fsm.unsafe_reason = self.intervention_fsm._coerce_unsafe_reason(unsafe_reason)
        info.update(self.intervention_fsm.diagnostics())
        if extra:
            info.update(dict(extra))

    def _classify_unsafe_reason(
        self,
        info: Mapping[str, Any],
        evaluation: DeformationEvaluation | None = None,
    ) -> UnsafeReason:
        source = info.get("nominal_collision_source")
        if source in {"current_human_geometry", "predicted_human_motion"}:
            return UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION
        if evaluation is not None and not evaluation.admissible:
            return UnsafeReason.DEFORMATION_INFEASIBLE
        return UnsafeReason.TRANSIENT_PATH_OBSTRUCTION

    def _candidate_progress(
        self,
        nominal_q_seq: np.ndarray | None,
        candidate_q_seq: np.ndarray | None,
        nominal_chunk: np.ndarray,
        candidate_chunk: np.ndarray,
    ) -> float:
        try:
            nominal_q = np.asarray(nominal_q_seq, dtype=np.float32)
            candidate_q = np.asarray(candidate_q_seq, dtype=np.float32)
            if nominal_q.ndim == 2 and candidate_q.ndim == 2 and nominal_q.shape[0] > 1 and candidate_q.shape[0] > 1:
                n = min(nominal_q.shape[1], candidate_q.shape[1])
                nominal_delta = nominal_q[min(nominal_q.shape[0] - 1, candidate_q.shape[0] - 1), :n] - nominal_q[0, :n]
                candidate_delta = candidate_q[min(candidate_q.shape[0] - 1, nominal_q.shape[0] - 1), :n] - candidate_q[0, :n]
                denom = float(np.linalg.norm(nominal_delta))
                if denom > 1e-8:
                    return float(np.dot(candidate_delta, nominal_delta) / denom)
        except Exception:  # noqa: BLE001
            pass
        try:
            cand = np.asarray(candidate_chunk, dtype=np.float32)
            nom = np.asarray(nominal_chunk, dtype=np.float32)
            valid = self.controlled_action_indices[self.controlled_action_indices < cand.shape[1]]
            horizon = min(cand.shape[0], nom.shape[0])
            if horizon > 0 and valid.size:
                return float(np.linalg.norm(cand[:horizon, valid] - cand[0:1, valid]))
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _candidate_velocity_ratio(
        self,
        nominal_chunk: np.ndarray,
        candidate_chunk: np.ndarray,
    ) -> float:
        try:
            nominal = np.asarray(nominal_chunk, dtype=np.float32)
            candidate = np.asarray(candidate_chunk, dtype=np.float32)
            if nominal.ndim != 2 or candidate.ndim != 2:
                return 0.0
            horizon = min(nominal.shape[0], candidate.shape[0], 4)
            valid = self.controlled_action_indices[self.controlled_action_indices < min(nominal.shape[1], candidate.shape[1])]
            if horizon <= 0 or valid.size == 0:
                return 0.0
            ratios = []
            eps = 1e-6
            for i in range(horizon):
                cand_norm = float(np.linalg.norm(candidate[i, valid]))
                nom_norm = float(np.linalg.norm(nominal[i, valid]))
                ratios.append(cand_norm / (nom_norm + eps))
            return float(min(ratios)) if ratios else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _evaluate_deformation_admissibility(
        self,
        obs: Any,
        nominal_chunk: np.ndarray,
        candidate_chunk: np.ndarray | None,
        nominal_q_seq: np.ndarray | None,
        info: Mapping[str, Any],
    ) -> DeformationEvaluation:
        del obs, nominal_q_seq
        if candidate_chunk is None:
            return DeformationEvaluation.rejected("missing_candidate")
        try:
            candidate = np.asarray(candidate_chunk, dtype=np.float32)
            if candidate.ndim != 2 or candidate.shape[0] == 0:
                return DeformationEvaluation.rejected("bad_candidate_shape")
        except Exception:  # noqa: BLE001
            return DeformationEvaluation.rejected("bad_candidate")

        # Reuse the optimizer/recovery diagnostics that were already computed by
        # the current SafeChunk pipeline.  Do not perform another simulator
        # rollout here; that can perturb the mirrored safety state and shift the
        # ACT resume timing we are trying to preserve.
        horizon_safe = bool(
            info.get("optimized_accepted", False)
            or info.get("recover_accepted", False)
            or info.get("return_accepted", False)
            or (
                info.get("deform_safe", False)
                and info.get("recover_path_safe", True)
                and info.get("recover_immediate_safe", True)
                and info.get("recover_prefix_safe", True)
            )
        )
        distance_candidates = (
            info.get("recover_path_min_clearance"),
            info.get("recover_min_clearance"),
            info.get("deform_min_clearance"),
            info.get("min_clearance"),
        )
        min_distance = float("-inf")
        for value in distance_candidates:
            try:
                value_f = float(value)
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(value_f):
                min_distance = value_f if not np.isfinite(min_distance) else min(min_distance, value_f)

        progress = 0.0
        for key in (
            "recover_act_progress_projection",
            "recover_projection_on_nominal",
            "recover_task_progress_score",
            "recover_resume_tube_progress_score",
            "mpc_handoff_progress_projection",
        ):
            try:
                value_f = float(info.get(key))
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(value_f):
                progress = max(progress, value_f)
        if progress <= 0.0:
            progress = self._candidate_progress(None, None, nominal_chunk, candidate)
        has_progress = bool(progress >= self.intervention_fsm.config.min_deform_progress)
        min_velocity_ratio = self._candidate_velocity_ratio(nominal_chunk, candidate)
        executable = bool(min_velocity_ratio >= self.intervention_fsm.config.min_deform_velocity_ratio)
        if not horizon_safe:
            reason = "horizon_unsafe"
        elif not has_progress:
            reason = "no_task_progress"
        elif not executable:
            reason = "immediate_safety_stop"
        else:
            reason = ""
        return DeformationEvaluation(
            admissible=bool(horizon_safe and has_progress and executable),
            horizon_safe=horizon_safe,
            has_progress=has_progress,
            executable=executable,
            min_distance=float(min_distance),
            progress=float(progress),
            min_velocity_ratio=float(min_velocity_ratio),
            failure_reason=reason,
        )

    def _latch_deformation_failure(
        self,
        info: dict[str, Any],
        *,
        reason: str,
        obs: Any,
        chunk: np.ndarray,
        q_seq: np.ndarray | None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        kwargs = kwargs or {}
        self.intervention_fsm.latch_failure(
            step=int(self.recovery.latest_nominal_step),
            reason=reason,
            human_state=self._human_signature(obs, kwargs),
            robot_state=self._robot_signature(q_seq),
            nominal_signature=self._nominal_signature(chunk, q_seq),
        )
        self._update_intervention_info(
            info,
            unsafe_reason=UnsafeReason.DEFORMATION_INFEASIBLE,
        )

    def _perform_fsm_safe_stop(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        safety_info: Mapping[str, Any],
        info: dict[str, Any],
        *,
        reason: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        braked_chunk, brake_info = self.brake.horizon_brake(obs, chunk, safety_info)
        info.update(brake_info)
        info.update(
            {
                "safety_mode": "horizon_brake",
                "mode": "horizon_brake",
                "deform_mode": "fsm_safe_stop_latched",
                "deformation_deferred": True,
                "fallback_used": True,
                "fallback_reason": reason,
                "safe_stop_latched": True,
            }
        )
        self._update_intervention_info(info)
        try:
            self.intervention_fsm.previous_safe_action = np.asarray(
                braked_chunk[0],
                dtype=np.float32,
            ).reshape(-1).copy()
        except Exception:  # noqa: BLE001
            pass
        slowdown_result = self._maybe_perform_pause_guard_slowdown(
            obs,
            chunk,
            original_shape,
            safety_info,
            info,
            deform_trigger_reason=reason,
            fallback_reason=reason,
            **kwargs,
        )
        if slowdown_result is not None:
            return slowdown_result
        self.last_info = info
        return self.brake._hold_return_or_emergency_deform(
            obs,
            chunk,
            braked_chunk,
            info,
            original_shape,
            **kwargs,
        )

    def _goal_block_release_wait_needed(
        self,
        info: dict[str, Any],
        *,
        goal_blocked: bool,
        kwargs: Mapping[str, Any],
    ) -> bool:
        """Hold briefly after a goal release while the human is still exiting.

        A goal hold must not be converted into deformation. When the goal
        region becomes clear, the human can still occupy the nominal path for
        a few frames. Waiting for the horizon to become safe preserves the
        stopped ACT history and avoids starting a recovery on an exit transient.
        """
        if goal_blocked:
            self._goal_block_hold_active = True
            self._goal_block_release_wait_count = 0
            return False
        if not self._goal_block_hold_active:
            return False
        phase = str(kwargs.get("human_phase", "")).strip().lower()
        if phase not in {"hold", "exit"}:
            self._goal_block_hold_active = False
            self._goal_block_release_wait_count = 0
            return False
        if self._goal_block_release_wait_count >= self.goal_block_release_wait_steps:
            info.update(
                {
                    "goal_block_release_wait_active": False,
                    "goal_block_release_wait_expired": True,
                    "goal_block_release_wait_count": int(
                        self._goal_block_release_wait_count
                    ),
                }
            )
            self._goal_block_hold_active = False
            return False
        self._goal_block_release_wait_count += 1
        info.update(
            {
                "goal_block_release_wait_active": True,
                "goal_block_release_wait_expired": False,
                "goal_block_release_wait_count": int(
                    self._goal_block_release_wait_count
                ),
                "nominal_blockage_route": "goal_block_release_wait",
                "goal_block_release_wait_reason": "human_exit_horizon_still_unsafe",
            }
        )
        return True

    @staticmethod
    def _prefixed_safety_info(prefix: str, safety_info: Mapping[str, Any]) -> dict[str, Any]:
        """Expose a compact, log-friendly view of one horizon safety check."""
        return {
            f"{prefix}_horizon_safe": bool(safety_info.get("horizon_safe", False)),
            f"{prefix}_min_clearance": safety_info.get("min_clearance"),
            f"{prefix}_first_violation": safety_info.get("first_violation"),
            f"{prefix}_unsafe_count": safety_info.get("unsafe_count"),
            f"{prefix}_safety_eval_available": bool(
                safety_info.get("safety_eval_available", False)
            ),
        }

    def _conflict_aware_enabled(self) -> bool:
        return bool(
            self.intervention_fsm.config.enabled
            and self.intervention_fsm.config.intervention_policy
            == InterventionPolicy.CONFLICT_AWARE.value
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _evaluate_stop_counterfactual(
        self,
        brake_info: Mapping[str, Any],
    ) -> StopCounterfactualResult:
        """Evaluate whether the existing brake/pause trajectory is safe."""
        if not self.intervention_fsm.config.stop_counterfactual_enabled:
            return StopCounterfactualResult(
                safe=False,
                min_distance=float("-inf"),
                violation_step=None,
                unsafe_count=None,
                source="disabled",
            )
        min_distance = self._finite_float(brake_info.get("brake_min_clearance"))
        if min_distance is None:
            min_distance = self._finite_float(brake_info.get("min_clearance"))
        if min_distance is None:
            min_distance = float("-inf")
        return StopCounterfactualResult(
            safe=bool(brake_info.get("brake_safe", False)),
            min_distance=float(min_distance),
            violation_step=self._optional_int(brake_info.get("brake_first_violation")),
            unsafe_count=self._optional_int(brake_info.get("brake_unsafe_count")),
        )

    def _goal_blockage_info(self, kwargs: Mapping[str, Any]) -> tuple[bool, float | None, bool, str]:
        """Classify optional task-goal blockage without task-specific solver coupling."""
        if not self.intervention_fsm.config.goal_block_check_enabled:
            return False, None, False, "disabled"
        context = self._normalize_resume_affordance_context(kwargs)
        for key in ("goal_region_blocked", "resume_goal_blocked", "task_goal_blocked"):
            if key in kwargs and kwargs.get(key) is not None:
                return bool(kwargs.get(key)), None, True, key
            if key in context and context.get(key) is not None:
                return bool(context.get(key)), None, True, f"resume_context.{key}"
        radius = float(self.intervention_fsm.config.goal_block_radius)
        for key in (
            "goal_region_human_distance",
            "human_goal_min_distance",
            "resume_target_human_distance",
        ):
            dist = self._finite_float(kwargs.get(key))
            if dist is None:
                dist = self._finite_float(context.get(key))
            if dist is not None:
                return bool(dist <= radius), float(dist), True, key

        goal = None
        for key in ("task_goal_position", "resume_target_position", "goal_position"):
            goal = self._flatten_numeric(kwargs.get(key))
            if goal is None:
                goal = self._flatten_numeric(context.get(key))
            if goal is not None:
                break
        human = None
        for key in (
            "human_hand_position",
            "human_palm_position",
            "human_position",
            "human_goal_position",
        ):
            human = self._flatten_numeric(kwargs.get(key))
            if human is None:
                human = self._flatten_numeric(context.get(key))
            if human is not None:
                break
        if goal is None or human is None:
            return False, None, False, "unavailable"
        n = min(goal.size, human.size)
        if n <= 0:
            return False, None, False, "unavailable"
        dist = float(np.linalg.norm(goal[:n] - human[:n]))
        return bool(dist <= radius), dist, True, "goal_human_distance"

    def _path_blockage_info(
        self,
        safety_info: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> tuple[bool, bool, str]:
        """Classify whether the nominal ACT horizon is blocked on its path."""
        for key in (
            "path_blocked",
            "trajectory_blocked",
            "human_path_blocked",
            "path_obstructed",
        ):
            if key in kwargs:
                return bool(kwargs.get(key)), True, key
        available = bool(safety_info.get("safety_eval_available", False))
        if not available:
            return False, False, "unavailable"
        return not bool(safety_info.get("horizon_safe", False)), True, "nominal_horizon"

    def _path_block_pause_sufficiency_info(
        self,
        kwargs: Mapping[str, Any],
    ) -> tuple[bool, bool, str]:
        """Return explicit evidence that a path blocker will clear while paused.

        Stop-counterfactual safety only establishes physical safety. It does not
        establish task liveness, so a blocked path defaults to requiring a bypass
        unless the caller explicitly predicts clearance within the pause budget.
        """
        context = self._normalize_resume_affordance_context(kwargs)
        for key in (
            "path_block_clear_within_pause_budget",
            "path_block_predicted_clear",
            "human_path_predicted_clear",
            "temporary_path_blocker_predicted_clear",
        ):
            if key in kwargs and kwargs.get(key) is not None:
                return self._signal_value_active(kwargs.get(key)), True, key
            if key in context and context.get(key) is not None:
                return (
                    self._signal_value_active(context.get(key)),
                    True,
                    f"resume_context.{key}",
                )
        return False, False, "unavailable"

    @staticmethod
    def _stop_pause_is_task_sufficient(
        *,
        stop_safe: bool,
        path_blocked: bool,
        path_block_pause_sufficient: bool,
    ) -> bool:
        """Require liveness evidence before using a safe stop for path blockage."""
        return bool(stop_safe and (not path_blocked or path_block_pause_sufficient))

    def _candidate_terminal_deviation(
        self,
        nominal_chunk: np.ndarray,
        candidate_chunk: np.ndarray | None,
    ) -> float:
        if candidate_chunk is None:
            return float("inf")
        try:
            nominal = np.asarray(nominal_chunk, dtype=np.float32)
            candidate = np.asarray(candidate_chunk, dtype=np.float32)
            if nominal.ndim != 2 or candidate.ndim != 2:
                return float("inf")
            horizon = min(nominal.shape[0], candidate.shape[0])
            if horizon <= 0:
                return float("inf")
            valid = self.controlled_action_indices[
                self.controlled_action_indices < min(nominal.shape[1], candidate.shape[1])
            ]
            if valid.size == 0:
                return 0.0
            return float(np.linalg.norm(candidate[horizon - 1, valid] - nominal[horizon - 1, valid]))
        except Exception:  # noqa: BLE001
            return float("inf")

    @staticmethod
    def _info_bool_if_available(info: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[bool, bool]:
        for key in keys:
            value = info.get(key)
            if value is not None:
                return bool(value), True
        return True, False

    def _evaluate_conflict_aware_deformation_admissibility(
        self,
        nominal_chunk: np.ndarray,
        candidate_chunk: np.ndarray | None,
        base: DeformationEvaluation,
        info: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> DeformationAdmissibilityResult:
        goal_blocked, goal_distance, goal_available, goal_source = self._goal_blockage_info(kwargs)
        terminal_deviation = self._candidate_terminal_deviation(nominal_chunk, candidate_chunk)
        terminal_ok = bool(
            terminal_deviation <= float(self.intervention_fsm.config.max_terminal_deviation)
        )
        progress_value = max(float(base.progress), self._finite_float(kwargs.get("task_progress_delta")) or 0.0)
        progress_ok = bool(progress_value >= float(self.intervention_fsm.config.min_task_progress))
        resumable, resumable_available = self._info_bool_if_available(
            info,
            (
                "act_resumable_ok",
                "recover_resume_tube_ok",
                "mpc_handoff_resume_tube_ok",
                "resume_affordance_ok",
            ),
        )
        deformation_safe = bool(base.horizon_safe)
        if not deformation_safe:
            reason = "DEFORMATION_UNSAFE"
        elif goal_blocked:
            reason = "GOAL_REGION_BLOCKED"
        elif not progress_ok:
            reason = "TASK_PROGRESS_INSUFFICIENT"
        elif not terminal_ok:
            reason = "TERMINAL_DEVIATION_EXCEEDED"
        elif not resumable:
            reason = "DEFORMATION_NOT_RESUMABLE"
        else:
            reason = "PATH_BYPASS_ADMISSIBLE"
        bypassable = bool(
            deformation_safe
            and not goal_blocked
            and progress_ok
            and terminal_ok
            and resumable
        )
        return DeformationAdmissibilityResult(
            safe=deformation_safe,
            bypassable=bypassable,
            goal_blocked=bool(goal_blocked),
            progress_ok=progress_ok,
            terminal_deviation_ok=terminal_ok,
            resumable=bool(resumable),
            reason=reason,
            task_progress_value=float(progress_value),
            terminal_deviation=float(terminal_deviation),
            goal_distance=goal_distance,
            goal_check_available=bool(goal_available),
        )

    def _update_conflict_aware_info(
        self,
        info: dict[str, Any],
        *,
        stop_result: StopCounterfactualResult | None = None,
        admissibility: DeformationAdmissibilityResult | None = None,
        selected_mode: str,
        decision_reason: str,
        nominal_safety: Mapping[str, Any] | None = None,
        pause_budget_used: int | None = None,
    ) -> None:
        cfg = self.intervention_fsm.config
        commit_remaining = max(0, int(cfg.deform_commit_steps) - int(self.intervention_fsm.deform_commit_counter))
        info.update(
            {
                "conflict_aware_policy_enabled": True,
                "conflict_aware_previous_mode": self.intervention_fsm.mode.value,
                "conflict_aware_selected_mode": selected_mode,
                "conflict_aware_decision_reason": decision_reason,
                "conflict_aware_nominal_min_distance": (
                    None if nominal_safety is None else nominal_safety.get("min_clearance")
                ),
                "conflict_aware_stop_min_distance": (
                    None if stop_result is None else float(stop_result.min_distance)
                ),
                "conflict_aware_stopping_sufficient": (
                    None if stop_result is None else bool(stop_result.safe)
                ),
                "conflict_aware_goal_blocked": (
                    None if admissibility is None else bool(admissibility.goal_blocked)
                ),
                "conflict_aware_deformation_safe": (
                    None if admissibility is None else bool(admissibility.safe)
                ),
                "conflict_aware_deformation_admissible": (
                    None if admissibility is None else bool(admissibility.bypassable)
                ),
                "conflict_aware_task_progress_value": (
                    None if admissibility is None else float(admissibility.task_progress_value)
                ),
                "conflict_aware_terminal_deviation": (
                    None if admissibility is None else float(admissibility.terminal_deviation)
                ),
                "conflict_aware_pause_budget_used": pause_budget_used,
                "conflict_aware_commit_steps_remaining": commit_remaining,
                "conflict_aware_resume_hysteresis_count": int(
                    self.intervention_fsm.nominal_clear_counter
                ),
                "conflict_aware_goal_distance": (
                    None if admissibility is None else admissibility.goal_distance
                ),
                "conflict_aware_goal_check_available": (
                    None if admissibility is None else bool(admissibility.goal_check_available)
                ),
            }
        )

    def _nominal_collision_attribution(
        self,
        obs: Any,
        q_seq: np.ndarray,
        predicted_safety: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attribute nominal ACT risk to current-human geometry or predicted motion."""
        predicted_safe = bool(predicted_safety.get("horizon_safe", False))
        predicted_available = bool(predicted_safety.get("safety_eval_available", False))
        static_safety = self.intervention_factory.evaluate_horizon_safety(
            obs,
            q_seq,
            predict_human_motion=False,
        )
        static_safe = bool(static_safety.get("horizon_safe", False))
        static_available = bool(static_safety.get("safety_eval_available", False))

        if not predicted_available or not static_available:
            source = "unknown"
        elif static_safe and predicted_safe:
            source = "none"
        elif static_safe and not predicted_safe:
            source = "predicted_human_motion"
        elif not static_safe and not predicted_safe:
            source = "current_human_geometry"
        else:
            source = "prediction_inconsistent"

        pause_recommended = source == "current_human_geometry"
        deform_recommended = source == "predicted_human_motion"
        info = {
            "nominal_collision_attribution_enabled": True,
            "nominal_collision_source": source,
            "nominal_current_human_unsafe": bool(static_available and not static_safe),
            "nominal_predicted_human_unsafe": bool(
                predicted_available and not predicted_safe
            ),
            "nominal_predicted_only_collision": bool(deform_recommended),
            "nominal_attribution_pause_recommended": bool(pause_recommended),
            "nominal_attribution_deform_recommended": bool(deform_recommended),
        }
        info.update(self._prefixed_safety_info("nominal_current_human", static_safety))
        info.update(
            self._prefixed_safety_info(
                "nominal_predicted_human",
                predicted_safety,
            )
        )
        return info

    def _arm_pause_exit_smoothing(
        self,
        anchor_chunk: Any | None,
        info: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        """Start a short safety-gated action ramp when leaving a pause."""
        info.update(
            {
                "pause_exit_smoothing_enabled": bool(
                    self.pause_exit_smoothing_enabled
                ),
                "pause_exit_smoothing_arm_reason": str(reason),
            }
        )
        if (
            not self.pause_exit_smoothing_enabled
            or self.pause_exit_smoothing_steps <= 0
            or self._pause_exit_smoothing_rearm_blocked
        ):
            info.update(
                {
                    "pause_exit_smoothing_armed": False,
                    "pause_exit_smoothing_remaining": int(
                        self._pause_exit_smoothing_remaining
                    ),
                }
            )
            return
        if anchor_chunk is None:
            info.update(
                {
                    "pause_exit_smoothing_armed": False,
                    "pause_exit_smoothing_skip_reason": "missing_anchor",
                }
            )
            return
        try:
            anchor_arr, _ = self.intervention_factory._as_chunk(anchor_chunk)
        except Exception:  # noqa: BLE001
            info.update(
                {
                    "pause_exit_smoothing_armed": False,
                    "pause_exit_smoothing_skip_reason": "bad_anchor",
                }
            )
            return
        if anchor_arr.shape[0] == 0:
            info.update(
                {
                    "pause_exit_smoothing_armed": False,
                    "pause_exit_smoothing_skip_reason": "empty_anchor",
                }
            )
            return
        self._pause_exit_smoothing_anchor_action = np.asarray(
            anchor_arr[0],
            dtype=np.float32,
        ).copy()
        self._pause_exit_smoothing_total_steps = int(self.pause_exit_smoothing_steps)
        self._pause_exit_smoothing_remaining = int(self.pause_exit_smoothing_steps)
        self._pause_exit_smoothing_rearm_blocked = True
        info.update(
            {
                "pause_exit_smoothing_armed": True,
                "pause_exit_smoothing_remaining": int(
                    self._pause_exit_smoothing_remaining
                ),
            }
        )

    def _reset_pause_exit_smoothing_rearm(self) -> None:
        """Allow a future pause episode to arm its own exit ramp."""
        self._pause_exit_smoothing_rearm_blocked = False

    def _apply_pause_exit_smoothing(
        self,
        obs: Any,
        action_chunk: Any,
        original_shape: tuple[int, ...],
        info: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Blend the first post-pause action toward the previous hold action."""
        info.setdefault(
            "pause_exit_smoothing_enabled",
            bool(self.pause_exit_smoothing_enabled),
        )
        if (
            not self.pause_exit_smoothing_enabled
            or self._pause_exit_smoothing_remaining <= 0
            or self._pause_exit_smoothing_anchor_action is None
        ):
            info.setdefault("pause_exit_smoothing_applied", False)
            return np.asarray(action_chunk, dtype=np.float32).reshape(original_shape), info

        try:
            chunk, _ = self.intervention_factory._as_chunk(action_chunk)
        except Exception:  # noqa: BLE001
            info.update(
                {
                    "pause_exit_smoothing_applied": False,
                    "pause_exit_smoothing_skip_reason": "bad_chunk",
                }
            )
            return np.asarray(action_chunk, dtype=np.float32).reshape(original_shape), info
        if chunk.shape[0] == 0:
            info.update(
                {
                    "pause_exit_smoothing_applied": False,
                    "pause_exit_smoothing_skip_reason": "empty_chunk",
                }
            )
            return chunk.reshape(original_shape), info

        anchor = np.asarray(self._pause_exit_smoothing_anchor_action, dtype=np.float32).reshape(-1)
        target = np.asarray(chunk[0], dtype=np.float32).reshape(-1)
        if anchor.shape != target.shape:
            info.update(
                {
                    "pause_exit_smoothing_applied": False,
                    "pause_exit_smoothing_skip_reason": "shape_mismatch",
                }
            )
            self._pause_exit_smoothing_remaining = 0
            return chunk.reshape(original_shape), info

        step_index = int(self._pause_exit_smoothing_total_steps - self._pause_exit_smoothing_remaining)
        alpha = float(
            min(
                1.0,
                max(
                    0.0,
                    self.pause_exit_smoothing_first_alpha
                    + step_index * self.pause_exit_smoothing_alpha_increment,
                ),
            )
        )
        controlled = np.asarray(self.controlled_action_indices, dtype=np.int64)
        controlled = controlled[(controlled >= 0) & (controlled < target.shape[0])]
        if controlled.size == 0:
            info.update(
                {
                    "pause_exit_smoothing_applied": False,
                    "pause_exit_smoothing_skip_reason": "no_controlled_indices",
                }
            )
            self._pause_exit_smoothing_remaining = 0
            return chunk.reshape(original_shape), info

        candidate = chunk.copy()
        smoothed_first = target.copy()
        smoothed_first[controlled] = (
            (1.0 - alpha) * anchor[controlled] + alpha * target[controlled]
        )
        candidate[0] = smoothed_first.astype(np.float32, copy=False)

        original_clearance = None
        smoothed_clearance = None
        try:
            original_q = self.intervention_factory.rollout_nominal_chunk(obs, chunk[:1])
            original_safety = self.intervention_factory.evaluate_horizon_safety(
                obs,
                original_q,
            )
            smoothed_q = self.intervention_factory.rollout_nominal_chunk(obs, candidate[:1])
            smoothed_safety = self.intervention_factory.evaluate_horizon_safety(
                obs,
                smoothed_q,
            )
            original_clearance = float(original_safety.get("min_clearance"))
            smoothed_clearance = float(smoothed_safety.get("min_clearance"))
            available = bool(
                original_safety.get("safety_eval_available", False)
                and smoothed_safety.get("safety_eval_available", False)
            )
        except Exception:  # noqa: BLE001
            available = False

        if not available:
            info.update(
                {
                    "pause_exit_smoothing_applied": False,
                    "pause_exit_smoothing_skip_reason": "safety_eval_unavailable",
                }
            )
            self._pause_exit_smoothing_remaining = 0
            return chunk.reshape(original_shape), info

        tolerance = float(self.pause_exit_smoothing_clearance_tolerance)
        if smoothed_clearance + tolerance < original_clearance:
            info.update(
                {
                    "pause_exit_smoothing_applied": False,
                    "pause_exit_smoothing_skip_reason": "clearance_worse_than_original",
                    "pause_exit_smoothing_alpha": alpha,
                    "pause_exit_smoothing_original_min_clearance": original_clearance,
                    "pause_exit_smoothing_smoothed_min_clearance": smoothed_clearance,
                    "pause_exit_smoothing_action_l2_before": float(
                        np.linalg.norm(target[controlled] - anchor[controlled])
                    ),
                }
            )
            self._pause_exit_smoothing_remaining = 0
            return chunk.reshape(original_shape), info

        before_l2 = float(np.linalg.norm(target[controlled] - anchor[controlled]))
        after_l2 = float(np.linalg.norm(smoothed_first[controlled] - anchor[controlled]))
        self._pause_exit_smoothing_remaining = max(
            0,
            int(self._pause_exit_smoothing_remaining) - 1,
        )
        if self._pause_exit_smoothing_remaining <= 0:
            self._pause_exit_smoothing_anchor_action = None
        info.update(
            {
                "pause_exit_smoothing_applied": True,
                "pause_exit_smoothing_alpha": alpha,
                "pause_exit_smoothing_step_index": int(step_index),
                "pause_exit_smoothing_remaining": int(
                    self._pause_exit_smoothing_remaining
                ),
                "pause_exit_smoothing_action_l2_before": before_l2,
                "pause_exit_smoothing_action_l2_after": after_l2,
                "pause_exit_smoothing_original_min_clearance": original_clearance,
                "pause_exit_smoothing_smoothed_min_clearance": smoothed_clearance,
            }
        )
        return candidate.reshape(original_shape), info

    def _maybe_apply_resume_blend(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        info: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        if (
            not self.intervention_fsm.config.enabled
            or self.intervention_fsm.mode != InterventionMode.RESUME_BLEND
        ):
            return None
        anchor = self.intervention_fsm.previous_safe_action
        if anchor is None:
            self.intervention_fsm.transition(
                InterventionMode.NOMINAL,
                "resume_blend_missing_anchor",
                UnsafeReason.NONE,
            )
            self._update_intervention_info(info)
            return None
        candidate = np.asarray(chunk, dtype=np.float32).copy()
        if candidate.ndim != 2 or candidate.shape[0] == 0:
            return None
        anchor_arr = np.asarray(anchor, dtype=np.float32).reshape(-1)
        if anchor_arr.shape[0] != candidate.shape[1]:
            self.intervention_fsm.transition(
                InterventionMode.NOMINAL,
                "resume_blend_shape_mismatch",
                UnsafeReason.NONE,
            )
            self._update_intervention_info(info)
            return None
        self.intervention_fsm.resume_blend_counter += 1
        alpha = min(
            1.0,
            float(self.intervention_fsm.resume_blend_counter)
            / float(max(1, self.intervention_fsm.config.resume_blend_steps)),
        )
        valid = self.controlled_action_indices[
            self.controlled_action_indices < candidate.shape[1]
        ]
        if valid.size:
            candidate[0, valid] = (
                (1.0 - alpha) * anchor_arr[valid]
                + alpha * candidate[0, valid]
            )
        try:
            blended_q_seq = self.deform.rollout_nominal_chunk(obs, candidate)
            blended_safety = self.intervention_factory.evaluate_horizon_safety(
                obs,
                blended_q_seq,
            )
        except Exception:  # noqa: BLE001
            blended_safety = {"horizon_safe": False, "min_clearance": float("-inf")}
        if not bool(blended_safety.get("horizon_safe", False)):
            self.intervention_fsm.transition(
                InterventionMode.PAUSE_GUARD,
                "resume_blend_nominal_unsafe",
                UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
            )
            info.update(blended_safety)
            self._update_intervention_info(
                info,
                extra={"intervention_resume_blend_blocked": True},
            )
            return self._perform_fsm_safe_stop(
                obs,
                chunk,
                original_shape,
                blended_safety,
                info,
                reason="resume_blend_nominal_unsafe",
            )
        if alpha >= 1.0:
            self.intervention_fsm.transition(
                InterventionMode.NOMINAL,
                "resume_blend_complete",
                UnsafeReason.NONE,
            )
        info.update(
            {
                "intervention_resume_blend_active": True,
                "intervention_resume_blend_alpha": float(alpha),
                "intervention_resume_blend_step": int(
                    self.intervention_fsm.resume_blend_counter
                ),
                "intervention_resume_blend_steps": int(
                    self.intervention_fsm.config.resume_blend_steps
                ),
                "intervention_resume_blend_min_clearance": blended_safety.get(
                    "min_clearance"
                ),
            }
        )
        self._update_intervention_info(info)
        self.brake._update_last_safe_execution(obs, candidate, info)
        self.intervention_fsm.previous_safe_action = np.asarray(
            candidate[0],
            dtype=np.float32,
        ).reshape(-1).copy()
        self.last_info = info
        return candidate.reshape(original_shape), info

    def _maybe_perform_policy_collision_slowdown(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        safety_info: Mapping[str, Any],
        info: dict[str, Any],
        *,
        deform_trigger_reason: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        if not bool(self.intervention_fsm.config.policy_collision_slowdown_enabled):
            info.update(
                {
                    "policy_collision_slowdown_enabled": False,
                    "policy_collision_slowdown_applied": False,
                    "policy_collision_slowdown_skip_reason": "slowdown_disabled",
                    "policy_collision_slowdown_counter": int(
                        self.intervention_fsm.slowdown_counter
                    ),
                    "policy_collision_slowdown_max_steps": int(
                        self.intervention_fsm.config.policy_collision_slowdown_max_steps
                    ),
                    "policy_collision_slowdown_min_first_violation": int(
                        self.intervention_fsm.config.policy_collision_slowdown_min_first_violation
                    ),
                }
            )
            return None

        min_first_violation = int(
            self.intervention_fsm.config.policy_collision_slowdown_min_first_violation
        )
        first_violation_raw = None if safety_info is None else safety_info.get("first_violation")
        try:
            first_violation_idx = int(first_violation_raw)
        except Exception:  # noqa: BLE001
            first_violation_idx = -1
        if first_violation_idx < min_first_violation:
            info.update(
                {
                    "policy_collision_slowdown_enabled": bool(
                        self.intervention_fsm.config.policy_collision_slowdown_enabled
                    ),
                    "policy_collision_slowdown_applied": False,
                    "policy_collision_slowdown_skip_reason": "first_violation_too_soon",
                    "policy_collision_slowdown_first_violation": (
                        first_violation_idx if first_violation_idx >= 0 else None
                    ),
                    "policy_collision_slowdown_min_first_violation": min_first_violation,
                    "policy_collision_slowdown_counter": int(
                        self.intervention_fsm.slowdown_counter
                    ),
                    "policy_collision_slowdown_max_steps": int(
                        self.intervention_fsm.config.policy_collision_slowdown_max_steps
                    ),
                }
            )
            return None

        if not self.intervention_fsm.can_slowdown_for_policy_collision():
            info.update(
                {
                    "policy_collision_slowdown_enabled": bool(
                        self.intervention_fsm.config.policy_collision_slowdown_enabled
                    ),
                    "policy_collision_slowdown_applied": False,
                    "policy_collision_slowdown_skip_reason": "slowdown_budget_exhausted",
                    "policy_collision_slowdown_counter": int(
                        self.intervention_fsm.slowdown_counter
                    ),
                    "policy_collision_slowdown_max_steps": int(
                        self.intervention_fsm.config.policy_collision_slowdown_max_steps
                    ),
                }
            )
            return None

        slowed_chunk, slowdown_info = self.brake.horizon_slowdown(
            obs,
            chunk,
            safety_info,
            factors=self.policy_collision_slowdown_factors,
        )
        info.update(slowdown_info)
        info.update(
            {
                "policy_collision_slowdown_enabled": True,
                "policy_collision_slowdown_counter": int(
                    self.intervention_fsm.slowdown_counter
                ),
                "policy_collision_slowdown_max_steps": int(
                    self.intervention_fsm.config.policy_collision_slowdown_max_steps
                ),
            }
        )
        if not bool(slowdown_info.get("slowdown_safe", False)):
            info.update(
                {
                    "policy_collision_slowdown_applied": False,
                    "policy_collision_slowdown_skip_reason": slowdown_info.get(
                        "slowdown_skip_reason",
                        "unsafe_slowdown",
                    ),
                }
            )
            return None

        self._reset_pause_exit_smoothing_rearm()
        self.brake.brake_streak += 1
        self.intervention_fsm.transition(
            InterventionMode.SLOWDOWN_GUARD,
            "policy_induced_collision_slowdown",
            UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
        )
        self.intervention_fsm.note_slowdown_step()
        try:
            self.intervention_fsm.previous_safe_action = np.asarray(
                slowed_chunk[0],
                dtype=np.float32,
            ).reshape(-1).copy()
        except Exception:  # noqa: BLE001
            pass
        info.update(
            {
                "safety_mode": "horizon_slowdown",
                "mode": "horizon_slowdown",
                "deform_mode": "policy_induced_collision_slowdown",
                "deformation_source": "horizon_slowdown",
                "deformation_deferred": True,
                "fallback_used": False,
                "fallback_reason": "policy_collision_slowdown",
                "nominal_attribution_pause_applied": False,
                "policy_collision_slowdown_applied": True,
                "policy_collision_slowdown_trigger_reason": deform_trigger_reason,
            }
        )
        info.update(
            self.brake._temporary_streak_info(
                trigger_reason=deform_trigger_reason,
                waiting=True,
            )
        )
        self._update_intervention_info(info)
        self.brake._update_last_safe_execution(obs, slowed_chunk, info, **kwargs)
        self.last_info = info
        return slowed_chunk.reshape(original_shape), info

    def _maybe_perform_pause_guard_slowdown(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        safety_info: Mapping[str, Any],
        info: dict[str, Any],
        *,
        deform_trigger_reason: str,
        fallback_reason: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        if not self.intervention_fsm.can_slowdown_for_pause_guard():
            info.setdefault("slowdown_applied", False)
            if not bool(self.intervention_fsm.config.pause_guard_slowdown_enabled):
                info.setdefault("slowdown_skip_reason", "pause_guard_slowdown_disabled")
            else:
                info.setdefault("slowdown_skip_reason", "pause_guard_slowdown_budget_exhausted")
            return None

        slowed_chunk, slowdown_info = self.brake.horizon_slowdown(
            obs,
            chunk,
            safety_info,
            factors=self.pause_guard_slowdown_factors,
        )
        info.update(slowdown_info)
        info.update(
            {
                "pause_guard_slowdown_enabled": bool(
                    self.intervention_fsm.config.pause_guard_slowdown_enabled
                ),
                "pause_guard_slowdown_counter": int(
                    self.intervention_fsm.slowdown_counter
                ),
                "pause_guard_slowdown_max_steps": int(
                    self.intervention_fsm.config.pause_guard_slowdown_max_steps
                ),
            }
        )
        if not bool(slowdown_info.get("slowdown_safe", False)):
            info.update(
                {
                    "slowdown_applied": False,
                    "slowdown_skip_reason": slowdown_info.get(
                        "slowdown_skip_reason",
                        "pause_guard_slowdown_unsafe",
                    ),
                }
            )
            return None

        self._reset_pause_exit_smoothing_rearm()
        self.brake.brake_streak += 1
        self.intervention_fsm.transition(
            InterventionMode.SLOWDOWN_GUARD,
            "pause_guard_gradual_brake",
            UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
        )
        self.intervention_fsm.note_slowdown_step()
        try:
            self.intervention_fsm.previous_safe_action = np.asarray(
                slowed_chunk[0],
                dtype=np.float32,
            ).reshape(-1).copy()
        except Exception:  # noqa: BLE001
            pass
        info.update(
            {
                "safety_mode": "horizon_slowdown",
                "mode": "horizon_slowdown",
                "deform_mode": "pause_guard_gradual_brake",
                "deformation_source": "horizon_slowdown",
                "deformation_deferred": True,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "nominal_attribution_pause_applied": False,
                "pause_guard_slowdown_applied": True,
                "pause_guard_slowdown_trigger_reason": deform_trigger_reason,
            }
        )
        info.update(
            self.brake._temporary_streak_info(
                trigger_reason=deform_trigger_reason,
                waiting=True,
            )
        )
        self._update_intervention_info(info)
        self.brake._update_last_safe_execution(obs, slowed_chunk, info, **kwargs)
        self.last_info = info
        return slowed_chunk.reshape(original_shape), info

    def _perform_attributed_nominal_pause(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        info: dict[str, Any],
        *,
        braked_chunk: np.ndarray,
        brake_info: Mapping[str, Any] | None = None,
        deform_trigger_reason: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Pause instead of deforming when the nominal ACT path is unsafe now."""
        self._reset_pause_exit_smoothing_rearm()
        self.brake.brake_streak += 1
        info.update(
            {
                "safety_mode": "horizon_brake",
                "mode": "horizon_brake",
                "deform_mode": "attributed_nominal_pause",
                "deformation_source": "horizon_brake",
                "deformation_deferred": True,
                "fallback_used": True,
                "fallback_reason": "nominal_current_human_collision_pause",
                "nominal_attribution_pause_applied": True,
            }
        )
        info.update(
            self.brake._temporary_streak_info(
                trigger_reason=deform_trigger_reason,
                waiting=False,
            )
        )
        slowdown_result = self._maybe_perform_pause_guard_slowdown(
            obs,
            chunk,
            original_shape,
            info,
            info,
            deform_trigger_reason=deform_trigger_reason,
            fallback_reason="nominal_current_human_collision_pause",
            **kwargs,
        )
        if slowdown_result is not None:
            return slowdown_result
        return self.brake._hold_return_or_emergency_deform(
            obs,
            chunk,
            braked_chunk,
            info,
            original_shape,
            **kwargs,
        )

    def _maybe_soft_full_horizon_pass_through(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        info: dict[str, Any],
        safety_info: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Ignore full-horizon-only risk when the executable prefix is live-safe."""
        if not self.full_horizon_soft_when_prefix_safe:
            return None
        if bool(safety_info.get("horizon_safe", False)):
            return None
        prefix_safe = self._hard_executable_prefix_safe(safety_info)
        live_clear = self._live_clear_to_continue(kwargs)
        info.update(
            {
                "hard_safety_prefix_len": int(self.hard_safety_prefix_len),
                "hard_executable_prefix_safe": bool(prefix_safe),
                "full_horizon_soft_when_prefix_safe": bool(
                    self.full_horizon_soft_when_prefix_safe
                ),
                "full_horizon_soft_live_clear": bool(live_clear),
            }
        )
        if not (prefix_safe and live_clear):
            return None
        self.unsafe_streak = 0
        self.brake.brake_streak = 0
        info.update(
            {
                "safety_mode": "pass_through",
                "mode": "pass_through",
                "deform_mode": None,
                "deformation_source": None,
                "optimized_fallback": None,
                "fallback_reason": None,
                "fallback_used": False,
                "full_horizon_soft_pass_through": True,
                "horizon_unsafe_ignored_due_to_executable_prefix_safe": True,
            }
        )
        try:
            self.brake._update_last_safe_execution(obs, chunk, info, **dict(kwargs))
        except Exception:  # noqa: BLE001
            pass
        self.last_info = info
        return chunk.reshape(original_shape), info

    def _apply_rollout_residual_correction(
        self,
        q_seq: np.ndarray,
        kwargs: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Lightly bias the next q rollout by the latest Bigym one-step residual."""
        info: dict[str, Any] = {
            "rollout_residual_correction_applied": False,
            "rollout_prediction_untrusted": self._rollout_prediction_untrusted(kwargs),
        }
        if not (
            self.rollout_mismatch_mitigation_enabled
            and self.rollout_residual_correction_enabled
        ):
            return q_seq, info
        residual = kwargs.get("rollout_residual_state")
        if residual is None:
            return q_seq, info
        try:
            residual_arr = np.asarray(residual, dtype=np.float32).reshape(-1)
            q_arr = np.asarray(q_seq, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return q_seq, info
        if residual_arr.size == 0 or q_arr.size == 0:
            return q_seq, info
        original_shape = q_arr.shape
        flat = q_arr.reshape(1, -1) if q_arr.ndim == 1 else q_arr.reshape(q_arr.shape[0], -1)
        n = min(int(flat.shape[1]), int(residual_arr.size))
        if n <= 0:
            return q_seq, info
        max_abs = max(0.0, float(self.rollout_residual_correction_max_abs))
        residual_used = residual_arr[:n]
        if max_abs > 0.0:
            residual_used = np.clip(residual_used, -max_abs, max_abs)
        corrected = flat.copy()
        scale = float(self.rollout_residual_correction_scale)
        decay = float(self.rollout_residual_correction_decay)
        for t in range(corrected.shape[0]):
            corrected[t, :n] += scale * (decay ** t) * residual_used
        corrected = corrected.reshape(original_shape).astype(q_arr.dtype, copy=False)
        info.update(
            {
                "rollout_residual_correction_applied": True,
                "rollout_residual_correction_scale": scale,
                "rollout_residual_correction_decay": decay,
                "rollout_residual_correction_max_abs": max_abs,
                "rollout_residual_state_l2": float(np.linalg.norm(residual_arr)),
                "rollout_residual_state_max_abs": float(np.max(np.abs(residual_arr))),
                "rollout_residual_corrected_dims": int(n),
            }
        )
        return corrected, info

    def _maybe_rollout_mismatch_pass_through(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        info: dict[str, Any],
        kwargs: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Escape repeated recover/deform loops when only stale rollout predicts danger."""
        prediction_untrusted = self._rollout_prediction_untrusted(kwargs)
        live_clear = self._live_clear_to_continue(kwargs)
        info.update(
            {
                "rollout_mismatch_prediction_untrusted": bool(prediction_untrusted),
                "rollout_mismatch_live_clear_to_continue": bool(live_clear),
                "rollout_mismatch_live_safe_steps": int(self._rollout_mismatch_live_safe_steps),
                "rollout_mismatch_no_progress_steps": int(self._rollout_mismatch_no_progress_steps),
                "rollout_mismatch_escape_steps_remaining": int(
                    self._rollout_mismatch_escape_steps_remaining
                ),
            }
        )
        prefix_safe = self._hard_executable_prefix_safe(info)
        info["rollout_mismatch_prefix_safe"] = bool(prefix_safe)
        if not (self.rollout_mismatch_mitigation_enabled and live_clear and prefix_safe):
            if not live_clear:
                self._rollout_mismatch_live_safe_steps = 0
                self._rollout_mismatch_no_progress_steps = 0
            return None

        progress = self._finite_float(kwargs.get("task_progress"))
        if progress is not None:
            best = self._rollout_mismatch_best_progress
            if best is None or progress > best + self.rollout_mismatch_progress_eps:
                self._rollout_mismatch_best_progress = progress
                self._rollout_mismatch_no_progress_steps = 0
            else:
                self._rollout_mismatch_no_progress_steps += 1
        else:
            self._rollout_mismatch_no_progress_steps += 1
        self._rollout_mismatch_live_safe_steps += 1

        reason: str | None = None
        if self._rollout_mismatch_escape_steps_remaining > 0:
            self._rollout_mismatch_escape_steps_remaining -= 1
            reason = "active_escape_window"
        elif (
            self._rollout_mismatch_live_safe_steps >= self.rollout_mismatch_escape_trigger_steps
            and self._rollout_mismatch_no_progress_steps >= self.rollout_mismatch_escape_trigger_steps
        ):
            self._rollout_mismatch_escape_steps_remaining = max(
                0,
                self.rollout_mismatch_escape_pass_through_steps - 1,
            )
            reason = "live_safe_no_progress_rollout_mismatch"

        info.update(
            {
                "rollout_mismatch_live_safe_steps": int(self._rollout_mismatch_live_safe_steps),
                "rollout_mismatch_no_progress_steps": int(self._rollout_mismatch_no_progress_steps),
                "rollout_mismatch_escape_steps_remaining": int(
                    self._rollout_mismatch_escape_steps_remaining
                ),
            }
        )
        if reason is None:
            return None

        self.unsafe_streak = 0
        self.brake.brake_streak = 0
        self.recovery.recovery_failure_streak = 0
        self.recovery.recovery_optimizer_cooldown_remaining = 0
        self.recovery.recovery_attempts_in_unsafe_streak = 0
        info.update(
            {
                "safety_mode": "pass_through",
                "mode": "pass_through",
                "deform_mode": None,
                "deformation_source": None,
                "optimized_fallback": None,
                "fallback_reason": None,
                "fallback_used": False,
                "rollout_mismatch_pass_through": True,
                "rollout_mismatch_escape_reason": reason,
                "horizon_unsafe_ignored_due_to_rollout_mismatch": True,
            }
        )
        try:
            self.brake._update_last_safe_execution(obs, chunk, info, **dict(kwargs))
        except Exception:  # noqa: BLE001
            pass
        self.last_info = info
        return chunk.reshape(original_shape), info

    def _mark_contact_rich_pause_info(
        self,
        info: dict[str, Any],
        *,
        pause_only: bool,
        fallback_reason: str,
    ) -> None:
        """Annotate an intervention as contact-rich pause-only."""
        info.update(
            {
                "contact_rich_state": self.contact_rich_state,
                "contact_rich_pause_active": bool(self._contact_rich_pause_active),
                "contact_rich_pause_only": bool(pause_only),
                "deformation_blocked_by_contact_rich_state": True,
                "deformation_deferred": True,
                "deformation_source": "horizon_brake",
                "fallback_reason": fallback_reason,
            }
        )
        if pause_only:
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deform_mode": "contact_rich_pause_only",
                    "fallback_used": True,
                }
            )

    @staticmethod
    def _rollout_context_key(action_chunk: Any) -> tuple[tuple[int, ...], str, bytes] | None:
        """Build a stable same-step key for a small ACT chunk."""
        try:
            arr = np.asarray(action_chunk, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return None
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arr = np.ascontiguousarray(arr)
        return tuple(int(x) for x in arr.shape), str(arr.dtype), arr.tobytes()

    def _reset_rollout_context(self) -> None:
        """Clear cached nominal rollout state at the start of every filter call."""
        self._rollout_context = {}

    def _store_nominal_rollout_context(
        self,
        obs: Any,
        chunk: np.ndarray,
        *,
        raw_q_seq: np.ndarray,
        q_seq: np.ndarray | None = None,
        safety_info: Mapping[str, Any] | None = None,
        rollout_mismatch_info: Mapping[str, Any] | None = None,
    ) -> None:
        """Expose this step's nominal ACT rollout to shared intervention helpers."""
        key = self._rollout_context_key(chunk)
        if key is None:
            self._rollout_context = {}
            return
        existing_step = self._rollout_context.get("step")
        if existing_step is None:
            self._rollout_context_step += 1
            existing_step = self._rollout_context_step
        nominal: dict[str, Any] = {
            "raw_q_seq": np.asarray(raw_q_seq, dtype=np.float32).copy(),
        }
        if q_seq is not None:
            nominal["q_seq"] = np.asarray(q_seq, dtype=np.float32).copy()
        if safety_info is not None:
            nominal["safety_info"] = dict(safety_info)
        if rollout_mismatch_info is not None:
            nominal["rollout_mismatch_info"] = dict(rollout_mismatch_info)
        self._rollout_context = {
            "step": int(existing_step),
            "obs_id": id(obs),
            "nominal_key": key,
            "nominal": nominal,
            "cache": {},
        }

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Callable adapter that accepts either (action, obs) or (obs, action)."""
        if len(args) < 2:
            action = kwargs.pop("action", None)
            obs = kwargs.pop("obs", kwargs.pop("observations", None))
            if action is None:
                raise TypeError("SafeChunkDeformFilter requires an action argument")
        else:
            first, second = args[0], args[1]
            # Infer argument order from the array-like action shape.
            first_arr = self.intervention_factory.maybe_array(first)
            second_arr = self.intervention_factory.maybe_array(second)
            if first_arr is not None and first_arr.ndim in (1, 2):
                action, obs = first, second
            elif second_arr is not None and second_arr.ndim in (1, 2):
                obs, action = first, second
            else:
                action, obs = first, second

        chunk, was_single = self.intervention_factory.as_chunk(action)
        if not self.enabled:
            self.last_info = {"safety_mode": "disabled", "mode": "disabled"}
            return chunk.reshape(np.asarray(action).shape)
        if was_single:
            safe_action = self.intervention_factory.filter_single_action(chunk[0], obs, **kwargs)
            self.last_info = {
                "safety_mode": "single_step_oscbf",
                "mode": "single_step_oscbf",
            }
            return np.asarray(safe_action).reshape(np.asarray(action).shape)

        safe_chunk, info = self.filter_chunk(obs, chunk, **kwargs)
        self.last_info = info
        return safe_chunk.reshape(np.asarray(action).shape)

    def filter_chunk(
        self,
        obs: Any,
        action_chunk: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Route one ACT chunk through pass-through, brake, deform, and recovery phases."""
        self._reset_rollout_context()
        resume_affordance_context = self._normalize_resume_affordance_context(kwargs)
        self.recovery.set_resume_affordance_context(resume_affordance_context)
        original_shape = np.asarray(action_chunk).shape
        chunk, was_single = self.intervention_factory.as_chunk(action_chunk)
        if not self.enabled:
            info = {"safety_mode": "disabled", "mode": "disabled"}
            self.last_info = info
            return chunk.reshape(original_shape), info
        if was_single:
            safe_action = self.intervention_factory.filter_single_action(chunk[0], obs, **kwargs)
            info = {
                "safety_mode": "single_step_oscbf",
                "mode": "single_step_oscbf",
            }
            self.last_info = info
            return np.asarray(safe_action).reshape(original_shape), info

        # The filter owns perception and routing: cache the nominal chunk, serve
        # any already-committed recovery first, then evaluate the fresh horizon.
        self.recovery.latest_nominal_chunk = np.asarray(chunk, dtype=np.float32).copy()
        self.recovery.latest_nominal_step += 1
        raw_q_seq: np.ndarray = self.deform.rollout_nominal_chunk(obs, chunk)
        self._store_nominal_rollout_context(obs, chunk, raw_q_seq=raw_q_seq)

        if (
            self.intervention_fsm.config.enabled
            and self.intervention_fsm.mode == InterventionMode.SAFE_STOP_LATCHED
        ):
            latch_q_seq, latch_rollout_info = self._apply_rollout_residual_correction(
                raw_q_seq,
                kwargs,
            )
            latch_safety_info = self.intervention_factory.evaluate_horizon_safety(
                obs,
                latch_q_seq,
            )
            latch_info = dict(latch_safety_info)
            latch_info.update(latch_rollout_info)
            latch_info.update(self._nominal_collision_attribution(obs, latch_q_seq, latch_safety_info))
            self.intervention_fsm.note_nominal_clear(
                bool(latch_safety_info.get("horizon_safe", False))
            )
            release, human_motion, nominal_change, release_reason = (
                self.intervention_fsm.latch_release_status(
                    human_state=self._human_signature(obs, kwargs),
                    nominal_signature=self._nominal_signature(chunk, latch_q_seq),
                    nominal_clear=bool(latch_safety_info.get("horizon_safe", False)),
                )
            )
            latch_info.update(
                {
                    "intervention_human_motion_since_failure": human_motion,
                    "intervention_nominal_change_since_failure": nominal_change,
                    "intervention_latch_release_reason": release_reason,
                }
            )
            if not release:
                self._update_intervention_info(latch_info)
                return self._perform_fsm_safe_stop(
                    obs,
                    chunk,
                    original_shape,
                    latch_safety_info,
                    latch_info,
                    reason="safe_stop_latched_no_meaningful_change",
                    **kwargs,
                )
            self.intervention_fsm.transition(
                InterventionMode.PAUSE_GUARD,
                f"safe_stop_latch_released:{release_reason}",
                UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
            )

        committed_result = self.recovery._serve_committed_chunk(obs, chunk, original_shape, **kwargs)
        if committed_result is not None:
            committed_chunk, committed_info = committed_result
            committed_chunk, committed_info = self._apply_pause_exit_smoothing(
                obs,
                committed_chunk,
                original_shape,
                committed_info,
            )
            if self.intervention_fsm.config.enabled:
                if bool(committed_info.get("committed_aborted_due_to_safety", False)):
                    self._latch_deformation_failure(
                        committed_info,
                        reason="committed_deformation_aborted",
                        obs=obs,
                        chunk=chunk,
                        q_seq=raw_q_seq,
                        kwargs=kwargs,
                    )
                elif bool(committed_info.get("committed_chunk_active", False)):
                    if self.intervention_fsm.mode != InterventionMode.DEFORM_COMMIT:
                        self.intervention_fsm.transition(
                            InterventionMode.DEFORM_COMMIT,
                            "serving_committed_chunk",
                        )
                    try:
                        served_arr, _ = self.intervention_factory._as_chunk(committed_chunk)
                        valid = self.controlled_action_indices[
                            self.controlled_action_indices < served_arr.shape[1]
                        ]
                        action_norm = float(np.linalg.norm(served_arr[0, valid])) if valid.size else None
                    except Exception:  # noqa: BLE001
                        action_norm = None
                    stalled = self.intervention_fsm.note_deform_commit_step(
                        action_norm=action_norm,
                    )
                    if stalled:
                        self._latch_deformation_failure(
                            committed_info,
                            reason="committed_deformation_stalled",
                            obs=obs,
                            chunk=chunk,
                            q_seq=raw_q_seq,
                            kwargs=kwargs,
                        )
                self._update_intervention_info(committed_info)
            self.last_info = committed_info
            return committed_chunk, committed_info
        pending_committed_replan_info = self.recovery._pop_pending_committed_replan_info()

        # Shared factory evaluates nominal horizon safety before routing interventions.
        q_seq, rollout_mismatch_info = self._apply_rollout_residual_correction(raw_q_seq, kwargs)
        safety_info = self.intervention_factory.evaluate_horizon_safety(obs, q_seq)
        attribution_info = self._nominal_collision_attribution(obs, q_seq, safety_info)
        self._store_nominal_rollout_context(
            obs,
            chunk,
            raw_q_seq=raw_q_seq,
            q_seq=q_seq,
            safety_info=safety_info,
            rollout_mismatch_info=rollout_mismatch_info,
        )
        info = dict(safety_info)
        info.update(attribution_info)
        info.update(rollout_mismatch_info)
        path_blocked, path_available, path_source = self._path_blockage_info(
            safety_info, kwargs
        )
        path_pause_sufficient, path_pause_available, path_pause_source = (
            self._path_block_pause_sufficiency_info(kwargs)
        )
        path_requires_bypass = bool(path_blocked and not path_pause_sufficient)
        goal_blocked, goal_distance, goal_available, goal_source = self._goal_blockage_info(
            kwargs
        )
        if goal_blocked:
            self._goal_block_hold_active = True
            self._goal_block_release_wait_count = 0
        info.update(
            {
                "nominal_path_blocked": bool(path_blocked),
                "nominal_path_blockage_check_available": bool(path_available),
                "nominal_path_blockage_source": path_source,
                "path_block_pause_sufficient": bool(path_pause_sufficient),
                "path_block_pause_sufficiency_available": bool(path_pause_available),
                "path_block_pause_sufficiency_source": path_pause_source,
                "path_block_requires_bypass": bool(path_requires_bypass),
                "nominal_goal_blocked": bool(goal_blocked),
                "nominal_goal_distance": goal_distance,
                "nominal_goal_blockage_check_available": bool(goal_available),
                "nominal_goal_blockage_source": goal_source,
                "nominal_blockage_route": (
                    "goal_blocked_hold"
                    if goal_blocked
                    else "path_blockage_candidate"
                    if path_blocked
                    else "no_blockage"
                ),
            }
        )
        info.update(self.intervention_factory.safechunk_replan_info())
        info.update(self.recovery.current_resume_affordance_info())
        if pending_committed_replan_info:
            info.update(pending_committed_replan_info)
        self.brake._temporary_update_progress(kwargs.get("task_progress"))
        info.update(
            self._update_contact_rich_pause_state(
                obs,
                chunk,
                safety_info,
                info,
                **kwargs,
            )
        )

        if safety_info["horizon_safe"]:
            return self._perform_pass_through(
                obs,
                chunk,
                original_shape,
                info,
                q_seq=q_seq,
                safety_info=safety_info,
                **kwargs,
            )

        if self._contact_rich_gripper_only_clear_to_continue(info, kwargs):
            return self._perform_contact_rich_clear_pass_through(
                chunk,
                original_shape,
                info,
                reason="gripper_only_clear_before_brake",
            )

        attributed_nominal_collision = info.get("nominal_collision_source") in {
            "current_human_geometry",
            "predicted_human_motion",
        }
        info["nominal_attribution_bypass_soft_pass_through"] = bool(
            attributed_nominal_collision
        )
        if not attributed_nominal_collision:
            soft_prefix_pass_through = self._maybe_soft_full_horizon_pass_through(
                obs,
                chunk,
                original_shape,
                info,
                safety_info,
                kwargs,
            )
            if soft_prefix_pass_through is not None:
                return soft_prefix_pass_through

            mismatch_pass_through = self._maybe_rollout_mismatch_pass_through(
                obs,
                chunk,
                original_shape,
                info,
                kwargs,
            )
            if mismatch_pass_through is not None:
                return mismatch_pass_through

        if self.intervention_fsm.config.enabled:
            self.intervention_fsm.note_nominal_clear(False)

        if self.recovery.post_recovery_act_window_active:
            info.update(self.recovery._post_recovery_act_window_info(interrupted=True))

        # Unsafe horizons enter the intervention ladder: brake first, then only
        # attempt deformation/recovery when brake policy says a stronger action
        # is warranted.
        brake_decision = self._perform_brake(
            obs,
            chunk,
            original_shape,
            safety_info,
            info,
            **kwargs,
        )
        if brake_decision["result"] is not None:
            if self._contact_rich_pause_is_active():
                self._mark_contact_rich_pause_info(
                    info,
                    pause_only=False,
                    fallback_reason="contact_rich_brake_wait",
                )
            return brake_decision["result"]

        goal_release_wait = self._goal_block_release_wait_needed(
            info,
            goal_blocked=bool(info.get("nominal_goal_blocked", False)),
            kwargs=kwargs,
        )
        if goal_release_wait:
            if self.intervention_fsm.config.enabled:
                self.intervention_fsm.transition(
                    InterventionMode.PAUSE_GUARD,
                    "goal_block_release_wait",
                    UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
                )
                self.intervention_fsm.note_pause_step()
            self._arm_pause_exit_smoothing(
                brake_decision["braked_chunk"],
                info,
                reason="goal_block_release_wait",
            )
            return self._perform_fsm_safe_stop(
                obs,
                chunk,
                original_shape,
                safety_info,
                info,
                reason="goal_block_release_wait",
                **kwargs,
            )

        conflict_aware_result = self._maybe_handle_conflict_aware_intervention(
            obs,
            chunk,
            original_shape,
            safety_info,
            info,
            brake_decision=brake_decision,
            nominal_q_seq=q_seq,
            **kwargs,
        )
        if conflict_aware_result is not None:
            return conflict_aware_result

        # A blocked task goal is a task conflict, not a path-bypass problem:
        # deformation would move away from the goal without resolving it.
        if bool(info.get("nominal_goal_blocked", False)):
            info.update(
                {
                    "nominal_blockage_route": "goal_blocked_hold",
                    "nominal_goal_blocked_deform_suppressed": True,
                    "nominal_goal_blocked_deform_suppression_reason": "goal_blockage_cannot_be_solved_by_deformation",
                }
            )
            if self.intervention_fsm.config.enabled:
                self.intervention_fsm.transition(
                    InterventionMode.PAUSE_GUARD,
                    "goal_blocked_hold",
                    UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
                )
                self.intervention_fsm.note_pause_step()
            self._arm_pause_exit_smoothing(
                brake_decision["braked_chunk"],
                info,
                reason="goal_blocked_hold",
            )
            return self._perform_fsm_safe_stop(
                obs,
                chunk,
                original_shape,
                safety_info,
                info,
                reason="goal_blocked_hold",
                **kwargs,
            )

        if bool(info.get("nominal_attribution_pause_recommended", False)):
            trigger_reason = brake_decision.get("deform_trigger_reason")
            attribution_pause_reason = None
            min_unsafe_steps = max(
                1,
                int(self.brake.temporary_min_unsafe_steps_before_deform),
            )
            max_brake_steps = max(
                1,
                int(self.brake.temporary_max_brake_steps_before_deform),
            )
            if self.unsafe_streak >= min_unsafe_steps:
                attribution_pause_reason = "persistent_unsafe"
            elif self.brake.brake_streak >= max_brake_steps:
                attribution_pause_reason = "brake_timeout"

            current_human_unsafe = bool(
                info.get("nominal_current_human_unsafe", False)
            )
            current_min_clearance = info.get(
                "nominal_current_human_min_clearance",
                safety_info.get("min_clearance") if safety_info is not None else None,
            )
            try:
                current_min_clearance_f = float(current_min_clearance)
            except Exception:  # noqa: BLE001
                current_min_clearance_f = float("inf")
            if not np.isfinite(current_min_clearance_f):
                current_min_clearance_f = float("inf")
            pause_deform_threshold = float(
                self.intervention_fsm.config.pause_deform_min_clearance_threshold
            )
            close_current_human = bool(
                current_min_clearance_f < pause_deform_threshold
            )
            force_deform_now = bool(
                self.intervention_fsm.config.pause_deform_on_current_human_unsafe
                and (current_human_unsafe or close_current_human)
            )
            force_deform_reason = None
            pre_force_attribution_pause_reason = attribution_pause_reason
            if force_deform_now:
                if current_human_unsafe:
                    force_deform_reason = "current_human_unsafe"
                else:
                    force_deform_reason = "current_human_close"
                attribution_pause_reason = force_deform_reason

            stop_result = None
            goal_blocked = False
            goal_distance = None
            goal_available = False
            goal_source = "not_checked"
            human_motion_speed = self._finite_float(info.get("human_motion_prediction_speed"))
            if human_motion_speed is None and safety_info is not None:
                human_motion_speed = self._finite_float(
                    safety_info.get("human_motion_prediction_speed")
                )
            if human_motion_speed is None:
                human_motion_speed = self._finite_float(
                    kwargs.get("human_motion_prediction_speed")
                )
            static_human_threshold = float(
                self.intervention_fsm.config.pause_deform_static_human_speed_threshold
            )
            human_motion_static = bool(
                human_motion_speed is not None
                and human_motion_speed <= static_human_threshold
            )
            forced_deform_suppressed = False
            forced_deform_suppression_reason = None
            goal_allows_suppression = False
            if (
                force_deform_now
                and self.intervention_fsm.config.pause_deform_suppress_when_stop_sufficient
            ):
                stop_result = self._evaluate_stop_counterfactual(brake_decision)
                (
                    goal_blocked,
                    goal_distance,
                    goal_available,
                    goal_source,
                ) = self._goal_blockage_info(kwargs)
                goal_allows_suppression = bool(
                    not goal_blocked
                    and (
                        goal_available
                        or not self.intervention_fsm.config.pause_deform_suppress_requires_goal_check
                    )
                )
                stop_pause_task_sufficient = self._stop_pause_is_task_sufficient(
                    stop_safe=bool(stop_result.safe),
                    path_blocked=bool(info.get("nominal_path_blocked", False)),
                    path_block_pause_sufficient=bool(
                        info.get("path_block_pause_sufficient", False)
                    ),
                )
                if stop_pause_task_sufficient and human_motion_static and goal_allows_suppression:
                    forced_deform_suppressed = True
                    forced_deform_suppression_reason = (
                        "stop_safe_static_blocker_pause_task_sufficient"
                    )
                    force_deform_now = False
                    force_deform_reason = None
                    attribution_pause_reason = pre_force_attribution_pause_reason
                    if trigger_reason in (None, "normal") and attribution_pause_reason is None:
                        trigger_reason = "stop_sufficient_static_human_wait"
                elif not stop_result.safe:
                    forced_deform_suppression_reason = "stop_counterfactual_unsafe"
                elif not human_motion_static:
                    forced_deform_suppression_reason = "human_motion_not_static"
                elif goal_blocked:
                    forced_deform_suppression_reason = "goal_blocked"
                elif bool(info.get("path_block_requires_bypass", False)):
                    forced_deform_suppression_reason = "path_block_requires_bypass"
                elif not goal_available:
                    forced_deform_suppression_reason = "goal_check_unavailable"
                else:
                    forced_deform_suppression_reason = "suppression_condition_failed"

            temporary_budget_exhausted = bool(
                trigger_reason
                in {
                    "persistent_unsafe",
                    "brake_timeout",
                    "progress_deadlock",
                }
                or attribution_pause_reason is not None
            )
            if trigger_reason == "normal" and attribution_pause_reason is not None:
                trigger_reason = attribution_pause_reason
            info.update(
                {
                    "nominal_attribution_pause_budget_open": not temporary_budget_exhausted,
                    "nominal_attribution_pause_budget_exhausted": temporary_budget_exhausted,
                    "nominal_attribution_pause_trigger_reason": trigger_reason,
                    "nominal_attribution_deform_after_pause_budget": temporary_budget_exhausted,
                    "nominal_attribution_pause_budget_brake_streak": int(
                        self.brake.brake_streak
                    ),
                    "nominal_attribution_pause_budget_unsafe_streak": int(
                        self.unsafe_streak
                    ),
                    "nominal_attribution_pause_budget_max_brake_steps": int(
                        max_brake_steps
                    ),
                    "nominal_attribution_pause_budget_min_unsafe_steps": int(
                        min_unsafe_steps
                    ),
                    "nominal_attribution_pause_forced_deform": force_deform_now,
                    "nominal_attribution_pause_force_deform_reason": force_deform_reason,
                    "nominal_attribution_pause_forced_deform_suppressed": forced_deform_suppressed,
                    "nominal_attribution_pause_force_deform_suppression_reason": forced_deform_suppression_reason,
                    "nominal_attribution_stop_counterfactual_safe": (
                        None if stop_result is None else bool(stop_result.safe)
                    ),
                    "nominal_attribution_stop_counterfactual_min_clearance": (
                        None if stop_result is None else float(stop_result.min_distance)
                    ),
                    "nominal_attribution_stop_counterfactual_first_violation": (
                        None if stop_result is None else stop_result.violation_step
                    ),
                    "nominal_attribution_goal_blocked": bool(goal_blocked),
                    "nominal_attribution_goal_distance": goal_distance,
                    "nominal_attribution_goal_check_available": bool(goal_available),
                    "nominal_attribution_goal_check_source": goal_source,
                    "nominal_attribution_goal_allows_suppression": bool(goal_allows_suppression),
                    "nominal_attribution_human_motion_speed": human_motion_speed,
                    "nominal_attribution_human_motion_static": bool(human_motion_static),
                    "nominal_attribution_human_motion_static_threshold": static_human_threshold,
                    "nominal_attribution_pause_current_human_unsafe": current_human_unsafe,
                    "nominal_attribution_pause_current_min_clearance": current_min_clearance_f,
                    "nominal_attribution_pause_min_clearance_threshold": pause_deform_threshold,
                }
            )
            slowdown_result = None
            if not temporary_budget_exhausted:
                slowdown_result = self._maybe_perform_policy_collision_slowdown(
                    obs,
                    chunk,
                    original_shape,
                    safety_info,
                    info,
                    deform_trigger_reason=str(trigger_reason),
                    **kwargs,
                )
            if slowdown_result is not None:
                return slowdown_result
            if not temporary_budget_exhausted:
                if self.intervention_fsm.config.enabled:
                    self.intervention_fsm.transition(
                        InterventionMode.PAUSE_GUARD,
                        "policy_induced_collision_pause_after_slowdown",
                        UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
                    )
                    self.intervention_fsm.note_pause_step()
                    self._update_intervention_info(
                        info,
                        extra={
                            "policy_collision_pause_after_slowdown": True,
                            "policy_collision_slowdown_to_pause": True,
                        },
                    )
                self._arm_pause_exit_smoothing(
                    brake_decision["braked_chunk"],
                    info,
                    reason=str(trigger_reason),
                )
                return self._perform_attributed_nominal_pause(
                    obs,
                    chunk,
                    original_shape,
                    info,
                    braked_chunk=brake_decision["braked_chunk"],
                    deform_trigger_reason=str(trigger_reason),
                    **kwargs,
                )
            if self.intervention_fsm.config.enabled:
                self.intervention_fsm.transition(
                    InterventionMode.PAUSE_GUARD,
                    "pause_budget_expired_evaluate_deform",
                    UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
                )
            self._arm_pause_exit_smoothing(
                brake_decision["braked_chunk"],
                info,
                reason=str(trigger_reason),
            )

        if self._contact_rich_deform_block_is_active():
            if self._contact_rich_gripper_only_clear_to_continue(info, kwargs):
                return self._perform_contact_rich_clear_pass_through(
                    chunk,
                    original_shape,
                    info,
                    reason="gripper_only_clear_after_brake_check",
                )
            return self._perform_contact_rich_brake_only(
                obs,
                chunk,
                original_shape,
                info,
                braked_chunk=brake_decision["braked_chunk"],
                deform_trigger_reason=info.get(
                    "nominal_attribution_pause_trigger_reason",
                    brake_decision["deform_trigger_reason"],
                ),
                **kwargs,
            )

        deform_decision = self._perform_deform(
            obs,
            chunk,
            original_shape,
            safety_info,
            info,
            braked_chunk=brake_decision["braked_chunk"],
            brake_info=info,
            nominal_q_seq=q_seq,
            deform_trigger_reason=info.get(
                "nominal_attribution_pause_trigger_reason",
                brake_decision["deform_trigger_reason"],
            ),
            **kwargs,
        )
        if deform_decision["result"] is not None:
            return deform_decision["result"]

        if (
            self.intervention_fsm.config.enabled
            and bool(info.get("nominal_attribution_pause_recommended", False))
            and bool(info.get("nominal_attribution_pause_budget_exhausted", False))
        ):
            deformation_evaluation = self._evaluate_deformation_admissibility(
                obs,
                chunk,
                deform_decision["safe_chunk"],
                q_seq,
                deform_decision["info"],
            )
            deform_decision["info"].update(
                {
                    "nominal_attribution_deform_candidate_evaluated": True,
                    "nominal_attribution_deform_candidate_admissible": bool(
                        deformation_evaluation.admissible
                    ),
                }
            )
            may_commit = self.intervention_fsm.can_enter_deform_commit(
                pause_budget_expired=True,
                evaluation=deformation_evaluation,
            )
            unsafe_reason = self._classify_unsafe_reason(
                deform_decision["info"],
                deformation_evaluation,
            )
            if not may_commit:
                self.intervention_fsm.transition(
                    InterventionMode.PAUSE_GUARD,
                    f"deform_not_admissible:{deformation_evaluation.failure_reason or 'waiting_for_stable_candidate'}",
                    unsafe_reason,
                )
                self._update_intervention_info(
                    deform_decision["info"],
                    evaluation=deformation_evaluation,
                    unsafe_reason=unsafe_reason,
                    extra={
                        "intervention_deform_commit_blocked": True,
                        "intervention_deform_commit_block_reason": (
                            deformation_evaluation.failure_reason
                            or "deform_valid_counter_not_satisfied"
                        ),
                    },
                )
                return self._perform_fsm_safe_stop(
                    obs,
                    chunk,
                    original_shape,
                    safety_info,
                    deform_decision["info"],
                    reason="pause_guard_deform_not_admissible",
                    **kwargs,
                )
            self.intervention_fsm.transition(
                InterventionMode.DEFORM_COMMIT,
                "deformation_admissible_after_pause_guard",
                unsafe_reason,
            )
            self._update_intervention_info(
                deform_decision["info"],
                evaluation=deformation_evaluation,
                unsafe_reason=unsafe_reason,
                extra={"intervention_deform_commit_blocked": False},
            )
        elif self.intervention_fsm.config.enabled and not bool(
            info.get("nominal_attribution_pause_recommended", False)
        ):
            if bool(deform_decision["info"].get("optimized_accepted", False)):
                stop_result = self._evaluate_stop_counterfactual(brake_decision)
                human_motion_speed = self._finite_float(
                    deform_decision["info"].get("human_motion_prediction_speed")
                )
                if human_motion_speed is None:
                    human_motion_speed = self._finite_float(
                        safety_info.get("human_motion_prediction_speed")
                    )
                human_motion_displacement = self._finite_float(
                    deform_decision["info"].get("human_motion_prediction_max_displacement")
                )
                if human_motion_displacement is None:
                    human_motion_displacement = self._finite_float(
                        safety_info.get("human_motion_prediction_max_displacement")
                    )
                static_speed_threshold = float(
                    self.intervention_fsm.config.pause_deform_static_human_speed_threshold
                )
                static_displacement_threshold = float(
                    self.intervention_fsm.config.early_deform_static_human_displacement_threshold
                )
                human_motion_static = bool(
                    (
                        human_motion_speed is not None
                        and human_motion_speed <= static_speed_threshold
                    )
                    or (
                        human_motion_displacement is not None
                        and human_motion_displacement <= static_displacement_threshold
                    )
                )
                (
                    goal_blocked,
                    goal_distance,
                    goal_available,
                    goal_source,
                ) = self._goal_blockage_info(kwargs)
                goal_allows_suppression = bool(
                    not goal_blocked
                    and (
                        goal_available
                        or not self.intervention_fsm.config.pause_deform_suppress_requires_goal_check
                    )
                )
                early_deform_suppressed = bool(
                    self.intervention_fsm.config.early_deform_suppress_when_stop_sufficient
                    and self._stop_pause_is_task_sufficient(
                        stop_safe=bool(stop_result.safe),
                        path_blocked=bool(
                            deform_decision["info"].get("nominal_path_blocked", False)
                        ),
                        path_block_pause_sufficient=bool(
                            deform_decision["info"].get("path_block_pause_sufficient", False)
                        ),
                    )
                    and human_motion_static
                    and goal_allows_suppression
                )
                if early_deform_suppressed:
                    deform_decision["info"].update(
                        {
                            "nominal_attribution_early_deform_suppressed": True,
                            "nominal_attribution_early_deform_suppression_reason": (
                                "stop_safe_static_blocker_pause_task_sufficient"
                            ),
                            "nominal_attribution_early_deform_candidate_accepted_before_suppression": True,
                            "optimized_accepted": False,
                            "optimized_fallback": "brake",
                            "optimized_reject_reason": "early_deform_suppressed_stop_sufficient_static_human",
                            "fallback_reason": "early_deform_suppressed_stop_sufficient_static_human",
                            "recover_accepted": False,
                            "return_accepted": False,
                            "is_recoverable": False,
                            "recovery_rejected": True,
                            "deformation_deferred": True,
                            "nominal_attribution_early_deform_stop_counterfactual_safe": bool(stop_result.safe),
                            "nominal_attribution_early_deform_stop_counterfactual_min_clearance": float(stop_result.min_distance),
                            "nominal_attribution_early_deform_human_motion_speed": human_motion_speed,
                            "nominal_attribution_early_deform_human_motion_displacement": human_motion_displacement,
                            "nominal_attribution_early_deform_human_motion_static": True,
                            "nominal_attribution_early_deform_goal_blocked": bool(goal_blocked),
                            "nominal_attribution_early_deform_goal_distance": goal_distance,
                            "nominal_attribution_early_deform_goal_check_available": bool(goal_available),
                            "nominal_attribution_early_deform_goal_check_source": goal_source,
                        }
                    )
                    self.intervention_fsm.transition(
                        InterventionMode.PAUSE_GUARD,
                        "early_deform_suppressed_stop_sufficient_static_human",
                        UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
                    )
                    return self._perform_fsm_safe_stop(
                        obs,
                        chunk,
                        original_shape,
                        safety_info,
                        deform_decision["info"],
                        reason="early_deform_suppressed_stop_sufficient_static_human",
                        **kwargs,
                    )
                deform_decision["info"].update(
                    {
                        "nominal_attribution_early_deform_suppressed": False,
                        "nominal_attribution_early_deform_suppression_reason": (
                            "stop_counterfactual_unsafe"
                            if not stop_result.safe
                            else "human_motion_not_static"
                            if not human_motion_static
                            else "goal_blocked"
                            if goal_blocked
                            else "path_block_requires_bypass"
                            if bool(
                                deform_decision["info"].get("path_block_requires_bypass", False)
                            )
                            else "goal_check_unavailable"
                            if not goal_available
                            else "suppression_condition_failed"
                        ),
                        "nominal_attribution_early_deform_stop_counterfactual_safe": bool(stop_result.safe),
                        "nominal_attribution_early_deform_stop_counterfactual_min_clearance": float(stop_result.min_distance),
                        "nominal_attribution_early_deform_human_motion_speed": human_motion_speed,
                        "nominal_attribution_early_deform_human_motion_displacement": human_motion_displacement,
                        "nominal_attribution_early_deform_human_motion_static": bool(human_motion_static),
                        "nominal_attribution_early_deform_goal_blocked": bool(goal_blocked),
                        "nominal_attribution_early_deform_goal_distance": goal_distance,
                        "nominal_attribution_early_deform_goal_check_available": bool(goal_available),
                        "nominal_attribution_early_deform_goal_check_source": goal_source,
                    }
                )
            self.intervention_fsm.transition(
                InterventionMode.DEFORM_COMMIT,
                "transient_obstruction_deform_first",
                UnsafeReason.TRANSIENT_PATH_OBSTRUCTION,
            )
            self._update_intervention_info(deform_decision["info"])

        recover_commit_result = self._perform_recover_commit(
            obs,
            chunk,
            original_shape,
            safe_chunk=deform_decision["safe_chunk"],
            braked_chunk=brake_decision["braked_chunk"],
            info=deform_decision["info"],
            **kwargs,
        )
        if recover_commit_result is not None:
            return recover_commit_result

        return self._perform_deform_fallback_or_finalize(
            obs,
            chunk,
            original_shape,
            safety_info=safety_info,
            safe_chunk=deform_decision["safe_chunk"],
            braked_chunk=brake_decision["braked_chunk"],
            info=deform_decision["info"],
            **kwargs,
        )

    def _maybe_handle_conflict_aware_intervention(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        safety_info: Mapping[str, Any],
        info: dict[str, Any],
        *,
        brake_decision: Mapping[str, Any],
        nominal_q_seq: np.ndarray | None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        if not self._conflict_aware_enabled():
            return None

        stop_result = self._evaluate_stop_counterfactual(brake_decision)
        pause_budget_limit = int(self.intervention_fsm.config.pause_budget_steps)
        if pause_budget_limit <= 0:
            pause_budget_limit = max(
                1,
                int(getattr(self.brake, "temporary_max_brake_steps_before_deform", 1)),
            )
        pause_budget_used = max(int(self.unsafe_streak), int(self.brake.brake_streak))
        pause_budget_expired = bool(pause_budget_used >= pause_budget_limit)

        if not stop_result.safe:
            self._update_conflict_aware_info(
                info,
                stop_result=stop_result,
                selected_mode=InterventionMode.DEFORM_COMMIT.value,
                decision_reason="STOP_INSUFFICIENT_HUMAN_INTRUSION",
                nominal_safety=safety_info,
                pause_budget_used=pause_budget_used,
            )
            info["nominal_attribution_pause_budget_exhausted"] = True
            info["nominal_attribution_pause_budget_open"] = False
            info["nominal_attribution_pause_trigger_reason"] = "stop_counterfactual_unsafe"
            return None

        deform_decision = self._perform_deform(
            obs,
            chunk,
            original_shape,
            dict(safety_info),
            info,
            braked_chunk=brake_decision["braked_chunk"],
            brake_info=info,
            nominal_q_seq=nominal_q_seq,
            deform_trigger_reason="conflict_aware_stop_sufficient",
            **kwargs,
        )
        if deform_decision["result"] is not None:
            result_chunk, result_info = deform_decision["result"]
            self._update_conflict_aware_info(
                result_info,
                stop_result=stop_result,
                selected_mode=str(result_info.get("mode", result_info.get("safety_mode", "safe_stop"))),
                decision_reason="SAFE_STOP_LATCHED",
                nominal_safety=safety_info,
                pause_budget_used=pause_budget_used,
            )
            return result_chunk, result_info

        base_eval = self._evaluate_deformation_admissibility(
            obs,
            chunk,
            deform_decision["safe_chunk"],
            nominal_q_seq,
            deform_decision["info"],
        )
        admissibility = self._evaluate_conflict_aware_deformation_admissibility(
            chunk,
            deform_decision["safe_chunk"],
            base_eval,
            deform_decision["info"],
            kwargs,
        )
        deform_decision["info"].update(
            {
                "nominal_attribution_deform_candidate_evaluated": True,
                "nominal_attribution_deform_candidate_admissible": bool(
                    admissibility.bypassable
                ),
            }
        )

        if deform_decision["safe_chunk"] is not None and (
            admissibility.bypassable
            or not self.intervention_fsm.config.deformation_admissibility_enabled
        ):
            self.intervention_fsm.can_enter_deform_commit(
                pause_budget_expired=True,
                evaluation=base_eval,
            )
            unsafe_reason = self._classify_unsafe_reason(deform_decision["info"], base_eval)
            self.intervention_fsm.transition(
                InterventionMode.DEFORM_COMMIT,
                "conflict_aware_path_bypass_admissible",
                unsafe_reason,
            )
            self._update_conflict_aware_info(
                deform_decision["info"],
                stop_result=stop_result,
                admissibility=admissibility,
                selected_mode=InterventionMode.DEFORM_COMMIT.value,
                decision_reason=admissibility.reason,
                nominal_safety=safety_info,
                pause_budget_used=pause_budget_used,
            )
            self._update_intervention_info(
                deform_decision["info"],
                evaluation=base_eval,
                unsafe_reason=unsafe_reason,
                extra={"intervention_deform_commit_blocked": False},
            )
            recover_commit_result = self._perform_recover_commit(
                obs,
                chunk,
                original_shape,
                safe_chunk=deform_decision["safe_chunk"],
                braked_chunk=brake_decision["braked_chunk"],
                info=deform_decision["info"],
                **kwargs,
            )
            if recover_commit_result is not None:
                return recover_commit_result
            return self._perform_deform_fallback_or_finalize(
                obs,
                chunk,
                original_shape,
                safety_info=safety_info,
                safe_chunk=deform_decision["safe_chunk"],
                braked_chunk=brake_decision["braked_chunk"],
                info=deform_decision["info"],
                **kwargs,
            )

        selected_mode = (
            InterventionMode.SAFE_STOP_LATCHED.value
            if pause_budget_expired
            else InterventionMode.PAUSE_GUARD.value
        )
        decision_reason = (
            "PAUSE_BUDGET_EXHAUSTED"
            if pause_budget_expired
            else "STOP_SUFFICIENT_TEMPORARY_BLOCKAGE"
        )
        if admissibility.reason != "PATH_BYPASS_ADMISSIBLE":
            decision_reason = admissibility.reason if pause_budget_expired else decision_reason
        self._update_conflict_aware_info(
            deform_decision["info"],
            stop_result=stop_result,
            admissibility=admissibility,
            selected_mode=selected_mode,
            decision_reason=decision_reason,
            nominal_safety=safety_info,
            pause_budget_used=pause_budget_used,
        )
        self.intervention_fsm.transition(
            InterventionMode.PAUSE_GUARD,
            f"conflict_aware_deform_rejected:{admissibility.reason}",
            UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
        )
        self._update_intervention_info(
            deform_decision["info"],
            evaluation=base_eval,
            unsafe_reason=UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
            extra={
                "intervention_deform_commit_blocked": True,
                "intervention_deform_commit_block_reason": admissibility.reason,
            },
        )
        if pause_budget_expired:
            self.intervention_fsm.transition(
                InterventionMode.SAFE_STOP_LATCHED,
                f"conflict_aware_pause_budget_exhausted:{admissibility.reason}",
                UnsafeReason.DEFORMATION_INFEASIBLE,
            )
            return self._perform_fsm_safe_stop(
                obs,
                chunk,
                original_shape,
                safety_info,
                deform_decision["info"],
                reason="conflict_aware_deformation_inadmissible",
                **kwargs,
            )

        self.intervention_fsm.note_pause_step()
        deform_decision["info"].update(
            {
                "nominal_attribution_pause_applied": True,
                "nominal_attribution_pause_budget_open": True,
                "nominal_attribution_pause_budget_exhausted": False,
                "nominal_attribution_pause_trigger_reason": "stop_sufficient_wait",
                "nominal_attribution_deform_after_pause_budget": False,
            }
        )
        self._arm_pause_exit_smoothing(
            brake_decision["braked_chunk"],
            deform_decision["info"],
            reason="conflict_aware_stop_sufficient",
        )
        return self._perform_attributed_nominal_pause(
            obs,
            chunk,
            original_shape,
            deform_decision["info"],
            braked_chunk=brake_decision["braked_chunk"],
            deform_trigger_reason="conflict_aware_stop_sufficient",
            **kwargs,
        )

    def _perform_pass_through(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        info: dict[str, Any],
        *,
        q_seq: np.ndarray,
        safety_info: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Handle nominal-safe chunks, including temporary-wait recovery exits."""
        if (
            self._goal_block_hold_active
            and not bool(info.get("nominal_goal_blocked", False))
        ):
            info.update(
                {
                    "goal_block_release_wait_active": False,
                    "goal_block_release_wait_completed": True,
                    "goal_block_release_wait_count": int(
                        self._goal_block_release_wait_count
                    ),
                }
            )
            self._goal_block_hold_active = False
            self._goal_block_release_wait_count = 0
        self._reset_rollout_mismatch_state()
        self.recovery.committed_recover_steps_since_act = 0
        self.recovery.committed_suffix_replans_in_current_recovery = 0
        waited_unsafe_streak: int = int(self.unsafe_streak)
        waited_brake_streak: int = int(self.brake.brake_streak)
        nominal_became_safe = bool(waited_unsafe_streak > 0 or waited_brake_streak > 0)
        resume_after_wait = bool(
            nominal_became_safe
            and waited_brake_streak > 0
            and not getattr(self.brake, "exit_on_nominal_safe", True)
        )
        if resume_after_wait:
            info.update(
                self.brake._temporary_streak_info(
                    nominal_became_safe=nominal_became_safe,
                    resume_after_wait=resume_after_wait,
                )
            )
            recovery_result = self.recovery.Perform(
                obs,
                chunk,
                original_shape,
                info,
                q_seq=q_seq,
                safety_info=safety_info,
                waited_unsafe_streak=waited_unsafe_streak,
                waited_brake_streak=waited_brake_streak,
                mode="temporary_wait",
                **kwargs,
            )
            if recovery_result is not None:
                return recovery_result
        if self.brake.temporary_reset_on_nominal_safe:
            self.unsafe_streak = 0
            self.brake.brake_streak = 0
            self.intervention_fsm.slowdown_counter = 0
            self._stationary_human_local_escape_counter = 0
            self.recovery.recovery_failure_streak = 0
            self.recovery.recovery_optimizer_cooldown_remaining = 0
            self.recovery.recovery_attempts_in_unsafe_streak = 0
        if self.recovery.clear_failed_recovery_on_nominal_safe:
            self.recovery.failed_recovery_targets = []
            self.recovery.failed_recovery_paths = []
            self.recovery.recovery_target_failure_counts = {}
            self.recovery._unsafe_recovery_cooldowns = {}
            self.recovery.recovery_path_failure_streak = 0
        self.recovery.recover_step_since_deform = 0
        self.brake._deadlock_count = 0
        if self.recovery.post_recovery_act_window_active:
            remaining = int(self.recovery.post_recovery_act_steps_remaining)
            info.update({"safety_mode": "pass_through", "mode": "pass_through"})
            info.update(
                self.brake._temporary_streak_info(
                    nominal_became_safe=nominal_became_safe,
                    resume_after_wait=resume_after_wait,
                )
            )
            info.update(
                {
                    "post_recovery_act_window_active": True,
                    "post_recovery_act_steps_remaining": remaining,
                    "post_recovery_act_window_interrupted": False,
                }
            )
            self.recovery.post_recovery_act_steps_remaining = max(0, remaining - 1)
            if self.recovery.post_recovery_act_steps_remaining <= 0:
                self.recovery.post_recovery_act_window_active = False
            self.brake._update_last_safe_execution(obs, chunk, info, **kwargs)
            self.last_info = info
            return chunk.reshape(original_shape), info
        info.update({"safety_mode": "pass_through", "mode": "pass_through"})
        info.update(
            self.brake._temporary_streak_info(
                nominal_became_safe=nominal_became_safe,
                resume_after_wait=resume_after_wait,
            )
        )
        if (
            self.brake.safechunk_active_safety_enabled
            and self.brake.check_hold_horizon_safety
            and not getattr(self.brake, "exit_on_nominal_safe", True)
        ):
            info["active_safety_nominal_gate"] = True
            return self.brake._hold_return_or_emergency_deform(
                obs,
                nominal_chunk=chunk,
                braked_chunk=chunk,
                info=info,
                original_shape=original_shape,
                **kwargs,
            )
        output_chunk = np.asarray(chunk, dtype=np.float32).reshape(original_shape)
        if (
            nominal_became_safe
            and self._pause_exit_smoothing_remaining > 0
            and self._pause_exit_smoothing_anchor_action is not None
        ):
            info["pause_exit_smoothing_context"] = "nominal_pass_through_after_pause"
            output_chunk, info = self._apply_pause_exit_smoothing(
                obs,
                chunk,
                original_shape,
                info,
            )
        else:
            info.setdefault("pause_exit_smoothing_applied", False)
        self._reset_pause_exit_smoothing_rearm()
        self.brake._update_last_safe_execution(obs, output_chunk, info, **kwargs)
        self.last_info = info
        return output_chunk, info

    def _perform_brake(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        safety_info: Mapping[str, Any],
        info: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run horizon brake and decide whether the filter should stop there."""
        recovery_attempt_reset_info: dict[str, Any] = {}
        if self.unsafe_streak == 0:
            self.recovery.blocked_nominal_chunk = np.asarray(chunk, dtype=np.float32).copy()
            self.recovery.blocked_nominal_step = int(self.recovery.latest_nominal_step)
        self.unsafe_streak += 1
        # Brake is computed as a last-resort fallback; unsafe chunks should first
        # try the deform/recovery route when it is enabled.
        braked_chunk, brake_info = self.brake.horizon_brake(obs, chunk, safety_info)
        info.update(brake_info)
        if brake_info["deadlock"]:
            self.brake._deadlock_count += 1
        else:
            self.brake._deadlock_count = 0
        info["deadlock_count"] = int(self.brake._deadlock_count)

        safe_prefix_len = int(brake_info.get("safe_prefix_len") or 0)
        if (
            safe_prefix_len > 0
            and not bool(brake_info.get("brake_hold_current", False))
            and not self.deform.deformation_enabled
        ):
            info.update(
                {
                    "safety_mode": "safe_prefix",
                    "mode": "safe_prefix",
                    "deform_mode": "safe_prefix_before_violation",
                    "deformation_source": "safe_prefix_then_brake",
                    "deformation_deferred": True,
                    "fallback_used": False,
                    "fallback_reason": "safe_prefix_before_violation",
                    "safe_prefix_execution": True,
                    "safe_prefix_len": int(safe_prefix_len),
                    "brake_only_on_violation": True,
                }
            )
            self.brake.brake_streak = 0
            self.brake._update_last_safe_execution(obs, brake_decision_chunk := braked_chunk, info, **kwargs)
            self.last_info = info
            return {
                "result": (braked_chunk.reshape(original_shape), info),
                "braked_chunk": braked_chunk,
                "deform_trigger_reason": "safe_prefix_before_violation",
            }

        brake_progress = float(brake_info.get("progress_scale", 1.0))
        prefer_deform_for_task = (
            self.deform.deformation_enabled
            and brake_progress < self.brake.task_progress_brake_threshold
        )
        progress_deadlock, progress_available = self.brake._temporary_progress_deadlocked()
        deform_trigger_reason = "normal"
        if self.brake.temporary_blocker_enabled and self.brake.temporary_prefer_brake_before_deform:
            reason = self.brake._temporary_deform_trigger_reason(
                progress_deadlock=progress_deadlock,
                progress_available=progress_available,
            )
            progress_gate_open = (
                not self.brake.temporary_require_progress_deadlock_before_deform
                or not progress_available
                or progress_deadlock
                or reason == "brake_timeout"
            )
            if reason is None or not progress_gate_open:
                self.brake.brake_streak += 1
                wait_info = {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deformation_deferred": True,
                    "fallback_reason": "temporary_blocker_wait",
                }
                if bool(info.get("nominal_attribution_pause_recommended", False)):
                    wait_info.update(
                        {
                            "nominal_attribution_pause_applied": True,
                            "nominal_attribution_pause_budget_open": True,
                            "nominal_attribution_pause_budget_exhausted": False,
                            "nominal_attribution_pause_trigger_reason": reason,
                        }
                    )
                info.update(wait_info)
                info.update(self.brake._temporary_streak_info(waiting=True))
                return {
                    "result": self.brake._hold_return_or_emergency_deform(
                        obs, chunk, braked_chunk, info, original_shape, **kwargs
                    ),
                    "braked_chunk": braked_chunk,
                    "deform_trigger_reason": reason,
                }
            deform_trigger_reason = reason

        if (
            brake_info.get("brake_hold_current", False)
            and self.deform.deform_after_deadlock_window
            and self.brake._deadlock_count < self.brake.deadlock_window
            and not prefer_deform_for_task
            and not self.deform.deformation_enabled
        ):
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deformation_deferred": True,
                    "fallback_reason": "immediate_violation_brake",
                }
            )
            self.brake.brake_streak += 1
            info.update(self.brake._temporary_streak_info(waiting=False))
            return {
                "result": self.brake._hold_return_or_emergency_deform(
                    obs, chunk, braked_chunk, info, original_shape, **kwargs
                ),
                "braked_chunk": braked_chunk,
                "deform_trigger_reason": deform_trigger_reason,
            }

        if (
            brake_info["brake_safe"]
            and not prefer_deform_for_task
            and not self.deform.deformation_enabled
            and (
                not brake_info["deadlock"]
                or (
                    self.deform.deform_after_deadlock_window
                    and self.brake._deadlock_count < self.brake.deadlock_window
                )
            )
        ):
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deformation_deferred": bool(brake_info["deadlock"]),
                }
            )
            self.brake.brake_streak += 1
            info.update(self.brake._temporary_streak_info(waiting=False))
            return {
                "result": self.brake._hold_return_or_emergency_deform(
                    obs, chunk, braked_chunk, info, original_shape, **kwargs
                ),
                "braked_chunk": braked_chunk,
                "deform_trigger_reason": deform_trigger_reason,
            }

        if not self.deform.deformation_enabled:
            info.update({"safety_mode": "stop", "mode": "stop"})
            info.update(
                self.brake._temporary_streak_info(
                    trigger_reason=deform_trigger_reason
                )
            )
            if recovery_attempt_reset_info:
                info.update(recovery_attempt_reset_info)
            return {
                "result": self.brake._hold_return_or_emergency_deform(
                    obs, chunk, braked_chunk, info, original_shape, **kwargs
                ),
                "braked_chunk": braked_chunk,
                "deform_trigger_reason": deform_trigger_reason,
            }

        return {
            "result": None,
            "braked_chunk": braked_chunk,
            "deform_trigger_reason": deform_trigger_reason,
        }

    def _perform_contact_rich_brake_only(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        info: dict[str, Any],
        *,
        braked_chunk: np.ndarray,
        deform_trigger_reason: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """During contact-rich mode, forbid spatial recover/deform and brake only."""
        self.brake.brake_streak += 1
        self._mark_contact_rich_pause_info(
            info,
            pause_only=True,
            fallback_reason="contact_rich_brake_only_no_spatial_deform",
        )
        info.update(
            {
                "contact_rich_deform_block_active": True,
                "recover_optimization_blocked_by_contact_rich_state": True,
                "optimized_fallback": "brake",
            }
        )
        info.update(
            self.brake._temporary_streak_info(
                trigger_reason=deform_trigger_reason,
                waiting=False,
            )
        )
        self.last_info = info
        return self.brake._hold_return_or_emergency_deform(
            obs,
            chunk,
            braked_chunk,
            info,
            original_shape,
            **kwargs,
        )

    def _perform_deform(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        safety_info: dict[str, Any],
        info: dict[str, Any],
        *,
        braked_chunk: np.ndarray,
        brake_info: Mapping[str, Any] | None = None,
        nominal_q_seq: np.ndarray | None = None,
        deform_trigger_reason: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run deform/recover optimization and update recovery attempt state."""
        recovery_optimizer_skip_reason: str | None = None
        recovery_attempt_reset_info: dict[str, Any] = {}
        if (
            self.recovery.recoverable_deform_enabled
            and self.recovery.explicit_return
            and self.recovery.safechunk_recover_enabled
        ):
            recovery_attempt_reset_info = (
                self.recovery._maybe_reset_recovery_attempts_after_brake_timeout()
            )
            if self.recovery.recovery_optimizer_cooldown_remaining > 0:
                recovery_optimizer_skip_reason = "cooldown"
                self.recovery.recovery_optimizer_cooldown_remaining = max(
                    0,
                    int(self.recovery.recovery_optimizer_cooldown_remaining) - 1,
                )
            elif (
                self.recovery.recover_max_attempts_per_unsafe_streak > 0
                and self.recovery.recovery_attempts_in_unsafe_streak
                >= self.recovery.recover_max_attempts_per_unsafe_streak
            ):
                recovery_optimizer_skip_reason = "attempt_cap"

        if recovery_optimizer_skip_reason is not None:
            self.recovery.recovery_optimization_skipped_count += 1
            self.brake.brake_streak += 1
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deform_mode": "recovery_optimization_skipped",
                    "deformation_source": "horizon_brake",
                    "deformation_deferred": True,
                    "fallback_reason": f"recovery_optimizer_{recovery_optimizer_skip_reason}",
                    "fallback_used": True,
                    "recovery_optimization_skipped": True,
                    "recovery_optimization_skip_reason": recovery_optimizer_skip_reason,
                }
            )
            info.update(
                self.brake._temporary_streak_info(
                    trigger_reason=deform_trigger_reason
                )
            )
            return {
                "result": self.brake._hold_return_or_emergency_deform(
                    obs, chunk, braked_chunk, info, original_shape, **kwargs
                ),
                "safe_chunk": None,
                "info": info,
            }

        if (
            self.recovery.recoverable_deform_enabled
            and self.recovery.explicit_return
            and self.recovery.safechunk_recover_enabled
        ):
            self.recovery.recovery_attempts_in_unsafe_streak += 1

        # Deform owns optimization; the filter only records the routing outcome.
        safe_chunk, deform_info = self.deform.deform_chunk(
            obs,
            chunk,
            safety_info=safety_info,
            braked_chunk=braked_chunk,
            nominal_q_seq=nominal_q_seq,
            **kwargs,
        )
        info.update(deform_info)
        if recovery_attempt_reset_info:
            info.update(recovery_attempt_reset_info)
        local_escape_result = self._maybe_perform_stationary_human_local_escape(
            obs,
            chunk,
            original_shape,
            safety_info,
            info,
            braked_chunk=braked_chunk,
            brake_info=brake_info,
            deform_trigger_reason=deform_trigger_reason,
            **kwargs,
        )
        if local_escape_result is not None:
            return {
                "result": local_escape_result,
                "safe_chunk": None,
                "info": local_escape_result[1],
            }
        if bool(info.get("optimized_accepted", False)):
            self.recovery.recovery_failure_streak = 0
            self.recovery.recovery_optimizer_cooldown_remaining = 0
            self.recovery.recovery_attempts_in_unsafe_streak = 0
        elif info.get("optimized_accepted") is not None or info.get("fallback_used") is not None:
            self.recovery.recovery_failure_streak += 1
            self.recovery.recovery_failure_streak_max = max(
                self.recovery.recovery_failure_streak_max,
                self.recovery.recovery_failure_streak,
            )
            if (
                self.recovery.recoverable_deform_enabled
                and self.recovery.explicit_return
                and self.recovery.safechunk_recover_enabled
                and self.recovery.recover_retry_cooldown_steps > 0
            ):
                self.recovery.recovery_optimizer_cooldown_remaining = max(
                    int(self.recovery.recovery_optimizer_cooldown_remaining),
                    int(self.recovery.recover_retry_cooldown_steps),
                )
        info.update(
            self.brake._temporary_streak_info(
                trigger_reason=deform_trigger_reason
            )
        )
        return {"result": None, "safe_chunk": safe_chunk, "info": info}

    def _maybe_perform_stationary_human_local_escape(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        safety_info: Mapping[str, Any],
        info: dict[str, Any],
        *,
        braked_chunk: np.ndarray,
        brake_info: Mapping[str, Any] | None = None,
        deform_trigger_reason: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        cfg = self.intervention_fsm.config
        if not bool(cfg.stationary_human_local_escape_enabled):
            info["stationary_human_local_escape_enabled"] = False
            info["stationary_human_local_escape_applied"] = False
            info["stationary_human_local_escape_skip_reason"] = "disabled"
            return None
        max_steps = int(cfg.stationary_human_local_escape_max_steps)
        info.update(
            {
                "stationary_human_local_escape_enabled": True,
                "stationary_human_local_escape_counter": int(
                    self._stationary_human_local_escape_counter
                ),
                "stationary_human_local_escape_max_steps": int(max_steps),
            }
        )
        if max_steps <= 0 or self._stationary_human_local_escape_counter >= max_steps:
            info["stationary_human_local_escape_applied"] = False
            info["stationary_human_local_escape_skip_reason"] = "budget_exhausted"
            return None

        stop_source = brake_info if brake_info is not None else info
        stop_result = self._evaluate_stop_counterfactual(stop_source)
        human_motion_speed = self._finite_float(info.get("human_motion_prediction_speed"))
        if human_motion_speed is None:
            human_motion_speed = self._finite_float(safety_info.get("human_motion_prediction_speed"))
        if human_motion_speed is None:
            human_motion_speed = self._finite_float(kwargs.get("human_motion_prediction_speed"))
        human_motion_displacement = self._finite_float(
            info.get("human_motion_prediction_max_displacement")
        )
        if human_motion_displacement is None:
            human_motion_displacement = self._finite_float(
                safety_info.get("human_motion_prediction_max_displacement")
            )
        static_speed_threshold = float(cfg.pause_deform_static_human_speed_threshold)
        static_displacement_threshold = float(
            cfg.early_deform_static_human_displacement_threshold
        )
        human_motion_static = bool(
            (human_motion_speed is not None and human_motion_speed <= static_speed_threshold)
            or (
                human_motion_displacement is not None
                and human_motion_displacement <= static_displacement_threshold
            )
        )
        path_blocked, path_available, path_source = self._path_blockage_info(
            safety_info, kwargs
        )
        path_pause_sufficient, path_pause_available, path_pause_source = (
            self._path_block_pause_sufficiency_info(kwargs)
        )
        path_requires_bypass = bool(path_blocked and not path_pause_sufficient)
        goal_blocked, goal_distance, goal_available, goal_source = self._goal_blockage_info(kwargs)
        goal_allows_local_escape = bool(
            not goal_blocked
            and (goal_available or not bool(cfg.pause_deform_suppress_requires_goal_check))
        )
        info.update(
            {
                "stationary_human_local_escape_path_blocked": bool(path_blocked),
                "stationary_human_local_escape_path_check_available": bool(path_available),
                "stationary_human_local_escape_path_check_source": path_source,
                "path_block_pause_sufficient": bool(path_pause_sufficient),
                "path_block_pause_sufficiency_available": bool(path_pause_available),
                "path_block_pause_sufficiency_source": path_pause_source,
                "path_block_requires_bypass": bool(path_requires_bypass),
                "stationary_human_local_escape_stop_counterfactual_safe": bool(stop_result.safe),
                "stationary_human_local_escape_stop_counterfactual_min_clearance": float(stop_result.min_distance),
                "stationary_human_local_escape_human_motion_speed": human_motion_speed,
                "stationary_human_local_escape_human_motion_displacement": human_motion_displacement,
                "stationary_human_local_escape_human_motion_static": bool(human_motion_static),
                "stationary_human_local_escape_goal_blocked": bool(goal_blocked),
                "stationary_human_local_escape_goal_distance": goal_distance,
                "stationary_human_local_escape_goal_check_available": bool(goal_available),
                "stationary_human_local_escape_goal_check_source": goal_source,
            }
        )

        clearance_values = [
            self._finite_float(info.get("hold_horizon_min_clearance")),
            self._finite_float(info.get("min_h")),
            self._finite_float(safety_info.get("min_h")),
            self._finite_float(stop_result.min_distance),
        ]
        finite_clearances = [v for v in clearance_values if v is not None and np.isfinite(v)]
        current_human_wait_clearance = (
            max(finite_clearances) if finite_clearances else float("-inf")
        )
        nominal_source = str(
            info.get(
                "nominal_collision_source",
                safety_info.get("nominal_collision_source", "unknown"),
            )
        )
        current_human_clearance_wait = bool(
            nominal_source == "current_human_geometry"
            and not path_blocked
            and goal_allows_local_escape
            and current_human_wait_clearance >= 0.0
        )
        info.update(
            {
                "stationary_human_local_escape_current_human_wait": bool(current_human_clearance_wait),
                "stationary_human_local_escape_current_human_wait_clearance": float(current_human_wait_clearance),
                "stationary_human_local_escape_nominal_source": nominal_source,
            }
        )
        if current_human_clearance_wait and bool(info.get("optimized_accepted", False)):
            self.brake.brake_streak += 1
            self._reset_pause_exit_smoothing_rearm()
            result_info = dict(info)
            result_info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deform_mode": "current_human_clearance_wait_no_recovery",
                    "deformation_source": "horizon_brake",
                    "stationary_human_local_escape_applied": False,
                    "stationary_human_local_escape_skip_reason": "current_human_clearance_wait_no_recovery",
                    "optimized_candidate_suppressed_by_local_escape": True,
                    "optimized_accepted_before_local_escape": bool(info.get("optimized_accepted", False)),
                    "recover_accepted_before_local_escape": bool(info.get("recover_accepted", False)),
                    "return_accepted_before_local_escape": bool(info.get("return_accepted", False)),
                    "optimized_accepted": False,
                    "recover_accepted": False,
                    "return_accepted": False,
                    "is_recoverable": False,
                    "recovery_rejected": True,
                    "deformation_deferred": True,
                    "fallback_used": True,
                    "fallback_reason": "current_human_clearance_wait_no_recovery",
                }
            )
            result_info.update(
                self.brake._temporary_streak_info(
                    trigger_reason=deform_trigger_reason,
                    waiting=True,
                )
            )
            if self.intervention_fsm.config.enabled:
                self.intervention_fsm.transition(
                    InterventionMode.PAUSE_GUARD,
                    "current_human_clearance_wait_no_recovery",
                    UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
                )
                self.intervention_fsm.note_pause_step()
                self._update_intervention_info(result_info)
            self._arm_pause_exit_smoothing(
                braked_chunk,
                result_info,
                reason="current_human_clearance_wait_no_recovery",
            )
            result = self.brake._hold_return_or_emergency_deform(
                obs,
                chunk,
                braked_chunk,
                result_info,
                original_shape,
                **kwargs,
            )
            self.last_info = result[1]
            return result
        if self._stop_pause_is_task_sufficient(
            stop_safe=bool(stop_result.safe),
            path_blocked=bool(path_blocked),
            path_block_pause_sufficient=bool(path_pause_sufficient),
        ):
            self.brake.brake_streak += 1
            self._reset_pause_exit_smoothing_rearm()
            result_info = dict(info)
            result_info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deform_mode": "stop_counterfactual_safe_no_recovery",
                    "deformation_source": "horizon_brake",
                    "stationary_human_local_escape_applied": False,
                    "stationary_human_local_escape_skip_reason": "stop_counterfactual_safe_no_recovery",
                    "optimized_candidate_suppressed_by_local_escape": True,
                    "optimized_accepted_before_local_escape": bool(info.get("optimized_accepted", False)),
                    "recover_accepted_before_local_escape": bool(info.get("recover_accepted", False)),
                    "return_accepted_before_local_escape": bool(info.get("return_accepted", False)),
                    "optimized_accepted": False,
                    "recover_accepted": False,
                    "return_accepted": False,
                    "is_recoverable": False,
                    "recovery_rejected": True,
                    "deformation_deferred": True,
                    "fallback_used": True,
                    "fallback_reason": "stop_counterfactual_safe_no_recovery",
                }
            )
            result_info.update(
                self.brake._temporary_streak_info(
                    trigger_reason=deform_trigger_reason,
                    waiting=True,
                )
            )
            if self.intervention_fsm.config.enabled:
                self.intervention_fsm.transition(
                    InterventionMode.PAUSE_GUARD,
                    "stop_counterfactual_safe_no_recovery",
                    UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
                )
                self.intervention_fsm.note_pause_step()
                self._update_intervention_info(result_info)
            self._arm_pause_exit_smoothing(
                braked_chunk,
                result_info,
                reason="stop_counterfactual_safe_no_recovery",
            )
            self.brake._update_last_safe_execution(obs, braked_chunk, result_info, **kwargs)
            self.last_info = result_info
            return braked_chunk.reshape(original_shape), result_info
        if not human_motion_static:
            info["stationary_human_local_escape_applied"] = False
            info["stationary_human_local_escape_skip_reason"] = "human_motion_not_static"
            return None
        if not path_blocked:
            info["stationary_human_local_escape_applied"] = False
            info["stationary_human_local_escape_skip_reason"] = "path_not_blocked"
            return None
        if not goal_allows_local_escape:
            info["stationary_human_local_escape_applied"] = False
            info["stationary_human_local_escape_skip_reason"] = (
                "goal_blocked" if goal_blocked else "goal_check_required_unavailable"
            )
            return None
        if not bool(info.get("optimized_accepted", False)):
            info["stationary_human_local_escape_candidate_accepted"] = False
            info["stationary_human_local_escape_applied"] = False
            info["stationary_human_local_escape_skip_reason"] = "no_accepted_recovery_candidate"
            return None

        escape_chunk, escape_info = self.brake.emergency_deform_away(
            obs,
            braked_chunk,
            nominal_chunk=chunk,
            hold_info=info,
            **kwargs,
        )
        accepted = bool(escape_info.get("accepted", False))
        info.update(
            {
                "stationary_human_local_escape_candidate_accepted": bool(accepted),
                "stationary_human_local_escape_candidate_path": escape_info.get("accepted_path_name"),
                "stationary_human_local_escape_candidate_hold_clearance": escape_info.get("hold_horizon_min_clearance"),
            }
        )
        if not accepted:
            info["stationary_human_local_escape_applied"] = False
            info["stationary_human_local_escape_skip_reason"] = escape_info.get(
                "hold_rejected_reason",
                "local_escape_candidate_rejected",
            )
            return None

        self._stationary_human_local_escape_counter += 1
        self.brake.brake_streak += 1
        self._reset_pause_exit_smoothing_rearm()
        result_info = dict(info)
        result_info.update(escape_info)
        result_info.update(
            {
                "safety_mode": "emergency_deform_away",
                "mode": "emergency_deform_away",
                "deform_mode": "stationary_human_local_escape",
                "deformation_source": "stationary_human_local_escape",
                "stationary_human_local_escape_applied": True,
                "stationary_human_local_escape_skip_reason": None,
                "stationary_human_local_escape_counter": int(
                    self._stationary_human_local_escape_counter
                ),
                "stationary_human_local_escape_trigger_reason": deform_trigger_reason,
                "optimized_candidate_suppressed_by_local_escape": True,
                "optimized_accepted_before_local_escape": bool(info.get("optimized_accepted", False)),
                "recover_accepted_before_local_escape": bool(info.get("recover_accepted", False)),
                "return_accepted_before_local_escape": bool(info.get("return_accepted", False)),
                "optimized_accepted": False,
                "recover_accepted": False,
                "return_accepted": False,
                "is_recoverable": False,
                "recovery_rejected": True,
                "deformation_deferred": True,
                "fallback_used": False,
                "fallback_reason": "stationary_human_local_escape",
            }
        )
        result_info.update(
            self.brake._temporary_streak_info(
                trigger_reason=deform_trigger_reason,
                waiting=True,
            )
        )
        if self.intervention_fsm.config.enabled:
            self.intervention_fsm.transition(
                InterventionMode.SLOWDOWN_GUARD,
                "stationary_human_local_escape",
                UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
            )
            self.intervention_fsm.note_slowdown_step()
            self._update_intervention_info(result_info)
        try:
            self.intervention_fsm.previous_safe_action = np.asarray(
                escape_chunk[0],
                dtype=np.float32,
            ).reshape(-1).copy()
        except Exception:  # noqa: BLE001
            pass
        self.brake._update_last_safe_execution(obs, escape_chunk, result_info, **kwargs)
        self.last_info = result_info
        return escape_chunk.reshape(original_shape), result_info

    def _perform_recover_commit(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        *,
        safe_chunk: np.ndarray | None,
        braked_chunk: np.ndarray,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Commit accepted explicit recovery chunks and serve the first step."""
        if not (
            info.get("optimized_accepted", False)
            and self.recovery.explicit_return
            and self.recovery.commit_accepted_chunks
        ):
            return None

        # Recovery owns commit validation and serving of accepted explicit-return plans.
        committed, commit_reject_info = self.recovery._commit_explicit_recovery_chunk(
            obs,
            safe_chunk,
            info,
            **kwargs,
        )
        if not committed:
            info.update(commit_reject_info)
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deform_mode": "committed_recovery_commit_rejected",
                    "deformation_source": "horizon_brake",
                    "optimized_accepted": False,
                    "optimized_fallback": "brake",
                    "optimized_reject_reason": "committed_rejected_missing_planned_q",
                    "fallback_reason": "committed_rejected_missing_planned_q",
                    "fallback_used": True,
                    "recover_corridor_accepted": bool(
                        info.get(
                            "recover_corridor_accepted",
                            info.get("return_accepted", info.get("recover_accepted", False)),
                        )
                    ),
                    "recover_accepted": False,
                    "return_accepted": False,
                    "is_recoverable": False,
                    "recovery_rejected": True,
                }
            )
            return self.brake._hold_return_or_emergency_deform(
                obs, chunk, braked_chunk, info, original_shape, **kwargs
            )
        if self.intervention_fsm.config.enabled:
            self.intervention_fsm.transition(
                InterventionMode.DEFORM_COMMIT,
                "accepted_deformation_committed",
            )
            self._update_intervention_info(info)
        committed_result = self.recovery._serve_committed_chunk(obs, chunk, original_shape, **kwargs)
        pending_committed_replan_info = self.recovery._pop_pending_committed_replan_info()
        if pending_committed_replan_info:
            info.update(pending_committed_replan_info)
        if committed_result is None:
            return None

        committed_chunk, committed_info = committed_result
        for key in self._RECOVERY_COMMIT_DIAGNOSTIC_KEYS:
            if key in info:
                committed_info[key] = info[key]
        committed_chunk, committed_info = self._apply_pause_exit_smoothing(
            obs,
            committed_chunk,
            original_shape,
            committed_info,
        )
        self.last_info = committed_info
        return committed_chunk, committed_info

    def _perform_deform_fallback_or_finalize(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        *,
        safety_info: Mapping[str, Any],
        safe_chunk: np.ndarray,
        braked_chunk: np.ndarray,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Choose brake fallback for rejected deforms or return final deform chunk."""
        if (
            not info.get("deform_safe", False)
            and self.deform.unsafe_deformation_fallback == "brake"
        ):
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deformation_rejected": True,
                    "fallback_reason": info.get("fallback_reason", "deform_unsafe"),
                }
            )
            slowdown_result = self._maybe_perform_pause_guard_slowdown(
                obs,
                chunk,
                original_shape,
                safety_info,
                info,
                deform_trigger_reason=str(
                    info.get("nominal_attribution_pause_trigger_reason", "deform_rejected")
                ),
                fallback_reason=str(info.get("fallback_reason", "deform_unsafe")),
                **kwargs,
            )
            if slowdown_result is not None:
                return slowdown_result
            self.last_info = info
            return braked_chunk.reshape(original_shape), info

        info.update({"safety_mode": "horizon_deform", "mode": "horizon_deform"})
        valid = self.deform._valid_control_indices(chunk)
        if np.any(valid):
            action_idx = self.controlled_action_indices[valid]
            safe_chunk = self.deform._project_optimized_chunk(safe_chunk, chunk, action_idx)
        output_chunk, info = self._apply_pause_exit_smoothing(
            obs,
            safe_chunk,
            original_shape,
            info,
        )
        if self.intervention_fsm.config.enabled:
            if self.intervention_fsm.mode != InterventionMode.DEFORM_COMMIT:
                self.intervention_fsm.transition(
                    InterventionMode.DEFORM_COMMIT,
                    "direct_deform_commit",
                )
            try:
                served_arr, _ = self.intervention_factory._as_chunk(output_chunk)
                valid = self.controlled_action_indices[
                    self.controlled_action_indices < served_arr.shape[1]
                ]
                action_norm = float(np.linalg.norm(served_arr[0, valid])) if valid.size else None
                self.intervention_fsm.previous_safe_action = np.asarray(
                    served_arr[0],
                    dtype=np.float32,
                ).reshape(-1).copy()
            except Exception:  # noqa: BLE001
                action_norm = None
            stalled = self.intervention_fsm.note_deform_commit_step(action_norm=action_norm)
            if stalled:
                self._latch_deformation_failure(
                    info,
                    reason="deformation_stalled",
                    obs=obs,
                    chunk=chunk,
                    q_seq=None,
                )
            self._update_intervention_info(info)
        self.last_info = info
        return output_chunk, info
