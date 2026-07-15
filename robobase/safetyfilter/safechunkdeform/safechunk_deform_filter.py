from __future__ import annotations

import logging
import time
from typing import Any, Optional, Mapping

import numpy as np

from .safechunk_brake import Brake
from .safechunk_deform import Deform, DeformConfig
from .safechunk_recovery import Recovery, RecoveryContext
from .safechunk_intervention_factory import InterventionExecutionFactory


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

        committed_result = self.recovery._serve_committed_chunk(obs, chunk, original_shape, **kwargs)
        if committed_result is not None:
            return committed_result
        pending_committed_replan_info = self.recovery._pop_pending_committed_replan_info()

        # Shared factory evaluates nominal horizon safety before routing interventions.
        q_seq, rollout_mismatch_info = self._apply_rollout_residual_correction(raw_q_seq, kwargs)
        safety_info = self.intervention_factory.evaluate_horizon_safety(obs, q_seq)
        self._store_nominal_rollout_context(
            obs,
            chunk,
            raw_q_seq=raw_q_seq,
            q_seq=q_seq,
            safety_info=safety_info,
            rollout_mismatch_info=rollout_mismatch_info,
        )
        info = dict(safety_info)
        info.update(rollout_mismatch_info)
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
                deform_trigger_reason=brake_decision["deform_trigger_reason"],
                **kwargs,
            )

        deform_decision = self._perform_deform(
            obs,
            chunk,
            original_shape,
            safety_info,
            info,
            braked_chunk=brake_decision["braked_chunk"],
            nominal_q_seq=q_seq,
            deform_trigger_reason=brake_decision["deform_trigger_reason"],
            **kwargs,
        )
        if deform_decision["result"] is not None:
            return deform_decision["result"]

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
            chunk,
            original_shape,
            safe_chunk=deform_decision["safe_chunk"],
            braked_chunk=brake_decision["braked_chunk"],
            info=deform_decision["info"],
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
        self.brake._update_last_safe_execution(obs, chunk, info, **kwargs)
        self.last_info = info
        return chunk.reshape(original_shape), info

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
        if self.unsafe_streak == 0:
            self.recovery.blocked_nominal_chunk = np.asarray(chunk, dtype=np.float32).copy()
            self.recovery.blocked_nominal_step = int(self.recovery.latest_nominal_step)
        self.unsafe_streak += 1
        # Brake is always attempted first; deformation is gated by brake outcome.
        braked_chunk, brake_info = self.brake.horizon_brake(obs, chunk, safety_info)
        info.update(brake_info)
        if brake_info["deadlock"]:
            self.brake._deadlock_count += 1
        else:
            self.brake._deadlock_count = 0
        info["deadlock_count"] = int(self.brake._deadlock_count)

        safe_prefix_len = int(brake_info.get("safe_prefix_len") or 0)
        if safe_prefix_len > 0 and not bool(brake_info.get("brake_hold_current", False)):
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
                info.update(
                    {
                        "safety_mode": "horizon_brake",
                        "mode": "horizon_brake",
                        "deformation_deferred": True,
                        "fallback_reason": "temporary_blocker_wait",
                    }
                )
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
        nominal_q_seq: np.ndarray | None,
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
        self.last_info = committed_info
        return committed_chunk, committed_info

    def _perform_deform_fallback_or_finalize(
        self,
        chunk: np.ndarray,
        original_shape: tuple[int, ...],
        *,
        safe_chunk: np.ndarray,
        braked_chunk: np.ndarray,
        info: dict[str, Any],
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
            self.last_info = info
            return braked_chunk.reshape(original_shape), info

        info.update({"safety_mode": "horizon_deform", "mode": "horizon_deform"})
        valid = self.deform._valid_control_indices(chunk)
        if np.any(valid):
            action_idx = self.controlled_action_indices[valid]
            safe_chunk = self.deform._project_optimized_chunk(safe_chunk, chunk, action_idx)
        self.last_info = info
        return safe_chunk.reshape(original_shape), info
