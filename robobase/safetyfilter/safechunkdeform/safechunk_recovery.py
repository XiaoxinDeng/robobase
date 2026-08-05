from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping

import time
import numpy as np

from .safechunk_intervention_factory import InterventionExecutionFactory
from .safechunk_mpc import MPCRecoveryController


logger = logging.getLogger(__name__)


ConfigDict = dict[str, Any]
InfoDict = dict[str, Any]
RecoveryResult = tuple[np.ndarray, InfoDict]
CostResult = tuple[float, InfoDict]


@dataclass
class RecoveryContext:
    """Cached nominal trajectory and bookkeeping for one recovery attempt."""

    nominal_chunk: np.ndarray
    nominal_q_seq: np.ndarray
    nominal_ee_seq: np.ndarray | None = None
    start_chunk_index: int | None = None
    trigger_step: int | None = None
    active: bool = True
    target_rejoin_index: int | None = None
    phase: str = "horizon_deform"
    observation_history: Any | None = None
    policy_buffer_metadata: Any | None = None
    return_retries: int = 0
    recover_retries: int = 0
    mpc_reference_index: int | None = None
    mpc_replans: int = 0
    mpc_no_progress_count: int = 0


class Recovery(InterventionExecutionFactory):
    """Recovery-only execution paths."""

    def __init__(
        self,
        parent: Any,
        sync: bool | None = None,
        *,
        deformation_enabled: bool = True,
        temporary_blocker: dict[str, Any] | None = None,
        recoverable_deform: dict[str, Any] | None = None,
        optimized_deform: dict[str, Any] | None = None,
        explicit_recovery: dict[str, Any] | None = None,
        safechunk_replan: dict[str, Any] | None = None,
        safechunk_recover: dict[str, Any] | None = None,
        safechunk_recovery_corridor: dict[str, Any] | None = None,
        intervention: Mapping[str, Any] | None = None,
        intervention_factory: Any | None = None,
        lambda_rejoin: float = 5.0,
        lambda_smooth: float = 0.1,
        rejoin_threshold: float = 0.03,
        min_rejoin_offset: int = 2,
        use_ee_pose_rejoin: bool = False,
        use_object_state_rejoin: bool = False,
        brake_if_unrecoverable: bool = True,
    ) -> None:
        super().__init__(
            parent,
            intervention=intervention,
            intervention_factory=intervention_factory,
        )
        self.rollout_model_default_variant = "recovery"
        del sync
        self._init_config(
            deformation_enabled=deformation_enabled,
            temporary_blocker=temporary_blocker,
            recoverable_deform=recoverable_deform,
            optimized_deform=optimized_deform,
            explicit_recovery=explicit_recovery,
            safechunk_replan=safechunk_replan,
            safechunk_recover=safechunk_recover,
            safechunk_recovery_corridor=safechunk_recovery_corridor,
            intervention=intervention,
            lambda_rejoin=lambda_rejoin,
            lambda_smooth=lambda_smooth,
            rejoin_threshold=rejoin_threshold,
            min_rejoin_offset=min_rejoin_offset,
            use_ee_pose_rejoin=use_ee_pose_rejoin,
            use_object_state_rejoin=use_object_state_rejoin,
            brake_if_unrecoverable=brake_if_unrecoverable,
        )
        self.mpc: MPCRecoveryController = MPCRecoveryController(self)
        self._init_execution_state()

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """Coerce common scalar/array-like structures to a boolean flag."""
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return False
            if value.ndim == 0:
                return bool(value.item())
            return bool(np.all(value))

        if isinstance(value, (list, tuple, set)):
            arr = np.asarray(list(value))
            if arr.size == 0:
                return False
            if arr.ndim == 0:
                return bool(arr.item())
            return bool(np.all(arr))

        try:
            return bool(value)
        except ValueError:
            arr = np.asarray(value)
            if arr.size == 0:
                return False
            return bool(np.all(arr))

    def _init_config(
        self,
        *,
        deformation_enabled: bool,
        temporary_blocker: Mapping[str, Any] | None,
        recoverable_deform: Mapping[str, Any] | None,
        optimized_deform: Mapping[str, Any] | None,
        explicit_recovery: Mapping[str, Any] | None,
        safechunk_replan: Mapping[str, Any] | None,
        safechunk_recover: Mapping[str, Any] | None,
        safechunk_recovery_corridor: Mapping[str, Any] | None,
        intervention: Mapping[str, Any] | None,
        lambda_rejoin: float,
        lambda_smooth: float,
        rejoin_threshold: float,
        min_rejoin_offset: int,
        use_ee_pose_rejoin: bool,
        use_object_state_rejoin: bool,
        brake_if_unrecoverable: bool,
    ) -> None:
        """Populate recovery runtime config from nested config dictionaries."""
        # Core toggles shared across recovery and deform branches.
        self.deformation_enabled: bool = bool(deformation_enabled)
        self.lambda_smooth: float = float(lambda_smooth)

        # Temporary blocker thresholds decide whether to pause recovery briefly.
        temporary_cfg: ConfigDict = self._temporary_blocker_config(temporary_blocker)
        self.temporary_blocker_enabled: bool = bool(temporary_cfg["enabled"])
        self.temporary_recover_after_wait = bool(
            temporary_cfg["recover_after_wait"]
        )
        self.temporary_recover_after_wait_min_brake_steps = int(
            temporary_cfg["recover_after_wait_min_brake_steps"]
        )

        # Optimizable recovery path (for explicit fallback/return modes).
        recoverable_optimization_cfg: ConfigDict = (
            self._recoverable_optimization_config(optimized_deform)
        )
        self.optimizer_method: str = str(recoverable_optimization_cfg["optimizer_method"]).lower()
        self.opt_iters: int = max(0, int(recoverable_optimization_cfg["opt_iters"]))
        self.opt_lr: float = max(1e-9, float(recoverable_optimization_cfg["opt_lr"]))
        self.opt_population: int = max(4, int(recoverable_optimization_cfg["opt_population"]))
        self.opt_elite_frac: float = float(recoverable_optimization_cfg["opt_elite_frac"])
        self.opt_seed: int | None = recoverable_optimization_cfg["opt_seed"]
        self._rng: np.random.Generator = np.random.default_rng(self.opt_seed)
        self.jax_batched_optimizer = bool(
            recoverable_optimization_cfg["jax_batched_optimizer"]
        )
        self.jax_batched_optimizer_fallback = bool(
            recoverable_optimization_cfg["jax_batched_optimizer_fallback"]
        )
        self.gradient_samples = max(1, int(recoverable_optimization_cfg["gradient_samples"]))
        self.gradient_eps = max(1e-9, float(recoverable_optimization_cfg["gradient_eps"]))
        self.gradient_adam_beta1 = float(recoverable_optimization_cfg["gradient_adam_beta1"])
        self.gradient_adam_beta2 = float(recoverable_optimization_cfg["gradient_adam_beta2"])
        self.gradient_min_improvement = max(
            0.0, float(recoverable_optimization_cfg["gradient_min_improvement"])
        )
        self.gradient_line_search_scales = recoverable_optimization_cfg[
            "gradient_line_search_scales"
        ]
        self.gradient_batched_line_search = bool(
            recoverable_optimization_cfg["gradient_batched_line_search"]
        )
        self.gradient_early_stop_on_candidate = bool(
            recoverable_optimization_cfg["gradient_early_stop_on_candidate"]
        )
        self.debug_safety_feasibility: bool = bool(
            recoverable_optimization_cfg.get("debug_safety_feasibility", False)
        )

        # Recoverable deformation thresholds and objective terms.
        recoverable_cfg: ConfigDict = self._recoverable_deform_config(
            recoverable_deform,
            intervention=intervention,
            lambda_rejoin=lambda_rejoin,
            rejoin_threshold=rejoin_threshold,
            min_rejoin_offset=min_rejoin_offset,
            use_ee_pose_rejoin=use_ee_pose_rejoin,
            use_object_state_rejoin=use_object_state_rejoin,
            brake_if_unrecoverable=brake_if_unrecoverable,
        )
        final_rejoin_metric: str = str(recoverable_cfg["final_rejoin_metric"])
        if (
            not bool(recoverable_cfg["use_ee_final_check"])
            and final_rejoin_metric == "ee_pose"
        ):
            final_rejoin_metric = "q_state"
        self.recoverable_deform_enabled: bool = bool(recoverable_cfg["enabled"])
        self.lambda_rejoin: float = float(recoverable_cfg["lambda_rejoin"])
        self.rejoin_threshold: float = float(recoverable_cfg["rejoin_threshold"])
        self.q_rejoin_threshold: float = float(recoverable_cfg["q_rejoin_threshold"])
        self.qd_rejoin_threshold: float = float(recoverable_cfg["qd_rejoin_threshold"])
        self.require_qd_rejoin: bool = bool(recoverable_cfg["require_qd_rejoin"])
        self.qd_rejoin_hard_threshold = float(recoverable_cfg["qd_rejoin_hard_threshold"])
        self.ee_rejoin_threshold = float(recoverable_cfg["ee_rejoin_threshold"])
        self.min_rejoin_offset: int = max(0, int(recoverable_cfg["min_rejoin_offset"]))
        self.use_ee_pose_rejoin = bool(recoverable_cfg["use_ee_pose_rejoin"])
        self.use_object_state_rejoin = bool(recoverable_cfg["use_object_state_rejoin"])
        self.brake_if_unrecoverable = bool(recoverable_cfg["brake_if_unrecoverable"])
        self.inner_rejoin_metric: str = str(recoverable_cfg["inner_rejoin_metric"])
        self.final_rejoin_metric: str = final_rejoin_metric
        self.cache_nominal_ee = bool(recoverable_cfg["cache_nominal_ee"])
        self.ee_rejoin_in_inner_loop = bool(recoverable_cfg["ee_rejoin_in_inner_loop"])
        self.q_rejoin_weights: Any = recoverable_cfg["q_rejoin_weights"]
        self.explicit_return: bool = bool(recoverable_cfg["explicit_return"])
        self.acceptance_clearance_tol = float(recoverable_cfg["acceptance_clearance_tol"])
        self.lambda_deform_safety = float(recoverable_cfg["lambda_deform_safety"])
        self.lambda_deform_action = float(recoverable_cfg["lambda_deform_action"])
        self.lambda_deform_smooth = float(recoverable_cfg["lambda_deform_smooth"])
        self.lambda_retreat = float(recoverable_cfg["lambda_retreat"])
        self.lambda_return_safety = float(recoverable_cfg["lambda_return_safety"])
        self.lambda_return_rejoin = float(recoverable_cfg["lambda_return_rejoin"])
        self.lambda_return_smooth = float(recoverable_cfg["lambda_return_smooth"])
        self.lambda_return_action = float(recoverable_cfg["lambda_return_action"])
        self.deform_horizon: int = max(1, int(recoverable_cfg["deform_horizon"]))
        self.return_horizon: int = max(1, int(recoverable_cfg["return_horizon"]))
        self.max_return_retries = max(0, int(recoverable_cfg["max_return_retries"]))
        self.use_ee_final_check = bool(recoverable_cfg["use_ee_final_check"])

        # Explicit recovery/committed plan policy.
        explicit_cfg: ConfigDict = self._explicit_recovery_config(explicit_recovery)
        committed_state_error_action = str(
            explicit_cfg["committed_state_error_action"]
        ).lower()
        if committed_state_error_action not in {"replan", "abort_to_brake"}:
            raise ValueError(
                "explicit_recovery.committed_state_error_action must be one of "
                "[replan, abort_to_brake], got "
                f"{committed_state_error_action}"
            )
        raw_resume_q_threshold = explicit_cfg["opportunistic_resume_q_threshold"]
        self.commit_accepted_chunks = bool(explicit_cfg["commit_accepted_chunks"])
        self.committed_chunk_safety_check = bool(
            explicit_cfg["committed_chunk_safety_check"]
        )
        self.committed_safety_tol = float(explicit_cfg["committed_safety_tol"])
        self.committed_abort_only_if_contact_risk = bool(
            explicit_cfg["committed_abort_only_if_contact_risk"]
        )
        self.committed_min_clearance_for_abort = float(
            explicit_cfg["committed_min_clearance_for_abort"]
        )
        self.committed_deform_min_clearance_for_abort = (
            None
            if explicit_cfg["committed_deform_min_clearance_for_abort"] is None
            else float(explicit_cfg["committed_deform_min_clearance_for_abort"])
        )
        self.repair_committed_action = bool(explicit_cfg["repair_committed_action"])
        self.monotonic_committed_repair = bool(
            explicit_cfg["monotonic_committed_repair"]
        )
        self.committed_execution_margin = float(explicit_cfg["committed_execution_margin"])
        self.committed_state_error_threshold = float(
            explicit_cfg["committed_state_error_threshold"]
        )
        self.committed_state_error_action = committed_state_error_action
        self.committed_state_mismatch_abort_requires_unsafe = bool(
            explicit_cfg["committed_state_mismatch_abort_requires_unsafe"]
        )
        self.replan_committed_suffix_on_state_mismatch = bool(
            explicit_cfg["replan_committed_suffix_on_state_mismatch"]
        )
        self.committed_suffix_replan_min_remaining = max(
            1,
            int(explicit_cfg["committed_suffix_replan_min_remaining"]),
        )
        self.opportunistic_act_resume = bool(explicit_cfg["opportunistic_act_resume"])
        self.opportunistic_resume_q_threshold = (
            None
            if raw_resume_q_threshold is None
            else float(raw_resume_q_threshold)
        )
        self.opportunistic_resume_min_clearance = float(
            explicit_cfg["opportunistic_resume_min_clearance"]
        )
        self.max_recover_steps_before_act_resume = max(
            0,
            int(explicit_cfg["max_recover_steps_before_act_resume"]),
        )
        self.max_suffix_replans_per_recovery = max(
            0,
            int(explicit_cfg["max_suffix_replans_per_recovery"]),
        )
        self.committed_rollout_gain = float(explicit_cfg["committed_rollout_gain"])
        self.closed_loop_recovery_tracking = bool(
            explicit_cfg["closed_loop_recovery_tracking"]
        )
        self.adaptive_committed_rollout_gain = bool(
            explicit_cfg["adaptive_committed_rollout_gain"]
        )
        self.rollout_gain_ema = float(explicit_cfg["rollout_gain_ema"])
        self.rollout_gain_min = float(explicit_cfg["rollout_gain_min"])
        self.rollout_gain_max = float(explicit_cfg["rollout_gain_max"])
        self.cancel_committed_on_nominal_safe = bool(
            explicit_cfg["cancel_committed_on_nominal_safe"]
        )
        self.committed_cancel_min_clearance = float(
            explicit_cfg["committed_cancel_min_clearance"]
        )
        self.mpc_recovery_enabled = bool(explicit_cfg["mpc_recovery_enabled"])
        self.mpc_recovery_horizon = max(1, int(explicit_cfg["mpc_recovery_horizon"]))
        self.mpc_recovery_prefix_len = max(
            1,
            int(explicit_cfg["mpc_recovery_prefix_len"]),
        )
        self.mpc_recovery_max_replans_per_recovery = max(
            0,
            int(explicit_cfg["mpc_recovery_max_replans_per_recovery"]),
        )
        self.mpc_recovery_require_ordered_progress = bool(
            explicit_cfg["mpc_recovery_require_ordered_progress"]
        )
        self.mpc_recovery_require_live_progress = bool(
            explicit_cfg["mpc_recovery_require_live_progress"]
        )
        self.mpc_recovery_min_progress_delta = float(
            explicit_cfg["mpc_recovery_min_progress_delta"]
        )
        self.mpc_recovery_no_progress_limit = max(
            0,
            int(explicit_cfg["mpc_recovery_no_progress_limit"]),
        )
        self.mpc_recovery_target_tube_enabled: bool = bool(
            explicit_cfg["mpc_recovery_target_tube_enabled"]
        )
        raw_tube_radius = explicit_cfg["mpc_recovery_target_tube_radius"]
        self.mpc_recovery_target_tube_radius: float | None = (
            None if raw_tube_radius is None else max(0.0, float(raw_tube_radius))
        )
        self.mpc_recovery_target_tube_weight: float = max(
            0.0,
            float(explicit_cfg["mpc_recovery_target_tube_weight"]),
        )
        self.mpc_recovery_target_tube_require_progress: bool = bool(
            explicit_cfg["mpc_recovery_target_tube_require_progress"]
        )
        self.mpc_recovery_target_tube_window_len: int = max(
            1,
            int(explicit_cfg["mpc_recovery_target_tube_window_len"]),
        )
        self.mpc_recovery_target_tube_window_weight: float = max(
            0.0,
            float(explicit_cfg["mpc_recovery_target_tube_window_weight"]),
        )
        self.mpc_recovery_target_tube_window_dq_weight: float = max(
            0.0,
            float(explicit_cfg["mpc_recovery_target_tube_window_dq_weight"]),
        )
        self.mpc_recovery_target_tube_window_action_weight: float = max(
            0.0,
            float(explicit_cfg["mpc_recovery_target_tube_window_action_weight"]),
        )
        self.mpc_recovery_target_tube_window_max_q_frame_l2_mean: float = float(
            explicit_cfg["mpc_recovery_target_tube_window_max_q_frame_l2_mean"]
        )
        self.mpc_recovery_target_tube_window_max_q_frame_l2_max: float = float(
            explicit_cfg["mpc_recovery_target_tube_window_max_q_frame_l2_max"]
        )
        self.mpc_recovery_target_tube_window_min_dq_cosine: float = float(
            explicit_cfg["mpc_recovery_target_tube_window_min_dq_cosine"]
        )
        self.mpc_recovery_target_tube_window_max_step_l2_error: float = float(
            explicit_cfg["mpc_recovery_target_tube_window_max_step_l2_error"]
        )
        self.mpc_bridge_replan_cooldown_steps: int = max(
            0,
            int(explicit_cfg["mpc_bridge_replan_cooldown_steps"]),
        )
        self.mpc_bridge_max_replans_per_recovery: int = max(
            0,
            int(explicit_cfg["mpc_bridge_max_replans_per_recovery"]),
        )
        self.mpc_bridge_direction_hard_gate: bool = bool(
            explicit_cfg["mpc_bridge_direction_hard_gate"]
        )
        self.mpc_bridge_min_heading_improvement: float = max(
            0.0,
            float(explicit_cfg["mpc_bridge_min_heading_improvement"]),
        )
        self.mpc_bridge_min_progress_improvement: float = max(
            0.0,
            float(explicit_cfg["mpc_bridge_min_progress_improvement"]),
        )
        self.mpc_bridge_min_clearance_improvement: float = max(
            0.0,
            float(explicit_cfg["mpc_bridge_min_clearance_improvement"]),
        )
        self.mpc_handoff_action_agreement_override_enabled: bool = bool(
            explicit_cfg.get("mpc_handoff_action_agreement_override_enabled", False)
        )
        self.mpc_handoff_action_agreement_source: str = str(
            explicit_cfg.get("mpc_handoff_action_agreement_source", "release")
        ).lower()
        if self.mpc_handoff_action_agreement_source not in {"release", "target", "both"}:
            self.mpc_handoff_action_agreement_source = "release"
        self.mpc_handoff_require_resume_readiness: bool = bool(
            explicit_cfg.get("mpc_handoff_require_resume_readiness", False)
        )
        self.mpc_handoff_use_selected_act_action_if_safe: bool = bool(
            explicit_cfg.get("mpc_handoff_use_selected_act_action_if_safe", False)
        )
        self.mpc_handoff_bridge_ramp_on_resume_not_ready: bool = bool(
            explicit_cfg.get("mpc_handoff_bridge_ramp_on_resume_not_ready", False)
        )
        self.mpc_handoff_bridge_ramp_max_steps: int = max(
            0, int(explicit_cfg.get("mpc_handoff_bridge_ramp_max_steps", 0))
        )
        self.mpc_handoff_shadow_prefix_len: int = max(
            1, int(explicit_cfg.get("mpc_handoff_shadow_prefix_len", 4))
        )
        self.mpc_handoff_require_shadow_prefix: bool = bool(
            explicit_cfg.get("mpc_handoff_require_shadow_prefix", False)
        )
        self.mpc_handoff_action_l2_threshold: float = max(
            0.0,
            float(explicit_cfg.get("mpc_handoff_action_l2_threshold", 0.15)),
        )
        self.mpc_handoff_action_cosine_threshold: float = float(
            explicit_cfg.get("mpc_handoff_action_cosine_threshold", 0.98)
        )
        self.mpc_handoff_action_arm_l2_threshold: float = max(
            0.0,
            float(explicit_cfg.get("mpc_handoff_action_arm_l2_threshold", 0.15)),
        )
        self.mpc_state_error_weights = self._make_mpc_state_weights(
            explicit_cfg.get("mpc_state_error_weights"),
            base_weight=float(explicit_cfg.get("mpc_state_error_base_weight", 1.0)),
            base_yaw_weight=float(explicit_cfg.get("mpc_state_error_base_yaw_weight", 0.5)),
            arm_weight=float(explicit_cfg.get("mpc_state_error_arm_weight", 1.0)),
            ignore_indices=explicit_cfg.get("mpc_state_error_ignore_indices", [2]),
        )
        self.mpc_handoff_state_weights = self._make_mpc_state_weights(
            explicit_cfg.get(
                "mpc_handoff_state_weights",
                explicit_cfg.get("mpc_state_error_weights"),
            ),
            base_weight=float(explicit_cfg.get("mpc_handoff_base_weight", 1.0)),
            base_yaw_weight=float(explicit_cfg.get("mpc_handoff_base_yaw_weight", 0.5)),
            arm_weight=float(explicit_cfg.get("mpc_handoff_arm_weight", 1.0)),
            ignore_indices=explicit_cfg.get(
                "mpc_handoff_ignore_indices",
                explicit_cfg.get("mpc_state_error_ignore_indices", [2]),
            ),
        )
        self.mpc_replan_every_recovery_step: bool = bool(
            explicit_cfg.get("mpc_replan_every_recovery_step", True)
        )
        self.committed_receding_recover_steps: int = max(
            0,
            int(explicit_cfg.get("committed_receding_recover_steps", 1)),
        )
        self.committed_nominal_tube_tracking_enabled: bool = bool(
            explicit_cfg.get("committed_nominal_tube_tracking_enabled", True)
        )
        self.committed_nominal_tube_tracking_arm_gain: float = max(
            0.0,
            float(explicit_cfg.get("committed_nominal_tube_tracking_arm_gain", 0.45)),
        )
        self.committed_nominal_tube_tracking_base_gain: float = max(
            0.0,
            float(explicit_cfg.get("committed_nominal_tube_tracking_base_gain", 0.75)),
        )
        self.committed_nominal_tube_tracking_max_arm_step: float = max(
            0.0,
            float(explicit_cfg.get("committed_nominal_tube_tracking_max_arm_step", 0.24)),
        )
        self.committed_nominal_tube_tracking_max_base_delta: float = max(
            0.0,
            float(explicit_cfg.get("committed_nominal_tube_tracking_max_base_delta", 0.06)),
        )
        self.committed_nominal_tube_tracking_done_threshold: float = max(
            0.0,
            float(
                explicit_cfg.get(
                    "committed_nominal_tube_tracking_done_threshold", 0.24
                )
            ),
        )
        self.committed_nominal_tube_tracking_max_recover_steps: int = max(
            0,
            int(
                explicit_cfg.get(
                    "committed_nominal_tube_tracking_max_recover_steps", 0
                )
            ),
        )
        self.committed_nominal_tube_tracking_rollout_solver: bool = bool(
            explicit_cfg.get("committed_nominal_tube_tracking_rollout_solver", True)
        )
        self.committed_nominal_tube_tracking_score_scale: float = max(
            1e-6,
            float(explicit_cfg.get("committed_nominal_tube_tracking_score_scale", 0.35)),
        )
        self.committed_nominal_tube_tracking_action_smooth_weight: float = max(
            0.0,
            float(
                explicit_cfg.get(
                    "committed_nominal_tube_tracking_action_smooth_weight", 0.0
                )
            ),
        )
        self.committed_nominal_tube_tracking_heading_weight: float = max(
            0.0,
            float(
                explicit_cfg.get(
                    "committed_nominal_tube_tracking_heading_weight", 0.15
                )
            ),
        )
        self.committed_nominal_tube_tracking_min_predicted_improvement: float = max(
            0.0,
            float(
                explicit_cfg.get(
                    "committed_nominal_tube_tracking_min_predicted_improvement",
                    1e-4,
                )
            ),
        )
        self.committed_nominal_tube_tracking_max_negative_actual_steps: int = max(
            1,
            int(
                explicit_cfg.get(
                    "committed_nominal_tube_tracking_max_negative_actual_steps",
                    2,
                )
            ),
        )
        self.committed_nominal_tube_tracking_servo_mode: bool = bool(
            explicit_cfg.get("committed_nominal_tube_tracking_servo_mode", True)
        )
        self.committed_nominal_tube_tracking_servo_scale: float = max(
            0.0,
            float(explicit_cfg.get("committed_nominal_tube_tracking_servo_scale", 1.5)),
        )
        self.committed_nominal_tube_tracking_servo_boost_scale: float = max(
            self.committed_nominal_tube_tracking_servo_scale,
            float(
                explicit_cfg.get(
                    "committed_nominal_tube_tracking_servo_boost_scale",
                    2.0,
                )
            ),
        )
        self.committed_nominal_tube_tracking_window_heading_weight: float = max(
            0.0,
            float(
                explicit_cfg.get(
                    "committed_nominal_tube_tracking_window_heading_weight", 0.10
                )
            ),
        )
        self.extend_recovery_budget_on_progress = bool(
            explicit_cfg["extend_recovery_budget_on_progress"]
        )
        self.max_recover_steps_with_progress = max(
            0,
            int(explicit_cfg["max_recover_steps_with_progress"]),
        )
        self.recovery_budget_progress_epsilon = float(
            explicit_cfg["recovery_budget_progress_epsilon"]
        )
        self.recovery_budget_no_progress_limit = max(
            0,
            int(explicit_cfg["recovery_budget_no_progress_limit"]),
        )

        # Replan and corridor policy sections.
        replan_cfg: ConfigDict = self._safechunk_replan_config(safechunk_replan)
        self.safechunk_replan_enabled = bool(replan_cfg["enabled"])
        self.replan_deform_from_current_state = bool(
            replan_cfg["replan_deform_from_current_state"]
        )
        self.replan_recovery_from_current_state = bool(
            replan_cfg["replan_recovery_from_current_state"]
        )
        self.suppress_stale_return = bool(replan_cfg["suppress_stale_return"])
        self.max_recovery_failure_before_replan = int(
            replan_cfg["max_recovery_failure_before_replan"]
        )
        self.allow_recovery_to_nominal_only_if_feasible = bool(
            replan_cfg["allow_recovery_to_nominal_only_if_feasible"]
        )
        self.recovery_target_mode = str(replan_cfg["recovery_target_mode"])
        self.clear_failed_recovery_on_nominal_safe = bool(
            replan_cfg["clear_failed_recovery_on_nominal_safe"]
        )

        recover_cfg: ConfigDict = self._safechunk_recover_config(safechunk_recover)
        self.safechunk_recover_enabled = bool(recover_cfg["enabled"])
        self.recover_rejoin_nominal_weight = float(recover_cfg["rejoin_nominal_weight"])
        self.recover_task_progress_weight = float(recover_cfg["task_progress_weight"])
        self.recover_act_progress_weight = float(recover_cfg["act_progress_weight"])
        self.recover_act_heading_weight = float(recover_cfg["act_heading_weight"])
        self.recover_min_act_heading_cosine = float(
            recover_cfg["min_act_heading_cosine"]
        )
        self.recover_direction_alignment_weight = float(
            recover_cfg["direction_alignment_weight"]
        )
        self.recover_min_direction_cosine = float(recover_cfg["min_direction_cosine"])
        self.require_recover_direction_alignment = bool(
            recover_cfg["require_direction_alignment"]
        )
        self.recover_direction_alignment_margin = float(
            recover_cfg["direction_alignment_margin"]
        )
        self.recover_ordered_pose_weight: float = float(recover_cfg["ordered_pose_weight"])
        self.recover_ordered_delta_weight: float = float(recover_cfg["ordered_delta_weight"])
        self.recover_ordered_heading_weight: float = float(
            recover_cfg["ordered_heading_weight"]
        )
        self.recover_ordered_pose_threshold: float = float(
            recover_cfg["ordered_pose_threshold"]
        )
        self.recover_ordered_delta_threshold: float = float(
            recover_cfg["ordered_delta_threshold"]
        )
        self.recover_ordered_heading_cosine_threshold: float = float(
            recover_cfg["ordered_heading_cosine_threshold"]
        )
        self.recover_ordered_backtrack_tolerance: int = int(
            recover_cfg["ordered_backtrack_tolerance"]
        )
        self.require_recover_ordered_path = bool(recover_cfg["require_ordered_path"])
        self.recover_retry_cooldown_steps = max(
            0,
            int(recover_cfg["retry_cooldown_steps"]),
        )
        self.recover_max_attempts_per_unsafe_streak = max(
            0,
            int(recover_cfg["max_attempts_per_unsafe_streak"]),
        )
        self.recovery_attempt_reset_after_brake_timeout = bool(
            recover_cfg["reset_attempts_after_brake_timeout"]
        )
        self.recovery_attempt_reset_brake_timeout_steps = max(
            1,
            int(recover_cfg["reset_attempts_brake_timeout_steps"]),
        )
        raw_reset_clearance = recover_cfg["reset_attempts_min_hold_clearance"]
        self.recovery_attempt_reset_min_hold_clearance = (
            None if raw_reset_clearance is None else float(raw_reset_clearance)
        )
        self.recovery_attempt_reset_require_safe_hold = bool(
            recover_cfg["reset_attempts_require_safe_hold"]
        )
        self.recover_safety_weight = float(recover_cfg["safety_weight"])
        self.recover_clearance_penalty_scale: float = float(
            recover_cfg.get("clearance_penalty_scale", 5.0)
        )
        # Task-progress recovery is the delayed/bridge path back to a future ACT
        # window.  It must value clearance more strongly than direct rejoin,
        # otherwise a smooth bridge can cut through the human before landing on
        # an otherwise-safe waypoint.
        self.recover_task_progress_clearance_penalty_scale: float = float(
            recover_cfg.get(
                "task_progress_clearance_penalty_scale",
                self.recover_clearance_penalty_scale * 4.0,
            )
        )
        self.recover_action_deviation_weight = float(
            recover_cfg["action_deviation_weight"]
        )
        self.recover_smoothness_weight = float(recover_cfg["smoothness_weight"])
        self.recover_action_rate_limit = float(recover_cfg["action_rate_limit"])
        self.recover_action_rate_limit_weight = float(
            recover_cfg["action_rate_limit_weight"]
        )
        self.require_nominal_prefix_safe_for_rejoin = bool(
            recover_cfg["require_nominal_prefix_safe_for_rejoin"]
        )
        self.nominal_rejoin_prefix_min_clearance = float(
            recover_cfg["nominal_rejoin_prefix_min_clearance"]
        )
        self.recover_resume_tube_weight = max(
            0.0,
            float(recover_cfg.get("resume_tube_weight", 1.5)),
        )
        self.recover_resume_tube_min_score = float(
            recover_cfg.get("resume_tube_min_score", 0.6)
        )
        self.recover_resume_tube_min_component_score = float(
            recover_cfg.get("resume_tube_min_component_score", 0.35)
        )
        self.recover_resume_tube_distance_scale = max(
            1e-6,
            float(recover_cfg.get("resume_tube_distance_scale", 0.75)),
        )
        self.recover_resume_tube_min_clearance = float(
            recover_cfg.get(
                "resume_tube_min_clearance",
                self.nominal_rejoin_prefix_min_clearance,
            )
        )
        self.recover_resume_window_len: int = max(
            1,
            int(recover_cfg.get("resume_window_len", recover_cfg.get("act_frame_stack", 4))),
        )
        self.recover_resume_window_weight: float = max(
            0.0,
            float(recover_cfg.get("resume_window_weight", 1.0)),
        )
        self.recover_resume_window_dq_weight: float = max(
            0.0,
            float(recover_cfg.get("resume_window_dq_weight", 0.5)),
        )
        self.recover_resume_window_action_weight: float = max(
            0.0,
            float(recover_cfg.get("resume_window_action_weight", 0.25)),
        )
        self.recover_resume_window_max_q_frame_l2_mean: float = float(
            recover_cfg.get("resume_window_max_q_frame_l2_mean", 0.24)
        )
        self.recover_resume_window_max_q_frame_l2_max: float = float(
            recover_cfg.get("resume_window_max_q_frame_l2_max", 0.32)
        )
        self.recover_resume_window_min_dq_cosine: float = float(
            recover_cfg.get("resume_window_min_dq_cosine", 0.20)
        )
        self.recover_resume_window_max_step_l2_error: float = float(
            recover_cfg.get("resume_window_max_step_l2_error", 0.12)
        )
        self.recover_resume_affordance_enabled = bool(
            recover_cfg.get("resume_affordance_enabled", True)
        )
        self.recover_resume_affordance_weight = max(
            0.0,
            float(recover_cfg.get("resume_affordance_weight", 1.0)),
        )
        self.recover_resume_affordance_min_score = float(
            recover_cfg.get("resume_affordance_min_score", 0.45)
        )
        self.recover_resume_affordance_min_component_score = float(
            recover_cfg.get("resume_affordance_min_component_score", 0.25)
        )
        self.recover_resume_affordance_required_for_accept = bool(
            recover_cfg.get("resume_affordance_required_for_accept", True)
        )
        self.recover_resume_affordance_min_component_for_accept = float(
            recover_cfg.get(
                "resume_affordance_min_component_for_accept",
                self.recover_resume_affordance_min_component_score,
            )
        )
        self.recover_resume_affordance_target_distance_good = max(
            0.0,
            float(recover_cfg.get("resume_affordance_target_distance_good", 0.12)),
        )
        self.recover_resume_affordance_target_distance_scale = max(
            1e-6,
            float(recover_cfg.get("resume_affordance_target_distance_scale", 0.45)),
        )
        self.recover_resume_affordance_progress_scale = max(
            1e-6,
            float(recover_cfg.get("resume_affordance_progress_scale", 0.10)),
        )
        self.recover_resume_affordance_progress_epsilon = max(
            1e-9,
            float(recover_cfg.get("resume_affordance_progress_epsilon", 0.005)),
        )
        self.recover_resume_affordance_progress_distance_gain = max(
            0.0,
            float(recover_cfg.get("resume_affordance_progress_distance_gain", 1.0)),
        )
        self.recover_resume_affordance_taskspace_in_optimizer = bool(
            recover_cfg.get("resume_affordance_taskspace_in_optimizer", False)
        )
        self.recover_resume_affordance_terminal_distance_weight = max(
            0.0,
            float(recover_cfg.get("resume_affordance_terminal_distance_weight", 8.0)),
        )
        self.recover_act_frame_stack: int = max(
            1,
            int(recover_cfg["act_frame_stack"]),
        )
        # A full ACT rejoin requires one policy history window.  Staging is an
        # internal fallback for partial safe windows: if ACT uses more than two
        # frames, two consecutive safe points are the shortest segment that still
        # defines an ACT-like direction.  A single point is ignored because its
        # heading would be invented at the beginning/end of a window.
        self.recover_safe_rejoin_window_len: int = self.recover_act_frame_stack
        self.recover_staging_rejoin_window_min_len: int = (
            2 if self.recover_act_frame_stack > 2 else self.recover_act_frame_stack
        )
        self.use_latest_nominal_for_rejoin = bool(
            recover_cfg["use_latest_nominal_for_rejoin"]
        )
        self.suppress_stale_nominal_rejoin = bool(
            recover_cfg["suppress_stale_nominal_rejoin"]
        )
        self.rejoin_weight_schedule = str(recover_cfg["rejoin_weight_schedule"]).lower()
        self.rejoin_ramp_steps = max(1, int(recover_cfg["rejoin_ramp_steps"]))

        corridor_cfg: ConfigDict = self._safechunk_recovery_corridor_config(
            safechunk_recovery_corridor
        )
        self.safechunk_recovery_corridor_enabled = bool(corridor_cfg["enabled"])
        self.require_recover_path_safe = bool(corridor_cfg["require_recover_path_safe"])
        # Post-escape recovery must satisfy the same clearance required for
        # execution.  The corridor config is kept for backward-compatible YAML
        # parsing, but it no longer weakens this final recovery-return gate.
        self.recover_execution_min_clearance: float = float(
            self._acceptance_clearance_threshold()
        )
        self.recover_path_min_clearance: float = self.recover_execution_min_clearance
        self.recover_immediate_hard_clearance: float = self.recover_execution_min_clearance
        self.recover_prefix_min_clearance: float = self.recover_execution_min_clearance
        self.enable_direct_rejoin = bool(corridor_cfg["enable_direct_rejoin"])
        self.enable_recovery_bridge_seeds = bool(
            corridor_cfg["enable_recovery_bridge_seeds"]
        )
        # Legacy detour mode is folded into recovery seed generation. Keep the
        # old flag false so recovery has one final accept/reject path.
        self.enable_detour_rejoin = False
        self.enable_delayed_rejoin = bool(corridor_cfg["enable_delayed_rejoin"])
        self.suppress_repeated_unsafe_recovery = bool(
            corridor_cfg["suppress_repeated_unsafe_recovery"]
        )
        self.unsafe_recovery_cooldown_steps = max(
            0,
            int(corridor_cfg["unsafe_recovery_cooldown_steps"]),
        )
        self.max_same_target_failures = max(
            1,
            int(corridor_cfg["max_same_target_failures"]),
        )
        self.bridge_seed_scales: tuple[float, ...] = tuple(
            float(x) for x in corridor_cfg["bridge_seed_scales"]
        )
        self.detour_scales: tuple[float, ...] = self.bridge_seed_scales
        self.detour_clearance_weight = float(corridor_cfg["detour_clearance_weight"])
        self.detour_task_rejoin_weight = float(corridor_cfg["detour_task_rejoin_weight"])
        self.detour_action_norm_weight = float(corridor_cfg["detour_action_norm_weight"])
        self.delayed_rejoin_wait_steps = max(
            0,
            int(corridor_cfg["delayed_rejoin_wait_steps"]),
        )
        self.delayed_rejoin_requires_nominal_prefix_safe = bool(
            corridor_cfg["delayed_rejoin_requires_nominal_prefix_safe"]
        )
        self.require_safe_corridor_for_recovery_complete = bool(
            corridor_cfg["require_safe_corridor_for_recovery_complete"]
        )
        self.require_post_recovery_act_window = bool(
            corridor_cfg["require_post_recovery_act_window"]
        )
        self.post_recovery_min_act_steps = max(
            0,
            int(corridor_cfg["post_recovery_min_act_steps"]),
        )


    def _init_execution_state(self) -> None:
        """Reset mutable recovery state and all bookkeeping counters."""
        # Transient recovery plan and target/cooldown accounting.
        self.recovery_context: RecoveryContext | None = None
        self.current_deform_plan: np.ndarray | None = None
        self.current_recovery_plan: np.ndarray | None = None
        self.deform_anchor_state: np.ndarray | None = None
        self.recovery_anchor_state: np.ndarray | None = None
        self._trigger_count: int = 0
        self.deform_replan_count: int = 0
        self.recovery_replan_count: int = 0
        self.stale_recovery_suppressed_count: int = 0
        self.recovery_target_infeasible_count: int = 0
        self.emergency_brake_steps: int = 0
        self.recovery_failure_streak: int = 0
        self.recovery_failure_streak_max: int = 0
        self.recovery_optimizer_cooldown_remaining: int = 0
        self.recovery_attempts_in_unsafe_streak: int = 0
        self.recovery_optimization_skipped_count: int = 0
        self.recovery_attempt_reset_count: int = 0
        self.recovery_attempt_reset_last_brake_streak: int = 0
        self.recovery_attempt_reset_last_hold_clearance: float | None = None
        self.recovery_attempt_reset_last_reason: str | None = None
        self.failed_recovery_targets: list[Any] = []
        self.failed_recovery_paths: list[Any] = []
        self.recovery_path_failure_streak: int = 0
        self.recovery_path_failure_streak_max: int = 0
        self.recovery_target_failure_counts: dict[Any, int] = {}
        self._unsafe_recovery_cooldowns: dict[Any, int] = {}
        self.delayed_rejoin_active: bool = False
        self.delayed_rejoin_steps: int = 0

        # Diagnostic counters reported to the outer safety filter.
        self.safe_corridor_recovery_count: int = 0
        self.direct_rejoin_attempt_count: int = 0
        self.direct_rejoin_reject_count: int = 0
        self.detour_rejoin_attempt_count: int = 0
        self.detour_rejoin_accept_count: int = 0
        self.recovery_bridge_seed_total_count: int = 0
        self.delayed_rejoin_count: int = 0
        self.delayed_rejoin_suppressed_count: int = 0
        self.recover_path_unsafe_count: int = 0
        self.contact_during_recover_count: int = 0
        self.post_recovery_act_window_count: int = 0
        self.repeated_unsafe_target_count: int = 0
        self.post_recovery_act_window_active: bool = False
        self.post_recovery_act_steps_remaining: int = 0
        self.post_recovery_act_window_interrupted_count: int = 0
        self._recover_path_min_clearance_history: list[float] = []
        self.latest_nominal_chunk: np.ndarray | None = None
        self.latest_nominal_step: int = 0
        self.blocked_nominal_chunk: np.ndarray | None = None
        self.blocked_nominal_step: int | None = None
        self.recover_step_since_deform: int = 0
        self.nominal_rejoin_available_count: int = 0
        self.nominal_rejoin_suppressed_count: int = 0
        self.stale_nominal_rejoin_suppressed_count: int = 0
        self.nominal_prefix_unsafe_suppressed_count: int = 0
        self.recover_positive_projection_count: int = 0
        self.recover_nonpositive_projection_count: int = 0
        self._recover_projection_history: list[float] = []
        self._recover_cosine_history: list[float] = []
        self._recover_task_progress_history: list[float] = []
        self._recover_ordered_pose_loss_history: list[float] = []
        self._recover_ordered_delta_loss_history: list[float] = []
        self._recover_ordered_heading_loss_history: list[float] = []
        self._recover_ordered_loss_history: list[float] = []

        # Committed explicit-recovery state served over multiple control ticks.
        self.committed_chunk: np.ndarray | None = None
        self.committed_chunk_index: int = 0
        self.committed_chunk_mode: str | None = None
        self.committed_chunk_modes: list[str] = []
        self.committed_sequence_id: int = 0
        self.committed_rejoin_index: int | None = None
        self.committed_until_complete: bool = False
        self.committed_planned_q_seq: np.ndarray | None = None
        self.committed_planned_post_q_seq: np.ndarray | None = None
        self.committed_planned_h_seq: np.ndarray | None = None
        self.committed_planned_min_clearance_seq: np.ndarray | None = None
        self.committed_planned_clearance_pre_seq: np.ndarray | None = None
        self.committed_planned_clearance_post_seq: np.ndarray | None = None
        self.committed_planned_actions: np.ndarray | None = None
        self.committed_accepted_min_clearance: float | None = None
        self.committed_accepted_clearance_margin: float | None = None
        self.committed_accepted_human_state_snapshot: Any | None = None
        self.committed_planning_human_state_snapshot: Any | None = None
        self.committed_planning_obs: Any | None = None
        self.committed_rejoin_diagnostics: InfoDict = {}
        self.committed_closed_loop_tracking_count: int = 0
        self._pending_committed_replan_info: InfoDict | None = None
        self.committed_suffix_replan_attempt_count: int = 0
        self.committed_suffix_replan_accepted_count: int = 0
        self.committed_suffix_replan_rejected_count: int = 0
        self.committed_suffix_replan_budget_suppressed_count: int = 0
        self.committed_opportunistic_resume_count: int = 0
        self.committed_recovery_budget_exit_count: int = 0
        self.committed_recover_steps_since_act: int = 0
        self.committed_suffix_replans_in_current_recovery: int = 0
        self.recovery_budget_extended_count: int = 0
        self.recovery_budget_no_progress_count: int = 0
        self.mpc_recovery_replan_count: int = 0
        self.mpc_recovery_accepted_count: int = 0
        self.mpc_recovery_rejected_count: int = 0
        self.mpc_recovery_no_progress_reject_count: int = 0
        self.mpc_recovery_budget_escape_count: int = 0
        self.mpc_recovery_replans_in_current_recovery: int = 0
        self.mpc_bridge_replans_in_current_recovery: int = 0
        self.mpc_bridge_replan_cooldown_remaining: int = 0
        self.mpc_bridge_context_id: int | None = None
        self.mpc_bridge_last_heading_cosine: float | None = None
        self.mpc_bridge_last_progress_projection: float | None = None
        self.mpc_bridge_last_prefix_clearance: float | None = None
        self.mpc_bridge_last_improved: bool = True
        self.resume_affordance_context: InfoDict = {}
        if hasattr(self, "mpc"):
            self.mpc.reset()

    def reset_execution_state(self) -> None:
        """Reset recovery state and restart the optimizer stream for this episode."""
        self._init_execution_state()
        self._rng = np.random.default_rng(self.opt_seed)

    def set_resume_affordance_context(self, context: Mapping[str, Any] | None) -> None:
        """Set task-adapter resume features for the current control step.

        The recovery logic consumes generic affordance fields only.  Task-specific
        code should translate its own semantics into names such as
        ``interaction_context``, ``resume_target_distance``,
        ``resume_target_contact``, and ``resume_task_progress`` before calling
        this method.
        """
        if context is None:
            self.resume_affordance_context = {}
            return
        if hasattr(context, "items"):
            self.resume_affordance_context = dict(context.items())
        else:
            self.resume_affordance_context = dict(context)


    def _candidate_resume_affordance_features(
        self,
        q_seq: Any,
        *,
        obs: Any | None = None,
        source: str = "candidate_ee_pose",
    ) -> InfoDict:
        """Convert a candidate q rollout into generic task-space resume features.

        The resume affordance scorer is intentionally generic.  This helper keeps
        it generic while making the MPC objective actionable: if the eval/task
        adapter provides a target position or target direction, candidate q states
        are mapped to an EE pose and scored in task space instead of by q-space
        progress alone.  When absolute EE/target frames look inconsistent, it
        falls back to conservative EE-motion projection toward the target.
        """

        context = getattr(self, "resume_affordance_context", {}) or {}
        if not hasattr(context, "get"):
            return {}

        def _finite(value, default=None):
            try:
                value_f = float(value)
            except Exception:  # noqa: BLE001
                return default
            return value_f if np.isfinite(value_f) else default

        def _finite_vec(value) -> np.ndarray | None:
            if value is None:
                return None
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:  # noqa: BLE001
                return None
            if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
                return None
            return arr

        q_arr = np.asarray(q_seq, dtype=np.float32)
        if q_arr.size == 0:
            return {}
        q_arr = q_arr.reshape(1, -1) if q_arr.ndim == 1 else q_arr.reshape(q_arr.shape[0], -1)
        if q_arr.shape[0] == 0:
            return {}

        ee_input = q_arr
        prepend_current = False
        if obs is not None:
            try:
                current_q = self._current_replay_q(obs)
                if current_q.shape[0] == q_arr.shape[1]:
                    ee_input = np.vstack([current_q.reshape(1, -1), q_arr[-1:].reshape(1, -1)])
                    prepend_current = True
            except Exception:  # noqa: BLE001
                prepend_current = False
                ee_input = q_arr

        try:
            ee_seq = self._ee_pose_sequence(ee_input)
        except Exception as exc:  # pragma: no cover - optional integration path
            logger.debug("resume affordance EE pose failed: %s", exc)
            ee_seq = None
        if ee_seq is None:
            return {}
        ee_seq = np.asarray(ee_seq, dtype=np.float32)
        if ee_seq.size == 0 or ee_seq.shape[0] == 0:
            return {}
        ee_seq = ee_seq.reshape(ee_seq.shape[0], -1)
        terminal_ee = ee_seq[-1]
        model_current_ee = ee_seq[0] if prepend_current and ee_seq.shape[0] >= 2 else None

        context_current_ee = _finite_vec(context.get("resume_current_ee_position"))
        target_pos = _finite_vec(context.get("resume_target_position"))
        target_vec = _finite_vec(context.get("resume_target_vector"))
        current_distance = _finite(context.get("resume_target_distance"), None)
        distance_good = float(
            getattr(self, "recover_resume_affordance_target_distance_good", 0.12)
        )
        distance_scale = float(
            getattr(self, "recover_resume_affordance_target_distance_scale", 0.45)
        )

        features: InfoDict = {
            "resume_taskspace_affordance_source": source,
            "resume_taskspace_affordance_available": True,
        }
        # Only model-current and model-terminal EE are guaranteed to share a
        # coordinate frame.  Do not mix a terminal JAX EE pose with a Bigym-world
        # current EE pose; if no model-current pose is available, fall back to
        # context-only affordance terms instead of inventing an absolute distance.
        current_ee_for_motion = model_current_ee
        candidate_distance = None
        current_model_distance = None
        frame_error = None

        if target_pos is not None:
            dims = min(3, terminal_ee.size, target_pos.size)
            if dims > 0:
                candidate_direct = float(np.linalg.norm(terminal_ee[:dims] - target_pos[:dims]))
                if current_ee_for_motion is not None and current_ee_for_motion.size >= dims:
                    current_model_distance = float(
                        np.linalg.norm(current_ee_for_motion[:dims] - target_pos[:dims])
                    )
                    if current_distance is not None:
                        frame_error = abs(current_model_distance - current_distance)
                frame_tol = max(0.25, 2.0 * distance_good, 0.5 * distance_scale)
                if model_current_ee is not None and (
                    frame_error is None or frame_error <= frame_tol
                ):
                    candidate_distance = candidate_direct
                    features["resume_target_distance_source"] = f"{source}.ee_to_target_position"
                features["resume_candidate_target_distance_direct"] = float(candidate_direct)
                if current_model_distance is not None:
                    features["resume_current_target_distance_model"] = float(current_model_distance)
                if frame_error is not None:
                    features["resume_target_distance_frame_error"] = float(frame_error)

        if candidate_distance is None and current_distance is not None:
            desired_vec = target_vec
            if desired_vec is None and target_pos is not None and current_ee_for_motion is not None:
                dims = min(3, target_pos.size, current_ee_for_motion.size)
                if dims > 0:
                    desired_vec = target_pos[:dims] - current_ee_for_motion[:dims]
            if desired_vec is not None and current_ee_for_motion is not None:
                dims = min(3, terminal_ee.size, current_ee_for_motion.size, desired_vec.size)
                if dims > 0:
                    desired = desired_vec[:dims]
                    desired_norm = float(np.linalg.norm(desired))
                    motion = terminal_ee[:dims] - current_ee_for_motion[:dims]
                    motion_norm = float(np.linalg.norm(motion))
                    if desired_norm > 1e-8:
                        projection = float(np.dot(motion, desired / desired_norm))
                        candidate_distance = float(max(0.0, current_distance - projection))
                        features["resume_target_distance_source"] = f"{source}.ee_motion_projection"
                        features["resume_candidate_target_projection"] = float(projection)
                        features["resume_candidate_ee_displacement"] = float(motion_norm)
                        if motion_norm > 1e-8:
                            cosine = float(
                                np.dot(motion, desired)
                                / max(1e-8, motion_norm * desired_norm)
                            )
                            cosine = float(min(1.0, max(-1.0, cosine)))
                            features["resume_alignment_cosine"] = cosine
                            features["resume_alignment_score"] = float(
                                min(1.0, max(0.0, 0.5 * (cosine + 1.0)))
                            )

        if candidate_distance is None:
            return {}

        reference_distance = current_distance
        if reference_distance is None:
            reference_distance = current_model_distance
        features["resume_target_distance"] = float(candidate_distance)
        features["resume_candidate_target_distance"] = float(candidate_distance)
        if reference_distance is not None:
            distance_delta = float(reference_distance - candidate_distance)
            features["resume_current_target_distance"] = float(reference_distance)
            features["resume_taskspace_distance_delta"] = distance_delta
            features["resume_task_progress_delta"] = distance_delta
        return features

    def current_resume_affordance_info(self) -> InfoDict:
        """Return current-step affordance diagnostics without a candidate path."""
        return self._resume_affordance_score_terms({})

    def _current_brake_streak_for_attempt_reset(self) -> int:
        """Return the active brake streak across current/legacy owners."""
        holders = (getattr(self.parent, "brake", None), self.parent, self)
        for holder in holders:
            if holder is None or not hasattr(holder, "brake_streak"):
                continue
            try:
                return max(0, int(getattr(holder, "brake_streak")))
            except Exception:  # noqa: BLE001
                continue
        return 0

    def _last_hold_horizon_clearance_for_attempt_reset(self) -> float | None:
        """Return the most recent hold/brake horizon clearance, if available."""
        holders = (getattr(self.parent, "brake", None), self.parent, self)
        for holder in holders:
            if holder is None:
                continue
            history = getattr(holder, "_hold_horizon_min_clearance_history", None)
            if not history:
                continue
            try:
                value = float(history[-1])
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(value):
                return value
        return None

    def _attempt_reset_clearance_threshold(self) -> float:
        """Clearance required before a brake timeout may reopen recovery attempts."""
        configured = self.recovery_attempt_reset_min_hold_clearance
        if configured is not None:
            return float(configured)
        try:
            return float(self._acceptance_clearance_threshold())
        except Exception:  # noqa: BLE001
            return float(getattr(self, "min_clearance", 0.0))

    def _maybe_reset_recovery_attempts_after_brake_timeout(self) -> InfoDict:
        """Let braking serve as a settle period, then reopen recovery attempts."""
        if not self.recovery_attempt_reset_after_brake_timeout:
            return {}

        previous_attempts = max(0, int(self.recovery_attempts_in_unsafe_streak))
        previous_cooldown = max(0, int(self.recovery_optimizer_cooldown_remaining))
        if previous_attempts <= 0 and previous_cooldown <= 0:
            return {}

        brake_streak = self._current_brake_streak_for_attempt_reset()
        steps_since_reset = brake_streak - int(self.recovery_attempt_reset_last_brake_streak)
        if steps_since_reset < int(self.recovery_attempt_reset_brake_timeout_steps):
            return {}

        hold_clearance = self._last_hold_horizon_clearance_for_attempt_reset()
        min_hold_clearance = self._attempt_reset_clearance_threshold()
        if self.recovery_attempt_reset_require_safe_hold:
            if hold_clearance is None or hold_clearance < min_hold_clearance:
                return {}

        self.recovery_optimizer_cooldown_remaining = 0
        self.recovery_attempts_in_unsafe_streak = 0
        self.recovery_failure_streak = 0
        self.recovery_attempt_reset_count += 1
        self.recovery_attempt_reset_last_brake_streak = int(brake_streak)
        self.recovery_attempt_reset_last_hold_clearance = hold_clearance
        self.recovery_attempt_reset_last_reason = "brake_timeout"
        return {
            "recovery_attempt_reset_after_brake_timeout": True,
            "recovery_attempt_reset_reason": "brake_timeout",
            "recovery_attempt_reset_count": int(self.recovery_attempt_reset_count),
            "recovery_attempt_reset_brake_streak": int(brake_streak),
            "recovery_attempt_reset_steps_since_previous": int(steps_since_reset),
            "recovery_attempt_reset_previous_attempts": int(previous_attempts),
            "recovery_attempt_reset_previous_cooldown": int(previous_cooldown),
            "recovery_attempt_reset_hold_clearance": hold_clearance,
            "recovery_attempt_reset_min_hold_clearance": float(min_hold_clearance),
            "recovery_attempt_reset_require_safe_hold": bool(
                self.recovery_attempt_reset_require_safe_hold
            ),
        }

    def _deprecated_config_value(
        self,
        cfg: Mapping[str, Any],
        old_key: str,
        new_key: str,
        default: Any,
    ) -> Any:
        """Read a deprecated key while preserving backward compatibility behavior."""
        if new_key in cfg:
            return cfg[new_key]
        if old_key in cfg:
            logger.warning(
                "Deprecated SafeChunk-Deform config key '%s'; use '%s' instead.",
                old_key,
                new_key,
            )
            return cfg[old_key]
        return default

    def Perform(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: Any,
        info: InfoDict,
        q_seq: np.ndarray | None = None,
        safety_info: Mapping[str, Any] | None = None,
        waited_unsafe_streak: int = 0,
        waited_brake_streak: int = 0,
        mode: str = "temporary_wait",
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Dispatch recovery-only execution modes requested by the parent filter."""
        if mode == "post_recovery_act_window":
            return self._perform_post_recovery_act_window(
                obs,
                chunk,
                original_shape,
                info=info,
                **kwargs,
            )
        if mode == "temporary_wait":
            return self._perform_recover_after_temporary_wait(
                obs,
                chunk,
                q_seq=q_seq,
                safety_info=safety_info,
                info=info,
                original_shape=original_shape,
                waited_unsafe_streak=waited_unsafe_streak,
                waited_brake_streak=waited_brake_streak,
                **kwargs,
            )
        return None

    def deform_chunk_optimized_explicit_return(
        self,
        nominal_chunk: np.ndarray,
        obs: Any,
        first_violation: int | None = None,
        nominal_q_seq: np.ndarray | None = None,
        nominal_ee_seq: np.ndarray | None = None,
        human_state: Any | None = None,
        safety_info: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> RecoveryResult:
        """Plan a safe deform prefix followed by an explicit recovery suffix."""
        del human_state
        chunk, _ = self._as_chunk(nominal_chunk)
        if nominal_q_seq is None:
            nominal_q_seq = self.rollout_nominal_chunk(obs, chunk)
        else:
            nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        self.deform_replan_count += 1
        self.deform_anchor_state = self.extract_current_q(obs, chunk).copy()
        self.current_deform_plan = np.asarray(chunk, dtype=np.float32).copy()
        context = self._create_recovery_context(
            chunk,
            nominal_q_seq,
            nominal_ee_seq=nominal_ee_seq,
            start_chunk_index=kwargs.get(
                "act_chunk_index",
                kwargs.get("chunk_index", kwargs.get("current_act_chunk_index")),
            ),
            observation_history=kwargs.get("observation_history"),
            policy_buffer_metadata=kwargs.get("policy_buffer_metadata"),
        )
        if safety_info is None:
            safety_info = self.evaluate_horizon_safety(obs, nominal_q_seq)
        if first_violation is None:
            first_violation = safety_info.get("first_violation")

        valid: np.ndarray = self._valid_control_indices(chunk)
        if not np.any(valid):
            return chunk.copy(), self._explicit_return_info(
                obs=obs,
                chunk=chunk,
                nominal=chunk,
                context=context,
                recovery_phase="horizon_deform",
                deform_chunk=chunk,
                deform_losses={},
                deform_eval=self.evaluate_horizon_safety(obs, nominal_q_seq),
                deform_stage_accepted=False,
                return_chunk=None,
                return_losses={},
                return_eval=None,
                return_accepted=False,
                return_rejoin_loss=float("inf"),
                return_target_index=None,
                fallback_used=True,
                rejection_cause="unsafe_and_unrecoverable",
            )

        action_idx: np.ndarray = self.controlled_action_indices[valid]
        deform_len: int = min(chunk.shape[0], self.deform_horizon)
        deform_nominal: np.ndarray = chunk[:deform_len].copy()
        deform_seed_chunks = [candidate[:deform_len].copy() for _, candidate in self._make_chunk_deformation_candidates(obs, chunk, safety_info or {})]

        def deform_cost(candidate: np.ndarray) -> CostResult:
            return self._deform_stage_deformation_cost(obs, candidate, deform_nominal, action_idx)

        def deform_early_stop(record: Mapping[str, Any]) -> bool:
            losses = record.get("losses", {})
            return float(losses.get("min_clearance", float("-inf"))) >= self._acceptance_clearance_threshold()

        deform_record = self._optimize_controlled_chunk(
            obs,
            deform_nominal,
            action_idx,
            deform_cost,
            seed_chunks=deform_seed_chunks,
            batch_cost_fn=lambda candidates: self._deform_stage_deformation_cost_batch(
                obs,
                candidates,
                deform_nominal,
                action_idx,
            ),
            early_stop_fn=deform_early_stop,
            optimizer_stage="deform",
        )
        deform_chunk = deform_record["chunk"]
        deform_q_seq = self.rollout_nominal_chunk(obs, deform_chunk)
        deform_eval = self.evaluate_horizon_safety(obs, deform_q_seq)
        deform_min_clearance_stage = float(deform_eval.get("min_clearance", float("-inf")))
        deform_stage_accepted = bool(
            deform_min_clearance_stage >= self._acceptance_clearance_threshold()
        )
        if not deform_stage_accepted:
            context.phase = "horizon_deform"
            return chunk.copy(), self._explicit_return_info(
                obs=obs,
                chunk=chunk,
                nominal=chunk,
                context=context,
                recovery_phase="horizon_deform",
                deform_chunk=deform_chunk,
                deform_losses=deform_record["losses"],
                deform_eval=deform_eval,
                deform_stage_accepted=False,
                return_chunk=None,
                return_losses={},
                return_eval=None,
                return_accepted=False,
                return_rejoin_loss=float("inf"),
                return_target_index=None,
                fallback_used=True,
                rejection_cause="unsafe",
            )

        context.phase = "recover"
        self.recover_step_since_deform = max(1, int(self.recover_step_since_deform) + 1)
        self.recovery_replan_count += 1
        self.recovery_anchor_state = np.asarray(deform_q_seq[-1], dtype=np.float32).copy()
        return_obs = self._obs_with_q(obs, deform_q_seq[-1])
        stale_recovery_attempted = bool(self.safechunk_replan_enabled)
        return_to_old_path_suppressed = False
        recover_to_task_progress = bool(
            self.safechunk_replan_enabled
            and self.recovery_target_mode == "task_progress"
        )
        nominal_return, seed_target_index = self._make_return_seed_chunk(
            context,
            deform_q_seq[-1],
            chunk,
            action_idx,
        )
        nominal_recover_feasible, _nominal_recover_eval, nominal_seed_immediate_safe = (
            self._recover_seed_feasible(return_obs, nominal_return)
        )
        recovery_target_feasible = bool(nominal_recover_feasible)
        if (
            self.safechunk_replan_enabled
            and self.suppress_stale_return
            and self.allow_recovery_to_nominal_only_if_feasible
            and not nominal_recover_feasible
        ):
            return_to_old_path_suppressed = True
            recover_to_task_progress = True
            self.stale_recovery_suppressed_count += 1
            self.recovery_target_infeasible_count += 1
            self.failed_recovery_targets.append(seed_target_index)
        if recover_to_task_progress:
            return_nominal, seed_target_index = self._make_task_progress_recover_chunk(
                return_obs,
                deform_q_seq[-1],
                chunk,
                action_idx,
                context=context,
                default_target_index=seed_target_index,
            )
        else:
            return_nominal = nominal_return
        task_recover_feasible, _task_recover_eval, seed_immediate_safe = self._recover_seed_feasible(
            return_obs,
            return_nominal,
        )
        recovery_target_feasible = bool(task_recover_feasible)
        if not seed_immediate_safe:
            self.emergency_brake_steps += 1
        self.current_recovery_plan = np.asarray(return_nominal, dtype=np.float32).copy()
        return_rejoin_context = self._make_rejoin_context(
            context.nominal_q_seq,
            context.nominal_ee_seq,
        )

        seed_min_clearance_floor = float(
            _task_recover_eval.get("min_clearance", float("-inf"))
        )
        seed_required_clearance = float(self._acceptance_clearance_threshold())
        recovery_seed_chunks: list[np.ndarray] = [
            np.asarray(return_nominal, dtype=np.float32).copy()
        ]
        best_seed_chunk = recovery_seed_chunks[0].copy()
        best_seed_name = "direct"
        recovery_bridge_seed_chunks: list[np.ndarray] = []
        recovery_bridge_seed_names: list[str] = []
        if self.enable_recovery_bridge_seeds:
            for seed_name, seed_chunk in self._make_recovery_bridge_seed_candidates(
                return_obs,
                return_nominal,
                action_idx,
            ):
                seed_arr = np.asarray(seed_chunk, dtype=np.float32).copy()
                if seed_arr.shape != np.asarray(return_nominal).shape:
                    continue
                recovery_bridge_seed_chunks.append(seed_arr)
                recovery_bridge_seed_names.append(str(seed_name))
                recovery_seed_chunks.append(seed_arr)
                _seed_feasible, seed_eval, _seed_immediate_safe = self._recover_seed_feasible(
                    return_obs,
                    seed_arr,
                )
                del _seed_feasible, _seed_immediate_safe
                if seed_eval is not None:
                    seed_clearance = float(seed_eval.get("min_clearance", float("-inf")))
                    if seed_clearance > seed_min_clearance_floor:
                        seed_min_clearance_floor = seed_clearance
                        best_seed_chunk = seed_arr.copy()
                        best_seed_name = str(seed_name)
        self.recovery_bridge_seed_total_count += len(recovery_bridge_seed_chunks)
        seed_path_safe = bool(seed_min_clearance_floor >= seed_required_clearance)

        strict_resume_context_objective = bool(
            float(getattr(self, "recover_resume_window_weight", 0.0)) > 0.0
            or float(getattr(self, "recover_resume_window_dq_weight", 0.0)) > 0.0
            or float(getattr(self, "recover_resume_window_action_weight", 0.0)) > 0.0
            or float(getattr(self, "recover_resume_affordance_weight", 0.0)) > 0.0
            or bool(getattr(self, "recover_resume_affordance_required_for_accept", False))
        )

        def _direct_resume_context_terms(candidate: np.ndarray) -> tuple[float, InfoDict]:
            # Reuse the task-progress context scorer for ordinary direct recovery,
            # but only add its ACT-resume-specific terms to the old return loss.
            # This keeps the original q-rejoin objective while preventing a
            # recovery from being accepted with an OOD last-4 observation window.
            _, resume_losses = self._recover_task_progress_cost(
                return_obs,
                candidate,
                return_nominal,
                action_idx,
                reference_chunk=return_nominal,
            )
            resume_losses = dict(resume_losses)
            resume_extra_loss = float(
                resume_losses.get("recover_resume_window_total_loss", 0.0)
                + resume_losses.get("recover_resume_affordance_loss", 0.0)
                + resume_losses.get("recover_resume_affordance_terminal_distance_loss", 0.0)
            )
            return resume_extra_loss, resume_losses

        def return_cost(candidate: np.ndarray) -> CostResult:
            if recover_to_task_progress:
                return self._recover_task_progress_cost(
                    return_obs,
                    candidate,
                    return_nominal,
                    action_idx,
                    reference_chunk=return_nominal,
                )
            base_cost, base_losses = self._return_deformation_cost(
                return_obs,
                candidate,
                return_nominal,
                context.nominal_q_seq,
                return_rejoin_context,
                action_idx,
            )
            base_losses = dict(base_losses)
            if strict_resume_context_objective:
                resume_extra_loss, resume_losses = _direct_resume_context_terms(candidate)
                base_losses.update(resume_losses)
                base_losses["explicit_recovery_resume_context_extra_loss"] = float(
                    resume_extra_loss
                )
                total_loss = float(base_cost + resume_extra_loss)
                base_losses["total_loss"] = total_loss
                base_losses["existing_optimization_loss"] = float(
                    base_losses.get("existing_optimization_loss", base_cost)
                    + resume_extra_loss
                )
                return total_loss, base_losses
            return base_cost, base_losses

        def return_early_stop(record: Mapping[str, Any]) -> bool:
            losses = record.get("losses", {})
            min_clearance = float(losses.get("min_clearance", float("-inf")))
            if min_clearance < self._acceptance_clearance_threshold():
                return False
            if (
                np.isfinite(seed_min_clearance_floor)
                and min_clearance + 1e-6 < seed_min_clearance_floor
            ):
                return False
            if recover_to_task_progress:
                if self.require_recover_ordered_path and not bool(
                    losses.get(
                        "recover_ordered_waypoint_tube_ok",
                        losses.get("recover_ordered_ok", False),
                    )
                ):
                    return False
                if not bool(losses.get("recover_act_progress_ok", True)):
                    return False
                if not bool(losses.get("recover_act_heading_ok", True)):
                    return False
                return float(losses.get("recover_task_progress_score", 0.0)) > 0.0
            rejoin_loss = float(losses.get("rejoin_loss", losses.get("return_rejoin_loss", float("inf"))))
            if self._sqrt_loss(rejoin_loss) >= self.q_rejoin_threshold:
                return False
            if strict_resume_context_objective:
                gate_info = {**losses, "q_rejoin_ok": True}
                gate_path = {
                    "immediate_safe": True,
                    "prefix_safe": True,
                    "path_safe": True,
                }
                return self._recovery_reject_reason(
                    gate_info,
                    gate_path,
                    task_progress_ok=True,
                    direction_ok=True,
                    ordered_ok=True,
                ) is None
            return True

        return_record = self._optimize_controlled_chunk(
            return_obs,
            return_nominal,
            action_idx,
            return_cost,
            seed_chunks=recovery_seed_chunks,
            batch_cost_fn=(
                (lambda candidates: self._recover_task_progress_cost_batch(
                    return_obs,
                    candidates,
                    return_nominal,
                    action_idx,
                    reference_chunk=return_nominal,
                ))
                if recover_to_task_progress
                else (
                    None
                    if strict_resume_context_objective
                    else (lambda candidates: self._return_deformation_cost_batch(
                        return_obs,
                        candidates,
                        return_nominal,
                        context.nominal_q_seq,
                        return_rejoin_context,
                        action_idx,
                    ))
                )
            ),
            early_stop_fn=return_early_stop,
            optimizer_stage="return",
        )
        optimized_return_losses = dict(return_record.get("losses", {}))
        optimized_min_clearance = float(
            optimized_return_losses.get("min_clearance", float("-inf"))
        )
        if (
            np.isfinite(seed_min_clearance_floor)
            and optimized_min_clearance + 1e-6 < seed_min_clearance_floor
        ):
            seed_cost, seed_losses = return_cost(best_seed_chunk)
            preserved_optimizer_losses = {
                key: value
                for key, value in optimized_return_losses.items()
                if key == "optimizer_method"
                or key.startswith((
                    "gradient_",
                    "cem_",
                    "return_optimizer_",
                    "optimizer_",
                ))
            }
            seed_losses.update(preserved_optimizer_losses)
            seed_losses.update(
                {
                    "recovery_seed_clearance_floor": float(seed_min_clearance_floor),
                    "recovery_seed_required_clearance": float(seed_required_clearance),
                    "recovery_seed_min_clearance": float(seed_min_clearance_floor),
                    "recovery_seed_path_safe": bool(seed_path_safe),
                    "recovery_seed_path_unsafe": not bool(seed_path_safe),
                    "recovery_optimized_min_clearance_before_seed_guard": float(
                        optimized_min_clearance
                    ),
                    "recovery_seed_clearance_guard_applied": True,
                    "recovery_seed_clearance_guard_source": str(best_seed_name),
                }
            )
            return_record = {
                "cost": float(seed_cost),
                "chunk": np.asarray(best_seed_chunk, dtype=np.float32).copy(),
                "losses": seed_losses,
            }
        else:
            return_record.setdefault("losses", {}).update(
                {
                    "recovery_seed_clearance_floor": float(seed_min_clearance_floor),
                    "recovery_seed_required_clearance": float(seed_required_clearance),
                    "recovery_seed_min_clearance": float(seed_min_clearance_floor),
                    "recovery_seed_path_safe": bool(seed_path_safe),
                    "recovery_seed_path_unsafe": not bool(seed_path_safe),
                    "recovery_optimized_min_clearance_before_seed_guard": float(
                        optimized_min_clearance
                    ),
                    "recovery_seed_clearance_guard_applied": False,
                    "recovery_seed_clearance_guard_source": str(best_seed_name),
                }
            )
        return_record.setdefault("losses", {}).update(
            {
                "recovery_seed_candidate_count": int(len(recovery_seed_chunks)),
                "recovery_bridge_seed_candidate_count": int(
                    len(recovery_bridge_seed_chunks)
                ),
                "recovery_bridge_seed_names": list(recovery_bridge_seed_names),
                "recovery_bridge_seeds_enabled": bool(self.enable_recovery_bridge_seeds),
            }
        )
        return_chunk = return_record["chunk"]
        self._tick_unsafe_recovery_cooldowns()
        direct_rejoin_attempted = False
        direct_rejoin_rejected = False
        detour_rejoin_attempted = False
        detour_rejoin_accepted = False
        repeated_unsafe_target = False
        recovery_candidate_class = "optimized_recovery"
        recover_reject_reason = None
        target_key = self.make_recovery_target_key(return_chunk)
        path_key = self._make_recovery_path_key(return_chunk, target_key)
        if self.recovery_target_failure_counts.get(target_key, 0) > 0:
            repeated_unsafe_target = True
        direct_suppressed = self._recovery_target_is_suppressed(target_key)
        if direct_suppressed:
            repeated_unsafe_target = True
            self.repeated_unsafe_target_count += 1

        direct_terminal = self._recovery_terminal_rejoin_info(
            return_obs,
            return_chunk,
            context,
            return_rejoin_context,
            default_target_index=seed_target_index,
        )
        selected_terminal = direct_terminal
        direct_path = self.evaluate_recovery_path_safety(
            return_obs,
            return_chunk,
            candidate_name="recover_direct",
        )
        # Seed feasibility is measured before optimization.  The optimizer may
        # repair the first recovery action, so final rejection must use the
        # optimized candidate path safety rather than the stale seed result.
        candidate_immediate_safe = bool(direct_path.get("immediate_safe", False))
        selected_direction_info = self.compute_nominal_rejoin_score(
            return_chunk,
            return_nominal,
            obs=return_obs,
        )
        selected_direction_terms = self._recover_direction_alignment_terms(
            selected_direction_info
        )
        if self.enable_direct_rejoin:
            direct_rejoin_attempted = True
            self.direct_rejoin_attempt_count += 1
            return_losses = return_record.get("losses", {})
            act_direction_ok = bool(
                return_losses.get("recover_act_progress_ok", True)
            ) and bool(
                return_losses.get("recover_act_heading_ok", True)
            )
            ordered_tube_ok = bool(
                direct_terminal.get(
                    "recover_ordered_waypoint_tube_ok",
                    direct_terminal.get("recover_ordered_ok", True),
                )
            )
            direct_accept_info = {**direct_terminal, **return_losses}
            recover_reject_reason = self._recovery_reject_reason(
                direct_accept_info,
                direct_path,
                repeated_unsafe_target=direct_suppressed,
                task_progress_ok=act_direction_ok,
                direction_ok=bool(selected_direction_terms["recover_direction_ok"]),
                ordered_ok=ordered_tube_ok,
            )
            if (
                recover_to_task_progress
                and recover_reject_reason == "q_rejoin_failed"
                and ordered_tube_ok
                and act_direction_ok
                and bool(selected_direction_terms["recover_direction_ok"])
            ):
                override_accept_info = {
                    **direct_accept_info,
                    "q_rejoin_ok": True,
                    "q_rejoin_overridden_by_ordered_tube": True,
                }
                override_reject_reason = self._recovery_reject_reason(
                    override_accept_info,
                    direct_path,
                    repeated_unsafe_target=direct_suppressed,
                    task_progress_ok=act_direction_ok,
                    direction_ok=bool(selected_direction_terms["recover_direction_ok"]),
                    ordered_ok=ordered_tube_ok,
                )
                if override_reject_reason is None:
                    recover_reject_reason = None
                    return_record.setdefault("losses", {})[
                        "recover_q_rejoin_overridden_by_ordered_tube"
                    ] = True
                    return_record.setdefault("losses", {})[
                        "recover_original_reject_reason"
                    ] = "q_rejoin_failed"
                    direct_terminal["q_rejoin_overridden_by_ordered_tube"] = True
                    direct_terminal["q_rejoin_ok"] = True
                else:
                    recover_reject_reason = override_reject_reason
                    return_record.setdefault("losses", {})[
                        "recover_q_rejoin_override_blocked_reason"
                    ] = str(override_reject_reason)
        else:
            recover_reject_reason = "unknown"

        return_q_seq = direct_terminal["q_seq"]
        return_eval = direct_terminal["eval"]
        return_min_clearance = float(direct_terminal["min_clearance"])
        return_rejoin_loss = float(direct_terminal["q_rejoin_loss"])
        return_target_index = direct_terminal["target_index"]
        return_q_time_ms = float(direct_terminal["q_eval_time_ms"])
        return_qd_rejoin_loss = float(direct_terminal["qd_rejoin_loss"])
        return_qd_rejoin_index = direct_terminal["qd_rejoin_index"]
        return_qd_time_ms = float(direct_terminal["qd_eval_time_ms"])
        return_q_rejoin_dist = float(direct_terminal["q_rejoin_dist"])
        return_qd_rejoin_dist = float(direct_terminal["qd_rejoin_dist"])
        return_rejoin_ok = self._coerce_bool(direct_terminal.get("q_rejoin_ok", False)) and self._coerce_bool(
            direct_terminal.get("qd_rejoin_ok", False)
        )
        return_safe = bool(
            direct_path["path_safe"]
            if self.safechunk_recovery_corridor_enabled
            else return_min_clearance >= self._acceptance_clearance_threshold()
        )
        return_accepted = bool(recover_reject_reason is None)

        if not return_accepted:
            direct_rejoin_rejected = bool(direct_rejoin_attempted)
            if direct_rejoin_rejected:
                self.direct_rejoin_reject_count += 1
            if recover_reject_reason in {
                "path_unsafe",
                "prefix_unsafe",
                "immediate_unsafe",
            }:
                self._mark_recovery_path_failure(
                    target_key,
                    path_key,
                    recover_reject_reason,
                )

        # Bridge alternatives are now optimizer seeds, so recovery has one
        # final candidate path instead of a separate detour accept branch.
        if not return_accepted and self.enable_delayed_rejoin:
            delayed_bridge_path_safe = bool(direct_path.get("path_safe", False))
            delayed_bridge_reject_reason = str(
                direct_path.get("reject_reason") or recover_reject_reason or "unknown"
            )
            delayed_bridge_safe_prefix_len = int(
                direct_path.get("safe_prefix_len", 0) or 0
            )
            delayed_can_wait = bool(delayed_bridge_path_safe and candidate_immediate_safe)
            return_record["losses"].update(
                {
                    "delayed_rejoin_bridge_path_safe": delayed_bridge_path_safe,
                    "delayed_rejoin_bridge_reject_reason": delayed_bridge_reject_reason,
                    "delayed_rejoin_bridge_safe_prefix_len": delayed_bridge_safe_prefix_len,
                    "delayed_rejoin_suppressed": not delayed_can_wait,
                    "delayed_rejoin_suppressed_reason": (
                        None if delayed_can_wait else "unsafe_bridge_path"
                    ),
                }
            )
            if delayed_can_wait:
                recovery_candidate_class = "delayed_rejoin"
                self.delayed_rejoin_active = True
                self.delayed_rejoin_steps = int(self.delayed_rejoin_wait_steps)
                self.delayed_rejoin_count += 1
            else:
                recovery_candidate_class = "delayed_rejoin_suppressed"
                self.delayed_rejoin_active = False
                self.delayed_rejoin_steps = 0
                self.delayed_rejoin_suppressed_count += 1

        # Re-read from the selected optimized recovery path so diagnostics
        # stay correct after seed-diverse optimization.
        candidate_immediate_safe = bool(direct_path.get("immediate_safe", False))

        if return_accepted:
            losses = return_record["losses"]
            try:
                if losses.get("recover_resume_window_q_frame_l2_mean") is None:
                    target_q_seq_for_resume = self.rollout_nominal_chunk(
                        return_obs,
                        return_nominal,
                    )
                    losses.update(
                        self._recover_resume_window_terms(
                            return_q_seq,
                            target_q_seq_for_resume,
                            candidate=return_chunk,
                            target_chunk=return_nominal,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                losses["recover_resume_window_eval_error"] = str(exc)
            try:
                losses.update(self._ensure_recover_resume_affordance_terms(losses))
            except Exception as exc:  # noqa: BLE001
                losses["recover_resume_affordance_eval_error"] = str(exc)

            def _finite_from_losses(*names: str) -> float | None:
                for name in names:
                    try:
                        value = losses.get(name)
                    except Exception:  # noqa: BLE001
                        value = None
                    try:
                        value_f = float(value)
                    except Exception:  # noqa: BLE001
                        continue
                    if np.isfinite(value_f):
                        return value_f
                return None

            def _bool_from_losses(*names: str, default: bool | None = None) -> bool | None:
                for name in names:
                    try:
                        value = losses.get(name)
                    except Exception:  # noqa: BLE001
                        value = None
                    if value is None:
                        continue
                    return bool(self._coerce_bool(value))
                return default

            def _finite_attr(name: str, default: float) -> float:
                try:
                    value_f = float(getattr(self, name, default))
                except Exception:  # noqa: BLE001
                    return float(default)
                return value_f

            window_mean = _finite_from_losses(
                "recover_resume_window_q_frame_l2_mean",
                "resume_window_q_frame_l2_mean",
                "mpc_recovery_target_tube_window_q_frame_l2_mean",
            )
            window_max = _finite_from_losses(
                "recover_resume_window_q_frame_l2_max",
                "resume_window_q_frame_l2_max",
                "mpc_recovery_target_tube_window_q_frame_l2_max",
            )
            dq_cosine_min = _finite_from_losses(
                "recover_resume_window_dq_cosine_min",
                "resume_window_dq_cosine_min",
                "mpc_recovery_target_tube_window_dq_cosine_min",
            )
            step_l2_max = _finite_from_losses(
                "recover_resume_window_step_l2_error_max",
                "resume_window_step_l2_error_max",
                "mpc_recovery_target_tube_window_step_l2_error_max",
            )
            window_mean_threshold = _finite_attr(
                "recover_resume_window_max_q_frame_l2_mean",
                float("inf"),
            )
            window_max_threshold = _finite_attr(
                "recover_resume_window_max_q_frame_l2_max",
                float("inf"),
            )
            dq_cosine_threshold = _finite_attr(
                "recover_resume_window_min_dq_cosine",
                -float("inf"),
            )
            step_l2_threshold = _finite_attr(
                "recover_resume_window_max_step_l2_error",
                float("inf"),
            )
            affordance_available = _bool_from_losses(
                "recover_resume_affordance_available",
                "resume_affordance_available",
            )
            affordance_task_relevant = _bool_from_losses(
                "recover_resume_affordance_task_relevant",
                "resume_affordance_task_relevant",
            )
            affordance_ok = _bool_from_losses(
                "recover_resume_affordance_ok",
                "resume_affordance_ok",
            )
            affordance_score = _finite_from_losses(
                "recover_resume_affordance_score",
                "resume_affordance_score",
            )
            affordance_component_score = _finite_from_losses(
                "recover_resume_affordance_component_score",
                "resume_affordance_component_score",
            )
            affordance_target_distance = _finite_from_losses(
                "recover_resume_affordance_target_distance",
                "resume_affordance_target_distance",
                "resume_target_distance",
            )
            affordance_component_threshold = _finite_attr(
                "recover_resume_affordance_min_component_for_accept",
                -float("inf"),
            )
            final_resume_reject_reason = self._recovery_reject_reason(
                {**selected_terminal, **losses},
                direct_path,
                repeated_unsafe_target=direct_suppressed,
                task_progress_ok=bool(losses.get("recover_act_progress_ok", True))
                and bool(losses.get("recover_act_heading_ok", True)),
                direction_ok=bool(selected_direction_terms["recover_direction_ok"]),
                ordered_ok=bool(
                    selected_terminal.get(
                        "recover_ordered_waypoint_tube_ok",
                        selected_terminal.get("recover_ordered_ok", True),
                    )
                ),
            )
            if (
                final_resume_reject_reason is None
                and np.isfinite(window_mean_threshold)
                and window_mean is not None
                and window_mean > window_mean_threshold
            ):
                final_resume_reject_reason = "resume_window_q_l2_failed"
            if (
                final_resume_reject_reason is None
                and np.isfinite(window_max_threshold)
                and window_max is not None
                and window_max > window_max_threshold
            ):
                final_resume_reject_reason = "resume_window_q_l2_failed"
            if (
                final_resume_reject_reason is None
                and np.isfinite(dq_cosine_threshold)
                and dq_cosine_min is not None
                and dq_cosine_min < dq_cosine_threshold
            ):
                final_resume_reject_reason = "resume_window_dq_alignment_failed"
            if (
                final_resume_reject_reason is None
                and np.isfinite(step_l2_threshold)
                and step_l2_max is not None
                and step_l2_max > step_l2_threshold
            ):
                final_resume_reject_reason = "resume_window_step_l2_failed"
            affordance_gate_active = bool(
                getattr(self, "recover_resume_affordance_required_for_accept", False)
            ) and (
                affordance_ok is False
                or bool(affordance_available and affordance_task_relevant)
            )
            if (
                final_resume_reject_reason is None
                and affordance_gate_active
                and affordance_ok is not True
            ):
                final_resume_reject_reason = "resume_affordance_not_ready"
            if (
                final_resume_reject_reason is None
                and np.isfinite(affordance_component_threshold)
                and affordance_component_score is not None
                and affordance_component_score < affordance_component_threshold
            ):
                final_resume_reject_reason = "resume_affordance_component_failed"
            losses.update(
                {
                    "recover_final_resume_gate_checked": True,
                    "recover_final_resume_gate_allowed": final_resume_reject_reason is None,
                    "recover_final_resume_gate_rejected": final_resume_reject_reason is not None,
                    "recover_final_resume_gate_reject_reason": (
                        None
                        if final_resume_reject_reason is None
                        else str(final_resume_reject_reason)
                    ),
                    "recover_final_resume_gate_affordance_available": affordance_available,
                    "recover_final_resume_gate_affordance_task_relevant": affordance_task_relevant,
                    "recover_final_resume_gate_affordance_ok": affordance_ok,
                    "recover_final_resume_gate_affordance_score": affordance_score,
                    "recover_final_resume_gate_affordance_component_score": affordance_component_score,
                    "recover_final_resume_gate_affordance_component_threshold": affordance_component_threshold,
                    "recover_final_resume_gate_affordance_target_distance": affordance_target_distance,
                    "recover_final_resume_gate_window_q_frame_l2_mean": window_mean,
                    "recover_final_resume_gate_window_q_frame_l2_mean_threshold": window_mean_threshold,
                    "recover_final_resume_gate_window_q_frame_l2_max": window_max,
                    "recover_final_resume_gate_window_q_frame_l2_max_threshold": window_max_threshold,
                    "recover_final_resume_gate_window_dq_cosine_min": dq_cosine_min,
                    "recover_final_resume_gate_window_dq_cosine_threshold": dq_cosine_threshold,
                    "recover_final_resume_gate_window_step_l2_error_max": step_l2_max,
                    "recover_final_resume_gate_window_step_l2_error_threshold": step_l2_threshold,
                }
            )
            if final_resume_reject_reason is not None:
                return_accepted = False
                recover_reject_reason = str(final_resume_reject_reason)
                losses["recover_reject_reason"] = recover_reject_reason
                if direct_rejoin_attempted and not direct_rejoin_rejected:
                    direct_rejoin_rejected = True
                    self.direct_rejoin_reject_count += 1

        if return_accepted:
            self.safe_corridor_recovery_count += int(
                self.safechunk_recovery_corridor_enabled
            )
            self._clear_recovery_path_failure_streak()
        self._recover_path_min_clearance_history.append(
            float(direct_path.get("recover_path_min_clearance", return_min_clearance))
        )
        if recover_to_task_progress:
            _target_info, _rejoin_info, _progress_score, _progress_available, _effective_weight = (
                self._recover_nominal_rejoin_terms(return_obs, return_chunk, record=True)
            )
            return_record["losses"].update(
                {
                    "recover_task_progress_score": float(_progress_score),
                    "progress_score_available": bool(_progress_available),
                    "recover_rejoin_weight_effective": float(_effective_weight),
                    "recover_step_since_deform": int(self.recover_step_since_deform),
                    "nominal_rejoin_available": bool(_target_info.get("available")),
                    "nominal_rejoin_suppressed_reason": _target_info.get("suppressed_reason"),
                    "nominal_rejoin_clearance": float(_target_info.get("nominal_rejoin_clearance", float("-inf"))),
                    "nominal_rejoin_safe_prefix_len": int(_target_info.get("safe_prefix_len", 0) or 0),
                    "nominal_rejoin_window_start": _target_info.get("nominal_rejoin_window_start"),
                    "nominal_rejoin_window_end": _target_info.get("nominal_rejoin_window_end"),
                    "nominal_rejoin_window_len": _target_info.get("nominal_rejoin_window_len"),
                    "nominal_rejoin_window_type": _target_info.get("nominal_rejoin_window_type"),
                    "safe_rejoin_window_found": bool(_target_info.get("safe_rejoin_window_found", False)),
                    "short_staging_window_found": bool(_target_info.get("short_staging_window_found", False)),
                    **_rejoin_info,
                }
            )
        return_record["losses"].update(
            {
                "recover_direction_alignment_weight": float(
                    self.recover_direction_alignment_weight
                ),
                **selected_direction_info,
                **selected_direction_terms,
            }
        )
        return_record["losses"].update(
            {
                key: selected_terminal.get(key)
                for key in (
                    "recover_ordered_path_available",
                    "recover_ordered_target_index",
                    "recover_ordered_horizon",
                    "recover_ordered_pose_loss",
                    "recover_ordered_delta_loss",
                    "recover_ordered_waypoint_pose_loss",
                    "recover_ordered_waypoint_rmse",
                    "recover_ordered_heading_loss",
                    "recover_ordered_heading_cosine",
                    "recover_ordered_heading_cosine_min",
                    "recover_ordered_heading_cosine_threshold",
                    "recover_ordered_backtrack_count",
                    "recover_ordered_monotonic_ok",
                    "recover_ordered_pose_tube_threshold",
                    "recover_ordered_pose_tube_ok",
                    "recover_ordered_waypoint_tube_ok",
                    "recover_ordered_strict_ok",
                    "recover_ordered_waypoint_index_start",
                    "recover_ordered_waypoint_index_end",
                    "recover_ordered_loss",
                    "recover_ordered_pose_weight",
                    "recover_ordered_delta_weight",
                    "recover_ordered_heading_weight",
                    "recover_ordered_pose_threshold",
                    "recover_ordered_delta_threshold",
                    "recover_ordered_ok",
                )
                if selected_terminal.get(key) is not None
            }
        )
        self._record_ordered_recovery_terms(return_record["losses"])
        return_record["losses"].update(
            self._safechunk_replan_info(
                recovery_target_feasible=recovery_target_feasible,
                stale_recovery_attempted=stale_recovery_attempted,
                stale_recovery_suppressed=return_to_old_path_suppressed,
                return_to_old_path_suppressed=return_to_old_path_suppressed,
                recover_to_task_progress=recover_to_task_progress,
                recovery_replanned_from_current_state=bool(
                    self.safechunk_replan_enabled
                    and self.replan_recovery_from_current_state
                ),
                emergency_brake_immediate_unsafe=not candidate_immediate_safe,
                recovery_seed_immediate_safe=bool(seed_immediate_safe),
                recovery_seed_immediate_unsafe=not bool(seed_immediate_safe),
                recover_candidate_immediate_safe=bool(candidate_immediate_safe),
            )
        )
        return_record["losses"].update(
            {
                "recover_required": True,
                "recovery_candidate_class": recovery_candidate_class,
                "recover_reject_reason": recover_reject_reason,
                "recover_q_rejoin_overridden_by_ordered_tube": return_record.get("losses", {}).get(
                    "recover_q_rejoin_overridden_by_ordered_tube"
                ),
                "recover_original_reject_reason": return_record.get("losses", {}).get(
                    "recover_original_reject_reason"
                ),
                "recovery_seed_immediate_safe": bool(seed_immediate_safe),
                "recovery_seed_immediate_unsafe": not bool(seed_immediate_safe),
                "recovery_seed_required_clearance": float(seed_required_clearance),
                "recovery_seed_min_clearance": float(seed_min_clearance_floor),
                "recovery_seed_path_safe": bool(seed_path_safe),
                "recovery_seed_path_unsafe": not bool(seed_path_safe),
                "recover_task_progress_clearance_penalty_scale": float(
                    self.recover_task_progress_clearance_penalty_scale
                ),
                "recover_candidate_immediate_safe": bool(candidate_immediate_safe),
                "recover_path_min_clearance": float(
                    direct_path.get("recover_path_min_clearance", return_min_clearance)
                ),
                "recover_immediate_clearance": float(
                    direct_path.get("recover_immediate_clearance", float("-inf"))
                ),
                "recover_prefix_min_clearance": float(
                    direct_path.get("recover_prefix_min_clearance", float("-inf"))
                ),
                "recover_path_safe": bool(direct_path.get("path_safe", False)),
                "recover_immediate_safe": bool(
                    direct_path.get("immediate_safe", False)
                ),
                "recover_prefix_safe": bool(direct_path.get("prefix_safe", False)),
                "recover_safe_prefix_len": int(
                    direct_path.get("safe_prefix_len", 0) or 0
                ),
                "recover_target_key": target_key,
                **selected_direction_terms,
                "recovery_path_failure_streak": int(
                    self.recovery_path_failure_streak
                ),
                "direct_rejoin_attempted": bool(direct_rejoin_attempted),
                "direct_rejoin_rejected": bool(direct_rejoin_rejected),
                "detour_rejoin_attempted": bool(detour_rejoin_attempted),
                "detour_rejoin_accepted": bool(detour_rejoin_accepted),
                "delayed_rejoin_active": bool(self.delayed_rejoin_active),
                "delayed_rejoin_steps": int(self.delayed_rejoin_steps),
                "delayed_rejoin_suppressed": bool(
                    return_record["losses"].get("delayed_rejoin_suppressed", False)
                ),
                "delayed_rejoin_suppressed_reason": return_record["losses"].get(
                    "delayed_rejoin_suppressed_reason"
                ),
                "delayed_rejoin_bridge_path_safe": return_record["losses"].get(
                    "delayed_rejoin_bridge_path_safe"
                ),
                "delayed_rejoin_bridge_reject_reason": return_record["losses"].get(
                    "delayed_rejoin_bridge_reject_reason"
                ),
                "delayed_rejoin_bridge_safe_prefix_len": return_record["losses"].get(
                    "delayed_rejoin_bridge_safe_prefix_len"
                ),
                "repeated_unsafe_target": bool(repeated_unsafe_target),
                "post_recovery_act_window_active": bool(
                    self.post_recovery_act_window_active
                ),
                "post_recovery_act_steps_remaining": int(
                    self.post_recovery_act_steps_remaining
                ),
                "post_recovery_act_window_interrupted": False,
            }
        )
        return_record["losses"]["rejoin_q_eval_time_ms"] = return_q_time_ms
        return_record["losses"]["rejoin_qd_eval_time_ms"] = return_qd_time_ms
        return_record["losses"]["return_rejoin_loss"] = float(return_rejoin_loss)
        return_record["losses"]["return_qd_rejoin_loss"] = float(return_qd_rejoin_loss)
        return_record["losses"]["return_target_index"] = return_target_index
        return_record["losses"]["return_qd_rejoin_index"] = return_qd_rejoin_index
        return_record["losses"]["qd_rejoin_loss"] = float(return_qd_rejoin_loss)
        return_record["losses"]["qd_rejoin_dist"] = float(return_qd_rejoin_dist)
        return_record["losses"]["qd_rejoin_threshold"] = float(self.qd_rejoin_threshold)
        return_record["losses"]["qd_rejoin_index"] = return_qd_rejoin_index
        for key in (
            "qd_rejoin_ok",
            "qd_rejoin_required",
            "qd_rejoin_hard_threshold",
            "qd_rejoin_hard_failed",
            "qd_rejoin_soft_ok",
        ):
            if key in selected_terminal:
                return_record["losses"][key] = selected_terminal[key]
        context.target_rejoin_index = None if return_target_index is None else int(return_target_index)

        if return_accepted:
            context.phase = "resume"
            context.active = False
            recovered_chunk = self._splice_explicit_return_chunk(
                chunk,
                deform_chunk,
                return_chunk,
                action_idx,
                return_target_index,
            )
            return recovered_chunk, self._explicit_return_info(
                obs=obs,
                chunk=recovered_chunk,
                nominal=chunk,
                context=context,
                recovery_phase="recover",
                deform_chunk=deform_chunk,
                deform_losses=deform_record["losses"],
                deform_eval=deform_eval,
                deform_stage_accepted=True,
                return_chunk=return_chunk,
                return_losses=return_record["losses"],
                return_eval=return_eval,
                return_accepted=True,
                return_rejoin_loss=return_rejoin_loss,
                return_target_index=return_target_index,
                fallback_used=False,
                rejection_cause=None,
            )

        context.phase = "recover"
        context.return_retries += 1
        context.recover_retries += 1
        if recover_reject_reason in {
            "path_unsafe",
            "prefix_unsafe",
            "immediate_unsafe",
            "nominal_prefix_unsafe",
        }:
            rejection_cause = "unsafe"
        elif recover_reject_reason in {
            "q_rejoin_failed",
            "qdot_rejoin_failed",
            "task_progress_failed",
            "direction_alignment_failed",
            "ordered_path_failed",
            "repeated_unsafe_target",
        }:
            rejection_cause = "unrecoverable"
        else:
            rejection_cause = self._optimized_reject_reason_from_flags(
                not return_safe,
                not return_rejoin_ok,
            )
        return chunk.copy(), self._explicit_return_info(
            obs=obs,
            chunk=chunk,
            nominal=chunk,
            context=context,
            recovery_phase="recover",
            deform_chunk=deform_chunk,
            deform_losses=deform_record["losses"],
            deform_eval=deform_eval,
            deform_stage_accepted=True,
            return_chunk=return_chunk,
            return_losses=return_record["losses"],
            return_eval=return_eval,
            return_accepted=False,
            return_rejoin_loss=return_rejoin_loss,
            return_target_index=return_target_index,
            fallback_used=True,
            rejection_cause=rejection_cause,
        )

    def _create_recovery_context(
        self,
        nominal_chunk: np.ndarray,
        nominal_q_seq: np.ndarray,
        nominal_ee_seq: np.ndarray | None = None,
        start_chunk_index: int | None = None,
        observation_history: Any | None = None,
        policy_buffer_metadata: Any | None = None,
    ) -> RecoveryContext:
        """Cache nominal rollout data used by later recovery/rejoin checks."""
        self._trigger_count += 1
        context = RecoveryContext(
            nominal_chunk=np.asarray(nominal_chunk, dtype=np.float32).copy(),
            nominal_q_seq=np.asarray(nominal_q_seq, dtype=np.float32).copy(),
            nominal_ee_seq=(
                None
                if nominal_ee_seq is None
                else np.asarray(nominal_ee_seq, dtype=np.float32).copy()
            ),
            start_chunk_index=start_chunk_index,
            trigger_step=self._trigger_count,
            active=True,
            phase="horizon_deform",
            observation_history=observation_history,
            policy_buffer_metadata=policy_buffer_metadata,
        )
        self.recovery_context = context
        return context

    def _explicit_return_info(
        self,
        obs: Any,
        chunk: np.ndarray,
        nominal: np.ndarray,
        context: RecoveryContext,
        recovery_phase: str,
        deform_chunk: np.ndarray,
        deform_losses: Mapping[str, Any] | None,
        deform_eval: Mapping[str, Any] | None,
        deform_stage_accepted: bool,
        return_chunk: np.ndarray | None,
        return_losses: Mapping[str, Any] | None,
        return_eval: Mapping[str, Any] | None,
        return_accepted: bool,
        return_rejoin_loss: float,
        return_target_index: int | None,
        fallback_used: bool,
        rejection_cause: str | None,
    ) -> InfoDict:
        """Build diagnostics returned with explicit recovery decisions."""
        del obs
        deform_losses = dict(deform_losses or {})
        return_losses = dict(return_losses or {})
        deform_eval = dict(deform_eval or {})
        return_eval = dict(return_eval or {})

        def optimizer_loss(name: str, default: Any | None = None) -> Any | None:
            """Return aggregate optimizer diagnostics, preferring recovery-stage data."""
            value = return_losses.get(name)
            if value is None:
                value = deform_losses.get(name, default)
            return value

        def prefixed_loss(prefix: str, losses: Mapping[str, Any], name: str) -> Any | None:
            """Expose stage-specific optimizer diagnostics without mutating loss records."""
            return losses.get(f"{prefix}_{name}", losses.get(name))

        replan_info = self._safechunk_replan_info()
        deform_min_clearance_stage = float(
            deform_eval.get(
                "min_clearance",
                deform_losses.get("min_clearance", float("-inf")),
            )
        )
        return_min_clearance = (
            None
            if return_chunk is None
            else float(
                return_eval.get(
                    "min_clearance",
                    return_losses.get("min_clearance", float("-inf")),
                )
            )
        )
        recover_losses = return_losses
        recover_path_safe = (
            self._coerce_bool(recover_losses["recover_path_safe"])
            if "recover_path_safe" in recover_losses
            else None
        )
        q_rejoin_dist = self._sqrt_loss(return_rejoin_loss)
        qd_rejoin_dist = self._sqrt_loss(
            return_losses.get(
                "qd_rejoin_loss",
                return_losses.get("return_qd_rejoin_loss", 0.0),
            )
        )
        qd_rejoin_ok = return_losses.get("qd_rejoin_ok")
        qd_acceptance_ok, qd_acceptance = self._qd_rejoin_acceptance(
            return_losses.get(
                "qd_rejoin_index",
                return_losses.get("return_qd_rejoin_index"),
            ),
            qd_rejoin_dist,
        )
        if qd_rejoin_ok is None:
            qd_rejoin_ok = qd_acceptance_ok
        else:
            qd_rejoin_ok = self._coerce_bool(qd_rejoin_ok) and bool(qd_acceptance_ok)
        combined_min_clearance = float(
            max(
                deform_min_clearance_stage,
                return_min_clearance
                if return_min_clearance is not None
                else float("-inf"),
            )
        )
        is_safe = bool(
            deform_stage_accepted
            and (
                return_accepted
                or return_chunk is None
                or (
                    return_min_clearance is not None
                    and return_min_clearance >= self._acceptance_clearance_threshold()
                )
            )
        )
        is_recoverable = bool(return_accepted)
        recovery_rejected = bool(deform_stage_accepted and not return_accepted)
        safety_rejected = bool(not deform_stage_accepted or rejection_cause == "unsafe")
        mode = "recover" if return_accepted else str(recovery_phase)
        if not return_accepted and fallback_used:
            mode = "recover_rejected"
        info = {
            "mode": mode,
            "deform_mode": mode,
            "recovery_mode": str(recovery_phase),
            "deformation_source": "explicit_recover_deform",
            "optimized_accepted": bool(return_accepted),
            "deform_safe": bool(is_safe),
            "is_safe": bool(is_safe),
            "is_recoverable": bool(is_recoverable),
            "safety_rejected": bool(safety_rejected),
            "recovery_rejected": bool(recovery_rejected),
            "rejection_cause": rejection_cause,
            "optimized_fallback": "brake" if fallback_used else None,
            "optimized_reject_reason": self._optimized_reject_reason_from_flags(
                safety_rejected,
                recovery_rejected,
            )
            if fallback_used
            else None,
            "fallback_reason": rejection_cause if fallback_used else None,
            "fallback_used": bool(fallback_used),
            "recover_required": True,
            "recover_corridor_accepted": bool(return_accepted),
            "recover_accepted": bool(return_accepted),
            "return_accepted": bool(return_accepted),
            "deform_stage_accepted": bool(deform_stage_accepted),
            "recovery_phase": str(recovery_phase),
            "cached_motion_active": bool(getattr(context, "active", False)),
            "recovery_context_active": bool(getattr(context, "active", False)),
            "target_rejoin_index": (
                None
                if getattr(context, "target_rejoin_index", None) is None
                else int(context.target_rejoin_index)
            ),
            "return_target_index": (
                None if return_target_index is None else int(return_target_index)
            ),
            "recover_target_index": (
                None if return_target_index is None else int(return_target_index)
            ),
            "act_resume_index": (
                None if return_target_index is None else int(return_target_index)
            ),
            "deform_min_clearance": combined_min_clearance,
            "min_clearance": combined_min_clearance,
            "deform_min_clearance_stage": float(deform_min_clearance_stage),
            "recover_min_clearance": return_min_clearance,
            "return_min_clearance": return_min_clearance,
            "required_min_clearance": float(self.min_clearance),
            "clearance_gap": float(self.min_clearance - combined_min_clearance),
            "safety_loss": float(
                return_losses.get(
                    "safety_loss",
                    deform_losses.get("safety_loss", 0.0),
                )
            ),
            "action_deviation_loss": float(
                return_losses.get(
                    "action_deviation_loss",
                    deform_losses.get("action_deviation_loss", 0.0),
                )
            ),
            "path_loss": 0.0,
            "rejoin_loss": float(return_rejoin_loss),
            "recover_rejoin_loss": float(return_rejoin_loss),
            "q_rejoin_loss": float(return_rejoin_loss),
            "q_rejoin_dist": float(q_rejoin_dist),
            "q_rejoin_threshold": float(self.q_rejoin_threshold),
            "q_rejoin_index": (
                None if return_target_index is None else int(return_target_index)
            ),
            "qd_rejoin_loss": float(
                return_losses.get(
                    "qd_rejoin_loss",
                    return_losses.get("return_qd_rejoin_loss", 0.0),
                )
            ),
            "qd_rejoin_dist": float(qd_rejoin_dist),
            "qd_rejoin_threshold": float(self.qd_rejoin_threshold),
            "qd_rejoin_index": return_losses.get(
                "qd_rejoin_index",
                return_losses.get("return_qd_rejoin_index"),
            ),
            "qd_rejoin_ok": self._coerce_bool(qd_rejoin_ok),
            **qd_acceptance,
            "return_qd_rejoin_loss": return_losses.get("return_qd_rejoin_loss"),
            "return_qd_rejoin_index": return_losses.get("return_qd_rejoin_index"),
            "rejoin_index": (
                None if return_target_index is None else int(return_target_index)
            ),
            "j_best": (
                None if return_target_index is None else int(return_target_index)
            ),
            "recover_retries": int(context.recover_retries),
            "max_recover_retries": int(self.max_return_retries),
            "deform_chunk_length": int(
                0 if deform_chunk is None else np.asarray(deform_chunk).shape[0]
            ),
            "recover_chunk_length": int(
                0 if return_chunk is None else np.asarray(return_chunk).shape[0]
            ),
            "return_chunk_length": int(
                0 if return_chunk is None else np.asarray(return_chunk).shape[0]
            ),
            "committed_chunk_total_length": int(np.asarray(chunk).shape[0]),
            "explicit_recovery": True,
            "final_rejoin_metric": self.final_rejoin_metric,
            "inner_rejoin_metric": self.inner_rejoin_metric,
            "ee_final_check_available": False if not self.use_ee_final_check else None,
            "ee_rejoin_loss": 0.0,
            "ee_rejoin_dist": 0.0,
            "ee_nom_cache_time_ms": 0.0,
            "ee_final_check_time_ms": 0.0,
            "rejoin_q_eval_time_ms": float(
                return_losses.get("rejoin_q_eval_time_ms", 0.0)
            ),
            "rejoin_qd_eval_time_ms": float(
                return_losses.get("rejoin_qd_eval_time_ms", 0.0)
            ),
            "smoothness_loss": float(
                return_losses.get(
                    "smoothness_loss",
                    deform_losses.get("smoothness_loss", 0.0),
                )
            ),
            "existing_optimization_loss": float(
                return_losses.get(
                    "existing_optimization_loss",
                    return_losses.get(
                        "total_loss",
                        deform_losses.get("total_loss", 0.0),
                    ),
                )
            ),
            "total_loss": float(
                return_losses.get(
                    "total_loss",
                    deform_losses.get("total_loss", 0.0),
                )
            ),
            "deformation_norm": self.deform._controlled_deformation_norm(
                chunk,
                nominal,
            ),
            "recover_path_min_clearance": recover_losses.get(
                "recover_path_min_clearance"
            ),
            "recover_immediate_clearance": recover_losses.get(
                "recover_immediate_clearance"
            ),
            "recover_prefix_min_clearance": recover_losses.get(
                "recover_prefix_min_clearance"
            ),
            "recover_path_safe": recover_path_safe,
            "recover_immediate_safe": recover_losses.get("recover_immediate_safe"),
            "recover_prefix_safe": recover_losses.get("recover_prefix_safe"),
            "recover_safe_prefix_len": recover_losses.get("recover_safe_prefix_len"),
            "recovery_seed_clearance_floor": recover_losses.get(
                "recovery_seed_clearance_floor"
            ),
            "recovery_seed_required_clearance": recover_losses.get(
                "recovery_seed_required_clearance"
            ),
            "recovery_seed_min_clearance": recover_losses.get(
                "recovery_seed_min_clearance"
            ),
            "recovery_seed_path_safe": recover_losses.get(
                "recovery_seed_path_safe"
            ),
            "recovery_seed_path_unsafe": recover_losses.get(
                "recovery_seed_path_unsafe"
            ),
            "recovery_optimized_min_clearance_before_seed_guard": recover_losses.get(
                "recovery_optimized_min_clearance_before_seed_guard"
            ),
            "recovery_seed_clearance_guard_applied": recover_losses.get(
                "recovery_seed_clearance_guard_applied"
            ),
            "recovery_seed_clearance_guard_source": recover_losses.get(
                "recovery_seed_clearance_guard_source"
            ),
            "recover_task_progress_clearance_penalty_scale": recover_losses.get(
                "recover_task_progress_clearance_penalty_scale"
            ),
            "recovery_seed_candidate_count": recover_losses.get(
                "recovery_seed_candidate_count"
            ),
            "recovery_bridge_seed_candidate_count": recover_losses.get(
                "recovery_bridge_seed_candidate_count"
            ),
            "recovery_bridge_seed_names": recover_losses.get(
                "recovery_bridge_seed_names"
            ),
            "recovery_bridge_seeds_enabled": recover_losses.get(
                "recovery_bridge_seeds_enabled"
            ),
            "recover_target_key": recover_losses.get("recover_target_key"),
            "recovery_candidate_class": recover_losses.get(
                "recovery_candidate_class"
            ),
            "recover_reject_reason": recover_losses.get("recover_reject_reason"),
            "recovery_path_failure_streak": recover_losses.get(
                "recovery_path_failure_streak",
                self.recovery_path_failure_streak,
            ),
            "direct_rejoin_attempted": recover_losses.get("direct_rejoin_attempted"),
            "direct_rejoin_rejected": recover_losses.get("direct_rejoin_rejected"),
            "detour_rejoin_attempted": recover_losses.get("detour_rejoin_attempted"),
            "detour_rejoin_accepted": recover_losses.get("detour_rejoin_accepted"),
            "delayed_rejoin_active": recover_losses.get("delayed_rejoin_active"),
            "delayed_rejoin_steps": recover_losses.get("delayed_rejoin_steps"),
            "delayed_rejoin_suppressed": recover_losses.get(
                "delayed_rejoin_suppressed"
            ),
            "delayed_rejoin_suppressed_reason": recover_losses.get(
                "delayed_rejoin_suppressed_reason"
            ),
            "delayed_rejoin_bridge_path_safe": recover_losses.get(
                "delayed_rejoin_bridge_path_safe"
            ),
            "delayed_rejoin_bridge_reject_reason": recover_losses.get(
                "delayed_rejoin_bridge_reject_reason"
            ),
            "delayed_rejoin_bridge_safe_prefix_len": recover_losses.get(
                "delayed_rejoin_bridge_safe_prefix_len"
            ),
            "repeated_unsafe_target": recover_losses.get("repeated_unsafe_target"),
            "post_recovery_act_window_active": recover_losses.get(
                "post_recovery_act_window_active",
                bool(self.post_recovery_act_window_active),
            ),
            "post_recovery_act_steps_remaining": recover_losses.get(
                "post_recovery_act_steps_remaining",
                int(self.post_recovery_act_steps_remaining),
            ),
            "post_recovery_act_window_interrupted": recover_losses.get(
                "post_recovery_act_window_interrupted",
                False,
            ),
            "recover_projection_on_nominal": return_losses.get(
                "recover_projection_on_nominal"
            ),
            "recover_cosine_to_nominal": return_losses.get(
                "recover_cosine_to_nominal"
            ),
            "recover_direction_cosine": return_losses.get(
                "recover_direction_cosine"
            ),
            "recover_direction_cosine_threshold": return_losses.get(
                "recover_direction_cosine_threshold"
            ),
            "recover_direction_loss": return_losses.get("recover_direction_loss"),
            "recover_direction_ok": return_losses.get("recover_direction_ok"),
            "recover_direction_alignment_available": return_losses.get(
                "recover_direction_alignment_available"
            ),
            "recover_direction_alignment_weight": return_losses.get(
                "recover_direction_alignment_weight"
            ),
            "recover_act_direction_available": return_losses.get(
                "recover_act_direction_available"
            ),
            "recover_act_progress_loss": return_losses.get(
                "recover_act_progress_loss"
            ),
            "recover_act_heading_loss": return_losses.get(
                "recover_act_heading_loss"
            ),
            "recover_act_direction_loss": return_losses.get(
                "recover_act_direction_loss"
            ),
            "recover_act_progress_projection": return_losses.get(
                "recover_act_progress_projection"
            ),
            "recover_act_target_progress": return_losses.get(
                "recover_act_target_progress"
            ),
            "recover_act_heading_cosine": return_losses.get(
                "recover_act_heading_cosine"
            ),
            "recover_act_heading_cosine_min": return_losses.get(
                "recover_act_heading_cosine_min"
            ),
            "recover_act_progress_ok": return_losses.get("recover_act_progress_ok"),
            "recover_act_heading_ok": return_losses.get("recover_act_heading_ok"),
            "recover_act_progress_weight": return_losses.get(
                "recover_act_progress_weight"
            ),
            "recover_act_heading_weight": return_losses.get(
                "recover_act_heading_weight"
            ),
            "recover_min_act_heading_cosine": return_losses.get(
                "recover_min_act_heading_cosine"
            ),
            "recover_resume_tube_weight": return_losses.get("recover_resume_tube_weight"),
            "recover_resume_tube_score": return_losses.get("resume_tube_score"),
            "recover_resume_tube_ok": return_losses.get("resume_tube_ok"),
            "recover_resume_tube_min_score": return_losses.get("resume_tube_min_score"),
            "recover_resume_tube_component_score": return_losses.get("resume_tube_component_score"),
            "recover_resume_tube_min_component_score": return_losses.get("resume_tube_min_component_score"),
            "recover_resume_tube_component_ok": return_losses.get("resume_tube_component_ok"),
            "recover_resume_tube_terminal_score": return_losses.get("resume_tube_terminal_score"),
            "recover_resume_tube_path_score": return_losses.get("resume_tube_path_score"),
            "recover_resume_tube_progress_score": return_losses.get("resume_tube_progress_score"),
            "recover_resume_tube_heading_score": return_losses.get("resume_tube_heading_score"),
            "recover_resume_tube_clearance_score": return_losses.get("resume_tube_clearance_score"),
            "recover_resume_tube_terminal_dist": return_losses.get("resume_tube_terminal_dist"),
            "recover_resume_tube_terminal_delta": return_losses.get("resume_tube_terminal_delta"),
            "recover_resume_tube_q_error": return_losses.get("resume_tube_q_error"),
            "recover_resume_tube_terminal_threshold": return_losses.get("resume_tube_terminal_threshold"),
            "recover_resume_tube_ordered_loss": return_losses.get("resume_tube_ordered_loss"),
            "recover_resume_tube_prefix_min_clearance": return_losses.get("resume_tube_prefix_min_clearance"),
            "recover_resume_tube_required_clearance": return_losses.get("resume_tube_required_clearance"),
            "recover_resume_tube_prefix_safe": return_losses.get("resume_tube_prefix_safe"),
            "recover_resume_tube_terminal_ok": return_losses.get("resume_tube_terminal_ok"),
            "recover_resume_window_available": return_losses.get("recover_resume_window_available"),
            "recover_resume_window_len": return_losses.get("recover_resume_window_len"),
            "recover_resume_window_requested_len": return_losses.get("recover_resume_window_requested_len"),
            "recover_resume_window_loss": return_losses.get("recover_resume_window_loss"),
            "recover_resume_window_total_loss": return_losses.get("recover_resume_window_total_loss"),
            "recover_resume_window_dist": return_losses.get("recover_resume_window_dist"),
            "recover_resume_window_error_l2": return_losses.get("recover_resume_window_error_l2"),
            "recover_resume_window_dq_loss": return_losses.get("recover_resume_window_dq_loss"),
            "recover_resume_window_dq_dist": return_losses.get("recover_resume_window_dq_dist"),
            "recover_resume_window_action_loss": return_losses.get("recover_resume_window_action_loss"),
            "recover_resume_window_action_dist": return_losses.get("recover_resume_window_action_dist"),
            "recover_resume_window_q_frame_l2": return_losses.get("recover_resume_window_q_frame_l2"),
            "recover_resume_window_q_frame_l2_mean": return_losses.get("recover_resume_window_q_frame_l2_mean"),
            "recover_resume_window_q_frame_l2_max": return_losses.get("recover_resume_window_q_frame_l2_max"),
            "recover_resume_window_wrist_l2": return_losses.get("recover_resume_window_wrist_l2"),
            "recover_resume_window_wrist_l2_mean": return_losses.get("recover_resume_window_wrist_l2_mean"),
            "recover_resume_window_wrist_l2_max": return_losses.get("recover_resume_window_wrist_l2_max"),
            "recover_resume_window_left_wrist_abs": return_losses.get("recover_resume_window_left_wrist_abs"),
            "recover_resume_window_left_wrist_abs_mean": return_losses.get("recover_resume_window_left_wrist_abs_mean"),
            "recover_resume_window_left_wrist_abs_max": return_losses.get("recover_resume_window_left_wrist_abs_max"),
            "recover_resume_window_right_wrist_abs": return_losses.get("recover_resume_window_right_wrist_abs"),
            "recover_resume_window_right_wrist_abs_mean": return_losses.get("recover_resume_window_right_wrist_abs_mean"),
            "recover_resume_window_right_wrist_abs_max": return_losses.get("recover_resume_window_right_wrist_abs_max"),
            "recover_resume_window_recovery_step_l2": return_losses.get("recover_resume_window_recovery_step_l2"),
            "recover_resume_window_target_step_l2": return_losses.get("recover_resume_window_target_step_l2"),
            "recover_resume_window_step_l2_error": return_losses.get("recover_resume_window_step_l2_error"),
            "recover_resume_window_step_l2_error_mean": return_losses.get("recover_resume_window_step_l2_error_mean"),
            "recover_resume_window_step_l2_error_max": return_losses.get("recover_resume_window_step_l2_error_max"),
            "recover_resume_window_dq_error_l2": return_losses.get("recover_resume_window_dq_error_l2"),
            "recover_resume_window_dq_cosine": return_losses.get("recover_resume_window_dq_cosine"),
            "recover_resume_window_dq_cosine_mean": return_losses.get("recover_resume_window_dq_cosine_mean"),
            "recover_resume_window_dq_cosine_min": return_losses.get("recover_resume_window_dq_cosine_min"),
            "recover_resume_window_dq_norm_ratio": return_losses.get("recover_resume_window_dq_norm_ratio"),
            "recover_resume_window_dq_norm_ratio_mean": return_losses.get("recover_resume_window_dq_norm_ratio_mean"),
            "recover_resume_window_dq_norm_ratio_min": return_losses.get("recover_resume_window_dq_norm_ratio_min"),
            "recover_resume_window_start_local_index": return_losses.get("recover_resume_window_start_local_index"),
            "recover_resume_window_end_local_index": return_losses.get("recover_resume_window_end_local_index"),
            "recover_resume_window_weight": return_losses.get("recover_resume_window_weight"),
            "recover_resume_window_dq_weight": return_losses.get("recover_resume_window_dq_weight"),
            "recover_resume_window_action_weight": return_losses.get("recover_resume_window_action_weight"),
            "recover_resume_window_q": return_losses.get("recover_resume_window_q"),
            "recover_resume_window_target_q": return_losses.get("recover_resume_window_target_q"),
            "recover_resume_window_action": return_losses.get("recover_resume_window_action"),
            "recover_resume_window_target_action": return_losses.get("recover_resume_window_target_action"),
            "recover_resume_affordance_weight": return_losses.get("recover_resume_affordance_weight"),
            "recover_resume_affordance_bonus": return_losses.get("recover_resume_affordance_bonus"),
            "recover_resume_affordance_enabled": return_losses.get("resume_affordance_enabled"),
            "recover_resume_affordance_available": return_losses.get("resume_affordance_available"),
            "recover_resume_affordance_task_relevant": return_losses.get("resume_affordance_task_relevant"),
            "recover_resume_affordance_score": return_losses.get("resume_affordance_score"),
            "recover_resume_affordance_ok": return_losses.get("resume_affordance_ok"),
            "recover_resume_affordance_min_score": return_losses.get("resume_affordance_min_score"),
            "recover_resume_affordance_component_score": return_losses.get("resume_affordance_component_score"),
            "recover_resume_affordance_min_component_score": return_losses.get("resume_affordance_min_component_score"),
            "recover_resume_affordance_target_distance": return_losses.get("resume_affordance_target_distance"),
            "recover_resume_affordance_target_distance_score": return_losses.get("resume_affordance_target_distance_score"),
            "recover_resume_affordance_contact_score": return_losses.get("resume_affordance_contact_score"),
            "recover_resume_affordance_progress_score": return_losses.get("resume_affordance_progress_score"),
            "recover_resume_affordance_alignment_score": return_losses.get("resume_affordance_alignment_score"),
            "recover_resume_affordance_continuity_score": return_losses.get("resume_affordance_continuity_score"),
            "recover_resume_affordance_safety_score": return_losses.get("resume_affordance_safety_score"),
            "recover_resume_affordance_interaction_context": return_losses.get("interaction_context"),
            "recover_final_resume_gate_checked": return_losses.get("recover_final_resume_gate_checked"),
            "recover_final_resume_gate_allowed": return_losses.get("recover_final_resume_gate_allowed"),
            "recover_final_resume_gate_rejected": return_losses.get("recover_final_resume_gate_rejected"),
            "recover_final_resume_gate_reject_reason": return_losses.get("recover_final_resume_gate_reject_reason"),
            "recover_final_resume_gate_affordance_available": return_losses.get("recover_final_resume_gate_affordance_available"),
            "recover_final_resume_gate_affordance_task_relevant": return_losses.get("recover_final_resume_gate_affordance_task_relevant"),
            "recover_final_resume_gate_affordance_ok": return_losses.get("recover_final_resume_gate_affordance_ok"),
            "recover_final_resume_gate_affordance_score": return_losses.get("recover_final_resume_gate_affordance_score"),
            "recover_final_resume_gate_affordance_component_score": return_losses.get("recover_final_resume_gate_affordance_component_score"),
            "recover_final_resume_gate_affordance_component_threshold": return_losses.get("recover_final_resume_gate_affordance_component_threshold"),
            "recover_final_resume_gate_affordance_target_distance": return_losses.get("recover_final_resume_gate_affordance_target_distance"),
            "recover_final_resume_gate_window_q_frame_l2_mean": return_losses.get("recover_final_resume_gate_window_q_frame_l2_mean"),
            "recover_final_resume_gate_window_q_frame_l2_mean_threshold": return_losses.get("recover_final_resume_gate_window_q_frame_l2_mean_threshold"),
            "recover_final_resume_gate_window_q_frame_l2_max": return_losses.get("recover_final_resume_gate_window_q_frame_l2_max"),
            "recover_final_resume_gate_window_q_frame_l2_max_threshold": return_losses.get("recover_final_resume_gate_window_q_frame_l2_max_threshold"),
            "recover_final_resume_gate_window_dq_cosine_min": return_losses.get("recover_final_resume_gate_window_dq_cosine_min"),
            "recover_final_resume_gate_window_dq_cosine_threshold": return_losses.get("recover_final_resume_gate_window_dq_cosine_threshold"),
            "recover_final_resume_gate_window_step_l2_error_max": return_losses.get("recover_final_resume_gate_window_step_l2_error_max"),
            "recover_final_resume_gate_window_step_l2_error_threshold": return_losses.get("recover_final_resume_gate_window_step_l2_error_threshold"),
            "recover_ordered_path_available": return_losses.get(
                "recover_ordered_path_available"
            ),
            "recover_ordered_target_index": return_losses.get(
                "recover_ordered_target_index"
            ),
            "recover_ordered_horizon": return_losses.get("recover_ordered_horizon"),
            "recover_ordered_pose_loss": return_losses.get(
                "recover_ordered_pose_loss"
            ),
            "recover_ordered_delta_loss": return_losses.get(
                "recover_ordered_delta_loss"
            ),
            "recover_ordered_waypoint_pose_loss": return_losses.get(
                "recover_ordered_waypoint_pose_loss"
            ),
            "recover_ordered_waypoint_rmse": return_losses.get(
                "recover_ordered_waypoint_rmse"
            ),
            "recover_ordered_heading_loss": return_losses.get(
                "recover_ordered_heading_loss"
            ),
            "recover_ordered_heading_cosine": return_losses.get(
                "recover_ordered_heading_cosine"
            ),
            "recover_ordered_heading_cosine_min": return_losses.get(
                "recover_ordered_heading_cosine_min"
            ),
            "recover_ordered_heading_cosine_threshold": return_losses.get(
                "recover_ordered_heading_cosine_threshold"
            ),
            "recover_ordered_backtrack_count": return_losses.get(
                "recover_ordered_backtrack_count"
            ),
            "recover_ordered_monotonic_ok": return_losses.get(
                "recover_ordered_monotonic_ok"
            ),
            "recover_ordered_pose_tube_threshold": return_losses.get(
                "recover_ordered_pose_tube_threshold"
            ),
            "recover_ordered_pose_tube_ok": return_losses.get(
                "recover_ordered_pose_tube_ok"
            ),
            "recover_ordered_waypoint_tube_ok": return_losses.get(
                "recover_ordered_waypoint_tube_ok"
            ),
            "recover_ordered_strict_ok": return_losses.get(
                "recover_ordered_strict_ok"
            ),
            "recover_ordered_waypoint_index_start": return_losses.get(
                "recover_ordered_waypoint_index_start"
            ),
            "recover_ordered_waypoint_index_end": return_losses.get(
                "recover_ordered_waypoint_index_end"
            ),
            "recover_ordered_loss": return_losses.get("recover_ordered_loss"),
            "recover_ordered_pose_weight": return_losses.get(
                "recover_ordered_pose_weight"
            ),
            "recover_ordered_delta_weight": return_losses.get(
                "recover_ordered_delta_weight"
            ),
            "recover_ordered_heading_weight": return_losses.get(
                "recover_ordered_heading_weight"
            ),
            "recover_ordered_pose_threshold": return_losses.get(
                "recover_ordered_pose_threshold"
            ),
            "recover_ordered_delta_threshold": return_losses.get(
                "recover_ordered_delta_threshold"
            ),
            "recover_ordered_ok": return_losses.get("recover_ordered_ok"),
            "nominal_rejoin_score": return_losses.get("nominal_rejoin_score"),
            "nominal_rejoin_available": return_losses.get(
                "nominal_rejoin_available"
            ),
            "nominal_rejoin_suppressed_reason": return_losses.get(
                "nominal_rejoin_suppressed_reason"
            ),
            "nominal_rejoin_clearance": return_losses.get(
                "nominal_rejoin_clearance"
            ),
            "nominal_rejoin_safe_prefix_len": return_losses.get(
                "nominal_rejoin_safe_prefix_len"
            ),
            "nominal_rejoin_window_start": return_losses.get(
                "nominal_rejoin_window_start"
            ),
            "nominal_rejoin_window_end": return_losses.get(
                "nominal_rejoin_window_end"
            ),
            "nominal_rejoin_window_len": return_losses.get(
                "nominal_rejoin_window_len"
            ),
            "nominal_rejoin_window_type": return_losses.get(
                "nominal_rejoin_window_type"
            ),
            "safe_rejoin_window_found": return_losses.get(
                "safe_rejoin_window_found"
            ),
            "short_staging_window_found": return_losses.get(
                "short_staging_window_found"
            ),
            "deform_rejoin_available": deform_losses.get("deform_rejoin_available"),
            "deform_rejoin_window_loss": deform_losses.get("deform_rejoin_window_loss"),
            "deform_rejoin_q_loss": deform_losses.get("deform_rejoin_q_loss"),
            "deform_rejoin_qd_loss": deform_losses.get("deform_rejoin_qd_loss"),
            "deform_rejoin_action_loss": deform_losses.get("deform_rejoin_action_loss"),
            "deform_rejoin_heading_loss": deform_losses.get("deform_rejoin_heading_loss"),
            "deform_rejoin_q_dist": deform_losses.get("deform_rejoin_q_dist"),
            "deform_rejoin_qd_dist": deform_losses.get("deform_rejoin_qd_dist"),
            "deform_rejoin_action_dist": deform_losses.get("deform_rejoin_action_dist"),
            "deform_rejoin_heading_cosine": deform_losses.get("deform_rejoin_heading_cosine"),
            "deform_rejoin_best_window_offset": deform_losses.get(
                "deform_rejoin_best_window_offset"
            ),
            "deform_rejoin_weight": deform_losses.get("deform_rejoin_weight"),
            "deform_rejoin_velocity_weight": deform_losses.get(
                "deform_rejoin_velocity_weight"
            ),
            "deform_rejoin_action_weight": deform_losses.get(
                "deform_rejoin_action_weight"
            ),
            "deform_rejoin_heading_weight": deform_losses.get(
                "deform_rejoin_heading_weight"
            ),
            "recover_task_progress_score": return_losses.get(
                "recover_task_progress_score"
            ),
            "recover_score_total": return_losses.get("recover_score_total"),
            "recover_rejoin_weight_effective": return_losses.get(
                "recover_rejoin_weight_effective"
            ),
            "recover_step_since_deform": return_losses.get(
                "recover_step_since_deform"
            ),
            "optimizer_method": optimizer_loss("optimizer_method"),
            "deform_optimizer_method": prefixed_loss(
                "deform", deform_losses, "optimizer_method"
            ),
            "return_optimizer_method": prefixed_loss(
                "return", return_losses, "optimizer_method"
            ),
            "explicit_optimizer_method": optimizer_loss("explicit_optimizer_method"),
            "optimizer_evaluations": optimizer_loss("optimizer_evaluations"),
            "deform_optimizer_time_ms": deform_losses.get("deform_optimizer_time_ms"),
            "return_optimizer_time_ms": return_losses.get("return_optimizer_time_ms"),
            "explicit_optimizer_time_ms": optimizer_loss("explicit_optimizer_time_ms"),
            "gradient_iterations_run": optimizer_loss("gradient_iterations_run"),
            "gradient_max_iters": optimizer_loss("gradient_max_iters"),
            "gradient_samples": optimizer_loss("gradient_samples"),
            "gradient_eps": optimizer_loss("gradient_eps"),
            "gradient_early_stopped": optimizer_loss("gradient_early_stopped"),
            "gradient_candidate_early_stopped": optimizer_loss(
                "gradient_candidate_early_stopped"
            ),
            "gradient_batched_line_search": optimizer_loss(
                "gradient_batched_line_search"
            ),
            "gradient_line_search_batch_evaluations": optimizer_loss(
                "gradient_line_search_batch_evaluations"
            ),
            "gradient_line_search_batch_size": optimizer_loss(
                "gradient_line_search_batch_size"
            ),
            "gradient_jax_scan_used": optimizer_loss("gradient_jax_scan_used"),
            "gradient_jax_scan_used_count": optimizer_loss(
                "gradient_jax_scan_used_count"
            ),
            "gradient_full_jax_scan_used": optimizer_loss(
                "gradient_full_jax_scan_used"
            ),
            "gradient_full_jax_scan_time_ms": optimizer_loss(
                "gradient_full_jax_scan_time_ms"
            ),
            "gradient_initial_records_time_ms": optimizer_loss(
                "gradient_initial_records_time_ms"
            ),
            "gradient_initial_project_time_ms": optimizer_loss(
                "gradient_initial_project_time_ms"
            ),
            "gradient_initial_batch_cost_time_ms": optimizer_loss(
                "gradient_initial_batch_cost_time_ms"
            ),
            "gradient_initial_record_build_time_ms": optimizer_loss(
                "gradient_initial_record_build_time_ms"
            ),
            "gradient_initial_sort_time_ms": optimizer_loss(
                "gradient_initial_sort_time_ms"
            ),
            "gradient_direction_sample_time_ms": optimizer_loss(
                "gradient_direction_sample_time_ms"
            ),
            "gradient_perturb_control_time_ms": optimizer_loss(
                "gradient_perturb_control_time_ms"
            ),
            "gradient_perturb_project_time_ms": optimizer_loss(
                "gradient_perturb_project_time_ms"
            ),
            "gradient_perturb_records_time_ms": optimizer_loss(
                "gradient_perturb_records_time_ms"
            ),
            "gradient_line_control_time_ms": optimizer_loss(
                "gradient_line_control_time_ms"
            ),
            "gradient_line_project_time_ms": optimizer_loss(
                "gradient_line_project_time_ms"
            ),
            "gradient_line_records_time_ms": optimizer_loss(
                "gradient_line_records_time_ms"
            ),
            "deform_gradient_iterations_run": deform_losses.get(
                "gradient_iterations_run"
            ),
            "return_gradient_iterations_run": return_losses.get(
                "gradient_iterations_run"
            ),
            "deform_gradient_initial_records_time_ms": deform_losses.get(
                "gradient_initial_records_time_ms"
            ),
            "return_gradient_initial_records_time_ms": return_losses.get(
                "gradient_initial_records_time_ms"
            ),
            "deform_gradient_initial_batch_cost_time_ms": deform_losses.get(
                "gradient_initial_batch_cost_time_ms"
            ),
            "return_gradient_initial_batch_cost_time_ms": return_losses.get(
                "gradient_initial_batch_cost_time_ms"
            ),
            "cem_iterations_run": return_losses.get(
                "cem_iterations_run",
                deform_losses.get("cem_iterations_run"),
            ),
            "cem_early_stopped": return_losses.get(
                "cem_early_stopped",
                deform_losses.get("cem_early_stopped"),
            ),
            "cem_max_iters": return_losses.get(
                "cem_max_iters",
                deform_losses.get("cem_max_iters"),
            ),
            "cem_population": return_losses.get(
                "cem_population",
                deform_losses.get("cem_population"),
            ),
            "deform_cem_iterations_run": deform_losses.get("cem_iterations_run"),
            "deform_cem_early_stopped": deform_losses.get("cem_early_stopped"),
            "return_cem_iterations_run": return_losses.get("cem_iterations_run"),
            "return_cem_early_stopped": return_losses.get("cem_early_stopped"),
        }
        info.update(replan_info)
        log_fn = logger.info if self.debug_safety_feasibility else logger.debug
        log_fn(
            "explicit SafeChunk recovery: phase=%s cached_active=%s "
            "deform_min_clearance=%.4f deform_accepted=%s recover_min_clearance=%s "
            "recover_rejoin_loss=%.6f recover_target_index=%s recover_accepted=%s "
            "resumed_from_recover_index=%s fallback_used=%s rejection_cause=%s",
            info.get("recovery_phase"),
            info.get("cached_motion_active"),
            info.get("deform_min_clearance_stage"),
            info.get("deform_stage_accepted"),
            info.get("recover_min_clearance"),
            info.get("recover_rejoin_loss"),
            info.get("recover_target_index"),
            info.get("recover_accepted"),
            info.get("resumed_from_recover_index"),
            info.get("fallback_used"),
            info.get("rejection_cause"),
        )
        return info

    def _return_deformation_cost(
        self,
        obs: Any,
        candidate: np.ndarray,
        nominal: np.ndarray,
        nominal_q_seq: np.ndarray,
        rejoin_context: Any,
        action_idx: np.ndarray,
    ) -> CostResult:
        """Score a return chunk by safety, rejoin distance, and smoothness."""
        q_seq = self.rollout_nominal_chunk(obs, candidate)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        required_clearance = float(self._acceptance_clearance_threshold())
        constraint = self.clearance_constraint_from_eval(
            safety_eval,
            q_seq.shape[0],
            required_clearance,
        )
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        safety_loss = constraint.margin_loss
        rejoin_loss, j_best, q_time_ms = self._q_rejoin_loss(
            q_seq,
            nominal_q_seq=nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        action_deviation_loss = (
            float(np.square(candidate[:, action_idx] - nominal[:, action_idx]).mean())
            if len(action_idx)
            else 0.0
        )
        smoothness_loss = self._smoothness_loss(candidate, action_idx)
        ordered_target_index = self._ordered_recovery_start_index(
            j_best,
            q_seq.shape[0],
            nominal_q_seq,
        )
        ordered_terms = self._ordered_recovery_path_terms(
            q_seq,
            nominal_q_seq,
            target_index=ordered_target_index,
            rejoin_context=rejoin_context,
        )
        ordered_loss = float(ordered_terms["recover_ordered_loss"])
        total_loss = float(
            self.lambda_return_rejoin * rejoin_loss
            + self.lambda_return_safety * self.recover_clearance_penalty_scale * safety_loss
            + self.lambda_return_smooth * smoothness_loss
            + self.lambda_return_action * action_deviation_loss
            + ordered_loss
        )
        return total_loss, {
            "safety_loss": safety_loss,
            "recover_required_min_clearance": required_clearance,
            "recover_clearance_margin_loss": safety_loss,
            "rejoin_loss": float(rejoin_loss),
            "return_rejoin_loss": float(rejoin_loss),
            "action_deviation_loss": action_deviation_loss,
            "smoothness_loss": smoothness_loss,
            **ordered_terms,
            "total_loss": total_loss,
            "min_clearance": float(np.min(h_seq)),
            "j_best": j_best,
            "return_target_index": j_best,
            "rejoin_q_eval_time_ms": float(q_time_ms),
        }

    def _perform_post_recovery_act_window(
        self,
        obs: Any,
        chunk: np.ndarray,
        original_shape: Any,
        info: InfoDict,
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Serve a short pass-through window after committed recovery completes."""
        parent = self.parent
        if not parent.post_recovery_act_window_active:
            return None
        remaining = int(parent.post_recovery_act_steps_remaining)
        nominal_became_safe = bool(parent.unsafe_streak > 0 or parent.brake_streak > 0)
        resume_after_wait = bool(
            nominal_became_safe and parent.brake_streak > 0 and remaining > 0
        )
        info.update({"safety_mode": "pass_through", "mode": "pass_through"})
        info.update(
            parent.brake._temporary_streak_info(
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
        parent.post_recovery_act_steps_remaining = max(0, remaining - 1)
        if parent.post_recovery_act_steps_remaining <= 0:
            parent.post_recovery_act_window_active = False
        self.brake._update_last_safe_execution(obs, chunk, info, **kwargs)
        parent.last_info = info
        if resume_after_wait and parent.safechunk_active_safety_enabled and parent.check_hold_horizon_safety:
            info["active_safety_nominal_gate"] = True
            return self._hold_return_or_emergency_deform(
                obs=obs,
                nominal_chunk=chunk,
                braked_chunk=chunk,
                info=info,
                original_shape=original_shape,
                **kwargs,
            )
        return chunk.reshape(original_shape), info

    def _perform_recover_after_temporary_wait(
        self,
        obs: Any,
        chunk: np.ndarray,
        q_seq: np.ndarray | None,
        safety_info: Mapping[str, Any] | None,
        info: InfoDict,
        original_shape: Any,
        waited_unsafe_streak: int,
        waited_brake_streak: int,
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Attempt recovery after a temporary wait/brake period has elapsed."""
        parent = self.parent
        if not (
            self.temporary_blocker_enabled
            and self.temporary_recover_after_wait
            and self.recoverable_deform_enabled
            and self.explicit_return
            and self.deformation_enabled
        ):
            return None
        if waited_brake_streak < self.temporary_recover_after_wait_min_brake_steps:
            return None
        if parent.post_recovery_act_window_active or parent.committed_chunk is not None:
            return None

        recovery_safety = dict(safety_info or {})
        recovery_safety.update({
            "horizon_safe": False,
            "first_violation": 0,
            "unsafe_count": max(1, int(recovery_safety.get("unsafe_count", 0) or 0)),
            "temporary_recover_after_wait_forced": True,
        })
        try:
            recovery_chunk, recovery_info = self.deform.deform_chunk(
                obs,
                chunk,
                safety_info=recovery_safety,
                braked_chunk=chunk,
                nominal_q_seq=q_seq,
                **kwargs,
            )
        except Exception as exc:
            info.update(
                {
                    "temporary_recover_after_wait_attempted": True,
                    "temporary_recover_after_wait_accepted": False,
                    "temporary_recover_after_wait_error": type(exc).__name__,
                    "temporary_recover_after_wait_error_message": str(exc),
                }
            )
            parent.last_info = info
            return None

        info.update(recovery_info)
        info.update(
            {
                "temporary_recover_after_wait_attempted": True,
                "temporary_recover_after_wait_waited_unsafe_streak": int(
                    waited_unsafe_streak
                ),
                "temporary_recover_after_wait_waited_brake_streak": int(
                    waited_brake_streak
                ),
                "temporary_recover_after_wait_accepted": bool(
                    recovery_info.get("optimized_accepted", False)
                ),
                "temporary_recover_after_wait_recover_accepted": bool(
                    recovery_info.get("recover_accepted", False)
                ),
            }
        )
        if not bool(recovery_info.get("optimized_accepted", False)):
            return None

        committed, commit_reject_info = self._commit_explicit_recovery_chunk(
            obs,
            recovery_chunk,
            info,
            **kwargs,
        )
        if not committed:
            info.update(commit_reject_info)
            info.update(
                {"temporary_recover_after_wait_committed": False,
                    "temporary_recover_after_wait_commit_rejected": True,
                }
            )
            return None

        parent.unsafe_streak = 0
        parent.brake_streak = 0
        parent.recovery_failure_streak = 0
        info.update(
            {
                "temporary_recover_after_wait_committed": True,
                "safety_mode": "horizon_deform",
                "mode": "temporary_recover_after_wait",
                "deform_mode": "temporary_recover_after_wait",
                "deformation_source": "temporary_recover_after_wait",
            }
        )
        committed_result = self._serve_committed_chunk(
            obs,
            chunk,
            original_shape,
            **kwargs,
        )
        if committed_result is None:
            return None
        committed_chunk, committed_info = committed_result
        committed_info.update({k: v for k, v in info.items() if k not in committed_info})
        committed_info.update(
            {
                "temporary_recover_after_wait_attempted": True,
                "temporary_recover_after_wait_accepted": True,
                "temporary_recover_after_wait_committed": True,
            }
        )
        parent.last_info = committed_info
        return committed_chunk, committed_info

    def _temporary_blocker_config(self, config: Any) -> ConfigDict:
        """Normalize temporary blocker config with defaults."""
        cfg: ConfigDict = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "recover_after_wait": bool(cfg.get("recover_after_wait", False)),
            "recover_after_wait_min_brake_steps": int(
                cfg.get("recover_after_wait_min_brake_steps", 1)
            ),
        }

    def _recoverable_optimization_config(
        self,
        config: Any,
        **defaults: Any,
    ) -> ConfigDict:
        """Normalize optimizer config for deform/return candidate search."""
        cfg: ConfigDict = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        scales = tuple(
            float(value)
            for value in cfg.get(
                "gradient_line_search_scales",
                defaults.get("gradient_line_search_scales", (1.0, 0.5, 0.25)),
            )
            if float(value) > 0.0
        )
        return {
            "optimizer_method": str(
                cfg.get("optimizer_method", defaults.get("optimizer_method", "cem"))
            ).lower(),
            "opt_iters": max(0, int(cfg.get("opt_iters", defaults.get("opt_iters", 20))),
            ),
            "opt_lr": max(1e-9, float(cfg.get("opt_lr", defaults.get("opt_lr", 0.03)))),
            "opt_population": max(4, int(cfg.get("opt_population", defaults.get("opt_population", 32)))),
            "opt_elite_frac": float(cfg.get("opt_elite_frac", defaults.get("opt_elite_frac", 0.25))),
            "opt_seed": (None if cfg.get("opt_seed", defaults.get("opt_seed", 0)) is None else int(cfg.get("opt_seed", defaults.get("opt_seed", 0)))),
            "jax_batched_optimizer": bool(cfg.get("jax_batched_optimizer", defaults.get("jax_batched_optimizer", True))),
            "jax_batched_optimizer_fallback": bool(cfg.get("jax_batched_optimizer_fallback", defaults.get("jax_batched_optimizer_fallback", True))),
            "gradient_samples": max(1, int(cfg.get("gradient_samples", defaults.get("gradient_samples", 4))),
            ),
            "gradient_eps": max(1e-9, float(cfg.get("gradient_eps", defaults.get("gradient_eps", 0.01)))),
            "gradient_adam_beta1": float(cfg.get("gradient_adam_beta1", defaults.get("gradient_adam_beta1", 0.9))),
            "gradient_adam_beta2": float(cfg.get("gradient_adam_beta2", defaults.get("gradient_adam_beta2", 0.999))),
            "gradient_min_improvement": max(0.0, float(cfg.get("gradient_min_improvement", defaults.get("gradient_min_improvement", 1e-6)))),
            "gradient_line_search_scales": scales or (1.0, 0.5, 0.25),
            "gradient_batched_line_search": bool(cfg.get("gradient_batched_line_search", defaults.get("gradient_batched_line_search", True))),
            "gradient_early_stop_on_candidate": bool(
                cfg.get(
                    "gradient_early_stop_on_path",
                    cfg.get("gradient_early_stop_on_candidate", defaults.get("gradient_early_stop_on_candidate", True)),
                )
            ),
        }

    def _safechunk_replan_config(self, config: Any) -> ConfigDict:
        """Normalize stale-recovery replan policy config."""
        cfg: ConfigDict = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        target_mode = str(cfg.get("recovery_target_mode", "task_progress")).lower()
        if target_mode not in {"task_progress", "nominal"}:
            raise ValueError(
                "safechunk_replan.recovery_target_mode must be one of "
                "['task_progress', 'nominal'], got "
                f"{target_mode}"
            )
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "replan_deform_from_current_state": bool(
                cfg.get("replan_deform_from_current_state", True)
            ),
            "replan_recovery_from_current_state": bool(
                cfg.get("replan_recovery_from_current_state", True)
            ),
            "suppress_stale_return": bool(cfg.get("suppress_stale_return", True)),
            "max_recovery_failure_before_replan": int(
                cfg.get("max_recovery_failure_before_replan", 1)
            ),
            "allow_recovery_to_nominal_only_if_feasible": bool(
                cfg.get("allow_recovery_to_nominal_only_if_feasible", True)
            ),
            "recovery_target_mode": target_mode,
            "clear_failed_recovery_on_nominal_safe": bool(
                cfg.get("clear_failed_recovery_on_nominal_safe", True)
            ),
        }

    def _recoverable_deform_config(
        self,
        config: Any,
        intervention: Mapping[str, Any] | None = None,
        **defaults: Any,
    ) -> ConfigDict:
        """Merge deform/rejoin config with intervention overrides."""
        cfg: ConfigDict = {}
        if config is not None:
            if hasattr(config, 'items'):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))

        intervention_cfg: ConfigDict = {}
        if intervention is not None:
            if hasattr(intervention, 'items'):
                intervention_cfg.update(dict(intervention.items()))
            else:
                intervention_cfg.update(dict(intervention))

        merged_cfg: ConfigDict = dict(cfg)
        merged_cfg.update(intervention_cfg)

        inner_metric = str(
            merged_cfg.get(
                'inner_rejoin_metric',
                'ee_pose' if bool(merged_cfg.get('ee_rejoin_in_inner_loop', False)) else 'q_state',
            )
        ).lower()
        final_metric = str(merged_cfg.get('final_rejoin_metric', 'ee_pose')).lower()
        if bool(merged_cfg.get('use_ee_pose_rejoin', False)) and bool(
            merged_cfg.get('ee_rejoin_in_inner_loop', False)
        ):
            inner_metric = 'ee_pose'
        if inner_metric not in {'q_state', 'ee_pose'}:
            raise ValueError(
                f"recoverable_deform.inner_rejoin_metric must be one of ['q_state', 'ee_pose'], got {inner_metric}"
            )
        if final_metric not in {'none', 'q_state', 'ee_pose'}:
            raise ValueError(
                f"recoverable_deform.final_rejoin_metric must be one of ['none', 'q_state', 'ee_pose'], got {final_metric}"
            )
        ee_rejoin_in_inner_loop = bool(
            merged_cfg.get('ee_rejoin_in_inner_loop', inner_metric == 'ee_pose')
        )
        if inner_metric == 'ee_pose':
            ee_rejoin_in_inner_loop = True

        rejoin_threshold = float(
            merged_cfg.get('rejoin_threshold', defaults['rejoin_threshold'])
        )
        return {
            'enabled': bool(merged_cfg.get('enabled', True)),
            'lambda_rejoin': float(merged_cfg.get('lambda_rejoin', defaults['lambda_rejoin'])),
            'rejoin_threshold': rejoin_threshold,
            'q_rejoin_threshold': float(
                merged_cfg.get('q_rejoin_threshold', merged_cfg.get('rejoin_threshold', 0.5))
            ),
            'qd_rejoin_threshold': float(merged_cfg.get('qd_rejoin_threshold', 5.0)),
            'require_qd_rejoin': bool(merged_cfg.get('require_qd_rejoin', False)),
            'qd_rejoin_hard_threshold': float(
                merged_cfg.get(
                    'qd_rejoin_hard_threshold',
                    merged_cfg.get('qd_rejoin_threshold', 5.0) * 4.0,
                )
            ),
            'ee_rejoin_threshold': float(merged_cfg.get('ee_rejoin_threshold', 0.08)),
            'min_rejoin_offset': int(merged_cfg.get('min_rejoin_offset', defaults['min_rejoin_offset'])),
            'use_ee_pose_rejoin': bool(
                merged_cfg.get(
                    'use_ee_pose_rejoin',
                    defaults['use_ee_pose_rejoin'],
                )
            ),
            'use_object_state_rejoin': bool(
                merged_cfg.get(
                    'use_object_state_rejoin',
                    defaults['use_object_state_rejoin'],
                )
            ),
            'brake_if_unrecoverable': bool(
                merged_cfg.get(
                    'brake_if_unrecoverable',
                    defaults['brake_if_unrecoverable'],
                )
            ),
            'inner_rejoin_metric': inner_metric,
            'final_rejoin_metric': final_metric,
            'cache_nominal_ee': bool(merged_cfg.get('cache_nominal_ee', True)),
            'ee_rejoin_in_inner_loop': ee_rejoin_in_inner_loop,
            'q_rejoin_weights': merged_cfg.get('q_rejoin_weights'),
            'explicit_return': bool(
                self._deprecated_config_value(
                    merged_cfg,
                    'explicit_return',
                    'explicit_recovery',
                    False,
                )
            ),
            'acceptance_clearance_tol': float(merged_cfg.get('acceptance_clearance_tol', 0.005)),
            'lambda_deform_safety': float(merged_cfg.get('lambda_deform_safety', 800.0)),
            'lambda_deform_action': float(merged_cfg.get('lambda_deform_action', 0.1)),
            'lambda_deform_smooth': float(merged_cfg.get('lambda_deform_smooth', 0.1)),
            'lambda_retreat': float(merged_cfg.get('lambda_retreat', 1.0)),
            'lambda_return_safety': float(
                self._deprecated_config_value(
                    merged_cfg,
                    'lambda_recover_safety',
                    'lambda_return_safety',
                    500.0,
                )
            ),
            'lambda_return_rejoin': float(
                self._deprecated_config_value(
                    merged_cfg,
                    'lambda_recover_rejoin',
                    'lambda_return_rejoin',
                    5.0,
                )
            ),
            'lambda_return_smooth': float(
                self._deprecated_config_value(
                    merged_cfg,
                    'lambda_recover_smooth',
                    'lambda_return_smooth',
                    0.2,
                )
            ),
            'lambda_return_action': float(
                self._deprecated_config_value(
                    merged_cfg,
                    'lambda_recover_action',
                    'lambda_return_action',
                    0.1,
                )
            ),
            'deform_horizon': int(merged_cfg.get('deform_horizon', defaults.get('deform_horizon', 4))),
            'return_horizon': int(
                self._deprecated_config_value(
                    merged_cfg,
                    'recover_horizon',
                    'return_horizon',
                    defaults.get('return_horizon', 8),
                )
            ),
            'max_return_retries': int(
                self._deprecated_config_value(
                    merged_cfg,
                    'max_recover_retries',
                    'max_return_retries',
                    3,
                )
            ),
            'use_ee_final_check': bool(merged_cfg.get('use_ee_final_check', True)),
        }
    def _explicit_recovery_config(self, config: Any) -> ConfigDict:
        """Normalize committed explicit-recovery execution config."""
        cfg: ConfigDict = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        return {
            "commit_accepted_chunks": bool(cfg.get("commit_accepted_chunks", True)),
            "committed_chunk_safety_check": bool(
                cfg.get("committed_chunk_safety_check", True)
            ),
            "committed_safety_tol": float(cfg.get("committed_safety_tol", 0.005)),
            "committed_abort_only_if_contact_risk": bool(
                cfg.get("committed_abort_only_if_contact_risk", True)
            ),
            "committed_min_clearance_for_abort": float(
                cfg.get("committed_min_clearance_for_abort", 0.08)
            ),
            "committed_deform_min_clearance_for_abort": cfg.get(
                "committed_deform_min_clearance_for_abort", None
            ),
            "repair_committed_action": bool(cfg.get("repair_committed_action", True)),
            "monotonic_committed_repair": bool(
                cfg.get("monotonic_committed_repair", True)
            ),
            "committed_execution_margin": float(
                cfg.get("committed_execution_margin", 0.02)
            ),
            "committed_state_error_threshold": float(
                cfg.get("committed_state_error_threshold", 0.25)
            ),
            "committed_state_error_action": str(
                cfg.get("committed_state_error_action", "replan")
            ).lower(),
            "committed_state_mismatch_abort_requires_unsafe": bool(
                cfg.get("committed_state_mismatch_abort_requires_unsafe", False)
            ),
            "replan_committed_suffix_on_state_mismatch": bool(
                cfg.get("replan_committed_suffix_on_state_mismatch", True)
            ),
            "committed_suffix_replan_min_remaining": int(
                cfg.get("committed_suffix_replan_min_remaining", 1)
            ),
            "opportunistic_act_resume": bool(
                cfg.get("opportunistic_act_resume", True)
            ),
            "opportunistic_resume_q_threshold": cfg.get(
                "opportunistic_resume_q_threshold", None
            ),
            "opportunistic_resume_min_clearance": float(
                cfg.get("opportunistic_resume_min_clearance", 0.08)
            ),
            "max_recover_steps_before_act_resume": int(
                cfg.get("max_recover_steps_before_act_resume", 8)
            ),
            "max_suffix_replans_per_recovery": int(
                cfg.get("max_suffix_replans_per_recovery", 3)
            ),
            "committed_rollout_gain": float(cfg.get("committed_rollout_gain", 1.0)),
            "closed_loop_recovery_tracking": bool(
                cfg.get("closed_loop_recovery_tracking", True)
            ),
            "adaptive_committed_rollout_gain": bool(
                cfg.get("adaptive_committed_rollout_gain", True)
            ),
            "rollout_gain_ema": float(cfg.get("rollout_gain_ema", 0.1)),
            "rollout_gain_min": float(cfg.get("rollout_gain_min", 0.2)),
            "rollout_gain_max": float(cfg.get("rollout_gain_max", 1.5)),
            "cancel_committed_on_nominal_safe": bool(
                cfg.get("cancel_committed_on_nominal_safe", True)
            ),
            "committed_cancel_min_clearance": float(
                cfg.get("committed_cancel_min_clearance", 0.08)
            ),
            "mpc_recovery_enabled": bool(cfg.get("mpc_recovery_enabled", False)),
            "mpc_recovery_horizon": int(cfg.get("mpc_recovery_horizon", 8)),
            "mpc_recovery_prefix_len": int(cfg.get("mpc_recovery_prefix_len", 2)),
            "committed_receding_recover_steps": int(
                cfg.get("committed_receding_recover_steps", 1)
            ),
            "committed_nominal_tube_tracking_enabled": bool(
                cfg.get("committed_nominal_tube_tracking_enabled", True)
            ),
            "committed_nominal_tube_tracking_arm_gain": float(
                cfg.get("committed_nominal_tube_tracking_arm_gain", 0.45)
            ),
            "committed_nominal_tube_tracking_base_gain": float(
                cfg.get("committed_nominal_tube_tracking_base_gain", 0.75)
            ),
            "committed_nominal_tube_tracking_max_arm_step": float(
                cfg.get("committed_nominal_tube_tracking_max_arm_step", 0.24)
            ),
            "committed_nominal_tube_tracking_max_base_delta": float(
                cfg.get("committed_nominal_tube_tracking_max_base_delta", 0.06)
            ),
            "committed_nominal_tube_tracking_done_threshold": float(
                cfg.get("committed_nominal_tube_tracking_done_threshold", 0.24)
            ),
            "committed_nominal_tube_tracking_max_recover_steps": int(
                cfg.get("committed_nominal_tube_tracking_max_recover_steps", 0)
            ),
            "committed_nominal_tube_tracking_rollout_solver": bool(
                cfg.get("committed_nominal_tube_tracking_rollout_solver", True)
            ),
            "committed_nominal_tube_tracking_score_scale": float(
                cfg.get("committed_nominal_tube_tracking_score_scale", 0.35)
            ),
            "committed_nominal_tube_tracking_action_smooth_weight": float(
                cfg.get("committed_nominal_tube_tracking_action_smooth_weight", 0.0)
            ),
            "committed_nominal_tube_tracking_heading_weight": float(
                cfg.get("committed_nominal_tube_tracking_heading_weight", 0.15)
            ),
            "committed_nominal_tube_tracking_window_heading_weight": float(
                cfg.get("committed_nominal_tube_tracking_window_heading_weight", 0.10)
            ),
            "mpc_recovery_max_replans_per_recovery": int(
                cfg.get("mpc_recovery_max_replans_per_recovery", 0)
            ),
            "mpc_recovery_require_ordered_progress": bool(
                cfg.get("mpc_recovery_require_ordered_progress", True)
            ),
            "mpc_recovery_require_live_progress": bool(
                cfg.get("mpc_recovery_require_live_progress", True)
            ),
            "mpc_recovery_min_progress_delta": float(
                cfg.get("mpc_recovery_min_progress_delta", 0.0001)
            ),
            "mpc_recovery_no_progress_limit": int(
                cfg.get("mpc_recovery_no_progress_limit", 3)
            ),
            "mpc_recovery_target_tube_enabled": bool(
                cfg.get("mpc_recovery_target_tube_enabled", True)
            ),
            "mpc_recovery_target_tube_radius": cfg.get(
                "mpc_recovery_target_tube_radius", None
            ),
            "mpc_recovery_target_tube_weight": float(
                cfg.get("mpc_recovery_target_tube_weight", 50.0)
            ),
            "mpc_recovery_target_tube_require_progress": bool(
                cfg.get("mpc_recovery_target_tube_require_progress", True)
            ),
            "mpc_recovery_target_tube_window_len": int(
                cfg.get("mpc_recovery_target_tube_window_len", cfg.get("act_frame_stack", 4))
            ),
            "mpc_recovery_target_tube_window_weight": float(
                cfg.get("mpc_recovery_target_tube_window_weight", 1.0)
            ),
            "mpc_recovery_target_tube_window_dq_weight": float(
                cfg.get("mpc_recovery_target_tube_window_dq_weight", 0.5)
            ),
            "mpc_recovery_target_tube_window_action_weight": float(
                cfg.get("mpc_recovery_target_tube_window_action_weight", 0.25)
            ),
            "mpc_recovery_target_tube_window_max_q_frame_l2_mean": float(
                cfg.get("mpc_recovery_target_tube_window_max_q_frame_l2_mean", 0.24)
            ),
            "mpc_recovery_target_tube_window_max_q_frame_l2_max": float(
                cfg.get("mpc_recovery_target_tube_window_max_q_frame_l2_max", 0.32)
            ),
            "mpc_recovery_target_tube_window_min_dq_cosine": float(
                cfg.get("mpc_recovery_target_tube_window_min_dq_cosine", 0.20)
            ),
            "mpc_recovery_target_tube_window_max_step_l2_error": float(
                cfg.get("mpc_recovery_target_tube_window_max_step_l2_error", 0.12)
            ),
            "mpc_bridge_replan_cooldown_steps": int(
                cfg.get("mpc_bridge_replan_cooldown_steps", 1)
            ),
            "mpc_bridge_max_replans_per_recovery": int(
                cfg.get("mpc_bridge_max_replans_per_recovery", 4)
            ),
            "mpc_bridge_direction_hard_gate": bool(
                cfg.get("mpc_bridge_direction_hard_gate", False)
            ),
            "mpc_bridge_min_heading_improvement": float(
                cfg.get("mpc_bridge_min_heading_improvement", 0.05)
            ),
            "mpc_bridge_min_progress_improvement": float(
                cfg.get("mpc_bridge_min_progress_improvement", 0.01)
            ),
            "mpc_bridge_min_clearance_improvement": float(
                cfg.get("mpc_bridge_min_clearance_improvement", 0.01)
            ),
            "mpc_handoff_action_agreement_override_enabled": bool(
                cfg.get("mpc_handoff_action_agreement_override_enabled", False)
            ),
            "mpc_handoff_action_agreement_source": str(
                cfg.get("mpc_handoff_action_agreement_source", "release")
            ).lower(),
            "mpc_handoff_require_resume_readiness": bool(
                cfg.get("mpc_handoff_require_resume_readiness", False)
            ),
            "mpc_handoff_use_selected_act_action_if_safe": bool(
                cfg.get("mpc_handoff_use_selected_act_action_if_safe", False)
            ),
            "mpc_handoff_bridge_ramp_on_resume_not_ready": bool(
                cfg.get("mpc_handoff_bridge_ramp_on_resume_not_ready", False)
            ),
            "mpc_handoff_bridge_ramp_max_steps": int(
                cfg.get("mpc_handoff_bridge_ramp_max_steps", 0)
            ),
            "mpc_handoff_shadow_prefix_len": int(
                cfg.get("mpc_handoff_shadow_prefix_len", 4)
            ),
            "mpc_handoff_require_shadow_prefix": bool(
                cfg.get("mpc_handoff_require_shadow_prefix", False)
            ),
            "mpc_handoff_action_l2_threshold": float(
                cfg.get("mpc_handoff_action_l2_threshold", 0.15)
            ),
            "mpc_handoff_action_cosine_threshold": float(
                cfg.get("mpc_handoff_action_cosine_threshold", 0.98)
            ),
            "mpc_handoff_action_arm_l2_threshold": float(
                cfg.get("mpc_handoff_action_arm_l2_threshold", 0.15)
            ),
            "extend_recovery_budget_on_progress": bool(
                cfg.get("extend_recovery_budget_on_progress", True)
            ),
            "max_recover_steps_with_progress": int(
                cfg.get("max_recover_steps_with_progress", 24)
            ),
            "recovery_budget_progress_epsilon": float(
                cfg.get("recovery_budget_progress_epsilon", 0.02)
            ),
            "recovery_budget_no_progress_limit": int(
                cfg.get("recovery_budget_no_progress_limit", 2)
            ),
        }

    def _safechunk_recovery_corridor_config(self, config: Any) -> ConfigDict:
        """Normalize corridor and detour policy config."""
        cfg: ConfigDict = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "require_recover_path_safe": bool(
                cfg.get("require_recover_path_safe", True)
            ),
            "recover_path_min_clearance": float(
                cfg.get("recover_path_min_clearance", 0.04)
            ),
            "recover_immediate_hard_clearance": float(
                cfg.get("recover_immediate_hard_clearance", 0.02)
            ),
            "recover_prefix_min_clearance": float(
                cfg.get("recover_prefix_min_clearance", 0.04)
            ),
            "enable_direct_rejoin": bool(cfg.get("enable_direct_rejoin", True)),
            "enable_recovery_bridge_seeds": bool(
                cfg.get(
                    "enable_recovery_bridge_seeds",
                    cfg.get("enable_detour_rejoin", False),
                )
            ),
            "enable_detour_rejoin": bool(cfg.get("enable_detour_rejoin", False)),
            "enable_delayed_rejoin": bool(cfg.get("enable_delayed_rejoin", True)),
            "suppress_repeated_unsafe_recovery": bool(
                cfg.get("suppress_repeated_unsafe_recovery", True)
            ),
            "unsafe_recovery_cooldown_steps": int(
                cfg.get("unsafe_recovery_cooldown_steps", 8)
            ),
            "max_same_target_failures": int(cfg.get("max_same_target_failures", 2)),
            "bridge_seed_scales": tuple(
                cfg.get(
                    "bridge_seed_scales",
                    cfg.get("detour_scales", (0.25, 0.5, 0.75, 1.0)),
                )
            ),
            "detour_scales": tuple(
                cfg.get("detour_scales", (0.25, 0.5, 0.75, 1.0))
            ),
            "detour_clearance_weight": float(
                cfg.get("detour_clearance_weight", 100.0)
            ),
            "detour_task_rejoin_weight": float(
                cfg.get("detour_task_rejoin_weight", 10.0)
            ),
            "detour_action_norm_weight": float(
                cfg.get("detour_action_norm_weight", 0.2)
            ),
            "delayed_rejoin_wait_steps": int(
                cfg.get("delayed_rejoin_wait_steps", 4)
            ),
            "delayed_rejoin_requires_nominal_prefix_safe": bool(
                cfg.get("delayed_rejoin_requires_nominal_prefix_safe", True)
            ),
            "require_safe_corridor_for_recovery_complete": bool(
                cfg.get("require_safe_corridor_for_recovery_complete", True)
            ),
            "require_post_recovery_act_window": bool(
                cfg.get("require_post_recovery_act_window", True)
            ),
            "post_recovery_min_act_steps": int(
                cfg.get("post_recovery_min_act_steps", 5)
            ),
        }

    def _safechunk_recover_config(self, config: Any) -> ConfigDict:
        """Normalize task-progress recovery objective config."""
        cfg: ConfigDict = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        schedule = str(cfg.get("rejoin_weight_schedule", "ramp")).lower()
        if schedule not in {"constant", "none", "ramp"}:
            raise ValueError(
                "safechunk_recover.rejoin_weight_schedule must be one of "
                "['constant', 'none', 'ramp'], got "
                f"{schedule}"
            )
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "rejoin_nominal_weight": float(cfg.get("rejoin_nominal_weight", 5.0)),
            "task_progress_weight": float(cfg.get("task_progress_weight", 10.0)),
            "act_progress_weight": float(cfg.get("act_progress_weight", 15.0)),
            "act_heading_weight": float(cfg.get("act_heading_weight", 10.0)),
            "min_act_heading_cosine": float(cfg.get("min_act_heading_cosine", 0.25)),
            "direction_alignment_weight": float(
                cfg.get("direction_alignment_weight", 5.0)
            ),
            "min_direction_cosine": float(cfg.get("min_direction_cosine", 0.05)),
            "require_direction_alignment": bool(
                cfg.get("require_direction_alignment", True)
            ),
            "direction_alignment_margin": float(
                cfg.get("direction_alignment_margin", 0.0)
            ),
            "ordered_pose_weight": float(cfg.get("ordered_pose_weight", 2.0)),
            "ordered_delta_weight": float(cfg.get("ordered_delta_weight", 1.0)),
            "ordered_heading_weight": float(cfg.get("ordered_heading_weight", 8.0)),
            "ordered_pose_threshold": float(cfg.get("ordered_pose_threshold", 0.02)),
            "ordered_delta_threshold": float(cfg.get("ordered_delta_threshold", 0.005)),
            "ordered_heading_cosine_threshold": float(
                cfg.get("ordered_heading_cosine_threshold", 0.75)
            ),
            "ordered_backtrack_tolerance": int(cfg.get("ordered_backtrack_tolerance", 0)),
            "require_ordered_path": bool(cfg.get("require_ordered_path", True)),
            "retry_cooldown_steps": int(cfg.get("retry_cooldown_steps", 0)),
            "max_attempts_per_unsafe_streak": int(
                cfg.get("max_attempts_per_unsafe_streak", 0)
            ),
            "reset_attempts_after_brake_timeout": bool(
                cfg.get("reset_attempts_after_brake_timeout", True)
            ),
            "reset_attempts_brake_timeout_steps": int(
                cfg.get("reset_attempts_brake_timeout_steps", 4)
            ),
            "reset_attempts_min_hold_clearance": cfg.get(
                "reset_attempts_min_hold_clearance", None
            ),
            "reset_attempts_require_safe_hold": bool(
                cfg.get("reset_attempts_require_safe_hold", True)
            ),
            "safety_weight": float(cfg.get("safety_weight", 100.0)),
            "clearance_penalty_scale": float(cfg.get("clearance_penalty_scale", 5.0)),
            "task_progress_clearance_penalty_scale": float(
                cfg.get("task_progress_clearance_penalty_scale", 20.0)
            ),
            "action_deviation_weight": float(cfg.get("action_deviation_weight", 0.2)),
            "smoothness_weight": float(cfg.get("smoothness_weight", 0.1)),
            "action_rate_limit": float(cfg.get("action_rate_limit", 0.0)),
            "action_rate_limit_weight": float(cfg.get("action_rate_limit_weight", 0.0)),
            "require_nominal_prefix_safe_for_rejoin": bool(
                cfg.get("require_nominal_prefix_safe_for_rejoin", True)
            ),
            "nominal_rejoin_prefix_min_clearance": float(
                cfg.get("nominal_rejoin_prefix_min_clearance", 0.04)
            ),
            "resume_window_len": int(cfg.get("resume_window_len", cfg.get("act_frame_stack", 4))),
            "resume_window_weight": float(cfg.get("resume_window_weight", 1.0)),
            "resume_window_dq_weight": float(cfg.get("resume_window_dq_weight", 0.5)),
            "resume_window_action_weight": float(cfg.get("resume_window_action_weight", 0.25)),
            "resume_window_max_q_frame_l2_mean": float(
                cfg.get("resume_window_max_q_frame_l2_mean", 0.24)
            ),
            "resume_window_max_q_frame_l2_max": float(
                cfg.get("resume_window_max_q_frame_l2_max", 0.32)
            ),
            "resume_window_min_dq_cosine": float(
                cfg.get("resume_window_min_dq_cosine", 0.20)
            ),
            "resume_window_max_step_l2_error": float(
                cfg.get("resume_window_max_step_l2_error", 0.12)
            ),
            "resume_tube_weight": float(cfg.get("resume_tube_weight", 1.5)),
            "resume_tube_min_score": float(cfg.get("resume_tube_min_score", 0.6)),
            "resume_tube_min_component_score": float(
                cfg.get("resume_tube_min_component_score", 0.35)
            ),
            "resume_tube_distance_scale": float(
                cfg.get("resume_tube_distance_scale", 0.75)
            ),
            "resume_tube_min_clearance": float(
                cfg.get("resume_tube_min_clearance", cfg.get("nominal_rejoin_prefix_min_clearance", 0.04))
            ),
            "resume_affordance_enabled": bool(
                cfg.get("resume_affordance_enabled", True)
            ),
            "resume_affordance_weight": float(
                cfg.get("resume_affordance_weight", 1.0)
            ),
            "resume_affordance_min_score": float(
                cfg.get("resume_affordance_min_score", 0.45)
            ),
            "resume_affordance_min_component_score": float(
                cfg.get("resume_affordance_min_component_score", 0.25)
            ),
            "resume_affordance_required_for_accept": bool(
                cfg.get("resume_affordance_required_for_accept", True)
            ),
            "resume_affordance_min_component_for_accept": float(
                cfg.get(
                    "resume_affordance_min_component_for_accept",
                    cfg.get("resume_affordance_min_component_score", 0.25),
                )
            ),
            "resume_affordance_target_distance_good": float(
                cfg.get("resume_affordance_target_distance_good", 0.12)
            ),
            "resume_affordance_target_distance_scale": float(
                cfg.get("resume_affordance_target_distance_scale", 0.45)
            ),
            "resume_affordance_progress_scale": float(
                cfg.get("resume_affordance_progress_scale", 0.10)
            ),
            "resume_affordance_progress_epsilon": float(
                cfg.get("resume_affordance_progress_epsilon", 0.005)
            ),
            "resume_affordance_progress_distance_gain": float(
                cfg.get("resume_affordance_progress_distance_gain", 1.0)
            ),
            "resume_affordance_taskspace_in_optimizer": bool(
                cfg.get("resume_affordance_taskspace_in_optimizer", False)
            ),
            "resume_affordance_terminal_distance_weight": float(
                cfg.get("resume_affordance_terminal_distance_weight", 8.0)
            ),
            "act_frame_stack": int(cfg.get("act_frame_stack", cfg.get("frame_stack", 4))),
            "use_latest_nominal_for_rejoin": bool(
                cfg.get("use_latest_nominal_for_rejoin", True)
            ),
            "suppress_stale_nominal_rejoin": bool(
                cfg.get("suppress_stale_nominal_rejoin", True)
            ),
            "rejoin_weight_schedule": schedule,
            "rejoin_ramp_steps": int(cfg.get("rejoin_ramp_steps", 5)),
        }

    def _clear_committed_chunk(self) -> None:
        """Drop any in-progress committed recovery plan."""
        self.committed_chunk = None
        self.committed_chunk_index = 0
        self.committed_chunk_mode = None
        self.committed_chunk_modes = []
        self.committed_rejoin_index = None
        self.committed_until_complete = False
        self.committed_planned_q_seq = None
        self.committed_planned_post_q_seq = None
        self.committed_planned_h_seq = None
        self.committed_planned_min_clearance_seq = None
        self.committed_planned_clearance_pre_seq = None
        self.committed_planned_clearance_post_seq = None
        self.committed_planned_actions = None
        self.committed_accepted_min_clearance = None
        self.committed_accepted_clearance_margin = None
        self.committed_accepted_human_state_snapshot = None
        self.committed_planning_human_state_snapshot = None
        self.committed_rejoin_diagnostics = {}

    def _pop_pending_committed_replan_info(self) -> Any:
        """Pop pending committed-replan diagnostics owned by recovery state."""
        info = self._pending_committed_replan_info
        self._pending_committed_replan_info = None
        return info


    def _make_mpc_state_weights(
        self,
        configured: Any,
        *,
        base_weight: float = 1.0,
        base_yaw_weight: float = 0.5,
        arm_weight: float = 1.0,
        ignore_indices: Any = (2,),
    ) -> np.ndarray:
        """Build per-q-dimension weights for MPC state comparisons.

        Pelvis z is excluded by default because it is not a useful handoff/tracking
        coordinate for the floating-base Bigym controller.  The rollout still
        predicts it; it just does not dominate state-mismatch or handoff gates.
        """
        dim = max(1, int(getattr(self, "expected_motion_dim", 14)))
        weights = np.full((dim,), float(arm_weight), dtype=np.float32)
        base_dim = min(4, dim)
        if base_dim > 0:
            weights[:base_dim] = float(base_weight)
        if dim > 3:
            weights[3] = float(base_yaw_weight)
        if configured is not None:
            try:
                arr = np.asarray(configured, dtype=np.float32).reshape(-1)
                if arr.size == dim:
                    weights = arr.copy()
                elif arr.size == np.asarray(self.controlled_state_indices).reshape(-1).size:
                    weights = np.zeros((dim,), dtype=np.float32)
                    controlled = np.asarray(self.controlled_state_indices, dtype=np.int64).reshape(-1)
                    valid = controlled < dim
                    weights[controlled[valid]] = arr[valid]
                else:
                    logger.warning(
                        "mpc_state weights length %d does not match state dim %d or controlled dim %d; using defaults.",
                        int(arr.size),
                        int(dim),
                        int(np.asarray(self.controlled_state_indices).reshape(-1).size),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Invalid mpc state weights; using defaults: %s", exc)
        try:
            ignored = {int(x) for x in (ignore_indices or [])}
        except Exception:  # noqa: BLE001
            ignored = set()
        for idx in ignored:
            if 0 <= idx < dim:
                weights[idx] = 0.0
        weights = np.where(np.isfinite(weights), weights, 0.0).astype(np.float32)
        return np.maximum(weights, 0.0)

    def _mpc_state_indices_and_weights(
        self,
        *state_dims: int,
        kind: str = "state",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return controlled q indices and weights for MPC comparisons."""
        dims = [int(d) for d in state_dims if d is not None]
        if not dims:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
        limit = min(dims)
        weights_source = (
            self.mpc_handoff_state_weights
            if kind == "handoff"
            else self.mpc_state_error_weights
        )
        controlled = np.asarray(self.controlled_state_indices, dtype=np.int64).reshape(-1)
        valid = (controlled < limit) & (controlled < weights_source.shape[0])
        idx = controlled[valid]
        weights = weights_source[idx].astype(np.float32, copy=True)
        keep = weights > 0.0
        return idx[keep], weights[keep]

    def _weighted_state_error(
        self,
        actual: Any,
        expected: Any,
        *,
        kind: str = "state",
    ) -> tuple[float | None, InfoDict, np.ndarray, np.ndarray, np.ndarray | None]:
        """Compute weighted controlled-state error with base/arm diagnostics."""
        try:
            actual_arr = np.asarray(actual, dtype=np.float32).reshape(-1)
            expected_arr = np.asarray(expected, dtype=np.float32).reshape(-1)
        except Exception:  # noqa: BLE001
            return None, {}, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), None
        idx, weights = self._mpc_state_indices_and_weights(
            actual_arr.shape[0],
            expected_arr.shape[0],
            kind=kind,
        )
        if idx.size == 0:
            return None, {}, idx, weights, None
        diff = actual_arr[idx] - expected_arr[idx]
        weighted = diff * weights
        base_mask = idx < min(4, int(getattr(self, "expected_motion_dim", 14)))
        arm_mask = ~base_mask

        def _l2(values: np.ndarray) -> float:
            return float(np.linalg.norm(values)) if values.size else 0.0

        stats: InfoDict = {
            "committed_state_error_metric": f"weighted_controlled_q_{kind}",
            "committed_state_error_indices": idx.astype(int).tolist(),
            "committed_state_error_weights": weights.astype(float).tolist(),
            "committed_state_error_raw_diff": diff.astype(float).tolist(),
            "committed_state_error_weighted_diff": weighted.astype(float).tolist(),
            "committed_state_error_unweighted_l2": _l2(diff),
            "committed_state_error_base_l2": _l2(weighted[base_mask]),
            "committed_state_error_arm_l2": _l2(weighted[arm_mask]),
            "committed_state_error_max_abs": float(np.max(np.abs(weighted))) if weighted.size else 0.0,
        }
        return _l2(weighted), stats, idx, weights, diff


    def _commit_explicit_recovery_chunk(
        self,
        obs: Any,
        chunk: np.ndarray,
        info: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[bool, InfoDict]:
        """Validate and cache an accepted recovery chunk for tick-by-tick replay."""
        reject_info: InfoDict = {}
        if not (self.explicit_return and self.commit_accepted_chunks):
            return False, reject_info
        if not bool(info.get("optimized_accepted", False)):
            return False, reject_info
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            return False, {
                "committed_rejected_missing_planned_q": True,
                "committed_reject_reason": "invalid_committed_chunk",
            }
        total = int(chunk.shape[0])
        deform_len = int(info.get("deform_chunk_length", min(self.deform_horizon, total)))
        return_len = int(info.get("recover_chunk_length", info.get("return_chunk_length", min(self.return_horizon, max(0, total - deform_len)))))
        deform_len = max(0, min(deform_len, total))
        return_len = max(0, min(return_len, total - deform_len))
        if info.get("recovery_candidate_class") != "committed_suffix_replan":
            self.committed_suffix_replans_in_current_recovery = 0
        modes = ["horizon_deform"] * deform_len
        modes.extend(["recover"] * return_len)
        modes.extend(["pass_through"] * max(0, total - len(modes)))
        if not modes:
            modes = ["horizon_deform"] * total

        try:
            current_q = self._current_replay_q(obs, **kwargs)
            post_q_seq = np.asarray(
                self._rollout_chunk_from_q(current_q, chunk),
                dtype=np.float32,
            )
            if post_q_seq.ndim != 2:
                raise ValueError(
                    f"planned rollout must be 2-D, got shape {post_q_seq.shape}"
                )
            if post_q_seq.shape[0] != total:
                raise ValueError(
                    "planned rollout horizon does not match committed chunk: "
                    f"{post_q_seq.shape[0]} != {total}"
                )
            current_q = np.asarray(current_q, dtype=np.float32).reshape(-1)
            if post_q_seq.shape[1] != current_q.shape[0]:
                raise ValueError(
                    "planned rollout state dim does not match current q: "
                    f"{post_q_seq.shape[1]} != {current_q.shape[0]}"
                )
            valid = self.controlled_state_indices < post_q_seq.shape[1]
            if not np.any(valid):
                raise ValueError("planned rollout has no controlled state dimensions")
            pre_q_seq = np.empty_like(post_q_seq)
            pre_q_seq[0] = current_q
            if total > 1:
                pre_q_seq[1:] = post_q_seq[:-1]
            planned_pre_safety = self.evaluate_horizon_safety(obs, pre_q_seq)
            planned_post_safety = self.evaluate_horizon_safety(obs, post_q_seq)
            planned_pre_h_seq = self._clearance_sequence_from_eval(
                planned_pre_safety,
                pre_q_seq.shape[0],
            )
            planned_post_h_seq = self._clearance_sequence_from_eval(
                planned_post_safety,
                post_q_seq.shape[0],
            )
            planned_pre_h_seq = np.asarray(planned_pre_h_seq, dtype=np.float32).reshape(-1)
            planned_post_h_seq = np.asarray(planned_post_h_seq, dtype=np.float32).reshape(-1)
            if planned_pre_h_seq.shape[0] != total:
                raise ValueError(
                    "planned pre-clearance horizon does not match committed chunk: "
                    f"{planned_pre_h_seq.shape[0]} != {total}"
                )
            if planned_post_h_seq.shape[0] != total:
                raise ValueError(
                    "planned post-clearance horizon does not match committed chunk: "
                    f"{planned_post_h_seq.shape[0]} != {total}"
                )
        except Exception as exc:  # noqa: BLE001
            self._clear_committed_chunk()
            logger.warning("Committed recovery plan diagnostics failed: %s", exc)
            return False, {
                "committed_rejected_missing_planned_q": True,
                "committed_reject_reason": str(exc),
            }

        self.committed_sequence_id += 1
        if hasattr(self, "mpc"):
            self.mpc.last_actual_q = None
            self.mpc.last_actual_q_key = None
        self.committed_chunk = chunk.copy()
        self.committed_chunk_index = 0
        self.committed_chunk_modes = modes[:total]
        self.committed_chunk_mode = self.committed_chunk_modes[0]
        self.committed_rejoin_index = info.get(
            "recover_target_index",
            info.get("return_target_index", info.get("rejoin_index", info.get("act_resume_index"))),
        )
        self.committed_until_complete = True
        self.committed_planned_actions = chunk.copy()
        self.committed_planned_q_seq = pre_q_seq.copy()
        self.committed_planned_post_q_seq = post_q_seq.copy()
        self.committed_planned_clearance_pre_seq = planned_pre_h_seq.copy()
        self.committed_planned_clearance_post_seq = planned_post_h_seq.copy()
        self.committed_planned_h_seq = self.committed_planned_clearance_post_seq.copy()
        self.committed_planned_min_clearance_seq = self.committed_planned_clearance_post_seq.copy()
        self._committed_tracking_last_target_key = None
        self._committed_tracking_last_target_q = None
        self._committed_tracking_last_q_l2_before = None
        self._committed_tracking_retarget_count = 0
        self._committed_tracking_negative_actual_improvement_streak = 0
        accepted_min = info.get(
            "recover_min_clearance",
            info.get("return_min_clearance", planned_post_safety.get("min_clearance", None)),
        )
        if accepted_min is not None:
            self.committed_accepted_min_clearance = float(accepted_min)
            self.committed_accepted_clearance_margin = float(
                self.committed_accepted_min_clearance - self.min_clearance
            )
        self.committed_accepted_human_state_snapshot = self._snapshot_human_state(
            kwargs.get("human_state")
        )
        self.committed_planning_human_state_snapshot = self.committed_accepted_human_state_snapshot
        rejoin_keys = (
            "recover_rejoin_loss",
            "q_rejoin_loss",
            "q_rejoin_dist",
            "q_rejoin_threshold",
            "q_rejoin_index",
            "qd_rejoin_loss",
            "qd_rejoin_dist",
            "qd_rejoin_threshold",
            "qd_rejoin_index",
            "return_qd_rejoin_loss",
            "return_qd_rejoin_index",
            "rejoin_q_eval_time_ms",
            "rejoin_qd_eval_time_ms",
            "recover_target_index",
            "recover_reject_reason",
            "recovery_candidate_class",
            "recover_path_min_clearance",
            "recover_immediate_clearance",
            "recover_prefix_min_clearance",
            "recover_target_key",
            "recover_projection_on_nominal",
            "recover_cosine_to_nominal",
            "recover_direction_cosine",
            "recover_direction_cosine_threshold",
            "recover_direction_loss",
            "recover_direction_ok",
            "recover_direction_alignment_available",
            "recover_direction_alignment_weight",
            "recover_act_direction_available",
            "recover_act_progress_loss",
            "recover_act_heading_loss",
            "recover_act_direction_loss",
            "recover_act_progress_projection",
            "recover_act_target_progress",
            "recover_act_heading_cosine",
            "recover_act_heading_cosine_min",
            "recover_act_progress_ok",
            "recover_act_heading_ok",
            "recover_act_progress_weight",
            "recover_act_heading_weight",
            "recover_min_act_heading_cosine",
            "recover_ordered_path_available",
            "recover_ordered_target_index",
            "recover_ordered_horizon",
            "recover_ordered_pose_loss",
            "recover_ordered_delta_loss",
            "recover_ordered_waypoint_pose_loss",
            "recover_ordered_waypoint_rmse",
            "recover_ordered_heading_loss",
            "recover_ordered_heading_cosine",
            "recover_ordered_heading_cosine_min",
            "recover_ordered_heading_cosine_threshold",
            "recover_ordered_backtrack_count",
            "recover_ordered_monotonic_ok",
            "recover_ordered_pose_tube_threshold",
            "recover_ordered_pose_tube_ok",
            "recover_ordered_waypoint_tube_ok",
            "recover_ordered_strict_ok",
            "recover_ordered_waypoint_index_start",
            "recover_ordered_waypoint_index_end",
            "recover_ordered_loss",
            "recover_ordered_pose_weight",
            "recover_ordered_delta_weight",
            "recover_ordered_heading_weight",
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
            "recovery_path_failure_streak",
            "committed_suffix_replan_attempted",
            "committed_suffix_replan_accepted",
            "committed_suffix_replan_rejected",
            "committed_suffix_replan_reject_reason",
            "committed_suffix_replan_from_index",
            "committed_suffix_replan_old_length",
            "committed_suffix_replan_new_length",
            "committed_suffix_replan_target_index",
            "resume_tube_score",
            "resume_tube_ok",
            "resume_tube_min_score",
            "resume_tube_component_score",
            "resume_tube_min_component_score",
            "resume_tube_component_ok",
            "resume_tube_terminal_score",
            "resume_tube_path_score",
            "resume_tube_progress_score",
            "resume_tube_heading_score",
            "resume_tube_clearance_score",
            "resume_tube_terminal_dist",
            "resume_tube_terminal_threshold",
            "resume_tube_ordered_loss",
            "resume_tube_prefix_min_clearance",
            "resume_tube_required_clearance",
            "resume_tube_prefix_safe",
            "resume_tube_terminal_ok",
            "recover_resume_tube_score",
            "recover_resume_tube_ok",
            "recover_resume_tube_min_score",
            "recover_resume_tube_component_score",
            "recover_resume_tube_min_component_score",
            "recover_resume_tube_component_ok",
            "recover_resume_tube_terminal_score",
            "recover_resume_tube_path_score",
            "recover_resume_tube_progress_score",
            "recover_resume_tube_heading_score",
            "recover_resume_tube_clearance_score",
            "recover_resume_tube_terminal_dist",
            "recover_resume_tube_terminal_threshold",
            "recover_resume_tube_ordered_loss",
            "recover_resume_tube_prefix_min_clearance",
            "recover_resume_tube_required_clearance",
            "recover_resume_tube_prefix_safe",
            "recover_resume_tube_terminal_ok",
            "recover_resume_window_available",
            "recover_resume_window_len",
            "recover_resume_window_requested_len",
            "recover_resume_window_loss",
            "recover_resume_window_total_loss",
            "recover_resume_window_dist",
            "recover_resume_window_error_l2",
            "recover_resume_window_dq_loss",
            "recover_resume_window_dq_dist",
            "recover_resume_window_action_loss",
            "recover_resume_window_action_dist",
            "recover_resume_window_q_frame_l2",
            "recover_resume_window_q_frame_l2_mean",
            "recover_resume_window_q_frame_l2_max",
            "recover_resume_window_wrist_l2",
            "recover_resume_window_wrist_l2_mean",
            "recover_resume_window_wrist_l2_max",
            "recover_resume_window_left_wrist_abs",
            "recover_resume_window_left_wrist_abs_mean",
            "recover_resume_window_left_wrist_abs_max",
            "recover_resume_window_right_wrist_abs",
            "recover_resume_window_right_wrist_abs_mean",
            "recover_resume_window_right_wrist_abs_max",
            "recover_resume_window_recovery_step_l2",
            "recover_resume_window_target_step_l2",
            "recover_resume_window_step_l2_error",
            "recover_resume_window_step_l2_error_mean",
            "recover_resume_window_step_l2_error_max",
            "recover_resume_window_dq_error_l2",
            "recover_resume_window_dq_cosine",
            "recover_resume_window_dq_cosine_mean",
            "recover_resume_window_dq_cosine_min",
            "recover_resume_window_dq_norm_ratio",
            "recover_resume_window_dq_norm_ratio_mean",
            "recover_resume_window_dq_norm_ratio_min",
            "recover_resume_window_start_local_index",
            "recover_resume_window_end_local_index",
            "recover_resume_window_weight",
            "recover_resume_window_dq_weight",
            "recover_resume_window_action_weight",
            "recover_resume_window_q",
            "recover_resume_window_target_q",
            "recover_resume_window_action",
            "recover_resume_window_target_action",
            "mpc_recovery_target_tube_available",
            "mpc_recovery_target_tube_ok",
            "mpc_recovery_target_tube_progress_ok",
            "mpc_recovery_target_tube_loss",
            "mpc_recovery_target_tube_terminal_loss",
            "mpc_recovery_target_tube_terminal_dist",
            "mpc_recovery_target_tube_min_path_loss",
            "mpc_recovery_target_tube_min_path_dist",
            "mpc_recovery_target_tube_loss_threshold",
            "mpc_recovery_target_tube_dist_threshold",
            "mpc_recovery_target_tube_current_local_index",
            "mpc_recovery_target_tube_terminal_local_index",
            "mpc_recovery_target_tube_local_index_progress",
            "mpc_recovery_target_tube_target_index",
            "mpc_recovery_target_tube_heading_cosine",
            "mpc_recovery_target_tube_progress_projection",
            "mpc_recovery_target_tube_target_tangent_norm",
            "mpc_recovery_target_tube_terminal_delta_norm",
            "mpc_recovery_target_tube_terminal_error_l2",
            "mpc_recovery_target_tube_window_len",
            "mpc_recovery_target_tube_requested_window_len",
            "mpc_recovery_target_tube_window_loss",
            "mpc_recovery_target_tube_window_total_loss",
            "mpc_recovery_target_tube_window_dist",
            "mpc_recovery_target_tube_window_error_l2",
            "mpc_recovery_target_tube_window_dq_loss",
            "mpc_recovery_target_tube_window_dq_dist",
            "mpc_recovery_target_tube_window_action_loss",
            "mpc_recovery_target_tube_window_action_dist",
            "mpc_recovery_target_tube_window_q_frame_l2",
            "mpc_recovery_target_tube_window_q_frame_l2_mean",
            "mpc_recovery_target_tube_window_q_frame_l2_max",
            "mpc_recovery_target_tube_window_wrist_l2",
            "mpc_recovery_target_tube_window_wrist_l2_mean",
            "mpc_recovery_target_tube_window_wrist_l2_max",
            "mpc_recovery_target_tube_window_left_wrist_abs",
            "mpc_recovery_target_tube_window_left_wrist_abs_mean",
            "mpc_recovery_target_tube_window_left_wrist_abs_max",
            "mpc_recovery_target_tube_window_right_wrist_abs",
            "mpc_recovery_target_tube_window_right_wrist_abs_mean",
            "mpc_recovery_target_tube_window_right_wrist_abs_max",
            "mpc_recovery_target_tube_window_recovery_step_l2",
            "mpc_recovery_target_tube_window_target_step_l2",
            "mpc_recovery_target_tube_window_step_l2_error",
            "mpc_recovery_target_tube_window_step_l2_error_mean",
            "mpc_recovery_target_tube_window_step_l2_error_max",
            "mpc_recovery_target_tube_window_dq_error_l2",
            "mpc_recovery_target_tube_window_dq_cosine",
            "mpc_recovery_target_tube_window_dq_cosine_mean",
            "mpc_recovery_target_tube_window_dq_cosine_min",
            "mpc_recovery_target_tube_window_dq_norm_ratio",
            "mpc_recovery_target_tube_window_dq_norm_ratio_mean",
            "mpc_recovery_target_tube_window_dq_norm_ratio_min",
            "mpc_recovery_target_tube_window_start_local_index",
            "mpc_recovery_target_tube_window_end_local_index",
            "mpc_recovery_target_tube_window_weight",
            "mpc_recovery_target_tube_window_dq_weight",
            "mpc_recovery_target_tube_window_action_weight",
            "mpc_recovery_target_tube_terminal_delta",
            "mpc_recovery_target_tube_q_error",
            "mpc_recovery_target_tube_terminal_q",
            "mpc_recovery_target_tube_target_q",
            "mpc_recovery_planned_q_seq",
            "mpc_recovery_planned_action_seq",
            "mpc_recovery_target_tube_window_q",
            "mpc_recovery_target_tube_target_window_q",
            "mpc_recovery_target_tube_window_action",
            "mpc_recovery_target_tube_target_window_action",
            "mpc_recovery_target_tube_terminal_valid_delta",
            "mpc_recovery_target_tube_terminal_weighted_delta",
            "mpc_recovery_target_tube_state_indices",
            "mpc_recovery_target_tube_state_weights",
            "mpc_recovery_target_tube_target_source",
            "mpc_recovery_target_tube_window_start",
            "mpc_recovery_target_tube_window_end",
            "mpc_recovery_target_tube_live_prefix_safe",
            "mpc_recovery_target_tube_live_prefix_best_min_clearance",
            "mpc_recovery_q_rejoin_overridden_by_tube",
            "mpc_recovery_original_reject_reason",
            "committed_suffix_replan_seed_start_index",
        )
        self.committed_rejoin_diagnostics = {
            key: info.get(key) for key in rejoin_keys if info.get(key) is not None
        }
        return True, {}

    def _committed_action_safety(
        self,
        obs: Any,
        pre_q: np.ndarray,
        action: np.ndarray,
        **kwargs: Any,
    ) -> tuple[bool, InfoDict]:
        """Evaluate one committed action from a supplied pre-action state."""
        try:
            pre_q = np.asarray(pre_q, dtype=np.float32).reshape(-1)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            post_q = self._rollout_one_step_from_q(pre_q, action)
            pre_safety = self.evaluate_horizon_safety(obs, pre_q.reshape(1, -1))
            post_safety = self.evaluate_horizon_safety(obs, post_q.reshape(1, -1))
            pre_clearance = float(pre_safety.get("min_clearance", float("-inf")))
            post_clearance = float(post_safety.get("min_clearance", float("-inf")))
            safe = bool(post_safety.get("horizon_safe", post_clearance >= self.min_clearance))
            if post_clearance < self.min_clearance:
                safe = False
            safety = dict(post_safety)
            safety.update(
                {
                    "horizon_safe": safe,
                    "min_clearance": post_clearance,
                    "min_clearances": np.asarray([post_clearance], dtype=np.float32),
                    "actual_pre_action_q": pre_q.copy(),
                    "replay_predicted_post_action_q": post_q.copy(),
                    "replay_clearance_pre": pre_clearance,
                    "replay_clearance_post": post_clearance,
                    "replay_pre_safety_available": bool(
                        pre_safety.get("safety_eval_available", True)
                    ),
                    "replay_post_safety_available": bool(
                        post_safety.get("safety_eval_available", True)
                    ),
                    "committed_action": action.copy(),
                    "control_type": self.control_type,
                    "dt": float(self.dt),
                    "controlled_state_indices": self.controlled_state_indices.copy(),
                    "controlled_action_indices": self.controlled_action_indices.copy(),
                    "action_conversion_mode": self.control_type,
                }
            )
            return safe, safety
        except Exception as exc:  # noqa: BLE001
            logger.warning("Committed recovery safety check failed: %s", exc)
            return False, {
                "horizon_safe": False,
                "min_clearance": float("-inf"),
                "replay_clearance_post": float("-inf"),
                "first_violation": 0,
                "unsafe_count": 1,
                "safety_eval_available": False,
                "safety_check_error": str(exc),
            }

    def _committed_nominal_tube_tracking_action(
        self,
        action: np.ndarray,
        actual_q: np.ndarray,
        idx: int,
        *,
        action_idx: np.ndarray,
    ) -> tuple[np.ndarray, InfoDict]:
        """Closed-loop track the stored nominal resume tube during recovery replay."""
        tracked = np.asarray(action, dtype=np.float32).copy()
        info: InfoDict = {
            "committed_nominal_tube_tracking_enabled": bool(
                self.committed_nominal_tube_tracking_enabled
            ),
            "committed_nominal_tube_tracking_applied": False,
        }
        if not self.committed_nominal_tube_tracking_enabled:
            return tracked, info
        actual = np.asarray(actual_q, dtype=np.float32).reshape(-1)
        if actual.size == 0 or tracked.size == 0:
            info["committed_nominal_tube_tracking_reason"] = "missing_actual_q_or_action"
            return tracked, info
        valid_action_set = {int(i) for i in np.asarray(action_idx).reshape(-1)}
        if not valid_action_set:
            info["committed_nominal_tube_tracking_reason"] = "missing_controlled_action_indices"
            return tracked, info

        target_window = None
        target_source = None
        diagnostics = self.committed_rejoin_diagnostics or {}
        for key in (
            "recover_resume_window_target_q",
            "recover_resume_window_q",
            "mpc_recovery_target_tube_target_window_q",
            "mpc_recovery_target_tube_window_q",
        ):
            raw_value = diagnostics.get(key)
            if raw_value is None:
                continue
            try:
                candidate = np.asarray(raw_value, dtype=np.float32)
            except Exception:  # noqa: BLE001
                continue
            if candidate.size == 0:
                continue
            if candidate.ndim == 1:
                candidate = candidate.reshape(1, -1)
            elif candidate.ndim > 2:
                candidate = candidate.reshape(-1, candidate.shape[-1])
            if candidate.ndim == 2 and candidate.shape[1] > 0:
                target_window = candidate
                target_source = key
                break
        if target_window is None or target_source is None:
            info["committed_nominal_tube_tracking_reason"] = "missing_nominal_tube_window"
            return tracked, info

        state_indices = []
        action_indices = []
        for state_i, action_i in zip(
            np.asarray(self.controlled_state_indices).reshape(-1),
            np.asarray(self.controlled_action_indices).reshape(-1),
        ):
            state_i = int(state_i)
            action_i = int(action_i)
            if action_i not in valid_action_set:
                continue
            if state_i < 0 or action_i < 0:
                continue
            if state_i >= actual.size or state_i >= target_window.shape[1]:
                continue
            if action_i >= tracked.size:
                continue
            state_indices.append(state_i)
            action_indices.append(action_i)
        if not state_indices:
            info["committed_nominal_tube_tracking_reason"] = "missing_valid_state_action_pairs"
            return tracked, info

        state_arr = np.asarray(state_indices, dtype=np.int64)
        action_arr = np.asarray(action_indices, dtype=np.int64)
        finite_rows = np.isfinite(target_window[:, state_arr]).all(axis=1)
        finite_rows &= bool(np.isfinite(actual[state_arr]).all())
        if not bool(np.any(finite_rows)):
            info["committed_nominal_tube_tracking_reason"] = "nonfinite_target_or_actual_q"
            return tracked, info
        diffs = target_window[:, state_arr] - actual[state_arr]
        row_dists = np.linalg.norm(diffs, axis=1)
        row_dists = np.where(finite_rows, row_dists, np.inf)

        def _window_tangent(row_index: int) -> np.ndarray | None:
            if target_window.shape[0] <= 1:
                return None
            row_i = int(np.clip(row_index, 0, target_window.shape[0] - 1))
            if row_i < target_window.shape[0] - 1:
                tangent = target_window[row_i + 1, state_arr] - target_window[row_i, state_arr]
            else:
                tangent = target_window[row_i, state_arr] - target_window[row_i - 1, state_arr]
            tangent = np.asarray(tangent, dtype=np.float32).reshape(-1)
            if tangent.size == 0 or not np.isfinite(tangent).all():
                return None
            if float(np.linalg.norm(tangent)) <= 1e-8:
                return None
            return tangent

        def _cosine_score(vec: Any, ref: Any) -> float:
            vec_arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            ref_arr = np.asarray(ref, dtype=np.float32).reshape(-1)
            if vec_arr.size == 0 or ref_arr.size == 0 or vec_arr.shape != ref_arr.shape:
                return 1.0
            vec_norm = float(np.linalg.norm(vec_arr))
            ref_norm = float(np.linalg.norm(ref_arr))
            if vec_norm <= 1e-8 or ref_norm <= 1e-8:
                return 1.0
            cosine = float(np.dot(vec_arr, ref_arr) / max(1e-8, vec_norm * ref_norm))
            return float(np.clip(0.5 * (cosine + 1.0), 0.0, 1.0))

        score_scale = float(self.committed_nominal_tube_tracking_score_scale)
        window_heading_weight = float(
            self.committed_nominal_tube_tracking_window_heading_weight
        )
        row_scores = np.full(row_dists.shape, -np.inf, dtype=np.float32)
        for row_i in range(target_window.shape[0]):
            if not np.isfinite(row_dists[row_i]):
                continue
            q_score = float(np.exp(-float(row_dists[row_i]) / max(score_scale, 1e-6)))
            tangent = _window_tangent(row_i)
            heading_score = (
                1.0
                if tangent is None
                else _cosine_score(target_window[row_i, state_arr] - actual[state_arr], tangent)
            )
            row_scores[row_i] = float(
                (1.0 - window_heading_weight) * q_score
                + window_heading_weight * heading_score
            )
        prev_negative_streak_for_action = int(
            getattr(self, "_committed_tracking_negative_actual_improvement_streak", 0)
        )
        previous_key = getattr(self, "_committed_tracking_last_target_key", None)
        sequence_key = (str(target_source), int(target_window.shape[0]))
        previous_sequence_key = getattr(self, "_committed_tracking_sequence_key", None)
        previous_sequence_step = int(
            getattr(self, "_committed_tracking_sequence_step", -1)
        )
        previous_replay_index = getattr(self, "_committed_tracking_last_replay_index", None)
        sequence_reset = bool(previous_sequence_key != sequence_key)
        try:
            replay_index = int(idx)
            if previous_replay_index is None or replay_index <= int(previous_replay_index):
                sequence_reset = True
        except Exception:  # noqa: BLE001
            replay_index = int(idx) if isinstance(idx, (int, np.integer)) else -1
        hold_sequence_due_to_actual_error = bool(
            prev_negative_streak_for_action > 0 and not sequence_reset
        )
        if sequence_reset:
            sequence_step = 0
        elif hold_sequence_due_to_actual_error:
            sequence_step = max(0, previous_sequence_step)
        else:
            sequence_step = previous_sequence_step + 1
        desired_target_index = int(
            np.clip(sequence_step, 0, max(0, target_window.shape[0] - 1))
        )
        target_index = desired_target_index
        adaptive_retargeted_from_failed_target = False
        if not (np.isfinite(row_dists[target_index]) and np.isfinite(row_scores[target_index])):
            valid_indices = np.flatnonzero(np.isfinite(row_dists) & np.isfinite(row_scores))
            if valid_indices.size > 0:
                target_index = int(
                    valid_indices[int(np.argmin(np.abs(valid_indices - desired_target_index)))]
                )
                adaptive_retargeted_from_failed_target = bool(target_index != desired_target_index)
        if not np.isfinite(row_dists[target_index]) or not np.isfinite(row_scores[target_index]):
            info["committed_nominal_tube_tracking_reason"] = "nonfinite_target_distance"
            return tracked, info
        target_q = target_window[target_index]
        target_tangent = _window_tangent(target_index)

        before = tracked.copy()
        arm_gain = float(self.committed_nominal_tube_tracking_arm_gain)
        base_gain = float(self.committed_nominal_tube_tracking_base_gain)
        max_arm_step = float(self.committed_nominal_tube_tracking_max_arm_step)
        max_base_delta = float(self.committed_nominal_tube_tracking_max_base_delta)
        response_gain_by_state = np.ones(actual.size, dtype=np.float32)
        response_gain_source = "default"
        previous_response_gain = getattr(
            self, "_committed_tracking_response_gain_by_state", None
        )
        previous_actual_for_response = getattr(
            self, "_committed_tracking_last_actual_q_before", None
        )
        previous_action_for_response = getattr(
            self, "_committed_tracking_last_tracked_action", None
        )
        if sequence_reset:
            previous_response_gain = None
            previous_actual_for_response = None
            previous_action_for_response = None
        if previous_response_gain is not None:
            try:
                prev_gain = np.asarray(previous_response_gain, dtype=np.float32).reshape(-1)
                n = min(prev_gain.size, response_gain_by_state.size)
                response_gain_by_state[:n] = np.clip(prev_gain[:n], 0.15, 1.5)
                response_gain_source = "cached"
            except Exception:  # noqa: BLE001
                response_gain_source = "default"
        if previous_actual_for_response is not None and previous_action_for_response is not None:
            try:
                prev_actual = np.asarray(previous_actual_for_response, dtype=np.float32).reshape(-1)
                prev_action = np.asarray(previous_action_for_response, dtype=np.float32).reshape(-1)
                measured_gain = response_gain_by_state.copy()
                measured_mask = []
                for state_i, action_i in zip(state_indices, action_indices):
                    if state_i >= prev_actual.size or action_i >= prev_action.size:
                        continue
                    actual_delta = float(actual[state_i] - prev_actual[state_i])
                    if state_i < 4:
                        command_delta = float(prev_action[action_i])
                    else:
                        command_delta = float(prev_action[action_i] - prev_actual[state_i])
                    if abs(command_delta) <= 1e-4:
                        continue
                    if actual_delta * command_delta <= 0.0:
                        continue
                    gain = actual_delta / command_delta
                    if not np.isfinite(gain):
                        continue
                    measured_gain[state_i] = np.float32(np.clip(gain, 0.15, 1.5))
                    measured_mask.append(state_i)
                if measured_mask:
                    alpha = 0.35
                    measured_idx = np.asarray(measured_mask, dtype=np.int64)
                    response_gain_by_state[measured_idx] = (
                        (1.0 - alpha) * response_gain_by_state[measured_idx]
                        + alpha * measured_gain[measured_idx]
                    )
                    response_gain_source = "measured_ema"
            except Exception:  # noqa: BLE001
                pass
        self._committed_tracking_response_gain_by_state = response_gain_by_state.copy()
        response_gain_values = np.asarray(
            [response_gain_by_state[int(i)] for i in state_indices], dtype=np.float32
        )
        response_gain_mean = float(np.mean(response_gain_values)) if response_gain_values.size else None
        response_gain_min = float(np.min(response_gain_values)) if response_gain_values.size else None
        response_gain_max = float(np.max(response_gain_values)) if response_gain_values.size else None

        def _direct_tracker(scale: float) -> np.ndarray:
            candidate = before.copy()
            scale = float(scale)
            for state_i, action_i in zip(state_indices, action_indices):
                delta = float(target_q[state_i] - actual[state_i])
                if not np.isfinite(delta):
                    continue
                gain_est = 1.0
                if 0 <= state_i < response_gain_by_state.size:
                    gain_est = float(np.clip(response_gain_by_state[state_i], 0.15, 1.5))
                if state_i < 4:
                    command = scale * base_gain * delta / max(gain_est, 0.15)
                    if max_base_delta > 0.0:
                        command = float(np.clip(command, -max_base_delta, max_base_delta))
                else:
                    step = scale * arm_gain * delta / max(gain_est, 0.15)
                    if max_arm_step > 0.0:
                        step = float(np.clip(step, -max_arm_step, max_arm_step))
                    command = float(actual[state_i] + step)
                if np.isfinite(command):
                    candidate[action_i] = np.float32(command)
            return candidate

        q_l2_before = float(row_dists[target_index])
        finite_row_dists = row_dists[np.isfinite(row_dists)]
        live_window_min_l2 = (
            None if finite_row_dists.size == 0 else float(np.min(finite_row_dists))
        )
        actual_window_history = [] if sequence_reset else list(
            getattr(self, "_committed_tracking_actual_window_history", [])
        )
        actual_window_history.append(actual.copy())
        max_window_len = max(1, int(target_window.shape[0]))
        actual_window_history = actual_window_history[-max_window_len:]
        self._committed_tracking_actual_window_history = actual_window_history
        live_window_l2_mean = None
        live_window_l2_max = None
        live_window_slot_count = int(len(actual_window_history))
        if live_window_slot_count > 0:
            try:
                paired = min(live_window_slot_count, int(target_window.shape[0]))
                actual_seq = np.stack(actual_window_history[-paired:], axis=0)
                target_seq = target_window[:paired]
                live_window_l2_values = np.linalg.norm(
                    target_seq[:, state_arr] - actual_seq[:, state_arr], axis=1
                )
                live_window_l2_mean = float(np.mean(live_window_l2_values))
                live_window_l2_max = float(np.max(live_window_l2_values))
            except Exception:  # noqa: BLE001
                live_window_l2_mean = None
                live_window_l2_max = None
        heading_error_before = None
        if target_tangent is not None:
            heading_error_before = float(
                1.0 - _cosine_score(target_q[state_arr] - actual[state_arr], target_tangent)
            )

        servo_mode = bool(self.committed_nominal_tube_tracking_servo_mode)
        servo_base_scale = float(self.committed_nominal_tube_tracking_servo_scale)
        servo_boost_scale = float(self.committed_nominal_tube_tracking_servo_boost_scale)
        adaptive_slowdown_active = bool(prev_negative_streak_for_action > 0)
        adaptive_slowdown_factor = float(
            0.5 ** min(max(prev_negative_streak_for_action, 0), 3)
        )
        servo_min_scale = float(min(servo_base_scale, max(0.5, 0.25 * servo_base_scale)))
        servo_scale = float(servo_base_scale)
        if adaptive_slowdown_active:
            servo_scale = float(max(servo_min_scale, servo_base_scale * adaptive_slowdown_factor))
        servo_boost_active = False

        # Closed-loop servo: do not rank candidate actions with the rollout model.
        # If the previous real Bigym step moved away from the target, become more
        # conservative and retarget instead of pushing harder into the same error.
        tracked = _direct_tracker(servo_scale)
        solver_enabled = False
        candidate_count = 1
        rollout_predicted_improvement = None
        q_l2_after = None
        heading_error_after = None
        best_solver_loss = None
        best_candidate_name = f"servo_scale_{servo_scale:g}" if servo_mode else "direct_scale_1"
        if not servo_mode:
            tracked = _direct_tracker(1.0)
            best_candidate_name = "direct_scale_1"
        try:
            pred_q = self._rollout_one_step_from_q(actual, tracked)
            pred_q = np.asarray(pred_q, dtype=np.float32).reshape(-1)
            q_l2_after = float(np.linalg.norm(target_q[state_arr] - pred_q[state_arr]))
            if target_tangent is not None:
                heading_error_after = float(
                    1.0 - _cosine_score(pred_q[state_arr] - actual[state_arr], target_tangent)
                )
            rollout_predicted_improvement = float(q_l2_before - q_l2_after)
            if q_l2_after is not None:
                best_solver_loss = float(q_l2_after)
        except Exception:  # noqa: BLE001
            q_l2_after = None
            rollout_predicted_improvement = None

        action_delta = tracked - before
        done_threshold = float(self.committed_nominal_tube_tracking_done_threshold)
        target_key = (str(target_source), int(target_index))
        previous_key = getattr(self, "_committed_tracking_last_target_key", None)
        retargeted = bool(previous_key is not None and previous_key != target_key)
        if retargeted:
            self._committed_tracking_retarget_count = int(
                getattr(self, "_committed_tracking_retarget_count", 0)
            ) + 1
        previous_target_q = getattr(self, "_committed_tracking_last_target_q", None)
        previous_q_l2_before = getattr(self, "_committed_tracking_last_q_l2_before", None)
        actual_improvement = None
        if previous_target_q is not None and previous_q_l2_before is not None:
            try:
                prev_target = np.asarray(previous_target_q, dtype=np.float32).reshape(-1)
                if prev_target.size > int(np.max(state_arr)):
                    current_prev_dist = float(np.linalg.norm(prev_target[state_arr] - actual[state_arr]))
                    actual_improvement = float(float(previous_q_l2_before) - current_prev_dist)
            except Exception:  # noqa: BLE001
                actual_improvement = None
        negative_streak = int(
            getattr(self, "_committed_tracking_negative_actual_improvement_streak", 0)
        )
        if actual_improvement is not None:
            if float(actual_improvement) < -1e-4:
                negative_streak += 1
            else:
                negative_streak = 0
            self._committed_tracking_negative_actual_improvement_streak = negative_streak
        max_negative_steps = int(
            self.committed_nominal_tube_tracking_max_negative_actual_steps
        )
        tracking_failed = bool(negative_streak > max_negative_steps)
        self._committed_tracking_last_target_key = target_key
        self._committed_tracking_last_target_q = target_q.copy()
        self._committed_tracking_last_q_l2_before = float(q_l2_before)
        ready_error = (
            live_window_l2_mean
            if live_window_l2_mean is not None
            else (live_window_min_l2 if live_window_min_l2 is not None else q_l2_before)
        )
        self._committed_tracking_sequence_key = sequence_key
        self._committed_tracking_sequence_step = int(sequence_step)
        self._committed_tracking_last_replay_index = int(replay_index)
        self._committed_tracking_last_actual_q_before = actual.copy()
        self._committed_tracking_last_tracked_action = tracked.copy()
        info.update(
            {
                "committed_nominal_tube_tracking_applied": True,
                "committed_nominal_tube_tracking_source": target_source,
                "committed_nominal_tube_tracking_target_index": target_index,
                "committed_nominal_tube_tracking_selected_resume_window_index": target_index,
                "committed_nominal_tube_tracking_sequential_targeting": True,
                "committed_nominal_tube_tracking_sequence_step": int(sequence_step),
                "committed_nominal_tube_tracking_desired_target_index": int(desired_target_index),
                "committed_nominal_tube_tracking_sequence_reset": bool(sequence_reset),
                "committed_nominal_tube_tracking_sequence_hold_due_to_actual_error": bool(
                    hold_sequence_due_to_actual_error
                ),
                "committed_nominal_tube_tracking_selected_resume_score": float(row_scores[target_index]),
                "committed_nominal_tube_tracking_resume_score_before": float(row_scores[target_index]),
                "committed_nominal_tube_tracking_resume_score_after": (
                    None
                    if q_l2_after is None
                    else float(np.exp(-float(q_l2_after) / max(score_scale, 1e-6)))
                ),
                "committed_nominal_tube_tracking_q_l2_before": q_l2_before,
                "committed_nominal_tube_tracking_q_l2_after": q_l2_after,
                "committed_nominal_tube_tracking_q_window_l2_before": q_l2_before,
                "committed_nominal_tube_tracking_q_window_l2_after": q_l2_after,
                "committed_nominal_tube_tracking_live_window_min_l2": live_window_min_l2,
                "committed_nominal_tube_tracking_live_window_l2_mean": live_window_l2_mean,
                "committed_nominal_tube_tracking_live_window_l2_max": live_window_l2_max,
                "committed_nominal_tube_tracking_live_window_slot_count": int(live_window_slot_count),
                "committed_nominal_tube_tracking_q_max_abs_before": float(
                    np.max(np.abs(target_q[state_arr] - actual[state_arr]))
                ),
                "committed_nominal_tube_tracking_heading_error_before": heading_error_before,
                "committed_nominal_tube_tracking_heading_error_after": heading_error_after,
                "committed_nominal_tube_tracking_rollout_solver": bool(solver_enabled),
                "committed_nominal_tube_tracking_servo_mode": bool(servo_mode),
                "committed_nominal_tube_tracking_servo_scale": float(servo_scale),
                "committed_nominal_tube_tracking_servo_boost_active": bool(servo_boost_active),
                "committed_nominal_tube_tracking_servo_adaptive_slowdown_active": bool(
                    adaptive_slowdown_active
                ),
                "committed_nominal_tube_tracking_servo_previous_negative_streak": int(
                    prev_negative_streak_for_action
                ),
                "committed_nominal_tube_tracking_adaptive_retargeted": bool(
                    adaptive_retargeted_from_failed_target
                ),
                "committed_nominal_tube_tracking_servo_boost_scale": float(servo_boost_scale),
                "committed_nominal_tube_tracking_response_gain_source": response_gain_source,
                "committed_nominal_tube_tracking_response_gain_mean": response_gain_mean,
                "committed_nominal_tube_tracking_response_gain_min": response_gain_min,
                "committed_nominal_tube_tracking_response_gain_max": response_gain_max,
                "committed_nominal_tube_tracking_solver_candidate_count": int(candidate_count),
                "committed_nominal_tube_tracking_best_candidate": best_candidate_name,
                "committed_nominal_tube_tracking_solver_loss": best_solver_loss,
                "committed_nominal_tube_tracking_rollout_predicted_improvement": rollout_predicted_improvement,
                "committed_nominal_tube_tracking_actual_improvement": actual_improvement,
                "committed_nominal_tube_tracking_negative_actual_improvement_streak": int(negative_streak),
                "committed_nominal_tube_tracking_max_negative_actual_steps": int(max_negative_steps),
                "committed_nominal_tube_tracking_failed": bool(tracking_failed),
                "committed_nominal_tube_tracking_min_predicted_improvement": float(
                    self.committed_nominal_tube_tracking_min_predicted_improvement
                ),
                "committed_nominal_tube_tracking_retargeted": bool(retargeted),
                "committed_nominal_tube_tracking_retarget_count": int(
                    getattr(self, "_committed_tracking_retarget_count", 0)
                ),
                "committed_nominal_tube_tracking_done_threshold": done_threshold,
                "committed_nominal_tube_tracking_ready": bool(ready_error <= done_threshold),
                "committed_nominal_tube_tracking_action_delta_l2": float(
                    np.linalg.norm(action_delta[action_arr])
                ),
                "committed_nominal_tube_tracking_arm_gain": arm_gain,
                "committed_nominal_tube_tracking_base_gain": base_gain,
                "committed_nominal_tube_tracking_max_arm_step": max_arm_step,
                "committed_nominal_tube_tracking_max_base_delta": max_base_delta,
                "committed_nominal_tube_tracking_score_scale": score_scale,
                "committed_nominal_tube_tracking_action_smooth_weight": float(
                    self.committed_nominal_tube_tracking_action_smooth_weight
                ),
                "committed_nominal_tube_tracking_heading_weight": float(
                    self.committed_nominal_tube_tracking_heading_weight
                ),
                "committed_nominal_tube_tracking_replay_index": int(idx),
            }
        )
        return tracked, info

    def _committed_abort_threshold(self) -> float:
        """Clearance threshold used to abort committed recovery replay."""
        if self.committed_abort_only_if_contact_risk:
            return float(self.committed_min_clearance_for_abort)
        return float(self.min_clearance - self.committed_safety_tol)

    def _committed_planned_value(self, arr: Any, idx: int) -> Any:
        """Read an indexed planned value, tolerating missing/short arrays."""
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 0 or arr.shape[0] == 0:
            return None
        return arr[min(max(int(idx), 0), arr.shape[0] - 1)]

    def _rebase_committed_plan_from_actual_step(
        self,
        obs: Any,
        idx: int,
        actual_pre_q: np.ndarray,
        actual_action: np.ndarray,
        action_safety: Mapping[str, Any] | None,
    ) -> InfoDict:
        """Update the cached committed rollout after repair/tracking changes an action."""
        if self.committed_planned_actions is None or self.committed_chunk is None:
            return {"committed_plan_rebase_available": False}
        try:
            idx = int(idx)
            planned_actions = np.asarray(
                self.committed_planned_actions,
                dtype=np.float32,
            ).copy()
            committed = np.asarray(self.committed_chunk, dtype=np.float32).copy()
            if idx < 0 or idx >= planned_actions.shape[0]:
                return {"committed_plan_rebase_available": False}
            actual_pre_q = np.asarray(actual_pre_q, dtype=np.float32).reshape(-1)
            actual_action = np.asarray(actual_action, dtype=np.float32).reshape(-1)
            if actual_pre_q.size == 0 or actual_action.size == 0:
                return {"committed_plan_rebase_available": False}
            planned_action = planned_actions[idx].copy()
            a_dim = min(actual_action.size, planned_action.size)
            action_delta = float(
                np.linalg.norm(actual_action[:a_dim] - planned_action[:a_dim])
            )
            planned_actions[idx, :a_dim] = actual_action[:a_dim]
            c_dim = min(actual_action.size, committed.shape[1])
            committed[idx, :c_dim] = actual_action[:c_dim]

            post_q = None
            if action_safety is not None:
                post_q = action_safety.get("replay_predicted_post_action_q")
            if post_q is None:
                post_q = self._rollout_one_step_from_q(actual_pre_q, actual_action)
            post_q = np.asarray(post_q, dtype=np.float32).reshape(-1)
            q_dim = min(actual_pre_q.size, post_q.size)
            if q_dim == 0:
                return {"committed_plan_rebase_available": False}

            suffix_len = planned_actions.shape[0] - idx
            pre_suffix = np.empty((suffix_len, q_dim), dtype=np.float32)
            post_suffix = np.empty((suffix_len, q_dim), dtype=np.float32)
            pre_suffix[0] = actual_pre_q[:q_dim]
            post_suffix[0] = post_q[:q_dim]
            for local in range(1, suffix_len):
                pre_suffix[local] = post_suffix[local - 1]
                next_post = self._rollout_one_step_from_q(
                    pre_suffix[local],
                    planned_actions[idx + local],
                )
                post_suffix[local] = np.asarray(
                    next_post,
                    dtype=np.float32,
                ).reshape(-1)[:q_dim]

            pre_safety = self.evaluate_horizon_safety(obs, pre_suffix)
            post_safety = self.evaluate_horizon_safety(obs, post_suffix)
            pre_h = np.asarray(
                self._clearance_sequence_from_eval(pre_safety, suffix_len),
                dtype=np.float32,
            ).reshape(-1)
            post_h = np.asarray(
                self._clearance_sequence_from_eval(post_safety, suffix_len),
                dtype=np.float32,
            ).reshape(-1)

            if self.committed_planned_q_seq is not None:
                arr = np.asarray(self.committed_planned_q_seq, dtype=np.float32).copy()
                arr[idx : idx + suffix_len, :q_dim] = pre_suffix
                self.committed_planned_q_seq = arr
            if self.committed_planned_post_q_seq is not None:
                arr = np.asarray(self.committed_planned_post_q_seq, dtype=np.float32).copy()
                arr[idx : idx + suffix_len, :q_dim] = post_suffix
                self.committed_planned_post_q_seq = arr
            if self.committed_planned_clearance_pre_seq is not None:
                arr = np.asarray(
                    self.committed_planned_clearance_pre_seq,
                    dtype=np.float32,
                ).copy()
                arr[idx : idx + suffix_len] = pre_h[:suffix_len]
                self.committed_planned_clearance_pre_seq = arr
            if self.committed_planned_clearance_post_seq is not None:
                arr = np.asarray(
                    self.committed_planned_clearance_post_seq,
                    dtype=np.float32,
                ).copy()
                arr[idx : idx + suffix_len] = post_h[:suffix_len]
                self.committed_planned_clearance_post_seq = arr
                self.committed_planned_h_seq = arr.copy()
                self.committed_planned_min_clearance_seq = arr.copy()
            self.committed_planned_actions = planned_actions
            self.committed_chunk = committed

            return {
                "committed_plan_rebase_available": True,
                "committed_plan_rebased_from_actual_step": True,
                "committed_plan_rebase_index": int(idx),
                "committed_plan_rebase_suffix_len": int(suffix_len),
                "committed_plan_rebase_action_l2": float(action_delta),
                "committed_plan_rebase_post_min_clearance": float(
                    np.min(post_h[:suffix_len])
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Committed plan rebase failed: %s", exc)
            return {
                "committed_plan_rebase_available": False,
                "committed_plan_rebase_error": str(exc),
            }

    def _committed_state_diagnostics(
        self,
        idx: int,
        obs: Any,
        **kwargs: Any,
    ) -> InfoDict:
        """Compare live replay state against cached committed-rollout state."""
        planned_q = self._committed_planned_value(self.committed_planned_q_seq, idx)
        q_actual = self._current_replay_q(obs, **kwargs)
        state_error = None
        state_error_stats: InfoDict = {}
        missing_planned_q = planned_q is None
        if planned_q is not None:
            try:
                planned_q_arr = np.asarray(planned_q, dtype=np.float32).reshape(-1)
                state_error, state_error_stats, _idx, _weights, _diff = self._weighted_state_error(
                    q_actual,
                    planned_q_arr,
                    kind="state",
                )
            except Exception:  # noqa: BLE001
                state_error = None
                state_error_stats = {}
                missing_planned_q = True
        else:
            missing_planned_q = True
        return {
            "committed_state_error": state_error,
            "committed_state_error_threshold": float(
                self.committed_state_error_threshold
            ),
            **state_error_stats,
            "committed_aborted_due_to_state_mismatch": False,
            "committed_replan_due_to_state_mismatch": False,
            "committed_rejected_missing_planned_q": bool(missing_planned_q),
            "planned_q_at_index": self._jsonable_snapshot(planned_q),
            "actual_q_at_replay": self._jsonable_snapshot(q_actual),
            "planned_vs_actual_q_error": state_error,
        }

    def _committed_state_mismatch_info(
        self,
        mode: str,
        idx: int,
        length: int,
        state_info: Mapping[str, Any],
    ) -> InfoDict:
        """Build diagnostics for a committed-plan state mismatch."""
        info = dict(state_info or {})
        state_error = info.get("committed_state_error")
        missing_planned_q = bool(info.get("committed_rejected_missing_planned_q"))
        if missing_planned_q:
            reason = "missing_planned_q"
        else:
            reason = "state_mismatch"
        info.update(
            {
                "committed_chunk_active": True,
                "committed_chunk_mode": mode,
                "committed_chunk_index": int(idx),
                "committed_chunk_length": int(length),
                "committed_rejoin_index": (
                    None
                    if self.committed_rejoin_index is None
                    else int(self.committed_rejoin_index)
                ),
                "committed_until_complete": bool(self.committed_until_complete),
                "committed_chunk_started": bool(idx == 0),
                "committed_chunk_completed": False,
                "committed_aborted_due_to_safety": False,
                "committed_aborted_due_to_state_mismatch": True,
                "committed_replan_due_to_state_mismatch": bool(
                    self.committed_state_error_action == "replan"
                ),
                "committed_abort_reason": reason,
                "committed_abort_step": int(idx),
                "committed_abort_mode": mode,
                "committed_abort_index": int(idx),
                "committed_abort_chunk_length": int(length),
                "recover_steps_executed": 0,
                "deform_steps_executed": 0,
                "return_steps_executed": 0,
                "deform_steps_executed": 0,
                "resume_from_committed_rejoin": False,
                "fallback_used": bool(
                    self.committed_state_error_action == "abort_to_brake"
                ),
                "optimized_accepted": False,
                "optimized_reject_reason": "committed_state_mismatch",
                "fallback_reason": reason,
                "deform_safe": False,
                "is_recoverable": False,
            }
        )
        if state_error is not None:
            info["planned_vs_actual_q_error"] = float(state_error)
        return info

    def _committed_state_mismatch_brake(
        self,
        obs: Any,
        nominal_chunk: np.ndarray,
        original_shape: Any,
        mode: str,
        idx: int,
        total: int,
        state_info: Mapping[str, Any],
    ) -> RecoveryResult:
        """Abort a committed plan by returning a brake action and diagnostics."""
        brake_safety = {
            "horizon_safe": False,
            "min_clearance": float("inf"),
            "first_violation": 0,
            "unsafe_count": 0,
            "safety_eval_available": False,
        }
        braked_chunk, brake_info = self.brake.horizon_brake(
            obs,
            nominal_chunk,
            brake_safety,
        )
        info = self._committed_state_mismatch_info(mode, idx, total, state_info)
        info.update(brake_info)
        info.update(
            {
                "safety_mode": "horizon_brake",
                "mode": "horizon_brake",
                "deform_mode": "committed_recovery_state_mismatch",
                "deformation_source": "horizon_brake",
                "optimized_accepted": False,
                "optimized_fallback": "brake",
                "optimized_reject_reason": "committed_state_mismatch",
                "fallback_reason": "committed_state_mismatch",
                "rejection_cause": "unsafe",
            }
        )
        self.last_info = info
        return braked_chunk.reshape(original_shape), info


    def _try_replan_committed_suffix_from_current_state(
        self,
        obs: Any,
        nominal_chunk: np.ndarray,
        original_shape: Any,
        mode: str,
        idx: int,
        total: int,
        state_info: InfoDict,
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Try replacing the remaining committed suffix after state mismatch."""
        state_info.update(
            {
                "committed_suffix_replan_attempted": False,
                "committed_suffix_replan_accepted": False,
                "committed_suffix_replan_rejected": False,
                "committed_suffix_replan_reject_reason": None,
            }
        )
        if not self.replan_committed_suffix_on_state_mismatch:
            state_info["committed_suffix_replan_reject_reason"] = "disabled"
            return None
        if self.committed_state_error_action != "replan":
            state_info["committed_suffix_replan_reject_reason"] = "not_replan_action"
            return None
        use_mpc_recovery_replan = bool(
            state_info.get("mpc_recovery_replan_attempted", False)
        )
        if (
            not use_mpc_recovery_replan
            and self.max_suffix_replans_per_recovery > 0
            and self.committed_suffix_replans_in_current_recovery
            >= self.max_suffix_replans_per_recovery
        ):
            self.committed_suffix_replan_budget_suppressed_count += 1
            self.recovery_optimizer_cooldown_remaining = max(
                int(self.recovery_optimizer_cooldown_remaining),
                int(self.recover_retry_cooldown_steps),
            )
            if self.recover_max_attempts_per_unsafe_streak > 0:
                self.recovery_attempts_in_unsafe_streak = max(
                    int(self.recovery_attempts_in_unsafe_streak),
                    int(self.recover_max_attempts_per_unsafe_streak),
                )
            state_info.update(
                {
                    "committed_suffix_replan_rejected": True,
                    "committed_suffix_replan_reject_reason": "suffix_replan_budget_exceeded",
                    "committed_suffix_replans_in_current_recovery": int(
                        self.committed_suffix_replans_in_current_recovery
                    ),
                    "max_suffix_replans_per_recovery": int(
                        self.max_suffix_replans_per_recovery
                    ),
                }
            )
            return None

        self.committed_suffix_replan_attempt_count += 1
        state_info.update(
            {
                "committed_suffix_replan_attempted": True,
                "committed_suffix_replan_from_index": int(idx),
                "committed_suffix_replan_old_length": int(total),
            }
        )

        def reject(reason: str, **extra: Any) -> None:
            self.committed_suffix_replan_rejected_count += 1
            state_info.update(
                {
                    "committed_suffix_replan_rejected": True,
                    "committed_suffix_replan_reject_reason": reason,
                    **extra,
                }
            )
            return None

        context = self.recovery_context
        if context is None or context.nominal_q_seq is None or context.nominal_chunk is None:
            return reject("missing_recovery_context")
        if self.committed_chunk is None:
            return reject("missing_committed_chunk")
        if bool(state_info.get("committed_rejected_missing_planned_q")):
            return reject("missing_planned_q")

        committed = np.asarray(self.committed_chunk, dtype=np.float32)
        nominal, _ = self._as_chunk(nominal_chunk)
        if committed.ndim != 2 or nominal.ndim != 2 or committed.shape[0] == 0:
            return reject("invalid_chunk_shape")
        remaining = int(total) - int(idx)
        if remaining < self.committed_suffix_replan_min_remaining:
            return reject("insufficient_remaining_horizon")
        suffix_horizon_limit = int(self.return_horizon)
        if bool(state_info.get("mpc_recovery_replan_attempted", False)):
            suffix_horizon_limit = max(
                1,
                int(state_info.get("mpc_recovery_horizon", self.mpc_recovery_horizon)),
            )
        h = min(remaining, suffix_horizon_limit, nominal.shape[0])
        if h <= 0:
            return reject("empty_suffix_horizon")

        current_q = self._current_replay_q(obs, **kwargs)
        valid = self._valid_control_indices(nominal)
        if not np.any(valid):
            return reject("no_controlled_actions")
        action_idx = self.controlled_action_indices[valid]

        old_suffix = committed[idx : idx + h].copy()
        if old_suffix.shape[0] != h:
            return reject("invalid_old_suffix")
        rejoin_target = self.committed_rejoin_index
        seed_start_index = self._ordered_recovery_start_index(
            rejoin_target,
            h,
            context.nominal_q_seq,
        )
        if seed_start_index is None and rejoin_target is not None:
            try:
                seed_start_index = int(rejoin_target)
            except Exception:  # noqa: BLE001
                seed_start_index = None

        return_obs = self._obs_with_q(obs, current_q)
        return_nominal, seed_target_index = self._make_task_progress_recover_chunk(
            return_obs,
            current_q,
            old_suffix,
            action_idx,
            context=context,
            default_target_index=seed_start_index,
        )
        if return_nominal.shape[0] == 0:
            return reject("empty_return_seed")

        return_rejoin_context = self._make_rejoin_context(
            context.nominal_q_seq,
            context.nominal_ee_seq,
        )
        use_recover_cost = bool(self.safechunk_recover_enabled)
        use_mpc_bridge_direction = bool(
            state_info.get("mpc_recovery_replan_attempted", False)
        )
        mpc_bridge_target_info = None
        if use_mpc_bridge_direction:
            try:
                mpc_bridge_target_info = self.get_nominal_rejoin_target(
                    return_obs,
                    candidate_q=current_q,
                    require_live_prefix_safe=True,
                    live_prefix_len=int(getattr(self, "recover_act_frame_stack", 1)),
                    allow_best_live_prefix_when_unsafe=True,
                )
            except Exception:  # noqa: BLE001
                mpc_bridge_target_info = None
        mpc_bridge_direction_weight = max(
            10.0,
            10.0
            * (
                float(getattr(self, "recover_act_heading_weight", 1.0))
                + float(getattr(self, "recover_act_progress_weight", 1.0))
            ),
        )
        mpc_target_tube_weight = float(
            getattr(self, "mpc_recovery_target_tube_weight", 0.0)
        )

        def add_mpc_target_tube_cost(candidate, cost, losses):
            if (
                not use_mpc_bridge_direction
                or not bool(getattr(self, "mpc_recovery_target_tube_enabled", False))
            ):
                return float(cost), losses
            tube_terms = self.mpc.recovery_target_tube_terms(
                return_obs,
                candidate,
                current_q=current_q,
                target_info=mpc_bridge_target_info,
            )
            losses.update(tube_terms)
            if bool(tube_terms.get("mpc_recovery_target_tube_available", False)):
                tube_loss = float(
                    tube_terms.get("mpc_recovery_target_tube_loss", 0.0)
                )
                cost = float(cost) + mpc_target_tube_weight * tube_loss
                losses["mpc_recovery_target_tube_weight"] = float(
                    mpc_target_tube_weight
                )
                losses["mpc_recovery_target_tube_weighted_loss"] = float(
                    mpc_target_tube_weight * tube_loss
                )
            return float(cost), losses

        def add_mpc_bridge_direction_cost(candidate, cost, losses):
            if not use_mpc_bridge_direction:
                return float(cost), losses
            bridge_terms = self.mpc.bridge_direction_terms(
                return_obs,
                candidate,
                current_q=current_q,
                target_info=mpc_bridge_target_info,
            )
            losses.update(bridge_terms)
            if bool(bridge_terms.get("mpc_bridge_direction_available", False)):
                cost = float(cost) + mpc_bridge_direction_weight * float(
                    bridge_terms.get("mpc_bridge_direction_loss", 0.0)
                )
                losses["mpc_bridge_direction_weight"] = float(
                    mpc_bridge_direction_weight
                )
                losses["mpc_bridge_weighted_direction_loss"] = float(
                    mpc_bridge_direction_weight
                    * float(bridge_terms.get("mpc_bridge_direction_loss", 0.0))
                )
            return float(cost), losses

        def return_cost(candidate):
            if use_recover_cost:
                cost, losses = self._recover_task_progress_cost(
                    return_obs,
                    candidate,
                    return_nominal,
                    action_idx,
                    reference_chunk=return_nominal,
                )
            else:
                cost, losses = self._return_deformation_cost(
                    return_obs,
                    candidate,
                    return_nominal,
                    context.nominal_q_seq,
                    return_rejoin_context,
                    action_idx,
                )
            cost, losses = add_mpc_bridge_direction_cost(candidate, cost, losses)
            return add_mpc_target_tube_cost(candidate, cost, losses)

        def return_batch_cost(candidates: np.ndarray) -> Any:
            if use_recover_cost:
                return self._recover_task_progress_cost_batch(
                    return_obs,
                    candidates,
                    return_nominal,
                    action_idx,
                    reference_chunk=return_nominal,
                )
            return self._return_deformation_cost_batch(
                return_obs,
                candidates,
                return_nominal,
                context.nominal_q_seq,
                return_rejoin_context,
                action_idx,
            )

        def return_early_stop(record):
            losses = record.get("losses", {})
            if float(losses.get("min_clearance", float("-inf"))) < self._acceptance_clearance_threshold():
                return False
            if (
                use_mpc_bridge_direction
                and self.mpc_bridge_direction_hard_gate
                and not bool(losses.get("mpc_bridge_direction_ok", True))
            ):
                return False
            if (
                use_mpc_bridge_direction
                and bool(getattr(self, "mpc_recovery_target_tube_enabled", False))
            ):
                if not bool(losses.get("mpc_recovery_target_tube_ok", False)):
                    return False
                if (
                    bool(getattr(self, "mpc_recovery_target_tube_require_progress", True))
                    and not bool(losses.get("mpc_recovery_target_tube_progress_ok", False))
                ):
                    return False
            if use_recover_cost:
                if not bool(losses.get("recover_direction_ok", True)):
                    return False
                if not bool(losses.get("recover_act_progress_ok", True)):
                    return False
                if not bool(losses.get("recover_act_heading_ok", True)):
                    return False
                if not bool(
                    losses.get(
                        "recover_ordered_waypoint_tube_ok",
                        losses.get("recover_ordered_ok", True),
                    )
                ):
                    return False
                return float(losses.get("recover_task_progress_score", 0.0)) > 0.0
            rejoin_loss = float(
                losses.get(
                    "rejoin_loss",
                    losses.get("return_rejoin_loss", float("inf")),
                )
            )
            return self._sqrt_loss(rejoin_loss) < self.q_rejoin_threshold

        self.recovery_replan_count += 1
        seed_chunks = [return_nominal]
        if old_suffix.shape == return_nominal.shape:
            seed_chunks.append(old_suffix)
        record = self._optimize_controlled_chunk(
            return_obs,
            return_nominal,
            action_idx,
            return_cost,
            seed_chunks=seed_chunks,
            batch_cost_fn=None if use_mpc_bridge_direction else return_batch_cost,
            early_stop_fn=return_early_stop,
            optimizer_stage="committed_suffix",
        )
        return_chunk = np.asarray(record["chunk"], dtype=np.float32)
        terminal = self._recovery_terminal_rejoin_info(
            return_obs,
            return_chunk,
            context,
            return_rejoin_context,
            default_target_index=seed_target_index,
        )
        path_info = self.evaluate_recovery_path_safety(
            return_obs,
            return_chunk,
            candidate_name="committed_suffix_replan",
        )
        direction_info = self.compute_nominal_rejoin_score(
            return_chunk,
            return_nominal,
            obs=return_obs,
        )
        direction_terms = self._recover_direction_alignment_terms(direction_info)
        suffix_losses = dict(record.get("losses", {}))
        if use_mpc_bridge_direction:
            bridge_terms = self.mpc.bridge_direction_terms(
                return_obs,
                return_chunk,
                current_q=current_q,
                target_info=mpc_bridge_target_info,
            )
            suffix_losses.update(bridge_terms)
            record.setdefault("losses", {}).update(bridge_terms)
            if bool(getattr(self, "mpc_recovery_target_tube_enabled", False)):
                tube_terms = self.mpc.recovery_target_tube_terms(
                    return_obs,
                    return_chunk,
                    current_q=current_q,
                    target_info=mpc_bridge_target_info,
                )
                suffix_losses.update(tube_terms)
                record.setdefault("losses", {}).update(tube_terms)
        suffix_task_progress_ok = bool(
            suffix_losses.get("recover_act_progress_ok", True)
        ) and bool(suffix_losses.get("recover_act_heading_ok", True))
        if use_mpc_bridge_direction:
            record.setdefault("losses", {})["mpc_bridge_direction_hard_gate"] = bool(
                self.mpc_bridge_direction_hard_gate
            )
        if use_mpc_bridge_direction and self.mpc_bridge_direction_hard_gate:
            suffix_task_progress_ok = suffix_task_progress_ok and bool(
                suffix_losses.get("mpc_bridge_direction_ok", True)
            )
        if (
            use_mpc_bridge_direction
            and bool(getattr(self, "mpc_recovery_target_tube_enabled", False))
            and bool(getattr(self, "mpc_recovery_target_tube_require_progress", True))
        ):
            suffix_task_progress_ok = suffix_task_progress_ok and bool(
                suffix_losses.get("mpc_recovery_target_tube_progress_ok", False)
            )
        mpc_target_tube_ok = bool(
            use_mpc_bridge_direction
            and bool(getattr(self, "mpc_recovery_target_tube_enabled", False))
            and bool(suffix_losses.get("mpc_recovery_target_tube_ok", False))
        )
        reject_reason = self._recovery_reject_reason(
            {**terminal, **suffix_losses},
            path_info,
            task_progress_ok=suffix_task_progress_ok,
            direction_ok=bool(direction_terms["recover_direction_ok"]),
            ordered_ok=bool(
                terminal.get(
                    "recover_ordered_waypoint_tube_ok",
                    terminal.get("recover_ordered_ok", True),
                )
            ),
        )
        required_margin = float(self.min_clearance + self.committed_execution_margin)
        if (
            reject_reason is None
            and use_mpc_bridge_direction
            and self.mpc_bridge_direction_hard_gate
            and not bool(suffix_losses.get("mpc_bridge_direction_ok", True))
        ):
            reject_reason = "mpc_bridge_direction"
        if (
            reject_reason == "q_rejoin_failed"
            and mpc_target_tube_ok
            and suffix_task_progress_ok
        ):
            override_accept_info = {
                **terminal,
                **suffix_losses,
                "q_rejoin_ok": True,
                "mpc_recovery_q_rejoin_overridden_by_tube": True,
            }
            override_reject_reason = self._recovery_reject_reason(
                override_accept_info,
                path_info,
                task_progress_ok=suffix_task_progress_ok,
                direction_ok=bool(direction_terms["recover_direction_ok"]),
                ordered_ok=bool(
                    terminal.get(
                        "recover_ordered_waypoint_tube_ok",
                        terminal.get("recover_ordered_ok", True),
                    )
                ),
            )
            if override_reject_reason is None:
                suffix_losses["mpc_recovery_q_rejoin_overridden_by_tube"] = True
                suffix_losses["mpc_recovery_original_reject_reason"] = "q_rejoin_failed"
                reject_reason = None
            else:
                suffix_losses["mpc_recovery_q_rejoin_override_blocked_reason"] = str(
                    override_reject_reason
                )
                reject_reason = override_reject_reason
        if reject_reason is None and float(terminal.get("min_clearance", float("-inf"))) < required_margin:
            reject_reason = "committed_margin"
        if reject_reason is not None:
            return reject(
                reject_reason,
                committed_suffix_replan_min_clearance=float(
                    terminal.get("min_clearance", float("-inf"))
                ),
                committed_suffix_replan_required_clearance=required_margin,
                committed_suffix_replan_target_index=terminal.get("target_index"),
                committed_suffix_replan_seed_start_index=seed_start_index,
                committed_suffix_replan_bridge_direction_ok=suffix_losses.get(
                    "mpc_bridge_direction_ok"
                ),
                committed_suffix_replan_bridge_heading_cosine=suffix_losses.get(
                    "mpc_bridge_heading_cosine"
                ),
                committed_suffix_replan_bridge_progress_projection=suffix_losses.get(
                    "mpc_bridge_progress_projection"
                ),
                committed_suffix_replan_target_tube_ok=suffix_losses.get(
                    "mpc_recovery_target_tube_ok"
                ),
                committed_suffix_replan_target_tube_dist=suffix_losses.get(
                    "mpc_recovery_target_tube_terminal_dist"
                ),
                committed_suffix_replan_target_tube_threshold=suffix_losses.get(
                    "mpc_recovery_target_tube_dist_threshold"
                ),
                committed_suffix_replan_target_tube_progress_ok=suffix_losses.get(
                    "mpc_recovery_target_tube_progress_ok"
                ),
            )

        losses = dict(record.get("losses", {}))
        losses.update(
            {
                "return_rejoin_loss": float(terminal["q_rejoin_loss"]),
                "recover_rejoin_loss": float(terminal["q_rejoin_loss"]),
                "q_rejoin_loss": float(terminal["q_rejoin_loss"]),
                "q_rejoin_dist": float(terminal["q_rejoin_dist"]),
                "q_rejoin_threshold": float(self.q_rejoin_threshold),
                "q_rejoin_index": terminal.get("target_index"),
                "qd_rejoin_loss": float(terminal["qd_rejoin_loss"]),
                "qd_rejoin_dist": float(terminal["qd_rejoin_dist"]),
                "qd_rejoin_threshold": float(self.qd_rejoin_threshold),
                "qd_rejoin_index": terminal.get("qd_rejoin_index"),
                "qd_rejoin_ok": self._coerce_bool(terminal.get("qd_rejoin_ok", False)),
                "qd_rejoin_required": bool(
                    terminal.get("qd_rejoin_required", self.require_qd_rejoin)
                ),
                "qd_rejoin_hard_threshold": float(
                    terminal.get(
                        "qd_rejoin_hard_threshold",
                        self.qd_rejoin_hard_threshold,
                    )
                ),
                "qd_rejoin_hard_failed": bool(
                    terminal.get("qd_rejoin_hard_failed", False)
                ),
                "qd_rejoin_soft_ok": bool(
                    terminal.get("qd_rejoin_soft_ok", False)
                ),
                "return_qd_rejoin_loss": float(terminal["qd_rejoin_loss"]),
                "return_qd_rejoin_index": terminal.get("qd_rejoin_index"),
                "rejoin_q_eval_time_ms": float(terminal["q_eval_time_ms"]),
                "rejoin_qd_eval_time_ms": float(terminal["qd_eval_time_ms"]),
                "recover_min_clearance": float(terminal["min_clearance"]),
                "recover_path_min_clearance": float(
                    path_info.get("recover_path_min_clearance", terminal["min_clearance"])
                ),
                "recover_immediate_clearance": float(
                    path_info.get("recover_immediate_clearance", float("-inf"))
                ),
                "recover_prefix_min_clearance": float(
                    path_info.get("recover_prefix_min_clearance", float("-inf"))
                ),
                "recover_path_safe": bool(path_info.get("path_safe", False)),
                "recover_immediate_safe": bool(path_info.get("immediate_safe", False)),
                "recover_prefix_safe": bool(path_info.get("prefix_safe", False)),
                "recover_safe_prefix_len": int(path_info.get("safe_prefix_len", 0) or 0),
                "recover_reject_reason": None,
                "recovery_candidate_class": "committed_suffix_replan",
                "committed_suffix_replan_attempted": True,
                "committed_suffix_replan_accepted": True,
                "committed_suffix_replan_rejected": False,
                "committed_suffix_replan_reject_reason": None,
                "committed_suffix_replan_from_index": int(idx),
                "committed_suffix_replan_old_length": int(total),
                "committed_suffix_replan_new_length": int(return_chunk.shape[0]),
                "committed_suffix_replan_target_index": terminal.get("target_index"),
                "committed_suffix_replan_seed_start_index": seed_start_index,
                **direction_info,
                **direction_terms,
            }
        )
        losses.update(
            {
                key: terminal.get(key)
                for key in (
                    "recover_ordered_path_available",
                    "recover_ordered_target_index",
                    "recover_ordered_horizon",
                    "recover_ordered_pose_loss",
                    "recover_ordered_delta_loss",
                    "recover_ordered_waypoint_pose_loss",
                    "recover_ordered_waypoint_rmse",
                    "recover_ordered_heading_loss",
                    "recover_ordered_heading_cosine",
                    "recover_ordered_heading_cosine_min",
                    "recover_ordered_heading_cosine_threshold",
                    "recover_ordered_backtrack_count",
                    "recover_ordered_monotonic_ok",
                    "recover_ordered_pose_tube_threshold",
                    "recover_ordered_pose_tube_ok",
                    "recover_ordered_waypoint_tube_ok",
                    "recover_ordered_strict_ok",
                    "recover_ordered_waypoint_index_start",
                    "recover_ordered_waypoint_index_end",
                    "recover_ordered_loss",
                    "recover_ordered_pose_weight",
                    "recover_ordered_delta_weight",
                    "recover_ordered_heading_weight",
                    "recover_ordered_pose_threshold",
                    "recover_ordered_delta_threshold",
                    "recover_ordered_ok",
                )
                if terminal.get(key) is not None
            }
        )
        self._record_ordered_recovery_terms(losses)

        commit_info = dict(losses)
        commit_info.update(
            {
                "optimized_accepted": True,
                "recover_accepted": True,
                "deform_stage_accepted": True,
                "deform_chunk_length": 0,
                "recover_chunk_length": int(return_chunk.shape[0]),
                "recover_target_index": terminal.get("target_index"),
                "return_target_index": terminal.get("target_index"),
                "rejoin_index": terminal.get("target_index"),
                "act_resume_index": terminal.get("target_index"),
                "recover_min_clearance": float(terminal["min_clearance"]),
                "recover_rejoin_loss": float(terminal["q_rejoin_loss"]),
                "rejection_cause": None,
                "fallback_used": False,
                "deform_safe": True,
                "is_safe": True,
                "is_recoverable": True,
            }
        )

        old_context = self.recovery_context
        self._clear_committed_chunk()
        self.recovery_context = old_context
        commit_kwargs = dict(kwargs)
        commit_kwargs["q_full"] = current_q
        committed, commit_reject_info = self._commit_explicit_recovery_chunk(
            return_obs,
            return_chunk,
            commit_info,
            **commit_kwargs,
        )
        if not committed:
            return reject(
                "commit_rejected",
                committed_suffix_replan_commit_reject_info=commit_reject_info,
            )

        self.committed_suffix_replan_accepted_count += 1
        self.committed_suffix_replans_in_current_recovery += 1
        served = self._serve_committed_chunk(
            return_obs,
            nominal,
            original_shape,
            **commit_kwargs,
        )
        if served is None:
            return reject("serve_replanned_suffix_failed")
        served_chunk, served_info = served
        served_info.update(
            {
                key: value
                for key, value in losses.items()
                if str(key).startswith(("mpc_bridge_", "mpc_recovery_target_tube_"))
                or key in {
                    "mpc_recovery_q_rejoin_overridden_by_tube",
                    "mpc_recovery_original_reject_reason",
                    "mpc_recovery_planned_q_seq",
                    "mpc_recovery_planned_action_seq",
                }
            }
        )
        served_info.update(
            {
                "committed_state_mismatch_detected": True,
                "committed_state_mismatch_recovered": True,
                "committed_aborted_due_to_state_mismatch": False,
                "committed_replan_due_to_state_mismatch": True,
                "committed_suffix_replan_attempted": True,
                "committed_suffix_replan_accepted": True,
                "committed_suffix_replan_rejected": False,
                "committed_suffix_replan_reject_reason": None,
                "committed_suffix_replan_from_index": int(idx),
                "committed_suffix_replan_old_length": int(total),
                "committed_suffix_replan_new_length": int(return_chunk.shape[0]),
                "committed_suffix_replan_target_index": terminal.get("target_index"),
                "committed_suffix_replan_seed_start_index": seed_start_index,
                "committed_state_error": state_info.get("committed_state_error"),
                "committed_state_error_threshold": state_info.get(
                    "committed_state_error_threshold"
                ),
                "planned_q_at_index_before_suffix_replan": state_info.get(
                    "planned_q_at_index"
                ),
                "actual_q_at_suffix_replan": state_info.get("actual_q_at_replay"),
            }
        )
        self.last_info = served_info
        return served_chunk, served_info

    def _committed_replay_diagnostics(
        self,
        idx: int,
        action: np.ndarray,
        safety_info: Mapping[str, Any],
        **kwargs: Any,
    ) -> InfoDict:
        """Summarize prediction-vs-replay errors for committed recovery."""
        planned_pre_q = self._committed_planned_value(self.committed_planned_q_seq, idx)
        planned_post_q = self._committed_planned_value(
            self.committed_planned_post_q_seq,
            idx,
        )
        planned_action = self._committed_planned_value(self.committed_planned_actions, idx)
        planned_clearance_pre = self._committed_planned_value(
            self.committed_planned_clearance_pre_seq,
            idx,
        )
        planned_clearance_post = self._committed_planned_value(
            self.committed_planned_clearance_post_seq,
            idx,
        )
        if planned_clearance_post is None:
            planned_clearance_post = self._committed_planned_value(
                self.committed_planned_h_seq,
                idx,
            )
        replay_pre_q = safety_info.get("actual_pre_action_q", kwargs.get("q_full"))
        replay_post_q = safety_info.get("replay_predicted_post_action_q")
        replay_clearance_pre = safety_info.get("replay_clearance_pre")
        replay_clearance_post = safety_info.get(
            "replay_clearance_post",
            safety_info.get("min_clearance", float("-inf")),
        )
        min_clearance = float(replay_clearance_post)
        qd_actual = kwargs.get("qd_full")

        def _norm_error(a: Any, b: Any) -> float | None:
            if a is None or b is None:
                return None
            try:
                arr_a = np.asarray(a, dtype=np.float32).reshape(-1)
                arr_b = np.asarray(b, dtype=np.float32).reshape(-1)
                n = min(arr_a.size, arr_b.size)
                if n == 0:
                    return None
                return float(np.linalg.norm(arr_a[:n] - arr_b[:n]))
            except Exception:  # noqa: BLE001
                return None

        def _float_or_none(value: Any) -> float | None:
            if value is None:
                return None
            try:
                value = float(value)
                return value if np.isfinite(value) else None
            except Exception:  # noqa: BLE001
                return None

        action_error = _norm_error(planned_action, action)
        pre_q_error = _norm_error(planned_pre_q, replay_pre_q)
        post_q_error = _norm_error(planned_post_q, replay_post_q)
        planned_pre_float = _float_or_none(planned_clearance_pre)
        planned_post_float = _float_or_none(planned_clearance_post)
        replay_pre_float = _float_or_none(replay_clearance_pre)
        replay_post_float = _float_or_none(replay_clearance_post)
        pre_clearance_error = (
            None
            if planned_pre_float is None or replay_pre_float is None
            else float(replay_pre_float - planned_pre_float)
        )
        post_clearance_error = (
            None
            if planned_post_float is None or replay_post_float is None
            else float(replay_post_float - planned_post_float)
        )
        human_motion = self._human_motion_since_plan(kwargs.get("human_state"))
        live_monitor_clearance = kwargs.get(
            "live_monitor_min_h",
            kwargs.get("min_h"),
        )

        def _finite_float(value: Any) -> float | None:
            try:
                value = float(value)
                return value if np.isfinite(value) else None
            except Exception:  # noqa: BLE001
                return None

        live_monitor_float = _finite_float(live_monitor_clearance)
        execution_clearances = [
            value
            for value in (replay_post_float,)
            if value is not None
        ]
        execution_min_clearance = (
            float(min(execution_clearances)) if execution_clearances else None
        )
        return {
            "actual_one_step_clearance": min_clearance,
            "committed_execution_min_clearance": execution_min_clearance,
            "committed_live_monitor_clearance": live_monitor_float,
            "committed_replay_pre_clearance": replay_pre_float,
            "committed_replay_post_clearance": replay_post_float,
            "planned_clearance_for_this_index": planned_post_float,
            "clearance_prediction_error": post_clearance_error,
            "planned_min_clearance_at_index": planned_post_float,
            "planned_h_at_index": planned_post_float,
            "planned_q_at_index": self._jsonable_snapshot(planned_pre_q),
            "planned_pre_action_q": self._jsonable_snapshot(planned_pre_q),
            "planned_post_action_q": self._jsonable_snapshot(planned_post_q),
            "planned_action_at_index": self._jsonable_snapshot(planned_action),
            "predicted_post_action_q": self._jsonable_snapshot(planned_post_q),
            "actual_pre_action_q": self._jsonable_snapshot(replay_pre_q),
            "replay_predicted_post_action_q": self._jsonable_snapshot(replay_post_q),
            "committed_action": self._jsonable_snapshot(action),
            "planned_clearance_pre": planned_pre_float,
            "planned_clearance_post": planned_post_float,
            "replay_clearance_pre": replay_pre_float,
            "replay_clearance_post": replay_post_float,
            "actual_vs_planned_pre_q_error": pre_q_error,
            "actual_vs_planned_post_q_error": post_q_error,
            "planned_vs_actual_q_error": pre_q_error,
            "planned_vs_actual_action_error": action_error,
            "planned_vs_actual_action_error_controlled": action_error,
            "planning_vs_replay_human_error": human_motion,
            "planning_vs_replay_clearance_pre_error": pre_clearance_error,
            "planning_vs_replay_clearance_post_error": post_clearance_error,
            "human_motion_since_plan": human_motion,
            "planning_human_state_snapshot": self.committed_planning_human_state_snapshot,
            "replay_human_state": self._snapshot_human_state(kwargs.get("human_state")),
            "control_type": self.control_type,
            "dt": float(self.dt),
            "controlled_state_indices": self._jsonable_snapshot(
                self.controlled_state_indices
            ),
            "controlled_action_indices": self._jsonable_snapshot(
                self.controlled_action_indices
            ),
            "action_conversion_mode": self.control_type,
            "accepted_min_clearance": self.committed_accepted_min_clearance,
            "accepted_clearance_margin": self.committed_accepted_clearance_margin,
            "accepted_human_state_snapshot": self.committed_accepted_human_state_snapshot,
            "committed_abort_robot_q": self._jsonable_snapshot(replay_pre_q),
            "committed_abort_robot_qd": self._jsonable_snapshot(qd_actual),
            "committed_abort_human_state": self._snapshot_human_state(
                kwargs.get("human_state")
            ),
        }

    def _committed_abort_reason(
        self,
        diagnostics: Mapping[str, Any],
        min_clearance: float,
    ) -> str:
        """Classify why committed recovery should abort."""
        human_motion = diagnostics.get("planning_vs_replay_human_error")
        if human_motion is None:
            human_motion = diagnostics.get("human_motion_since_plan")
        if human_motion is not None and float(human_motion) > 1e-6:
            return "human_motion_after_planning"
        post_error = diagnostics.get("planning_vs_replay_clearance_post_error")
        if post_error is None:
            post_error = diagnostics.get("clearance_prediction_error")
        if post_error is not None and float(post_error) < -self.committed_safety_tol:
            return "safety_semantics_mismatch"
        return "clearance_below_abort_threshold"

    def _repair_committed_action(
        self,
        obs: Any,
        action: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run one-step repair on a committed action, falling back to original."""
        try:
            repaired = self._call_single_step_operator(action, obs, **kwargs)
            return np.asarray(repaired, dtype=np.float32).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Committed action repair failed: %s", exc)
            return np.asarray(action, dtype=np.float32).reshape(-1)

    def _committed_recovery_phase(self, mode: str) -> str:
        """Map committed chunk modes to public recovery phase labels."""
        if mode == "recover":
            return "recover"
        if mode == "pass_through":
            return "pass_through"
        return "horizon_deform"


    def _resume_readiness_terms(
        self,
        info: Mapping[str, Any] | None,
        *,
        q: Any | None = None,
        obs: Any | None = None,
        source: str = "resume",
    ) -> InfoDict:
        """Single gate for every path that can release control back to ACT."""

        score_input: InfoDict = dict(info or {})
        if q is not None:
            try:
                score_input.update(
                    self._candidate_resume_affordance_features(
                        np.asarray(q, dtype=np.float32).reshape(1, -1),
                        obs=obs,
                        source=f"{source}_ee_pose",
                    )
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug("%s resume readiness EE feature failed: %s", source, exc)
        tube_terms = self._resume_tube_score_terms(score_input)
        affordance_terms = self._resume_affordance_score_terms(score_input)
        tube_ok = bool(tube_terms.get("resume_tube_ok", True))
        affordance_ok = bool(affordance_terms.get("resume_affordance_ok", True))
        allowed = bool(tube_ok and affordance_ok)
        if not tube_ok:
            block_reason = "resume_tube_not_ready"
        elif not affordance_ok:
            block_reason = "resume_affordance_not_ready"
        else:
            block_reason = None
        out: InfoDict = {}
        out.update(tube_terms)
        out.update(affordance_terms)
        out.update(
            {
                "resume_readiness_source": source,
                "resume_allowed": bool(allowed),
                "resume_block_reason": block_reason,
            }
        )
        return out

    def _act_release_terms(
        self,
        *,
        reason: str,
        allowed: bool,
        block_reason: str | None = None,
        reset_history: bool = True,
        resume_index: Any | None = None,
        activate_window: bool = True,
        degraded: bool = False,
    ) -> InfoDict:
        """Canonical flag writer for every path that can hand control back to ACT."""

        release_reason = str(reason)
        release_allowed = bool(allowed) and not bool(degraded)
        if not release_allowed and block_reason is None:
            block_reason = (
                "degraded_release_not_allowed" if degraded else "resume_not_ready"
            )
        act_resume_index = None
        if release_allowed and resume_index is not None:
            try:
                act_resume_index = int(resume_index)
            except (TypeError, ValueError):
                act_resume_index = None
        if release_allowed and activate_window:
            self._activate_post_recovery_act_window()

        released_for_act_resume = bool(
            release_allowed
            and release_reason
            in {
                "mpc_handoff",
                "opportunistic_resume",
                "committed_soft_handoff",
            }
        )
        from_committed_rejoin = bool(
            release_allowed and release_reason == "committed_rejoin"
        )
        terms: InfoDict = {
            "act_release_attempted": True,
            "act_release_reason": release_reason,
            "act_release_allowed": bool(release_allowed),
            "act_release_blocked": not bool(release_allowed),
            "act_release_block_reason": block_reason,
            "act_release_degraded": bool(degraded),
            "committed_released_for_act_resume": released_for_act_resume,
            "resume_from_committed_rejoin": from_committed_rejoin,
            "request_action_history_reset_after_recovery": bool(
                release_allowed and reset_history
            ),
            "act_resume_index": act_resume_index,
            "act_resume_supported": False,
        }
        if release_reason == "committed_rejoin":
            terms.update(
                {
                    "committed_rejoin_resume_allowed": bool(release_allowed),
                    "committed_rejoin_resume_blocked": not bool(release_allowed),
                    "committed_rejoin_resume_block_reason": block_reason,
                }
            )
        elif release_reason == "opportunistic_resume":
            terms.update(
                {
                    "committed_opportunistic_resume_allowed": bool(release_allowed),
                    "committed_opportunistic_resume_block_reason": block_reason,
                }
            )
        elif release_reason == "committed_soft_handoff":
            terms.update(
                {
                    "committed_soft_handoff_release_to_main_filter": bool(
                        release_allowed
                    ),
                    "committed_soft_handoff_release_block_reason": block_reason,
                }
            )
        elif release_reason == "emergency_budget_exit":
            terms.update(
                {
                    "committed_recovery_budget_exit_release_allowed": bool(
                        release_allowed
                    ),
                    "committed_recovery_budget_exit_release_block_reason": block_reason,
                }
            )
        return terms

    def _fill_recover_affordance_metrics(self, info: InfoDict | None) -> InfoDict:
        """Ensure selected recovery diagnostics expose recover_resume_affordance_* fields."""

        if info is None:
            return {}
        terms = self._resume_affordance_score_terms(info)
        mapping = {
            "recover_resume_affordance_enabled": terms.get("resume_affordance_enabled"),
            "recover_resume_affordance_available": terms.get("resume_affordance_available"),
            "recover_resume_affordance_task_relevant": terms.get("resume_affordance_task_relevant"),
            "recover_resume_affordance_score": terms.get("resume_affordance_score"),
            "recover_resume_affordance_ok": terms.get("resume_affordance_ok"),
            "recover_resume_affordance_min_score": terms.get("resume_affordance_min_score"),
            "recover_resume_affordance_component_score": terms.get("resume_affordance_component_score"),
            "recover_resume_affordance_min_component_score": terms.get("resume_affordance_min_component_score"),
            "recover_resume_affordance_target_distance": terms.get("resume_affordance_target_distance"),
            "recover_resume_affordance_target_distance_score": terms.get("resume_affordance_target_distance_score"),
            "recover_resume_affordance_contact_score": terms.get("resume_affordance_contact_score"),
            "recover_resume_affordance_progress_score": terms.get("resume_affordance_progress_score"),
            "recover_resume_affordance_alignment_score": terms.get("resume_affordance_alignment_score"),
            "recover_resume_affordance_continuity_score": terms.get("resume_affordance_continuity_score"),
            "recover_resume_affordance_safety_score": terms.get("resume_affordance_safety_score"),
            "recover_resume_affordance_interaction_context": terms.get("interaction_context"),
        }
        for key, value in mapping.items():
            if info.get(key) is None and value is not None:
                info[key] = value
        return info

    def _committed_rejoin_resume_tube_terms(self, info: Mapping[str, Any]) -> InfoDict:
        """Score the actual committed rejoin / ACT-history-reset handover point."""
        terminal_dist = info.get(
            "recover_resume_tube_terminal_dist",
            info.get(
                "resume_tube_terminal_dist",
                info.get("mpc_recovery_target_tube_terminal_dist", info.get("q_rejoin_dist")),
            ),
        )
        terminal_threshold = info.get(
            "recover_resume_tube_terminal_threshold",
            info.get(
                "resume_tube_terminal_threshold",
                info.get("mpc_recovery_target_tube_dist_threshold", info.get("q_rejoin_threshold")),
            ),
        )
        terminal_delta = info.get(
            "recover_resume_tube_terminal_delta",
            info.get(
                "resume_tube_terminal_delta",
                info.get(
                    "mpc_recovery_target_tube_terminal_delta",
                    info.get("mpc_recovery_target_tube_q_error"),
                ),
            ),
        )
        ordered_loss = info.get(
            "recover_resume_tube_ordered_loss",
            info.get(
                "resume_tube_ordered_loss",
                info.get("mpc_recovery_target_tube_ordered_loss", info.get("recover_ordered_loss")),
            ),
        )
        prefix_min_clearance = info.get(
            "recover_resume_tube_prefix_min_clearance",
            info.get(
                "resume_tube_prefix_min_clearance",
                info.get(
                    "recover_prefix_min_clearance",
                    info.get("accepted_min_clearance", info.get("min_clearance")),
                ),
            ),
        )
        prefix_safe = info.get(
            "recover_resume_tube_prefix_safe",
            info.get("resume_tube_prefix_safe", info.get("recover_prefix_safe", True)),
        )
        progress_projection = info.get(
            "resume_tube_progress_projection",
            info.get(
                "recover_act_progress_projection",
                info.get("mpc_recovery_target_tube_progress_projection"),
            ),
        )
        heading_cosine = info.get(
            "resume_tube_heading_cosine",
            info.get(
                "recover_act_heading_cosine_min",
                info.get(
                    "recover_act_heading_cosine",
                    info.get("mpc_recovery_target_tube_heading_cosine"),
                ),
            ),
        )
        score_input = {
            **dict(info),
            "resume_tube_terminal_dist": terminal_dist,
            "resume_tube_terminal_delta": terminal_delta,
            "resume_tube_q_error": terminal_delta,
            "resume_tube_terminal_threshold": terminal_threshold,
            "resume_tube_ordered_loss": ordered_loss,
            "resume_tube_prefix_min_clearance": prefix_min_clearance,
            "resume_tube_prefix_safe": prefix_safe,
            "resume_tube_progress_projection": progress_projection,
            "resume_tube_heading_cosine": heading_cosine,
        }
        q_for_affordance = info.get("actual_q_at_replay", info.get("current_q"))
        readiness_terms = self._resume_readiness_terms(
            score_input,
            q=q_for_affordance,
            source="committed_rejoin_actual",
        )
        out = {
            f"committed_rejoin_{key}": value
            for key, value in readiness_terms.items()
        }
        return out

    def _committed_info(
        self,
        safety_info: Mapping[str, Any] | None,
        mode: str,
        index: int,
        length: int,
        completed: bool = False,
        aborted: bool = False,
        repaired: bool = False,
        repair_info: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> InfoDict:
        """Assemble diagnostics for a served committed recovery step."""
        info = dict(safety_info or {})
        rejoin_index = self.committed_rejoin_index
        recovery_phase = self._committed_recovery_phase(mode)
        committed_resume_candidate = bool(
            completed and rejoin_index is not None and not aborted
        )
        info.update(
            {
                "safety_mode": "horizon_deform" if not aborted else "horizon_brake",
                "mode": "committed_explicit_recovery" if not aborted else "horizon_brake",
                "deform_mode": "committed_explicit_recovery" if not aborted else "committed_recovery_aborted",
                "deformation_source": "committed_explicit_recovery" if not aborted else "horizon_brake",
                "recovery_mode": recovery_phase,
                "recovery_phase": recovery_phase,
                "committed_chunk_active": True,
                "committed_chunk_mode": mode,
                "committed_chunk_index": int(index),
                "committed_chunk_length": int(length),
                "committed_rejoin_index": None if rejoin_index is None else int(rejoin_index),
                "committed_until_complete": bool(self.committed_until_complete),
                "committed_chunk_started": bool(index == 0),
                "committed_chunk_completed": bool(completed),
                "committed_aborted_due_to_safety": bool(aborted),
                "committed_repaired_step": bool(repaired),
                "committed_repair_min_clearance": (
                    None
                    if repair_info is None
                    else repair_info.get("committed_repair_min_clearance")
                ),
                "committed_repair_clearance_gain": (
                    None
                    if repair_info is None
                    else repair_info.get("committed_repair_clearance_gain")
                ),
                "recover_steps_executed": int(mode == "recover" and not aborted),
                "deform_steps_executed": int(mode == "horizon_deform" and not aborted),
                "return_steps_executed": int(mode == "recover" and not aborted),
                "deform_steps_executed": int(mode == "horizon_deform" and not aborted),
                "resume_from_committed_rejoin": False,
                "request_action_history_reset_after_recovery": False,
                "suppress_outer_pause": not bool(aborted),
                "act_resume_index": None,
                "act_resume_supported": False,
                "fallback_used": bool(aborted),
                "deform_safe": not bool(aborted),
                "is_recoverable": not bool(aborted),
            }
        )
        if self.committed_rejoin_diagnostics:
            for key, value in self.committed_rejoin_diagnostics.items():
                info.setdefault(key, value)
        info.update(self._safechunk_recovery_corridor_info())
        self._fill_recover_affordance_metrics(info)
        info.setdefault(
            "post_recovery_act_window_interrupted",
            bool(aborted),
        )
        if extra:
            info.update(extra)
        if committed_resume_candidate:
            committed_resume_terms = self._committed_rejoin_resume_tube_terms(info)
            info.update(committed_resume_terms)
            committed_resume_tube_ok = bool(
                committed_resume_terms.get("committed_rejoin_resume_tube_ok", True)
            )
            committed_resume_affordance_ok = bool(
                committed_resume_terms.get(
                    "committed_rejoin_resume_affordance_ok",
                    True,
                )
            )
            committed_resume_allowed = bool(
                committed_resume_terms.get(
                    "committed_rejoin_resume_allowed",
                    committed_resume_tube_ok and committed_resume_affordance_ok,
                )
            )
            if not committed_resume_tube_ok:
                block_reason = "resume_tube_not_ready"
            elif not committed_resume_affordance_ok:
                block_reason = "resume_affordance_not_ready"
            elif not committed_resume_allowed:
                block_reason = "resume_not_ready"
            else:
                block_reason = None
            info.update(
                self._act_release_terms(
                    reason="committed_rejoin",
                    allowed=committed_resume_allowed,
                    block_reason=block_reason,
                    reset_history=True,
                    resume_index=rejoin_index,
                    activate_window=True,
                )
            )
        if aborted:
            human_motion = info.get("human_motion_since_plan")
            prediction_error = info.get("clearance_prediction_error")
            abort_reason = info.get(
                "committed_abort_reason",
                "clearance_below_abort_threshold",
            )
            abort_due_to_human_motion = abort_reason == "human_motion_after_planning"
            if human_motion is not None:
                try:
                    abort_due_to_human_motion = (
                        abort_due_to_human_motion or float(human_motion) > 1e-6
                    )
                except (TypeError, ValueError):
                    pass
            semantics_error = info.get("planning_vs_replay_clearance_post_error")
            abort_due_to_semantics_mismatch = (
                abort_reason == "safety_semantics_mismatch"
            )
            if semantics_error is not None:
                try:
                    abort_due_to_semantics_mismatch = (
                        abort_due_to_semantics_mismatch
                        or (
                            float(semantics_error) < -self.committed_safety_tol
                            and not abort_due_to_human_motion
                        )
                    )
                except (TypeError, ValueError):
                    pass
            abort_due_to_prediction_error = abort_reason == "clearance_prediction_error"
            if prediction_error is not None:
                try:
                    abort_due_to_prediction_error = (
                        abort_due_to_prediction_error
                        or float(prediction_error) < -self.committed_safety_tol
                    )
                except (TypeError, ValueError):
                    pass
            info.update(
                {
                    "optimized_accepted": False,
                    "optimized_fallback": "brake",
                    "optimized_reject_reason": "committed_aborted_due_to_safety",
                    "fallback_reason": "committed_aborted_due_to_safety",
                    "rejection_cause": "unsafe",
                    "committed_abort_step": int(index),
                    "committed_abort_mode": mode,
                    "committed_abort_index": int(index),
                    "committed_abort_chunk_length": int(length),
                    "committed_abort_min_clearance": float(
                        info.get("min_clearance", float("-inf"))
                    ),
                    "committed_abort_required_clearance": self._committed_abort_threshold(),
                    "committed_abort_clearance_gap": float(
                        self._committed_abort_threshold()
                        - float(info.get("min_clearance", float("-inf")))
                    ),
                    "committed_abort_reason": abort_reason,
                    "committed_abort_due_to_human_motion": bool(
                        abort_due_to_human_motion
                    ),
                    "committed_abort_due_to_prediction_error": bool(
                        abort_due_to_prediction_error
                    ),
                    "committed_abort_due_to_safety_semantics_mismatch": bool(
                        abort_due_to_semantics_mismatch
                    ),
                }
            )
        return info

    def _try_resume_act_from_committed_recovery(
        self,
        obs: Any,
        nominal_chunk: np.ndarray,
        original_shape: Any,
        mode: str,
        idx: int,
        total: int,
        state_info: InfoDict,
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Resume nominal ACT execution early when committed recovery rejoins."""
        if not (self.opportunistic_act_resume and mode == "recover"):
            return None
        context = self.recovery_context
        if context is None or context.nominal_q_seq is None:
            state_info["committed_opportunistic_resume_available"] = False
            state_info["committed_opportunistic_resume_reason"] = "missing_recovery_context"
            return None
        nominal, _ = self._as_chunk(nominal_chunk)
        if nominal.shape[0] == 0:
            state_info["committed_opportunistic_resume_available"] = False
            state_info["committed_opportunistic_resume_reason"] = "empty_nominal_chunk"
            return None

        current_q = self._current_replay_q(obs, **kwargs)
        rejoin_context = self._make_rejoin_context(
            context.nominal_q_seq,
            context.nominal_ee_seq,
        )
        q_loss, target_index, q_time_ms = self._q_rejoin_loss(
            current_q.reshape(1, -1),
            nominal_q_seq=context.nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        q_dist = self._sqrt_loss(q_loss)
        q_threshold = (
            float(self.q_rejoin_threshold)
            if self.opportunistic_resume_q_threshold is None
            else float(self.opportunistic_resume_q_threshold)
        )
        close_enough = bool(target_index is not None and q_dist < q_threshold)

        nominal_action = np.asarray(nominal[0], dtype=np.float32).copy()
        _nominal_safe, action_safety = self._committed_action_safety(
            obs,
            current_q,
            nominal_action,
            **kwargs,
        )
        del _nominal_safe
        nominal_min_clearance = float(
            action_safety.get("min_clearance", float("-inf"))
        )
        nominal_step_safe = bool(
            nominal_min_clearance >= float(self.opportunistic_resume_min_clearance)
        )
        resume_score_input = {
            **state_info,
            **action_safety,
            "q_rejoin_dist": float(q_dist),
            "q_rejoin_threshold": float(q_threshold),
            "resume_tube_prefix_min_clearance": float(nominal_min_clearance),
            "resume_tube_prefix_safe": bool(nominal_step_safe),
            "handoff_current_clearance": float(nominal_min_clearance),
        }
        resume_readiness_terms = self._resume_readiness_terms(
            resume_score_input,
            q=current_q,
            obs=obs,
            source="committed_opportunistic_current",
        )
        resume_tube_terms = resume_readiness_terms
        resume_affordance_terms = resume_readiness_terms
        resume_tube_ok = bool(resume_readiness_terms.get("resume_tube_ok", True))
        resume_affordance_ok = bool(
            resume_readiness_terms.get("resume_affordance_ok", True)
        )
        resume_allowed = bool(
            resume_readiness_terms.get(
                "resume_allowed",
                resume_tube_ok and resume_affordance_ok,
            )
        )
        state_info.update(
            {
                "committed_opportunistic_resume_available": bool(
                    close_enough
                    and nominal_step_safe
                    and resume_allowed
                ),
                "committed_opportunistic_resume_close_enough": bool(close_enough),
                "committed_opportunistic_resume_nominal_step_safe": bool(
                    nominal_step_safe
                ),
                "committed_opportunistic_resume_tube_ok": bool(resume_tube_ok),
                "committed_opportunistic_resume_tube_score": float(
                    resume_tube_terms["resume_tube_score"]
                ),
                "committed_opportunistic_resume_tube_component_score": float(
                    resume_tube_terms["resume_tube_component_score"]
                ),
                "committed_opportunistic_resume_allowed": bool(resume_allowed),
                "committed_opportunistic_resume_block_reason": resume_readiness_terms.get(
                    "resume_block_reason"
                ),
                "committed_opportunistic_resume_affordance_ok": bool(
                    resume_affordance_ok
                ),
                "committed_opportunistic_resume_affordance_score": float(
                    resume_affordance_terms["resume_affordance_score"]
                ),
                "committed_opportunistic_resume_affordance_component_score": float(
                    resume_affordance_terms[
                        "resume_affordance_component_score"
                    ]
                ),
                "committed_opportunistic_resume_q_dist": float(q_dist),
                "committed_opportunistic_resume_q_threshold": float(q_threshold),
                "committed_opportunistic_resume_min_clearance": float(
                    nominal_min_clearance
                ),
                "committed_opportunistic_resume_required_clearance": float(
                    self.opportunistic_resume_min_clearance
                ),
                "committed_opportunistic_resume_rejoin_index": (
                    None if target_index is None else int(target_index)
                ),
                "committed_opportunistic_resume_q_eval_time_ms": float(q_time_ms),
            }
        )
        if not (
            close_enough
            and nominal_step_safe
            and resume_allowed
        ):
            if not close_enough:
                resume_reason = "q_rejoin_not_close"
            elif not nominal_step_safe:
                resume_reason = "nominal_step_unsafe"
            elif not resume_tube_ok:
                resume_reason = "resume_tube_not_ready"
            elif not resume_affordance_ok:
                resume_reason = "resume_affordance_not_ready"
            else:
                resume_reason = "resume_not_ready"
            state_info["committed_opportunistic_resume_reason"] = resume_reason
            return None

        self.committed_opportunistic_resume_count += 1
        self.committed_recover_steps_since_act = 0
        self.committed_suffix_replans_in_current_recovery = 0
        info = dict(action_safety)
        info.update(
            {
                "safety_mode": "pass_through",
                "mode": "pass_through",
                "deform_mode": "act_resume_from_recovery",
                "deformation_source": "act_resume_from_recovery",
                "recovery_mode": "resume_act",
                "recovery_phase": "resume_act",
                "committed_chunk_active": True,
                "committed_chunk_mode": mode,
                "committed_chunk_index": int(idx),
                "committed_chunk_length": int(total),
                "committed_chunk_completed": False,
                **self._act_release_terms(
                    reason="opportunistic_resume",
                    allowed=True,
                    block_reason=None,
                    reset_history=True,
                    resume_index=target_index,
                    activate_window=True,
                ),
                "committed_opportunistic_resume": True,
                "committed_recovery_budget_exit": False,
                "recover_steps_executed": 0,
                "return_steps_executed": 0,
                "deform_steps_executed": 0,
                "deform_steps_executed": 0,
                "fallback_used": False,
                "optimized_accepted": True,
                "deform_safe": True,
                "is_safe": True,
                "is_recoverable": True,
                "q_rejoin_loss": float(q_loss),
                "q_rejoin_dist": float(q_dist),
                "q_rejoin_threshold": float(q_threshold),
                "q_rejoin_index": None if target_index is None else int(target_index),
                "rejoin_q_eval_time_ms": float(q_time_ms),
                **state_info,
            }
        )
        if self.committed_rejoin_diagnostics:
            for key, value in self.committed_rejoin_diagnostics.items():
                info.setdefault(key, value)
        self._clear_committed_chunk()
        self.last_info = info
        return nominal.reshape(original_shape), info

    def _serve_committed_chunk(
        self,
        obs: Any,
        nominal_chunk: np.ndarray,
        original_shape: Any,
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Serve the next action from a cached committed recovery chunk."""
        if self.committed_chunk is None:
            return None
        committed = np.asarray(self.committed_chunk, dtype=np.float32)
        total = int(committed.shape[0])
        idx = int(self.committed_chunk_index)
        if idx >= total:
            self._clear_committed_chunk()
            return None

        mode = (
            self.committed_chunk_modes[idx]
            if idx < len(self.committed_chunk_modes)
            else self.committed_chunk_mode
        )
        self.committed_chunk_mode = mode
        action_idx = self.controlled_action_indices[
            self.controlled_action_indices < nominal_chunk.shape[1]
        ]
        actual_action = np.asarray(nominal_chunk[0], dtype=np.float32).copy()
        if action_idx.size:
            actual_action[action_idx] = committed[idx, action_idx]

        state_diagnostics = self._committed_state_diagnostics(idx, obs, **kwargs)
        actual_pre_q = np.asarray(
            state_diagnostics.get("actual_q_at_replay", []),
            dtype=np.float32,
        ).reshape(-1)
        if mode == "recover" and self.committed_nominal_tube_tracking_enabled:
            actual_action, tracking_info = self._committed_nominal_tube_tracking_action(
                actual_action,
                actual_pre_q,
                idx,
                action_idx=action_idx,
            )
            state_diagnostics.update(tracking_info)
        state_diagnostics["planned_action_at_index"] = self._jsonable_snapshot(
            self._committed_planned_value(self.committed_planned_actions, idx)
        )
        state_diagnostics["committed_abort_action"] = self._jsonable_snapshot(
            actual_action
        )
        budget_exceeded = bool(
            mode == "recover"
            and self.max_recover_steps_before_act_resume > 0
            and self.committed_recover_steps_since_act
            >= self.max_recover_steps_before_act_resume
        )
        mpc_bridge_context_id = 0 if self.recovery_context is None else id(self.recovery_context)
        if self.mpc_bridge_context_id != mpc_bridge_context_id:
            self.mpc_bridge_context_id = mpc_bridge_context_id
            self.mpc_bridge_replans_in_current_recovery = 0
            self.mpc_bridge_replan_cooldown_remaining = 0
            self.mpc_bridge_last_heading_cosine = None
            self.mpc_bridge_last_progress_projection = None
            self.mpc_bridge_last_prefix_clearance = None
            self.mpc_bridge_last_improved = True
        mpc_bridge_cooldown_active = bool(
            self.mpc_bridge_replan_cooldown_remaining > 0
        )
        if mpc_bridge_cooldown_active:
            self.mpc_bridge_replan_cooldown_remaining = max(
                0,
                int(self.mpc_bridge_replan_cooldown_remaining) - 1,
            )
        state_diagnostics.update(
            {
                "mpc_bridge_replans_in_current_recovery": int(
                    self.mpc_bridge_replans_in_current_recovery
                ),
                "mpc_bridge_max_replans_per_recovery": int(
                    self.mpc_bridge_max_replans_per_recovery
                ),
                "mpc_bridge_replan_cooldown_remaining": int(
                    self.mpc_bridge_replan_cooldown_remaining
                ),
                "mpc_bridge_replan_cooldown_active": bool(
                    mpc_bridge_cooldown_active
                ),
            }
        )
        # Prefer the committed recovery rejoin path for ACT restart.  This path
        # releases only after the live state is close enough to the recovery
        # rejoin target, then asks the eval loop to reset ACT action/visual
        # histories.  Direct MPC handoff remains a fallback repair diagnostic,
        # not the primary gate for resuming ACT.
        # Do not directly resume raw ACT from inside the committed-recovery path.
        # Release only after the current recovery/MPC action is safe, then let the
        # next ACT chunk re-enter the main safechunk brake->deform->recover loop.
        resume_result = None
        state_diagnostics["committed_opportunistic_resume_available"] = False
        state_diagnostics["committed_opportunistic_resume_reason"] = (
            "deferred_to_main_safechunk_filter"
        )
        fsm_handoff_min_commit_ok = True
        try:
            fsm = getattr(self.parent, "intervention_fsm", None)
            if (
                fsm is not None
                and getattr(fsm.config, "enabled", False)
                and str(getattr(fsm, "mode", ""))
                == "InterventionMode.DEFORM_COMMIT"
            ):
                fsm_handoff_min_commit_ok = bool(fsm.can_handoff_from_deform())
        except Exception:  # noqa: BLE001
            fsm_handoff_min_commit_ok = True
        state_diagnostics["intervention_fsm_handoff_min_commit_ok"] = bool(
            fsm_handoff_min_commit_ok
        )
        state_diagnostics["intervention_fsm_handoff_block_reason"] = (
            None if fsm_handoff_min_commit_ok else "deform_commit_min_steps"
        )
        if mode == "recover" and fsm_handoff_min_commit_ok:
            resume_result = self.mpc.try_handoff_to_act(
                obs,
                nominal_chunk,
                original_shape,
                mode,
                idx,
                total,
                state_diagnostics,
                handoff_reason="pre_committed_step",
                **kwargs,
            )
        if resume_result is not None:
            resumed_action, resumed_info = resume_result
            release_info = dict(resumed_info or {})
            release_to_act = bool(
                release_info.get("mpc_handoff_bridge_ramp_release_to_act", True)
            )
            release_info.update(
                self._act_release_terms(
                    reason="mpc_handoff" if release_to_act else "mpc_handoff_bridge_ramp",
                    allowed=release_to_act,
                    block_reason=None
                    if release_to_act
                    else str(
                        release_info.get(
                            "mpc_handoff_deferred_release_reason",
                            "resume_tube_not_ready",
                        )
                    ),
                    reset_history=bool(
                        release_to_act
                        and release_info.get(
                            "request_action_history_reset_after_recovery",
                            True,
                        )
                    ),
                    resume_index=release_info.get("act_resume_index") if release_to_act else None,
                    activate_window=False,
                )
            )
            if not release_to_act:
                release_info["committed_released_for_act_resume"] = False
                release_info["request_action_history_reset_after_recovery"] = False
                release_info["act_resume_index"] = None
            self._fill_recover_affordance_metrics(release_info)
            return resumed_action, release_info
        if (
            mode == "recover"
            and bool(self.mpc_replan_every_recovery_step)
            and self.mpc_recovery_enabled
            and idx > 0
            and idx < total - 1
        ):
            mpc_result = self.mpc.try_replan_committed_recovery(
                obs,
                nominal_chunk,
                original_shape,
                mode,
                idx,
                total,
                state_diagnostics,
                replan_reason="receding_horizon",
                **kwargs,
            )
            if mpc_result is not None:
                return mpc_result
        if budget_exceeded:
            mpc_result = self.mpc.try_replan_committed_recovery(
                obs,
                nominal_chunk,
                original_shape,
                mode,
                idx,
                total,
                state_diagnostics,
                replan_reason="recovery_budget_exit",
                **kwargs,
            )
            if mpc_result is not None:
                return mpc_result
            self.committed_recovery_budget_exit_count += 1
            budget_info = dict(state_diagnostics)
            budget_info.update(
                {
                    "committed_chunk_active": True,
                    "committed_chunk_mode": mode,
                    "committed_chunk_index": int(idx),
                    "committed_chunk_length": int(total),
                    "committed_recovery_budget_exit": True,
                    **self._act_release_terms(
                        reason="emergency_budget_exit",
                        allowed=False,
                        block_reason="recovery_budget_exit",
                        reset_history=False,
                        resume_index=None,
                        activate_window=False,
                        degraded=True,
                    ),
                    "committed_recover_steps_since_act": int(
                        self.committed_recover_steps_since_act
                    ),
                    "max_recover_steps_before_act_resume": int(
                        self.max_recover_steps_before_act_resume
                    ),
                    "committed_replan_due_to_recovery_budget": True,
                    "fallback_used": False,
                    "optimized_accepted": False,
                    "optimized_reject_reason": "recovery_budget_exit",
                    "fallback_reason": "recovery_budget_exit",
                }
            )
            self.recovery_optimizer_cooldown_remaining = max(
                int(self.recovery_optimizer_cooldown_remaining),
                int(self.recover_retry_cooldown_steps),
            )
            if self.recover_max_attempts_per_unsafe_streak > 0:
                self.recovery_attempts_in_unsafe_streak = max(
                    int(self.recovery_attempts_in_unsafe_streak),
                    int(self.recover_max_attempts_per_unsafe_streak),
                )
            self._clear_committed_chunk()
            self._pending_committed_replan_info = budget_info
            return None
        handoff_reject_reason = state_diagnostics.get("mpc_handoff_reject_reason")
        handoff_bridge_reason = str(handoff_reject_reason)
        current_heading = state_diagnostics.get("mpc_handoff_heading_cosine")
        current_progress = state_diagnostics.get("mpc_handoff_progress_projection")
        current_clearance = state_diagnostics.get("mpc_handoff_act_prefix_min_clearance")
        if current_clearance is None:
            current_clearance = state_diagnostics.get("mpc_handoff_live_prefix_best_min_clearance")
        if current_clearance is None:
            current_clearance = state_diagnostics.get("nominal_rejoin_clearance")

        def _finite_float_or_none(value):
            try:
                value_f = float(value)
            except Exception:  # noqa: BLE001
                return None
            return value_f if np.isfinite(value_f) else None

        current_heading_f = _finite_float_or_none(current_heading)
        current_progress_f = _finite_float_or_none(current_progress)
        current_clearance_f = _finite_float_or_none(current_clearance)

        def _contact_count_clear() -> bool:
            contact_count = kwargs.get("robot_human_contact_count")
            if contact_count is None:
                contact_count = kwargs.get("current_robot_human_contact_count")
            if contact_count is None:
                contact_count = kwargs.get("contact_count")
            try:
                return bool(contact_count is None or int(contact_count) <= 0)
            except (TypeError, ValueError):
                return False

        live_clearance = _finite_float_or_none(kwargs.get("live_monitor_min_clearance"))
        live_prefix_safe = bool(
            state_diagnostics.get("mpc_handoff_live_prefix_safe", False)
            or state_diagnostics.get("nominal_rejoin_live_prefix_safe", False)
        )
        live_safe = bool(
            not bool(kwargs.get("live_monitor_h_violation", False))
            and _contact_count_clear()
            and (
                live_prefix_safe
                or live_clearance is None
                or live_clearance >= 0.02
            )
        )
        prefix_release_clearance = max(
            0.0,
            min(
                float(getattr(self, "opportunistic_resume_min_clearance", 0.02)),
                float(getattr(self, "committed_min_clearance_for_abort", 0.02)),
            ),
        )
        prefix_safe = bool(
            current_clearance_f is not None
            and current_clearance_f >= prefix_release_clearance
        )
        resume_tube_input = dict(state_diagnostics)
        resume_tube_input.update(
            {
                "handoff_heading_cosine": current_heading_f,
                "handoff_progress_projection": current_progress_f,
                "handoff_current_clearance": current_clearance_f,
                "live_monitor_min_clearance": live_clearance,
                "live_prefix_safe": bool(live_prefix_safe),
                "prefix_safe": bool(prefix_safe),
                "q_rejoin_threshold": float(self.q_rejoin_threshold),
            }
        )
        actual_q_for_affordance = state_diagnostics.get("actual_q_at_replay")
        resume_readiness_terms = self._resume_readiness_terms(
            resume_tube_input,
            q=actual_q_for_affordance,
            obs=obs,
            source="mpc_handoff_actual",
        )
        resume_tube_terms = resume_readiness_terms
        resume_affordance_terms = resume_readiness_terms
        state_diagnostics.update(
            {
                f"mpc_handoff_{key}": value
                for key, value in resume_tube_terms.items()
            }
        )
        state_diagnostics.update(
            {
                f"mpc_handoff_{key}": value
                for key, value in resume_affordance_terms.items()
            }
        )
        resume_tube_ok = bool(resume_tube_terms["resume_tube_ok"])
        resume_affordance_ok = bool(
            resume_affordance_terms.get("resume_affordance_ok", True)
        )
        resume_allowed = bool(
            resume_tube_terms.get(
                "resume_allowed",
                resume_tube_ok and resume_affordance_ok,
            )
        )
        soft_handoff_reasons = {
            "actual_heading_not_aligned",
            "actual_progress_not_forward",
            "no_safe_nominal_window",
            "no_live_safe_nominal_window",
            "act_progress_failed",
        }
        soft_handoff_prefix_ok = bool(
            (prefix_safe or live_prefix_safe)
            and resume_allowed
        )
        if (
            mode == "recover"
            and fsm_handoff_min_commit_ok
            and handoff_reject_reason in soft_handoff_reasons
            and soft_handoff_prefix_ok
            and live_safe
        ):
            self.committed_opportunistic_resume_count += 1
            self.committed_recover_steps_since_act = 0
            self.committed_suffix_replans_in_current_recovery = 0
            release_info = dict(state_diagnostics)
            release_info.update(
                {
                    "committed_chunk_active": True,
                    "committed_chunk_mode": mode,
                    "committed_chunk_index": int(idx),
                    "committed_chunk_length": int(total),
                    **self._act_release_terms(
                        reason="committed_soft_handoff",
                        allowed=True,
                        block_reason=None,
                        reset_history=True,
                        resume_index=None,
                        activate_window=False,
                    ),
                    "committed_soft_handoff_release_reason": str(handoff_reject_reason),
                    "committed_soft_handoff_prefix_min_clearance": current_clearance_f,
                    "committed_soft_handoff_required_clearance": float(prefix_release_clearance),
                    "committed_soft_handoff_live_min_clearance": live_clearance,
                    "committed_soft_handoff_live_prefix_safe": bool(live_prefix_safe),
                    "committed_soft_handoff_prefix_safe": bool(prefix_safe),
                    "committed_soft_handoff_prefix_ok": bool(soft_handoff_prefix_ok),
                    "committed_soft_handoff_resume_tube_score": float(
                        resume_tube_terms["resume_tube_score"]
                    ),
                    "committed_soft_handoff_resume_tube_ok": bool(resume_tube_ok),
                    "committed_soft_handoff_resume_tube_component_score": float(
                        resume_tube_terms["resume_tube_component_score"]
                    ),
                    "committed_soft_handoff_resume_affordance_score": float(
                        resume_affordance_terms["resume_affordance_score"]
                    ),
                    "committed_soft_handoff_resume_affordance_ok": bool(
                        resume_affordance_ok
                    ),
                    "committed_soft_handoff_resume_affordance_component_score": float(
                        resume_affordance_terms["resume_affordance_component_score"]
                    ),
                    "fallback_used": False,
                    "optimized_accepted": True,
                    "optimized_reject_reason": None,
                    "fallback_reason": None,
                }
            )
            self._clear_committed_chunk()
            self._pending_committed_replan_info = release_info
            return None

        has_bridge_history = bool(
            self.mpc_bridge_last_heading_cosine is not None
            or self.mpc_bridge_last_progress_projection is not None
            or self.mpc_bridge_last_prefix_clearance is not None
        )
        heading_improved = bool(
            current_heading_f is not None
            and self.mpc_bridge_last_heading_cosine is not None
            and current_heading_f
            >= float(self.mpc_bridge_last_heading_cosine)
            + float(self.mpc_bridge_min_heading_improvement)
        )
        progress_improved = bool(
            current_progress_f is not None
            and self.mpc_bridge_last_progress_projection is not None
            and current_progress_f
            >= float(self.mpc_bridge_last_progress_projection)
            + float(self.mpc_bridge_min_progress_improvement)
        )
        clearance_improved = bool(
            current_clearance_f is not None
            and self.mpc_bridge_last_prefix_clearance is not None
            and current_clearance_f
            >= float(self.mpc_bridge_last_prefix_clearance)
            + float(self.mpc_bridge_min_clearance_improvement)
        )
        bridge_improved = bool(
            not has_bridge_history
            or heading_improved
            or progress_improved
            or clearance_improved
        )
        bridge_under_cap = bool(
            self.mpc_bridge_max_replans_per_recovery <= 0
            or self.mpc_bridge_replans_in_current_recovery
            < self.mpc_bridge_max_replans_per_recovery
        )
        state_diagnostics.update(
            {
                "mpc_bridge_replan_heading_improved": bool(heading_improved),
                "mpc_bridge_replan_progress_improved": bool(progress_improved),
                "mpc_bridge_replan_clearance_improved": bool(clearance_improved),
                "mpc_bridge_replan_metric_improved": bool(bridge_improved),
                "mpc_bridge_replan_under_cap": bool(bridge_under_cap),
                "mpc_bridge_last_heading_cosine": self.mpc_bridge_last_heading_cosine,
                "mpc_bridge_last_progress_projection": self.mpc_bridge_last_progress_projection,
                "mpc_bridge_last_prefix_clearance": self.mpc_bridge_last_prefix_clearance,
            }
        )
        handoff_needs_bridge_replan = bool(
            mode == "recover"
            and self.mpc_recovery_enabled
            and handoff_reject_reason in {
                "act_prefix_unsafe",
                "no_live_safe_nominal_window",
                "actual_heading_not_aligned",
                "actual_progress_not_forward",
            }
            and int(self.committed_recover_steps_since_act)
            >= int(self.mpc_recovery_prefix_len)
            and not mpc_bridge_cooldown_active
            and bridge_under_cap
            and bridge_improved
        )
        if handoff_reject_reason in {
            "act_prefix_unsafe",
            "no_live_safe_nominal_window",
            "actual_heading_not_aligned",
            "actual_progress_not_forward",
        }:
            if mpc_bridge_cooldown_active:
                handoff_bridge_reason = "bridge_cooldown"
            elif not bridge_under_cap:
                handoff_bridge_reason = "bridge_replan_cap"
            elif not bridge_improved:
                handoff_bridge_reason = "bridge_not_improving"
            state_diagnostics["mpc_bridge_replan_suppressed_reason"] = handoff_bridge_reason
        if handoff_needs_bridge_replan:
            self.mpc_bridge_replans_in_current_recovery += 1
            if current_heading_f is not None:
                self.mpc_bridge_last_heading_cosine = current_heading_f
            if current_progress_f is not None:
                self.mpc_bridge_last_progress_projection = current_progress_f
            if current_clearance_f is not None:
                self.mpc_bridge_last_prefix_clearance = current_clearance_f
            self.mpc_bridge_last_improved = bool(bridge_improved)
            state_diagnostics.update(
                {
                    "mpc_bridge_replan_suppressed_reason": None,
                    "mpc_bridge_replans_in_current_recovery": int(
                        self.mpc_bridge_replans_in_current_recovery
                    ),
                }
            )
            mpc_result = self.mpc.try_replan_committed_recovery(
                obs,
                nominal_chunk,
                original_shape,
                mode,
                idx,
                total,
                state_diagnostics,
                replan_reason=f"handoff_{handoff_reject_reason}",
                **kwargs,
            )
            if mpc_result is not None:
                self.mpc_bridge_replan_cooldown_remaining = int(
                    self.mpc_bridge_replan_cooldown_steps
                )
                return mpc_result
        state_error = state_diagnostics.get("committed_state_error")
        state_mismatch = bool(
            state_diagnostics.get("committed_rejected_missing_planned_q")
        )
        if state_error is not None:
            state_mismatch = (
                state_mismatch
                or float(state_error) > self.committed_state_error_threshold
            )
        if (
            state_mismatch
            and self.committed_state_mismatch_abort_requires_unsafe
            and self.mpc_recovery_enabled
            and mode == "recover"
        ):
            mpc_result = self.mpc.try_replan_committed_recovery(
                obs,
                nominal_chunk,
                original_shape,
                mode,
                idx,
                total,
                state_diagnostics,
                replan_reason="state_mismatch",
                **kwargs,
            )
            if mpc_result is not None:
                return mpc_result
        if (
            state_mismatch
            and not self.committed_state_mismatch_abort_requires_unsafe
        ):
            suffix_replan_result = self._try_replan_committed_suffix_from_current_state(
                obs,
                nominal_chunk,
                original_shape,
                mode,
                idx,
                total,
                state_diagnostics,
                **kwargs,
            )
            if suffix_replan_result is not None:
                return suffix_replan_result
            mismatch_info = self._committed_state_mismatch_info(
                mode,
                idx,
                total,
                state_diagnostics,
            )
            if self.committed_state_error_action == "replan":
                self._clear_committed_chunk()
                self._pending_committed_replan_info = mismatch_info
                return None
            result = self._committed_state_mismatch_brake(
                obs,
                nominal_chunk,
                original_shape,
                mode,
                idx,
                total,
                state_diagnostics,
            )
            self._clear_committed_chunk()
            return result

        repaired = False
        repair_info = None
        if self.committed_chunk_safety_check:
            _action_safe, action_safety = self._committed_action_safety(
                obs,
                actual_pre_q,
                actual_action,
                **kwargs,
            )
            del _action_safe
            min_clearance = float(action_safety.get("min_clearance", float("-inf")))
            diagnostics = self._committed_replay_diagnostics(
                idx,
                actual_action,
                action_safety,
                **kwargs,
            )
            diagnostics.update(state_diagnostics)
            diagnostics["committed_state_mismatch_detected"] = bool(state_mismatch)
            diagnostics["committed_state_mismatch_abort_requires_unsafe"] = bool(
                self.committed_state_mismatch_abort_requires_unsafe
            )
            abort_threshold = self._committed_abort_threshold()
            execution_min_clearance = diagnostics.get("committed_execution_min_clearance")
            try:
                execution_min_clearance = float(execution_min_clearance)
            except Exception:  # noqa: BLE001
                execution_min_clearance = min_clearance
            if not np.isfinite(execution_min_clearance):
                execution_min_clearance = min_clearance
            diagnostics["committed_abort_required_clearance"] = float(abort_threshold)
            abort_clearance_gap = float(abort_threshold - execution_min_clearance)
            abort_due_to_low_clearance = bool(execution_min_clearance < abort_threshold)
            if abort_due_to_low_clearance and state_mismatch:
                suffix_replan_result = self._try_replan_committed_suffix_from_current_state(
                    obs,
                    nominal_chunk,
                    original_shape,
                    mode,
                    idx,
                    total,
                    state_diagnostics,
                    **kwargs,
                )
                if suffix_replan_result is not None:
                    return suffix_replan_result
                diagnostics["committed_pre_abort_suffix_replan_attempted"] = True
                diagnostics["committed_pre_abort_suffix_replan_accepted"] = False

            marginal_positive_clearance = bool(
                state_mismatch
                and execution_min_clearance >= 0.0
                and abort_clearance_gap <= max(float(self.committed_safety_tol), 1e-6)
            )
            diagnostics["committed_abort_clearance_gap"] = float(abort_clearance_gap)
            diagnostics["committed_abort_marginal_positive_clearance"] = bool(
                marginal_positive_clearance
            )
            if abort_due_to_low_clearance and marginal_positive_clearance:
                # The committed rollout can be slightly pessimistic/noisy near the
                # abort boundary.  Do not throw away an accepted recovery because
                # live clearance is still positive and only misses the emergency
                # abort threshold by a small tolerance; keep executing and let the
                # next committed tick re-check safety from the live state.
                diagnostics["committed_abort_deferred_for_marginal_clearance"] = True
                abort_due_to_low_clearance = False

            if abort_due_to_low_clearance:
                diagnostics["committed_abort_reason"] = (
                    "state_mismatch_and_contact_risk"
                    if state_mismatch
                    else (
                        "current_clearance_below_abort_threshold"
                        if execution_min_clearance < min_clearance
                        else self._committed_abort_reason(diagnostics, min_clearance)
                    )
                )
                diagnostics["committed_abort_action"] = self._jsonable_snapshot(actual_action)
                diagnostics["committed_abort_min_clearance"] = float(execution_min_clearance)
                abort_safety = dict(action_safety)
                abort_safety["min_clearance"] = float(execution_min_clearance)
                abort_info = self._committed_info(
                    abort_safety,
                    mode,
                    idx,
                    total,
                    completed=False,
                    aborted=True,
                    extra=diagnostics,
                )
                self._clear_committed_chunk()
                brake_safety = dict(action_safety)
                brake_safety["first_violation"] = 0
                brake_safety["min_clearance"] = float(execution_min_clearance)
                braked_chunk, brake_info = self.brake.horizon_brake(
                    obs,
                    nominal_chunk,
                    brake_safety,
                )
                abort_info.update(brake_info)
                self.last_info = abort_info
                return braked_chunk.reshape(original_shape), abort_info

            if (
                self.repair_committed_action
                and min_clearance < self.min_clearance
                and min_clearance >= abort_threshold
            ):
                before_repair_clearance = min_clearance
                repaired_action = self._repair_committed_action(
                    obs,
                    actual_action,
                    **kwargs,
                )
                if repaired_action.shape[0] == actual_action.shape[0]:
                    candidate_action = np.asarray(actual_action, dtype=np.float32).copy()
                    if action_idx.size:
                        candidate_action[action_idx] = repaired_action[action_idx]
                    else:
                        candidate_action = repaired_action
                    _repair_safe, repaired_safety = self._committed_action_safety(
                        obs,
                        actual_pre_q,
                        candidate_action,
                        **kwargs,
                    )
                    del _repair_safe
                    repaired_min = float(
                        repaired_safety.get("min_clearance", float("-inf"))
                    )
                    repair_gain = float(repaired_min - before_repair_clearance)
                    repair_info = {
                        "committed_repair_min_clearance": repaired_min,
                        "committed_repair_clearance_gain": repair_gain,
                        "committed_repair_rejected": bool(
                            self.monotonic_committed_repair and repair_gain < 0.0
                        ),
                    }
                    diagnostics.update(repair_info)
                    if repair_gain >= 0.0 or not self.monotonic_committed_repair:
                        actual_action = candidate_action
                        repaired = True
                        action_safety = repaired_safety
                        min_clearance = repaired_min
                        diagnostics.update(
                            self._committed_replay_diagnostics(
                                idx,
                                actual_action,
                                action_safety,
                                **kwargs,
                            )
                        )
                        diagnostics.update(state_diagnostics)
                        diagnostics["committed_state_mismatch_detected"] = bool(
                            state_mismatch
                        )
                        diagnostics["committed_state_mismatch_abort_requires_unsafe"] = bool(
                            self.committed_state_mismatch_abort_requires_unsafe
                        )
                        diagnostics.update(repair_info)
                        if repaired_min < abort_threshold:
                            reason = self._committed_abort_reason(
                                diagnostics,
                                repaired_min,
                            )
                            if reason == "clearance_below_abort_threshold":
                                reason = "repair_failed_contact_risk"
                            diagnostics["committed_abort_reason"] = reason
                            diagnostics["committed_abort_action"] = self._jsonable_snapshot(actual_action)
                            abort_info = self._committed_info(
                                action_safety,
                                mode,
                                idx,
                                total,
                                completed=False,
                                aborted=True,
                                repaired=True,
                                repair_info=repair_info,
                                extra=diagnostics,
                            )
                            self._clear_committed_chunk()
                            brake_safety = dict(action_safety)
                            brake_safety["first_violation"] = 0
                            braked_chunk, brake_info = self.brake.horizon_brake(
                                obs,
                                nominal_chunk,
                                brake_safety,
                            )
                            abort_info.update(brake_info)
                            self.last_info = abort_info
                            return braked_chunk.reshape(original_shape), abort_info

        else:
            action_safety = {
                "horizon_safe": True,
                "min_clearance": float("inf"),
                "safety_eval_available": False,
            }
            diagnostics = self._committed_replay_diagnostics(
                idx,
                actual_action,
                action_safety,
                **kwargs,
            )
            diagnostics.update(state_diagnostics)

        rebase_info = self._rebase_committed_plan_from_actual_step(
            obs,
            idx,
            actual_pre_q,
            actual_action,
            action_safety,
        )
        diagnostics.update(rebase_info)
        committed = np.asarray(self.committed_chunk, dtype=np.float32)

        served = np.asarray(nominal_chunk, dtype=np.float32).copy()
        n = min(served.shape[0], total - idx)
        if action_idx.size:
            served[:n, action_idx] = committed[idx : idx + n, action_idx]
            served[0, action_idx] = actual_action[action_idx]
        self.committed_chunk_index = idx + 1
        if mode == "recover":
            self.committed_recover_steps_since_act += 1
        completed = self.committed_chunk_index >= total
        info = self._committed_info(
            action_safety,
            mode,
            idx,
            total,
            completed=completed,
            aborted=False,
            repaired=repaired,
            repair_info=repair_info,
            extra=diagnostics,
        )
        info.update(
            {
                "committed_recover_steps_since_act": int(
                    self.committed_recover_steps_since_act
                ),
                "max_recover_steps_before_act_resume": int(
                    self.max_recover_steps_before_act_resume
                ),
                "committed_suffix_replans_in_current_recovery": int(
                    self.committed_suffix_replans_in_current_recovery
                ),
                "max_suffix_replans_per_recovery": int(
                    self.max_suffix_replans_per_recovery
                ),
            }
        )
        receding_recover_triggered = False
        receding_recover_steps = 0
        receding_reason = "recover_prefix_executed"
        tracking_applied_for_receding = bool(
            info.get("committed_nominal_tube_tracking_applied", False)
        )
        tracking_ready_for_receding = bool(
            info.get("committed_nominal_tube_tracking_ready", False)
        )
        tracking_failed_for_receding = bool(
            info.get("committed_nominal_tube_tracking_failed", False)
        )
        tracking_step_cap = int(
            getattr(self, "committed_nominal_tube_tracking_max_recover_steps", 0)
        )
        tracking_step_cap_reached = False
        if (
            not completed
            and mode == "recover"
            and int(getattr(self, "committed_receding_recover_steps", 0)) > 0
        ):
            try:
                receding_recover_steps = int(
                    sum(
                        1
                        for _mode in self.committed_chunk_modes[: self.committed_chunk_index]
                        if _mode == "recover"
                    )
                )
            except Exception:  # noqa: BLE001
                receding_recover_steps = int(self.committed_recover_steps_since_act)
            tracking_step_cap_reached = bool(
                tracking_step_cap > 0 and receding_recover_steps >= tracking_step_cap
            )
            receding_due = bool(
                receding_recover_steps >= int(self.committed_receding_recover_steps)
                or tracking_failed_for_receding
            )
            if receding_due:
                if tracking_failed_for_receding:
                    receding_recover_triggered = True
                    receding_reason = "tracking_actual_diverged"
                elif tracking_applied_for_receding and not tracking_ready_for_receding:
                    if tracking_step_cap_reached:
                        receding_recover_triggered = True
                        receding_reason = "recover_tracking_step_cap"
                    else:
                        info.update(
                            {
                                "committed_receding_horizon_deferred": True,
                                "committed_receding_horizon_defer_reason": "tracking_not_ready",
                                "committed_receding_horizon_prefix_steps": int(
                                    receding_recover_steps
                                ),
                                "committed_receding_horizon_remaining_steps": int(
                                    max(0, total - self.committed_chunk_index)
                                ),
                                "committed_receding_recover_steps": int(
                                    self.committed_receding_recover_steps
                                ),
                                "committed_nominal_tube_tracking_max_recover_steps": int(
                                    tracking_step_cap
                                ),
                            }
                        )
                else:
                    receding_recover_triggered = True
                    receding_reason = (
                        "recover_tracking_ready"
                        if tracking_applied_for_receding
                        else "recover_prefix_executed"
                    )
        if receding_recover_triggered:
            info.update(
                {
                    "committed_receding_horizon_replan": True,
                    "committed_receding_horizon_reason": receding_reason,
                    "committed_receding_horizon_prefix_steps": int(receding_recover_steps),
                    "committed_receding_horizon_remaining_steps": int(
                        max(0, total - self.committed_chunk_index)
                    ),
                    "committed_receding_recover_steps": int(
                        self.committed_receding_recover_steps
                    ),
                    "committed_receding_horizon_step_cap_reached": bool(
                        tracking_step_cap_reached
                    ),
                    "committed_nominal_tube_tracking_max_recover_steps": int(
                        tracking_step_cap
                    ),
                }
            )
            self.recovery_optimizer_cooldown_remaining = 0
            self._clear_committed_chunk()
        elif completed:
            self._clear_committed_chunk()
        self.last_info = info
        return served.reshape(original_shape), info

    def _record_ordered_recovery_terms(self, terms: Mapping[str, Any]) -> None:
        """Append ordered-path loss diagnostics when present."""
        if not terms or not bool(terms.get("recover_ordered_path_available")):
            return
        for key, history in (
            ("recover_ordered_pose_loss", self._recover_ordered_pose_loss_history),
            ("recover_ordered_delta_loss", self._recover_ordered_delta_loss_history),
            ("recover_ordered_heading_loss", self._recover_ordered_heading_loss_history),
            ("recover_ordered_loss", self._recover_ordered_loss_history),
        ):
            value = terms.get(key)
            if value is not None and np.isfinite(float(value)):
                history.append(float(value))

    def _recover_resume_window_terms(
        self,
        q_seq,
        target_q_seq,
        *,
        candidate=None,
        target_chunk=None,
    ) -> Dict[str, Any]:
        requested_len = max(1, int(getattr(self, "recover_resume_window_len", 4)))
        result: Dict[str, Any] = {
            "recover_resume_window_available": False,
            "recover_resume_window_len": 0,
            "recover_resume_window_requested_len": requested_len,
            "recover_resume_window_loss": 0.0,
            "recover_resume_window_total_loss": 0.0,
            "recover_resume_window_dist": None,
            "recover_resume_window_error_l2": None,
            "recover_resume_window_dq_loss": 0.0,
            "recover_resume_window_dq_dist": None,
            "recover_resume_window_action_loss": 0.0,
            "recover_resume_window_action_dist": None,
            "recover_resume_window_q_frame_l2": None,
            "recover_resume_window_q_frame_l2_mean": None,
            "recover_resume_window_q_frame_l2_max": None,
            "recover_resume_window_wrist_l2": None,
            "recover_resume_window_wrist_l2_mean": None,
            "recover_resume_window_wrist_l2_max": None,
            "recover_resume_window_left_wrist_abs": None,
            "recover_resume_window_left_wrist_abs_mean": None,
            "recover_resume_window_left_wrist_abs_max": None,
            "recover_resume_window_right_wrist_abs": None,
            "recover_resume_window_right_wrist_abs_mean": None,
            "recover_resume_window_right_wrist_abs_max": None,
            "recover_resume_window_recovery_step_l2": None,
            "recover_resume_window_target_step_l2": None,
            "recover_resume_window_step_l2_error": None,
            "recover_resume_window_step_l2_error_mean": None,
            "recover_resume_window_step_l2_error_max": None,
            "recover_resume_window_dq_error_l2": None,
            "recover_resume_window_dq_cosine": None,
            "recover_resume_window_dq_cosine_mean": None,
            "recover_resume_window_dq_cosine_min": None,
            "recover_resume_window_dq_norm_ratio": None,
            "recover_resume_window_dq_norm_ratio_mean": None,
            "recover_resume_window_dq_norm_ratio_min": None,
            "recover_resume_window_start_local_index": None,
            "recover_resume_window_end_local_index": None,
            "recover_resume_window_weight": float(getattr(self, "recover_resume_window_weight", 0.0)),
            "recover_resume_window_dq_weight": float(getattr(self, "recover_resume_window_dq_weight", 0.0)),
            "recover_resume_window_action_weight": float(getattr(self, "recover_resume_window_action_weight", 0.0)),
            "recover_resume_window_q": None,
            "recover_resume_window_target_q": None,
            "recover_resume_window_action": None,
            "recover_resume_window_target_action": None,
        }
        try:
            q_arr = np.asarray(q_seq, dtype=np.float64)
            target_arr = np.asarray(target_q_seq, dtype=np.float64)
        except Exception:
            return result
        if q_arr.ndim != 2 or target_arr.ndim != 2:
            return result
        if q_arr.shape[0] <= 0 or target_arr.shape[0] <= 0:
            return result
        q_dim = min(q_arr.shape[1], target_arr.shape[1])
        if q_dim <= 0:
            return result
        window_len = min(requested_len, q_arr.shape[0], target_arr.shape[0])
        if window_len <= 0:
            return result

        state_indices = np.arange(q_dim, dtype=np.int64)
        state_weights = np.ones((q_dim,), dtype=np.float64)
        try:
            idx_weights = self._mpc_state_indices_and_weights(q_dim)
            if isinstance(idx_weights, tuple) and len(idx_weights) == 2:
                idx, weights = idx_weights
                idx = np.asarray(idx, dtype=np.int64)
                weights = np.asarray(weights, dtype=np.float64)
                valid = (idx >= 0) & (idx < q_dim)
                if np.any(valid):
                    state_indices = idx[valid]
                    state_weights = weights[valid]
        except Exception:
            try:
                idx = np.asarray(getattr(self, "controlled_state_indices", []), dtype=np.int64)
                valid = (idx >= 0) & (idx < q_dim)
                if np.any(valid):
                    state_indices = idx[valid]
                    state_weights = np.ones((len(state_indices),), dtype=np.float64)
            except Exception:
                pass
        if state_indices.size <= 0:
            return result
        state_weights = np.maximum(np.asarray(state_weights, dtype=np.float64), 1e-6)
        wrist_l2 = None
        wrist_l2_mean = None
        wrist_l2_max = None
        left_wrist_abs = None
        left_wrist_abs_mean = None
        left_wrist_abs_max = None
        right_wrist_abs = None
        right_wrist_abs_mean = None
        right_wrist_abs_max = None

        recovery_window = q_arr[-window_len:, :q_dim]
        best_start = 0
        best_loss = float("inf")
        best_diff = None
        best_target_window = None
        for start in range(0, target_arr.shape[0] - window_len + 1):
            target_window = target_arr[start : start + window_len, :q_dim]
            diff = (recovery_window[:, state_indices] - target_window[:, state_indices]) * state_weights[None, :]
            loss = float(np.mean(np.sum(diff * diff, axis=1)))
            if loss < best_loss:
                best_loss = loss
                best_start = start
                best_diff = diff
                best_target_window = target_window
        if best_diff is None or best_target_window is None:
            return result

        frame_l2 = np.linalg.norm(best_diff, axis=1)
        try:
            wrist_indices = [idx for idx in (8, 13) if idx < q_dim]
            if wrist_indices:
                wrist_diff = recovery_window[:, wrist_indices] - best_target_window[:, wrist_indices]
                wrist_l2_arr = np.linalg.norm(wrist_diff, axis=1).astype(np.float32)
                wrist_l2 = wrist_l2_arr.astype(float).tolist()
                wrist_l2_mean = float(np.mean(wrist_l2_arr)) if wrist_l2_arr.size else None
                wrist_l2_max = float(np.max(wrist_l2_arr)) if wrist_l2_arr.size else None
                if 8 in wrist_indices:
                    left_abs_arr = np.abs(wrist_diff[:, wrist_indices.index(8)]).astype(np.float32)
                    left_wrist_abs = left_abs_arr.astype(float).tolist()
                    left_wrist_abs_mean = float(np.mean(left_abs_arr)) if left_abs_arr.size else None
                    left_wrist_abs_max = float(np.max(left_abs_arr)) if left_abs_arr.size else None
                if 13 in wrist_indices:
                    right_abs_arr = np.abs(wrist_diff[:, wrist_indices.index(13)]).astype(np.float32)
                    right_wrist_abs = right_abs_arr.astype(float).tolist()
                    right_wrist_abs_mean = float(np.mean(right_abs_arr)) if right_abs_arr.size else None
                    right_wrist_abs_max = float(np.max(right_abs_arr)) if right_abs_arr.size else None
        except Exception:
            pass
        window_dist = float(np.linalg.norm(best_diff.reshape(-1)))
        window_loss = float(best_loss)
        dq_loss = 0.0
        dq_dist = None
        recovery_step_l2 = None
        target_step_l2 = None
        step_l2_error = None
        dq_error_l2 = None
        dq_cosine = None
        dq_norm_ratio = None
        if window_len >= 2:
            recovery_dq = np.diff(recovery_window[:, state_indices], axis=0) * state_weights[None, :]
            target_dq = np.diff(best_target_window[:, state_indices], axis=0) * state_weights[None, :]
            dq_diff = recovery_dq - target_dq
            dq_error_l2_arr = np.linalg.norm(dq_diff, axis=1)
            dq_loss = float(np.mean(np.sum(dq_diff * dq_diff, axis=1)))
            dq_dist = float(np.linalg.norm(dq_diff.reshape(-1)))
            recovery_norm = np.linalg.norm(recovery_dq, axis=1)
            target_norm = np.linalg.norm(target_dq, axis=1)
            step_l2_error_arr = np.abs(recovery_norm - target_norm)
            denom = np.maximum(recovery_norm * target_norm, 1e-8)
            cos_arr = np.sum(recovery_dq * target_dq, axis=1) / denom
            ratio_arr = recovery_norm / np.maximum(target_norm, 1e-8)
            recovery_step_l2 = recovery_norm.tolist()
            target_step_l2 = target_norm.tolist()
            step_l2_error = step_l2_error_arr.tolist()
            dq_error_l2 = dq_error_l2_arr.tolist()
            dq_cosine = cos_arr.tolist()
            dq_norm_ratio = ratio_arr.tolist()

        action_loss = 0.0
        action_dist = None
        recovery_action_payload = None
        target_action_payload = None
        if candidate is not None and target_chunk is not None:
            try:
                cand_arr = np.asarray(candidate, dtype=np.float64)
                target_action_arr = np.asarray(target_chunk, dtype=np.float64)
                if cand_arr.ndim == 1:
                    cand_arr = cand_arr[None, :]
                if target_action_arr.ndim == 1:
                    target_action_arr = target_action_arr[None, :]
                action_dim = min(cand_arr.shape[1], target_action_arr.shape[1])
                action_len = min(window_len, cand_arr.shape[0], max(0, target_action_arr.shape[0] - best_start))
                if action_dim > 0 and action_len > 0:
                    recovery_action_payload = cand_arr[-action_len:, :action_dim].astype(float).tolist()
                    target_action_payload = target_action_arr[
                        best_start : best_start + action_len, :action_dim
                    ].astype(float).tolist()
                    action_indices = np.arange(action_dim, dtype=np.int64)
                    try:
                        idx = np.asarray(getattr(self, "controlled_action_indices", []), dtype=np.int64)
                        valid = (idx >= 0) & (idx < action_dim)
                        if np.any(valid):
                            action_indices = idx[valid]
                    except Exception:
                        pass
                    action_diff = cand_arr[-action_len:, :action_dim][:, action_indices] - target_action_arr[
                        best_start : best_start + action_len, :action_dim
                    ][:, action_indices]
                    action_loss = float(np.mean(np.sum(action_diff * action_diff, axis=1)))
                    action_dist = float(np.linalg.norm(action_diff.reshape(-1)))
            except Exception:
                action_loss = 0.0
                action_dist = None
                recovery_action_payload = None
                target_action_payload = None

        total_loss = (
            window_loss
            + float(getattr(self, "recover_resume_window_dq_weight", 0.0)) * dq_loss
            + float(getattr(self, "recover_resume_window_action_weight", 0.0)) * action_loss
        )
        result.update(
            {
                "recover_resume_window_available": True,
                "recover_resume_window_len": int(window_len),
                "recover_resume_window_loss": float(window_loss),
                "recover_resume_window_total_loss": float(total_loss),
                "recover_resume_window_dist": float(window_dist),
                "recover_resume_window_error_l2": float(window_dist),
                "recover_resume_window_dq_loss": float(dq_loss),
                "recover_resume_window_dq_dist": dq_dist,
                "recover_resume_window_action_loss": float(action_loss),
                "recover_resume_window_action_dist": action_dist,
                "recover_resume_window_q_frame_l2": frame_l2.tolist(),
                "recover_resume_window_q_frame_l2_mean": float(np.mean(frame_l2)) if frame_l2.size else None,
                "recover_resume_window_q_frame_l2_max": float(np.max(frame_l2)) if frame_l2.size else None,
                "recover_resume_window_wrist_l2": wrist_l2,
                "recover_resume_window_wrist_l2_mean": wrist_l2_mean,
                "recover_resume_window_wrist_l2_max": wrist_l2_max,
                "recover_resume_window_left_wrist_abs": left_wrist_abs,
                "recover_resume_window_left_wrist_abs_mean": left_wrist_abs_mean,
                "recover_resume_window_left_wrist_abs_max": left_wrist_abs_max,
                "recover_resume_window_right_wrist_abs": right_wrist_abs,
                "recover_resume_window_right_wrist_abs_mean": right_wrist_abs_mean,
                "recover_resume_window_right_wrist_abs_max": right_wrist_abs_max,
                "recover_resume_window_recovery_step_l2": recovery_step_l2,
                "recover_resume_window_target_step_l2": target_step_l2,
                "recover_resume_window_step_l2_error": step_l2_error,
                "recover_resume_window_step_l2_error_mean": float(np.mean(step_l2_error)) if step_l2_error is not None else None,
                "recover_resume_window_step_l2_error_max": float(np.max(step_l2_error)) if step_l2_error is not None else None,
                "recover_resume_window_dq_error_l2": dq_error_l2,
                "recover_resume_window_dq_cosine": dq_cosine,
                "recover_resume_window_dq_cosine_mean": float(np.mean(dq_cosine)) if dq_cosine is not None else None,
                "recover_resume_window_dq_cosine_min": float(np.min(dq_cosine)) if dq_cosine is not None else None,
                "recover_resume_window_dq_norm_ratio": dq_norm_ratio,
                "recover_resume_window_dq_norm_ratio_mean": float(np.mean(dq_norm_ratio)) if dq_norm_ratio is not None else None,
                "recover_resume_window_dq_norm_ratio_min": float(np.min(dq_norm_ratio)) if dq_norm_ratio is not None else None,
                "recover_resume_window_start_local_index": int(best_start),
                "recover_resume_window_end_local_index": int(best_start + window_len - 1),
                "recover_resume_window_q": np.asarray(recovery_window, dtype=np.float64).astype(float).tolist(),
                "recover_resume_window_target_q": np.asarray(best_target_window, dtype=np.float64).astype(float).tolist(),
                "recover_resume_window_action": recovery_action_payload,
                "recover_resume_window_target_action": target_action_payload,
            }
        )
        return result

    def _recover_task_progress_cost(
        self,
        obs: Any,
        candidate: np.ndarray,
        nominal: np.ndarray,
        action_idx: np.ndarray,
        reference_chunk: np.ndarray | None = None,
    ) -> CostResult:
        """Score recovery by safe progress toward a viable nominal rejoin."""
        q_seq = self.rollout_nominal_chunk(obs, candidate)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        required_clearance = float(self._acceptance_clearance_threshold())
        constraint = self.clearance_constraint_from_eval(
            safety_eval,
            q_seq.shape[0],
            required_clearance,
        )
        safety_loss = float(constraint.margin_loss)
        action_deviation_loss = float(
            np.square(candidate[:, action_idx] - nominal[:, action_idx]).mean()
        ) if len(action_idx) else 0.0
        smoothness_loss = self._smoothness_loss(candidate, action_idx)
        if (
            len(action_idx)
            and candidate.shape[0] > 1
            and float(self.recover_action_rate_limit) > 0.0
            and float(self.recover_action_rate_limit_weight) > 0.0
        ):
            action_step = np.diff(candidate[:, action_idx], axis=0)
            rate_excess = np.maximum(
                np.abs(action_step) - float(self.recover_action_rate_limit),
                0.0,
            )
            action_rate_limit_loss = float(np.square(rate_excess).mean())
        else:
            action_rate_limit_loss = 0.0
        target_info, rejoin_info, progress_score, progress_available, effective_weight = (
            self._recover_nominal_rejoin_terms(obs, candidate, record=False)
        )
        if not target_info.get("available"):
            progress_score = 0.0
            progress_available = False
        reference_available = reference_chunk is not None
        if reference_available:
            rejoin_info = self.compute_nominal_rejoin_score(
                candidate,
                reference_chunk,
                obs=obs,
            )
        nominal_rejoin_score = float(rejoin_info.get("nominal_rejoin_score", 0.0))
        direction_terms = self._recover_direction_alignment_terms(rejoin_info)
        direction_loss = float(direction_terms["recover_direction_loss"])
        ordered_terms = self._zero_ordered_recovery_terms(0)
        target_q_seq = None
        if reference_available:
            target_q_seq = self.rollout_nominal_chunk(obs, reference_chunk)
            ordered_terms = self._ordered_recovery_path_terms(
                q_seq,
                target_q_seq,
                target_index=0,
            )
        elif target_info.get("available"):
            target_q_seq = self.rollout_nominal_chunk(obs, target_info["target_chunk"])
            ordered_terms = self._ordered_recovery_path_terms(
                q_seq,
                target_q_seq,
                target_index=0,
            )
        ordered_loss = float(ordered_terms["recover_ordered_loss"])
        act_direction_terms = self._recover_act_direction_terms(q_seq, target_q_seq)
        act_direction_loss = float(
            self.recover_act_progress_weight
            * act_direction_terms["recover_act_progress_loss"]
            + self.recover_act_heading_weight
            * act_direction_terms["recover_act_heading_loss"]
        )
        progress_projection_for_affordance = None
        try:
            progress_projection_f = float(
                act_direction_terms.get("recover_act_progress_projection")
            )
            if np.isfinite(progress_projection_f):
                progress_projection_for_affordance = progress_projection_f
        except (TypeError, ValueError):
            progress_projection_for_affordance = None

        resume_context = getattr(self, "resume_affordance_context", {}) or {}
        current_target_distance = None
        if hasattr(resume_context, "get"):
            try:
                current_target_distance_f = float(
                    resume_context.get("resume_target_distance")
                )
                if np.isfinite(current_target_distance_f):
                    current_target_distance = current_target_distance_f
            except (TypeError, ValueError):
                current_target_distance = None
        candidate_target_distance = current_target_distance
        if (
            current_target_distance is not None
            and progress_projection_for_affordance is not None
        ):
            candidate_target_distance = max(
                0.0,
                current_target_distance
                - max(0.0, progress_projection_for_affordance)
                * float(self.recover_resume_affordance_progress_distance_gain),
            )

        taskspace_affordance = {}
        if bool(getattr(self, "recover_resume_affordance_taskspace_in_optimizer", False)):
            taskspace_affordance = self._candidate_resume_affordance_features(
                q_seq[-1:].reshape(1, -1),
                obs=obs,
                source="recover_candidate_terminal_ee_pose",
            )
        taskspace_progress_delta = taskspace_affordance.get("resume_task_progress_delta")
        resume_progress_delta = (
            taskspace_progress_delta
            if taskspace_progress_delta is not None
            else progress_projection_for_affordance
        )

        resume_tube_input = {
            **rejoin_info,
            **ordered_terms,
            **act_direction_terms,
            "recover_prefix_min_clearance": float(constraint.prefix_min_clearance),
            "recover_prefix_safe": bool(constraint.prefix_safe),
            "q_rejoin_threshold": float(self.q_rejoin_threshold),
            "resume_task_progress_delta": resume_progress_delta,
            "resume_control_continuity_score": float(
                1.0 / (1.0 + max(0.0, action_deviation_loss))
            ),
        }
        if candidate_target_distance is not None:
            resume_tube_input.update(
                {
                    "resume_target_distance": float(candidate_target_distance),
                    "resume_target_distance_source": "terminal_progress_projection_surrogate",
                    "resume_current_target_distance": float(current_target_distance),
                    "resume_candidate_target_distance_gain": float(
                        self.recover_resume_affordance_progress_distance_gain
                    ),
                }
            )
        resume_tube_input.update(taskspace_affordance)
        resume_tube_terms = self._resume_tube_score_terms(resume_tube_input)
        resume_tube_score = float(resume_tube_terms["resume_tube_score"])
        target_chunk_for_window = None
        if reference_available:
            target_chunk_for_window = reference_chunk
        elif target_info.get("available"):
            target_chunk_for_window = target_info.get("target_chunk")
        resume_window_terms = self._recover_resume_window_terms(
            q_seq,
            target_q_seq,
            candidate=candidate,
            target_chunk=target_chunk_for_window,
        )
        resume_window_loss = float(
            resume_window_terms.get("recover_resume_window_total_loss", 0.0)
        )
        resume_affordance_terms = self._resume_affordance_score_terms(resume_tube_input)
        resume_affordance_score = float(
            resume_affordance_terms["resume_affordance_score"]
        )
        resume_affordance_component_score = float(
            resume_affordance_terms["resume_affordance_component_score"]
        )
        resume_affordance_available = bool(
            resume_affordance_terms["resume_affordance_available"]
            and resume_affordance_terms["resume_affordance_task_relevant"]
        )
        resume_affordance_score_gap = max(
            0.0,
            float(resume_affordance_terms["resume_affordance_min_score"])
            - resume_affordance_score,
        )
        resume_affordance_component_gap = max(
            0.0,
            float(resume_affordance_terms["resume_affordance_min_component_score"])
            - resume_affordance_component_score,
        )
        resume_affordance_loss = float(
            self.recover_resume_affordance_weight
            * (
                resume_affordance_score_gap * resume_affordance_score_gap
                + resume_affordance_component_gap * resume_affordance_component_gap
            )
            if resume_affordance_available
            else 0.0
        )
        resume_affordance_bonus = float(
            self.recover_resume_affordance_weight
            * resume_affordance_score
            if resume_affordance_available
            else 0.0
        )
        resume_terminal_distance = None
        resume_terminal_distance_source = resume_tube_input.get("resume_target_distance_source")
        try:
            terminal_distance_f = float(
                resume_affordance_terms.get("resume_affordance_target_distance")
            )
            if np.isfinite(terminal_distance_f):
                resume_terminal_distance = terminal_distance_f
        except (TypeError, ValueError):
            resume_terminal_distance = None
        resume_terminal_distance_gap = 0.0
        resume_terminal_distance_loss = 0.0
        if resume_affordance_available and resume_terminal_distance is not None:
            resume_terminal_distance_gap = max(
                0.0,
                float(resume_terminal_distance)
                - float(self.recover_resume_affordance_target_distance_good),
            )
            resume_terminal_distance_loss = float(
                self.recover_resume_affordance_terminal_distance_weight
                * (
                    resume_terminal_distance_gap
                    / max(1e-6, float(self.recover_resume_affordance_target_distance_scale))
                )
                ** 2
            )
        stalled_penalty = 5.0 if progress_score <= 0.0 and nominal_rejoin_score <= 0.0 else 0.0
        existing_loss = float(
            self.recover_safety_weight
            * self.recover_task_progress_clearance_penalty_scale
            * safety_loss
            + self.recover_action_deviation_weight * action_deviation_loss
            + self.recover_smoothness_weight * smoothness_loss
            + self.recover_action_rate_limit_weight * action_rate_limit_loss
            + self.recover_direction_alignment_weight * direction_loss
            + ordered_loss
            + act_direction_loss
            + self.recover_resume_window_weight * resume_window_loss
            + resume_affordance_loss
            + resume_terminal_distance_loss
        )
        recover_score_total = float(
            self.recover_task_progress_weight * progress_score
            + effective_weight * nominal_rejoin_score
            + self.recover_resume_tube_weight * resume_tube_score
            + resume_affordance_bonus
            - stalled_penalty
        )
        total_loss = float(existing_loss - recover_score_total)
        return total_loss, {
            "safety_loss": safety_loss,
            "recover_required_min_clearance": required_clearance,
            "recover_clearance_margin_loss": safety_loss,
            "recover_clearance_penalty_scale": float(self.recover_clearance_penalty_scale),
            "recover_task_progress_clearance_penalty_scale": float(
                self.recover_task_progress_clearance_penalty_scale
            ),
            "action_deviation_loss": action_deviation_loss,
            "smoothness_loss": smoothness_loss,
            "recover_action_rate_limit": float(self.recover_action_rate_limit),
            "recover_action_rate_limit_weight": float(self.recover_action_rate_limit_weight),
            "recover_action_rate_limit_loss": float(action_rate_limit_loss),
            "existing_optimization_loss": existing_loss,
            "total_loss": total_loss,
            "min_clearance": float(constraint.min_clearance),
            "recover_task_progress_score": float(progress_score),
            "progress_score_available": bool(progress_available),
            "recover_score_total": recover_score_total,
            "recover_rejoin_weight_effective": effective_weight,
            "recover_direction_alignment_weight": float(self.recover_direction_alignment_weight),
            "recover_act_progress_weight": float(self.recover_act_progress_weight),
            "recover_act_heading_weight": float(self.recover_act_heading_weight),
            "recover_min_act_heading_cosine": float(self.recover_min_act_heading_cosine),
            "recover_act_direction_loss": float(act_direction_loss),
            "recover_resume_tube_weight": float(self.recover_resume_tube_weight),
            "recover_resume_affordance_weight": float(
                self.recover_resume_affordance_weight
            ),
            "recover_resume_affordance_bonus": float(resume_affordance_bonus),
            "recover_resume_affordance_loss": float(resume_affordance_loss),
            "recover_resume_affordance_score_gap": float(resume_affordance_score_gap),
            "recover_resume_affordance_component_gap": float(
                resume_affordance_component_gap
            ),
            "recover_resume_affordance_terminal_distance": resume_terminal_distance,
            "recover_resume_affordance_terminal_distance_source": resume_terminal_distance_source,
            "recover_resume_affordance_terminal_distance_gap": float(
                resume_terminal_distance_gap
            ),
            "recover_resume_affordance_terminal_distance_loss": float(
                resume_terminal_distance_loss
            ),
            "recover_resume_affordance_terminal_distance_weight": float(
                self.recover_resume_affordance_terminal_distance_weight
            ),
            "recover_resume_affordance_enabled": resume_affordance_terms.get(
                "resume_affordance_enabled"
            ),
            "recover_resume_affordance_available": resume_affordance_terms.get(
                "resume_affordance_available"
            ),
            "recover_resume_affordance_task_relevant": resume_affordance_terms.get(
                "resume_affordance_task_relevant"
            ),
            "recover_resume_affordance_score": float(resume_affordance_score),
            "recover_resume_affordance_ok": resume_affordance_terms.get(
                "resume_affordance_ok"
            ),
            "recover_resume_affordance_min_score": resume_affordance_terms.get(
                "resume_affordance_min_score"
            ),
            "recover_resume_affordance_component_score": float(
                resume_affordance_component_score
            ),
            "recover_resume_affordance_min_component_score": resume_affordance_terms.get(
                "resume_affordance_min_component_score"
            ),
            "recover_resume_affordance_target_distance": resume_affordance_terms.get(
                "resume_affordance_target_distance"
            ),
            "recover_resume_affordance_target_distance_score": resume_affordance_terms.get(
                "resume_affordance_target_distance_score"
            ),
            "recover_resume_affordance_contact_score": resume_affordance_terms.get(
                "resume_affordance_contact_score"
            ),
            "recover_resume_affordance_progress_score": resume_affordance_terms.get(
                "resume_affordance_progress_score"
            ),
            "recover_resume_affordance_alignment_score": resume_affordance_terms.get(
                "resume_affordance_alignment_score"
            ),
            "recover_resume_affordance_continuity_score": resume_affordance_terms.get(
                "resume_affordance_continuity_score"
            ),
            "recover_resume_affordance_safety_score": resume_affordance_terms.get(
                "resume_affordance_safety_score"
            ),
            "recover_resume_affordance_interaction_context": resume_affordance_terms.get(
                "interaction_context"
            ),
            **direction_terms,
            **ordered_terms,
            **act_direction_terms,
            **resume_tube_terms,
            **resume_window_terms,
            **resume_affordance_terms,
            "recover_step_since_deform": int(self.recover_step_since_deform),
            "nominal_rejoin_available": bool(target_info.get("available")),
            "nominal_rejoin_suppressed_reason": target_info.get("suppressed_reason"),
            "nominal_rejoin_clearance": float(target_info.get("nominal_rejoin_clearance", float("-inf"))),
            "nominal_rejoin_safe_prefix_len": int(target_info.get("safe_prefix_len", 0) or 0),
            "nominal_rejoin_window_start": target_info.get("nominal_rejoin_window_start"),
            "nominal_rejoin_window_end": target_info.get("nominal_rejoin_window_end"),
            "nominal_rejoin_window_len": target_info.get("nominal_rejoin_window_len"),
            "nominal_rejoin_window_type": target_info.get("nominal_rejoin_window_type"),
            "safe_rejoin_window_found": bool(target_info.get("safe_rejoin_window_found", False)),
            "short_staging_window_found": bool(target_info.get("short_staging_window_found", False)),
            **rejoin_info,
        }


    def _zero_recover_act_direction_terms(self) -> InfoDict:
        """Return neutral rollout-direction terms when no ACT window exists."""
        return {
            "recover_act_direction_available": False,
            "recover_act_progress_loss": 0.0,
            "recover_act_heading_loss": 0.0,
            "recover_act_progress_projection": 0.0,
            "recover_act_target_progress": 0.0,
            "recover_act_heading_cosine": 1.0,
            "recover_act_heading_cosine_min": 1.0,
            "recover_act_progress_ok": True,
            "recover_act_heading_ok": True,
        }

    def _resume_tube_score_terms(self, info) -> InfoDict:
        """Score whether a recovery endpoint is resumable by ACT.

        This is deliberately task-agnostic: it does not know about handles,
        drawers, or object names.  It asks whether the recovery state is close
        to the nominal ACT state/action tube, whether the short executable
        prefix is safe, and whether ACT's local direction/progress is broadly
        compatible.
        """

        def _finite(value, default=None):
            try:
                value_f = float(value)
            except Exception:  # noqa: BLE001
                return default
            return value_f if np.isfinite(value_f) else default

        def _first_finite(*names, default=None):
            for name in names:
                value = _finite(info.get(name), None)
                if value is not None:
                    return value
            return default

        def _clamp01(value: float) -> float:
            return float(min(1.0, max(0.0, value)))

        terminal_dist = _first_finite(
            "resume_tube_terminal_dist",
            "mpc_handoff_target_tube_terminal_dist",
            "mpc_recovery_target_tube_terminal_dist",
            "mpc_handoff_pose_dist",
            "recover_ordered_terminal_dist",
            "q_rejoin_dist",
            "return_q_rejoin_dist",
            "mpc_handoff_q_rejoin_dist",
        )
        terminal_threshold = _first_finite(
            "resume_tube_terminal_threshold",
            "mpc_handoff_target_tube_dist_threshold",
            "mpc_recovery_target_tube_dist_threshold",
            "mpc_handoff_pose_tube_dist_threshold",
            "q_rejoin_threshold",
            default=float(getattr(self, "q_rejoin_threshold", 1.0)),
        )
        terminal_scale = max(
            float(getattr(self, "recover_resume_tube_distance_scale", 0.75)),
            float(terminal_threshold or 0.0),
            1e-6,
        )
        nominal_rejoin_score = _first_finite("nominal_rejoin_score", default=None)
        if terminal_dist is None:
            terminal_score = _clamp01(float(nominal_rejoin_score or 0.0))
            terminal_ok = terminal_score > 0.0
        else:
            terminal_score = _clamp01(1.0 - terminal_dist / terminal_scale)
            terminal_ok = bool(
                terminal_threshold is None or terminal_dist <= terminal_threshold
            )

        ordered_loss = _first_finite(
            "resume_tube_ordered_loss",
            "recover_ordered_loss",
            "mpc_handoff_recover_ordered_loss",
            default=None,
        )
        if ordered_loss is None:
            path_score = terminal_score
        else:
            path_score = float(1.0 / (1.0 + max(0.0, ordered_loss)))
        tube_score = _clamp01(0.55 * terminal_score + 0.45 * path_score)

        progress_projection = _first_finite(
            "resume_tube_progress_projection",
            "recover_act_progress_projection",
            "mpc_bridge_last_progress_projection",
            "handoff_progress_projection",
            default=None,
        )
        target_progress = max(
            _first_finite(
                "recover_act_target_progress",
                "resume_tube_target_progress",
                default=0.05,
            ),
            0.05,
        )
        progress_ok_raw = bool(info.get("recover_act_progress_ok", True))
        if progress_projection is None:
            progress_score = 1.0 if progress_ok_raw else 0.0
        else:
            progress_score = _clamp01(progress_projection / target_progress)

        heading_cosine = _first_finite(
            "resume_tube_heading_cosine",
            "recover_act_heading_cosine_min",
            "recover_act_heading_cosine",
            "mpc_bridge_last_heading_cosine",
            "handoff_heading_cosine",
            default=None,
        )
        if heading_cosine is None:
            heading_score = 1.0 if bool(info.get("recover_act_heading_ok", True)) else 0.0
        else:
            # Below 0.75 is clearly not ACT-like; 1.0 is perfectly aligned.
            heading_score = _clamp01((heading_cosine - 0.75) / 0.25)

        prefix_clearance = _first_finite(
            "resume_tube_prefix_min_clearance",
            "recover_prefix_min_clearance",
            "mpc_handoff_act_prefix_min_clearance",
            "mpc_handoff_live_prefix_best_min_clearance",
            "nominal_rejoin_clearance",
            "live_monitor_min_clearance",
            "handoff_current_clearance",
            default=None,
        )
        required_clearance = float(
            getattr(self, "recover_resume_tube_min_clearance", 0.02)
        )
        if prefix_clearance is None:
            clearance_score = 0.5
            clearance_ok = False
        else:
            clearance_score = _clamp01(
                (prefix_clearance - required_clearance + 0.04) / 0.04
            )
            clearance_ok = bool(prefix_clearance >= required_clearance)

        live_prefix_eval_count = _first_finite(
            "mpc_handoff_live_prefix_eval_count",
            "nominal_rejoin_live_prefix_eval_count",
            default=None,
        )
        live_prefix_checked = bool(
            live_prefix_eval_count is not None and live_prefix_eval_count >= 1.0
        )
        live_prefix_safe = bool(
            live_prefix_checked
            and (
                info.get("mpc_handoff_live_prefix_safe", False)
                or info.get("nominal_rejoin_live_prefix_safe", False)
                or info.get("live_prefix_safe", False)
            )
        )
        prefix_safe_like = bool(
            info.get("resume_tube_prefix_safe", False)
            or info.get("prefix_safe", False)
            or info.get("recover_prefix_safe", False)
            or live_prefix_safe
            or clearance_ok
        )
        score = _clamp01(
            0.65 * tube_score
            + 0.15 * progress_score
            + 0.10 * heading_score
            + 0.10 * clearance_score
        )
        min_score = float(getattr(self, "recover_resume_tube_min_score", 0.6))
        min_component = float(
            getattr(self, "recover_resume_tube_min_component_score", 0.35)
        )
        tube_component_ok = bool(tube_score >= min_component)
        resume_ok = bool(
            score >= min_score
            and tube_component_ok
            and prefix_safe_like
            and terminal_ok
        )
        return {
            "resume_tube_score": float(score),
            "resume_tube_ok": bool(resume_ok),
            "resume_tube_min_score": float(min_score),
            "resume_tube_component_score": float(tube_score),
            "resume_tube_min_component_score": float(min_component),
            "resume_tube_component_ok": bool(tube_component_ok),
            "resume_tube_terminal_score": float(terminal_score),
            "resume_tube_path_score": float(path_score),
            "resume_tube_progress_score": float(progress_score),
            "resume_tube_heading_score": float(heading_score),
            "resume_tube_clearance_score": float(clearance_score),
            "resume_tube_terminal_dist": terminal_dist,
            "resume_tube_terminal_threshold": terminal_threshold,
            "resume_tube_ordered_loss": ordered_loss,
            "resume_tube_prefix_min_clearance": prefix_clearance,
            "resume_tube_required_clearance": float(required_clearance),
            "resume_tube_prefix_safe": bool(prefix_safe_like),
            "resume_tube_prefix_evaluated": bool(live_prefix_checked),
            "resume_tube_live_prefix_safe": bool(live_prefix_safe),
            "resume_tube_terminal_ok": bool(terminal_ok),
        }

    def _resume_affordance_score_terms(self, info: Mapping[str, Any] | None = None) -> InfoDict:
        """Score whether the current/candidate state affords ACT task resume.

        This helper is intentionally task-adapter driven.  It does not know about
        drawers or handles.  Instead, adapters provide generic fields such as
        ``resume_target_distance``, ``resume_target_contact``,
        ``resume_task_progress``, ``resume_alignment_score``, and
        ``interaction_context``.  Missing features are neutral diagnostics rather
        than hard rejections, so non-manipulation tasks are not accidentally
        blocked by absent adapter data.
        """

        merged: InfoDict = {}
        context = getattr(self, "resume_affordance_context", {})
        if hasattr(context, "items"):
            merged.update(dict(context.items()))
        if info is not None and hasattr(info, "items"):
            merged.update(dict(info.items()))

        def _finite(value, default=None):
            try:
                value_f = float(value)
            except Exception:  # noqa: BLE001
                return default
            return value_f if np.isfinite(value_f) else default

        def _first_finite(*names, default=None):
            for name in names:
                value = _finite(merged.get(name), None)
                if value is not None:
                    return value
            return default

        def _clamp01(value: float) -> float:
            return float(min(1.0, max(0.0, value)))

        def _boolish(value) -> bool | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"", "0", "false", "none", "free", "open", "released"}:
                    return False
                if lowered in {
                    "1",
                    "true",
                    "contact",
                    "contact_rich",
                    "contact-rich",
                    "grasp",
                    "grasped",
                    "holding",
                    "latched",
                    "pull",
                }:
                    return True
                return None
            try:
                arr = np.asarray(value)
                if arr.size == 0:
                    return None
                if arr.dtype.kind in {"b"}:
                    return bool(np.any(arr))
                if arr.dtype.kind in {"i", "u", "f"}:
                    finite = np.asarray(arr, dtype=np.float64)
                    finite = finite[np.isfinite(finite)]
                    if finite.size == 0:
                        return None
                    return bool(np.nanmax(np.abs(finite)) > 0.0)
            except Exception:  # noqa: BLE001
                return None
            return bool(value)

        raw_context = str(
            merged.get("interaction_context")
            or merged.get("resume_interaction_context")
            or merged.get("contact_rich_state")
            or "unknown"
        ).strip()
        context_l = raw_context.lower().replace("-", "_")
        if context_l in {"pre_grasp", "pregrasp", "approach"}:
            interaction_context = "pre_contact"
        elif context_l in {"grasp", "grasping", "contact", "contact_rich", "contact_rich_pause"}:
            interaction_context = "contact_rich"
        elif context_l in {"pull", "pulling", "progress", "manipulation"}:
            interaction_context = "manipulation_progress"
        elif context_l in {"done", "complete", "success"}:
            interaction_context = "done"
        elif context_l in {"free", "free_motion", "pass_through"}:
            interaction_context = "free_motion"
        else:
            interaction_context = raw_context or "unknown"

        target_distance = _first_finite(
            "resume_target_distance",
            "target_distance",
            "ee_to_target_dist",
            "ee_object_distance",
            "ee_to_handle_dist",
            "phase_reanchor_ee_to_handle_dist",
            default=None,
        )
        distance_good = float(
            getattr(self, "recover_resume_affordance_target_distance_good", 0.12)
        )
        distance_scale = float(
            getattr(self, "recover_resume_affordance_target_distance_scale", 0.45)
        )
        if target_distance is None:
            target_distance_score = 0.5
        else:
            target_distance_score = _clamp01(
                1.0 - max(0.0, target_distance - distance_good) / distance_scale
            )
            if interaction_context == "pre_contact":
                # ACT must be allowed to plan the approach itself. Keep the live
                # distance diagnostic, but do not count pre-contact separation as
                # both a target and contact failure. Pose-tube, prefix-safety,
                # history, and action-agreement gates remain hard requirements.
                target_distance_score = max(0.5, target_distance_score)

        contact_value = None
        for key in (
            "resume_target_contact",
            "target_contact",
            "robot_object_contact",
            "ee_object_contact",
            "gripper_contact",
            "gripper_latched",
            "gripper_closed",
            "resume_gripper_closed",
        ):
            contact_value = _boolish(merged.get(key))
            if contact_value is not None:
                break
        contact_expected = interaction_context in {
            "contact_rich",
            "manipulation_progress",
        }
        contact_available = bool(contact_value is not None and (contact_expected or contact_value is True))
        if contact_value is True:
            contact_score = 1.0
        elif contact_value is False and contact_expected:
            contact_score = 0.0
        elif interaction_context == "pre_contact":
            contact_score = 0.5
        elif target_distance is not None:
            contact_score = 1.0 if target_distance <= distance_good else target_distance_score
        else:
            contact_score = 0.5

        progress_value = _first_finite(
            "resume_task_progress",
            "task_progress",
            "task_progress_after",
            "drawer_open_fraction",
            default=None,
        )
        progress_delta = _first_finite(
            "resume_task_progress_delta",
            "task_progress_delta",
            "recover_act_progress_projection",
            default=None,
        )
        progress_scale = float(getattr(self, "recover_resume_affordance_progress_scale", 0.10))
        progress_eps = float(getattr(self, "recover_resume_affordance_progress_epsilon", 0.005))
        if progress_delta is not None:
            progress_score = _clamp01((progress_delta + progress_eps) / (2.0 * progress_eps))
        elif interaction_context == "pre_contact":
            # Pregrasp resume anchors should not require object progress yet.
            progress_score = 0.5
        elif progress_value is not None:
            progress_score = _clamp01(progress_value / progress_scale)
        else:
            progress_score = 0.5

        alignment_score = _first_finite(
            "resume_alignment_score",
            "target_alignment_score",
            default=None,
        )
        alignment_cosine = _first_finite(
            "resume_alignment_cosine",
            "target_alignment_cosine",
            "recover_act_heading_cosine_min",
            "recover_act_heading_cosine",
            "handoff_heading_cosine",
            default=None,
        )
        if alignment_score is None:
            alignment_score = 0.5 if alignment_cosine is None else _clamp01(0.5 * (alignment_cosine + 1.0))
        else:
            alignment_score = _clamp01(alignment_score)

        continuity_score = _first_finite("resume_control_continuity_score", default=None)
        action_deviation_loss = _first_finite("action_deviation_loss", default=None)
        if continuity_score is None:
            continuity_score = (
                0.5
                if action_deviation_loss is None
                else float(1.0 / (1.0 + max(0.0, action_deviation_loss)))
            )
        else:
            continuity_score = _clamp01(continuity_score)

        prefix_clearance = _first_finite(
            "resume_tube_prefix_min_clearance",
            "recover_prefix_min_clearance",
            "mpc_handoff_live_prefix_best_min_clearance",
            "live_monitor_min_clearance",
            "handoff_current_clearance",
            "min_clearance",
            default=None,
        )
        required_clearance = float(getattr(self, "recover_resume_tube_min_clearance", 0.02))
        if prefix_clearance is None:
            safety_score = 0.5
        else:
            safety_score = _clamp01((prefix_clearance - required_clearance + 0.04) / 0.04)

        feature_available = bool(
            target_distance is not None
            or contact_available
            or progress_value is not None
            or progress_delta is not None
            or alignment_cosine is not None
            or merged.get("resume_alignment_score") is not None
            or merged.get("resume_control_continuity_score") is not None
        )
        enabled = bool(getattr(self, "recover_resume_affordance_enabled", True))
        task_relevant = bool(
            feature_available
            and interaction_context not in {"unknown", "free_motion", "done"}
        )
        component_score = float(
            min(
                target_distance_score,
                contact_score if contact_expected and contact_available else 1.0,
                progress_score,
                alignment_score,
                continuity_score,
                safety_score,
            )
        )
        score = _clamp01(
            0.35 * target_distance_score
            + 0.20 * contact_score
            + 0.15 * progress_score
            + 0.15 * alignment_score
            + 0.10 * continuity_score
            + 0.05 * safety_score
        )
        min_score = float(getattr(self, "recover_resume_affordance_min_score", 0.45))
        min_component = float(
            getattr(self, "recover_resume_affordance_min_component_score", 0.25)
        )
        ok = bool(
            (not enabled)
            or (not feature_available)
            or (not task_relevant)
            or (score >= min_score and component_score >= min_component)
        )
        return {
            "interaction_context": interaction_context,
            "resume_adapter": merged.get("resume_adapter"),
            "resume_context_source": merged.get("resume_context_source"),
            "resume_target_label": merged.get("resume_target_label"),
            "resume_affordance_enabled": bool(enabled),
            "resume_affordance_available": bool(feature_available),
            "resume_affordance_task_relevant": bool(task_relevant),
            "resume_affordance_score": float(score if feature_available else 0.0),
            "resume_affordance_ok": bool(ok),
            "resume_affordance_min_score": float(min_score),
            "resume_affordance_component_score": float(component_score),
            "resume_affordance_min_component_score": float(min_component),
            "resume_affordance_target_distance": target_distance,
            "resume_affordance_target_distance_score": float(target_distance_score),
            "resume_affordance_target_distance_good": float(distance_good),
            "resume_affordance_target_distance_scale": float(distance_scale),
            "resume_affordance_contact_score": float(contact_score),
            "resume_affordance_contact_available": bool(contact_available),
            "resume_affordance_progress": progress_value,
            "resume_affordance_progress_delta": progress_delta,
            "resume_affordance_progress_score": float(progress_score),
            "resume_affordance_alignment_score": float(alignment_score),
            "resume_affordance_continuity_score": float(continuity_score),
            "resume_affordance_safety_score": float(safety_score),
            "resume_affordance_prefix_min_clearance": prefix_clearance,
            "resume_affordance_required_clearance": float(required_clearance),
        }


    def _recover_act_direction_terms(
        self,
        candidate_q_seq: np.ndarray,
        target_q_seq: np.ndarray | None,
    ) -> InfoDict:
        """Measure rollout-state progress and heading against the selected ACT window.

        Raw action deltas are only command-space hints.  This helper compares the
        rolled-out robot states so recovery optimization prefers trajectories that
        move forward along the selected ACT rejoin window instead of merely being
        safe and stationary.
        """
        if target_q_seq is None:
            return self._zero_recover_act_direction_terms()
        cand = np.asarray(candidate_q_seq, dtype=np.float32)
        target = np.asarray(target_q_seq, dtype=np.float32)
        if cand.ndim != 2 or target.ndim != 2 or cand.shape[0] < 2 or target.shape[0] < 2:
            return self._zero_recover_act_direction_terms()
        state_idx = np.asarray(self.controlled_state_indices, dtype=np.int64)
        state_idx = state_idx[(state_idx >= 0) & (state_idx < cand.shape[1]) & (state_idx < target.shape[1])]
        if state_idx.size == 0:
            return self._zero_recover_act_direction_terms()
        horizon = min(cand.shape[0], target.shape[0])
        cand_state = cand[:horizon, state_idx]
        target_state = target[:horizon, state_idx]
        target_axis = target_state[-1] - target_state[0]
        axis_norm = float(np.linalg.norm(target_axis))
        eps = 1e-6
        if axis_norm <= eps:
            target_delta_seq = target_state[1:] - target_state[:-1]
            target_delta_norm = np.linalg.norm(target_delta_seq, axis=1)
            if target_delta_norm.size == 0 or float(np.max(target_delta_norm)) <= eps:
                return self._zero_recover_act_direction_terms()
            target_axis = target_delta_seq[int(np.argmax(target_delta_norm))]
            axis_norm = float(np.linalg.norm(target_axis))
        axis = target_axis / max(axis_norm, eps)
        cand_progress = (cand_state[1:] - cand_state[0]) @ axis
        target_progress = (target_state[1:] - target_state[0]) @ axis
        target_progress = np.maximum(target_progress, 0.0)
        progress_deficit = np.maximum(target_progress - cand_progress, 0.0)
        progress_loss = float(np.mean(np.square(progress_deficit))) if progress_deficit.size else 0.0
        final_projection = float(cand_progress[-1]) if cand_progress.size else 0.0
        target_final_progress = float(target_progress[-1]) if target_progress.size else 0.0

        cand_delta = cand_state[1:] - cand_state[:-1]
        target_delta = target_state[1:] - target_state[:-1]
        cand_norm = np.linalg.norm(cand_delta, axis=1)
        target_norm = np.linalg.norm(target_delta, axis=1)
        valid = (cand_norm > eps) & (target_norm > eps)
        if np.any(valid):
            cosine = np.sum(cand_delta[valid] * target_delta[valid], axis=1) / (
                cand_norm[valid] * target_norm[valid] + eps
            )
            cosine = np.clip(cosine, -1.0, 1.0)
            heading_deficit = np.maximum(float(self.recover_min_act_heading_cosine) - cosine, 0.0)
            heading_loss = float(np.mean(np.square(heading_deficit)))
            heading_mean = float(np.mean(cosine))
            heading_min = float(np.min(cosine))
        else:
            heading_loss = 0.0
            heading_mean = 1.0
            heading_min = 1.0
        progress_ok = bool(final_projection > 1e-4 or target_final_progress <= 1e-4)
        heading_ok = bool(heading_min >= float(self.recover_min_act_heading_cosine))
        return {
            "recover_act_direction_available": True,
            "recover_act_progress_loss": progress_loss,
            "recover_act_heading_loss": heading_loss,
            "recover_act_progress_projection": final_projection,
            "recover_act_target_progress": target_final_progress,
            "recover_act_heading_cosine": heading_mean,
            "recover_act_heading_cosine_min": heading_min,
            "recover_act_progress_ok": progress_ok,
            "recover_act_heading_ok": heading_ok,
        }

    def _safechunk_recovery_corridor_info(self) -> InfoDict:
        """Return corridor/recovery counters for diagnostics dictionaries."""
        h = np.asarray(self._recover_path_min_clearance_history, dtype=np.float32)
        return {
            "safechunk_recovery_corridor_enabled": bool(
                self.safechunk_recovery_corridor_enabled
            ),
            "safe_corridor_recovery_count": int(self.safe_corridor_recovery_count),
            "direct_rejoin_attempt_count": int(self.direct_rejoin_attempt_count),
            "direct_rejoin_reject_count": int(self.direct_rejoin_reject_count),
            "detour_rejoin_attempt_count": int(self.detour_rejoin_attempt_count),
            "detour_rejoin_accept_count": int(self.detour_rejoin_accept_count),
            "recovery_bridge_seed_total_count": int(
                self.recovery_bridge_seed_total_count
            ),
            "delayed_rejoin_count": int(self.delayed_rejoin_count),
            "delayed_rejoin_suppressed_count": int(
                self.delayed_rejoin_suppressed_count
            ),
            "recover_path_unsafe_count": int(self.recover_path_unsafe_count),
            "recovery_path_failure_streak": int(self.recovery_path_failure_streak),
            "recovery_path_failure_streak_max": int(
                self.recovery_path_failure_streak_max
            ),
            "repeated_unsafe_target_count": int(self.repeated_unsafe_target_count),
            "post_recovery_act_window_count": int(
                self.post_recovery_act_window_count
            ),
            "post_recovery_act_window_active": bool(
                self.post_recovery_act_window_active
            ),
            "post_recovery_act_steps_remaining": int(
                self.post_recovery_act_steps_remaining
            ),
            "post_recovery_act_window_interrupted_count": int(
                self.post_recovery_act_window_interrupted_count
            ),
            "mean_recover_path_min_clearance": (
                float(np.mean(h)) if h.size else None
            ),
            "min_recover_path_min_clearance": (
                float(np.min(h)) if h.size else None
            ),
            "failed_recovery_target_count": int(len(self.failed_recovery_targets)),
            "failed_recovery_path_count": int(len(self.failed_recovery_paths)),
        }

    def _tick_unsafe_recovery_cooldowns(self) -> None:
        """Age out suppressed unsafe recovery targets."""
        if not self._unsafe_recovery_cooldowns:
            return
        for key in list(self._unsafe_recovery_cooldowns):
            remaining = int(self._unsafe_recovery_cooldowns[key]) - 1
            if remaining <= 0:
                self._unsafe_recovery_cooldowns.pop(key, None)
            else:
                self._unsafe_recovery_cooldowns[key] = remaining

    def make_recovery_target_key(self, target_chunk_or_q: Any) -> str | None:
        """Create a stable rounded key for a recovery target endpoint."""
        arr = np.asarray(target_chunk_or_q, dtype=np.float32)
        if arr.size == 0:
            return "empty"
        if arr.ndim >= 2:
            vec = arr.reshape(arr.shape[0], -1)[-1]
            idx = self.controlled_action_indices[
                self.controlled_action_indices < vec.shape[0]
            ]
        else:
            vec = arr.reshape(-1)
            if vec.shape[0] <= self.expected_motion_dim:
                idx = self.controlled_state_indices[
                    self.controlled_state_indices < vec.shape[0]
                ]
            else:
                idx = self.controlled_action_indices[
                    self.controlled_action_indices < vec.shape[0]
                ]
        vals = vec[idx] if idx.size else vec
        vals = np.round(np.asarray(vals, dtype=np.float32).reshape(-1), 2)
        return ",".join(f"{float(v):.2f}" for v in vals)

    def _make_recovery_path_key(
        self,
        recover_chunk: np.ndarray,
        target_key: str | None = None,
    ) -> str | None:
        """Create a rounded key for the full recovery path."""
        chunk, _ = self._as_chunk(recover_chunk)
        if chunk.shape[0] == 0:
            return f"empty:{target_key}"
        idx = self.controlled_action_indices[
            self.controlled_action_indices < chunk.shape[1]
        ]
        if idx.size:
            sample = np.concatenate([chunk[0, idx], chunk[-1, idx]], axis=0)
        else:
            sample = np.concatenate([chunk[0], chunk[-1]], axis=0)
        sample = np.round(sample.astype(np.float32), 2)
        path_key = ",".join(f"{float(v):.2f}" for v in sample)
        return f"{target_key}:{path_key}"

    def _recovery_target_is_suppressed(self, target_key: Any) -> bool:
        """Check whether repeated unsafe attempts suppress this target."""
        if (
            not self.safechunk_recovery_corridor_enabled
            or not self.suppress_repeated_unsafe_recovery
            or target_key is None
        ):
            return False
        return int(self._unsafe_recovery_cooldowns.get(target_key, 0)) > 0

    def _mark_recovery_path_failure(
        self,
        target_key: Any,
        path_key: Any,
        reason: str,
    ) -> None:
        """Record unsafe recovery targets/paths and update suppression state."""
        if reason not in {"path_unsafe", "prefix_unsafe", "immediate_unsafe"}:
            return
        if target_key is not None:
            self.failed_recovery_targets.append(target_key)
            self.recovery_target_failure_counts[target_key] = (
                int(self.recovery_target_failure_counts.get(target_key, 0)) + 1
            )
            if (
                self.suppress_repeated_unsafe_recovery
                and self.recovery_target_failure_counts[target_key]
                >= self.max_same_target_failures
            ):
                self._unsafe_recovery_cooldowns[target_key] = int(
                    self.unsafe_recovery_cooldown_steps
                )
        if path_key is not None:
            self.failed_recovery_paths.append(path_key)
        self.recovery_path_failure_streak += 1
        self.recovery_path_failure_streak_max = max(
            self.recovery_path_failure_streak_max,
            self.recovery_path_failure_streak,
        )
        self.recover_path_unsafe_count += 1

    def _clear_recovery_path_failure_streak(self) -> None:
        """Clear path-failure streak state after a valid recovery path."""
        self.recovery_path_failure_streak = 0
        self.delayed_rejoin_active = False
        self.delayed_rejoin_steps = 0

    def _activate_post_recovery_act_window(self) -> None:
        """Open the post-recovery nominal pass-through window when configured."""
        if not (
            self.safechunk_recovery_corridor_enabled
            and self.require_post_recovery_act_window
            and self.post_recovery_min_act_steps > 0
        ):
            return
        self.post_recovery_act_window_active = True
        self.post_recovery_act_steps_remaining = int(self.post_recovery_min_act_steps)
        self.post_recovery_act_window_count += 1

    def evaluate_recovery_path_safety(
        self,
        obs: Any,
        recover_chunk: np.ndarray,
        candidate_name: str = "recover",
    ) -> InfoDict:
        """Evaluate immediate, prefix, and full-path safety for recovery chunks."""
        chunk, _ = self._as_chunk(recover_chunk)
        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        required_clearance = float(self._acceptance_clearance_threshold())
        constraint = self.clearance_constraint_from_eval(
            safety_eval,
            q_seq.shape[0],
            required_clearance,
            prefix_len=1,
            require_full_path=bool(self.require_recover_path_safe),
        )
        try:
            acceptance = self.evaluate_candidate_acceptance(
                obs,
                chunk,
                candidate_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Recovery path acceptance helper failed: %s", exc)
            acceptance = {}
        immediate = float(constraint.immediate_clearance)
        path_min = float(constraint.min_clearance)
        safe_prefix_len = int(constraint.safe_prefix_len)
        prefix_min = float(constraint.prefix_min_clearance)
        immediate_safe = bool(constraint.immediate_safe)
        prefix_safe = bool(safe_prefix_len >= 1 and constraint.prefix_safe)
        path_margin_safe = bool(constraint.path_safe)
        path_safe = path_margin_safe if self.require_recover_path_safe else bool(
            immediate_safe and prefix_safe
        )
        reject_reason = None
        if not immediate_safe:
            reject_reason = "immediate_unsafe"
        elif not prefix_safe:
            reject_reason = "prefix_unsafe"
        elif not path_safe:
            reject_reason = "path_unsafe"
        return {
            "path_safe": bool(path_safe),
            "immediate_safe": bool(immediate_safe),
            "prefix_safe": bool(prefix_safe),
            "recover_required_min_clearance": required_clearance,
            "recover_path_min_clearance": path_min,
            "recover_immediate_clearance": immediate,
            "recover_prefix_min_clearance": prefix_min,
            "safe_prefix_len": int(safe_prefix_len),
            "reject_reason": reject_reason,
            "candidate_name": candidate_name,
            "acceptance_type": acceptance.get("acceptance_type"),
            "acceptance_rejection_reason": acceptance.get("rejection_reason"),
        }

    def _recovery_reject_reason(
        self,
        terminal_info: Mapping[str, Any],
        path_info: Mapping[str, Any],
        *,
        repeated_unsafe_target: bool = False,
        task_progress_ok: bool = True,
        direction_ok: bool = True,
        ordered_ok: bool = True,
    ) -> str | None:
        """Return the first policy reason that rejects a recovery candidate."""
        if repeated_unsafe_target:
            return "repeated_unsafe_target"
        if self.safechunk_recovery_corridor_enabled:
            if not bool(path_info.get("immediate_safe")):
                return "immediate_unsafe"
            if not bool(path_info.get("prefix_safe")):
                return "prefix_unsafe"
            if (
                self.require_safe_corridor_for_recovery_complete
                and not bool(path_info.get("path_safe"))
            ):
                return path_info.get("reject_reason") or "path_unsafe"
        if not bool(terminal_info.get("q_rejoin_ok")):
            return "q_rejoin_failed"
        qd_rejoin_required = bool(terminal_info.get("qd_rejoin_required", self.require_qd_rejoin))
        # Treat qdot rejoin as a hard acceptance gate only when the recovery
        # policy explicitly requires qdot rejoin.  When require_qd_rejoin=False,
        # qdot diagnostics remain logged but should not veto an otherwise safe
        # recovery candidate.
        if qd_rejoin_required and bool(terminal_info.get("qd_rejoin_hard_failed", False)):
            return "qdot_rejoin_hard_failed"
        if qd_rejoin_required and not self._coerce_bool(terminal_info.get("qd_rejoin_ok")):
            return "qdot_rejoin_failed"

        def _finite_metric(*names):
            for name in names:
                try:
                    value = terminal_info.get(name)
                except Exception:  # noqa: BLE001
                    value = None
                try:
                    value_f = float(value)
                except Exception:  # noqa: BLE001
                    continue
                if np.isfinite(value_f):
                    return value_f
            return None

        def _finite_threshold(name, default):
            try:
                value_f = float(getattr(self, name, default))
            except Exception:  # noqa: BLE001
                return default
            return value_f

        max_window_mean = _finite_threshold(
            "recover_resume_window_max_q_frame_l2_mean",
            float("inf"),
        )
        if np.isfinite(max_window_mean):
            window_mean = _finite_metric(
                "recover_resume_window_q_frame_l2_mean",
                "mpc_recovery_target_tube_window_q_frame_l2_mean",
            )
            if window_mean is not None and window_mean > max_window_mean:
                return "resume_window_q_l2_failed"

        max_window_max = _finite_threshold(
            "recover_resume_window_max_q_frame_l2_max",
            float("inf"),
        )
        if np.isfinite(max_window_max):
            window_max = _finite_metric(
                "recover_resume_window_q_frame_l2_max",
                "mpc_recovery_target_tube_window_q_frame_l2_max",
            )
            if window_max is not None and window_max > max_window_max:
                return "resume_window_q_l2_failed"

        min_dq_cosine = _finite_threshold(
            "recover_resume_window_min_dq_cosine",
            -float("inf"),
        )
        if np.isfinite(min_dq_cosine):
            dq_cosine_min = _finite_metric(
                "recover_resume_window_dq_cosine_min",
                "mpc_recovery_target_tube_window_dq_cosine_min",
                "mpc_handoff_heading_cosine",
                "mpc_bridge_heading_cosine",
            )
            if dq_cosine_min is not None and dq_cosine_min < min_dq_cosine:
                return "resume_window_dq_alignment_failed"

        max_step_l2_error = _finite_threshold(
            "recover_resume_window_max_step_l2_error",
            float("inf"),
        )
        if np.isfinite(max_step_l2_error):
            step_l2_error = _finite_metric(
                "recover_resume_window_step_l2_error_max",
                "mpc_recovery_target_tube_window_step_l2_error_max",
            )
            if step_l2_error is not None and step_l2_error > max_step_l2_error:
                return "resume_window_step_continuity_failed"

        if bool(getattr(self, "recover_resume_affordance_required_for_accept", False)):
            affordance_available = bool(
                terminal_info.get("recover_resume_affordance_available", False)
                or terminal_info.get("resume_affordance_available", False)
            )
            affordance_task_relevant = bool(
                terminal_info.get("recover_resume_affordance_task_relevant", False)
                or terminal_info.get("resume_affordance_task_relevant", False)
            )
            if affordance_available and affordance_task_relevant:
                affordance_ok = bool(
                    terminal_info.get("recover_resume_affordance_ok", False)
                    or terminal_info.get("resume_affordance_ok", False)
                )
                if not affordance_ok:
                    return "resume_affordance_not_ready"

        min_component_for_accept = _finite_threshold(
            "recover_resume_affordance_min_component_for_accept",
            -float("inf"),
        )
        if np.isfinite(min_component_for_accept):
            affordance_component = _finite_metric(
                "recover_resume_affordance_component_score",
                "resume_affordance_component_score",
                "mpc_handoff_resume_affordance_component_score",
            )
            if affordance_component is not None and affordance_component < min_component_for_accept:
                return "resume_affordance_component_failed"

        if not bool(task_progress_ok):
            return "act_progress_failed"
        if not bool(direction_ok):
            return "direction_alignment_failed"
        if not bool(ordered_ok):
            return "ordered_path_failed"
        return None

    def _make_recovery_bridge_seed_candidates(
        self,
        obs: Any,
        direct_chunk: np.ndarray,
        action_idx: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        """Generate conservative bridge seeds for the recovery optimizer."""
        direct, _ = self._as_chunk(direct_chunk)
        candidates: list[tuple[str, np.ndarray]] = []
        if direct.shape[0] == 0:
            return candidates
        action_idx = np.asarray(action_idx, dtype=np.int64)
        action_idx = action_idx[action_idx < direct.shape[1]]
        passthrough_idx = [
            i for i in range(direct.shape[1]) if i not in set(action_idx.tolist())
        ]

        def add(name: str, candidate: Any) -> None:
            if candidate is None:
                return
            cand = np.asarray(candidate, dtype=np.float32).copy()
            if cand.shape != direct.shape:
                return
            if passthrough_idx:
                cand[:, passthrough_idx] = direct[:, passthrough_idx]
            candidates.append((name, cand))

        q = self.extract_current_q(obs, direct)
        hold_action = direct[0].copy()
        valid_hold = (
            (self.controlled_action_indices < direct.shape[1])
            & (self.controlled_state_indices < q.shape[0])
        )
        if np.any(valid_hold):
            hold_action[self.controlled_action_indices[valid_hold]] = self.deform._controlled_anchor(
                obs,
                direct,
                self.controlled_action_indices[valid_hold],
                self.controlled_state_indices[valid_hold],
            )

        h = direct.shape[0]
        if action_idx.size and h > 1:
            cand = direct.copy()
            cand[0] = hold_action
            for k in range(1, h):
                act_delta = direct[k, action_idx] - direct[k - 1, action_idx]
                cand[k, action_idx] = cand[k - 1, action_idx] + act_delta
            for k in range(h):
                alpha = float(k + 1) / float(max(h, 1))
                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                cand[k, action_idx] = (
                    (1.0 - smooth_alpha) * cand[k, action_idx]
                    + smooth_alpha * direct[k, action_idx]
                )
            add("act_progress_bridge", cand)

        h = direct.shape[0]
        for scale in self.detour_scales:
            if h <= 1:
                cand = np.repeat(hold_action.reshape(1, -1), h, axis=0)
                cand[-1] = direct[-1]
                add(f"hold_prefix_{float(scale):g}", cand)
                continue
            prefix_len = int(round(float(scale) * h))
            prefix_len = max(1, min(h - 1, prefix_len))
            cand = direct.copy()
            cand[:prefix_len] = hold_action
            denom = max(1, h - prefix_len)
            for k in range(prefix_len, h):
                alpha = float(k - prefix_len + 1) / float(denom)
                cand[k, action_idx] = (
                    (1.0 - alpha) * hold_action[action_idx]
                    + alpha * direct[-1, action_idx]
                )
            add(f"hold_prefix_{float(scale):g}", cand)

        last_safe_q = getattr(self, "last_safe_q", None)
        if last_safe_q is None:
            last_safe_q = getattr(self.parent, "last_safe_q", None)
        if last_safe_q is not None and h > 1:
            last_q = np.asarray(last_safe_q, dtype=np.float32).reshape(-1)
            valid_last = (
                (self.controlled_action_indices < direct.shape[1])
                & (self.controlled_state_indices < last_q.shape[0])
            )
            if np.any(valid_last):
                safe_action = hold_action.copy()
                action_idx_last = self.controlled_action_indices[valid_last]
                state_idx_last = self.controlled_state_indices[valid_last]
                modes = self._control_mode_ids_for_state_indices(state_idx_last)
                absolute = modes == 0
                if np.any(absolute):
                    safe_action[action_idx_last[absolute]] = last_q[state_idx_last[absolute]]
                for scale in self.detour_scales:
                    pivot = max(1, min(h - 1, int(round(float(scale) * h))))
                    cand = direct.copy()
                    cand[:pivot] = safe_action
                    denom = max(1, h - pivot)
                    for k in range(pivot, h):
                        alpha = float(k - pivot + 1) / float(denom)
                        cand[k, action_idx] = (
                            (1.0 - alpha) * safe_action[action_idx]
                            + alpha * direct[-1, action_idx]
                        )
                    add(f"last_safe_q_{float(scale):g}", cand)

        last_safe_chunk = getattr(self, "last_safe_chunk", None)
        if last_safe_chunk is None:
            last_safe_chunk = getattr(self.parent, "last_safe_chunk", None)
        if last_safe_chunk is not None:
            safe_chunk = np.asarray(last_safe_chunk, dtype=np.float32)
            if safe_chunk.shape == direct.shape:
                for scale in self.detour_scales:
                    cand = direct + float(scale) * (safe_chunk - direct)
                    cand[-1, action_idx] = direct[-1, action_idx]
                    add(f"last_safe_chunk_{float(scale):g}", cand)

        op = self._get_oscbf_operator()
        if callable(op):
            try:
                action = np.asarray(
                    self._call_single_step_operator(direct[0], obs),
                    dtype=np.float32,
                ).reshape(-1)
                if action.shape[0] == direct.shape[1]:
                    cand = direct.copy()
                    cand[0, action_idx] = action[action_idx]
                    add("oscbf_first_step", cand)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Recovery detour OSCBF candidate failed: %s", exc)

        return candidates

    def _recover_seed_feasible(
        self,
        obs: Any,
        seed_chunk: np.ndarray | None,
    ) -> tuple[bool, InfoDict | None, bool]:
        """Check whether a seed chunk is safe enough to optimize from."""
        if seed_chunk is None or np.asarray(seed_chunk).size == 0:
            return False, None, False
        q_seq = self.rollout_nominal_chunk(obs, seed_chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        min_clearance = float(safety_eval.get("min_clearance", float("-inf")))
        horizon_feasible = min_clearance >= self._acceptance_clearance_threshold()
        first_action = np.asarray(seed_chunk[0], dtype=np.float32)
        q0 = self.extract_current_q(obs, seed_chunk)
        _, one_step = self._committed_action_safety(obs, q0, first_action)
        one_step_clearance = float(one_step.get("min_clearance", float("-inf")))
        immediate_safe = one_step_clearance >= self._committed_abort_threshold()
        return bool(horizon_feasible and immediate_safe), safety_eval, bool(immediate_safe)

    def _make_return_seed_chunk(
        self,
        context: RecoveryContext,
        q_start: np.ndarray,
        current_chunk: np.ndarray,
        action_idx: np.ndarray,
    ) -> tuple[np.ndarray, int | None]:
        """Create a nominal return seed that tracks a future rejoin point."""
        h = min(current_chunk.shape[0], self.return_horizon)
        q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)
        allowed_actions = set(np.asarray(action_idx, dtype=np.int64).reshape(-1).tolist())
        valid = (
            (self.controlled_action_indices < current_chunk.shape[1])
            & (self.controlled_state_indices < context.nominal_q_seq.shape[1])
            & (self.controlled_state_indices < q_start.shape[0])
        )
        if allowed_actions:
            valid &= np.asarray(
                [idx in allowed_actions for idx in self.controlled_action_indices],
                dtype=np.bool_,
            )
        local_action_idx = self.controlled_action_indices[valid]
        state_idx = self.controlled_state_indices[valid]
        if state_idx.size:
            future = context.nominal_q_seq[self.min_rejoin_offset :, state_idx]
            loss, target_index = self._nearest_future_loss(
                q_start[state_idx],
                future,
                weights=None,
                start_index=self.min_rejoin_offset,
            )
            del loss
        else:
            target_index = min(self.min_rejoin_offset, context.nominal_chunk.shape[0] - 1)
        rows = []
        target_rows = []
        for k in range(h):
            src_idx = min(target_index + k, context.nominal_chunk.shape[0] - 1)
            rows.append(context.nominal_chunk[src_idx].copy())
            target_rows.append(context.nominal_q_seq[src_idx].copy())
        return_chunk = np.stack(rows, axis=0).astype(np.float32)
        if target_rows and state_idx.size and local_action_idx.size:
            target_q_seq = np.stack(target_rows, axis=0).astype(np.float32)
            return_chunk = self._write_state_tracking_actions(
                return_chunk,
                q_start,
                target_q_seq,
                local_action_idx,
                state_idx,
            )
        passthrough_idx = [i for i in range(current_chunk.shape[1]) if i not in set(action_idx.tolist())]
        return_chunk[:, passthrough_idx] = current_chunk[:h, passthrough_idx]
        return return_chunk, target_index

    def _optimized_reject_reason_from_flags(
        self,
        safety_rejected: bool,
        recovery_rejected: bool,
    ) -> str | None:
        """Map safety/recovery rejection flags to public reason strings."""
        if safety_rejected and recovery_rejected:
            return "unsafe_and_unrecoverable"
        if safety_rejected:
            return "unsafe"
        if recovery_rejected:
            return "unrecoverable"
        return None

    def _try_recover_after_temporary_wait(
        self,
        obs: Any,
        chunk: np.ndarray,
        q_seq: np.ndarray | None,
        safety_info: Mapping[str, Any] | None,
        info: InfoDict,
        original_shape: Any,
        waited_unsafe_streak: int,
        waited_brake_streak: int,
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Legacy entrypoint for recovery after temporary waiting."""
        if not (
            self.temporary_blocker_enabled
            and self.temporary_recover_after_wait
            and self.recoverable_deform_enabled
            and self.explicit_return
            and self.deformation_enabled
        ):
            return None
        if waited_brake_streak < self.temporary_recover_after_wait_min_brake_steps:
            return None
        if self.post_recovery_act_window_active or self.committed_chunk is not None:
            return None

        recovery_safety = dict(safety_info or {})
        recovery_safety.update(
            {
                "horizon_safe": False,
                "first_violation": 0,
                "unsafe_count": max(1, int(recovery_safety.get("unsafe_count", 0) or 0)),
                "temporary_recover_after_wait_forced": True,
            }
        )
        try:
            recovery_chunk, recovery_info = self.deform.deform_chunk(
                obs,
                chunk,
                safety_info=recovery_safety,
                braked_chunk=chunk,
                nominal_q_seq=q_seq,
                **kwargs,
            )
        except Exception as exc:
            info.update(
                {
                    "temporary_recover_after_wait_attempted": True,
                    "temporary_recover_after_wait_accepted": False,
                    "temporary_recover_after_wait_error": type(exc).__name__,
                    "temporary_recover_after_wait_error_message": str(exc),
                }
            )
            logger.warning("temporary recover-after-wait failed: %s", exc)
            return None

        info.update(recovery_info)
        info.update(
            {
                "temporary_recover_after_wait_attempted": True,
                "temporary_recover_after_wait_waited_unsafe_streak": int(waited_unsafe_streak),
                "temporary_recover_after_wait_waited_brake_streak": int(waited_brake_streak),
                "temporary_recover_after_wait_accepted": bool(
                    recovery_info.get("optimized_accepted", False)
                ),
                "temporary_recover_after_wait_recover_accepted": bool(
                    recovery_info.get("recover_accepted", False)
                ),
            }
        )
        if not bool(recovery_info.get("optimized_accepted", False)):
            return None

        committed, commit_reject_info = self._commit_explicit_recovery_chunk(
            obs,
            recovery_chunk,
            info,
            **kwargs,
        )
        if not committed:
            info.update(commit_reject_info)
            info.update(
                {
                    "temporary_recover_after_wait_committed": False,
                    "temporary_recover_after_wait_commit_rejected": True,
                }
            )
            return None

        self.unsafe_streak = 0
        self.brake_streak = 0
        self.recovery_failure_streak = 0
        info.update(
            {
                "temporary_recover_after_wait_committed": True,
                "safety_mode": "horizon_deform",
                "mode": "temporary_recover_after_wait",
                "deform_mode": "temporary_recover_after_wait",
                "deformation_source": "temporary_recover_after_wait",
            }
        )
        committed_result = self._serve_committed_chunk(
            obs,
            chunk,
            original_shape,
            **kwargs,
        )
        if committed_result is None:
            return None
        committed_chunk, committed_info = committed_result
        committed_info.update({k: v for k, v in info.items() if k not in committed_info})
        committed_info.update(
            {
                "temporary_recover_after_wait_attempted": True,
                "temporary_recover_after_wait_accepted": True,
                "temporary_recover_after_wait_committed": True,
            }
        )
        self.last_info = committed_info
        return committed_chunk, committed_info

    def _current_replay_q(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Return the live replay q vector padded/clipped to expected dimension."""
        q_actual = kwargs.get("q_full")
        if q_actual is None:
            q_actual = self.extract_current_q(obs)
        q_actual = np.asarray(q_actual, dtype=np.float32).reshape(-1)
        if q_actual.shape[0] >= self.expected_motion_dim:
            return q_actual[: self.expected_motion_dim].copy()
        padded = np.zeros(self.expected_motion_dim, dtype=np.float32)
        padded[: q_actual.shape[0]] = q_actual
        return padded

    def _splice_explicit_return_chunk(
        self,
        nominal: np.ndarray,
        deform_chunk: np.ndarray,
        return_chunk: np.ndarray,
        action_idx: np.ndarray,
        target_index: int | None,
    ) -> np.ndarray:
        """Combine deform, recovery, and nominal suffix actions into one chunk."""
        full = np.asarray(nominal, dtype=np.float32).copy()
        y = min(deform_chunk.shape[0], full.shape[0])
        full[:y] = deform_chunk[:y]
        end = min(full.shape[0], y + return_chunk.shape[0])
        if end > y:
            full[y:end] = return_chunk[: end - y]
        if target_index is not None and end < full.shape[0] and self.recovery_context is not None:
            suffix_start = int(target_index) + 1
            for local, k in enumerate(range(end, full.shape[0])):
                src_idx = min(suffix_start + local, self.recovery_context.nominal_chunk.shape[0] - 1)
                full[k] = self.recovery_context.nominal_chunk[src_idx]
        passthrough_idx = [i for i in range(nominal.shape[1]) if i not in set(action_idx.tolist())]
        full[:, passthrough_idx] = nominal[:, passthrough_idx]
        return full.astype(np.asarray(nominal).dtype, copy=False)
