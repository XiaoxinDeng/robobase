from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration path
    jax = None
    jnp = None
    _JAX_AVAILABLE = False

logger = logging.getLogger(__name__)


if _JAX_AVAILABLE:
    @jax.jit
    def _jax_project_candidate_population(nominal, ctrl_samples, action_idx, max_delta, low, high):
        batch = ctrl_samples.shape[0]
        candidates = jnp.broadcast_to(nominal[None, :, :], (batch,) + nominal.shape)
        candidates = candidates.at[:, :, action_idx].set(ctrl_samples)
        nominal_ctrl = nominal[None, :, action_idx]
        delta = candidates[:, :, action_idx] - nominal_ctrl
        clipped_delta = jnp.clip(delta, -max_delta, max_delta)
        ctrl = nominal_ctrl + clipped_delta
        ctrl = jnp.clip(ctrl, low, high)
        return candidates.at[:, :, action_idx].set(ctrl)

    @jax.jit
    def _jax_rollout_chunk_population(action_chunks, q0, state_idx, action_idx, control_mode_ids, dt):
        batch = action_chunks.shape[0]
        q = jnp.broadcast_to(q0[None, :], (batch, q0.shape[0]))
        actions_by_time = jnp.swapaxes(action_chunks, 0, 1)

        def step(q_prev, actions_t):
            q_next = q_prev
            selected = actions_t[:, action_idx]
            current = q_prev[:, state_idx]
            absolute = selected
            delta = current + selected
            velocity = current + dt * selected
            modes = control_mode_ids[None, :]
            updated = jnp.where(modes == 0, absolute, jnp.where(modes == 1, delta, velocity))
            q_next = q_next.at[:, state_idx].set(updated)
            return q_next, q_next

        _, q_seq_time_major = jax.lax.scan(step, q, actions_by_time)
        return jnp.swapaxes(q_seq_time_major, 0, 1)


@dataclass
class RecoveryContext:
    nominal_chunk: np.ndarray
    nominal_q_seq: np.ndarray
    nominal_ee_seq: Optional[np.ndarray] = None
    start_chunk_index: Optional[int] = None
    trigger_step: Optional[int] = None
    active: bool = True
    target_rejoin_index: Optional[int] = None
    phase: str = "horizon_deform"
    observation_history: Any = None
    policy_buffer_metadata: Any = None
    return_retries: int = 0
    recover_retries: int = 0


class SafeChunkDeformFilter:
    """Chunk-level safety wrapper around a single-step OSCBF-style operator."""

    def __init__(
        self,
        oscbf_operator=None,
        horizon: int = 16,
        dt: float = 1.0 / 20.0,
        action_dim: int = 16,
        expected_motion_dim: int = 14,
        control_type: str = "absolute",
        controlled_action_indices=None,
        controlled_state_indices=None,
        min_clearance: float = 0.08,
        brake_progress_threshold: float = 0.05,
        deadlock_window: int = 5,
        deformation_enabled: bool = True,
        mode: str = "candidate",
        chunk_deformation_scales=None,
        chunk_deformation_smoothing: int = 1,
        sequential_oscbf_fallback: bool = False,
        deform_after_deadlock_window: bool = True,
        unsafe_deformation_fallback: str = "brake",
        opt_iters: int = 20,
        opt_lr: float = 0.03,
        lambda_safety: float = 100.0,
        lambda_action: float = 1.0,
        lambda_path: float = 1.0,
        lambda_rejoin: float = 5.0,
        lambda_smooth: float = 0.1,
        rejoin_threshold: float = 0.03,
        min_rejoin_offset: int = 2,
        optimized_fallback: str = "brake",
        detach_passthrough_dims: bool = True,
        use_ee_pose_rejoin: bool = False,
        use_object_state_rejoin: bool = False,
        brake_if_unrecoverable: bool = True,
        recoverable_deform: Optional[dict[str, Any]] = None,
        optimized_deform: Optional[dict[str, Any]] = None,
        explicit_recovery: Optional[dict[str, Any]] = None,
        temporary_blocker: Optional[dict[str, Any]] = None,
        safechunk_replan: Optional[dict[str, Any]] = None,
        safechunk_acceptance: Optional[dict[str, Any]] = None,
        safechunk_recover: Optional[dict[str, Any]] = None,
        safechunk_active_safety: Optional[dict[str, Any]] = None,
        safechunk_recovery_corridor: Optional[dict[str, Any]] = None,
        diagnostics: Optional[dict[str, Any]] = None,
        debug_safety_feasibility: bool = False,
        opt_population: int = 32,
        opt_elite_frac: float = 0.25,
        opt_seed: Optional[int] = 0,
        action_low: Optional[float] = None,
        action_high: Optional[float] = None,
        max_action_delta: Optional[float] = None,
        debug: bool = True,
        enabled: bool = True,
        **kwargs,
    ):
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        if kwargs:
            logger.warning("Unused SafeChunkDeformFilter kwargs: %s", kwargs)

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
        if mode not in {"candidate", "optimized"}:
            raise ValueError(
                "mode must be one of ['candidate', 'optimized'], "
                f"got {mode}"
            )
        optimized_fallback = str(optimized_fallback).lower()
        if optimized_fallback not in {"candidate", "brake"}:
            raise ValueError(
                "optimized_fallback must be one of ['candidate', 'brake'], "
                f"got {optimized_fallback}"
            )

        self.oscbf_operator = oscbf_operator
        self._operator_instantiation_failed = False
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.action_dim = int(action_dim)
        self.expected_motion_dim = int(expected_motion_dim)
        self.control_type = control_type
        self.controlled_action_indices = np.asarray(
            controlled_action_indices
            if controlled_action_indices is not None
            else [4, 5, 6, 7, 9, 10, 11, 12],
            dtype=np.int64,
        )
        self.controlled_state_indices = np.asarray(
            controlled_state_indices
            if controlled_state_indices is not None
            else [4, 5, 6, 7, 9, 10, 11, 12],
            dtype=np.int64,
        )
        self.min_clearance = float(min_clearance)
        self.brake_progress_threshold = float(brake_progress_threshold)
        self.deadlock_window = int(deadlock_window)
        self.deformation_enabled = bool(deformation_enabled)
        self.mode = mode
        self.chunk_deformation_scales = tuple(
            float(x)
            for x in (
                chunk_deformation_scales
                if chunk_deformation_scales is not None
                else [0.0, 0.25, 0.5, 0.75]
            )
        )
        self.chunk_deformation_smoothing = max(0, int(chunk_deformation_smoothing))
        self.sequential_oscbf_fallback = bool(sequential_oscbf_fallback)
        self.deform_after_deadlock_window = bool(deform_after_deadlock_window)
        self.unsafe_deformation_fallback = unsafe_deformation_fallback
        self.opt_iters = max(0, int(opt_iters))
        self.opt_lr = max(1e-9, float(opt_lr))
        self.lambda_safety = float(lambda_safety)
        self.lambda_action = float(lambda_action)
        self.lambda_path = float(lambda_path)
        recoverable_cfg = self._recoverable_deform_config(
            recoverable_deform,
            lambda_rejoin=lambda_rejoin,
            rejoin_threshold=rejoin_threshold,
            min_rejoin_offset=min_rejoin_offset,
            use_ee_pose_rejoin=use_ee_pose_rejoin,
            use_object_state_rejoin=use_object_state_rejoin,
            brake_if_unrecoverable=brake_if_unrecoverable,
        )
        self.recoverable_deform_enabled = bool(recoverable_cfg["enabled"])
        self.lambda_rejoin = float(recoverable_cfg["lambda_rejoin"])
        self.lambda_smooth = float(lambda_smooth)
        self.rejoin_threshold = float(recoverable_cfg["rejoin_threshold"])
        self.q_rejoin_threshold = float(recoverable_cfg["q_rejoin_threshold"])
        self.qd_rejoin_threshold = float(recoverable_cfg["qd_rejoin_threshold"])
        self.require_qd_rejoin = bool(recoverable_cfg["require_qd_rejoin"])
        self.qd_rejoin_hard_threshold = float(
            recoverable_cfg["qd_rejoin_hard_threshold"]
        )
        self.ee_rejoin_threshold = float(recoverable_cfg["ee_rejoin_threshold"])
        self.min_rejoin_offset = max(0, int(recoverable_cfg["min_rejoin_offset"]))
        self.optimized_fallback = optimized_fallback
        self.detach_passthrough_dims = bool(detach_passthrough_dims)
        self.use_ee_pose_rejoin = bool(recoverable_cfg["use_ee_pose_rejoin"])
        self.use_object_state_rejoin = bool(recoverable_cfg["use_object_state_rejoin"])
        self.brake_if_unrecoverable = bool(recoverable_cfg["brake_if_unrecoverable"])
        self.inner_rejoin_metric = str(recoverable_cfg["inner_rejoin_metric"])
        self.final_rejoin_metric = str(recoverable_cfg["final_rejoin_metric"])
        self.cache_nominal_ee = bool(recoverable_cfg["cache_nominal_ee"])
        self.ee_rejoin_in_inner_loop = bool(recoverable_cfg["ee_rejoin_in_inner_loop"])
        self.q_rejoin_weights = recoverable_cfg["q_rejoin_weights"]
        self.explicit_return = bool(recoverable_cfg["explicit_return"])
        self.acceptance_clearance_tol = float(
            recoverable_cfg["acceptance_clearance_tol"]
        )
        self.lambda_yield_safety = float(recoverable_cfg["lambda_yield_safety"])
        self.lambda_yield_action = float(recoverable_cfg["lambda_yield_action"])
        self.lambda_yield_smooth = float(recoverable_cfg["lambda_yield_smooth"])
        self.lambda_retreat = float(recoverable_cfg["lambda_retreat"])
        self.lambda_return_safety = float(recoverable_cfg["lambda_return_safety"])
        self.lambda_return_rejoin = float(recoverable_cfg["lambda_return_rejoin"])
        self.lambda_return_smooth = float(recoverable_cfg["lambda_return_smooth"])
        self.lambda_return_action = float(recoverable_cfg["lambda_return_action"])
        self.yield_horizon = max(1, int(recoverable_cfg["yield_horizon"]))
        self.return_horizon = max(1, int(recoverable_cfg["return_horizon"]))
        self.max_return_retries = max(0, int(recoverable_cfg["max_return_retries"]))
        self.use_ee_final_check = bool(recoverable_cfg["use_ee_final_check"])
        if not self.use_ee_final_check and self.final_rejoin_metric == "ee_pose":
            self.final_rejoin_metric = "q_state"
        self.recovery_context: Optional[RecoveryContext] = None
        optimized_cfg = self._optimized_deform_config(
            optimized_deform,
            debug_safety_feasibility=debug_safety_feasibility,
            jax_batched_optimizer=True,
            jax_batched_optimizer_fallback=True,
        )
        self.debug_safety_feasibility = bool(
            optimized_cfg["debug_safety_feasibility"]
        )
        self.jax_batched_optimizer = bool(optimized_cfg["jax_batched_optimizer"])
        self.jax_batched_optimizer_fallback = bool(
            optimized_cfg["jax_batched_optimizer_fallback"]
        )
        self._warned_jax_unavailable = False
        if self.debug_safety_feasibility and self.final_rejoin_metric == "ee_pose":
            self.final_rejoin_metric = "none"
        explicit_recovery_cfg = self._explicit_recovery_config(explicit_recovery)
        self.commit_accepted_chunks = bool(
            explicit_recovery_cfg["commit_accepted_chunks"]
        )
        self.committed_chunk_safety_check = bool(
            explicit_recovery_cfg["committed_chunk_safety_check"]
        )
        self.committed_safety_tol = float(
            explicit_recovery_cfg["committed_safety_tol"]
        )
        self.committed_abort_only_if_contact_risk = bool(
            explicit_recovery_cfg["committed_abort_only_if_contact_risk"]
        )
        self.committed_min_clearance_for_abort = float(
            explicit_recovery_cfg["committed_min_clearance_for_abort"]
        )
        self.repair_committed_action = bool(
            explicit_recovery_cfg["repair_committed_action"]
        )
        self.monotonic_committed_repair = bool(
            explicit_recovery_cfg["monotonic_committed_repair"]
        )
        self.committed_execution_margin = float(
            explicit_recovery_cfg["committed_execution_margin"]
        )
        self.committed_state_error_threshold = float(
            explicit_recovery_cfg["committed_state_error_threshold"]
        )
        self.committed_state_error_action = str(
            explicit_recovery_cfg["committed_state_error_action"]
        ).lower()
        self.committed_state_mismatch_abort_requires_unsafe = bool(
            explicit_recovery_cfg["committed_state_mismatch_abort_requires_unsafe"]
        )
        self.replan_committed_suffix_on_state_mismatch = bool(
            explicit_recovery_cfg["replan_committed_suffix_on_state_mismatch"]
        )
        self.committed_suffix_replan_min_remaining = max(
            1,
            int(explicit_recovery_cfg["committed_suffix_replan_min_remaining"]),
        )
        self.opportunistic_act_resume = bool(
            explicit_recovery_cfg["opportunistic_act_resume"]
        )
        raw_resume_q_threshold = explicit_recovery_cfg[
            "opportunistic_resume_q_threshold"
        ]
        self.opportunistic_resume_q_threshold = (
            None
            if raw_resume_q_threshold is None
            else float(raw_resume_q_threshold)
        )
        self.opportunistic_resume_min_clearance = float(
            explicit_recovery_cfg["opportunistic_resume_min_clearance"]
        )
        self.max_recover_steps_before_act_resume = max(
            0,
            int(explicit_recovery_cfg["max_recover_steps_before_act_resume"]),
        )
        self.max_suffix_replans_per_recovery = max(
            0,
            int(explicit_recovery_cfg["max_suffix_replans_per_recovery"]),
        )
        if self.committed_state_error_action not in {"replan", "abort_to_brake"}:
            raise ValueError(
                "explicit_recovery.committed_state_error_action must be one of "
                "['replan', 'abort_to_brake'], got "
                f"{self.committed_state_error_action}"
            )
        temporary_blocker_cfg = self._temporary_blocker_config(temporary_blocker)
        self.temporary_blocker_enabled = bool(temporary_blocker_cfg["enabled"])
        self.temporary_prefer_brake_before_deform = bool(
            temporary_blocker_cfg["prefer_brake_before_deform"]
        )
        self.temporary_min_unsafe_steps_before_deform = int(
            temporary_blocker_cfg["min_unsafe_steps_before_deform"]
        )
        self.temporary_max_brake_steps_before_deform = int(
            temporary_blocker_cfg["max_brake_steps_before_deform"]
        )
        self.temporary_reset_on_nominal_safe = bool(
            temporary_blocker_cfg["reset_on_nominal_safe"]
        )
        self.temporary_require_progress_deadlock_before_deform = bool(
            temporary_blocker_cfg["require_progress_deadlock_before_deform"]
        )
        self.temporary_progress_window = int(
            temporary_blocker_cfg["progress_window"]
        )
        self.temporary_min_progress_delta = float(
            temporary_blocker_cfg["min_progress_delta"]
        )
        self.temporary_recover_after_wait = bool(
            temporary_blocker_cfg["recover_after_wait"]
        )
        self.temporary_recover_after_wait_min_brake_steps = int(
            temporary_blocker_cfg["recover_after_wait_min_brake_steps"]
        )
        replan_cfg = self._safechunk_replan_config(safechunk_replan)
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
        acceptance_cfg = self._safechunk_acceptance_config(safechunk_acceptance)
        self.safechunk_acceptance_enabled = bool(acceptance_cfg["enabled"])
        self.acceptance_hard_min_clearance = float(acceptance_cfg["hard_min_clearance"])
        self.acceptance_desired_min_clearance = float(acceptance_cfg["desired_min_clearance"])
        self.allow_safe_prefix_execution = bool(
            acceptance_cfg["allow_safe_prefix_execution"]
        )
        self.min_safe_prefix_len = int(acceptance_cfg["min_safe_prefix_len"])
        self.prefix_min_clearance = float(acceptance_cfg["prefix_min_clearance"])
        self.rolling_replan_on_prefix = bool(acceptance_cfg["rolling_replan_on_prefix"])
        self.full_horizon_required_for_recover = bool(
            acceptance_cfg["full_horizon_required_for_recover"]
        )
        self.full_horizon_required_for_deform = bool(
            acceptance_cfg["full_horizon_required_for_deform"]
        )
        self.emergency_brake_if_immediate_below_hard_margin = bool(
            acceptance_cfg["emergency_brake_if_immediate_below_hard_margin"]
        )
        self.allow_candidate_fallback = bool(acceptance_cfg["allow_candidate_fallback"])
        self.candidate_fallback_only_if_no_optimized_result = bool(
            acceptance_cfg["candidate_fallback_only_if_no_optimized_result"]
        )
        recover_cfg = self._safechunk_recover_config(safechunk_recover)
        self.safechunk_recover_enabled = bool(recover_cfg["enabled"])
        self.recover_rejoin_nominal_weight = float(recover_cfg["rejoin_nominal_weight"])
        self.recover_task_progress_weight = float(recover_cfg["task_progress_weight"])
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
        self.recover_ordered_pose_weight = float(recover_cfg["ordered_pose_weight"])
        self.recover_ordered_delta_weight = float(recover_cfg["ordered_delta_weight"])
        self.recover_ordered_pose_threshold = float(recover_cfg["ordered_pose_threshold"])
        self.recover_ordered_delta_threshold = float(recover_cfg["ordered_delta_threshold"])
        self.require_recover_ordered_path = bool(recover_cfg["require_ordered_path"])
        self.recover_retry_cooldown_steps = max(
            0,
            int(recover_cfg["retry_cooldown_steps"]),
        )
        self.recover_max_attempts_per_unsafe_streak = max(
            0,
            int(recover_cfg["max_attempts_per_unsafe_streak"]),
        )
        self.recover_safety_weight = float(recover_cfg["safety_weight"])
        self.recover_action_deviation_weight = float(recover_cfg["action_deviation_weight"])
        self.recover_smoothness_weight = float(recover_cfg["smoothness_weight"])
        self.require_nominal_prefix_safe_for_rejoin = bool(
            recover_cfg["require_nominal_prefix_safe_for_rejoin"]
        )
        self.nominal_rejoin_prefix_min_clearance = float(
            recover_cfg["nominal_rejoin_prefix_min_clearance"]
        )
        self.use_latest_nominal_for_rejoin = bool(recover_cfg["use_latest_nominal_for_rejoin"])
        self.suppress_stale_nominal_rejoin = bool(
            recover_cfg["suppress_stale_nominal_rejoin"]
        )
        self.rejoin_weight_schedule = str(recover_cfg["rejoin_weight_schedule"]).lower()
        self.rejoin_ramp_steps = max(1, int(recover_cfg["rejoin_ramp_steps"]))
        active_cfg = self._safechunk_active_safety_config(safechunk_active_safety)
        self.safechunk_active_safety_enabled = bool(active_cfg["enabled"])
        self.check_hold_horizon_safety = bool(active_cfg["check_hold_horizon_safety"])
        self.predict_human_motion_for_hold = bool(active_cfg["predict_human_motion_for_hold"])
        self.active_safety_hard_min_clearance = float(active_cfg["hard_min_clearance"])
        self.hold_prefix_min_clearance = float(active_cfg["hold_prefix_min_clearance"])
        self.hold_horizon_steps = max(1, int(active_cfg["hold_horizon_steps"]))
        self.emergency_deform_when_hold_unsafe = bool(
            active_cfg["emergency_deform_when_hold_unsafe"]
        )
        self.optimize_when_hold_unsafe = bool(active_cfg["optimize_when_hold_unsafe"])
        self.emergency_deform_candidate_scales = tuple(
            float(x) for x in active_cfg["emergency_deform_candidate_scales"]
        )
        self.prefer_last_safe_action = bool(active_cfg["prefer_last_safe_action"])
        self.prefer_last_safe_q_retract = bool(active_cfg["prefer_last_safe_q_retract"])
        self.emergency_deform_replan_next_step = bool(
            active_cfg["emergency_deform_replan_next_step"]
        )
        corridor_cfg = self._safechunk_recovery_corridor_config(
            safechunk_recovery_corridor
        )
        self.safechunk_recovery_corridor_enabled = bool(corridor_cfg["enabled"])
        self.require_recover_path_safe = bool(
            corridor_cfg["require_recover_path_safe"]
        )
        self.recover_path_min_clearance = float(
            corridor_cfg["recover_path_min_clearance"]
        )
        self.recover_immediate_hard_clearance = float(
            corridor_cfg["recover_immediate_hard_clearance"]
        )
        self.recover_prefix_min_clearance = float(
            corridor_cfg["recover_prefix_min_clearance"]
        )
        self.enable_direct_rejoin = bool(corridor_cfg["enable_direct_rejoin"])
        self.enable_detour_rejoin = bool(corridor_cfg["enable_detour_rejoin"])
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
        self.detour_scales = tuple(float(x) for x in corridor_cfg["detour_scales"])
        self.detour_clearance_weight = float(corridor_cfg["detour_clearance_weight"])
        self.detour_task_rejoin_weight = float(
            corridor_cfg["detour_task_rejoin_weight"]
        )
        self.detour_action_norm_weight = float(
            corridor_cfg["detour_action_norm_weight"]
        )
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
        self.diagnostics = dict(diagnostics or {})
        self.opt_population = max(4, int(opt_population))
        self.opt_elite_frac = float(np.clip(opt_elite_frac, 1.0 / self.opt_population, 1.0))
        self.action_low = action_low
        self.action_high = action_high
        self._rng = np.random.default_rng(opt_seed)
        self._optimizer_warmup_done = False
        self._optimizer_warmup_cache = set()
        self._optimizer_warmup_info: dict[str, Any] = {}
        self.max_action_delta = (
            None if max_action_delta is None else float(max_action_delta)
        )
        self.debug = bool(debug)
        self.enabled = bool(enabled)
        self.last_info: dict[str, Any] = {}
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
        self._pending_committed_replan_info = None
        self.committed_suffix_replan_attempt_count = 0
        self.committed_suffix_replan_accepted_count = 0
        self.committed_suffix_replan_rejected_count = 0
        self.committed_suffix_replan_budget_suppressed_count = 0
        self.committed_opportunistic_resume_count = 0
        self.committed_recovery_budget_exit_count = 0
        self.committed_recover_steps_since_act = 0
        self.committed_suffix_replans_in_current_recovery = 0
        self._trigger_count = 0
        self.unsafe_streak = 0
        self.brake_streak = 0
        self.recovery_failure_streak = 0
        self.recovery_failure_streak_max = 0
        self.recovery_optimizer_cooldown_remaining = 0
        self.recovery_attempts_in_unsafe_streak = 0
        self.recovery_optimization_skipped_count = 0
        self.current_deform_plan = None
        self.current_recovery_plan = None
        self.deform_anchor_state = None
        self.recovery_anchor_state = None
        self.failed_recovery_targets = []
        self.failed_recovery_paths = []
        self.recovery_path_failure_streak = 0
        self.recovery_path_failure_streak_max = 0
        self.recovery_target_failure_counts = {}
        self._unsafe_recovery_cooldowns = {}
        self.delayed_rejoin_active = False
        self.delayed_rejoin_steps = 0
        self.safe_corridor_recovery_count = 0
        self.direct_rejoin_attempt_count = 0
        self.direct_rejoin_reject_count = 0
        self.detour_rejoin_attempt_count = 0
        self.detour_rejoin_accept_count = 0
        self.delayed_rejoin_count = 0
        self.recover_path_unsafe_count = 0
        self.post_recovery_act_window_count = 0
        self.repeated_unsafe_target_count = 0
        self.post_recovery_act_window_active = False
        self.post_recovery_act_steps_remaining = 0
        self.post_recovery_act_window_interrupted_count = 0
        self._recover_path_min_clearance_history = []
        self.deform_replan_count = 0
        self.recovery_replan_count = 0
        self.stale_recovery_suppressed_count = 0
        self.recovery_target_infeasible_count = 0
        self.emergency_brake_steps = 0
        self.optimized_candidate_count = 0
        self.optimized_solution_count = 0
        self.fallback_candidate_count = 0
        self.fallback_candidate_accepted_count = 0
        self.optimized_rejected_count = 0
        self.deform_candidate_count = 0
        self.deform_accepted_count = 0
        self.deform_rejected_count = 0
        self.recover_candidate_count = 0
        self.recover_accepted_count = 0
        self.recover_rejected_count = 0
        self.safe_prefix_accepted_count = 0
        self.first_action_only_accepted_count = 0
        self.immediate_hard_reject_count = 0
        self.no_safe_prefix_reject_count = 0
        self.horizon_margin_reject_count = 0
        self.accepted_deform_steps = 0
        self.accepted_recover_steps = 0
        self.fallback_brake_after_reject_count = 0
        self.latest_nominal_chunk = None
        self.latest_nominal_step = 0
        self.blocked_nominal_chunk = None
        self.blocked_nominal_step = None
        self.recover_step_since_deform = 0
        self.nominal_rejoin_available_count = 0
        self.nominal_rejoin_suppressed_count = 0
        self.stale_nominal_rejoin_suppressed_count = 0
        self.nominal_prefix_unsafe_suppressed_count = 0
        self.recover_positive_projection_count = 0
        self.recover_nonpositive_projection_count = 0
        self._recover_projection_history = []
        self._recover_cosine_history = []
        self._recover_task_progress_history = []
        self._recover_ordered_pose_loss_history = []
        self._recover_ordered_delta_loss_history = []
        self._recover_ordered_loss_history = []
        self.last_safe_action = None
        self.last_safe_q = None
        self.last_safe_chunk = None
        self.emergency_deform_away_steps = 0
        self.emergency_deform_away_count = 0
        self.contact_during_hold_count = 0
        self.contact_during_brake_count = 0
        self.contact_during_deform_count = 0
        self.contact_during_recover_count = 0
        self.hold_unsafe_count = 0
        self.hold_predicted_contact_count = 0
        self._hold_horizon_min_clearance_history = []
        self._previous_human_snapshot = None
        self._temporary_progress_history = []
        self._warned_no_safety_eval = False
        self._deadlock_count = 0
        # A safe brake can still destroy the task by freezing most of the ACT
        # chunk. Try deformation first when braking would keep too little of the
        # nominal horizon.
        self.task_progress_brake_threshold = 0.5

    def _deprecated_config_value(self, cfg, old_key, new_key, default):
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

    def _recoverable_deform_config(self, config, **defaults):
        cfg = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))

        inner_metric = str(
            cfg.get(
                "inner_rejoin_metric",
                "ee_pose"
                if bool(cfg.get("ee_rejoin_in_inner_loop", False))
                else "q_state",
            )
        ).lower()
        final_metric = str(cfg.get("final_rejoin_metric", "ee_pose")).lower()
        if bool(cfg.get("use_ee_pose_rejoin", False)) and bool(
            cfg.get("ee_rejoin_in_inner_loop", False)
        ):
            inner_metric = "ee_pose"
        if inner_metric not in {"q_state", "ee_pose"}:
            raise ValueError(
                "recoverable_deform.inner_rejoin_metric must be one of "
                "['q_state', 'ee_pose'], got "
                f"{inner_metric}"
            )
        if final_metric not in {"none", "q_state", "ee_pose"}:
            raise ValueError(
                "recoverable_deform.final_rejoin_metric must be one of "
                "['none', 'q_state', 'ee_pose'], got "
                f"{final_metric}"
            )
        ee_rejoin_in_inner_loop = bool(
            cfg.get("ee_rejoin_in_inner_loop", inner_metric == "ee_pose")
        )
        if inner_metric == "ee_pose":
            ee_rejoin_in_inner_loop = True

        rejoin_threshold = float(
            cfg.get("rejoin_threshold", defaults["rejoin_threshold"])
        )
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "lambda_rejoin": float(cfg.get("lambda_rejoin", defaults["lambda_rejoin"])),
            "rejoin_threshold": rejoin_threshold,
            "q_rejoin_threshold": float(
                cfg.get("q_rejoin_threshold", cfg.get("rejoin_threshold", 0.5))
            ),
            "qd_rejoin_threshold": float(cfg.get("qd_rejoin_threshold", 5.0)),
            "require_qd_rejoin": bool(cfg.get("require_qd_rejoin", False)),
            "qd_rejoin_hard_threshold": float(
                cfg.get(
                    "qd_rejoin_hard_threshold",
                    cfg.get("qd_rejoin_threshold", 5.0) * 4.0,
                )
            ),
            "ee_rejoin_threshold": float(cfg.get("ee_rejoin_threshold", 0.08)),
            "min_rejoin_offset": int(
                cfg.get("min_rejoin_offset", defaults["min_rejoin_offset"])
            ),
            "use_ee_pose_rejoin": bool(
                cfg.get("use_ee_pose_rejoin", defaults["use_ee_pose_rejoin"])
            ),
            "use_object_state_rejoin": bool(
                cfg.get(
                    "use_object_state_rejoin",
                    defaults["use_object_state_rejoin"],
                )
            ),
            "brake_if_unrecoverable": bool(
                cfg.get(
                    "brake_if_unrecoverable",
                    defaults["brake_if_unrecoverable"],
                )
            ),
            "inner_rejoin_metric": inner_metric,
            "final_rejoin_metric": final_metric,
            "cache_nominal_ee": bool(cfg.get("cache_nominal_ee", True)),
            "ee_rejoin_in_inner_loop": ee_rejoin_in_inner_loop,
            "q_rejoin_weights": cfg.get("q_rejoin_weights"),
            "explicit_return": bool(
                self._deprecated_config_value(
                    cfg, "explicit_return", "explicit_recovery", False
                )
            ),
            "acceptance_clearance_tol": float(
                cfg.get("acceptance_clearance_tol", 0.005)
            ),
            "lambda_yield_safety": float(
                self._deprecated_config_value(
                    cfg, "lambda_yield_safety", "lambda_deform_safety", 800.0
                )
            ),
            "lambda_yield_action": float(
                self._deprecated_config_value(
                    cfg, "lambda_yield_action", "lambda_deform_action", 0.1
                )
            ),
            "lambda_yield_smooth": float(
                self._deprecated_config_value(
                    cfg, "lambda_yield_smooth", "lambda_deform_smooth", 0.1
                )
            ),
            "lambda_retreat": float(cfg.get("lambda_retreat", 1.0)),
            "lambda_return_safety": float(
                self._deprecated_config_value(
                    cfg, "lambda_return_safety", "lambda_recover_safety", 500.0
                )
            ),
            "lambda_return_rejoin": float(
                self._deprecated_config_value(
                    cfg, "lambda_return_rejoin", "lambda_recover_rejoin", 5.0
                )
            ),
            "lambda_return_smooth": float(
                self._deprecated_config_value(
                    cfg, "lambda_return_smooth", "lambda_recover_smooth", 0.2
                )
            ),
            "lambda_return_action": float(
                self._deprecated_config_value(
                    cfg, "lambda_return_action", "lambda_recover_action", 0.1
                )
            ),
            "yield_horizon": int(
                self._deprecated_config_value(cfg, "yield_horizon", "deform_horizon", 4)
            ),
            "return_horizon": int(
                self._deprecated_config_value(cfg, "return_horizon", "recover_horizon", 8)
            ),
            "max_return_retries": int(
                self._deprecated_config_value(
                    cfg, "max_return_retries", "max_recover_retries", 3
                )
            ),
            "use_ee_final_check": bool(cfg.get("use_ee_final_check", True)),
        }

    def _explicit_recovery_config(self, config):
        cfg = {}
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
        }

    def _temporary_blocker_config(self, config):
        cfg = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "prefer_brake_before_deform": bool(
                cfg.get("prefer_brake_before_deform", True)
            ),
            "min_unsafe_steps_before_deform": int(
                cfg.get("min_unsafe_steps_before_deform", 8)
            ),
            "max_brake_steps_before_deform": int(
                cfg.get("max_brake_steps_before_deform", 12)
            ),
            "reset_on_nominal_safe": bool(cfg.get("reset_on_nominal_safe", True)),
            "require_progress_deadlock_before_deform": bool(
                cfg.get("require_progress_deadlock_before_deform", True)
            ),
            "progress_window": int(cfg.get("progress_window", 10)),
            "min_progress_delta": float(cfg.get("min_progress_delta", 0.001)),
            "recover_after_wait": bool(cfg.get("recover_after_wait", False)),
            "recover_after_wait_min_brake_steps": int(
                cfg.get("recover_after_wait_min_brake_steps", 1)
            ),
        }

    def _safechunk_replan_config(self, config):
        cfg = {}
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

    def _safechunk_active_safety_config(self, config):
        cfg = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "check_hold_horizon_safety": bool(cfg.get("check_hold_horizon_safety", True)),
            "predict_human_motion_for_hold": bool(cfg.get("predict_human_motion_for_hold", True)),
            "hard_min_clearance": float(cfg.get("hard_min_clearance", 0.02)),
            "hold_prefix_min_clearance": float(cfg.get("hold_prefix_min_clearance", 0.04)),
            "hold_horizon_steps": int(cfg.get("hold_horizon_steps", 4)),
            "emergency_deform_when_hold_unsafe": bool(
                cfg.get("emergency_deform_when_hold_unsafe", True)
            ),
            "optimize_when_hold_unsafe": bool(
                cfg.get("optimize_when_hold_unsafe", True)
            ),
            "emergency_deform_candidate_scales": tuple(
                cfg.get("emergency_deform_candidate_scales", (0.25, 0.5, 0.75, 1.0))
            ),
            "prefer_last_safe_action": bool(cfg.get("prefer_last_safe_action", True)),
            "prefer_last_safe_q_retract": bool(cfg.get("prefer_last_safe_q_retract", True)),
            "emergency_deform_replan_next_step": bool(
                cfg.get("emergency_deform_replan_next_step", True)
            ),
        }

    def _safechunk_recovery_corridor_config(self, config):
        cfg = {}
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
            "enable_detour_rejoin": bool(cfg.get("enable_detour_rejoin", False)),
            "enable_delayed_rejoin": bool(cfg.get("enable_delayed_rejoin", True)),
            "suppress_repeated_unsafe_recovery": bool(
                cfg.get("suppress_repeated_unsafe_recovery", True)
            ),
            "unsafe_recovery_cooldown_steps": int(
                cfg.get("unsafe_recovery_cooldown_steps", 8)
            ),
            "max_same_target_failures": int(cfg.get("max_same_target_failures", 2)),
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

    def _safechunk_recover_config(self, config):
        cfg = {}
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
            "ordered_pose_threshold": float(cfg.get("ordered_pose_threshold", 0.02)),
            "ordered_delta_threshold": float(cfg.get("ordered_delta_threshold", 0.005)),
            "require_ordered_path": bool(cfg.get("require_ordered_path", True)),
            "retry_cooldown_steps": int(cfg.get("retry_cooldown_steps", 4)),
            "max_attempts_per_unsafe_streak": int(
                cfg.get("max_attempts_per_unsafe_streak", 3)
            ),
            "safety_weight": float(cfg.get("safety_weight", 100.0)),
            "action_deviation_weight": float(cfg.get("action_deviation_weight", 0.2)),
            "smoothness_weight": float(cfg.get("smoothness_weight", 0.1)),
            "require_nominal_prefix_safe_for_rejoin": bool(
                cfg.get("require_nominal_prefix_safe_for_rejoin", True)
            ),
            "nominal_rejoin_prefix_min_clearance": float(
                cfg.get("nominal_rejoin_prefix_min_clearance", 0.04)
            ),
            "use_latest_nominal_for_rejoin": bool(
                cfg.get("use_latest_nominal_for_rejoin", True)
            ),
            "suppress_stale_nominal_rejoin": bool(
                cfg.get("suppress_stale_nominal_rejoin", True)
            ),
            "rejoin_weight_schedule": schedule,
            "rejoin_ramp_steps": int(cfg.get("rejoin_ramp_steps", 5)),
        }

    def _safechunk_acceptance_config(self, config):
        cfg = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "hard_min_clearance": float(cfg.get("hard_min_clearance", 0.02)),
            "desired_min_clearance": float(cfg.get("desired_min_clearance", 0.08)),
            "allow_safe_prefix_execution": bool(
                cfg.get("allow_safe_prefix_execution", True)
            ),
            "min_safe_prefix_len": int(cfg.get("min_safe_prefix_len", 1)),
            "prefix_min_clearance": float(cfg.get("prefix_min_clearance", 0.04)),
            "rolling_replan_on_prefix": bool(cfg.get("rolling_replan_on_prefix", True)),
            "full_horizon_required_for_recover": bool(
                cfg.get("full_horizon_required_for_recover", False)
            ),
            "full_horizon_required_for_deform": bool(
                cfg.get("full_horizon_required_for_deform", False)
            ),
            "emergency_brake_if_immediate_below_hard_margin": bool(
                cfg.get("emergency_brake_if_immediate_below_hard_margin", True)
            ),
            "allow_candidate_fallback": bool(cfg.get("allow_candidate_fallback", False)),
            "candidate_fallback_only_if_no_optimized_result": bool(
                cfg.get("candidate_fallback_only_if_no_optimized_result", True)
            ),
        }

    def _optimized_deform_config(self, config, **defaults):
        cfg = {}
        if config is not None:
            if hasattr(config, "items"):
                cfg.update(dict(config.items()))
            else:
                cfg.update(dict(config))
        return {
            "debug_safety_feasibility": bool(
                cfg.get(
                    "debug_safety_feasibility",
                    defaults["debug_safety_feasibility"],
                )
            ),
            "jax_batched_optimizer": bool(
                cfg.get("jax_batched_optimizer", defaults.get("jax_batched_optimizer", True))
            ),
            "jax_batched_optimizer_fallback": bool(
                cfg.get(
                    "jax_batched_optimizer_fallback",
                    defaults.get("jax_batched_optimizer_fallback", True),
                )
            ),
        }

    def reset(self):
        self.last_info = {}
        self._deadlock_count = 0
        self._trigger_count = 0
        self.recovery_context = None
        self._pending_committed_replan_info = None
        self.unsafe_streak = 0
        self.brake_streak = 0
        self.recovery_failure_streak = 0
        self.recovery_failure_streak_max = 0
        self.recovery_optimizer_cooldown_remaining = 0
        self.recovery_attempts_in_unsafe_streak = 0
        self.recovery_optimization_skipped_count = 0
        self.committed_suffix_replan_attempt_count = 0
        self.committed_suffix_replan_accepted_count = 0
        self.committed_suffix_replan_rejected_count = 0
        self.committed_suffix_replan_budget_suppressed_count = 0
        self.committed_opportunistic_resume_count = 0
        self.committed_recovery_budget_exit_count = 0
        self.committed_recover_steps_since_act = 0
        self.committed_suffix_replans_in_current_recovery = 0
        self.post_recovery_act_window_active = False
        self.current_deform_plan = None
        self.current_recovery_plan = None
        self.deform_anchor_state = None
        self.recovery_anchor_state = None
        self.failed_recovery_targets = []
        self.failed_recovery_paths = []
        self.recovery_path_failure_streak = 0
        self.recovery_path_failure_streak_max = 0
        self.recovery_target_failure_counts = {}
        self._unsafe_recovery_cooldowns = {}
        self.delayed_rejoin_active = False
        self.delayed_rejoin_steps = 0
        self.safe_corridor_recovery_count = 0
        self.direct_rejoin_attempt_count = 0
        self.direct_rejoin_reject_count = 0
        self.detour_rejoin_attempt_count = 0
        self.detour_rejoin_accept_count = 0
        self.delayed_rejoin_count = 0
        self.recover_path_unsafe_count = 0
        self.post_recovery_act_window_count = 0
        self.repeated_unsafe_target_count = 0
        self.post_recovery_act_window_active = False
        self.post_recovery_act_steps_remaining = 0
        self.post_recovery_act_window_interrupted_count = 0
        self._recover_path_min_clearance_history = []
        self.deform_replan_count = 0
        self.recovery_replan_count = 0
        self.stale_recovery_suppressed_count = 0
        self.recovery_target_infeasible_count = 0
        self.emergency_brake_steps = 0
        self.optimized_candidate_count = 0
        self.optimized_solution_count = 0
        self.fallback_candidate_count = 0
        self.fallback_candidate_accepted_count = 0
        self.optimized_rejected_count = 0
        self.deform_candidate_count = 0
        self.deform_accepted_count = 0
        self.deform_rejected_count = 0
        self.recover_candidate_count = 0
        self.recover_accepted_count = 0
        self.recover_rejected_count = 0
        self.safe_prefix_accepted_count = 0
        self.first_action_only_accepted_count = 0
        self.immediate_hard_reject_count = 0
        self.no_safe_prefix_reject_count = 0
        self.horizon_margin_reject_count = 0
        self.accepted_deform_steps = 0
        self.accepted_recover_steps = 0
        self.fallback_brake_after_reject_count = 0
        self.latest_nominal_chunk = None
        self.latest_nominal_step = 0
        self.blocked_nominal_chunk = None
        self.blocked_nominal_step = None
        self.recover_step_since_deform = 0
        self.nominal_rejoin_available_count = 0
        self.nominal_rejoin_suppressed_count = 0
        self.stale_nominal_rejoin_suppressed_count = 0
        self.nominal_prefix_unsafe_suppressed_count = 0
        self.recover_positive_projection_count = 0
        self.recover_nonpositive_projection_count = 0
        self._recover_projection_history = []
        self._recover_cosine_history = []
        self._recover_task_progress_history = []
        self._recover_ordered_pose_loss_history = []
        self._recover_ordered_delta_loss_history = []
        self._recover_ordered_loss_history = []
        self.last_safe_action = None
        self.last_safe_q = None
        self.last_safe_chunk = None
        self.emergency_deform_away_steps = 0
        self.emergency_deform_away_count = 0
        self.contact_during_hold_count = 0
        self.contact_during_brake_count = 0
        self.contact_during_deform_count = 0
        self.contact_during_recover_count = 0
        self.hold_unsafe_count = 0
        self.hold_predicted_contact_count = 0
        self._hold_horizon_min_clearance_history = []
        self._previous_human_snapshot = None
        self._temporary_progress_history = []
        self._clear_committed_chunk()

    def _clear_committed_chunk(self):
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

    def _jsonable_snapshot(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return {str(k): self._jsonable_snapshot(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable_snapshot(v) for v in value]
        try:
            arr = np.asarray(value)
            if arr.ndim > 0:
                return arr.astype(float).tolist()
            if np.issubdtype(arr.dtype, np.number):
                return float(arr)
        except Exception:  # noqa: BLE001
            pass
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return str(value)

    def _snapshot_human_state(self, human_state):
        return self._jsonable_snapshot(human_state)

    def _human_motion_since_plan(self, human_state):
        before = self.committed_accepted_human_state_snapshot
        after = self._snapshot_human_state(human_state)
        if before is None or after is None:
            return None
        try:
            a = np.asarray(after, dtype=np.float32).reshape(-1)
            b = np.asarray(before, dtype=np.float32).reshape(-1)
            if a.shape != b.shape:
                return None
            return float(np.linalg.norm(a - b))
        except Exception:  # noqa: BLE001
            return None

    def _rollout_chunk_from_q(self, q, action_chunk) -> np.ndarray:
        chunk, _ = self._as_chunk(action_chunk)
        q = np.asarray(q, dtype=np.float32).reshape(-1).copy()
        q_seq = np.zeros((chunk.shape[0], q.shape[0]), dtype=np.float32)

        valid = (
            (self.controlled_state_indices < q.shape[0])
            & (self.controlled_action_indices < chunk.shape[1])
        )
        state_idx = self.controlled_state_indices[valid]
        action_idx = self.controlled_action_indices[valid]

        for k, action in enumerate(chunk):
            q_next = self._apply_controlled_action_step(q, action, state_idx, action_idx)
            q_seq[k] = q_next
            q = q_next
        return q_seq

    def _rollout_one_step_from_q(self, q, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        return self._rollout_chunk_from_q(q, action)[0]

    def _commit_explicit_recovery_chunk(self, obs, chunk, info, **kwargs):
        reject_info = {}
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
        yield_len = int(info.get("deform_chunk_length", info.get("yield_chunk_length", min(self.yield_horizon, total))))
        return_len = int(info.get("recover_chunk_length", info.get("return_chunk_length", min(self.return_horizon, max(0, total - yield_len)))))
        yield_len = max(0, min(yield_len, total))
        return_len = max(0, min(return_len, total - yield_len))
        if info.get("recovery_candidate_class") != "committed_suffix_replan":
            self.committed_suffix_replans_in_current_recovery = 0
        modes = ["horizon_deform"] * yield_len
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
            "candidate_delta_norm",
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
            "committed_suffix_replan_seed_start_index",
        )
        self.committed_rejoin_diagnostics = {
            key: info.get(key) for key in rejoin_keys if info.get(key) is not None
        }
        return True, {}

    def _committed_action_safety(self, obs, pre_q, action, **kwargs):
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

    def _committed_abort_threshold(self):
        if self.committed_abort_only_if_contact_risk:
            return float(self.committed_min_clearance_for_abort)
        return float(self.min_clearance - self.committed_safety_tol)

    def _committed_planned_value(self, arr, idx):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 0 or arr.shape[0] == 0:
            return None
        return arr[min(max(int(idx), 0), arr.shape[0] - 1)]

    def _pop_pending_committed_replan_info(self):
        info = self._pending_committed_replan_info
        self._pending_committed_replan_info = None
        return info

    def _current_replay_q(self, obs, **kwargs):
        q_actual = kwargs.get("q_full")
        if q_actual is None:
            q_actual = self.extract_current_q(obs)
        q_actual = np.asarray(q_actual, dtype=np.float32).reshape(-1)
        if q_actual.shape[0] >= self.expected_motion_dim:
            return q_actual[: self.expected_motion_dim].copy()
        padded = np.zeros(self.expected_motion_dim, dtype=np.float32)
        padded[: q_actual.shape[0]] = q_actual
        return padded

    def _committed_state_diagnostics(self, idx, obs, **kwargs):
        planned_q = self._committed_planned_value(self.committed_planned_q_seq, idx)
        q_actual = self._current_replay_q(obs, **kwargs)
        state_error = None
        if planned_q is not None:
            try:
                planned_q_arr = np.asarray(planned_q, dtype=np.float32).reshape(-1)
                valid = (
                    (self.controlled_state_indices < q_actual.shape[0])
                    & (self.controlled_state_indices < planned_q_arr.shape[0])
                )
                if np.any(valid):
                    state_idx = self.controlled_state_indices[valid]
                    state_error = float(
                        np.linalg.norm(q_actual[state_idx] - planned_q_arr[state_idx])
                    )
            except Exception:  # noqa: BLE001
                state_error = None
        missing_planned_q = planned_q is None or state_error is None
        return {
            "committed_state_error": state_error,
            "committed_state_error_threshold": float(
                self.committed_state_error_threshold
            ),
            "committed_aborted_due_to_state_mismatch": False,
            "committed_replan_due_to_state_mismatch": False,
            "committed_rejected_missing_planned_q": bool(missing_planned_q),
            "planned_q_at_index": self._jsonable_snapshot(planned_q),
            "actual_q_at_replay": self._jsonable_snapshot(q_actual),
            "planned_vs_actual_q_error": state_error,
        }

    def _committed_state_mismatch_info(self, mode, idx, length, state_info):
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
                "yield_steps_executed": 0,
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
        obs,
        nominal_chunk,
        original_shape,
        mode,
        idx,
        total,
        state_info,
    ):
        brake_safety = {
            "horizon_safe": False,
            "min_clearance": float("inf"),
            "first_violation": 0,
            "unsafe_count": 0,
            "safety_eval_available": False,
        }
        braked_chunk, brake_info = self.horizon_brake(
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
        obs,
        nominal_chunk,
        original_shape,
        mode,
        idx,
        total,
        state_info,
        **kwargs,
    ):
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
        if (
            self.max_suffix_replans_per_recovery > 0
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

        def reject(reason, **extra):
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
        h = min(remaining, self.return_horizon, nominal.shape[0])
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

        return_nominal, seed_target_index = self._make_task_progress_recover_chunk(
            current_q,
            old_suffix,
            action_idx,
            context=context,
            default_target_index=seed_start_index,
        )
        if return_nominal.shape[0] == 0:
            return reject("empty_return_seed")

        return_obs = self._obs_with_q(obs, current_q)
        return_rejoin_context = self._make_rejoin_context(
            context.nominal_q_seq,
            context.nominal_ee_seq,
        )
        use_recover_cost = bool(self.safechunk_recover_enabled)

        def return_cost(candidate):
            if use_recover_cost:
                return self._recover_task_progress_cost(
                    return_obs,
                    candidate,
                    return_nominal,
                    action_idx,
                    reference_chunk=return_nominal,
                )
            return self._return_deformation_cost(
                return_obs,
                candidate,
                return_nominal,
                context.nominal_q_seq,
                return_rejoin_context,
                action_idx,
            )

        def return_batch_cost(candidates):
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
            if use_recover_cost:
                if not bool(losses.get("recover_direction_ok", True)):
                    return False
                if not bool(losses.get("recover_ordered_ok", True)):
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
            batch_cost_fn=return_batch_cost,
            early_stop_fn=return_early_stop,
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
        reject_reason = self._recovery_reject_reason(
            terminal,
            path_info,
            direction_ok=bool(direction_terms["recover_direction_ok"]),
            ordered_ok=bool(terminal.get("recover_ordered_ok", True)),
        )
        required_margin = float(self.min_clearance + self.committed_execution_margin)
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
                "qd_rejoin_ok": bool(terminal.get("qd_rejoin_ok", False)),
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
                    "recover_ordered_loss",
                    "recover_ordered_pose_weight",
                    "recover_ordered_delta_weight",
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

    def _committed_replay_diagnostics(self, idx, action, safety_info, **kwargs):
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

        def _norm_error(a, b):
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

        def _float_or_none(value):
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

        def _finite_float(value):
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

    def _committed_abort_reason(self, diagnostics, min_clearance):
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

    def _repair_committed_action(self, obs, action, **kwargs):
        try:
            repaired = self._call_single_step_operator(action, obs, **kwargs)
            return np.asarray(repaired, dtype=np.float32).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Committed action repair failed: %s", exc)
            return np.asarray(action, dtype=np.float32).reshape(-1)

    def _committed_recovery_phase(self, mode):
        if mode == "recover":
            return "recover"
        if mode == "pass_through":
            return "pass_through"
        return "horizon_deform"

    def _committed_info(
        self,
        safety_info,
        mode,
        index,
        length,
        completed=False,
        aborted=False,
        repaired=False,
        repair_info=None,
        extra=None,
    ):
        info = dict(safety_info or {})
        rejoin_index = self.committed_rejoin_index
        recovery_phase = self._committed_recovery_phase(mode)
        if completed and rejoin_index is not None and not aborted:
            self._activate_post_recovery_act_window()
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
                "yield_steps_executed": int(mode == "horizon_deform" and not aborted),
                "resume_from_committed_rejoin": bool(
                    completed and rejoin_index is not None and not aborted
                ),
                "request_action_history_reset_after_recovery": bool(
                    completed and rejoin_index is not None and not aborted
                ),
                "suppress_outer_pause": not bool(aborted),
                "act_resume_index": (
                    None
                    if not (completed and rejoin_index is not None and not aborted)
                    else int(rejoin_index)
                ),
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
        info.setdefault(
            "post_recovery_act_window_interrupted",
            bool(aborted),
        )
        if extra:
            info.update(extra)
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
        obs,
        nominal_chunk,
        original_shape,
        mode,
        idx,
        total,
        state_info,
        **kwargs,
    ):
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
        state_info.update(
            {
                "committed_opportunistic_resume_available": bool(
                    close_enough and nominal_step_safe
                ),
                "committed_opportunistic_resume_close_enough": bool(close_enough),
                "committed_opportunistic_resume_nominal_step_safe": bool(
                    nominal_step_safe
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
        if not (close_enough and nominal_step_safe):
            state_info["committed_opportunistic_resume_reason"] = (
                "q_rejoin_not_close" if not close_enough else "nominal_step_unsafe"
            )
            return None

        self.committed_opportunistic_resume_count += 1
        self.committed_recover_steps_since_act = 0
        self.committed_suffix_replans_in_current_recovery = 0
        self._activate_post_recovery_act_window()
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
                "committed_released_for_act_resume": True,
                "committed_opportunistic_resume": True,
                "committed_recovery_budget_exit": False,
                "resume_from_committed_rejoin": True,
                "request_action_history_reset_after_recovery": True,
                "act_resume_index": None if target_index is None else int(target_index),
                "act_resume_supported": False,
                "recover_steps_executed": 0,
                "return_steps_executed": 0,
                "yield_steps_executed": 0,
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

    def _serve_committed_chunk(self, obs, nominal_chunk, original_shape, **kwargs):
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
        resume_result = self._try_resume_act_from_committed_recovery(
            obs,
            nominal_chunk,
            original_shape,
            mode,
            idx,
            total,
            state_diagnostics,
            **kwargs,
        )
        if resume_result is not None:
            return resume_result
        if budget_exceeded:
            self.committed_recovery_budget_exit_count += 1
            budget_info = dict(state_diagnostics)
            budget_info.update(
                {
                    "committed_chunk_active": True,
                    "committed_chunk_mode": mode,
                    "committed_chunk_index": int(idx),
                    "committed_chunk_length": int(total),
                    "committed_recovery_budget_exit": True,
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
            if execution_min_clearance < abort_threshold:
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
                braked_chunk, brake_info = self.horizon_brake(
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
                            braked_chunk, brake_info = self.horizon_brake(
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
        if completed:
            self._clear_committed_chunk()
        self.last_info = info
        return served.reshape(original_shape), info

    def _as_chunk(self, action) -> tuple[np.ndarray, bool]:
        arr = np.asarray(action)
        if arr.ndim == 1:
            if arr.shape[0] != self.action_dim:
                logger.debug(
                    "Single action dim %d differs from configured action_dim %d",
                    arr.shape[0],
                    self.action_dim,
                )
            return arr.reshape(1, -1).copy(), True
        if arr.ndim == 2:
            return arr.copy(), False
        raise ValueError(
            f"Expected action shape ({self.action_dim},) or (H, {self.action_dim}), "
            f"got {arr.shape}"
        )

    def extract_current_q(self, obs, action_chunk: Optional[np.ndarray] = None) -> np.ndarray:
        candidates = ("q", "qpos", "robot_state", "state")
        value = None
        if isinstance(obs, dict):
            for name in candidates:
                if name in obs:
                    value = obs[name]
                    break
        else:
            for name in candidates:
                if hasattr(obs, name):
                    value = getattr(obs, name)
                    break

        if value is not None:
            q = np.asarray(value, dtype=np.float32).reshape(-1)
            if q.shape[0] >= self.expected_motion_dim:
                return q[: self.expected_motion_dim].copy()
            padded = np.zeros(self.expected_motion_dim, dtype=np.float32)
            padded[: q.shape[0]] = q
            return padded

        q = np.zeros(self.expected_motion_dim, dtype=np.float32)
        if action_chunk is not None and action_chunk.size > 0:
            valid = self.controlled_state_indices < q.shape[0]
            state_idx = self.controlled_state_indices[valid]
            action_idx = self.controlled_action_indices[valid]
            modes = self._control_mode_ids_for_state_indices(state_idx)
            absolute = modes == 0
            if np.any(absolute):
                q[state_idx[absolute]] = action_chunk[0][action_idx[absolute]]
        return q

    def rollout_nominal_chunk(self, obs, action_chunk) -> np.ndarray:
        chunk, _ = self._as_chunk(action_chunk)
        q = self.extract_current_q(obs, chunk)
        q_seq = np.zeros((chunk.shape[0], q.shape[0]), dtype=np.float32)

        valid = (
            (self.controlled_state_indices < q.shape[0])
            & (self.controlled_action_indices < chunk.shape[1])
        )
        state_idx = self.controlled_state_indices[valid]
        action_idx = self.controlled_action_indices[valid]

        for k, action in enumerate(chunk):
            q_next = self._apply_controlled_action_step(q, action, state_idx, action_idx)
            q_seq[k] = q_next
            q = q_next

        return q_seq

    def _control_mode_id(self):
        if self.control_type == "absolute":
            return 0
        if self.control_type == "delta":
            return 1
        return 2

    def _control_mode_ids_for_state_indices(self, state_idx):
        state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
        modes = np.full(state_idx.shape, self._control_mode_id(), dtype=np.int32)
        # BiGym floating-base action dimensions are command deltas, even when
        # the arm policy outputs absolute joint targets.
        modes[state_idx < min(4, self.expected_motion_dim)] = 1
        return modes

    def _apply_controlled_action_step(self, q, action, state_idx, action_idx):
        q_next = np.asarray(q, dtype=np.float32).reshape(-1).copy()
        if len(action_idx) == 0:
            return q_next
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        modes = self._control_mode_ids_for_state_indices(state_idx)
        selected = action[action_idx]
        current = q_next[state_idx]
        updated = selected.copy()
        delta_mask = modes == 1
        velocity_mask = modes == 2
        updated[delta_mask] = current[delta_mask] + selected[delta_mask]
        updated[velocity_mask] = current[velocity_mask] + self.dt * selected[velocity_mask]
        q_next[state_idx] = updated
        return q_next

    def _apply_controlled_action_step_batch(self, q, actions, state_idx, action_idx):
        q_next = np.asarray(q, dtype=np.float32).copy()
        if len(action_idx) == 0:
            return q_next
        actions = np.asarray(actions, dtype=np.float32)
        modes = self._control_mode_ids_for_state_indices(state_idx)
        selected = actions[:, action_idx]
        current = q_next[:, state_idx]
        updated = selected.copy()
        delta_mask = modes == 1
        velocity_mask = modes == 2
        updated[:, delta_mask] = current[:, delta_mask] + selected[:, delta_mask]
        updated[:, velocity_mask] = current[:, velocity_mask] + self.dt * selected[:, velocity_mask]
        q_next[:, state_idx] = updated
        return q_next

    def _jax_optimizer_ready(self):
        if not self.jax_batched_optimizer:
            return False
        if not _JAX_AVAILABLE:
            if not self._warned_jax_unavailable:
                logger.warning("JAX batched optimizer requested but JAX is unavailable; using NumPy optimizer path.")
                self._warned_jax_unavailable = True
            return False
        return True

    def _make_optimizer_warmup_chunk(self, obs, horizon):
        horizon = max(1, int(horizon))
        chunk = np.zeros((horizon, self.action_dim), dtype=np.float32)
        try:
            q = self.extract_current_q(obs, chunk)
            valid = (
                (self.controlled_state_indices < q.shape[0])
                & (self.controlled_action_indices < chunk.shape[1])
            )
            if np.any(valid):
                state_idx = self.controlled_state_indices[valid]
                action_idx = self.controlled_action_indices[valid]
                modes = self._control_mode_ids_for_state_indices(state_idx)
                absolute = modes == 0
                if np.any(absolute):
                    chunk[:, action_idx[absolute]] = q[state_idx[absolute]][None, :]
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk optimizer warmup q-seeded chunk failed: %s", exc)
        return chunk

    def _warmup_optimizer_shape(self, obs, horizon):
        if not self._jax_optimizer_ready():
            return {"compiled": False, "reason": "jax_unavailable"}
        horizon = max(1, int(horizon))
        valid_key = self._valid_control_indices(np.zeros((horizon, self.action_dim), dtype=np.float32))
        mode_key = tuple(
            self._control_mode_ids_for_state_indices(
                self.controlled_state_indices[valid_key]
            ).tolist()
        )
        key = (int(self.opt_population), horizon, int(self.action_dim), mode_key)
        if key in self._optimizer_warmup_cache:
            return {"compiled": False, "reason": "already_warmed", "key": key}

        nominal = self._make_optimizer_warmup_chunk(obs, horizon)
        valid = self._valid_control_indices(nominal)
        if not np.any(valid):
            return {"compiled": False, "reason": "no_control_indices", "key": key}
        action_idx = self.controlled_action_indices[valid]
        ctrl = np.broadcast_to(
            nominal[None, :, action_idx],
            (int(self.opt_population), horizon, len(action_idx)),
        ).copy()
        t0 = time.perf_counter()
        candidates = self._jax_project_candidate_population(nominal, ctrl, action_idx)
        if candidates is None:
            return {"compiled": False, "reason": "projection_fallback", "key": key}
        q_seq_batch = self.rollout_nominal_chunk_batch(obs, candidates)
        safety_eval = self.evaluate_horizon_safety_batch(obs, q_seq_batch)
        _ = self._clearance_sequence_batch_from_eval(
            safety_eval,
            candidates.shape[0],
            candidates.shape[1],
        )
        nominal_q_seq = self.rollout_nominal_chunk(obs, nominal)
        rejoin_context = self._make_rejoin_context(nominal_q_seq, None)
        _ = self._q_rejoin_loss_batch(
            q_seq_batch,
            nominal_q_seq=nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        try:
            self._yield_deformation_cost_batch(obs, candidates, nominal, action_idx)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk warmup yield cost skipped: %s", exc)
        try:
            self._return_deformation_cost_batch(
                obs,
                candidates,
                nominal,
                nominal_q_seq,
                rejoin_context,
                action_idx,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk warmup return cost skipped: %s", exc)
        try:
            self._recover_task_progress_cost_batch(obs, candidates, nominal, action_idx)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk warmup task-progress recover cost skipped: %s", exc)
        try:
            context = RecoveryContext(
                nominal_chunk=nominal.copy(),
                nominal_q_seq=nominal_q_seq.copy(),
                nominal_ee_seq=None,
            )
            self._recovery_terminal_rejoin_info(
                obs,
                nominal,
                context,
                rejoin_context,
                default_target_index=min(horizon - 1, self.min_rejoin_offset),
            )
            self.evaluate_recovery_path_safety(obs, nominal, candidate_name="warmup")
            self.evaluate_candidate_acceptance(obs, nominal, candidate_type="recover")
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk warmup recovery checks skipped: %s", exc)
        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        self._optimizer_warmup_cache.add(key)
        return {"compiled": True, "key": key, "time_ms": float(elapsed_ms)}

    def _warmup_optimizer_live_path(self, obs, nominal_chunk=None):
        if not (self.recoverable_deform_enabled and self.explicit_return):
            return {"compiled": False, "reason": "explicit_return_disabled"}
        t0 = time.perf_counter()
        rng_state = None
        try:
            rng_state = self._rng.bit_generator.state
        except Exception:  # noqa: BLE001
            rng_state = None
        try:
            self.reset()
            if nominal_chunk is None:
                nominal = self._make_optimizer_warmup_chunk(obs, self.horizon)
            else:
                nominal, _ = self._as_chunk(nominal_chunk)
                nominal = np.asarray(nominal, dtype=np.float32).copy()
            nominal_q_seq = self.rollout_nominal_chunk(obs, nominal)
            safety_info = self.evaluate_horizon_safety(obs, nominal_q_seq)
            _, info = self.deform_chunk_optimized(
                nominal,
                obs=obs,
                nominal_q_seq=nominal_q_seq,
                safety_info=safety_info,
            )
            return {
                "compiled": True,
                "path": "explicit_return_optimizer",
                "time_ms": float(1000.0 * (time.perf_counter() - t0)),
                "optimized_accepted": bool(info.get("optimized_accepted", False)),
                "yield_cem_iterations_run": info.get("yield_cem_iterations_run"),
                "return_cem_iterations_run": info.get("return_cem_iterations_run"),
                "cem_early_stopped": info.get("cem_early_stopped"),
                "min_clearance": info.get("min_clearance"),
                "recover_min_clearance": info.get("recover_min_clearance"),
                "rejection_cause": info.get("rejection_cause"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk live-path optimizer warmup failed: %s", exc)
            return {
                "compiled": False,
                "path": "explicit_return_optimizer",
                "reason": str(exc),
                "time_ms": float(1000.0 * (time.perf_counter() - t0)),
            }
        finally:
            self.reset()
            if rng_state is not None:
                try:
                    self._rng.bit_generator.state = rng_state
                except Exception:  # noqa: BLE001
                    pass

    def warmup_optimizer(self, obs, nominal_chunk=None, *, force=False):
        if self._optimizer_warmup_done and nominal_chunk is None and not force:
            return dict(self._optimizer_warmup_info)
        t0 = time.perf_counter()
        horizons = [self.horizon]
        if self.recoverable_deform_enabled and self.explicit_return:
            horizons.extend([self.yield_horizon, self.return_horizon])
        seen = []
        results = []
        for horizon in horizons:
            horizon = int(horizon)
            if horizon in seen:
                continue
            seen.append(horizon)
            try:
                results.append(self._warmup_optimizer_shape(obs, horizon))
            except Exception as exc:  # noqa: BLE001
                logger.debug("SafeChunk optimizer warmup failed for horizon %s: %s", horizon, exc)
                results.append({"compiled": False, "horizon": horizon, "reason": str(exc)})
        live_path_result = self._warmup_optimizer_live_path(obs, nominal_chunk=nominal_chunk)
        info = {
            "optimizer_warmup_enabled": True,
            "optimizer_warmup_done": True,
            "optimizer_warmup_time_ms": float(1000.0 * (time.perf_counter() - t0)),
            "optimizer_warmup_results": results,
            "optimizer_warmup_live_path_result": live_path_result,
        }
        if nominal_chunk is None:
            self._optimizer_warmup_done = True
            self._optimizer_warmup_info = dict(info)
        return info

    def _jax_project_candidate_population(self, nominal, ctrl_samples, action_idx):
        if not self._jax_optimizer_ready() or len(action_idx) == 0:
            return None
        try:
            nominal_arr = np.asarray(nominal, dtype=np.float32)
            ctrl_arr = np.asarray(ctrl_samples, dtype=np.float32)
            action_idx_arr = np.asarray(action_idx, dtype=np.int32)
            max_delta = np.inf if self.max_action_delta is None else float(self.max_action_delta)
            low = -np.inf if self.action_low is None else float(self.action_low)
            high = np.inf if self.action_high is None else float(self.action_high)
            return np.asarray(
                _jax_project_candidate_population(
                    jnp.asarray(nominal_arr),
                    jnp.asarray(ctrl_arr),
                    jnp.asarray(action_idx_arr),
                    jnp.asarray(max_delta, dtype=jnp.float32),
                    jnp.asarray(low, dtype=jnp.float32),
                    jnp.asarray(high, dtype=jnp.float32),
                ),
                dtype=np.float32,
            )
        except Exception as exc:  # noqa: BLE001
            if not self.jax_batched_optimizer_fallback:
                raise
            logger.debug("JAX candidate projection failed; using NumPy path: %s", exc)
            return None

    def _jax_rollout_nominal_chunk_batch(self, obs, action_chunks):
        if not self._jax_optimizer_ready():
            return None
        try:
            chunks = np.asarray(action_chunks, dtype=np.float32)
            if chunks.ndim == 2:
                chunks = chunks[None, :, :]
            if chunks.ndim != 3:
                return None
            q0 = self.extract_current_q(obs, chunks[0] if chunks.shape[0] else None)
            valid = (
                (self.controlled_state_indices < q0.shape[0])
                & (self.controlled_action_indices < chunks.shape[2])
            )
            state_idx = self.controlled_state_indices[valid].astype(np.int32)
            action_idx = self.controlled_action_indices[valid].astype(np.int32)
            if state_idx.size == 0 or action_idx.size == 0:
                return None
            return np.asarray(
                _jax_rollout_chunk_population(
                    jnp.asarray(chunks),
                    jnp.asarray(q0, dtype=jnp.float32),
                    jnp.asarray(state_idx, dtype=jnp.int32),
                    jnp.asarray(action_idx, dtype=jnp.int32),
                    jnp.asarray(
                        self._control_mode_ids_for_state_indices(state_idx),
                        dtype=jnp.int32,
                    ),
                    jnp.asarray(float(self.dt), dtype=jnp.float32),
                ),
                dtype=np.float32,
            )
        except Exception as exc:  # noqa: BLE001
            if not self.jax_batched_optimizer_fallback:
                raise
            logger.debug("JAX batched rollout failed; using NumPy path: %s", exc)
            return None

    def rollout_nominal_chunk_batch(self, obs, action_chunks) -> np.ndarray:
        chunks = np.asarray(action_chunks, dtype=np.float32)
        if chunks.ndim == 2:
            chunks = chunks[None, :, :]
        if chunks.ndim != 3:
            raise ValueError(
                "Expected action_chunks with shape (B, H, A), "
                f"got {chunks.shape}"
            )
        jax_q_seq = self._jax_rollout_nominal_chunk_batch(obs, chunks)
        if jax_q_seq is not None:
            return jax_q_seq

        batch, horizon = chunks.shape[:2]
        q0 = self.extract_current_q(obs, chunks[0] if batch else None)
        q = np.broadcast_to(q0[None, :], (batch, q0.shape[0])).copy()
        q_seq = np.zeros((batch, horizon, q0.shape[0]), dtype=np.float32)

        valid = (
            (self.controlled_state_indices < q0.shape[0])
            & (self.controlled_action_indices < chunks.shape[2])
        )
        state_idx = self.controlled_state_indices[valid]
        action_idx = self.controlled_action_indices[valid]

        for k in range(horizon):
            q_next = self._apply_controlled_action_step_batch(
                q,
                chunks[:, k, :],
                state_idx,
                action_idx,
            )
            q_seq[:, k, :] = q_next
            q = q_next

        return q_seq

    def evaluate_horizon_safety_batch(self, obs, q_seq_batch) -> dict[str, Any]:
        q_seq_batch = np.asarray(q_seq_batch, dtype=np.float32)
        if q_seq_batch.ndim == 2:
            q_seq_batch = q_seq_batch[None, :, :]
        if q_seq_batch.ndim != 3:
            raise ValueError(
                "Expected q_seq_batch with shape (B, H, Q), "
                f"got {q_seq_batch.shape}"
            )
        op = self._get_oscbf_operator()
        for method_name in (
            "evaluate_safety_batch",
            "compute_min_clearance_batch",
            "get_min_clearance_batch",
        ):
            method = getattr(op, method_name, None)
            if method is None:
                continue
            try:
                result = self._call_safety_batch_method(method, obs, q_seq_batch)
                return self._normalize_safety_batch_result(result, q_seq_batch.shape[0], q_seq_batch.shape[1])
            except Exception as exc:  # pragma: no cover - defensive integration path
                logger.debug(
                    "SafeChunk-Deform batched safety evaluation via %s failed: %s",
                    method_name,
                    exc,
                )

        per_candidate = [
            self.evaluate_horizon_safety(obs, q_seq)
            for q_seq in q_seq_batch
        ]
        min_clearances = np.stack(
            [self._clearance_sequence_from_eval(item, q_seq_batch.shape[1]) for item in per_candidate],
            axis=0,
        ).astype(np.float32)
        unsafe = min_clearances < self.min_clearance
        unsafe_any = np.any(unsafe, axis=1)
        first_violation = np.full(q_seq_batch.shape[0], -1, dtype=np.int32)
        if np.any(unsafe_any):
            first_violation[unsafe_any] = np.argmax(unsafe[unsafe_any], axis=1)
        return {
            "horizon_safe": ~unsafe_any,
            "min_clearance": np.min(min_clearances, axis=1).astype(np.float32),
            "min_clearances": min_clearances,
            "first_violation": first_violation,
            "unsafe_count": np.count_nonzero(unsafe, axis=1).astype(np.int32),
            "safety_eval_available": all(
                bool(item.get("safety_eval_available", True)) for item in per_candidate
            ),
        }

    def _call_safety_batch_method(self, method, obs, q_seq_batch):
        attempts = (
            lambda: method(obs=obs, q_seq_batch=q_seq_batch),
            lambda: method(q_seq_batch=q_seq_batch, obs=obs),
            lambda: method(obs, q_seq_batch),
            lambda: method(q_seq_batch),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Batched safety method could not be called")

    def _normalize_safety_batch_result(self, result, batch: int, horizon: int) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise ValueError("Batched safety result must be a dict")
        min_clearances = np.asarray(
            result.get("min_clearances", result.get("clearances", [])),
            dtype=np.float32,
        )
        if min_clearances.shape != (batch, horizon):
            min_by_candidate = np.asarray(
                result.get("min_clearance", np.inf),
                dtype=np.float32,
            ).reshape(-1)
            if min_by_candidate.size == 1:
                min_by_candidate = np.full(batch, float(min_by_candidate[0]), dtype=np.float32)
            if min_by_candidate.size != batch:
                min_by_candidate = np.full(batch, np.inf, dtype=np.float32)
            min_clearances = np.repeat(min_by_candidate[:, None], horizon, axis=1)
        unsafe = min_clearances < self.min_clearance
        unsafe_any = np.any(unsafe, axis=1)
        horizon_safe = np.asarray(
            result.get("horizon_safe", result.get("safe", ~unsafe_any)),
            dtype=np.bool_,
        ).reshape(-1)
        if horizon_safe.size == 1:
            horizon_safe = np.full(batch, bool(horizon_safe[0]), dtype=np.bool_)
        if horizon_safe.size != batch:
            horizon_safe = ~unsafe_any
        first_violation = result.get("first_violation")
        if first_violation is None:
            first_violation = np.full(batch, -1, dtype=np.int32)
            if np.any(unsafe_any):
                first_violation[unsafe_any] = np.argmax(unsafe[unsafe_any], axis=1)
        else:
            first_violation = np.asarray(first_violation, dtype=np.int32).reshape(-1)
            if first_violation.size == 1:
                first_violation = np.full(batch, int(first_violation[0]), dtype=np.int32)
            if first_violation.size != batch:
                first_violation = np.full(batch, -1, dtype=np.int32)
        info = dict(result)
        info.update(
            {
                "horizon_safe": horizon_safe,
                "min_clearance": np.min(min_clearances, axis=1).astype(np.float32),
                "min_clearances": min_clearances.astype(np.float32),
                "first_violation": first_violation,
                "unsafe_count": np.count_nonzero(unsafe, axis=1).astype(np.int32),
                "safety_eval_available": bool(result.get("safety_eval_available", True)),
            }
        )
        return info

    def evaluate_horizon_safety(self, obs, q_seq) -> dict[str, Any]:
        q_seq = np.asarray(q_seq, dtype=np.float32)
        op = self._get_oscbf_operator()
        methods = (
            "compute_min_clearance",
            "get_min_clearance",
            "evaluate_safety",
            "is_safe",
        )

        for method_name in methods:
            method = getattr(op, method_name, None)
            if method is None:
                continue
            try:
                result = self._call_safety_method(method, obs, q_seq)
                return self._normalize_safety_result(result, q_seq.shape[0])
            except Exception as exc:  # pragma: no cover - defensive integration path
                logger.warning(
                    "SafeChunk-Deform safety evaluation via %s failed: %s",
                    method_name,
                    exc,
                )

        if not self._warned_no_safety_eval:
            logger.warning(
                "SafeChunk-Deform could not find a horizon clearance evaluator; "
                "using conservative pass-through horizon evaluation."
            )
            self._warned_no_safety_eval = True
        h = q_seq.shape[0]
        return {
            "horizon_safe": True,
            "min_clearance": float("inf"),
            "min_clearances": np.full(h, np.inf, dtype=np.float32),
            "first_violation": None,
            "unsafe_count": 0,
            "safety_eval_available": False,
        }

    def _call_safety_method(self, method, obs, q_seq):
        attempts = (
            lambda: method(obs=obs, q_seq=q_seq),
            lambda: method(q_seq=q_seq, obs=obs),
            lambda: method(obs, q_seq),
            lambda: method(q_seq),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        raise last_error

    def _normalize_safety_result(self, result, horizon: int) -> dict[str, Any]:
        if isinstance(result, dict):
            min_clearances = np.asarray(
                result.get("min_clearances", result.get("clearances", [])),
                dtype=np.float32,
            ).reshape(-1)
            if min_clearances.size == 0:
                min_clearance = float(result.get("min_clearance", np.inf))
                min_clearances = np.full(horizon, min_clearance, dtype=np.float32)
            horizon_safe = bool(
                result.get(
                    "horizon_safe",
                    result.get("safe", np.all(min_clearances >= self.min_clearance)),
                )
            )
            first_violation = result.get("first_violation")
            if first_violation is None:
                unsafe = np.flatnonzero(min_clearances < self.min_clearance)
                first_violation = int(unsafe[0]) if unsafe.size else None
            unsafe_count = int(
                result.get(
                    "unsafe_count",
                    np.count_nonzero(min_clearances < self.min_clearance),
                )
            )
            info = {
                "horizon_safe": horizon_safe,
                "min_clearance": float(result.get("min_clearance", np.min(min_clearances))),
                "min_clearances": min_clearances,
                "first_violation": first_violation,
                "unsafe_count": unsafe_count,
                "safety_eval_available": True,
            }
            for key, value in result.items():
                if key not in info and key not in {"clearances"}:
                    info[key] = value
            return info

        arr = np.asarray(result)
        if arr.dtype == np.bool_:
            horizon_safe = bool(arr.all())
            min_clearances = np.full(horizon, np.inf if horizon_safe else -np.inf)
        else:
            min_clearances = arr.astype(np.float32).reshape(-1)
            if min_clearances.size == 1 and horizon > 1:
                min_clearances = np.full(horizon, float(min_clearances[0]), dtype=np.float32)
            horizon_safe = bool(np.all(min_clearances >= self.min_clearance))
        unsafe = np.flatnonzero(min_clearances < self.min_clearance)
        return {
            "horizon_safe": horizon_safe,
            "min_clearance": float(np.min(min_clearances)),
            "min_clearances": min_clearances,
            "first_violation": int(unsafe[0]) if unsafe.size else None,
            "unsafe_count": int(unsafe.size),
            "safety_eval_available": True,
        }

    def horizon_brake(self, obs, action_chunk, safety_info):
        chunk, _ = self._as_chunk(action_chunk)
        first_violation = safety_info.get("first_violation")
        if first_violation is None:
            q_seq = self.rollout_nominal_chunk(obs, chunk)
            brake_safety = self.evaluate_horizon_safety(obs, q_seq)
            return chunk, {
                "brake_safe": bool(brake_safety["horizon_safe"]),
                "deadlock": False,
                "progress_scale": 1.0,
                "brake_stop_idx": None,
                "brake_min_clearance": float(brake_safety["min_clearance"]),
            }

        stop_idx = max(0, int(first_violation) - 1)
        stop_idx = min(stop_idx, chunk.shape[0] - 1)
        braked = chunk.copy()
        valid = self._valid_control_indices(chunk)
        action_idx = self.controlled_action_indices[valid]
        if action_idx.size:
            if stop_idx == 0:
                state_idx = self.controlled_state_indices[valid]
                anchor = self._controlled_anchor(obs, chunk, action_idx, state_idx)
                braked[:, action_idx] = anchor
            else:
                braked[stop_idx:, action_idx] = chunk[stop_idx, action_idx]

        q_seq = self.rollout_nominal_chunk(obs, braked)
        brake_safety = self.evaluate_horizon_safety(obs, q_seq)
        progress_scale = stop_idx / max(1, chunk.shape[0] - 1)
        deadlock = progress_scale < self.brake_progress_threshold
        return braked, {
            "brake_safe": bool(brake_safety["horizon_safe"]),
            "deadlock": bool(deadlock),
            "progress_scale": float(progress_scale),
            "brake_stop_idx": int(stop_idx),
            "brake_min_clearance": float(brake_safety["min_clearance"]),
            "brake_hold_current": bool(stop_idx == 0),
        }

    def deform_chunk(self, obs, action_chunk, safety_info=None, braked_chunk=None, **kwargs):
        if self.mode == "optimized":
            return self._deform_chunk_optimized_with_fallback(
                obs,
                action_chunk,
                safety_info=safety_info,
                braked_chunk=braked_chunk,
                **kwargs,
            )
        return self.deform_chunk_candidate(
            obs,
            action_chunk,
            safety_info=safety_info,
            **kwargs,
        )

    def deform_chunk_candidate(self, obs, action_chunk, safety_info=None, **kwargs):
        """Deform the controlled chunk trajectory using fixed suffix scales.

        This is the original SafeChunk-Deform candidate search. It is intentionally
        derivative-free: with only scalar clearance feedback available, it generates
        smooth whole-chunk candidates by retracting the unsafe suffix toward a
        no-motion anchor and selects the first safe, least distorted candidate.
        """
        chunk, _ = self._as_chunk(action_chunk)
        safety_info = safety_info or {}
        candidates = self._make_chunk_deformation_candidates(obs, chunk, safety_info)

        best_chunk = chunk.copy()
        best_eval = None
        best_norm = float("inf")
        best_scale = None
        best_safe = False

        for scale, candidate in candidates:
            q_seq = self.rollout_nominal_chunk(obs, candidate)
            candidate_eval = self.evaluate_horizon_safety(obs, q_seq)
            candidate_norm = self._controlled_deformation_norm(candidate, chunk)
            candidate_progress = self._controlled_progress_retention(
                candidate, chunk, obs
            )
            candidate_safe = bool(candidate_eval["horizon_safe"])
            candidate_eval["task_progress_retention"] = candidate_progress

            if self._is_better_deformation_candidate(
                candidate_eval,
                candidate_norm,
                candidate_safe,
                best_eval,
                best_norm,
                best_safe,
            ):
                best_chunk = candidate
                best_eval = candidate_eval
                best_norm = candidate_norm
                best_scale = scale
                best_safe = candidate_safe

        if best_eval is None:
            q_seq = self.rollout_nominal_chunk(obs, best_chunk)
            best_eval = self.evaluate_horizon_safety(obs, q_seq)
            best_norm = self._controlled_deformation_norm(best_chunk, chunk)
            best_eval["task_progress_retention"] = self._controlled_progress_retention(
                best_chunk, chunk, obs
            )

        info = {
            "deform_mode": "candidate",
            "deform_safe": bool(best_eval["horizon_safe"]),
            "deform_min_clearance": float(best_eval["min_clearance"]),
            "deformation_norm": float(best_norm),
            "deformation_source": "chunk_deform",
            "chunk_deform_scale": best_scale,
            "chunk_deform_attempts": len(candidates),
            "task_progress_retention": float(
                best_eval.get("task_progress_retention", 1.0)
            ),
        }

        if (
            not info["deform_safe"]
            and self.sequential_oscbf_fallback
            and callable(self._get_oscbf_operator())
        ):
            fallback_chunk, fallback_info = self.deform_chunk_with_oscbf(
                obs, chunk, **kwargs
            )
            fallback_info["deform_mode"] = "candidate"
            fallback_info["deformation_source"] = "sequential_oscbf_fallback"
            return fallback_chunk, fallback_info

        return best_chunk, info

    def _rolling_prefix_candidate_fallback(
        self,
        obs,
        chunk,
        optimized_chunk,
        optimized_info,
        safety_info=None,
        braked_chunk=None,
        candidate_type="deform",
        **kwargs,
    ):
        valid = self._valid_control_indices(chunk)
        action_idx = self.controlled_action_indices[valid]
        candidates = []

        def add_candidate(name, candidate):
            if candidate is None:
                return
            arr, _ = self._as_chunk(candidate)
            if arr.shape == chunk.shape:
                candidates.append((name, arr.copy()))

        add_candidate("optimized", optimized_chunk)
        add_candidate("nominal", chunk)
        add_candidate("horizon_brake", braked_chunk)
        for scale, candidate in self._make_chunk_deformation_candidates(obs, chunk, safety_info or {}):
            add_candidate(f"scaled_deform_{scale}", candidate)
        if callable(self._get_oscbf_operator()):
            try:
                seq_chunk, _seq_info = self.deform_chunk_with_oscbf(obs, chunk, **kwargs)
                add_candidate("sequential_oscbf", seq_chunk)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sequential OSCBF fallback candidate failed: %s", exc)
        try:
            q_current = self.extract_current_q(obs, chunk)
            recover_seed, _ = self._make_task_progress_recover_chunk(
                q_current,
                chunk,
                action_idx,
            )
            if recover_seed.shape[0] < chunk.shape[0]:
                padded = chunk.copy()
                padded[: recover_seed.shape[0]] = recover_seed
                padded[recover_seed.shape[0] :] = recover_seed[-1]
                recover_seed = padded
            add_candidate("recover_to_task_progress", recover_seed)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Recover-to-task-progress candidate failed: %s", exc)
        hold = chunk.copy()
        if hold.shape[0]:
            q = self.extract_current_q(obs, chunk)
            valid_hold = (
                (self.controlled_action_indices < hold.shape[1])
                & (self.controlled_state_indices < q.shape[0])
            )
            if np.any(valid_hold):
                action_idx = self.controlled_action_indices[valid_hold]
                state_idx = self.controlled_state_indices[valid_hold]
                anchor = self._controlled_anchor(obs, hold, action_idx, state_idx)
                hold[:, action_idx] = anchor[None, :]
            else:
                hold[1:] = hold[0]
        add_candidate("hold", hold)

        best = None
        rejected = []
        for name, candidate in candidates:
            acceptance = self.evaluate_candidate_acceptance(obs, candidate, candidate_type)
            self.fallback_candidate_count += 1
            if candidate_type == "recover":
                self.recover_candidate_count += 1
            else:
                self.deform_candidate_count += 1
            if not acceptance["accepted"]:
                rejected.append(acceptance.get("rejection_reason"))
                continue
            score, score_info = self._score_accepted_candidate(
                obs,
                candidate,
                chunk,
                acceptance,
                candidate_type=candidate_type,
            )
            if best is None or score > best[0]:
                best = (score, name, candidate, acceptance, score_info)

        if best is None:
            if any(reason == "immediate_below_hard_margin" for reason in rejected):
                self.immediate_hard_reject_count += 1
            elif any(reason == "no_safe_prefix" for reason in rejected):
                self.no_safe_prefix_reject_count += 1
            elif any(reason == "horizon_below_desired_margin" for reason in rejected):
                self.horizon_margin_reject_count += 1
            return None, None

        _score, name, candidate, acceptance, score_info = best
        accepted_chunk = self._truncate_chunk_to_safe_prefix(candidate, acceptance)
        self.fallback_candidate_accepted_count += 1
        if acceptance["acceptance_type"] == "safe_prefix":
            self.safe_prefix_accepted_count += 1
        elif acceptance["acceptance_type"] == "first_action_only":
            self.first_action_only_accepted_count += 1
        if candidate_type == "recover":
            self.recover_accepted_count += 1
            self.accepted_recover_steps += 1
        else:
            self.deform_accepted_count += 1
            self.accepted_deform_steps += 1
        info = dict(optimized_info or {})
        mode = "recover_safe_prefix" if candidate_type == "recover" else "deform_safe_prefix"
        if acceptance["acceptance_type"] == "full_horizon":
            mode = "recover" if candidate_type == "recover" else "horizon_deform"
        info.update(
            {
                "optimized_accepted": True,
                "optimized_fallback": "safe_prefix_candidate",
                "optimized_reject_reason": None,
                "fallback_used": False,
                "deform_safe": True,
                "is_safe": True,
                "is_recoverable": True if self.recoverable_deform_enabled else None,
                "safety_rejected": False,
                "recovery_rejected": False,
                "rejection_cause": None,
                "mode": mode,
                "deform_mode": mode,
                "recovery_mode": mode,
                "deformation_source": name,
                "candidate_count": len(candidates),
                "accepted_candidate_name": name,
                "accepted_candidate_type": candidate_type,
                "acceptance_type": acceptance["acceptance_type"],
                "safe_prefix_len": acceptance["safe_prefix_len"],
                "immediate_clearance": acceptance["immediate_clearance"],
                "prefix_min_clearance": acceptance["prefix_min_clearance"],
                "horizon_min_clearance": acceptance["horizon_min_clearance"],
                "rejection_reason": None,
                "full_horizon_required": acceptance["full_horizon_required"],
                "rolling_replan_on_prefix": acceptance["rolling_replan_on_prefix"],
                "safe_prefix_execution": acceptance["safe_prefix_execution"],
                "progress_score": score_info["progress_score"],
                "progress_score_available": score_info["progress_score_available"],
                "deformation_norm": score_info["deformation_norm"],
                "recover_projection_on_nominal": score_info.get("recover_projection_on_nominal"),
                "recover_cosine_to_nominal": score_info.get("recover_cosine_to_nominal"),
                "nominal_rejoin_score": score_info.get("nominal_rejoin_score"),
                "nominal_rejoin_available": score_info.get("nominal_rejoin_available"),
                "nominal_rejoin_suppressed_reason": score_info.get("nominal_rejoin_suppressed_reason"),
                "nominal_rejoin_clearance": score_info.get("nominal_rejoin_clearance"),
                "nominal_rejoin_safe_prefix_len": score_info.get("nominal_rejoin_safe_prefix_len"),
                "recover_task_progress_score": score_info.get("recover_task_progress_score"),
                "recover_score_total": score_info.get("recover_score_total"),
                "recover_rejoin_weight_effective": score_info.get("recover_rejoin_weight_effective"),
                "recover_step_since_deform": score_info.get("recover_step_since_deform"),
                "deform_min_clearance": acceptance["immediate_clearance"],
                "min_clearance": acceptance["immediate_clearance"],
            }
        )
        if candidate_type == "recover":
            info["recover_accepted"] = True
        else:
            info["deform_stage_accepted"] = True
        info.update(self._safechunk_replan_info())
        return accepted_chunk, info

    def _deform_chunk_optimized_with_fallback(
        self,
        obs,
        action_chunk,
        safety_info=None,
        braked_chunk=None,
        **kwargs,
    ):
        chunk, _ = self._as_chunk(action_chunk)
        nominal_q_seq = kwargs.pop("nominal_q_seq", None)
        first_violation = None if safety_info is None else safety_info.get("first_violation")
        try:
            optimized_chunk, optimized_info = self.deform_chunk_optimized(
                nominal_chunk=chunk,
                obs=obs,
                first_violation=first_violation,
                nominal_q_seq=nominal_q_seq,
                safety_info=safety_info,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - defensive integration path
            logger.warning("Optimized SafeChunk deformation failed: %s", exc)
            optimized_chunk = chunk.copy()
            optimized_info = self._optimized_failure_info(str(exc))

        candidate_type = "recover" if self.recoverable_deform_enabled and self.explicit_return else "deform"
        if self.safechunk_acceptance_enabled:
            optimized_acceptance = self.evaluate_candidate_acceptance(
                obs,
                optimized_chunk,
                candidate_type,
            )
        else:
            old_style_accepted = bool(
                optimized_info.get("deform_safe", False)
                and (
                    not self.recoverable_deform_enabled
                    or optimized_info.get("is_recoverable", False)
                )
            )
            optimized_acceptance = {
                "accepted": old_style_accepted,
                "acceptance_type": "full_horizon" if old_style_accepted else "rejected",
                "safe_prefix_len": optimized_chunk.shape[0],
                "immediate_clearance": float(optimized_info.get("min_clearance", float("-inf"))),
                "prefix_min_clearance": float(self.prefix_min_clearance),
                "horizon_min_clearance": float(optimized_info.get("min_clearance", float("-inf"))),
                "rejection_reason": None if old_style_accepted else "horizon_below_desired_margin",
                "full_horizon_required": True,
                "rolling_replan_on_prefix": False,
                "safe_prefix_execution": False,
            }
        self.optimized_solution_count += 1
        self.optimized_candidate_count += 1
        if candidate_type == "recover":
            self.recover_candidate_count += 1
        else:
            self.deform_candidate_count += 1
        optimized_info.update(
            {
                "candidate_count": 1,
                "accepted_candidate_name": "optimized",
                "accepted_candidate_type": candidate_type,
                "acceptance_type": optimized_acceptance["acceptance_type"],
                "safe_prefix_len": optimized_acceptance["safe_prefix_len"],
                "immediate_clearance": optimized_acceptance["immediate_clearance"],
                "prefix_min_clearance": optimized_acceptance["prefix_min_clearance"],
                "horizon_min_clearance": optimized_acceptance["horizon_min_clearance"],
                "rejection_reason": optimized_acceptance["rejection_reason"],
                "full_horizon_required": optimized_acceptance["full_horizon_required"],
                "rolling_replan_on_prefix": optimized_acceptance["rolling_replan_on_prefix"],
                "safe_prefix_execution": optimized_acceptance["safe_prefix_execution"],
            }
        )
        accepted = bool(
            optimized_acceptance.get("accepted", False)
            and (
                optimized_acceptance["acceptance_type"] != "full_horizon"
                or (
                    optimized_info.get("deform_safe", False)
                    and (
                        not self.recoverable_deform_enabled
                        or optimized_info.get("is_recoverable", False)
                    )
                )
            )
        )
        if accepted and optimized_acceptance["acceptance_type"] != "full_horizon":
            optimized_chunk = self._truncate_chunk_to_safe_prefix(
                optimized_chunk,
                optimized_acceptance,
            )
            optimized_info.update(
                {
                    "deform_safe": True,
                    "is_safe": True,
                    "is_recoverable": True if self.recoverable_deform_enabled else None,
                    "safety_rejected": False,
                    "recovery_rejected": False,
                    "rejection_cause": None,
                    "deform_min_clearance": optimized_acceptance["immediate_clearance"],
                    "min_clearance": optimized_acceptance["immediate_clearance"],
                    "recover_accepted": candidate_type == "recover",
                    "deform_stage_accepted": candidate_type == "deform",
                    "mode": (
                        "recover_safe_prefix"
                        if candidate_type == "recover"
                        else "deform_safe_prefix"
                    ),
                    "deform_mode": (
                        "recover_safe_prefix"
                        if candidate_type == "recover"
                        else "deform_safe_prefix"
                    ),
                    "recovery_mode": (
                        "recover_safe_prefix"
                        if candidate_type == "recover"
                        else "deform_safe_prefix"
                    ),
                }
            )
        if (
            accepted
            and optimized_acceptance.get("acceptance_type") == "full_horizon"
            and self.recoverable_deform_enabled
            and self.explicit_return
            and self.commit_accepted_chunks
        ):
            return_min_clearance = optimized_info.get(
                "recover_min_clearance",
                optimized_info.get("return_min_clearance"),
            )
            required_return_clearance = float(
                self.min_clearance + self.committed_execution_margin
            )
            optimized_info["committed_execution_margin"] = float(
                self.committed_execution_margin
            )
            optimized_info["committed_return_acceptance_clearance"] = (
                required_return_clearance
            )
            if (
                return_min_clearance is None
                or float(return_min_clearance) < required_return_clearance
            ):
                accepted = False
                optimized_info.update(
                    {
                        "recover_accepted": False,
                        "deform_safe": False,
                        "is_safe": False,
                        "is_recoverable": False,
                        "safety_rejected": True,
                        "recovery_rejected": True,
                        "committed_margin_rejected": True,
                        "rejection_cause": "unsafe",
                        "clearance_gap": float(
                            required_return_clearance
                            - float(return_min_clearance if return_min_clearance is not None else float("-inf"))
                        ),
                    }
                )

        if accepted:
            if optimized_acceptance["acceptance_type"] == "safe_prefix":
                self.safe_prefix_accepted_count += 1
            elif optimized_acceptance["acceptance_type"] == "first_action_only":
                self.first_action_only_accepted_count += 1
            if candidate_type == "recover":
                self.recover_accepted_count += 1
                self.accepted_recover_steps += 1
            else:
                self.deform_accepted_count += 1
                self.accepted_deform_steps += 1
            optimized_info["optimized_accepted"] = True
            optimized_info["fallback_used"] = False
            optimized_info.update(self._safechunk_replan_info())
            if self.recoverable_deform_enabled and self.explicit_return:
                optimized_info.setdefault("mode", "recover")
                optimized_info.setdefault("deform_mode", "recover")
                optimized_info["deformation_source"] = "explicit_recover_deform"
                optimized_info.setdefault("recovery_mode", "resume_act")
                optimized_info["act_resume_index"] = optimized_info.get(
                    "return_target_index",
                    optimized_info.get("rejoin_index"),
                )
                optimized_info["act_resume_supported"] = False
                optimized_info.setdefault(
                    "act_resume_note",
                    "TODO: executor replans each outer step; use recover_target_index "
                    "when a persistent ACT chunk cursor is added.",
                )
            elif self.recoverable_deform_enabled:
                optimized_info["mode"] = "optimized_recoverable_deform"
                optimized_info["deform_mode"] = "optimized_recoverable_deform"
                optimized_info["deformation_source"] = "optimized_recoverable_deform"
                optimized_info["recovery_mode"] = "optimized_recoverable_deform"
                optimized_info["act_resume_index"] = optimized_info.get("rejoin_index")
                optimized_info["act_resume_supported"] = False
                optimized_info["act_resume_note"] = (
                    "TODO: executor replans each outer step; use rejoin_index "
                    "when a persistent ACT chunk cursor is added."
                )
            return optimized_chunk, optimized_info

        if not accepted:
            self.optimized_rejected_count += 1
            if candidate_type == "recover":
                self.recover_rejected_count += 1
            else:
                self.deform_rejected_count += 1
            reason = optimized_acceptance.get("rejection_reason")
            if reason == "immediate_below_hard_margin":
                self.immediate_hard_reject_count += 1
            elif reason == "no_safe_prefix":
                self.no_safe_prefix_reject_count += 1
            elif reason == "horizon_below_desired_margin":
                self.horizon_margin_reject_count += 1
        reject_reason = self._optimized_reject_reason(optimized_info)
        candidate_chunk, candidate_info = (None, None)
        should_try_candidate_fallback = (
            self.safechunk_acceptance_enabled
            and self.allow_candidate_fallback
            and (
                not self.candidate_fallback_only_if_no_optimized_result
                or self.optimized_solution_count == 0
            )
        )
        if should_try_candidate_fallback:
            candidate_chunk, candidate_info = self._rolling_prefix_candidate_fallback(
                obs,
                chunk,
                optimized_chunk,
                optimized_info,
                safety_info=safety_info,
                braked_chunk=braked_chunk,
                candidate_type=candidate_type,
                **kwargs,
            )
        if candidate_info is not None:
            return candidate_chunk, candidate_info
        fallback_mode = self.optimized_fallback
        if self.recoverable_deform_enabled and (
            self.brake_if_unrecoverable or self.explicit_return
        ):
            fallback_mode = "brake"
        if fallback_mode == "candidate":
            candidate_chunk, candidate_info = self.deform_chunk_candidate(
                obs,
                chunk,
                safety_info=safety_info,
                **kwargs,
            )
            candidate_info.update(
                self._prefixed_optimized_info(optimized_info)
            )
            candidate_info.update(
                {
                    "deform_mode": "optimized",
                    "optimized_accepted": False,
                    "optimized_fallback": "candidate",
                    "optimized_reject_reason": reject_reason,
                    "fallback_reason": reject_reason,
                    "fallback_used": True,
                }
            )
            return candidate_chunk, candidate_info

        self.fallback_brake_after_reject_count += 1
        info = dict(optimized_info)
        info.update(self._safechunk_replan_info())
        info.update(
            {
                "deform_safe": False,
                "optimized_accepted": False,
                "optimized_fallback": "brake",
                "optimized_reject_reason": reject_reason,
                "fallback_reason": reject_reason,
                "fallback_used": True,
            }
        )
        if braked_chunk is None:
            braked_chunk = chunk.copy()
        return braked_chunk, info

    def deform_chunk_optimized(
        self,
        nominal_chunk,
        q_current=None,
        qd_current=None,
        first_violation=None,
        nominal_q_seq=None,
        nominal_ee_seq=None,
        human_state=None,
        obs=None,
        safety_info=None,
        **kwargs,
    ):
        """Optimize a whole action chunk with a CEM gradient-free objective.

        The current rollout and horizon-clearance path is NumPy/JAX based, so this
        uses the requested differentiability fallback instead of pretending the
        computation is a PyTorch differentiable graph.
        """
        chunk, _ = self._as_chunk(nominal_chunk)
        obs = self._obs_with_q(obs, q_current)
        if nominal_q_seq is None:
            nominal_q_seq = self.rollout_chunk(chunk, q_current, qd_current, obs=obs)
        else:
            nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        if safety_info is None:
            safety_info = self.evaluate_horizon_safety(obs, nominal_q_seq)
        if first_violation is None:
            first_violation = safety_info.get("first_violation")

        if self.recoverable_deform_enabled and self.explicit_return:
            return self.deform_chunk_optimized_explicit_return(
                nominal_chunk=chunk,
                obs=obs,
                first_violation=first_violation,
                nominal_q_seq=nominal_q_seq,
                nominal_ee_seq=nominal_ee_seq,
                human_state=human_state,
                safety_info=safety_info,
                **kwargs,
            )

        rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)

        valid = self._valid_control_indices(chunk)
        if not np.any(valid):
            info = self._optimized_final_info(
                obs,
                chunk,
                chunk,
                nominal_q_seq,
                nominal_ee_seq,
                human_state,
                j_best=None,
                rejoin_loss=float("inf"),
                losses={},
                rejoin_context=rejoin_context,
            )
            return chunk.copy(), info

        action_idx = self.controlled_action_indices[valid]
        seed_chunks = self._optimized_seed_chunks(obs, chunk, safety_info)
        seed_ctrl = [candidate[:, action_idx].copy() for candidate in seed_chunks]
        mean = chunk[:, action_idx].copy()
        std = np.full_like(mean, self.opt_lr, dtype=np.float32)
        best_record = None

        num_iters = max(1, self.opt_iters)
        for _ in range(num_iters):
            ctrl_samples = [mean.copy()]
            ctrl_samples.extend(sample.copy() for sample in seed_ctrl)
            remaining = max(0, self.opt_population - len(ctrl_samples))
            if remaining:
                noise = self._rng.normal(
                    loc=0.0,
                    scale=std[None, :, :],
                    size=(remaining,) + mean.shape,
                ).astype(np.float32)
                ctrl_samples.extend(mean[None, :, :] + noise)

            ctrl_sample_batch = np.stack(ctrl_samples, axis=0).astype(np.float32)
            candidate_batch = self._jax_project_candidate_population(
                chunk,
                ctrl_sample_batch,
                action_idx,
            )
            if candidate_batch is None:
                candidates = []
                for ctrl_sample in ctrl_samples:
                    candidate = chunk.copy()
                    candidate[:, action_idx] = ctrl_sample
                    candidate = self._project_optimized_chunk(candidate, chunk, action_idx)
                    candidates.append(candidate)
                candidate_batch = np.stack(candidates, axis=0).astype(np.float32)
            else:
                candidates = [candidate_batch[i].copy() for i in range(candidate_batch.shape[0])]
            try:
                costs, losses_list = self._optimized_deformation_cost_batch(
                    obs,
                    candidate_batch,
                    chunk,
                    nominal_q_seq,
                    nominal_ee_seq,
                    human_state,
                    rejoin_context=rejoin_context,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Batched optimized cost failed; falling back to serial: %s", exc)
                costs = []
                losses_list = []
                for candidate in candidates:
                    cost, losses = self._optimized_deformation_cost(
                        obs,
                        candidate,
                        chunk,
                        nominal_q_seq,
                        nominal_ee_seq,
                        human_state,
                        rejoin_context=rejoin_context,
                    )
                    costs.append(cost)
                    losses_list.append(losses)

            records = [
                {
                    "cost": float(cost),
                    "losses": losses,
                    "chunk": candidate,
                    "ctrl": candidate[:, action_idx].copy(),
                }
                for cost, losses, candidate in zip(costs, losses_list, candidates)
            ]
            records.sort(key=lambda item: item["cost"])
            if best_record is None or records[0]["cost"] < best_record["cost"]:
                best_record = records[0]

            elite_count = max(1, int(round(self.opt_population * self.opt_elite_frac)))
            elite_ctrl = np.stack(
                [record["ctrl"] for record in records[:elite_count]],
                axis=0,
            )
            elite_mean = elite_ctrl.mean(axis=0).astype(np.float32)
            elite_std = elite_ctrl.std(axis=0).astype(np.float32)
            mean = (0.3 * mean + 0.7 * elite_mean).astype(np.float32)
            std = np.maximum(elite_std, self.opt_lr * 0.05).astype(np.float32)

        best_chunk = self._project_optimized_chunk(best_record["chunk"], chunk, action_idx)
        info = self._optimized_final_info(
            obs,
            best_chunk,
            chunk,
            nominal_q_seq,
            nominal_ee_seq,
            human_state,
            j_best=best_record["losses"].get("j_best"),
            rejoin_loss=best_record["losses"].get("rejoin_loss", float("inf")),
            losses=best_record["losses"],
            rejoin_context=rejoin_context,
        )
        return best_chunk, info

    def deform_chunk_optimized_explicit_return(
        self,
        nominal_chunk,
        obs,
        first_violation=None,
        nominal_q_seq=None,
        nominal_ee_seq=None,
        human_state=None,
        safety_info=None,
        **kwargs,
    ):
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

        valid = self._valid_control_indices(chunk)
        if not np.any(valid):
            return chunk.copy(), self._explicit_return_info(
                obs=obs,
                chunk=chunk,
                nominal=chunk,
                context=context,
                recovery_phase="horizon_deform",
                yield_chunk=chunk,
                yield_losses={},
                yield_eval=self.evaluate_horizon_safety(obs, nominal_q_seq),
                yield_accepted=False,
                return_chunk=None,
                return_losses={},
                return_eval=None,
                return_accepted=False,
                return_rejoin_loss=float("inf"),
                return_target_index=None,
                fallback_used=True,
                rejection_cause="unsafe_and_unrecoverable",
            )

        action_idx = self.controlled_action_indices[valid]
        yield_len = min(chunk.shape[0], self.yield_horizon)
        yield_nominal = chunk[:yield_len].copy()
        yield_seed_chunks = [candidate[:yield_len].copy() for _, candidate in self._make_chunk_deformation_candidates(obs, chunk, safety_info or {})]

        def yield_cost(candidate):
            return self._yield_deformation_cost(obs, candidate, yield_nominal, action_idx)

        def yield_early_stop(record):
            losses = record.get("losses", {})
            return float(losses.get("min_clearance", float("-inf"))) >= self._acceptance_clearance_threshold()

        yield_record = self._optimize_controlled_chunk(
            obs,
            yield_nominal,
            action_idx,
            yield_cost,
            seed_chunks=yield_seed_chunks,
            batch_cost_fn=lambda candidates: self._yield_deformation_cost_batch(
                obs,
                candidates,
                yield_nominal,
                action_idx,
            ),
            early_stop_fn=yield_early_stop,
        )
        yield_chunk = yield_record["chunk"]
        yield_q_seq = self.rollout_nominal_chunk(obs, yield_chunk)
        yield_eval = self.evaluate_horizon_safety(obs, yield_q_seq)
        yield_min_clearance = float(yield_eval.get("min_clearance", float("-inf")))
        yield_accepted = bool(
            yield_min_clearance >= self._acceptance_clearance_threshold()
        )
        if not yield_accepted:
            context.phase = "horizon_deform"
            return chunk.copy(), self._explicit_return_info(
                obs=obs,
                chunk=chunk,
                nominal=chunk,
                context=context,
                recovery_phase="horizon_deform",
                yield_chunk=yield_chunk,
                yield_losses=yield_record["losses"],
                yield_eval=yield_eval,
                yield_accepted=False,
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
        self.recovery_anchor_state = np.asarray(yield_q_seq[-1], dtype=np.float32).copy()
        return_obs = self._obs_with_q(obs, yield_q_seq[-1])
        stale_recovery_attempted = bool(self.safechunk_replan_enabled)
        return_to_old_path_suppressed = False
        recover_to_task_progress = bool(
            self.safechunk_replan_enabled
            and self.recovery_target_mode == "task_progress"
        )
        nominal_return, seed_target_index = self._make_return_seed_chunk(
            context,
            yield_q_seq[-1],
            chunk,
            action_idx,
        )
        nominal_recover_feasible, _nominal_recover_eval, immediate_safe = (
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
                yield_q_seq[-1],
                chunk,
                action_idx,
                context=context,
                default_target_index=seed_target_index,
            )
        else:
            return_nominal = nominal_return
        task_recover_feasible, _task_recover_eval, immediate_safe = self._recover_seed_feasible(
            return_obs,
            return_nominal,
        )
        recovery_target_feasible = bool(task_recover_feasible)
        if not immediate_safe:
            self.emergency_brake_steps += 1
        self.current_recovery_plan = np.asarray(return_nominal, dtype=np.float32).copy()
        return_rejoin_context = self._make_rejoin_context(
            context.nominal_q_seq,
            context.nominal_ee_seq,
        )

        def return_cost(candidate):
            if recover_to_task_progress:
                return self._recover_task_progress_cost(
                    return_obs,
                    candidate,
                    return_nominal,
                    action_idx,
                    reference_chunk=return_nominal,
                )
            return self._return_deformation_cost(
                return_obs,
                candidate,
                return_nominal,
                context.nominal_q_seq,
                return_rejoin_context,
                action_idx,
            )

        def return_early_stop(record):
            losses = record.get("losses", {})
            min_clearance = float(losses.get("min_clearance", float("-inf")))
            if min_clearance < self._acceptance_clearance_threshold():
                return False
            if recover_to_task_progress:
                return float(losses.get("recover_task_progress_score", 0.0)) > 0.0
            rejoin_loss = float(losses.get("rejoin_loss", losses.get("return_rejoin_loss", float("inf"))))
            return self._sqrt_loss(rejoin_loss) < self.q_rejoin_threshold

        return_record = self._optimize_controlled_chunk(
            return_obs,
            return_nominal,
            action_idx,
            return_cost,
            seed_chunks=[return_nominal],
            batch_cost_fn=(
                (lambda candidates: self._recover_task_progress_cost_batch(
                    return_obs,
                    candidates,
                    return_nominal,
                    action_idx,
                    reference_chunk=return_nominal,
                ))
                if recover_to_task_progress
                else (lambda candidates: self._return_deformation_cost_batch(
                    return_obs,
                    candidates,
                    return_nominal,
                    context.nominal_q_seq,
                    return_rejoin_context,
                    action_idx,
                ))
            ),
            early_stop_fn=return_early_stop,
        )
        return_chunk = return_record["chunk"]
        self._tick_unsafe_recovery_cooldowns()
        direct_rejoin_attempted = False
        direct_rejoin_rejected = False
        detour_rejoin_attempted = False
        detour_rejoin_accepted = False
        repeated_unsafe_target = False
        recovery_candidate_class = "direct_rejoin"
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
        if not immediate_safe:
            direct_path["immediate_safe"] = False
            direct_path["reject_reason"] = "immediate_unsafe"
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
            recover_reject_reason = self._recovery_reject_reason(
                direct_terminal,
                direct_path,
                repeated_unsafe_target=direct_suppressed,
                direction_ok=bool(selected_direction_terms["recover_direction_ok"]),
                ordered_ok=bool(direct_terminal.get("recover_ordered_ok", True)),
            )
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
        return_rejoin_ok = bool(
            direct_terminal["q_rejoin_ok"] and direct_terminal["qd_rejoin_ok"]
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

        best_detour = None
        should_try_detour = bool(
            not return_accepted
            and self.enable_detour_rejoin
            and recover_reject_reason
            in {
                "path_unsafe",
                "prefix_unsafe",
                "immediate_unsafe",
                "repeated_unsafe_target",
                "direction_alignment_failed",
                "ordered_path_failed",
            }
        )
        if should_try_detour:
            for detour_name, detour_chunk in self._make_recovery_detour_candidates(
                return_obs,
                return_chunk,
                action_idx,
            ):
                detour_rejoin_attempted = True
                self.detour_rejoin_attempt_count += 1
                detour_terminal = self._recovery_terminal_rejoin_info(
                    return_obs,
                    detour_chunk,
                    context,
                    return_rejoin_context,
                    default_target_index=seed_target_index,
                )
                detour_path = self.evaluate_recovery_path_safety(
                    return_obs,
                    detour_chunk,
                    candidate_name="recover_detour",
                )
                detour_direction_info = self.compute_nominal_rejoin_score(
                    detour_chunk,
                    chunk,
                    obs=return_obs,
                )
                detour_direction_terms = self._recover_direction_alignment_terms(
                    detour_direction_info
                )
                detour_reason = self._recovery_reject_reason(
                    detour_terminal,
                    detour_path,
                    direction_ok=bool(detour_direction_terms["recover_direction_ok"]),
                    ordered_ok=bool(detour_terminal.get("recover_ordered_ok", True)),
                )
                detour_target_key = self.make_recovery_target_key(detour_chunk)
                detour_path_key = self._make_recovery_path_key(
                    detour_chunk,
                    detour_target_key,
                )
                if detour_reason in {
                    "path_unsafe",
                    "prefix_unsafe",
                    "immediate_unsafe",
                }:
                    self._mark_recovery_path_failure(
                        detour_target_key,
                        detour_path_key,
                        detour_reason,
                    )
                    continue
                if detour_reason is not None:
                    continue
                score = self._score_recovery_detour_candidate(
                    detour_chunk,
                    return_chunk,
                    action_idx,
                    detour_path,
                    detour_terminal,
                )
                if best_detour is None or score > best_detour[0]:
                    best_detour = (
                        score,
                        detour_name,
                        detour_chunk,
                        detour_path,
                        detour_terminal,
                        detour_target_key,
                        detour_direction_info,
                        detour_direction_terms,
                    )

        if best_detour is not None:
            (
                _detour_score,
                detour_name,
                return_chunk,
                selected_path,
                selected_terminal,
                target_key,
                selected_direction_info,
                selected_direction_terms,
            ) = best_detour
            del _detour_score
            recovery_candidate_class = "detour_rejoin"
            detour_rejoin_accepted = True
            self.detour_rejoin_accept_count += 1
            recover_reject_reason = None
            return_q_seq = selected_terminal["q_seq"]
            return_eval = selected_terminal["eval"]
            return_min_clearance = float(selected_terminal["min_clearance"])
            return_rejoin_loss = float(selected_terminal["q_rejoin_loss"])
            return_target_index = selected_terminal["target_index"]
            return_q_time_ms = float(selected_terminal["q_eval_time_ms"])
            return_qd_rejoin_loss = float(selected_terminal["qd_rejoin_loss"])
            return_qd_rejoin_index = selected_terminal["qd_rejoin_index"]
            return_qd_time_ms = float(selected_terminal["qd_eval_time_ms"])
            return_q_rejoin_dist = float(selected_terminal["q_rejoin_dist"])
            return_qd_rejoin_dist = float(selected_terminal["qd_rejoin_dist"])
            return_rejoin_ok = True
            return_safe = bool(selected_path["path_safe"])
            direct_path = selected_path
            return_record["losses"]["accepted_candidate_name"] = detour_name
            return_accepted = True
        elif not return_accepted and self.enable_delayed_rejoin:
            recovery_candidate_class = "delayed_rejoin"
            self.delayed_rejoin_active = True
            self.delayed_rejoin_steps = int(self.delayed_rejoin_wait_steps)
            self.delayed_rejoin_count += 1

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
                    "recover_ordered_loss",
                    "recover_ordered_pose_weight",
                    "recover_ordered_delta_weight",
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
                emergency_brake_immediate_unsafe=not immediate_safe,
            )
        )
        return_record["losses"].update(
            {
                "recover_required": True,
                "recovery_candidate_class": recovery_candidate_class,
                "recover_reject_reason": recover_reject_reason,
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
                yield_chunk,
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
                yield_chunk=yield_chunk,
                yield_losses=yield_record["losses"],
                yield_eval=yield_eval,
                yield_accepted=True,
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
            yield_chunk=yield_chunk,
            yield_losses=yield_record["losses"],
            yield_eval=yield_eval,
            yield_accepted=True,
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
        nominal_chunk,
        nominal_q_seq,
        nominal_ee_seq=None,
        start_chunk_index=None,
        observation_history=None,
        policy_buffer_metadata=None,
    ):
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

    def _optimize_controlled_chunk(
        self,
        obs,
        nominal_chunk,
        action_idx,
        cost_fn,
        seed_chunks=None,
        batch_cost_fn=None,
        early_stop_fn=None,
        min_iters: int = 1,
    ):
        del obs
        nominal_chunk = np.asarray(nominal_chunk, dtype=np.float32)
        seed_chunks = seed_chunks or []
        seed_ctrl = []
        for seed in seed_chunks:
            seed = np.asarray(seed, dtype=np.float32)
            if seed.shape == nominal_chunk.shape:
                projected = self._project_optimized_chunk(seed, nominal_chunk, action_idx)
                seed_ctrl.append(projected[:, action_idx].copy())
        mean = nominal_chunk[:, action_idx].copy()
        std = np.full_like(mean, self.opt_lr, dtype=np.float32)
        best_record = None
        num_iters = max(1, self.opt_iters)
        min_iters = max(1, int(min_iters))
        iterations_run = 0
        early_stopped = False
        for iter_idx in range(num_iters):
            iterations_run = iter_idx + 1
            ctrl_samples = [mean.copy()]
            ctrl_samples.extend(sample.copy() for sample in seed_ctrl)
            remaining = max(0, self.opt_population - len(ctrl_samples))
            if remaining:
                noise = self._rng.normal(
                    loc=0.0,
                    scale=std[None, :, :],
                    size=(remaining,) + mean.shape,
                ).astype(np.float32)
                ctrl_samples.extend(mean[None, :, :] + noise)

            ctrl_sample_batch = np.stack(ctrl_samples, axis=0).astype(np.float32)
            candidate_batch = self._jax_project_candidate_population(
                nominal_chunk,
                ctrl_sample_batch,
                action_idx,
            )
            if candidate_batch is None:
                candidates = []
                for ctrl_sample in ctrl_samples:
                    candidate = nominal_chunk.copy()
                    candidate[:, action_idx] = ctrl_sample
                    candidate = self._project_optimized_chunk(candidate, nominal_chunk, action_idx)
                    candidates.append(candidate)
            else:
                candidates = [candidate_batch[i].copy() for i in range(candidate_batch.shape[0])]

            costs = None
            losses_list = None
            if batch_cost_fn is not None and candidates:
                try:
                    costs, losses_list = batch_cost_fn(
                        np.stack(candidates, axis=0).astype(np.float32)
                    )
                    if len(costs) != len(candidates) or len(losses_list) != len(candidates):
                        raise ValueError("batch cost result length mismatch")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Batched controlled-chunk cost failed; falling back to serial: %s", exc)
                    costs = None
                    losses_list = None

            records = []
            if costs is None or losses_list is None:
                for candidate in candidates:
                    cost, losses = cost_fn(candidate)
                    records.append(
                        {
                            "cost": float(cost),
                            "losses": losses,
                            "chunk": candidate,
                            "ctrl": candidate[:, action_idx].copy(),
                        }
                    )
            else:
                for cost, losses, candidate in zip(costs, losses_list, candidates):
                    records.append(
                        {
                            "cost": float(cost),
                            "losses": losses,
                            "chunk": candidate,
                            "ctrl": candidate[:, action_idx].copy(),
                        }
                    )
            records.sort(key=lambda item: item["cost"])
            if best_record is None or records[0]["cost"] < best_record["cost"]:
                best_record = records[0]
            if (
                early_stop_fn is not None
                and iterations_run >= min_iters
                and early_stop_fn(best_record)
            ):
                early_stopped = True
                break
            elite_count = max(1, int(round(self.opt_population * self.opt_elite_frac)))
            elite_ctrl = np.stack(
                [record["ctrl"] for record in records[:elite_count]],
                axis=0,
            )
            elite_mean = elite_ctrl.mean(axis=0).astype(np.float32)
            elite_std = elite_ctrl.std(axis=0).astype(np.float32)
            mean = (0.3 * mean + 0.7 * elite_mean).astype(np.float32)
            std = np.maximum(elite_std, self.opt_lr * 0.05).astype(np.float32)
        best_record["chunk"] = self._project_optimized_chunk(
            best_record["chunk"],
            nominal_chunk,
            action_idx,
        )
        best_record.setdefault("losses", {})
        best_record["losses"]["cem_iterations_run"] = int(iterations_run)
        best_record["losses"]["cem_early_stopped"] = bool(early_stopped)
        best_record["losses"]["cem_max_iters"] = int(num_iters)
        best_record["losses"]["cem_population"] = int(self.opt_population)
        return best_record

    def _yield_deformation_cost(self, obs, candidate, nominal, action_idx):
        q_seq = self.rollout_nominal_chunk(obs, candidate)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        safety_loss = float(np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum())
        action_deviation_loss = float(
            np.square(candidate[:, action_idx] - nominal[:, action_idx]).mean()
        ) if len(action_idx) else 0.0
        smoothness_loss = self._smoothness_loss(candidate, action_idx)
        finite_h = np.nan_to_num(h_seq, nan=0.0, posinf=self.min_clearance, neginf=-1.0)
        retreat_loss = -float(np.mean(np.clip(finite_h, -1.0, 1.0)))
        total_loss = float(
            self.lambda_yield_safety * safety_loss
            + self.lambda_yield_action * action_deviation_loss
            + self.lambda_yield_smooth * smoothness_loss
            + self.lambda_retreat * retreat_loss
        )
        return total_loss, {
            "safety_loss": safety_loss,
            "action_deviation_loss": action_deviation_loss,
            "smoothness_loss": smoothness_loss,
            "retreat_loss": retreat_loss,
            "total_loss": total_loss,
            "min_clearance": float(np.min(h_seq)),
        }

    def _yield_deformation_cost_batch(self, obs, candidates, nominal, action_idx):
        candidates = np.asarray(candidates, dtype=np.float32)
        rollout_t0 = time.perf_counter()
        q_seq_batch = self.rollout_nominal_chunk_batch(obs, candidates)
        rollout_time_ms = 1000.0 * (time.perf_counter() - rollout_t0)
        safety_eval = self.evaluate_horizon_safety_batch(obs, q_seq_batch)
        h_seq = self._clearance_sequence_batch_from_eval(
            safety_eval,
            candidates.shape[0],
            candidates.shape[1],
        )
        safety_loss = np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum(axis=1)
        if len(action_idx):
            action_deviation_loss = np.square(
                candidates[:, :, action_idx] - nominal[None, :, action_idx]
            ).mean(axis=(1, 2))
        else:
            action_deviation_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        smoothness_loss = self._smoothness_loss_batch(candidates, action_idx)
        finite_h = np.nan_to_num(h_seq, nan=0.0, posinf=self.min_clearance, neginf=-1.0)
        retreat_loss = -np.mean(np.clip(finite_h, -1.0, 1.0), axis=1)
        total_loss = (
            self.lambda_yield_safety * safety_loss
            + self.lambda_yield_action * action_deviation_loss
            + self.lambda_yield_smooth * smoothness_loss
            + self.lambda_retreat * retreat_loss
        )
        losses = [
            {
                "safety_loss": float(safety_loss[i]),
                "action_deviation_loss": float(action_deviation_loss[i]),
                "smoothness_loss": float(smoothness_loss[i]),
                "retreat_loss": float(retreat_loss[i]),
                "total_loss": float(total_loss[i]),
                "min_clearance": float(np.min(h_seq[i])),
                "batched_optimizer": True,
                "jax_batched_optimizer": bool(self._jax_optimizer_ready()),
                "jax_rollout_time_ms": float(rollout_time_ms) / max(1, candidates.shape[0]),
            }
            for i in range(candidates.shape[0])
        ]
        return total_loss.astype(np.float32), losses

    def _return_deformation_cost(
        self,
        obs,
        candidate,
        nominal,
        nominal_q_seq,
        rejoin_context,
        action_idx,
    ):
        q_seq = self.rollout_nominal_chunk(obs, candidate)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        safety_loss = float(np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum())
        rejoin_loss, j_best, q_time_ms = self._q_rejoin_loss(
            q_seq,
            nominal_q_seq=nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        action_deviation_loss = float(
            np.square(candidate[:, action_idx] - nominal[:, action_idx]).mean()
        ) if len(action_idx) else 0.0
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
            + self.lambda_return_safety * safety_loss
            + self.lambda_return_smooth * smoothness_loss
            + self.lambda_return_action * action_deviation_loss
            + ordered_loss
        )
        return total_loss, {
            "safety_loss": safety_loss,
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

    def _return_deformation_cost_batch(
        self,
        obs,
        candidates,
        nominal,
        nominal_q_seq,
        rejoin_context,
        action_idx,
    ):
        candidates = np.asarray(candidates, dtype=np.float32)
        rollout_t0 = time.perf_counter()
        q_seq_batch = self.rollout_nominal_chunk_batch(obs, candidates)
        rollout_time_ms = 1000.0 * (time.perf_counter() - rollout_t0)
        safety_eval = self.evaluate_horizon_safety_batch(obs, q_seq_batch)
        h_seq = self._clearance_sequence_batch_from_eval(
            safety_eval,
            candidates.shape[0],
            candidates.shape[1],
        )
        safety_loss = np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum(axis=1)
        rejoin_loss, j_best, q_time_ms = self._q_rejoin_loss_batch(
            q_seq_batch,
            nominal_q_seq=nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        if len(action_idx):
            action_deviation_loss = np.square(
                candidates[:, :, action_idx] - nominal[None, :, action_idx]
            ).mean(axis=(1, 2))
        else:
            action_deviation_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        smoothness_loss = self._smoothness_loss_batch(candidates, action_idx)
        ordered_terms_list = []
        for i in range(candidates.shape[0]):
            ordered_target_index = self._ordered_recovery_start_index(
                j_best[i],
                q_seq_batch.shape[1],
                nominal_q_seq,
            )
            ordered_terms_list.append(
                self._ordered_recovery_path_terms(
                    q_seq_batch[i],
                    nominal_q_seq,
                    target_index=ordered_target_index,
                    rejoin_context=rejoin_context,
                )
            )
        ordered_loss = np.asarray(
            [float(item["recover_ordered_loss"]) for item in ordered_terms_list],
            dtype=np.float32,
        )
        total_loss = (
            self.lambda_return_rejoin * rejoin_loss
            + self.lambda_return_safety * safety_loss
            + self.lambda_return_smooth * smoothness_loss
            + self.lambda_return_action * action_deviation_loss
            + ordered_loss
        )
        per_q_ms = float(q_time_ms) / max(1, candidates.shape[0])
        losses = [
            {
                "safety_loss": float(safety_loss[i]),
                "rejoin_loss": float(rejoin_loss[i]),
                "return_rejoin_loss": float(rejoin_loss[i]),
                "action_deviation_loss": float(action_deviation_loss[i]),
                "smoothness_loss": float(smoothness_loss[i]),
                **ordered_terms_list[i],
                "total_loss": float(total_loss[i]),
                "min_clearance": float(np.min(h_seq[i])),
                "j_best": j_best[i],
                "return_target_index": j_best[i],
                "rejoin_q_eval_time_ms": per_q_ms,
                "batched_optimizer": True,
            }
            for i in range(candidates.shape[0])
        ]
        return total_loss.astype(np.float32), losses

    def _record_nominal_rejoin_target(self, target_info, rejoin_info=None, progress_score=None):
        if target_info.get("available"):
            self.nominal_rejoin_available_count += 1
        else:
            self.nominal_rejoin_suppressed_count += 1
            reason = target_info.get("suppressed_reason")
            if reason == "stale_blocked_nominal":
                self.stale_nominal_rejoin_suppressed_count += 1
            elif reason == "nominal_prefix_unsafe":
                self.nominal_prefix_unsafe_suppressed_count += 1
        if rejoin_info is not None:
            projection = float(rejoin_info.get("recover_projection_on_nominal", 0.0))
            if projection > 0.0:
                self.recover_positive_projection_count += 1
            else:
                self.recover_nonpositive_projection_count += 1
            cosine = float(rejoin_info.get("recover_cosine_to_nominal", 0.0))
            if np.isfinite(projection):
                self._recover_projection_history.append(projection)
            if np.isfinite(cosine):
                self._recover_cosine_history.append(cosine)
        if progress_score is not None and np.isfinite(float(progress_score)):
            self._recover_task_progress_history.append(float(progress_score))

    def _recover_rejoin_weight_effective(self):
        weight = float(self.recover_rejoin_nominal_weight)
        if not self.safechunk_recover_enabled:
            return 0.0
        if self.rejoin_weight_schedule == "none":
            return 0.0
        if self.rejoin_weight_schedule == "ramp":
            ramp = min(1.0, float(self.recover_step_since_deform) / float(max(1, self.rejoin_ramp_steps)))
            weight *= ramp
        return float(weight)


    def _recover_direction_alignment_terms(self, rejoin_info):
        available = bool(
            rejoin_info is not None
            and float(rejoin_info.get("nominal_delta_norm", 0.0) or 0.0) > 1e-6
            and float(rejoin_info.get("candidate_delta_norm", 0.0) or 0.0) > 1e-6
        )
        cosine = float(
            0.0
            if rejoin_info is None
            else rejoin_info.get("recover_cosine_to_nominal", 0.0)
        )
        if not np.isfinite(cosine):
            cosine = 0.0
        threshold = float(self.recover_min_direction_cosine)
        margin = float(self.recover_direction_alignment_margin)
        loss = float(max(0.0, threshold + margin - cosine) ** 2)
        ok = bool(
            (not self.require_recover_direction_alignment)
            or (not available)
            or cosine >= threshold
        )
        return {
            "recover_direction_alignment_available": bool(available),
            "recover_direction_cosine": float(cosine),
            "recover_direction_cosine_threshold": float(threshold),
            "recover_direction_loss": float(loss),
            "recover_direction_ok": bool(ok),
        }

    def _zero_ordered_recovery_terms(self, target_index=None):
        return {
            "recover_ordered_path_available": False,
            "recover_ordered_target_index": (
                None if target_index is None else int(target_index)
            ),
            "recover_ordered_horizon": 0,
            "recover_ordered_pose_loss": 0.0,
            "recover_ordered_delta_loss": 0.0,
            "recover_ordered_loss": 0.0,
            "recover_ordered_pose_weight": float(self.recover_ordered_pose_weight),
            "recover_ordered_delta_weight": float(self.recover_ordered_delta_weight),
            "recover_ordered_pose_threshold": float(self.recover_ordered_pose_threshold),
            "recover_ordered_delta_threshold": float(self.recover_ordered_delta_threshold),
            "recover_ordered_ok": bool(not self.require_recover_ordered_path),
        }

    def _ordered_recovery_start_index(self, terminal_index, horizon, nominal_q_seq):
        if terminal_index is None:
            return None
        try:
            terminal_index = int(terminal_index)
        except Exception:  # noqa: BLE001
            return None
        horizon = max(1, int(horizon))
        nominal_len = 0 if nominal_q_seq is None else int(np.asarray(nominal_q_seq).shape[0])
        if nominal_len <= 0:
            return None
        start = terminal_index - horizon + 1
        start = max(0, min(start, nominal_len - 1))
        return int(start)

    def _ordered_recovery_path_terms(
        self,
        q_seq,
        nominal_q_seq,
        *,
        target_index=0,
        rejoin_context=None,
    ):
        if (
            not self.safechunk_recover_enabled
            or (
                self.recover_ordered_pose_weight <= 0.0
                and self.recover_ordered_delta_weight <= 0.0
            )
        ):
            return self._zero_ordered_recovery_terms(target_index)
        if target_index is None or q_seq is None or nominal_q_seq is None:
            return self._zero_ordered_recovery_terms(target_index)
        q_seq = np.asarray(q_seq, dtype=np.float32)
        nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        if q_seq.ndim != 2 or nominal_q_seq.ndim != 2 or q_seq.shape[0] == 0:
            return self._zero_ordered_recovery_terms(target_index)
        try:
            target_index = int(target_index)
        except Exception:  # noqa: BLE001
            return self._zero_ordered_recovery_terms(None)
        if target_index < 0 or target_index >= nominal_q_seq.shape[0]:
            return self._zero_ordered_recovery_terms(target_index)
        valid = self.controlled_state_indices < min(q_seq.shape[1], nominal_q_seq.shape[1])
        state_idx = self.controlled_state_indices[valid]
        if state_idx.size == 0:
            return self._zero_ordered_recovery_terms(target_index)
        horizon = min(q_seq.shape[0], nominal_q_seq.shape[0] - target_index)
        if horizon <= 0:
            return self._zero_ordered_recovery_terms(target_index)
        candidate = q_seq[:horizon, state_idx]
        nominal = nominal_q_seq[target_index : target_index + horizon, state_idx]
        weights = np.ones(state_idx.shape[0], dtype=np.float32)
        if rejoin_context is not None:
            ctx_idx = rejoin_context.get("q_state_indices")
            ctx_weights = rejoin_context.get("q_weights")
            if ctx_idx is not None and ctx_weights is not None:
                weight_by_idx = {
                    int(idx): float(weight)
                    for idx, weight in zip(
                        np.asarray(ctx_idx).reshape(-1),
                        np.asarray(ctx_weights, dtype=np.float32).reshape(-1),
                    )
                }
                weights = np.asarray(
                    [weight_by_idx.get(int(idx), 1.0) for idx in state_idx],
                    dtype=np.float32,
                )
        diff = (candidate - nominal) * weights.reshape(1, -1)
        pose_loss = float(np.square(diff).mean())
        if horizon >= 2:
            candidate_delta = candidate[1:] - candidate[:-1]
            nominal_delta = nominal[1:] - nominal[:-1]
            delta_diff = (candidate_delta - nominal_delta) * weights.reshape(1, -1)
            delta_loss = float(np.square(delta_diff).mean())
        else:
            delta_loss = 0.0
        ordered_loss = float(
            self.recover_ordered_pose_weight * pose_loss
            + self.recover_ordered_delta_weight * delta_loss
        )
        ordered_ok = bool(
            (not self.require_recover_ordered_path)
            or (
                pose_loss <= float(self.recover_ordered_pose_threshold)
                and delta_loss <= float(self.recover_ordered_delta_threshold)
            )
        )
        return {
            "recover_ordered_path_available": True,
            "recover_ordered_target_index": int(target_index),
            "recover_ordered_horizon": int(horizon),
            "recover_ordered_pose_loss": pose_loss,
            "recover_ordered_delta_loss": delta_loss,
            "recover_ordered_loss": ordered_loss,
            "recover_ordered_pose_weight": float(self.recover_ordered_pose_weight),
            "recover_ordered_delta_weight": float(self.recover_ordered_delta_weight),
            "recover_ordered_pose_threshold": float(self.recover_ordered_pose_threshold),
            "recover_ordered_delta_threshold": float(self.recover_ordered_delta_threshold),
            "recover_ordered_ok": ordered_ok,
        }

    def _record_ordered_recovery_terms(self, terms):
        if not terms or not bool(terms.get("recover_ordered_path_available")):
            return
        for key, history in (
            ("recover_ordered_pose_loss", self._recover_ordered_pose_loss_history),
            ("recover_ordered_delta_loss", self._recover_ordered_delta_loss_history),
            ("recover_ordered_loss", self._recover_ordered_loss_history),
        ):
            value = terms.get(key)
            if value is not None and np.isfinite(float(value)):
                history.append(float(value))

    def get_nominal_rejoin_target(self, obs, candidate_chunk=None):
        del candidate_chunk
        if not self.safechunk_recover_enabled or not self.use_latest_nominal_for_rejoin:
            return {
                "available": False,
                "target_chunk": None,
                "safe_prefix_len": 0,
                "suppressed_reason": "no_latest_nominal",
                "nominal_rejoin_clearance": float("-inf"),
            }
        if self.latest_nominal_chunk is None:
            return {
                "available": False,
                "target_chunk": None,
                "safe_prefix_len": 0,
                "suppressed_reason": "no_latest_nominal",
                "nominal_rejoin_clearance": float("-inf"),
            }
        target = np.asarray(self.latest_nominal_chunk, dtype=np.float32).copy()
        if (
            self.suppress_stale_nominal_rejoin
            and self.blocked_nominal_chunk is not None
            and target.shape == np.asarray(self.blocked_nominal_chunk).shape
            and self.latest_nominal_step <= (self.blocked_nominal_step or -1)
            and np.allclose(target, self.blocked_nominal_chunk)
        ):
            return {
                "available": False,
                "target_chunk": None,
                "safe_prefix_len": 0,
                "suppressed_reason": "stale_blocked_nominal",
                "nominal_rejoin_clearance": float("-inf"),
            }
        try:
            acceptance = self.evaluate_candidate_acceptance(
                obs,
                target,
                candidate_type="nominal_rejoin_target",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Nominal rejoin target acceptance failed: %s", exc)
            return {
                "available": False,
                "target_chunk": None,
                "safe_prefix_len": 0,
                "suppressed_reason": "acceptance_unavailable",
                "nominal_rejoin_clearance": float("-inf"),
            }
        clearance = max(
            float(acceptance.get("immediate_clearance", float("-inf"))),
            float(acceptance.get("prefix_min_clearance", float("-inf"))),
        )
        safe_prefix_len = int(acceptance.get("safe_prefix_len", 0) or 0)
        if self.require_nominal_prefix_safe_for_rejoin and (
            safe_prefix_len < 1
            or clearance < float(self.nominal_rejoin_prefix_min_clearance)
            or acceptance.get("rejection_reason") == "immediate_below_hard_margin"
        ):
            return {
                "available": False,
                "target_chunk": None,
                "safe_prefix_len": safe_prefix_len,
                "suppressed_reason": "nominal_prefix_unsafe",
                "nominal_rejoin_clearance": float(clearance),
            }
        return {
            "available": True,
            "target_chunk": target,
            "safe_prefix_len": safe_prefix_len,
            "suppressed_reason": None,
            "nominal_rejoin_clearance": float(clearance),
        }

    def compute_nominal_rejoin_score(self, candidate_chunk, nominal_chunk, obs=None):
        candidate, _ = self._as_chunk(candidate_chunk)
        nominal, _ = self._as_chunk(nominal_chunk)
        valid = self._valid_control_indices(candidate)
        if not np.any(valid) or candidate.shape[0] == 0 or nominal.shape[0] == 0:
            return {
                "nominal_rejoin_score": 0.0,
                "recover_projection_on_nominal": 0.0,
                "recover_cosine_to_nominal": 0.0,
                "nominal_delta_norm": 0.0,
                "candidate_delta_norm": 0.0,
            }
        action_idx = self.controlled_action_indices[valid]
        delta_cand = candidate[0, action_idx].astype(np.float64, copy=False)
        delta_nom = nominal[0, action_idx].astype(np.float64, copy=False)
        if obs is not None:
            try:
                q = self.extract_current_q(obs, candidate)
                state_idx = self.controlled_state_indices[valid]
                valid_state = state_idx < q.shape[0]
                modes = self._control_mode_ids_for_state_indices(state_idx)
                absolute = valid_state & (modes == 0)
                if np.any(absolute):
                    delta_cand = delta_cand.copy()
                    delta_nom = delta_nom.copy()
                    delta_cand[absolute] = (
                        candidate[0, action_idx[absolute]].astype(np.float64, copy=False)
                        - q[state_idx[absolute]].astype(np.float64, copy=False)
                    )
                    delta_nom[absolute] = (
                        nominal[0, action_idx[absolute]].astype(np.float64, copy=False)
                        - q[state_idx[absolute]].astype(np.float64, copy=False)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Nominal rejoin score q extraction failed: %s", exc)
        eps = 1e-9
        dot = float(np.dot(delta_cand, delta_nom))
        nominal_norm = float(np.linalg.norm(delta_nom))
        candidate_norm = float(np.linalg.norm(delta_cand))
        projection = dot / (nominal_norm * nominal_norm + eps)
        cosine = dot / (candidate_norm * nominal_norm + eps)
        return {
            "nominal_rejoin_score": float(max(0.0, projection)),
            "recover_projection_on_nominal": float(projection),
            "recover_cosine_to_nominal": float(cosine),
            "nominal_delta_norm": float(nominal_norm),
            "candidate_delta_norm": float(candidate_norm),
        }

    def _recover_nominal_rejoin_terms(self, obs, candidate, *, record=False):
        target_info = self.get_nominal_rejoin_target(obs, candidate)
        rejoin_info = {
            "nominal_rejoin_score": 0.0,
            "recover_projection_on_nominal": 0.0,
            "recover_cosine_to_nominal": 0.0,
            "nominal_delta_norm": 0.0,
            "candidate_delta_norm": 0.0,
        }
        if target_info.get("available"):
            rejoin_info = self.compute_nominal_rejoin_score(
                candidate,
                target_info["target_chunk"],
                obs=obs,
            )
        progress_score, progress_available = self._candidate_progress_score(obs, candidate)
        effective_weight = self._recover_rejoin_weight_effective()
        if record:
            self._record_nominal_rejoin_target(
                target_info,
                rejoin_info if target_info.get("available") else None,
                progress_score=progress_score,
            )
        return target_info, rejoin_info, float(progress_score), bool(progress_available), float(effective_weight)

    def _recover_task_progress_cost(
        self,
        obs,
        candidate,
        nominal,
        action_idx,
        reference_chunk=None,
    ):
        q_seq = self.rollout_nominal_chunk(obs, candidate)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        safety_loss = float(np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum())
        action_deviation_loss = float(
            np.square(candidate[:, action_idx] - nominal[:, action_idx]).mean()
        ) if len(action_idx) else 0.0
        smoothness_loss = self._smoothness_loss(candidate, action_idx)
        target_info, rejoin_info, progress_score, progress_available, effective_weight = (
            self._recover_nominal_rejoin_terms(obs, candidate, record=False)
        )
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
        stalled_penalty = 5.0 if progress_score <= 0.0 and nominal_rejoin_score <= 0.0 else 0.0
        existing_loss = float(
            self.recover_safety_weight * safety_loss
            + self.recover_action_deviation_weight * action_deviation_loss
            + self.recover_smoothness_weight * smoothness_loss
            + self.recover_direction_alignment_weight * direction_loss
            + ordered_loss
        )
        recover_score_total = float(
            self.recover_task_progress_weight * progress_score
            + effective_weight * nominal_rejoin_score
            - stalled_penalty
        )
        total_loss = float(existing_loss - recover_score_total)
        return total_loss, {
            "safety_loss": safety_loss,
            "action_deviation_loss": action_deviation_loss,
            "smoothness_loss": smoothness_loss,
            "existing_optimization_loss": existing_loss,
            "total_loss": total_loss,
            "min_clearance": float(np.min(h_seq)),
            "recover_task_progress_score": float(progress_score),
            "progress_score_available": bool(progress_available),
            "recover_score_total": recover_score_total,
            "recover_rejoin_weight_effective": effective_weight,
            "recover_direction_alignment_weight": float(self.recover_direction_alignment_weight),
            **direction_terms,
            **ordered_terms,
            "recover_step_since_deform": int(self.recover_step_since_deform),
            "nominal_rejoin_available": bool(target_info.get("available")),
            "nominal_rejoin_suppressed_reason": target_info.get("suppressed_reason"),
            "nominal_rejoin_clearance": float(target_info.get("nominal_rejoin_clearance", float("-inf"))),
            "nominal_rejoin_safe_prefix_len": int(target_info.get("safe_prefix_len", 0) or 0),
            **rejoin_info,
        }

    def _recover_task_progress_cost_batch(
        self,
        obs,
        candidates,
        nominal,
        action_idx,
        reference_chunk=None,
    ):
        candidates = np.asarray(candidates, dtype=np.float32)
        q_seq_batch = self.rollout_nominal_chunk_batch(obs, candidates)
        safety_eval = self.evaluate_horizon_safety_batch(obs, q_seq_batch)
        h_seq = self._clearance_sequence_batch_from_eval(
            safety_eval,
            candidates.shape[0],
            candidates.shape[1],
        )
        safety_loss = np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum(axis=1)
        if len(action_idx):
            action_deviation_loss = np.square(
                candidates[:, :, action_idx] - nominal[None, :, action_idx]
            ).mean(axis=(1, 2))
        else:
            action_deviation_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        smoothness_loss = self._smoothness_loss_batch(candidates, action_idx)
        target_info = self.get_nominal_rejoin_target(obs)
        reference_available = reference_chunk is not None
        target_q_seq = None
        if reference_available:
            target_q_seq = self.rollout_nominal_chunk(obs, reference_chunk)
        elif target_info.get("available"):
            target_q_seq = self.rollout_nominal_chunk(obs, target_info["target_chunk"])
        effective_weight = self._recover_rejoin_weight_effective()
        progress_scores = []
        progress_available = []
        rejoin_infos = []
        ordered_terms_list = []
        for i, candidate in enumerate(candidates):
            progress_score, progress_ok = self._candidate_progress_score(obs, candidate)
            progress_scores.append(float(progress_score))
            progress_available.append(bool(progress_ok))
            if reference_available:
                rejoin_infos.append(
                    self.compute_nominal_rejoin_score(
                        candidate,
                        reference_chunk,
                        obs=obs,
                    )
                )
            elif target_info.get("available"):
                rejoin_infos.append(
                    self.compute_nominal_rejoin_score(
                        candidate,
                        target_info["target_chunk"],
                        obs=obs,
                    )
                )
            else:
                rejoin_infos.append(
                    {
                        "nominal_rejoin_score": 0.0,
                        "recover_projection_on_nominal": 0.0,
                        "recover_cosine_to_nominal": 0.0,
                        "nominal_delta_norm": 0.0,
                        "candidate_delta_norm": 0.0,
                    }
                )
            ordered_terms_list.append(
                self._ordered_recovery_path_terms(
                    q_seq_batch[i],
                    target_q_seq,
                    target_index=0,
                )
            )
        progress_scores = np.asarray(progress_scores, dtype=np.float32)
        nominal_rejoin_scores = np.asarray(
            [float(info.get("nominal_rejoin_score", 0.0)) for info in rejoin_infos],
            dtype=np.float32,
        )
        direction_terms_list = [
            self._recover_direction_alignment_terms(info) for info in rejoin_infos
        ]
        direction_loss = np.asarray(
            [float(info["recover_direction_loss"]) for info in direction_terms_list],
            dtype=np.float32,
        )
        ordered_loss = np.asarray(
            [float(info["recover_ordered_loss"]) for info in ordered_terms_list],
            dtype=np.float32,
        )
        stalled_penalty = np.where(
            (progress_scores <= 0.0) & (nominal_rejoin_scores <= 0.0),
            5.0,
            0.0,
        ).astype(np.float32)
        existing_loss = (
            self.recover_safety_weight * safety_loss
            + self.recover_action_deviation_weight * action_deviation_loss
            + self.recover_smoothness_weight * smoothness_loss
            + self.recover_direction_alignment_weight * direction_loss
            + ordered_loss
        )
        recover_score_total = (
            self.recover_task_progress_weight * progress_scores
            + effective_weight * nominal_rejoin_scores
            - stalled_penalty
        )
        total_loss = existing_loss - recover_score_total
        losses = []
        for i, rejoin_info in enumerate(rejoin_infos):
            losses.append(
                {
                    "safety_loss": float(safety_loss[i]),
                    "action_deviation_loss": float(action_deviation_loss[i]),
                    "smoothness_loss": float(smoothness_loss[i]),
                    "existing_optimization_loss": float(existing_loss[i]),
                    "total_loss": float(total_loss[i]),
                    "min_clearance": float(np.min(h_seq[i])),
                    "recover_task_progress_score": float(progress_scores[i]),
                    "progress_score_available": bool(progress_available[i]),
                    "recover_score_total": float(recover_score_total[i]),
                    "recover_rejoin_weight_effective": float(effective_weight),
                    "recover_direction_alignment_weight": float(self.recover_direction_alignment_weight),
                    **direction_terms_list[i],
                    **ordered_terms_list[i],
                    "recover_step_since_deform": int(self.recover_step_since_deform),
                    "nominal_rejoin_available": bool(target_info.get("available")),
                    "nominal_rejoin_suppressed_reason": target_info.get("suppressed_reason"),
                    "nominal_rejoin_clearance": float(target_info.get("nominal_rejoin_clearance", float("-inf"))),
                    "nominal_rejoin_safe_prefix_len": int(target_info.get("safe_prefix_len", 0) or 0),
                    "batched_optimizer": True,
                    **rejoin_info,
                }
            )
        return total_loss.astype(np.float32), losses

    def _safechunk_recovery_corridor_info(self):
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
            "delayed_rejoin_count": int(self.delayed_rejoin_count),
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

    def _tick_unsafe_recovery_cooldowns(self):
        if not self._unsafe_recovery_cooldowns:
            return
        for key in list(self._unsafe_recovery_cooldowns):
            remaining = int(self._unsafe_recovery_cooldowns[key]) - 1
            if remaining <= 0:
                self._unsafe_recovery_cooldowns.pop(key, None)
            else:
                self._unsafe_recovery_cooldowns[key] = remaining

    def make_recovery_target_key(self, target_chunk_or_q):
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

    def _make_recovery_path_key(self, recover_chunk, target_key=None):
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

    def _recovery_target_is_suppressed(self, target_key):
        if (
            not self.safechunk_recovery_corridor_enabled
            or not self.suppress_repeated_unsafe_recovery
            or target_key is None
        ):
            return False
        return int(self._unsafe_recovery_cooldowns.get(target_key, 0)) > 0

    def _mark_recovery_path_failure(self, target_key, path_key, reason):
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

    def _clear_recovery_path_failure_streak(self):
        self.recovery_path_failure_streak = 0
        self.delayed_rejoin_active = False
        self.delayed_rejoin_steps = 0

    def _activate_post_recovery_act_window(self):
        if not (
            self.safechunk_recovery_corridor_enabled
            and self.require_post_recovery_act_window
            and self.post_recovery_min_act_steps > 0
        ):
            return
        self.post_recovery_act_window_active = True
        self.post_recovery_act_steps_remaining = int(self.post_recovery_min_act_steps)
        self.post_recovery_act_window_count += 1

    def _post_recovery_act_window_info(self, *, interrupted=False):
        if interrupted and self.post_recovery_act_window_active:
            self.post_recovery_act_window_interrupted_count += 1
            self.post_recovery_act_window_active = False
            self.post_recovery_act_steps_remaining = 0
        return {
            "post_recovery_act_window_active": bool(
                self.post_recovery_act_window_active
            ),
            "post_recovery_act_steps_remaining": int(
                self.post_recovery_act_steps_remaining
            ),
            "post_recovery_act_window_interrupted": bool(interrupted),
        }

    def evaluate_recovery_path_safety(self, obs, recover_chunk, candidate_name="recover"):
        chunk, _ = self._as_chunk(recover_chunk)
        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        h_seq = np.asarray(h_seq, dtype=np.float32).reshape(-1)
        if h_seq.size == 0:
            h_seq = np.asarray([float("-inf")], dtype=np.float32)
        try:
            acceptance = self.evaluate_candidate_acceptance(
                obs,
                chunk,
                candidate_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Recovery path acceptance helper failed: %s", exc)
            acceptance = {}
        immediate = float(h_seq[0])
        path_min = float(np.min(h_seq))
        prefix_threshold = float(self.recover_prefix_min_clearance)
        safe_prefix_len = 0
        for value in h_seq:
            if float(value) >= prefix_threshold:
                safe_prefix_len += 1
            else:
                break
        if safe_prefix_len > 0:
            prefix_min = float(np.min(h_seq[:safe_prefix_len]))
        else:
            prefix_min = immediate
        immediate_safe = bool(immediate >= self.recover_immediate_hard_clearance)
        prefix_safe = bool(safe_prefix_len >= 1 and prefix_min >= prefix_threshold)
        path_margin_safe = bool(path_min >= self.recover_path_min_clearance)
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
            "recover_path_min_clearance": path_min,
            "recover_immediate_clearance": immediate,
            "recover_prefix_min_clearance": prefix_min,
            "safe_prefix_len": int(safe_prefix_len),
            "reject_reason": reject_reason,
            "candidate_name": candidate_name,
            "acceptance_type": acceptance.get("acceptance_type"),
            "acceptance_rejection_reason": acceptance.get("rejection_reason"),
        }

    def _recovery_terminal_rejoin_info(
        self,
        obs,
        candidate_chunk,
        context,
        rejoin_context,
        default_target_index=None,
    ):
        q_seq = self.rollout_nominal_chunk(obs, candidate_chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        min_clearance = float(safety_eval.get("min_clearance", float("-inf")))
        q_loss, target_index, q_time_ms = self._q_rejoin_loss(
            q_seq,
            nominal_q_seq=context.nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        if target_index is None:
            target_index = default_target_index
        qd_loss, qd_index, qd_time_ms = self._qd_rejoin_loss(
            q_seq,
            nominal_q_seq=context.nominal_q_seq,
            target_index=target_index,
            rejoin_context=rejoin_context,
        )
        ordered_target_index = self._ordered_recovery_start_index(
            target_index,
            q_seq.shape[0],
            context.nominal_q_seq,
        )
        ordered_terms = self._ordered_recovery_path_terms(
            q_seq,
            context.nominal_q_seq,
            target_index=ordered_target_index,
            rejoin_context=rejoin_context,
        )
        q_dist = self._sqrt_loss(q_loss)
        qd_dist = self._sqrt_loss(qd_loss)
        q_ok = bool(target_index is not None and q_dist < self.q_rejoin_threshold)
        qd_ok, qd_acceptance = self._qd_rejoin_acceptance(qd_index, qd_dist)
        return {
            "q_seq": q_seq,
            "eval": safety_eval,
            "min_clearance": min_clearance,
            "q_rejoin_loss": float(q_loss),
            "q_rejoin_dist": float(q_dist),
            "q_rejoin_ok": bool(q_ok),
            "target_index": target_index,
            "q_eval_time_ms": float(q_time_ms),
            "qd_rejoin_loss": float(qd_loss),
            "qd_rejoin_dist": float(qd_dist),
            "qd_rejoin_ok": bool(qd_ok),
            "qd_rejoin_index": qd_index,
            "qd_eval_time_ms": float(qd_time_ms),
            **qd_acceptance,
            **ordered_terms,
        }

    def _recovery_reject_reason(
        self,
        terminal_info,
        path_info,
        *,
        repeated_unsafe_target=False,
        task_progress_ok=True,
        direction_ok=True,
        ordered_ok=True,
    ):
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
        if bool(terminal_info.get("qd_rejoin_hard_failed", False)):
            return "qdot_rejoin_hard_failed"
        if bool(terminal_info.get("qd_rejoin_required", self.require_qd_rejoin)) and not bool(
            terminal_info.get("qd_rejoin_ok")
        ):
            return "qdot_rejoin_failed"
        if not bool(task_progress_ok):
            return "task_progress_failed"
        if not bool(direction_ok):
            return "direction_alignment_failed"
        if not bool(ordered_ok):
            return "ordered_path_failed"
        return None

    def _make_recovery_detour_candidates(self, obs, direct_chunk, action_idx):
        direct, _ = self._as_chunk(direct_chunk)
        candidates = []
        if direct.shape[0] == 0:
            return candidates
        action_idx = np.asarray(action_idx, dtype=np.int64)
        action_idx = action_idx[action_idx < direct.shape[1]]
        passthrough_idx = [
            i for i in range(direct.shape[1]) if i not in set(action_idx.tolist())
        ]

        def add(name, candidate):
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
            hold_action[self.controlled_action_indices[valid_hold]] = self._controlled_anchor(
                obs,
                direct,
                self.controlled_action_indices[valid_hold],
                self.controlled_state_indices[valid_hold],
            )

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

        if self.last_safe_q is not None and h > 1:
            last_q = np.asarray(self.last_safe_q, dtype=np.float32).reshape(-1)
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

        if self.last_safe_chunk is not None:
            safe_chunk = np.asarray(self.last_safe_chunk, dtype=np.float32)
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

    def _score_recovery_detour_candidate(
        self,
        candidate,
        direct_chunk,
        action_idx,
        path_info,
        terminal_info,
    ):
        candidate = np.asarray(candidate, dtype=np.float32)
        direct = np.asarray(direct_chunk, dtype=np.float32)
        action_idx = np.asarray(action_idx, dtype=np.int64)
        action_idx = action_idx[action_idx < candidate.shape[1]]
        if action_idx.size:
            action_norm = float(np.linalg.norm(candidate[:, action_idx] - direct[:, action_idx]))
        else:
            action_norm = float(np.linalg.norm(candidate - direct))
        clearance = float(path_info.get("recover_path_min_clearance", float("-inf")))
        if not np.isfinite(clearance):
            clearance = -1.0
        q_dist = float(terminal_info.get("q_rejoin_dist", float("inf")))
        qd_dist = float(terminal_info.get("qd_rejoin_dist", float("inf")))
        if not np.isfinite(q_dist):
            q_dist = 1e3
        if not np.isfinite(qd_dist):
            qd_dist = 1e3
        return float(
            self.detour_clearance_weight * clearance
            + 10.0 * float(path_info.get("safe_prefix_len", 0) or 0)
            - self.detour_task_rejoin_weight * (q_dist + 0.1 * qd_dist)
            - self.detour_action_norm_weight * action_norm
        )

    def evaluate_candidate_acceptance(self, obs, candidate_chunk, candidate_type):
        chunk, _ = self._as_chunk(candidate_chunk)
        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        h_seq = np.asarray(h_seq, dtype=np.float32).reshape(-1)
        if h_seq.size == 0:
            h_seq = np.asarray([float("-inf")], dtype=np.float32)
        immediate_clearance = float(h_seq[0])
        horizon_min_clearance = float(np.min(h_seq))
        hard = float(self.acceptance_hard_min_clearance)
        desired = float(self.acceptance_desired_min_clearance)
        prefix_threshold = float(self.prefix_min_clearance)
        full_required = bool(
            (candidate_type == "recover" and self.full_horizon_required_for_recover)
            or (candidate_type == "deform" and self.full_horizon_required_for_deform)
        )

        accepted = False
        acceptance_type = "rejected"
        rejection_reason = None
        safe_prefix_len = 0
        for value in h_seq:
            if float(value) >= prefix_threshold:
                safe_prefix_len += 1
            else:
                break

        if immediate_clearance < hard:
            acceptance_type = "emergency_brake"
            rejection_reason = "immediate_below_hard_margin"
        elif horizon_min_clearance >= desired:
            accepted = True
            acceptance_type = "full_horizon"
            safe_prefix_len = int(h_seq.size)
        elif full_required:
            rejection_reason = "horizon_below_desired_margin"
        elif self.allow_safe_prefix_execution:
            if safe_prefix_len >= self.min_safe_prefix_len:
                accepted = True
                acceptance_type = "safe_prefix"
            elif immediate_clearance >= prefix_threshold:
                accepted = True
                acceptance_type = "first_action_only"
                safe_prefix_len = 1
            else:
                rejection_reason = "no_safe_prefix"
        else:
            rejection_reason = "horizon_below_desired_margin"

        return {
            "accepted": bool(accepted),
            "acceptance_type": acceptance_type,
            "safe_prefix_len": int(safe_prefix_len),
            "immediate_clearance": immediate_clearance,
            "prefix_min_clearance": prefix_threshold,
            "horizon_min_clearance": horizon_min_clearance,
            "desired_min_clearance": desired,
            "hard_min_clearance": hard,
            "rejection_reason": rejection_reason,
            "candidate_type": candidate_type,
            "full_horizon_required": full_required,
            "rolling_replan_on_prefix": bool(self.rolling_replan_on_prefix),
            "safe_prefix_execution": bool(accepted and acceptance_type != "full_horizon"),
            "horizon_safe": bool(safety_eval.get("horizon_safe", horizon_min_clearance >= desired)),
        }

    def _truncate_chunk_to_safe_prefix(self, candidate_chunk, acceptance):
        chunk, was_single = self._as_chunk(candidate_chunk)
        safe = np.asarray(chunk, dtype=np.float32).copy()
        if safe.shape[0] == 0:
            return safe.reshape(np.asarray(candidate_chunk).shape)
        acceptance_type = acceptance.get("acceptance_type")
        prefix_len = int(acceptance.get("safe_prefix_len", 0) or 0)
        if acceptance_type == "first_action_only":
            prefix_len = 1
        if acceptance_type in {"safe_prefix", "first_action_only"}:
            prefix_len = max(1, min(prefix_len, safe.shape[0]))
            hold = safe[prefix_len - 1].copy()
            safe[prefix_len:] = hold
        return safe[0].copy() if was_single else safe.astype(np.asarray(candidate_chunk).dtype, copy=False)

    def _candidate_progress_score(self, obs, candidate_chunk):
        chunk, _ = self._as_chunk(candidate_chunk)
        valid = self._valid_control_indices(chunk)
        if not np.any(valid) or chunk.shape[0] == 0:
            return 0.0, False
        action_idx = self.controlled_action_indices[valid]
        state_idx = self.controlled_state_indices[valid]
        q = self.extract_current_q(obs, chunk)
        valid_state = state_idx < q.shape[0]
        if not np.any(valid_state):
            return 0.0, False
        action_idx = action_idx[valid_state]
        state_idx = state_idx[valid_state]
        return float(np.linalg.norm(chunk[0, action_idx] - q[state_idx])), True

    def _score_accepted_candidate(self, obs, candidate, nominal, acceptance, candidate_type="deform"):
        progress_score, progress_available = self._candidate_progress_score(obs, candidate)
        deformation_norm = self._controlled_deformation_norm(candidate, nominal)
        recover_extra = {}
        task_weight = 2.0
        nominal_rejoin_score = 0.0
        rejoin_weight = 0.0
        if candidate_type == "recover" and self.safechunk_recover_enabled:
            target_info, rejoin_info, progress_score, progress_available, rejoin_weight = (
                self._recover_nominal_rejoin_terms(obs, candidate, record=True)
            )
            nominal_rejoin_score = float(rejoin_info.get("nominal_rejoin_score", 0.0))
            recover_extra = {
                "recover_task_progress_score": float(progress_score),
                "recover_rejoin_weight_effective": float(rejoin_weight),
                "recover_step_since_deform": int(self.recover_step_since_deform),
                "nominal_rejoin_available": bool(target_info.get("available")),
                "nominal_rejoin_suppressed_reason": target_info.get("suppressed_reason"),
                "nominal_rejoin_clearance": float(target_info.get("nominal_rejoin_clearance", float("-inf"))),
                "nominal_rejoin_safe_prefix_len": int(target_info.get("safe_prefix_len", 0) or 0),
                **rejoin_info,
            }
            task_weight = float(self.recover_task_progress_weight)
        score = (
            1000.0 * float(bool(acceptance.get("accepted")))
            + 10.0 * float(acceptance.get("safe_prefix_len", 0) or 0)
            + 5.0 * float(acceptance.get("immediate_clearance", 0.0) or 0.0)
            + task_weight * progress_score
            + rejoin_weight * nominal_rejoin_score
            - 0.1 * deformation_norm
        )
        if candidate_type == "recover" and progress_score <= 0.0 and nominal_rejoin_score <= 0.0:
            score -= 5.0
        recover_extra["recover_score_total"] = float(score)
        return float(score), {
            "progress_score": float(progress_score),
            "progress_score_available": bool(progress_available),
            "deformation_norm": float(deformation_norm),
            **recover_extra,
        }

    def _write_state_tracking_actions(
        self,
        seed_chunk,
        q_start,
        target_q_seq,
        action_idx,
        state_idx,
    ):
        chunk = np.asarray(seed_chunk, dtype=np.float32).copy()
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            return chunk
        q_prev = np.asarray(q_start, dtype=np.float32).reshape(-1).copy()
        target_q_seq = np.asarray(target_q_seq, dtype=np.float32)
        if target_q_seq.ndim == 1:
            target_q_seq = target_q_seq.reshape(1, -1)
        if target_q_seq.ndim != 2 or target_q_seq.shape[0] == 0:
            return chunk
        action_idx = np.asarray(action_idx, dtype=np.int64).reshape(-1)
        state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
        valid = (
            (action_idx < chunk.shape[1])
            & (state_idx < q_prev.shape[0])
            & (state_idx < target_q_seq.shape[1])
        )
        if not np.any(valid):
            return chunk
        action_idx = action_idx[valid]
        state_idx = state_idx[valid]
        modes = self._control_mode_ids_for_state_indices(state_idx)
        absolute = modes == 0
        delta = modes == 1
        velocity = modes == 2
        horizon = min(chunk.shape[0], target_q_seq.shape[0])
        dt = max(float(self.dt), 1e-9)
        for k in range(horizon):
            desired = target_q_seq[k]
            selected = chunk[k, action_idx].copy()
            if np.any(absolute):
                selected[absolute] = desired[state_idx[absolute]]
            if np.any(delta):
                selected[delta] = desired[state_idx[delta]] - q_prev[state_idx[delta]]
            if np.any(velocity):
                selected[velocity] = (
                    desired[state_idx[velocity]] - q_prev[state_idx[velocity]]
                ) / dt
            chunk[k, action_idx] = selected
            q_prev[state_idx] = desired[state_idx]
        return chunk

    def _make_task_progress_recover_chunk(
        self,
        q_start,
        current_chunk,
        action_idx,
        context=None,
        default_target_index=None,
    ):
        h = min(current_chunk.shape[0], self.return_horizon)
        recover_chunk = np.asarray(current_chunk[:h], dtype=np.float32).copy()
        target_index = min(self.min_rejoin_offset, max(0, current_chunk.shape[0] - 1))
        if action_idx.size:
            q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)
            valid = (
                (self.controlled_action_indices < recover_chunk.shape[1])
                & (self.controlled_state_indices < q_start.shape[0])
            )
            allowed_actions = set(np.asarray(action_idx, dtype=np.int64).reshape(-1).tolist())
            if allowed_actions:
                valid &= np.asarray(
                    [idx in allowed_actions for idx in self.controlled_action_indices],
                    dtype=np.bool_,
                )
            local_action_idx = self.controlled_action_indices[valid]
            state_idx = self.controlled_state_indices[valid]
            if local_action_idx.size:
                nominal_q_seq = None if context is None else getattr(context, "nominal_q_seq", None)
                if nominal_q_seq is not None:
                    nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
                if (
                    nominal_q_seq is not None
                    and nominal_q_seq.ndim == 2
                    and nominal_q_seq.shape[0] > 0
                    and np.all(state_idx < nominal_q_seq.shape[1])
                ):
                    if default_target_index is not None:
                        target_index = int(np.clip(
                            int(default_target_index),
                            0,
                            max(0, nominal_q_seq.shape[0] - 1),
                        ))
                    elif nominal_q_seq.shape[0] > self.min_rejoin_offset:
                        future = nominal_q_seq[self.min_rejoin_offset :, state_idx]
                        _loss, target_index = self._nearest_future_loss(
                            q_start[state_idx],
                            future,
                            weights=None,
                            start_index=self.min_rejoin_offset,
                        )
                    else:
                        target_index = 0
                    target_rows = []
                    for k in range(h):
                        src_idx = min(target_index + k, nominal_q_seq.shape[0] - 1)
                        target_rows.append(nominal_q_seq[src_idx].copy())
                    if target_rows:
                        recover_chunk = self._write_state_tracking_actions(
                            recover_chunk,
                            q_start,
                            np.stack(target_rows, axis=0).astype(np.float32),
                            local_action_idx,
                            state_idx,
                        )
                else:
                    modes = self._control_mode_ids_for_state_indices(state_idx)
                    first = np.zeros(local_action_idx.shape, dtype=recover_chunk.dtype)
                    absolute = modes == 0
                    if np.any(absolute):
                        first[absolute] = q_start[state_idx[absolute]]
                    recover_chunk[0, local_action_idx] = first
        passthrough_idx = [
            i for i in range(current_chunk.shape[1]) if i not in set(action_idx.tolist())
        ]
        recover_chunk[:, passthrough_idx] = current_chunk[:h, passthrough_idx]
        return recover_chunk, target_index

    def _recover_seed_feasible(self, obs, seed_chunk):
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

    def _make_return_seed_chunk(self, context, q_start, current_chunk, action_idx):
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

    def _splice_explicit_return_chunk(
        self,
        nominal,
        yield_chunk,
        return_chunk,
        action_idx,
        target_index,
    ):
        full = np.asarray(nominal, dtype=np.float32).copy()
        y = min(yield_chunk.shape[0], full.shape[0])
        full[:y] = yield_chunk[:y]
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

    def _explicit_return_info(
        self,
        obs,
        chunk,
        nominal,
        context,
        recovery_phase,
        yield_chunk,
        yield_losses,
        yield_eval,
        yield_accepted,
        return_chunk,
        return_losses,
        return_eval,
        return_accepted,
        return_rejoin_loss,
        return_target_index,
        fallback_used,
        rejection_cause,
    ):
        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        min_clearance = float(safety_eval.get("min_clearance", float("-inf")))
        yield_min_clearance = float(yield_eval.get("min_clearance", float("-inf")))
        return_min_clearance = (
            None
            if return_eval is None
            else float(return_eval.get("min_clearance", float("-inf")))
        )
        recover_losses = dict(return_losses or {})
        if "recover_path_safe" in recover_losses:
            return_safe = bool(
                recover_losses.get("recover_path_safe")
                and recover_losses.get("recover_immediate_safe", True)
                and (
                    not self.require_safe_corridor_for_recovery_complete
                    or recover_losses.get("recover_prefix_safe", True)
                )
            )
        else:
            return_safe = bool(
                return_min_clearance is not None
                and return_min_clearance >= self._acceptance_clearance_threshold()
            )
        q_rejoin_dist = self._sqrt_loss(return_rejoin_loss)
        qd_rejoin_dist = float(
            recover_losses.get(
                "qd_rejoin_dist",
                self._sqrt_loss(recover_losses.get("return_qd_rejoin_loss", 0.0)),
            )
        )
        qd_rejoin_ok = bool(
            recover_losses.get(
                "qd_rejoin_ok",
                self._qd_rejoin_acceptance(
                    recover_losses.get(
                        "qd_rejoin_index",
                        recover_losses.get("return_qd_rejoin_index"),
                    ),
                    qd_rejoin_dist,
                )[0],
            )
        )
        qd_acceptance = self._qd_rejoin_acceptance(
            recover_losses.get(
                "qd_rejoin_index",
                recover_losses.get("return_qd_rejoin_index"),
            ),
            qd_rejoin_dist,
        )[1]
        return_rejoin_ok = bool(
            return_target_index is not None
            and q_rejoin_dist < self.q_rejoin_threshold
            and qd_rejoin_ok
        )
        is_safe = bool(yield_accepted and (return_accepted or return_safe))
        is_recoverable = bool(return_accepted and return_rejoin_ok)
        if rejection_cause is None and not return_accepted:
            rejection_cause = self._optimized_reject_reason_from_flags(
                not return_safe,
                not return_rejoin_ok,
            )
        context.target_rejoin_index = None if return_target_index is None else int(return_target_index)
        public_phase = (
            "horizon_deform" if recovery_phase in {"yield", "yield_deform"} else recovery_phase
        )
        if public_phase in {"return", "resume_act", "return_to_cached_motion"}:
            public_phase = "recover"
        replan_info = self._safechunk_replan_info()
        replan_info.update(
            {k: recover_losses[k] for k in recover_losses if k in replan_info}
        )
        info = dict(safety_eval)
        info.update(
            {
                "mode": public_phase,
                "deform_mode": public_phase,
                "deformation_source": "explicit_recover_deform",
                "recovery_mode": public_phase,
                "recover_mode": public_phase,
                "recovery_phase": public_phase,
                "cached_motion_active": bool(context.active),
                "recovery_context_active": bool(context.active),
                "start_chunk_index": context.start_chunk_index,
                "trigger_step": context.trigger_step,
                "target_rejoin_index": context.target_rejoin_index,
                "deform_min_clearance_stage": yield_min_clearance,
                "deform_stage_accepted": bool(yield_accepted),
                "recover_min_clearance": return_min_clearance,
                "recover_rejoin_loss": float(return_rejoin_loss),
                "recover_target_index": None if return_target_index is None else int(return_target_index),
                "recover_accepted": bool(return_accepted),
                "recover_required": bool(recover_losses.get("recover_required", True)),
                "recovery_candidate_class": recover_losses.get(
                    "recovery_candidate_class"
                ),
                "recover_reject_reason": recover_losses.get(
                    "recover_reject_reason",
                    None if return_accepted else rejection_cause,
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
                "recover_path_safe": recover_losses.get("recover_path_safe"),
                "recover_immediate_safe": recover_losses.get(
                    "recover_immediate_safe"
                ),
                "recover_prefix_safe": recover_losses.get(
                    "recover_prefix_safe"
                ),
                "recover_safe_prefix_len": recover_losses.get(
                    "recover_safe_prefix_len"
                ),
                "recover_target_key": recover_losses.get("recover_target_key"),
                "recovery_path_failure_streak": recover_losses.get(
                    "recovery_path_failure_streak",
                    self.recovery_path_failure_streak,
                ),
                "direct_rejoin_attempted": recover_losses.get(
                    "direct_rejoin_attempted", False
                ),
                "direct_rejoin_rejected": recover_losses.get(
                    "direct_rejoin_rejected", False
                ),
                "detour_rejoin_attempted": recover_losses.get(
                    "detour_rejoin_attempted", False
                ),
                "detour_rejoin_accepted": recover_losses.get(
                    "detour_rejoin_accepted", False
                ),
                "delayed_rejoin_active": recover_losses.get(
                    "delayed_rejoin_active", self.delayed_rejoin_active
                ),
                "delayed_rejoin_steps": recover_losses.get(
                    "delayed_rejoin_steps", self.delayed_rejoin_steps
                ),
                "repeated_unsafe_target": recover_losses.get(
                    "repeated_unsafe_target", False
                ),
                "post_recovery_act_window_active": recover_losses.get(
                    "post_recovery_act_window_active",
                    self.post_recovery_act_window_active,
                ),
                "post_recovery_act_steps_remaining": recover_losses.get(
                    "post_recovery_act_steps_remaining",
                    self.post_recovery_act_steps_remaining,
                ),
                "post_recovery_act_window_interrupted": recover_losses.get(
                    "post_recovery_act_window_interrupted", False
                ),
                "resumed_from_recover_index": (
                    None if not return_accepted or return_target_index is None else int(return_target_index)
                ),
                "act_resume_index": (
                    None if not return_accepted or return_target_index is None else int(return_target_index)
                ),
                "act_resume_supported": False,
                "act_resume_note": (
                    "TODO: executor replans each outer step; use recover_target_index "
                    "when a persistent ACT chunk cursor is added."
                ),
                "deform_safe": bool(return_safe),
                "is_safe": bool(return_safe),
                "is_recoverable": bool(is_recoverable),
                "safety_rejected": not bool(return_safe),
                "recovery_rejected": not bool(is_recoverable),
                "rejection_cause": rejection_cause,
                "fallback_used": bool(fallback_used),
                "min_clearance": min_clearance,
                "deform_min_clearance": min_clearance,
                "best_min_clearance": max(
                    yield_min_clearance,
                    return_min_clearance if return_min_clearance is not None else float("-inf"),
                ),
                "required_min_clearance": float(self.min_clearance),
                "clearance_gap": float(
                    self.min_clearance
                    - max(
                        yield_min_clearance,
                        return_min_clearance if return_min_clearance is not None else float("-inf"),
                    )
                ),
                "safety_loss": float(return_losses.get("safety_loss", yield_losses.get("safety_loss", 0.0))),
                "action_deviation_loss": float(return_losses.get("action_deviation_loss", yield_losses.get("action_deviation_loss", 0.0))),
                "path_loss": 0.0,
                "rejoin_loss": float(return_rejoin_loss),
                "q_rejoin_loss": float(return_rejoin_loss),
                "q_rejoin_dist": q_rejoin_dist,
                "q_rejoin_threshold": float(self.q_rejoin_threshold),
                "q_rejoin_index": None if return_target_index is None else int(return_target_index),
                "qd_rejoin_loss": float(return_losses.get("qd_rejoin_loss", return_losses.get("return_qd_rejoin_loss", 0.0))),
                "qd_rejoin_dist": qd_rejoin_dist,
                "qd_rejoin_threshold": float(self.qd_rejoin_threshold),
                "qd_rejoin_index": return_losses.get("qd_rejoin_index", return_losses.get("return_qd_rejoin_index")),
                "qd_rejoin_ok": bool(qd_rejoin_ok),
                **qd_acceptance,
                "return_qd_rejoin_loss": return_losses.get("return_qd_rejoin_loss"),
                "return_qd_rejoin_index": return_losses.get("return_qd_rejoin_index"),
                "rejoin_index": None if return_target_index is None else int(return_target_index),
                "j_best": None if return_target_index is None else int(return_target_index),
                "recover_retries": int(context.recover_retries),
                "max_recover_retries": int(self.max_return_retries),
                "deform_chunk_length": int(
                    0 if yield_chunk is None else np.asarray(yield_chunk).shape[0]
                ),
                "recover_chunk_length": int(
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
                "rejoin_q_eval_time_ms": float(return_losses.get("rejoin_q_eval_time_ms", 0.0)),
                "rejoin_qd_eval_time_ms": float(return_losses.get("rejoin_qd_eval_time_ms", 0.0)),
                "smoothness_loss": float(return_losses.get("smoothness_loss", yield_losses.get("smoothness_loss", 0.0))),
                "existing_optimization_loss": float(return_losses.get("existing_optimization_loss", return_losses.get("total_loss", yield_losses.get("total_loss", 0.0)))),
                "total_loss": float(return_losses.get("total_loss", yield_losses.get("total_loss", 0.0))),
                "deformation_norm": self._controlled_deformation_norm(chunk, nominal),
                "recover_projection_on_nominal": return_losses.get("recover_projection_on_nominal"),
                "recover_cosine_to_nominal": return_losses.get("recover_cosine_to_nominal"),
                "recover_direction_cosine": return_losses.get("recover_direction_cosine"),
                "recover_direction_cosine_threshold": return_losses.get("recover_direction_cosine_threshold"),
                "recover_direction_loss": return_losses.get("recover_direction_loss"),
                "recover_direction_ok": return_losses.get("recover_direction_ok"),
                "recover_direction_alignment_available": return_losses.get("recover_direction_alignment_available"),
                "recover_direction_alignment_weight": return_losses.get("recover_direction_alignment_weight"),
                "recover_ordered_path_available": return_losses.get("recover_ordered_path_available"),
                "recover_ordered_target_index": return_losses.get("recover_ordered_target_index"),
                "recover_ordered_horizon": return_losses.get("recover_ordered_horizon"),
                "recover_ordered_pose_loss": return_losses.get("recover_ordered_pose_loss"),
                "recover_ordered_delta_loss": return_losses.get("recover_ordered_delta_loss"),
                "recover_ordered_loss": return_losses.get("recover_ordered_loss"),
                "recover_ordered_pose_weight": return_losses.get("recover_ordered_pose_weight"),
                "recover_ordered_delta_weight": return_losses.get("recover_ordered_delta_weight"),
                "recover_ordered_pose_threshold": return_losses.get("recover_ordered_pose_threshold"),
                "recover_ordered_delta_threshold": return_losses.get("recover_ordered_delta_threshold"),
                "recover_ordered_ok": return_losses.get("recover_ordered_ok"),
                "nominal_rejoin_score": return_losses.get("nominal_rejoin_score"),
                "nominal_rejoin_available": return_losses.get("nominal_rejoin_available"),
                "nominal_rejoin_suppressed_reason": return_losses.get("nominal_rejoin_suppressed_reason"),
                "nominal_rejoin_clearance": return_losses.get("nominal_rejoin_clearance"),
                "nominal_rejoin_safe_prefix_len": return_losses.get("nominal_rejoin_safe_prefix_len"),
                "recover_task_progress_score": return_losses.get("recover_task_progress_score"),
                "recover_score_total": return_losses.get("recover_score_total"),
                "recover_rejoin_weight_effective": return_losses.get("recover_rejoin_weight_effective"),
                "recover_step_since_deform": return_losses.get("recover_step_since_deform"),
                "cem_iterations_run": return_losses.get(
                    "cem_iterations_run",
                    yield_losses.get("cem_iterations_run"),
                ),
                "cem_early_stopped": return_losses.get(
                    "cem_early_stopped",
                    yield_losses.get("cem_early_stopped"),
                ),
                "cem_max_iters": return_losses.get(
                    "cem_max_iters",
                    yield_losses.get("cem_max_iters"),
                ),
                "cem_population": return_losses.get(
                    "cem_population",
                    yield_losses.get("cem_population"),
                ),
                "yield_cem_iterations_run": yield_losses.get("cem_iterations_run"),
                "yield_cem_early_stopped": yield_losses.get("cem_early_stopped"),
                "return_cem_iterations_run": return_losses.get("cem_iterations_run"),
                "return_cem_early_stopped": return_losses.get("cem_early_stopped"),
            }
        )
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

    def _acceptance_clearance_threshold(self):
        return float(self.min_clearance - self.acceptance_clearance_tol)

    def rollout_chunk(self, action_chunk, q_current=None, qd_current=None, obs=None):
        del qd_current
        return self.rollout_nominal_chunk(self._obs_with_q(obs, q_current), action_chunk)

    def compute_horizon_clearance(self, q_seq, human_state=None, obs=None):
        eval_obs = obs if obs is not None else (human_state if human_state is not None else {})
        safety = self.evaluate_horizon_safety(eval_obs, q_seq)
        return self._clearance_sequence_from_eval(safety, np.asarray(q_seq).shape[0])

    def _optimized_seed_chunks(self, obs, chunk, safety_info):
        seeds = [chunk.copy()]
        for _, candidate in self._make_chunk_deformation_candidates(obs, chunk, safety_info or {}):
            if not any(np.allclose(candidate, seen) for seen in seeds):
                seeds.append(candidate.copy())
        return seeds

    def _optimized_deformation_cost(
        self,
        obs,
        candidate,
        nominal,
        nominal_q_seq,
        nominal_ee_seq,
        human_state,
        rejoin_context=None,
    ):
        del human_state
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)
        q_seq = self.rollout_nominal_chunk(obs, candidate)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        valid = self._valid_control_indices(nominal)
        action_idx = self.controlled_action_indices[valid]

        safety_loss = float(np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum())
        action_deviation_loss = float(
            np.square(candidate[:, action_idx] - nominal[:, action_idx]).mean()
        ) if action_idx.size else 0.0
        path_loss = self.nominal_path_deviation_loss(q_seq, nominal_q_seq)
        rejoin_loss = 0.0
        j_best = None
        rejoin_space = None
        rejoin_q_eval_time_ms = 0.0
        if self.recoverable_deform_enabled:
            if self.inner_rejoin_metric == "ee_pose":
                (
                    rejoin_loss,
                    j_best,
                    rejoin_q_eval_time_ms,
                    _ee_available,
                ) = self._ee_rejoin_loss(
                    q_seq,
                    nominal_q_seq=nominal_q_seq,
                    nominal_ee_seq=nominal_ee_seq,
                    rejoin_context=rejoin_context,
                )
                rejoin_space = "ee_pose"
            else:
                rejoin_loss, j_best, rejoin_q_eval_time_ms = self._q_rejoin_loss(
                    q_seq,
                    nominal_q_seq=nominal_q_seq,
                    rejoin_context=rejoin_context,
                )
                rejoin_space = "q_state"
        smoothness_loss = self._smoothness_loss(candidate, action_idx)
        existing_optimization_loss = float(
            self.lambda_safety * safety_loss
            + self.lambda_action * action_deviation_loss
            + self.lambda_path * path_loss
            + self.lambda_smooth * smoothness_loss
        )
        total_loss = existing_optimization_loss
        if self.recoverable_deform_enabled:
            total_loss = float(total_loss + self.lambda_rejoin * rejoin_loss)
        losses = {
            "safety_loss": safety_loss,
            "action_deviation_loss": action_deviation_loss,
            "path_loss": path_loss,
            "existing_optimization_loss": existing_optimization_loss,
            "rejoin_loss": rejoin_loss,
            "smoothness_loss": smoothness_loss,
            "total_loss": total_loss,
            "min_clearance": float(np.min(h_seq)),
            "j_best": j_best,
            "rejoin_space": rejoin_space,
            "inner_rejoin_metric": self.inner_rejoin_metric,
            "final_rejoin_metric": self.final_rejoin_metric,
            "rejoin_q_eval_time_ms": float(rejoin_q_eval_time_ms),
            "ee_nom_cache_time_ms": float(
                rejoin_context.get("ee_nom_cache_time_ms", 0.0)
            ),
            "ee_final_check_time_ms": 0.0,
        }
        if rejoin_space == "q_state":
            losses["q_rejoin_loss"] = rejoin_loss
            losses["q_rejoin_dist"] = self._sqrt_loss(rejoin_loss)
            losses["q_rejoin_index"] = j_best
        return total_loss, losses

    def _clearance_sequence_batch_from_eval(self, safety_eval, batch: int, horizon: int):
        h_seq = np.asarray(
            safety_eval.get("min_clearances", safety_eval.get("clearances", [])),
            dtype=np.float32,
        )
        if h_seq.shape == (batch, horizon):
            return h_seq
        min_h = np.asarray(safety_eval.get("min_clearance", np.inf), dtype=np.float32).reshape(-1)
        if min_h.size == 1:
            min_h = np.full(batch, float(min_h[0]), dtype=np.float32)
        if min_h.size != batch:
            min_h = np.full(batch, np.inf, dtype=np.float32)
        return np.repeat(min_h[:, None], horizon, axis=1).astype(np.float32)

    def nominal_path_deviation_loss_batch(self, q_seq_batch, nominal_q_seq):
        if nominal_q_seq is None:
            return np.zeros(q_seq_batch.shape[0], dtype=np.float32)
        q_seq_batch = np.asarray(q_seq_batch, dtype=np.float32)
        nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        horizon = min(q_seq_batch.shape[1], nominal_q_seq.shape[0])
        if horizon == 0:
            return np.zeros(q_seq_batch.shape[0], dtype=np.float32)
        q_dim = min(q_seq_batch.shape[2], nominal_q_seq.shape[1])
        delta = q_seq_batch[:, :horizon, :q_dim] - nominal_q_seq[None, :horizon, :q_dim]
        return np.square(delta).mean(axis=(1, 2)).astype(np.float32)

    def _smoothness_loss_batch(self, chunks, action_idx):
        chunks = np.asarray(chunks, dtype=np.float32)
        if len(action_idx) == 0 or chunks.shape[1] <= 1:
            return np.zeros(chunks.shape[0], dtype=np.float32)
        controlled = chunks[:, :, action_idx]
        velocity_loss = np.square(np.diff(controlled, axis=1)).mean(axis=(1, 2))
        if chunks.shape[1] <= 2:
            return velocity_loss.astype(np.float32)
        acc = controlled[:, 2:, :] - 2.0 * controlled[:, 1:-1, :] + controlled[:, :-2, :]
        return (velocity_loss + 0.5 * np.square(acc).mean(axis=(1, 2))).astype(np.float32)

    def _q_rejoin_loss_batch(self, q_seq_batch, nominal_q_seq=None, rejoin_context=None):
        t0 = time.perf_counter()
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq)
        q_seq_batch = np.asarray(q_seq_batch, dtype=np.float32)
        batch = q_seq_batch.shape[0]
        state_idx = rejoin_context.get("q_state_indices")
        future = rejoin_context.get("q_nom_future")
        weights = rejoin_context.get("q_weights")
        if state_idx is None or future is None or q_seq_batch.shape[1] == 0:
            return (
                np.full(batch, np.inf, dtype=np.float32),
                [None] * batch,
                (time.perf_counter() - t0) * 1000.0,
            )
        valid = state_idx < q_seq_batch.shape[2]
        if not np.all(valid):
            state_idx = state_idx[valid]
            future = future[:, valid]
            weights = None if weights is None else weights[valid]
        if state_idx.size == 0:
            return (
                np.full(batch, np.inf, dtype=np.float32),
                [None] * batch,
                (time.perf_counter() - t0) * 1000.0,
            )
        final_state = q_seq_batch[:, -1, :][:, state_idx]
        diff = final_state[:, None, :] - future[None, :, :]
        if weights is not None:
            diff = diff * np.asarray(weights, dtype=np.float32).reshape(1, 1, -1)
        losses_by_index = np.square(diff).sum(axis=2)
        j_best = np.argmin(losses_by_index, axis=1)
        losses = losses_by_index[np.arange(batch), j_best].astype(np.float32)
        start_index = int(
            rejoin_context.get("q_nom_future_start_index", self.min_rejoin_offset)
        )
        indices = [
            None if not np.isfinite(losses[i]) else int(j_best[i] + start_index)
            for i in range(batch)
        ]
        return losses, indices, (time.perf_counter() - t0) * 1000.0

    def _optimized_deformation_cost_batch(
        self,
        obs,
        candidates,
        nominal,
        nominal_q_seq,
        nominal_ee_seq,
        human_state,
        rejoin_context=None,
    ):
        del human_state
        if self.recoverable_deform_enabled and self.inner_rejoin_metric == "ee_pose":
            raise NotImplementedError("Batched EE-pose rejoin is not implemented")
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)
        candidates = np.asarray(candidates, dtype=np.float32)
        rollout_t0 = time.perf_counter()
        q_seq_batch = self.rollout_nominal_chunk_batch(obs, candidates)
        rollout_time_ms = 1000.0 * (time.perf_counter() - rollout_t0)
        safety_eval = self.evaluate_horizon_safety_batch(obs, q_seq_batch)
        h_seq = self._clearance_sequence_batch_from_eval(
            safety_eval,
            candidates.shape[0],
            candidates.shape[1],
        )
        valid = self._valid_control_indices(nominal)
        action_idx = self.controlled_action_indices[valid]
        safety_loss = np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum(axis=1)
        if action_idx.size:
            action_deviation_loss = np.square(
                candidates[:, :, action_idx] - nominal[None, :, action_idx]
            ).mean(axis=(1, 2))
        else:
            action_deviation_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        path_loss = self.nominal_path_deviation_loss_batch(q_seq_batch, nominal_q_seq)
        smoothness_loss = self._smoothness_loss_batch(candidates, action_idx)
        rejoin_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        j_best = [None] * candidates.shape[0]
        rejoin_q_eval_time_ms = 0.0
        rejoin_space = None
        if self.recoverable_deform_enabled:
            rejoin_loss, j_best, rejoin_q_eval_time_ms = self._q_rejoin_loss_batch(
                q_seq_batch,
                nominal_q_seq=nominal_q_seq,
                rejoin_context=rejoin_context,
            )
            rejoin_space = "q_state"
        existing_optimization_loss = (
            self.lambda_safety * safety_loss
            + self.lambda_action * action_deviation_loss
            + self.lambda_path * path_loss
            + self.lambda_smooth * smoothness_loss
        )
        total_loss = existing_optimization_loss.copy()
        if self.recoverable_deform_enabled:
            total_loss = total_loss + self.lambda_rejoin * rejoin_loss
        per_rejoin_ms = float(rejoin_q_eval_time_ms) / max(1, candidates.shape[0])
        losses = []
        for i in range(candidates.shape[0]):
            item = {
                "safety_loss": float(safety_loss[i]),
                "action_deviation_loss": float(action_deviation_loss[i]),
                "path_loss": float(path_loss[i]),
                "existing_optimization_loss": float(existing_optimization_loss[i]),
                "rejoin_loss": float(rejoin_loss[i]),
                "smoothness_loss": float(smoothness_loss[i]),
                "total_loss": float(total_loss[i]),
                "min_clearance": float(np.min(h_seq[i])),
                "j_best": j_best[i],
                "rejoin_space": rejoin_space,
                "inner_rejoin_metric": self.inner_rejoin_metric,
                "final_rejoin_metric": self.final_rejoin_metric,
                "rejoin_q_eval_time_ms": per_rejoin_ms,
                "ee_nom_cache_time_ms": float(rejoin_context.get("ee_nom_cache_time_ms", 0.0)),
                "ee_final_check_time_ms": 0.0,
                "batched_optimizer": True,
                "jax_batched_optimizer": bool(self._jax_optimizer_ready()),
                "jax_rollout_time_ms": float(rollout_time_ms) / max(1, candidates.shape[0]),
            }
            if rejoin_space == "q_state":
                item["q_rejoin_loss"] = float(rejoin_loss[i])
                item["q_rejoin_dist"] = self._sqrt_loss(float(rejoin_loss[i]))
                item["q_rejoin_index"] = j_best[i]
            losses.append(item)
        return total_loss.astype(np.float32), losses

    def nominal_path_deviation_loss(self, q_seq, nominal_q_seq):
        if nominal_q_seq is None:
            return 0.0
        q_seq = np.asarray(q_seq, dtype=np.float32)
        nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        horizon = min(q_seq.shape[0], nominal_q_seq.shape[0])
        if horizon == 0:
            return 0.0
        valid = self.controlled_state_indices < min(q_seq.shape[1], nominal_q_seq.shape[1])
        state_idx = self.controlled_state_indices[valid]
        if state_idx.size == 0:
            return 0.0
        delta = q_seq[:horizon, state_idx] - nominal_q_seq[:horizon, state_idx]
        return float(np.square(delta).mean())

    def _make_rejoin_context(self, nominal_q_seq, nominal_ee_seq=None):
        context = {
            "nominal_q_seq": None,
            "q_state_indices": None,
            "q_weights": None,
            "q_nom_future": None,
            "ee_nom_seq": None,
            "ee_nom_future": None,
            "ee_nom_cache_time_ms": 0.0,
        }
        if not self.recoverable_deform_enabled or nominal_q_seq is None:
            return context

        nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        context["nominal_q_seq"] = nominal_q_seq
        if nominal_q_seq.shape[0] > self.min_rejoin_offset:
            valid = self.controlled_state_indices < nominal_q_seq.shape[1]
            state_idx = self.controlled_state_indices[valid]
            if state_idx.size:
                context["q_state_indices"] = state_idx
                context["q_weights"] = self._q_rejoin_weight_vector(
                    nominal_q_seq.shape[1], state_idx
                )
                context["q_nom_future"] = nominal_q_seq[
                    self.min_rejoin_offset :, state_idx
                ]
                context["q_nom_future_start_index"] = int(self.min_rejoin_offset)

        needs_ee = self.inner_rejoin_metric == "ee_pose" or (
            self.use_ee_final_check and self.final_rejoin_metric == "ee_pose"
        )
        if not needs_ee:
            return context

        ee_seq = None
        if nominal_ee_seq is not None:
            ee_seq = np.asarray(nominal_ee_seq, dtype=np.float32)
        elif self.cache_nominal_ee:
            t0 = time.perf_counter()
            ee_seq = self._ee_pose_sequence(nominal_q_seq)
            context["ee_nom_cache_time_ms"] = (time.perf_counter() - t0) * 1000.0
        if ee_seq is not None and ee_seq.shape[0] == nominal_q_seq.shape[0]:
            ee_seq = ee_seq.reshape(ee_seq.shape[0], -1).astype(np.float32)
            context["ee_nom_seq"] = ee_seq
            if ee_seq.shape[0] > self.min_rejoin_offset:
                context["ee_nom_future"] = ee_seq[self.min_rejoin_offset :]
        return context

    def _q_rejoin_weight_vector(self, q_dim, state_idx):
        weights = self.q_rejoin_weights
        if weights is None:
            return np.ones(state_idx.shape[0], dtype=np.float32)
        try:
            weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        except Exception:  # pragma: no cover - defensive config path
            logger.warning("Invalid q_rejoin_weights; using unit weights.")
            return np.ones(state_idx.shape[0], dtype=np.float32)
        if weights.size == 1:
            return np.full(state_idx.shape[0], float(weights[0]), dtype=np.float32)
        if weights.size == state_idx.shape[0]:
            return weights.astype(np.float32)
        if weights.size >= int(np.max(state_idx)) + 1:
            return weights[state_idx].astype(np.float32)
        logger.warning(
            "q_rejoin_weights length %d does not match state dim %d or controlled dim %d; "
            "using unit weights.",
            weights.size,
            q_dim,
            state_idx.shape[0],
        )
        return np.ones(state_idx.shape[0], dtype=np.float32)

    def _q_rejoin_loss(self, q_seq, nominal_q_seq=None, rejoin_context=None):
        t0 = time.perf_counter()
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq)
        q_seq = np.asarray(q_seq, dtype=np.float32)
        state_idx = rejoin_context.get("q_state_indices")
        future = rejoin_context.get("q_nom_future")
        weights = rejoin_context.get("q_weights")
        if state_idx is None or future is None or q_seq.shape[0] == 0:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0
        valid = state_idx < q_seq.shape[1]
        if not np.all(valid):
            state_idx = state_idx[valid]
            future = future[:, valid]
            weights = None if weights is None else weights[valid]
        if state_idx.size == 0:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0
        loss, j_best = self._nearest_future_loss(
            q_seq[-1, state_idx],
            future,
            weights=weights,
            start_index=self.min_rejoin_offset,
        )
        return loss, j_best, (time.perf_counter() - t0) * 1000.0

    def _qd_rejoin_loss(self, q_seq, nominal_q_seq=None, target_index=None, rejoin_context=None):
        t0 = time.perf_counter()
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq)
        q_seq = np.asarray(q_seq, dtype=np.float32)
        nominal = rejoin_context.get("nominal_q_seq")
        if nominal is None and nominal_q_seq is not None:
            nominal = np.asarray(nominal_q_seq, dtype=np.float32)
        state_idx = rejoin_context.get("q_state_indices")
        weights = rejoin_context.get("q_weights")
        if (
            target_index is None
            or nominal is None
            or state_idx is None
            or q_seq.shape[0] < 2
        ):
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0
        nominal = np.asarray(nominal, dtype=np.float32)
        target_index = int(target_index)
        if target_index <= 0 or target_index >= nominal.shape[0]:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0
        valid = (state_idx < q_seq.shape[1]) & (state_idx < nominal.shape[1])
        if not np.all(valid):
            state_idx = state_idx[valid]
            weights = None if weights is None else weights[valid]
        if state_idx.size == 0:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0
        dt = max(float(self.dt), 1e-9)
        candidate_qd = (q_seq[-1, state_idx] - q_seq[-2, state_idx]) / dt
        nominal_qd = (
            nominal[target_index, state_idx] - nominal[target_index - 1, state_idx]
        ) / dt
        diff = candidate_qd - nominal_qd
        if weights is not None:
            diff = diff * np.asarray(weights, dtype=np.float32).reshape(-1)
        loss = float(np.square(diff).sum())
        return loss, target_index, (time.perf_counter() - t0) * 1000.0

    def _ee_rejoin_loss(
        self,
        q_seq,
        nominal_q_seq=None,
        nominal_ee_seq=None,
        rejoin_context=None,
    ):
        t0 = time.perf_counter()
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)
        ee_future = rejoin_context.get("ee_nom_future")
        if ee_future is None and nominal_ee_seq is not None:
            ee_seq = np.asarray(nominal_ee_seq, dtype=np.float32)
            if ee_seq.shape[0] > self.min_rejoin_offset:
                ee_future = ee_seq.reshape(ee_seq.shape[0], -1)[self.min_rejoin_offset :]
        if ee_future is None and nominal_q_seq is not None:
            ee_seq = self._ee_pose_sequence(nominal_q_seq)
            if ee_seq is not None and ee_seq.shape[0] > self.min_rejoin_offset:
                ee_future = ee_seq.reshape(ee_seq.shape[0], -1)[self.min_rejoin_offset :]
        opt_ee_seq = self._ee_pose_sequence(q_seq)
        if opt_ee_seq is None or ee_future is None:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0, False
        loss, j_best = self._nearest_future_loss(
            opt_ee_seq[-1],
            ee_future,
            start_index=self.min_rejoin_offset,
        )
        return loss, j_best, (time.perf_counter() - t0) * 1000.0, True

    def _final_rejoin_check(
        self,
        q_seq,
        nominal_q_seq=None,
        nominal_ee_seq=None,
        rejoin_context=None,
    ):
        if not self.recoverable_deform_enabled:
            return {
                "is_recoverable": None,
                "q_rejoin_loss": 0.0,
                "q_rejoin_dist": 0.0,
                "q_rejoin_index": None,
                "qd_rejoin_loss": 0.0,
                "qd_rejoin_dist": 0.0,
                "qd_rejoin_index": None,
                "qd_rejoin_threshold": float(self.qd_rejoin_threshold),
                "ee_rejoin_loss": 0.0,
                "ee_rejoin_dist": 0.0,
                "ee_rejoin_index": None,
                "ee_final_check_available": None,
                "rejoin_q_eval_time_ms": 0.0,
                "rejoin_qd_eval_time_ms": 0.0,
                "ee_final_check_time_ms": 0.0,
            }
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)

        q_loss, q_j_best, q_time_ms = self._q_rejoin_loss(
            q_seq,
            nominal_q_seq=nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        q_dist = self._sqrt_loss(q_loss)
        q_recoverable = bool(
            q_j_best is not None and q_dist < self.q_rejoin_threshold
        )
        qd_loss, qd_j_best, qd_time_ms = self._qd_rejoin_loss(
            q_seq,
            nominal_q_seq=nominal_q_seq,
            target_index=q_j_best,
            rejoin_context=rejoin_context,
        )
        qd_dist = self._sqrt_loss(qd_loss)
        qd_recoverable, qd_acceptance = self._qd_rejoin_acceptance(
            qd_j_best,
            qd_dist,
        )

        ee_loss = 0.0
        ee_dist = 0.0
        ee_j_best = None
        ee_time_ms = 0.0
        ee_available = None
        ee_recoverable = True
        if self.use_ee_final_check and self.final_rejoin_metric == "ee_pose":
            t0 = time.perf_counter()
            ee_future = rejoin_context.get("ee_nom_future")
            if ee_future is None and nominal_ee_seq is not None:
                ee_seq = np.asarray(nominal_ee_seq, dtype=np.float32)
                if ee_seq.shape[0] > self.min_rejoin_offset:
                    ee_future = ee_seq.reshape(ee_seq.shape[0], -1)[
                        self.min_rejoin_offset :
                    ]
            if ee_future is None and nominal_q_seq is not None:
                ee_seq = self._ee_pose_sequence(nominal_q_seq)
                if ee_seq is not None and ee_seq.shape[0] > self.min_rejoin_offset:
                    ee_future = ee_seq.reshape(ee_seq.shape[0], -1)[
                        self.min_rejoin_offset :
                    ]
            ee_opt_end = self._ee_pose(q_seq[-1]) if q_seq.shape[0] else None
            if ee_opt_end is None or ee_future is None:
                ee_available = False
                ee_loss = float("inf")
                ee_dist = float("inf")
                ee_recoverable = False
            else:
                ee_available = True
                ee_loss, ee_j_best = self._nearest_future_loss(
                    ee_opt_end,
                    ee_future,
                    start_index=self.min_rejoin_offset,
                )
                ee_dist = self._sqrt_loss(ee_loss)
                ee_recoverable = bool(ee_dist < self.ee_rejoin_threshold)
            ee_time_ms = (time.perf_counter() - t0) * 1000.0
        elif self.final_rejoin_metric == "q_state":
            ee_available = None
        elif self.final_rejoin_metric == "none":
            ee_available = None

        return {
            "is_recoverable": bool(q_recoverable and qd_recoverable and ee_recoverable),
            "q_rejoin_loss": float(q_loss),
            "q_rejoin_dist": float(q_dist),
            "q_rejoin_index": None if q_j_best is None else int(q_j_best),
            "q_rejoin_threshold": float(self.q_rejoin_threshold),
            "qd_rejoin_loss": float(qd_loss),
            "qd_rejoin_dist": float(qd_dist),
            "qd_rejoin_index": None if qd_j_best is None else int(qd_j_best),
            "qd_rejoin_threshold": float(self.qd_rejoin_threshold),
            **qd_acceptance,
            "ee_rejoin_loss": float(ee_loss),
            "ee_rejoin_dist": float(ee_dist),
            "ee_rejoin_index": None if ee_j_best is None else int(ee_j_best),
            "ee_rejoin_threshold": float(self.ee_rejoin_threshold),
            "ee_final_check_available": ee_available,
            "rejoin_q_eval_time_ms": float(q_time_ms),
            "rejoin_qd_eval_time_ms": float(qd_time_ms),
            "ee_final_check_time_ms": float(ee_time_ms),
        }

    def _sqrt_loss(self, loss):
        if not np.isfinite(loss):
            return float("inf")
        return float(np.sqrt(max(float(loss), 0.0)))

    def _qd_rejoin_acceptance(self, qd_index, qd_dist):
        try:
            qd_dist = float(qd_dist)
        except Exception:  # noqa: BLE001
            qd_dist = float("inf")
        hard_threshold = float(self.qd_rejoin_hard_threshold)
        finite_qd = bool(np.isfinite(qd_dist))
        hard_enabled = bool(np.isfinite(hard_threshold) and hard_threshold > 0.0)
        hard_failed = bool(hard_enabled and finite_qd and qd_dist >= hard_threshold)
        required = bool(self.require_qd_rejoin)
        threshold_ok = bool(
            qd_index is not None
            and finite_qd
            and qd_dist < float(self.qd_rejoin_threshold)
        )
        ok = bool((threshold_ok or not required) and not hard_failed)
        return ok, {
            "qd_rejoin_required": required,
            "qd_rejoin_hard_threshold": hard_threshold,
            "qd_rejoin_hard_failed": hard_failed,
            "qd_rejoin_soft_ok": threshold_ok,
        }

    def _nearest_future_loss(self, final_state, nominal_seq, weights=None, start_index=None):
        nominal_seq = np.asarray(nominal_seq, dtype=np.float32)
        if start_index is None:
            if nominal_seq.shape[0] <= self.min_rejoin_offset:
                return float("inf"), None
            future = nominal_seq[self.min_rejoin_offset :]
            index_offset = self.min_rejoin_offset
        else:
            if nominal_seq.shape[0] == 0:
                return float("inf"), None
            future = nominal_seq
            index_offset = int(start_index)
        final = np.asarray(final_state, dtype=np.float32).reshape(-1)
        future = future.reshape(future.shape[0], -1)
        diff = future - final[None, :]
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float32).reshape(1, -1)
            diff = diff * weights
        dist_to_future = np.square(diff).sum(axis=1)
        best_local_idx = int(np.argmin(dist_to_future))
        return float(dist_to_future[best_local_idx]), best_local_idx + index_offset

    def _ee_pose(self, q):
        op = self._get_oscbf_operator()
        for method_name in ("ee_pose", "compute_ee_pose"):
            method = getattr(op, method_name, None)
            if method is None:
                continue
            try:
                ee_pose = np.asarray(method(q), dtype=np.float32).reshape(-1)
            except Exception as exc:  # pragma: no cover - optional integration path
                logger.debug("EE rejoin pose via %s failed: %s", method_name, exc)
                continue
            return ee_pose
        ee_seq = self._ee_pose_sequence(np.asarray(q, dtype=np.float32).reshape(1, -1))
        if ee_seq is None or ee_seq.shape[0] == 0:
            return None
        return ee_seq[-1]

    def _ee_pose_sequence(self, q_seq):
        op = self._get_oscbf_operator()
        for method_name in ("ee_pose_sequence", "compute_ee_pose_sequence"):
            method = getattr(op, method_name, None)
            if method is None:
                continue
            try:
                ee_seq = np.asarray(method(q_seq), dtype=np.float32)
            except Exception as exc:  # pragma: no cover - optional integration path
                logger.debug("EE rejoin rollout via %s failed: %s", method_name, exc)
                continue
            if ee_seq.shape[0] == np.asarray(q_seq).shape[0]:
                return ee_seq.reshape(ee_seq.shape[0], -1)
        return None

    def _smoothness_loss(self, chunk, action_idx):
        if len(action_idx) == 0 or chunk.shape[0] <= 1:
            return 0.0
        velocity_loss = float(np.square(np.diff(chunk[:, action_idx], axis=0)).mean())
        if chunk.shape[0] <= 2:
            return velocity_loss
        acc = chunk[2:, action_idx] - 2.0 * chunk[1:-1, action_idx] + chunk[:-2, action_idx]
        return velocity_loss + 0.5 * float(np.square(acc).mean())

    def _project_optimized_chunk(self, candidate, nominal, action_idx):
        projected = np.asarray(candidate, dtype=nominal.dtype).copy()
        passthrough_idx = [i for i in range(nominal.shape[1]) if i not in set(action_idx.tolist())]
        projected[:, passthrough_idx] = nominal[:, passthrough_idx]
        projected = self._clip_controlled_delta(projected, nominal, action_idx)
        if self.action_low is not None or self.action_high is not None:
            low = -np.inf if self.action_low is None else float(self.action_low)
            high = np.inf if self.action_high is None else float(self.action_high)
            projected[:, action_idx] = np.clip(projected[:, action_idx], low, high)
        projected[:, passthrough_idx] = nominal[:, passthrough_idx]
        return projected

    def _optimized_final_info(
        self,
        obs,
        chunk,
        nominal,
        nominal_q_seq,
        nominal_ee_seq,
        human_state,
        j_best,
        rejoin_loss,
        losses,
        rejoin_context=None,
    ):
        del human_state
        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        min_clearance = float(np.min(h_seq))
        is_safe = bool(min_clearance >= self.min_clearance)
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)
        if not losses:
            _, losses = self._optimized_deformation_cost(
                obs,
                chunk,
                nominal,
                nominal_q_seq,
                nominal_ee_seq,
                None,
                rejoin_context=rejoin_context,
            )
            j_best = losses.get("j_best")
            rejoin_loss = losses.get("rejoin_loss", rejoin_loss)

        final_rejoin = self._final_rejoin_check(
            q_seq,
            nominal_q_seq=nominal_q_seq,
            nominal_ee_seq=nominal_ee_seq,
            rejoin_context=rejoin_context,
        )
        if self.recoverable_deform_enabled:
            is_recoverable = bool(final_rejoin.get("is_recoverable", False))
            rejoin_index = final_rejoin.get("q_rejoin_index")
            if rejoin_index is None:
                rejoin_index = final_rejoin.get("ee_rejoin_index")
            rejoin_cost = float(final_rejoin.get("q_rejoin_loss", float("inf")))
        else:
            is_recoverable = None
            rejoin_index = None if j_best is None else int(j_best)
            rejoin_cost = float(rejoin_loss)
        best_min_clearance = float(losses.get("min_clearance", min_clearance))
        required_min_clearance = float(self.min_clearance)
        clearance_gap = float(required_min_clearance - best_min_clearance)
        safety_rejected = not is_safe
        recovery_rejected = bool(
            self.recoverable_deform_enabled and not bool(is_recoverable)
        )
        rejection_cause = self._optimized_reject_reason_from_flags(
            safety_rejected,
            recovery_rejected,
        )

        info = dict(safety_eval)
        info.update(
            {
                "deform_mode": "optimized",
                "deformation_source": "optimized_deform",
                "deform_safe": is_safe,
                "deform_min_clearance": min_clearance,
                "deformation_norm": self._controlled_deformation_norm(chunk, nominal),
                "recoverable_deform_enabled": self.recoverable_deform_enabled,
                "brake_if_unrecoverable": self.brake_if_unrecoverable,
                "inner_rejoin_metric": self.inner_rejoin_metric,
                "final_rejoin_metric": self.final_rejoin_metric,
                "cache_nominal_ee": self.cache_nominal_ee,
                "ee_rejoin_in_inner_loop": self.ee_rejoin_in_inner_loop,
                "debug_safety_feasibility": self.debug_safety_feasibility,
                "is_safe": is_safe,
                "is_recoverable": is_recoverable,
                "safety_rejected": safety_rejected,
                "recovery_rejected": recovery_rejected,
                "rejection_cause": rejection_cause,
                "best_min_clearance": best_min_clearance,
                "required_min_clearance": required_min_clearance,
                "clearance_gap": clearance_gap,
                "rejoin_cost": rejoin_cost,
                "rejoin_loss": float(losses.get("rejoin_loss", rejoin_loss)),
                "q_rejoin_loss": float(final_rejoin.get("q_rejoin_loss", 0.0)),
                "q_rejoin_dist": float(final_rejoin.get("q_rejoin_dist", 0.0)),
                "q_rejoin_threshold": float(final_rejoin.get(
                    "q_rejoin_threshold", self.q_rejoin_threshold
                )),
                "q_rejoin_index": final_rejoin.get("q_rejoin_index"),
                "qd_rejoin_loss": float(final_rejoin.get("qd_rejoin_loss", 0.0)),
                "qd_rejoin_dist": float(final_rejoin.get("qd_rejoin_dist", 0.0)),
                "qd_rejoin_threshold": float(final_rejoin.get(
                    "qd_rejoin_threshold", self.qd_rejoin_threshold
                )),
                "qd_rejoin_index": final_rejoin.get("qd_rejoin_index"),
                "ee_rejoin_loss": float(final_rejoin.get("ee_rejoin_loss", 0.0)),
                "ee_rejoin_dist": float(final_rejoin.get("ee_rejoin_dist", 0.0)),
                "ee_rejoin_threshold": float(final_rejoin.get(
                    "ee_rejoin_threshold", self.ee_rejoin_threshold
                )),
                "ee_rejoin_index": final_rejoin.get("ee_rejoin_index"),
                "ee_final_check_available": final_rejoin.get(
                    "ee_final_check_available"
                ),
                "rejoin_index": None if rejoin_index is None else int(rejoin_index),
                "j_best": None if rejoin_index is None else int(rejoin_index),
                "optimizer_j_best": None if j_best is None else int(j_best),
                "rejoin_space": losses.get("rejoin_space"),
                "safety_loss": float(losses.get("safety_loss", 0.0)),
                "action_deviation_loss": float(losses.get("action_deviation_loss", 0.0)),
                "path_loss": float(losses.get("path_loss", 0.0)),
                "existing_optimization_loss": float(
                    losses.get("existing_optimization_loss", 0.0)
                ),
                "smoothness_loss": float(losses.get("smoothness_loss", 0.0)),
                "total_loss": float(losses.get("total_loss", 0.0)),
                "rejoin_q_eval_time_ms": float(final_rejoin.get(
                    "rejoin_q_eval_time_ms",
                    losses.get("rejoin_q_eval_time_ms", 0.0),
                )),
                "rejoin_qd_eval_time_ms": float(final_rejoin.get(
                    "rejoin_qd_eval_time_ms",
                    losses.get("rejoin_qd_eval_time_ms", 0.0),
                )),
                "ee_nom_cache_time_ms": float(
                    rejoin_context.get(
                        "ee_nom_cache_time_ms",
                        losses.get("ee_nom_cache_time_ms", 0.0),
                    )
                ),
                "ee_final_check_time_ms": float(final_rejoin.get(
                    "ee_final_check_time_ms", 0.0
                )),
                "jax_batched_optimizer": bool(losses.get(
                    "jax_batched_optimizer", self._jax_optimizer_ready()
                )),
                "jax_rollout_time_ms": float(losses.get(
                    "jax_rollout_time_ms", 0.0
                )),
                "fallback_used": False,
                "recovery_mode": (
                    "optimized_recoverable_deform"
                    if self.recoverable_deform_enabled
                    else None
                ),
                "min_clearance": min_clearance,
            }
        )
        log_fn = logger.info if self.debug_safety_feasibility else logger.debug
        log_fn(
            "optimized SafeChunk-Deform final: mode=%s best_min_clearance=%.4f "
            "required_min_clearance=%.4f clearance_gap=%.4f safety_loss=%.6f "
            "existing_loss=%.6f rejoin_loss=%.6f q_dist=%.6f ee_dist=%.6f "
            "rejoin_index=%s safe=%s recoverable=%s rejection_cause=%s",
            info.get("recovery_mode") or info.get("deform_mode"),
            info["best_min_clearance"],
            info["required_min_clearance"],
            info["clearance_gap"],
            info["safety_loss"],
            info["existing_optimization_loss"],
            info["rejoin_loss"],
            info["q_rejoin_dist"],
            info["ee_rejoin_dist"],
            info["rejoin_index"],
            info["is_safe"],
            info["is_recoverable"],
            info["rejection_cause"],
        )
        return info

    def _optimized_failure_info(self, error):
        return {
            "deform_mode": "optimized",
            "deformation_source": "optimized_deform",
            "deform_safe": False,
            "is_safe": False,
            "recoverable_deform_enabled": self.recoverable_deform_enabled,
            "brake_if_unrecoverable": self.brake_if_unrecoverable,
            "is_recoverable": False if self.recoverable_deform_enabled else None,
            "inner_rejoin_metric": self.inner_rejoin_metric,
            "final_rejoin_metric": self.final_rejoin_metric,
            "cache_nominal_ee": self.cache_nominal_ee,
            "ee_rejoin_in_inner_loop": self.ee_rejoin_in_inner_loop,
            "debug_safety_feasibility": self.debug_safety_feasibility,
            "safety_rejected": True,
            "recovery_rejected": bool(self.recoverable_deform_enabled),
            "rejection_cause": (
                "unsafe_and_unrecoverable"
                if self.recoverable_deform_enabled
                else "unsafe"
            ),
            "best_min_clearance": float("-inf"),
            "required_min_clearance": float(self.min_clearance),
            "clearance_gap": float("inf"),
            "rejoin_index": None,
            "j_best": None,
            "rejoin_cost": float("inf"),
            "rejoin_loss": float("inf"),
            "q_rejoin_loss": float("inf"),
            "q_rejoin_dist": float("inf"),
            "q_rejoin_threshold": self.q_rejoin_threshold,
            "q_rejoin_index": None,
            "qd_rejoin_loss": float("inf"),
            "qd_rejoin_dist": float("inf"),
            "qd_rejoin_threshold": self.qd_rejoin_threshold,
            "qd_rejoin_index": None,
            "ee_rejoin_loss": float("inf"),
            "ee_rejoin_dist": float("inf"),
            "ee_rejoin_threshold": self.ee_rejoin_threshold,
            "ee_rejoin_index": None,
            "ee_final_check_available": False,
            "rejoin_q_eval_time_ms": 0.0,
            "rejoin_qd_eval_time_ms": 0.0,
            "ee_nom_cache_time_ms": 0.0,
            "ee_final_check_time_ms": 0.0,
            "safety_loss": float("inf"),
            "action_deviation_loss": 0.0,
            "path_loss": 0.0,
            "existing_optimization_loss": float("inf"),
            "smoothness_loss": 0.0,
            "total_loss": float("inf"),
            "fallback_used": True,
            "deform_min_clearance": float("-inf"),
            "min_clearance": float("-inf"),
            "optimized_error": error,
        }

    def _optimized_reject_reason(self, info):
        safety_rejected = not bool(info.get("deform_safe", False))
        recovery_rejected = bool(
            self.recoverable_deform_enabled and not info.get("is_recoverable", False)
        )
        return self._optimized_reject_reason_from_flags(
            safety_rejected,
            recovery_rejected,
        ) or "rejected"

    def _optimized_reject_reason_from_flags(self, safety_rejected, recovery_rejected):
        if safety_rejected and recovery_rejected:
            return "unsafe_and_unrecoverable"
        if safety_rejected:
            return "unsafe"
        if recovery_rejected:
            return "unrecoverable"
        return None

    def _prefixed_optimized_info(self, info):
        keys = (
            "safety_loss",
            "action_deviation_loss",
            "path_loss",
            "existing_optimization_loss",
            "rejoin_loss",
            "smoothness_loss",
            "total_loss",
            "min_clearance",
            "is_safe",
            "is_recoverable",
            "j_best",
            "rejoin_index",
            "rejoin_cost",
            "rejoin_space",
            "explicit_return",
            "recovery_phase",
            "cached_motion_active",
            "recovery_context_active",
            "start_chunk_index",
            "trigger_step",
            "target_rejoin_index",
            "yield_min_clearance",
            "yield_accepted",
            "return_min_clearance",
            "return_rejoin_loss",
            "return_target_index",
            "return_accepted",
            "resumed_from_cached_index",
            "return_retries",
            "max_return_retries",
            "inner_rejoin_metric",
            "final_rejoin_metric",
            "debug_safety_feasibility",
            "safety_rejected",
            "recovery_rejected",
            "rejection_cause",
            "best_min_clearance",
            "required_min_clearance",
            "clearance_gap",
            "q_rejoin_loss",
            "q_rejoin_dist",
            "q_rejoin_threshold",
            "q_rejoin_index",
            "ee_rejoin_loss",
            "ee_rejoin_dist",
            "ee_rejoin_threshold",
            "ee_rejoin_index",
            "ee_final_check_available",
            "rejoin_q_eval_time_ms",
            "ee_nom_cache_time_ms",
            "ee_final_check_time_ms",
            "recoverable_deform_enabled",
            "fallback_used",
            "recovery_mode",
            "deform_min_clearance",
        )
        prefixed = {f"optimized_{key}": info.get(key) for key in keys if key in info}
        for key in keys:
            if key in info:
                prefixed.setdefault(key, info[key])
        return prefixed

    def _clearance_sequence_from_eval(self, safety_eval, horizon):
        if isinstance(safety_eval, dict):
            h_seq = np.asarray(
                safety_eval.get("min_clearances", safety_eval.get("clearances", [])),
                dtype=np.float32,
            ).reshape(-1)
            if h_seq.size == 0:
                h_seq = np.full(
                    int(horizon),
                    float(safety_eval.get("min_clearance", np.inf)),
                    dtype=np.float32,
                )
        else:
            h_seq = np.asarray(safety_eval, dtype=np.float32).reshape(-1)
        if h_seq.size == 1 and horizon > 1:
            h_seq = np.full(int(horizon), float(h_seq[0]), dtype=np.float32)
        if h_seq.size == 0:
            h_seq = np.full(int(horizon), np.inf, dtype=np.float32)
        return h_seq

    def _obs_with_q(self, obs, q_current):
        if q_current is None:
            return {} if obs is None else obs
        q = np.asarray(q_current, dtype=np.float32).reshape(-1)
        if obs is None:
            return {"q": q}
        if isinstance(obs, dict):
            merged = dict(obs)
            merged["q"] = q
            return merged
        return {"q": q}

    def _make_chunk_deformation_candidates(self, obs, chunk, safety_info):
        valid = self._valid_control_indices(chunk)
        if not np.any(valid):
            return [(None, chunk.copy())]

        action_idx = self.controlled_action_indices[valid]
        state_idx = self.controlled_state_indices[valid]
        anchor = self._controlled_anchor(obs, chunk, action_idx, state_idx)
        start_idx = self._deformation_start_idx(safety_info, chunk.shape[0])
        candidates = []
        seen = set()

        for scale in self.chunk_deformation_scales:
            scale = float(np.clip(scale, 0.0, 1.0))
            candidate = chunk.copy()
            profile = np.ones(chunk.shape[0], dtype=np.float32)
            profile[start_idx:] = scale
            nominal = chunk[:, action_idx]
            candidate[:, action_idx] = anchor + profile[:, None] * (nominal - anchor)
            candidate = self._clip_controlled_delta(candidate, chunk, action_idx)
            candidate = self._smooth_controlled_suffix(candidate, action_idx, start_idx)
            candidate = self._clip_controlled_delta(candidate, chunk, action_idx)
            passthrough_idx = [
                i for i in range(chunk.shape[1]) if i not in set(action_idx.tolist())
            ]
            candidate[:, passthrough_idx] = chunk[:, passthrough_idx]
            key = tuple(np.round(candidate[:, action_idx].reshape(-1), 8))
            if key not in seen:
                candidates.append((scale, candidate))
                seen.add(key)

        return candidates

    def _controlled_anchor(self, obs, chunk, action_idx, state_idx):
        anchor = np.zeros(len(action_idx), dtype=chunk.dtype)
        q = self.extract_current_q(obs, chunk)
        valid = state_idx < q.shape[0]
        if np.any(valid):
            modes = self._control_mode_ids_for_state_indices(state_idx)
            absolute = valid & (modes == 0)
            if np.any(absolute):
                anchor[absolute] = q[state_idx[absolute]].astype(chunk.dtype, copy=False)
        return anchor

    def _deformation_start_idx(self, safety_info, horizon):
        first_violation = safety_info.get("first_violation")
        if first_violation is None:
            return 0
        return max(0, min(int(first_violation) - 1, horizon - 1))

    def _smooth_controlled_suffix(self, candidate, action_idx, start_idx):
        if self.chunk_deformation_smoothing <= 0 or candidate.shape[0] <= 2:
            return candidate
        smoothed = candidate.copy()
        for _ in range(self.chunk_deformation_smoothing):
            prev = smoothed.copy()
            for k in range(max(1, start_idx), candidate.shape[0] - 1):
                smoothed[k, action_idx] = (
                    0.25 * prev[k - 1, action_idx]
                    + 0.5 * prev[k, action_idx]
                    + 0.25 * prev[k + 1, action_idx]
                )
        return smoothed

    def _clip_controlled_delta(self, candidate, nominal, action_idx):
        if self.max_action_delta is None or len(action_idx) == 0:
            return candidate
        clipped = candidate.copy()
        delta = clipped[:, action_idx] - nominal[:, action_idx]
        delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
        clipped[:, action_idx] = nominal[:, action_idx] + delta
        return clipped

    def _valid_control_indices(self, chunk):
        return self.controlled_action_indices < chunk.shape[1]

    def _controlled_deformation_norm(self, candidate, nominal):
        valid = self._valid_control_indices(nominal)
        if not np.any(valid):
            return 0.0
        action_idx = self.controlled_action_indices[valid]
        delta = candidate[:, action_idx] - nominal[:, action_idx]
        return float(np.mean(np.linalg.norm(delta, axis=1)))

    def _controlled_progress_retention(self, candidate, nominal, obs):
        valid = self._valid_control_indices(nominal)
        if not np.any(valid):
            return 1.0
        action_idx = self.controlled_action_indices[valid]
        state_idx = self.controlled_state_indices[valid]
        anchor = self._controlled_anchor(obs, nominal, action_idx, state_idx)
        nominal_delta = nominal[:, action_idx] - anchor[None, :]
        candidate_delta = candidate[:, action_idx] - anchor[None, :]
        denom = float(np.mean(np.linalg.norm(nominal_delta, axis=1)))
        if denom <= 1e-9:
            return 1.0
        numer = float(np.mean(np.linalg.norm(candidate_delta, axis=1)))
        return float(np.clip(numer / denom, 0.0, 2.0))

    def _is_better_deformation_candidate(
        self,
        candidate_eval,
        candidate_norm,
        candidate_safe,
        best_eval,
        best_norm,
        best_safe,
    ):
        if best_eval is None:
            return True
        if candidate_safe != best_safe:
            return candidate_safe
        if candidate_safe:
            candidate_progress = float(
                candidate_eval.get("task_progress_retention", 1.0)
            )
            best_progress = float(best_eval.get("task_progress_retention", 1.0))
            if abs(candidate_progress - best_progress) > 1e-6:
                return candidate_progress > best_progress
            return candidate_norm < best_norm
        clearance_margin = (
            float(candidate_eval["min_clearance"])
            - float(best_eval["min_clearance"])
        )
        if abs(clearance_margin) > 1e-9:
            return clearance_margin > 0.0
        return candidate_norm < best_norm

    def deform_chunk_with_oscbf(self, obs, action_chunk, **kwargs):
        chunk, _ = self._as_chunk(action_chunk)
        safe_chunk = chunk.copy()
        op = self._get_oscbf_operator()
        batch_filter_info = {}
        batch_filter_t0 = time.perf_counter()
        if op is not None:
            for method_name in (
                "filter_chunk",
                "filter_action_chunk",
            ):
                method = getattr(op, method_name, None)
                if method is None:
                    continue
                try:
                    result = self._call_oscbf_chunk_method(method, obs, chunk, **kwargs)
                    if isinstance(result, tuple):
                        candidate, candidate_info = result
                    else:
                        candidate, candidate_info = result, {}
                    candidate = np.asarray(candidate, dtype=chunk.dtype)
                    if candidate.shape != chunk.shape:
                        raise ValueError(
                            "Chunk safety operator returned shape "
                            f"{candidate.shape}, expected {chunk.shape}"
                        )
                    safe_chunk = chunk.copy()
                    safe_chunk[:, self.controlled_action_indices] = candidate[
                        :, self.controlled_action_indices
                    ]
                    batch_filter_info = dict(candidate_info or {})
                    batch_filter_info.update(
                        {
                            "sequential_oscbf_batched": True,
                            "sequential_oscbf_batch_method": method_name,
                            "sequential_oscbf_batch_filter_time_ms": float(
                                1000.0 * (time.perf_counter() - batch_filter_t0)
                            ),
                        }
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Batched sequential OSCBF via %s failed; using per-step path: %s",
                        method_name,
                        exc,
                    )
        if callable(op) and not batch_filter_info:
            for k, action in enumerate(chunk):
                safe_action = self._call_single_step_operator(action, obs, **kwargs)
                safe_action = np.asarray(safe_action, dtype=chunk.dtype).reshape(-1)
                if safe_action.shape[0] != chunk.shape[1]:
                    raise ValueError(
                        "Single-step safety operator returned shape "
                        f"{safe_action.shape}, expected ({chunk.shape[1]},)"
                    )
                safe_chunk[k, self.controlled_action_indices] = safe_action[
                    self.controlled_action_indices
                ]
            batch_filter_info = {
                "sequential_oscbf_batched": False,
                "sequential_oscbf_batch_method": None,
                "sequential_oscbf_batch_filter_time_ms": float(
                    1000.0 * (time.perf_counter() - batch_filter_t0)
                ),
            }

        delta = (
            safe_chunk[:, self.controlled_action_indices]
            - chunk[:, self.controlled_action_indices]
        )
        deformation_norm = float(np.mean(np.linalg.norm(delta, axis=1)))
        q_seq = self.rollout_nominal_chunk(obs, safe_chunk)
        deform_safety = self.evaluate_horizon_safety(obs, q_seq)
        info = dict(deform_safety)
        info.update(
            {
                "deform_safe": bool(deform_safety["horizon_safe"]),
                "deform_min_clearance": float(deform_safety["min_clearance"]),
                "deformation_norm": deformation_norm,
            }
        )
        info.update(batch_filter_info)
        return safe_chunk, info

    def _call_oscbf_chunk_method(self, method, obs, chunk, **kwargs):
        attempts = (
            lambda: method(action_chunk=chunk, obs=obs, **kwargs),
            lambda: method(action_chunk=chunk, observations=obs, **kwargs),
            lambda: method(obs=obs, action_chunk=chunk, **kwargs),
            lambda: method(observations=obs, action_chunk=chunk, **kwargs),
            lambda: method(chunk, obs, **kwargs),
            lambda: method(chunk, **kwargs),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Chunk OSCBF method could not be called")

    def _contact_count_from_kwargs(self, kwargs):
        for key in ("contact_count", "contacts", "robot_human_contact_count"):
            if key in kwargs and kwargs[key] is not None:
                try:
                    return int(kwargs[key])
                except Exception:  # noqa: BLE001
                    return None
        return None

    def _update_last_safe_execution(self, obs, executed_chunk, info, **kwargs):
        chunk, _ = self._as_chunk(executed_chunk)
        if chunk.shape[0] == 0:
            return
        contact_count = self._contact_count_from_kwargs(kwargs)
        if contact_count is not None and contact_count > 0:
            return
        clearance = info.get("immediate_clearance", info.get("min_clearance"))
        if clearance is None:
            clearance = info.get("deform_min_clearance", info.get("brake_min_clearance"))
        try:
            clearance = float(clearance)
        except Exception:  # noqa: BLE001
            clearance = float("inf")
        if clearance < float(self.active_safety_hard_min_clearance):
            return
        self.last_safe_action = np.asarray(chunk[0], dtype=np.float32).copy()
        self.last_safe_chunk = np.asarray(chunk, dtype=np.float32).copy()
        try:
            self.last_safe_q = self._rollout_one_step_from_q(
                self.extract_current_q(obs, chunk),
                chunk[0],
            ).copy()
        except Exception:  # noqa: BLE001
            self.last_safe_q = self.extract_current_q(obs, chunk).copy()

    def _active_safety_info(self):
        h = np.asarray(self._hold_horizon_min_clearance_history, dtype=np.float32)
        return {
            "hold_unsafe_count": int(self.hold_unsafe_count),
            "hold_predicted_contact_count": int(self.hold_predicted_contact_count),
            "emergency_deform_away_steps": int(self.emergency_deform_away_steps),
            "emergency_deform_away_count": int(self.emergency_deform_away_count),
            "contact_during_hold_count": int(self.contact_during_hold_count),
            "contact_during_brake_count": int(self.contact_during_brake_count),
            "contact_during_deform_count": int(self.contact_during_deform_count),
            "contact_during_recover_count": int(self.contact_during_recover_count),
            "mean_hold_horizon_min_clearance": float(np.mean(h)) if h.size else None,
            "min_hold_horizon_min_clearance": float(np.min(h)) if h.size else None,
        }

    def _hold_prediction_metadata(self, obs=None, **kwargs):
        human_state = kwargs.get("human_state")
        current = self._snapshot_human_state(human_state)
        velocity_toward_robot = None
        prediction_available = False
        if current is not None and self._previous_human_snapshot is not None:
            try:
                cur = np.asarray(current, dtype=np.float32).reshape(-1)
                prev = np.asarray(self._previous_human_snapshot, dtype=np.float32).reshape(-1)
                if cur.shape == prev.shape:
                    velocity_toward_robot = float(np.linalg.norm(cur - prev) / max(self.dt, 1e-9))
                    prediction_available = True
            except Exception:  # noqa: BLE001
                velocity_toward_robot = None
        self._previous_human_snapshot = current
        return prediction_available, velocity_toward_robot

    def evaluate_hold_or_brake_acceptance(self, obs, braked_chunk, **kwargs):
        chunk, _ = self._as_chunk(braked_chunk)
        horizon = min(chunk.shape[0], max(1, int(self.hold_horizon_steps)))
        hold_chunk = chunk[:horizon].copy()
        human_prediction_available, human_velocity_toward_robot = self._hold_prediction_metadata(obs, **kwargs)
        try:
            acceptance = self.evaluate_candidate_acceptance(
                obs,
                hold_chunk,
                candidate_type="hold_or_brake",
            )
        except TypeError:
            acceptance = self.evaluate_candidate_acceptance(obs, hold_chunk, "hold_or_brake")
        model_immediate = float(acceptance.get("immediate_clearance", float("-inf")))
        model_horizon_min = float(acceptance.get("horizon_min_clearance", model_immediate))
        immediate = model_immediate
        horizon_min = model_horizon_min
        safe_prefix_len = int(acceptance.get("safe_prefix_len", 0) or 0)
        hard = float(self.active_safety_hard_min_clearance)
        live_monitor_min_h = kwargs.get("live_monitor_min_h", kwargs.get("min_h"))
        live_monitor_contact_risk = False
        if kwargs.get("gate_live_monitor_clearance", False) and live_monitor_min_h is not None:
            try:
                live_monitor_min_h = float(live_monitor_min_h)
                if np.isfinite(live_monitor_min_h):
                    immediate = min(immediate, live_monitor_min_h)
                    horizon_min = min(horizon_min, live_monitor_min_h)
                    live_monitor_contact_risk = bool(live_monitor_min_h < hard)
            except Exception:  # noqa: BLE001
                live_monitor_min_h = None
        predicted_contact = bool(horizon_min < hard)
        accepted = bool(
            immediate >= hard
            and not predicted_contact
            and safe_prefix_len >= 1
            and immediate >= float(self.hold_prefix_min_clearance)
        )
        reason = None
        if live_monitor_contact_risk:
            reason = "live_monitor_below_hard_margin"
        elif immediate < hard:
            reason = "immediate_below_hard_margin"
        elif predicted_contact:
            reason = "hold_predicted_contact"
        elif safe_prefix_len < 1 or immediate < float(self.hold_prefix_min_clearance):
            reason = "hold_prefix_unsafe"
        info = dict(acceptance)
        info.update(
            {
                "accepted": accepted,
                "hold_immediate_clearance": immediate,
                "hold_horizon_min_clearance": horizon_min,
                "model_hold_immediate_clearance": model_immediate,
                "model_hold_horizon_min_clearance": model_horizon_min,
                "live_monitor_min_h": live_monitor_min_h,
                "live_monitor_contact_risk": bool(live_monitor_contact_risk),
                "hold_acceptance_type": "hold_or_brake" if accepted else "rejected",
                "hold_rejected_reason": reason,
                "hold_predicted_contact": predicted_contact,
                "human_prediction_available": bool(human_prediction_available),
                "human_velocity_toward_robot": human_velocity_toward_robot,
            }
        )
        self._hold_horizon_min_clearance_history.append(horizon_min)
        if not accepted:
            self.hold_unsafe_count += 1
        if predicted_contact:
            self.hold_predicted_contact_count += 1
        return info

    def _optimize_instead_of_unsafe_hold(
        self,
        obs,
        nominal_chunk,
        braked_chunk,
        info,
        hold_info,
        original_shape,
        **kwargs,
    ):
        nominal, _ = self._as_chunk(nominal_chunk)
        braked, _ = self._as_chunk(braked_chunk)
        nominal_q_seq = self.rollout_nominal_chunk(obs, nominal)
        safety_info = self.evaluate_horizon_safety(obs, nominal_q_seq)
        forced_safety = dict(safety_info)
        live_min = hold_info.get("live_monitor_min_h")
        hold_min = hold_info.get("hold_horizon_min_clearance")
        clearance_values = [forced_safety.get("min_clearance")]
        for value in (live_min, hold_min):
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(value):
                clearance_values.append(value)
        forced_min = min(float(v) for v in clearance_values if v is not None)
        if forced_min < float(self.min_clearance) or hold_info.get("live_monitor_contact_risk"):
            forced_safety.update(
                {
                    "horizon_safe": False,
                    "min_clearance": float(forced_min),
                    "first_violation": 0,
                    "unsafe_count": max(1, int(forced_safety.get("unsafe_count", 0) or 0)),
                }
            )
        safe_chunk, deform_info = self.deform_chunk(
            obs,
            nominal,
            safety_info=forced_safety,
            braked_chunk=braked,
            nominal_q_seq=nominal_q_seq,
            **kwargs,
        )
        info.update(deform_info)
        info.update(
            {
                "unsafe_hold_replaced_by_optimization": True,
                "emergency_deform_away": False,
                "deform_trigger_reason": info.get("deform_trigger_reason", "unsafe_hold_optimization"),
            }
        )
        if bool(info.get("optimized_accepted", False)):
            self.recovery_failure_streak = 0
        elif info.get("optimized_accepted") is not None or info.get("fallback_used") is not None:
            self.recovery_failure_streak += 1
            self.recovery_failure_streak_max = max(
                self.recovery_failure_streak_max,
                self.recovery_failure_streak,
            )
        if (
            info.get("optimized_accepted", False)
            and self.explicit_return
            and self.commit_accepted_chunks
        ):
            committed, commit_reject_info = self._commit_explicit_recovery_chunk(
                obs,
                safe_chunk,
                info,
                **kwargs,
            )
            if committed:
                committed_result = self._serve_committed_chunk(obs, nominal, original_shape, **kwargs)
                pending_committed_replan_info = self._pop_pending_committed_replan_info()
                if pending_committed_replan_info:
                    info.update(pending_committed_replan_info)
                if committed_result is not None:
                    committed_chunk, committed_info = committed_result
                    committed_info.update({k: v for k, v in info.items() if k not in committed_info})
                    self.last_info = committed_info
                    return committed_chunk, committed_info
            else:
                info.update(commit_reject_info)
        if not info.get("deform_safe", info.get("optimized_accepted", False)) and self.unsafe_deformation_fallback == "brake":
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deformation_rejected": True,
                    "optimized_fallback": "brake",
                    "fallback_used": True,
                }
            )
            self.last_info = info
            return braked.reshape(original_shape), info
        info.update({"safety_mode": "horizon_deform", "mode": "horizon_deform"})
        self.last_info = info
        return np.asarray(safe_chunk, dtype=np.float32).reshape(original_shape), info

    def _hold_return_or_emergency_deform(self, obs, nominal_chunk, braked_chunk, info, original_shape, **kwargs):
        if not (self.safechunk_active_safety_enabled and self.check_hold_horizon_safety):
            self._update_last_safe_execution(obs, braked_chunk, info, **kwargs)
            self.last_info = info
            return braked_chunk.reshape(original_shape), info
        hold_info = self.evaluate_hold_or_brake_acceptance(
            obs,
            braked_chunk,
            gate_live_monitor_clearance=True,
            **kwargs,
        )
        info.update(hold_info)
        if hold_info.get("accepted"):
            self._update_last_safe_execution(obs, braked_chunk, info, **kwargs)
            self.last_info = info
            return braked_chunk.reshape(original_shape), info
        if self.optimize_when_hold_unsafe and self.deformation_enabled:
            return self._optimize_instead_of_unsafe_hold(
                obs,
                nominal_chunk,
                braked_chunk,
                info,
                hold_info,
                original_shape,
                **kwargs,
            )
        if self.emergency_deform_when_hold_unsafe:
            emergency_chunk, emergency_info = self.emergency_deform_away(
                obs,
                reference_action=braked_chunk,
                nominal_chunk=nominal_chunk,
                hold_info=hold_info,
                **kwargs,
            )
            info.update(emergency_info)
            self._update_last_safe_execution(obs, emergency_chunk, info, **kwargs)
            self.last_info = info
            return emergency_chunk.reshape(original_shape), info
        self.last_info = info
        return braked_chunk.reshape(original_shape), info

    def emergency_deform_away(self, obs, reference_action, nominal_chunk=None, hold_info=None, **kwargs):
        reference, _ = self._as_chunk(reference_action)
        nominal = reference if nominal_chunk is None else self._as_chunk(nominal_chunk)[0]
        candidates = []

        def add(name, candidate):
            if candidate is None:
                return
            cand, _ = self._as_chunk(candidate)
            if cand.shape[1] != reference.shape[1]:
                return
            if cand.shape[0] < reference.shape[0]:
                pad = np.repeat(cand[-1:], reference.shape[0] - cand.shape[0], axis=0)
                cand = np.concatenate([cand, pad], axis=0)
            elif cand.shape[0] > reference.shape[0]:
                cand = cand[: reference.shape[0]]
            controlled = set(self.controlled_action_indices.tolist())
            passthrough_idx = [i for i in range(reference.shape[1]) if i not in controlled]
            if passthrough_idx:
                cand[:, passthrough_idx] = reference[:, passthrough_idx]
            candidates.append((name, cand.astype(np.float32, copy=True)))

        if not bool((hold_info or {}).get("live_monitor_contact_risk", False)):
            add("hold", reference)
        op = self._get_oscbf_operator()
        if callable(op):
            try:
                action = np.asarray(self._call_single_step_operator(reference[0], obs, **kwargs), dtype=np.float32)
                add("oscbf_hold", np.repeat(action.reshape(1, -1), reference.shape[0], axis=0))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Emergency deform-away OSCBF candidate failed: %s", exc)
        if self.prefer_last_safe_action and self.last_safe_action is not None:
            add("last_safe_action", np.repeat(np.asarray(self.last_safe_action, dtype=np.float32).reshape(1, -1), reference.shape[0], axis=0))
        if self.prefer_last_safe_q_retract and self.last_safe_q is not None:
            cand = reference.copy()
            valid = (
                (self.controlled_action_indices < cand.shape[1])
                & (self.controlled_state_indices < np.asarray(self.last_safe_q).shape[0])
            )
            last_q = np.asarray(self.last_safe_q, dtype=np.float32)
            action_idx = self.controlled_action_indices[valid]
            state_idx = self.controlled_state_indices[valid]
            modes = self._control_mode_ids_for_state_indices(state_idx)
            absolute = modes == 0
            if np.any(absolute):
                cand[:, action_idx[absolute]] = last_q[state_idx[absolute]]
            add("last_safe_q_retract", cand)
        if self.last_safe_chunk is not None:
            for scale in self.emergency_deform_candidate_scales:
                target = np.asarray(self.last_safe_chunk, dtype=np.float32)
                if target.shape == reference.shape:
                    add(f"scaled_last_safe_{scale:g}", reference + float(scale) * (target - reference))
        best = None
        best_rejected = None
        rejected = []
        live_contact_risk = bool((hold_info or {}).get("live_monitor_contact_risk", False))
        for name, candidate in candidates:
            acc = self.evaluate_hold_or_brake_acceptance(
                obs,
                candidate,
                gate_live_monitor_clearance=False,
                **kwargs,
            )
            action_norm = float(np.linalg.norm(candidate[0, self.controlled_action_indices[self._valid_control_indices(candidate)]])) if candidate.shape[0] else 0.0
            clearance_score = float(acc.get("hold_horizon_min_clearance", acc.get("hold_immediate_clearance", float("-inf"))) or float("-inf"))
            rejected_score = (
                100.0 * clearance_score
                + 10.0 * float(acc.get("safe_prefix_len", 0) or 0)
                - 0.001 * action_norm
            )
            if not acc.get("accepted"):
                rejected.append((name, acc.get("hold_rejected_reason")))
                if name != "hold" and (best_rejected is None or rejected_score > best_rejected[0]):
                    best_rejected = (rejected_score, name, candidate, acc, action_norm)
                continue
            score = (
                1000.0
                + 100.0 * float(acc.get("hold_immediate_clearance", 0.0) or 0.0)
                + 10.0 * float(acc.get("safe_prefix_len", 0) or 0)
                - 0.01 * action_norm
            )
            if best is None or score > best[0]:
                best = (score, name, candidate, acc, action_norm)
        used_rejected_escape = False
        if best is None and best_rejected is not None and live_contact_risk:
            _score, name, candidate, acc, action_norm = best_rejected
            used_rejected_escape = True
        elif best is None:
            candidate = reference.copy()
            name = "hold_fallback"
            acc = hold_info or {}
            action_norm = float(np.linalg.norm(candidate[0])) if candidate.shape[0] else 0.0
        else:
            _score, name, candidate, acc, action_norm = best
        self.emergency_deform_away_steps += 1
        self.emergency_deform_away_count += 1
        info = {
            "safety_mode": "emergency_deform_away",
            "mode": "emergency_deform_away",
            "deform_mode": "emergency_deform_away",
            "deformation_source": "emergency_deform_away",
            "fallback_used": False,
            "emergency_deform_away": True,
            "emergency_deform_away_steps": int(self.emergency_deform_away_steps),
            "emergency_deform_away_count": int(self.emergency_deform_away_count),
            "emergency_deform_replan_next_step": bool(self.emergency_deform_replan_next_step),
            "accepted_candidate_name": name,
            "accepted_candidate_type": "emergency_deform_away",
            "chosen_action_norm": float(action_norm),
            "controlled_action_delta_norm": self._controlled_deformation_norm(candidate, nominal),
            "arm_delta_norm": self._controlled_deformation_norm(candidate, nominal),
            "hold_rejected_candidates": rejected,
            "emergency_deform_used_rejected_escape": bool(used_rejected_escape),
        }
        info.update(acc)
        info.update(self._active_safety_info())
        return candidate, info

    def _temporary_update_progress(self, progress):
        if progress is None:
            return
        try:
            value = float(progress)
        except Exception:  # noqa: BLE001
            return
        if not np.isfinite(value):
            return
        self._temporary_progress_history.append(value)
        keep = max(1, self.temporary_progress_window)
        if len(self._temporary_progress_history) > keep:
            self._temporary_progress_history = self._temporary_progress_history[-keep:]

    def _temporary_progress_deadlocked(self):
        if len(self._temporary_progress_history) < max(1, self.temporary_progress_window):
            return False, False
        delta = self._temporary_progress_history[-1] - self._temporary_progress_history[0]
        return bool(delta < self.temporary_min_progress_delta), True

    def _temporary_streak_info(
        self,
        *,
        waiting=False,
        trigger_reason=None,
        nominal_became_safe=False,
        resume_after_wait=False,
    ):
        return {
            "unsafe_streak": int(self.unsafe_streak),
            "brake_streak": int(self.brake_streak),
            "recovery_failure_streak": int(self.recovery_failure_streak),
            "recovery_failure_streak_max": int(self.recovery_failure_streak_max),
            "recovery_optimizer_cooldown_remaining": int(
                self.recovery_optimizer_cooldown_remaining
            ),
            "recovery_retry_cooldown_steps": int(self.recover_retry_cooldown_steps),
            "recovery_attempts_in_unsafe_streak": int(
                self.recovery_attempts_in_unsafe_streak
            ),
            "recovery_max_attempts_per_unsafe_streak": int(
                self.recover_max_attempts_per_unsafe_streak
            ),
            "recovery_optimization_skipped_count": int(
                self.recovery_optimization_skipped_count
            ),
            "temporary_blocker_waiting": bool(waiting),
            "deform_trigger_reason": trigger_reason,
            "nominal_became_safe_after_brake": bool(nominal_became_safe),
            "resume_act_after_wait": bool(resume_after_wait),
            "temporary_wait_step": bool(waiting),
            "deform_suppressed_by_temporary_wait": bool(waiting),
            "deform_after_persistent_block": bool(
                trigger_reason in {"persistent_unsafe", "brake_timeout", "progress_deadlock"}
            ),
        }

    def _safechunk_replan_info(self, **overrides):
        info = {
            "safechunk_replan_enabled": bool(self.safechunk_replan_enabled),
            "deform_replan_count": int(self.deform_replan_count),
            "recover_replan_count": int(self.recovery_replan_count),
            "recovery_replan_count": int(self.recovery_replan_count),
            "recovery_failure_streak": int(self.recovery_failure_streak),
            "recovery_failure_streak_max": int(self.recovery_failure_streak_max),
            "recovery_optimizer_cooldown_remaining": int(
                self.recovery_optimizer_cooldown_remaining
            ),
            "recovery_retry_cooldown_steps": int(self.recover_retry_cooldown_steps),
            "recovery_attempts_in_unsafe_streak": int(
                self.recovery_attempts_in_unsafe_streak
            ),
            "recovery_max_attempts_per_unsafe_streak": int(
                self.recover_max_attempts_per_unsafe_streak
            ),
            "recovery_optimization_skipped_count": int(
                self.recovery_optimization_skipped_count
            ),
            "committed_suffix_replan_attempt_count": int(
                self.committed_suffix_replan_attempt_count
            ),
            "committed_suffix_replan_accepted_count": int(
                self.committed_suffix_replan_accepted_count
            ),
            "committed_suffix_replan_rejected_count": int(
                self.committed_suffix_replan_rejected_count
            ),
            "committed_suffix_replan_budget_suppressed_count": int(
                self.committed_suffix_replan_budget_suppressed_count
            ),
            "committed_opportunistic_resume_count": int(
                self.committed_opportunistic_resume_count
            ),
            "committed_recovery_budget_exit_count": int(
                self.committed_recovery_budget_exit_count
            ),
            "committed_recover_steps_since_act": int(
                self.committed_recover_steps_since_act
            ),
            "committed_suffix_replans_in_current_recovery": int(
                self.committed_suffix_replans_in_current_recovery
            ),
            "recovery_optimization_skipped": False,
            "recovery_optimization_skip_reason": None,
            "stale_recovery_suppressed_count": int(self.stale_recovery_suppressed_count),
            "recovery_target_infeasible_count": int(self.recovery_target_infeasible_count),
            "emergency_brake_steps": int(self.emergency_brake_steps),
            "optimized_candidate_count": int(self.optimized_candidate_count),
            "optimized_solution_count": int(self.optimized_solution_count),
            "fallback_candidate_count": int(self.fallback_candidate_count),
            "fallback_candidate_accepted_count": int(self.fallback_candidate_accepted_count),
            "candidate_fallback_enabled": bool(self.allow_candidate_fallback),
            "optimized_rejected_count": int(self.optimized_rejected_count),
            "deform_candidate_count": int(self.deform_candidate_count),
            "deform_accepted_count": int(self.deform_accepted_count),
            "deform_rejected_count": int(self.deform_rejected_count),
            "recover_candidate_count": int(self.recover_candidate_count),
            "recover_accepted_count": int(self.recover_accepted_count),
            "recover_rejected_count": int(self.recover_rejected_count),
            "safe_prefix_accepted_count": int(self.safe_prefix_accepted_count),
            "first_action_only_accepted_count": int(self.first_action_only_accepted_count),
            "immediate_hard_reject_count": int(self.immediate_hard_reject_count),
            "no_safe_prefix_reject_count": int(self.no_safe_prefix_reject_count),
            "horizon_margin_reject_count": int(self.horizon_margin_reject_count),
            "accepted_deform_steps": int(self.accepted_deform_steps),
            "accepted_recover_steps": int(self.accepted_recover_steps),
            "fallback_brake_after_reject_count": int(self.fallback_brake_after_reject_count),
            "recover_step_since_deform": int(self.recover_step_since_deform),
            "nominal_rejoin_available_count": int(self.nominal_rejoin_available_count),
            "nominal_rejoin_suppressed_count": int(self.nominal_rejoin_suppressed_count),
            "stale_nominal_rejoin_suppressed_count": int(self.stale_nominal_rejoin_suppressed_count),
            "nominal_prefix_unsafe_suppressed_count": int(self.nominal_prefix_unsafe_suppressed_count),
            "recover_positive_projection_count": int(self.recover_positive_projection_count),
            "recover_nonpositive_projection_count": int(self.recover_nonpositive_projection_count),
            "mean_recover_projection_on_nominal": (
                float(np.mean(self._recover_projection_history))
                if self._recover_projection_history else None
            ),
            "mean_recover_cosine_to_nominal": (
                float(np.mean(self._recover_cosine_history))
                if self._recover_cosine_history else None
            ),
            "mean_recover_task_progress_score": (
                float(np.mean(self._recover_task_progress_history))
                if self._recover_task_progress_history else None
            ),
            "mean_recover_ordered_pose_loss": (
                float(np.mean(self._recover_ordered_pose_loss_history))
                if self._recover_ordered_pose_loss_history else None
            ),
            "mean_recover_ordered_delta_loss": (
                float(np.mean(self._recover_ordered_delta_loss_history))
                if self._recover_ordered_delta_loss_history else None
            ),
            "mean_recover_ordered_loss": (
                float(np.mean(self._recover_ordered_loss_history))
                if self._recover_ordered_loss_history else None
            ),
            **self._active_safety_info(),
            **self._safechunk_recovery_corridor_info(),
            "deform_anchor_is_current": self.deform_anchor_state is not None,
            "recover_anchor_is_current": self.recovery_anchor_state is not None,
            "recovery_anchor_is_current": self.recovery_anchor_state is not None,
            "recovery_target_mode": self.recovery_target_mode,
            "recovery_target_feasible": None,
            "stale_recovery_attempted": False,
            "stale_recovery_suppressed": False,
            "recover_to_task_progress": self.recovery_target_mode == "task_progress",
            "recovery_replanned_from_current_state": False,
            "return_to_old_path_suppressed": False,
            "emergency_brake_immediate_unsafe": False,
        }
        info.update(overrides)
        return info

    def _temporary_deform_trigger_reason(self, *, progress_deadlock=False, progress_available=False):
        if self.unsafe_streak >= self.temporary_min_unsafe_steps_before_deform:
            return "persistent_unsafe"
        if self.brake_streak >= self.temporary_max_brake_steps_before_deform:
            return "brake_timeout"
        if progress_available and progress_deadlock:
            return "progress_deadlock"
        return None

    def _try_recover_after_temporary_wait(
        self,
        obs,
        chunk,
        q_seq,
        safety_info,
        info,
        original_shape,
        waited_unsafe_streak,
        waited_brake_streak,
        **kwargs,
    ):
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
            recovery_chunk, recovery_info = self.deform_chunk(
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

    def filter_chunk(self, obs, action_chunk, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        original_shape = np.asarray(action_chunk).shape
        chunk, was_single = self._as_chunk(action_chunk)
        if not self.enabled:
            info = {"safety_mode": "disabled", "mode": "disabled"}
            self.last_info = info
            return chunk.reshape(original_shape), info
        if was_single:
            safe_action = self._call_single_step_operator(chunk[0], obs, **kwargs)
            info = {
                "safety_mode": "single_step_oscbf",
                "mode": "single_step_oscbf",
            }
            self.last_info = info
            return np.asarray(safe_action).reshape(original_shape), info

        self.latest_nominal_chunk = np.asarray(chunk, dtype=np.float32).copy()
        self.latest_nominal_step += 1

        committed_result = self._serve_committed_chunk(obs, chunk, original_shape, **kwargs)
        if committed_result is not None:
            return committed_result
        pending_committed_replan_info = self._pop_pending_committed_replan_info()

        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_info = self.evaluate_horizon_safety(obs, q_seq)
        info = dict(safety_info)
        info.update(self._safechunk_replan_info())
        if pending_committed_replan_info:
            info.update(pending_committed_replan_info)
        self._temporary_update_progress(kwargs.get("task_progress"))

        if safety_info["horizon_safe"]:
            self.committed_recover_steps_since_act = 0
            self.committed_suffix_replans_in_current_recovery = 0
            waited_unsafe_streak = int(self.unsafe_streak)
            waited_brake_streak = int(self.brake_streak)
            nominal_became_safe = bool(waited_unsafe_streak > 0 or waited_brake_streak > 0)
            resume_after_wait = bool(nominal_became_safe and waited_brake_streak > 0)
            if resume_after_wait:
                info.update(
                    self._temporary_streak_info(
                        nominal_became_safe=nominal_became_safe,
                        resume_after_wait=resume_after_wait,
                    )
                )
                recovery_result = self._try_recover_after_temporary_wait(
                    obs,
                    chunk,
                    q_seq,
                    safety_info,
                    info,
                    original_shape,
                    waited_unsafe_streak,
                    waited_brake_streak,
                    **kwargs,
                )
                if recovery_result is not None:
                    return recovery_result
            if self.temporary_reset_on_nominal_safe:
                self.unsafe_streak = 0
                self.brake_streak = 0
                self.recovery_failure_streak = 0
                self.recovery_optimizer_cooldown_remaining = 0
                self.recovery_attempts_in_unsafe_streak = 0
            if self.clear_failed_recovery_on_nominal_safe:
                self.failed_recovery_targets = []
                self.failed_recovery_paths = []
                self.recovery_target_failure_counts = {}
                self._unsafe_recovery_cooldowns = {}
                self.recovery_path_failure_streak = 0
            self.recover_step_since_deform = 0
            self._deadlock_count = 0
            if self.post_recovery_act_window_active:
                remaining = int(self.post_recovery_act_steps_remaining)
                info.update({"safety_mode": "pass_through", "mode": "pass_through"})
                info.update(
                    self._temporary_streak_info(
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
                self.post_recovery_act_steps_remaining = max(0, remaining - 1)
                if self.post_recovery_act_steps_remaining <= 0:
                    self.post_recovery_act_window_active = False
                self._update_last_safe_execution(obs, chunk, info, **kwargs)
                self.last_info = info
                return chunk.reshape(original_shape), info
            info.update({"safety_mode": "pass_through", "mode": "pass_through"})
            info.update(
                self._temporary_streak_info(
                    nominal_became_safe=nominal_became_safe,
                    resume_after_wait=resume_after_wait,
                )
            )
            if self.safechunk_active_safety_enabled and self.check_hold_horizon_safety:
                info["active_safety_nominal_gate"] = True
                return self._hold_return_or_emergency_deform(
                    obs,
                    nominal_chunk=chunk,
                    braked_chunk=chunk,
                    info=info,
                    original_shape=original_shape,
                    **kwargs,
                )
            self._update_last_safe_execution(obs, chunk, info, **kwargs)
            self.last_info = info
            return chunk.reshape(original_shape), info

        if self.post_recovery_act_window_active:
            info.update(self._post_recovery_act_window_info(interrupted=True))

        if self.unsafe_streak == 0:
            self.blocked_nominal_chunk = np.asarray(chunk, dtype=np.float32).copy()
            self.blocked_nominal_step = int(self.latest_nominal_step)
        self.unsafe_streak += 1
        braked_chunk, brake_info = self.horizon_brake(obs, chunk, safety_info)
        info.update(brake_info)
        if brake_info["deadlock"]:
            self._deadlock_count += 1
        else:
            self._deadlock_count = 0
        info["deadlock_count"] = int(self._deadlock_count)

        brake_progress = float(brake_info.get("progress_scale", 1.0))
        prefer_deform_for_task = (
            self.deformation_enabled
            and brake_progress < self.task_progress_brake_threshold
        )
        progress_deadlock, progress_available = self._temporary_progress_deadlocked()
        deform_trigger_reason = "normal"
        if self.temporary_blocker_enabled and self.temporary_prefer_brake_before_deform:
            reason = self._temporary_deform_trigger_reason(
                progress_deadlock=progress_deadlock,
                progress_available=progress_available,
            )
            progress_gate_open = (
                not self.temporary_require_progress_deadlock_before_deform
                or not progress_available
                or progress_deadlock
                or reason == "brake_timeout"
            )
            if reason is None or not progress_gate_open:
                self.brake_streak += 1
                info.update(
                    {
                        "safety_mode": "horizon_brake",
                        "mode": "horizon_brake",
                        "deformation_deferred": True,
                        "fallback_reason": "temporary_blocker_wait",
                    }
                )
                info.update(self._temporary_streak_info(waiting=True))
                return self._hold_return_or_emergency_deform(
                    obs, chunk, braked_chunk, info, original_shape, **kwargs
                )
            deform_trigger_reason = reason

        if (
            brake_info.get("brake_hold_current", False)
            and self.deform_after_deadlock_window
            and self._deadlock_count < self.deadlock_window
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
            self.brake_streak += 1
            info.update(self._temporary_streak_info(waiting=False))
            return self._hold_return_or_emergency_deform(
                obs, chunk, braked_chunk, info, original_shape, **kwargs
            )

        if (
            brake_info["brake_safe"]
            and not prefer_deform_for_task
            and (
                not brake_info["deadlock"]
                or (
                    self.deform_after_deadlock_window
                    and self._deadlock_count < self.deadlock_window
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
            self.brake_streak += 1
            info.update(self._temporary_streak_info(waiting=False))
            return self._hold_return_or_emergency_deform(
                obs, chunk, braked_chunk, info, original_shape, **kwargs
            )

        if not self.deformation_enabled:
            info.update({"safety_mode": "stop", "mode": "stop"})
            info.update(self._temporary_streak_info(trigger_reason=deform_trigger_reason))
            return self._hold_return_or_emergency_deform(
                obs, chunk, braked_chunk, info, original_shape, **kwargs
            )

        recovery_optimizer_skip_reason = None
        if (
            self.recoverable_deform_enabled
            and self.explicit_return
            and self.safechunk_recover_enabled
        ):
            if self.recovery_optimizer_cooldown_remaining > 0:
                recovery_optimizer_skip_reason = "cooldown"
                self.recovery_optimizer_cooldown_remaining = max(
                    0,
                    int(self.recovery_optimizer_cooldown_remaining) - 1,
                )
            elif (
                self.recover_max_attempts_per_unsafe_streak > 0
                and self.recovery_attempts_in_unsafe_streak
                >= self.recover_max_attempts_per_unsafe_streak
            ):
                recovery_optimizer_skip_reason = "attempt_cap"

        if recovery_optimizer_skip_reason is not None:
            self.recovery_optimization_skipped_count += 1
            self.brake_streak += 1
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
            info.update(self._temporary_streak_info(trigger_reason=deform_trigger_reason))
            return self._hold_return_or_emergency_deform(
                obs, chunk, braked_chunk, info, original_shape, **kwargs
            )

        if (
            self.recoverable_deform_enabled
            and self.explicit_return
            and self.safechunk_recover_enabled
        ):
            self.recovery_attempts_in_unsafe_streak += 1

        safe_chunk, deform_info = self.deform_chunk(
            obs,
            chunk,
            safety_info=safety_info,
            braked_chunk=braked_chunk,
            nominal_q_seq=q_seq,
            **kwargs,
        )
        info.update(deform_info)
        if bool(info.get("optimized_accepted", False)):
            self.recovery_failure_streak = 0
            self.recovery_optimizer_cooldown_remaining = 0
            self.recovery_attempts_in_unsafe_streak = 0
        elif info.get("optimized_accepted") is not None or info.get("fallback_used") is not None:
            self.recovery_failure_streak += 1
            self.recovery_failure_streak_max = max(
                self.recovery_failure_streak_max,
                self.recovery_failure_streak,
            )
            if (
                self.recoverable_deform_enabled
                and self.explicit_return
                and self.safechunk_recover_enabled
                and self.recover_retry_cooldown_steps > 0
            ):
                self.recovery_optimizer_cooldown_remaining = max(
                    int(self.recovery_optimizer_cooldown_remaining),
                    int(self.recover_retry_cooldown_steps),
                )
        info.update(self._temporary_streak_info(trigger_reason=deform_trigger_reason))
        if (
            info.get("optimized_accepted", False)
            and self.explicit_return
            and self.commit_accepted_chunks
        ):
            committed, commit_reject_info = self._commit_explicit_recovery_chunk(
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
                    }
                )
                return self._hold_return_or_emergency_deform(
                    obs, chunk, braked_chunk, info, original_shape, **kwargs
                )
            committed_result = self._serve_committed_chunk(obs, chunk, original_shape, **kwargs)
            pending_committed_replan_info = self._pop_pending_committed_replan_info()
            if pending_committed_replan_info:
                info.update(pending_committed_replan_info)
            if committed_result is not None:
                committed_chunk, committed_info = committed_result
                for key in (
                    "optimized_accepted",
                    "deform_stage_accepted",
                    "recover_accepted",
                    "recover_target_index",
                    "resumed_from_recover_index",
                    "yield_accepted",
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
                    "candidate_delta_norm",
                    "nominal_rejoin_score",
                    "nominal_rejoin_available",
                    "nominal_rejoin_suppressed_reason",
                    "nominal_rejoin_clearance",
                    "nominal_rejoin_safe_prefix_len",
                    "deform_min_clearance_stage",
                    "recover_min_clearance",
                    "return_rejoin_loss",
                    "yield_min_clearance",
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
                    "yield_chunk_length",
                    "return_chunk_length",
                    "committed_chunk_total_length",
                ):
                    if key in info:
                        committed_info[key] = info[key]
                self.last_info = committed_info
                return committed_chunk, committed_info

        if (
            not info.get("deform_safe", False)
            and self.unsafe_deformation_fallback == "brake"
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
        valid = self._valid_control_indices(chunk)
        if np.any(valid):
            action_idx = self.controlled_action_indices[valid]
            safe_chunk = self._project_optimized_chunk(safe_chunk, chunk, action_idx)
        self.last_info = info
        return safe_chunk.reshape(original_shape), info

    def __call__(self, *args, **kwargs):
        if len(args) < 2:
            action = kwargs.pop("action", None)
            obs = kwargs.pop("obs", kwargs.pop("observations", None))
            if action is None:
                raise TypeError("SafeChunkDeformFilter requires an action argument")
        else:
            first, second = args[0], args[1]
            first_arr = self._maybe_array(first)
            second_arr = self._maybe_array(second)
            if first_arr is not None and first_arr.ndim in (1, 2):
                action, obs = first, second
            elif second_arr is not None and second_arr.ndim in (1, 2):
                obs, action = first, second
            else:
                action, obs = first, second

        chunk, was_single = self._as_chunk(action)
        if not self.enabled:
            self.last_info = {"safety_mode": "disabled", "mode": "disabled"}
            return chunk.reshape(np.asarray(action).shape)
        if was_single:
            safe_action = self._call_single_step_operator(chunk[0], obs, **kwargs)
            self.last_info = {
                "safety_mode": "single_step_oscbf",
                "mode": "single_step_oscbf",
            }
            return np.asarray(safe_action).reshape(np.asarray(action).shape)

        safe_chunk, info = self.filter_chunk(obs, chunk, **kwargs)
        self.last_info = info
        return safe_chunk.reshape(np.asarray(action).shape)

    def _maybe_array(self, value):
        try:
            return np.asarray(value)
        except Exception:
            return None

    def _call_single_step_operator(self, action, obs=None, **kwargs):
        op = self._get_oscbf_operator()
        if not callable(op):
            return np.asarray(action).copy()
        attempts = (
            lambda: op(action=action, observations=obs, **kwargs),
            lambda: op(action=action, obs=obs, **kwargs),
            lambda: op(action, obs, **kwargs),
            lambda: op(action, **kwargs),
            lambda: op(action),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        raise last_error

    def _get_oscbf_operator(self):
        if callable(self.oscbf_operator) or self.oscbf_operator is None:
            return self.oscbf_operator
        if self._operator_instantiation_failed:
            return None
        target = None
        try:
            target = self.oscbf_operator.get("_target_")
        except AttributeError:
            if isinstance(self.oscbf_operator, dict):
                target = self.oscbf_operator.get("_target_")
        if target is None:
            return self.oscbf_operator
        try:
            import hydra

            self.oscbf_operator = hydra.utils.instantiate(self.oscbf_operator)
        except Exception as exc:  # pragma: no cover - depends on deployment config
            logger.warning(
                "SafeChunk-Deform could not instantiate oscbf_operator; "
                "falling back to identity single-step deformation: %s",
                exc,
            )
            self._operator_instantiation_failed = True
            return None
        return self.oscbf_operator
