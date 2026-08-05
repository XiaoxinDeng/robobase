from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .safechunk_recovery import InfoDict, Recovery, RecoveryResult


class MPCRecoveryController:
    """Closed-loop MPC repair and ACT handoff logic for committed recovery."""

    def __init__(self, recovery: Recovery) -> None:
        """Bind the controller to the recovery executor that owns shared helpers."""
        self.recovery: Recovery = recovery
        self.last_actual_q: np.ndarray | None = None
        self.last_actual_q_key: tuple[int, int, int] | None = None
        self.handoff_attempt_count: int = 0
        self.handoff_accept_count: int = 0
        self.handoff_reject_count: int = 0

    def reset(self) -> None:
        """Clear MPC-local state at episode/reset boundaries."""
        self.last_actual_q = None
        self.last_actual_q_key = None
        self.handoff_attempt_count = 0
        self.handoff_accept_count = 0
        self.handoff_reject_count = 0

    def _action_agreement_terms(
        self,
        lhs: Any,
        rhs: Any,
        prefix: str,
        *,
        arm_indices: Any | None = None,
    ) -> InfoDict:
        """Compare two first actions for ACT handoff control compatibility."""
        try:
            lhs_arr = np.asarray(lhs, dtype=np.float64).reshape(-1)
            rhs_arr = np.asarray(rhs, dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            return {}
        if lhs_arr.size == 0 or rhs_arr.size == 0:
            return {}
        dim = min(lhs_arr.size, rhs_arr.size)
        lhs_arr = lhs_arr[:dim]
        rhs_arr = rhs_arr[:dim]
        delta = lhs_arr - rhs_arr
        lhs_norm = float(np.linalg.norm(lhs_arr))
        rhs_norm = float(np.linalg.norm(rhs_arr))
        denom = lhs_norm * rhs_norm
        cosine = None if denom <= 1e-12 else float(np.dot(lhs_arr, rhs_arr) / denom)
        out: InfoDict = {
            f"{prefix}_l2": float(np.linalg.norm(delta)),
            f"{prefix}_max_abs": float(np.max(np.abs(delta))),
            f"{prefix}_cosine": cosine,
            f"{prefix}_dim": int(dim),
        }
        if arm_indices is not None:
            try:
                arm_idx = np.asarray(arm_indices, dtype=np.int64).reshape(-1)
                arm_idx = np.where(arm_idx < 0, arm_idx + dim, arm_idx)
                arm_idx = arm_idx[(arm_idx >= 0) & (arm_idx < dim)]
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
        return out

    def _annotate_replan_defaults(
        self,
        state_info: InfoDict,
        *,
        idx: int,
    ) -> None:
        """Populate stable MPC replan fields before any accept/reject branch."""
        recovery = self.recovery
        state_info.update(
            {
                "mpc_recovery_enabled": bool(recovery.mpc_recovery_enabled),
                "mpc_recovery_active": False,
                "mpc_recovery_replan_attempted": False,
                "mpc_recovery_replan_accepted": False,
                "mpc_recovery_replan_rejected": False,
                "mpc_recovery_replan_reject_reason": None,
                "mpc_recovery_reference_index": int(idx),
                "mpc_recovery_horizon": int(recovery.mpc_recovery_horizon),
                "mpc_recovery_prefix_len": int(recovery.mpc_recovery_prefix_len),
                "mpc_recovery_replan_count": int(recovery.mpc_recovery_replan_count),
                "mpc_recovery_accepted_count": int(recovery.mpc_recovery_accepted_count),
                "mpc_recovery_rejected_count": int(recovery.mpc_recovery_rejected_count),
                "mpc_recovery_replans_in_current_recovery": int(
                    recovery.mpc_recovery_replans_in_current_recovery
                ),
                "mpc_recovery_max_replans_per_recovery": int(
                    recovery.mpc_recovery_max_replans_per_recovery
                ),
                "mpc_recovery_require_live_progress": bool(
                    recovery.mpc_recovery_require_live_progress
                ),
                "mpc_recovery_min_progress_delta": float(
                    recovery.mpc_recovery_min_progress_delta
                ),
                "mpc_recovery_prefix_replay_step": False,
                "mpc_recovery_recover_local_index": int(
                    recovery.committed_recover_steps_since_act
                ),
                "committed_state_mismatch_ignored_for_mpc_prefix": False,
                "mpc_recovery_no_progress_count": int(
                    getattr(recovery, "mpc_recovery_no_progress_count", 0)
                ),
                "mpc_recovery_no_progress_limit": int(
                    recovery.mpc_recovery_no_progress_limit
                ),
                "mpc_recovery_budget_escape": False,
                "mpc_recovery_budget_escape_count": int(
                    recovery.mpc_recovery_budget_escape_count
                ),
            }
        )

    def _current_actual_direction(
        self,
        current_q: np.ndarray,
        state_idx: np.ndarray,
        *,
        qd_full: Any | None,
        previous_q: np.ndarray | None,
    ) -> tuple[np.ndarray | None, str | None]:
        """Return the best available actual robot direction on controlled states.

        ACT's OOD sensitivity comes from observed state transitions.  Prefer the
        adjacent executed state delta when it is available, and only fall back to
        simulator qd when no adjacent observation transition exists.
        """
        if state_idx.size == 0:
            return None, "no_controlled_state_indices"
        if previous_q is not None:
            previous = np.asarray(previous_q, dtype=np.float32).reshape(-1)
            valid = state_idx[
                (state_idx < previous.shape[0]) & (state_idx < current_q.shape[0])
            ]
            if valid.size > 0:
                direction = current_q[valid] - previous[valid]
                if np.linalg.norm(direction) > 1e-8:
                    return direction.astype(np.float32, copy=True), "actual_delta_q"
        if qd_full is not None:
            qd = np.asarray(qd_full, dtype=np.float32).reshape(-1)
            valid = state_idx[state_idx < qd.shape[0]]
            if valid.size > 0:
                direction = qd[valid].astype(np.float32, copy=True)
                if np.linalg.norm(direction) > 1e-8:
                    return direction, "actual_qd_fallback"
        return None, "actual_direction_unavailable"

    def _shadow_prefix_safety_terms(
        self,
        obs: Any,
        actions: np.ndarray,
        source: str,
    ) -> InfoDict:
        recovery = self.recovery
        chunk = np.asarray(actions, dtype=np.float32)
        prefix = chunk[: max(1, int(getattr(recovery, "mpc_handoff_shadow_prefix_len", 4)))]
        base = f"mpc_handoff_shadow_{source}_prefix"
        if prefix.ndim != 2 or prefix.shape[0] == 0:
            return {
                f"{base}_available": False,
                f"{base}_safe": False,
                f"{base}_reason": "empty_prefix",
            }
        try:
            q_seq = np.asarray(recovery.rollout_nominal_chunk(obs, prefix), dtype=np.float32)
            safety_eval = recovery.evaluate_horizon_safety(obs, q_seq)
            h_seq = np.asarray(
                recovery._clearance_sequence_from_eval(safety_eval, q_seq.shape[0]),
                dtype=np.float32,
            ).reshape(-1)
            if h_seq.size == 0:
                h_seq = np.asarray([float("-inf")], dtype=np.float32)
            min_clearance = float(np.min(h_seq))
            required = max(
                float(recovery.opportunistic_resume_min_clearance),
                float(recovery._acceptance_clearance_threshold()),
            )
            return {
                f"{base}_available": True,
                f"{base}_safe": bool(min_clearance >= required),
                f"{base}_reason": None,
                f"{base}_len": int(prefix.shape[0]),
                f"{base}_min_clearance": min_clearance,
                f"{base}_required_clearance": float(required),
                f"{base}_clearance_margin": float(min_clearance - required),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                f"{base}_available": False,
                f"{base}_safe": False,
                f"{base}_reason": f"evaluation_failed:{exc}",
            }

    def _shadow_handoff_prefix_terms(
        self,
        obs: Any,
        release_action: np.ndarray,
        act_chunk: np.ndarray,
        target_chunk: np.ndarray,
        target_action_index: int,
    ) -> InfoDict:
        act = np.asarray(act_chunk, dtype=np.float32)
        target = np.asarray(target_chunk, dtype=np.float32)
        release = np.asarray(release_action, dtype=np.float32).reshape(-1)
        prefix_len = max(1, int(getattr(self.recovery, "mpc_handoff_shadow_prefix_len", 4)))
        start = min(max(int(target_action_index), 0), max(0, target.shape[0] - 1))
        target_prefix = target[start : start + prefix_len]
        if target_prefix.shape[0] < prefix_len and target.ndim == 2 and target.shape[0] > 0:
            target_prefix = np.concatenate(
                (target_prefix, np.repeat(target[-1:], prefix_len - target_prefix.shape[0], axis=0)),
                axis=0,
            )
        release_act_prefix = np.concatenate(
            (release[None, :], act[: max(0, prefix_len - 1)]), axis=0
        )
        terms = {}
        terms.update(self._shadow_prefix_safety_terms(obs, act, "act"))
        terms.update(self._shadow_prefix_safety_terms(obs, target_prefix, "target"))
        terms.update(self._shadow_prefix_safety_terms(obs, release_act_prefix, "release_act"))
        terms.update(
            {
                "mpc_handoff_shadow_prefix_available": terms.get("mpc_handoff_shadow_release_act_prefix_available"),
                "mpc_handoff_shadow_prefix_safe": terms.get("mpc_handoff_shadow_release_act_prefix_safe"),
                "mpc_handoff_shadow_prefix_reason": terms.get("mpc_handoff_shadow_release_act_prefix_reason"),
                "mpc_handoff_shadow_prefix_len": terms.get("mpc_handoff_shadow_release_act_prefix_len"),
                "mpc_handoff_shadow_prefix_target_start": int(start),
                "mpc_handoff_shadow_prefix_min_clearance": terms.get("mpc_handoff_shadow_release_act_prefix_min_clearance"),
                "mpc_handoff_shadow_prefix_required_clearance": terms.get("mpc_handoff_shadow_release_act_prefix_required_clearance"),
                "mpc_handoff_shadow_prefix_clearance_margin": terms.get("mpc_handoff_shadow_release_act_prefix_clearance_margin"),
            }
        )
        return terms

    def _nominal_prefix_safety(
        self,
        obs: Any,
        nominal_chunk: np.ndarray,
        *,
        prefix_len: int,
    ) -> tuple[bool, float, InfoDict]:
        """Check whether ACT can safely own the next nominal prefix."""
        recovery = self.recovery
        nominal = np.asarray(nominal_chunk, dtype=np.float32)
        prefix_horizon = max(1, min(int(prefix_len), int(nominal.shape[0])))
        if prefix_horizon <= 0:
            return False, float("-inf"), {"reason": "empty_nominal_prefix"}
        nominal_q_seq = recovery.rollout_nominal_chunk(obs, nominal[:prefix_horizon])
        safety_eval = recovery.evaluate_horizon_safety(obs, nominal_q_seq)
        h_seq = np.asarray(
            recovery._clearance_sequence_from_eval(
                safety_eval,
                np.asarray(nominal_q_seq).shape[0],
            ),
            dtype=np.float32,
        ).reshape(-1)
        if h_seq.size == 0:
            h_seq = np.asarray([float("-inf")], dtype=np.float32)
        min_clearance = float(np.min(h_seq))
        required = max(
            float(recovery.opportunistic_resume_min_clearance),
            float(recovery._acceptance_clearance_threshold()),
        )
        info = dict(safety_eval)
        info.update(
            {
                "mpc_handoff_act_prefix_len": int(prefix_horizon),
                "mpc_handoff_act_prefix_min_clearance": float(min_clearance),
                "mpc_handoff_act_prefix_required_clearance": float(required),
            }
        )
        return bool(min_clearance >= required), min_clearance, info

    def _reject_handoff(
        self,
        state_info: InfoDict,
        reason: str,
        *,
        handoff_reason: str,
        extra: InfoDict | None = None,
    ) -> None:
        """Record a failed MPC-to-ACT handoff check."""
        self.handoff_reject_count += 1
        state_info.update(
            {
                "mpc_handoff_attempted": True,
                "mpc_handoff_accepted": False,
                "mpc_handoff_rejected": True,
                "mpc_handoff_reject_reason": str(reason),
                "mpc_handoff_reason": str(handoff_reason),
                "mpc_handoff_attempt_count": int(self.handoff_attempt_count),
                "mpc_handoff_accept_count": int(self.handoff_accept_count),
                "mpc_handoff_reject_count": int(self.handoff_reject_count),
            }
        )
        if extra:
            state_info.update(extra)

    def try_handoff_to_act(
        self,
        obs: Any,
        nominal_chunk: np.ndarray,
        original_shape: Any,
        mode: str,
        idx: int,
        total: int,
        state_info: InfoDict,
        *,
        handoff_reason: str,
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Release committed recovery when the live state is ACT-compatible."""
        recovery = self.recovery
        if mode != "recover":
            state_info.update(
                {
                    "mpc_handoff_attempted": False,
                    "mpc_handoff_accepted": False,
                    "mpc_handoff_rejected": False,
                    "mpc_handoff_reject_reason": None,
                    "mpc_handoff_skip_reason": "not_recover_mode",
                }
            )
            return None

        self.handoff_attempt_count += 1
        state_info.update(
            {
                "mpc_handoff_attempted": True,
                "mpc_handoff_accepted": False,
                "mpc_handoff_rejected": False,
                "mpc_handoff_reject_reason": None,
                "mpc_handoff_reason": str(handoff_reason),
                "mpc_handoff_attempt_count": int(self.handoff_attempt_count),
                "mpc_handoff_accept_count": int(self.handoff_accept_count),
                "mpc_handoff_reject_count": int(self.handoff_reject_count),
            }
        )
        if not recovery.opportunistic_act_resume:
            self._reject_handoff(
                state_info,
                "opportunistic_resume_disabled",
                handoff_reason=handoff_reason,
            )
            return None

        nominal, _ = recovery._as_chunk(nominal_chunk)
        if nominal.shape[0] == 0:
            self._reject_handoff(
                state_info,
                "empty_nominal_chunk",
                handoff_reason=handoff_reason,
            )
            return None

        current_q = np.asarray(
            recovery._current_replay_q(obs, **kwargs),
            dtype=np.float32,
        ).reshape(-1)
        committed_id = int(getattr(recovery, "committed_sequence_id", 0))
        previous_q_available = self.last_actual_q is not None
        previous_q_adjacent = bool(
            self.last_actual_q_key is not None
            and self.last_actual_q_key[0] == committed_id
            and self.last_actual_q_key[1] == int(total)
            and self.last_actual_q_key[2] == int(idx) - 1
        )
        previous_q = (
            self.last_actual_q.copy()
            if previous_q_available and previous_q_adjacent
            else None
        )
        self.last_actual_q = current_q.copy()
        self.last_actual_q_key = (committed_id, int(total), int(idx))

        try:
            target_info = recovery.get_nominal_rejoin_target(
                obs,
                candidate_q=current_q,
                require_live_prefix_safe=True,
                live_prefix_len=int(getattr(recovery, "recover_act_frame_stack", 1)),
                allow_best_live_prefix_when_unsafe=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._reject_handoff(
                state_info,
                "target_selection_failed",
                handoff_reason=handoff_reason,
                extra={"mpc_handoff_error": str(exc)},
            )
            return None
        if not bool(target_info.get("available", False)):
            self._reject_handoff(
                state_info,
                str(target_info.get("suppressed_reason", "no_safe_act_window")),
                handoff_reason=handoff_reason,
                extra={
                    "mpc_handoff_act_window_available": False,
                    "mpc_handoff_act_prefix_safe": False,
                    "mpc_handoff_act_prefix_min_clearance": target_info.get(
                        "nominal_rejoin_live_prefix_best_min_clearance"
                    ),
                    "mpc_handoff_nominal_rejoin_clearance": target_info.get(
                        "nominal_rejoin_clearance"
                    ),
                    "mpc_handoff_nominal_rejoin_required_clearance": target_info.get(
                        "nominal_rejoin_required_clearance"
                    ),
                    "mpc_handoff_live_prefix_required": target_info.get(
                        "nominal_rejoin_live_prefix_required"
                    ),
                    "mpc_handoff_live_prefix_safe_count": target_info.get(
                        "nominal_rejoin_live_prefix_safe_count"
                    ),
                    "mpc_handoff_live_prefix_eval_count": target_info.get(
                        "nominal_rejoin_live_prefix_eval_count"
                    ),
                    "mpc_handoff_live_prefix_best_min_clearance": target_info.get(
                        "nominal_rejoin_live_prefix_best_min_clearance"
                    ),
                    "mpc_handoff_live_prefix_required_clearance": target_info.get(
                        "nominal_rejoin_live_prefix_required_clearance"
                    ),
                    "mpc_handoff_live_prefix_best_start": target_info.get(
                        "nominal_rejoin_live_prefix_best_start"
                    ),
                    "safe_rejoin_window_found": bool(
                        target_info.get("safe_rejoin_window_found", False)
                    ),
                },
            )
            return None

        target_q_seq = np.asarray(target_info.get("target_q_seq"), dtype=np.float32)
        if target_q_seq.ndim != 2 or target_q_seq.shape[0] == 0:
            self._reject_handoff(
                state_info,
                "empty_target_q_window",
                handoff_reason=handoff_reason,
            )
            return None
        target_chunk = np.asarray(target_info.get("target_chunk"), dtype=np.float32)
        if target_chunk.ndim != 2 or target_chunk.shape[0] == 0:
            self._reject_handoff(
                state_info,
                "empty_target_action_window",
                handoff_reason=handoff_reason,
            )
            return None
        if target_chunk.shape[1] != nominal.shape[1]:
            self._reject_handoff(
                state_info,
                "target_action_dim_mismatch",
                handoff_reason=handoff_reason,
                extra={
                    "mpc_handoff_target_action_dim": int(target_chunk.shape[1]),
                    "mpc_handoff_nominal_action_dim": int(nominal.shape[1]),
                },
            )
            return None

        state_idx, state_weights = recovery._mpc_state_indices_and_weights(
            current_q.shape[0],
            target_q_seq.shape[1],
            kind="handoff",
        )
        if state_idx.size == 0:
            self._reject_handoff(
                state_info,
                "no_controlled_state_indices",
                handoff_reason=handoff_reason,
            )
            return None

        state_weights_2d = state_weights.reshape(1, -1)
        target_controlled = target_q_seq[:, state_idx] * state_weights_2d
        current_controlled = current_q[state_idx].reshape(1, -1) * state_weights_2d
        waypoint_loss = np.square(target_controlled - current_controlled).mean(axis=1)
        nearest_local = int(np.argmin(waypoint_loss))
        pose_loss = float(waypoint_loss[nearest_local])
        pose_dist = float(np.sqrt(max(pose_loss, 0.0)))
        pose_threshold = float(recovery.recover_ordered_pose_threshold)
        pose_tube_loss_threshold = max(pose_threshold * 4.0, pose_threshold + 0.05)
        pose_tube_dist_threshold = float(np.sqrt(max(pose_tube_loss_threshold, 0.0)))
        pose_tube_ok = bool(pose_loss <= pose_tube_loss_threshold)

        if target_q_seq.shape[0] >= 2:
            tangent_start = min(nearest_local, target_q_seq.shape[0] - 2)
            tangent_end = tangent_start + 1
            target_tangent = (
                target_q_seq[tangent_end, state_idx]
                - target_q_seq[tangent_start, state_idx]
            ) * state_weights
        else:
            target_tangent = np.zeros((state_idx.shape[0],), dtype=np.float32)
        actual_direction, direction_source = self._current_actual_direction(
            current_q,
            state_idx,
            qd_full=kwargs.get("qd_full"),
            previous_q=previous_q,
        )
        if actual_direction is not None:
            actual_direction = actual_direction * state_weights[: actual_direction.shape[0]]
        target_norm = float(np.linalg.norm(target_tangent))
        actual_norm = (
            0.0
            if actual_direction is None
            else float(np.linalg.norm(actual_direction))
        )
        if actual_direction is None or actual_norm <= 1e-8 or target_norm <= 1e-8:
            heading_cosine = None
            progress_projection = None
            heading_ok = False
            progress_ok = False
        else:
            heading_cosine = float(
                np.dot(actual_direction, target_tangent)
                / (actual_norm * target_norm + 1e-8)
            )
            heading_cosine = float(np.clip(heading_cosine, -1.0, 1.0))
            progress_projection = float(
                np.dot(actual_direction, target_tangent) / (target_norm + 1e-8)
            )
            heading_ok = bool(
                heading_cosine
                >= float(getattr(recovery, "recover_min_act_heading_cosine", 0.0))
            )
            progress_ok = bool(
                progress_projection
                >= -abs(float(recovery.mpc_recovery_min_progress_delta))
            )

        target_action_index = min(max(int(nearest_local), 0), target_chunk.shape[0] - 1)
        target_action = np.asarray(
            target_chunk[target_action_index], dtype=np.float32
        ).reshape(1, -1)
        handoff_affordance_q = current_q
        handoff_affordance_q_source = "current_q"
        try:
            predicted_target_q = np.asarray(
                recovery.rollout_nominal_chunk(obs, target_action),
                dtype=np.float32,
            )
            if predicted_target_q.ndim == 2 and predicted_target_q.shape[0] > 0:
                handoff_affordance_q = predicted_target_q[-1]
                handoff_affordance_q_source = "selected_act_action_one_step"
        except Exception as exc:  # pragma: no cover - optional task adapter path
            logger.debug("MPC handoff one-step affordance rollout failed: %s", exc)

        prefix_len = int(
            target_info.get(
                "nominal_rejoin_target_window_len",
                target_chunk.shape[0],
            )
        )
        # ACT-prefix safety is intentionally deferred to the main safechunk loop.
        # Handoff only checks that the recovery/MPC output itself is safe and in
        # the nominal trajectory tube; the next ACT chunk will undergo the normal
        # brake->deform->recover pipeline.
        act_prefix_safe = None
        act_prefix_min_clearance = None
        start = target_info.get("nominal_rejoin_window_start")
        nearest_index = None
        if start is not None:
            nearest_index = int(start) + int(nearest_local)

        handoff_terms: InfoDict = {
            "mpc_handoff_act_window_available": True,
            "mpc_handoff_pose_loss": float(pose_loss),
            "mpc_handoff_pose_dist": float(pose_dist),
            "mpc_handoff_pose_tube_loss_threshold": float(pose_tube_loss_threshold),
            "mpc_handoff_pose_tube_dist_threshold": float(pose_tube_dist_threshold),
            "mpc_handoff_pose_tube_ok": bool(pose_tube_ok),
            "mpc_handoff_actual_direction_available": bool(
                actual_direction is not None
            ),
            "mpc_handoff_actual_direction_source": direction_source,
            "mpc_handoff_committed_sequence_id": int(committed_id),
            "mpc_handoff_previous_q_available": bool(previous_q_available),
            "mpc_handoff_previous_q_adjacent": bool(previous_q_adjacent),
            "mpc_handoff_target_source": "selected_safe_rejoin_window",
            "mpc_handoff_state_indices": state_idx.astype(int).tolist(),
            "mpc_handoff_state_weights": state_weights.astype(float).tolist(),
            "mpc_handoff_target_action_len": int(target_chunk.shape[0]),
            "mpc_handoff_target_action_index": int(target_action_index),
            "mpc_handoff_affordance_q_source": handoff_affordance_q_source,
            "mpc_handoff_target_tangent_norm": float(target_norm),
            "mpc_handoff_actual_direction_norm": float(actual_norm),
            "mpc_handoff_heading_cosine": heading_cosine,
            "mpc_handoff_heading_cosine_threshold": float(
                getattr(recovery, "recover_min_act_heading_cosine", 0.0)
            ),
            "mpc_handoff_heading_ok": bool(heading_ok),
            "mpc_handoff_progress_projection": progress_projection,
            "mpc_handoff_progress_ok": bool(progress_ok),
            "mpc_handoff_act_prefix_checked": False,
            "mpc_handoff_act_prefix_check_deferred_to_main_filter": True,
            "mpc_handoff_act_prefix_safe": act_prefix_safe,
            "mpc_handoff_act_prefix_min_clearance": act_prefix_min_clearance,
            "mpc_handoff_live_prefix_required": target_info.get(
                "nominal_rejoin_live_prefix_required"
            ),
            "mpc_handoff_live_prefix_safe": target_info.get(
                "nominal_rejoin_live_prefix_safe"
            ),
            "mpc_handoff_live_prefix_safe_count": target_info.get(
                "nominal_rejoin_live_prefix_safe_count"
            ),
            "mpc_handoff_live_prefix_eval_count": target_info.get(
                "nominal_rejoin_live_prefix_eval_count"
            ),
            "mpc_handoff_live_prefix_best_min_clearance": target_info.get(
                "nominal_rejoin_live_prefix_best_min_clearance"
            ),
            "mpc_handoff_live_prefix_required_clearance": target_info.get(
                "nominal_rejoin_live_prefix_required_clearance"
            ),
            "mpc_handoff_live_prefix_best_start": target_info.get(
                "nominal_rejoin_live_prefix_best_start"
            ),
            "mpc_handoff_rejoin_index": nearest_index,
            "nominal_rejoin_window_start": target_info.get(
                "nominal_rejoin_window_start"
            ),
            "nominal_rejoin_window_end": target_info.get("nominal_rejoin_window_end"),
            "nominal_rejoin_window_len": target_info.get("nominal_rejoin_window_len"),
            "nominal_rejoin_window_type": target_info.get(
                "nominal_rejoin_window_type"
            ),
            "safe_rejoin_window_found": bool(
                target_info.get("safe_rejoin_window_found", False)
            ),
            "short_staging_window_found": bool(
                target_info.get("short_staging_window_found", False)
            ),
        }
        resume_readiness_terms = recovery._resume_readiness_terms(
            handoff_terms,
            q=handoff_affordance_q,
            obs=obs,
            source="mpc_handoff_selected_act_action",
        )
        handoff_terms.update(
            {f"mpc_handoff_{key}": value for key, value in resume_readiness_terms.items()}
        )
        handoff_terms["mpc_handoff_resume_readiness_required"] = bool(
            getattr(recovery, "mpc_handoff_require_resume_readiness", False)
        )
        state_info.update(handoff_terms)

        release_chunk = nominal.copy()
        release_len = 0
        release_source = "missing_committed_recovery"
        release_action_safety: InfoDict = {}
        committed = recovery.committed_chunk
        if committed is not None:
            committed_arr = np.asarray(committed, dtype=np.float32)
            if committed_arr.ndim == 2 and committed_arr.shape[0] > 0:
                src_idx = min(max(int(idx), 0), committed_arr.shape[0] - 1)
                release_len = min(
                    int(release_chunk.shape[0]),
                    int(committed_arr.shape[0]) - int(src_idx),
                )
                if release_len > 0:
                    release_chunk[:release_len] = committed_arr[
                        src_idx : src_idx + release_len
                    ]
                    release_source = "committed_recovery_suffix"
                    release_safe, release_action_safety = recovery._committed_action_safety(
                        obs,
                        current_q,
                        committed_arr[src_idx],
                        **kwargs,
                    )
                    if not release_safe:
                        self._reject_handoff(
                            state_info,
                            "mpc_release_action_unsafe",
                            handoff_reason=handoff_reason,
                            extra={**handoff_terms, **release_action_safety},
                        )
                        return None
        if release_len <= 0:
            self._reject_handoff(
                state_info,
                "missing_committed_recovery_action",
                handoff_reason=handoff_reason,
                extra=handoff_terms,
            )
            return None

        original_release_first_action = release_chunk[0].copy()
        bridge_action_source = release_source
        bridge_target_action_safe = None
        bridge_target_action_safety: InfoDict = {}
        if bool(
            getattr(recovery, "mpc_handoff_use_selected_act_action_if_safe", False)
        ):
            bridge_target_action_safe, bridge_target_action_safety = (
                recovery._committed_action_safety(
                    obs,
                    current_q,
                    target_action.reshape(-1),
                    **kwargs,
                )
            )
            if bridge_target_action_safe:
                release_chunk[0] = target_action.reshape(-1).astype(
                    release_chunk.dtype, copy=True
                )
                bridge_action_source = "selected_act_action_safe_bridge"
            else:
                bridge_action_source = "recovery_release_target_unsafe"
        shadow_prefix_terms = self._shadow_handoff_prefix_terms(
            obs,
            release_chunk[0],
            nominal,
            target_chunk,
            target_action_index,
        )
        handoff_terms.update(shadow_prefix_terms)
        release_action_safety.update(
            {
                "mpc_handoff_bridge_action_source": bridge_action_source,
                "mpc_handoff_bridge_target_action_safe": bridge_target_action_safe,
                "mpc_handoff_bridge_target_action_safety": bridge_target_action_safety,
                "mpc_handoff_original_release_action_l2": float(
                    np.linalg.norm(nominal[0] - original_release_first_action)
                ),
            }
        )
        if bool(getattr(recovery, "mpc_handoff_require_shadow_prefix", False)) and not bool(
            shadow_prefix_terms.get("mpc_handoff_shadow_prefix_safe", False)
        ):
            self._reject_handoff(
                state_info,
                "shadow_prefix_unsafe",
                handoff_reason=handoff_reason,
                extra={**handoff_terms, **release_action_safety},
            )
            return None
        if (
            bool(getattr(recovery, "mpc_handoff_use_selected_act_action_if_safe", False))
            and not bool(bridge_target_action_safe)
        ):
            self._reject_handoff(
                state_info,
                "mpc_target_bridge_action_unsafe",
                handoff_reason=handoff_reason,
                extra={**handoff_terms, **release_action_safety},
            )
            return None

        action_idx = getattr(recovery, "controlled_action_indices", None)
        if action_idx is None:
            action_idx = getattr(recovery, "action_indices", None)
        handoff_action_terms = self._action_agreement_terms(
            nominal[0],
            release_chunk[0],
            "mpc_handoff_act_vs_release_action",
            arm_indices=action_idx,
        )
        handoff_action_terms.update(
            self._action_agreement_terms(
                nominal[0],
                target_chunk[target_action_index],
                "mpc_handoff_act_vs_target_action",
                arm_indices=action_idx,
            )
        )
        action_source = str(
            getattr(recovery, "mpc_handoff_action_agreement_source", "release")
        ).lower()
        if action_source not in {"release", "target", "both"}:
            action_source = "release"
        action_l2_threshold = float(recovery.mpc_handoff_action_l2_threshold)
        action_cosine_threshold = float(recovery.mpc_handoff_action_cosine_threshold)
        action_arm_l2_threshold = float(recovery.mpc_handoff_action_arm_l2_threshold)
        action_prefix = (
            "mpc_handoff_act_vs_target_action"
            if action_source == "target"
            else "mpc_handoff_act_vs_release_action"
        )
        if action_source == "both":
            agreement_prefixes = (
                "mpc_handoff_act_vs_release_action",
                "mpc_handoff_act_vs_target_action",
            )
            l2_values = [handoff_action_terms.get(f"{prefix}_l2") for prefix in agreement_prefixes]
            cosine_values = [handoff_action_terms.get(f"{prefix}_cosine") for prefix in agreement_prefixes]
            arm_l2_values = [handoff_action_terms.get(f"{prefix}_arm_l2") for prefix in agreement_prefixes]
            action_l2 = max((float(value) for value in l2_values if value is not None), default=None)
            action_cosine = min((float(value) for value in cosine_values if value is not None), default=None)
            action_arm_l2 = max((float(value) for value in arm_l2_values if value is not None), default=None)
            action_l2_ok = bool(all(value is not None and float(value) <= action_l2_threshold for value in l2_values))
            action_cosine_ok = bool(all(value is not None and float(value) >= action_cosine_threshold for value in cosine_values))
            action_arm_l2_ok = bool(all(value is not None and float(value) <= action_arm_l2_threshold for value in arm_l2_values))
        else:
            action_l2 = handoff_action_terms.get(f"{action_prefix}_l2")
            action_cosine = handoff_action_terms.get(f"{action_prefix}_cosine")
            action_arm_l2 = handoff_action_terms.get(f"{action_prefix}_arm_l2")
            action_l2_ok = bool(action_l2 is not None and float(action_l2) <= action_l2_threshold)
            action_cosine_ok = bool(action_cosine is None or float(action_cosine) >= action_cosine_threshold)
            action_arm_l2_ok = bool(action_arm_l2 is None or float(action_arm_l2) <= action_arm_l2_threshold)
        action_agreement_ok = bool(action_l2_ok and action_cosine_ok and action_arm_l2_ok)
        resume_readiness_required = bool(
            getattr(recovery, "mpc_handoff_require_resume_readiness", False)
        )
        resume_allowed = bool(handoff_terms.get("mpc_handoff_resume_allowed", True))
        resume_block_reason = handoff_terms.get("mpc_handoff_resume_block_reason")
        action_live_ok = bool((not resume_readiness_required) or resume_allowed)
        action_override_allowed = bool(
            recovery.mpc_handoff_action_agreement_override_enabled
            and action_agreement_ok
            and action_live_ok
            and release_len > 0
        )
        heading_ok_raw = bool(heading_ok)
        progress_ok_raw = bool(progress_ok)
        heading_ok_effective = bool(heading_ok_raw or action_override_allowed)
        progress_ok_effective = bool(progress_ok_raw or action_override_allowed)
        override_reason = None
        if not bool(recovery.mpc_handoff_action_agreement_override_enabled):
            override_reason = "disabled"
        elif not action_agreement_ok:
            override_reason = "action_agreement_failed"
        elif not action_live_ok:
            override_reason = str(resume_block_reason or "resume_readiness_not_ready")
        elif not action_override_allowed:
            override_reason = "not_allowed"
        elif (not heading_ok_raw) or (not progress_ok_raw):
            override_reason = f"act_{action_source}_action_agrees"
        else:
            override_reason = "not_needed"
        handoff_terms.update(handoff_action_terms)
        handoff_terms.update(
            {
                "mpc_handoff_release_action_safe": True,
                "mpc_handoff_bridge_action_source": bridge_action_source,
                "mpc_handoff_bridge_target_action_safe": bridge_target_action_safe,
                "mpc_handoff_use_selected_act_action_if_safe": bool(
                    getattr(recovery, "mpc_handoff_use_selected_act_action_if_safe", False)
                ),
                "mpc_handoff_action_agreement_override_enabled": bool(
                    recovery.mpc_handoff_action_agreement_override_enabled
                ),
                "mpc_handoff_action_agreement_source": action_source,
                "mpc_handoff_action_agreement_l2_threshold": float(action_l2_threshold),
                "mpc_handoff_action_agreement_cosine_threshold": float(action_cosine_threshold),
                "mpc_handoff_action_agreement_arm_l2_threshold": float(action_arm_l2_threshold),
                "mpc_handoff_action_agreement_l2_ok": bool(action_l2_ok),
                "mpc_handoff_action_agreement_cosine_ok": bool(action_cosine_ok),
                "mpc_handoff_action_agreement_arm_l2_ok": bool(action_arm_l2_ok),
                "mpc_handoff_action_agreement_ok": bool(action_agreement_ok),
                "mpc_handoff_action_agreement_live_ok": bool(action_live_ok),
                "mpc_handoff_action_agreement_override_allowed": bool(action_override_allowed),
                "mpc_handoff_action_agreement_override_reason": override_reason,
                "mpc_handoff_heading_ok_raw": bool(heading_ok_raw),
                "mpc_handoff_progress_ok_raw": bool(progress_ok_raw),
                "mpc_handoff_heading_ok_effective": bool(heading_ok_effective),
                "mpc_handoff_progress_ok_effective": bool(progress_ok_effective),
                "mpc_handoff_heading_overridden_by_action_agreement": bool(
                    action_override_allowed and not heading_ok_raw
                ),
                "mpc_handoff_progress_overridden_by_action_agreement": bool(
                    action_override_allowed and not progress_ok_raw
                ),
                "mpc_handoff_heading_ok": bool(heading_ok_effective),
                "mpc_handoff_progress_ok": bool(progress_ok_effective),
            }
        )
        bridge_ramp_enabled = bool(
            getattr(recovery, "mpc_handoff_bridge_ramp_on_resume_not_ready", False)
        )
        bridge_ramp_max_steps = max(
            0, int(getattr(recovery, "mpc_handoff_bridge_ramp_max_steps", 0))
        )
        bridge_ramp_budget_ok = bool(
            bridge_ramp_max_steps <= 0
            or int(recovery.committed_recover_steps_since_act) < bridge_ramp_max_steps
        )
        bridge_ramp_reason_ok = bool(
            resume_readiness_required
            and not resume_allowed
            and str(resume_block_reason or "") == "resume_tube_not_ready"
        )
        bridge_ramp_allowed = bool(
            bridge_ramp_enabled
            and bridge_ramp_reason_ok
            and action_agreement_ok
            and bool(bridge_target_action_safe)
            and release_len > 0
            and bridge_ramp_budget_ok
        )
        handoff_terms.update(
            {
                "mpc_handoff_bridge_ramp_enabled": bool(bridge_ramp_enabled),
                "mpc_handoff_bridge_ramp_max_steps": int(bridge_ramp_max_steps),
                "mpc_handoff_bridge_ramp_budget_ok": bool(bridge_ramp_budget_ok),
                "mpc_handoff_bridge_ramp_reason_ok": bool(bridge_ramp_reason_ok),
                "mpc_handoff_bridge_ramp_allowed": bool(bridge_ramp_allowed),
                "mpc_handoff_bridge_ramp_executed": False,
                "mpc_handoff_bridge_ramp_block_reason": None
                if bridge_ramp_allowed
                else (
                    "disabled"
                    if not bridge_ramp_enabled
                    else "not_resume_tube_blocked"
                    if not bridge_ramp_reason_ok
                    else "action_agreement_failed"
                    if not action_agreement_ok
                    else "target_action_unsafe"
                    if not bool(bridge_target_action_safe)
                    else "bridge_ramp_budget_exhausted"
                    if not bridge_ramp_budget_ok
                    else "unknown"
                ),
                "mpc_handoff_bridge_ramp_steps_since_act": int(
                    recovery.committed_recover_steps_since_act
                ),
            }
        )
        state_info.update(handoff_terms)

        if bridge_ramp_allowed:
            recovery.committed_chunk_index = min(int(idx) + 1, int(total))
            if mode == "recover":
                recovery.committed_recover_steps_since_act += 1
            completed = bool(recovery.committed_chunk_index >= int(total))
            self.handoff_reject_count += 1
            info = dict(release_action_safety)
            info.update(bridge_target_action_safety)
            info.update(
                {
                    **state_info,
                    **handoff_terms,
                    "safety_mode": "horizon_deform",
                    "mode": "committed_explicit_recovery",
                    "deform_mode": "mpc_handoff_bridge_ramp_to_resume_tube",
                    "deformation_source": "committed_explicit_recovery",
                    "recovery_mode": "recover",
                    "recovery_phase": "mpc_handoff_bridge_ramp_to_resume_tube",
                    "committed_chunk_active": True,
                    "committed_chunk_mode": mode,
                    "committed_chunk_index": int(idx),
                    "committed_chunk_length": int(total),
                    "committed_chunk_completed": completed,
                    "committed_released_for_act_resume": False,
                    "committed_opportunistic_resume": False,
                    "committed_recovery_budget_exit": False,
                    "resume_from_committed_rejoin": False,
                    "request_action_history_reset_after_recovery": False,
                    "act_resume_supported": False,
                    "mpc_handoff_attempted": True,
                    "mpc_handoff_accepted": False,
                    "mpc_handoff_rejected": True,
                    "mpc_handoff_reject_reason": "bridge_ramp_resume_tube_not_ready",
                    "mpc_handoff_reason": str(handoff_reason),
                    "mpc_handoff_attempt_count": int(self.handoff_attempt_count),
                    "mpc_handoff_accept_count": int(self.handoff_accept_count),
                    "mpc_handoff_reject_count": int(self.handoff_reject_count),
                    "mpc_handoff_bridge_ramp_executed": True,
                    "mpc_handoff_bridge_ramp_block_reason": None,
                    "mpc_handoff_bridge_ramp_release_to_act": False,
                    "mpc_handoff_deferred_release_reason": str(
                        resume_block_reason or "resume_readiness_not_ready"
                    ),
                    "recover_steps_executed": 1,
                    "return_steps_executed": 0,
                    "deform_steps_executed": 0,
                    "fallback_used": False,
                    "optimized_accepted": True,
                    "deform_safe": True,
                    "is_safe": True,
                    "is_recoverable": True,
                }
            )
            if recovery.committed_rejoin_diagnostics:
                for key, value in recovery.committed_rejoin_diagnostics.items():
                    info.setdefault(key, value)
            if completed:
                recovery._clear_committed_chunk()
            recovery.last_info = info
            return release_chunk.reshape(original_shape), info

        if resume_readiness_required and not resume_allowed:
            self._reject_handoff(
                state_info,
                str(resume_block_reason or "resume_readiness_not_ready"),
                handoff_reason=handoff_reason,
                extra=handoff_terms,
            )
            return None
        if not pose_tube_ok:
            self._reject_handoff(
                state_info,
                "outside_rejoin_tube",
                handoff_reason=handoff_reason,
                extra=handoff_terms,
            )
            return None
        if not heading_ok_effective:
            self._reject_handoff(
                state_info,
                "actual_heading_not_aligned",
                handoff_reason=handoff_reason,
                extra=handoff_terms,
            )
            return None
        if not progress_ok_effective:
            self._reject_handoff(
                state_info,
                "actual_progress_not_forward",
                handoff_reason=handoff_reason,
                extra=handoff_terms,
            )
            return None

        self.handoff_accept_count += 1
        state_info.update(
            {
                "mpc_handoff_attempted": True,
                "mpc_handoff_accepted": True,
                "mpc_handoff_rejected": False,
                "mpc_handoff_reject_reason": None,
                "mpc_handoff_reason": str(handoff_reason),
                "mpc_handoff_attempt_count": int(self.handoff_attempt_count),
                "mpc_handoff_accept_count": int(self.handoff_accept_count),
                "mpc_handoff_reject_count": int(self.handoff_reject_count),
            }
        )
        recovery.committed_opportunistic_resume_count += 1
        recovery.committed_recover_steps_since_act = 0
        recovery.committed_suffix_replans_in_current_recovery = 0
        recovery.mpc_recovery_replans_in_current_recovery = 0
        info = dict(release_action_safety)
        info.update(
            {
                "safety_mode": "horizon_deform",
                "mode": "committed_explicit_recovery",
                "deform_mode": "mpc_handoff_release_to_main_filter",
                "deformation_source": "committed_explicit_recovery",
                "recovery_mode": "resume_act",
                "recovery_phase": "mpc_handoff_release_to_main_filter",
                "committed_chunk_active": True,
                "committed_chunk_mode": mode,
                "committed_chunk_index": int(idx),
                "committed_chunk_length": int(total),
                "committed_chunk_completed": True,
                "committed_released_for_act_resume": True,
                "committed_opportunistic_resume": True,
                "committed_recovery_budget_exit": False,
                "resume_from_committed_rejoin": False,
                "request_action_history_reset_after_recovery": True,
                "act_resume_index": nearest_index,
                "act_resume_supported": False,
                "mpc_handoff_resume_chunk_uses_target_window": False,
                "mpc_handoff_release_to_main_filter": True,
                "mpc_handoff_defer_act_safety_to_main_filter": True,
                "mpc_handoff_release_chunk_source": release_source,
                "mpc_handoff_resume_window_len": int(release_len),
                "mpc_handoff_release_chunk_len": int(release_len),
                "recover_steps_executed": 0,
                "return_steps_executed": 0,
                "deform_steps_executed": 0,
                "fallback_used": False,
                "optimized_accepted": True,
                "deform_safe": True,
                "is_safe": True,
                "is_recoverable": True,
                "mpc_handoff_attempted": True,
                "mpc_handoff_accepted": True,
                "mpc_handoff_rejected": False,
                "mpc_handoff_reject_reason": None,
                "mpc_handoff_reason": str(handoff_reason),
                "mpc_handoff_attempt_count": int(self.handoff_attempt_count),
                "mpc_handoff_accept_count": int(self.handoff_accept_count),
                "mpc_handoff_reject_count": int(self.handoff_reject_count),
                **handoff_terms,
                **state_info,
            }
        )
        if recovery.committed_rejoin_diagnostics:
            for key, value in recovery.committed_rejoin_diagnostics.items():
                info.setdefault(key, value)
        recovery._clear_committed_chunk()
        recovery.last_info = info
        return release_chunk.reshape(original_shape), info

    def _target_tube_loss_threshold(self) -> tuple[float, float]:
        """Return (loss_threshold, distance_threshold) for MPC ACT-tube recovery."""
        recovery = self.recovery
        configured = getattr(recovery, "mpc_recovery_target_tube_radius", None)
        if configured is not None and float(configured) > 0.0:
            dist_threshold = float(configured)
            return float(dist_threshold**2), dist_threshold
        pose_threshold = float(getattr(recovery, "recover_ordered_pose_threshold", 0.02))
        loss_threshold = max(pose_threshold * 4.0, pose_threshold + 0.05)
        return float(loss_threshold), float(np.sqrt(max(loss_threshold, 0.0)))

    def recovery_target_tube_terms(
        self,
        obs: Any,
        candidate_chunk: np.ndarray,
        *,
        current_q: np.ndarray,
        target_info: InfoDict | None = None,
    ) -> InfoDict:
        """Score candidate recovery by distance to the nominal ACT trajectory tube.

        This deliberately compares the candidate terminal state to every waypoint in
        the selected live-safe ACT window, instead of requiring one exact rejoin
        index.  The first-step bridge term still handles immediate heading, while
        this term asks whether the suffix ends inside the ACT trajectory tube and
        has not moved backward along that tube.
        """
        recovery = self.recovery
        chunk, _ = recovery._as_chunk(candidate_chunk)
        if chunk.shape[0] == 0:
            return {
                "mpc_recovery_target_tube_available": False,
                "mpc_recovery_target_tube_ok": False,
                "mpc_recovery_target_tube_progress_ok": False,
                "mpc_recovery_target_tube_loss": 0.0,
                "mpc_recovery_target_tube_unavailable_reason": "empty_candidate_chunk",
            }
        current_q_arr = np.asarray(current_q, dtype=np.float32).reshape(-1)
        if target_info is None:
            try:
                target_info = recovery.get_nominal_rejoin_target(
                    obs,
                    candidate_q=current_q_arr,
                    require_live_prefix_safe=True,
                    live_prefix_len=int(getattr(recovery, "recover_act_frame_stack", 1)),
                    allow_best_live_prefix_when_unsafe=True,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "mpc_recovery_target_tube_available": False,
                    "mpc_recovery_target_tube_ok": False,
                    "mpc_recovery_target_tube_progress_ok": False,
                    "mpc_recovery_target_tube_loss": 0.0,
                    "mpc_recovery_target_tube_unavailable_reason": (
                        f"target_selection_failed:{exc}"
                    ),
                }
        if not bool(target_info.get("available", False)):
            return {
                "mpc_recovery_target_tube_available": False,
                "mpc_recovery_target_tube_ok": False,
                "mpc_recovery_target_tube_progress_ok": False,
                "mpc_recovery_target_tube_loss": 0.0,
                "mpc_recovery_target_tube_unavailable_reason": str(
                    target_info.get("suppressed_reason", "no_safe_rejoin_window")
                ),
                "mpc_recovery_target_tube_live_prefix_safe": target_info.get(
                    "nominal_rejoin_live_prefix_safe"
                ),
                "mpc_recovery_target_tube_live_prefix_best_min_clearance": (
                    target_info.get("nominal_rejoin_live_prefix_best_min_clearance")
                ),
            }
        target_q_seq = np.asarray(target_info.get("target_q_seq"), dtype=np.float32)
        if target_q_seq.ndim != 2 or target_q_seq.shape[0] == 0:
            return {
                "mpc_recovery_target_tube_available": False,
                "mpc_recovery_target_tube_ok": False,
                "mpc_recovery_target_tube_progress_ok": False,
                "mpc_recovery_target_tube_loss": 0.0,
                "mpc_recovery_target_tube_unavailable_reason": "empty_target_q_window",
            }
        q_seq = np.asarray(recovery.rollout_nominal_chunk(obs, chunk), dtype=np.float32)
        if q_seq.ndim != 2 or q_seq.shape[0] == 0:
            return {
                "mpc_recovery_target_tube_available": False,
                "mpc_recovery_target_tube_ok": False,
                "mpc_recovery_target_tube_progress_ok": False,
                "mpc_recovery_target_tube_loss": 0.0,
                "mpc_recovery_target_tube_unavailable_reason": "candidate_rollout_unavailable",
            }
        state_idx, state_weights = recovery._mpc_state_indices_and_weights(
            current_q_arr.shape[0],
            target_q_seq.shape[1],
            kind="handoff",
        )
        valid_mask = (
            (state_idx < q_seq.shape[1])
            & (state_idx < target_q_seq.shape[1])
            & (state_idx < current_q_arr.shape[0])
        )
        valid = state_idx[valid_mask]
        if valid.size == 0:
            return {
                "mpc_recovery_target_tube_available": False,
                "mpc_recovery_target_tube_ok": False,
                "mpc_recovery_target_tube_progress_ok": False,
                "mpc_recovery_target_tube_loss": 0.0,
                "mpc_recovery_target_tube_unavailable_reason": "no_controlled_state_indices",
            }
        valid_weights = state_weights[valid_mask]
        weights_2d = valid_weights.reshape(1, -1)
        target_controlled = target_q_seq[:, valid] * weights_2d
        current_controlled = current_q_arr[valid].reshape(1, -1) * weights_2d
        q_controlled = q_seq[:, valid] * weights_2d

        current_loss_seq = np.square(target_controlled - current_controlled).mean(axis=1)
        current_nearest_local = int(np.argmin(current_loss_seq))
        terminal = q_controlled[-1].reshape(1, -1)
        terminal_loss_seq = np.square(target_controlled - terminal).mean(axis=1)
        terminal_nearest_local = int(np.argmin(terminal_loss_seq))
        terminal_loss = float(terminal_loss_seq[terminal_nearest_local])
        terminal_dist = float(np.sqrt(max(terminal_loss, 0.0)))
        terminal_q = np.asarray(q_seq[-1], dtype=np.float32).reshape(-1)
        target_q = np.asarray(target_q_seq[terminal_nearest_local], dtype=np.float32).reshape(-1)
        terminal_dim = min(terminal_q.shape[0], target_q.shape[0])
        terminal_error = terminal_q[:terminal_dim] - target_q[:terminal_dim]
        terminal_valid_delta = terminal_q[valid] - target_q[valid]
        terminal_weighted_delta = terminal_valid_delta * valid_weights
        terminal_error_l2 = float(np.linalg.norm(terminal_error))

        all_loss = np.square(q_controlled[:, None, :] - target_controlled[None, :, :]).mean(axis=2)
        min_path_loss = float(np.min(all_loss))
        min_path_dist = float(np.sqrt(max(min_path_loss, 0.0)))

        requested_window_len = int(
            max(
                1,
                getattr(
                    recovery,
                    "mpc_recovery_target_tube_window_len",
                    getattr(recovery, "recover_act_frame_stack", 4),
                ),
            )
        )
        window_len = int(min(requested_window_len, q_controlled.shape[0], target_controlled.shape[0]))
        window_loss = 0.0
        window_dist = 0.0
        window_error_l2 = 0.0
        window_dq_loss = 0.0
        window_dq_dist = 0.0
        window_action_loss = 0.0
        window_action_dist = 0.0
        window_q_frame_l2 = []
        window_q_frame_l2_mean = 0.0
        window_q_frame_l2_max = 0.0
        window_wrist_l2 = []
        window_wrist_l2_mean = None
        window_wrist_l2_max = None
        window_left_wrist_abs = []
        window_left_wrist_abs_mean = None
        window_left_wrist_abs_max = None
        window_right_wrist_abs = []
        window_right_wrist_abs_mean = None
        window_right_wrist_abs_max = None
        window_recovery_step_l2 = []
        window_target_step_l2 = []
        window_step_l2_error = []
        window_step_l2_error_mean = 0.0
        window_step_l2_error_max = 0.0
        window_dq_error_l2 = 0.0
        window_dq_cosine = []
        window_dq_cosine_mean = None
        window_dq_cosine_min = None
        window_dq_norm_ratio = []
        window_dq_norm_ratio_mean = None
        window_dq_norm_ratio_min = None
        window_best_start = terminal_nearest_local
        window_best_end = terminal_nearest_local
        if window_len > 0:
            recovery_window = q_controlled[-window_len:]
            num_windows = max(1, target_controlled.shape[0] - window_len + 1)
            q_window_losses = []
            q_window_error_l2s = []
            dq_window_losses = []
            action_window_losses = []
            action_idx_for_valid = None
            target_chunk = target_info.get("target_chunk")
            try:
                target_chunk = np.asarray(target_chunk, dtype=np.float32)
            except Exception:  # noqa: BLE001
                target_chunk = None
            try:
                state_to_action = {
                    int(state): int(action)
                    for state, action in zip(
                        np.asarray(recovery.controlled_state_indices, dtype=np.int64).reshape(-1),
                        np.asarray(recovery.controlled_action_indices, dtype=np.int64).reshape(-1),
                    )
                }
                action_idx_for_valid = np.asarray(
                    [state_to_action.get(int(state), -1) for state in valid],
                    dtype=np.int64,
                )
            except Exception:  # noqa: BLE001
                action_idx_for_valid = None
            for start_i in range(num_windows):
                target_window = target_controlled[start_i : start_i + window_len]
                diff = recovery_window - target_window
                q_window_losses.append(float(np.square(diff).mean()))
                q_window_error_l2s.append(float(np.linalg.norm(diff)))
                if window_len >= 2:
                    dq_recovery = np.diff(recovery_window, axis=0)
                    dq_target = np.diff(target_window, axis=0)
                    dq_window_losses.append(float(np.square(dq_recovery - dq_target).mean()))
                else:
                    dq_window_losses.append(0.0)
                action_loss_i = 0.0
                if (
                    action_idx_for_valid is not None
                    and target_chunk is not None
                    and target_chunk.ndim == 2
                    and chunk.ndim == 2
                    and start_i + window_len <= target_chunk.shape[0]
                ):
                    action_valid = (
                        (action_idx_for_valid >= 0)
                        & (action_idx_for_valid < chunk.shape[1])
                        & (action_idx_for_valid < target_chunk.shape[1])
                    )
                    if np.any(action_valid):
                        action_idx_valid = action_idx_for_valid[action_valid]
                        action_weights = valid_weights[action_valid].reshape(1, -1)
                        recovery_action_window = chunk[-window_len:, action_idx_valid] * action_weights
                        target_action_window = target_chunk[start_i : start_i + window_len, action_idx_valid] * action_weights
                        action_loss_i = float(np.square(recovery_action_window - target_action_window).mean())
                action_window_losses.append(action_loss_i)
            q_window_losses_arr = np.asarray(q_window_losses, dtype=np.float32)
            window_best_start = int(np.argmin(q_window_losses_arr))
            window_best_end = int(window_best_start + window_len - 1)
            window_loss = float(q_window_losses_arr[window_best_start])
            window_dist = float(np.sqrt(max(window_loss, 0.0)))
            window_error_l2 = float(q_window_error_l2s[window_best_start])
            window_dq_loss = float(dq_window_losses[window_best_start])
            window_dq_dist = float(np.sqrt(max(window_dq_loss, 0.0)))
            window_action_loss = float(action_window_losses[window_best_start])
            window_action_dist = float(np.sqrt(max(window_action_loss, 0.0)))
            best_target_window = target_controlled[window_best_start : window_best_start + window_len]
            best_q_diff = recovery_window - best_target_window
            q_frame_l2_arr = np.linalg.norm(best_q_diff, axis=1).astype(np.float32)
            window_q_frame_l2 = q_frame_l2_arr.astype(float).tolist()
            window_q_frame_l2_mean = float(np.mean(q_frame_l2_arr)) if q_frame_l2_arr.size else 0.0
            window_q_frame_l2_max = float(np.max(q_frame_l2_arr)) if q_frame_l2_arr.size else 0.0
            try:
                full_dim = min(int(q_seq.shape[1]), int(target_q_seq.shape[1]))
                wrist_indices = [idx for idx in (8, 13) if idx < full_dim]
                if wrist_indices:
                    recovery_window_full = np.asarray(q_seq[-window_len:, :full_dim], dtype=np.float32)
                    target_window_full = np.asarray(
                        target_q_seq[window_best_start : window_best_start + window_len, :full_dim],
                        dtype=np.float32,
                    )
                    wrist_diff = recovery_window_full[:, wrist_indices] - target_window_full[:, wrist_indices]
                    wrist_l2_arr = np.linalg.norm(wrist_diff, axis=1).astype(np.float32)
                    window_wrist_l2 = wrist_l2_arr.astype(float).tolist()
                    window_wrist_l2_mean = float(np.mean(wrist_l2_arr)) if wrist_l2_arr.size else None
                    window_wrist_l2_max = float(np.max(wrist_l2_arr)) if wrist_l2_arr.size else None
                    if 8 in wrist_indices:
                        left_abs_arr = np.abs(wrist_diff[:, wrist_indices.index(8)]).astype(np.float32)
                        window_left_wrist_abs = left_abs_arr.astype(float).tolist()
                        window_left_wrist_abs_mean = float(np.mean(left_abs_arr)) if left_abs_arr.size else None
                        window_left_wrist_abs_max = float(np.max(left_abs_arr)) if left_abs_arr.size else None
                    if 13 in wrist_indices:
                        right_abs_arr = np.abs(wrist_diff[:, wrist_indices.index(13)]).astype(np.float32)
                        window_right_wrist_abs = right_abs_arr.astype(float).tolist()
                        window_right_wrist_abs_mean = float(np.mean(right_abs_arr)) if right_abs_arr.size else None
                        window_right_wrist_abs_max = float(np.max(right_abs_arr)) if right_abs_arr.size else None
            except Exception:
                pass
            if window_len >= 2:
                dq_recovery = np.diff(recovery_window, axis=0)
                dq_target = np.diff(best_target_window, axis=0)
                dq_diff = dq_recovery - dq_target
                recovery_step_l2_arr = np.linalg.norm(dq_recovery, axis=1).astype(np.float32)
                target_step_l2_arr = np.linalg.norm(dq_target, axis=1).astype(np.float32)
                step_l2_error_arr = np.abs(recovery_step_l2_arr - target_step_l2_arr).astype(np.float32)
                window_recovery_step_l2 = recovery_step_l2_arr.astype(float).tolist()
                window_target_step_l2 = target_step_l2_arr.astype(float).tolist()
                window_step_l2_error = step_l2_error_arr.astype(float).tolist()
                window_step_l2_error_mean = float(np.mean(step_l2_error_arr)) if step_l2_error_arr.size else 0.0
                window_step_l2_error_max = float(np.max(step_l2_error_arr)) if step_l2_error_arr.size else 0.0
                window_dq_error_l2 = float(np.linalg.norm(dq_diff))
                denom = recovery_step_l2_arr * target_step_l2_arr
                valid_cos = denom > 1e-8
                if np.any(valid_cos):
                    cos_arr = np.sum(dq_recovery[valid_cos] * dq_target[valid_cos], axis=1) / (denom[valid_cos] + 1e-8)
                    cos_arr = np.clip(cos_arr, -1.0, 1.0).astype(np.float32)
                    window_dq_cosine = cos_arr.astype(float).tolist()
                    window_dq_cosine_mean = float(np.mean(cos_arr))
                    window_dq_cosine_min = float(np.min(cos_arr))
                valid_ratio = target_step_l2_arr > 1e-8
                if np.any(valid_ratio):
                    ratio_arr = (recovery_step_l2_arr[valid_ratio] / (target_step_l2_arr[valid_ratio] + 1e-8)).astype(np.float32)
                    window_dq_norm_ratio = ratio_arr.astype(float).tolist()
                    window_dq_norm_ratio_mean = float(np.mean(ratio_arr))
                    window_dq_norm_ratio_min = float(np.min(ratio_arr))

        loss_threshold, dist_threshold = self._target_tube_loss_threshold()

        def _threshold(name: str, fallback_name: str | None = None, default: float = float("inf")) -> float:
            for attr in (name, fallback_name):
                if not attr:
                    continue
                try:
                    value = float(getattr(recovery, attr))
                except Exception:  # noqa: BLE001
                    continue
                return value
            return float(default)

        window_q_frame_l2_mean_threshold = _threshold(
            "mpc_recovery_target_tube_window_max_q_frame_l2_mean",
            "recover_resume_window_max_q_frame_l2_mean",
        )
        window_q_frame_l2_max_threshold = _threshold(
            "mpc_recovery_target_tube_window_max_q_frame_l2_max",
            "recover_resume_window_max_q_frame_l2_max",
        )
        window_dq_cosine_threshold = _threshold(
            "mpc_recovery_target_tube_window_min_dq_cosine",
            "recover_resume_window_min_dq_cosine",
            default=-float("inf"),
        )
        window_step_l2_error_threshold = _threshold(
            "mpc_recovery_target_tube_window_max_step_l2_error",
            "recover_resume_window_max_step_l2_error",
        )
        window_q_frame_l2_mean_ok = bool(
            (not np.isfinite(window_q_frame_l2_mean_threshold))
            or window_q_frame_l2_mean <= window_q_frame_l2_mean_threshold
        )
        window_q_frame_l2_max_ok = bool(
            (not np.isfinite(window_q_frame_l2_max_threshold))
            or window_q_frame_l2_max <= window_q_frame_l2_max_threshold
        )
        window_dq_cosine_ok = bool(
            (not np.isfinite(window_dq_cosine_threshold))
            or window_dq_cosine_min is None
            or window_dq_cosine_min >= window_dq_cosine_threshold
        )
        window_step_l2_error_ok = bool(
            (not np.isfinite(window_step_l2_error_threshold))
            or window_step_l2_error_max <= window_step_l2_error_threshold
        )
        tube_ok = bool(
            terminal_loss <= loss_threshold
            and window_q_frame_l2_mean_ok
            and window_q_frame_l2_max_ok
            and window_dq_cosine_ok
            and window_step_l2_error_ok
        )

        local_index_progress = int(terminal_nearest_local - current_nearest_local)
        target_tangent = np.zeros((valid.shape[0],), dtype=np.float32)
        if target_q_seq.shape[0] >= 2:
            tangent_start = min(terminal_nearest_local, target_q_seq.shape[0] - 2)
            tangent_end = tangent_start + 1
            target_tangent = (
                target_q_seq[tangent_end, valid] - target_q_seq[tangent_start, valid]
            ) * valid_weights
        terminal_delta = q_controlled[-1] - current_controlled.reshape(-1)
        tangent_norm = float(np.linalg.norm(target_tangent))
        delta_norm = float(np.linalg.norm(terminal_delta))
        if tangent_norm <= 1e-8 or delta_norm <= 1e-8:
            heading_cosine = None
            progress_projection = None
            projection_ok = bool(local_index_progress >= 0)
        else:
            heading_cosine = float(
                np.dot(terminal_delta, target_tangent)
                / (delta_norm * tangent_norm + 1e-8)
            )
            heading_cosine = float(np.clip(heading_cosine, -1.0, 1.0))
            progress_projection = float(
                np.dot(terminal_delta, target_tangent) / (tangent_norm + 1e-8)
            )
            projection_ok = bool(
                progress_projection >= -abs(float(recovery.mpc_recovery_min_progress_delta))
            )
        progress_ok = bool(local_index_progress >= 0 or projection_ok)
        tube_violation = max(terminal_dist - dist_threshold, 0.0)
        progress_loss = 0.0 if progress_ok else float((abs(local_index_progress) + 1) ** 2)
        window_weight = max(0.0, float(getattr(recovery, "mpc_recovery_target_tube_window_weight", 1.0)))
        window_dq_weight = max(0.0, float(getattr(recovery, "mpc_recovery_target_tube_window_dq_weight", 0.5)))
        window_action_weight = max(0.0, float(getattr(recovery, "mpc_recovery_target_tube_window_action_weight", 0.25)))
        window_total_loss = float(
            window_loss
            + window_dq_weight * window_dq_loss
            + window_action_weight * window_action_loss
        )
        tube_loss = float(tube_violation**2 + progress_loss + window_weight * window_total_loss)
        start = target_info.get("nominal_rejoin_window_start")
        target_index = None if start is None else int(start) + terminal_nearest_local
        recovery_window_q = []
        target_window_q = []
        recovery_window_action = []
        target_window_action = []
        if window_len > 0:
            recovery_window_q = np.asarray(q_seq[-window_len:], dtype=np.float32).astype(float).tolist()
            target_window_q = np.asarray(
                target_q_seq[window_best_start : window_best_start + window_len],
                dtype=np.float32,
            ).astype(float).tolist()
            if chunk.ndim == 2 and chunk.shape[0] > 0:
                recovery_window_action = np.asarray(chunk[-window_len:], dtype=np.float32).astype(float).tolist()
            try:
                target_chunk_payload = np.asarray(target_info.get("target_chunk"), dtype=np.float32)
                if target_chunk_payload.ndim == 2 and target_chunk_payload.shape[0] >= window_best_start + window_len:
                    target_window_action = target_chunk_payload[
                        window_best_start : window_best_start + window_len
                    ].astype(float).tolist()
            except Exception:  # noqa: BLE001
                target_window_action = []

        return {
            "mpc_recovery_target_tube_available": True,
            "mpc_recovery_target_tube_ok": bool(tube_ok),
            "mpc_recovery_target_tube_progress_ok": bool(progress_ok),
            "mpc_recovery_target_tube_loss": float(tube_loss),
            "mpc_recovery_target_tube_terminal_loss": float(terminal_loss),
            "mpc_recovery_target_tube_terminal_dist": float(terminal_dist),
            "mpc_recovery_target_tube_terminal_error_l2": float(terminal_error_l2),
            "mpc_recovery_target_tube_window_len": int(window_len),
            "mpc_recovery_target_tube_requested_window_len": int(requested_window_len),
            "mpc_recovery_target_tube_window_loss": float(window_loss),
            "mpc_recovery_target_tube_window_total_loss": float(window_total_loss),
            "mpc_recovery_target_tube_window_dist": float(window_dist),
            "mpc_recovery_target_tube_window_error_l2": float(window_error_l2),
            "mpc_recovery_target_tube_window_dq_loss": float(window_dq_loss),
            "mpc_recovery_target_tube_window_dq_dist": float(window_dq_dist),
            "mpc_recovery_target_tube_window_action_loss": float(window_action_loss),
            "mpc_recovery_target_tube_window_action_dist": float(window_action_dist),
            "mpc_recovery_target_tube_window_q_frame_l2": window_q_frame_l2,
            "mpc_recovery_target_tube_window_q_frame_l2_mean": float(window_q_frame_l2_mean),
            "mpc_recovery_target_tube_window_q_frame_l2_max": float(window_q_frame_l2_max),
            "mpc_recovery_target_tube_window_q_frame_l2_mean_threshold": float(window_q_frame_l2_mean_threshold),
            "mpc_recovery_target_tube_window_q_frame_l2_max_threshold": float(window_q_frame_l2_max_threshold),
            "mpc_recovery_target_tube_window_q_frame_l2_mean_ok": bool(window_q_frame_l2_mean_ok),
            "mpc_recovery_target_tube_window_q_frame_l2_max_ok": bool(window_q_frame_l2_max_ok),
            "mpc_recovery_target_tube_window_dq_cosine_threshold": float(window_dq_cosine_threshold),
            "mpc_recovery_target_tube_window_dq_cosine_ok": bool(window_dq_cosine_ok),
            "mpc_recovery_target_tube_window_step_l2_error_threshold": float(window_step_l2_error_threshold),
            "mpc_recovery_target_tube_window_step_l2_error_ok": bool(window_step_l2_error_ok),
            "mpc_recovery_target_tube_window_wrist_l2": window_wrist_l2,
            "mpc_recovery_target_tube_window_wrist_l2_mean": window_wrist_l2_mean,
            "mpc_recovery_target_tube_window_wrist_l2_max": window_wrist_l2_max,
            "mpc_recovery_target_tube_window_left_wrist_abs": window_left_wrist_abs,
            "mpc_recovery_target_tube_window_left_wrist_abs_mean": window_left_wrist_abs_mean,
            "mpc_recovery_target_tube_window_left_wrist_abs_max": window_left_wrist_abs_max,
            "mpc_recovery_target_tube_window_right_wrist_abs": window_right_wrist_abs,
            "mpc_recovery_target_tube_window_right_wrist_abs_mean": window_right_wrist_abs_mean,
            "mpc_recovery_target_tube_window_right_wrist_abs_max": window_right_wrist_abs_max,
            "mpc_recovery_target_tube_window_recovery_step_l2": window_recovery_step_l2,
            "mpc_recovery_target_tube_window_target_step_l2": window_target_step_l2,
            "mpc_recovery_target_tube_window_step_l2_error": window_step_l2_error,
            "mpc_recovery_target_tube_window_step_l2_error_mean": float(window_step_l2_error_mean),
            "mpc_recovery_target_tube_window_step_l2_error_max": float(window_step_l2_error_max),
            "mpc_recovery_target_tube_window_dq_error_l2": float(window_dq_error_l2),
            "mpc_recovery_target_tube_window_dq_cosine": window_dq_cosine,
            "mpc_recovery_target_tube_window_dq_cosine_mean": window_dq_cosine_mean,
            "mpc_recovery_target_tube_window_dq_cosine_min": window_dq_cosine_min,
            "mpc_recovery_target_tube_window_dq_norm_ratio": window_dq_norm_ratio,
            "mpc_recovery_target_tube_window_dq_norm_ratio_mean": window_dq_norm_ratio_mean,
            "mpc_recovery_target_tube_window_dq_norm_ratio_min": window_dq_norm_ratio_min,
            "mpc_recovery_target_tube_window_start_local_index": int(window_best_start),
            "mpc_recovery_target_tube_window_end_local_index": int(window_best_end),
            "mpc_recovery_target_tube_window_weight": float(window_weight),
            "mpc_recovery_target_tube_window_dq_weight": float(window_dq_weight),
            "mpc_recovery_target_tube_window_action_weight": float(window_action_weight),
            "mpc_recovery_target_tube_terminal_delta": terminal_error.astype(float).tolist(),
            "mpc_recovery_target_tube_q_error": terminal_error.astype(float).tolist(),
            "mpc_recovery_target_tube_terminal_q": terminal_q.astype(float).tolist(),
            "mpc_recovery_target_tube_target_q": target_q.astype(float).tolist(),
            "mpc_recovery_planned_q_seq": np.asarray(q_seq, dtype=np.float32).astype(float).tolist(),
            "mpc_recovery_planned_action_seq": np.asarray(chunk, dtype=np.float32).astype(float).tolist(),
            "mpc_recovery_target_tube_window_q": recovery_window_q,
            "mpc_recovery_target_tube_target_window_q": target_window_q,
            "mpc_recovery_target_tube_window_action": recovery_window_action,
            "mpc_recovery_target_tube_target_window_action": target_window_action,
            "mpc_recovery_target_tube_terminal_valid_delta": terminal_valid_delta.astype(float).tolist(),
            "mpc_recovery_target_tube_terminal_weighted_delta": terminal_weighted_delta.astype(float).tolist(),
            "mpc_recovery_target_tube_min_path_loss": float(min_path_loss),
            "mpc_recovery_target_tube_min_path_dist": float(min_path_dist),
            "mpc_recovery_target_tube_loss_threshold": float(loss_threshold),
            "mpc_recovery_target_tube_dist_threshold": float(dist_threshold),
            "mpc_recovery_target_tube_current_local_index": int(current_nearest_local),
            "mpc_recovery_target_tube_terminal_local_index": int(terminal_nearest_local),
            "mpc_recovery_target_tube_local_index_progress": int(local_index_progress),
            "mpc_recovery_target_tube_target_index": target_index,
            "mpc_recovery_target_tube_heading_cosine": heading_cosine,
            "mpc_recovery_target_tube_progress_projection": progress_projection,
            "mpc_recovery_target_tube_target_tangent_norm": float(tangent_norm),
            "mpc_recovery_target_tube_terminal_delta_norm": float(delta_norm),
            "mpc_recovery_target_tube_state_indices": valid.astype(int).tolist(),
            "mpc_recovery_target_tube_state_weights": valid_weights.astype(float).tolist(),
            "mpc_recovery_target_tube_target_source": target_info.get(
                "nominal_rejoin_target_source"
            ),
            "mpc_recovery_target_tube_window_start": target_info.get(
                "nominal_rejoin_window_start"
            ),
            "mpc_recovery_target_tube_window_end": target_info.get(
                "nominal_rejoin_window_end"
            ),
            "mpc_recovery_target_tube_live_prefix_safe": target_info.get(
                "nominal_rejoin_live_prefix_safe"
            ),
            "mpc_recovery_target_tube_live_prefix_best_min_clearance": (
                target_info.get("nominal_rejoin_live_prefix_best_min_clearance")
            ),
        }

    def bridge_direction_terms(
        self,
        obs: Any,
        candidate_chunk: np.ndarray,
        *,
        current_q: np.ndarray,
        target_info: InfoDict | None = None,
    ) -> InfoDict:
        """Score the next MPC bridge step against the selected ACT tangent.

        Recovery optimization can satisfy final path/rejoin losses while the first
        action still moves opposite the ACT trajectory. MPC executes receding
        horizon first actions, so this term judges the first rolled-out state
        transition from the live state to the candidate's next state.
        """
        recovery = self.recovery
        chunk, _ = recovery._as_chunk(candidate_chunk)
        if chunk.shape[0] == 0:
            return {
                "mpc_bridge_direction_available": False,
                "mpc_bridge_direction_ok": True,
                "mpc_bridge_heading_ok": True,
                "mpc_bridge_progress_ok": True,
                "mpc_bridge_direction_loss": 0.0,
                "mpc_bridge_heading_loss": 0.0,
                "mpc_bridge_progress_loss": 0.0,
                "mpc_bridge_unavailable_reason": "empty_candidate_chunk",
            }
        current_q_arr = np.asarray(current_q, dtype=np.float32).reshape(-1)
        if target_info is None:
            try:
                target_info = recovery.get_nominal_rejoin_target(
                    obs,
                    candidate_q=current_q_arr,
                    require_live_prefix_safe=True,
                    live_prefix_len=int(getattr(recovery, "recover_act_frame_stack", 1)),
                    allow_best_live_prefix_when_unsafe=True,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "mpc_bridge_direction_available": False,
                    "mpc_bridge_direction_ok": True,
                    "mpc_bridge_heading_ok": True,
                    "mpc_bridge_progress_ok": True,
                    "mpc_bridge_direction_loss": 0.0,
                    "mpc_bridge_heading_loss": 0.0,
                    "mpc_bridge_progress_loss": 0.0,
                    "mpc_bridge_unavailable_reason": f"target_selection_failed:{exc}",
                }
        if not bool(target_info.get("available", False)):
            return {
                "mpc_bridge_direction_available": False,
                "mpc_bridge_direction_ok": True,
                "mpc_bridge_heading_ok": True,
                "mpc_bridge_progress_ok": True,
                "mpc_bridge_direction_loss": 0.0,
                "mpc_bridge_heading_loss": 0.0,
                "mpc_bridge_progress_loss": 0.0,
                "mpc_bridge_unavailable_reason": str(
                    target_info.get("suppressed_reason", "no_safe_rejoin_window")
                ),
            }
        target_q_seq = np.asarray(target_info.get("target_q_seq"), dtype=np.float32)
        if target_q_seq.ndim != 2 or target_q_seq.shape[0] < 2:
            return {
                "mpc_bridge_direction_available": False,
                "mpc_bridge_direction_ok": True,
                "mpc_bridge_heading_ok": True,
                "mpc_bridge_progress_ok": True,
                "mpc_bridge_direction_loss": 0.0,
                "mpc_bridge_heading_loss": 0.0,
                "mpc_bridge_progress_loss": 0.0,
                "mpc_bridge_unavailable_reason": "target_tangent_unavailable",
            }
        state_idx, state_weights = recovery._mpc_state_indices_and_weights(
            current_q_arr.shape[0],
            target_q_seq.shape[1],
            kind="handoff",
        )
        if state_idx.size == 0:
            return {
                "mpc_bridge_direction_available": False,
                "mpc_bridge_direction_ok": True,
                "mpc_bridge_heading_ok": True,
                "mpc_bridge_progress_ok": True,
                "mpc_bridge_direction_loss": 0.0,
                "mpc_bridge_heading_loss": 0.0,
                "mpc_bridge_progress_loss": 0.0,
                "mpc_bridge_unavailable_reason": "no_controlled_state_indices",
            }

        state_weights_2d = state_weights.reshape(1, -1)
        target_controlled = target_q_seq[:, state_idx] * state_weights_2d
        current_controlled = current_q_arr[state_idx].reshape(1, -1) * state_weights_2d
        waypoint_loss = np.square(target_controlled - current_controlled).mean(axis=1)
        nearest_local = int(np.argmin(waypoint_loss))
        tangent_start = min(nearest_local, target_q_seq.shape[0] - 2)
        tangent_end = tangent_start + 1
        target_tangent = (
            target_q_seq[tangent_end, state_idx]
            - target_q_seq[tangent_start, state_idx]
        ) * state_weights
        q_seq = np.asarray(
            recovery.rollout_nominal_chunk(obs, chunk[:1]),
            dtype=np.float32,
        )
        if q_seq.ndim != 2 or q_seq.shape[0] == 0:
            return {
                "mpc_bridge_direction_available": False,
                "mpc_bridge_direction_ok": True,
                "mpc_bridge_heading_ok": True,
                "mpc_bridge_progress_ok": True,
                "mpc_bridge_direction_loss": 0.0,
                "mpc_bridge_heading_loss": 0.0,
                "mpc_bridge_progress_loss": 0.0,
                "mpc_bridge_unavailable_reason": "candidate_rollout_unavailable",
            }
        valid_mask = state_idx < q_seq.shape[1]
        valid = state_idx[valid_mask]
        if valid.size == 0:
            return {
                "mpc_bridge_direction_available": False,
                "mpc_bridge_direction_ok": True,
                "mpc_bridge_heading_ok": True,
                "mpc_bridge_progress_ok": True,
                "mpc_bridge_direction_loss": 0.0,
                "mpc_bridge_heading_loss": 0.0,
                "mpc_bridge_progress_loss": 0.0,
                "mpc_bridge_unavailable_reason": "candidate_state_dim_mismatch",
            }
        next_q = q_seq[0, valid]
        current_valid = current_q_arr[valid]
        valid_weights = state_weights[valid_mask]
        tangent_valid = target_tangent[valid_mask]
        bridge_delta = (next_q - current_valid) * valid_weights
        delta_norm = float(np.linalg.norm(bridge_delta))
        tangent_norm = float(np.linalg.norm(tangent_valid))
        heading_threshold = float(
            getattr(recovery, "recover_min_act_heading_cosine", 0.0)
        )
        progress_threshold = max(0.0, float(recovery.mpc_recovery_min_progress_delta))
        if delta_norm <= 1e-8 or tangent_norm <= 1e-8:
            heading_cosine = None
            progress_projection = None
            heading_loss = float(heading_threshold**2)
            progress_loss = float(progress_threshold**2)
            heading_ok = False
            progress_ok = False
        else:
            heading_cosine = float(
                np.dot(bridge_delta, tangent_valid)
                / (delta_norm * tangent_norm + 1e-8)
            )
            heading_cosine = float(np.clip(heading_cosine, -1.0, 1.0))
            progress_projection = float(
                np.dot(bridge_delta, tangent_valid) / (tangent_norm + 1e-8)
            )
            heading_loss = float(max(heading_threshold - heading_cosine, 0.0) ** 2)
            progress_loss = float(max(progress_threshold - progress_projection, 0.0) ** 2)
            heading_ok = bool(heading_cosine >= heading_threshold)
            progress_ok = bool(progress_projection >= progress_threshold)

        bridge_prefix_available = False
        bridge_prefix_min_clearance = None
        bridge_prefix_required_clearance = max(
            float(recovery._acceptance_clearance_threshold()),
            float(
                getattr(
                    recovery,
                    "opportunistic_resume_min_clearance",
                    recovery._acceptance_clearance_threshold(),
                )
            ),
        )
        bridge_prefix_clearance_ok = True
        bridge_prefix_clearance_loss = 0.0
        # Keep this diagnostic-only for now.  Penalizing this term, even softly,
        # made suffix repair trade away recovery stability before ACT restart could
        # use the committed-rejoin/history-reset path.
        bridge_prefix_clearance_weight = 0.0
        target_chunk = np.asarray(target_info.get("target_chunk"), dtype=np.float32)
        if target_chunk.ndim == 2 and target_chunk.shape[0] > 0:
            try:
                prefix_horizon = max(
                    1,
                    min(
                        int(target_chunk.shape[0]),
                        int(getattr(recovery, "recover_act_frame_stack", 1)),
                    ),
                )
                bridge_prefix_chunk = np.concatenate(
                    [chunk[:1], target_chunk[:prefix_horizon]],
                    axis=0,
                )
                bridge_prefix_q = recovery.rollout_nominal_chunk(
                    obs,
                    bridge_prefix_chunk,
                )
                bridge_prefix_eval = recovery.evaluate_horizon_safety(
                    obs,
                    bridge_prefix_q,
                )
                bridge_prefix_h = np.asarray(
                    recovery._clearance_sequence_from_eval(
                        bridge_prefix_eval,
                        np.asarray(bridge_prefix_q).shape[0],
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                if bridge_prefix_h.size == 0:
                    bridge_prefix_h = np.asarray([float("-inf")], dtype=np.float32)
                bridge_prefix_available = True
                bridge_prefix_min_clearance = float(np.min(bridge_prefix_h))
                bridge_prefix_clearance_ok = bool(
                    bridge_prefix_min_clearance >= bridge_prefix_required_clearance
                )
                bridge_prefix_clearance_loss = float(
                    max(
                        bridge_prefix_required_clearance - bridge_prefix_min_clearance,
                        0.0,
                    )
                    ** 2
                )
            except Exception:  # noqa: BLE001
                bridge_prefix_available = False
                bridge_prefix_clearance_ok = True
                bridge_prefix_clearance_loss = 0.0
        direction_loss = float(
            heading_loss
            + progress_loss
            + bridge_prefix_clearance_weight * bridge_prefix_clearance_loss
        )
        return {
            "mpc_bridge_direction_available": True,
            "mpc_bridge_direction_ok": bool(heading_ok and progress_ok),
            "mpc_bridge_heading_ok": bool(heading_ok),
            "mpc_bridge_progress_ok": bool(progress_ok),
            "mpc_bridge_heading_cosine": heading_cosine,
            "mpc_bridge_heading_cosine_threshold": float(heading_threshold),
            "mpc_bridge_progress_projection": progress_projection,
            "mpc_bridge_progress_threshold": float(progress_threshold),
            "mpc_bridge_direction_loss": float(direction_loss),
            "mpc_bridge_heading_loss": float(heading_loss),
            "mpc_bridge_progress_loss": float(progress_loss),
            "mpc_bridge_live_prefix_available": bool(bridge_prefix_available),
            "mpc_bridge_live_prefix_clearance_ok": bool(bridge_prefix_clearance_ok),
            "mpc_bridge_live_prefix_min_clearance": bridge_prefix_min_clearance,
            "mpc_bridge_live_prefix_required_clearance": float(
                bridge_prefix_required_clearance
            ),
            "mpc_bridge_live_prefix_clearance_loss": float(
                bridge_prefix_clearance_loss
            ),
            "mpc_bridge_live_prefix_clearance_weight": float(
                bridge_prefix_clearance_weight
            ),
            "mpc_bridge_target_local_index": int(nearest_local),
            "mpc_bridge_state_indices": state_idx.astype(int).tolist(),
            "mpc_bridge_state_weights": state_weights.astype(float).tolist(),
            "mpc_bridge_delta_norm": float(delta_norm),
            "mpc_bridge_target_tangent_norm": float(tangent_norm),
            "mpc_bridge_target_source": target_info.get(
                "nominal_rejoin_target_source"
            ),
            "mpc_bridge_target_live_prefix_safe": target_info.get(
                "nominal_rejoin_live_prefix_safe"
            ),
            "mpc_bridge_target_live_prefix_fallback_selected": target_info.get(
                "nominal_rejoin_live_prefix_fallback_selected"
            ),
            "mpc_bridge_target_live_prefix_min_clearance": target_info.get(
                "nominal_rejoin_live_prefix_min_clearance"
            ),
            "mpc_bridge_target_live_prefix_best_min_clearance": target_info.get(
                "nominal_rejoin_live_prefix_best_min_clearance"
            ),
            "mpc_bridge_target_live_prefix_required_clearance": target_info.get(
                "nominal_rejoin_live_prefix_required_clearance"
            ),
            "mpc_bridge_target_window_start": target_info.get(
                "nominal_rejoin_window_start"
            ),
        }

    def try_replan_committed_recovery(
        self,
        obs: Any,
        nominal_chunk: np.ndarray,
        original_shape: Any,
        mode: str,
        idx: int,
        total: int,
        state_info: InfoDict,
        *,
        replan_reason: str,
        **kwargs: Any,
    ) -> RecoveryResult | None:
        """Repair a committed recovery suffix from the live robot state."""
        recovery = self.recovery
        self._annotate_replan_defaults(state_info, idx=idx)
        if not recovery.mpc_recovery_enabled:
            return None
        if mode != "recover":
            state_info.update(
                {
                    "mpc_recovery_replan_rejected": True,
                    "mpc_recovery_replan_reject_reason": "not_recover_mode",
                }
            )
            return None
        if (
            replan_reason == "state_mismatch"
            and int(recovery.committed_recover_steps_since_act)
            < int(recovery.mpc_recovery_prefix_len)
        ):
            state_info.update(
                {
                    "mpc_recovery_prefix_replay_step": True,
                    "committed_state_mismatch_ignored_for_mpc_prefix": True,
                    "mpc_recovery_replan_reject_reason": "prefix_replay_window",
                }
            )
            return None
        if (
            recovery.mpc_recovery_max_replans_per_recovery > 0
            and recovery.mpc_recovery_replans_in_current_recovery
            >= recovery.mpc_recovery_max_replans_per_recovery
        ):
            recovery.mpc_recovery_rejected_count += 1
            state_info.update(
                {
                    "mpc_recovery_replan_rejected": True,
                    "mpc_recovery_replan_reject_reason": "mpc_replan_budget_exceeded",
                    "mpc_recovery_rejected_count": int(
                        recovery.mpc_recovery_rejected_count
                    ),
                }
            )
            return None

        recovery.mpc_recovery_replan_count += 1
        recovery.mpc_recovery_replans_in_current_recovery += 1
        state_info.update(
            {
                "mpc_recovery_active": True,
                "mpc_recovery_replan_attempted": True,
                "mpc_recovery_replan_count": int(
                    recovery.mpc_recovery_replan_count
                ),
                "mpc_recovery_replans_in_current_recovery": int(
                    recovery.mpc_recovery_replans_in_current_recovery
                ),
                "mpc_recovery_replan_reason": str(replan_reason),
            }
        )
        old_recover_steps_since_act = int(recovery.committed_recover_steps_since_act)
        if replan_reason == "recovery_budget_exit":
            recovery.committed_recover_steps_since_act = 0
        result = recovery._try_replan_committed_suffix_from_current_state(
            obs,
            nominal_chunk,
            original_shape,
            mode,
            idx,
            total,
            state_info,
            **kwargs,
        )
        if result is None:
            recovery.committed_recover_steps_since_act = old_recover_steps_since_act
            recovery.mpc_recovery_rejected_count += 1
            state_info.update(
                {
                    "mpc_recovery_replan_rejected": True,
                    "mpc_recovery_replan_reject_reason": state_info.get(
                        "committed_suffix_replan_reject_reason",
                        "suffix_replan_failed",
                    ),
                    "mpc_recovery_rejected_count": int(
                        recovery.mpc_recovery_rejected_count
                    ),
                }
            )
            return None

        safe_chunk, info = result
        recovery.mpc_recovery_accepted_count += 1
        if replan_reason == "recovery_budget_exit":
            recovery.mpc_recovery_budget_escape_count += 1
        accepted_terms = {
            key: value
            for key, value in state_info.items()
            if key.startswith("mpc_recovery_")
            or key.startswith("mpc_handoff_")
            or key == "committed_state_mismatch_ignored_for_mpc_prefix"
        }
        info.update(accepted_terms)
        info.update(
            {
                "mpc_recovery_active": True,
                "mpc_recovery_replan_attempted": True,
                "mpc_recovery_replan_accepted": True,
                "mpc_recovery_replan_rejected": False,
                "mpc_recovery_replan_reject_reason": None,
                "mpc_recovery_accepted_count": int(
                    recovery.mpc_recovery_accepted_count
                ),
                "mpc_recovery_rejected_count": int(
                    recovery.mpc_recovery_rejected_count
                ),
                "mpc_recovery_budget_escape": bool(
                    replan_reason == "recovery_budget_exit"
                ),
                "mpc_recovery_budget_escape_count": int(
                    recovery.mpc_recovery_budget_escape_count
                ),
                "committed_recovery_budget_exit": False,
                "committed_replan_due_to_recovery_budget": bool(
                    replan_reason == "recovery_budget_exit"
                ),
                "mpc_recovery_replan_reason": str(replan_reason),
            }
        )
        # Do not hand off immediately after planning a repaired suffix.  The
        # bridge action must be executed first so the next tick can evaluate the
        # actual state transition and heading produced by MPC.
        return safe_chunk, info
