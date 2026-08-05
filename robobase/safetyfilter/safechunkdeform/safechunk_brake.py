from __future__ import annotations

from typing import Any, Mapping
import logging

import numpy as np

from .safechunk_intervention_factory import InterventionExecutionFactory


logger = logging.getLogger(__name__)


class Brake(InterventionExecutionFactory):
    """Brake and temporary hold execution paths."""

    brake_progress_threshold: float
    deadlock_window: int
    task_progress_brake_threshold: float
    deformation_enabled: bool
    unsafe_deformation_fallback: str
    recover_retry_cooldown_steps: int
    recover_max_attempts_per_unsafe_streak: int
    explicit_return: bool
    commit_accepted_chunks: bool

    temporary_blocker_enabled: bool
    temporary_prefer_brake_before_deform: bool
    temporary_min_unsafe_steps_before_deform: int
    temporary_max_brake_steps_before_deform: int
    temporary_reset_on_nominal_safe: bool
    temporary_require_progress_deadlock_before_deform: bool
    temporary_progress_window: int
    temporary_min_progress_delta: float
    temporary_recover_after_wait: bool
    temporary_recover_after_wait_min_brake_steps: int

    safechunk_active_safety_enabled: bool
    check_hold_horizon_safety: bool
    predict_human_motion_for_hold: bool
    active_safety_hard_min_clearance: float
    hold_prefix_min_clearance: float
    hold_horizon_steps: int
    brake_hold_damping_time: float
    brake_hold_max_correction: float
    emergency_deform_when_hold_unsafe: bool
    optimize_when_hold_unsafe: bool
    emergency_deform_candidate_scales: tuple[float, ...]
    prefer_last_safe_action: bool
    prefer_last_safe_q_retract: bool
    emergency_deform_replan_next_step: bool
    exit_on_nominal_safe: bool

    unsafe_streak: int
    brake_streak: int
    _deadlock_count: int
    recovery_failure_streak: int
    recovery_failure_streak_max: int
    recovery_optimizer_cooldown_remaining: int
    recovery_attempts_in_unsafe_streak: int
    recovery_optimization_skipped_count: int
    last_safe_action: Any | None
    last_safe_q: Any | None
    last_safe_chunk: Any | None
    emergency_deform_away_steps: int
    emergency_deform_away_count: int
    contact_during_hold_count: int
    contact_during_brake_count: int
    contact_during_deform_count: int
    contact_during_recover_count: int
    hold_unsafe_count: int
    hold_predicted_contact_count: int
    _hold_horizon_min_clearance_history: list[float]
    _previous_human_snapshot: Any | None
    _temporary_progress_history: list[float]

    def __init__(
        self,
        parent: Any,
        *,
        brake_progress_threshold: float = 0.05,
        deadlock_window: int = 5,
        task_progress_brake_threshold: float = 0.5,
        deformation_enabled: bool = True,
        unsafe_deformation_fallback: str = "brake",
        recover_retry_cooldown_steps: int = 4,
        recover_max_attempts_per_unsafe_streak: int = 3,
        explicit_return: bool = False,
        commit_accepted_chunks: bool = True,
        temporary_blocker: dict[str, Any] | None = None,
        safechunk_active_safety: dict[str, Any] | None = None,
        intervention: Mapping[str, Any] | None = None,
        intervention_factory: Any | None = None,
        sync: bool | None = None,
    ) -> None:
        """Initialize brake settings and runtime counters for hold/deform fallback."""
        del sync  # Legacy parent call compatibility; Brake no longer copies parent config.
        super().__init__(
            parent,
            intervention=intervention,
            intervention_factory=intervention_factory,
        )
        self._init_config(
            brake_progress_threshold=brake_progress_threshold,
            deadlock_window=deadlock_window,
            task_progress_brake_threshold=task_progress_brake_threshold,
            deformation_enabled=deformation_enabled,
            unsafe_deformation_fallback=unsafe_deformation_fallback,
            recover_retry_cooldown_steps=recover_retry_cooldown_steps,
            recover_max_attempts_per_unsafe_streak=(
                recover_max_attempts_per_unsafe_streak
            ),
            explicit_return=explicit_return,
            commit_accepted_chunks=commit_accepted_chunks,
            temporary_blocker=temporary_blocker,
            safechunk_active_safety=safechunk_active_safety,
        )
        self._init_execution_state()

    @staticmethod
    def _as_config_dict(config: Any | None) -> dict[str, Any]:
        """Normalize optional config-like inputs to a plain dictionary."""
        if config is None:
            return {}
        if hasattr(config, "items"):
            return dict(config.items())
        return dict(config)

    def _temporary_blocker_config(
        self,
        config: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Extract temporary blocker behavior from the brake blocker config block."""
        cfg = self._as_config_dict(config)
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "prefer_brake_before_deform": bool(
                cfg.get("prefer_brake_before_deform", False)
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

    def _safechunk_active_safety_config(
        self,
        config: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Extract hold/deform safety settings from the brake safety config block."""
        cfg = self._as_config_dict(config)
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "check_hold_horizon_safety": bool(
                cfg.get("check_hold_horizon_safety", True)
            ),
            "predict_human_motion_for_hold": bool(
                cfg.get("predict_human_motion_for_hold", True)
            ),
            "hard_min_clearance": float(cfg.get("hard_min_clearance", 0.02)),
            "hold_prefix_min_clearance": float(
                cfg.get("hold_prefix_min_clearance", 0.04)
            ),
            "hold_horizon_steps": int(cfg.get("hold_horizon_steps", 4)),
            "brake_hold_damping_time": float(
                cfg.get("brake_hold_damping_time", 0.0)
            ),
            "brake_hold_max_correction": float(
                cfg.get("brake_hold_max_correction", 0.0)
            ),
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
            "prefer_last_safe_q_retract": bool(
                cfg.get("prefer_last_safe_q_retract", True)
            ),
            "emergency_deform_replan_next_step": bool(
                cfg.get("emergency_deform_replan_next_step", True)
            ),
            "exit_on_nominal_safe": bool(cfg.get("exit_on_nominal_safe", True)),
        }

    def _init_config(
        self,
        *,
        brake_progress_threshold: float,
        deadlock_window: int,
        task_progress_brake_threshold: float,
        deformation_enabled: bool,
        unsafe_deformation_fallback: str,
        recover_retry_cooldown_steps: int,
        recover_max_attempts_per_unsafe_streak: int,
        explicit_return: bool,
        commit_accepted_chunks: bool,
        temporary_blocker: Mapping[str, Any] | None,
        safechunk_active_safety: Mapping[str, Any] | None,
    ) -> None:
        """Initialize explicit brake runtime configuration on the parent filter."""
        unsafe_deformation_fallback = str(unsafe_deformation_fallback)
        if unsafe_deformation_fallback not in {"brake", "best"}:
            raise ValueError(
                "unsafe_deformation_fallback must be one of [brake, best], "
                f"got {unsafe_deformation_fallback}"
            )
        temporary_blocker_cfg: dict[str, Any] = self._temporary_blocker_config(
            temporary_blocker
        )
        active_cfg: dict[str, Any] = self._safechunk_active_safety_config(
            safechunk_active_safety
        )

        self.brake_progress_threshold = float(brake_progress_threshold)
        self.deadlock_window = int(deadlock_window)
        self.task_progress_brake_threshold = float(task_progress_brake_threshold)
        self.deformation_enabled = bool(deformation_enabled)
        self.unsafe_deformation_fallback = unsafe_deformation_fallback
        self.recover_retry_cooldown_steps = max(0, int(recover_retry_cooldown_steps))
        self.recover_max_attempts_per_unsafe_streak = max(
            0,
            int(recover_max_attempts_per_unsafe_streak),
        )
        self.explicit_return = bool(explicit_return)
        self.commit_accepted_chunks = bool(commit_accepted_chunks)

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

        self.safechunk_active_safety_enabled = bool(active_cfg["enabled"])
        self.check_hold_horizon_safety = bool(active_cfg["check_hold_horizon_safety"])
        self.predict_human_motion_for_hold = bool(
            active_cfg["predict_human_motion_for_hold"]
        )
        self.active_safety_hard_min_clearance = float(
            active_cfg["hard_min_clearance"]
        )
        self.hold_prefix_min_clearance = float(active_cfg["hold_prefix_min_clearance"])
        self.hold_horizon_steps = max(1, int(active_cfg["hold_horizon_steps"]))
        self.brake_hold_damping_time = float(active_cfg["brake_hold_damping_time"])
        self.brake_hold_max_correction = float(active_cfg["brake_hold_max_correction"])
        self.emergency_deform_when_hold_unsafe = bool(
            active_cfg["emergency_deform_when_hold_unsafe"]
        )
        self.optimize_when_hold_unsafe = bool(active_cfg["optimize_when_hold_unsafe"])
        self.emergency_deform_candidate_scales = tuple(
            float(x) for x in active_cfg["emergency_deform_candidate_scales"]
        )
        self.prefer_last_safe_action = bool(active_cfg["prefer_last_safe_action"])
        self.prefer_last_safe_q_retract = bool(
            active_cfg["prefer_last_safe_q_retract"]
        )
        self.emergency_deform_replan_next_step = bool(
            active_cfg["emergency_deform_replan_next_step"]
        )
        self.exit_on_nominal_safe = bool(active_cfg["exit_on_nominal_safe"])

    def _init_execution_state(self) -> None:
        """Reset all mutable brake execution counters and rolling histories."""
        self.unsafe_streak: int = 0
        self.brake_streak: int = 0
        self._deadlock_count: int = 0
        self.recovery_failure_streak: int = 0
        self.recovery_failure_streak_max: int = 0
        self.recovery_optimizer_cooldown_remaining: int = 0
        self.recovery_attempts_in_unsafe_streak: int = 0
        self.recovery_optimization_skipped_count: int = 0
        self.last_safe_action: Any | None = None
        self.last_safe_q: Any | None = None
        self.last_safe_chunk: Any | None = None
        self.emergency_deform_away_steps: int = 0
        self.emergency_deform_away_count: int = 0
        self.contact_during_hold_count: int = 0
        self.contact_during_brake_count: int = 0
        self.contact_during_deform_count: int = 0
        self.contact_during_recover_count: int = 0
        self.hold_unsafe_count: int = 0
        self.hold_predicted_contact_count: int = 0
        self._hold_horizon_min_clearance_history: list[float] = []
        self._previous_human_snapshot: Any | None = None
        self._temporary_progress_history: list[float] = []

    def reset_execution_state(self) -> None:
        """Reset counters for a new episode or intervention streak."""
        self._init_execution_state()

    def horizon_slowdown(
        self,
        obs: Any,
        action_chunk: Any,
        safety_info: Mapping[str, Any] | None,
        *,
        factors: tuple[float, ...] = (0.95,),
    ) -> tuple[np.ndarray, dict[str, Any]]:
        chunk, _ = self._as_chunk(action_chunk)
        valid = self._valid_control_indices(chunk)
        action_idx = self.controlled_action_indices[valid]
        if not action_idx.size:
            return chunk.copy(), {
                "slowdown_safe": False,
                "slowdown_applied": False,
                "slowdown_skip_reason": "no_controlled_indices",
            }

        first_violation = None if safety_info is None else safety_info.get("first_violation")
        try:
            state_idx = self.controlled_state_indices[valid]
            anchor = np.asarray(
                self.deform._controlled_anchor(obs, chunk, action_idx, state_idx),
                dtype=np.float32,
            ).reshape(-1)
        except Exception:  # noqa: BLE001
            anchor = np.asarray(chunk[0, action_idx], dtype=np.float32).reshape(-1)

        best_chunk: np.ndarray | None = None
        best_info: dict[str, Any] | None = None
        best_factor = None
        best_clearance = float("-inf")
        checked: list[float] = []
        for raw_factor in factors:
            try:
                factor = float(raw_factor)
            except Exception:  # noqa: BLE001
                continue
            factor = float(np.clip(factor, 0.0, 1.0))
            checked.append(factor)
            candidate = chunk.copy()
            candidate[:, action_idx] = anchor[None, :] + factor * (
                candidate[:, action_idx] - anchor[None, :]
            )
            q_seq = self.rollout_nominal_chunk(obs, candidate)
            slow_safety = self.evaluate_horizon_safety(obs, q_seq)
            clearance = float(slow_safety.get("min_clearance", float("-inf")))
            if bool(slow_safety.get("horizon_safe", False)):
                if best_chunk is None or factor > float(best_factor):
                    best_chunk = candidate
                    best_info = dict(slow_safety)
                    best_factor = factor
                    best_clearance = clearance
            elif best_chunk is None and clearance > best_clearance:
                best_info = dict(slow_safety)
                best_clearance = clearance

        if best_chunk is None:
            return chunk.copy(), {
                "slowdown_safe": False,
                "slowdown_applied": False,
                "slowdown_checked_factors": checked,
                "slowdown_factor": None,
                "slowdown_min_clearance": best_clearance,
                "slowdown_first_violation": first_violation,
                "slowdown_skip_reason": "no_safe_slowdown_factor",
            }

        info = {
            "slowdown_safe": True,
            "slowdown_applied": True,
            "slowdown_checked_factors": checked,
            "slowdown_factor": float(best_factor),
            "slowdown_min_clearance": float(best_info.get("min_clearance", best_clearance)),
            "slowdown_first_violation": first_violation,
            "slowdown_unsafe_count": best_info.get("unsafe_count"),
            "slowdown_horizon_safe": bool(best_info.get("horizon_safe", False)),
            "slowdown_progress_scale": float(best_factor),
        }
        return best_chunk, info

    def horizon_brake(
        self,
        obs: Any,
        action_chunk: Any,
        safety_info: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Clamp controlled actions from the last safe step after a violation."""
        chunk: np.ndarray
        chunk, _ = self._as_chunk(action_chunk)
        first_violation: Any = safety_info.get("first_violation")
        if first_violation is None:
            # No predicted violation: preserve the original chunk and report safety.
            q_seq: np.ndarray = self.rollout_nominal_chunk(obs, chunk)
            brake_safety: Mapping[str, Any] = self.evaluate_horizon_safety(
                obs,
                q_seq,
            )
            return chunk, {
                "brake_safe": bool(brake_safety["horizon_safe"]),
                "deadlock": False,
                "progress_scale": 1.0,
                "brake_stop_idx": None,
                "brake_min_clearance": float(brake_safety["min_clearance"]),
                "safe_prefix_execution": False,
                "safe_prefix_len": int(chunk.shape[0]),
                "brake_only_on_violation": True,
            }

        safe_prefix_len: int = max(0, min(int(first_violation), chunk.shape[0]))
        stop_idx: int = max(0, int(first_violation) - 1)
        stop_idx = min(stop_idx, chunk.shape[0] - 1)
        braked: np.ndarray = chunk.copy()
        valid: np.ndarray = self._valid_control_indices(chunk)
        action_idx: np.ndarray = self.controlled_action_indices[valid]
        if action_idx.size:
            # Controlled dimensions are held; pass-through action channels are untouched.
            if stop_idx == 0:
                state_idx: np.ndarray = self.controlled_state_indices[valid]
                anchor: Any = self.deform._controlled_anchor(
                    obs,
                    chunk,
                    action_idx,
                    state_idx,
                )
                braked[:, action_idx] = anchor
            else:
                braked[stop_idx:, action_idx] = chunk[stop_idx, action_idx]

        q_seq = self.rollout_nominal_chunk(obs, braked)
        brake_safety = self.evaluate_horizon_safety(obs, q_seq)
        progress_scale: float = stop_idx / max(1, chunk.shape[0] - 1)
        deadlock: bool = progress_scale < self.brake_progress_threshold
        return braked, {
            "brake_safe": bool(brake_safety["horizon_safe"]),
            "deadlock": bool(deadlock),
            "progress_scale": float(progress_scale),
            "brake_stop_idx": int(stop_idx),
            "brake_min_clearance": float(brake_safety["min_clearance"]),
            "brake_hold_current": bool(stop_idx == 0),
            "safe_prefix_execution": bool(safe_prefix_len > 0),
            "safe_prefix_len": int(safe_prefix_len),
            "brake_only_on_violation": True,
        }

    def emergency_deform_away(
        self,
        obs: Any,
        reference_action: Any,
        nominal_chunk: Any | None = None,
        hold_info: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Choose an emergency escape candidate when braking/holding is unsafe.

        Candidate chunks are normalized to the reference shape, preserve
        uncontrolled channels, then run through the same hold acceptance gate.
        """
        reference: np.ndarray
        reference, _ = self._as_chunk(reference_action)
        nominal: np.ndarray = (
            reference if nominal_chunk is None else self._as_chunk(nominal_chunk)[0]
        )
        candidates: list[tuple[str, np.ndarray]] = []

        def add(name: str, candidate: Any) -> None:
            """Normalize candidate chunk shape before scoring it."""
            if candidate is None:
                return
            cand: np.ndarray
            cand, _ = self._as_chunk(candidate)
            if cand.shape[1] != reference.shape[1]:
                return
            if cand.shape[0] < reference.shape[0]:
                pad: np.ndarray = np.repeat(
                    cand[-1:],
                    reference.shape[0] - cand.shape[0],
                    axis=0,
                )
                cand = np.concatenate([cand, pad], axis=0)
            elif cand.shape[0] > reference.shape[0]:
                cand = cand[: reference.shape[0]]
            controlled: set[int] = set(self.controlled_action_indices.tolist())
            passthrough_idx: list[int] = [
                i for i in range(reference.shape[1]) if i not in controlled
            ]
            if passthrough_idx:
                cand[:, passthrough_idx] = reference[:, passthrough_idx]
            candidates.append((name, cand.astype(np.float32, copy=True)))

        if not bool((hold_info or {}).get("live_monitor_contact_risk", False)):
            add("hold", reference)
        op: Any = self._get_oscbf_operator()
        if callable(op):
            try:
                action: np.ndarray = np.asarray(
                    self._call_single_step_operator(reference[0], obs, **kwargs),
                    dtype=np.float32,
                )
                add(
                    "oscbf_hold",
                    np.repeat(action.reshape(1, -1), reference.shape[0], axis=0),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Emergency deform-away OSCBF candidate failed: %s", exc)
        if self.prefer_last_safe_action and self.last_safe_action is not None:
            add(
                "last_safe_action",
                np.repeat(
                    np.asarray(self.last_safe_action, dtype=np.float32).reshape(1, -1),
                    reference.shape[0],
                    axis=0,
                ),
            )
        if self.prefer_last_safe_q_retract and self.last_safe_q is not None:
            cand: np.ndarray = reference.copy()
            valid: np.ndarray = (
                (self.controlled_action_indices < cand.shape[1])
                & (self.controlled_state_indices < np.asarray(self.last_safe_q).shape[0])
            )
            last_q: np.ndarray = np.asarray(self.last_safe_q, dtype=np.float32)
            action_idx: np.ndarray = self.controlled_action_indices[valid]
            state_idx: np.ndarray = self.controlled_state_indices[valid]
            modes: np.ndarray = self._control_mode_ids_for_state_indices(state_idx)
            absolute: np.ndarray = modes == 0
            if np.any(absolute):
                cand[:, action_idx[absolute]] = last_q[state_idx[absolute]]
            add("last_safe_q_retract", cand)
        if self.last_safe_chunk is not None:
            for scale in self.emergency_deform_candidate_scales:
                target: np.ndarray = np.asarray(
                    self.last_safe_chunk,
                    dtype=np.float32,
                )
                if target.shape == reference.shape:
                    add(
                        f"scaled_last_safe_{scale:g}",
                        reference + float(scale) * (target - reference),
                    )
        best: tuple[float, str, np.ndarray, Mapping[str, Any], float] | None = None
        best_rejected: (
            tuple[float, str, np.ndarray, Mapping[str, Any], float] | None
        ) = None
        rejected: list[tuple[str, Any]] = []
        live_contact_risk: bool = bool(
            (hold_info or {}).get("live_monitor_contact_risk", False)
        )
        # Score accepted candidates first; keep one rejected escape only for live risk.
        for name, candidate in candidates:
            acc: Mapping[str, Any] = self.evaluate_hold_or_brake_acceptance(
                obs,
                candidate,
                gate_live_monitor_clearance=False,
                **kwargs,
            )
            if candidate.shape[0]:
                valid_control: np.ndarray = self._valid_control_indices(candidate)
                controlled_idx: np.ndarray = self.controlled_action_indices[
                    valid_control
                ]
                action_norm: float = float(
                    np.linalg.norm(candidate[0, controlled_idx])
                )
            else:
                action_norm = 0.0
            clearance_score: float = float(
                acc.get(
                    "hold_horizon_min_clearance",
                    acc.get("hold_immediate_clearance", float("-inf")),
                )
                or float("-inf")
            )
            rejected_score: float = (
                100.0 * clearance_score
                + 10.0 * float(acc.get("safe_prefix_len", 0) or 0)
                - 0.001 * action_norm
            )
            if not acc.get("accepted"):
                rejected.append((name, acc.get("hold_rejected_reason")))
                if name != "hold" and (best_rejected is None or rejected_score > best_rejected[0]):
                    best_rejected = (rejected_score, name, candidate, acc, action_norm)
                continue
            score: float = (
                1000.0
                + 100.0 * float(acc.get("hold_immediate_clearance", 0.0) or 0.0)
                + 10.0 * float(acc.get("safe_prefix_len", 0) or 0)
                - 0.01 * action_norm
            )
            if best is None or score > best[0]:
                best = (score, name, candidate, acc, action_norm)
        used_rejected_escape: bool = False
        if best is None and best_rejected is not None and live_contact_risk:
            _score, name, candidate, acc, action_norm = best_rejected
            used_rejected_escape = True
        elif best is None:
            # Preserve legacy fallback: hold the reference if no escape is available.
            candidate = reference.copy()
            name = "hold_fallback"
            acc = hold_info or {}
            action_norm = float(np.linalg.norm(candidate[0])) if candidate.shape[0] else 0.0
        else:
            _score, name, candidate, acc, action_norm = best
        self.emergency_deform_away_steps += 1
        self.emergency_deform_away_count += 1
        info: dict[str, Any] = {
            "safety_mode": "emergency_deform_away",
            "mode": "emergency_deform_away",
            "deform_mode": "emergency_deform_away",
            "deformation_source": "emergency_deform_away",
            "fallback_used": False,
            "emergency_deform_away": True,
            "emergency_deform_away_steps": int(self.emergency_deform_away_steps),
            "emergency_deform_away_count": int(self.emergency_deform_away_count),
            "emergency_deform_replan_next_step": bool(self.emergency_deform_replan_next_step),
            "accepted_path_name": name,
            "accepted_path_type": "emergency_deform_away",
            "chosen_action_norm": float(action_norm),
            "controlled_action_delta_norm": self.deform._controlled_deformation_norm(candidate, nominal),
            "arm_delta_norm": self.deform._controlled_deformation_norm(candidate, nominal),
            "hold_rejected_candidates": rejected,
            "emergency_deform_used_rejected_escape": bool(used_rejected_escape),
        }
        info.update(acc)
        info.update(self._active_safety_info())
        return candidate, info

    def _active_safety_info(self) -> dict[str, Any]:
        """Aggregate current brake safety and contact counters."""
        h: np.ndarray = np.asarray(
            self._hold_horizon_min_clearance_history,
            dtype=np.float32,
        )
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

    def _hold_prediction_metadata(
        self,
        obs: Any | None = None,
        **kwargs: Any,
    ) -> tuple[bool, float | None]:
        """Compute optional human-motion metadata used by hold acceptance checks."""
        human_state: Any = kwargs.get("human_state")
        current: Any | None = self._snapshot_human_state(human_state)
        velocity_toward_robot: float | None = None
        prediction_available: bool = False
        if current is not None and self._previous_human_snapshot is not None:
            try:
                cur: np.ndarray = np.asarray(current, dtype=np.float32).reshape(-1)
                prev: np.ndarray = np.asarray(
                    self._previous_human_snapshot,
                    dtype=np.float32,
                ).reshape(-1)
                if cur.shape == prev.shape:
                    velocity_toward_robot = float(
                        np.linalg.norm(cur - prev) / max(self.dt, 1e-9)
                    )
                    prediction_available = True
            except Exception:  # noqa: BLE001
                velocity_toward_robot = None
        self._previous_human_snapshot = current
        return prediction_available, velocity_toward_robot

    def evaluate_hold_or_brake_acceptance(
        self,
        obs: Any,
        braked_chunk: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate a hold/brake chunk against model and live-monitor margins."""
        chunk: np.ndarray
        chunk, _ = self._as_chunk(braked_chunk)
        horizon: int = min(chunk.shape[0], max(1, int(self.hold_horizon_steps)))
        hold_chunk: np.ndarray = chunk[:horizon].copy()
        human_prediction_available: bool
        human_velocity_toward_robot: float | None
        human_prediction_available, human_velocity_toward_robot = (
            self._hold_prediction_metadata(obs, **kwargs)
        )
        try:
            acceptance: Mapping[str, Any] = self.evaluate_candidate_acceptance(
                obs,
                hold_chunk,
                candidate_type="hold_or_brake",
            )
        except TypeError:
            acceptance = self.evaluate_candidate_acceptance(obs, hold_chunk, "hold_or_brake")
        model_immediate: float = float(
            acceptance.get("immediate_clearance", float("-inf"))
        )
        model_horizon_min: float = float(
            acceptance.get("horizon_min_clearance", model_immediate)
        )
        immediate: float = model_immediate
        horizon_min: float = model_horizon_min
        safe_prefix_len: int = int(acceptance.get("safe_prefix_len", 0) or 0)
        hard: float = float(self.active_safety_hard_min_clearance)
        live_monitor_min_h: Any = kwargs.get("live_monitor_min_h", kwargs.get("min_h"))
        live_monitor_contact_risk: bool = False
        if kwargs.get("gate_live_monitor_clearance", False) and live_monitor_min_h is not None:
            try:
                live_monitor_min_h = float(live_monitor_min_h)
                if np.isfinite(live_monitor_min_h):
                    immediate = min(immediate, live_monitor_min_h)
                    horizon_min = min(horizon_min, live_monitor_min_h)
                    live_monitor_contact_risk = bool(live_monitor_min_h < hard)
            except Exception:  # noqa: BLE001
                live_monitor_min_h = None
        predicted_contact: bool = bool(horizon_min < hard)
        accepted: bool = bool(
            immediate >= hard
            and not predicted_contact
            and safe_prefix_len >= 1
            and immediate >= float(self.hold_prefix_min_clearance)
        )
        reason: str | None = None
        # The live monitor can only tighten the model margins, never relax them.
        if live_monitor_contact_risk:
            reason = "live_monitor_below_hard_margin"
        elif immediate < hard:
            reason = "immediate_below_hard_margin"
        elif predicted_contact:
            reason = "hold_predicted_contact"
        elif safe_prefix_len < 1 or immediate < float(self.hold_prefix_min_clearance):
            reason = "hold_prefix_unsafe"
        info: dict[str, Any] = dict(acceptance)
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

    def _temporary_update_progress(self, progress: float | None) -> None:
        """Track a short rolling window of temporary-progress measurements."""
        if progress is None:
            return
        try:
            value: float = float(progress)
        except Exception:  # noqa: BLE001
            return
        if not np.isfinite(value):
            return
        self._temporary_progress_history.append(value)
        keep: int = max(1, self.temporary_progress_window)
        if len(self._temporary_progress_history) > keep:
            self._temporary_progress_history = self._temporary_progress_history[-keep:]

    def _temporary_deform_trigger_reason(
        self,
        *,
        progress_deadlock: bool = False,
        progress_available: bool = False,
    ) -> str | None:
        """Return the temporary-deform trigger reason, or None if not active."""
        if self.unsafe_streak >= self.temporary_min_unsafe_steps_before_deform:
            return "persistent_unsafe"
        if self.brake_streak >= self.temporary_max_brake_steps_before_deform:
            return "brake_timeout"
        if progress_available and progress_deadlock:
            return "progress_deadlock"
        return None
