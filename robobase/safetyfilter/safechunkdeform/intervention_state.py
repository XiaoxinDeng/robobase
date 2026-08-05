from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class InterventionMode(str, Enum):
    """Explicit SafeChunk-Deform intervention state."""

    NOMINAL = "nominal"
    SLOWDOWN_GUARD = "slowdown_guard"
    PAUSE_GUARD = "pause_guard"
    DEFORM_COMMIT = "deform_commit"
    SAFE_STOP_LATCHED = "safe_stop_latched"
    RESUME_BLEND = "resume_blend"


class UnsafeReason(str, Enum):
    """Coarse route classification for unsafe nominal ACT chunks."""

    NONE = "none"
    TRANSIENT_PATH_OBSTRUCTION = "transient_path_obstruction"
    POLICY_INDUCED_HUMAN_COLLISION = "policy_induced_human_collision"
    DEFORMATION_INFEASIBLE = "deformation_infeasible"


class InterventionPolicy(str, Enum):
    """Selectable intervention policy.

    Legacy preserves the current SafeChunk-Deform routing. Conflict-aware adds
    stop counterfactual and task-consistency checks before deformation commit.
    """

    LEGACY = "legacy"
    CONFLICT_AWARE = "conflict_aware"


@dataclass(frozen=True)
class StopCounterfactualResult:
    """Safety result for the existing brake/pause trajectory."""

    safe: bool
    min_distance: float
    violation_step: int | None = None
    unsafe_count: int | None = None
    source: str = "horizon_brake"


@dataclass(frozen=True)
class DeformationAdmissibilityResult:
    """Conflict-aware task-consistency wrapper around deformation evaluation."""

    safe: bool
    bypassable: bool
    goal_blocked: bool
    progress_ok: bool
    terminal_deviation_ok: bool
    resumable: bool
    reason: str
    task_progress_value: float
    terminal_deviation: float
    goal_distance: float | None = None
    goal_check_available: bool = False


@dataclass(frozen=True)
class DeformationEvaluation:
    """Candidate-level deformation admissibility result."""

    admissible: bool
    horizon_safe: bool
    has_progress: bool
    executable: bool
    min_distance: float
    progress: float
    min_velocity_ratio: float
    failure_reason: str = ""

    @classmethod
    def rejected(cls, reason: str) -> "DeformationEvaluation":
        return cls(
            admissible=False,
            horizon_safe=False,
            has_progress=False,
            executable=False,
            min_distance=float("-inf"),
            progress=0.0,
            min_velocity_ratio=0.0,
            failure_reason=reason,
        )


@dataclass(frozen=True)
class InterventionFSMConfig:
    """Tunable state machine thresholds."""

    enabled: bool = True
    intervention_policy: str = InterventionPolicy.LEGACY.value
    deform_valid_required_steps: int = 3
    nominal_clear_required_steps: int = 3
    deform_commit_min_steps: int = 5
    deform_stall_required_steps: int = 3
    min_deform_progress: float = 1e-3
    min_step_progress: float = 1e-4
    min_deform_velocity_ratio: float = 0.1
    stop_action_threshold: float = 1e-3
    human_reconsider_distance: float = 0.03
    nominal_change_threshold: float = 0.05
    resume_blend_steps: int = 5
    pause_deform_on_current_human_unsafe: bool = True
    pause_deform_min_clearance_threshold: float = 0.03
    pause_deform_suppress_when_stop_sufficient: bool = True
    pause_deform_static_human_speed_threshold: float = 0.03
    pause_deform_suppress_requires_goal_check: bool = False
    early_deform_suppress_when_stop_sufficient: bool = True
    early_deform_static_human_displacement_threshold: float = 0.02
    stationary_human_local_escape_enabled: bool = True
    stationary_human_local_escape_max_steps: int = 6
    policy_collision_slowdown_enabled: bool = False
    policy_collision_slowdown_max_steps: int = 1
    policy_collision_slowdown_min_first_violation: int = 6
    pause_guard_slowdown_enabled: bool = True
    pause_guard_slowdown_max_steps: int = 2
    stop_counterfactual_enabled: bool = True
    goal_block_check_enabled: bool = True
    deformation_admissibility_enabled: bool = True
    pause_budget_steps: int = 0
    resume_hysteresis_steps: int = 3
    deform_commit_steps: int = 4
    goal_block_radius: float = 0.10
    max_terminal_deviation: float = 0.50
    min_task_progress: float = 0.0


class InterventionStateMachine:
    """Small memory component for SafeChunk-Deform mode selection.

    The filter still owns all rollout, OSCBF, and action execution.  This class
    only keeps counters/latches so pause budget expiry means "start evaluating
    deformation" rather than "must execute deformation now".
    """

    def __init__(self, config: InterventionFSMConfig | None = None) -> None:
        self.config = config or InterventionFSMConfig()
        self.reset()

    def reset(self) -> None:
        self.mode = InterventionMode.NOMINAL
        self.previous_mode = InterventionMode.NOMINAL
        self.transition_reason = "reset"
        self.unsafe_reason = UnsafeReason.NONE
        self.pause_counter = 0
        self.slowdown_counter = 0
        self.deform_valid_counter = 0
        self.nominal_clear_counter = 0
        self.deform_commit_counter = 0
        self.deform_stall_counter = 0
        self.resume_blend_counter = 0
        self.deform_failure_latched = False
        self.deform_failure_human_state: np.ndarray | None = None
        self.deform_failure_robot_state: np.ndarray | None = None
        self.deform_failure_nominal_signature: np.ndarray | None = None
        self.deform_failure_step: int | None = None
        self.last_evaluation: DeformationEvaluation | None = None
        self.previous_safe_action: np.ndarray | None = None

    def transition(
        self,
        new_mode: InterventionMode,
        reason: str,
        unsafe_reason: UnsafeReason | str | None = None,
    ) -> bool:
        if not isinstance(new_mode, InterventionMode):
            new_mode = InterventionMode(str(new_mode))
        changed = new_mode != self.mode
        self.previous_mode = self.mode
        self.mode = new_mode
        self.transition_reason = str(reason)
        if unsafe_reason is not None:
            self.unsafe_reason = self._coerce_unsafe_reason(unsafe_reason)
        if changed:
            if new_mode == InterventionMode.NOMINAL:
                self.pause_counter = 0
                self.slowdown_counter = 0
                self.deform_valid_counter = 0
                self.deform_commit_counter = 0
                self.deform_stall_counter = 0
                self.resume_blend_counter = 0
                self.deform_failure_latched = False
            elif new_mode in {
                InterventionMode.SLOWDOWN_GUARD,
                InterventionMode.PAUSE_GUARD,
            }:
                self.deform_valid_counter = 0
                self.deform_commit_counter = 0
                self.deform_stall_counter = 0
                self.resume_blend_counter = 0
            elif new_mode == InterventionMode.DEFORM_COMMIT:
                self.deform_commit_counter = 0
                self.deform_stall_counter = 0
                self.resume_blend_counter = 0
            elif new_mode == InterventionMode.RESUME_BLEND:
                self.resume_blend_counter = 0
        return changed

    def note_pause_step(self) -> None:
        self.pause_counter += 1
        self.nominal_clear_counter = 0
        if self.mode != InterventionMode.PAUSE_GUARD:
            self.transition(InterventionMode.PAUSE_GUARD, "pause_step")

    def note_slowdown_step(self) -> None:
        self.slowdown_counter += 1
        self.nominal_clear_counter = 0
        if self.mode != InterventionMode.SLOWDOWN_GUARD:
            self.transition(InterventionMode.SLOWDOWN_GUARD, "slowdown_step")

    def can_slowdown_for_policy_collision(self) -> bool:
        return (
            bool(self.config.policy_collision_slowdown_enabled)
            and self.slowdown_counter < max(1, self.config.policy_collision_slowdown_max_steps)
        )

    def can_slowdown_for_pause_guard(self) -> bool:
        return (
            bool(self.config.pause_guard_slowdown_enabled)
            and self.slowdown_counter < max(0, self.config.pause_guard_slowdown_max_steps)
        )

    def note_nominal_clear(self, clear: bool) -> None:
        if clear:
            self.nominal_clear_counter += 1
        else:
            self.nominal_clear_counter = 0

    def note_deformation_evaluation(
        self,
        evaluation: DeformationEvaluation,
    ) -> bool:
        self.last_evaluation = evaluation
        if evaluation.admissible:
            self.deform_valid_counter += 1
        else:
            self.deform_valid_counter = 0
        return self.deform_valid_counter >= max(1, self.config.deform_valid_required_steps)

    def can_enter_deform_commit(
        self,
        pause_budget_expired: bool,
        evaluation: DeformationEvaluation,
    ) -> bool:
        valid = self.note_deformation_evaluation(evaluation)
        return bool(pause_budget_expired and valid and evaluation.admissible)

    def note_deform_commit_step(
        self,
        action_norm: float | None = None,
        step_progress: float | None = None,
    ) -> bool:
        self.deform_commit_counter += 1
        stalled = False
        if action_norm is not None:
            try:
                stalled = stalled or float(action_norm) < self.config.stop_action_threshold
            except Exception:
                pass
        if step_progress is not None:
            try:
                stalled = stalled or float(step_progress) < self.config.min_step_progress
            except Exception:
                pass
        if stalled:
            self.deform_stall_counter += 1
        else:
            self.deform_stall_counter = 0
        return self.deform_stall_counter >= max(1, self.config.deform_stall_required_steps)

    def latch_failure(
        self,
        *,
        step: int | None,
        reason: str,
        human_state: Any = None,
        robot_state: Any = None,
        nominal_signature: Any = None,
    ) -> None:
        self.deform_failure_latched = True
        self.deform_failure_step = None if step is None else int(step)
        self.deform_failure_human_state = self._array_or_none(human_state)
        self.deform_failure_robot_state = self._array_or_none(robot_state)
        self.deform_failure_nominal_signature = self._array_or_none(nominal_signature)
        self.transition(
            InterventionMode.SAFE_STOP_LATCHED,
            reason,
            UnsafeReason.DEFORMATION_INFEASIBLE,
        )

    def latch_release_status(
        self,
        *,
        human_state: Any = None,
        nominal_signature: Any = None,
        nominal_clear: bool = False,
    ) -> tuple[bool, float | None, float | None, str]:
        human_motion = self._distance(self.deform_failure_human_state, human_state)
        nominal_change = self._distance(
            self.deform_failure_nominal_signature,
            nominal_signature,
        )
        required_clear = self.config.nominal_clear_required_steps
        if self.config.intervention_policy == InterventionPolicy.CONFLICT_AWARE.value:
            required_clear = self.config.resume_hysteresis_steps
        if nominal_clear and self.nominal_clear_counter >= max(1, required_clear):
            return True, human_motion, nominal_change, "nominal_clear_hysteresis"
        if human_motion is not None and human_motion >= self.config.human_reconsider_distance:
            return True, human_motion, nominal_change, "human_moved"
        if nominal_change is not None and nominal_change >= self.config.nominal_change_threshold:
            return True, human_motion, nominal_change, "nominal_changed"
        return False, human_motion, nominal_change, "no_meaningful_change"

    def can_handoff_from_deform(self) -> bool:
        required_steps = self.config.deform_commit_min_steps
        if self.config.intervention_policy == InterventionPolicy.CONFLICT_AWARE.value:
            required_steps = self.config.deform_commit_steps
        return self.deform_commit_counter >= max(0, required_steps)

    def diagnostics(self) -> dict[str, Any]:
        eval_result = self.last_evaluation
        info: dict[str, Any] = {
            "intervention_policy": self.config.intervention_policy,
            "intervention_previous_mode": self.previous_mode.value,
            "intervention_mode": self.mode.value,
            "intervention_transition_reason": self.transition_reason,
            "intervention_unsafe_reason": self.unsafe_reason.value,
            "intervention_pause_counter": int(self.pause_counter),
            "intervention_slowdown_counter": int(self.slowdown_counter),
            "intervention_deform_valid_counter": int(self.deform_valid_counter),
            "intervention_nominal_clear_counter": int(self.nominal_clear_counter),
            "intervention_deform_commit_counter": int(self.deform_commit_counter),
            "intervention_deform_stall_counter": int(self.deform_stall_counter),
            "intervention_deform_failure_latched": bool(self.deform_failure_latched),
        }
        if eval_result is not None:
            info.update(
                {
                    "intervention_deformation_admissible": bool(eval_result.admissible),
                    "intervention_deform_horizon_safe": bool(eval_result.horizon_safe),
                    "intervention_deform_has_progress": bool(eval_result.has_progress),
                    "intervention_deform_executable": bool(eval_result.executable),
                    "intervention_deform_min_distance": float(eval_result.min_distance),
                    "intervention_deform_progress": float(eval_result.progress),
                    "intervention_deform_min_velocity_ratio": float(eval_result.min_velocity_ratio),
                    "intervention_deform_failure_reason": eval_result.failure_reason,
                }
            )
        return info

    @staticmethod
    def _coerce_unsafe_reason(value: UnsafeReason | str) -> UnsafeReason:
        if isinstance(value, UnsafeReason):
            return value
        try:
            return UnsafeReason(str(value))
        except ValueError:
            return UnsafeReason.NONE

    @staticmethod
    def _array_or_none(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        return arr.copy()

    @classmethod
    def _distance(cls, previous: Any, current: Any) -> float | None:
        prev = cls._array_or_none(previous)
        cur = cls._array_or_none(current)
        if prev is None or cur is None:
            return None
        n = min(prev.size, cur.size)
        if n <= 0:
            return None
        return float(np.linalg.norm(cur[:n] - prev[:n]))
