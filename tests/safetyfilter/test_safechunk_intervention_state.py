import numpy as np

from robobase.safetyfilter.safechunkdeform.intervention_state import (
    DeformationEvaluation,
    InterventionFSMConfig,
    InterventionMode,
    InterventionPolicy,
    InterventionStateMachine,
    UnsafeReason,
)
from robobase.safetyfilter.safechunkdeform.safechunk_deform_filter import (
    SafeChunkDeformFilter,
)


def eval_result(admissible=True, progress=0.01, velocity_ratio=0.5, reason=""):
    return DeformationEvaluation(
        admissible=admissible,
        horizon_safe=admissible,
        has_progress=progress >= 0.001,
        executable=velocity_ratio >= 0.1,
        min_distance=0.12 if admissible else 0.0,
        progress=progress,
        min_velocity_ratio=velocity_ratio,
        failure_reason=reason,
    )


def test_pause_budget_expired_but_deformation_inadmissible_stays_paused():
    fsm = InterventionStateMachine(InterventionFSMConfig(deform_valid_required_steps=1))
    fsm.transition(
        InterventionMode.PAUSE_GUARD,
        "policy_induced_collision_pause",
        UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
    )
    allowed = fsm.can_enter_deform_commit(
        pause_budget_expired=True,
        evaluation=eval_result(False, progress=0.0, velocity_ratio=0.0, reason="immediate_safety_stop"),
    )
    if not allowed:
        fsm.transition(InterventionMode.PAUSE_GUARD, "deform_not_admissible")
    assert not allowed
    assert fsm.mode == InterventionMode.PAUSE_GUARD


def test_valid_deformation_requires_consecutive_frames_before_commit():
    fsm = InterventionStateMachine(InterventionFSMConfig(deform_valid_required_steps=3))
    fsm.transition(InterventionMode.PAUSE_GUARD, "pause")
    assert not fsm.can_enter_deform_commit(True, eval_result(True))
    assert not fsm.can_enter_deform_commit(True, eval_result(True))
    assert fsm.can_enter_deform_commit(True, eval_result(True))
    fsm.transition(InterventionMode.DEFORM_COMMIT, "admissible")
    assert fsm.mode == InterventionMode.DEFORM_COMMIT


def test_deformation_stall_enters_latched_stop():
    fsm = InterventionStateMachine(InterventionFSMConfig(deform_stall_required_steps=3))
    fsm.transition(InterventionMode.DEFORM_COMMIT, "start")
    assert not fsm.note_deform_commit_step(action_norm=0.0)
    assert not fsm.note_deform_commit_step(action_norm=0.0)
    assert fsm.note_deform_commit_step(action_norm=0.0)
    fsm.latch_failure(step=10, reason="deformation_stalled")
    assert fsm.mode == InterventionMode.SAFE_STOP_LATCHED
    assert fsm.deform_failure_latched


def test_latched_stop_does_not_immediately_redeform_without_change():
    fsm = InterventionStateMachine(InterventionFSMConfig())
    fsm.latch_failure(
        step=4,
        reason="unsafe",
        human_state=np.array([0.0, 0.0, 0.0]),
        nominal_signature=np.array([1.0, 0.0]),
    )
    release, human_motion, nominal_change, reason = fsm.latch_release_status(
        human_state=np.array([0.0, 0.0, 0.0]),
        nominal_signature=np.array([1.0, 0.0]),
        nominal_clear=False,
    )
    assert not release
    assert human_motion == 0.0
    assert nominal_change == 0.0
    assert reason == "no_meaningful_change"
    assert fsm.mode == InterventionMode.SAFE_STOP_LATCHED


def test_human_motion_releases_latch_to_pause_guard_not_deform():
    fsm = InterventionStateMachine(InterventionFSMConfig(human_reconsider_distance=0.03))
    fsm.latch_failure(step=4, reason="unsafe", human_state=np.array([0.0, 0.0]))
    release, *_ = fsm.latch_release_status(human_state=np.array([0.04, 0.0]))
    assert release
    fsm.transition(InterventionMode.PAUSE_GUARD, "safe_stop_latch_released:human_moved")
    assert fsm.mode == InterventionMode.PAUSE_GUARD


def test_resume_requires_consecutive_nominal_clear_frames():
    fsm = InterventionStateMachine(InterventionFSMConfig(nominal_clear_required_steps=3))
    fsm.transition(InterventionMode.PAUSE_GUARD, "pause")
    fsm.note_nominal_clear(True)
    fsm.note_nominal_clear(False)
    fsm.note_nominal_clear(True)
    fsm.note_nominal_clear(True)
    assert fsm.nominal_clear_counter == 2
    assert fsm.nominal_clear_counter < fsm.config.nominal_clear_required_steps
    fsm.note_nominal_clear(True)
    assert fsm.nominal_clear_counter == 3


def test_deform_commit_not_interrupted_before_min_window():
    fsm = InterventionStateMachine(InterventionFSMConfig(deform_commit_min_steps=5))
    fsm.transition(InterventionMode.DEFORM_COMMIT, "start")
    for _ in range(4):
        fsm.note_deform_commit_step(action_norm=1.0)
    assert not fsm.can_handoff_from_deform()
    fsm.note_deform_commit_step(action_norm=1.0)
    assert fsm.can_handoff_from_deform()


def test_transient_obstruction_can_still_go_deform_first():
    fsm = InterventionStateMachine(InterventionFSMConfig(deform_valid_required_steps=1))
    allowed = fsm.can_enter_deform_commit(True, eval_result(True))
    assert allowed
    fsm.transition(
        InterventionMode.DEFORM_COMMIT,
        "transient_obstruction_deform_first",
        UnsafeReason.TRANSIENT_PATH_OBSTRUCTION,
    )
    assert fsm.mode == InterventionMode.DEFORM_COMMIT
    assert fsm.unsafe_reason == UnsafeReason.TRANSIENT_PATH_OBSTRUCTION


def test_policy_collision_slowdown_guard_has_budget():
    fsm = InterventionStateMachine(
        InterventionFSMConfig(policy_collision_slowdown_enabled=True, policy_collision_slowdown_max_steps=2)
    )
    assert fsm.can_slowdown_for_policy_collision()
    fsm.transition(
        InterventionMode.SLOWDOWN_GUARD,
        "policy_induced_collision_slowdown",
        UnsafeReason.POLICY_INDUCED_HUMAN_COLLISION,
    )
    fsm.note_slowdown_step()
    assert fsm.mode == InterventionMode.SLOWDOWN_GUARD
    assert fsm.can_slowdown_for_policy_collision()
    fsm.note_slowdown_step()
    assert not fsm.can_slowdown_for_policy_collision()
    diagnostics = fsm.diagnostics()
    assert diagnostics["intervention_slowdown_counter"] == 2



def filter_shell(policy=InterventionPolicy.CONFLICT_AWARE.value):
    filt = object.__new__(SafeChunkDeformFilter)
    filt.controlled_action_indices = np.array([0, 1], dtype=np.int64)
    filt.intervention_fsm = InterventionStateMachine(
        InterventionFSMConfig(
            intervention_policy=policy,
            goal_block_radius=0.1,
            max_terminal_deviation=0.5,
            min_task_progress=0.0,
            deform_commit_min_steps=0,
            deform_commit_steps=4,
            nominal_clear_required_steps=2,
            resume_hysteresis_steps=3,
        )
    )
    return filt


def test_default_policy_is_legacy_for_backward_compatibility():
    cfg = InterventionFSMConfig()
    assert cfg.intervention_policy == InterventionPolicy.LEGACY.value


def test_legacy_handoff_uses_existing_deform_commit_min_steps():
    fsm = InterventionStateMachine(
        InterventionFSMConfig(
            intervention_policy=InterventionPolicy.LEGACY.value,
            deform_commit_min_steps=1,
            deform_commit_steps=4,
        )
    )
    fsm.transition(InterventionMode.DEFORM_COMMIT, "legacy")
    assert not fsm.can_handoff_from_deform()
    fsm.note_deform_commit_step(action_norm=1.0)
    assert fsm.can_handoff_from_deform()


def test_conflict_aware_handoff_uses_deform_commit_steps():
    fsm = InterventionStateMachine(
        InterventionFSMConfig(
            intervention_policy=InterventionPolicy.CONFLICT_AWARE.value,
            deform_commit_min_steps=0,
            deform_commit_steps=2,
        )
    )
    fsm.transition(InterventionMode.DEFORM_COMMIT, "conflict")
    assert not fsm.can_handoff_from_deform()
    fsm.note_deform_commit_step(action_norm=1.0)
    assert not fsm.can_handoff_from_deform()
    fsm.note_deform_commit_step(action_norm=1.0)
    assert fsm.can_handoff_from_deform()


def test_conflict_aware_stop_counterfactual_reuses_brake_safety():
    filt = filter_shell()
    result = filt._evaluate_stop_counterfactual(
        {"brake_safe": True, "brake_min_clearance": 0.12}
    )
    assert result.safe
    assert result.min_distance == 0.12


def test_conflict_aware_stopping_unsafe_requires_active_evasion():
    filt = filter_shell()
    result = filt._evaluate_stop_counterfactual(
        {"brake_safe": False, "brake_min_clearance": -0.01}
    )
    assert not result.safe
    assert result.min_distance == -0.01


def test_conflict_aware_path_bypass_admissible_when_goal_clear_and_terminal_bounded():
    filt = filter_shell()
    nominal = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    candidate = np.array([[0.0, 0.0], [0.8, 0.1]], dtype=np.float32)
    base = DeformationEvaluation(
        admissible=True,
        horizon_safe=True,
        has_progress=True,
        executable=True,
        min_distance=0.1,
        progress=0.2,
        min_velocity_ratio=0.5,
    )
    result = filt._evaluate_conflict_aware_deformation_admissibility(
        nominal,
        candidate,
        base,
        {},
        {},
    )
    assert result.bypassable
    assert result.reason == "PATH_BYPASS_ADMISSIBLE"


def test_conflict_aware_goal_blockage_rejects_collision_free_local_deform():
    filt = filter_shell()
    nominal = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    candidate = np.array([[0.0, 0.0], [0.8, 0.1]], dtype=np.float32)
    base = DeformationEvaluation(
        admissible=True,
        horizon_safe=True,
        has_progress=True,
        executable=True,
        min_distance=0.1,
        progress=0.2,
        min_velocity_ratio=0.5,
    )
    result = filt._evaluate_conflict_aware_deformation_admissibility(
        nominal,
        candidate,
        base,
        {},
        {"goal_region_blocked": True},
    )
    assert not result.bypassable
    assert result.goal_blocked
    assert result.reason == "GOAL_REGION_BLOCKED"


def test_conflict_aware_resume_uses_resume_hysteresis_steps():
    fsm = InterventionStateMachine(
        InterventionFSMConfig(
            intervention_policy=InterventionPolicy.CONFLICT_AWARE.value,
            nominal_clear_required_steps=1,
            resume_hysteresis_steps=3,
        )
    )
    fsm.latch_failure(step=1, reason="unsafe")
    fsm.note_nominal_clear(True)
    release, *_ = fsm.latch_release_status(nominal_clear=True)
    assert not release
    fsm.note_nominal_clear(True)
    fsm.note_nominal_clear(True)
    release, *_ = fsm.latch_release_status(nominal_clear=True)
    assert release


def test_stop_safe_persistent_path_block_still_requires_bypass():
    assert not SafeChunkDeformFilter._stop_pause_is_task_sufficient(
        stop_safe=True,
        path_blocked=True,
        path_block_pause_sufficient=False,
    )


def test_stop_safe_path_block_can_pause_with_explicit_clear_prediction():
    assert SafeChunkDeformFilter._stop_pause_is_task_sufficient(
        stop_safe=True,
        path_blocked=True,
        path_block_pause_sufficient=True,
    )


def test_stop_safe_non_path_conflict_can_still_pause():
    assert SafeChunkDeformFilter._stop_pause_is_task_sufficient(
        stop_safe=True,
        path_blocked=False,
        path_block_pause_sufficient=False,
    )


def test_path_pause_sufficiency_requires_explicit_clear_prediction():
    filt = filter_shell()
    sufficient, available, source = filt._path_block_pause_sufficiency_info({})
    assert not sufficient
    assert not available
    assert source == "unavailable"
