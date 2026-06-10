import logging

import numpy as np

from robobase.safetyfilter.safechunk_deform_filter import RecoveryContext, SafeChunkDeformFilter
from robobase.safetyfilter.path_consistent_brake_filter import PathConsistentBrakeFilter


CTRL = np.asarray([4, 5, 6, 7, 9, 10, 11, 12])
BASE_ARM_CTRL = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12])
PASS = np.asarray([i for i in range(16) if i not in set(CTRL.tolist())])
BASE_ARM_PASS = np.asarray([i for i in range(16) if i not in set(BASE_ARM_CTRL.tolist())])


class FakeOSCBF:
    def __call__(self, action, obs=None, **kwargs):
        out = np.array(action, copy=True)
        out[CTRL] *= 0.5
        return out


class FakeChunkOSCBF:
    def __init__(self):
        self.chunk_calls = 0
        self.single_calls = 0

    def filter_chunk(self, action_chunk, obs=None, **kwargs):
        self.chunk_calls += 1
        out = np.array(action_chunk, copy=True)
        out[:, CTRL] *= 0.25
        return out, {"operator_chunk_info": True}

    def __call__(self, action, obs=None, **kwargs):
        self.single_calls += 1
        out = np.array(action, copy=True)
        out[CTRL] *= 0.5
        return out


class CountingEEOperator:
    def __init__(self):
        self.sequence_calls = 0
        self.pose_calls = 0

    def ee_pose(self, q):
        self.pose_calls += 1
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        return q[:3].copy()

    def ee_pose_sequence(self, q_seq):
        self.sequence_calls += 1
        q_seq = np.asarray(q_seq, dtype=np.float32)
        return q_seq[:, :3].copy()


def make_chunk(h=16):
    return np.arange(h * 16, dtype=np.float32).reshape(h, 16) / 100.0


def unsafe_filter(first_violation, safe=False, **kwargs):
    filt = SafeChunkDeformFilter(oscbf_operator=FakeOSCBF(), debug=False, **kwargs)

    def evaluate(_obs, q_seq):
        clearances = np.ones(q_seq.shape[0], dtype=np.float32)
        if first_violation is not None:
            clearances[first_violation:] = 0.0
        return {
            "horizon_safe": safe,
            "min_clearance": float(clearances.min()),
            "min_clearances": clearances,
            "first_violation": first_violation,
            "unsafe_count": int(np.count_nonzero(clearances < filt.min_clearance)),
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = evaluate
    return filt




def test_horizon_operator_human_capsule_rollout_adds_velocity_prediction():
    from eval_act_oscbf_safety_metrics import HorizonOSCBFOperator

    op = object.__new__(HorizonOSCBFOperator)
    op.predict_human_motion = True
    op.dt = 0.05
    op.human_prediction_max_time = 0.10
    op.human_prediction_max_speed = 3.0
    op._human_motion_prediction_available = True
    op._human_motion_prediction_speed = 1.0
    op._capsule_a_velocity_world = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    op._capsule_b_velocity_world = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    capsule_a = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    capsule_b = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    capsule_radii = np.asarray([0.1], dtype=np.float32)

    a_seq, b_seq, radii, info = op._human_capsule_rollout(
        capsule_a,
        capsule_b,
        capsule_radii,
        horizon=3,
    )

    assert info["human_motion_prediction_available"] is True
    assert np.isclose(info["human_motion_prediction_max_displacement"], 0.1)
    assert a_seq.shape == (3, 2, 3)
    assert b_seq.shape == (3, 2, 3)
    np.testing.assert_allclose(radii, [0.1, 0.1])
    np.testing.assert_allclose(a_seq[:, 0, :], np.zeros((3, 3)))
    np.testing.assert_allclose(a_seq[:, 1, 0], [0.05, 0.10, 0.10])
    np.testing.assert_allclose(b_seq[:, 1, 0], [0.05, 0.10, 0.10])


def test_horizon_operator_human_capsule_rollout_is_static_without_velocity():
    from eval_act_oscbf_safety_metrics import HorizonOSCBFOperator

    op = object.__new__(HorizonOSCBFOperator)
    op.predict_human_motion = True
    op.dt = 0.05
    op.human_prediction_max_time = 0.10
    op.human_prediction_max_speed = 3.0
    op._human_motion_prediction_available = False
    op._human_motion_prediction_speed = 0.0
    op._capsule_a_velocity_world = None
    op._capsule_b_velocity_world = None

    capsule_a = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    capsule_b = np.asarray([[1.0, 2.0, 4.0]], dtype=np.float32)
    capsule_radii = np.asarray([0.1], dtype=np.float32)

    a_seq, b_seq, radii, info = op._human_capsule_rollout(
        capsule_a,
        capsule_b,
        capsule_radii,
        horizon=2,
    )

    assert info["human_motion_prediction_available"] is False
    assert a_seq.shape == (2, 1, 3)
    assert b_seq.shape == (2, 1, 3)
    np.testing.assert_allclose(radii, [0.1])
    np.testing.assert_allclose(a_seq[:, 0, :], np.repeat(capsule_a, 2, axis=0))
    np.testing.assert_allclose(b_seq[:, 0, :], np.repeat(capsule_b, 2, axis=0))


def test_normalize_safety_result_preserves_prediction_metadata():
    filt = SafeChunkDeformFilter(oscbf_operator=FakeOSCBF(), debug=False)
    result = {
        "horizon_safe": True,
        "min_clearance": 0.2,
        "min_clearances": np.asarray([0.2, 0.3], dtype=np.float32),
        "human_motion_prediction_available": True,
        "human_motion_prediction_speed": 0.75,
    }

    info = filt._normalize_safety_result(result, horizon=2)

    assert info["human_motion_prediction_available"] is True
    assert info["human_motion_prediction_speed"] == 0.75


def test_safe_chunk_returns_same_shape_and_values():
    filt = SafeChunkDeformFilter(debug=False)
    chunk = make_chunk()

    safe, info = filt.filter_chunk({"q": np.zeros(14)}, chunk)

    assert safe.shape == chunk.shape
    np.testing.assert_allclose(safe, chunk)
    assert info["safety_mode"] == "pass_through"
    assert filt.last_info is info


def test_single_action_returns_same_shape():
    filt = SafeChunkDeformFilter(oscbf_operator=FakeOSCBF(), debug=False)
    action = make_chunk(1)[0]

    safe = filt({"q": np.zeros(14)}, action)

    assert safe.shape == action.shape
    assert filt.last_info["safety_mode"] == "single_step_oscbf"


def test_jax_batched_optimizer_is_default_and_matches_serial_rollout():
    filt = SafeChunkDeformFilter(mode="optimized", debug=False)
    assert filt.jax_batched_optimizer is True

    chunk = make_chunk(4)
    chunks = np.stack([chunk, chunk + 0.05], axis=0).astype(np.float32)
    obs = {"q": np.linspace(0.0, 0.13, 14, dtype=np.float32)}

    batch_rollout = filt.rollout_nominal_chunk_batch(obs, chunks)
    serial_rollout = np.stack(
        [filt.rollout_nominal_chunk(obs, candidate) for candidate in chunks],
        axis=0,
    )

    np.testing.assert_allclose(batch_rollout, serial_rollout, rtol=1e-6, atol=1e-6)


def test_base_indices_roll_out_as_delta_when_arm_is_absolute():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        debug=False,
        controlled_action_indices=BASE_ARM_CTRL,
        controlled_state_indices=BASE_ARM_CTRL,
        control_type="absolute",
    )
    q0 = np.zeros(14, dtype=np.float32)
    q0[:5] = [1.0, 2.0, 3.0, 0.5, 0.25]
    chunk = np.zeros((3, 16), dtype=np.float32)
    chunk[:, 0] = [0.1, 0.2, -0.05]
    chunk[:, 1] = [0.0, 0.3, 0.0]
    chunk[:, 2] = [0.01, 0.01, 0.01]
    chunk[:, 3] = [0.2, 0.0, -0.1]
    chunk[:, 4] = [0.9, 0.8, 0.7]
    chunk[:, 5] = [-0.4, -0.3, -0.2]

    q_seq = filt.rollout_nominal_chunk({"q": q0}, chunk)

    np.testing.assert_allclose(q_seq[:, 0], [1.1, 1.3, 1.25])
    np.testing.assert_allclose(q_seq[:, 1], [2.0, 2.3, 2.3])
    np.testing.assert_allclose(q_seq[:, 2], [3.01, 3.02, 3.03])
    np.testing.assert_allclose(q_seq[:, 3], [0.7, 0.7, 0.6])
    np.testing.assert_allclose(q_seq[:, 4], [0.9, 0.8, 0.7])
    np.testing.assert_allclose(q_seq[:, 5], [-0.4, -0.3, -0.2])


def test_base_delta_absolute_arm_rollout_batch_matches_serial():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        debug=False,
        controlled_action_indices=BASE_ARM_CTRL,
        controlled_state_indices=BASE_ARM_CTRL,
        control_type="absolute",
    )
    q0 = np.linspace(-0.2, 0.45, 14, dtype=np.float32)
    chunk = np.zeros((4, 16), dtype=np.float32)
    chunk[:, 0] = [0.05, -0.02, 0.03, 0.01]
    chunk[:, 1] = [0.0, 0.04, 0.0, -0.02]
    chunk[:, 2] = [0.01, 0.01, -0.02, 0.0]
    chunk[:, 3] = [-0.1, 0.1, 0.0, 0.05]
    chunk[:, CTRL] = np.linspace(0.1, 0.4, chunk.shape[0], dtype=np.float32)[:, None]
    chunks = np.stack([chunk, chunk + 0.03], axis=0).astype(np.float32)

    batch_rollout = filt.rollout_nominal_chunk_batch({"q": q0}, chunks)
    serial_rollout = np.stack(
        [filt.rollout_nominal_chunk({"q": q0}, candidate) for candidate in chunks],
        axis=0,
    )

    np.testing.assert_allclose(batch_rollout, serial_rollout, rtol=1e-6, atol=1e-6)


def test_return_seed_tracks_future_nominal_q_with_base_delta_actions():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        debug=False,
        controlled_action_indices=BASE_ARM_CTRL,
        controlled_state_indices=BASE_ARM_CTRL,
        control_type="absolute",
        min_rejoin_offset=2,
    )
    filt.return_horizon = 3
    nominal_chunk = np.zeros((8, 16), dtype=np.float32)
    nominal_chunk[:, :4] = 0.01
    nominal_q_seq = np.zeros((8, 14), dtype=np.float32)
    nominal_q_seq[2, :4] = [0.10, -0.05, 0.02, 0.04]
    nominal_q_seq[3, :4] = [0.18, -0.03, 0.03, 0.07]
    nominal_q_seq[4, :4] = [0.25, -0.01, 0.05, 0.10]
    nominal_q_seq[2, CTRL] = 0.20
    nominal_q_seq[3, CTRL] = 0.30
    nominal_q_seq[4, CTRL] = 0.40
    nominal_q_seq[5:, BASE_ARM_CTRL] = 2.0
    nominal_chunk[2:5, CTRL] = nominal_q_seq[2:5, CTRL]
    context = RecoveryContext(
        nominal_chunk=nominal_chunk,
        nominal_q_seq=nominal_q_seq,
    )
    q_start = np.zeros(14, dtype=np.float32)
    current_chunk = np.zeros_like(nominal_chunk)

    return_chunk, target_index = filt._make_return_seed_chunk(
        context,
        q_start,
        current_chunk,
        BASE_ARM_CTRL,
    )
    q_seq = filt.rollout_nominal_chunk({"q": q_start}, return_chunk)

    assert target_index == 2
    np.testing.assert_allclose(return_chunk[0, :4], nominal_q_seq[2, :4], atol=1e-6)
    np.testing.assert_allclose(
        return_chunk[1, :4],
        nominal_q_seq[3, :4] - nominal_q_seq[2, :4],
        atol=1e-6,
    )
    np.testing.assert_allclose(q_seq[:3, :4], nominal_q_seq[2:5, :4], atol=1e-6)
    np.testing.assert_allclose(q_seq[:3, CTRL], nominal_q_seq[2:5, CTRL], atol=1e-6)


def test_recovery_terminal_ordered_path_ends_at_terminal_rejoin_index():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        debug=False,
        controlled_action_indices=BASE_ARM_CTRL,
        controlled_state_indices=BASE_ARM_CTRL,
        control_type="absolute",
        min_rejoin_offset=2,
        safechunk_recover={
            "ordered_pose_weight": 10.0,
            "ordered_delta_weight": 5.0,
            "require_ordered_path": True,
        },
    )
    filt.return_horizon = 3
    filt.evaluate_horizon_safety = lambda _obs, q_seq: {
        "horizon_safe": True,
        "min_clearance": 1.0,
        "min_clearances": np.ones(q_seq.shape[0], dtype=np.float32),
    }
    nominal_chunk = np.zeros((8, 16), dtype=np.float32)
    nominal_q_seq = np.zeros((8, 14), dtype=np.float32)
    nominal_q_seq[2, :4] = [0.10, -0.05, 0.02, 0.04]
    nominal_q_seq[3, :4] = [0.18, -0.03, 0.03, 0.07]
    nominal_q_seq[4, :4] = [0.25, -0.01, 0.05, 0.10]
    nominal_q_seq[2, CTRL] = 0.20
    nominal_q_seq[3, CTRL] = 0.30
    nominal_q_seq[4, CTRL] = 0.40
    nominal_q_seq[5:, BASE_ARM_CTRL] = 2.0
    context = RecoveryContext(
        nominal_chunk=nominal_chunk,
        nominal_q_seq=nominal_q_seq,
    )
    q_start = np.zeros(14, dtype=np.float32)
    return_chunk, seed_index = filt._make_return_seed_chunk(
        context,
        q_start,
        np.zeros_like(nominal_chunk),
        BASE_ARM_CTRL,
    )
    terminal = filt._recovery_terminal_rejoin_info(
        {"q": q_start},
        return_chunk,
        context,
        filt._make_rejoin_context(nominal_q_seq),
        default_target_index=seed_index,
    )

    assert terminal["target_index"] == 4
    assert terminal["recover_ordered_target_index"] == 2
    assert terminal["recover_ordered_ok"] is True
    np.testing.assert_allclose(terminal["recover_ordered_pose_loss"], 0.0, atol=1e-6)
    np.testing.assert_allclose(terminal["recover_ordered_delta_loss"], 0.0, atol=1e-6)


def test_task_progress_recover_seed_tracks_context_nominal_with_base_delta_actions():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        debug=False,
        controlled_action_indices=BASE_ARM_CTRL,
        controlled_state_indices=BASE_ARM_CTRL,
        control_type="absolute",
        min_rejoin_offset=2,
    )
    filt.return_horizon = 3
    nominal_chunk = np.zeros((8, 16), dtype=np.float32)
    nominal_q_seq = np.zeros((8, 14), dtype=np.float32)
    nominal_q_seq[2, :4] = [0.10, -0.05, 0.02, 0.04]
    nominal_q_seq[3, :4] = [0.18, -0.03, 0.03, 0.07]
    nominal_q_seq[4, :4] = [0.25, -0.01, 0.05, 0.10]
    nominal_q_seq[2, CTRL] = 0.20
    nominal_q_seq[3, CTRL] = 0.30
    nominal_q_seq[4, CTRL] = 0.40
    nominal_q_seq[5:, BASE_ARM_CTRL] = 2.0
    context = RecoveryContext(
        nominal_chunk=nominal_chunk,
        nominal_q_seq=nominal_q_seq,
    )
    q_start = np.zeros(14, dtype=np.float32)
    current_chunk = np.zeros_like(nominal_chunk)

    recover_chunk, target_index = filt._make_task_progress_recover_chunk(
        q_start,
        current_chunk,
        BASE_ARM_CTRL,
        context=context,
        default_target_index=2,
    )
    q_seq = filt.rollout_nominal_chunk({"q": q_start}, recover_chunk)

    assert target_index == 2
    np.testing.assert_allclose(recover_chunk[0, :4], nominal_q_seq[2, :4], atol=1e-6)
    np.testing.assert_allclose(
        recover_chunk[1, :4],
        nominal_q_seq[3, :4] - nominal_q_seq[2, :4],
        atol=1e-6,
    )
    np.testing.assert_allclose(q_seq[:3, :4], nominal_q_seq[2:5, :4], atol=1e-6)
    np.testing.assert_allclose(q_seq[:3, CTRL], nominal_q_seq[2:5, CTRL], atol=1e-6)


def test_immediate_base_arm_absolute_brake_zeroes_base_delta_and_holds_arm_q():
    filt = unsafe_filter(
        first_violation=0,
        safe=False,
        controlled_action_indices=BASE_ARM_CTRL,
        controlled_state_indices=BASE_ARM_CTRL,
        control_type="absolute",
    )
    chunk = make_chunk()
    q = np.linspace(-0.7, 0.7, 14, dtype=np.float32)

    braked, info = filt.horizon_brake({"q": q}, chunk, {"first_violation": 0})

    assert info["brake_stop_idx"] == 0
    assert info["brake_hold_current"] is True
    np.testing.assert_allclose(braked[:, :4], 0.0)
    np.testing.assert_allclose(braked[:, CTRL], np.repeat(q[None, CTRL], chunk.shape[0], axis=0))
    np.testing.assert_allclose(braked[:, BASE_ARM_PASS], chunk[:, BASE_ARM_PASS])


def test_chunk_trajectory_trace_logs_only_receding_horizon_first_action():
    from eval_act_oscbf_safety_metrics import _collect_chunk_trajectory_trace

    class Args:
        condition = "path_consistent_brake"
        chunk_trajectory_include_q_states = True

    class FakeHorizonOperator:
        def ee_pose_sequence(self, q_seq):
            q_seq = np.asarray(q_seq, dtype=np.float32)
            return q_seq[:, :3]

    filt = SafeChunkDeformFilter(debug=False)
    nominal = make_chunk(h=4)
    generated = nominal.copy()
    generated[:, CTRL] = 0.0

    record = _collect_chunk_trajectory_trace(
        args=Args(),
        episode=0,
        step=3,
        safechunk=filt,
        horizon_operator=FakeHorizonOperator(),
        obs={"q": np.zeros(14, dtype=np.float32)},
        nominal_chunk=nominal,
        generated_chunk=generated,
        safety_info={
            "safety_mode": "path_consistent_brake",
            "deformation_source": "path_consistent_brake",
        },
    )

    assert record["planned_action_horizon"] == 4
    assert record["executed_action_horizon"] == 1
    assert record["segment_lengths"]["planned_total"] == 4
    assert record["segment_lengths"]["total"] == 1
    for trace_name in ("nominal", "braking", "generated"):
        trace = record["traces"][trace_name]
        assert trace["horizon"] == 1
        assert trace["action_shape"] == [1, 16]
        assert len(trace["action_chunk"]) == 1
        assert len(trace["q_seq"]) == 1

    executed = record["executed_policy_sample"]
    assert executed["source"] == "transformed_safe_action_sequence"
    assert executed["step"] == 4
    np.testing.assert_allclose(executed["ee_pos"], record["traces"]["generated"]["ee_xyz"][0])
    np.testing.assert_allclose(executed["q"], record["traces"]["generated"]["q_seq"][0])
    np.testing.assert_allclose(executed["action"], record["traces"]["generated"]["action_chunk"][0])


def test_actual_execution_segments_are_colored_by_adjacent_modes():
    from eval_act_oscbf_safety_metrics import (
        _execution_marker_segments,
        _execution_pair_segments,
    )

    samples = [
        {"episode": 0, "step": 0, "ee_pos": [0.0, 0.0, 0.0], "execution_mode": "policy"},
        {"episode": 0, "step": 1, "ee_pos": [1.0, 0.0, 0.0], "execution_mode": "policy"},
        {"episode": 0, "step": 2, "ee_pos": [2.0, 0.0, 0.0], "execution_mode": "braking"},
        {"episode": 0, "step": 3, "ee_pos": [3.0, 0.0, 0.0], "execution_mode": "braking"},
        {"episode": 0, "step": 4, "ee_pos": [4.0, 0.0, 0.0], "execution_mode": "deform"},
    ]

    policy_segments = _execution_pair_segments(samples, "policy")
    intervention_segments = _execution_pair_segments(samples, "intervention")
    transition_segments = _execution_pair_segments(samples, "transition")
    policy_markers = _execution_marker_segments(samples, "policy", "policy")
    intervention_markers = _execution_marker_segments(samples, "intervention", "intervention")

    assert [segment["steps"] for segment in policy_segments] == [[0, 1]]
    assert [segment["steps"] for segment in intervention_segments] == [[2, 3]]
    assert [segment["steps"] for segment in transition_segments] == [[1, 2], [3, 4]]
    assert policy_markers[0]["steps"] == [0, 1]
    assert intervention_markers[0]["steps"] == [2, 3, 4]


def test_drawer_reference_uses_logged_handle_and_open_distance():
    from eval_act_oscbf_safety_metrics import _drawer_reference_from_samples

    ref = _drawer_reference_from_samples([
        {
            "episode": 0,
            "step": 1,
            "ee_pos": [0.0, 0.0, 0.0],
            "handle_pos": [0.875, -0.1, 0.666],
            "drawer_open_distance": 0.1,
            "drawer_open_fraction": 0.25,
        }
    ])

    assert ref is not None
    np.testing.assert_allclose(ref["handle_pos"], [0.875, -0.1, 0.666])
    assert ref["source"] in {"base_cabinet_600_xml", "handle_fallback"}
    assert ref["cabinet"]
    assert ref["drawer"]
    assert ref["handle"]
    if ref["source"] == "base_cabinet_600_xml":
        assert ref["absolute"] is True
        np.testing.assert_allclose(ref["origin"], [0.0, 0.0, 0.0])
        drawer_points = np.asarray(ref["drawer"], dtype=np.float64).reshape(-1, 3)
        assert drawer_points[:, 0].min() < 0.875 < drawer_points[:, 0].max()
        assert drawer_points[:, 1].min() < -0.1 < drawer_points[:, 1].max()
    else:
        np.testing.assert_allclose(ref["origin"], [0.875, 0.0, 0.666])
        np.testing.assert_allclose(ref["open_axis"], [0.0, -1.0, 0.0])
        assert ref["default_open"] == 0.1


def test_drawer_reference_prefers_logged_mujoco_geometry():
    from eval_act_oscbf_safety_metrics import _drawer_reference_from_samples

    cabinet = [[[1.0, 2.0, 3.0], [1.5, 2.0, 3.0]]]
    drawer = [[[2.0, 2.0, 3.0], [2.5, 2.0, 3.0]]]
    ref = _drawer_reference_from_samples([
        {
            "episode": 0,
            "step": 1,
            "ee_pos": [0.0, 0.0, 0.0],
            "object_state": {
                "handle_pos": [0.875, -0.1, 0.666],
                "drawer_open_distance": 0.1,
                "drawer_open_fraction": 0.25,
                "drawer_scene_geometry": {
                    "absolute": True,
                    "cabinet": cabinet,
                    "drawer": drawer,
                },
            },
        }
    ])

    assert ref is not None
    assert ref["absolute"] is True
    assert ref["source"] == "mujoco_geoms"
    assert ref["cabinet"] == cabinet
    assert ref["drawer"] == drawer
    np.testing.assert_allclose(ref["handle_pos"], [0.875, -0.1, 0.666])


def test_pause_scaling_keeps_base_as_delta_and_arm_as_absolute():
    from eval_act_oscbf_safety_metrics import (
        _pause_arm_at_current_q,
        _scale_controlled_motion_from_current_q,
    )

    q = np.linspace(-0.7, 0.7, 14, dtype=np.float32)
    action = np.zeros(16, dtype=np.float32)
    action[:4] = [0.2, -0.1, 0.05, 0.3]
    action[CTRL] = q[CTRL] + 0.4

    paused = _pause_arm_at_current_q(action, q, BASE_ARM_CTRL, BASE_ARM_CTRL)
    scaled = _scale_controlled_motion_from_current_q(
        action, q, BASE_ARM_CTRL, BASE_ARM_CTRL, scale=0.5
    )

    np.testing.assert_allclose(paused[:4], 0.0)
    np.testing.assert_allclose(paused[CTRL], q[CTRL])
    np.testing.assert_allclose(scaled[:4], 0.5 * action[:4])
    np.testing.assert_allclose(scaled[CTRL], q[CTRL] + 0.5 * (action[CTRL] - q[CTRL]))


def test_non_controlled_dimensions_are_preserved_by_deformation():
    filt = unsafe_filter(
        first_violation=1,
        safe=False,
        unsafe_deformation_fallback="best",
        deadlock_window=0,
    )
    filt.brake_progress_threshold = 1.0
    chunk = make_chunk()

    safe, info = filt.filter_chunk({"q": np.zeros(14)}, chunk)

    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])
    assert info["safety_mode"] == "horizon_deform"


def test_braking_holds_from_previous_safe_index():
    filt = unsafe_filter(first_violation=5, safe=False)
    chunk = make_chunk()

    braked, info = filt.horizon_brake(
        {"q": np.zeros(14)},
        chunk,
        {"first_violation": 5},
    )

    assert info["brake_stop_idx"] == 4
    np.testing.assert_allclose(braked[:4], chunk[:4])
    np.testing.assert_allclose(braked[4:, CTRL], np.repeat(chunk[4:5, CTRL], 12, axis=0))
    np.testing.assert_allclose(braked[:, PASS], chunk[:, PASS])


def test_immediate_violation_brake_holds_current_q():
    filt = unsafe_filter(first_violation=0, safe=False)
    chunk = make_chunk()
    q = np.linspace(-0.7, 0.7, 14, dtype=np.float32)

    braked, info = filt.horizon_brake(
        {"q": q},
        chunk,
        {"first_violation": 0},
    )

    assert info["brake_stop_idx"] == 0
    assert info["brake_hold_current"] is True
    np.testing.assert_allclose(braked[:, CTRL], np.repeat(q[None, CTRL], chunk.shape[0], axis=0))
    np.testing.assert_allclose(braked[:, PASS], chunk[:, PASS])



def test_path_consistent_q_transition_keeps_base_as_delta():
    filt = PathConsistentBrakeFilter(
        waypoint_substeps=1,
        certified_backup_enabled=False,
        controlled_action_indices=BASE_ARM_CTRL,
        controlled_state_indices=BASE_ARM_CTRL,
        control_type="absolute",
        debug=False,
    )
    action = np.zeros(16, dtype=np.float32)
    q_prev = np.linspace(-0.5, 0.8, 14, dtype=np.float32)
    q_next = q_prev.copy()
    q_next[:4] += [0.10, -0.05, 0.02, 0.30]
    q_next[CTRL] = np.linspace(0.2, 0.9, CTRL.shape[0], dtype=np.float32)

    converted = filt._action_from_q_transition(
        action, q_prev, q_next, BASE_ARM_CTRL, BASE_ARM_CTRL
    )
    hold = filt._make_hold_chunk_from_q(
        {"q": q_prev}, np.ones((3, 16), dtype=np.float32), q_prev, horizon=3
    )

    np.testing.assert_allclose(converted[:4], q_next[:4] - q_prev[:4])
    np.testing.assert_allclose(converted[CTRL], q_next[CTRL])
    np.testing.assert_allclose(hold[:, :4], 0.0)
    np.testing.assert_allclose(hold[:, CTRL], np.repeat(q_prev[None, CTRL], 3, axis=0))


def test_path_consistent_brake_filter_is_restored_under_requested_name():
    filt = PathConsistentBrakeFilter(
        waypoint_substeps=1,
        certified_backup_enabled=False,
        debug=False,
    )
    chunk = np.zeros((2, 16), dtype=np.float32)

    def evaluate(_obs, q_seq):
        return {
            "horizon_safe": True,
            "min_clearance": 1.0,
            "min_clearances": np.ones(q_seq.shape[0], dtype=np.float32),
            "first_violation": None,
            "unsafe_count": 0,
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = evaluate
    safe, info = filt.filter_chunk({"q": np.zeros(14, dtype=np.float32)}, chunk)

    assert isinstance(filt, PathConsistentBrakeFilter)
    assert info["filter_name"] == "path_consistent_brake"
    assert info["safety_mode"] == "pass_through"
    np.testing.assert_allclose(safe, chunk)

def test_deadlock_defers_deformation_until_window():
    filt = SafeChunkDeformFilter(
        brake_progress_threshold=0.5,
        deadlock_window=2,
        debug=False,
    )
    filt.task_progress_brake_threshold = 0.0
    chunk = np.zeros((16, 16), dtype=np.float32)
    chunk[:, CTRL] = 1.0

    def evaluate(_obs, q_seq):
        controlled = q_seq[:, CTRL]
        clearances = 0.2 - np.max(np.abs(controlled), axis=1)
        unsafe = np.flatnonzero(clearances < filt.min_clearance)
        return {
            "horizon_safe": bool(unsafe.size == 0),
            "min_clearance": float(clearances.min()),
            "min_clearances": clearances,
            "first_violation": int(unsafe[0]) if unsafe.size else None,
            "unsafe_count": int(unsafe.size),
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = evaluate

    safe, info = filt.filter_chunk({"q": np.zeros(14)}, chunk)

    assert info["deadlock"] is True
    assert info["deformation_deferred"] is True
    assert info["safety_mode"] == "horizon_brake"
    np.testing.assert_allclose(safe[:, CTRL], 0.0)

    safe, info = filt.filter_chunk({"q": np.zeros(14)}, chunk)

    assert info["deadlock_count"] == 2
    assert info["safety_mode"] == "horizon_deform"
    assert info["deform_safe"] is True
    assert safe.shape == chunk.shape


def test_unsafe_deformation_falls_back_to_brake():
    filt = unsafe_filter(first_violation=0, safe=False, deadlock_window=0)
    chunk = make_chunk()

    safe, info = filt.filter_chunk({"q": np.zeros(14)}, chunk)

    assert info["deformation_rejected"] is True
    assert info["fallback_reason"] == "deform_unsafe"
    assert info["safety_mode"] == "horizon_brake"
    np.testing.assert_allclose(safe[:, CTRL], 0.0)
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])


def test_fake_oscbf_changes_only_controlled_dimensions_and_reports_norm():
    filt = SafeChunkDeformFilter(oscbf_operator=FakeOSCBF(), debug=False)
    chunk = make_chunk()

    safe, info = filt.deform_chunk_with_oscbf({"q": np.zeros(14)}, chunk)

    np.testing.assert_allclose(safe[:, CTRL], chunk[:, CTRL] * 0.5)
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])
    assert info["deformation_norm"] > 0.0


def test_deform_chunk_uses_batched_oscbf_operator_when_available():
    op = FakeChunkOSCBF()
    filt = SafeChunkDeformFilter(oscbf_operator=op, debug=False)
    chunk = make_chunk()

    safe, info = filt.deform_chunk_with_oscbf({"q": np.zeros(14)}, chunk)

    assert op.chunk_calls == 1
    assert op.single_calls == 0
    np.testing.assert_allclose(safe[:, CTRL], chunk[:, CTRL] * 0.25)
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])
    assert info["operator_chunk_info"] is True
    assert info["sequential_oscbf_batched"] is True
    assert info["sequential_oscbf_batch_method"] == "filter_chunk"
    assert info["deformation_norm"] > 0.0


def test_existing_oscbf_import_is_not_broken():
    from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter

    assert OSCBFFilter is not None


def test_oscbf_base_conversion_uses_delta_under_absolute_control():
    from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter

    filt = object.__new__(OSCBFFilter)
    filt.control_type = "absolute"
    filt.dt = 0.05
    q_base = np.asarray([1.0, -2.0, 0.5, 0.25], dtype=np.float32)
    base_delta = np.asarray([0.10, -0.05, 0.02, 0.30], dtype=np.float32)

    velocity = filt._bigym_base_action_to_velocity(q_base, base_delta)
    recovered_action = filt._bigym_base_velocity_to_action(q_base, velocity)

    np.testing.assert_allclose(velocity, base_delta / filt.dt)
    np.testing.assert_allclose(recovered_action, base_delta)


class RaisingOSCBF:
    def __call__(self, *args, **kwargs):
        raise AssertionError("sequential OSCBF should not be used for chunk deformation")


def test_horizon_deform_uses_chunk_deformation_not_sequential_oscbf():
    filt = SafeChunkDeformFilter(
        oscbf_operator=RaisingOSCBF(),
        brake_progress_threshold=0.5,
        deadlock_window=1,
        debug=False,
    )
    chunk = np.zeros((16, 16), dtype=np.float32)
    chunk[:, CTRL] = 1.0

    def evaluate(_obs, q_seq):
        controlled = q_seq[:, CTRL]
        clearances = 0.2 - np.max(np.abs(controlled), axis=1)
        unsafe = np.flatnonzero(clearances < filt.min_clearance)
        return {
            "horizon_safe": bool(unsafe.size == 0),
            "min_clearance": float(clearances.min()),
            "min_clearances": clearances,
            "first_violation": int(unsafe[0]) if unsafe.size else None,
            "unsafe_count": int(unsafe.size),
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = evaluate

    safe, info = filt.filter_chunk({"q": np.zeros(14)}, chunk)

    assert info["safety_mode"] == "horizon_deform"
    assert info["deformation_source"] == "chunk_deform"
    assert info["deform_safe"] is True
    assert info["deformation_norm"] > 0.0
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])
    np.testing.assert_allclose(safe[:, CTRL], 0.0)



def _controlled_limit_safety(filt, limit):
    def evaluate(_obs, q_seq):
        controlled = q_seq[:, CTRL]
        clearances = limit - np.max(np.abs(controlled), axis=1)
        unsafe = np.flatnonzero(clearances < filt.min_clearance)
        return {
            "horizon_safe": bool(unsafe.size == 0),
            "min_clearance": float(clearances.min()),
            "min_clearances": clearances.astype(np.float32),
            "first_violation": int(unsafe[0]) if unsafe.size else None,
            "unsafe_count": int(unsafe.size),
            "safety_eval_available": True,
        }

    return evaluate


def test_optimized_mode_preserves_passthrough_and_improves_clearance():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        optimized_fallback="brake",
        opt_iters=2,
        opt_population=8,
        opt_seed=0,
        rejoin_threshold=1.0,
        recoverable_deform={"final_rejoin_metric": "q_state"},
        use_ee_pose_rejoin=False,
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=0.22)
    filt.task_progress_brake_threshold = 1.0
    chunk = make_chunk()
    chunk[:, CTRL] = np.linspace(0.0, 0.2, chunk.shape[0], dtype=np.float32)[:, None]
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)
    nominal_safety = filt.evaluate_horizon_safety(obs, nominal_q_seq)

    safe, info = filt.filter_chunk(obs, chunk)

    assert info["deform_mode"] == "optimized_recoverable_deform"
    assert info["deformation_source"] == "optimized_recoverable_deform"
    assert info["recovery_mode"] == "optimized_recoverable_deform"
    assert info["optimized_accepted"] is True
    assert info["deform_safe"] is True
    assert info["is_recoverable"] is True
    assert info["rejoin_index"] >= filt.min_rejoin_offset
    assert info["rejoin_index"] < chunk.shape[0]
    assert info["act_resume_index"] == info["rejoin_index"]
    assert info["act_resume_supported"] is False
    assert info["deform_min_clearance"] > nominal_safety["min_clearance"]
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])


def test_optimized_mode_direct_result_rejects_high_rejoin_loss():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        opt_iters=1,
        opt_population=6,
        opt_seed=0,
        rejoin_threshold=1e-6,
        recoverable_deform={
            "final_rejoin_metric": "q_state",
            "q_rejoin_threshold": 1e-6,
        },
        use_ee_pose_rejoin=False,
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=0.2)
    chunk = make_chunk()
    chunk[:, CTRL] = 1.0
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert info["deform_safe"] is True
    assert info["is_recoverable"] is False
    assert info["q_rejoin_dist"] > filt.q_rejoin_threshold
    assert info["rejoin_index"] >= filt.min_rejoin_offset
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])


def test_optimized_mode_falls_back_to_candidate_when_unrecoverable():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        optimized_fallback="candidate",
        unsafe_deformation_fallback="best",
        opt_iters=1,
        opt_population=6,
        opt_seed=0,
        rejoin_threshold=1e-6,
        recoverable_deform={
            "final_rejoin_metric": "q_state",
            "q_rejoin_threshold": 1e-6,
        },
        brake_if_unrecoverable=False,
        safechunk_acceptance={"enabled": False},
        deadlock_window=0,
        use_ee_pose_rejoin=False,
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=0.2)
    chunk = make_chunk()
    chunk[:, CTRL] = 1.0

    safe, info = filt.filter_chunk({"q": np.zeros(14, dtype=np.float32)}, chunk)

    assert info["deform_mode"] == "optimized"
    assert info["optimized_accepted"] is False
    assert info["optimized_fallback"] == "candidate"
    assert info["optimized_reject_reason"] == "unrecoverable"
    assert info["rejection_cause"] == "unrecoverable"
    assert info["deformation_source"] == "chunk_deform"
    assert info["deform_safe"] is True
    np.testing.assert_allclose(safe[:, CTRL], 0.0)
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])


def test_optimized_mode_falls_back_to_brake_when_unrecoverable():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        optimized_fallback="brake",
        opt_iters=1,
        opt_population=6,
        opt_seed=0,
        rejoin_threshold=1e-6,
        recoverable_deform={
            "final_rejoin_metric": "q_state",
            "q_rejoin_threshold": 1e-6,
        },
        deadlock_window=0,
        safechunk_acceptance={"enabled": False},
        use_ee_pose_rejoin=False,
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=0.2)
    chunk = make_chunk()
    chunk[:, CTRL] = 1.0

    safe, info = filt.filter_chunk({"q": np.zeros(14, dtype=np.float32)}, chunk)

    assert info["optimized_accepted"] is False
    assert info["optimized_fallback"] == "brake"
    assert info["fallback_reason"] == "unrecoverable"
    assert info["rejection_cause"] == "unrecoverable"
    assert info["safety_mode"] == "horizon_brake"
    np.testing.assert_allclose(safe[:, CTRL], 0.0)
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])


def test_recovery_disabled_preserves_safe_optimized_acceptance():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        optimized_fallback="brake",
        opt_iters=1,
        opt_population=6,
        opt_seed=0,
        rejoin_threshold=1e-12,
        recoverable_deform={"enabled": False},
        deadlock_window=0,
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=0.2)
    filt.task_progress_brake_threshold = 1.0
    chunk = make_chunk()
    chunk[:, CTRL] = 1.0

    safe, info = filt.filter_chunk({"q": np.zeros(14, dtype=np.float32)}, chunk)

    assert info["optimized_accepted"] is True
    assert info["recoverable_deform_enabled"] is False
    assert info["is_recoverable"] is None
    assert info["rejoin_loss"] == 0.0
    assert info["total_loss"] == info["existing_optimization_loss"]
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])


def test_recovery_enabled_adds_rejoin_loss_to_objective():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        lambda_rejoin=5.0,
        recoverable_deform={
            "enabled": True,
            "lambda_rejoin": 5.0,
            "inner_rejoin_metric": "q_state",
            "final_rejoin_metric": "none",
            "use_ee_pose_rejoin": False,
        },
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal = make_chunk()
    nominal[:, CTRL] = 1.0
    candidate = nominal.copy()
    candidate[:, CTRL] = 0.0
    nominal_q_seq = filt.rollout_nominal_chunk(obs, nominal)

    cost, losses = filt._optimized_deformation_cost(
        obs,
        candidate,
        nominal,
        nominal_q_seq,
        None,
        None,
    )

    assert losses["rejoin_loss"] > 0.0
    assert losses["j_best"] >= filt.min_rejoin_offset
    assert losses["j_best"] < nominal.shape[0]
    np.testing.assert_allclose(
        cost,
        losses["existing_optimization_loss"]
        + filt.lambda_rejoin * losses["rejoin_loss"],
    )


def test_recovery_disabled_removes_rejoin_from_objective():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        lambda_rejoin=5.0,
        recoverable_deform={"enabled": False},
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal = make_chunk()
    nominal[:, CTRL] = 1.0
    candidate = nominal.copy()
    candidate[:, CTRL] = 0.0
    nominal_q_seq = filt.rollout_nominal_chunk(obs, nominal)

    cost, losses = filt._optimized_deformation_cost(
        obs,
        candidate,
        nominal,
        nominal_q_seq,
        None,
        None,
    )

    assert losses["rejoin_loss"] == 0.0
    assert losses["j_best"] is None
    assert cost == losses["existing_optimization_loss"]



def test_recovery_default_inner_q_caches_nominal_ee_for_final_check():
    op = CountingEEOperator()
    filt = SafeChunkDeformFilter(
        oscbf_operator=op,
        mode="optimized",
        opt_iters=1,
        opt_population=6,
        opt_seed=0,
        recoverable_deform={
            "enabled": True,
            "inner_rejoin_metric": "q_state",
            "final_rejoin_metric": "ee_pose",
            "cache_nominal_ee": True,
            "q_rejoin_threshold": 10.0,
            "ee_rejoin_threshold": 10.0,
        },
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert info["inner_rejoin_metric"] == "q_state"
    assert info["final_rejoin_metric"] == "ee_pose"
    assert info["ee_final_check_available"] is True
    assert info["is_recoverable"] is True
    assert op.sequence_calls == 1
    assert op.pose_calls == 1
    assert info["ee_nom_cache_time_ms"] >= 0.0
    assert info["ee_final_check_time_ms"] >= 0.0
    assert info["rejoin_q_eval_time_ms"] >= 0.0


def test_legacy_ee_inner_loop_mode_is_config_only():
    op = CountingEEOperator()
    filt = SafeChunkDeformFilter(
        oscbf_operator=op,
        mode="optimized",
        lambda_rejoin=5.0,
        recoverable_deform={
            "enabled": True,
            "lambda_rejoin": 5.0,
            "inner_rejoin_metric": "ee_pose",
            "final_rejoin_metric": "none",
            "cache_nominal_ee": True,
            "q_rejoin_threshold": 10.0,
        },
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal = make_chunk()
    candidate = nominal.copy()
    candidate[:, CTRL] = 0.0
    nominal_q_seq = filt.rollout_nominal_chunk(obs, nominal)
    rejoin_context = filt._make_rejoin_context(nominal_q_seq)

    cost, losses = filt._optimized_deformation_cost(
        obs,
        candidate,
        nominal,
        nominal_q_seq,
        None,
        None,
        rejoin_context=rejoin_context,
    )

    assert losses["rejoin_space"] == "ee_pose"
    assert op.sequence_calls == 2
    assert cost == losses["total_loss"]



def test_debug_safety_feasibility_skips_ee_final_check_and_reports_gap():
    op = CountingEEOperator()
    filt = SafeChunkDeformFilter(
        oscbf_operator=op,
        mode="optimized",
        opt_iters=1,
        opt_population=6,
        opt_seed=0,
        min_clearance=0.3,
        recoverable_deform={
            "enabled": True,
            "inner_rejoin_metric": "q_state",
            "final_rejoin_metric": "ee_pose",
            "cache_nominal_ee": True,
            "q_rejoin_threshold": 10.0,
            "ee_rejoin_threshold": 10.0,
        },
        optimized_deform={"debug_safety_feasibility": True},
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=0.2)
    chunk = make_chunk()
    chunk[:, CTRL] = 1.0
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert filt.final_rejoin_metric == "none"
    assert info["final_rejoin_metric"] == "none"
    assert info["debug_safety_feasibility"] is True
    assert info["ee_final_check_available"] is None
    assert info["ee_nom_cache_time_ms"] == 0.0
    assert info["ee_final_check_time_ms"] == 0.0
    assert op.sequence_calls == 0
    assert op.pose_calls == 0
    assert info["best_min_clearance"] == info["min_clearance"]
    assert info["required_min_clearance"] == filt.min_clearance
    np.testing.assert_allclose(
        info["clearance_gap"],
        filt.min_clearance - info["best_min_clearance"],
    )
    assert info["rejection_cause"] in {
        "unsafe",
        "unsafe_and_unrecoverable",
        None,
    }



def _explicit_recovery_filter(**kwargs):
    recoverable = {
        "enabled": True,
        "explicit_recovery": True,
        "final_rejoin_metric": "q_state",
        "cache_nominal_ee": False,
        "use_ee_final_check": False,
        "q_rejoin_threshold": kwargs.pop("q_rejoin_threshold", 10.0),
        "deform_horizon": kwargs.pop("yield_horizon", kwargs.pop("deform_horizon", 4)),
        "recover_horizon": kwargs.pop("return_horizon", kwargs.pop("recover_horizon", 4)),
        "acceptance_clearance_tol": 0.005,
    }
    recoverable.update(kwargs.pop("recoverable_deform", {}))
    filt = SafeChunkDeformFilter(
        mode="optimized",
        opt_iters=1,
        opt_population=6,
        opt_seed=0,
        recoverable_deform=recoverable,
        debug=False,
        **kwargs,
    )
    return filt



def test_deprecated_explicit_recovery_config_keys_are_mapped(caplog):
    with caplog.at_level(logging.WARNING):
        filt = SafeChunkDeformFilter(
            mode="optimized",
            recoverable_deform={
                "enabled": True,
                "explicit_return": True,
                "yield_horizon": 3,
                "return_horizon": 5,
                "lambda_yield_safety": 123.0,
                "lambda_return_safety": 456.0,
            },
            debug=False,
        )

    assert filt.explicit_return is True
    assert filt.yield_horizon == 3
    assert filt.return_horizon == 5
    assert filt.lambda_yield_safety == 123.0
    assert filt.lambda_return_safety == 456.0
    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "explicit_return" in warning_text
    assert "explicit_recovery" in warning_text
    assert "yield_horizon" in warning_text
    assert "deform_horizon" in warning_text
    assert "return_horizon" in warning_text
    assert "recover_horizon" in warning_text

def _acceptance_filter(clearances):
    filt = SafeChunkDeformFilter(
        mode="optimized",
        safechunk_acceptance={
            "enabled": True,
            "hard_min_clearance": 0.02,
            "desired_min_clearance": 0.08,
            "prefix_min_clearance": 0.04,
            "min_safe_prefix_len": 1,
            "allow_safe_prefix_execution": True,
            "full_horizon_required_for_recover": False,
            "full_horizon_required_for_deform": False,
        },
        debug=False,
    )
    clearances = np.asarray(clearances, dtype=np.float32)

    def fake_safety(_obs, q_seq):
        h = clearances[: q_seq.shape[0]]
        if h.shape[0] < q_seq.shape[0]:
            h = np.pad(h, (0, q_seq.shape[0] - h.shape[0]), mode="edge")
        unsafe_idx = np.flatnonzero(h < filt.min_clearance)
        return {
            "horizon_safe": bool(unsafe_idx.size == 0),
            "min_clearance": float(np.min(h)),
            "min_clearances": h,
            "first_violation": int(unsafe_idx[0]) if unsafe_idx.size else None,
            "unsafe_count": int(unsafe_idx.size),
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = fake_safety
    return filt


def test_candidate_acceptance_full_horizon():
    filt = _acceptance_filter([0.09, 0.10, 0.11, 0.12])
    chunk = make_chunk(h=4)
    info = filt.evaluate_candidate_acceptance({"q": np.zeros(14, dtype=np.float32)}, chunk, "deform")

    assert info["accepted"] is True
    assert info["acceptance_type"] == "full_horizon"


def test_candidate_acceptance_safe_prefix():
    filt = _acceptance_filter([0.06, 0.05, 0.03, 0.01])
    chunk = make_chunk(h=4)
    info = filt.evaluate_candidate_acceptance({"q": np.zeros(14, dtype=np.float32)}, chunk, "deform")

    assert info["accepted"] is True
    assert info["acceptance_type"] == "safe_prefix"
    assert info["safe_prefix_len"] == 2


def test_candidate_acceptance_first_action_only_truncates_suffix():
    filt = _acceptance_filter([0.05, 0.03, 0.02, 0.01])
    chunk = make_chunk(h=4)
    chunk[:, CTRL] = np.arange(4, dtype=np.float32)[:, None]
    info = filt.evaluate_candidate_acceptance({"q": np.zeros(14, dtype=np.float32)}, chunk, "deform")
    safe = filt._truncate_chunk_to_safe_prefix(chunk, info)

    assert info["accepted"] is True
    assert info["acceptance_type"] in {"safe_prefix", "first_action_only"}
    assert info["safe_prefix_len"] == 1
    np.testing.assert_allclose(safe[1:], np.repeat(safe[0][None, :], safe.shape[0] - 1, axis=0))


def test_candidate_acceptance_immediate_hard_reject():
    filt = _acceptance_filter([0.01, 0.09, 0.09])
    chunk = make_chunk(h=3)
    info = filt.evaluate_candidate_acceptance({"q": np.zeros(14, dtype=np.float32)}, chunk, "deform")

    assert info["accepted"] is False
    assert info["acceptance_type"] == "emergency_brake"
    assert info["rejection_reason"] == "immediate_below_hard_margin"


def test_recover_low_full_horizon_safe_first_action_is_accepted():
    filt = _acceptance_filter([0.05, 0.04, 0.035, 0.03])
    chunk = make_chunk(h=4)
    info = filt.evaluate_candidate_acceptance({"q": np.zeros(14, dtype=np.float32)}, chunk, "recover")

    assert info["accepted"] is True
    assert info["acceptance_type"] in {"safe_prefix", "first_action_only"}
    assert info["horizon_min_clearance"] < info["desired_min_clearance"]


def test_explicit_recovery_info_uses_no_old_public_keys():
    filt = _explicit_recovery_filter(q_rejoin_threshold=10.0)
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    for old_key in (
        "yield_accepted",
        "return_accepted",
        "yield_min_clearance",
        "return_min_clearance",
        "return_rejoin_loss",
        "return_target_index",
    ):
        assert old_key not in info
    assert "deform_stage_accepted" in info
    assert "recover_accepted" in info


def test_explicit_recovery_creates_recovery_context():
    filt = _explicit_recovery_filter()
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
        act_chunk_index=3,
    )

    assert filt.recovery_context is not None
    assert filt.recovery_context.start_chunk_index == 3
    np.testing.assert_allclose(filt.recovery_context.nominal_chunk, chunk)
    np.testing.assert_allclose(filt.recovery_context.nominal_q_seq, nominal_q_seq)
    assert info["explicit_recovery"] is True


def test_explicit_recovery_yield_accepts_without_rejoin_passing():
    filt = _explicit_recovery_filter(q_rejoin_threshold=0.0)
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert info["deform_stage_accepted"] is True
    assert info["recover_accepted"] is False
    assert info["is_recoverable"] is False
    assert info["rejection_cause"] == "unrecoverable"


def test_explicit_recovery_requires_return_rejoin_loss_to_pass():
    filt = _explicit_recovery_filter(q_rejoin_threshold=0.0)
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert info["recover_min_clearance"] >= filt.min_clearance - filt.acceptance_clearance_tol
    assert info["recover_rejoin_loss"] >= filt.q_rejoin_threshold
    assert info["recover_accepted"] is False
    assert info["optimized_accepted"] is not True if "optimized_accepted" in info else True


def test_task_progress_recovery_cost_uses_return_seed_as_direction_reference():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        debug=False,
        controlled_action_indices=BASE_ARM_CTRL,
        controlled_state_indices=BASE_ARM_CTRL,
        control_type="absolute",
        safechunk_recover={
            "direction_alignment_weight": 10.0,
            "min_direction_cosine": 0.7,
            "ordered_pose_weight": 1.0,
            "ordered_delta_weight": 1.0,
        },
    )
    filt.evaluate_horizon_safety = lambda _obs, q_seq: {
        "horizon_safe": True,
        "min_clearance": 1.0,
        "min_clearances": np.ones(q_seq.shape[0], dtype=np.float32),
    }
    obs = {"q": np.zeros(14, dtype=np.float32)}
    reference = np.zeros((3, 16), dtype=np.float32)
    reference[:, 0] = 0.1
    reference[:, CTRL] = 0.2
    stale_latest = -reference.copy()
    filt.latest_nominal_chunk = stale_latest

    _cost, losses = filt._recover_task_progress_cost(
        obs,
        reference.copy(),
        reference.copy(),
        BASE_ARM_CTRL,
        reference_chunk=reference,
    )

    assert losses["recover_direction_ok"] is True
    assert losses["recover_direction_cosine"] > 0.99
    assert losses["recover_ordered_ok"] is True


def test_task_progress_recovery_checks_q_rejoin_without_optimizing_it():
    filt = _explicit_recovery_filter(q_rejoin_threshold=0.5)
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)
    calls = {"q": 0, "qd": 0}

    def fake_q_rejoin_loss(q_seq, nominal_q_seq=None, rejoin_context=None):
        del q_seq, nominal_q_seq, rejoin_context
        calls["q"] += 1
        return 1.25, 3, 0.5

    def fake_qd_rejoin_loss(q_seq, nominal_q_seq=None, target_index=None, rejoin_context=None):
        del q_seq, nominal_q_seq, target_index, rejoin_context
        calls["qd"] += 1
        return 0.01, 3, 0.25

    filt._q_rejoin_loss = fake_q_rejoin_loss
    filt._qd_rejoin_loss = fake_qd_rejoin_loss

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert calls["q"] == 1
    assert calls["qd"] == 1
    assert info["recover_to_task_progress"] is True
    assert info["recover_rejoin_loss"] == 1.25
    assert info["q_rejoin_loss"] == 1.25
    assert info["q_rejoin_dist"] == np.sqrt(1.25)
    assert info["qd_rejoin_loss"] == 0.01
    assert info["qd_rejoin_dist"] == 0.1
    assert info["recover_target_index"] == 3
    assert info["recover_accepted"] is False
    assert info["rejection_cause"] == "unrecoverable"


def test_task_progress_recovery_treats_bad_qd_rejoin_as_soft_by_default():
    filt = _explicit_recovery_filter(q_rejoin_threshold=0.5)
    filt.qd_rejoin_threshold = 0.5

    reason = filt._recovery_reject_reason(
        {
            "q_rejoin_ok": True,
            "qd_rejoin_ok": False,
            "qd_rejoin_required": False,
            "qd_rejoin_hard_failed": False,
        },
        {"immediate_safe": True, "prefix_safe": True, "path_safe": True},
        direction_ok=True,
        ordered_ok=True,
    )

    assert reason is None



def test_task_progress_recovery_rejects_bad_qd_rejoin_when_required():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=0.5,
        recoverable_deform={"require_qd_rejoin": True},
    )
    filt.qd_rejoin_threshold = 0.5
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    def fake_q_rejoin_loss(q_seq, nominal_q_seq=None, rejoin_context=None):
        del q_seq, nominal_q_seq, rejoin_context
        return 0.01, 3, 0.5

    def fake_qd_rejoin_loss(q_seq, nominal_q_seq=None, target_index=None, rejoin_context=None):
        del q_seq, nominal_q_seq, target_index, rejoin_context
        return 4.0, 3, 0.25

    filt._q_rejoin_loss = fake_q_rejoin_loss
    filt._qd_rejoin_loss = fake_qd_rejoin_loss

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert info["recover_rejoin_loss"] == 0.01
    assert info["q_rejoin_dist"] == 0.1
    assert info["qd_rejoin_loss"] == 4.0
    assert info["qd_rejoin_dist"] == 2.0
    assert info["recover_accepted"] is False
    assert info["rejection_cause"] == "unrecoverable"


def test_recovery_path_safety_distinguishes_path_from_terminal_rejoin():
    filt = _explicit_recovery_filter(
        safechunk_recovery_corridor={
            "enabled": True,
            "recover_path_min_clearance": 0.04,
            "recover_immediate_hard_clearance": 0.02,
            "recover_prefix_min_clearance": 0.04,
        }
    )
    clearances = np.asarray([0.05, 0.03, 0.05], dtype=np.float32)

    def fake_safety(_obs, q_seq):
        h = clearances[: q_seq.shape[0]]
        if h.shape[0] < q_seq.shape[0]:
            h = np.pad(h, (0, q_seq.shape[0] - h.shape[0]), mode="edge")
        return {
            "horizon_safe": bool(np.min(h) >= filt.min_clearance),
            "min_clearance": float(np.min(h)),
            "min_clearances": h,
            "first_violation": 1,
            "unsafe_count": int(np.count_nonzero(h < filt.min_clearance)),
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = fake_safety
    chunk = np.zeros((3, 16), dtype=np.float32)

    info = filt.evaluate_recovery_path_safety(
        {"q": np.zeros(14, dtype=np.float32)},
        chunk,
    )

    assert info["immediate_safe"] is True
    assert info["prefix_safe"] is True
    assert info["path_safe"] is False
    assert info["reject_reason"] == "path_unsafe"
    assert np.isclose(info["recover_path_min_clearance"], 0.03)


def test_recovery_target_failure_memory_cools_down_repeated_path_failures():
    filt = _explicit_recovery_filter(
        safechunk_recovery_corridor={
            "enabled": True,
            "unsafe_recovery_cooldown_steps": 3,
            "max_same_target_failures": 2,
        }
    )
    target = np.zeros((2, 16), dtype=np.float32)
    key = filt.make_recovery_target_key(target)
    path_key = filt._make_recovery_path_key(target, key)

    filt._mark_recovery_path_failure(key, path_key, "path_unsafe")
    assert filt._recovery_target_is_suppressed(key) is False
    filt._mark_recovery_path_failure(key, path_key, "prefix_unsafe")

    assert filt._recovery_target_is_suppressed(key) is True
    assert filt.recovery_target_failure_counts[key] == 2
    assert filt.recover_path_unsafe_count == 2
    assert filt.recovery_path_failure_streak == 2

    for _ in range(3):
        filt._tick_unsafe_recovery_cooldowns()

    assert filt._recovery_target_is_suppressed(key) is False


def test_explicit_recovery_delays_when_direct_recovery_corridor_is_unsafe():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        safechunk_recovery_corridor={
            "enabled": True,
            "enable_detour_rejoin": False,
            "enable_delayed_rejoin": True,
            "delayed_rejoin_wait_steps": 4,
        },
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)

    def fake_recovery_path(_obs, _chunk, candidate_name="recover"):
        return {
            "path_safe": False,
            "immediate_safe": True,
            "prefix_safe": True,
            "recover_path_min_clearance": 0.03,
            "recover_immediate_clearance": 0.05,
            "recover_prefix_min_clearance": 0.05,
            "safe_prefix_len": 1,
            "reject_reason": "path_unsafe",
            "candidate_name": candidate_name,
        }

    filt.evaluate_recovery_path_safety = fake_recovery_path
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert info["recover_accepted"] is False
    assert info["recover_reject_reason"] == "path_unsafe"
    assert info["recovery_candidate_class"] == "delayed_rejoin"
    assert info["direct_rejoin_attempted"] is True
    assert info["direct_rejoin_rejected"] is True
    assert info["delayed_rejoin_active"] is True
    assert info["recover_path_unsafe_count"] == 1


def test_explicit_recovery_accepts_safe_detour_after_direct_corridor_reject():
    filt = _explicit_recovery_filter(q_rejoin_threshold=10.0)
    filt.enable_detour_rejoin = True
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)

    def fake_recovery_path(_obs, _chunk, candidate_name="recover"):
        safe = candidate_name == "recover_detour"
        return {
            "path_safe": safe,
            "immediate_safe": True,
            "prefix_safe": True,
            "recover_path_min_clearance": 0.08 if safe else 0.03,
            "recover_immediate_clearance": 0.08,
            "recover_prefix_min_clearance": 0.08,
            "safe_prefix_len": 2,
            "reject_reason": None if safe else "path_unsafe",
            "candidate_name": candidate_name,
        }

    def fake_detours(_obs, direct_chunk, _action_idx):
        return [("test_detour", np.asarray(direct_chunk, dtype=np.float32).copy())]

    filt.evaluate_recovery_path_safety = fake_recovery_path
    filt._make_recovery_detour_candidates = fake_detours
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert info["recover_accepted"] is True
    assert info["recovery_candidate_class"] == "detour_rejoin"
    assert info["recover_reject_reason"] is None
    assert info["direct_rejoin_rejected"] is True
    assert info["detour_rejoin_attempted"] is True
    assert info["detour_rejoin_accepted"] is True
    assert info["recover_path_min_clearance"] == 0.08


def test_committed_recovery_completion_requests_action_history_reset():
    filt = _explicit_recovery_filter()
    filt.committed_rejoin_index = 7

    info = filt._committed_info({}, "recover", 1, 2, completed=True)

    assert info["resume_from_committed_rejoin"] is True
    assert info["request_action_history_reset_after_recovery"] is True


def test_explicit_recovery_resume_index_matches_target_index():
    filt = _explicit_recovery_filter(q_rejoin_threshold=10.0)
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert info["recover_accepted"] is True
    assert info["recover_target_index"] is not None
    assert info["act_resume_index"] == info["recover_target_index"]
    assert info["resumed_from_recover_index"] == info["recover_target_index"]


def test_explicit_recovery_fallback_used_when_return_fails():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=0.0,
        deadlock_window=0,
        safechunk_acceptance={"enabled": False},
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=0.2)
    chunk = make_chunk()
    chunk[:, CTRL] = 1.0

    safe, info = filt.filter_chunk({"q": np.zeros(14, dtype=np.float32)}, chunk)

    assert info["deform_stage_accepted"] is True
    assert info["recover_accepted"] is False
    assert info["fallback_used"] is True
    assert info["optimized_accepted"] is False
    assert info["safety_mode"] == "horizon_brake"
    np.testing.assert_allclose(safe[:, CTRL], 0.0)




def test_explicit_recovery_commits_accepted_chunk_and_serves_return_step():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        yield_horizon=1,
        return_horizon=2,
        deadlock_window=0,
        explicit_recovery={"opportunistic_act_resume": False},
    )
    filt.task_progress_brake_threshold = 1.0
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1

    def staged_safety(_obs, q_seq):
        clearances = np.ones(q_seq.shape[0], dtype=np.float32)
        if q_seq.shape[0] >= chunk.shape[0]:
            clearances[-1] = 0.0
        unsafe_idx = np.flatnonzero(clearances < filt.min_clearance)
        return {
            "horizon_safe": bool(unsafe_idx.size == 0),
            "min_clearance": float(clearances.min()),
            "min_clearances": clearances,
            "first_violation": int(unsafe_idx[0]) if unsafe_idx.size else None,
            "unsafe_count": int(unsafe_idx.size),
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = staged_safety
    obs = {"q": np.zeros(14, dtype=np.float32)}

    safe0, info0 = filt.filter_chunk(obs, chunk)

    np.testing.assert_allclose(safe0[:, PASS], chunk[:, PASS])
    assert info0["optimized_accepted"] is True
    assert info0["committed_chunk_active"] is True
    assert info0["committed_chunk_mode"] == "horizon_deform"
    assert info0["deform_steps_executed"] == 1
    assert filt.committed_chunk is not None
    assert filt.committed_chunk_index == 1

    replay_q = np.zeros(14, dtype=np.float32)
    replay_q[CTRL] = safe0[0, CTRL]
    safe1, info1 = filt.filter_chunk({"q": replay_q}, chunk)

    np.testing.assert_allclose(safe1[:, PASS], chunk[:, PASS])
    assert info1["committed_chunk_active"] is True
    assert info1["committed_chunk_mode"] == "recover"
    assert info1["recover_steps_executed"] == 1
    assert info1["committed_chunk_index"] == 1
    assert filt.committed_chunk_index == 2


def test_committed_recovery_opportunistically_resumes_act_when_rejoined():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        yield_horizon=1,
        return_horizon=2,
        explicit_recovery={
            "opportunistic_act_resume": True,
            "opportunistic_resume_min_clearance": 0.08,
        },
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)
    filt.recovery_context = RecoveryContext(
        nominal_chunk=chunk.copy(),
        nominal_q_seq=nominal_q_seq.copy(),
        active=False,
        target_rejoin_index=3,
        phase="recover",
    )
    info = {
        "optimized_accepted": True,
        "deform_chunk_length": 1,
        "recover_chunk_length": 2,
        "recover_target_index": 3,
        "recover_min_clearance": 1.0,
    }

    committed, reject_info = filt._commit_explicit_recovery_chunk(obs, chunk, info)
    assert committed is True
    assert reject_info == {}
    filt.committed_chunk_index = 1
    replay_q = nominal_q_seq[0].copy()

    result = filt._serve_committed_chunk(
        {"q": replay_q},
        chunk,
        chunk.shape,
        q_full=replay_q,
    )

    assert result is not None
    safe_chunk, replay_info = result
    np.testing.assert_allclose(safe_chunk, chunk)
    assert replay_info["committed_opportunistic_resume"] is True
    assert replay_info["committed_released_for_act_resume"] is True
    assert replay_info["mode"] == "pass_through"
    assert replay_info["recover_steps_executed"] == 0
    assert replay_info["act_resume_index"] is not None
    assert filt.committed_chunk is None
    assert filt.committed_opportunistic_resume_count == 1


def test_committed_recovery_budget_exit_replans_when_not_rejoined():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        yield_horizon=1,
        return_horizon=2,
        explicit_recovery={
            "opportunistic_act_resume": True,
            "opportunistic_resume_q_threshold": 0.0,
            "max_recover_steps_before_act_resume": 1,
        },
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)
    filt.recovery_context = RecoveryContext(
        nominal_chunk=chunk.copy(),
        nominal_q_seq=nominal_q_seq.copy(),
        active=False,
        target_rejoin_index=3,
        phase="recover",
    )
    info = {
        "optimized_accepted": True,
        "deform_chunk_length": 1,
        "recover_chunk_length": 2,
        "recover_target_index": 3,
        "recover_min_clearance": 1.0,
    }

    committed, reject_info = filt._commit_explicit_recovery_chunk(obs, chunk, info)
    assert committed is True
    assert reject_info == {}
    filt.committed_chunk_index = 1
    filt.committed_recover_steps_since_act = 1
    replay_q = nominal_q_seq[0].copy()

    result = filt._serve_committed_chunk(
        {"q": replay_q},
        chunk,
        chunk.shape,
        q_full=replay_q,
    )

    assert result is None
    assert filt.committed_chunk is None
    pending = filt._pop_pending_committed_replan_info()
    assert pending["committed_recovery_budget_exit"] is True
    assert pending["committed_replan_due_to_recovery_budget"] is True
    assert filt.committed_recovery_budget_exit_count == 1


def test_explicit_recovery_commit_disabled_preserves_replanning_behavior():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        yield_horizon=1,
        return_horizon=2,
        deadlock_window=0,
        explicit_recovery={"commit_accepted_chunks": False},
    )
    filt.task_progress_brake_threshold = 1.0
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1

    def staged_safety(_obs, q_seq):
        clearances = np.ones(q_seq.shape[0], dtype=np.float32)
        if q_seq.shape[0] >= chunk.shape[0]:
            clearances[-1] = 0.0
        unsafe_idx = np.flatnonzero(clearances < filt.min_clearance)
        return {
            "horizon_safe": bool(unsafe_idx.size == 0),
            "min_clearance": float(clearances.min()),
            "min_clearances": clearances,
            "first_violation": int(unsafe_idx[0]) if unsafe_idx.size else None,
            "unsafe_count": int(unsafe_idx.size),
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = staged_safety

    _safe, info = filt.filter_chunk({"q": np.zeros(14, dtype=np.float32)}, chunk)

    assert info["optimized_accepted"] is True
    assert info.get("committed_chunk_active") is None
    assert filt.committed_chunk is None

def test_committed_replay_compares_post_clearance_to_planned_post_clearance():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        yield_horizon=1,
        return_horizon=2,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    info = {
        "optimized_accepted": True,
        "deform_chunk_length": 1,
        "recover_chunk_length": 2,
        "recover_target_index": 3,
        "recover_min_clearance": 1.0,
    }

    committed, reject_info = filt._commit_explicit_recovery_chunk(obs, chunk, info)
    served = filt._serve_committed_chunk(obs, chunk, chunk.shape)

    assert committed is True
    assert reject_info == {}
    assert served is not None
    _safe, replay_info = served
    np.testing.assert_allclose(replay_info["planned_clearance_pre"], 2.0)
    np.testing.assert_allclose(replay_info["planned_clearance_post"], 1.9)
    np.testing.assert_allclose(replay_info["replay_clearance_post"], 1.9)
    np.testing.assert_allclose(
        replay_info["planning_vs_replay_clearance_post_error"],
        0.0,
    )
    np.testing.assert_allclose(
        replay_info["actual_vs_planned_post_q_error"],
        0.0,
    )


def test_committed_replay_allows_low_current_clearance_when_post_action_safe():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        yield_horizon=1,
        return_horizon=2,
        explicit_recovery={
            "committed_abort_only_if_contact_risk": True,
            "committed_min_clearance_for_abort": 0.08,
            "committed_state_mismatch_abort_requires_unsafe": True,
        },
    )

    def current_low_post_safe(_obs, q_seq):
        controlled = np.max(np.abs(q_seq[:, CTRL]), axis=1)
        clearances = np.where(controlled < 0.05, 0.03, 1.0).astype(np.float32)
        unsafe = np.flatnonzero(clearances < filt.min_clearance)
        return {
            "horizon_safe": bool(unsafe.size == 0),
            "min_clearance": float(clearances.min()),
            "min_clearances": clearances,
            "first_violation": int(unsafe[0]) if unsafe.size else None,
            "unsafe_count": int(unsafe.size),
            "safety_eval_available": True,
        }

    filt.evaluate_horizon_safety = current_low_post_safe
    chunk = make_chunk()
    chunk[:, CTRL] = 0.2
    obs = {"q": np.zeros(14, dtype=np.float32)}
    info = {
        "optimized_accepted": True,
        "deform_chunk_length": 1,
        "recover_chunk_length": 2,
        "recover_target_index": 3,
        "recover_min_clearance": 1.0,
    }

    committed, reject_info = filt._commit_explicit_recovery_chunk(obs, chunk, info)
    served = filt._serve_committed_chunk(
        obs,
        chunk,
        chunk.shape,
        q_full=obs["q"],
        live_monitor_min_h=0.03,
    )

    assert committed is True
    assert reject_info == {}
    assert served is not None
    _safe, replay_info = served
    assert replay_info["committed_aborted_due_to_safety"] is False
    assert replay_info["committed_live_monitor_clearance"] == 0.03
    np.testing.assert_allclose(replay_info["committed_execution_min_clearance"], 1.0)
    assert replay_info["replay_clearance_post"] == 1.0


def test_committed_replay_state_mismatch_replans_before_safety_repair():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        yield_horizon=1,
        return_horizon=2,
    )
    assert filt.committed_state_mismatch_abort_requires_unsafe is False
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    info = {
        "optimized_accepted": True,
        "deform_chunk_length": 1,
        "recover_chunk_length": 2,
        "recover_target_index": 3,
        "recover_min_clearance": 1.0,
    }

    committed, reject_info = filt._commit_explicit_recovery_chunk(obs, chunk, info)

    assert committed is True
    assert reject_info == {}
    assert filt.committed_chunk is not None
    filt.committed_chunk_index = 1
    mismatched_q = np.zeros(14, dtype=np.float32)
    mismatched_q[CTRL] = 1.0

    result = filt._serve_committed_chunk(
        {"q": mismatched_q},
        chunk,
        chunk.shape,
        q_full=mismatched_q,
    )

    assert result is None
    assert filt.committed_chunk is None
    pending = filt._pop_pending_committed_replan_info()
    assert pending["committed_aborted_due_to_state_mismatch"] is True
    assert pending["committed_replan_due_to_state_mismatch"] is True
    assert pending["committed_state_error"] > filt.committed_state_error_threshold
    assert pending["committed_state_error_threshold"] == 0.25
    assert pending["actual_q_at_replay"] is not None
    assert pending["planned_q_at_index"] is not None



def test_committed_state_mismatch_replans_recovery_suffix_from_actual_q():
    filt = _explicit_recovery_filter(
        q_rejoin_threshold=10.0,
        yield_horizon=1,
        return_horizon=3,
        explicit_recovery={
            "committed_state_error_threshold": 0.05,
            "replan_committed_suffix_on_state_mismatch": True,
            "committed_execution_margin": 0.0,
            "opportunistic_act_resume": False,
        },
        safechunk_recover={
            "enabled": True,
            "require_direction_alignment": False,
            "require_ordered_path": False,
        },
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=10.0)
    chunk = make_chunk(h=5)
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)
    filt.recovery_context = RecoveryContext(
        nominal_chunk=chunk.copy(),
        nominal_q_seq=nominal_q_seq.copy(),
        active=False,
        target_rejoin_index=3,
        phase="recover",
    )
    info = {
        "optimized_accepted": True,
        "deform_chunk_length": 1,
        "recover_chunk_length": 3,
        "recover_target_index": 3,
        "recover_min_clearance": 1.0,
    }

    committed, reject_info = filt._commit_explicit_recovery_chunk(obs, chunk, info)
    assert committed is True
    assert reject_info == {}
    filt.committed_chunk_index = 1
    mismatched_q = np.zeros(14, dtype=np.float32)
    mismatched_q[CTRL] = 0.4

    result = filt._serve_committed_chunk(
        {"q": mismatched_q},
        chunk,
        chunk.shape,
        q_full=mismatched_q,
    )

    assert result is not None
    safe_chunk, replay_info = result
    np.testing.assert_allclose(safe_chunk[:, PASS], chunk[:, PASS])
    assert replay_info["committed_state_mismatch_detected"] is True
    assert replay_info["committed_state_mismatch_recovered"] is True
    assert replay_info["committed_suffix_replan_attempted"] is True
    assert replay_info["committed_suffix_replan_accepted"] is True
    assert replay_info["committed_chunk_mode"] == "recover"
    assert replay_info["recover_steps_executed"] == 1
    assert filt.committed_suffix_replan_attempt_count == 1
    assert filt.committed_suffix_replan_accepted_count == 1
    assert filt.committed_chunk is not None
    assert filt.committed_chunk_index == 1


def test_committed_chunk_rejects_missing_or_malformed_planned_q():
    filt = _explicit_recovery_filter(q_rejoin_threshold=10.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}

    def malformed_rollout(_q, action_chunk):
        return np.zeros((action_chunk.shape[0] - 1, 14), dtype=np.float32)

    filt._rollout_chunk_from_q = malformed_rollout
    info = {
        "optimized_accepted": True,
        "deform_chunk_length": 1,
        "recover_chunk_length": 2,
        "recover_target_index": 3,
        "recover_min_clearance": 1.0,
    }

    committed, reject_info = filt._commit_explicit_recovery_chunk(obs, chunk, info)

    assert committed is False
    assert reject_info["committed_rejected_missing_planned_q"] is True
    assert filt.committed_chunk is None


def test_explicit_recovery_false_preserves_one_stage_behavior():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        opt_iters=1,
        opt_population=6,
        opt_seed=0,
        recoverable_deform={
            "enabled": True,
            "explicit_recovery": False,
            "final_rejoin_metric": "q_state",
            "q_rejoin_threshold": 10.0,
        },
        debug=False,
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    chunk = make_chunk()
    chunk[:, CTRL] = 0.1
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal_q_seq = filt.rollout_nominal_chunk(obs, chunk)

    _safe, info = filt.deform_chunk_optimized(
        nominal_chunk=chunk,
        obs=obs,
        nominal_q_seq=nominal_q_seq,
        first_violation=0,
    )

    assert filt.explicit_return is False
    assert filt.recovery_context is None
    assert "deform_stage_accepted" not in info
    assert info["deformation_source"] == "optimized_deform"



def _recover_rejoin_filter(clearances=(0.10, 0.10, 0.10, 0.10), **recover_cfg):
    filt = _acceptance_filter(clearances)
    cfg = {
        "enabled": True,
        "rejoin_nominal_weight": 5.0,
        "task_progress_weight": 10.0,
        "rejoin_weight_schedule": "constant",
    }
    cfg.update(recover_cfg)
    parsed = filt._safechunk_recover_config(cfg)
    filt.safechunk_recover_enabled = parsed["enabled"]
    filt.recover_rejoin_nominal_weight = parsed["rejoin_nominal_weight"]
    filt.recover_task_progress_weight = parsed["task_progress_weight"]
    filt.recover_ordered_pose_weight = parsed["ordered_pose_weight"]
    filt.recover_ordered_delta_weight = parsed["ordered_delta_weight"]
    filt.recover_ordered_pose_threshold = parsed["ordered_pose_threshold"]
    filt.recover_ordered_delta_threshold = parsed["ordered_delta_threshold"]
    filt.require_recover_ordered_path = parsed["require_ordered_path"]
    filt.recover_safety_weight = parsed["safety_weight"]
    filt.recover_action_deviation_weight = parsed["action_deviation_weight"]
    filt.recover_smoothness_weight = parsed["smoothness_weight"]
    filt.require_nominal_prefix_safe_for_rejoin = parsed["require_nominal_prefix_safe_for_rejoin"]
    filt.nominal_rejoin_prefix_min_clearance = parsed["nominal_rejoin_prefix_min_clearance"]
    filt.use_latest_nominal_for_rejoin = parsed["use_latest_nominal_for_rejoin"]
    filt.suppress_stale_nominal_rejoin = parsed["suppress_stale_nominal_rejoin"]
    filt.rejoin_weight_schedule = parsed["rejoin_weight_schedule"]
    filt.rejoin_ramp_steps = parsed["rejoin_ramp_steps"]
    return filt


def test_nominal_rejoin_score_positive_for_aligned_candidate():
    filt = _recover_rejoin_filter()
    nominal = np.zeros((4, 16), dtype=np.float32)
    candidate = np.zeros_like(nominal)
    nominal[0, CTRL] = 0.2
    candidate[0, CTRL] = 0.1

    info = filt.compute_nominal_rejoin_score(candidate, nominal, obs={"q": np.zeros(14, dtype=np.float32)})

    assert info["nominal_rejoin_score"] > 0.0
    assert info["recover_cosine_to_nominal"] > 0.0


def test_nominal_rejoin_score_zero_for_opposite_candidate():
    filt = _recover_rejoin_filter()
    nominal = np.zeros((4, 16), dtype=np.float32)
    candidate = np.zeros_like(nominal)
    nominal[0, CTRL] = 0.2
    candidate[0, CTRL] = -0.1

    info = filt.compute_nominal_rejoin_score(candidate, nominal, obs={"q": np.zeros(14, dtype=np.float32)})

    assert info["nominal_rejoin_score"] == 0.0
    assert info["recover_cosine_to_nominal"] < 0.0


def test_ordered_recovery_path_loss_tracks_ordered_nominal_slice():
    filt = _recover_rejoin_filter(ordered_pose_weight=2.0, ordered_delta_weight=3.0)
    nominal_q_seq = np.zeros((6, 14), dtype=np.float32)
    for k in range(nominal_q_seq.shape[0]):
        nominal_q_seq[k, CTRL] = 0.1 * k
    q_seq = nominal_q_seq[2:5].copy()

    terms = filt._ordered_recovery_path_terms(q_seq, nominal_q_seq, target_index=2)

    assert terms["recover_ordered_path_available"] is True
    assert terms["recover_ordered_target_index"] == 2
    assert terms["recover_ordered_horizon"] == 3
    assert terms["recover_ordered_pose_loss"] == 0.0
    assert terms["recover_ordered_delta_loss"] == 0.0
    assert terms["recover_ordered_loss"] == 0.0
    assert terms["recover_ordered_ok"] is True

    deviated = q_seq.copy()
    deviated[1, CTRL] += 0.25
    bad_terms = filt._ordered_recovery_path_terms(deviated, nominal_q_seq, target_index=2)

    assert bad_terms["recover_ordered_pose_loss"] > 0.0
    assert bad_terms["recover_ordered_delta_loss"] > 0.0
    assert bad_terms["recover_ordered_loss"] > bad_terms["recover_ordered_pose_loss"]
    assert bad_terms["recover_ordered_ok"] is False


def test_ordered_recovery_path_reject_reason_is_unrecoverable():
    filt = _recover_rejoin_filter()
    reason = filt._recovery_reject_reason(
        {"q_rejoin_ok": True, "qd_rejoin_ok": True},
        {"immediate_safe": True, "prefix_safe": True, "path_safe": True},
        direction_ok=True,
        ordered_ok=False,
    )

    assert reason == "ordered_path_failed"


def test_return_deformation_cost_adds_ordered_recovery_loss():
    filt = _explicit_recovery_filter(
        safechunk_recover={
            "ordered_pose_weight": 2.0,
            "ordered_delta_weight": 3.0,
        }
    )
    filt.evaluate_horizon_safety = _controlled_limit_safety(filt, limit=2.0)
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal = make_chunk(h=5)
    for k in range(nominal.shape[0]):
        nominal[k, CTRL] = 0.1 * k
    candidate = nominal.copy()
    candidate[:, CTRL] = 0.0
    nominal_q_seq = filt.rollout_nominal_chunk(obs, nominal)
    rejoin_context = filt._make_rejoin_context(nominal_q_seq)

    cost, losses = filt._return_deformation_cost(
        obs,
        candidate,
        nominal,
        nominal_q_seq,
        rejoin_context,
        CTRL,
    )

    assert losses["recover_ordered_path_available"] is True
    assert losses["recover_ordered_pose_loss"] > 0.0
    assert losses["recover_ordered_loss"] > 0.0
    expected = (
        filt.lambda_return_rejoin * losses["rejoin_loss"]
        + filt.lambda_return_safety * losses["safety_loss"]
        + filt.lambda_return_smooth * losses["smoothness_loss"]
        + filt.lambda_return_action * losses["action_deviation_loss"]
        + losses["recover_ordered_loss"]
    )
    np.testing.assert_allclose(cost, expected)


def test_batched_q_rejoin_indices_are_absolute_future_indices():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        recoverable_deform={"enabled": True, "inner_rejoin_metric": "q_state"},
        debug=False,
    )
    nominal_q_seq = np.zeros((6, 14), dtype=np.float32)
    for k in range(nominal_q_seq.shape[0]):
        nominal_q_seq[k, CTRL] = float(k)
    q_seq_batch = np.zeros((1, 3, 14), dtype=np.float32)
    q_seq_batch[0, -1, CTRL] = nominal_q_seq[3, CTRL]

    _losses, indices, _time_ms = filt._q_rejoin_loss_batch(
        q_seq_batch,
        nominal_q_seq=nominal_q_seq,
    )

    assert indices == [3]


def test_stale_nominal_rejoin_target_is_suppressed():
    filt = _recover_rejoin_filter()
    chunk = make_chunk(h=4)
    filt.latest_nominal_chunk = chunk.copy()
    filt.blocked_nominal_chunk = chunk.copy()
    filt.latest_nominal_step = 3
    filt.blocked_nominal_step = 3

    info = filt.get_nominal_rejoin_target({"q": np.zeros(14, dtype=np.float32)}, chunk)

    assert info["available"] is False
    assert info["suppressed_reason"] == "stale_blocked_nominal"


def test_unsafe_nominal_prefix_rejoin_target_is_suppressed():
    filt = _recover_rejoin_filter(clearances=(0.03, 0.10, 0.10, 0.10))
    chunk = make_chunk(h=4)
    filt.latest_nominal_chunk = chunk.copy()
    filt.latest_nominal_step = 1

    info = filt.get_nominal_rejoin_target({"q": np.zeros(14, dtype=np.float32)}, chunk)

    assert info["available"] is False
    assert info["suppressed_reason"] == "nominal_prefix_unsafe"


def test_recover_candidate_score_increases_with_nominal_rejoin_weight():
    obs = {"q": np.zeros(14, dtype=np.float32)}
    nominal = np.zeros((4, 16), dtype=np.float32)
    candidate = np.zeros_like(nominal)
    nominal[0, CTRL] = 0.2
    candidate[0, CTRL] = 0.1
    acceptance = {
        "accepted": True,
        "safe_prefix_len": 4,
        "immediate_clearance": 0.1,
    }
    filt0 = _recover_rejoin_filter(rejoin_nominal_weight=0.0)
    filt0.latest_nominal_chunk = nominal.copy()
    filt0.latest_nominal_step = 1
    score0, _ = filt0._score_accepted_candidate(
        obs,
        candidate,
        nominal,
        acceptance,
        candidate_type="recover",
    )
    filt1 = _recover_rejoin_filter(rejoin_nominal_weight=5.0)
    filt1.latest_nominal_chunk = nominal.copy()
    filt1.latest_nominal_step = 1
    score1, info1 = filt1._score_accepted_candidate(
        obs,
        candidate,
        nominal,
        acceptance,
        candidate_type="recover",
    )

    assert info1["nominal_rejoin_score"] > 0.0
    assert score1 > score0


def test_immediate_hard_reject_overrides_high_rejoin_score():
    filt = _recover_rejoin_filter(clearances=(0.01, 0.10, 0.10, 0.10), rejoin_nominal_weight=1000.0)
    chunk = make_chunk(h=4)
    chunk[0, CTRL] = 1.0
    filt.latest_nominal_chunk = chunk.copy()
    filt.latest_nominal_step = 1

    acceptance = filt.evaluate_candidate_acceptance(
        {"q": np.zeros(14, dtype=np.float32)},
        chunk,
        "recover",
    )

    assert acceptance["accepted"] is False
    assert acceptance["rejection_reason"] == "immediate_below_hard_margin"



def _active_safety_filter_for_hold_tests():
    filt = SafeChunkDeformFilter(
        mode="optimized",
        safechunk_active_safety={
            "enabled": True,
            "hard_min_clearance": 0.02,
            "hold_prefix_min_clearance": 0.04,
            "hold_horizon_steps": 3,
            "emergency_deform_when_hold_unsafe": True,
            "optimize_when_hold_unsafe": False,
        },
        debug=False,
    )
    filt.evaluate_horizon_safety = lambda _obs, q_seq: {
        "horizon_safe": True,
        "min_clearance": 1.0,
        "min_clearances": np.ones(q_seq.shape[0], dtype=np.float32),
        "first_violation": None,
        "unsafe_count": 0,
        "safety_eval_available": True,
    }
    return filt


def _hold_acceptance(clearances):
    clearances = np.asarray(clearances, dtype=np.float32)

    def fake(_obs, candidate, candidate_type="deform"):
        h = clearances[: np.asarray(candidate).reshape((-1, np.asarray(candidate).shape[-1])).shape[0]]
        if h.size == 0:
            h = clearances
        if h.shape[0] < np.asarray(candidate).reshape((-1, np.asarray(candidate).shape[-1])).shape[0]:
            h = np.pad(h, (0, np.asarray(candidate).reshape((-1, np.asarray(candidate).shape[-1])).shape[0] - h.shape[0]), mode="edge")
        safe_prefix_len = 0
        for value in h:
            if float(value) >= 0.04:
                safe_prefix_len += 1
            else:
                break
        return {
            "accepted": bool(np.min(h) >= 0.02 and h[0] >= 0.04),
            "acceptance_type": "full_horizon" if np.min(h) >= 0.08 else "safe_prefix",
            "safe_prefix_len": safe_prefix_len,
            "immediate_clearance": float(h[0]),
            "prefix_min_clearance": 0.04,
            "horizon_min_clearance": float(np.min(h)),
            "desired_min_clearance": 0.08,
            "hard_min_clearance": 0.02,
            "rejection_reason": None if np.min(h) >= 0.02 else "hold_predicted_contact",
            "candidate_type": candidate_type,
            "full_horizon_required": False,
            "rolling_replan_on_prefix": True,
            "safe_prefix_execution": False,
            "horizon_safe": bool(np.min(h) >= 0.08),
        }

    return fake


def test_hold_brake_accepted_when_predicted_clearance_safe():
    filt = _active_safety_filter_for_hold_tests()
    filt.evaluate_candidate_acceptance = _hold_acceptance([0.06, 0.05, 0.05])
    chunk = np.zeros((3, 16), dtype=np.float32)
    info = {"safety_mode": "horizon_brake", "mode": "horizon_brake"}

    safe, out = filt._hold_return_or_emergency_deform(
        {"q": np.zeros(14, dtype=np.float32)},
        chunk,
        chunk,
        info,
        chunk.shape,
    )

    np.testing.assert_allclose(safe, chunk)
    assert out["safety_mode"] == "horizon_brake"
    assert out["hold_acceptance_type"] == "hold_or_brake"
    assert out.get("emergency_deform_away") is not True


def test_hold_rejected_when_predicted_human_sweep_violates_hard_margin():
    filt = _active_safety_filter_for_hold_tests()
    filt.evaluate_candidate_acceptance = _hold_acceptance([0.05, 0.03, 0.015])
    chunk = np.zeros((3, 16), dtype=np.float32)
    info = {"safety_mode": "horizon_brake", "mode": "horizon_brake"}

    _safe, out = filt._hold_return_or_emergency_deform(
        {"q": np.zeros(14, dtype=np.float32)},
        chunk,
        chunk,
        info,
        chunk.shape,
    )

    assert out["safety_mode"] == "emergency_deform_away"
    assert out["hold_predicted_contact"] is True
    assert out["hold_rejected_reason"] == "hold_predicted_contact"
    assert out["emergency_deform_away"] is True


def test_zero_action_hold_is_not_automatically_safe():
    filt = _active_safety_filter_for_hold_tests()
    filt.evaluate_candidate_acceptance = _hold_acceptance([0.05, 0.03, 0.015])
    zero_hold = np.zeros((3, 16), dtype=np.float32)

    info = filt.evaluate_hold_or_brake_acceptance(
        {"q": np.zeros(14, dtype=np.float32)},
        zero_hold,
    )

    assert info["accepted"] is False
    assert info["hold_predicted_contact"] is True


def test_emergency_deform_away_chooses_safe_last_safe_action_over_unsafe_hold():
    filt = _active_safety_filter_for_hold_tests()
    hold = np.zeros((3, 16), dtype=np.float32)
    hold[:, PASS] = np.arange(PASS.size, dtype=np.float32)
    last_safe = np.zeros(16, dtype=np.float32)
    last_safe[CTRL] = 0.25
    last_safe[PASS] = 9.0
    filt.last_safe_action = last_safe.copy()

    def fake_acceptance(_obs, candidate, candidate_type="deform"):
        cand = np.asarray(candidate, dtype=np.float32).reshape((-1, 16))
        if np.allclose(cand[0, CTRL], last_safe[CTRL]):
            h = np.asarray([0.09, 0.09, 0.09], dtype=np.float32)
        else:
            h = np.asarray([0.05, 0.03, 0.015], dtype=np.float32)
        safe_prefix_len = int(np.sum(h >= 0.04)) if np.all(h >= 0.04) else 1
        return {
            "accepted": bool(np.min(h) >= 0.02 and h[0] >= 0.04),
            "acceptance_type": "full_horizon",
            "safe_prefix_len": safe_prefix_len,
            "immediate_clearance": float(h[0]),
            "prefix_min_clearance": 0.04,
            "horizon_min_clearance": float(np.min(h)),
            "desired_min_clearance": 0.08,
            "hard_min_clearance": 0.02,
            "rejection_reason": None if np.min(h) >= 0.02 else "hold_predicted_contact",
            "candidate_type": candidate_type,
            "full_horizon_required": False,
            "rolling_replan_on_prefix": True,
            "safe_prefix_execution": False,
            "horizon_safe": bool(np.min(h) >= 0.08),
        }

    filt.evaluate_candidate_acceptance = fake_acceptance

    safe, info = filt.emergency_deform_away(
        {"q": np.zeros(14, dtype=np.float32)},
        hold,
        nominal_chunk=hold,
    )

    assert info["safety_mode"] == "emergency_deform_away"
    assert info["accepted_candidate_name"] == "last_safe_action"
    np.testing.assert_allclose(safe[0, CTRL], last_safe[CTRL])
    np.testing.assert_allclose(safe[:, PASS], hold[:, PASS])


def test_emergency_deform_away_logs_do_not_use_yield_return_terms():
    filt = _active_safety_filter_for_hold_tests()
    filt.evaluate_candidate_acceptance = _hold_acceptance([0.05, 0.03, 0.015])
    hold = np.zeros((3, 16), dtype=np.float32)

    _safe, info = filt.emergency_deform_away(
        {"q": np.zeros(14, dtype=np.float32)},
        hold,
        nominal_chunk=hold,
    )

    assert not any("yield" in key or "return" in key for key in info)
    assert info["safety_mode"] == "emergency_deform_away"
