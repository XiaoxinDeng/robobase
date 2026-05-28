import numpy as np

from robobase.safetyfilter.safechunk_deform_filter import SafeChunkDeformFilter


CTRL = np.asarray([4, 5, 6, 7, 9, 10, 11, 12])
PASS = np.asarray([i for i in range(16) if i not in set(CTRL.tolist())])


class FakeOSCBF:
    def __call__(self, action, obs=None, **kwargs):
        out = np.array(action, copy=True)
        out[CTRL] *= 0.5
        return out


def make_chunk(h=16):
    return np.arange(h * 16, dtype=np.float32).reshape(h, 16) / 100.0


def unsafe_filter(first_violation, safe=False):
    filt = SafeChunkDeformFilter(oscbf_operator=FakeOSCBF(), debug=False)

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


def test_non_controlled_dimensions_are_preserved_by_deformation():
    filt = unsafe_filter(first_violation=1, safe=False)
    filt.brake_progress_threshold = 1.0
    chunk = make_chunk()

    safe, info = filt.filter_chunk({"q": np.zeros(14)}, chunk)

    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])
    assert info["safety_mode"] == "horizon_deform"


def test_braking_holds_from_previous_safe_index():
    filt = unsafe_filter(first_violation=5, safe=False)
    chunk = make_chunk()

    braked, info = filt.path_consistent_brake(
        {"q": np.zeros(14)},
        chunk,
        {"first_violation": 5},
    )

    assert info["brake_stop_idx"] == 4
    np.testing.assert_allclose(braked[:4], chunk[:4])
    np.testing.assert_allclose(braked[4:], np.repeat(chunk[4:5], 12, axis=0))


def test_deadlock_uses_deformation_path():
    filt = unsafe_filter(first_violation=1, safe=False)
    filt.brake_progress_threshold = 0.5
    chunk = make_chunk()

    safe, info = filt.filter_chunk({"q": np.zeros(14)}, chunk)

    assert info["deadlock"] is True
    assert info["safety_mode"] == "horizon_deform"
    assert safe.shape == chunk.shape


def test_fake_oscbf_changes_only_controlled_dimensions_and_reports_norm():
    filt = SafeChunkDeformFilter(oscbf_operator=FakeOSCBF(), debug=False)
    chunk = make_chunk()

    safe, info = filt.deform_chunk_with_oscbf({"q": np.zeros(14)}, chunk)

    np.testing.assert_allclose(safe[:, CTRL], chunk[:, CTRL] * 0.5)
    np.testing.assert_allclose(safe[:, PASS], chunk[:, PASS])
    assert info["deformation_norm"] > 0.0


def test_existing_oscbf_import_is_not_broken():
    from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter

    assert OSCBFFilter is not None
