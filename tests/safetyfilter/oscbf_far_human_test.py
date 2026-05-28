import numpy as np

from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter

URDF_PATH = "/home/xd1125/Workspace/safe_bigym_hoi/external/oscbf/oscbf/assets/h1/h1.urdf"

filt = OSCBFFilter(
    urdf_path=URDF_PATH,
    use_dummy_filter=False,
    debug=True,
    control_type="absolute",
)

def fake_far_human(env=None, observations=None):
    # q_bigym is zero in this test, and T_world_urdf is identity,
    # so these world-frame capsule positions are also effectively URDF-frame far-away points.
    capsule_a = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.5],
        ],
        dtype=np.float32,
    )
    capsule_b = np.array(
        [
            [10.0, 10.0, 10.3],
            [10.0, 10.0, 10.8],
        ],
        dtype=np.float32,
    )
    capsule_radii = np.array([0.1, 0.1], dtype=np.float32)

    return {
        "human_arm_qpos": np.zeros(1, dtype=np.float32),
        "human_arm_qvel": np.zeros(1, dtype=np.float32),
        "capsule_a": capsule_a,
        "capsule_b": capsule_b,
        "capsule_radii": capsule_radii,
    }

filt._extract_human_obstacles = fake_far_human

action = np.zeros(16, dtype=np.float32)
q = np.zeros(14, dtype=np.float32)
qd = np.zeros(14, dtype=np.float32)

action[filt.bigym_action_arm_indices] = np.array(
    [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05],
    dtype=np.float32,
)

safe = filt(
    action=action,
    q_full=q,
    qd_full=qd,
)

arm_idx = filt.bigym_action_arm_indices
non_arm_idx = np.array(
    [i for i in range(action.shape[0]) if i not in set(arm_idx.tolist())],
    dtype=np.int64,
)

arm_delta = np.linalg.norm(safe[arm_idx] - action[arm_idx])
non_arm_unchanged = np.allclose(safe[non_arm_idx], action[non_arm_idx])

print("nominal action:", action)
print("safe action:", safe)
print("nominal arm:", action[arm_idx])
print("safe arm:", safe[arm_idx])
print("arm delta:", arm_delta)
print("non-arm unchanged:", non_arm_unchanged)

assert safe.shape == action.shape
assert non_arm_unchanged

# For far obstacle, the filter should usually do almost nothing.
# Keep this tolerance loose first; tighten after seeing actual QP behaviour.
assert arm_delta < 1e-3, f"Far human changed action too much: {arm_delta}"

print("PASS: far-human OSCBF test")