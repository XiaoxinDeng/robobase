import logging
import numpy as np
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter


logging.basicConfig(level=logging.DEBUG)


URDF_PATH = "/home/xd1125/Workspace/safe_bigym_hoi/external/oscbf/oscbf/assets/h1/h1.urdf"

filt = OSCBFFilter(
    urdf_path=URDF_PATH,
    use_dummy_filter=True,
    dummy_scale=0.5,
    debug=True,
    control_type="absolute",
)

action = np.zeros(16, dtype=np.float32)
q = np.zeros(14, dtype=np.float32)
qd = np.zeros(14, dtype=np.float32)

# Make the nominal arm action different from current arm q.
action[filt.bigym_action_arm_indices] = np.array(
    [0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5],
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

expected_arm = q[filt.bigym_state_arm_indices] + 0.5 * (
    action[arm_idx] - q[filt.bigym_state_arm_indices]
)

print("action:", action)
print("safe:", safe)
print("nominal arm:", action[arm_idx])
print("safe arm:", safe[arm_idx])
print("expected arm:", expected_arm)
print("non-arm unchanged:", np.allclose(safe[non_arm_idx], action[non_arm_idx]))
print("arm correct:", np.allclose(safe[arm_idx], expected_arm))

assert safe.shape == action.shape
assert np.allclose(safe[non_arm_idx], action[non_arm_idx])
assert np.allclose(safe[arm_idx], expected_arm)

print("PASS: dummy slicing test")