import numpy as np
import jax.numpy as jnp

from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter

URDF_PATH = "/home/xd1125/Workspace/safe_bigym_hoi/external/oscbf/oscbf/assets/h1/h1.urdf"

filt = OSCBFFilter(
    urdf_path=URDF_PATH,
    use_dummy_filter=False,
    debug=True,
    control_type="absolute",
)

q = np.zeros(14, dtype=np.float32)
qd = np.zeros(14, dtype=np.float32)

q_urdf, qd_urdf, q_arm_bigym, q_arm_urdf = (
    filt._build_urdf_surrogate_state_from_bigym(q, qd)
)

ee = np.asarray(
    filt.robot_model.ee_position(jnp.asarray(q_urdf, dtype=jnp.float32)),
    dtype=np.float32,
).reshape(3)

print("URDF protected EE/right elbow point:", ee)

def fake_near_human(env=None, observations=None):
    capsule_a = np.array(
        [
            ee + np.array([0.00, 0.00, 0.00], dtype=np.float32),
            ee + np.array([0.02, 0.00, 0.00], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    capsule_b = np.array(
        [
            ee + np.array([0.00, 0.00, 0.20], dtype=np.float32),
            ee + np.array([0.02, 0.00, 0.20], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    capsule_radii = np.array([0.15, 0.15], dtype=np.float32)

    return {
        "human_arm_qpos": np.zeros(1, dtype=np.float32),
        "human_arm_qvel": np.zeros(1, dtype=np.float32),
        "capsule_a": capsule_a,
        "capsule_b": capsule_b,
        "capsule_radii": capsule_radii,
    }

filt._extract_human_obstacles = fake_near_human

action = np.zeros(16, dtype=np.float32)
action[filt.bigym_action_arm_indices] = np.array(
    [0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2],
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

print("nominal arm:", action[arm_idx])
print("safe arm:", safe[arm_idx])
print("arm delta:", arm_delta)
print("non-arm unchanged:", non_arm_unchanged)

assert safe.shape == action.shape
assert non_arm_unchanged
assert arm_delta > 1e-6, f"Near human did not change action: {arm_delta}"

print("PASS: near-human OSCBF intervention test")