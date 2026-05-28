import numpy as np

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

q[filt.bigym_state_arm_indices] = np.array(
    [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4],
    dtype=np.float32,
)

q_urdf, qd_urdf, q_arm_bigym, q_arm_urdf = (
    filt._build_urdf_surrogate_state_from_bigym(q, qd)
)

print("robot_model.num_joints:", filt.robot_model.num_joints)
print("robot_model.num_controls:", filt.robot_model.num_controls)
print("q_urdf shape:", q_urdf.shape)
print("qd_urdf shape:", qd_urdf.shape)
print("urdf_arm_joint_indices:", filt.urdf_arm_joint_indices)
print("q_arm_bigym:", q_arm_bigym)
print("q_arm_urdf:", q_arm_urdf)
print("q_urdf arm:", q_urdf[filt.urdf_arm_joint_indices])

non_arm_mask = np.ones(filt.robot_model.num_joints, dtype=bool)
non_arm_mask[filt.urdf_arm_joint_indices] = False

print("non-arm neutral:", np.allclose(q_urdf[non_arm_mask], filt.urdf_neutral_q[non_arm_mask]))
print("arm inserted correctly:", np.allclose(q_urdf[filt.urdf_arm_joint_indices], q_arm_urdf))

assert q_urdf.shape[0] == filt.robot_model.num_joints
assert qd_urdf.shape[0] == filt.robot_model.num_joints
assert np.allclose(q_urdf[filt.urdf_arm_joint_indices], q_arm_urdf)
assert np.allclose(q_urdf[non_arm_mask], filt.urdf_neutral_q[non_arm_mask])

print("PASS: fixed-torso URDF surrogate test")