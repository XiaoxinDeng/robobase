from oscbf.core.treemanipulator import TreeManipulator

URDF_PATH = "/home/xd1125/Workspace/safe_bigym_hoi/external/oscbf/oscbf/assets/h1/h1.urdf"

robot_model = TreeManipulator.from_urdf(
    urdf_filename=URDF_PATH,
    ee_joint_idx=0,
    controlled_joint_indices=None,
)

print("num_joints:", robot_model.num_joints)
print("joint_names:")
for i, name in enumerate(robot_model.joint_names):
    print(i, name)