import logging
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter

logging.basicConfig(level=logging.DEBUG)

URDF_PATH = "/home/xd1125/Workspace/safe_bigym_hoi/external/oscbf/oscbf/assets/h1/h1.urdf"

def main():
    filt = OSCBFFilter(
        urdf_path=URDF_PATH,
        debug=True,
        use_dummy_filter=False,
        expected_motion_dim=14,
        dt=0.05,
        control_type="absolute",
        filter_all_except_gripper=True,
    )
    if filt.robot_model.num_joints != filt.expected_motion_dim:
        raise ValueError(
            f"Current H1 upper-body OSCBF expects robot_model.num_joints "
            f"to equal expected_motion_dim. Got robot_model.num_joints="
            f"{filt.robot_model.num_joints}, expected_motion_dim="
            f"{filt.expected_motion_dim}. If using a full-body H1 URDF, "
            f"enable controlled_joint_indices/subset mode."
        )
    print("robot_model:", type(filt.robot_model).__name__)
    print("oscbf_config:", type(filt.oscbf_config).__name__)
    print("cbf:", type(filt.cbf).__name__)
    print("[PASS] real OSCBF init works")

if __name__ == "__main__":
    main()