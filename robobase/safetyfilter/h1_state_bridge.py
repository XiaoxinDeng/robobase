from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# This must match the validated TreeManipulator / URDF order exactly
TREE_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

CONTROLLED_JOINT_INDICES: Tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16, 17, 18)


@dataclass
class H1State:
    q_full: np.ndarray
    qd_full: np.ndarray
    q_ctrl: np.ndarray
    qd_ctrl: np.ndarray


def get_mujoco_joint_names(env) -> List[str]:
    """
    Return MuJoCo joint names in the order used by qpos/qvel access for the H1 articulated joints.

    You may need to adapt this depending on how BiGym exposes the underlying physics object.
    """
    physics = env.unwrapped._physics
    model = physics.model

    joint_names = []
    for j in range(model.njnt):
        name = model.id2name(j, "joint")
        if name is not None:
            joint_names.append(name)
    return joint_names


def build_tree_to_mujoco_index_map(env, tree_joint_names: Sequence[str] = TREE_JOINT_NAMES) -> Dict[str, int]:
    """
    Build a name-based mapping from TreeManipulator joint names to MuJoCo joint indices.
    """
    mujoco_joint_names = get_mujoco_joint_names(env)
    mujoco_name_to_idx = {name: i for i, name in enumerate(mujoco_joint_names)}

    missing = [name for name in tree_joint_names if name not in mujoco_name_to_idx]
    if missing:
        raise KeyError(
            "These TreeManipulator joints were not found in the MuJoCo model: "
            + ", ".join(missing)
        )

    return {name: mujoco_name_to_idx[name] for name in tree_joint_names}


def extract_h1_q_qd(env, tree_joint_names: Sequence[str] = TREE_JOINT_NAMES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract full H1 q and qd in TreeManipulator order.

    Assumes each listed joint is 1-DoF and corresponds to one MuJoCo joint entry.
    """
    physics = env.unwrapped._physics
    data = physics.data

    tree_to_mj = build_tree_to_mujoco_index_map(env, tree_joint_names)

    q = np.zeros(len(tree_joint_names), dtype=np.float64)
    qd = np.zeros(len(tree_joint_names), dtype=np.float64)

    # IMPORTANT:
    # This assumes the MuJoCo qpos/qvel indexing for these joints is aligned with joint ids
    # for 1-DoF hinge joints. If BiGym uses free joints or more complex indexing,
    # replace this with explicit jnt_qposadr / jnt_dofadr lookup.
    model = physics.model
    for i, joint_name in enumerate(tree_joint_names):
        mj_joint_id = tree_to_mj[joint_name]
        qpos_adr = model.jnt_qposadr[mj_joint_id]
        dof_adr = model.jnt_dofadr[mj_joint_id]

        q[i] = data.qpos[qpos_adr]
        qd[i] = data.qvel[dof_adr]

    return q, qd


def extract_h1_state(
    env,
    tree_joint_names: Sequence[str] = TREE_JOINT_NAMES,
    controlled_joint_indices: Sequence[int] = CONTROLLED_JOINT_INDICES,
) -> H1State:
    q_full, qd_full = extract_h1_q_qd(env, tree_joint_names)

    ctrl_idx = np.asarray(controlled_joint_indices, dtype=int)
    q_ctrl = q_full[ctrl_idx]
    qd_ctrl = qd_full[ctrl_idx]

    return H1State(
        q_full=q_full,
        qd_full=qd_full,
        q_ctrl=q_ctrl,
        qd_ctrl=qd_ctrl,
    )


def print_joint_name_alignment(env, tree_joint_names: Sequence[str] = TREE_JOINT_NAMES) -> None:
    """
    Debug helper to verify MuJoCo and TreeManipulator naming alignment.
    """
    mujoco_joint_names = get_mujoco_joint_names(env)
    print("MuJoCo joints:")
    for i, name in enumerate(mujoco_joint_names):
        print(f"{i:2d}  {name}")

    print("\nTreeManipulator joints:")
    for i, name in enumerate(tree_joint_names):
        print(f"{i:2d}  {name}")

    print("\nTree -> MuJoCo mapping:")
    mapping = build_tree_to_mujoco_index_map(env, tree_joint_names)
    for i, name in enumerate(tree_joint_names):
        print(f"{i:2d}  {name:30s} -> mj joint {mapping[name]}")