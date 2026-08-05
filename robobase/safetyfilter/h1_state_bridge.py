from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import mujoco

# This must match the runtime BiGym MuJoCo joint order used for the current H1 embodiment.
TREE_JOINT_NAMES: Tuple[str, ...] = (
    "h1/pelvis_x",
    "h1/pelvis_y",
    "h1/pelvis_z",
    "h1/pelvis_rz",
    "h1/left_shoulder_pitch",
    "h1/left_shoulder_roll",
    "h1/left_shoulder_yaw",
    "h1/left_elbow",
    "h1/left_wrist",
    "h1/right_shoulder_pitch",
    "h1/right_shoulder_roll",
    "h1/right_shoulder_yaw",
    "h1/right_elbow",
    "h1/right_wrist",
)

# Gripper joints are excluded for now because the current safety-filter test
# only targets pelvis + upper-body arm joints. Add them back later if needed.
LEFT_GRIPPER_JOINT_NAMES: Tuple[str, ...] = (
    'h1/robotiq_2f85_left/right_driver_joint',
    'h1/robotiq_2f85_left/right_coupler_joint',
    'h1/robotiq_2f85_left/right_spring_link_joint',
    'h1/robotiq_2f85_left/right_follower_joint',
    'h1/robotiq_2f85_left/left_driver_joint',
    'h1/robotiq_2f85_left/left_coupler_joint',
    'h1/robotiq_2f85_left/left_spring_link_joint',
    'h1/robotiq_2f85_left/left_follower_joint',
)
RIGHT_GRIPPER_JOINT_NAMES: Tuple[str, ...] = (
    'h1/robotiq_2f85_right/right_driver_joint',
    'h1/robotiq_2f85_right/right_coupler_joint',
    'h1/robotiq_2f85_right/right_spring_link_joint',
    'h1/robotiq_2f85_right/right_follower_joint',
    'h1/robotiq_2f85_right/left_driver_joint',
    'h1/robotiq_2f85_right/left_coupler_joint',
    'h1/robotiq_2f85_right/left_spring_link_joint',
    'h1/robotiq_2f85_right/left_follower_joint',
)

# Two arms controlled
# CONTROLLED_JOINT_INDICES: Tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

# All joints controlled (including pelvis)
CONTROLLED_JOINT_INDICES: Tuple[int, ...] = tuple(range(14))

@dataclass
class H1State:
    q_full: np.ndarray
    qd_full: np.ndarray
    q_ctrl: np.ndarray
    qd_ctrl: np.ndarray

def get_bigym_task(env):
    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))

        if hasattr(cur, "_mojo"):
            return cur
        if hasattr(cur, "unwrapped") and hasattr(cur.unwrapped, "_mojo"):
            return cur.unwrapped

        cur = getattr(cur, "env", None)

    raise AttributeError("Could not find base BiGym task object with '_mojo'")

def get_bigym_mojo(env):
    task = get_bigym_task(env)
    if not hasattr(task, "_mojo"):
        raise AttributeError(f"{type(task).__name__} has no attribute '_mojo'")
    return task._mojo

def get_mujoco_joint_names(env) -> List[str]:
    mojo = get_bigym_mojo(env)
    model = mojo.model

    joint_names = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
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
    mojo = get_bigym_mojo(env)
    model = mojo.model
    data = mojo.data

    tree_to_mj = build_tree_to_mujoco_index_map(env, tree_joint_names)

    q = np.zeros(len(tree_joint_names), dtype=np.float64)
    qd = np.zeros(len(tree_joint_names), dtype=np.float64)

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
    print_diagnostics: bool = False,
) -> H1State:
    q_full, qd_full = extract_h1_q_qd(env, tree_joint_names)

    ctrl_idx = np.asarray(controlled_joint_indices, dtype=int)
    q_ctrl = q_full[ctrl_idx]
    qd_ctrl = qd_full[ctrl_idx]

    if print_diagnostics:
        print("[safetyfilter] q_full shape:", q_full.shape)
        print("[safetyfilter] qd_full shape:", qd_full.shape)
        print("[safetyfilter] q_ctrl shape:", q_ctrl.shape)
        print("[safetyfilter] qd_ctrl shape:", qd_ctrl.shape)
        print("[safetyfilter] q_full:", q_full)

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

def debug_print_mujoco_joint_names(env):
    names = get_mujoco_joint_names(env)
    print("[safetyfilter] MuJoCo joint names:")
    for i, name in enumerate(names):
        print(f"{i:2d}: {name}")
