from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import mujoco
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from cbfpy import CBF
from oscbf.core.treemanipulator import TreeManipulator
from robobase.safetyfilter.h1_state_bridge import TREE_JOINT_NAMES, get_bigym_task
from robobase.safetyfilter.oscbf.oscbf_eehumancapsule_velocity_config import (
    OSCBFEEHumanCapsuleVelocityConfig,
    OSCBFPelvisArmHumanCapsuleVelocityConfig,
)
import logging
logger = logging.getLogger(__name__)

class OSCBFFilter:
    """
    Fused RoboBase/BiGym OSCBF safety filter.

    Input:
        full ACT/env action, e.g. shape (16,)

    Internally:
        motion action: action[:expected_motion_dim]
        passthrough action: action[expected_motion_dim:]

    Output:
        full safe action with same shape as input.
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        debug: bool = True,
        use_dummy_filter: bool = False,
        dummy_scale: float = 0.5,
        runtime_joint_names: Optional[Sequence[str]] = None,
        expected_motion_dim: int = 14,
        dt: float = 1.0 / 20.0,
        human_margin: float = 0.08,
        alpha_gain: float = 10.0,
        control_type: str = "absolute",
        filter_all_except_gripper: bool = True,
        max_action_delta: Optional[float] = None,
        enabled: bool = True,
        build_cbf_eagerly: bool = False,
        enable_pelvis_cbf: bool = True,
        pelvis_velocity_limits: Sequence[float] = (0.6, 0.6, 0.4, 1.5),
        pelvis_cbf_weight: float = 0.5,
        arm_cbf_weight: float = 1.0,
        **kwargs,
    ):
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        if kwargs:
            logger.warning("Unused OSCBFFilter kwargs: %s", kwargs)
        
        self.enabled = bool(enabled)
        if self.enabled and not use_dummy_filter:
            assert urdf_path is not None, "URDF path is required when OSCBF is enabled"
            assert Path(urdf_path).is_file(), f"URDF not found: {urdf_path}"

        if control_type not in ["absolute", "velocity", "delta"]:
            raise ValueError(
                f"control_type must be one of ['absolute', 'velocity', 'delta'], "
                f"got {control_type}"
            )

        self.urdf_path = Path(urdf_path) if urdf_path is not None else None
        self.debug = debug
        self.use_dummy_filter = use_dummy_filter
        self.dummy_scale = float(dummy_scale)
        
        self.runtime_joint_names = (
            tuple(runtime_joint_names)
            if runtime_joint_names is not None
            else TREE_JOINT_NAMES
        )

        self.expected_motion_dim = int(expected_motion_dim)
        self.dt = float(dt)
        self.human_margin = float(human_margin)
        self.alpha_gain = float(alpha_gain)
        self.control_type = control_type
        self.filter_all_except_gripper = bool(filter_all_except_gripper)
        self.max_action_delta = max_action_delta
        self.build_cbf_eagerly = bool(build_cbf_eagerly)
        self.enable_pelvis_cbf = bool(enable_pelvis_cbf)
        self.pelvis_velocity_limits = tuple(float(x) for x in pelvis_velocity_limits)
        self.pelvis_cbf_weight = float(pelvis_cbf_weight)
        self.arm_cbf_weight = float(arm_cbf_weight)
        self._warned_pelvis_cbf_fallback = False

        self._printed_frame_debug = False
        self._human_capsule_kinematic_cache = None
        self._human_capsule_fast_cache_hits = 0
        self._human_capsule_fast_cache_misses = 0
        self._human_capsule_fast_cache_failures = 0
        self._live_h_fn = None


        # Indices in the 16D BiGym / RoboBase action. The CBF model itself
        # still controls the 8 arm joints; chunk-level deformation can also
        # edit the floating pelvis so the whole kinematic chain can yield.
        self.bigym_action_base_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
        self.bigym_action_arm_indices = np.asarray(
            [4, 5, 6, 7, 9, 10, 11, 12],
            dtype=np.int64,
        )
        self.bigym_action_safety_indices = np.concatenate(
            [self.bigym_action_base_indices, self.bigym_action_arm_indices]
        )

        # Indices in the 14D BiGym / RoboBase robot state.
        self.bigym_state_base_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
        self.bigym_state_arm_indices = np.asarray(
            [4, 5, 6, 7, 9, 10, 11, 12],
            dtype=np.int64,
        )
        self.bigym_state_safety_indices = np.concatenate(
            [self.bigym_state_base_indices, self.bigym_state_arm_indices]
        )

        # Backward-compatible aliases for the true 8D OSCBF control model.
        self.oscbf_action_indices = self.bigym_action_arm_indices
        self.oscbf_state_indices = self.bigym_state_arm_indices

        # BiGym arm joint -> URDF arm joint mapping.
        # Start with identity signs and zero offsets.
        # Later calibrate these joint-by-joint if directions/zero poses differ.
        self.arm_sign = np.ones(8, dtype=np.float32)
        self.arm_offset = np.zeros(8, dtype=np.float32)

        # Fixed calibration from BiGym pelvis frame to URDF surrogate base frame.
        # These are initial guesses. Tune them so that URDF shoulders align with BiGym shoulders.
        self.pelvis_to_urdf_translation = np.asarray(
            [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.pelvis_to_urdf_rpy = np.asarray(
            [0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        self.T_pelvis_urdf = self._make_transform_from_xyz_rpy(
            self.pelvis_to_urdf_translation,
            self.pelvis_to_urdf_rpy,
        )

        self.oscbf_joint_names = (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        )

        if (not self.enabled) or self.use_dummy_filter:
            self.robot_model = None
            self.oscbf_config = None
            self.pelvis_oscbf_config = None
            self.cbf = None
            self.pelvis_cbf = None
        else:
            self.robot_model = self._build_robot_model(self.urdf_path)
            self.oscbf_config = OSCBFEEHumanCapsuleVelocityConfig(
                robot=self.robot_model,
                capsule_a_init=np.zeros((2, 3), dtype=np.float32),
                capsule_b_init=np.zeros((2, 3), dtype=np.float32),
                capsule_radii_init=np.ones((2,), dtype=np.float32) * 0.1,
                alpha_gain=self.alpha_gain,
            )
            self.pelvis_oscbf_config = OSCBFPelvisArmHumanCapsuleVelocityConfig(
                robot=self.robot_model,
                capsule_a_init=np.zeros((2, 3), dtype=np.float32),
                capsule_b_init=np.zeros((2, 3), dtype=np.float32),
                capsule_radii_init=np.ones((2,), dtype=np.float32) * 0.1,
                pelvis_to_urdf_transform=self.T_pelvis_urdf,
                alpha_gain=self.alpha_gain,
                pelvis_velocity_limits=self.pelvis_velocity_limits,
                pelvis_obj_weight=self.pelvis_cbf_weight,
                arm_obj_weight=self.arm_cbf_weight,
            )
            self.cbf = None
            self.pelvis_cbf = None
            if self.build_cbf_eagerly:
                self._ensure_cbf()
                if self.enable_pelvis_cbf:
                    self._ensure_pelvis_cbf()
            
        logger.debug(
            f"\n[OSCBFFilter][INIT]" 
            f"\n  enabled: {self.enabled}"
            f"\n  urdf_path: {self.urdf_path}"
            f"\n  expected_motion_dim: {self.expected_motion_dim}" 
            f"\n  filter_all_except_gripper: {self.filter_all_except_gripper}" 
            f"\n  use_dummy_filter: {self.use_dummy_filter}" 
            f"\n  dummy_scale: {self.dummy_scale}" 
            f"\n  dt: {self.dt}" 
            f"\n  human_margin: {self.human_margin}" 
            f"\n  alpha_gain: {self.alpha_gain}" 
            f"\n  control_type: {self.control_type}" 
            f"\n  enable_pelvis_cbf: {self.enable_pelvis_cbf}" 
            f"\n  runtime_joint_names: {self.runtime_joint_names}" 
            f"\n  robot_model: {type(self.robot_model).__name__}"
        )

    def _ensure_live_h_fn(self):
        if self.oscbf_config is None:
            return None
        if self._live_h_fn is None:
            self._live_h_fn = jax.jit(
                lambda q, capsule_a, capsule_b, capsule_radii: self.oscbf_config.h_1(
                    q,
                    capsule_a=capsule_a,
                    capsule_b=capsule_b,
                    capsule_radii=capsule_radii,
                )
            )
        return self._live_h_fn

    def compute_live_h_values(self, q_urdf, capsule_a_urdf, capsule_b_urdf, capsule_radii):
        self._validate_capsules(capsule_a_urdf, capsule_b_urdf, capsule_radii)
        h_fn = self._ensure_live_h_fn()
        if h_fn is None:
            return None
        return np.asarray(
            h_fn(
                jnp.asarray(q_urdf, dtype=jnp.float32),
                jnp.asarray(capsule_a_urdf, dtype=jnp.float32),
                jnp.asarray(capsule_b_urdf, dtype=jnp.float32),
                jnp.asarray(capsule_radii, dtype=jnp.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)

    def _ensure_cbf(self):
        if self.cbf is None:
            if self.oscbf_config is None:
                raise RuntimeError("Cannot build CBF without an OSCBF config.")
            logger.info("[OSCBFFilter] building CBF safety filter lazily")
            self.cbf = CBF.from_config(self.oscbf_config)
        return self.cbf

    def _ensure_pelvis_cbf(self):
        if self.pelvis_cbf is None:
            if self.pelvis_oscbf_config is None:
                raise RuntimeError("Cannot build pelvis CBF without a config.")
            logger.info("[OSCBFFilter] building pelvis+arm CBF safety filter lazily")
            self.pelvis_cbf = CBF.from_config(self.pelvis_oscbf_config)
        return self.pelvis_cbf

    def _build_robot_model(self, urdf_path: Path):
        robot_model = TreeManipulator.from_urdf(
            urdf_filename=str(urdf_path),
            ee_joint_idx=0,
            controlled_joint_indices=None,
        )

        # Check that the requested OSCBF arm joints exist in the URDF.
        missing_joint_names = [
            name for name in self.oscbf_joint_names
            if name not in robot_model.joint_names
        ]

        if missing_joint_names:
            raise ValueError(
                "OSCBF controlled joints not found in URDF.\n"
                f"Missing: {missing_joint_names}\n"
                f"Available URDF joints: {robot_model.joint_names}"
            )

        controlled_joint_indices = tuple(
            robot_model.joint_index(name)
            for name in self.oscbf_joint_names
        )

        robot_model.set_controlled_joints(controlled_joint_indices)

        ee_joint_name = "right_elbow_joint"

        if ee_joint_name not in robot_model.joint_names:
            raise ValueError(
                f"EE joint {ee_joint_name} not found in URDF.\n"
                f"Available URDF joints: {robot_model.joint_names}"
            )

        robot_model.ee_joint_idx = robot_model.joint_index(ee_joint_name)

        if robot_model.num_controls != len(self.bigym_action_arm_indices):
            raise ValueError(
                f"OSCBF action/joint mismatch: "
                f"{len(self.bigym_action_arm_indices)} action dims but "
                f"{robot_model.num_controls} robot controls"
            )

        # These are URDF joint indices, not BiGym indices.
        self.urdf_arm_joint_indices = np.asarray(
            controlled_joint_indices,
            dtype=np.int64,
        )

        # Fixed-torso / fixed-leg surrogate posture.
        # Everything not in urdf_arm_joint_indices remains fixed here.
        self.urdf_neutral_q = np.zeros(robot_model.num_joints, dtype=np.float32)
        self.urdf_neutral_qd = np.zeros(robot_model.num_joints, dtype=np.float32)

        logger.info(
            "[OSCBFFilter] robot_model loaded | "
            "num_joints=%d | num_controls=%d | "
            "controlled_joint_indices=%s | bigym_action_arm_indices=%s | "
            "bigym_action_safety_indices=%s | ee_joint_idx=%d",
            robot_model.num_joints,
            robot_model.num_controls,
            robot_model.controlled_joint_indices,
            self.bigym_action_arm_indices.tolist(),
            self.bigym_action_safety_indices.tolist(),
            robot_model.ee_joint_idx,
        )

        return robot_model

    def __call__(
        self,
        action: np.ndarray,
        env=None,
        observations=None,
        q_full: Optional[np.ndarray] = None,
        qd_full: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not self.enabled:
            return action
        
        if self.debug and not hasattr(self, "_printed_action_layout"):
            passthrough_indices = [
                i for i in range(action.shape[0])
                if i not in set(self.bigym_action_arm_indices.tolist())
            ]

            logger.debug("\n[OSCBFFilter][ACTION LAYOUT CHECK]")
            logger.debug("  full action dim: %d", action.shape[0])
            logger.debug("  arm action indices: %s", self.bigym_action_arm_indices.tolist())
            logger.debug("  passthrough indices: %s", passthrough_indices)
            logger.debug("  arm action values: %s", action[self.bigym_action_arm_indices])
            logger.debug("  passthrough values: %s", action[passthrough_indices])

            self._printed_action_layout = True

        if q_full is None or qd_full is None:
            raise ValueError("q_full and qd_full must be provided.")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        q_bigym = np.asarray(q_full, dtype=np.float32).reshape(-1)
        qd_bigym = np.asarray(qd_full, dtype=np.float32).reshape(-1)

        self._validate_full_inputs(action, q_bigym, qd_bigym)

        a_arm_bigym_nom = action[self.bigym_action_arm_indices]
        q_arm_bigym = q_bigym[self.bigym_state_arm_indices]

        # Build fixed-torso URDF surrogate state.
        q_urdf, qd_urdf, q_arm_bigym, q_arm_urdf = (
            self._build_urdf_surrogate_state_from_bigym(q_bigym, qd_bigym)
        )

        if self.enable_pelvis_cbf and self.pelvis_oscbf_config is not None:
            try:
                return self._filter_safety_action_with_pelvis_cbf(
                    action=action,
                    q_bigym=q_bigym,
                    qd_bigym=qd_bigym,
                    q_urdf=q_urdf,
                    qd_urdf=qd_urdf,
                    q_arm_bigym=q_arm_bigym,
                    q_arm_urdf=q_arm_urdf,
                    env=env,
                    observations=observations,
                ).astype(np.float32)
            except Exception as exc:
                if not self._warned_pelvis_cbf_fallback:
                    logger.warning(
                        "[OSCBFFilter] pelvis+arm CBF failed; falling back to arm-only CBF: %s",
                        exc,
                    )
                    self._warned_pelvis_cbf_fallback = True

        a_arm_bigym_safe = self._filter_motion_action(
            q_bigym=q_bigym,
            qd_bigym=qd_bigym,
            q_urdf=q_urdf,
            qd_urdf=qd_urdf,
            q_arm_bigym=q_arm_bigym,
            q_arm_urdf=q_arm_urdf,
            a_arm_bigym_nom=a_arm_bigym_nom,
            env=env,
            observations=observations,
        )

        safe_action = action.copy()
        safe_action[self.bigym_action_arm_indices] = a_arm_bigym_safe

        if self.max_action_delta is not None:
            delta = (
                safe_action[self.bigym_action_arm_indices]
                - action[self.bigym_action_arm_indices]
            )
            delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
            safe_action[self.bigym_action_arm_indices] = (
                action[self.bigym_action_arm_indices] + delta
            )

        return safe_action.astype(np.float32)

    def _validate_full_inputs(
        self,
        action: np.ndarray,
        q_full: np.ndarray,
        qd_full: np.ndarray,
    ):
        if action.ndim != 1:
            raise ValueError(f"action must be 1D, got {action.shape}")

        if q_full.ndim != 1:
            raise ValueError(f"q_full must be 1D, got {q_full.shape}")

        if qd_full.ndim != 1:
            raise ValueError(f"qd_full must be 1D, got {qd_full.shape}")

        if action.shape[0] != 16:
            raise ValueError(f"Expected action dim 16, got {action.shape[0]}")

        if q_full.shape[0] != 14:
            raise ValueError(f"Expected BiGym/RoboBase q_full dim 14, got {q_full.shape[0]}")

        if qd_full.shape[0] != 14:
            raise ValueError(f"Expected BiGym/RoboBase qd_full dim 14, got {qd_full.shape[0]}")

        if np.max(self.bigym_action_arm_indices) >= action.shape[0]:
            raise ValueError(
                f"bigym_action_arm_indices {self.bigym_action_arm_indices.tolist()} "
                f"out of range for action dim {action.shape[0]}"
            )

        if np.max(self.bigym_state_arm_indices) >= q_full.shape[0]:
            raise ValueError(
                f"bigym_state_arm_indices {self.bigym_state_arm_indices.tolist()} "
                f"out of range for q_full dim {q_full.shape[0]}"
            )

        if len(self.bigym_action_arm_indices) != 8:
            raise ValueError(
                f"Expected 8 action arm indices, got {len(self.bigym_action_arm_indices)}"
            )

        if len(self.bigym_state_arm_indices) != 8:
            raise ValueError(
                f"Expected 8 state arm indices, got {len(self.bigym_state_arm_indices)}"
            )
        

    def _filter_safety_action_with_pelvis_cbf(
        self,
        action: np.ndarray,
        q_bigym: np.ndarray,
        qd_bigym: np.ndarray,
        q_urdf: np.ndarray,
        qd_urdf: np.ndarray,
        q_arm_bigym: np.ndarray,
        q_arm_urdf: np.ndarray,
        env=None,
        observations=None,
    ) -> np.ndarray:
        human_obstacles = self._extract_human_obstacles(env, observations)
        capsule_a_world = human_obstacles["capsule_a"]
        capsule_b_world = human_obstacles["capsule_b"]
        capsule_radii = human_obstacles["capsule_radii"]
        self._validate_capsules(capsule_a_world, capsule_b_world, capsule_radii)

        if capsule_a_world.shape[0] != 2:
            raise ValueError(
                f"Current pelvis OSCBF config was initialised for 2 human capsules, "
                f"but extracted {capsule_a_world.shape[0]}."
            )

        self.pelvis_oscbf_config.set_human_capsules(
            capsule_a_world,
            capsule_b_world,
            capsule_radii,
        )

        q_base_bigym = q_bigym[self.bigym_state_base_indices]
        a_base_bigym_nom = action[self.bigym_action_base_indices]
        a_arm_bigym_nom = action[self.bigym_action_arm_indices]

        u_base_nom = self._bigym_base_action_to_velocity(
            q_base_bigym=q_base_bigym,
            a_base_bigym_nom=a_base_bigym_nom,
        )
        u_arm_urdf_nom = self._bigym_action_to_urdf_velocity(
            q_arm_bigym=q_arm_bigym,
            q_arm_urdf=q_arm_urdf,
            a_arm_bigym_nom=a_arm_bigym_nom,
        )
        z_aug = np.concatenate([q_base_bigym, q_urdf], axis=0).astype(np.float32)
        u_aug_nom = np.concatenate([u_base_nom, u_arm_urdf_nom], axis=0).astype(np.float32)

        cbf = self._ensure_pelvis_cbf()
        u_aug_safe = np.asarray(
            cbf.safety_filter(
                jnp.asarray(z_aug, dtype=jnp.float32),
                jnp.asarray(u_aug_nom, dtype=jnp.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)

        expected_dim = len(self.bigym_action_base_indices) + len(self.bigym_action_arm_indices)
        if u_aug_safe.shape[0] != expected_dim:
            raise ValueError(
                f"Expected pelvis OSCBF output dim {expected_dim}, got {u_aug_safe.shape[0]}"
            )

        u_base_safe = u_aug_safe[: len(self.bigym_action_base_indices)]
        u_arm_safe = u_aug_safe[len(self.bigym_action_base_indices) :]
        a_base_bigym_safe = self._bigym_base_velocity_to_action(
            q_base_bigym=q_base_bigym,
            u_base_safe=u_base_safe,
        )
        a_arm_bigym_safe = self._urdf_velocity_to_bigym_action(
            q_arm_bigym=q_arm_bigym,
            q_arm_urdf=q_arm_urdf,
            u_arm_urdf_safe=u_arm_safe,
        )

        safe_action = action.copy()
        safe_action[self.bigym_action_base_indices] = a_base_bigym_safe
        safe_action[self.bigym_action_arm_indices] = a_arm_bigym_safe

        if self.max_action_delta is not None:
            delta = safe_action[self.bigym_action_safety_indices] - action[self.bigym_action_safety_indices]
            delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
            safe_action[self.bigym_action_safety_indices] = action[self.bigym_action_safety_indices] + delta

        if self.debug and not hasattr(self, "_printed_pelvis_cbf_debug"):
            logger.debug("\n[OSCBFFilter][PELVIS+ARM OSCBF]")
            logger.debug("  z_aug dim: %d", z_aug.shape[0])
            logger.debug("  u_aug_nom dim: %d", u_aug_nom.shape[0])
            logger.debug("  ||u_aug_safe-u_aug_nom||: %s", float(np.linalg.norm(u_aug_safe - u_aug_nom)))
            logger.debug("  base action delta: %s", safe_action[self.bigym_action_base_indices] - action[self.bigym_action_base_indices])
            logger.debug("  arm action delta norm: %s", float(np.linalg.norm(a_arm_bigym_safe - a_arm_bigym_nom)))
            self._printed_pelvis_cbf_debug = True

        return safe_action.astype(np.float32)

    def _filter_motion_action(
        self,
        q_bigym: np.ndarray,
        qd_bigym: np.ndarray,
        q_urdf: np.ndarray,
        qd_urdf: np.ndarray,
        q_arm_bigym: np.ndarray,
        q_arm_urdf: np.ndarray,
        a_arm_bigym_nom: np.ndarray,
        env=None,
        observations=None,
    ) -> np.ndarray:
        if self.use_dummy_filter:
            if self.control_type == "absolute":
                # Correct dummy slowdown for absolute joint targets.
                a_arm_bigym_safe = q_arm_bigym + self.dummy_scale * (
                    a_arm_bigym_nom - q_arm_bigym
                )
            else:
                a_arm_bigym_safe = self.dummy_scale * a_arm_bigym_nom

            logger.debug("\n[OSCBFFilter][DUMMY]")
            logger.debug(
                "  a_arm_bigym_nom min/max: %s, %s",
                float(np.min(a_arm_bigym_nom)),
                float(np.max(a_arm_bigym_nom)),
            )
            logger.debug(
                "  a_arm_bigym_safe min/max: %s, %s",
                float(np.min(a_arm_bigym_safe)),
                float(np.max(a_arm_bigym_safe)),
            )
            logger.debug(
                "  ||a_arm_bigym_safe-a_arm_bigym_nom||: %s",
                float(np.linalg.norm(a_arm_bigym_safe - a_arm_bigym_nom)),
            )

            return a_arm_bigym_safe.astype(np.float32)

        # Extract human capsules in MuJoCo world frame.
        human_obstacles = self._extract_human_obstacles(env, observations)

        capsule_a_world = human_obstacles["capsule_a"]
        capsule_b_world = human_obstacles["capsule_b"]
        capsule_radii = human_obstacles["capsule_radii"]

        self._validate_capsules(capsule_a_world, capsule_b_world, capsule_radii)

        # Build world -> URDF-surrogate transform.
        # Since torso is fixed, this uses only floating pelvis translation/yaw
        # and a fixed pelvis-to-URDF calibration.
        T_world_urdf = self._get_world_T_urdf_from_bigym_state(q_bigym)
        T_urdf_world = np.linalg.inv(T_world_urdf)

        # Transform human capsules into URDF surrogate base frame.
        capsule_a_urdf = self._transform_points(T_urdf_world, capsule_a_world)
        capsule_b_urdf = self._transform_points(T_urdf_world, capsule_b_world)

        self._validate_capsules(capsule_a_urdf, capsule_b_urdf, capsule_radii)

        if capsule_a_urdf.shape[0] != 2:
            raise ValueError(
                f"Current OSCBF config was initialised for 2 human capsules, "
                f"but extracted {capsule_a_urdf.shape[0]}. "
                f"Update capsule_a_init/capsule_b_init/capsule_radii_init accordingly."
            )

        self.oscbf_config.set_human_capsules(
            capsule_a_urdf,
            capsule_b_urdf,
            capsule_radii,
        )

        if self.debug:
            self._print_frame_and_h_debug(
                q_urdf,
                capsule_a_urdf,
                capsule_b_urdf,
                capsule_radii,
            )

        # Convert BiGym action into URDF 8D arm velocity.
        u_arm_urdf_nom = self._bigym_action_to_urdf_velocity(
            q_arm_bigym=q_arm_bigym,
            q_arm_urdf=q_arm_urdf,
            a_arm_bigym_nom=a_arm_bigym_nom,
        )

        # OSCBF sees:
        #   state   = fixed-torso full URDF surrogate q
        #   control = 8D URDF arm velocity
        cbf = self._ensure_cbf()
        u_arm_urdf_safe = np.asarray(
            cbf.safety_filter(
                jnp.asarray(q_urdf, dtype=jnp.float32),
                jnp.asarray(u_arm_urdf_nom, dtype=jnp.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)

        if u_arm_urdf_safe.shape[0] != len(self.bigym_action_arm_indices):
            raise ValueError(
                f"Expected OSCBF output dim {len(self.bigym_action_arm_indices)}, "
                f"got {u_arm_urdf_safe.shape[0]}"
            )

        # Convert safe URDF arm velocity back into BiGym action format.
        a_arm_bigym_safe = self._urdf_velocity_to_bigym_action(
            q_arm_bigym=q_arm_bigym,
            q_arm_urdf=q_arm_urdf,
            u_arm_urdf_safe=u_arm_urdf_safe,
        )

        if self.debug and not self._printed_frame_debug:
            self._print_surrogate_alignment_debug(
                q_urdf=q_urdf,
                T_world_urdf=T_world_urdf,
                capsule_a_world=capsule_a_world,
                capsule_b_world=capsule_b_world,
                capsule_a_urdf=capsule_a_urdf,
                capsule_b_urdf=capsule_b_urdf,
            )
            logger.debug("\n[OSCBFFilter][REAL OSCBF - FIXED TORSO SURROGATE]")
            logger.debug("  q_bigym dim: %d", q_bigym.shape[0])
            logger.debug("  q_urdf dim: %d", q_urdf.shape[0])
            logger.debug("  u_arm_urdf_nom dim: %d", u_arm_urdf_nom.shape[0])
            logger.debug("  u_arm_urdf_safe dim: %d", u_arm_urdf_safe.shape[0])
            logger.debug(
                "  ||u_arm_urdf_safe-u_arm_urdf_nom||: %s",
                float(np.linalg.norm(u_arm_urdf_safe - u_arm_urdf_nom)),
            )
            logger.debug(
                "  ||a_arm_bigym_safe-a_arm_bigym_nom||: %s",
                float(np.linalg.norm(a_arm_bigym_safe - a_arm_bigym_nom)),
            )
            logger.debug(
                "  finite a_arm_bigym_safe: %s",
                bool(np.isfinite(a_arm_bigym_safe).all()),
            )

        if a_arm_bigym_safe.shape != a_arm_bigym_nom.shape:
            raise ValueError(
                f"a_arm_bigym_safe shape {a_arm_bigym_safe.shape}, "
                f"expected {a_arm_bigym_nom.shape}"
            )

        return a_arm_bigym_safe.astype(np.float32)
    
    def _action_to_velocity(self, q_arm, a_arm_nom):
        if self.control_type == "absolute":
            return (a_arm_nom - q_arm) / self.dt
        if self.control_type == "delta":
            return a_arm_nom / self.dt
        if self.control_type == "velocity":
            return a_arm_nom
        raise RuntimeError(f"Unexpected control_type: {self.control_type}")
        

    def _velocity_to_action(self, q_arm: np.ndarray, qdot_safe: np.ndarray) -> np.ndarray:
        if qdot_safe.shape[0] != q_arm.shape[0]:
            raise ValueError(
                f"Expected qdot_safe dim {q_arm.shape[0]}, got {qdot_safe.shape[0]}"
            )

        if self.control_type == "velocity":
            return qdot_safe

        if self.control_type == "delta":
            return qdot_safe * self.dt

        if self.control_type == "absolute":
            return q_arm + qdot_safe * self.dt

        raise RuntimeError(f"Unexpected control_type: {self.control_type}")
    

    def _extract_robot_state(self, env, observations=None):
        """
        TODO: connect this to your existing H1 state bridge.

        For now, I recommend passing q_full and qd_full explicitly from the
        current place where OSCBFFilter already calls bridge.filter_first_action().
        """
        raise NotImplementedError(
            "Pass q_full and qd_full explicitly, or implement _extract_robot_state()."
        )

    def _extract_human_obstacles(self, env, observations=None):
        obs = observations if observations is not None else {}
        task = get_bigym_task(env)

        if "human_arm_qpos" not in obs or "human_arm_qvel" not in obs:
            obs = task._get_task_privileged_obs()

        capsule_a, capsule_b, capsule_radii, capsule_source = (
            self._extract_human_arm_capsules_from_qpos(task, obs)
        )

        return {
            "human_arm_qpos": np.asarray(obs["human_arm_qpos"], dtype=np.float32).reshape(-1),
            "human_arm_qvel": np.asarray(obs["human_arm_qvel"], dtype=np.float32).reshape(-1),
            "human_arm_carrier_qpos": np.asarray(
                obs.get("human_arm_carrier_qpos", []),
                dtype=np.float32,
            ).reshape(-1),
            "human_arm_carrier_qvel": np.asarray(
                obs.get("human_arm_carrier_qvel", []),
                dtype=np.float32,
            ).reshape(-1),
            "capsule_a": capsule_a,
            "capsule_b": capsule_b,
            "capsule_radii": capsule_radii,
            "capsule_source": capsule_source,
            "capsule_fast_cache_hits": int(self._human_capsule_fast_cache_hits),
            "capsule_fast_cache_misses": int(self._human_capsule_fast_cache_misses),
            "capsule_fast_cache_failures": int(self._human_capsule_fast_cache_failures),
        }

    def _extract_human_arm_capsules_from_qpos(self, task, obs):
        try:
            cache = self._get_or_build_human_capsule_kinematic_cache(task)
            qpos = np.asarray(obs["human_arm_qpos"], dtype=np.float32).reshape(-1)
            carrier_qpos = np.asarray(
                obs.get("human_arm_carrier_qpos", []),
                dtype=np.float32,
            ).reshape(-1)

            human_count = len(cache["carrier_base_pos"])
            if qpos.size != human_count * 4:
                raise RuntimeError(
                    f"Expected {human_count * 4} human arm qpos values, got {qpos.size}"
                )
            if carrier_qpos.size != human_count * 2:
                carrier_qpos = self._read_human_carrier_qpos_from_task(task)
            if carrier_qpos.size != human_count * 2:
                raise RuntimeError(
                    f"Expected {human_count * 2} human carrier qpos values, got {carrier_qpos.size}"
                )

            qpos = qpos.reshape(human_count, 4)
            carrier_qpos = carrier_qpos.reshape(human_count, 2)
            capsule_a = []
            capsule_b = []
            z_upper = np.asarray([0.0, 0.0, -0.34], dtype=np.float32)
            z_fore = np.asarray([0.0, 0.0, -0.30], dtype=np.float32)

            for human_idx in range(human_count):
                shoulder = cache["carrier_base_pos"][human_idx].copy()
                shoulder[0:2] += carrier_qpos[human_idx]

                q_base, q_yaw, q_pitch, q_elbow = qpos[human_idx]
                r_shoulder = (
                    self._rot_z(q_base)
                    @ self._rot_z(q_yaw)
                    @ self._rot_y(q_pitch)
                )
                elbow = shoulder + r_shoulder @ z_upper
                r_forearm = r_shoulder @ self._rot_y(q_elbow)
                wrist = elbow + r_forearm @ z_fore

                capsule_a.append(elbow)
                capsule_b.append(shoulder)
                capsule_a.append(wrist)
                capsule_b.append(elbow)

            self._human_capsule_fast_cache_hits += 1
            return (
                np.asarray(capsule_a, dtype=np.float32),
                np.asarray(capsule_b, dtype=np.float32),
                cache["radii"].copy(),
                "cached_qpos_fk",
            )
        except Exception as exc:  # noqa: BLE001
            self._human_capsule_fast_cache_failures += 1
            logger.debug(
                "Qpos human capsule extraction failed; using MuJoCo body/geom fallback: %s",
                exc,
            )
            return self._extract_human_arm_capsules_fast(task)

    def _read_human_carrier_qpos_from_task(self, task):
        try:
            return np.concatenate(
                [
                    np.asarray(task._mojo.data.qpos[human._carrier_qpos_adr], dtype=np.float32)
                    for human in task.humanarms
                ],
                dtype=np.float32,
            )
        except Exception:  # noqa: BLE001
            return np.asarray([], dtype=np.float32)

    @staticmethod
    def _rot_z(theta: float):
        c, s = np.cos(float(theta)), np.sin(float(theta))
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    @staticmethod
    def _rot_y(theta: float):
        c, s = np.cos(float(theta)), np.sin(float(theta))
        return np.asarray(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float32,
        )

    def _extract_human_arm_capsules(self, task):
        if not hasattr(task, "humanarms"):
            raise AttributeError(
                f"{type(task).__name__} has no 'humanarms'; "
                "cannot extract human-arm capsules"
            )

        model = task._mojo.model
        data = task._mojo.data

        geom_specs = [
            ("upperarm_geom", 0.035, 0.34 / 2.0),
            ("forearm_geom", 0.032, 0.30 / 2.0),
        ]

        capsule_a = []
        capsule_b = []
        capsule_radii = []

        for human in task.humanarms:
            for geom_name, radius_xml, half_length in geom_specs:
                geom_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    human._pref(geom_name),
                )

                if geom_id < 0:
                    raise RuntimeError(
                        f"Human arm geom not found: {human._pref(geom_name)}"
                    )

                center = np.asarray(data.geom_xpos[geom_id], dtype=np.float32)
                rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)

                local_z_world = rot[:, 2].astype(np.float32)

                capsule_a.append(center - half_length * local_z_world)
                capsule_b.append(center + half_length * local_z_world)
                capsule_radii.append(radius_xml + self.human_margin)

        return (
            np.asarray(capsule_a, dtype=np.float32),
            np.asarray(capsule_b, dtype=np.float32),
            np.asarray(capsule_radii, dtype=np.float32),
        )

    def _extract_human_arm_capsules_fast(self, task):
        try:
            cache = self._get_or_build_human_capsule_kinematic_cache(task)
            model = task._mojo.model
            data = task._mojo.data
            if cache["model_id"] != id(model):
                raise RuntimeError("MuJoCo model changed since human capsule cache was built")

            capsule_a = []
            capsule_b = []
            for body_id, local_a, local_b in zip(
                cache["body_ids"],
                cache["local_a"],
                cache["local_b"],
            ):
                body_pos = np.asarray(data.xpos[int(body_id)], dtype=np.float32)
                body_rot = np.asarray(data.xmat[int(body_id)], dtype=np.float32).reshape(3, 3)
                capsule_a.append(body_pos + body_rot @ local_a)
                capsule_b.append(body_pos + body_rot @ local_b)

            self._human_capsule_fast_cache_hits += 1
            return (
                np.asarray(capsule_a, dtype=np.float32),
                np.asarray(capsule_b, dtype=np.float32),
                cache["radii"].copy(),
                "cached_body_fk",
            )
        except Exception as exc:  # noqa: BLE001
            self._human_capsule_fast_cache_failures += 1
            logger.debug(
                "Fast human capsule extraction failed; using geom_xpos fallback: %s",
                exc,
            )
            capsule_a, capsule_b, capsule_radii = self._extract_human_arm_capsules(task)
            return capsule_a, capsule_b, capsule_radii, "geom_xpos_fallback"

    def _get_or_build_human_capsule_kinematic_cache(self, task):
        if not hasattr(task, "humanarms"):
            raise AttributeError(
                f"{type(task).__name__} has no 'humanarms'; "
                "cannot extract human-arm capsules"
            )

        model = task._mojo.model
        data = task._mojo.data
        cache_key = (
            id(model),
            tuple(id(human) for human in task.humanarms),
            float(self.human_margin),
        )
        cache = self._human_capsule_kinematic_cache
        if cache is not None and cache.get("cache_key") == cache_key:
            return cache

        geom_specs = [
            ("upperarm_geom", 0.035, 0.34 / 2.0),
            ("forearm_geom", 0.032, 0.30 / 2.0),
        ]
        body_ids = []
        local_a = []
        local_b = []
        radii = []
        carrier_base_pos = []

        for human in task.humanarms:
            carrier_body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                human._pref("arm_carrier"),
            )
            if carrier_body_id < 0:
                raise RuntimeError(f"Human arm carrier body not found: {human._pref('arm_carrier')}")
            carrier_qpos = np.asarray(data.qpos[human._carrier_qpos_adr], dtype=np.float32)
            carrier_pos = np.asarray(data.xpos[carrier_body_id], dtype=np.float32).copy()
            carrier_pos[0:2] -= carrier_qpos
            carrier_base_pos.append(carrier_pos)
            for geom_name, radius_xml, half_length in geom_specs:
                geom_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    human._pref(geom_name),
                )
                if geom_id < 0:
                    raise RuntimeError(
                        f"Human arm geom not found: {human._pref(geom_name)}"
                    )

                body_id = int(model.geom_bodyid[geom_id])
                geom_center = np.asarray(data.geom_xpos[geom_id], dtype=np.float32)
                geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
                local_z_world = geom_rot[:, 2].astype(np.float32)
                endpoint_a_world = geom_center - half_length * local_z_world
                endpoint_b_world = geom_center + half_length * local_z_world

                body_pos = np.asarray(data.xpos[body_id], dtype=np.float32)
                body_rot = np.asarray(data.xmat[body_id], dtype=np.float32).reshape(3, 3)
                body_rot_t = body_rot.T
                body_ids.append(body_id)
                local_a.append(body_rot_t @ (endpoint_a_world - body_pos))
                local_b.append(body_rot_t @ (endpoint_b_world - body_pos))
                radii.append(radius_xml + self.human_margin)

        cache = {
            "cache_key": cache_key,
            "model_id": id(model),
            "carrier_base_pos": np.asarray(carrier_base_pos, dtype=np.float32),
            "body_ids": np.asarray(body_ids, dtype=np.int32),
            "local_a": np.asarray(local_a, dtype=np.float32),
            "local_b": np.asarray(local_b, dtype=np.float32),
            "radii": np.asarray(radii, dtype=np.float32),
        }
        self._validate_capsules(cache["local_a"], cache["local_b"], cache["radii"])
        self._human_capsule_kinematic_cache = cache
        self._human_capsule_fast_cache_misses += 1
        return cache

    def _validate_capsules(self, capsule_a, capsule_b, capsule_radii):
        if capsule_a.ndim != 2 or capsule_a.shape[1] != 3:
            raise ValueError(f"capsule_a must have shape (N, 3), got {capsule_a.shape}")

        if capsule_b.shape != capsule_a.shape:
            raise ValueError(
                f"capsule_b must match capsule_a, got {capsule_b.shape}"
            )

        if capsule_radii.shape != (capsule_a.shape[0],):
            raise ValueError(
                f"capsule_radii must have shape ({capsule_a.shape[0]},), "
                f"got {capsule_radii.shape}"
            )

    def _print_frame_and_h_debug(
        self,
        q_full: np.ndarray,
        capsule_a: np.ndarray,
        capsule_b: np.ndarray,
        capsule_radii: np.ndarray,
    ):
        h = np.asarray(
            self.oscbf_config.h_1(jnp.asarray(q_full, dtype=jnp.float32))
        )

        logger.debug(f"\n[OSCBFFilter][SAFETY]")
        logger.debug(f"  min_h: {float(np.min(h))}")
        logger.debug(f"  h: {h}")
        logger.debug(f"  num_capsules: {h.shape[0]}")

        if not self._printed_frame_debug:
            ee_pos = np.asarray(
                self.robot_model.ee_position(jnp.asarray(q_full, dtype=jnp.float32))
            )

            logger.debug(f"\n[OSCBFFilter][FRAME CHECK]")
            logger.debug(f"  ee_pos: {ee_pos}")
            logger.debug(f"  capsule_a: {capsule_a}")
            logger.debug(f"  capsule_b: {capsule_b}")
            logger.debug(f"  capsule_radii: {capsule_radii}")

            self._printed_frame_debug = True

    def _print_action_debug(
        self,
        action: np.ndarray,
        safe_action: np.ndarray,
        motion_nom: np.ndarray,
        motion_safe: np.ndarray,
    ):
        logger.debug(f"\n[OSCBFFilter][ACTION]")
        logger.debug(f"  action shape: {action.shape}")
        logger.debug(f"  safe_action shape: {safe_action.shape}")
        logger.debug(f"  motion dim: {self.expected_motion_dim}")
        logger.debug(f"  motion delta norm: {float(np.linalg.norm(motion_safe - motion_nom))}")
        logger.debug(f"  full action delta norm: {float(np.linalg.norm(safe_action - action))}")

        if action.shape[0] > self.expected_motion_dim:
            logger.debug(f"  passthrough nominal: {action[self.expected_motion_dim:]}")
            logger.debug(f"  passthrough safe: {safe_action[self.expected_motion_dim:]}")
            logger.debug(f"  passthrough unchanged: {bool(np.allclose(safe_action[self.expected_motion_dim:], action[self.expected_motion_dim:]))}")

    def _build_urdf_surrogate_state_from_bigym(
        self,
        q_bigym: np.ndarray,
        qd_bigym: np.ndarray,
    ):
        q_bigym = np.asarray(q_bigym, dtype=np.float32).reshape(-1)
        qd_bigym = np.asarray(qd_bigym, dtype=np.float32).reshape(-1)

        q_arm_bigym = q_bigym[self.bigym_state_arm_indices]
        qd_arm_bigym = qd_bigym[self.bigym_state_arm_indices]

        if q_arm_bigym.shape[0] != 8:
            raise ValueError(f"Expected 8D q_arm_bigym, got {q_arm_bigym.shape}")

        q_arm_urdf = self.arm_sign * q_arm_bigym + self.arm_offset
        qd_arm_urdf = self.arm_sign * qd_arm_bigym

        q_urdf = self.urdf_neutral_q.copy()
        qd_urdf = self.urdf_neutral_qd.copy()

        # Fixed torso/legs:
        # only these 8 URDF arm joints are overwritten.
        q_urdf[self.urdf_arm_joint_indices] = q_arm_urdf
        qd_urdf[self.urdf_arm_joint_indices] = qd_arm_urdf

        return q_urdf, qd_urdf, q_arm_bigym, q_arm_urdf

    def _bigym_base_action_to_velocity(
        self,
        q_base_bigym: np.ndarray,
        a_base_bigym_nom: np.ndarray,
    ) -> np.ndarray:
        del q_base_bigym
        if self.control_type in {"absolute", "delta"}:
            return a_base_bigym_nom / self.dt
        if self.control_type == "velocity":
            return a_base_bigym_nom
        raise RuntimeError(f"Unexpected control_type: {self.control_type}")

    def _bigym_base_velocity_to_action(
        self,
        q_base_bigym: np.ndarray,
        u_base_safe: np.ndarray,
    ) -> np.ndarray:
        del q_base_bigym
        if self.control_type == "velocity":
            return u_base_safe
        if self.control_type in {"absolute", "delta"}:
            return u_base_safe * self.dt
        raise RuntimeError(f"Unexpected control_type: {self.control_type}")

    def _bigym_action_to_urdf_velocity(
        self,
        q_arm_bigym: np.ndarray,
        q_arm_urdf: np.ndarray,
        a_arm_bigym_nom: np.ndarray,
    ):
        if self.control_type == "absolute":
            a_arm_urdf_nom = self.arm_sign * a_arm_bigym_nom + self.arm_offset
            u_arm_urdf_nom = (a_arm_urdf_nom - q_arm_urdf) / self.dt
            return u_arm_urdf_nom

        if self.control_type == "delta":
            return self.arm_sign * a_arm_bigym_nom / self.dt

        if self.control_type == "velocity":
            return self.arm_sign * a_arm_bigym_nom

        raise RuntimeError(f"Unexpected control_type: {self.control_type}")
    
    def _urdf_velocity_to_bigym_action(
        self,
        q_arm_bigym: np.ndarray,
        q_arm_urdf: np.ndarray,
        u_arm_urdf_safe: np.ndarray,
    ):
        if self.control_type == "velocity":
            return self.arm_sign * u_arm_urdf_safe

        if self.control_type == "delta":
            return self.arm_sign * (u_arm_urdf_safe * self.dt)

        if self.control_type == "absolute":
            q_arm_urdf_safe = q_arm_urdf + u_arm_urdf_safe * self.dt
            q_arm_bigym_safe = self.arm_sign * (q_arm_urdf_safe - self.arm_offset)
            return q_arm_bigym_safe

        raise RuntimeError(f"Unexpected control_type: {self.control_type}")

    def _transform_points(self, T: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([points, ones], axis=1)
        return (T @ points_h.T).T[:, :3].astype(np.float32)
    
    def _make_transform_from_xyz_rpy(
        self,
        xyz: np.ndarray,
        rpy: np.ndarray,
    ) -> np.ndarray:
        """
        Return T_parent_child.

        Convention:
            p_parent = T_parent_child @ p_child

        xyz/rpy describe the child frame expressed in the parent frame.
        """
        xyz = np.asarray(xyz, dtype=np.float32).reshape(3)
        rpy = np.asarray(rpy, dtype=np.float32).reshape(3)

        roll, pitch, yaw = rpy

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=np.float32,
        )

        R_y = np.asarray(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float32,
        )

        R_z = np.asarray(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        R = R_z @ R_y @ R_x

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T
    
    def _make_transform_from_xyz_yaw(
        self,
        xyz: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        """
        Return T_parent_child for translation + yaw only.

        Convention:
            p_parent = T_parent_child @ p_child
        """
        xyz = np.asarray(xyz, dtype=np.float32).reshape(3)
        yaw = float(yaw)

        cy, sy = np.cos(yaw), np.sin(yaw)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = np.asarray(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        T[:3, 3] = xyz
        return T
    
    def _get_world_T_urdf_from_bigym_state(self, q_bigym: np.ndarray) -> np.ndarray:
        """
        Build the world-frame pose of the fixed-torso URDF surrogate.

        Assumption:
            q_bigym[0:4] = floating pelvis x, y, z, yaw

        If your actual state order is different, only this method needs changing.
        """
        q_bigym = np.asarray(q_bigym, dtype=np.float32).reshape(-1)

        pelvis_xyz = np.asarray(
            [q_bigym[0], q_bigym[1], q_bigym[2]],
            dtype=np.float32,
        )
        pelvis_yaw = float(q_bigym[3])

        T_world_pelvis = self._make_transform_from_xyz_yaw(
            pelvis_xyz,
            pelvis_yaw,
        )
        # Convention:
        #   T_parent_child maps child-frame points into parent frame.
        #   T_pelvis_urdf maps URDF-frame points into BiGym pelvis frame.
        #   Therefore:
        #       p_world = T_world_pelvis @ T_pelvis_urdf @ p_urdf
        T_world_urdf = T_world_pelvis @ self.T_pelvis_urdf
        return T_world_urdf.astype(np.float32)

    def _transform_points_homogeneous(self, T: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 3)

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([points, ones], axis=1)
        return (T @ points_h.T).T[:, :3].astype(np.float32)


    def _print_surrogate_alignment_debug(
        self,
        q_urdf: np.ndarray,
        T_world_urdf: np.ndarray,
        capsule_a_world: np.ndarray,
        capsule_b_world: np.ndarray,
        capsule_a_urdf: np.ndarray,
        capsule_b_urdf: np.ndarray,
    ):
        ee_urdf = np.asarray(
            self.robot_model.ee_position(jnp.asarray(q_urdf, dtype=jnp.float32)),
            dtype=np.float32,
        ).reshape(1, 3)

        ee_world = self._transform_points_homogeneous(T_world_urdf, ee_urdf)

        logger.debug("\n[OSCBFFilter][SURROGATE ALIGNMENT]")
        logger.debug("  ee_urdf: %s", ee_urdf.reshape(-1))
        logger.debug("  ee_world_from_surrogate: %s", ee_world.reshape(-1))
        logger.debug("  capsule_a_world[0]: %s", capsule_a_world[0])
        logger.debug("  capsule_b_world[0]: %s", capsule_b_world[0])
        logger.debug("  capsule_a_urdf[0]: %s", capsule_a_urdf[0])
        logger.debug("  capsule_b_urdf[0]: %s", capsule_b_urdf[0])
        logger.debug("  T_world_urdf:\n%s", T_world_urdf)
