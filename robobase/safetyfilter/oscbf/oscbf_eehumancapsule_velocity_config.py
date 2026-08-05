import numpy as np
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from cbfpy.config.cbf_config import CBFConfig


class OSCBFEEHumanCapsuleVelocityConfig(CBFConfig):
    """
    Human-aware velocity-level OSCBF for TreeManipulator.

    State:
        z = q_full, shape (N,)

    Control:
        u = qdot_ctrl, shape (M,)

    where:
        N = robot.num_joints
        M = robot.num_controls
    """

    def __init__(
        self,
        robot,
        capsule_a_init,
        capsule_b_init,
        capsule_radii_init,
        alpha_gain: float = 10.0,
        pos_obj_weight: float = 1.0,
        rot_obj_weight: float = 1.0,
        joint_obj_weight: float = 1.0,
    ):
        self.robot = robot
        self.num_joints = robot.num_joints          # N = 19
        self.num_controls = robot.num_controls      # M = 14
        self.task_dim = 6
        self.is_redundant = self.num_controls > self.task_dim

        self.alpha_gain = float(alpha_gain)
        self.pos_obj_weight = float(pos_obj_weight)
        self.rot_obj_weight = float(rot_obj_weight)
        self.joint_space_obj_weight = float(joint_obj_weight)
        # MuJoCo contact geoms for the H1 right shoulder/elbow are offset from
        # the URDF joint-to-joint surrogate axes used here. Inflate the robot
        # capsules so h becomes conservative relative to those contact geoms.
        self.right_arm_contact_margin = 0.065
        self.right_arm_capsule_radii = jnp.asarray(
            (
                0.03 + self.right_arm_contact_margin,
                0.025 + self.right_arm_contact_margin,
            ),
            dtype=jnp.float32,
        )
        # The URDF surrogate ends at the right elbow. Approximate the physical arm
        # with a short forearm capsule and a small end-gripper sphere.
        self.right_forearm_capsule_offset = jnp.asarray(
            [0.30, 0.0, 0.0],
            dtype=jnp.float32,
        )
        self.right_gripper_sphere_offset = jnp.asarray(
            [0.36, 0.0, 0.0],
            dtype=jnp.float32,
        )
        self.right_gripper_sphere_radius = jnp.asarray(0.13, dtype=jnp.float32)

        self.right_shoulder_yaw_idx = self.robot.joint_index(
            "right_shoulder_yaw_joint"
        )
        self.right_elbow_idx = self.robot.joint_index(
            "right_elbow_joint"
        )

        self.W_T_W_task_diag = tuple(
            np.array([self.pos_obj_weight] * 3 + [self.rot_obj_weight] * 3) ** 2
        )
        self.W_T_W_joint_diag = tuple(
            np.array([self.joint_space_obj_weight] * self.num_controls) ** 2
        )

        self.set_human_capsules(
            capsule_a_init,
            capsule_b_init,
            capsule_radii_init,
        )

        ctrl = np.asarray(self.robot.controlled_joint_indices, dtype=np.int64)
        joint_max_velocities = np.asarray(self.robot.joint_max_velocities)[ctrl]

        super().__init__(
            n=self.num_joints,
            m=self.num_controls,
            u_min=-joint_max_velocities,
            u_max=joint_max_velocities,
            init_args=(
                self.capsule_a,
                self.capsule_b,
                self.capsule_radii,
            ),
        )

    def set_human_capsules(self, capsule_a, capsule_b, capsule_radii):
        self.capsule_a = jnp.asarray(capsule_a, dtype=jnp.float32)
        self.capsule_b = jnp.asarray(capsule_b, dtype=jnp.float32)
        self.capsule_radii = jnp.asarray(capsule_radii, dtype=jnp.float32)

    def f(self, z, *args, **kwargs):
        """
        Drift dynamics for velocity-level kinematic system.

        qdot = f(q) + g(q)u
        Here f(q) = 0.
        """
        return jnp.zeros(
            (self.num_joints,),
            dtype=z.dtype,
        )

    def g(self, z, *args, **kwargs):
        """
        State dynamics:
            qdot_full = S u_ctrl

        Shape:
            g(z): (N, M)
        """
        ctrl = jnp.asarray(self.robot.controlled_joint_indices, dtype=jnp.int32)
        col = jnp.arange(self.num_controls, dtype=jnp.int32)

        G = jnp.zeros(
            (self.num_joints, self.num_controls),
            dtype=z.dtype,
        )

        one = jnp.asarray(1.0, dtype=z.dtype)

        return G.at[ctrl, col].set(one)

    def _segment_segment_distance_sq(self, p1, q1, p2, q2):
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2

        a = jnp.dot(d1, d1)
        e = jnp.dot(d2, d2)
        f = jnp.dot(d2, r)
        eps = jnp.asarray(1e-8, dtype=p1.dtype)

        s_unclamped = (jnp.dot(d1, d2) * f - e * jnp.dot(d1, r)) / (a * e - jnp.dot(d1, d2) ** 2 + eps)
        s = jnp.clip(s_unclamped, 0.0, 1.0)
        t = (jnp.dot(d1, d2) * s + f) / (e + eps)

        t_clamped = jnp.clip(t, 0.0, 1.0)
        s_for_t_edge = jnp.clip((jnp.dot(d1, d2) * t_clamped - jnp.dot(d1, r)) / (a + eps), 0.0, 1.0)
        t_is_clamped = jnp.logical_or(t < 0.0, t > 1.0)
        s = jnp.where(t_is_clamped, s_for_t_edge, s)
        t = t_clamped

        c1 = p1 + d1 * s
        c2 = p2 + d2 * t
        diff = c1 - c2
        return jnp.dot(diff, diff)

    def _right_arm_capsules(self, q):
        transforms = self.robot.joint_to_world_transforms(q)
        shoulder = transforms[self.right_shoulder_yaw_idx, :3, 3]
        elbow = transforms[self.right_elbow_idx, :3, 3]
        elbow_tf = transforms[self.right_elbow_idx]
        forearm_end = elbow_tf[:3, 3] + elbow_tf[:3, :3] @ self.right_forearm_capsule_offset

        capsule_a = jnp.vstack([shoulder, elbow])
        capsule_b = jnp.vstack([elbow, forearm_end])
        return capsule_a, capsule_b, self.right_arm_capsule_radii

    def _right_gripper_sphere(self, q):
        transforms = self.robot.joint_to_world_transforms(q)
        elbow_tf = transforms[self.right_elbow_idx]
        center = elbow_tf[:3, 3] + elbow_tf[:3, :3] @ self.right_gripper_sphere_offset
        return center, self.right_gripper_sphere_radius

    def h_1(self, z, capsule_a=None, capsule_b=None, capsule_radii=None, **kwargs):
        q = z[: self.num_joints]
        robot_a, robot_b, robot_radii = self._right_arm_capsules(q)
        gripper_center, gripper_radius = self._right_gripper_sphere(q)

        if capsule_a is None:
            capsule_a = self.capsule_a
        if capsule_b is None:
            capsule_b = self.capsule_b
        if capsule_radii is None:
            capsule_radii = self.capsule_radii

        h_values = []

        for robot_idx in range(robot_a.shape[0]):
            for human_idx in range(capsule_a.shape[0]):
                combined_radius = robot_radii[robot_idx] + capsule_radii[human_idx]
                dist_sq = self._segment_segment_distance_sq(
                    robot_a[robot_idx],
                    robot_b[robot_idx],
                    capsule_a[human_idx],
                    capsule_b[human_idx],
                )
                h_values.append(dist_sq - combined_radius ** 2)

        for human_idx in range(capsule_a.shape[0]):
            combined_radius = gripper_radius + capsule_radii[human_idx]
            dist_sq = self._segment_segment_distance_sq(
                gripper_center,
                gripper_center,
                capsule_a[human_idx],
                capsule_b[human_idx],
            )
            h_values.append(dist_sq - combined_radius ** 2)

        return jnp.asarray(h_values)

    def alpha(self, h):
        return jnp.asarray(self.alpha_gain, dtype=h.dtype) * h

    def alpha_2(self, h_2):
        return jnp.asarray(self.alpha_gain, dtype=h_2.dtype) * h_2

    def _P(self, z):
        q = z[: self.num_joints]

        transforms = self.robot.joint_to_world_transforms(q)

        # These should already return controlled-space matrices:
        # J: (6, M)
        # M: (M, M)
        J = self.robot._ee_jacobian(transforms)
        M = self.robot._mass_matrix(transforms)

        M_inv = jnp.linalg.inv(M)

        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.pinv(task_inertia_inv)

        J_bar = M_inv @ J.T @ task_inertia
        N = jnp.eye(self.num_controls) - J_bar @ J

        W_T_W_joint = jnp.diag(jnp.asarray(self.W_T_W_joint_diag))
        W_T_W_task = jnp.diag(jnp.asarray(self.W_T_W_task_diag))

        return N.T @ W_T_W_joint @ N + J.T @ W_T_W_task @ J

    def P(self, z, u_des, *args, **kwargs):
        return self._P(z)

    def q(self, z, u_des, *args, **kwargs):
        return -u_des.T @ self._P(z)


class OSCBFPelvisArmHumanCapsuleVelocityConfig(OSCBFEEHumanCapsuleVelocityConfig):
    """
    World-frame velocity-level CBF for floating pelvis + arm control.

    State:
        z = [pelvis_x, pelvis_y, pelvis_z, pelvis_yaw, q_urdf], shape (4 + N,)

    Control:
        u = [pelvis_vel_xyz_yaw, qdot_arm_ctrl], shape (4 + M_arm,)

    The URDF still provides arm FK, but h is evaluated in world frame after a
    differentiable pelvis transform. This makes h directly sensitive to moving
    the robot body, not only to changing the arm joints.
    """

    def __init__(
        self,
        robot,
        capsule_a_init,
        capsule_b_init,
        capsule_radii_init,
        pelvis_to_urdf_transform,
        alpha_gain: float = 10.0,
        pelvis_velocity_limits=(0.6, 0.6, 0.4, 1.5),
        pelvis_obj_weight: float = 0.5,
        arm_obj_weight: float = 1.0,
    ):
        self.robot = robot
        self.num_joints = robot.num_joints
        self.num_controls = robot.num_controls
        self.pelvis_dim = 4
        self.task_dim = 6
        self.is_redundant = True
        self.alpha_gain = float(alpha_gain)
        self.pelvis_obj_weight = float(pelvis_obj_weight)
        self.arm_obj_weight = float(arm_obj_weight)
        self.pelvis_to_urdf_transform = jnp.asarray(
            pelvis_to_urdf_transform,
            dtype=jnp.float32,
        )
        self.pelvis_velocity_limits = tuple(float(x) for x in pelvis_velocity_limits)

        # Reuse the same conservative robot envelope as the arm-only config and
        # model the missing wrist segment explicitly as a forearm capsule.
        self.right_forearm_capsule_offset = jnp.asarray(
            [0.30, 0.0, 0.0],
            dtype=jnp.float32,
        )
        self.right_arm_contact_margin = 0.065
        self.right_arm_capsule_radii = jnp.asarray(
            (
                0.03 + self.right_arm_contact_margin,
                0.025 + self.right_arm_contact_margin,
            ),
            dtype=jnp.float32,
        )
        self.right_gripper_sphere_offset = jnp.asarray(
            [0.36, 0.0, 0.0],
            dtype=jnp.float32,
        )
        self.right_gripper_sphere_radius = jnp.asarray(0.13, dtype=jnp.float32)
        self.right_shoulder_yaw_idx = self.robot.joint_index(
            "right_shoulder_yaw_joint"
        )
        self.right_elbow_idx = self.robot.joint_index("right_elbow_joint")

        self.set_human_capsules(capsule_a_init, capsule_b_init, capsule_radii_init)

        ctrl = np.asarray(self.robot.controlled_joint_indices, dtype=np.int64)
        arm_max_velocities = np.asarray(self.robot.joint_max_velocities)[ctrl]
        u_max = np.concatenate(
            [np.asarray(self.pelvis_velocity_limits, dtype=float), arm_max_velocities],
            axis=0,
        )

        CBFConfig.__init__(
            self,
            n=self.pelvis_dim + self.num_joints,
            m=self.pelvis_dim + self.num_controls,
            u_min=-u_max,
            u_max=u_max,
            init_args=(self.capsule_a, self.capsule_b, self.capsule_radii),
        )

    def _world_from_urdf_points(self, pelvis, points_urdf):
        xyz = pelvis[:3]
        yaw = pelvis[3]
        cy = jnp.cos(yaw)
        sy = jnp.sin(yaw)
        r_world_pelvis = jnp.asarray(
            (
                (cy, -sy, 0.0),
                (sy, cy, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=points_urdf.dtype,
        )
        r_pelvis_urdf = self.pelvis_to_urdf_transform[:3, :3]
        t_pelvis_urdf = self.pelvis_to_urdf_transform[:3, 3]
        points_pelvis = points_urdf @ r_pelvis_urdf.T + t_pelvis_urdf
        return points_pelvis @ r_world_pelvis.T + xyz

    def _right_arm_capsules_world(self, pelvis, q_urdf):
        robot_a, robot_b, robot_radii = self._right_arm_capsules(q_urdf)
        gripper_center, gripper_radius = self._right_gripper_sphere(q_urdf)
        robot_a_world = self._world_from_urdf_points(pelvis, robot_a)
        robot_b_world = self._world_from_urdf_points(pelvis, robot_b)
        gripper_center_world = self._world_from_urdf_points(
            pelvis,
            gripper_center.reshape(1, 3),
        )[0]
        return robot_a_world, robot_b_world, robot_radii, gripper_center_world, gripper_radius

    def f(self, z, *args, **kwargs):
        return jnp.zeros((self.pelvis_dim + self.num_joints,), dtype=z.dtype)

    def g(self, z, *args, **kwargs):
        ctrl = jnp.asarray(self.robot.controlled_joint_indices, dtype=jnp.int32)
        col = jnp.arange(self.num_controls, dtype=jnp.int32)
        g = jnp.zeros(
            (self.pelvis_dim + self.num_joints, self.pelvis_dim + self.num_controls),
            dtype=z.dtype,
        )
        g = g.at[: self.pelvis_dim, : self.pelvis_dim].set(
            jnp.eye(self.pelvis_dim, dtype=z.dtype)
        )
        g = g.at[self.pelvis_dim + ctrl, self.pelvis_dim + col].set(
            jnp.asarray(1.0, dtype=z.dtype)
        )
        return g

    def h_1(self, z, capsule_a=None, capsule_b=None, capsule_radii=None, **kwargs):
        pelvis = z[: self.pelvis_dim]
        q_urdf = z[self.pelvis_dim : self.pelvis_dim + self.num_joints]
        robot_a, robot_b, robot_radii, gripper_center, gripper_radius = (
            self._right_arm_capsules_world(pelvis, q_urdf)
        )

        if capsule_a is None:
            capsule_a = self.capsule_a
        if capsule_b is None:
            capsule_b = self.capsule_b
        if capsule_radii is None:
            capsule_radii = self.capsule_radii

        h_values = []
        for robot_idx in range(robot_a.shape[0]):
            for human_idx in range(capsule_a.shape[0]):
                combined_radius = robot_radii[robot_idx] + capsule_radii[human_idx]
                dist_sq = self._segment_segment_distance_sq(
                    robot_a[robot_idx],
                    robot_b[robot_idx],
                    capsule_a[human_idx],
                    capsule_b[human_idx],
                )
                h_values.append(dist_sq - combined_radius ** 2)

        for human_idx in range(capsule_a.shape[0]):
            combined_radius = gripper_radius + capsule_radii[human_idx]
            dist_sq = self._segment_segment_distance_sq(
                gripper_center,
                gripper_center,
                capsule_a[human_idx],
                capsule_b[human_idx],
            )
            h_values.append(dist_sq - combined_radius ** 2)

        return jnp.asarray(h_values)

    def _P_augmented(self, z):
        weights = jnp.concatenate(
            [
                jnp.ones((self.pelvis_dim,), dtype=z.dtype) * self.pelvis_obj_weight,
                jnp.ones((self.num_controls,), dtype=z.dtype) * self.arm_obj_weight,
            ]
        )
        return jnp.diag(weights ** 2)

    def P(self, z, u_des, *args, **kwargs):
        return self._P_augmented(z)

    def q(self, z, u_des, *args, **kwargs):
        return -u_des.T @ self._P_augmented(z)
