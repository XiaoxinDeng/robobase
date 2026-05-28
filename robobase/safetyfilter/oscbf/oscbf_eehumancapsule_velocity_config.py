import numpy as np
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

    def _point_segment_distance_sq(self, p, a, b):
        ab = b - a
        ap = p - a

        denom = jnp.dot(ab, ab) + 1e-8
        t = jnp.dot(ap, ab) / denom
        t = jnp.clip(t, 0.0, 1.0)

        closest = a + t * ab
        diff = p - closest

        return jnp.dot(diff, diff)

    def h_1(self, z, capsule_a=None, capsule_b=None, capsule_radii=None, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)

        if capsule_a is None:
            capsule_a = self.capsule_a
        if capsule_b is None:
            capsule_b = self.capsule_b
        if capsule_radii is None:
            capsule_radii = self.capsule_radii

        h_values = []

        for i in range(capsule_a.shape[0]):
            dist_sq = self._point_segment_distance_sq(
                ee_pos,
                capsule_a[i],
                capsule_b[i],
            )
            h_values.append(dist_sq - capsule_radii[i] ** 2)

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