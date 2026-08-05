from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter


logger = logging.getLogger(__name__)


_H_ROBOT_CAPSULE_PARTS = ("upper_arm", "forearm")
_H_HUMAN_CAPSULE_PARTS = ("human_upper_arm", "human_forearm")


def _jax_prepare_horizon_clearance_inputs(
    q_bigym_flat,
    capsule_a_world_flat,
    capsule_b_world_flat,
    bigym_state_arm_indices,
    urdf_arm_joint_indices,
    arm_sign,
    arm_offset,
    urdf_neutral_q,
    t_pelvis_urdf,
):
    q_arm_bigym = q_bigym_flat[:, bigym_state_arm_indices]
    q_arm_urdf = arm_sign[None, :] * q_arm_bigym + arm_offset[None, :]
    q_urdf = jnp.broadcast_to(
        urdf_neutral_q[None, :],
        (q_bigym_flat.shape[0], urdf_neutral_q.shape[0]),
    )
    q_urdf = q_urdf.at[:, urdf_arm_joint_indices].set(q_arm_urdf)

    xyz = q_bigym_flat[:, :3]
    yaw = q_bigym_flat[:, 3]
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    zeros = jnp.zeros_like(cy)
    ones = jnp.ones_like(cy)
    r_world_pelvis = jnp.stack(
        (
            jnp.stack((cy, -sy, zeros), axis=-1),
            jnp.stack((sy, cy, zeros), axis=-1),
            jnp.stack((zeros, zeros, ones), axis=-1),
        ),
        axis=-2,
    )
    r_pelvis_urdf = t_pelvis_urdf[:3, :3]
    t_pelvis_urdf_vec = t_pelvis_urdf[:3, 3]
    r_world_urdf = jnp.einsum("nij,jk->nik", r_world_pelvis, r_pelvis_urdf)
    t_world_urdf = xyz + jnp.einsum("nij,j->ni", r_world_pelvis, t_pelvis_urdf_vec)
    bottom = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, 0.0, 1.0), dtype=q_bigym_flat.dtype),
        (q_bigym_flat.shape[0], 1, 4),
    )
    t_world_urdf_h = jnp.concatenate(
        (jnp.concatenate((r_world_urdf, t_world_urdf[:, :, None]), axis=2), bottom),
        axis=1,
    )
    t_urdf_world_h = jnp.linalg.inv(t_world_urdf_h)

    ones_a = jnp.ones(capsule_a_world_flat.shape[:-1] + (1,), dtype=q_bigym_flat.dtype)
    ones_b = jnp.ones(capsule_b_world_flat.shape[:-1] + (1,), dtype=q_bigym_flat.dtype)
    capsule_a_world_h = jnp.concatenate((capsule_a_world_flat, ones_a), axis=-1)
    capsule_b_world_h = jnp.concatenate((capsule_b_world_flat, ones_b), axis=-1)
    capsule_a_urdf = jnp.einsum("nij,ncj->nci", t_urdf_world_h, capsule_a_world_h)[:, :, :3]
    capsule_b_urdf = jnp.einsum("nij,ncj->nci", t_urdf_world_h, capsule_b_world_h)[:, :, :3]
    return q_urdf, capsule_a_urdf, capsule_b_urdf

def _warmup_oscbf_cbf_paths(
    filt: OSCBFFilter,
    env,
    obs,
    q_full: np.ndarray,
    qd_full: np.ndarray,
    action: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "arm_cbf_built": False,
        "arm_cbf_compiled": False,
        "pelvis_cbf_built": False,
        "pelvis_cbf_compiled": False,
    }
    if filt is None or getattr(filt, "use_dummy_filter", False):
        info["skipped"] = True
        return info
    if getattr(filt, "oscbf_config", None) is None:
        info["skipped"] = True
        info["reason"] = "missing_oscbf_config"
        return info

    q_bigym = np.asarray(q_full, dtype=np.float32).reshape(-1)
    qd_bigym = np.asarray(qd_full, dtype=np.float32).reshape(-1)
    if action is None:
        action = np.zeros((16,), dtype=np.float32)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != 16:
        padded = np.zeros((16,), dtype=np.float32)
        n = min(action.size, padded.size)
        padded[:n] = action[:n]
        action = padded

    try:
        q_urdf, _qd_urdf, q_arm_bigym, q_arm_urdf = (
            filt._build_urdf_surrogate_state_from_bigym(q_bigym, qd_bigym)
        )
        human_obstacles = filt._extract_human_obstacles(env, obs)
        capsule_a_world = human_obstacles["capsule_a"]
        capsule_b_world = human_obstacles["capsule_b"]
        capsule_radii = human_obstacles["capsule_radii"]
        filt._validate_capsules(capsule_a_world, capsule_b_world, capsule_radii)

        t_world_urdf = filt._get_world_T_urdf_from_bigym_state(q_bigym)
        t_urdf_world = np.linalg.inv(t_world_urdf)
        capsule_a_urdf = filt._transform_points(t_urdf_world, capsule_a_world)
        capsule_b_urdf = filt._transform_points(t_urdf_world, capsule_b_world)
        filt._validate_capsules(capsule_a_urdf, capsule_b_urdf, capsule_radii)
    except Exception as exc:  # noqa: BLE001
        info["setup_error"] = str(exc)
        return info

    try:
        build_t0 = time.perf_counter()
        arm_cbf = filt._ensure_cbf()
        info["arm_cbf_build_time_ms"] = float(1000.0 * (time.perf_counter() - build_t0))
        info["arm_cbf_built"] = True
        filt.oscbf_config.set_human_capsules(
            capsule_a_urdf,
            capsule_b_urdf,
            capsule_radii,
        )
        u_arm_nom = filt._bigym_action_to_urdf_velocity(
            q_arm_bigym=q_arm_bigym,
            q_arm_urdf=q_arm_urdf,
            a_arm_bigym_nom=action[filt.bigym_action_arm_indices],
        )
        compile_t0 = time.perf_counter()
        arm_result = arm_cbf.safety_filter(
            jnp.asarray(q_urdf, dtype=jnp.float32),
            jnp.asarray(u_arm_nom, dtype=jnp.float32),
        )
        jax.block_until_ready(arm_result)
        info["arm_cbf_compile_time_ms"] = float(1000.0 * (time.perf_counter() - compile_t0))
        info["arm_cbf_compiled"] = True
    except Exception as exc:  # noqa: BLE001
        info["arm_cbf_error"] = str(exc)

    if getattr(filt, "pelvis_oscbf_config", None) is not None:
        try:
            build_t0 = time.perf_counter()
            pelvis_cbf = filt._ensure_pelvis_cbf()
            info["pelvis_cbf_build_time_ms"] = float(1000.0 * (time.perf_counter() - build_t0))
            info["pelvis_cbf_built"] = True
            filt.pelvis_oscbf_config.set_human_capsules(
                capsule_a_world,
                capsule_b_world,
                capsule_radii,
            )
            q_base_bigym = q_bigym[filt.bigym_state_base_indices]
            u_base_nom = filt._bigym_base_action_to_velocity(
                q_base_bigym=q_base_bigym,
                a_base_bigym_nom=action[filt.bigym_action_base_indices],
            )
            u_arm_nom = filt._bigym_action_to_urdf_velocity(
                q_arm_bigym=q_arm_bigym,
                q_arm_urdf=q_arm_urdf,
                a_arm_bigym_nom=action[filt.bigym_action_arm_indices],
            )
            z_aug = np.concatenate([q_base_bigym, q_urdf], axis=0).astype(np.float32)
            u_aug_nom = np.concatenate([u_base_nom, u_arm_nom], axis=0).astype(np.float32)
            compile_t0 = time.perf_counter()
            pelvis_result = pelvis_cbf.safety_filter(
                jnp.asarray(z_aug, dtype=jnp.float32),
                jnp.asarray(u_aug_nom, dtype=jnp.float32),
            )
            jax.block_until_ready(pelvis_result)
            info["pelvis_cbf_compile_time_ms"] = float(
                1000.0 * (time.perf_counter() - compile_t0)
            )
            info["pelvis_cbf_compiled"] = True
        except Exception as exc:  # noqa: BLE001
            info["pelvis_cbf_error"] = str(exc)

    info["total_time_ms"] = float(
        sum(
            float(info.get(key, 0.0) or 0.0)
            for key in (
                "arm_cbf_build_time_ms",
                "arm_cbf_compile_time_ms",
                "pelvis_cbf_build_time_ms",
                "pelvis_cbf_compile_time_ms",
            )
        )
    )
    return info

def _h_pair_label(pair_index: Optional[int], pair_count: Optional[int]) -> dict[str, Any]:
    if pair_index is None or pair_count is None:
        return {
            "h_argmin_robot_part": None,
            "h_argmin_human_part": None,
            "h_argmin_human_capsule_index": None,
            "h_argmin_human_arm_index": None,
        }
    pair_index = int(pair_index)
    pair_count = int(pair_count)
    if pair_index < 0 or pair_count <= 0:
        return {
            "h_argmin_robot_part": None,
            "h_argmin_human_part": None,
            "h_argmin_human_capsule_index": None,
            "h_argmin_human_arm_index": None,
        }

    human_capsule_count = pair_count // 3 if pair_count % 3 == 0 else None
    if human_capsule_count is None or human_capsule_count <= 0:
        return {
            "h_argmin_robot_part": f"pair_{pair_index}",
            "h_argmin_human_part": None,
            "h_argmin_human_capsule_index": None,
            "h_argmin_human_arm_index": None,
        }

    robot_capsule_pair_count = len(_H_ROBOT_CAPSULE_PARTS) * human_capsule_count
    if pair_index < robot_capsule_pair_count:
        robot_idx = pair_index // human_capsule_count
        human_idx = pair_index % human_capsule_count
        robot_part = (
            _H_ROBOT_CAPSULE_PARTS[robot_idx]
            if robot_idx < len(_H_ROBOT_CAPSULE_PARTS)
            else f"robot_capsule_{robot_idx}"
        )
    else:
        robot_part = "gripper"
        human_idx = pair_index - robot_capsule_pair_count

    if 0 <= human_idx < human_capsule_count:
        human_part = _H_HUMAN_CAPSULE_PARTS[human_idx % len(_H_HUMAN_CAPSULE_PARTS)]
        human_arm_index = human_idx // len(_H_HUMAN_CAPSULE_PARTS)
    else:
        human_part = None
        human_arm_index = None

    return {
        "h_argmin_robot_part": robot_part,
        "h_argmin_human_part": human_part,
        "h_argmin_human_capsule_index": int(human_idx) if human_part is not None else None,
        "h_argmin_human_arm_index": int(human_arm_index) if human_arm_index is not None else None,
    }

def _h_argmin_metadata(h_values, include_pair_values: bool = True) -> dict[str, Any]:
    try:
        arr = np.asarray(h_values, dtype=np.float32)
    except Exception:  # noqa: BLE001
        arr = np.asarray([], dtype=np.float32)
    if arr.size == 0:
        return {}

    finite = np.isfinite(arr)
    if not np.any(finite):
        return {}
    scored = np.where(finite, arr, np.inf)
    flat_scored = scored.reshape(-1)
    flat_argmin = int(np.argmin(flat_scored))

    if scored.ndim == 1:
        pair_count = int(scored.shape[0])
        pair_index = flat_argmin
        horizon_index = None
        pair_values = scored
    else:
        unraveled = np.unravel_index(flat_argmin, scored.shape)
        horizon_index = int(unraveled[-2]) if scored.ndim >= 2 else None
        pair_index = int(unraveled[-1])
        pair_count = int(scored.shape[-1])
        pair_values = scored.reshape(-1, pair_count)[flat_argmin // pair_count]

    metadata = {
        "h_argmin_horizon_index": horizon_index,
        "h_argmin_pair_index": pair_index,
        "h_argmin_value": float(flat_scored[flat_argmin]),
    }
    metadata.update(_h_pair_label(pair_index, pair_count))
    if include_pair_values:
        metadata["h_pair_values_at_argmin"] = pair_values.astype(float).tolist()
    return metadata

def _selected_trace_indices(length: int, trace_indices=None) -> list[int]:
    if length <= 0:
        return []
    if trace_indices is None:
        return [0]
    out = []
    for idx in trace_indices:
        try:
            i = int(idx)
        except Exception:  # noqa: BLE001
            continue
        if i < 0:
            i += int(length)
        if 0 <= i < int(length):
            out.append(i)
    return out or [0]


def _robot_h_compute_geometry_sequence(q_seq, oscbf, trace_indices=None):
    if oscbf is None:
        return None
    q_arr = np.asarray(q_seq, dtype=np.float32)
    if q_arr.ndim != 2 or q_arr.shape[0] == 0:
        return None
    capsules = []
    spheres = []
    try:
        for idx in _selected_trace_indices(q_arr.shape[0], trace_indices):
            q_bigym = q_arr[int(idx)]
            qd_bigym = np.zeros_like(q_bigym, dtype=np.float32)
            q_urdf, _, _, _ = oscbf._build_urdf_surrogate_state_from_bigym(
                q_bigym,
                qd_bigym,
            )
            t_world_urdf = oscbf._get_world_T_urdf_from_bigym_state(q_bigym)
            robot_a, robot_b, robot_radii = oscbf.oscbf_config._right_arm_capsules(
                jnp.asarray(q_urdf, dtype=jnp.float32)
            )
            gripper_center, gripper_radius = oscbf.oscbf_config._right_gripper_sphere(
                jnp.asarray(q_urdf, dtype=jnp.float32)
            )
            robot_a_world = oscbf._transform_points_homogeneous(
                t_world_urdf,
                np.asarray(robot_a, dtype=np.float32),
            )
            robot_b_world = oscbf._transform_points_homogeneous(
                t_world_urdf,
                np.asarray(robot_b, dtype=np.float32),
            )
            gripper_world = oscbf._transform_points_homogeneous(
                t_world_urdf,
                np.asarray(gripper_center, dtype=np.float32).reshape(1, 3),
            ).reshape(3)
            for seg_idx, (a, b, radius) in enumerate(
                zip(robot_a_world, robot_b_world, np.asarray(robot_radii, dtype=np.float32))
            ):
                capsules.append(
                    {
                        "trace_index": int(idx),
                        "segment_index": int(seg_idx),
                        "a": np.asarray(a, dtype=np.float32).astype(float).tolist(),
                        "b": np.asarray(b, dtype=np.float32).astype(float).tolist(),
                        "radius": float(radius),
                    }
                )
            spheres.append(
                {
                    "trace_index": int(idx),
                    "center": gripper_world.astype(float).tolist(),
                    "radius": float(np.asarray(gripper_radius, dtype=np.float32)),
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Robot h-compute geometry extraction failed: %s", exc)
        return None
    return {
        "frame": "mujoco_world_from_safety_model",
        "source": "oscbf_h_compute_robot_capsules",
        "capsules": capsules,
        "spheres": spheres,
    }


class HorizonOSCBFOperator:
    def __init__(
        self,
        oscbf: OSCBFFilter,
        min_clearance: float,
        dt: float = 0.05,
        predict_human_motion: bool = True,
        human_prediction_max_time: Optional[float] = 0.25,
        human_prediction_max_speed: Optional[float] = 3.0,
    ):
        self.oscbf = oscbf
        self.min_clearance = float(min_clearance)
        self.dt = float(dt)
        self.predict_human_motion = bool(predict_human_motion)
        self.human_prediction_max_time = (
            None
            if human_prediction_max_time is None or human_prediction_max_time <= 0
            else float(human_prediction_max_time)
        )
        self.human_prediction_max_speed = (
            None
            if human_prediction_max_speed is None or human_prediction_max_speed <= 0
            else float(human_prediction_max_speed)
        )
        self.env = None
        self.obs = None
        self.q_full = None
        self.qd_full = None
        self._prev_capsule_a_world = None
        self._prev_capsule_b_world = None
        self._prev_capsule_radii = None
        self._capsule_a_velocity_world = None
        self._capsule_b_velocity_world = None
        self._human_motion_prediction_available = False
        self._human_motion_prediction_speed = 0.0
        self._human_obstacles_cache = None
        self._human_rollout_cache = {}
        self._human_obstacle_extract_time_ms = 0.0
        self._human_obstacle_cache_hits = 0
        self._human_obstacle_cache_misses = 0
        self._batched_h_fn = jax.jit(
            jax.vmap(
                lambda q, capsule_a, capsule_b, capsule_radii: self.oscbf.oscbf_config.h_1(
                    q,
                    capsule_a=capsule_a,
                    capsule_b=capsule_b,
                    capsule_radii=capsule_radii,
                ),
                in_axes=(0, 0, 0, None),
            )
        )
        self._robot_pair_radii = self._static_robot_pair_radii()
        self._chunk_filter_fns = {}
        self._safechunk_gradient_scan_fn = None

    def _static_robot_pair_radii(self):
        """Robot radii in the same pair order as OSCBF h_1 outputs."""
        cfg = self.oscbf.oscbf_config
        if cfg is None:
            return np.asarray([], dtype=np.float32)
        robot_radii = np.asarray(
            getattr(cfg, "right_arm_capsule_radii", []),
            dtype=np.float32,
        ).reshape(-1)
        gripper_radius = getattr(cfg, "right_gripper_sphere_radius", None)
        if gripper_radius is not None:
            robot_radii = np.concatenate(
                [robot_radii, np.asarray([float(np.asarray(gripper_radius))], dtype=np.float32)],
                axis=0,
            )
        return robot_radii.astype(np.float32)

    def _combined_pair_radii(self, capsule_radii):
        """Combined robot+human radii matching the flattened raw h pair order."""
        robot_radii = np.asarray(self._robot_pair_radii, dtype=np.float32).reshape(-1)
        human_radii = np.asarray(capsule_radii, dtype=np.float32).reshape(-1)
        if robot_radii.size == 0 or human_radii.size == 0:
            return np.asarray([], dtype=np.float32)
        return (robot_radii[:, None] + human_radii[None, :]).reshape(-1).astype(np.float32)

    def _h_values_to_signed_clearances(self, h_values, capsule_radii):
        """Convert raw h = dist^2 - r^2 into signed linear clearance in meters."""
        h_arr = np.asarray(h_values, dtype=np.float32)
        pair_radii = self._combined_pair_radii(capsule_radii)
        if h_arr.ndim == 0 or pair_radii.size == 0 or h_arr.shape[-1] != pair_radii.size:
            return h_arr
        radii = pair_radii.reshape((1,) * (h_arr.ndim - 1) + (pair_radii.size,))
        dist_sq = np.maximum(h_arr + np.square(radii), 0.0)
        return (np.sqrt(dist_sq) - radii).astype(np.float32)

    def _ensure_safechunk_gradient_scan_fn(self):
        if self._safechunk_gradient_scan_fn is not None:
            return self._safechunk_gradient_scan_fn
        h_fn = self._batched_h_fn
        robot_pair_radii = jnp.asarray(self._robot_pair_radii, dtype=jnp.float32)

        @jax.jit
        def _optimize(
            nominal_chunk,
            q0,
            action_idx,
            state_idx,
            mode_ids,
            initial_ctrl,
            initial_valid,
            directions,
            line_scales,
            capsule_a_world_seq,
            capsule_b_world_seq,
            capsule_radii,
            bigym_state_arm_indices,
            urdf_arm_joint_indices,
            arm_sign,
            arm_offset,
            urdf_neutral_q,
            t_pelvis_urdf,
            nominal_q_seq,
            q_future,
            q_future_valid,
            q_weights,
            rejoin_state_idx,
            target_chunk,
            target_q_seq,
            min_clearance,
            max_action_delta,
            action_low,
            action_high,
            dt,
            rollout_gain,
            opt_lr,
            gradient_eps,
            beta1,
            beta2,
            min_improvement,
            lambda_safety,
            lambda_action,
            lambda_path,
            lambda_smooth,
            lambda_rejoin,
            lambda_retreat,
            use_path,
            use_rejoin,
            use_ordered,
            use_recover_task,
            recover_task_progress_weight,
            recover_rejoin_weight,
            recover_direction_alignment_weight,
            recover_direction_threshold,
            recover_direction_margin,
            recover_ordered_pose_weight,
            recover_ordered_delta_weight,
        ):
            horizon = nominal_chunk.shape[0]

            def project_ctrl(ctrl_batch):
                candidates = jnp.broadcast_to(
                    nominal_chunk[None, :, :],
                    (ctrl_batch.shape[0],) + nominal_chunk.shape,
                )
                candidates = candidates.at[:, :, action_idx].set(ctrl_batch)
                nominal_ctrl = nominal_chunk[None, :, action_idx]
                delta = candidates[:, :, action_idx] - nominal_ctrl
                clipped_delta = jnp.clip(delta, -max_action_delta, max_action_delta)
                ctrl = nominal_ctrl + clipped_delta
                ctrl = jnp.clip(ctrl, action_low, action_high)
                return candidates.at[:, :, action_idx].set(ctrl)

            def rollout(candidates):
                q = jnp.broadcast_to(q0[None, :], (candidates.shape[0], q0.shape[0]))
                actions_by_time = jnp.swapaxes(candidates, 0, 1)

                def step(q_prev, actions_t):
                    selected = actions_t[:, action_idx]
                    current = q_prev[:, state_idx]
                    absolute = current + rollout_gain * (selected - current)
                    delta = current + rollout_gain * selected
                    velocity = current + rollout_gain * dt * selected
                    modes = mode_ids[None, :]
                    updated = jnp.where(
                        modes == 0,
                        absolute,
                        jnp.where(modes == 1, delta, velocity),
                    )
                    q_next = q_prev.at[:, state_idx].set(updated)
                    return q_next, q_next

                _, q_seq_time_major = jax.lax.scan(step, q, actions_by_time)
                return jnp.swapaxes(q_seq_time_major, 0, 1)

            def clearance_sequence(q_seq_batch):
                batch = q_seq_batch.shape[0]
                q_bigym_flat = q_seq_batch.reshape(batch * horizon, q_seq_batch.shape[-1])
                capsule_a_world_flat = jnp.broadcast_to(
                    capsule_a_world_seq[None, :, :, :],
                    (batch,) + capsule_a_world_seq.shape,
                ).reshape(batch * horizon, capsule_a_world_seq.shape[1], 3)
                capsule_b_world_flat = jnp.broadcast_to(
                    capsule_b_world_seq[None, :, :, :],
                    (batch,) + capsule_b_world_seq.shape,
                ).reshape(batch * horizon, capsule_b_world_seq.shape[1], 3)
                q_urdf, capsule_a_urdf, capsule_b_urdf = _jax_prepare_horizon_clearance_inputs(
                    q_bigym_flat,
                    capsule_a_world_flat,
                    capsule_b_world_flat,
                    bigym_state_arm_indices,
                    urdf_arm_joint_indices,
                    arm_sign,
                    arm_offset,
                    urdf_neutral_q,
                    t_pelvis_urdf,
                )
                h_values = h_fn(q_urdf, capsule_a_urdf, capsule_b_urdf, capsule_radii)
                combined_radii = (robot_pair_radii[:, None] + capsule_radii[None, :]).reshape(-1)
                signed_clearances = jnp.sqrt(
                    jnp.maximum(h_values + jnp.square(combined_radii)[None, :], 0.0)
                ) - combined_radii[None, :]
                return jnp.min(signed_clearances.reshape(batch, horizon, -1), axis=2)

            def ordered_loss_terms(q_seq_batch, j_best_rel):
                target_len = target_q_seq.shape[0]
                terminal_index = j_best_rel
                terminal_index = jnp.where(terminal_index < 0, horizon - 1, terminal_index)
                start = jnp.clip(terminal_index - horizon + 1, 0, jnp.maximum(0, target_len - horizon))
                start = jnp.where(use_recover_task > 0.0, 0, start)

                def one(candidate_q, start_idx):
                    nominal_slice = jax.lax.dynamic_slice(
                        target_q_seq,
                        (start_idx.astype(jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
                        (horizon, target_q_seq.shape[1]),
                    )
                    cand = candidate_q[:, state_idx]
                    nom = nominal_slice[:, state_idx]
                    diff = cand - nom
                    pose = jnp.square(diff).mean()
                    cand_delta = cand[1:] - cand[:-1]
                    nom_delta = nom[1:] - nom[:-1]
                    delta_loss = jnp.where(
                        horizon > 1,
                        jnp.square(cand_delta - nom_delta).mean(),
                        jnp.asarray(0.0, dtype=candidate_q.dtype),
                    )
                    return recover_ordered_pose_weight * pose + recover_ordered_delta_weight * delta_loss

                losses = jax.vmap(one)(q_seq_batch, start.astype(jnp.int32))
                return jnp.where(use_ordered > 0.0, losses, jnp.zeros_like(losses))

            def cost_ctrl_batch(ctrl_batch):
                candidates = project_ctrl(ctrl_batch)
                q_seq_batch = rollout(candidates)
                h_seq = clearance_sequence(q_seq_batch)
                safety_loss = jnp.square(jnp.maximum(min_clearance - h_seq, 0.0)).sum(axis=1)
                controlled_delta = candidates[:, :, action_idx] - nominal_chunk[None, :, action_idx]
                action_deviation_loss = jnp.square(controlled_delta).mean(axis=(1, 2))
                if horizon <= 1:
                    smoothness_loss = jnp.zeros((candidates.shape[0],), dtype=jnp.float32)
                else:
                    controlled = candidates[:, :, action_idx]
                    velocity_loss = jnp.square(controlled[:, 1:, :] - controlled[:, :-1, :]).mean(axis=(1, 2))
                    if horizon <= 2:
                        smoothness_loss = velocity_loss
                    else:
                        acc = controlled[:, 2:, :] - 2.0 * controlled[:, 1:-1, :] + controlled[:, :-2, :]
                        smoothness_loss = velocity_loss + 0.5 * jnp.square(acc).mean(axis=(1, 2))
                path_delta = q_seq_batch - nominal_q_seq[None, :horizon, :]
                path_loss = jnp.where(
                    use_path > 0.0,
                    jnp.square(path_delta).mean(axis=(1, 2)),
                    jnp.zeros((candidates.shape[0],), dtype=jnp.float32),
                )
                final_state = q_seq_batch[:, -1, :][:, rejoin_state_idx]
                future_delta = (final_state[:, None, :] - q_future[None, :, :]) * q_weights[None, None, :]
                future_losses = jnp.square(future_delta).sum(axis=2)
                future_losses = jnp.where(q_future_valid[None, :], future_losses, jnp.inf)
                j_best_rel = jnp.argmin(future_losses, axis=1).astype(jnp.int32)
                best_rejoin_loss = jnp.take_along_axis(future_losses, j_best_rel[:, None], axis=1)[:, 0]
                future_available = jnp.any(q_future_valid)
                rejoin_active = (use_rejoin > 0.0) & future_available
                rejoin_loss = jnp.where(
                    rejoin_active,
                    best_rejoin_loss,
                    jnp.zeros((candidates.shape[0],), dtype=jnp.float32),
                )
                j_best_rel = jnp.where(rejoin_active, j_best_rel, jnp.full_like(j_best_rel, -1))
                finite_h = jnp.nan_to_num(h_seq, nan=0.0, posinf=min_clearance, neginf=-1.0)
                retreat_loss = -jnp.mean(jnp.clip(finite_h, -1.0, 1.0), axis=1)
                ordered_loss = ordered_loss_terms(q_seq_batch, j_best_rel)

                progress_delta = candidates[:, 0, action_idx] - q0[state_idx][None, :]
                progress_score = jnp.linalg.norm(progress_delta, axis=1)
                delta_cand = candidates[:, 0, action_idx]
                delta_nom = jnp.broadcast_to(target_chunk[None, 0, action_idx], delta_cand.shape)
                absolute = mode_ids == 0
                q_state = q0[state_idx]
                delta_cand = jnp.where(absolute[None, :], delta_cand - q_state[None, :], delta_cand)
                delta_nom = jnp.where(absolute[None, :], delta_nom - q_state[None, :], delta_nom)
                dot = jnp.sum(delta_cand * delta_nom, axis=1)
                nominal_norm = jnp.linalg.norm(delta_nom, axis=1)
                candidate_norm = jnp.linalg.norm(delta_cand, axis=1)
                projection = dot / (jnp.square(nominal_norm) + 1e-9)
                cosine = dot / (candidate_norm * nominal_norm + 1e-9)
                nominal_rejoin_score = jnp.maximum(0.0, projection)
                direction_loss = jnp.square(
                    jnp.maximum(0.0, recover_direction_threshold + recover_direction_margin - cosine)
                )
                stalled_penalty = jnp.where(
                    (progress_score <= 0.0) & (nominal_rejoin_score <= 0.0),
                    5.0,
                    0.0,
                )
                recover_extra_loss = (
                    recover_direction_alignment_weight * direction_loss
                    + ordered_loss
                    - (recover_task_progress_weight * progress_score + recover_rejoin_weight * nominal_rejoin_score - stalled_penalty)
                )
                existing_loss = (
                    lambda_safety * safety_loss
                    + lambda_action * action_deviation_loss
                    + lambda_path * path_loss
                    + lambda_smooth * smoothness_loss
                )
                total = existing_loss + lambda_rejoin * rejoin_loss + lambda_retreat * retreat_loss
                total = jnp.where(use_ordered > 0.0, total + ordered_loss, total)
                total = jnp.where(use_recover_task > 0.0, existing_loss + recover_extra_loss, total)
                return total, candidates

            initial_costs, initial_candidates = cost_ctrl_batch(initial_ctrl)
            initial_costs = jnp.where(initial_valid, initial_costs, jnp.inf)
            initial_idx = jnp.argmin(initial_costs)
            current_ctrl = initial_ctrl[initial_idx]
            current_cost = initial_costs[initial_idx]
            best_ctrl = current_ctrl
            best_cost = current_cost
            m = jnp.zeros_like(current_ctrl)
            v = jnp.zeros_like(current_ctrl)

            def scan_step(carry, step_input):
                current_ctrl, current_cost, best_ctrl, best_cost, m, v, iteration = carry
                dirs = step_input
                plus = current_ctrl[None, :, :] + gradient_eps * dirs
                minus = current_ctrl[None, :, :] - gradient_eps * dirs
                paired_ctrl = jnp.stack([plus, minus], axis=1).reshape((dirs.shape[0] * 2,) + current_ctrl.shape)
                paired_costs, paired_candidates = cost_ctrl_batch(paired_ctrl)
                finite_pair_costs = jnp.where(jnp.isfinite(paired_costs), paired_costs, jnp.inf)
                pair_best_idx = jnp.argmin(finite_pair_costs)
                pair_best_cost = finite_pair_costs[pair_best_idx]
                pair_best_ctrl = jnp.take(paired_candidates[pair_best_idx], action_idx, axis=1)
                improve_best_pair = pair_best_cost + min_improvement < best_cost
                best_ctrl = jnp.where(improve_best_pair, pair_best_ctrl, best_ctrl)
                best_cost = jnp.where(improve_best_pair, pair_best_cost, best_cost)

                costs = paired_costs.reshape((dirs.shape[0], 2))
                coeff = jnp.where(
                    jnp.isfinite(costs[:, 0]) & jnp.isfinite(costs[:, 1]),
                    (costs[:, 0] - costs[:, 1]) / (2.0 * gradient_eps),
                    0.0,
                )
                grad = jnp.mean(coeff[:, None, None] * dirs, axis=0)
                grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_norm = jnp.linalg.norm(grad)
                grad = jnp.where(grad_norm > 1e6, grad * (1e6 / grad_norm), grad)
                m_next = beta1 * m + (1.0 - beta1) * grad
                v_next = beta2 * v + (1.0 - beta2) * jnp.square(grad)
                t = iteration.astype(jnp.float32)
                m_hat = m_next / jnp.maximum(1e-9, 1.0 - beta1 ** t)
                v_hat = v_next / jnp.maximum(1e-9, 1.0 - beta2 ** t)
                update = opt_lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
                trial_ctrl = current_ctrl[None, :, :] - line_scales[:, None, None] * update
                trial_costs, trial_candidates = cost_ctrl_batch(trial_ctrl)
                finite_trial_costs = jnp.where(jnp.isfinite(trial_costs), trial_costs, jnp.inf)
                trial_best_idx = jnp.argmin(finite_trial_costs)
                trial_best_cost = finite_trial_costs[trial_best_idx]
                trial_best_ctrl = jnp.take(trial_candidates[trial_best_idx], action_idx, axis=1)
                improve_best_trial = trial_best_cost + min_improvement < best_cost
                best_ctrl = jnp.where(improve_best_trial, trial_best_ctrl, best_ctrl)
                best_cost = jnp.where(improve_best_trial, trial_best_cost, best_cost)

                trial_improved = trial_costs + min_improvement < current_cost
                any_trial = jnp.any(trial_improved)
                first_trial_idx = jnp.argmax(trial_improved.astype(jnp.int32))
                accepted_trial_ctrl = trial_ctrl[first_trial_idx]
                accepted_trial_cost = trial_costs[first_trial_idx]
                pair_improved = pair_best_cost + min_improvement < current_cost
                next_ctrl = jnp.where(
                    any_trial,
                    accepted_trial_ctrl,
                    jnp.where(pair_improved, pair_best_ctrl, current_ctrl),
                )
                next_cost = jnp.where(
                    any_trial,
                    accepted_trial_cost,
                    jnp.where(pair_improved, pair_best_cost, current_cost),
                )
                ctrl_dtype = current_ctrl.dtype
                cost_dtype = current_cost.dtype
                next_ctrl = next_ctrl.astype(ctrl_dtype)
                best_ctrl = best_ctrl.astype(ctrl_dtype)
                m_next = m_next.astype(m.dtype)
                v_next = v_next.astype(v.dtype)
                next_cost = next_cost.astype(cost_dtype)
                best_cost = best_cost.astype(cost_dtype)
                return (next_ctrl, next_cost, best_ctrl, best_cost, m_next, v_next, iteration + 1), None

            init_carry = (
                current_ctrl,
                current_cost,
                best_ctrl,
                best_cost,
                m,
                v,
                jnp.asarray(1, dtype=jnp.int32),
            )
            final_carry, _ = jax.lax.scan(scan_step, init_carry, directions)
            _current_ctrl, _current_cost, best_ctrl, best_cost, _m, _v, _iteration = final_carry
            best_chunk = project_ctrl(best_ctrl[None, :, :])[0]
            optimizer_evaluations = initial_ctrl.shape[0] + directions.shape[0] * (2 * directions.shape[1] + line_scales.shape[0])
            return best_chunk, best_cost, optimizer_evaluations

        self._safechunk_gradient_scan_fn = _optimize
        return _optimize

    def safechunk_jax_scan_optimize(self, *, obs=None, **kwargs):
        nominal_chunk = np.asarray(kwargs.pop("nominal_chunk"), dtype=np.float32)
        horizon = int(nominal_chunk.shape[0])
        (
            capsule_a_world_seq,
            capsule_b_world_seq,
            capsule_radii_eval,
            _prediction_info,
        ) = self._human_capsule_rollout_cached(obs if obs is not None else self.obs, horizon)
        fn = self._ensure_safechunk_gradient_scan_fn()
        result = fn(
            jnp.asarray(nominal_chunk, dtype=jnp.float32),
            jnp.asarray(kwargs.pop("q0"), dtype=jnp.float32),
            jnp.asarray(kwargs.pop("action_idx"), dtype=jnp.int32),
            jnp.asarray(kwargs.pop("state_idx"), dtype=jnp.int32),
            jnp.asarray(kwargs.pop("mode_ids"), dtype=jnp.int32),
            jnp.asarray(kwargs.pop("initial_ctrl"), dtype=jnp.float32),
            jnp.asarray(kwargs.pop("initial_valid"), dtype=jnp.bool_),
            jnp.asarray(kwargs.pop("directions"), dtype=jnp.float32),
            jnp.asarray(kwargs.pop("line_scales"), dtype=jnp.float32),
            jnp.asarray(capsule_a_world_seq, dtype=jnp.float32),
            jnp.asarray(capsule_b_world_seq, dtype=jnp.float32),
            jnp.asarray(capsule_radii_eval, dtype=jnp.float32),
            jnp.asarray(self.oscbf.bigym_state_arm_indices, dtype=jnp.int32),
            jnp.asarray(self.oscbf.urdf_arm_joint_indices, dtype=jnp.int32),
            jnp.asarray(self.oscbf.arm_sign, dtype=jnp.float32),
            jnp.asarray(self.oscbf.arm_offset, dtype=jnp.float32),
            jnp.asarray(self.oscbf.urdf_neutral_q, dtype=jnp.float32),
            jnp.asarray(self.oscbf.T_pelvis_urdf, dtype=jnp.float32),
            jnp.asarray(kwargs.pop("nominal_q_seq"), dtype=jnp.float32),
            jnp.asarray(kwargs.pop("q_future"), dtype=jnp.float32),
            jnp.asarray(kwargs.pop("q_future_valid"), dtype=jnp.bool_),
            jnp.asarray(kwargs.pop("q_weights"), dtype=jnp.float32),
            jnp.asarray(kwargs.pop("rejoin_state_idx"), dtype=jnp.int32),
            jnp.asarray(kwargs.pop("target_chunk"), dtype=jnp.float32),
            jnp.asarray(kwargs.pop("target_q_seq"), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("min_clearance")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("max_action_delta")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("action_low")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("action_high")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("dt")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("rollout_gain")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("opt_lr")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("gradient_eps")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("beta1")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("beta2")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("min_improvement")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("lambda_safety")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("lambda_action")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("lambda_path")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("lambda_smooth")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("lambda_rejoin")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("lambda_retreat")), dtype=jnp.float32),
            jnp.asarray(1.0 if kwargs.pop("use_path") else 0.0, dtype=jnp.float32),
            jnp.asarray(1.0 if kwargs.pop("use_rejoin") else 0.0, dtype=jnp.float32),
            jnp.asarray(1.0 if kwargs.pop("use_ordered") else 0.0, dtype=jnp.float32),
            jnp.asarray(1.0 if kwargs.pop("use_recover_task") else 0.0, dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("recover_task_progress_weight")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("recover_rejoin_weight")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("recover_direction_alignment_weight")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("recover_direction_threshold")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("recover_direction_margin")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("recover_ordered_pose_weight")), dtype=jnp.float32),
            jnp.asarray(float(kwargs.pop("recover_ordered_delta_weight")), dtype=jnp.float32),
        )
        if kwargs:
            raise ValueError(f"Unused safechunk_jax_scan_optimize kwargs: {sorted(kwargs)}")
        best_chunk, best_cost, optimizer_evaluations = result
        jax.block_until_ready(result)
        return {
            "best_chunk": np.asarray(best_chunk, dtype=np.float32),
            "best_cost": float(np.asarray(best_cost)),
            "optimizer_evaluations": int(np.asarray(optimizer_evaluations)),
        }

    def set_context(self, env, obs, q_full: np.ndarray, qd_full: np.ndarray):
        self.env = env
        self.obs = obs
        self.q_full = np.asarray(q_full, dtype=np.float32).reshape(-1)
        self.qd_full = np.asarray(qd_full, dtype=np.float32).reshape(-1)
        self._human_obstacles_cache = None
        self._human_rollout_cache = {}
        self._human_obstacle_extract_time_ms = 0.0
        self._human_obstacle_cache_hits = 0
        self._human_obstacle_cache_misses = 0
        self._update_human_capsule_velocity()

    def reset_human_motion_prediction(self):
        self._prev_capsule_a_world = None
        self._prev_capsule_b_world = None
        self._prev_capsule_radii = None
        self._capsule_a_velocity_world = None
        self._capsule_b_velocity_world = None
        self._human_motion_prediction_available = False
        self._human_motion_prediction_speed = 0.0
        self._human_obstacles_cache = None
        self._human_rollout_cache = {}
        self._human_obstacle_extract_time_ms = 0.0
        self._human_obstacle_cache_hits = 0
        self._human_obstacle_cache_misses = 0

    def _limit_capsule_velocity(self, velocity):
        if self.human_prediction_max_speed is None:
            return velocity
        velocity = np.asarray(velocity, dtype=np.float32)
        norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
        scale = np.minimum(
            1.0,
            self.human_prediction_max_speed / np.maximum(norm, 1e-9),
        )
        return velocity * scale

    def _update_human_capsule_velocity(self):
        self._human_motion_prediction_available = False
        self._human_motion_prediction_speed = 0.0
        if self.env is None:
            return
        try:
            t0 = time.perf_counter()
            human_obstacles = self.oscbf._extract_human_obstacles(self.env, self.obs)
            self._human_obstacle_extract_time_ms = 1000.0 * (time.perf_counter() - t0)
            capsule_a = np.asarray(human_obstacles["capsule_a"], dtype=np.float32)
            capsule_b = np.asarray(human_obstacles["capsule_b"], dtype=np.float32)
            capsule_radii = np.asarray(
                human_obstacles["capsule_radii"],
                dtype=np.float32,
            )
            self._human_obstacles_cache = {
                "capsule_a": capsule_a.copy(),
                "capsule_b": capsule_b.copy(),
                "capsule_radii": capsule_radii.copy(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Human capsule velocity update failed: %s", exc)
            return

        if not self.predict_human_motion:
            self._prev_capsule_a_world = capsule_a.copy()
            self._prev_capsule_b_world = capsule_b.copy()
            self._prev_capsule_radii = capsule_radii.copy()
            return

        if (
            self._prev_capsule_a_world is not None
            and self._prev_capsule_b_world is not None
            and self._prev_capsule_radii is not None
            and capsule_a.shape == self._prev_capsule_a_world.shape
            and capsule_b.shape == self._prev_capsule_b_world.shape
            and capsule_radii.shape == self._prev_capsule_radii.shape
        ):
            dt = max(float(self.dt), 1e-6)
            a_velocity = (capsule_a - self._prev_capsule_a_world) / dt
            b_velocity = (capsule_b - self._prev_capsule_b_world) / dt
            a_velocity = self._limit_capsule_velocity(a_velocity)
            b_velocity = self._limit_capsule_velocity(b_velocity)
            self._capsule_a_velocity_world = a_velocity.astype(np.float32)
            self._capsule_b_velocity_world = b_velocity.astype(np.float32)
            endpoint_speeds = np.concatenate(
                [
                    np.linalg.norm(a_velocity, axis=-1),
                    np.linalg.norm(b_velocity, axis=-1),
                ]
            )
            self._human_motion_prediction_speed = float(np.max(endpoint_speeds))
            self._human_motion_prediction_available = bool(
                np.isfinite(self._human_motion_prediction_speed)
                and self._human_motion_prediction_speed > 1e-9
            )
        else:
            self._capsule_a_velocity_world = None
            self._capsule_b_velocity_world = None

        self._prev_capsule_a_world = capsule_a.copy()
        self._prev_capsule_b_world = capsule_b.copy()
        self._prev_capsule_radii = capsule_radii.copy()

    def _current_human_obstacles(self, obs=None):
        if self._human_obstacles_cache is not None:
            self._human_obstacle_cache_hits += 1
            return self._human_obstacles_cache, True
        self._human_obstacle_cache_misses += 1
        t0 = time.perf_counter()
        human_obstacles = self.oscbf._extract_human_obstacles(
            self.env,
            self.obs if obs is None else obs,
        )
        self._human_obstacle_extract_time_ms += 1000.0 * (time.perf_counter() - t0)
        capsule_a = np.asarray(human_obstacles["capsule_a"], dtype=np.float32)
        capsule_b = np.asarray(human_obstacles["capsule_b"], dtype=np.float32)
        capsule_radii = np.asarray(human_obstacles["capsule_radii"], dtype=np.float32)
        self._human_obstacles_cache = {
            "capsule_a": capsule_a.copy(),
            "capsule_b": capsule_b.copy(),
            "capsule_radii": capsule_radii.copy(),
        }
        return self._human_obstacles_cache, False

    def _human_capsule_rollout_cached(self, obs, horizon):
        horizon = int(horizon)
        cached = self._human_rollout_cache.get(horizon)
        if cached is not None:
            a_seq, b_seq, radii, info = cached
            info = dict(info)
            info.update(
                {
                    "human_obstacles_cached": True,
                    "human_rollout_cached": True,
                    "human_obstacle_cache_hits": int(self._human_obstacle_cache_hits),
                    "human_obstacle_cache_misses": int(self._human_obstacle_cache_misses),
                    "human_obstacle_extract_time_ms": float(self._human_obstacle_extract_time_ms),
                }
            )
            return a_seq, b_seq, radii, info

        human_obstacles, obstacle_cached = self._current_human_obstacles(obs)
        a_seq, b_seq, radii, info = self._human_capsule_rollout(
            human_obstacles["capsule_a"],
            human_obstacles["capsule_b"],
            human_obstacles["capsule_radii"],
            horizon,
        )
        info = dict(info)
        info.update(
            {
                "human_obstacles_cached": bool(obstacle_cached),
                "human_rollout_cached": False,
                "human_obstacle_cache_hits": int(self._human_obstacle_cache_hits),
                "human_obstacle_cache_misses": int(self._human_obstacle_cache_misses),
                "human_obstacle_extract_time_ms": float(self._human_obstacle_extract_time_ms),
            }
        )
        self._human_rollout_cache[horizon] = (
            np.asarray(a_seq, dtype=np.float32).copy(),
            np.asarray(b_seq, dtype=np.float32).copy(),
            np.asarray(radii, dtype=np.float32).copy(),
            dict(info),
        )
        return a_seq, b_seq, radii, info

    def _human_capsule_rollout(self, capsule_a_world, capsule_b_world, capsule_radii, horizon):
        capsule_a_world = np.asarray(capsule_a_world, dtype=np.float32)
        capsule_b_world = np.asarray(capsule_b_world, dtype=np.float32)
        capsule_radii = np.asarray(capsule_radii, dtype=np.float32)
        current_a = np.broadcast_to(
            capsule_a_world[None, :, :],
            (horizon,) + capsule_a_world.shape,
        ).copy()
        current_b = np.broadcast_to(
            capsule_b_world[None, :, :],
            (horizon,) + capsule_b_world.shape,
        ).copy()

        info = {
            "human_motion_prediction_enabled": bool(self.predict_human_motion),
            "human_motion_prediction_available": False,
            "human_motion_prediction_dt": float(self.dt),
            "human_motion_prediction_max_time": self.human_prediction_max_time,
            "human_motion_prediction_max_speed": self.human_prediction_max_speed,
            "human_motion_prediction_speed": float(self._human_motion_prediction_speed),
            "human_motion_prediction_max_displacement": 0.0,
        }
        if (
            not self.predict_human_motion
            or not self._human_motion_prediction_available
            or self._capsule_a_velocity_world is None
            or self._capsule_b_velocity_world is None
            or self._capsule_a_velocity_world.shape != capsule_a_world.shape
            or self._capsule_b_velocity_world.shape != capsule_b_world.shape
        ):
            return current_a, current_b, capsule_radii, info

        times = (np.arange(horizon, dtype=np.float32) + 1.0) * float(self.dt)
        if self.human_prediction_max_time is not None:
            times = np.minimum(times, float(self.human_prediction_max_time))
        predicted_a = (
            capsule_a_world[None, :, :]
            + times[:, None, None] * self._capsule_a_velocity_world[None, :, :]
        )
        predicted_b = (
            capsule_b_world[None, :, :]
            + times[:, None, None] * self._capsule_b_velocity_world[None, :, :]
        )
        capsule_a_seq = np.concatenate([current_a, predicted_a], axis=1)
        capsule_b_seq = np.concatenate([current_b, predicted_b], axis=1)
        capsule_radii_pred = np.concatenate([capsule_radii, capsule_radii], axis=0)
        info.update(
            {
                "human_motion_prediction_available": True,
                "human_motion_prediction_max_displacement": float(
                    self._human_motion_prediction_speed * float(np.max(times))
                ),
            }
        )
        return capsule_a_seq, capsule_b_seq, capsule_radii_pred, info

    def __call__(self, action, obs=None, **kwargs):
        return self.oscbf(
            action=action,
            env=kwargs.pop("env", self.env),
            observations=kwargs.pop("observations", obs if obs is not None else self.obs),
            q_full=kwargs.pop("q_full", self.q_full),
            qd_full=kwargs.pop("qd_full", self.qd_full),
            **kwargs,
        )

    def _ensure_chunk_filter_fn(self, use_pelvis_cbf: bool):
        key = "pelvis" if use_pelvis_cbf else "arm"
        cached = self._chunk_filter_fns.get(key)
        if cached is not None:
            return cached

        cbf = (
            self.oscbf._ensure_pelvis_cbf()
            if use_pelvis_cbf
            else self.oscbf._ensure_cbf()
        )

        @jax.jit
        def _filter_chunk(
            action_chunk,
            q0_bigym,
            capsule_a_world_seq,
            capsule_b_world_seq,
            capsule_radii,
            bigym_action_base_indices,
            bigym_action_arm_indices,
            bigym_action_clip_indices,
            bigym_state_base_indices,
            bigym_state_arm_indices,
            urdf_arm_joint_indices,
            rollout_state_indices,
            rollout_action_indices,
            rollout_mode_ids,
            arm_sign,
            arm_offset,
            urdf_neutral_q,
            t_pelvis_urdf,
            dt,
            control_mode_id,
            max_action_delta,
        ):
            r_pelvis_urdf = t_pelvis_urdf[:3, :3]
            t_pelvis_urdf_vec = t_pelvis_urdf[:3, 3]

            def urdf_state_and_world_pose(q_bigym):
                q_arm_bigym = q_bigym[bigym_state_arm_indices]
                q_arm_urdf = arm_sign * q_arm_bigym + arm_offset
                q_urdf = urdf_neutral_q.at[urdf_arm_joint_indices].set(q_arm_urdf)

                yaw = q_bigym[3]
                cy = jnp.cos(yaw)
                sy = jnp.sin(yaw)
                zeros = jnp.asarray(0.0, dtype=q_bigym.dtype)
                one = jnp.asarray(1.0, dtype=q_bigym.dtype)
                r_world_pelvis = jnp.stack(
                    (
                        jnp.stack((cy, -sy, zeros)),
                        jnp.stack((sy, cy, zeros)),
                        jnp.stack((zeros, zeros, one)),
                    ),
                    axis=0,
                )
                r_world_urdf = r_world_pelvis @ r_pelvis_urdf
                t_world_urdf = q_bigym[:3] + r_world_pelvis @ t_pelvis_urdf_vec
                return q_urdf, q_arm_bigym, q_arm_urdf, r_world_urdf, t_world_urdf

            def arm_action_to_urdf_velocity(action, q_arm_urdf):
                a_arm_bigym = action[bigym_action_arm_indices]
                a_arm_urdf = arm_sign * a_arm_bigym + arm_offset
                u_abs = (a_arm_urdf - q_arm_urdf) / dt
                u_delta = arm_sign * a_arm_bigym / dt
                u_velocity = arm_sign * a_arm_bigym
                return jnp.where(
                    control_mode_id == 0,
                    u_abs,
                    jnp.where(control_mode_id == 1, u_delta, u_velocity),
                )

            def urdf_velocity_to_arm_action(q_arm_urdf, u_arm_urdf):
                a_abs = arm_sign * (q_arm_urdf + u_arm_urdf * dt - arm_offset)
                a_delta = arm_sign * (u_arm_urdf * dt)
                a_velocity = arm_sign * u_arm_urdf
                return jnp.where(
                    control_mode_id == 0,
                    a_abs,
                    jnp.where(control_mode_id == 1, a_delta, a_velocity),
                )

            def base_action_to_velocity(action):
                a_base = action[bigym_action_base_indices]
                u_delta = a_base / dt
                return jnp.where(control_mode_id == 2, a_base, u_delta)

            def base_velocity_to_action(u_base):
                return jnp.where(control_mode_id == 2, u_base, u_base * dt)

            def rollout_step(q_bigym, action):
                selected = action[rollout_action_indices]
                current = q_bigym[rollout_state_indices]
                updated = jnp.where(
                    rollout_mode_ids == 0,
                    selected,
                    jnp.where(
                        rollout_mode_ids == 1,
                        current + selected,
                        current + dt * selected,
                    ),
                )
                return q_bigym.at[rollout_state_indices].set(updated)

            def step(q_bigym, step_inputs):
                action, capsule_a_world, capsule_b_world = step_inputs
                q_urdf, q_arm_bigym, q_arm_urdf, r_world_urdf, t_world_urdf = (
                    urdf_state_and_world_pose(q_bigym)
                )
                del q_arm_bigym

                if use_pelvis_cbf:
                    z = jnp.concatenate((q_bigym[bigym_state_base_indices], q_urdf), axis=0)
                    u_base_nom = base_action_to_velocity(action)
                    u_arm_nom = arm_action_to_urdf_velocity(action, q_arm_urdf)
                    u_aug_nom = jnp.concatenate((u_base_nom, u_arm_nom), axis=0)
                    u_aug_safe = cbf.safety_filter(
                        z,
                        u_aug_nom,
                        capsule_a_world,
                        capsule_b_world,
                        capsule_radii,
                    )
                    u_base_safe = u_aug_safe[: bigym_action_base_indices.shape[0]]
                    u_arm_safe = u_aug_safe[bigym_action_base_indices.shape[0] :]
                    safe_action = action
                    safe_action = safe_action.at[bigym_action_base_indices].set(
                        base_velocity_to_action(u_base_safe)
                    )
                    safe_action = safe_action.at[bigym_action_arm_indices].set(
                        urdf_velocity_to_arm_action(q_arm_urdf, u_arm_safe)
                    )
                else:
                    capsule_a_urdf = (capsule_a_world - t_world_urdf[None, :]) @ r_world_urdf
                    capsule_b_urdf = (capsule_b_world - t_world_urdf[None, :]) @ r_world_urdf
                    u_arm_nom = arm_action_to_urdf_velocity(action, q_arm_urdf)
                    u_arm_safe = cbf.safety_filter(
                        q_urdf,
                        u_arm_nom,
                        capsule_a_urdf,
                        capsule_b_urdf,
                        capsule_radii,
                    )
                    safe_action = action.at[bigym_action_arm_indices].set(
                        urdf_velocity_to_arm_action(q_arm_urdf, u_arm_safe)
                    )

                delta = safe_action[bigym_action_clip_indices] - action[bigym_action_clip_indices]
                clipped_delta = jnp.clip(delta, -max_action_delta, max_action_delta)
                safe_action = safe_action.at[bigym_action_clip_indices].set(
                    action[bigym_action_clip_indices] + clipped_delta
                )
                q_next = rollout_step(q_bigym, safe_action)
                return q_next, safe_action

            _, safe_actions = jax.lax.scan(
                step,
                q0_bigym,
                (action_chunk, capsule_a_world_seq, capsule_b_world_seq),
            )
            return safe_actions

        self._chunk_filter_fns[key] = _filter_chunk
        return _filter_chunk

    def _control_mode_id(self):
        if self.oscbf.control_type == "absolute":
            return 0
        if self.oscbf.control_type == "delta":
            return 1
        return 2

    def _chunk_rollout_mode_ids(self, state_indices):
        state_indices = np.asarray(state_indices, dtype=np.int64).reshape(-1)
        modes = np.full(state_indices.shape, self._control_mode_id(), dtype=np.int32)
        modes[state_indices < 4] = 1
        return modes

    def filter_chunk(self, action_chunk=None, obs=None, observations=None, **kwargs):
        if action_chunk is None:
            action_chunk = kwargs.pop("chunk", None)
        if action_chunk is None:
            raise ValueError("action_chunk must be provided")
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected action_chunk shape (H, A), got {chunk.shape}")
        if not self.oscbf.enabled:
            return chunk.copy(), {"jax_sequential_oscbf_used": False, "sequential_oscbf_passthrough": True}
        if self.oscbf.use_dummy_filter:
            raise RuntimeError("JAX chunk OSCBF is not used for dummy OSCBF filters")

        obs_eval = observations if observations is not None else (obs if obs is not None else self.obs)
        q_full = np.asarray(kwargs.pop("q_full", self.q_full), dtype=np.float32).reshape(-1)
        _ = np.asarray(kwargs.pop("qd_full", self.qd_full), dtype=np.float32).reshape(-1)
        if q_full.shape[0] != self.oscbf.expected_motion_dim:
            raise ValueError(
                f"Expected q_full dim {self.oscbf.expected_motion_dim}, got {q_full.shape[0]}"
            )

        use_pelvis_cbf = bool(
            getattr(self.oscbf, "enable_pelvis_cbf", False)
            and getattr(self.oscbf, "pelvis_oscbf_config", None) is not None
        )
        if use_pelvis_cbf:
            self.oscbf._ensure_pelvis_cbf()
            clip_indices = self.oscbf.bigym_action_safety_indices
        else:
            self.oscbf._ensure_cbf()
            clip_indices = self.oscbf.bigym_action_arm_indices

        (
            capsule_a_world_seq,
            capsule_b_world_seq,
            capsule_radii_eval,
            prediction_info,
        ) = self._human_capsule_rollout_cached(obs_eval, chunk.shape[0])

        valid = (
            (self.oscbf.bigym_state_safety_indices < q_full.shape[0])
            & (self.oscbf.bigym_action_safety_indices < chunk.shape[1])
        )
        rollout_state_indices = self.oscbf.bigym_state_safety_indices[valid].astype(np.int32)
        rollout_action_indices = self.oscbf.bigym_action_safety_indices[valid].astype(np.int32)
        if rollout_state_indices.size == 0:
            raise ValueError("No valid state/action indices for sequential OSCBF rollout")

        max_action_delta = (
            np.inf
            if self.oscbf.max_action_delta is None
            else float(self.oscbf.max_action_delta)
        )
        filter_fn = self._ensure_chunk_filter_fn(use_pelvis_cbf)
        t0 = time.perf_counter()
        safe_chunk = np.asarray(
            filter_fn(
                jnp.asarray(chunk, dtype=jnp.float32),
                jnp.asarray(q_full, dtype=jnp.float32),
                jnp.asarray(capsule_a_world_seq, dtype=jnp.float32),
                jnp.asarray(capsule_b_world_seq, dtype=jnp.float32),
                jnp.asarray(capsule_radii_eval, dtype=jnp.float32),
                jnp.asarray(self.oscbf.bigym_action_base_indices, dtype=jnp.int32),
                jnp.asarray(self.oscbf.bigym_action_arm_indices, dtype=jnp.int32),
                jnp.asarray(clip_indices, dtype=jnp.int32),
                jnp.asarray(self.oscbf.bigym_state_base_indices, dtype=jnp.int32),
                jnp.asarray(self.oscbf.bigym_state_arm_indices, dtype=jnp.int32),
                jnp.asarray(self.oscbf.urdf_arm_joint_indices, dtype=jnp.int32),
                jnp.asarray(rollout_state_indices, dtype=jnp.int32),
                jnp.asarray(rollout_action_indices, dtype=jnp.int32),
                jnp.asarray(
                    self._chunk_rollout_mode_ids(rollout_state_indices),
                    dtype=jnp.int32,
                ),
                jnp.asarray(self.oscbf.arm_sign, dtype=jnp.float32),
                jnp.asarray(self.oscbf.arm_offset, dtype=jnp.float32),
                jnp.asarray(self.oscbf.urdf_neutral_q, dtype=jnp.float32),
                jnp.asarray(self.oscbf.T_pelvis_urdf, dtype=jnp.float32),
                jnp.asarray(float(self.oscbf.dt), dtype=jnp.float32),
                jnp.asarray(self._control_mode_id(), dtype=jnp.int32),
                jnp.asarray(max_action_delta, dtype=jnp.float32),
            ),
            dtype=np.float32,
        )
        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        info = dict(prediction_info)
        info.update(
            {
                "jax_sequential_oscbf_used": True,
                "jax_sequential_oscbf_use_pelvis_cbf": bool(use_pelvis_cbf),
                "jax_sequential_oscbf_time_ms": float(elapsed_ms),
            }
        )
        return safe_chunk, info

    def evaluate_safety(self, obs, q_seq):
        if self.oscbf.oscbf_config is None or self.env is None:
            return self._unavailable(q_seq)
        q_seq = np.asarray(q_seq, dtype=np.float32)
        try:
            (
                capsule_a_world_seq,
                capsule_b_world_seq,
                capsule_radii_eval,
                prediction_info,
            ) = self._human_capsule_rollout_cached(obs, q_seq.shape[0])
            qd_seq = np.zeros_like(q_seq, dtype=np.float32)
            q_urdf_seq = []
            capsule_a_urdf_seq = []
            capsule_b_urdf_seq = []
            for k, (q_bigym, qd_bigym) in enumerate(zip(q_seq, qd_seq)):
                q_urdf, _, _, _ = self.oscbf._build_urdf_surrogate_state_from_bigym(q_bigym, qd_bigym)
                t_world_urdf = self.oscbf._get_world_T_urdf_from_bigym_state(q_bigym)
                t_urdf_world = np.linalg.inv(t_world_urdf)
                capsule_a_urdf = self.oscbf._transform_points(
                    t_urdf_world,
                    capsule_a_world_seq[k],
                )
                capsule_b_urdf = self.oscbf._transform_points(
                    t_urdf_world,
                    capsule_b_world_seq[k],
                )
                self.oscbf._validate_capsules(
                    capsule_a_urdf,
                    capsule_b_urdf,
                    capsule_radii_eval,
                )
                q_urdf_seq.append(q_urdf)
                capsule_a_urdf_seq.append(capsule_a_urdf)
                capsule_b_urdf_seq.append(capsule_b_urdf)

            h_values = np.asarray(
                self._batched_h_fn(
                    jnp.asarray(q_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_a_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_b_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_radii_eval, dtype=jnp.float32),
                ),
                dtype=np.float32,
            )
            signed_clearances = self._h_values_to_signed_clearances(
                h_values,
                capsule_radii_eval,
            )
            min_clearances = np.min(signed_clearances, axis=1).astype(np.float32)
            min_h_values = np.min(h_values, axis=1).astype(np.float32)
            unsafe = np.flatnonzero(min_clearances < self.min_clearance)
            info = {
                "horizon_safe": bool(unsafe.size == 0),
                "min_clearance": float(np.min(min_clearances)),
                "min_clearances": min_clearances,
                "clearance_units": "signed_distance_m",
                "min_h": float(np.min(min_h_values)),
                "min_h_values": min_h_values,
                "h_values": h_values,
                "h_units": "distance_sq_minus_combined_radius_sq_m2",
                "first_violation": int(unsafe[0]) if unsafe.size else None,
                "unsafe_count": int(unsafe.size),
                "safety_eval_available": True,
            }
            info.update(_h_argmin_metadata(h_values))
            info.update(prediction_info)
            return info
        except Exception as exc:
            logger.warning("Chunk horizon OSCBF monitor failed: %s", exc)
            return self._unavailable(q_seq)

    def evaluate_safety_batch(self, obs, q_seq_batch):
        q_seq_batch = np.asarray(q_seq_batch, dtype=np.float32)
        if q_seq_batch.ndim == 2:
            q_seq_batch = q_seq_batch[None, :, :]
        if q_seq_batch.ndim != 3:
            raise ValueError(
                "Expected q_seq_batch with shape (B, H, Q), "
                f"got {q_seq_batch.shape}"
            )
        if self.oscbf.oscbf_config is None or self.env is None:
            return self._unavailable_batch(q_seq_batch)
        batch, horizon = q_seq_batch.shape[:2]
        try:
            (
                capsule_a_world_seq,
                capsule_b_world_seq,
                capsule_radii_eval,
                prediction_info,
            ) = self._human_capsule_rollout_cached(obs, horizon)

            q_bigym_flat = q_seq_batch.reshape(batch * horizon, q_seq_batch.shape[-1])
            capsule_a_world_flat = np.broadcast_to(
                capsule_a_world_seq[None, :, :, :],
                (batch,) + capsule_a_world_seq.shape,
            ).reshape(batch * horizon, capsule_a_world_seq.shape[1], 3)
            capsule_b_world_flat = np.broadcast_to(
                capsule_b_world_seq[None, :, :, :],
                (batch,) + capsule_b_world_seq.shape,
            ).reshape(batch * horizon, capsule_b_world_seq.shape[1], 3)

            jax_prep_used = False
            prep_t0 = time.perf_counter()
            try:
                q_urdf_seq, capsule_a_urdf_seq, capsule_b_urdf_seq = (
                    _jax_prepare_horizon_clearance_inputs(
                        jnp.asarray(q_bigym_flat, dtype=jnp.float32),
                        jnp.asarray(capsule_a_world_flat, dtype=jnp.float32),
                        jnp.asarray(capsule_b_world_flat, dtype=jnp.float32),
                        jnp.asarray(self.oscbf.bigym_state_arm_indices, dtype=jnp.int32),
                        jnp.asarray(self.oscbf.urdf_arm_joint_indices, dtype=jnp.int32),
                        jnp.asarray(self.oscbf.arm_sign, dtype=jnp.float32),
                        jnp.asarray(self.oscbf.arm_offset, dtype=jnp.float32),
                        jnp.asarray(self.oscbf.urdf_neutral_q, dtype=jnp.float32),
                        jnp.asarray(self.oscbf.T_pelvis_urdf, dtype=jnp.float32),
                    )
                )
                jax_prep_used = True
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "JAX horizon clearance input preparation failed; using Python preparation: %s",
                    exc,
                )
                q_urdf_seq = []
                capsule_a_urdf_seq = []
                capsule_b_urdf_seq = []
                qd_zero = np.zeros(q_seq_batch.shape[-1], dtype=np.float32)
                for candidate_q_seq in q_seq_batch:
                    for k, q_bigym in enumerate(candidate_q_seq):
                        q_urdf, _, _, _ = self.oscbf._build_urdf_surrogate_state_from_bigym(
                            q_bigym,
                            qd_zero,
                        )
                        t_world_urdf = self.oscbf._get_world_T_urdf_from_bigym_state(q_bigym)
                        t_urdf_world = np.linalg.inv(t_world_urdf)
                        capsule_a_urdf = self.oscbf._transform_points(
                            t_urdf_world,
                            capsule_a_world_seq[k],
                        )
                        capsule_b_urdf = self.oscbf._transform_points(
                            t_urdf_world,
                            capsule_b_world_seq[k],
                        )
                        self.oscbf._validate_capsules(
                            capsule_a_urdf,
                            capsule_b_urdf,
                            capsule_radii_eval,
                        )
                        q_urdf_seq.append(q_urdf)
                        capsule_a_urdf_seq.append(capsule_a_urdf)
                        capsule_b_urdf_seq.append(capsule_b_urdf)
            prep_time_ms = 1000.0 * (time.perf_counter() - prep_t0)

            h_t0 = time.perf_counter()
            h_values = np.asarray(
                self._batched_h_fn(
                    jnp.asarray(q_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_a_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_b_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_radii_eval, dtype=jnp.float32),
                ),
                dtype=np.float32,
            ).reshape(batch, horizon, -1)
            h_eval_time_ms = 1000.0 * (time.perf_counter() - h_t0)
            signed_clearances = self._h_values_to_signed_clearances(
                h_values,
                capsule_radii_eval,
            )
            min_clearances = np.min(signed_clearances, axis=2).astype(np.float32)
            min_h_values = np.min(h_values, axis=2).astype(np.float32)
            unsafe = min_clearances < self.min_clearance
            unsafe_any = np.any(unsafe, axis=1)
            first_violation = np.full(batch, -1, dtype=np.int32)
            if np.any(unsafe_any):
                first_violation[unsafe_any] = np.argmax(unsafe[unsafe_any], axis=1)
            info = {
                "horizon_safe": ~unsafe_any,
                "min_clearance": np.min(min_clearances, axis=1).astype(np.float32),
                "min_clearances": min_clearances,
                "clearance_units": "signed_distance_m",
                "min_h": np.min(min_h_values, axis=1).astype(np.float32),
                "min_h_values": min_h_values,
                "h_units": "distance_sq_minus_combined_radius_sq_m2",
                "first_violation": first_violation,
                "unsafe_count": np.count_nonzero(unsafe, axis=1).astype(np.int32),
                "safety_eval_available": True,
                "jax_clearance_prep_used": bool(jax_prep_used),
                "jax_clearance_prep_time_ms": float(prep_time_ms),
                "jax_h_eval_time_ms": float(h_eval_time_ms),
            }
            info.update(prediction_info)
            return info
        except Exception as exc:
            logger.warning("Batched chunk horizon OSCBF monitor failed: %s", exc)
            return self._unavailable_batch(q_seq_batch)

    def ee_pose(self, q):
        ee_seq = self.ee_pose_sequence(np.asarray(q, dtype=np.float32).reshape(1, -1))
        if ee_seq is None or ee_seq.shape[0] == 0:
            return None
        return ee_seq[0]

    def ee_pose_sequence(self, q_seq):
        if self.oscbf.robot_model is None:
            return None
        q_seq = np.asarray(q_seq, dtype=np.float32)
        qd_seq = np.zeros_like(q_seq, dtype=np.float32)
        ee_seq = []
        for q_bigym, qd_bigym in zip(q_seq, qd_seq):
            q_urdf, _, _, _ = self.oscbf._build_urdf_surrogate_state_from_bigym(
                q_bigym, qd_bigym
            )
            ee_urdf = np.asarray(
                self.oscbf.robot_model.ee_position(jnp.asarray(q_urdf, dtype=jnp.float32)),
                dtype=np.float32,
            ).reshape(-1, 3)
            t_world_urdf = self.oscbf._get_world_T_urdf_from_bigym_state(q_bigym)
            ee_world = self.oscbf._transform_points_homogeneous(t_world_urdf, ee_urdf)
            ee_seq.append(np.asarray(ee_world, dtype=np.float32).reshape(-1))
        return np.stack(ee_seq, axis=0).astype(np.float32)


    def robot_safety_geometry_sequence(self, q_seq, trace_indices=None):
        return _robot_h_compute_geometry_sequence(
            np.asarray(q_seq, dtype=np.float32),
            self.oscbf,
            trace_indices=trace_indices,
        )

    def _unavailable(self, q_seq):
        h = int(np.asarray(q_seq).shape[0])
        return {
            "horizon_safe": True,
            "min_clearance": float("inf"),
            "min_clearances": np.full(h, np.inf, dtype=np.float32),
            "first_violation": None,
            "unsafe_count": 0,
            "safety_eval_available": False,
        }

    def _unavailable_batch(self, q_seq_batch):
        q_seq_batch = np.asarray(q_seq_batch)
        if q_seq_batch.ndim == 2:
            q_seq_batch = q_seq_batch[None, :, :]
        batch = int(q_seq_batch.shape[0])
        horizon = int(q_seq_batch.shape[1])
        return {
            "horizon_safe": np.ones(batch, dtype=np.bool_),
            "min_clearance": np.full(batch, np.inf, dtype=np.float32),
            "min_clearances": np.full((batch, horizon), np.inf, dtype=np.float32),
            "first_violation": np.full(batch, -1, dtype=np.int32),
            "unsafe_count": np.zeros(batch, dtype=np.int32),
            "safety_eval_available": False,
        }
