from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from robobase.safetyfilter.h1_state_bridge import get_bigym_task

logger = logging.getLogger(__name__)

_H_ROBOT_CAPSULE_PARTS = ("upper_arm", "forearm")
_H_ROBOT_PART_GEOM_TOKEN_MAP = {
    "upper_arm": ("upperarm", "upper arm", "upper-arm", "shoulder", "shoulder_upper", "shoulder-upper"),
    "forearm": ("forearm", "forearm", "elbow"),
    "wrist": ("wrist", "hand", "gripper", "robotiq", "right_wrist", "h1/right_wrist"),
}
_H_ROBOT_PART_GEOM_FALLBACK_NAME_MAP = {
    "upper_arm": (
        "right_shoulder_upper",
        "right_upperarm",
        "upperarm",
        "h1/upperarm",
        "h1/right_upperarm",
    ),
    "forearm": (
        "right_forearm",
        "h1/right_forearm",
        "forearm",
        "h1/forearm",
    ),
    "wrist": (
        "right_wrist",
        "h1/right_wrist",
        "right_wrist_yaw",
        "h1/right_wrist_yaw",
        "right_gripper",
        "h1/right_gripper",
        "robotiq_2f85_right",
    ),
}
_H_ROBOT_PART_GEOM_CACHE: dict[tuple[int, str], list[int]] = {}
_H_ROBOT_PART_RED = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
_H_ROBOT_PART_BLUE = np.array([0.0, 0.35, 1.0, 1.0], dtype=np.float32)


def _get_non_arm_indices(action_dim: int, arm_indices: Sequence[int]) -> np.ndarray:
    arm_set = set(int(i) for i in arm_indices)
    return np.asarray(
        [i for i in range(action_dim) if i not in arm_set],
        dtype=np.int64,
    )

def assert_action_properties(
    nominal_action: np.ndarray,
    safe_action: np.ndarray,
    arm_indices: np.ndarray,
):
    assert safe_action.shape == nominal_action.shape
    assert np.isfinite(safe_action).all()

    non_arm_idx = _get_non_arm_indices(nominal_action.shape[0], arm_indices)

    if not np.allclose(
        nominal_action[non_arm_idx],
        safe_action[non_arm_idx],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError(
            "Non-arm dimensions changed.\n"
            f"non_arm_idx={non_arm_idx}\n"
            f"nominal={nominal_action[non_arm_idx]}\n"
            f"safe={safe_action[non_arm_idx]}"
        )



def _jsonable_trace_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable_trace_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_trace_value(v) for v in value]
    try:
        arr = np.asarray(value)
        if arr.ndim > 0:
            if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
                return arr.tolist()
            return arr.astype(str).tolist()
        if np.issubdtype(arr.dtype, np.number):
            return arr.item()
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return str(value)


def _as_chunk(action) -> tuple[np.ndarray, bool]:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        return action.reshape(1, -1), True
    if action.ndim == 2:
        return action, False
    raise ValueError(f"Unsupported action shape: {action.shape}")


def _restore_action_shape(chunk: np.ndarray, was_single: bool) -> np.ndarray:
    return chunk[0].copy() if was_single else chunk.copy()


def _raw_scaled_first_action(env, action):
    rescale_wrapper = _find_wrapped_env_with_attr(env, "action_stats")
    if rescale_wrapper is None or not hasattr(rescale_wrapper, "action"):
        return None
    return np.asarray(rescale_wrapper.action(np.asarray(action, dtype=np.float32)), dtype=np.float32)


def _raw_action_to_normalized(env, raw_action):
    raw = np.asarray(raw_action, dtype=np.float32)

    rescale_wrapper = _find_wrapped_env_with_attr(env, "action_stats")
    if rescale_wrapper is not None:
        action_stats = getattr(rescale_wrapper, "action_stats", None)
        transform = getattr(type(rescale_wrapper), "transform_to_tanh", None)
        if callable(transform) and action_stats is not None:
            try:
                if hasattr(rescale_wrapper, "min_max_margin"):
                    normalized = transform(
                        raw,
                        action_stats,
                        float(getattr(rescale_wrapper, "min_max_margin", 0.0)),
                    )
                else:
                    normalized = transform(raw, action_stats)
                return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)
            except Exception:  # noqa: BLE001
                pass

    tanh_wrapper = _find_wrapped_env_with_attr(env, "orig_action_space")
    if tanh_wrapper is not None:
        transform = getattr(type(tanh_wrapper), "transform_to_tanh", None)
        action_space = getattr(tanh_wrapper, "orig_action_space", None)
        if callable(transform) and action_space is not None:
            try:
                normalized = transform(raw, action_space)
                return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)
            except Exception:  # noqa: BLE001
                pass
    return None


def _env_chain(env):
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "env", None)


def _has_direct_attr(obj, attr_name):
    if attr_name in getattr(obj, "__dict__", {}):
        return True
    return any(attr_name in cls.__dict__ for cls in type(obj).mro())


def _find_wrapped_attr(env, attr_name):
    for candidate in _env_chain(env):
        if _has_direct_attr(candidate, attr_name):
            return getattr(candidate, attr_name)
    return None


def _find_wrapped_env_with_attr(env, attr_name):
    for candidate in _env_chain(env):
        if _has_direct_attr(candidate, attr_name):
            return candidate
    return None


def _apply_robot_spawn_offset_xy(env, offset_xy) -> Optional[dict[str, list[float]]]:
    if offset_xy is None:
        return None

    task = _find_wrapped_env_with_attr(env, "RESET_ROBOT_POS")
    if task is None:
        raise RuntimeError("Could not find raw BiGym env with RESET_ROBOT_POS.")

    if hasattr(task, "_eval_default_reset_robot_pos"):
        default_pos = np.asarray(
            task._eval_default_reset_robot_pos, dtype=np.float64
        ).copy()
    else:
        default_pos = np.asarray(task.RESET_ROBOT_POS, dtype=np.float64).copy()
        task._eval_default_reset_robot_pos = default_pos.copy()

    offset = np.asarray(offset_xy, dtype=np.float64).reshape(2)
    spawn_pos = default_pos.copy()
    spawn_pos[:2] = spawn_pos[:2] + offset
    task.RESET_ROBOT_POS = spawn_pos

    return {
        "default_pos": default_pos.astype(float).tolist(),
        "offset_xy": offset.astype(float).tolist(),
        "spawn_pos": spawn_pos.astype(float).tolist(),
    }


def _reset_action_sequence_history(env) -> int:
    reset_count = 0
    for candidate in _env_chain(env):
        reset = getattr(candidate, "_init_action_history", None)
        if callable(reset):
            reset()
            reset_count += 1
    return reset_count


def _set_action_sequence_temporal_ensemble(env, enabled: bool) -> list[tuple[Any, bool]]:
    records: list[tuple[Any, bool]] = []
    for candidate in _env_chain(env):
        if hasattr(candidate, "_temporal_ensemble"):
            records.append((candidate, bool(getattr(candidate, "_temporal_ensemble"))))
            setattr(candidate, "_temporal_ensemble", bool(enabled))
    return records


def _restore_action_sequence_temporal_ensemble(records: list[tuple[Any, bool]]) -> None:
    for candidate, previous_enabled in records:
        setattr(candidate, "_temporal_ensemble", bool(previous_enabled))


def _zero_bound_joint_velocity(bound_joint) -> None:
    for attr in ("qvel", "qacc"):
        value = getattr(bound_joint, attr, None)
        if value is None:
            continue
        try:
            value *= 0
        except Exception:  # noqa: BLE001
            try:
                setattr(bound_joint, attr, 0.0)
            except Exception:  # noqa: BLE001
                pass


def _set_robot_freeze_next_step(env) -> int:
    task = _find_wrapped_env_with_attr(env, "_robot")
    if task is None:
        return 0
    try:
        setattr(task, "_freeze_robot_state_next_step", True)
        return 1
    except Exception:  # noqa: BLE001
        return 0


def _sync_robot_low_level_hold_state(env) -> int:
    task = _find_wrapped_env_with_attr(env, "_robot")
    if task is None:
        return 0
    robot = getattr(task, "_robot", None)
    mojo = getattr(task, "mojo", None) or getattr(task, "_mojo", None)
    physics = getattr(mojo, "physics", None) if mojo is not None else None
    if robot is None or physics is None:
        return 0

    synced = 0
    floating_base = getattr(robot, "floating_base", None)
    if floating_base is not None:
        for name in ("_accumulated_actions", "_last_action"):
            arr = getattr(floating_base, name, None)
            if arr is not None:
                try:
                    arr[...] = 0.0
                except Exception:  # noqa: BLE001
                    pass
        for actuator in getattr(floating_base, "all_actuators", []) or []:
            joint = getattr(actuator, "joint", None)
            if joint is None:
                continue
            try:
                bound_joint = physics.bind(joint)
                bound_actuator = physics.bind(actuator)
                bound_actuator.ctrl = bound_joint.qpos
                _zero_bound_joint_velocity(bound_joint)
                synced += 1
            except Exception:  # noqa: BLE001
                continue

    for actuator in getattr(robot, "limb_actuators", []) or []:
        joint = getattr(actuator, "joint", None)
        if joint is None:
            continue
        try:
            bound_joint = physics.bind(joint)
            bound_actuator = physics.bind(actuator)
            bound_actuator.ctrl = bound_joint.qpos
            _zero_bound_joint_velocity(bound_joint)
            synced += 1
        except Exception:  # noqa: BLE001
            continue

    for gripper in (getattr(robot, "grippers", {}) or {}).values():
        try:
            current = float(gripper.qpos)
            gripper.set_control(current)
            synced += len(getattr(gripper, "actuators", []) or [])
        except Exception:  # noqa: BLE001
            continue
        for joint in getattr(gripper, "_actuated_joints", []) or []:
            try:
                _zero_bound_joint_velocity(physics.bind(joint))
            except Exception:  # noqa: BLE001
                continue

    forward = getattr(physics, "forward", None)
    if callable(forward):
        forward()
    return synced


def _hard_hold_action_from_live_robot(env, env_action):
    task = _find_wrapped_env_with_attr(env, "_robot")
    robot = getattr(task, "_robot", None) if task is not None else None
    if robot is None:
        return env_action, [], 0.0

    qpos = getattr(robot, "qpos_actuated", None)
    if qpos is None:
        return env_action, [], 0.0
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.size == 0:
        return env_action, [], 0.0

    safe = np.asarray(env_action, dtype=np.float32).copy()
    action_dim = int(safe.shape[-1])
    hold_dim = min(action_dim, qpos.size)
    if hold_dim <= 0:
        return safe, [], 0.0

    hold = qpos[:hold_dim].astype(safe.dtype, copy=True)
    base_dof = 0
    floating_base = getattr(robot, "floating_base", None)
    if floating_base is not None:
        try:
            base_dof = min(int(floating_base.dof_amount), hold_dim)
        except Exception:  # noqa: BLE001
            base_dof = 0
    if base_dof > 0:
        # BiGym floating-base actions are deltas even when arm joints are absolute.
        hold[:base_dof] = 0.0

    indices = np.arange(hold_dim, dtype=np.int64)
    old = safe[..., indices].copy()
    safe[..., indices] = hold.reshape((1,) * (safe.ndim - 1) + (hold_dim,))
    delta_norm = float(np.linalg.norm(safe[..., indices] - old))
    return safe, indices.astype(int).tolist(), delta_norm


def _reset_visual_observation_history(env) -> int:
    reset_count = 0
    reset_names = (
        "_init_obs_history",
        "_init_observation_history",
        "_init_frame_stack",
        "reset_obs_history",
        "reset_observation_history",
        "reset_frame_stack",
    )
    for candidate in _env_chain(env):
        for name in reset_names:
            reset = getattr(candidate, name, None)
            if callable(reset):
                reset()
                reset_count += 1
                break
    return reset_count


def _flush_visual_stack_to_current(policy_obs):
    if not isinstance(policy_obs, dict):
        return policy_obs, 0
    flushed = copy.deepcopy(policy_obs)
    reset_count = 0
    for key, value in list(flushed.items()):
        if not str(key).startswith("rgb_"):
            continue
        arr = np.asarray(value)
        if arr.ndim < 4 or arr.shape[0] <= 1:
            continue
        current = arr[-1:].copy()
        flushed[key] = np.repeat(current, arr.shape[0], axis=0).astype(arr.dtype, copy=False)
        reset_count += 1
    return flushed, reset_count


def _reset_policy_visual_history_after_recovery(env, policy_env, policy_obs):
    reset_count = _reset_visual_observation_history(env)
    if policy_env is not None:
        reset_count += _reset_visual_observation_history(policy_env)
    policy_obs, flushed_count = _flush_visual_stack_to_current(policy_obs)
    return policy_obs, int(reset_count + flushed_count)


def _seed_frame_stack_history(env, seeded_policy_obs) -> int:
    if env is None or not isinstance(seeded_policy_obs, dict):
        return 0
    seed_count = 0
    seen: set[int] = set()
    for candidate in _env_chain(env):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        frames = getattr(candidate, "frames", None)
        if not isinstance(frames, dict):
            continue
        for key, value in list(frames.items()):
            if key not in seeded_policy_obs:
                continue
            target = np.asarray(seeded_policy_obs[key])
            current = np.asarray(value)
            if target.shape != current.shape:
                continue
            frames[key] = target.astype(current.dtype, copy=True)
            seed_count += 1
    return int(seed_count)


def _seed_policy_visual_history_after_recovery(
    policy_obs,
    recovery_policy_obs_history,
    *,
    env=None,
    policy_env=None,
):
    """Seed stacked policy observations from real recovery observations.

    This is used when the caller does not want to reset/flush visual history to
    the current frame. For each stacked observation entry, we take the newest
    frame from each recorded recovery observation and rebuild the stack with the
    last N real frames, padding with the earliest available recovery frame when
    fewer than N frames are available.
    """

    if not isinstance(policy_obs, dict) or not recovery_policy_obs_history:
        return policy_obs, 0
    seeded = copy.deepcopy(policy_obs)
    seed_count = 0
    for key, value in list(seeded.items()):
        arr = np.asarray(value)
        if arr.ndim < 2 or arr.shape[0] <= 1:
            continue
        frames = []
        expected_frame_shape = arr.shape[1:]
        for hist_obs in recovery_policy_obs_history:
            if not isinstance(hist_obs, dict) or key not in hist_obs:
                continue
            hist_arr = np.asarray(hist_obs[key])
            frame = None
            if hist_arr.shape == arr.shape and hist_arr.ndim >= 2 and hist_arr.shape[0] > 0:
                frame = hist_arr[-1]
            elif hist_arr.shape == expected_frame_shape:
                frame = hist_arr
            if frame is None or frame.shape != expected_frame_shape:
                continue
            frames.append(np.asarray(frame).copy())
        if not frames:
            continue
        stack_len = int(arr.shape[0])
        selected = frames[-stack_len:]
        if len(selected) < stack_len:
            selected = [selected[0].copy() for _ in range(stack_len - len(selected))] + selected
        try:
            seeded[key] = np.stack(selected, axis=0).astype(arr.dtype, copy=False)
            seed_count += 1
        except Exception:  # noqa: BLE001
            continue
    frame_stack_seed_count = _seed_frame_stack_history(env, seeded)
    if policy_env is not None and policy_env is not env:
        frame_stack_seed_count += _seed_frame_stack_history(policy_env, seeded)
    return seeded, int(seed_count), int(frame_stack_seed_count)


def _disable_human_arm_collisions(env) -> int:
    humanarms = _find_wrapped_attr(env, "humanarms")
    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if not humanarms or base_env is None:
        return 0

    import mujoco

    model = base_env.mojo.physics.model.ptr
    disabled = 0
    for name in ["cylinder_arm/upperarm_geom", "cylinder_arm/forearm_geom"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        model.geom_margin[gid] = 0.0
        disabled += 1
    base_env.mojo.physics.forward()
    return disabled


def _enable_human_arm_collisions(env) -> int:
    humanarms = _find_wrapped_attr(env, "humanarms")
    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if not humanarms or base_env is None:
        return 0

    import mujoco

    model = base_env.mojo.physics.model.ptr
    enabled = 0
    for name in ["cylinder_arm/upperarm_geom", "cylinder_arm/forearm_geom"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        model.geom_contype[gid] = 2
        model.geom_conaffinity[gid] = 1
        model.geom_margin[gid] = max(float(model.geom_margin[gid]), 0.01)
        enabled += 1
    base_env.mojo.physics.forward()
    return enabled


def _freeze_human_arm(env) -> int:
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0

    frozen = 0
    for human in humanarms:
        state = human.get_state()
        # set_qpos_target also switches HumanArm out of scripted mode and into
        # position mode, so scripted primitives stop advancing.
        human.set_qpos_target(state["qpos"])
        if hasattr(human, "_qpos_filt"):
            human._qpos_filt = state["qpos"].copy()
        if hasattr(human, "_qvel_filt"):
            human._qvel_filt[:] = 0.0
        if hasattr(human, "_walk_enable"):
            human._walk_enable = False
        if hasattr(human, "_walk_v"):
            human._walk_v[:] = 0.0
        if hasattr(human, "_carrier_dwell"):
            human._carrier_dwell = 1e9
        frozen += 1
    return frozen


def _update_temporary_human_blocker_if_present(env) -> Optional[dict]:
    blocker = _find_wrapped_attr(env, "_temporary_human_blocker")
    if blocker is None:
        return None
    task = _find_wrapped_env_with_attr(env, "get_dt")
    if task is None:
        return None
    info = dict(blocker.update(task.get_dt()))
    if hasattr(task, "_temporary_human_blocker_info"):
        task._temporary_human_blocker_info = dict(info)
    return info


def _configure_human_arm_challenge(env, args) -> int:
    _reset_human_arm_final_clear_state(args)
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0

    aggression = float(np.clip(args.human_arm_aggression, 0.1, 3.0))
    configured = 0
    for human in humanarms:
        if hasattr(human, "_style_speed"):
            human._style_speed = float(np.clip(human._style_speed * aggression, 0.1, 3.0))
        if hasattr(human, "_style_amp"):
            human._style_amp = float(np.clip(human._style_amp * aggression, 0.1, 1.75))
        if hasattr(human, "_style_dwell") and args.human_arm_zero_dwell:
            human._style_dwell = min(float(human._style_dwell), 0.2)
        if hasattr(human, "_carrier_dwell") and args.human_arm_zero_dwell:
            human._carrier_dwell = 0.0
        if args.human_arm_walk_radius is not None and hasattr(human, "_walk_radius"):
            human._walk_radius = float(max(0.0, args.human_arm_walk_radius))
        if args.human_arm_keepout_min_clear is not None and hasattr(human, "MIN_CLEAR"):
            min_clear = float(max(0.0, args.human_arm_keepout_min_clear))
            human.MIN_CLEAR = min_clear
            if hasattr(human, "KEEP_SOFT"):
                human.KEEP_SOFT = max(float(human.KEEP_SOFT), min_clear + 0.01)
            if hasattr(human, "KEEP_HARD"):
                human.KEEP_HARD = min(float(human.KEEP_HARD), max(0.0, min_clear * 0.4))
        if args.human_arm_disable_keepout:
            _disable_human_arm_internal_keepout(human)
        _bias_human_arm_goal(human, args.human_arm_goal_xy)
        configured += 1
    _force_human_arm_carrier_xy(env, _forced_human_arm_carrier_xy(args, step=0), args=args)
    _apply_natural_human_arm_contact_motion(env, args, step=0)
    _force_human_arm_carrier_xy(env, _forced_human_arm_carrier_xy(args, step=0), args=args)
    return configured


def _disable_human_arm_internal_keepout(human) -> None:
    if hasattr(human, "_robot_geom_ids"):
        human._robot_geom_ids = np.asarray([], dtype=np.int32)
    if hasattr(human, "_robot_keepout_r"):
        human._robot_keepout_r = np.asarray([], dtype=np.float64)
    if hasattr(human, "MIN_CLEAR"):
        human.MIN_CLEAR = -1.0
    if hasattr(human, "KEEP_SOFT"):
        human.KEEP_SOFT = -1.0
    if hasattr(human, "KEEP_HARD"):
        human.KEEP_HARD = -1.0
    if hasattr(human, "_debug_keepout_clear"):
        human._debug_keepout_clear = float("inf")
    if hasattr(human, "_debug_keepout_active"):
        human._debug_keepout_active = False


def _bias_human_arm_goal(human, goal_xy) -> bool:
    if goal_xy is None or not hasattr(human, "_walk_goal_xy"):
        return False
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    radius = float(getattr(human, "_walk_radius", np.linalg.norm(goal)))
    norm = float(np.linalg.norm(goal))
    if radius > 0.0 and norm > radius:
        goal = goal / (norm + 1e-12) * radius
    human._walk_goal_xy = goal
    if hasattr(human, "_carrier_dwell"):
        human._carrier_dwell = 0.0
    return True


def _transient_human_arm_alpha(args, step: int) -> float:
    if not args.human_arm_transient_obstruction:
        return 0.0
    start = int(args.human_arm_release_after_steps)
    duration = max(1, int(args.human_arm_release_duration_steps))
    return float(np.clip((int(step) - start) / duration, 0.0, 1.0))


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _human_arm_retracted_q(human):
    q = np.array(
        [
            np.deg2rad(-8.0),
            np.deg2rad(4.0),
            np.deg2rad(8.0),
            np.deg2rad(92.0),
        ],
        dtype=np.float64,
    )
    if hasattr(human, "_clip_joint_vec"):
        q = human._clip_joint_vec(q)
    return q

def _apply_human_arm_yaw_offset(
    human, args, q: np.ndarray, *, current_state: bool = False
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    offset = np.deg2rad(float(getattr(args, "human_arm_yaw_offset_deg", 0.0)))
    if q.shape[0] > 1:
        if current_state:
            previous = float(getattr(human, "_eval_human_arm_yaw_offset_rad", 0.0))
            q[1] += offset - previous
        else:
            q[1] += offset
    setattr(human, "_eval_human_arm_yaw_offset_rad", float(offset))
    if hasattr(human, "_clip_joint_vec"):
        q = human._clip_joint_vec(q)
    return q


def _natural_human_arm_contact_q(human, args, step: int = 0, dt: float = 0.05):
    phase_step = float(step) + float(getattr(args, "human_arm_natural_motion_phase_offset_steps", 0.0))
    phase = 2.0 * np.pi * float(args.human_arm_natural_motion_frequency) * phase_step * float(dt)
    reach = 0.5 * (1.0 - np.cos(phase))
    sweep = np.sin(phase)
    settle = np.sin(0.5 * phase + 0.35)
    lateral_scale = float(max(0.0, getattr(args, "human_arm_natural_lateral_scale", 1.0)))
    curl_scale = float(max(0.0, getattr(args, "human_arm_natural_return_curl_scale", 0.0)))
    return_phase = float(np.clip(-sweep, 0.0, 1.0))

    q = np.array(
        [
            np.deg2rad(1.0) + lateral_scale * np.deg2rad(4.0) * settle,
            np.deg2rad(0.0) + lateral_scale * np.deg2rad(7.0) * sweep,
            np.deg2rad(-30.0) - np.deg2rad(30.0) * reach + np.deg2rad(2.0) * settle + np.deg2rad(8.0) * curl_scale * return_phase,
            np.deg2rad(66.0) - np.deg2rad(30.0) * reach + np.deg2rad(4.0) * np.sin(phase + 0.8) + np.deg2rad(28.0) * curl_scale * return_phase,
        ],
        dtype=np.float64,
    )
    if hasattr(human, "_clip_joint_vec"):
        q = human._clip_joint_vec(q)
    return q


def _apply_natural_human_arm_contact_motion(env, args, step: int = 0, dt: float = 0.05) -> int:
    if not args.human_arm_natural_contact_motion:
        return 0
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0
    applied = 0
    for human in humanarms:
        if not hasattr(human, "_set_kinematic_state"):
            continue
        q = _natural_human_arm_contact_q(human, args, step=step, dt=dt)
        alpha = _smoothstep(_transient_human_arm_alpha(args, step))
        if alpha > 0.0:
            q = (1.0 - alpha) * q + alpha * _human_arm_retracted_q(human)
        q = _apply_human_arm_yaw_offset(human, args, q)
        if hasattr(human, "_qpos_filt"):
            human._qpos_filt = q.copy()
        if hasattr(human, "_qvel_filt"):
            human._qvel_filt[:] = 0.0
        if hasattr(human, "_qpos_target"):
            human._qpos_target[:] = q
        if hasattr(human, "_walk_xy"):
            xy = np.asarray(human._walk_xy, dtype=np.float64).copy()
        else:
            state = human.get_state()
            xy = np.zeros(2, dtype=np.float64)
        human._set_kinematic_state(xy, q)
        applied += 1
    return applied


def _robot_ee_world_xy(oscbf, q_full: np.ndarray, qd_full: np.ndarray, offset_xy=None):
    if oscbf is None or oscbf.robot_model is None:
        return None
    q_urdf, _, _, _ = oscbf._build_urdf_surrogate_state_from_bigym(q_full, qd_full)
    ee_urdf = np.asarray(
        oscbf.robot_model.ee_position(jnp.asarray(q_urdf, dtype=jnp.float32)),
        dtype=np.float32,
    ).reshape(3)
    t_world_urdf = oscbf._get_world_T_urdf_from_bigym_state(q_full)
    ee_world = np.asarray(
        oscbf._transform_points_homogeneous(t_world_urdf, ee_urdf),
        dtype=np.float64,
    ).reshape(-1)
    xy = ee_world[:2].copy()
    if offset_xy is not None:
        xy = xy + np.asarray(offset_xy, dtype=np.float64).reshape(2)
    return xy

def _robot_gripper_geom_world_xy(env, offset_xy=None):
    try:
        task = get_bigym_task(env)
        model = task._mojo.model
        data = task._mojo.data
    except Exception:  # noqa: BLE001
        return None

    priority_patterns = (
        ("robotiq_2f85_right", "finger"),
        ("robotiq_2f85_right", "pad"),
        ("robotiq_2f85_right", "driver"),
        ("robotiq_2f85_right",),
        ("right_wrist",),
        ("wrist",),
    )
    exclude_patterns = ("visual", "camera", "left")

    for patterns in priority_patterns:
        points = []
        for geom_id in range(model.ngeom):
            name = (model.geom(geom_id).name or "").lower()
            if not name:
                continue
            if any(excluded in name for excluded in exclude_patterns):
                continue
            if all(pattern in name for pattern in patterns):
                points.append(np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3))
        if points:
            xy = np.mean(np.stack(points, axis=0), axis=0)[:2]
            if offset_xy is not None:
                xy = xy + np.asarray(offset_xy, dtype=np.float64).reshape(2)
            return xy

    return None


def _drawer_obstruction_carrier_xy(args, step: int = 0, dt: float = 0.05):
    xy = np.asarray(args.human_arm_drawer_obstruction_xy, dtype=np.float64).reshape(2)
    amp = np.asarray(args.human_arm_drawer_obstruction_amp_xy, dtype=np.float64).reshape(2)
    phase = 2.0 * np.pi * float(args.human_arm_force_carrier_frequency) * float(step) * float(dt)
    # Move locally around the drawer area without anchoring to the robot EE.
    offset = np.array(
        [
            0.55 * np.sin(phase + 0.4) + 0.20 * np.sin(1.9 * phase),
            0.45 * np.sin(phase + 1.7) + 0.15 * np.sin(1.4 * phase + 0.8),
        ],
        dtype=np.float64,
    )
    alpha = _smoothstep(_transient_human_arm_alpha(args, step))
    return xy + (1.0 - alpha) * amp * offset


def _ee_side_sweep_carrier_xy(args, anchor_xy, step: int = 0, dt: float = 0.05):
    xy = np.asarray(anchor_xy, dtype=np.float64).reshape(2)
    amp = np.asarray(args.human_arm_ee_side_sweep_amp_xy, dtype=np.float64).reshape(2)
    phase = (
        2.0 * np.pi * float(args.human_arm_ee_side_sweep_frequency) * float(step) * float(dt)
        + float(getattr(args, "human_arm_ee_side_sweep_phase", 0.0))
    )
    offset = np.array(
        [
            0.35 * np.sin(0.5 * phase + 0.3),
            np.sin(phase),
        ],
        dtype=np.float64,
    )
    alpha = _smoothstep(_transient_human_arm_alpha(args, step))
    return xy + (1.0 - alpha) * amp * offset


def _forced_human_arm_carrier_xy(args, step: int = 0, dt: float = 0.05, anchor_xy=None):
    if (
        anchor_xy is None
        and args.human_arm_force_carrier_xy is None
        and not args.human_arm_transient_obstruction
        and not args.human_arm_drawer_obstruction
    ):
        return None
    if anchor_xy is not None:
        if args.human_arm_ee_side_sweep:
            xy = _ee_side_sweep_carrier_xy(args, anchor_xy, step=step, dt=dt)
        else:
            xy = np.asarray(anchor_xy, dtype=np.float64).reshape(2)
    elif args.human_arm_drawer_obstruction:
        xy = _drawer_obstruction_carrier_xy(args, step=step, dt=dt)
    elif args.human_arm_force_carrier_xy is None:
        xy = np.asarray([-0.5, 0.2], dtype=np.float64)
    else:
        xy = np.asarray(args.human_arm_force_carrier_xy, dtype=np.float64).reshape(2)

    if args.human_arm_force_carrier_amp_xy is not None:
        amp = np.asarray(args.human_arm_force_carrier_amp_xy, dtype=np.float64).reshape(2)
        phase = 2.0 * np.pi * float(args.human_arm_force_carrier_frequency) * float(step) * float(dt)
        offset = np.array([np.sin(phase), np.sin(phase + 0.5 * np.pi)], dtype=np.float64)
        alpha = _smoothstep(_transient_human_arm_alpha(args, step))
        xy = xy + (1.0 - alpha) * amp * offset

    alpha = _smoothstep(_transient_human_arm_alpha(args, step))
    if alpha > 0.0:
        release_xy = np.asarray(args.human_arm_release_carrier_xy, dtype=np.float64).reshape(2)
        xy = (1.0 - alpha) * xy + alpha * release_xy
    return xy


def _force_human_arm_carrier_xy(env, carrier_xy, args=None) -> int:
    if carrier_xy is None:
        return 0
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0
    xy = np.asarray(carrier_xy, dtype=np.float64).reshape(2)
    forced = 0
    for human in humanarms:
        if not hasattr(human, "_set_kinematic_state"):
            continue
        if hasattr(human, "_qpos_filt"):
            joint_q = np.asarray(human._qpos_filt, dtype=np.float64).copy()
        else:
            joint_q = np.asarray(human.get_state()["qpos"], dtype=np.float64).copy()
        joint_q = _apply_human_arm_yaw_offset(human, args, joint_q, current_state=True) if args is not None else joint_q
        human._set_kinematic_state(xy, joint_q)
        try:
            task = get_bigym_task(env)
            task._mojo.physics.forward()
        except Exception:  # noqa: BLE001
            pass
        if hasattr(human, "_walk_xy"):
            human._walk_xy = xy.copy()
        if hasattr(human, "_walk_goal_xy"):
            human._walk_goal_xy = xy.copy()
        if hasattr(human, "_walk_v"):
            human._walk_v[:] = 0.0
        forced += 1
    return forced


def _reset_human_arm_final_clear_state(args) -> None:
    for name in (
        "_human_arm_final_clear_y_last",
        "_human_arm_final_clear_y_prev_delta",
        "_human_arm_final_clear_y_peak_step",
    ):
        if hasattr(args, name):
            delattr(args, name)


def _human_arm_final_clear_start_step(args) -> int:
    configured_start = int(getattr(args, "human_arm_final_clear_after_steps", -1))
    if configured_start < 0:
        return -1
    trigger = str(getattr(args, "human_arm_final_clear_trigger", "step"))
    peak_step = getattr(args, "_human_arm_final_clear_y_peak_step", None)
    if trigger == "carrier-y-peak" and peak_step is not None:
        return int(peak_step)
    return configured_start


def _human_arm_final_clear_alpha(args, step: int) -> float:
    start = _human_arm_final_clear_start_step(args)
    if start < 0:
        return 0.0
    duration = max(1, int(getattr(args, "human_arm_final_clear_duration_steps", 20)))
    return _smoothstep(float(np.clip((int(step) - start) / duration, 0.0, 1.0)))


def _update_human_arm_final_clear_y_peak_trigger(args, step: int, carrier_xy) -> None:
    if int(getattr(args, "human_arm_final_clear_after_steps", -1)) < 0:
        return
    if str(getattr(args, "human_arm_final_clear_trigger", "step")) != "carrier-y-peak":
        return
    if getattr(args, "_human_arm_final_clear_y_peak_step", None) is not None:
        return
    if carrier_xy is None:
        return
    y = float(np.asarray(carrier_xy, dtype=np.float64).reshape(2)[1])
    last_y = getattr(args, "_human_arm_final_clear_y_last", None)
    prev_delta = getattr(args, "_human_arm_final_clear_y_prev_delta", None)
    if last_y is not None:
        delta = y - float(last_y)
        eps = 1e-5
        if prev_delta is not None and float(prev_delta) > eps and delta <= eps:
            setattr(args, "_human_arm_final_clear_y_peak_step", int(step))
        setattr(args, "_human_arm_final_clear_y_prev_delta", float(delta))
    setattr(args, "_human_arm_final_clear_y_last", y)


def _limited_step_toward(current, target, max_step: float) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_step or norm <= 1e-12:
        return target.copy()
    return current + delta * (float(max_step) / norm)


def _apply_final_human_arm_clearance(env, args, step: int, dt: float = 0.05) -> int:
    alpha = _human_arm_final_clear_alpha(args, step)
    if alpha <= 0.0:
        return 0
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0

    start_step = _human_arm_final_clear_start_step(args)
    dt = max(float(dt), 1e-9)
    max_carrier_step = float(args.human_arm_final_clear_max_carrier_speed) * dt
    max_joint_step = float(args.human_arm_final_clear_max_joint_speed) * dt
    target_xy = np.asarray(
        getattr(args, "human_arm_final_clear_carrier_xy", [-0.85, 0.55]),
        dtype=np.float64,
    ).reshape(2)
    applied = 0
    for human in humanarms:
        if not hasattr(human, "_set_kinematic_state"):
            continue
        if hasattr(human, "_walk_xy"):
            current_xy = np.asarray(human._walk_xy, dtype=np.float64).reshape(2)
        elif hasattr(human, "_carrier_qpos_adr") and hasattr(human, "_physics"):
            current_xy = np.asarray(
                human._physics.data.qpos[human._carrier_qpos_adr],
                dtype=np.float64,
            ).reshape(2)
        else:
            current_xy = target_xy.copy()

        if hasattr(human, "_qpos_filt"):
            current_q = np.asarray(human._qpos_filt, dtype=np.float64).copy()
        else:
            current_q = np.asarray(human.get_state()["qpos"], dtype=np.float64).copy()
        if getattr(human, "_eval_final_clear_start_step", None) != start_step:
            human._eval_final_clear_start_step = start_step
            human._eval_final_clear_start_xy = current_xy.copy()
            human._eval_final_clear_start_q = current_q.copy()

        start_xy = np.asarray(human._eval_final_clear_start_xy, dtype=np.float64).reshape(2)
        start_q = np.asarray(human._eval_final_clear_start_q, dtype=np.float64).copy()
        target_q = _apply_human_arm_yaw_offset(
            human,
            args,
            _human_arm_retracted_q(human),
        )
        desired_xy = (1.0 - alpha) * start_xy + alpha * target_xy
        desired_q = (1.0 - alpha) * start_q + alpha * target_q
        xy = _limited_step_toward(current_xy, desired_xy, max_carrier_step)
        q = _limited_step_toward(current_q, desired_q, max_joint_step)
        if hasattr(human, "_clip_joint_vec"):
            q = human._clip_joint_vec(q)

        if hasattr(human, "_qpos_filt"):
            human._qpos_filt = q.copy()
        if hasattr(human, "_qvel_filt"):
            human._qvel_filt[:] = (q - current_q) / dt
        if hasattr(human, "_qpos_target"):
            human._qpos_target[:] = q
        if hasattr(human, "_walk_xy"):
            human._walk_xy = xy.copy()
        if hasattr(human, "_walk_goal_xy"):
            human._walk_goal_xy = target_xy.copy()
        if hasattr(human, "_walk_v"):
            human._walk_v[:] = 0.0
        if hasattr(human, "_carrier_dwell"):
            human._carrier_dwell = 0.0
        human._set_kinematic_state(xy, q)
        applied += 1

    try:
        task = get_bigym_task(env)
        task._mojo.physics.forward()
    except Exception:  # noqa: BLE001
        pass
    return applied


def _human_arm_contact_geom_center_xy(env):
    try:
        import mujoco

        task = get_bigym_task(env)
        model = task._mojo.model
        data = task._mojo.data
        centers = []
        humanarms = getattr(task, "humanarms", [])
        for human in humanarms:
            for geom_name in ("forearm_geom", "upperarm_geom"):
                gid = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    human._pref(geom_name),
                )
                if gid >= 0:
                    centers.append(np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3))
        if not centers:
            return None
        return np.mean(np.stack(centers, axis=0), axis=0)[:2]
    except Exception:  # noqa: BLE001
        return None


def _align_human_arm_contact_geoms_to_xy(env, target_xy) -> bool:
    if target_xy is None:
        return False
    center_xy = _human_arm_contact_geom_center_xy(env)
    if center_xy is None:
        return False
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return False
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    shifted = False
    for human in humanarms:
        if not hasattr(human, "_set_kinematic_state"):
            continue
        if hasattr(human, "_walk_xy"):
            current_xy = np.asarray(human._walk_xy, dtype=np.float64).reshape(2)
        else:
            state = human.get_state()
            current_xy = np.asarray(state.get("walk_xy", [0.0, 0.0]), dtype=np.float64).reshape(2)
        shifted_xy = current_xy + (target_xy - center_xy)
        _force_human_arm_carrier_xy(env, shifted_xy)
        shifted = True
    return shifted


def _make_policy_env_cfg(cfg, policy_env: str):
    if not policy_env.startswith("bigym/"):
        raise ValueError(
            "--policy-env currently expects a BiGym env name like "
            f"'bigym/drawer_top_open', got {policy_env!r}."
        )
    policy_cfg = copy.deepcopy(cfg)
    policy_cfg.env.task_name = policy_env.split("/", 1)[1]
    return policy_cfg


def _joint_qpos_qvel_dims(model, joint_id):
    import mujoco

    joint_type = model.jnt_type[joint_id]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def _sync_named_mujoco_state(source_env, target_env) -> dict[str, int]:
    source_task = _find_wrapped_env_with_attr(source_env, "mojo")
    target_task = _find_wrapped_env_with_attr(target_env, "mojo")
    if source_task is None or target_task is None:
        raise RuntimeError("Could not find source/target BiGym envs for state mirroring.")

    import mujoco

    src_model = source_task.mojo.physics.model.ptr
    src_data = source_task.mojo.physics.data
    dst_model = target_task.mojo.physics.model.ptr
    dst_data = target_task.mojo.physics.data

    dst_data.time = src_data.time

    copied_joints = 0
    for src_jid in range(src_model.njnt):
        name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_JOINT, src_jid)
        if not name:
            continue
        dst_jid = mujoco.mj_name2id(dst_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if dst_jid < 0:
            continue

        src_nq, src_nv = _joint_qpos_qvel_dims(src_model, src_jid)
        dst_nq, dst_nv = _joint_qpos_qvel_dims(dst_model, dst_jid)
        if src_nq != dst_nq or src_nv != dst_nv:
            continue

        src_qadr = src_model.jnt_qposadr[src_jid]
        dst_qadr = dst_model.jnt_qposadr[dst_jid]
        src_dadr = src_model.jnt_dofadr[src_jid]
        dst_dadr = dst_model.jnt_dofadr[dst_jid]
        dst_data.qpos[dst_qadr : dst_qadr + dst_nq] = src_data.qpos[src_qadr : src_qadr + src_nq]
        dst_data.qvel[dst_dadr : dst_dadr + dst_nv] = src_data.qvel[src_dadr : src_dadr + src_nv]
        copied_joints += 1

    copied_actuators = 0
    for src_aid in range(src_model.nu):
        name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_ACTUATOR, src_aid)
        if not name:
            continue
        dst_aid = mujoco.mj_name2id(dst_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if dst_aid < 0:
            continue
        dst_data.ctrl[dst_aid] = src_data.ctrl[src_aid]
        copied_actuators += 1

    target_task.mojo.physics.forward()
    return {"joints": copied_joints, "actuators": copied_actuators}


def _sync_animated_legs(env, is_moving: bool = True) -> bool:
    task = _find_wrapped_env_with_attr(env, "_robot")
    if task is None:
        return False
    floating_base = getattr(task._robot, "floating_base", None)
    if floating_base is None:
        return False
    animated_legs = getattr(floating_base, "_animated_legs", None)
    if animated_legs is None:
        return False
    animated_legs.step(floating_base._pelvis_z, is_moving=is_moving)
    task.mojo.physics.forward()
    return True


def _update_scripted_human_arm_pose(env, args, step: int, anchor_xy=None) -> int:
    if env is None or args.freeze_human_arm:
        return 0
    runtime = env.unwrapped if hasattr(env, "unwrapped") else env
    human_motion_dt = runtime.dt if hasattr(runtime, "dt") else 0.05
    forced_xy = _forced_human_arm_carrier_xy(
        args,
        step=step,
        dt=human_motion_dt,
        anchor_xy=anchor_xy,
    )
    _update_human_arm_final_clear_y_peak_trigger(args, step, forced_xy)
    final_clear_active = _human_arm_final_clear_alpha(args, step) > 0.0
    advanced = 0
    if not final_clear_active:
        advanced = _advance_human_arm_only(
            env,
            substeps=args.human_arm_substeps,
            goal_xy=args.human_arm_goal_xy,
        )
        _force_human_arm_carrier_xy(env, forced_xy, args=args)
        _apply_natural_human_arm_contact_motion(
            env,
            args,
            step=step,
            dt=human_motion_dt,
        )
        _force_human_arm_carrier_xy(env, forced_xy, args=args)
        if anchor_xy is not None and forced_xy is not None:
            _align_human_arm_contact_geoms_to_xy(env, forced_xy)
    _apply_final_human_arm_clearance(env, args, step, dt=human_motion_dt)
    _sync_animated_legs(env, is_moving=True)
    return advanced


def _advance_human_arm_only(env, substeps: int = 1, goal_xy=None) -> int:
    task = _find_wrapped_env_with_attr(env, "humanarms")
    if task is None:
        return 0
    dt = task.get_dt() if hasattr(task, "get_dt") else 0.05
    substeps = max(1, int(substeps))
    advanced = 0
    for _ in range(substeps):
        for human in task.humanarms:
            _bias_human_arm_goal(human, goal_xy)
            human._on_step(dt)
            advanced += 1
    task.mojo.physics.forward()
    return advanced


def _human_arm_geom_ids(env) -> list[int]:
    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if base_env is None:
        return []

    import mujoco

    model = base_env.mojo.physics.model.ptr
    geom_ids = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("cylinder_arm/"):
            geom_ids.append(gid)
    return geom_ids


def _render_visual_obs_with_hidden_human_arm(env) -> dict[str, np.ndarray]:
    base_env = _find_wrapped_env_with_attr(env, "_get_visual_obs")
    if base_env is None or not hasattr(base_env, "mojo"):
        raise RuntimeError("Could not find underlying BiGym env for clean policy rendering.")

    geom_ids = _human_arm_geom_ids(env)
    if not geom_ids:
        return base_env._get_visual_obs()

    model = base_env.mojo.physics.model.ptr
    old_rgba = model.geom_rgba[geom_ids].copy()
    try:
        model.geom_rgba[geom_ids, 3] = 0.0
        return base_env._get_visual_obs()
    finally:
        model.geom_rgba[geom_ids] = old_rgba


def _policy_obs_with_hidden_human_arm(env, obs, prev_policy_obs=None):
    policy_obs = copy.deepcopy(obs)
    visual_obs = _render_visual_obs_with_hidden_human_arm(env)

    for key, clean_frame in visual_obs.items():
        if not key.startswith("rgb_") or key not in policy_obs:
            continue

        current = np.asarray(policy_obs[key])
        clean_frame = np.asarray(clean_frame, dtype=current.dtype)

        if current.ndim == clean_frame.ndim + 1:
            if prev_policy_obs is None or key not in prev_policy_obs:
                policy_obs[key] = np.repeat(clean_frame[None], current.shape[0], axis=0)
            else:
                previous = np.asarray(prev_policy_obs[key], dtype=current.dtype)
                policy_obs[key] = np.concatenate([previous[1:], clean_frame[None]], axis=0)
        elif current.shape == clean_frame.shape:
            policy_obs[key] = clean_frame
        else:
            raise ValueError(
                f"Cannot replace policy RGB observation {key}: "
                f"wrapped shape={current.shape}, clean frame shape={clean_frame.shape}."
            )

    return policy_obs


def _adapt_policy_obs_to_space(policy_obs, observation_space):
    if observation_space is None or not isinstance(policy_obs, dict):
        return policy_obs

    adapted = dict(policy_obs)
    for key, space in observation_space.items():
        if key not in adapted or not hasattr(space, "shape"):
            continue
        expected_shape = tuple(int(x) for x in space.shape)
        value = np.asarray(adapted[key])
        if value.shape == expected_shape:
            continue

        if (
            key == "low_dim_state"
            and value.ndim == len(expected_shape)
            and value.shape[:-1] == expected_shape[:-1]
            and value.shape[-1] >= expected_shape[-1]
        ):
            adapted[key] = value[..., : expected_shape[-1]].astype(value.dtype, copy=False)
            continue

        if key == "low_dim_state" and value.size >= int(np.prod(expected_shape)):
            flat = value.reshape(-1)[: int(np.prod(expected_shape))]
            adapted[key] = flat.reshape(expected_shape).astype(value.dtype, copy=False)
            continue

        raise ValueError(
            f"Policy observation {key!r} has shape {value.shape}, "
            f"but the loaded policy expects {expected_shape}."
        )
    return adapted


def _normalize_h_robot_part(part: Optional[str]) -> Optional[str]:
    if part is None:
        return None
    part = str(part).strip().lower()
    if not part:
        return None
    if part.startswith("right_"):
        part = part[len("right_"):]
    if part.startswith("left_"):
        part = part[len("left_"):]
    if part.startswith("h1/"):
        part = part[len("h1/"):]

    if part in {"shoulder_upper", "upperarm", "upper-arm", "upper_arm", "upperarm_link", "right_shoulder_upper"}:
        return "upper_arm"
    if part in {"right_upperarm", "upper_arm", "upper-arm", "shoulder"}:
        return "upper_arm"
    if part in {"right_forearm", "forearm", "lower_arm", "lowerarm"}:
        return "forearm"
    if part in {"right_wrist", "wrist", "gripper", "hand", "robotiq"}:
        return "wrist"
    return None


def _get_robot_part_geom_ids(env, robot_part: Optional[str]) -> list[int]:
    normalized = _normalize_h_robot_part(robot_part)
    if normalized is None:
        return []

    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if base_env is None or not hasattr(base_env, "mojo"):
        return []

    try:
        model = base_env.mojo.physics.model.ptr
    except Exception:  # noqa: BLE001
        return []

    cache_key = (id(model), normalized)
    cached = _H_ROBOT_PART_GEOM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import mujoco

    tokens = _H_ROBOT_PART_GEOM_TOKEN_MAP.get(normalized, (normalized,))
    found = set()
    for gid in range(int(model.ngeom)):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        if not name:
            continue
        if name.startswith("cylinder_arm/") or "human" in name:
            continue
        if any(token in name for token in tokens):
            found.add(int(gid))

    if not found:
        for name in _H_ROBOT_PART_GEOM_FALLBACK_NAME_MAP.get(normalized, ()):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if int(gid) >= 0:
                found.add(int(gid))

    geom_ids = sorted(found)
    _H_ROBOT_PART_GEOM_CACHE[cache_key] = geom_ids
    return geom_ids


def _collect_robot_part_geom_ids(env, robot_parts: set[str]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for part in robot_parts:
        for gid in _get_robot_part_geom_ids(env, part):
            if gid not in seen:
                ids.append(gid)
                seen.add(gid)
    return ids



def _mujoco_model_data(task):
    mojo = getattr(task, "_mojo", None) or getattr(task, "mojo", None)
    if mojo is None:
        return None, None
    model = getattr(mojo, "model", None)
    data = getattr(mojo, "data", None)
    if model is not None and data is not None:
        return model, data
    physics = getattr(mojo, "physics", None)
    if physics is not None:
        return getattr(physics.model, "ptr", physics.model), getattr(physics.data, "ptr", physics.data)
    return None, None


def _mujoco_site_position(model, data, names: Sequence[str]):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None
    for name in names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3)
    return None


def _box_edge_segments_world(center, size, xmat=None):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    size = np.asarray(size, dtype=np.float64).reshape(3)
    rot = np.eye(3, dtype=np.float64) if xmat is None else np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    corners = []
    labels = []
    for dx in (-1.0, 1.0):
        for dy in (-1.0, 1.0):
            for dz in (-1.0, 1.0):
                local = np.asarray([dx * size[0], dy * size[1], dz * size[2]], dtype=np.float64)
                corners.append((center + rot @ local).astype(float).tolist())
                labels.append((int(dx), int(dy), int(dz)))
    idx = {label: i for i, label in enumerate(labels)}
    pairs = []
    for dx in (-1, 1):
        for dy in (-1, 1):
            pairs.append((idx[(dx, dy, -1)], idx[(dx, dy, 1)]))
        for dz in (-1, 1):
            pairs.append((idx[(dx, -1, dz)], idx[(dx, 1, dz)]))
    for dy in (-1, 1):
        for dz in (-1, 1):
            pairs.append((idx[(-1, dy, dz)], idx[(1, dy, dz)]))
    return [[corners[a], corners[b]] for a, b in pairs]


def _mujoco_find_body_id(model, names: Sequence[str]):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return -1
    for name in names:
        try:
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        except Exception:  # noqa: BLE001
            body_id = -1
        if body_id >= 0:
            return body_id
    suffixes = tuple(f"/{name}" for name in names)
    for body_id in range(int(model.nbody)):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name in names or body_name.endswith(suffixes):
            return int(body_id)
    return -1


def _mujoco_body_is_descendant(model, body_id: int, ancestor_id: int) -> bool:
    if ancestor_id < 0 or body_id < 0:
        return False
    body_id = int(body_id)
    ancestor_id = int(ancestor_id)
    while body_id >= 0:
        if body_id == ancestor_id:
            return True
        parent = int(model.body_parentid[body_id])
        if parent == body_id:
            break
        body_id = parent
    return False


def _mujoco_body_name(model, body_id: int) -> str:
    try:
        import mujoco
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""
    except Exception:  # noqa: BLE001
        return ""


def _drawer_motion_axis(model, data):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None
    try:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_small_4")
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "drawer_small_4")
        if joint_id < 0 or site_id < 0:
            return None
        qpos_adr = int(model.jnt_qposadr[joint_id])
        old_qpos = float(data.qpos[qpos_adr])
        p0 = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3).copy()
        eps = 0.01
        data.qpos[qpos_adr] = old_qpos + eps
        mujoco.mj_forward(model, data)
        p1 = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3).copy()
        data.qpos[qpos_adr] = old_qpos
        mujoco.mj_forward(model, data)
        axis = p1 - p0
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9 or not np.isfinite(norm):
            return None
        return (axis / norm).astype(float).tolist()
    except Exception:  # noqa: BLE001
        return None

def _drawer_scene_geometry(model, data):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None

    cabinet_body_id = _mujoco_find_body_id(model, ("base_cabinet_600",))
    drawer_body_id = _mujoco_find_body_id(
        model,
        ("base_cabinet_600/drawer_small_4", "drawer_small_4"),
    )

    cabinet_segments = []
    drawer_segments = []
    cabinet_geoms = []
    drawer_geoms = []
    for geom_id in range(int(model.ngeom)):
        try:
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            body_id = int(model.geom_bodyid[geom_id])
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_name = _mujoco_body_name(model, body_id)
            in_cabinet = (
                _mujoco_body_is_descendant(model, body_id, cabinet_body_id)
                or geom_name.startswith("base_cabinet_600")
                or body_name.startswith("base_cabinet_600")
                or "/base_cabinet_600" in body_name
            )
            if not in_cabinet:
                continue
            segments = _box_edge_segments_world(
                data.geom_xpos[geom_id],
                model.geom_size[geom_id],
                data.geom_xmat[geom_id],
            )
        except Exception:  # noqa: BLE001
            continue
        entry = {
            "name": geom_name or body_name or f"geom_{geom_id}",
            "body": body_name,
            "center": np.asarray(data.geom_xpos[geom_id], dtype=np.float64).astype(float).tolist(),
            "size": np.asarray(model.geom_size[geom_id], dtype=np.float64).astype(float).tolist(),
            "segments": segments,
        }
        is_drawer_geom = (
            _mujoco_body_is_descendant(model, body_id, drawer_body_id)
            or "/drawer_small_4" in body_name
            or body_name.endswith("drawer_small_4")
            or "/drawer_small_4" in geom_name
            or geom_name.endswith("drawer_small_4")
        )
        if is_drawer_geom:
            drawer_segments.extend(segments)
            drawer_geoms.append(entry)
        else:
            cabinet_segments.extend(segments)
            cabinet_geoms.append(entry)

    if not cabinet_segments and not drawer_segments:
        return None
    return {
        "absolute": True,
        "cabinet": cabinet_segments,
        "drawer": drawer_segments,
        "open_axis": _drawer_motion_axis(model, data),
        "cabinet_geoms": cabinet_geoms,
        "drawer_geoms": drawer_geoms,
        "source": "mujoco_geoms",
    }


def _drawer_open_distance_and_fraction(task, model, data):
    distance = None
    joint_range = None
    try:
        import mujoco
        for name in ("base_cabinet_600/drawer_small_4", "drawer_small_4"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            distance = float(data.qpos[qpos_adr])
            joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64).reshape(2)
            break
    except Exception:  # noqa: BLE001
        distance = None

    if distance is None and hasattr(task, "cabinet_drawers"):
        try:
            distance = float(np.asarray(task.cabinet_drawers.get_state()).reshape(-1)[-1])
        except Exception:  # noqa: BLE001
            distance = None

    if distance is None:
        return None, None

    if (
        joint_range is not None
        and np.isfinite(joint_range).all()
        and float(joint_range[1]) > float(joint_range[0])
    ):
        fraction = (distance - float(joint_range[0])) / (float(joint_range[1]) - float(joint_range[0]))
    else:
        fraction = distance / 0.38
    return float(distance), float(np.clip(fraction, 0.0, 1.0))


def _diagnostic_task_state(env) -> dict[str, Any]:
    """Best-effort task progress/object state for failure diagnosis logs."""
    try:
        task = get_bigym_task(env)
    except Exception:  # noqa: BLE001
        task = _find_wrapped_env_with_attr(env, "mojo")
    if task is None:
        return {
            "drawer_open_distance": None,
            "drawer_open_fraction": None,
            "drawer_joint_position": None,
            "task_progress": None,
            "ee_object_distance": None,
            "object_state": None,
        }

    model, data = _mujoco_model_data(task)
    drawer_distance = None
    drawer_fraction = None
    handle_pos = None
    ee_pos = None
    drawer_scene_geometry = None
    ee_object_distance = None
    if model is not None and data is not None:
        drawer_distance, drawer_fraction = _drawer_open_distance_and_fraction(task, model, data)
        drawer_scene_geometry = _drawer_scene_geometry(model, data)
        handle_pos = _mujoco_site_position(
            model,
            data,
            ("base_cabinet_600/drawer_small_4", "drawer_small_4"),
        )
        ee_pos = _mujoco_site_position(
            model,
            data,
            ("h1/right_end_effector", "right_end_effector"),
        )
        if handle_pos is not None and ee_pos is not None:
            ee_object_distance = float(np.linalg.norm(ee_pos - handle_pos))
    elif hasattr(task, "cabinet_drawers"):
        drawer_distance, drawer_fraction = _drawer_open_distance_and_fraction(task, model, data)

    task_progress = drawer_fraction if drawer_fraction is not None else drawer_distance
    object_state = {}
    if drawer_distance is not None:
        object_state["drawer_open_distance"] = float(drawer_distance)
    if drawer_fraction is not None:
        object_state["drawer_open_fraction"] = float(drawer_fraction)
    if handle_pos is not None:
        object_state["handle_pos"] = np.asarray(handle_pos, dtype=np.float64).astype(float).tolist()
    if ee_pos is not None:
        object_state["ee_pos"] = np.asarray(ee_pos, dtype=np.float64).astype(float).tolist()
    if drawer_scene_geometry is not None:
        object_state["drawer_scene_geometry"] = drawer_scene_geometry
    return {
        "drawer_open_distance": None if drawer_distance is None else float(drawer_distance),
        "drawer_open_fraction": None if drawer_fraction is None else float(drawer_fraction),
        "drawer_joint_position": None if drawer_distance is None else float(drawer_distance),
        "task_progress": None if task_progress is None else float(task_progress),
        "ee_object_distance": ee_object_distance,
        "object_state": object_state or None,
    }


def _diagnostic_progress_delta(before: dict[str, Any], after: dict[str, Any]) -> Optional[float]:
    before_progress = before.get("task_progress") if before else None
    after_progress = after.get("task_progress") if after else None
    if before_progress is None or after_progress is None:
        return None
    before_progress = float(before_progress)
    after_progress = float(after_progress)
    if not (np.isfinite(before_progress) and np.isfinite(after_progress)):
        return None
    return float(after_progress - before_progress)


def _finite_task_progress(task_state: Optional[dict[str, Any]]) -> Optional[float]:
    if not task_state:
        return None
    progress = task_state.get("task_progress")
    if progress is None:
        return None
    progress = float(progress)
    return progress if np.isfinite(progress) else None


def _post_recovery_task_guard_reanchor_allowed(phase_state, args):
    phase = str((phase_state or {}).get("phase", "pre_grasp"))
    allowed = set(getattr(args, "post_recovery_task_guard_reanchor_phases", ["grasp"]))
    return phase in allowed, phase


def _post_recovery_task_guard_ready(task_state, phase_state, args):
    _ = phase_state
    progress = _finite_task_progress(task_state)
    if progress is None:
        return False, "task_progress_unavailable"
    if progress > float(args.post_recovery_task_guard_min_progress):
        return True, "task_progress"
    return False, "insufficient_task_progress"


def _phase_reanchor_state(env, args):
    try:
        task = get_bigym_task(env)
    except Exception:  # noqa: BLE001
        task = _find_wrapped_env_with_attr(env, "mojo")
    if task is None:
        return None

    model, data = _mujoco_model_data(task)
    if model is None or data is None:
        return None

    handle_pos = _mujoco_site_position(
        model,
        data,
        ("base_cabinet_600/drawer_small_4", "drawer_small_4"),
    )
    ee_pos = _mujoco_site_position(
        model,
        data,
        ("h1/right_end_effector", "right_end_effector"),
    )
    if handle_pos is None or ee_pos is None:
        return None

    drawer_distance, drawer_fraction = _drawer_open_distance_and_fraction(task, model, data)
    if drawer_fraction is None:
        drawer_fraction = 0.0

    gripper_qpos = None
    gripper_closed = False
    robot = getattr(task, "robot", None)
    if robot is not None and hasattr(robot, "qpos_grippers"):
        try:
            qpos_grippers = np.asarray(robot.qpos_grippers, dtype=np.float64).reshape(-1)
            if qpos_grippers.size:
                gripper_qpos = float(qpos_grippers[-1])
                gripper_closed = gripper_qpos >= float(args.phase_reanchor_gripper_closed_threshold)
        except Exception:  # noqa: BLE001
            pass

    gripper_xy = _robot_gripper_geom_world_xy(env)
    gripper_site_xy_error = None
    if gripper_xy is not None:
        gripper_xy = np.asarray(gripper_xy, dtype=np.float64).reshape(2)
        gripper_site_xy_error = float(np.linalg.norm(gripper_xy - ee_pos[:2]))

    requested_task_point_source = str(getattr(args, "phase_reanchor_task_point_source", "ee_site"))
    task_point_source = "mujoco_site:h1/right_end_effector"
    task_point_fallback_reason = None
    task_point_xy = np.asarray(ee_pos[:2], dtype=np.float64).reshape(2).copy()
    if requested_task_point_source == "gripper_geom":
        if gripper_xy is not None:
            task_point_xy = gripper_xy.copy()
            task_point_source = "mujoco_gripper_geom"
        else:
            task_point_fallback_reason = "gripper_geom_unavailable"

    site_ee_to_handle_xy = handle_pos[:2] - ee_pos[:2]
    site_ee_to_handle_dist = float(np.linalg.norm(site_ee_to_handle_xy))
    gripper_to_handle_dist = None
    if gripper_xy is not None:
        gripper_to_handle_dist = float(np.linalg.norm(handle_pos[:2] - gripper_xy))

    control_ee_to_handle_xy = handle_pos[:2] - task_point_xy
    control_ee_to_handle_dist = float(np.linalg.norm(control_ee_to_handle_xy))
    if drawer_fraction >= float(args.phase_reanchor_done_threshold):
        phase = "done"
    elif (
        drawer_fraction >= float(args.phase_reanchor_pull_open_threshold)
        or (gripper_closed and control_ee_to_handle_dist <= float(args.phase_reanchor_grasp_dist))
    ):
        phase = "pull"
    elif control_ee_to_handle_dist <= float(args.phase_reanchor_grasp_dist):
        phase = "grasp"
    else:
        phase = "pre_grasp"

    preload_target_grasp = False
    preload_grasp_limit = float(
        getattr(
            args,
            "phase_reanchor_bridge_handle_dist",
            getattr(args, "phase_reanchor_live_release_handle_dist", 0.24),
        )
    )
    if (
        phase == "pre_grasp"
        and bool(getattr(args, "phase_reanchor_preload_target_grasp", False))
        and gripper_to_handle_dist is not None
        and np.isfinite(gripper_to_handle_dist)
        and gripper_to_handle_dist <= preload_grasp_limit
    ):
        phase = "grasp"
        preload_target_grasp = True

    target_ee_xy = handle_pos[:2] + _phase_reanchor_offset_xy(args, phase)
    control_ee_to_target_xy = target_ee_xy - task_point_xy
    control_ee_to_target_dist = float(np.linalg.norm(control_ee_to_target_xy))
    site_ee_to_target_dist = float(np.linalg.norm(target_ee_xy - ee_pos[:2]))
    gripper_to_target_dist = None
    if gripper_xy is not None:
        gripper_to_target_dist = float(np.linalg.norm(target_ee_xy - gripper_xy))

    requested_measurement_source = str(getattr(args, "phase_reanchor_measurement_task_point_source", "control"))
    measurement_task_point_xy = task_point_xy.copy()
    measurement_task_point_source = task_point_source
    measurement_task_point_fallback_reason = None
    if requested_measurement_source == "ee_site":
        measurement_task_point_xy = np.asarray(ee_pos[:2], dtype=np.float64).reshape(2).copy()
        measurement_task_point_source = "mujoco_site:h1/right_end_effector"
    elif requested_measurement_source == "gripper_geom":
        if gripper_xy is not None:
            measurement_task_point_xy = gripper_xy.copy()
            measurement_task_point_source = "mujoco_gripper_geom"
        else:
            measurement_task_point_fallback_reason = "gripper_geom_unavailable"

    ee_to_handle_xy = handle_pos[:2] - measurement_task_point_xy
    ee_to_handle_dist = float(np.linalg.norm(ee_to_handle_xy))
    ee_to_target_xy = target_ee_xy - measurement_task_point_xy
    ee_to_target_dist = float(np.linalg.norm(ee_to_target_xy))

    control_error_source = str(getattr(args, "phase_reanchor_control_error_source", "control"))
    if control_error_source == "measurement":
        control_ee_to_target_xy = ee_to_target_xy.copy()
        control_ee_to_target_dist = float(np.linalg.norm(control_ee_to_target_xy))
    else:
        control_error_source = "control"

    trust_limit = float(getattr(args, "phase_reanchor_task_point_geometry_trust_error", 0.08))
    task_point_geometry_untrusted = bool(
        (requested_task_point_source == "gripper_geom" and gripper_xy is None)
        or (requested_measurement_source == "gripper_geom" and gripper_xy is None)
        or (
            requested_measurement_source in {"control", "ee_site"}
            and measurement_task_point_source == "mujoco_site:h1/right_end_effector"
            and gripper_site_xy_error is not None
            and np.isfinite(gripper_site_xy_error)
            and gripper_site_xy_error > trust_limit
        )
    )

    return {
        "task": task,
        "phase": phase,
        "handle_pos": handle_pos,
        "ee_pos": ee_pos,
        "gripper_xy": gripper_xy,
        "task_point_xy": measurement_task_point_xy,
        "task_point_source": measurement_task_point_source,
        "task_point_requested_source": requested_measurement_source,
        "task_point_fallback_reason": measurement_task_point_fallback_reason,
        "control_task_point_xy": task_point_xy,
        "control_task_point_source": task_point_source,
        "control_task_point_requested_source": requested_task_point_source,
        "control_task_point_fallback_reason": task_point_fallback_reason,
        "control_ee_to_handle_xy": control_ee_to_handle_xy,
        "control_ee_to_handle_dist": control_ee_to_handle_dist,
        "control_ee_to_target_xy": control_ee_to_target_xy,
        "control_ee_to_target_dist": control_ee_to_target_dist,
        "control_error_source": control_error_source,
        "preload_target_grasp": bool(preload_target_grasp),
        "preload_grasp_limit": float(preload_grasp_limit),
        "site_ee_to_handle_dist": site_ee_to_handle_dist,
        "site_ee_to_target_dist": site_ee_to_target_dist,
        "gripper_site_xy_error": gripper_site_xy_error,
        "gripper_to_handle_dist": gripper_to_handle_dist,
        "gripper_to_target_dist": gripper_to_target_dist,
        "task_point_geometry_untrusted": task_point_geometry_untrusted,
        "ee_to_handle_xy": ee_to_handle_xy,
        "ee_to_handle_dist": ee_to_handle_dist,
        "target_ee_xy": target_ee_xy,
        "ee_to_target_xy": ee_to_target_xy,
        "ee_to_target_dist": ee_to_target_dist,
        "drawer_open_distance": drawer_distance,
        "drawer_open_fraction": float(drawer_fraction),
        "gripper_qpos": gripper_qpos,
        "gripper_closed": bool(gripper_closed),
    }


def _phase_reanchor_offset_xy(args, phase: str) -> np.ndarray:
    if phase == "pull":
        offset = args.phase_reanchor_pull_offset_xy
    elif phase == "grasp":
        offset = args.phase_reanchor_grasp_offset_xy
    else:
        offset = args.phase_reanchor_pregrasp_offset_xy
    return np.asarray(offset, dtype=np.float64).reshape(2)


def _should_start_phase_reanchor(args, step: int, state, drawer_history, cooldown_left: int):
    if not args.phase_reanchor or state is None or cooldown_left > 0:
        return False, None
    if state.get("phase") == "done" or step < int(args.phase_reanchor_check_after_steps):
        return False, None

    window = int(args.phase_reanchor_no_progress_window)
    if len(drawer_history) < window:
        return False, None
    recent = np.asarray(drawer_history[-window:], dtype=np.float64)
    if not np.isfinite(recent).any():
        return False, None
    progress = float(np.nanmax(recent) - np.nanmin(recent))
    return progress < float(args.phase_reanchor_min_drawer_progress), progress



def _phase_reanchor_arm_q_window_command(env, chunk, q_full, oscbf, args, state):
    """Closed-loop q-space tracker for ACT re-entry during phase reanchor.

    Prefer an externally supplied nominal re-entry q-window when available.  With
    a nominal window, track the full controlled state: floating-base dimensions
    are sent as bounded delta commands, while arm/wrist dimensions are sent as
    bounded absolute joint targets.  If no reference window is supplied, fall
    back to the live ACT chunk for arm/wrist only.
    """
    info = {
        "arm_servo_enabled": False,
        "arm_servo_reason": None,
        "arm_servo_rank": None,
        "arm_servo_error_norm": None,
        "arm_servo_delta_norm": None,
        "arm_servo_command_norm": None,
        "arm_servo_action_delta_norm": None,
        "arm_servo_target_source": None,
        "arm_servo_target_episode": None,
        "arm_servo_target_start_step": None,
        "arm_servo_target_step": None,
        "arm_servo_target_window_index": None,
        "arm_servo_target_window_score": None,
        "live_taskspace_guard_suppress_q_servo": None,
        "live_taskspace_guard_distance": None,
        "live_taskspace_guard_best_distance": None,
    }
    if not bool(getattr(args, "phase_reanchor_arm_servo", False)):
        info["arm_servo_reason"] = "disabled"
        return None, None, info
    if state is not None and bool(state.get("nominal_reentry_suppress_q_servo", False)):
        suppress_reason = state.get("nominal_reentry_suppress_q_servo_reason") or "live_taskspace_worsening"
        info.update(
            {
                "arm_servo_reason": suppress_reason,
                "live_taskspace_guard_suppress_q_servo": True,
                "live_taskspace_guard_distance": state.get("live_taskspace_guard_distance"),
                "live_taskspace_guard_best_distance": state.get("live_taskspace_guard_best_distance"),
            }
        )
        return None, None, info

    base_action_idx = np.asarray(getattr(oscbf, "bigym_action_base_indices", []), dtype=np.int64)
    base_state_idx = np.asarray(getattr(oscbf, "bigym_state_base_indices", []), dtype=np.int64)
    arm_action_idx = np.asarray(getattr(oscbf, "bigym_action_arm_indices", []), dtype=np.int64)
    arm_state_idx = np.asarray(getattr(oscbf, "bigym_state_arm_indices", []), dtype=np.int64)
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    chunk_arr, _ = _as_chunk(chunk)

    target_source = "live_act_chunk"
    target_episode = None
    target_start_step = None
    target_steps = None
    nominal_q_window = None if state is None else state.get("nominal_reentry_q_window")

    if nominal_q_window is not None:
        try:
            q_window = np.asarray(nominal_q_window, dtype=np.float32)
        except Exception:  # noqa: BLE001
            q_window = None
        if q_window is not None and q_window.ndim == 2 and q_window.shape[0] > 0:
            track_base = bool(getattr(args, "phase_reanchor_nominal_window_track_base", True))
            base_count = min(base_action_idx.size, base_state_idx.size) if track_base else 0
            arm_count = min(arm_action_idx.size, arm_state_idx.size)
            base_action = base_action_idx[:base_count]
            base_state = base_state_idx[:base_count]
            arm_action = arm_action_idx[:arm_count]
            arm_state = arm_state_idx[:arm_count]
            base_valid = (
                (base_action < chunk_arr.shape[1])
                & (base_state < q.shape[0])
                & (base_state < q_window.shape[1])
            )
            arm_valid = (
                (arm_action < chunk_arr.shape[1])
                & (arm_state < q.shape[0])
                & (arm_state < q_window.shape[1])
            )
            base_action = base_action[base_valid]
            base_state = base_state[base_valid]
            arm_action = arm_action[arm_valid]
            arm_state = arm_state[arm_valid]
            if base_action.size + arm_action.size <= 0:
                info["arm_servo_reason"] = "nominal_window_no_valid_control_pairs"
                return None, None, info

            action_idx = np.concatenate([base_action, arm_action]).astype(np.int64, copy=False)
            state_idx = np.concatenate([base_state, arm_state]).astype(np.int64, copy=False)
            target_seq = q_window[:, state_idx].astype(np.float32, copy=False)
            q_ctrl = q[state_idx].astype(np.float32, copy=False)
            finite_rows = np.isfinite(target_seq).all(axis=1)
            if not np.any(finite_rows):
                info["arm_servo_reason"] = "nonfinite_nominal_target_window"
                return None, None, info
            finite_indices = np.flatnonzero(finite_rows)
            target_seq = target_seq[finite_rows]

            weights = np.ones(q_ctrl.shape[0], dtype=np.float32)
            if base_action.size:
                weights[: base_action.size] = np.float32(1.0)
            wrist_weight = float(getattr(args, "phase_reanchor_wrist_servo_weight", 1.0))
            if wrist_weight > 0.0 and arm_action.size >= 2:
                weights[base_action.size + arm_action.size - 2 : base_action.size + arm_action.size] = np.float32(wrist_weight)

            errors = target_seq - q_ctrl.reshape(1, -1)
            weighted_errors = errors * weights.reshape(1, -1)
            row_scores = np.linalg.norm(weighted_errors, axis=1)
            if not np.isfinite(row_scores).any():
                info["arm_servo_reason"] = "nonfinite_nominal_window_score"
                return None, None, info
            target_local = int(np.nanargmin(row_scores))
            target_row = int(finite_indices[target_local]) if finite_indices.size else target_local
            error = errors[target_local]
            target = target_seq[target_local]

            command = np.zeros_like(q_ctrl, dtype=np.float32)
            delta_for_metrics = np.zeros_like(q_ctrl, dtype=np.float32)
            if base_action.size:
                base_error = error[: base_action.size]
                base_cmd = float(getattr(args, "phase_reanchor_base_gain", 0.45)) * base_error
                max_base = float(getattr(args, "phase_reanchor_max_base_step", 0.012))
                if max_base > 0.0:
                    base_cmd = np.clip(base_cmd, -max_base, max_base)
                command[: base_action.size] = base_cmd.astype(np.float32, copy=False)
                delta_for_metrics[: base_action.size] = base_cmd.astype(np.float32, copy=False)
            if arm_action.size:
                arm_slice = slice(base_action.size, base_action.size + arm_action.size)
                arm_error = error[arm_slice]
                arm_q = q_ctrl[arm_slice]
                gain = float(getattr(args, "phase_reanchor_arm_gain", 1.0))
                arm_correction = gain * arm_error
                wrist_weight = float(getattr(args, "phase_reanchor_wrist_servo_weight", 1.0))
                if wrist_weight > 0.0 and arm_correction.size >= 2:
                    arm_correction = arm_correction.astype(np.float32, copy=True)
                    arm_correction[-2:] = gain * wrist_weight * arm_error[-2:]
                max_step = float(getattr(args, "phase_reanchor_arm_max_step", 0.08))
                if max_step > 0.0:
                    arm_correction = np.clip(arm_correction, -max_step, max_step)
                command[arm_slice] = arm_q + arm_correction.astype(np.float32, copy=False)
                delta_for_metrics[arm_slice] = arm_correction.astype(np.float32, copy=False)

            target_step = None
            target_steps = state.get("nominal_reentry_window_steps") if state is not None else None
            if target_steps is not None:
                try:
                    target_step = list(target_steps)[target_row]
                except Exception:  # noqa: BLE001
                    target_step = None
            info.update(
                {
                    "arm_servo_enabled": True,
                    "arm_servo_reason": (
                        "nominal_full_q_window_tracker"
                        if base_action.size
                        else "nominal_arm_q_window_tracker"
                    ),
                    "arm_servo_rank": int(action_idx.size),
                    "arm_servo_error_norm": float(np.linalg.norm(error)),
                    "arm_servo_delta_norm": float(np.linalg.norm(delta_for_metrics)),
                    "arm_servo_command_norm": float(np.linalg.norm(command)),
                    "arm_servo_target_source": str(state.get("nominal_reentry_source", "nominal_reentry_window")) if state is not None else "nominal_reentry_window",
                    "arm_servo_target_episode": state.get("nominal_reentry_episode") if state is not None else None,
                    "arm_servo_target_start_step": state.get("nominal_reentry_start_step") if state is not None else None,
                    "arm_servo_target_step": target_step,
                    "arm_servo_target_window_index": int(target_row),
                    "arm_servo_target_window_score": float(row_scores[target_local]),
                }
            )
            return action_idx, command.astype(np.float32, copy=False), info

    # Fallback: no stable reference window, so only track the local ACT arm chunk.
    if arm_action_idx.size == 0 or arm_state_idx.size == 0:
        info["arm_servo_reason"] = "no_arm_pairs"
        return None, None, info
    pair_count = min(arm_action_idx.size, arm_state_idx.size)
    action_idx = arm_action_idx[:pair_count]
    state_idx = arm_state_idx[:pair_count]
    window = min(4, int(chunk_arr.shape[0]))
    if window <= 0:
        info["arm_servo_reason"] = "empty_action_window"
        return None, None, info
    raw_rows = []
    for row in chunk_arr[:window]:
        raw_row = _raw_scaled_first_action(env, row)
        if raw_row is None:
            info["arm_servo_reason"] = "raw_action_unavailable"
            return None, None, info
        raw_rows.append(np.asarray(raw_row, dtype=np.float32).reshape(-1))
    raw_chunk = np.stack(raw_rows, axis=0)
    valid = (action_idx < raw_chunk.shape[1]) & (state_idx < q.shape[0])
    if not np.any(valid):
        info["arm_servo_reason"] = "no_valid_arm_pairs"
        return None, None, info
    action_idx = action_idx[valid]
    state_idx = state_idx[valid]
    q_arm = q[state_idx].astype(np.float32, copy=False)
    target_seq = raw_chunk[:, action_idx].astype(np.float32, copy=False)
    finite_rows = np.isfinite(target_seq).all(axis=1)
    if not np.any(finite_rows):
        info["arm_servo_reason"] = "nonfinite_target_window"
        return None, None, info
    target_seq = target_seq[finite_rows]
    weights = np.ones(q_arm.shape[0], dtype=np.float32)
    wrist_weight = float(getattr(args, "phase_reanchor_wrist_servo_weight", 1.0))
    if wrist_weight > 0.0 and weights.size >= 2:
        weights[-2:] = np.float32(wrist_weight)
    errors = target_seq - q_arm.reshape(1, -1)
    row_scores = np.linalg.norm(errors * weights.reshape(1, -1), axis=1)
    if not np.isfinite(row_scores).any():
        info["arm_servo_reason"] = "nonfinite_window_score"
        return None, None, info
    target_local = int(np.nanargmin(row_scores))
    error = target_seq[target_local] - q_arm
    correction = float(getattr(args, "phase_reanchor_arm_gain", 1.0)) * error
    if wrist_weight > 0.0 and correction.size >= 2:
        correction = correction.astype(np.float32, copy=True)
        correction[-2:] = float(getattr(args, "phase_reanchor_arm_gain", 1.0)) * wrist_weight * error[-2:]
    max_step = float(getattr(args, "phase_reanchor_arm_max_step", 0.08))
    if max_step > 0.0:
        correction = np.clip(correction, -max_step, max_step)
    command = q_arm + correction.astype(np.float32, copy=False)
    info.update(
        {
            "arm_servo_enabled": True,
            "arm_servo_reason": "q_window_tracker",
            "arm_servo_rank": int(action_idx.size),
            "arm_servo_error_norm": float(np.linalg.norm(error)),
            "arm_servo_delta_norm": float(np.linalg.norm(command - q_arm)),
            "arm_servo_command_norm": float(np.linalg.norm(command)),
            "arm_servo_target_source": target_source,
            "arm_servo_target_window_index": int(target_local),
            "arm_servo_target_window_score": float(row_scores[target_local]),
        }
    )
    return action_idx, command.astype(np.float32, copy=False), info


def _phase_reanchor_live_ee_arm_command(
    q_full,
    oscbf,
    args,
    state,
    target_ee_xy,
    ee_xy,
    action_idx,
    state_idx,
    action_dim,
):
    """Small damped task-space arm servo for ACT re-entry.

    This is only intended for the live-taskspace-worsening branch, where the
    nominal q-window tracker is known to be pulling the robot away from the
    live handle geometry.  It uses the safety model EE FK for a local XY
    Jacobian and keeps a weak nominal-window q prior when available.
    """
    info = {
        "live_ee_servo_enabled": False,
        "live_ee_servo_reason": "not_attempted",
        "live_ee_servo_error_norm": None,
        "live_ee_servo_delta_norm": None,
        "live_ee_servo_command_norm": None,
        "live_ee_servo_jacobian_rank": None,
        "live_ee_servo_nominal_reg": None,
        "live_ee_servo_fk_site_xy_error": None,
        "live_ee_servo_fk_gripper_xy_error": None,
        "live_ee_servo_predicted_error_before": None,
        "live_ee_servo_predicted_error_after": None,
        "live_ee_servo_geometry_untrusted": None,
        "suppressed_q_servo_arm_hold": False,
    }
    if not bool(getattr(args, "phase_reanchor_live_ee_servo", False)):
        info["live_ee_servo_reason"] = "disabled"
        return None, None, info
    allow_with_q_window = bool(getattr(args, "phase_reanchor_live_ee_servo_with_q_window", False))
    q_servo_suppressed = bool(state.get("nominal_reentry_suppress_q_servo", False)) if state is not None else False
    if state is None or (not q_servo_suppressed and not allow_with_q_window):
        info["live_ee_servo_reason"] = "q_servo_not_suppressed"
        return None, None, info
    if oscbf is None or getattr(oscbf, "robot_model", None) is None:
        info["live_ee_servo_reason"] = "missing_safety_fk"
        return None, None, info

    action_idx = np.asarray(action_idx, dtype=np.int64).reshape(-1)
    state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    pair_count = min(action_idx.size, state_idx.size)
    if pair_count <= 0:
        info["live_ee_servo_reason"] = "no_arm_pairs"
        return None, None, info
    action_idx = action_idx[:pair_count]
    state_idx = state_idx[:pair_count]
    valid = (action_idx < int(action_dim)) & (state_idx < q.shape[0])
    if not np.any(valid):
        info["live_ee_servo_reason"] = "no_valid_arm_pairs"
        return None, None, info
    action_idx = action_idx[valid]
    state_idx = state_idx[valid]
    q_arm = q[state_idx].astype(np.float32, copy=False)

    target_ee_xy = np.asarray(target_ee_xy, dtype=np.float64).reshape(2)
    ee_xy = np.asarray(ee_xy, dtype=np.float64).reshape(2)
    error_xy = target_ee_xy - ee_xy
    err_norm = float(np.linalg.norm(error_xy))
    if not np.isfinite(err_norm) or err_norm <= 1e-9:
        info.update(
            {
                "live_ee_servo_reason": "zero_or_nonfinite_error",
                "live_ee_servo_error_norm": err_norm if np.isfinite(err_norm) else None,
            }
        )
        return None, None, info
    error_clip = float(getattr(args, "phase_reanchor_arm_error_clip", 0.25))
    if error_clip > 0.0 and err_norm > error_clip:
        error_xy = error_xy * (error_clip / err_norm)

    qd_zero = np.zeros_like(q, dtype=np.float32)
    base_xy = _robot_ee_world_xy(oscbf, q, qd_zero)
    if base_xy is None:
        info["live_ee_servo_reason"] = "base_ee_fk_unavailable"
        return None, None, info
    base_xy = np.asarray(base_xy, dtype=np.float64).reshape(2)
    site_xy = np.asarray(ee_xy, dtype=np.float64).reshape(2)
    fk_site_error = float(np.linalg.norm(base_xy - site_xy))
    fk_gripper_error = None
    gripper_xy = None
    if isinstance(state, dict) and state.get("gripper_xy") is not None:
        try:
            gripper_xy = np.asarray(state.get("gripper_xy"), dtype=np.float64).reshape(2)
            fk_gripper_error = float(np.linalg.norm(base_xy - gripper_xy))
        except Exception:  # noqa: BLE001
            gripper_xy = None
            fk_gripper_error = None
    trust_limit = float(getattr(args, "phase_reanchor_task_point_geometry_trust_error", 0.08))
    geometry_untrusted = bool(
        (np.isfinite(fk_site_error) and fk_site_error > trust_limit)
        or (fk_gripper_error is not None and np.isfinite(fk_gripper_error) and fk_gripper_error > trust_limit)
        or bool(state.get("task_point_geometry_untrusted", False))
    )
    info.update(
        {
            "live_ee_servo_fk_site_xy_error": fk_site_error,
            "live_ee_servo_fk_gripper_xy_error": fk_gripper_error,
            "live_ee_servo_geometry_untrusted": geometry_untrusted,
        }
    )
    if geometry_untrusted:
        info["live_ee_servo_reason"] = "live_taskspace_geometry_untrusted"
        return None, None, info

    eps = float(getattr(args, "phase_reanchor_arm_fd_eps", 1e-3))
    if eps <= 0.0:
        info["live_ee_servo_reason"] = "nonpositive_fd_eps"
        return None, None, info
    jac = np.zeros((2, state_idx.size), dtype=np.float64)
    for col, sidx in enumerate(state_idx):
        q_pert = q.copy()
        q_pert[int(sidx)] += np.float32(eps)
        pert_xy = _robot_ee_world_xy(oscbf, q_pert, qd_zero)
        if pert_xy is None:
            info["live_ee_servo_reason"] = "perturbed_ee_fk_unavailable"
            return None, None, info
        jac[:, col] = (np.asarray(pert_xy, dtype=np.float64).reshape(2) - base_xy) / eps
    if not np.isfinite(jac).all():
        info["live_ee_servo_reason"] = "nonfinite_jacobian"
        return None, None, info

    gain = float(getattr(args, "phase_reanchor_live_ee_gain", getattr(args, "phase_reanchor_arm_gain", 0.9)))
    desired_xy = gain * error_xy
    nominal_delta = np.zeros_like(q_arm, dtype=np.float64)
    nominal_reg = float(getattr(args, "phase_reanchor_live_ee_nominal_reg", 0.15))
    nominal_q_window = state.get("nominal_reentry_q_window") if isinstance(state, dict) else None
    if nominal_q_window is not None and nominal_reg > 0.0:
        try:
            q_window = np.asarray(nominal_q_window, dtype=np.float32)
            if q_window.ndim == 2 and q_window.shape[0] > 0:
                valid_cols = state_idx < q_window.shape[1]
                if np.any(valid_cols):
                    target_seq = q_window[:, state_idx[valid_cols]].astype(np.float64, copy=False)
                    q_sel = q[state_idx[valid_cols]].astype(np.float64, copy=False)
                    row_scores = np.linalg.norm(target_seq - q_sel.reshape(1, -1), axis=1)
                    if np.isfinite(row_scores).any():
                        target_row = int(np.nanargmin(row_scores))
                        nominal_delta[valid_cols] = target_seq[target_row] - q_sel
        except Exception:  # noqa: BLE001
            nominal_delta[:] = 0.0

    damping = float(getattr(args, "phase_reanchor_arm_damping", 1e-3))
    damping = max(0.0, damping)
    rows = [jac]
    rhs = [desired_xy.astype(np.float64, copy=False)]
    if nominal_reg > 0.0:
        scale = float(np.sqrt(nominal_reg))
        rows.append(scale * np.eye(state_idx.size, dtype=np.float64))
        rhs.append(scale * nominal_delta.astype(np.float64, copy=False))
    if damping > 0.0:
        scale = float(np.sqrt(damping))
        rows.append(scale * np.eye(state_idx.size, dtype=np.float64))
        rhs.append(np.zeros(state_idx.size, dtype=np.float64))
    lhs = np.vstack(rows)
    target = np.concatenate(rhs)
    try:
        delta, *_ = np.linalg.lstsq(lhs, target, rcond=None)
    except np.linalg.LinAlgError:
        info["live_ee_servo_reason"] = "lstsq_failed"
        return None, None, info
    if not np.isfinite(delta).all():
        info["live_ee_servo_reason"] = "nonfinite_delta"
        return None, None, info
    max_step = float(getattr(args, "phase_reanchor_arm_max_step", 0.08))
    if max_step > 0.0:
        delta = np.clip(delta, -max_step, max_step)
    predicted_before = float(np.linalg.norm(target_ee_xy - base_xy))
    predicted_after_xy = base_xy + jac @ delta
    predicted_after = float(np.linalg.norm(target_ee_xy - predicted_after_xy))
    info.update(
        {
            "live_ee_servo_predicted_error_before": predicted_before,
            "live_ee_servo_predicted_error_after": predicted_after,
        }
    )
    min_improvement = float(getattr(args, "phase_reanchor_live_ee_servo_min_predicted_improvement", 0.0))
    if (not np.isfinite(predicted_before)) or (not np.isfinite(predicted_after)) or predicted_after > predicted_before - min_improvement:
        info["live_ee_servo_reason"] = "live_ee_prediction_not_improving"
        return None, None, info
    command = q_arm + delta.astype(np.float32, copy=False)
    info.update(
        {
            "live_ee_servo_enabled": True,
            "live_ee_servo_reason": "live_ee_taskspace_servo",
            "live_ee_servo_error_norm": err_norm,
            "live_ee_servo_delta_norm": float(np.linalg.norm(delta)),
            "live_ee_servo_command_norm": float(np.linalg.norm(command)),
            "live_ee_servo_jacobian_rank": int(np.linalg.matrix_rank(jac)),
            "live_ee_servo_nominal_reg": float(nominal_reg),
        }
    )
    return action_idx.astype(np.int64, copy=False), command.astype(np.float32, copy=False), info


def _phase_reanchor_action(env, safe_env_action, q_full, oscbf, args, state):
    chunk, was_single_chunk = _as_chunk(safe_env_action)
    raw_first = _raw_scaled_first_action(env, chunk[0])
    if raw_first is None:
        return None, None

    raw_first = np.asarray(raw_first, dtype=np.float32).reshape(-1).copy()
    phase = str(state.get("phase", "pre_grasp"))
    task_point_xy = state.get("control_task_point_xy", state.get("task_point_xy"))
    if task_point_xy is None:
        ee_xy = np.asarray(state["ee_pos"][:2], dtype=np.float64)
    else:
        ee_xy = np.asarray(task_point_xy, dtype=np.float64).reshape(2)
    target_ee_xy = state.get("target_ee_xy")
    if target_ee_xy is None:
        target_ee_xy = np.asarray(state["handle_pos"][:2], dtype=np.float64) + _phase_reanchor_offset_xy(args, phase)
    else:
        target_ee_xy = np.asarray(target_ee_xy, dtype=np.float64).reshape(2)
    ee_error_xy = state.get("control_ee_to_target_xy", state.get("ee_to_target_xy"))
    if ee_error_xy is None:
        ee_error_xy = target_ee_xy - ee_xy
    else:
        ee_error_xy = np.asarray(ee_error_xy, dtype=np.float64).reshape(2)
    base_cmd_zeroed_reason = None
    handle_assist_enabled = False
    handle_assist_reason = "disabled"
    handle_assist_error_norm = None
    handle_assist_base_cmd_xy = None
    preload_pull_probe_enabled = False
    preload_pull_probe_reason = "disabled"
    preload_pull_probe_axis_xy = None
    preload_pull_probe_step = None
    preload_pull_probe_delta_norm = None
    if phase == "pull":
        base_cmd_xy = np.zeros(2, dtype=np.float32)
        base_cmd_zeroed_reason = "pull_phase"
    else:
        max_base_step_xy = getattr(args, "phase_reanchor_max_base_step_xy", None)
        if max_base_step_xy is None:
            max_base_step_xy = np.full(2, float(args.phase_reanchor_max_base_step), dtype=np.float64)
        else:
            try:
                max_base_step_xy = np.asarray(max_base_step_xy, dtype=np.float64).reshape(2)
            except Exception:
                max_base_step_xy = np.full(2, float(args.phase_reanchor_max_base_step), dtype=np.float64)
            max_base_step_xy = np.maximum(max_base_step_xy, 1e-6)
        command_error_xy = ee_error_xy
        command_gain = float(args.phase_reanchor_base_gain)
        if bool(getattr(args, "phase_reanchor_live_handle_assist", False)):
            try:
                live_target_dist = float(state.get("ee_to_target_dist", np.inf))
                live_handle_dist = float(state.get("ee_to_handle_dist", np.inf))
                trigger_dist = float(getattr(args, "phase_reanchor_live_handle_assist_trigger_target_dist", 0.22))
                handle_limit = float(getattr(args, "phase_reanchor_live_release_handle_dist", 0.24))
                handle_error_xy = np.asarray(state.get("ee_to_handle_xy"), dtype=np.float64).reshape(2)
                geometry_trusted = not bool(state.get("task_point_geometry_untrusted", False))
                if (
                    phase != "pull"
                    and geometry_trusted
                    and np.isfinite(live_target_dist)
                    and np.isfinite(live_handle_dist)
                    and live_target_dist <= trigger_dist
                    and live_handle_dist > handle_limit
                    and np.isfinite(handle_error_xy).all()
                ):
                    command_error_xy = handle_error_xy
                    assist_max_base_step_xy = getattr(
                        args,
                        "phase_reanchor_live_handle_assist_max_base_step_xy",
                        None,
                    )
                    if assist_max_base_step_xy is not None:
                        try:
                            max_base_step_xy = np.asarray(
                                assist_max_base_step_xy,
                                dtype=np.float64,
                            ).reshape(2)
                            max_base_step_xy = np.maximum(max_base_step_xy, 1e-6)
                        except Exception:  # noqa: BLE001
                            pass
                    command_gain = float(getattr(args, "phase_reanchor_live_handle_assist_gain", command_gain))
                    handle_assist_enabled = True
                    handle_assist_reason = "near_release_handle_gap"
                    handle_assist_error_norm = float(np.linalg.norm(handle_error_xy))
                else:
                    handle_assist_reason = "not_triggered"
            except Exception:  # noqa: BLE001
                handle_assist_reason = "invalid_handle_assist_state"
        base_cmd_xy = np.clip(
            command_gain * command_error_xy,
            -max_base_step_xy,
            max_base_step_xy,
        ).astype(np.float32)
        if handle_assist_enabled:
            handle_assist_base_cmd_xy = base_cmd_xy.copy()

    if bool(getattr(args, "phase_reanchor_bridge_preload_pull_probe", False)) and phase == "grasp":
        preload_pull_probe_reason = "not_triggered"
        try:
            probe_handle_dist = float(state.get("gripper_to_handle_dist"))
            probe_limit = float(
                getattr(
                    args,
                    "phase_reanchor_bridge_handle_dist",
                    getattr(args, "phase_reanchor_live_release_handle_dist", 0.24),
                )
            )
            geometry_trusted = not bool(state.get("task_point_geometry_untrusted", False))
            if geometry_trusted and np.isfinite(probe_handle_dist) and probe_handle_dist <= probe_limit:
                pull_axis = np.asarray(
                    getattr(args, "phase_reanchor_pull_offset_xy", [0.0, -0.1]),
                    dtype=np.float64,
                ).reshape(2)
                axis_norm = float(np.linalg.norm(pull_axis))
                if not np.isfinite(axis_norm) or axis_norm < 1e-6:
                    pull_axis = np.asarray([0.0, -1.0], dtype=np.float64)
                else:
                    pull_axis = pull_axis / axis_norm
                pull_step = float(getattr(args, "phase_reanchor_bridge_preload_pull_step", 0.006))
                previous_base_cmd = np.asarray(base_cmd_xy, dtype=np.float32).copy()
                base_cmd_xy = np.clip(
                    np.asarray(base_cmd_xy, dtype=np.float64) + pull_step * pull_axis,
                    -max_base_step_xy,
                    max_base_step_xy,
                ).astype(np.float32)
                preload_pull_probe_enabled = True
                preload_pull_probe_reason = "near_handle_pull_probe"
                preload_pull_probe_axis_xy = pull_axis.astype(float).tolist()
                preload_pull_probe_step = float(pull_step)
                preload_pull_probe_delta_norm = float(np.linalg.norm(base_cmd_xy - previous_base_cmd))
            elif not geometry_trusted:
                preload_pull_probe_reason = "geometry_untrusted"
            else:
                preload_pull_probe_reason = "handle_not_ready"
        except Exception:  # noqa: BLE001
            preload_pull_probe_reason = "invalid_probe_state"

    if raw_first.shape[0] >= 2:
        raw_first[:2] = base_cmd_xy
    if raw_first.shape[0] >= 4:
        raw_first[2:4] = 0.0

    action_idx = np.asarray(getattr(oscbf, "bigym_action_arm_indices", []), dtype=np.int64)
    state_idx = np.asarray(getattr(oscbf, "bigym_state_arm_indices", []), dtype=np.int64)
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    pair_count = min(action_idx.size, state_idx.size)
    action_idx = action_idx[:pair_count]
    state_idx = state_idx[:pair_count]
    valid = (action_idx < raw_first.shape[0]) & (state_idx < q.shape[0])
    arm_hold_enabled = bool(phase == "pull")
    arm_hold_reason = "pull_phase" if arm_hold_enabled else None
    if arm_hold_enabled and np.any(valid):
        raw_first[action_idx[valid]] = q[state_idx[valid]]
    arm_servo_info = {
        "arm_servo_enabled": False,
        "arm_servo_reason": "not_attempted",
        "arm_servo_rank": None,
        "arm_servo_error_norm": None,
        "arm_servo_delta_norm": None,
        "arm_servo_command_norm": None,
        "arm_servo_action_delta_norm": None,
        "live_ee_servo_enabled": False,
        "live_ee_servo_reason": "not_attempted",
        "live_ee_servo_error_norm": None,
        "live_ee_servo_delta_norm": None,
        "live_ee_servo_command_norm": None,
        "live_ee_servo_jacobian_rank": None,
        "live_ee_servo_nominal_reg": None,
        "live_ee_servo_fk_site_xy_error": None,
        "live_ee_servo_fk_gripper_xy_error": None,
        "live_ee_servo_predicted_error_before": None,
        "live_ee_servo_predicted_error_after": None,
        "live_ee_servo_geometry_untrusted": None,
    }
    near_live_target_suppress_q_servo = False
    if bool(getattr(args, "phase_reanchor_near_live_target_suppress_q_servo", False)):
        try:
            live_target_dist = float(state.get("ee_to_target_dist", np.inf))
            suppress_dist = float(getattr(args, "phase_reanchor_near_live_target_suppress_dist", 0.22))
            geometry_trusted = not bool(state.get("task_point_geometry_untrusted", False))
            near_live_target_suppress_q_servo = bool(
                geometry_trusted
                and np.isfinite(live_target_dist)
                and live_target_dist <= suppress_dist
            )
        except Exception:  # noqa: BLE001
            near_live_target_suppress_q_servo = False
    if handle_assist_enabled:
        arm_servo_info["arm_servo_reason"] = "handle_assist_suppresses_q_servo"
    elif near_live_target_suppress_q_servo:
        arm_servo_info["arm_servo_reason"] = "near_live_target_suppresses_q_servo"
    if not arm_hold_enabled and not handle_assist_enabled and not near_live_target_suppress_q_servo:
        servo_action_idx, servo_command, arm_servo_info = _phase_reanchor_arm_q_window_command(
            env,
            chunk,
            q_full,
            oscbf,
            args,
            state,
        )
        if servo_action_idx is not None and servo_command is not None and len(servo_action_idx):
            servo_action_idx = np.asarray(servo_action_idx, dtype=np.int64)
            valid_servo = servo_action_idx < raw_first.shape[0]
            if np.any(valid_servo):
                servo_action_idx = servo_action_idx[valid_servo]
                servo_command = np.asarray(servo_command, dtype=np.float32).reshape(-1)[valid_servo]
                previous_arm = raw_first[servo_action_idx].copy()
                mix = float(getattr(args, "phase_reanchor_arm_servo_mix", 0.75))
                mix = float(np.clip(mix, 0.0, 1.0))
                raw_first[servo_action_idx] = (1.0 - mix) * previous_arm + mix * servo_command
                arm_servo_info["arm_servo_action_delta_norm"] = float(
                    np.linalg.norm(raw_first[servo_action_idx] - previous_arm)
                )
        live_ee_with_q_window = bool(getattr(args, "phase_reanchor_live_ee_servo_with_q_window", False))
        if bool(state.get("nominal_reentry_suppress_q_servo", False)) or live_ee_with_q_window:
            live_action_idx, live_command, live_info = _phase_reanchor_live_ee_arm_command(
                q_full,
                oscbf,
                args,
                state,
                target_ee_xy,
                ee_xy,
                action_idx,
                state_idx,
                raw_first.shape[0],
            )
            arm_servo_info.update(live_info)
            live_command_applied = False
            if live_action_idx is not None and live_command is not None and len(live_action_idx):
                live_action_idx = np.asarray(live_action_idx, dtype=np.int64)
                valid_live = live_action_idx < raw_first.shape[0]
                if np.any(valid_live):
                    live_action_idx = live_action_idx[valid_live]
                    live_command = np.asarray(live_command, dtype=np.float32).reshape(-1)[valid_live]
                    previous_arm = raw_first[live_action_idx].copy()
                    mix = float(getattr(args, "phase_reanchor_live_ee_servo_mix", 1.0))
                    mix = float(np.clip(mix, 0.0, 1.0))
                    raw_first[live_action_idx] = (1.0 - mix) * previous_arm + mix * live_command
                    arm_servo_info["arm_servo_enabled"] = True
                    arm_servo_info["arm_servo_reason"] = live_info.get(
                        "live_ee_servo_reason",
                        "live_ee_taskspace_servo",
                    )
                    arm_servo_info["arm_servo_rank"] = int(live_action_idx.size)
                    arm_servo_info["arm_servo_error_norm"] = live_info.get("live_ee_servo_error_norm")
                    arm_servo_info["arm_servo_delta_norm"] = live_info.get("live_ee_servo_delta_norm")
                    arm_servo_info["arm_servo_command_norm"] = live_info.get("live_ee_servo_command_norm")
                    arm_servo_info["arm_servo_action_delta_norm"] = float(
                        np.linalg.norm(raw_first[live_action_idx] - previous_arm)
                    )
                    live_command_applied = True
            if (
                not live_command_applied
                and bool(getattr(args, "phase_reanchor_hold_arm_when_q_servo_suppressed", True))
                and np.any(valid)
            ):
                previous_arm = raw_first[action_idx[valid]].copy()
                hold_command = q[state_idx[valid]].astype(np.float32, copy=False)
                raw_first[action_idx[valid]] = hold_command
                arm_hold_enabled = True
                arm_hold_reason = "live_taskspace_worsening_q_servo_suppressed"
                arm_servo_info.update(
                    {
                        "arm_servo_enabled": True,
                        "arm_servo_reason": "live_taskspace_worsening_arm_hold",
                        "arm_servo_rank": int(np.count_nonzero(valid)),
                        "arm_servo_error_norm": 0.0,
                        "arm_servo_delta_norm": 0.0,
                        "arm_servo_command_norm": float(np.linalg.norm(hold_command)),
                        "arm_servo_action_delta_norm": float(
                            np.linalg.norm(raw_first[action_idx[valid]] - previous_arm)
                        ),
                        "suppressed_q_servo_arm_hold": True,
                    }
                )

    normalized_first = _raw_action_to_normalized(env, raw_first)
    if normalized_first is None:
        return None, None
    normalized_first = np.asarray(normalized_first, dtype=np.float32).reshape(-1)
    preload_gripper_forced = False
    preload_gripper_limit = float(
        getattr(
            args,
            "phase_reanchor_bridge_handle_dist",
            getattr(args, "phase_reanchor_grasp_dist", 0.12),
        )
    )
    preload_gripper_handle_dist = state.get("gripper_to_handle_dist")
    try:
        preload_gripper_handle_dist = float(preload_gripper_handle_dist)
        if not np.isfinite(preload_gripper_handle_dist):
            preload_gripper_handle_dist = None
    except (TypeError, ValueError):
        preload_gripper_handle_dist = None
    if (
        phase in {"grasp", "pull"}
        or (
            bool(getattr(args, "phase_reanchor_force_gripper_during_preload", False))
            and preload_gripper_handle_dist is not None
            and preload_gripper_handle_dist <= preload_gripper_limit
        )
    ):
        normalized_first[-1] = float(np.clip(args.phase_reanchor_gripper_value, -1.0, 1.0))
        preload_gripper_forced = True
    effective_raw_first = _raw_scaled_first_action(env, normalized_first)
    effective_base_cmd_xy = None
    base_cmd_clip_delta_norm = None
    base_cmd_normalized_xy = None
    if raw_first.shape[0] >= 2:
        base_cmd_normalized_xy = normalized_first[:2].astype(float).tolist()
        if effective_raw_first is not None:
            effective_raw_first = np.asarray(effective_raw_first, dtype=np.float32).reshape(-1)
            if effective_raw_first.shape[0] >= 2:
                effective_base_cmd = effective_raw_first[:2].astype(np.float32, copy=False)
                effective_base_cmd_xy = effective_base_cmd.astype(float).tolist()
                base_cmd_clip_delta_norm = float(np.linalg.norm(effective_base_cmd - raw_first[:2]))

    reanchored_chunk = np.repeat(normalized_first[None, :], chunk.shape[0], axis=0)
    info = {
        "phase": phase,
        "target_ee_xy": target_ee_xy.astype(float).tolist(),
        "ee_error_xy": ee_error_xy.astype(float).tolist(),
        "ee_to_target_dist": float(state.get("ee_to_target_dist", np.linalg.norm(ee_error_xy))),
        "base_cmd_xy": base_cmd_xy.astype(float).tolist(),
        "base_cmd_normalized_xy": base_cmd_normalized_xy,
        "base_cmd_effective_raw_xy": effective_base_cmd_xy,
        "base_cmd_clip_delta_norm": base_cmd_clip_delta_norm,
        "base_cmd_zeroed_reason": base_cmd_zeroed_reason,
        "preload_gripper_forced": bool(preload_gripper_forced),
        "preload_gripper_limit": float(preload_gripper_limit),
        "preload_pull_probe_enabled": bool(preload_pull_probe_enabled),
        "preload_pull_probe_reason": preload_pull_probe_reason,
        "preload_pull_probe_axis_xy": preload_pull_probe_axis_xy,
        "preload_pull_probe_step": preload_pull_probe_step,
        "preload_pull_probe_delta_norm": preload_pull_probe_delta_norm,
        "handle_assist_enabled": handle_assist_enabled,
        "handle_assist_reason": handle_assist_reason,
        "handle_assist_error_norm": handle_assist_error_norm,
        "handle_assist_base_cmd_xy": (
            handle_assist_base_cmd_xy.astype(float).tolist()
            if handle_assist_base_cmd_xy is not None
            else None
        ),
        "preload_target_grasp": bool(state.get("preload_target_grasp", False)),
        "preload_grasp_limit": state.get("preload_grasp_limit"),
        "arm_hold_enabled": bool(arm_hold_enabled),
        "arm_hold_reason": arm_hold_reason,
        "drawer_open_fraction": float(state.get("drawer_open_fraction", 0.0)),
        "ee_to_handle_dist": float(state.get("ee_to_handle_dist", np.nan)),
        "task_point_source": state.get("task_point_source"),
        "task_point_requested_source": state.get("task_point_requested_source"),
        "task_point_fallback_reason": state.get("task_point_fallback_reason"),
        "control_task_point_source": state.get("control_task_point_source"),
        "control_task_point_requested_source": state.get("control_task_point_requested_source"),
        "control_task_point_fallback_reason": state.get("control_task_point_fallback_reason"),
        "control_ee_to_handle_dist": state.get("control_ee_to_handle_dist"),
        "control_ee_to_target_dist": state.get("control_ee_to_target_dist"),
        "site_ee_to_handle_dist": state.get("site_ee_to_handle_dist"),
        "site_ee_to_target_dist": state.get("site_ee_to_target_dist"),
        "gripper_to_handle_dist": state.get("gripper_to_handle_dist"),
        "gripper_to_target_dist": state.get("gripper_to_target_dist"),
        "gripper_site_xy_error": state.get("gripper_site_xy_error"),
        "task_point_geometry_untrusted": state.get("task_point_geometry_untrusted"),
    }
    info.update(arm_servo_info)
    return _restore_action_shape(reanchored_chunk, was_single_chunk), info


def _data_position(data, idx: int, *field_names):
    for field_name in field_names:
        values = getattr(data, field_name, None)
        if values is None:
            continue
        try:
            return np.asarray(values[int(idx)], dtype=np.float64).reshape(3)
        except Exception:  # noqa: BLE001
            continue
    return None


def _mujoco_named_position(model, data, obj_type, names: Sequence[str], *field_names):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None, None

    for name in names:
        obj_id = mujoco.mj_name2id(model, obj_type, name)
        if obj_id < 0:
            continue
        pos = _data_position(data, int(obj_id), *field_names)
        if pos is not None and np.isfinite(pos).all():
            return pos, name
    return None, None


def _human_arm_wrist_position(model, data):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None, None

    pos, name = _mujoco_named_position(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        ("cylinder_arm/wrist", "wrist"),
        "xpos",
        "body_xpos",
    )
    if pos is not None:
        return pos, name

    pos, name = _mujoco_named_position(
        model,
        data,
        mujoco.mjtObj.mjOBJ_GEOM,
        ("cylinder_arm/vis_wrist", "vis_wrist"),
        "geom_xpos",
    )
    if pos is not None:
        return pos, name

    return None, None


def _human_arm_trajectory_sample(env, episode: int, step: int):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None

    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if base_env is None:
        return None
    try:
        model = base_env.mojo.physics.model.ptr
        data = base_env.mojo.physics.data
    except Exception:  # noqa: BLE001
        try:
            model = base_env.mojo.model
            data = base_env.mojo.data
        except Exception:  # noqa: BLE001
            return None

    geoms = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        lower = name.lower()
        if not (
            lower.startswith("cylinder_arm/")
            or lower.endswith("upperarm_geom")
            or lower.endswith("forearm_geom")
        ):
            continue
        try:
            pos = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
        except Exception:  # noqa: BLE001
            continue
        geoms.append({"name": name, "pos": pos.astype(float).tolist()})

    wrist_pos, wrist_name = _human_arm_wrist_position(model, data)
    if not geoms and wrist_pos is None:
        return None

    sample = {
        "episode": int(episode),
        "step": int(step),
        "time": float(getattr(data, "time", np.nan)),
        "geoms": geoms,
    }
    if geoms:
        centers = np.asarray([g["pos"] for g in geoms], dtype=np.float64)
        sample["center"] = np.mean(centers, axis=0).astype(float).tolist()
    if wrist_pos is not None:
        sample["wrist_name"] = str(wrist_name)
        sample["wrist_pos"] = np.asarray(wrist_pos, dtype=np.float64).astype(float).tolist()
    return sample


def _robot_ee_position(model, data):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None, None

    pos, name = _mujoco_named_position(
        model,
        data,
        mujoco.mjtObj.mjOBJ_SITE,
        (
            "right_wrist",
            "h1/right_wrist",
            "right_wrist_yaw",
            "h1/right_wrist_yaw",
            "wrist",
            "h1/wrist",
            "right_end_effector",
            "h1/right_end_effector",
            "right_gripper",
            "h1/right_gripper",
        ),
        "site_xpos",
    )
    if pos is not None:
        return pos, f"site:{name}"

    priority_patterns = (
        ("right_wrist",),
        ("robotiq_2f85_right", "driver"),
        ("robotiq_2f85_right",),
        ("robotiq_2f85_right", "finger"),
        ("robotiq_2f85_right", "pad"),
    )
    exclude_patterns = ("visual", "camera", "left")
    for patterns in priority_patterns:
        points = []
        names = []
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            lower = geom_name.lower()
            if not lower or any(excluded in lower for excluded in exclude_patterns):
                continue
            if all(pattern in lower for pattern in patterns):
                pos = _data_position(data, geom_id, "geom_xpos")
                if pos is not None and np.isfinite(pos).all():
                    points.append(pos)
                    names.append(geom_name)
        if points:
            return np.mean(np.stack(points, axis=0), axis=0), "geom_average:" + ",".join(names[:4])
    return None, None


def _robot_ee_trajectory_sample(
    env,
    episode: int,
    step: int,
    task_state=None,
    horizon_operator=None,
    q_full=None,
):
    pos = None
    source = None
    data = None

    if horizon_operator is not None and q_full is not None:
        try:
            candidate = horizon_operator.ee_pose(np.asarray(q_full, dtype=np.float32))
            if candidate is not None:
                candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
                if candidate.size >= 3 and np.isfinite(candidate[:3]).all():
                    pos = candidate[:3]
                    source = "safety_model_ee"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Safety-model executed EE extraction failed: %s", exc)

    try:
        task = get_bigym_task(env)
        model, data = _mujoco_model_data(task)
        if pos is None and model is not None and data is not None:
            pos, source = _robot_ee_position(model, data)
    except Exception:  # noqa: BLE001
        if pos is None:
            source = None

    if pos is None and isinstance(task_state, dict):
        object_state = task_state.get("object_state")
        if isinstance(object_state, dict) and object_state.get("ee_pos") is not None:
            try:
                candidate = np.asarray(object_state.get("ee_pos"), dtype=np.float64).reshape(3)
                if np.isfinite(candidate).all():
                    pos = candidate
                    source = "diagnostic_task_state.ee_pos"
            except Exception:  # noqa: BLE001
                pos = None

    if pos is None:
        return None

    sample = {
        "episode": int(episode),
        "step": int(step),
        "time": float(getattr(data, "time", np.nan)) if data is not None else None,
        "ee_pos": np.asarray(pos, dtype=np.float64).astype(float).tolist(),
        "source": source,
    }
    if isinstance(task_state, dict):
        for key in ("drawer_open_distance", "drawer_open_fraction", "drawer_joint_position"):
            value = task_state.get(key)
            if value is not None:
                try:
                    sample[key] = float(value)
                except (TypeError, ValueError):
                    pass
        object_state = task_state.get("object_state")
        if isinstance(object_state, dict):
            copied_state = {}
            for key in ("handle_pos", "drawer_open_distance", "drawer_open_fraction", "drawer_scene_geometry"):
                value = object_state.get(key)
                if value is not None:
                    copied_state[key] = _jsonable_trace_value(value)
            if copied_state:
                sample["object_state"] = copied_state
            if object_state.get("handle_pos") is not None:
                try:
                    handle_pos = np.asarray(object_state.get("handle_pos"), dtype=np.float64).reshape(3)
                    if np.isfinite(handle_pos).all():
                        sample["handle_pos"] = handle_pos.astype(float).tolist()
                except Exception:  # noqa: BLE001
                    pass
    return sample
