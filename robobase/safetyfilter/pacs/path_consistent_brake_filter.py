"""Path-consistent braking filter.

This module intentionally uses source-backed SafeChunk-Deform code instead of
loading a sourceless ``.pyc`` implementation. The previous cache loader could
re-enter this wrapper from ``__pycache__`` and fail before eval startup.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robobase.safetyfilter.safechunkdeform.safechunk_deform_filter import (
    SafeChunkDeformFilter,
)


_FILTER_KEYS = {
    "oscbf_operator",
    "horizon",
    "dt",
    "action_dim",
    "expected_motion_dim",
    "control_type",
    "controlled_action_indices",
    "controlled_state_indices",
    "min_clearance",
    "diagnostics",
    "debug",
    "enabled",
}

_BRAKE_KEYS = {
    "brake_progress_threshold",
    "deadlock_window",
    "temporary_blocker",
    "safechunk_active_safety",
}

_DEFORM_KEYS = {
    "deformation_enabled",
    "mode",
    "chunk_deformation_scales",
    "chunk_deformation_smoothing",
    "sequential_oscbf_fallback",
    "deform_after_deadlock_window",
    "unsafe_deformation_fallback",
    "optimized_fallback",
    "detach_passthrough_dims",
    "opt_iters",
    "opt_lr",
    "opt_population",
    "opt_elite_frac",
    "opt_seed",
    "lambda_safety",
    "lambda_action",
    "lambda_path",
    "lambda_smooth",
    "optimized_deform",
    "safechunk_acceptance",
    "debug_safety_feasibility",
    "action_low",
    "action_high",
    "max_action_delta",
}

_RECOVERY_KEYS = {
    "recoverable_deform",
    "explicit_recovery",
    "safechunk_replan",
    "safechunk_recover",
    "safechunk_recovery_corridor",
    "lambda_rejoin",
    "rejoin_threshold",
    "min_rejoin_offset",
    "use_ee_pose_rejoin",
    "use_object_state_rejoin",
    "brake_if_unrecoverable",
}

_PATH_CONSISTENT_KEYS = {
    "waypoint_substeps",
    "max_waypoint_delta",
    "slowdown_enabled",
    "slowdown_lookahead",
    "slowdown_min_scale",
    "certified_backup_enabled",
    "trajectory_generation_enabled",
    "trajectory_max_velocity",
    "trajectory_max_acceleration",
    "trajectory_max_jerk",
    "trajectory_initial_speed",
    "trajectory_backend",
    "trajectory_min_position",
    "trajectory_max_position",
    "shield_substeps",
    "inner_shield_verification_enabled",
    "skip_inner_shield_when_rejected",
    "reuse_operator_human_rollout_cache",
    "background_check_only",
    "reachability_certification_enabled",
    "reachability_fail_closed",
    "reachability_robot_radius",
    "reachability_obstacle_radius",
    "reachability_robot_points_source",
    "reachability_inflation_enabled",
    "reachability_tracking_error",
    "reachability_measurement_error",
    "reachability_object_speed",
    "reachability_object_acceleration",
    "reachability_sensor_delay",
    "safety_constraint_type",
    "pfl_energy_threshold",
    "pfl_contact_margin",
    "pfl_joint_inertia",
    "pfl_energy_thresholds",
    "pfl_active_threshold_key",
}


class PathRetimingResult:
    """Small compatibility container for old PACS retiming diagnostics."""

    def __init__(self, *values: Any, **fields: Any) -> None:
        self.values = tuple(values)
        self.__dict__.update(fields)

    def asdict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k != "values"}
        if self.values:
            data["values"] = self.values
        return data


def _rename_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value == "horizon_brake":
        return "path_consistent_brake"
    if value == "horizon_brake_intended_step":
        return "path_consistent_brake_intended_step"
    if value == "horizon_brake_slowdown":
        return "path_consistent_brake_slowdown"
    return value


def _rename_info_keys(info: dict[str, Any]) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    for key, value in info.items():
        new_key = key
        if key.startswith("horizon_brake_"):
            new_key = "path_consistent_brake_" + key[len("horizon_brake_"):]
        renamed[new_key] = _rename_value(value)
    if renamed.get("filter_name") == "horizon_brake":
        renamed["filter_name"] = "path_consistent_brake"
    source = renamed.get("deformation_source")
    if (
        renamed.get("retiming_source") is None
        and source in {"path_consistent_brake", "path_consistent_brake_slowdown"}
    ):
        renamed["retiming_source"] = source
    if "retiming_norm" not in renamed and "deformation_norm" in renamed:
        renamed["retiming_norm"] = renamed.get("deformation_norm")
    return renamed


def _nested_cfg_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "safety_filter": {},
        "intervention": {"brake": {}, "deform": {}, "recovery": {}},
        "path_consistent_brake": {},
    }
    for key, value in kwargs.items():
        if key in _FILTER_KEYS:
            cfg["safety_filter"][key] = value
        elif key in _BRAKE_KEYS:
            cfg["intervention"]["brake"][key] = value
        elif key in _DEFORM_KEYS:
            cfg["intervention"]["deform"][key] = value
        elif key in _RECOVERY_KEYS:
            cfg["intervention"]["recovery"][key] = value
        elif key in _PATH_CONSISTENT_KEYS:
            cfg["path_consistent_brake"][key] = value
        else:
            # Keep unknown legacy kwargs visible to SafeChunkDeformFilter's
            # compatibility lookup instead of dropping user-provided settings.
            cfg[key] = value
    return cfg


class PathConsistentBrakeFilter(SafeChunkDeformFilter):
    """Source-backed PACS-compatible braking filter.

    The old experimental implementation was only available as a compiled cache
    artifact in this checkout. This wrapper keeps the public constructor and info
    keys used by the eval script while delegating safety rollout, braking, and
    recovery behavior to the maintained SafeChunk-Deform implementation.
    """

    def __init__(self, cfg: Any | None = None, **kwargs: Any) -> None:
        if cfg is None:
            cfg_dict = _nested_cfg_from_kwargs(dict(kwargs))
        else:
            cfg_dict = self._cfg_to_dict(cfg)
            if kwargs:
                merged = dict(cfg_dict)
                merged.update(kwargs)
                cfg_dict = _nested_cfg_from_kwargs(merged)
        self.path_consistent_config = dict(cfg_dict.get("path_consistent_brake", {}))
        super().__init__(cfg=cfg_dict)

    def _control_mode_ids_for_state_indices(self, state_idx: Any) -> np.ndarray:
        return self.intervention_factory._control_mode_ids_for_state_indices(state_idx)

    def _action_from_q_transition(self, action, q_prev, q_next, action_idx, state_idx):
        safe = np.asarray(action, dtype=np.float32).copy()
        q_prev = np.asarray(q_prev, dtype=np.float32).reshape(-1)
        q_next = np.asarray(q_next, dtype=np.float32).reshape(-1)
        action_idx = np.asarray(action_idx, dtype=np.int64).reshape(-1)
        state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
        valid = (
            (action_idx < safe.shape[0])
            & (state_idx < q_prev.shape[0])
            & (state_idx < q_next.shape[0])
        )
        if not np.any(valid):
            return safe
        action_idx = action_idx[valid]
        state_idx = state_idx[valid]
        modes = self._control_mode_ids_for_state_indices(state_idx)
        value = q_next[state_idx].copy()
        delta = q_next[state_idx] - q_prev[state_idx]
        delta_mask = modes == 1
        velocity_mask = modes == 2
        value[delta_mask] = delta[delta_mask]
        value[velocity_mask] = delta[velocity_mask] / float(self.dt)
        safe[action_idx] = value
        return safe

    def _chunk_from_q_sequence(self, reference_chunk, q0, q_seq, action_idx, state_idx):
        chunk = np.asarray(reference_chunk, dtype=np.float32).copy()
        q_seq = np.asarray(q_seq, dtype=np.float32)
        if chunk.ndim != 2 or q_seq.ndim != 2:
            return chunk
        q_prev = np.asarray(q0, dtype=np.float32).reshape(-1)
        horizon = min(chunk.shape[0], q_seq.shape[0])
        for k in range(horizon):
            chunk[k] = self._action_from_q_transition(
                chunk[k], q_prev, q_seq[k], action_idx, state_idx
            )
            q_prev = q_seq[k]
        return chunk

    def _make_hold_chunk_from_q(self, obs, reference_chunk, q_anchor, horizon: int):
        del obs
        reference = np.asarray(reference_chunk, dtype=np.float32)
        horizon = max(0, int(horizon))
        if reference.ndim != 2:
            return reference.copy()
        chunk = np.zeros((horizon, reference.shape[1]), dtype=np.float32)
        if horizon == 0:
            return chunk
        rows = min(horizon, reference.shape[0])
        chunk[:rows] = reference[:rows]
        if rows < horizon and reference.shape[0] > 0:
            chunk[rows:] = reference[-1]
        q = np.asarray(q_anchor, dtype=np.float32).reshape(-1)
        valid = (
            (self.controlled_action_indices < chunk.shape[1])
            & (self.controlled_state_indices < q.shape[0])
        )
        if np.any(valid):
            action_idx = self.controlled_action_indices[valid]
            state_idx = self.controlled_state_indices[valid]
            anchor = np.zeros(action_idx.shape, dtype=chunk.dtype)
            modes = self._control_mode_ids_for_state_indices(state_idx)
            absolute = modes == 0
            if np.any(absolute):
                anchor[absolute] = q[state_idx[absolute]].astype(chunk.dtype, copy=False)
            chunk[:, action_idx] = anchor[None, :]
        return chunk

    def filter_chunk(self, obs, action_chunk, **kwargs):
        nominal_chunk = np.asarray(action_chunk).copy()
        safe_chunk, info = super().filter_chunk(obs, action_chunk, **kwargs)
        info = _rename_info_keys(dict(info))
        mode = info.get("safety_mode") or info.get("mode")
        if mode == "path_consistent_brake_intended_step":
            info.update(
                {
                    "path_consistent_brake_intended_passthrough": True,
                    "path_consistent_brake_passthrough_reason": (
                        "certified_intended_chunk"
                    ),
                    "deformation_source": None,
                    "deformation_norm": 0.0,
                    "retiming_source": None,
                    "retiming_norm": 0.0,
                }
            )
            self.last_info = info
            return nominal_chunk, info
        info["path_consistent_brake_intended_passthrough"] = False
        self.last_info = info
        return safe_chunk, info

    def make_path_consistent_failsafe_tail(self, obs, reference_chunk, q_anchor=None, horizon=None, *args, **kwargs):
        del args, kwargs
        if horizon is None:
            horizon = np.asarray(reference_chunk).shape[0]
        if q_anchor is None:
            return np.asarray(reference_chunk, dtype=np.float32).copy()
        return self._make_hold_chunk_from_q(obs, reference_chunk, q_anchor, int(horizon))

    def path_consistent_slowdown_or_brake(self, obs, action_chunk, safety_info=None, *args, **kwargs):
        del args, kwargs
        chunk, info = self.brake.horizon_brake(obs, action_chunk, safety_info or {})
        info.update({"safety_mode": "horizon_brake", "mode": "horizon_brake"})
        return chunk, _rename_info_keys(dict(info))


__all__ = ["PathConsistentBrakeFilter", "PathRetimingResult"]
