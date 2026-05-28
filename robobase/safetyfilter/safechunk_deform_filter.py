from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SafeChunkDeformFilter:
    """Chunk-level safety wrapper around a single-step OSCBF-style operator."""

    def __init__(
        self,
        oscbf_operator=None,
        horizon: int = 16,
        dt: float = 1.0 / 20.0,
        action_dim: int = 16,
        expected_motion_dim: int = 14,
        control_type: str = "absolute",
        controlled_action_indices=None,
        controlled_state_indices=None,
        min_clearance: float = 0.08,
        brake_progress_threshold: float = 0.05,
        deadlock_window: int = 5,
        deformation_enabled: bool = True,
        debug: bool = True,
        enabled: bool = True,
        **kwargs,
    ):
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        if kwargs:
            logger.warning("Unused SafeChunkDeformFilter kwargs: %s", kwargs)

        if control_type not in {"absolute", "delta", "velocity"}:
            raise ValueError(
                "control_type must be one of ['absolute', 'delta', 'velocity'], "
                f"got {control_type}"
            )

        self.oscbf_operator = oscbf_operator
        self._operator_instantiation_failed = False
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.action_dim = int(action_dim)
        self.expected_motion_dim = int(expected_motion_dim)
        self.control_type = control_type
        self.controlled_action_indices = np.asarray(
            controlled_action_indices
            if controlled_action_indices is not None
            else [4, 5, 6, 7, 9, 10, 11, 12],
            dtype=np.int64,
        )
        self.controlled_state_indices = np.asarray(
            controlled_state_indices
            if controlled_state_indices is not None
            else [4, 5, 6, 7, 9, 10, 11, 12],
            dtype=np.int64,
        )
        self.min_clearance = float(min_clearance)
        self.brake_progress_threshold = float(brake_progress_threshold)
        self.deadlock_window = int(deadlock_window)
        self.deformation_enabled = bool(deformation_enabled)
        self.debug = bool(debug)
        self.enabled = bool(enabled)
        self.last_info: dict[str, Any] = {}
        self._warned_no_safety_eval = False

    def _as_chunk(self, action) -> tuple[np.ndarray, bool]:
        arr = np.asarray(action)
        if arr.ndim == 1:
            if arr.shape[0] != self.action_dim:
                logger.debug(
                    "Single action dim %d differs from configured action_dim %d",
                    arr.shape[0],
                    self.action_dim,
                )
            return arr.reshape(1, -1).copy(), True
        if arr.ndim == 2:
            return arr.copy(), False
        raise ValueError(
            f"Expected action shape ({self.action_dim},) or (H, {self.action_dim}), "
            f"got {arr.shape}"
        )

    def extract_current_q(self, obs, action_chunk: Optional[np.ndarray] = None) -> np.ndarray:
        candidates = ("q", "qpos", "robot_state", "state")
        value = None
        if isinstance(obs, dict):
            for name in candidates:
                if name in obs:
                    value = obs[name]
                    break
        else:
            for name in candidates:
                if hasattr(obs, name):
                    value = getattr(obs, name)
                    break

        if value is not None:
            q = np.asarray(value, dtype=np.float32).reshape(-1)
            if q.shape[0] >= self.expected_motion_dim:
                return q[: self.expected_motion_dim].copy()
            padded = np.zeros(self.expected_motion_dim, dtype=np.float32)
            padded[: q.shape[0]] = q
            return padded

        q = np.zeros(self.expected_motion_dim, dtype=np.float32)
        if action_chunk is not None and action_chunk.size > 0:
            valid = self.controlled_state_indices < q.shape[0]
            q[self.controlled_state_indices[valid]] = action_chunk[0][
                self.controlled_action_indices[valid]
            ]
        return q

    def rollout_nominal_chunk(self, obs, action_chunk) -> np.ndarray:
        chunk, _ = self._as_chunk(action_chunk)
        q = self.extract_current_q(obs, chunk)
        q_seq = np.zeros((chunk.shape[0], q.shape[0]), dtype=np.float32)

        valid = (
            (self.controlled_state_indices < q.shape[0])
            & (self.controlled_action_indices < chunk.shape[1])
        )
        state_idx = self.controlled_state_indices[valid]
        action_idx = self.controlled_action_indices[valid]

        for k, action in enumerate(chunk):
            q_next = q.copy()
            if self.control_type == "absolute":
                q_next[state_idx] = action[action_idx]
            elif self.control_type == "delta":
                q_next[state_idx] += action[action_idx]
            elif self.control_type == "velocity":
                q_next[state_idx] += self.dt * action[action_idx]
            q_seq[k] = q_next
            q = q_next

        return q_seq

    def evaluate_horizon_safety(self, obs, q_seq) -> dict[str, Any]:
        q_seq = np.asarray(q_seq, dtype=np.float32)
        op = self._get_oscbf_operator()
        methods = (
            "compute_min_clearance",
            "get_min_clearance",
            "evaluate_safety",
            "is_safe",
        )

        for method_name in methods:
            method = getattr(op, method_name, None)
            if method is None:
                continue
            try:
                result = self._call_safety_method(method, obs, q_seq)
                return self._normalize_safety_result(result, q_seq.shape[0])
            except Exception as exc:  # pragma: no cover - defensive integration path
                logger.warning(
                    "SafeChunk-Deform safety evaluation via %s failed: %s",
                    method_name,
                    exc,
                )

        if not self._warned_no_safety_eval:
            logger.warning(
                "SafeChunk-Deform could not find a horizon clearance evaluator; "
                "using conservative pass-through horizon evaluation."
            )
            self._warned_no_safety_eval = True
        h = q_seq.shape[0]
        return {
            "horizon_safe": True,
            "min_clearance": float("inf"),
            "min_clearances": np.full(h, np.inf, dtype=np.float32),
            "first_violation": None,
            "unsafe_count": 0,
            "safety_eval_available": False,
        }

    def _call_safety_method(self, method, obs, q_seq):
        attempts = (
            lambda: method(obs=obs, q_seq=q_seq),
            lambda: method(q_seq=q_seq, obs=obs),
            lambda: method(obs, q_seq),
            lambda: method(q_seq),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        raise last_error

    def _normalize_safety_result(self, result, horizon: int) -> dict[str, Any]:
        if isinstance(result, dict):
            min_clearances = np.asarray(
                result.get("min_clearances", result.get("clearances", [])),
                dtype=np.float32,
            ).reshape(-1)
            if min_clearances.size == 0:
                min_clearance = float(result.get("min_clearance", np.inf))
                min_clearances = np.full(horizon, min_clearance, dtype=np.float32)
            horizon_safe = bool(
                result.get(
                    "horizon_safe",
                    result.get("safe", np.all(min_clearances >= self.min_clearance)),
                )
            )
            first_violation = result.get("first_violation")
            if first_violation is None:
                unsafe = np.flatnonzero(min_clearances < self.min_clearance)
                first_violation = int(unsafe[0]) if unsafe.size else None
            unsafe_count = int(
                result.get(
                    "unsafe_count",
                    np.count_nonzero(min_clearances < self.min_clearance),
                )
            )
            return {
                "horizon_safe": horizon_safe,
                "min_clearance": float(result.get("min_clearance", np.min(min_clearances))),
                "min_clearances": min_clearances,
                "first_violation": first_violation,
                "unsafe_count": unsafe_count,
                "safety_eval_available": True,
            }

        arr = np.asarray(result)
        if arr.dtype == np.bool_:
            horizon_safe = bool(arr.all())
            min_clearances = np.full(horizon, np.inf if horizon_safe else -np.inf)
        else:
            min_clearances = arr.astype(np.float32).reshape(-1)
            if min_clearances.size == 1 and horizon > 1:
                min_clearances = np.full(horizon, float(min_clearances[0]), dtype=np.float32)
            horizon_safe = bool(np.all(min_clearances >= self.min_clearance))
        unsafe = np.flatnonzero(min_clearances < self.min_clearance)
        return {
            "horizon_safe": horizon_safe,
            "min_clearance": float(np.min(min_clearances)),
            "min_clearances": min_clearances,
            "first_violation": int(unsafe[0]) if unsafe.size else None,
            "unsafe_count": int(unsafe.size),
            "safety_eval_available": True,
        }

    def path_consistent_brake(self, obs, action_chunk, safety_info):
        chunk, _ = self._as_chunk(action_chunk)
        first_violation = safety_info.get("first_violation")
        if first_violation is None:
            q_seq = self.rollout_nominal_chunk(obs, chunk)
            brake_safety = self.evaluate_horizon_safety(obs, q_seq)
            return chunk, {
                "brake_safe": bool(brake_safety["horizon_safe"]),
                "deadlock": False,
                "progress_scale": 1.0,
                "brake_stop_idx": None,
                "brake_min_clearance": float(brake_safety["min_clearance"]),
            }

        stop_idx = max(0, int(first_violation) - 1)
        stop_idx = min(stop_idx, chunk.shape[0] - 1)
        braked = chunk.copy()
        braked[stop_idx:] = chunk[stop_idx]

        q_seq = self.rollout_nominal_chunk(obs, braked)
        brake_safety = self.evaluate_horizon_safety(obs, q_seq)
        progress_scale = stop_idx / max(1, chunk.shape[0] - 1)
        deadlock = progress_scale < self.brake_progress_threshold
        return braked, {
            "brake_safe": bool(brake_safety["horizon_safe"]),
            "deadlock": bool(deadlock),
            "progress_scale": float(progress_scale),
            "brake_stop_idx": int(stop_idx),
            "brake_min_clearance": float(brake_safety["min_clearance"]),
        }

    def deform_chunk_with_oscbf(self, obs, action_chunk, **kwargs):
        chunk, _ = self._as_chunk(action_chunk)
        safe_chunk = chunk.copy()
        op = self._get_oscbf_operator()
        if callable(op):
            for k, action in enumerate(chunk):
                safe_action = self._call_single_step_operator(action, obs, **kwargs)
                safe_action = np.asarray(safe_action, dtype=chunk.dtype).reshape(-1)
                if safe_action.shape[0] != chunk.shape[1]:
                    raise ValueError(
                        "Single-step safety operator returned shape "
                        f"{safe_action.shape}, expected ({chunk.shape[1]},)"
                    )
                safe_chunk[k, self.controlled_action_indices] = safe_action[
                    self.controlled_action_indices
                ]

        delta = (
            safe_chunk[:, self.controlled_action_indices]
            - chunk[:, self.controlled_action_indices]
        )
        deformation_norm = float(np.mean(np.linalg.norm(delta, axis=1)))
        q_seq = self.rollout_nominal_chunk(obs, safe_chunk)
        deform_safety = self.evaluate_horizon_safety(obs, q_seq)
        return safe_chunk, {
            "deform_safe": bool(deform_safety["horizon_safe"]),
            "deform_min_clearance": float(deform_safety["min_clearance"]),
            "deformation_norm": deformation_norm,
        }

    def filter_chunk(self, obs, action_chunk, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        original_shape = np.asarray(action_chunk).shape
        chunk, was_single = self._as_chunk(action_chunk)
        if not self.enabled:
            info = {"safety_mode": "disabled", "mode": "disabled"}
            self.last_info = info
            return chunk.reshape(original_shape), info
        if was_single:
            safe_action = self._call_single_step_operator(chunk[0], obs, **kwargs)
            info = {
                "safety_mode": "single_step_oscbf",
                "mode": "single_step_oscbf",
            }
            self.last_info = info
            return np.asarray(safe_action).reshape(original_shape), info

        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_info = self.evaluate_horizon_safety(obs, q_seq)
        info = dict(safety_info)

        if safety_info["horizon_safe"]:
            info.update({"safety_mode": "pass_through", "mode": "pass_through"})
            self.last_info = info
            return chunk.reshape(original_shape), info

        braked_chunk, brake_info = self.path_consistent_brake(obs, chunk, safety_info)
        info.update(brake_info)
        if brake_info["brake_safe"] and not brake_info["deadlock"]:
            info.update(
                {
                    "safety_mode": "path_consistent_brake",
                    "mode": "path_consistent_brake",
                }
            )
            self.last_info = info
            return braked_chunk.reshape(original_shape), info

        if not self.deformation_enabled:
            info.update({"safety_mode": "stop", "mode": "stop"})
            self.last_info = info
            return braked_chunk.reshape(original_shape), info

        safe_chunk, deform_info = self.deform_chunk_with_oscbf(
            obs, chunk, **kwargs
        )
        info.update(deform_info)
        info.update({"safety_mode": "horizon_deform", "mode": "horizon_deform"})
        self.last_info = info
        return safe_chunk.reshape(original_shape), info

    def __call__(self, *args, **kwargs):
        if len(args) < 2:
            action = kwargs.pop("action", None)
            obs = kwargs.pop("obs", kwargs.pop("observations", None))
            if action is None:
                raise TypeError("SafeChunkDeformFilter requires an action argument")
        else:
            first, second = args[0], args[1]
            first_arr = self._maybe_array(first)
            second_arr = self._maybe_array(second)
            if first_arr is not None and first_arr.ndim in (1, 2):
                action, obs = first, second
            elif second_arr is not None and second_arr.ndim in (1, 2):
                obs, action = first, second
            else:
                action, obs = first, second

        chunk, was_single = self._as_chunk(action)
        if not self.enabled:
            self.last_info = {"safety_mode": "disabled", "mode": "disabled"}
            return chunk.reshape(np.asarray(action).shape)
        if was_single:
            safe_action = self._call_single_step_operator(chunk[0], obs, **kwargs)
            self.last_info = {
                "safety_mode": "single_step_oscbf",
                "mode": "single_step_oscbf",
            }
            return np.asarray(safe_action).reshape(np.asarray(action).shape)

        safe_chunk, info = self.filter_chunk(obs, chunk, **kwargs)
        self.last_info = info
        return safe_chunk.reshape(np.asarray(action).shape)

    def _maybe_array(self, value):
        try:
            return np.asarray(value)
        except Exception:
            return None

    def _call_single_step_operator(self, action, obs=None, **kwargs):
        op = self._get_oscbf_operator()
        if not callable(op):
            return np.asarray(action).copy()
        attempts = (
            lambda: op(action=action, observations=obs, **kwargs),
            lambda: op(action=action, obs=obs, **kwargs),
            lambda: op(action, obs, **kwargs),
            lambda: op(action, **kwargs),
            lambda: op(action),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        raise last_error

    def _get_oscbf_operator(self):
        if callable(self.oscbf_operator) or self.oscbf_operator is None:
            return self.oscbf_operator
        if self._operator_instantiation_failed:
            return None
        target = None
        try:
            target = self.oscbf_operator.get("_target_")
        except AttributeError:
            if isinstance(self.oscbf_operator, dict):
                target = self.oscbf_operator.get("_target_")
        if target is None:
            return self.oscbf_operator
        try:
            import hydra

            self.oscbf_operator = hydra.utils.instantiate(self.oscbf_operator)
        except Exception as exc:  # pragma: no cover - depends on deployment config
            logger.warning(
                "SafeChunk-Deform could not instantiate oscbf_operator; "
                "falling back to identity single-step deformation: %s",
                exc,
            )
            self._operator_instantiation_failed = True
            return None
        return self.oscbf_operator
