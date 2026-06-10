"""Path-consistent braking filter.

This restores the experimental path-consistent braking implementation under
the intended public class name, ``PathConsistentBrakeFilter``. The implementation
body is loaded from the compiled cache that was produced before the accidental
source removal in this workspace.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_compiled_module():
    cache_dir = Path(__file__).with_name("__pycache__")
    version_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    candidates = [
        cache_dir / f"horizon_brake_filter.{version_tag}.pyc",
        cache_dir / "horizon_brake_filter.cpython-310.pyc",
        cache_dir / "horizon_brake_filter.cpython-313.pyc",
    ]
    for candidate in candidates:
        if candidate.exists():
            loader = importlib.machinery.SourcelessFileLoader(
                __name__ + "._compiled_path_consistent_brake",
                str(candidate),
            )
            spec = importlib.util.spec_from_loader(loader.name, loader)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            return module
    searched = ", ".join(str(path) for path in candidates)
    raise ImportError(
        "PathConsistentBrakeFilter compiled implementation is missing. "
        f"Searched: {searched}"
    )


_compiled = _load_compiled_module()
PathRetimingResult = _compiled.PathRetimingResult


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


class PathConsistentBrakeFilter(_compiled.HorizonBrakeFilter):
    """Experimental path-consistent braking filter.

    The filter keeps the barrier-function interface used in this repository while
    adding waypoint insertion, trajectory retiming, failsafe verification,
    optional reachability inflation/certification, and inner shield checks.
    """

    def _action_from_q_transition(self, action, q_prev, q_next, action_idx, state_idx):
        safe = np.asarray(action, dtype=np.float32).copy()
        q_prev = np.asarray(q_prev, dtype=np.float32).reshape(-1)
        q_next = np.asarray(q_next, dtype=np.float32).reshape(-1)
        action_idx = np.asarray(action_idx, dtype=np.int64).reshape(-1)
        state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
        valid = (action_idx < safe.shape[0]) & (state_idx < q_prev.shape[0]) & (state_idx < q_next.shape[0])
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
        valid = (self.controlled_action_indices < chunk.shape[1]) & (self.controlled_state_indices < q.shape[0])
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

    def make_path_consistent_failsafe_tail(self, *args, **kwargs):
        return self.make_horizon_brake_failsafe_tail(*args, **kwargs)

    def path_consistent_slowdown_or_brake(self, *args, **kwargs):
        chunk, info = self.horizon_brake_slowdown_or_brake(*args, **kwargs)
        return chunk, _rename_info_keys(dict(info))


__all__ = ["PathConsistentBrakeFilter", "PathRetimingResult"]
