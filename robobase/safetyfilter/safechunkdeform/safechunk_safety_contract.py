from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SafetyTrace:
    """Canonical safety signal for a predicted trajectory.

    clearance_seq is signed linear clearance in meters. Raw h is kept only as
    optional debug/barrier data and must not be treated as a distance margin.
    """

    clearance_seq: np.ndarray
    min_clearance: float
    immediate_clearance: float
    raw_h_seq: np.ndarray | None = None
    min_raw_h: float | None = None
    argmin_pair: str | None = None


@dataclass(frozen=True)
class SafetyConstraintResult:
    """Shared clearance-margin constraint result for deform/recovery/MPC."""

    required_clearance: float
    min_clearance: float
    immediate_clearance: float
    prefix_min_clearance: float
    safe_prefix_len: int
    immediate_safe: bool
    prefix_safe: bool
    path_safe: bool
    margin_loss: float
    reject_reason: str | None


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def clearance_sequence_from_eval(
    safety_eval: Mapping[str, Any] | None,
    horizon: int,
    *,
    default: float = float("-inf"),
) -> np.ndarray:
    """Extract signed clearance sequence in meters from a SafetyEval-like dict."""
    horizon = max(0, int(horizon))
    if horizon == 0:
        return np.zeros((0,), dtype=np.float32)
    if not isinstance(safety_eval, Mapping):
        try:
            arr = np.asarray(safety_eval, dtype=np.float32).reshape(-1)
            if arr.size == horizon:
                return arr.astype(np.float32, copy=True)
            if arr.size == 1:
                return np.full((horizon,), float(arr[0]), dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass
        return np.full((horizon,), float(default), dtype=np.float32)

    clearances = safety_eval.get("min_clearances", safety_eval.get("clearances"))
    if clearances is not None:
        try:
            arr = np.asarray(clearances, dtype=np.float32).reshape(-1)
            if arr.size == horizon:
                return arr.astype(np.float32, copy=True)
            if arr.size == 1:
                return np.full((horizon,), float(arr[0]), dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass

    min_clearance = _finite_float(safety_eval.get("min_clearance"), default)
    return np.full((horizon,), min_clearance, dtype=np.float32)


def raw_h_sequence_from_eval(
    safety_eval: Mapping[str, Any] | None,
    horizon: int,
) -> np.ndarray | None:
    """Extract raw h sequence for diagnostics when available."""
    if not isinstance(safety_eval, Mapping):
        return None
    raw = safety_eval.get("min_h_values")
    if raw is None:
        raw = safety_eval.get("h_values")
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim >= 2:
            arr = np.min(arr.reshape(arr.shape[0], -1), axis=1)
        arr = arr.reshape(-1)
        if arr.size == int(horizon):
            return arr.astype(np.float32, copy=True)
        if arr.size == 1 and int(horizon) > 1:
            return np.full((int(horizon),), float(arr[0]), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    return None


def safety_trace_from_eval(
    safety_eval: Mapping[str, Any] | None,
    horizon: int,
) -> SafetyTrace:
    """Build the canonical safety trace from a horizon safety evaluation."""
    clearance_seq = clearance_sequence_from_eval(safety_eval, horizon)
    if clearance_seq.size:
        min_clearance = float(np.min(clearance_seq))
        immediate_clearance = float(clearance_seq[0])
    else:
        min_clearance = float("inf")
        immediate_clearance = float("inf")

    raw_h_seq = raw_h_sequence_from_eval(safety_eval, horizon)
    min_raw_h = float(np.min(raw_h_seq)) if raw_h_seq is not None and raw_h_seq.size else None
    argmin_pair = None
    if isinstance(safety_eval, Mapping):
        argmin_pair = safety_eval.get("h_argmin_pair_label") or safety_eval.get("argmin_pair")
    return SafetyTrace(
        clearance_seq=clearance_seq,
        min_clearance=min_clearance,
        immediate_clearance=immediate_clearance,
        raw_h_seq=raw_h_seq,
        min_raw_h=min_raw_h,
        argmin_pair=None if argmin_pair is None else str(argmin_pair),
    )


def clearance_margin_loss(clearance_seq: Any, required_clearance: float) -> float:
    """Squared hinge loss for clearance-margin violations."""
    seq = np.asarray(clearance_seq, dtype=np.float32).reshape(-1)
    if seq.size == 0:
        return 0.0
    deficit = np.maximum(float(required_clearance) - seq, 0.0)
    if not np.all(np.isfinite(deficit)):
        return float("inf")
    return float(np.square(deficit).sum())


def evaluate_clearance_constraint(
    trace: SafetyTrace,
    required_clearance: float,
    *,
    prefix_len: int = 1,
    require_full_path: bool = True,
) -> SafetyConstraintResult:
    """Evaluate immediate, prefix, and full-path clearance constraints."""
    seq = np.asarray(trace.clearance_seq, dtype=np.float32).reshape(-1)
    required = float(required_clearance)
    if seq.size == 0:
        return SafetyConstraintResult(
            required_clearance=required,
            min_clearance=float("inf"),
            immediate_clearance=float("inf"),
            prefix_min_clearance=float("inf"),
            safe_prefix_len=0,
            immediate_safe=True,
            prefix_safe=True,
            path_safe=True,
            margin_loss=0.0,
            reject_reason=None,
        )

    prefix_n = max(1, min(int(prefix_len), int(seq.size)))
    prefix_seq = seq[:prefix_n]
    safe_prefix_len = 0
    for value in seq:
        if float(value) >= required:
            safe_prefix_len += 1
        else:
            break

    immediate = float(seq[0])
    min_clearance = float(np.min(seq))
    prefix_min = float(np.min(prefix_seq)) if prefix_seq.size else immediate
    immediate_safe = bool(immediate >= required)
    prefix_safe = bool(prefix_min >= required)
    full_path_safe = bool(min_clearance >= required)

    reject_reason = None
    if not immediate_safe:
        reject_reason = "immediate_unsafe"
    elif not prefix_safe:
        reject_reason = "prefix_unsafe"
    elif require_full_path and not full_path_safe:
        reject_reason = "path_unsafe"

    return SafetyConstraintResult(
        required_clearance=required,
        min_clearance=min_clearance,
        immediate_clearance=immediate,
        prefix_min_clearance=prefix_min,
        safe_prefix_len=int(safe_prefix_len),
        immediate_safe=immediate_safe,
        prefix_safe=prefix_safe,
        path_safe=full_path_safe,
        margin_loss=clearance_margin_loss(seq, required),
        reject_reason=reject_reason,
    )
