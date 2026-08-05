#!/usr/bin/env python3
"""Fit fixed SafeChunk rollout calibration from saved Bigym diagnostics.

This is an offline tool: it reads diagnostic JSONL files and estimates a
simple shared action->q model. Runtime SafeChunk does not need Bigym feedback
when using the calibrated constants from the method config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ACTION_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
DEFAULT_STATE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
STATE_DIM = 14


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _samples_from_nominal_record(record: dict[str, Any]):
    q0 = _as_array(record.get("q_before"))
    q1 = _as_array(record.get("q_after"))
    action = _as_array(record.get("safe_first_action") or record.get("nominal_first_action"))
    if q0 is None or q1 is None or action is None:
        return None
    return q0, q1, action


def _samples_from_mpc_record(record: dict[str, Any]):
    q0 = _as_array(record.get("bigym_actual_pre_action_q"))
    q1 = _as_array(record.get("bigym_actual_post_action_q"))
    action = _as_array(record.get("executed_first_action") or record.get("committed_action"))
    if q0 is None or q1 is None or action is None:
        return None
    return q0, q1, action


def load_samples(paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    samples = []
    for path in paths:
        for record in _load_jsonl(path):
            sample = _samples_from_nominal_record(record)
            if sample is None:
                sample = _samples_from_mpc_record(record)
            if sample is not None:
                samples.append(sample)
    return samples


def _fit_slope_bias(x: np.ndarray, y: np.ndarray, min_abs_x: float) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y) & (np.abs(x) >= min_abs_x)
    if int(np.count_nonzero(valid)) < 3:
        return 1.0, 0.0, int(np.count_nonzero(valid))
    a = np.stack([x[valid], np.ones(int(np.count_nonzero(valid)), dtype=np.float32)], axis=1)
    slope, bias = np.linalg.lstsq(a, y[valid], rcond=None)[0]
    return float(slope), float(bias), int(np.count_nonzero(valid))


def predict(
    q0: np.ndarray,
    action: np.ndarray,
    state_indices: list[int],
    action_indices: list[int],
    delta_scale: np.ndarray,
    target_alpha: np.ndarray,
    action_bias: np.ndarray,
) -> np.ndarray:
    pred = q0[:STATE_DIM].astype(np.float32, copy=True)
    for state, act in zip(state_indices, action_indices):
        if state >= pred.size or act >= action.size:
            continue
        command = float(action[act]) + float(action_bias[state])
        if state < 4:
            pred[state] = pred[state] + float(delta_scale[state]) * command
        else:
            pred[state] = pred[state] + float(target_alpha[state]) * (command - pred[state])
    return pred


def error_summary(samples, state_indices, action_indices, delta_scale, target_alpha, action_bias):
    errs = []
    for q0, q1, action in samples:
        if q0.size < STATE_DIM or q1.size < STATE_DIM:
            continue
        pred = predict(q0, action, state_indices, action_indices, delta_scale, target_alpha, action_bias)
        errs.append(q1[:STATE_DIM] - pred)
    if not errs:
        return {"count": 0}
    arr = np.stack(errs, axis=0)
    return {
        "count": int(arr.shape[0]),
        "mean_l2": float(np.mean(np.linalg.norm(arr, axis=1))),
        "max_l2": float(np.max(np.linalg.norm(arr, axis=1))),
        "mean_max_abs": float(np.mean(np.max(np.abs(arr), axis=1))),
        "mean_base_l2": float(np.mean(np.linalg.norm(arr[:, :4], axis=1))),
        "mean_arm_l2": float(np.mean(np.linalg.norm(arr[:, 4:], axis=1))),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="diagnostic JSONL files")
    parser.add_argument("--output", required=True, help="summary JSON path")
    parser.add_argument("--min-abs-x", type=float, default=1e-4)
    parser.add_argument("--state-indices", nargs="*", type=int, default=DEFAULT_STATE_INDICES)
    parser.add_argument("--action-indices", nargs="*", type=int, default=DEFAULT_ACTION_INDICES)
    args = parser.parse_args()

    if len(args.state_indices) != len(args.action_indices):
        raise ValueError("state/action index lists must have the same length")
    samples = load_samples([Path(p) for p in args.input])
    delta_scale = np.ones((STATE_DIM,), dtype=np.float32)
    target_alpha = np.ones((STATE_DIM,), dtype=np.float32)
    action_bias = np.zeros((STATE_DIM,), dtype=np.float32)
    fit_counts: dict[str, int] = {}

    for state, act in zip(args.state_indices, args.action_indices):
        x_vals = []
        y_vals = []
        for q0, q1, action in samples:
            if state >= min(q0.size, q1.size, STATE_DIM) or act >= action.size:
                continue
            y = float(q1[state] - q0[state])
            x = float(action[act]) if state < 4 else float(action[act] - q0[state])
            x_vals.append(x)
            y_vals.append(y)
        if not x_vals:
            continue
        slope, bias, count = _fit_slope_bias(
            np.asarray(x_vals, dtype=np.float32),
            np.asarray(y_vals, dtype=np.float32),
            float(args.min_abs_x),
        )
        fit_counts[str(state)] = count
        if state < 4:
            delta_scale[state] = slope
            action_bias[state] = bias / slope if abs(slope) > 1e-8 else 0.0
        else:
            target_alpha[state] = max(0.0, slope)
            action_bias[state] = bias / slope if abs(slope) > 1e-8 else 0.0

    default_summary = error_summary(
        samples,
        args.state_indices,
        args.action_indices,
        np.ones((STATE_DIM,), dtype=np.float32),
        np.ones((STATE_DIM,), dtype=np.float32),
        np.zeros((STATE_DIM,), dtype=np.float32),
    )
    calibrated_summary = error_summary(
        samples,
        args.state_indices,
        args.action_indices,
        delta_scale,
        target_alpha,
        action_bias,
    )
    summary = {
        "sample_count": len(samples),
        "state_indices": args.state_indices,
        "action_indices": args.action_indices,
        "fit_counts_by_state": fit_counts,
        "default_error": default_summary,
        "calibrated_error": calibrated_summary,
        "per_state_delta_scale": delta_scale.astype(float).tolist(),
        "per_state_target_alpha": target_alpha.astype(float).tolist(),
        "per_state_action_bias": action_bias.astype(float).tolist(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
