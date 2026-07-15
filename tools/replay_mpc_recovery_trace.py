#!/usr/bin/env python3
"""Replay SafeChunk MPC recovery traces without running ACT, MuJoCo, or OSCBF.

This script consumes ``recovery_plan_traces.jsonl`` records.  Each record already
contains the planned recovery actions and the q-state sequence that the online
planner produced.  The offline runner checks two things quickly:

1. The recorded action-to-q conversion is internally consistent.
2. A receding-horizon kinematic MPC repair can converge back onto the recorded
   q path from each recorded start state, optionally with injected perturbations.

It intentionally does not recompute live collision/OSCBF clearances; recorded
clearance fields are carried through as diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

JsonDict = dict[str, Any]

DEFAULT_CONTROLLED = "0,1,2,3,4,5,6,7,9,10,11,12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline replay of MPC recovery convergence on prerecorded traces."
    )
    parser.add_argument("--input", nargs="+", required=True, help="recovery_plan_traces JSONL/JSON files")
    parser.add_argument("--output", default=None, help="Optional JSON summary path")
    parser.add_argument("--output-jsonl", default=None, help="Optional per-case JSONL path")
    parser.add_argument("--limit-records", type=int, default=0, help="Limit loaded records; 0 means all")
    parser.add_argument("--horizon", type=int, default=8, help="Offline MPC planning horizon")
    parser.add_argument("--max-attempts", type=int, default=0, help="Attempts per start; 0 means 4x remaining path")
    parser.add_argument("--start-index", default="all", help="all, or comma list of local start indices")
    parser.add_argument("--target-tolerance", type=float, default=0.08, help="Per-waypoint controlled q tolerance")
    parser.add_argument("--terminal-tolerance", type=float, default=0.12, help="Final controlled q convergence tolerance")
    parser.add_argument("--min-progress", type=float, default=1e-6, help="Diagnostic progress threshold")
    parser.add_argument("--no-progress-limit", type=int, default=0, help="Abort after N no-progress attempts; 0 disables")
    parser.add_argument("--perturb-scale", type=float, default=0.0, help="Stddev perturbation applied to controlled q at each start")
    parser.add_argument("--max-state-step", type=float, default=0.0, help="Clamp controlled q movement per offline step; 0 disables")
    parser.add_argument("--nearest-target", action="store_true", default=True, help="Allow snapping to nearest future target")
    parser.add_argument("--sequential-targets", dest="nearest_target", action="store_false", help="Track targets strictly in order")
    parser.add_argument("--dt", type=float, default=0.05, help="Control dt for velocity-mode dimensions")
    parser.add_argument("--control-type", default="absolute", choices=["absolute", "delta", "velocity"])
    parser.add_argument("--expected-motion-dim", type=int, default=14)
    parser.add_argument("--controlled-action-indices", default=DEFAULT_CONTROLLED)
    parser.add_argument("--controlled-state-indices", default=DEFAULT_CONTROLLED)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_indices(spec: str) -> np.ndarray:
    values = [int(tok.strip()) for tok in str(spec).split(",") if tok.strip()]
    return np.asarray(values, dtype=np.int64)


def load_records(paths: Sequence[str], limit: int = 0) -> list[JsonDict]:
    records: list[JsonDict] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
                    if limit > 0 and len(records) >= limit:
                        return records
        else:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("records"), list):
                loaded = loaded["records"]
            if isinstance(loaded, list):
                for record in loaded:
                    if isinstance(record, Mapping):
                        records.append(dict(record))
                        if limit > 0 and len(records) >= limit:
                            return records
            elif isinstance(loaded, Mapping):
                records.append(dict(loaded))
            else:
                raise ValueError(f"Unsupported JSON root in {path}: {type(loaded)!r}")
    return records


def finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    return value_f if math.isfinite(value_f) else default


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def control_mode_ids(state_idx: np.ndarray, control_type: str, expected_motion_dim: int) -> np.ndarray:
    if control_type == "absolute":
        mode = 0
    elif control_type == "delta":
        mode = 1
    else:
        mode = 2
    modes = np.full(state_idx.shape, mode, dtype=np.int32)
    modes[state_idx < min(4, int(expected_motion_dim))] = 1
    return modes


def valid_control_indices(
    action_dim: int,
    q_dim: int,
    action_idx: np.ndarray,
    state_idx: np.ndarray,
) -> np.ndarray:
    return (action_idx < int(action_dim)) & (state_idx < int(q_dim))


def apply_action(
    q: np.ndarray,
    action: np.ndarray,
    action_idx: np.ndarray,
    state_idx: np.ndarray,
    *,
    control_type: str,
    expected_motion_dim: int,
    dt: float,
) -> np.ndarray:
    q_next = np.asarray(q, dtype=np.float32).reshape(-1).copy()
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    valid = valid_control_indices(action.shape[0], q_next.shape[0], action_idx, state_idx)
    if not np.any(valid):
        return q_next
    ai = action_idx[valid]
    si = state_idx[valid]
    modes = control_mode_ids(si, control_type, expected_motion_dim)
    selected = action[ai].astype(np.float32, copy=True)
    current = q_next[si]
    updated = selected.copy()
    delta_mask = modes == 1
    velocity_mask = modes == 2
    updated[delta_mask] = current[delta_mask] + selected[delta_mask]
    updated[velocity_mask] = current[velocity_mask] + float(dt) * selected[velocity_mask]
    q_next[si] = updated
    return q_next


def action_from_transition(
    reference_action: np.ndarray,
    q_prev: np.ndarray,
    q_next: np.ndarray,
    action_idx: np.ndarray,
    state_idx: np.ndarray,
    *,
    control_type: str,
    expected_motion_dim: int,
    dt: float,
) -> np.ndarray:
    safe = np.asarray(reference_action, dtype=np.float32).reshape(-1).copy()
    q_prev = np.asarray(q_prev, dtype=np.float32).reshape(-1)
    q_next = np.asarray(q_next, dtype=np.float32).reshape(-1)
    valid = valid_control_indices(safe.shape[0], min(q_prev.shape[0], q_next.shape[0]), action_idx, state_idx)
    if not np.any(valid):
        return safe
    ai = action_idx[valid]
    si = state_idx[valid]
    modes = control_mode_ids(si, control_type, expected_motion_dim)
    values = q_next[si].astype(np.float32, copy=True)
    delta = q_next[si] - q_prev[si]
    values[modes == 1] = delta[modes == 1]
    values[modes == 2] = delta[modes == 2] / float(dt)
    safe[ai] = values
    return safe


def controlled_dist(a: np.ndarray, b: np.ndarray, state_idx: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    valid = state_idx[(state_idx < a.shape[0]) & (state_idx < b.shape[0])]
    if valid.size == 0:
        return float("inf")
    diff = a[valid] - b[valid]
    return float(np.linalg.norm(diff) / math.sqrt(max(1, diff.size)))


def extract_plan(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    planned = record.get("planned")
    if not isinstance(planned, Mapping):
        return None
    actions = np.asarray(planned.get("actions"), dtype=np.float32)
    q_seq = np.asarray(planned.get("q_seq"), dtype=np.float32)
    post_q_seq = np.asarray(planned.get("post_q_seq", planned.get("q_seq")), dtype=np.float32)
    if actions.ndim != 2 or q_seq.ndim != 2 or post_q_seq.ndim != 2:
        return None
    n = min(actions.shape[0], q_seq.shape[0], post_q_seq.shape[0])
    if n <= 0:
        return None
    return actions[:n], q_seq[:n], post_q_seq[:n]


def start_indices_for(length: int, spec: str) -> list[int]:
    if length <= 0:
        return []
    if str(spec).strip().lower() == "all":
        return list(range(length))
    result: list[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        idx = int(tok)
        if idx < 0:
            idx = length + idx
        if 0 <= idx < length:
            result.append(idx)
    return sorted(set(result))


def action_model_error(
    actions: np.ndarray,
    q_seq: np.ndarray,
    post_q_seq: np.ndarray,
    args: argparse.Namespace,
    action_idx: np.ndarray,
    state_idx: np.ndarray,
) -> dict[str, float]:
    errors = []
    for action, q, target in zip(actions, q_seq, post_q_seq):
        predicted = apply_action(
            q,
            action,
            action_idx,
            state_idx,
            control_type=args.control_type,
            expected_motion_dim=args.expected_motion_dim,
            dt=args.dt,
        )
        errors.append(controlled_dist(predicted, target, state_idx))
    if not errors:
        return {"mean": float("inf"), "max": float("inf")}
    return {"mean": float(np.mean(errors)), "max": float(np.max(errors))}


def clamp_toward(q_current: np.ndarray, q_target: np.ndarray, state_idx: np.ndarray, max_state_step: float) -> np.ndarray:
    if max_state_step <= 0.0:
        return np.asarray(q_target, dtype=np.float32).copy()
    q_current = np.asarray(q_current, dtype=np.float32).reshape(-1)
    desired = q_current.copy()
    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)
    valid = state_idx[(state_idx < q_current.shape[0]) & (state_idx < q_target.shape[0])]
    if valid.size:
        delta = q_target[valid] - q_current[valid]
        desired[valid] = q_current[valid] + np.clip(delta, -max_state_step, max_state_step)
    return desired


def plan_actions_to_targets(
    current_q: np.ndarray,
    targets: np.ndarray,
    references: np.ndarray,
    args: argparse.Namespace,
    action_idx: np.ndarray,
    state_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_roll = np.asarray(current_q, dtype=np.float32).reshape(-1).copy()
    planned_actions = []
    planned_q = []
    horizon = min(int(args.horizon), targets.shape[0])
    for k in range(horizon):
        desired_q = clamp_toward(q_roll, targets[k], state_idx, float(args.max_state_step))
        ref = references[min(k, references.shape[0] - 1)]
        action = action_from_transition(
            ref,
            q_roll,
            desired_q,
            action_idx,
            state_idx,
            control_type=args.control_type,
            expected_motion_dim=args.expected_motion_dim,
            dt=args.dt,
        )
        q_roll = apply_action(
            q_roll,
            action,
            action_idx,
            state_idx,
            control_type=args.control_type,
            expected_motion_dim=args.expected_motion_dim,
            dt=args.dt,
        )
        planned_actions.append(action)
        planned_q.append(q_roll.copy())
    return np.asarray(planned_actions, dtype=np.float32), np.asarray(planned_q, dtype=np.float32)


def replay_start(
    record: Mapping[str, Any],
    record_index: int,
    start_idx: int,
    actions: np.ndarray,
    q_seq: np.ndarray,
    post_q_seq: np.ndarray,
    args: argparse.Namespace,
    action_idx: np.ndarray,
    state_idx: np.ndarray,
) -> JsonDict:
    rng = np.random.default_rng(int(args.seed) + record_index * 1009 + start_idx)
    current_q = q_seq[start_idx].astype(np.float32, copy=True)
    if float(args.perturb_scale) > 0.0:
        valid = state_idx[state_idx < current_q.shape[0]]
        current_q[valid] += rng.normal(0.0, float(args.perturb_scale), size=valid.shape[0]).astype(np.float32)

    targets = post_q_seq[start_idx:]
    refs = actions[start_idx:]
    if targets.shape[0] == 0:
        return {"success": False, "reject_reason": "empty_target_path"}

    max_attempts = int(args.max_attempts)
    if max_attempts <= 0:
        max_attempts = max(1, int(targets.shape[0]) * 4)

    cursor = 0
    no_progress_count = 0
    best_terminal = controlled_dist(current_q, targets[-1], state_idx)
    best_target = controlled_dist(current_q, targets[0], state_idx)
    trace = []
    success = False
    reject_reason = "max_attempts"

    for attempt in range(1, max_attempts + 1):
        if bool(args.nearest_target):
            future = targets[cursor:]
            if future.shape[0] > 0:
                dists = [controlled_dist(current_q, q, state_idx) for q in future]
                nearest = int(np.argmin(dists))
                cursor += nearest
        cursor = min(cursor, targets.shape[0] - 1)
        target_before = controlled_dist(current_q, targets[cursor], state_idx)
        terminal_before = controlled_dist(current_q, targets[-1], state_idx)
        plan_targets = targets[cursor : cursor + int(args.horizon)]
        plan_refs = refs[cursor : cursor + int(args.horizon)]
        if plan_targets.shape[0] == 0 or plan_refs.shape[0] == 0:
            reject_reason = "empty_plan_window"
            break
        planned_actions, planned_q = plan_actions_to_targets(
            current_q,
            plan_targets,
            plan_refs,
            args,
            action_idx,
            state_idx,
        )
        if planned_actions.shape[0] == 0:
            reject_reason = "empty_mpc_plan"
            break
        current_q = planned_q[0]
        target_after = controlled_dist(current_q, targets[cursor], state_idx)
        terminal_after = controlled_dist(current_q, targets[-1], state_idx)
        progress = target_before - target_after
        best_terminal = min(best_terminal, terminal_after)
        best_target = min(best_target, target_after)
        if progress <= float(args.min_progress):
            no_progress_count += 1
        else:
            no_progress_count = 0
        while cursor < targets.shape[0] and controlled_dist(current_q, targets[cursor], state_idx) <= float(args.target_tolerance):
            cursor += 1
        trace.append(
            {
                "attempt": attempt,
                "cursor": int(cursor),
                "target_dist_before": float(target_before),
                "target_dist_after": float(target_after),
                "terminal_dist_before": float(terminal_before),
                "terminal_dist_after": float(terminal_after),
                "progress": float(progress),
            }
        )
        if terminal_after <= float(args.terminal_tolerance) or cursor >= targets.shape[0]:
            success = True
            reject_reason = None
            break
        if int(args.no_progress_limit) > 0 and no_progress_count >= int(args.no_progress_limit):
            reject_reason = "no_progress"
            break

    return {
        "episode": record.get("episode"),
        "step": record.get("step"),
        "record_index": int(record_index),
        "start_index": int(start_idx),
        "path_length": int(targets.shape[0]),
        "success": bool(success),
        "reject_reason": reject_reason,
        "attempts": int(trace[-1]["attempt"] if trace else 0),
        "final_cursor": int(cursor),
        "final_terminal_dist": float(controlled_dist(current_q, targets[-1], state_idx)),
        "best_terminal_dist": float(best_terminal),
        "best_target_dist": float(best_target),
        "planned_min_clearance": finite_float(record.get("planned_min_clearance")),
        "mpc_recovery_replan_reason": record.get("mpc_recovery_replan_reason"),
        "mpc_recovery_replan_accepted": record.get("mpc_recovery_replan_accepted"),
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    action_idx = parse_indices(args.controlled_action_indices)
    state_idx = parse_indices(args.controlled_state_indices)
    records = load_records(args.input, limit=int(args.limit_records))
    results: list[JsonDict] = []
    model_errors: list[float] = []
    skipped = 0

    for record_index, record in enumerate(records):
        plan = extract_plan(record)
        if plan is None:
            skipped += 1
            continue
        actions, q_seq, post_q_seq = plan
        err = action_model_error(actions, q_seq, post_q_seq, args, action_idx, state_idx)
        model_errors.append(float(err["max"]))
        starts = start_indices_for(actions.shape[0], args.start_index)
        for start_idx in starts:
            item = replay_start(
                record,
                record_index,
                start_idx,
                actions,
                q_seq,
                post_q_seq,
                args,
                action_idx,
                state_idx,
            )
            item["action_model_error_mean"] = float(err["mean"])
            item["action_model_error_max"] = float(err["max"])
            results.append(item)

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    summary: JsonDict = {
        "stage": "offline_mpc_recovery_trace_replay",
        "records_loaded": int(len(records)),
        "records_skipped": int(skipped),
        "cases": int(len(results)),
        "successes": int(len(successes)),
        "failures": int(len(failures)),
        "success_rate": float(len(successes) / len(results)) if results else 0.0,
        "mean_attempts_success": float(np.mean([r["attempts"] for r in successes])) if successes else None,
        "mean_best_terminal_dist": float(np.mean([r["best_terminal_dist"] for r in results])) if results else None,
        "max_action_model_error": float(np.max(model_errors)) if model_errors else None,
        "failure_reasons": {},
        "settings": {
            "horizon": int(args.horizon),
            "max_attempts": int(args.max_attempts),
            "target_tolerance": float(args.target_tolerance),
            "terminal_tolerance": float(args.terminal_tolerance),
            "perturb_scale": float(args.perturb_scale),
            "max_state_step": float(args.max_state_step),
            "nearest_target": bool(args.nearest_target),
        },
    }
    reasons: dict[str, int] = {}
    for item in failures:
        reason = str(item.get("reject_reason") or "failed")
        reasons[reason] = reasons.get(reason, 0) + 1
    summary["failure_reasons"] = reasons
    summary["worst_cases"] = sorted(
        [
            {
                "episode": r.get("episode"),
                "step": r.get("step"),
                "record_index": r.get("record_index"),
                "start_index": r.get("start_index"),
                "success": r.get("success"),
                "reject_reason": r.get("reject_reason"),
                "attempts": r.get("attempts"),
                "best_terminal_dist": r.get("best_terminal_dist"),
                "final_terminal_dist": r.get("final_terminal_dist"),
                "planned_min_clearance": r.get("planned_min_clearance"),
                "action_model_error_max": r.get("action_model_error_max"),
            }
            for r in results
        ],
        key=lambda x: float(x.get("best_terminal_dist") or 0.0),
        reverse=True,
    )[:10]

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(jsonable(item), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
