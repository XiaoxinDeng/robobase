#!/usr/bin/env python3
"""Tune SafeChunk explicit MPC replan settings from recorded recovery-plan traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

JsonDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-tune SafeChunk explicit MPC replan parameters."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more recovery plan trace JSONL files.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON summary output path.")
    parser.add_argument(
        "--write-yaml",
        default=None,
        help="Optional YAML override snippet output path.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of top parameter sets to print/save.")
    parser.add_argument("--horizon", default="6,8,10,12", help="MPC recovery horizon grid.")
    parser.add_argument("--prefix-len", default="0,1,2,3", help="Recovery prefix length grid.")
    parser.add_argument(
        "--max-replans",
        default="0,1,2,3,4",
        help="Max re-plans per recovery grid (0 means unlimited).",
    )
    parser.add_argument(
        "--require-ordered-progress",
        default="0,1",
        help="Grid for explicit_recovery.mpc_recovery_require_ordered_progress (0/1).",
    )
    parser.add_argument(
        "--require-live-progress",
        default="0,1",
        help="Grid for explicit_recovery.mpc_recovery_require_live_progress (0/1).",
    )
    parser.add_argument(
        "--min-progress-delta",
        default="0,0.0001,0.0005,0.001",
        help="Grid for explicit_recovery.mpc_recovery_min_progress_delta.",
    )
    parser.add_argument(
        "--no-progress-limit",
        default="1,2,3,4,6",
        help="Grid for explicit_recovery.mpc_recovery_no_progress_limit.",
    )
    parser.add_argument(
        "--strict-budget-match",
        action="store_true",
        help="Count budget-exceeded attempts as hard failures unless replan budget is strictly larger.",
    )
    return parser.parse_args()


def parse_grid(spec: str) -> list[float]:
    values: list[float] = []
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            pieces = [float(p) for p in token.split(":")]
            if len(pieces) != 3:
                raise ValueError(f"range grid must be start:stop:step, got {token!r}")
            start, stop, step = pieces
            if step == 0:
                raise ValueError(f"range step cannot be zero: {token!r}")
            current = start
            eps = abs(step) * 1e-6
            if step > 0:
                while current <= stop + eps:
                    values.append(float(current))
                    current += step
            else:
                while current >= stop - eps:
                    values.append(float(current))
                    current += step
        else:
            values.append(float(token))
    return sorted(set(values))


def parse_bool_grid(spec: str) -> list[bool]:
    values: list[bool] = []
    for token in str(spec).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"1", "true", "t", "yes", "y", "on"}:
            values.append(True)
        elif token in {"0", "false", "f", "no", "n", "off"}:
            values.append(False)
        else:
            raise ValueError(f"Unknown bool token for grid: {token!r}")
    return sorted(set(values))


def finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value_f):
        return default
    return value_f


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return default


def load_records(inputs: Sequence[str]) -> list[JsonDict]:
    records: list[JsonDict] = []
    for raw in inputs:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records.extend(loaded)
            elif isinstance(loaded, dict) and "records" in loaded:
                records.extend(loaded["records"])
            else:
                raise ValueError(f"Unsupported JSON shape for tuning input: {path}")
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}")
    return records


def simulate_mpc_acceptance(record: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[bool, str]:
    if not safe_bool(record.get("mpc_recovery_replan_attempted")):
        return False, "not_attempted"
    if safe_bool(record.get("mpc_recovery_replan_accepted")):
        return True, "accepted"

    reject_reason = str(record.get("mpc_recovery_replan_reject_reason") or "")
    suffix_reason = str(record.get("committed_suffix_replan_reject_reason") or "")
    reason = reject_reason.lower()
    suffix_reason_lower = suffix_reason.lower()

    if reason == "not_recover_mode":
        return False, "not_recover_mode"
    if reason == "prefix_replay_window":
        local_idx = record.get("committed_recover_local_index")
        local_idx = int(local_idx) if local_idx is not None else int(record.get("mpc_recovery_recover_local_index") or 0)
        return params["mpc_recovery_prefix_len"] <= int(local_idx), "prefix_replay_window"
    if reason == "mpc_replan_budget_exceeded":
        max_replans = int(params["mpc_recovery_max_replans_per_recovery"])
        if max_replans <= 0:
            return True, "unbounded_budget"
        tried = int(record.get("mpc_recovery_replans_in_current_recovery") or record.get("mpc_recovery_replan_count") or 0)
        return int(tried) < max_replans, "mpc_replan_budget_exceeded"

    if (not params["mpc_recovery_require_ordered_progress"]) and ("ordered" in reason or "ordered" in suffix_reason_lower):
        return True, reason or suffix_reason_lower or "ordered_relaxed"
    if (not params["mpc_recovery_require_live_progress"]) and ("progress" in reason or "progress" in suffix_reason_lower):
        return True, reason or suffix_reason_lower or "live_progress_relaxed"

    if params["mpc_recovery_require_live_progress"] and params["mpc_recovery_min_progress_delta"] <= 0.0:
        if "progress" in reason:
            return True, "progress_delta_zeroed"

    return False, reason or suffix_reason or "rejected"


def score_params(records: Sequence[JsonDict], params: Mapping[str, Any]) -> JsonDict:
    attempted = 0
    accepted = 0
    reject_counts: Counter[str] = Counter()
    clearances: list[float] = []
    timings: list[float] = []
    total = 0

    for record in records:
        if not safe_bool(record.get("mpc_recovery_replan_attempted")):
            continue
        total += 1
        simulated_ok, reason = simulate_mpc_acceptance(record, params)
        if simulated_ok:
            accepted += 1
        reject_counts[str(reason)] += 1 if not simulated_ok else 0
        attempted += 1

        clearance = finite_float(record.get("planned_min_clearance"))
        if clearance is not None:
            clearances.append(clearance)
        timing = finite_float(record.get("planner_timing_ms"))
        if timing is not None:
            timings.append(timing)

    accept_rate = float(accepted / attempted) if attempted else 0.0
    mean_clearance = float(np.mean(clearances)) if clearances else 0.0
    mean_timing = float(np.mean(timings)) if timings else 0.0

    objective = accept_rate + 0.01 * max(0.0, mean_clearance) - 0.0005 * mean_timing

    return {
        "params": dict(params),
        "objective": objective,
        "attempted": int(attempted),
        "accepted": int(accepted),
        "accept_rate": float(accept_rate),
        "mean_planned_clearance": float(mean_clearance),
        "mean_planner_timing_ms": float(mean_timing),
        "reject_reason_counts": dict(reject_counts),
        "strict_rejected": int(reject_counts.get("not_recover_mode", 0) + reject_counts.get("mpc_replan_budget_exceeded", 0)),
    }


def parameter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    keys = [
        "mpc_recovery_horizon",
        "mpc_recovery_prefix_len",
        "mpc_recovery_max_replans_per_recovery",
        "mpc_recovery_require_ordered_progress",
        "mpc_recovery_require_live_progress",
        "mpc_recovery_min_progress_delta",
        "mpc_recovery_no_progress_limit",
    ]

    value_lists = [
        [int(v) for v in parse_grid(args.horizon)],
        [int(v) for v in parse_grid(args.prefix_len)],
        [int(v) for v in parse_grid(args.max_replans)],
        parse_bool_grid(args.require_ordered_progress),
        parse_bool_grid(args.require_live_progress),
        parse_grid(args.min_progress_delta),
        [int(v) for v in parse_grid(args.no_progress_limit)],
    ]

    combos: list[dict[str, Any]] = []
    _grid = np.array(np.meshgrid(*value_lists), dtype=object)
    for idx in range(_grid.reshape(len(keys), -1).shape[1]):
        combo = {k: _grid[i].flatten()[idx] for i, k in enumerate(keys)}
        if (not combo["mpc_recovery_require_ordered_progress"]) and combo["mpc_recovery_require_live_progress"]:
            pass
        combos.append(combo)
    return combos


def yaml_snippet(best: Mapping[str, Any]) -> str:
    p = best
    return "\n".join(
        [
            "safety_filter:",
            "  overrides:",
            "    explicit_recovery:",
            f"      mpc_recovery_horizon: {p['mpc_recovery_horizon']}",
            f"      mpc_recovery_prefix_len: {p['mpc_recovery_prefix_len']}",
            f"      mpc_recovery_max_replans_per_recovery: {p['mpc_recovery_max_replans_per_recovery']}",
            f"      mpc_recovery_require_ordered_progress: {str(bool(p['mpc_recovery_require_ordered_progress'])).lower()}",
            f"      mpc_recovery_require_live_progress: {str(bool(p['mpc_recovery_require_live_progress'])).lower()}",
            f"      mpc_recovery_min_progress_delta: {p['mpc_recovery_min_progress_delta']}",
            f"      mpc_recovery_no_progress_limit: {p['mpc_recovery_no_progress_limit']}",
        ]
    )


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    if not records:
        raise SystemExit("No records loaded from inputs.")

    grid = parameter_grid(args)
    results = [score_params(records, params) for params in grid]
    results.sort(key=lambda item: item["objective"], reverse=True)

    top_k = max(1, int(args.top_k))
    top = results[:top_k]

    if args.strict_budget_match:
        for item in results:
            if item["params"]["mpc_recovery_max_replans_per_recovery"] <= 0:
                item["objective"] *= 0.98
        results.sort(key=lambda item: item["objective"], reverse=True)
        top = results[:top_k]

    summary = {
        "stage": "safechunk_mpc_recovery",
        "records": int(len(records)),
        "attempted_records": int(sum(1 for r in records if safe_bool(r.get("mpc_recovery_replan_attempted")))),
        "grid_size": int(len(results)),
        "best": top[0],
        "top": top,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.write_yaml:
        Path(args.write_yaml).write_text(yaml_snippet(top[0]["params"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
