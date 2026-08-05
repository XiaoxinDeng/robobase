#!/usr/bin/env python3
"""Offline tuner for SafeChunk-Deform candidate selection.

This script decouples the deform stage from the live safety filter loop.  It does
not step an environment.  Instead, it consumes saved candidate/case records from
JSONL, JSON, or NPZ files and asks: given the recorded deform losses and safety
metrics, which deform weights/gates would have selected and accepted the best
candidate most often?

Supported input shapes:

1. JSONL/JSON records with a top-level candidate, for example eval metrics rows:
   {
     "episode": 0,
     "step": 10,
     "candidate_type": "deform",
     "min_clearance": 0.02,
     "immediate_clearance": 0.04,
     "prefix_min_clearance": 0.01,
     "horizon_min_clearance": 0.02,
     "safety_loss": 0.1,
     "action_deviation_loss": 0.03,
     "smoothness_loss": 0.02,
     "path_loss": 0.04
   }

2. JSONL/JSON records with multiple saved candidates:
   {
     "case_id": "episode0_step10",
     "candidates": [
       {"candidate_name": "gd_seed", "min_clearance": ..., "safety_loss": ...},
       {"candidate_name": "gd_best", "min_clearance": ..., "safety_loss": ...}
     ]
   }

3. NPZ files where each first-dimension row is a candidate/case.  Arrays are
   converted into candidate dictionaries by key.

The script is intentionally scalar-metric based so deform can be tuned from saved
failure cases without rerunning ACT, MuJoCo, OSCBF, or recovery.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

Number = int | float
JsonDict = dict[str, Any]


# Deform scalar keys accepted from eval rows or candidate dumps.  The aliases let
# the tuner work with both optimizer loss records and StepMetrics records.
CLEARANCE_ALIASES: dict[str, tuple[str, ...]] = {
    "immediate": ("immediate_clearance", "deform_immediate_clearance", "best_immediate_clearance"),
    "prefix": ("prefix_min_clearance", "deform_prefix_min_clearance", "best_prefix_min_clearance"),
    "horizon": (
        "horizon_min_clearance",
        "deform_min_clearance",
        "deform_stage_min_clearance",
        "best_min_clearance",
        "min_clearance",
    ),
}
LOSS_ALIASES: dict[str, tuple[str, ...]] = {
    "safety": ("safety_loss", "deform_safety_loss"),
    "action": ("action_deviation_loss", "action_loss", "deform_action_loss"),
    "smooth": ("smoothness_loss", "smooth_loss", "deform_smoothness_loss"),
    "path": ("path_loss", "deform_path_loss", "existing_optimization_loss"),
    "envelope": ("deform_envelope_loss", "envelope_loss"),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the offline deform tuner."""
    parser = argparse.ArgumentParser(
        description="Tune SafeChunk-Deform weights/gates from saved candidate records."
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL, JSON, or NPZ candidate records.")
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--write-yaml", default=None, help="Optional YAML override snippet path.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of best parameter sets to print/save.")
    parser.add_argument("--require-full-horizon", action="store_true", help="Require horizon clearance for acceptance.")
    parser.add_argument("--allow-prefix", action="store_true", help="Allow safe-prefix acceptance when full horizon fails.")
    parser.add_argument("--lambda-safety", default="100,300,600", help="Comma/range values for lambda_safety.")
    parser.add_argument("--lambda-action", default="0.02,0.05,0.1,0.2", help="Comma/range values for lambda_action.")
    parser.add_argument("--lambda-smooth", default="0.02,0.05,0.1", help="Comma/range values for lambda_smooth.")
    parser.add_argument("--lambda-path", default="0.05,0.1,0.2", help="Comma/range values for lambda_path.")
    parser.add_argument("--lambda-envelope", default="0,0.1,1.0", help="Comma/range values for envelope loss weight.")
    parser.add_argument("--hard-min-clearance", default="0,0.02,0.04", help="Candidate immediate hard gate.")
    parser.add_argument("--desired-min-clearance", default="0,0.02,0.04,0.05", help="Full horizon desired gate.")
    parser.add_argument("--prefix-min-clearance", default="0,0.02,0.04", help="Safe-prefix gate.")
    return parser.parse_args()


def parse_grid(spec: str) -> list[float]:
    """Parse comma values plus optional start:stop:step ranges into floats."""
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


def load_records(paths: Sequence[str]) -> list[JsonDict]:
    """Load JSONL, JSON, or NPZ records and normalize them into case dicts."""
    records: list[JsonDict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() == ".npz":
            records.extend(load_npz_records(path))
        elif path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict) and isinstance(data.get("records"), list):
                records.extend(data["records"])
            elif isinstance(data, dict):
                records.append(data)
            else:
                raise ValueError(f"Unsupported JSON root in {path}: {type(data)!r}")
    return records


def load_npz_records(path: Path) -> list[JsonDict]:
    """Convert an NPZ archive into row-wise candidate records."""
    archive = np.load(path, allow_pickle=True)
    keys = list(archive.files)
    if not keys:
        return []
    first = archive[keys[0]]
    count = int(first.shape[0]) if getattr(first, "ndim", 0) > 0 else 1
    records: list[JsonDict] = []
    for idx in range(count):
        record: JsonDict = {"case_id": f"{path.stem}:{idx}"}
        for key in keys:
            value = archive[key]
            if getattr(value, "ndim", 0) == 0:
                record[key] = np_to_jsonable(value.item())
            elif value.shape[0] == count:
                record[key] = np_to_jsonable(value[idx])
            else:
                record[key] = np_to_jsonable(value)
        records.append(record)
    return records


def np_to_jsonable(value: Any) -> Any:
    """Convert NumPy values into plain Python containers for scoring/output."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return np_to_jsonable(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def finite_float(value: Any, default: float | None = None) -> float | None:
    """Coerce numeric values while treating None, NaN, and inf as missing."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def first_float(record: Mapping[str, Any], keys: Iterable[str], default: float | None = None) -> float | None:
    """Read the first finite scalar from a list of aliases."""
    for key in keys:
        value = finite_float(record.get(key), None)
        if value is not None:
            return value
    return default


def case_candidates(record: JsonDict) -> list[JsonDict]:
    """Return candidates for a case, falling back to the record itself."""
    candidates = record.get("candidates")
    if isinstance(candidates, list) and candidates:
        return [dict(c) for c in candidates if isinstance(c, Mapping)]
    candidate = dict(record)
    candidate.pop("candidates", None)
    return [candidate]


def deform_cost(candidate: Mapping[str, Any], params: Mapping[str, float]) -> float:
    """Compute the weighted deform objective used to select among candidates."""
    safety = first_float(candidate, LOSS_ALIASES["safety"], 0.0) or 0.0
    action = first_float(candidate, LOSS_ALIASES["action"], 0.0) or 0.0
    smooth = first_float(candidate, LOSS_ALIASES["smooth"], 0.0) or 0.0
    path = first_float(candidate, LOSS_ALIASES["path"], 0.0) or 0.0
    envelope = first_float(candidate, LOSS_ALIASES["envelope"], 0.0) or 0.0
    return float(
        params["lambda_safety"] * safety
        + params["lambda_action"] * action
        + params["lambda_smooth"] * smooth
        + params["lambda_path"] * path
        + params["lambda_envelope"] * envelope
    )


def deform_acceptance(candidate: Mapping[str, Any], params: Mapping[str, float], *, require_full: bool, allow_prefix: bool) -> tuple[bool, str]:
    """Apply deform acceptance gates to a selected candidate."""
    immediate = first_float(candidate, CLEARANCE_ALIASES["immediate"])
    prefix = first_float(candidate, CLEARANCE_ALIASES["prefix"], immediate)
    horizon = first_float(candidate, CLEARANCE_ALIASES["horizon"], prefix)
    hard = params["hard_min_clearance"]
    desired = params["desired_min_clearance"]
    prefix_gate = params["prefix_min_clearance"]

    if immediate is None:
        return False, "missing_immediate_clearance"
    if immediate < hard:
        return False, "immediate_below_hard_margin"
    if horizon is not None and horizon >= desired:
        return True, "full_horizon"
    if not require_full and allow_prefix and prefix is not None and prefix >= prefix_gate:
        return True, "safe_prefix"
    if horizon is None:
        return False, "missing_horizon_clearance"
    return False, "horizon_below_desired_margin"


def score_params(cases: Sequence[JsonDict], params: Mapping[str, float], *, require_full: bool, allow_prefix: bool) -> JsonDict:
    """Score one deform parameter set across all cases."""
    accepted = 0
    total = 0
    reason_counts: Counter[str] = Counter()
    clearance_values: list[float] = []
    cost_values: list[float] = []
    selected_names: Counter[str] = Counter()

    for case in cases:
        candidates = case_candidates(case)
        if not candidates:
            continue
        total += 1
        selected = min(candidates, key=lambda item: deform_cost(item, params))
        accepted_flag, reason = deform_acceptance(selected, params, require_full=require_full, allow_prefix=allow_prefix)
        accepted += int(accepted_flag)
        reason_counts[reason] += 1
        selected_names[str(selected.get("candidate_name", selected.get("name", "candidate")))] += 1
        clearance = first_float(selected, CLEARANCE_ALIASES["horizon"])
        if clearance is not None:
            clearance_values.append(clearance)
        cost_values.append(deform_cost(selected, params))

    accept_rate = float(accepted / total) if total else 0.0
    mean_clearance = float(np.mean(clearance_values)) if clearance_values else None
    mean_cost = float(np.mean(cost_values)) if cost_values else None
    # Prefer acceptance, then clearance, then lower weighted cost.
    objective = accept_rate + 0.01 * (mean_clearance or 0.0) - 1e-6 * (mean_cost or 0.0)
    return {
        "params": dict(params),
        "objective": objective,
        "accepted": accepted,
        "total": total,
        "accept_rate": accept_rate,
        "mean_selected_clearance": mean_clearance,
        "mean_selected_cost": mean_cost,
        "reject_reason_counts": dict(reason_counts),
        "selected_candidate_counts": dict(selected_names),
    }


def parameter_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    """Build the deform tuning grid from CLI values."""
    keys = [
        "lambda_safety",
        "lambda_action",
        "lambda_smooth",
        "lambda_path",
        "lambda_envelope",
        "hard_min_clearance",
        "desired_min_clearance",
        "prefix_min_clearance",
    ]
    values = [
        parse_grid(args.lambda_safety),
        parse_grid(args.lambda_action),
        parse_grid(args.lambda_smooth),
        parse_grid(args.lambda_path),
        parse_grid(args.lambda_envelope),
        parse_grid(args.hard_min_clearance),
        parse_grid(args.desired_min_clearance),
        parse_grid(args.prefix_min_clearance),
    ]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def yaml_snippet(params: Mapping[str, float]) -> str:
    """Render a nested eval override snippet for the deform stage."""
    return "\n".join(
        [
            "safety_filter:",
            "  overrides:",
            "    optimized_deform:",
            "      optimizer_method: gradient",
            f"    lambda_safety: {params['lambda_safety']}",
            f"    lambda_action: {params['lambda_action']}",
            f"    lambda_smooth: {params['lambda_smooth']}",
            f"    lambda_path: {params['lambda_path']}",
            "    safechunk_acceptance:",
            f"      hard_min_clearance: {params['hard_min_clearance']}",
            f"      desired_min_clearance: {params['desired_min_clearance']}",
            f"      prefix_min_clearance: {params['prefix_min_clearance']}",
            "",
        ]
    )


def main() -> None:
    """Run the offline deform tuning grid and write summaries."""
    args = parse_args()
    cases = load_records(args.input)
    if not cases:
        raise SystemExit("No deform cases loaded.")
    results = [
        score_params(cases, params, require_full=args.require_full_horizon, allow_prefix=args.allow_prefix)
        for params in parameter_grid(args)
    ]
    results.sort(key=lambda item: item["objective"], reverse=True)
    top = results[: max(1, int(args.top_k))]
    summary = {
        "stage": "deform",
        "num_cases": len(cases),
        "num_grid_points": len(results),
        "require_full_horizon": bool(args.require_full_horizon),
        "allow_prefix": bool(args.allow_prefix),
        "best": top[0],
        "top": top,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_yaml:
        Path(args.write_yaml).write_text(yaml_snippet(top[0]["params"]), encoding="utf-8")


if __name__ == "__main__":
    main()
