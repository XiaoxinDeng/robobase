#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SNAPSHOT = Path("exp_local/pixel_act/bigym_drawer_top_open_20260528034109/snapshots/3000_snapshot.pt")
MANIFEST = Path("/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/manifest.json")
ENV = "bigym/human_arm_drawer_top_open"

@dataclass(frozen=True)
class Method:
    key: str
    label: str
    condition: str
    args: tuple[str, ...]
    safety_pause: bool = False

METHODS = (
    Method("act_baseline", "ACT baseline", "act", ()),
    Method("single_step_oscbf", "ACT + single-step OSCBF", "oscbf", ()),
    Method("sequential_oscbf", "ACT + sequential OSCBF", "sequential", ()),
    Method(
        "brake_hold_only",
        "ACT + brake/hold only",
        "chunk_deform",
        ("--chunk-deform-mode", "optimized", "--no-chunk-deformation-enabled", "--no-chunk-explicit-return"),
        True,
    ),
    Method(
        "chunk_deform",
        "ACT + SafeChunk-Deform",
        "chunk_deform",
        (
            "--chunk-deform-mode", "optimized",
            "--chunk-deformation-enabled",
            "--chunk-explicit-return",
            "--chunk-commit-accepted-chunks",
            "--chunk-repair-committed-action",
            "--chunk-monotonic-committed-repair",
        ),
        True,
    ),
)

FIELDS = (
    "method", "label", "returncode", "task_success_rate", "contact_rate", "contact_steps",
    "h_violation_rate", "final_task_progress", "max_task_progress", "act_ratio",
    "safety_mode_ratio", "fallback_ratio", "yield_steps", "return_steps", "brake_steps",
    "fallback_steps", "optimized_attempts", "optimized_accepted_count",
    "committed_chunk_started_count", "committed_chunk_completed_count",
    "committed_chunk_abort_count", "committed_repaired_step_count",
    "mean_planning_vs_replay_post_clearance_error", "mean_filter_ms", "p50_filter_ms",
    "p95_filter_ms", "max_filter_ms", "summary_json", "step_jsonl", "episodes_json", "run_log",
)

MAP = {
    "task_success_rate": ("task_success_rate", "success_rate"),
    "contact_rate": ("collision_episode_rate", "mean_contact_step_rate"),
    "contact_steps": ("total_contact_steps",),
    "h_violation_rate": ("mean_h_violation_rate",),
    "final_task_progress": ("final_task_progress",),
    "max_task_progress": ("max_task_progress",),
    "act_ratio": ("mean_act_ratio",),
    "safety_mode_ratio": ("mean_safety_mode_ratio",),
    "fallback_ratio": ("mean_fallback_ratio",),
    "yield_steps": ("yield_steps",),
    "return_steps": ("return_steps",),
    "brake_steps": ("brake_steps", "total_brake_steps"),
    "fallback_steps": ("fallback_steps",),
    "optimized_attempts": ("optimized_attempts", "total_optimized_attempt_count"),
    "optimized_accepted_count": ("optimized_accepted_count",),
    "committed_chunk_started_count": ("committed_chunk_started_count",),
    "committed_chunk_completed_count": ("committed_chunk_completed_count",),
    "committed_chunk_abort_count": ("committed_chunk_abort_count",),
    "committed_repaired_step_count": ("committed_repaired_step_count",),
    "mean_planning_vs_replay_post_clearance_error": ("mean_planning_vs_replay_clearance_post_error",),
    "mean_filter_ms": ("mean_filter_time_ms",),
    "max_filter_ms": ("max_filter_time_ms_over_episodes",),
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demos", type=int, default=40)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--methods", nargs="+", default=[m.key for m in METHODS])
    ap.add_argument("--stop-on-failure", action="store_true")
    return ap.parse_args()

def repo_root():
    return Path(__file__).resolve().parents[1]

def percentile(vals, pct):
    vals = sorted(vals)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

def load_times(path):
    vals = []
    if not path.exists():
        return vals
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                value = row.get("filter_time_ms")
                if value is not None:
                    vals.append(float(value))
            except Exception:
                pass
    return vals

def first(summary, keys):
    for k in keys:
        if k in summary and summary[k] is not None:
            return summary[k]
    return None

def common_args(args):
    return [
        "--snapshot", str(SNAPSHOT),
        "--env", ENV,
        "--episodes", str(args.episodes),
        "--steps", str(args.steps),
        "--demos", str(args.demos),
        "--seed", str(args.seed),
        "--normalization-source", "snapshot",
        "--enable-human-arm-collisions",
        "--diagnostics",
        "--no-record-video",
        "--no-progress",
        "--chunk-min-clearance", "0.12",
        "--chunk-lambda-safety", "500.0",
        "--chunk-lambda-rejoin", "0.5",
        "--chunk-lambda-path", "0.2",
        "--chunk-lambda-action", "0.1",
        "--chunk-opt-iters", "2",
        "--chunk-opt-population", "8",
        "--chunk-opt-elite-frac", "0.25",
        "--chunk-inner-rejoin-metric", "q_state",
        "--chunk-final-rejoin-metric", "q_state",
        "--no-chunk-use-ee-final-check",
        "--no-chunk-cache-nominal-ee",
        "--chunk-q-rejoin-threshold", "0.5",
        "--chunk-brake-if-unrecoverable",
        "--chunk-debug-safety-feasibility",
        "--override", f"env.manifest={MANIFEST}",
        "--override", "env.privileged_information=false",
        "--override", "env.require_mode_label=false",
        "--override", "frame_stack=4",
    ]

def build_cmd(method, args, out_dir):
    out_name = f"{method.key}.jsonl"
    cmd = [args.python, "eval_act_oscbf_safety_metrics.py", "--condition", method.condition, "--output-dir", str(out_dir), "--out", out_name]
    cmd.extend(common_args(args))
    if method.safety_pause:
        cmd.extend(["--pause-on-unsafe", "--pause-clearance-threshold", "0.0", "--reset-action-history-after-human-exit"])
    cmd.extend(method.args)
    return cmd

def aggregate(method, rc, out_dir):
    step_jsonl = out_dir / f"{method.key}.jsonl"
    episodes_json = out_dir / f"{method.key}_episodes.json"
    summary_json = out_dir / f"{method.key}_summary.json"
    run_log = out_dir / "run.log"
    summary = {}
    if summary_json.exists():
        summary = json.loads(summary_json.read_text())
    times = load_times(step_jsonl)
    row = {"method": method.key, "label": method.label, "returncode": rc,
           "summary_json": str(summary_json), "step_jsonl": str(step_jsonl),
           "episodes_json": str(episodes_json), "run_log": str(run_log),
           "p50_filter_ms": percentile(times, 50), "p95_filter_ms": percentile(times, 95)}
    for out, keys in MAP.items():
        row[out] = first(summary, keys)
    if row.get("max_filter_ms") is None and times:
        row["max_filter_ms"] = max(times)
    return row

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in FIELDS})

def main():
    args = parse_args()
    all_methods = {m.key: m for m in METHODS}
    methods = [all_methods[k] for k in args.methods]
    if args.output_root is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = repo_root() / "eval_safety" / f"publish_safety_set_{stamp}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        out_dir = args.output_root / method.key
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(method, args, out_dir)
        log_path = out_dir / "run.log"
        print(f"\n=== Running {method.key}: {method.label} ===", flush=True)
        print("$", shlex.join(cmd), flush=True)
        with log_path.open("w") as log:
            log.write("$ " + shlex.join(cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=repo_root(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            rc = proc.wait()
        print(f"=== {method.key} returncode={rc} ===", flush=True)
        rows.append(aggregate(method, rc, out_dir))
        write_csv(args.output_root / "aggregate.csv", rows)
        if rc and args.stop_on_failure:
            break
    write_csv(args.output_root / "aggregate.csv", rows)
    print(f"\nAggregate CSV: {args.output_root / 'aggregate.csv'}")
    return 1 if any((r.get("returncode") or 0) != 0 for r in rows) else 0

if __name__ == "__main__":
    raise SystemExit(main())
