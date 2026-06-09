#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SNAPSHOT = Path(
    "exp_local/pixel_act/bigym_drawer_top_open_20260528034109/"
    "snapshots/3000_snapshot.pt"
)
DEFAULT_MANIFEST = Path(
    "/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/"
    "JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/"
    "lightweight/manifest.json"
)
DEFAULT_ENV = "bigym/human_arm_drawer_top_open"


@dataclass(frozen=True)
class Method:
    name: str
    condition: str
    description: str
    args: tuple[str, ...]


METHODS: tuple[Method, ...] = (
    Method(
        name="act_baseline",
        condition="act",
        description="Raw ACT policy, monitored for contacts only.",
        args=(),
    ),
    Method(
        name="brake_hold_only",
        condition="chunk_deform",
        description="SafeChunk path-consistent brake/hold fallback with deformation disabled.",
        args=(
            "--chunk-deform-mode",
            "optimized",
            "--no-chunk-deformation-enabled",
            "--no-chunk-explicit-return",
        ),
    ),
    Method(
        name="one_stage_optimized_recoverable",
        condition="chunk_deform",
        description="One-stage optimized recoverable deformation without explicit yield-return.",
        args=(
            "--chunk-deform-mode",
            "optimized",
            "--chunk-deformation-enabled",
            "--no-chunk-explicit-return",
            "--chunk-commit-accepted-chunks",
        ),
    ),
    Method(
        name="explicit_recovery",
        condition="chunk_deform",
        description="Optimized explicit yield-return recovery with committed replay and monotonic repair.",
        args=(
            "--chunk-deform-mode",
            "optimized",
            "--chunk-deformation-enabled",
            "--chunk-explicit-return",
            "--chunk-commit-accepted-chunks",
            "--chunk-repair-committed-action",
            "--chunk-monotonic-committed-repair",
        ),
    ),
    Method(
        name="explicit_no_monotonic_repair",
        condition="chunk_deform",
        description="Explicit recovery ablation: committed repair can apply even when clearance worsens.",
        args=(
            "--chunk-deform-mode",
            "optimized",
            "--chunk-deformation-enabled",
            "--chunk-explicit-return",
            "--chunk-commit-accepted-chunks",
            "--chunk-repair-committed-action",
            "--no-chunk-monotonic-committed-repair",
        ),
    ),
    Method(
        name="explicit_no_committed_replay",
        condition="chunk_deform",
        description="Explicit recovery ablation: accepted chunks are not committed across outer steps.",
        args=(
            "--chunk-deform-mode",
            "optimized",
            "--chunk-deformation-enabled",
            "--chunk-explicit-return",
            "--no-chunk-commit-accepted-chunks",
            "--chunk-repair-committed-action",
            "--chunk-monotonic-committed-repair",
        ),
    ),
    Method(
        name="explicit_fixed_replay_semantics",
        condition="chunk_deform",
        description=(
            "Current fixed implementation: committed replay safety checks predicted "
            "post-action clearance, matching planning semantics. Legacy pre-action "
            "mismatch is not toggled by this runner."
        ),
        args=(
            "--chunk-deform-mode",
            "optimized",
            "--chunk-deformation-enabled",
            "--chunk-explicit-return",
            "--chunk-commit-accepted-chunks",
            "--chunk-repair-committed-action",
            "--chunk-monotonic-committed-repair",
        ),
    ),
)
METHOD_BY_NAME = {method.name: method for method in METHODS}


REQUESTED_FIELDS = (
    "method",
    "description",
    "returncode",
    "task_success_rate",
    "contact_rate",
    "contact_steps",
    "h_violation_rate",
    "final_task_progress",
    "max_task_progress",
    "act_ratio",
    "yield_steps",
    "return_steps",
    "brake_steps",
    "fallback_steps",
    "optimized_attempts",
    "optimized_accepted_count",
    "committed_chunk_started_count",
    "committed_chunk_completed_count",
    "committed_chunk_abort_count",
    "committed_abort_due_to_state_mismatch_count",
    "committed_abort_due_to_safety_semantics_mismatch_count",
    "committed_repaired_step_count",
    "mean_planning_vs_replay_post_clearance_error",
    "mean_filter_ms",
    "p50_filter_ms",
    "p95_filter_ms",
    "max_filter_ms",
    "summary_json",
    "step_jsonl",
    "log_file",
)


SUMMARY_MAP = {
    "task_success_rate": ("task_success_rate", "success_rate"),
    "contact_rate": ("collision_episode_rate", "mean_contact_step_rate"),
    "contact_steps": ("total_contact_steps",),
    "h_violation_rate": ("mean_h_violation_rate",),
    "final_task_progress": ("final_task_progress",),
    "max_task_progress": ("max_task_progress",),
    "act_ratio": ("mean_act_ratio",),
    "yield_steps": ("yield_steps",),
    "return_steps": ("return_steps",),
    "brake_steps": ("brake_steps", "total_brake_steps"),
    "fallback_steps": ("fallback_steps",),
    "optimized_attempts": ("optimized_attempts", "total_optimized_attempt_count"),
    "optimized_accepted_count": ("optimized_accepted_count",),
    "committed_chunk_started_count": ("committed_chunk_started_count",),
    "committed_chunk_completed_count": ("committed_chunk_completed_count",),
    "committed_chunk_abort_count": ("committed_chunk_abort_count",),
    "committed_abort_due_to_state_mismatch_count": (
        "committed_state_mismatch_abort_count",
    ),
    "committed_abort_due_to_safety_semantics_mismatch_count": (
        "committed_abort_due_to_safety_semantics_mismatch_count",
    ),
    "committed_repaired_step_count": ("committed_repaired_step_count",),
    "mean_planning_vs_replay_post_clearance_error": (
        "mean_planning_vs_replay_clearance_post_error",
    ),
    "mean_filter_ms": ("mean_filter_time_ms",),
    "max_filter_ms": ("max_filter_time_ms_over_episodes",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SafeChunk-Deform longer evals and ablations reproducibly."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demos", type=int, default=40)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root folder for all ablation outputs. Defaults to eval_safety/safechunk_ablation_<timestamp>.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Method names to run, or 'all'.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra evaluator argument token. Repeat for each token, e.g. --extra-arg --debug.",
    )
    parser.add_argument(
        "--opt-iters",
        type=int,
        default=2,
        help="CEM iterations for optimized deformation. Matches fast smoke defaults.",
    )
    parser.add_argument("--opt-population", type=int, default=8)
    parser.add_argument("--chunk-min-clearance", type=float, default=0.12)
    parser.add_argument("--chunk-lambda-safety", type=float, default=500.0)
    parser.add_argument("--chunk-lambda-rejoin", type=float, default=0.5)
    parser.add_argument("--chunk-lambda-path", type=float, default=0.2)
    parser.add_argument("--chunk-lambda-action", type=float, default=0.1)
    parser.add_argument("--q-rejoin-threshold", type=float, default=0.5)
    return parser.parse_args()


def selected_methods(names: Iterable[str]) -> list[Method]:
    names = list(names)
    if "all" in names:
        return list(METHODS)
    unknown = [name for name in names if name not in METHOD_BY_NAME]
    if unknown:
        valid = ", ".join(method.name for method in METHODS)
        raise SystemExit(f"Unknown methods {unknown}. Valid methods: {valid}")
    return [METHOD_BY_NAME[name] for name in names]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def build_common_args(args: argparse.Namespace) -> list[str]:
    manifest = str(args.manifest)
    return [
        "--snapshot",
        str(args.snapshot),
        "--env",
        args.env,
        "--episodes",
        str(args.episodes),
        "--steps",
        str(args.steps),
        "--demos",
        str(args.demos),
        "--seed",
        str(args.seed),
        "--normalization-source",
        "snapshot",
        "--enable-human-arm-collisions",
        "--diagnostics",
        "--no-record-video",
        "--no-progress",
        "--chunk-min-clearance",
        str(args.chunk_min_clearance),
        "--chunk-lambda-safety",
        str(args.chunk_lambda_safety),
        "--chunk-lambda-rejoin",
        str(args.chunk_lambda_rejoin),
        "--chunk-lambda-path",
        str(args.chunk_lambda_path),
        "--chunk-lambda-action",
        str(args.chunk_lambda_action),
        "--chunk-opt-iters",
        str(args.opt_iters),
        "--chunk-opt-population",
        str(args.opt_population),
        "--chunk-opt-elite-frac",
        "0.25",
        "--chunk-inner-rejoin-metric",
        "q_state",
        "--chunk-final-rejoin-metric",
        "q_state",
        "--no-chunk-use-ee-final-check",
        "--no-chunk-cache-nominal-ee",
        "--chunk-q-rejoin-threshold",
        str(args.q_rejoin_threshold),
        "--chunk-brake-if-unrecoverable",
        "--chunk-debug-safety-feasibility",
        "--override",
        f"env.manifest={manifest}",
        "--override",
        "env.privileged_information=false",
        "--override",
        "env.require_mode_label=false",
        "--override",
        "frame_stack=4",
    ]


def build_command(
    method: Method,
    args: argparse.Namespace,
    method_dir: Path,
) -> tuple[list[str], Path, Path, Path]:
    out_name = f"{method.name}.jsonl"
    step_jsonl = method_dir / out_name
    summary_json = method_dir / f"{method.name}_summary.json"
    log_file = method_dir / "run.log"
    cmd = [
        args.python,
        "eval_act_oscbf_safety_metrics.py",
        "--condition",
        method.condition,
        "--output-dir",
        str(method_dir),
        "--out",
        out_name,
    ]
    cmd.extend(build_common_args(args))
    if method.condition != "act":
        cmd.extend([
            "--pause-on-unsafe",
            "--pause-clearance-threshold",
            "0.0",
            "--reset-action-history-after-human-exit",
        ])
    cmd.extend(method.args)
    cmd.extend(args.extra_arg)
    return cmd, summary_json, step_jsonl, log_file


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def load_filter_times(step_jsonl: Path) -> list[float]:
    values: list[float] = []
    if not step_jsonl.exists():
        return values
    with step_jsonl.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get("filter_time_ms")
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return values


def first_summary_value(summary: dict, names: tuple[str, ...]):
    for name in names:
        if name in summary and summary[name] is not None:
            return summary[name]
    return None


def aggregate_row(
    method: Method,
    returncode: int | None,
    summary_json: Path,
    step_jsonl: Path,
    log_file: Path,
) -> dict:
    summary = {}
    if summary_json.exists():
        with summary_json.open() as f:
            summary = json.load(f)
    filter_times = load_filter_times(step_jsonl)
    row = {
        "method": method.name,
        "description": method.description,
        "returncode": returncode,
        "summary_json": str(summary_json),
        "step_jsonl": str(step_jsonl),
        "log_file": str(log_file),
        "p50_filter_ms": percentile(filter_times, 50.0),
        "p95_filter_ms": percentile(filter_times, 95.0),
    }
    for output_name, summary_names in SUMMARY_MAP.items():
        row[output_name] = first_summary_value(summary, summary_names)
    if row.get("max_filter_ms") is None and filter_times:
        row["max_filter_ms"] = max(filter_times)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUESTED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in REQUESTED_FIELDS})


def run_method(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("$", shlex.join(cmd), flush=True)
    with log_file.open("w") as log:
        log.write("$ " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if args.output_root is None:
        args.output_root = root / "eval_safety" / f"safechunk_ablation_{timestamp()}"
    args.output_root.mkdir(parents=True, exist_ok=True)

    methods = selected_methods(args.methods)
    rows: list[dict] = []
    for method in methods:
        method_dir = args.output_root / method.name
        cmd, summary_json, step_jsonl, log_file = build_command(method, args, method_dir)
        if args.dry_run:
            print("$", shlex.join(cmd))
            rows.append(aggregate_row(method, None, summary_json, step_jsonl, log_file))
            continue
        if args.skip_existing and summary_json.exists():
            print(f"Skipping {method.name}: found {summary_json}", flush=True)
            returncode = 0
        else:
            print(f"\n=== Running {method.name}: {method.description} ===", flush=True)
            returncode = run_method(cmd, log_file)
            print(f"=== {method.name} returncode={returncode} ===", flush=True)
        rows.append(aggregate_row(method, returncode, summary_json, step_jsonl, log_file))
        write_csv(args.output_root / "aggregate.csv", rows)
        if returncode and args.stop_on_failure:
            break

    csv_path = args.output_root / "aggregate.csv"
    write_csv(csv_path, rows)
    print(f"\nAggregate CSV: {csv_path}")
    return 1 if any((row.get("returncode") or 0) != 0 for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
