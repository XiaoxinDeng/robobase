#!/usr/bin/env python3
"""Export requested safety-eval metrics for one or more run folders.

The exporter is dependency-free and writes an XLSX workbook with:
- runs: headline metrics per run
- time_breakdown: timing mean/p50/p95/max per run
- episodes: per-episode provenance rows

A run folder may contain direct eval outputs, or split subfolders such as
``media_ep0`` and ``metrics_ep1_9``. In both cases the script scans for
``*_episodes.json`` and matching ``*.jsonl`` step logs below each run root.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


HEADLINE_COLUMNS = [
    "run",
    "run_dir",
    "num_episodes",
    "success_rate",
    "h_violation_rate",
    "h_violations_total",
    "mean_steps",
    "mean_all_episode_length",
    "mean_failed_episode_length",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "max_latency_ms",
    "mean_path_deviation",
    "max_path_deviation",
    "mean_intervention_rate",
    "weighted_intervention_rate",
    "contact_count_total",
    "collision_episode_rate",
    "step_count",
]

EPISODE_COLUMNS = [
    "run",
    "combined_episode",
    "source_segment",
    "source_episode",
    "success",
    "task_success",
    "episode_length",
    "h_violation_count",
    "h_violation_rate",
    "episode_min_h",
    "mean_filter_time_ms",
    "mean_path_deviation",
    "max_path_deviation",
    "mean_final_path_deviation",
    "intervention_frequency",
    "total_brake_steps",
    "likely_failure_cause",
    "contact_count_total",
    "contact_step_count",
]

TIME_METRICS = [
    "filter_time_ms",
    "monitor_time_ms",
    "env_step_time_ms",
    "policy_obs_adapt_time_ms",
    "policy_action_time_ms",
    "policy_obs_update_time_ms",
    "policy_total_time_ms_recomputed",
    "step_wall_time_ms",
]


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if finite_number(v)]
    return sum(vals) / len(vals) if vals else None


def total(values: Iterable[Any]) -> float:
    return sum(float(v) for v in values if finite_number(v))


def percentile(values: Iterable[Any], pct: float) -> float | None:
    vals = sorted(float(v) for v in values if finite_number(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def max_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if finite_number(v)]
    return max(vals) if vals else None


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def step_jsonl_for_episode_json(path: Path) -> Path:
    suffix = "_episodes.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"Not an episode summary path: {path}")
    return path.with_name(path.name[: -len(suffix)] + ".jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def collect_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_files = sorted(run_dir.rglob("*_episodes.json"))
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    combined_episode = 0

    for episode_file in episode_files:
        try:
            loaded = read_json(episode_file)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, list):
            continue

        segment = episode_file.parent.relative_to(run_dir).as_posix()
        if segment == ".":
            segment = run_dir.name
        episode_map: dict[Any, int] = {}
        for ep in loaded:
            if not isinstance(ep, dict):
                continue
            row = dict(ep)
            original_episode = row.get("episode")
            row["source_segment"] = segment
            row["source_episode"] = original_episode
            row["combined_episode"] = combined_episode
            row["episodes_json"] = str(episode_file)
            episode_map[original_episode] = combined_episode
            combined_episode += 1
            episodes.append(row)

        step_file = step_jsonl_for_episode_json(episode_file)
        for row in load_jsonl(step_file):
            out = dict(row)
            out["source_segment"] = segment
            out["source_episode"] = out.get("episode")
            out["combined_episode"] = episode_map.get(out.get("episode"), out.get("episode"))
            out["step_jsonl"] = str(step_file)
            steps.append(out)

    if not episodes:
        combined = run_dir / "combined_summary.json"
        if combined.exists():
            data = read_json(combined)
            if isinstance(data, dict) and isinstance(data.get("episodes"), list):
                for i, ep in enumerate(data["episodes"]):
                    if not isinstance(ep, dict):
                        continue
                    row = dict(ep)
                    row.setdefault("combined_episode", i)
                    row.setdefault("source_segment", "combined_summary")
                    row.setdefault("source_episode", row.get("episode"))
                    episodes.append(row)

    return episodes, steps


def episode_success(ep: dict[str, Any]) -> bool:
    if ep.get("success") is not None:
        return bool(ep.get("success"))
    return bool(ep.get("task_success"))


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if finite_number(row.get(key))]


def recomputed_policy_total(rows: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    keys = (
        "policy_obs_adapt_time_ms",
        "policy_action_time_ms",
        "policy_obs_update_time_ms",
    )
    for row in rows:
        parts = [row.get(key) for key in keys]
        if all(finite_number(part) for part in parts):
            out.append(sum(float(part) for part in parts))
    return out


def time_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    if metric == "policy_total_time_ms_recomputed":
        return recomputed_policy_total(rows)
    if metric == "step_wall_time_ms":
        return [1000.0 * float(v) for v in values(rows, "step_wall_time_s")]
    return values(rows, metric)


def first_non_null(summary: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return value
    return None


def fallback_aggregate(run_dir: Path) -> dict[str, Any]:
    combined = run_dir / "combined_summary.json"
    if combined.exists():
        data = read_json(combined)
        if isinstance(data, dict) and isinstance(data.get("aggregate"), dict):
            return dict(data["aggregate"])
    summary_files = sorted(run_dir.rglob("*_summary.json"))
    if summary_files:
        try:
            summary = read_json(summary_files[0])
        except (OSError, json.JSONDecodeError):
            return {}
        return summary if isinstance(summary, dict) else {}
    return {}


def summarize_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episodes, steps = collect_run(run_dir)
    fallback = fallback_aggregate(run_dir)

    successful = [ep for ep in episodes if episode_success(ep)]
    failed = [ep for ep in episodes if not episode_success(ep)]

    h_rows = [row for row in steps if row.get("h_violation") is not None]
    h_rate = (
        sum(1 for row in h_rows if bool(row.get("h_violation"))) / len(h_rows)
        if h_rows
        else first_non_null(fallback, ("h_violation_step_rate", "mean_h_violation_rate"))
    )
    h_total = (
        sum(1 for row in h_rows if bool(row.get("h_violation")))
        if h_rows
        else first_non_null(fallback, ("h_violations", "total_h_violations"))
    )

    filter_times = values(steps, "filter_time_ms")
    path_mean = values(steps, "path_mean_deviation")
    path_max = values(steps, "path_max_deviation")
    intervention_rows = [row for row in steps if row.get("intervention_active") is not None]
    weighted_intervention = (
        sum(1 for row in intervention_rows if bool(row.get("intervention_active")))
        / len(intervention_rows)
        if intervention_rows
        else None
    )

    row = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "num_episodes": len(episodes) or fallback.get("num_episodes"),
        "success_rate": (
            mean(1.0 if episode_success(ep) else 0.0 for ep in episodes)
            if episodes
            else fallback.get("success_rate")
        ),
        "h_violation_rate": h_rate,
        "h_violations_total": h_total,
        "mean_steps": (
            mean(ep.get("episode_length") for ep in successful)
            if episodes
            else first_non_null(fallback, ("mean_steps", "mean_success_episode_length", "mean_episode_length"))
        ),
        "mean_all_episode_length": (
            mean(ep.get("episode_length") for ep in episodes)
            if episodes
            else fallback.get("mean_all_episode_length")
        ),
        "mean_failed_episode_length": (
            mean(ep.get("episode_length") for ep in failed)
            if episodes
            else fallback.get("mean_failed_episode_length")
        ),
        "mean_latency_ms": mean(filter_times) if filter_times else first_non_null(fallback, ("latency_ms_mean_filter_time", "mean_filter_time_ms")),
        "p50_latency_ms": percentile(filter_times, 50.0) if filter_times else fallback.get("latency_ms_p50_filter_time"),
        "p95_latency_ms": percentile(filter_times, 95.0) if filter_times else first_non_null(fallback, ("latency_ms_p95_filter_time", "p95_filter_ms")),
        "max_latency_ms": max_or_none(filter_times) if filter_times else first_non_null(fallback, ("latency_ms_max_filter_time", "max_filter_time_ms")),
        "mean_path_deviation": mean(path_mean) if path_mean else first_non_null(fallback, ("overall_path_mean_deviation", "mean_path_deviation")),
        "max_path_deviation": max_or_none(path_max) if path_max else first_non_null(fallback, ("overall_path_max_deviation", "max_path_deviation_over_episodes")),
        "mean_intervention_rate": (
            mean(ep.get("intervention_frequency") for ep in episodes)
            if episodes
            else fallback.get("mean_intervention_frequency")
        ),
        "weighted_intervention_rate": weighted_intervention,
        "contact_count_total": total(ep.get("contact_count_total") for ep in episodes) if episodes else fallback.get("total_contacts"),
        "collision_episode_rate": (
            mean(1.0 if ep.get("contact_episode") else 0.0 for ep in episodes)
            if episodes
            else fallback.get("collision_episode_rate")
        ),
        "step_count": len(steps),
    }
    return row, episodes, steps


def column_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(value: Any, row_index: int, column_index: int) -> str:
    ref = f"{column_name(column_index)}{row_index}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if finite_number(value):
        return f'<c r="{ref}"><v>{float(value):.15g}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def sheet_xml(rows: list[list[Any]]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>',
    ]
    for row_index, row in enumerate(rows, 1):
        parts.append(f'<row r="{row_index}">')
        for column_index, value in enumerate(row, 1):
            parts.append(cell_xml(value, row_index, column_index))
        parts.append("</row>")
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def write_xlsx(path: Path, sheets: dict[str, list[list[Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets.keys())
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, len(sheet_names) + 1)
    )
    workbook_sheets = "".join(
        f'<sheet name="{escape(name[:31])}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(sheet_names, 1)
    )
    workbook_rels = "".join(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(sheet_names) + 1)
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{overrides}</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>',
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{workbook_rels}"
            f'<Relationship Id="rId{len(sheet_names) + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/></Relationships>',
        )
        zf.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '</styleSheet>',
        )
        for index, name in enumerate(sheet_names, 1):
            zf.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(sheets[name]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export requested h/path/latency/intervention metrics for eval runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more eval run directories. Each may contain split sub-runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("requested_metrics_time_summary.xlsx"),
        help="Output XLSX path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON copy of the per-run headline rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_rows: list[dict[str, Any]] = []
    time_rows: list[list[Any]] = [["run", "time_metric_ms", "mean", "p50", "p95", "max", "count"]]
    episode_rows: list[list[Any]] = [EPISODE_COLUMNS]

    for run_dir in args.run_dirs:
        run_dir = run_dir.expanduser().resolve()
        row, episodes, steps = summarize_run(run_dir)
        run_rows.append(row)

        for metric in TIME_METRICS:
            vals = time_values(steps, metric)
            time_rows.append([
                row["run"],
                metric,
                mean(vals),
                percentile(vals, 50.0),
                percentile(vals, 95.0),
                max_or_none(vals),
                len(vals),
            ])

        for ep in episodes:
            episode_rows.append([
                row["run"],
                ep.get("combined_episode"),
                ep.get("source_segment"),
                ep.get("source_episode"),
                ep.get("success"),
                ep.get("task_success"),
                ep.get("episode_length"),
                ep.get("h_violation_count"),
                ep.get("h_violation_rate"),
                ep.get("episode_min_h"),
                ep.get("mean_filter_time_ms"),
                ep.get("mean_path_deviation"),
                ep.get("max_path_deviation"),
                ep.get("mean_final_path_deviation"),
                ep.get("intervention_frequency"),
                ep.get("total_brake_steps"),
                ep.get("likely_failure_cause"),
                ep.get("contact_count_total"),
                ep.get("contact_step_count"),
            ])

    run_sheet = [HEADLINE_COLUMNS] + [[row.get(col) for col in HEADLINE_COLUMNS] for row in run_rows]
    write_xlsx(args.output, {"runs": run_sheet, "time_breakdown": time_rows, "episodes": episode_rows})
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(run_rows, indent=2))

    print(f"Wrote {args.output}")
    for row in run_rows:
        print(
            f"{row['run']}: h_rate={row.get('h_violation_rate')} "
            f"mean_steps={row.get('mean_steps')} "
            f"latency_ms={row.get('mean_latency_ms')} "
            f"path_dev={row.get('mean_path_deviation')} "
            f"intervention={row.get('mean_intervention_rate')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
