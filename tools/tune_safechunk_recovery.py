#!/usr/bin/env python3
"""Offline tuner for SafeChunk recovery acceptance and candidate selection.

This script decouples recovery from the live SafeChunk-Deform filter.  It uses
saved recovery candidate metrics from JSONL, JSON, or NPZ files and sweeps the
recovery objective weights plus hard gates.  This lets us answer questions like:
"would this candidate be accepted if ordered-path were softened?" or "which
recovery safety/rejoin weights select the least rejected candidate?" without
rerunning ACT, MuJoCo, OSCBF, or the deform stage.

The input can be raw eval metrics rows, one row per attempted recovery, or richer
case records with a "candidates" list.  Candidate dictionaries may include:
recover_immediate_clearance, recover_prefix_min_clearance,
recover_path_min_clearance, q_rejoin_dist, qd_rejoin_dist,
recover_ordered_pose_loss, recover_ordered_delta_loss, recover_task_progress_score,
safety_loss, action_deviation_loss, smoothness_loss, and recover_reject_reason.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

JsonDict = dict[str, Any]

HUMAN_ARM_CHUNK_DEFORM_ENV = "bigym/human_arm_drawer_top_open"
DEFAULT_BIGYM_ACT_SNAPSHOT = "exp_local/pixel_act/bigym_drawer_top_open_20260528034109/snapshots/3000_snapshot.pt"
DEFAULT_BIGYM_EXEC_OVERRIDES = [
    "env.episode_length=400000",
    "env.manifest=/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/manifest.json",
    "env.privileged_information=false",
    "env.require_mode_label=false",
    "frame_stack=4",
]
STANDALONE_HUMAN_ARM_CHUNK_DEFORM_LIVE_THRESHOLDS: dict[str, float | bool] = {
    "q_rejoin_threshold": 0.25,
    "require_qd_rejoin": True,
    "qd_rejoin_threshold": 3.0,
    "qd_rejoin_hard_threshold": 6.0,
    "ordered_pose_threshold": 0.315,
    "ordered_delta_threshold": 0.25,
    "require_ordered_path": True,
    "recover_path_min_clearance": 0.05,
    "recover_immediate_hard_clearance": 0.05,
    "recover_prefix_min_clearance": 0.05,
}


CLEARANCE_ALIASES: dict[str, tuple[str, ...]] = {
    "immediate": (
        "recover_immediate_clearance",
        "return_immediate_clearance",
        "immediate_clearance",
        "planned_clearance_post_min",
        "planned_min_clearance",
    ),
    "prefix": (
        "recover_prefix_min_clearance",
        "return_prefix_min_clearance",
        "prefix_min_clearance",
        "planned_clearance_pre_min",
        "planned_clearance_post_min",
        "planned_min_clearance",
    ),
    "path": (
        "recover_path_min_clearance",
        "return_min_clearance",
        "recover_min_clearance",
        "planned_min_clearance",
        "planned_clearance_post_min",
        "min_clearance",
    ),
}
LOSS_ALIASES: dict[str, tuple[str, ...]] = {
    "safety": ("safety_loss", "recover_safety_loss", "recover_clearance_margin_loss", "return_safety_loss"),
    "action": ("action_deviation_loss", "recover_action_loss", "return_action_loss"),
    "smooth": ("smoothness_loss", "recover_smoothness_loss", "return_smoothness_loss"),
    "rejoin": ("recover_rejoin_loss", "return_rejoin_loss", "q_rejoin_loss", "rejoin_loss"),
    "ordered_pose": ("recover_ordered_pose_loss", "ordered_pose_loss"),
    "ordered_delta": ("recover_ordered_delta_loss", "ordered_delta_loss"),
    "task_progress": ("recover_task_progress_score", "task_progress_score"),
    "direction": ("recover_direction_loss", "direction_loss"),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the offline recovery tuner."""
    parser = argparse.ArgumentParser(
        description="Tune SafeChunk recovery weights/gates from saved candidate records."
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL, JSON, or NPZ recovery records.")
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--write-yaml", default=None, help="Optional YAML override snippet path.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of best parameter sets to print/save.")
    parser.add_argument("--require-ordered-path", action="store_true", default=None, help="Force ordered path to be a hard gate.")
    parser.add_argument("--soften-ordered-path", action="store_true", help="Do not hard reject on ordered_path_failed.")
    parser.add_argument("--require-qd-rejoin", action="store_true", help="Keep qdot rejoin as a hard gate.")
    parser.add_argument("--safety-weight", default="50,100,300", help="Comma/range values for recovery safety weight.")
    parser.add_argument("--clearance-penalty-scale", default="1,5,10", help="Comma/range values for recovery clearance penalty scale.")
    parser.add_argument("--rejoin-weight", default="0.2,1,2,5,10", help="Comma/range values for q rejoin weight.")
    parser.add_argument("--task-progress-weight", default="1,5,10,20", help="Comma/range values for task progress reward weight.")
    parser.add_argument("--ordered-pose-weight", default="0,1,5", help="Comma/range values for ordered pose weight.")
    parser.add_argument("--ordered-delta-weight", default="0,1,3", help="Comma/range values for ordered delta weight.")
    parser.add_argument("--action-weight", default="0.05,0.2,1", help="Comma/range values for action deviation weight.")
    parser.add_argument("--smoothness-weight", default="0.02,0.1,0.5", help="Comma/range values for smoothness weight.")
    parser.add_argument("--path-min-clearance", default="0,0.02,0.04", help="Full recovery path clearance gate.")
    parser.add_argument("--immediate-hard-clearance", default="0,0.02", help="First recovery step clearance gate.")
    parser.add_argument("--prefix-min-clearance", default="0,0.02,0.04", help="Recovery prefix clearance gate.")
    parser.add_argument("--q-rejoin-threshold", default="0.5,1.0,1.5", help="q-state rejoin distance gate.")
    parser.add_argument("--qd-rejoin-threshold", default="3,6,8", help="qdot diagnostic/hard gate if enabled.")
    parser.add_argument("--ordered-pose-threshold", default="0.01,0.02,0.05", help="Ordered pose hard gate.")
    parser.add_argument("--ordered-delta-threshold", default="0.002,0.01,0.03", help="Ordered delta hard gate.")
    parser.add_argument("--execute-bigym-replans", action="store_true", help="Step recorded replanned actions in a real Bigym env.")
    parser.add_argument("--bigym-execute-only", action="store_true", help="Only run Bigym execution replay; skip offline grid scoring.")
    parser.add_argument("--bigym-output-jsonl", default=None, help="Optional per-trace Bigym execution JSONL path.")
    parser.add_argument("--bigym-env", default=HUMAN_ARM_CHUNK_DEFORM_ENV, help="Bigym env used for execution replay.")
    parser.add_argument("--bigym-demos", type=int, default=40, help="Demos used for Bigym action normalization.")
    parser.add_argument("--bigym-seed", type=int, default=1, help="Base seed for Bigym execution replay.")
    parser.add_argument("--bigym-limit-records", type=int, default=0, help="Limit Bigym-executed trace records; 0 means all.")
    parser.add_argument("--bigym-warmup-to-trace-step", action="store_true", help="Step zero actions until the trace step before replay.")
    parser.add_argument("--bigym-set-start-q", action=argparse.BooleanOptionalAction, default=True, help="Seed robot q from planned.q_seq[0] before replay.")
    parser.add_argument("--bigym-act-after-replan-steps", type=int, default=0, help="Run this many ACT policy steps after replayed recovery.")
    parser.add_argument("--bigym-act-snapshot", default=DEFAULT_BIGYM_ACT_SNAPSHOT, help="ACT snapshot used for post-replan handoff continuation.")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override for Bigym execution mode.")
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


def bool_value(value: Any, default: bool = False) -> bool:
    """Parse booleans from bool/int/string diagnostics."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def case_candidates(record: JsonDict) -> list[JsonDict]:
    """Return candidates for a case, falling back to the record itself."""
    candidates = record.get("candidates")
    if isinstance(candidates, list) and candidates:
        return [dict(c) for c in candidates if isinstance(c, Mapping)]
    candidate = dict(record)
    candidate.pop("candidates", None)
    return [candidate]


def recovery_cost(candidate: Mapping[str, Any], params: Mapping[str, float]) -> float:
    """Compute the weighted recovery objective used to select candidates."""
    safety = first_float(candidate, LOSS_ALIASES["safety"], 0.0) or 0.0
    action = first_float(candidate, LOSS_ALIASES["action"], 0.0) or 0.0
    smooth = first_float(candidate, LOSS_ALIASES["smooth"], 0.0) or 0.0
    rejoin = first_float(candidate, LOSS_ALIASES["rejoin"], 0.0) or 0.0
    ordered_pose = first_float(candidate, LOSS_ALIASES["ordered_pose"], 0.0) or 0.0
    ordered_delta = first_float(candidate, LOSS_ALIASES["ordered_delta"], 0.0) or 0.0
    direction = first_float(candidate, LOSS_ALIASES["direction"], 0.0) or 0.0
    progress = first_float(candidate, LOSS_ALIASES["task_progress"], 0.0) or 0.0
    return float(
        params["safety_weight"] * params["clearance_penalty_scale"] * safety
        + params["action_weight"] * action
        + params["smoothness_weight"] * smooth
        + params["rejoin_weight"] * rejoin
        + params["ordered_pose_weight"] * ordered_pose
        + params["ordered_delta_weight"] * ordered_delta
        + direction
        - params["task_progress_weight"] * progress
    )


def ordered_ok(candidate: Mapping[str, Any], params: Mapping[str, float]) -> bool:
    """Evaluate ordered-path diagnostics against swept thresholds."""
    if "recover_ordered_ok" in candidate:
        original = bool_value(candidate.get("recover_ordered_ok"), True)
        if original:
            return True
    pose = first_float(candidate, LOSS_ALIASES["ordered_pose"])
    delta = first_float(candidate, LOSS_ALIASES["ordered_delta"])
    if pose is None and delta is None:
        return True
    pose_ok = True if pose is None else pose <= params["ordered_pose_threshold"]
    delta_ok = True if delta is None else delta <= params["ordered_delta_threshold"]
    return bool(pose_ok and delta_ok)


def recovery_acceptance(candidate: Mapping[str, Any], params: Mapping[str, float]) -> tuple[bool, str]:
    """Apply recovery hard gates to a selected candidate."""
    immediate = first_float(candidate, CLEARANCE_ALIASES["immediate"])
    prefix = first_float(candidate, CLEARANCE_ALIASES["prefix"], immediate)
    path_min = first_float(candidate, CLEARANCE_ALIASES["path"], prefix)
    if immediate is None:
        return False, "missing_immediate_clearance"
    if immediate < params["recover_immediate_hard_clearance"]:
        return False, "immediate_unsafe"
    if prefix is None or prefix < params["recover_prefix_min_clearance"]:
        return False, "prefix_unsafe"
    if path_min is None or path_min < params["recover_path_min_clearance"]:
        return False, "path_unsafe"

    q_dist = first_float(candidate, ("q_rejoin_dist", "recover_q_rejoin_dist"))
    q_ok = bool_value(candidate.get("q_rejoin_ok"), True)
    if q_dist is not None:
        q_ok = q_dist <= params["q_rejoin_threshold"]
    if not q_ok:
        return False, "q_rejoin_failed"

    if params["require_qd_rejoin"]:
        qd_dist = first_float(candidate, ("qd_rejoin_dist", "recover_qd_rejoin_dist"))
        qd_ok = bool_value(candidate.get("qd_rejoin_ok"), True)
        if qd_dist is not None:
            qd_ok = qd_dist <= params["qd_rejoin_threshold"]
        if not qd_ok:
            return False, "qdot_rejoin_failed"

    direction_ok = bool_value(candidate.get("recover_direction_ok"), True)
    if not direction_ok:
        return False, "direction_alignment_failed"

    if params["require_ordered_path"] and not ordered_ok(candidate, params):
        return False, "ordered_path_failed"
    return True, "accepted"


def score_params(cases: Sequence[JsonDict], params: Mapping[str, float]) -> JsonDict:
    """Score one recovery parameter set across all cases."""
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
        selected = min(candidates, key=lambda item: recovery_cost(item, params))
        accepted_flag, reason = recovery_acceptance(selected, params)
        accepted += int(accepted_flag)
        reason_counts[reason] += 1
        selected_names[str(selected.get("candidate_name", selected.get("name", "candidate")))] += 1
        clearance = first_float(selected, CLEARANCE_ALIASES["path"])
        if clearance is not None:
            clearance_values.append(clearance)
        cost_values.append(recovery_cost(selected, params))

    accept_rate = float(accepted / total) if total else 0.0
    mean_clearance = float(np.mean(clearance_values)) if clearance_values else None
    mean_cost = float(np.mean(cost_values)) if cost_values else None
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


def _parameter_grid_parts(
    args: argparse.Namespace,
) -> tuple[list[str], list[list[float | bool]]]:
    """Build the recovery tuning grid axes from CLI values."""
    ordered_options = [not args.soften_ordered_path]
    if args.require_ordered_path:
        ordered_options = [True]
    keys = [
        "safety_weight",
        "clearance_penalty_scale",
        "rejoin_weight",
        "task_progress_weight",
        "ordered_pose_weight",
        "ordered_delta_weight",
        "action_weight",
        "smoothness_weight",
        "recover_path_min_clearance",
        "recover_immediate_hard_clearance",
        "recover_prefix_min_clearance",
        "q_rejoin_threshold",
        "qd_rejoin_threshold",
        "ordered_pose_threshold",
        "ordered_delta_threshold",
        "require_ordered_path",
        "require_qd_rejoin",
    ]
    value_lists: list[list[float | bool]] = [
        parse_grid(args.safety_weight),
        parse_grid(args.clearance_penalty_scale),
        parse_grid(args.rejoin_weight),
        parse_grid(args.task_progress_weight),
        parse_grid(args.ordered_pose_weight),
        parse_grid(args.ordered_delta_weight),
        parse_grid(args.action_weight),
        parse_grid(args.smoothness_weight),
        parse_grid(args.path_min_clearance),
        parse_grid(args.immediate_hard_clearance),
        parse_grid(args.prefix_min_clearance),
        parse_grid(args.q_rejoin_threshold),
        parse_grid(args.qd_rejoin_threshold),
        parse_grid(args.ordered_pose_threshold),
        parse_grid(args.ordered_delta_threshold),
        ordered_options,
        [bool(args.require_qd_rejoin)],
    ]
    return keys, value_lists


def parameter_grid_size(args: argparse.Namespace) -> int:
    """Return the number of recovery tuning parameter combinations."""
    _, value_lists = _parameter_grid_parts(args)
    total = 1
    for values in value_lists:
        total *= len(values)
    return int(total)


def parameter_grid(args: argparse.Namespace) -> Iterable[dict[str, float | bool]]:
    """Stream recovery tuning parameters instead of materializing the full grid."""
    keys, value_lists = _parameter_grid_parts(args)
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def _jsonable(value: Any) -> Any:
    """Convert NumPy containers into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _planned_arrays(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    """Extract planned actions and optional q traces from a recovery trace record."""
    planned = record.get("planned")
    if not isinstance(planned, Mapping):
        return None
    actions = np.asarray(planned.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] <= 0:
        return None
    q_seq = planned.get("q_seq")
    post_q_seq = planned.get("post_q_seq")
    q_arr = None if q_seq is None else np.asarray(q_seq, dtype=np.float32)
    post_q_arr = None if post_q_seq is None else np.asarray(post_q_seq, dtype=np.float32)
    if q_arr is not None and q_arr.ndim != 2:
        q_arr = None
    if post_q_arr is not None and post_q_arr.ndim != 2:
        post_q_arr = None
    return actions, q_arr, post_q_arr


def _make_bigym_execution_cfg(args: argparse.Namespace):
    """Build the same Bigym eval wrapper stack used by the full safety eval."""
    from robobase.safetyfilter.eval_utils.eval_utils import make_cfg

    overrides = list(DEFAULT_BIGYM_EXEC_OVERRIDES)
    overrides.extend(args.override or [])
    return make_cfg(
        SimpleNamespace(
            env=args.bigym_env,
            demos=int(args.bigym_demos),
            episodes=1,
            override=overrides,
        )
    )


def _reset_env(env, seed: int):
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _step_env(env, action: np.ndarray):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        return out
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), False, info
    raise ValueError(f"Unsupported env.step output from Bigym env: {type(out)!r}")


def _env_action_from_trace_action(action: np.ndarray, action_shape: tuple[int, ...], env) -> np.ndarray:
    """Adapt one planned action row to the eval env action-space shape."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if len(action_shape) == 1:
        if action.shape[0] != action_shape[0]:
            raise ValueError(f"Trace action dim {action.shape[0]} != env action dim {action_shape[0]}")
        env_action = action
    elif len(action_shape) == 2:
        if action.shape[0] != action_shape[1]:
            raise ValueError(f"Trace action dim {action.shape[0]} != env action dim {action_shape[1]}")
        env_action = np.repeat(action[None, :], action_shape[0], axis=0)
    else:
        raise ValueError(f"Unsupported env action shape: {action_shape}")

    low = getattr(getattr(env, "action_space", None), "low", None)
    high = getattr(getattr(env, "action_space", None), "high", None)
    if low is not None and high is not None:
        env_action = np.clip(env_action, low, high)
    return env_action.astype(np.float32)


def _set_bigym_h1_q(env, q: np.ndarray) -> bool:
    """Seed the Bigym robot q state from a recorded H1 q vector."""
    from robobase.safetyfilter.h1_state_bridge import TREE_JOINT_NAMES, get_bigym_mojo
    import mujoco

    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] < len(TREE_JOINT_NAMES):
        return False

    mojo = get_bigym_mojo(env)
    model = mojo.model
    data = mojo.data
    for idx, joint_name in enumerate(TREE_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return False
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = float(q[idx])
        data.qvel[dof_adr] = 0.0
    mojo.physics.forward()
    return True


def _q_rmse(env, target_q: np.ndarray | None) -> float | None:
    if target_q is None:
        return None
    try:
        from robobase.safetyfilter.h1_state_bridge import extract_h1_state

        q = extract_h1_state(env).q_full.astype(np.float32)
    except Exception:  # noqa: BLE001
        return None
    target = np.asarray(target_q, dtype=np.float32).reshape(-1)
    n = min(q.shape[0], target.shape[0])
    if n <= 0:
        return None
    diff = q[:n] - target[:n]
    return float(np.linalg.norm(diff) / math.sqrt(n))


def execute_bigym_replans(cases: Sequence[JsonDict], args: argparse.Namespace) -> JsonDict:
    """Execute recorded recovery replans through a real Bigym env."""
    from robobase.safetyfilter.eval_utils.eval_utils import (
        count_robot_human_contacts,
        extract_success,
        infer_env_action_shape,
        make_workspace_and_load_snapshot,
        normalise_env_action_shape,
        policy_action,
    )
    from robobase.safetyfilter.eval_utils.eval_video import _make_eval_env_with_normalization
    from robobase.safetyfilter.eval_utils.eval_environment import _reset_action_sequence_history

    ws = None
    if args.bigym_act_after_replan_steps > 0:
        ws = make_workspace_and_load_snapshot(
            _make_bigym_execution_cfg(args),
            Path(args.bigym_act_snapshot),
        )

    records = [record for record in cases if _planned_arrays(record) is not None]
    if args.bigym_limit_records > 0:
        records = records[: int(args.bigym_limit_records)]
    if not records:
        return {
            "stage": "bigym_replan_execution",
            "env": args.bigym_env,
            "total": 0,
            "executed": 0,
            "state_replay_exact": False,
            "note": "No records with planned.actions were found.",
        }

    cfg = _make_bigym_execution_cfg(args)
    env = _make_eval_env_with_normalization(cfg)
    action_shape = infer_env_action_shape(env)
    per_case: list[JsonDict] = []
    success_count = 0
    handover_success_count = 0
    terminated_count = 0
    truncated_count = 0
    contact_count = 0
    final_q_errors: list[float] = []
    step_q_errors: list[float] = []

    try:
        for index, record in enumerate(records):
            plan = _planned_arrays(record)
            if plan is None:
                continue
            actions, q_seq, post_q_seq = plan
            seed = int(args.bigym_seed) + int(record.get("episode") or 0)
            obs, _info = _reset_env(env, seed=seed)
            _reset_action_sequence_history(env)

            start_q_set = False
            if args.bigym_set_start_q and q_seq is not None and q_seq.shape[0] > 0:
                start_q_set = _set_bigym_h1_q(env, q_seq[0])

            if args.bigym_warmup_to_trace_step:
                zero = np.zeros(action_shape, dtype=np.float32)
                for _ in range(max(0, int(record.get("step") or 0))):
                    obs, _reward, terminated, truncated, _info = _step_env(env, zero)
                    if terminated or truncated:
                        break

            rewards: list[float] = []
            terminated = False
            truncated = False
            success = False
            step_errors: list[float] = []
            contacts: list[int] = []
            for local_idx, action in enumerate(actions):
                env_action = _env_action_from_trace_action(action, action_shape, env)
                obs, reward, terminated, truncated, info = _step_env(env, env_action)
                reward_scalar = float(np.asarray(reward).sum())
                rewards.append(reward_scalar)
                success = bool(success or extract_success(info, reward_scalar, bool(terminated)))
                contact_value = count_robot_human_contacts(env)
                if contact_value is not None:
                    contacts.append(int(contact_value))
                if post_q_seq is not None and local_idx < post_q_seq.shape[0]:
                    err = _q_rmse(env, post_q_seq[local_idx])
                    if err is not None:
                        step_errors.append(err)
                        step_q_errors.append(err)
                if terminated or truncated:
                    break

            act_rewards: list[float] = []
            handover_success = False
            if ws is not None and not (terminated or truncated):
                base_step = int(record.get("step") or 0) + len(rewards)
                for act_idx in range(max(0, int(args.bigym_act_after_replan_steps))):
                    act_action = policy_action(ws, obs, step=base_step + act_idx)
                    env_action = normalise_env_action_shape(act_action, action_shape)
                    obs, reward, terminated, truncated, info = _step_env(env, env_action)
                    reward_scalar = float(np.asarray(reward).sum())
                    act_rewards.append(reward_scalar)
                    handover_success = bool(
                        handover_success
                        or extract_success(info, reward_scalar, bool(terminated))
                    )
                    success = bool(success or handover_success)
                    contact_value = count_robot_human_contacts(env)
                    if contact_value is not None:
                        contacts.append(int(contact_value))
                    if terminated or truncated:
                        break

            final_target = None
            if post_q_seq is not None and post_q_seq.shape[0] > 0 and rewards:
                final_target = post_q_seq[min(len(rewards), post_q_seq.shape[0]) - 1]
            final_q_error = _q_rmse(env, final_target)
            if final_q_error is not None:
                final_q_errors.append(final_q_error)

            case_contacts = int(max(contacts)) if contacts else 0
            contact_count += int(case_contacts > 0)
            success_count += int(success)
            handover_success_count += int(handover_success)
            terminated_count += int(terminated)
            truncated_count += int(truncated)
            per_case.append(
                {
                    "case_index": index,
                    "episode": record.get("episode"),
                    "trace_step": record.get("step"),
                    "seed": seed,
                    "planned_action_count": int(actions.shape[0]),
                    "executed_action_count": int(len(rewards)),
                    "act_after_replan_steps_executed": int(len(act_rewards)),
                    "start_q_set": bool(start_q_set),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(success),
                    "handover_success": bool(handover_success),
                    "max_robot_human_contacts": case_contacts,
                    "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
                    "act_reward_sum": float(np.sum(act_rewards)) if act_rewards else 0.0,
                    "final_q_rmse": final_q_error,
                    "mean_step_q_rmse": float(np.mean(step_errors)) if step_errors else None,
                    "max_step_q_rmse": float(np.max(step_errors)) if step_errors else None,
                    "online_mpc_replan_accepted": record.get("mpc_recovery_replan_accepted"),
                    "online_mpc_replan_reject_reason": record.get("mpc_recovery_replan_reject_reason"),
                }
            )
    finally:
        env.close()

    if args.bigym_output_jsonl:
        out_path = Path(args.bigym_output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for item in per_case:
                handle.write(json.dumps(_jsonable(item), sort_keys=True) + "\n")

    total = len(per_case)
    return {
        "stage": "bigym_replan_execution",
        "env": args.bigym_env,
        "state_replay_exact": False,
        "start_q_seeded_from_trace": bool(args.bigym_set_start_q),
        "warmup_to_trace_step": bool(args.bigym_warmup_to_trace_step),
        "action_shape": list(action_shape),
        "total": total,
        "executed": total,
        "success_count": int(success_count),
        "success_rate": float(success_count / total) if total else 0.0,
        "handover_success_count": int(handover_success_count),
        "act_after_replan_steps": int(args.bigym_act_after_replan_steps),
        "terminated_count": int(terminated_count),
        "truncated_count": int(truncated_count),
        "contact_case_count": int(contact_count),
        "mean_final_q_rmse": float(np.mean(final_q_errors)) if final_q_errors else None,
        "max_final_q_rmse": float(np.max(final_q_errors)) if final_q_errors else None,
        "mean_step_q_rmse": float(np.mean(step_q_errors)) if step_q_errors else None,
        "max_step_q_rmse": float(np.max(step_q_errors)) if step_q_errors else None,
        "output_jsonl": args.bigym_output_jsonl,
        "note": "Executes recorded replanned actions in Bigym; full exact replay needs recorded MuJoCo state snapshots.",
    }

def yaml_snippet(params: Mapping[str, float | bool]) -> str:
    """Render an executable human-arm chunk-deform eval override."""
    live = STANDALONE_HUMAN_ARM_CHUNK_DEFORM_LIVE_THRESHOLDS
    return "\n".join(
        [
            "# @package _global_",
            "",
            "environment:",
            f"  env: {HUMAN_ARM_CHUNK_DEFORM_ENV}",
            "  safety_env: null",
            "",
            "safety_filter:",
            "  config: safechunk_deform.yaml",
            "  overrides:",
            "    # Keep live execution gates identical to standalone/default",
            "    # human-arm SafeChunk-Deform. The tuner only exports objective",
            "    # weights below; threshold sweeps stay offline diagnostics.",
            "    recoverable_deform:",
            f"      q_rejoin_threshold: {live['q_rejoin_threshold']}",
            f"      require_qd_rejoin: {str(bool(live['require_qd_rejoin'])).lower()}",
            f"      qd_rejoin_threshold: {live['qd_rejoin_threshold']}",
            f"      qd_rejoin_hard_threshold: {live['qd_rejoin_hard_threshold']}",
            "    safechunk_recover:",
            f"      safety_weight: {params['safety_weight']}",
            f"      clearance_penalty_scale: {params['clearance_penalty_scale']}",
            f"      rejoin_nominal_weight: {params['rejoin_weight']}",
            f"      task_progress_weight: {params['task_progress_weight']}",
            f"      action_deviation_weight: {params['action_weight']}",
            f"      smoothness_weight: {params['smoothness_weight']}",
            f"      ordered_pose_weight: {params['ordered_pose_weight']}",
            f"      ordered_delta_weight: {params['ordered_delta_weight']}",
            f"      ordered_pose_threshold: {live['ordered_pose_threshold']}",
            f"      ordered_delta_threshold: {live['ordered_delta_threshold']}",
            f"      require_ordered_path: {str(bool(live['require_ordered_path'])).lower()}",
            "    safechunk_recovery_corridor:",
            f"      recover_path_min_clearance: {live['recover_path_min_clearance']}",
            f"      recover_immediate_hard_clearance: {live['recover_immediate_hard_clearance']}",
            f"      recover_prefix_min_clearance: {live['recover_prefix_min_clearance']}",
            "",
        ]
    )


def main() -> None:
    """Run offline recovery tuning and optional Bigym replan execution."""
    args = parse_args()
    cases = load_records(args.input)
    if not cases:
        raise SystemExit("No recovery cases loaded.")

    bigym_summary = None
    if args.execute_bigym_replans:
        print(
            f"Loaded {len(cases)} recovery cases; executing replans in Bigym.",
            flush=True,
        )
        bigym_summary = execute_bigym_replans(cases, args)
        print(json.dumps(bigym_summary, indent=2, sort_keys=True))
        if args.bigym_execute_only:
            if args.output:
                Path(args.output).write_text(
                    json.dumps(bigym_summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return

    grid_size = parameter_grid_size(args)
    keep = max(1, int(args.top_k))
    print(
        f"Loaded {len(cases)} recovery cases; scoring {grid_size} parameter sets.",
        flush=True,
    )
    top: list[JsonDict] = []
    for index, params in enumerate(parameter_grid(args), start=1):
        result = score_params(cases, params)
        top.append(result)
        top.sort(key=lambda item: item["objective"], reverse=True)
        del top[keep:]
        if index == 1 or index == grid_size or index % 10000 == 0:
            best = top[0]
            print(
                "progress "
                f"{index}/{grid_size} "
                f"best_objective={best['objective']:.6f} "
                f"accept_rate={best['accept_rate']:.3f}",
                flush=True,
            )
    summary = {
        "stage": "recovery",
        "num_cases": len(cases),
        "num_grid_points": grid_size,
        "best": top[0],
        "top": top,
    }
    if bigym_summary is not None:
        summary["bigym_execution"] = bigym_summary
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_yaml:
        Path(args.write_yaml).write_text(yaml_snippet(top[0]["params"]), encoding="utf-8")


if __name__ == "__main__":
    main()
