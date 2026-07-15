from __future__ import annotations

import numpy as np

from robobase.safetyfilter.eval_utils.eval_metrics import _finite_float


def _trajectory_sample_step(sample: dict) -> int | None:
    if not isinstance(sample, dict):
        return None
    raw_step = sample.get("step")
    if raw_step is None:
        return None
    try:
        step = int(raw_step)
    except (TypeError, ValueError):
        return None
    return step


def _execution_sample_handle_distance(sample: dict) -> float | None:
    if not isinstance(sample, dict):
        return None
    distance = _finite_float(sample.get("ee_object_distance"))
    if distance is not None:
        return distance

    object_state = sample.get("object_state")
    if isinstance(object_state, dict):
        distance = _finite_float(object_state.get("ee_object_distance"))
        if distance is not None:
            return distance

    handle_pos = sample.get("handle_pos")
    if handle_pos is None and isinstance(object_state, dict):
        handle_pos = object_state.get("handle_pos")
    ee_pos = sample.get("ee_pos")
    if ee_pos is None or handle_pos is None:
        return None
    try:
        ee_array = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        handle_array = np.asarray(handle_pos, dtype=np.float64).reshape(3)
        if np.isfinite(ee_array).all() and np.isfinite(handle_array).all():
            return float(np.linalg.norm(ee_array - handle_array))
    except Exception:  # noqa: BLE001
        return None
    return None


def _closest_handle_distance_step(trajectory_samples: list[dict]) -> int | None:
    best_distance: float | None = None
    best_step: int | None = None
    for sample in trajectory_samples:
        distance = _execution_sample_handle_distance(sample)
        if distance is None:
            continue
        step = _trajectory_sample_step(sample)
        if step is None:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_step = step
    return best_step

def _truncate_trajectory_samples_after_step(samples: list[dict], cutoff_step: int | None) -> int:
    if cutoff_step is None:
        return 0
    kept_samples: list[dict] = []
    removed = 0
    for sample in samples:
        step = _trajectory_sample_step(sample)
        if step is None or step <= cutoff_step:
            kept_samples.append(sample)
        else:
            removed += 1
    if removed:
        samples[:] = kept_samples
    return removed

def _execution_sample_mode(sample: dict) -> str:
    mode = sample.get("execution_mode") or sample.get("diagnostic_step_mode") or sample.get("safety_mode")
    if mode in {None, "act", "pass_through", "path_consistent_brake_intended_step"}:
        return "policy"
    if bool(sample.get("brake_step")):
        return "braking"
    if bool(sample.get("deform_step")):
        return "deform"
    if bool(sample.get("recover_step")):
        return "recover"
    if bool(sample.get("intervention_active")):
        return str(mode) if str(mode) not in {"act", "pass_through"} else "intervention"
    return str(mode) if str(mode) not in {"act", "pass_through"} else "policy"

def _execution_sample_point(sample: dict):
    point = sample.get("ee_pos")
    if point is None:
        return None
    try:
        arr = np.asarray(point, dtype=np.float64).reshape(3)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _group_execution_samples_by_episode(executed_samples: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for sample in executed_samples:
        if not isinstance(sample, dict):
            continue
        point = _execution_sample_point(sample)
        if point is None:
            continue
        step = _trajectory_sample_step(sample)
        if step is None:
            step = 0
        try:
            episode = int(sample.get("episode", 0))
        except (TypeError, ValueError):
            episode = 0
        mode = _execution_sample_mode(sample)
        item = dict(sample)
        item["_point"] = point
        item["_mode"] = mode
        item["_group"] = "policy" if mode == "policy" else "intervention"
        item["step"] = step
        grouped.setdefault(episode, []).append(item)

    for items in grouped.values():
        items.sort(key=lambda item: int(item.get("step", 0)))
    return grouped
