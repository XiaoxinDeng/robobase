from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robobase.safetyfilter.eval_utils.eval_environment import (
    _H_ROBOT_PART_BLUE,
    _H_ROBOT_PART_RED,
    _collect_robot_part_geom_ids,
    _find_wrapped_env_with_attr,
)
from robobase.safetyfilter.eval_utils.eval_trajectory import (
    _execution_sample_point,
    _group_execution_samples_by_episode,
)


logger = logging.getLogger(__name__)


def _apply_robot_part_color_overrides(
    env,
    red_parts: set[str],
    blue_parts: set[str],
) -> Optional[dict[str, Any]]:
    if not red_parts and not blue_parts:
        return None

    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if base_env is None or not hasattr(base_env, "mojo"):
        return None
    try:
        model = base_env.mojo.physics.model.ptr
    except Exception:  # noqa: BLE001
        return None

    red_part_ids = _collect_robot_part_geom_ids(env, red_parts)
    blue_part_ids = _collect_robot_part_geom_ids(env, blue_parts)
    blue_only = sorted(set(blue_part_ids) - set(red_part_ids))
    if not red_part_ids and not blue_only:
        return None

    highlight_ids = sorted(set(red_part_ids) | set(blue_only))
    old_rgba = model.geom_rgba[highlight_ids].copy()
    if red_part_ids:
        model.geom_rgba[np.asarray(red_part_ids, dtype=np.int64)] = _H_ROBOT_PART_RED
    if blue_only:
        model.geom_rgba[np.asarray(blue_only, dtype=np.int64)] = _H_ROBOT_PART_BLUE
    try:
        base_env.mojo.physics.forward()
    except Exception:  # noqa: BLE001
        pass
    return {
        "geom_ids": np.asarray(highlight_ids, dtype=np.int64),
        "old_rgba": old_rgba,
    }


def _restore_robot_part_color_overrides(env, overrides: Optional[dict[str, Any]]) -> None:
    if overrides is None:
        return
    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if base_env is None or not hasattr(base_env, "mojo"):
        return
    try:
        model = base_env.mojo.physics.model.ptr
    except Exception:  # noqa: BLE001
        return

    geom_ids = np.asarray(overrides.get("geom_ids", []), dtype=np.int64)
    old_rgba = overrides.get("old_rgba")
    if geom_ids.size == 0 or old_rgba is None:
        return
    if len(geom_ids) != len(old_rgba):
        return
    model.geom_rgba[geom_ids] = old_rgba
    try:
        base_env.mojo.physics.forward()
    except Exception:  # noqa: BLE001
        pass


def _downsample(values, width):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.size <= width:
        return values
    x = np.linspace(0, values.size - 1, num=values.size)
    xp = np.linspace(0, values.size - 1, num=width)
    return np.interp(xp, x, values)

def _ascii_plot_lines(title, values, width=80, height=10):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []
    finite = np.isfinite(values)
    if not finite.any():
        return []
    values = np.where(finite, values, np.nan)
    values = _downsample(values, min(width, max(1, len(values))))
    min_v = float(np.nanmin(values))
    max_v = float(np.nanmax(values))
    if max_v == min_v:
        min_v -= 0.5
        max_v += 0.5
    span = max_v - min_v

    lines = [f"{title} (steps={len(values)}): {min_v:.4g} .. {max_v:.4g}"]
    for row in range(height, 0, -1):
        threshold = min_v + (row - 1) / (height - 1) * span
        line = "".join(
            "*" if (not np.isnan(v) and v >= threshold) else " "
            for v in values
        )
        lines.append(line)
    lines.append("-" * len(values))
    return lines

def _ascii_plot(title, values, width=80, height=10):
    for line in _ascii_plot_lines(title, values, width=width, height=height):
        print(line)

def _plot_episode_metrics(episode, episode_metrics):
    reward_values = [m.reward for m in episode_metrics]
    min_h_values = [float("nan") if m.min_h is None else m.min_h for m in episode_metrics]
    arm_delta_values = [m.arm_delta for m in episode_metrics]
    non_arm_delta_values = [m.non_arm_delta for m in episode_metrics]
    contact_values = [m.contact_count for m in episode_metrics]

    _ascii_plot(f"Episode {episode:03d} reward", reward_values)
    _ascii_plot(f"Episode {episode:03d} min_h", min_h_values)
    _ascii_plot(f"Episode {episode:03d} arm_delta", arm_delta_values)
    _ascii_plot(f"Episode {episode:03d} non_arm_delta", non_arm_delta_values)
    _ascii_plot(f"Episode {episode:03d} contact_count", contact_values)

def _jsonable_trace_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable_trace_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_trace_value(v) for v in value]
    try:
        arr = np.asarray(value)
        if arr.ndim > 0:
            if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
                return arr.tolist()
            return arr.astype(str).tolist()
        if np.issubdtype(arr.dtype, np.number):
            return arr.item()
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return str(value)

def _clearance_sequence_payload(safety_eval, horizon: int):
    if not isinstance(safety_eval, dict):
        return None
    clearances = safety_eval.get("min_clearances")
    if clearances is None:
        return None
    try:
        arr = np.asarray(clearances, dtype=np.float32).reshape(-1)
        if horizon > 0:
            arr = arr[:horizon]
        return arr.astype(float).tolist()
    except Exception:  # noqa: BLE001
        return None

def _trace_xyz_array(trace):
    if not isinstance(trace, dict):
        return None
    xyz = trace.get("ee_xyz")
    if xyz is None:
        return None
    try:
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    except Exception:  # noqa: BLE001
        return None
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    return arr if arr.size else None

def _record_uses_receding_first_action(record: dict, trace_payload=None) -> bool:
    horizon = record.get("executed_action_horizon")
    if horizon is not None:
        try:
            return int(horizon) == 1
        except Exception:  # noqa: BLE001
            return False
    if isinstance(trace_payload, dict):
        shape = trace_payload.get("action_shape")
        if isinstance(shape, list) and shape:
            try:
                return int(shape[0]) > 1
            except Exception:  # noqa: BLE001
                return False
    return False

def _trace_first_xyz(trace_payload):
    arr = _trace_xyz_array(trace_payload)
    if arr is None or arr.shape[0] < 1:
        return None
    first = np.asarray(arr[0], dtype=np.float64).reshape(3)
    if not np.isfinite(first).all():
        return None
    return first

def _trajectory_sample_segments(samples: list[dict], point_key: str, label_prefix: str):
    grouped: dict[int, list[tuple[int, list[float]]]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        point = sample.get(point_key)
        if point is None:
            continue
        try:
            arr = np.asarray(point, dtype=np.float64).reshape(3)
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(arr).all():
            continue
        episode = int(sample.get("episode", 0))
        step = int(sample.get("step", 0))
        grouped.setdefault(episode, []).append((step, arr.astype(float).tolist()))

    segments = []
    for episode, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item[0])
        if not items:
            continue
        segments.append({"label": f"{label_prefix} episode {episode:03d}", "points": [point for _step, point in items], "steps": [int(step) for step, _point in items]})
    return segments

def _trajectory_trace_segments(trace_records: list[dict], trace_name: str, label_prefix: str, max_events: int):
    selected = trace_records
    if max_events > 0 and len(selected) > max_events:
        selected = selected[-max_events:]

    segments = []
    receding_points: dict[int, list[tuple[int, list[float]]]] = {}
    for record in selected:
        if not isinstance(record, dict):
            continue
        traces = record.get("traces", {})
        if not isinstance(traces, dict):
            continue
        arr = _trace_xyz_array(traces.get(trace_name))
        if arr is None or arr.shape[0] < 1:
            continue
        episode = int(record.get("episode", 0))
        step = int(record.get("step", 0))
        if _record_uses_receding_first_action(record, traces.get(trace_name)):
            receding_points.setdefault(episode, []).append((step + 1, arr[0].astype(float).tolist()))
            continue
        segments.append(
            {
                "label": f"{label_prefix} e{episode:03d} step {step}",
                "points": arr.astype(float).tolist(),
                "episode": episode,
                "step": step,
                "steps": [int(step + offset) for offset in range(arr.shape[0])],
                "horizon_steps": [int(offset) for offset in range(arr.shape[0])],
            }
        )
    for episode, items in sorted(receding_points.items()):
        items = sorted(items, key=lambda item: item[0])
        if not items:
            continue
        segments.append(
            {
                "label": f"{label_prefix} executed first-action episode {episode:03d}",
                "points": [point for _step, point in items],
                "episode": episode,
                "step": int(items[0][0]),
                "steps": [int(step) for step, _point in items],
                "horizon_steps": [0 for _step, _point in items],
                "trace_source": "executed_first_action",
            }
        )
    return segments

def _braking_trajectory_segments(trace_records: list[dict], max_events: int):
    selected = trace_records
    if max_events > 0 and len(selected) > max_events:
        selected = selected[-max_events:]

    segments = []
    receding_points: dict[tuple[int, str], list[tuple[int, list[float]]]] = {}
    for record in selected:
        if not isinstance(record, dict):
            continue
        traces = record.get("traces", {})
        if not isinstance(traces, dict):
            continue
        source = (
            record.get("deformation_source")
            or record.get("retiming_source")
            or record.get("safety_mode")
            or record.get("mode")
        )
        trace = traces.get("braking")
        trace_source = "braking"
        if (
            source in {
                "path_consistent_brake",
                "path_consistent_brake_slowdown",
                "unverified_emergency_failsafe",
            }
            and isinstance(traces.get("generated"), dict)
        ):
            trace = traces.get("generated")
            trace_source = "generated_safe_first_action"
        arr = _trace_xyz_array(trace)
        if arr is None or arr.shape[0] < 1:
            continue
        episode = int(record.get("episode", 0))
        step = int(record.get("step", 0))
        if _record_uses_receding_first_action(record, trace):
            receding_points.setdefault((episode, trace_source), []).append((step + 1, arr[0].astype(float).tolist()))
            continue
        segments.append(
            {
                "label": f"braking e{episode:03d} step {step}",
                "points": arr.astype(float).tolist(),
                "episode": episode,
                "step": step,
                "steps": [int(step + offset) for offset in range(arr.shape[0])],
                "horizon_steps": [int(offset) for offset in range(arr.shape[0])],
                "trace_source": trace_source,
            }
        )
    for (episode, trace_source), items in sorted(receding_points.items()):
        items = sorted(items, key=lambda item: item[0])
        if not items:
            continue
        segments.append(
            {
                "label": f"braking executed first-action episode {episode:03d}",
                "points": [point for _step, point in items],
                "episode": episode,
                "step": int(items[0][0]),
                "steps": [int(step) for step, _point in items],
                "horizon_steps": [0 for _step, _point in items],
                "trace_source": trace_source,
            }
        )
    return segments

def _segment_timestep_labels(segment: dict, count: int):
    raw_steps = segment.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    raw_horizon_steps = segment.get("horizon_steps")
    horizon_steps = raw_horizon_steps if isinstance(raw_horizon_steps, list) else []
    labels = []
    for idx in range(count):
        parts = []
        if idx < len(steps):
            parts.append(f"timestep {steps[idx]}")
        elif segment.get("step") is not None:
            parts.append(f"timestep {int(segment['step']) + idx}")
        else:
            parts.append(f"timestep {idx}")
        if idx < len(horizon_steps):
            parts.append(f"horizon {horizon_steps[idx]}")
        labels.append(", ".join(parts))
    return labels

def _execution_marker_segments(executed_samples: list[dict], group: str, label: str, mode_filter: str | None = None):
    segments = []
    for episode, items in sorted(_group_execution_samples_by_episode(executed_samples).items()):
        selected = [
            item
            for item in items
            if item.get("_group") == group
            and (mode_filter is None or str(item.get("_mode")) == str(mode_filter))
        ]
        if not selected:
            continue
        segments.append(
            {
                "label": f"{label} episode {episode:03d}",
                "points": [item["_point"].astype(float).tolist() for item in selected],
                "steps": [int(item.get("step", 0)) for item in selected],
                "execution_modes": [str(item.get("_mode")) for item in selected],
            }
        )
    return segments

def _execution_pair_segments(executed_samples: list[dict], segment_class: str, mode_filter: str | None = None):
    segments = []
    for episode, items in sorted(_group_execution_samples_by_episode(executed_samples).items()):
        for prev, curr in zip(items, items[1:]):
            prev_group = str(prev.get("_group"))
            curr_group = str(curr.get("_group"))
            prev_mode = str(prev.get("_mode"))
            curr_mode = str(curr.get("_mode"))
            is_policy = prev_group == curr_group == "policy"
            is_intervention = prev_group == curr_group == "intervention" and prev_mode == curr_mode
            is_transition = prev_group != curr_group or (prev_group == curr_group == "intervention" and prev_mode != curr_mode)
            if mode_filter is not None and (prev_mode != str(mode_filter) or curr_mode != str(mode_filter)):
                continue
            if segment_class == "policy" and not is_policy:
                continue
            if segment_class == "intervention" and not is_intervention:
                continue
            if segment_class == "transition" and not is_transition:
                continue
            segments.append(
                {
                    "label": (
                        f"actual execution {prev_mode}->{curr_mode} e{episode:03d} "
                        f"steps {int(prev.get('step', 0))}-{int(curr.get('step', 0))}"
                    ),
                    "points": [prev["_point"].astype(float).tolist(), curr["_point"].astype(float).tolist()],
                    "episode": episode,
                    "step": int(prev.get("step", 0)),
                    "steps": [int(prev.get("step", 0)), int(curr.get("step", 0))],
                    "execution_modes": [prev_mode, curr_mode],
                }
            )
    return segments


def _box_edge_segments_world(center, size):
    try:
        center_arr = np.asarray(center, dtype=np.float64).reshape(3)
        half_size = 0.5 * np.asarray(size, dtype=np.float64).reshape(3)
    except Exception:  # noqa: BLE001
        return []
    if not np.isfinite(center_arr).all() or not np.isfinite(half_size).all():
        return []

    def point(dx, dy, dz):
        signs = np.asarray([dx, dy, dz], dtype=np.float64)
        return (center_arr + signs * half_size).astype(float).tolist()

    edge_pairs = []
    for dx in (-1, 1):
        for dy in (-1, 1):
            edge_pairs.append(((dx, dy, -1), (dx, dy, 1)))
    for dx in (-1, 1):
        for dz in (-1, 1):
            edge_pairs.append(((dx, -1, dz), (dx, 1, dz)))
    for dy in (-1, 1):
        for dz in (-1, 1):
            edge_pairs.append(((-1, dy, dz), (1, dy, dz)))
    return [[point(*a), point(*b)] for a, b in edge_pairs]


def _drawer_scene_geometry_from_cabinet_xml(handle_pos, open_distance, open_fraction):
    try:
        handle_arr = np.asarray(handle_pos, dtype=np.float64).reshape(3)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(handle_arr).all():
        return None

    distance = 0.0
    try:
        distance = float(open_distance)
    except (TypeError, ValueError):
        try:
            distance = 0.35 * float(open_fraction)
        except (TypeError, ValueError):
            distance = 0.0
    if not np.isfinite(distance):
        distance = 0.0

    open_axis = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    origin = handle_arr - open_axis * distance
    cabinet_segments = _box_edge_segments_world([0.0, 0.20, -0.07], [0.82, 0.48, 0.72])
    drawer_segments = _box_edge_segments_world([0.0, 0.12, 0.0], [0.58, 0.30, 0.18])
    front_segments = _box_edge_segments_world([0.0, 0.0, 0.0], [0.64, 0.025, 0.24])
    handle_segments = [[[-0.16, -0.035, 0.0], [0.16, -0.035, 0.0]]]
    return {
        "absolute": False,
        "origin": origin.astype(float).tolist(),
        "handle_pos": handle_arr.astype(float).tolist(),
        "open_axis": open_axis.astype(float).tolist(),
        "default_open": float(distance),
        "open_fraction": None if open_fraction is None else float(open_fraction),
        "cabinet": cabinet_segments,
        "drawer": drawer_segments + front_segments,
        "handle": handle_segments,
        "handleDisplacement": [],
        "source": "cabinet_xml_fallback",
    }

def _drawer_reference_from_samples(executed_samples: list[dict]):
    handle_pos = None
    handle_trajectory = []
    open_distance_trajectory = []
    open_distance = None
    open_fraction = None
    scene_geometry = None

    def _point_or_none(value):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(3)
        except Exception:  # noqa: BLE001
            return None
        return arr if np.isfinite(arr).all() else None

    for sample in executed_samples:
        if not isinstance(sample, dict):
            continue
        candidate = sample.get("handle_pos")
        object_state = sample.get("object_state")
        if candidate is None and isinstance(object_state, dict):
            candidate = object_state.get("handle_pos")
        if candidate is not None:
            arr = _point_or_none(candidate)
            if arr is not None:
                if handle_pos is None:
                    handle_pos = arr
                step = int(sample.get("step", len(handle_trajectory)))
                if not handle_trajectory or np.linalg.norm(
                    arr - np.asarray(handle_trajectory[-1]["point"], dtype=np.float64)
                ) > 1e-6:
                    handle_trajectory.append({"step": step, "point": arr.astype(float).tolist()})

        if scene_geometry is None:
            value = sample.get("drawer_scene_geometry")
            if value is None and isinstance(object_state, dict):
                value = object_state.get("drawer_scene_geometry")
            if isinstance(value, dict) and (value.get("cabinet") or value.get("drawer") or value.get("handle")):
                scene_geometry = value

        value = sample.get("drawer_open_distance")
        if value is None and isinstance(object_state, dict):
            value = object_state.get("drawer_open_distance")
        if value is not None:
            try:
                distance_value = float(value)
                if np.isfinite(distance_value):
                    if open_distance is None:
                        open_distance = distance_value
                    step = int(sample.get("step", len(open_distance_trajectory)))
                    if (
                        not open_distance_trajectory
                        or abs(distance_value - float(open_distance_trajectory[-1]["distance"])) > 1e-6
                    ):
                        open_distance_trajectory.append({"step": step, "distance": distance_value})
            except (TypeError, ValueError):
                pass
        if open_fraction is None:
            value = sample.get("drawer_open_fraction")
            if value is None and isinstance(object_state, dict):
                value = object_state.get("drawer_open_fraction")
            if value is not None:
                try:
                    open_fraction = float(value)
                except (TypeError, ValueError):
                    pass
    if scene_geometry is not None and handle_pos is not None:
        try:
            axis_arr = np.asarray(scene_geometry.get("open_axis"), dtype=np.float64).reshape(3)
            axis_norm = float(np.linalg.norm(axis_arr))
        except Exception:  # noqa: BLE001
            axis_norm = 0.0
        if axis_norm <= 1e-9 or not np.isfinite(axis_norm):
            xml_geometry = _drawer_scene_geometry_from_cabinet_xml(handle_pos, open_distance, open_fraction)
            if xml_geometry is not None and xml_geometry.get("open_axis") is not None:
                scene_geometry = dict(scene_geometry)
                scene_geometry["open_axis"] = xml_geometry.get("open_axis")

    scene_open_axis = None
    if scene_geometry is not None:
        try:
            axis_arr = np.asarray(scene_geometry.get("open_axis"), dtype=np.float64).reshape(3)
            axis_norm = float(np.linalg.norm(axis_arr))
            if axis_norm > 1e-9 and np.isfinite(axis_norm):
                scene_open_axis = axis_arr / axis_norm
        except Exception:  # noqa: BLE001
            scene_open_axis = None
    if len(handle_trajectory) < 2 and handle_pos is not None and scene_open_axis is not None and open_distance_trajectory:
        base_distance = float(open_distance_trajectory[0]["distance"])
        handle_trajectory = [
            {
                "step": int(item["step"]),
                "point": (handle_pos + scene_open_axis * (float(item["distance"]) - base_distance)).astype(float).tolist(),
            }
            for item in open_distance_trajectory
        ]

    if scene_geometry is not None:
        handle_segments = scene_geometry.get("handle") or []
        if not handle_segments and handle_pos is not None:
            handle_segments = _box_edge_segments_world(handle_pos, [0.025, 0.025, 0.025])
        return {
            "absolute": True,
            "origin": [0.0, 0.0, 0.0],
            "handle_pos": None if handle_pos is None else handle_pos.astype(float).tolist(),
            "open_axis": [0.0, 0.0, 0.0] if scene_open_axis is None else scene_open_axis.astype(float).tolist(),
            "default_open": 0.0 if open_distance is None else float(open_distance),
            "open_fraction": None if open_fraction is None else float(open_fraction),
            "cabinet": scene_geometry.get("cabinet") or [],
            "drawer": scene_geometry.get("drawer") or [],
            "handle": handle_segments,
            "handleDisplacement": handle_trajectory,
            "source": scene_geometry.get("source") or "mujoco_geoms",
        }

    if handle_pos is None:
        return None

    open_distance = 0.0 if open_distance is None or not np.isfinite(open_distance) else float(open_distance)
    xml_geometry = _drawer_scene_geometry_from_cabinet_xml(handle_pos, open_distance, open_fraction)
    if xml_geometry is not None:
        if handle_trajectory and not xml_geometry.get("handleDisplacement"):
            xml_geometry = dict(xml_geometry)
            xml_geometry["handleDisplacement"] = handle_trajectory
        return xml_geometry

    open_axis = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    if len(handle_trajectory) < 2 and open_distance_trajectory:
        base_distance = float(open_distance_trajectory[0]["distance"])
        handle_trajectory = [
            {
                "step": int(item["step"]),
                "point": (handle_pos + open_axis * (float(item["distance"]) - base_distance)).astype(float).tolist(),
            }
            for item in open_distance_trajectory
        ]
    origin = handle_pos - open_axis * open_distance

    def edges(center, size):
        cx, cy, cz = [float(v) for v in center]
        sx, sy, sz = [0.5 * float(v) for v in size]
        corners = [
            [cx + dx * sx, cy + dy * sy, cz + dz * sz]
            for dx in (-1, 1)
            for dy in (-1, 1)
            for dz in (-1, 1)
        ]
        idx = {(dx, dy, dz): i for i, (dx, dy, dz) in enumerate((
            (dx, dy, dz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)
        ))}
        pairs = []
        for dx in (-1, 1):
            for dy in (-1, 1):
                pairs.append((idx[(dx, dy, -1)], idx[(dx, dy, 1)]))
            for dz in (-1, 1):
                pairs.append((idx[(dx, -1, dz)], idx[(dx, 1, dz)]))
        for dy in (-1, 1):
            for dz in (-1, 1):
                pairs.append((idx[(-1, dy, dz)], idx[(1, dy, dz)]))
        return [[corners[a], corners[b]] for a, b in pairs]

    cabinet_segments = edges([0.0, 0.20, -0.07], [0.82, 0.48, 0.72])
    drawer_segments = edges([0.0, 0.12, 0.0], [0.58, 0.30, 0.18])
    front_segments = edges([0.0, 0.0, 0.0], [0.64, 0.025, 0.24])
    handle_segments = [[[-0.16, -0.035, 0.0], [0.16, -0.035, 0.0]]]
    return {
        "absolute": False,
        "origin": origin.astype(float).tolist(),
        "handle_pos": handle_pos.astype(float).tolist(),
        "open_axis": open_axis.astype(float).tolist(),
        "default_open": float(open_distance),
        "open_fraction": None if open_fraction is None else float(open_fraction),
        "cabinet": cabinet_segments,
        "drawer": drawer_segments + front_segments,
        "handle": handle_segments,
        "handleDisplacement": handle_trajectory,
        "source": "handle_fallback",
    }

def _add_plotly_segments(fig, go, segments, *, name: str, color: str, width: int, dash=None, line: bool = True, markers: bool = True):
    first = True
    for segment in segments:
        points = np.asarray(segment.get("points", []), dtype=np.float64).reshape(-1, 3)
        if points.size == 0:
            continue
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.size == 0:
            continue
        labels = _segment_timestep_labels(segment, points.shape[0])
        line_style = {"color": color, "width": width}
        if dash:
            line_style["dash"] = "dash"
        if line:
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="lines",
                    name=name if first else segment.get("label", name),
                    legendgroup=name,
                    showlegend=first,
                    line=line_style,
                    text=[segment.get("label", name)] * points.shape[0],
                    customdata=labels,
                    hovertemplate="%{text}<br>%{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
                )
            )
        if markers:
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    name=f"{name} timestep dots" if first else f"{segment.get('label', name)} timestep dots",
                    legendgroup=f"{name} timestep dots",
                    showlegend=first,
                    marker={"color": color, "size": max(2, width + 1)},
                    text=[segment.get("label", name)] * points.shape[0],
                    customdata=labels,
                    hovertemplate="%{text}<br>%{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
                )
            )
        first = False

def _executed_policy_segments_from_traces(trace_records: list[dict], executed_samples: list[dict], max_events: int):
    segments = _trajectory_sample_segments(executed_samples, "ee_pos", "executed policy")
    if segments:
        return segments
    return _braking_trajectory_segments(trace_records, max_events)

def _first_trace_point(trace_payload):
    arr = _trace_xyz_array(trace_payload)
    if arr is None or arr.shape[0] == 0:
        return None
    return np.asarray(arr[0], dtype=np.float64).reshape(3)

def _trajectory_frame_diagnostics(trace_records: list[dict], executed_samples: list[dict]):
    executed_by_step = {}
    for sample in executed_samples:
        if not isinstance(sample, dict):
            continue
        point = _execution_sample_point(sample)
        if point is None:
            continue
        executed_by_step[(int(sample.get("episode", 0)), int(sample.get("step", 0)))] = point

    comparisons = []
    for record in trace_records:
        if not isinstance(record, dict):
            continue
        episode = int(record.get("episode", 0))
        step = int(record.get("step", 0))
        anchor = record.get("policy_anchor_sample")
        if isinstance(anchor, dict) and anchor.get("ee_pos") is not None:
            try:
                src = np.asarray(anchor.get("ee_pos"), dtype=np.float64).reshape(3)
                ref = executed_by_step.get((episode, int(anchor.get("step", step))))
                if ref is not None and np.isfinite(src).all():
                    diff = ref - src
                    comparisons.append({
                        "label": "policy anchor vs MuJoCo executed same timestep",
                        "source": str(anchor.get("source")),
                        "episode": episode,
                        "step": int(anchor.get("step", step)),
                        "norm": float(np.linalg.norm(diff)),
                        "diff": diff.astype(float).tolist(),
                    })
            except Exception:  # noqa: BLE001
                pass
        traces = record.get("traces")
        if not isinstance(traces, dict):
            continue
        for name in ("generated", "braking", "deformed", "recovery"):
            point = _first_trace_point(traces.get(name))
            ref = executed_by_step.get((episode, step + 1))
            if point is None or ref is None:
                continue
            diff = ref - point
            comparisons.append({
                "label": f"{name} first-action FK vs MuJoCo executed next timestep",
                "source": str((traces.get(name) or {}).get("frame", "unknown")),
                "episode": episode,
                "step": int(step + 1),
                "norm": float(np.linalg.norm(diff)),
                "diff": diff.astype(float).tolist(),
            })

    if not comparisons:
        return {"count": 0, "note": "No comparable safety-model FK and MuJoCo execution samples were available."}
    norms = np.asarray([c["norm"] for c in comparisons], dtype=np.float64)
    worst = comparisons[int(np.argmax(norms))]
    return {
        "count": int(len(comparisons)),
        "mean_norm": float(np.mean(norms)),
        "max_norm": float(np.max(norms)),
        "worst": worst,
        "note": "Actual execution/human/drawer layers are MuJoCo world-frame samples. Planned objective layers are safety-model FK world estimates and may not align with MuJoCo.",
    }

def _viewer_segments(segments, *, line: bool = True, markers: bool = True):
    return [
        {**segment, "line": bool(line), "markers": bool(markers)}
        for segment in segments
    ]

def _execution_trajectory_layer(
    *,
    name: str,
    color: str,
    width: int,
    line_segments: list[dict],
    marker_segments: list[dict],
):
    return {
        "name": name,
        "legend": name,
        "color": color,
        "width": width,
        "segments": _viewer_segments(line_segments, line=True, markers=False)
        + _viewer_segments(marker_segments, line=False, markers=True),
    }

def _chunk_trajectory_viewer_layers(trace_records, human_samples, executed_samples, max_events):
    return [
        _execution_trajectory_layer(
            name="policy trajectory",
            color="#111827",
            width=4,
            line_segments=_execution_pair_segments(executed_samples, "policy"),
            marker_segments=_execution_marker_segments(executed_samples, "policy", "policy pose"),
        ),
        _execution_trajectory_layer(
            name="braking trajectory",
            color="#f59e0b",
            width=4,
            line_segments=_execution_pair_segments(executed_samples, "intervention", "braking"),
            marker_segments=_execution_marker_segments(executed_samples, "intervention", "braking pose", "braking"),
        ),
        _execution_trajectory_layer(
            name="deformed trajectory",
            color="#2563eb",
            width=4,
            line_segments=_execution_pair_segments(executed_samples, "intervention", "deform"),
            marker_segments=_execution_marker_segments(executed_samples, "intervention", "deformed pose", "deform"),
        ),
        _execution_trajectory_layer(
            name="recovered trajectory",
            color="#16a34a",
            width=4,
            line_segments=_execution_pair_segments(executed_samples, "intervention", "recover"),
            marker_segments=_execution_marker_segments(executed_samples, "intervention", "recovered pose", "recover"),
        ),
        {"name": "mode transition trajectory", "legend": "mode transition trajectory", "color": "#a855f7", "width": 3, "markers": False, "segments": _execution_pair_segments(executed_samples, "transition")},
        {"name": "human arm wrist joint trajectory", "legend": "human arm wrist trajectory", "color": "#d62728", "width": 4, "segments": _trajectory_sample_segments(human_samples, "wrist_pos", "human wrist")},
        {"name": "planned braking first-action objective (safety-model FK)", "legend": "braking trajectory (model FK diagnostic)", "color": "#fbbf24", "width": 2, "dash": [5, 5], "markers": False, "visible": False, "segments": _braking_trajectory_segments(trace_records, max_events)},
        {"name": "planned deformed first-action objective (safety-model FK)", "legend": "deformed trajectory (model FK diagnostic)", "color": "#2563eb", "width": 2, "dash": [5, 5], "markers": False, "visible": False, "segments": _trajectory_trace_segments(trace_records, "deformed", "deformed", max_events)},
        {"name": "planned recovered first-action objective (safety-model FK)", "legend": "recovered trajectory (model FK diagnostic)", "color": "#16a34a", "width": 2, "dash": [5, 5], "markers": False, "visible": False, "segments": _trajectory_trace_segments(trace_records, "recovery", "recovered", max_events)},
    ]

def _save_chunk_trajectory_canvas_viewer(path: Path, title: str, trace_records: list[dict], human_samples: list[dict], executed_samples: list[dict], max_events: int):
    layers = _chunk_trajectory_viewer_layers(trace_records, human_samples, executed_samples, max_events)
    if not any(segment.get("points") for layer in layers for segment in layer.get("segments", [])):
        return None
    payload = {
        "title": title,
        "layers": layers,
        "drawerReference": _drawer_reference_from_samples(executed_samples),
        "frameDiagnostics": _trajectory_frame_diagnostics(trace_records, executed_samples),
        "safetyGeometry": None,
    }
    payload_json = json.dumps(_jsonable_trace_value(payload), separators=(",", ":"))
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeChunk 3D Trajectory Viewer</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;font-family:Arial,sans-serif;background:transparent;color:#151515}
canvas{position:fixed;inset:0;width:100vw;height:100vh;background:transparent}
.panel{position:fixed;left:14px;top:14px;width:min(430px,calc(100vw - 28px));max-height:calc(100vh - 28px);overflow:auto;background:rgba(255,255,255,.92);border:1px solid #d6d2c8;border-radius:8px;padding:12px;box-shadow:0 10px 30px rgba(0,0,0,.14)}
.panel[hidden]{display:none}.panel-tab{position:fixed;left:14px;top:14px;z-index:2}
.legend{position:fixed;right:14px;bottom:14px;max-width:min(340px,calc(100vw - 28px));background:rgba(255,255,255,.84);border:1px solid #d6d2c8;border-radius:8px;padding:8px 10px;font-size:12px;color:#1f2937;box-shadow:0 8px 24px rgba(0,0,0,.12);pointer-events:none}
.legend[hidden]{display:none}.legend-row{display:grid;grid-template-columns:24px 1fr;gap:7px;align-items:center;margin:3px 0}.legend-line{height:0;border-top:2px solid currentColor}
h1{font-size:16px;margin:0 0 8px}.hint{font-size:12px;color:#555;line-height:1.35;margin-bottom:8px}
.dot-toggle{display:flex;align-items:center;gap:8px;border-top:1px solid #ebe7dd;margin-top:2px;padding:8px 0 4px;font-size:13px}
.layer{border-top:1px solid #ebe7dd;padding:7px 0;font-size:13px}.layer-main{display:grid;grid-template-columns:18px 18px 1fr auto;gap:8px;align-items:center}.layer-style{display:grid;grid-template-columns:52px 1fr 42px;gap:8px;align-items:center;margin:6px 0 0 44px}.layer-style input[type=color]{width:100%;height:24px;padding:0;border:1px solid #c8c3b8;background:white}.layer-style input[type=range]{width:100%}
.sw{width:14px;height:14px;border-radius:50%;border:1px solid rgba(0,0,0,.24)}
button{border:1px solid #c8c3b8;background:white;border-radius:6px;padding:6px 9px;margin:2px 4px 8px 0;cursor:pointer}.count{font-size:12px;color:#666}
.view-presets{display:flex;align-items:center;gap:6px;flex-wrap:wrap;border-top:1px solid #ebe7dd;margin-top:2px;padding:8px 0 4px;font-size:13px}.view-presets span{color:#555;margin-right:2px}.view-presets button{margin:0;padding:4px 8px}
.drawer-controls{border-top:1px solid #ebe7dd;margin-top:4px;padding:8px 0;font-size:13px}.drawer-controls h2{font-size:13px;margin:0 0 6px}.drawer-control{display:grid;grid-template-columns:58px 1fr 48px;gap:8px;align-items:center;margin:5px 0}.drawer-control input{width:100%}
.frame-diagnostics{border-top:1px solid #ebe7dd;margin-top:4px;padding:8px 0;font-size:12px;color:#374151;line-height:1.35}.frame-diagnostics strong{color:#111827}
</style>
</head>
<body>
<canvas id="view"></canvas>
<button id="showPanel" class="panel-tab" hidden>Show panel</button>
<div id="legend" class="legend" hidden></div>
<section id="panel" class="panel">
<h1 id="title"></h1>
<div class="hint">Choose Orbit or Pan, then left-drag. Mouse wheel zooms. Arrow keys pan; A/D and W/S orbit. Controls are remembered in this browser tab even if VS Code reloads the HTML.</div>
<button id="reset">Reset view</button><button id="shot">Download screenshot</button><button id="legendShot">Download legend</button><button id="hidePanel">Hide panel</button><button id="clearSettings">Reset controls</button>
<label class="dot-toggle"><input id="dots" type="checkbox" checked> Timestep dots</label>
<label class="dot-toggle"><input id="pauseRedraw" type="checkbox"> Pause redraw while editing</label>
<label class="dot-toggle">Drag mode <select id="dragMode"><option value="orbit">Orbit</option><option value="pan">Pan</option></select></label>
<div id="viewPresets" class="view-presets"><span>View</span><button data-view="iso">ISO</button><button data-view="top">Top</button><button data-view="left">Left</button><button data-view="right">Right</button><button data-view="front">Front</button></div>
<div id="axisControls" class="drawer-controls"></div>
<div id="fadeControls"></div>
<div id="frameDiagnostics" class="frame-diagnostics" hidden></div>
<div id="hModelControls" class="drawer-controls"></div>
<div id="drawerControls" class="drawer-controls"></div>
<div id="layers"></div>
</section>
<script>
const DATA=__DATA__;
const canvas=document.getElementById("view"),ctx=canvas.getContext("2d"),vis=new Map(),styles=new Map(),drawerRef=DATA.drawerReference||null,hGeom=DATA.safetyGeometry||null;
const STORE_KEY="safechunk_trajectory_viewer:shared",LEGACY_STORE_KEY="safechunk_trajectory_viewer:"+location.pathname;
let savedState={};try{savedState=JSON.parse(localStorage.getItem(STORE_KEY)||localStorage.getItem(LEGACY_STORE_KEY)||"{}")}catch(_e){savedState={}}
const hEventCount=hGeom&&Array.isArray(hGeom.events)?hGeom.events.length:0;
const hasHandleDisplacement=!!(drawerRef&&Array.isArray(drawerRef.handleDisplacement)&&drawerRef.handleDisplacement.length);
let showDots=savedState.showDots!==undefined?!!savedState.showDots:true,drawDrawer=savedState.drawDrawer!==undefined?!!savedState.drawDrawer:!!drawerRef,drawHandleDisplacement=savedState.drawHandleDisplacement!==undefined?!!savedState.drawHandleDisplacement:hasHandleDisplacement,drawHModel=savedState.drawHModel!==undefined?!!savedState.drawHModel:false,drawHHuman=savedState.drawHHuman!==undefined?!!savedState.drawHHuman:true,drawHRobot=savedState.drawHRobot!==undefined?!!savedState.drawHRobot:true,drawHAllEvents=savedState.drawHAllEvents!==undefined?!!savedState.drawHAllEvents:false,hModelStep=savedState.hModelStep!==undefined?Number(savedState.hModelStep):0,lineAlphaMin=savedState.lineAlphaMin!==undefined?Number(savedState.lineAlphaMin):.08,lineAlphaMax=savedState.lineAlphaMax!==undefined?Number(savedState.lineAlphaMax):1,trajectoryLineWidth=savedState.trajectoryLineWidth!==undefined?Number(savedState.trajectoryLineWidth):.5,trajectoryDotSize=savedState.trajectoryDotSize!==undefined?Number(savedState.trajectoryDotSize):1,policyDotSize=savedState.policyDotSize!==undefined?Number(savedState.policyDotSize):trajectoryDotSize,hModelAlpha=savedState.hModelAlpha!==undefined?Number(savedState.hModelAlpha):.28,drawerOffset=Array.isArray(savedState.drawerOffset)?savedState.drawerOffset.slice(0,3):[0,0,0],drawerOpen=savedState.drawerOpen!==undefined?Number(savedState.drawerOpen):(drawerRef?drawerRef.default_open:0),axisOrigin=Array.isArray(savedState.axisOrigin)?savedState.axisOrigin.slice(0,3):[0,0,0],axisLengthValue=savedState.axisLengthValue!==undefined?Number(savedState.axisLengthValue):.5,showAxes=savedState.showAxes!==undefined?!!savedState.showAxes:true,dragMode=savedState.dragMode||"orbit",panelHidden=!!savedState.panelHidden,pauseRedraw=false,yaw=-.85,pitch=.48,dist=1,target=[0,0,0],drag=false,pan=false,lx=0,ly=0,restoredCamera=false;
if(!hasHandleDisplacement)drawHandleDisplacement=false;
const savedCamera=savedState.camera||{};if(Array.isArray(savedCamera.target)&&savedCamera.target.length===3&&Number.isFinite(Number(savedCamera.yaw))&&Number.isFinite(Number(savedCamera.pitch))&&Number.isFinite(Number(savedCamera.dist))){yaw=Number(savedCamera.yaw);pitch=Number(savedCamera.pitch);dist=Math.max(.03,Number(savedCamera.dist));target=savedCamera.target.map(Number);if(target.every(Number.isFinite))restoredCamera=true;else target=[0,0,0]}
lineAlphaMin=Math.max(.03,Math.min(.95,Number.isFinite(lineAlphaMin)?lineAlphaMin:.08));lineAlphaMax=Math.max(lineAlphaMin+.01,Math.min(1,Number.isFinite(lineAlphaMax)?lineAlphaMax:1));trajectoryLineWidth=Math.max(.1,Math.min(12,Number.isFinite(trajectoryLineWidth)?trajectoryLineWidth:.5));trajectoryDotSize=Math.max(.5,Math.min(14,Number.isFinite(trajectoryDotSize)?trajectoryDotSize:1));policyDotSize=Math.max(.5,Math.min(20,Number.isFinite(policyDotSize)?policyDotSize:trajectoryDotSize));hModelAlpha=Math.max(.02,Math.min(1,Number.isFinite(hModelAlpha)?hModelAlpha:.28));hModelStep=Math.max(0,Math.min(Math.max(0,hEventCount-1),Number.isFinite(hModelStep)?Math.round(hModelStep):0));
while(drawerOffset.length<3)drawerOffset.push(0);
while(axisOrigin.length<3)axisOrigin.push(0);axisOrigin=axisOrigin.slice(0,3).map(v=>Number.isFinite(Number(v))?Number(v):0);axisLengthValue=Math.max(.01,Math.min(5,Number.isFinite(axisLengthValue)?axisLengthValue:.5));
const stepRange=computeStepRange();
function add(a,b){return[a[0]+b[0],a[1]+b[1],a[2]+b[2]]}function sub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}function mul(a,s){return[a[0]*s,a[1]*s,a[2]*s]}function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function len(a){return Math.sqrt(Math.max(1e-18,dot(a,a)))}function norm(a){return mul(a,1/len(a))}
function serialize(){const layerVis={},layerStyles={};for(const [k,v] of vis.entries())layerVis[k]=v;for(const [k,v] of styles.entries())layerStyles[k]=v;const panel=document.getElementById("panel");return{showDots,drawDrawer,drawHandleDisplacement,drawHModel,drawHHuman,drawHRobot,drawHAllEvents,hModelStep,hModelAlpha,lineAlphaMin,lineAlphaMax,trajectoryLineWidth,trajectoryDotSize,policyDotSize,showAxes,axisOrigin,axisLengthValue,drawerOffset,drawerOpen,dragMode,panelHidden,panelScroll:panel?panel.scrollTop:0,camera:{yaw,pitch,dist,target:target.slice(0,3)},layerVis,layerStyles}}
function saveSettings(){try{localStorage.setItem(STORE_KEY,JSON.stringify(serialize()))}catch(_e){}}
function maybeDraw(){saveSettings();if(!pauseRedraw)draw()}
function axisBoundsPoints(){const o=axisOrigin,L=axisLengthValue;return showAxes?[o,[o[0]+L,o[1],o[2]],[o[0],o[1]+L,o[2]],[o[0],o[1],o[2]+L]]:[]}
function allPts(){const out=axisBoundsPoints().slice();for(const l of DATA.layers)for(const s of l.segments)for(const p of s.points)out.push(p);if(drawerRef){for(const seg of drawerRef.drawer||[])for(const p of seg)out.push(drawerPointRaw(p,true));for(const p of drawerHandleDisplacementPoints())out.push(p)}if(drawHModel)for(const p of hModelPoints())out.push(p);return out}
function bounds(){const pts=allPts();if(!pts.length)return{c:[0,0,0],r:1};const mn=[Infinity,Infinity,Infinity],mx=[-Infinity,-Infinity,-Infinity];for(const p of pts)for(let i=0;i<3;i++){mn[i]=Math.min(mn[i],p[i]);mx[i]=Math.max(mx[i],p[i])}return{c:[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2],r:Math.max(.05,Math.hypot(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2])/2)}}
let box;
const VIEW_PRESETS={iso:{yaw:-.85,pitch:.48},top:{yaw:-Math.PI/2,pitch:1.45},left:{yaw:Math.PI,pitch:0},right:{yaw:0,pitch:0},front:{yaw:-Math.PI/2,pitch:0}};
function setViewPreset(name,fit){const v=VIEW_PRESETS[name]||VIEW_PRESETS.iso;if(fit||!box){box=bounds();target=box.c.slice();dist=Math.max(.25,box.r*3.2)}yaw=v.yaw;pitch=v.pitch;restoredCamera=true;draw();saveSettings()}
function reset(){box=bounds();target=box.c.slice();dist=Math.max(.25,box.r*3.2);setViewPreset("iso",false)}
function cam(){const cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw),pos=[target[0]+dist*cp*cy,target[1]+dist*cp*sy,target[2]+dist*sp],f=norm(sub(target,pos));let r=cross(f,[0,0,1]);r=len(r)<1e-5?[1,0,0]:norm(r);return{pos:pos,f:f,r:r,u:norm(cross(r,f))}}
function proj(p,c){const rel=sub(p,c.pos),z=dot(rel,c.f);if(z<=1e-5)return null;const q=Math.min(canvas.width,canvas.height)*.82;return{x:canvas.width/2+dot(rel,c.r)*q/z,y:canvas.height/2-dot(rel,c.u)*q/z}}
function numericStep(v){const n=Number(v);return Number.isFinite(n)?n:null}
function computeStepRange(){let mn=Infinity,mx=-Infinity;for(const layer of DATA.layers||[])for(const seg of layer.segments||[]){const vals=Array.isArray(seg.steps)?seg.steps:[seg.step];for(const v of vals){const n=numericStep(v);if(n===null)continue;mn=Math.min(mn,n);mx=Math.max(mx,n)}}return Number.isFinite(mn)&&Number.isFinite(mx)?{min:mn,max:mx}:{min:0,max:0}}
function alphaForStep(step){const n=numericStep(step);if(n===null||stepRange.max<=stepRange.min)return lineAlphaMax;const t=Math.max(0,Math.min(1,(n-stepRange.min)/(stepRange.max-stepRange.min)));return lineAlphaMin+(lineAlphaMax-lineAlphaMin)*t}
function alphaForSteps(steps){if(Array.isArray(steps)&&steps.length){let n=null;for(const v of steps){const x=numericStep(v);if(x!==null)n=n===null?x:Math.max(n,x)}return n===null?1:alphaForStep(n)}return alphaForStep(steps)}
function line(points,color,w,c,dash,steps){ctx.save();ctx.strokeStyle=color;ctx.lineWidth=w;ctx.lineJoin="round";ctx.lineCap="round";if(dash)ctx.setLineDash(dash.map(v=>v*(window.devicePixelRatio||1)));if(Array.isArray(steps)&&steps.length>=points.length&&points.length>1&&stepRange.max>stepRange.min){for(let i=1;i<points.length;i++){const a=proj(points[i-1],c),b=proj(points[i],c);if(!a||!b)continue;ctx.globalAlpha=alphaForStep(steps[i]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}}else{ctx.globalAlpha=alphaForSteps(steps);let on=false;for(const p of points){const q=proj(p,c);if(!q){on=false;continue}if(!on){ctx.beginPath();ctx.moveTo(q.x,q.y);on=true}else ctx.lineTo(q.x,q.y)}if(on)ctx.stroke()}ctx.restore()}
function point(p,color,r,c,alpha=1){const q=proj(p,c);if(!q)return;ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.beginPath();ctx.arc(q.x,q.y,r,0,Math.PI*2);ctx.fill();ctx.restore()}
function drawAxisLabel(text,p,color,c){const q=proj(p,c);if(!q)return;const dpr=window.devicePixelRatio||1;ctx.save();ctx.font=(12*dpr)+"px Arial, sans-serif";ctx.fillStyle=color;ctx.strokeStyle="rgba(255,255,255,.82)";ctx.lineWidth=3*dpr;ctx.strokeText(text,q.x+5*dpr,q.y-5*dpr);ctx.fillText(text,q.x+5*dpr,q.y-5*dpr);ctx.restore()}
function drawWorldAxes(c){if(!showAxes)return;const dpr=window.devicePixelRatio||1,L=axisLengthValue,o=axisOrigin,x=[o[0]+L,o[1],o[2]],y=[o[0],o[1]+L,o[2]],z=[o[0],o[1],o[2]+L];line([o,x],"#ef4444",1.4*dpr,c);line([o,y],"#22c55e",1.4*dpr,c);line([o,z],"#3b82f6",1.4*dpr,c);point(o,"#111827",2*dpr,c);drawAxisLabel("X",x,"#ef4444",c);drawAxisLabel("Y",y,"#22c55e",c);drawAxisLabel("Z",z,"#3b82f6",c)}
function defaultLayerStyle(layer){return{color:layer.color,width:.5,pointSize:1}}
function layerStyle(layer){return styles.get(layer.name)||defaultLayerStyle(layer)}
function drawerPointRaw(p,move){if(!drawerRef)return p;if(drawerRef.absolute)return[p[0]+drawerOffset[0],p[1]+drawerOffset[1],p[2]+drawerOffset[2]];const o=drawerRef.origin,a=drawerRef.open_axis,m=move?drawerOpen:0;return[o[0]+drawerOffset[0]+p[0]+a[0]*m,o[1]+drawerOffset[1]+p[1]+a[1]*m,o[2]+drawerOffset[2]+p[2]+a[2]*m]}
function drawerPoint(p,move){return drawerPointRaw(p,move)}function drawerSegs(segs,color,w,c,move){for(const seg of segs)line(seg.map(p=>drawerPoint(p,move)),color,w,c)}
function drawerHandleDisplacementPoints(){if(!drawerRef)return[];return (drawerRef.handleDisplacement||[]).map(v=>Array.isArray(v)?v:v.point).filter(Boolean).map(p=>[p[0]+drawerOffset[0],p[1]+drawerOffset[1],p[2]+drawerOffset[2]])}
function drawDrawerHandleDisplacement(c){if(!drawerRef||!drawHandleDisplacement)return;const pts=drawerHandleDisplacementPoints();if(!pts.length)return;const dpr=window.devicePixelRatio||1,st=styles.get("__drawer_displacement");if(pts.length>1)line(pts,st.color,st.width*dpr,c);for(const p of pts)point(p,st.color,st.pointSize*dpr,c)}
function drawDrawerRef(c){if(!drawerRef||!drawDrawer)return;const dpr=window.devicePixelRatio||1;drawerSegs(drawerRef.drawer||[],styles.get("__drawer_box").color,styles.get("__drawer_box").width*dpr,c,true)}
function hEventsToDraw(){if(!hGeom||!Array.isArray(hGeom.events))return[];if(drawHAllEvents)return hGeom.events;const ev=hGeom.events[hModelStep];return ev?[ev]:[]}
function hModelPoints(){const out=[];for(const ev of hEventsToDraw()){const human=((ev.human||{}).capsules)||[];for(const cap of human){if(cap.a)out.push(cap.a);if(cap.b)out.push(cap.b)}for(const rt of ev.robotTraces||[]){const geom=rt.geometry||{};for(const cap of geom.capsules||[]){if(cap.a)out.push(cap.a);if(cap.b)out.push(cap.b)}for(const sp of geom.spheres||[])if(sp.center)out.push(sp.center)}}return out}
function projectedRadius(p,r,c){const q=proj(p,c);if(!q)return 0;const edge=proj(add(p,mul(c.r,Math.max(0,Number(r)||0))),c);if(!edge)return 0;return Math.min(14,Math.max(.5,Math.hypot(edge.x-q.x,edge.y-q.y)))}
function capsule(a,b,r,color,alpha,c){const qa=proj(a,c),qb=proj(b,c);if(!qa||!qb)return;const mid=mul(add(a,b),.5),w=Math.max(1,2*projectedRadius(mid,r,c));ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=w;ctx.lineJoin="round";ctx.lineCap="round";ctx.beginPath();ctx.moveTo(qa.x,qa.y);ctx.lineTo(qb.x,qb.y);ctx.stroke();ctx.restore()}
function sphere(center,r,color,alpha,c){const q=proj(center,c);if(!q)return;const pr=Math.max(1,projectedRadius(center,r,c));ctx.save();ctx.globalAlpha=alpha*.35;ctx.fillStyle=color;ctx.beginPath();ctx.arc(q.x,q.y,pr,0,Math.PI*2);ctx.fill();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=Math.max(1,window.devicePixelRatio||1);ctx.stroke();ctx.restore()}
function drawHComputeGeometry(c){if(!hGeom||!drawHModel)return;const humanSt=styles.get("__h_human"),robotSt=styles.get("__h_robot"),gripperSt=styles.get("__h_gripper");for(const ev of hEventsToDraw()){if(drawHHuman&&ev.human)for(const cap of ev.human.capsules||[])capsule(cap.a,cap.b,cap.radius,humanSt.color,hModelAlpha,c);if(drawHRobot)for(const rt of ev.robotTraces||[]){const geom=rt.geometry||{};for(const cap of geom.capsules||[])capsule(cap.a,cap.b,cap.radius,robotSt.color,hModelAlpha,c);for(const sp of geom.spheres||[])sphere(sp.center,sp.radius,gripperSt.color,Math.min(1,hModelAlpha+.1),c)}}}
function segmentHasLine(layer,seg){return seg.line!==undefined?!!seg.line:layer.line!==false}
function segmentHasMarkers(layer,seg){return seg.markers!==undefined?!!seg.markers:layer.markers!==false}
function layerHasLines(layer){return (layer.segments||[]).some(seg=>segmentHasLine(layer,seg))}
function layerHasDots(layer){return (layer.segments||[]).some(seg=>segmentHasMarkers(layer,seg))}
function isPolicyLayer(layer){const text=(String(layer.name||"")+" "+String(layer.legend||"")).toLowerCase();return text.includes("policy trajectory")||text.includes("policy pose")}
function layerDotSize(layer){return isPolicyLayer(layer)?policyDotSize:trajectoryDotSize}
function markerAlpha(layer,seg,i){if(!String(layer.name||"").toLowerCase().includes("human arm"))return 1;if(Array.isArray(seg.steps)&&i<seg.steps.length)return alphaForStep(seg.steps[i]);return alphaForSteps(seg.steps||seg.step)}
function draw(){const dpr=window.devicePixelRatio||1,w=Math.max(1,Math.floor(innerWidth*dpr)),h=Math.max(1,Math.floor(innerHeight*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}ctx.clearRect(0,0,w,h);const c=cam();drawWorldAxes(c);drawHComputeGeometry(c);for(const layer of DATA.layers){if(!vis.get(layer.name))continue;const st=layerStyle(layer);for(const seg of layer.segments){if(segmentHasLine(layer,seg))line(seg.points,st.color,trajectoryLineWidth*dpr,c,layer.dash,seg.steps||seg.step);if(showDots&&segmentHasMarkers(layer,seg))for(let i=0;i<seg.points.length;i++)point(seg.points[i],st.color,layerDotSize(layer)*dpr,c,markerAlpha(layer,seg,i))}}drawDrawerRef(c);drawDrawerHandleDisplacement(c);updateLegend()}
function styleRow(label,value,min,max,step,onInput){const row=document.createElement("label");row.className="layer-style";const name=document.createElement("span");name.textContent=label;const input=document.createElement("input");input.type="range";input.min=min;input.max=max;input.step=step;input.value=value;const val=document.createElement("span");val.textContent=(+value).toFixed(1);input.oninput=()=>{val.textContent=(+input.value).toFixed(1);onInput(+input.value);maybeDraw()};row.append(name,input,val);return row}
function colorRow(value,onInput){const row=document.createElement("label");row.className="layer-style";const name=document.createElement("span");name.textContent="Color";const input=document.createElement("input");input.type="color";input.value=value;const val=document.createElement("span");val.textContent="";input.oninput=()=>{onInput(input.value);maybeDraw()};row.append(name,input,val);return row}
function alphaRow(label,value,onInput){const row=document.createElement("label");row.className="layer-style alpha-style";const name=document.createElement("span");name.textContent=label;const input=document.createElement("input");input.type="range";input.min=.03;input.max=1;input.step=.01;input.value=value;const val=document.createElement("span");val.textContent=(+value).toFixed(2);input.oninput=()=>{onInput(+input.value);input.value=label.startsWith("Min")?lineAlphaMin:lineAlphaMax;val.textContent=(+input.value).toFixed(2);maybeDraw()};row.append(name,input,val);return row}
function axisNumberRow(label,value,min,max,step,onInput){const row=document.createElement("label");row.className="drawer-control";const name=document.createElement("span");name.textContent=label;const input=document.createElement("input");input.type="number";input.step=step;input.value=Number(value).toFixed(3);if(min!==null)input.min=min;if(max!==null)input.max=max;const unit=document.createElement("span");unit.textContent="m";const apply=()=>{let v=Number(input.value);if(!Number.isFinite(v))return;if(min!==null)v=Math.max(min,v);if(max!==null)v=Math.min(max,v);input.value=v.toFixed(3);onInput(v);maybeDraw()};input.onchange=apply;input.oninput=()=>{const v=Number(input.value);if(Number.isFinite(v)){let x=v;if(min!==null)x=Math.max(min,x);if(max!==null)x=Math.min(max,x);onInput(x);maybeDraw()}};row.append(name,input,unit);return row}
function setupAxisControls(){let root=document.getElementById("axisControls");if(!root){root=document.createElement("div");root.id="axisControls";root.className="drawer-controls";const anchor=document.getElementById("fadeControls")||document.getElementById("layers");if(anchor&&anchor.parentNode)anchor.parentNode.insertBefore(root,anchor);else return}root.innerHTML="<h2>XYZ axis</h2>";const visible=document.createElement("label");visible.className="dot-toggle";visible.innerHTML="<input type=\"checkbox\"> Show axis";visible.querySelector("input").checked=showAxes;visible.querySelector("input").onchange=e=>{showAxes=e.target.checked;maybeDraw()};root.appendChild(visible);root.appendChild(axisNumberRow("Origin X",axisOrigin[0],null,null,.005,v=>axisOrigin[0]=v));root.appendChild(axisNumberRow("Origin Y",axisOrigin[1],null,null,.005,v=>axisOrigin[1]=v));root.appendChild(axisNumberRow("Origin Z",axisOrigin[2],null,null,.005,v=>axisOrigin[2]=v));root.appendChild(axisNumberRow("Length",axisLengthValue,.01,5,.01,v=>axisLengthValue=v))}
function setupFadeControls(){const root=document.getElementById("fadeControls");if(!root)return;root.appendChild(styleRow("Trajectory line",trajectoryLineWidth,.1,12,.1,v=>trajectoryLineWidth=v));root.appendChild(styleRow("Trajectory dots",trajectoryDotSize,.5,14,.5,v=>trajectoryDotSize=v));if(DATA.layers.some(layer=>isPolicyLayer(layer)&&layerHasDots(layer)))root.appendChild(styleRow("Black policy dots",policyDotSize,.5,20,.5,v=>policyDotSize=v));root.appendChild(alphaRow("Min alpha",lineAlphaMin,v=>{lineAlphaMin=Math.max(.03,Math.min(v,lineAlphaMax-.01))}));root.appendChild(alphaRow("Max alpha",lineAlphaMax,v=>{lineAlphaMax=Math.max(lineAlphaMin+.01,Math.min(1,v))}))}
function legendRows(){const rows=[];function addLegend(name,color){if(rows.find(r=>r.name===name))return;rows.push({name:name,color:color})}for(const layer of DATA.layers){if(!vis.get(layer.name)||!layer.segments.length)continue;const st=layerStyle(layer);addLegend(layer.legend||layer.name,st.color)}if(drawerRef&&drawDrawer)addLegend("drawer",styles.get("__drawer_box")?styles.get("__drawer_box").color:"#0f766e");if(drawerRef&&drawHandleDisplacement&&drawerHandleDisplacementPoints().length)addLegend("drawer handle displacement",styles.get("__drawer_displacement")?styles.get("__drawer_displacement").color:"#dc2626");return rows}
function updateLegend(){const root=document.getElementById("legend");if(!root)return;const rows=legendRows();if(!rows.length){root.hidden=true;return}root.hidden=false;root.innerHTML=rows.map(r=>"<div class=\"legend-row\"><span class=\"legend-line\" style=\"color:"+r.color+"\"></span><span>"+r.name+"</span></div>").join("")}
function roundRectPath(g,x,y,w,h,r){g.beginPath();g.moveTo(x+r,y);g.lineTo(x+w-r,y);g.quadraticCurveTo(x+w,y,x+w,y+r);g.lineTo(x+w,y+h-r);g.quadraticCurveTo(x+w,y+h,x+w-r,y+h);g.lineTo(x+r,y+h);g.quadraticCurveTo(x,y+h,x,y+h-r);g.lineTo(x,y+r);g.quadraticCurveTo(x,y,x+r,y);g.closePath()}
function downloadName(suffix){return (DATA.title||"safechunk_trajectory").replace(/[^a-z0-9]+/gi,"_").toLowerCase()+suffix+".png"}
function downloadCanvas(out,name){const a=document.createElement("a");a.download=name;a.href=out.toDataURL("image/png");a.click()}
function downloadScreenshot(){draw();downloadCanvas(canvas,downloadName(""))}
function drawLegendCanvas(){const rows=legendRows(),out=document.createElement("canvas"),g=out.getContext("2d"),dpr=window.devicePixelRatio||1;if(!rows.length)return null;g.font=(12*dpr)+"px Arial, sans-serif";const pad=10*dpr,rowH=19*dpr,lineW=24*dpr,gap=7*dpr,maxText=Math.max(...rows.map(r=>g.measureText(r.name).width));const boxW=Math.ceil(Math.min(340*dpr,Math.max(150*dpr,pad*2+lineW+gap+maxText))),boxH=Math.ceil(pad*2+rowH*rows.length);out.width=boxW;out.height=boxH;g.font=(12*dpr)+"px Arial, sans-serif";g.fillStyle="rgba(255,255,255,.84)";roundRectPath(g,0,0,boxW,boxH,8*dpr);g.fill();g.strokeStyle="#d6d2c8";g.lineWidth=dpr;g.stroke();g.fillStyle="#1f2937";g.textBaseline="middle";rows.forEach((r,i)=>{const cy=pad+rowH*(i+.5);g.strokeStyle=r.color;g.lineWidth=2*dpr;g.beginPath();g.moveTo(pad,cy);g.lineTo(pad+lineW,cy);g.stroke();g.fillStyle="#1f2937";g.fillText(r.name,pad+lineW+gap,cy)});return out}
function downloadLegend(){const out=drawLegendCanvas();if(out)downloadCanvas(out,downloadName("_legend"))}
function setupLayerControls(){const root=document.getElementById("layers");for(const layer of DATA.layers){const st={...defaultLayerStyle(layer),...((savedState.layerStyles||{})[layer.name]||{})};styles.set(layer.name,st);const defaultVisible=layer.visible!==false&&!layer.name.toLowerCase().includes("objective");const visible=(savedState.layerVis&&savedState.layerVis[layer.name]!==undefined)?!!savedState.layerVis[layer.name]:defaultVisible;vis.set(layer.name,visible);const row=document.createElement("div");row.className="layer";const main=document.createElement("label");main.className="layer-main";const cb=document.createElement("input");cb.type="checkbox";cb.checked=visible;cb.onchange=()=>{vis.set(layer.name,cb.checked);maybeDraw()};const sw=document.createElement("span");sw.className="sw";sw.style.background=st.color;const nm=document.createElement("span");nm.textContent=layer.name;const ct=document.createElement("span");ct.className="count";ct.textContent=layer.segments.length+" seg";main.append(cb,sw,nm,ct);row.appendChild(main);row.appendChild(colorRow(st.color,v=>{st.color=v;sw.style.background=v}));root.appendChild(row)}}
function setupFrameDiagnostics(){const root=document.getElementById("frameDiagnostics"),d=DATA.frameDiagnostics;if(!root||!d)return;if(!d.count){root.textContent=d.note||"No frame diagnostics available.";root.hidden=false;return}const w=d.worst||{};root.innerHTML="<strong>Frame check</strong><br>MuJoCo world execution vs safety-model FK: mean "+Number(d.mean_norm||0).toFixed(4)+" m, max "+Number(d.max_norm||0).toFixed(4)+" m over "+d.count+" comparisons.<br>Worst: "+(w.label||"unknown")+" at timestep "+(w.step??"?")+", diff ["+((w.diff||[]).map(v=>Number(v).toFixed(4)).join(", "))+"] m.<br>"+(d.note||"");root.hidden=false}
function setupHModelControls(){
const saved=savedState.layerStyles||{};
styles.set("__h_human",{color:"#ef4444",width:.5,pointSize:1,...(saved.__h_human||{})});
styles.set("__h_robot",{color:"#0891b2",width:.5,pointSize:1,...(saved.__h_robot||{})});
styles.set("__h_gripper",{color:"#7c3aed",width:.5,pointSize:1,...(saved.__h_gripper||{})});
const root=document.getElementById("hModelControls");if(!root)return;if(!hGeom){root.hidden=true;return}
root.innerHTML="<h2>Inflated h-model</h2>";
const main=document.createElement("label");main.className="dot-toggle";main.innerHTML="<input type=\"checkbox\"> Show h model";main.querySelector("input").checked=drawHModel;main.querySelector("input").onchange=e=>{drawHModel=e.target.checked;maybeDraw()};root.appendChild(main);
const human=document.createElement("label");human.className="dot-toggle";human.innerHTML="<input type=\"checkbox\"> Human capsules";human.querySelector("input").checked=drawHHuman;human.querySelector("input").onchange=e=>{drawHHuman=e.target.checked;maybeDraw()};root.appendChild(human);
const robot=document.createElement("label");robot.className="dot-toggle";robot.innerHTML="<input type=\"checkbox\"> Robot capsules + gripper";robot.querySelector("input").checked=drawHRobot;robot.querySelector("input").onchange=e=>{drawHRobot=e.target.checked;maybeDraw()};root.appendChild(robot);
const all=document.createElement("label");all.className="dot-toggle";all.innerHTML="<input type=\"checkbox\"> Show all h timesteps";all.querySelector("input").checked=drawHAllEvents;all.querySelector("input").onchange=e=>{drawHAllEvents=e.target.checked;maybeDraw()};root.appendChild(all);
if(hEventCount>1){const row=document.createElement("label");row.className="drawer-control";const name=document.createElement("span");name.textContent="h step";const slider=document.createElement("input");slider.type="range";slider.min=0;slider.max=hEventCount-1;slider.step=1;slider.value=hModelStep;const val=document.createElement("span");val.textContent=String(hModelStep);slider.oninput=()=>{hModelStep=Math.max(0,Math.min(hEventCount-1,Math.round(+slider.value)));val.textContent=String(hModelStep);maybeDraw()};row.append(name,slider,val);root.appendChild(row)}
root.appendChild(styleRow("Alpha",hModelAlpha,.02,1,.01,v=>hModelAlpha=v));
root.appendChild(colorRow(styles.get("__h_human").color,v=>styles.get("__h_human").color=v));root.lastChild.firstChild.textContent="Human";
root.appendChild(colorRow(styles.get("__h_robot").color,v=>styles.get("__h_robot").color=v));root.lastChild.firstChild.textContent="Robot";
root.appendChild(colorRow(styles.get("__h_gripper").color,v=>styles.get("__h_gripper").color=v));root.lastChild.firstChild.textContent="Gripper";
}
function setupDrawerControls(){
const saved=savedState.layerStyles||{};
styles.set("__drawer_box",{color:"#0f766e",width:.5,pointSize:1,...(saved.__drawer_box||{})});
styles.set("__drawer_displacement",{color:"#dc2626",width:.5,pointSize:1,...(saved.__drawer_displacement||{})});
const root=document.getElementById("drawerControls");if(!root)return;if(!drawerRef){root.hidden=true;return}
root.innerHTML="<h2>Drawer reference</h2>";
const visible=document.createElement("label");visible.className="dot-toggle";visible.innerHTML="<input type=\"checkbox\"> Show drawer";visible.querySelector("input").checked=drawDrawer;visible.querySelector("input").onchange=e=>{drawDrawer=e.target.checked;maybeDraw()};root.appendChild(visible);
if(hasHandleDisplacement){const dispVisible=document.createElement("label");dispVisible.className="dot-toggle";dispVisible.innerHTML="<input type=\"checkbox\"> Drawer handle displacement";dispVisible.querySelector("input").checked=drawHandleDisplacement;dispVisible.querySelector("input").onchange=e=>{drawHandleDisplacement=e.target.checked;maybeDraw()};root.appendChild(dispVisible)}
const drawerStyleKeys=[["__drawer_box","Drawer",false]];
if(hasHandleDisplacement)drawerStyleKeys.push(["__drawer_displacement","Handle displacement",true]);
for(const key of drawerStyleKeys){const st=styles.get(key[0]);root.appendChild(colorRow(st.color,v=>st.color=v));root.lastChild.firstChild.textContent=key[1];root.appendChild(styleRow("Line",st.width,.1,12,.1,v=>st.width=v));if(key[2])root.appendChild(styleRow("Dots",st.pointSize,.5,14,.5,v=>st.pointSize=v))}
const defs=drawerRef.absolute?[["x","X",-.35,.35,drawerOffset[0]],["y","Y",-.35,.35,drawerOffset[1]],["z","Z",-.25,.25,drawerOffset[2]]]:[["open","Open",-.05,.55,drawerOpen],["x","X",-.35,.35,drawerOffset[0]],["y","Y",-.35,.35,drawerOffset[1]],["z","Z",-.25,.25,drawerOffset[2]]];
for(const d of defs){const row=document.createElement("label");row.className="drawer-control";const name=document.createElement("span");name.textContent=d[1];const slider=document.createElement("input");slider.type="range";slider.min=d[2];slider.max=d[3];slider.step=.005;slider.value=d[4];const val=document.createElement("span");val.textContent=(+slider.value).toFixed(3);slider.oninput=()=>{const v=+slider.value;val.textContent=v.toFixed(3);if(d[0]==="open")drawerOpen=v;else drawerOffset[{x:0,y:1,z:2}[d[0]]]=v;maybeDraw()};row.append(name,slider,val);root.appendChild(row)}
}
function setPanelHidden(hidden){panelHidden=!!hidden;document.getElementById("panel").hidden=panelHidden;document.getElementById("showPanel").hidden=!panelHidden;saveSettings()}
function setupViewPresets(){const root=document.getElementById("viewPresets");if(!root)return;for(const btn of root.querySelectorAll("button[data-view]"))btn.onclick=()=>setViewPreset(btn.dataset.view,false)}
function setup(){document.getElementById("title").textContent=DATA.title||"SafeChunk 3D trajectories";const dots=document.getElementById("dots");dots.checked=showDots;dots.onchange=e=>{showDots=e.target.checked;maybeDraw()};const mode=document.getElementById("dragMode");mode.value=dragMode;mode.onchange=e=>{dragMode=e.target.value;saveSettings()};document.getElementById("pauseRedraw").onchange=e=>{pauseRedraw=e.target.checked;if(!pauseRedraw)draw()};document.getElementById("clearSettings").onclick=()=>{localStorage.removeItem(STORE_KEY);localStorage.removeItem(LEGACY_STORE_KEY);location.reload()};setupViewPresets();setupAxisControls();setupFadeControls();setupFrameDiagnostics();setupHModelControls();setupDrawerControls();setupLayerControls();setPanelHidden(panelHidden);const panel=document.getElementById("panel");panel.scrollTop=Math.max(0,Number(savedState.panelScroll)||0);panel.onscroll=()=>saveSettings();document.getElementById("hidePanel").onclick=()=>setPanelHidden(true);document.getElementById("showPanel").onclick=()=>setPanelHidden(false);saveSettings()}
canvas.onmousedown=e=>{drag=true;pan=dragMode==="pan"||e.shiftKey||e.button===2;lx=e.clientX;ly=e.clientY};canvas.oncontextmenu=e=>e.preventDefault();window.onmouseup=()=>drag=false;window.onmousemove=e=>{if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;if(pan){const c=cam(),f=dist/Math.max(300,Math.min(canvas.width,canvas.height));target=add(target,add(mul(c.r,-dx*f),mul(c.u,dy*f)))}else{yaw+=dx*.006;pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*.006))}draw();saveSettings()};window.onkeydown=e=>{if(["INPUT","SELECT","TEXTAREA"].includes(document.activeElement&&document.activeElement.tagName))return;const c=cam(),step=dist*.045;let used=true;if(e.key==="ArrowLeft")target=add(target,mul(c.r,-step));else if(e.key==="ArrowRight")target=add(target,mul(c.r,step));else if(e.key==="ArrowUp")target=add(target,mul(c.u,step));else if(e.key==="ArrowDown")target=add(target,mul(c.u,-step));else if(e.key==="a"||e.key==="A")yaw-=.08;else if(e.key==="d"||e.key==="D")yaw+=.08;else if(e.key==="w"||e.key==="W")pitch=Math.max(-1.45,Math.min(1.45,pitch+.08));else if(e.key==="s"||e.key==="S")pitch=Math.max(-1.45,Math.min(1.45,pitch-.08));else used=false;if(used){e.preventDefault();draw();saveSettings()}};canvas.onwheel=e=>{e.preventDefault();dist=Math.max(.03,dist*Math.exp(e.deltaY*.001));draw();saveSettings()};window.onresize=()=>{if(!pauseRedraw)draw()};document.getElementById("reset").onclick=reset;document.getElementById("shot").onclick=downloadScreenshot;document.getElementById("legendShot").onclick=downloadLegend;setup();if(restoredCamera)draw();else reset();
</script>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__DATA__", payload_json))
    return str(path)

def _save_chunk_trajectory_viewer(path: Path, title: str, trace_records: list[dict], human_samples: list[dict], executed_samples: list[dict], max_events: int):
    return _save_chunk_trajectory_canvas_viewer(
        path,
        title,
        trace_records,
        human_samples,
        executed_samples,
        max_events,
    )

    try:
        import plotly.graph_objects as go
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not import plotly for interactive trajectory viewer; "
            "writing canvas fallback: %s",
            exc,
        )
        return _save_chunk_trajectory_canvas_viewer(
            path,
            title,
            trace_records,
            human_samples,
            executed_samples,
            max_events,
        )

    layers = _chunk_trajectory_viewer_layers(
        trace_records,
        human_samples,
        executed_samples,
        max_events,
    )
    if not any(segment.get("points") for layer in layers for segment in layer.get("segments", [])):
        return None

    fig = go.Figure()
    for layer in layers:
        _add_plotly_segments(
            fig,
            go,
            layer.get("segments", []),
            name=layer.get("name", "trajectory"),
            color=layer.get("color", "#111827"),
            width=int(layer.get("width", 3)),
            dash=layer.get("dash"),
            line=bool(layer.get("line", True)),
            markers=bool(layer.get("markers", True)),
        )

    fig.update_layout(
        title=title,
        scene={"xaxis_title": "x world", "yaxis_title": "y world", "zaxis_title": "z world", "aspectmode": "data"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "groupclick": "togglegroup"},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        template="plotly_white",
    )
    config = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": Path(path).stem, "height": 1000, "width": 1400, "scale": 2}}
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True, full_html=True, config=config)
    return str(path)
