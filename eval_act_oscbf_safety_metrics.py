"""
export SNAPSHOT=exp_local/pixel_act/bigym_drawer_top_open_20260527214324/snapshots/15000_snapshot.pt
ACT-only safety-monitor evaluation:
    cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase

    /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
    eval_act_oscbf_safety_metrics.py \
    --condition act \
    --snapshot $SNAPSHOT \
    --env bigym/human_arm_drawer_top_open \
    --episodes 2 \
    --steps 200 \
    --demos 1 \
    --out debug_act_human_env_drawer_stats.jsonl \
    --override env.manifest=/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/manifest.json \
    --override env.privileged_information=false \
    --override env.require_mode_label=false \
    --plot-terminal \
    --debug 

ACT-only safety-monitor evaluation on Original env (no demo override, with privileged info):
    cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase

    /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
    eval_act_oscbf_safety_metrics.py \
    --condition act \
    --snapshot $SNAPSHOT \
    --env bigym/drawer_top_open \
    --episodes 5 \
    --steps 3500 \
    --demos 1 \
    --out debug_act_human_env_drawer_stats.jsonl \
    --output-dir eval_safety/eval_15000 \
    --override env.episode_length=400000 \
    --override +env.manifest=/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/manifest.json \
    --override +env.privileged_information=false \
    --override +env.require_mode_label=false \
    --debug \
    --stop-video-at 2:30

ACT + single-action OSCBF evaluation:
    cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase

    /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
    eval_act_oscbf_safety_metrics.py \
    --condition oscbf \
    --snapshot $SNAPSHOT \
    --env bigym/human_arm_drawer_top_open \
    --episodes 2 \
    --steps 500 \
    --demos 1 \
    --out metrics_act_single_step_oscbf_human.jsonl \
    --override env.episode_length=20000 \
    --override env.manifest=/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/manifest.json \
    --override env.privileged_information=false \
    --override env.require_mode_label=false \
    --debug \
    --plot-terminal
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm may not be installed in every environment
    tqdm = None

from robobase.eval_utils import (
    WallClockVideoRecorder,
    infer_env_action_shape,
    extract_first_action,
    replace_first_action,
    get_non_arm_indices,
    normalise_env_action_shape,
    make_output_paths,
    make_cfg,
    make_eval_env,
    make_workspace_and_load_snapshot,
    policy_action,
    compute_oscbf_h_monitor,
    count_robot_human_contacts,
    extract_success,
    assert_action_properties,
    summarise_episode,
    summarise_all_episodes,
)

from robobase.safetyfilter.h1_state_bridge import extract_h1_state
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from robobase.safetyfilter.h1_state_bridge import extract_h1_state
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _IgnoreBigymVersionMismatchFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Installed version of bigym" not in message

logging.getLogger().addFilter(_IgnoreBigymVersionMismatchFilter())


REPO = Path("/home/xd1125/Workspace/safe_bigym_hoi")
ROBOBASE_CFG = REPO / "external/robobase/robobase/cfgs"
H1_URDF = REPO / "external/oscbf/oscbf/assets/h1/h1.urdf"


@dataclass
class StepMetrics:
    condition: str
    episode: int
    step: int

    reward: float
    terminated: bool
    truncated: bool
    success: bool

    min_h: Optional[float]
    h_values: Optional[list[float]]
    h_violation: Optional[bool]

    contact_count: Optional[int]

    arm_delta: float
    non_arm_delta: float
    full_delta: float
    intervention_active: bool

    nominal_arm_min: float
    nominal_arm_max: float
    safe_arm_min: float
    safe_arm_max: float

    action_norm: float
    safe_action_norm: float

    filter_time_ms: float
    monitor_time_ms: float


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        choices=["act", "oscbf"],
        required=True,
        help="act = monitor only; oscbf = apply single-action OSCBF.",
    )

    parser.add_argument(
        "--snapshot",
        required=True,
        type=str,
        help="Path to ACT snapshot, e.g. snapshots/latest_snapshot.pt.",
    )

    parser.add_argument(
        "--env",
        default="bigym/human_arm_drawer_top_open",
        help="Evaluation env, e.g. bigym/human_arm_drawer_top_open.",
    )

    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--demos", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out", type=str, default="eval/act_oscbf_metrics.jsonl")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_safety",
        help=(
            "Root output folder for step logs, summaries, and videos. "
            "If unset, derives a folder from the --out file path."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--intervention-eps", type=float, default=1e-4)
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Disable video recording.",
    )
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument(
        "--stop-video-at",
        type=str,
        default="2:30",
        help=(
            "Stop each episode once the recorded video reaches this duration. "
            "Use seconds, M:S, H:M:S, or 'none' to disable."
        ),
    )
    parser.add_argument(
        "--plot-terminal",
        action="store_true",
        help="Render ASCII terminal plots for step metrics after each episode.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars for episodes and steps.",
    )

    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=None,
        help="Optional max per-dimension OSCBF action edit.",
    )

    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override. Can be used multiple times.",
    )

    args = parser.parse_args()
    args.record_video = not args.no_record_video
    try:
        args.stop_video_at_seconds = _parse_duration_seconds(args.stop_video_at)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def _parse_duration_seconds(value: str) -> Optional[float]:
    value = value.strip().lower()
    if value in {"", "none", "off", "false", "0"}:
        return None

    parts = value.split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes, seconds_part = parts
            seconds = int(minutes) * 60 + float(seconds_part)
        elif len(parts) == 3:
            hours, minutes, seconds_part = parts
            seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds_part)
        else:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --stop-video-at duration: {value!r}"
        ) from exc

    if seconds <= 0:
        return None
    return seconds


def _video_duration_seconds(video_recorder) -> float:
    timestamps = getattr(video_recorder, "timestamps", [])
    if len(timestamps) < 2:
        return 0.0
    return max(0.0, float(timestamps[-1] - timestamps[0]))


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


def _make_progress_bar(*args, **kwargs):
    if tqdm is None:
        return None
    return tqdm(
        *args,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        **kwargs,
    )


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


def _plot_episode_metrics_lines(episode, episode_metrics, width=80, height=10):
    reward_values = [m.reward for m in episode_metrics]
    min_h_values = [float("nan") if m.min_h is None else m.min_h for m in episode_metrics]
    arm_delta_values = [m.arm_delta for m in episode_metrics]
    non_arm_delta_values = [m.non_arm_delta for m in episode_metrics]
    contact_values = [m.contact_count for m in episode_metrics]

    lines = []
    lines.extend(_ascii_plot_lines(f"Episode {episode:03d} reward", reward_values, width=width, height=height))
    lines.extend(_ascii_plot_lines(f"Episode {episode:03d} min_h", min_h_values, width=width, height=height))
    lines.extend(_ascii_plot_lines(f"Episode {episode:03d} arm_delta", arm_delta_values, width=width, height=height))
    lines.extend(_ascii_plot_lines(f"Episode {episode:03d} non_arm_delta", non_arm_delta_values, width=width, height=height))
    lines.extend(_ascii_plot_lines(f"Episode {episode:03d} contact_count", contact_values, width=width, height=height))
    return lines


def make_oscbf_filter(args) -> OSCBFFilter:
    return OSCBFFilter(
        urdf_path=str(H1_URDF),
        debug=args.debug,
        use_dummy_filter=False,
        dummy_scale=0.5,
        control_type="absolute",
        max_action_delta=args.max_action_delta,
    )




def main():
    args = parse_args()
    np.random.seed(args.seed)

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    cfg = make_cfg(args)

    print("\n=== Creating Workspace and loading ACT snapshot ===")
    ws = make_workspace_and_load_snapshot(cfg, snapshot_path)

    print("\n=== Creating evaluation env ===")
    env = make_eval_env(cfg)
    env_action_shape = infer_env_action_shape(env, fallback=(16, 16))
    print("env_action_shape:", env_action_shape)

    output_root, step_jsonl_path, episode_summary_path, final_summary_path = make_output_paths(args)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.video_dir is not None:
        video_dir = Path(args.video_dir)
    else:
        video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_recorder = WallClockVideoRecorder(video_dir if args.record_video else None)

    print("record_video:", args.record_video)
    print("video_dir:", video_dir)
    print("stop_video_at_seconds:", args.stop_video_at_seconds)

    print("\n=== Creating OSCBF monitor/filter ===")
    oscbf = make_oscbf_filter(args)
    print("condition:", args.condition)
    print("arm indices:", oscbf.bigym_action_arm_indices.tolist())


    all_step_metrics: list[StepMetrics] = []
    all_episode_summaries: list[dict] = []
    show_progress = tqdm is not None and not args.no_progress
    episode_bar = None

    if show_progress:
        episode_bar = _make_progress_bar(
            total=args.episodes,
            desc="episodes",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
        episode_bar.set_postfix(episodes_left=args.episodes)

    try:
        for episode in range(args.episodes):
            print(f"\n========== Episode {episode} ==========")

            obs, info = env.reset()
            video_recorder.init(env, enabled=args.record_video)
            episode_metrics: list[StepMetrics] = []
            if show_progress:
                progress_bar = _make_progress_bar(
                    total=args.steps,
                    desc=f"ep={episode:03d} steps",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                )
                progress_bar.set_postfix(steps_left=args.steps)
            else:
                progress_bar = None

            for step in range(args.steps):
                h1state = extract_h1_state(
                    env,
                    print_diagnostics=(episode == 0 and step == 0),
                )

                q_full = np.asarray(h1state.q_full, dtype=np.float32).reshape(-1)
                qd_full = np.asarray(h1state.qd_full, dtype=np.float32).reshape(-1)

                if q_full.shape != (14,):
                    raise ValueError(f"Expected q_full shape (14,), got {q_full.shape}")
                if qd_full.shape != (14,):
                    raise ValueError(f"Expected qd_full shape (14,), got {qd_full.shape}")

                env_action = policy_action(ws, obs, step=step)
                env_action = normalise_env_action_shape(env_action, env_action_shape)

                first_action = extract_first_action(env_action)

                monitor_t0 = time.perf_counter()
                min_h, h_values, h_violation = compute_oscbf_h_monitor(
                    filt=oscbf,
                    env=env,
                    obs=obs,
                    q_full=q_full,
                    qd_full=qd_full,
                )
                monitor_time_ms = 1000.0 * (time.perf_counter() - monitor_t0)

                filter_t0 = time.perf_counter()

                if args.condition == "oscbf":
                    safe_first_action = oscbf(
                        action=first_action,
                        env=env,
                        observations=obs,
                        q_full=q_full,
                        qd_full=qd_full,
                    )
                else:
                    safe_first_action = first_action.copy()

                filter_time_ms = 1000.0 * (time.perf_counter() - filter_t0)

                assert_action_properties(
                    nominal_action=first_action,
                    safe_action=safe_first_action,
                    arm_indices=oscbf.bigym_action_arm_indices,
                )

                safe_env_action = replace_first_action(
                    env_action=env_action,
                    safe_first_action=safe_first_action,
                )

                obs, reward, terminated, truncated, info = env.step(safe_env_action)
                video_recorder.record(env)

                arm_idx = oscbf.bigym_action_arm_indices
                non_arm_idx = get_non_arm_indices(first_action.shape[0], arm_idx)

                nominal_arm = first_action[arm_idx]
                safe_arm = safe_first_action[arm_idx]

                arm_delta = float(np.linalg.norm(safe_arm - nominal_arm))
                non_arm_delta = float(
                    np.linalg.norm(
                        safe_first_action[non_arm_idx] - first_action[non_arm_idx]
                    )
                )
                full_delta = float(np.linalg.norm(safe_first_action - first_action))

                success = extract_success(info, float(reward), bool(terminated))
                contact_count = count_robot_human_contacts(env)

                step_metrics = StepMetrics(
                    condition=args.condition,
                    episode=episode,
                    step=step,

                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    success=success,

                    min_h=min_h,
                    h_values=h_values,
                    h_violation=h_violation,

                    contact_count=contact_count,

                    arm_delta=arm_delta,
                    non_arm_delta=non_arm_delta,
                    full_delta=full_delta,
                    intervention_active=bool(arm_delta > args.intervention_eps),

                    nominal_arm_min=float(np.min(nominal_arm)),
                    nominal_arm_max=float(np.max(nominal_arm)),
                    safe_arm_min=float(np.min(safe_arm)),
                    safe_arm_max=float(np.max(safe_arm)),

                    action_norm=float(np.linalg.norm(first_action)),
                    safe_action_norm=float(np.linalg.norm(safe_first_action)),

                    filter_time_ms=float(filter_time_ms),
                    monitor_time_ms=float(monitor_time_ms),
                )

                episode_metrics.append(step_metrics)
                all_step_metrics.append(step_metrics)

                video_duration_s = _video_duration_seconds(video_recorder)
                video_left_s = None
                if args.record_video and args.stop_video_at_seconds is not None:
                    video_left_s = max(0.0, args.stop_video_at_seconds - video_duration_s)

                if progress_bar is not None:
                    progress_bar.update(1)
                    postfix = {"steps_left": args.steps - progress_bar.n}
                    if video_left_s is not None:
                        postfix["video_left"] = f"{video_left_s:.1f}s"
                    progress_bar.set_postfix(postfix)
                elif args.debug:
                    print(
                        f"ep={episode:03d} step={step:04d} "
                        f"reward={float(reward):.3f} "
                        f"min_h={min_h} "
                        f"arm_delta={arm_delta:.5f} "
                        f"non_arm_delta={non_arm_delta:.5f} "
                        f"contact_count={contact_count} "
                        f"filter_ms={filter_time_ms:.2f}"
                    )

                reached_video_limit = (
                    args.record_video
                    and args.stop_video_at_seconds is not None
                    and video_duration_s >= args.stop_video_at_seconds
                )
                if reached_video_limit:
                    print(
                        f"Stopping episode {episode} at "
                        f"{video_duration_s:.1f}s of recorded video "
                        f"(target {args.stop_video_at_seconds:.1f}s)."
                    )
                    break
                if terminated or truncated or success:
                    break

            episode_summary = summarise_episode(episode_metrics)
            all_episode_summaries.append(episode_summary)

            if progress_bar is not None:
                progress_bar.close()

            print("\nEpisode summary:")
            for key, value in episode_summary.items():
                print(f"  {key}: {value}")

            if args.plot_terminal:
                _plot_episode_metrics(episode, episode_metrics)

            video_recorder.save(f"{args.condition}_episode_{episode:03d}.mp4")

            if episode_bar is not None:
                episode_bar.update(1)
                episode_bar.set_postfix(episodes_left=args.episodes - episode_bar.n)

    finally:
        if episode_bar is not None:
            episode_bar.close()
        env.close()

    final_summary = summarise_all_episodes(all_episode_summaries)

    with step_jsonl_path.open("w") as f:
        for metric in all_step_metrics:
            f.write(json.dumps(asdict(metric)) + "\n")

    with episode_summary_path.open("w") as f:
        json.dump(all_episode_summaries, f, indent=2)

    with final_summary_path.open("w") as f:
        json.dump(final_summary, f, indent=2)

    print("\n========== Final summary ==========")
    for key, value in final_summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print("  step metrics:", step_jsonl_path)
    print("  episode summaries:", episode_summary_path)
    print("  final summary:", final_summary_path)


if __name__ == "__main__":
    main()