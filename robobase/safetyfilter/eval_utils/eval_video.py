from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from omegaconf import OmegaConf

from robobase.envs.bigym import BiGymEnvFactory
from robobase.safetyfilter.eval_utils.eval_utils import make_eval_env


def _video_duration_seconds(video_recorder) -> float:
    timestamps = getattr(video_recorder, "timestamps", [])
    if len(timestamps) < 2:
        return 0.0
    return max(0.0, float(timestamps[-1] - timestamps[0]))

def _video_recorded_steps(video_recorder) -> int:
    frames = getattr(video_recorder, "frames", [])
    if len(frames) > 0:
        return max(0, len(frames) - 1)
    states = getattr(video_recorder, "_states", [])
    return max(0, len(states) - 1)

def _resolve_video_stop_steps(args, video_recorder) -> Optional[int]:
    if not args.record_video or args.video_time_base != "sim":
        return None
    if args.stop_video_at_steps is not None:
        return int(args.stop_video_at_steps)
    if args.stop_video_at_seconds is None:
        return None
    fps = float(getattr(video_recorder, "fps", 20))
    return max(1, int(round(args.stop_video_at_seconds * fps)))

def _load_snapshot_normalization_cfg(snapshot_path: Path):
    cfg_path = snapshot_path.parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            "Could not load normalization stats from snapshot because the Hydra "
            f"config is missing: {cfg_path}"
        )
    return OmegaConf.load(cfg_path)

def _make_eval_env_with_normalization(cfg, normalization_cfg=None):
    if normalization_cfg is None:
        return make_eval_env(cfg)

    stats_factory = BiGymEnvFactory()
    stats_factory.collect_or_fetch_demos(
        normalization_cfg,
        normalization_cfg.demos,
    )

    env_factory = BiGymEnvFactory()
    env_factory._action_stats = copy.deepcopy(stats_factory._action_stats)
    env_factory._obs_stats = copy.deepcopy(stats_factory._obs_stats)
    return env_factory.make_eval_env(cfg)

def _normalization_context(cfg):
    manifest = cfg.env.get("manifest", None)
    return (
        str(cfg.env.task_name),
        None if manifest is None else str(manifest),
        str(cfg.demos),
    )

def _resolve_normalization_cfg(args, cfg, snapshot_path: Path):
    if args.normalization_source == "eval":
        return "eval", None

    snapshot_cfg = _load_snapshot_normalization_cfg(snapshot_path)
    if args.normalization_source == "snapshot":
        return "snapshot", snapshot_cfg

    eval_context = _normalization_context(cfg)
    snapshot_context = _normalization_context(snapshot_cfg)
    if eval_context != snapshot_context:
        return "snapshot(auto)", snapshot_cfg
    return "eval(auto)", None

def _print_normalization_source(name: str, cfg):
    manifest = cfg.env.get("manifest", None)
    print("normalization_source:", name)
    print("normalization_task:", cfg.env.task_name)
    print("normalization_demos:", cfg.demos)
    print(
        "normalization_manifest:",
        manifest if manifest is not None else "<DemoStore/default>",
    )

def _normalize_rgb_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3:
        return None

    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] > 3:
        frame = frame[..., :3]

    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if frame.size and np.nanmax(frame) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame

def _policy_obs_rgb_frame(policy_obs):
    frames = []
    for key in sorted(k for k in policy_obs if k.startswith("rgb_")):
        value = np.asarray(policy_obs[key])
        if value.ndim == 4:
            frame = value[-1]
        elif value.ndim == 3:
            frame = value
        else:
            continue

        frame = _normalize_rgb_frame(frame)
        if frame is not None:
            frames.append(frame)

    if not frames:
        return None

    height = min(frame.shape[0] for frame in frames)
    frames = [frame[:height] for frame in frames]
    return np.concatenate(frames, axis=1)

def _save_policy_obs_video(frames, timestamps, path: Path, default_fps=20):
    if not frames:
        return
    fps = default_fps
    if len(timestamps) > 1:
        duration = timestamps[-1] - timestamps[0]
        if duration > 0:
            fps = len(frames) / duration
    imageio.mimsave(str(path), np.asarray(frames), fps=fps)
