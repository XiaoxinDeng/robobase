"""
Evaluate ACT checkpoints with chunk-level safety filters.

Example chunk-deform evaluation:
    cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase

    /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
    eval_act_safechunk_safety_metrics.py \
    --condition chunk_deform \
    --snapshot $SNAPSHOT \
    --env bigym/human_arm_drawer_top_open \
    --episodes 2 \
    --steps 500 \
    --demos 1 \
    --out metrics_act_safechunk_deform_human.jsonl \
    --override env.episode_length=20000 \
    --debug \
    --plot-terminal

Example sequential OSCBF-over-chunk evaluation:
    /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
    eval_act_safechunk_safety_metrics.py \
    --condition sequential_oscbf \
    --snapshot $SNAPSHOT \
    --env bigym/human_arm_drawer_top_open \
    --episodes 2 \
    --steps 500 \
    --demos 1 \
    --out metrics_act_sequential_oscbf_chunk_human.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm may not be installed in every environment
    tqdm = None

from eval_act_oscbf_safety_metrics import (
    _make_progress_bar,
    _parse_duration_seconds,
    _path_consistent_brake_kwargs_from_config,
    _plot_episode_metrics,
    _video_duration_seconds,
)
from robobase.eval_utils import (
    WallClockVideoRecorder,
    assert_action_properties,
    compute_oscbf_h_monitor,
    count_robot_human_contacts,
    extract_first_action,
    extract_success,
    get_non_arm_indices,
    infer_env_action_shape,
    make_cfg,
    make_eval_env,
    make_output_paths,
    make_workspace_and_load_snapshot,
    normalise_env_action_shape,
    policy_action,
    summarise_all_episodes,
    summarise_episode,
)
from robobase.safetyfilter.h1_state_bridge import extract_h1_state
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter
from robobase.safetyfilter.safechunk_deform_filter import SafeChunkDeformFilter
from robobase.safetyfilter.path_consistent_brake_filter import PathConsistentBrakeFilter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _IgnoreBigymVersionMismatchFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Installed version of bigym" not in record.getMessage()


logging.getLogger().addFilter(_IgnoreBigymVersionMismatchFilter())

REPO = Path("/home/xd1125/Workspace/safe_bigym_hoi")
H1_URDF = REPO / "external/oscbf/oscbf/assets/h1/h1.urdf"


@dataclass
class ChunkStepMetrics:
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
    chunk_min_clearance: Optional[float]
    chunk_first_violation: Optional[int]
    chunk_unsafe_count: Optional[int]

    contact_count: Optional[int]

    arm_delta: float
    non_arm_delta: float
    full_delta: float
    chunk_arm_delta: float
    chunk_non_arm_delta: float
    chunk_full_delta: float
    intervention_active: bool

    nominal_arm_min: float
    nominal_arm_max: float
    safe_arm_min: float
    safe_arm_max: float

    action_norm: float
    safe_action_norm: float
    chunk_action_norm: float
    safe_chunk_action_norm: float

    safety_mode: Optional[str]
    deformation_source: Optional[str]
    progress_scale: Optional[float]
    deadlock: Optional[bool]
    brake_stop_idx: Optional[int]
    deformation_norm: Optional[float]
    deform_safe: Optional[bool]
    deform_min_clearance: Optional[float]
    chunk_deform_scale: Optional[float]
    chunk_deform_attempts: Optional[int]

    filter_time_ms: float
    monitor_time_ms: float


class HorizonOSCBFOperator:
    """Expose OSCBF single-step calls and horizon h evaluation to SafeChunk."""

    def __init__(
        self,
        oscbf: OSCBFFilter,
        min_clearance: float,
        dt: float = 0.05,
        predict_human_motion: bool = True,
        human_prediction_max_time: Optional[float] = 0.25,
        human_prediction_max_speed: Optional[float] = 3.0,
    ):
        self.oscbf = oscbf
        self.min_clearance = float(min_clearance)
        self.dt = float(dt)
        self.predict_human_motion = bool(predict_human_motion)
        self.human_prediction_max_time = (
            None
            if human_prediction_max_time is None or human_prediction_max_time <= 0
            else float(human_prediction_max_time)
        )
        self.human_prediction_max_speed = (
            None
            if human_prediction_max_speed is None or human_prediction_max_speed <= 0
            else float(human_prediction_max_speed)
        )
        self.env = None
        self.obs = None
        self.q_full = None
        self.qd_full = None
        self._prev_capsule_a_world = None
        self._prev_capsule_b_world = None
        self._prev_capsule_radii = None
        self._capsule_a_velocity_world = None
        self._capsule_b_velocity_world = None
        self._human_motion_prediction_available = False
        self._human_motion_prediction_speed = 0.0

    def set_context(self, env, obs, q_full: np.ndarray, qd_full: np.ndarray):
        self.env = env
        self.obs = obs
        self.q_full = np.asarray(q_full, dtype=np.float32).reshape(-1)
        self.qd_full = np.asarray(qd_full, dtype=np.float32).reshape(-1)
        self._update_human_capsule_velocity()

    def _limit_capsule_velocity(self, velocity):
        if self.human_prediction_max_speed is None:
            return velocity
        velocity = np.asarray(velocity, dtype=np.float32)
        norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
        scale = np.minimum(
            1.0,
            self.human_prediction_max_speed / np.maximum(norm, 1e-9),
        )
        return velocity * scale

    def _update_human_capsule_velocity(self):
        self._human_motion_prediction_available = False
        self._human_motion_prediction_speed = 0.0
        if not self.predict_human_motion or self.env is None:
            return
        try:
            human_obstacles = self.oscbf._extract_human_obstacles(self.env, self.obs)
            capsule_a = np.asarray(human_obstacles["capsule_a"], dtype=np.float32)
            capsule_b = np.asarray(human_obstacles["capsule_b"], dtype=np.float32)
            capsule_radii = np.asarray(
                human_obstacles["capsule_radii"],
                dtype=np.float32,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Human capsule velocity update failed: %s", exc)
            return

        if (
            self._prev_capsule_a_world is not None
            and self._prev_capsule_b_world is not None
            and self._prev_capsule_radii is not None
            and capsule_a.shape == self._prev_capsule_a_world.shape
            and capsule_b.shape == self._prev_capsule_b_world.shape
            and capsule_radii.shape == self._prev_capsule_radii.shape
        ):
            dt = max(float(self.dt), 1e-6)
            a_velocity = (capsule_a - self._prev_capsule_a_world) / dt
            b_velocity = (capsule_b - self._prev_capsule_b_world) / dt
            a_velocity = self._limit_capsule_velocity(a_velocity)
            b_velocity = self._limit_capsule_velocity(b_velocity)
            self._capsule_a_velocity_world = a_velocity.astype(np.float32)
            self._capsule_b_velocity_world = b_velocity.astype(np.float32)
            endpoint_speeds = np.concatenate(
                [
                    np.linalg.norm(a_velocity, axis=-1),
                    np.linalg.norm(b_velocity, axis=-1),
                ]
            )
            self._human_motion_prediction_speed = float(np.max(endpoint_speeds))
            self._human_motion_prediction_available = bool(
                np.isfinite(self._human_motion_prediction_speed)
                and self._human_motion_prediction_speed > 1e-9
            )
        else:
            self._capsule_a_velocity_world = None
            self._capsule_b_velocity_world = None

        self._prev_capsule_a_world = capsule_a.copy()
        self._prev_capsule_b_world = capsule_b.copy()
        self._prev_capsule_radii = capsule_radii.copy()

    def _human_capsule_rollout(self, capsule_a_world, capsule_b_world, capsule_radii, horizon):
        capsule_a_world = np.asarray(capsule_a_world, dtype=np.float32)
        capsule_b_world = np.asarray(capsule_b_world, dtype=np.float32)
        capsule_radii = np.asarray(capsule_radii, dtype=np.float32)
        current_a = np.broadcast_to(
            capsule_a_world[None, :, :],
            (horizon,) + capsule_a_world.shape,
        ).copy()
        current_b = np.broadcast_to(
            capsule_b_world[None, :, :],
            (horizon,) + capsule_b_world.shape,
        ).copy()

        info = {
            "human_motion_prediction_enabled": bool(self.predict_human_motion),
            "human_motion_prediction_available": False,
            "human_motion_prediction_dt": float(self.dt),
            "human_motion_prediction_max_time": self.human_prediction_max_time,
            "human_motion_prediction_max_speed": self.human_prediction_max_speed,
            "human_motion_prediction_speed": float(self._human_motion_prediction_speed),
            "human_motion_prediction_max_displacement": 0.0,
        }
        if (
            not self.predict_human_motion
            or not self._human_motion_prediction_available
            or self._capsule_a_velocity_world is None
            or self._capsule_b_velocity_world is None
            or self._capsule_a_velocity_world.shape != capsule_a_world.shape
            or self._capsule_b_velocity_world.shape != capsule_b_world.shape
        ):
            return current_a, current_b, capsule_radii, info

        times = (np.arange(horizon, dtype=np.float32) + 1.0) * float(self.dt)
        if self.human_prediction_max_time is not None:
            times = np.minimum(times, float(self.human_prediction_max_time))
        predicted_a = (
            capsule_a_world[None, :, :]
            + times[:, None, None] * self._capsule_a_velocity_world[None, :, :]
        )
        predicted_b = (
            capsule_b_world[None, :, :]
            + times[:, None, None] * self._capsule_b_velocity_world[None, :, :]
        )
        capsule_a_seq = np.concatenate([current_a, predicted_a], axis=1)
        capsule_b_seq = np.concatenate([current_b, predicted_b], axis=1)
        capsule_radii_pred = np.concatenate([capsule_radii, capsule_radii], axis=0)
        info.update(
            {
                "human_motion_prediction_available": True,
                "human_motion_prediction_max_displacement": float(
                    self._human_motion_prediction_speed * float(np.max(times))
                ),
            }
        )
        return capsule_a_seq, capsule_b_seq, capsule_radii_pred, info

    @property
    def bigym_action_arm_indices(self):
        return self.oscbf.bigym_action_arm_indices

    def __call__(self, action, obs=None, **kwargs):
        q_full = kwargs.pop("q_full", self.q_full)
        qd_full = kwargs.pop("qd_full", self.qd_full)
        env = kwargs.pop("env", self.env)
        observations = kwargs.pop("observations", obs if obs is not None else self.obs)
        return self.oscbf(
            action=action,
            env=env,
            observations=observations,
            q_full=q_full,
            qd_full=qd_full,
            **kwargs,
        )

    def evaluate_safety(self, obs, q_seq):
        if self.oscbf.oscbf_config is None:
            return self._unavailable(q_seq)
        if self.env is None or self.q_full is None:
            return self._unavailable(q_seq)

        q_seq = np.asarray(q_seq, dtype=np.float32)
        try:
            human_obstacles = self.oscbf._extract_human_obstacles(self.env, obs)
            capsule_a_world = human_obstacles["capsule_a"]
            capsule_b_world = human_obstacles["capsule_b"]
            capsule_radii = human_obstacles["capsule_radii"]
            (
                capsule_a_world_seq,
                capsule_b_world_seq,
                capsule_radii_eval,
                prediction_info,
            ) = self._human_capsule_rollout(
                capsule_a_world,
                capsule_b_world,
                capsule_radii,
                q_seq.shape[0],
            )

            min_clearances = []
            for k, q_bigym in enumerate(q_seq):
                q_bigym = np.asarray(q_bigym, dtype=np.float32).reshape(-1)
                qd_bigym = np.zeros_like(q_bigym, dtype=np.float32)

                q_urdf, _, _, _ = self.oscbf._build_urdf_surrogate_state_from_bigym(
                    q_bigym,
                    qd_bigym,
                )

                t_world_urdf = self.oscbf._get_world_T_urdf_from_bigym_state(q_bigym)
                t_urdf_world = np.linalg.inv(t_world_urdf)
                capsule_a_urdf = self.oscbf._transform_points(
                    t_urdf_world,
                    capsule_a_world_seq[k],
                )
                capsule_b_urdf = self.oscbf._transform_points(
                    t_urdf_world,
                    capsule_b_world_seq[k],
                )
                self.oscbf._validate_capsules(
                    capsule_a_urdf,
                    capsule_b_urdf,
                    capsule_radii_eval,
                )
                self.oscbf.oscbf_config.set_human_capsules(
                    capsule_a_urdf,
                    capsule_b_urdf,
                    capsule_radii_eval,
                )
                h_values = np.asarray(
                    self.oscbf.oscbf_config.h_1(
                        jnp.asarray(q_urdf, dtype=jnp.float32)
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                min_clearances.append(float(np.min(h_values)))

            min_clearances = np.asarray(min_clearances, dtype=np.float32)
            unsafe = np.flatnonzero(min_clearances < self.min_clearance)
            info = {
                "horizon_safe": bool(unsafe.size == 0),
                "min_clearance": float(np.min(min_clearances)),
                "min_clearances": min_clearances,
                "first_violation": int(unsafe[0]) if unsafe.size else None,
                "unsafe_count": int(unsafe.size),
                "safety_eval_available": True,
            }
            info.update(prediction_info)
            return info
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chunk horizon OSCBF monitor failed: %s", exc)
            return self._unavailable(q_seq)

    def _unavailable(self, q_seq):
        h = int(np.asarray(q_seq).shape[0])
        return {
            "horizon_safe": True,
            "min_clearance": float("inf"),
            "min_clearances": np.full(h, np.inf, dtype=np.float32),
            "first_violation": None,
            "unsafe_count": 0,
            "safety_eval_available": False,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["act", "chunk_deform", "path_consistent_brake", "sequential_oscbf"],
        required=True,
        help=(
            "act = monitor only; chunk_deform = SafeChunk-Deform horizon "
            "deformation; path_consistent_brake = standalone path-consistent braking filter; "
            "sequential_oscbf = apply OSCBF to each chunk action."
        ),
    )
    parser.add_argument(
        "--path-consistent-brake-config",
        type=str,
        default=None,
        help=(
            "Optional YAML overlay for PathConsistentBrake parameters. Relative "
            "names are resolved under robobase/cfgs/safety_filter."
        ),
    )
    parser.add_argument("--snapshot", required=True, type=str)
    parser.add_argument("--env", default="bigym/human_arm_drawer_top_open")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--demos", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="eval/act_safechunk_metrics.jsonl")
    parser.add_argument("--output-dir", type=str, default="eval_safety")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--intervention-eps", type=float, default=1e-4)
    parser.add_argument("--no-record-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--stop-video-at", type=str, default="2:30")
    parser.add_argument("--plot-terminal", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-action-delta", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--chunk-min-clearance", type=float, default=0.08)
    parser.add_argument("--chunk-horizon-predict-human-motion", dest="chunk_horizon_predict_human_motion", action="store_true", default=True, help="Propagate human-arm capsules with finite-difference velocity during SafeChunk horizon safety checks.")
    parser.add_argument("--no-chunk-horizon-predict-human-motion", dest="chunk_horizon_predict_human_motion", action="store_false")
    parser.add_argument("--chunk-human-motion-prediction-max-time", type=float, default=0.25)
    parser.add_argument("--chunk-human-motion-prediction-max-speed", type=float, default=3.0)
    parser.add_argument(
        "--chunk-deformation-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
    )
    parser.add_argument("--chunk-deformation-smoothing", type=int, default=1)
    parser.add_argument(
        "--sequential-oscbf-fallback",
        action="store_true",
        help="Allow chunk_deform to fall back to sequential OSCBF if no candidate is safe.",
    )
    parser.add_argument("--override", action="append", default=[])

    args = parser.parse_args()
    args.record_video = not args.no_record_video
    try:
        args.stop_video_at_seconds = _parse_duration_seconds(args.stop_video_at)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def make_oscbf_filter(args) -> OSCBFFilter:
    return OSCBFFilter(
        urdf_path=str(H1_URDF),
        debug=args.debug,
        use_dummy_filter=False,
        dummy_scale=0.5,
        control_type="absolute",
        max_action_delta=args.max_action_delta,
    )


def make_safechunk_filter(args, operator: HorizonOSCBFOperator) -> SafeChunkDeformFilter:
    filter_cls = (
        PathConsistentBrakeFilter
        if args.condition == "path_consistent_brake"
        else SafeChunkDeformFilter
    )
    path_consistent_brake_kwargs = {}
    if args.condition == "path_consistent_brake":
        path_consistent_brake_kwargs = _path_consistent_brake_kwargs_from_config(args)

    return filter_cls(
        oscbf_operator=operator,
        horizon=args.horizon,
        dt=0.05,
        action_dim=16,
        expected_motion_dim=14,
        control_type="absolute",
        min_clearance=args.chunk_min_clearance,
        brake_progress_threshold=0.05,
        deformation_enabled=args.condition != "path_consistent_brake",
        chunk_deformation_scales=args.chunk_deformation_scales,
        chunk_deformation_smoothing=args.chunk_deformation_smoothing,
        sequential_oscbf_fallback=(
            False
            if args.condition == "path_consistent_brake"
            else args.sequential_oscbf_fallback
        ),
        **path_consistent_brake_kwargs,
        debug=args.debug,
    )


def as_chunk(action: np.ndarray) -> tuple[np.ndarray, bool]:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        return action.reshape(1, -1), True
    if action.ndim == 2:
        return action, False
    raise ValueError(f"Unsupported action shape: {action.shape}")


def restore_action_shape(chunk: np.ndarray, was_single: bool) -> np.ndarray:
    return chunk[0].copy() if was_single else chunk.copy()


def apply_filter(args, safechunk, env_action, obs, env, q_full, qd_full):
    chunk, was_single = as_chunk(env_action)
    filter_t0 = time.perf_counter()

    if args.condition == "act":
        safe_chunk = chunk.copy()
        q_seq = safechunk.rollout_nominal_chunk(obs, chunk)
        safety_info = safechunk.evaluate_horizon_safety(obs, q_seq)
        safety_info = dict(safety_info)
        safety_info.update({"safety_mode": "act", "mode": "act"})
    elif args.condition in {"chunk_deform", "path_consistent_brake"}:
        safe_chunk, safety_info = safechunk.filter_chunk(
            obs,
            chunk,
            env=env,
            q_full=q_full,
            qd_full=qd_full,
        )
    elif args.condition == "sequential_oscbf":
        safe_chunk, safety_info = safechunk.deform_chunk_with_oscbf(
            obs,
            chunk,
            env=env,
            q_full=q_full,
            qd_full=qd_full,
        )
        safety_info = dict(safety_info)
        safety_info.update(
            {
                "safety_mode": "sequential_oscbf",
                "mode": "sequential_oscbf",
                "deformation_source": "sequential_oscbf",
            }
        )
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(args.condition)

    filter_time_ms = 1000.0 * (time.perf_counter() - filter_t0)
    return restore_action_shape(np.asarray(safe_chunk, dtype=np.float32), was_single), safety_info, filter_time_ms


def assert_chunk_properties(nominal_chunk, safe_chunk, arm_indices):
    nominal_chunk, was_single = as_chunk(nominal_chunk)
    safe_chunk, _ = as_chunk(safe_chunk)
    if safe_chunk.shape != nominal_chunk.shape:
        raise AssertionError(
            f"Safe chunk shape {safe_chunk.shape} != nominal chunk shape {nominal_chunk.shape}"
        )
    if not np.isfinite(safe_chunk).all():
        raise AssertionError("Safe chunk contains non-finite values")
    for k in range(nominal_chunk.shape[0]):
        assert_action_properties(nominal_chunk[k], safe_chunk[k], arm_indices)


def safe_info_get(info: dict[str, Any], key: str):
    value = info.get(key)
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarise_chunk_episode(metrics: list[ChunkStepMetrics]) -> dict:
    summary = summarise_episode(metrics)
    if len(metrics) == 0:
        return summary

    chunk_arm_delta = np.asarray([m.chunk_arm_delta for m in metrics], dtype=np.float32)
    chunk_non_arm_delta = np.asarray([m.chunk_non_arm_delta for m in metrics], dtype=np.float32)
    chunk_full_delta = np.asarray([m.chunk_full_delta for m in metrics], dtype=np.float32)
    chunk_interventions = np.asarray([m.intervention_active for m in metrics], dtype=np.float32)
    deform_norms = [m.deformation_norm for m in metrics if m.deformation_norm is not None]
    deform_safe = [m.deform_safe for m in metrics if m.deform_safe is not None]
    modes = [m.safety_mode for m in metrics if m.safety_mode is not None]
    sources = [m.deformation_source for m in metrics if m.deformation_source is not None]

    def rate(values, target):
        return float(np.mean([v == target for v in values])) if values else None

    summary.update(
        {
            "mean_chunk_arm_delta": float(np.mean(chunk_arm_delta)),
            "max_chunk_arm_delta": float(np.max(chunk_arm_delta)),
            "mean_chunk_non_arm_delta": float(np.mean(chunk_non_arm_delta)),
            "max_chunk_non_arm_delta": float(np.max(chunk_non_arm_delta)),
            "mean_chunk_full_delta": float(np.mean(chunk_full_delta)),
            "max_chunk_full_delta": float(np.max(chunk_full_delta)),
            "chunk_intervention_frequency": float(np.mean(chunk_interventions)),
            "mean_deformation_norm": float(np.mean(deform_norms)) if deform_norms else None,
            "deform_safe_rate": float(np.mean(deform_safe)) if deform_safe else None,
            "pass_through_rate": rate(modes, "pass_through"),
            "horizon_brake_rate": rate(modes, "horizon_brake"),
            "path_consistent_brake_rate": rate(modes, "path_consistent_brake"),
            "path_consistent_brake_intended_rate": rate(modes, "path_consistent_brake_intended_step"),
            "horizon_brake_intended_rate": rate(modes, "horizon_brake_intended_step"),
            "verified_failsafe_rate": rate(modes, "verified_failsafe"),
            "unverified_emergency_failsafe_rate": rate(modes, "unverified_emergency_failsafe"),
            "horizon_deform_rate": rate(modes, "horizon_deform"),
            "sequential_oscbf_rate": rate(modes, "sequential_oscbf"),
            "chunk_deform_source_rate": rate(sources, "chunk_deform"),
            "sequential_oscbf_source_rate": rate(sources, "sequential_oscbf"),
        }
    )
    return summary


def summarise_all_chunk_episodes(episode_summaries: list[dict]) -> dict:
    summary = summarise_all_episodes(episode_summaries)

    def mean_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def max_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.max(vals)) if vals else None

    summary.update(
        {
            "mean_chunk_arm_delta": mean_of("mean_chunk_arm_delta"),
            "max_chunk_arm_delta_over_episodes": max_of("max_chunk_arm_delta"),
            "mean_chunk_intervention_frequency": mean_of("chunk_intervention_frequency"),
            "mean_deformation_norm": mean_of("mean_deformation_norm"),
            "mean_deform_safe_rate": mean_of("deform_safe_rate"),
            "mean_pass_through_rate": mean_of("pass_through_rate"),
            "mean_horizon_brake_rate": mean_of("horizon_brake_rate"),
            "mean_path_consistent_brake_rate": mean_of("path_consistent_brake_rate"),
            "mean_path_consistent_brake_intended_rate": mean_of("path_consistent_brake_intended_rate"),
            "mean_horizon_brake_intended_rate": mean_of("horizon_brake_intended_rate"),
            "mean_verified_failsafe_rate": mean_of("verified_failsafe_rate"),
            "mean_unverified_emergency_failsafe_rate": mean_of("unverified_emergency_failsafe_rate"),
            "mean_horizon_deform_rate": mean_of("horizon_deform_rate"),
            "mean_sequential_oscbf_rate": mean_of("sequential_oscbf_rate"),
            "mean_chunk_deform_source_rate": mean_of("chunk_deform_source_rate"),
            "mean_sequential_oscbf_source_rate": mean_of("sequential_oscbf_source_rate"),
        }
    )
    return summary


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

    video_dir = Path(args.video_dir) if args.video_dir is not None else output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_recorder = WallClockVideoRecorder(video_dir if args.record_video else None)

    print("record_video:", args.record_video)
    print("video_dir:", video_dir)
    print("stop_video_at_seconds:", args.stop_video_at_seconds)

    print("\n=== Creating OSCBF monitor and SafeChunk filter ===")
    oscbf = make_oscbf_filter(args)
    horizon_operator = HorizonOSCBFOperator(
        oscbf,
        min_clearance=args.chunk_min_clearance,
        dt=0.05,
        predict_human_motion=args.chunk_horizon_predict_human_motion,
        human_prediction_max_time=args.chunk_human_motion_prediction_max_time,
        human_prediction_max_speed=args.chunk_human_motion_prediction_max_speed,
    )
    safechunk = make_safechunk_filter(args, horizon_operator)
    print("condition:", args.condition)
    print("arm indices:", oscbf.bigym_action_arm_indices.tolist())
    print("chunk_deformation_scales:", safechunk.chunk_deformation_scales)
    print("sequential_oscbf_fallback:", safechunk.sequential_oscbf_fallback)
    print("chunk_horizon_predict_human_motion:", horizon_operator.predict_human_motion)
    print("chunk_human_motion_prediction_max_time:", horizon_operator.human_prediction_max_time)
    print("chunk_human_motion_prediction_max_speed:", horizon_operator.human_prediction_max_speed)

    all_step_metrics: list[ChunkStepMetrics] = []
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
            episode_metrics: list[ChunkStepMetrics] = []

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

                horizon_operator.set_context(env, obs, q_full, qd_full)

                monitor_t0 = time.perf_counter()
                min_h, h_values, h_violation = compute_oscbf_h_monitor(
                    filt=oscbf,
                    env=env,
                    obs=obs,
                    q_full=q_full,
                    qd_full=qd_full,
                )
                monitor_time_ms = 1000.0 * (time.perf_counter() - monitor_t0)

                safe_env_action, safety_info, filter_time_ms = apply_filter(
                    args,
                    safechunk,
                    env_action,
                    obs,
                    env,
                    q_full,
                    qd_full,
                )
                assert_chunk_properties(env_action, safe_env_action, oscbf.bigym_action_arm_indices)

                safe_first_action = extract_first_action(safe_env_action)
                obs, reward, terminated, truncated, info = env.step(safe_env_action)
                video_recorder.record(env)

                arm_idx = oscbf.bigym_action_arm_indices
                non_arm_idx = get_non_arm_indices(first_action.shape[0], arm_idx)
                nominal_arm = first_action[arm_idx]
                safe_arm = safe_first_action[arm_idx]

                nominal_chunk, _ = as_chunk(env_action)
                safe_chunk, _ = as_chunk(safe_env_action)
                chunk_arm_delta = float(
                    np.linalg.norm(safe_chunk[:, arm_idx] - nominal_chunk[:, arm_idx])
                )
                chunk_non_arm_delta = float(
                    np.linalg.norm(safe_chunk[:, non_arm_idx] - nominal_chunk[:, non_arm_idx])
                )
                chunk_full_delta = float(np.linalg.norm(safe_chunk - nominal_chunk))

                arm_delta = float(np.linalg.norm(safe_arm - nominal_arm))
                non_arm_delta = float(
                    np.linalg.norm(safe_first_action[non_arm_idx] - first_action[non_arm_idx])
                )
                full_delta = float(np.linalg.norm(safe_first_action - first_action))

                success = extract_success(info, float(reward), bool(terminated))
                contact_count = count_robot_human_contacts(env)

                step_metrics = ChunkStepMetrics(
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
                    chunk_min_clearance=safe_info_get(safety_info, "min_clearance"),
                    chunk_first_violation=safe_info_get(safety_info, "first_violation"),
                    chunk_unsafe_count=safe_info_get(safety_info, "unsafe_count"),
                    contact_count=contact_count,
                    arm_delta=arm_delta,
                    non_arm_delta=non_arm_delta,
                    full_delta=full_delta,
                    chunk_arm_delta=chunk_arm_delta,
                    chunk_non_arm_delta=chunk_non_arm_delta,
                    chunk_full_delta=chunk_full_delta,
                    intervention_active=bool(chunk_arm_delta > args.intervention_eps),
                    nominal_arm_min=float(np.min(nominal_arm)),
                    nominal_arm_max=float(np.max(nominal_arm)),
                    safe_arm_min=float(np.min(safe_arm)),
                    safe_arm_max=float(np.max(safe_arm)),
                    action_norm=float(np.linalg.norm(first_action)),
                    safe_action_norm=float(np.linalg.norm(safe_first_action)),
                    chunk_action_norm=float(np.linalg.norm(nominal_chunk)),
                    safe_chunk_action_norm=float(np.linalg.norm(safe_chunk)),
                    safety_mode=safe_info_get(safety_info, "safety_mode") or safe_info_get(safety_info, "mode"),
                    deformation_source=safe_info_get(safety_info, "deformation_source"),
                    progress_scale=safe_info_get(safety_info, "progress_scale"),
                    deadlock=safe_info_get(safety_info, "deadlock"),
                    brake_stop_idx=safe_info_get(safety_info, "brake_stop_idx"),
                    deformation_norm=safe_info_get(safety_info, "deformation_norm"),
                    deform_safe=safe_info_get(safety_info, "deform_safe"),
                    deform_min_clearance=safe_info_get(safety_info, "deform_min_clearance"),
                    chunk_deform_scale=safe_info_get(safety_info, "chunk_deform_scale"),
                    chunk_deform_attempts=safe_info_get(safety_info, "chunk_deform_attempts"),
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
                        f"reward={float(reward):.3f} min_h={min_h} "
                        f"mode={step_metrics.safety_mode} "
                        f"source={step_metrics.deformation_source} "
                        f"chunk_delta={chunk_arm_delta:.5f} "
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
                        f"Stopping episode {episode} at {video_duration_s:.1f}s "
                        f"of recorded video (target {args.stop_video_at_seconds:.1f}s)."
                    )
                    break
                if terminated or truncated or success:
                    break

            episode_summary = summarise_chunk_episode(episode_metrics)
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

    final_summary = summarise_all_chunk_episodes(all_episode_summaries)

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
