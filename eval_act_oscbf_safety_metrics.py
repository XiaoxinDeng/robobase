"""
export SNAPSHOT_V0=exp_local/pixel_act/bigym_drawer_top_open_20260527214324/snapshots/15000_snapshot.pt
export SNAPSHOT_V1=exp_local/pixel_act/bigym_drawer_top_open_20260528010800/snapshots/30000_snapshot.pt
export SNAPSHOT_V2=exp_local/pixel_act/bigym_drawer_top_open_20260528034109/snapshots/3000_snapshot.pt
export SNAPSHOT=$SNAPSHOT_V2
export MANIFEST=/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/manifest.json

ACT evaluation:
    /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
    eval_act_oscbf_safety_metrics.py \
    --condition act \
    --snapshot $SNAPSHOT_V2 \
    --env bigym/drawer_top_open \
    --episodes 5 \
    --steps 3500 \
    --stop-video-at 2:00 \
    --demos 40 \
    --out debug_act_human_env_drawer_stats.jsonl \
    --output-dir eval_safety/v2_eval_3000 \
    --override frame_stack=4 \
    --debug




ACT + single-action OSCBF evaluation:
    cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase

    /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
    eval_act_oscbf_safety_metrics.py \
    --condition oscbf \
    --snapshot $SNAPSHOT \
    --env bigym/human_arm_drawer_top_open \
    --episodes 2 \
    --steps 1000 \
    --demos 40 \
    --out debug_act_human_env_drawer_stats.jsonl \
    --output-dir eval_safety/act_monitor_human_env \
    --override env.manifest=$MANIFEST \
    --override env.privileged_information=false \
    --override env.require_mode_label=false \
    --override frame_stack=4 \
    --debug




ACT + single-action OSCBF evaluation:
    cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase

    /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
    eval_act_oscbf_safety_metrics.py \
    --condition oscbf \
    --snapshot $SNAPSHOT \
    --env bigym/human_arm_drawer_top_open \
    --episodes 2 \
    --steps 500 \
    --demos 40 \
    --out metrics_act_single_step_oscbf_human.jsonl \
    --override env.episode_length=20000 \
    --override env.manifest=$MANIFEST \
    --override env.privileged_information=false \
    --override env.require_mode_label=false \
    --debug \
    --plot-terminal
"""

from __future__ import annotations

import argparse
import copy
import imageio
import json
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

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
    robot_human_contact_pairs,
    extract_success,
    assert_action_properties,
    summarise_episode,
    summarise_all_episodes,
)

from robobase.envs.bigym import BiGymEnvFactory
from robobase.safetyfilter.h1_state_bridge import extract_h1_state, get_bigym_task
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter
from robobase.safetyfilter.safechunk_deform_filter import SafeChunkDeformFilter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

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

    human_phase: Optional[str]
    ee_to_handle_dist: Optional[float]
    human_blocker_triggered: Optional[bool]
    human_time_in_phase: Optional[float]
    min_robot_human_distance: Optional[float]
    drawer_open_distance: Optional[float]
    drawer_open_fraction: Optional[float]
    drawer_joint_position: Optional[float]
    task_progress: Optional[float]
    task_progress_before: Optional[float]
    task_progress_after: Optional[float]
    task_progress_delta: Optional[float]
    ee_object_distance: Optional[float]
    object_state: Optional[dict[str, Any]]

    min_h: Optional[float]
    h_values: Optional[list[float]]
    h_violation: Optional[bool]
    chunk_min_clearance: Optional[float]
    chunk_first_violation: Optional[int]
    chunk_unsafe_count: Optional[int]
    horizon_risk_gap: Optional[float]
    horizon_risk_gap_active: Optional[bool]
    horizon_clearance_drop: Optional[float]

    contact_count: Optional[int]
    contact_pairs: Optional[list[str]]

    arm_delta: float
    base_delta: float
    non_arm_delta: float
    full_delta: float
    per_step_action_delta_norm: float
    per_step_arm_delta_norm: float
    per_step_base_delta_norm: float
    chunk_arm_delta: Optional[float]
    chunk_base_delta: Optional[float]
    chunk_non_arm_delta: Optional[float]
    chunk_full_delta: Optional[float]
    chunk_action_delta_norm: Optional[float]
    chunk_arm_delta_norm: Optional[float]
    chunk_base_delta_norm: Optional[float]
    chunk_modified_fraction: Optional[float]
    chunk_modified_steps: Optional[int]
    chunk_first_modified_step: Optional[int]
    chunk_last_modified_step: Optional[int]
    chunk_mean_step_arm_delta: Optional[float]
    chunk_max_step_arm_delta: Optional[float]
    chunk_future_arm_delta: Optional[float]
    chunk_future_edit_fraction: Optional[float]
    chunk_first_edit_fraction: Optional[float]
    chunk_safe_arm_variation: Optional[float]
    chunk_nominal_arm_variation: Optional[float]
    chunk_arm_variation_delta: Optional[float]
    chunk_edit_variation: Optional[float]
    path_mean_deviation: Optional[float]
    path_max_deviation: Optional[float]
    path_final_deviation: Optional[float]
    chunk_preemptive_intervention: Optional[bool]
    intervention_active: bool

    nominal_arm_min: float
    nominal_arm_max: float
    safe_arm_min: float
    safe_arm_max: float

    action_norm: float
    safe_action_norm: float
    raw_action_norm: Optional[float]
    raw_arm_min: Optional[float]
    raw_arm_max: Optional[float]
    chunk_action_norm: Optional[float]
    safe_chunk_action_norm: Optional[float]
    safety_mode: Optional[str]
    pause_reason: Optional[str]
    deformation_source: Optional[str]
    deformation_norm: Optional[float]
    deform_safe: Optional[bool]
    deform_min_clearance: Optional[float]
    chunk_deform_scale: Optional[float]
    chunk_deform_attempts: Optional[int]
    deform_mode: Optional[str]
    optimized_accepted: Optional[bool]
    optimized_fallback: Optional[str]
    optimized_reject_reason: Optional[str]
    debug_safety_feasibility: Optional[bool]
    safety_rejected: Optional[bool]
    recovery_rejected: Optional[bool]
    rejection_cause: Optional[str]
    best_min_clearance: Optional[float]
    required_min_clearance: Optional[float]
    clearance_gap: Optional[float]
    recovery_mode: Optional[str]
    recovery_phase: Optional[str]
    cached_motion_active: Optional[bool]
    deform_stage_min_clearance: Optional[float]
    deform_stage_accepted: Optional[bool]
    recover_min_clearance: Optional[float]
    recover_rejoin_loss: Optional[float]
    recover_target_index: Optional[int]
    recover_accepted: Optional[bool]
    recover_required: Optional[bool]
    recovery_candidate_class: Optional[str]
    recover_reject_reason: Optional[str]
    recover_path_min_clearance: Optional[float]
    recover_immediate_clearance: Optional[float]
    recover_prefix_min_clearance: Optional[float]
    recover_path_safe: Optional[bool]
    recover_immediate_safe: Optional[bool]
    recover_prefix_safe: Optional[bool]
    recover_safe_prefix_len: Optional[int]
    recover_target_key: Optional[str]
    recovery_path_failure_streak: Optional[int]
    direct_rejoin_attempted: Optional[bool]
    direct_rejoin_rejected: Optional[bool]
    detour_rejoin_attempted: Optional[bool]
    detour_rejoin_accepted: Optional[bool]
    delayed_rejoin_active: Optional[bool]
    delayed_rejoin_steps: Optional[int]
    repeated_unsafe_target: Optional[bool]
    post_recovery_act_window_active: Optional[bool]
    post_recovery_act_steps_remaining: Optional[int]
    post_recovery_act_window_interrupted: Optional[bool]
    resumed_from_cached_index: Optional[int]
    is_recoverable: Optional[bool]
    rejoin_index: Optional[int]
    rejoin_cost: Optional[float]
    safety_loss: Optional[float]
    action_deviation_loss: Optional[float]
    path_loss: Optional[float]
    rejoin_loss: Optional[float]
    q_rejoin_loss: Optional[float]
    q_rejoin_dist: Optional[float]
    q_rejoin_threshold: Optional[float]
    q_rejoin_index: Optional[int]
    qd_rejoin_loss: Optional[float]
    qd_rejoin_dist: Optional[float]
    qd_rejoin_threshold: Optional[float]
    qd_rejoin_index: Optional[int]
    ee_rejoin_loss: Optional[float]
    ee_rejoin_dist: Optional[float]
    ee_rejoin_threshold: Optional[float]
    ee_rejoin_index: Optional[int]
    ee_final_check_available: Optional[bool]
    inner_rejoin_metric: Optional[str]
    final_rejoin_metric: Optional[str]
    rejoin_q_eval_time_ms: Optional[float]
    rejoin_qd_eval_time_ms: Optional[float]
    ee_nom_cache_time_ms: Optional[float]
    ee_final_check_time_ms: Optional[float]
    existing_optimization_loss: Optional[float]
    smoothness_loss: Optional[float]
    total_loss: Optional[float]
    fallback_used: Optional[bool]
    act_resume_index: Optional[int]
    act_resume_supported: Optional[bool]
    committed_chunk_active: Optional[bool]
    committed_chunk_mode: Optional[str]
    committed_chunk_index: Optional[int]
    committed_chunk_length: Optional[int]
    committed_rejoin_index: Optional[int]
    committed_chunk_started: Optional[bool]
    committed_chunk_completed: Optional[bool]
    committed_aborted_due_to_safety: Optional[bool]
    committed_repaired_step: Optional[bool]
    committed_repair_min_clearance: Optional[float]
    committed_repair_clearance_gain: Optional[float]
    recover_steps_executed: Optional[int]
    deform_steps_executed: Optional[int]
    resume_from_committed_rejoin: Optional[bool]
    request_action_history_reset_after_recovery: Optional[bool]
    recovery_action_history_reset: Optional[bool]
    recovery_action_history_reset_count: Optional[int]
    committed_abort_step: Optional[int]
    committed_abort_mode: Optional[str]
    committed_abort_index: Optional[int]
    committed_abort_chunk_length: Optional[int]
    committed_abort_action: Optional[list[float]]
    committed_abort_min_clearance: Optional[float]
    committed_abort_required_clearance: Optional[float]
    committed_abort_clearance_gap: Optional[float]
    committed_abort_human_state: Optional[Any]
    committed_abort_robot_q: Optional[list[float]]
    committed_abort_robot_qd: Optional[list[float]]
    committed_abort_reason: Optional[str]
    planned_min_clearance_at_index: Optional[float]
    planned_h_at_index: Optional[float]
    planned_q_at_index: Optional[list[float]]
    planned_action_at_index: Optional[list[float]]
    planned_vs_actual_q_error: Optional[float]
    planned_vs_actual_action_error: Optional[float]
    actual_one_step_clearance: Optional[float]
    planned_clearance_for_this_index: Optional[float]
    clearance_prediction_error: Optional[float]
    planned_pre_action_q: Optional[list[float]]
    planned_post_action_q: Optional[list[float]]
    predicted_post_action_q: Optional[list[float]]
    actual_pre_action_q: Optional[list[float]]
    replay_predicted_post_action_q: Optional[list[float]]
    committed_action: Optional[list[float]]
    planned_clearance_pre: Optional[float]
    planned_clearance_post: Optional[float]
    replay_clearance_pre: Optional[float]
    replay_clearance_post: Optional[float]
    actual_vs_planned_pre_q_error: Optional[float]
    actual_vs_planned_post_q_error: Optional[float]
    planning_vs_replay_human_error: Optional[float]
    planning_vs_replay_clearance_pre_error: Optional[float]
    planning_vs_replay_clearance_post_error: Optional[float]
    planning_human_state_snapshot: Optional[Any]
    replay_human_state: Optional[Any]
    control_type: Optional[str]
    dt: Optional[float]
    controlled_state_indices: Optional[list[int]]
    controlled_action_indices: Optional[list[int]]
    action_conversion_mode: Optional[str]
    human_motion_since_plan: Optional[float]
    accepted_min_clearance: Optional[float]
    accepted_clearance_margin: Optional[float]
    committed_abort_due_to_human_motion: Optional[bool]
    committed_abort_due_to_prediction_error: Optional[bool]
    committed_abort_due_to_safety_semantics_mismatch: Optional[bool]
    committed_state_error: Optional[float]
    committed_state_error_threshold: Optional[float]
    committed_aborted_due_to_state_mismatch: Optional[bool]
    committed_replan_due_to_state_mismatch: Optional[bool]
    committed_rejected_missing_planned_q: Optional[bool]
    actual_q_at_replay: Optional[list[float]]
    diagnostic_step_mode: Optional[str]
    mode_transition: Optional[str]
    act_step: Optional[bool]
    deform_step: Optional[bool]
    recover_step: Optional[bool]
    brake_step: Optional[bool]
    fallback_step: Optional[bool]
    optimized_attempt_step: Optional[bool]
    optimized_accepted_step: Optional[bool]
    unsafe_streak: Optional[int]
    brake_streak: Optional[int]
    recovery_failure_streak: Optional[int]
    recovery_failure_streak_max: Optional[int]
    temporary_blocker_waiting: Optional[bool]
    deform_trigger_reason: Optional[str]
    nominal_became_safe_after_brake: Optional[bool]
    resume_act_after_wait: Optional[bool]
    temporary_wait_step: Optional[bool]
    deform_suppressed_by_temporary_wait: Optional[bool]
    deform_after_persistent_block: Optional[bool]
    deform_replan_count: Optional[int]
    recover_replan_count: Optional[int]
    recovery_replan_count: Optional[int]
    recovery_target_feasible: Optional[bool]
    stale_recovery_attempted: Optional[bool]
    stale_recovery_suppressed_count: Optional[int]
    recovery_target_infeasible_count: Optional[int]
    recover_to_task_progress: Optional[bool]
    recover_anchor_is_current: Optional[bool]
    deform_anchor_is_current: Optional[bool]
    emergency_brake_steps: Optional[int]
    emergency_brake_immediate_unsafe: Optional[bool]
    optimized_candidate_count: Optional[int]
    optimized_solution_count: Optional[int]
    fallback_candidate_count: Optional[int]
    fallback_candidate_accepted_count: Optional[int]
    candidate_fallback_enabled: Optional[bool]
    optimized_rejected_count: Optional[int]
    deform_candidate_count: Optional[int]
    deform_accepted_count: Optional[int]
    deform_rejected_count: Optional[int]
    recover_candidate_count: Optional[int]
    recover_accepted_count: Optional[int]
    recover_rejected_count: Optional[int]
    safe_corridor_recovery_count: Optional[int]
    direct_rejoin_attempt_count: Optional[int]
    direct_rejoin_reject_count: Optional[int]
    detour_rejoin_attempt_count: Optional[int]
    detour_rejoin_accept_count: Optional[int]
    delayed_rejoin_count: Optional[int]
    recover_path_unsafe_count: Optional[int]
    recovery_path_failure_streak_max: Optional[int]
    repeated_unsafe_target_count: Optional[int]
    post_recovery_act_window_count: Optional[int]
    post_recovery_act_window_interrupted_count: Optional[int]
    mean_recover_path_min_clearance: Optional[float]
    min_recover_path_min_clearance: Optional[float]
    safe_prefix_accepted_count: Optional[int]
    first_action_only_accepted_count: Optional[int]
    immediate_hard_reject_count: Optional[int]
    no_safe_prefix_reject_count: Optional[int]
    horizon_margin_reject_count: Optional[int]
    accepted_deform_steps: Optional[int]
    accepted_recover_steps: Optional[int]
    fallback_brake_after_reject_count: Optional[int]
    accepted_candidate_type: Optional[str]
    accepted_candidate_name: Optional[str]
    acceptance_type: Optional[str]
    safe_prefix_len: Optional[int]
    immediate_clearance: Optional[float]
    prefix_min_clearance: Optional[float]
    horizon_min_clearance: Optional[float]
    full_horizon_required: Optional[bool]
    rolling_replan_on_prefix: Optional[bool]
    safe_prefix_execution: Optional[bool]
    recover_projection_on_nominal: Optional[float]
    recover_cosine_to_nominal: Optional[float]
    nominal_rejoin_score: Optional[float]
    nominal_rejoin_available: Optional[bool]
    nominal_rejoin_suppressed_reason: Optional[str]
    nominal_rejoin_clearance: Optional[float]
    nominal_rejoin_safe_prefix_len: Optional[int]
    recover_task_progress_score: Optional[float]
    recover_score_total: Optional[float]
    recover_rejoin_weight_effective: Optional[float]
    recover_step_since_deform: Optional[int]
    nominal_rejoin_available_count: Optional[int]
    nominal_rejoin_suppressed_count: Optional[int]
    stale_nominal_rejoin_suppressed_count: Optional[int]
    nominal_prefix_unsafe_suppressed_count: Optional[int]
    recover_positive_projection_count: Optional[int]
    recover_nonpositive_projection_count: Optional[int]
    mean_recover_projection_on_nominal: Optional[float]
    mean_recover_cosine_to_nominal: Optional[float]
    mean_recover_task_progress_score: Optional[float]
    contact_during_hold: Optional[bool]
    contact_during_brake: Optional[bool]
    contact_during_deform: Optional[bool]
    contact_during_recover: Optional[bool]
    chosen_action_norm: Optional[float]
    controlled_action_delta_norm: Optional[float]
    arm_delta_norm: Optional[float]
    gripper_latched: Optional[bool]
    gripper_latch_dim: Optional[int]
    safe_gripper_action: Optional[float]
    raw_gripper_action: Optional[float]
    phase_reanchor_steps_left: Optional[int]
    phase_reanchor_base_cmd_xy: Optional[list[float]]
    phase_reanchor_ee_error_xy: Optional[list[float]]
    phase_reanchor_drawer_fraction: Optional[float]
    phase_reanchor_ee_to_handle_dist: Optional[float]
    post_recovery_task_guard_active: Optional[bool]
    post_recovery_task_guard_steps_left: Optional[int]
    post_recovery_task_guard_reason: Optional[str]
    post_recovery_task_guard_best_progress: Optional[float]
    post_recovery_progress_regression: Optional[float]
    post_recovery_reanchor_started: Optional[bool]
    hold_immediate_clearance: Optional[float]
    hold_horizon_min_clearance: Optional[float]
    hold_acceptance_type: Optional[str]
    hold_rejected_reason: Optional[str]
    hold_predicted_contact: Optional[bool]
    human_prediction_available: Optional[bool]
    human_velocity_toward_robot: Optional[float]
    human_motion_prediction_enabled: Optional[bool]
    human_motion_prediction_available: Optional[bool]
    human_motion_prediction_speed: Optional[float]
    human_motion_prediction_max_displacement: Optional[float]
    emergency_deform_away: Optional[bool]
    emergency_deform_away_steps: Optional[int]
    emergency_deform_away_count: Optional[int]
    hold_unsafe_count: Optional[int]
    hold_predicted_contact_count: Optional[int]
    contact_during_hold_count: Optional[int]
    contact_during_brake_count: Optional[int]
    contact_during_deform_count: Optional[int]
    contact_during_recover_count: Optional[int]
    mean_hold_horizon_min_clearance: Optional[float]
    min_hold_horizon_min_clearance: Optional[float]

    elapsed_wall_time_s: float
    step_wall_time_s: float

    filter_time_ms: float
    monitor_time_ms: float


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        choices=["act", "oscbf", "sequential", "sequential_oscbf", "chunk_deform"],
        required=True,
        help=("act = monitor only; oscbf = single-action OSCBF; "
              "sequential/sequential_oscbf = apply OSCBF across the ACT chunk; "
              "chunk_deform = SafeChunk horizon deformation."),
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
    parser.add_argument(
        "--robot-spawn-offset-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("DX", "DY"),
        help=(
            "Eval-only offset added to the task default RESET_ROBOT_POS x/y "
            "before each reset. Useful for spawn-location robustness sweeps."
        ),
    )
    parser.add_argument(
        "--normalization-source",
        choices=["auto", "eval", "snapshot"],
        default="auto",
        help=(
            "Where to compute demo-based action min/max and low-dim observation "
            "normalization stats. 'auto' uses snapshot stats when the eval "
            "task/manifest/demo source differs from the checkpoint training config."
        ),
    )
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
        "--continue-after-success",
        action="store_true",
        help=(
            "Stress-test mode: keep stepping/recording after task success until "
            "termination, truncation, step limit, or video limit. Useful when the "
            "collision window occurs just after the nominal task success trigger."
        ),
    )
    parser.add_argument(
        "--gripper-latch",
        action="store_true",
        help=(
            "Eval-only helper: once the selected gripper command crosses the "
            "trigger threshold, hold that gripper closed for the rest of the "
            "episode. Useful for diagnosing drawer-pull slip failures."
        ),
    )
    parser.add_argument(
        "--gripper-latch-dim",
        type=int,
        default=-1,
        help="Action dimension to latch. For BiGym DrawerTopOpen, -1 is the right gripper.",
    )
    parser.add_argument(
        "--gripper-latch-trigger",
        type=float,
        default=0.5,
        help="Latch activates when the selected normalized gripper action is >= this value.",
    )
    parser.add_argument(
        "--gripper-latch-value",
        type=float,
        default=1.0,
        help="Normalized action value to write to the latched gripper dimension.",
    )
    parser.add_argument(
        "--gripper-latch-start-step",
        type=int,
        default=0,
        help="Do not allow the latch to activate before this outer eval step.",
    )
    parser.add_argument(
        "--disable-human-arm-collisions",
        action="store_true",
        help="Disable collision geoms for BiGym's cylinder human arm in the eval env.",
    )
    parser.add_argument(
        "--enable-human-arm-collisions",
        action="store_true",
        help=(
            "Force BiGym's cylinder human arm geoms to physical MuJoCo contact "
            "masks in the eval env."
        ),
    )
    parser.add_argument(
        "--visual-only-human-arm",
        action="store_true",
        help=(
            "Render and update the human arm, but disable its physical MuJoCo contact "
            "response. Barrier/proximity metrics still provide the collision check."
        ),
    )
    parser.add_argument(
        "--freeze-human-arm",
        action="store_true",
        help="Freeze BiGym's cylinder human arm joints and carrier motion after reset.",
    )
    parser.add_argument(
        "--human-arm-aggression",
        type=float,
        default=1.0,
        help=(
            "Eval-only multiplier for scripted human-arm speed/amplitude. "
            "Values >1 make the arm more active; defaults to unchanged behavior."
        ),
    )
    parser.add_argument(
        "--human-arm-substeps",
        type=int,
        default=1,
        help=(
            "Number of scripted human-arm updates per outer eval step. "
            "Use >1 to make the human move faster relative to the robot."
        ),
    )
    parser.add_argument(
        "--human-arm-zero-dwell",
        action="store_true",
        help="Remove most scripted human-arm dwell/pauses after reset.",
    )
    parser.add_argument(
        "--human-arm-walk-radius",
        type=float,
        default=None,
        help="Optional carrier walk radius override for the scripted human arm.",
    )
    parser.add_argument(
        "--human-arm-goal-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help=(
            "Optional carrier goal in the human-arm local XY walk disk. "
            "When set, the aggressive eval keeps biasing the carrier toward this point."
        ),
    )
    parser.add_argument(
        "--human-arm-keepout-min-clear",
        type=float,
        default=None,
        help=(
            "Optional override for the human arm's internal carrier keepout MIN_CLEAR. "
            "This keeps the aggressive arm near the robot without allowing deep penetration."
        ),
    )
    parser.add_argument(
        "--human-arm-disable-keepout",
        action="store_true",
        help=(
            "Disable the scripted human arm's internal robot-avoidance keepout. "
            "MuJoCo collisions remain enabled, so this is useful for contact stress tests."
        ),
    )
    parser.add_argument(
        "--human-arm-ee-obstruction",
        action="store_true",
        help=(
            "During --human-arm-transient-obstruction, place the human-arm "
            "carrier near the robot end-effector XY before releasing it."
        ),
    )
    parser.add_argument(
        "--human-arm-ee-side-sweep",
        action="store_true",
        help=(
            "With --human-arm-ee-obstruction, sweep the human arm sideways "
            "across the end-effector collision zone instead of parking it."
        ),
    )
    parser.add_argument(
        "--human-arm-drawer-obstruction",
        action="store_true",
        help=(
            "During a transient obstruction, place the human-arm carrier between "
            "the robot and drawer and move it locally, instead of anchoring "
            "it to the robot end-effector."
        ),
    )
    parser.add_argument(
        "--human-arm-drawer-obstruction-xy",
        type=float,
        nargs=2,
        default=[-0.50, 0.20],
        metavar=("X", "Y"),
        help="Carrier XY near the drawer area used by --human-arm-drawer-obstruction.",
    )
    parser.add_argument(
        "--human-arm-drawer-obstruction-amp-xy",
        type=float,
        nargs=2,
        default=[0.03, 0.16],
        metavar=("AX", "AY"),
        help=(
            "Local carrier sweep around --human-arm-drawer-obstruction-xy "
            "while the arm temporarily blocks the robot-to-drawer path."
        ),
    )
    parser.add_argument(
        "--human-arm-yaw-offset-deg",
        type=float,
        default=90.0,
        help=(
            "Extra shoulder-yaw offset for the scripted human arm in degrees. "
            "Positive values rotate anticlockwise in the top-down view."
        ),
    )
    parser.add_argument(
        "--human-arm-ee-offset-xy",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("DX", "DY"),
        help="XY offset added to the robot end-effector obstruction anchor.",
    )
    parser.add_argument(
        "--human-arm-ee-side-sweep-amp-xy",
        type=float,
        nargs=2,
        default=[0.03, 0.30],
        metavar=("AX", "AY"),
        help="Sideways sweep amplitude around the end-effector obstruction anchor.",
    )
    parser.add_argument(
        "--human-arm-ee-side-sweep-frequency",
        type=float,
        default=0.35,
        help="Sideways sweep frequency in cycles per second.",
    )
    parser.add_argument(
        "--human-arm-ee-side-sweep-phase",
        type=float,
        default=2.0,
        help="Initial side-sweep phase in radians, used to enter the EE zone early at slow speed.",
    )
    parser.add_argument(
        "--human-arm-force-carrier-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help=(
            "Force the scripted human-arm carrier to this XY position after reset "
            "and after each step. Useful for controlled contact stress tests."
        ),
    )
    parser.add_argument(
        "--human-arm-force-carrier-amp-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("AX", "AY"),
        help=(
            "Oscillation amplitude around --human-arm-force-carrier-xy. "
            "Use with the forced carrier option to make contact tests move around."
        ),
    )
    parser.add_argument(
        "--human-arm-force-carrier-frequency",
        type=float,
        default=0.35,
        help="Carrier oscillation frequency in cycles per second for forced contact tests.",
    )
    parser.add_argument(
        "--human-arm-natural-contact-motion",
        action="store_true",
        help=(
            "Override the human-arm joints with a bounded reach/sweep motion for contact tests, "
            "avoiding unnatural shoulder/base spinning."
        ),
    )
    parser.add_argument(
        "--human-arm-natural-motion-frequency",
        type=float,
        default=0.45,
        help="Natural contact joint-motion frequency in cycles per second.",
    )
    parser.add_argument(
        "--human-arm-natural-motion-phase-offset-steps",
        type=float,
        default=50.0,
        help=(
            "Eval-only phase lead for --human-arm-natural-contact-motion. "
            "Positive values make the arm reach the drawer lane earlier in the episode."
        ),
    )
    parser.add_argument(
        "--human-arm-natural-lateral-scale",
        type=float,
        default=0.35,
        help=(
            "Scale for side-raise / lateral shoulder motion in natural human-arm motion. "
            "Lower values keep the upper arm more tucked."
        ),
    )
    parser.add_argument(
        "--human-arm-natural-return-curl-scale",
        type=float,
        default=1.0,
        help=(
            "Scale for elbow curl during the return/retraction phase of natural motion. "
            "Higher values make the arm come back in a more human-like curled pose."
        ),
    )
    parser.add_argument(
        "--human-arm-transient-obstruction",
        action="store_true",
        help=(
            "Start the human arm in the robot/drawer operation area, then move it "
            "away after a short obstruction window so the robot can finish."
        ),
    )
    parser.add_argument(
        "--human-arm-release-after-steps",
        type=int,
        default=180,
        help="For transient obstruction, begin moving the human arm away after this eval step.",
    )
    parser.add_argument(
        "--human-arm-release-duration-steps",
        type=int,
        default=60,
        help="For transient obstruction, number of eval steps used to move out of the contact zone.",
    )
    parser.add_argument(
        "--human-arm-release-carrier-xy",
        type=float,
        nargs=2,
        default=[-0.85, 0.55],
        metavar=("X", "Y"),
        help="Carrier XY target after the transient obstruction moves away.",
    )
    parser.add_argument(
        "--human-arm-final-clear-after-steps",
        type=int,
        default=-1,
        help=(
            "Eval-only late-phase override. When >= 0, begin moving the human "
            "arm carrier and joints out of the robot/drawer collision zone at "
            "this eval step."
        ),
    )
    parser.add_argument(
        "--human-arm-final-clear-duration-steps",
        type=int,
        default=20,
        help="Number of eval steps used by the late-phase human-arm clear override.",
    )
    parser.add_argument(
        "--human-arm-final-clear-trigger",
        choices=["step", "carrier-y-peak"],
        default="carrier-y-peak",
        help=(
            "Trigger the late clear at the fixed step, or at the first forced "
            "carrier-Y local maximum after the obstruction sweep starts. "
            "If no peak is observed, the fixed step is used as a fallback."
        ),
    )
    parser.add_argument(
        "--human-arm-final-clear-max-carrier-speed",
        type=float,
        default=0.35,
        help="Maximum carrier XY speed used by the late human-arm clear override.",
    )
    parser.add_argument(
        "--human-arm-final-clear-max-joint-speed",
        type=float,
        default=1.3,
        help="Maximum joint-space speed used by the late human-arm clear override.",
    )
    parser.add_argument(
        "--human-arm-final-clear-carrier-xy",
        type=float,
        nargs=2,
        default=[-0.85, 0.55],
        metavar=("X", "Y"),
        help="Carrier XY target for --human-arm-final-clear-after-steps.",
    )
    parser.add_argument(
        "--policy-env",
        type=str,
        default=None,
        help=(
            "Optional clean shadow env used only for ACT policy observations. "
            "Actions still execute in --env, so safety/video/contacts use the real env. "
            "For human-arm safety eval, prefer --hide-human-arm-policy-obs first."
        ),
    )
    parser.add_argument(
        "--safety-env",
        type=str,
        default=None,
        help=(
            "Optional human/safety mirror env. --env remains the task plant that "
            "executes actions; this env receives mirrored robot/drawer state and is "
            "used for OSCBF human capsules, contact counts, and video."
        ),
    )
    parser.add_argument(
        "--hide-human-arm-policy-obs",
        action="store_true",
        help=(
            "Use the real eval env state, but hide cylinder-arm geoms while "
            "rendering RGB observations for ACT. Video/safety still see the normal env."
        ),
    )
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Disable video recording.",
    )
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument(
        "--record-policy-video",
        action="store_true",
        help=(
            "Save a video of the RGB observations passed to ACT after any "
            "policy-observation modifications, e.g. hidden human arm."
        ),
    )
    parser.add_argument(
        "--policy-video-every",
        type=int,
        default=1,
        help="Record every N policy-observation frames when --record-policy-video is set.",
    )
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
        "--video-time-base",
        choices=["sim", "wall"],
        default="sim",
        help=(
            "Video timing. 'sim' saves fixed-FPS videos by env step, so slow filters "
            "do not make playback look laggy. 'wall' preserves real evaluation latency."
        ),
    )
    parser.add_argument(
        "--stop-video-at-steps",
        type=int,
        default=None,
        help=(
            "Stop each episode after this many recorded env steps. Only used for "
            "--video-time-base sim. If unset, --stop-video-at is converted to steps "
            "using the video FPS."
        ),
    )
    parser.add_argument(
        "--plot-terminal",
        action="store_true",
        help="Render ASCII terminal plots for step metrics after each episode.",
    )
    parser.add_argument(
        "--log-chunk-trajectories",
        dest="log_chunk_trajectories",
        action="store_true",
        default=True,
        help=(
            "Log 3D SafeChunk trajectory traces for intervention steps. Writes "
            "chunk_trajectory_traces.jsonl and human_arm_trajectory.jsonl under "
            "--output-dir."
        ),
    )
    parser.add_argument(
        "--no-log-chunk-trajectories",
        dest="log_chunk_trajectories",
        action="store_false",
        help="Disable SafeChunk trajectory trace logging.",
    )
    parser.add_argument(
        "--chunk-trajectory-max-events",
        type=int,
        default=300,
        help="Maximum number of chunk trajectory intervention events to store.",
    )
    parser.add_argument(
        "--chunk-trajectory-include-q-states",
        dest="chunk_trajectory_include_q_states",
        action="store_true",
        default=True,
        help="Include rolled-out q-state arrays in chunk_trajectory_traces.jsonl.",
    )
    parser.add_argument(
        "--no-chunk-trajectory-q-states",
        dest="chunk_trajectory_include_q_states",
        action="store_false",
        help="Only log 3D EE trajectories, not full rolled-out q-state arrays.",
    )
    parser.add_argument(
        "--plot-chunk-trajectories-3d",
        dest="plot_chunk_trajectories_3d",
        action="store_true",
        default=True,
        help="Save per-episode 3D trajectory plots for chunk interventions.",
    )
    parser.add_argument(
        "--no-plot-chunk-trajectories-3d",
        dest="plot_chunk_trajectories_3d",
        action="store_false",
        help="Disable per-episode 3D trajectory plot images.",
    )
    parser.add_argument(
        "--chunk-trajectory-plot-max-events",
        type=int,
        default=25,
        help="Maximum chunk intervention events drawn in each 3D trajectory plot.",
    )
    parser.add_argument(
        "--human-arm-trajectory-stride",
        type=int,
        default=1,
        help="Record every Nth human-arm trajectory sample for the graph/log.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars for episodes and steps.",
    )
    parser.add_argument(
        "--diagnostics",
        dest="diagnostics_enabled",
        action="store_true",
        default=True,
        help="Enable failure-diagnosis logging in step and summary JSON.",
    )
    parser.add_argument(
        "--no-diagnostics",
        dest="diagnostics_enabled",
        action="store_false",
        help="Disable failure-diagnosis logging fields.",
    )
    parser.add_argument("--diagnostics-large-arm-delta-threshold", type=float, default=3.0)
    parser.add_argument("--diagnostics-large-base-delta-threshold", type=float, default=0.5)
    parser.add_argument("--diagnostics-low-act-ratio-threshold", type=float, default=0.3)
    parser.add_argument("--diagnostics-high-fallback-ratio-threshold", type=float, default=0.5)

    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=None,
        help="Optional max per-dimension OSCBF action edit.",
    )
    parser.add_argument(
        "--oscbf-human-margin",
        type=float,
        default=0.08,
        help="OSCBF human capsule inflation margin.",
    )
    parser.add_argument(
        "--oscbf-alpha-gain",
        type=float,
        default=10.0,
        help="OSCBF barrier gain; larger values react more aggressively.",
    )
    parser.add_argument(
        "--oscbf-pelvis-velocity-limits",
        type=float,
        nargs=4,
        default=[0.6, 0.6, 0.4, 1.5],
        metavar=("VX", "VY", "VZ", "WYAW"),
        help="Pelvis velocity limits for the augmented pelvis+arm OSCBF.",
    )
    parser.add_argument(
        "--oscbf-pelvis-weight",
        type=float,
        default=0.5,
        help="QP objective weight for pelvis controls in augmented OSCBF.",
    )
    parser.add_argument(
        "--oscbf-arm-weight",
        type=float,
        default=1.0,
        help="QP objective weight for arm controls in augmented OSCBF.",
    )
    parser.add_argument(
        "--pause-on-unsafe",
        action="store_true",
        help=(
            "If current/horizon clearance is unsafe, hold controlled arm joints "
            "at current q. Gripper and non-arm action dimensions remain bypassed."
        ),
    )
    parser.add_argument(
        "--pause-clearance-threshold",
        type=float,
        default=0.0,
        help="Pause fallback threshold on OSCBF clearance h. Use 0 for collision/violation only.",
    )
    parser.add_argument(
        "--no-pause-policy-step-on-brake",
        action="store_false",
        dest="pause_policy_step_on_brake",
        help="Keep advancing the ACT policy step counter even when the robot arm is held/braked.",
    )
    parser.set_defaults(pause_policy_step_on_brake=True)
    parser.add_argument(
        "--reset-action-history-after-human-exit",
        action="store_true",
        help=(
            "After a temporary human blocker reaches done and clears the robot, "
            "reset receding-horizon action history so ACT resumes from fresh observations."
        ),
    )
    parser.add_argument(
        "--reset-action-history-after-recovery",
        dest="reset_action_history_after_recovery",
        action="store_true",
        default=True,
        help="Reset ACT temporal action history after committed SafeChunk recovery completes.",
    )
    parser.add_argument(
        "--no-reset-action-history-after-recovery",
        dest="reset_action_history_after_recovery",
        action="store_false",
        help="Do not clear ACT temporal action history after committed SafeChunk recovery.",
    )
    parser.add_argument(
        "--pause-and-restart-on-human-blocker",
        action="store_true",
        help=(
            "Ablation: hold controlled robot motion while the temporary human "
            "blocker is in enter/hold/exit, then reset ACT/action-sequence "
            "history after the blocker reaches done and clears."
        ),
    )
    parser.add_argument(
        "--initial-pause-restart-steps",
        type=int,
        default=0,
        help=(
            "Ablation: hold controlled robot motion for the first N eval steps, "
            "then reset ACT/action-sequence history and continue. Use this to "
            "test whether a pure delay breaks the trained ACT rollout."
        ),
    )
    parser.add_argument(
        "--pause-motion-scale",
        type=float,
        default=0.0,
        help=(
            "Ablation: during pause/restart holds, execute this fraction of the "
            "ACT controlled motion instead of a hard hold. 0.0 is a full pause; "
            "0.2 keeps 20 percent of the nominal motion."
        ),
    )
    parser.add_argument(
        "--resume-clearance-threshold",
        type=float,
        default=0.08,
        help="Minimum robot-human distance before resetting action history after human exit.",
    )
    parser.add_argument(
        "--resume-clear-steps",
        type=int,
        default=3,
        help="Consecutive clear done-phase steps required before action-history reset.",
    )
    parser.add_argument(
        "--phase-reanchor",
        action="store_true",
        help=(
            "When drawer progress stalls, pause ACT, run a short handle-relative "
            "base re-anchor primitive for the nearest useful task phase, then "
            "reset ACT action history and resume."
        ),
    )
    parser.add_argument(
        "--phase-reanchor-check-after-steps",
        type=int,
        default=70,
        help="Earliest eval step at which stalled drawer progress can trigger re-anchor.",
    )
    parser.add_argument(
        "--phase-reanchor-no-progress-window",
        type=int,
        default=20,
        help="Recent step window used to decide whether drawer progress has stalled.",
    )
    parser.add_argument(
        "--phase-reanchor-min-drawer-progress",
        type=float,
        default=0.02,
        help="Minimum drawer-open fraction increase over the stall window.",
    )
    parser.add_argument(
        "--phase-reanchor-steps",
        type=int,
        default=24,
        help="Number of env steps to execute each re-anchor primitive.",
    )
    parser.add_argument(
        "--phase-reanchor-cooldown-steps",
        type=int,
        default=50,
        help="Minimum steps after a re-anchor before another one can start.",
    )
    parser.add_argument(
        "--phase-reanchor-base-gain",
        type=float,
        default=0.45,
        help="Proportional gain from hand-anchor XY error to floating-base XY command.",
    )
    parser.add_argument(
        "--phase-reanchor-max-base-step",
        type=float,
        default=0.012,
        help="Raw per-step XY floating-base command limit during re-anchor.",
    )
    parser.add_argument(
        "--phase-reanchor-pregrasp-offset-xy",
        type=float,
        nargs=2,
        default=[-0.12, -0.06],
        metavar=("DX", "DY"),
        help="Desired end-effector XY offset from the handle in pre-grasp phase.",
    )
    parser.add_argument(
        "--phase-reanchor-grasp-offset-xy",
        type=float,
        nargs=2,
        default=[-0.03, 0.0],
        metavar=("DX", "DY"),
        help="Desired end-effector XY offset from the handle in grasp phase.",
    )
    parser.add_argument(
        "--phase-reanchor-pull-offset-xy",
        type=float,
        nargs=2,
        default=[0.0, -0.10],
        metavar=("DX", "DY"),
        help="Desired end-effector XY offset from the handle once pulling has started.",
    )
    parser.add_argument(
        "--phase-reanchor-grasp-dist",
        type=float,
        default=0.12,
        help="EE-handle XY distance below which the recovery phase is considered grasp.",
    )
    parser.add_argument(
        "--phase-reanchor-pull-open-threshold",
        type=float,
        default=0.15,
        help="Drawer-open fraction above which the recovery phase is considered pull.",
    )
    parser.add_argument(
        "--phase-reanchor-done-threshold",
        type=float,
        default=0.90,
        help="Drawer-open fraction above which no re-anchor is attempted.",
    )
    parser.add_argument(
        "--phase-reanchor-gripper-closed-threshold",
        type=float,
        default=0.5,
        help="Raw gripper qpos threshold used by the phase classifier.",
    )
    parser.add_argument(
        "--phase-reanchor-gripper-value",
        type=float,
        default=1.0,
        help="Normalized gripper command used during grasp/pull re-anchor phases.",
    )
    parser.add_argument(
        "--post-recovery-task-guard",
        action="store_true",
        help=(
            "After committed SafeChunk recovery completes, keep the gripper "
            "closed and optionally run a short handle-relative re-anchor "
            "before passing ACT through."
        ),
    )
    parser.add_argument(
        "--post-recovery-task-guard-steps",
        type=int,
        default=24,
        help="Number of post-recovery steps protected by the task guard.",
    )
    parser.add_argument(
        "--post-recovery-progress-tolerance",
        type=float,
        default=1e-5,
        help=(
            "Allowed task-progress drop before the post-recovery guard treats "
            "the drawer as regressing and re-runs re-anchor."
        ),
    )
    parser.add_argument(
        "--post-recovery-task-guard-min-progress",
        type=float,
        default=1e-6,
        help="Minimum task progress required to arm the post-recovery task guard.",
    )
    parser.add_argument(
        "--post-recovery-task-guard-max-ee-distance",
        type=float,
        default=0.15,
        help=(
            "Also arm the post-recovery task guard when the EE-handle "
            "distance is at or below this value. Non-positive disables this gate."
        ),
    )
    parser.add_argument(
        "--post-recovery-task-guard-reanchor-phases",
        nargs="+",
        default=["grasp"],
        choices=["pre_grasp", "grasp", "pull"],
        help=(
            "Task phases where the post-recovery task guard may override ACT "
            "with the base re-anchor primitive. The default avoids pre-grasp "
            "and pull so a closed gripper does not reel the robot toward the drawer."
        ),
    )
    parser.add_argument(
        "--post-recovery-task-guard-check-safety",
        dest="post_recovery_task_guard_check_safety",
        action="store_true",
        default=True,
        help="Run SafeChunk horizon acceptance before executing post-recovery re-anchor actions.",
    )
    parser.add_argument(
        "--no-post-recovery-task-guard-check-safety",
        dest="post_recovery_task_guard_check_safety",
        action="store_false",
        help="Bypass SafeChunk acceptance for post-recovery re-anchor actions.",
    )
    parser.add_argument(
        "--post-recovery-task-guard-force-gripper",
        dest="post_recovery_task_guard_force_gripper",
        action="store_true",
        default=True,
        help="Force the selected gripper command closed while the post-recovery task guard is active.",
    )
    parser.add_argument(
        "--no-post-recovery-task-guard-force-gripper",
        dest="post_recovery_task_guard_force_gripper",
        action="store_false",
        help="Do not force gripper closure during the post-recovery task guard.",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--brake-progress-threshold",
        type=float,
        default=0.05,
        help="Minimum chunk progress before braking is treated as non-deadlocked.",
    )
    parser.add_argument(
        "--deadlock-window",
        type=int,
        default=5,
        help="Consecutive deadlocked brake steps before horizon deformation is allowed.",
    )
    parser.add_argument("--chunk-min-clearance", type=float, default=0.12)
    parser.add_argument(
        "--chunk-deformation-enabled",
        dest="chunk_deformation_enabled",
        action="store_true",
        default=True,
        help="Allow SafeChunk-Deform to replace unsafe chunks with deformed chunks.",
    )
    parser.add_argument(
        "--no-chunk-deformation-enabled",
        dest="chunk_deformation_enabled",
        action="store_false",
        help="Disable chunk deformation and use the configured brake/fallback behavior.",
    )
    parser.add_argument(
        "--chunk-deformation-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
    )
    parser.add_argument("--chunk-deformation-smoothing", type=int, default=1)
    parser.add_argument(
        "--unsafe-deformation-fallback",
        choices=["brake", "best"],
        default="brake",
        help=(
            "When no safe chunk deformation is found, either brake/hold the "
            "controlled arm joints or keep the best unsafe deformed candidate."
        ),
    )
    parser.add_argument(
        "--deform-immediately-on-deadlock",
        action="store_true",
        help="Legacy behavior: allow deformation on the first deadlocked brake step.",
    )
    parser.add_argument(
        "--sequential-oscbf-fallback",
        action="store_true",
        help="Allow chunk_deform to fall back to sequential OSCBF if no candidate is safe.",
    )
    parser.add_argument(
        "--chunk-deform-mode",
        choices=["candidate", "optimized"],
        default="candidate",
        help="SafeChunk deformation mode. Candidate preserves the original fixed-scale search.",
    )
    parser.add_argument("--chunk-opt-iters", type=int, default=20)
    parser.add_argument("--chunk-opt-lr", type=float, default=0.03)
    parser.add_argument("--chunk-lambda-safety", type=float, default=500.0)
    parser.add_argument("--chunk-lambda-action", type=float, default=0.1)
    parser.add_argument("--chunk-lambda-path", type=float, default=0.2)
    parser.add_argument("--chunk-lambda-rejoin", type=float, default=0.5)
    parser.add_argument("--chunk-lambda-smooth", type=float, default=0.1)
    parser.add_argument(
        "--chunk-rejoin-threshold",
        type=float,
        default=0.03,
        help="Legacy optimized-deform rejoin threshold; q/EE thresholds are preferred.",
    )
    parser.add_argument("--chunk-min-rejoin-offset", type=int, default=2)
    parser.add_argument(
        "--chunk-inner-rejoin-metric",
        choices=["q_state", "ee_pose"],
        default="q_state",
        help="Metric used inside the CEM objective. q_state is the fast default.",
    )
    parser.add_argument(
        "--chunk-final-rejoin-metric",
        choices=["none", "q_state", "ee_pose"],
        default="q_state",
        help="Optional post-optimization recoverability check metric.",
    )
    parser.add_argument("--chunk-q-rejoin-threshold", type=float, default=0.5)
    parser.add_argument(
        "--chunk-qd-rejoin-threshold",
        type=float,
        default=5.0,
        help="Maximum terminal qdot distance allowed for recovery rejoin.",
    )
    parser.add_argument("--chunk-ee-rejoin-threshold", type=float, default=0.08)
    parser.add_argument(
        "--chunk-explicit-recovery",
        "--chunk-explicit-return",
        dest="chunk_explicit_return",
        action="store_true",
        default=True,
        help="Use two-stage deform + recover optimized SafeChunk-Deform.",
    )
    parser.add_argument(
        "--no-chunk-explicit-recovery",
        "--no-chunk-explicit-return",
        dest="chunk_explicit_return",
        action="store_false",
        help="Use the previous one-stage optimized recoverable deformation.",
    )
    parser.add_argument(
        "--chunk-commit-accepted-chunks",
        dest="chunk_commit_accepted_chunks",
        action="store_true",
        default=True,
        help="Execute accepted explicit-recovery chunks from a persistent buffer.",
    )
    parser.add_argument(
        "--no-chunk-commit-accepted-chunks",
        dest="chunk_commit_accepted_chunks",
        action="store_false",
        help="Preserve legacy behavior: replan after the first accepted action.",
    )
    parser.add_argument(
        "--chunk-committed-chunk-safety-check",
        dest="chunk_committed_chunk_safety_check",
        action="store_true",
        default=True,
        help="Run a cheap one-step safety check before serving committed recovery actions.",
    )
    parser.add_argument(
        "--no-chunk-committed-chunk-safety-check",
        dest="chunk_committed_chunk_safety_check",
        action="store_false",
    )
    parser.add_argument("--chunk-committed-safety-tol", type=float, default=0.005)
    parser.add_argument(
        "--chunk-committed-abort-only-if-contact-risk",
        dest="chunk_committed_abort_only_if_contact_risk",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-chunk-committed-abort-only-if-contact-risk",
        dest="chunk_committed_abort_only_if_contact_risk",
        action="store_false",
    )
    parser.add_argument("--chunk-committed-min-clearance-for-abort", type=float, default=0.08)
    parser.add_argument(
        "--chunk-repair-committed-action",
        dest="chunk_repair_committed_action",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-chunk-repair-committed-action",
        dest="chunk_repair_committed_action",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-monotonic-committed-repair",
        dest="chunk_monotonic_committed_repair",
        action="store_true",
        default=True,
        help="Reject committed-action repair candidates that reduce post-action clearance.",
    )
    parser.add_argument(
        "--no-chunk-monotonic-committed-repair",
        dest="chunk_monotonic_committed_repair",
        action="store_false",
        help="Legacy ablation: apply committed-action repair even if it worsens clearance.",
    )
    parser.add_argument("--chunk-committed-execution-margin", type=float, default=0.02)
    parser.add_argument("--chunk-committed-state-error-threshold", type=float, default=0.25)
    parser.add_argument(
        "--chunk-committed-state-error-action",
        choices=("replan", "abort_to_brake"),
        default="replan",
    )
    parser.add_argument(
        "--chunk-temporary-blocker-enabled",
        dest="chunk_temporary_blocker_enabled",
        action="store_true",
        default=True,
        help="Prefer wait/brake before SafeChunk deformation for temporary blockers.",
    )
    parser.add_argument(
        "--no-chunk-temporary-blocker-enabled",
        dest="chunk_temporary_blocker_enabled",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-temporary-prefer-brake-before-deform",
        dest="chunk_temporary_prefer_brake_before_deform",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-chunk-temporary-prefer-brake-before-deform",
        dest="chunk_temporary_prefer_brake_before_deform",
        action="store_false",
    )
    parser.add_argument("--chunk-temporary-min-unsafe-steps-before-deform", type=int, default=8)
    parser.add_argument("--chunk-temporary-max-brake-steps-before-deform", type=int, default=12)
    parser.add_argument(
        "--chunk-temporary-reset-on-nominal-safe",
        dest="chunk_temporary_reset_on_nominal_safe",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-chunk-temporary-reset-on-nominal-safe",
        dest="chunk_temporary_reset_on_nominal_safe",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-temporary-require-progress-deadlock-before-deform",
        dest="chunk_temporary_require_progress_deadlock_before_deform",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-chunk-temporary-require-progress-deadlock-before-deform",
        dest="chunk_temporary_require_progress_deadlock_before_deform",
        action="store_false",
    )
    parser.add_argument("--chunk-temporary-progress-window", type=int, default=10)
    parser.add_argument("--chunk-temporary-min-progress-delta", type=float, default=0.001)
    parser.add_argument(
        "--chunk-temporary-recover-after-wait",
        dest="chunk_temporary_recover_after_wait",
        action="store_true",
        default=False,
        help="After temporary wait/brake becomes nominal-safe, run existing SafeChunk recovery before releasing ACT.",
    )
    parser.add_argument(
        "--no-chunk-temporary-recover-after-wait",
        dest="chunk_temporary_recover_after_wait",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-temporary-recover-after-wait-min-brake-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chunk-safechunk-replan-enabled",
        dest="chunk_safechunk_replan_enabled",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-chunk-safechunk-replan-enabled",
        dest="chunk_safechunk_replan_enabled",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-recovery-target-mode",
        choices=("task_progress", "nominal"),
        default="task_progress",
    )
    parser.add_argument("--chunk-max-recovery-failure-before-replan", type=int, default=1)
    parser.add_argument("--chunk-acceptance-enabled", dest="chunk_acceptance_enabled", action="store_true", default=True)
    parser.add_argument("--no-chunk-acceptance-enabled", dest="chunk_acceptance_enabled", action="store_false")
    parser.add_argument("--chunk-acceptance-hard-min-clearance", type=float, default=0.02)
    parser.add_argument("--chunk-acceptance-desired-min-clearance", type=float, default=0.08)
    parser.add_argument("--chunk-acceptance-prefix-min-clearance", type=float, default=0.04)
    parser.add_argument("--chunk-acceptance-min-safe-prefix-len", type=int, default=1)
    parser.add_argument("--chunk-allow-safe-prefix-execution", dest="chunk_allow_safe_prefix_execution", action="store_true", default=True)
    parser.add_argument("--no-chunk-allow-safe-prefix-execution", dest="chunk_allow_safe_prefix_execution", action="store_false")
    parser.add_argument("--chunk-rolling-replan-on-prefix", dest="chunk_rolling_replan_on_prefix", action="store_true", default=True)
    parser.add_argument("--no-chunk-rolling-replan-on-prefix", dest="chunk_rolling_replan_on_prefix", action="store_false")
    parser.add_argument("--chunk-full-horizon-required-for-recover", dest="chunk_full_horizon_required_for_recover", action="store_true", default=False)
    parser.add_argument("--chunk-full-horizon-required-for-deform", dest="chunk_full_horizon_required_for_deform", action="store_true", default=False)
    parser.add_argument("--chunk-emergency-brake-if-immediate-below-hard-margin", dest="chunk_emergency_brake_if_immediate_below_hard_margin", action="store_true", default=True)
    parser.add_argument("--no-chunk-emergency-brake-if-immediate-below-hard-margin", dest="chunk_emergency_brake_if_immediate_below_hard_margin", action="store_false")
    parser.add_argument("--chunk-allow-candidate-fallback", dest="chunk_allow_candidate_fallback", action="store_true", default=False, help="Ablation/debug only: allow non-optimized candidate fallback inside optimized SafeChunk.")
    parser.add_argument("--no-chunk-allow-candidate-fallback", dest="chunk_allow_candidate_fallback", action="store_false")
    parser.add_argument("--chunk-candidate-fallback-after-any-optimized-result", dest="chunk_candidate_fallback_only_if_no_optimized_result", action="store_false", default=True)
    parser.add_argument("--chunk-safechunk-recover-enabled", dest="chunk_safechunk_recover_enabled", action="store_true", default=True)
    parser.add_argument("--no-chunk-safechunk-recover-enabled", dest="chunk_safechunk_recover_enabled", action="store_false")
    parser.add_argument("--chunk-recover-rejoin-nominal-weight", type=float, default=5.0)
    parser.add_argument("--chunk-recover-task-progress-weight", type=float, default=10.0)
    parser.add_argument("--chunk-recover-safety-weight", type=float, default=100.0)
    parser.add_argument("--chunk-recover-action-deviation-weight", type=float, default=0.2)
    parser.add_argument("--chunk-recover-smoothness-weight", type=float, default=0.1)
    parser.add_argument("--chunk-recover-require-nominal-prefix-safe", dest="chunk_recover_require_nominal_prefix_safe", action="store_true", default=True)
    parser.add_argument("--no-chunk-recover-require-nominal-prefix-safe", dest="chunk_recover_require_nominal_prefix_safe", action="store_false")
    parser.add_argument("--chunk-recover-nominal-rejoin-prefix-min-clearance", type=float, default=0.04)
    parser.add_argument("--chunk-recover-use-latest-nominal", dest="chunk_recover_use_latest_nominal", action="store_true", default=True)
    parser.add_argument("--no-chunk-recover-use-latest-nominal", dest="chunk_recover_use_latest_nominal", action="store_false")
    parser.add_argument("--chunk-recover-suppress-stale-nominal", dest="chunk_recover_suppress_stale_nominal", action="store_true", default=True)
    parser.add_argument("--no-chunk-recover-suppress-stale-nominal", dest="chunk_recover_suppress_stale_nominal", action="store_false")
    parser.add_argument("--chunk-recover-rejoin-weight-schedule", choices=("ramp", "constant", "none"), default="ramp")
    parser.add_argument("--chunk-recover-rejoin-ramp-steps", type=int, default=5)
    parser.add_argument("--chunk-active-safety-enabled", dest="chunk_active_safety_enabled", action="store_true", default=True)
    parser.add_argument("--no-chunk-active-safety-enabled", dest="chunk_active_safety_enabled", action="store_false")
    parser.add_argument("--chunk-active-check-hold-horizon-safety", dest="chunk_active_check_hold_horizon_safety", action="store_true", default=True)
    parser.add_argument("--no-chunk-active-check-hold-horizon-safety", dest="chunk_active_check_hold_horizon_safety", action="store_false")
    parser.add_argument("--chunk-active-predict-human-motion-for-hold", dest="chunk_active_predict_human_motion_for_hold", action="store_true", default=True)
    parser.add_argument("--no-chunk-active-predict-human-motion-for-hold", dest="chunk_active_predict_human_motion_for_hold", action="store_false")
    parser.add_argument("--chunk-horizon-predict-human-motion", dest="chunk_horizon_predict_human_motion", action="store_true", default=True, help="Propagate human-arm capsules with finite-difference velocity during SafeChunk horizon safety checks.")
    parser.add_argument("--no-chunk-horizon-predict-human-motion", dest="chunk_horizon_predict_human_motion", action="store_false")
    parser.add_argument("--chunk-human-motion-prediction-max-time", type=float, default=0.25, help="Maximum lookahead time, in seconds, for constant-velocity human capsule prediction. Non-positive disables the cap.")
    parser.add_argument("--chunk-human-motion-prediction-max-speed", type=float, default=3.0, help="Maximum endpoint speed, in m/s, used to clip finite-difference human capsule velocities. Non-positive disables clipping.")
    parser.add_argument("--chunk-active-hard-min-clearance", type=float, default=0.02)
    parser.add_argument("--chunk-active-hold-prefix-min-clearance", type=float, default=0.04)
    parser.add_argument("--chunk-active-hold-horizon-steps", type=int, default=4)
    parser.add_argument("--chunk-active-emergency-deform-when-hold-unsafe", dest="chunk_active_emergency_deform_when_hold_unsafe", action="store_true", default=True)
    parser.add_argument("--no-chunk-active-emergency-deform-when-hold-unsafe", dest="chunk_active_emergency_deform_when_hold_unsafe", action="store_false")
    parser.add_argument("--chunk-active-optimize-when-hold-unsafe", dest="chunk_active_optimize_when_hold_unsafe", action="store_true", default=True)
    parser.add_argument("--no-chunk-active-optimize-when-hold-unsafe", dest="chunk_active_optimize_when_hold_unsafe", action="store_false")
    parser.add_argument("--chunk-active-emergency-deform-candidate-scales", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--chunk-active-prefer-last-safe-action", dest="chunk_active_prefer_last_safe_action", action="store_true", default=True)
    parser.add_argument("--no-chunk-active-prefer-last-safe-action", dest="chunk_active_prefer_last_safe_action", action="store_false")
    parser.add_argument("--chunk-active-prefer-last-safe-q-retract", dest="chunk_active_prefer_last_safe_q_retract", action="store_true", default=True)
    parser.add_argument("--no-chunk-active-prefer-last-safe-q-retract", dest="chunk_active_prefer_last_safe_q_retract", action="store_false")
    parser.add_argument("--chunk-active-emergency-deform-replan-next-step", dest="chunk_active_emergency_deform_replan_next_step", action="store_true", default=True)
    parser.add_argument("--no-chunk-active-emergency-deform-replan-next-step", dest="chunk_active_emergency_deform_replan_next_step", action="store_false")
    parser.add_argument("--chunk-recovery-corridor-enabled", dest="chunk_recovery_corridor_enabled", action="store_true", default=True)
    parser.add_argument("--no-chunk-recovery-corridor-enabled", dest="chunk_recovery_corridor_enabled", action="store_false")
    parser.add_argument("--chunk-require-recover-path-safe", dest="chunk_require_recover_path_safe", action="store_true", default=True)
    parser.add_argument("--no-chunk-require-recover-path-safe", dest="chunk_require_recover_path_safe", action="store_false")
    parser.add_argument("--chunk-recover-path-min-clearance", type=float, default=0.04)
    parser.add_argument("--chunk-recover-immediate-hard-clearance", type=float, default=0.02)
    parser.add_argument("--chunk-recover-prefix-min-clearance", type=float, default=0.04)
    parser.add_argument("--chunk-enable-direct-rejoin", dest="chunk_enable_direct_rejoin", action="store_true", default=True)
    parser.add_argument("--no-chunk-enable-direct-rejoin", dest="chunk_enable_direct_rejoin", action="store_false")
    parser.add_argument("--chunk-enable-detour-rejoin", dest="chunk_enable_detour_rejoin", action="store_true", default=True)
    parser.add_argument("--no-chunk-enable-detour-rejoin", dest="chunk_enable_detour_rejoin", action="store_false")
    parser.add_argument("--chunk-enable-delayed-rejoin", dest="chunk_enable_delayed_rejoin", action="store_true", default=True)
    parser.add_argument("--no-chunk-enable-delayed-rejoin", dest="chunk_enable_delayed_rejoin", action="store_false")
    parser.add_argument("--chunk-suppress-repeated-unsafe-recovery", dest="chunk_suppress_repeated_unsafe_recovery", action="store_true", default=True)
    parser.add_argument("--no-chunk-suppress-repeated-unsafe-recovery", dest="chunk_suppress_repeated_unsafe_recovery", action="store_false")
    parser.add_argument("--chunk-unsafe-recovery-cooldown-steps", type=int, default=8)
    parser.add_argument("--chunk-max-same-target-failures", type=int, default=2)
    parser.add_argument("--chunk-detour-scales", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--chunk-detour-clearance-weight", type=float, default=100.0)
    parser.add_argument("--chunk-detour-task-rejoin-weight", type=float, default=10.0)
    parser.add_argument("--chunk-detour-action-norm-weight", type=float, default=0.2)
    parser.add_argument("--chunk-delayed-rejoin-wait-steps", type=int, default=4)
    parser.add_argument("--chunk-delayed-rejoin-requires-nominal-prefix-safe", dest="chunk_delayed_rejoin_requires_nominal_prefix_safe", action="store_true", default=True)
    parser.add_argument("--no-chunk-delayed-rejoin-requires-nominal-prefix-safe", dest="chunk_delayed_rejoin_requires_nominal_prefix_safe", action="store_false")
    parser.add_argument("--chunk-require-safe-corridor-for-recovery-complete", dest="chunk_require_safe_corridor_for_recovery_complete", action="store_true", default=True)
    parser.add_argument("--no-chunk-require-safe-corridor-for-recovery-complete", dest="chunk_require_safe_corridor_for_recovery_complete", action="store_false")
    parser.add_argument("--chunk-require-post-recovery-act-window", dest="chunk_require_post_recovery_act_window", action="store_true", default=True)
    parser.add_argument("--no-chunk-require-post-recovery-act-window", dest="chunk_require_post_recovery_act_window", action="store_false")
    parser.add_argument("--chunk-post-recovery-min-act-steps", type=int, default=5)
    parser.add_argument("--chunk-acceptance-clearance-tol", type=float, default=0.005)
    parser.add_argument("--chunk-lambda-deform-safety", "--chunk-lambda-yield-safety", dest="chunk_lambda_yield_safety", type=float, default=800.0)
    parser.add_argument("--chunk-lambda-deform-action", "--chunk-lambda-yield-action", dest="chunk_lambda_yield_action", type=float, default=0.1)
    parser.add_argument("--chunk-lambda-deform-smooth", "--chunk-lambda-yield-smooth", dest="chunk_lambda_yield_smooth", type=float, default=0.1)
    parser.add_argument("--chunk-lambda-retreat", type=float, default=1.0)
    parser.add_argument("--chunk-lambda-recover-safety", "--chunk-lambda-return-safety", dest="chunk_lambda_return_safety", type=float, default=500.0)
    parser.add_argument("--chunk-lambda-recover-rejoin", "--chunk-lambda-return-rejoin", dest="chunk_lambda_return_rejoin", type=float, default=5.0)
    parser.add_argument("--chunk-lambda-recover-smooth", "--chunk-lambda-return-smooth", dest="chunk_lambda_return_smooth", type=float, default=0.2)
    parser.add_argument("--chunk-lambda-recover-action", "--chunk-lambda-return-action", dest="chunk_lambda_return_action", type=float, default=0.1)
    parser.add_argument("--chunk-deform-horizon", "--chunk-yield-horizon", dest="chunk_yield_horizon", type=int, default=4)
    parser.add_argument("--chunk-recover-horizon", "--chunk-return-horizon", dest="chunk_return_horizon", type=int, default=8)
    parser.add_argument("--chunk-max-recover-retries", "--chunk-max-return-retries", dest="chunk_max_return_retries", type=int, default=3)
    parser.add_argument(
        "--chunk-use-ee-final-check",
        dest="chunk_use_ee_final_check",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-chunk-use-ee-final-check",
        dest="chunk_use_ee_final_check",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-cache-nominal-ee",
        dest="chunk_cache_nominal_ee",
        action="store_true",
        default=False,
        help="Precompute nominal EE poses once per ACT chunk when needed.",
    )
    parser.add_argument(
        "--no-chunk-cache-nominal-ee",
        dest="chunk_cache_nominal_ee",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-ee-rejoin-in-inner-loop",
        dest="chunk_ee_rejoin_in_inner_loop",
        action="store_true",
        default=False,
        help="Legacy slow mode: run EE FK inside the inner optimization loop.",
    )
    parser.add_argument(
        "--chunk-debug-safety-feasibility",
        dest="chunk_debug_safety_feasibility",
        action="store_true",
        default=True,
        help="Skip EE final checks and log optimizer safety feasibility diagnostics.",
    )
    parser.add_argument(
        "--no-chunk-debug-safety-feasibility",
        dest="chunk_debug_safety_feasibility",
        action="store_false",
    )
    parser.add_argument(
        "--no-chunk-ee-rejoin-in-inner-loop",
        dest="chunk_ee_rejoin_in_inner_loop",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-recoverable-deform",
        dest="chunk_recoverable_deform_enabled",
        action="store_true",
        default=True,
        help="Add rejoin loss and recoverability rejection to optimized deformation.",
    )
    parser.add_argument(
        "--no-chunk-recoverable-deform",
        dest="chunk_recoverable_deform_enabled",
        action="store_false",
        help="Disable optimized-deform recovery terms and preserve legacy optimized behavior.",
    )
    parser.add_argument(
        "--chunk-brake-if-unrecoverable",
        dest="chunk_brake_if_unrecoverable",
        action="store_true",
        default=True,
        help="Use the existing brake/hold chunk when optimized deformation is unsafe or unrecoverable.",
    )
    parser.add_argument(
        "--no-chunk-brake-if-unrecoverable",
        dest="chunk_brake_if_unrecoverable",
        action="store_false",
        help="Use --chunk-optimized-fallback when optimized deformation is rejected.",
    )
    parser.add_argument(
        "--chunk-optimized-fallback",
        choices=["candidate", "brake"],
        default="brake",
        help="Fallback used when optimized deformation is unsafe or unrecoverable. Use candidate only for ablation.",
    )
    parser.add_argument("--chunk-opt-population", type=int, default=32)
    parser.add_argument("--chunk-opt-elite-frac", type=float, default=0.25)
    parser.add_argument("--chunk-opt-seed", type=int, default=0)
    parser.add_argument(
        "--chunk-detach-passthrough-dims",
        dest="chunk_detach_passthrough_dims",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-chunk-detach-passthrough-dims",
        dest="chunk_detach_passthrough_dims",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-use-ee-pose-rejoin",
        dest="chunk_use_ee_pose_rejoin",
        action="store_true",
        default=False,
        help="Legacy alias for enabling EE-pose inner-loop rejoin when combined with --chunk-ee-rejoin-in-inner-loop.",
    )
    parser.add_argument(
        "--no-chunk-use-ee-pose-rejoin",
        dest="chunk_use_ee_pose_rejoin",
        action="store_false",
    )
    parser.add_argument(
        "--chunk-use-object-state-rejoin",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save-actions",
        type=str,
        default=None,
        help="Save normalized action chunks sent to env.step() as an NPZ file.",
    )
    parser.add_argument(
        "--replay-actions",
        type=str,
        default=None,
        help="Replay normalized action chunks from an NPZ file instead of querying ACT.",
    )

    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override. Can be used multiple times.",
    )

    args = parser.parse_args()
    args.record_video = not args.no_record_video
    if args.save_actions is not None and args.replay_actions is not None:
        parser.error("Use either --save-actions or --replay-actions, not both.")
    if args.enable_human_arm_collisions and (
        args.disable_human_arm_collisions or args.visual_only_human_arm
    ):
        parser.error(
            "--enable-human-arm-collisions cannot be combined with "
            "--disable-human-arm-collisions or --visual-only-human-arm"
        )
    if args.policy_video_every < 1:
        parser.error("--policy-video-every must be >= 1")
    if args.human_arm_aggression <= 0:
        parser.error("--human-arm-aggression must be > 0")
    if args.human_arm_substeps < 1:
        parser.error("--human-arm-substeps must be >= 1")
    if args.human_arm_force_carrier_frequency <= 0:
        parser.error("--human-arm-force-carrier-frequency must be > 0")
    if args.human_arm_ee_obstruction and args.human_arm_drawer_obstruction:
        parser.error(
            "Use either --human-arm-ee-obstruction or "
            "--human-arm-drawer-obstruction, not both."
        )
    if (
        args.human_arm_force_carrier_amp_xy is not None
        and args.human_arm_force_carrier_xy is None
        and not args.human_arm_ee_obstruction
        and not args.human_arm_drawer_obstruction
    ):
        parser.error(
            "--human-arm-force-carrier-amp-xy requires --human-arm-force-carrier-xy "
            "unless an obstruction mode is set"
        )
    if args.human_arm_natural_motion_frequency <= 0:
        parser.error("--human-arm-natural-motion-frequency must be > 0")
    if args.human_arm_natural_lateral_scale < 0:
        parser.error("--human-arm-natural-lateral-scale must be >= 0")
    if args.human_arm_natural_return_curl_scale < 0:
        parser.error("--human-arm-natural-return-curl-scale must be >= 0")
    if args.human_arm_transient_obstruction:
        if args.human_arm_release_after_steps < 0:
            parser.error("--human-arm-release-after-steps must be >= 0")
        if args.human_arm_release_duration_steps < 1:
            parser.error("--human-arm-release-duration-steps must be >= 1")
    if args.human_arm_final_clear_after_steps >= 0:
        if args.human_arm_final_clear_duration_steps < 1:
            parser.error("--human-arm-final-clear-duration-steps must be >= 1")
    if args.stop_video_at_steps is not None and args.stop_video_at_steps <= 0:
        parser.error("--stop-video-at-steps must be > 0")
    if args.resume_clear_steps < 1:
        parser.error("--resume-clear-steps must be >= 1")
    if args.resume_clearance_threshold < 0:
        parser.error("--resume-clearance-threshold must be >= 0")
    if args.phase_reanchor_check_after_steps < 0:
        parser.error("--phase-reanchor-check-after-steps must be >= 0")
    if args.phase_reanchor_no_progress_window < 1:
        parser.error("--phase-reanchor-no-progress-window must be >= 1")
    if args.phase_reanchor_steps < 1:
        parser.error("--phase-reanchor-steps must be >= 1")
    if args.phase_reanchor_cooldown_steps < 0:
        parser.error("--phase-reanchor-cooldown-steps must be >= 0")
    if args.phase_reanchor_max_base_step <= 0:
        parser.error("--phase-reanchor-max-base-step must be > 0")
    if args.phase_reanchor_base_gain < 0:
        parser.error("--phase-reanchor-base-gain must be >= 0")
    if args.phase_reanchor_grasp_dist < 0:
        parser.error("--phase-reanchor-grasp-dist must be >= 0")
    if args.post_recovery_task_guard_steps < 1:
        parser.error("--post-recovery-task-guard-steps must be >= 1")
    if args.post_recovery_progress_tolerance < 0:
        parser.error("--post-recovery-progress-tolerance must be >= 0")
    if args.post_recovery_task_guard_min_progress < 0:
        parser.error("--post-recovery-task-guard-min-progress must be >= 0")
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


def _raw_scaled_first_action(env, action):
    rescale_wrapper = _find_wrapped_env_with_attr(env, "action_stats")
    if rescale_wrapper is None or not hasattr(rescale_wrapper, "action"):
        return None
    return np.asarray(rescale_wrapper.action(np.asarray(action, dtype=np.float32)), dtype=np.float32)


def _raw_action_to_normalized(env, raw_action):
    rescale_wrapper = _find_wrapped_env_with_attr(env, "action_stats")
    if rescale_wrapper is None:
        return None
    action_stats = getattr(rescale_wrapper, "action_stats", None)
    if action_stats is None or "min" not in action_stats or "max" not in action_stats:
        return None

    action_min = np.asarray(action_stats["min"], dtype=np.float32)
    action_max = np.asarray(action_stats["max"], dtype=np.float32)
    margin = float(getattr(rescale_wrapper, "min_max_margin", 0.0))
    action_min = action_min - np.fabs(action_min) * margin
    action_max = action_max + np.fabs(action_max) * margin

    raw = np.asarray(raw_action, dtype=np.float32)
    normalized = (raw - action_min) / (action_max - action_min + 1e-8)
    normalized = normalized * 2.0 - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def _env_chain(env):
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "env", None)


def _has_direct_attr(obj, attr_name):
    if attr_name in getattr(obj, "__dict__", {}):
        return True
    return any(attr_name in cls.__dict__ for cls in type(obj).mro())


def _find_wrapped_attr(env, attr_name):
    for candidate in _env_chain(env):
        if _has_direct_attr(candidate, attr_name):
            return getattr(candidate, attr_name)
    return None


def _find_wrapped_env_with_attr(env, attr_name):
    for candidate in _env_chain(env):
        if _has_direct_attr(candidate, attr_name):
            return candidate
    return None


def _apply_robot_spawn_offset_xy(env, offset_xy) -> Optional[dict[str, list[float]]]:
    if offset_xy is None:
        return None

    task = _find_wrapped_env_with_attr(env, "RESET_ROBOT_POS")
    if task is None:
        raise RuntimeError("Could not find raw BiGym env with RESET_ROBOT_POS.")

    if hasattr(task, "_eval_default_reset_robot_pos"):
        default_pos = np.asarray(
            task._eval_default_reset_robot_pos, dtype=np.float64
        ).copy()
    else:
        default_pos = np.asarray(task.RESET_ROBOT_POS, dtype=np.float64).copy()
        task._eval_default_reset_robot_pos = default_pos.copy()

    offset = np.asarray(offset_xy, dtype=np.float64).reshape(2)
    spawn_pos = default_pos.copy()
    spawn_pos[:2] = spawn_pos[:2] + offset
    task.RESET_ROBOT_POS = spawn_pos

    return {
        "default_pos": default_pos.astype(float).tolist(),
        "offset_xy": offset.astype(float).tolist(),
        "spawn_pos": spawn_pos.astype(float).tolist(),
    }


def _reset_action_sequence_history(env) -> int:
    reset_count = 0
    for candidate in _env_chain(env):
        reset = getattr(candidate, "_init_action_history", None)
        if callable(reset):
            reset()
            reset_count += 1
    return reset_count


def _is_safety_intervention_mode(safety_info) -> bool:
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    if mode in {
        "path_consistent_brake",
        "pause_on_unsafe",
        "stop",
        "horizon_deform",
        "sequential_oscbf",
        "single_step_oscbf",
        "phase_reanchor",
    }:
        return True
    return source in {"chunk_deform", "sequential_oscbf", "sequential_oscbf_fallback"}


def _should_hold_policy_step(safety_info, first_action, safe_first_action, arm_idx, eps) -> bool:
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    if mode not in {"path_consistent_brake", "pause_on_unsafe", "stop"}:
        return False
    if bool(_safe_info_get(safety_info, "brake_hold_current")):
        return True
    first_action = np.asarray(first_action, dtype=np.float32).reshape(-1)
    safe_first_action = np.asarray(safe_first_action, dtype=np.float32).reshape(-1)
    arm_idx = np.asarray(arm_idx, dtype=np.int64)
    valid = arm_idx < min(first_action.shape[0], safe_first_action.shape[0])
    if not np.any(valid):
        return False
    delta = np.linalg.norm(safe_first_action[arm_idx[valid]] - first_action[arm_idx[valid]])
    return bool(delta > float(eps))


def _disable_human_arm_collisions(env) -> int:
    humanarms = _find_wrapped_attr(env, "humanarms")
    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if not humanarms or base_env is None:
        return 0

    import mujoco

    model = base_env.mojo.physics.model.ptr
    disabled = 0
    for name in ["cylinder_arm/upperarm_geom", "cylinder_arm/forearm_geom"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        model.geom_margin[gid] = 0.0
        disabled += 1
    base_env.mojo.physics.forward()
    return disabled


def _enable_human_arm_collisions(env) -> int:
    humanarms = _find_wrapped_attr(env, "humanarms")
    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if not humanarms or base_env is None:
        return 0

    import mujoco

    model = base_env.mojo.physics.model.ptr
    enabled = 0
    for name in ["cylinder_arm/upperarm_geom", "cylinder_arm/forearm_geom"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        model.geom_contype[gid] = 2
        model.geom_conaffinity[gid] = 1
        model.geom_margin[gid] = max(float(model.geom_margin[gid]), 0.01)
        enabled += 1
    base_env.mojo.physics.forward()
    return enabled


def _freeze_human_arm(env) -> int:
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0

    frozen = 0
    for human in humanarms:
        state = human.get_state()
        # set_qpos_target also switches HumanArm out of scripted mode and into
        # position mode, so scripted primitives stop advancing.
        human.set_qpos_target(state["qpos"])
        if hasattr(human, "_qpos_filt"):
            human._qpos_filt = state["qpos"].copy()
        if hasattr(human, "_qvel_filt"):
            human._qvel_filt[:] = 0.0
        if hasattr(human, "_walk_enable"):
            human._walk_enable = False
        if hasattr(human, "_walk_v"):
            human._walk_v[:] = 0.0
        if hasattr(human, "_carrier_dwell"):
            human._carrier_dwell = 1e9
        frozen += 1
    return frozen


def _update_temporary_human_blocker_if_present(env) -> Optional[dict]:
    blocker = _find_wrapped_attr(env, "_temporary_human_blocker")
    if blocker is None:
        return None
    task = _find_wrapped_env_with_attr(env, "get_dt")
    if task is None:
        return None
    info = dict(blocker.update(task.get_dt()))
    if hasattr(task, "_temporary_human_blocker_info"):
        task._temporary_human_blocker_info = dict(info)
    return info


def _configure_human_arm_challenge(env, args) -> int:
    _reset_human_arm_final_clear_state(args)
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0

    aggression = float(np.clip(args.human_arm_aggression, 0.1, 3.0))
    configured = 0
    for human in humanarms:
        if hasattr(human, "_style_speed"):
            human._style_speed = float(np.clip(human._style_speed * aggression, 0.1, 3.0))
        if hasattr(human, "_style_amp"):
            human._style_amp = float(np.clip(human._style_amp * aggression, 0.1, 1.75))
        if hasattr(human, "_style_dwell") and args.human_arm_zero_dwell:
            human._style_dwell = min(float(human._style_dwell), 0.2)
        if hasattr(human, "_carrier_dwell") and args.human_arm_zero_dwell:
            human._carrier_dwell = 0.0
        if args.human_arm_walk_radius is not None and hasattr(human, "_walk_radius"):
            human._walk_radius = float(max(0.0, args.human_arm_walk_radius))
        if args.human_arm_keepout_min_clear is not None and hasattr(human, "MIN_CLEAR"):
            min_clear = float(max(0.0, args.human_arm_keepout_min_clear))
            human.MIN_CLEAR = min_clear
            if hasattr(human, "KEEP_SOFT"):
                human.KEEP_SOFT = max(float(human.KEEP_SOFT), min_clear + 0.01)
            if hasattr(human, "KEEP_HARD"):
                human.KEEP_HARD = min(float(human.KEEP_HARD), max(0.0, min_clear * 0.4))
        if args.human_arm_disable_keepout:
            _disable_human_arm_internal_keepout(human)
        _bias_human_arm_goal(human, args.human_arm_goal_xy)
        configured += 1
    _force_human_arm_carrier_xy(env, _forced_human_arm_carrier_xy(args, step=0), args=args)
    _apply_natural_human_arm_contact_motion(env, args, step=0)
    _force_human_arm_carrier_xy(env, _forced_human_arm_carrier_xy(args, step=0), args=args)
    return configured


def _disable_human_arm_internal_keepout(human) -> None:
    if hasattr(human, "_robot_geom_ids"):
        human._robot_geom_ids = np.asarray([], dtype=np.int32)
    if hasattr(human, "_robot_keepout_r"):
        human._robot_keepout_r = np.asarray([], dtype=np.float64)
    if hasattr(human, "MIN_CLEAR"):
        human.MIN_CLEAR = -1.0
    if hasattr(human, "KEEP_SOFT"):
        human.KEEP_SOFT = -1.0
    if hasattr(human, "KEEP_HARD"):
        human.KEEP_HARD = -1.0
    if hasattr(human, "_debug_keepout_clear"):
        human._debug_keepout_clear = float("inf")
    if hasattr(human, "_debug_keepout_active"):
        human._debug_keepout_active = False


def _bias_human_arm_goal(human, goal_xy) -> bool:
    if goal_xy is None or not hasattr(human, "_walk_goal_xy"):
        return False
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    radius = float(getattr(human, "_walk_radius", np.linalg.norm(goal)))
    norm = float(np.linalg.norm(goal))
    if radius > 0.0 and norm > radius:
        goal = goal / (norm + 1e-12) * radius
    human._walk_goal_xy = goal
    if hasattr(human, "_carrier_dwell"):
        human._carrier_dwell = 0.0
    return True



def _transient_human_arm_alpha(args, step: int) -> float:
    if not args.human_arm_transient_obstruction:
        return 0.0
    start = int(args.human_arm_release_after_steps)
    duration = max(1, int(args.human_arm_release_duration_steps))
    return float(np.clip((int(step) - start) / duration, 0.0, 1.0))


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _human_arm_retracted_q(human):
    q = np.array(
        [
            np.deg2rad(-8.0),
            np.deg2rad(4.0),
            np.deg2rad(8.0),
            np.deg2rad(92.0),
        ],
        dtype=np.float64,
    )
    if hasattr(human, "_clip_joint_vec"):
        q = human._clip_joint_vec(q)
    return q

def _apply_human_arm_yaw_offset(
    human, args, q: np.ndarray, *, current_state: bool = False
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    offset = np.deg2rad(float(getattr(args, "human_arm_yaw_offset_deg", 0.0)))
    if q.shape[0] > 1:
        if current_state:
            previous = float(getattr(human, "_eval_human_arm_yaw_offset_rad", 0.0))
            q[1] += offset - previous
        else:
            q[1] += offset
    setattr(human, "_eval_human_arm_yaw_offset_rad", float(offset))
    if hasattr(human, "_clip_joint_vec"):
        q = human._clip_joint_vec(q)
    return q


def _natural_human_arm_contact_q(human, args, step: int = 0, dt: float = 0.05):
    phase_step = float(step) + float(getattr(args, "human_arm_natural_motion_phase_offset_steps", 0.0))
    phase = 2.0 * np.pi * float(args.human_arm_natural_motion_frequency) * phase_step * float(dt)
    reach = 0.5 * (1.0 - np.cos(phase))
    sweep = np.sin(phase)
    settle = np.sin(0.5 * phase + 0.35)
    lateral_scale = float(max(0.0, getattr(args, "human_arm_natural_lateral_scale", 1.0)))
    curl_scale = float(max(0.0, getattr(args, "human_arm_natural_return_curl_scale", 0.0)))
    return_phase = float(np.clip(-sweep, 0.0, 1.0))

    q = np.array(
        [
            np.deg2rad(1.0) + lateral_scale * np.deg2rad(4.0) * settle,
            np.deg2rad(0.0) + lateral_scale * np.deg2rad(7.0) * sweep,
            np.deg2rad(-30.0) - np.deg2rad(30.0) * reach + np.deg2rad(2.0) * settle + np.deg2rad(8.0) * curl_scale * return_phase,
            np.deg2rad(66.0) - np.deg2rad(30.0) * reach + np.deg2rad(4.0) * np.sin(phase + 0.8) + np.deg2rad(28.0) * curl_scale * return_phase,
        ],
        dtype=np.float64,
    )
    if hasattr(human, "_clip_joint_vec"):
        q = human._clip_joint_vec(q)
    return q


def _apply_natural_human_arm_contact_motion(env, args, step: int = 0, dt: float = 0.05) -> int:
    if not args.human_arm_natural_contact_motion:
        return 0
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0
    applied = 0
    for human in humanarms:
        if not hasattr(human, "_set_kinematic_state"):
            continue
        q = _natural_human_arm_contact_q(human, args, step=step, dt=dt)
        alpha = _smoothstep(_transient_human_arm_alpha(args, step))
        if alpha > 0.0:
            q = (1.0 - alpha) * q + alpha * _human_arm_retracted_q(human)
        q = _apply_human_arm_yaw_offset(human, args, q)
        if hasattr(human, "_qpos_filt"):
            human._qpos_filt = q.copy()
        if hasattr(human, "_qvel_filt"):
            human._qvel_filt[:] = 0.0
        if hasattr(human, "_qpos_target"):
            human._qpos_target[:] = q
        if hasattr(human, "_walk_xy"):
            xy = np.asarray(human._walk_xy, dtype=np.float64).copy()
        else:
            state = human.get_state()
            xy = np.zeros(2, dtype=np.float64)
        human._set_kinematic_state(xy, q)
        applied += 1
    return applied



def _robot_ee_world_xy(oscbf, q_full: np.ndarray, qd_full: np.ndarray, offset_xy=None):
    if oscbf is None or oscbf.robot_model is None:
        return None
    q_urdf, _, _, _ = oscbf._build_urdf_surrogate_state_from_bigym(q_full, qd_full)
    ee_urdf = np.asarray(
        oscbf.robot_model.ee_position(jnp.asarray(q_urdf, dtype=jnp.float32)),
        dtype=np.float32,
    ).reshape(3)
    t_world_urdf = oscbf._get_world_T_urdf_from_bigym_state(q_full)
    ee_world = np.asarray(
        oscbf._transform_points_homogeneous(t_world_urdf, ee_urdf),
        dtype=np.float64,
    ).reshape(-1)
    xy = ee_world[:2].copy()
    if offset_xy is not None:
        xy = xy + np.asarray(offset_xy, dtype=np.float64).reshape(2)
    return xy

def _robot_gripper_geom_world_xy(env, offset_xy=None):
    try:
        task = get_bigym_task(env)
        model = task._mojo.model
        data = task._mojo.data
    except Exception:  # noqa: BLE001
        return None

    priority_patterns = (
        ("robotiq_2f85_right", "finger"),
        ("robotiq_2f85_right", "pad"),
        ("robotiq_2f85_right", "driver"),
        ("robotiq_2f85_right",),
        ("right_wrist",),
        ("wrist",),
    )
    exclude_patterns = ("visual", "camera", "left")

    for patterns in priority_patterns:
        points = []
        for geom_id in range(model.ngeom):
            name = (model.geom(geom_id).name or "").lower()
            if not name:
                continue
            if any(excluded in name for excluded in exclude_patterns):
                continue
            if all(pattern in name for pattern in patterns):
                points.append(np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3))
        if points:
            xy = np.mean(np.stack(points, axis=0), axis=0)[:2]
            if offset_xy is not None:
                xy = xy + np.asarray(offset_xy, dtype=np.float64).reshape(2)
            return xy

    return None


def _drawer_obstruction_carrier_xy(args, step: int = 0, dt: float = 0.05):
    xy = np.asarray(args.human_arm_drawer_obstruction_xy, dtype=np.float64).reshape(2)
    amp = np.asarray(args.human_arm_drawer_obstruction_amp_xy, dtype=np.float64).reshape(2)
    phase = 2.0 * np.pi * float(args.human_arm_force_carrier_frequency) * float(step) * float(dt)
    # Move locally around the drawer area without anchoring to the robot EE.
    offset = np.array(
        [
            0.55 * np.sin(phase + 0.4) + 0.20 * np.sin(1.9 * phase),
            0.45 * np.sin(phase + 1.7) + 0.15 * np.sin(1.4 * phase + 0.8),
        ],
        dtype=np.float64,
    )
    alpha = _smoothstep(_transient_human_arm_alpha(args, step))
    return xy + (1.0 - alpha) * amp * offset


def _ee_side_sweep_carrier_xy(args, anchor_xy, step: int = 0, dt: float = 0.05):
    xy = np.asarray(anchor_xy, dtype=np.float64).reshape(2)
    amp = np.asarray(args.human_arm_ee_side_sweep_amp_xy, dtype=np.float64).reshape(2)
    phase = (
        2.0 * np.pi * float(args.human_arm_ee_side_sweep_frequency) * float(step) * float(dt)
        + float(getattr(args, "human_arm_ee_side_sweep_phase", 0.0))
    )
    offset = np.array(
        [
            0.35 * np.sin(0.5 * phase + 0.3),
            np.sin(phase),
        ],
        dtype=np.float64,
    )
    alpha = _smoothstep(_transient_human_arm_alpha(args, step))
    return xy + (1.0 - alpha) * amp * offset


def _forced_human_arm_carrier_xy(args, step: int = 0, dt: float = 0.05, anchor_xy=None):
    if (
        anchor_xy is None
        and args.human_arm_force_carrier_xy is None
        and not args.human_arm_transient_obstruction
        and not args.human_arm_drawer_obstruction
    ):
        return None
    if anchor_xy is not None:
        if args.human_arm_ee_side_sweep:
            xy = _ee_side_sweep_carrier_xy(args, anchor_xy, step=step, dt=dt)
        else:
            xy = np.asarray(anchor_xy, dtype=np.float64).reshape(2)
    elif args.human_arm_drawer_obstruction:
        xy = _drawer_obstruction_carrier_xy(args, step=step, dt=dt)
    elif args.human_arm_force_carrier_xy is None:
        xy = np.asarray([-0.5, 0.2], dtype=np.float64)
    else:
        xy = np.asarray(args.human_arm_force_carrier_xy, dtype=np.float64).reshape(2)

    if args.human_arm_force_carrier_amp_xy is not None:
        amp = np.asarray(args.human_arm_force_carrier_amp_xy, dtype=np.float64).reshape(2)
        phase = 2.0 * np.pi * float(args.human_arm_force_carrier_frequency) * float(step) * float(dt)
        offset = np.array([np.sin(phase), np.sin(phase + 0.5 * np.pi)], dtype=np.float64)
        alpha = _smoothstep(_transient_human_arm_alpha(args, step))
        xy = xy + (1.0 - alpha) * amp * offset

    alpha = _smoothstep(_transient_human_arm_alpha(args, step))
    if alpha > 0.0:
        release_xy = np.asarray(args.human_arm_release_carrier_xy, dtype=np.float64).reshape(2)
        xy = (1.0 - alpha) * xy + alpha * release_xy
    return xy


def _force_human_arm_carrier_xy(env, carrier_xy, args=None) -> int:
    if carrier_xy is None:
        return 0
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0
    xy = np.asarray(carrier_xy, dtype=np.float64).reshape(2)
    forced = 0
    for human in humanarms:
        if not hasattr(human, "_set_kinematic_state"):
            continue
        if hasattr(human, "_qpos_filt"):
            joint_q = np.asarray(human._qpos_filt, dtype=np.float64).copy()
        else:
            joint_q = np.asarray(human.get_state()["qpos"], dtype=np.float64).copy()
        joint_q = _apply_human_arm_yaw_offset(human, args, joint_q, current_state=True) if args is not None else joint_q
        human._set_kinematic_state(xy, joint_q)
        try:
            task = get_bigym_task(env)
            task._mojo.physics.forward()
        except Exception:  # noqa: BLE001
            pass
        if hasattr(human, "_walk_xy"):
            human._walk_xy = xy.copy()
        if hasattr(human, "_walk_goal_xy"):
            human._walk_goal_xy = xy.copy()
        if hasattr(human, "_walk_v"):
            human._walk_v[:] = 0.0
        forced += 1
    return forced



def _reset_human_arm_final_clear_state(args) -> None:
    for name in (
        "_human_arm_final_clear_y_last",
        "_human_arm_final_clear_y_prev_delta",
        "_human_arm_final_clear_y_peak_step",
    ):
        if hasattr(args, name):
            delattr(args, name)


def _human_arm_final_clear_start_step(args) -> int:
    configured_start = int(getattr(args, "human_arm_final_clear_after_steps", -1))
    if configured_start < 0:
        return -1
    trigger = str(getattr(args, "human_arm_final_clear_trigger", "step"))
    peak_step = getattr(args, "_human_arm_final_clear_y_peak_step", None)
    if trigger == "carrier-y-peak" and peak_step is not None:
        return int(peak_step)
    return configured_start


def _human_arm_final_clear_alpha(args, step: int) -> float:
    start = _human_arm_final_clear_start_step(args)
    if start < 0:
        return 0.0
    duration = max(1, int(getattr(args, "human_arm_final_clear_duration_steps", 20)))
    return _smoothstep(float(np.clip((int(step) - start) / duration, 0.0, 1.0)))


def _update_human_arm_final_clear_y_peak_trigger(args, step: int, carrier_xy) -> None:
    if int(getattr(args, "human_arm_final_clear_after_steps", -1)) < 0:
        return
    if str(getattr(args, "human_arm_final_clear_trigger", "step")) != "carrier-y-peak":
        return
    if getattr(args, "_human_arm_final_clear_y_peak_step", None) is not None:
        return
    if carrier_xy is None:
        return
    y = float(np.asarray(carrier_xy, dtype=np.float64).reshape(2)[1])
    last_y = getattr(args, "_human_arm_final_clear_y_last", None)
    prev_delta = getattr(args, "_human_arm_final_clear_y_prev_delta", None)
    if last_y is not None:
        delta = y - float(last_y)
        eps = 1e-5
        if prev_delta is not None and float(prev_delta) > eps and delta <= eps:
            setattr(args, "_human_arm_final_clear_y_peak_step", int(step))
        setattr(args, "_human_arm_final_clear_y_prev_delta", float(delta))
    setattr(args, "_human_arm_final_clear_y_last", y)


def _limited_step_toward(current, target, max_step: float) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_step or norm <= 1e-12:
        return target.copy()
    return current + delta * (float(max_step) / norm)


def _apply_final_human_arm_clearance(env, args, step: int, dt: float = 0.05) -> int:
    alpha = _human_arm_final_clear_alpha(args, step)
    if alpha <= 0.0:
        return 0
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0

    start_step = _human_arm_final_clear_start_step(args)
    dt = max(float(dt), 1e-9)
    max_carrier_step = float(args.human_arm_final_clear_max_carrier_speed) * dt
    max_joint_step = float(args.human_arm_final_clear_max_joint_speed) * dt
    target_xy = np.asarray(
        getattr(args, "human_arm_final_clear_carrier_xy", [-0.85, 0.55]),
        dtype=np.float64,
    ).reshape(2)
    applied = 0
    for human in humanarms:
        if not hasattr(human, "_set_kinematic_state"):
            continue
        if hasattr(human, "_walk_xy"):
            current_xy = np.asarray(human._walk_xy, dtype=np.float64).reshape(2)
        elif hasattr(human, "_carrier_qpos_adr") and hasattr(human, "_physics"):
            current_xy = np.asarray(
                human._physics.data.qpos[human._carrier_qpos_adr],
                dtype=np.float64,
            ).reshape(2)
        else:
            current_xy = target_xy.copy()

        if hasattr(human, "_qpos_filt"):
            current_q = np.asarray(human._qpos_filt, dtype=np.float64).copy()
        else:
            current_q = np.asarray(human.get_state()["qpos"], dtype=np.float64).copy()
        if getattr(human, "_eval_final_clear_start_step", None) != start_step:
            human._eval_final_clear_start_step = start_step
            human._eval_final_clear_start_xy = current_xy.copy()
            human._eval_final_clear_start_q = current_q.copy()

        start_xy = np.asarray(human._eval_final_clear_start_xy, dtype=np.float64).reshape(2)
        start_q = np.asarray(human._eval_final_clear_start_q, dtype=np.float64).copy()
        target_q = _apply_human_arm_yaw_offset(
            human,
            args,
            _human_arm_retracted_q(human),
        )
        desired_xy = (1.0 - alpha) * start_xy + alpha * target_xy
        desired_q = (1.0 - alpha) * start_q + alpha * target_q
        xy = _limited_step_toward(current_xy, desired_xy, max_carrier_step)
        q = _limited_step_toward(current_q, desired_q, max_joint_step)
        if hasattr(human, "_clip_joint_vec"):
            q = human._clip_joint_vec(q)

        if hasattr(human, "_qpos_filt"):
            human._qpos_filt = q.copy()
        if hasattr(human, "_qvel_filt"):
            human._qvel_filt[:] = (q - current_q) / dt
        if hasattr(human, "_qpos_target"):
            human._qpos_target[:] = q
        if hasattr(human, "_walk_xy"):
            human._walk_xy = xy.copy()
        if hasattr(human, "_walk_goal_xy"):
            human._walk_goal_xy = target_xy.copy()
        if hasattr(human, "_walk_v"):
            human._walk_v[:] = 0.0
        if hasattr(human, "_carrier_dwell"):
            human._carrier_dwell = 0.0
        human._set_kinematic_state(xy, q)
        applied += 1

    try:
        task = get_bigym_task(env)
        task._mojo.physics.forward()
    except Exception:  # noqa: BLE001
        pass
    return applied


def _human_arm_contact_geom_center_xy(env):
    try:
        import mujoco

        task = get_bigym_task(env)
        model = task._mojo.model
        data = task._mojo.data
        centers = []
        humanarms = getattr(task, "humanarms", [])
        for human in humanarms:
            for geom_name in ("forearm_geom", "upperarm_geom"):
                gid = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    human._pref(geom_name),
                )
                if gid >= 0:
                    centers.append(np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3))
        if not centers:
            return None
        return np.mean(np.stack(centers, axis=0), axis=0)[:2]
    except Exception:  # noqa: BLE001
        return None


def _align_human_arm_contact_geoms_to_xy(env, target_xy) -> bool:
    if target_xy is None:
        return False
    center_xy = _human_arm_contact_geom_center_xy(env)
    if center_xy is None:
        return False
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return False
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    shifted = False
    for human in humanarms:
        if not hasattr(human, "_set_kinematic_state"):
            continue
        if hasattr(human, "_walk_xy"):
            current_xy = np.asarray(human._walk_xy, dtype=np.float64).reshape(2)
        else:
            state = human.get_state()
            current_xy = np.asarray(state.get("walk_xy", [0.0, 0.0]), dtype=np.float64).reshape(2)
        shifted_xy = current_xy + (target_xy - center_xy)
        _force_human_arm_carrier_xy(env, shifted_xy)
        shifted = True
    return shifted


def _bias_human_arm_goals(env, goal_xy) -> int:
    if goal_xy is None:
        return 0
    humanarms = _find_wrapped_attr(env, "humanarms")
    if not humanarms:
        return 0
    return sum(1 for human in humanarms if _bias_human_arm_goal(human, goal_xy))


def _make_policy_env_cfg(cfg, policy_env: str):
    if not policy_env.startswith("bigym/"):
        raise ValueError(
            "--policy-env currently expects a BiGym env name like "
            f"'bigym/drawer_top_open', got {policy_env!r}."
        )
    policy_cfg = copy.deepcopy(cfg)
    policy_cfg.env.task_name = policy_env.split("/", 1)[1]
    return policy_cfg


def _joint_qpos_qvel_dims(model, joint_id):
    import mujoco

    joint_type = model.jnt_type[joint_id]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def _sync_named_mujoco_state(source_env, target_env) -> dict[str, int]:
    source_task = _find_wrapped_env_with_attr(source_env, "mojo")
    target_task = _find_wrapped_env_with_attr(target_env, "mojo")
    if source_task is None or target_task is None:
        raise RuntimeError("Could not find source/target BiGym envs for state mirroring.")

    import mujoco

    src_model = source_task.mojo.physics.model.ptr
    src_data = source_task.mojo.physics.data
    dst_model = target_task.mojo.physics.model.ptr
    dst_data = target_task.mojo.physics.data

    dst_data.time = src_data.time

    copied_joints = 0
    for src_jid in range(src_model.njnt):
        name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_JOINT, src_jid)
        if not name:
            continue
        dst_jid = mujoco.mj_name2id(dst_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if dst_jid < 0:
            continue

        src_nq, src_nv = _joint_qpos_qvel_dims(src_model, src_jid)
        dst_nq, dst_nv = _joint_qpos_qvel_dims(dst_model, dst_jid)
        if src_nq != dst_nq or src_nv != dst_nv:
            continue

        src_qadr = src_model.jnt_qposadr[src_jid]
        dst_qadr = dst_model.jnt_qposadr[dst_jid]
        src_dadr = src_model.jnt_dofadr[src_jid]
        dst_dadr = dst_model.jnt_dofadr[dst_jid]
        dst_data.qpos[dst_qadr : dst_qadr + dst_nq] = src_data.qpos[src_qadr : src_qadr + src_nq]
        dst_data.qvel[dst_dadr : dst_dadr + dst_nv] = src_data.qvel[src_dadr : src_dadr + src_nv]
        copied_joints += 1

    copied_actuators = 0
    for src_aid in range(src_model.nu):
        name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_ACTUATOR, src_aid)
        if not name:
            continue
        dst_aid = mujoco.mj_name2id(dst_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if dst_aid < 0:
            continue
        dst_data.ctrl[dst_aid] = src_data.ctrl[src_aid]
        copied_actuators += 1

    target_task.mojo.physics.forward()
    return {"joints": copied_joints, "actuators": copied_actuators}


def _sync_animated_legs(env, is_moving: bool = True) -> bool:
    task = _find_wrapped_env_with_attr(env, "_robot")
    if task is None:
        return False
    floating_base = getattr(task._robot, "floating_base", None)
    if floating_base is None:
        return False
    animated_legs = getattr(floating_base, "_animated_legs", None)
    if animated_legs is None:
        return False
    animated_legs.step(floating_base._pelvis_z, is_moving=is_moving)
    task.mojo.physics.forward()
    return True


def _update_scripted_human_arm_pose(env, args, step: int, anchor_xy=None) -> int:
    if env is None or args.freeze_human_arm:
        return 0
    runtime = env.unwrapped if hasattr(env, "unwrapped") else env
    human_motion_dt = runtime.dt if hasattr(runtime, "dt") else 0.05
    forced_xy = _forced_human_arm_carrier_xy(
        args,
        step=step,
        dt=human_motion_dt,
        anchor_xy=anchor_xy,
    )
    _update_human_arm_final_clear_y_peak_trigger(args, step, forced_xy)
    final_clear_active = _human_arm_final_clear_alpha(args, step) > 0.0
    advanced = 0
    if not final_clear_active:
        advanced = _advance_human_arm_only(
            env,
            substeps=args.human_arm_substeps,
            goal_xy=args.human_arm_goal_xy,
        )
        _force_human_arm_carrier_xy(env, forced_xy, args=args)
        _apply_natural_human_arm_contact_motion(
            env,
            args,
            step=step,
            dt=human_motion_dt,
        )
        _force_human_arm_carrier_xy(env, forced_xy, args=args)
        if anchor_xy is not None and forced_xy is not None:
            _align_human_arm_contact_geoms_to_xy(env, forced_xy)
    _apply_final_human_arm_clearance(env, args, step, dt=human_motion_dt)
    _sync_animated_legs(env, is_moving=True)
    return advanced


def _advance_human_arm_only(env, substeps: int = 1, goal_xy=None) -> int:
    task = _find_wrapped_env_with_attr(env, "humanarms")
    if task is None:
        return 0
    dt = task.get_dt() if hasattr(task, "get_dt") else 0.05
    substeps = max(1, int(substeps))
    advanced = 0
    for _ in range(substeps):
        for human in task.humanarms:
            _bias_human_arm_goal(human, goal_xy)
            human._on_step(dt)
            advanced += 1
    task.mojo.physics.forward()
    return advanced


def _human_arm_geom_ids(env) -> list[int]:
    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if base_env is None:
        return []

    import mujoco

    model = base_env.mojo.physics.model.ptr
    geom_ids = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("cylinder_arm/"):
            geom_ids.append(gid)
    return geom_ids


def _render_visual_obs_with_hidden_human_arm(env) -> dict[str, np.ndarray]:
    base_env = _find_wrapped_env_with_attr(env, "_get_visual_obs")
    if base_env is None or not hasattr(base_env, "mojo"):
        raise RuntimeError("Could not find underlying BiGym env for clean policy rendering.")

    geom_ids = _human_arm_geom_ids(env)
    if not geom_ids:
        return base_env._get_visual_obs()

    model = base_env.mojo.physics.model.ptr
    old_rgba = model.geom_rgba[geom_ids].copy()
    try:
        model.geom_rgba[geom_ids, 3] = 0.0
        return base_env._get_visual_obs()
    finally:
        model.geom_rgba[geom_ids] = old_rgba


def _policy_obs_with_hidden_human_arm(env, obs, prev_policy_obs=None):
    policy_obs = copy.deepcopy(obs)
    visual_obs = _render_visual_obs_with_hidden_human_arm(env)

    for key, clean_frame in visual_obs.items():
        if not key.startswith("rgb_") or key not in policy_obs:
            continue

        current = np.asarray(policy_obs[key])
        clean_frame = np.asarray(clean_frame, dtype=current.dtype)

        if current.ndim == clean_frame.ndim + 1:
            if prev_policy_obs is None or key not in prev_policy_obs:
                policy_obs[key] = np.repeat(clean_frame[None], current.shape[0], axis=0)
            else:
                previous = np.asarray(prev_policy_obs[key], dtype=current.dtype)
                policy_obs[key] = np.concatenate([previous[1:], clean_frame[None]], axis=0)
        elif current.shape == clean_frame.shape:
            policy_obs[key] = clean_frame
        else:
            raise ValueError(
                f"Cannot replace policy RGB observation {key}: "
                f"wrapped shape={current.shape}, clean frame shape={clean_frame.shape}."
            )

    return policy_obs


def _adapt_policy_obs_to_space(policy_obs, observation_space):
    if observation_space is None or not isinstance(policy_obs, dict):
        return policy_obs

    adapted = dict(policy_obs)
    for key, space in observation_space.items():
        if key not in adapted or not hasattr(space, "shape"):
            continue
        expected_shape = tuple(int(x) for x in space.shape)
        value = np.asarray(adapted[key])
        if value.shape == expected_shape:
            continue

        if (
            key == "low_dim_state"
            and value.ndim == len(expected_shape)
            and value.shape[:-1] == expected_shape[:-1]
            and value.shape[-1] >= expected_shape[-1]
        ):
            adapted[key] = value[..., : expected_shape[-1]].astype(value.dtype, copy=False)
            continue

        if key == "low_dim_state" and value.size >= int(np.prod(expected_shape)):
            flat = value.reshape(-1)[: int(np.prod(expected_shape))]
            adapted[key] = flat.reshape(expected_shape).astype(value.dtype, copy=False)
            continue

        raise ValueError(
            f"Policy observation {key!r} has shape {value.shape}, "
            f"but the loaded policy expects {expected_shape}."
        )
    return adapted


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
        human_margin=args.oscbf_human_margin,
        alpha_gain=args.oscbf_alpha_gain,
        pelvis_velocity_limits=args.oscbf_pelvis_velocity_limits,
        pelvis_cbf_weight=args.oscbf_pelvis_weight,
        arm_cbf_weight=args.oscbf_arm_weight,
    )



class HorizonOSCBFOperator:
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
        self._batched_h_fn = jax.jit(
            jax.vmap(
                lambda q, capsule_a, capsule_b, capsule_radii: self.oscbf.oscbf_config.h_1(
                    q,
                    capsule_a=capsule_a,
                    capsule_b=capsule_b,
                    capsule_radii=capsule_radii,
                ),
                in_axes=(0, 0, 0, None),
            )
        )

    def set_context(self, env, obs, q_full: np.ndarray, qd_full: np.ndarray):
        self.env = env
        self.obs = obs
        self.q_full = np.asarray(q_full, dtype=np.float32).reshape(-1)
        self.qd_full = np.asarray(qd_full, dtype=np.float32).reshape(-1)
        self._update_human_capsule_velocity()

    def reset_human_motion_prediction(self):
        self._prev_capsule_a_world = None
        self._prev_capsule_b_world = None
        self._prev_capsule_radii = None
        self._capsule_a_velocity_world = None
        self._capsule_b_velocity_world = None
        self._human_motion_prediction_available = False
        self._human_motion_prediction_speed = 0.0

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

    def __call__(self, action, obs=None, **kwargs):
        return self.oscbf(
            action=action,
            env=kwargs.pop("env", self.env),
            observations=kwargs.pop("observations", obs if obs is not None else self.obs),
            q_full=kwargs.pop("q_full", self.q_full),
            qd_full=kwargs.pop("qd_full", self.qd_full),
            **kwargs,
        )

    def evaluate_safety(self, obs, q_seq):
        if self.oscbf.oscbf_config is None or self.env is None:
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
            qd_seq = np.zeros_like(q_seq, dtype=np.float32)
            q_urdf_seq = []
            capsule_a_urdf_seq = []
            capsule_b_urdf_seq = []
            for k, (q_bigym, qd_bigym) in enumerate(zip(q_seq, qd_seq)):
                q_urdf, _, _, _ = self.oscbf._build_urdf_surrogate_state_from_bigym(q_bigym, qd_bigym)
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
                q_urdf_seq.append(q_urdf)
                capsule_a_urdf_seq.append(capsule_a_urdf)
                capsule_b_urdf_seq.append(capsule_b_urdf)

            h_values = np.asarray(
                self._batched_h_fn(
                    jnp.asarray(q_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_a_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_b_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_radii_eval, dtype=jnp.float32),
                ),
                dtype=np.float32,
            )
            min_clearances = np.min(h_values, axis=1).astype(np.float32)
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
        except Exception as exc:
            logger.warning("Chunk horizon OSCBF monitor failed: %s", exc)
            return self._unavailable(q_seq)

    def evaluate_safety_batch(self, obs, q_seq_batch):
        q_seq_batch = np.asarray(q_seq_batch, dtype=np.float32)
        if q_seq_batch.ndim == 2:
            q_seq_batch = q_seq_batch[None, :, :]
        if q_seq_batch.ndim != 3:
            raise ValueError(
                "Expected q_seq_batch with shape (B, H, Q), "
                f"got {q_seq_batch.shape}"
            )
        if self.oscbf.oscbf_config is None or self.env is None:
            return self._unavailable_batch(q_seq_batch)
        batch, horizon = q_seq_batch.shape[:2]
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
                horizon,
            )

            q_urdf_seq = []
            capsule_a_urdf_seq = []
            capsule_b_urdf_seq = []
            qd_zero = np.zeros(q_seq_batch.shape[-1], dtype=np.float32)
            for candidate_q_seq in q_seq_batch:
                for k, q_bigym in enumerate(candidate_q_seq):
                    q_urdf, _, _, _ = self.oscbf._build_urdf_surrogate_state_from_bigym(
                        q_bigym,
                        qd_zero,
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
                    q_urdf_seq.append(q_urdf)
                    capsule_a_urdf_seq.append(capsule_a_urdf)
                    capsule_b_urdf_seq.append(capsule_b_urdf)

            h_values = np.asarray(
                self._batched_h_fn(
                    jnp.asarray(q_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_a_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_b_urdf_seq, dtype=jnp.float32),
                    jnp.asarray(capsule_radii_eval, dtype=jnp.float32),
                ),
                dtype=np.float32,
            ).reshape(batch, horizon, -1)
            min_clearances = np.min(h_values, axis=2).astype(np.float32)
            unsafe = min_clearances < self.min_clearance
            unsafe_any = np.any(unsafe, axis=1)
            first_violation = np.full(batch, -1, dtype=np.int32)
            if np.any(unsafe_any):
                first_violation[unsafe_any] = np.argmax(unsafe[unsafe_any], axis=1)
            info = {
                "horizon_safe": ~unsafe_any,
                "min_clearance": np.min(min_clearances, axis=1).astype(np.float32),
                "min_clearances": min_clearances,
                "first_violation": first_violation,
                "unsafe_count": np.count_nonzero(unsafe, axis=1).astype(np.int32),
                "safety_eval_available": True,
            }
            info.update(prediction_info)
            return info
        except Exception as exc:
            logger.warning("Batched chunk horizon OSCBF monitor failed: %s", exc)
            return self._unavailable_batch(q_seq_batch)

    def ee_pose(self, q):
        ee_seq = self.ee_pose_sequence(np.asarray(q, dtype=np.float32).reshape(1, -1))
        if ee_seq is None or ee_seq.shape[0] == 0:
            return None
        return ee_seq[0]

    def ee_pose_sequence(self, q_seq):
        if self.oscbf.robot_model is None:
            return None
        q_seq = np.asarray(q_seq, dtype=np.float32)
        qd_seq = np.zeros_like(q_seq, dtype=np.float32)
        ee_seq = []
        for q_bigym, qd_bigym in zip(q_seq, qd_seq):
            q_urdf, _, _, _ = self.oscbf._build_urdf_surrogate_state_from_bigym(
                q_bigym, qd_bigym
            )
            ee_urdf = np.asarray(
                self.oscbf.robot_model.ee_position(jnp.asarray(q_urdf, dtype=jnp.float32)),
                dtype=np.float32,
            ).reshape(-1, 3)
            t_world_urdf = self.oscbf._get_world_T_urdf_from_bigym_state(q_bigym)
            ee_world = self.oscbf._transform_points_homogeneous(t_world_urdf, ee_urdf)
            ee_seq.append(np.asarray(ee_world, dtype=np.float32).reshape(-1))
        return np.stack(ee_seq, axis=0).astype(np.float32)

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

    def _unavailable_batch(self, q_seq_batch):
        q_seq_batch = np.asarray(q_seq_batch)
        if q_seq_batch.ndim == 2:
            q_seq_batch = q_seq_batch[None, :, :]
        batch = int(q_seq_batch.shape[0])
        horizon = int(q_seq_batch.shape[1])
        return {
            "horizon_safe": np.ones(batch, dtype=np.bool_),
            "min_clearance": np.full(batch, np.inf, dtype=np.float32),
            "min_clearances": np.full((batch, horizon), np.inf, dtype=np.float32),
            "first_violation": np.full(batch, -1, dtype=np.int32),
            "unsafe_count": np.zeros(batch, dtype=np.int32),
            "safety_eval_available": False,
        }


def make_safechunk_filter(
    args,
    operator: HorizonOSCBFOperator,
    oscbf: Optional[OSCBFFilter] = None,
) -> SafeChunkDeformFilter:
    controlled_action_indices = None
    controlled_state_indices = None
    if oscbf is not None:
        controlled_action_indices = getattr(oscbf, "bigym_action_safety_indices", None)
        controlled_state_indices = getattr(oscbf, "bigym_state_safety_indices", None)

    return SafeChunkDeformFilter(
        oscbf_operator=operator,
        horizon=args.horizon,
        dt=0.05,
        action_dim=16,
        expected_motion_dim=14,
        control_type="absolute",
        controlled_action_indices=controlled_action_indices,
        controlled_state_indices=controlled_state_indices,
        min_clearance=args.chunk_min_clearance,
        brake_progress_threshold=args.brake_progress_threshold,
        deadlock_window=args.deadlock_window,
        deformation_enabled=args.chunk_deformation_enabled,
        mode=args.chunk_deform_mode,
        chunk_deformation_scales=args.chunk_deformation_scales,
        chunk_deformation_smoothing=args.chunk_deformation_smoothing,
        sequential_oscbf_fallback=args.sequential_oscbf_fallback,
        deform_after_deadlock_window=not args.deform_immediately_on_deadlock,
        unsafe_deformation_fallback=args.unsafe_deformation_fallback,
        opt_iters=args.chunk_opt_iters,
        opt_lr=args.chunk_opt_lr,
        lambda_safety=args.chunk_lambda_safety,
        lambda_action=args.chunk_lambda_action,
        lambda_path=args.chunk_lambda_path,
        lambda_smooth=args.chunk_lambda_smooth,
        optimized_fallback=args.chunk_optimized_fallback,
        detach_passthrough_dims=args.chunk_detach_passthrough_dims,
        recoverable_deform={
            "enabled": args.chunk_recoverable_deform_enabled,
            "lambda_rejoin": args.chunk_lambda_rejoin,
            "rejoin_threshold": args.chunk_rejoin_threshold,
            "q_rejoin_threshold": args.chunk_q_rejoin_threshold,
            "qd_rejoin_threshold": args.chunk_qd_rejoin_threshold,
            "ee_rejoin_threshold": args.chunk_ee_rejoin_threshold,
            "explicit_recovery": args.chunk_explicit_return,
            "acceptance_clearance_tol": args.chunk_acceptance_clearance_tol,
            "lambda_deform_safety": args.chunk_lambda_yield_safety,
            "lambda_deform_action": args.chunk_lambda_yield_action,
            "lambda_deform_smooth": args.chunk_lambda_yield_smooth,
            "lambda_retreat": args.chunk_lambda_retreat,
            "lambda_recover_safety": args.chunk_lambda_return_safety,
            "lambda_recover_rejoin": args.chunk_lambda_return_rejoin,
            "lambda_recover_smooth": args.chunk_lambda_return_smooth,
            "lambda_recover_action": args.chunk_lambda_return_action,
            "deform_horizon": args.chunk_yield_horizon,
            "recover_horizon": args.chunk_return_horizon,
            "max_recover_retries": args.chunk_max_return_retries,
            "use_ee_final_check": args.chunk_use_ee_final_check,
            "min_rejoin_offset": args.chunk_min_rejoin_offset,
            "inner_rejoin_metric": args.chunk_inner_rejoin_metric,
            "final_rejoin_metric": args.chunk_final_rejoin_metric,
            "cache_nominal_ee": args.chunk_cache_nominal_ee,
            "ee_rejoin_in_inner_loop": args.chunk_ee_rejoin_in_inner_loop,
            "use_ee_pose_rejoin": args.chunk_use_ee_pose_rejoin,
            "use_object_state_rejoin": args.chunk_use_object_state_rejoin,
            "brake_if_unrecoverable": args.chunk_brake_if_unrecoverable,
        },
        optimized_deform={
            "debug_safety_feasibility": args.chunk_debug_safety_feasibility,
        },
        explicit_recovery={
            "commit_accepted_chunks": args.chunk_commit_accepted_chunks,
            "committed_chunk_safety_check": args.chunk_committed_chunk_safety_check,
            "committed_safety_tol": args.chunk_committed_safety_tol,
            "committed_abort_only_if_contact_risk": args.chunk_committed_abort_only_if_contact_risk,
            "committed_min_clearance_for_abort": args.chunk_committed_min_clearance_for_abort,
            "repair_committed_action": args.chunk_repair_committed_action,
            "monotonic_committed_repair": args.chunk_monotonic_committed_repair,
            "committed_execution_margin": args.chunk_committed_execution_margin,
            "committed_state_error_threshold": args.chunk_committed_state_error_threshold,
            "committed_state_error_action": args.chunk_committed_state_error_action,
        },
        temporary_blocker={
            "enabled": args.chunk_temporary_blocker_enabled,
            "prefer_brake_before_deform": args.chunk_temporary_prefer_brake_before_deform,
            "min_unsafe_steps_before_deform": args.chunk_temporary_min_unsafe_steps_before_deform,
            "max_brake_steps_before_deform": args.chunk_temporary_max_brake_steps_before_deform,
            "reset_on_nominal_safe": args.chunk_temporary_reset_on_nominal_safe,
            "require_progress_deadlock_before_deform": args.chunk_temporary_require_progress_deadlock_before_deform,
            "progress_window": args.chunk_temporary_progress_window,
            "min_progress_delta": args.chunk_temporary_min_progress_delta,
            "recover_after_wait": args.chunk_temporary_recover_after_wait,
            "recover_after_wait_min_brake_steps": args.chunk_temporary_recover_after_wait_min_brake_steps,
        },
        safechunk_replan={
            "enabled": args.chunk_safechunk_replan_enabled,
            "replan_deform_from_current_state": True,
            "replan_recovery_from_current_state": True,
            "suppress_stale_return": True,
            "max_recovery_failure_before_replan": args.chunk_max_recovery_failure_before_replan,
            "allow_recovery_to_nominal_only_if_feasible": True,
            "recovery_target_mode": args.chunk_recovery_target_mode,
            "clear_failed_recovery_on_nominal_safe": True,
        },
        safechunk_acceptance={
            "enabled": args.chunk_acceptance_enabled,
            "hard_min_clearance": args.chunk_acceptance_hard_min_clearance,
            "desired_min_clearance": args.chunk_acceptance_desired_min_clearance,
            "allow_safe_prefix_execution": args.chunk_allow_safe_prefix_execution,
            "min_safe_prefix_len": args.chunk_acceptance_min_safe_prefix_len,
            "prefix_min_clearance": args.chunk_acceptance_prefix_min_clearance,
            "rolling_replan_on_prefix": args.chunk_rolling_replan_on_prefix,
            "full_horizon_required_for_recover": args.chunk_full_horizon_required_for_recover,
            "full_horizon_required_for_deform": args.chunk_full_horizon_required_for_deform,
            "emergency_brake_if_immediate_below_hard_margin": args.chunk_emergency_brake_if_immediate_below_hard_margin,
            "allow_candidate_fallback": args.chunk_allow_candidate_fallback,
            "candidate_fallback_only_if_no_optimized_result": args.chunk_candidate_fallback_only_if_no_optimized_result,
        },
        safechunk_recover={
            "enabled": args.chunk_safechunk_recover_enabled,
            "rejoin_nominal_weight": args.chunk_recover_rejoin_nominal_weight,
            "task_progress_weight": args.chunk_recover_task_progress_weight,
            "safety_weight": args.chunk_recover_safety_weight,
            "action_deviation_weight": args.chunk_recover_action_deviation_weight,
            "smoothness_weight": args.chunk_recover_smoothness_weight,
            "require_nominal_prefix_safe_for_rejoin": args.chunk_recover_require_nominal_prefix_safe,
            "nominal_rejoin_prefix_min_clearance": args.chunk_recover_nominal_rejoin_prefix_min_clearance,
            "use_latest_nominal_for_rejoin": args.chunk_recover_use_latest_nominal,
            "suppress_stale_nominal_rejoin": args.chunk_recover_suppress_stale_nominal,
            "rejoin_weight_schedule": args.chunk_recover_rejoin_weight_schedule,
            "rejoin_ramp_steps": args.chunk_recover_rejoin_ramp_steps,
        },
        safechunk_active_safety={
            "enabled": args.chunk_active_safety_enabled,
            "check_hold_horizon_safety": args.chunk_active_check_hold_horizon_safety,
            "predict_human_motion_for_hold": args.chunk_active_predict_human_motion_for_hold,
            "hard_min_clearance": args.chunk_active_hard_min_clearance,
            "hold_prefix_min_clearance": args.chunk_active_hold_prefix_min_clearance,
            "hold_horizon_steps": args.chunk_active_hold_horizon_steps,
            "emergency_deform_when_hold_unsafe": args.chunk_active_emergency_deform_when_hold_unsafe,
            "optimize_when_hold_unsafe": args.chunk_active_optimize_when_hold_unsafe,
            "emergency_deform_candidate_scales": args.chunk_active_emergency_deform_candidate_scales,
            "prefer_last_safe_action": args.chunk_active_prefer_last_safe_action,
            "prefer_last_safe_q_retract": args.chunk_active_prefer_last_safe_q_retract,
            "emergency_deform_replan_next_step": args.chunk_active_emergency_deform_replan_next_step,
        },
        safechunk_recovery_corridor={
            "enabled": args.chunk_recovery_corridor_enabled,
            "require_recover_path_safe": args.chunk_require_recover_path_safe,
            "recover_path_min_clearance": args.chunk_recover_path_min_clearance,
            "recover_immediate_hard_clearance": args.chunk_recover_immediate_hard_clearance,
            "recover_prefix_min_clearance": args.chunk_recover_prefix_min_clearance,
            "enable_direct_rejoin": args.chunk_enable_direct_rejoin,
            "enable_detour_rejoin": args.chunk_enable_detour_rejoin,
            "enable_delayed_rejoin": args.chunk_enable_delayed_rejoin,
            "suppress_repeated_unsafe_recovery": args.chunk_suppress_repeated_unsafe_recovery,
            "unsafe_recovery_cooldown_steps": args.chunk_unsafe_recovery_cooldown_steps,
            "max_same_target_failures": args.chunk_max_same_target_failures,
            "detour_scales": args.chunk_detour_scales,
            "detour_clearance_weight": args.chunk_detour_clearance_weight,
            "detour_task_rejoin_weight": args.chunk_detour_task_rejoin_weight,
            "detour_action_norm_weight": args.chunk_detour_action_norm_weight,
            "delayed_rejoin_wait_steps": args.chunk_delayed_rejoin_wait_steps,
            "delayed_rejoin_requires_nominal_prefix_safe": args.chunk_delayed_rejoin_requires_nominal_prefix_safe,
            "require_safe_corridor_for_recovery_complete": args.chunk_require_safe_corridor_for_recovery_complete,
            "require_post_recovery_act_window": args.chunk_require_post_recovery_act_window,
            "post_recovery_min_act_steps": args.chunk_post_recovery_min_act_steps,
        },
        diagnostics={
            "enabled": args.diagnostics_enabled,
            "large_arm_delta_threshold": args.diagnostics_large_arm_delta_threshold,
            "large_base_delta_threshold": args.diagnostics_large_base_delta_threshold,
            "low_act_ratio_threshold": args.diagnostics_low_act_ratio_threshold,
            "high_fallback_ratio_threshold": args.diagnostics_high_fallback_ratio_threshold,
        },
        opt_population=args.chunk_opt_population,
        opt_elite_frac=args.chunk_opt_elite_frac,
        opt_seed=args.chunk_opt_seed,
        max_action_delta=args.max_action_delta,
        debug=args.debug,
    )



def _pause_arm_at_current_q(action, q_full, action_indices, state_indices):
    safe = np.asarray(action, dtype=np.float32).copy()
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    action_idx = np.asarray(action_indices, dtype=np.int64)
    state_idx = np.asarray(state_indices, dtype=np.int64)
    valid = state_idx < q.shape[0]
    action_idx = action_idx[valid]
    state_idx = state_idx[valid]
    if safe.ndim == 1:
        safe[action_idx] = q[state_idx]
    elif safe.ndim == 2:
        safe[:, action_idx] = q[state_idx][None, :]
    else:
        raise ValueError(f"Unsupported action shape for pause fallback: {safe.shape}")
    return safe


def _scale_controlled_motion_from_current_q(
    action, q_full, action_indices, state_indices, scale: float
):
    scale = float(np.clip(scale, 0.0, 1.0))
    if scale <= 0.0:
        return _pause_arm_at_current_q(action, q_full, action_indices, state_indices)
    safe = np.asarray(action, dtype=np.float32).copy()
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    action_idx = np.asarray(action_indices, dtype=np.int64)
    state_idx = np.asarray(state_indices, dtype=np.int64)
    valid = state_idx < q.shape[0]
    action_idx = action_idx[valid]
    state_idx = state_idx[valid]
    if not np.any(valid):
        return safe
    anchor = q[state_idx].astype(safe.dtype, copy=False)
    if safe.ndim == 1:
        safe[action_idx] = anchor + scale * (safe[action_idx] - anchor)
    elif safe.ndim == 2:
        safe[:, action_idx] = anchor[None, :] + scale * (
            safe[:, action_idx] - anchor[None, :]
        )
    else:
        raise ValueError(f"Unsupported action shape for pause scaling: {safe.shape}")
    return safe


def _should_pause_for_safety(args, min_h, safety_info):
    if not args.pause_on_unsafe:
        return False, None
    threshold = float(args.pause_clearance_threshold)
    if min_h is not None and np.isfinite(float(min_h)) and float(min_h) < threshold:
        return True, "current_clearance"
    chunk_min = _safe_info_get(safety_info, "min_clearance")
    if chunk_min is not None and np.isfinite(float(chunk_min)) and float(chunk_min) < threshold:
        return True, "horizon_clearance"
    deform_safe = _safe_info_get(safety_info, "deform_safe")
    deform_min = _safe_info_get(safety_info, "deform_min_clearance")
    if (
        deform_safe is False
        and deform_min is not None
        and np.isfinite(float(deform_min))
        and float(deform_min) < threshold
    ):
        return True, "deform_clearance"
    return False, None


def _mujoco_model_data(task):
    mojo = getattr(task, "_mojo", None) or getattr(task, "mojo", None)
    if mojo is None:
        return None, None
    model = getattr(mojo, "model", None)
    data = getattr(mojo, "data", None)
    if model is not None and data is not None:
        return model, data
    physics = getattr(mojo, "physics", None)
    if physics is not None:
        return getattr(physics.model, "ptr", physics.model), getattr(physics.data, "ptr", physics.data)
    return None, None


def _mujoco_site_position(model, data, names: Sequence[str]):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None
    for name in names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3)
    return None


def _drawer_open_distance_and_fraction(task, model, data):
    distance = None
    joint_range = None
    try:
        import mujoco
        for name in ("base_cabinet_600/drawer_small_4", "drawer_small_4"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            distance = float(data.qpos[qpos_adr])
            joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64).reshape(2)
            break
    except Exception:  # noqa: BLE001
        distance = None

    if distance is None and hasattr(task, "cabinet_drawers"):
        try:
            distance = float(np.asarray(task.cabinet_drawers.get_state()).reshape(-1)[-1])
        except Exception:  # noqa: BLE001
            distance = None

    if distance is None:
        return None, None

    if (
        joint_range is not None
        and np.isfinite(joint_range).all()
        and float(joint_range[1]) > float(joint_range[0])
    ):
        fraction = (distance - float(joint_range[0])) / (float(joint_range[1]) - float(joint_range[0]))
    else:
        fraction = distance / 0.38
    return float(distance), float(np.clip(fraction, 0.0, 1.0))



def _diagnostic_task_state(env) -> dict[str, Any]:
    """Best-effort task progress/object state for failure diagnosis logs."""
    try:
        task = get_bigym_task(env)
    except Exception:  # noqa: BLE001
        task = _find_wrapped_env_with_attr(env, "mojo")
    if task is None:
        return {
            "drawer_open_distance": None,
            "drawer_open_fraction": None,
            "drawer_joint_position": None,
            "task_progress": None,
            "ee_object_distance": None,
            "object_state": None,
        }

    model, data = _mujoco_model_data(task)
    drawer_distance = None
    drawer_fraction = None
    handle_pos = None
    ee_pos = None
    ee_object_distance = None
    if model is not None and data is not None:
        drawer_distance, drawer_fraction = _drawer_open_distance_and_fraction(task, model, data)
        handle_pos = _mujoco_site_position(
            model,
            data,
            ("base_cabinet_600/drawer_small_4", "drawer_small_4"),
        )
        ee_pos = _mujoco_site_position(
            model,
            data,
            ("h1/right_end_effector", "right_end_effector"),
        )
        if handle_pos is not None and ee_pos is not None:
            ee_object_distance = float(np.linalg.norm(ee_pos - handle_pos))
    elif hasattr(task, "cabinet_drawers"):
        drawer_distance, drawer_fraction = _drawer_open_distance_and_fraction(task, model, data)

    task_progress = drawer_fraction if drawer_fraction is not None else drawer_distance
    object_state = {}
    if drawer_distance is not None:
        object_state["drawer_open_distance"] = float(drawer_distance)
    if drawer_fraction is not None:
        object_state["drawer_open_fraction"] = float(drawer_fraction)
    if handle_pos is not None:
        object_state["handle_pos"] = np.asarray(handle_pos, dtype=np.float64).astype(float).tolist()
    if ee_pos is not None:
        object_state["ee_pos"] = np.asarray(ee_pos, dtype=np.float64).astype(float).tolist()
    return {
        "drawer_open_distance": None if drawer_distance is None else float(drawer_distance),
        "drawer_open_fraction": None if drawer_fraction is None else float(drawer_fraction),
        "drawer_joint_position": None if drawer_distance is None else float(drawer_distance),
        "task_progress": None if task_progress is None else float(task_progress),
        "ee_object_distance": ee_object_distance,
        "object_state": object_state or None,
    }


def _diagnostic_progress_delta(before: dict[str, Any], after: dict[str, Any]) -> Optional[float]:
    before_progress = before.get("task_progress") if before else None
    after_progress = after.get("task_progress") if after else None
    if before_progress is None or after_progress is None:
        return None
    before_progress = float(before_progress)
    after_progress = float(after_progress)
    if not (np.isfinite(before_progress) and np.isfinite(after_progress)):
        return None
    return float(after_progress - before_progress)


def _finite_task_progress(task_state: Optional[dict[str, Any]]) -> Optional[float]:
    if not task_state:
        return None
    progress = task_state.get("task_progress")
    if progress is None:
        return None
    progress = float(progress)
    return progress if np.isfinite(progress) else None


def _post_recovery_task_guard_reanchor_allowed(phase_state, args):
    phase = str((phase_state or {}).get("phase", "pre_grasp"))
    allowed = set(getattr(args, "post_recovery_task_guard_reanchor_phases", ["grasp"]))
    return phase in allowed, phase


def _post_recovery_task_guard_ready(task_state, phase_state, args):
    phase = str((phase_state or {}).get("phase", "pre_grasp"))
    if phase == "pre_grasp":
        return False, "pre_grasp"

    ee_distance = None
    if task_state:
        ee_distance = task_state.get("ee_object_distance")
    if ee_distance is None and phase_state is not None:
        ee_distance = phase_state.get("ee_to_handle_dist")

    near_handle = False
    if ee_distance is not None:
        ee_distance = float(ee_distance)
        near_handle = (
            np.isfinite(ee_distance)
            and float(args.post_recovery_task_guard_max_ee_distance) > 0.0
            and ee_distance <= float(args.post_recovery_task_guard_max_ee_distance)
        )

    progress = _finite_task_progress(task_state)
    has_progress = (
        progress is not None
        and progress > float(args.post_recovery_task_guard_min_progress)
    )

    if near_handle:
        return True, "near_handle"
    if phase == "pull" and has_progress:
        return True, "task_progress"

    return False, phase


def _diagnostic_mode_flags(safety_info: dict, arm_delta: float, eps: float) -> dict[str, Any]:
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    recovery_phase = _safe_info_get(safety_info, "recovery_phase")
    fallback_step = bool(_safe_info_get(safety_info, "fallback_used"))
    committed_active = bool(_safe_info_get(safety_info, "committed_chunk_active"))
    committed_mode = _safe_info_get(safety_info, "committed_chunk_mode")
    brake_step = bool(
        mode in {"path_consistent_brake", "pause_on_unsafe", "pause_and_restart", "stop"}
        or source == "path_consistent_brake"
        or _safe_info_get(safety_info, "pause_reason") is not None
    )
    optimized_attempt_step = _safe_info_get(safety_info, "optimized_accepted") is not None
    optimized_accepted_step = bool(_safe_info_get(safety_info, "optimized_accepted"))
    deform_step = bool(
        not fallback_step
        and not brake_step
        and (
            mode == "emergency_deform_away"
            or committed_mode == "horizon_deform"
            or recovery_phase == "horizon_deform"
            or (source in {"explicit_recover_deform", "explicit_return_deform"} and optimized_accepted_step)
        )
    )
    recover_step = bool(
        not fallback_step
        and not brake_step
        and (
            committed_mode == "recover"
            or recovery_phase == "recover"
        )
    )
    act_step = bool(
        not deform_step
        and not recover_step
        and not brake_step
        and not fallback_step
        and arm_delta <= float(eps)
    )
    if fallback_step:
        step_mode = "fallback"
    elif brake_step:
        step_mode = "brake"
    elif recover_step:
        step_mode = "recover"
    elif deform_step:
        step_mode = "horizon_deform"
    else:
        step_mode = "act"
    return {
        "diagnostic_step_mode": step_mode,
        "act_step": act_step,
        "deform_step": deform_step,
        "recover_step": recover_step,
        "brake_step": brake_step,
        "fallback_step": fallback_step,
        "optimized_attempt_step": bool(optimized_attempt_step),
        "optimized_accepted_step": bool(optimized_accepted_step),
        "committed_active": committed_active,
    }


def _phase_reanchor_state(env, args):
    try:
        task = get_bigym_task(env)
    except Exception:  # noqa: BLE001
        task = _find_wrapped_env_with_attr(env, "mojo")
    if task is None:
        return None

    model, data = _mujoco_model_data(task)
    if model is None or data is None:
        return None

    handle_pos = _mujoco_site_position(
        model,
        data,
        ("base_cabinet_600/drawer_small_4", "drawer_small_4"),
    )
    ee_pos = _mujoco_site_position(
        model,
        data,
        ("h1/right_end_effector", "right_end_effector"),
    )
    if handle_pos is None or ee_pos is None:
        return None

    drawer_distance, drawer_fraction = _drawer_open_distance_and_fraction(task, model, data)
    if drawer_fraction is None:
        drawer_fraction = 0.0

    gripper_qpos = None
    gripper_closed = False
    robot = getattr(task, "robot", None)
    if robot is not None and hasattr(robot, "qpos_grippers"):
        try:
            qpos_grippers = np.asarray(robot.qpos_grippers, dtype=np.float64).reshape(-1)
            if qpos_grippers.size:
                gripper_qpos = float(qpos_grippers[-1])
                gripper_closed = gripper_qpos >= float(args.phase_reanchor_gripper_closed_threshold)
        except Exception:  # noqa: BLE001
            pass

    ee_to_handle_xy = handle_pos[:2] - ee_pos[:2]
    ee_to_handle_dist = float(np.linalg.norm(ee_to_handle_xy))
    if drawer_fraction >= float(args.phase_reanchor_done_threshold):
        phase = "done"
    elif (
        drawer_fraction >= float(args.phase_reanchor_pull_open_threshold)
        or (gripper_closed and ee_to_handle_dist <= float(args.phase_reanchor_grasp_dist))
    ):
        phase = "pull"
    elif ee_to_handle_dist <= float(args.phase_reanchor_grasp_dist):
        phase = "grasp"
    else:
        phase = "pre_grasp"

    return {
        "task": task,
        "phase": phase,
        "handle_pos": handle_pos,
        "ee_pos": ee_pos,
        "ee_to_handle_xy": ee_to_handle_xy,
        "ee_to_handle_dist": ee_to_handle_dist,
        "drawer_open_distance": drawer_distance,
        "drawer_open_fraction": float(drawer_fraction),
        "gripper_qpos": gripper_qpos,
        "gripper_closed": bool(gripper_closed),
    }


def _phase_reanchor_offset_xy(args, phase: str) -> np.ndarray:
    if phase == "pull":
        offset = args.phase_reanchor_pull_offset_xy
    elif phase == "grasp":
        offset = args.phase_reanchor_grasp_offset_xy
    else:
        offset = args.phase_reanchor_pregrasp_offset_xy
    return np.asarray(offset, dtype=np.float64).reshape(2)


def _should_start_phase_reanchor(args, step: int, state, drawer_history, cooldown_left: int):
    if not args.phase_reanchor or state is None or cooldown_left > 0:
        return False, None
    if state.get("phase") == "done" or step < int(args.phase_reanchor_check_after_steps):
        return False, None

    window = int(args.phase_reanchor_no_progress_window)
    if len(drawer_history) < window:
        return False, None
    recent = np.asarray(drawer_history[-window:], dtype=np.float64)
    if not np.isfinite(recent).any():
        return False, None
    progress = float(np.nanmax(recent) - np.nanmin(recent))
    return progress < float(args.phase_reanchor_min_drawer_progress), progress


def _phase_reanchor_action(env, safe_env_action, q_full, oscbf, args, state):
    chunk, was_single_chunk = _as_chunk(safe_env_action)
    raw_first = _raw_scaled_first_action(env, chunk[0])
    if raw_first is None:
        return None, None

    raw_first = np.asarray(raw_first, dtype=np.float32).reshape(-1).copy()
    phase = str(state.get("phase", "pre_grasp"))
    target_ee_xy = np.asarray(state["handle_pos"][:2], dtype=np.float64) + _phase_reanchor_offset_xy(args, phase)
    ee_xy = np.asarray(state["ee_pos"][:2], dtype=np.float64)
    ee_error_xy = target_ee_xy - ee_xy
    base_cmd_zeroed_reason = None
    if phase == "pull" or bool(state.get("gripper_closed")):
        base_cmd_xy = np.zeros(2, dtype=np.float32)
        base_cmd_zeroed_reason = "closed_gripper_or_pull_phase"
    else:
        base_cmd_xy = np.clip(
            float(args.phase_reanchor_base_gain) * ee_error_xy,
            -float(args.phase_reanchor_max_base_step),
            float(args.phase_reanchor_max_base_step),
        ).astype(np.float32)

    if raw_first.shape[0] >= 2:
        raw_first[:2] = base_cmd_xy
    if raw_first.shape[0] >= 4:
        raw_first[2:4] = 0.0

    action_idx = np.asarray(getattr(oscbf, "bigym_action_arm_indices", []), dtype=np.int64)
    state_idx = np.asarray(getattr(oscbf, "bigym_state_arm_indices", []), dtype=np.int64)
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    pair_count = min(action_idx.size, state_idx.size)
    action_idx = action_idx[:pair_count]
    state_idx = state_idx[:pair_count]
    valid = (action_idx < raw_first.shape[0]) & (state_idx < q.shape[0])
    if np.any(valid):
        raw_first[action_idx[valid]] = q[state_idx[valid]]

    normalized_first = _raw_action_to_normalized(env, raw_first)
    if normalized_first is None:
        return None, None
    normalized_first = np.asarray(normalized_first, dtype=np.float32).reshape(-1)
    if phase in {"grasp", "pull"}:
        normalized_first[-1] = float(np.clip(args.phase_reanchor_gripper_value, -1.0, 1.0))

    reanchored_chunk = np.repeat(normalized_first[None, :], chunk.shape[0], axis=0)
    info = {
        "phase": phase,
        "target_ee_xy": target_ee_xy.astype(float).tolist(),
        "ee_error_xy": ee_error_xy.astype(float).tolist(),
        "base_cmd_xy": base_cmd_xy.astype(float).tolist(),
        "base_cmd_zeroed_reason": base_cmd_zeroed_reason,
        "drawer_open_fraction": float(state.get("drawer_open_fraction", 0.0)),
        "ee_to_handle_dist": float(state.get("ee_to_handle_dist", np.nan)),
    }
    return _restore_action_shape(reanchored_chunk, was_single_chunk), info


def _as_chunk(action) -> tuple[np.ndarray, bool]:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        return action.reshape(1, -1), True
    if action.ndim == 2:
        return action, False
    raise ValueError(f"Unsupported action shape: {action.shape}")


def _restore_action_shape(chunk: np.ndarray, was_single: bool) -> np.ndarray:
    return chunk[0].copy() if was_single else chunk.copy()


def _safe_info_get(info: dict, key: str, default=None):
    value = info.get(key, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_float(value):
    if value is None:
        return None
    return float(value)


def _optional_bool(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.asarray(value).any())
    return bool(value)


def _optional_str(value):
    if value is None:
        return None
    return str(value)


def _unmodelled_robot_contact_reason(contact_pairs):
    if not contact_pairs:
        return None
    unmodelled_tokens = ("head", "helmet")
    for pair in contact_pairs:
        lower = str(pair).lower()
        if any(token in lower for token in unmodelled_tokens):
            return f"unmodelled_robot_contact:{pair}"
    return None


def _chunk_obs_with_q(obs, q_full: np.ndarray):
    if isinstance(obs, dict):
        chunk_obs = dict(obs)
    else:
        chunk_obs = {"obs": obs}
    chunk_obs["q"] = np.asarray(q_full, dtype=np.float32).reshape(-1)
    return chunk_obs


def _assert_chunk_properties(nominal_chunk, safe_chunk, arm_indices):
    nominal_chunk, _ = _as_chunk(nominal_chunk)
    safe_chunk, _ = _as_chunk(safe_chunk)
    if safe_chunk.shape != nominal_chunk.shape:
        raise AssertionError(
            f"Safe chunk shape {safe_chunk.shape} != nominal chunk shape {nominal_chunk.shape}"
        )
    if not np.isfinite(safe_chunk).all():
        raise AssertionError("Safe chunk contains non-finite values")
    for k in range(nominal_chunk.shape[0]):
        assert_action_properties(nominal_chunk[k], safe_chunk[k], arm_indices)


def _chunk_filter_advantage_metrics(
    nominal_chunk,
    safe_chunk,
    arm_indices,
    intervention_eps: float,
) -> dict[str, Optional[float]]:
    nominal_chunk, _ = _as_chunk(nominal_chunk)
    safe_chunk, _ = _as_chunk(safe_chunk)
    arm_indices = np.asarray(arm_indices, dtype=np.int64)

    nominal_arm = nominal_chunk[:, arm_indices]
    safe_arm = safe_chunk[:, arm_indices]
    delta_arm = safe_arm - nominal_arm
    step_arm_delta = np.linalg.norm(delta_arm, axis=1)
    edited = step_arm_delta > float(intervention_eps)
    edited_steps = np.flatnonzero(edited)
    chunk_arm_delta = float(np.linalg.norm(delta_arm))
    first_delta = float(step_arm_delta[0]) if step_arm_delta.size else 0.0
    future_delta = (
        float(np.linalg.norm(delta_arm[1:])) if delta_arm.shape[0] > 1 else 0.0
    )

    if nominal_arm.shape[0] > 1:
        nominal_variation = float(np.mean(np.linalg.norm(np.diff(nominal_arm, axis=0), axis=1)))
        safe_variation = float(np.mean(np.linalg.norm(np.diff(safe_arm, axis=0), axis=1)))
        edit_variation = float(np.mean(np.linalg.norm(np.diff(delta_arm, axis=0), axis=1)))
    else:
        nominal_variation = 0.0
        safe_variation = 0.0
        edit_variation = 0.0

    denom = max(chunk_arm_delta, 1e-12)
    return {
        "chunk_modified_fraction": float(np.mean(edited)) if edited.size else 0.0,
        "chunk_modified_steps": int(edited_steps.size),
        "chunk_first_modified_step": int(edited_steps[0]) if edited_steps.size else None,
        "chunk_last_modified_step": int(edited_steps[-1]) if edited_steps.size else None,
        "chunk_mean_step_arm_delta": float(np.mean(step_arm_delta)) if step_arm_delta.size else 0.0,
        "chunk_max_step_arm_delta": float(np.max(step_arm_delta)) if step_arm_delta.size else 0.0,
        "chunk_future_arm_delta": future_delta,
        "chunk_future_edit_fraction": float(future_delta / denom),
        "chunk_first_edit_fraction": float(first_delta / denom),
        "chunk_safe_arm_variation": safe_variation,
        "chunk_nominal_arm_variation": nominal_variation,
        "chunk_arm_variation_delta": float(safe_variation - nominal_variation),
        "chunk_edit_variation": edit_variation,
        "chunk_preemptive_intervention": bool(first_delta <= float(intervention_eps) and future_delta > float(intervention_eps)),
    }




def _path_consistency_metrics(safechunk, obs, nominal_chunk, safe_chunk) -> dict[str, Optional[float]]:
    try:
        nominal_q = np.asarray(
            safechunk.rollout_nominal_chunk(obs, nominal_chunk),
            dtype=np.float32,
        )
        safe_q = np.asarray(
            safechunk.rollout_nominal_chunk(obs, safe_chunk),
            dtype=np.float32,
        )
        if nominal_q.shape != safe_q.shape or nominal_q.ndim != 2:
            return {
                "path_mean_deviation": None,
                "path_max_deviation": None,
                "path_final_deviation": None,
            }
        state_idx = np.asarray(safechunk.controlled_state_indices, dtype=np.int64)
        valid = state_idx < nominal_q.shape[1]
        state_idx = state_idx[valid]
        if state_idx.size == 0:
            return {
                "path_mean_deviation": None,
                "path_max_deviation": None,
                "path_final_deviation": None,
            }
        step_deviation = np.linalg.norm(safe_q[:, state_idx] - nominal_q[:, state_idx], axis=1)
        return {
            "path_mean_deviation": float(np.mean(step_deviation)) if step_deviation.size else 0.0,
            "path_max_deviation": float(np.max(step_deviation)) if step_deviation.size else 0.0,
            "path_final_deviation": float(step_deviation[-1]) if step_deviation.size else 0.0,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Path consistency metric failed: %s", exc)
        return {
            "path_mean_deviation": None,
            "path_max_deviation": None,
            "path_final_deviation": None,
        }




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


def _sliced_safety_eval(safety_eval, start: int, end: int):
    if not isinstance(safety_eval, dict):
        return None
    sliced = dict(safety_eval)
    clearances = safety_eval.get("min_clearances")
    if clearances is None:
        return sliced
    try:
        arr = np.asarray(clearances, dtype=np.float32).reshape(-1)
        arr = arr[max(0, int(start)) : max(0, int(end))]
        sliced["min_clearances"] = arr
        if arr.size:
            sliced["min_clearance"] = float(np.min(arr))
    except Exception:  # noqa: BLE001
        pass
    return sliced


def _ee_xyz_from_q_seq(horizon_operator, q_seq):
    if horizon_operator is None or q_seq is None:
        return None
    try:
        q_seq = np.asarray(q_seq, dtype=np.float32)
        if q_seq.size == 0:
            return []
        ee_seq = horizon_operator.ee_pose_sequence(q_seq)
        if ee_seq is None:
            return None
        ee_seq = np.asarray(ee_seq, dtype=np.float32)
        if ee_seq.ndim == 3 and ee_seq.shape[-1] == 3:
            xyz = ee_seq[:, 0, :]
        elif ee_seq.ndim == 2 and ee_seq.shape[1] >= 3:
            xyz = ee_seq[:, :3]
        elif ee_seq.ndim == 1 and ee_seq.size >= 3:
            xyz = ee_seq.reshape(1, -1)[:, :3]
        else:
            return None
        return xyz.astype(float).tolist()
    except Exception as exc:  # noqa: BLE001
        logger.debug("EE trajectory extraction failed: %s", exc)
        return None


def _state_trace_payload(
    name: str,
    q_seq,
    action_chunk,
    horizon_operator,
    include_q_states: bool,
    safety_eval=None,
):
    q_arr = np.asarray(q_seq, dtype=np.float32)
    action_arr = None if action_chunk is None else np.asarray(action_chunk, dtype=np.float32)
    payload = {
        "name": name,
        "horizon": int(q_arr.shape[0]) if q_arr.ndim >= 1 else 0,
        "state_shape": list(q_arr.shape),
        "action_shape": None if action_arr is None else list(action_arr.shape),
        "ee_xyz": _ee_xyz_from_q_seq(horizon_operator, q_arr),
        "min_clearance": (
            None
            if not isinstance(safety_eval, dict) or safety_eval.get("min_clearance") is None
            else float(safety_eval.get("min_clearance"))
        ),
        "min_clearances": _clearance_sequence_payload(
            safety_eval,
            int(q_arr.shape[0]) if q_arr.ndim >= 1 else 0,
        ),
    }
    if action_arr is not None:
        payload["action_chunk"] = action_arr.astype(float).tolist()
    if include_q_states:
        payload["q_seq"] = q_arr.astype(float).tolist()
    return payload


def _human_arm_trajectory_sample(env, episode: int, step: int):
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return None

    base_env = _find_wrapped_env_with_attr(env, "mojo")
    if base_env is None:
        return None
    try:
        model = base_env.mojo.physics.model.ptr
        data = base_env.mojo.physics.data
    except Exception:  # noqa: BLE001
        try:
            model = base_env.mojo.model
            data = base_env.mojo.data
        except Exception:  # noqa: BLE001
            return None

    geoms = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        lower = name.lower()
        if not (
            lower.startswith("cylinder_arm/")
            or lower.endswith("upperarm_geom")
            or lower.endswith("forearm_geom")
        ):
            continue
        try:
            pos = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
        except Exception:  # noqa: BLE001
            continue
        geoms.append({"name": name, "pos": pos.astype(float).tolist()})

    if not geoms:
        return None
    centers = np.asarray([g["pos"] for g in geoms], dtype=np.float64)
    sample = {
        "episode": int(episode),
        "step": int(step),
        "time": float(getattr(data, "time", np.nan)),
        "center": np.mean(centers, axis=0).astype(float).tolist(),
        "geoms": geoms,
    }
    return sample


def _horizon_human_capsule_trace(horizon_operator, obs, horizon: int):
    if horizon_operator is None or horizon <= 0:
        return None
    try:
        human_obstacles = horizon_operator.oscbf._extract_human_obstacles(
            horizon_operator.env,
            obs,
        )
        capsule_a = np.asarray(human_obstacles["capsule_a"], dtype=np.float32)
        capsule_b = np.asarray(human_obstacles["capsule_b"], dtype=np.float32)
        capsule_radii = np.asarray(human_obstacles["capsule_radii"], dtype=np.float32)
        a_seq, b_seq, radii_eval, prediction_info = horizon_operator._human_capsule_rollout(
            capsule_a,
            capsule_b,
            capsule_radii,
            int(horizon),
        )
        centers = 0.5 * (np.asarray(a_seq, dtype=np.float32) + np.asarray(b_seq, dtype=np.float32))
        return {
            "capsule_a_world": np.asarray(a_seq, dtype=np.float32).astype(float).tolist(),
            "capsule_b_world": np.asarray(b_seq, dtype=np.float32).astype(float).tolist(),
            "capsule_centers_world": centers.astype(float).tolist(),
            "capsule_radii": np.asarray(radii_eval, dtype=np.float32).astype(float).tolist(),
            "prediction_info": _jsonable_trace_value(prediction_info),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Human capsule trajectory extraction failed: %s", exc)
        return None


def _should_log_chunk_trajectory_trace(
    args,
    safety_info: dict,
    nominal_action,
    safe_action,
    eps: float,
) -> bool:
    if not getattr(args, "log_chunk_trajectories", False):
        return False
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    committed_active = bool(_safe_info_get(safety_info, "committed_chunk_active"))
    optimized_known = _safe_info_get(safety_info, "optimized_accepted") is not None
    try:
        nominal, _ = _as_chunk(nominal_action)
        safe, _ = _as_chunk(safe_action)
        chunk_delta = float(np.linalg.norm(safe - nominal)) if safe.shape == nominal.shape else float("inf")
    except Exception:  # noqa: BLE001
        chunk_delta = 0.0
    return bool(
        committed_active
        or optimized_known
        or chunk_delta > float(eps)
        or mode in {
            "path_consistent_brake",
            "pause_on_unsafe",
            "pause_and_restart",
            "horizon_deform",
            "phase_reanchor",
            "committed_explicit_recovery",
            "recover",
            "recover_safe_prefix",
            "deform_safe_prefix",
        }
        or source in {
            "chunk_deform",
            "explicit_recover_deform",
            "explicit_return_deform",
            "committed_explicit_recovery",
            "path_consistent_brake",
            "sequential_oscbf_fallback",
        }
    )


def _segment_len_from_info(safety_info, *keys, default=0):
    for key in keys:
        value = _safe_info_get(safety_info, key)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except Exception:  # noqa: BLE001
            continue
    return int(default)


def _collect_chunk_trajectory_trace(
    *,
    args,
    episode: int,
    step: int,
    safechunk,
    horizon_operator,
    obs,
    nominal_chunk,
    generated_chunk,
    safety_info: dict,
    human_sample=None,
):
    try:
        nominal, _ = _as_chunk(nominal_chunk)
        generated, _ = _as_chunk(generated_chunk)
        nominal_q = safechunk.rollout_nominal_chunk(obs, nominal)
        nominal_eval = safechunk.evaluate_horizon_safety(obs, nominal_q)
        braked_chunk, brake_info = safechunk.path_consistent_brake(
            obs,
            nominal,
            nominal_eval,
        )
        braked_q = safechunk.rollout_nominal_chunk(obs, braked_chunk)
        brake_eval = safechunk.evaluate_horizon_safety(obs, braked_q)
        generated_q = safechunk.rollout_nominal_chunk(obs, generated)
        generated_eval = safechunk.evaluate_horizon_safety(obs, generated_q)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Chunk trajectory trace collection failed: %s", exc)
        return None

    include_q = bool(getattr(args, "chunk_trajectory_include_q_states", True))
    traces = {
        "nominal": _state_trace_payload(
            "nominal",
            nominal_q,
            nominal,
            horizon_operator,
            include_q,
            nominal_eval,
        ),
        "braking": _state_trace_payload(
            "braking",
            braked_q,
            braked_chunk,
            horizon_operator,
            include_q,
            brake_eval,
        ),
        "generated": _state_trace_payload(
            "generated",
            generated_q,
            generated,
            horizon_operator,
            include_q,
            generated_eval,
        ),
    }

    total = int(generated.shape[0]) if generated.ndim == 2 else 0
    deform_len = _segment_len_from_info(
        safety_info,
        "deform_chunk_length",
        "yield_chunk_length",
        default=0,
    )
    recover_len = _segment_len_from_info(
        safety_info,
        "recover_chunk_length",
        "return_chunk_length",
        default=0,
    )
    committed_mode = _safe_info_get(safety_info, "committed_chunk_mode")
    mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
    source = _safe_info_get(safety_info, "deformation_source")
    if deform_len <= 0 and total > 0 and committed_mode == "horizon_deform":
        deform_len = total
    elif deform_len <= 0 and recover_len <= 0 and total > 0 and (
        mode in {"horizon_deform", "deform_safe_prefix"}
        or source in {"chunk_deform", "explicit_recover_deform", "explicit_return_deform"}
    ):
        deform_len = total
    if recover_len <= 0 and total > 0 and committed_mode == "recover":
        recover_len = total

    deform_len = max(0, min(deform_len, total))
    recover_start = deform_len
    if recover_start >= total and committed_mode == "recover":
        recover_start = 0
    recover_len = max(0, min(recover_len, total - recover_start))

    if deform_len > 0:
        traces["deformed"] = _state_trace_payload(
            "deformed",
            np.asarray(generated_q, dtype=np.float32)[:deform_len],
            generated[:deform_len],
            horizon_operator,
            include_q,
            _sliced_safety_eval(generated_eval, 0, deform_len),
        )
    if recover_len > 0:
        end = recover_start + recover_len
        traces["recovery"] = _state_trace_payload(
            "recovery",
            np.asarray(generated_q, dtype=np.float32)[recover_start:end],
            generated[recover_start:end],
            horizon_operator,
            include_q,
            _sliced_safety_eval(generated_eval, recover_start, end),
        )

    horizon = max(
        int(np.asarray(nominal_q).shape[0]),
        int(np.asarray(generated_q).shape[0]),
    )
    record = {
        "episode": int(episode),
        "step": int(step),
        "condition": getattr(args, "condition", None),
        "safety_mode": mode,
        "deform_mode": _safe_info_get(safety_info, "deform_mode"),
        "recovery_phase": _safe_info_get(safety_info, "recovery_phase"),
        "deformation_source": source,
        "accepted_candidate_name": _safe_info_get(safety_info, "accepted_candidate_name"),
        "accepted_candidate_type": _safe_info_get(safety_info, "accepted_candidate_type"),
        "optimized_accepted": _safe_info_get(safety_info, "optimized_accepted"),
        "fallback_used": _safe_info_get(safety_info, "fallback_used"),
        "first_violation": _safe_info_get(safety_info, "first_violation"),
        "brake_info": _jsonable_trace_value(brake_info),
        "segment_lengths": {
            "deform": int(deform_len),
            "recover": int(recover_len),
            "recover_start": int(recover_start),
            "total": int(total),
        },
        "controlled_state_indices": _jsonable_trace_value(
            getattr(safechunk, "controlled_state_indices", None)
        ),
        "controlled_action_indices": _jsonable_trace_value(
            getattr(safechunk, "controlled_action_indices", None)
        ),
        "traces": traces,
        "human_arm_sample": _jsonable_trace_value(human_sample),
        "human_capsule_prediction": _horizon_human_capsule_trace(
            horizon_operator,
            obs,
            horizon,
        ),
    }
    return _jsonable_trace_value(record)


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


def _set_3d_axes_equal(ax, point_arrays):
    arrays = [np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in point_arrays if a is not None and np.asarray(a).size]
    if not arrays:
        return
    points = np.concatenate(arrays, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _save_chunk_trajectory_plot(
    path: Path,
    episode: int,
    trace_records: list[dict],
    human_samples: list[dict],
    max_events: int,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not import matplotlib for chunk trajectory plot: %s", exc)
        return None

    selected = trace_records
    if max_events > 0 and len(selected) > max_events:
        selected = selected[-max_events:]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    styles = {
        "nominal": ("0.55", "--", 1.0, 0.35),
        "braking": ("tab:orange", "-", 2.0, 0.75),
        "deformed": ("tab:blue", "-", 2.2, 0.85),
        "recovery": ("tab:green", "-", 2.2, 0.85),
        "generated": ("black", ":", 1.4, 0.55),
    }
    plotted_labels = set()
    point_arrays = []

    centers = []
    for sample in human_samples:
        center = sample.get("center") if isinstance(sample, dict) else None
        if center is None:
            continue
        try:
            center_arr = np.asarray(center, dtype=np.float64).reshape(3)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(center_arr).all():
            centers.append(center_arr)
    if centers:
        center_arr = np.stack(centers, axis=0)
        point_arrays.append(center_arr)
        ax.plot(
            center_arr[:, 0],
            center_arr[:, 1],
            center_arr[:, 2],
            color="crimson",
            linewidth=2.6,
            alpha=0.9,
            label="human arm center",
        )
        plotted_labels.add("human arm center")

    geom_names = sorted(
        {
            geom.get("name")
            for sample in human_samples
            if isinstance(sample, dict)
            for geom in sample.get("geoms", [])
            if isinstance(geom, dict) and geom.get("name")
        }
    )
    for geom_name in geom_names:
        points = []
        for sample in human_samples:
            if not isinstance(sample, dict):
                continue
            for geom in sample.get("geoms", []):
                if geom.get("name") != geom_name:
                    continue
                try:
                    pos = np.asarray(geom.get("pos"), dtype=np.float64).reshape(3)
                except Exception:  # noqa: BLE001
                    continue
                if np.isfinite(pos).all():
                    points.append(pos)
        if len(points) < 2:
            continue
        arr = np.stack(points, axis=0)
        point_arrays.append(arr)
        short_name = geom_name.split("/")[-1]
        label = short_name if short_name not in plotted_labels else None
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            color="lightcoral",
            linewidth=0.9,
            alpha=0.35,
            label=label,
        )
        if label:
            plotted_labels.add(label)

    for record in selected:
        traces = record.get("traces", {}) if isinstance(record, dict) else {}
        for name in ("braking", "deformed", "recovery", "generated", "nominal"):
            arr = _trace_xyz_array(traces.get(name))
            if arr is None or arr.shape[0] < 1:
                continue
            color, linestyle, linewidth, alpha = styles[name]
            label = name if name not in plotted_labels else None
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )
            if arr.shape[0] > 0:
                ax.scatter(
                    [arr[0, 0]],
                    [arr[0, 1]],
                    [arr[0, 2]],
                    color=color,
                    s=12,
                    alpha=min(1.0, alpha + 0.1),
                )
            point_arrays.append(arr)
            if label:
                plotted_labels.add(label)

    ax.set_title(f"Episode {episode:03d} SafeChunk 3D trajectories")
    ax.set_xlabel("x world")
    ax.set_ylabel("y world")
    ax.set_zlabel("z world")
    _set_3d_axes_equal(ax, point_arrays)
    if plotted_labels:
        ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)

def _horizon_risk_gap(current_h, horizon_min_clearance, eps: float = 1e-9):
    if current_h is None or horizon_min_clearance is None:
        return None, None, None
    current = float(current_h)
    horizon = float(horizon_min_clearance)
    if not np.isfinite(current) or not np.isfinite(horizon):
        return None, None, None
    clearance_drop = current - horizon
    risk_gap = max(0.0, clearance_drop)
    return float(risk_gap), bool(risk_gap > eps), float(clearance_drop)


def _metric_safety_violation(metric: StepMetrics) -> bool:
    if metric.h_violation is not None:
        return bool(metric.h_violation)
    return False


def _metric_is_brake_step(metric: StepMetrics) -> bool:
    mode = metric.safety_mode
    source = metric.deformation_source
    if mode in {"path_consistent_brake", "pause_on_unsafe", "pause_and_restart", "stop"}:
        return True
    if source == "path_consistent_brake":
        return True
    return metric.pause_reason is not None


def _metric_is_deformation_step(metric: StepMetrics) -> bool:
    mode = metric.safety_mode
    source = metric.deformation_source
    if mode in {"horizon_deform", "chunk_deform", "emergency_deform_away"}:
        return True
    return source == "chunk_deform"


def _resume_latency_after_human_exit(metrics: list[StepMetrics], default_dt: float = 0.05):
    if not metrics:
        return None
    phases = [m.human_phase for m in metrics]
    if "done" not in phases or not any(phase in {"enter", "hold", "exit"} for phase in phases):
        return None

    first_done = next(i for i, phase in enumerate(phases) if phase == "done")
    for resume_idx in range(first_done, len(metrics)):
        if not _metric_is_brake_step(metrics[resume_idx]) and not _metric_is_deformation_step(metrics[resume_idx]):
            return float((resume_idx - first_done) * default_dt)
    return None

def summarise_chunk_episode(metrics: list[StepMetrics], diagnostics_cfg: Optional[dict[str, float]] = None) -> dict:
    summary = summarise_episode(metrics)
    if len(metrics) == 0:
        return summary

    diagnostics_cfg = dict(diagnostics_cfg or {})
    large_arm_delta_threshold = float(diagnostics_cfg.get("large_arm_delta_threshold", 3.0))
    large_base_delta_threshold = float(diagnostics_cfg.get("large_base_delta_threshold", 0.5))
    low_act_ratio_threshold = float(diagnostics_cfg.get("low_act_ratio_threshold", 0.3))
    high_fallback_ratio_threshold = float(diagnostics_cfg.get("high_fallback_ratio_threshold", 0.5))
    success_threshold = float(diagnostics_cfg.get("success_threshold", 0.9))

    chunk_arm_delta = np.asarray([m.chunk_arm_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_non_arm_delta = np.asarray([m.chunk_non_arm_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_full_delta = np.asarray([m.chunk_full_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_interventions = np.asarray([m.intervention_active for m in metrics], dtype=np.float32)
    chunk_modified_fraction = np.asarray([m.chunk_modified_fraction or 0.0 for m in metrics], dtype=np.float32)
    chunk_modified_steps = np.asarray([m.chunk_modified_steps or 0 for m in metrics], dtype=np.float32)
    chunk_mean_step_arm_delta = np.asarray([m.chunk_mean_step_arm_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_max_step_arm_delta = np.asarray([m.chunk_max_step_arm_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_future_edit_fraction = np.asarray([m.chunk_future_edit_fraction or 0.0 for m in metrics], dtype=np.float32)
    chunk_first_edit_fraction = np.asarray([m.chunk_first_edit_fraction or 0.0 for m in metrics], dtype=np.float32)
    chunk_safe_arm_variation = np.asarray([m.chunk_safe_arm_variation or 0.0 for m in metrics], dtype=np.float32)
    chunk_nominal_arm_variation = np.asarray([m.chunk_nominal_arm_variation or 0.0 for m in metrics], dtype=np.float32)
    chunk_arm_variation_delta = np.asarray([m.chunk_arm_variation_delta or 0.0 for m in metrics], dtype=np.float32)
    chunk_edit_variation = np.asarray([m.chunk_edit_variation or 0.0 for m in metrics], dtype=np.float32)
    path_mean_deviation = np.asarray([m.path_mean_deviation for m in metrics if m.path_mean_deviation is not None], dtype=np.float32)
    path_max_deviation = np.asarray([m.path_max_deviation for m in metrics if m.path_max_deviation is not None], dtype=np.float32)
    path_final_deviation = np.asarray([m.path_final_deviation for m in metrics if m.path_final_deviation is not None], dtype=np.float32)
    chunk_preemptive_interventions = np.asarray([bool(m.chunk_preemptive_intervention) for m in metrics], dtype=np.float32)
    horizon_risk_gaps = [m.horizon_risk_gap for m in metrics if m.horizon_risk_gap is not None]
    horizon_clearance_drops = [m.horizon_clearance_drop for m in metrics if m.horizon_clearance_drop is not None]
    horizon_risk_gap_active = [m.horizon_risk_gap_active for m in metrics if m.horizon_risk_gap_active is not None]
    horizon_only_risk = [
        bool(m.horizon_risk_gap_active) and not bool(m.h_violation)
        for m in metrics
        if m.horizon_risk_gap_active is not None and m.h_violation is not None
    ]
    first_modified_steps = [m.chunk_first_modified_step for m in metrics if m.chunk_first_modified_step is not None]
    deform_norms = [m.deformation_norm for m in metrics if m.deformation_norm is not None]
    deform_safe = [m.deform_safe for m in metrics if m.deform_safe is not None]
    optimized_records = [m for m in metrics if m.optimized_accepted is not None]
    optimized_attempts = [m.optimized_accepted for m in optimized_records]
    optimized_safe = [bool(m.deform_safe) for m in optimized_records]
    recoverable_checks = [
        bool(m.is_recoverable) for m in optimized_records if m.is_recoverable is not None
    ]
    fallback_steps = [m.fallback_used for m in metrics if m.fallback_used is not None]
    rejection_causes = [m.rejection_cause for m in optimized_records]
    deform_stage_checks = [m.deform_stage_accepted for m in metrics if m.deform_stage_accepted is not None]
    recover_checks = [m.recover_accepted for m in metrics if m.recover_accepted is not None]
    recover_reject_reasons = [
        m.recover_reject_reason
        for m in metrics
        if m.recover_reject_reason is not None
    ]
    direct_rejoin_attempted_steps = int(
        np.sum([bool(m.direct_rejoin_attempted) for m in metrics])
    )
    direct_rejoin_rejected_steps = int(
        np.sum([bool(m.direct_rejoin_rejected) for m in metrics])
    )
    detour_rejoin_attempted_steps = int(
        np.sum([bool(m.detour_rejoin_attempted) for m in metrics])
    )
    detour_rejoin_accepted_steps = int(
        np.sum([bool(m.detour_rejoin_accepted) for m in metrics])
    )
    delayed_rejoin_active_steps = int(
        np.sum([bool(m.delayed_rejoin_active) for m in metrics])
    )
    repeated_unsafe_target_steps = int(
        np.sum([bool(m.repeated_unsafe_target) for m in metrics])
    )
    post_recovery_act_window_steps = int(
        np.sum([bool(m.post_recovery_act_window_active) for m in metrics])
    )
    post_recovery_act_window_interrupted_steps = int(
        np.sum([bool(m.post_recovery_act_window_interrupted) for m in metrics])
    )
    cached_motion = [m.cached_motion_active for m in metrics if m.cached_motion_active is not None]
    resumed_indices = [
        m.resumed_from_cached_index
        for m in metrics
        if m.resumed_from_cached_index is not None
    ]

    def finite_metric(name):
        vals = [getattr(m, name) for m in metrics if getattr(m, name) is not None]
        vals = [float(v) for v in vals if np.isfinite(float(v))]
        return np.asarray(vals, dtype=np.float32)

    q_rejoin_dist = finite_metric("q_rejoin_dist")
    qd_rejoin_dist = finite_metric("qd_rejoin_dist")
    ee_rejoin_dist = finite_metric("ee_rejoin_dist")
    rejoin_q_eval_time_ms = finite_metric("rejoin_q_eval_time_ms")
    rejoin_qd_eval_time_ms = finite_metric("rejoin_qd_eval_time_ms")
    ee_nom_cache_time_ms = finite_metric("ee_nom_cache_time_ms")
    ee_final_check_time_ms = finite_metric("ee_final_check_time_ms")
    deform_stage_min_clearance = finite_metric("deform_stage_min_clearance")
    recover_min_clearance = finite_metric("recover_min_clearance")
    recover_rejoin_loss = finite_metric("recover_rejoin_loss")
    recover_path_min_clearance = finite_metric("recover_path_min_clearance")
    recover_immediate_clearance = finite_metric("recover_immediate_clearance")
    recover_prefix_min_clearance = finite_metric("recover_prefix_min_clearance")
    committed_clearance_prediction_error = finite_metric("clearance_prediction_error")
    committed_planned_vs_actual_q_error = finite_metric("planned_vs_actual_q_error")
    committed_human_motion_since_plan = finite_metric("human_motion_since_plan")
    committed_accepted_clearance_margin = finite_metric("accepted_clearance_margin")
    committed_state_error = finite_metric("committed_state_error")
    planning_vs_replay_clearance_post_error = finite_metric(
        "planning_vs_replay_clearance_post_error"
    )
    planning_vs_replay_human_error = finite_metric("planning_vs_replay_human_error")
    actual_vs_planned_post_q_error = finite_metric("actual_vs_planned_post_q_error")
    recover_projection_on_nominal = finite_metric("recover_projection_on_nominal")
    recover_cosine_to_nominal = finite_metric("recover_cosine_to_nominal")
    recover_task_progress_score = finite_metric("recover_task_progress_score")
    hold_horizon_min_clearance = finite_metric("hold_horizon_min_clearance")
    task_progress = finite_metric("task_progress")
    task_progress_delta = finite_metric("task_progress_delta")
    total_steps = len(metrics)
    act_step_flags = np.asarray([bool(m.act_step) for m in metrics], dtype=np.bool_)
    deform_step_flags = np.asarray([bool(m.deform_step) for m in metrics], dtype=np.bool_)
    recover_step_flags = np.asarray([bool(m.recover_step) for m in metrics], dtype=np.bool_)
    brake_step_flags = np.asarray([bool(m.brake_step) for m in metrics], dtype=np.bool_)
    fallback_step_flags = np.asarray([bool(m.fallback_step) for m in metrics], dtype=np.bool_)
    optimized_attempt_step_flags = np.asarray([bool(m.optimized_attempt_step) for m in metrics], dtype=np.bool_)
    optimized_accepted_step_flags = np.asarray([bool(m.optimized_accepted_step) for m in metrics], dtype=np.bool_)
    temporary_wait_steps = int(np.sum([bool(m.temporary_wait_step) for m in metrics]))
    resume_after_wait_count = int(np.sum([bool(m.resume_act_after_wait) for m in metrics]))
    deform_after_persistent_block_count = int(np.sum([bool(m.deform_after_persistent_block) for m in metrics]))
    deform_suppressed_by_temporary_wait_count = int(
        np.sum([bool(m.deform_suppressed_by_temporary_wait) for m in metrics])
    )
    recovery_failure_streak_vals = [
        int(m.recovery_failure_streak_max)
        for m in metrics
        if m.recovery_failure_streak_max is not None
    ]
    recovery_failure_streak_max = (
        int(np.max(recovery_failure_streak_vals))
        if recovery_failure_streak_vals
        else 0
    )

    def max_int_metric(name):
        vals = [
            int(getattr(m, name))
            for m in metrics
            if getattr(m, name) is not None
        ]
        return int(np.max(vals)) if vals else 0

    deform_replan_count = max_int_metric("deform_replan_count")
    recover_replan_count = max_int_metric("recover_replan_count")
    recovery_replan_count = max_int_metric("recovery_replan_count")
    stale_recovery_suppressed_count = max_int_metric("stale_recovery_suppressed_count")
    recovery_target_infeasible_count = max_int_metric("recovery_target_infeasible_count")
    emergency_brake_steps = max_int_metric("emergency_brake_steps")
    optimized_candidate_count = max_int_metric("optimized_candidate_count")
    optimized_solution_count = max_int_metric("optimized_solution_count")
    fallback_candidate_count = max_int_metric("fallback_candidate_count")
    fallback_candidate_accepted_count = max_int_metric("fallback_candidate_accepted_count")
    optimized_rejected_count = max_int_metric("optimized_rejected_count")
    deform_candidate_count = max_int_metric("deform_candidate_count")
    deform_accepted_count = max_int_metric("deform_accepted_count")
    deform_rejected_count = max_int_metric("deform_rejected_count")
    recover_candidate_count = max_int_metric("recover_candidate_count")
    recover_accepted_count = max_int_metric("recover_accepted_count")
    recover_rejected_count = max_int_metric("recover_rejected_count")
    safe_corridor_recovery_count = max_int_metric("safe_corridor_recovery_count")
    direct_rejoin_attempt_count = max_int_metric("direct_rejoin_attempt_count")
    direct_rejoin_reject_count = max_int_metric("direct_rejoin_reject_count")
    detour_rejoin_attempt_count = max_int_metric("detour_rejoin_attempt_count")
    detour_rejoin_accept_count = max_int_metric("detour_rejoin_accept_count")
    delayed_rejoin_count = max_int_metric("delayed_rejoin_count")
    recover_path_unsafe_count = max_int_metric("recover_path_unsafe_count")
    recovery_path_failure_streak_max = max_int_metric("recovery_path_failure_streak_max")
    repeated_unsafe_target_count = max_int_metric("repeated_unsafe_target_count")
    post_recovery_act_window_count = max_int_metric("post_recovery_act_window_count")
    post_recovery_act_window_interrupted_count = max_int_metric("post_recovery_act_window_interrupted_count")
    safe_prefix_accepted_count = max_int_metric("safe_prefix_accepted_count")
    first_action_only_accepted_count = max_int_metric("first_action_only_accepted_count")
    immediate_hard_reject_count = max_int_metric("immediate_hard_reject_count")
    no_safe_prefix_reject_count = max_int_metric("no_safe_prefix_reject_count")
    horizon_margin_reject_count = max_int_metric("horizon_margin_reject_count")
    accepted_deform_steps = max_int_metric("accepted_deform_steps")
    accepted_recover_steps = max_int_metric("accepted_recover_steps")
    fallback_brake_after_reject_count = max_int_metric("fallback_brake_after_reject_count")
    nominal_rejoin_available_count = max_int_metric("nominal_rejoin_available_count")
    nominal_rejoin_suppressed_count = max_int_metric("nominal_rejoin_suppressed_count")
    stale_nominal_rejoin_suppressed_count = max_int_metric("stale_nominal_rejoin_suppressed_count")
    nominal_prefix_unsafe_suppressed_count = max_int_metric("nominal_prefix_unsafe_suppressed_count")
    recover_positive_projection_count = max_int_metric("recover_positive_projection_count")
    recover_nonpositive_projection_count = max_int_metric("recover_nonpositive_projection_count")
    emergency_deform_away_steps = max_int_metric("emergency_deform_away_steps")
    emergency_deform_away_count = max_int_metric("emergency_deform_away_count")
    hold_unsafe_count = max_int_metric("hold_unsafe_count")
    hold_predicted_contact_count = max_int_metric("hold_predicted_contact_count")
    contact_during_hold_count = max_int_metric("contact_during_hold_count")
    contact_during_brake_count = max_int_metric("contact_during_brake_count")
    contact_during_deform_count = max_int_metric("contact_during_deform_count")
    contact_during_recover_count = max_int_metric("contact_during_recover_count")

    def mean_progress_for(flag_name):
        vals = [
            m.task_progress_delta
            for m in metrics
            if bool(getattr(m, flag_name))
            and m.task_progress_delta is not None
            and np.isfinite(float(m.task_progress_delta))
        ]
        return float(np.mean(vals)) if vals else None

    act_steps = int(np.sum(act_step_flags))
    deform_steps = int(np.sum(deform_step_flags))
    recover_steps = int(np.sum(recover_step_flags))
    brake_step_count = int(np.sum(brake_step_flags))
    fallback_step_count = int(np.sum(fallback_step_flags))
    optimized_attempt_steps = int(np.sum(optimized_attempt_step_flags))
    optimized_accepted_steps = int(np.sum(optimized_accepted_step_flags))
    committed_chunk_started_count = int(
        np.sum([bool(m.committed_chunk_started) for m in metrics])
    )
    committed_chunk_completed_count = int(
        np.sum([bool(m.committed_chunk_completed) for m in metrics])
    )
    committed_state_mismatch_abort_count = int(
        np.sum([bool(m.committed_aborted_due_to_state_mismatch) for m in metrics])
    )
    committed_chunk_abort_count = int(
        np.sum(
            [
                bool(m.committed_aborted_due_to_safety)
                or bool(m.committed_aborted_due_to_state_mismatch)
                for m in metrics
            ]
        )
    )
    committed_repaired_step_count = int(
        np.sum([bool(m.committed_repaired_step) for m in metrics])
    )
    committed_abort_due_to_human_motion_count = int(
        np.sum([bool(m.committed_abort_due_to_human_motion) for m in metrics])
    )
    committed_abort_due_to_prediction_error_count = int(
        np.sum([bool(m.committed_abort_due_to_prediction_error) for m in metrics])
    )
    committed_abort_due_to_safety_semantics_mismatch_count = int(
        np.sum(
            [
                bool(m.committed_abort_due_to_safety_semantics_mismatch)
                for m in metrics
            ]
        )
    )
    committed_deform_steps_executed = int(
        np.sum([m.deform_steps_executed or 0 for m in metrics])
    )
    committed_recover_steps_executed = int(
        np.sum([m.recover_steps_executed or 0 for m in metrics])
    )
    resume_from_committed_rejoin_count = int(
        np.sum([bool(m.resume_from_committed_rejoin) for m in metrics])
    )
    recovery_action_history_reset_count = int(
        np.sum([bool(m.recovery_action_history_reset) for m in metrics])
    )
    contact_during_hold_count = int(np.sum([bool(m.contact_during_hold) for m in metrics]))
    contact_during_brake_count = int(np.sum([bool(m.contact_during_brake) for m in metrics]))
    contact_during_deform_count = int(np.sum([bool(m.contact_during_deform) for m in metrics]))
    contact_during_recover_count = int(np.sum([bool(m.contact_during_recover) for m in metrics]))
    act_ratio = float(act_steps / total_steps) if total_steps else None
    safety_mode_ratio = (
        float((deform_steps + recover_steps + brake_step_count + fallback_step_count) / total_steps)
        if total_steps
        else None
    )
    fallback_ratio = float(fallback_step_count / total_steps) if total_steps else None
    final_task_progress = float(task_progress[-1]) if task_progress.size else None
    max_task_progress = float(np.max(task_progress)) if task_progress.size else None
    mean_chunk_arm_delta_for_failure = float(np.mean(chunk_arm_delta)) if chunk_arm_delta.size else 0.0
    accepted_recover_chunks_not_executed = bool(
        recover_checks and np.sum(recover_checks) > 0 and recover_steps == 0
    )
    diagnostic_warning = (
        "accepted_recover_chunks_not_executed"
        if accepted_recover_chunks_not_executed
        else None
    )

    if act_ratio is not None and act_ratio < low_act_ratio_threshold:
        likely_failure_cause = "low_act_utilization"
    elif (
        max_task_progress is not None
        and final_task_progress is not None
        and max_task_progress > success_threshold * 0.7
        and final_task_progress < max_task_progress * 0.5
    ):
        likely_failure_cause = "progress_lost_after_intervention"
    elif mean_chunk_arm_delta_for_failure > large_arm_delta_threshold:
        likely_failure_cause = "large_deformation_ood"
    elif fallback_ratio is not None and fallback_ratio > high_fallback_ratio_threshold:
        likely_failure_cause = "fallback_braking_timeout"
    else:
        likely_failure_cause = "unknown"

    modes = [m.safety_mode for m in metrics]
    pause_reasons = [m.pause_reason for m in metrics]
    sources = [m.deformation_source for m in metrics]
    robot_human_distances = [m.min_robot_human_distance for m in metrics if m.min_robot_human_distance is not None]
    drawer_open_distances = [m.drawer_open_distance for m in metrics if m.drawer_open_distance is not None]
    safety_violations = [_metric_safety_violation(m) for m in metrics]
    brake_steps = [_metric_is_brake_step(m) for m in metrics]
    deformation_steps = [_metric_is_deformation_step(m) for m in metrics]
    phase_reanchor_steps = [m.safety_mode == "phase_reanchor" for m in metrics]
    gripper_latched_steps = int(np.sum([bool(m.gripper_latched) for m in metrics]))
    post_recovery_task_guard_steps = int(
        np.sum([bool(m.post_recovery_task_guard_active) for m in metrics])
    )
    post_recovery_reanchor_started_count = int(
        np.sum([bool(m.post_recovery_reanchor_started) for m in metrics])
    )
    post_recovery_progress_regression_count = int(
        np.sum(
            [
                (m.post_recovery_progress_regression or 0.0) > 0.0
                for m in metrics
            ]
        )
    )

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
            "mean_chunk_modified_fraction": float(np.mean(chunk_modified_fraction)),
            "mean_chunk_modified_steps": float(np.mean(chunk_modified_steps)),
            "mean_chunk_first_modified_step": float(np.mean(first_modified_steps)) if first_modified_steps else None,
            "mean_chunk_mean_step_arm_delta": float(np.mean(chunk_mean_step_arm_delta)),
            "max_chunk_step_arm_delta": float(np.max(chunk_max_step_arm_delta)),
            "mean_chunk_future_edit_fraction": float(np.mean(chunk_future_edit_fraction)),
            "mean_chunk_first_edit_fraction": float(np.mean(chunk_first_edit_fraction)),
            "mean_chunk_safe_arm_variation": float(np.mean(chunk_safe_arm_variation)),
            "mean_chunk_nominal_arm_variation": float(np.mean(chunk_nominal_arm_variation)),
            "mean_chunk_arm_variation_delta": float(np.mean(chunk_arm_variation_delta)),
            "mean_chunk_edit_variation": float(np.mean(chunk_edit_variation)),
            "mean_path_deviation": float(np.mean(path_mean_deviation)) if path_mean_deviation.size else None,
            "max_path_deviation": float(np.max(path_max_deviation)) if path_max_deviation.size else None,
            "mean_final_path_deviation": float(np.mean(path_final_deviation)) if path_final_deviation.size else None,
            "chunk_preemptive_intervention_frequency": float(np.mean(chunk_preemptive_interventions)),
            "mean_horizon_risk_gap": float(np.mean(horizon_risk_gaps)) if horizon_risk_gaps else None,
            "max_horizon_risk_gap": float(np.max(horizon_risk_gaps)) if horizon_risk_gaps else None,
            "mean_horizon_clearance_drop": float(np.mean(horizon_clearance_drops)) if horizon_clearance_drops else None,
            "horizon_risk_gap_rate": float(np.mean(horizon_risk_gap_active)) if horizon_risk_gap_active else None,
            "horizon_only_risk_rate": float(np.mean(horizon_only_risk)) if horizon_only_risk else None,
            "mean_deformation_norm": float(np.mean(deform_norms)) if deform_norms else None,
            "deform_safe_rate": float(np.mean(deform_safe)) if deform_safe else None,
            "optimized_attempts": int(len(optimized_records)),
            "optimized_safe_count": int(np.sum(optimized_safe)) if optimized_records else 0,
            "optimized_recoverable_count": int(np.sum(recoverable_checks)) if recoverable_checks else 0,
            "optimized_accepted_count": int(np.sum(optimized_attempts)) if optimized_attempts else 0,
            "rejected_unsafe_count": int(np.sum([c == "unsafe" for c in rejection_causes])),
            "rejected_unrecoverable_count": int(np.sum([c == "unrecoverable" for c in rejection_causes])),
            "rejected_both_count": int(np.sum([c == "unsafe_and_unrecoverable" for c in rejection_causes])),
            "fallback_used_count": int(np.sum(fallback_steps)) if fallback_steps else 0,
            "deform_stage_accepted_count": int(np.sum(deform_stage_checks)) if deform_stage_checks else 0,
            "recover_accepted_count": int(np.sum(recover_checks)) if recover_checks else 0,
            "cached_motion_active_count": int(np.sum(cached_motion)) if cached_motion else 0,
            "resumed_from_cached_count": int(len(resumed_indices)),
            "mean_deform_stage_min_clearance": float(np.mean(deform_stage_min_clearance)) if deform_stage_min_clearance.size else None,
            "mean_recover_min_clearance": float(np.mean(recover_min_clearance)) if recover_min_clearance.size else None,
            "mean_recover_rejoin_loss": float(np.mean(recover_rejoin_loss)) if recover_rejoin_loss.size else None,
            "mean_recover_path_min_clearance": float(np.mean(recover_path_min_clearance)) if recover_path_min_clearance.size else None,
            "min_recover_path_min_clearance": float(np.min(recover_path_min_clearance)) if recover_path_min_clearance.size else None,
            "mean_recover_immediate_clearance": float(np.mean(recover_immediate_clearance)) if recover_immediate_clearance.size else None,
            "mean_recover_prefix_min_clearance": float(np.mean(recover_prefix_min_clearance)) if recover_prefix_min_clearance.size else None,
            "optimized_attempt_count": int(len(optimized_attempts)),
            "optimized_accept_rate": float(np.mean(optimized_attempts)) if optimized_attempts else None,
            "recoverable_rate": float(np.mean(recoverable_checks)) if recoverable_checks else None,
            "fallback_used_rate": float(np.mean(fallback_steps)) if fallback_steps else None,
            "mean_q_rejoin_dist": float(np.mean(q_rejoin_dist)) if q_rejoin_dist.size else None,
            "max_q_rejoin_dist": float(np.max(q_rejoin_dist)) if q_rejoin_dist.size else None,
            "mean_qd_rejoin_dist": float(np.mean(qd_rejoin_dist)) if qd_rejoin_dist.size else None,
            "max_qd_rejoin_dist": float(np.max(qd_rejoin_dist)) if qd_rejoin_dist.size else None,
            "mean_ee_rejoin_dist": float(np.mean(ee_rejoin_dist)) if ee_rejoin_dist.size else None,
            "max_ee_rejoin_dist": float(np.max(ee_rejoin_dist)) if ee_rejoin_dist.size else None,
            "mean_rejoin_q_eval_time_ms": float(np.mean(rejoin_q_eval_time_ms)) if rejoin_q_eval_time_ms.size else None,
            "mean_rejoin_qd_eval_time_ms": float(np.mean(rejoin_qd_eval_time_ms)) if rejoin_qd_eval_time_ms.size else None,
            "mean_ee_nom_cache_time_ms": float(np.mean(ee_nom_cache_time_ms)) if ee_nom_cache_time_ms.size else None,
            "mean_ee_final_check_time_ms": float(np.mean(ee_final_check_time_ms)) if ee_final_check_time_ms.size else None,
            "act_steps": act_steps,
            "deform_steps": deform_steps,
            "recover_steps": recover_steps,
            "brake_steps": brake_step_count,
            "fallback_steps": fallback_step_count,
            "optimized_attempt_steps": optimized_attempt_steps,
            "optimized_accepted_steps": optimized_accepted_steps,
            "committed_chunk_started_count": committed_chunk_started_count,
            "committed_chunk_completed_count": committed_chunk_completed_count,
            "committed_chunk_abort_count": committed_chunk_abort_count,
            "committed_repaired_step_count": committed_repaired_step_count,
            "committed_abort_due_to_human_motion_count": committed_abort_due_to_human_motion_count,
            "committed_abort_due_to_prediction_error_count": committed_abort_due_to_prediction_error_count,
            "committed_abort_due_to_safety_semantics_mismatch_count": committed_abort_due_to_safety_semantics_mismatch_count,
            "committed_state_mismatch_abort_count": committed_state_mismatch_abort_count,
            "mean_planning_vs_replay_clearance_post_error": (
                float(np.mean(planning_vs_replay_clearance_post_error))
                if planning_vs_replay_clearance_post_error.size
                else None
            ),
            "mean_planning_vs_replay_human_error": (
                float(np.mean(planning_vs_replay_human_error))
                if planning_vs_replay_human_error.size
                else None
            ),
            "mean_actual_vs_planned_post_q_error": (
                float(np.mean(actual_vs_planned_post_q_error))
                if actual_vs_planned_post_q_error.size
                else None
            ),
            "mean_committed_state_error": (
                float(np.mean(committed_state_error))
                if committed_state_error.size
                else None
            ),
            "max_committed_state_error": (
                float(np.max(committed_state_error))
                if committed_state_error.size
                else None
            ),
            "mean_committed_clearance_prediction_error": (
                float(np.mean(committed_clearance_prediction_error))
                if committed_clearance_prediction_error.size
                else None
            ),
            "mean_committed_planned_vs_actual_q_error": (
                float(np.mean(committed_planned_vs_actual_q_error))
                if committed_planned_vs_actual_q_error.size
                else None
            ),
            "mean_committed_human_motion_since_plan": (
                float(np.mean(committed_human_motion_since_plan))
                if committed_human_motion_since_plan.size
                else None
            ),
            "mean_committed_accepted_clearance_margin": (
                float(np.mean(committed_accepted_clearance_margin))
                if committed_accepted_clearance_margin.size
                else None
            ),
            "committed_deform_steps_executed": committed_deform_steps_executed,
            "committed_recover_steps_executed": committed_recover_steps_executed,
            "resume_from_committed_rejoin_count": resume_from_committed_rejoin_count,
            "recovery_action_history_reset_count": recovery_action_history_reset_count,
            "accepted_recover_chunks_not_executed": accepted_recover_chunks_not_executed,
            "diagnostic_warning": diagnostic_warning,
            "temporary_wait_steps": temporary_wait_steps,
            "resume_after_wait_count": resume_after_wait_count,
            "deform_after_persistent_block_count": deform_after_persistent_block_count,
            "deform_suppressed_by_temporary_wait_count": deform_suppressed_by_temporary_wait_count,
            "recovery_failure_streak_max": recovery_failure_streak_max,
            "deform_replan_count": deform_replan_count,
            "recover_replan_count": recover_replan_count,
            "recovery_replan_count": recovery_replan_count,
            "stale_recovery_suppressed_count": stale_recovery_suppressed_count,
            "recovery_target_infeasible_count": recovery_target_infeasible_count,
            "emergency_brake_steps": emergency_brake_steps,
            "optimized_candidate_count": optimized_candidate_count,
            "optimized_solution_count": optimized_solution_count,
            "fallback_candidate_count": fallback_candidate_count,
            "fallback_candidate_accepted_count": fallback_candidate_accepted_count,
            "optimized_rejected_count": optimized_rejected_count,
            "deform_candidate_count": deform_candidate_count,
            "deform_accepted_count": deform_accepted_count,
            "deform_rejected_count": deform_rejected_count,
            "recover_candidate_count": recover_candidate_count,
            "recover_accepted_count": recover_accepted_count,
            "recover_rejected_count": recover_rejected_count,
            "safe_corridor_recovery_count": safe_corridor_recovery_count,
            "direct_rejoin_attempt_count": direct_rejoin_attempt_count,
            "direct_rejoin_reject_count": direct_rejoin_reject_count,
            "detour_rejoin_attempt_count": detour_rejoin_attempt_count,
            "detour_rejoin_accept_count": detour_rejoin_accept_count,
            "delayed_rejoin_count": delayed_rejoin_count,
            "recover_path_unsafe_count": recover_path_unsafe_count,
            "recovery_path_failure_streak_max": recovery_path_failure_streak_max,
            "repeated_unsafe_target_count": repeated_unsafe_target_count,
            "post_recovery_act_window_count": post_recovery_act_window_count,
            "post_recovery_act_window_interrupted_count": post_recovery_act_window_interrupted_count,
            "direct_rejoin_attempted_steps": direct_rejoin_attempted_steps,
            "direct_rejoin_rejected_steps": direct_rejoin_rejected_steps,
            "detour_rejoin_attempted_steps": detour_rejoin_attempted_steps,
            "detour_rejoin_accepted_steps": detour_rejoin_accepted_steps,
            "delayed_rejoin_active_steps": delayed_rejoin_active_steps,
            "repeated_unsafe_target_steps": repeated_unsafe_target_steps,
            "post_recovery_act_window_steps": post_recovery_act_window_steps,
            "post_recovery_act_window_interrupted_steps": post_recovery_act_window_interrupted_steps,
            "recover_reject_reason_counts": {
                reason: recover_reject_reasons.count(reason)
                for reason in sorted(set(recover_reject_reasons))
            },
            "safe_prefix_accepted_count": safe_prefix_accepted_count,
            "first_action_only_accepted_count": first_action_only_accepted_count,
            "immediate_hard_reject_count": immediate_hard_reject_count,
            "no_safe_prefix_reject_count": no_safe_prefix_reject_count,
            "horizon_margin_reject_count": horizon_margin_reject_count,
            "accepted_deform_steps": accepted_deform_steps,
            "accepted_recover_steps": accepted_recover_steps,
            "fallback_brake_after_reject_count": fallback_brake_after_reject_count,
            "nominal_rejoin_available_count": nominal_rejoin_available_count,
            "nominal_rejoin_suppressed_count": nominal_rejoin_suppressed_count,
            "stale_nominal_rejoin_suppressed_count": stale_nominal_rejoin_suppressed_count,
            "nominal_prefix_unsafe_suppressed_count": nominal_prefix_unsafe_suppressed_count,
            "recover_positive_projection_count": recover_positive_projection_count,
            "recover_nonpositive_projection_count": recover_nonpositive_projection_count,
            "mean_recover_projection_on_nominal": (
                float(np.mean(recover_projection_on_nominal))
                if recover_projection_on_nominal.size
                else (
                    float(np.mean(finite_metric("mean_recover_projection_on_nominal")))
                    if finite_metric("mean_recover_projection_on_nominal").size
                    else None
                )
            ),
            "mean_recover_cosine_to_nominal": (
                float(np.mean(recover_cosine_to_nominal))
                if recover_cosine_to_nominal.size
                else (
                    float(np.mean(finite_metric("mean_recover_cosine_to_nominal")))
                    if finite_metric("mean_recover_cosine_to_nominal").size
                    else None
                )
            ),
            "mean_recover_task_progress_score": (
                float(np.mean(recover_task_progress_score))
                if recover_task_progress_score.size
                else (
                    float(np.mean(finite_metric("mean_recover_task_progress_score")))
                    if finite_metric("mean_recover_task_progress_score").size
                    else None
                )
            ),
            "hold_unsafe_count": hold_unsafe_count,
            "hold_predicted_contact_count": hold_predicted_contact_count,
            "emergency_deform_away_steps": emergency_deform_away_steps,
            "emergency_deform_away_count": emergency_deform_away_count,
            "contact_during_hold_count": contact_during_hold_count,
            "contact_during_brake_count": contact_during_brake_count,
            "contact_during_deform_count": contact_during_deform_count,
            "contact_during_recover_count": contact_during_recover_count,
            "mean_hold_horizon_min_clearance": (
                float(np.mean(hold_horizon_min_clearance))
                if hold_horizon_min_clearance.size
                else None
            ),
            "min_hold_horizon_min_clearance": (
                float(np.min(hold_horizon_min_clearance))
                if hold_horizon_min_clearance.size
                else None
            ),
            "act_ratio": act_ratio,
            "safety_mode_ratio": safety_mode_ratio,
            "fallback_ratio": fallback_ratio,
            "mean_task_progress": float(np.mean(task_progress)) if task_progress.size else None,
            "max_task_progress": max_task_progress,
            "final_task_progress": final_task_progress,
            "mean_task_progress_delta": float(np.mean(task_progress_delta)) if task_progress_delta.size else None,
            "mean_progress_during_act": mean_progress_for("act_step"),
            "mean_progress_during_deform": mean_progress_for("deform_step"),
            "mean_progress_during_recover": mean_progress_for("recover_step"),
            "mean_progress_during_brake": mean_progress_for("brake_step"),
            "mean_progress_during_fallback": mean_progress_for("fallback_step"),
            "num_progress_regressions": int(np.sum(task_progress_delta < -1e-6)) if task_progress_delta.size else 0,
            "num_large_arm_delta_events": int(
                np.sum([m.per_step_arm_delta_norm > large_arm_delta_threshold for m in metrics])
            ),
            "num_large_base_delta_events": int(
                np.sum([m.per_step_base_delta_norm > large_base_delta_threshold for m in metrics])
            ),
            "large_arm_delta_threshold": large_arm_delta_threshold,
            "large_base_delta_threshold": large_base_delta_threshold,
            "low_act_ratio_threshold": low_act_ratio_threshold,
            "high_fallback_ratio_threshold": high_fallback_ratio_threshold,
            "task_success_threshold": success_threshold,
            "likely_failure_cause": likely_failure_cause,
            "pass_through_rate": rate(modes, "pass_through"),
            "path_consistent_brake_rate": rate(modes, "path_consistent_brake"),
            "horizon_deform_rate": rate(modes, "horizon_deform"),
            "sequential_oscbf_rate": rate(modes, "sequential_oscbf"),
            "pause_on_unsafe_rate": rate(modes, "pause_on_unsafe"),
            "pause_and_restart_rate": rate(modes, "pause_and_restart"),
            "phase_reanchor_rate": rate(modes, "phase_reanchor"),
            "phase_reanchor_source_rate": rate(sources, "phase_reanchor"),
            "phase_reanchor_steps": int(np.sum(phase_reanchor_steps)),
            "gripper_latched_steps": gripper_latched_steps,
            "post_recovery_task_guard_steps": post_recovery_task_guard_steps,
            "post_recovery_reanchor_started_count": post_recovery_reanchor_started_count,
            "post_recovery_progress_regression_count": post_recovery_progress_regression_count,
            "pause_current_clearance_rate": rate(pause_reasons, "current_clearance"),
            "pause_horizon_clearance_rate": rate(pause_reasons, "horizon_clearance"),
            "pause_deform_clearance_rate": rate(pause_reasons, "deform_clearance"),
            "chunk_deform_source_rate": rate(sources, "chunk_deform"),
            "sequential_oscbf_source_rate": rate(sources, "sequential_oscbf"),
            "min_robot_human_distance": float(np.min(robot_human_distances)) if robot_human_distances else None,
            "num_safety_violations": int(np.sum(safety_violations)),
            "num_filter_activations": int(np.sum(chunk_interventions)),
            "total_brake_steps": int(np.sum(brake_steps)),
            "total_deformation_steps": int(np.sum(deformation_steps)),
            "task_success": bool(any(m.success for m in metrics)),
            "drawer_open_distance": float(drawer_open_distances[-1]) if drawer_open_distances else None,
            "resume_latency_after_human_exit": _resume_latency_after_human_exit(metrics),
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

    def min_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return float(np.min(vals)) if vals else None

    def sum_of(key):
        vals = [s[key] for s in episode_summaries if s.get(key) is not None]
        return int(np.sum(vals)) if vals else None

    summary.update(
        {
            "mean_chunk_arm_delta": mean_of("mean_chunk_arm_delta"),
            "max_chunk_arm_delta_over_episodes": max_of("max_chunk_arm_delta"),
            "mean_chunk_intervention_frequency": mean_of("chunk_intervention_frequency"),
            "mean_chunk_modified_fraction": mean_of("mean_chunk_modified_fraction"),
            "mean_chunk_modified_steps": mean_of("mean_chunk_modified_steps"),
            "mean_chunk_first_modified_step": mean_of("mean_chunk_first_modified_step"),
            "mean_chunk_mean_step_arm_delta": mean_of("mean_chunk_mean_step_arm_delta"),
            "max_chunk_step_arm_delta_over_episodes": max_of("max_chunk_step_arm_delta"),
            "mean_chunk_future_edit_fraction": mean_of("mean_chunk_future_edit_fraction"),
            "mean_chunk_first_edit_fraction": mean_of("mean_chunk_first_edit_fraction"),
            "mean_chunk_safe_arm_variation": mean_of("mean_chunk_safe_arm_variation"),
            "mean_chunk_nominal_arm_variation": mean_of("mean_chunk_nominal_arm_variation"),
            "mean_chunk_arm_variation_delta": mean_of("mean_chunk_arm_variation_delta"),
            "mean_chunk_edit_variation": mean_of("mean_chunk_edit_variation"),
            "mean_path_deviation": mean_of("mean_path_deviation"),
            "max_path_deviation_over_episodes": max_of("max_path_deviation"),
            "mean_final_path_deviation": mean_of("mean_final_path_deviation"),
            "mean_chunk_preemptive_intervention_frequency": mean_of("chunk_preemptive_intervention_frequency"),
            "mean_horizon_risk_gap": mean_of("mean_horizon_risk_gap"),
            "max_horizon_risk_gap_over_episodes": max_of("max_horizon_risk_gap"),
            "mean_horizon_clearance_drop": mean_of("mean_horizon_clearance_drop"),
            "mean_horizon_risk_gap_rate": mean_of("horizon_risk_gap_rate"),
            "mean_horizon_only_risk_rate": mean_of("horizon_only_risk_rate"),
            "mean_deformation_norm": mean_of("mean_deformation_norm"),
            "mean_deform_safe_rate": mean_of("deform_safe_rate"),
            "optimized_attempts": sum_of("optimized_attempts"),
            "optimized_safe_count": sum_of("optimized_safe_count"),
            "optimized_recoverable_count": sum_of("optimized_recoverable_count"),
            "optimized_accepted_count": sum_of("optimized_accepted_count"),
            "rejected_unsafe_count": sum_of("rejected_unsafe_count"),
            "rejected_unrecoverable_count": sum_of("rejected_unrecoverable_count"),
            "rejected_both_count": sum_of("rejected_both_count"),
            "fallback_used_count": sum_of("fallback_used_count"),
            "deform_stage_accepted_count": sum_of("deform_stage_accepted_count"),
            "recover_accepted_count": sum_of("recover_accepted_count"),
            "cached_motion_active_count": sum_of("cached_motion_active_count"),
            "resumed_from_cached_count": sum_of("resumed_from_cached_count"),
            "mean_deform_stage_min_clearance": mean_of("mean_deform_stage_min_clearance"),
            "mean_recover_min_clearance": mean_of("mean_recover_min_clearance"),
            "mean_recover_rejoin_loss": mean_of("mean_recover_rejoin_loss"),
            "mean_recover_path_min_clearance": mean_of("mean_recover_path_min_clearance"),
            "min_recover_path_min_clearance": min_of("min_recover_path_min_clearance"),
            "mean_recover_immediate_clearance": mean_of("mean_recover_immediate_clearance"),
            "mean_recover_prefix_min_clearance": mean_of("mean_recover_prefix_min_clearance"),
            "total_optimized_attempt_count": sum_of("optimized_attempt_count"),
            "mean_optimized_accept_rate": mean_of("optimized_accept_rate"),
            "mean_recoverable_rate": mean_of("recoverable_rate"),
            "mean_fallback_used_rate": mean_of("fallback_used_rate"),
            "mean_q_rejoin_dist": mean_of("mean_q_rejoin_dist"),
            "max_q_rejoin_dist_over_episodes": max_of("max_q_rejoin_dist"),
            "mean_qd_rejoin_dist": mean_of("mean_qd_rejoin_dist"),
            "max_qd_rejoin_dist_over_episodes": max_of("max_qd_rejoin_dist"),
            "mean_ee_rejoin_dist": mean_of("mean_ee_rejoin_dist"),
            "max_ee_rejoin_dist_over_episodes": max_of("max_ee_rejoin_dist"),
            "mean_rejoin_q_eval_time_ms": mean_of("mean_rejoin_q_eval_time_ms"),
            "mean_rejoin_qd_eval_time_ms": mean_of("mean_rejoin_qd_eval_time_ms"),
            "mean_ee_nom_cache_time_ms": mean_of("mean_ee_nom_cache_time_ms"),
            "mean_ee_final_check_time_ms": mean_of("mean_ee_final_check_time_ms"),
            "act_steps": sum_of("act_steps"),
            "deform_steps": sum_of("deform_steps"),
            "recover_steps": sum_of("recover_steps"),
            "brake_steps": sum_of("brake_steps"),
            "fallback_steps": sum_of("fallback_steps"),
            "optimized_attempt_steps": sum_of("optimized_attempt_steps"),
            "optimized_accepted_steps": sum_of("optimized_accepted_steps"),
            "committed_chunk_started_count": sum_of("committed_chunk_started_count"),
            "committed_chunk_completed_count": sum_of("committed_chunk_completed_count"),
            "committed_chunk_abort_count": sum_of("committed_chunk_abort_count"),
            "committed_repaired_step_count": sum_of("committed_repaired_step_count"),
            "committed_abort_due_to_human_motion_count": sum_of("committed_abort_due_to_human_motion_count"),
            "committed_abort_due_to_prediction_error_count": sum_of("committed_abort_due_to_prediction_error_count"),
            "committed_abort_due_to_safety_semantics_mismatch_count": sum_of("committed_abort_due_to_safety_semantics_mismatch_count"),
            "committed_state_mismatch_abort_count": sum_of("committed_state_mismatch_abort_count"),
            "mean_planning_vs_replay_clearance_post_error": mean_of("mean_planning_vs_replay_clearance_post_error"),
            "mean_planning_vs_replay_human_error": mean_of("mean_planning_vs_replay_human_error"),
            "mean_actual_vs_planned_post_q_error": mean_of("mean_actual_vs_planned_post_q_error"),
            "mean_committed_state_error": mean_of("mean_committed_state_error"),
            "max_committed_state_error": max_of("max_committed_state_error"),
            "mean_committed_clearance_prediction_error": mean_of("mean_committed_clearance_prediction_error"),
            "mean_committed_planned_vs_actual_q_error": mean_of("mean_committed_planned_vs_actual_q_error"),
            "mean_committed_human_motion_since_plan": mean_of("mean_committed_human_motion_since_plan"),
            "mean_committed_accepted_clearance_margin": mean_of("mean_committed_accepted_clearance_margin"),
            "committed_deform_steps_executed": sum_of("committed_deform_steps_executed"),
            "committed_recover_steps_executed": sum_of("committed_recover_steps_executed"),
            "resume_from_committed_rejoin_count": sum_of("resume_from_committed_rejoin_count"),
            "recovery_action_history_reset_count": sum_of("recovery_action_history_reset_count"),
            "accepted_recover_chunks_not_executed": any(
                bool(s.get("accepted_recover_chunks_not_executed"))
                for s in episode_summaries
            ),
            "diagnostic_warning": (
                "accepted_recover_chunks_not_executed"
                if any(bool(s.get("accepted_recover_chunks_not_executed")) for s in episode_summaries)
                else None
            ),
            "temporary_wait_steps": sum_of("temporary_wait_steps"),
            "resume_after_wait_count": sum_of("resume_after_wait_count"),
            "deform_after_persistent_block_count": sum_of("deform_after_persistent_block_count"),
            "deform_suppressed_by_temporary_wait_count": sum_of("deform_suppressed_by_temporary_wait_count"),
            "recovery_failure_streak_max": max_of("recovery_failure_streak_max"),
            "deform_replan_count": max_of("deform_replan_count"),
            "recover_replan_count": max_of("recover_replan_count"),
            "recovery_replan_count": max_of("recovery_replan_count"),
            "stale_recovery_suppressed_count": max_of("stale_recovery_suppressed_count"),
            "recovery_target_infeasible_count": max_of("recovery_target_infeasible_count"),
            "emergency_brake_steps": max_of("emergency_brake_steps"),
            "optimized_candidate_count": max_of("optimized_candidate_count"),
            "optimized_solution_count": max_of("optimized_solution_count"),
            "fallback_candidate_count": max_of("fallback_candidate_count"),
            "fallback_candidate_accepted_count": max_of("fallback_candidate_accepted_count"),
            "optimized_rejected_count": max_of("optimized_rejected_count"),
            "deform_candidate_count": max_of("deform_candidate_count"),
            "deform_accepted_count": max_of("deform_accepted_count"),
            "deform_rejected_count": max_of("deform_rejected_count"),
            "recover_candidate_count": max_of("recover_candidate_count"),
            "recover_accepted_count": max_of("recover_accepted_count"),
            "recover_rejected_count": max_of("recover_rejected_count"),
            "safe_corridor_recovery_count": max_of("safe_corridor_recovery_count"),
            "direct_rejoin_attempt_count": max_of("direct_rejoin_attempt_count"),
            "direct_rejoin_reject_count": max_of("direct_rejoin_reject_count"),
            "detour_rejoin_attempt_count": max_of("detour_rejoin_attempt_count"),
            "detour_rejoin_accept_count": max_of("detour_rejoin_accept_count"),
            "delayed_rejoin_count": max_of("delayed_rejoin_count"),
            "recover_path_unsafe_count": max_of("recover_path_unsafe_count"),
            "recovery_path_failure_streak_max": max_of("recovery_path_failure_streak_max"),
            "repeated_unsafe_target_count": max_of("repeated_unsafe_target_count"),
            "post_recovery_act_window_count": max_of("post_recovery_act_window_count"),
            "post_recovery_act_window_interrupted_count": max_of("post_recovery_act_window_interrupted_count"),
            "direct_rejoin_attempted_steps": sum_of("direct_rejoin_attempted_steps"),
            "direct_rejoin_rejected_steps": sum_of("direct_rejoin_rejected_steps"),
            "detour_rejoin_attempted_steps": sum_of("detour_rejoin_attempted_steps"),
            "detour_rejoin_accepted_steps": sum_of("detour_rejoin_accepted_steps"),
            "delayed_rejoin_active_steps": sum_of("delayed_rejoin_active_steps"),
            "repeated_unsafe_target_steps": sum_of("repeated_unsafe_target_steps"),
            "post_recovery_act_window_steps": sum_of("post_recovery_act_window_steps"),
            "post_recovery_act_window_interrupted_steps": sum_of("post_recovery_act_window_interrupted_steps"),
            "safe_prefix_accepted_count": max_of("safe_prefix_accepted_count"),
            "first_action_only_accepted_count": max_of("first_action_only_accepted_count"),
            "immediate_hard_reject_count": max_of("immediate_hard_reject_count"),
            "no_safe_prefix_reject_count": max_of("no_safe_prefix_reject_count"),
            "horizon_margin_reject_count": max_of("horizon_margin_reject_count"),
            "accepted_deform_steps": max_of("accepted_deform_steps"),
            "accepted_recover_steps": max_of("accepted_recover_steps"),
            "fallback_brake_after_reject_count": max_of("fallback_brake_after_reject_count"),
            "nominal_rejoin_available_count": max_of("nominal_rejoin_available_count"),
            "nominal_rejoin_suppressed_count": max_of("nominal_rejoin_suppressed_count"),
            "stale_nominal_rejoin_suppressed_count": max_of("stale_nominal_rejoin_suppressed_count"),
            "nominal_prefix_unsafe_suppressed_count": max_of("nominal_prefix_unsafe_suppressed_count"),
            "recover_positive_projection_count": max_of("recover_positive_projection_count"),
            "recover_nonpositive_projection_count": max_of("recover_nonpositive_projection_count"),
            "mean_recover_projection_on_nominal": mean_of("mean_recover_projection_on_nominal"),
            "mean_recover_cosine_to_nominal": mean_of("mean_recover_cosine_to_nominal"),
            "mean_recover_task_progress_score": mean_of("mean_recover_task_progress_score"),
            "hold_unsafe_count": max_of("hold_unsafe_count"),
            "hold_predicted_contact_count": max_of("hold_predicted_contact_count"),
            "emergency_deform_away_steps": max_of("emergency_deform_away_steps"),
            "emergency_deform_away_count": max_of("emergency_deform_away_count"),
            "contact_during_hold_count": max_of("contact_during_hold_count"),
            "contact_during_brake_count": max_of("contact_during_brake_count"),
            "contact_during_deform_count": max_of("contact_during_deform_count"),
            "contact_during_recover_count": max_of("contact_during_recover_count"),
            "mean_hold_horizon_min_clearance": mean_of("mean_hold_horizon_min_clearance"),
            "min_hold_horizon_min_clearance": min(
                [s.get("min_hold_horizon_min_clearance") for s in episode_summaries if s.get("min_hold_horizon_min_clearance") is not None],
                default=None,
            ),
            "mean_act_ratio": mean_of("act_ratio"),
            "mean_safety_mode_ratio": mean_of("safety_mode_ratio"),
            "mean_fallback_ratio": mean_of("fallback_ratio"),
            "mean_task_progress": mean_of("mean_task_progress"),
            "max_task_progress": max_of("max_task_progress"),
            "final_task_progress": mean_of("final_task_progress"),
            "mean_task_progress_delta": mean_of("mean_task_progress_delta"),
            "mean_progress_during_act": mean_of("mean_progress_during_act"),
            "mean_progress_during_deform": mean_of("mean_progress_during_deform"),
            "mean_progress_during_recover": mean_of("mean_progress_during_recover"),
            "mean_progress_during_brake": mean_of("mean_progress_during_brake"),
            "mean_progress_during_fallback": mean_of("mean_progress_during_fallback"),
            "num_progress_regressions": sum_of("num_progress_regressions"),
            "num_large_arm_delta_events": sum_of("num_large_arm_delta_events"),
            "num_large_base_delta_events": sum_of("num_large_base_delta_events"),
            "likely_failure_cause": (
                episode_summaries[-1].get("likely_failure_cause")
                if len(episode_summaries) == 1
                else None
            ),
            "likely_failure_cause_counts": {
                cause: sum(1 for s in episode_summaries if s.get("likely_failure_cause") == cause)
                for cause in sorted({s.get("likely_failure_cause") for s in episode_summaries if s.get("likely_failure_cause") is not None})
            },
            "mean_pass_through_rate": mean_of("pass_through_rate"),
            "mean_path_consistent_brake_rate": mean_of("path_consistent_brake_rate"),
            "mean_horizon_deform_rate": mean_of("horizon_deform_rate"),
            "mean_sequential_oscbf_rate": mean_of("sequential_oscbf_rate"),
            "mean_pause_on_unsafe_rate": mean_of("pause_on_unsafe_rate"),
            "mean_pause_and_restart_rate": mean_of("pause_and_restart_rate"),
            "mean_phase_reanchor_rate": mean_of("phase_reanchor_rate"),
            "mean_phase_reanchor_source_rate": mean_of("phase_reanchor_source_rate"),
            "total_phase_reanchor_steps": sum_of("phase_reanchor_steps"),
            "total_gripper_latched_steps": sum_of("gripper_latched_steps"),
            "total_post_recovery_task_guard_steps": sum_of("post_recovery_task_guard_steps"),
            "total_post_recovery_reanchor_started_count": sum_of("post_recovery_reanchor_started_count"),
            "total_post_recovery_progress_regression_count": sum_of("post_recovery_progress_regression_count"),
            "mean_chunk_deform_source_rate": mean_of("chunk_deform_source_rate"),
            "mean_sequential_oscbf_source_rate": mean_of("sequential_oscbf_source_rate"),
            "min_robot_human_distance": min_of("min_robot_human_distance"),
            "mean_min_robot_human_distance": mean_of("min_robot_human_distance"),
            "total_safety_violations": sum_of("num_safety_violations"),
            "total_filter_activations": sum_of("num_filter_activations"),
            "total_brake_steps": sum_of("total_brake_steps"),
            "total_deformation_steps": sum_of("total_deformation_steps"),
            "task_success_rate": mean_of("task_success"),
            "mean_drawer_open_distance": mean_of("drawer_open_distance"),
            "mean_resume_latency_after_human_exit": mean_of("resume_latency_after_human_exit"),
        }
    )
    return summary


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    cfg = make_cfg(args)
    runtime_cfg = cfg
    workspace_cfg = cfg
    direct_human_runtime = False
    snapshot_cfg = None
    try:
        snapshot_cfg = _load_snapshot_normalization_cfg(snapshot_path)
    except FileNotFoundError:
        snapshot_cfg = None
    if snapshot_cfg is not None:
        runtime_task = str(cfg.env.task_name)
        snapshot_task = str(snapshot_cfg.env.task_name)
        if (
            args.policy_env is None
            and args.safety_env is None
            and runtime_task != snapshot_task
            and runtime_task.startswith("human_arm_")
        ):
            workspace_cfg = _make_policy_env_cfg(cfg, f"bigym/{snapshot_task}")
            for key in ("manifest", "privileged_information", "require_mode_label"):
                if key in snapshot_cfg.env:
                    workspace_cfg.env[key] = snapshot_cfg.env[key]
                elif key == "manifest" and key in workspace_cfg.env:
                    workspace_cfg.env[key] = None
            direct_human_runtime = True
            if not args.hide_human_arm_policy_obs:
                args.hide_human_arm_policy_obs = True
                print("direct_human_runtime: hiding human arm from policy observations")
            print("direct_human_runtime: using checkpoint task for policy workspace")
            print("policy_workspace_task:", workspace_cfg.env.task_name)
            print("runtime_task:", runtime_cfg.env.task_name)

    normalization_source, normalization_cfg = _resolve_normalization_cfg(
        args,
        workspace_cfg if direct_human_runtime else cfg,
        snapshot_path,
    )
    if direct_human_runtime and normalization_cfg is None:
        normalization_cfg = workspace_cfg
        normalization_source = f"{normalization_source}+policy_workspace"

    print("\n=== Normalization stats source ===")
    _print_normalization_source(
        normalization_source,
        normalization_cfg if normalization_cfg is not None else cfg,
    )

    print("\n=== Eval control config ===")
    print("action_sequence:", workspace_cfg.get("action_sequence", None))
    print("execution_length:", workspace_cfg.get("execution_length", None))
    print("temporal_ensemble:", workspace_cfg.get("temporal_ensemble", None))

    print("\n=== Creating Workspace and loading ACT snapshot ===")
    ws = make_workspace_and_load_snapshot(workspace_cfg, snapshot_path)
    policy_observation_space = getattr(ws.eval_env, "observation_space", None)

    print("\n=== Creating evaluation env ===")
    env = _make_eval_env_with_normalization(runtime_cfg, normalization_cfg)
    robot_spawn_info = _apply_robot_spawn_offset_xy(env, args.robot_spawn_offset_xy)
    if robot_spawn_info is not None:
        print("robot_spawn:", robot_spawn_info)
    if args.freeze_human_arm:
        print("freeze_human_arm: will freeze after each reset")
    if (
        args.human_arm_aggression != 1.0
        or args.human_arm_substeps != 1
        or args.human_arm_zero_dwell
        or args.human_arm_walk_radius is not None
        or args.human_arm_goal_xy is not None
        or args.human_arm_keepout_min_clear is not None
        or args.human_arm_disable_keepout
        or args.human_arm_force_carrier_xy is not None
        or args.human_arm_force_carrier_amp_xy is not None
        or args.human_arm_drawer_obstruction
        or args.human_arm_natural_contact_motion
        or args.human_arm_final_clear_after_steps >= 0
    ):
        print(
            "human_arm_challenge:",
            f"aggression={args.human_arm_aggression}",
            f"substeps={args.human_arm_substeps}",
            f"zero_dwell={args.human_arm_zero_dwell}",
            f"walk_radius={args.human_arm_walk_radius}",
            f"goal_xy={args.human_arm_goal_xy}",
            f"keepout_min_clear={args.human_arm_keepout_min_clear}",
            f"disable_keepout={args.human_arm_disable_keepout}",
            f"force_carrier_xy={args.human_arm_force_carrier_xy}",
            f"force_carrier_amp_xy={args.human_arm_force_carrier_amp_xy}",
            f"force_carrier_frequency={args.human_arm_force_carrier_frequency}",
            f"drawer_obstruction={args.human_arm_drawer_obstruction}",
            f"drawer_obstruction_xy={args.human_arm_drawer_obstruction_xy}",
            f"drawer_obstruction_amp_xy={args.human_arm_drawer_obstruction_amp_xy}",
            f"yaw_offset_deg={args.human_arm_yaw_offset_deg}",
            f"natural_contact_motion={args.human_arm_natural_contact_motion}",
            f"natural_motion_frequency={args.human_arm_natural_motion_frequency}",
            f"natural_lateral_scale={args.human_arm_natural_lateral_scale}",
            f"natural_return_curl_scale={args.human_arm_natural_return_curl_scale}",
            f"final_clear_after_steps={args.human_arm_final_clear_after_steps}",
            f"final_clear_duration_steps={args.human_arm_final_clear_duration_steps}",
            f"final_clear_trigger={args.human_arm_final_clear_trigger}",
            f"final_clear_max_carrier_speed={args.human_arm_final_clear_max_carrier_speed}",
            f"final_clear_max_joint_speed={args.human_arm_final_clear_max_joint_speed}",
            f"final_clear_carrier_xy={args.human_arm_final_clear_carrier_xy}",
        )
    if args.disable_human_arm_collisions or args.visual_only_human_arm:
        disabled = _disable_human_arm_collisions(env)
        label = (
            "visual_only_human_arm"
            if args.visual_only_human_arm
            else "disable_human_arm_collisions"
        )
        print(f"{label}: disabled_physical_contact_geoms={disabled}")
        if args.visual_only_human_arm:
            print(
                "visual_only_human_arm: actual contact response is disabled; "
                "use h_violation/min_robot_human_distance as the collision check."
            )
    elif args.enable_human_arm_collisions:
        enabled = _enable_human_arm_collisions(env)
        print(f"enable_human_arm_collisions: enabled_physical_contact_geoms={enabled}")
    env_action_shape = infer_env_action_shape(env, fallback=(16, 16))
    print("env_action_shape:", env_action_shape)

    if args.hide_human_arm_policy_obs and args.policy_env is not None:
        raise ValueError(
            "Use either --hide-human-arm-policy-obs or --policy-env, not both."
        )

    policy_env = None
    policy_env_action_shape = None
    policy_robot_spawn_info = None
    if args.policy_env is not None:
        print("\n=== Creating clean policy observation env ===")
        policy_cfg = _make_policy_env_cfg(cfg, args.policy_env)
        policy_env = _make_eval_env_with_normalization(
            policy_cfg,
            normalization_cfg,
        )
        policy_env_action_shape = infer_env_action_shape(
            policy_env,
            fallback=env_action_shape,
        )
        if policy_env_action_shape != env_action_shape:
            raise ValueError(
                "Policy env action shape does not match eval env action shape: "
                f"policy={policy_env_action_shape}, eval={env_action_shape}."
            )
        policy_robot_spawn_info = _apply_robot_spawn_offset_xy(
            policy_env, args.robot_spawn_offset_xy
        )
        print("policy_env:", args.policy_env)
        print("policy_env_action_shape:", policy_env_action_shape)
        if policy_robot_spawn_info is not None:
            print("policy_robot_spawn:", policy_robot_spawn_info)

    safety_env = None
    safety_robot_spawn_info = None
    if args.safety_env is not None:
        print("\n=== Creating mirrored safety env ===")
        safety_cfg = _make_policy_env_cfg(runtime_cfg, args.safety_env)
        safety_env = _make_eval_env_with_normalization(
            safety_cfg,
            normalization_cfg if normalization_cfg is not None else cfg,
        )
        safety_env_action_shape = infer_env_action_shape(
            safety_env,
            fallback=env_action_shape,
        )
        if safety_env_action_shape != env_action_shape:
            raise ValueError(
                "Safety env action shape does not match task env action shape: "
                f"safety={safety_env_action_shape}, task={env_action_shape}."
            )
        if args.disable_human_arm_collisions or args.visual_only_human_arm:
            disabled = _disable_human_arm_collisions(safety_env)
            label = (
                "visual_only_human_arm"
                if args.visual_only_human_arm
                else "disable_human_arm_collisions"
            )
            print(f"safety_env {label}: disabled_physical_contact_geoms={disabled}")
        elif args.enable_human_arm_collisions:
            enabled = _enable_human_arm_collisions(safety_env)
            print(
                "safety_env enable_human_arm_collisions: "
                f"enabled_physical_contact_geoms={enabled}"
            )
        safety_robot_spawn_info = _apply_robot_spawn_offset_xy(
            safety_env, args.robot_spawn_offset_xy
        )
        print("safety_env:", args.safety_env)
        print("safety_env_action_shape:", safety_env_action_shape)
        if safety_robot_spawn_info is not None:
            print("safety_robot_spawn:", safety_robot_spawn_info)

    output_root, step_jsonl_path, episode_summary_path, final_summary_path = make_output_paths(args)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.video_dir is not None:
        video_dir = Path(args.video_dir)
    else:
        video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_recorder = WallClockVideoRecorder(
        video_dir if args.record_video else None,
        fps=20,
        time_base=args.video_time_base,
    )
    video_stop_steps = _resolve_video_stop_steps(args, video_recorder)

    trajectory_logging_enabled = bool(
        args.log_chunk_trajectories
        and args.condition in {"sequential", "sequential_oscbf", "chunk_deform"}
    )
    chunk_trajectory_jsonl_path = output_root / "chunk_trajectory_traces.jsonl"
    human_arm_trajectory_jsonl_path = output_root / "human_arm_trajectory.jsonl"
    trajectory_plot_dir = output_root / "trajectory_plots"
    if trajectory_logging_enabled and args.plot_chunk_trajectories_3d:
        trajectory_plot_dir.mkdir(parents=True, exist_ok=True)

    print("record_video:", args.record_video)
    print("video_dir:", video_dir)
    print("stop_video_at_seconds:", args.stop_video_at_seconds)
    print("video_time_base:", args.video_time_base)
    print("stop_video_at_steps:", video_stop_steps)
    print("log_chunk_trajectories:", trajectory_logging_enabled)
    if trajectory_logging_enabled:
        print("chunk_trajectory_jsonl:", chunk_trajectory_jsonl_path)
        print("human_arm_trajectory_jsonl:", human_arm_trajectory_jsonl_path)
        print("plot_chunk_trajectories_3d:", args.plot_chunk_trajectories_3d)
        print("trajectory_plot_dir:", trajectory_plot_dir)

    replay_actions = None
    if args.replay_actions is not None:
        replay_path = Path(args.replay_actions)
        if not replay_path.is_file():
            raise FileNotFoundError(f"Replay actions not found: {replay_path}")
        replay_npz = np.load(replay_path)
        replay_actions = np.asarray(replay_npz["actions"], dtype=np.float32)
        if replay_actions.ndim == len(env_action_shape) + 1:
            replay_actions = replay_actions[None]
        if replay_actions.ndim != len(env_action_shape) + 2:
            raise ValueError(
                f"Expected replay actions with shape (episodes, steps, {env_action_shape}), "
                f"got {replay_actions.shape}."
            )
        if tuple(replay_actions.shape[2:]) != env_action_shape:
            raise ValueError(
                f"Replay action chunk shape {replay_actions.shape[2:]} does not match "
                f"env_action_shape {env_action_shape}."
            )
        print("replay_actions:", replay_path)
        print("replay_actions_shape:", replay_actions.shape)

    print("\n=== Creating OSCBF monitor/filter ===")
    oscbf = make_oscbf_filter(args)
    horizon_operator = HorizonOSCBFOperator(
        oscbf,
        min_clearance=args.chunk_min_clearance,
        dt=0.05,
        predict_human_motion=args.chunk_horizon_predict_human_motion,
        human_prediction_max_time=args.chunk_human_motion_prediction_max_time,
        human_prediction_max_speed=args.chunk_human_motion_prediction_max_speed,
    )
    safechunk = make_safechunk_filter(args, horizon_operator, oscbf=oscbf)
    print("condition:", args.condition)
    print("arm indices:", oscbf.bigym_action_arm_indices.tolist())
    if args.condition in {"sequential", "sequential_oscbf", "chunk_deform"}:
        print("chunk controlled indices:", safechunk.controlled_action_indices.tolist())
        print("chunk_deform_mode:", safechunk.mode)
        print("chunk_deformation_enabled:", safechunk.deformation_enabled)
        print("chunk_deformation_scales:", safechunk.chunk_deformation_scales)
        print("recoverable_deform_enabled:", safechunk.recoverable_deform_enabled)
        print("recoverable_inner_rejoin_metric:", safechunk.inner_rejoin_metric)
        print("recoverable_final_rejoin_metric:", safechunk.final_rejoin_metric)
        print("recoverable_q_rejoin_threshold:", safechunk.q_rejoin_threshold)
        print("recoverable_ee_rejoin_threshold:", safechunk.ee_rejoin_threshold)
        print("recoverable_cache_nominal_ee:", safechunk.cache_nominal_ee)
        print("recoverable_ee_rejoin_in_inner_loop:", safechunk.ee_rejoin_in_inner_loop)
        print("recoverable_explicit_recovery:", safechunk.explicit_return)
        print("explicit_recovery_commit_accepted_chunks:", safechunk.commit_accepted_chunks)
        print("explicit_recovery_committed_chunk_safety_check:", safechunk.committed_chunk_safety_check)
        print("explicit_recovery_committed_min_clearance_for_abort:", safechunk.committed_min_clearance_for_abort)
        print("explicit_recovery_repair_committed_action:", safechunk.repair_committed_action)
        print("explicit_recovery_monotonic_committed_repair:", safechunk.monotonic_committed_repair)
        print("explicit_recovery_committed_execution_margin:", safechunk.committed_execution_margin)
        print("explicit_recovery_committed_state_error_threshold:", safechunk.committed_state_error_threshold)
        print("explicit_recovery_committed_state_error_action:", safechunk.committed_state_error_action)
        print("recoverable_deform_horizon:", safechunk.yield_horizon)
        print("recoverable_recover_horizon:", safechunk.return_horizon)
        print("recoverable_use_ee_final_check:", safechunk.use_ee_final_check)
        print("optimized_debug_safety_feasibility:", safechunk.debug_safety_feasibility)
        print("chunk_horizon_predict_human_motion:", horizon_operator.predict_human_motion)
        print("chunk_human_motion_prediction_max_time:", horizon_operator.human_prediction_max_time)
        print("chunk_human_motion_prediction_max_speed:", horizon_operator.human_prediction_max_speed)
        print("diagnostics_enabled:", args.diagnostics_enabled)
        print("diagnostics_large_arm_delta_threshold:", args.diagnostics_large_arm_delta_threshold)
        print("diagnostics_large_base_delta_threshold:", args.diagnostics_large_base_delta_threshold)
        print("diagnostics_low_act_ratio_threshold:", args.diagnostics_low_act_ratio_threshold)
        print("diagnostics_high_fallback_ratio_threshold:", args.diagnostics_high_fallback_ratio_threshold)
        print("recoverable_brake_if_unrecoverable:", safechunk.brake_if_unrecoverable)
        print("sequential_oscbf_fallback:", safechunk.sequential_oscbf_fallback)


    all_step_metrics: list[StepMetrics] = []
    saved_action_episodes = []
    all_episode_summaries: list[dict] = []
    all_chunk_trajectory_records: list[dict] = []
    all_human_arm_trajectory_samples: list[dict] = []
    trajectory_plot_paths: list[str] = []
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

            reset_seed = args.seed + episode
            if policy_env is None:
                obs, info = env.reset(seed=reset_seed)
                if args.freeze_human_arm:
                    frozen = _freeze_human_arm(env)
                    print(f"freeze_human_arm: episode={episode} frozen={frozen}")
                else:
                    challenged = _configure_human_arm_challenge(env, args)
                    if challenged and episode == 0:
                        print(f"human_arm_challenge: episode={episode} configured={challenged}")
                if args.hide_human_arm_policy_obs:
                    policy_obs = _policy_obs_with_hidden_human_arm(env, obs)
                else:
                    policy_obs = obs
            else:
                obs, info = env.reset(seed=reset_seed)
                if args.freeze_human_arm:
                    frozen = _freeze_human_arm(env)
                    print(f"freeze_human_arm: episode={episode} frozen={frozen}")
                else:
                    challenged = _configure_human_arm_challenge(env, args)
                    if challenged and episode == 0:
                        print(f"human_arm_challenge: episode={episode} configured={challenged}")
                policy_obs, _policy_info = policy_env.reset(seed=reset_seed)

            safety_runtime_env = safety_env if safety_env is not None else env
            if safety_env is not None:
                safety_env.reset(seed=reset_seed)
                sync_counts = _sync_named_mujoco_state(env, safety_env)
                legs_synced = _sync_animated_legs(safety_env, is_moving=False)
                if args.freeze_human_arm:
                    frozen = _freeze_human_arm(safety_env)
                    print(f"safety_env freeze_human_arm: episode={episode} frozen={frozen}")
                else:
                    challenged = _configure_human_arm_challenge(safety_env, args)
                    if challenged and episode == 0:
                        print(f"safety_env human_arm_challenge: episode={episode} configured={challenged}")
                if episode == 0:
                    print(
                        "safety_env mirrored_state: "
                        f"joints={sync_counts['joints']} actuators={sync_counts['actuators']} "
                        f"animated_legs={legs_synced}"
                    )
            video_recorder.init(safety_runtime_env, enabled=args.record_video)
            episode_metrics: list[StepMetrics] = []
            episode_chunk_trajectory_records: list[dict] = []
            episode_human_arm_trajectory_samples: list[dict] = []
            saved_episode_actions = []
            episode_stop_reason = None
            policy_video_frames = []
            policy_video_timestamps = []
            gripper_latched = False
            policy_step = 0
            human_done_clear_steps = 0
            action_history_reset_after_exit = False
            pause_restart_reset_after_exit = False
            initial_pause_restart_reset = False
            last_safety_intervention_active = False
            last_diagnostic_step_mode = None
            phase_reanchor_steps_left = 0
            phase_reanchor_cooldown_left = 0
            phase_reanchor_drawer_history = []
            phase_reanchor_reset_after_step = False
            post_recovery_task_guard_steps_left = 0
            post_recovery_task_guard_reason = None
            post_recovery_task_guard_best_progress = None
            post_recovery_progress_regression = None
            post_recovery_reanchor_started = False
            if hasattr(safechunk, "reset"):
                safechunk.reset()
            episode_wall_t0 = time.perf_counter()
            last_step_wall_t = episode_wall_t0
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
                blocker_info = {}
                if safety_env is not None:
                    _sync_named_mujoco_state(env, safety_env)

                if not args.freeze_human_arm:
                    maybe_blocker_info = _update_temporary_human_blocker_if_present(safety_runtime_env)
                    if maybe_blocker_info is None:
                        human_anchor_xy = None
                        if args.human_arm_ee_obstruction and not args.human_arm_drawer_obstruction:
                            human_anchor_xy = _robot_gripper_geom_world_xy(
                                safety_runtime_env,
                                offset_xy=args.human_arm_ee_offset_xy,
                            )
                            if human_anchor_xy is None:
                                ee_state = extract_h1_state(env)
                                human_anchor_xy = _robot_ee_world_xy(
                                    oscbf,
                                    np.asarray(ee_state.q_full, dtype=np.float32).reshape(-1),
                                    np.asarray(ee_state.qd_full, dtype=np.float32).reshape(-1),
                                    offset_xy=args.human_arm_ee_offset_xy,
                                )
                        _update_scripted_human_arm_pose(
                            safety_runtime_env,
                            args,
                            step=step,
                            anchor_xy=human_anchor_xy,
                        )
                    else:
                        blocker_info = maybe_blocker_info
                elif safety_env is not None:
                    _sync_animated_legs(safety_env, is_moving=False)

                human_arm_trace_sample = None
                human_arm_stride = max(1, int(args.human_arm_trajectory_stride))
                if trajectory_logging_enabled and step % human_arm_stride == 0:
                    human_arm_trace_sample = _human_arm_trajectory_sample(
                        safety_runtime_env,
                        episode,
                        step,
                    )
                    if human_arm_trace_sample is not None:
                        episode_human_arm_trajectory_samples.append(human_arm_trace_sample)
                        all_human_arm_trajectory_samples.append(human_arm_trace_sample)

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

                task_state_before = (
                    _diagnostic_task_state(env)
                    if args.diagnostics_enabled
                    else {
                        "drawer_open_distance": None,
                        "drawer_open_fraction": None,
                        "drawer_joint_position": None,
                        "task_progress": None,
                        "ee_object_distance": None,
                        "object_state": None,
                    }
                )

                phase_reanchor_state = None
                phase_reanchor_drawer_progress = None
                phase_reanchor_reset_after_step = False
                if args.phase_reanchor or args.post_recovery_task_guard:
                    if phase_reanchor_cooldown_left > 0:
                        phase_reanchor_cooldown_left -= 1
                    phase_reanchor_state = _phase_reanchor_state(env, args)
                    if phase_reanchor_state is not None:
                        drawer_fraction = phase_reanchor_state.get("drawer_open_fraction")
                        if drawer_fraction is not None and np.isfinite(float(drawer_fraction)):
                            phase_reanchor_drawer_history.append(float(drawer_fraction))

                post_recovery_progress_regression = None
                post_recovery_reanchor_started = False
                post_recovery_task_guard_ready = False
                post_recovery_task_guard_phase_reason = None
                if args.post_recovery_task_guard:
                    (
                        post_recovery_task_guard_ready,
                        post_recovery_task_guard_phase_reason,
                    ) = _post_recovery_task_guard_ready(
                        task_state_before,
                        phase_reanchor_state,
                        args,
                    )
                    progress_before = _finite_task_progress(task_state_before)
                    if progress_before is not None:
                        if (
                            post_recovery_task_guard_best_progress is None
                            or progress_before > post_recovery_task_guard_best_progress
                        ):
                            post_recovery_task_guard_best_progress = progress_before
                        regression = post_recovery_task_guard_best_progress - progress_before
                        if (
                            regression > args.post_recovery_progress_tolerance
                            and post_recovery_task_guard_ready
                        ):
                            post_recovery_progress_regression = float(regression)
                            post_recovery_task_guard_steps_left = max(
                                post_recovery_task_guard_steps_left,
                                int(args.post_recovery_task_guard_steps),
                            )
                            post_recovery_task_guard_reason = "progress_regression:" + str(
                                post_recovery_task_guard_phase_reason
                            )

                    if post_recovery_task_guard_steps_left > 0:
                        if args.post_recovery_task_guard_force_gripper:
                            gripper_latched = True
                        reanchor_allowed, _guard_phase = _post_recovery_task_guard_reanchor_allowed(
                            phase_reanchor_state,
                            args,
                        )
                        if reanchor_allowed and phase_reanchor_steps_left <= 0:
                            phase_reanchor_steps_left = max(
                                1,
                                int(post_recovery_task_guard_steps_left),
                            )
                            phase_reanchor_cooldown_left = 0
                            post_recovery_reanchor_started = True
                        post_recovery_task_guard_steps_left = max(
                            0,
                            post_recovery_task_guard_steps_left - 1,
                        )

                if (
                    args.initial_pause_restart_steps > 0
                    and not initial_pause_restart_reset
                    and step >= args.initial_pause_restart_steps
                    and replay_actions is None
                ):
                    reset_count = _reset_action_sequence_history(env)
                    if policy_env is not None:
                        reset_count += _reset_action_sequence_history(policy_env)
                    if hasattr(safechunk, "reset"):
                        safechunk.reset()
                    policy_step = 0
                    initial_pause_restart_reset = True
                    if episode == 0 or args.debug:
                        print(
                            "initial_pause_restart: reset_action_history "
                            f"step={step} reset_wrappers={reset_count}"
                        )

                if args.record_policy_video and step % args.policy_video_every == 0:
                    policy_frame = _policy_obs_rgb_frame(policy_obs)
                    if policy_frame is not None:
                        policy_video_frames.append(policy_frame)
                        policy_video_timestamps.append(time.perf_counter())

                if replay_actions is None:
                    policy_obs_for_action = _adapt_policy_obs_to_space(
                        policy_obs,
                        policy_observation_space,
                    )
                    env_action = policy_action(ws, policy_obs_for_action, step=policy_step)
                    env_action = normalise_env_action_shape(env_action, env_action_shape)
                else:
                    if episode >= replay_actions.shape[0] or step >= replay_actions.shape[1]:
                        print(
                            f"Stopping episode {episode}: replay actions ended at "
                            f"shape {replay_actions.shape}."
                        )
                        break
                    env_action = replay_actions[episode, step].copy()

                first_action = extract_first_action(env_action)
                chunk_filter_mode = args.condition in {"sequential", "sequential_oscbf", "chunk_deform"}
                pelvis_cbf_mode = (
                    args.condition == "oscbf"
                    and getattr(oscbf, "enable_pelvis_cbf", False)
                    and getattr(oscbf, "pelvis_oscbf_config", None) is not None
                )
                arm_idx = (
                    safechunk.controlled_action_indices
                    if chunk_filter_mode
                    else (
                        oscbf.bigym_action_safety_indices
                        if pelvis_cbf_mode
                        else oscbf.bigym_action_arm_indices
                    )
                )
                state_idx = (
                    safechunk.controlled_state_indices
                    if chunk_filter_mode
                    else (
                        oscbf.bigym_state_safety_indices
                        if pelvis_cbf_mode
                        else oscbf.bigym_state_arm_indices
                    )
                )
                base_idx = getattr(oscbf, "bigym_action_base_indices", np.asarray([], dtype=np.int64))
                valid_base_idx = base_idx[base_idx < first_action.shape[0]]
                non_arm_idx = get_non_arm_indices(first_action.shape[0], arm_idx)

                monitor_t0 = time.perf_counter()
                min_h, h_values, h_violation = compute_oscbf_h_monitor(
                    filt=oscbf,
                    env=safety_runtime_env,
                    obs=obs,
                    q_full=q_full,
                    qd_full=qd_full,
                )
                monitor_time_ms = 1000.0 * (time.perf_counter() - monitor_t0)

                filter_t0 = time.perf_counter()
                safety_info = {}
                chunk_trace_context = None

                if chunk_filter_mode:
                    chunk_obs = _chunk_obs_with_q(obs, q_full)
                    horizon_operator.set_context(safety_runtime_env, chunk_obs, q_full, qd_full)
                    nominal_chunk, was_single_chunk = _as_chunk(env_action)
                    chunk_trace_context = {
                        "obs": chunk_obs,
                        "nominal_chunk": np.asarray(nominal_chunk, dtype=np.float32).copy(),
                    }
                    if args.condition in {"sequential", "sequential_oscbf"}:
                        safe_chunk, safety_info = safechunk.deform_chunk_with_oscbf(
                            chunk_obs,
                            nominal_chunk,
                            env=safety_runtime_env,
                            q_full=q_full,
                            qd_full=qd_full,
                        )
                        safety_info = dict(safety_info)
                        safety_info.update({
                            "safety_mode": "sequential_oscbf",
                            "mode": "sequential_oscbf",
                            "deformation_source": "sequential_oscbf",
                        })
                    else:
                        safe_chunk, safety_info = safechunk.filter_chunk(
                            chunk_obs,
                            nominal_chunk,
                            env=safety_runtime_env,
                            q_full=q_full,
                            qd_full=qd_full,
                            task_progress=task_state_before.get("task_progress"),
                            live_monitor_min_h=min_h,
                            live_monitor_h_violation=h_violation,
                        )
                    safe_env_action = _restore_action_shape(
                        np.asarray(safe_chunk, dtype=np.float32),
                        was_single_chunk,
                    )
                    safe_first_action = extract_first_action(safe_env_action)
                elif args.condition == "oscbf":
                    safe_first_action = oscbf(
                        action=first_action,
                        env=safety_runtime_env,
                        observations=obs,
                        q_full=q_full,
                        qd_full=qd_full,
                    )
                    safe_env_action = replace_first_action(
                        env_action=env_action,
                        safe_first_action=safe_first_action,
                    )
                else:
                    safe_first_action = first_action.copy()
                    safe_env_action = env_action.copy()

                policy_hold_active = False
                if args.pause_policy_step_on_brake and replay_actions is None:
                    policy_hold_active = _should_hold_policy_step(
                        safety_info,
                        first_action,
                        safe_first_action,
                        arm_idx,
                        args.intervention_eps,
                    )

                phase_for_pause_restart = _safe_info_get(blocker_info, "human_phase")
                pause_restart_active = (
                    args.pause_and_restart_on_human_blocker
                    and phase_for_pause_restart in {"enter", "hold", "exit"}
                )
                initial_pause_restart_active = (
                    args.initial_pause_restart_steps > 0
                    and step < args.initial_pause_restart_steps
                )
                pause_active, pause_reason = _should_pause_for_safety(args, min_h, safety_info)
                if bool(_safe_info_get(safety_info, "suppress_outer_pause")):
                    pause_active = False
                    pause_reason = None
                if pause_restart_active or initial_pause_restart_active:
                    pause_active = True
                    pause_reason = (
                        "initial_pause_restart"
                        if initial_pause_restart_active
                        else "human_blocker_pause_restart"
                    )
                if pause_active:
                    pause_action_idx = arm_idx
                    pause_state_idx = state_idx
                    if pause_restart_active or initial_pause_restart_active:
                        pause_action_idx = getattr(
                            oscbf,
                            "bigym_action_safety_indices",
                            arm_idx,
                        )
                        pause_state_idx = getattr(
                            oscbf,
                            "bigym_state_safety_indices",
                            state_idx,
                        )
                    safe_env_action = _scale_controlled_motion_from_current_q(
                        safe_env_action,
                        q_full,
                        pause_action_idx,
                        pause_state_idx,
                        args.pause_motion_scale,
                    )
                    safe_first_action = extract_first_action(safe_env_action)
                    safety_info = dict(safety_info)
                    safety_info.update(
                        {
                            "safety_mode": (
                                "pause_and_restart"
                                if pause_restart_active
                                else "pause_on_unsafe"
                            ),
                            "mode": (
                                "pause_and_restart"
                                if pause_restart_active
                                else "pause_on_unsafe"
                            ),
                            "pause_reason": pause_reason,
                        }
                    )
                    if args.pause_policy_step_on_brake and replay_actions is None:
                        policy_hold_active = True

                filter_time_ms = 1000.0 * (time.perf_counter() - filter_t0)

                assertion_idx = arm_idx
                if pause_active and (pause_restart_active or initial_pause_restart_active):
                    assertion_idx = pause_action_idx
                if chunk_filter_mode:
                    _assert_chunk_properties(
                        env_action,
                        safe_env_action,
                        assertion_idx,
                    )
                else:
                    assert_action_properties(
                        nominal_action=first_action,
                        safe_action=safe_first_action,
                        arm_indices=assertion_idx,
                    )

                if (args.phase_reanchor or args.post_recovery_task_guard) and replay_actions is None:
                    if phase_reanchor_steps_left <= 0 and args.phase_reanchor:
                        should_start_reanchor, phase_reanchor_drawer_progress = _should_start_phase_reanchor(
                            args,
                            step,
                            phase_reanchor_state,
                            phase_reanchor_drawer_history,
                            phase_reanchor_cooldown_left,
                        )
                        if should_start_reanchor:
                            phase_reanchor_steps_left = int(args.phase_reanchor_steps)
                            reset_count = _reset_action_sequence_history(env)
                            if policy_env is not None:
                                reset_count += _reset_action_sequence_history(policy_env)
                            if hasattr(safechunk, "reset"):
                                safechunk.reset()
                            policy_step = 0
                            if episode == 0 or args.debug:
                                print(
                                    "phase_reanchor: start "
                                    f"episode={episode} step={step} "
                                    f"phase={phase_reanchor_state.get('phase')} "
                                    f"drawer_fraction={phase_reanchor_state.get('drawer_open_fraction'):.3f} "
                                    f"window_progress={phase_reanchor_drawer_progress:.3f} "
                                    f"reset_wrappers={reset_count}"
                                )

                    if phase_reanchor_steps_left > 0:
                        if (
                            phase_reanchor_state is None
                            or phase_reanchor_state.get("phase") == "done"
                        ):
                            phase_reanchor_steps_left = 0
                            phase_reanchor_cooldown_left = int(args.phase_reanchor_cooldown_steps)
                            phase_reanchor_reset_after_step = True
                        else:
                            reanchor_action, reanchor_info = _phase_reanchor_action(
                                env,
                                safe_env_action,
                                q_full,
                                oscbf,
                                args,
                                phase_reanchor_state,
                            )
                            if reanchor_action is None:
                                phase_reanchor_steps_left = 0
                                phase_reanchor_cooldown_left = int(args.phase_reanchor_cooldown_steps)
                                if episode == 0 or args.debug:
                                    print(
                                        "phase_reanchor: unavailable "
                                        f"episode={episode} step={step}"
                                    )
                            else:
                                reanchor_to_execute = reanchor_action
                                reanchor_acceptance = None
                                reanchor_accepted = True
                                if (
                                    args.post_recovery_task_guard_check_safety
                                    and chunk_filter_mode
                                    and hasattr(safechunk, "evaluate_candidate_acceptance")
                                ):
                                    try:
                                        reanchor_acceptance = safechunk.evaluate_candidate_acceptance(
                                            _chunk_obs_with_q(obs, q_full),
                                            reanchor_action,
                                            "deform",
                                        )
                                        reanchor_accepted = bool(
                                            reanchor_acceptance.get("accepted")
                                        )
                                        if (
                                            reanchor_accepted
                                            and reanchor_acceptance.get("safe_prefix_execution")
                                            and hasattr(safechunk, "_truncate_chunk_to_safe_prefix")
                                        ):
                                            reanchor_to_execute = safechunk._truncate_chunk_to_safe_prefix(
                                                reanchor_action,
                                                reanchor_acceptance,
                                            )
                                    except Exception as exc:  # noqa: BLE001
                                        reanchor_accepted = False
                                        reanchor_acceptance = {
                                            "rejection_reason": f"acceptance_error:{type(exc).__name__}",
                                        }

                                if not reanchor_accepted:
                                    phase_reanchor_steps_left = 0
                                    phase_reanchor_cooldown_left = int(args.phase_reanchor_cooldown_steps)
                                    phase_reanchor_reset_after_step = True
                                    safety_info = dict(safety_info)
                                    safety_info.update(
                                        {
                                            "phase_reanchor_rejected": True,
                                            "phase_reanchor_reject_reason": (
                                                reanchor_acceptance or {}
                                            ).get("rejection_reason"),
                                            "phase_reanchor_acceptance_type": (
                                                reanchor_acceptance or {}
                                            ).get("acceptance_type"),
                                            "phase_reanchor_immediate_clearance": (
                                                reanchor_acceptance or {}
                                            ).get("immediate_clearance"),
                                            "phase_reanchor_horizon_min_clearance": (
                                                reanchor_acceptance or {}
                                            ).get("horizon_min_clearance"),
                                        }
                                    )
                                else:
                                    safe_env_action = reanchor_to_execute
                                    safe_first_action = extract_first_action(safe_env_action)
                                    phase_reanchor_steps_left -= 1
                                    if phase_reanchor_steps_left <= 0:
                                        phase_reanchor_cooldown_left = int(args.phase_reanchor_cooldown_steps)
                                        phase_reanchor_reset_after_step = True
                                    safety_info = dict(safety_info)
                                    safety_info.update(
                                        {
                                            "safety_mode": "phase_reanchor",
                                            "mode": "phase_reanchor",
                                            "pause_reason": (
                                                "phase_reanchor:"
                                                f"{reanchor_info.get('phase', 'unknown')}"
                                            ),
                                            "deformation_source": "phase_reanchor",
                                            "phase_reanchor_steps_left": int(phase_reanchor_steps_left),
                                            "phase_reanchor_base_cmd_xy": reanchor_info.get("base_cmd_xy"),
                                            "phase_reanchor_ee_error_xy": reanchor_info.get("ee_error_xy"),
                                            "phase_reanchor_drawer_fraction": reanchor_info.get("drawer_open_fraction"),
                                            "phase_reanchor_ee_to_handle_dist": reanchor_info.get("ee_to_handle_dist"),
                                            "phase_reanchor_rejected": False,
                                            "phase_reanchor_acceptance_type": (
                                                reanchor_acceptance or {}
                                            ).get("acceptance_type"),
                                            "phase_reanchor_immediate_clearance": (
                                                reanchor_acceptance or {}
                                            ).get("immediate_clearance"),
                                            "phase_reanchor_horizon_min_clearance": (
                                                reanchor_acceptance or {}
                                            ).get("horizon_min_clearance"),
                                        }
                                    )
                                    policy_hold_active = True

                latch_dim = int(args.gripper_latch_dim)
                latch_requested = bool(
                    args.gripper_latch
                    or (
                        args.post_recovery_task_guard
                        and args.post_recovery_task_guard_force_gripper
                        and gripper_latched
                    )
                )
                if latch_requested:
                    if not -safe_first_action.shape[0] <= latch_dim < safe_first_action.shape[0]:
                        raise ValueError(
                            f"--gripper-latch-dim {latch_dim} is out of range "
                            f"for action shape {safe_first_action.shape}."
                        )
                    if (
                        args.gripper_latch
                        and not gripper_latched
                        and step >= args.gripper_latch_start_step
                        and safe_first_action[latch_dim] >= args.gripper_latch_trigger
                    ):
                        gripper_latched = True
                        if args.debug:
                            print(
                                f"gripper latch activated at episode={episode} "
                                f"step={step} dim={latch_dim} "
                                f"value={safe_first_action[latch_dim]:.3f}"
                            )
                    if gripper_latched:
                        safe_first_action[latch_dim] = args.gripper_latch_value

                raw_first_action = _raw_scaled_first_action(env, safe_first_action)
                if raw_first_action is None:
                    raw_action_norm = None
                    raw_arm_min = None
                    raw_arm_max = None
                else:
                    raw_arm = raw_first_action[arm_idx]
                    raw_action_norm = float(np.linalg.norm(raw_first_action))
                    raw_arm_min = float(np.min(raw_arm))
                    raw_arm_max = float(np.max(raw_arm))

                if latch_requested and gripper_latched:
                    if safe_env_action.ndim == 1:
                        safe_env_action[latch_dim] = args.gripper_latch_value
                    else:
                        safe_env_action[:, latch_dim] = args.gripper_latch_value

                safe_gripper_action = (
                    float(safe_first_action[latch_dim])
                    if -safe_first_action.shape[0] <= latch_dim < safe_first_action.shape[0]
                    else None
                )
                raw_gripper_action = (
                    float(raw_first_action[latch_dim])
                    if raw_first_action is not None
                    and -raw_first_action.shape[0] <= latch_dim < raw_first_action.shape[0]
                    else None
                )

                post_recovery_task_guard_active = bool(
                    args.post_recovery_task_guard
                    and (
                        post_recovery_task_guard_steps_left > 0
                        or post_recovery_reanchor_started
                        or post_recovery_progress_regression is not None
                        or (
                            post_recovery_task_guard_reason is not None
                            and phase_reanchor_steps_left > 0
                        )
                    )
                )
                safety_info = dict(safety_info)
                safety_info.update(
                    {
                        "gripper_latched": bool(gripper_latched),
                        "gripper_latch_dim": int(latch_dim),
                        "safe_gripper_action": safe_gripper_action,
                        "raw_gripper_action": raw_gripper_action,
                        "post_recovery_task_guard_active": post_recovery_task_guard_active,
                        "post_recovery_task_guard_steps_left": int(post_recovery_task_guard_steps_left),
                        "post_recovery_task_guard_reason": post_recovery_task_guard_reason,
                        "post_recovery_task_guard_best_progress": post_recovery_task_guard_best_progress,
                        "post_recovery_progress_regression": post_recovery_progress_regression,
                        "post_recovery_reanchor_started": bool(post_recovery_reanchor_started),
                    }
                )

                if (
                    trajectory_logging_enabled
                    and chunk_filter_mode
                    and chunk_trace_context is not None
                    and len(all_chunk_trajectory_records) < max(0, int(args.chunk_trajectory_max_events))
                    and _should_log_chunk_trajectory_trace(
                        args,
                        safety_info,
                        env_action,
                        safe_env_action,
                        args.intervention_eps,
                    )
                ):
                    trace_record = _collect_chunk_trajectory_trace(
                        args=args,
                        episode=episode,
                        step=step,
                        safechunk=safechunk,
                        horizon_operator=horizon_operator,
                        obs=chunk_trace_context["obs"],
                        nominal_chunk=chunk_trace_context["nominal_chunk"],
                        generated_chunk=safe_env_action,
                        safety_info=safety_info,
                        human_sample=human_arm_trace_sample,
                    )
                    if trace_record is not None:
                        episode_chunk_trajectory_records.append(trace_record)
                        all_chunk_trajectory_records.append(trace_record)

                saved_episode_actions.append(
                    np.asarray(safe_env_action, dtype=np.float32).copy()
                )

                safety_intervention_active = _is_safety_intervention_mode(safety_info)
                reset_policy_after_intervention = (
                    replay_actions is None
                    and chunk_filter_mode
                    and safety_intervention_active
                    and not last_safety_intervention_active
                )

                obs, reward, terminated, truncated, info = env.step(safe_env_action)
                if policy_env is None:
                    if args.hide_human_arm_policy_obs:
                        policy_obs = _policy_obs_with_hidden_human_arm(
                            env,
                            obs,
                            prev_policy_obs=policy_obs,
                        )
                    else:
                        policy_obs = obs
                else:
                    policy_obs, _policy_reward, policy_terminated, policy_truncated, _policy_info = policy_env.step(
                        safe_env_action
                    )
                    if (policy_terminated or policy_truncated) and not (
                        terminated or truncated or extract_success(info, float(reward), bool(terminated))
                    ):
                        print(
                            "Warning: clean policy env ended before eval env; "
                            "policy/eval states may have diverged."
                        )

                task_state_after = (
                    _diagnostic_task_state(env)
                    if args.diagnostics_enabled
                    else {
                        "drawer_open_distance": None,
                        "drawer_open_fraction": None,
                        "drawer_joint_position": None,
                        "task_progress": None,
                        "ee_object_distance": None,
                        "object_state": None,
                    }
                )
                task_progress_delta = _diagnostic_progress_delta(
                    task_state_before,
                    task_state_after,
                )

                if replay_actions is None:
                    if reset_policy_after_intervention:
                        reset_count = _reset_action_sequence_history(env)
                        if policy_env is not None:
                            reset_count += _reset_action_sequence_history(policy_env)
                        if episode == 0 and args.debug and reset_count > 0:
                            mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
                            print(
                                "intervention_requery: reset_action_history "
                                f"step={step} mode={mode} reset_wrappers={reset_count}"
                            )
                    if (
                        args.reset_action_history_after_recovery
                        and chunk_filter_mode
                        and bool(_safe_info_get(safety_info, "request_action_history_reset_after_recovery"))
                    ):
                        reset_count = _reset_action_sequence_history(env)
                        if policy_env is not None:
                            reset_count += _reset_action_sequence_history(policy_env)
                        policy_step = 0
                        policy_hold_active = True
                        safety_info = dict(safety_info)
                        safety_info.update(
                            {
                                "recovery_action_history_reset": True,
                                "recovery_action_history_reset_count": int(reset_count),
                            }
                        )
                        if args.post_recovery_task_guard:
                            progress_after_recovery = _finite_task_progress(task_state_after)
                            if progress_after_recovery is not None:
                                if (
                                    post_recovery_task_guard_best_progress is None
                                    or progress_after_recovery > post_recovery_task_guard_best_progress
                                ):
                                    post_recovery_task_guard_best_progress = progress_after_recovery
                            guard_ready, guard_phase_reason = _post_recovery_task_guard_ready(
                                task_state_after,
                                phase_reanchor_state,
                                args,
                            )
                            if guard_ready:
                                post_recovery_task_guard_steps_left = max(
                                    post_recovery_task_guard_steps_left,
                                    int(args.post_recovery_task_guard_steps),
                                )
                                post_recovery_task_guard_reason = "recovery_completed:" + str(
                                    guard_phase_reason
                                )
                                if args.post_recovery_task_guard_force_gripper:
                                    gripper_latched = True
                                reanchor_allowed, _guard_phase = _post_recovery_task_guard_reanchor_allowed(
                                    phase_reanchor_state,
                                    args,
                                )
                                if reanchor_allowed:
                                    phase_reanchor_steps_left = max(
                                        phase_reanchor_steps_left,
                                        int(args.post_recovery_task_guard_steps),
                                    )
                                    phase_reanchor_cooldown_left = 0
                                post_recovery_guard_active_after_reset = True
                                post_recovery_guard_reanchor_after_reset = bool(reanchor_allowed)
                            else:
                                post_recovery_task_guard_reason = "suppressed:" + str(
                                    guard_phase_reason
                                )
                                post_recovery_guard_active_after_reset = False
                                post_recovery_guard_reanchor_after_reset = False
                            safety_info.update(
                                {
                                    "gripper_latched": bool(gripper_latched),
                                    "post_recovery_task_guard_active": bool(post_recovery_guard_active_after_reset),
                                    "post_recovery_task_guard_steps_left": int(post_recovery_task_guard_steps_left),
                                    "post_recovery_task_guard_reason": post_recovery_task_guard_reason,
                                    "post_recovery_task_guard_best_progress": post_recovery_task_guard_best_progress,
                                    "post_recovery_progress_regression": post_recovery_progress_regression,
                                    "post_recovery_reanchor_started": bool(post_recovery_guard_reanchor_after_reset),
                                }
                            )
                        if episode == 0 or args.debug:
                            mode = _safe_info_get(safety_info, "safety_mode") or _safe_info_get(safety_info, "mode")
                            print(
                                "recovery_requery: reset_action_history "
                                f"step={step} mode={mode} reset_wrappers={reset_count}"
                            )

                    if phase_reanchor_reset_after_step:
                        reset_count = _reset_action_sequence_history(env)
                        if policy_env is not None:
                            reset_count += _reset_action_sequence_history(policy_env)
                        if hasattr(safechunk, "reset"):
                            safechunk.reset()
                        policy_step = 0
                        policy_hold_active = True
                        if episode == 0 or args.debug:
                            print(
                                "phase_reanchor: finish "
                                f"episode={episode} step={step} reset_wrappers={reset_count} "
                                f"cooldown={phase_reanchor_cooldown_left}"
                            )
                    if not policy_hold_active:
                        policy_step += 1

                last_safety_intervention_active = bool(safety_intervention_active)

                if safety_env is not None:
                    # Mirror the robot/task state after the action, but keep the human
                    # pose fixed to the one the filter just evaluated. The next human
                    # update happens at the start of the next control step, before
                    # monitor/filter, so contacts are measured against a visible pose.
                    _sync_named_mujoco_state(env, safety_env)
                    _sync_animated_legs(safety_env, is_moving=True)

                phase_for_resume = _safe_info_get(blocker_info, "human_phase")
                dist_for_resume = _safe_info_get(blocker_info, "min_robot_human_distance")
                should_reset_after_human_exit = (
                    args.reset_action_history_after_human_exit
                    and not action_history_reset_after_exit
                )
                should_restart_after_blocker_pause = (
                    args.pause_and_restart_on_human_blocker
                    and not pause_restart_reset_after_exit
                )
                if (
                    (should_reset_after_human_exit or should_restart_after_blocker_pause)
                    and phase_for_resume == "done"
                ):
                    if (
                        dist_for_resume is None
                        or float(dist_for_resume) >= args.resume_clearance_threshold
                    ):
                        human_done_clear_steps += 1
                    else:
                        human_done_clear_steps = 0
                    if human_done_clear_steps >= args.resume_clear_steps:
                        reset_count = _reset_action_sequence_history(env)
                        if policy_env is not None:
                            reset_count += _reset_action_sequence_history(policy_env)
                        if hasattr(safechunk, "reset"):
                            safechunk.reset()
                        policy_step = 0
                        if should_reset_after_human_exit:
                            action_history_reset_after_exit = True
                        if should_restart_after_blocker_pause:
                            pause_restart_reset_after_exit = True
                        if episode == 0 or args.debug:
                            reason = (
                                "pause_and_restart_after_human_blocker"
                                if should_restart_after_blocker_pause
                                else "reset_action_history_after_human_exit"
                            )
                            print(
                                f"resume_supervisor: {reason} "
                                f"step={step} reset_wrappers={reset_count}"
                            )
                elif phase_for_resume != "done":
                    human_done_clear_steps = 0

                video_recorder.record(safety_runtime_env)
                step_wall_t = time.perf_counter()
                elapsed_wall_time_s = step_wall_t - episode_wall_t0
                step_wall_time_s = step_wall_t - last_step_wall_t
                last_step_wall_t = step_wall_t

                nominal_arm = first_action[arm_idx]
                safe_arm = safe_first_action[arm_idx]

                arm_delta = float(np.linalg.norm(safe_arm - nominal_arm))
                base_delta = float(
                    np.linalg.norm(
                        safe_first_action[valid_base_idx] - first_action[valid_base_idx]
                    )
                ) if valid_base_idx.size else 0.0
                non_arm_delta = float(
                    np.linalg.norm(
                        safe_first_action[non_arm_idx] - first_action[non_arm_idx]
                    )
                )
                full_delta = float(np.linalg.norm(safe_first_action - first_action))

                nominal_chunk_for_metrics, _ = _as_chunk(env_action)
                safe_chunk_for_metrics, _ = _as_chunk(safe_env_action)
                chunk_arm_delta = float(
                    np.linalg.norm(
                        safe_chunk_for_metrics[:, arm_idx]
                        - nominal_chunk_for_metrics[:, arm_idx]
                    )
                )
                chunk_base_delta = float(
                    np.linalg.norm(
                        safe_chunk_for_metrics[:, valid_base_idx]
                        - nominal_chunk_for_metrics[:, valid_base_idx]
                    )
                ) if valid_base_idx.size else 0.0
                chunk_non_arm_delta = float(
                    np.linalg.norm(
                        safe_chunk_for_metrics[:, non_arm_idx]
                        - nominal_chunk_for_metrics[:, non_arm_idx]
                    )
                )
                chunk_full_delta = float(
                    np.linalg.norm(safe_chunk_for_metrics - nominal_chunk_for_metrics)
                )
                chunk_advantage_metrics = _chunk_filter_advantage_metrics(
                    nominal_chunk_for_metrics,
                    safe_chunk_for_metrics,
                    arm_idx,
                    args.intervention_eps,
                )
                path_consistency_metrics = (
                    _path_consistency_metrics(
                        safechunk,
                        _chunk_obs_with_q(obs, q_full),
                        nominal_chunk_for_metrics,
                        safe_chunk_for_metrics,
                    )
                    if chunk_filter_mode
                    else {
                        "path_mean_deviation": None,
                        "path_max_deviation": None,
                        "path_final_deviation": None,
                    }
                )
                if args.diagnostics_enabled:
                    diagnostic_flags = _diagnostic_mode_flags(
                        safety_info,
                        arm_delta=arm_delta,
                        eps=args.intervention_eps,
                    )
                    current_diagnostic_mode = diagnostic_flags["diagnostic_step_mode"]
                    mode_transition = (
                        f"{last_diagnostic_step_mode}->{current_diagnostic_mode}"
                        if last_diagnostic_step_mode is not None
                        and current_diagnostic_mode != last_diagnostic_step_mode
                        else None
                    )
                    last_diagnostic_step_mode = current_diagnostic_mode
                else:
                    diagnostic_flags = {
                        "diagnostic_step_mode": None,
                        "act_step": None,
                        "deform_step": None,
                        "recover_step": None,
                        "brake_step": None,
                        "fallback_step": None,
                        "optimized_attempt_step": None,
                        "optimized_accepted_step": None,
                    }
                    mode_transition = None
                (
                    horizon_risk_gap,
                    horizon_risk_gap_active,
                    horizon_clearance_drop,
                ) = _horizon_risk_gap(
                    min_h,
                    _safe_info_get(safety_info, "min_clearance"),
                )

                success = extract_success(info, float(reward), bool(terminated))
                contact_pairs = robot_human_contact_pairs(safety_runtime_env)
                contact_count = None if contact_pairs is None else len(contact_pairs)
                contact_now = bool(contact_count is not None and contact_count > 0)
                safety_info["contact_during_hold"] = bool(
                    contact_now
                    and diagnostic_flags.get("brake_step")
                    and _safe_info_get(safety_info, "brake_hold_current")
                )
                safety_info["contact_during_brake"] = bool(
                    contact_now and diagnostic_flags.get("brake_step")
                )
                safety_info["contact_during_deform"] = bool(
                    contact_now and diagnostic_flags.get("deform_step")
                )
                safety_info["contact_during_recover"] = bool(
                    contact_now and diagnostic_flags.get("recover_step")
                )
                unmodelled_contact_reason = _unmodelled_robot_contact_reason(contact_pairs)
                info_dict = info if isinstance(info, dict) else {}
                if blocker_info:
                    info_dict = {**info_dict, **blocker_info}

                step_metrics = StepMetrics(
                    condition=args.condition,
                    episode=episode,
                    step=step,

                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    success=success,

                    human_phase=_optional_str(_safe_info_get(info_dict, "human_phase")),
                    ee_to_handle_dist=_optional_float(_safe_info_get(info_dict, "ee_to_handle_dist")),
                    human_blocker_triggered=_optional_bool(_safe_info_get(info_dict, "human_blocker_triggered")),
                    human_time_in_phase=_optional_float(_safe_info_get(info_dict, "human_time_in_phase")),
                    min_robot_human_distance=_optional_float(
                        _safe_info_get(info_dict, "min_robot_human_distance")
                        if _safe_info_get(info_dict, "min_robot_human_distance") is not None
                        else _safe_info_get(info_dict, "human_min_robot_distance")
                    ),
                    drawer_open_distance=_optional_float(
                        task_state_after.get("drawer_open_distance")
                        if task_state_after.get("drawer_open_distance") is not None
                        else _safe_info_get(info_dict, "drawer_open_distance")
                    ),
                    drawer_open_fraction=_optional_float(task_state_after.get("drawer_open_fraction")),
                    drawer_joint_position=_optional_float(task_state_after.get("drawer_joint_position")),
                    task_progress=_optional_float(task_state_after.get("task_progress")),
                    task_progress_before=_optional_float(task_state_before.get("task_progress")),
                    task_progress_after=_optional_float(task_state_after.get("task_progress")),
                    task_progress_delta=_optional_float(task_progress_delta),
                    ee_object_distance=_optional_float(task_state_after.get("ee_object_distance")),
                    object_state=task_state_after.get("object_state"),

                    min_h=min_h,
                    h_values=h_values,
                    h_violation=h_violation,
                    chunk_min_clearance=_safe_info_get(safety_info, "min_clearance"),
                    chunk_first_violation=_safe_info_get(safety_info, "first_violation"),
                    chunk_unsafe_count=_safe_info_get(safety_info, "unsafe_count"),
                    horizon_risk_gap=horizon_risk_gap,
                    horizon_risk_gap_active=horizon_risk_gap_active,
                    horizon_clearance_drop=horizon_clearance_drop,

                    contact_count=contact_count,
                    contact_pairs=contact_pairs,

                    arm_delta=arm_delta,
                    base_delta=base_delta,
                    non_arm_delta=non_arm_delta,
                    full_delta=full_delta,
                    per_step_action_delta_norm=full_delta,
                    per_step_arm_delta_norm=arm_delta,
                    per_step_base_delta_norm=base_delta,
                    chunk_arm_delta=chunk_arm_delta,
                    chunk_base_delta=chunk_base_delta,
                    chunk_non_arm_delta=chunk_non_arm_delta,
                    chunk_full_delta=chunk_full_delta,
                    chunk_action_delta_norm=chunk_full_delta,
                    chunk_arm_delta_norm=chunk_arm_delta,
                    chunk_base_delta_norm=chunk_base_delta,
                    chunk_modified_fraction=chunk_advantage_metrics["chunk_modified_fraction"],
                    chunk_modified_steps=chunk_advantage_metrics["chunk_modified_steps"],
                    chunk_first_modified_step=chunk_advantage_metrics["chunk_first_modified_step"],
                    chunk_last_modified_step=chunk_advantage_metrics["chunk_last_modified_step"],
                    chunk_mean_step_arm_delta=chunk_advantage_metrics["chunk_mean_step_arm_delta"],
                    chunk_max_step_arm_delta=chunk_advantage_metrics["chunk_max_step_arm_delta"],
                    chunk_future_arm_delta=chunk_advantage_metrics["chunk_future_arm_delta"],
                    chunk_future_edit_fraction=chunk_advantage_metrics["chunk_future_edit_fraction"],
                    chunk_first_edit_fraction=chunk_advantage_metrics["chunk_first_edit_fraction"],
                    chunk_safe_arm_variation=chunk_advantage_metrics["chunk_safe_arm_variation"],
                    chunk_nominal_arm_variation=chunk_advantage_metrics["chunk_nominal_arm_variation"],
                    chunk_arm_variation_delta=chunk_advantage_metrics["chunk_arm_variation_delta"],
                    chunk_edit_variation=chunk_advantage_metrics["chunk_edit_variation"],
                    path_mean_deviation=path_consistency_metrics["path_mean_deviation"],
                    path_max_deviation=path_consistency_metrics["path_max_deviation"],
                    path_final_deviation=path_consistency_metrics["path_final_deviation"],
                    chunk_preemptive_intervention=chunk_advantage_metrics["chunk_preemptive_intervention"],
                    intervention_active=bool(chunk_arm_delta > args.intervention_eps),

                    nominal_arm_min=float(np.min(nominal_arm)),
                    nominal_arm_max=float(np.max(nominal_arm)),
                    safe_arm_min=float(np.min(safe_arm)),
                    safe_arm_max=float(np.max(safe_arm)),

                    action_norm=float(np.linalg.norm(first_action)),
                    safe_action_norm=float(np.linalg.norm(safe_first_action)),
                    raw_action_norm=raw_action_norm,
                    raw_arm_min=raw_arm_min,
                    raw_arm_max=raw_arm_max,
                    chunk_action_norm=float(np.linalg.norm(nominal_chunk_for_metrics)),
                    safe_chunk_action_norm=float(np.linalg.norm(safe_chunk_for_metrics)),
                    safety_mode=(
                        _safe_info_get(safety_info, "safety_mode")
                        or _safe_info_get(safety_info, "mode")
                    ),
                    pause_reason=_safe_info_get(safety_info, "pause_reason"),
                    deformation_source=_safe_info_get(safety_info, "deformation_source"),
                    deformation_norm=_safe_info_get(safety_info, "deformation_norm"),
                    deform_safe=_safe_info_get(safety_info, "deform_safe"),
                    deform_min_clearance=_safe_info_get(safety_info, "deform_min_clearance"),
                    chunk_deform_scale=_safe_info_get(safety_info, "chunk_deform_scale"),
                    chunk_deform_attempts=_safe_info_get(safety_info, "chunk_deform_attempts"),
                    deform_mode=_safe_info_get(safety_info, "deform_mode"),
                    optimized_accepted=_safe_info_get(safety_info, "optimized_accepted"),
                    optimized_fallback=_safe_info_get(safety_info, "optimized_fallback"),
                    optimized_reject_reason=_safe_info_get(safety_info, "optimized_reject_reason"),
                    debug_safety_feasibility=_safe_info_get(
                        safety_info, "debug_safety_feasibility"
                    ),
                    safety_rejected=_safe_info_get(safety_info, "safety_rejected"),
                    recovery_rejected=_safe_info_get(safety_info, "recovery_rejected"),
                    rejection_cause=_safe_info_get(safety_info, "rejection_cause"),
                    best_min_clearance=_safe_info_get(safety_info, "best_min_clearance"),
                    required_min_clearance=_safe_info_get(
                        safety_info, "required_min_clearance"
                    ),
                    clearance_gap=_safe_info_get(safety_info, "clearance_gap"),
                    recovery_mode=_safe_info_get(safety_info, "recovery_mode"),
                    recovery_phase=_safe_info_get(safety_info, "recovery_phase"),
                    cached_motion_active=_safe_info_get(
                        safety_info, "cached_motion_active"
                    ),
                    deform_stage_min_clearance=_safe_info_get(
                        safety_info, "deform_stage_min_clearance"
                    ),
                    deform_stage_accepted=_safe_info_get(safety_info, "deform_stage_accepted", _safe_info_get(safety_info, "yield_accepted")),
                    recover_min_clearance=_safe_info_get(
                        safety_info, "recover_min_clearance"
                    ),
                    recover_rejoin_loss=_safe_info_get(
                        safety_info, "recover_rejoin_loss"
                    ),
                    recover_target_index=_safe_info_get(
                        safety_info, "recover_target_index"
                    ),
                    recover_accepted=_safe_info_get(safety_info, "recover_accepted", _safe_info_get(safety_info, "return_accepted")),
                    recover_required=_safe_info_get(safety_info, "recover_required"),
                    recovery_candidate_class=_safe_info_get(safety_info, "recovery_candidate_class"),
                    recover_reject_reason=_safe_info_get(safety_info, "recover_reject_reason"),
                    recover_path_min_clearance=_safe_info_get(safety_info, "recover_path_min_clearance"),
                    recover_immediate_clearance=_safe_info_get(safety_info, "recover_immediate_clearance"),
                    recover_prefix_min_clearance=_safe_info_get(safety_info, "recover_prefix_min_clearance"),
                    recover_path_safe=_safe_info_get(safety_info, "recover_path_safe"),
                    recover_immediate_safe=_safe_info_get(safety_info, "recover_immediate_safe"),
                    recover_prefix_safe=_safe_info_get(safety_info, "recover_prefix_safe"),
                    recover_safe_prefix_len=_safe_info_get(safety_info, "recover_safe_prefix_len"),
                    recover_target_key=_safe_info_get(safety_info, "recover_target_key"),
                    recovery_path_failure_streak=_safe_info_get(safety_info, "recovery_path_failure_streak"),
                    direct_rejoin_attempted=_safe_info_get(safety_info, "direct_rejoin_attempted"),
                    direct_rejoin_rejected=_safe_info_get(safety_info, "direct_rejoin_rejected"),
                    detour_rejoin_attempted=_safe_info_get(safety_info, "detour_rejoin_attempted"),
                    detour_rejoin_accepted=_safe_info_get(safety_info, "detour_rejoin_accepted"),
                    delayed_rejoin_active=_safe_info_get(safety_info, "delayed_rejoin_active"),
                    delayed_rejoin_steps=_safe_info_get(safety_info, "delayed_rejoin_steps"),
                    repeated_unsafe_target=_safe_info_get(safety_info, "repeated_unsafe_target"),
                    post_recovery_act_window_active=_safe_info_get(safety_info, "post_recovery_act_window_active"),
                    post_recovery_act_steps_remaining=_safe_info_get(safety_info, "post_recovery_act_steps_remaining"),
                    post_recovery_act_window_interrupted=_safe_info_get(safety_info, "post_recovery_act_window_interrupted"),
                    resumed_from_cached_index=_safe_info_get(
                        safety_info, "resumed_from_cached_index"
                    ),
                    is_recoverable=_safe_info_get(safety_info, "is_recoverable"),
                    rejoin_index=_safe_info_get(safety_info, "rejoin_index"),
                    rejoin_cost=_safe_info_get(safety_info, "rejoin_cost"),
                    safety_loss=_safe_info_get(safety_info, "safety_loss"),
                    action_deviation_loss=_safe_info_get(safety_info, "action_deviation_loss"),
                    path_loss=_safe_info_get(safety_info, "path_loss"),
                    rejoin_loss=_safe_info_get(safety_info, "rejoin_loss"),
                    q_rejoin_loss=_safe_info_get(safety_info, "q_rejoin_loss"),
                    q_rejoin_dist=_safe_info_get(safety_info, "q_rejoin_dist"),
                    q_rejoin_threshold=_safe_info_get(
                        safety_info, "q_rejoin_threshold"
                    ),
                    q_rejoin_index=_safe_info_get(safety_info, "q_rejoin_index"),
                    qd_rejoin_loss=_safe_info_get(safety_info, "qd_rejoin_loss"),
                    qd_rejoin_dist=_safe_info_get(safety_info, "qd_rejoin_dist"),
                    qd_rejoin_threshold=_safe_info_get(
                        safety_info, "qd_rejoin_threshold"
                    ),
                    qd_rejoin_index=_safe_info_get(safety_info, "qd_rejoin_index"),
                    ee_rejoin_loss=_safe_info_get(safety_info, "ee_rejoin_loss"),
                    ee_rejoin_dist=_safe_info_get(safety_info, "ee_rejoin_dist"),
                    ee_rejoin_threshold=_safe_info_get(
                        safety_info, "ee_rejoin_threshold"
                    ),
                    ee_rejoin_index=_safe_info_get(safety_info, "ee_rejoin_index"),
                    ee_final_check_available=_safe_info_get(
                        safety_info, "ee_final_check_available"
                    ),
                    inner_rejoin_metric=_safe_info_get(
                        safety_info, "inner_rejoin_metric"
                    ),
                    final_rejoin_metric=_safe_info_get(
                        safety_info, "final_rejoin_metric"
                    ),
                    rejoin_q_eval_time_ms=_safe_info_get(
                        safety_info, "rejoin_q_eval_time_ms"
                    ),
                    rejoin_qd_eval_time_ms=_safe_info_get(
                        safety_info, "rejoin_qd_eval_time_ms"
                    ),
                    ee_nom_cache_time_ms=_safe_info_get(
                        safety_info, "ee_nom_cache_time_ms"
                    ),
                    ee_final_check_time_ms=_safe_info_get(
                        safety_info, "ee_final_check_time_ms"
                    ),
                    existing_optimization_loss=_safe_info_get(
                        safety_info, "existing_optimization_loss"
                    ),
                    smoothness_loss=_safe_info_get(safety_info, "smoothness_loss"),
                    total_loss=_safe_info_get(safety_info, "total_loss"),
                    fallback_used=_safe_info_get(safety_info, "fallback_used"),
                    act_resume_index=_safe_info_get(safety_info, "act_resume_index"),
                    act_resume_supported=_safe_info_get(
                        safety_info, "act_resume_supported"
                    ),
                    committed_chunk_active=_safe_info_get(safety_info, "committed_chunk_active"),
                    committed_chunk_mode=_safe_info_get(safety_info, "committed_chunk_mode"),
                    committed_chunk_index=_safe_info_get(safety_info, "committed_chunk_index"),
                    committed_chunk_length=_safe_info_get(safety_info, "committed_chunk_length"),
                    committed_rejoin_index=_safe_info_get(safety_info, "committed_rejoin_index"),
                    committed_chunk_started=_safe_info_get(safety_info, "committed_chunk_started"),
                    committed_chunk_completed=_safe_info_get(safety_info, "committed_chunk_completed"),
                    committed_aborted_due_to_safety=_safe_info_get(safety_info, "committed_aborted_due_to_safety"),
                    committed_repaired_step=_safe_info_get(safety_info, "committed_repaired_step"),
                    committed_repair_min_clearance=_safe_info_get(safety_info, "committed_repair_min_clearance"),
                    committed_repair_clearance_gain=_safe_info_get(safety_info, "committed_repair_clearance_gain"),
                    recover_steps_executed=_safe_info_get(safety_info, "recover_steps_executed", _safe_info_get(safety_info, "return_steps_executed")),
                    deform_steps_executed=_safe_info_get(safety_info, "deform_steps_executed", _safe_info_get(safety_info, "yield_steps_executed")),
                    resume_from_committed_rejoin=_safe_info_get(safety_info, "resume_from_committed_rejoin"),
                    request_action_history_reset_after_recovery=_safe_info_get(safety_info, "request_action_history_reset_after_recovery"),
                    recovery_action_history_reset=_safe_info_get(safety_info, "recovery_action_history_reset"),
                    recovery_action_history_reset_count=_safe_info_get(safety_info, "recovery_action_history_reset_count"),
                    committed_abort_step=_safe_info_get(safety_info, "committed_abort_step"),
                    committed_abort_mode=_safe_info_get(safety_info, "committed_abort_mode"),
                    committed_abort_index=_safe_info_get(safety_info, "committed_abort_index"),
                    committed_abort_chunk_length=_safe_info_get(safety_info, "committed_abort_chunk_length"),
                    committed_abort_action=_safe_info_get(safety_info, "committed_abort_action"),
                    committed_abort_min_clearance=_safe_info_get(safety_info, "committed_abort_min_clearance"),
                    committed_abort_required_clearance=_safe_info_get(safety_info, "committed_abort_required_clearance"),
                    committed_abort_clearance_gap=_safe_info_get(safety_info, "committed_abort_clearance_gap"),
                    committed_abort_human_state=_safe_info_get(safety_info, "committed_abort_human_state"),
                    committed_abort_robot_q=_safe_info_get(safety_info, "committed_abort_robot_q"),
                    committed_abort_robot_qd=_safe_info_get(safety_info, "committed_abort_robot_qd"),
                    committed_abort_reason=_safe_info_get(safety_info, "committed_abort_reason"),
                    planned_min_clearance_at_index=_safe_info_get(safety_info, "planned_min_clearance_at_index"),
                    planned_h_at_index=_safe_info_get(safety_info, "planned_h_at_index"),
                    planned_q_at_index=_safe_info_get(safety_info, "planned_q_at_index"),
                    planned_action_at_index=_safe_info_get(safety_info, "planned_action_at_index"),
                    planned_vs_actual_q_error=_safe_info_get(safety_info, "planned_vs_actual_q_error"),
                    planned_vs_actual_action_error=_safe_info_get(safety_info, "planned_vs_actual_action_error"),
                    actual_one_step_clearance=_safe_info_get(safety_info, "actual_one_step_clearance"),
                    planned_clearance_for_this_index=_safe_info_get(safety_info, "planned_clearance_for_this_index"),
                    clearance_prediction_error=_safe_info_get(safety_info, "clearance_prediction_error"),
                    planned_pre_action_q=_safe_info_get(safety_info, "planned_pre_action_q"),
                    planned_post_action_q=_safe_info_get(safety_info, "planned_post_action_q"),
                    predicted_post_action_q=_safe_info_get(safety_info, "predicted_post_action_q"),
                    actual_pre_action_q=_safe_info_get(safety_info, "actual_pre_action_q"),
                    replay_predicted_post_action_q=_safe_info_get(safety_info, "replay_predicted_post_action_q"),
                    committed_action=_safe_info_get(safety_info, "committed_action"),
                    planned_clearance_pre=_safe_info_get(safety_info, "planned_clearance_pre"),
                    planned_clearance_post=_safe_info_get(safety_info, "planned_clearance_post"),
                    replay_clearance_pre=_safe_info_get(safety_info, "replay_clearance_pre"),
                    replay_clearance_post=_safe_info_get(safety_info, "replay_clearance_post"),
                    actual_vs_planned_pre_q_error=_safe_info_get(safety_info, "actual_vs_planned_pre_q_error"),
                    actual_vs_planned_post_q_error=_safe_info_get(safety_info, "actual_vs_planned_post_q_error"),
                    planning_vs_replay_human_error=_safe_info_get(safety_info, "planning_vs_replay_human_error"),
                    planning_vs_replay_clearance_pre_error=_safe_info_get(safety_info, "planning_vs_replay_clearance_pre_error"),
                    planning_vs_replay_clearance_post_error=_safe_info_get(safety_info, "planning_vs_replay_clearance_post_error"),
                    planning_human_state_snapshot=_safe_info_get(safety_info, "planning_human_state_snapshot"),
                    replay_human_state=_safe_info_get(safety_info, "replay_human_state"),
                    control_type=_safe_info_get(safety_info, "control_type"),
                    dt=_safe_info_get(safety_info, "dt"),
                    controlled_state_indices=_safe_info_get(safety_info, "controlled_state_indices"),
                    controlled_action_indices=_safe_info_get(safety_info, "controlled_action_indices"),
                    action_conversion_mode=_safe_info_get(safety_info, "action_conversion_mode"),
                    human_motion_since_plan=_safe_info_get(safety_info, "human_motion_since_plan"),
                    accepted_min_clearance=_safe_info_get(safety_info, "accepted_min_clearance"),
                    accepted_clearance_margin=_safe_info_get(safety_info, "accepted_clearance_margin"),
                    committed_abort_due_to_human_motion=_safe_info_get(safety_info, "committed_abort_due_to_human_motion"),
                    committed_abort_due_to_prediction_error=_safe_info_get(safety_info, "committed_abort_due_to_prediction_error"),
                    committed_abort_due_to_safety_semantics_mismatch=_safe_info_get(safety_info, "committed_abort_due_to_safety_semantics_mismatch"),
                    committed_state_error=_safe_info_get(safety_info, "committed_state_error"),
                    committed_state_error_threshold=_safe_info_get(safety_info, "committed_state_error_threshold"),
                    committed_aborted_due_to_state_mismatch=_safe_info_get(safety_info, "committed_aborted_due_to_state_mismatch"),
                    committed_replan_due_to_state_mismatch=_safe_info_get(safety_info, "committed_replan_due_to_state_mismatch"),
                    committed_rejected_missing_planned_q=_safe_info_get(safety_info, "committed_rejected_missing_planned_q"),
                    actual_q_at_replay=_safe_info_get(safety_info, "actual_q_at_replay"),
                    diagnostic_step_mode=diagnostic_flags["diagnostic_step_mode"],
                    mode_transition=mode_transition,
                    act_step=diagnostic_flags["act_step"],
                    deform_step=diagnostic_flags["deform_step"],
                    recover_step=diagnostic_flags["recover_step"],
                    brake_step=diagnostic_flags["brake_step"],
                    fallback_step=diagnostic_flags["fallback_step"],
                    optimized_attempt_step=diagnostic_flags["optimized_attempt_step"],
                    optimized_accepted_step=diagnostic_flags["optimized_accepted_step"],
                    unsafe_streak=_safe_info_get(safety_info, "unsafe_streak"),
                    brake_streak=_safe_info_get(safety_info, "brake_streak"),
                    recovery_failure_streak=_safe_info_get(safety_info, "recovery_failure_streak"),
                    recovery_failure_streak_max=_safe_info_get(safety_info, "recovery_failure_streak_max"),
                    temporary_blocker_waiting=_safe_info_get(safety_info, "temporary_blocker_waiting"),
                    deform_trigger_reason=_safe_info_get(safety_info, "deform_trigger_reason"),
                    nominal_became_safe_after_brake=_safe_info_get(safety_info, "nominal_became_safe_after_brake"),
                    resume_act_after_wait=_safe_info_get(safety_info, "resume_act_after_wait"),
                    temporary_wait_step=_safe_info_get(safety_info, "temporary_wait_step"),
                    deform_suppressed_by_temporary_wait=_safe_info_get(safety_info, "deform_suppressed_by_temporary_wait"),
                    deform_after_persistent_block=_safe_info_get(safety_info, "deform_after_persistent_block"),
                    deform_replan_count=_safe_info_get(safety_info, "deform_replan_count"),
                    recover_replan_count=_safe_info_get(
                        safety_info,
                        "recover_replan_count",
                        _safe_info_get(safety_info, "recovery_replan_count"),
                    ),
                    recovery_replan_count=_safe_info_get(safety_info, "recovery_replan_count"),
                    recovery_target_feasible=_safe_info_get(safety_info, "recovery_target_feasible"),
                    stale_recovery_attempted=_safe_info_get(safety_info, "stale_recovery_attempted"),
                    stale_recovery_suppressed_count=_safe_info_get(safety_info, "stale_recovery_suppressed_count"),
                    recovery_target_infeasible_count=_safe_info_get(safety_info, "recovery_target_infeasible_count"),
                    recover_to_task_progress=_safe_info_get(safety_info, "recover_to_task_progress"),
                    recover_anchor_is_current=_safe_info_get(safety_info, "recover_anchor_is_current"),
                    deform_anchor_is_current=_safe_info_get(safety_info, "deform_anchor_is_current"),
                    emergency_brake_steps=_safe_info_get(safety_info, "emergency_brake_steps"),
                    emergency_brake_immediate_unsafe=_safe_info_get(safety_info, "emergency_brake_immediate_unsafe"),
                    optimized_candidate_count=_safe_info_get(safety_info, "optimized_candidate_count"),
                    optimized_solution_count=_safe_info_get(safety_info, "optimized_solution_count"),
                    fallback_candidate_count=_safe_info_get(safety_info, "fallback_candidate_count"),
                    fallback_candidate_accepted_count=_safe_info_get(safety_info, "fallback_candidate_accepted_count"),
                    candidate_fallback_enabled=_safe_info_get(safety_info, "candidate_fallback_enabled"),
                    optimized_rejected_count=_safe_info_get(safety_info, "optimized_rejected_count"),
                    deform_candidate_count=_safe_info_get(safety_info, "deform_candidate_count"),
                    deform_accepted_count=_safe_info_get(safety_info, "deform_accepted_count"),
                    deform_rejected_count=_safe_info_get(safety_info, "deform_rejected_count"),
                    recover_candidate_count=_safe_info_get(safety_info, "recover_candidate_count"),
                    recover_accepted_count=_safe_info_get(safety_info, "recover_accepted_count"),
                    recover_rejected_count=_safe_info_get(safety_info, "recover_rejected_count"),
                    safe_corridor_recovery_count=_safe_info_get(safety_info, "safe_corridor_recovery_count"),
                    direct_rejoin_attempt_count=_safe_info_get(safety_info, "direct_rejoin_attempt_count"),
                    direct_rejoin_reject_count=_safe_info_get(safety_info, "direct_rejoin_reject_count"),
                    detour_rejoin_attempt_count=_safe_info_get(safety_info, "detour_rejoin_attempt_count"),
                    detour_rejoin_accept_count=_safe_info_get(safety_info, "detour_rejoin_accept_count"),
                    delayed_rejoin_count=_safe_info_get(safety_info, "delayed_rejoin_count"),
                    recover_path_unsafe_count=_safe_info_get(safety_info, "recover_path_unsafe_count"),
                    recovery_path_failure_streak_max=_safe_info_get(safety_info, "recovery_path_failure_streak_max"),
                    repeated_unsafe_target_count=_safe_info_get(safety_info, "repeated_unsafe_target_count"),
                    post_recovery_act_window_count=_safe_info_get(safety_info, "post_recovery_act_window_count"),
                    post_recovery_act_window_interrupted_count=_safe_info_get(safety_info, "post_recovery_act_window_interrupted_count"),
                    mean_recover_path_min_clearance=_safe_info_get(safety_info, "mean_recover_path_min_clearance"),
                    min_recover_path_min_clearance=_safe_info_get(safety_info, "min_recover_path_min_clearance"),
                    safe_prefix_accepted_count=_safe_info_get(safety_info, "safe_prefix_accepted_count"),
                    first_action_only_accepted_count=_safe_info_get(safety_info, "first_action_only_accepted_count"),
                    immediate_hard_reject_count=_safe_info_get(safety_info, "immediate_hard_reject_count"),
                    no_safe_prefix_reject_count=_safe_info_get(safety_info, "no_safe_prefix_reject_count"),
                    horizon_margin_reject_count=_safe_info_get(safety_info, "horizon_margin_reject_count"),
                    accepted_deform_steps=_safe_info_get(safety_info, "accepted_deform_steps"),
                    accepted_recover_steps=_safe_info_get(safety_info, "accepted_recover_steps"),
                    fallback_brake_after_reject_count=_safe_info_get(safety_info, "fallback_brake_after_reject_count"),
                    accepted_candidate_type=_safe_info_get(safety_info, "accepted_candidate_type"),
                    accepted_candidate_name=_safe_info_get(safety_info, "accepted_candidate_name"),
                    acceptance_type=_safe_info_get(safety_info, "acceptance_type"),
                    safe_prefix_len=_safe_info_get(safety_info, "safe_prefix_len"),
                    immediate_clearance=_safe_info_get(safety_info, "immediate_clearance"),
                    prefix_min_clearance=_safe_info_get(safety_info, "prefix_min_clearance"),
                    horizon_min_clearance=_safe_info_get(safety_info, "horizon_min_clearance"),
                    full_horizon_required=_safe_info_get(safety_info, "full_horizon_required"),
                    rolling_replan_on_prefix=_safe_info_get(safety_info, "rolling_replan_on_prefix"),
                    safe_prefix_execution=_safe_info_get(safety_info, "safe_prefix_execution"),
                    recover_projection_on_nominal=_safe_info_get(safety_info, "recover_projection_on_nominal"),
                    recover_cosine_to_nominal=_safe_info_get(safety_info, "recover_cosine_to_nominal"),
                    nominal_rejoin_score=_safe_info_get(safety_info, "nominal_rejoin_score"),
                    nominal_rejoin_available=_safe_info_get(safety_info, "nominal_rejoin_available"),
                    nominal_rejoin_suppressed_reason=_safe_info_get(safety_info, "nominal_rejoin_suppressed_reason"),
                    nominal_rejoin_clearance=_safe_info_get(safety_info, "nominal_rejoin_clearance"),
                    nominal_rejoin_safe_prefix_len=_safe_info_get(safety_info, "nominal_rejoin_safe_prefix_len"),
                    recover_task_progress_score=_safe_info_get(safety_info, "recover_task_progress_score"),
                    recover_score_total=_safe_info_get(safety_info, "recover_score_total"),
                    recover_rejoin_weight_effective=_safe_info_get(safety_info, "recover_rejoin_weight_effective"),
                    recover_step_since_deform=_safe_info_get(safety_info, "recover_step_since_deform"),
                    nominal_rejoin_available_count=_safe_info_get(safety_info, "nominal_rejoin_available_count"),
                    nominal_rejoin_suppressed_count=_safe_info_get(safety_info, "nominal_rejoin_suppressed_count"),
                    stale_nominal_rejoin_suppressed_count=_safe_info_get(safety_info, "stale_nominal_rejoin_suppressed_count"),
                    nominal_prefix_unsafe_suppressed_count=_safe_info_get(safety_info, "nominal_prefix_unsafe_suppressed_count"),
                    recover_positive_projection_count=_safe_info_get(safety_info, "recover_positive_projection_count"),
                    recover_nonpositive_projection_count=_safe_info_get(safety_info, "recover_nonpositive_projection_count"),
                    mean_recover_projection_on_nominal=_safe_info_get(safety_info, "mean_recover_projection_on_nominal"),
                    mean_recover_cosine_to_nominal=_safe_info_get(safety_info, "mean_recover_cosine_to_nominal"),
                    mean_recover_task_progress_score=_safe_info_get(safety_info, "mean_recover_task_progress_score"),
                    contact_during_hold=_safe_info_get(safety_info, "contact_during_hold"),
                    contact_during_brake=_safe_info_get(safety_info, "contact_during_brake"),
                    contact_during_deform=_safe_info_get(safety_info, "contact_during_deform"),
                    contact_during_recover=_safe_info_get(safety_info, "contact_during_recover"),
                    chosen_action_norm=_safe_info_get(safety_info, "chosen_action_norm"),
                    controlled_action_delta_norm=_safe_info_get(safety_info, "controlled_action_delta_norm"),
                    arm_delta_norm=_safe_info_get(safety_info, "arm_delta_norm"),
                    gripper_latched=_safe_info_get(safety_info, "gripper_latched"),
                    gripper_latch_dim=_safe_info_get(safety_info, "gripper_latch_dim"),
                    safe_gripper_action=_safe_info_get(safety_info, "safe_gripper_action"),
                    raw_gripper_action=_safe_info_get(safety_info, "raw_gripper_action"),
                    phase_reanchor_steps_left=_safe_info_get(safety_info, "phase_reanchor_steps_left"),
                    phase_reanchor_base_cmd_xy=_safe_info_get(safety_info, "phase_reanchor_base_cmd_xy"),
                    phase_reanchor_ee_error_xy=_safe_info_get(safety_info, "phase_reanchor_ee_error_xy"),
                    phase_reanchor_drawer_fraction=_safe_info_get(safety_info, "phase_reanchor_drawer_fraction"),
                    phase_reanchor_ee_to_handle_dist=_safe_info_get(safety_info, "phase_reanchor_ee_to_handle_dist"),
                    post_recovery_task_guard_active=_safe_info_get(safety_info, "post_recovery_task_guard_active"),
                    post_recovery_task_guard_steps_left=_safe_info_get(safety_info, "post_recovery_task_guard_steps_left"),
                    post_recovery_task_guard_reason=_safe_info_get(safety_info, "post_recovery_task_guard_reason"),
                    post_recovery_task_guard_best_progress=_safe_info_get(safety_info, "post_recovery_task_guard_best_progress"),
                    post_recovery_progress_regression=_safe_info_get(safety_info, "post_recovery_progress_regression"),
                    post_recovery_reanchor_started=_safe_info_get(safety_info, "post_recovery_reanchor_started"),
                    hold_immediate_clearance=_safe_info_get(safety_info, "hold_immediate_clearance"),
                    hold_horizon_min_clearance=_safe_info_get(safety_info, "hold_horizon_min_clearance"),
                    hold_acceptance_type=_safe_info_get(safety_info, "hold_acceptance_type"),
                    hold_rejected_reason=_safe_info_get(safety_info, "hold_rejected_reason"),
                    hold_predicted_contact=_safe_info_get(safety_info, "hold_predicted_contact"),
                    human_prediction_available=_safe_info_get(safety_info, "human_prediction_available"),
                    human_velocity_toward_robot=_safe_info_get(safety_info, "human_velocity_toward_robot"),
                    human_motion_prediction_enabled=_safe_info_get(safety_info, "human_motion_prediction_enabled"),
                    human_motion_prediction_available=_safe_info_get(safety_info, "human_motion_prediction_available"),
                    human_motion_prediction_speed=_safe_info_get(safety_info, "human_motion_prediction_speed"),
                    human_motion_prediction_max_displacement=_safe_info_get(safety_info, "human_motion_prediction_max_displacement"),
                    emergency_deform_away=_safe_info_get(safety_info, "emergency_deform_away"),
                    emergency_deform_away_steps=_safe_info_get(safety_info, "emergency_deform_away_steps"),
                    emergency_deform_away_count=_safe_info_get(safety_info, "emergency_deform_away_count"),
                    hold_unsafe_count=_safe_info_get(safety_info, "hold_unsafe_count"),
                    hold_predicted_contact_count=_safe_info_get(safety_info, "hold_predicted_contact_count"),
                    contact_during_hold_count=_safe_info_get(safety_info, "contact_during_hold_count"),
                    contact_during_brake_count=_safe_info_get(safety_info, "contact_during_brake_count"),
                    contact_during_deform_count=_safe_info_get(safety_info, "contact_during_deform_count"),
                    contact_during_recover_count=_safe_info_get(safety_info, "contact_during_recover_count"),
                    mean_hold_horizon_min_clearance=_safe_info_get(safety_info, "mean_hold_horizon_min_clearance"),
                    min_hold_horizon_min_clearance=_safe_info_get(safety_info, "min_hold_horizon_min_clearance"),

                    elapsed_wall_time_s=float(elapsed_wall_time_s),
                    step_wall_time_s=float(step_wall_time_s),

                    filter_time_ms=float(filter_time_ms),
                    monitor_time_ms=float(monitor_time_ms),
                )

                episode_metrics.append(step_metrics)
                all_step_metrics.append(step_metrics)

                video_duration_s = _video_duration_seconds(video_recorder)
                video_recorded_steps = _video_recorded_steps(video_recorder)
                video_left_s = None
                video_left_steps = None
                if args.record_video and args.video_time_base == "sim" and video_stop_steps is not None:
                    video_left_steps = max(0, video_stop_steps - video_recorded_steps)
                elif args.record_video and args.stop_video_at_seconds is not None:
                    video_left_s = max(0.0, args.stop_video_at_seconds - video_duration_s)

                if progress_bar is not None:
                    progress_bar.update(1)
                    postfix = {"steps_left": args.steps - progress_bar.n}
                    if video_left_steps is not None:
                        postfix["video_steps_left"] = video_left_steps
                    elif video_left_s is not None:
                        postfix["video_left"] = f"{video_left_s:.1f}s"
                    progress_bar.set_postfix(postfix)
                elif args.debug:
                    print(
                        f"ep={episode:03d} step={step:04d} "
                        f"reward={float(reward):.3f} "
                        f"min_h={min_h} "
                        f"arm_delta={arm_delta:.5f} "
                        f"non_arm_delta={non_arm_delta:.5f} "
                        f"gripper_latched={gripper_latched} "
                        f"contact_count={contact_count} "
                        f"filter_ms={filter_time_ms:.2f}"
                    )

                if args.video_time_base == "sim":
                    reached_video_limit = (
                        args.record_video
                        and video_stop_steps is not None
                        and video_recorded_steps >= video_stop_steps
                    )
                else:
                    reached_video_limit = (
                        args.record_video
                        and args.stop_video_at_seconds is not None
                        and video_duration_s >= args.stop_video_at_seconds
                    )
                if reached_video_limit:
                    if args.video_time_base == "sim":
                        episode_stop_reason = f"video_step_limit:{video_recorded_steps}"
                        print(
                            f"Stopping episode {episode} at "
                            f"{video_recorded_steps} recorded env steps "
                            f"(target {video_stop_steps})."
                        )
                    else:
                        episode_stop_reason = f"video_wall_limit:{video_duration_s:.3f}"
                        print(
                            f"Stopping episode {episode} at "
                            f"{video_duration_s:.1f}s of recorded video "
                            f"(target {args.stop_video_at_seconds:.1f}s)."
                        )
                    break
                if unmodelled_contact_reason is not None:
                    episode_stop_reason = unmodelled_contact_reason
                    if episode == 0 or args.debug:
                        print(
                            f"Stopping episode {episode} at step {step}: "
                            f"{unmodelled_contact_reason}"
                        )
                    break
                if terminated or truncated or (success and not args.continue_after_success):
                    if terminated:
                        episode_stop_reason = "terminated"
                    elif truncated:
                        episode_stop_reason = "truncated"
                    elif success and not args.continue_after_success:
                        episode_stop_reason = "success"
                    break

            saved_action_episodes.append(
                np.asarray(saved_episode_actions, dtype=np.float32)
            )

            episode_summary = summarise_chunk_episode(
                episode_metrics,
                diagnostics_cfg={
                    "large_arm_delta_threshold": args.diagnostics_large_arm_delta_threshold,
                    "large_base_delta_threshold": args.diagnostics_large_base_delta_threshold,
                    "low_act_ratio_threshold": args.diagnostics_low_act_ratio_threshold,
                    "high_fallback_ratio_threshold": args.diagnostics_high_fallback_ratio_threshold,
                    "success_threshold": args.phase_reanchor_done_threshold,
                },
            )
            if episode_metrics:
                wall_time_s = episode_metrics[-1].elapsed_wall_time_s
                step_wall_times = np.asarray(
                    [m.step_wall_time_s for m in episode_metrics],
                    dtype=np.float32,
                )
                episode_summary["wall_time_s"] = float(wall_time_s)
                episode_summary["mean_step_wall_time_s"] = float(
                    np.mean(step_wall_times)
                )
                episode_summary["steps_per_wall_second"] = float(
                    len(episode_metrics) / max(wall_time_s, 1e-9)
                )
                video_recorded_duration_s = float(_video_duration_seconds(video_recorder))
                episode_summary["video_recorded_duration_s"] = video_recorded_duration_s
                episode_summary["video_recorded_wall_time_s"] = video_recorded_duration_s
                episode_summary["video_recorded_steps"] = int(
                    _video_recorded_steps(video_recorder)
                )
            episode_summary["stop_reason"] = episode_stop_reason
            episode_summary["video_stop_steps"] = video_stop_steps
            episode_summary["normalization_source"] = normalization_source
            episode_summary["robot_spawn"] = robot_spawn_info
            episode_summary["policy_robot_spawn"] = policy_robot_spawn_info
            episode_summary["safety_robot_spawn"] = safety_robot_spawn_info
            episode_summary["execution_length"] = workspace_cfg.get("execution_length", None)
            episode_summary["action_sequence"] = workspace_cfg.get("action_sequence", None)
            episode_summary["video_time_base"] = args.video_time_base
            if trajectory_logging_enabled:
                episode_summary["chunk_trajectory_trace_events"] = int(
                    len(episode_chunk_trajectory_records)
                )
                episode_summary["human_arm_trajectory_samples"] = int(
                    len(episode_human_arm_trajectory_samples)
                )
                if args.plot_chunk_trajectories_3d and (
                    episode_chunk_trajectory_records
                    or episode_human_arm_trajectory_samples
                ):
                    plot_path = trajectory_plot_dir / (
                        f"{args.condition}_episode_{episode:03d}_trajectories_3d.png"
                    )
                    saved_plot = _save_chunk_trajectory_plot(
                        plot_path,
                        episode,
                        episode_chunk_trajectory_records,
                        episode_human_arm_trajectory_samples,
                        args.chunk_trajectory_plot_max_events,
                    )
                    if saved_plot is not None:
                        episode_summary["chunk_trajectory_plot_3d"] = saved_plot
                        trajectory_plot_paths.append(saved_plot)
            all_episode_summaries.append(episode_summary)

            if progress_bar is not None:
                progress_bar.close()

            print("\nEpisode summary:")
            for key, value in episode_summary.items():
                print(f"  {key}: {value}")

            if args.plot_terminal:
                _plot_episode_metrics(episode, episode_metrics)

            video_recorder.save(f"{args.condition}_episode_{episode:03d}.mp4")
            if args.record_policy_video:
                policy_video_path = video_dir / f"{args.condition}_policy_obs_episode_{episode:03d}.mp4"
                _save_policy_obs_video(
                    policy_video_frames,
                    policy_video_timestamps,
                    policy_video_path,
                )
                print("  policy obs video:", policy_video_path)

            if episode_bar is not None:
                episode_bar.update(1)
                episode_bar.set_postfix(episodes_left=args.episodes - episode_bar.n)

    finally:
        if episode_bar is not None:
            episode_bar.close()
        if policy_env is not None:
            policy_env.close()
        if safety_env is not None:
            safety_env.close()
        env.close()

    final_summary = summarise_all_chunk_episodes(all_episode_summaries)
    final_summary["normalization_source"] = normalization_source
    final_summary["robot_spawn"] = robot_spawn_info
    final_summary["policy_robot_spawn"] = policy_robot_spawn_info
    final_summary["safety_robot_spawn"] = safety_robot_spawn_info
    final_summary["execution_length"] = workspace_cfg.get("execution_length", None)
    final_summary["action_sequence"] = workspace_cfg.get("action_sequence", None)
    final_summary["video_time_base"] = args.video_time_base
    final_summary["video_stop_steps"] = video_stop_steps
    if trajectory_logging_enabled:
        final_summary["chunk_trajectory_trace_events"] = int(
            len(all_chunk_trajectory_records)
        )
        final_summary["human_arm_trajectory_samples"] = int(
            len(all_human_arm_trajectory_samples)
        )
        final_summary["chunk_trajectory_trace_jsonl"] = str(chunk_trajectory_jsonl_path)
        final_summary["human_arm_trajectory_jsonl"] = str(human_arm_trajectory_jsonl_path)
        final_summary["chunk_trajectory_plot_count"] = int(len(trajectory_plot_paths))
        final_summary["chunk_trajectory_plots"] = list(trajectory_plot_paths)
        final_summary["chunk_trajectory_include_q_states"] = bool(
            args.chunk_trajectory_include_q_states
        )

    if args.save_actions is not None:
        save_actions_path = Path(args.save_actions)
        if not save_actions_path.is_absolute():
            save_actions_path = output_root / save_actions_path
        save_actions_path.parent.mkdir(parents=True, exist_ok=True)

        if saved_action_episodes:
            min_steps = min(actions.shape[0] for actions in saved_action_episodes)
            actions_to_save = np.stack(
                [actions[:min_steps] for actions in saved_action_episodes],
                axis=0,
            ).astype(np.float32)
        else:
            actions_to_save = np.empty((0, 0) + env_action_shape, dtype=np.float32)

        np.savez_compressed(
            save_actions_path,
            actions=actions_to_save,
            env_action_shape=np.asarray(env_action_shape, dtype=np.int64),
            normalization_source=np.asarray(str(normalization_source)),
        )
        final_summary["saved_actions"] = str(save_actions_path)

    if trajectory_logging_enabled:
        with chunk_trajectory_jsonl_path.open("w") as f:
            for record in all_chunk_trajectory_records:
                f.write(json.dumps(_jsonable_trace_value(record)) + "\n")
        with human_arm_trajectory_jsonl_path.open("w") as f:
            for sample in all_human_arm_trajectory_samples:
                f.write(json.dumps(_jsonable_trace_value(sample)) + "\n")

    with step_jsonl_path.open("w") as f:
        for metric in all_step_metrics:
            f.write(json.dumps(asdict(metric)) + "\n")

    with episode_summary_path.open("w") as f:
        json.dump(all_episode_summaries, f, indent=2)

    with final_summary_path.open("w") as f:
        json.dump(final_summary, f, indent=2)

    if final_summary.get("diagnostic_warning"):
        print(f"WARNING: {final_summary['diagnostic_warning']}")

    print("\n========== Final summary ==========")
    for key, value in final_summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print("  step metrics:", step_jsonl_path)
    print("  episode summaries:", episode_summary_path)
    print("  final summary:", final_summary_path)
    if trajectory_logging_enabled:
        print("  chunk trajectory traces:", chunk_trajectory_jsonl_path)
        print("  human arm trajectory:", human_arm_trajectory_jsonl_path)
        if trajectory_plot_paths:
            print("  trajectory plots:", trajectory_plot_dir)
    if args.save_actions is not None:
        print("  saved actions:", final_summary["saved_actions"])


if __name__ == "__main__":
    main()
