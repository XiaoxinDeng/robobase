from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


REPO = Path("/home/xd1125/Workspace/safe_bigym_hoi")

ROBOBASE_CFG = REPO / "external/robobase/robobase/cfgs"

EVAL_SCENARIO_CONFIG_DIR = ROBOBASE_CFG / "eval_scenarios"

SAFETY_FILTER_CONFIG_DIR = ROBOBASE_CFG / "safety_filter"

PATH_CONSISTENT_BRAKE_FILTER_CONFIG = (
    ROBOBASE_CFG / "safety_filter" / "path_consistent_brake.yaml"
)

PATH_CONSISTENT_BRAKE_CONFIG_KEYS = (
    "waypoint_substeps",
    "max_waypoint_delta",
    "slowdown_enabled",
    "slowdown_lookahead",
    "slowdown_min_scale",
    "certified_backup_enabled",
    "trajectory_generation_enabled",
    "trajectory_max_velocity",
    "trajectory_max_acceleration",
    "trajectory_max_jerk",
    "trajectory_initial_speed",
    "trajectory_backend",
    "trajectory_min_position",
    "trajectory_max_position",
    "shield_substeps",
    "inner_shield_verification_enabled",
    "skip_inner_shield_when_rejected",
    "reuse_operator_human_rollout_cache",
    "reachability_certification_enabled",
    "reachability_fail_closed",
    "reachability_robot_radius",
    "reachability_obstacle_radius",
    "reachability_robot_points_source",
    "reachability_inflation_enabled",
    "reachability_tracking_error",
    "reachability_measurement_error",
    "reachability_object_speed",
    "reachability_object_acceleration",
    "reachability_sensor_delay",
    "safety_constraint_type",
    "pfl_energy_threshold",
    "pfl_contact_margin",
    "pfl_joint_inertia",
    "pfl_energy_thresholds",
    "pfl_active_threshold_key",
)

PATH_CONSISTENT_BRAKE_LIMIT_KEYS = (
    "trajectory_max_velocity",
    "trajectory_max_acceleration",
    "trajectory_max_jerk",
)

IGNORED_EVAL_CONFIG_KEYS = {
    "notes",
    "command_notes",
    "temporary_blocker",
}

DEFAULT_SAFETY_FILTER_CONFIG = SAFETY_FILTER_CONFIG_DIR / "safechunk_deform.yaml"

DEFAULT_EVAL_ARGS: dict[str, Any] = {'ablation_force_planned_recovery_q': False,
 'ablation_force_planned_recovery_q_hold_current_step': True,
 'ablation_force_planned_recovery_q_mode': 'controlled',
 'ablation_force_planned_recovery_q_pure_act_resume_steps': 0,
 'ablation_force_planned_recovery_q_replay_sequence': False,
 'ablation_force_planned_recovery_q_window_mode': 'default',
 'ablation_force_planned_recovery_q_once_per_episode': True,
 'ablation_force_planned_recovery_q_reset_filter': True,
 'ablation_force_planned_recovery_q_sync_low_level_state': False,
 'ablation_force_planned_recovery_q_reset_history': False,
 'ablation_force_planned_recovery_q_seed_policy_window': False,
 'ablation_force_planned_recovery_q_source_mode': 'planned_terminal',
 'ablation_force_planned_recovery_q_window_interpolate': True,
 'ablation_force_planned_recovery_q_window_len': 4,
 'ablation_force_planned_recovery_q_trigger': 'accepted',
 'ablation_force_planned_recovery_q_zero_velocity': True,
 'brake_progress_threshold': 0.05,
 'chunk_acceptance_clearance_tol': 0.005,
 'chunk_acceptance_desired_min_clearance': 0.08,
 'chunk_acceptance_enabled': True,
 'chunk_acceptance_hard_min_clearance': 0.02,
 'chunk_acceptance_min_safe_prefix_len': 1,
 'chunk_acceptance_prefix_min_clearance': 0.04,
 'chunk_active_check_hold_horizon_safety': True,
 'chunk_active_emergency_deform_candidate_scales': [0.25, 0.5, 0.75, 1.0],
 'chunk_active_emergency_deform_replan_next_step': True,
 'chunk_active_emergency_deform_when_hold_unsafe': True,
 'chunk_active_hard_min_clearance': 0.02,
 'chunk_active_hold_horizon_steps': 4,
 'chunk_active_hold_prefix_min_clearance': 0.04,
 'chunk_active_optimize_when_hold_unsafe': True,
 'chunk_active_predict_human_motion_for_hold': True,
 'chunk_active_prefer_last_safe_action': True,
 'chunk_active_prefer_last_safe_q_retract': True,
 'chunk_active_safety_enabled': True,
 'chunk_allow_fallback_path': True,
 'chunk_allow_ordered_path_soft_fail_on_live_progress': True,
 'chunk_allow_safe_prefix_execution': True,
 'chunk_brake_if_unrecoverable': True,
 'chunk_cache_nominal_ee': True,
 'chunk_fallback_only_if_no_optimized_result': True,
 'chunk_commit_accepted_chunks': True,
 'chunk_committed_abort_only_if_contact_risk': True,
 'chunk_committed_chunk_safety_check': True,
 'chunk_committed_deform_min_clearance_for_abort': None,
 'chunk_committed_execution_margin': 0.0,
 'chunk_committed_min_clearance_for_abort': 0.08,
 'chunk_committed_safety_tol': 0.005,
 'chunk_committed_state_error_action': 'replan',
 'chunk_committed_state_error_threshold': 0.25,
 'chunk_committed_state_mismatch_abort_requires_unsafe': True,
 'chunk_debug_safety_feasibility': True,
 'chunk_deform_mode': 'optimized',
 'chunk_deformation_enabled': True,
 'chunk_deformation_scales': [0.0, 0.25, 0.5, 0.75],
 'chunk_deformation_smoothing': 1,
 'chunk_delayed_rejoin_requires_nominal_prefix_safe': True,
 'chunk_delayed_rejoin_wait_steps': 4,
 'chunk_detach_passthrough_dims': True,
 'chunk_detour_action_norm_weight': 0.2,
 'chunk_detour_clearance_weight': 100.0,
 'chunk_detour_scales': [0.25, 0.5, 0.75, 1.0],
 'chunk_detour_task_rejoin_weight': 10.0,
 'chunk_ee_rejoin_in_inner_loop': True,
 'chunk_ee_rejoin_threshold': 0.08,
 'chunk_emergency_brake_if_immediate_below_hard_margin': True,
 'chunk_enable_delayed_rejoin': True,
 'chunk_enable_detour_rejoin': True,
 'chunk_enable_direct_rejoin': True,
 'chunk_explicit_return': True,
 'chunk_extend_recovery_budget_on_progress': True,
 'chunk_final_rejoin_metric': 'q_state',
 'chunk_full_horizon_required_for_deform': False,
 'chunk_full_horizon_required_for_recover': False,
 'chunk_gradient_adam_beta1': 0.9,
 'chunk_gradient_adam_beta2': 0.999,
 'chunk_gradient_batched_line_search': True,
 'chunk_gradient_early_stop_on_path': True,
 'chunk_gradient_eps': None,
 'chunk_gradient_line_search_scales': [1.0, 0.5, 0.25],
 'chunk_gradient_min_improvement': 1e-06,
 'chunk_gradient_samples': 4,
 'chunk_horizon_predict_human_motion': True,
 'chunk_human_motion_prediction_max_speed': 3.0,
 'chunk_human_motion_prediction_max_time': 0.25,
 'chunk_inner_rejoin_metric': 'q_state',
 'chunk_lambda_action': 0.1,
 'chunk_lambda_path': 0.2,
 'chunk_lambda_rejoin': 0.5,
 'chunk_lambda_retreat': 1.0,
 'chunk_lambda_return_action': 0.1,
 'chunk_lambda_return_rejoin': 10.0,
 'chunk_lambda_return_safety': 500.0,
 'chunk_lambda_return_smooth': 0.2,
 'chunk_lambda_safety': 500.0,
 'chunk_lambda_smooth': 0.1,
 'chunk_lambda_deform_action': 0.1,
 'chunk_lambda_deform_safety': 800.0,
 'chunk_lambda_deform_smooth': 0.1,
 'chunk_max_recover_steps_before_act_resume': 16,
 'chunk_max_recover_steps_with_progress': 24,
 'chunk_max_recovery_failure_before_replan': 1,
 'chunk_max_return_retries': 3,
 'chunk_max_same_target_failures': 2,
 'chunk_min_clearance': 0.12,
 'chunk_min_rejoin_offset': 2,
 'chunk_monotonic_committed_repair': True,
 'chunk_mpc_recovery': True,
 'chunk_mpc_recovery_horizon': 8,
 'chunk_mpc_recovery_max_replans_per_recovery': 0,
 'chunk_mpc_recovery_min_progress_delta': 0.0001,
 'chunk_mpc_recovery_no_progress_limit': 3,
 'chunk_mpc_recovery_prefix_len': 2,
 'chunk_mpc_recovery_require_live_progress': True,
 'chunk_mpc_recovery_require_ordered_progress': True,
 'chunk_opt_elite_frac': 0.25,
 'chunk_opt_iters': 20,
 'chunk_opt_lr': 0.03,
 'chunk_opt_method': 'gradient',
 'chunk_opt_population': 32,
 'chunk_opt_seed': 0,
 'chunk_optimized_fallback': 'brake',
 'chunk_post_recovery_min_act_steps': 5,
 'chunk_q_rejoin_threshold': 0.25,
 'chunk_qd_rejoin_hard_threshold': 6.0,
 'chunk_qd_rejoin_threshold': 3.0,
 'chunk_recover_action_deviation_weight': 0.2,
 'chunk_recover_direction_alignment_margin': 0.0,
 'chunk_recover_direction_alignment_weight': 5.0,
 'chunk_recover_immediate_hard_clearance': 0.02,
 'chunk_recover_max_attempts_per_unsafe_streak': 3,
 'chunk_recover_min_direction_cosine': 0.05,
 'chunk_recover_nominal_rejoin_prefix_min_clearance': 0.04,
 'chunk_recover_ordered_delta_threshold': 0.002,
 'chunk_recover_ordered_delta_weight': 3.0,
 'chunk_recover_ordered_pose_threshold': 0.01,
 'chunk_recover_ordered_pose_weight': 5.0,
 'chunk_recover_path_min_clearance': 0.04,
 'chunk_recover_prefix_min_clearance': 0.04,
 'chunk_recover_rejoin_nominal_weight': 10.0,
 'chunk_recover_rejoin_ramp_steps': 5,
 'chunk_recover_rejoin_weight_schedule': 'ramp',
 'chunk_recover_require_nominal_prefix_safe': True,
 'chunk_recover_retry_cooldown_steps': 4,
 'chunk_recover_safety_weight': 100.0,
 'chunk_recover_smoothness_weight': 0.1,
 'chunk_recover_suppress_stale_nominal': True,
 'chunk_recover_task_progress_weight': 10.0,
 'chunk_recover_use_latest_nominal': True,
 'chunk_recoverable_deform_enabled': True,
 'chunk_recovery_budget_no_progress_limit': 2,
 'chunk_recovery_budget_progress_epsilon': 0.02,
 'chunk_recovery_corridor_enabled': True,
 'chunk_recovery_target_mode': 'task_progress',
 'chunk_rejoin_threshold': 0.03,
 'chunk_repair_committed_action': True,
 'chunk_require_post_recovery_act_window': True,
 'chunk_require_qd_rejoin': True,
 'chunk_require_recover_direction_alignment': True,
 'chunk_require_recover_ordered_path': True,
 'chunk_require_recover_path_safe': True,
 'chunk_require_safe_corridor_for_recovery_complete': True,
 'chunk_return_horizon': 8,
 'chunk_rolling_replan_on_prefix': True,
 'chunk_safechunk_recover_enabled': True,
 'chunk_safechunk_replan_enabled': True,
 'chunk_staged_recovery': True,
 'chunk_staged_recovery_min_progress_delta': 0.05,
 'chunk_suppress_repeated_unsafe_recovery': True,
 'chunk_temporary_blocker_enabled': True,
 'chunk_temporary_max_brake_steps_before_deform': 12,
 'chunk_temporary_min_progress_delta': 0.001,
 'chunk_temporary_min_unsafe_steps_before_deform': 8,
 'chunk_temporary_prefer_brake_before_deform': True,
 'chunk_temporary_progress_window': 10,
 'chunk_temporary_recover_after_wait': True,
 'chunk_temporary_recover_after_wait_min_brake_steps': 1,
 'chunk_temporary_require_progress_deadlock_before_deform': True,
 'chunk_temporary_reset_on_nominal_safe': True,
 'chunk_trajectory_include_q_states': True,
 'chunk_trajectory_max_events': 300,
 'chunk_trajectory_plot_max_events': 25,
 'chunk_unsafe_recovery_cooldown_steps': 8,
 'chunk_use_ee_final_check': True,
 'chunk_use_ee_pose_rejoin': True,
 'chunk_use_object_state_rejoin': False,
 'chunk_deform_horizon': 4,
 'condition': None,
 'continue_after_success': False,
 'deadlock_window': 5,
 'debug': False,
 'deform_immediately_on_deadlock': False,
 'demos': 1,
 'diagnostics_enabled': True,
 'diagnostics_high_fallback_ratio_threshold': 0.5,
 'diagnostics_large_arm_delta_threshold': 3.0,
 'diagnostics_large_base_delta_threshold': 0.5,
 'diagnostics_low_act_ratio_threshold': 0.3,
 'enable_human_arm_collisions': False,
 'env': 'bigym/human_arm_drawer_top_open',
 'episodes': 20,
 'episode_index_offset': 0,
 'eval_config': None,
 'frame_image_dir': None,
 'frame_image_every': 10,
 'freeze_human_arm': False,
 'gripper_latch': False,
 'gripper_latch_dim': -1,
 'gripper_latch_start_step': 0,
 'gripper_latch_trigger': 0.5,
 'gripper_latch_value': 1.0,
 'hide_human_arm_policy_obs': False,
 'horizon': 16,
 'human_arm_aggression': 1.0,
 'human_arm_disable_keepout': False,
 'human_arm_drawer_obstruction': False,
 'human_arm_drawer_obstruction_amp_xy': [0.03, 0.16],
 'human_arm_drawer_obstruction_xy': [-0.5, 0.2],
 'human_arm_ee_obstruction': False,
 'human_arm_ee_offset_xy': [0.0, 0.0],
 'human_arm_ee_side_sweep': False,
 'human_arm_ee_side_sweep_amp_xy': [0.03, 0.3],
 'human_arm_ee_side_sweep_frequency': 0.35,
 'human_arm_ee_side_sweep_phase': 2.0,
 'human_arm_final_clear_after_steps': -1,
 'human_arm_final_clear_carrier_xy': [-0.85, 0.55],
 'human_arm_final_clear_duration_steps': 20,
 'human_arm_final_clear_max_carrier_speed': 0.35,
 'human_arm_final_clear_max_joint_speed': 1.3,
 'human_arm_final_clear_trigger': 'carrier-y-peak',
 'human_arm_force_carrier_amp_xy': None,
 'human_arm_force_carrier_frequency': 0.35,
 'human_arm_force_carrier_xy': None,
 'human_arm_goal_xy': None,
 'human_arm_keepout_min_clear': None,
 'human_arm_natural_contact_motion': False,
 'human_arm_natural_lateral_scale': 0.35,
 'human_arm_natural_motion_frequency': 0.45,
 'human_arm_natural_motion_phase_offset_steps': 50.0,
 'human_arm_natural_return_curl_scale': 1.0,
 'human_arm_release_after_steps': 180,
 'human_arm_release_carrier_xy': [-0.85, 0.55],
 'human_arm_release_duration_steps': 60,
 'human_arm_substeps': 1,
 'human_arm_trajectory_stride': 1,
 'human_arm_transient_obstruction': False,
 'human_arm_walk_radius': None,
 'human_arm_yaw_offset_deg': 90.0,
 'human_arm_zero_dwell': False,
 'initial_pause_restart_steps': 0,
 'intervention_eps': 0.0001,
 'live_h_monitor': True,
 'log_chunk_trajectories': True,
 'log_mpc_replay_diagnostics': False,
 'mpc_replay_diagnostics_max_events': 300,
 'max_action_delta': None,
 'no_progress': False,
 'no_record_video': False,
 'normalization_source': 'auto',
 'oscbf_alpha_gain': 10.0,
 'oscbf_arm_weight': 1.0,
 'oscbf_human_margin': 0.08,
 'oscbf_pelvis_velocity_limits': [0.6, 0.6, 0.4, 1.5],
 'oscbf_pelvis_weight': 0.5,
 'out': 'eval/act_oscbf_metrics.jsonl',
 'output_dir': 'eval_safety',
 'override': [],
 'path_consistent_brake_config': None,
 'pause_and_restart_on_human_blocker': False,
 'pause_clearance_threshold': 0.0,
 'pause_motion_scale': 0.0,
 'pause_on_unsafe': False,
 'pause_policy_step_on_brake': True,
 'phase_reanchor': False,
 'phase_reanchor_base_gain': 0.45,
 'phase_reanchor_bypass_temporal_ensemble': False,
 'phase_reanchor_arm_servo': False,
 'phase_reanchor_arm_gain': 0.9,
 'phase_reanchor_arm_max_step': 0.08,
 'phase_reanchor_arm_servo_mix': 0.75,
 'phase_reanchor_live_taskspace_guard': True,
 'phase_reanchor_live_taskspace_worsen_tolerance': 0.005,
 'phase_reanchor_live_taskspace_worsen_patience': 1,
 'phase_reanchor_live_taskspace_guard_disable_steps': 6,
 'phase_reanchor_live_taskspace_stop_on_worsening': False,
 'phase_reanchor_live_taskspace_stop_min_steps': 12,
 'phase_reanchor_suppress_q_servo_far_target': False,
 'phase_reanchor_q_servo_enable_target_dist': 0.24,
 'phase_reanchor_release_requires_live_taskspace': True,
 'phase_reanchor_live_release_target_error': 0.16,
 'phase_reanchor_live_release_handle_dist': 0.24,
 'phase_reanchor_live_release_require_both': False,
 'phase_reanchor_bridge_requires_handle_proximity': False,
 'phase_reanchor_bridge_handle_dist': 0.24,
 'phase_reanchor_bridge_seed_mode': 'live_taskspace',
 'phase_reanchor_bridge_seed_obs_source': 'recovery',
 'phase_reanchor_bridge_policy_step_source': 'reset_zero',
 'phase_reanchor_bridge_reseed_action_history_with_post_seed_act': False,
 'phase_reanchor_bridge_requires_resume_affordance': True,
 'phase_reanchor_bridge_requires_post_seed_action_agreement': False,
 'phase_reanchor_bridge_action_agreement_l2': 1.0,
 'phase_reanchor_bridge_action_agreement_cosine': 0.8,
 'phase_reanchor_bridge_action_agreement_mode': 'and',
 'phase_reanchor_bridge_preload_validation': False,
 'phase_reanchor_bridge_preload_steps': 8,
 'phase_reanchor_bridge_preload_progress_delta': 0.0002,
 'phase_reanchor_bridge_preload_progress_min_abs': 0.0,
 'phase_reanchor_bridge_preload_handle_dist': 0.245,
 'phase_reanchor_bridge_preload_allow_handle_only': False,
 'phase_reanchor_bridge_preload_pull_probe': False,
 'phase_reanchor_bridge_preload_pull_step': 0.006,
 'phase_reanchor_force_gripper_during_preload': False,
 'phase_reanchor_preload_target_grasp': False,
 'phase_reanchor_live_extend_on_not_ready': False,
 'phase_reanchor_live_extend_steps': 16,
 'phase_reanchor_live_max_extensions': 3,
 'phase_reanchor_early_release_on_resumable_window': False,
 'phase_reanchor_early_release_min_steps': 8,
 'phase_reanchor_early_release_arm_q_error': 0.12,
 'phase_reanchor_early_release_act_grace_steps': 16,
 'phase_reanchor_live_ee_servo': False,
 'phase_reanchor_live_ee_servo_with_q_window': False,
 'phase_reanchor_live_ee_gain': 0.9,
 'phase_reanchor_live_ee_nominal_reg': 0.15,
 'phase_reanchor_live_ee_servo_mix': 1.0,
 'phase_reanchor_task_point_source': 'ee_site',
 'phase_reanchor_measurement_task_point_source': 'control',
 'phase_reanchor_control_error_source': 'control',
 'phase_reanchor_live_handle_assist': False,
 'phase_reanchor_live_handle_assist_trigger_target_dist': 0.22,
 'phase_reanchor_live_handle_assist_gain': 0.45,
 'phase_reanchor_live_handle_assist_max_base_step_xy': None,
 'phase_reanchor_near_live_target_suppress_q_servo': False,
 'phase_reanchor_near_live_target_suppress_dist': 0.22,
 'phase_reanchor_task_point_geometry_trust_error': 0.08,
 'phase_reanchor_live_ee_servo_min_predicted_improvement': 0.0,
 'phase_reanchor_hold_arm_when_q_servo_suppressed': False,
 'phase_reanchor_nominal_window_enabled': False,
 'phase_reanchor_nominal_window_track_base': True,
 'phase_reanchor_nominal_window_source': None,
 'phase_reanchor_nominal_window_len': 4,
 'phase_reanchor_nominal_window_lead_steps': 3,
 'phase_reanchor_nominal_window_pregrasp_lags': [],
 'phase_reanchor_nominal_window_selector': 'taskspace_aware',
 'phase_reanchor_nominal_window_stage': 'auto',
 'phase_reanchor_arm_fd_eps': 0.001,
 'phase_reanchor_arm_damping': 0.001,
 'phase_reanchor_arm_error_clip': 0.25,
 'phase_reanchor_wrist_servo_weight': 1.5,
 'phase_reanchor_check_after_steps': 70,
 'phase_reanchor_cooldown_steps': 50,
 'phase_reanchor_done_threshold': 0.9,
 'phase_reanchor_grasp_dist': 0.12,
 'phase_reanchor_grasp_offset_xy': [-0.03, 0.0],
 'phase_reanchor_gripper_closed_threshold': 0.5,
 'phase_reanchor_gripper_value': 1.0,
 'phase_reanchor_max_base_step': 0.012,
 'phase_reanchor_max_base_step_xy': None,
 'phase_reanchor_min_drawer_progress': 0.02,
 'phase_reanchor_no_progress_window': 20,
 'phase_reanchor_pregrasp_offset_xy': [-0.12, -0.06],
 'phase_reanchor_pull_offset_xy': [0.0, -0.1],
 'phase_reanchor_pull_open_threshold': 0.15,
 'phase_reanchor_steps': 24,
 'plot_chunk_trajectories_3d': True,
 'plot_terminal': False,
 'policy_env': None,
 'policy_video_every': 1,
 'post_recovery_progress_tolerance': 1e-05,
 'post_recovery_no_progress_reanchor': False,
 'post_recovery_no_progress_start_step': 90,
 'post_recovery_no_progress_patience': 8,
 'post_recovery_no_progress_max_progress': 0.04,
 'post_recovery_no_progress_min_target_distance': 0.62,
 'post_recovery_no_progress_distance_source': 'measurement',
 'post_recovery_mid_progress_no_progress_reanchor': False,
 'post_recovery_mid_progress_min_progress': 0.35,
 'post_recovery_mid_progress_patience': 8,
 'post_recovery_mid_progress_epsilon': 0.001,
 'post_recovery_mid_progress_distance_regression': 0.06,
 'post_recovery_mid_progress_min_target_distance': 0.42,
 'post_recovery_mid_progress_reseed_action_history': False,
 'post_recovery_mid_progress_reseed_phases': ['pull'],
 'post_recovery_mid_progress_reseed_max_count': 1,
 'post_recovery_mid_progress_reseed_prior_progress_action': False,
 'post_recovery_mid_progress_reseed_prior_max_age': 8,
 'post_recovery_act_bridge_no_progress_monitor_steps': 0,
 'post_recovery_act_bridge_no_progress_min_steps': 8,
 'post_recovery_act_bridge_no_progress_patience': 8,
 'post_recovery_act_bridge_no_progress_epsilon': 1e-05,
 'post_recovery_task_guard': False,
 'post_recovery_task_guard_check_safety': True,
 'post_recovery_task_guard_force_gripper': True,
 'post_recovery_task_guard_max_ee_distance': 0.0,
 'post_recovery_task_guard_min_progress': 1e-06,
 'post_recovery_task_guard_reanchor_phases': ['grasp'],
 'post_recovery_task_guard_steps': 24,
 'handoff_sanitize_controller_state': True,
 'handoff_seed_action_history_from_fresh_act': True,
 'record_policy_video': False,
 'replay_actions': None,
 'reset_action_history_after_human_exit': False,
 'reset_visual_history_after_human_exit': False,
 'reset_action_history_after_recovery': True,
 'reset_action_history_after_intervention_boundary': False,
 'reset_visual_history_after_recovery': True,
 'post_recovery_act_bridge_steps': 4,
 'resume_clear_steps': 3,
 'resume_clearance_threshold': 0.08,
 'robot_spawn_offset_xy': None,
 'safety_env': None,
 'save_actions': None,
 'save_frame_images': False,
 'seed': 0,
 'sequential_oscbf_fallback': False,
 'snapshot': None,
 'steps': 200,
 'stop_video_at': '2:30',
 'stop_video_at_steps': None,
 'unsafe_deformation_fallback': 'brake',
 'video_dir': None,
 'video_time_base': 'sim',
 'visual_only_human_arm': False}


def _ensure_eval_config_resolvers() -> None:
    if not OmegaConf.has_resolver("now"):
        from datetime import datetime

        OmegaConf.register_new_resolver(
            "now",
            lambda pattern="%Y-%m-%d_%H-%M-%S": datetime.now().strftime(pattern),
        )

def _resolve_eval_config_path(config_path: Optional[str], base_dir: Optional[Path] = None) -> Path:
    raw_path = Path(config_path).expanduser()
    if raw_path.is_absolute():
        return raw_path
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / raw_path)
    candidates.extend(
        (
            Path.cwd() / raw_path,
            EVAL_SCENARIO_CONFIG_DIR / raw_path,
            ROBOBASE_CFG / raw_path,
            REPO / raw_path,
        )
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def _resolve_safety_filter_config_path(config_path: Optional[str]) -> Path:
    raw_path = Path(config_path).expanduser()
    if raw_path.is_absolute():
        return raw_path
    candidates = (
        Path.cwd() / raw_path,
        SAFETY_FILTER_CONFIG_DIR / raw_path,
        ROBOBASE_CFG / raw_path,
        REPO / raw_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def _resolved_safety_filter_config(group: Any, resolved_eval_path: Path) -> dict[str, Any]:
    if group is None:
        group = {}
    if isinstance(group, str):
        group = {"config": group}
    if not isinstance(group, dict):
        raise ValueError(
            f"Eval config safety_filter must be a mapping or string: {resolved_eval_path}"
        )

    cfg = OmegaConf.create({})
    if DEFAULT_SAFETY_FILTER_CONFIG.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(DEFAULT_SAFETY_FILTER_CONFIG))

    config_name = group.get("config", group.get("base"))
    if config_name is not None:
        resolved_filter = _resolve_safety_filter_config_path(config_name)
        if not resolved_filter.exists():
            raise FileNotFoundError(f"Safety filter config not found: {resolved_filter}")
        if not (
            DEFAULT_SAFETY_FILTER_CONFIG.exists()
            and resolved_filter.resolve() == DEFAULT_SAFETY_FILTER_CONFIG.resolve()
        ):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(resolved_filter))

    overlay = {
        key: value
        for key, value in group.items()
        if key not in {"config", "base", "overrides"}
    }
    if group.get("overrides") is not None:
        overrides = group["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(
                f"Eval config safety_filter.overrides must be a mapping: {resolved_eval_path}"
            )
        overlay = OmegaConf.merge(OmegaConf.create(overlay), OmegaConf.create(overrides))
    if overlay:
        cfg = OmegaConf.merge(cfg, {"safety_filter": overlay})

    safety_cfg = cfg.get("safety_filter", cfg)
    container = OmegaConf.to_container(safety_cfg, resolve=True) or {}
    if not isinstance(container, dict):
        raise ValueError(
            f"Resolved safety_filter config must be a mapping: {resolved_eval_path}"
        )
    return dict(container)

def _load_eval_config_container(
    config_path: str,
    *,
    base_dir: Optional[Path] = None,
    seen: Optional[set[Path]] = None,
) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_eval_config_path(config_path, base_dir=base_dir).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Eval config not found: {resolved}")
    seen = set() if seen is None else set(seen)
    if resolved in seen:
        cycle = " -> ".join(str(path) for path in (*seen, resolved))
        raise ValueError(f"Eval config extends cycle: {cycle}")
    seen.add(resolved)

    _ensure_eval_config_resolvers()
    cfg = OmegaConf.load(resolved)
    container = OmegaConf.to_container(cfg, resolve=True) or {}
    if not isinstance(container, dict):
        raise ValueError(f"Eval config must be a mapping: {resolved}")

    extends = container.get("extends", []) or []
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list):
        raise ValueError(f"Eval config extends must be a string or list: {resolved}")

    merged = OmegaConf.create({})
    for parent in extends:
        parent_container, _ = _load_eval_config_container(
            str(parent),
            base_dir=resolved.parent,
            seen=seen,
        )
        merged = OmegaConf.merge(merged, OmegaConf.create(parent_container))

    current = dict(container)
    current.pop("extends", None)
    merged = OmegaConf.merge(merged, OmegaConf.create(current))
    out = OmegaConf.to_container(merged, resolve=True) or {}
    if not isinstance(out, dict):
        raise ValueError(f"Merged eval config must be a mapping: {resolved}")
    return dict(out), resolved

def _flatten_eval_config_paths(config_paths: Any) -> list[str]:
    if config_paths is None:
        return []
    if isinstance(config_paths, (str, Path)):
        return [str(config_paths)]
    flattened: list[str] = []
    for item in config_paths:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            flattened.extend(str(value) for value in item)
        else:
            flattened.append(str(item))
    return flattened

def _hydra_run_dir_default(container: dict[str, Any], resolved_eval_path: Path) -> Optional[str]:
    hydra = container.get("hydra", {}) or {}
    if not hydra:
        return None
    if not isinstance(hydra, dict):
        raise ValueError(f"Eval config hydra must be a mapping: {resolved_eval_path}")

    run = hydra.get("run", {}) or {}
    if not run:
        return None
    if not isinstance(run, dict):
        raise ValueError(f"Eval config hydra.run must be a mapping: {resolved_eval_path}")

    run_dir = run.get("dir")
    if run_dir is None or run_dir == "":
        return None
    return str(run_dir)

def _load_eval_config_defaults(config_paths: Any) -> tuple[dict[str, Any], list[Path]]:
    config_paths = _flatten_eval_config_paths(config_paths)
    if not config_paths:
        return {}, []

    merged = OmegaConf.create({})
    resolved_paths: list[Path] = []
    for config_path in config_paths:
        container_part, resolved_part = _load_eval_config_container(config_path)
        merged = OmegaConf.merge(merged, OmegaConf.create(container_part))
        resolved_paths.append(resolved_part)

    container = OmegaConf.to_container(merged, resolve=True) or {}
    if not isinstance(container, dict):
        raise ValueError(
            "Merged eval configs must produce a mapping: "
            f"{', '.join(str(path) for path in resolved_paths)}"
        )
    resolved = resolved_paths[-1]

    grouped_defaults = any(
        key in container for key in ("environment", "safety_filter", "eval_args")
    )
    if grouped_defaults:
        defaults: dict[str, Any] = {}
        for group_name in ("environment", "eval_args"):
            group = container.get(group_name, {}) or {}
            if not isinstance(group, dict):
                raise ValueError(
                    f"Eval config {group_name} must be a mapping: {resolved}"
                )
            defaults.update(group)
        defaults["safety_filter"] = _resolved_safety_filter_config(
            container.get("safety_filter"), resolved
        )
    else:
        eval_args = container.get("eval_args", container)
        if eval_args is None:
            eval_args = {}
        if not isinstance(eval_args, dict):
            raise ValueError(f"Eval config eval_args must be a mapping: {resolved}")
        defaults = dict(eval_args)
        defaults["safety_filter"] = _resolved_safety_filter_config({}, resolved)

    hydra_output_dir = _hydra_run_dir_default(container, resolved)
    if hydra_output_dir is not None and defaults.get("output_dir") is None:
        defaults["output_dir"] = hydra_output_dir

    for alias in ("overrides", "hydra_overrides"):
        if alias in defaults:
            existing = defaults.pop("override", []) or []
            extra = defaults.pop(alias) or []
            defaults["override"] = list(existing) + list(extra)

    fallback_aliases = {
        "chunk_allow_candidate_fallback": "chunk_allow_fallback_path",
        "chunk_candidate_fallback_only_if_no_optimized_result": "chunk_fallback_only_if_no_optimized_result",
        "chunk_gradient_early_stop_on_candidate": "chunk_gradient_early_stop_on_path",
    }
    for old_key, new_key in fallback_aliases.items():
        if old_key in defaults:
            if new_key not in defaults:
                defaults[new_key] = defaults.pop(old_key)
            else:
                defaults.pop(old_key)

    return defaults, resolved_paths

def _validate_eval_config_defaults(
    defaults: dict[str, Any],
    resolved_paths: list[Path],
    parser: argparse.ArgumentParser,
) -> None:
    known_dests = set(DEFAULT_EVAL_ARGS) | {"safety_filter"}
    unknown = sorted(
        key
        for key in defaults
        if key not in known_dests and key not in IGNORED_EVAL_CONFIG_KEYS
    )
    if unknown:
        config_label = ", ".join(str(path) for path in resolved_paths)
        parser.error(
            f"{config_label} contains unsupported eval config keys: "
            f"{', '.join(unknown)}"
        )

def _namespace_from_eval_config(
    config_paths: Any,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    defaults, resolved_paths = _load_eval_config_defaults(config_paths)
    _validate_eval_config_defaults(defaults, resolved_paths, parser)

    safety_filter_config = defaults.get("safety_filter") or {}
    values = copy.deepcopy(DEFAULT_EVAL_ARGS)
    values.update(
        {
            key: value
            for key, value in defaults.items()
            if key in DEFAULT_EVAL_ARGS
        }
    )
    values["eval_config"] = config_paths
    values["safety_filter"] = copy.deepcopy(safety_filter_config)
    return argparse.Namespace(**values)

def _args_safety_filter(args) -> dict[str, Any]:
    cfg = getattr(args, "safety_filter", None) or {}
    if not isinstance(cfg, dict):
        return {}
    return cfg

def _safety_filter_section(args, name: str) -> dict[str, Any]:
    section = _args_safety_filter(args).get(name, {}) or {}
    if not isinstance(section, dict):
        return {}
    return dict(section)

def _safety_filter_value(args, key: str, default: Any = None) -> Any:
    return _args_safety_filter(args).get(key, default)

def _safety_filter_debug(args) -> bool:
    return bool(_safety_filter_value(args, "debug", getattr(args, "debug", False)))

def _resolve_path_consistent_brake_config_path(config_path: Optional[str]) -> Optional[Path]:
    if config_path is None:
        return None
    raw_path = Path(config_path).expanduser()
    if raw_path.is_absolute():
        return raw_path
    candidates = (
        Path.cwd() / raw_path,
        ROBOBASE_CFG / "safety_filter" / raw_path,
        ROBOBASE_CFG / raw_path,
        REPO / raw_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def _load_path_consistent_brake_filter_config(config_path: Optional[str] = None) -> dict[str, Any]:
    cfg = OmegaConf.create({})
    if PATH_CONSISTENT_BRAKE_FILTER_CONFIG.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(PATH_CONSISTENT_BRAKE_FILTER_CONFIG))
    else:
        logger.warning("PathConsistentBrake base config missing: %s", PATH_CONSISTENT_BRAKE_FILTER_CONFIG)

    overlay_path = _resolve_path_consistent_brake_config_path(config_path)
    if overlay_path is not None:
        if not overlay_path.exists():
            raise FileNotFoundError(f"PathConsistentBrake config not found: {overlay_path}")
        if not (
            PATH_CONSISTENT_BRAKE_FILTER_CONFIG.exists()
            and overlay_path.resolve() == PATH_CONSISTENT_BRAKE_FILTER_CONFIG.resolve()
        ):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(overlay_path))

    safety_cfg = cfg.get("safety_filter", cfg)
    container = OmegaConf.to_container(safety_cfg, resolve=True)
    return dict(container or {})

def _positive_path_consistent_brake_limit_or_none(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)) and float(value) <= 0.0:
        return None
    return value

def _path_consistent_brake_kwargs_from_config(
    args,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if config is None:
        config = _load_path_consistent_brake_filter_config(
            getattr(args, "path_consistent_brake_config", None)
        )
    kwargs = {
        key: config[key]
        for key in PATH_CONSISTENT_BRAKE_CONFIG_KEYS
        if key in config
    }
    for key in PATH_CONSISTENT_BRAKE_LIMIT_KEYS:
        if key in kwargs:
            kwargs[key] = _positive_path_consistent_brake_limit_or_none(kwargs[key])
    return kwargs

def _path_consistent_brake_eval_config(args) -> dict[str, Any]:
    if getattr(args, "condition", None) != "path_consistent_brake":
        return {}
    return _args_safety_filter(args)

def parse_args(argv: Optional[Sequence[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=(
            "Run ACT safety evaluation from composable eval_scenarios YAML. "
            "Runtime method, environment, safety-filter, and output parameters "
            "belong in config files; the public CLI intentionally stays tiny."
        )
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        nargs="+",
        action="append",
        default=None,
        metavar="YAML",
        help=(
            "One or more eval_scenarios YAML presets. Repeat or list multiple "
            "configs to compose them; later configs override earlier ones."
        ),
    )

    cli_args = parser.parse_args(argv)
    if not cli_args.eval_config:
        parser.error(
            "--eval-config is required. Put method, environment, safety-filter, "
            "and runtime parameters in eval_scenarios YAML instead of CLI flags."
        )

    args = _namespace_from_eval_config(cli_args.eval_config, parser)
    if args.condition is None:
        parser.error("--condition is required unless supplied by --eval-config")
    if args.snapshot is None:
        parser.error("--snapshot is required unless supplied by --eval-config")
    args.record_video = not args.no_record_video
    if args.frame_image_every <= 0:
        parser.error("--frame-image-every must be positive.")
    if args.save_actions is not None and args.replay_actions is not None:
        parser.error("Use either --save-actions or --replay-actions, not both.")
    if args.enable_human_arm_collisions and args.visual_only_human_arm:
        parser.error(
            "--enable-human-arm-collisions cannot be combined with "
            "--visual-only-human-arm"
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
    if args.episode_index_offset < 0:
        parser.error("--episode-index-offset must be >= 0")
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
    if args.phase_reanchor_max_base_step_xy is not None:
        try:
            _phase_reanchor_max_base_step_xy = [float(v) for v in args.phase_reanchor_max_base_step_xy]
        except TypeError:
            parser.error("--phase-reanchor-max-base-step-xy must contain two positive values")
        if len(_phase_reanchor_max_base_step_xy) != 2 or any(v <= 0 for v in _phase_reanchor_max_base_step_xy):
            parser.error("--phase-reanchor-max-base-step-xy must contain two positive values")
    if args.phase_reanchor_base_gain < 0:
        parser.error("--phase-reanchor-base-gain must be >= 0")
    if args.phase_reanchor_arm_gain < 0:
        parser.error("--phase-reanchor-arm-gain must be >= 0")
    if args.phase_reanchor_arm_max_step < 0:
        parser.error("--phase-reanchor-arm-max-step must be >= 0")
    if args.phase_reanchor_live_taskspace_worsen_tolerance < 0:
        parser.error("--phase-reanchor-live-taskspace-worsen-tolerance must be >= 0")
    if args.phase_reanchor_live_taskspace_worsen_patience < 1:
        parser.error("--phase-reanchor-live-taskspace-worsen-patience must be >= 1")
    if args.phase_reanchor_live_taskspace_guard_disable_steps < 1:
        parser.error("--phase-reanchor-live-taskspace-guard-disable-steps must be >= 1")
    if args.phase_reanchor_live_taskspace_stop_min_steps < 0:
        parser.error("--phase-reanchor-live-taskspace-stop-min-steps must be >= 0")
    if args.phase_reanchor_q_servo_enable_target_dist < 0:
        parser.error("--phase-reanchor-q-servo-enable-target-dist must be >= 0")
    if args.phase_reanchor_live_release_target_error < 0:
        parser.error("--phase-reanchor-live-release-target-error must be >= 0")
    if args.phase_reanchor_live_release_handle_dist < 0:
        parser.error("--phase-reanchor-live-release-handle-dist must be >= 0")
    if args.phase_reanchor_nominal_window_selector not in {"taskspace_aware", "distribution_first"}:
        parser.error(
            "--phase-reanchor-nominal-window-selector must be one of: "
            "taskspace_aware, distribution_first"
        )
    if args.phase_reanchor_nominal_window_stage not in {"auto", "pregrasp", "manipulation", "unknown"}:
        parser.error(
            "--phase-reanchor-nominal-window-stage must be one of: "
            "auto, pregrasp, manipulation, unknown"
        )
    if args.phase_reanchor_bridge_handle_dist <= 0:
        parser.error("--phase-reanchor-bridge-handle-dist must be > 0")
    if args.phase_reanchor_bridge_seed_mode not in {
        "live_taskspace",
        "nominal_history",
        "nominal_history_with_live_veto",
    }:
        parser.error(
            "--phase-reanchor-bridge-seed-mode must be one of: "
            "live_taskspace, nominal_history, nominal_history_with_live_veto"
        )
    if args.phase_reanchor_bridge_seed_obs_source not in {"recovery", "nominal_q_window"}:
        parser.error(
            "--phase-reanchor-bridge-seed-obs-source must be one of: "
            "recovery, nominal_q_window"
        )
    if args.phase_reanchor_bridge_policy_step_source not in {"reset_zero", "nominal_window"}:
        parser.error(
            "--phase-reanchor-bridge-policy-step-source must be one of: "
            "reset_zero, nominal_window"
        )
    if args.phase_reanchor_bridge_action_agreement_l2 < 0:
        parser.error("--phase-reanchor-bridge-action-agreement-l2 must be >= 0")
    if not (0.0 <= args.phase_reanchor_bridge_action_agreement_cosine <= 1.0):
        parser.error("--phase-reanchor-bridge-action-agreement-cosine must be in [0, 1]")
    if args.phase_reanchor_bridge_action_agreement_mode not in {"and", "or"}:
        parser.error("--phase-reanchor-bridge-action-agreement-mode must be one of: and, or")
    if args.phase_reanchor_bridge_preload_steps < 1:
        parser.error("--phase-reanchor-bridge-preload-steps must be >= 1")
    if args.phase_reanchor_bridge_preload_progress_delta < 0:
        parser.error("--phase-reanchor-bridge-preload-progress-delta must be >= 0")
    if args.phase_reanchor_bridge_preload_progress_min_abs < 0:
        parser.error("--phase-reanchor-bridge-preload-progress-min-abs must be >= 0")
    if args.phase_reanchor_bridge_preload_handle_dist <= 0:
        parser.error("--phase-reanchor-bridge-preload-handle-dist must be > 0")
    if args.phase_reanchor_bridge_preload_pull_step < 0:
        parser.error("--phase-reanchor-bridge-preload-pull-step must be >= 0")
    if args.phase_reanchor_live_extend_steps < 1:
        parser.error("--phase-reanchor-live-extend-steps must be >= 1")
    if args.phase_reanchor_live_max_extensions < 0:
        parser.error("--phase-reanchor-live-max-extensions must be >= 0")
    if args.phase_reanchor_live_ee_gain < 0:
        parser.error("--phase-reanchor-live-ee-gain must be >= 0")
    if args.phase_reanchor_live_ee_nominal_reg < 0:
        parser.error("--phase-reanchor-live-ee-nominal-reg must be >= 0")
    if not 0 <= args.phase_reanchor_live_ee_servo_mix <= 1:
        parser.error("--phase-reanchor-live-ee-servo-mix must be between 0 and 1")
    if str(args.phase_reanchor_task_point_source) not in {"ee_site", "gripper_geom"}:
        parser.error("--phase-reanchor-task-point-source must be ee_site or gripper_geom")
    if str(args.phase_reanchor_measurement_task_point_source) not in {"control", "ee_site", "gripper_geom"}:
        parser.error("--phase-reanchor-measurement-task-point-source must be control, ee_site, or gripper_geom")
    if str(args.phase_reanchor_control_error_source) not in {"control", "measurement"}:
        parser.error("--phase-reanchor-control-error-source must be control or measurement")
    if args.phase_reanchor_live_handle_assist_trigger_target_dist < 0:
        parser.error("--phase-reanchor-live-handle-assist-trigger-target-dist must be >= 0")
    if args.phase_reanchor_live_handle_assist_gain < 0:
        parser.error("--phase-reanchor-live-handle-assist-gain must be >= 0")
    if args.phase_reanchor_live_handle_assist_max_base_step_xy is not None:
        try:
            _phase_reanchor_live_handle_assist_max_base_step_xy = [
                float(v) for v in args.phase_reanchor_live_handle_assist_max_base_step_xy
            ]
        except TypeError:
            parser.error("--phase-reanchor-live-handle-assist-max-base-step-xy must contain two positive values")
        if (
            len(_phase_reanchor_live_handle_assist_max_base_step_xy) != 2
            or any(v <= 0 for v in _phase_reanchor_live_handle_assist_max_base_step_xy)
        ):
            parser.error("--phase-reanchor-live-handle-assist-max-base-step-xy must contain two positive values")
    if args.phase_reanchor_near_live_target_suppress_dist < 0:
        parser.error("--phase-reanchor-near-live-target-suppress-dist must be >= 0")
    if args.phase_reanchor_task_point_geometry_trust_error < 0:
        parser.error("--phase-reanchor-task-point-geometry-trust-error must be >= 0")
    if args.phase_reanchor_live_ee_servo_min_predicted_improvement < 0:
        parser.error("--phase-reanchor-live-ee-servo-min-predicted-improvement must be >= 0")
    if args.phase_reanchor_arm_fd_eps <= 0:
        parser.error("--phase-reanchor-arm-fd-eps must be > 0")
    if args.phase_reanchor_arm_damping < 0:
        parser.error("--phase-reanchor-arm-damping must be >= 0")
    if args.phase_reanchor_nominal_window_len < 1:
        parser.error("--phase-reanchor-nominal-window-len must be >= 1")
    if args.phase_reanchor_nominal_window_lead_steps < 0:
        parser.error("--phase-reanchor-nominal-window-lead-steps must be >= 0")
    if args.phase_reanchor_arm_error_clip < 0:
        parser.error("--phase-reanchor-arm-error-clip must be >= 0")
    if args.phase_reanchor_wrist_servo_weight < 0:
        parser.error("--phase-reanchor-wrist-servo-weight must be >= 0")
    if args.phase_reanchor_grasp_dist < 0:
        parser.error("--phase-reanchor-grasp-dist must be >= 0")
    if args.post_recovery_task_guard_steps < 1:
        parser.error("--post-recovery-task-guard-steps must be >= 1")
    if args.post_recovery_progress_tolerance < 0:
        parser.error("--post-recovery-progress-tolerance must be >= 0")
    if args.post_recovery_no_progress_start_step < 0:
        parser.error("--post-recovery-no-progress-start-step must be >= 0")
    if args.post_recovery_no_progress_patience < 1:
        parser.error("--post-recovery-no-progress-patience must be >= 1")
    if args.post_recovery_no_progress_max_progress < 0:
        parser.error("--post-recovery-no-progress-max-progress must be >= 0")
    if args.post_recovery_no_progress_min_target_distance < 0:
        parser.error("--post-recovery-no-progress-min-target-distance must be >= 0")
    if args.post_recovery_mid_progress_patience < 1:
        parser.error("--post-recovery-mid-progress-patience must be >= 1")
    if args.post_recovery_mid_progress_epsilon < 0:
        parser.error("--post-recovery-mid-progress-epsilon must be >= 0")
    if args.post_recovery_mid_progress_distance_regression < 0:
        parser.error("--post-recovery-mid-progress-distance-regression must be >= 0")
    if args.post_recovery_mid_progress_min_target_distance < 0:
        parser.error("--post-recovery-mid-progress-min-target-distance must be >= 0")
    if args.post_recovery_mid_progress_reseed_max_count < 0:
        parser.error("--post-recovery-mid-progress-reseed-max-count must be >= 0")
    if args.post_recovery_mid_progress_reseed_prior_max_age < 0:
        parser.error("--post-recovery-mid-progress-reseed-prior-max-age must be >= 0")
    if args.post_recovery_act_bridge_no_progress_monitor_steps < 0:
        parser.error("--post-recovery-act-bridge-no-progress-monitor-steps must be >= 0")
    if args.post_recovery_act_bridge_no_progress_min_steps < 0:
        parser.error("--post-recovery-act-bridge-no-progress-min-steps must be >= 0")
    if args.post_recovery_act_bridge_no_progress_patience < 1:
        parser.error("--post-recovery-act-bridge-no-progress-patience must be >= 1")
    if args.post_recovery_act_bridge_no_progress_epsilon < 0:
        parser.error("--post-recovery-act-bridge-no-progress-epsilon must be >= 0")
    if str(args.post_recovery_no_progress_distance_source) not in {"measurement", "control", "site", "gripper"}:
        parser.error("--post-recovery-no-progress-distance-source must be measurement, control, site, or gripper")
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
