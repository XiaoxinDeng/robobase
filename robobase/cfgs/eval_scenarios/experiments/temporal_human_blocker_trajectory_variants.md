# Temporary Human-Arm Trajectory Variants

These scenarios all use SafeChunk-Deform with the temporary blocker enabled and guarded ACT action/visual history reset after human exit.

The human keyframe order is:

```text
[arm_tx, arm_ty, shoulder_base, shoulder_yaw, shoulder_pitch, elbow]
```

## Variants

- `temporal_human_blocker_trajectory_left_gate_chunk_deform.yaml`: explicit strong gate near the known successful 50ep blocker.
- `temporal_human_blocker_trajectory_short_lateral_chunk_deform.yaml`: shorter lateral sweep for brief blocker tests.
- `temporal_human_blocker_trajectory_slow_drift_chunk_deform.yaml`: slow entry and longer swaying hold.
- `temporal_human_blocker_trajectory_fast_probe_chunk_deform.yaml`: fast incursion and quick exit.
- `temporal_human_blocker_trajectory_opposite_exit_chunk_deform.yaml`: crosses in and exits on the opposite side.
- `temporal_human_blocker_trajectory_high_elbow_chunk_deform.yaml`: folded high-elbow forearm geometry.

## Run

```bash
cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase
TF_CPP_MIN_LOG_LEVEL=2 /home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
  eval_act_oscbf_safety_metrics.py \
  --eval-config robobase/cfgs/eval_scenarios/experiments/temporal_human_blocker_trajectory_fast_probe_chunk_deform.yaml
```

## Smoke Overlay

Use `temporal_human_blocker_trajectory_smoke_overlay.yaml` as a second config for a 1-episode, 140-step, no-video smoke run:

```bash
cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase
/home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
  eval_act_oscbf_safety_metrics.py \
  --eval-config \
  robobase/cfgs/eval_scenarios/experiments/temporal_human_blocker_trajectory_fast_probe_chunk_deform.yaml \
  robobase/cfgs/eval_scenarios/experiments/temporal_human_blocker_trajectory_smoke_overlay.yaml
```
