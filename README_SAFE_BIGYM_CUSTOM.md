# Safe BiGym HOI custom notes

This file tracks local safety-filter and ACT evaluation tooling that is specific to this workspace, separate from the upstream RoboBase README.

## ACT velocity tolerance sweep

Use this diagnostic to test whether ACT can recover after local speed changes in a predicted action horizon. The command below runs the human-arm runtime environment while hiding the human arm from policy observations, uses the checkpoint drawer-top-open workspace, keeps human-arm collisions disabled from `default_human_arm_motion.yaml`, and disables action clipping so `scale=1.0` remains a valid baseline.

```commandline
cd /home/xd1125/Workspace/safe_bigym_hoi/external/robobase && \
TF_CPP_MIN_LOG_LEVEL=2 \
/home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python \
tools/sweep_act_velocity_tolerance.py \
--eval-config robobase/cfgs/eval_scenarios/default_human_arm_motion.yaml \
--episodes 1 \
--steps 300 \
--perturb-steps 60,90,120 \
--window-starts 0 \
--window-len 4 \
--scales 1.0,0.75,0.5,0.25,1.25,1.5,2.0,3.0 \
--profiles constant,accelerate,decelerate \
--execution-mode waypoint \
--no-clip-action \
--hide-human-arm-policy-obs \
--log-steps
```

A validated scale-1 baseline under this setting succeeded with `success_rate=1.0`, `clip_count=0`, and near-zero perturbation norm.
