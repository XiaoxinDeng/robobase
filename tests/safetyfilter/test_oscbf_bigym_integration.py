from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from hydra import compose, initialize_config_dir

from robobase.envs.bigym import BiGymEnvFactory
from robobase.safetyfilter.h1_state_bridge import extract_h1_state, H1State
from robobase.safetyfilter.oscbf.oscbffilter import OSCBFFilter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


REPO = Path("/home/xd1125/Workspace/safe_bigym_hoi")
ROBOBASE_CFG = REPO / "external/robobase/robobase/cfgs"
H1_URDF = REPO / "external/oscbf/oscbf/assets/h1/h1.urdf"

def make_random_env_action(env_action_shape: tuple[int, ...], scale: float) -> np.ndarray:
    return np.random.uniform(
        low=-scale,
        high=scale,
        size=env_action_shape,
    ).astype(np.float32)


def extract_first_action(env_action: np.ndarray) -> np.ndarray:
    env_action = np.asarray(env_action, dtype=np.float32)

    if env_action.ndim == 1:
        return env_action.copy()

    if env_action.ndim == 2:
        return env_action[0].copy()

    raise ValueError(f"Unsupported env_action shape: {env_action.shape}")


def replace_first_action(
    env_action: np.ndarray,
    safe_first_action: np.ndarray,
) -> np.ndarray:
    env_action = np.asarray(env_action, dtype=np.float32).copy()
    safe_first_action = np.asarray(safe_first_action, dtype=np.float32).reshape(-1)

    if env_action.ndim == 1:
        if env_action.shape != safe_first_action.shape:
            raise ValueError(
                f"env_action shape {env_action.shape} does not match "
                f"safe_first_action shape {safe_first_action.shape}"
            )
        return safe_first_action.astype(np.float32)

    if env_action.ndim == 2:
        if env_action.shape[1] != safe_first_action.shape[0]:
            raise ValueError(
                f"env_action second dim {env_action.shape[1]} does not match "
                f"safe_first_action dim {safe_first_action.shape[0]}"
            )
        env_action[0] = safe_first_action
        return env_action.astype(np.float32)

    raise ValueError(f"Unsupported env_action shape: {env_action.shape}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["dummy", "real"],
        default="dummy",
        help=(
            "dummy: tests interface and slicing only. "
            "real: runs actual OSCBF and human-capsule extraction."
        ),
    )

    parser.add_argument(
        "--env",
        default="bigym/drawer_top_open",
        help=(
            "RoboBase env override. "
            "For real OSCBF, use a human-arm task, e.g. "
            "bigym/human_arm_drawer_top_open if available."
        ),
    )

    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--demos", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--dummy-scale", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def make_cfg(args):
    overrides = [
        f"env={args.env}",
        "method=act",
        "launch=act_pixel_bigym",
        f"demos={args.demos}",
        "num_train_envs=0",
        "num_eval_episodes=1",
        "log_eval_video=false",
        "replay.num_workers=0",
    ]

    with initialize_config_dir(config_dir=str(ROBOBASE_CFG), version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=overrides,
        )

    return cfg


def make_eval_env(cfg):
    env_factory = BiGymEnvFactory()

    if cfg.demos != 0:
        env_factory.collect_or_fetch_demos(cfg, cfg.demos)

    env = env_factory.make_eval_env(cfg)

    if cfg.demos != 0:
        env_factory.post_collect_or_fetch_demos(cfg)

    return env


def make_filter(args) -> OSCBFFilter:
    return OSCBFFilter(
        urdf_path=str(H1_URDF),
        debug=args.debug,
        use_dummy_filter=(args.mode == "dummy"),
        dummy_scale=args.dummy_scale,
        control_type="absolute",
    )


def infer_action_dim(env, fallback: int = 16) -> int:
    if hasattr(env, "action_space"):
        shape = getattr(env.action_space, "shape", None)
        if shape is not None and len(shape) == 1:
            return int(shape[0])
    return fallback


def make_random_action(action_dim: int, scale: float) -> np.ndarray:
    return np.random.uniform(
        low=-scale,
        high=scale,
        size=(action_dim,),
    ).astype(np.float32)


def get_non_arm_indices(action_dim: int, arm_indices: Sequence[int]) -> np.ndarray:
    arm_set = set(int(i) for i in arm_indices)
    return np.asarray(
        [i for i in range(action_dim) if i not in arm_set],
        dtype=np.int64,
    )


def assert_common_filter_properties(
    action: np.ndarray,
    safe_action: np.ndarray,
    arm_indices: np.ndarray,
):
    non_arm_indices = get_non_arm_indices(action.shape[0], arm_indices)

    assert safe_action.shape == action.shape, (
        f"safe_action shape {safe_action.shape}, expected {action.shape}"
    )

    assert np.isfinite(safe_action).all(), "safe_action contains NaN or Inf"

    assert np.allclose(
        safe_action[non_arm_indices],
        action[non_arm_indices],
        atol=1e-6,
        rtol=1e-6,
    ), (
        "Non-arm action dimensions changed.\n"
        f"non_arm_indices={non_arm_indices}\n"
        f"nominal_non_arm={action[non_arm_indices]}\n"
        f"safe_non_arm={safe_action[non_arm_indices]}"
    )


def assert_dummy_expected_behavior(
    filt: OSCBFFilter,
    action: np.ndarray,
    safe_action: np.ndarray,
    q_full: np.ndarray,
):
    arm_idx = filt.bigym_action_arm_indices
    q_arm = q_full[filt.bigym_state_arm_indices]
    a_arm = action[arm_idx]

    if filt.control_type == "absolute":
        expected_arm = q_arm + filt.dummy_scale * (a_arm - q_arm)
    else:
        expected_arm = filt.dummy_scale * a_arm

    assert np.allclose(
        safe_action[arm_idx],
        expected_arm,
        atol=1e-6,
        rtol=1e-6,
    ), (
        "Dummy filter arm output is not as expected.\n"
        f"nominal_arm={a_arm}\n"
        f"safe_arm={safe_action[arm_idx]}\n"
        f"expected_arm={expected_arm}"
    )

def infer_env_action_shape(env, fallback=(16, 16)) -> tuple[int, ...]:
    if hasattr(env, "action_space"):
        shape = getattr(env.action_space, "shape", None)
        if shape is not None:
            return tuple(int(x) for x in shape)
    return fallback


def print_step_summary(
    step: int,
    action: np.ndarray,
    safe_action: np.ndarray,
    arm_indices: np.ndarray,
):
    non_arm_indices = get_non_arm_indices(action.shape[0], arm_indices)

    arm_delta = float(np.linalg.norm(safe_action[arm_indices] - action[arm_indices]))
    full_delta = float(np.linalg.norm(safe_action - action))
    non_arm_delta = float(
        np.linalg.norm(safe_action[non_arm_indices] - action[non_arm_indices])
    )

    print(f"\n--- Step {step} ---")
    print("arm_indices:", arm_indices.tolist())
    print("non_arm_indices:", non_arm_indices.tolist())
    print("nominal_arm:", action[arm_indices])
    print("safe_arm:", safe_action[arm_indices])
    print("arm_delta:", arm_delta)
    print("non_arm_delta:", non_arm_delta)
    print("full_delta:", full_delta)


def main():
    args = parse_args()

    if args.mode == "real" and "human" not in args.env.lower():
        print(
            "\n[WARNING] You are running real OSCBF on an env name that does not "
            "look like a human-arm task.\n"
            "If the task has no `humanarms`, _extract_human_obstacles() will fail.\n"
            "Use something like --env=bigym/human_arm_drawer_top_open if available.\n"
        )

    print("\n=== Building RoboBase config ===")
    cfg = make_cfg(args)

    print("\n=== Creating BiGym eval env ===")
    env = make_eval_env(cfg)

    print("\n=== Creating OSCBF filter ===")
    filt = make_filter(args)

    print("\n=== Resetting env ===")
    obs, info = env.reset()
    env_action_shape = infer_env_action_shape(env, fallback=(16, 16))

    if len(env_action_shape) == 1:
        action_seq_len = None
        action_dim = env_action_shape[0]
    elif len(env_action_shape) == 2:
        action_seq_len, action_dim = env_action_shape
    else:
        raise ValueError(f"Unsupported env action shape: {env_action_shape}")


    action_dim = infer_action_dim(env, fallback=16)

    print("env_action_shape:", env_action_shape)
    print("action_seq_len:", action_seq_len)
    print("single_action_dim:", action_dim)
    print("action_dim:", action_dim)
    print("filter mode:", args.mode)
    print("arm action indices:", filt.bigym_action_arm_indices.tolist())
    print("arm state indices:", filt.bigym_state_arm_indices.tolist())

    try:
        for step in range(args.steps):
            print(f"\n=== Integration step {step} ===")

            h1state: H1State = extract_h1_state(
                env,
                print_diagnostics=(step == 0),
            )

            q_full = np.asarray(h1state.q_full, dtype=np.float32).reshape(-1)
            qd_full = np.asarray(h1state.qd_full, dtype=np.float32).reshape(-1)

            print("q_full shape:", q_full.shape)
            print("qd_full shape:", qd_full.shape)
            print("q finite:", bool(np.isfinite(q_full).all()))
            print("qd finite:", bool(np.isfinite(qd_full).all()))

            assert q_full.shape == (14,), f"Expected q_full shape (14,), got {q_full.shape}"
            assert qd_full.shape == (14,), f"Expected qd_full shape (14,), got {qd_full.shape}"
            assert np.isfinite(q_full).all()
            assert np.isfinite(qd_full).all()

            # Env expects either:
            #   single action: (16,)
            #   action sequence/chunk: (16, 16)
            env_action = make_random_env_action(env_action_shape, args.action_scale)

            # OSCBFFilter operates on one 16D action.
            first_action = extract_first_action(env_action)

            safe_first_action = filt(
                action=first_action,
                env=env,
                observations=obs,
                q_full=q_full,
                qd_full=qd_full,
            )

            assert_common_filter_properties(
                action=first_action,
                safe_action=safe_first_action,
                arm_indices=filt.bigym_action_arm_indices,
            )

            if args.mode == "dummy":
                assert_dummy_expected_behavior(
                    filt=filt,
                    action=first_action,
                    safe_action=safe_first_action,
                    q_full=q_full,
                )

            print_step_summary(
                step=step,
                action=first_action,
                safe_action=safe_first_action,
                arm_indices=filt.bigym_action_arm_indices,
            )

            # Put the filtered first action back into the full env action/chunk.
            safe_env_action = replace_first_action(
                env_action=env_action,
                safe_first_action=safe_first_action,
            )

            print("\n=== Calling env.step(safe_env_action) ===")
            print("env_action shape:", env_action.shape)
            print("safe_env_action shape:", safe_env_action.shape)

            obs, reward, terminated, truncated, info = env.step(safe_env_action)

            print("reward:", reward)
            print("terminated:", terminated)
            print("truncated:", truncated)

            if terminated or truncated:
                print("Episode ended early.")
                break

    except AttributeError as exc:
        if args.mode == "real":
            raise RuntimeError(
                "Real OSCBF integration failed while extracting human obstacles. "
                "This usually means the selected BiGym task has no `humanarms`. "
                "Use a human-arm task for real OSCBF testing."
            ) from exc
        raise

    finally:
        env.close()

    print("\n[PASS] OSCBF + RoboBase/BiGym integration test passed.")


if __name__ == "__main__":
    main()