from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.action_modes import JointPositionActionMode
from robobase.utils import DemoEnv, add_demo_to_replay_buffer
from robobase.envs.utils.bigym_utils import TASK_MAP
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    RescaleFromTanhWithMinMax,
    OnehotTime,
    ActionSequence,
    AppendDemoInfo,
    FrameStack,
    ConcatDim,
    RecedingHorizonControl,
)
from omegaconf import DictConfig
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.action_modes import PelvisDof
import multiprocessing as mp
import logging
import numpy as np

from demonstrations.demo import DemoStep, Demo
from demonstrations.demo_store import DemoStore, DemoNotFoundError
from demonstrations.demo_converter import DemoConverter
from demonstrations.utils import Metadata

from typing import List, Dict, Tuple, Callable
import copy
import inspect
from bigym.const import CACHE_PATH
import os
import json
from pathlib import Path
from pathlib import Path
from demonstrations.demo import Demo
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

UNIT_TEST = False

def _validate_mode_labels_from_info(cfg, demos):
    require_mode_label = cfg.env.get("require_mode_label", False)

    if not require_mode_label:
        return demos

    kept = []
    dropped = []
    error_type = {'1': "success", 
                  "-1": "Missing Info", 
                  "-2":"mode_label Not found in info", 
                  "-3": "Failed to convert to int type"}
    for demo in demos:
        demo_uuid = str(demo.uuid)
        ok = 1

        for i, ts in enumerate(demo.timesteps):
            if ts.info is None:
                ok = -1
                break
            if "mode_label" not in ts.info:
                ok = -2
                break

            try:
                ts.info["mode_label"] = int(ts.info["mode_label"])
            except Exception:
                ok = -3
                break

        if ok:
            kept.append(demo)
        else:
            dropped.append(demo_uuid)
            logging.warning(f"Dropping demo; {error_type[ok]}")

    print(f"[mode_label] kept {len(kept)} demos with info labels, dropped {len(dropped)} without valid labels")
    if dropped:
        print("[mode_label] dropped uuids:", dropped[:10])

    return kept

def rescale_demo_actions(
    rescale_fn: Callable, demos: List[List[DemoStep]], cfg: DictConfig
):
    """Rescale actions in demonstrations to [-1, 1] Tanh space.
    This is because RoboBase assumes everything to be in [-1, 1] space.

    Args:
        rescale_fn: callable that takes info containing demo action and cfg and
            outputs the rescaled action
        demos: list of demo episodes whose actions are raw, i.e., not scaled
        cfg: Configs

    Returns:
        List[Demo]: list of demo episodes whose actions are rescaled
    """
    for demo in demos:
        for step in demo:
            info = step.info
            if "demo_action" in info:
                # Rescale demo actions
                info["demo_action"] = rescale_fn(info, cfg)
    return demos


def _task_name_to_env_class(task_name: str) -> type[BiGymEnv]:
    return TASK_MAP[task_name]


class BiGymEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg, demo_env=False, train=True, return_raw_spaces=False):
        # last two are grippers
        assert cfg.demos != 0
        assert cfg.action_repeat == 1

        action_space = copy.deepcopy(env.action_space)
        observation_space = copy.deepcopy(env.observation_space)

        env = RescaleFromTanhWithMinMax(
            env=env,
            action_stats=self._action_stats,
            min_max_margin=cfg.min_max_margin,
        )
        obs_stats = None
        if cfg.norm_obs:
            obs_stats = self._obs_stats

        # We normalize the low dimensional observations in the ConcatDim wrapper.
        # This is to be consistent with the original ACT implementation.
        env = ConcatDim(
            env,
            shape_length=1,
            dim=-1,
            new_name="low_dim_state",
            norm_obs=cfg.norm_obs,
            obs_stats=obs_stats,
            keys_to_ignore=["proprioception_floating_base_actions"],
        )
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack)
        env = TimeLimit(
            env,
            cfg.env.episode_length // cfg.env.demo_down_sample_rate,
        )

        if not demo_env:
            if not train:
                env = RecedingHorizonControl(
                    env,
                    cfg.action_sequence,
                    cfg.env.episode_length // (cfg.env.demo_down_sample_rate),
                    cfg.execution_length,
                    temporal_ensemble=cfg.temporal_ensemble,
                    gain=cfg.temporal_ensemble_gain,
                )
            else:
                env = ActionSequence(
                    env,
                    cfg.action_sequence,
                )

        env = AppendDemoInfo(env)

        if return_raw_spaces:
            return env, action_space, observation_space
        else:
            return env

    def _create_env(self, cfg: DictConfig) -> BiGymEnv:
        bigym_class = _task_name_to_env_class(cfg.env.task_name)
        camera_configs = [
            CameraConfig(
                name=camera_name,
                rgb=True,
                depth=False,
                resolution=cfg.visual_observation_shape,
            )
            for camera_name in cfg.env.cameras
        ]

        if cfg.env.enable_all_floating_dof:
            action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
                floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
            )
        else:
            action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
            )

        env_kwargs = dict(
            render_mode=cfg.env.render_mode,
            action_mode=action_mode,
            observation_config=ObservationConfig(
                cameras=camera_configs if cfg.pixels else [],
                proprioception=True,
                privileged_information=cfg.env.get(
                    "privileged_information", (not cfg.pixels)
                ),
            ),
            control_frequency=CONTROL_FREQUENCY_MAX // cfg.env.demo_down_sample_rate,
        )
        if "arm_action_mode" in inspect.signature(bigym_class.__init__).parameters:
            env_kwargs["arm_action_mode"] = cfg.env.get("arm_action_mode", "scripted")

        return bigym_class(**env_kwargs)

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.SyncVectorEnv
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    self._create_env(cfg),
                    cfg,
                    demo_env=False,
                    train=True,
                )
                for _ in range(cfg.num_train_envs)
            ],
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        env, self._action_space, self._observation_space = self._wrap_env(
            env=self._create_env(cfg),
            cfg=cfg,
            demo_env=False,
            train=False,
            return_raw_spaces=True,
        )
        return env

    def _get_demo_fn(self, cfg: DictConfig, num_demos: int):
        demos = []

        logging.info("Start to load demos.")
        env = self._create_env(cfg)
        target_frequency = CONTROL_FREQUENCY_MAX // cfg.env.demo_down_sample_rate
        demo_manifest = cfg.env.get("manifest", None)
        demo_store = DemoStore()

        if np.isinf(num_demos):
            num_demos = -1
        if demo_manifest is not None:
            logging.info(
                "Loading Task %s from manifest: '%s'",
                cfg.env.task_name, demo_manifest,
            )
            demos = self._load_demos_from_manifest(demo_manifest, num_demos)
            logging.info(f"Loaded {len(demos)} demos from manifest.")
        else:
            logging.info(
                "Loading from DemoStore: %s",
                cfg.env.task_name,
            )
            try:
                demos = demo_store.get_demos(
                    Metadata.from_env(env),
                    amount=num_demos,
                    frequency=target_frequency,
                )
            except DemoNotFoundError:                
                env.close()
                raise
        if len(demos) == 0:
            raise RuntimeError(
                f"No demos loaded from manifest: {demo_manifest}"
            )
        
        for demo in demos:
            for ts in demo.timesteps:
                ts.observation = {
                    k: np.array(v, dtype=np.float32) for k, v in ts.observation.items()
                }
        env.close()
        logging.info("Finished loading demos.")
        return demos

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        demos = self._get_demo_fn(cfg, num_demos)
        demos = _validate_mode_labels_from_info(cfg, demos)
        self._raw_demos = demos
        self._action_stats = self._compute_action_stats(cfg, demos)
        self._obs_stats = self._compute_obs_stats(cfg, demos)

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        demo_list = [demo.timesteps for demo in self._raw_demos]
        demo_list = rescale_demo_actions(
            self._rescale_demo_action_helper, demo_list, cfg
        )
        self._demos = self._demo_to_steps(cfg, demo_list)

    def _load_demos_from_manifest(self, manifest_path: str, amount: int = -1):
        manifest_path = Path(manifest_path).expanduser()
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        if amount is not None and amount > 0:
            entries = entries[:amount]

        demos = []
        for entry in entries:
            # Filter unsuccessful tasks
            if entry["success"] == 1:
                demo_path = Path(entry["target_path"]).expanduser()
                demo = Demo.from_safetensors(demo_path)   # replace with your actual demo loader
                demos.append(demo)

        return demos
    
    def load_demos_into_replay(self, cfg: DictConfig, buffer, is_demo_buffer):
        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        
        if is_demo_buffer:
            # Filter successful demonstrations
            demos = []
            for i, demo in enumerate(self._demos):
                successful = (demo[0][-1]["demo"] == 1)
                if successful:
                    demos.append(demo)
                else:
                    print(f"Skipping failed demonstration {i}")
                    continue
        else:
            demos = self._demos

        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(demos), self._action_space, self._observation_space
            ),
            cfg,
            demo_env=True,
            train=False,
        )
        for _ in range(len(demos)):
            add_demo_to_replay_buffer(demo_env, buffer)
            
    def _demo_to_steps(
        self, cfg: DictConfig, demo_list: List[List[DemoStep]]
    ) -> List[DemoStep]:
        ret_demos = []

        demos_from_manifest = cfg.env.get("manifest", None) is not None

        for demo in demo_list:
            cur_demo = []
            last_timestep = False

            if len(demo) == 0:
                continue

            if demos_from_manifest:
                # Manifest-selected demos are treated as successful by construction.
                successful_demo = True
            else:
                # Compute success from transition rewards only.
                # Skip i == 0 because the first timestep is the reset / initial state
                # and may not have a valid reward.
                rewards = []
                for i, step in enumerate(demo):
                    if i == 0:
                        continue

                    reward = step.reward
                    if reward is None:
                        raise RuntimeError(
                            f"Reward is None in demo at transition step {i}"
                        )

                    rewards.append(float(reward))

                successful_demo = sum(rewards) > 0.25

            for i, step in enumerate(demo):
                if step.info is None:
                    step.info = {}

                step.info.update({"demo": int(successful_demo)})

                if i == 0:
                    # Initial observation timestep: no reward / term / trunc payload.
                    cur_demo.append((step.observation, step.info))
                    continue

                term, trunc = step.termination, step.truncation
                reward = step.reward

                # Be defensive in case reward is missing in some demos.
                if reward is None:
                    reward = 0.0

                if demos_from_manifest:
                    # Manifest demos are already filtered to successful recordings.
                    # End on the final step or on explicit success flag in info.
                    if i == len(demo) - 1 or bool(step.info.get("success", False)):
                        if not (term or trunc):
                            term = False
                            trunc = True
                        last_timestep = True
                else:
                    # Non-manifest demos: infer end-of-demo from final step or positive reward.
                    if i == len(demo) - 1 or reward > 0:
                        if not (term or trunc):
                            term = False
                            trunc = True
                        last_timestep = True

                cur_demo.append((step.observation, reward, term, trunc, step.info))

                if last_timestep:
                    break

            ret_demos.append(cur_demo)

        return ret_demos
    
    def _compute_action_stats(
        self, cfg: DictConfig, demos: List[List[DemoStep]]
    ) -> Dict:
        actions = []
        for demo in demos:
            for step in demo.timesteps:
                info = step.info
                if "demo_action" in info:
                    actions.append(info["demo_action"])
        actions = np.stack(actions)

        mean, std, gmax, gmin = self._get_gripper_action_stats(cfg)
        action_mean = np.hstack([np.mean(actions, 0)[:-2], mean, mean])
        action_std = np.hstack([np.std(actions, 0)[:-2], std, std])
        action_max = np.hstack([np.max(actions, 0)[:-2], gmax, gmax])
        action_min = np.hstack([np.min(actions, 0)[:-2], gmin, gmin])
        action_stats = {
            "mean": action_mean,
            "std": action_std,
            "max": action_max,
            "min": action_min,
        }
        return action_stats



    def _compute_obs_stats(self, cfg: DictConfig, demos: List[List[DemoStep]]) -> Dict:
        count = defaultdict(int)
        mean = {}
        M2 = {}
        min_val = {}
        max_val = {}

        for demo in demos:
            for step in demo.timesteps:
                obs.append(step.observation)

        keys = obs[0].keys()
        obs = {key: np.stack([o[key] for o in obs], axis=0) for key in keys}
        obs_mean = {key: np.mean(obs[key], 0) for key in keys}
        obs_std = {key: np.std(obs[key], 0) for key in keys}
        obs_min = {key: np.min(obs[key], 0) for key in keys}
        obs_max = {key: np.max(obs[key], 0) for key in keys}
        obs_stats = {
            "mean": obs_mean,
            "std": obs_std,
            "max": obs_max,
            "min": obs_min,
        }
        return obs_stats
    
    def _get_gripper_action_stats(
        self, cfg: DictConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if cfg.env.action_mode in ["absolute", "delta"]:
            return (0.5, 0.25, 1, 0)
        else:
            raise NotImplementedError("Unsupported action mode.")

    def _rescale_demo_action_helper(self, info, cfg: DictConfig):
        return RescaleFromTanhWithMinMax.transform_to_tanh(
            info["demo_action"],
            action_stats=self._action_stats,
            min_max_margin=cfg.min_max_margin,
        )



