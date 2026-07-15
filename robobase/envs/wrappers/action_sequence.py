"""Wrapper for allowing action sequences."""

from typing import Any, Dict

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class ActionSequence(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Wrapper for allowing action sequences."""

    def __init__(self, env: gym.Env, sequence_length: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        self._sequence_length = sequence_length
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.is_demo_env = getattr(env, "is_demo_env", False)
        if self.is_vector_env:
            raise NotImplementedError(
                "It is not possible to use this wrapper with a VecEnv."
            )
        low, high = env.action_space.low, env.action_space.high
        self.action_space = Box(
            np.expand_dims(low, 0).repeat(sequence_length, 0),
            np.expand_dims(high, 0).repeat(sequence_length, 0),
            dtype=self.action_space.dtype,
        )

    def _step_sequence(self, action):
        total_reward = np.array(0.0)
        action_idx_reached = 0
        if self.is_demo_env:
            demo_actions = np.array(action)
        for i, sub_action in enumerate(action):
            observation, reward, termination, truncation, info = self.env.step(
                sub_action
            )
            if self.is_demo_env:
                demo_actions[i] = info.pop("demo_action")
            total_reward += reward
            action_idx_reached += 1
            if termination or truncation:
                break
        assert action_idx_reached <= self._sequence_length
        info["action_sequence_mask"] = (
            np.arange(self._sequence_length) < action_idx_reached
        ).astype(int)
        if self.is_demo_env:
            info["demo_action"] = np.array(demo_actions)
        return observation, total_reward, termination, truncation, info

    def step(self, action):
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action to be of shape {self.action_space.shape}, "
                f"but got action of shape {action.shape}."
            )
        return self._step_sequence(action)


class RecedingHorizonControl(ActionSequence):
    """Receding horizon control with temporal ensembling of ACT.

    This wrapper allows agent predict an action sequence of length N,
    but performs receding horizon control of only K <= N steps of actions.
    We also support temporal ensembling (from ALOHA https://arxiv.org/abs/2304.13705),
    which caches the previous actions and outputs a weighted average of them.
    """

    def __init__(
        self,
        env: gym.Env,
        sequence_length: int,
        time_limit: int,
        execution_length: int,
        temporal_ensemble: bool = True,
        gain: float = 0.01,
        action_smoothing_enabled: bool = False,
        action_smoothing_alpha: float = 0.0,
        action_smoothing_ignore_last_dims: int = 0,
    ):
        """Init.

        Args:
            env: The gym env to wrap.
            sequence_length: Action sequence length.
            time_limit: The time limit of the env for creating buffers.
            execution_length: The execution length of the receding horizion control.
            temporal_ensemble: Whether to use temporal ensembling. Defaults to True.
            gain: Temporal ensembling gain. Defaults to 0.01.
            action_smoothing_enabled: Whether to smooth executed actions with EMA.
            action_smoothing_alpha: Previous-action EMA weight. 0 disables smoothing.
            action_smoothing_ignore_last_dims: Number of final action dims to leave
                unsmoothed, useful for gripper commands.
        """
        super().__init__(env, sequence_length)
        if not 0.0 <= action_smoothing_alpha < 1.0:
            raise ValueError(
                "action_smoothing_alpha must be in [0, 1), got "
                f"{action_smoothing_alpha}."
            )
        if action_smoothing_ignore_last_dims < 0:
            raise ValueError(
                "action_smoothing_ignore_last_dims must be non-negative, got "
                f"{action_smoothing_ignore_last_dims}."
            )
        self._time_limit = time_limit
        self._execution_length = execution_length
        self._temporal_ensemble = temporal_ensemble
        self._gain = gain
        self._action_smoothing_enabled = action_smoothing_enabled
        self._action_smoothing_alpha = action_smoothing_alpha
        self._action_smoothing_ignore_last_dims = action_smoothing_ignore_last_dims
        self._init_action_history()

    def _init_action_history(self):
        """Initialize the action history buffer.

        We store the history actions within a buffer of shape [T, T + L, A],
        where T is the time limit, L is the sequence length, and A is the action size.

        For example, self._action_history[t, t:t + L] stores the predicted action
        sequence of size A and length L at time step t.
        """
        self._action_history = np.zeros(
            [
                self._time_limit,
                self._time_limit + self._sequence_length,
                self.action_space.shape[-1],
            ],
            dtype=self.action_space.dtype,
        )
        self._cur_step = 0
        self._last_smoothed_action = None
        self._last_requested_action = None
        self._last_executed_action = None
        self._last_execution_index = None

    def _smooth_action(self, sub_action):
        if (
            not self._action_smoothing_enabled
            or self._action_smoothing_alpha <= 0.0
        ):
            return sub_action

        current = np.asarray(sub_action, dtype=self.env.action_space.dtype)
        if self._last_smoothed_action is None:
            smoothed = current.copy()
        else:
            smoothed = (
                self._action_smoothing_alpha * self._last_smoothed_action
                + (1.0 - self._action_smoothing_alpha) * current
            )

        ignore_dims = min(self._action_smoothing_ignore_last_dims, smoothed.shape[-1])
        if ignore_dims > 0:
            smoothed[-ignore_dims:] = current[-ignore_dims:]

        smoothed = np.clip(
            smoothed,
            self.env.action_space.low,
            self.env.action_space.high,
        ).astype(self.env.action_space.dtype)
        self._last_smoothed_action = smoothed
        return smoothed

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._init_action_history()
        return super().reset(seed=seed, options=options)

    def _step_sequence(self, action):
        total_reward = np.array(0.0)
        action_idx_reached = 0
        if self.is_demo_env:
            demo_actions = np.array(action)

        self._action_history[
            self._cur_step, self._cur_step : self._cur_step + self._sequence_length
        ] = action
        for i, sub_action in enumerate(action):
            if self._temporal_ensemble and self._sequence_length > 1:
                # Select all predicted actions for self._cur_step. This will cover the
                # actions from [cur_step - sequence_length + 1, cur_step)
                # Note that not all actions in this range will be valid as we might have
                # execution_length > 1, which skips some of the intermediate steps.
                cur_actions = self._action_history[:, self._cur_step]
                indices = np.all(cur_actions != 0, axis=1)
                cur_actions = cur_actions[indices]

                # earlier predicted actions will have smaller weights.
                exp_weights = np.exp(-self._gain * np.arange(len(cur_actions)))
                exp_weights = (exp_weights / exp_weights.sum())[:, None]
                sub_action = (cur_actions * exp_weights).sum(axis=0)

            sub_action = self._smooth_action(sub_action)
            executed_sub_action = np.asarray(sub_action, dtype=np.float32).reshape(-1).copy()
            requested_sub_action = np.asarray(action[i], dtype=np.float32).reshape(-1).copy()
            self._last_requested_action = requested_sub_action.copy()
            self._last_executed_action = executed_sub_action.copy()
            self._last_execution_index = int(action_idx_reached)
            observation, reward, termination, truncation, info = self.env.step(
                sub_action
            )
            if isinstance(info, dict):
                requested_dim = min(requested_sub_action.size, executed_sub_action.size)
                requested_delta = executed_sub_action[:requested_dim] - requested_sub_action[:requested_dim]
                requested_norm = float(np.linalg.norm(requested_sub_action[:requested_dim]))
                executed_norm = float(np.linalg.norm(executed_sub_action[:requested_dim]))
                info["rhc_executed_action_available"] = True
                info["rhc_executed_action"] = executed_sub_action
                info["rhc_requested_action"] = requested_sub_action
                info["rhc_execution_index"] = int(action_idx_reached)
                info["rhc_requested_vs_executed_l2"] = float(np.linalg.norm(requested_delta))
                info["rhc_requested_vs_executed_max_abs"] = float(np.max(np.abs(requested_delta))) if requested_dim else 0.0
                info["rhc_requested_vs_executed_cosine"] = float(
                    np.dot(requested_sub_action[:requested_dim], executed_sub_action[:requested_dim])
                    / (requested_norm * executed_norm + 1e-8)
                ) if requested_dim else None
            self._cur_step += 1
            if self.is_demo_env:
                demo_actions[i] = info.pop("demo_action")
            total_reward += reward
            action_idx_reached += 1
            if termination or truncation:
                break

            if not self.is_demo_env:
                if action_idx_reached == self._execution_length:
                    break

        assert action_idx_reached <= self._sequence_length
        # TODO not sure this is correct in the case of receding horizon control
        #      Currently, for every action_sequence, all actions that are not applied
        #      will be masked out!!
        info["action_sequence_mask"] = (
            np.arange(self._sequence_length) < action_idx_reached
        ).astype(int)
        if self.is_demo_env:
            info["demo_action"] = np.array(demo_actions)
        return (
            observation,
            total_reward,
            termination,
            truncation,
            info,
        )
