from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import logging
from typing import Any, Mapping

import time
import numpy as np

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE: bool = True
except Exception:  # pragma: no cover - optional acceleration path
    jax = None
    jnp = None
    _JAX_AVAILABLE: bool = False


if _JAX_AVAILABLE:
    @jax.jit
    def _jax_project_candidate_population(
        nominal: Any,
        ctrl_samples: Any,
        action_idx: Any,
        max_delta: Any,
        low: Any,
        high: Any,
    ) -> Any:
        batch = ctrl_samples.shape[0]
        candidates = jnp.broadcast_to(nominal[None, :, :], (batch,) + nominal.shape)
        candidates = candidates.at[:, :, action_idx].set(ctrl_samples)
        nominal_ctrl = nominal[None, :, action_idx]
        delta = candidates[:, :, action_idx] - nominal_ctrl
        clipped_delta = jnp.clip(delta, -max_delta, max_delta)
        ctrl = nominal_ctrl + clipped_delta
        ctrl = jnp.clip(ctrl, low, high)
        return candidates.at[:, :, action_idx].set(ctrl)

from .safechunk_intervention_factory import InterventionExecutionFactory
from .safechunk_recovery import RecoveryContext


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeformConfig:
    """Configuration owned by the SafeChunk deformation executor."""

    mode: str = "optimized"
    deformation_enabled: bool = True
    recoverable_deform_enabled: bool = False
    explicit_return: bool = False
    safechunk_recover_enabled: bool = False
    recover_retry_cooldown_steps: int = 0
    recover_max_attempts_per_unsafe_streak: int = 0
    unsafe_deformation_fallback: str = "brake"
    commit_accepted_chunks: bool = False
    safechunk_acceptance_enabled: bool = False
    allow_candidate_fallback: bool = False
    candidate_fallback_only_if_no_optimized_result: bool = True
    optimized_fallback: str = "brake"
    chunk_deformation_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    chunk_deformation_smoothing: int = 1
    sequential_oscbf_fallback: bool = False
    deform_after_deadlock_window: bool = True
    opt_iters: int = 20
    opt_lr: float = 0.03
    opt_population: int = 32
    opt_elite_frac: float = 0.25
    opt_seed: int | None = 0
    optimizer_method: str = "cem"
    gradient_samples: int = 4
    gradient_eps: float = 0.01
    gradient_adam_beta1: float = 0.9
    gradient_adam_beta2: float = 0.999
    gradient_min_improvement: float = 1e-6
    gradient_line_search_scales: tuple[float, ...] = (1.0, 0.5, 0.25)
    gradient_batched_line_search: bool = True
    gradient_early_stop_on_candidate: bool = True
    lambda_safety: float = 100.0
    lambda_action: float = 1.0
    lambda_path: float = 1.0
    lambda_rejoin: float = 5.0
    lambda_smooth: float = 0.1
    lambda_deform_safety: float = 800.0
    lambda_deform_action: float = 0.1
    lambda_deform_smooth: float = 0.1
    lambda_retreat: float = 1.0
    jax_batched_optimizer: bool = True
    jax_batched_optimizer_fallback: bool = True
    action_low: float | None = None
    action_high: float | None = None
    max_action_delta: float | None = None
    brake_if_unrecoverable: bool = True
    inner_rejoin_metric: str = "q_state"
    final_rejoin_metric: str = "q_state"
    cache_nominal_ee: bool = False
    ee_rejoin_in_inner_loop: bool = False
    debug_safety_feasibility: bool = False
    min_rejoin_offset: int = 2
    q_rejoin_threshold: float = 0.03
    qd_rejoin_threshold: float = 0.1
    qd_rejoin_hard_threshold: float = float("inf")
    require_qd_rejoin: bool = False
    ee_rejoin_threshold: float = 0.03
    q_rejoin_weights: Any = None
    use_ee_final_check: bool = False
    deform_horizon: int = 16
    return_horizon: int = 16
    committed_execution_margin: float = 0.0
    acceptance_clearance_tol: float = 1e-6
    acceptance_hard_min_clearance: float = float("-inf")
    acceptance_desired_min_clearance: float = 0.0
    allow_safe_prefix_execution: bool = False
    min_safe_prefix_len: int = 1
    prefix_min_clearance: float = 0.0
    rolling_replan_on_prefix: bool = False
    full_horizon_required_for_recover: bool = True
    full_horizon_required_for_deform: bool = False
    emergency_brake_if_immediate_below_hard_margin: bool = True
    recover_task_progress_weight: float = 2.0

    @classmethod
    def from_parent(cls, parent: Any, **overrides: Any) -> DeformConfig:
        """Create config by reading matching attributes from a parent filter."""
        values: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            if hasattr(parent, name):
                values[name] = getattr(parent, name)
        values.update(overrides)
        return cls(**values)


class Deform(InterventionExecutionFactory):

    """Low-level SafeChunk deformation execution and optimization."""

    def __init__(
        self,
        parent: Any,
        config: DeformConfig | dict[str, Any] | None = None,
        *,
        sync: bool | None = None,
        intervention: Mapping[str, Any] | None = None,
        intervention_factory: Any | None = None,
        rng: Any | None = None,
        optimizer_warmup_done: bool = False,
        optimizer_warmup_cache: Iterable[Any] | None = None,
        optimizer_warmup_info: dict[str, Any] | None = None,
        **overrides,
    ) -> None:
        """Initialize deformation solver and warmup state for this parent filter."""
        super().__init__(
            parent,
            intervention=intervention,
            intervention_factory=intervention_factory,
        )
        del sync
        if config is None:
            config = DeformConfig.from_parent(parent, **overrides)
        else:
            config = self._coerce_config(config, **overrides)
        self._init_deform_config(config)
        if rng is None and hasattr(parent, "_rng"):
            rng = getattr(parent, "_rng")
        self._init_optimizer_state(
            rng=rng,
            optimizer_warmup_done=optimizer_warmup_done,
            optimizer_warmup_cache=optimizer_warmup_cache,
            optimizer_warmup_info=optimizer_warmup_info,
        )
        self._init_execution_state()

    @staticmethod
    def _coerce_config(config: DeformConfig | dict[str, Any], **overrides: Any) -> DeformConfig:
        """Convert generic config payloads into a typed DeformConfig."""
        field_names: set[str] = set(DeformConfig.__dataclass_fields__)
        if isinstance(config, DeformConfig):
            values: dict[str, Any] = {
                name: getattr(config, name) for name in field_names
            }
        elif hasattr(config, "items"):
            values = {
                name: value
                for name, value in dict(config.items()).items()
                if name in field_names
            }
        else:
            values = {
                name: value
                for name, value in dict(config).items()
                if name in field_names
            }
        values.update({k: v for k, v in overrides.items() if k in field_names})
        return DeformConfig(**values)

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """Coerce common array-like scalars to boolean values."""
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return False
            if value.ndim == 0:
                return bool(value.item())
            return bool(np.all(value))
        if isinstance(value, (list, tuple, set)):
            arr = np.asarray(list(value))
            if arr.size == 0:
                return False
            if arr.ndim == 0:
                return bool(arr.item())
            return bool(np.all(arr))
        try:
            return bool(value)
        except ValueError:
            arr = np.asarray(value)
            if arr.size == 0:
                return False
            return bool(np.all(arr))

    @staticmethod
    def _as_config_dict(config: Any | None) -> dict[str, Any]:
        """Normalize a config-like object into a dictionary."""
        if config is None:
            return {}
        if hasattr(config, "items"):
            return dict(config.items())
        return dict(config)

    @classmethod
    def safechunk_acceptance_config(cls, config: Any | None) -> dict[str, Any]:
        """Build safe-chunk acceptance policy defaults and toggles."""
        cfg: dict[str, Any] = cls._as_config_dict(config)
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "hard_min_clearance": float(cfg.get("hard_min_clearance", 0.02)),
            "desired_min_clearance": float(cfg.get("desired_min_clearance", 0.0)),
            "allow_safe_prefix_execution": bool(
                cfg.get("allow_safe_prefix_execution", True)
            ),
            "min_safe_prefix_len": int(cfg.get("min_safe_prefix_len", 1)),
            "prefix_min_clearance": float(cfg.get("prefix_min_clearance", 0.04)),
            "rolling_replan_on_prefix": bool(
                cfg.get("rolling_replan_on_prefix", True)
            ),
            "full_horizon_required_for_recover": bool(
                cfg.get("full_horizon_required_for_recover", True)
            ),
            "full_horizon_required_for_deform": bool(
                cfg.get("full_horizon_required_for_deform", False)
            ),
            "emergency_brake_if_immediate_below_hard_margin": bool(
                cfg.get("emergency_brake_if_immediate_below_hard_margin", True)
            ),
            "allow_fallback_path": bool(
                cfg.get(
                    "allow_fallback_path", cfg.get("allow_candidate_fallback", False)
                )
            ),
            "fallback_only_if_no_optimized_result": bool(
                cfg.get(
                    "fallback_only_if_no_optimized_result",
                    cfg.get("candidate_fallback_only_if_no_optimized_result", True),
                )
            ),
        }

    @classmethod
    def optimized_deform_config(
        cls,
        config: Any | None,
        **defaults: Any,
    ) -> dict[str, Any]:
        """Build optimized deformation defaults merged with user overrides."""
        cfg: dict[str, Any] = cls._as_config_dict(config)
        return {
            "debug_safety_feasibility": bool(
                cfg.get(
                    "debug_safety_feasibility",
                    defaults.get("debug_safety_feasibility", False),
                )
            ),
            "jax_batched_optimizer": bool(
                cfg.get(
                    "jax_batched_optimizer",
                    defaults.get("jax_batched_optimizer", True),
                )
            ),
            "jax_batched_optimizer_fallback": bool(
                cfg.get(
                    "jax_batched_optimizer_fallback",
                    defaults.get("jax_batched_optimizer_fallback", True),
                )
            ),
            "optimizer_method": str(
                cfg.get("optimizer_method", defaults.get("optimizer_method", "cem"))
            ).lower(),
            "opt_iters": max(0, int(cfg.get("opt_iters", defaults.get("opt_iters", 20))),
            ),
            "opt_lr": max(1e-9, float(cfg.get("opt_lr", defaults.get("opt_lr", 0.03))),
            ),
            "opt_population": max(
                4,
                int(cfg.get("opt_population", defaults.get("opt_population", 32))),
            ),
            "opt_elite_frac": float(
                cfg.get("opt_elite_frac", defaults.get("opt_elite_frac", 0.25))
            ),
            "opt_seed": (
                None
                if cfg.get("opt_seed", defaults.get("opt_seed", 0)) is None
                else int(cfg.get("opt_seed", defaults.get("opt_seed", 0)))
            ),
            "gradient_samples": max(
                1,
                int(cfg.get("gradient_samples", defaults.get("gradient_samples", 4))),
            ),
            "gradient_eps": max(
                1e-9,
                float(cfg.get("gradient_eps", defaults.get("gradient_eps", 0.01))),
            ),
            "gradient_adam_beta1": float(
                cfg.get("gradient_adam_beta1", defaults.get("gradient_adam_beta1", 0.9))
            ),
            "gradient_adam_beta2": float(
                cfg.get("gradient_adam_beta2", defaults.get("gradient_adam_beta2", 0.999))
            ),
            "gradient_min_improvement": max(
                0.0,
                float(
                    cfg.get(
                        "gradient_min_improvement",
                        defaults.get("gradient_min_improvement", 1e-6),
                    )
                ),
            ),
            "gradient_line_search_scales": tuple(
                float(v)
                for v in cfg.get(
                    "gradient_line_search_scales",
                    defaults.get("gradient_line_search_scales", (1.0, 0.5, 0.25)),
                )
                if float(v) > 0.0
            )
            or (1.0, 0.5, 0.25),
            "gradient_batched_line_search": bool(
                cfg.get(
                    "gradient_batched_line_search",
                    defaults.get("gradient_batched_line_search", True),
                )
            ),
            "gradient_early_stop_on_path": bool(
                cfg.get(
                    "gradient_early_stop_on_path",
                    cfg.get(
                        "gradient_early_stop_on_candidate",
                        defaults.get("gradient_early_stop_on_candidate", True),
                    ),
                )
            ),
        }

    def _init_deform_config(self, config: DeformConfig) -> None:
        """Copy normalized deformation config onto the instance."""
        gradient_line_search_scales: tuple[float, ...] = tuple(
            float(value) for value in config.gradient_line_search_scales if float(value) > 0.0
        )

        # Config is copied to attributes because hot-path code reads these often.
        self.mode: str = str(config.mode).lower()
        if self.mode != "optimized":
            raise ValueError("SafeChunkDeform mode must be \"optimized\"; candidate mode has been removed.")
        self.deformation_enabled: bool = bool(config.deformation_enabled)
        self.recoverable_deform_enabled: bool = bool(config.recoverable_deform_enabled)
        self.explicit_return: bool = bool(config.explicit_return)
        self.safechunk_recover_enabled: bool = bool(config.safechunk_recover_enabled)
        self.recover_retry_cooldown_steps: int = max(
            0, int(config.recover_retry_cooldown_steps)
        )
        self.recover_max_attempts_per_unsafe_streak: int = max(
            0,
            int(config.recover_max_attempts_per_unsafe_streak),
        )
        self.unsafe_deformation_fallback: str = str(config.unsafe_deformation_fallback)
        self.commit_accepted_chunks: bool = bool(config.commit_accepted_chunks)
        self.safechunk_acceptance_enabled: bool = bool(config.safechunk_acceptance_enabled)
        self.allow_candidate_fallback: bool = bool(config.allow_candidate_fallback)
        self.candidate_fallback_only_if_no_optimized_result: bool = bool(
            config.candidate_fallback_only_if_no_optimized_result
        )
        self.optimized_fallback: str = str(config.optimized_fallback).lower()
        if self.optimized_fallback != "brake":
            raise ValueError("SafeChunkDeform optimized_fallback must be 'brake'; candidate fallback has been removed.")
        self.chunk_deformation_scales: tuple[float, ...] = tuple(
            float(x) for x in config.chunk_deformation_scales
        )
        self.chunk_deformation_smoothing: int = max(0, int(config.chunk_deformation_smoothing))
        self.sequential_oscbf_fallback: bool = bool(config.sequential_oscbf_fallback)
        self.deform_after_deadlock_window: bool = bool(config.deform_after_deadlock_window)
        self.opt_iters: int = max(0, int(config.opt_iters))
        self.opt_lr: float = max(1e-9, float(config.opt_lr))
        self.opt_population: int = max(4, int(config.opt_population))
        self.opt_seed: int | None = config.opt_seed
        self.optimizer_method: str = str(config.optimizer_method).lower()
        if self.optimizer_method not in {"gradient", "cem"}:
            raise ValueError("SafeChunkDeform optimizer_method must be either 'gradient' or 'cem'.")
        self.gradient_samples: int = max(1, int(config.gradient_samples))
        self.gradient_eps: float = max(1e-9, float(config.gradient_eps))
        self.gradient_adam_beta1: float = float(np.clip(config.gradient_adam_beta1, 0.0, 1.0))
        self.gradient_adam_beta2: float = float(np.clip(config.gradient_adam_beta2, 0.0, 0.9999))
        self.gradient_min_improvement: float = max(0.0, float(config.gradient_min_improvement))
        self.gradient_line_search_scales: tuple[float, ...] = (
            gradient_line_search_scales
            if gradient_line_search_scales
            else (1.0, 0.5, 0.25)
        )
        self.gradient_batched_line_search: bool = bool(config.gradient_batched_line_search)
        self.gradient_early_stop_on_candidate: bool = bool(config.gradient_early_stop_on_candidate)
        self.lambda_safety: float = float(config.lambda_safety)
        self.lambda_action: float = float(config.lambda_action)
        self.lambda_path: float = float(config.lambda_path)
        self.lambda_rejoin: float = float(config.lambda_rejoin)
        self.lambda_smooth: float = float(config.lambda_smooth)
        self.lambda_deform_safety: float = float(config.lambda_deform_safety)
        self.lambda_deform_action: float = float(config.lambda_deform_action)
        self.lambda_deform_smooth: float = float(config.lambda_deform_smooth)
        self.lambda_retreat: float = float(config.lambda_retreat)
        self.jax_batched_optimizer: bool = bool(config.jax_batched_optimizer)
        self.jax_batched_optimizer_fallback: bool = bool(
            config.jax_batched_optimizer_fallback
        )
        self.action_low: float | None = config.action_low
        self.action_high: float | None = config.action_high
        self.max_action_delta: float | None = None if config.max_action_delta is None else float(config.max_action_delta)
        self.brake_if_unrecoverable: bool = bool(config.brake_if_unrecoverable)
        self.inner_rejoin_metric: str = str(config.inner_rejoin_metric)
        self.final_rejoin_metric: str = str(config.final_rejoin_metric)
        self.cache_nominal_ee: bool = bool(config.cache_nominal_ee)
        self.ee_rejoin_in_inner_loop: bool = bool(config.ee_rejoin_in_inner_loop)
        self.debug_safety_feasibility: bool = bool(config.debug_safety_feasibility)
        self.min_rejoin_offset: int = max(0, int(config.min_rejoin_offset))
        self.q_rejoin_threshold: float = float(config.q_rejoin_threshold)
        self.qd_rejoin_threshold: float = float(config.qd_rejoin_threshold)
        self.qd_rejoin_hard_threshold: float = float(config.qd_rejoin_hard_threshold)
        self.require_qd_rejoin: bool = bool(config.require_qd_rejoin)
        self.ee_rejoin_threshold: float = float(config.ee_rejoin_threshold)
        self.q_rejoin_weights: Any = config.q_rejoin_weights
        self.use_ee_final_check: bool = bool(config.use_ee_final_check)
        self.deform_horizon: int = max(1, int(config.deform_horizon))
        self.return_horizon: int = max(1, int(config.return_horizon))
        self.committed_execution_margin: float = float(config.committed_execution_margin)
        self.acceptance_clearance_tol: float = float(config.acceptance_clearance_tol)
        self.acceptance_hard_min_clearance: float = float(config.acceptance_hard_min_clearance)
        self.acceptance_desired_min_clearance: float = float(config.acceptance_desired_min_clearance)
        self.allow_safe_prefix_execution: bool = bool(config.allow_safe_prefix_execution)
        self.min_safe_prefix_len: int = int(config.min_safe_prefix_len)
        self.prefix_min_clearance: float = float(config.prefix_min_clearance)
        self.rolling_replan_on_prefix: bool = bool(config.rolling_replan_on_prefix)
        self.full_horizon_required_for_recover: bool = bool(config.full_horizon_required_for_recover)
        self.full_horizon_required_for_deform: bool = bool(config.full_horizon_required_for_deform)
        self.emergency_brake_if_immediate_below_hard_margin: bool = bool(
            config.emergency_brake_if_immediate_below_hard_margin
        )
        self.recover_task_progress_weight: float = float(config.recover_task_progress_weight)
        self.opt_elite_frac: float = float(np.clip(config.opt_elite_frac, 1.0 / self.opt_population, 1.0))

    def _init_optimizer_state(
        self,
        *,
        rng: Any | None = None,
        optimizer_warmup_done: bool = False,
        optimizer_warmup_cache: Iterable[Any] | None = None,
        optimizer_warmup_info: dict[str, Any] | None = None,
    ) -> None:
        """Initialize RNG and warmup state for optimizer-backed methods."""
        if rng is None:
            rng = np.random.default_rng(self.opt_seed)
        self._rng: Any = rng
        self._warned_jax_unavailable: bool = False
        self._optimizer_warmup_done: bool = bool(optimizer_warmup_done)
        self._optimizer_warmup_cache: set[Any] = set() if optimizer_warmup_cache is None else set(
            optimizer_warmup_cache
        )
        self._optimizer_warmup_info: dict[str, Any] = {} if optimizer_warmup_info is None else dict(
            optimizer_warmup_info
        )

    def _sync_config(self) -> None:
        """Deprecated compatibility hook; config is owned by the constructor."""
        return None

    def _init_execution_state(self) -> None:
        """Reset mutable deformation counters used for stats and tracing."""
        self.current_deform_plan: Any | None = None
        self.deform_anchor_state: Any | None = None
        self.deform_replan_count: int = 0
        self.optimized_attempt_count: int = 0
        self.optimized_solution_count: int = 0
        self.fallback_attempt_count: int = 0
        self.fallback_attempt_accepted_count: int = 0
        self.optimized_rejected_count: int = 0
        self.deform_option_attempt_count: int = 0
        self.deform_accepted_count: int = 0
        self.deform_rejected_count: int = 0
        self.contact_during_deform_count: int = 0
        self.recover_option_attempt_count: int = 0
        self.recover_accepted_count: int = 0
        self.recover_rejected_count: int = 0
        self.safe_prefix_accepted_count: int = 0
        self.first_action_only_accepted_count: int = 0
        self.immediate_hard_reject_count: int = 0
        self.no_safe_prefix_reject_count: int = 0
        self.horizon_margin_reject_count: int = 0
        self.accepted_deform_steps: int = 0
        self.accepted_recover_steps: int = 0
        self.fallback_brake_after_reject_count: int = 0

    def reset_execution_state(self) -> None:
        """Clear runtime state and restart the optimizer stream for this episode."""
        self._init_execution_state()
        self._rng = np.random.default_rng(self.opt_seed)

    def deform_chunk(
        self,
        obs: Any,
        action_chunk: Any,
        safety_info: dict[str, Any] | None = None,
        braked_chunk: Any | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run optimized SafeChunk deformation."""
        if self.mode != "optimized":
            raise ValueError("SafeChunkDeform mode must be \"optimized\"; candidate mode has been removed.")
        return self._deform_chunk_optimized_with_fallback(
            obs,
            action_chunk,
            safety_info=safety_info,
            braked_chunk=braked_chunk,
            **kwargs,
        )

    def deform_chunk_candidate(
        self,
        obs: Any,
        action_chunk: Any,
        safety_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Deform the controlled chunk trajectory with fixed candidate scales."""
        """Deform the controlled chunk trajectory using fixed suffix scales.

        This is the original SafeChunk-Deform candidate search. It is intentionally
        derivative-free: with only scalar clearance feedback available, it generates
        smooth whole-chunk candidates by retracting the unsafe suffix toward a
        no-motion anchor and selects the first safe, least distorted candidate.
        """
        chunk, _ = self._as_chunk(action_chunk)
        safety_info = safety_info or {}
        candidates = self._make_chunk_deformation_candidates(obs, chunk, safety_info)

        best_chunk = chunk.copy()
        best_eval = None
        best_norm = float("inf")
        best_scale = None
        best_safe = False

        for scale, candidate in candidates:
            q_seq = self.rollout_nominal_chunk(obs, candidate)
            candidate_eval = self.evaluate_horizon_safety(obs, q_seq)
            candidate_norm = self._controlled_deformation_norm(candidate, chunk)
            candidate_progress = self._controlled_progress_retention(
                candidate, chunk, obs
            )
            candidate_safe = bool(candidate_eval["horizon_safe"])
            candidate_eval["task_progress_retention"] = candidate_progress

            if self._is_better_deformation_candidate(
                candidate_eval,
                candidate_norm,
                candidate_safe,
                best_eval,
                best_norm,
                best_safe,
            ):
                best_chunk = candidate
                best_eval = candidate_eval
                best_norm = candidate_norm
                best_scale = scale
                best_safe = candidate_safe

        if best_eval is None:
            q_seq = self.rollout_nominal_chunk(obs, best_chunk)
            best_eval = self.evaluate_horizon_safety(obs, q_seq)
            best_norm = self._controlled_deformation_norm(best_chunk, chunk)
            best_eval["task_progress_retention"] = self._controlled_progress_retention(
                best_chunk, chunk, obs
            )

        info = {
            "deform_mode": "deform_safe_prefix",
            "deform_safe": bool(best_eval["horizon_safe"]),
            "deform_min_clearance": float(best_eval["min_clearance"]),
            "deformation_norm": float(best_norm),
            "deformation_source": "chunk_deform",
            "chunk_deform_scale": best_scale,
            "chunk_deform_attempts": len(candidates),
            "task_progress_retention": float(
                best_eval.get("task_progress_retention", 1.0)
            ),
        }

        if (
            not info["deform_safe"]
            and self.sequential_oscbf_fallback
            and callable(self._get_oscbf_operator())
        ):
            fallback_chunk, fallback_info = self.deform_chunk_with_oscbf(
                obs, chunk, **kwargs
            )
            fallback_info["deform_mode"] = "deform_safe_prefix"
            fallback_info["deformation_source"] = "sequential_oscbf_fallback"
            return fallback_chunk, fallback_info

        return best_chunk, info

    def _rolling_prefix_candidate_fallback(
        self,
        obs: Any,
        chunk: np.ndarray,
        optimized_chunk: Any,
        optimized_info: dict[str, Any] | None,
        safety_info: dict[str, Any] | None = None,
        braked_chunk: Any | None = None,
        candidate_type: str = "deform",
        **kwargs: Any,
    ) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        """Try safe-prefix alternatives after the optimized candidate is rejected."""
        valid = self._valid_control_indices(chunk)
        action_idx = self.controlled_action_indices[valid]
        candidates: list[tuple[str, np.ndarray]] = []

        def add_candidate(name: str, candidate: Any) -> None:
            if candidate is None:
                return
            arr, _ = self._as_chunk(candidate)
            if arr.shape == chunk.shape:
                candidates.append((name, arr.copy()))

        add_candidate("optimized", optimized_chunk)
        add_candidate("nominal", chunk)
        add_candidate("horizon_brake", braked_chunk)
        for scale, candidate in self._make_chunk_deformation_candidates(obs, chunk, safety_info or {}):
            add_candidate(f"scaled_deform_{scale}", candidate)
        if callable(self._get_oscbf_operator()):
            try:
                seq_chunk, _seq_info = self.deform_chunk_with_oscbf(obs, chunk, **kwargs)
                add_candidate("sequential_oscbf", seq_chunk)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sequential OSCBF fallback candidate failed: %s", exc)
        try:
            q_current = self.extract_current_q(obs, chunk)
            recover_seed, _ = self._make_task_progress_recover_chunk(
                obs,
                q_current,
                chunk,
                action_idx,
            )
            if recover_seed.shape[0] < chunk.shape[0]:
                padded = chunk.copy()
                padded[: recover_seed.shape[0]] = recover_seed
                padded[recover_seed.shape[0] :] = recover_seed[-1]
                recover_seed = padded
            add_candidate("recover_to_task_progress", recover_seed)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Recover-to-task-progress candidate failed: %s", exc)
        hold = chunk.copy()
        if hold.shape[0]:
            q = self.extract_current_q(obs, chunk)
            valid_hold = (
                (self.controlled_action_indices < hold.shape[1])
                & (self.controlled_state_indices < q.shape[0])
            )
            if np.any(valid_hold):
                action_idx = self.controlled_action_indices[valid_hold]
                state_idx = self.controlled_state_indices[valid_hold]
                anchor = self._controlled_anchor(obs, hold, action_idx, state_idx)
                hold[:, action_idx] = anchor[None, :]
            else:
                hold[1:] = hold[0]
        add_candidate("hold", hold)

        best: tuple[float, str, np.ndarray, dict[str, Any], dict[str, Any]] | None = None
        rejected: list[Any] = []
        for name, candidate in candidates:
            acceptance = self.evaluate_candidate_acceptance(obs, candidate, candidate_type)
            self.fallback_attempt_count += 1
            if candidate_type == "recover":
                self.recover_option_attempt_count += 1
            else:
                self.deform_option_attempt_count += 1
            if not acceptance["accepted"]:
                rejected.append(acceptance.get("rejection_reason"))
                continue
            score, score_info = self._score_accepted_candidate(
                obs,
                candidate,
                chunk,
                acceptance,
                candidate_type=candidate_type,
            )
            if best is None or score > best[0]:
                best = (score, name, candidate, acceptance, score_info)

        if best is None:
            if any(reason == "immediate_below_hard_margin" for reason in rejected):
                self.immediate_hard_reject_count += 1
            elif any(reason == "no_safe_prefix" for reason in rejected):
                self.no_safe_prefix_reject_count += 1
            elif any(reason == "horizon_below_desired_margin" for reason in rejected):
                self.horizon_margin_reject_count += 1
            return None, None

        _score, name, candidate, acceptance, score_info = best
        accepted_chunk = self._truncate_chunk_to_safe_prefix(candidate, acceptance)
        self.fallback_attempt_accepted_count += 1
        if acceptance["acceptance_type"] == "safe_prefix":
            self.safe_prefix_accepted_count += 1
        elif acceptance["acceptance_type"] == "first_action_only":
            self.first_action_only_accepted_count += 1
        if candidate_type == "recover":
            self.recover_accepted_count += 1
            self.accepted_recover_steps += 1
        else:
            self.deform_accepted_count += 1
            self.accepted_deform_steps += 1
        info = dict(optimized_info or {})
        mode = "recover_safe_prefix" if candidate_type == "recover" else "deform_safe_prefix"
        if acceptance["acceptance_type"] == "full_horizon":
            mode = "recover" if candidate_type == "recover" else "horizon_deform"
        info.update(
            {
                "optimized_accepted": True,
                "optimized_fallback": "safe_prefix_fallback",
                "optimized_reject_reason": None,
                "fallback_used": False,
                "deform_safe": True,
                "is_safe": True,
                "is_recoverable": True if self.recoverable_deform_enabled else None,
                "safety_rejected": False,
                "recovery_rejected": False,
                "rejection_cause": None,
                "mode": mode,
                "deform_mode": mode,
                "recovery_mode": mode,
                "deformation_source": name,
                "path_count": len(candidates),
                "accepted_path_name": name,
                "accepted_path_type": candidate_type,
                "acceptance_type": acceptance["acceptance_type"],
                "safe_prefix_len": acceptance["safe_prefix_len"],
                "immediate_clearance": acceptance["immediate_clearance"],
                "prefix_min_clearance": acceptance["prefix_min_clearance"],
                "horizon_min_clearance": acceptance["horizon_min_clearance"],
                "rejection_reason": None,
                "full_horizon_required": acceptance["full_horizon_required"],
                "rolling_replan_on_prefix": acceptance["rolling_replan_on_prefix"],
                "safe_prefix_execution": acceptance["safe_prefix_execution"],
                "progress_score": score_info["progress_score"],
                "progress_score_available": score_info["progress_score_available"],
                "deformation_norm": score_info["deformation_norm"],
                "recover_projection_on_nominal": score_info.get("recover_projection_on_nominal"),
                "recover_cosine_to_nominal": score_info.get("recover_cosine_to_nominal"),
                "nominal_rejoin_score": score_info.get("nominal_rejoin_score"),
                "nominal_rejoin_available": score_info.get("nominal_rejoin_available"),
                "nominal_rejoin_suppressed_reason": score_info.get("nominal_rejoin_suppressed_reason"),
                "nominal_rejoin_clearance": score_info.get("nominal_rejoin_clearance"),
                "nominal_rejoin_safe_prefix_len": score_info.get("nominal_rejoin_safe_prefix_len"),
                "recover_task_progress_score": score_info.get("recover_task_progress_score"),
                "recover_score_total": score_info.get("recover_score_total"),
                "recover_rejoin_weight_effective": score_info.get("recover_rejoin_weight_effective"),
                "recover_step_since_deform": score_info.get("recover_step_since_deform"),
                "deform_min_clearance": acceptance["immediate_clearance"],
                "min_clearance": acceptance["immediate_clearance"],
            }
        )
        if candidate_type == "recover":
            info["recover_accepted"] = True
        else:
            info["deform_stage_accepted"] = True
        info.update(self._safechunk_replan_info())
        return accepted_chunk, info

    def _deform_chunk_optimized_with_fallback(
        self,
        obs: Any,
        action_chunk: Any,
        safety_info: dict[str, Any] | None = None,
        braked_chunk: Any | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run optimized deformation and apply configured brake/candidate fallback."""
        chunk, _ = self._as_chunk(action_chunk)
        nominal_q_seq = kwargs.pop("nominal_q_seq", None)
        first_violation = None if safety_info is None else safety_info.get("first_violation")
        try:
            optimized_chunk, optimized_info = self.deform_chunk_optimized(
                nominal_chunk=chunk,
                obs=obs,
                first_violation=first_violation,
                nominal_q_seq=nominal_q_seq,
                safety_info=safety_info,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - defensive integration path
            logger.warning("Optimized SafeChunk deformation failed: %s", exc)
            optimized_chunk = chunk.copy()
            optimized_info = self._optimized_failure_info(str(exc))

        candidate_type = "recover" if self.recoverable_deform_enabled and self.explicit_return else "deform"
        if self.safechunk_acceptance_enabled:
            optimized_acceptance = self.evaluate_candidate_acceptance(
                obs,
                optimized_chunk,
                candidate_type,
            )
        else:
            old_style_accepted = bool(
                optimized_info.get("deform_safe", False)
                and (
                    not self.recoverable_deform_enabled
                    or optimized_info.get("is_recoverable", False)
                )
            )
            optimized_acceptance = {
                "accepted": old_style_accepted,
                "acceptance_type": "full_horizon" if old_style_accepted else "rejected",
                "safe_prefix_len": optimized_chunk.shape[0],
                "immediate_clearance": float(optimized_info.get("min_clearance", float("-inf"))),
                "prefix_min_clearance": float(self.prefix_min_clearance),
                "horizon_min_clearance": float(optimized_info.get("min_clearance", float("-inf"))),
                "rejection_reason": None if old_style_accepted else "horizon_below_desired_margin",
                "full_horizon_required": True,
                "rolling_replan_on_prefix": False,
                "safe_prefix_execution": False,
            }
        self.optimized_solution_count += 1
        self.optimized_attempt_count += 1
        if candidate_type == "recover":
            self.recover_option_attempt_count += 1
        else:
            self.deform_option_attempt_count += 1
        optimized_info.update(
            {
                "path_count": 1,
                "accepted_path_name": "optimized",
                "accepted_path_type": candidate_type,
                "acceptance_type": optimized_acceptance["acceptance_type"],
                "safe_prefix_len": optimized_acceptance["safe_prefix_len"],
                "immediate_clearance": optimized_acceptance["immediate_clearance"],
                "prefix_min_clearance": optimized_acceptance["prefix_min_clearance"],
                "horizon_min_clearance": optimized_acceptance["horizon_min_clearance"],
                "rejection_reason": optimized_acceptance["rejection_reason"],
                "full_horizon_required": optimized_acceptance["full_horizon_required"],
                "rolling_replan_on_prefix": optimized_acceptance["rolling_replan_on_prefix"],
                "safe_prefix_execution": optimized_acceptance["safe_prefix_execution"],
            }
        )
        accepted = bool(
            optimized_acceptance.get("accepted", False)
            and (
                optimized_acceptance["acceptance_type"] != "full_horizon"
                or (
                    optimized_info.get("deform_safe", False)
                    and (
                        not self.recoverable_deform_enabled
                        or optimized_info.get("is_recoverable", False)
                    )
                )
            )
        )
        if accepted and optimized_acceptance["acceptance_type"] != "full_horizon":
            optimized_chunk = self._truncate_chunk_to_safe_prefix(
                optimized_chunk,
                optimized_acceptance,
            )
            optimized_info.update(
                {
                    "deform_safe": True,
                    "is_safe": True,
                    "is_recoverable": True if self.recoverable_deform_enabled else None,
                    "safety_rejected": False,
                    "recovery_rejected": False,
                    "rejection_cause": None,
                    "deform_min_clearance": optimized_acceptance["immediate_clearance"],
                    "min_clearance": optimized_acceptance["immediate_clearance"],
                    "recover_accepted": candidate_type == "recover",
                    "deform_stage_accepted": candidate_type == "deform",
                    "mode": (
                        "recover_safe_prefix"
                        if candidate_type == "recover"
                        else "deform_safe_prefix"
                    ),
                    "deform_mode": (
                        "recover_safe_prefix"
                        if candidate_type == "recover"
                        else "deform_safe_prefix"
                    ),
                    "recovery_mode": (
                        "recover_safe_prefix"
                        if candidate_type == "recover"
                        else "deform_safe_prefix"
                    ),
                }
            )
        if (
            accepted
            and optimized_acceptance.get("acceptance_type") == "full_horizon"
            and self.recoverable_deform_enabled
            and self.explicit_return
            and self.commit_accepted_chunks
        ):
            return_min_clearance = optimized_info.get(
                "recover_min_clearance",
                optimized_info.get("return_min_clearance"),
            )
            required_return_clearance = float(
                self.min_clearance + self.committed_execution_margin
            )
            optimized_info["committed_execution_margin"] = float(
                self.committed_execution_margin
            )
            optimized_info["committed_return_acceptance_clearance"] = (
                required_return_clearance
            )
            if (
                return_min_clearance is None
                or float(return_min_clearance) < required_return_clearance
            ):
                accepted = False
                optimized_info.update(
                    {
                        "recover_corridor_accepted": bool(
                            optimized_info.get(
                                "recover_corridor_accepted",
                                optimized_info.get(
                                    "return_accepted",
                                    optimized_info.get("recover_accepted", False),
                                ),
                            )
                        ),
                        "recover_accepted": False,
                        "return_accepted": False,
                        "deform_safe": False,
                        "is_safe": False,
                        "is_recoverable": False,
                        "safety_rejected": True,
                        "recovery_rejected": True,
                        "committed_margin_rejected": True,
                        "rejection_cause": "unsafe",
                        "clearance_gap": float(
                            required_return_clearance
                            - float(return_min_clearance if return_min_clearance is not None else float("-inf"))
                        ),
                    }
                )

        if accepted:
            if optimized_acceptance["acceptance_type"] == "safe_prefix":
                self.safe_prefix_accepted_count += 1
            elif optimized_acceptance["acceptance_type"] == "first_action_only":
                self.first_action_only_accepted_count += 1
            if candidate_type == "recover":
                self.recover_accepted_count += 1
                self.accepted_recover_steps += 1
            else:
                self.deform_accepted_count += 1
                self.accepted_deform_steps += 1
            optimized_info["optimized_accepted"] = True
            optimized_info["fallback_used"] = False
            if candidate_type == "recover":
                optimized_info["recover_corridor_accepted"] = bool(
                    optimized_info.get(
                        "recover_corridor_accepted",
                        optimized_info.get(
                            "return_accepted",
                            optimized_info.get("recover_accepted", True),
                        ),
                    )
                )
                optimized_info["recover_accepted"] = True
                optimized_info["return_accepted"] = True
                optimized_info["recovery_rejected"] = False
                optimized_info["is_recoverable"] = True
            optimized_info.update(self._safechunk_replan_info())
            if self.recoverable_deform_enabled and self.explicit_return:
                optimized_info.setdefault("mode", "recover")
                optimized_info.setdefault("deform_mode", "recover")
                optimized_info["deformation_source"] = "explicit_recover_deform"
                optimized_info.setdefault("recovery_mode", "resume_act")
                optimized_info["act_resume_index"] = optimized_info.get(
                    "return_target_index",
                    optimized_info.get("rejoin_index"),
                )
                optimized_info["act_resume_supported"] = False
                optimized_info.setdefault(
                    "act_resume_note",
                    "TODO: executor replans each outer step; use recover_target_index "
                    "when a persistent ACT chunk cursor is added.",
                )
            elif self.recoverable_deform_enabled:
                optimized_info["mode"] = "optimized_recoverable_deform"
                optimized_info["deform_mode"] = "optimized_recoverable_deform"
                optimized_info["deformation_source"] = "optimized_recoverable_deform"
                optimized_info["recovery_mode"] = "optimized_recoverable_deform"
                optimized_info["act_resume_index"] = optimized_info.get("rejoin_index")
                optimized_info["act_resume_supported"] = False
                optimized_info["act_resume_note"] = (
                    "TODO: executor replans each outer step; use rejoin_index "
                    "when a persistent ACT chunk cursor is added."
                )
            return optimized_chunk, optimized_info

        if not accepted:
            self.optimized_rejected_count += 1
            if candidate_type == "recover":
                self.recover_rejected_count += 1
            else:
                self.deform_rejected_count += 1
            reason = optimized_acceptance.get("rejection_reason")
            if reason == "immediate_below_hard_margin":
                self.immediate_hard_reject_count += 1
            elif reason == "no_safe_prefix":
                self.no_safe_prefix_reject_count += 1
            elif reason == "horizon_below_desired_margin":
                self.horizon_margin_reject_count += 1
        reject_reason = self._optimized_reject_reason(optimized_info)
        candidate_chunk, candidate_info = (None, None)
        should_try_candidate_fallback = (
            self.safechunk_acceptance_enabled
            and self.allow_candidate_fallback
            and (
                not self.candidate_fallback_only_if_no_optimized_result
                or self.optimized_solution_count == 0
            )
        )
        if should_try_candidate_fallback:
            candidate_chunk, candidate_info = self._rolling_prefix_candidate_fallback(
                obs,
                chunk,
                optimized_chunk,
                optimized_info,
                safety_info=safety_info,
                braked_chunk=braked_chunk,
                candidate_type=candidate_type,
                **kwargs,
            )
        if candidate_info is not None:
            return candidate_chunk, candidate_info
        self.fallback_brake_after_reject_count += 1
        info = dict(optimized_info)
        info.update(self._safechunk_replan_info())
        info.update(
            {
                "deform_safe": False,
                "optimized_accepted": False,
                "optimized_fallback": "brake",
                "optimized_reject_reason": reject_reason,
                "fallback_reason": reject_reason,
                "fallback_used": True,
            }
        )
        if candidate_type == "recover":
            recover_corridor_accepted = bool(
                info.get(
                    "recover_corridor_accepted",
                    info.get("return_accepted", info.get("recover_accepted", False)),
                )
            )
            info.update(
                {
                    "recover_corridor_accepted": recover_corridor_accepted,
                    "recover_accepted": False,
                    "return_accepted": False,
                    "is_recoverable": False,
                    "recovery_rejected": True,
                }
            )
        if braked_chunk is None:
            braked_chunk = chunk.copy()
        return braked_chunk, info

    def deform_chunk_optimized(
        self,
        nominal_chunk: Any,
        q_current: Any | None = None,
        qd_current: Any | None = None,
        first_violation: int | None = None,
        nominal_q_seq: Any | None = None,
        nominal_ee_seq: Any | None = None,
        human_state: Any | None = None,
        obs: Any | None = None,
        safety_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Optimize a whole action chunk with a CEM gradient-free objective.

        The current rollout and horizon-clearance path is NumPy/JAX based, so this
        uses the requested differentiability fallback instead of pretending the
        computation is a PyTorch differentiable graph.
        """
        chunk, _ = self._as_chunk(nominal_chunk)
        obs = self._obs_with_q(obs, q_current)
        if nominal_q_seq is None:
            nominal_q_seq = self.rollout_chunk(chunk, q_current, qd_current, obs=obs)
        else:
            nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        if safety_info is None:
            safety_info = self.evaluate_horizon_safety(obs, nominal_q_seq)
        if first_violation is None:
            first_violation = safety_info.get("first_violation")

        if self.recoverable_deform_enabled and self.explicit_return:
            return self.parent.recovery.deform_chunk_optimized_explicit_return(
                nominal_chunk=chunk,
                obs=obs,
                first_violation=first_violation,
                nominal_q_seq=nominal_q_seq,
                nominal_ee_seq=nominal_ee_seq,
                human_state=human_state,
                safety_info=safety_info,
                **kwargs,
            )

        rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)

        valid = self._valid_control_indices(chunk)
        if not np.any(valid):
            info = self._optimized_final_info(
                obs,
                chunk,
                chunk,
                nominal_q_seq,
                nominal_ee_seq,
                human_state,
                j_best=None,
                rejoin_loss=float("inf"),
                losses={},
                rejoin_context=rejoin_context,
            )
            return chunk.copy(), info

        action_idx = self.controlled_action_indices[valid]
        seed_chunks = self._optimized_seed_chunks(obs, chunk, safety_info)

        def cost_fn(candidate: np.ndarray) -> tuple[float, dict[str, Any]]:
            return self._optimized_deformation_cost(
                obs,
                candidate,
                chunk,
                nominal_q_seq,
                nominal_ee_seq,
                human_state,
                rejoin_context=rejoin_context,
            )

        def batch_cost_fn(candidates: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
            return self._optimized_deformation_cost_batch(
                obs,
                candidates,
                chunk,
                nominal_q_seq,
                nominal_ee_seq,
                human_state,
                rejoin_context=rejoin_context,
            )

        best_record = self._optimize_controlled_chunk(
            obs,
            chunk,
            action_idx,
            cost_fn,
            seed_chunks=seed_chunks,
            batch_cost_fn=batch_cost_fn,
            optimizer_stage="deform",
        )
        best_chunk = best_record["chunk"]
        info = self._optimized_final_info(
            obs,
            best_chunk,
            chunk,
            nominal_q_seq,
            nominal_ee_seq,
            human_state,
            j_best=best_record["losses"].get("j_best"),
            rejoin_loss=best_record["losses"].get("rejoin_loss", float("inf")),
            losses=best_record["losses"],
            rejoin_context=rejoin_context,
        )
        return best_chunk, info

    def deform_chunk_with_oscbf(
        self,
        obs: Any,
        action_chunk: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Filter a full chunk through a batched OSCBF method or per-step fallback."""
        chunk, _ = self._as_chunk(action_chunk)
        safe_chunk = chunk.copy()
        op = self._get_oscbf_operator()
        batch_filter_info: dict[str, Any] = {}
        batch_filter_t0 = time.perf_counter()
        if op is not None:
            for method_name in (
                "filter_chunk",
                "filter_action_chunk",
            ):
                method = getattr(op, method_name, None)
                if method is None:
                    continue
                try:
                    result = self._call_oscbf_chunk_method(method, obs, chunk, **kwargs)
                    if isinstance(result, tuple):
                        candidate, candidate_info = result
                    else:
                        candidate, candidate_info = result, {}
                    candidate = np.asarray(candidate, dtype=chunk.dtype)
                    if candidate.shape != chunk.shape:
                        raise ValueError(
                            "Chunk safety operator returned shape "
                            f"{candidate.shape}, expected {chunk.shape}"
                        )
                    safe_chunk = chunk.copy()
                    safe_chunk[:, self.controlled_action_indices] = candidate[
                        :, self.controlled_action_indices
                    ]
                    batch_filter_info = dict(candidate_info or {})
                    batch_filter_info.update(
                        {
                            "sequential_oscbf_batched": True,
                            "sequential_oscbf_batch_method": method_name,
                            "sequential_oscbf_batch_filter_time_ms": float(
                                1000.0 * (time.perf_counter() - batch_filter_t0)
                            ),
                        }
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Batched sequential OSCBF via %s failed; using per-step path: %s",
                        method_name,
                        exc,
                    )
        if callable(op) and not batch_filter_info:
            for k, action in enumerate(chunk):
                safe_action = self._call_single_step_operator(action, obs, **kwargs)
                safe_action = np.asarray(safe_action, dtype=chunk.dtype).reshape(-1)
                if safe_action.shape[0] != chunk.shape[1]:
                    raise ValueError(
                        "Single-step safety operator returned shape "
                        f"{safe_action.shape}, expected ({chunk.shape[1]},)"
                    )
                safe_chunk[k, self.controlled_action_indices] = safe_action[
                    self.controlled_action_indices
                ]
            batch_filter_info = {
                "sequential_oscbf_batched": False,
                "sequential_oscbf_batch_method": None,
                "sequential_oscbf_batch_filter_time_ms": float(
                    1000.0 * (time.perf_counter() - batch_filter_t0)
                ),
            }

        delta = (
            safe_chunk[:, self.controlled_action_indices]
            - chunk[:, self.controlled_action_indices]
        )
        deformation_norm = float(np.mean(np.linalg.norm(delta, axis=1)))
        q_seq = self.rollout_nominal_chunk(obs, safe_chunk)
        deform_safety = self.evaluate_horizon_safety(obs, q_seq)
        info = dict(deform_safety)
        info.update(
            {
                "deform_safe": bool(deform_safety["horizon_safe"]),
                "deform_min_clearance": float(deform_safety["min_clearance"]),
                "deformation_norm": deformation_norm,
            }
        )
        info.update(batch_filter_info)
        return safe_chunk, info

    def _score_accepted_candidate(
        self,
        obs: Any,
        candidate: np.ndarray,
        nominal: np.ndarray,
        acceptance: dict[str, Any],
        candidate_type: str = "deform",
    ) -> tuple[float, dict[str, Any]]:
        """Rank accepted fallback candidates by safety, progress, and distortion."""

        progress_score, progress_available = self._candidate_progress_score(obs, candidate)
        deformation_norm = self._controlled_deformation_norm(candidate, nominal)
        recover_extra: dict[str, Any] = {}
        task_weight = 2.0
        nominal_rejoin_score = 0.0
        rejoin_weight = 0.0
        if candidate_type == "recover" and self.safechunk_recover_enabled:
            target_info, rejoin_info, progress_score, progress_available, rejoin_weight = (
                self._recover_nominal_rejoin_terms(obs, candidate, record=True)
            )
            nominal_rejoin_score = float(rejoin_info.get("nominal_rejoin_score", 0.0))
            recover_extra = {
                "recover_task_progress_score": float(progress_score),
                "recover_rejoin_weight_effective": float(rejoin_weight),
                "recover_step_since_deform": int(self.recover_step_since_deform),
                "nominal_rejoin_available": bool(target_info.get("available")),
                "nominal_rejoin_suppressed_reason": target_info.get("suppressed_reason"),
                "nominal_rejoin_clearance": float(target_info.get("nominal_rejoin_clearance", float("-inf"))),
                "nominal_rejoin_safe_prefix_len": int(target_info.get("safe_prefix_len", 0) or 0),
                **rejoin_info,
            }
            task_weight = float(self.recover_task_progress_weight)
        score = (
            1000.0 * float(bool(acceptance.get("accepted")))
            + 10.0 * float(acceptance.get("safe_prefix_len", 0) or 0)
            + 5.0 * float(acceptance.get("immediate_clearance", 0.0) or 0.0)
            + task_weight * progress_score
            + rejoin_weight * nominal_rejoin_score
            - 0.1 * deformation_norm
        )
        if candidate_type == "recover" and progress_score <= 0.0 and nominal_rejoin_score <= 0.0:
            score -= 5.0
        recover_extra["recover_score_total"] = float(score)
        return float(score), {
            "progress_score": float(progress_score),
            "progress_score_available": bool(progress_available),
            "deformation_norm": float(deformation_norm),
            **recover_extra,
        }

    def _optimized_deformation_cost(
        self,
        obs: Any,
        candidate: np.ndarray,
        nominal: np.ndarray,
        nominal_q_seq: Any,
        nominal_ee_seq: Any,
        human_state: Any,
        rejoin_context: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Compute scalar objective and diagnostics for one optimized chunk."""
        del human_state
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)
        q_seq = self.rollout_nominal_chunk(obs, candidate)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        valid = self._valid_control_indices(nominal)
        action_idx = self.controlled_action_indices[valid]

        safety_loss = float(np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum())
        action_deviation_loss = float(
            np.square(candidate[:, action_idx] - nominal[:, action_idx]).mean()
        ) if action_idx.size else 0.0
        path_loss = self.nominal_path_deviation_loss(q_seq, nominal_q_seq)
        rejoin_loss = 0.0
        j_best = None
        rejoin_space = None
        rejoin_q_eval_time_ms = 0.0
        if self.recoverable_deform_enabled:
            if self.inner_rejoin_metric == "ee_pose":
                (
                    rejoin_loss,
                    j_best,
                    rejoin_q_eval_time_ms,
                    _ee_available,
                ) = self._ee_rejoin_loss(
                    q_seq,
                    nominal_q_seq=nominal_q_seq,
                    nominal_ee_seq=nominal_ee_seq,
                    rejoin_context=rejoin_context,
                )
                rejoin_space = "ee_pose"
            else:
                rejoin_loss, j_best, rejoin_q_eval_time_ms = self._q_rejoin_loss(
                    q_seq,
                    nominal_q_seq=nominal_q_seq,
                    rejoin_context=rejoin_context,
                )
                rejoin_space = "q_state"
        smoothness_loss = self._smoothness_loss(candidate, action_idx)
        existing_optimization_loss = float(
            self.lambda_safety * safety_loss
            + self.lambda_action * action_deviation_loss
            + self.lambda_path * path_loss
            + self.lambda_smooth * smoothness_loss
        )
        total_loss = existing_optimization_loss
        if self.recoverable_deform_enabled:
            total_loss = float(total_loss + self.lambda_rejoin * rejoin_loss)
        losses: dict[str, Any] = {
            "safety_loss": safety_loss,
            "action_deviation_loss": action_deviation_loss,
            "path_loss": path_loss,
            "existing_optimization_loss": existing_optimization_loss,
            "rejoin_loss": rejoin_loss,
            "smoothness_loss": smoothness_loss,
            "total_loss": total_loss,
            "min_clearance": float(np.min(h_seq)),
            "j_best": j_best,
            "rejoin_space": rejoin_space,
            "inner_rejoin_metric": self.inner_rejoin_metric,
            "final_rejoin_metric": self.final_rejoin_metric,
            "rejoin_q_eval_time_ms": float(rejoin_q_eval_time_ms),
            "ee_nom_cache_time_ms": float(
                rejoin_context.get("ee_nom_cache_time_ms", 0.0)
            ),
            "ee_final_check_time_ms": 0.0,
        }
        if rejoin_space == "q_state":
            losses["q_rejoin_loss"] = rejoin_loss
            losses["q_rejoin_dist"] = self._sqrt_loss(rejoin_loss)
            losses["q_rejoin_index"] = j_best
        return total_loss, losses

    def _optimized_deformation_cost_batch(
        self,
        obs: Any,
        candidates: Any,
        nominal: np.ndarray,
        nominal_q_seq: Any,
        nominal_ee_seq: Any,
        human_state: Any,
        rejoin_context: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Vectorized objective for CEM populations; mirrors single-candidate cost."""
        del human_state
        if self.recoverable_deform_enabled and self.inner_rejoin_metric == "ee_pose":
            raise NotImplementedError("Batched EE-pose rejoin is not implemented")
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)
        candidates = np.asarray(candidates, dtype=np.float32)
        rollout_t0 = time.perf_counter()
        q_seq_batch = self.rollout_nominal_chunk_batch(obs, candidates)
        rollout_time_ms = 1000.0 * (time.perf_counter() - rollout_t0)
        safety_eval = self.evaluate_horizon_safety_batch(obs, q_seq_batch)
        h_seq = self._clearance_sequence_batch_from_eval(
            safety_eval,
            candidates.shape[0],
            candidates.shape[1],
        )
        valid = self._valid_control_indices(nominal)
        action_idx = self.controlled_action_indices[valid]
        safety_loss = np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum(axis=1)
        if action_idx.size:
            action_deviation_loss = np.square(
                candidates[:, :, action_idx] - nominal[None, :, action_idx]
            ).mean(axis=(1, 2))
        else:
            action_deviation_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        path_loss = self.nominal_path_deviation_loss_batch(q_seq_batch, nominal_q_seq)
        smoothness_loss = self._smoothness_loss_batch(candidates, action_idx)
        rejoin_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        j_best = [None] * candidates.shape[0]
        rejoin_q_eval_time_ms = 0.0
        rejoin_space = None
        if self.recoverable_deform_enabled:
            rejoin_loss, j_best, rejoin_q_eval_time_ms = self._q_rejoin_loss_batch(
                q_seq_batch,
                nominal_q_seq=nominal_q_seq,
                rejoin_context=rejoin_context,
            )
            rejoin_space = "q_state"
        existing_optimization_loss = (
            self.lambda_safety * safety_loss
            + self.lambda_action * action_deviation_loss
            + self.lambda_path * path_loss
            + self.lambda_smooth * smoothness_loss
        )
        total_loss = existing_optimization_loss.copy()
        if self.recoverable_deform_enabled:
            total_loss = total_loss + self.lambda_rejoin * rejoin_loss
        per_rejoin_ms = float(rejoin_q_eval_time_ms) / max(1, candidates.shape[0])
        losses: list[dict[str, Any]] = []
        for i in range(candidates.shape[0]):
            item: dict[str, Any] = {
                "safety_loss": float(safety_loss[i]),
                "action_deviation_loss": float(action_deviation_loss[i]),
                "path_loss": float(path_loss[i]),
                "existing_optimization_loss": float(existing_optimization_loss[i]),
                "rejoin_loss": float(rejoin_loss[i]),
                "smoothness_loss": float(smoothness_loss[i]),
                "total_loss": float(total_loss[i]),
                "min_clearance": float(np.min(h_seq[i])),
                "j_best": j_best[i],
                "rejoin_space": rejoin_space,
                "inner_rejoin_metric": self.inner_rejoin_metric,
                "final_rejoin_metric": self.final_rejoin_metric,
                "rejoin_q_eval_time_ms": per_rejoin_ms,
                "ee_nom_cache_time_ms": float(rejoin_context.get("ee_nom_cache_time_ms", 0.0)),
                "ee_final_check_time_ms": 0.0,
                "batched_optimizer": True,
                "jax_batched_optimizer": bool(self._jax_optimizer_ready()),
                "jax_rollout_time_ms": float(rollout_time_ms) / max(1, candidates.shape[0]),
            }
            if rejoin_space == "q_state":
                item["q_rejoin_loss"] = float(rejoin_loss[i])
                item["q_rejoin_dist"] = self._sqrt_loss(float(rejoin_loss[i]))
                item["q_rejoin_index"] = j_best[i]
            losses.append(item)
        return total_loss.astype(np.float32), losses

    def _controlled_anchor(
        self,
        obs: Any,
        chunk: np.ndarray,
        action_idx: np.ndarray,
        state_idx: np.ndarray,
    ) -> np.ndarray:
        """Return no-motion anchors for controlled action dimensions."""
        anchor = np.zeros(len(action_idx), dtype=chunk.dtype)
        q = self.extract_current_q(obs, chunk)
        valid = state_idx < q.shape[0]
        if np.any(valid):
            modes = self._control_mode_ids_for_state_indices(state_idx)
            absolute = valid & (modes == 0)
            if np.any(absolute):
                anchor[absolute] = q[state_idx[absolute]].astype(chunk.dtype, copy=False)
        return anchor


    def _clip_controlled_delta(
        self,
        candidate: np.ndarray,
        nominal: np.ndarray,
        action_idx: np.ndarray,
    ) -> np.ndarray:
        """Clamp controlled deltas around the nominal chunk when configured."""
        if self.max_action_delta is None or len(action_idx) == 0:
            return candidate
        clipped = candidate.copy()
        delta = clipped[:, action_idx] - nominal[:, action_idx]
        delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
        clipped[:, action_idx] = nominal[:, action_idx] + delta
        return clipped

    def _valid_control_indices(self, chunk: np.ndarray) -> np.ndarray:
        """Mask controlled action indices that fit the given chunk width."""
        return self.controlled_action_indices < chunk.shape[1]

    def _controlled_deformation_norm(
        self,
        candidate: np.ndarray,
        nominal: np.ndarray,
    ) -> float:
        """Mean L2 deformation over controlled action dimensions."""
        valid = self._valid_control_indices(nominal)
        if not np.any(valid):
            return 0.0
        action_idx = self.controlled_action_indices[valid]
        delta = candidate[:, action_idx] - nominal[:, action_idx]
        return float(np.mean(np.linalg.norm(delta, axis=1)))

    def _is_better_deformation_candidate(
        self,
        candidate_eval: dict[str, Any],
        candidate_norm: float,
        candidate_safe: bool,
        best_eval: dict[str, Any] | None,
        best_norm: float,
        best_safe: bool,
    ) -> bool:
        """Prefer safe candidates, then progress/clearance, then smaller deformation."""
        if best_eval is None:
            return True
        if candidate_safe != best_safe:
            return candidate_safe
        if candidate_safe:
            candidate_progress = float(
                candidate_eval.get("task_progress_retention", 1.0)
            )
            best_progress = float(best_eval.get("task_progress_retention", 1.0))
            if abs(candidate_progress - best_progress) > 1e-6:
                return candidate_progress > best_progress
            return candidate_norm < best_norm
        clearance_margin = (
            float(candidate_eval["min_clearance"])
            - float(best_eval["min_clearance"])
        )
        if abs(clearance_margin) > 1e-9:
            return clearance_margin > 0.0
        return candidate_norm < best_norm

    def _jax_optimizer_ready(self) -> bool:
        """Return whether JAX acceleration can be used for batched optimizer paths."""
        if not self.jax_batched_optimizer:
            return False
        if not _JAX_AVAILABLE:
            if not self._warned_jax_unavailable:
                logger.warning("JAX batched optimizer requested but JAX is unavailable; using NumPy optimizer path.")
                self._warned_jax_unavailable = True
            return False
        return True

    def _jax_project_candidate_population(
        self,
        nominal: np.ndarray,
        ctrl_samples: np.ndarray,
        action_idx: np.ndarray,
    ) -> np.ndarray | None:
        """Project sampled controlled actions without bouncing through the filter wrapper."""
        if not self._jax_optimizer_ready():
            return None
        max_delta = np.inf if self.max_action_delta is None else float(self.max_action_delta)
        low = -np.inf if self.action_low is None else float(self.action_low)
        high = np.inf if self.action_high is None else float(self.action_high)
        try:
            projected = _jax_project_candidate_population(
                jnp.asarray(nominal, dtype=jnp.float32),
                jnp.asarray(ctrl_samples, dtype=jnp.float32),
                jnp.asarray(action_idx, dtype=jnp.int32),
                np.float32(max_delta),
                np.float32(low),
                np.float32(high),
            )
            return np.asarray(projected, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            if self.jax_batched_optimizer_fallback:
                logger.debug("JAX candidate projection failed; using NumPy optimizer path: %s", exc)
                return None
            raise

    def _make_optimizer_warmup_chunk(self, obs: Any, horizon: int) -> np.ndarray:
        """Build a zero/hold chunk that is safe to use for compilation warmup."""
        horizon = max(1, int(horizon))
        chunk = np.zeros((horizon, self.action_dim), dtype=np.float32)
        try:
            q = self.extract_current_q(obs, chunk)
            valid = (
                (self.controlled_state_indices < q.shape[0])
                & (self.controlled_action_indices < chunk.shape[1])
            )
            if np.any(valid):
                state_idx = self.controlled_state_indices[valid]
                action_idx = self.controlled_action_indices[valid]
                modes = self._control_mode_ids_for_state_indices(state_idx)
                absolute = modes == 0
                if np.any(absolute):
                    chunk[:, action_idx[absolute]] = q[state_idx[absolute]][None, :]
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk optimizer warmup q-seeded chunk failed: %s", exc)
        return chunk

    def _warmup_optimizer_shape(self, obs: Any, horizon: int) -> dict[str, Any]:
        """Compile fixed-shape JAX helper paths for one horizon size."""
        if not self._jax_optimizer_ready():
            return {"compiled": False, "reason": "jax_unavailable"}
        horizon = max(1, int(horizon))
        valid_key = self._valid_control_indices(np.zeros((horizon, self.action_dim), dtype=np.float32))
        mode_key = tuple(
            self._control_mode_ids_for_state_indices(
                self.controlled_state_indices[valid_key]
            ).tolist()
        )
        key = (int(self.opt_population), horizon, int(self.action_dim), mode_key)
        if key in self._optimizer_warmup_cache:
            return {"compiled": False, "reason": "already_warmed", "key": key}

        nominal = self._make_optimizer_warmup_chunk(obs, horizon)
        valid = self._valid_control_indices(nominal)
        if not np.any(valid):
            return {"compiled": False, "reason": "no_control_indices", "key": key}
        action_idx = self.controlled_action_indices[valid]
        ctrl = np.broadcast_to(
            nominal[None, :, action_idx],
            (int(self.opt_population), horizon, len(action_idx)),
        ).copy()
        t0 = time.perf_counter()
        candidates = self._jax_project_candidate_population(nominal, ctrl, action_idx)
        if candidates is None:
            return {"compiled": False, "reason": "projection_fallback", "key": key}
        q_seq_batch = self.rollout_nominal_chunk_batch(obs, candidates)
        safety_eval = self.evaluate_horizon_safety_batch(obs, q_seq_batch)
        _ = self._clearance_sequence_batch_from_eval(
            safety_eval,
            candidates.shape[0],
            candidates.shape[1],
        )
        nominal_q_seq = self.rollout_nominal_chunk(obs, nominal)
        rejoin_context = self._make_rejoin_context(nominal_q_seq, None)
        _ = self._q_rejoin_loss_batch(
            q_seq_batch,
            nominal_q_seq=nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        try:
            self._deform_stage_deformation_cost_batch(obs, candidates, nominal, action_idx)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk warmup deform-stage cost skipped: %s", exc)
        try:
            self._return_deformation_cost_batch(
                obs,
                candidates,
                nominal,
                nominal_q_seq,
                rejoin_context,
                action_idx,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk warmup return cost skipped: %s", exc)
        try:
            self._recover_task_progress_cost_batch(obs, candidates, nominal, action_idx)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk warmup task-progress recover cost skipped: %s", exc)
        try:
            context = RecoveryContext(
                nominal_chunk=nominal.copy(),
                nominal_q_seq=nominal_q_seq.copy(),
                nominal_ee_seq=None,
            )
            self._recovery_terminal_rejoin_info(
                obs,
                nominal,
                context,
                rejoin_context,
                default_target_index=min(horizon - 1, self.min_rejoin_offset),
            )
            if hasattr(self, "evaluate_recovery_path_safety"):
                self.evaluate_recovery_path_safety(obs, nominal, candidate_name="warmup")
            self.evaluate_candidate_acceptance(obs, nominal, candidate_type="recover")
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk warmup recovery checks skipped: %s", exc)
        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        self._optimizer_warmup_cache.add(key)
        return {"compiled": True, "key": key, "time_ms": float(elapsed_ms)}

    def _warmup_optimizer_live_path(
        self,
        obs: Any,
        nominal_chunk: Any | None = None,
    ) -> dict[str, Any]:
        """Exercise the explicit-return optimizer while restoring mutable state."""
        if not (self.recoverable_deform_enabled and self.explicit_return):
            return {"compiled": False, "reason": "explicit_return_disabled"}
        t0 = time.perf_counter()
        rng_state = None
        try:
            rng_state = self._rng.bit_generator.state
        except Exception:  # noqa: BLE001
            rng_state = None
        try:
            self.reset()
            if nominal_chunk is None:
                nominal = self._make_optimizer_warmup_chunk(obs, self.horizon)
            else:
                nominal, _ = self._as_chunk(nominal_chunk)
                nominal = np.asarray(nominal, dtype=np.float32).copy()
            nominal_q_seq = self.rollout_nominal_chunk(obs, nominal)
            safety_info = self.evaluate_horizon_safety(obs, nominal_q_seq)
            _, info = self.deform_chunk_optimized(
                nominal,
                obs=obs,
                nominal_q_seq=nominal_q_seq,
                safety_info=safety_info,
            )
            return {
                "compiled": True,
                "path": "explicit_return_optimizer",
                "time_ms": float(1000.0 * (time.perf_counter() - t0)),
                "optimized_accepted": self._coerce_bool(info.get("optimized_accepted", False)),
                "deform_cem_iterations_run": info.get("deform_cem_iterations_run"),
                "return_cem_iterations_run": info.get("return_cem_iterations_run"),
                "cem_early_stopped": info.get("cem_early_stopped"),
                "min_clearance": info.get("min_clearance"),
                "recover_min_clearance": info.get("recover_min_clearance"),
                "rejection_cause": info.get("rejection_cause"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("SafeChunk live-path optimizer warmup failed: %s", exc)
            return {
                "compiled": False,
                "path": "explicit_return_optimizer",
                "reason": str(exc),
                "time_ms": float(1000.0 * (time.perf_counter() - t0)),
            }
        finally:
            self.reset()
            if rng_state is not None:
                try:
                    self._rng.bit_generator.state = rng_state
                except Exception:  # noqa: BLE001
                    pass

    def warmup_optimizer(
        self,
        obs: Any,
        nominal_chunk: Any | None = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Warm JAX/CEM helper paths for configured horizons and optional live path."""
        if self._optimizer_warmup_done and nominal_chunk is None and not force:
            return dict(self._optimizer_warmup_info)
        t0 = time.perf_counter()
        horizons: list[int] = [self.horizon]
        if self.recoverable_deform_enabled and self.explicit_return:
            horizons.extend([self.deform_horizon, self.return_horizon])
        seen: list[int] = []
        results: list[dict[str, Any]] = []
        for horizon in horizons:
            horizon = int(horizon)
            if horizon in seen:
                continue
            seen.append(horizon)
            try:
                results.append(self._warmup_optimizer_shape(obs, horizon))
            except Exception as exc:  # noqa: BLE001
                logger.debug("SafeChunk optimizer warmup failed for horizon %s: %s", horizon, exc)
                results.append({"compiled": False, "horizon": horizon, "reason": str(exc)})
        live_path_result = self._warmup_optimizer_live_path(obs, nominal_chunk=nominal_chunk)
        info = {
            "optimizer_warmup_enabled": True,
            "optimizer_warmup_done": True,
            "optimizer_warmup_time_ms": float(1000.0 * (time.perf_counter() - t0)),
            "optimizer_warmup_results": results,
            "optimizer_warmup_live_path_result": live_path_result,
        }
        if nominal_chunk is None:
            self._optimizer_warmup_done = True
            self._optimizer_warmup_info = dict(info)
        return info


    def rollout_nominal_chunk_batch(self, obs: Any, action_chunks: Any) -> np.ndarray:
        """Roll out a batch of action chunks with JAX or NumPy fallback."""
        chunks = np.asarray(action_chunks, dtype=np.float32)
        if chunks.ndim == 2:
            chunks = chunks[None, :, :]
        if chunks.ndim != 3:
            raise ValueError(
                "Expected action_chunks with shape (B, H, A), "
                f"got {chunks.shape}"
            )
        jax_q_seq = self._jax_rollout_nominal_chunk_batch(obs, chunks)
        if jax_q_seq is not None:
            return jax_q_seq

        batch, horizon = chunks.shape[:2]
        q0 = self.extract_current_q(obs, chunks[0] if batch else None)
        q = np.broadcast_to(q0[None, :], (batch, q0.shape[0])).copy()
        q_seq = np.zeros((batch, horizon, q0.shape[0]), dtype=np.float32)

        valid = (
            (self.controlled_state_indices < q0.shape[0])
            & (self.controlled_action_indices < chunks.shape[2])
        )
        state_idx = self.controlled_state_indices[valid]
        action_idx = self.controlled_action_indices[valid]

        for k in range(horizon):
            q_next = self._apply_controlled_action_step_batch(
                q,
                chunks[:, k, :],
                state_idx,
                action_idx,
            )
            q_seq[:, k, :] = q_next
            q = q_next

        return q_seq

    def _optimized_seed_chunks(
        self,
        obs: Any,
        chunk: np.ndarray,
        safety_info: dict[str, Any] | None,
    ) -> list[np.ndarray]:
        """Seed the optimizer with nominal plus unique fixed-scale candidates."""
        seeds: list[np.ndarray] = [chunk.copy()]
        for _, candidate in self._make_chunk_deformation_candidates(obs, chunk, safety_info or {}):
            if not any(np.allclose(candidate, seen) for seen in seeds):
                seeds.append(candidate.copy())
        return seeds

    def nominal_path_deviation_loss_batch(
        self,
        q_seq_batch: np.ndarray,
        nominal_q_seq: Any,
    ) -> np.ndarray:
        """Vectorized mean squared deviation from the nominal state rollout."""
        if nominal_q_seq is None:
            return np.zeros(q_seq_batch.shape[0], dtype=np.float32)
        q_seq_batch = np.asarray(q_seq_batch, dtype=np.float32)
        nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        horizon = min(q_seq_batch.shape[1], nominal_q_seq.shape[0])
        if horizon == 0:
            return np.zeros(q_seq_batch.shape[0], dtype=np.float32)
        q_dim = min(q_seq_batch.shape[2], nominal_q_seq.shape[1])
        delta = q_seq_batch[:, :horizon, :q_dim] - nominal_q_seq[None, :horizon, :q_dim]
        return np.square(delta).mean(axis=(1, 2)).astype(np.float32)

    def _ee_rejoin_loss(
        self,
        q_seq: np.ndarray,
        nominal_q_seq: Any | None = None,
        nominal_ee_seq: Any | None = None,
        rejoin_context: dict[str, Any] | None = None,
    ) -> tuple[float, int | None, float, bool]:
        """Evaluate terminal EE-pose rejoin against future nominal poses."""
        t0 = time.perf_counter()
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)
        ee_future = rejoin_context.get("ee_nom_future")
        if ee_future is None and nominal_ee_seq is not None:
            ee_seq = np.asarray(nominal_ee_seq, dtype=np.float32)
            if ee_seq.shape[0] > self.min_rejoin_offset:
                ee_future = ee_seq.reshape(ee_seq.shape[0], -1)[self.min_rejoin_offset :]
        if ee_future is None and nominal_q_seq is not None:
            ee_seq = self._ee_pose_sequence(nominal_q_seq)
            if ee_seq is not None and ee_seq.shape[0] > self.min_rejoin_offset:
                ee_future = ee_seq.reshape(ee_seq.shape[0], -1)[self.min_rejoin_offset :]
        opt_ee_seq = self._ee_pose_sequence(q_seq)
        if opt_ee_seq is None or ee_future is None:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0, False
        loss, j_best = self._nearest_future_loss(
            opt_ee_seq[-1],
            ee_future,
            start_index=self.min_rejoin_offset,
        )
        return loss, j_best, (time.perf_counter() - t0) * 1000.0, True

    def _final_rejoin_check(
        self,
        q_seq: np.ndarray,
        nominal_q_seq: Any | None = None,
        nominal_ee_seq: Any | None = None,
        rejoin_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Combine q, qd, and optional EE checks into final recoverability info."""
        if not self.recoverable_deform_enabled:
            return {
                "is_recoverable": None,
                "q_rejoin_loss": 0.0,
                "q_rejoin_dist": 0.0,
                "q_rejoin_index": None,
                "qd_rejoin_loss": 0.0,
                "qd_rejoin_dist": 0.0,
                "qd_rejoin_index": None,
                "qd_rejoin_threshold": float(self.qd_rejoin_threshold),
                "ee_rejoin_loss": 0.0,
                "ee_rejoin_dist": 0.0,
                "ee_rejoin_index": None,
                "ee_final_check_available": None,
                "rejoin_q_eval_time_ms": 0.0,
                "rejoin_qd_eval_time_ms": 0.0,
                "ee_final_check_time_ms": 0.0,
            }
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)

        q_loss, q_j_best, q_time_ms = self._q_rejoin_loss(
            q_seq,
            nominal_q_seq=nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        q_dist = self._sqrt_loss(q_loss)
        q_recoverable = bool(
            q_j_best is not None and q_dist < self.q_rejoin_threshold
        )
        qd_loss, qd_j_best, qd_time_ms = self._qd_rejoin_loss(
            q_seq,
            nominal_q_seq=nominal_q_seq,
            target_index=q_j_best,
            rejoin_context=rejoin_context,
        )
        qd_dist = self._sqrt_loss(qd_loss)
        qd_recoverable, qd_acceptance = self._qd_rejoin_acceptance(
            qd_j_best,
            qd_dist,
        )

        ee_loss = 0.0
        ee_dist = 0.0
        ee_j_best = None
        ee_time_ms = 0.0
        ee_available = None
        ee_recoverable = True
        if self.use_ee_final_check and self.final_rejoin_metric == "ee_pose":
            t0 = time.perf_counter()
            ee_future = rejoin_context.get("ee_nom_future")
            if ee_future is None and nominal_ee_seq is not None:
                ee_seq = np.asarray(nominal_ee_seq, dtype=np.float32)
                if ee_seq.shape[0] > self.min_rejoin_offset:
                    ee_future = ee_seq.reshape(ee_seq.shape[0], -1)[
                        self.min_rejoin_offset :
                    ]
            if ee_future is None and nominal_q_seq is not None:
                ee_seq = self._ee_pose_sequence(nominal_q_seq)
                if ee_seq is not None and ee_seq.shape[0] > self.min_rejoin_offset:
                    ee_future = ee_seq.reshape(ee_seq.shape[0], -1)[
                        self.min_rejoin_offset :
                    ]
            ee_opt_end = self._ee_pose(q_seq[-1]) if q_seq.shape[0] else None
            if ee_opt_end is None or ee_future is None:
                ee_available = False
                ee_loss = float("inf")
                ee_dist = float("inf")
                ee_recoverable = False
            else:
                ee_available = True
                ee_loss, ee_j_best = self._nearest_future_loss(
                    ee_opt_end,
                    ee_future,
                    start_index=self.min_rejoin_offset,
                )
                ee_dist = self._sqrt_loss(ee_loss)
                ee_recoverable = bool(ee_dist < self.ee_rejoin_threshold)
            ee_time_ms = (time.perf_counter() - t0) * 1000.0
        elif self.final_rejoin_metric == "q_state":
            ee_available = None
        elif self.final_rejoin_metric == "none":
            ee_available = None

        return {
            "is_recoverable": bool(q_recoverable and qd_recoverable and ee_recoverable),
            "q_rejoin_loss": float(q_loss),
            "q_rejoin_dist": float(q_dist),
            "q_rejoin_index": None if q_j_best is None else int(q_j_best),
            "q_rejoin_threshold": float(self.q_rejoin_threshold),
            "qd_rejoin_loss": float(qd_loss),
            "qd_rejoin_dist": float(qd_dist),
            "qd_rejoin_index": None if qd_j_best is None else int(qd_j_best),
            "qd_rejoin_threshold": float(self.qd_rejoin_threshold),
            **qd_acceptance,
            "ee_rejoin_loss": float(ee_loss),
            "ee_rejoin_dist": float(ee_dist),
            "ee_rejoin_index": None if ee_j_best is None else int(ee_j_best),
            "ee_rejoin_threshold": float(self.ee_rejoin_threshold),
            "ee_final_check_available": ee_available,
            "rejoin_q_eval_time_ms": float(q_time_ms),
            "rejoin_qd_eval_time_ms": float(qd_time_ms),
            "ee_final_check_time_ms": float(ee_time_ms),
        }

    def _ee_pose(self, q: Any) -> np.ndarray | None:
        """Return a flattened end-effector pose for one state when supported."""
        op = self._get_oscbf_operator()
        for method_name in ("ee_pose", "compute_ee_pose"):
            method = getattr(op, method_name, None)
            if method is None:
                continue
            try:
                ee_pose = np.asarray(method(q), dtype=np.float32).reshape(-1)
            except Exception as exc:  # pragma: no cover - optional integration path
                logger.debug("EE rejoin pose via %s failed: %s", method_name, exc)
                continue
            return ee_pose
        ee_seq = self._ee_pose_sequence(np.asarray(q, dtype=np.float32).reshape(1, -1))
        if ee_seq is None or ee_seq.shape[0] == 0:
            return None
        return ee_seq[-1]

    def _ee_pose_sequence(self, q_seq: Any) -> np.ndarray | None:
        """Return flattened end-effector poses for a state sequence when supported."""
        op = self._get_oscbf_operator()
        for method_name in ("ee_pose_sequence", "compute_ee_pose_sequence"):
            method = getattr(op, method_name, None)
            if method is None:
                continue
            try:
                ee_seq = np.asarray(method(q_seq), dtype=np.float32)
            except Exception as exc:  # pragma: no cover - optional integration path
                logger.debug("EE rejoin rollout via %s failed: %s", method_name, exc)
                continue
            if ee_seq.shape[0] == np.asarray(q_seq).shape[0]:
                return ee_seq.reshape(ee_seq.shape[0], -1)
        return None

    def _project_optimized_chunk(
        self,
        candidate: Any,
        nominal: np.ndarray,
        action_idx: np.ndarray,
    ) -> np.ndarray:
        """Restore passthrough actions and enforce configured action bounds."""
        projected = np.asarray(candidate, dtype=nominal.dtype).copy()
        passthrough_idx = [i for i in range(nominal.shape[1]) if i not in set(action_idx.tolist())]
        projected[:, passthrough_idx] = nominal[:, passthrough_idx]
        projected = self._clip_controlled_delta(projected, nominal, action_idx)
        if self.action_low is not None or self.action_high is not None:
            low = -np.inf if self.action_low is None else float(self.action_low)
            high = np.inf if self.action_high is None else float(self.action_high)
            projected[:, action_idx] = np.clip(projected[:, action_idx], low, high)
        projected[:, passthrough_idx] = nominal[:, passthrough_idx]
        return projected

    def _optimized_final_info(
        self,
        obs: Any,
        chunk: np.ndarray,
        nominal: np.ndarray,
        nominal_q_seq: Any,
        nominal_ee_seq: Any,
        human_state: Any,
        j_best: int | None,
        rejoin_loss: float,
        losses: dict[str, Any],
        rejoin_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Package optimizer rollout, safety, and recoverability diagnostics."""
        del human_state
        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        min_clearance = float(np.min(h_seq))
        is_safe = bool(min_clearance >= self.min_clearance)
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq, nominal_ee_seq)
        if not losses:
            _, losses = self._optimized_deformation_cost(
                obs,
                chunk,
                nominal,
                nominal_q_seq,
                nominal_ee_seq,
                None,
                rejoin_context=rejoin_context,
            )
            j_best = losses.get("j_best")
            rejoin_loss = losses.get("rejoin_loss", rejoin_loss)

        final_rejoin = self._final_rejoin_check(
            q_seq,
            nominal_q_seq=nominal_q_seq,
            nominal_ee_seq=nominal_ee_seq,
            rejoin_context=rejoin_context,
        )
        if self.recoverable_deform_enabled:
            is_recoverable = bool(final_rejoin.get("is_recoverable", False))
            rejoin_index = final_rejoin.get("q_rejoin_index")
            if rejoin_index is None:
                rejoin_index = final_rejoin.get("ee_rejoin_index")
            rejoin_cost = float(final_rejoin.get("q_rejoin_loss", float("inf")))
        else:
            is_recoverable = None
            rejoin_index = None if j_best is None else int(j_best)
            rejoin_cost = float(rejoin_loss)
        best_min_clearance = float(losses.get("min_clearance", min_clearance))
        required_min_clearance = float(self.min_clearance)
        clearance_gap = float(required_min_clearance - best_min_clearance)
        safety_rejected = not is_safe
        recovery_rejected = bool(
            self.recoverable_deform_enabled and not bool(is_recoverable)
        )
        rejection_cause = self._optimized_reject_reason_from_flags(
            safety_rejected,
            recovery_rejected,
        )

        info: dict[str, Any] = dict(safety_eval)
        info.update(
            {
                "deform_mode": "optimized",
                "deformation_source": "optimized_deform",
                "deform_safe": is_safe,
                "deform_min_clearance": min_clearance,
                "deformation_norm": self._controlled_deformation_norm(chunk, nominal),
                "recoverable_deform_enabled": self.recoverable_deform_enabled,
                "brake_if_unrecoverable": self.brake_if_unrecoverable,
                "inner_rejoin_metric": self.inner_rejoin_metric,
                "final_rejoin_metric": self.final_rejoin_metric,
                "cache_nominal_ee": self.cache_nominal_ee,
                "ee_rejoin_in_inner_loop": self.ee_rejoin_in_inner_loop,
                "debug_safety_feasibility": self.debug_safety_feasibility,
                "is_safe": is_safe,
                "is_recoverable": is_recoverable,
                "safety_rejected": safety_rejected,
                "recovery_rejected": recovery_rejected,
                "rejection_cause": rejection_cause,
                "best_min_clearance": best_min_clearance,
                "required_min_clearance": required_min_clearance,
                "clearance_gap": clearance_gap,
                "rejoin_cost": rejoin_cost,
                "rejoin_loss": float(losses.get("rejoin_loss", rejoin_loss)),
                "q_rejoin_loss": float(final_rejoin.get("q_rejoin_loss", 0.0)),
                "q_rejoin_dist": float(final_rejoin.get("q_rejoin_dist", 0.0)),
                "q_rejoin_threshold": float(final_rejoin.get(
                    "q_rejoin_threshold", self.q_rejoin_threshold
                )),
                "q_rejoin_index": final_rejoin.get("q_rejoin_index"),
                "qd_rejoin_loss": float(final_rejoin.get("qd_rejoin_loss", 0.0)),
                "qd_rejoin_dist": float(final_rejoin.get("qd_rejoin_dist", 0.0)),
                "qd_rejoin_threshold": float(final_rejoin.get(
                    "qd_rejoin_threshold", self.qd_rejoin_threshold
                )),
                "qd_rejoin_index": final_rejoin.get("qd_rejoin_index"),
                "ee_rejoin_loss": float(final_rejoin.get("ee_rejoin_loss", 0.0)),
                "ee_rejoin_dist": float(final_rejoin.get("ee_rejoin_dist", 0.0)),
                "ee_rejoin_threshold": float(final_rejoin.get(
                    "ee_rejoin_threshold", self.ee_rejoin_threshold
                )),
                "ee_rejoin_index": final_rejoin.get("ee_rejoin_index"),
                "ee_final_check_available": final_rejoin.get(
                    "ee_final_check_available"
                ),
                "rejoin_index": None if rejoin_index is None else int(rejoin_index),
                "j_best": None if rejoin_index is None else int(rejoin_index),
                "optimizer_j_best": None if j_best is None else int(j_best),
                "rejoin_space": losses.get("rejoin_space"),
                "safety_loss": float(losses.get("safety_loss", 0.0)),
                "action_deviation_loss": float(losses.get("action_deviation_loss", 0.0)),
                "path_loss": float(losses.get("path_loss", 0.0)),
                "existing_optimization_loss": float(
                    losses.get("existing_optimization_loss", 0.0)
                ),
                "smoothness_loss": float(losses.get("smoothness_loss", 0.0)),
                "total_loss": float(losses.get("total_loss", 0.0)),
                "rejoin_q_eval_time_ms": float(final_rejoin.get(
                    "rejoin_q_eval_time_ms",
                    losses.get("rejoin_q_eval_time_ms", 0.0),
                )),
                "rejoin_qd_eval_time_ms": float(final_rejoin.get(
                    "rejoin_qd_eval_time_ms",
                    losses.get("rejoin_qd_eval_time_ms", 0.0),
                )),
                "ee_nom_cache_time_ms": float(
                    rejoin_context.get(
                        "ee_nom_cache_time_ms",
                        losses.get("ee_nom_cache_time_ms", 0.0),
                    )
                ),
                "ee_final_check_time_ms": float(final_rejoin.get(
                    "ee_final_check_time_ms", 0.0
                )),
                "jax_batched_optimizer": bool(losses.get(
                    "jax_batched_optimizer", self._jax_optimizer_ready()
                )),
                "jax_rollout_time_ms": float(losses.get(
                    "jax_rollout_time_ms", 0.0
                )),
                "fallback_used": False,
                "recovery_mode": (
                    "optimized_recoverable_deform"
                    if self.recoverable_deform_enabled
                    else None
                ),
                "min_clearance": min_clearance,
            }
        )
        log_fn = logger.info if self.debug_safety_feasibility else logger.debug
        log_fn(
            "optimized SafeChunk-Deform final: mode=%s best_min_clearance=%.4f "
            "required_min_clearance=%.4f clearance_gap=%.4f safety_loss=%.6f "
            "existing_loss=%.6f rejoin_loss=%.6f q_dist=%.6f ee_dist=%.6f "
            "rejoin_index=%s safe=%s recoverable=%s rejection_cause=%s",
            info.get("recovery_mode") or info.get("deform_mode"),
            info["best_min_clearance"],
            info["required_min_clearance"],
            info["clearance_gap"],
            info["safety_loss"],
            info["existing_optimization_loss"],
            info["rejoin_loss"],
            info["q_rejoin_dist"],
            info["ee_rejoin_dist"],
            info["rejoin_index"],
            info["is_safe"],
            info["is_recoverable"],
            info["rejection_cause"],
        )
        return info

    def _optimized_failure_info(self, error: str) -> dict[str, Any]:
        """Create a standard diagnostics payload for optimizer exceptions."""
        return {
            "deform_mode": "optimized",
            "deformation_source": "optimized_deform",
            "deform_safe": False,
            "is_safe": False,
            "recoverable_deform_enabled": self.recoverable_deform_enabled,
            "brake_if_unrecoverable": self.brake_if_unrecoverable,
            "is_recoverable": False if self.recoverable_deform_enabled else None,
            "inner_rejoin_metric": self.inner_rejoin_metric,
            "final_rejoin_metric": self.final_rejoin_metric,
            "cache_nominal_ee": self.cache_nominal_ee,
            "ee_rejoin_in_inner_loop": self.ee_rejoin_in_inner_loop,
            "debug_safety_feasibility": self.debug_safety_feasibility,
            "safety_rejected": True,
            "recovery_rejected": bool(self.recoverable_deform_enabled),
            "rejection_cause": (
                "unsafe_and_unrecoverable"
                if self.recoverable_deform_enabled
                else "unsafe"
            ),
            "best_min_clearance": float("-inf"),
            "required_min_clearance": float(self.min_clearance),
            "clearance_gap": float("inf"),
            "rejoin_index": None,
            "j_best": None,
            "rejoin_cost": float("inf"),
            "rejoin_loss": float("inf"),
            "q_rejoin_loss": float("inf"),
            "q_rejoin_dist": float("inf"),
            "q_rejoin_threshold": self.q_rejoin_threshold,
            "q_rejoin_index": None,
            "qd_rejoin_loss": float("inf"),
            "qd_rejoin_dist": float("inf"),
            "qd_rejoin_threshold": self.qd_rejoin_threshold,
            "qd_rejoin_index": None,
            "ee_rejoin_loss": float("inf"),
            "ee_rejoin_dist": float("inf"),
            "ee_rejoin_threshold": self.ee_rejoin_threshold,
            "ee_rejoin_index": None,
            "ee_final_check_available": False,
            "rejoin_q_eval_time_ms": 0.0,
            "rejoin_qd_eval_time_ms": 0.0,
            "ee_nom_cache_time_ms": 0.0,
            "ee_final_check_time_ms": 0.0,
            "safety_loss": float("inf"),
            "action_deviation_loss": 0.0,
            "path_loss": 0.0,
            "existing_optimization_loss": float("inf"),
            "smoothness_loss": 0.0,
            "total_loss": float("inf"),
            "fallback_used": True,
            "deform_min_clearance": float("-inf"),
            "min_clearance": float("-inf"),
            "optimized_error": error,
        }

    def _optimized_reject_reason(self, info: dict[str, Any]) -> str:
        """Derive a compact rejection reason from optimizer diagnostics."""
        safety_rejected = not bool(info.get("deform_safe", False))
        recovery_rejected = bool(
            self.recoverable_deform_enabled and not info.get("is_recoverable", False)
        )
        return self._optimized_reject_reason_from_flags(
            safety_rejected,
            recovery_rejected,
        ) or "rejected"

    def _optimized_reject_reason_from_flags(
        self,
        safety_rejected: bool,
        recovery_rejected: bool,
    ) -> str | None:
        """Map safety/recovery rejection booleans to an info-string reason."""
        if safety_rejected and recovery_rejected:
            return "unsafe_and_unrecoverable"
        if safety_rejected:
            return "unsafe"
        if recovery_rejected:
            return "unrecoverable"
        return None

    def _prefixed_optimized_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Copy optimizer diagnostics under both raw and optimized-prefixed keys."""
        keys = (
            "safety_loss",
            "action_deviation_loss",
            "path_loss",
            "existing_optimization_loss",
            "rejoin_loss",
            "smoothness_loss",
            "total_loss",
            "min_clearance",
            "is_safe",
            "is_recoverable",
            "j_best",
            "rejoin_index",
            "rejoin_cost",
            "rejoin_space",
            "explicit_return",
            "recovery_phase",
            "cached_motion_active",
            "recovery_context_active",
            "start_chunk_index",
            "trigger_step",
            "target_rejoin_index",
            "deform_min_clearance_stage",
            "deform_stage_accepted",
            "return_min_clearance",
            "return_rejoin_loss",
            "return_target_index",
            "return_accepted",
            "resumed_from_cached_index",
            "return_retries",
            "max_return_retries",
            "inner_rejoin_metric",
            "final_rejoin_metric",
            "debug_safety_feasibility",
            "safety_rejected",
            "recovery_rejected",
            "rejection_cause",
            "best_min_clearance",
            "required_min_clearance",
            "clearance_gap",
            "q_rejoin_loss",
            "q_rejoin_dist",
            "q_rejoin_threshold",
            "q_rejoin_index",
            "ee_rejoin_loss",
            "ee_rejoin_dist",
            "ee_rejoin_threshold",
            "ee_rejoin_index",
            "ee_final_check_available",
            "rejoin_q_eval_time_ms",
            "ee_nom_cache_time_ms",
            "ee_final_check_time_ms",
            "recoverable_deform_enabled",
            "fallback_used",
            "recovery_mode",
            "deform_min_clearance",
        )
        prefixed: dict[str, Any] = {f"optimized_{key}": info.get(key) for key in keys if key in info}
        for key in keys:
            if key in info:
                prefixed.setdefault(key, info[key])
        return prefixed

    def _controlled_progress_retention(
        self,
        candidate: np.ndarray,
        nominal: np.ndarray,
        obs: Any,
    ) -> float:
        """Estimate how much controlled motion remains relative to nominal."""
        valid = self._valid_control_indices(nominal)
        if not np.any(valid):
            return 1.0
        action_idx = self.controlled_action_indices[valid]
        state_idx = self.controlled_state_indices[valid]
        anchor = self._controlled_anchor(obs, nominal, action_idx, state_idx)
        nominal_delta = nominal[:, action_idx] - anchor[None, :]
        candidate_delta = candidate[:, action_idx] - anchor[None, :]
        denom = float(np.mean(np.linalg.norm(nominal_delta, axis=1)))
        if denom <= 1e-9:
            return 1.0
        numer = float(np.mean(np.linalg.norm(candidate_delta, axis=1)))
        return float(np.clip(numer / denom, 0.0, 2.0))

    def _call_oscbf_chunk_method(
        self,
        method: Callable[..., Any],
        obs: Any,
        chunk: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        """Call OSCBF chunk filters across supported legacy argument names."""
        attempts = (
            lambda: method(action_chunk=chunk, obs=obs, **kwargs),
            lambda: method(action_chunk=chunk, observations=obs, **kwargs),
            lambda: method(obs=obs, action_chunk=chunk, **kwargs),
            lambda: method(observations=obs, action_chunk=chunk, **kwargs),
            lambda: method(chunk, obs, **kwargs),
            lambda: method(chunk, **kwargs),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Chunk OSCBF method could not be called")

    def Perform(
        self,
        obs: Any,
        chunk: np.ndarray,
        q_seq: np.ndarray,
        safety_info: dict[str, Any],
        braked_chunk: np.ndarray,
        info: dict[str, Any],
        original_shape: tuple[int, ...],
        deform_trigger_reason: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute one deformation/recovery attempt and update parent bookkeeping."""
        parent = self.parent

        # Cooldown and attempt caps are enforced before launching expensive recovery.
        recovery_optimizer_skip_reason: str | None = None
        recovery_attempt_reset_info: dict[str, Any] = {}
        if (
            self.recoverable_deform_enabled
            and self.explicit_return
            and self.safechunk_recover_enabled
        ):
            recovery_owner = getattr(self, "recovery", None)
            maybe_reset = getattr(
                recovery_owner,
                "_maybe_reset_recovery_attempts_after_brake_timeout",
                None,
            )
            if callable(maybe_reset):
                recovery_attempt_reset_info = maybe_reset()
                if recovery_attempt_reset_info.get(
                    "recovery_attempt_reset_after_brake_timeout", False
                ):
                    parent.recovery_optimizer_cooldown_remaining = 0
                    parent.recovery_attempts_in_unsafe_streak = 0
                    parent.recovery_failure_streak = 0
            if parent.recovery_optimizer_cooldown_remaining > 0:
                recovery_optimizer_skip_reason = "cooldown"
                parent.recovery_optimizer_cooldown_remaining = max(
                    0,
                    int(parent.recovery_optimizer_cooldown_remaining) - 1,
                )
            elif (
                self.recover_max_attempts_per_unsafe_streak > 0
                and parent.recovery_attempts_in_unsafe_streak
                >= self.recover_max_attempts_per_unsafe_streak
            ):
                recovery_optimizer_skip_reason = "attempt_cap"

        if recovery_optimizer_skip_reason is not None:
            parent.recovery_optimization_skipped_count += 1
            parent.brake_streak += 1
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deform_mode": "recovery_optimization_skipped",
                    "deformation_source": "horizon_brake",
                    "deformation_deferred": True,
                    "fallback_reason": f"recovery_optimizer_{recovery_optimizer_skip_reason}",
                    "fallback_used": True,
                    "recovery_optimization_skipped": True,
                    "recovery_optimization_skip_reason": recovery_optimizer_skip_reason,
                }
            )
            info.update(self._temporary_streak_info(trigger_reason=deform_trigger_reason))
            if recovery_attempt_reset_info:
                info.update(recovery_attempt_reset_info)
            return self._hold_return_or_emergency_deform(
                obs,
                chunk,
                braked_chunk,
                info,
                original_shape,
                **kwargs,
            )

        if (
            self.recoverable_deform_enabled
            and self.explicit_return
            and self.safechunk_recover_enabled
        ):
            parent.recovery_attempts_in_unsafe_streak += 1

        safe_chunk, deform_info = self.deform_chunk(
            obs,
            chunk,
            safety_info=safety_info,
            braked_chunk=braked_chunk,
            nominal_q_seq=q_seq,
            **kwargs,
        )
        info.update(deform_info)
        if recovery_attempt_reset_info:
            info.update(recovery_attempt_reset_info)
        if self._coerce_bool(info.get("optimized_accepted", False)):
            parent.recovery_failure_streak = 0
            parent.recovery_optimizer_cooldown_remaining = 0
            parent.recovery_attempts_in_unsafe_streak = 0
        elif info.get("optimized_accepted") is not None or info.get("fallback_used") is not None:
            parent.recovery_failure_streak += 1
            parent.recovery_failure_streak_max = max(
                parent.recovery_failure_streak_max,
                parent.recovery_failure_streak,
            )
            if (
                self.recoverable_deform_enabled
                and self.explicit_return
                and self.safechunk_recover_enabled
                and self.recover_retry_cooldown_steps > 0
            ):
                parent.recovery_optimizer_cooldown_remaining = max(
                    int(parent.recovery_optimizer_cooldown_remaining),
                    int(self.recover_retry_cooldown_steps),
                )
        info.update(self._temporary_streak_info(trigger_reason=deform_trigger_reason))
        if (
            self._coerce_bool(info.get("optimized_accepted", False))
            and self.explicit_return
            and self.commit_accepted_chunks
        ):
            # Persisted recovery chunks must pass an extra commit-time safety check.
            committed, commit_reject_info = self.recovery._commit_explicit_recovery_chunk(
                obs,
                safe_chunk,
                info,
                **kwargs,
            )
            if not committed:
                info.update(commit_reject_info)
                info.update(
                    {
                        "safety_mode": "horizon_brake",
                        "mode": "horizon_brake",
                        "deform_mode": "committed_recovery_commit_rejected",
                        "deformation_source": "horizon_brake",
                        "optimized_accepted": False,
                        "optimized_fallback": "brake",
                        "optimized_reject_reason": "committed_rejected_missing_planned_q",
                        "fallback_reason": "committed_rejected_missing_planned_q",
                        "fallback_used": True,
                    }
                )
                return self._hold_return_or_emergency_deform(
                    obs,
                    chunk,
                    braked_chunk,
                    info,
                    original_shape,
                    **kwargs,
                )

            committed_result = self.recovery._serve_committed_chunk(obs, chunk, original_shape, **kwargs)
            pending_committed_replan_info = self.recovery._pop_pending_committed_replan_info()
            if pending_committed_replan_info:
                info.update(pending_committed_replan_info)
            if committed_result is not None:
                committed_chunk, committed_info = committed_result
                for key in (
                    "optimized_accepted",
                    "deform_stage_accepted",
                    "recover_accepted",
                    "recover_target_index",
                    "resumed_from_recover_index",
                    "deform_stage_accepted",
                    "return_accepted",
                    "return_target_index",
                    "resumed_from_cached_index",
                    "is_safe",
                    "is_recoverable",
                    "rejoin_index",
                    "q_rejoin_dist",
                    "recover_rejoin_loss",
                    "recover_projection_on_nominal",
                    "recover_cosine_to_nominal",
                    "recover_direction_cosine",
                    "recover_direction_cosine_threshold",
                    "recover_direction_loss",
                    "recover_direction_ok",
                    "recover_direction_alignment_available",
                    "recover_direction_alignment_weight",
                    "recover_ordered_path_available",
                    "recover_ordered_target_index",
                    "recover_ordered_horizon",
                    "recover_ordered_pose_loss",
                    "recover_ordered_delta_loss",
                    "recover_ordered_loss",
                    "recover_ordered_pose_weight",
                    "recover_ordered_delta_weight",
                    "recover_ordered_pose_threshold",
                    "recover_ordered_delta_threshold",
                    "recover_ordered_ok",
                    "nominal_delta_norm",
                    "path_delta_norm",
                    "nominal_rejoin_score",
                    "nominal_rejoin_available",
                    "nominal_rejoin_suppressed_reason",
                    "nominal_rejoin_clearance",
                    "nominal_rejoin_safe_prefix_len",
                    "deform_min_clearance_stage",
                    "recover_min_clearance",
                    "return_rejoin_loss",
                    "deform_min_clearance_stage",
                    "return_min_clearance",
                    "deformation_norm",
                    "deform_min_clearance",
                    "best_min_clearance",
                    "required_min_clearance",
                    "clearance_gap",
                    "explicit_recovery",
                    "deform_chunk_length",
                    "recover_chunk_length",
                    "explicit_return",
                    "deform_chunk_length",
                    "return_chunk_length",
                    "committed_chunk_total_length",
                    "optimizer_method",
                    "deform_optimizer_method",
                    "return_optimizer_method",
                    "gradient_iterations_run",
                    "gradient_max_iters",
                    "gradient_samples",
                    "gradient_eps",
                    "gradient_early_stopped",
                    "gradient_candidate_early_stopped",
                    "gradient_batched_line_search",
                    "gradient_line_search_batch_evaluations",
                    "gradient_line_search_batch_size",
                    "gradient_jax_scan_used",
                    "gradient_jax_scan_used_count",
                    "optimizer_evaluations",
                    "deform_optimizer_time_ms",
                    "return_optimizer_time_ms",
                    "explicit_optimizer_time_ms",
                    "committed_suffix_optimizer_time_ms",
                    "committed_plan_rollout_time_ms",
                    "committed_plan_safety_time_ms",
                    "committed_plan_diagnostics_time_ms",
                    "fixed_shape_jax_cost",
                    "fixed_shape_jax_safety",
                    "fixed_shape_safety_method",
                    "fixed_shape_cost_batch_size",
                    "fixed_shape_cost_original_batch_size",
                    "fixed_shape_cost_padding",
                    "fixed_shape_rollout_time_ms",
                    "fixed_shape_safety_time_ms",
                    "fixed_shape_jax_reduce_time_ms",
                ):
                    if key in info:
                        committed_info[key] = info[key]
                parent.last_info = committed_info
                return committed_chunk, committed_info

        if (
            not info.get("deform_safe", False)
            and self.unsafe_deformation_fallback == "brake"
        ):
            # Last-resort behavior preserves the existing horizon-brake fallback.
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deformation_rejected": True,
                    "fallback_reason": info.get("fallback_reason", "deform_unsafe"),
                }
            )
            parent.last_info = info
            return braked_chunk.reshape(original_shape), info

        self._clear_brake_hold()
        info.update({"safety_mode": "horizon_deform", "mode": "horizon_deform"})
        valid = self._valid_control_indices(chunk)
        if np.any(valid):
            action_idx = parent.controlled_action_indices[valid]
            safe_chunk = self._project_optimized_chunk(safe_chunk, chunk, action_idx)
        parent.last_info = info
        return safe_chunk.reshape(original_shape), info
