from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .safechunk_safety_contract import (
    SafetyConstraintResult,
    SafetyTrace,
    clearance_margin_loss,
    clearance_sequence_from_eval,
    evaluate_clearance_constraint,
    safety_trace_from_eval,
)
from robobase.safetyfilter.gradient_descent import GradientDescent
from robobase.safetyfilter.cross_entropy_method import CrossEntropyMethod

if TYPE_CHECKING:
    from .safechunk_brake import Brake
    from .safechunk_deform import Deform
    from .safechunk_deform_filter import SafeChunkDeformFilter
    from .safechunk_recovery import Recovery


logger = logging.getLogger(__name__)


class InterventionExecutionFactory:
    """Common helpers shared by brake/deform/recovery executors."""

    def __init__(
        self,
        parent: "SafeChunkDeformFilter",
        intervention: Mapping[str, Any] | None = None,
        intervention_factory: "InterventionExecutionFactory" | None = None,
    ) -> None:
        """Create shared intervention state from the full intervention config."""
        self.parent: "SafeChunkDeformFilter" = parent
        source_config = intervention
        if intervention_factory is not None:
            source_config = intervention_factory.intervention
        if source_config is None:
            source_config = getattr(parent, "intervention", None)

        self.intervention: dict[str, Any] = self._coerce_intervention_config(source_config)
        self.intervention_shared_config: dict[str, Any] = {
            key: value
            for key, value in self.intervention.items()
            if key not in {"brake", "deform", "recovery"}
        }
        self.intervention_brake_config: dict[str, Any] = {
            **self.intervention_shared_config,
            **self._coerce_intervention_config(self.intervention.get("brake")),
        }
        self.intervention_deform_config: dict[str, Any] = {
            **self.intervention_shared_config,
            **self._coerce_intervention_config(self.intervention.get("deform")),
        }
        self.intervention_recovery_config: dict[str, Any] = {
            **self.intervention_shared_config,
            **self.intervention_deform_config,
            **self._coerce_intervention_config(self.intervention.get("recovery")),
        }

        self.lambda_deform_safety: float = self._config_float(
            self.intervention_deform_config,
            "lambda_deform_safety",
            800.0,
        )
        self.lambda_deform_action: float = self._config_float(
            self.intervention_deform_config,
            "lambda_deform_action",
            0.1,
        )
        self.lambda_deform_smooth: float = self._config_float(
            self.intervention_deform_config,
            "lambda_deform_smooth",
            0.1,
        )
        self.lambda_retreat: float = self._config_float(
            self.intervention_deform_config,
            "lambda_retreat",
            1.0,
        )
        self.lambda_deform_rejoin: float = self._config_float(
            self.intervention_deform_config,
            "lambda_deform_rejoin",
            1.0,
        )
        self.lambda_deform_rejoin_velocity: float = self._config_float(
            self.intervention_deform_config,
            "lambda_deform_rejoin_velocity",
            0.5,
        )
        self.lambda_deform_rejoin_action: float = self._config_float(
            self.intervention_deform_config,
            "lambda_deform_rejoin_action",
            0.25,
        )
        self.lambda_deform_rejoin_heading: float = self._config_float(
            self.intervention_deform_config,
            "lambda_deform_rejoin_heading",
            0.5,
        )
        self.lambda_return_safety: float = self._config_float(
            self.intervention_recovery_config,
            "lambda_return_safety",
            500.0,
        )
        self.lambda_return_rejoin: float = self._config_float(
            self.intervention_recovery_config,
            "lambda_return_rejoin",
            5.0,
        )
        self.lambda_return_smooth: float = self._config_float(
            self.intervention_recovery_config,
            "lambda_return_smooth",
            0.2,
        )
        self.lambda_return_action: float = self._config_float(
            self.intervention_recovery_config,
            "lambda_return_action",
            0.1,
        )
        self.acceptance_clearance_tol: float = self._config_float(
            self.intervention_deform_config,
            "acceptance_clearance_tol",
            0.005,
        )
        self.deform_horizon: int = self._config_int(
            self.intervention_recovery_config,
            "deform_horizon",
            self._config_int(self.intervention_deform_config, "deform_horizon", 4),
        )
        self.return_horizon: int = self._config_int(
            self.intervention_recovery_config,
            "return_horizon",
            8,
        )
        self.max_return_retries: int = self._config_int(
            self.intervention_recovery_config,
            "max_return_retries",
            3,
        )
        self.latest_nominal_rejoin_target_info: dict[str, Any] | None = None
        self.latest_nominal_rejoin_target_step: int | None = None
        self.latest_nominal_rejoin_target_obs_id: int | None = None
        self._intervention_weights_synced: bool = True

        self.sync_filter_context()
        self.attach_executors(
            brake=getattr(parent, "brake", None),
            deform=getattr(parent, "deform", None),
            recovery=getattr(parent, "recovery", None),
        )

    def sync_filter_context(self) -> None:
        """Copy filter-owned constructor fields used by shared intervention helpers."""
        self.oscbf_operator: Any = self.parent.oscbf_operator
        self._operator_instantiation_failed: bool = self.parent._operator_instantiation_failed
        self.horizon: int = self.parent.horizon
        self.dt: float = self.parent.dt
        self.action_dim: int = self.parent.action_dim
        self.expected_motion_dim: int = self.parent.expected_motion_dim
        self.control_type: str = self.parent.control_type
        self.controlled_action_indices: np.ndarray = np.asarray(
            self.parent.controlled_action_indices,
            dtype=np.int64,
        ).copy()
        self.controlled_state_indices: np.ndarray = np.asarray(
            self.parent.controlled_state_indices,
            dtype=np.int64,
        ).copy()
        self.min_clearance: float = self.parent.min_clearance
        self.diagnostics: dict[str, Any] = dict(self.parent.diagnostics)
        self.rollout_model_config: dict[str, Any] = dict(
            getattr(self.parent, "rollout_model_config", {}) or {}
        )
        self._sync_rollout_model_config()
        self.debug: bool = self.parent.debug
        self.enabled: bool = self.parent.enabled
        self._warned_no_safety_eval: bool = self.parent._warned_no_safety_eval

    def attach_executors(
        self,
        brake: "Brake | None" = None,
        deform: "Deform | None" = None,
        recovery: "Recovery | None" = None,
    ) -> None:
        """Attach sibling executors explicitly after the filter finishes construction."""
        self.brake: "Brake | None" = brake
        self.deform: "Deform | None" = deform
        self.recovery: "Recovery | None" = recovery

    def _optimizer_source(self) -> Any:
        """Return the executor that owns optimizer hyperparameters for this call."""
        if self.deform is self or self.recovery is self:
            return self
        if self.deform is not None:
            return self.deform
        if self.recovery is not None:
            return self.recovery
        raise AttributeError("No optimizer-owning executor is attached")

    @staticmethod
    def _coerce_intervention_config(config: Any | None) -> dict[str, Any]:
        """Normalize intervention config into a string-keyed mapping."""
        if config is None:
            return {}
        if hasattr(config, "items"):
            return {str(k): v for k, v in config.items()}
        return {str(k): v for k, v in dict(config).items()}

    def _config_float(
        self,
        config: Mapping[str, Any],
        key: str,
        default: float,
    ) -> float:
        """Read a float config value from a normalized intervention section."""
        return self._coerce_float(config.get(key, default), default)

    def _config_int(
        self,
        config: Mapping[str, Any],
        key: str,
        default: int,
    ) -> int:
        """Read an int config value from a normalized intervention section."""
        try:
            return int(config.get(key, default))
        except (TypeError, ValueError):
            return int(default)

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        """Coerce config-like values to float with a safe fallback."""
        try:
            return float(value)
        except (TypeError, ValueError):
            arr = np.asarray(value)
            if arr.size == 0:
                return float(default)
            return float(np.asarray(arr, dtype=np.float64).reshape(-1)[0])

    def _sync_intervention_weights(self) -> None:
        """Compatibility hook; intervention fields are initialized explicitly."""
        self._intervention_weights_synced = True

    def _rollout_vector_config(
        self,
        key: str,
        *,
        length: int,
        default: float,
        positive: bool = False,
        cfg: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Read scalar/list rollout calibration values as a fixed state vector."""
        source = self.rollout_model_config if cfg is None else cfg
        value = source.get(key, None)
        if value is None:
            arr = np.full((length,), float(default), dtype=np.float32)
        else:
            try:
                raw = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:  # noqa: BLE001
                raw = np.asarray([], dtype=np.float32)
            if raw.size == 1:
                arr = np.full((length,), float(raw[0]), dtype=np.float32)
            elif raw.size >= length:
                arr = raw[:length].astype(np.float32, copy=True)
            else:
                arr = np.full((length,), float(default), dtype=np.float32)
                if raw.size:
                    arr[: raw.size] = raw
        arr = np.where(np.isfinite(arr), arr, float(default)).astype(np.float32)
        if positive:
            arr = np.maximum(arr, 0.0)
        return arr

    def _rollout_index_set_config(
        self,
        key: str,
        cfg: Mapping[str, Any] | None = None,
    ) -> set[int]:
        """Read optional state-index override lists for rollout control modes."""
        source = self.rollout_model_config if cfg is None else cfg
        value = source.get(key, None)
        if value is None:
            return set()
        try:
            return {int(x) for x in np.asarray(value).reshape(-1).tolist()}
        except Exception:  # noqa: BLE001
            return set()

    def _rollout_profile_from_config(
        self,
        cfg: Mapping[str, Any],
        dim: int,
    ) -> dict[str, Any]:
        """Build one calibrated rollout parameter profile."""
        profile_cfg = dict(cfg or {})
        action_scale = self._rollout_vector_config(
            "per_state_action_scale",
            length=dim,
            default=1.0,
            cfg=profile_cfg,
        )
        action_bias = self._rollout_vector_config(
            "per_state_action_bias",
            length=dim,
            default=0.0,
            cfg=profile_cfg,
        )
        delta_scale = self._rollout_vector_config(
            "per_state_delta_scale",
            length=dim,
            default=1.0,
            cfg=profile_cfg,
        )
        velocity_scale = self._rollout_vector_config(
            "per_state_velocity_scale",
            length=dim,
            default=1.0,
            cfg=profile_cfg,
        )
        target_alpha = self._rollout_vector_config(
            "per_state_target_alpha",
            length=dim,
            default=1.0,
            positive=True,
            cfg=profile_cfg,
        )
        max_step = self._rollout_vector_config(
            "per_state_max_step",
            length=dim,
            default=np.inf,
            positive=True,
            cfg=profile_cfg,
        )

        # Scalar shorthand values are fallback defaults only.  Per-state
        # calibration arrays must win; otherwise a default base_delta_scale=1.0
        # or arm_target_alpha=1.0 silently clobbers calibrated damping gains.
        base_delta_scale = profile_cfg.get("base_delta_scale", None)
        if base_delta_scale is not None and profile_cfg.get("per_state_delta_scale", None) is None:
            base_n = min(4, dim)
            delta_scale[:base_n] = self._rollout_vector_config(
                "base_delta_scale",
                length=base_n,
                default=1.0,
                cfg=profile_cfg,
            )[:base_n]
        arm_target_alpha = profile_cfg.get("arm_target_alpha", None)
        if (
            arm_target_alpha is not None
            and dim > 4
            and profile_cfg.get("per_state_target_alpha", None) is None
        ):
            target_alpha[4:] = self._rollout_vector_config(
                "arm_target_alpha",
                length=dim - 4,
                default=1.0,
                positive=True,
                cfg=profile_cfg,
            )[: dim - 4]

        return {
            "enabled": bool(profile_cfg.get("enabled", True)),
            "base_delta_always": bool(profile_cfg.get("base_delta_always", True)),
            "action_scale": action_scale,
            "action_bias": action_bias,
            "delta_scale": delta_scale,
            "velocity_scale": velocity_scale,
            "target_alpha": target_alpha,
            "max_step": max_step,
            "absolute_state_indices": self._rollout_index_set_config(
                "absolute_state_indices", cfg=profile_cfg
            ),
            "delta_state_indices": self._rollout_index_set_config(
                "delta_state_indices", cfg=profile_cfg
            ),
            "velocity_state_indices": self._rollout_index_set_config(
                "velocity_state_indices", cfg=profile_cfg
            ),
        }

    def _rollout_profile_config(
        self,
        cfg: Mapping[str, Any],
        name: str,
    ) -> dict[str, Any]:
        """Merge a named rollout profile override onto the root profile."""
        merged = {
            k: v
            for k, v in dict(cfg or {}).items()
            if not isinstance(v, Mapping) or k not in {"nominal", "recovery", "mpc"}
        }
        override = cfg.get(name, None) if isinstance(cfg, Mapping) else None
        if override is None and name == "recovery" and isinstance(cfg, Mapping):
            override = cfg.get("mpc", None)
        if isinstance(override, Mapping):
            merged.update(dict(override))
        return merged

    def _set_rollout_profile_attrs(self, profile: Mapping[str, Any]) -> None:
        """Keep legacy attribute names pointing at the active nominal profile."""
        self.rollout_model_enabled = bool(profile.get("enabled", True))
        self.rollout_base_delta_always = bool(profile.get("base_delta_always", True))
        self.rollout_action_scale_by_state = profile["action_scale"]
        self.rollout_action_bias_by_state = profile["action_bias"]
        self.rollout_delta_scale_by_state = profile["delta_scale"]
        self.rollout_velocity_scale_by_state = profile["velocity_scale"]
        self.rollout_target_alpha_by_state = profile["target_alpha"]
        self.rollout_max_step_by_state = profile["max_step"]
        self.rollout_absolute_state_indices = profile["absolute_state_indices"]
        self.rollout_delta_state_indices = profile["delta_state_indices"]
        self.rollout_velocity_state_indices = profile["velocity_state_indices"]

    def _rollout_profile(self) -> Mapping[str, Any]:
        """Return the rollout profile for the current executor/regime."""
        profiles = getattr(self, "rollout_model_profiles", {}) or {}
        variant = str(getattr(self, "rollout_model_default_variant", "nominal") or "nominal")
        return profiles.get(variant) or profiles.get("nominal") or {}

    def _sync_rollout_model_config(self) -> None:
        """Prepare shared calibrated rollout profiles for SafeChunk users."""
        cfg = self.rollout_model_config
        self.rollout_model_state_dim = max(1, int(self.expected_motion_dim))
        dim = self.rollout_model_state_dim
        nominal_cfg = self._rollout_profile_config(cfg, "nominal")
        recovery_cfg = self._rollout_profile_config(cfg, "recovery")
        nominal_profile = self._rollout_profile_from_config(nominal_cfg, dim)
        recovery_profile = self._rollout_profile_from_config(recovery_cfg, dim)
        self.rollout_model_profiles = {
            "nominal": nominal_profile,
            "act": nominal_profile,
            "default": nominal_profile,
            "recovery": recovery_profile,
            "mpc": recovery_profile,
        }
        self._set_rollout_profile_attrs(nominal_profile)

    @property
    def info(self) -> dict[str, Any]:
        """Expose the parent filter info dictionary for shared helpers."""
        return self.parent.last_info

    def _optimize_controlled_chunk(
        self,
        obs,
        nominal_chunk,
        action_idx,
        cost_fn,
        seed_chunks=None,
        batch_cost_fn=None,
        early_stop_fn=None,
        min_iters: int = 1,
        optimizer_stage: str | None = None,
    ):
        del obs
        nominal_chunk = np.asarray(nominal_chunk, dtype=np.float32)
        optimizer_source = self._optimizer_source()
        optimizer_method = str(getattr(optimizer_source, "optimizer_method", "cem")).lower()
        opt_start = time.perf_counter()

        if optimizer_method == "gradient":
            optimizer = GradientDescent(
                rng=optimizer_source._rng,
                max_iters=max(1, optimizer_source.opt_iters),
                min_iters=max(1, int(min_iters)),
                lr=optimizer_source.opt_lr,
                gradient_samples=max(1, int(getattr(optimizer_source, "gradient_samples", 4))),
                eps=float(getattr(optimizer_source, "gradient_eps", 0.01)),
                adam_beta1=float(getattr(optimizer_source, "gradient_adam_beta1", 0.9)),
                adam_beta2=float(getattr(optimizer_source, "gradient_adam_beta2", 0.999)),
                min_improvement=float(getattr(optimizer_source, "gradient_min_improvement", 1e-6)),
                line_search_scales=tuple(
                    getattr(optimizer_source, "gradient_line_search_scales", (1.0, 0.5, 0.25))
                ),
                batched_line_search=bool(
                    getattr(optimizer_source, "gradient_batched_line_search", True)
                ),
                early_stop_on_candidate=bool(
                    getattr(optimizer_source, "gradient_early_stop_on_candidate", True)
                ),
            )
            best_record = optimizer.optimize(
                nominal_chunk=nominal_chunk,
                action_idx=action_idx,
                cost_fn=cost_fn,
                seed_chunks=seed_chunks,
                batch_cost_fn=batch_cost_fn,
                early_stop_fn=early_stop_fn,
                project_fn=lambda candidate: self.deform._project_optimized_chunk(
                    candidate,
                    nominal_chunk,
                    action_idx,
                ),
            )
        elif optimizer_method == "cem":
            optimizer = CrossEntropyMethod(
                rng=optimizer_source._rng,
                max_iters=max(1, optimizer_source.opt_iters),
                min_iters=max(1, int(min_iters)),
                population=max(1, int(optimizer_source.opt_population)),
                elite_frac=float(optimizer_source.opt_elite_frac),
                sigma=float(optimizer_source.opt_lr),
            )
            best_record = optimizer.optimize(
                nominal_chunk=nominal_chunk,
                action_idx=action_idx,
                cost_fn=cost_fn,
                seed_chunks=seed_chunks,
                batch_cost_fn=batch_cost_fn,
                early_stop_fn=early_stop_fn,
                project_fn=lambda candidate: self.deform._project_optimized_chunk(
                    candidate,
                    nominal_chunk,
                    action_idx,
                ),
                project_population_fn=optimizer_source._jax_project_candidate_population,
            )
        else:
            raise ValueError(f"Unknown optimizer_method={optimizer_method}; must be 'gradient' or 'cem'.")
        if best_record is None:
            raise RuntimeError("No candidate produced by optimizer")

        opt_elapsed_ms = 1000.0 * (time.perf_counter() - opt_start)
        best_record.setdefault("losses", {})
        best_record["losses"]["optimizer_method"] = optimizer_method
        if optimizer_stage == "deform":
            best_record["losses"]["deform_optimizer_method"] = optimizer_method
            best_record["losses"]["deform_optimizer_time_ms"] = float(opt_elapsed_ms)
        elif optimizer_stage == "return":
            best_record["losses"]["return_optimizer_method"] = optimizer_method
            best_record["losses"]["return_optimizer_time_ms"] = float(opt_elapsed_ms)
        elif optimizer_stage == "explicit":
            best_record["losses"]["explicit_optimizer_method"] = optimizer_method
            best_record["losses"]["explicit_optimizer_time_ms"] = float(opt_elapsed_ms)
        elif optimizer_stage == "committed_suffix":
            best_record["losses"]["committed_suffix_optimizer_method"] = optimizer_method
            best_record["losses"]["committed_suffix_optimizer_time_ms"] = float(
                opt_elapsed_ms
            )
        return best_record

    def _make_task_progress_recover_chunk(
        self,
        obs,
        q_start,
        current_chunk,
        action_idx,
        context=None,
        default_target_index=None,
    ):
        h = min(current_chunk.shape[0], self.return_horizon)
        recover_chunk = np.asarray(current_chunk[:h], dtype=np.float32).copy()
        target_index = min(self.recovery.min_rejoin_offset, max(0, current_chunk.shape[0] - 1))
        if action_idx.size:
            q_start = np.asarray(q_start, dtype=np.float32).reshape(-1)
            valid = (
                (self.controlled_action_indices < recover_chunk.shape[1])
                & (self.controlled_state_indices < q_start.shape[0])
            )
            allowed_actions = set(np.asarray(action_idx, dtype=np.int64).reshape(-1).tolist())
            if allowed_actions:
                valid &= np.asarray(
                    [idx in allowed_actions for idx in self.controlled_action_indices],
                    dtype=np.bool_,
                )
            local_action_idx = self.controlled_action_indices[valid]
            state_idx = self.controlled_state_indices[valid]
            if local_action_idx.size:
                nominal_q_seq = None if context is None else getattr(context, "nominal_q_seq", None)
                if nominal_q_seq is not None:
                    nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
                if (
                    nominal_q_seq is not None
                    and nominal_q_seq.ndim == 2
                    and nominal_q_seq.shape[0] > 0
                    and np.all(state_idx < nominal_q_seq.shape[1])
                ):
                    window_info = InterventionExecutionFactory._select_safe_nominal_rejoin_window(
                        self,
                        obs,
                        nominal_q_seq,
                        candidate_q=q_start,
                        state_idx=state_idx,
                    )
                    if window_info.get("available"):
                        target_index = int(window_info["target_index"])
                    elif default_target_index is not None:
                        target_index = int(np.clip(
                            int(default_target_index),
                            0,
                            max(0, nominal_q_seq.shape[0] - 1),
                        ))
                    elif nominal_q_seq.shape[0] > self.recovery.min_rejoin_offset:
                        future = nominal_q_seq[self.recovery.min_rejoin_offset :, state_idx]
                        _loss, target_index = self._nearest_future_loss(
                            q_start[state_idx],
                            future,
                            weights=None,
                            start_index=self.recovery.min_rejoin_offset,
                        )
                    else:
                        target_index = 0
                    target_rows = []
                    if window_info.get("available"):
                        window_end = int(window_info["end"])
                        target_chunk = InterventionExecutionFactory._frame_stack_rejoin_target_window(
                            self,
                            nominal_q_seq,
                            start=target_index,
                            end=window_end,
                        )
                        target_chunk = np.asarray(target_chunk, dtype=np.float32)
                        if target_chunk.ndim == 2 and target_chunk.shape[0] > 0:
                            for k in range(h):
                                alpha = float(k + 1) / float(max(h, 1))
                                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                                target_row = target_chunk[
                                    min(k, target_chunk.shape[0] - 1)
                                ].copy()
                                bridge_row = q_start.copy()
                                bridge_row[state_idx] = (
                                    (1.0 - smooth_alpha) * q_start[state_idx]
                                    + smooth_alpha * target_row[state_idx]
                                )
                                target_rows.append(bridge_row)
                    else:
                        target_index = None
                        target_rows = [q_start.copy() for _ in range(h)]
                    if target_rows:
                        recover_chunk = self._write_state_tracking_actions(
                            recover_chunk,
                            q_start,
                            np.stack(target_rows, axis=0).astype(np.float32),
                            local_action_idx,
                            state_idx,
                        )
                else:
                    modes = self._control_mode_ids_for_state_indices(state_idx)
                    first = np.zeros(local_action_idx.shape, dtype=recover_chunk.dtype)
                    absolute = modes == 0
                    if np.any(absolute):
                        first[absolute] = q_start[state_idx[absolute]]
                    recover_chunk[0, local_action_idx] = first
        passthrough_idx = [
            i for i in range(current_chunk.shape[1]) if i not in set(action_idx.tolist())
        ]
        recover_chunk[:, passthrough_idx] = current_chunk[:h, passthrough_idx]
        return recover_chunk, target_index

    def _clearance_sequence_batch_from_eval(self, safety_eval, batch: int, horizon: int):
        h_seq = np.asarray(
            safety_eval.get("min_clearances", safety_eval.get("clearances", [])),
            dtype=np.float32,
        )
        if h_seq.shape == (batch, horizon):
            return h_seq
        min_h = np.asarray(safety_eval.get("min_clearance", np.inf), dtype=np.float32).reshape(-1)
        if min_h.size == 1:
            min_h = np.full(batch, float(min_h[0]), dtype=np.float32)
        if min_h.size != batch:
            min_h = np.full(batch, np.inf, dtype=np.float32)
        return np.repeat(min_h[:, None], horizon, axis=1).astype(np.float32)

    def _deform_stage_deformation_cost(self, obs, candidate, nominal, action_idx):
        self._sync_intervention_weights()
        q_seq = self.rollout_nominal_chunk(obs, candidate)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        constraint = self.clearance_constraint_from_eval(
            safety_eval,
            q_seq.shape[0],
            self.min_clearance,
        )
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        safety_loss = constraint.margin_loss
        action_deviation_loss = (
            float(np.square(candidate[:, action_idx] - nominal[:, action_idx]).mean())
            if len(action_idx)
            else 0.0
        )
        smoothness_loss = self._smoothness_loss(candidate, action_idx)
        finite_h = np.nan_to_num(h_seq, nan=0.0, posinf=self.min_clearance, neginf=-1.0)
        retreat_loss = -float(np.mean(np.clip(finite_h, -1.0, 1.0)))
        rejoin_terms = self._deform_terminal_rejoin_terms(
            obs,
            candidate,
            q_seq,
            action_idx,
        )
        deform_rejoin_loss = float(rejoin_terms["deform_rejoin_window_loss"])
        total_loss = float(
            self.lambda_deform_safety * safety_loss
            + self.lambda_deform_action * action_deviation_loss
            + self.lambda_deform_smooth * smoothness_loss
            + self.lambda_retreat * retreat_loss
            + self.lambda_deform_rejoin * deform_rejoin_loss
        )
        return total_loss, {
            "safety_loss": safety_loss,
            "action_deviation_loss": action_deviation_loss,
            "smoothness_loss": smoothness_loss,
            "retreat_loss": retreat_loss,
            "deform_rejoin_window_loss": deform_rejoin_loss,
            "deform_rejoin_weight": float(self.lambda_deform_rejoin),
            "total_loss": total_loss,
            "min_clearance": float(constraint.min_clearance),
            **rejoin_terms,
        }

    def _deform_stage_deformation_cost_batch(self, obs, candidates, nominal, action_idx):
        self._sync_intervention_weights()
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
        required_clearance = float(self.recovery._acceptance_clearance_threshold())
        safety_loss = np.square(np.maximum(required_clearance - h_seq, 0.0)).sum(axis=1)
        if len(action_idx):
            action_deviation_loss = np.square(
                candidates[:, :, action_idx] - nominal[None, :, action_idx]
            ).mean(axis=(1, 2))
        else:
            action_deviation_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        smoothness_loss = self._smoothness_loss_batch(candidates, action_idx)
        finite_h = np.nan_to_num(h_seq, nan=0.0, posinf=self.min_clearance, neginf=-1.0)
        retreat_loss = -np.mean(np.clip(finite_h, -1.0, 1.0), axis=1)
        deform_rejoin_loss, rejoin_terms_list = self._deform_terminal_rejoin_terms_batch(
            obs,
            candidates,
            q_seq_batch,
            action_idx,
        )
        total_loss = (
            self.lambda_deform_safety * safety_loss
            + self.lambda_deform_action * action_deviation_loss
            + self.lambda_deform_smooth * smoothness_loss
            + self.lambda_retreat * retreat_loss
            + self.lambda_deform_rejoin * deform_rejoin_loss
        )
        losses = [
            {
                "safety_loss": float(safety_loss[i]),
                "action_deviation_loss": float(action_deviation_loss[i]),
                "smoothness_loss": float(smoothness_loss[i]),
                "retreat_loss": float(retreat_loss[i]),
                "deform_rejoin_window_loss": float(deform_rejoin_loss[i]),
                "deform_rejoin_weight": float(self.lambda_deform_rejoin),
                "total_loss": float(total_loss[i]),
                "min_clearance": float(np.min(h_seq[i])),
                "batched_optimizer": True,
                "jax_batched_optimizer": bool(self._jax_optimizer_ready()),
                "jax_rollout_time_ms": float(rollout_time_ms) / max(1, candidates.shape[0]),
                **rejoin_terms_list[i],
            }
            for i in range(candidates.shape[0])
        ]
        return total_loss.astype(np.float32), losses

    def _make_chunk_deformation_candidates(self, obs, chunk, safety_info):
        valid = self._valid_control_indices(chunk)
        if not np.any(valid):
            return [(None, chunk.copy())]

        action_idx = self.controlled_action_indices[valid]
        state_idx = self.controlled_state_indices[valid]
        anchor = self.deform._controlled_anchor(obs, chunk, action_idx, state_idx)
        start_idx = self._deformation_start_idx(safety_info, chunk.shape[0])
        candidates = []
        seen = set()

        for scale in self.deform.chunk_deformation_scales:
            scale = float(np.clip(scale, 0.0, 1.0))
            candidate = chunk.copy()
            profile = np.ones(chunk.shape[0], dtype=np.float32)
            profile[start_idx:] = scale
            nominal = chunk[:, action_idx]
            candidate[:, action_idx] = anchor + profile[:, None] * (nominal - anchor)
            candidate = self.deform._clip_controlled_delta(candidate, chunk, action_idx)
            candidate = self._smooth_controlled_suffix(candidate, action_idx, start_idx)
            candidate = self.deform._clip_controlled_delta(candidate, chunk, action_idx)
            passthrough_idx = [
                i for i in range(chunk.shape[1]) if i not in set(action_idx.tolist())
            ]
            candidate[:, passthrough_idx] = chunk[:, passthrough_idx]
            key = tuple(np.round(candidate[:, action_idx].reshape(-1), 8))
            if key not in seen:
                candidates.append((scale, candidate))
                seen.add(key)

        return candidates

    def _deformation_start_idx(self, safety_info, horizon):
        first_violation = safety_info.get("first_violation")
        if first_violation is None:
            return 0
        return max(0, min(int(first_violation) - 1, horizon - 1))

    def _smooth_controlled_suffix(self, candidate, action_idx, start_idx):
        if self.deform.chunk_deformation_smoothing <= 0 or candidate.shape[0] <= 2:
            return candidate
        smoothed = candidate.copy()
        for _ in range(self.deform.chunk_deformation_smoothing):
            prev = smoothed.copy()
            for k in range(max(1, start_idx), candidate.shape[0] - 1):
                smoothed[k, action_idx] = (
                    0.25 * prev[k - 1, action_idx]
                    + 0.5 * prev[k, action_idx]
                    + 0.25 * prev[k + 1, action_idx]
                )
        return smoothed

    def _q_rejoin_loss_batch(self, q_seq_batch, nominal_q_seq=None, rejoin_context=None):
        t0 = time.perf_counter()
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq)
        q_seq_batch = np.asarray(q_seq_batch, dtype=np.float32)
        batch = q_seq_batch.shape[0]
        state_idx = rejoin_context.get("q_state_indices")
        future = rejoin_context.get("q_nom_future")
        weights = rejoin_context.get("q_weights")
        if state_idx is None or future is None or q_seq_batch.shape[1] == 0:
            return (
                np.full(batch, np.inf, dtype=np.float32),
                [None] * batch,
                (time.perf_counter() - t0) * 1000.0,
            )
        valid = state_idx < q_seq_batch.shape[2]
        if not np.all(valid):
            state_idx = state_idx[valid]
            future = future[:, valid]
            weights = None if weights is None else weights[valid]
        if state_idx.size == 0:
            return (
                np.full(batch, np.inf, dtype=np.float32),
                [None] * batch,
                (time.perf_counter() - t0) * 1000.0,
            )
        final_state = q_seq_batch[:, -1, :][:, state_idx]
        diff = final_state[:, None, :] - future[None, :, :]
        if weights is not None:
            diff = diff * np.asarray(weights, dtype=np.float32).reshape(1, 1, -1)
        losses_by_index = np.square(diff).sum(axis=2)
        j_best = np.argmin(losses_by_index, axis=1)
        losses = losses_by_index[np.arange(batch), j_best].astype(np.float32)
        start_index = int(
            rejoin_context.get("q_nom_future_start_index", self.recovery.min_rejoin_offset)
        )
        indices = [
            None if not np.isfinite(losses[i]) else int(j_best[i] + start_index)
            for i in range(batch)
        ]
        return losses, indices, (time.perf_counter() - t0) * 1000.0

    def _smoothness_loss_batch(self, chunks, action_idx):
        chunks = np.asarray(chunks, dtype=np.float32)
        if len(action_idx) == 0 or chunks.shape[1] <= 1:
            return np.zeros(chunks.shape[0], dtype=np.float32)
        controlled = chunks[:, :, action_idx]
        velocity_loss = np.square(np.diff(controlled, axis=1)).mean(axis=(1, 2))
        if chunks.shape[1] <= 2:
            return velocity_loss.astype(np.float32)
        acc = controlled[:, 2:, :] - 2.0 * controlled[:, 1:-1, :] + controlled[:, :-2, :]
        return (velocity_loss + 0.5 * np.square(acc).mean(axis=(1, 2))).astype(np.float32)

    def _nominal_rejoin_store(self) -> "InterventionExecutionFactory":
        """Return the central factory object that owns shared rejoin-window state."""
        store = getattr(self.parent, "intervention_factory", None)
        return store if store is not None else self

    def _recovery_owner(self) -> Any:
        """Return the recovery executor that owns recovery-specific config/state."""
        return self.recovery if self.recovery is not None else self

    def _nominal_rejoin_empty_info(
        self,
        reason: str,
        *,
        safe_prefix_len: int = 0,
        clearance: float = float("-inf"),
        stale_blocked: bool = False,
        window_count: int = 0,
        full_window_count: int = 0,
        staging_window_count: int = 0,
    ) -> dict[str, Any]:
        """Build unavailable nominal-window diagnostics for shared rejoin search."""
        required_clearance: float = float(self._acceptance_clearance_threshold())
        return {
            "available": False,
            "target_chunk": None,
            "target_q_seq": None,
            "safe_prefix_len": int(safe_prefix_len),
            "suppressed_reason": str(reason),
            "nominal_rejoin_clearance": float(clearance),
            "nominal_rejoin_required_clearance": required_clearance,
            "nominal_rejoin_stale_blocked": bool(stale_blocked),
            "nominal_rejoin_window_count": int(window_count),
            "nominal_rejoin_full_window_count": int(full_window_count),
            "nominal_rejoin_staging_window_count": int(staging_window_count),
            "safe_rejoin_window_found": False,
            "short_staging_window_found": False,
        }

    def _safe_nominal_windows_from_clearance(
        self,
        h_seq: Any,
        *,
        start_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Split nominal ACT clearance into consecutive full/staging safe windows."""
        recovery = self._recovery_owner()
        h_arr: np.ndarray = np.asarray(h_seq, dtype=np.float32).reshape(-1)
        if h_arr.size == 0:
            return []
        required_clearance: float = float(self._acceptance_clearance_threshold())
        full_len: int = max(1, int(getattr(recovery, "recover_safe_rejoin_window_len", 1)))
        staging_len: int = max(1, int(getattr(recovery, "recover_staging_rejoin_window_min_len", full_len)))
        default_start = int(getattr(recovery, "min_rejoin_offset", 0))
        start_min: int = default_start if start_index is None else int(start_index)
        start_min = max(0, min(start_min, h_arr.shape[0] - 1))
        safe_mask: np.ndarray = h_arr >= required_clearance
        windows: list[dict[str, Any]] = []
        idx: int = start_min
        while idx < h_arr.shape[0]:
            if not bool(safe_mask[idx]):
                idx += 1
                continue
            start: int = idx
            while idx + 1 < h_arr.shape[0] and bool(safe_mask[idx + 1]):
                idx += 1
            end: int = idx
            length: int = end - start + 1
            if length >= staging_len:
                windows.append(
                    {
                        "start": int(start),
                        "end": int(end),
                        "length": int(length),
                        "min_clearance": float(np.min(h_arr[start : end + 1])),
                        "window_type": "full" if length >= full_len else "staging",
                    }
                )
            idx += 1
        return windows

    def _frame_stack_rejoin_target_window(
        self,
        nominal_seq: Any,
        *,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Build an ACT frame-stack-sized target from a safe nominal/staging window.

        Full windows are capped to one ACT history. Short staging windows are
        extended with their local tangent, avoiding unsafe nominal samples while
        still giving the optimizer a heading-consistent target.
        """
        recovery = self._recovery_owner()
        nominal_arr: np.ndarray = np.asarray(nominal_seq, dtype=np.float32)
        if nominal_arr.ndim != 2 or nominal_arr.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)

        target_len: int = max(1, int(getattr(recovery, "recover_act_frame_stack", 1)))
        start_i: int = max(0, min(int(start), nominal_arr.shape[0] - 1))
        end_i: int = max(start_i, min(int(end), nominal_arr.shape[0] - 1))
        safe_segment: np.ndarray = nominal_arr[start_i : end_i + 1].copy()
        if safe_segment.shape[0] >= target_len:
            return safe_segment[:target_len].astype(np.float32, copy=False)
        if safe_segment.shape[0] < 2:
            return safe_segment.astype(np.float32, copy=False)

        tangent: np.ndarray = safe_segment[-1] - safe_segment[-2]
        extra_count: int = target_len - int(safe_segment.shape[0])
        extra_scale: np.ndarray = np.arange(
            1,
            extra_count + 1,
            dtype=np.float32,
        ).reshape(-1, 1)
        extrapolated: np.ndarray = (
            safe_segment[-1].reshape(1, -1) + extra_scale * tangent.reshape(1, -1)
        )
        return np.concatenate([safe_segment, extrapolated.astype(np.float32)], axis=0)

    def _select_safe_nominal_rejoin_window(
        self,
        obs: Any,
        nominal_q_seq: Any,
        *,
        candidate_q: Any | None = None,
        state_idx: Any | None = None,
        nominal_action_seq: Any | None = None,
        require_live_prefix_safe: bool = False,
        live_prefix_len: int | None = None,
        allow_best_live_prefix_when_unsafe: bool = False,
    ) -> dict[str, Any]:
        """Choose the safe nominal ACT-like window that is reachable from a candidate."""
        nominal_arr: np.ndarray = np.asarray(nominal_q_seq, dtype=np.float32)
        if nominal_arr.ndim != 2 or nominal_arr.shape[0] == 0:
            return {"available": False, "reason": "empty_nominal_q_seq"}
        recovery = self._recovery_owner()
        required_clearance: float = float(self._acceptance_clearance_threshold())
        live_required_clearance: float = max(
            required_clearance,
            float(
                getattr(
                    recovery,
                    "opportunistic_resume_min_clearance",
                    required_clearance,
                )
            ),
        )
        safety_eval: dict[str, Any] = self.evaluate_horizon_safety(obs, nominal_arr)
        h_seq: np.ndarray = np.asarray(
            self._clearance_sequence_from_eval(safety_eval, nominal_arr.shape[0]),
            dtype=np.float32,
        ).reshape(-1)
        windows: list[dict[str, Any]] = self._safe_nominal_windows_from_clearance(h_seq)
        if not windows:
            return {
                "available": False,
                "reason": "no_safe_nominal_window",
                "h_seq": h_seq,
                "windows": [],
                "clearance": float(np.max(h_seq)) if h_seq.size else float("-inf"),
            }
        full_windows: list[dict[str, Any]] = [w for w in windows if w["window_type"] == "full"]
        staging_windows: list[dict[str, Any]] = [w for w in windows if w["window_type"] == "staging"]
        candidates: list[dict[str, Any]] = full_windows if full_windows else staging_windows
        action_arr: np.ndarray | None = None
        if nominal_action_seq is not None:
            action_arr = np.asarray(nominal_action_seq, dtype=np.float32)
            if action_arr.ndim != 2 or action_arr.shape[0] == 0:
                action_arr = None
        if candidate_q is not None and state_idx is not None:
            q = np.asarray(candidate_q, dtype=np.float32).reshape(-1)
            idx_arr = np.asarray(state_idx, dtype=np.int64).reshape(-1)
            idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < min(q.shape[0], nominal_arr.shape[1]))]
        else:
            q = None
            idx_arr = np.asarray([], dtype=np.int64)
        best: dict[str, Any] | None = None
        best_score: float | None = None
        live_eval_count: int = 0
        live_safe_count: int = 0
        live_best_min: float = float("-inf")
        live_best_start: int | None = None
        live_skip_reason: str | None = None
        live_fallback_best: dict[str, Any] | None = None
        live_fallback_score: tuple[float, float, float] | None = None
        live_fallback_selected: bool = False
        for window in candidates:
            live_prefix_safe: bool = True
            live_prefix_min: float | None = None
            live_prefix_horizon: int | None = None
            if require_live_prefix_safe:
                live_eval_count += 1
                live_prefix_safe = False
                if action_arr is None:
                    live_skip_reason = "missing_nominal_action_chunk"
                    continue
                try:
                    target_window = self._frame_stack_rejoin_target_window(
                        action_arr,
                        start=int(window["start"]),
                        end=int(window["end"]),
                    )
                    if target_window.ndim != 2 or target_window.shape[0] == 0:
                        live_skip_reason = "empty_live_prefix_action_window"
                        continue
                    live_prefix_horizon = max(
                        1,
                        min(
                            int(target_window.shape[0]),
                            int(live_prefix_len)
                            if live_prefix_len is not None
                            else int(target_window.shape[0]),
                        ),
                    )
                    live_q_seq = self.rollout_nominal_chunk(
                        obs,
                        target_window[:live_prefix_horizon],
                    )
                    live_safety_eval = self.evaluate_horizon_safety(obs, live_q_seq)
                    live_h_seq = np.asarray(
                        self._clearance_sequence_from_eval(
                            live_safety_eval,
                            np.asarray(live_q_seq).shape[0],
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    if live_h_seq.size == 0:
                        live_h_seq = np.asarray([float("-inf")], dtype=np.float32)
                    live_prefix_min = float(np.min(live_h_seq))
                    if live_prefix_min > live_best_min:
                        live_best_min = float(live_prefix_min)
                        live_best_start = int(window["start"])
                    live_prefix_safe = bool(live_prefix_min >= live_required_clearance)
                    live_candidate = dict(window)
                    live_candidate.update(
                        {
                            "live_prefix_required": bool(require_live_prefix_safe),
                            "live_prefix_safe": bool(live_prefix_safe),
                            "live_prefix_min_clearance": float(live_prefix_min),
                            "live_prefix_required_clearance": float(live_required_clearance),
                            "live_prefix_horizon": int(live_prefix_horizon),
                        }
                    )
                    nearest_dist = 0.0
                    if q is not None and idx_arr.size:
                        segment = nominal_arr[
                            window["start"] : window["end"] + 1,
                            idx_arr,
                        ]
                        dists = np.linalg.norm(
                            segment - q[idx_arr].reshape(1, -1),
                            axis=1,
                        )
                        nearest_dist = float(np.min(dists))
                    fallback_score = (
                        float(live_prefix_min),
                        -float(nearest_dist),
                        -float(window["start"]),
                    )
                    if live_fallback_score is None or fallback_score > live_fallback_score:
                        live_fallback_best = live_candidate
                        live_fallback_score = fallback_score
                    if not live_prefix_safe:
                        live_skip_reason = "live_prefix_unsafe"
                        continue
                    live_safe_count += 1
                except Exception as exc:  # noqa: BLE001
                    live_skip_reason = f"live_prefix_check_failed:{exc}"
                    continue
            score: float = float(window["start"])
            if q is not None and idx_arr.size:
                segment = nominal_arr[window["start"] : window["end"] + 1, idx_arr]
                dists = np.linalg.norm(segment - q[idx_arr].reshape(1, -1), axis=1)
                score += 0.25 * float(np.min(dists))
            score -= 0.01 * float(window["min_clearance"])
            if best_score is None or score < best_score:
                best = dict(window)
                best.update(
                    {
                        "live_prefix_required": bool(require_live_prefix_safe),
                        "live_prefix_safe": bool(live_prefix_safe),
                        "live_prefix_min_clearance": live_prefix_min,
                        "live_prefix_required_clearance": float(live_required_clearance),
                        "live_prefix_horizon": live_prefix_horizon,
                    }
                )
                best_score = score
        if (
            best is None
            and require_live_prefix_safe
            and allow_best_live_prefix_when_unsafe
            and live_fallback_best is not None
        ):
            best = live_fallback_best
            live_fallback_selected = True
        if best is None:
            return {
                "available": False,
                "reason": (
                    "no_live_safe_nominal_window"
                    if require_live_prefix_safe
                    else "no_reachable_safe_nominal_window"
                ),
                "windows": windows,
                "full_window_count": len(full_windows),
                "staging_window_count": len(staging_windows),
                "live_prefix_required": bool(require_live_prefix_safe),
                "live_prefix_allow_fallback": bool(allow_best_live_prefix_when_unsafe),
                "live_prefix_fallback_selected": False,
                "live_prefix_eval_count": int(live_eval_count),
                "live_prefix_safe_count": int(live_safe_count),
                "live_prefix_best_min_clearance": float(live_best_min),
                "live_prefix_best_start": live_best_start,
                "live_prefix_required_clearance": float(live_required_clearance),
                "live_prefix_skip_reason": live_skip_reason,
            }
        return {
            "available": True,
            "target_index": int(best["start"]),
            "start": int(best["start"]),
            "end": int(best["end"]),
            "length": int(best["length"]),
            "window_type": str(best["window_type"]),
            "min_clearance": float(best["min_clearance"]),
            "h_seq": h_seq,
            "windows": windows,
            "full_window_count": len(full_windows),
            "staging_window_count": len(staging_windows),
            "live_prefix_required": bool(require_live_prefix_safe),
            "live_prefix_allow_fallback": bool(allow_best_live_prefix_when_unsafe),
            "live_prefix_fallback_selected": bool(live_fallback_selected),
            "target_source": (
                "best_live_clearance_rejoin_window"
                if live_fallback_selected
                else (
                    "live_safe_rejoin_window"
                    if require_live_prefix_safe
                    else "nominal_safe_rejoin_window"
                )
            ),
            "live_prefix_safe": bool(best.get("live_prefix_safe", not require_live_prefix_safe)),
            "live_prefix_min_clearance": best.get("live_prefix_min_clearance"),
            "live_prefix_required_clearance": float(live_required_clearance),
            "live_prefix_horizon": best.get("live_prefix_horizon"),
            "live_prefix_eval_count": int(live_eval_count),
            "live_prefix_safe_count": int(live_safe_count),
            "live_prefix_best_min_clearance": float(live_best_min),
            "live_prefix_best_start": live_best_start,
        }

    def _cache_nominal_rejoin_target(
        self,
        obs: Any,
        target_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Store the latest shared rejoin target on the central factory object."""
        recovery = self._recovery_owner()
        store = self._nominal_rejoin_store()
        cached = dict(target_info)
        if bool(cached.get("nominal_rejoin_live_prefix_required", False)):
            return cached
        store.latest_nominal_rejoin_target_info = cached
        store.latest_nominal_rejoin_target_step = getattr(recovery, "latest_nominal_step", None)
        store.latest_nominal_rejoin_target_obs_id = id(obs)
        return cached

    def _cached_nominal_rejoin_target(self, obs: Any) -> dict[str, Any] | None:
        """Return the shared target cached for this exact observation/nominal step."""
        recovery = self._recovery_owner()
        store = self._nominal_rejoin_store()
        cached = getattr(store, "latest_nominal_rejoin_target_info", None)
        if not isinstance(cached, dict):
            return None
        if getattr(store, "latest_nominal_rejoin_target_obs_id", None) != id(obs):
            return None
        if getattr(store, "latest_nominal_rejoin_target_step", None) != getattr(
            recovery,
            "latest_nominal_step",
            None,
        ):
            return None
        return dict(cached)

    def get_nominal_rejoin_target(
        self,
        obs: Any,
        candidate_chunk: Any | None = None,
        *,
        candidate_q: Any | None = None,
        require_live_prefix_safe: bool = False,
        live_prefix_len: int | None = None,
        allow_best_live_prefix_when_unsafe: bool = False,
    ) -> dict[str, Any]:
        """Return and cache the shared safe ACT-like rejoin window.

        The selected window is shared by deform and recovery. Deform can ask for
        a candidate-reachable target while optimizing; recovery can then reuse
        the cached target for the same live observation/nominal step.
        """
        recovery = self._recovery_owner()
        if (
            candidate_chunk is None
            and candidate_q is None
            and not require_live_prefix_safe
        ):
            cached = self._cached_nominal_rejoin_target(obs)
            if cached is not None:
                return cached

        if not getattr(recovery, "safechunk_recover_enabled", False) or not getattr(
            recovery,
            "use_latest_nominal_for_rejoin",
            False,
        ):
            return self._cache_nominal_rejoin_target(
                obs,
                self._nominal_rejoin_empty_info("no_latest_nominal"),
            )
        if getattr(recovery, "latest_nominal_chunk", None) is None:
            return self._cache_nominal_rejoin_target(
                obs,
                self._nominal_rejoin_empty_info("no_latest_nominal"),
            )

        target: np.ndarray = np.asarray(recovery.latest_nominal_chunk, dtype=np.float32).copy()
        blocked_nominal = getattr(recovery, "blocked_nominal_chunk", None)
        blocked_step = getattr(recovery, "blocked_nominal_step", None)
        latest_step = getattr(recovery, "latest_nominal_step", None)
        stale_blocked: bool = bool(
            getattr(recovery, "suppress_stale_nominal_rejoin", False)
            and blocked_nominal is not None
            and target.shape == np.asarray(blocked_nominal).shape
            and latest_step <= (blocked_step or -1)
            and np.allclose(target, blocked_nominal)
        )
        try:
            nominal_q_seq = self.rollout_nominal_chunk(obs, target)
            candidate_q_arr = None
            if candidate_q is not None:
                candidate_q_arr = np.asarray(candidate_q, dtype=np.float32).reshape(-1)
            elif candidate_chunk is not None:
                candidate_q_seq = self.rollout_nominal_chunk(obs, candidate_chunk)
                candidate_q_seq_arr = np.asarray(candidate_q_seq, dtype=np.float32)
                if candidate_q_seq_arr.ndim == 2 and candidate_q_seq_arr.shape[0] > 0:
                    candidate_q_arr = candidate_q_seq_arr[-1]
            valid = self.controlled_state_indices < np.asarray(nominal_q_seq).shape[1]
            state_idx = self.controlled_state_indices[valid]
            window_info = self._select_safe_nominal_rejoin_window(
                obs,
                nominal_q_seq,
                candidate_q=candidate_q_arr,
                state_idx=state_idx,
                nominal_action_seq=target,
                require_live_prefix_safe=require_live_prefix_safe,
                live_prefix_len=live_prefix_len,
                allow_best_live_prefix_when_unsafe=allow_best_live_prefix_when_unsafe,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Shared nominal safe-window target selection failed: %s", exc)
            return self._cache_nominal_rejoin_target(
                obs,
                self._nominal_rejoin_empty_info(
                    "acceptance_unavailable",
                    stale_blocked=stale_blocked,
                ),
            )

        windows = list(window_info.get("windows", []))
        full_count = int(window_info.get("full_window_count", 0) or 0)
        staging_count = int(window_info.get("staging_window_count", 0) or 0)
        if not bool(window_info.get("available", False)):
            h_seq = np.asarray(window_info.get("h_seq", []), dtype=np.float32).reshape(-1)
            clearance = float(np.max(h_seq)) if h_seq.size else float("-inf")
            reason = (
                "stale_blocked_nominal"
                if stale_blocked and not windows
                else str(window_info.get("reason", "no_safe_nominal_window"))
            )
            info = self._nominal_rejoin_empty_info(
                reason,
                clearance=clearance,
                stale_blocked=stale_blocked,
                window_count=len(windows),
                full_window_count=full_count,
                staging_window_count=staging_count,
            )
            info.update(
                {
                    "nominal_rejoin_live_prefix_required": bool(
                        window_info.get("live_prefix_required", require_live_prefix_safe)
                    ),
                    "nominal_rejoin_live_prefix_allow_fallback": bool(
                        window_info.get(
                            "live_prefix_allow_fallback",
                            allow_best_live_prefix_when_unsafe,
                        )
                    ),
                    "nominal_rejoin_live_prefix_fallback_selected": bool(
                        window_info.get("live_prefix_fallback_selected", False)
                    ),
                    "nominal_rejoin_target_source": window_info.get("target_source"),
                    "nominal_rejoin_live_prefix_safe_count": int(
                        window_info.get("live_prefix_safe_count", 0) or 0
                    ),
                    "nominal_rejoin_live_prefix_eval_count": int(
                        window_info.get("live_prefix_eval_count", 0) or 0
                    ),
                    "nominal_rejoin_live_prefix_best_min_clearance": float(
                        window_info.get("live_prefix_best_min_clearance", float("-inf"))
                    ),
                    "nominal_rejoin_live_prefix_best_start": window_info.get(
                        "live_prefix_best_start"
                    ),
                    "nominal_rejoin_live_prefix_required_clearance": float(
                        window_info.get(
                            "live_prefix_required_clearance",
                            self._acceptance_clearance_threshold(),
                        )
                    ),
                    "nominal_rejoin_live_prefix_skip_reason": window_info.get(
                        "live_prefix_skip_reason"
                    ),
                }
            )
            return self._cache_nominal_rejoin_target(obs, info)

        start = int(window_info["start"])
        end = int(window_info["end"])
        target_window = self._frame_stack_rejoin_target_window(target, start=start, end=end)
        target_q_window = self._frame_stack_rejoin_target_window(
            nominal_q_seq,
            start=start,
            end=end,
        )
        window_type = str(window_info["window_type"])
        safe_prefix_len = int(window_info["length"])
        clearance = float(window_info["min_clearance"])
        return self._cache_nominal_rejoin_target(
            obs,
            {
                "available": True,
                "target_chunk": target_window,
                "target_q_seq": target_q_window,
                "nominal_rejoin_target_window_len": int(target_window.shape[0]),
                "safe_prefix_len": safe_prefix_len,
                "suppressed_reason": None,
                "nominal_rejoin_clearance": clearance,
                "nominal_rejoin_required_clearance": float(self._acceptance_clearance_threshold()),
                "nominal_rejoin_stale_blocked": stale_blocked,
                "nominal_rejoin_window_count": len(windows),
                "nominal_rejoin_full_window_count": full_count,
                "nominal_rejoin_staging_window_count": staging_count,
                "nominal_rejoin_window_start": start,
                "nominal_rejoin_window_end": end,
                "nominal_rejoin_window_len": safe_prefix_len,
                "nominal_rejoin_window_type": window_type,
                "safe_rejoin_window_found": window_type == "full",
                "short_staging_window_found": window_type == "staging",
                "nominal_rejoin_live_prefix_required": bool(
                    window_info.get("live_prefix_required", require_live_prefix_safe)
                ),
                "nominal_rejoin_live_prefix_allow_fallback": bool(
                    window_info.get(
                        "live_prefix_allow_fallback",
                        allow_best_live_prefix_when_unsafe,
                    )
                ),
                "nominal_rejoin_live_prefix_fallback_selected": bool(
                    window_info.get("live_prefix_fallback_selected", False)
                ),
                "nominal_rejoin_target_source": window_info.get(
                    "target_source",
                    "nominal_safe_rejoin_window",
                ),
                "nominal_rejoin_live_prefix_safe": bool(
                    window_info.get("live_prefix_safe", not require_live_prefix_safe)
                ),
                "nominal_rejoin_live_prefix_min_clearance": window_info.get(
                    "live_prefix_min_clearance"
                ),
                "nominal_rejoin_live_prefix_required_clearance": float(
                    window_info.get(
                        "live_prefix_required_clearance",
                        self._acceptance_clearance_threshold(),
                    )
                ),
                "nominal_rejoin_live_prefix_horizon": window_info.get(
                    "live_prefix_horizon"
                ),
                "nominal_rejoin_live_prefix_safe_count": int(
                    window_info.get("live_prefix_safe_count", 0) or 0
                ),
                "nominal_rejoin_live_prefix_eval_count": int(
                    window_info.get("live_prefix_eval_count", 0) or 0
                ),
                "nominal_rejoin_live_prefix_best_min_clearance": float(
                    window_info.get("live_prefix_best_min_clearance", float("-inf"))
                ),
                "nominal_rejoin_live_prefix_best_start": window_info.get(
                    "live_prefix_best_start"
                ),
            },
        )

    def _deform_terminal_rejoin_terms(
        self,
        obs: Any,
        candidate: Any,
        q_seq: Any,
        action_idx: Any,
    ) -> dict[str, Any]:
        """Score whether a deformation endpoint is action/tangent-rejoinable.

        Position alone is not enough for ACT.  This term also compares the last
        deformation action and terminal state tangent with the selected ACT-like
        rejoin window, so deform prefers escape states that recovery can enter
        smoothly.
        """
        candidate_arr = np.asarray(candidate, dtype=np.float32)
        q_seq_arr = np.asarray(q_seq, dtype=np.float32)
        action_idx_arr = np.asarray(action_idx, dtype=np.int64).reshape(-1)
        target_info = InterventionExecutionFactory.get_nominal_rejoin_target(
            self,
            obs,
            candidate,
            candidate_q=q_seq_arr[-1] if q_seq_arr.ndim == 2 and q_seq_arr.shape[0] else None,
        )
        base_terms: dict[str, Any] = {
            "deform_rejoin_available": bool(target_info.get("available", False)),
            "deform_rejoin_window_loss": 0.0,
            "deform_rejoin_q_loss": 0.0,
            "deform_rejoin_qd_loss": 0.0,
            "deform_rejoin_action_loss": 0.0,
            "deform_rejoin_heading_loss": 0.0,
            "deform_rejoin_q_dist": None,
            "deform_rejoin_qd_dist": None,
            "deform_rejoin_action_dist": None,
            "deform_rejoin_heading_cosine": None,
            "deform_rejoin_best_window_offset": None,
            "deform_rejoin_velocity_weight": float(self.lambda_deform_rejoin_velocity),
            "deform_rejoin_action_weight": float(self.lambda_deform_rejoin_action),
            "deform_rejoin_heading_weight": float(self.lambda_deform_rejoin_heading),
            "nominal_rejoin_available": bool(target_info.get("available", False)),
            "nominal_rejoin_suppressed_reason": target_info.get("suppressed_reason"),
            "nominal_rejoin_clearance": float(target_info.get("nominal_rejoin_clearance", float("-inf"))),
            "nominal_rejoin_safe_prefix_len": int(target_info.get("safe_prefix_len", 0) or 0),
            "nominal_rejoin_window_start": target_info.get("nominal_rejoin_window_start"),
            "nominal_rejoin_window_end": target_info.get("nominal_rejoin_window_end"),
            "nominal_rejoin_window_len": target_info.get("nominal_rejoin_window_len"),
            "nominal_rejoin_window_type": target_info.get("nominal_rejoin_window_type"),
            "safe_rejoin_window_found": bool(target_info.get("safe_rejoin_window_found", False)),
            "short_staging_window_found": bool(target_info.get("short_staging_window_found", False)),
        }
        if (
            not bool(target_info.get("available", False))
            or q_seq_arr.ndim != 2
            or q_seq_arr.shape[0] == 0
        ):
            return base_terms

        target_q_raw = target_info.get("target_q_seq")
        target_q_seq = (
            np.asarray(target_q_raw, dtype=np.float32)
            if target_q_raw is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        if target_q_seq.ndim != 2 or target_q_seq.shape[0] == 0:
            target_chunk = target_info.get("target_chunk")
            if target_chunk is None:
                return base_terms
            target_q_seq = np.asarray(self.rollout_nominal_chunk(obs, target_chunk), dtype=np.float32)
        if target_q_seq.ndim != 2 or target_q_seq.shape[0] == 0:
            return base_terms

        valid_state = self.controlled_state_indices < min(q_seq_arr.shape[1], target_q_seq.shape[1])
        state_idx: np.ndarray = self.controlled_state_indices[valid_state]
        if state_idx.size == 0:
            return base_terms
        weights = self._q_rejoin_weight_vector(target_q_seq.shape[1], state_idx)
        final_q = q_seq_arr[-1, state_idx].reshape(1, -1)
        q_diff = (target_q_seq[:, state_idx] - final_q) * weights.reshape(1, -1)
        q_losses = np.square(q_diff).mean(axis=1)
        best = int(np.argmin(q_losses))
        q_loss = float(q_losses[best])
        q_dist = float(np.sqrt(max(q_loss, 0.0)))

        if q_seq_arr.shape[0] >= 2:
            deform_tangent = q_seq_arr[-1, state_idx] - q_seq_arr[-2, state_idx]
        else:
            deform_tangent = np.zeros(state_idx.shape[0], dtype=np.float32)
        if target_q_seq.shape[0] >= 2:
            next_idx = min(best + 1, target_q_seq.shape[0] - 1)
            prev_idx = max(best - 1, 0)
            if next_idx > best:
                target_tangent = target_q_seq[next_idx, state_idx] - target_q_seq[best, state_idx]
            else:
                target_tangent = target_q_seq[best, state_idx] - target_q_seq[prev_idx, state_idx]
        else:
            target_tangent = np.zeros(state_idx.shape[0], dtype=np.float32)
        weighted_deform_tangent = deform_tangent * weights
        weighted_target_tangent = target_tangent * weights
        qd_diff = weighted_deform_tangent - weighted_target_tangent
        qd_loss = float(np.square(qd_diff).mean())
        qd_dist = float(np.sqrt(max(qd_loss, 0.0)))
        deform_norm = float(np.linalg.norm(weighted_deform_tangent))
        target_norm = float(np.linalg.norm(weighted_target_tangent))
        if deform_norm > 1e-8 and target_norm > 1e-8:
            heading_cosine = float(
                np.clip(
                    np.dot(weighted_deform_tangent, weighted_target_tangent)
                    / (deform_norm * target_norm + 1e-8),
                    -1.0,
                    1.0,
                )
            )
            heading_loss = float(np.square(max(0.0, 1.0 - heading_cosine)))
        else:
            heading_cosine = None
            heading_loss = 0.0

        action_loss = 0.0
        action_dist = None
        target_chunk = target_info.get("target_chunk")
        if (
            target_chunk is not None
            and candidate_arr.ndim == 2
            and candidate_arr.shape[0] > 0
            and action_idx_arr.size > 0
        ):
            target_action_arr = np.asarray(target_chunk, dtype=np.float32)
            valid_action = action_idx_arr < min(candidate_arr.shape[1], target_action_arr.shape[1])
            local_action_idx = action_idx_arr[valid_action]
            if local_action_idx.size > 0 and target_action_arr.ndim == 2 and target_action_arr.shape[0] > 0:
                action_offset = min(best, target_action_arr.shape[0] - 1)
                action_diff = (
                    candidate_arr[-1, local_action_idx]
                    - target_action_arr[action_offset, local_action_idx]
                )
                action_loss = float(np.square(action_diff).mean())
                action_dist = float(np.sqrt(max(action_loss, 0.0)))

        total_rejoin_loss = float(
            q_loss
            + self.lambda_deform_rejoin_velocity * qd_loss
            + self.lambda_deform_rejoin_action * action_loss
            + self.lambda_deform_rejoin_heading * heading_loss
        )
        base_terms.update(
            {
                "deform_rejoin_window_loss": total_rejoin_loss,
                "deform_rejoin_q_loss": q_loss,
                "deform_rejoin_qd_loss": qd_loss,
                "deform_rejoin_action_loss": action_loss,
                "deform_rejoin_heading_loss": heading_loss,
                "deform_rejoin_q_dist": q_dist,
                "deform_rejoin_qd_dist": qd_dist,
                "deform_rejoin_action_dist": action_dist,
                "deform_rejoin_heading_cosine": heading_cosine,
                "deform_rejoin_best_window_offset": int(best),
            }
        )
        return base_terms

    def _deform_terminal_rejoin_terms_batch(
        self,
        obs: Any,
        candidates: Any,
        q_seq_batch: Any,
        action_idx: Any,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Vector wrapper for deformation endpoint-to-window costs."""
        candidates_arr = np.asarray(candidates, dtype=np.float32)
        q_batch = np.asarray(q_seq_batch, dtype=np.float32)
        terms: list[dict[str, Any]] = []
        losses: list[float] = []
        for i in range(candidates_arr.shape[0]):
            item = self._deform_terminal_rejoin_terms(
                obs,
                candidates_arr[i],
                q_batch[i],
                action_idx,
            )
            terms.append(item)
            losses.append(float(item.get("deform_rejoin_window_loss", 0.0)))
        return np.asarray(losses, dtype=np.float32), terms

    def _recover_nominal_rejoin_terms(self, obs, candidate, *, record=False):
        target_info = InterventionExecutionFactory.get_nominal_rejoin_target(
            self,
            obs,
            candidate,
        )
        rejoin_info = {
            "nominal_rejoin_score": 0.0,
            "recover_projection_on_nominal": 0.0,
            "recover_cosine_to_nominal": 0.0,
            "nominal_delta_norm": 0.0,
            "path_delta_norm": 0.0,
        }
        if target_info.get("available"):
            rejoin_info = self.compute_nominal_rejoin_score(
                candidate,
                target_info["target_chunk"],
                obs=obs,
            )
        progress_score, progress_available = self._candidate_progress_score(obs, candidate)
        effective_weight = self._recover_rejoin_weight_effective()
        if record:
            self._record_nominal_rejoin_target(
                target_info,
                rejoin_info if target_info.get("available") else None,
                progress_score=progress_score,
            )
        return target_info, rejoin_info, float(progress_score), bool(progress_available), float(effective_weight)

    def _recover_task_progress_cost_batch(
        self,
        obs,
        candidates,
        nominal,
        action_idx,
        reference_chunk=None,
    ):
        candidates = np.asarray(candidates, dtype=np.float32)
        q_seq_batch = self.rollout_nominal_chunk_batch(obs, candidates)
        safety_eval = self.evaluate_horizon_safety_batch(obs, q_seq_batch)
        h_seq = self._clearance_sequence_batch_from_eval(
            safety_eval,
            candidates.shape[0],
            candidates.shape[1],
        )
        required_clearance = float(self.recovery._acceptance_clearance_threshold())
        safety_loss = np.square(np.maximum(required_clearance - h_seq, 0.0)).sum(axis=1)
        task_progress_clearance_scale = float(
            getattr(
                self.recovery,
                "recover_task_progress_clearance_penalty_scale",
                getattr(self.recovery, "recover_clearance_penalty_scale", 5.0),
            )
        )
        if len(action_idx):
            action_deviation_loss = np.square(
                candidates[:, :, action_idx] - nominal[None, :, action_idx]
            ).mean(axis=(1, 2))
        else:
            action_deviation_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        smoothness_loss = self._smoothness_loss_batch(candidates, action_idx)
        target_info = InterventionExecutionFactory.get_nominal_rejoin_target(self, obs)
        reference_available = reference_chunk is not None
        target_q_seq = None
        if reference_available:
            target_q_seq = self.rollout_nominal_chunk(obs, reference_chunk)
        elif target_info.get("available"):
            target_q_seq = target_info.get("target_q_seq")
            if target_q_seq is not None:
                target_q_seq = np.asarray(target_q_seq, dtype=np.float32)
            if target_q_seq is None or target_q_seq.ndim != 2 or target_q_seq.shape[0] == 0:
                target_q_seq = self.rollout_nominal_chunk(obs, target_info["target_chunk"])
        effective_weight = self._recover_rejoin_weight_effective()
        progress_scores = []
        progress_available = []
        rejoin_infos = []
        ordered_terms_list = []
        act_direction_terms_list = []
        resume_window_terms_list = []
        target_chunk_for_window = reference_chunk if reference_available else (
            target_info.get("target_chunk") if target_info.get("available") else None
        )
        for i, candidate in enumerate(candidates):
            progress_score, progress_ok = self._candidate_progress_score(obs, candidate)
            if not target_info.get("available"):
                progress_score = 0.0
                progress_ok = False
            progress_scores.append(float(progress_score))
            progress_available.append(bool(progress_ok))
            if reference_available:
                rejoin_infos.append(
                    self.compute_nominal_rejoin_score(
                        candidate,
                        reference_chunk,
                        obs=obs,
                    )
                )
            elif target_info.get("available"):
                rejoin_infos.append(
                    self.compute_nominal_rejoin_score(
                        candidate,
                        target_info["target_chunk"],
                        obs=obs,
                    )
                )
            else:
                rejoin_infos.append(
                    {
                        "nominal_rejoin_score": 0.0,
                        "recover_projection_on_nominal": 0.0,
                        "recover_cosine_to_nominal": 0.0,
                        "nominal_delta_norm": 0.0,
                        "path_delta_norm": 0.0,
                    }
                )
            ordered_terms_list.append(
                self._ordered_recovery_path_terms(
                    q_seq_batch[i],
                    target_q_seq,
                    target_index=0,
                )
            )
            act_direction_terms_list.append(
                self.recovery._recover_act_direction_terms(q_seq_batch[i], target_q_seq)
            )
            resume_window_terms_list.append(
                self.recovery._recover_resume_window_terms(
                    q_seq_batch[i],
                    target_q_seq,
                    candidate=candidate,
                    target_chunk=target_chunk_for_window,
                )
            )
        progress_scores = np.asarray(progress_scores, dtype=np.float32)
        nominal_rejoin_scores = np.asarray(
            [float(info.get("nominal_rejoin_score", 0.0)) for info in rejoin_infos],
            dtype=np.float32,
        )
        direction_terms_list = [
            self._recover_direction_alignment_terms(info) for info in rejoin_infos
        ]
        direction_loss = np.asarray(
            [float(info["recover_direction_loss"]) for info in direction_terms_list],
            dtype=np.float32,
        )
        ordered_loss = np.asarray(
            [float(info["recover_ordered_loss"]) for info in ordered_terms_list],
            dtype=np.float32,
        )
        act_progress_loss = np.asarray(
            [float(info["recover_act_progress_loss"]) for info in act_direction_terms_list],
            dtype=np.float32,
        )
        act_heading_loss = np.asarray(
            [float(info["recover_act_heading_loss"]) for info in act_direction_terms_list],
            dtype=np.float32,
        )
        act_direction_loss = (
            self.recovery.recover_act_progress_weight * act_progress_loss
            + self.recovery.recover_act_heading_weight * act_heading_loss
        ).astype(np.float32)
        resume_window_loss = np.asarray(
            [
                float(info.get("recover_resume_window_total_loss", 0.0))
                for info in resume_window_terms_list
            ],
            dtype=np.float32,
        )
        stalled_penalty = np.where(
            (progress_scores <= 0.0) & (nominal_rejoin_scores <= 0.0),
            5.0,
            0.0,
        ).astype(np.float32)
        existing_loss = (
            self.recovery.recover_safety_weight
            * task_progress_clearance_scale
            * safety_loss
            + self.recovery.recover_action_deviation_weight * action_deviation_loss
            + self.recovery.recover_smoothness_weight * smoothness_loss
            + self.recovery.recover_direction_alignment_weight * direction_loss
            + ordered_loss
            + act_direction_loss
            + self.recovery.recover_resume_window_weight * resume_window_loss
        )
        recover_score_total = (
            self.recovery.recover_task_progress_weight * progress_scores
            + effective_weight * nominal_rejoin_scores
            - stalled_penalty
        )
        total_loss = existing_loss - recover_score_total
        losses = []
        for i, rejoin_info in enumerate(rejoin_infos):
            losses.append(
                {
                    "safety_loss": float(safety_loss[i]),
                    "recover_required_min_clearance": float(required_clearance),
                    "recover_clearance_margin_loss": float(safety_loss[i]),
                    "recover_clearance_penalty_scale": float(
                        getattr(self.recovery, "recover_clearance_penalty_scale", 5.0)
                    ),
                    "recover_task_progress_clearance_penalty_scale": float(
                        task_progress_clearance_scale
                    ),
                    "action_deviation_loss": float(action_deviation_loss[i]),
                    "smoothness_loss": float(smoothness_loss[i]),
                    "existing_optimization_loss": float(existing_loss[i]),
                    "total_loss": float(total_loss[i]),
                    "min_clearance": float(np.min(h_seq[i])),
                    "recover_task_progress_score": float(progress_scores[i]),
                    "progress_score_available": bool(progress_available[i]),
                    "recover_score_total": float(recover_score_total[i]),
                    "recover_rejoin_weight_effective": float(effective_weight),
                    "recover_direction_alignment_weight": float(self.recovery.recover_direction_alignment_weight),
                    "recover_act_progress_weight": float(self.recovery.recover_act_progress_weight),
                    "recover_act_heading_weight": float(self.recovery.recover_act_heading_weight),
                    "recover_min_act_heading_cosine": float(self.recovery.recover_min_act_heading_cosine),
                    "recover_act_direction_loss": float(act_direction_loss[i]),
                    **direction_terms_list[i],
                    **ordered_terms_list[i],
                    **act_direction_terms_list[i],
                    **resume_window_terms_list[i],
                    "recover_step_since_deform": int(self.recovery.recover_step_since_deform),
                    "nominal_rejoin_available": bool(target_info.get("available")),
                    "nominal_rejoin_suppressed_reason": target_info.get("suppressed_reason"),
                    "nominal_rejoin_clearance": float(target_info.get("nominal_rejoin_clearance", float("-inf"))),
                    "nominal_rejoin_safe_prefix_len": int(target_info.get("safe_prefix_len", 0) or 0),
                    "nominal_rejoin_window_start": target_info.get("nominal_rejoin_window_start"),
                    "nominal_rejoin_window_end": target_info.get("nominal_rejoin_window_end"),
                    "nominal_rejoin_window_len": target_info.get("nominal_rejoin_window_len"),
                    "nominal_rejoin_window_type": target_info.get("nominal_rejoin_window_type"),
                    "safe_rejoin_window_found": bool(target_info.get("safe_rejoin_window_found", False)),
                    "short_staging_window_found": bool(target_info.get("short_staging_window_found", False)),
                    "batched_optimizer": True,
                    **rejoin_info,
                }
            )
        return total_loss.astype(np.float32), losses

    def _return_deformation_cost_batch(
        self,
        obs,
        candidates,
        nominal,
        nominal_q_seq,
        rejoin_context,
        action_idx,
    ):
        self._sync_intervention_weights()
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
        safety_loss = np.square(np.maximum(self.min_clearance - h_seq, 0.0)).sum(
            axis=1
        )
        rejoin_loss, j_best, q_time_ms = self._q_rejoin_loss_batch(
            q_seq_batch,
            nominal_q_seq=nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        if len(action_idx):
            action_deviation_loss = np.square(
                candidates[:, :, action_idx] - nominal[None, :, action_idx]
            ).mean(axis=(1, 2))
        else:
            action_deviation_loss = np.zeros(candidates.shape[0], dtype=np.float32)
        smoothness_loss = self._smoothness_loss_batch(candidates, action_idx)
        ordered_terms_list = []
        for i in range(candidates.shape[0]):
            ordered_target_index = self._ordered_recovery_start_index(
                j_best[i],
                q_seq_batch.shape[1],
                nominal_q_seq,
            )
            ordered_terms_list.append(
                self._ordered_recovery_path_terms(
                    q_seq_batch[i],
                    nominal_q_seq,
                    target_index=ordered_target_index,
                    rejoin_context=rejoin_context,
                )
            )
        ordered_loss = np.asarray(
            [float(item["recover_ordered_loss"]) for item in ordered_terms_list],
            dtype=np.float32,
        )
        required_clearance = float(self.recovery._acceptance_clearance_threshold())
        safety_loss = np.square(np.maximum(required_clearance - h_seq, 0.0)).sum(axis=1)
        total_loss = (
            self.lambda_return_rejoin * rejoin_loss
            + self.lambda_return_safety
            * float(getattr(self.recovery, "recover_clearance_penalty_scale", 5.0))
            * safety_loss
            + self.lambda_return_smooth * smoothness_loss
            + self.lambda_return_action * action_deviation_loss
            + ordered_loss
        )
        per_q_ms = float(q_time_ms) / max(1, candidates.shape[0])
        per_rollout_ms = float(rollout_time_ms) / max(1, candidates.shape[0])
        losses = [
            {
                "safety_loss": float(safety_loss[i]),
                "recover_required_min_clearance": float(required_clearance),
                "recover_clearance_margin_loss": float(safety_loss[i]),
                "rejoin_loss": float(rejoin_loss[i]),
                "return_rejoin_loss": float(rejoin_loss[i]),
                "action_deviation_loss": float(action_deviation_loss[i]),
                "smoothness_loss": float(smoothness_loss[i]),
                **ordered_terms_list[i],
                "total_loss": float(total_loss[i]),
                "min_clearance": float(np.min(h_seq[i])),
                "j_best": j_best[i],
                "return_target_index": j_best[i],
                "rejoin_q_eval_time_ms": per_q_ms,
                "batched_optimizer": True,
                "jax_rollout_time_ms": per_rollout_ms,
            }
            for i in range(candidates.shape[0])
        ]
        return total_loss.astype(np.float32), losses

    def _recovery_terminal_rejoin_info(
        self,
        obs,
        candidate_chunk,
        context,
        rejoin_context,
        default_target_index=None,
    ):
        q_seq = self.rollout_nominal_chunk(obs, candidate_chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        min_clearance = float(safety_eval.get("min_clearance", float("-inf")))
        q_loss, target_index, q_time_ms = self._q_rejoin_loss(
            q_seq,
            nominal_q_seq=context.nominal_q_seq,
            rejoin_context=rejoin_context,
        )
        if target_index is None:
            target_index = default_target_index
        qd_loss, qd_index, qd_time_ms = self._qd_rejoin_loss(
            q_seq,
            nominal_q_seq=context.nominal_q_seq,
            target_index=target_index,
            rejoin_context=rejoin_context,
        )
        ordered_target_index = self._ordered_recovery_start_index(
            target_index,
            q_seq.shape[0],
            context.nominal_q_seq,
        )
        ordered_terms = self._ordered_recovery_path_terms(
            q_seq,
            context.nominal_q_seq,
            target_index=ordered_target_index,
            rejoin_context=rejoin_context,
        )
        q_dist = self._sqrt_loss(q_loss)
        qd_dist = self._sqrt_loss(qd_loss)
        q_ok = bool(target_index is not None and q_dist < self.recovery.q_rejoin_threshold)
        qd_ok, qd_acceptance = self._qd_rejoin_acceptance(qd_index, qd_dist)
        return {
            "q_seq": q_seq,
            "eval": safety_eval,
            "min_clearance": min_clearance,
            "q_rejoin_loss": float(q_loss),
            "q_rejoin_dist": float(q_dist),
            "q_rejoin_ok": bool(q_ok),
            "target_index": target_index,
            "q_eval_time_ms": float(q_time_ms),
            "qd_rejoin_loss": float(qd_loss),
            "qd_rejoin_dist": float(qd_dist),
            "qd_rejoin_ok": bool(qd_ok),
            "qd_rejoin_index": qd_index,
            "qd_eval_time_ms": float(qd_time_ms),
            **qd_acceptance,
            **ordered_terms,
        }

    def as_chunk(self, action: Any) -> tuple[np.ndarray, bool]:
        """Public wrapper for action-chunk normalization used by filter routing."""
        return self._as_chunk(action)

    @staticmethod
    def maybe_array(value: Any) -> np.ndarray | None:
        """Best-effort conversion used by public API argument disambiguation."""
        try:
            return np.asarray(value)
        except Exception:  # noqa: BLE001
            return None

    def _as_chunk(self, action) -> tuple[np.ndarray, bool]:
        arr = np.asarray(action)
        if arr.ndim == 1:
            if arr.shape[0] != self.action_dim:
                logger.debug(
                    "Single action dim %d differs from configured action_dim %d",
                    arr.shape[0],
                    self.action_dim,
                )
            return arr.reshape(1, -1).copy(), True
        if arr.ndim == 2:
            return arr.copy(), False
        raise ValueError(
            f"Expected action shape ({self.action_dim},) or (H, {self.action_dim}), "
            f"got {arr.shape}"
        )

    def _jsonable_snapshot(self, value: Any) -> Any:
        """Convert numpy values and nested containers to JSON-friendly snapshots."""
        if value is None:
            return None
        if isinstance(value, dict):
            return {str(k): self._jsonable_snapshot(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable_snapshot(v) for v in value]
        try:
            arr = np.asarray(value)
            if arr.ndim > 0:
                return arr.astype(float).tolist()
            if np.issubdtype(arr.dtype, np.number):
                return float(arr)
        except Exception:  # noqa: BLE001
            pass
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return str(value)

    def _snapshot_human_state(self, human_state: Any) -> Any:
        """Normalize human state payload for JSON snapshots and motion checks."""
        return self._jsonable_snapshot(human_state)

    def _human_motion_since_plan(self, human_state: Any) -> float | None:
        """Compute L2 human-state motion since the last accepted recovery plan."""
        recovery = self.recovery if self.recovery is not None else self
        before = getattr(recovery, "committed_accepted_human_state_snapshot", None)
        after = self._snapshot_human_state(human_state)
        if before is None or after is None:
            return None
        try:
            a = np.asarray(after, dtype=np.float32).reshape(-1)
            b = np.asarray(before, dtype=np.float32).reshape(-1)
            if a.shape != b.shape:
                return None
            return float(np.linalg.norm(a - b))
        except Exception:  # noqa: BLE001
            return None

    def _rollout_chunk_from_q(self, q: Any, action_chunk: Any) -> np.ndarray:
        """Roll a full action chunk forward from a starting state."""
        chunk, _ = self._as_chunk(action_chunk)
        q = np.asarray(q, dtype=np.float32).reshape(-1).copy()
        q_seq = np.zeros((chunk.shape[0], q.shape[0]), dtype=np.float32)
        valid = (
            (self.controlled_state_indices < q.shape[0])
            & (self.controlled_action_indices < chunk.shape[1])
        )
        state_idx = self.controlled_state_indices[valid]
        action_idx = self.controlled_action_indices[valid]
        for k, action in enumerate(chunk):
            q_next = self._apply_controlled_action_step(q, action, state_idx, action_idx)
            q_seq[k] = q_next
            q = q_next
        return q_seq

    def _cached_nominal_rollout(self, obs: Any, chunk: np.ndarray) -> np.ndarray | None:
        """Return the current filter-step nominal rollout when this is the same chunk."""
        parent = getattr(self, "parent", None)
        context = getattr(parent, "_rollout_context", None)
        if not isinstance(context, dict) or not context:
            return None
        if context.get("obs_id") != id(obs):
            return None
        key_fn = getattr(parent, "_rollout_context_key", None)
        if not callable(key_fn):
            return None
        key = key_fn(chunk)
        if key is None or key != context.get("nominal_key"):
            return None
        nominal = context.get("nominal")
        if not isinstance(nominal, dict):
            return None
        q_seq = nominal.get("raw_q_seq")
        if q_seq is None:
            return None
        return np.asarray(q_seq, dtype=np.float32).copy()

    def _rollout_one_step_from_q(self, q: Any, action: Any) -> np.ndarray:
        """Roll exactly one action step from the given state."""
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        return self._rollout_chunk_from_q(q, action)[0]

    def rollout_nominal_chunk(self, obs, action_chunk) -> np.ndarray:
        """Roll out a single action chunk using shared control-mode semantics."""
        chunk, _ = self._as_chunk(action_chunk)
        cached_q_seq = self._cached_nominal_rollout(obs, chunk)
        if cached_q_seq is not None:
            return cached_q_seq
        q = self.extract_current_q(obs, chunk)
        q_seq = np.zeros((chunk.shape[0], q.shape[0]), dtype=np.float32)

        valid = (
            (self.controlled_state_indices < q.shape[0])
            & (self.controlled_action_indices < chunk.shape[1])
        )
        state_idx = self.controlled_state_indices[valid]
        action_idx = self.controlled_action_indices[valid]

        for k, action in enumerate(chunk):
            q_next = self._apply_controlled_action_step(q, action, state_idx, action_idx)
            q_seq[k] = q_next
            q = q_next
        return q_seq

    def rollout_nominal_chunk_batch(self, obs, action_chunks) -> np.ndarray:
        """Roll out a batch of action chunks and preserve the default numpy path."""
        rollout_impl = type(self).__dict__.get("rollout_nominal_chunk_batch")
        if rollout_impl is not None and rollout_impl is not InterventionExecutionFactory.rollout_nominal_chunk_batch:
            return rollout_impl(self, obs, action_chunks)

        # Batch rollout is intentionally owned here so Deform/Recovery/MPC share
        # the same calibrated action-to-q transition semantics.

        chunks = np.asarray(action_chunks, dtype=np.float32)
        if chunks.ndim == 2:
            chunks = chunks[None, :, :]
        if chunks.ndim != 3:
            raise ValueError(
                "Expected action_chunks with shape (B, H, A), "
                f"got {chunks.shape}"
            )
        q0 = self.extract_current_q(obs, chunks[0] if chunks.shape[0] else None)
        q = np.broadcast_to(q0[None, :], (chunks.shape[0], q0.shape[0])).copy()
        q_seq = np.zeros((chunks.shape[0], chunks.shape[1], q0.shape[0]), dtype=np.float32)

        valid = (
            (self.controlled_state_indices < q0.shape[0])
            & (self.controlled_action_indices < chunks.shape[2])
        )
        state_idx = self.controlled_state_indices[valid]
        action_idx = self.controlled_action_indices[valid]
        for k in range(chunks.shape[1]):
            q = self._apply_controlled_action_step_batch(
                q,
                chunks[:, k, :],
                state_idx,
                action_idx,
            )
            q_seq[:, k, :] = q
        return q_seq

    def extract_current_q(
        self,
        obs: Any,
        action_chunk: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract the current robot configuration from an observation payload."""
        candidates = ("q", "qpos", "robot_state", "state")
        value = None
        if isinstance(obs, dict):
            for name in candidates:
                if name in obs:
                    value = obs[name]
                    break
        else:
            for name in candidates:
                if hasattr(obs, name):
                    value = getattr(obs, name)
                    break

        if value is not None:
            q = np.asarray(value, dtype=np.float32).reshape(-1)
            if q.shape[0] >= self.expected_motion_dim:
                return q[: self.expected_motion_dim].copy()
            padded = np.zeros(self.expected_motion_dim, dtype=np.float32)
            padded[: q.shape[0]] = q
            return padded

        q = np.zeros(self.expected_motion_dim, dtype=np.float32)
        if action_chunk is not None and action_chunk.size > 0:
            valid = self.controlled_state_indices < q.shape[0]
            state_idx = self.controlled_state_indices[valid]
            action_idx = self.controlled_action_indices[valid]
            modes = self._control_mode_ids_for_state_indices(state_idx)
            absolute = modes == 0
            if np.any(absolute):
                q[state_idx[absolute]] = action_chunk[0][action_idx[absolute]]
        return q

    def evaluate_horizon_safety_batch(self, obs: Any, q_seq_batch: Any) -> dict[str, Any]:
        """Evaluate clearance metrics for a batch of planned state trajectories."""
        q_seq_batch = np.asarray(q_seq_batch, dtype=np.float32)
        if q_seq_batch.ndim == 2:
            q_seq_batch = q_seq_batch[None, :, :]
        if q_seq_batch.ndim != 3:
            raise ValueError(
                "Expected q_seq_batch with shape (B, H, Q), "
                f"got {q_seq_batch.shape}"
            )
        op = self._get_oscbf_operator()
        for method_name in (
            "evaluate_safety_batch",
            "compute_min_clearance_batch",
            "get_min_clearance_batch",
        ):
            method = getattr(op, method_name, None)
            if method is None:
                continue
            try:
                result = self._call_safety_batch_method(method, obs, q_seq_batch)
                return self._normalize_safety_batch_result(
                    result,
                    q_seq_batch.shape[0],
                    q_seq_batch.shape[1],
                )
            except Exception as exc:  # pragma: no cover - defensive integration path
                logger.debug(
                    "SafeChunk-Deform batched safety evaluation via %s failed: %s",
                    method_name,
                    exc,
                )

        per_candidate = [self.evaluate_horizon_safety(obs, q_seq) for q_seq in q_seq_batch]
        min_clearances = np.stack(
            [
                self._clearance_sequence_from_eval(item, q_seq_batch.shape[1])
                for item in per_candidate
            ],
            axis=0,
        ).astype(np.float32)
        unsafe = min_clearances < self.min_clearance
        unsafe_any = np.any(unsafe, axis=1)
        first_violation = np.full(q_seq_batch.shape[0], -1, dtype=np.int32)
        if np.any(unsafe_any):
            first_violation[unsafe_any] = np.argmax(unsafe[unsafe_any], axis=1)
        return {
            "horizon_safe": ~unsafe_any,
            "min_clearance": np.min(min_clearances, axis=1).astype(np.float32),
            "min_clearances": min_clearances,
            "first_violation": first_violation,
            "unsafe_count": np.count_nonzero(unsafe, axis=1).astype(np.int32),
            "safety_eval_available": all(
                bool(item.get("safety_eval_available", True)) for item in per_candidate
            ),
        }

    def _call_safety_batch_method(self, method: Any, obs: Any, q_seq_batch: np.ndarray):
        """Call a batch safety method across supported operator signatures."""
        attempts = (
            lambda: method(obs=obs, q_seq_batch=q_seq_batch),
            lambda: method(q_seq_batch=q_seq_batch, obs=obs),
            lambda: method(obs, q_seq_batch),
            lambda: method(q_seq_batch),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Batched safety method could not be called")

    def _normalize_safety_batch_result(
        self,
        result: Any,
        batch: int,
        horizon: int,
    ) -> dict[str, Any]:
        """Normalize a batched safety result into standard clearance fields."""
        if not isinstance(result, dict):
            raise ValueError("Batched safety result must be a dict")
        min_clearances = np.asarray(
            result.get("min_clearances", result.get("clearances", [])),
            dtype=np.float32,
        )
        if min_clearances.shape != (batch, horizon):
            min_by_candidate = np.asarray(
                result.get("min_clearance", np.inf),
                dtype=np.float32,
            ).reshape(-1)
            if min_by_candidate.size == 1:
                min_by_candidate = np.full(
                    batch,
                    float(min_by_candidate[0]),
                    dtype=np.float32,
                )
            if min_by_candidate.size != batch:
                min_by_candidate = np.full(batch, np.inf, dtype=np.float32)
            min_clearances = np.repeat(min_by_candidate[:, None], horizon, axis=1)
        unsafe = min_clearances < self.min_clearance
        unsafe_any = np.any(unsafe, axis=1)
        horizon_safe = np.asarray(
            result.get("horizon_safe", result.get("safe", ~unsafe_any)),
            dtype=np.bool_,
        ).reshape(-1)
        if horizon_safe.size == 1:
            horizon_safe = np.full(batch, bool(horizon_safe[0]), dtype=np.bool_)
        if horizon_safe.size != batch:
            horizon_safe = ~unsafe_any
        first_violation = result.get("first_violation")
        if first_violation is None:
            first_violation = np.full(batch, -1, dtype=np.int32)
            if np.any(unsafe_any):
                first_violation[unsafe_any] = np.argmax(unsafe[unsafe_any], axis=1)
        else:
            first_violation = np.asarray(first_violation, dtype=np.int32).reshape(-1)
            if first_violation.size == 1:
                first_violation = np.full(batch, int(first_violation[0]), dtype=np.int32)
            if first_violation.size != batch:
                first_violation = np.full(batch, -1, dtype=np.int32)
        info = dict(result)
        info.update(
            {
                "horizon_safe": horizon_safe,
                "min_clearance": np.min(min_clearances, axis=1).astype(np.float32),
                "min_clearances": min_clearances.astype(np.float32),
                "first_violation": first_violation,
                "unsafe_count": np.count_nonzero(unsafe, axis=1).astype(np.int32),
                "safety_eval_available": bool(result.get("safety_eval_available", True)),
            }
        )
        return info

    def evaluate_horizon_safety(
        self,
        obs: Any,
        q_seq: Any,
        *,
        predict_human_motion: bool | None = None,
    ) -> dict[str, Any]:
        """Evaluate safety and clearance metrics for one planned trajectory."""
        q_seq = np.asarray(q_seq, dtype=np.float32)
        op = self._get_oscbf_operator()
        prediction_attr_was_set = False
        previous_prediction_value = None
        if predict_human_motion is not None and op is not None and hasattr(
            op,
            "predict_human_motion",
        ):
            previous_prediction_value = getattr(op, "predict_human_motion")
            setattr(op, "predict_human_motion", bool(predict_human_motion))
            prediction_attr_was_set = True
        try:
            for method_name in (
                "compute_min_clearance",
                "get_min_clearance",
                "evaluate_safety",
                "is_safe",
            ):
                method = getattr(op, method_name, None)
                if method is None:
                    continue
                try:
                    result = self._call_safety_method(method, obs, q_seq)
                    info = self._normalize_safety_result(result, q_seq.shape[0])
                    if predict_human_motion is not None:
                        info["human_motion_prediction_override"] = bool(
                            predict_human_motion
                        )
                        info["human_motion_prediction_override_applied"] = bool(
                            prediction_attr_was_set
                        )
                    return info
                except Exception as exc:  # pragma: no cover - defensive integration path
                    logger.warning(
                        "SafeChunk-Deform safety evaluation via %s failed: %s",
                        method_name,
                        exc,
                    )

            if not self._warned_no_safety_eval:
                logger.warning(
                    "SafeChunk-Deform could not find a horizon clearance evaluator; "
                    "using conservative pass-through horizon evaluation."
                )
                self._warned_no_safety_eval = True
            h = q_seq.shape[0]
            info = {
                "horizon_safe": True,
                "min_clearance": float("inf"),
                "min_clearances": np.full(h, np.inf, dtype=np.float32),
                "first_violation": None,
                "unsafe_count": 0,
                "safety_eval_available": False,
            }
            if predict_human_motion is not None:
                info["human_motion_prediction_override"] = bool(predict_human_motion)
                info["human_motion_prediction_override_applied"] = bool(
                    prediction_attr_was_set
                )
            return info
        finally:
            if prediction_attr_was_set:
                setattr(op, "predict_human_motion", previous_prediction_value)

    def _call_safety_method(self, method: Any, obs: Any, q_seq: np.ndarray):
        """Call a single safety method across supported operator signatures."""
        attempts = (
            lambda: method(obs=obs, q_seq=q_seq),
            lambda: method(q_seq=q_seq, obs=obs),
            lambda: method(obs, q_seq),
            lambda: method(q_seq),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        raise last_error

    def _normalize_safety_result(self, result: Any, horizon: int) -> dict[str, Any]:
        """Normalize a safety operator result into standard clearance fields."""
        if isinstance(result, dict):
            min_clearances = np.asarray(
                result.get("min_clearances", result.get("clearances", [])),
                dtype=np.float32,
            ).reshape(-1)
            if min_clearances.size == 0:
                min_clearance = float(result.get("min_clearance", np.inf))
                min_clearances = np.full(horizon, min_clearance, dtype=np.float32)
            horizon_safe = bool(
                result.get(
                    "horizon_safe",
                    result.get("safe", np.all(min_clearances >= self.min_clearance)),
                )
            )
            first_violation = result.get("first_violation")
            if first_violation is None:
                unsafe = np.flatnonzero(min_clearances < self.min_clearance)
                first_violation = int(unsafe[0]) if unsafe.size else None
            unsafe_count = int(
                result.get(
                    "unsafe_count",
                    np.count_nonzero(min_clearances < self.min_clearance),
                )
            )
            info = {
                "horizon_safe": horizon_safe,
                "min_clearance": float(result.get("min_clearance", np.min(min_clearances))),
                "min_clearances": min_clearances,
                "first_violation": first_violation,
                "unsafe_count": unsafe_count,
                "safety_eval_available": bool(result.get("safety_eval_available", True)),
            }
            for key, value in result.items():
                if key not in info and key != "clearances":
                    info[key] = value
            return info

        arr = np.asarray(result)
        if arr.dtype == np.bool_:
            horizon_safe = bool(arr.all())
            min_clearances = np.full(horizon, np.inf if horizon_safe else -np.inf)
        else:
            min_clearances = arr.astype(np.float32).reshape(-1)
            if min_clearances.size == 1 and horizon > 1:
                min_clearances = np.full(horizon, float(min_clearances[0]), dtype=np.float32)
            horizon_safe = bool(np.all(min_clearances >= self.min_clearance))
        unsafe = np.flatnonzero(min_clearances < self.min_clearance)
        return {
            "horizon_safe": horizon_safe,
            "min_clearance": float(np.min(min_clearances)),
            "min_clearances": min_clearances,
            "first_violation": int(unsafe[0]) if unsafe.size else None,
            "unsafe_count": int(unsafe.size),
            "safety_eval_available": True,
        }

    def evaluate_candidate_acceptance(
        self,
        obs: Any,
        candidate_chunk: Any,
        candidate_type: str,
    ) -> dict[str, Any]:
        """Evaluate whether a deform/recover candidate can be executed safely."""
        chunk, _ = self._as_chunk(candidate_chunk)
        q_seq = self.rollout_nominal_chunk(obs, chunk)
        safety_eval = self.evaluate_horizon_safety(obs, q_seq)
        h_seq = self._clearance_sequence_from_eval(safety_eval, q_seq.shape[0])
        h_seq = np.asarray(h_seq, dtype=np.float32).reshape(-1)
        if h_seq.size == 0:
            h_seq = np.asarray([float("-inf")], dtype=np.float32)

        immediate_clearance = float(h_seq[0])
        horizon_min_clearance = float(np.min(h_seq))
        hard = float(self.deform.acceptance_hard_min_clearance)
        desired = float(self.deform.acceptance_desired_min_clearance)
        prefix_threshold = float(self.deform.prefix_min_clearance)
        full_required = bool(
            (candidate_type == "recover" and self.deform.full_horizon_required_for_recover)
            or (candidate_type == "deform" and self.deform.full_horizon_required_for_deform)
        )

        accepted = False
        acceptance_type = "rejected"
        rejection_reason = None
        safe_prefix_len = 0
        for value in h_seq:
            if float(value) >= prefix_threshold:
                safe_prefix_len += 1
            else:
                break

        if immediate_clearance < hard:
            acceptance_type = "emergency_brake"
            rejection_reason = "immediate_below_hard_margin"
        elif horizon_min_clearance >= desired:
            accepted = True
            acceptance_type = "full_horizon"
            safe_prefix_len = int(h_seq.size)
        elif full_required:
            rejection_reason = "horizon_below_desired_margin"
        elif self.deform.allow_safe_prefix_execution:
            if safe_prefix_len >= self.deform.min_safe_prefix_len:
                accepted = True
                acceptance_type = "safe_prefix"
            elif immediate_clearance >= prefix_threshold:
                accepted = True
                acceptance_type = "first_action_only"
                safe_prefix_len = 1
            else:
                rejection_reason = "no_safe_prefix"
        else:
            rejection_reason = "horizon_below_desired_margin"

        return {
            "accepted": bool(accepted),
            "acceptance_type": acceptance_type,
            "safe_prefix_len": int(safe_prefix_len),
            "immediate_clearance": immediate_clearance,
            "prefix_min_clearance": prefix_threshold,
            "horizon_min_clearance": horizon_min_clearance,
            "desired_min_clearance": desired,
            "hard_min_clearance": hard,
            "rejection_reason": rejection_reason,
            "candidate_type": candidate_type,
            "full_horizon_required": full_required,
            "rolling_replan_on_prefix": bool(self.deform.rolling_replan_on_prefix),
            "safe_prefix_execution": bool(accepted and acceptance_type != "full_horizon"),
            "horizon_safe": bool(
                safety_eval.get("horizon_safe", horizon_min_clearance >= desired)
            ),
        }

    def evaluate_hold_or_brake_acceptance(
        self,
        obs: Any,
        braked_chunk: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate hold/brake acceptance through the brake executor."""
        return self.parent.brake.evaluate_hold_or_brake_acceptance(
            obs,
            braked_chunk,
            **kwargs,
        )

    def deform_chunk_candidate(self, *args: Any, **kwargs: Any):
        """Run candidate deformation through the deformation executor."""
        return self.parent.deform.deform_chunk_candidate(*args, **kwargs)

    def emergency_deform_away(self, *args: Any, **kwargs: Any):
        """Run emergency deformation through the brake executor."""
        return self.parent.brake.emergency_deform_away(*args, **kwargs)

    def _ee_pose_sequence(self, *args: Any, **kwargs: Any):
        """Compute end-effector pose sequence through the deform executor."""
        return self.parent.deform._ee_pose_sequence(*args, **kwargs)

    def compute_nominal_rejoin_score(self, candidate_chunk, nominal_chunk, obs=None):
        """Compute simple alignment metrics between a candidate and nominal trajectories."""
        candidate, _ = self._as_chunk(candidate_chunk)
        nominal, _ = self._as_chunk(nominal_chunk)
        valid = self._valid_control_indices(candidate)
        if not np.any(valid) or candidate.shape[0] == 0 or nominal.shape[0] == 0:
            return {
                "nominal_rejoin_score": 0.0,
                "recover_projection_on_nominal": 0.0,
                "recover_cosine_to_nominal": 0.0,
                "nominal_delta_norm": 0.0,
                "path_delta_norm": 0.0,
            }
        action_idx = self.controlled_action_indices[valid]
        delta_cand = candidate[0, action_idx].astype(np.float64, copy=False)
        delta_nom = nominal[0, action_idx].astype(np.float64, copy=False)
        if obs is not None:
            try:
                q = self.extract_current_q(obs, candidate)
                state_idx = self.controlled_state_indices[valid]
                valid_state = state_idx < q.shape[0]
                modes = self._control_mode_ids_for_state_indices(state_idx)
                absolute = valid_state & (modes == 0)
                if np.any(absolute):
                    delta_cand = delta_cand.copy()
                    delta_nom = delta_nom.copy()
                    delta_cand[absolute] = (
                        candidate[0, action_idx[absolute]].astype(np.float64, copy=False)
                        - q[state_idx[absolute]].astype(np.float64, copy=False)
                    )
                    delta_nom[absolute] = (
                        nominal[0, action_idx[absolute]].astype(np.float64, copy=False)
                        - q[state_idx[absolute]].astype(np.float64, copy=False)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Nominal rejoin score q extraction failed: %s", exc)
        eps = 1e-9
        dot = float(np.dot(delta_cand, delta_nom))
        nominal_norm = float(np.linalg.norm(delta_nom))
        candidate_norm = float(np.linalg.norm(delta_cand))
        projection = dot / (nominal_norm * nominal_norm + eps)
        cosine = dot / (candidate_norm * nominal_norm + eps)
        return {
            "nominal_rejoin_score": float(max(0.0, projection)),
            "recover_projection_on_nominal": float(projection),
            "recover_cosine_to_nominal": float(cosine),
            "nominal_delta_norm": float(nominal_norm),
            "path_delta_norm": float(candidate_norm),
        }

    def _record_nominal_rejoin_target(
        self,
        target_info: Mapping[str, Any],
        rejoin_info: Mapping[str, Any] | None = None,
        progress_score: float | None = None,
    ) -> None:
        """Track nominal rejoin diagnostics shared by deform and recovery paths."""
        if target_info.get("available"):
            self.nominal_rejoin_available_count += 1
        else:
            self.nominal_rejoin_suppressed_count += 1
            reason = target_info.get("suppressed_reason")
            if reason == "stale_blocked_nominal":
                self.stale_nominal_rejoin_suppressed_count += 1
            elif reason == "nominal_prefix_unsafe":
                self.nominal_prefix_unsafe_suppressed_count += 1
        if rejoin_info is not None:
            projection = float(rejoin_info.get("recover_projection_on_nominal", 0.0))
            if projection > 0.0:
                self.recover_positive_projection_count += 1
            else:
                self.recover_nonpositive_projection_count += 1
            cosine = float(rejoin_info.get("recover_cosine_to_nominal", 0.0))
            if np.isfinite(projection):
                self.recovery._recover_projection_history.append(projection)
            if np.isfinite(cosine):
                self.recovery._recover_cosine_history.append(cosine)
        if progress_score is not None and np.isfinite(float(progress_score)):
            self.recovery._recover_task_progress_history.append(float(progress_score))

    def _write_state_tracking_actions(
        self,
        seed_chunk: Any,
        q_start: Any,
        target_q_seq: Any,
        action_idx: Any,
        state_idx: Any,
    ) -> np.ndarray:
        """Rewrite controlled actions so they track provided target state values."""
        chunk = np.asarray(seed_chunk, dtype=np.float32).copy()
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            return chunk
        q_prev = np.asarray(q_start, dtype=np.float32).reshape(-1).copy()
        target_q_seq = np.asarray(target_q_seq, dtype=np.float32)
        if target_q_seq.ndim == 1:
            target_q_seq = target_q_seq.reshape(1, -1)
        if target_q_seq.ndim != 2 or target_q_seq.shape[0] == 0:
            return chunk
        action_idx = np.asarray(action_idx, dtype=np.int64).reshape(-1)
        state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
        valid = (
            (action_idx < chunk.shape[1])
            & (state_idx < q_prev.shape[0])
            & (state_idx < target_q_seq.shape[1])
        )
        if not np.any(valid):
            return chunk
        action_idx = action_idx[valid]
        state_idx = state_idx[valid]
        modes = self._control_mode_ids_for_state_indices(state_idx)
        absolute = modes == 0
        delta = modes == 1
        velocity = modes == 2
        horizon = min(chunk.shape[0], target_q_seq.shape[0])
        dt = max(float(self.dt), 1e-9)
        for k in range(horizon):
            desired = target_q_seq[k]
            selected = chunk[k, action_idx].copy()
            if np.any(absolute):
                selected[absolute] = desired[state_idx[absolute]]
            if np.any(delta):
                selected[delta] = desired[state_idx[delta]] - q_prev[state_idx[delta]]
            if np.any(velocity):
                selected[velocity] = (
                    desired[state_idx[velocity]] - q_prev[state_idx[velocity]]
                ) / dt
            chunk[k, action_idx] = selected
            q_prev[state_idx] = desired[state_idx]
        return chunk

    def _acceptance_clearance_threshold(self) -> float:
        """Threshold between unsafe and acceptable for intervention handoff."""
        self._sync_intervention_weights()
        return float(self.min_clearance - self.acceptance_clearance_tol)

    def _make_rejoin_context(self, nominal_q_seq, nominal_ee_seq=None) -> dict[str, Any]:
        """Build rejoin context used by q/eepose distance helpers in both executors."""
        context: dict[str, Any] = {
            "nominal_q_seq": None,
            "q_state_indices": None,
            "q_weights": None,
            "q_nom_future": None,
            "ee_nom_seq": None,
            "ee_nom_future": None,
            "ee_nom_cache_time_ms": 0.0,
        }
        if not self.recovery.recoverable_deform_enabled or nominal_q_seq is None:
            return context

        nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        context["nominal_q_seq"] = nominal_q_seq
        if nominal_q_seq.shape[0] > self.recovery.min_rejoin_offset:
            valid = self.controlled_state_indices < nominal_q_seq.shape[1]
            state_idx = self.controlled_state_indices[valid]
            if state_idx.size:
                context["q_state_indices"] = state_idx
                context["q_weights"] = self._q_rejoin_weight_vector(
                    nominal_q_seq.shape[1], state_idx
                )
                context["q_nom_future"] = nominal_q_seq[
                    self.recovery.min_rejoin_offset :, state_idx
                ]
                context["q_nom_future_start_index"] = int(self.recovery.min_rejoin_offset)

        needs_ee = self.recovery.inner_rejoin_metric == "ee_pose" or (
            self.recovery.use_ee_final_check and self.recovery.final_rejoin_metric == "ee_pose"
        )
        if not needs_ee:
            return context

        ee_seq = None
        if nominal_ee_seq is not None:
            ee_seq = np.asarray(nominal_ee_seq, dtype=np.float32)
        elif self.recovery.cache_nominal_ee:
            t0 = time.perf_counter()
            ee_seq = self._ee_pose_sequence(nominal_q_seq)
            context["ee_nom_cache_time_ms"] = (time.perf_counter() - t0) * 1000.0
        if ee_seq is not None and ee_seq.shape[0] == nominal_q_seq.shape[0]:
            ee_seq = ee_seq.reshape(ee_seq.shape[0], -1).astype(np.float32)
            context["ee_nom_seq"] = ee_seq
            if ee_seq.shape[0] > self.recovery.min_rejoin_offset:
                context["ee_nom_future"] = ee_seq[self.recovery.min_rejoin_offset :]
        return context

    def _q_rejoin_weight_vector(self, q_dim: int, state_idx: np.ndarray) -> np.ndarray:
        """Create per-state rejoin weights from configured q_rejoin_weights."""
        del q_dim
        weights = self.recovery.q_rejoin_weights
        if weights is None:
            return np.ones(state_idx.shape[0], dtype=np.float32)
        try:
            weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        except Exception:  # pragma: no cover - defensive config path
            logger.warning("Invalid q_rejoin_weights; using unit weights.")
            return np.ones(state_idx.shape[0], dtype=np.float32)
        if weights.size == 1:
            return np.full(state_idx.shape[0], float(weights[0]), dtype=np.float32)
        if weights.size == state_idx.shape[0]:
            return weights.astype(np.float32)
        if weights.size >= int(np.max(state_idx)) + 1:
            return weights[state_idx].astype(np.float32)
        logger.warning(
            "q_rejoin_weights length %d does not match state dim %d or controlled dim %d; using unit weights.",
            weights.size,
            q_dim,
            state_idx.shape[0],
        )
        return np.ones(state_idx.shape[0], dtype=np.float32)

    def _nearest_future_loss(
        self,
        final_state: Any,
        nominal_seq: Any,
        weights: np.ndarray | None = None,
        start_index: int | None = None,
    ) -> tuple[float, int]:
        """Find the closest suffix point to a final state in a future trajectory."""
        nominal_seq = np.asarray(nominal_seq, dtype=np.float32)
        if start_index is None:
            if nominal_seq.shape[0] <= self.recovery.min_rejoin_offset:
                return float("inf"), None
            future = nominal_seq[self.recovery.min_rejoin_offset :]
            index_offset = self.recovery.min_rejoin_offset
        else:
            if nominal_seq.shape[0] == 0:
                return float("inf"), None
            future = nominal_seq
            index_offset = int(start_index)
        final = np.asarray(final_state, dtype=np.float32).reshape(-1)
        future = future.reshape(future.shape[0], -1)
        diff = future - final[None, :]
        if weights is not None:
            diff = diff * np.asarray(weights, dtype=np.float32).reshape(1, -1)
        dist_to_future = np.square(diff).sum(axis=1)
        best_local_idx = int(np.argmin(dist_to_future))
        return float(dist_to_future[best_local_idx]), best_local_idx + index_offset

    def _smoothness_loss(self, chunk, action_idx) -> float:
        """Compute smoothness score for candidate chunk actions."""
        chunk_arr = np.asarray(chunk, dtype=np.float32)
        action_idx = np.asarray(action_idx, dtype=np.int64)
        if len(action_idx) == 0 or chunk_arr.shape[0] <= 1:
            return 0.0
        velocity_loss = float(np.square(np.diff(chunk_arr[:, action_idx], axis=0)).mean())
        if chunk_arr.shape[0] <= 2:
            return velocity_loss
        acc = chunk_arr[2:, action_idx] - 2.0 * chunk_arr[1:-1, action_idx] + chunk_arr[:-2, action_idx]
        return velocity_loss + 0.5 * float(np.square(acc).mean())

    def rollout_chunk(
        self,
        action_chunk: Any,
        q_current: Any = None,
        qd_current: Any = None,
        obs: Any = None,
    ) -> np.ndarray:
        """Roll out an action chunk from an optional explicit current state."""
        del qd_current
        return self.rollout_nominal_chunk(self._obs_with_q(obs, q_current), action_chunk)

    def nominal_path_deviation_loss(self, q_seq: Any, nominal_q_seq: Any) -> float:
        """Measure mean squared deviation over controlled state dimensions."""
        if nominal_q_seq is None:
            return 0.0
        q_seq = np.asarray(q_seq, dtype=np.float32)
        nominal_q_seq = np.asarray(nominal_q_seq, dtype=np.float32)
        horizon = min(q_seq.shape[0], nominal_q_seq.shape[0])
        if horizon == 0:
            return 0.0
        valid = self.controlled_state_indices < min(q_seq.shape[1], nominal_q_seq.shape[1])
        state_idx = self.controlled_state_indices[valid]
        if state_idx.size == 0:
            return 0.0
        delta = q_seq[:horizon, state_idx] - nominal_q_seq[:horizon, state_idx]
        return float(np.square(delta).mean())

    def safety_trace_from_eval(self, safety_eval, horizon) -> SafetyTrace:
        """Return the canonical signed-clearance safety trace for one rollout."""
        return safety_trace_from_eval(safety_eval, horizon)

    def clearance_constraint_from_eval(
        self,
        safety_eval,
        horizon,
        required_clearance,
        *,
        prefix_len: int = 1,
        require_full_path: bool = True,
    ) -> SafetyConstraintResult:
        """Evaluate the shared clearance-margin safety contract."""
        return evaluate_clearance_constraint(
            self.safety_trace_from_eval(safety_eval, horizon),
            required_clearance,
            prefix_len=prefix_len,
            require_full_path=require_full_path,
        )

    def clearance_margin_loss(self, clearance_seq, required_clearance) -> float:
        """Shared squared hinge loss for clearance violations."""
        return clearance_margin_loss(clearance_seq, required_clearance)

    def _clearance_sequence_from_eval(self, safety_eval: Any, horizon: int):
        """Compatibility wrapper returning signed clearance sequence in meters."""
        return clearance_sequence_from_eval(safety_eval, horizon)

    def _obs_with_q(self, obs: Any, q_current: Any) -> Any:
        """Return an observation-like payload with q overridden for rollout scoring."""
        if q_current is None:
            return {} if obs is None else obs
        q = np.asarray(q_current, dtype=np.float32).reshape(-1)
        if obs is None:
            return {"q": q}
        if isinstance(obs, dict):
            merged = dict(obs)
            merged["q"] = q
            return merged
        return {"q": q}

    def _control_mode_id(self) -> int:
        """Return the scalar control-mode id for configured action semantics."""
        if self.control_type == "absolute":
            return 0
        if self.control_type == "delta":
            return 1
        return 2

    def _control_mode_ids_for_state_indices(self, state_idx: Any) -> np.ndarray:
        """Return per-state control modes, forcing floating-base dimensions to delta."""
        state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
        modes = np.full(state_idx.shape, self._control_mode_id(), dtype=np.int32)
        profile = self._rollout_profile()
        if bool(profile.get("base_delta_always", getattr(self, "rollout_base_delta_always", True))):
            modes[state_idx < min(4, self.expected_motion_dim)] = 1
        if bool(profile.get("enabled", getattr(self, "rollout_model_enabled", True))):
            absolute_indices = profile.get("absolute_state_indices", getattr(self, "rollout_absolute_state_indices", set()))
            delta_indices = profile.get("delta_state_indices", getattr(self, "rollout_delta_state_indices", set()))
            velocity_indices = profile.get("velocity_state_indices", getattr(self, "rollout_velocity_state_indices", set()))
            for i, state in enumerate(state_idx.tolist()):
                state_i = int(state)
                if state_i in absolute_indices:
                    modes[i] = 0
                if state_i in delta_indices:
                    modes[i] = 1
                if state_i in velocity_indices:
                    modes[i] = 2
        return modes

    def _apply_controlled_action_step(
        self,
        q: Any,
        action: Any,
        state_idx: Any,
        action_idx: Any,
    ) -> np.ndarray:
        """Apply one calibrated controlled action step to a state vector."""
        q_next = np.asarray(q, dtype=np.float32).reshape(-1).copy()
        if len(action_idx) == 0:
            return q_next
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
        action_idx = np.asarray(action_idx, dtype=np.int64).reshape(-1)
        modes = self._control_mode_ids_for_state_indices(state_idx)
        profile = self._rollout_profile()
        selected = action[action_idx].astype(np.float32, copy=True)
        if bool(profile.get("enabled", getattr(self, "rollout_model_enabled", True))):
            valid_state = state_idx < int(getattr(self, "rollout_model_state_dim", self.expected_motion_dim))
            scale = np.ones_like(selected, dtype=np.float32)
            bias = np.zeros_like(selected, dtype=np.float32)
            action_scale = profile.get("action_scale", self.rollout_action_scale_by_state)
            action_bias = profile.get("action_bias", self.rollout_action_bias_by_state)
            scale[valid_state] = action_scale[state_idx[valid_state]]
            bias[valid_state] = action_bias[state_idx[valid_state]]
            selected = selected * scale + bias
        current = q_next[state_idx]
        updated = selected.copy()
        absolute_mask = modes == 0
        delta_mask = modes == 1
        velocity_mask = modes == 2
        if bool(profile.get("enabled", getattr(self, "rollout_model_enabled", True))):
            valid_state = state_idx < int(getattr(self, "rollout_model_state_dim", self.expected_motion_dim))
            alpha = np.ones_like(selected, dtype=np.float32)
            delta_scale = np.ones_like(selected, dtype=np.float32)
            velocity_scale = np.ones_like(selected, dtype=np.float32)
            max_step = np.full_like(selected, np.inf, dtype=np.float32)
            target_alpha = profile.get("target_alpha", self.rollout_target_alpha_by_state)
            profile_delta_scale = profile.get("delta_scale", self.rollout_delta_scale_by_state)
            profile_velocity_scale = profile.get("velocity_scale", self.rollout_velocity_scale_by_state)
            profile_max_step = profile.get("max_step", self.rollout_max_step_by_state)
            alpha[valid_state] = target_alpha[state_idx[valid_state]]
            delta_scale[valid_state] = profile_delta_scale[state_idx[valid_state]]
            velocity_scale[valid_state] = profile_velocity_scale[state_idx[valid_state]]
            max_step[valid_state] = profile_max_step[state_idx[valid_state]]
        else:
            alpha = np.ones_like(selected, dtype=np.float32)
            delta_scale = np.ones_like(selected, dtype=np.float32)
            velocity_scale = np.ones_like(selected, dtype=np.float32)
            max_step = np.full_like(selected, np.inf, dtype=np.float32)
        updated[absolute_mask] = current[absolute_mask] + alpha[absolute_mask] * (
            selected[absolute_mask] - current[absolute_mask]
        )
        updated[delta_mask] = current[delta_mask] + delta_scale[delta_mask] * selected[delta_mask]
        updated[velocity_mask] = current[velocity_mask] + self.dt * velocity_scale[velocity_mask] * selected[velocity_mask]
        step = updated - current
        finite_step_limit = np.isfinite(max_step)
        if np.any(finite_step_limit):
            step[finite_step_limit] = np.clip(
                step[finite_step_limit],
                -max_step[finite_step_limit],
                max_step[finite_step_limit],
            )
            updated = current + step
        q_next[state_idx] = updated
        return q_next

    def _apply_controlled_action_step_batch(
        self,
        q: Any,
        actions: Any,
        state_idx: Any,
        action_idx: Any,
    ) -> np.ndarray:
        """Apply one calibrated controlled action step to a batch of states."""
        q_next = np.asarray(q, dtype=np.float32).copy()
        if len(action_idx) == 0:
            return q_next
        actions = np.asarray(actions, dtype=np.float32)
        state_idx = np.asarray(state_idx, dtype=np.int64).reshape(-1)
        action_idx = np.asarray(action_idx, dtype=np.int64).reshape(-1)
        modes = self._control_mode_ids_for_state_indices(state_idx)
        profile = self._rollout_profile()
        selected = actions[:, action_idx].astype(np.float32, copy=True)
        if bool(profile.get("enabled", getattr(self, "rollout_model_enabled", True))):
            valid_state = state_idx < int(getattr(self, "rollout_model_state_dim", self.expected_motion_dim))
            scale = np.ones((state_idx.shape[0],), dtype=np.float32)
            bias = np.zeros((state_idx.shape[0],), dtype=np.float32)
            action_scale = profile.get("action_scale", self.rollout_action_scale_by_state)
            action_bias = profile.get("action_bias", self.rollout_action_bias_by_state)
            scale[valid_state] = action_scale[state_idx[valid_state]]
            bias[valid_state] = action_bias[state_idx[valid_state]]
            selected = selected * scale.reshape(1, -1) + bias.reshape(1, -1)
        current = q_next[:, state_idx]
        updated = selected.copy()
        absolute_mask = modes == 0
        delta_mask = modes == 1
        velocity_mask = modes == 2
        if bool(profile.get("enabled", getattr(self, "rollout_model_enabled", True))):
            valid_state = state_idx < int(getattr(self, "rollout_model_state_dim", self.expected_motion_dim))
            alpha = np.ones((state_idx.shape[0],), dtype=np.float32)
            delta_scale = np.ones((state_idx.shape[0],), dtype=np.float32)
            velocity_scale = np.ones((state_idx.shape[0],), dtype=np.float32)
            max_step = np.full((state_idx.shape[0],), np.inf, dtype=np.float32)
            target_alpha = profile.get("target_alpha", self.rollout_target_alpha_by_state)
            profile_delta_scale = profile.get("delta_scale", self.rollout_delta_scale_by_state)
            profile_velocity_scale = profile.get("velocity_scale", self.rollout_velocity_scale_by_state)
            profile_max_step = profile.get("max_step", self.rollout_max_step_by_state)
            alpha[valid_state] = target_alpha[state_idx[valid_state]]
            delta_scale[valid_state] = profile_delta_scale[state_idx[valid_state]]
            velocity_scale[valid_state] = profile_velocity_scale[state_idx[valid_state]]
            max_step[valid_state] = profile_max_step[state_idx[valid_state]]
        else:
            alpha = np.ones((state_idx.shape[0],), dtype=np.float32)
            delta_scale = np.ones((state_idx.shape[0],), dtype=np.float32)
            velocity_scale = np.ones((state_idx.shape[0],), dtype=np.float32)
            max_step = np.full((state_idx.shape[0],), np.inf, dtype=np.float32)
        updated[:, absolute_mask] = current[:, absolute_mask] + alpha[absolute_mask].reshape(1, -1) * (
            selected[:, absolute_mask] - current[:, absolute_mask]
        )
        updated[:, delta_mask] = current[:, delta_mask] + delta_scale[delta_mask].reshape(1, -1) * selected[:, delta_mask]
        updated[:, velocity_mask] = current[:, velocity_mask] + self.dt * velocity_scale[velocity_mask].reshape(1, -1) * selected[:, velocity_mask]
        step = updated - current
        finite_step_limit = np.isfinite(max_step)
        if np.any(finite_step_limit):
            step[:, finite_step_limit] = np.clip(
                step[:, finite_step_limit],
                -max_step[finite_step_limit].reshape(1, -1),
                max_step[finite_step_limit].reshape(1, -1),
            )
            updated = current + step
        q_next[:, state_idx] = updated
        return q_next

    def _active_safety_info(self):
        return self.brake._active_safety_info()

    def _safechunk_recovery_corridor_info(self, *args, **kwargs):
        return self.recovery._safechunk_recovery_corridor_info(*args, **kwargs)

    def safechunk_replan_info(self, **overrides: Any) -> dict[str, Any]:
        """Build structured safechunk replan diagnostics from executor counters."""
        info = {
            "safechunk_replan_enabled": bool(self.recovery.safechunk_replan_enabled),
            "deform_replan_count": int(self.deform.deform_replan_count),
            "recover_replan_count": int(self.recovery.recovery_replan_count),
            "recovery_replan_count": int(self.recovery.recovery_replan_count),
            "recovery_failure_streak": int(self.recovery.recovery_failure_streak),
            "recovery_failure_streak_max": int(self.recovery.recovery_failure_streak_max),
            "recovery_optimizer_cooldown_remaining": int(
                self.recovery.recovery_optimizer_cooldown_remaining
            ),
            "recovery_retry_cooldown_steps": int(self.recovery.recover_retry_cooldown_steps),
            "recovery_attempts_in_unsafe_streak": int(
                self.recovery.recovery_attempts_in_unsafe_streak
            ),
            "recovery_max_attempts_per_unsafe_streak": int(
                self.recovery.recover_max_attempts_per_unsafe_streak
            ),
            "recovery_optimization_skipped_count": int(
                self.recovery.recovery_optimization_skipped_count
            ),
            "recovery_attempt_reset_after_brake_timeout_enabled": bool(
                getattr(self.recovery, "recovery_attempt_reset_after_brake_timeout", False)
            ),
            "recovery_attempt_reset_brake_timeout_steps": int(
                getattr(self.recovery, "recovery_attempt_reset_brake_timeout_steps", 0)
            ),
            "recovery_attempt_reset_min_hold_clearance": getattr(
                self.recovery, "recovery_attempt_reset_min_hold_clearance", None
            ),
            "recovery_attempt_reset_count": int(
                getattr(self.recovery, "recovery_attempt_reset_count", 0)
            ),
            "recovery_attempt_reset_last_brake_streak": int(
                getattr(self.recovery, "recovery_attempt_reset_last_brake_streak", 0)
            ),
            "recovery_attempt_reset_last_hold_clearance": getattr(
                self.recovery, "recovery_attempt_reset_last_hold_clearance", None
            ),
            "recovery_attempt_reset_last_reason": getattr(
                self.recovery, "recovery_attempt_reset_last_reason", None
            ),
            "committed_suffix_replan_attempt_count": int(
                self.recovery.committed_suffix_replan_attempt_count
            ),
            "committed_suffix_replan_accepted_count": int(
                self.recovery.committed_suffix_replan_accepted_count
            ),
            "committed_suffix_replan_rejected_count": int(
                self.recovery.committed_suffix_replan_rejected_count
            ),
            "committed_suffix_replan_budget_suppressed_count": int(
                self.recovery.committed_suffix_replan_budget_suppressed_count
            ),
            "committed_opportunistic_resume_count": int(
                self.recovery.committed_opportunistic_resume_count
            ),
            "committed_recovery_budget_exit_count": int(
                self.recovery.committed_recovery_budget_exit_count
            ),
            "committed_recover_steps_since_act": int(
                self.recovery.committed_recover_steps_since_act
            ),
            "committed_suffix_replans_in_current_recovery": int(
                self.recovery.committed_suffix_replans_in_current_recovery
            ),
            "recovery_optimization_skipped": False,
            "recovery_optimization_skip_reason": None,
            "stale_recovery_suppressed_count": int(self.recovery.stale_recovery_suppressed_count),
            "recovery_target_infeasible_count": int(self.recovery.recovery_target_infeasible_count),
            "emergency_brake_steps": int(self.recovery.emergency_brake_steps),
            "optimized_attempt_count": int(self.deform.optimized_attempt_count),
            "optimized_solution_count": int(self.deform.optimized_solution_count),
            "fallback_attempt_count": int(self.deform.fallback_attempt_count),
            "fallback_attempt_accepted_count": int(self.deform.fallback_attempt_accepted_count),
            "fallback_path_enabled": bool(self.deform.allow_candidate_fallback),
            "optimized_rejected_count": int(self.deform.optimized_rejected_count),
            "deform_option_attempt_count": int(self.deform.deform_option_attempt_count),
            "deform_accepted_count": int(self.deform.deform_accepted_count),
            "deform_rejected_count": int(self.deform.deform_rejected_count),
            "recover_option_attempt_count": int(self.deform.recover_option_attempt_count),
            "recover_accepted_count": int(self.deform.recover_accepted_count),
            "recover_rejected_count": int(self.deform.recover_rejected_count),
            "safe_prefix_accepted_count": int(self.deform.safe_prefix_accepted_count),
            "first_action_only_accepted_count": int(self.deform.first_action_only_accepted_count),
            "immediate_hard_reject_count": int(self.deform.immediate_hard_reject_count),
            "no_safe_prefix_reject_count": int(self.deform.no_safe_prefix_reject_count),
            "horizon_margin_reject_count": int(self.deform.horizon_margin_reject_count),
            "accepted_deform_steps": int(self.deform.accepted_deform_steps),
            "accepted_recover_steps": int(self.deform.accepted_recover_steps),
            "fallback_brake_after_reject_count": int(self.deform.fallback_brake_after_reject_count),
            "recover_step_since_deform": int(self.recovery.recover_step_since_deform),
            "nominal_rejoin_available_count": int(self.recovery.nominal_rejoin_available_count),
            "nominal_rejoin_suppressed_count": int(self.recovery.nominal_rejoin_suppressed_count),
            "stale_nominal_rejoin_suppressed_count": int(self.recovery.stale_nominal_rejoin_suppressed_count),
            "nominal_prefix_unsafe_suppressed_count": int(self.recovery.nominal_prefix_unsafe_suppressed_count),
            "recover_positive_projection_count": int(self.recovery.recover_positive_projection_count),
            "recover_nonpositive_projection_count": int(self.recovery.recover_nonpositive_projection_count),
            "mean_recover_projection_on_nominal": (
                float(np.mean(self.recovery._recover_projection_history))
                if self.recovery._recover_projection_history else None
            ),
            "mean_recover_cosine_to_nominal": (
                float(np.mean(self.recovery._recover_cosine_history))
                if self.recovery._recover_cosine_history else None
            ),
            "mean_recover_task_progress_score": (
                float(np.mean(self.recovery._recover_task_progress_history))
                if self.recovery._recover_task_progress_history else None
            ),
            "mean_recover_ordered_pose_loss": (
                float(np.mean(self.recovery._recover_ordered_pose_loss_history))
                if self.recovery._recover_ordered_pose_loss_history else None
            ),
            "mean_recover_ordered_delta_loss": (
                float(np.mean(self.recovery._recover_ordered_delta_loss_history))
                if self.recovery._recover_ordered_delta_loss_history else None
            ),
            "mean_recover_ordered_loss": (
                float(np.mean(self.recovery._recover_ordered_loss_history))
                if self.recovery._recover_ordered_loss_history else None
            ),
            **self.brake._active_safety_info(),
            **self.recovery._safechunk_recovery_corridor_info(),
            "deform_anchor_is_current": self.deform.deform_anchor_state is not None,
            "recover_anchor_is_current": self.recovery.recovery_anchor_state is not None,
            "recovery_anchor_is_current": self.recovery.recovery_anchor_state is not None,
            "recovery_target_mode": self.recovery.recovery_target_mode,
            "recovery_target_feasible": None,
            "stale_recovery_attempted": False,
            "stale_recovery_suppressed": False,
            "recover_to_task_progress": self.recovery.recovery_target_mode == "task_progress",
            "recovery_replanned_from_current_state": False,
            "return_to_old_path_suppressed": False,
            "emergency_brake_immediate_unsafe": False,
        }
        info.update(overrides)
        return info

    def _safechunk_replan_info(self, **overrides: Any) -> dict[str, Any]:
        """Backward-compatible private alias for safechunk replan diagnostics."""
        return self.safechunk_replan_info(**overrides)

    def _candidate_progress_score(self, obs: Any, candidate_chunk: Any) -> tuple[float, bool]:
        """Compute a simple progress score for a candidate's first action."""
        # Convert candidate to chunk form for shape-safe indexing.
        chunk: np.ndarray
        chunk, _ = self._as_chunk(candidate_chunk)
        # Only score controlled action dimensions.
        valid: np.ndarray = self._valid_control_indices(chunk)
        if not np.any(valid) or chunk.shape[0] == 0:
            return 0.0, False

        action_idx: np.ndarray = self.controlled_action_indices[valid]
        state_idx: np.ndarray = self.controlled_state_indices[valid]
        current_q: np.ndarray = self.extract_current_q(obs, chunk)
        valid_state: np.ndarray = state_idx < current_q.shape[0]
        if not np.any(valid_state):
            return 0.0, False

        action_idx = action_idx[valid_state]
        state_idx = state_idx[valid_state]
        return float(np.linalg.norm(chunk[0, action_idx] - current_q[state_idx])), True

    def _truncate_chunk_to_safe_prefix(self, candidate_chunk: Any, acceptance: Mapping[str, Any]) -> np.ndarray:
        """Clamp a candidate chunk to the accepted safe prefix length."""
        chunk, was_single = self._as_chunk(candidate_chunk)
        safe: np.ndarray = np.asarray(chunk, dtype=np.float32).copy()

        # Empty candidates should preserve the input shape exactly.
        if safe.shape[0] == 0:
            return safe.reshape(np.asarray(candidate_chunk).shape)

        acceptance_type = acceptance.get("acceptance_type")
        prefix_len: int = int(acceptance.get("safe_prefix_len", 0) or 0)
        if acceptance_type == "first_action_only":
            prefix_len = 1

        # For explicit safe-prefix acceptance, hold the last safe action afterwards.
        if acceptance_type in {"safe_prefix", "first_action_only"}:
            prefix_len = max(1, min(prefix_len, safe.shape[0]))
            hold = safe[prefix_len - 1].copy()
            safe[prefix_len:] = hold

        return safe[:1].astype(np.asarray(candidate_chunk).dtype, copy=False) if was_single else safe.astype(np.asarray(candidate_chunk).dtype, copy=False)

    def _q_rejoin_loss(
        self,
        q_seq: Any,
        nominal_q_seq: np.ndarray | None = None,
        rejoin_context: Mapping[str, Any] | None = None,
    ) -> tuple[float, int | None, float]:
        """Compute rejoin distance against the nominal trajectory suffix."""
        t0: float = time.perf_counter()
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq)

        q_seq_arr: np.ndarray = np.asarray(q_seq, dtype=np.float32)
        state_idx = rejoin_context.get("q_state_indices")
        future = rejoin_context.get("q_nom_future")
        weights = rejoin_context.get("q_weights")

        if state_idx is None or future is None or q_seq_arr.shape[0] == 0:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0

        state_idx_arr: np.ndarray = np.asarray(state_idx)
        future_arr: np.ndarray = np.asarray(future)
        valid: np.ndarray = state_idx_arr < q_seq_arr.shape[1]

        if not np.all(valid):
            state_idx_arr = state_idx_arr[valid]
            future_arr = future_arr[:, valid]
            weights = None if weights is None else np.asarray(weights)[valid]
        if state_idx_arr.size == 0:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0

        loss, j_best = self._nearest_future_loss(
            q_seq_arr[-1, state_idx_arr],
            future_arr,
            weights=weights,
            start_index=self.recovery.min_rejoin_offset,
        )
        return loss, int(j_best), (time.perf_counter() - t0) * 1000.0

    def _qd_rejoin_loss(
        self,
        q_seq: Any,
        nominal_q_seq: np.ndarray | None = None,
        target_index: int | None = None,
        rejoin_context: Mapping[str, Any] | None = None,
    ) -> tuple[float, int | None, float]:
        """Compute velocity-rejoin loss between final and nominal trajectory velocity."""
        t0: float = time.perf_counter()
        if rejoin_context is None:
            rejoin_context = self._make_rejoin_context(nominal_q_seq)

        q_seq_arr: np.ndarray = np.asarray(q_seq, dtype=np.float32)
        nominal = rejoin_context.get("nominal_q_seq")
        if nominal is None and nominal_q_seq is not None:
            nominal = np.asarray(nominal_q_seq, dtype=np.float32)

        state_idx = rejoin_context.get("q_state_indices")
        weights = rejoin_context.get("q_weights")

        if (
            target_index is None
            or nominal is None
            or state_idx is None
            or q_seq_arr.shape[0] < 2
        ):
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0

        nominal_arr: np.ndarray = np.asarray(nominal, dtype=np.float32)
        target_index_i: int = int(target_index)
        if target_index_i <= 0 or target_index_i >= nominal_arr.shape[0]:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0

        state_idx_arr: np.ndarray = np.asarray(state_idx)
        valid: np.ndarray = (state_idx_arr < q_seq_arr.shape[1]) & (
            state_idx_arr < nominal_arr.shape[1]
        )
        if not np.all(valid):
            state_idx_arr = state_idx_arr[valid]
            weights = None if weights is None else np.asarray(weights)[valid]

        if state_idx_arr.size == 0:
            return float("inf"), None, (time.perf_counter() - t0) * 1000.0

        dt: float = max(float(self.dt), 1e-9)
        candidate_qd: np.ndarray = (q_seq_arr[-1, state_idx_arr] - q_seq_arr[-2, state_idx_arr]) / dt
        nominal_qd: np.ndarray = (
            nominal_arr[target_index_i, state_idx_arr] - nominal_arr[target_index_i - 1, state_idx_arr]
        ) / dt
        diff: np.ndarray = candidate_qd - nominal_qd

        if weights is not None:
            weights_arr: np.ndarray = np.asarray(weights, dtype=np.float32).reshape(-1)
            diff = diff * weights_arr

        loss: float = float(np.square(diff).sum())
        return loss, target_index_i, (time.perf_counter() - t0) * 1000.0

    def _sqrt_loss(self, loss: float | np.ndarray) -> float:
        """Return root mean-square distance from a raw sum-of-squares loss."""
        loss_arr = np.asarray(loss, dtype=np.float32)
        if not np.isfinite(loss_arr).all():
            return float("inf")
        return float(np.sqrt(max(float(np.asarray(loss_arr).item()), 0.0)))

    def _qd_rejoin_acceptance(self, qd_index: int | None, qd_dist: float | np.ndarray) -> tuple[bool, dict[str, Any]]:
        """Evaluate whether the velocity rejoin is soft/hard-constraint acceptable."""
        try:
            qd_dist_val: float = float(qd_dist)
        except Exception:  # noqa: BLE001
            qd_dist_val = float("inf")

        hard_threshold: float = float(self.recovery.qd_rejoin_hard_threshold)
        finite_qd: bool = bool(np.isfinite(qd_dist_val))
        hard_enabled: bool = bool(np.isfinite(hard_threshold) and hard_threshold > 0.0)
        hard_failed: bool = bool(hard_enabled and finite_qd and qd_dist_val >= hard_threshold)

        required: bool = bool(self.recovery.require_qd_rejoin)
        threshold_ok: bool = bool(
            qd_index is not None
            and finite_qd
            and qd_dist_val < float(self.recovery.qd_rejoin_threshold)
        )
        ok: bool = bool((threshold_ok or not required) and not hard_failed)

        return ok, {
            "qd_rejoin_required": required,
            "qd_rejoin_hard_threshold": hard_threshold,
            "qd_rejoin_hard_failed": hard_failed,
            "qd_rejoin_soft_ok": threshold_ok,
        }

    def _recover_rejoin_weight_effective(self) -> float:
        """Compute ramped rejoin weight used by recovery objective terms."""
        weight: float = float(self.recovery.recover_rejoin_nominal_weight)
        if not self.recovery.safechunk_recover_enabled:
            return 0.0
        if self.recovery.rejoin_weight_schedule == "none":
            return 0.0
        if self.recovery.rejoin_weight_schedule == "ramp":
            ramp: float = min(
                1.0, float(self.recovery.recover_step_since_deform) / float(max(1, self.recovery.rejoin_ramp_steps))
            )
            weight *= ramp
        return float(weight)

    def _recover_direction_alignment_terms(self, rejoin_info: dict[str, Any] | None) -> dict[str, Any]:
        """Compute directional alignment penalty for recovery rejoin candidates."""
        available: bool = bool(
            rejoin_info is not None
            and float(rejoin_info.get("nominal_delta_norm", 0.0) or 0.0) > 1e-6
            and float(rejoin_info.get("path_delta_norm", 0.0) or 0.0) > 1e-6
        )

        cosine: float = float(
            0.0 if rejoin_info is None else rejoin_info.get("recover_cosine_to_nominal", 0.0)
        )
        if not np.isfinite(cosine):
            cosine = 0.0

        threshold: float = float(self.recovery.recover_min_direction_cosine)
        margin: float = float(self.recovery.recover_direction_alignment_margin)
        loss: float = float(max(0.0, threshold + margin - cosine) ** 2)
        ok: bool = bool(
            (not self.recovery.require_recover_direction_alignment)
            or (not available)
            or cosine >= threshold
        )

        return {
            "recover_direction_alignment_available": bool(available),
            "recover_direction_cosine": float(cosine),
            "recover_direction_cosine_threshold": float(threshold),
            "recover_direction_loss": float(loss),
            "recover_direction_ok": bool(ok),
        }

    def _zero_ordered_recovery_terms(self, target_index: int | None = None) -> dict[str, Any]:
        """Return disabled ordered-recovery diagnostics."""
        heading_threshold = float(
            getattr(self.recovery, "recover_ordered_heading_cosine_threshold", 0.75)
        )
        pose_threshold = float(self.recovery.recover_ordered_pose_threshold)
        pose_tube_threshold = max(pose_threshold * 4.0, pose_threshold + 0.05)
        return {
            "recover_ordered_path_available": False,
            "recover_ordered_target_index": target_index,
            "recover_ordered_horizon": 0,
            "recover_ordered_pose_loss": 0.0,
            "recover_ordered_delta_loss": 0.0,
            "recover_ordered_waypoint_pose_loss": 0.0,
            "recover_ordered_waypoint_rmse": 0.0,
            "recover_ordered_heading_loss": 0.0,
            "recover_ordered_heading_cosine": None,
            "recover_ordered_heading_cosine_min": None,
            "recover_ordered_heading_cosine_threshold": heading_threshold,
            "recover_ordered_backtrack_count": 0,
            "recover_ordered_monotonic_ok": True,
            "recover_ordered_pose_tube_threshold": float(pose_tube_threshold),
            "recover_ordered_pose_tube_ok": True,
            "recover_ordered_waypoint_tube_ok": True,
            "recover_ordered_strict_ok": True,
            "recover_ordered_waypoint_index_start": target_index,
            "recover_ordered_waypoint_index_end": target_index,
            "recover_ordered_loss": 0.0,
            "recover_ordered_pose_weight": float(self.recovery.recover_ordered_pose_weight),
            "recover_ordered_delta_weight": float(self.recovery.recover_ordered_delta_weight),
            "recover_ordered_heading_weight": float(
                getattr(self.recovery, "recover_ordered_heading_weight", 0.0)
            ),
            "recover_ordered_pose_threshold": float(self.recovery.recover_ordered_pose_threshold),
            "recover_ordered_delta_threshold": float(self.recovery.recover_ordered_delta_threshold),
            "recover_ordered_ok": True,
        }

    def _ordered_recovery_start_index(self, terminal_index: int, horizon: int, nominal_q_seq: Any) -> int | None:
        """Compute where ordered-recovery alignment starts in the nominal sequence."""
        if terminal_index is None:
            return None
        try:
            terminal_index_i: int = int(terminal_index)
        except Exception:  # noqa: BLE001
            return None

        horizon_i: int = max(1, int(horizon))
        nominal_len: int = 0 if nominal_q_seq is None else int(np.asarray(nominal_q_seq).shape[0])
        if nominal_len <= 0:
            return None

        start: int = terminal_index_i - horizon_i + 1
        return int(max(0, min(start, nominal_len - 1)))

    def _ordered_recovery_path_terms(
        self,
        q_seq: Any,
        nominal_q_seq: Any,
        *,
        target_index: int = 0,
        rejoin_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score recovered rollout against nearby ACT waypoints and their headings.

        The optimizer should not simply reproduce a same-index pose sequence.  Each
        recovered action is judged by where its rolled-out robot state actually
        lands, then that landing state is coupled to the nearest nominal ACT
        waypoint at or after ``target_index``.  The heading check compares the
        actual state-space tangent of the optimized rollout with the nominal
        tangent at the landed waypoint, which preserves ACT's heading sensitivity
        while allowing modest waypoint-position error.
        """
        if not self.recovery.safechunk_recover_enabled or (
            self.recovery.recover_ordered_pose_weight <= 0.0
            and self.recovery.recover_ordered_delta_weight <= 0.0
            and getattr(self.recovery, "recover_ordered_heading_weight", 0.0) <= 0.0
        ):
            return self._zero_ordered_recovery_terms(target_index)
        if target_index is None or q_seq is None or nominal_q_seq is None:
            return self._zero_ordered_recovery_terms(target_index)

        q_seq_arr = np.asarray(q_seq, dtype=np.float32)
        nominal_arr = np.asarray(nominal_q_seq, dtype=np.float32)
        if q_seq_arr.ndim != 2 or nominal_arr.ndim != 2 or q_seq_arr.shape[0] == 0:
            return self._zero_ordered_recovery_terms(target_index)

        target_index_i = int(target_index)
        if target_index_i < 0 or target_index_i >= nominal_arr.shape[0]:
            return self._zero_ordered_recovery_terms(target_index_i)

        valid = self.controlled_state_indices < min(q_seq_arr.shape[1], nominal_arr.shape[1])
        state_idx: np.ndarray = self.controlled_state_indices[valid]
        if state_idx.size == 0:
            return self._zero_ordered_recovery_terms(target_index_i)

        candidate_horizon: int = int(q_seq_arr.shape[0])
        nominal_window_len: int = max(
            1,
            int(getattr(self.recovery, "recover_act_frame_stack", candidate_horizon)),
        )
        if rejoin_context is not None:
            context_window_len = rejoin_context.get("nominal_rejoin_target_window_len")
            if context_window_len is not None:
                try:
                    nominal_window_len = max(1, int(context_window_len))
                except (TypeError, ValueError):
                    pass
        nominal_tail_end: int = min(nominal_arr.shape[0], target_index_i + nominal_window_len)
        nominal_tail: np.ndarray = nominal_arr[target_index_i:nominal_tail_end, :]
        if candidate_horizon <= 0 or nominal_tail.shape[0] <= 0:
            return self._zero_ordered_recovery_terms(target_index_i)

        candidate: np.ndarray = q_seq_arr[:candidate_horizon, state_idx]
        nominal_tail_controlled: np.ndarray = nominal_tail[:, state_idx]
        weights: np.ndarray = np.ones((state_idx.shape[0],), dtype=np.float32)
        if rejoin_context is not None:
            q_state_indices = np.asarray(
                rejoin_context.get("q_state_indices", []),
                dtype=np.int64,
            )
            q_weights = np.asarray(
                rejoin_context.get("q_weights", []),
                dtype=np.float32,
            )
            if q_state_indices.size > 0 and q_weights.size == q_state_indices.size:
                max_idx = int(max(np.max(state_idx), np.max(q_state_indices))) + 1
                weight_by_idx = np.ones((max_idx,), dtype=np.float32)
                for idx, weight in zip(q_state_indices, q_weights, strict=False):
                    if 0 <= int(idx) < weight_by_idx.shape[0]:
                        weight_by_idx[int(idx)] = float(
                            weight if np.isfinite(weight) and weight > 0.0 else 1.0
                        )
                weights = weight_by_idx[state_idx].astype(np.float32, copy=False)

        weighted_candidate: np.ndarray = candidate * weights.reshape(1, -1)
        weighted_nominal_tail: np.ndarray = nominal_tail_controlled * weights.reshape(1, -1)
        waypoint_dist_sq: np.ndarray = np.square(
            weighted_candidate[:, None, :] - weighted_nominal_tail[None, :, :]
        ).mean(axis=2)
        local_waypoint_idx: np.ndarray = np.argmin(waypoint_dist_sq, axis=1).astype(np.int64)
        matched_nominal: np.ndarray = nominal_tail_controlled[local_waypoint_idx]
        diff: np.ndarray = (candidate - matched_nominal) * weights.reshape(1, -1)
        waypoint_pose_loss: float = float(np.square(diff).mean())
        waypoint_rmse: float = float(np.sqrt(max(waypoint_pose_loss, 0.0)))

        if candidate_horizon >= 2 and nominal_tail_controlled.shape[0] >= 2:
            candidate_delta: np.ndarray = candidate[1:] - candidate[:-1]
            landing_local_idx: np.ndarray = local_waypoint_idx[1:]
            next_idx: np.ndarray = np.minimum(
                landing_local_idx + 1,
                nominal_tail_controlled.shape[0] - 1,
            )
            prev_idx: np.ndarray = np.maximum(landing_local_idx - 1, 0)
            tangent_start_idx: np.ndarray = np.where(
                next_idx > landing_local_idx,
                landing_local_idx,
                prev_idx,
            )
            tangent_end_idx: np.ndarray = np.where(
                next_idx > landing_local_idx,
                next_idx,
                landing_local_idx,
            )
            nominal_tangent: np.ndarray = (
                nominal_tail_controlled[tangent_end_idx]
                - nominal_tail_controlled[tangent_start_idx]
            )
            weighted_candidate_delta: np.ndarray = candidate_delta * weights.reshape(1, -1)
            weighted_nominal_tangent: np.ndarray = nominal_tangent * weights.reshape(1, -1)
            candidate_norm: np.ndarray = np.linalg.norm(weighted_candidate_delta, axis=1)
            tangent_norm: np.ndarray = np.linalg.norm(weighted_nominal_tangent, axis=1)
            valid_heading: np.ndarray = (candidate_norm > 1e-8) & (tangent_norm > 1e-8)
            heading_threshold: float = float(
                getattr(self.recovery, "recover_ordered_heading_cosine_threshold", 0.75)
            )
            if np.any(valid_heading):
                heading_cosines_arr: np.ndarray = np.sum(
                    weighted_candidate_delta[valid_heading]
                    * weighted_nominal_tangent[valid_heading],
                    axis=1,
                ) / (
                    candidate_norm[valid_heading] * tangent_norm[valid_heading] + 1e-8
                )
                heading_cosines_arr = np.clip(heading_cosines_arr, -1.0, 1.0)
                heading_cosine: float | None = float(np.mean(heading_cosines_arr))
                heading_cosine_min: float | None = float(np.min(heading_cosines_arr))
                heading_loss: float = float(
                    np.square(np.maximum(heading_threshold - heading_cosines_arr, 0.0)).mean()
                )
            else:
                heading_cosine = None
                heading_cosine_min = None
                heading_loss = 0.0

            waypoint_steps: np.ndarray = np.diff(local_waypoint_idx)
            backtrack_count: int = int(np.sum(waypoint_steps < 0))
            if waypoint_steps.size > 0:
                backtrack_loss: float = float(np.square(np.minimum(waypoint_steps, 0)).mean())
            else:
                backtrack_loss = 0.0
        else:
            heading_threshold = float(
                getattr(self.recovery, "recover_ordered_heading_cosine_threshold", 0.75)
            )
            heading_cosine = None
            heading_cosine_min = None
            heading_loss = 0.0
            backtrack_count = 0
            backtrack_loss = 0.0

        ordered_loss: float = float(
            self.recovery.recover_ordered_pose_weight * waypoint_pose_loss
            + getattr(self.recovery, "recover_ordered_heading_weight", 0.0) * heading_loss
            + self.recovery.recover_ordered_delta_weight * backtrack_loss
        )

        pose_threshold: float = float(self.recovery.recover_ordered_pose_threshold)
        pose_tube_threshold: float = max(pose_threshold * 4.0, pose_threshold + 0.05)
        monotonic_ok: bool = bool(
            backtrack_count
            <= int(getattr(self.recovery, "recover_ordered_backtrack_tolerance", 0))
        )
        heading_ok: bool = bool(
            heading_cosine_min is None or heading_cosine_min >= heading_threshold
        )
        pose_tube_ok: bool = bool(waypoint_pose_loss <= pose_tube_threshold)
        strict_ordered_ok: bool = bool(
            waypoint_pose_loss <= pose_threshold and heading_ok and monotonic_ok
        )
        # Ordered heading is a soft trajectory-shape diagnostic here.  The hard
        # OOD-style heading gate is recover_act_heading_ok, so ordered acceptance
        # only requires a forward monotonic match inside the waypoint tube.
        ordered_ok: bool = bool(
            (not self.recovery.require_recover_ordered_path)
            or (pose_tube_ok and monotonic_ok)
        )

        return {
            "recover_ordered_path_available": True,
            "recover_ordered_target_index": int(target_index_i),
            "recover_ordered_horizon": int(candidate_horizon),
            "recover_ordered_pose_loss": waypoint_pose_loss,
            "recover_ordered_delta_loss": backtrack_loss,
            "recover_ordered_waypoint_pose_loss": waypoint_pose_loss,
            "recover_ordered_waypoint_rmse": waypoint_rmse,
            "recover_ordered_heading_loss": heading_loss,
            "recover_ordered_heading_cosine": heading_cosine,
            "recover_ordered_heading_cosine_min": heading_cosine_min,
            "recover_ordered_heading_cosine_threshold": float(heading_threshold),
            "recover_ordered_backtrack_count": int(backtrack_count),
            "recover_ordered_monotonic_ok": bool(monotonic_ok),
            "recover_ordered_pose_tube_threshold": float(pose_tube_threshold),
            "recover_ordered_pose_tube_ok": bool(pose_tube_ok),
            "recover_ordered_waypoint_tube_ok": bool(
                pose_tube_ok and monotonic_ok
            ),
            "recover_ordered_strict_ok": bool(strict_ordered_ok),
            "recover_ordered_waypoint_index_start": int(target_index_i + local_waypoint_idx[0]),
            "recover_ordered_waypoint_index_end": int(target_index_i + local_waypoint_idx[-1]),
            "recover_ordered_loss": ordered_loss,
            "recover_ordered_pose_weight": float(self.recovery.recover_ordered_pose_weight),
            "recover_ordered_delta_weight": float(self.recovery.recover_ordered_delta_weight),
            "recover_ordered_heading_weight": float(
                getattr(self.recovery, "recover_ordered_heading_weight", 0.0)
            ),
            "recover_ordered_pose_threshold": float(self.recovery.recover_ordered_pose_threshold),
            "recover_ordered_delta_threshold": float(self.recovery.recover_ordered_delta_threshold),
            "recover_ordered_ok": ordered_ok,
        }

    def _contact_count_from_kwargs(self, kwargs: Mapping[str, Any]) -> int | None:
        """Extract contact count from dynamic call-time metadata."""
        for key in ("contact_count", "contacts", "robot_human_contact_count"):
            if key in kwargs and kwargs[key] is not None:
                try:
                    return int(kwargs[key])
                except Exception:  # noqa: BLE001
                    return None
        return None

    def _optimize_instead_of_unsafe_hold(
        self,
        obs: Any,
        nominal_chunk: Any,
        braked_chunk: Any,
        info: dict[str, Any],
        hold_info: Mapping[str, Any],
        original_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Replace an unsafe hold response with a deformation optimization path."""
        # Evaluate nominal rollout under current safety model first.
        nominal: np.ndarray
        nominal, _ = self._as_chunk(nominal_chunk)
        braked: np.ndarray
        braked, _ = self._as_chunk(braked_chunk)
        nominal_q_seq: np.ndarray = self.rollout_nominal_chunk(obs, nominal)

        safety_info: dict[str, Any] = self.evaluate_horizon_safety(obs, nominal_q_seq)
        forced_safety: dict[str, Any] = dict(safety_info)

        live_min = hold_info.get("live_monitor_min_h")
        hold_min = hold_info.get("hold_horizon_min_clearance")
        clearance_values: list[float | None] = [forced_safety.get("min_clearance")]

        for value in (live_min, hold_min):
            if value is None:
                continue
            try:
                value_f = float(value)
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(value_f):
                clearance_values.append(value_f)

        # Force safety failure when either monitor indicates a live hold risk.
        forced_min: float = min(float(v) for v in clearance_values if v is not None)
        if forced_min < float(self.min_clearance) or hold_info.get("live_monitor_contact_risk"):
            forced_safety.update(
                {
                    "horizon_safe": False,
                    "min_clearance": float(forced_min),
                    "first_violation": 0,
                    "unsafe_count": max(1, int(forced_safety.get("unsafe_count", 0) or 0)),
                }
            )

        safe_chunk, deform_info = self.deform.deform_chunk(
            obs,
            nominal,
            safety_info=forced_safety,
            braked_chunk=braked,
            nominal_q_seq=nominal_q_seq,
            **kwargs,
        )
        info.update(deform_info)
        info.update(
            {
                "unsafe_hold_replaced_by_optimization": True,
                "emergency_deform_away": False,
                "deform_trigger_reason": info.get("deform_trigger_reason", "unsafe_hold_optimization"),
            }
        )

        if bool(info.get("optimized_accepted", False)):
            self.recovery.recovery_failure_streak = 0
        elif info.get("optimized_accepted") is not None or info.get("fallback_used") is not None:
            self.recovery.recovery_failure_streak += 1
            self.recovery.recovery_failure_streak_max = max(
                self.recovery.recovery_failure_streak_max,
                self.recovery.recovery_failure_streak,
            )

        # If this is a committed recovery flow, try to bind the committed suffix path.
        if (
            info.get("optimized_accepted", False)
            and self.recovery.explicit_return
            and self.recovery.commit_accepted_chunks
        ):
            committed, commit_reject_info = self.recovery._commit_explicit_recovery_chunk(
                obs,
                safe_chunk,
                info,
                **kwargs,
            )
            if committed:
                committed_result = self.recovery._serve_committed_chunk(obs, nominal, original_shape, **kwargs)
                pending_committed_replan_info = self.recovery._pop_pending_committed_replan_info()
                if pending_committed_replan_info:
                    info.update(pending_committed_replan_info)
                if committed_result is not None:
                    committed_chunk, committed_info = committed_result
                    committed_info.update({
                        k: v for k, v in info.items() if k not in committed_info
                    })
                    self.last_info = committed_info
                    return committed_chunk, committed_info
            else:
                info.update(commit_reject_info)

        if not info.get("deform_safe", info.get("optimized_accepted", False)) and self.deform.unsafe_deformation_fallback == "brake":
            info.update(
                {
                    "safety_mode": "horizon_brake",
                    "mode": "horizon_brake",
                    "deformation_rejected": True,
                    "optimized_fallback": "brake",
                    "fallback_used": True,
                }
            )
            self.last_info = info
            return braked.reshape(original_shape), info

        info.update({"safety_mode": "horizon_deform", "mode": "horizon_deform"})
        self.last_info = info
        return np.asarray(safe_chunk, dtype=np.float32).reshape(original_shape), info

    def _update_last_safe_execution(
        self,
        obs: Any,
        executed_chunk: Any,
        info: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        """Persist the last known-safe action/sequence for fallback and recovery use."""
        chunk: np.ndarray
        chunk, _ = self._as_chunk(executed_chunk)
        if chunk.shape[0] == 0:
            return

        contact_count: int | None = self._contact_count_from_kwargs(kwargs)
        if contact_count is not None and contact_count > 0:
            return

        clearance_raw: Any = info.get("immediate_clearance", info.get("min_clearance"))
        if clearance_raw is None:
            clearance_raw = info.get("deform_min_clearance", info.get("brake_min_clearance"))

        clearance: float
        try:
            clearance = float(clearance_raw)
        except Exception:  # noqa: BLE001
            clearance = float("inf")

        # Do not stamp unsafe trajectories as last safe examples.
        if clearance < float(self.brake.active_safety_hard_min_clearance):
            return

        self.last_safe_action = np.asarray(chunk[0], dtype=np.float32).copy()
        self.last_safe_chunk = np.asarray(chunk, dtype=np.float32).copy()

        try:
            self.last_safe_q = self._rollout_one_step_from_q(
                self.extract_current_q(obs, chunk),
                chunk[0],
            ).copy()
        except Exception:  # noqa: BLE001
            self.last_safe_q = self.extract_current_q(obs, chunk).copy()

    def _get_oscbf_operator(self) -> Any:
        """Instantiate and return the shared OSCBF operator when configured."""
        if callable(self.oscbf_operator) or self.oscbf_operator is None:
            return self.oscbf_operator
        if self._operator_instantiation_failed:
            return None
        target = None
        try:
            target = self.oscbf_operator.get("_target_")
        except AttributeError:
            if isinstance(self.oscbf_operator, dict):
                target = self.oscbf_operator.get("_target_")
        if target is None:
            return self.oscbf_operator
        try:
            import hydra

            self.oscbf_operator = hydra.utils.instantiate(self.oscbf_operator)
            self.parent.oscbf_operator = self.oscbf_operator
        except Exception as exc:  # pragma: no cover - depends on deployment config
            logger.warning(
                "SafeChunk-Deform could not instantiate oscbf_operator; "
                "falling back to identity single-step deformation: %s",
                exc,
            )
            self._operator_instantiation_failed = True
            self.parent._operator_instantiation_failed = True
            return None
        return self.oscbf_operator

    def filter_single_action(self, action: Any, obs: Any = None, **kwargs: Any) -> np.ndarray:
        """Apply the configured single-step safety operator to one action."""
        return self._call_single_step_operator(action, obs=obs, **kwargs)

    def _call_single_step_operator(self, action: Any, obs: Any = None, **kwargs: Any) -> np.ndarray:
        """Call the OSCBF single-step operator with supported calling conventions."""
        op = self._get_oscbf_operator()
        if not callable(op):
            return np.asarray(action).copy()
        attempts = (
            lambda: op(action=action, observations=obs, **kwargs),
            lambda: op(action=action, obs=obs, **kwargs),
            lambda: op(action, obs, **kwargs),
            lambda: op(action, **kwargs),
            lambda: op(action),
        )
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        raise last_error

    def _jax_optimizer_ready(self):
        return self.deform._jax_optimizer_ready()

    def _deform_chunk_optimized_with_fallback(self, *args, **kwargs):
        return self.parent.deform._deform_chunk_optimized_with_fallback(*args, **kwargs)

    def _post_recovery_act_window_info(self, *, interrupted=False):
        if interrupted and self.recovery.post_recovery_act_window_active:
            self.recovery.post_recovery_act_window_interrupted_count += 1
            self.recovery.post_recovery_act_window_active = False
            self.recovery.post_recovery_act_steps_remaining = 0
        return {
            "post_recovery_act_window_active": bool(
                self.recovery.post_recovery_act_window_active
            ),
            "post_recovery_act_steps_remaining": int(
                self.recovery.post_recovery_act_steps_remaining
            ),
            "post_recovery_act_window_interrupted": bool(interrupted),
        }

    def _committed_replay_obs(self, q_current, obs=None):
        base_obs = self.recovery.committed_planning_obs
        if base_obs is None:
            base_obs = self._copy_obs_for_committed_replay(obs)
        if base_obs is None:
            base_obs = obs
        return self._obs_with_q(base_obs, q_current)

    def _copy_obs_for_committed_replay(self, obs):
        """Snapshot only safety-relevant observation state for committed replay.

        Policy observations may contain large RGB frame stacks.  Committed replay
        diagnostics only need robot/human low-dimensional state, and _obs_with_q
        injects the planned robot q before safety evaluation.  Keep this copy
        intentionally narrow so optimizer warmup does not spend time cloning
        visual observations that safety never reads.
        """
        blocked_tokens = (
            "rgb",
            "image",
            "pixel",
            "camera",
            "depth",
            "mask",
            "visual",
        )
        safety_tokens = (
            "q",
            "qd",
            "qpos",
            "qvel",
            "state",
            "human",
            "capsule",
            "obstacle",
            "robot",
            "joint",
            "carrier",
        )

        def keep_key(key):
            text = str(key).lower()
            if any(token in text for token in blocked_tokens):
                return False
            return any(token in text for token in safety_tokens)

        def copy_value(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, dict):
                return {
                    key: copy_value(item)
                    for key, item in value.items()
                    if keep_key(key)
                }
            if isinstance(value, list):
                return [copy_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(copy_value(item) for item in value)
            copy_method = getattr(value, "copy", None)
            if callable(copy_method):
                try:
                    return copy_method()
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Observation copy fallback kept original %s",
                        type(value).__name__,
                    )
            return value

        if obs is None:
            return None
        if isinstance(obs, dict):
            copied = {
                key: copy_value(value)
                for key, value in obs.items()
                if keep_key(key)
            }
            return copied if copied else None
        return copy_value(obs)

    def _clear_brake_hold(self):
        self.brake_hold_anchor = None
        self.brake_hold_anchor_state = None

    def _valid_control_indices(self, chunk):
        return self.controlled_action_indices < chunk.shape[1]

    def _hold_return_or_emergency_deform(self, obs, nominal_chunk, braked_chunk, info, original_shape, **kwargs):
        if not (self.brake.safechunk_active_safety_enabled and self.brake.check_hold_horizon_safety):
            self.brake._update_last_safe_execution(obs, braked_chunk, info, **kwargs)
            self.last_info = info
            return braked_chunk.reshape(original_shape), info
        hold_info = self.evaluate_hold_or_brake_acceptance(
            obs,
            braked_chunk,
            gate_live_monitor_clearance=True,
            **kwargs,
        )
        info.update(hold_info)
        if hold_info.get("accepted"):
            self.brake._update_last_safe_execution(obs, braked_chunk, info, **kwargs)
            self.last_info = info
            return braked_chunk.reshape(original_shape), info
        if self.brake.optimize_when_hold_unsafe and self.deform.deformation_enabled:
            return self._optimize_instead_of_unsafe_hold(
                obs,
                nominal_chunk,
                braked_chunk,
                info,
                hold_info,
                original_shape,
                **kwargs,
            )
        if self.brake.emergency_deform_when_hold_unsafe:
            emergency_chunk, emergency_info = self.emergency_deform_away(
                obs,
                reference_action=braked_chunk,
                nominal_chunk=nominal_chunk,
                hold_info=hold_info,
                **kwargs,
            )
            info.update(emergency_info)
            self.brake._update_last_safe_execution(obs, emergency_chunk, info, **kwargs)
            self.last_info = info
            return emergency_chunk.reshape(original_shape), info
        self.last_info = info
        return braked_chunk.reshape(original_shape), info

    def _temporary_progress_deadlocked(self):
        if len(self.brake._temporary_progress_history) < max(1, self.brake.temporary_progress_window):
            return False, False
        delta = self.brake._temporary_progress_history[-1] - self.brake._temporary_progress_history[0]
        return bool(delta < self.brake.temporary_min_progress_delta), True

    def _temporary_streak_info(
             self,
             *,
             waiting=False,
             trigger_reason=None,
             nominal_became_safe=False,
             resume_after_wait=False,
        ):
            return {
                "unsafe_streak": int(self.parent.unsafe_streak),
                "brake_streak": int(self.brake.brake_streak),
                "recovery_failure_streak": int(self.recovery.recovery_failure_streak),
                "recovery_failure_streak_max": int(self.recovery.recovery_failure_streak_max),
                "recovery_optimizer_cooldown_remaining": int(
                    self.recovery.recovery_optimizer_cooldown_remaining
                ),
                "recovery_retry_cooldown_steps": int(self.recovery.recover_retry_cooldown_steps),
                "recovery_attempts_in_unsafe_streak": int(
                    self.recovery.recovery_attempts_in_unsafe_streak
                ),
                "recovery_max_attempts_per_unsafe_streak": int(
                    self.recovery.recover_max_attempts_per_unsafe_streak
                ),
                "recovery_optimization_skipped_count": int(
                    self.recovery.recovery_optimization_skipped_count
                ),
                "recovery_attempt_reset_after_brake_timeout_enabled": bool(
                    getattr(self.recovery, "recovery_attempt_reset_after_brake_timeout", False)
                ),
                "recovery_attempt_reset_brake_timeout_steps": int(
                    getattr(self.recovery, "recovery_attempt_reset_brake_timeout_steps", 0)
                ),
                "recovery_attempt_reset_min_hold_clearance": getattr(
                    self.recovery, "recovery_attempt_reset_min_hold_clearance", None
                ),
                "recovery_attempt_reset_count": int(
                    getattr(self.recovery, "recovery_attempt_reset_count", 0)
                ),
                "recovery_attempt_reset_last_brake_streak": int(
                    getattr(self.recovery, "recovery_attempt_reset_last_brake_streak", 0)
                ),
                "recovery_attempt_reset_last_hold_clearance": getattr(
                    self.recovery, "recovery_attempt_reset_last_hold_clearance", None
                ),
                "recovery_attempt_reset_last_reason": getattr(
                    self.recovery, "recovery_attempt_reset_last_reason", None
                ),
                "temporary_blocker_waiting": bool(waiting),
                "deform_trigger_reason": trigger_reason,
                "nominal_became_safe_after_brake": bool(nominal_became_safe),
                "resume_act_after_wait": bool(resume_after_wait),
                "temporary_wait_step": bool(waiting),
                "deform_suppressed_by_temporary_wait": bool(waiting),
                "deform_after_persistent_block": bool(
                    trigger_reason in {"persistent_unsafe", "brake_timeout", "progress_deadlock"}
                ),
            }