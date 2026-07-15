import time
import numpy as np

class GradientDescent:
    """Finite-difference gradient-descent optimizer used by the chunk helper."""

    def __init__(
        self,
        *,
        rng,
        max_iters: int,
        min_iters: int,
        lr: float,
        gradient_samples: int,
        eps: float,
        adam_beta1: float,
        adam_beta2: float,
        min_improvement: float,
        line_search_scales: tuple[float, ...],
        batched_line_search: bool,
        early_stop_on_candidate: bool,
    ):
        self.rng = rng
        self.max_iters = max(1, int(max_iters))
        self.min_iters = max(1, int(min_iters))
        self.lr = float(lr)
        self.gradient_samples = max(1, int(gradient_samples))
        self.eps = max(1e-9, float(eps))
        self.beta1 = float(np.clip(adam_beta1, 0.0, 0.999))
        self.beta2 = float(np.clip(adam_beta2, 0.0, 0.9999))
        self.min_improvement = max(0.0, float(min_improvement))

        scales = []
        for value in line_search_scales:
            scale = float(value)
            if scale > 0.0:
                scales.append(scale)
        self.line_search_scales = tuple(scales or [1.0])
        self.batched_line_search = bool(batched_line_search)
        self.early_stop_on_candidate = bool(early_stop_on_candidate)

        self._adam_m: np.ndarray | None = None
        self._adam_v: np.ndarray | None = None
        self._beta1_pow = 1.0
        self._beta2_pow = 1.0

        self.metrics = {
            "optimizer_method": "gradient",
            "gradient_iterations_run": 0,
            "gradient_max_iters": int(self.max_iters),
            "gradient_samples": int(self.gradient_samples),
            "gradient_eps": float(self.eps),
            "gradient_early_stopped": False,
            "gradient_candidate_early_stopped": False,
            "gradient_batched_line_search": bool(self.batched_line_search),
            "gradient_line_search_batch_evaluations": 0,
            "gradient_line_search_batch_size": int(len(self.line_search_scales)),
            "gradient_jax_scan_used": False,
            "gradient_jax_scan_used_count": 0,
            "gradient_full_jax_scan_used": False,
            "gradient_full_jax_scan_time_ms": 0.0,
            "gradient_initial_records_time_ms": 0.0,
            "gradient_initial_project_time_ms": 0.0,
            "gradient_initial_batch_cost_time_ms": 0.0,
            "gradient_initial_record_build_time_ms": 0.0,
            "gradient_initial_sort_time_ms": 0.0,
            "gradient_direction_sample_time_ms": 0.0,
            "gradient_perturb_control_time_ms": 0.0,
            "gradient_perturb_project_time_ms": 0.0,
            "gradient_perturb_records_time_ms": 0.0,
            "gradient_line_control_time_ms": 0.0,
            "gradient_line_project_time_ms": 0.0,
            "gradient_line_records_time_ms": 0.0,
        }

    @staticmethod
    def _pick_best(records, key="cost"):
        if not records:
            raise ValueError("No candidate records available")
        return min(records, key=lambda item: float(item[key]))

    def _evaluate(
        self,
        candidates,
        cost_fn,
        batch_cost_fn,
        action_idx,
    ):
        if candidates is None or np.asarray(candidates).size == 0:
            return []
        if batch_cost_fn is not None:
            try:
                costs, losses_list = batch_cost_fn(candidates)
                if len(costs) != len(candidates) or len(losses_list) != len(candidates):
                    raise ValueError("batch cost result length mismatch")
                return [
                    {
                        "cost": float(cost),
                        "losses": losses,
                        "chunk": np.asarray(candidate, dtype=np.float32),
                        "ctrl": np.asarray(candidate[:, action_idx], dtype=np.float32),
                    }
                    for cost, losses, candidate in zip(costs, losses_list, candidates)
                ]
            except Exception:
                pass

        records = []
        for candidate in candidates:
            cost, losses = cost_fn(candidate)
            records.append(
                {
                    "cost": float(cost),
                    "losses": losses,
                    "chunk": np.asarray(candidate, dtype=np.float32),
                    "ctrl": np.asarray(candidate[:, action_idx], dtype=np.float32),
                }
            )
        return records

    def _project(self, nominal_chunk, action_idx, candidate_ctrl, project_fn):
        projected = nominal_chunk.copy()
        projected[:, action_idx] = candidate_ctrl
        return project_fn(projected)

    def optimize(
        self,
        nominal_chunk,
        action_idx,
        cost_fn,
        seed_chunks,
        batch_cost_fn,
        early_stop_fn,
        project_fn,
    ):
        nominal_chunk = np.asarray(nominal_chunk, dtype=np.float32)
        initial_t0 = time.perf_counter()

        candidate_chunks = [
            project_fn(np.asarray(nominal_chunk, dtype=np.float32))
        ]
        for seed in (seed_chunks or []):
            seed = np.asarray(seed, dtype=np.float32)
            if seed.shape == nominal_chunk.shape:
                candidate_chunks.append(project_fn(seed))

        records = self._evaluate(candidate_chunks, cost_fn, batch_cost_fn, action_idx)
        if not records:
            raise RuntimeError("Gradient optimization produced no initial records")

        best_record = self._pick_best(records)
        self.metrics["gradient_initial_records_time_ms"] = 1000.0 * (
            time.perf_counter() - initial_t0
        )

        # Adam state is over the control tensor, not full action chunk.
        ctrl_shape = best_record["chunk"][:, action_idx].shape
        self._adam_m = np.zeros(ctrl_shape, dtype=np.float32)
        self._adam_v = np.zeros(ctrl_shape, dtype=np.float32)

        best_record.setdefault("losses", {})
        best_record["losses"].update(self.metrics)

        # The initial seed may already satisfy a coarse progress predicate while
        # still failing stricter downstream gates, such as ordered ACT rejoin.
        # Always run the optimization loop at least once; min_iters then controls
        # when the predicate is allowed to stop later iterations.
        for iteration in range(self.max_iters):
            self.metrics["gradient_iterations_run"] = int(iteration + 1)
            base_cost = float(best_record["cost"])
            base_ctrl = best_record["chunk"][:, action_idx]

            dir_t0 = time.perf_counter()
            dirs = self.rng.normal(size=(self.gradient_samples,) + base_ctrl.shape).astype(np.float32)
            flat = dirs.reshape(self.gradient_samples, -1)
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            norms = np.where(norms <= 1e-12, 1.0, norms)
            dirs = (flat / norms).reshape((self.gradient_samples,) + base_ctrl.shape)
            self.metrics["gradient_direction_sample_time_ms"] += 1000.0 * (
                time.perf_counter() - dir_t0
            )

            perturb_t0 = time.perf_counter()
            perturb_deltas = [self.eps * d for d in dirs]
            perturb_chunks = [
                self._project(nominal_chunk, action_idx, base_ctrl + delta, project_fn)
                for delta in perturb_deltas
            ]
            self.metrics["gradient_perturb_project_time_ms"] += 1000.0 * (
                time.perf_counter() - perturb_t0
            )

            perturb_eval_t0 = time.perf_counter()
            perturb_records = self._evaluate(
                np.stack(perturb_chunks, axis=0).astype(np.float32),
                cost_fn,
                batch_cost_fn,
                action_idx,
            )
            self.metrics["gradient_perturb_records_time_ms"] += 1000.0 * (
                time.perf_counter() - perturb_eval_t0
            )

            perturb_records.sort(key=lambda item: item["cost"])
            perturb_costs = np.asarray([item["cost"] for item in perturb_records], dtype=np.float32)
            grad = np.mean((perturb_costs - base_cost)[:, None, None] * dirs, axis=0) / self.eps
            grad = np.asarray(grad, dtype=np.float32)

            self._beta1_pow *= self.beta1
            self._beta2_pow *= self.beta2
            self._adam_m = self.beta1 * self._adam_m + (1.0 - self.beta1) * grad
            self._adam_v = self.beta2 * self._adam_v + (1.0 - self.beta2) * (grad * grad)
            m_hat = self._adam_m / max(1e-12, 1.0 - self._beta1_pow)
            v_hat = self._adam_v / max(1e-12, 1.0 - self._beta2_pow)
            direction = -m_hat / (np.sqrt(v_hat) + 1e-8)

            line_t0 = time.perf_counter()
            line_deltas = [scale * self.lr * direction for scale in self.line_search_scales]
            line_chunks = [
                self._project(nominal_chunk, action_idx, base_ctrl + line_delta, project_fn)
                for line_delta in line_deltas
            ]
            self.metrics["gradient_line_control_time_ms"] += 1000.0 * (
                time.perf_counter() - line_t0
            )

            line_eval_t0 = time.perf_counter()
            line_records = self._evaluate(
                np.stack(line_chunks, axis=0).astype(np.float32),
                cost_fn,
                batch_cost_fn if self.batched_line_search else None,
                action_idx,
            )
            self.metrics["gradient_line_records_time_ms"] += 1000.0 * (
                time.perf_counter() - line_eval_t0
            )
            self.metrics["gradient_line_search_batch_evaluations"] += len(line_chunks)

            if line_records:
                line_best = self._pick_best(line_records)
                if line_best["cost"] + self.min_improvement < base_cost:
                    best_record = line_best
                    if early_stop_fn is not None and self.metrics["gradient_iterations_run"] >= self.min_iters:
                        if early_stop_fn(best_record):
                            self.metrics["gradient_early_stopped"] = True
                            break
                else:
                    self.metrics["gradient_candidate_early_stopped"] = True
                    if self.early_stop_on_candidate:
                        self.metrics["gradient_early_stopped"] = True
                        break

        best_record = self._pick_best([best_record] + perturb_records)
        best_record.setdefault("losses", {})
        best_record["losses"].update(self.metrics)
        best_record["losses"]["gradient_perturb_control_time_ms"] = 0.0
        return best_record
