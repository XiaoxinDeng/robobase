from __future__ import annotations

import numpy as np


class CrossEntropyMethod:
    """Cross-entropy optimizer used by the SafeChunk deformation filter."""

    def __init__(
        self,
        *,
        rng,
        max_iters: int,
        min_iters: int,
        population: int,
        elite_frac: float,
        sigma: float,
    ):
        self.rng = rng
        self.max_iters = max(1, int(max_iters))
        self.min_iters = max(1, int(min_iters))
        self.population = max(1, int(population))
        self.elite_frac = max(1.0 / self.population, float(elite_frac))
        self.sigma = max(1e-9, float(sigma))
        self.metrics = {
            "optimizer_method": "cem",
            "cem_iterations_run": 0,
            "cem_early_stopped": False,
            "cem_max_iters": int(self.max_iters),
            "cem_population": int(self.population),
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

    def optimize(
        self,
        nominal_chunk,
        action_idx,
        cost_fn,
        seed_chunks,
        batch_cost_fn,
        early_stop_fn,
        project_fn,
        project_population_fn=None,
    ):
        nominal_chunk = np.asarray(nominal_chunk, dtype=np.float32)
        seed_chunks = seed_chunks or []
        seed_ctrl = []
        for seed in seed_chunks:
            seed = np.asarray(seed, dtype=np.float32)
            if seed.shape == nominal_chunk.shape:
                projected = project_fn(seed, nominal_chunk, action_idx)
                seed_ctrl.append(projected[:, action_idx].copy())

        mean = nominal_chunk[:, action_idx].copy()
        sigma = np.full_like(mean, self.sigma, dtype=np.float32)
        best_record = None

        num_iters = int(self.max_iters)
        min_iters = int(self.min_iters)
        iterations_run = 0
        early_stopped = False

        for iter_idx in range(num_iters):
            iterations_run = iter_idx + 1
            ctrl_samples = [mean.copy()]
            ctrl_samples.extend(sample.copy() for sample in seed_ctrl)
            remaining = max(0, self.population - len(ctrl_samples))
            if remaining:
                noise = self.rng.normal(
                    loc=0.0,
                    scale=sigma[None, :, :],
                    size=(remaining,) + mean.shape,
                ).astype(np.float32)
                ctrl_samples.extend(mean[None, :, :] + noise)

            ctrl_sample_batch = np.stack(ctrl_samples, axis=0).astype(np.float32)
            candidate_batch = None
            if project_population_fn is not None:
                try:
                    candidate_batch = project_population_fn(
                        nominal_chunk,
                        ctrl_sample_batch,
                        action_idx,
                    )
                except TypeError:
                    candidate_batch = None
                except Exception:  # pragma: no cover - passthrough to NumPy path
                    candidate_batch = None

            if candidate_batch is None:
                candidates = []
                for ctrl_sample in ctrl_samples:
                    candidate = nominal_chunk.copy()
                    candidate[:, action_idx] = ctrl_sample
                    candidate = project_fn(
                        candidate,
                        nominal_chunk,
                        action_idx,
                    )
                    candidates.append(candidate)
            else:
                candidates = [candidate_batch[i].copy() for i in range(candidate_batch.shape[0])]

            records = self._evaluate(
                np.stack(candidates, axis=0).astype(np.float32),
                cost_fn,
                batch_cost_fn,
                action_idx,
            )
            records.sort(key=lambda item: item["cost"])
            if best_record is None or records[0]["cost"] < best_record["cost"]:
                best_record = records[0]

            if (
                early_stop_fn is not None
                and iterations_run >= min_iters
                and early_stop_fn(best_record)
            ):
                early_stopped = True
                break

            elite_count = max(1, int(round(self.population * self.elite_frac)))
            elite_ctrl = np.stack([record["ctrl"] for record in records[:elite_count]], axis=0)
            elite_mean = elite_ctrl.mean(axis=0).astype(np.float32)
            elite_sigma = elite_ctrl.std(axis=0).astype(np.float32)
            mean = (0.3 * mean + 0.7 * elite_mean).astype(np.float32)
            sigma = np.maximum(elite_sigma, self.sigma * 0.05).astype(np.float32)

        if best_record is None:
            raise RuntimeError("No candidate produced by optimizer")

        best_record["chunk"] = project_fn(
            best_record["chunk"],
            nominal_chunk,
            action_idx,
        )
        self.metrics["cem_iterations_run"] = int(iterations_run)
        self.metrics["cem_early_stopped"] = bool(early_stopped)
        best_record.setdefault("losses", {})
        best_record["losses"].update(self.metrics)
        return best_record
