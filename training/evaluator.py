"""Window-based and full-rollout evaluation in normalized and physical units."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist

from datasets.normalization import FeatureNormalizer
from training.metrics import channel_error_sums, overall_horizon_sums, summarize_channel_metrics, summarize_horizon_curve


class Evaluator:
    """Evaluate fixed windows and full trajectories in normalized and physical units."""

    def __init__(self, model: torch.nn.Module, normalizer: FeatureNormalizer, device: torch.device, amp_mode: str = "none") -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.amp_mode = amp_mode

    def _autocast(self):
        if self.device.type != "cuda" or self.amp_mode == "none":
            return nullcontext()
        dtype = torch.bfloat16 if self.amp_mode == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.no_grad()
    def evaluate_loader(self, loader) -> dict[str, float]:
        """Evaluate a fixed-window loader over the full stored horizon."""

        self.model.eval()
        sample = loader.dataset[0]
        state_dim = int(sample.y_future.shape[-1])
        norm_sq_sum = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        norm_abs_sum = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        phys_sq_sum = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        phys_abs_sum = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        count = torch.zeros(state_dim, device=self.device, dtype=torch.float64)

        for batch in loader:
            batch = batch.to(self.device)
            with self._autocast():
                y_pred = self.model(batch)
            if not torch.isfinite(y_pred).all():
                raise FloatingPointError("Non-finite model outputs encountered during window evaluation.")
            pred_norm = y_pred.float()
            target_norm = batch.y_future.float()
            batch_sq, batch_abs, batch_count = channel_error_sums(pred_norm, target_norm)
            norm_sq_sum += batch_sq.to(dtype=torch.float64)
            norm_abs_sum += batch_abs.to(dtype=torch.float64)
            count += batch_count.to(dtype=torch.float64)
            pred_phys = self.normalizer.inverse_state(pred_norm)
            target_phys = self.normalizer.inverse_state(target_norm)
            batch_sq, batch_abs, _ = channel_error_sums(pred_phys, target_phys)
            phys_sq_sum += batch_sq.to(dtype=torch.float64)
            phys_abs_sum += batch_abs.to(dtype=torch.float64)

        self._all_reduce(norm_sq_sum)
        self._all_reduce(norm_abs_sum)
        self._all_reduce(phys_sq_sum)
        self._all_reduce(phys_abs_sum)
        self._all_reduce(count)
        metrics = summarize_channel_metrics(phys_sq_sum, phys_abs_sum, count)
        metrics.update(summarize_channel_metrics(norm_sq_sum, norm_abs_sum, count, prefix="norm_"))
        return metrics

    @torch.no_grad()
    def evaluate_full_rollout(self, dataset, start_t: int | None = None) -> dict[str, float | list[float]]:
        """Run whole-trajectory rollout evaluation using known future forcing."""

        self.model.eval()
        rollout_start_t = dataset.history_len - 1 if start_t is None else int(start_t)
        min_steps = min(int(info["length"]) for info in dataset.scenario_infos)
        if rollout_start_t < dataset.history_len - 1:
            raise ValueError(
                f"Full-rollout start_t={rollout_start_t} must be at least history_len - 1 "
                f"({dataset.history_len - 1})."
            )
        if rollout_start_t >= min_steps - 1:
            raise ValueError(
                f"Full-rollout start_t={rollout_start_t} must leave at least one future step; "
                f"the shortest trajectory has {min_steps} steps."
            )

        reference = dataset.get_rollout_data(0, start_t=rollout_start_t)
        state_dim = int(reference.y_future.shape[-1])
        max_horizon = max(int(info["length"]) - rollout_start_t - 1 for info in dataset.scenario_infos)

        norm_whole_sq = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        norm_whole_abs = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        whole_sq = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        whole_abs = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        whole_count = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        norm_final_sq = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        norm_final_abs = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        final_sq = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        final_abs = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        final_count = torch.zeros(state_dim, device=self.device, dtype=torch.float64)
        norm_horizon_sq = torch.zeros(max_horizon, device=self.device, dtype=torch.float64)
        horizon_sq = torch.zeros(max_horizon, device=self.device, dtype=torch.float64)
        horizon_count = torch.zeros(max_horizon, device=self.device, dtype=torch.float64)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        for scenario_index in range(rank, len(dataset.scenario_infos), world_size):
            sample = dataset.get_rollout_data(scenario_index, start_t=rollout_start_t).to(self.device)
            with self._autocast():
                y_pred = self.model(sample)
            if not torch.isfinite(y_pred).all():
                raise FloatingPointError("Non-finite model outputs encountered during full-rollout evaluation.")

            pred_norm = y_pred.float()
            target_norm = sample.y_future.float()
            batch_sq, batch_abs, batch_count = channel_error_sums(pred_norm, target_norm)
            norm_whole_sq += batch_sq.to(dtype=torch.float64)
            norm_whole_abs += batch_abs.to(dtype=torch.float64)
            whole_count += batch_count.to(dtype=torch.float64)
            pred_phys = self.normalizer.inverse_state(pred_norm)
            target_phys = self.normalizer.inverse_state(target_norm)
            batch_sq, batch_abs, batch_count = channel_error_sums(pred_phys, target_phys)
            whole_sq += batch_sq.to(dtype=torch.float64)
            whole_abs += batch_abs.to(dtype=torch.float64)

            norm_final_pred = pred_norm[:, -1:, :]
            norm_final_target = target_norm[:, -1:, :]
            batch_sq, batch_abs, batch_count = channel_error_sums(norm_final_pred, norm_final_target)
            norm_final_sq += batch_sq.to(dtype=torch.float64)
            norm_final_abs += batch_abs.to(dtype=torch.float64)
            final_count += batch_count.to(dtype=torch.float64)
            final_pred = pred_phys[:, -1:, :]
            final_target = target_phys[:, -1:, :]
            batch_sq, batch_abs, batch_count = channel_error_sums(final_pred, final_target)
            final_sq += batch_sq.to(dtype=torch.float64)
            final_abs += batch_abs.to(dtype=torch.float64)

            curve_sq, curve_count = overall_horizon_sums(pred_norm, target_norm)
            norm_horizon_sq[: curve_sq.shape[0]] += curve_sq.to(dtype=torch.float64)
            horizon_count[: curve_count.shape[0]] += curve_count.to(dtype=torch.float64)
            curve_sq, _ = overall_horizon_sums(pred_phys, target_phys)
            horizon_sq[: curve_sq.shape[0]] += curve_sq.to(dtype=torch.float64)

        for tensor in (
            norm_whole_sq,
            norm_whole_abs,
            whole_sq,
            whole_abs,
            whole_count,
            norm_final_sq,
            norm_final_abs,
            final_sq,
            final_abs,
            final_count,
            norm_horizon_sq,
            horizon_sq,
            horizon_count,
        ):
            self._all_reduce(tensor)

        metrics = summarize_channel_metrics(whole_sq, whole_abs, whole_count, prefix="whole_rollout_")
        metrics.update(summarize_channel_metrics(norm_whole_sq, norm_whole_abs, whole_count, prefix="whole_rollout_norm_"))
        metrics.update(summarize_channel_metrics(final_sq, final_abs, final_count, prefix="final_step_"))
        metrics.update(summarize_channel_metrics(norm_final_sq, norm_final_abs, final_count, prefix="final_step_norm_"))
        metrics["horizon_norm_rmse_curve"] = summarize_horizon_curve(norm_horizon_sq, horizon_count)
        metrics["horizon_rmse_curve"] = summarize_horizon_curve(horizon_sq, horizon_count)
        return metrics

    @staticmethod
    def _all_reduce(tensor: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
