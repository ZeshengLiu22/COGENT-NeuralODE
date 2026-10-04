"""Rollout-to-end evaluation in normalized and physical units."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.normalization import FeatureNormalizer
from training.metrics import channel_error_sums, overall_horizon_sums, summarize_channel_metrics, summarize_horizon_curve
from training.metrics import add_issm_metrics, issm_speed_error_sums


class Evaluator:
    """Evaluate full trajectories in normalized and physical units."""

    def __init__(self, model: torch.nn.Module, normalizer: FeatureNormalizer, device: torch.device, amp_mode: str = "none") -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.amp_mode = amp_mode
        if amp_mode not in ("none", "bf16", "fp16"):
            raise ValueError(f"Unknown evaluation amp_mode: {amp_mode}")

    def _predict(self, batch):
        # Non-padding shards can have unequal batch counts. Bypass DDP forward
        # buffer broadcasts; evaluation weights were synchronized by training.
        model = self.model.module if isinstance(self.model, DDP) else self.model
        return model(batch)

    def _autocast(self):
        if self.device.type != "cuda" or self.amp_mode == "none":
            return nullcontext()
        dtype = torch.bfloat16 if self.amp_mode == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.no_grad()
    def evaluate_full_rollout(self, dataset, *, known_steps: int) -> dict[str, float | list[float]]:
        """Run whole-trajectory rollout evaluation using known future forcing."""

        self.model.eval()
        known_steps = int(known_steps)
        if not dataset.scenario_infos:
            raise ValueError("Rollout evaluation requires at least one scenario.")
        if known_steps < dataset.history_len:
            raise ValueError(
                f"known_steps={known_steps} must be >= history_len={dataset.history_len}."
            )
        min_steps = min(int(info["length"]) for info in dataset.scenario_infos)
        if known_steps >= min_steps:
            raise ValueError(
                f"known_steps={known_steps} must be < trajectory length ({min_steps})."
            )
        rollout_start_t = known_steps - 1

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
        is_issm = getattr(dataset, "dataset_name", None) == "issm"
        speed_sums = torch.zeros(2, device=self.device, dtype=torch.float64)
        final_speed_sums = torch.zeros_like(speed_sums)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        for scenario_index in range(rank, len(dataset.scenario_infos), world_size):
            sample = dataset.get_rollout_data(scenario_index, start_t=rollout_start_t).to(self.device)
            with self._autocast():
                y_pred = self._predict(sample)
            if not torch.isfinite(y_pred).all():
                raise FloatingPointError("Non-finite model outputs encountered during full-rollout evaluation.")

            pred_norm = y_pred.float()
            target_norm = sample.y_future.float()
            batch_sq, batch_abs, batch_count = channel_error_sums(pred_norm.double(), target_norm.double())
            norm_whole_sq += batch_sq.to(dtype=torch.float64)
            norm_whole_abs += batch_abs.to(dtype=torch.float64)
            whole_count += batch_count.to(dtype=torch.float64)
            pred_phys = self.normalizer.inverse_state(pred_norm)
            target_phys = self.normalizer.inverse_state(target_norm)
            if is_issm:
                speed_sums += issm_speed_error_sums(pred_phys, target_phys)
                final_speed_sums += issm_speed_error_sums(pred_phys[:, -1:], target_phys[:, -1:])
            batch_sq, batch_abs, batch_count = channel_error_sums(pred_phys.double(), target_phys.double())
            whole_sq += batch_sq.to(dtype=torch.float64)
            whole_abs += batch_abs.to(dtype=torch.float64)

            norm_final_pred = pred_norm[:, -1:, :]
            norm_final_target = target_norm[:, -1:, :]
            batch_sq, batch_abs, batch_count = channel_error_sums(norm_final_pred.double(), norm_final_target.double())
            norm_final_sq += batch_sq.to(dtype=torch.float64)
            norm_final_abs += batch_abs.to(dtype=torch.float64)
            final_count += batch_count.to(dtype=torch.float64)
            final_pred = pred_phys[:, -1:, :]
            final_target = target_phys[:, -1:, :]
            batch_sq, batch_abs, batch_count = channel_error_sums(final_pred.double(), final_target.double())
            final_sq += batch_sq.to(dtype=torch.float64)
            final_abs += batch_abs.to(dtype=torch.float64)

            curve_sq, curve_count = overall_horizon_sums(pred_norm.double(), target_norm.double())
            norm_horizon_sq[: curve_sq.shape[0]] += curve_sq.to(dtype=torch.float64)
            horizon_count[: curve_count.shape[0]] += curve_count.to(dtype=torch.float64)
            curve_sq, _ = overall_horizon_sums(pred_phys.double(), target_phys.double())
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
        metrics["horizon_lead_steps"] = list(range(1, max_horizon + 1))
        metrics["horizon_norm_rmse_curve"] = summarize_horizon_curve(norm_horizon_sq, horizon_count)
        metrics["horizon_rmse_curve"] = summarize_horizon_curve(horizon_sq, horizon_count)
        if is_issm:
            self._all_reduce(speed_sums)
            self._all_reduce(final_speed_sums)
            add_issm_metrics(metrics, speed_sums, prefix="whole_rollout_")
            add_issm_metrics(metrics, final_speed_sums, prefix="final_step_")
        return metrics

    @staticmethod
    def _all_reduce(tensor: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
