"""Training loop with DDP-safe horizon sampling, AMP, and checkpointing."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from training.evaluator import Evaluator
from training.horizon_sampling import curriculum_horizon_max, synchronized_horizon
from training.losses import rollout_mse
from utils.io import save_checkpoint, save_json
from utils.logging_utils import rank0_log
from utils.seed import get_rank


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _truncate_future_horizon(batch: Any, k_eff: int) -> Any:
    """Trim future-aligned batch tensors so training rolls out only ``k_eff`` steps."""

    if k_eff < 1:
        raise ValueError(f"k_eff must be >= 1, got {k_eff}")

    node_future_fields = ("force_future", "y_future", "target_mask")
    graph_future_fields = ("t_future", "future_idx", "future_time")

    for name in node_future_fields:
        value = getattr(batch, name, None)
        if value is None:
            continue
        if value.dim() < 2:
            raise ValueError(f"{name} must have a time dimension, got shape {tuple(value.shape)}")
        if value.shape[1] < k_eff:
            raise ValueError(f"{name} has only {value.shape[1]} future steps, cannot trim to k_eff={k_eff}")
        setattr(batch, name, value[:, :k_eff, ...].contiguous())

    for name in graph_future_fields:
        value = getattr(batch, name, None)
        if value is None:
            continue
        if value.dim() == 1:
            if value.shape[0] < k_eff:
                raise ValueError(f"{name} has only {value.shape[0]} future steps, cannot trim to k_eff={k_eff}")
            trimmed = value[:k_eff]
        else:
            if value.shape[1] < k_eff:
                raise ValueError(f"{name} has only {value.shape[1]} future steps, cannot trim to k_eff={k_eff}")
            trimmed = value[:, :k_eff, ...]
        setattr(batch, name, trimmed.contiguous())

    return batch


def _build_grad_scaler(enabled: bool):
    """Construct a GradScaler without triggering deprecation warnings across torch versions."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _clip_grad_norm_fp64(parameters: list[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    """Clip gradients using an fp64 total norm to avoid false overflow failures."""

    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    grad_norms: list[torch.Tensor] = []
    for param in parameters:
        grad = param.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            return torch.as_tensor(float("nan"), device=grad.device, dtype=torch.float64)
        grad_norms.append(torch.linalg.vector_norm(grad.detach().to(dtype=torch.float64), ord=2))

    if not grad_norms:
        return torch.zeros((), dtype=torch.float64)

    total_norm = torch.linalg.vector_norm(torch.stack(grad_norms), ord=2)
    if torch.isfinite(total_norm):
        clip_coef = float(max_norm) / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in parameters:
                if param.grad is not None:
                    param.grad.detach().mul_(clip_coef.to(device=param.grad.device, dtype=param.grad.dtype))
    return total_norm


class Trainer:
    """End-to-end trainer for the continuous graph emulator baselines."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
        normalizer,
        config: dict[str, Any],
        device: torch.device,
        output_dir: Path,
        logger,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.normalizer = normalizer
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.logger = logger

        training_cfg = config["training"]
        self.epochs = int(training_cfg["epochs"])
        self.grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 0.0))
        self.grad_clip_norm_dtype = str(training_cfg.get("grad_clip_norm_dtype", "fp32")).lower()
        if self.grad_clip_norm_dtype not in {"fp32", "fp64"}:
            raise ValueError(
                "training.grad_clip_norm_dtype must be 'fp32' or 'fp64', "
                f"got {self.grad_clip_norm_dtype!r}"
            )
        self.loss_scale_factor = float(training_cfg.get("loss_scale_factor", training_cfg.get("scale_factor", 1.0)))
        self.log_every = int(training_cfg.get("log_every", 10))
        self.val_every = int(training_cfg.get("val_every", 1))
        evaluation_cfg = config["evaluation"]
        self.full_rollout_on_val = bool(evaluation_cfg.get("full_rollout_on_val", True))
        self.full_rollout_known_steps = evaluation_cfg.get("full_rollout_known_steps", None)
        self.checkpoint_metric = str(evaluation_cfg.get("checkpoint_metric", "whole_rollout_rmse"))
        self.checkpoint_metric_scale = self._metric_scale(self.checkpoint_metric)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            _unwrap_model(self.model).parameters(),
            lr=float(training_cfg["lr"]),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        )
        scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
        if scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=float(training_cfg.get("min_lr", 0.0)))
        else:
            self.scheduler = None

        amp_mode = str(config["amp"]["mode"]).lower()
        self.amp_mode = amp_mode
        self.scaler = _build_grad_scaler(enabled=(self.device.type == "cuda" and amp_mode == "fp16"))
        self.evaluator = Evaluator(self.model, normalizer, device=device, amp_mode=amp_mode)
        self.best_metric = float("inf")
        self.best_epoch = -1

    def _target_train_horizon_max(self, training_cfg: dict[str, Any]) -> int:
        configured_horizon_max = training_cfg.get("train_horizon_max", None)
        if configured_horizon_max is None:
            horizon_max = int(self.train_dataset.future_len)
        else:
            horizon_max = int(configured_horizon_max)
        if horizon_max > int(self.train_dataset.future_len):
            raise ValueError(
                f"train_horizon_max must be <= dataset.future_len, got {horizon_max} > {self.train_dataset.future_len}"
            )
        return horizon_max

    def _curriculum_config(self, training_cfg: dict[str, Any]) -> dict[str, Any]:
        raw_cfg = training_cfg.get("train_horizon_curriculum", True)
        if isinstance(raw_cfg, dict):
            enabled = bool(raw_cfg.get("enabled", True))
            curriculum_epochs = int(raw_cfg.get("epochs", training_cfg.get("train_horizon_curriculum_epochs", 120)))
            warmup_fractions = raw_cfg.get(
                "warmup_fractions",
                training_cfg.get("train_horizon_curriculum_warmup_fractions", [0.40, 0.55, 0.70, 0.85]),
            )
        else:
            enabled = bool(raw_cfg)
            curriculum_epochs = int(training_cfg.get("train_horizon_curriculum_epochs", 120))
            warmup_fractions = training_cfg.get("train_horizon_curriculum_warmup_fractions", [0.40, 0.55, 0.70, 0.85])
        return {
            "enabled": enabled,
            "epochs": curriculum_epochs,
            "warmup_fractions": warmup_fractions,
        }

    def _epoch_train_horizon_max(self, epoch: int, training_cfg: dict[str, Any], horizon_min: int, target_horizon_max: int) -> int:
        curriculum_cfg = self._curriculum_config(training_cfg)
        return curriculum_horizon_max(
            epoch=epoch,
            k_min=horizon_min,
            target_k_max=target_horizon_max,
            enabled=bool(curriculum_cfg["enabled"]),
            curriculum_epochs=int(curriculum_cfg["epochs"]),
            warmup_fractions=curriculum_cfg["warmup_fractions"],
        )

    def fit(self) -> dict[str, Any]:
        """Run training, validation, checkpointing, and final test evaluation."""

        history: list[dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            self.train_dataset.set_epoch(epoch)

            train_metrics = self.train_epoch(epoch)
            epoch_record: dict[str, Any] = {"epoch": epoch, "train": train_metrics}

            if epoch % self.val_every == 0:
                val_metrics = self.evaluator.evaluate_loader(self.val_loader)
                epoch_record["val_window"] = val_metrics
                if self.full_rollout_on_val:
                    val_rollout = self.evaluator.evaluate_full_rollout(
                        self.val_dataset,
                        start_t=self._full_rollout_start_t(self.val_dataset),
                    )
                    epoch_record["val_rollout"] = val_rollout
                    metric_value = float(val_rollout[self.checkpoint_metric])
                else:
                    metric_value = float(val_metrics[self.checkpoint_metric])

                if metric_value < self.best_metric:
                    self.best_metric = metric_value
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, metric_value)

            if self.scheduler is not None:
                self.scheduler.step()

            history.append(epoch_record)
            if get_rank() == 0:
                rank0_log(self.logger, self._format_epoch_log(epoch, epoch_record))
                save_json(self.output_dir / "history.json", history)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._load_best_checkpoint()
        test_window = self.evaluator.evaluate_loader(self.test_loader)
        test_rollout = self.evaluator.evaluate_full_rollout(
            self.test_dataset,
            start_t=self._full_rollout_start_t(self.test_dataset),
        )
        summary = {
            "best_epoch": self.best_epoch,
            "best_metric_name": self.checkpoint_metric,
            "best_metric_scale": self.checkpoint_metric_scale,
            "best_metric": self.best_metric,
            "full_rollout_known_steps": self.full_rollout_known_steps,
            "test_window": test_window,
            "test_rollout": test_rollout,
        }
        if get_rank() == 0:
            save_json(self.output_dir / "final_metrics.json", summary)
        return summary

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch with synchronized effective horizon sampling."""

        self.model.train()
        training_cfg = self.config["training"]
        horizon_mode = str(training_cfg.get("train_horizon_mode", "uniform_random"))
        horizon_min = int(training_cfg.get("train_horizon_min", 2))
        target_horizon_max = self._target_train_horizon_max(training_cfg)
        if horizon_min < 1:
            raise ValueError(f"train_horizon_min must be >= 1, got {horizon_min}")
        horizon_max = self._epoch_train_horizon_max(epoch, training_cfg, horizon_min, target_horizon_max)

        norm_sq_sum = torch.zeros(1, device=self.device, dtype=torch.float64)
        phys_sq_sum = torch.zeros(1, device=self.device, dtype=torch.float64)
        phys_abs_sum = torch.zeros(1, device=self.device, dtype=torch.float64)
        element_count = torch.zeros(1, device=self.device, dtype=torch.float64)
        self.optimizer.zero_grad(set_to_none=True)

        current_k_eff = None
        for step, batch in enumerate(self.train_loader, start=1):
            if (step - 1) % self.grad_accum_steps == 0:
                current_k_eff = synchronized_horizon(horizon_mode, horizon_min, horizon_max, self.device)
            if current_k_eff is None:
                raise RuntimeError("Effective training horizon was not sampled before model forward.")

            batch = batch.to(self.device)
            batch = _truncate_future_horizon(batch, current_k_eff)
            with self._autocast():
                y_pred = self.model(batch)
            y_pred_loss = y_pred.float()
            y_true_loss = batch.y_future.float()
            if self.loss_scale_factor != 1.0:
                y_pred_loss = y_pred_loss * self.loss_scale_factor
                y_true_loss = y_true_loss * self.loss_scale_factor
            loss = rollout_mse(y_pred_loss, y_true_loss)
            loss = loss / self.grad_accum_steps

            if not torch.isfinite(y_pred).all():
                raise FloatingPointError(
                    f"Non-finite model outputs at step={step}, k_eff={current_k_eff}, "
                    f"t_future_max={float(batch.t_future.max())}, device={self.device}."
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step={step}, k_eff={current_k_eff}, "
                    f"t_future_max={float(batch.t_future.max())}, device={self.device}."
                )

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step % self.grad_accum_steps == 0) or (step == len(self.train_loader))
            if should_step:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    self._check_and_clip_gradients(
                        step=step,
                        k_eff=current_k_eff,
                        t_future_max=float(batch.t_future.max()),
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self._check_and_clip_gradients(
                        step=step,
                        k_eff=current_k_eff,
                        t_future_max=float(batch.t_future.max()),
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            pred_norm = y_pred.detach().float()
            target_norm = batch.y_future.detach().float()
            norm_diff = pred_norm - target_norm
            norm_sq_sum += (norm_diff * norm_diff).sum().to(dtype=torch.float64)

            pred_phys = self.normalizer.inverse_state(pred_norm)
            target_phys = self.normalizer.inverse_state(target_norm)
            phys_diff = pred_phys - target_phys
            phys_sq_sum += (phys_diff * phys_diff).sum().to(dtype=torch.float64)
            phys_abs_sum += phys_diff.abs().sum().to(dtype=torch.float64)
            element_count += float(pred_norm.numel())

        self._all_reduce(norm_sq_sum)
        self._all_reduce(phys_sq_sum)
        self._all_reduce(phys_abs_sum)
        self._all_reduce(element_count)
        safe_count = torch.clamp(element_count, min=1.0)
        train_norm_mse = norm_sq_sum / safe_count
        return {
            "train_loss": float(train_norm_mse.item()),
            "train_norm_mse": float(train_norm_mse.item()),
            "train_norm_rmse": float(torch.sqrt(train_norm_mse).item()),
            "train_phys_rmse": float(torch.sqrt(phys_sq_sum / safe_count).item()),
            "train_phys_mae": float((phys_abs_sum / safe_count).item()),
            "train_horizon_min": float(horizon_min),
            "train_horizon_max": float(horizon_max),
            "train_horizon_target_max": float(target_horizon_max),
        }

    def _save_checkpoint(self, epoch: int, metric_value: float) -> None:
        if get_rank() != 0:
            return
        checkpoint = {
            "epoch": epoch,
            "metric_name": self.checkpoint_metric,
            "metric_scale": self.checkpoint_metric_scale,
            "metric_value": metric_value,
            "model_state": _unwrap_model(self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "normalizer": self.normalizer.to_dict(),
        }
        save_checkpoint(self.output_dir / "best.pt", checkpoint)

    def _load_best_checkpoint(self) -> None:
        checkpoint_path = self.output_dir / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        _unwrap_model(self.model).load_state_dict(checkpoint["model_state"])

    def _check_and_clip_gradients(self, *, step: int, k_eff: int, t_future_max: float) -> None:
        parameters = [param for param in _unwrap_model(self.model).parameters() if param.grad is not None]
        if not parameters:
            return

        if self.max_grad_norm > 0.0:
            if self.grad_clip_norm_dtype == "fp64":
                total_norm = _clip_grad_norm_fp64(parameters, self.max_grad_norm)
                if not torch.isfinite(total_norm):
                    raise FloatingPointError(
                        f"Non-finite gradients before optimizer step at step={step}, "
                        f"k_eff={k_eff}, t_future_max={t_future_max}, device={self.device}."
                    )
            else:
                try:
                    torch.nn.utils.clip_grad_norm_(
                        parameters,
                        self.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise FloatingPointError(
                        f"Non-finite gradients before optimizer step at step={step}, "
                        f"k_eff={k_eff}, t_future_max={t_future_max}, device={self.device}."
                    ) from exc
            return

        for param in parameters:
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError(
                    f"Non-finite gradients before optimizer step at step={step}, "
                    f"k_eff={k_eff}, t_future_max={t_future_max}, device={self.device}."
                )

    def _autocast(self):
        if self.device.type != "cuda" or self.amp_mode == "none":
            return nullcontext()
        dtype = torch.bfloat16 if self.amp_mode == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _format_epoch_log(self, epoch: int, epoch_record: dict[str, Any]) -> str:
        train_metrics = epoch_record["train"]
        parts = [
            f"Epoch {epoch:03d}",
            f"train_horizon_max={train_metrics['train_horizon_max']:.0f}/{train_metrics['train_horizon_target_max']:.0f}",
            f"train_norm_mse={train_metrics['train_norm_mse']:.6f}",
            f"train_phys_rmse={train_metrics['train_phys_rmse']:.6f}",
        ]

        val_window = epoch_record.get("val_window")
        if val_window is not None:
            parts.append(f"val_window_norm_rmse={val_window['norm_rmse']:.6f}")
            parts.append(f"val_window_phys_rmse={val_window['rmse']:.6f}")

        val_rollout = epoch_record.get("val_rollout")
        if val_rollout is not None:
            parts.append(f"val_rollout_norm_rmse={val_rollout['whole_rollout_norm_rmse']:.6f}")
            parts.append(f"val_rollout_phys_rmse={val_rollout['whole_rollout_rmse']:.6f}")

        parts.append(f"best_{self.checkpoint_metric}={self.best_metric:.6f}")
        return " | ".join(parts)

    @staticmethod
    def _metric_scale(metric_name: str) -> str:
        if metric_name.startswith("norm_") or "_norm_" in metric_name:
            return "normalized"
        return "physical"

    def _full_rollout_start_t(self, dataset) -> int | None:
        if self.full_rollout_known_steps is None:
            return None
        known_steps = int(self.full_rollout_known_steps)
        if known_steps < dataset.history_len:
            raise ValueError(
                f"evaluation.full_rollout_known_steps must be >= dataset.history_len "
                f"({dataset.history_len}), got {known_steps}."
            )
        return known_steps - 1

    @staticmethod
    def _all_reduce(tensor: torch.Tensor) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
