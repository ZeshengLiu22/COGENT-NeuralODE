"""Regression checks for accumulation, reporting, and evaluation precision."""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data

from training.evaluator import Evaluator
from training.trainer import Trainer
from utils.eval_artifacts import _metric_rows_and_values


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        return self.weight.expand_as(data.y_future)


class IdentityNormalizer:
    def inverse_state(self, x):
        return x


class FoundationTrainingTest(unittest.TestCase):
    def make_trainer(self, root, training_amp="bf16", evaluation=None):
        config = {
            "dataset": {"data_dir": str(root)},
            "training": {"epochs": 1, "lr": 0.0, "grad_accum_steps": 2,
                         "train_horizon_mode": "fixed", "train_horizon_min": 1,
                         "train_horizon_curriculum": {"enabled": False}},
            "evaluation": evaluation or {}, "amp": {"mode": training_amp},
        }
        datasets = []
        for split in ("train", "val", "test"):
            path = root / f"{split}.npz"
            path.touch()
            datasets.append(SimpleNamespace(scenario_files=[path], future_len=1))
        batches = [Data(y_future=torch.full((1, 1, 1), target),
                        t_future=torch.tensor([[1.0]])) for target in (1.0, 3.0, 5.0)]
        return Trainer(ScalarModel(), batches, [], [], *datasets, IdentityNormalizer(),
                       config, torch.device("cpu"), root / "output", logging.getLogger(__name__))

    def test_accumulation_tail_uses_actual_group_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            gradients = []
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
            original_step = trainer.optimizer.step

            def record_step(*args, **kwargs):
                gradients.append(float(trainer.model.weight.grad))
                return original_step(*args, **kwargs)

            with patch.object(trainer.optimizer, "step", side_effect=record_step), patch(
                "training.trainer.synchronized_horizon", return_value=1
            ) as horizon:
                trainer.train_epoch(1)
            self.assertEqual(gradients, [-4.0, -10.0])
            self.assertEqual(horizon.call_count, 2)

    def test_evaluation_fp32_independent_of_bf16_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            self.assertEqual(trainer.amp_mode, "bf16")
            self.assertEqual(trainer.evaluator.amp_mode, "none")
            self.assertEqual(trainer.checkpoint_metric, "whole_rollout_norm_rmse")
            self.assertEqual(trainer.checkpoint_metric_scale, "normalized")
            explicit = self.make_trainer(Path(tmp), evaluation={"amp_mode": "bf16"})
            self.assertEqual(explicit.evaluator.amp_mode, "bf16")

    def test_evaluation_bypasses_ddp_forward_collectives(self):
        # Uneven non-padding shards must not enter per-forward DDP collectives.
        wrapper = DDP.__new__(DDP)
        torch.nn.Module.__init__(wrapper)
        wrapper.module = ScalarModel()
        evaluator = Evaluator(wrapper, IdentityNormalizer(), torch.device("cpu"))
        sample = Data(y_future=torch.ones(1, 1, 1))
        with patch.object(wrapper, "forward", side_effect=AssertionError("DDP broadcast")):
            torch.testing.assert_close(evaluator._predict(sample), torch.zeros(1, 1, 1))

    def test_issm_named_metrics_use_derived_physical_speed(self):
        prediction = torch.tensor([[[3.0, 4.0, 15.0], [-3.0, -4.0, 15.0]]])
        target = torch.tensor([[[0.0, 0.0, 10.0], [3.0, 4.0, 10.0]]])
        sample = Data(y_future=target)

        class PhysicalModel(torch.nn.Module):
            def forward(self, _):
                return prediction

        class Dataset:
            dataset_name = "issm"
            history_len = 1
            scenario_infos = [{"length": 3}]

            def __getitem__(self, _):
                return sample

            def get_rollout_data(self, *args, **kwargs):
                return sample

        class Loader(list):
            dataset = Dataset()

        evaluator = Evaluator(PhysicalModel(), IdentityNormalizer(), torch.device("cpu"))
        window = evaluator.evaluate_loader(Loader([sample]))
        rollout = evaluator.evaluate_full_rollout(Dataset())
        self.assertAlmostEqual(window["speed_rmse_m_per_yr"], np.sqrt(12.5))
        self.assertEqual(window["thickness_rmse_m"], 5.0)
        self.assertAlmostEqual(rollout["whole_rollout_speed_rmse_m_per_yr"], np.sqrt(12.5))
        self.assertEqual(rollout["final_step_speed_rmse_m_per_yr"], 0.0)
        self.assertEqual(len(rollout["horizon_rmse_curve"]), 2)
        pred = prediction.numpy().reshape(-1, 3)
        true = target.numpy().reshape(-1, 3)
        standalone, _ = _metric_rows_and_values(
            scope="window", pred_phys=pred, target_phys=true, pred_norm=pred,
            target_norm=true, channel_names=["vx", "vy", "thickness"], metric_prefix="",
        )
        self.assertAlmostEqual(standalone["speed_rmse_m_per_yr"], window["speed_rmse_m_per_yr"])
        self.assertEqual(standalone["thickness_rmse_m"], window["thickness_rmse_m"])


if __name__ == "__main__":
    unittest.main()
