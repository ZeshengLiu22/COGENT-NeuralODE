"""Regression checks for accumulation, reporting, and evaluation precision."""
from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch, Data

from training.evaluator import Evaluator
from training.losses import compute_temporal_consistency
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

    def to_dict(self):
        return {"kind": "identity"}


class SlopeModel(ScalarModel):
    """A tiny trajectory model whose temporal loss has a nonzero gradient."""

    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, data):
        self.forward_calls += 1
        time = data.t_future[0].view(1, -1, 1)
        return (self.weight * time).expand_as(data.y_future)


class BatchLoader(list):
    sampler = None


class DropoutSlopeModel(SlopeModel):
    def __init__(self):
        super().__init__()
        self.dropout_masks = []

    def forward(self, data):
        mask = torch.nn.functional.dropout(torch.ones_like(data.y_future), p=0.5, training=self.training)
        self.dropout_masks.append(mask.detach().clone())
        return (super().forward(data) + 0.25) * mask


class FoundationTrainingTest(unittest.TestCase):
    def make_trainer(self, root, training_amp="bf16", evaluation=None, *,
                     training=None, future_len=1, model=None, node_counts=(1, 1, 1), seed=42):
        config = {
            "seed": seed,
            "dataset": {"data_dir": str(root)},
            "training": {"epochs": 1, "lr": 0.0, "grad_accum_steps": 2,
                         "train_horizon_mode": "uniform_random", "train_horizon_min": future_len,
                         "train_horizon_curriculum": {"enabled": False}},
            "evaluation": evaluation or {}, "amp": {"mode": training_amp},
        }
        config["training"].update(training or {})
        datasets = []
        for split in ("train", "val", "test"):
            path = root / f"{split}.npz"
            path.touch()
            datasets.append(SimpleNamespace(scenario_files=[path], future_len=future_len,
                                            set_epoch=lambda epoch: None))
        time = torch.arange(1, future_len + 1, dtype=torch.float32)
        batches = BatchLoader(
            Data(y_future=(target * time).view(1, -1, 1).repeat(nodes, 1, 1),
                 t_future=time.view(1, -1).clone())
            for target, nodes in zip((1.0, 3.0, 5.0), node_counts)
        )
        return Trainer(model if model is not None else ScalarModel(), batches, [], [],
                       *datasets, IdentityNormalizer(),
                       config, torch.device("cpu"), root / "output", logging.getLogger(__name__))

    def test_temporal_fit_logs_scaled_objectives_and_preserves_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(
                Path(tmp), future_len=3, model=SlopeModel(), node_counts=(1, 2, 3),
                training={"loss_scale_factor": 2.0, "temporal_consistency": {
                    "enabled": True, "mode": "adjacent_increment", "penalty": "mse",
                }},
            )
            # Exercise the actual fit/history/checkpoint path; evaluation remains
            # independent of the training objective and contributes no forwards.
            with patch.object(trainer.evaluator, "evaluate_loader", return_value={
                "norm_rmse": 2.0, "rmse": 3.0,
            }), patch.object(trainer.evaluator, "evaluate_full_rollout", return_value={
                "whole_rollout_norm_rmse": 2.0, "whole_rollout_rmse": 3.0,
            }):
                summary = trainer.fit()

            history = json.loads((trainer.output_dir / "history.json").read_text())
            metrics = history[0]["train"]
            objective_keys = (
                "train_state_objective", "train_tc_raw", "train_tc_weighted",
                "train_total_objective", "train_tc_adjacent", "train_tc_random_pair",
                "train_tc_rate", "train_tc_curvature",
            )
            for key in objective_keys:
                self.assertIn(key, metrics)
                self.assertTrue(np.isfinite(metrics[key]), key)
            self.assertEqual(trainer.model.forward_calls, len(trainer.train_loader))
            self.assertEqual(trainer.tc_weight, 1.0)
            self.assertEqual(summary["best_metric_name"], "whole_rollout_norm_rmse")
            self.assertEqual(summary["best_metric"], 2.0)

            # Optimization diagnostics average microbatches equally, whereas
            # prediction metrics retain their existing per-element weighting.
            expected_state = 4.0 * np.mean([1.0, 9.0, 25.0]) * np.mean([1.0, 4.0, 9.0])
            expected_tc = 4.0 * np.mean([1.0, 9.0, 25.0])
            expected_norm_mse = np.average([1.0, 9.0, 25.0], weights=[1, 2, 3]) * (14.0 / 3.0)
            self.assertAlmostEqual(metrics["train_state_objective"], expected_state, places=4)
            self.assertAlmostEqual(metrics["train_tc_raw"], expected_tc, places=4)
            self.assertEqual(metrics["train_tc_adjacent"], metrics["train_tc_raw"])
            self.assertEqual(metrics["train_tc_weighted"], metrics["train_tc_raw"])
            self.assertAlmostEqual(
                metrics["train_total_objective"],
                metrics["train_state_objective"] + metrics["train_tc_weighted"], places=4,
            )
            self.assertEqual(metrics["train_loss"], metrics["train_total_objective"])
            self.assertNotEqual(metrics["train_loss"], metrics["train_norm_mse"])
            self.assertAlmostEqual(metrics["train_norm_mse"], expected_norm_mse)
            self.assertAlmostEqual(metrics["train_phys_rmse"], np.sqrt(expected_norm_mse))
            for key in ("train_tc_random_pair", "train_tc_rate", "train_tc_curvature"):
                self.assertEqual(metrics[key], 0.0)

    def test_temporal_gradient_accumulation_includes_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(
                Path(tmp), future_len=3, model=SlopeModel(),
                training={"loss_scale_factor": 2.0, "temporal_consistency": {
                    "enabled": True, "mode": "adjacent_increment", "weight": 0.5,
                }},
            )
            gradients = []
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
            original_step = trainer.optimizer.step

            def record_step(*args, **kwargs):
                gradients.append(float(trainer.model.weight.grad))
                return original_step(*args, **kwargs)

            with patch.object(trainer.optimizer, "step", side_effect=record_step):
                trainer.train_epoch(1)
            # For slope w and target slope s: L = 4*(w-s)^2*(14/3 + 0.5).
            # First accumulation group averages s=1,3; the final group is s=5.
            torch.testing.assert_close(
                torch.tensor(gradients), torch.tensor([-248.0 / 3.0, -620.0 / 3.0]),
            )
            self.assertEqual(trainer.model.forward_calls, 3)

    def test_disabled_temporal_matches_baseline_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = self.make_trainer(
                Path(tmp), future_len=3, model=SlopeModel(),
                training={"lr": 0.01, "loss_scale_factor": 2.0},
            )
            disabled = self.make_trainer(
                Path(tmp), future_len=3, model=SlopeModel(),
                training={"lr": 0.01, "loss_scale_factor": 2.0,
                          "temporal_consistency": {"enabled": False,
                                                   "mode": "random_pair_increment",
                                                   "weight": 10.0}},
            )
            rng_before = torch.random.get_rng_state().clone()
            baseline_metrics = baseline.train_epoch(1)
            baseline_rng_after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng_before)
            disabled_metrics = disabled.train_epoch(1)
            torch.testing.assert_close(torch.random.get_rng_state(), baseline_rng_after)
            torch.testing.assert_close(baseline.model.weight, disabled.model.weight, rtol=0, atol=0)
            self.assertEqual(baseline_metrics, disabled_metrics)
            self.assertEqual(disabled_metrics["train_tc_raw"], 0.0)
            self.assertEqual(disabled_metrics["train_tc_weighted"], 0.0)
            self.assertEqual(disabled_metrics["train_total_objective"],
                             disabled_metrics["train_state_objective"])
            self.assertEqual(disabled_metrics["train_loss"], disabled_metrics["train_state_objective"])
            self.assertIsNone(disabled.tc_generator)

    def test_random_pair_uses_graph_membership_and_preserves_dropout_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = self.make_trainer(Path(tmp), future_len=4, model=DropoutSlopeModel())
            temporal = self.make_trainer(
                Path(tmp), future_len=4, model=DropoutSlopeModel(),
                training={"temporal_consistency": {
                    "enabled": True, "mode": "random_pair_increment",
                }},
            )
            for trainer in (baseline, temporal):
                trainer.train_loader = BatchLoader(
                    Batch.from_data_list([
                        Data(y_future=sample.y_future.repeat(nodes, 1, 1),
                             t_future=sample.t_future.clone(), num_nodes=nodes)
                        for nodes in (2, 3)
                    ]) for sample in trainer.train_loader
                )
            initial_rng = torch.random.get_rng_state().clone()
            initial_pair_rng = temporal.tc_generator.get_state().clone()
            baseline.train_epoch(1)
            baseline_rng = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(initial_rng)
            with patch("training.trainer.compute_temporal_consistency", wraps=compute_temporal_consistency) as compute:
                metrics = temporal.train_epoch(1)
            torch.testing.assert_close(torch.random.get_rng_state(), baseline_rng, rtol=0, atol=0)
            self.assertFalse(torch.equal(temporal.tc_generator.get_state(), initial_pair_rng))
            self.assertGreater(metrics["train_tc_random_pair"], 0.0)
            self.assertEqual(temporal.model.forward_calls, len(temporal.train_loader))
            for actual_mask, baseline_mask in zip(temporal.model.dropout_masks, baseline.model.dropout_masks):
                torch.testing.assert_close(actual_mask, baseline_mask, rtol=0, atol=0)
            for call in compute.call_args_list:
                torch.testing.assert_close(call.kwargs["node_batch"], torch.tensor([0, 0, 1, 1, 1]))
                self.assertIs(call.kwargs["generator"], temporal.tc_generator)

    def test_random_pair_generator_seed_is_rank_specific_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial_rng = torch.random.get_rng_state().clone()
            generators = []
            for rank in (0, 1, 0):
                with patch("training.trainer.get_rank", return_value=rank):
                    trainer = self.make_trainer(
                        Path(tmp), seed=123, training={"temporal_consistency": {
                            "enabled": True, "mode": "random_pair_increment",
                        }},
                    )
                self.assertEqual(trainer.tc_generator.initial_seed(), 123 + rank)
                generators.append(trainer.tc_generator.get_state())
            torch.testing.assert_close(generators[0], generators[2], rtol=0, atol=0)
            self.assertFalse(torch.equal(generators[0], generators[1]))
            torch.testing.assert_close(torch.random.get_rng_state(), initial_rng, rtol=0, atol=0)

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
