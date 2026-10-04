"""Fixed per-scenario exposure with natural anchors and variable rollout lengths."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from datasets.factory import build_dataset
from datasets.normalization import FeatureNormalizer
from datasets.window_utils import expand_windows, sample_training_series_for_epoch
from models import build_model
from test_foundation_datasets import _SyntheticDataset
from test_smoke import _base_config
from training.horizon_sampling import synchronized_horizon
from training.losses import rollout_mse
from training.trainer import _truncate_future_horizon
from utils.io import load_yaml


class TrainingSeriesSamplingTest(unittest.TestCase):
    def sample(self, *, total_steps, history, future, budget, seed=42):
        natural = expand_windows(0, total_steps, history, future, 1)
        rng = Mock(wraps=np.random.default_rng(seed))
        selected = sample_training_series_for_epoch(natural, 1, rng, budget)
        return natural, selected, rng

    def test_formal_protocol_budgets(self):
        for name, expected in (("issm", 60), ("anuga", 9)):
            config = load_yaml(PROJECT_ROOT / f"configs/protocols/{name}/main.yaml")
            self.assertEqual(config["dataset"]["train_series_per_scenario_per_epoch"], expected)

    def test_issm_short_k_selects_exactly_60_distinct_natural_anchors(self):
        natural, selected, rng = self.sample(total_steps=240, history=4, future=30, budget=60)
        self.assertEqual(len(natural), 207)
        self.assertEqual(len(selected), 60)
        self.assertEqual(len(set(selected)), 60)
        self.assertTrue(set(selected).issubset(natural))
        rng.choice.assert_called_once_with(207, size=60, replace=False)

    def test_anuga_includes_all_unique_anchors_before_replacement_extras(self):
        for history, expected in ((1, list(range(9))), (4, list(range(3, 9))), (8, [7, 8])):
            with self.subTest(history=history):
                natural, selected, rng = self.sample(total_steps=73, history=history, future=64, budget=9)
                self.assertEqual([series.t_end for series in natural], expected)
                counts = Counter(series.t_end for series in selected)
                self.assertEqual(set(counts), set(expected))
                self.assertEqual(sum(counts.values()), 9)
                self.assertEqual(sum(count - 1 for count in counts.values()), 9 - len(expected))
                if len(expected) == 9:
                    self.assertEqual(set(counts.values()), {1})
                    rng.choice.assert_called_once_with(9, size=9, replace=False)
                else:
                    rng.choice.assert_called_once_with(len(expected), size=9 - len(expected), replace=True)
                    rng.shuffle.assert_called_once()

    def test_each_scenario_keeps_its_own_budget_and_valid_anchors(self):
        natural = expand_windows(0, 73, 4, 64, 1) + expand_windows(1, 120, 4, 64, 1)
        selected = sample_training_series_for_epoch(natural, 2, np.random.default_rng(6), 9)
        self.assertEqual(Counter(series.scenario_index for series in selected), {0: 9, 1: 9})
        self.assertEqual({series.t_end for series in selected if series.scenario_index == 0}, set(range(3, 9)))
        second = [series for series in selected if series.scenario_index == 1]
        self.assertEqual(len(set(second)), 9)
        self.assertTrue(set(selected).issubset(natural))

    def test_epoch_seed_is_deterministic_and_can_change_anchor_subset(self):
        dataset = _SyntheticDataset([Path("240")], 4, 30, "train",
                                    train_series_per_scenario_per_epoch=60, seed=19)
        dataset.set_epoch(4)
        first = list(dataset.active_windows)
        dataset.set_epoch(4)
        self.assertEqual(first, dataset.active_windows)
        duplicate = _SyntheticDataset([Path("240")], 4, 30, "train",
                                      train_series_per_scenario_per_epoch=60, seed=19)
        duplicate.set_epoch(4)
        self.assertEqual(first, duplicate.active_windows)
        dataset.set_epoch(5)
        self.assertNotEqual(set(first), set(dataset.active_windows))
        self.assertEqual(len(dataset), 60)

    def test_empty_scenario_and_invalid_budgets_fail_instead_of_changing_weighting(self):
        for invalid in (0, -1, 1.5, True, "9"):
            with self.subTest(budget=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                self.sample(total_steps=73, history=1, future=64, budget=invalid)
        with self.assertRaisesRegex(ValueError, "no natural anchors"):
            self.sample(total_steps=73, history=10, future=64, budget=9)

    def test_formal_sweeps_keep_dataset_lengths_and_four_rank_shards_constant(self):
        for name, total_steps, scenarios, budget, futures, expected in (
            ("issm", 240, 28, 60, [30, 45, 60, 75, 90, 120, 150, 180], 1680),
            ("anuga", 73, 12, 9, [8, 16, 24, 32, 40, 48, 56, 64], 108),
        ):
            files = [Path(f"scenario_{index}/{total_steps}") for index in range(scenarios)]
            for history in range(1, 9):
                for future in futures:
                    with self.subTest(dataset=name, history=history, future=future):
                        config = {"seed": 42, "dataset": {"history_len": history, "future_len": future,
                                  "train_series_per_scenario_per_epoch": budget}}
                        with patch.dict("datasets.factory.DATASET_REGISTRY", {"synthetic": _SyntheticDataset}):
                            dataset = build_dataset("synthetic", files, "train", config)
                        self.assertEqual(len(dataset), expected)
                        self.assertEqual(Counter(series.scenario_index for series in dataset.active_windows),
                                         {index: budget for index in range(scenarios)})
                        shards = [list(DistributedSampler(dataset, num_replicas=4, rank=rank, shuffle=True))
                                  for rank in range(4)]
                        self.assertEqual([len(shard) for shard in shards], [expected // 4] * 4)
                        self.assertEqual(sorted(index for shard in shards for index in shard), list(range(expected)))


class RolloutDatasetConstructionTest(unittest.TestCase):
    def test_rollout_only_construction_still_validates_temporal_dimensions(self):
        for history, future, stride in ((0, 64, 1), (4, 0, 1), (4, 64, 0)):
            with self.subTest(history=history, future=future, stride=stride):
                with self.assertRaisesRegex(ValueError, "must be >= 1"):
                    _SyntheticDataset([Path("73")], history, future, "test", stride=stride)

    def test_rollout_splits_skip_anchor_expansion_and_preserve_trajectory_access(self):
        for split in ("val", "test", "eval", "train"):
            with self.subTest(split=split):
                config = {"seed": 42, "dataset": {"history_len": 4, "future_len": 64,
                          "train_series_per_scenario_per_epoch": 9}}
                kwargs = {"build_training_series": False} if split == "train" else {}
                with patch.dict("datasets.factory.DATASET_REGISTRY", {"synthetic": _SyntheticDataset}), \
                     patch("datasets.base_dataset.expand_windows", side_effect=AssertionError("training anchors built")):
                    dataset = build_dataset("synthetic", [Path("73")], split, config, **kwargs)
                    dataset.set_epoch(2)
                self.assertEqual(dataset.all_windows, [])
                self.assertEqual(dataset.active_windows, [])
                self.assertEqual(len(dataset), 0)
                self.assertEqual(dataset.scenario_infos[0]["length"], 73)
                self.assertEqual(dataset.scenario_infos[0]["scenario_id"], "73")
                trajectories = dataset.iter_trajectories()
                dataset.normalizer = FeatureNormalizer.fit_from_trajectories(trajectories)
                sample = dataset.get_rollout_data(0, start_t=7)
                self.assertEqual(sample.history_idx[0].tolist(), [4, 5, 6, 7])
                self.assertEqual(sample.future_idx[0].tolist(), list(range(8, 73)))
                self.assertEqual(sample.y_future.shape[1], 65)
                torch.testing.assert_close(dataset.normalizer.inverse_state(sample.y_future)[0, :, 0],
                                           torch.arange(8, 73).float())
                np.testing.assert_array_equal(sample.edge_index.numpy(), trajectories[0].edge_index)
                self.assertEqual(sample.scenario_id.item(), dataset.scenario_id_to_code["73"])
                self.assertEqual(sample.sim_id.item(), dataset.sim_id_to_code["73"])


class TrainingSeriesHorizonRegressionTest(unittest.TestCase):
    def test_repeated_anchor_keeps_independent_horizon_and_complete_future_predictions(self):
        dataset = _SyntheticDataset([Path("73")], 8, 64, "train",
                                    train_series_per_scenario_per_epoch=9)
        selected = list(dataset.active_windows)
        # Series sampling uses its own NumPy generator and cannot advance the
        # Torch generator used by the synchronized effective-horizon sampler.
        torch.manual_seed(43)
        rng_before = torch.random.get_rng_state().clone()
        dataset.set_epoch(0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertEqual(dataset.active_windows, selected)
        horizons = [synchronized_horizon("uniform_random", 8, 64, torch.device("cpu")) for _ in range(12)]
        self.assertGreater(len(set(horizons)), 1)
        duplicate = next(series for series, count in Counter(selected).items() if count > 1)
        index = selected.index(duplicate)
        config = _base_config()
        config["model"]["relative_time_scale"] = 65.0
        model = build_model(config, num_static=4, num_force=2, num_state=3).eval()
        for k_eff in (min(horizons), max(horizons)):
            maximum = dataset[index]
            self.assertEqual(maximum.y_future.shape[1], 64)
            batch = _truncate_future_horizon(maximum, k_eff)
            expected = torch.arange(duplicate.t_end + 1, duplicate.t_end + 1 + k_eff)
            self.assertEqual(batch.force_future.shape[1], k_eff)
            torch.testing.assert_close(batch.future_idx[0], expected)
            torch.testing.assert_close(batch.y_future[0, :, 0], expected.float())
            torch.testing.assert_close(batch.t_future[0], torch.arange(1, k_eff + 1).float())
            with torch.no_grad():
                prediction = model(batch)
            self.assertEqual(prediction.shape, (3, k_eff, 3))
            torch.testing.assert_close(rollout_mse(prediction, batch.y_future),
                                       ((prediction - batch.y_future) ** 2).mean())
            self.assertEqual(len(dataset), 9)
        self.assertEqual(dataset.active_windows, selected)


if __name__ == "__main__":
    unittest.main()
