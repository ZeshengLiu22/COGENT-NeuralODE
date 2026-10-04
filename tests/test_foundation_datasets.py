"""Regression tests for the foundation input, normalization, and window repairs."""

from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.base_dataset import BaseTemporalGraphDataset, TrajectoryData
from datasets.factory import DistributedEvaluationSampler, build_dataset, build_loader
from datasets.issm_dataset import ISSMDataset
from datasets.normalization import FeatureNormalizer
from datasets.split_utils import issm_rate_modulo_split, parse_issm_rate_from_filename
from datasets.window_utils import enumerate_window_end_indices


def _issm_payload():
    t = np.arange(12, dtype=np.float32)[:, None]
    node = np.arange(3, dtype=np.float32)[None, :]
    fields = [
        np.array([0, 1000, 0], dtype=np.float32),
        np.array([0, 0, 1000], dtype=np.float32),
        np.array([[1, 2, 3]]),
        0.5 + t + node,
        3 + t + node,
        4 + t + node,
        np.zeros((12, 3), dtype=np.float32),
        100 + t + node,
        -200 + t + node,
        300 + t + node,
        np.broadcast_to(np.array([-10, 0, 20], dtype=np.float32), (12, 3)).copy(),
    ]
    return {"S": [[fields]]}


def _issm_dataset(payload):
    with patch("datasets.issm_dataset._load_container", return_value=payload):
        return ISSMDataset([Path("PIG_transient_r010.mat")], 2, 4, "train", cache_in_memory=True)


class _SyntheticDataset(BaseTemporalGraphDataset):
    def _load_trajectory(self, path):
        steps = int(path.stem)
        return TrajectoryData(
            x_static=np.zeros((3, 4), dtype=np.float32),
            force=np.zeros((steps, 3, 2), dtype=np.float32),
            state=np.zeros((steps, 3, 3), dtype=np.float32),
            edge_index=np.array([[0, 1], [1, 0]]),
            times=np.arange(steps, dtype=np.float32),
            scenario_id=str(path),
            sim_id=str(path),
        )


class ISSMProtocolTest(unittest.TestCase):
    def test_canonical_dimensions_initial_features_and_state(self):
        payload = _issm_payload()
        fields = payload["S"][0][0]
        trajectory = _issm_dataset(payload).iter_trajectories()[0]
        self.assertEqual(trajectory.x_static.shape, (3, 4))
        self.assertEqual(trajectory.force.shape, (12, 3, 2))
        self.assertEqual(trajectory.state.shape, (12, 3, 3))
        np.testing.assert_array_equal(trajectory.x_static[:, 0], fields[8][0])
        np.testing.assert_array_equal(trajectory.x_static[:, 1], fields[7][0])
        np.testing.assert_allclose(trajectory.x_static[:, 2], np.sqrt(fields[4][0] ** 2 + fields[5][0] ** 2))
        np.testing.assert_array_equal(trajectory.x_static[:, 3], [1, 0, 0])
        np.testing.assert_array_equal(trajectory.force[..., 0], np.broadcast_to([10, 0, 0], (12, 3)))
        np.testing.assert_array_equal(trajectory.force[..., 1], fields[3])
        np.testing.assert_array_equal(trajectory.state, np.stack([fields[4], fields[5], fields[9]], axis=-1))

    def test_future_floating_cannot_change_any_model_input(self):
        payload = _issm_payload()
        changed = deepcopy(payload)
        changed["S"][0][0][10][1:] = np.arange(33).reshape(11, 3) * -1234.0
        before = _issm_dataset(payload)
        after = _issm_dataset(changed)
        for index in range(len(before)):
            for key, value in before[index]:
                actual = after[index][key]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, actual), msg=key)
                else:
                    self.assertEqual(value, actual)

    def test_only_initial_static_quantities_are_used(self):
        payload = _issm_payload()
        changed = deepcopy(payload)
        for field_index in [4, 5, 7, 8]:
            changed["S"][0][0][field_index][1:] += 1e5
        original = _issm_dataset(payload).iter_trajectories()[0]
        modified = _issm_dataset(changed).iter_trajectories()[0]
        np.testing.assert_array_equal(original.x_static, modified.x_static)
        np.testing.assert_array_equal(original.edge_attr, modified.edge_attr)

    def test_initial_floating_changes_binary_and_melt_masks(self):
        payload = _issm_payload()
        payload["S"][0][0][10][0] = np.array([0, -0.1, -1000])
        trajectory = _issm_dataset(payload).iter_trajectories()[0]
        np.testing.assert_array_equal(trajectory.x_static[:, 3], [0, 1, 1])
        np.testing.assert_array_equal(trajectory.force[..., 0], np.broadcast_to([0, 10, 10], (12, 3)))

    def test_generic_legacy_dimensions_are_rejected(self):
        payload = dict(x_static=np.zeros((3, 2)), force=np.zeros((12, 3, 3)),
                       state=np.zeros((12, 3, 3)), edge_index=np.array([[0, 1], [1, 0]]))
        with self.assertRaisesRegex(ValueError, "canonical protocol"):
            _issm_dataset(payload)

    def test_published_rate_split_is_unchanged(self):
        files = [Path(f"PIG_transient_r{rate:03d}.mat") for rate in range(0, 72, 2)]
        train, val, test = issm_rate_modulo_split(files)
        rates = lambda paths: [parse_issm_rate_from_filename(path) for path in paths]
        self.assertEqual(rates(val), [0, 20, 40, 60])
        self.assertEqual(rates(test), [10, 30, 50, 70])
        self.assertEqual(len(train), 28)
        self.assertEqual(set(train) | set(val) | set(test), set(files))
        self.assertFalse(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))


class NormalizationTest(unittest.TestCase):
    def test_small_real_variance_constant_channels_and_roundtrip(self):
        unit = np.linspace(-1, 1, 1001, dtype=np.float64)
        unit /= unit.std()
        features = np.stack([1.2e-5 + unit * 5.855e-6, 2e-8 + unit * 1e-8,
                             np.full_like(unit, 0.03), 2 + unit * 3], axis=-1).astype(np.float32)
        trajectories = [SimpleNamespace(x_static=part, force=part[:, None, :], state=part[:, None, :])
                        for part in np.array_split(features, 3)]
        normalizer = FeatureNormalizer.fit_from_trajectories(trajectories)
        value = torch.from_numpy(features)
        normalized = normalizer.transform_force(value)
        self.assertEqual(normalized.dtype, torch.float32)
        torch.testing.assert_close(normalized[:, [0, 1, 3]].std(dim=0, correction=0), torch.ones(3), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(normalized[:, [0, 1, 3]].mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.equal(normalized[:, 2], torch.zeros(len(features))))
        self.assertEqual(normalizer.force_std[2], 1)
        self.assertAlmostEqual(float(normalizer.force_std[0]), 5.855e-6, delta=1e-12)
        restored = FeatureNormalizer.from_dict(json.loads(json.dumps(normalizer.to_dict())))
        for method in ["transform_force", "transform_static", "transform_state"]:
            self.assertTrue(torch.equal(getattr(restored, method)(value), getattr(normalizer, method)(value)))
        torch.testing.assert_close(normalizer.inverse_state(normalizer.transform_state(value)), value)

    def test_constant_detection_is_independent_of_physical_scale(self):
        pattern = np.array([1, 2, 4, 8], dtype=np.float32)[:, None]
        for scale in [1.0, 1e-8, 1e-12]:
            values = pattern * scale
            trajectory = SimpleNamespace(x_static=values, force=values, state=values)
            normalized = FeatureNormalizer.fit_from_trajectories([trajectory]).transform_force(torch.from_numpy(values))
            self.assertAlmostEqual(normalized.std(correction=0).item(), 1.0, places=6)
        zeros = np.zeros((4, 1), dtype=np.float32)
        normalizer = FeatureNormalizer.fit_from_trajectories([SimpleNamespace(x_static=zeros, force=zeros, state=zeros)])
        self.assertTrue(torch.equal(normalizer.transform_force(torch.from_numpy(zeros)), torch.zeros(4, 1)))


class ReferenceWindowTest(unittest.TestCase):
    def test_missing_reference_preserves_existing_windows(self):
        self.assertEqual(enumerate_window_end_indices(12, 3, 4, 1), list(range(2, 8)))
        self.assertEqual(enumerate_window_end_indices(12, 3, 4, 2), [2, 4, 6])

    def test_all_history_and_future_scans_share_prediction_anchors(self):
        scans = [
            (73, [(h, 64) for h in range(1, 9)], {"history_len": 8, "future_len": 64}, [7, 8]),
            (240, [(h, 120) for h in range(1, 9)], {"history_len": 8, "future_len": 120}, list(range(7, 120))),
            (240, [(6, k) for k in [30, 45, 60, 75, 90, 120, 150, 180]], {"history_len": 6, "future_len": 180}, list(range(5, 60))),
        ]
        for steps, variants, reference, expected in scans:
            for history, future in variants:
                config = {"seed": 42, "dataset": {"history_len": history, "future_len": future, "window_reference": reference}}
                with patch.dict("datasets.factory.DATASET_REGISTRY", {"synthetic": _SyntheticDataset}):
                    dataset = build_dataset("synthetic", [Path(str(steps))], "train", config)
                self.assertEqual([window.t_end for window in dataset.all_windows], expected)
                self.assertEqual(len(dataset), len(expected))
                self.assertEqual(dataset[0].state_hist.shape[1], history)
                self.assertEqual(dataset[0].y_future.shape[1], future)

    def test_current_constraints_remain_valid_and_bad_reference_is_rejected(self):
        self.assertEqual(enumerate_window_end_indices(20, 8, 6, 1, {"history_len": 2, "future_len": 3}), list(range(7, 14)))
        self.assertEqual(enumerate_window_end_indices(10, 1, 2, 1, {"history_len": 8, "future_len": 8}), [])
        with self.assertRaisesRegex(ValueError, "window_reference"):
            enumerate_window_end_indices(10, 1, 2, 1, {"history_len": 0})


class LoaderRepairTest(unittest.TestCase):
    def test_nonpadding_eval_sampler_has_no_duplicates_or_missing_samples(self):
        for length in [0, 1, 2, 8, 9, 11, 12]:
            for world_size in [1, 2, 4]:
                shards = []
                for rank in range(world_size):
                    sampler = DistributedEvaluationSampler(range(length), world_size, rank)
                    indices = list(sampler)
                    self.assertEqual(indices, list(range(rank, length, world_size)))
                    self.assertEqual(len(sampler), len(indices))
                    shards.extend(indices)
                self.assertEqual(sorted(shards), list(range(length)))
                self.assertEqual(len(shards), len(set(shards)))

    def test_loader_uses_nonpadding_sampler_for_evaluation(self):
        dataset = _SyntheticDataset([Path("12")], 2, 3, "val")
        with patch("datasets.factory.dist.get_world_size", return_value=3), patch("datasets.factory.dist.get_rank", return_value=1):
            loader = build_loader(dataset, 2, 0, distributed=True, shuffle=False)
        self.assertIsInstance(loader.sampler, DistributedEvaluationSampler)
        self.assertEqual(list(loader.sampler), [1, 4, 7])

    def test_persistent_workers_disabled_only_for_training_resampling(self):
        for sampling in [{"windows_per_scenario": 2}, {"epoch_num_windows": 2}]:
            dataset = _SyntheticDataset([Path("12")], 2, 3, "train", **sampling)
            with self.assertLogs("datasets.factory", level="WARNING") as messages:
                loader = build_loader(dataset, 2, 1, False, True, persistent_workers=True)
            self.assertFalse(loader.persistent_workers)
            self.assertIn("epoch window resampling", messages.output[0])
        for split in ["train", "val"]:
            dataset = _SyntheticDataset([Path("12")], 2, 3, split)
            loader = build_loader(dataset, 2, 1, False, split == "train", persistent_workers=True)
            self.assertTrue(loader.persistent_workers)
        loader = build_loader(dataset, 2, 0, False, False, persistent_workers=True)
        self.assertFalse(loader.persistent_workers)


if __name__ == "__main__":
    unittest.main()
