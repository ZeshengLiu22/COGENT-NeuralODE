"""Regression tests for the foundation input, normalization, and window repairs."""

from __future__ import annotations

import inspect
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
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_foundation import validate_main_protocol, validate_temporal_settings
from datasets.base_dataset import BaseTemporalGraphDataset, TrajectoryData
from datasets.factory import build_dataset, build_loader
from torch.utils.data.distributed import DistributedSampler
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
            force=np.broadcast_to(np.arange(steps, dtype=np.float32)[:, None, None], (steps, 3, 2)).copy(),
            state=np.broadcast_to(np.arange(steps, dtype=np.float32)[:, None, None], (steps, 3, 3)).copy(),
            edge_index=np.array([[0, 1], [1, 0]]),
            times=12.0 + 2.5 * np.arange(steps, dtype=np.float32),
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


class TrainingWindowTest(unittest.TestCase):
    def test_enumeration_has_only_h_k_stride_and_inclusive_valid_anchors(self):
        self.assertEqual(list(inspect.signature(enumerate_window_end_indices).parameters),
                         ["total_steps", "history_len", "future_len", "stride"])
        self.assertEqual(enumerate_window_end_indices(12, 3, 4, 1), list(range(2, 8)))
        self.assertEqual(enumerate_window_end_indices(12, 3, 4, 2), [2, 4, 6])
        self.assertEqual(enumerate_window_end_indices(7, 3, 4, 1), [2])
        self.assertEqual(enumerate_window_end_indices(6, 3, 4, 1), [])

    def test_h_and_k_naturally_change_training_anchors(self):
        for history in range(1, 9):
            for future in [30, 45, 60, 75, 90, 120, 150, 180]:
                config = {"seed": 42, "dataset": {"history_len": history, "future_len": future}}
                with patch.dict("datasets.factory.DATASET_REGISTRY", {"synthetic": _SyntheticDataset}):
                    dataset = build_dataset("synthetic", [Path("240")], "train", config)
                expected = list(range(history - 1, 240 - future))
                self.assertEqual([window.t_end for window in dataset.all_windows], expected)
                self.assertEqual(len(dataset), 240 - future - history + 1)
                self.assertEqual(dataset[0].state_hist.shape[1], history)
                self.assertEqual(dataset[0].y_future.shape[1], future)
                torch.testing.assert_close(dataset[0].state_hist[0, :, 0], torch.arange(history).float())
                torch.testing.assert_close(dataset[0].y_future[0, :, 0], torch.arange(history, history + future).float())

    def test_h_k_and_stride_must_be_positive(self):
        for history, future, stride in [(0, 2, 1), (2, 0, 1), (2, 3, 0)]:
            with self.assertRaisesRegex(ValueError, "must be >= 1"):
                enumerate_window_end_indices(20, history, future, stride)


class RolloutIndexingTest(unittest.TestCase):
    def test_absolute_rollout_start_selects_the_preceding_h_true_states(self):
        for history, known in [(1, 60), (6, 60), (6, 90), (6, 120)]:
            with self.subTest(history=history, known=known):
                dataset = _SyntheticDataset([Path("240")], history, 180, "test")
                sample = dataset.get_rollout_data(0, start_t=known - 1)
                history_indices = torch.arange(known - history, known)
                future_indices = torch.arange(known, 240)
                torch.testing.assert_close(sample.history_idx[0], history_indices)
                torch.testing.assert_close(sample.state_hist[0, :, 0], history_indices.float())
                torch.testing.assert_close(sample.force_hist[0, :, 0], history_indices.float())
                torch.testing.assert_close(sample.future_idx[0], future_indices)
                torch.testing.assert_close(sample.y_future[0, :, 0], future_indices.float())
                torch.testing.assert_close(sample.force_future[0, :, 0], future_indices.float())
                torch.testing.assert_close(sample.history_time[0], 12 + 2.5 * history_indices.float())
                torch.testing.assert_close(sample.future_time[0], 12 + 2.5 * future_indices.float())
                torch.testing.assert_close(sample.t_hist[0], torch.arange(1 - history, 1).float())
                torch.testing.assert_close(sample.t_future[0], torch.arange(1, 241 - known).float())
                self.assertEqual(sample.rollout_length.item(), 240 - known)

    def test_k_does_not_limit_rollout_or_require_any_training_windows(self):
        for future in [30, 60, 180, 240]:
            with self.subTest(future=future):
                dataset = _SyntheticDataset([Path("240")], 6, future, "test")
                sample = dataset.get_rollout_data(0, start_t=59)
                self.assertEqual(sample.y_future.shape, (3, 180, 3))
                self.assertEqual(sample.future_idx[0].tolist(), list(range(60, 240)))
                if future == 240:
                    self.assertEqual(len(dataset), 0)
                    self.assertEqual(len(dataset.scenario_infos), 1)

    def test_known_steps_must_include_history_and_leave_a_future(self):
        dataset = _SyntheticDataset([Path("240")], 6, 30, "test")
        for known in [0, 5]:
            with self.subTest(known=known), self.assertRaisesRegex(ValueError, ">= history_len"):
                dataset.get_rollout_data(0, start_t=known - 1)
        for known in [240, 241]:
            with self.subTest(known=known), self.assertRaisesRegex(ValueError, "< trajectory length"):
                dataset.get_rollout_data(0, start_t=known - 1)
        self.assertEqual(dataset.get_rollout_data(0, start_t=5).history_idx[0].tolist(), list(range(6)))
        self.assertEqual(dataset.get_rollout_data(0, start_t=238).future_idx[0].tolist(), [239])


class LoaderRepairTest(unittest.TestCase):
    def test_distributed_loader_is_only_for_training_windows(self):
        dataset = _SyntheticDataset([Path("12")], 2, 3, "train")
        with patch("torch.distributed.get_world_size", return_value=3), patch("torch.distributed.get_rank", return_value=1):
            loader = build_loader(dataset, 2, 0, distributed=True, shuffle=False)
        self.assertIsInstance(loader.sampler, DistributedSampler)
        self.assertEqual(list(loader.sampler), [1, 4, 7])
        dataset.split = "val"
        with self.assertRaisesRegex(ValueError, "evaluate complete trajectories directly"):
            build_loader(dataset, 2, 0, distributed=True, shuffle=False)

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


class TemporalAuditTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "dataset": {"name": "issm", "data_dir": "/data/ISSM/PIG_5000", "history_len": 1, "future_len": 180},
            "evaluation": {"known_steps": 60},
            "model": {"relative_time_scale": 180.0},
            "training": {"train_horizon_min": 24, "train_horizon_max": None},
        }

    def test_canonical_protocols_and_null_horizon_max(self):
        validate_main_protocol(self.config, "issm")
        semantics = validate_temporal_settings(self.config, {"simulation": 240})
        self.assertEqual(semantics["target_train_horizon_max"], 180)
        self.assertEqual(semantics["history_indices"], [59])
        self.assertEqual(semantics["rollout_lengths"], [180])
        self.config["dataset"].update(name="anuga", history_len=1, future_len=64)
        self.config["evaluation"]["known_steps"] = 8
        self.config["model"]["relative_time_scale"] = 65
        validate_main_protocol(self.config, "anuga")
        self.assertEqual(validate_temporal_settings(self.config, {"simulation": 73})["rollout_lengths"], [65])

    def test_training_k_and_rollout_start_are_independent(self):
        self.config["dataset"].update(history_len=6, future_len=30)
        for known in [60, 90, 120]:
            self.config["evaluation"]["known_steps"] = known
            result = validate_temporal_settings(self.config, {"simulation": 240})
            self.assertEqual(result["target_train_horizon_max"], 30)
            self.assertEqual(result["rollout_lengths"], [240 - known])
            self.assertEqual(result["relative_time_scale"], 180)

    def test_invalid_temporal_settings_are_rejected(self):
        for section, key, value, message in [
            ("dataset", "history_len", 61, "known_steps"),
            ("evaluation", "known_steps", 240, "trajectory length"),
            ("dataset", "future_len", 23, "train_horizon_min"),
            ("training", "train_horizon_max", 181, "train_horizon_max"),
            ("training", "train_horizon_max", 23, "train_horizon_max"),
        ]:
            invalid = deepcopy(self.config)
            invalid[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                validate_temporal_settings(invalid, {"simulation": 240})


if __name__ == "__main__":
    unittest.main()
