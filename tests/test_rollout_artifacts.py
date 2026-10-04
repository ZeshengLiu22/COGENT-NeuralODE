"""Authoritative rollout artifacts, model-free slices and exact DDP statistics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.normalization import FeatureNormalizer
from training.evaluator import Evaluator
from utils.eval_artifacts import (collect_full_rollout_prediction_bundle, save_evaluation_artifacts,
                                 summarize_full_rollout_bundle, validate_full_rollout_bundle)
from utils.rollout_postprocess import compare_rollout_artifacts, load_rollout_artifact, save_comparison_report


class SyntheticDataset:
    dataset_name = "issm"
    cache_in_memory = True
    future_len = 1  # Deliberately shorter than every evaluated rollout.

    def __init__(self, *, history_len=2, length=9, order=(0, 1, 2), variable_lengths=True):
        self.history_len = history_len
        self.normalizer = FeatureNormalizer(torch.zeros(2), torch.ones(2), torch.zeros(1), torch.ones(1),
                                            torch.tensor([0.35, -0.28, 10.0]), torch.tensor([2.3, 3.1, 7.3]))
        self.scenario_infos = []
        self.trajectories = []
        for index in order:
            steps = length + index if variable_lengths else length
            nodes = 2 + index
            state = np.arange(steps * nodes * 3, dtype=np.float32).reshape(steps, nodes, 3) / 7.0
            times = np.arange(steps, dtype=np.float32) * 1.25 + 2000.0
            self.trajectories.append(SimpleNamespace(state=state, times=times, x_static=np.arange(nodes * 2, dtype=np.float32).reshape(nodes, 2),
                                                     edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64), edge_attr=None))
            self.scenario_infos.append({"scenario_id": f"scenario-{index}", "sim_id": f"simulation-{index}",
                                        "length": steps, "path": Path(f"scenario-{index}.npz")})

    def _get_trajectory(self, index, cache=True):
        return self.trajectories[index]

    def get_rollout_data(self, index, *, start_t):
        trajectory = self.trajectories[index]
        if not self.history_len - 1 <= start_t < len(trajectory.times) - 1:
            raise ValueError("Invalid rollout start.")
        return Data(y_future=self.normalizer.transform_state(torch.from_numpy(trajectory.state[start_t + 1:]).permute(1, 0, 2)),
                    x_static=torch.from_numpy(trajectory.x_static), scenario_idx=torch.tensor([index]),
                    t_idx=torch.tensor([start_t]), t_value=torch.tensor([trajectory.times[start_t]]),
                    future_idx=torch.arange(start_t + 1, len(trajectory.times)).unsqueeze(0),
                    future_time=torch.from_numpy(trajectory.times[start_t + 1:]).unsqueeze(0))


class ForecastModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, data):
        self.seen.append(int(data.scenario_idx.item()))
        steps = torch.arange(1, data.y_future.shape[1] + 1, dtype=torch.float32).view(1, -1, 1)
        return data.y_future + steps * torch.tensor([0.017, -0.013, 0.011]).view(1, 1, 3)


class RolloutArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = SyntheticDataset()
        self.channels = ["vx", "vy", "thickness"]

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, known_steps, *, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        bundle = collect_full_rollout_prediction_bundle(ForecastModel(), dataset, dataset.normalizer,
                                                        torch.device("cpu"), known_steps=known_steps)
        summary = summarize_full_rollout_bundle(bundle, self.channels)
        stem = self.root / f"known{known_steps}"
        with patch("utils.eval_artifacts._plot_trajectory_metrics"), patch("utils.eval_artifacts._plot_leadtime_metrics"):
            paths = save_evaluation_artifacts(stem, bundle=bundle, summary=summary, dataset_name="issm", split="test",
                                              mode="full_rollout", channel_names=self.channels, scenario_infos=dataset.scenario_infos)
        return Path(paths["predictions_npz"]), bundle, summary

    def test_complete_artifacts_exactly_recompute_direct_metrics(self):
        path, original, saved_summary = self.save(3)
        bundle, meta = load_rollout_artifact(path)
        self.assertEqual(meta["known_steps"], 3)
        self.assertEqual(meta["history_len"], 2)
        for key in original:
            np.testing.assert_array_equal(original[key], bundle[key])
        self.assertEqual(len(bundle["pred_phys"]), sum((info["length"] - 3) * (2 + index)
                                                        for index, info in enumerate(self.dataset.scenario_infos)))
        self.assertGreater(len(saved_summary["metrics"]["horizon_rmse_curve"]), self.dataset.future_len)
        direct = Evaluator(ForecastModel(), self.dataset.normalizer, torch.device("cpu")).evaluate_full_rollout(self.dataset, known_steps=3)
        recomputed = compare_rollout_artifacts([path])["results"][0]["metrics"]
        self.assertEqual(set(direct), set(recomputed))
        for key in direct:
            np.testing.assert_allclose(direct[key], recomputed[key], rtol=1e-13, atol=1e-13, err_msg=key)
        self.assertEqual(recomputed, saved_summary["metrics"])

    def test_equal_lead_and_common_tail_with_scenario_reordering(self):
        paths = [self.save(2)[0], self.save(4, dataset=SyntheticDataset(order=(2, 0, 1)))[0], self.save(6)[0]]
        equal = compare_rollout_artifacts(paths, mode="equal-lead", lead_steps=3)
        for result in equal["results"]:
            for scenario in result["scenarios"]:
                self.assertEqual(scenario["lead_steps"], [1, 2, 3])
                self.assertEqual(scenario["absolute_indices"], list(range(result["known_steps"], result["known_steps"] + 3)))
        tail = compare_rollout_artifacts(paths, mode="common-tail")
        for result in tail["results"]:
            for scenario in result["scenarios"]:
                expected_end = 9 + int(scenario["scenario_id"].split("-")[1])
                self.assertEqual(scenario["absolute_indices"], list(range(6, expected_end)))
        self.assertGreater(tail["results"][0]["metrics"]["whole_rollout_rmse"], tail["results"][-1]["metrics"]["whole_rollout_rmse"])
        saved = save_comparison_report(tail, self.root / "comparison")
        self.assertTrue(all(Path(path).is_file() for path in saved.values()))
        with self.assertRaisesRegex(ValueError, "fewer than"):
            compare_rollout_artifacts(paths, mode="equal-lead", lead_steps=4)

    def test_requested_240_step_comparisons(self):
        dataset = SyntheticDataset(history_len=6, length=240, order=(0,), variable_lengths=False)
        paths = [self.save(start, dataset=dataset)[0] for start in (60, 90, 120)]
        equal = compare_rollout_artifacts(paths, mode="equal-lead", lead_steps=120)
        tail = compare_rollout_artifacts(paths, mode="common-tail")
        for result, start in zip(equal["results"], (60, 90, 120)):
            self.assertEqual(result["scenarios"][0]["absolute_indices"], list(range(start, start + 120)))
        for result in tail["results"]:
            self.assertEqual(result["scenarios"][0]["absolute_indices"], list(range(120, 240)))

    def test_arbitrary_lead_absolute_and_actual_time_slices(self):
        path = self.save(2)[0]
        lead = compare_rollout_artifacts([path], mode="lead-slice", lead_start=2, lead_stop=4)
        absolute = compare_rollout_artifacts([path], mode="absolute-slice", absolute_start=3, absolute_stop=6)
        physical = compare_rollout_artifacts([path], mode="time-slice", time_start=2003.75, time_stop=2007.5)
        self.assertEqual(lead["results"][0]["metrics"], absolute["results"][0]["metrics"])
        self.assertEqual(absolute["results"][0]["metrics"], physical["results"][0]["metrics"])
        self.assertEqual(lead["results"][0]["metrics"]["horizon_lead_steps"], [2, 3, 4])
        self.assertEqual(len(lead["results"][0]["metrics"]["horizon_rmse_curve"]), 3)

    def test_absolute_and_physical_time_comparisons_reject_silent_clipping(self):
        paths = [self.save(2)[0], self.save(6)[0]]
        for options in (
            {"mode": "absolute-slice", "absolute_start": 4, "absolute_stop": 8},
            {"mode": "absolute-slice", "absolute_stop": 8},
            {"mode": "time-slice", "time_start": 2005.0, "time_stop": 2010.0},
            {"mode": "time-slice", "time_stop": 2010.0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare_rollout_artifacts(paths, **options)
        with self.assertRaisesRegex(ValueError, "requested absolute slice"):
            compare_rollout_artifacts([paths[1]], mode="absolute-slice", absolute_start=4, absolute_stop=8)
        matched = compare_rollout_artifacts(paths, mode="absolute-slice", absolute_start=6, absolute_stop=8)
        for result in matched["results"]:
            self.assertTrue(all(row["absolute_indices"] == [6, 7] for row in result["scenarios"]))
        # A bound strictly between anchor and first forecast contains no missing discrete state.
        physical = compare_rollout_artifacts(paths, mode="time-slice", time_start=2006.5, time_stop=2010.0)
        for result in physical["results"]:
            self.assertTrue(all(row["absolute_indices"] == [6, 7] for row in result["scenarios"]))

    def test_scenario_time_node_channel_mismatches_fail(self):
        first, _, _ = self.save(2)
        second, _, _ = self.save(4)
        meta_path = second.with_name(second.name.replace("_predictions.npz", "_predictions_meta.json"))
        original_meta = json.loads(meta_path.read_text())
        with np.load(second) as archive:
            original_bundle = {name: archive[name] for name in archive.files}
        for mismatch in ("scenario", "time", "node", "channel", "target", "normalization"):
            bundle = {name: array.copy() for name, array in original_bundle.items()}
            meta = json.loads(json.dumps(original_meta))
            if mismatch == "scenario":
                bundle["scenario_id"][0] = "different"
                meta["scenario_lookup"][0]["scenario_id"] = "different"
            elif mismatch == "time":
                bundle["trajectory_time"] += 0.1
                bundle["history_end_time"] += 0.1
            elif mismatch == "node":
                bundle["mesh_signature"][0] = "a" * 64
                meta["scenario_lookup"][0]["mesh_signature"] = "a" * 64
            elif mismatch == "normalization":
                bundle["normalization_signature"][:] = "b" * 64
                for info in meta["scenario_lookup"]:
                    info["normalization_signature"] = "b" * 64
            elif mismatch == "channel":
                meta["channel_names"] = ["vy", "vx", "thickness"]
            else:
                bundle["target_phys"][0] += 1
            np.savez_compressed(second, **bundle)
            meta_path.write_text(json.dumps(meta))
            with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                compare_rollout_artifacts([first, second], mode="common-tail")

    def test_duplicate_node_time_rows_fail_even_if_marginal_counts_match(self):
        _, bundle, _ = self.save(3)
        # Duplicate node 0 and discard node 1 at one timestep, preserving timestep counts.
        row = np.flatnonzero((bundle["scenario_index"] == 0) & (bundle["node_index"] == 1))[0]
        bundle["node_index"][row] = 0
        with self.assertRaisesRegex(ValueError, "duplicate or missing"):
            validate_full_rollout_bundle(bundle, scenario_infos=self.dataset.scenario_infos, known_steps=3, node_counts=[2, 3, 4])

    def test_invalid_known_steps_rejected_in_both_inference_paths(self):
        evaluator = Evaluator(ForecastModel(), self.dataset.normalizer, torch.device("cpu"))
        for known in (1, 9):
            with self.subTest(known=known), self.assertRaises(ValueError):
                evaluator.evaluate_full_rollout(self.dataset, known_steps=known)
            with self.subTest(known=known), self.assertRaises(ValueError):
                collect_full_rollout_prediction_bundle(ForecastModel(), self.dataset, self.dataset.normalizer,
                                                        torch.device("cpu"), known_steps=known)

    def test_ddp_nonpadding_shards_sum_exact_statistics_with_empty_rank(self):
        def evaluate(rank=None, world_size=1):
            model = ForecastModel()
            evaluator = Evaluator(model, self.dataset.normalizer, torch.device("cpu"))
            statistics = []
            with patch.object(evaluator, "_all_reduce", side_effect=lambda tensor: statistics.append(tensor.clone())):
                if rank is None:
                    evaluator.evaluate_full_rollout(self.dataset, known_steps=3)
                else:
                    with patch("training.evaluator.dist.is_initialized", return_value=True), \
                         patch("training.evaluator.dist.get_rank", return_value=rank), \
                         patch("training.evaluator.dist.get_world_size", return_value=world_size):
                        evaluator.evaluate_full_rollout(self.dataset, known_steps=3)
            return model.seen, statistics
        expected_visits, expected_stats = evaluate()
        visited, shards = [], []
        for rank in range(5):
            indices, statistics = evaluate(rank, 5)
            self.assertEqual(indices, list(range(rank, 3, 5)))
            visited.extend(indices)
            shards.append(statistics)
        self.assertEqual(sorted(visited), expected_visits)
        self.assertEqual(len(visited), len(set(visited)))
        for index, expected in enumerate(expected_stats):
            total = sum(shard[index] for shard in shards)
            torch.testing.assert_close(total, expected, rtol=1e-14, atol=1e-14)
            self.assertEqual(total.dtype, torch.float64)
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in shards[-1]))


if __name__ == "__main__":
    unittest.main()
