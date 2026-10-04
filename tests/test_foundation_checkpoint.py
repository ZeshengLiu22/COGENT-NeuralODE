"""Regression tests for checkpoint provenance and the solver diagnostic."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import check_solver_convergence
import evaluate
import run_full_rollout
from datasets.factory import build_dataset
from datasets.normalization import FeatureNormalizer
from datasets.split_utils import make_split_manifest, resolve_split_manifest
from models import build_model
from training import Evaluator
from utils.checkpoint_evaluation import restore_checkpoint_splits, restore_evaluation_config
from utils.io import load_config_bundle


class CheckpointFoundationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        for index in range(3):
            np.savez(
                self.data_dir / f"case_{index}.npz",
                x_static=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                force=np.arange(6, dtype=np.float32).reshape(6, 1, 1).repeat(2, axis=1) / 5,
                state=(np.arange(12, dtype=np.float32).reshape(6, 2, 1) + index) / 10,
                edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                times=np.arange(6, dtype=np.float32),
            )
        self.config = load_config_bundle([PROJECT_ROOT / "configs/base_sample.yaml"])
        self.config["dataset"].update(name="adcirc", data_dir=str(self.data_dir), history_len=2, future_len=2)
        self.config["evaluation"].update(num_workers=0, full_rollout_known_steps=2)
        self.config["amp"]["mode"] = "bf16"
        model = self.config["model"]
        model.update(latent_dim=4, decoder_hidden_dims=[4], use_residual_decoder=True, relative_time_scale=4.0)
        model["history_encoder"].update(
            static_hidden_dims=[4], static_embed_dim=4, gnn_hidden_dim=4, gnn_num_layers=1,
            lstm_hidden_dim=4, history_encoder_type="lstm", dropout=0.0,
        )
        model["continuous"].update(hidden_dim=4, num_layers=1)
        self.config["solver"]["interpolation"] = "linear"
        self.split_files = {split: [self.data_dir / f"case_{index}.npz"] for index, split in enumerate(("train", "val", "test"))}
        dataset = build_dataset("adcirc", self.split_files["train"], "train", self.config)
        self.normalizer = FeatureNormalizer.fit_from_trajectories(dataset.iter_trajectories())
        dataset.normalizer = self.normalizer
        sample = dataset[0]
        torch.manual_seed(3)
        self.model = build_model(self.config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1]).eval()
        self.checkpoint = {
            "config": deepcopy(self.config),
            "normalizer": self.normalizer.to_dict(),
            "model_state": self.model.state_dict(),
            "split_manifest": make_split_manifest(self.split_files, self.data_dir),
        }
        self.checkpoint_path = self.root / "best.pt"
        torch.save(self.checkpoint, self.checkpoint_path)

    def test_model_config_is_authoritative(self) -> None:
        for section, override in (
            ("model", {"use_residual_decoder": False}),
            ("model", {"relative_time_scale": 99.0}),
            ("model", {"history_encoder": {"history_encoder_type": "transformer"}}),
            ("solver", {"ode_options": {"step_size": 0.5}}),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "checkpoint training config is authoritative"):
                restore_evaluation_config(self.checkpoint, {section: override})
        restored = restore_evaluation_config(self.checkpoint, self.config)
        self.assertEqual(restored["model"], self.checkpoint["config"]["model"])
        reloaded = build_model(restored, 2, 1, 1).eval()
        reloaded.load_state_dict(self.checkpoint["model_state"])
        dataset = build_dataset("adcirc", self.split_files["test"], "test", restored, self.normalizer)
        with torch.no_grad():
            torch.testing.assert_close(self.model(dataset[0]), reloaded(dataset[0]), rtol=0, atol=0)

    def test_only_explicit_runtime_overrides_and_fp32_default(self) -> None:
        self.checkpoint["config"]["evaluation"]["amp_mode"] = "bf16"
        restored = restore_evaluation_config(self.checkpoint, runtime_overrides={
            "dataset": {"history_len": 1, "future_len": 3},
            "evaluation": {"batch_size": 2, "num_workers": 0},
        })
        self.assertEqual(restored["evaluation"]["amp_mode"], "none")
        self.assertEqual(restored["dataset"]["history_len"], 1)
        self.assertEqual(self.checkpoint["config"]["dataset"]["history_len"], 2)
        explicit = restore_evaluation_config(self.checkpoint, {"evaluation": {"amp_mode": "bf16"}})
        self.assertEqual(explicit["evaluation"]["amp_mode"], "bf16")
        with self.assertRaises(ValueError):
            restore_evaluation_config(self.checkpoint, {"dataset": {"stride": 3}})

    def test_saved_split_survives_extra_files_and_relocation(self) -> None:
        expected = restore_checkpoint_splits(self.checkpoint, self.data_dir)
        (self.data_dir / "unrelated.npz").touch()
        self.assertEqual(restore_checkpoint_splits(self.checkpoint, self.data_dir), expected)
        relocated = self.root / "relocated"
        shutil.copytree(self.data_dir, relocated)
        shutil.rmtree(self.data_dir)
        restored = restore_checkpoint_splits(self.checkpoint, relocated)
        self.assertEqual(restored["test"], [relocated / "case_2.npz"])
        self.assertEqual(self.checkpoint["split_manifest"]["test"], ["case_2.npz"])

    def test_manifest_rejects_duplicate_or_missing_scenarios(self) -> None:
        for source, target in (("train", "val"), ("train", "test"), ("val", "test")):
            manifest = deepcopy(self.checkpoint["split_manifest"])
            manifest[target] = manifest[source]
            with self.subTest(pair=(source, target)), self.assertRaisesRegex(ValueError, "occurs in both"):
                resolve_split_manifest(manifest, self.data_dir)
        missing = deepcopy(self.checkpoint["split_manifest"])
        missing["test"] = ["missing.npz"]
        with self.assertRaises(FileNotFoundError):
            resolve_split_manifest(missing, self.data_dir)
        with self.assertRaisesRegex(ValueError, "no split_manifest"):
            restore_checkpoint_splits({"config": self.config}, self.data_dir)

    def test_standalone_evaluation_matches_evaluator_without_config(self) -> None:
        (self.data_dir / "unrelated.npz").touch()
        output_dir = self.root / "window_artifacts"
        with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", str(self.checkpoint_path), "--device", "cpu", "--output-dir", str(output_dir)]):
            evaluate.main()
        metrics = json.loads((output_dir / "best.window_metrics.json").read_text())
        dataset = build_dataset("adcirc", self.split_files["test"], "test", self.config, self.normalizer)
        expected = Evaluator(self.model, self.normalizer, torch.device("cpu")).evaluate_loader(DataLoader(dataset, batch_size=1))
        self.assertAlmostEqual(metrics["norm_rmse"], expected["norm_rmse"], places=6)
        metadata = json.loads((output_dir / "best.window_predictions_meta.json").read_text())["evaluation_metadata"]
        self.assertEqual(metadata["amp_mode"], "none")
        self.assertEqual(metadata["split_source"], "checkpoint.split_manifest")

    def test_full_rollout_uses_embedded_split_after_relocation(self) -> None:
        relocated = self.root / "relocated"
        shutil.copytree(self.data_dir, relocated)
        shutil.rmtree(self.data_dir)
        output_dir = self.root / "rollout_artifacts"
        with patch.object(sys, "argv", [
            "run_full_rollout.py", "--checkpoint", str(self.checkpoint_path), "--device", "cpu",
            "--data-dir", str(relocated), "--output-dir", str(output_dir),
        ]):
            run_full_rollout.main()
        metadata = json.loads((output_dir / "best.full_rollout_predictions_meta.json").read_text())["evaluation_metadata"]
        self.assertEqual(metadata["split_manifest"], "checkpoint.split_manifest")
        self.assertEqual(metadata["amp_mode"], "none")

    def test_solver_diagnostic_runs_from_checkpoint_and_restores_solver(self) -> None:
        with patch.object(sys, "argv", ["check_solver_convergence.py", "--checkpoint", str(self.checkpoint_path), "--device", "cpu"]):
            check_solver_convergence.main()
        report = json.loads((self.root / "solver_convergence.json").read_text())
        self.assertEqual(report["split"], "val")
        self.assertEqual(report["scenario_count"], 1)
        self.assertEqual(report["scalar_prediction_count"], 8)
        self.assertEqual(len(report["settings"]), 3)
        for setting in report["settings"]:
            self.assertTrue(np.isfinite(setting["validation_normalized_rmse"]))
        self.assertEqual(report["settings"][1]["reference_solution"], "step_size_0.25")
        self.assertEqual(report["settings"][2]["prediction_rmse_difference"], 0.0)
        dataset = build_dataset("adcirc", self.split_files["val"], "val", self.config, self.normalizer)
        original = deepcopy(self.model.solver_cfg)
        check_solver_convergence.check_convergence(self.model, dataset, self.normalizer, torch.device("cpu"), start_t=1)
        self.assertEqual(self.model.solver_cfg, original)


if __name__ == "__main__":
    unittest.main()
