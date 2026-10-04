"""Regression coverage for fixed relative time and formal configuration protocol."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from datasets.window_utils import enumerate_window_end_indices
from models import build_model
from test_smoke import _base_config
from utils.io import load_config_bundle, load_yaml


class FoundationModelTest(unittest.TestCase):
    def test_relative_time_prediction_prefix_is_independent_of_horizon(self) -> None:
        torch.manual_seed(31)
        sample = Data(
            x_static=torch.randn(4, 4),
            state_hist=torch.randn(4, 3, 3),
            force_hist=torch.randn(4, 3, 2),
            force_future=torch.randn(4, 8, 2),
            t_future=torch.arange(1, 9, dtype=torch.float32).unsqueeze(0),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        )
        shorter = sample.clone()
        shorter.force_future = shorter.force_future[:, :4]
        shorter.t_future = shorter.t_future[:, :4]
        for interpolation in ("linear", "hermite_cubic_backward"):
            with self.subTest(interpolation=interpolation):
                config = _base_config()
                config["model"]["use_relative_time"] = True
                config["model"]["relative_time_scale"] = 180.0
                config["solver"]["interpolation"] = interpolation
                model = build_model(config, 4, 2, 3).eval()
                with torch.no_grad():
                    short_prediction = model(shorter)
                    long_prediction = model(sample)
                torch.testing.assert_close(short_prediction, long_prediction[:, :4], rtol=1e-6, atol=1e-6)

    def test_relative_time_uses_fixed_scale_and_is_not_endpoint_clipped(self) -> None:
        config = _base_config()
        config["model"]["relative_time_scale"] = 180.0
        model = build_model(config, 4, 2, 3)
        feature = model.dynamics._relative_time_feature(torch.tensor(360.0), torch.zeros(4, 16))
        torch.testing.assert_close(feature, torch.full((4, 1), 2.0))

    def test_invalid_relative_time_scale_is_rejected(self) -> None:
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                config = _base_config()
                config["model"]["relative_time_scale"] = scale
                with self.assertRaisesRegex(ValueError, "relative_time_scale"):
                    build_model(config, 4, 2, 3)

    def test_active_model_only_imports_standard_ode_solver(self) -> None:
        source = (PROJECT_ROOT / "models/node2_model.py").read_text()
        module = ast.parse(source)
        solver_imports = [
            alias.name for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom) and node.module == "torchdiffeq"
            for alias in node.names
        ]
        self.assertEqual(solver_imports, ["odeint"])
        self.assertNotIn("adjoint", source)

    def test_dataset_time_scale_survives_model_overlay(self) -> None:
        for dataset, scale in (("issm", 180.0), ("anuga", 65.0)):
            with self.subTest(dataset=dataset):
                config = load_config_bundle([
                    PROJECT_ROOT / "configs/default.yaml",
                    PROJECT_ROOT / f"configs/datasets/{dataset}.yaml",
                    PROJECT_ROOT / f"configs/protocols/{dataset}/main.yaml",
                    PROJECT_ROOT / "configs/models/node2.yaml",
                ])
                self.assertEqual(config["model"]["relative_time_scale"], scale)
                self.assertTrue(config["model"]["use_relative_time"])
                self.assertEqual(config["evaluation"]["amp_mode"], "none")
                self.assertEqual(config["evaluation"]["checkpoint_metric"], "whole_rollout_norm_rmse")

    def test_history_and_future_variants_have_natural_training_anchors(self) -> None:
        h1 = enumerate_window_end_indices(240, 1, 180, 1)
        h8 = enumerate_window_end_indices(240, 8, 180, 1)
        k30 = enumerate_window_end_indices(240, 1, 30, 1)
        self.assertEqual(h1, list(range(60)))
        self.assertEqual(h8, list(range(7, 60)))
        self.assertEqual(k30, list(range(210)))


if __name__ == "__main__":
    unittest.main()
