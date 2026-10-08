"""Lightweight stability checks; synthetic graphs do not establish training stability."""

from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data
from torchdiffeq import odeint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model
from models.common.gnn_blocks import GraphNetwork
from models.continuous.node_latent_block import LatentNODEFunc
from training.losses import rollout_mse
from training.trainer import Trainer, _clip_grad_norm_fp64
from utils.io import load_config_bundle


def issm_config(history_len=1):
    return load_config_bundle([
        ROOT / "configs/default.yaml",
        ROOT / "configs/datasets/issm.yaml",
        ROOT / "configs/protocols/issm/main.yaml",
        ROOT / "configs/models/node2.yaml",
        ROOT / f"configs/ablations/issm/history/h{history_len}.yaml",
    ])


class NumericalStabilityTest(unittest.TestCase):
    def test_zero_derivative_for_each_supported_graph_projection(self):
        edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        for gnn_type in ("sage", "gcn", "graphconv"):
            for num_layers in (1, 2):
                with self.subTest(gnn_type=gnn_type, num_layers=num_layers):
                    torch.manual_seed(42)
                    config = issm_config()
                    config["model"]["continuous"].update(
                        gnn_type=gnn_type, num_layers=num_layers,
                    )
                    dynamics = LatentNODEFunc(4, 2, 3, 5, config)
                    control = SimpleNamespace(evaluate=lambda t: torch.ones(4, 2) * (1 + t))
                    dynamics.set_context(edges, torch.randn(4, 3), control, torch.randn(4, 5))
                    for t in (0.0, 90.0, 180.0):
                        derivative = dynamics(torch.tensor(t), torch.randn(4, 4) * 1e6)
                        torch.testing.assert_close(
                            derivative, torch.zeros_like(derivative), rtol=0, atol=0,
                        )

    def test_issm_defaults_to_zero_without_protocol_overlay(self):
        for dataset in ("issm", "anuga", "adcirc"):
            with self.subTest(dataset=dataset):
                config = load_config_bundle([
                    ROOT / "configs/default.yaml",
                    ROOT / f"configs/datasets/{dataset}.yaml",
                    ROOT / "configs/models/node2.yaml",
                ])
                self.assertNotIn("zero_init_output", config["model"]["continuous"])
                model = build_model(config, 4, 2, 3)
                output_is_zero = all(
                    torch.count_nonzero(p).item() == 0
                    for p in model.dynamics.net.convs[-1].parameters()
                )
                self.assertEqual(output_is_zero, dataset == "issm")

    def test_disabled_or_missing_flag_preserves_original_random_initialization(self):
        for gnn_type in ("sage", "gcn", "graphconv"):
            config = issm_config()
            continuous = config["model"]["continuous"]
            continuous["gnn_type"] = gnn_type
            torch.manual_seed(42)
            reference = GraphNetwork(
                input_dim=4 + 2 + 3 + 5 + 1, hidden_dim=continuous["hidden_dim"],
                output_dim=4, num_layers=continuous["num_layers"], layer_type=gnn_type,
                activation=continuous["activation"], dropout=continuous["dropout"],
            )
            for enabled, dataset in ((False, "issm"), (None, "anuga"), (None, "adcirc"), (None, None)):
                if dataset is None:
                    config.pop("dataset", None)
                else:
                    config["dataset"]["name"] = dataset
                with self.subTest(gnn_type=gnn_type, zero_init_output=enabled, dataset=dataset):
                    if enabled is None:
                        continuous.pop("zero_init_output", None)
                    else:
                        continuous["zero_init_output"] = enabled
                    torch.manual_seed(42)
                    dynamics = LatentNODEFunc(4, 2, 3, 5, config)
                    for name, value in reference.state_dict().items():
                        torch.testing.assert_close(dynamics.net.state_dict()[name], value, rtol=0, atol=0)
                    self.assertGreater(
                        sum(float(p.detach().abs().sum()) for p in dynamics.net.convs[-1].parameters()),
                        0.0,
                    )

    def test_seed42_h1_h8_k180_initial_rollout_backward_and_checkpoint(self):
        # Preserve the formal model, solver, loss scaling, batch size and LR;
        # only the normalized input graphs are small synthetic fixtures.
        for history_len in (1, 8):
            with self.subTest(history_len=history_len):
                torch.manual_seed(42)
                config = issm_config(history_len)
                self.assertTrue(config["model"]["continuous"]["zero_init_output"])
                horizon = config["dataset"]["future_len"]
                samples = [
                    Data(
                        num_nodes=4,
                        x_static=torch.randn(4, 4),
                        state_hist=torch.randn(4, history_len, 3),
                        force_hist=torch.randn(4, history_len, 2),
                        force_future=torch.randn(4, horizon, 2),
                        y_future=torch.randn(4, horizon, 3),
                        t_future=torch.arange(1, horizon + 1, dtype=torch.float32).unsqueeze(0),
                        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
                    )
                    for _ in range(config["training"]["batch_size"])
                ]
                batch = Batch.from_data_list(samples)
                model = build_model(config, 4, 2, 3).train()

                def check_latent_rollout(func, z0, times, **kwargs):
                    latent = odeint(func, z0, times, **kwargs)
                    self.assertEqual(latent.shape[0], 181)
                    torch.testing.assert_close(
                        latent, z0.unsqueeze(0).expand_as(latent), rtol=0, atol=0,
                    )
                    return latent

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config["training"]["lr"],
                    weight_decay=config["training"]["weight_decay"],
                )
                # Check initialization and the state after each of three updates.
                for updates_done in range(4):
                    optimizer.zero_grad(set_to_none=True)
                    if updates_done == 0:
                        with patch("models.node2_model.odeint", side_effect=check_latent_rollout):
                            prediction = model(batch)
                    else:
                        prediction = model(batch)
                    self.assertTrue(torch.isfinite(prediction).all())
                    scale = config["training"]["loss_scale_factor"]
                    loss = rollout_mse(prediction.float() * scale, batch.y_future.float() * scale)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for parameter in model.parameters():
                        self.assertEqual(parameter.dtype, torch.float32)
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                            self.assertEqual(parameter.grad.dtype, torch.float32)
                    for module in (model.history_encoder, model.init_mlp, model.dynamics.net.convs[-1], model.decoder):
                        gradients = [p.grad for p in module.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)
                    for parameter in model.dynamics.net.convs[-1].parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
                    if updates_done > 0:
                        for conv in model.dynamics.net.convs[:-1]:
                            gradients = [p.grad for p in conv.parameters() if p.grad is not None]
                            self.assertTrue(gradients)
                            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)
                    if updates_done == 3:
                        break
                    norm = _clip_grad_norm_fp64(
                        list(model.parameters()), config["training"]["max_grad_norm"],
                    )
                    self.assertTrue(torch.isfinite(norm))
                    optimizer.step()
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                    self.assertGreater(
                        sum(float(p.detach().abs().sum()) for p in model.dynamics.net.convs[-1].parameters()),
                        0.0,
                    )

                checkpoint = io.BytesIO()
                torch.save({"model_state": model.state_dict()}, checkpoint)
                for enabled in (None, False, True):
                    with self.subTest(checkpoint_zero_init_output=enabled):
                        restore_config = deepcopy(config)
                        continuous = restore_config["model"]["continuous"]
                        if enabled is None:
                            continuous.pop("zero_init_output", None)
                        else:
                            continuous["zero_init_output"] = enabled
                        restored = build_model(restore_config, 4, 2, 3)
                        checkpoint.seek(0)
                        restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state"])
                        for name, value in model.state_dict().items():
                            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_fp64_norm_clips_large_finite_fp32_gradients(self):
        parameters = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
        parameters[0].grad = torch.tensor([3e20, 4e20])
        parameters[1].grad = torch.tensor([12e20])
        flat = torch.cat([p.grad for p in parameters])
        self.assertTrue(torch.isinf(torch.linalg.vector_norm(flat)))
        expected = torch.linalg.vector_norm(flat.double())
        pointers = [p.grad.data_ptr() for p in parameters]
        norm = _clip_grad_norm_fp64(parameters, 1.0)
        self.assertEqual(norm.dtype, torch.float64)
        torch.testing.assert_close(norm, expected)
        for parameter, pointer in zip(parameters, pointers):
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.grad.dtype, torch.float32)
            self.assertEqual(parameter.grad.data_ptr(), pointer)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        clipped = torch.cat([p.grad for p in parameters]).double()
        torch.testing.assert_close(clipped, flat.double() / expected, rtol=1e-6, atol=0)
        torch.testing.assert_close(torch.linalg.vector_norm(clipped), torch.tensor(1.0).double())

    def test_genuinely_nonfinite_gradients_raise_without_being_zeroed(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                model = torch.nn.Linear(2, 1)
                model.weight.grad = torch.full_like(model.weight, 1e20)
                model.bias.grad = torch.full_like(model.bias, bad)
                before = [p.grad.clone() for p in model.parameters()]
                trainer = Trainer.__new__(Trainer)
                trainer.model = model
                trainer.max_grad_norm = 1.0
                trainer.grad_clip_norm_dtype = "fp64"
                trainer.device = torch.device("cpu")
                with self.assertRaises(FloatingPointError):
                    trainer._check_and_clip_gradients(step=1, k_eff=180, t_future_max=180.0)
                for parameter, original in zip(model.parameters(), before):
                    torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0, equal_nan=True)

    def test_issm_overlays_preserve_stability_options(self):
        base = [
            ROOT / "configs/default.yaml",
            ROOT / "configs/datasets/issm.yaml",
            ROOT / "configs/protocols/issm/main.yaml",
            ROOT / "configs/models/node2.yaml",
        ]
        expected_continuous = load_config_bundle([base[-1]])["model"]["continuous"]
        expected_continuous["zero_init_output"] = True
        overlays = sorted((ROOT / "configs/ablations/issm").rglob("*.yaml"))
        for overlay in overlays:
            with self.subTest(overlay=overlay):
                config = load_config_bundle([
                    *base, overlay, ROOT / "configs/runtime/issm_fast.yaml",
                    ROOT / "configs/runtime/issm_derecho.yaml",
                ])
                self.assertEqual(config["model"]["continuous"], expected_continuous)
                self.assertEqual(config["model"]["relative_time_scale"], 180.0)
                self.assertEqual(config["training"]["grad_clip_norm_dtype"], "fp64")
                self.assertEqual(config["training"]["max_grad_norm"], 1.0)
                self.assertEqual(config["training"]["batch_size"], 4)
                self.assertEqual(config["training"]["grad_accum_steps"], 2)


if __name__ == "__main__":
    unittest.main()
