"""Lightweight smoke tests for the unified pipeline."""

from __future__ import annotations

import json
import logging
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from datasets import ADCIRCDataset, ANUGADataset, ISSMDataset
from datasets.normalization import FeatureNormalizer
from models import build_model
from models.continuous.node_latent_block import LatentNODEFunc
from models.decoders.mlp_decoder import MLPDecoder
from models.encoders.history_encoder import HistoryEncoder
from utils.checkpoint_evaluation import restore_checkpoint_splits
from training import Evaluator
from training.horizon_sampling import curriculum_horizon_max
from training.trainer import Trainer, _truncate_future_horizon
from utils.anuga_postprocess import generate_anuga_flood_maps
from utils.eval_artifacts import (
    collect_full_rollout_prediction_bundle,
    infer_state_channel_names,
    save_evaluation_artifacts,
    summarize_full_rollout_bundle,
    validate_full_rollout_bundle,
)


def _base_config() -> dict:
    return {
        "model": {
            "latent_dim": 16,
            "dropout": 0.0,
            "decoder_hidden_dims": [16],
            "decoder_activation": "gelu",
            "use_residual_decoder": True,
            "use_history_in_ode": True,
            "use_relative_time": True,
            "relative_time_scale": 8.0,
            "history_encoder": {
                "static_hidden_dims": [16],
                "static_embed_dim": 16,
                "gnn_hidden_dim": 16,
                "gnn_num_layers": 2,
                "gnn_type": "sage",
                "gnn_activation": "gelu",
                "lstm_hidden_dim": 16,
                "lstm_num_layers": 1,
                "history_encoder_type": "transformer",
                "history_transformer_num_layers": 1,
                "history_transformer_num_heads": 4,
                "history_transformer_ff_dim": 64,
                "history_transformer_dropout": 0.0,
                "history_use_positional_encoding": True,
                "history_context_pooling": "last",
                "dropout": 0.0,
            },
            "continuous": {
                "hidden_dim": 16,
                "num_layers": 2,
                "gnn_type": "sage",
                "activation": "softplus",
                "dropout": 0.0,
            },
        },
        "solver": {
            "ode_method": "midpoint",
            "interpolation": "linear",
            "rtol": 1.0e-4,
            "atol": 1.0e-5,
            "ode_options": None,
        },
    }


def _node2_upgrade_off_config() -> dict:
    config = _base_config()
    config["model"]["use_residual_decoder"] = False
    config["model"]["use_history_in_ode"] = False
    config["model"]["use_relative_time"] = False
    config["model"]["history_encoder"]["history_encoder_type"] = "lstm"
    config["model"]["history_encoder"]["history_use_positional_encoding"] = False
    return config


class SmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)

        time = 600.0 * np.arange(6, dtype=np.float32)
        x = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        y = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elevation = np.array([1.0, 1.2, 0.8], dtype=np.float32)
        friction = np.array([0.03, 0.04, 0.05], dtype=np.float32)
        volumes = np.array([[0, 1, 2]], dtype=np.int64)
        rain_rate = np.linspace(0.0, 1.0, len(time), dtype=np.float32)
        stage = np.stack([elevation + 0.1 * idx for idx in range(len(time))], axis=0).astype(np.float32)
        xmomentum = np.ones((len(time), len(x)), dtype=np.float32) * 0.2
        ymomentum = np.ones((len(time), len(x)), dtype=np.float32) * 0.1
        np.savez(
            root / "sim_000_merged.npz",
            time=time,
            rain_rate=rain_rate,
            rain_cum=np.cumsum(rain_rate),
            x=x,
            y=y,
            elevation=elevation,
            friction=friction,
            volumes=volumes,
            stage=stage,
            xmomentum=xmomentum,
            ymomentum=ymomentum,
        )

        coords = np.stack([x, y], axis=1)
        adcirc_force = np.stack(
            [
                np.tile(rain_rate[:, None], (1, len(x))),
                np.tile((rain_rate + 1.0)[:, None], (1, len(x))),
                np.tile((rain_rate + 2.0)[:, None], (1, len(x))),
            ],
            axis=-1,
        ).astype(np.float32)
        surge = np.stack([0.05 * idx * np.ones(len(x), dtype=np.float32) for idx in range(len(time))], axis=0)[..., None]
        np.savez(root / "adcirc_case.npz", x_static=coords, force=adcirc_force, state=surge, edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64), times=time)

        issm_force = np.stack(
            [
                np.tile(np.array([80.0, 80.0, 0.0], dtype=np.float32), (len(time), 1)),
                np.tile(np.array([0.5, 0.4, 0.6], dtype=np.float32), (len(time), 1)),
            ],
            axis=-1,
        ).astype(np.float32)
        issm_state = np.stack(
            [
                np.tile(np.array([1.0, 0.8, 0.6], dtype=np.float32), (len(time), 1)),
                np.tile(np.array([0.4, 0.5, 0.6], dtype=np.float32), (len(time), 1)),
                np.tile(np.array([100.0, 120.0, 140.0], dtype=np.float32), (len(time), 1)),
            ],
            axis=-1,
        ).astype(np.float32)
        issm_static = np.stack([
            np.array([-20.0, -30.0, -40.0], dtype=np.float32),
            np.array([80.0, 90.0, 100.0], dtype=np.float32),
            np.linalg.norm(issm_state[0, :, :2], axis=-1),
            np.array([1.0, 1.0, 0.0], dtype=np.float32),
        ], axis=-1)
        np.savez(root / "PIG_transient_m100_r080.npz", x_static=issm_static, force=issm_force, state=issm_state, edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64), times=time)

        self.root = root

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_datasets_and_models(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        anuga_normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        anuga.normalizer = anuga_normalizer

        adcirc = ADCIRCDataset([self.root / "adcirc_case.npz"], history_len=3, future_len=2, split="train")
        adcirc_normalizer = FeatureNormalizer.fit_from_trajectories(adcirc.iter_trajectories())
        adcirc.normalizer = adcirc_normalizer

        issm = ISSMDataset([self.root / "PIG_transient_m100_r080.npz"], history_len=3, future_len=2, split="train")
        issm_normalizer = FeatureNormalizer.fit_from_trajectories(issm.iter_trajectories())
        issm.normalizer = issm_normalizer

        self.assertEqual(anuga[0].state_hist.shape[1], 3)
        self.assertEqual(adcirc[0].force_future.shape[1], 2)
        self.assertEqual(issm[0].y_future.shape[-1], 3)
        self.assertTrue(torch.allclose(anuga[0].t_future[0], torch.tensor([1.0, 2.0], dtype=torch.float32)))

        for dataset in (anuga, adcirc, issm):
            with self.subTest(dataset=type(dataset).__name__):
                sample = dataset[0]
                model = build_model(_base_config(), sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
                y_pred = model(sample)
                self.assertEqual(tuple(y_pred.shape), tuple(sample.y_future.shape))
                self.assertTrue(torch.isfinite(y_pred).all())

    def test_node2_upgrade_flags_and_baseline_ablation(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        sample = anuga[0]

        upgraded = build_model(
            _base_config(),
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
        )
        encoded = upgraded.history_encoder(sample.x_static, sample.state_hist, sample.force_hist, sample.edge_index)
        self.assertEqual(upgraded.history_encoder.history_encoder_type, "transformer")
        self.assertIsInstance(upgraded.dynamics, LatentNODEFunc)
        self.assertTrue(upgraded.dynamics.use_history_in_ode)
        self.assertTrue(upgraded.dynamics.use_relative_time)
        self.assertTrue(upgraded.use_residual_decoder)
        self.assertEqual(tuple(encoded["step_embeds"].shape), (sample.x_static.shape[0], 3, 16))
        self.assertEqual(tuple(encoded["hist_context"].shape), (sample.x_static.shape[0], 16))
        self.assertEqual(tuple(upgraded(sample).shape), tuple(sample.y_future.shape))

        baseline_like = build_model(
            _node2_upgrade_off_config(),
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
        )
        self.assertEqual(baseline_like.history_encoder.history_encoder_type, "lstm")
        self.assertFalse(baseline_like.dynamics.use_history_in_ode)
        self.assertFalse(baseline_like.dynamics.use_relative_time)
        self.assertFalse(baseline_like.use_residual_decoder)
        self.assertEqual(tuple(baseline_like(sample).shape), tuple(sample.y_future.shape))

    def test_chunked_transformer_history_matches_unchunked(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        sample = anuga[0]
        full_config = _base_config()
        full_config["model"]["history_encoder"]["history_transformer_chunk_size"] = None
        chunked_config = deepcopy(full_config)
        chunked_config["model"]["history_encoder"]["history_transformer_chunk_size"] = 2

        full = HistoryEncoder(sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1], full_config)
        chunked = HistoryEncoder(
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
            chunked_config,
        )
        chunked.load_state_dict(full.state_dict())
        full.eval()
        chunked.eval()

        full_encoded = full(sample.x_static, sample.state_hist, sample.force_hist, sample.edge_index)
        chunked_encoded = chunked(sample.x_static, sample.state_hist, sample.force_hist, sample.edge_index)
        self.assertTrue(torch.allclose(full_encoded["hist_context"], chunked_encoded["hist_context"], atol=1.0e-6))

    def test_chunked_decoder_matches_unchunked(self) -> None:
        torch.manual_seed(0)
        full = MLPDecoder(8, 3, [8], chunk_size=None)
        chunked = MLPDecoder(8, 3, [8], chunk_size=5)
        chunked.load_state_dict(full.state_dict())
        x = torch.randn(7, 4, 8)

        self.assertTrue(torch.allclose(full(x), chunked(x), atol=1.0e-6))

    def test_node2_integrates_from_zero_then_drops_initial_state(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        sample = anuga[0]
        expected_times = torch.cat([torch.zeros(1), sample.t_future.squeeze(0).float()])
        captured: dict[str, torch.Tensor] = {}

        def fake_odeint(func, y0, t, method=None, rtol=None, atol=None, options=None):
            del func, method, rtol, atol, options
            captured["times"] = t.detach().cpu()
            return torch.stack([y0 + float(idx) for idx in range(t.numel())], dim=0)

        model = build_model(
            _base_config(),
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
        )
        with patch("models.node2_model.odeint", side_effect=fake_odeint):
            y_pred = model(sample)

        self.assertEqual(tuple(y_pred.shape), tuple(sample.y_future.shape))
        self.assertTrue(torch.allclose(captured["times"], expected_times))

    def test_node2_backward_reaches_encoder_dynamics_and_decoder(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        anuga.normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        batch = next(iter(DataLoader(anuga, batch_size=2, shuffle=False)))

        torch.manual_seed(7)
        config = _base_config()
        model = build_model(config, batch.x_static.shape[-1], batch.force_hist.shape[-1], batch.state_hist.shape[-1])
        prediction = model(batch)
        loss = torch.nn.functional.mse_loss(prediction, batch.y_future)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for module in (model.history_encoder, model.init_mlp, model.dynamics, model.decoder):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients, type(module).__name__)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_residual_decoder_anchors_predictions_to_last_observed_state(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        sample = anuga[0]
        model = build_model(_base_config(), sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
        with torch.no_grad():
            for parameter in model.decoder.parameters():
                parameter.zero_()
            prediction = model(sample)
        expected = sample.state_hist[:, -1:].expand_as(sample.y_future)
        self.assertTrue(torch.equal(prediction, expected))

    def test_horizon_curriculum_reaches_full_horizon_after_warmup(self) -> None:
        curriculum = {
            "k_min": 2,
            "target_k_max": 12,
            "enabled": True,
            "curriculum_epochs": 8,
            "warmup_fractions": [0.40, 0.55, 0.70, 0.85],
        }
        caps = [curriculum_horizon_max(epoch=epoch, **curriculum) for epoch in range(1, 10)]
        self.assertEqual(caps, [6, 6, 8, 8, 9, 9, 11, 11, 12])
        self.assertEqual(curriculum_horizon_max(epoch=1, **{**curriculum, "enabled": False}), 12)

    def test_training_saves_and_reloads_checkpoint_with_curriculum(self) -> None:
        torch.manual_seed(7)
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        anuga.normalizer = normalizer
        loader = DataLoader(anuga, batch_size=2, shuffle=False)
        val_path = self.root / "sim_val_merged.npz"
        test_path = self.root / "sim_test_merged.npz"
        shutil.copyfile(self.root / "sim_000_merged.npz", val_path)
        shutil.copyfile(self.root / "sim_000_merged.npz", test_path)
        val_dataset = ANUGADataset([val_path], history_len=3, future_len=2, split="val", normalizer=normalizer)
        test_dataset = ANUGADataset([test_path], history_len=3, future_len=2, split="test", normalizer=normalizer)
        sample = anuga[0]
        config = _base_config()
        config.update({
            "dataset": {"name": "anuga", "data_dir": str(self.root), "history_len": 3, "future_len": 2},
            "training": {
                "epochs": 1,
                "lr": 1.0e-3,
                "max_grad_norm": 1.0,
                "train_horizon_min": 1,
                "train_horizon_max": 2,
                "train_horizon_curriculum": {
                    "enabled": True,
                    "epochs": 4,
                    "warmup_fractions": [0.4, 0.55, 0.7, 0.85],
                },
            },
            "evaluation": {"known_steps": 3, "checkpoint_metric": "whole_rollout_norm_rmse", "amp_mode": "none"},
            "amp": {"mode": "none"},
        })
        model = build_model(config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
        initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        output_dir = self.root / "training"
        trainer = Trainer(
            model, loader, anuga, val_dataset, test_dataset, normalizer,
            config, torch.device("cpu"), output_dir, logging.getLogger(__name__),
        )
        summary = trainer.fit()
        self.assertEqual(summary["best_epoch"], 1)
        self.assertTrue(np.isfinite(summary["best_metric"]))
        self.assertTrue(np.isfinite(summary["test_rollout"]["whole_rollout_rmse"]))
        self.assertTrue((output_dir / "best.pt").exists())
        checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
        self.assertEqual(checkpoint["split_manifest"], {
            "train": ["sim_000_merged.npz"], "val": [val_path.name], "test": [test_path.name],
        })
        self.assertEqual(json.loads((output_dir / "split_files.json").read_text()), checkpoint["split_manifest"])
        self.assertEqual(checkpoint["config"], config)
        self.assertTrue((output_dir / "final_metrics.json").exists())
        history = json.loads((output_dir / "history.json").read_text())
        self.assertEqual(history[0]["train"]["train_horizon_max"], 1.0)
        self.assertTrue(any(not torch.equal(initial_parameters[name], parameter) for name, parameter in model.named_parameters()))

    def test_temporal_training_checkpoint_smoke_for_anuga_and_issm(self) -> None:
        # Give the synthetic ISSM reference genuine evolution, so TC is tested
        # with nonconstant targets as well as nonconstant model predictions.
        issm_path = self.root / "PIG_transient_m100_r080.npz"
        with np.load(issm_path) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["state"] = payload["state"] + np.arange(6, dtype=np.float32)[:, None, None] * 0.1
        np.savez(issm_path, **payload)

        for dataset_name, dataset_type, train_path in (
            ("anuga", ANUGADataset, self.root / "sim_000_merged.npz"),
            ("issm", ISSMDataset, issm_path),
        ):
            for mode in ("none", "random_pair_increment", "multiscale_rate"):
                with self.subTest(dataset=dataset_name, mode=mode):
                    torch.manual_seed(17)
                    train_dataset = dataset_type([train_path], history_len=2, future_len=3, split="train")
                    normalizer = FeatureNormalizer.fit_from_trajectories(train_dataset.iter_trajectories())
                    train_dataset.normalizer = normalizer
                    datasets = [train_dataset]
                    for split in ("val", "test"):
                        path = self.root / f"{dataset_name}_{mode}_{split}.npz"
                        shutil.copyfile(train_path, path)
                        datasets.append(dataset_type(
                            [path], history_len=2, future_len=3, split=split, normalizer=normalizer,
                        ))
                    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
                    config = _base_config()
                    config.update({
                        "seed": 17,
                        "dataset": {"name": dataset_name, "data_dir": str(self.root)},
                        "training": {
                            "epochs": 1, "lr": 1e-3, "max_grad_norm": 1.0,
                            "loss_scale_factor": 2.0,
                            "train_horizon_min": 3, "train_horizon_max": 3,
                            "train_horizon_curriculum": {"enabled": False},
                            "temporal_consistency": {"enabled": mode != "none", "mode": mode},
                        },
                        "evaluation": {"known_steps": 2, "checkpoint_metric": "whole_rollout_norm_rmse", "amp_mode": "none"},
                        "amp": {"mode": "none"},
                    })
                    sample = train_dataset[0]
                    dimensions = (sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
                    model = build_model(config, *dimensions)
                    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                    output_dir = self.root / f"training_{dataset_name}_{mode}"
                    trainer = Trainer(
                        model, loader, *datasets, normalizer, config, torch.device("cpu"),
                        output_dir, logging.getLogger(__name__),
                    )
                    summary = trainer.fit()
                    self.assertTrue(np.isfinite(summary["best_metric"]))
                    self.assertTrue(np.isfinite(summary["test_rollout"]["whole_rollout_norm_rmse"]))
                    self.assertTrue(any(
                        not torch.equal(initial[name], parameter) for name, parameter in model.named_parameters()
                    ))
                    metrics = json.loads((output_dir / "history.json").read_text())[0]["train"]
                    self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                    self.assertEqual(metrics["train_loss"], metrics["train_total_objective"])
                    if mode == "none":
                        self.assertEqual(metrics["train_tc_raw"], 0.0)
                    else:
                        self.assertGreater(metrics["train_tc_raw"], 0.0)
                    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
                    reloaded = build_model(checkpoint["config"], *dimensions)
                    reloaded.load_state_dict(checkpoint["model_state"])
                    model.eval()
                    reloaded.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(model(sample), reloaded(sample), rtol=0, atol=0)

    def test_training_horizon_truncation_keeps_future_fields_aligned(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        batch = next(iter(DataLoader(anuga, batch_size=2, shuffle=False)))

        _truncate_future_horizon(batch, k_eff=1)

        self.assertEqual(batch.force_future.shape[1], 1)
        self.assertEqual(batch.y_future.shape[1], 1)
        self.assertEqual(batch.t_future.shape[1], 1)
        self.assertEqual(batch.future_idx.shape[1], 1)
        self.assertEqual(batch.future_time.shape[1], 1)

    def test_evaluator_reports_normalized_and_physical_metrics(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, data):
                return torch.zeros_like(data.y_future)

        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        anuga_normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        anuga.normalizer = anuga_normalizer

        loader = DataLoader(anuga, batch_size=1, shuffle=False)
        evaluator = Evaluator(ZeroModel(), anuga_normalizer, device=torch.device("cpu"))
        rollout_metrics = evaluator.evaluate_full_rollout(anuga, known_steps=3)
        shifted_rollout_metrics = evaluator.evaluate_full_rollout(anuga, known_steps=4)

        self.assertIn("whole_rollout_norm_rmse", rollout_metrics)
        self.assertIn("whole_rollout_rmse", rollout_metrics)
        self.assertIn("horizon_norm_rmse_curve", rollout_metrics)
        self.assertIn("horizon_rmse_curve", rollout_metrics)
        self.assertEqual(len(shifted_rollout_metrics["horizon_rmse_curve"]), 2)
        with self.assertRaises(ValueError):
            evaluator.evaluate_full_rollout(anuga, known_steps=2)

    def test_eval_artifacts_are_saved_for_rollout(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, data):
                return torch.zeros_like(data.y_future)

        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        anuga_normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        anuga.normalizer = anuga_normalizer
        channel_names = infer_state_channel_names("anuga", state_dim=anuga[0].y_future.shape[-1])

        loader = DataLoader(anuga, batch_size=2, shuffle=False)
        model = ZeroModel()

        rollout_bundle = collect_full_rollout_prediction_bundle(model, anuga, anuga_normalizer, device=torch.device("cpu"), known_steps=3)
        self.assertEqual(np.unique(rollout_bundle["trajectory_idx0"]).tolist(), [3, 4, 5])
        self.assertEqual(np.unique(rollout_bundle["horizon_idx0"]).tolist(), [0, 1, 2])
        rollout_coverage = validate_full_rollout_bundle(
            rollout_bundle,
            scenario_infos=anuga.scenario_infos,
            known_steps=3,
            node_counts=[3],
        )
        self.assertTrue(rollout_coverage["complete"])
        self.assertEqual(rollout_coverage["max_rollout_steps"], 3)
        rollout_summary = summarize_full_rollout_bundle(
            rollout_bundle,
            channel_names,
            max_lead_time_step=2,
            summary_method="node2_full_rollout",
        )
        rollout_stem = self.root / "rollout_eval"
        rollout_artifacts = save_evaluation_artifacts(
            rollout_stem,
            bundle=rollout_bundle,
            summary=rollout_summary,
            dataset_name="anuga",
            split="train",
            mode="full_rollout",
            channel_names=channel_names,
            scenario_infos=anuga.scenario_infos,
            metadata={
                "known_steps": 3,
                "rollout_start_idx0": 2,
                "rollout_start_idx1": 3,
                "eval_history_len": 3,
                "train_config_history_len": 3,
                "train_config_future_len": 2,
            },
        )
        self.assertTrue((self.root / "rollout_eval_metrics.json").exists())
        self.assertTrue((self.root / "rollout_eval_predictions_meta.json").exists())
        self.assertTrue((self.root / "rollout_eval_metric_table.csv").exists())
        self.assertTrue((self.root / "rollout_eval_summary_table.csv").exists())
        self.assertTrue((self.root / "rollout_eval_leadtime_metrics.png").exists())
        self.assertTrue((self.root / "rollout_eval_leadtime_metrics.csv").exists())
        self.assertTrue((self.root / "rollout_eval_leadtime_rmse_curve.csv").exists())
        self.assertTrue((self.root / "rollout_eval_leadtime_rmse_curve.npz").exists())
        self.assertTrue((self.root / "rollout_eval_leadtime_mae_curve.csv").exists())
        self.assertTrue((self.root / "rollout_eval_leadtime_mae_curve.npz").exists())
        self.assertIn("metrics_json", rollout_artifacts)
        self.assertIn("summary_table_csv", rollout_artifacts)
        self.assertIn("leadtime_metrics_csv", rollout_artifacts)
        self.assertIn("leadtime_metrics_plot", rollout_artifacts)
        self.assertIn("leadtime_rmse_curve_csv", rollout_artifacts)
        self.assertIn("leadtime_rmse_curve_npz", rollout_artifacts)
        self.assertIn("leadtime_mae_curve_csv", rollout_artifacts)
        self.assertIn("leadtime_mae_curve_npz", rollout_artifacts)
        self.assertIn("whole_rollout_rmse", rollout_summary["metrics"])
        self.assertIn("final_step_rmse", rollout_summary["metrics"])
        self.assertEqual(len(rollout_summary["leadtime_metrics"]), 2)
        self.assertEqual(
            [row["summary_row"] for row in rollout_summary["summary_table"]],
            ["min_lead_time", "max_lead_time", "avg_over_lead_time"],
        )
        self.assertEqual(rollout_summary["summary_table"][0]["method"], "node2_full_rollout")
        self.assertEqual(rollout_summary["summary_table"][0]["lead_time_step"], 1)
        self.assertEqual(rollout_summary["summary_table"][1]["lead_time_step"], 2)

        shifted_rollout_bundle = collect_full_rollout_prediction_bundle(
            model,
            anuga,
            anuga_normalizer,
            device=torch.device("cpu"),
            known_steps=4,
        )
        self.assertTrue(np.all(shifted_rollout_bundle["history_end_idx0"] == 3))
        self.assertEqual(int(shifted_rollout_bundle["trajectory_idx0"].min()), 4)
        self.assertEqual(int(shifted_rollout_bundle["trajectory_idx0"].max()), 5)
        self.assertEqual(shifted_rollout_bundle["pred_phys"].shape, (3 * 2, 3))
        shifted_coverage = validate_full_rollout_bundle(
            shifted_rollout_bundle,
            scenario_infos=anuga.scenario_infos,
            known_steps=4,
            node_counts=[3],
        )
        self.assertEqual(shifted_coverage["max_rollout_steps"], 2)

    def test_anuga_postprocess_generates_flood_maps(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, data):
                return torch.zeros_like(data.y_future)

        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        anuga_normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        anuga.normalizer = anuga_normalizer
        channel_names = infer_state_channel_names("anuga", state_dim=anuga[0].y_future.shape[-1])

        loader = DataLoader(anuga, batch_size=2, shuffle=False)
        bundle = collect_full_rollout_prediction_bundle(ZeroModel(), anuga, anuga_normalizer, device=torch.device("cpu"), known_steps=3)
        summary = summarize_full_rollout_bundle(bundle, channel_names)
        artifact_stem = self.root / "rollout_eval"
        save_evaluation_artifacts(
            artifact_stem,
            bundle=bundle,
            summary=summary,
            dataset_name="anuga",
            split="train",
            mode="full_rollout",
            channel_names=channel_names,
            scenario_infos=anuga.scenario_infos,
        )

        flood_summary = generate_anuga_flood_maps(
            self.root / "rollout_eval_predictions.npz",
            output_dir=self.root / "rollout_eval_flood_maps",
            num_frames=2,
            trajectory_indices1=[4, 6],
        )

        self.assertEqual(len(flood_summary["scenarios"]), 1)
        scenario_summary = flood_summary["scenarios"][0]
        self.assertEqual(scenario_summary["scenario_id"], "sim_000")
        self.assertEqual(scenario_summary["predicted_trajectory_indices1"], [4, 5, 6])
        self.assertEqual(scenario_summary["chosen_trajectory_indices1"], [4, 6])
        self.assertTrue((self.root / "rollout_eval_flood_maps" / "summary.json").exists())
        self.assertTrue((self.root / "rollout_eval_flood_maps" / "sim_000" / "summary.json").exists())
        self.assertTrue((self.root / "rollout_eval_flood_maps" / "sim_000" / "depth_timeseries.npz").exists())
        self.assertEqual(len(scenario_summary["figure_paths"]), 2)

        depth_timeseries = np.load(self.root / "rollout_eval_flood_maps" / "sim_000" / "depth_timeseries.npz", allow_pickle=True)
        self.assertEqual(depth_timeseries["gt_depth"].shape, (6, 3))
        self.assertEqual(depth_timeseries["pred_depth"].shape, (6, 3))
        self.assertTrue(np.any(depth_timeseries["pred_available"] > 0))

    def test_full_rollout_anuga_export_saves_every_future_step(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, data):
                return torch.zeros_like(data.y_future)

        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="test")
        normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        anuga.normalizer = normalizer
        channel_names = infer_state_channel_names("anuga", state_dim=3)
        bundle = collect_full_rollout_prediction_bundle(
            ZeroModel(),
            anuga,
            normalizer,
            device=torch.device("cpu"),
            known_steps=3,
        )
        summary = summarize_full_rollout_bundle(bundle, channel_names)
        artifact_stem = self.root / "full_rollout_eval"
        save_evaluation_artifacts(
            artifact_stem,
            bundle=bundle,
            summary=summary,
            dataset_name="anuga",
            split="test",
            mode="full_rollout",
            channel_names=channel_names,
            scenario_infos=anuga.scenario_infos,
            metadata={
                "known_steps": 3,
                "rollout_start_idx0": 2,
                "rollout_start_idx1": 3,
                "eval_history_len": 3,
                "train_config_history_len": 3,
                "train_config_future_len": 2,
            },
        )

        flood_summary = generate_anuga_flood_maps(
            self.root / "full_rollout_eval_predictions.npz",
            output_dir=self.root / "full_rollout_eval_flood_maps",
            num_frames=0,
        )
        scenario_summary = flood_summary["scenarios"][0]
        self.assertEqual(scenario_summary["known_steps"], 3)
        self.assertEqual(scenario_summary["rollout_num_steps"], 3)
        self.assertEqual(scenario_summary["predicted_trajectory_indices1"], [4, 5, 6])
        self.assertEqual(scenario_summary["figure_paths"], [])

        timeseries = np.load(scenario_summary["timeseries_npz"], allow_pickle=True)
        self.assertEqual(timeseries["gt_depth"].shape, (6, 3))
        self.assertEqual(timeseries["pred_depth"].shape, (6, 3))
        self.assertEqual(timeseries["gt_xmomentum"].shape, (6, 3))
        self.assertEqual(timeseries["pred_ymomentum"].shape, (6, 3))
        self.assertEqual(timeseries["pred_available"].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertTrue(np.all(timeseries["pred_count"][:3] == 0))
        self.assertTrue(np.all(timeseries["pred_count"][3:] == 1))
        self.assertEqual(timeseries["prediction_indices0"].tolist(), [3, 4, 5])
        self.assertEqual(int(timeseries["known_steps"][0]), 3)
        self.assertEqual(int(timeseries["rollout_num_steps"][0]), 3)

    def test_rollout_preserves_saved_split_order(self) -> None:
        first = self.root / "sim_000_merged.npz"
        second = self.root / "sim_001_merged.npz"
        second.write_bytes(first.read_bytes())
        val = self.root / "sim_val_merged.npz"
        test = self.root / "sim_test_merged.npz"
        shutil.copyfile(first, val)
        shutil.copyfile(first, test)
        manifest_path = self.root / "split_files.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "train": [second.name, first.name],
                    "val": [val.name],
                    "test": [test.name],
                },
                handle,
            )
        split_files = restore_checkpoint_splits({"split_manifest": json.loads(manifest_path.read_text())}, self.root)
        self.assertEqual(split_files["train"], [second, first])
        self.assertEqual(split_files["test"], [test])

    def test_node2_checkpoint_roundtrip_after_forward(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        sample = anuga[0]
        config = _base_config()
        dimensions = (sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
        model = build_model(config, *dimensions).eval()
        with torch.no_grad():
            expected = model(sample)
        checkpoint_path = self.root / "model.pt"
        torch.save({"config": config, "model_state": model.state_dict()}, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        restored = build_model(checkpoint["config"], *dimensions).eval()
        restored.load_state_dict(checkpoint["model_state"])
        with torch.no_grad():
            actual = restored(sample)
        self.assertTrue(torch.equal(expected, actual))
        with self.assertRaises(RuntimeError):
            restored.load_state_dict({})


if __name__ == "__main__":
    unittest.main()
