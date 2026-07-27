"""Lightweight smoke tests for the unified pipeline."""

from __future__ import annotations

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

from datasets import ADCIRCDataset, ANUGADataset, ISSMDataset
from datasets.normalization import FeatureNormalizer
from models import build_model
from models.continuous.node_latent_block import LatentNODEFunc
from models.continuous.node_latent_block_structured import StructuredLatentNODEFunc
from models.decoders.mlp_decoder import MLPDecoder
from models.encoders.history_encoder import HistoryEncoder
from training import Evaluator
from training.trainer import _truncate_future_horizon
from utils.anuga_postprocess import generate_anuga_flood_maps
from utils.eval_artifacts import (
    collect_full_rollout_prediction_bundle,
    collect_window_prediction_bundle,
    infer_state_channel_names,
    save_evaluation_artifacts,
    summarize_full_rollout_bundle,
    summarize_window_bundle,
)


def _base_config(model_name: str) -> dict:
    return {
        "model": {
            "name": model_name,
            "latent_dim": 16,
            "dropout": 0.0,
            "decoder_hidden_dims": [16],
            "decoder_activation": "gelu",
            "use_residual_decoder": True,
            "use_history_in_ode": True,
            "use_relative_time": True,
            "relative_time_mode": "normalized",
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
                "use_transformer_history": True,
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
            "cde_method": "rk4",
            "interpolation": "linear",
            "rtol": 1.0e-4,
            "atol": 1.0e-5,
            "use_adjoint": False,
            "ode_options": None,
            "cde_options": None,
        },
    }


def _node2_upgrade_off_config() -> dict:
    config = _base_config("node2")
    config["model"]["use_residual_decoder"] = False
    config["model"]["use_history_in_ode"] = False
    config["model"]["use_relative_time"] = False
    config["model"]["history_encoder"]["history_encoder_type"] = "lstm"
    config["model"]["history_encoder"]["use_transformer_history"] = False
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
                np.tile(np.array([-1.0, -1.0, 1.0], dtype=np.float32), (len(time), 1)),
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
        np.savez(root / "PIG_transient_m100_r080.npz", x_static=coords, force=issm_force, state=issm_state, edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64), times=time)

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

        sample = anuga[0]
        for model_name in ("node1", "node2", "ncde1"):
            model = build_model(_base_config(model_name), sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
            y_pred = model(sample)
            self.assertEqual(tuple(y_pred.shape), tuple(sample.y_future.shape))

    def test_node2_upgrade_flags_and_baseline_ablation(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        sample = anuga[0]

        upgraded = build_model(
            _base_config("node2"),
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
        )
        encoded = upgraded.history_encoder(sample.x_static, sample.state_hist, sample.force_hist, sample.edge_index)
        self.assertTrue(upgraded.history_encoder.use_transformer_history)
        self.assertEqual(upgraded.node2_vector_field_type, "v1_base")
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
        self.assertFalse(baseline_like.history_encoder.use_transformer_history)
        self.assertFalse(baseline_like.dynamics.use_history_in_ode)
        self.assertFalse(baseline_like.dynamics.use_relative_time)
        self.assertFalse(baseline_like.use_residual_decoder)
        self.assertEqual(tuple(baseline_like(sample).shape), tuple(sample.y_future.shape))

        structured_config = _base_config("node2")
        structured_config["model"]["node2_vector_field_type"] = "structured_v2"
        structured_config["model"]["structured_dynamics"] = {
            "use_f_local": True,
            "use_f_spatial": True,
            "use_f_forcing": True,
            "use_f_coupling": True,
        }
        structured = build_model(
            structured_config,
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
        )
        self.assertEqual(structured.node2_vector_field_type, "structured_v2")
        self.assertIsInstance(structured.dynamics, StructuredLatentNODEFunc)
        self.assertTrue(structured.dynamics.use_f_local)
        self.assertTrue(structured.dynamics.use_f_spatial)
        self.assertTrue(structured.dynamics.use_f_forcing)
        self.assertTrue(structured.dynamics.use_f_coupling)
        self.assertTrue(structured.dynamics.use_history_in_ode)
        self.assertTrue(structured.dynamics.use_relative_time)
        self.assertEqual(structured.dynamics.term_norm_kind, "none")
        self.assertEqual(structured.dynamics.fusion, "sum")
        self.assertEqual(tuple(structured(sample).shape), tuple(sample.y_future.shape))

        softmax_gated_config = deepcopy(structured_config)
        softmax_gated_config["model"]["structured_dynamics"].update(
            {
                "term_norm": "rmsnorm",
                "fusion": "softmax_gated",
                "gate_init": "active_mean",
            }
        )
        softmax_gated = build_model(
            softmax_gated_config,
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
        )
        self.assertEqual(softmax_gated.dynamics.term_norm_kind, "rmsnorm")
        self.assertEqual(softmax_gated.dynamics.fusion, "softmax_gated")
        self.assertIsNotNone(softmax_gated.dynamics.term_gate_logits)
        softmax_weights = torch.softmax(softmax_gated.dynamics.term_gate_logits, dim=0)
        self.assertTrue(torch.allclose(softmax_weights, torch.full((4,), 0.25)))
        self.assertEqual(tuple(softmax_gated(sample).shape), tuple(sample.y_future.shape))

        direct_gated_config = deepcopy(structured_config)
        direct_gated_config["model"]["structured_dynamics"].update(
            {
                "use_f_coupling": False,
                "term_norm": "layernorm",
                "fusion": "direct_gated",
                "gate_init": "active_mean",
            }
        )
        direct_gated = build_model(
            direct_gated_config,
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
        )
        self.assertEqual(direct_gated.dynamics.term_norm_kind, "layernorm")
        self.assertEqual(direct_gated.dynamics.fusion, "direct_gated")
        self.assertEqual(direct_gated.dynamics.num_active_terms, 3)
        self.assertIsNotNone(direct_gated.dynamics.term_gates)
        active_gates = direct_gated.dynamics.term_gates[direct_gated.dynamics.active_term_indices]
        self.assertTrue(torch.allclose(active_gates, torch.full((3,), 1.0 / 3.0)))
        self.assertEqual(float(direct_gated.dynamics.term_gates[-1].detach()), 0.0)
        self.assertEqual(tuple(direct_gated(sample).shape), tuple(sample.y_future.shape))

        zero_terms_config = deepcopy(structured_config)
        zero_terms_config["model"]["structured_dynamics"] = {
            "use_f_local": False,
            "use_f_spatial": False,
            "use_f_forcing": False,
            "use_f_coupling": False,
        }
        zero_terms = build_model(
            zero_terms_config,
            sample.x_static.shape[-1],
            sample.force_hist.shape[-1],
            sample.state_hist.shape[-1],
        )
        self.assertFalse(zero_terms.dynamics.use_f_local)
        self.assertFalse(zero_terms.dynamics.use_f_spatial)
        self.assertFalse(zero_terms.dynamics.use_f_forcing)
        self.assertFalse(zero_terms.dynamics.use_f_coupling)
        self.assertEqual(tuple(zero_terms(sample).shape), tuple(sample.y_future.shape))

        invalid_fusion_config = deepcopy(structured_config)
        invalid_fusion_config["model"]["structured_dynamics"]["fusion"] = "mystery"
        with self.assertRaises(ValueError):
            build_model(
                invalid_fusion_config,
                sample.x_static.shape[-1],
                sample.force_hist.shape[-1],
                sample.state_hist.shape[-1],
            )

    def test_chunked_transformer_history_matches_unchunked(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        sample = anuga[0]
        full_config = _base_config("node2")
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

    def test_node_models_integrate_from_zero_then_drop_initial_state(self) -> None:
        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        sample = anuga[0]
        expected_times = torch.cat([torch.zeros(1), sample.t_future.squeeze(0).float()])

        for model_name, patch_path in (
            ("node1", "models.node1_model.odeint"),
            ("node2", "models.node2_model.odeint"),
        ):
            with self.subTest(model=model_name):
                captured: dict[str, torch.Tensor] = {}

                def fake_odeint(func, y0, t, method=None, rtol=None, atol=None, options=None):
                    del func, method, rtol, atol, options
                    captured["times"] = t.detach().cpu()
                    return torch.stack([y0 + float(idx) for idx in range(t.numel())], dim=0)

                model = build_model(
                    _base_config(model_name),
                    sample.x_static.shape[-1],
                    sample.force_hist.shape[-1],
                    sample.state_hist.shape[-1],
                )
                with patch(patch_path, side_effect=fake_odeint):
                    y_pred = model(sample)

                self.assertEqual(tuple(y_pred.shape), tuple(sample.y_future.shape))
                self.assertTrue(torch.allclose(captured["times"], expected_times))

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
        window_metrics = evaluator.evaluate_loader(loader)
        rollout_metrics = evaluator.evaluate_full_rollout(anuga)
        shifted_rollout_metrics = evaluator.evaluate_full_rollout(anuga, start_t=3)

        self.assertIn("norm_rmse", window_metrics)
        self.assertIn("rmse", window_metrics)
        self.assertTrue(np.isfinite(window_metrics["norm_rmse"]))
        self.assertTrue(np.isfinite(window_metrics["rmse"]))
        self.assertIn("whole_rollout_norm_rmse", rollout_metrics)
        self.assertIn("whole_rollout_rmse", rollout_metrics)
        self.assertIn("horizon_norm_rmse_curve", rollout_metrics)
        self.assertIn("horizon_rmse_curve", rollout_metrics)
        self.assertEqual(len(shifted_rollout_metrics["horizon_rmse_curve"]), 2)
        with self.assertRaises(ValueError):
            evaluator.evaluate_full_rollout(anuga, start_t=1)

    def test_eval_artifacts_are_saved_for_window_and_rollout(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, data):
                return torch.zeros_like(data.y_future)

        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        anuga_normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        anuga.normalizer = anuga_normalizer
        channel_names = infer_state_channel_names("anuga", state_dim=anuga[0].y_future.shape[-1])

        loader = DataLoader(anuga, batch_size=2, shuffle=False)
        model = ZeroModel()

        window_bundle = collect_window_prediction_bundle(model, loader, anuga_normalizer, device=torch.device("cpu"))
        window_summary = summarize_window_bundle(window_bundle, channel_names)
        window_stem = self.root / "window_eval"
        window_artifacts = save_evaluation_artifacts(
            window_stem,
            bundle=window_bundle,
            summary=window_summary,
            dataset_name="anuga",
            split="train",
            mode="fixed_window",
            channel_names=channel_names,
            scenario_infos=anuga.scenario_infos,
        )
        self.assertTrue((self.root / "window_eval_metrics.json").exists())
        self.assertTrue((self.root / "window_eval_predictions.npz").exists())
        self.assertTrue((self.root / "window_eval_trajectory_metrics.png").exists())
        self.assertTrue((self.root / "window_eval_leadtime_metrics.png").exists())
        self.assertTrue((self.root / "window_eval_leadtime_metrics.csv").exists())
        self.assertTrue((self.root / "window_eval_leadtime_rmse_curve.csv").exists())
        self.assertTrue((self.root / "window_eval_leadtime_rmse_curve.npz").exists())
        self.assertTrue((self.root / "window_eval_leadtime_mae_curve.csv").exists())
        self.assertTrue((self.root / "window_eval_leadtime_mae_curve.npz").exists())
        self.assertIn("metrics_json", window_artifacts)
        self.assertIn("leadtime_metrics_csv", window_artifacts)
        self.assertIn("leadtime_metrics_plot", window_artifacts)
        self.assertIn("leadtime_rmse_curve_csv", window_artifacts)
        self.assertIn("leadtime_rmse_curve_npz", window_artifacts)
        self.assertIn("leadtime_mae_curve_csv", window_artifacts)
        self.assertIn("leadtime_mae_curve_npz", window_artifacts)
        self.assertGreater(len(window_summary["trajectory_metrics"]), 0)
        self.assertEqual(len(window_summary["leadtime_metrics"]), 2)

        rollout_bundle = collect_full_rollout_prediction_bundle(model, anuga, anuga_normalizer, device=torch.device("cpu"))
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
            start_t=3,
        )
        self.assertTrue(np.all(shifted_rollout_bundle["history_end_idx0"] == 3))
        self.assertEqual(int(shifted_rollout_bundle["trajectory_idx0"].min()), 4)

    def test_anuga_postprocess_generates_flood_maps(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, data):
                return torch.zeros_like(data.y_future)

        anuga = ANUGADataset([self.root / "sim_000_merged.npz"], history_len=3, future_len=2, split="train")
        anuga_normalizer = FeatureNormalizer.fit_from_trajectories(anuga.iter_trajectories())
        anuga.normalizer = anuga_normalizer
        channel_names = infer_state_channel_names("anuga", state_dim=anuga[0].y_future.shape[-1])

        loader = DataLoader(anuga, batch_size=2, shuffle=False)
        bundle = collect_window_prediction_bundle(ZeroModel(), loader, anuga_normalizer, device=torch.device("cpu"))
        summary = summarize_window_bundle(bundle, channel_names)
        artifact_stem = self.root / "window_eval"
        save_evaluation_artifacts(
            artifact_stem,
            bundle=bundle,
            summary=summary,
            dataset_name="anuga",
            split="train",
            mode="fixed_window",
            channel_names=channel_names,
            scenario_infos=anuga.scenario_infos,
        )

        flood_summary = generate_anuga_flood_maps(
            self.root / "window_eval_predictions.npz",
            output_dir=self.root / "window_eval_flood_maps",
            num_frames=2,
            trajectory_indices1=[4, 6],
        )

        self.assertEqual(len(flood_summary["scenarios"]), 1)
        scenario_summary = flood_summary["scenarios"][0]
        self.assertEqual(scenario_summary["scenario_id"], "sim_000")
        self.assertEqual(scenario_summary["predicted_trajectory_indices1"], [4, 5, 6])
        self.assertEqual(scenario_summary["chosen_trajectory_indices1"], [4, 6])
        self.assertTrue((self.root / "window_eval_flood_maps" / "summary.json").exists())
        self.assertTrue((self.root / "window_eval_flood_maps" / "sim_000" / "summary.json").exists())
        self.assertTrue((self.root / "window_eval_flood_maps" / "sim_000" / "depth_timeseries.npz").exists())
        self.assertEqual(len(scenario_summary["figure_paths"]), 2)

        depth_timeseries = np.load(self.root / "window_eval_flood_maps" / "sim_000" / "depth_timeseries.npz", allow_pickle=True)
        self.assertEqual(depth_timeseries["gt_depth"].shape, (6, 3))
        self.assertEqual(depth_timeseries["pred_depth"].shape, (6, 3))
        self.assertTrue(np.any(depth_timeseries["pred_available"] > 0))


if __name__ == "__main__":
    unittest.main()
