"""Analytic and configuration regressions for output-space temporal losses."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from training.losses import (
    compute_temporal_consistency,
    rollout_mse,
    temporal_adjacent_increment,
    temporal_curvature,
    temporal_multiscale_rate,
    temporal_random_pair_increment,
    validate_temporal_consistency_config,
)
from utils.io import load_config_bundle


MODES = (
    "adjacent_increment",
    "random_pair_increment",
    "multiscale_rate",
    "rate_curvature",
    "hybrid",
)
COMPONENTS = {"total", "adjacent", "random_pair", "rate", "curvature"}


def trajectory(values):
    """Use multiple nodes and features so the temporal axis is unambiguous."""
    values = torch.as_tensor(values, dtype=torch.float64)
    return values.reshape(1, -1, 1).expand(2, -1, 3).clone()


def config(mode, **overrides):
    result = {"enabled": True, "mode": mode, "weight": 1.0, "penalty": "mse"}
    result.update(overrides)
    return result


class TemporalLossTest(unittest.TestCase):
    def setUp(self):
        self.t = torch.tensor([1.0, 2.0, 4.0, 7.0], dtype=torch.float64)

    def test_perfect_prediction_all_modes(self):
        true = trajectory(self.t.square())
        pred = true.clone().requires_grad_()
        for mode in MODES:
            with self.subTest(mode=mode):
                result = compute_temporal_consistency(pred, true, self.t, config(mode))
                self.assertEqual(set(result), COMPONENTS)
                for component in result.values():
                    self.assertEqual(component.item(), 0.0)
                    self.assertTrue(component.requires_grad)

    def test_constant_bias_changes_state_but_not_evolution(self):
        true = trajectory(self.t.square())
        pred = true + 5.0
        self.assertEqual(rollout_mse(pred, true).item(), 25.0)
        for mode in MODES:
            with self.subTest(mode=mode):
                result = compute_temporal_consistency(pred, true, self.t, config(mode))
                for component in result.values():
                    self.assertAlmostEqual(component.item(), 0.0, places=12)

    def test_linear_slope_mismatch_has_no_curvature(self):
        true = trajectory(self.t)
        pred = 2.0 * true
        self.assertGreater(rollout_mse(pred, true).item(), 0.0)
        self.assertGreater(temporal_adjacent_increment(pred, true).item(), 0.0)
        self.assertGreater(temporal_random_pair_increment(pred, true).item(), 0.0)
        self.assertEqual(temporal_multiscale_rate(pred, true, self.t).item(), 1.0)
        self.assertEqual(temporal_curvature(pred, true, self.t).item(), 0.0)

    def test_quadratic_curvature_matches_target_curvature(self):
        pred = trajectory(self.t.square())
        linear_true = trajectory(self.t)
        # The nonuniform finite difference is exact for a quadratic: d2(t2)/dt2=2.
        self.assertEqual(temporal_curvature(pred, linear_true, self.t).item(), 4.0)
        self.assertEqual(temporal_curvature(pred, pred, self.t).item(), 0.0)

    def test_nonuniform_times_drive_all_rate_modes(self):
        true = trajectory(self.t)
        pred = 2.0 * true
        for mode in ("multiscale_rate", "rate_curvature", "hybrid"):
            with self.subTest(mode=mode):
                result = compute_temporal_consistency(pred, true, self.t, config(mode))
                self.assertEqual(result["rate"].item(), 1.0)
                self.assertEqual(result["curvature"].item(), 0.0)
                expected_total = 1.0 + (14.0 / 3.0 if mode == "hybrid" else 0.0)
                self.assertAlmostEqual(result["total"].item(), expected_total)

    def test_shared_time_grid_supported_shapes(self):
        true = trajectory(self.t)
        pred = 2.0 * true
        grids = (self.t, self.t[None, :], self.t[None, :].repeat(3, 1))
        for grid in grids:
            with self.subTest(shape=tuple(grid.shape)):
                self.assertEqual(temporal_multiscale_rate(pred, true, grid).item(), 1.0)
                self.assertEqual(temporal_curvature(pred, true, grid).item(), 0.0)

    def test_invalid_time_grids_fail_clearly(self):
        pred = trajectory(self.t)
        invalid_grids = (
            torch.tensor([1.0, 2.0, 2.0, 3.0]),
            torch.tensor([1.0, 3.0, 2.0, 4.0]),
            torch.tensor([1.0, 2.0, float("nan"), 4.0]),
            torch.tensor([1.0, 2.0, 3.0, float("inf")]),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[1.0, 2.0, 4.0, 7.0], [1.0, 2.0, 4.0, 8.0]]),
            torch.ones(1, 1, 4),
        )
        for helper in (temporal_multiscale_rate, temporal_curvature):
            for grid in invalid_grids:
                with self.subTest(helper=helper.__name__, grid=grid):
                    with self.assertRaises(ValueError):
                        helper(pred, pred, grid)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            temporal_curvature(pred, pred, invalid_grids[0])

    def test_increment_modes_do_not_require_time(self):
        pred = trajectory(self.t)
        true = torch.zeros_like(pred)
        for mode in ("adjacent_increment", "random_pair_increment"):
            with self.subTest(mode=mode):
                result = compute_temporal_consistency(pred, true, None, config(mode))
                self.assertGreater(result["total"].item(), 0.0)

    def test_one_step_all_modes_return_differentiable_zero(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                pred = trajectory([4.0]).requires_grad_()
                result = compute_temporal_consistency(
                    pred, torch.zeros_like(pred), torch.tensor([1.0]), config(mode)
                )
                for component in result.values():
                    self.assertEqual(component.item(), 0.0)
                    self.assertTrue(component.requires_grad)
                result["total"].backward()
                torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_two_steps_keep_valid_terms_and_skip_curvature(self):
        pred = trajectory([1.0, 3.0]).requires_grad_()
        true = torch.zeros_like(pred)
        t = torch.tensor([1.0, 3.0])
        self.assertEqual(temporal_adjacent_increment(pred, true).item(), 4.0)
        self.assertEqual(temporal_random_pair_increment(pred, true).item(), 4.0)
        self.assertEqual(temporal_multiscale_rate(pred, true, t).item(), 1.0)
        curvature = temporal_curvature(pred, true, t)
        self.assertEqual(curvature.item(), 0.0)
        curvature.backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_unavailable_pairs_and_lags_return_differentiable_zero(self):
        pred = trajectory([1.0, 3.0]).requires_grad_()
        true = torch.zeros_like(pred)
        zero_losses = (
            temporal_random_pair_increment(pred, true, min_lag=2),
            temporal_multiscale_rate(pred, true, torch.tensor([1.0, 3.0]), lags=[3, 6]),
            temporal_multiscale_rate(
                pred, true, torch.tensor([1.0, 3.0]), lags=[1, 6], lag_weights=[0.0, 1.0]
            ),
        )
        for loss in zero_losses:
            self.assertEqual(loss.item(), 0.0)
            self.assertTrue(loss.requires_grad)
        sum(zero_losses).backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_pair_lag_filtering_and_no_rate_normalization(self):
        pred = trajectory([0.0, 1.0, 4.0, 9.0])
        true = torch.zeros_like(pred)
        # Lag two gives increments 4 and 8; lag three gives increment 9.
        self.assertEqual(
            temporal_random_pair_increment(pred, true, num_pairs=99, min_lag=2, max_lag=2).item(),
            40.0,
        )
        self.assertEqual(
            temporal_random_pair_increment(pred, true, num_pairs=99, min_lag=3, max_lag=3).item(),
            81.0,
        )
        self.assertAlmostEqual(
            temporal_random_pair_increment(pred, true, num_pairs=99, min_lag=2).item(),
            161.0 / 3.0,
        )

    def test_pair_sampling_without_replacement_caps_at_candidates(self):
        pred = trajectory([0.0, 1.0, 4.0, 9.0])
        true = torch.zeros_like(pred)
        expected = (1.0 + 16.0 + 81.0 + 9.0 + 64.0 + 25.0) / 6.0
        with torch.random.fork_rng():
            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                self.assertAlmostEqual(
                    temporal_random_pair_increment(pred, true, num_pairs=99).item(), expected
                )

    def test_pair_sampling_respects_torch_seed(self):
        pred = trajectory([0.0, 1.0, 4.0, 9.0])
        true = torch.zeros_like(pred)
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            first = temporal_random_pair_increment(pred, true, num_pairs=2)
            torch.manual_seed(1729)
            second = temporal_random_pair_increment(pred, true, num_pairs=2)
        torch.testing.assert_close(first, second)

    def test_pairs_are_shared_within_graph_and_independent_across_graphs(self):
        # A single increment yields nonzero gradients at exactly its two times.
        # Identical trajectories let us observe pair selection without exposing
        # or duplicating the sampler implementation in the test.
        node_batch = torch.arange(12).repeat_interleave(3)
        pred = torch.arange(8, dtype=torch.float64)[None, :, None].repeat(36, 1, 2)
        pred.requires_grad_()
        loss = compute_temporal_consistency(
            pred, torch.zeros_like(pred), None, config("random_pair_increment"),
            node_batch=node_batch, generator=torch.Generator().manual_seed(1729),
        )["total"]
        loss.backward()
        support = pred.grad.abs().sum(dim=2).ne(0)
        pairs_per_graph = []
        for graph in range(12):
            nodes = support[node_batch == graph]
            self.assertTrue(torch.equal(nodes, nodes[0:1].expand_as(nodes)))
            self.assertEqual(nodes[0].sum().item(), 2)
            pairs_per_graph.append(tuple(nodes[0].nonzero().flatten().tolist()))
        self.assertGreater(len(set(pairs_per_graph)), 1)

    def test_graph_sampling_preserves_node_average_and_no_replacement(self):
        # Graph IDs need not be contiguous or nodes adjacent. The three-node
        # graph has three times the weight of the one-node graph, as in MSE.
        node_batch = torch.tensor([7, 2, 7, 7])
        pred = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float64)[None, :, None]
        pred = pred * torch.tensor([1.0, 2.0, 3.0, 4.0])[:, None, None]
        pred.requires_grad_()
        # Sorted graph 2 selects (0,1),(0,3), graph 7 selects (2,3),(1,3).
        scores = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]])
        with patch("training.losses.torch.rand", return_value=scores):
            loss = temporal_random_pair_increment(
                pred, torch.zeros_like(pred), num_pairs=2, node_batch=node_batch,
            )
        expected = (26.0 * (25.0 + 64.0) + 4.0 * (1.0 + 81.0)) / 8.0
        self.assertEqual(loss.item(), expected)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_private_pair_generator_is_reproducible_and_isolates_global_rng(self):
        pred = trajectory([0.0, 1.0, 4.0, 9.0, 16.0])
        true = torch.zeros_like(pred)
        membership = torch.tensor([0, 1])
        first_gen = torch.Generator().manual_seed(42)
        second_gen = torch.Generator().manual_seed(42)
        initial_gen_state = first_gen.get_state().clone()
        global_state = torch.random.get_rng_state().clone()
        first_run = [
            compute_temporal_consistency(
                pred, true, None, config("random_pair_increment", random_pair={"num_pairs": 2}),
                node_batch=membership, generator=first_gen,
            )["total"]
            for _ in range(3)
        ]
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_state))
        self.assertFalse(torch.equal(first_gen.get_state(), initial_gen_state))
        with torch.random.fork_rng():
            # Model/dropout randomness cannot change the private pair stream.
            torch.manual_seed(123)
            torch.rand(100)
            second_run = [
                compute_temporal_consistency(
                    pred, true, None, config("random_pair_increment", random_pair={"num_pairs": 2}),
                    node_batch=membership, generator=second_gen,
                )["total"]
                for _ in range(3)
            ]
        torch.testing.assert_close(torch.stack(first_run), torch.stack(second_run))
        self.assertTrue(torch.equal(first_gen.get_state(), second_gen.get_state()))

    def test_omitted_membership_samples_one_graph(self):
        pred = trajectory([0.0, 1.0, 4.0, 9.0])
        true = torch.zeros_like(pred)
        implicit = temporal_random_pair_increment(
            pred, true, generator=torch.Generator().manual_seed(12),
        )
        explicit = temporal_random_pair_increment(
            pred, true, node_batch=torch.zeros(pred.shape[0], dtype=torch.long),
            generator=torch.Generator().manual_seed(12),
        )
        torch.testing.assert_close(implicit, explicit)

    def test_invalid_graph_membership_fails_clearly(self):
        pred = trajectory([0.0, 1.0, 4.0, 9.0])
        for membership in (torch.tensor([0]), torch.tensor([[0, 1]]),
                           torch.tensor([0.0, 1.0]), torch.tensor([-1, 0])):
            with self.subTest(membership=membership):
                with self.assertRaisesRegex(ValueError, "node_batch"):
                    temporal_random_pair_increment(pred, pred, node_batch=membership)

    def test_unavailable_or_disabled_pairs_do_not_advance_private_generator(self):
        pred = trajectory([1.0])
        generator = torch.Generator().manual_seed(42)
        before = generator.get_state().clone()
        temporal_random_pair_increment(pred, pred, generator=generator)
        pred = trajectory([0.0, 1.0])
        temporal_random_pair_increment(pred, pred, min_lag=3, generator=generator)
        compute_temporal_consistency(
            pred, pred, None, config("random_pair_increment", enabled=False),
            generator=generator,
        )
        self.assertTrue(torch.equal(generator.get_state(), before))

    def test_rate_equal_and_explicit_weights_average_per_scale(self):
        pred = trajectory([0.0, 1.0, 4.0, 9.0])
        true = torch.zeros_like(pred)
        t = torch.arange(4, dtype=torch.float64)
        # Lag one has squared rates 1,9,25; lag two has squared rates 4,16.
        equal = temporal_multiscale_rate(pred, true, t, lags=[1, 2])
        self.assertAlmostEqual(equal.item(), (35.0 / 3.0 + 10.0) / 2.0)
        weighted = temporal_multiscale_rate(pred, true, t, lags=[1, 2], lag_weights=[1.0, 3.0])
        self.assertAlmostEqual(weighted.item(), (35.0 / 3.0 + 30.0) / 4.0)

    def test_rate_weight_denominator_excludes_unavailable_lags(self):
        pred = trajectory([0.0, 1.0, 4.0, 9.0])
        true = torch.zeros_like(pred)
        rate = temporal_multiscale_rate(
            pred, true, torch.arange(4, dtype=torch.float64),
            lags=[1, 2, 12], lag_weights=[1.0, 3.0, 100.0],
        )
        self.assertAlmostEqual(rate.item(), (35.0 / 3.0 + 30.0) / 4.0)

    def test_composite_formulations_use_configured_weights(self):
        pred = trajectory(self.t.square())
        true = trajectory(self.t)
        common = {"rate": {"lags": [1, 2]}}
        rate_curvature = compute_temporal_consistency(
            pred, true, self.t,
            config("rate_curvature", **common,
                   rate_curvature={"rate_weight": 2.0, "curvature_weight": 3.0}),
        )
        torch.testing.assert_close(
            rate_curvature["total"],
            2.0 * rate_curvature["rate"] + 3.0 * rate_curvature["curvature"],
        )
        hybrid = compute_temporal_consistency(
            pred, true, self.t,
            config("hybrid", **common,
                   hybrid={"adjacent_weight": 2.0, "rate_weight": 3.0, "curvature_weight": 4.0}),
        )
        torch.testing.assert_close(
            hybrid["total"],
            2.0 * hybrid["adjacent"] + 3.0 * hybrid["rate"] + 4.0 * hybrid["curvature"],
        )
        self.assertEqual(hybrid["random_pair"].item(), 0.0)

    def test_dispatcher_returns_raw_loss_independent_of_overall_weight(self):
        pred = trajectory(self.t.square())
        true = trajectory(self.t)
        for mode in ("adjacent_increment", "multiscale_rate", "rate_curvature", "hybrid"):
            with self.subTest(mode=mode):
                first = compute_temporal_consistency(pred, true, self.t, config(mode, weight=0.0))
                second = compute_temporal_consistency(pred, true, self.t, config(mode, weight=7.0))
                torch.testing.assert_close(first["total"], second["total"])

    def test_optional_rmse_applies_only_to_temporal_penalty(self):
        pred = trajectory([0.0, 2.0, 4.0])
        true = torch.zeros_like(pred)
        result = compute_temporal_consistency(
            pred, true, None, config("adjacent_increment", penalty="rmse", rmse_eps=0.01)
        )
        self.assertAlmostEqual(result["total"].item(), (4.0 + 0.01) ** 0.5 - 0.1)
        self.assertAlmostEqual(rollout_mse(pred, true).item(), 20.0 / 3.0)
        perfect = temporal_adjacent_increment(pred, pred, penalty="rmse", rmse_eps=0.01)
        self.assertEqual(perfect.item(), 0.0)

    def test_rmse_perfect_match_has_zero_loss_and_finite_zero_gradient(self):
        for dtype in (torch.float32, torch.float64):
            for mode in MODES:
                with self.subTest(dtype=dtype, mode=mode):
                    true = trajectory(self.t.square()).to(dtype=dtype)
                    pred = true.clone().requires_grad_()
                    result = compute_temporal_consistency(
                        pred, true, self.t, config(mode, penalty="rmse"),
                    )
                    for component in result.values():
                        self.assertEqual(component.item(), 0.0)
                    result["total"].backward()
                    self.assertTrue(torch.isfinite(pred.grad).all())
                    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_rmse_epsilon_must_be_positive_for_direct_calls(self):
        pred = trajectory(self.t)
        for eps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(eps=eps):
                with self.assertRaisesRegex(ValueError, "rmse_eps"):
                    temporal_adjacent_increment(pred, pred, penalty="rmse", rmse_eps=eps)

    def test_all_modes_backpropagate_finite_nonzero_gradients(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                pred = trajectory(self.t.square()).requires_grad_()
                true = trajectory(self.t)
                result = compute_temporal_consistency(pred, true, self.t, config(mode))
                result["total"].backward()
                self.assertTrue(torch.isfinite(pred.grad).all())
                self.assertGreater(pred.grad.abs().sum().item(), 0.0)

    def test_disabled_or_absent_config_preserves_baseline_gradients_and_rng(self):
        true = trajectory(self.t)
        initial = trajectory(self.t.square())
        baseline = initial.clone().requires_grad_()
        baseline_loss = rollout_mse(baseline, true)
        baseline_loss.backward()
        for tc_cfg in ({}, {"enabled": False}, config("random_pair_increment", enabled=False)):
            with self.subTest(tc_cfg=tc_cfg):
                pred = initial.clone().requires_grad_()
                rng_before = torch.random.get_rng_state().clone()
                result = compute_temporal_consistency(pred, true, None, tc_cfg)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
                for value in result.values():
                    self.assertEqual(value.item(), 0.0)
                loss = rollout_mse(pred, true) + result["total"]
                loss.backward()
                torch.testing.assert_close(loss, baseline_loss, rtol=0, atol=0)
                torch.testing.assert_close(pred.grad, baseline.grad, rtol=0, atol=0)


class TemporalConfigTest(unittest.TestCase):
    def test_invalid_rmse_epsilon(self):
        for eps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(eps=eps):
                with self.assertRaisesRegex(ValueError, "rmse_eps"):
                    validate_temporal_consistency_config(
                        config("adjacent_increment", penalty="rmse", rmse_eps=eps)
                    )
        validate_temporal_consistency_config(config("adjacent_increment", rmse_eps=0.0))

    def test_invalid_general_config(self):
        invalid = (
            config("unsupported"),
            config("none"),
            config("adjacent_increment", weight=-1.0),
            config("adjacent_increment", penalty="mae"),
        )
        for tc_cfg in invalid:
            with self.subTest(tc_cfg=tc_cfg):
                with self.assertRaises(ValueError):
                    validate_temporal_consistency_config(tc_cfg)

    def test_invalid_random_pair_config(self):
        for values in ({"num_pairs": 0}, {"min_lag": 0}, {"min_lag": 3, "max_lag": 2}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_temporal_consistency_config(config("random_pair_increment", random_pair=values))

    def test_invalid_rate_config(self):
        invalid = (
            {"lags": []},
            {"lags": [0, 1]},
            {"lags": [-1, 2]},
            {"lags": [1.5, 2]},
            {"lags": [1, 2], "lag_weights": [1.0]},
            {"lags": [1, 2], "lag_weights": [-1.0, 1.0]},
            {"lags": [1, 2], "lag_weights": [0.0, 0.0]},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_temporal_consistency_config(config("multiscale_rate", rate=values))

    def test_invalid_composite_weights(self):
        for mode, key in (("rate_curvature", "rate_weight"), ("rate_curvature", "curvature_weight"),
                          ("hybrid", "adjacent_weight"), ("hybrid", "rate_weight"),
                          ("hybrid", "curvature_weight")):
            with self.subTest(mode=mode, key=key):
                with self.assertRaises(ValueError):
                    validate_temporal_consistency_config(config(mode, **{mode: {key: -1.0}}))

    def test_unused_subsections_are_ignored(self):
        validate_temporal_consistency_config(config(
            "adjacent_increment", random_pair={"num_pairs": 0},
            rate={"lags": []}, rate_curvature={"rate_weight": -1.0}, hybrid=None,
        ))
        validate_temporal_consistency_config({})
        validate_temporal_consistency_config({"enabled": False})

    def test_six_overlays_validate_and_merge_last(self):
        root = Path(__file__).resolve().parents[1]
        names = tuple(f"tc{index}.yaml" for index in range(6))
        for dataset in ("issm", "anuga"):
            overlay_dir = root / "configs" / "ablations" / dataset / "temporal_consistency"
            self.assertEqual({path.name for path in overlay_dir.glob("*.yaml")}, set(names))
            for name, mode in zip(names, ("none", *MODES)):
                with self.subTest(dataset=dataset, name=name):
                    merged = load_config_bundle([root / "configs" / "default.yaml", overlay_dir / name])
                    tc_cfg = merged["training"]["temporal_consistency"]
                    validate_temporal_consistency_config(tc_cfg)
                    self.assertEqual(tc_cfg["mode"], mode)
                    self.assertEqual(tc_cfg["enabled"], mode != "none")
                    self.assertEqual(tc_cfg["weight"], 1.0)
                    self.assertNotIn("balancing", tc_cfg)
                    if mode != "none":
                        self.assertEqual(tc_cfg["penalty"], "mse")
                    if mode == "random_pair_increment":
                        self.assertEqual(tc_cfg["random_pair"]["num_pairs"], 1)
                    if mode == "hybrid":
                        self.assertEqual(tc_cfg["rate"]["lags"], [3, 6, 12])
                        self.assertTrue(all(value == 1.0 for value in tc_cfg["hybrid"].values()))


if __name__ == "__main__":
    unittest.main()
