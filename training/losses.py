"""Rollout MSE and optional output-space temporal consistency losses."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral

import torch


def rollout_mse(y_pred: torch.Tensor, y_true: torch.Tensor, k_eff: int | None = None) -> torch.Tensor:
    """Compute rollout MSE over the first ``k_eff`` steps when provided."""

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch in rollout_mse: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}")
    if k_eff is not None:
        y_pred = y_pred[:, :k_eff, :]
        y_true = y_true[:, :k_eff, :]
    return torch.mean((y_pred - y_true) ** 2)


_TEMPORAL_MODES = {
    "none",
    "adjacent_increment",
    "random_pair_increment",
    "multiscale_rate",
    "rate_curvature",
    "hybrid",
}
_RATE_MODES = {"multiscale_rate", "rate_curvature", "hybrid"}


def _nonnegative_number(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _config_section(tc_cfg: Mapping, name: str) -> Mapping:
    section = tc_cfg.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"temporal_consistency.{name} must be a mapping")
    return section


def _default_rate_lags(mode: str) -> tuple[int, ...]:
    return (3, 6, 12) if mode == "hybrid" else (1, 3, 6, 12)


def validate_temporal_consistency_config(tc_cfg: dict) -> None:
    """Validate the TC subsection once at trainer initialization.

    Only the selected mode's subsections are used, so leftover fields from a
    recursively merged overlay cannot affect a different temporal formulation.
    """
    if not isinstance(tc_cfg, Mapping):
        raise ValueError("training.temporal_consistency must be a mapping")
    enabled = tc_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("temporal_consistency.enabled must be a boolean")
    mode = str(tc_cfg.get("mode", "none")).lower()
    if mode not in _TEMPORAL_MODES:
        raise ValueError(f"Unsupported temporal consistency mode: {mode}")
    if enabled and mode == "none":
        raise ValueError("Enabled temporal consistency requires a mode other than 'none'")
    _nonnegative_number(tc_cfg.get("weight", 1.0), "temporal_consistency.weight")
    penalty = str(tc_cfg.get("penalty", "mse")).lower()
    if penalty not in {"mse", "rmse"}:
        raise ValueError(f"Unsupported temporal consistency penalty: {penalty}")
    rmse_eps = _nonnegative_number(tc_cfg.get("rmse_eps", 1e-8), "temporal_consistency.rmse_eps")
    if penalty == "rmse" and rmse_eps == 0:
        raise ValueError("temporal_consistency.rmse_eps must be positive for RMSE")

    if mode == "random_pair_increment":
        pair_cfg = _config_section(tc_cfg, "random_pair")
        _positive_integer(pair_cfg.get("num_pairs", 1), "random_pair.num_pairs")
        min_lag = _positive_integer(pair_cfg.get("min_lag", 1), "random_pair.min_lag")
        max_lag = pair_cfg.get("max_lag")
        if max_lag is not None:
            max_lag = _positive_integer(max_lag, "random_pair.max_lag")
            if max_lag < min_lag:
                raise ValueError("random_pair.max_lag must be >= random_pair.min_lag")

    if mode in _RATE_MODES:
        rate_cfg = _config_section(tc_cfg, "rate")
        lags = rate_cfg.get("lags", _default_rate_lags(mode))
        if isinstance(lags, (str, bytes)) or not isinstance(lags, Sequence) or not lags:
            raise ValueError("rate.lags must be a non-empty sequence of positive integers")
        for lag in lags:
            _positive_integer(lag, "rate.lags entries")
        lag_weights = rate_cfg.get("lag_weights")
        if lag_weights is not None:
            if (
                isinstance(lag_weights, (str, bytes))
                or not isinstance(lag_weights, Sequence)
                or len(lag_weights) != len(lags)
            ):
                raise ValueError("rate.lag_weights must have the same length as rate.lags")
            weights = [_nonnegative_number(weight, "rate.lag_weights entries") for weight in lag_weights]
            if not any(weight > 0 for weight in weights):
                raise ValueError("rate.lag_weights must include at least one positive weight")

    if mode in {"rate_curvature", "hybrid"}:
        component_cfg = _config_section(tc_cfg, mode)
        names = ("rate_weight", "curvature_weight")
        if mode == "hybrid":
            names = ("adjacent_weight", *names)
        for name in names:
            _nonnegative_number(component_cfg.get(name, 1.0), f"{mode}.{name}")


def _zero_loss(y_pred: torch.Tensor) -> torch.Tensor:
    """Keep empty/unused temporal terms attached to the prediction graph."""
    return y_pred.sum() * 0.0


def _temporal_penalty(error: torch.Tensor, kind: str = "mse", eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean(error ** 2)
    if kind == "mse":
        return mse
    if kind == "rmse":
        eps = _nonnegative_number(eps, "rmse_eps")
        if eps == 0:
            raise ValueError("rmse_eps must be positive for RMSE")
        offset = mse.new_tensor(eps)
        return torch.sqrt(mse + offset) - torch.sqrt(offset)
    raise ValueError(f"Unsupported temporal consistency penalty: {kind}")


def _check_temporal_shapes(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    if y_pred.ndim != 3 or y_pred.shape != y_true.shape:
        raise ValueError(
            "Temporal losses require matching [nodes, time, features] tensors; "
            f"got {tuple(y_pred.shape)} and {tuple(y_true.shape)}"
        )


def _get_shared_time_grid(t_future: torch.Tensor | None, expected_k: int) -> torch.Tensor:
    """Return a finite, strictly increasing grid shared by every graph."""
    if not isinstance(t_future, torch.Tensor):
        raise ValueError("t_future is required as a tensor for rate and curvature temporal losses")
    if t_future.ndim == 1:
        t = t_future
    elif t_future.ndim == 2 and t_future.shape[0] > 0:
        t = t_future[0]
        if not torch.allclose(t_future, t.unsqueeze(0).expand_as(t_future)):
            raise ValueError("All batch rows of t_future must share the same temporal grid")
    else:
        raise ValueError("t_future must have shape [K], [1, K], or [B, K]")
    if t.numel() != expected_k:
        raise ValueError(f"t_future has {t.numel()} steps; expected {expected_k}")
    if not torch.isfinite(t).all():
        raise ValueError("temporal grid must contain only finite values")
    if not torch.all(t[1:] > t[:-1]):
        raise ValueError("temporal grid must be strictly increasing")
    return t


def temporal_adjacent_increment(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    penalty: str = "mse",
    rmse_eps: float = 1e-8,
) -> torch.Tensor:
    """Match local one-step increments along time dimension 1."""
    _check_temporal_shapes(y_pred, y_true)
    if y_pred.shape[1] < 2:
        return _zero_loss(y_pred)
    pred_delta = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    true_delta = y_true[:, 1:, :] - y_true[:, :-1, :]
    return _temporal_penalty(pred_delta - true_delta, penalty, rmse_eps)


def temporal_random_pair_increment(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_pairs: int = 1,
    min_lag: int = 1,
    max_lag: int | None = None,
    penalty: str = "mse",
    rmse_eps: float = 1e-8,
    *,
    node_batch: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Match sampled integrated changes without dividing by elapsed time.

    Each graph samples independently without replacement; its nodes share the
    selected pairs. ``node_batch`` maps nodes to graphs, or all nodes belong to
    one graph when omitted. The penalty still averages over node elements.

    Pass a generator on the prediction device to isolate sampling from model
    randomness. Direct calls without one use the standard global torch RNG.
    Hyperparameters are validated at trainer startup.
    """
    _check_temporal_shapes(y_pred, y_true)
    k = y_pred.shape[1]
    if k < 2:
        return _zero_loss(y_pred)
    pairs = torch.triu_indices(k, k, offset=1, device=y_pred.device)
    lags = pairs[1] - pairs[0]
    mask = lags >= min_lag
    if max_lag is not None:
        mask &= lags <= max_lag
    pairs = pairs[:, mask]
    num_candidates = pairs.shape[1]
    if num_candidates == 0:
        return _zero_loss(y_pred)
    if node_batch is None:
        node_graph = torch.zeros(y_pred.shape[0], device=y_pred.device, dtype=torch.long)
        num_graphs = 1
    else:
        if (
            not isinstance(node_batch, torch.Tensor)
            or node_batch.ndim != 1
            or node_batch.numel() != y_pred.shape[0]
            or node_batch.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise ValueError("node_batch must be an integer tensor of shape [nodes]")
        if torch.any(node_batch < 0):
            raise ValueError("node_batch graph IDs must be nonnegative")
        graph_ids, node_graph = torch.unique(
            node_batch.to(device=y_pred.device, dtype=torch.long), return_inverse=True
        )
        num_graphs = graph_ids.numel()

    # Independent random priorities are a vectorized permutation per graph.
    # Top-k selects distinct candidates, including when all are requested.
    scores = torch.rand(
        (num_graphs, num_candidates), device=y_pred.device, generator=generator,
        dtype=torch.float32,
    )
    selected = scores.topk(min(num_pairs, num_candidates), dim=1).indices
    i = pairs[0, selected][node_graph]
    j = pairs[1, selected][node_graph]
    nodes = torch.arange(y_pred.shape[0], device=y_pred.device)[:, None]
    pred_delta = y_pred[nodes, j, :] - y_pred[nodes, i, :]
    true_delta = y_true[nodes, j, :] - y_true[nodes, i, :]
    return _temporal_penalty(pred_delta - true_delta, penalty, rmse_eps)


def temporal_multiscale_rate(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    t_future: torch.Tensor | None,
    lags: Sequence[int] = (1, 3, 6, 12),
    lag_weights: Sequence[float] | None = None,
    penalty: str = "mse",
    rmse_eps: float = 1e-8,
) -> torch.Tensor:
    """Average separate rate penalties across available, weighted time scales."""
    _check_temporal_shapes(y_pred, y_true)
    k = y_pred.shape[1]
    zero = _zero_loss(y_pred)
    if k < 2:
        return zero
    weights = [1.0] * len(lags) if lag_weights is None else lag_weights
    valid_lags = [(lag, float(weight)) for lag, weight in zip(lags, weights) if lag < k and float(weight) > 0]
    if not valid_lags:
        return zero
    t = _get_shared_time_grid(t_future, k).to(device=y_pred.device, dtype=y_pred.dtype)
    weighted_loss = zero
    total_weight = 0.0
    for lag, weight in valid_lags:
        dt = t[lag:] - t[:-lag]
        if not torch.all(dt > 0):
            raise ValueError("temporal grid must be strictly increasing")
        dt = dt.view(1, -1, 1)
        pred_rate = (y_pred[:, lag:, :] - y_pred[:, :-lag, :]) / dt
        true_rate = (y_true[:, lag:, :] - y_true[:, :-lag, :]) / dt
        weighted_loss = weighted_loss + weight * _temporal_penalty(pred_rate - true_rate, penalty, rmse_eps)
        total_weight += weight
    return weighted_loss / total_weight


def temporal_curvature(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    t_future: torch.Tensor | None,
    penalty: str = "mse",
    rmse_eps: float = 1e-8,
) -> torch.Tensor:
    """Match target second derivatives, including on nonuniform time grids.

    This compares predicted and reference curvature; it does not smooth the
    prediction toward zero curvature.
    """
    _check_temporal_shapes(y_pred, y_true)
    k = y_pred.shape[1]
    if k < 3:
        return _zero_loss(y_pred)
    t = _get_shared_time_grid(t_future, k).to(device=y_pred.device, dtype=y_pred.dtype)
    dt = t[1:] - t[:-1]
    if not torch.all(dt > 0):
        raise ValueError("temporal grid must be strictly increasing")
    dt = dt.view(1, -1, 1)
    span = (t[2:] - t[:-2]).view(1, -1, 1)
    pred_rate = (y_pred[:, 1:, :] - y_pred[:, :-1, :]) / dt
    true_rate = (y_true[:, 1:, :] - y_true[:, :-1, :]) / dt
    pred_curvature = 2.0 * (pred_rate[:, 1:, :] - pred_rate[:, :-1, :]) / span
    true_curvature = 2.0 * (true_rate[:, 1:, :] - true_rate[:, :-1, :]) / span
    return _temporal_penalty(pred_curvature - true_curvature, penalty, rmse_eps)


def compute_temporal_consistency(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    t_future: torch.Tensor | None,
    tc_cfg: dict,
    *,
    node_batch: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Dispatch a validated TC subsection to one formulation.

    ``weight`` belongs to the trainer's state + weight * TC objective. Returned
    components and total are raw temporal losses with no automatic balancing.
    Call :func:`validate_temporal_consistency_config` once before training.
    """
    zero = _zero_loss(y_pred)
    result = dict.fromkeys(("total", "adjacent", "random_pair", "rate", "curvature"), zero)
    mode = str(tc_cfg.get("mode", "none")).lower()
    if mode not in _TEMPORAL_MODES:
        raise ValueError(f"Unsupported temporal consistency mode: {mode}")
    if not tc_cfg.get("enabled", False):
        return result
    if mode == "none":
        raise ValueError("Enabled temporal consistency requires a mode other than 'none'")

    penalty = str(tc_cfg.get("penalty", "mse")).lower()
    rmse_eps = float(tc_cfg.get("rmse_eps", 1e-8))
    if mode in {"adjacent_increment", "hybrid"}:
        result["adjacent"] = temporal_adjacent_increment(y_pred, y_true, penalty, rmse_eps)
    if mode == "random_pair_increment":
        pair_cfg = tc_cfg.get("random_pair", {})
        result["random_pair"] = temporal_random_pair_increment(
            y_pred,
            y_true,
            num_pairs=pair_cfg.get("num_pairs", 1),
            min_lag=pair_cfg.get("min_lag", 1),
            max_lag=pair_cfg.get("max_lag"),
            penalty=penalty,
            rmse_eps=rmse_eps,
            node_batch=node_batch,
            generator=generator,
        )
    if mode in _RATE_MODES:
        rate_cfg = tc_cfg.get("rate", {})
        result["rate"] = temporal_multiscale_rate(
            y_pred,
            y_true,
            t_future,
            lags=rate_cfg.get("lags", _default_rate_lags(mode)),
            lag_weights=rate_cfg.get("lag_weights"),
            penalty=penalty,
            rmse_eps=rmse_eps,
        )
    if mode in {"rate_curvature", "hybrid"}:
        result["curvature"] = temporal_curvature(y_pred, y_true, t_future, penalty, rmse_eps)

    if mode == "adjacent_increment":
        result["total"] = result["adjacent"]
    elif mode == "random_pair_increment":
        result["total"] = result["random_pair"]
    elif mode == "multiscale_rate":
        result["total"] = result["rate"]
    else:
        component_cfg = tc_cfg.get(mode, {})
        result["total"] = (
            float(component_cfg.get("rate_weight", 1.0)) * result["rate"]
            + float(component_cfg.get("curvature_weight", 1.0)) * result["curvature"]
        )
        if mode == "hybrid":
            result["total"] = result["total"] + float(component_cfg.get("adjacent_weight", 1.0)) * result["adjacent"]
    return result
