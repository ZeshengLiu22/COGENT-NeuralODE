"""Prediction collection, summaries, and reporting for evaluation scripts."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import save_json

SCENARIO_FIELDS = frozenset({"scenario_id", "sim_id", "known_steps", "history_len", "mesh_signature", "normalization_signature"})


def infer_state_channel_names(dataset_name: str, state_dim: int) -> list[str]:
    """Return human-readable state-channel names when the dataset has known semantics."""

    if dataset_name == "anuga" and state_dim == 3:
        return ["depth", "xmomentum", "ymomentum"]
    if dataset_name == "adcirc" and state_dim == 1:
        return ["surge"]
    if dataset_name == "issm" and state_dim == 3:
        return ["vx", "vy", "thickness"]
    return [f"state_ch{idx}" for idx in range(state_dim)]


def _summary_channel_aliases(channel_names: list[str]) -> list[str]:
    """Return compact channel aliases for summary-table columns."""

    alias_overrides = {
        "depth": "h",
        "xmomentum": "mx",
        "ymomentum": "my",
    }
    aliases: list[str] = []
    used: set[str] = set()
    for channel_name in channel_names:
        alias = alias_overrides.get(channel_name, re.sub(r"[^0-9a-zA-Z]+", "_", channel_name.strip().lower()).strip("_"))
        if not alias:
            alias = "state"
        if alias in used:
            suffix = 1
            candidate = f"{alias}_{suffix}"
            while candidate in used:
                suffix += 1
                candidate = f"{alias}_{suffix}"
            alias = candidate
        used.add(alias)
        aliases.append(alias)
    return aliases


def _truncate_index_rows(rows: list[dict[str, Any]], *, index1_key: str, max_index1: int | None) -> list[dict[str, Any]]:
    """Return the first ``max_index1`` 1-based rows for a curve-style metric table."""

    if max_index1 is None:
        return list(rows)

    max_index1 = int(max_index1)
    if max_index1 < 1:
        raise ValueError(f"max_index1 must be >= 1, got {max_index1}.")

    truncated = [row for row in rows if int(row[index1_key]) <= max_index1]
    if not truncated:
        raise ValueError(f"No rows remain after truncating to {index1_key} <= {max_index1}.")
    return truncated


def _build_leadtime_summary_rows(
    leadtime_rows: list[dict[str, Any]],
    channel_names: list[str],
    *,
    method: str,
    history_end_idx0: int,
) -> list[dict[str, Any]]:
    """Build Multi-H-style lead-1 / max-lead / avg-over-lead RMSE summary rows."""

    if not leadtime_rows:
        return []

    aliases = _summary_channel_aliases(channel_names)

    def _build_row(
        summary_row: str,
        *,
        source_row: dict[str, Any] | None = None,
        aggregate_rmse: float,
        channel_rmses: list[float],
        lead_time_step: int | str,
        target_step_index: int | str,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": method,
            "summary_row": summary_row,
            "lead_time_step": lead_time_step,
            "target_step_index": target_step_index,
        }
        for channel_index, alias in enumerate(aliases):
            if source_row is not None:
                row[f"rmse_{alias}"] = float(source_row[f"phys_rmse_ch{channel_index}"])
            else:
                row[f"rmse_{alias}"] = float(channel_rmses[channel_index])
        row["rmse_aggregate"] = float(source_row["phys_rmse"]) if source_row is not None else float(aggregate_rmse)
        return row

    first_row = leadtime_rows[0]
    last_row = leadtime_rows[-1]
    avg_channel_rmses = [
        float(np.nanmean([float(row[f"phys_rmse_ch{channel_index}"]) for row in leadtime_rows]))
        for channel_index in range(len(channel_names))
    ]
    avg_aggregate_rmse = float(np.nanmean([float(row["phys_rmse"]) for row in leadtime_rows]))

    return [
        _build_row(
            "min_lead_time",
            source_row=first_row,
            aggregate_rmse=0.0,
            channel_rmses=[],
            lead_time_step=int(first_row["leadtime_idx1"]),
            target_step_index=history_end_idx0 + int(first_row["leadtime_idx1"]),
        ),
        _build_row(
            "max_lead_time",
            source_row=last_row,
            aggregate_rmse=0.0,
            channel_rmses=[],
            lead_time_step=int(last_row["leadtime_idx1"]),
            target_step_index=history_end_idx0 + int(last_row["leadtime_idx1"]),
        ),
        _build_row(
            "avg_over_lead_time",
            source_row=None,
            aggregate_rmse=avg_aggregate_rmse,
            channel_rmses=avg_channel_rmses,
            lead_time_step="",
            target_step_index="",
        ),
    ]


def _mesh_signature(trajectory) -> str:
    """Identify node ordering and topology independently of normalization/paths."""
    digest = hashlib.sha256()
    for name in ("x_static", "edge_index", "edge_attr"):
        value = getattr(trajectory, name, None)
        digest.update(name.encode("ascii"))
        if value is not None:
            array = np.ascontiguousarray(value)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _empty_bundle() -> dict[str, list[np.ndarray]]:
    return {
        "pred_norm": [],
        "target_norm": [],
        "pred_phys": [],
        "target_phys": [],
        "sample_index": [],
        "scenario_index": [],
        "node_index": [],
        "trajectory_idx0": [],
        "trajectory_idx1": [],
        "horizon_idx0": [],
        "horizon_idx1": [],
        "trajectory_time": [],
        "history_end_idx0": [],
        "history_end_idx1": [],
        "history_end_time": [],
        "scenario_id": [],
        "sim_id": [],
        "known_steps": [],
        "history_len": [],
        "mesh_signature": [],
        "normalization_signature": [],
    }


def _append_prediction_rows(
    bundle: dict[str, list[np.ndarray]],
    *,
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    pred_phys: np.ndarray,
    target_phys: np.ndarray,
    sample_index: int,
    scenario_index: int,
    history_end_idx0: int,
    history_end_time: float,
    future_idx0: np.ndarray,
    future_time: np.ndarray,
) -> None:
    node_count, horizon_len, state_dim = pred_norm.shape
    flat_count = node_count * horizon_len
    horizon_idx0 = np.arange(horizon_len, dtype=np.int64)

    bundle["pred_norm"].append(pred_norm.reshape(flat_count, state_dim).astype(np.float32, copy=False))
    bundle["target_norm"].append(target_norm.reshape(flat_count, state_dim).astype(np.float32, copy=False))
    bundle["pred_phys"].append(pred_phys.reshape(flat_count, state_dim).astype(np.float32, copy=False))
    bundle["target_phys"].append(target_phys.reshape(flat_count, state_dim).astype(np.float32, copy=False))
    bundle["sample_index"].append(np.full(flat_count, sample_index, dtype=np.int64))
    bundle["scenario_index"].append(np.full(flat_count, scenario_index, dtype=np.int64))
    bundle["node_index"].append(np.repeat(np.arange(node_count, dtype=np.int64), horizon_len))
    bundle["trajectory_idx0"].append(np.tile(future_idx0.astype(np.int64, copy=False), node_count))
    bundle["trajectory_idx1"].append(np.tile(future_idx0.astype(np.int64, copy=False) + 1, node_count))
    bundle["horizon_idx0"].append(np.tile(horizon_idx0, node_count))
    bundle["horizon_idx1"].append(np.tile(horizon_idx0 + 1, node_count))
    bundle["trajectory_time"].append(np.tile(future_time.astype(np.float64, copy=False), node_count))
    bundle["history_end_idx0"].append(np.full(flat_count, history_end_idx0, dtype=np.int64))
    bundle["history_end_idx1"].append(np.full(flat_count, history_end_idx0 + 1, dtype=np.int64))
    bundle["history_end_time"].append(np.full(flat_count, history_end_time, dtype=np.float64))


def _finalize_bundle(bundle: dict[str, list[np.ndarray]], state_dim: int) -> dict[str, np.ndarray]:
    if not bundle["pred_phys"]:
        empty_2d = np.zeros((0, state_dim), dtype=np.float32)
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_f32 = np.zeros((0,), dtype=np.float32)
        return {
            "pred_norm": empty_2d,
            "target_norm": empty_2d.copy(),
            "pred_phys": empty_2d.copy(),
            "target_phys": empty_2d.copy(),
            "sample_index": empty_i64,
            "scenario_index": empty_i64.copy(),
            "node_index": empty_i64.copy(),
            "trajectory_idx0": empty_i64.copy(),
            "trajectory_idx1": empty_i64.copy(),
            "horizon_idx0": empty_i64.copy(),
            "horizon_idx1": empty_i64.copy(),
            "trajectory_time": empty_f32,
            "history_end_idx0": empty_i64.copy(),
            "history_end_idx1": empty_i64.copy(),
            "history_end_time": empty_f32.copy(),
        }
    return {key: np.concatenate(parts, axis=0) for key, parts in bundle.items()}


@torch.no_grad()
def collect_full_rollout_prediction_bundle(
    model: torch.nn.Module,
    dataset,
    normalizer,
    device: torch.device,
    *,
    known_steps: int,
) -> dict[str, np.ndarray]:
    """Collect full-rollout predictions and metadata in normalized and physical units."""

    model.eval()
    if not dataset.scenario_infos:
        raise ValueError("Rollout collection requires at least one scenario.")
    known_steps = int(known_steps)
    if known_steps < dataset.history_len:
        raise ValueError("known_steps must be >= history_len.")
    if any(known_steps >= int(info["length"]) for info in dataset.scenario_infos):
        raise ValueError("known_steps must be < every trajectory length.")
    start_t = known_steps - 1
    first_sample = dataset.get_rollout_data(0, start_t=start_t)
    state_dim = int(first_sample.y_future.shape[-1])
    bundle = _empty_bundle()
    state_statistics = np.concatenate((normalizer.state_mean.cpu().numpy(), normalizer.state_std.cpu().numpy()))
    normalization_signature = hashlib.sha256(np.ascontiguousarray(state_statistics).tobytes()).hexdigest()

    for sample_index in range(len(dataset.scenario_infos)):
        sample = dataset.get_rollout_data(sample_index, start_t=start_t).to(device)
        prediction_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        y_pred = prediction_model(sample)
        if not torch.isfinite(y_pred).all():
            raise FloatingPointError("Non-finite model outputs encountered during full-rollout prediction collection.")

        y_pred_norm = y_pred.float()
        y_true_norm = sample.y_future.float()
        y_pred_phys = normalizer.inverse_state(y_pred_norm)
        y_true_phys = normalizer.inverse_state(y_true_norm)

        info = dataset.scenario_infos[sample_index]
        trajectory = dataset._get_trajectory(sample_index, cache=dataset.cache_in_memory)
        mesh_signature = _mesh_signature(trajectory)
        for name, value in (
            ("scenario_id", str(info["scenario_id"])),
            ("sim_id", str(info.get("sim_id", ""))),
            ("known_steps", known_steps),
            ("history_len", int(dataset.history_len)),
            ("mesh_signature", mesh_signature),
            ("normalization_signature", normalization_signature),
        ):
            bundle[name].append(np.asarray([value]))
        _append_prediction_rows(
            bundle,
            pred_norm=y_pred_norm.detach().cpu().numpy(),
            target_norm=y_true_norm.detach().cpu().numpy(),
            pred_phys=y_pred_phys.detach().cpu().numpy(),
            target_phys=y_true_phys.detach().cpu().numpy(),
            sample_index=sample_index,
            scenario_index=int(sample.scenario_idx.item()),
            history_end_idx0=int(sample.t_idx.item()),
            history_end_time=float(sample.t_value.item()),
            future_idx0=sample.future_idx.detach().cpu().numpy().reshape(-1),
            future_time=sample.future_time.detach().cpu().numpy().reshape(-1),
        )

    return _finalize_bundle(bundle, state_dim=state_dim)


def validate_full_rollout_bundle(
    bundle: dict[str, np.ndarray],
    *,
    scenario_infos: list[dict[str, Any]],
    known_steps: int,
    node_counts: list[int],
) -> dict[str, Any]:
    """Require exactly one finite prediction for every scenario/node/future index."""
    if not scenario_infos or len(node_counts) != len(scenario_infos):
        raise ValueError("Expected nonempty scenarios and one node count per scenario.")
    required = set(_empty_bundle())
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Rollout bundle is missing required fields: {sorted(missing)}")
    row_count = len(bundle["scenario_index"])
    if any(len(bundle[key]) != row_count for key in required - SCENARIO_FIELDS):
        raise ValueError("Rollout row fields must share the same row count.")
    if any(len(bundle[key]) != len(scenario_infos) for key in SCENARIO_FIELDS):
        raise ValueError("Rollout scenario fields must contain one entry per scenario.")
    shape = bundle["pred_phys"].shape
    if len(shape) != 2 or shape[1] < 1:
        raise ValueError("Predictions must have shape [rows, channels].")
    for key in ("pred_phys", "target_phys", "pred_norm", "target_norm"):
        if bundle[key].shape != shape or not np.isfinite(bundle[key]).all():
            raise ValueError(f"Invalid shape or non-finite values in {key}.")
    for key in ("trajectory_time", "history_end_time"):
        if not np.isfinite(bundle[key]).all():
            raise ValueError(f"Non-finite actual times in {key}.")
    for key in ("scenario_index", "sample_index", "node_index", "trajectory_idx0", "trajectory_idx1",
                "horizon_idx0", "horizon_idx1", "history_end_idx0", "history_end_idx1", "history_len", "known_steps"):
        if not np.issubdtype(bundle[key].dtype, np.integer) or bundle[key].ndim != 1:
            raise ValueError(f"{key} must contain integer indices.")
    if set(np.unique(bundle["scenario_index"])) != set(range(len(scenario_infos))):
        raise ValueError("Scenario indices do not match the metadata scenarios.")
    ids = [str(info["scenario_id"]) for info in scenario_infos]
    if len(set(ids)) != len(ids):
        raise ValueError("Scenario IDs must be unique for unambiguous alignment.")
    rows = []
    for scenario_index, (info, node_count) in enumerate(zip(scenario_infos, node_counts)):
        total_steps, node_count = int(info["length"]), int(node_count)
        steps = total_steps - known_steps
        mask = bundle["scenario_index"] == scenario_index
        if steps < 1 or node_count < 1 or int(mask.sum()) != steps * node_count:
            raise ValueError(f"Scenario {info['scenario_id']} has incomplete rollout coverage.")
        nodes = bundle["node_index"][mask]
        absolute = bundle["trajectory_idx0"][mask]
        lead = bundle["horizon_idx0"][mask]
        if (np.any(nodes < 0) or np.any(nodes >= node_count)
                or np.any(absolute < known_steps) or np.any(absolute >= total_steps)
                or not np.array_equal(lead, absolute - known_steps)):
            raise ValueError(f"Scenario {info['scenario_id']} has inconsistent node/time/lead indices.")
        pairs = nodes * steps + lead
        if not np.array_equal(np.sort(pairs), np.arange(node_count * steps)):
            raise ValueError(f"Scenario {info['scenario_id']} has duplicate or missing node/time predictions.")
        for key, value in (("scenario_id", str(info["scenario_id"])),
                           ("sim_id", str(info.get("sim_id", ""))), ("known_steps", known_steps)):
            if bundle[key][scenario_index] != value:
                raise ValueError(f"Scenario {info['scenario_id']} has inconsistent {key}.")
        for key, value in (("sample_index", scenario_index), ("history_end_idx0", known_steps - 1),
                           ("history_end_idx1", known_steps)):
            if not np.all(bundle[key][mask] == value):
                raise ValueError(f"Scenario {info['scenario_id']} has inconsistent {key}.")
        history = int(bundle["history_len"][scenario_index])
        if not 1 <= history <= known_steps:
            raise ValueError("history_len must be positive and <= known_steps.")
        for key in ("mesh_signature", "normalization_signature"):
            if not re.fullmatch(r"[a-f0-9]{64}", str(bundle[key][scenario_index])):
                raise ValueError(f"Invalid or missing {key}.")
        if (not np.array_equal(bundle["trajectory_idx1"][mask], absolute + 1)
                or not np.array_equal(bundle["horizon_idx1"][mask], lead + 1)):
            raise ValueError("Inconsistent one-based indices.")
        order = np.argsort(pairs)
        times = bundle["trajectory_time"][mask][order].reshape(node_count, steps)
        anchors = bundle["history_end_time"][mask]
        if (not np.array_equal(times, np.broadcast_to(times[0], times.shape))
                or np.any(np.diff(times[0]) <= 0) or len(np.unique(anchors)) != 1
                or times[0, 0] <= anchors[0]):
            raise ValueError("Inconsistent or nonincreasing actual trajectory times.")
        rows.append({"scenario_index": scenario_index, "scenario_id": str(info["scenario_id"]),
                     "total_steps": total_steps, "known_steps": known_steps, "history_len": history,
                     "rollout_steps": steps, "node_count": node_count, "saved_rows": int(mask.sum()),
                     "first_prediction_idx0": known_steps, "last_prediction_idx0": total_steps - 1})
    return {"complete": True, "scenario_count": len(rows), "known_steps": known_steps,
            "min_rollout_steps": min(row["rollout_steps"] for row in rows),
            "max_rollout_steps": max(row["rollout_steps"] for row in rows), "scenarios": rows}


def _error_sums(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    sq_sum = np.square(diff).sum(axis=0)
    abs_sum = np.abs(diff).sum(axis=0)
    return sq_sum, abs_sum, int(diff.shape[0])


def _metric_rows_and_values(
    *,
    scope: str,
    pred_phys: np.ndarray,
    target_phys: np.ndarray,
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    channel_names: list[str],
    metric_prefix: str,
) -> tuple[dict[str, float | list[float]], list[dict[str, Any]]]:
    phys_sq_sum, phys_abs_sum, row_count = _error_sums(pred_phys, target_phys)
    norm_sq_sum, norm_abs_sum, _ = _error_sums(pred_norm, target_norm)
    state_dim = int(pred_phys.shape[1]) if pred_phys.ndim == 2 else len(channel_names)
    safe_count = max(float(row_count), 1.0)
    safe_total = max(float(row_count * state_dim), 1.0)

    phys_rmse_ch = np.sqrt(phys_sq_sum / safe_count)
    phys_mae_ch = phys_abs_sum / safe_count
    norm_rmse_ch = np.sqrt(norm_sq_sum / safe_count)
    norm_mae_ch = norm_abs_sum / safe_count

    metrics: dict[str, float | list[float]] = {
        f"{metric_prefix}rmse": float(np.sqrt(phys_sq_sum.sum() / safe_total)),
        f"{metric_prefix}mae": float(phys_abs_sum.sum() / safe_total),
        f"{metric_prefix}norm_rmse": float(np.sqrt(norm_sq_sum.sum() / safe_total)),
        f"{metric_prefix}norm_mae": float(norm_abs_sum.sum() / safe_total),
    }
    if channel_names == ["vx", "vy", "thickness"]:
        speed_diff = (np.linalg.norm(pred_phys[:, :2].astype(np.float64), axis=-1)
                      - np.linalg.norm(target_phys[:, :2].astype(np.float64), axis=-1))
        metrics[f"{metric_prefix}speed_rmse_m_per_yr"] = float(np.sqrt(np.square(speed_diff).sum() / safe_count))
        metrics[f"{metric_prefix}thickness_rmse_m"] = float(phys_rmse_ch[2])
    rows = [
        {
            "scope": scope,
            "channel": "aggregate",
            "count": int(row_count * state_dim),
            "phys_rmse": float(metrics[f"{metric_prefix}rmse"]),
            "phys_mae": float(metrics[f"{metric_prefix}mae"]),
            "norm_rmse": float(metrics[f"{metric_prefix}norm_rmse"]),
            "norm_mae": float(metrics[f"{metric_prefix}norm_mae"]),
        }
    ]

    for channel_index, channel_name in enumerate(channel_names):
        metrics[f"{metric_prefix}rmse_ch{channel_index}"] = float(phys_rmse_ch[channel_index])
        metrics[f"{metric_prefix}mae_ch{channel_index}"] = float(phys_mae_ch[channel_index])
        metrics[f"{metric_prefix}norm_rmse_ch{channel_index}"] = float(norm_rmse_ch[channel_index])
        metrics[f"{metric_prefix}norm_mae_ch{channel_index}"] = float(norm_mae_ch[channel_index])
        rows.append(
            {
                "scope": scope,
                "channel": channel_name,
                "count": int(row_count),
                "phys_rmse": float(phys_rmse_ch[channel_index]),
                "phys_mae": float(phys_mae_ch[channel_index]),
                "norm_rmse": float(norm_rmse_ch[channel_index]),
                "norm_mae": float(norm_mae_ch[channel_index]),
            }
        )
    return metrics, rows


def _overall_curve(diff: np.ndarray, index_values: np.ndarray) -> list[float]:
    if diff.shape[0] == 0:
        return []

    max_index = int(index_values.max())
    sq_sum = np.zeros((max_index + 1,), dtype=np.float64)
    count = np.zeros((max_index + 1,), dtype=np.float64)
    diff64 = diff.astype(np.float64)
    np.add.at(sq_sum, index_values, np.square(diff64).sum(axis=1))
    np.add.at(count, index_values, float(diff.shape[1]))
    available = count > 0
    return np.sqrt(sq_sum[available] / count[available]).tolist()


def _indexed_metric_rows(
    bundle: dict[str, np.ndarray],
    channel_names: list[str],
    *,
    index_values: np.ndarray,
    index0_key: str,
    index1_key: str,
    time_values: np.ndarray,
    mean_time_key: str,
) -> list[dict[str, Any]]:
    if bundle["pred_phys"].shape[0] == 0:
        return []

    state_dim = int(bundle["pred_phys"].shape[1])
    index_values = index_values.astype(np.int64, copy=False)
    max_index = int(index_values.max())

    phys_diff = bundle["pred_phys"].astype(np.float64) - bundle["target_phys"].astype(np.float64)
    norm_diff = bundle["pred_norm"].astype(np.float64) - bundle["target_norm"].astype(np.float64)

    phys_sq = np.zeros((max_index + 1, state_dim), dtype=np.float64)
    phys_abs = np.zeros((max_index + 1, state_dim), dtype=np.float64)
    norm_sq = np.zeros((max_index + 1, state_dim), dtype=np.float64)
    norm_abs = np.zeros((max_index + 1, state_dim), dtype=np.float64)
    count = np.zeros((max_index + 1,), dtype=np.float64)
    time_sum = np.zeros((max_index + 1,), dtype=np.float64)

    for channel_index in range(state_dim):
        np.add.at(phys_sq[:, channel_index], index_values, np.square(phys_diff[:, channel_index]))
        np.add.at(phys_abs[:, channel_index], index_values, np.abs(phys_diff[:, channel_index]))
        np.add.at(norm_sq[:, channel_index], index_values, np.square(norm_diff[:, channel_index]))
        np.add.at(norm_abs[:, channel_index], index_values, np.abs(norm_diff[:, channel_index]))
    np.add.at(count, index_values, 1.0)
    np.add.at(time_sum, index_values, time_values.astype(np.float64, copy=False))

    rows: list[dict[str, Any]] = []
    valid_indices = np.nonzero(count > 0.0)[0]
    for index_value in valid_indices.tolist():
        safe_count = float(max(count[index_value], 1.0))
        safe_total = float(max(count[index_value] * state_dim, 1.0))
        row = {
            index0_key: int(index_value),
            index1_key: int(index_value + 1),
            mean_time_key: float(time_sum[index_value] / safe_count),
            "count_per_channel": int(count[index_value]),
            "count_aggregate": int(count[index_value] * state_dim),
            "phys_rmse": float(np.sqrt(phys_sq[index_value].sum() / safe_total)),
            "phys_mae": float(phys_abs[index_value].sum() / safe_total),
            "norm_rmse": float(np.sqrt(norm_sq[index_value].sum() / safe_total)),
            "norm_mae": float(norm_abs[index_value].sum() / safe_total),
        }
        for channel_index in range(len(channel_names)):
            row[f"phys_rmse_ch{channel_index}"] = float(np.sqrt(phys_sq[index_value, channel_index] / safe_count))
            row[f"phys_mae_ch{channel_index}"] = float(phys_abs[index_value, channel_index] / safe_count)
            row[f"norm_rmse_ch{channel_index}"] = float(np.sqrt(norm_sq[index_value, channel_index] / safe_count))
            row[f"norm_mae_ch{channel_index}"] = float(norm_abs[index_value, channel_index] / safe_count)
        rows.append(row)
    return rows


def _trajectory_metric_rows(bundle: dict[str, np.ndarray], channel_names: list[str]) -> list[dict[str, Any]]:
    return _indexed_metric_rows(
        bundle,
        channel_names,
        index_values=bundle["trajectory_idx0"],
        index0_key="trajectory_idx0",
        index1_key="trajectory_idx1",
        time_values=bundle["trajectory_time"],
        mean_time_key="mean_time",
    )


def _leadtime_metric_rows(bundle: dict[str, np.ndarray], channel_names: list[str]) -> list[dict[str, Any]]:
    lead_times = bundle["trajectory_time"].astype(np.float64) - bundle["history_end_time"].astype(np.float64)
    return _indexed_metric_rows(
        bundle,
        channel_names,
        index_values=bundle["horizon_idx0"],
        index0_key="leadtime_idx0",
        index1_key="leadtime_idx1",
        time_values=lead_times,
        mean_time_key="mean_lead_time",
    )


def summarize_full_rollout_bundle(
    bundle: dict[str, np.ndarray],
    channel_names: list[str],
    *,
    max_lead_time_step: int | None = None,
    summary_method: str = "full_rollout",
) -> dict[str, Any]:
    """Summarize a full-rollout prediction bundle."""

    metrics, table_rows = _metric_rows_and_values(
        scope="whole_rollout",
        pred_phys=bundle["pred_phys"],
        target_phys=bundle["target_phys"],
        pred_norm=bundle["pred_norm"],
        target_norm=bundle["target_norm"],
        channel_names=channel_names,
        metric_prefix="whole_rollout_",
    )

    if bundle["sample_index"].shape[0]:
        last_horizon = np.full(int(bundle["sample_index"].max()) + 1, -1, dtype=np.int64)
        np.maximum.at(last_horizon, bundle["sample_index"], bundle["horizon_idx0"])
        final_mask = bundle["horizon_idx0"] == last_horizon[bundle["sample_index"]]
    else:
        final_mask = np.zeros((0,), dtype=bool)

    final_metrics, final_rows = _metric_rows_and_values(
        scope="final_step",
        pred_phys=bundle["pred_phys"][final_mask],
        target_phys=bundle["target_phys"][final_mask],
        pred_norm=bundle["pred_norm"][final_mask],
        target_norm=bundle["target_norm"][final_mask],
        channel_names=channel_names,
        metric_prefix="final_step_",
    )
    metrics.update(final_metrics)
    metrics["horizon_lead_steps"] = (np.unique(bundle["horizon_idx0"]) + 1).tolist()
    metrics["horizon_rmse_curve"] = _overall_curve(bundle["pred_phys"].astype(np.float64) - bundle["target_phys"].astype(np.float64), bundle["horizon_idx0"])
    metrics["horizon_norm_rmse_curve"] = _overall_curve(bundle["pred_norm"].astype(np.float64) - bundle["target_norm"].astype(np.float64), bundle["horizon_idx0"])
    leadtime_metrics = _truncate_index_rows(
        _leadtime_metric_rows(bundle, channel_names),
        index1_key="leadtime_idx1",
        max_index1=max_lead_time_step,
    )
    history_end_idx0 = int(bundle["history_end_idx0"].min()) if bundle["history_end_idx0"].size else 0
    return {
        "metrics": metrics,
        "metric_table": table_rows + final_rows,
        "trajectory_metrics": _trajectory_metric_rows(bundle, channel_names),
        "leadtime_metrics": leadtime_metrics,
        "summary_table": _build_leadtime_summary_rows(
            leadtime_metrics,
            channel_names,
            method=summary_method,
            history_end_idx0=history_end_idx0,
        ),
    }


def format_metric_table(rows: list[dict[str, Any]]) -> str:
    """Render a compact plain-text metric table for logger output."""

    if not rows:
        return "(no rows)"

    headers = ["scope", "channel", "count", "phys_rmse", "phys_mae", "norm_rmse", "norm_mae"]
    formatted_rows: list[list[str]] = []
    for row in rows:
        formatted_rows.append(
            [
                str(row["scope"]),
                str(row["channel"]),
                str(row["count"]),
                f"{float(row['phys_rmse']):.6f}",
                f"{float(row['phys_mae']):.6f}",
                f"{float(row['norm_rmse']):.6f}",
                f"{float(row['norm_mae']):.6f}",
            ]
        )

    widths = []
    for column_index, header in enumerate(headers):
        widths.append(max(len(header), *(len(row[column_index]) for row in formatted_rows)))

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in formatted_rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _curve_rows_from_index_rows(
    rows: list[dict[str, Any]],
    channel_names: list[str],
    *,
    index_key: str,
    time_key: str,
    metric_key: str,
    metric_channel_prefix: str,
    metric_output_prefix: str,
) -> list[dict[str, Any]]:
    curve_rows: list[dict[str, Any]] = []
    for row in rows:
        curve_row: dict[str, Any] = {
            index_key: int(row[index_key]),
            time_key: float(row[time_key]),
            f"{metric_output_prefix}_aggregate": float(row[metric_key]),
        }
        for channel_index, channel_name in enumerate(channel_names):
            curve_row[f"{metric_output_prefix}_{channel_name}"] = float(row[f"{metric_channel_prefix}{channel_index}"])
        curve_rows.append(curve_row)
    return curve_rows


def _save_curve_npz(
    path: Path,
    rows: list[dict[str, Any]],
    channel_names: list[str],
    *,
    index_key: str,
    time_key: str,
    metric_key: str,
    metric_channel_prefix: str,
    metric_output_prefix: str,
) -> None:
    if not rows:
        np.savez_compressed(
            path,
            channel_names=np.asarray(channel_names),
            **{
                index_key: np.zeros((0,), dtype=np.int64),
                time_key: np.zeros((0,), dtype=np.float64),
                f"{metric_output_prefix}_aggregate": np.zeros((0,), dtype=np.float64),
                **{f"{metric_output_prefix}_{channel_name}": np.zeros((0,), dtype=np.float64) for channel_name in channel_names},
            },
        )
        return

    payload: dict[str, np.ndarray] = {
        "channel_names": np.asarray(channel_names),
        index_key: np.asarray([int(row[index_key]) for row in rows], dtype=np.int64),
        time_key: np.asarray([float(row[time_key]) for row in rows], dtype=np.float64),
        f"{metric_output_prefix}_aggregate": np.asarray([float(row[metric_key]) for row in rows], dtype=np.float64),
    }
    for channel_index, channel_name in enumerate(channel_names):
        payload[f"{metric_output_prefix}_{channel_name}"] = np.asarray(
            [float(row[f"{metric_channel_prefix}{channel_index}"]) for row in rows],
            dtype=np.float64,
        )
    np.savez_compressed(path, **payload)


def _plot_index_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    channel_names: list[str],
    title: str,
    *,
    x_key: str,
    x_label: str,
    x_title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Metric")
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return

    x_values = [int(row[x_key]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    metrics_to_plot = [("RMSE", "phys_rmse", "phys_rmse_ch"), ("MAE", "phys_mae", "phys_mae_ch")]

    for axis, (ylabel, aggregate_key, channel_prefix) in zip(axes, metrics_to_plot):
        axis.plot(x_values, [float(row[aggregate_key]) for row in rows], linewidth=2.5, color="black", label="aggregate")
        for channel_index, channel_name in enumerate(channel_names):
            axis.plot(
                x_values,
                [float(row[f"{channel_prefix}{channel_index}"]) for row in rows],
                linewidth=1.8,
                label=channel_name,
            )
        axis.set_title(f"{ylabel} vs {x_title}")
        axis.set_xlabel(x_label)
        axis.set_ylabel(f"{ylabel} (physical units)")
        axis.grid(True, alpha=0.3)

    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_trajectory_metrics(path: Path, rows: list[dict[str, Any]], channel_names: list[str], title: str) -> None:
    _plot_index_metrics(
        path,
        rows,
        channel_names,
        title,
        x_key="trajectory_idx1",
        x_label="Trajectory Index (1-based)",
        x_title="Trajectory Index",
    )


def _plot_leadtime_metrics(path: Path, rows: list[dict[str, Any]], channel_names: list[str], title: str) -> None:
    _plot_index_metrics(
        path,
        rows,
        channel_names,
        title,
        x_key="leadtime_idx1",
        x_label="Lead Time Step (1-based)",
        x_title="Lead Time Step",
    )


def save_evaluation_artifacts(
    artifact_stem: str | Path,
    *,
    bundle: dict[str, np.ndarray],
    summary: dict[str, Any],
    dataset_name: str,
    split: str,
    mode: str,
    channel_names: list[str],
    scenario_infos: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist predictions, summaries, tables, and trajectory plots for evaluation."""

    if mode != "full_rollout":
        raise ValueError("Only full_rollout evaluation artifacts are supported.")
    artifact_stem = Path(artifact_stem)
    artifact_stem.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(f"{artifact_stem}_metrics.json")
    predictions_path = Path(f"{artifact_stem}_predictions.npz")
    predictions_meta_path = Path(f"{artifact_stem}_predictions_meta.json")
    metric_table_path = Path(f"{artifact_stem}_metric_table.csv")
    summary_table_path = Path(f"{artifact_stem}_summary_table.csv")
    trajectory_metrics_path = Path(f"{artifact_stem}_trajectory_metrics.csv")
    trajectory_plot_path = Path(f"{artifact_stem}_trajectory_metrics.png")
    leadtime_metrics_path = Path(f"{artifact_stem}_leadtime_metrics.csv")
    leadtime_plot_path = Path(f"{artifact_stem}_leadtime_metrics.png")
    leadtime_rmse_curve_csv_path = Path(f"{artifact_stem}_leadtime_rmse_curve.csv")
    leadtime_rmse_curve_npz_path = Path(f"{artifact_stem}_leadtime_rmse_curve.npz")
    leadtime_mae_curve_csv_path = Path(f"{artifact_stem}_leadtime_mae_curve.csv")
    leadtime_mae_curve_npz_path = Path(f"{artifact_stem}_leadtime_mae_curve.npz")

    validate_full_rollout_bundle(
        bundle, scenario_infos=scenario_infos, known_steps=int(bundle["known_steps"][0]),
        node_counts=[int(np.unique(bundle["node_index"][bundle["scenario_index"] == i]).size)
                     for i in range(len(scenario_infos))],
    )
    np.savez_compressed(predictions_path, **bundle)

    scenario_lookup = []
    for scenario_index, info in enumerate(scenario_infos):
        scenario_lookup.append(
            {
                "scenario_index": scenario_index,
                "scenario_id": info["scenario_id"],
                "sim_id": str(info.get("sim_id", "")),
                "known_steps": int(bundle["known_steps"][scenario_index]),
                "history_len": int(bundle["history_len"][scenario_index]),
                "node_count": int(np.unique(bundle["node_index"][bundle["scenario_index"] == scenario_index]).size),
                "mesh_signature": str(bundle["mesh_signature"][scenario_index]),
                "normalization_signature": str(bundle["normalization_signature"][scenario_index]),
                "length": info["length"],
                "path": str(info["path"]),
            }
        )
    save_json(
        predictions_meta_path,
        {
            "schema_version": 2,
            "dataset_name": dataset_name,
            "known_steps": int(bundle["known_steps"][0]),
            "history_len": int(bundle["history_len"][0]),
            "time_coordinate_source": "TrajectoryData.times; original adapter units",
            "time_semantics": "trajectory_time preserves the adapter-provided coordinate, never normalized relative solver time; ISSM cell trajectories use snapshot indices and ANUGA uses supplied simulation time",
            "split": split,
            "mode": mode,
            "channel_names": channel_names,
            "row_format": "Prediction row arrays describe one (scenario_index, node_index, trajectory_idx0) with all state channels across columns.",
            "scenario_fields": sorted(SCENARIO_FIELDS),
            "scenario_format": "scenario_id, sim_id, known_steps, history_len, mesh_signature and normalization_signature contain one entry per scenario, indexed by scenario_index.",
            "scenario_lookup": scenario_lookup,
            "evaluation_metadata": metadata or {},
        },
    )

    _write_csv(metric_table_path, summary["metric_table"])
    _write_csv(trajectory_metrics_path, summary["trajectory_metrics"])
    trajectory_title = f"{dataset_name.upper()} {split} {mode.replace('_', ' ').title()}"
    _plot_trajectory_metrics(
        trajectory_plot_path,
        summary["trajectory_metrics"],
        channel_names,
        title=trajectory_title,
    )

    artifact_paths = {
        "metrics_json": str(metrics_path),
        "predictions_npz": str(predictions_path),
        "predictions_meta_json": str(predictions_meta_path),
        "metric_table_csv": str(metric_table_path),
        "trajectory_metrics_csv": str(trajectory_metrics_path),
        "trajectory_metrics_plot": str(trajectory_plot_path),
    }
    if "summary_table" in summary:
        _write_csv(summary_table_path, summary["summary_table"])
        artifact_paths["summary_table_csv"] = str(summary_table_path)
    if "leadtime_metrics" in summary:
        _write_csv(leadtime_metrics_path, summary["leadtime_metrics"])
        _plot_leadtime_metrics(
            leadtime_plot_path,
            summary["leadtime_metrics"],
            channel_names,
            title=f"{dataset_name.upper()} {split} {mode.replace('_', ' ').title()} Lead Time",
        )
        rmse_curve_rows = _curve_rows_from_index_rows(
            summary["leadtime_metrics"],
            channel_names,
            index_key="leadtime_idx1",
            time_key="mean_lead_time",
            metric_key="phys_rmse",
            metric_channel_prefix="phys_rmse_ch",
            metric_output_prefix="rmse",
        )
        mae_curve_rows = _curve_rows_from_index_rows(
            summary["leadtime_metrics"],
            channel_names,
            index_key="leadtime_idx1",
            time_key="mean_lead_time",
            metric_key="phys_mae",
            metric_channel_prefix="phys_mae_ch",
            metric_output_prefix="mae",
        )
        _write_csv(leadtime_rmse_curve_csv_path, rmse_curve_rows)
        _write_csv(leadtime_mae_curve_csv_path, mae_curve_rows)
        _save_curve_npz(
            leadtime_rmse_curve_npz_path,
            summary["leadtime_metrics"],
            channel_names,
            index_key="leadtime_idx1",
            time_key="mean_lead_time",
            metric_key="phys_rmse",
            metric_channel_prefix="phys_rmse_ch",
            metric_output_prefix="rmse",
        )
        _save_curve_npz(
            leadtime_mae_curve_npz_path,
            summary["leadtime_metrics"],
            channel_names,
            index_key="leadtime_idx1",
            time_key="mean_lead_time",
            metric_key="phys_mae",
            metric_channel_prefix="phys_mae_ch",
            metric_output_prefix="mae",
        )
        artifact_paths["leadtime_metrics_csv"] = str(leadtime_metrics_path)
        artifact_paths["leadtime_metrics_plot"] = str(leadtime_plot_path)
        artifact_paths["leadtime_rmse_curve_csv"] = str(leadtime_rmse_curve_csv_path)
        artifact_paths["leadtime_rmse_curve_npz"] = str(leadtime_rmse_curve_npz_path)
        artifact_paths["leadtime_mae_curve_csv"] = str(leadtime_mae_curve_csv_path)
        artifact_paths["leadtime_mae_curve_npz"] = str(leadtime_mae_curve_npz_path)
    metrics_payload = dict(summary["metrics"])
    metrics_payload["channel_names"] = channel_names
    metrics_payload["metric_table"] = summary["metric_table"]
    if "summary_table" in summary:
        metrics_payload["summary_table"] = summary["summary_table"]
    if "leadtime_metrics" in summary:
        metrics_payload["leadtime_rmse_curve"] = _curve_rows_from_index_rows(
            summary["leadtime_metrics"],
            channel_names,
            index_key="leadtime_idx1",
            time_key="mean_lead_time",
            metric_key="phys_rmse",
            metric_channel_prefix="phys_rmse_ch",
            metric_output_prefix="rmse",
        )
        metrics_payload["leadtime_mae_curve"] = _curve_rows_from_index_rows(
            summary["leadtime_metrics"],
            channel_names,
            index_key="leadtime_idx1",
            time_key="mean_lead_time",
            metric_key="phys_mae",
            metric_channel_prefix="phys_mae_ch",
            metric_output_prefix="mae",
        )
    if metadata:
        metrics_payload["evaluation_metadata"] = metadata
    metrics_payload["artifacts"] = artifact_paths
    save_json(metrics_path, metrics_payload)
    return artifact_paths
