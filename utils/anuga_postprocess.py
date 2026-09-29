"""Post-processing helpers for ANUGA flood-map visualization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from datasets.anuga_dataset import compute_depth
from .io import ensure_dir, save_json


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _load_predictions(predictions_path: str | Path, predictions_meta_path: str | Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    predictions_path = Path(predictions_path)
    if predictions_meta_path is None:
        predictions_meta_path = Path(str(predictions_path).replace("_predictions.npz", "_predictions_meta.json"))
    predictions_meta_path = Path(predictions_meta_path)

    with np.load(predictions_path) as archive:
        bundle = {key: archive[key] for key in archive.files}
    with predictions_meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return bundle, meta


def _resolve_scenario_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _build_gt_state(raw: dict[str, np.ndarray], channel_names: list[str]) -> np.ndarray:
    if channel_names != ["depth", "xmomentum", "ymomentum"]:
        raise ValueError(f"Expected ANUGA channel names ['depth', 'xmomentum', 'ymomentum'], got {channel_names!r}.")

    stage = np.asarray(raw["stage"], dtype=np.float32)
    elevation = np.asarray(raw["elevation"], dtype=np.float32)
    xmomentum = np.asarray(raw["xmomentum"], dtype=np.float32)
    ymomentum = np.asarray(raw["ymomentum"], dtype=np.float32)
    if "h" in raw:
        depth = np.asarray(raw["h"], dtype=np.float32)
    else:
        depth = compute_depth(stage, elevation)
    return np.stack([depth, xmomentum, ymomentum], axis=-1).astype(np.float32)


def _aggregate_predictions_for_scenario(
    bundle: dict[str, np.ndarray],
    *,
    scenario_index: int,
    total_steps: int,
    num_nodes: int,
    state_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenario_mask = bundle["scenario_index"] == int(scenario_index)
    pred_state = np.full((total_steps, num_nodes, state_dim), np.nan, dtype=np.float32)
    pred_count = np.zeros((total_steps, num_nodes), dtype=np.int32)
    pred_available = np.zeros((total_steps,), dtype=np.uint8)
    if not np.any(scenario_mask):
        return pred_state, pred_count, pred_available

    trajectory_idx0 = bundle["trajectory_idx0"][scenario_mask].astype(np.int64, copy=False)
    node_index = bundle["node_index"][scenario_mask].astype(np.int64, copy=False)
    pred_phys = bundle["pred_phys"][scenario_mask].astype(np.float64, copy=False)

    pred_sum = np.zeros((total_steps, num_nodes, state_dim), dtype=np.float64)
    np.add.at(pred_count, (trajectory_idx0, node_index), 1)
    for channel_index in range(state_dim):
        np.add.at(pred_sum[:, :, channel_index], (trajectory_idx0, node_index), pred_phys[:, channel_index])

    valid = pred_count > 0
    denom = np.where(valid, pred_count, 1).astype(np.float64)
    for channel_index in range(state_dim):
        averaged = pred_sum[:, :, channel_index] / denom
        pred_state[:, :, channel_index] = np.where(valid, averaged, np.nan).astype(np.float32)

    # A timestep is available only when every mesh node has a prediction.
    pred_available = valid.all(axis=1).astype(np.uint8)
    return pred_state, pred_count, pred_available


def _select_trajectory_indices(
    pred_available: np.ndarray,
    *,
    num_frames: int,
    trajectory_indices1: list[int] | None,
) -> list[int]:
    available_idx0 = np.where(pred_available > 0)[0]
    if available_idx0.size == 0:
        return []

    if trajectory_indices1:
        chosen = []
        for idx1 in trajectory_indices1:
            idx0 = int(idx1) - 1
            if idx0 < 0 or idx0 >= pred_available.shape[0]:
                continue
            if pred_available[idx0] > 0:
                chosen.append(idx0)
        return sorted(set(chosen))

    if num_frames <= 0:
        return []
    if available_idx0.size <= num_frames:
        return [int(value) for value in available_idx0.tolist()]
    pick = [int(round(i * (available_idx0.size - 1) / max(num_frames - 1, 1))) for i in range(num_frames)]
    return [int(available_idx0[idx]) for idx in pick]


def _save_scenario_timeseries(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    volumes: np.ndarray,
    time_s: np.ndarray,
    gt_state: np.ndarray,
    pred_state: np.ndarray,
    pred_count: np.ndarray,
    pred_available: np.ndarray,
    channel_names: list[str],
    scenario_id: str,
    sim_id: str,
    evaluation_metadata: dict[str, Any] | None = None,
) -> None:
    prediction_indices0 = np.where(pred_available > 0)[0]
    payload: dict[str, Any] = {
        "x": x.astype(np.float32, copy=False),
        "y": y.astype(np.float32, copy=False),
        "volumes": volumes.astype(np.int32, copy=False),
        "time_s": time_s.astype(np.float64, copy=False),
        "pred_count": pred_count.astype(np.int32, copy=False),
        "pred_available": pred_available.astype(np.uint8, copy=False),
        "channel_names": np.asarray(channel_names, dtype=object),
        "scenario_id": np.asarray([scenario_id], dtype=object),
        "sim_id": np.asarray([sim_id], dtype=object),
        "total_steps": np.asarray([time_s.shape[0]], dtype=np.int64),
        "rollout_num_steps": np.asarray([prediction_indices0.size], dtype=np.int64),
        "prediction_indices0": prediction_indices0.astype(np.int64, copy=False),
        "prediction_indices1": (prediction_indices0 + 1).astype(np.int64, copy=False),
    }
    if prediction_indices0.size:
        payload["prediction_start_idx0"] = np.asarray([prediction_indices0[0]], dtype=np.int64)
        payload["prediction_end_idx0"] = np.asarray([prediction_indices0[-1]], dtype=np.int64)
    if evaluation_metadata:
        for key in (
            "known_steps",
            "rollout_start_idx0",
            "rollout_start_idx1",
            "eval_history_len",
            "train_config_history_len",
            "train_config_future_len",
        ):
            value = evaluation_metadata.get(key)
            if value is not None:
                payload[key] = np.asarray([int(value)], dtype=np.int64)
    for channel_index, channel_name in enumerate(channel_names):
        safe_name = _safe_name(channel_name)
        payload[f"gt_{safe_name}"] = gt_state[:, :, channel_index].astype(np.float32, copy=False)
        payload[f"pred_{safe_name}"] = pred_state[:, :, channel_index].astype(np.float32, copy=False)
    np.savez_compressed(path, **payload)


def _plot_depth_maps_for_scenario(
    *,
    scenario_out_dir: Path,
    scenario_stem: str,
    x: np.ndarray,
    y: np.ndarray,
    volumes: np.ndarray,
    time_s: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    chosen_idx0: list[int],
    levels: int,
) -> list[str]:
    if not chosen_idx0:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if volumes.size >= 3 and volumes.ndim == 2 and volumes.shape[1] == 3:
        triang = mtri.Triangulation(x, y, triangles=volumes)
    else:
        triang = mtri.Triangulation(x, y)

    saved_paths: list[str] = []
    for frame_id, trajectory_idx0 in enumerate(chosen_idx0):
        gt = np.maximum(gt_depth[trajectory_idx0], 0.0)
        pred = np.maximum(pred_depth[trajectory_idx0], 0.0)
        finite_values = np.concatenate([gt[np.isfinite(gt)], pred[np.isfinite(pred)]])
        if finite_values.size == 0:
            continue
        vmin = float(np.nanmin(finite_values))
        vmax = float(np.nanmax(finite_values))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        contour = None
        for axis, values, title in ((axes[0], gt, "GT"), (axes[1], pred, "Pred")):
            contour = axis.tricontourf(triang, values, levels=levels, vmin=vmin, vmax=vmax)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            axis.set_title(f"{title} flood depth h (m) @ t={time_s[trajectory_idx0]:.0f} s")
        fig.colorbar(contour, ax=axes.ravel().tolist(), label="flood depth h (m)")
        fig.suptitle(f"{scenario_stem} (trajectory_idx={trajectory_idx0 + 1})")
        fig_path = scenario_out_dir / f"depth_map_{frame_id:02d}_t{trajectory_idx0 + 1:04d}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        saved_paths.append(str(fig_path))
    return saved_paths


def generate_anuga_flood_maps(
    predictions_path: str | Path,
    *,
    predictions_meta_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    scenario_ids: list[str] | None = None,
    num_frames: int = 3,
    levels: int = 30,
    trajectory_indices1: list[int] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load saved evaluation artifacts and render ANUGA GT-vs-Pred flood maps."""

    bundle, meta = _load_predictions(predictions_path, predictions_meta_path)
    channel_names = list(meta.get("channel_names", []))
    if channel_names != ["depth", "xmomentum", "ymomentum"]:
        raise ValueError(
            "ANUGA flood-map post-processing expects channel_names=['depth', 'xmomentum', 'ymomentum']; "
            f"got {channel_names!r}."
        )

    predictions_path = Path(predictions_path)
    project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    if output_dir is None:
        output_dir = predictions_path.with_name(predictions_path.stem.replace("_predictions", "_flood_maps"))
    output_dir = ensure_dir(output_dir).resolve()

    wanted_ids = None if not scenario_ids else set(scenario_ids)
    scenario_lookup = list(meta.get("scenario_lookup", []))
    evaluation_metadata = dict(meta.get("evaluation_metadata", {}))
    known_steps_value = evaluation_metadata.get("known_steps")
    known_steps = int(known_steps_value) if known_steps_value is not None else None
    scenario_summaries: list[dict[str, Any]] = []

    for scenario_info in scenario_lookup:
        scenario_id = str(scenario_info["scenario_id"])
        if wanted_ids is not None and scenario_id not in wanted_ids:
            continue

        raw_path = _resolve_scenario_path(str(scenario_info["path"]), project_root=project_root).resolve()
        raw = np.load(raw_path)
        x = np.asarray(raw["x"], dtype=np.float32)
        y = np.asarray(raw["y"], dtype=np.float32)
        volumes = np.asarray(raw["volumes"], dtype=np.int32) if "volumes" in raw else np.empty((0, 3), dtype=np.int32)
        time_s = np.asarray(raw["time"], dtype=np.float64)
        gt_state = _build_gt_state(raw, channel_names)
        pred_state, pred_count, pred_available = _aggregate_predictions_for_scenario(
            bundle,
            scenario_index=int(scenario_info["scenario_index"]),
            total_steps=int(gt_state.shape[0]),
            num_nodes=int(gt_state.shape[1]),
            state_dim=int(gt_state.shape[2]),
        )
        if str(meta.get("mode", "")) == "full_rollout" and known_steps is not None:
            expected_available = np.zeros((gt_state.shape[0],), dtype=np.uint8)
            expected_available[known_steps:] = 1
            if not np.array_equal(pred_available, expected_available):
                actual_indices1 = (np.where(pred_available > 0)[0] + 1).astype(int).tolist()
                expected_indices1 = (np.where(expected_available > 0)[0] + 1).astype(int).tolist()
                raise ValueError(
                    f"Incomplete full-rollout coverage for {scenario_id}: "
                    f"predicted trajectory indices={actual_indices1}, expected={expected_indices1}."
                )
            if np.any(pred_count[:known_steps] != 0) or np.any(pred_count[known_steps:] != 1):
                raise ValueError(
                    f"Expected exactly one prediction per node and future timestep for {scenario_id}."
                )

        scenario_dir = ensure_dir(output_dir / _safe_name(scenario_id))
        timeseries_path = scenario_dir / "depth_timeseries.npz"
        _save_scenario_timeseries(
            timeseries_path,
            x=x,
            y=y,
            volumes=volumes,
            time_s=time_s,
            gt_state=gt_state,
            pred_state=pred_state,
            pred_count=pred_count,
            pred_available=pred_available,
            channel_names=channel_names,
            scenario_id=scenario_id,
            sim_id=str(scenario_info["sim_id"]),
            evaluation_metadata=evaluation_metadata,
        )

        chosen_idx0 = _select_trajectory_indices(
            pred_available,
            num_frames=max(int(num_frames), 0),
            trajectory_indices1=trajectory_indices1,
        )
        figure_paths = _plot_depth_maps_for_scenario(
            scenario_out_dir=scenario_dir,
            scenario_stem=_safe_name(scenario_id),
            x=x,
            y=y,
            volumes=volumes,
            time_s=time_s,
            gt_depth=gt_state[:, :, 0],
            pred_depth=pred_state[:, :, 0],
            chosen_idx0=chosen_idx0,
            levels=max(int(levels), 2),
        )

        available_indices0 = np.where(pred_available > 0)[0]
        scenario_summary = {
            "scenario_id": scenario_id,
            "sim_id": str(scenario_info["sim_id"]),
            "source_path": str(raw_path),
            "timeseries_npz": str(timeseries_path.resolve()),
            "figure_paths": [str(Path(path).resolve()) for path in figure_paths],
            "predicted_trajectory_indices1": (np.where(pred_available > 0)[0] + 1).astype(int).tolist(),
            "chosen_trajectory_indices1": [int(idx + 1) for idx in chosen_idx0],
            "summary_json": str((scenario_dir / "summary.json").resolve()),
            "total_steps": int(gt_state.shape[0]),
            "known_steps": known_steps,
            "rollout_num_steps": int(np.count_nonzero(pred_available)),
            "prediction_start_idx0": int(available_indices0[0]) if available_indices0.size else None,
            "prediction_end_idx0": int(available_indices0[-1]) if available_indices0.size else None,
        }
        save_json(scenario_dir / "summary.json", scenario_summary)
        scenario_summaries.append(scenario_summary)

    if not scenario_summaries:
        raise ValueError("No ANUGA scenarios were rendered. Check the requested scenario ids and predictions metadata.")

    summary = {
        "predictions_path": str(Path(predictions_path).resolve()),
        "mode": str(meta.get("mode", "")),
        "channel_names": channel_names,
        "output_dir": str(output_dir),
        "summary_json": str((Path(output_dir) / "summary.json").resolve()),
        "prediction_aggregation": "mean over repeated predictions for the same (trajectory_idx, node_index) pair",
        "evaluation_metadata": evaluation_metadata,
        "scenarios": scenario_summaries,
    }
    save_json(Path(output_dir) / "summary.json", summary)
    return summary
