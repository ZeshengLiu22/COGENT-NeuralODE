#!/usr/bin/env python3
"""Full-trajectory rollout evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

import torch

from datasets import ADCIRCDataset, ANUGADataset, ISSMDataset
from datasets.normalization import FeatureNormalizer
from datasets.split_utils import discover_files, issm_rate_modulo_split, random_split
from models import build_model
from utils import configure_logging, load_config_bundle
from utils.eval_artifacts import (
    collect_full_rollout_prediction_bundle,
    format_metric_table,
    infer_state_channel_names,
    save_evaluation_artifacts,
    summarize_full_rollout_bundle,
    validate_full_rollout_bundle,
)
from utils.anuga_postprocess import generate_anuga_flood_maps


DATASET_REGISTRY = {
    "anuga": ANUGADataset,
    "adcirc": ADCIRCDataset,
    "issm": ISSMDataset,
}

TRANSIENT_CHECKPOINT_PREFIXES = ("dynamics.control.",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Fallback YAML/JSON config files for legacy checkpoints without an embedded config. "
            "Modern checkpoints use their own saved config."
        ),
    )
    parser.add_argument(
        "--split-files",
        type=str,
        default=None,
        help=(
            "Optional exact split manifest. Defaults to split_files.json beside the checkpoint; "
            "only falls back to recomputing the split when no saved manifest exists."
        ),
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--history-len",
        type=int,
        default=None,
        help="Optional evaluation-only history length override. Defaults to dataset.history_len from the config.",
    )
    parser.add_argument(
        "--known-steps",
        type=int,
        default=None,
        help=(
            "Number of initial true timesteps before rollout begins. Defaults to "
            "evaluation.full_rollout_known_steps from the saved checkpoint config, then to history length."
        ),
    )
    parser.add_argument(
        "--plot-max-lead-step",
        type=int,
        default=None,
        help="Only save the first this many lead-time steps in the full-rollout summary and curve artifacts. The rollout itself still runs to the end.",
    )
    parser.add_argument(
        "--anuga-output-dir",
        type=str,
        default=None,
        help="Optional directory for structured per-scenario ANUGA time series and flood maps.",
    )
    parser.add_argument(
        "--anuga-num-frames",
        type=int,
        default=3,
        help=(
            "Number of representative GT-vs-Pred flood-map frames per ANUGA scenario. "
            "Use 0 to save only the complete per-step time series."
        ),
    )
    parser.add_argument(
        "--skip-anuga-export",
        action="store_true",
        help="Skip automatic structured ANUGA time-series/flood-map export.",
    )
    return parser.parse_args()


def build_split_files(config: dict) -> dict[str, list[Path]]:
    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["split"]
    files = discover_files(dataset_cfg["data_dir"], dataset_cfg["file_patterns"])
    strategy = str(split_cfg.get("strategy", "random")).lower()
    if strategy == "issm_rate_modulo":
        train_files, val_files, test_files = issm_rate_modulo_split(
            files,
            modulo=int(split_cfg.get("modulo", 20)),
            val_remainder=int(split_cfg.get("val_remainder", 0)),
            test_remainder=int(split_cfg.get("test_remainder", 10)),
        )
    else:
        train_files, val_files, test_files = random_split(
            files,
            train=float(split_cfg["train"]),
            val=float(split_cfg["val"]),
            test=float(split_cfg["test"]),
            seed=int(split_cfg.get("seed", config["seed"])),
        )
    return {"train": train_files, "val": val_files, "test": test_files}


def load_evaluation_config(checkpoint: dict, config_paths: list[str] | None) -> tuple[dict, str]:
    """Prefer the exact merged config embedded in the training checkpoint."""

    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise TypeError(f"Expected checkpoint['config'] to be a dict, got {type(checkpoint_config)!r}.")
        return deepcopy(checkpoint_config), "checkpoint"
    if not config_paths:
        raise ValueError(
            "The checkpoint does not contain an embedded config. Supply the original config with --config."
        )
    return load_config_bundle(config_paths), "cli"


def _resolve_saved_scenario_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_saved_split_files(path: str | Path) -> dict[str, list[Path]]:
    """Load the exact train/val/test file lists recorded at training time."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a split mapping in {manifest_path}, got {type(payload)!r}.")

    split_files: dict[str, list[Path]] = {}
    for split_name in ("train", "val", "test"):
        values = payload.get(split_name)
        if not isinstance(values, list):
            raise ValueError(f"Missing list-valued '{split_name}' entry in {manifest_path}.")
        split_files[split_name] = [_resolve_saved_scenario_path(value) for value in values]
    return split_files


def resolve_split_files(
    config: dict,
    *,
    checkpoint_path: Path,
    split_manifest_path: str | None,
) -> tuple[dict[str, list[Path]], Path | None]:
    """Use the saved split manifest when available, with a legacy fallback."""

    manifest_path = Path(split_manifest_path) if split_manifest_path else checkpoint_path.parent / "split_files.json"
    if manifest_path.exists():
        return load_saved_split_files(manifest_path), manifest_path.resolve()
    if split_manifest_path:
        raise FileNotFoundError(f"Requested split manifest does not exist: {manifest_path}")
    return build_split_files(config), None


def load_model_state_strict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Strictly load learned state while dropping known runtime-only control buffers."""

    transient_keys = sorted(
        key
        for key in state_dict
        if key.startswith(TRANSIENT_CHECKPOINT_PREFIXES)
    )
    learned_state = {
        key: value
        for key, value in state_dict.items()
        if key not in transient_keys
    }
    model.load_state_dict(learned_state, strict=True)
    return transient_keys


def build_dataset(
    dataset_name: str,
    files: list[Path],
    split: str,
    config: dict,
    normalizer,
    *,
    history_len: int | None = None,
):
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset_cfg = config["dataset"]
    return dataset_cls(
        scenario_files=files,
        history_len=int(dataset_cfg["history_len"] if history_len is None else history_len),
        future_len=int(dataset_cfg["future_len"]),
        split=split,
        stride=int(dataset_cfg.get("stride", 1)),
        normalizer=normalizer,
        cache_in_memory=bool(dataset_cfg.get("cache_in_memory", False)),
        seed=int(dataset_cfg.get("seed", config["seed"])),
        adapter_kwargs=dataset_cfg.get(dataset_name, {}),
    )


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for full-rollout evaluation, but no CUDA device is available.")
    return device


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config, config_source = load_evaluation_config(checkpoint, args.config)
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    logger = configure_logging()
    device = resolve_device(args.device)
    if config_source == "checkpoint" and args.config:
        logger.info("Using the config embedded in %s; supplied --config files are not needed.", checkpoint_path)

    dataset_name = str(config["dataset"]["name"]).lower()
    train_history_len = int(config["dataset"]["history_len"])
    eval_history_len = train_history_len if args.history_len is None else int(args.history_len)
    if eval_history_len < 1:
        raise ValueError(f"--history-len must be >= 1, got {eval_history_len}.")
    configured_known_steps = config.get("evaluation", {}).get("full_rollout_known_steps")
    if args.known_steps is not None:
        known_steps = int(args.known_steps)
    elif configured_known_steps is not None:
        known_steps = int(configured_known_steps)
    else:
        known_steps = eval_history_len
    if known_steps < eval_history_len:
        raise ValueError(
            f"--known-steps must be >= evaluation history length ({eval_history_len}), got {known_steps}."
        )
    split_files, split_manifest = resolve_split_files(
        config,
        checkpoint_path=checkpoint_path,
        split_manifest_path=args.split_files,
    )
    missing_files = [path for path in split_files[args.split] if not path.is_file()]
    if missing_files:
        preview = ", ".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(
            f"{len(missing_files)} saved {args.split} scenario file(s) are missing. First entries: {preview}"
        )
    dataset = build_dataset(
        dataset_name,
        split_files[args.split],
        args.split,
        config,
        normalizer,
        history_len=eval_history_len,
    )
    if not dataset.scenario_infos:
        raise RuntimeError(f"No scenarios are available for split={args.split}.")
    min_steps = min(int(info["length"]) for info in dataset.scenario_infos)
    if known_steps >= min_steps:
        raise ValueError(
            f"--known-steps must leave at least one future step across the {args.split} split; "
            f"got known_steps={known_steps}, but the shortest trajectory has only {min_steps} steps."
        )
    start_t = known_steps - 1
    rollout_lengths = [int(info["length"]) - known_steps for info in dataset.scenario_infos]

    sample = dataset.get_rollout_data(0, start_t=start_t)
    model = build_model(config, sample.x_static.shape[-1], sample.force_hist.shape[-1], sample.state_hist.shape[-1])
    skipped_transient_keys = load_model_state_strict(model, checkpoint["model_state"])
    model.to(device)
    model.eval()
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info("Using evaluation device: %s (%s)", device, torch.cuda.get_device_name(device_index))
    else:
        logger.info("Using evaluation device: %s", device)
    logger.info(
        "Full-rollout evaluation context: history_len=%d known_steps=%d rollout_steps=%d..%d "
        "plot_max_lead_step=%s (training config: history_len=%d future_len=%d)",
        eval_history_len,
        known_steps,
        min(rollout_lengths),
        max(rollout_lengths),
        "all" if args.plot_max_lead_step is None else str(int(args.plot_max_lead_step)),
        train_history_len,
        int(config["dataset"]["future_len"]),
    )
    logger.info(
        "Evaluation provenance: config_source=%s split_source=%s",
        config_source,
        str(split_manifest) if split_manifest is not None else "recomputed_from_config",
    )
    if skipped_transient_keys:
        logger.info(
            "Ignored %d runtime-only forcing-interpolator buffer(s) during strict learned-state load: %s",
            len(skipped_transient_keys),
            skipped_transient_keys,
        )
    bundle = collect_full_rollout_prediction_bundle(model, dataset, normalizer, device=device, start_t=start_t)
    coverage = validate_full_rollout_bundle(
        bundle,
        scenario_infos=dataset.scenario_infos,
        known_steps=known_steps,
        node_counts=[
            int(dataset.get_rollout_data(index, start_t=start_t).num_nodes)
            for index in range(len(dataset.scenario_infos))
        ],
    )
    channel_names = infer_state_channel_names(dataset_name, state_dim=int(bundle["pred_phys"].shape[1]))
    summary_method = f"{config['model']['name']}_full_rollout"
    summary = summarize_full_rollout_bundle(
        bundle,
        channel_names,
        max_lead_time_step=args.plot_max_lead_step,
        summary_method=summary_method,
    )
    artifact_stem = Path(str(checkpoint_path.with_suffix("")) + ".full_rollout")
    evaluation_metadata = {
        "eval_history_len": eval_history_len,
        "known_steps": known_steps,
        "rollout_start_idx0": start_t,
        "rollout_start_idx1": start_t + 1,
        "train_config_history_len": train_history_len,
        "train_config_future_len": int(config["dataset"]["future_len"]),
        "plot_max_lead_step": args.plot_max_lead_step,
        "full_rollout_num_lead_steps": len(summary["metrics"]["horizon_rmse_curve"]),
        "artifact_num_lead_steps": len(summary["leadtime_metrics"]),
        "summary_method": summary_method,
        "config_source": config_source,
        "split_manifest": str(split_manifest) if split_manifest is not None else None,
        "rollout_coverage": coverage,
    }
    artifact_paths = save_evaluation_artifacts(
        artifact_stem,
        bundle=bundle,
        summary=summary,
        dataset_name=dataset_name,
        split=args.split,
        mode="full_rollout",
        channel_names=channel_names,
        scenario_infos=dataset.scenario_infos,
        metadata=evaluation_metadata,
    )

    logger.info("Full-rollout metrics: %s", summary["metrics"])
    logger.info("Detailed metric table:\n%s", format_metric_table(summary["metric_table"]))
    logger.info("Saved evaluation artifacts: %s", artifact_paths)
    del bundle
    if dataset_name == "anuga" and not args.skip_anuga_export:
        flood_summary = generate_anuga_flood_maps(
            artifact_paths["predictions_npz"],
            predictions_meta_path=artifact_paths["predictions_meta_json"],
            output_dir=args.anuga_output_dir,
            num_frames=max(int(args.anuga_num_frames), 0),
            project_root=PROJECT_ROOT,
        )
        logger.info(
            "Saved complete per-step ANUGA GT/prediction time series and flood maps: %s",
            flood_summary["summary_json"],
        )


if __name__ == "__main__":
    main()
