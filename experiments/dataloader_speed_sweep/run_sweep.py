#!/usr/bin/env python3
"""Launch a compact dataloader-speed sweep for ISSM NODE2 upgrade-v1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = EXPERIMENT_DIR / "train_timed.py"
DEFAULT_DATA_DIR = "./data/ISSM/PIG_5000"


DEFAULT_COMBOS: list[dict[str, Any]] = [
    {
        "name": "nw4_vw4_pin0_pfnone_persist0",
        "num_workers": 4,
        "val_num_workers": 4,
        "pin_memory": False,
        "prefetch_factor": None,
        "persistent_workers": False,
    },
    {
        "name": "nw0_vw0_pin0_pfnone_persist0",
        "num_workers": 0,
        "val_num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": None,
        "persistent_workers": False,
    },
    {
        "name": "nw2_vw2_pin1_pf2_persist1",
        "num_workers": 2,
        "val_num_workers": 2,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    },
    {
        "name": "nw4_vw4_pin1_pf2_persist0",
        "num_workers": 4,
        "val_num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": False,
    },
    {
        "name": "nw4_vw4_pin1_pf2_persist1",
        "num_workers": 4,
        "val_num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    },
    {
        "name": "nw4_vw4_pin1_pf4_persist1",
        "num_workers": 4,
        "val_num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    },
    {
        "name": "nw8_vw4_pin1_pf2_persist1",
        "num_workers": 8,
        "val_num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    },
    {
        "name": "nw8_vw8_pin1_pf2_persist1",
        "num_workers": 8,
        "val_num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    },
    {
        "name": "nw8_vw8_pin1_pf4_persist1",
        "num_workers": 8,
        "val_num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/ISSM_History_Scan/base_ISSM_history_4.yaml")
    parser.add_argument("--dataset-config", default="configs/issm.yaml")
    parser.add_argument("--model-config", default="configs/model_node2.yaml")
    parser.add_argument("--extra-config", action="append", default=[])
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="ISSM subset to sweep. Defaults to the PIG_5000-only directory, not PIG_data.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument(
        "--train-horizon-max",
        type=int,
        default=None,
        help="Optional timing cap for training horizon. Omit for the formal config behavior.",
    )
    parser.add_argument(
        "--horizon-curriculum",
        choices=("on", "off"),
        default=None,
        help="Override training.train_horizon_curriculum.enabled for each timed run.",
    )
    parser.add_argument(
        "--skip-full-rollout-on-val",
        action="store_true",
        help="Disable full-rollout validation for train-loader-only timing.",
    )
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--output-root", default="experiments/dataloader_speed_sweep/outputs")
    parser.add_argument("--work-root", default="experiments/dataloader_speed_sweep/runs")
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--combo-index", action="append", type=int, default=None, help="Run only selected 0-based combo index.")
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def selected_combos(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(DEFAULT_COMBOS))
    if args.combo_index is not None:
        wanted = set(args.combo_index)
        indexed = [(idx, combo) for idx, combo in indexed if idx in wanted]
    if args.max_combos is not None:
        indexed = indexed[: args.max_combos]
    return indexed


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def stream_command(cmd: list[str], *, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def flatten_result(row: dict[str, Any]) -> dict[str, Any]:
    combo = row["combo"]
    summary = row.get("summary") or {}
    flat = {
        "combo_index": row["combo_index"],
        "name": combo["name"],
        "returncode": row["returncode"],
        "run_name": row["run_name"],
        "num_workers": combo["num_workers"],
        "val_num_workers": combo["val_num_workers"],
        "pin_memory": combo["pin_memory"],
        "prefetch_factor": combo["prefetch_factor"],
        "persistent_workers": combo["persistent_workers"],
        "mean_train_seconds_warm": summary.get("mean_train_seconds_warm"),
        "mean_train_windows_per_sec_warm": summary.get("mean_train_windows_per_sec_warm"),
        "mean_train_seconds_all": summary.get("mean_train_seconds_all"),
        "mean_train_windows_per_sec_all": summary.get("mean_train_windows_per_sec_all"),
        "mean_val_seconds_all": summary.get("mean_val_seconds_all"),
        "mean_epoch_seconds_all": summary.get("mean_epoch_seconds_all"),
        "summary_path": row.get("summary_path"),
        "log_path": row["log_path"],
    }
    return flat


def write_results(results_path: Path, csv_path: Path, results: list[dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    rows = [flatten_result(row) for row in results]
    if not rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_prefix or f"issm_node2_h4_up1_dataloader_{stamp}"
    output_root = resolve_project_path(args.output_root)
    work_root = resolve_project_path(args.work_root) / run_prefix
    config_root = work_root / "configs"
    log_root = work_root / "logs"
    results_path = work_root / "sweep_results.json"
    csv_path = work_root / "sweep_results.csv"
    best_path = work_root / "best_result.json"
    work_root.mkdir(parents=True, exist_ok=True)
    data_dir_path = resolve_project_path(args.data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Configured data directory does not exist: {data_dir_path}")

    base_configs = [
        resolve_project_path(args.base_config),
        resolve_project_path(args.dataset_config),
        resolve_project_path(args.model_config),
        *[resolve_project_path(path) for path in args.extra_config],
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    results: list[dict[str, Any]] = []
    for combo_index, combo in selected_combos(args):
        run_name = f"{run_prefix}_c{combo_index:02d}_{combo['name']}"
        override = {
            "output_dir": str(output_root),
            "dataset": {
                "data_dir": args.data_dir,
                "num_workers": int(combo["num_workers"]),
                "val_num_workers": int(combo["val_num_workers"]),
                "pin_memory": bool(combo["pin_memory"]),
                "prefetch_factor": combo["prefetch_factor"],
                "persistent_workers": bool(combo["persistent_workers"]),
            },
            "training": {
                "epochs": int(args.epochs),
                "val_every": int(args.val_every),
            },
            "evaluation": {
                "full_rollout_on_val": not bool(args.skip_full_rollout_on_val),
                "checkpoint_metric": "norm_rmse" if args.skip_full_rollout_on_val else "whole_rollout_norm_rmse",
            },
        }
        if args.train_horizon_max is not None:
            override["training"]["train_horizon_max"] = int(args.train_horizon_max)
        if args.horizon_curriculum is not None:
            override["training"]["train_horizon_curriculum"] = {"enabled": args.horizon_curriculum == "on"}
        override_path = config_root / f"combo_{combo_index:02d}_{combo['name']}.yaml"
        write_yaml(override_path, override)

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.nproc}",
            "--master_port",
            str(29600 + combo_index),
            str(TRAIN_SCRIPT),
        ]
        for config_path in [*base_configs, override_path]:
            cmd.extend(["--config", str(config_path)])
        cmd.extend(["--run-name", run_name])

        log_path = log_root / f"{run_name}.log"
        print(f"\n=== combo {combo_index}: {combo['name']} ===")
        print(f"data_dir={args.data_dir}")
        print(" ".join(cmd))
        start = time.perf_counter()
        returncode = 0 if args.dry_run else stream_command(cmd, log_path=log_path, env=env)
        elapsed_seconds = time.perf_counter() - start

        summary_path = output_root / run_name / "speed_summary.json"
        summary = None
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

        result = {
            "combo_index": combo_index,
            "combo": combo,
            "run_name": run_name,
            "returncode": returncode,
            "elapsed_seconds": elapsed_seconds,
            "log_path": str(log_path),
            "override_path": str(override_path),
            "summary_path": str(summary_path) if summary_path.exists() else None,
            "summary": summary,
        }
        results.append(result)
        write_results(results_path, csv_path, results)
        if returncode != 0:
            print(f"combo {combo_index} failed with returncode={returncode}; stopping sweep")
            break

    successful = [row for row in results if row["returncode"] == 0 and row.get("summary")]
    if successful:
        best = max(successful, key=lambda row: float(row["summary"]["mean_train_windows_per_sec_warm"]))
        with best_path.open("w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2, sort_keys=True)
        print("\nBest combo by warm train windows/sec:")
        print(json.dumps(flatten_result(best), indent=2, sort_keys=True))

    print(f"\nResults JSON: {results_path}")
    print(f"Results CSV:  {csv_path}")
    if best_path.exists():
        print(f"Best JSON:    {best_path}")


if __name__ == "__main__":
    main()
