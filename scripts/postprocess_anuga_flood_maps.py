#!/usr/bin/env python3
"""Render ANUGA GT-vs-Pred flood maps from saved evaluation artifacts."""

from __future__ import annotations

import argparse

from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from utils import configure_logging
from utils.anuga_postprocess import generate_anuga_flood_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to a saved *_predictions.npz archive.")
    parser.add_argument(
        "--predictions-meta",
        type=str,
        default=None,
        help="Optional path to the companion *_predictions_meta.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to a sibling *_flood_maps directory.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=None,
        help="Optional scenario id filter. Repeat to render multiple scenarios.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=3,
        help="How many trajectory indices to render per scenario when --trajectory-idx is not provided.",
    )
    parser.add_argument("--levels", type=int, default=30, help="Contour levels for the flood depth maps.")
    parser.add_argument(
        "--trajectory-idx",
        action="append",
        type=int,
        default=None,
        help="Optional 1-based absolute trajectory index to render. Repeat to render multiple frames.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root used to resolve relative scenario paths saved in metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    summary = generate_anuga_flood_maps(
        args.predictions,
        predictions_meta_path=args.predictions_meta,
        output_dir=args.output_dir,
        scenario_ids=args.scenario_id,
        num_frames=args.num_frames,
        levels=args.levels,
        trajectory_indices1=args.trajectory_idx,
        project_root=args.project_root,
    )

    logger.info("Saved ANUGA flood-map artifacts to %s", summary["output_dir"])
    logger.info("Top-level summary: %s", summary["summary_json"])
    for scenario in summary["scenarios"]:
        logger.info(
            "Scenario %s: %d figure(s), timeseries=%s",
            scenario["scenario_id"],
            len(scenario["figure_paths"]),
            scenario["timeseries_npz"],
        )


if __name__ == "__main__":
    main()
