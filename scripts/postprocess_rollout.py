#!/usr/bin/env python3
"""Recompute/compare saved rollouts using predictions and targets alone."""

from __future__ import annotations

import argparse

from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from utils.rollout_postprocess import compare_rollout_artifacts, save_comparison_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", nargs="+", required=True, help="Complete *_predictions.npz artifacts with matching metadata JSON.")
    parser.add_argument("--labels", nargs="+", help="One unique report label per artifact.")
    parser.add_argument("--mode", choices=["full", "equal-lead", "common-tail", "lead-slice", "absolute-slice", "time-slice"], default="full")
    parser.add_argument("--lead-steps", type=int, help="Required equal-lead length, e.g. 120.")
    parser.add_argument("--lead-start", type=int, default=1, help="Inclusive one-based lead step.")
    parser.add_argument("--lead-stop", type=int, help="Inclusive one-based lead step.")
    parser.add_argument("--absolute-start", type=int, help="Inclusive zero-based absolute state index.")
    parser.add_argument("--absolute-stop", type=int, help="Exclusive zero-based absolute state index.")
    parser.add_argument("--time-start", type=float, help="Inclusive actual physical trajectory time.")
    parser.add_argument("--time-stop", type=float, help="Exclusive actual physical trajectory time.")
    parser.add_argument("--output-prefix", required=True, help="Prefix for JSON and CSV reports.")
    args = parser.parse_args()
    values = vars(args)
    output_prefix = values.pop("output_prefix")
    paths = values.pop("artifacts")
    report = compare_rollout_artifacts(paths, **values)
    outputs = save_comparison_report(report, output_prefix)
    print("label,known_steps,rmse,norm_rmse")
    for result in report["results"]:
        metrics = result["metrics"]
        print(f"{result['label']},{result['known_steps']},{metrics['whole_rollout_rmse']:.8g},{metrics['whole_rollout_norm_rmse']:.8g}")
    print(f"Saved report: {outputs['json']}")


if __name__ == "__main__":
    main()
