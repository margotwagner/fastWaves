#!/usr/bin/env python3
"""Aggregate the three-seed FastWave tuning experiment."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RUN_RE_FASTWAVE = re.compile(
    r"^fwtune_fastwave_eta(?P<eta>\d+p\d+)_beta(?P<beta>\d+p\d+)"
    r"_commitment_xv_h1_s(?P<settle>\d+)_seed(?P<seed>\d+)$"
)
RUN_RE_WAVE = re.compile(
    r"^fwtune_wave_commitment_xv_h1_s(?P<settle>\d+)_seed(?P<seed>\d+)$"
)

PRIMARY_METRICS = [
    "eightarm_rollout/dynamic_first_action_valid_rate",
    "eightarm_rollout/dynamic_action_valid_unvisited_rate",
    "eightarm_rollout/dynamic_routing_departure_rate",
    "eightarm_rollout/dynamic_routing_matches_conditioning_action_rate",
    "eightarm_rollout/dynamic_step_success_rate",
    "eightarm_rollout/dynamic_trial_complete_success_rate",
    "eightarm_rollout/dynamic_unique_unvisited_arms_routed_mean",
    "eightarm_tf/routing_exact_target_arm_acc",
    "extras/fast_weight_norm_mean",
    "extras/fast_drive_norm_mean",
]


def parse_float_tag(value: str) -> float:
    return float(value.replace("p", "."))


def parse_run(run: str) -> dict:
    match = RUN_RE_FASTWAVE.match(run)
    if match:
        return {
            "model": "fastwave",
            "eta": parse_float_tag(match.group("eta")),
            "beta": parse_float_tag(match.group("beta")),
            "settle_steps": int(match.group("settle")),
            "seed": int(match.group("seed")),
        }

    match = RUN_RE_WAVE.match(run)
    if match:
        return {
            "model": "wave",
            "eta": 0.0,
            "beta": 0.0,
            "settle_steps": int(match.group("settle")),
            "seed": int(match.group("seed")),
        }

    raise ValueError(f"Unrecognized run name: {run}")


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)

    metadata = pd.DataFrame([parse_run(run) for run in df["run"]])
    df = pd.concat([metadata, df], axis=1)

    metrics = [metric for metric in PRIMARY_METRICS if metric in df.columns]
    group_cols = ["model", "eta", "beta", "settle_steps"]

    grouped = df.groupby(group_cols, dropna=False)
    mean_df = grouped[metrics].mean().add_suffix("_mean")
    std_df = grouped[metrics].std(ddof=1).add_suffix("_std")
    count_df = grouped.size().rename("n_seeds")

    summary = pd.concat([count_df, mean_df, std_df], axis=1).reset_index()

    sort_metric = "eightarm_rollout/dynamic_step_success_rate_mean"
    secondary = "eightarm_rollout/dynamic_action_valid_unvisited_rate_mean"
    sort_cols = [c for c in [sort_metric, secondary] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output}")

    fastwave = summary[summary["model"] == "fastwave"]
    if not fastwave.empty and sort_metric in fastwave.columns:
        best = fastwave.sort_values(
            [sort_metric, secondary],
            ascending=False,
        ).iloc[0]
        print(
            "\nBest FastWave by mean step success: "
            f"eta={best['eta']}, beta={best['beta']}, "
            f"step_success={best[sort_metric]:.4f}, "
            f"valid_actions={best[secondary]:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/runs/fastwave_tuning_run_level.csv",
    )
    parser.add_argument(
        "--output",
        default="data/runs/fastwave_tuning_summary.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
