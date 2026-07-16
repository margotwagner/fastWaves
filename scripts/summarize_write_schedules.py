#!/usr/bin/env python3
"""Aggregate Wave and FastWave write-schedule comparisons across seeds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PATTERNS = [
    (
        re.compile(
            r"^fastwrite_forced_reward_hold_eta(?P<eta>\d+p\d+)_beta(?P<beta>\d+p\d+)"
            r"_commitment_xv_h1_s(?P<settle>\d+)_seed(?P<seed>\d+)$"
        ),
        "forced_reward_hold",
        "forced_reward",
        "hold",
    ),
    (
        re.compile(
            r"^fastwrite_forced_(?P<mode>decay|hold)_eta(?P<eta>\d+p\d+)"
            r"_beta(?P<beta>\d+p\d+)_commitment_xv_h1_s(?P<settle>\d+)"
            r"_seed(?P<seed>\d+)$"
        ),
        None,
        "forced",
        None,
    ),
    (
        re.compile(
            r"^corecmp_fastwave_eta(?P<eta>\d+p\d+)_beta(?P<beta>\d+p\d+)"
            r"_commitment_xv_h1_s(?P<settle>\d+)_seed(?P<seed>\d+)$"
        ),
        "all_write",
        "all",
        "not_applicable",
    ),
    (
        re.compile(
            r"^corecmp_wave_.*_commitment_xv_h1_s(?P<settle>\d+)_seed(?P<seed>\d+)$"
        ),
        "wave",
        "not_applicable",
        "not_applicable",
    ),
]

PRIMARY_METRICS = [
    "eightarm_rollout/dynamic_first_action_valid_rate",
    "eightarm_rollout/dynamic_action_valid_unvisited_rate",
    "eightarm_rollout/action_selection_entropy",
    "eightarm_rollout/action_selection_top1_probability",
    "eightarm_rollout/action_selection_top1_top2_margin",
    "eightarm_rollout/dynamic_routing_departure_rate",
    "eightarm_rollout/dynamic_routing_matches_conditioning_action_rate",
    "eightarm_rollout/dynamic_step_success_rate",
    "eightarm_rollout/dynamic_trial_complete_success_rate",
    "eightarm_rollout/dynamic_unique_unvisited_arms_routed_mean",
    "eightarm_tf/action_selection_valid_under_teacher_history_rate",
    "eightarm_tf/routing_exact_target_arm_acc",
    "extras/fast_weight_norm_mean",
    "extras/fast_drive_norm_mean",
    "extras/fast_write_gate_mean",
]


def parse_float_tag(value: str | None) -> float | None:
    return None if value is None else float(value.replace("p", "."))


def parse_run(run: str) -> dict:
    for regex, condition, phase, mode in PATTERNS:
        match = regex.match(run)
        if match is None:
            continue
        groups = match.groupdict()
        parsed_mode = mode or groups.get("mode")
        parsed_condition = condition or f"forced_{parsed_mode}"
        return {
            "condition": parsed_condition,
            "fast_write_phase": phase,
            "fast_nonwrite_mode": parsed_mode,
            "eta": parse_float_tag(groups.get("eta")),
            "beta": parse_float_tag(groups.get("beta")),
            "settle_steps": int(groups["settle"]),
            "seed": int(groups["seed"]),
        }
    raise ValueError(f"Unrecognized run name: {run}")


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    metadata = pd.DataFrame([parse_run(run) for run in df["run"]])
    df = pd.concat([metadata, df], axis=1)

    metrics = [metric for metric in PRIMARY_METRICS if metric in df.columns]
    group_cols = [
        "condition",
        "fast_write_phase",
        "fast_nonwrite_mode",
        "eta",
        "beta",
        "settle_steps",
    ]
    grouped = df.groupby(group_cols, dropna=False)
    summary = pd.concat(
        [
            grouped.size().rename("n_seeds"),
            grouped[metrics].mean().add_suffix("_mean"),
            grouped[metrics].std(ddof=1).add_suffix("_std"),
        ],
        axis=1,
    ).reset_index()

    sort_metric = "eightarm_rollout/dynamic_step_success_rate_mean"
    if sort_metric in summary.columns:
        summary = summary.sort_values(sort_metric, ascending=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/runs/write_schedule_run_level.csv")
    parser.add_argument("--output", default="data/runs/write_schedule_summary.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
