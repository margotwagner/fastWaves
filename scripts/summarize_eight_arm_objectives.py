#!/usr/bin/env python3
"""Aggregate the three-seed objective comparison by model and objective."""

import argparse
import re
from pathlib import Path

import pandas as pd

RUN_RE = re.compile(
    r"^objectivecmp_(wave|fastwave)_(valid_set|commitment)_xv_h1_s\d+_seed\d+$"
)

KEY_METRICS = [
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
]


def parse_run(run: str) -> tuple[str, str]:
    match = RUN_RE.match(run)
    if match is None:
        raise ValueError(f"Unexpected run name: {run}")
    return match.group(1), match.group(2)


def main(args):
    df = pd.read_csv(args.input)
    if "run" not in df.columns:
        raise ValueError("Input CSV must contain a 'run' column.")

    parsed = df["run"].map(parse_run)
    df["model"] = parsed.map(lambda x: x[0])
    df["objective"] = parsed.map(lambda x: x[1])

    metrics = [metric for metric in KEY_METRICS if metric in df.columns]
    if not metrics:
        raise ValueError("None of the expected eight-arm metrics were found.")

    grouped = df.groupby(["model", "objective"], sort=True)
    rows = []
    for (model, objective), group in grouped:
        row = {
            "model": model,
            "objective": objective,
            "n_seeds": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["model", "objective"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
