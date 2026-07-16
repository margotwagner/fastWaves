#!/usr/bin/env python3
"""Aggregate multi-seed transition-recall results."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


def parse_run_name(name: str):
    pattern = (
        r"transition_(sanity|independent)(?:_norm)?_"
        r"(vanilla|gru|wave|fastwave)_"
        r"p(\d+)_d(\d+)_q(\d+)_seed(\d+)$"
    )
    match = re.match(pattern, name)
    if match is None:
        raise ValueError(f"Unexpected run name: {name}")
    condition, model, pairs, delay, query_hold, seed = match.groups()
    return condition, model, int(pairs), int(delay), int(query_hold), int(seed)


def read_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return {str(row.metric): float(row.value) for row in frame.itertuples()}


def main(args: argparse.Namespace) -> None:
    paths = sorted(
        glob.glob(
            str(Path(args.runs_dir) / args.pattern / "transition_analysis/metrics.csv")
        )
    )
    if not paths:
        raise FileNotFoundError("No transition-analysis metrics matched the pattern")

    rows = []
    for raw in paths:
        path = Path(raw)
        run_name = path.parent.parent.name
        condition, model, pairs, delay, query_hold, seed = parse_run_name(run_name)
        row = {
            "run": run_name,
            "condition": condition,
            "model": model,
            "n_pairs": pairs,
            "delay_steps": delay,
            "query_hold_steps": query_hold,
            "seed": seed,
        }
        row.update(read_metrics(path))
        rows.append(row)

    run_level = pd.DataFrame(rows).sort_values(["condition", "model", "seed"])
    out_run = Path(args.out_run_level)
    out_run.parent.mkdir(parents=True, exist_ok=True)
    run_level.to_csv(out_run, index=False)

    metrics = [
        "transition_test_accuracy",
        "transition_test_cross_entropy",
        "transition_test_mean_top1_probability",
    ]
    summary_rows = []
    for (condition, model), group in run_level.groupby(
        ["condition", "model"], sort=False
    ):
        row = {"condition": condition, "model": model, "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    out_summary = Path(args.out_summary)
    summary.to_csv(out_summary, index=False)

    print(run_level.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved {out_run} and {out_summary}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="data/runs")
    p.add_argument("--pattern", required=True)
    p.add_argument("--out-run-level", required=True)
    p.add_argument("--out-summary", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
