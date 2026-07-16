#!/usr/bin/env python3
"""Aggregate multi-seed successor-recall results."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


def parse_run_name(name: str):
    match = re.match(
        r"successor_(vanilla|gru|wave|fastwave)_L(\d+)_d(\d+)_q(\d+)_seed(\d+)$",
        name,
    )
    if match is None:
        raise ValueError(f"Unexpected run name: {name}")
    model, length, delay, query_hold, seed = match.groups()
    return model, int(length), int(delay), int(query_hold), int(seed)


def read_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return {str(row.metric): float(row.value) for row in frame.itertuples()}


def main(args: argparse.Namespace) -> None:
    paths = sorted(
        glob.glob(str(Path(args.runs_dir) / args.pattern / "successor_analysis/metrics.csv"))
    )
    if not paths:
        raise FileNotFoundError("No successor analysis metrics matched the pattern")

    rows = []
    for raw in paths:
        path = Path(raw)
        run_name = path.parent.parent.name
        model, length, delay, query_hold, seed = parse_run_name(run_name)
        row = {
            "run": run_name,
            "model": model,
            "sequence_length": length,
            "delay_steps": delay,
            "query_hold_steps": query_hold,
            "seed": seed,
        }
        row.update(read_metrics(path))
        rows.append(row)

    run_level = pd.DataFrame(rows).sort_values(["model", "seed"])
    out_run = Path(args.out_run_level)
    out_run.parent.mkdir(parents=True, exist_ok=True)
    run_level.to_csv(out_run, index=False)

    metrics = [
        "successor_test_accuracy",
        "successor_test_cross_entropy",
        "successor_test_mean_top1_probability",
    ]
    summary_parts = []
    for model, group in run_level.groupby("model", sort=False):
        row = {"model": model, "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_parts.append(row)
    summary = pd.DataFrame(summary_parts)
    out_summary = Path(args.out_summary)
    summary.to_csv(out_summary, index=False)

    print(run_level.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved {out_run} and {out_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="data/runs")
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out-run-level", required=True)
    parser.add_argument("--out-summary", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
