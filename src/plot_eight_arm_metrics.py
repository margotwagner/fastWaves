#!/usr/bin/env python3
"""
Plot the eight-arm radial-maze delay sweep.

Main output:
    eight_arm_behavior_summary.png/.pdf
        A 2x2 figure showing:
        A. Valid unvisited action selection
        B. Valid unvisited spatial routing
        C. Action-routing agreement, conditioned on departure
        D. End-to-end action-to-target routing success

Optional output:
    eight_arm_mse_vs_behavior.png/.pdf
        Rollout MSE versus end-to-end task success.

Example:
    python plot_eight_arm_metrics.py \
        --csv data/runs/eight_arm_bump_traj_action_routingw50_s0to10.csv \
        --out-dir data/runs/figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["vanilla", "globalfast", "localfast", "wave", "fastwave"]
MODEL_LABELS = {
    "vanilla": "Vanilla RNN",
    "globalfast": "Global fast",
    "localfast": "Local fast",
    "wave": "Wave",
    "fastwave": "FastWave",
}

MAIN_METRICS = [
    (
        "eightarm_rollout/action_selection_valid_unvisited_rate",
        "A  Valid action selection",
        "Valid unvisited rate",
    ),
    (
        "eightarm_rollout/routing_valid_unvisited_rate",
        "B  Valid spatial routing",
        "Valid unvisited rate",
    ),
    (
        "eightarm_rollout/routing_spatial_matches_arm_head_given_departure_rate",
        "C  Action–routing agreement after departure",
        "Agreement rate",
    ),
    (
        "eightarm_rollout/action_to_target_routing_success_rate",
        "D  End-to-end task success",
        "Success rate",
    ),
]


def parse_model(run_name: str) -> str:
    """Infer model name from the beginning of a run directory/name."""
    run_name = str(run_name).lower()
    for model in ["globalfast", "localfast", "fastwave", "vanilla", "wave"]:
        if run_name.startswith(model + "_") or run_name == model:
            return model
    raise ValueError(f"Could not infer model from run name: {run_name!r}")


def parse_settle_steps(run_name: str) -> int:
    """Extract s0, s2, s5, s10, etc. from a run name."""
    match = re.search(r"(?:^|_)s(\d+)(?:_|$)", str(run_name).lower())
    if match is None:
        raise ValueError(f"Could not infer settle steps from run name: {run_name!r}")
    return int(match.group(1))


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "run" not in df.columns:
        raise ValueError("CSV must contain a 'run' column.")

    missing = [metric for metric, _, _ in MAIN_METRICS if metric not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required metric columns:\n  " + "\n  ".join(missing)
        )

    df = df.copy()
    df["model"] = df["run"].map(parse_model)
    df["settle_steps"] = df["run"].map(parse_settle_steps)

    # Ensure numeric columns are numeric.
    metric_columns = [metric for metric, _, _ in MAIN_METRICS]
    optional = ["rollout_mse_mean"]
    for column in metric_columns + optional:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    duplicates = df.duplicated(["model", "settle_steps"], keep=False)
    if duplicates.any():
        names = df.loc[duplicates, ["run", "model", "settle_steps"]]
        raise ValueError(
            "Expected one run per model/delay, but found duplicates:\n"
            + names.to_string(index=False)
        )

    return df


def available_models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"])
    return [model for model in MODEL_ORDER if model in present]


def style_axis(ax: plt.Axes, settle_values: list[int]) -> None:
    ax.set_xticks(settle_values)
    ax.set_xlabel("Settle delay (steps)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_behavior_summary(
    df: pd.DataFrame,
    out_dir: Path,
    basename: str = "eight_arm_behavior_summary",
) -> None:
    models = available_models(df)
    settle_values = sorted(df["settle_steps"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(axes, MAIN_METRICS):
        for model in models:
            model_df = (
                df.loc[df["model"] == model]
                .sort_values("settle_steps")
            )
            ax.plot(
                model_df["settle_steps"],
                model_df[metric],
                marker="o",
                linewidth=2.1,
                markersize=5.5,
                label=MODEL_LABELS[model],
            )

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        style_axis(ax, settle_values)

        # Chance level for choosing one of four valid arms out of eight.
        if metric == "eightarm_rollout/action_selection_valid_unvisited_rate":
            ax.axhline(
                0.5,
                linestyle="--",
                linewidth=1.2,
                alpha=0.65,
                label="Random-arm baseline",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        "Memory-guided action selection and spatial routing",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        "Eight-arm bump-trajectory task; autonomous rollout; routing weight = 50",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{basename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mse_vs_success(
    df: pd.DataFrame,
    out_dir: Path,
    basename: str = "eight_arm_mse_vs_behavior",
) -> None:
    if "rollout_mse_mean" not in df.columns:
        print("Skipping MSE-versus-success figure: rollout_mse_mean is absent.")
        return

    success_col = "eightarm_rollout/action_to_target_routing_success_rate"
    models = available_models(df)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    for model in models:
        model_df = df.loc[df["model"] == model].sort_values("settle_steps")
        ax.scatter(
            model_df["rollout_mse_mean"],
            model_df[success_col],
            s=65,
            label=MODEL_LABELS[model],
        )
    # Add delay labels safely using row indexing.
    for _, row in df.iterrows():
        if pd.notna(row["rollout_mse_mean"]) and pd.notna(row[success_col]):
            ax.annotate(
                f"s{int(row['settle_steps'])}",
                (row["rollout_mse_mean"], row[success_col]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

    ax.set_title(
        "Low prediction error does not guarantee task success",
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Autonomous rollout MSE (lower is better)")
    ax.set_ylabel("End-to-end action-to-target routing success")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{basename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plot_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Save only the columns used in the figures for easy inspection."""
    columns = [
        "run",
        "model",
        "settle_steps",
        *[metric for metric, _, _ in MAIN_METRICS],
    ]
    if "rollout_mse_mean" in df.columns:
        columns.append("rollout_mse_mean")

    (
        df[columns]
        .sort_values(["model", "settle_steps"])
        .to_csv(out_dir / "eight_arm_plot_data.csv", index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.csv)

    plot_behavior_summary(df, args.out_dir)
    plot_mse_vs_success(df, args.out_dir)
    save_plot_table(df, args.out_dir)

    print(f"Saved figures and plot data to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
