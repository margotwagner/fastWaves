#!/usr/bin/env python3
"""
Visualization demo for the eight-arm bump trajectory working-memory task.

Creates a publication/demo-style figure with:
  1. Eight-arm radial-maze task schematic.
  2. Example trial timeline from EightArmBumpTrajectoryDataset.
  3. Model metric comparison: teacher-forced vs rollout.
  4. Rollout-specific decision/routing summary.

Example:
  python visualize_eight_arm_results.py \
    --csv eight_arm_bump_traj_s5_action_routingw20_comparison.csv \
    --tasks-py tasks.py \
    --out eight_arm_task_results_demo.png
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


MODEL_ORDER = ["vanilla", "wave", "globalfast", "localfast", "fastwave"]
MODEL_LABELS = {
    "vanilla": "Vanilla RNN",
    "wave": "Wave RNN",
    "globalfast": "GlobalFast RNN",
    "localfast": "LocalFast RNN",
    "fastwave": "FastWave RNN",
}


def clean_model_name(run_name: str) -> str:
    """Extract compact model name from run string."""
    s = str(run_name).lower()
    for key in ["globalfast", "localfast", "fastwave", "vanilla", "wave"]:
        if s.startswith(key) or key in s:
            return key
    return str(run_name)


def load_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "run" not in df.columns:
        raise ValueError("Expected a 'run' column in the comparison CSV.")
    df = df.copy()
    df["model"] = df["run"].map(clean_model_name)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["_order"] = df["model"].map(order).fillna(999)
    return df.sort_values("_order").reset_index(drop=True)


def load_tasks_module(tasks_py: str | Path | None) -> Any | None:
    if tasks_py is None:
        return None
    path = Path(tasks_py)
    if not path.exists():
        raise FileNotFoundError(f"Could not find tasks.py at {path}")
    spec = importlib.util.spec_from_file_location("uploaded_tasks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_example_trial(tasks_module: Any | None, seed: int = 5):
    """Return dataset object and one example x,y pair if tasks.py is available."""
    if tasks_module is None or not hasattr(tasks_module, "EightArmBumpTrajectoryDataset"):
        return None, None, None
    ds = tasks_module.EightArmBumpTrajectoryDataset(
        n_samples=1,
        seq_len=65,
        n_space=40,
        n_arms=8,
        n_forced=4,
        arm_len=3,
        settle_steps=2,
        reward_hold_steps=1,
        center_hold_steps=0,
        choice_order="random",
        bump_sigma=0.75,
        seed=seed,
    )
    item = ds[0]
    x, y = item[0].numpy(), item[1].numpy()
    return ds, x, y


def radial_coordinates(n_arms: int = 8, arm_len: int = 3):
    """Coordinates for center plus arm depth nodes."""
    coords = {0: (0.0, 0.0)}
    for arm in range(n_arms):
        theta = 2 * math.pi * arm / n_arms + math.pi / 2
        for depth in range(arm_len):
            r = depth + 1
            idx = 1 + arm * arm_len + depth
            coords[idx] = (r * math.cos(theta), r * math.sin(theta))
    return coords


def decode_pos(ds: Any, frame: np.ndarray) -> int:
    return int(np.argmax(frame[: ds.n_pos]))


def decode_phase(ds: Any, frame: np.ndarray) -> str:
    if frame[ds.cue_forced] > 0.5:
        return "forced"
    if frame[ds.cue_choice] > 0.5:
        return "choice"
    if frame[ds.cue_settle] > 0.5:
        return "settle"
    return "none"


def add_task_schematic(ax, ds: Any | None, x: np.ndarray | None):
    ax.set_title("Eight-arm memory task", fontsize=14, fontweight="bold", loc="left")
    ax.set_aspect("equal")
    ax.axis("off")

    n_arms, arm_len = 8, 3
    coords = radial_coordinates(n_arms, arm_len)

    forced, choice = [], []
    if ds is not None:
        forced = [int(a) for a in ds.forced_orders[0].tolist()]
        choice = [int(a) for a in ds.choice_orders[0].tolist()]
    else:
        forced, choice = [0, 3, 5, 6], [1, 2, 4, 7]

    for arm in range(n_arms):
        xs, ys = [0.0], [0.0]
        for depth in range(arm_len):
            px, py = coords[1 + arm * arm_len + depth]
            xs.append(px); ys.append(py)
        lw = 3.0 if arm in forced or arm in choice else 1.5
        alpha = 0.95 if arm in forced or arm in choice else 0.35
        linestyle = "-" if arm in forced else "--" if arm in choice else ":"
        ax.plot(xs, ys, linewidth=lw, alpha=alpha, linestyle=linestyle)
        ox, oy = coords[1 + arm * arm_len + (arm_len - 1)]
        ax.text(1.12 * ox, 1.12 * oy, str(arm), ha="center", va="center", fontsize=10)

    # Nodes and center
    for idx, (px, py) in coords.items():
        size = 95 if idx == 0 else 35
        ax.scatter([px], [py], s=size, edgecolor="black", linewidth=0.6, zorder=5)
    ax.text(0, -0.32, "center", ha="center", va="top", fontsize=9)

    ax.text(
        -3.55, -3.25,
        "Solid = forced visits\nDashed = correct unvisited choices\nAt center, sensory state is ambiguous",
        fontsize=9,
        va="bottom",
    )


def add_trial_timeline(ax, ds: Any | None, x: np.ndarray | None):
    ax.set_title("Example trial timeline", fontsize=14, fontweight="bold", loc="left")
    if ds is None or x is None:
        ax.text(0.5, 0.5, "Pass --tasks-py to render a real trial", ha="center", va="center")
        ax.axis("off")
        return

    pos = np.array([decode_pos(ds, frame) for frame in x])
    phase = np.array([decode_phase(ds, frame) for frame in x])
    arm = np.full_like(pos, fill_value=-1)
    depth = np.full_like(pos, fill_value=-1)
    for t, p in enumerate(pos):
        if p > 0:
            z = p - 1
            arm[t] = z // ds.arm_len
            depth[t] = z % ds.arm_len

    # Plot position index as a timeline, with phase bands underneath.
    t = np.arange(len(pos))
    ax.plot(t, pos, marker="o", markersize=2.7, linewidth=1.5, label="Spatial position channel")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Position index")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    phase_to_y = {"forced": -3, "settle": -5, "choice": -7}
    for ph, y in phase_to_y.items():
        mask = phase == ph
        ax.scatter(t[mask], np.full(mask.sum(), y), marker="s", s=16, label=ph.capitalize())
    ax.set_ylim(-8.5, ds.n_pos + 1)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")

    forced_txt = "Forced: " + " → ".join(map(str, ds.forced_orders[0].tolist()))
    choice_txt = "Choice target: " + " → ".join(map(str, ds.choice_orders[0].tolist()))
    ax.text(0.01, 0.96, forced_txt + "\n" + choice_txt, transform=ax.transAxes, va="top", fontsize=9)


def add_metric_bars(ax, df: pd.DataFrame, tf_col: str, ro_col: str, title: str, ylabel: str, lower_is_better: bool):
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    models = df["model_label"].tolist()
    x = np.arange(len(models))
    width = 0.36
    ax.bar(x - width / 2, df[tf_col], width, label="Teacher forced")
    ax.bar(x + width / 2, df[ro_col], width, label="Rollout")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    note = "Lower is better" if lower_is_better else "Higher is better"
    ax.text(0.98, 0.96, note, transform=ax.transAxes, ha="right", va="top", fontsize=8)


def add_rollout_summary(ax, df: pd.DataFrame):
    ax.set_title("Autonomous rollout summary", fontsize=14, fontweight="bold", loc="left")
    models = df["model_label"].tolist()
    x = np.arange(len(models))
    width = 0.28
    cols = [
        ("eightarm_rollout/position_acc", "Position acc."),
        ("eightarm_rollout/arm_choice_head_acc", "Arm-head acc."),
        ("eightarm_rollout/valid_unvisited_choice_rate", "Valid unvisited"),
    ]
    for i, (col, label) in enumerate(cols):
        ax.bar(x + (i - 1) * width, df[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy / rate")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8, ncol=1)


def add_horizon_plot(ax, df: pd.DataFrame):
    ax.set_title("Prediction horizon", fontsize=14, fontweight="bold", loc="left")
    models = df["model_label"].tolist()
    vals = df["prediction_horizon_steps"].values
    y = np.arange(len(models))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("Steps maintained during rollout")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.35)
    best = int(np.argmax(vals))
    ax.text(vals[best], best, f"  best: {vals[best]:.1f}", va="center", fontsize=9)


def make_figure(csv_path: str | Path, tasks_py: str | Path | None, out_path: str | Path, seed: int = 5):
    df = load_results(csv_path)
    tasks_module = load_tasks_module(tasks_py) if tasks_py else None
    ds, x, y = make_example_trial(tasks_module, seed=seed)

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=False)
    gs = GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[1.08, 1, 1],
        width_ratios=[1.05, 1, 1],
        left=0.055,
        right=0.985,
        bottom=0.07,
        top=0.86,
        wspace=0.40,
        hspace=0.48,
    )

    ax_task = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1:])
    ax_mse = fig.add_subplot(gs[1, 0])
    ax_angle = fig.add_subplot(gs[1, 1])
    ax_horizon = fig.add_subplot(gs[1, 2])
    ax_rollout = fig.add_subplot(gs[2, :])

    add_task_schematic(ax_task, ds, x)
    add_trial_timeline(ax_time, ds, x)
    add_metric_bars(ax_mse, df, "teacher_forced_mse", "rollout_mse_mean", "MSE: supervised vs autonomous", "MSE", True)
    add_metric_bars(ax_angle, df, "teacher_forced_angle_error_rad", "rollout_angle_error_rad", "Angular error", "Radians", True)
    add_horizon_plot(ax_horizon, df)
    add_rollout_summary(ax_rollout, df)

    title = "Eight-arm bump trajectory task: memory-dependent routing and model rollouts"
    subtitle = "Forced visits define trial-specific memory; after the settle period, the model must autonomously choose and route through unvisited arms."
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.975)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Comparison CSV from compare_metrics.py")
    parser.add_argument("--tasks-py", default=None, help="Path to tasks.py for task reconstruction")
    parser.add_argument("--out", default="eight_arm_task_results_demo.png", help="Output PNG/PDF/SVG path")
    parser.add_argument("--seed", type=int, default=5, help="Example trial seed")
    args = parser.parse_args()
    make_figure(args.csv, args.tasks_py, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
