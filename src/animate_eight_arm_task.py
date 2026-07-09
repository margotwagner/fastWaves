#!/usr/bin/env python3
"""
Animate the eight-arm bump trajectory task.

The video shows:
  1. The eight-arm radial maze with a smooth Gaussian bump routed through it.
  2. The current phase: forced, settle, or choice.
  3. The explicit 8-channel arm-choice/action head.
  4. The forced-arm memory set and the correct unvisited choice set.
  5. A compact time cursor showing when the task switches from forced to choice.

Example:
  python animate_eight_arm_task.py \
    --tasks-py tasks.py \
    --out eight_arm_task_movie.mp4 \
    --seed 5

Optional: pass --pred-npy rollout.npy to animate model outputs instead of the target
input sequence. The array should be [T, n_space] or [1, T, n_space].
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.gridspec import GridSpec


def load_tasks_module(tasks_py: str | Path) -> Any:
    path = Path(tasks_py)
    if not path.exists():
        raise FileNotFoundError(f"Could not find tasks.py at {path}")
    spec = importlib.util.spec_from_file_location("uploaded_tasks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_dataset(tasks_module: Any, args: argparse.Namespace):
    if not hasattr(tasks_module, "EightArmBumpTrajectoryDataset"):
        raise AttributeError("tasks.py does not define EightArmBumpTrajectoryDataset")

    return tasks_module.EightArmBumpTrajectoryDataset(
        n_samples=1,
        seq_len=args.seq_len,
        n_space=args.n_space,
        n_arms=args.n_arms,
        n_forced=args.n_forced,
        arm_len=args.arm_len,
        settle_steps=args.settle_steps,
        reward_hold_steps=args.reward_hold_steps,
        center_hold_steps=args.center_hold_steps,
        choice_order=args.choice_order,
        bump_sigma=args.bump_sigma,
        seed=args.seed,
    )


def radial_coordinates(n_arms: int = 8, arm_len: int = 3) -> np.ndarray:
    """Return [n_pos, 2] coordinates for center plus radial arm-depth positions."""
    coords = [(0.0, 0.0)]
    for arm in range(n_arms):
        theta = 2 * math.pi * arm / n_arms + math.pi / 2
        for depth in range(arm_len):
            r = depth + 1
            coords.append((r * math.cos(theta), r * math.sin(theta)))
    return np.asarray(coords, dtype=float)


def decode_phase(ds: Any, frame: np.ndarray) -> str:
    if frame[ds.cue_forced] > 0.5:
        return "forced"
    if frame[ds.cue_choice] > 0.5:
        return "choice"
    if frame[ds.cue_settle] > 0.5:
        return "settle"
    return "none"


def decode_direction(ds: Any, frame: np.ndarray) -> str:
    if frame[ds.cue_outbound] > 0.5:
        return "outbound"
    if frame[ds.cue_inbound] > 0.5:
        return "inbound"
    if frame[ds.cue_reward] > 0.5:
        return "reward"
    if frame[ds.cue_center] > 0.5:
        return "center"
    return "none"


def decode_peak_position(ds: Any, frame: np.ndarray) -> int:
    return int(np.argmax(frame[: ds.n_pos]))


def pos_to_arm_depth(ds: Any, pos_idx: int) -> tuple[int | None, int | None]:
    if pos_idx == ds.center_idx:
        return None, None
    z = pos_idx - 1
    return int(z // ds.arm_len), int(z % ds.arm_len)


def phase_color(phase: str) -> str:
    return {
        "forced": "#4C78A8",
        "settle": "#9E9E9E",
        "choice": "#F58518",
    }.get(phase, "#BBBBBB")


def load_sequence(ds: Any, args: argparse.Namespace) -> np.ndarray:
    item = ds[0]
    x = item[0].detach().cpu().numpy()

    if args.pred_npy is None:
        return x

    pred = np.load(args.pred_npy)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError("--pred-npy must have shape [T, n_space] or [1, T, n_space]")
    if pred.shape[1] < ds.min_n_space:
        raise ValueError(
            f"Prediction has only {pred.shape[1]} channels, but this task needs at least {ds.min_n_space}."
        )
    return pred[: x.shape[0], :]


def draw_static_maze(ax, ds: Any, coords: np.ndarray, forced: list[int], choice: list[int]):
    ax.set_aspect("equal")
    ax.axis("off")

    for arm in range(ds.n_arms):
        arm_coords = [coords[0]] + [coords[ds.arm_pos_idx(arm, d)] for d in range(ds.arm_len)]
        arm_coords = np.asarray(arm_coords)
        if arm in forced:
            color, lw, ls, alpha = "#4C78A8", 4.0, "-", 0.95
        elif arm in choice:
            color, lw, ls, alpha = "#F58518", 4.0, "--", 0.95
        else:
            color, lw, ls, alpha = "#CCCCCC", 2.0, ":", 0.55
        ax.plot(arm_coords[:, 0], arm_coords[:, 1], color=color, lw=lw, ls=ls, alpha=alpha, zorder=1)

        outer = arm_coords[-1]
        ax.text(1.17 * outer[0], 1.17 * outer[1], str(arm), ha="center", va="center", fontsize=11)

    ax.scatter(coords[:, 0], coords[:, 1], s=45, c="white", edgecolors="black", linewidths=0.8, zorder=2)
    ax.scatter([0], [0], s=120, c="white", edgecolors="black", linewidths=1.1, zorder=3)
    ax.text(0, -0.36, "center", ha="center", va="top", fontsize=9)

    ax.set_xlim(-4.05, 4.05)
    ax.set_ylim(-4.05, 4.05)


def make_animation(args: argparse.Namespace):
    tasks_module = load_tasks_module(args.tasks_py)
    ds = make_dataset(tasks_module, args)
    seq = load_sequence(ds, args)
    target = ds[0][0].detach().cpu().numpy()

    forced = [int(a) for a in ds.forced_orders[0].tolist()]
    choice = [int(a) for a in ds.choice_orders[0].tolist()]
    coords = radial_coordinates(ds.n_arms, ds.arm_len)
    T = seq.shape[0]

    pos_mat = seq[:, : ds.n_pos]
    arm_head = seq[:, ds.arm_choice_start : ds.arm_choice_end]
    true_phase = [decode_phase(ds, frame) for frame in target]

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(13.5, 7.7), constrained_layout=False)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.25, 1.0, 1.0],
        height_ratios=[1.25, 0.82],
        left=0.055,
        right=0.985,
        bottom=0.08,
        top=0.88,
        wspace=0.35,
        hspace=0.34,
    )

    ax_maze = fig.add_subplot(gs[:, 0])
    ax_bump = fig.add_subplot(gs[0, 1:])
    ax_arm = fig.add_subplot(gs[1, 1])
    ax_time = fig.add_subplot(gs[1, 2])

    draw_static_maze(ax_maze, ds, coords, forced, choice)
    bump_scatter = ax_maze.scatter([], [], s=[], c=[], cmap="viridis", vmin=0, vmax=1, edgecolors="black", linewidths=0.35, zorder=6)
    peak_marker = ax_maze.scatter([], [], s=240, facecolors="none", edgecolors="#D62728", linewidths=2.4, zorder=7)
    title_text = ax_maze.text(-3.9, 3.85, "", ha="left", va="top", fontsize=14, fontweight="bold")
    state_text = ax_maze.text(-3.9, -3.85, "", ha="left", va="bottom", fontsize=10)

    x_pos = np.arange(ds.n_pos)
    bump_line, = ax_bump.plot(x_pos, pos_mat[0], lw=2.2)
    peak_vline = ax_bump.axvline(0, lw=1.5, ls="--", alpha=0.8)
    ax_bump.set_xlim(0, ds.n_pos - 1)
    ax_bump.set_ylim(-0.03, 1.08)
    ax_bump.set_xlabel("Spatial channel: center + arm-depth nodes")
    ax_bump.set_ylabel("Gaussian bump amplitude")
    ax_bump.set_title("Smooth bump routed through radial-maze graph", loc="left", fontweight="bold")
    ax_bump.grid(True, alpha=0.28, linewidth=0.5)
    for arm in range(ds.n_arms):
        start = ds.arm_pos_idx(arm, 0)
        end = ds.arm_pos_idx(arm, ds.arm_len - 1)
        ax_bump.axvspan(start - 0.45, end + 0.45, alpha=0.04)
        ax_bump.text((start + end) / 2, 1.03, f"A{arm}", ha="center", va="bottom", fontsize=8)

    arm_bars = ax_arm.bar(np.arange(ds.n_arms), arm_head[0], width=0.72)
    ax_arm.set_ylim(0, 1.08)
    ax_arm.set_xticks(np.arange(ds.n_arms))
    ax_arm.set_xlabel("Arm")
    ax_arm.set_ylabel("Action / arm-choice head")
    ax_arm.set_title("Predicted/chosen arm channel", loc="left", fontweight="bold")
    ax_arm.grid(True, axis="y", alpha=0.28, linewidth=0.5)

    # Compact phase timeline: draw colored bands and an updating cursor.
    ax_time.set_title("Trial phases and required memory", loc="left", fontweight="bold")
    for t, ph in enumerate(true_phase):
        ax_time.axvspan(t - 0.5, t + 0.5, color=phase_color(ph), alpha=0.36, lw=0)
    time_cursor = ax_time.axvline(0, color="black", lw=2.0)
    ax_time.set_xlim(-0.5, T - 0.5)
    ax_time.set_ylim(0, 1)
    ax_time.set_yticks([])
    ax_time.set_xlabel("Time step")

    forced_text = "Forced memory set: " + ", ".join(map(str, forced))
    choice_text = "Correct unvisited choices: " + ", ".join(map(str, choice))
    ax_time.text(0.02, 0.78, forced_text, transform=ax_time.transAxes, color="#4C78A8", fontsize=10)
    ax_time.text(0.02, 0.58, choice_text, transform=ax_time.transAxes, color="#F58518", fontsize=10)
    ax_time.text(
        0.02,
        0.17,
        "At choice-center frames, the spatial input is at center;\nthe next arm depends on remembered forced visits.",
        transform=ax_time.transAxes,
        fontsize=9,
    )

    fig.suptitle("Eight-arm bump trajectory task: forced phase → choice phase", fontsize=16, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.925,
        "Animation shows the target task sequence by default; pass --pred-npy to show a model rollout sequence.",
        ha="center",
        va="top",
        fontsize=10,
    )

    def update(t: int):
        frame = seq[t]
        target_frame = target[t]
        pos_vals = np.clip(frame[: ds.n_pos], 0.0, 1.0)
        peak = decode_peak_position(ds, frame)
        arm, depth = pos_to_arm_depth(ds, peak)
        ph = decode_phase(ds, target_frame)
        direction = decode_direction(ds, target_frame)
        arm_vals = np.clip(frame[ds.arm_choice_start : ds.arm_choice_end], 0.0, 1.0)
        chosen = int(np.argmax(arm_vals)) if arm_vals.max() > 0.05 else None

        sizes = 40 + 680 * (pos_vals ** 1.5)
        bump_scatter.set_offsets(coords)
        bump_scatter.set_sizes(sizes)
        bump_scatter.set_array(pos_vals)
        peak_marker.set_offsets(coords[[peak]])

        if arm is None:
            pos_label = "center"
        else:
            pos_label = f"arm {arm}, depth {depth}"
        chosen_label = "none" if chosen is None else f"arm {chosen}"

        title_text.set_text(f"t = {t:02d} | {ph.upper()} | {direction}")
        state_text.set_text(
            f"Bump peak: {pos_label}\n"
            f"Arm-choice head: {chosen_label}\n"
            f"Forced: {forced}\nChoice target: {choice}"
        )

        bump_line.set_ydata(pos_vals)
        peak_vline.set_xdata([peak, peak])

        for i, bar in enumerate(arm_bars):
            bar.set_height(float(arm_vals[i]))
            if i in forced:
                bar.set_facecolor("#4C78A8")
                bar.set_alpha(0.45)
            elif i in choice:
                bar.set_facecolor("#F58518")
                bar.set_alpha(0.75)
            else:
                bar.set_facecolor("#BBBBBB")
                bar.set_alpha(0.35)
            if chosen is not None and i == chosen:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.0)
            else:
                bar.set_edgecolor("none")
                bar.set_linewidth(0.0)

        time_cursor.set_xdata([t, t])
        return [bump_scatter, peak_marker, title_text, state_text, bump_line, peak_vline, *arm_bars, time_cursor]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / args.fps, blit=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=args.bitrate)
    anim.save(out, writer=writer)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-py", required=True, help="Path to your tasks.py")
    parser.add_argument("--out", default="eight_arm_task_movie.mp4", help="Output .mp4 or .gif")
    parser.add_argument("--pred-npy", default=None, help="Optional [T,N] or [1,T,N] model rollout array")
    parser.add_argument("--seed", type=int, default=5, help="Trial seed")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--bitrate", type=int, default=2400, help="FFmpeg bitrate")

    # Match your current task defaults from the routingw20 experiments.
    parser.add_argument("--seq-len", type=int, default=65)
    parser.add_argument("--n-space", type=int, default=40)
    parser.add_argument("--n-arms", type=int, default=8)
    parser.add_argument("--n-forced", type=int, default=4)
    parser.add_argument("--arm-len", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--reward-hold-steps", type=int, default=1)
    parser.add_argument("--center-hold-steps", type=int, default=0)
    parser.add_argument("--choice-order", choices=["random", "ascending"], default="random")
    parser.add_argument("--bump-sigma", type=float, default=0.75)

    args = parser.parse_args()
    make_animation(args)


if __name__ == "__main__":
    main()
