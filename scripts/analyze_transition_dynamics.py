#!/usr/bin/env python3
"""Inspect hidden activity and fast-weight writes/retrievals over a transition trial."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.tasks import build_dataset
from src.train import build_model


def safe_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = a.norm() * b.norm()
    if float(denom) < eps:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def phase_label(ds, frame: torch.Tensor, t: int) -> str:
    if frame[ds.cue_reset] > 0.5:
        return "reset"
    if frame[ds.cue_write] > 0.5:
        return "target/write"
    if frame[ds.cue_source] > 0.5:
        return "query" if t >= ds.query_start else "source"
    if frame[ds.cue_center] > 0.5:
        return "delay/center"
    return "other"


def save_state_plot(extras: dict, out_dir: Path, title: str) -> None:
    if "wave_state" in extras:
        x = extras["wave_state"][0].detach().cpu()
        v = extras["wave_velocity"][0].detach().cpu()
        # Collapse channels only for visualization.
        x_view = x.pow(2).sum(dim=1).sqrt().T
        v_view = v.pow(2).sum(dim=1).sqrt().T

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        im0 = axes[0].imshow(x_view, aspect="auto", origin="lower")
        axes[0].set_ylabel("Latent site")
        axes[0].set_title("Wave-state magnitude")
        fig.colorbar(im0, ax=axes[0], fraction=0.025)
        im1 = axes[1].imshow(v_view, aspect="auto", origin="lower")
        axes[1].set_ylabel("Latent site")
        axes[1].set_xlabel("Time")
        axes[1].set_title("Velocity magnitude")
        fig.colorbar(im1, ax=axes[1], fraction=0.025)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / "hidden_state_progression.png", dpi=180)
        plt.close(fig)
    elif "hidden" in extras:
        h = extras["hidden"][0].detach().cpu().T
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(h, aspect="auto", origin="lower")
        ax.set_xlabel("Time")
        ax.set_ylabel("Hidden unit")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "hidden_state_progression.png", dpi=180)
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = safe_load(args.ckpt, device)
    train_args = SimpleNamespace(**ckpt["args"])
    if getattr(train_args, "task", None) != "eight_arm_transition_recall":
        raise ValueError("Checkpoint was not trained on eight_arm_transition_recall")

    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ds = build_dataset(train_args, n_samples=max(args.trial + 1, 1), seed=args.seed)

    x = ds.x[args.trial : args.trial + 1].to(device)
    with torch.no_grad():
        yhat, extras = model(x, record_all=True)

    out_dir = Path(args.out_dir)
    if args.out_dir == "auto":
        out_dir = Path(args.ckpt).parent / "transition_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_state_plot(extras, out_dir, f"{train_args.model}: transition trial {args.trial}")

    pred_t = int(ds.prediction_times[args.trial])
    logits = yhat[0, pred_t, ds.arm_choice_start : ds.arm_choice_end]
    probs = torch.softmax(logits, dim=-1).detach().cpu()
    predicted = int(probs.argmax())
    target = int(ds.successor_targets[args.trial])

    rows = []
    T = x.shape[1]
    for t in range(T):
        row = {
            "time": t,
            "phase": phase_label(ds, x[0, t].detach().cpu(), t),
            "write_gate": float(extras.get("fast_write_gate", torch.zeros(1, T))[0, t]),
            "reset_gate": float(extras.get("state_reset_gate", torch.zeros(1, T))[0, t]),
        }
        if "wave_state" in extras:
            row["x_norm"] = float(extras["wave_state"][0, t].norm())
            row["v_norm"] = float(extras["wave_velocity"][0, t].norm())
        if "hidden" in extras:
            row["hidden_norm"] = float(extras["hidden"][0, t].norm())
        if "fast_memory_post" in extras:
            row["F_norm"] = float(extras["fast_memory_post"][0, t].norm())
            row["delta_F_norm"] = float(extras["fast_delta"][0, t].norm())
            if "fast_query_raw" in extras:
                row["query_raw_norm"] = float(
                    extras["fast_query_raw"][0, t].norm()
                )
            row["query_norm"] = float(extras["fast_query"][0, t].norm())
            if "fast_value_raw" in extras:
                row["value_raw_norm"] = float(
                    extras["fast_value_raw"][0, t].norm()
                )
            row["value_norm"] = float(extras["fast_value"][0, t].norm())
            row["retrieved_norm"] = float(extras["fast_retrieved"][0, t].norm())
            row["fast_drive_norm"] = float(extras["fast_drive"][0, t].norm())
        rows.append(row)

    with (out_dir / "dynamics_trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "trial": args.trial,
        "target_successor": target,
        "predicted_successor": predicted,
        "correct": int(predicted == target),
        "top1_probability": float(probs.max()),
        "fast_patch_norm": getattr(train_args, "fast_patch_norm", "none"),
        "fast_readout_bias": getattr(train_args, "fast_readout_bias", True),
    }
    if hasattr(model, "fast_to_site"):
        bias = model.fast_to_site.bias
        summary["fast_to_site_bias_norm"] = (
            0.0 if bias is None else float(bias.detach().norm().cpu())
        )

    if "fast_memory_post" in extras:
        pair_target_times = ds.pair_target_times[args.trial].tolist()
        queried_pair = int(ds.queried_pair_indices[args.trial])
        query_read_time = pred_t
        query_q = extras["fast_query"][0, query_read_time]
        query_r = extras["fast_retrieved"][0, query_read_time]

        key_sim = []
        retrieval_sim = []
        for pair_idx, target_time in enumerate(pair_target_times):
            stored_key = extras["fast_query"][0, target_time]
            stored_value = extras["fast_value"][0, target_time]
            key_sim.append(cosine(query_q, stored_key))
            retrieval_sim.append(cosine(query_r, stored_value))

        summary["queried_pair_index"] = queried_pair
        summary["query_to_correct_key_cosine"] = key_sim[queried_pair]
        summary["retrieval_to_correct_value_cosine"] = retrieval_sim[queried_pair]

        with (out_dir / "pair_similarity.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "pair_index",
                    "source_arm",
                    "target_arm",
                    "query_key_cosine",
                    "retrieval_value_cosine",
                    "is_queried_pair",
                ]
            )
            for j in range(ds.transition_n_pairs):
                writer.writerow(
                    [
                        j,
                        int(ds.pair_sources[args.trial, j]),
                        int(ds.pair_targets[args.trial, j]),
                        key_sim[j],
                        retrieval_sim[j],
                        int(j == queried_pair),
                    ]
                )

        F_site = extras["fast_memory_post"][0].norm(dim=(-1, -2)).T
        dF_site = extras["fast_delta"][0].norm(dim=(-1, -2)).T
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        im0 = axes[0].imshow(F_site, aspect="auto", origin="lower")
        axes[0].set_ylabel("Latent site")
        axes[0].set_title("Fast-memory norm")
        fig.colorbar(im0, ax=axes[0], fraction=0.025)
        im1 = axes[1].imshow(dF_site, aspect="auto", origin="lower")
        axes[1].set_ylabel("Latent site")
        axes[1].set_xlabel("Time")
        axes[1].set_title("Fast-weight update magnitude |ΔF|")
        fig.colorbar(im1, ax=axes[1], fraction=0.025)
        fig.tight_layout()
        fig.savefig(out_dir / "fast_weight_progression.png", dpi=180)
        plt.close(fig)

        labels = [
            f"{int(ds.pair_sources[args.trial, j])}→{int(ds.pair_targets[args.trial, j])}"
            for j in range(ds.transition_n_pairs)
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, key_sim)
        axes[0].set_ylim(-1, 1)
        axes[0].set_title("Query key vs encoded keys")
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].bar(labels, retrieval_sim)
        axes[1].set_ylim(-1, 1)
        axes[1].set_title("Retrieved vector vs encoded values")
        axes[1].tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / "key_and_retrieval_similarity.png", dpi=180)
        plt.close(fig)

        np.savez_compressed(
            out_dir / "transition_memory_trace.npz",
            x=extras["wave_state"].detach().cpu().numpy(),
            v=extras["wave_velocity"].detach().cpu().numpy(),
            F_pre=extras["fast_memory_pre"].numpy(),
            F_post=extras["fast_memory_post"].numpy(),
            delta_F=extras["fast_delta"].numpy(),
            query_raw=(
                extras["fast_query_raw"].numpy()
                if "fast_query_raw" in extras
                else extras["fast_query"].numpy()
            ),
            query=extras["fast_query"].numpy(),
            value_raw=(
                extras["fast_value_raw"].numpy()
                if "fast_value_raw" in extras
                else extras["fast_value"].numpy()
            ),
            value=extras["fast_value"].numpy(),
            retrieved=extras["fast_retrieved"].numpy(),
            fast_drive=extras["fast_drive"].numpy(),
            input=x.detach().cpu().numpy(),
            output=yhat.detach().cpu().numpy(),
        )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())

    print(f"Checkpoint: {args.ckpt}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved dynamics to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--out-dir", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
