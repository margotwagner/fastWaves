#!/usr/bin/env python3
"""Evaluate whether transition recall causally depends on FastWave memory."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.tasks import build_dataset
from src.train import build_model


def safe_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate(model, ds, device, batch_size: int, mode: str) -> dict[str, float]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    confidence_sum = 0.0
    entropy_sum = 0.0

    model.eval()
    with torch.no_grad():
        offset = 0
        for batch in loader:
            x = batch[0].to(device)
            targets = batch[3].to(device)
            query_mask = batch[4].to(device)
            yhat, _ = model(x, fast_ablation=mode)

            b_idx, t_idx = torch.where(query_mask)
            logits = yhat[b_idx, t_idx, ds.arm_choice_start : ds.arm_choice_end]
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)

            correct += int(pred.eq(targets).sum().item())
            total += int(targets.numel())
            confidence_sum += float(probs.max(dim=-1).values.sum().item())
            entropy_sum += float(
                (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).sum().item()
            )
            offset += x.shape[0]

    return {
        "ablation": mode,
        "successor_accuracy": correct / max(total, 1),
        "mean_top1_probability": confidence_sum / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "n_examples": float(total),
    }


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = safe_load(args.ckpt, device)
    train_args = SimpleNamespace(**ckpt["args"])
    if getattr(train_args, "model", None) != "fastwave":
        raise ValueError("This causal ablation analysis requires a FastWave checkpoint")
    if not getattr(train_args, "transition_reset_before_query", False):
        raise ValueError(
            "The checkpoint must use --transition-reset-before-query so the "
            "ablation has a well-defined query-onset reset event."
        )

    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model_state"])
    ds = build_dataset(train_args, n_samples=args.n_test, seed=args.seed)

    modes = [
        "none",
        "erase_at_reset",
        "shuffle_at_reset",
        "disable_drive_after_reset",
    ]
    rows = [evaluate(model, ds, device, args.batch_size, mode) for mode in modes]
    baseline = rows[0]["successor_accuracy"]
    for row in rows:
        row["accuracy_change_from_baseline"] = row["successor_accuracy"] - baseline

    out_dir = (
        Path(args.ckpt).parent / "transition_ablations"
        if args.out_dir == "auto"
        else Path(args.out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Checkpoint: {args.ckpt}")
    for row in rows:
        print(
            f"{row['ablation']:>26s} | accuracy "
            f"{row['successor_accuracy']:.4f} | delta "
            f"{row['accuracy_change_from_baseline']:+.4f}"
        )
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=54321)
    p.add_argument("--out-dir", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
