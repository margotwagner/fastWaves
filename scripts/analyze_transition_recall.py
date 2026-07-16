#!/usr/bin/env python3
"""Evaluate an eight-arm transition-recall checkpoint."""

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
from torch.utils.data import DataLoader

from src.tasks import build_dataset
from src.train import build_model, unpack_batch, successor_recall_loss_and_accuracy


def safe_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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

    ds = build_dataset(train_args, n_samples=args.n_test, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    preds, targets, probs_all = [], [], []
    total_loss = 0.0
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            unpacked = unpack_batch(batch, device)
            yhat, _ = model(unpacked["x"])
            loss, _ = successor_recall_loss_and_accuracy(
                yhat,
                unpacked["successor_targets"],
                unpacked["successor_query_masks"],
                ds,
            )
            logits = yhat[..., ds.arm_choice_start : ds.arm_choice_end][
                unpacked["successor_query_masks"].bool()
            ]
            probs = torch.softmax(logits, dim=-1)
            target = unpacked["successor_targets"].long()
            pred = probs.argmax(dim=-1)

            n = target.shape[0]
            total_loss += loss.item() * n
            n_seen += n
            preds.append(pred.cpu())
            targets.append(target.cpu())
            probs_all.append(probs.cpu())

    pred = torch.cat(preds)
    target = torch.cat(targets)
    probs = torch.cat(probs_all)
    correct = pred.eq(target)

    out_dir = Path(args.out_dir)
    if args.out_dir == "auto":
        out_dir = Path(args.ckpt).parent / "transition_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "transition_test_accuracy": correct.float().mean().item(),
        "transition_test_cross_entropy": total_loss / max(n_seen, 1),
        "transition_test_mean_top1_probability": probs.max(-1).values.mean().item(),
        "chance_accuracy": 1.0 / ds.n_arms,
        "n_test": float(args.n_test),
    }
    for pair_idx in range(ds.transition_n_pairs):
        mask = ds.queried_pair_indices == pair_idx
        metrics[f"accuracy_query_pair_{pair_idx}"] = (
            correct[mask].float().mean().item() if mask.any() else float("nan")
        )

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    confusion = torch.zeros(ds.n_arms, ds.n_arms, dtype=torch.long)
    for t, p in zip(target.tolist(), pred.tolist()):
        confusion[t, p] += 1
    np.savetxt(out_dir / "confusion_matrix.csv", confusion.numpy(), fmt="%d", delimiter=",")

    rows = []
    for i in range(args.n_test):
        rows.append(
            {
                "trial": i,
                "pair_sources": " ".join(map(str, ds.pair_sources[i].tolist())),
                "pair_targets": " ".join(map(str, ds.pair_targets[i].tolist())),
                "queried_pair_index": int(ds.queried_pair_indices[i]),
                "query_arm": int(ds.query_arms[i]),
                "target_successor": int(target[i]),
                "predicted_successor": int(pred[i]),
                "correct": int(correct[i]),
                "top1_probability": float(probs[i].max()),
            }
        )
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion.numpy())
    ax.set_xlabel("Predicted successor arm")
    ax.set_ylabel("True successor arm")
    ax.set_xticks(range(ds.n_arms))
    ax.set_yticks(range(ds.n_arms))
    ax.set_title(f"Transition recall\naccuracy = {metrics['transition_test_accuracy']:.3f}")
    fig.colorbar(im, ax=ax, label="Trial count")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(f"Checkpoint: {args.ckpt}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"Saved analysis to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out-dir", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
