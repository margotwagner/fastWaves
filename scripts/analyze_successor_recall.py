#!/usr/bin/env python3
"""Evaluate a trained eight-arm episodic successor-recall checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.tasks import build_dataset
from src.train import build_model, unpack_batch, successor_recall_loss_and_accuracy


def safe_torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = safe_torch_load(args.ckpt, device)
    train_args = SimpleNamespace(**ckpt["args"])
    train_args.device = str(device)

    if getattr(train_args, "task", None) != "eight_arm_successor":
        raise ValueError("Checkpoint was not trained on eight_arm_successor")

    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = build_dataset(train_args, n_samples=args.n_test, seed=args.seed)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_pred = []
    all_target = []
    all_prob = []
    total_loss = 0.0
    n_seen = 0

    with torch.no_grad():
        for batch in loader:
            unpacked = unpack_batch(batch, device)
            yhat, _ = model(unpacked["x"])
            loss, _ = successor_recall_loss_and_accuracy(
                yhat=yhat,
                successor_targets=unpacked["successor_targets"],
                successor_query_masks=unpacked["successor_query_masks"],
                dataset=test_ds,
            )
            logits = yhat[
                ..., test_ds.arm_choice_start : test_ds.arm_choice_end
            ][unpacked["successor_query_masks"].bool()]
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            target = unpacked["successor_targets"].long()

            batch_n = target.shape[0]
            total_loss += loss.item() * batch_n
            n_seen += batch_n
            all_pred.append(pred.cpu())
            all_target.append(target.cpu())
            all_prob.append(probs.cpu())

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    probs = torch.cat(all_prob)
    accuracy = pred.eq(target).float().mean().item()
    mean_confidence = probs.max(dim=-1).values.mean().item()
    cross_entropy = total_loss / max(n_seen, 1)

    out_dir = Path(args.out_dir)
    if args.out_dir == "auto":
        out_dir = Path(args.ckpt).parent / "successor_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "successor_test_accuracy": accuracy,
        "successor_test_cross_entropy": cross_entropy,
        "successor_test_mean_top1_probability": mean_confidence,
        "chance_accuracy": 1.0 / test_ds.n_arms,
        "n_test": float(args.n_test),
    }
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    confusion = torch.zeros(test_ds.n_arms, test_ds.n_arms, dtype=torch.long)
    for t, p in zip(target.tolist(), pred.tolist()):
        confusion[t, p] += 1
    np.savetxt(out_dir / "confusion_matrix.csv", confusion.numpy(), fmt="%d", delimiter=",")

    rows = []
    for i in range(args.n_test):
        sequence = test_ds.sequences[i].tolist()
        rows.append(
            {
                "trial": i,
                "sequence": " ".join(map(str, sequence)),
                "query_index": int(test_ds.query_indices[i].item()),
                "query_arm": int(test_ds.query_arms[i].item()),
                "target_successor": int(target[i].item()),
                "predicted_successor": int(pred[i].item()),
                "correct": int(pred[i].item() == target[i].item()),
                "top1_probability": float(probs[i].max().item()),
            }
        )
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion.numpy())
    ax.set_xlabel("Predicted successor arm")
    ax.set_ylabel("True successor arm")
    ax.set_xticks(range(test_ds.n_arms))
    ax.set_yticks(range(test_ds.n_arms))
    ax.set_title(f"Successor recall confusion matrix\naccuracy = {accuracy:.3f}")
    fig.colorbar(image, ax=ax, label="Trial count")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(f"Checkpoint: {args.ckpt}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"Saved analysis to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-dir", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
