#!/usr/bin/env python3
"""Verify FastWave write gates against the task's phase/reward cues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks import build_dataset  # noqa: E402
from src.train import build_model  # noqa: E402


def safe_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = safe_load(args.ckpt, device)
    train_args = SimpleNamespace(**ckpt["args"])
    train_args.device = str(device)

    dataset = build_dataset(train_args, n_samples=1, seed=123)
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        _, extras = model(dataset.x[:1].to(device))

    if "fast_write_gate" not in extras:
        raise RuntimeError("Checkpoint/model did not return fast_write_gate")

    gate = extras["fast_write_gate"][0].detach().cpu() > 0.5
    x = dataset.x[0]
    phase = getattr(train_args, "fast_write_phase", "all")

    if phase == "all":
        expected = torch.ones_like(gate)
    elif phase == "forced":
        expected = x[:, dataset.cue_forced] > 0.5
    elif phase == "forced_reward":
        expected = (x[:, dataset.cue_forced] > 0.5) | (
            x[:, dataset.cue_reward] > 0.5
        )
    else:
        raise ValueError(f"Unknown fast_write_phase: {phase}")

    print(f"fast_write_phase: {phase}")
    print(f"fast_nonwrite_mode: {getattr(train_args, 'fast_nonwrite_mode', 'decay')}")
    print(f"forced cue index: {getattr(train_args, 'fast_write_cue_index', None)}")
    print(
        "reward cue index: "
        f"{getattr(train_args, 'fast_write_reward_cue_index', None)}"
    )
    print(f"gate sum: {int(gate.sum().item())} / {len(gate)}")
    print("gate:    ", "".join("1" if z else "0" for z in gate.tolist()))
    print("expected:", "".join("1" if z else "0" for z in expected.tolist()))

    if not torch.equal(gate, expected):
        mismatch = torch.where(gate != expected)[0].tolist()
        raise RuntimeError(f"Unexpected write gate at timesteps: {mismatch}")

    reward_only = (
        (x[:, dataset.cue_reward] > 0.5)
        & ~(x[:, dataset.cue_forced] > 0.5)
    )
    print(f"choice reward write events: {int(reward_only.sum().item())}")
    print("PASS: the write gate matches the configured schedule.")


if __name__ == "__main__":
    main()
