#!/usr/bin/env python3
"""Verify that a forced-only FastWave checkpoint writes only in forced frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks import build_dataset
from src.train import build_model


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

    gate = extras["fast_write_gate"][0].detach().cpu()
    expected_end = int(dataset.n_forced * dataset.visit_len)
    expected = torch.zeros_like(gate)
    expected[:expected_end] = 1.0

    print(f"fast_write_phase: {getattr(train_args, 'fast_write_phase', 'all')}")
    print(f"fast_nonwrite_mode: {getattr(train_args, 'fast_nonwrite_mode', 'decay')}")
    print(f"fast_write_cue_index: {getattr(train_args, 'fast_write_cue_index', None)}")
    print(f"forced input frames: 0..{expected_end - 1}")
    print(f"gate sum: {int(gate.sum().item())} / {len(gate)}")
    print("gate:", "".join("1" if value > 0.5 else "0" for value in gate.tolist()))

    if getattr(train_args, "fast_write_phase", "all") == "forced":
        if not torch.equal(gate, expected):
            mismatch = torch.where(gate != expected)[0].tolist()
            raise RuntimeError(f"Unexpected write gate at timesteps: {mismatch}")
        print("PASS: writing is active only during the forced phase.")
    else:
        if not torch.all(gate == 1):
            raise RuntimeError("All-write checkpoint did not write at every timestep")
        print("PASS: writing is active at every timestep.")


if __name__ == "__main__":
    main()
