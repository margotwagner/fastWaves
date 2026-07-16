#!/usr/bin/env python3
"""Sanity checks for the two-arm and independent transition-pair tasks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.tasks import EightArmTransitionRecallDataset


def check_dataset(ds: EightArmTransitionRecallDataset, name: str) -> None:
    x, y, weights, target, query_mask = ds[0]
    write_times = torch.where(x[:, ds.cue_write] > 0.5)[0]
    reset_times = torch.where(x[:, ds.cue_reset] > 0.5)[0]
    scored_times = torch.where(query_mask)[0]

    assert len(write_times) == ds.transition_n_pairs
    assert torch.equal(write_times, ds.pair_target_times[0])
    assert len(scored_times) == 1
    assert int(scored_times[0]) == ds.successor_prediction_time
    assert int(y[scored_times[0], ds.arm_choice_start : ds.arm_choice_end].argmax()) == int(target)
    assert float(weights.sum()) == float(ds.n_arms)

    expected_resets = 0
    if ds.transition_reset_between_pairs:
        expected_resets += ds.transition_n_pairs - 1
    if ds.transition_reset_before_query:
        expected_resets += 1
    assert len(reset_times) == expected_resets

    print(f"{name}: PASS")
    print(f"  sources: {ds.pair_sources[0].tolist()}")
    print(f"  targets: {ds.pair_targets[0].tolist()}")
    print(f"  source times: {ds.pair_source_times[0].tolist()}")
    print(f"  write/target times: {write_times.tolist()}")
    print(f"  reset times: {reset_times.tolist()}")
    print(f"  query arm: {int(ds.query_arms[0])}")
    print(f"  target successor: {int(target)}")
    print(f"  prediction time: {ds.successor_prediction_time}\n")


def main() -> None:
    two_arm = EightArmTransitionRecallDataset(
        n_samples=4,
        n_space=40,
        transition_n_pairs=1,
        transition_delay_steps=0,
        transition_query_hold_steps=2,
        transition_reset_between_pairs=False,
        transition_reset_before_query=False,
        seed=42,
    )
    independent = EightArmTransitionRecallDataset(
        n_samples=4,
        n_space=40,
        transition_n_pairs=3,
        transition_delay_steps=0,
        transition_query_hold_steps=2,
        transition_reset_between_pairs=True,
        transition_reset_before_query=True,
        seed=42,
    )
    check_dataset(two_arm, "two-arm sanity")
    check_dataset(independent, "independent transition pairs")
    print("PASS: all transition-task timing and targets are consistent")


if __name__ == "__main__":
    main()
