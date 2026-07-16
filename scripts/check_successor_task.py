#!/usr/bin/env python3
"""Sanity-check the successor-recall timing and targets."""

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from types import SimpleNamespace

from src.tasks import build_dataset


args = SimpleNamespace(
    task="eight_arm_successor",
    seq_len=11,
    n_space=40,
    n_arms=8,
    arm_len=3,
    successor_seq_length=4,
    successor_delay_steps=5,
    successor_query_hold_steps=2,
    bump_sigma=0.75,
)

ds = build_dataset(args, n_samples=8, seed=42)
for i in range(8):
    sequence = ds.sequences[i].tolist()
    query_index = int(ds.query_indices[i])
    query = int(ds.query_arms[i])
    target = int(ds.successor_targets[i])
    assert sequence[query_index] == query
    assert sequence[query_index + 1] == target
    assert int(ds.successor_query_masks[i].sum()) == 1
    print(f"trial {i}: {sequence} | query {query} -> target {target}")

print(f"prediction timestep: {ds.successor_prediction_time}")
print("PASS: successor targets and timing are consistent")
