# Forced + reward-event FastWave writing

This update adds:

```text
--fast-write-phase forced_reward
--fast-nonwrite-mode hold
```

The resulting schedule is:

```text
forced visits:       write
settle:              hold
choice traversal:    hold
choice reward frame: write once
return to center:    hold
```

The reward gate uses the existing `cue_reward` input channel. No change to
`tasks.py` is needed.

## Files to install

```bash
cp eight_arm_forced_reward_update/src/models.py src/models.py
cp eight_arm_forced_reward_update/src/train.py src/train.py
cp eight_arm_forced_reward_update/src/models_diagnostics.py src/models_diagnostics.py
cp eight_arm_forced_reward_update/scripts/analyze_fastwave_states.py scripts/analyze_fastwave_states.py
cp eight_arm_forced_reward_update/scripts/check_fast_write_gate.py scripts/check_fast_write_gate.py
cp eight_arm_forced_reward_update/scripts/run_eight_arm_forced_reward_sweep.sh scripts/run_eight_arm_forced_reward_sweep.sh
cp eight_arm_forced_reward_update/scripts/summarize_write_schedules.py scripts/summarize_write_schedules.py
chmod +x scripts/check_fast_write_gate.py scripts/run_eight_arm_forced_reward_sweep.sh scripts/summarize_write_schedules.py
```

## Smoke test

```bash
python -m src.train \
  --model fastwave \
  --task eight_arm_bump_traj \
  --choice-order random \
  --choice-objective commitment \
  --valid-choice-loss-weight 0.1 \
  --wave-readout xv \
  --action-hold-steps 1 \
  --n-space 40 \
  --seq-len 68 \
  --n-arms 8 \
  --arm-len 3 \
  --reward-hold-steps 1 \
  --settle-steps 5 \
  --n-train 32 \
  --n-val 16 \
  --batch-size 16 \
  --epochs 1 \
  --routing-weight 50 \
  --arm-choice-weight 50 \
  --fast-update transition \
  --fast-write-phase forced_reward \
  --fast-nonwrite-mode hold \
  --lam 0.95 \
  --eta 0.10 \
  --beta 0.25 \
  --seed 42 \
  --device cpu \
  --run-name forced_reward_smoke
```

```bash
python scripts/check_fast_write_gate.py \
  --ckpt data/runs/forced_reward_smoke/best.pt \
  --device cpu
```

The checker should report four choice-phase reward write events.

## Three-seed experiment

```bash
DEVICE=cuda \
EPOCHS=100 \
N_TRAIN=2048 \
N_VAL=256 \
N_TEST=256 \
BATCH_SIZE=64 \
./scripts/run_eight_arm_forced_reward_sweep.sh
```

Restart while retaining completed runs:

```bash
DEVICE=cpu SKIP_EXISTING=1 \
./scripts/run_eight_arm_forced_reward_sweep.sh
```

Outputs:

```text
data/runs/write_schedule_run_level.csv
data/runs/write_schedule_summary.csv
```
