#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"

# Shared task settings: settle=5 -> natural seq_len=63+5=68.
COMMON=(
  --task eight_arm_bump_traj
  --n-space 40
  --seq-len 68
  --settle-steps 5
  --n-train 1024
  --n-val 256
  --batch-size 64
  --epochs "$EPOCHS"
  --device "$DEVICE"
  --routing-weight 50
  --arm-choice-weight 50
  --valid-choice-loss-weight 1.0
  --fast-update transition
  --lam 0.95
  --eta 0.1
  --beta 1.0
  --print-every 5
)

# 1) Deterministic sanity check: exact ascending target.
for model in vanilla fastwave; do
  python -m src.train \
    --model "$model" \
    --choice-order ascending \
    --choice-objective exact \
    --seed 42 \
    --run-name "debug_exact_${model}_s5_seed42" \
    "${COMMON[@]}"
done

# 2) Main any-unvisited-arm objective, one seed, all five models.
for model in vanilla globalfast localfast wave fastwave; do
  python -m src.train \
    --model "$model" \
    --choice-order random \
    --choice-objective valid_set \
    --seed 42 \
    --run-name "validset_${model}_s5_seed42" \
    "${COMMON[@]}"
done

# Analyze every stage-1 run. --prefix-len -1 infers the first choice point.
for ckpt in data/runs/debug_exact_*_s5_seed42/best.pt \
            data/runs/validset_*_s5_seed42/best.pt; do
  python -m src.analyze \
    --ckpt "$ckpt" \
    --device "$DEVICE" \
    --n-test 64 \
    --batch-size 64 \
    --prefix-len -1 \
    --debug-trials 16
done

python -m src.compare_metrics \
  data/runs/debug_exact_*_s5_seed42/analysis/metrics.csv \
  data/runs/validset_*_s5_seed42/analysis/metrics.csv \
  --mode eight_arm \
  --sort \
  --out data/runs/stage1_choice_objective_comparison.csv
