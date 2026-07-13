#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
N_TRAIN="${N_TRAIN:-2048}"
N_VAL="${N_VAL:-256}"

models=(vanilla globalfast localfast wave fastwave)
settles=(0 2 5 10)
seeds=(42 43 44)

for settle in "${settles[@]}"; do
  seq_len=$((63 + settle))
  for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
      run="validset_${model}_s${settle}_seed${seed}"
      python -m src.train \
        --model "$model" \
        --task eight_arm_bump_traj \
        --choice-order random \
        --choice-objective valid_set \
        --valid-choice-loss-weight 1.0 \
        --n-space 40 \
        --seq-len "$seq_len" \
        --settle-steps "$settle" \
        --n-train "$N_TRAIN" \
        --n-val "$N_VAL" \
        --batch-size 64 \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --routing-weight 50 \
        --arm-choice-weight 50 \
        --fast-update transition \
        --lam 0.95 \
        --eta 0.1 \
        --beta 1.0 \
        --seed "$seed" \
        --run-name "$run" \
        --print-every 5
    done
  done
done

for ckpt in data/runs/validset_*_s*_seed*/best.pt; do
  python -m src.analyze \
    --ckpt "$ckpt" \
    --device "$DEVICE" \
    --n-test 64 \
    --batch-size 64 \
    --prefix-len -1 \
    --debug-trials 8
done

python -m src.compare_metrics \
  data/runs/validset_*_s*_seed*/analysis/metrics.csv \
  --mode eight_arm \
  --sort \
  --out data/runs/validset_main_sweep_comparison.csv
