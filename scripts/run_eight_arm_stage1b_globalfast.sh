#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   DEVICE=cuda ./scripts/run_eight_arm_stage1b_globalfast.sh
#
# Optional overrides:
#   EPOCHS=100 N_TRAIN=1024 N_VAL=256 BATCH_SIZE=64 SEED=42 SETTLE_STEPS=5

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
N_TRAIN="${N_TRAIN:-1024}"
N_VAL="${N_VAL:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
SETTLE_STEPS="${SETTLE_STEPS:-5}"
N_TEST="${N_TEST:-128}"
DEBUG_TRIALS="${DEBUG_TRIALS:-16}"
VALID_CHOICE_WEIGHT="${VALID_CHOICE_WEIGHT:-0.1}"

ETAS=(0.005 0.01)
BETAS=(0.1 0.25)

for eta in "${ETAS[@]}"; do
  eta_tag="${eta/./p}"

  for beta in "${BETAS[@]}"; do
    beta_tag="${beta/./p}"
    run_name="stage1b_globalfast_eta${eta_tag}_beta${beta_tag}_s${SETTLE_STEPS}_seed${SEED}"

    echo
    echo "============================================================"
    echo "Training ${run_name}"
    echo "============================================================"

    python -m src.train \
      --model globalfast \
      --task eight_arm_bump_traj \
      --choice-order random \
      --choice-objective valid_set \
      --valid-choice-loss-weight "${VALID_CHOICE_WEIGHT}" \
      --n-space 40 \
      --seq-len 68 \
      --settle-steps "${SETTLE_STEPS}" \
      --n-train "${N_TRAIN}" \
      --n-val "${N_VAL}" \
      --batch-size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --routing-weight 50 \
      --arm-choice-weight 50 \
      --fast-update transition \
      --lam 0.95 \
      --eta "${eta}" \
      --beta "${beta}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --run-name "${run_name}"

    echo
    echo "Analyzing ${run_name}"

    python -m src.analyze \
      --ckpt "data/runs/${run_name}/best.pt" \
      --device "${DEVICE}" \
      --n-test "${N_TEST}" \
      --batch-size "${BATCH_SIZE}" \
      --prefix-len -1 \
      --debug-trials "${DEBUG_TRIALS}"
  done
done

shopt -s nullglob
metric_files=(data/runs/stage1b_globalfast_eta*_beta*_s"${SETTLE_STEPS}"_seed"${SEED}"/analysis/metrics.csv)

if ((${#metric_files[@]} == 0)); then
  echo "No GlobalFast Stage 1B metric files found." >&2
  exit 1
fi

python -m src.compare_metrics \
  "${metric_files[@]}" \
  --mode eight_arm \
  --sort \
  --out data/runs/stage1b_globalfast_comparison.csv

echo
echo "Saved comparison:"
echo "  data/runs/stage1b_globalfast_comparison.csv"
