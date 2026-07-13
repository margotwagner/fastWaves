#!/usr/bin/env bash
set -euo pipefail

# Diagnostic sweep for the three changes:
#   1) commitment choice loss,
#   2) Wave/FastWave readout from [x, v],
#   3) multiple action-cued center preparation steps.
#
# Place this file in scripts/ and run from the repository root:
#   DEVICE=cuda ./scripts/run_eight_arm_commitment_diagnostic.sh
#
# Optional overrides:
#   EPOCHS=100 N_TRAIN=1024 N_VAL=256 N_TEST=128 BATCH_SIZE=64
#   SEED=42 SETTLE_STEPS=5 CHOICE_WEIGHT=0.1

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
N_TRAIN="${N_TRAIN:-1024}"
N_VAL="${N_VAL:-256}"
N_TEST="${N_TEST:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
SETTLE_STEPS="${SETTLE_STEPS:-5}"
CHOICE_WEIGHT="${CHOICE_WEIGHT:-0.1}"
DEBUG_TRIALS="${DEBUG_TRIALS:-16}"

run_one () {
  local model="$1"
  local label="$2"
  local action_holds="$3"
  local wave_readout="$4"
  local seq_len="$5"
  local eta="$6"
  local beta="$7"

  local run_name="commitdiag_${model}_${label}_s${SETTLE_STEPS}_seed${SEED}"

  echo
  echo "============================================================"
  echo "Training ${run_name}"
  echo "============================================================"

  python -m src.train \
    --model "${model}" \
    --task eight_arm_bump_traj \
    --choice-order random \
    --choice-objective commitment \
    --valid-choice-loss-weight "${CHOICE_WEIGHT}" \
    --wave-readout "${wave_readout}" \
    --action-hold-steps "${action_holds}" \
    --n-space 40 \
    --seq-len "${seq_len}" \
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
}

# With settle_steps=5:
#   action_holds=1 -> natural seq_len=68
#   action_holds=3 -> natural seq_len=84
#
# A: commitment loss only, original x-only readout, one action hold.
# B: add [x, v] readout, still one action hold.
# C: add three action preparation frames.
for model in wave fastwave; do
  run_one "${model}" "commit_x_h1"  1 x  68 0.1 1.0
  run_one "${model}" "commit_xv_h1" 1 xv 68 0.1 1.0
  run_one "${model}" "commit_xv_h3" 3 xv 84 0.1 1.0
done

# Stabilized GlobalFast control using the full h3 task.
run_one globalfast "commit_h3" 3 x 84 0.01 0.25

shopt -s nullglob
metric_files=(
  data/runs/commitdiag_*_s"${SETTLE_STEPS}"_seed"${SEED}"/analysis/metrics.csv
)

if ((${#metric_files[@]} == 0)); then
  echo "No diagnostic metric files found." >&2
  exit 1
fi

python -m src.compare_metrics \
  "${metric_files[@]}" \
  --mode eight_arm \
  --sort \
  --out data/runs/commitment_diagnostic_comparison.csv

echo
echo "Saved comparison:"
echo "  data/runs/commitment_diagnostic_comparison.csv"
