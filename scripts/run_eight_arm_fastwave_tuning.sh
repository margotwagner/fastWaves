#!/usr/bin/env bash
set -euo pipefail

# Tune FastWave fast-weight strength against a matched Wave reference.
#
# Place this file in scripts/ and run it from the repository root:
#   chmod +x scripts/run_eight_arm_fastwave_tuning.sh
#   DEVICE=cuda ./scripts/run_eight_arm_fastwave_tuning.sh
#
# Optional overrides:
#   EPOCHS=100 N_TRAIN=1024 N_VAL=256 N_TEST=256 BATCH_SIZE=64
#   SETTLE_STEPS=5 CHOICE_WEIGHT=0.1 DEBUG_TRIALS=16 SKIP_EXISTING=1
#
# This script runs:
#   Wave reference: 3 seeds
#   FastWave: 4 (eta, beta) settings x 3 seeds
# Total: 15 runs

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
N_TRAIN="${N_TRAIN:-1024}"
N_VAL="${N_VAL:-256}"
N_TEST="${N_TEST:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SETTLE_STEPS="${SETTLE_STEPS:-5}"
CHOICE_WEIGHT="${CHOICE_WEIGHT:-0.1}"
DEBUG_TRIALS="${DEBUG_TRIALS:-16}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

ACTION_HOLD_STEPS=1
WAVE_READOUT="xv"
N_ARMS=8
ARM_LEN=3
REWARD_HOLD_STEPS=1
LAM=0.95

# visit_len = center(no action) + action holds + 3 outbound/reward
#             + 2 inbound + final center = 7 + action_hold_steps
VISIT_LEN=$((7 + ACTION_HOLD_STEPS))
SEQ_LEN=$((N_ARMS * VISIT_LEN + SETTLE_STEPS - 1))

SEEDS=(42 43 44)

# Candidate settings selected to reduce interference with wave routing while
# preserving useful transient memory.
FASTWAVE_CONFIGS=(
  "0.05 0.25"
  "0.05 0.50"
  "0.10 0.25"
  "0.10 0.50"
)

float_tag () {
  local value="$1"
  echo "${value/./p}"
}

train_and_analyze () {
  local model="$1"
  local eta="$2"
  local beta="$3"
  local seed="$4"

  local eta_tag
  local beta_tag
  eta_tag="$(float_tag "${eta}")"
  beta_tag="$(float_tag "${beta}")"

  local run_name
  if [[ "${model}" == "wave" ]]; then
    run_name="fwtune_wave_commitment_xv_h1_s${SETTLE_STEPS}_seed${seed}"
  else
    run_name="fwtune_fastwave_eta${eta_tag}_beta${beta_tag}_commitment_xv_h1_s${SETTLE_STEPS}_seed${seed}"
  fi

  local ckpt="data/runs/${run_name}/best.pt"
  local metrics="data/runs/${run_name}/analysis/metrics.csv"

  echo
  echo "============================================================"
  echo "${run_name}"
  echo "============================================================"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${ckpt}" ]]; then
    echo "Checkpoint exists; skipping training."
  else
    python -m src.train \
      --model "${model}" \
      --task eight_arm_bump_traj \
      --choice-order random \
      --choice-objective commitment \
      --valid-choice-loss-weight "${CHOICE_WEIGHT}" \
      --wave-readout "${WAVE_READOUT}" \
      --action-hold-steps "${ACTION_HOLD_STEPS}" \
      --n-space 40 \
      --seq-len "${SEQ_LEN}" \
      --n-arms "${N_ARMS}" \
      --arm-len "${ARM_LEN}" \
      --reward-hold-steps "${REWARD_HOLD_STEPS}" \
      --settle-steps "${SETTLE_STEPS}" \
      --n-train "${N_TRAIN}" \
      --n-val "${N_VAL}" \
      --batch-size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --routing-weight 50 \
      --arm-choice-weight 50 \
      --fast-update transition \
      --lam "${LAM}" \
      --eta "${eta}" \
      --beta "${beta}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --run-name "${run_name}"
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${metrics}" ]]; then
    echo "Analysis exists; skipping analysis."
  else
    python -m src.analyze \
      --ckpt "${ckpt}" \
      --device "${DEVICE}" \
      --n-test "${N_TEST}" \
      --batch-size "${BATCH_SIZE}" \
      --prefix-len -1 \
      --debug-trials "${DEBUG_TRIALS}"
  fi
}

# Matched Wave reference.
for seed in "${SEEDS[@]}"; do
  train_and_analyze "wave" "0.10" "1.00" "${seed}"
done

# FastWave candidates.
for config in "${FASTWAVE_CONFIGS[@]}"; do
  read -r eta beta <<< "${config}"
  for seed in "${SEEDS[@]}"; do
    train_and_analyze "fastwave" "${eta}" "${beta}" "${seed}"
  done
done

shopt -s nullglob
metric_files=(
  data/runs/fwtune_*_commitment_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
)

if ((${#metric_files[@]} == 0)); then
  echo "No FastWave-tuning metric files found." >&2
  exit 1
fi

python -m src.compare_metrics \
  "${metric_files[@]}" \
  --mode eight_arm \
  --sort \
  --out data/runs/fastwave_tuning_run_level.csv

python scripts/summarize_fastwave_tuning.py \
  --input data/runs/fastwave_tuning_run_level.csv \
  --output data/runs/fastwave_tuning_summary.csv

echo
echo "Saved:"
echo "  data/runs/fastwave_tuning_run_level.csv"
echo "  data/runs/fastwave_tuning_summary.csv"
