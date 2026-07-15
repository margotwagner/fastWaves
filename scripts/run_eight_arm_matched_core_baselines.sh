#!/usr/bin/env bash
set -euo pipefail

# Run the five existing core models under one matched task configuration.
#
# Run this only after inspecting fastwave_tuning_summary.csv. Supply the selected
# FastWave parameters as environment variables:
#
#   FASTWAVE_ETA=0.05 FASTWAVE_BETA=0.25 DEVICE=cuda \
#     ./scripts/run_eight_arm_matched_core_baselines.sh
#
# Optional overrides:
#   EPOCHS=100 N_TRAIN=2048 N_VAL=256 N_TEST=256 BATCH_SIZE=64
#   SETTLE_STEPS=5 CHOICE_WEIGHT=0.1 DEBUG_TRIALS=16 SKIP_EXISTING=1
#   GLOBALFAST_ETA=0.01 GLOBALFAST_BETA=0.25
#   LOCALFAST_ETA=0.1 LOCALFAST_BETA=1.0

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
N_TRAIN="${N_TRAIN:-2048}"
N_VAL="${N_VAL:-256}"
N_TEST="${N_TEST:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SETTLE_STEPS="${SETTLE_STEPS:-5}"
CHOICE_WEIGHT="${CHOICE_WEIGHT:-0.1}"
DEBUG_TRIALS="${DEBUG_TRIALS:-16}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

FASTWAVE_ETA="${FASTWAVE_ETA:-0.05}"
FASTWAVE_BETA="${FASTWAVE_BETA:-0.25}"
GLOBALFAST_ETA="${GLOBALFAST_ETA:-0.01}"
GLOBALFAST_BETA="${GLOBALFAST_BETA:-0.25}"
LOCALFAST_ETA="${LOCALFAST_ETA:-0.10}"
LOCALFAST_BETA="${LOCALFAST_BETA:-1.00}"

ACTION_HOLD_STEPS=1
WAVE_READOUT="xv"
N_ARMS=8
ARM_LEN=3
REWARD_HOLD_STEPS=1
LAM=0.95
VISIT_LEN=$((7 + ACTION_HOLD_STEPS))
SEQ_LEN=$((N_ARMS * VISIT_LEN + SETTLE_STEPS - 1))

MODELS=(vanilla globalfast localfast wave fastwave)
SEEDS=(42 43 44)

float_tag () {
  local value="$1"
  echo "${value/./p}"
}

params_for_model () {
  local model="$1"
  case "${model}" in
    globalfast)
      echo "${GLOBALFAST_ETA} ${GLOBALFAST_BETA}"
      ;;
    localfast)
      echo "${LOCALFAST_ETA} ${LOCALFAST_BETA}"
      ;;
    fastwave)
      echo "${FASTWAVE_ETA} ${FASTWAVE_BETA}"
      ;;
    *)
      # Ignored by Vanilla and Wave, but supplied so the command is uniform.
      echo "0.10 1.00"
      ;;
  esac
}

for model in "${MODELS[@]}"; do
  read -r eta beta <<< "$(params_for_model "${model}")"
  eta_tag="$(float_tag "${eta}")"
  beta_tag="$(float_tag "${beta}")"

  for seed in "${SEEDS[@]}"; do
    run_name="corecmp_${model}_eta${eta_tag}_beta${beta_tag}_commitment_xv_h1_s${SETTLE_STEPS}_seed${seed}"
    ckpt="data/runs/${run_name}/best.pt"
    metrics="data/runs/${run_name}/analysis/metrics.csv"

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
  done
done

shopt -s nullglob
metric_files=(
  data/runs/corecmp_*_commitment_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
)

if ((${#metric_files[@]} == 0)); then
  echo "No matched-baseline metric files found." >&2
  exit 1
fi

python -m src.compare_metrics \
  "${metric_files[@]}" \
  --mode eight_arm \
  --sort \
  --out data/runs/core_baselines_run_level.csv

python scripts/summarize_core_baselines.py \
  --input data/runs/core_baselines_run_level.csv \
  --output data/runs/core_baselines_summary.csv

echo
echo "Saved:"
echo "  data/runs/core_baselines_run_level.csv"
echo "  data/runs/core_baselines_summary.csv"
