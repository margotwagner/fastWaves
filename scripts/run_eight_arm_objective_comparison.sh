#!/usr/bin/env bash
set -euo pipefail

# Compare valid-set vs commitment objectives using the corrected Wave/FastWave
# configuration: full [x,v] readout and one action-hold step.
#
# Place in scripts/ and run from the repository root:
#   DEVICE=cuda ./scripts/run_eight_arm_objective_comparison.sh
#
# Optional overrides:
#   EPOCHS=100 N_TRAIN=1024 N_VAL=256 N_TEST=256 BATCH_SIZE=64
#   SETTLE_STEPS=5 CHOICE_WEIGHT=0.1 DEBUG_TRIALS=16 SKIP_EXISTING=1

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

# For arm_len=3 and reward_hold_steps=1:
# visit_len = center(no action) + action holds + 3 outbound/reward
#             + 2 inbound + final center = 7 + action_hold_steps.
VISIT_LEN=$((7 + ACTION_HOLD_STEPS))
SEQ_LEN=$((N_ARMS * VISIT_LEN + SETTLE_STEPS - 1))

MODELS=(wave fastwave)
OBJECTIVES=(valid_set commitment)
SEEDS=(42 43 44)

run_one () {
  local model="$1"
  local objective="$2"
  local seed="$3"
  local run_name="objectivecmp_${model}_${objective}_xv_h1_s${SETTLE_STEPS}_seed${seed}"
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
      --choice-objective "${objective}" \
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
      --lam 0.95 \
      --eta 0.1 \
      --beta 1.0 \
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

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for objective in "${OBJECTIVES[@]}"; do
      run_one "${model}" "${objective}" "${seed}"
    done
  done
done

shopt -s nullglob
metric_files=(
  data/runs/objectivecmp_*_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
)

if ((${#metric_files[@]} == 0)); then
  echo "No objective-comparison metric files found." >&2
  exit 1
fi

python -m src.compare_metrics \
  "${metric_files[@]}" \
  --mode eight_arm \
  --sort \
  --out data/runs/objective_comparison_run_level.csv

python scripts/summarize_eight_arm_objectives.py \
  --input data/runs/objective_comparison_run_level.csv \
  --output data/runs/objective_comparison_summary.csv

echo
echo "Saved:"
echo "  data/runs/objective_comparison_run_level.csv"
echo "  data/runs/objective_comparison_summary.csv"
