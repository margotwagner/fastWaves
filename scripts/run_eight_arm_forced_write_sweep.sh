#!/usr/bin/env bash
set -euo pipefail

# Compare forced-phase-only FastWave writing with decay versus hold outside
# the forced phase. Existing all-timestep FastWave core-comparison runs are
# included in the summary when they are present.
#
# Example:
#   DEVICE=cuda ./scripts/run_eight_arm_forced_write_sweep.sh
#
# Optional overrides:
#   EPOCHS=100 N_TRAIN=2048 N_VAL=256 N_TEST=256 BATCH_SIZE=64
#   SETTLE_STEPS=5 ETA=0.10 BETA=0.25 SKIP_EXISTING=1

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
ETA="${ETA:-0.10}"
BETA="${BETA:-0.25}"
LAM="${LAM:-0.95}"

ACTION_HOLD_STEPS=1
N_ARMS=8
ARM_LEN=3
REWARD_HOLD_STEPS=1
VISIT_LEN=$((7 + ACTION_HOLD_STEPS))
SEQ_LEN=$((N_ARMS * VISIT_LEN + SETTLE_STEPS - 1))
SEEDS=(42 43 44)
NONWRITE_MODES=(decay hold)

float_tag () {
  local value="$1"
  echo "${value/./p}"
}

eta_tag="$(float_tag "${ETA}")"
beta_tag="$(float_tag "${BETA}")"

for mode in "${NONWRITE_MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="fastwrite_forced_${mode}_eta${eta_tag}_beta${beta_tag}_commitment_xv_h1_s${SETTLE_STEPS}_seed${seed}"
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
        --model fastwave \
        --task eight_arm_bump_traj \
        --choice-order random \
        --choice-objective commitment \
        --valid-choice-loss-weight "${CHOICE_WEIGHT}" \
        --wave-readout xv \
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
        --fast-write-phase forced \
        --fast-nonwrite-mode "${mode}" \
        --lam "${LAM}" \
        --eta "${ETA}" \
        --beta "${BETA}" \
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
  data/runs/fastwrite_forced_*_eta"${eta_tag}"_beta"${beta_tag}"_commitment_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
)

# Add the already-completed always-write FastWave runs when available.
for seed in "${SEEDS[@]}"; do
  baseline="data/runs/corecmp_fastwave_eta${eta_tag}_beta${beta_tag}_commitment_xv_h1_s${SETTLE_STEPS}_seed${seed}/analysis/metrics.csv"
  if [[ -f "${baseline}" ]]; then
    metric_files+=("${baseline}")
  else
    echo "Note: always-write baseline not found for seed ${seed}: ${baseline}"
  fi
done

if ((${#metric_files[@]} == 0)); then
  echo "No forced-write metric files found." >&2
  exit 1
fi

python -m src.compare_metrics \
  "${metric_files[@]}" \
  --mode all \
  --sort \
  --out data/runs/forced_write_run_level.csv

python scripts/summarize_forced_write.py \
  --input data/runs/forced_write_run_level.csv \
  --output data/runs/forced_write_summary.csv

echo
echo "Saved:"
echo "  data/runs/forced_write_run_level.csv"
echo "  data/runs/forced_write_summary.csv"
