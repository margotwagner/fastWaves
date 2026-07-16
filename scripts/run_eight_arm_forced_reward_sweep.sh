#!/usr/bin/env bash
set -euo pipefail

# Train only the new forced+reward-event / hold condition across three seeds.
# The final summary also includes previously completed Wave, always-write,
# forced-decay, and forced-hold runs when those files are present.

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

float_tag () {
  local value="$1"
  echo "${value/./p}"
}

eta_tag="$(float_tag "${ETA}")"
beta_tag="$(float_tag "${BETA}")"

for seed in "${SEEDS[@]}"; do
  run_name="fastwrite_forced_reward_hold_eta${eta_tag}_beta${beta_tag}_commitment_xv_h1_s${SETTLE_STEPS}_seed${seed}"
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
      --fast-write-phase forced_reward \
      --fast-nonwrite-mode hold \
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

shopt -s nullglob
metric_files=(
  data/runs/fastwrite_forced_reward_hold_eta"${eta_tag}"_beta"${beta_tag}"_commitment_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
  data/runs/fastwrite_forced_decay_eta"${eta_tag}"_beta"${beta_tag}"_commitment_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
  data/runs/fastwrite_forced_hold_eta"${eta_tag}"_beta"${beta_tag}"_commitment_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
  data/runs/corecmp_fastwave_eta"${eta_tag}"_beta"${beta_tag}"_commitment_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
  data/runs/corecmp_wave_*_commitment_xv_h1_s"${SETTLE_STEPS}"_seed*/analysis/metrics.csv
)

if ((${#metric_files[@]} == 0)); then
  echo "No metric files found." >&2
  exit 1
fi

python -m src.compare_metrics \
  "${metric_files[@]}" \
  --mode all \
  --sort \
  --out data/runs/write_schedule_run_level.csv

python scripts/summarize_write_schedules.py \
  --input data/runs/write_schedule_run_level.csv \
  --output data/runs/write_schedule_summary.csv

echo
echo "Saved:"
echo "  data/runs/write_schedule_run_level.csv"
echo "  data/runs/write_schedule_summary.csv"
