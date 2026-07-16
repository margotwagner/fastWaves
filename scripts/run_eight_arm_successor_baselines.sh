#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
N_TRAIN="${N_TRAIN:-2048}"
N_VAL="${N_VAL:-256}"
N_TEST="${N_TEST:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-40}"
SEEDS="${SEEDS:-42 43 44}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SEQ_LENGTH="${SEQ_LENGTH:-4}"
DELAY_STEPS="${DELAY_STEPS:-5}"
QUERY_HOLD_STEPS="${QUERY_HOLD_STEPS:-2}"
SEQ_LEN=$((SEQ_LENGTH + DELAY_STEPS + QUERY_HOLD_STEPS))

mkdir -p data/runs

COMMON=(
  --task eight_arm_successor
  --n-space 40
  --seq-len "${SEQ_LEN}"
  --n-arms 8
  --arm-len 3
  --successor-seq-length "${SEQ_LENGTH}"
  --successor-delay-steps "${DELAY_STEPS}"
  --successor-query-hold-steps "${QUERY_HOLD_STEPS}"
  --bump-sigma 0.75
  --n-train "${N_TRAIN}"
  --n-val "${N_VAL}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --hidden-dim "${HIDDEN_DIM}"
  --channels 1
  --kernel-size 7
  --patch-size 5
  --wave-readout xv
  --fast-update transition
  --lam 0.95
  --eta 0.10
  --beta 0.25
  --lr 1e-3
  --device "${DEVICE}"
)

run_one() {
  local model="$1"
  local seed="$2"
  local run_name="successor_${model}_L${SEQ_LENGTH}_d${DELAY_STEPS}_q${QUERY_HOLD_STEPS}_seed${seed}"
  local run_dir="data/runs/${run_name}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${run_dir}/successor_analysis/metrics.csv" ]]; then
    echo "Skipping completed ${run_name}"
    return
  fi

  if [[ ! -f "${run_dir}/best.pt" ]]; then
    extra=()
    if [[ "${model}" == "fastwave" ]]; then
      extra+=(--fast-write-phase forced --fast-nonwrite-mode hold)
    fi

    python -m src.train \
      --model "${model}" \
      --seed "${seed}" \
      --run-name "${run_name}" \
      "${COMMON[@]}" \
      "${extra[@]}"
  else
    echo "Training checkpoint already exists for ${run_name}; running analysis"
  fi

  python scripts/analyze_successor_recall.py \
    --ckpt "${run_dir}/best.pt" \
    --device "${DEVICE}" \
    --n-test "${N_TEST}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "$((10000 + seed))"
}

for seed in ${SEEDS}; do
  for model in vanilla gru wave fastwave; do
    run_one "${model}" "${seed}"
  done
done

python scripts/summarize_successor_recall.py \
  --runs-dir data/runs \
  --pattern "successor_*_L${SEQ_LENGTH}_d${DELAY_STEPS}_q${QUERY_HOLD_STEPS}_seed*" \
  --out-run-level data/runs/successor_recall_run_level.csv \
  --out-summary data/runs/successor_recall_summary.csv
