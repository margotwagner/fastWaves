#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
N_TRAIN="${N_TRAIN:-4096}"
N_VAL="${N_VAL:-512}"
N_TEST="${N_TEST:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-40}"
SEEDS="${SEEDS:-42 43 44}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
N_PAIRS="${N_PAIRS:-3}"
DELAY_STEPS="${DELAY_STEPS:-0}"
QUERY_HOLD_STEPS="${QUERY_HOLD_STEPS:-2}"
# Two frames per pair, one reset between adjacent pairs, one reset before query.
SEQ_LEN=$((2 * N_PAIRS + (N_PAIRS - 1) + DELAY_STEPS + 1 + QUERY_HOLD_STEPS))

COMMON=(
  --task eight_arm_transition_recall
  --n-space 40
  --seq-len "${SEQ_LEN}"
  --n-arms 8
  --arm-len 3
  --transition-n-pairs "${N_PAIRS}"
  --transition-delay-steps "${DELAY_STEPS}"
  --transition-query-hold-steps "${QUERY_HOLD_STEPS}"
  --transition-reset-between-pairs
  --transition-reset-before-query
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
  --lam 1.0
  --eta "${ETA:-0.50}"
  --beta "${BETA:-0.25}"
  --lr 1e-3
  --device "${DEVICE}"
)

run_one() {
  local model="$1"
  local seed="$2"
  local run_name="transition_independent_norm_${model}_p${N_PAIRS}_d${DELAY_STEPS}_q${QUERY_HOLD_STEPS}_seed${seed}"
  local run_dir="data/runs/${run_name}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${run_dir}/transition_analysis/metrics.csv" ]]; then
    if [[ "${model}" != "fastwave" || -f "${run_dir}/transition_ablations/metrics.csv" ]]; then
      echo "Skipping completed ${run_name}"
      return
    fi
  fi

  if [[ ! -f "${run_dir}/best.pt" ]]; then
    extra=()
    if [[ "${model}" == "fastwave" ]]; then
      extra+=(--fast-write-phase forced --fast-nonwrite-mode hold --fast-patch-norm l2 --no-fast-readout-bias)
    fi
    python -m src.train \
      --model "${model}" \
      --seed "${seed}" \
      --run-name "${run_name}" \
      "${COMMON[@]}" \
      "${extra[@]}"
  fi

  python scripts/analyze_transition_recall.py \
    --ckpt "${run_dir}/best.pt" \
    --device "${DEVICE}" \
    --n-test "${N_TEST}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "$((20000 + seed))"

  if [[ "${model}" == "fastwave" ]]; then
    python scripts/analyze_transition_ablations.py \
      --ckpt "${run_dir}/best.pt" \
      --device "${DEVICE}" \
      --n-test "${N_TEST}" \
      --batch-size "${BATCH_SIZE}" \
      --seed "$((30000 + seed))"
  fi
}

for seed in ${SEEDS}; do
  for model in vanilla gru wave fastwave; do
    run_one "${model}" "${seed}"
  done
done

python scripts/summarize_transition_recall.py \
  --runs-dir data/runs \
  --pattern "transition_independent_norm_*_p${N_PAIRS}_d${DELAY_STEPS}_q${QUERY_HOLD_STEPS}_seed*" \
  --out-run-level data/runs/transition_independent_norm_run_level.csv \
  --out-summary data/runs/transition_independent_norm_summary.csv
