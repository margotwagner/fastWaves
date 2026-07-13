#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper. Run from the repository root:
#   DEVICE=cuda ./scripts/run_eight_arm_stage1b_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_eight_arm_stage1b_loss_balance.sh"
"${SCRIPT_DIR}/run_eight_arm_stage1b_globalfast.sh"

echo
echo "Stage 1B complete."
echo "Outputs:"
echo "  data/runs/stage1b_loss_balance_comparison.csv"
echo "  data/runs/stage1b_globalfast_comparison.csv"
