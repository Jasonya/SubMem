#!/bin/bash
# Run BCB driver for each retrieval-based config sequentially.
# Each produces <SUBMEM_RESULTS>/<TS>_bcb_<model>/<CFG>/preds_bcb.jsonl.
#
# Usage:
#   bash scripts/scan_bcb_configs.sh [SUBSET_IDS] [CFG1 CFG2 ...]
#
# Defaults:
#   SUBSET_IDS=$SUBMEM_RESULTS/bcb_subset50_ids.txt
#   CFGS=C1 C2 C3 C4 C5 C6 C7 C13 C14 C15 C16
# Env: SUBMEM_ROOT (default: the checkout containing this script),
#      SUBMEM_RESULTS (default: $SUBMEM_ROOT/results), CONDA_SH / CONDA_ENV.
set -eo pipefail

# Repo root: $SUBMEM_ROOT, else derived from this script's location (scripts/..).
SUBMEM_ROOT="${SUBMEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# .env may set SUBMEM_RESULTS; load it before deriving the defaults below.
set -a; . "$SUBMEM_ROOT/.env"; set +a
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"

SUBSET_IDS="${1:-$SUBMEM_RESULTS/bcb_subset50_ids.txt}"
shift || true
if [[ $# -gt 0 ]]; then
  CFGS=("$@")
else
  CFGS=(C1 C2 C3 C4 C5 C6 C7 C13 C14 C15 C16)
fi

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"

MAX_Q=$(wc -l < "$SUBSET_IDS")
echo "Scanning ${#CFGS[@]} configs on $MAX_Q tasks: ${CFGS[*]}"
echo

SCAN_LOG="$SUBMEM_RESULTS/scan_bcb_$(date +%Y%m%d_%H%M%S).log"
echo "scan log: $SCAN_LOG"

FAILED_CFGS=()
for CFG in "${CFGS[@]}"; do
  echo "=== $CFG ==="
  TS=$(date +%Y%m%d_%H%M%S)
  # Per-config failure isolated: set +e around the python call.
  set +e
  python SubMem/runners/run_agentic_bcb.py \
    --memory "$CFG" \
    --subset-ids "$SUBSET_IDS" \
    --max-queries "$MAX_Q" \
    --top-k 3 \
    --max-new-tokens 1024 \
    2>&1 | tee -a "$SCAN_LOG" | tail -4
  RC=${PIPESTATUS[0]}
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "*** $CFG FAILED (exit=$RC) — continuing ***"
    FAILED_CFGS+=("$CFG")
  fi
  echo
done

if (( ${#FAILED_CFGS[@]} > 0 )); then
  echo "=== FAILED CONFIGS ==="
  for c in "${FAILED_CFGS[@]}"; do echo "  $c"; done
fi

# List the directories produced (for grading script to pick up)
echo
echo "=== Runs produced ==="
for CFG in "${CFGS[@]}"; do
  latest=$(ls -dt "$SUBMEM_RESULTS"/*_bcb_*/"$CFG" 2>/dev/null | head -1)
  echo "  $CFG: $latest"
done
