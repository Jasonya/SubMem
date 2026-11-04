#!/bin/bash
# Find the latest run dir per config and submit a grading sbatch for each.
# Usage: bash scripts/grade_all_bcb.sh [CFG1 CFG2 ...]
# Env: SUBMEM_ROOT (default: the checkout containing this script),
#      SUBMEM_RESULTS (default: $SUBMEM_ROOT/results).
set -eo pipefail

# Repo root: $SUBMEM_ROOT, else derived from this script's location (scripts/..).
SUBMEM_ROOT="${SUBMEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"

if [[ $# -gt 0 ]]; then
  CFGS=("$@")
else
  CFGS=(C1 C2 C3 C4 C5 C6 C7 C13 C14 C15 C16)
fi

echo "Grading ${#CFGS[@]} configs"
for CFG in "${CFGS[@]}"; do
  # Pick most recent run dir containing this CFG
  latest=$(ls -dt "$SUBMEM_RESULTS"/*_bcb_*/"$CFG" 2>/dev/null | head -1)
  if [[ -z "$latest" || ! -f "$latest/preds_bcb.jsonl" ]]; then
    echo "  $CFG: NO preds_bcb.jsonl found — skipping"
    continue
  fi
  echo "  $CFG: $latest"
  bash "$SUBMEM_ROOT/scripts/eval_bcb_run.sh" "$latest" | grep -E "Submitted|pass_rate" | head -2
done
