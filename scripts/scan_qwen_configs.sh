#!/bin/bash
# Submit sequential sbatch runs for each config on Qwen3-32B.
# Uses _vllm_driver_sbatch.sh which brings up vLLM + runs one driver.
# Each sbatch gets its own GPU + cold vLLM start; acceptable tradeoff for
# fair comparison (same model version, same server config).
#
# Usage:
#   bash scripts/scan_qwen_configs.sh [CFG1 CFG2 ...]
# Env: SUBMEM_ROOT (default: the checkout containing this script),
#      SUBMEM_RESULTS (default: $SUBMEM_ROOT/results). Both are exported so
#      the sbatch jobs (--export=ALL) inherit them.
set -eo pipefail

# Repo root: $SUBMEM_ROOT, else derived from this script's location (scripts/..).
export SUBMEM_ROOT="${SUBMEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# .env may set SUBMEM_RESULTS (the jobs source it too); load it first so the
# SUBSET_IDS default below points at the same results root the jobs will use.
[[ -f "$SUBMEM_ROOT/.env" ]] && { set -a; . "$SUBMEM_ROOT/.env"; set +a; }
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"
# The job script's own #SBATCH --output is cwd-relative; pass an absolute log
# path so submitting from any directory works.
mkdir -p "$SUBMEM_RESULTS"

if [[ $# -gt 0 ]]; then
  CFGS=("$@")
else
  CFGS=(C2 C3 C4 C5 C6 C7 C13 C14 C15 C16)
fi

SUBSET_IDS="$SUBMEM_RESULTS/bcb_subset50_ids.txt"
MAX_Q=50

echo "Submitting ${#CFGS[@]} configs: ${CFGS[*]}"
JOBS=()
for CFG in "${CFGS[@]}"; do
  JID=$(CFG="$CFG" SUBSET_IDS="$SUBSET_IDS" MAX_Q="$MAX_Q" \
        sbatch --parsable --export=ALL --job-name="bcb-qwen-$CFG" \
          --output="$SUBMEM_RESULTS/bcb_qwen_%x_%j.log" \
          "$SUBMEM_ROOT/scripts/_vllm_driver_sbatch.sh")
  echo "  $CFG -> job $JID"
  JOBS+=("$JID")
done

echo
echo "All jobs submitted. Monitor: squeue -j $(IFS=,; echo "${JOBS[*]}")"
echo "Grade each after completion via scripts/eval_bcb_run.sh"
