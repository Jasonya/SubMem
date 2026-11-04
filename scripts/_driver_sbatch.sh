#!/bin/bash
#SBATCH --job-name=bcb-driver
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/bcb_driver_%x_%j.log
#
# Run a BCB driver for a single config on the configured subset.
# Usage (via --export; SUBMEM_ROOT must point at the SubMem checkout):
#   SUBMEM_ROOT=<path/to/SubMem> CFG=C7 SUBSET_IDS=... MAX_Q=50 \
#     sbatch --export=ALL --job-name=bcb-C7 scripts/_driver_sbatch.sh

set -eo pipefail

: "${CFG:?CFG required}"
: "${SUBSET_IDS:?SUBSET_IDS required}"
MAX_Q="${MAX_Q:-50}"
SUBMEM_ROOT="${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"
set -a; . "$SUBMEM_ROOT/.env"; set +a
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"

cd "$(dirname "$SUBMEM_ROOT")"
python SubMem/runners/run_agentic_bcb.py \
  --memory "$CFG" \
  --subset-ids "$SUBSET_IDS" \
  --max-queries "$MAX_Q" \
  --top-k 3 \
  --max-new-tokens 1024 2>&1 | tail -30
