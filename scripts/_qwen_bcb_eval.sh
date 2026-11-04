#!/bin/bash
#SBATCH --job-name=qwen-bcb-eval
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/qwen_bcb_eval_%j.log
set -eo pipefail
SUBMEM_ROOT="${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"
SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"

SAMPLES="$SUBMEM_RESULTS/qwen_bcb_gen/qwen3-32b-all.jsonl"
cd "$SUBMEM_RESULTS"
bigcodebench.evaluate \
  --split instruct --subset full \
  --samples "$SAMPLES" \
  --execution local --parallel 4 \
  --calibrated False 2>&1 | tail -50
