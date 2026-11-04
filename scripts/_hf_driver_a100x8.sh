#!/bin/bash
#SBATCH --job-name=bcb-a8
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuA100x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=results/bcb_a8_%x_%j.log
#
# BCB driver on A100-80G (gpuA100x8 = 8x80GB nodes, we take 1 GPU).
#   C9  — Qwen3-32B + QLoRA 4-bit  (pass EXTRA_ARGS="--load-in-4bit")
#   C11 — Qwen3-32B-AWQ (inference only, ~30-40GB fits 80G easily)
#   C12 — same as C11
#
# Env vars:
#   SUBMEM_ROOT (required) path to the SubMem checkout
#   CFG         (required) memory config
#   MODEL_NAME  (required) e.g. Qwen/Qwen3-32B or Qwen/Qwen3-32B-AWQ
#   SUBSET_IDS  (required)
#   MAX_Q       (default 148)
#   EXTRA_ARGS  optional extra args forwarded to run_agentic_bcb.py
#   SUBMEM_RESULTS / HF_HOME  (default $SUBMEM_ROOT/results, $HOME/.cache/huggingface)

set -eo pipefail

: "${CFG:?CFG required}"
: "${MODEL_NAME:?MODEL_NAME required}"
: "${SUBSET_IDS:?SUBSET_IDS required}"
MAX_Q="${MAX_Q:-148}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SUBMEM_ROOT="${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set -a; . "$SUBMEM_ROOT/.env"; set +a
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=0

cd "$(dirname "$SUBMEM_ROOT")"
TOP_K="${TOP_K:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---balanced-retrieval}"
python SubMem/runners/run_agentic_bcb.py \
  --memory "$CFG" \
  --model-name "$MODEL_NAME" \
  --subset-ids "$SUBSET_IDS" \
  --max-queries "$MAX_Q" \
  --top-k "$TOP_K" \
  --max-new-tokens 1024 \
  --no-think $EXTRA_ARGS 2>&1
