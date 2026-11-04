#!/bin/bash
#SBATCH --job-name=bcb-hf
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/bcb_hf_%x_%j.log
#
# Run SubMem BCB driver with a LOCAL HuggingFace model (for C9/C11/C12 which
# need direct torch access and can't use vLLM OpenAI API).
#
# Env vars:
#   SUBMEM_ROOT (required) path to the SubMem checkout
#   CFG         (required) memory config (C9/C11/C12)
#   MODEL_NAME  (default Qwen/Qwen3-32B-AWQ)   # AWQ OK for inference-only C11/C12
#                (C9 LoRA needs bf16; try Qwen/Qwen3-8B instead)
#   SUBSET_IDS  (required)
#   MAX_Q       (default 50)
#   SUBMEM_RESULTS / HF_HOME  (default $SUBMEM_ROOT/results, $HOME/.cache/huggingface)

set -eo pipefail

: "${CFG:?CFG required}"
: "${SUBSET_IDS:?SUBSET_IDS required}"
MAX_Q="${MAX_Q:-50}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B-AWQ}"
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
