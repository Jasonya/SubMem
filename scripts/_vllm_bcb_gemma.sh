#!/bin/bash
#SBATCH --job-name=bcb-gem
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=results/bcb_gem_%x_%j.log
# vLLM Gemma-4-26B-A4B-it (MoE 26B/4B-active) for BCB. H200 only.
# Env: SUBMEM_ROOT (required), CFG, SUBSET_IDS, MAX_Q, TOP_K, EXTRA_ARGS,
#      SUBMEM_RESULTS / HF_HOME (default $SUBMEM_ROOT/results, $HOME/.cache/huggingface).

set -eo pipefail
: "${CFG:?CFG required}"
: "${SUBSET_IDS:?SUBSET_IDS required}"
MAX_Q="${MAX_Q:-148}"
TOP_K="${TOP_K:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---balanced-retrieval}"
SUBMEM_ROOT="${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set -a; . "$SUBMEM_ROOT/.env"; set +a
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export FAISS_OPT_LEVEL=AVX2

MODEL=google/gemma-4-26B-A4B-it
PORT=$((8700 + RANDOM % 100))
VLLM_LOG="$SUBMEM_RESULTS/bcb_gem_vllm_${SLURM_JOB_ID}.log"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port $PORT \
  --served-model-name gemma-4-26b-a4b-it gpt-4o-mini \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 180); do
  grep -q "Application startup complete" "$VLLM_LOG" 2>/dev/null && { echo "vLLM ready after ${i}*15s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "ERROR: vLLM died"; tail -80 "$VLLM_LOG"; exit 1; }
  sleep 15
done

# Route embeddings to real OpenAI (vLLM has no /v1/embeddings).
export OPENAI_EMBEDDINGS_API_KEY="$OPENAI_API_KEY"
export OPENAI_EMBEDDINGS_BASE_URL="https://api.openai.com/v1"
# Chat goes to local vLLM:
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export SUBMEM_REAL_OPENAI_KEY="$OPENAI_API_KEY"
export SUBMEM_REAL_OPENAI_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY=dummy

cd "$(dirname "$SUBMEM_ROOT")"
python SubMem/runners/run_agentic_bcb.py \
  --memory "$CFG" \
  --model-name "openai/gemma-4-26b-a4b-it" \
  --subset-ids "$SUBSET_IDS" \
  --max-queries "$MAX_Q" \
  --top-k "$TOP_K" \
  --max-new-tokens 1024 \
  --no-think $EXTRA_ARGS 2>&1
RC=$?
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
exit $RC
