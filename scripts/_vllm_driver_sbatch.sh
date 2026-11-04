#!/bin/bash
#SBATCH --job-name=bcb-qwen
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/bcb_qwen_%x_%j.log
#
# Start Qwen3-32B-AWQ via vLLM, then run SubMem BCB driver.
#
# Env vars:
#   SUBMEM_ROOT (required) path to the SubMem checkout
#   VLLM_SIF    (required) vLLM Apptainer image (e.g. vllm-openai_latest.sif)
#   CFG         (required) memory config (C1..C16)
#   SUBSET_IDS  (required) path to task_id list
#   MAX_Q       (default 50)
#   SUBMEM_RESULTS / HF_HOME  (default $SUBMEM_ROOT/results, $HOME/.cache/huggingface)

set -eo pipefail

: "${CFG:?CFG required}"
: "${SUBSET_IDS:?SUBSET_IDS required}"
MAX_Q="${MAX_Q:-50}"
SUBMEM_ROOT="${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"
# Ensure conda's libstdc++ is preferred over the system one (fixes
# CXXABI_1.3.15 missing error for C5 Mem0 / C6 Graphiti / C16 A-Mem deps).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set -a; . "$SUBMEM_ROOT/.env"; set +a
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

SIF="${VLLM_SIF:?set VLLM_SIF to the vLLM Apptainer image}"
PORT=8765
MODEL_ID=Qwen/Qwen3-32B-AWQ
# Use HF model id; weights are cached under $HF_HOME which we bind into the container.
echo "Using model id: $MODEL_ID (cache: $HF_HOME)"

# Start vLLM server in background (logs to a separate file).
VLLM_LOG="$SUBMEM_RESULTS/bcb_qwen_vllm_${SLURM_JOB_ID}.log"
echo "Starting vLLM on :$PORT ..."
apptainer run --nv \
  --env "HF_HOME=/root/.cache/huggingface" \
  --bind "$HF_HOME:/root/.cache/huggingface" \
  "$SIF" \
    --model "$MODEL_ID" \
    --served-model-name qwen3-32b gpt-4o-mini \
    --quantization awq_marlin \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port $PORT \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID; log: $VLLM_LOG"

# Wait until ready (poll /v1/models). Cold weight-load + AOT cache warmup
# can take 10-15 min on Lustre; allow up to 30 min.
echo "Waiting for vLLM to come up..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready after ${i}*15s"; break
  fi
  sleep 15
done
if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERR: vLLM did not come up; last log:"
  tail -40 "$VLLM_LOG"
  kill $VLLM_PID 2>/dev/null; exit 1
fi

# Route chat/completions to local vLLM, embeddings to real OpenAI.
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_EMBEDDINGS_API_KEY="$OPENAI_API_KEY"
# Preserve real key + default OpenAI URL so C5 (mem0) can bypass vLLM for
# its internal embedding + mgmt-LLM calls. Any substrate that wants to talk
# to the real OpenAI API reads SUBMEM_REAL_OPENAI_KEY / SUBMEM_REAL_OPENAI_BASE.
export SUBMEM_REAL_OPENAI_KEY="$OPENAI_API_KEY"
export SUBMEM_REAL_OPENAI_BASE="https://api.openai.com/v1"
# The Submem driver uses the same OPENAI_API_KEY for chat; vLLM ignores it.
export OPENAI_API_KEY="dummy"

cd "$(dirname "$SUBMEM_ROOT")"
TOP_K="${TOP_K:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---balanced-retrieval}"
python SubMem/runners/run_agentic_bcb.py \
  --memory "$CFG" \
  --model-name "openai/qwen3-32b" \
  --subset-ids "$SUBSET_IDS" \
  --max-queries "$MAX_Q" \
  --top-k "$TOP_K" \
  --max-new-tokens 1024 \
  --no-think $EXTRA_ARGS 2>&1

RC=$?
echo "Driver rc=$RC"
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
exit $RC
