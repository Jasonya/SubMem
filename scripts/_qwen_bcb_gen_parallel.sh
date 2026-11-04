#!/bin/bash
#SBATCH --job-name=qwen-bcb-gen-par
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/qwen_bcb_gen_par_%j.log

set -eo pipefail
SUBMEM_ROOT="${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"
set -a; . "$SUBMEM_ROOT/.env"; set +a
SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"
# Host-side HF cache, bound into the container as /root/.cache/huggingface.
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

SIF="${VLLM_SIF:?set VLLM_SIF to the vLLM Apptainer image}"
PORT=8765
MODEL_ID=Qwen/Qwen3-32B-AWQ

VLLM_LOG="$SUBMEM_RESULTS/qwen_bcb_gen_par_vllm_${SLURM_JOB_ID}.log"
apptainer run --nv \
  --env "HF_HOME=/root/.cache/huggingface" \
  --bind "$HF_CACHE:/root/.cache/huggingface" \
  "$SIF" \
  --model "$MODEL_ID" --served-model-name qwen3-32b \
  --quantization awq_marlin --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --host 0.0.0.0 --port $PORT \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { echo "vLLM ready ${i}*15s"; break; }
  sleep 15
done

export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="dummy"

cd "$SUBMEM_ROOT"
python scripts/qwen_bcb_generate_parallel.py \
  --output "$SUBMEM_RESULTS/qwen_bcb_gen/qwen3-32b-all.jsonl" \
  --model qwen3-32b --concurrency 32 --max-new-tokens 1024 2>&1 | tail -40

kill $VLLM_PID 2>/dev/null
