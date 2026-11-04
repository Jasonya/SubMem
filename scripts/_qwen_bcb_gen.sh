#!/bin/bash
#SBATCH --job-name=qwen-bcb-gen
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/qwen_bcb_gen_%j.log

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

# start vLLM
VLLM_LOG="$SUBMEM_RESULTS/qwen_bcb_gen_vllm_${SLURM_JOB_ID}.log"
apptainer run --nv \
  --env "HF_HOME=/root/.cache/huggingface" \
  --bind "$HF_CACHE:/root/.cache/huggingface" \
  "$SIF" \
  --model "$MODEL_ID" --served-model-name qwen3-32b \
  --quantization awq_marlin --tensor-parallel-size 1 \
  --max-model-len 40960 --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 --host 0.0.0.0 --port $PORT \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { echo "vLLM ready ${i}*15s"; break; }
  sleep 15
done

export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="dummy"

cd "$SUBMEM_RESULTS"
mkdir -p qwen_bcb_gen
bigcodebench.generate \
  qwen3-32b instruct full \
  --backend openai \
  --bs 16 \
  --temperature 0.0 --greedy \
  --max_new_tokens 1024 \
  --root "$SUBMEM_RESULTS/qwen_bcb_gen" \
  --resume 2>&1 | tail -50

kill $VLLM_PID 2>/dev/null
