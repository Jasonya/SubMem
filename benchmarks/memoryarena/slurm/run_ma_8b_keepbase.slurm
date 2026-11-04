#!/bin/bash
#SBATCH --job-name=ma-8b-kb
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuH200x8,gpuA40x4,gpuA40x4-preempt,gpuA100x4,gpuA100x4-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
# --output cannot expand variables; the path is relative to the directory
# sbatch is run from, which must contain results/.
#SBATCH --output=results/memoryarena_%x_%j.log

# Full-270-group travel for a set of substrates, 2 shards each, all concurrent.
# Pass the substrate list via TRAVEL_SUBS (space-separated). Group-level skip
# makes restarts and overlap safe.
set -o pipefail
: "${TRAVEL_SUBS:?need TRAVEL_SUBS}"
SUBMEM_ROOT="${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"
SUBMEM_PARENT="${SUBMEM_PARENT:-$(dirname "$SUBMEM_ROOT")}"
export SUBMEM_ROOT SUBMEM_PARENT
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-submem}"
[ -n "${CONDA_PREFIX:-}" ] && export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
cd ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}
set -a; source "$SUBMEM_ROOT/.env" 2>/dev/null; set +a
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_BASE="https://api.openai.com/v1"
export PYTHONPATH="${SUBMEM_PARENT}:${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}"
export MA_MEM_TIMEOUT=1800
export SUBMEM_KEEP_BASE=1
export MA_MAX_COMPLETION_TOKENS=1024
# travel tool results are already huge; keep memory injection small enough that
# prompt + completion fits the 32k window (M10 full-context otherwise overflows)
export SUBMEM_MEM_CHAR_CAP=20000
PORT=$((11000 + SLURM_JOB_ID % 4000))
MEMP=$((16000 + SLURM_JOB_ID % 4000))
ENVP=$((21000 + SLURM_JOB_ID % 4000))
export MA_LLM_URL="http://localhost:${PORT}/v1"
export MA_MEM_URL="http://localhost:${MEMP}"
export MA_ENV_URL="http://localhost:${ENVP}"

echo "=== $(date) vLLM Qwen3-8B :$PORT (tools enabled) ==="
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port $PORT \
    --enforce-eager --max-model-len 32768 --gpu-memory-utilization 0.88 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/vllm_tr_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 90); do curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break; sleep 10; done
curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -40 vllm_tr_${SLURM_JOB_ID}.log; exit 1; }
( cd memory && PYTHONPATH=${SUBMEM_PARENT}:${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/memory MEM_PORT=$MEMP python server.py > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/mem8010_tr_${SLURM_JOB_ID}.log 2>&1 & )
( cd env && PYTHONPATH=${SUBMEM_PARENT}:${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/env ENV_PORT=$ENVP python env_server.py > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/env8011_tr_${SLURM_JOB_ID}.log 2>&1 & )
for i in $(seq 1 40); do curl -s http://localhost:${MEMP}/ >/dev/null 2>&1 && curl -s http://localhost:${ENVP}/ >/dev/null 2>&1 && break; sleep 3; done
echo "=== $(date) servers up; substrates: $TRAVEL_SUBS ==="

export MA_MAX_GROUPS=25
run_shard () {  # $1=substrate $2=shard-idx $3=nshards
  OPENAI_API_BASE="http://localhost:${PORT}/v1" OPENAI_BASE_URL="http://localhost:${PORT}/v1" \
  MA_GROUP_SHARD="$2,$3" \
    python run_travel.py --config configs/travel_planner_configs/${1}_8b_keepbase.json \
    > travel8b_${1}_s${2}_${SLURM_JOB_ID}.log 2>&1
  echo "===== $(date) TRAVEL SHARD DONE ${1} shard ${2}/${3} ====="
}
PIDS=()
for S in $TRAVEL_SUBS; do
  run_shard $S 0 2 & PIDS+=($!)
  run_shard $S 1 2 & PIDS+=($!)
done
wait "${PIDS[@]}"   # explicit PIDs: bare wait would block on the vLLM server forever

echo "########## EVAL PASS (all groups present, skip + combine + eval) ##########"
unset MA_GROUP_SHARD
for S in $TRAVEL_SUBS; do
  echo "===== EVAL $S ====="
  OPENAI_API_BASE="http://localhost:${PORT}/v1" OPENAI_BASE_URL="http://localhost:${PORT}/v1" \
    python run_travel.py --config configs/travel_planner_configs/${S}_8b_keepbase.json 2>&1 | grep -E 'Groups:|PS:|SPS:|SR:' | tail -5
done


echo "=== $(date) ALL DONE ==="
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
