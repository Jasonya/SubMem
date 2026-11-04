#!/bin/bash
#SBATCH --job-name=ma-8b-m9
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuH200x8,gpuA40x4,gpuA40x4-preempt,gpuA100x4,gpuA100x4-preempt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
# --output cannot expand variables; the path is relative to the directory
# sbatch is run from, which must contain results/.
#SBATCH --output=results/memoryarena_%x_%j.log

set -o pipefail
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
export MA_MAX_COMPLETION_TOKENS=4096
PORT=$((11000 + SLURM_JOB_ID % 4000))
MEMP=$((16000 + SLURM_JOB_ID % 4000))
ENVP=$((21000 + SLURM_JOB_ID % 4000))
export MA_LLM_URL="http://localhost:${PORT}/v1"
export MA_MEM_URL="http://localhost:${MEMP}"
export MA_ENV_URL="http://localhost:${ENVP}"

echo "=== $(date) vLLM Qwen3-8B on GPU0 :$PORT ==="
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port $PORT \
    --enforce-eager --max-model-len 32768 --gpu-memory-utilization 0.88 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/vllm_m9_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 90); do curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break; sleep 10; done
curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -40 vllm_m9_${SLURM_JOB_ID}.log; exit 1; }
echo "=== $(date) vLLM ready ==="

# memory server on GPU1 with local HF backbone for M9 internals
( cd memory && CUDA_VISIBLE_DEVICES=1 SUBMEM_M9_HF_MODEL="Qwen/Qwen3-8B" \
  PYTHONPATH=${SUBMEM_PARENT}:${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/memory MEM_PORT=$MEMP \
  python server.py > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/mem8010_m9_${SLURM_JOB_ID}.log 2>&1 & )
( cd env && PYTHONPATH=${SUBMEM_PARENT}:${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/env ENV_PORT=$ENVP \
  python env_server.py > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/env8011_m9_${SLURM_JOB_ID}.log 2>&1 & )
for i in $(seq 1 40); do curl -s http://localhost:${MEMP}/ >/dev/null 2>&1 && curl -s http://localhost:${ENVP}/ >/dev/null 2>&1 && break; sleep 3; done
echo "=== $(date) servers up ==="

# SMOKE: M9 math, 1 paper — proves HF-backend read works end to end
python - <<PY
import json
c=json.load(open('configs/formal_reasoning_configs/math_submem_m9_8b.json'))
c['task_specific']['max_papers']=1
c['output']['json_output_dir']='./results/json/math_8b_smoke_m9/submem_m9'
json.dump(c,open('configs/formal_reasoning_configs/math_submem_m9_8b_smoke.json','w'),indent=2)
PY
rm -rf ./results/json/math_8b_smoke_m9
echo "=== $(date) SMOKE M9 (1 paper) ==="
python run_math.py -c configs/formal_reasoning_configs/math_submem_m9_8b_smoke.json 2>&1 | tail -15
SMOKE=$(find ./results/json/math_8b_smoke_m9 -name 'result.jsonl' | head -1)
if [ -z "$SMOKE" ] || [ ! -s "$SMOKE" ]; then echo "SMOKE FAILED: no result.jsonl"; kill $VLLM_PID; exit 1; fi
grep -q "read failed" mem8010_m9_${SLURM_JOB_ID}.log && { echo "SMOKE FAILED: M9 read errors in mem server log"; kill $VLLM_PID; exit 1; }
echo "SMOKE OK: $SMOKE"

echo "########## MATH M9 (40 papers) ##########"
python run_math.py -c configs/formal_reasoning_configs/math_submem_m9_8b.json 2>&1 | grep -iE 'progress|passrate|error|paper' | tail -6

echo "########## TRAVEL submem_m9 (270 groups, 2 shards) ##########"
export MA_MAX_GROUPS=0
run_shard () {
  OPENAI_API_BASE="http://localhost:${PORT}/v1" OPENAI_BASE_URL="http://localhost:${PORT}/v1" \
  MA_GROUP_SHARD="$1,2" \
    python run_travel.py --config configs/travel_planner_configs/submem_m9_8b.json \
    > travel8b_submem_m9_s${1}_${SLURM_JOB_ID}.log 2>&1
}
run_shard 0 & P1=$!
run_shard 1 & P2=$!
wait $P1 $P2   # explicit PIDs: bare wait would block on the vLLM server forever
unset MA_GROUP_SHARD
OPENAI_API_BASE="http://localhost:${PORT}/v1" OPENAI_BASE_URL="http://localhost:${PORT}/v1" \
  python run_travel.py --config configs/travel_planner_configs/submem_m9_8b.json 2>&1 | grep -E 'Groups:|PS:|SPS:|SR:' | tail -5

# GATE: ALL DONE only counts if travel actually produced results
N=$(ls results/travel_8b/submem_m9/generated_plan_*.json 2>/dev/null | wc -l)
echo "TRAVEL RESULT COUNT submem_m9: $N/270"
if [ "$N" -lt 250 ]; then echo "GATE FAILED: incomplete travel results"; kill $VLLM_PID 2>/dev/null; exit 1; fi

echo "=== $(date) ALL DONE ==="
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
