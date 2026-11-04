#!/bin/bash
#SBATCH --job-name=ma-8b-mb3
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuH200x8,gpuA40x4,gpuA40x4-preempt,gpuA100x4,gpuA100x4-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

# real OpenAI for aux (gpt-4o-mini), embeddings, and the benchmark judge
set -a; source "$SUBMEM_ROOT/.env" 2>/dev/null; set +a
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_BASE="https://api.openai.com/v1"
export PYTHONPATH="${SUBMEM_PARENT}:${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}"
export MA_MEM_TIMEOUT=1800
PORT=$((11000 + SLURM_JOB_ID % 4000))
MEMP=$((16000 + SLURM_JOB_ID % 4000))
ENVP=$((21000 + SLURM_JOB_ID % 4000))
export MA_LLM_URL="http://localhost:${PORT}/v1"
export MA_MEM_URL="http://localhost:${MEMP}"
export MA_ENV_URL="http://localhost:${ENVP}"

echo "=== $(date) starting vLLM Qwen3-8B on :$PORT ==="
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port $PORT \
    --enforce-eager --max-model-len 32768 --gpu-memory-utilization 0.88 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 90); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && break
    sleep 10
done
curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 || { echo "vLLM FAILED"; tail -60 vllm_${SLURM_JOB_ID}.log; kill $VLLM_PID; exit 1; }
echo "=== $(date) vLLM ready ==="

# memory server (:8010) uses real OpenAI for aux+embeddings; env server (:8011)
( cd memory && PYTHONPATH=${SUBMEM_PARENT}:${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/memory MEM_PORT=$MEMP python server.py > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/mem8010_${SLURM_JOB_ID}.log 2>&1 & )
( cd env && PYTHONPATH=${SUBMEM_PARENT}:${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/env ENV_PORT=$ENVP python env_server.py > ${MEMORYARENA_ROOT:?set MEMORYARENA_ROOT}/env8011_${SLURM_JOB_ID}.log 2>&1 & )
for i in $(seq 1 40); do curl -sf http://localhost:${MEMP}/ >/dev/null 2>&1 && curl -sf http://localhost:${ENVP}/ >/dev/null 2>&1 && break; sleep 3; done
echo "=== $(date) servers up (mem log / env log tails) ==="; tail -3 mem8010_${SLURM_JOB_ID}.log; tail -3 env8011_${SLURM_JOB_ID}.log

run_math_smoke () {
  rm -rf ./results/json/math_8b_smoke
  python - <<PY
import json
c=json.load(open('configs/formal_reasoning_configs/math_bm25_8b.json'))
c['task_specific']['max_papers']=1
c['output']['json_output_dir']='./results/json/math_8b_smoke/bm25'
json.dump(c,open('configs/formal_reasoning_configs/math_bm25_8b_smoke.json','w'),indent=2)
PY
  python run_math.py -c configs/formal_reasoning_configs/math_bm25_8b_smoke.json 2>&1 | tail -25
}

echo "=== $(date) SMOKE: math bm25, 1 paper (agent=vLLM 8B, judge=OpenAI) ==="
run_math_smoke
SMOKE=$(find ./results/json/math_8b_smoke -name 'result.jsonl' | head -1)
if [ -z "$SMOKE" ] || [ ! -s "$SMOKE" ]; then echo "SMOKE FAILED: no result.jsonl"; kill $VLLM_PID; exit 1; fi
echo "SMOKE OK: $SMOKE ($(wc -l < "$SMOKE") lines)"
grep -c '<think>' "$SMOKE" >/dev/null 2>&1 && echo "WARNING: <think> present in smoke output" || echo "no <think> tags — thinking disabled OK"
echo "=== smoke ok, starting full runs ==="

echo "########## MATH (40 papers x 10, 3-way parallel) ##########"
run_one_math () {
  echo "===== $(date) MATH START $1 ====="
  python run_math.py -c configs/formal_reasoning_configs/math_${1}_8b.json > math8b_${1}_${SLURM_JOB_ID}.log 2>&1
  echo "===== $(date) MATH DONE $1 (tail) ====="; tail -3 math8b_${1}_${SLURM_JOB_ID}.log
}
for BATCH in "submem_m5"; do
  PIDS=()
  for S in $BATCH; do run_one_math $S & PIDS+=($!); done
  wait "${PIDS[@]}"   # explicit PIDs: bare wait would block on the vLLM server forever
done

echo "(travel moved to dedicated jobs)"

echo "=== $(date) ALL DONE ==="
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
