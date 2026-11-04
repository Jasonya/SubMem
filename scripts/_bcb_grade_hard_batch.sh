#!/bin/bash
#SBATCH --job-name=bcb-grade-hard-batch
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/bcb_grade_hard_batch_%j.log

set -eo pipefail
SUBMEM_ROOT="${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"
SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"

# List of hard-scan runs to grade
CFGS="C1 C2 C3 C4 C5 C6 C7 C9 C10 C13 C14"

for cfg in $CFGS; do
  run=$(ls -td "$SUBMEM_RESULTS"/20260421_1*_bcb_*/$cfg 2>/dev/null | head -1)
  if [[ -z "$run" ]]; then echo "SKIP $cfg: no run dir"; continue; fi
  if [[ -f "$run/preds_bcb_eval_results.json" ]]; then echo "SKIP $cfg: already graded"; continue; fi
  if [[ ! -f "$run/preds_bcb.jsonl" ]]; then echo "SKIP $cfg: no preds"; continue; fi

  echo "=== grading $cfg: $run ==="
  cd "$run"
  TASK_IDS=$(python -c "import json; ids=[json.loads(l)['task_id'] for l in open('preds_bcb.jsonl')]; print(','.join(ids))")
  bigcodebench.evaluate \
    --split instruct --subset full \
    --samples "$run/preds_bcb.jsonl" \
    --selective_evaluate "$TASK_IDS" \
    --execution local --parallel 4 \
    --calibrated True 2>&1 | tail -20
  python "$SUBMEM_ROOT/scripts/bcb_update_metrics.py" --submem-run-dir "$run" 2>&1 | tail -10
done
