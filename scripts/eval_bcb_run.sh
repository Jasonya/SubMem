#!/bin/bash
# Grade a BigCodeBench driver output: run bigcodebench.evaluate on the
# emitted preds_bcb.jsonl and merge pass_rate_true into 07_metrics.json.
#
# Usage:
#   bash scripts/eval_bcb_run.sh <submem_run_dir>
# Example:
#   bash scripts/eval_bcb_run.sh results/20260419_..._bcb_gpt-4o-mini/C1
#
# Env: SLURM_ACCOUNT (allocation for the grading job), SUBMEM_ROOT (default:
# the checkout containing this script), CONDA_SH / CONDA_ENV (conda setup).
set -eo pipefail

SUBMEM_DIR="${1:?usage: $0 <submem_run_dir>}"
JSONL="$SUBMEM_DIR/preds_bcb.jsonl"
[[ -f "$JSONL" ]] || { echo "ERR: $JSONL missing"; exit 1; }

# Repo root: $SUBMEM_ROOT, else derived from this script's location (scripts/..).
SUBMEM_ROOT="${SUBMEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Guard: skip if a bcb-grade for this dir is already queued/running, or the
# run is already graded. Prevents duplicate submissions from stale loops.
if [[ -f "$SUBMEM_DIR/preds_bcb_eval_results.json" ]]; then
  echo "SKIP: already graded ($SUBMEM_DIR/preds_bcb_eval_results.json exists)"
  exit 0
fi
EXISTING=$(squeue -u "$USER" -h -o "%j|%Z" 2>/dev/null \
  | awk -F'|' -v d="$SUBMEM_DIR" '$1=="bcb-grade" && index($2,d){c++} END{print c+0}')
if [[ "${EXISTING:-0}" -gt 0 ]]; then
  echo "SKIP: $EXISTING bcb-grade job(s) already in queue for $SUBMEM_DIR"
  exit 0
fi
# Global ceiling: hard cap on total bcb-grade jobs across all dirs.
TOTAL=$(squeue -u "$USER" -h -n bcb-grade 2>/dev/null | wc -l)
MAX_BCB_GRADE="${MAX_BCB_GRADE:-4}"
if [[ "${TOTAL:-0}" -ge "$MAX_BCB_GRADE" ]]; then
  echo "SKIP: $TOTAL bcb-grade already queued (MAX_BCB_GRADE=$MAX_BCB_GRADE)"
  exit 0
fi

cat > /tmp/_bcb_grade_$$.sh <<EOF
#!/bin/bash
#SBATCH --job-name=bcb-grade
#SBATCH --account=${SLURM_ACCOUNT:-<your-allocation>}
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=$SUBMEM_DIR/bcb_grade_%j.log

set -eo pipefail
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"

cd $SUBMEM_DIR
# Build --selective_evaluate CSV of the task_ids we actually have predictions for.
# BCB evaluator asserts len(preds) == 1140 unless we restrict the set.
TASK_IDS=\$(python -c "import json; ids=[json.loads(l)['task_id'] for l in open('$JSONL')]; print(','.join(ids))")

bigcodebench.evaluate \\
  --split instruct --subset full \\
  --samples $JSONL \\
  --selective_evaluate "\$TASK_IDS" \\
  --execution local --parallel 4 \\
  --calibrated True 2>&1 | tail -80

# Merge pass rate into 07_metrics.json
python "$SUBMEM_ROOT/scripts/bcb_update_metrics.py" \\
  --submem-run-dir $SUBMEM_DIR
EOF

JOB_ID=$(sbatch --parsable /tmp/_bcb_grade_$$.sh)
rm /tmp/_bcb_grade_$$.sh
echo "Submitted grading job $JOB_ID"
echo "Monitor: sacct -j $JOB_ID -X"
echo "After completion, pass_rate_true will be in $SUBMEM_DIR/07_metrics.json"
