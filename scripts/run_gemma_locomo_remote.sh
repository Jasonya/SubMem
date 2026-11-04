#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# Run Gemma-4-26B on LoCoMo from a clean checkout on another machine.
#
# Usage on the new machine:
#   git clone https://github.com/Jasonya/SubMem.git
#   cd SubMem
#   bash scripts/run_gemma_locomo_remote.sh <SUBSTRATE> [TOPK] [N_GPUS] [TIME_HRS]
#
# Examples:
#   bash scripts/run_gemma_locomo_remote.sh C16          # default k=10, 4 GPU, 8h
#   bash scripts/run_gemma_locomo_remote.sh C10 10 4 12  # 12h
#
# Substrates that need 4 GPUs to fit 26B bf16: C4 C10 C12 C16
# Other substrates fit on 1 GPU: C1 C2 C7 C9 C15
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SUB="${1:?usage: $0 SUBSTRATE [TOPK] [N_GPUS] [TIME_HRS]}"
TOPK="${2:-10}"
NGPU="${3:-4}"
TIME_HRS="${4:-8}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Verify .env exists with OPENAI_API_KEY ─────────────────────
if [[ ! -f .env ]]; then
    cat <<'EOF' >&2
[ERROR] .env not found at repo root. Create one with:

  OPENAI_API_KEY=sk-...
  SUBMEM_REAL_OPENAI_KEY=sk-...
  SUBMEM_REAL_OPENAI_BASE=https://api.openai.com/v1

OPENAI_API_KEY is required for: C4 (gpt-4o-mini gist), C16 (A-Mem evolution),
P4 LLM-judge, and the embedding service used by C1.
EOF
    exit 1
fi

# ── 2. Ensure conda env exists ────────────────────────────────────
if ! command -v conda >/dev/null; then
    echo "[ERROR] conda not on PATH. Install miniforge first." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^submem "; then
    echo "[setup] creating submem env from environment.yml ..."
    bash scripts/setup_env.sh
fi
conda activate submem

# ── 3. Pre-cache the Gemma model so SLURM job doesn't redownload ──
MODEL="google/gemma-4-26B-A4B-it"
echo "[setup] pre-fetching $MODEL ..."
python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('$MODEL', cache_dir=os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))
print('done.')
"

# ── 4. Build SLURM script with OOM-safe config ────────────────────
JOB_TAG="${SUB}_k${TOPK}_gemma26B"
SBATCH_FILE="/tmp/submem_gemma_${JOB_TAG}.sh"
LOG_DIR="${REPO_ROOT}/sweep_logs"
mkdir -p "$LOG_DIR"

cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=submem-gemma-${JOB_TAG}
#SBATCH --partition=${SLURM_PARTITION:-gpuA100x8}
#SBATCH --account=${SLURM_ACCOUNT:-<your-allocation>}
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=${TIME_HRS}:00:00
#SBATCH --output=${LOG_DIR}/${JOB_TAG}_%j.log

set -euo pipefail
cd "${REPO_ROOT}"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate submem
source scripts/common.sh

# OOM mitigations:
#   1. expandable_segments avoids fragmentation across long inferences
#   2. evo-disable after 5 JSON parse failures (already in c16_amem upstream patch)
#   3. C9 LoRA target_modules regex (already in c9_lora.py) for Gemma4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="\${HF_HOME:-\$HOME/.cache/huggingface}"

echo "[GPUs] \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python3 runners/run_c1c2_test.py \\
    --memory ${SUB} \\
    --model ${MODEL} \\
    --max-samples 10 \\
    --top-k ${TOPK} \\
    --aux-model openai/gpt-4o-mini \\
    --encoder-model text-embedding-3-small \\
    --dataset locomo \\
    --device-map auto \\
    --fast 2>&1
EOF

# ── 5. Submit ─────────────────────────────────────────────────────
echo
echo "[submit] sbatch $SBATCH_FILE"
JOBID=$(sbatch --parsable "$SBATCH_FILE")
echo "[ok] submitted job $JOBID — log: $LOG_DIR/${JOB_TAG}_${JOBID}.log"
echo "[track] squeue -j $JOBID"
echo "[result] $REPO_ROOT/results/<ts>_locomo_gemma-4-26B-A4B-it_pid*/${SUB}/07_metrics.json"
