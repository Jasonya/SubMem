#!/usr/bin/env bash
# One-shot env bootstrap for a fresh clone of SubMem.
#
# Creates a conda env named "submem" (override with CONDA_ENV_NAME) and
# installs every dep listed in environment.yml. Then clones the upstream
# A-mem repo locally and installs it editable so C16 can import
# `agentic_memory.memory_system`.
#
# Usage:
#   bash scripts/setup_env.sh          # creates env "submem"
#   CONDA_ENV_NAME=foo bash scripts/setup_env.sh   # custom name
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMEM_ROOT="$(cd "$_here/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-submem}"

# If ENV_NAME starts with "/", treat it as a prefix path (conda env --prefix).
# Otherwise use -n (named env in default envs_dirs).
if [[ "$ENV_NAME" == /* ]]; then
    ENV_FLAG="--prefix"
else
    ENV_FLAG="-n"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH — install miniforge/miniconda first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $NF}' | grep -qxF "$ENV_NAME"; then
    echo "[setup] conda env '$ENV_NAME' exists — skipping create."
else
    echo "[setup] creating conda env '$ENV_NAME' from environment.yml..."
    conda env create $ENV_FLAG "$ENV_NAME" -f "$SUBMEM_ROOT/environment.yml"
fi

set +u
conda activate "$ENV_NAME"
set -u

mkdir -p "$SUBMEM_ROOT/third_party"

# A-mem (C16): clone and install editable so C16 can `from
# agentic_memory.memory_system import AgenticMemorySystem`.
AMEM_DIR="$SUBMEM_ROOT/third_party/A-mem"
if [[ ! -d "$AMEM_DIR" ]]; then
    echo "[setup] cloning agiresearch/A-mem..."
    git clone https://github.com/agiresearch/A-mem.git "$AMEM_DIR"
fi
pip install -e "$AMEM_DIR"

# ml-epicache (C12 reference, optional). Our C12 runs an utterance-
# granularity adaptation (see memory/c12_epicache.py module docstring),
# so the upstream repo is not required to reproduce our numbers — we
# clone it only if INSTALL_EPICACHE=1 for users who want to compare.
EPI_DIR="$SUBMEM_ROOT/third_party/ml-epicache"
if [[ ! -d "$EPI_DIR" ]] && [[ "${INSTALL_EPICACHE:-0}" == "1" ]]; then
    echo "[setup] cloning apple/ml-epicache (optional reference)..."
    git clone https://github.com/apple/ml-epicache.git "$EPI_DIR"
fi

# SnapKV (C11 reference, optional). Same rationale — our C11 is an
# utterance-granularity adaptation; the reference repo is Llama/Mistral
# specific and needs flash_attn, which compute nodes may not have nvcc
# for. Clone with INSTALL_SNAPKV=1 if you want to compare directly.
SNAP_DIR="$SUBMEM_ROOT/third_party/snapkv"
if [[ ! -d "$SNAP_DIR" ]] && [[ "${INSTALL_SNAPKV:-0}" == "1" ]]; then
    echo "[setup] cloning FasterDecoding/SnapKV (optional reference)..."
    git clone https://github.com/FasterDecoding/SnapKV.git "$SNAP_DIR"
fi

echo
echo "[setup] done. Activate with:"
echo "    conda activate $ENV_NAME"
echo
echo "Required env vars (put in a .env at repo root):"
echo "    OPENAI_API_KEY=sk-..."
echo
echo "Optional overrides:"
echo "    HF_HOME=\$HOME/.cache/huggingface"
echo "    SUBMEM_RESULTS=./results"
echo "    MEM0_DIR=./results/mem0_db"
