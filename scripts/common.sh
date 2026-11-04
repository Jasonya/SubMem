#!/usr/bin/env bash
# Sourced by every run_c*_1conv.sh. Provides portable defaults that work on
# both a SLURM cluster and a clean clone on any Linux box.
#
# Every var honours its existing env value first, then falls back to a
# local default under the repo checkout or $HOME.
set -euo pipefail

# Repo root = parent of this scripts/ dir, resolved even through symlinks.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SUBMEM_ROOT:=$(cd "$_here/.." && pwd)}"

# Where HF caches models. Default: ~/.cache/huggingface.
if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME="$HOME/.cache/huggingface"
fi

# Result root. Default: repo/results.
if [[ -z "${SUBMEM_RESULTS:-}" ]]; then
    export SUBMEM_RESULTS="$SUBMEM_ROOT/results"
fi
mkdir -p "$SUBMEM_RESULTS"

# C5 Mem0 local SQLite. Default: $SUBMEM_RESULTS/mem0_db.
if [[ -z "${MEM0_DIR:-}" ]]; then
    export MEM0_DIR="$SUBMEM_RESULTS/mem0_db"
fi
mkdir -p "$MEM0_DIR"

# Auto-load .env at repo root (OPENAI_API_KEY etc.)
if [[ -f "$SUBMEM_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$SUBMEM_ROOT/.env"
    set +a
fi

# Activate conda env. Priority:
#   1. $CONDA_ENV env var (explicit path or name)
#   2. `submem` env in local miniconda/mamba
_activate_conda() {
    local env="${1:-}"
    if [[ -z "$env" ]]; then return; fi

    # Prefer 'conda activate' if conda is initialised; otherwise fall back to
    # the legacy 'source activate' shim.
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
        set +u
        if conda activate "$env" 2>/dev/null; then
            set -u
            return 0
        fi
        set -u
    fi
    # shellcheck disable=SC1091
    source activate "$env" 2>/dev/null || {
        echo "[common.sh] Could not activate '$env'. PATH python will be used." >&2
    }
}

if [[ -z "${CONDA_ENV:-}" ]]; then
    CONDA_ENV="submem"
fi
_activate_conda "$CONDA_ENV"

# Prepend conda's libstdc++ to LD_LIBRARY_PATH so CXXABI_1.3.15 (needed by
# spacy via libicui18n.so.78) resolves on compute nodes whose system
# /lib64/libstdc++.so.6 is older. Affects C18 LightRAG.
if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$SUBMEM_ROOT"

# Model + harness defaults — can all be overridden from the caller.
#
# Note: TOPK is intentionally *not* defaulted. Leaving it unset means the
# run_c*_1conv.sh scripts omit --top-k, and each substrate's read() uses
# its own paper-faithful default (e.g. C14 = 1, C16 = 5, C1 = 10). Set
# TOPK explicitly only when forcing a single k across all configs for an
# apples-to-apples sweep.
: "${MODEL:=Qwen/Qwen3-8B}"
: "${MAX_SAMPLES:=1}"
: "${AUX_MODEL:=openai/gpt-4o-mini}"
