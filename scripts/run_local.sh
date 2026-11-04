#!/usr/bin/env bash
# Non-SLURM local runner. Usage:
#   bash scripts/run_local.sh <config>     # e.g. C1, C3-Letta, C16
#   bash scripts/run_local.sh all          # runs C1..C16 sequentially (skip C14)
#
# Honours MODEL / MAX_SAMPLES / TOPK like the SLURM scripts do.
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMEM_ROOT="$(cd "$_here/.." && pwd)"
cd "$SUBMEM_ROOT"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <C1|C2|C3-Letta|C4|C5|C6|...|C16|all>"
    exit 1
fi

run_one() {
    local cfg="$1"
    local script
    case "$cfg" in
        C1|C2)    script=run_c1c2_1conv.sh ;;
        C3|C3-Letta) script=run_c3_1conv.sh ;;
        C4)       script=run_c4_1conv.sh ;;
        C5)       script=run_c5_1conv.sh ;;
        C6)       script=run_c6_1conv.sh ;;
        C7)       script=run_c7_1conv.sh ;;
        C9)       script=run_c9_1conv.sh ;;
        C10)      script=run_c10_1conv.sh ;;
        C11)      script=run_c11_1conv.sh ;;
        C12)      script=run_c12_1conv.sh ;;
        C13)      script=run_c13_1conv.sh ;;
        C15)      script=run_c15_1conv.sh ;;
        C16)      script=run_c16_1conv.sh ;;
        *) echo "unknown config: $cfg"; return 1 ;;
    esac
    echo "=== running $cfg via $script ==="
    bash "$SUBMEM_ROOT/$script"
}

if [[ "$1" == "all" ]]; then
    for c in C1 C3-Letta C4 C5 C6 C7 C9 C10 C11 C12 C13 C15 C16; do
        run_one "$c" || echo "[run_local] $c failed; continuing"
    done
else
    run_one "$1"
fi
