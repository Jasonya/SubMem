#!/usr/bin/env bash
# Set up the MemoryArena evaluation from a fresh clone of this repository.
#
#   ./setup.sh [target-dir]
#
# Clones the benchmark, applies our integration patch, installs the adapter and
# the run configurations, and checks that the substrates import. Nothing outside
# the target directory is touched.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMEM_ROOT="$(cd "$HERE/../.." && pwd)"
TARGET="${1:-$HERE/third_party/memoryarena}"
REPO="${MEMORYARENA_REPO:-https://github.com/Jasonya/ACRE.git}"

echo "SubMem root : $SUBMEM_ROOT"
echo "Benchmark   : $TARGET"

if [ -d "$TARGET/.git" ]; then
  echo "== benchmark already present, skipping clone"
else
  echo "== cloning $REPO"
  git clone --depth 1 "$REPO" "$TARGET"
fi

echo "== applying integration patch"
if git -C "$TARGET" apply --check "$HERE/patch/submem-integration.patch" 2>/dev/null; then
  git -C "$TARGET" apply "$HERE/patch/submem-integration.patch"
  echo "   applied"
else
  echo "   patch does not apply cleanly; the checkout may already contain it."
  echo "   Inspect $HERE/patch/submem-integration.patch and merge by hand if needed."
fi

echo "== installing adapter and configs"
install -d "$TARGET/memory/memory_systems" \
           "$TARGET/configs/formal_reasoning_configs" \
           "$TARGET/configs/travel_planner_configs"
cp "$HERE/adapter/submem_adapter.py" "$TARGET/memory/memory_systems/"
cp "$HERE/configs/math/"*.json   "$TARGET/configs/formal_reasoning_configs/"
cp "$HERE/configs/travel/"*.json "$TARGET/configs/travel_planner_configs/"

echo "== checking that the substrates import"
SUBMEM_PARENT="$(dirname "$SUBMEM_ROOT")" \
PYTHONPATH="$(dirname "$SUBMEM_ROOT"):$TARGET" \
python - <<'PY'
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location(
    "submem_adapter",
    os.path.join(os.environ["PYTHONPATH"].split(os.pathsep)[1],
                 "memory", "memory_systems", "submem_adapter.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert hasattr(m, "make_submem_system")
print("   adapter imports, make_submem_system available")
PY

cat <<EOF

Done. To run:

  export MEMORYARENA_ROOT="$TARGET"
  export SUBMEM_ROOT="$SUBMEM_ROOT"
  export SUBMEM_PARENT="$(dirname "$SUBMEM_ROOT")"
  export OPENAI_API_KEY=...          # auxiliary LLM, embeddings, judge
  export CONDA_ENV=submem            # conda env the job scripts activate (default: submem)
  export CONDA_SH=...                # conda.sh to source (default: \$HOME/miniforge3/etc/profile.d/conda.sh)

  cd "\$MEMORYARENA_ROOT" && mkdir -p results   # job logs: results/memoryarena_<jobname>_<jobid>.log
  sbatch $HERE/slurm/run_ma_8b.slurm                       # formal reasoning
  sbatch --export=ALL,TRAVEL_SUBS="submem_m1 submem_m2" \\
         $HERE/slurm/run_ma_8b_kbfull.slurm                # group travel

Scores from our runs are in $HERE/results/ for comparison.
EOF
