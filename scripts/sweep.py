"""Sweep submitter: submits SLURM jobs across (substrate, sample, top_k, gen_model).

The harness handles caching internally — cache hits skip the write phase
automatically. This wrapper just generates the cross-product and submits.

Usage:
    python scripts/sweep.py \\
        --benchmark locomo \\
        --memory C1 C3-Letta \\
        --max-samples 10 \\
        --top-k 5 10 20 \\
        --gen-model Qwen/Qwen3-8B \\
        --aux-model openai/gpt-4o-mini \\
        --encoder text-embedding-3-small \\
        --account <your-allocation>

Paths: the repo root is $SUBMEM_ROOT (default: the checkout containing this
file), logs go to --log-dir (default $SUBMEM_ROOT/sweep_logs), and --account
defaults to $SLURM_ACCOUNT.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Repo root: $SUBMEM_ROOT, else derived from this file's location (scripts/..).
SUBMEM_ROOT = Path(os.environ.get("SUBMEM_ROOT", Path(__file__).resolve().parents[1]))


_TPL = """#!/bin/bash
#SBATCH --job-name=submem-sweep-{tag}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log}

SUBMEM_ROOT={submem_root}
cd "$SUBMEM_ROOT"
source "${{CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}}"
source "$SUBMEM_ROOT/scripts/common.sh"

# Isolate per-job mem0 dir to avoid Qdrant lock collisions when running
# multiple C5 jobs concurrently. Default MEM0_DIR from common.sh is shared.
export MEM0_DIR="$SUBMEM_RESULTS/mem0_db_job${{SLURM_JOB_ID}}"
mkdir -p "$MEM0_DIR"

# C6 Graphiti: cap Kuzu buffer pool to prevent mmap-virtual explosion under SLURM cgroup.
export C6_BUFFER_POOL_GB="${{C6_BUFFER_POOL_GB:-32}}"

# C5 mem0: ChromaDB tolerates UPDATE/DELETE churn (qdrant-local has
# payload_mask vs deleted-mask shape-mismatch after ~200 ops).
export MEM0_VECTOR_STORE="${{MEM0_VECTOR_STORE:-chromadb}}"

# C3 Letta: run Qwen agent every N turns (skipped turns go straight to archival
# insert). N=3 → ~3x write speedup. Paper-default is 1.
export C3_AGENT_EVERY_N="${{C3_AGENT_EVERY_N:-3}}"

python3 runners/run_c1c2_test.py \\
    --memory {memory} \\
    --model {gen_model} \\
    --max-samples {max_samples} \\
    --top-k {top_k} \\
    --aux-model {aux_model} \\
    --encoder-model {encoder} \\
    --dataset {benchmark} \\
    --device-map auto \\
    --fast 2>&1
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="locomo")
    ap.add_argument("--memory", nargs="+", required=True)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--top-k", type=int, nargs="+", default=[10])
    ap.add_argument("--gen-model", nargs="+", default=["Qwen/Qwen3-8B"])
    ap.add_argument("--aux-model", default="openai/gpt-4o-mini")
    ap.add_argument("--encoder", default="text-embedding-3-small")
    ap.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT", "<your-allocation>"))
    ap.add_argument("--partition", default="gpuA100x4")
    ap.add_argument("--cpus", default="8")
    ap.add_argument("--gpus", default="1")
    ap.add_argument("--mem", default="96G")
    ap.add_argument("--time", default="04:00:00")
    ap.add_argument("--log-dir", default=str(SUBMEM_ROOT / "sweep_logs"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path("/tmp/submem_sweep_scripts"); scripts_dir.mkdir(parents=True, exist_ok=True)

    submitted = []
    for cfg in args.memory:
        for k in args.top_k:
            for gen in args.gen_model:
                tag = f"{cfg.replace('/','_')}_k{k}_{gen.replace('/','_')[:20]}"
                script_path = scripts_dir / f"{tag}.sh"
                log_path = log_dir / f"{tag}_%j.log"
                content = _TPL.format(
                    tag=tag, account=args.account, partition=args.partition,
                    cpus=args.cpus, gpus=args.gpus, mem=args.mem, time=args.time,
                    log=str(log_path),
                    submem_root=shlex.quote(str(SUBMEM_ROOT)),
                    memory=cfg, gen_model=shlex.quote(gen),
                    max_samples=args.max_samples, top_k=k,
                    aux_model=shlex.quote(args.aux_model),
                    encoder=shlex.quote(args.encoder),
                    benchmark=args.benchmark,
                )
                script_path.write_text(content)
                cmd = ["sbatch", str(script_path)]
                if args.dry_run:
                    print(f"[dry] {' '.join(cmd)}\n  → {script_path}")
                    continue
                out = subprocess.run(cmd, capture_output=True, text=True)
                jid = out.stdout.strip().split()[-1] if "Submitted" in out.stdout else "FAIL"
                print(f"  {tag} → job {jid}")
                submitted.append(jid)

    if not args.dry_run:
        print(f"\nSubmitted {len(submitted)} jobs.")
        print("Track: squeue -u $USER")
        print(f"Aggregate after: python scripts/aggregate.py --glob 'results/*locomo*/C*/09_raw_data.json'")


if __name__ == "__main__":
    main()
