# LoCoMo on a fresh machine — Quickstart

Run the LoCoMo benchmark sweep (14 substrates × 10 conversations) on any
Linux host with a CUDA GPU. Builds on `scripts/setup_env.sh` for the env
and `scripts/sweep.py` for batch submission (or `scripts/run_local.sh`
for non-SLURM runs).

---

## 1. Clone + setup

```bash
git clone https://github.com/Jasonya/SubMem.git
cd SubMem

# 1a. install conda env (one-shot; clones third_party/{memit, a-mem})
bash scripts/setup_env.sh                # creates env "submem"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate submem

# 1b. .env file with API keys
cat > .env <<EOF
OPENAI_API_KEY=sk-...
SUBMEM_REAL_OPENAI_KEY=sk-...           # same as OPENAI_API_KEY; consumed by C5
SUBMEM_REAL_OPENAI_BASE=https://api.openai.com/v1
EOF
```

`OPENAI_API_KEY` is needed by C3/C4/C5/C6/C7/C13/C15/C16 (auxiliary LLM),
the P4 LLM-judge, and the embedding service used by C1/C5/C6/C7.

---

## 2. Pick a runner mode

### Option A — SLURM (multi-node cluster, recommended)

```bash
# Defaults: account from $SLURM_ACCOUNT, gpuA40x4 partition, 4h time, 200G RAM,
# 1 GPU per job. Override with --account / --partition / --time / --mem.
python3 scripts/sweep.py \
    --benchmark locomo \
    --memory C1 C2 C3-Letta C4 C5 C6 C7 C9 C10 C11 C12 C13 C15 C16 \
    --max-samples 10 --top-k 10 \
    --gen-model Qwen/Qwen3-8B \
    --mem 200G --time 04:00:00 \
    --account <your-account> --partition <your-gpu-partition>
```

This emits 14 sbatch scripts under `/tmp/submem_sweep_scripts/` and
submits each one. Logs land in `sweep_logs/<config>_<jobid>.log`.
Track:

```bash
squeue -u $USER -n submem-sweep
```

### Option B — local (single host, no SLURM)

```bash
# Runs one substrate at a time on local GPU. ~5 min - 5 h per config.
bash scripts/run_local.sh C1                     # one substrate
bash scripts/run_local.sh all                    # all 14
# Or invoke the runner directly:
python3 runners/run_c1c2_test.py \
    --memory C1 --model Qwen/Qwen3-8B \
    --max-samples 10 --top-k 10 \
    --aux-model openai/gpt-4o-mini \
    --encoder-model text-embedding-3-small \
    --dataset locomo --device-map auto --fast
```

---

## 3. top_k sweep (optional)

For C1, C2, C6, C7, C13 — the substrates whose `read()` honours `top_k`:

```bash
python3 scripts/sweep.py \
    --benchmark locomo \
    --memory C1 C2 C6 C7 C13 \
    --max-samples 10 --top-k 1 2 5 10 20 \
    --gen-model Qwen/Qwen3-8B \
    --mem 200G --time 04:00:00 \
    --account <your-account> --partition <your-partition>
```

The cache (see §5) means write-phase work is reused across `k` values
for substrates with `save_state` (everyone except C6).

---

## 4. Result layout

Per run dir under `results/<ts>_locomo_<model>_pid<PID>/<Cx>/`:

```
01_data_summary.json
02_writes_all.json
02_writes_sample<i>.json       # i = 0..9, one per conversation
03_retrievals.json
04_prompts.json
05_predictions.json
06_memory_snapshot_sample<i>.json
06_run_meta.json
07_metrics.json                # P + E + Q + R + D family aggregates
08_llm_judge.json              # P4 per-query
09_faithfulness.json           # P5 per-query (NLI)
09_raw_data.json               # flat per-query dump for aggregation
```

Build the cross-substrate main table:

```bash
python3 scripts/aggregate.py \
    --glob 'results/*locomo*/C*/09_raw_data.json'
```

---

## 5. Cache (per-conversation, write-once-read-many)

Cache root: `~/.cache/submem/cache/locomo/`
- `CACHE_INDEX.jsonl` — append-only JSONL, one line per store/hit
- `by_hash/<sha16>/` — per-conv saved memory state (load via
  `BaseMemoryUnit.load_state(dir, config, model)`)

Cache key = SHA256 of:
- benchmark, substrate, sample_id
- encoder + dim, aux_model
- gen_model_at_write (only for substrates with
  `REQUIRES_GEN_MODEL_IN_KEY = True`)
- whitelisted hyperparams (`CACHE_KEY_EXTRA`)
- substrate `.py` file SHA → editing the source invalidates all of that
  substrate's cached convs

If a job times out mid-sweep, resubmitting the same command will HIT
already-completed convs and only rerun the missing ones.

**Note on C6 Graphiti**: cache is intentionally disabled (Kuzu DB
live-file copy interacts poorly with `graphiti_core`'s connection pool —
see `memory/c6_graphiti.py:564`). Each C6 run rebuilds the graph from
scratch (~2h write per conv). Use `--time 08:00:00` for `top_k=20`.

---

## 6. Knobs you'll probably touch

| Env var | Default | What |
|---|---|---|
| `MEM0_DIR` | `~/.cache/submem/mem0_db` | C5 mem0 root (qdrant + history.db) |
| `MEM0_VECTOR_STORE` | `qdrant` | C5: switch to `chroma` if mem0 hits qdrant UPDATE/DELETE shape-mismatch (200+ writes) |
| `C3_AGENT_EVERY_N` | `3` | C3 Letta: run Qwen tool-call loop every N turns (1=paper, 3=~3x faster) |
| `C6_BUFFER_POOL_GB` | `32` | C6: kuzu mmap cap (mmap virtual blew SLURM cgroup at default 0) |
| `C6_BFS_DEPTH` | `1` | C6 search BFS depth (recipe default 3 OOMs on 419-episode graphs) |
| `C6_CONCURRENCY` | `8` | C6: parallel `add_episode` workers (write-time speedup) |
| `SUBMEM_CACHE` | `~/.cache/submem/cache` | Cache root override |
| `SUBMEM_RESULTS` | `<repo>/results` | Result root override |

---

## 7. Quirks & known issues

1. **C6 cache disabled** — see `memory/c6_graphiti.py:564`. Will rerun
   write phase every time. Budget 2-5h per conv depending on `top_k`.
2. **C9 LoRA `substrate_src_sha`** — editing `c9_lora.py` invalidates
   all 10 cached adapters. Workaround in `scripts/c9_cache_remap.py`
   (compute new hash, copy old dir).
3. **C5 mem0 qdrant path** — fixed in `mem0_eval/src/memzero/add.py` to
   route through `$MEM0_DIR/qdrant` (was `/tmp/qdrant` and got clobbered
   by every other concurrent job).
4. **P5 faithfulness short-answer fix** — `runners/run_c1c2_test.py` rebuilds
   claims as `f"{question}: {answer}"` when the answer is < 5 words,
   because DeBERTa NLI labels bare entities as `neutral` regardless of
   premise. Affects all P5 numbers reported on QA benchmarks.
5. **E-metrics empty when run errors mid-inference** — always check
   `efficiency` is non-empty before trusting E1/E5/E14.
6. **C8 dropped** — MEMIT incompatible with Qwen3 layer geometry.

---

## 8. Moving results to scratch

Default `SUBMEM_RESULTS=<repo>/results`. On a cluster with limited home
quota:

```bash
# $SCRATCH = your cluster's scratch filesystem
mkdir -p "$SCRATCH/SubMem-results"
ln -s "$SCRATCH/SubMem-results" <repo>/results
echo "results/" >> .gitignore   # already gitignored, but doesn't hurt
```

Or set `SUBMEM_RESULTS=$SCRATCH/SubMem-results` in `.env`.

The cache (`~/.cache/submem/cache/`) can also grow to 2-3 GB per
substrate (C9 has 15 MB × 10 conv adapters; C6 graphs are 100 MB each).
Symlink it out the same way if `~/.cache` is on a small partition.
