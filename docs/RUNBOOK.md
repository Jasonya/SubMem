# SubMem LoCoMo Benchmark — Runbook

Last updated: 2026-04-17

This doc covers (a) the conda env layout, (b) how to launch each of the 15 memory configs on LoCoMo via SLURM, (c) why we run HF (not vLLM), and (d) the current 1-conversation result table.

**Porting to a fresh machine?** Jump to §7 — a clean clone now works on any Linux host via `scripts/setup_env.sh` + `scripts/run_local.sh`, without SLURM or any cluster-specific paths.

---

## 1. Paths

All locations are driven by environment variables (see §2.2); the defaults are relative to the checkout.

| What | Path |
|---|---|
| Source tree | `$SUBMEM_ROOT` (your SubMem checkout) |
| HF model cache | `$HF_HOME` (default `$HOME/.cache/huggingface`) |
| Result root | `$SUBMEM_RESULTS` (default `$SUBMEM_ROOT/results`) |
| SLURM logs | the relative `#SBATCH --output=` path in each script, under the submit directory (§7.3) |
| mem0 SQLite (C5) | `$MEM0_DIR` (default set by `scripts/common.sh`) |

Each run writes to `$SUBMEM_RESULTS/<TIMESTAMP>_locomo_<MODEL>/<CONFIG>/` with:
`01_data_summary.json`, `02_writes_*.json`, `03_retrievals.json`, `04_prompts.json`,
`05_predictions.json`, `06_memory_snapshot_*.json`, `06_run_meta.json`,
`07_metrics.json`, `08_llm_judge.json`.

---

## 2. Environment

### 2.1 Conda env

All configs share one env, created by `scripts/setup_env.sh` (§7.1) and selected with `CONDA_ENV` (a name or a prefix path; default `submem`):

```bash
conda activate "${CONDA_ENV:-submem}"
```

Already contains every third-party dep the substrates import:

- `transformers`, `torch`, `accelerate`, `peft` — HF backend (C1/C3/C8/C9/C10/C11/C12/C13/C15)
- `faiss-cpu` — dense retrieval (C1/C3/C7/C13/C15)
- `rank_bm25` — sparse retrieval (C2)
- `agentic_memory` (editable, from agiresearch/A-mem) — C16
- `graphiti_core` (editable, 0.28.2) — C6
- `chromadb` 1.5.5 — C16 vector store
- `mem0ai` 2.0.0 — C5 (note: v2 moved `user_id` into `filters={...}`)
- `openai` — C3/C5/C6/C7/C13/C16 (management LLM = gpt-4o-mini)

**Do not `pip install -r requirements.txt` from scratch** — the list is a minimal seed; the shared env is the source of truth. If you must rebuild, also install:
`mem0ai==2.0.0 graphiti-core==0.28.2 chromadb==1.5.5 peft openai tiktoken jinja2 bitsandbytes`
plus editable `/tmp/A-mem` and `/tmp/ml-epicache` clones.

### 2.2 Required env vars

```bash
export SUBMEM_ROOT=<path/to/SubMem>   # the checkout; SLURM scripts run from a spool dir and need this
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"
export OPENAI_API_KEY=...          # required for C3/C5/C6/C7/C13/C16 management LLM
export LETTA_MODEL=gpt-4o-mini     # optional; C3 agent LLM override
export MEM0_DIR=<path/to/mem0_db>  # C5 only; optional (scripts/common.sh sets a default)
```

`.env` at repo root is auto-loaded by C3 (`memory/c3_letta.py:_ENV_FILE`).

### 2.3 Why HF, not vLLM

vLLM 0.x cannot serve Qwen3.5-9B on our cluster:

- Qwen3.5-9B uses GDN (gated delta-net / mamba-style) layers; vLLM's mamba kernel JIT-compiles via `nvcc`, which is not installed on compute nodes.
- Workaround: **use `Qwen/Qwen3-8B`** (pure attention, full compat) and HF `model.generate` for all configs.
- For HF-only configs (C8–C12) we apply `k=2` calibration at batch boundaries so token accounting aligns with the vLLM-based configs elsewhere.

This choice is recorded in memory; do not retry vLLM for Qwen3.5-9B without first installing nvcc on the worker image.

---

## 3. Launching runs

All one-conversation (≈419 turns / 199 queries) SLURM scripts sit at the repo root. Submit with `sbatch <script>`.

### 3.1 Per-config scripts (all use Qwen3-8B + HF)

| Cfg | Script | Est. wall | Notes |
|---|---|---:|---|
| C1 Dense / C2 Sparse | `run_c1c2_1conv.sh` | 30 min | `--memory C1 C2` in one job |
| C3 Letta | `run_c3_1conv.sh` | 55 min | Standalone MemGPT; uses OpenAI gpt-4o-mini as agent LLM |
| C4 PageSum | `run_c4_1conv.sh` | 8 min | ReadAgent-style gist summaries |
| C5 Mem0 | `run_c5_1conv.sh` | 18 min | Goes through `runners/run_c5.py` (separate eval harness) |
| C6 Graphiti | `run_c6_1conv.sh` | 45 min | Cloned graphiti-core; temporal KG |
| C7 RAPTOR | `run_c7_1conv.sh` | 12 min | Hierarchical summary tree |
| C8 MEMIT | `run_c8_1conv.sh` | 2 min | Parametric; uses EasyEdit |
| C9 LoRA | `run_c9_1conv.sh` | 12 min | rank=8, q/k/v/o only |
| C10 FullCtx | `run_c10_1conv.sh` | 5 min | Full conversation in context |
| C11 SnapKV | `run_c11_1conv.sh` | 6 min | window=5, avgpool |
| C12 EpiCache | `run_c12_1conv.sh` | 2 min | n_medoid=8, power=1.3 |
| C13 Sleep | `run_c13_1conv.sh` | 17 min | Chunked rethinking (v0 revert) |
| C15 MemSkill | `run_c15_1conv.sh` | 21 min | |
| C16 A-Mem | `run_c16_1conv.sh` | 38 min | Official AgenticMemorySystem wrapper |

C14 is **not run on LoCoMo** — ReasoningBank is agentic-only (SWE-bench / AlfWorld). `write(text)` raises `NotImplementedError` for dialog inputs.

### 3.2 Common SLURM block

Every script uses:

```
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=<bad-node>  # optional: skip a node you know is faulty
```

Time budget varies per config (set `#SBATCH --time=...`). Log path: `slurm_<name>_%j.log`.

### 3.3 Entry points

- `runners/run_c1c2_test.py` — the main harness. Flags:
  `--memory <C1 C2 ...>`, `--model Qwen/Qwen3-8B`, `--max-samples N`,
  `--top-k K` (default 10), `--device-map auto`, `--aux-model openai/gpt-4o-mini`,
  `--fast` (disables unnecessary LLM re-evals).
- `runners/run_c5.py` — C5 only, because mem0 has its own eval loop.

### 3.4 Parallel full-batch

To submit 13 configs in one shot (C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C15, C16):

```bash
for s in run_c3_1conv.sh run_c4_1conv.sh run_c5_1conv.sh run_c6_1conv.sh \
         run_c7_1conv.sh run_c8_1conv.sh run_c9_1conv.sh run_c10_1conv.sh \
         run_c11_1conv.sh run_c12_1conv.sh run_c13_1conv.sh run_c15_1conv.sh \
         run_c16_1conv.sh; do
  sbatch "$s"
done
```

Monitor with `squeue -u $USER`.

---

## 4. Metrics

Recorded in `07_metrics.json` under `task_performance / robustness / efficiency / quality / dynamics`.

- **P1 EM / P2 F1** — exact-match + token F1 (rule-based, no LLM)
- **P3 BLEU-1** — n-gram overlap (rule-based)
- **P4 LLM-Judge** — gpt-4o-mini ranks correctness. **Only P-class metric allowed to call LLM.**
- **P5 ClaimDecomp** — rule-based decomposition (replaced the previous LLM path)
- **P6 Recall@k / P7 Precision@k** — dia_id overlap between retrieved entries and gold evidence
- **R2 DRR** — adversarial distractor rejection rate
- **E1–E15** — storage / latency / token / call counts (split construct/retrieval/mgmt)
- **Q1 Compression** — size ratio vs raw transcript
- **Q2 Coverage/Diversity** — FAISS-reused embeddings, no re-encode
- **D1 Recency** — correlation of retrieved entry age vs rank

---

## 5. Known issues / caveats

1. **C3 Letta dia_id back-matching** (fixed 2026-04-17, job 88544).
   Agent composes archival inserts late-bound, so `_current_write_dia_id` stamped
   the wrong turn (e.g. an LGBTQ fact from D1:3 got tagged D1:18 because the
   agent emitted the insert during Melanie's later goodbye turn).
   Fixed by maintaining a source-turn log `(text, dia_id, emb)` populated at
   every `write()`, and re-attributing each `archival_memory_insert` via
   cosine similarity (threshold 0.45, top-3). Archival entries now carry a
   list `dia_ids` instead of a scalar `dia_id`. Recall@k 0.018 → 0.638.

2. **C5 Mem0 hallucinated IDs** — mem0 sometimes emits UPDATE for IDs it
   never inserted. We tried patching to coerce UPDATE→ADD but it *hurt* F1
   (added noise). Accepted as library behavior.

3. **C8 deployment delta size** — `memory_size_bytes` reports info-theoretic
   triple bytes (~3 KB), not the 805 MB dense delta. The latter is still
   available as `deployment_delta_bytes` in stats.

4. **C13 latency split** — per-chunk write time is token-weighted (not
   evenly split) so mean/p50/p90 are non-degenerate.

5. **C14 excluded from LoCoMo** — agentic-only; the paper does not
   evaluate on dialog. `write()` raises `NotImplementedError`.

6. **Only P4 may call an LLM.** P5 uses rule-based claim decomposition.
   Q2 reuses write-time FAISS embeddings. No other P-metric invokes an LLM.

7. **Retrieval top-k now defaults to each substrate's paper default.**
   The harness argparse `--top-k` is `None` unless set. When unset, each
   `memory_unit.read()` uses its own method-level default (the one the
   paper actually evaluates with): C1=10, C2/C3/C4/C10-C13/C15/C16=5,
   C14=1 (ReasoningBank paper default). Set `TOPK=10` in the environment
   to force a single k across configs for apples-to-apples comparisons;
   this used to be the hard-wired default and silently overrode C14's
   paper default of 1.

8. **C11 SnapKV / C12 EpiCache are utterance-granularity adaptations,
   not paper-faithful reproductions.** The published methods perform
   per-head (SnapKV) or per-layer (EpiCache) KV-token eviction using
   monkey-patched attention forwards that require `flash_attn` (needs an
   `nvcc` toolchain which our compute nodes do not have) and have only
   been upstreamed for Llama / Mistral / Qwen2.5 — Qwen3's hybrid
   architecture (attention + DeltaNet layers) is not supported. Our
   substrates keep the paper's scoring logic (observation-window
   attention for C11, episodic medoid attention for C12) but select
   whole utterances and re-prefill them, so E2 reflects re-prefill cost
   rather than the paper's decode speedup. Numbers on LoCoMo should be
   interpreted as "attention-guided utterance selection with
   SnapKV-/EpiCache-style scoring," not as a reproduction of the
   underlying KV-cache-compression claim.

9. **C8 MEMIT first run precomputes Wikipedia covariance (~1 hour).**
   We now call the paper's `apply_memit_to_model` from the vendored
   `third_party/memit/memit/`, which lazy-computes `C₀ = E[k·kᵀ]` the
   first time `get_cov()` is invoked. The stats file lands under
   `$SUBMEM_RESULTS/memit_stats/` and is reused by every subsequent
   C8 job. No manual pre-computation step is required — just expect
   the first `sbatch run_c8_1conv.sh` on a fresh machine to take an
   extra hour.

9. **C9 LoRA E14/E15 include amortized training.**
   C9's `write()` only buffers; adapter fit happens lazily during the
   inference stage (LoRA epoch × 1125 QA pairs ≈ 1000s on A100). The
   harness measures `t_total = write_time + inference_time` and reports
   E14 from that. E15 = E14 / n_queries therefore bundles ~1000s of
   training into the "per-query" figure. For LoCoMo runs at
   n_queries=199 this inflates E15 to ~5-6s when actual per-query
   inference latency (E2) is <500ms. Do not compare E15 across
   parametric (C8/C9) and non-parametric configs directly — use E2
   for inference-only cost.

---

## 6. Reproducing C3's dia_id fix verification

```bash
cd "$SUBMEM_ROOT"                 # your SubMem checkout
sbatch run_c3_1conv.sh            # ~55 min on gpu-h200
# After completion:
python3 - <<'PY'
import json, glob, os
results = os.environ.get("SUBMEM_RESULTS", "results")
latest = sorted(glob.glob(f"{results}/*_locomo_Qwen3-8B/C3-Letta/07_metrics.json"))[-1]
d = json.load(open(latest))
print("P6 Recall@k:", d["task_performance"]["P6_recall_at_k"])
print("P7 Precision@k:", d["task_performance"]["P7_precision_at_k"])
PY
```

Expect Recall ≈ 0.638, Precision ≈ 0.046.

---

## 7. Porting to a fresh machine

Hard-coded cluster paths have been removed from all `run_c*_1conv.sh`. A single dispatcher at `scripts/common.sh` sets defaults from env vars (`SUBMEM_ROOT`, `HF_HOME`, `SUBMEM_RESULTS`, `MEM0_DIR`, `CONDA_ENV`), falling back to local paths under the checkout. You can therefore `git clone` on any Linux host with CUDA and run.

### 7.1 One-time setup

```bash
git clone <repo-url> SubMem && cd SubMem
# Requires conda (miniforge/miniconda).
bash scripts/setup_env.sh              # creates env "submem", clones A-mem, installs editable
conda activate submem

# Put your keys in a .env at repo root (auto-loaded by scripts/common.sh):
cat > .env <<EOF
OPENAI_API_KEY=sk-...
EOF
```

Optional overrides (any can go into `.env` too):

```bash
export HF_HOME=$HOME/.cache/huggingface     # default
export SUBMEM_RESULTS=./results             # default: <repo>/results
export MEM0_DIR=./results/mem0_db
export MODEL=Qwen/Qwen3-8B                  # default
export TOPK=10                              # default
export MAX_SAMPLES=1                        # default
```

### 7.2 Running without SLURM (local)

```bash
# Single config:
bash scripts/run_local.sh C3-Letta
# All (skips C14, which is agentic-only):
bash scripts/run_local.sh all
```

Each call shells into `run_c<N>_1conv.sh`, which now just sources `scripts/common.sh` and invokes the harness. The `#SBATCH` directives at the top of each script are ignored when run under plain `bash`.

### 7.3 Running with SLURM (same scripts)

`sbatch run_c3_1conv.sh` still works unchanged. `#SBATCH --output=` now uses a relative path (`slurm_c3_1conv_%j.log`), landing in the submit directory.

### 7.4 LoCoMo test data

The harness looks for `locomo_test.json`; if not found under `$SUBMEM_RESULTS`, download from the [LoCoMo release](https://github.com/snap-research/locomo) and place in the repo root (or set `LOCOMO_TEST=/path/to/locomo_test.json`).

### 7.5 GPU requirements

| Config | VRAM | Notes |
|---|---|---|
| C1, C2, C7, C15, C16 | ~20 GB | Qwen3-8B inference only |
| C3, C4, C5, C6, C13 | ~20 GB + OpenAI API | mgmt LLM runs on gpt-4o-mini |
| C8 MEMIT | 24 GB | EasyEdit weight delta |
| C9 LoRA | 32 GB | training pass over 8B base |
| C10, C11, C12 | ~24 GB | full-context / KV eviction |

A single H100/H200 (80 GB) runs everything. On a 24 GB card, skip C9 or use `--device-map auto` for offloading (already default on C3/C7/C8/C9/C10/C11/C12/C13).

### 7.6 File layout after setup

```
SubMem/
├── environment.yml
├── .env                      # you create this
├── scripts/
│   ├── common.sh             # sourced by every run_c*_1conv.sh
│   ├── setup_env.sh          # one-shot bootstrap
│   └── run_local.sh          # non-SLURM runner
├── third_party/
│   └── A-mem/                # cloned by setup_env.sh
├── run_c1c2_1conv.sh         # thin SBATCH wrappers
├── run_c3_1conv.sh
├── ...
├── memory/                   # C1–C16 substrates
├── metrics/                  # P/R/E/Q/D evaluators
├── runners/run_c1c2_test.py          # main harness
└── runners/run_c5.py                 # C5 separate entry
```
