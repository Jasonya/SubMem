# SubMem: Quick Start Guide

## What is SubMem?

SubMem benchmarks **12 memory substrates** (C1-C12) for LLM conversational agents. Each substrate is a different way to store, manage, and retrieve long-term conversation history. We evaluate on **LoCoMo** (10 multi-session conversations, 5 query types) using **27 metrics** across 6 families.

---

## Memory Substrates Overview

| C | Name | Reference | Write | Read | Management |
|---|------|-----------|-------|------|------------|
| **C1** | Dense Vector | Contriever (Izacard+, TMLR'22) | Embed utterance → FAISS add | Cosine ANN top-k | --- |
| **C2** | Sparse Vector | BM25 (Robertson+, '94) | Tokenize → BM25 corpus | BM25Okapi scoring | Lazy re-index |
| **C3** | Raw Buffer | MemGPT (Packer+, NeurIPS'23) | FIFO buffer; evict 50% on overflow | LLM-routed archival + recall + core memory | LLM core memory update |
| **C4** | Summarized Gist Index | ReadAgent (Lee+, ICML'24) | ~500-word pages → LLM gist → embed | LLM selects pages to expand | Text dedup |
| **C5** | Summarized Atomic Fact | Mem0 (Chhikara+, '25) | Per-pair LLM fact extraction | Dense retrieval over non-deleted facts | LLM classify: ADD/UPDATE/DELETE |
| **C6** | Temporal Knowledge Graph | Graphiti/Zep (Raghavan+, '25) | Entity + edge extraction + conflict resolution | BM25 + cosine + BFS → RRF | Entity dedup; temporal invalidation |
| **C7** | Hierarchical Store | H-MEM (Sun+, EACL'26) | LLM 4-level extraction | Top-down cosine filtering | Top-level dedup; optional decay |
| **C8** | Knowledge Editing | MEMIT (Meng+, ICLR'23) | LLM triple extraction → closed-form ΔW | Direct generation (no retrieval) | --- |
| **C9** | LoRA Adapter | LoRA (Hu+, ICLR'22) | LLM QA gen → LoRA training (r=64, α=128) | Direct generation on base+adapter | --- |
| **C10** | Full KV Cache | Baseline | Append to buffer | Concatenate; truncate to window | Truncate at limit |
| **C11** | Query-Filtered KV Cache | SnapKV (Li+, NeurIPS'24) | Append to buffer | Blockwise attention → select 20% → re-prefill | Recompute per query |
| **C12** | Episode-Clustered KV Cache | EpiCache (Kim+, '25) | Segment → K-means → per-episode attention select 25% | Match query to episode centroid → re-prefill | Episodic clustering; budget allocation |

---

## Metrics (6 families, 24 metrics)

| Family | ID | Metric | What it measures |
|--------|-----|--------|-----------------|
| **Performance** | P1 | Exact Match | Strict string match |
| | P2 | Token F1 | Token overlap |
| | P3 | BLEU-1 | Unigram precision + brevity penalty |
| | P4 | LLM Judge | LLM correctness rating (1-5 → [0,1]) |
| | P5 | Faithfulness | NLI entailment of claims by context |
| | P6 | Recall@k | Evidence coverage |
| | P7 | Precision@k | Retrieved relevance |
| **Robustness** | R1 | Contradiction Resolution | Answer uses latest version of updated facts |
| | R2 | Adversarial Rejection | Rejects injected distractors |
| **Efficiency** | E1-E11 | Size, latency, tokens, LLM calls | 11 fine-grained cost metrics |
| **Quality** | Q1 | Compression Ratio | Input/stored tokens |
| | Q2 | Coverage Diversity | Topical diversity of stored entries (k=5) |
| **Scaling** | S1 | Session Retention | F1 degradation as sessions grow |
| | S2 | Capacity Cliff | Utilization threshold for accuracy collapse |
| **Dynamics** | D1 | Recency Bias | EM gap between recent vs early evidence |

---

## How to Run

### Prerequisites

```bash
# Repository checkout (the Slurm scripts need this: they run from a spool directory)
export SUBMEM_ROOT=<path/to/SubMem>

# Conda environment (created by scripts/setup_env.sh; see docs/RUNBOOK.md §8)
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"

# HuggingFace cache (default $HOME/.cache/huggingface; point it at a shared model cache if you have one)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Results directory (optional, defaults to $SUBMEM_ROOT/results)
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"

# API key for C3-C7 aux-model and P4 LLM-Judge (optional)
export OPENAI_API_KEY=sk-...
```

#### Native library for M5 (LightRAG)

M5 links against a newer `libstdc++` than the one on many cluster images, and
without it the substrate fails at import with

```
ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

Preload the copy that ships with the conda environment. The Slurm scripts
already do this; an interactive shell has to set it too:

```bash
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
```

The other ten substrates import without it, so this failure looks specific to
M5 rather than to the environment, which is what makes it hard to place.

### Quick Local Test

```bash
cd "$SUBMEM_ROOT"   # your SubMem checkout

# Fast run: C1+C2 on 1 conversation, skip expensive metrics
python3 runners/run_c1c2_test.py \
    --memory C1 C2 \
    --model Qwen/Qwen3.5-9B \
    --max-samples 1 \
    --fast

# External memory with API aux-model
python3 runners/run_c1c2_test.py \
    --memory C3 C4 C5 C6 C7 \
    --model Qwen/Qwen3.5-9B \
    --aux-model openai/gpt-4o-mini \
    --max-samples 1 \
    --fast

# Parametric + KV Cache
python3 runners/run_c1c2_test.py \
    --memory C8 C9 C10 C11 C12 \
    --model Qwen/Qwen3.5-9B \
    --max-samples 1 \
    --fast
```

### Full Run via Slurm

```bash
# All external memory (C1-C7) on 2 conversations
sbatch run_c1c7.sh

# All parametric + KV (C8-C12)
sbatch run_c8c9.sh
sbatch run_c10c12.sh

# Full Qwen3-8B comparison
sbatch run_all_qwen3.sh
```

Example Slurm script:
```bash
#!/bin/bash
#SBATCH --job-name=submem
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/submem_%x_%j.log

# Export SUBMEM_ROOT before sbatch: Slurm copies this script to a spool
# directory, so the checkout cannot be derived from $0.
cd "${SUBMEM_ROOT:?set SUBMEM_ROOT to the SubMem checkout}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export SUBMEM_RESULTS="${SUBMEM_RESULTS:-$SUBMEM_ROOT/results}"

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-submem}"

python3 runners/run_c1c2_test.py \
    --memory C1 C2 C3 C4 C5 C6 C7 \
    --model Qwen/Qwen3.5-9B \
    --aux-model openai/gpt-4o-mini \
    --max-samples 2 \
    --fast 2>&1
```

### Post-hoc Evaluation

```bash
# Re-evaluate existing predictions (add P3 BLEU-1, P4 LLM-Judge, P5 Faithfulness)
python3 runners/run_c1c2_test.py \
    --memory C1 \
    --eval-only \
    --run-dir $SUBMEM_RESULTS/20260407_220823_locomo_Qwen3.5-9B

```

### CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--memory C1 C2 ...` | Substrates to run | required |
| `--model NAME` | Base model for generation | `Qwen/Qwen3.5-0.8B` |
| `--aux-model NAME` | API model for write-phase LLM calls | None |
| `--max-samples N` | Number of LoCoMo conversations | 3 |
| `--max-queries N` | Total query limit | unlimited |
| `--dataset NAME` | `locomo` or `longmemeval` | `locomo` |
| `--fast` | Skip P4/P5/Q metrics | off |
| `--no-nli` | Skip NLI model (P5 Faithfulness) | off |
| `--eval-only` | Evaluate from saved predictions | off |
| `--run-dir PATH` | Run directory for `--eval-only` | auto-detect |
| `--device-map auto` | Multi-GPU model sharding | single GPU |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key for aux-model and P4 LLM-Judge |
| `SUBMEM_JUDGE_MODEL` | LLM-Judge model (default: `gpt-4o`) |
| `SUBMEM_RESULTS` | Override results output directory |
| `HF_HOME` | HuggingFace model cache |

---

## Output Structure

```
results/20260407_220823_locomo_Qwen3.5-9B/
├── 01_data_summary.json          # Dataset statistics
├── C1/
│   ├── 02_writes_all.json        # All write logs
│   ├── 03_retrievals.json        # Per-query retrieval results
│   ├── 04_prompts.json           # Full prompts to LLM
│   ├── 05_predictions.json       # Predictions + gold answers
│   ├── 06_memory_snapshot_*.json # Memory state snapshots
│   ├── 06_run_meta.json          # Efficiency stats
│   ├── 07_metrics.json           # All computed metrics
│   ├── 08_llm_judge.json         # P4 LLM-Judge details
│   └── 09_faithfulness.json      # P5 Faithfulness details
├── C2/
│   └── ...
└── locomo_test.json              # Combined results
```

### Reading Results

```python
import json

with open("results/.../C1/07_metrics.json") as f:
    m = json.load(f)

print(f"P1 EM:      {m['task_performance']['P1_em']:.3f}")
print(f"P2 F1:      {m['task_performance']['P2_f1']:.3f}")
print(f"P3 BLEU-1:  {m['task_performance']['P3_bleu1']:.3f}")
print(f"P6 Recall:  {m['task_performance']['P6_recall_at_k']:.3f}")
print(f"R2 DRR:     {m['robustness']['R2_drr']:.3f}")
print(f"E1 Storage: {m['efficiency']['E1_storage_bytes']} bytes")
```

---

## Dependencies

```
torch>=2.1.0          transformers>=4.40.0     accelerate>=0.25.0
faiss-cpu>=1.7.4      rank_bm25>=0.2.2         networkx
scikit-learn>=1.3.0   sentence-transformers    peft
tiktoken              openai                    numpy
```
