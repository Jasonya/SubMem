# SubMem: Memory Substrate Comparison Framework

## Overview

SubMem evaluates 12 memory substrates (C1-C12) for LLM agents on conversational QA benchmarks. Each substrate represents a different approach to storing, managing, and retrieving long-term conversational memory.

---

## Memory Substrates (C1-C12)

### External Memory (C1-C7)

| C | Name | Reference | Write | Read |
|---|------|-----------|-------|------|
| **C1** | Dense Vector | Standard RAG | per-utterance → Contriever embed → FAISS | FAISS cosine ANN search |
| **C2** | Sparse Vector | BM25 (Robertson+, '94) | per-utterance → tokenize (punctuation strip + stopword removal) | BM25Okapi scoring |
| **C3** | Raw Text | MemGPT (Packer+, '23) | FIFO buffer (50 msgs) → overflow evict + LLM recursive summary → archival FAISS | summary + full buffer + archival search |
| **C4** | Summarized Text | ReadAgent (Lee+, ICLR'24) | ~500-word pages → LLM gist → embed gist → FAISS; store original page in lookup | LLM reads all gists in-context → decides page lookups → return original pages |
| **C5** | Extracted Facts | Mem0 (Chhikara+, '25) | per-message-pair → LLM extract facts → LLM classify (ADD/UPDATE/DELETE/NOOP) → FAISS | FAISS search over facts |
| **C6** | Temporal KG | Graphiti/Zep (Raghavan+, '25) | per-utterance episode → LLM entity extraction → entity resolution → LLM edge extraction with temporal bounds → conflict resolution | BM25 + cosine + BFS graph traversal → RRF reranking |
| **C7** | Hierarchical | H-MEM (Sun+, EACL'26) | per-utterance → LLM extract 4-level hierarchy (Domain→Category→Trace→Episode) | top-down layer-by-layer cosine filtering |

### Parametric Memory (C8-C9)

| C | Name | Reference | Write | Read |
|---|------|-----------|-------|------|
| **C8** | Knowledge Editing | MEMIT (Meng+, '22) | LLM extract triples → MEMIT batch edit MLP weights | direct forward pass (no retrieval) |
| **C9** | LoRA | LoRA (Hu+, '22) | LLM generate QA pairs → LoRA fine-tune adapter | direct forward pass (no retrieval) |

### Baseline (C10)

| C | Name | Write | Read |
|---|------|-------|------|
| **C10** | Full Context | append to buffer | concatenate all text, truncate to context window (8192 tokens) |

### KV Cache (C11-C12)

| C | Name | Reference | Write | Read |
|---|------|-----------|-------|------|
| **C11** | Query-Aware KV Cache | SnapKV (Li+, NeurIPS'24) | buffer text | per-query: full prefill → observation window attention → select top 20% utterances → re-prefill selected + query → generate |
| **C12** | Episodic KV Cache | EpiCache (Kim+, '25 Apple) | buffer text → cluster into episodes (K-means) → per-episode attention-guided text selection | match query to episode → re-prefill episode's selected text + query → generate |

### KV Cache Hybrid Architecture Note

C11/C12 use **attention-guided text selection + re-prefill** instead of direct KV cache manipulation. This is because Qwen3.5 uses a hybrid architecture (3:1 linear:standard attention). Compressing only the standard attention KV cache while leaving linear attention recurrent state unchanged causes layer inconsistency. The two-pass approach (prefill for attention scores → select text → re-prefill) lets both layer types process the same selected text consistently. The memory substrate remains KV cache; only the construction method differs.

---

## Inheritance Structure

```
BaseMemoryUnit (memory/base.py)
├── DenseRetrievalMemory (memory/dense_retrieval.py) — Contriever + FAISS
│   ├── C1 DenseVectorMemory    (c1_dense_vector.py)
│   ├── C3 RawTextMemory        (c3_raw_text.py) — 3-tier: buffer + summary + archival
│   ├── C4 SummarizedTextMemory (c4_summarized_text.py) — gist index + page lookup
│   └── C5 ExtractedFactsMemory (c5_extracted_facts.py) — fact extract + classify
├── C2  SparseVectorMemory      (c2_sparse_vector.py) — BM25
├── C6  KnowledgeGraphMemory    (c6_knowledge_graph.py) — networkx + BM25/cosine/BFS RRF
├── C7  HierarchicalMemory      (c7_hierarchical.py) — 4-level top-down
├── C8  MEMITMemory             (c8_memit.py) — weight editing via third_party/memit
├── C9  LoRAMemory              (c9_lora.py) — peft LoRA adapter
├── C10 FullContextMemory       (c10_full_context.py) — no memory baseline
├── C11 SnapKVMemory            (c11_snapkv.py) — per-query attention selection
└── C12 EpiCacheMemory          (c12_epicache.py) — episodic attention selection
```

---

## Metrics (6 families, 18 metrics)

### P — Performance (5)

| Metric | Description | Computation | When |
|--------|-------------|-------------|------|
| **P1 EM** | Exact Match | `normalize(pred) == normalize(gold)` | post-hoc from `05_predictions.json` |
| **P2 F1** | Token-level F1 | precision/recall harmonic mean | post-hoc |
| **P3 Faithfulness** | NLI-based grounding | rule-based claim decomposition → DeBERTa NLI entailment ratio | post-hoc via `--eval-only` |
| **P4 Recall@k** | Retrieval recall | `\|gold evidence ∩ retrieved dia_ids\| / \|gold\|` | post-hoc from `03_retrievals.json` |
| **P5 Precision@k** | Retrieval precision | `\|gold evidence ∩ retrieved dia_ids\| / \|retrieved\|` | post-hoc |

### R — Robustness (2)

| Metric | Description | Computation | When |
|--------|-------------|-------------|------|
| **R1 CRA** | Contradiction Resolution | LongMemEval `knowledge_update` category accuracy | requires LongMemEval run |
| **R2 DRR** | Distractor Rejection | LoCoMo adversarial: word overlap with `adversarial_answer` < 50% | post-hoc |

### E — Efficiency (5)

| Metric | Description | Computation | When |
|--------|-------------|-------------|------|
| **E1** | Storage (bytes) | `memory_unit.get_external_storage_bytes()` | inference time |
| **E2** | Read Latency (mean/p50/p90 ms) | per `read()` timing | inference time |
| **E3** | Write Latency (mean/p50/p90 ms) | per `write()` timing (amortized for lazy indexing) | inference time |
| **E4** | LLM Token Consumption | write-phase LLM input + output tokens | inference time |
| **E5** | LLM Calls | write-phase LLM call count | inference time |

### Q — Memory Quality (4)

| Metric | Description | Computation | When |
|--------|-------------|-------------|------|
| **Q1** | Write Fidelity | sample 20% entries, self-query, check top-1 is self | inference time (needs live memory) |
| **Q2** | Temporal Consistency | LLM extract temporal pairs, query latest value | inference time (needs live memory) |
| **Q3** | Compression Ratio | original tokens / stored tokens | post-hoc from snapshot |
| **Q4** | Memory Entropy | entries embed → K-means → Shannon entropy | post-hoc from snapshot |

### S — Scaling (1)

| Metric | Description | Computation | When |
|--------|-------------|-------------|------|
| **S1 CBPC** | Cross-Benchmark Consistency | `1 - CoV(scores across benchmarks)` | post-hoc, requires ≥2 benchmark runs |

### D — Dynamics (1)

| Metric | Description | Computation | When |
|--------|-------------|-------------|------|
| **D3 PRB** | Primacy-Recency Bias | EM difference between early 10% and recent 10% (by evidence position) | post-hoc |

---

## Benchmarks

### LoCoMo (primary)

- **Source**: snap-research/locomo, `locomo10.json`
- **Data**: 10 conversations, ~1986 QA total, avg 19 sessions / 300 turns per conversation
- **Categories**: single_hop (282), temporal (321), multi_hop (96), open_domain (841), adversarial (446)
- **Protocol**: per-conversation, write ALL sessions then answer ALL queries. Reset between conversations.
- **Primary metric**: F1

### LongMemEval

- **Source**: `xiaowu0162/longmemeval-cleaned`
- **Data**: 500 questions, 3 variants (oracle / s ~40 sessions / m ~500 sessions)
- **Categories**: single_session, multi_session, knowledge_update (78, used for R1 CRA), temporal_reasoning, abstention
- **Primary metric**: Accuracy

### ALFWorld (agentic rollout — isolated runner)

Unlike the QA benchmarks above, ALFWorld runs an **interactive TextWorld
environment** and measures **task success rate** — the canonical metric used
by ReAct, Reflexion, and MemSkill. The runner is intentionally separate from
`runners/run_c1c2_test.py` because the protocol is multi-turn rollout, not per-query
inference.

- **Source**: canonical ALFWorld data via `alfworld-download` →
  `$ALFWORLD_DATA/json_2.1.1/{train,valid_seen,valid_unseen}`
- **Eval split**: `eval_out_of_distribution` = 134 valid_unseen tasks
- **Task types**: `pick_and_place_simple`, `pick_two_obj_and_place`,
  `pick_clean_then_place_in_recep`, `pick_cool_then_place_in_recep`,
  `pick_heat_then_place_in_recep`, `look_at_obj_in_light`
- **Primary metric**: binary task success rate (env-grounded reward)
- **Secondary metrics**: avg steps to completion, per-type success

**Memory model — within-episode (Reflexion-style):**

For each task the memory unit is reset, then grows step-by-step as the agent
interacts with the live env:

```
env.reset() → obs_0
memory.reset()
memory.write(f"Initial observation: {obs_0}")

for step in range(max_steps):
    retrieved = memory.read(obs_t, top_k=5)          # relevant past steps
    prompt    = build_prompt(goal, obs_t, admissible, retrieved, history)
    action    = model.generate(prompt)                # parsed to admissible set
    obs_{t+1}, reward, done, _ = env.step(action)
    memory.write(f"At step {t}, I did `{action}` and saw: {obs_{t+1}}")
    if done: break
success = bool(reward)
```

The agent gets **no cross-task memory** — each episode starts fresh. This
tests the substrate's ability to compress and recall *within a single 30-50
step trajectory*. To add cross-episode transfer (MemSkill-style), a second
memory layer that persists across `env.reset()` calls would be needed.

**Action parsing.** Model outputs a free-form line; we extract the first line
after `Action:`, match it against `info["admissible_commands"]`
(exact → normalised → fuzzy substring → fallback). Unparseable outputs do not
crash — the episode continues with the closest admissible action.

**Prompt shape.** Goal + retrieved reflections + last 6 trajectory steps +
current observation + admissible actions (top 40) + `Action:` completion cue.
Explicit reminder that first action must be `go to <receptacle>` (empty
inventory semantics).

---

## Project Structure

```
SubMem/
├── memory/                          # Memory substrates
│   ├── base.py                      # BaseMemoryUnit interface
│   ├── dense_retrieval.py           # Shared Contriever + FAISS layer
│   ├── preprocessing.py             # chunk_by_tokens, chunk_by_tokens_with_metadata
│   ├── c1_dense_vector.py           # C1
│   ├── c2_sparse_vector.py          # C2
│   ├── c3_raw_text.py               # C3
│   ├── c4_summarized_text.py        # C4
│   ├── c5_extracted_facts.py        # C5
│   ├── c6_knowledge_graph.py        # C6
│   ├── c7_hierarchical.py           # C7
│   ├── c8_memit.py                  # C8
│   ├── c9_lora.py                   # C9
│   ├── c10_full_context.py          # C10
│   ├── c11_snapkv.py               # C11
│   ├── c12_epicache.py             # C12
│   └── __init__.py                  # Registry + factory
│
├── metrics/                         # Evaluation metrics
│   ├── __init__.py                  # LOCAL_METRIC_FAMILIES / GLOBAL_METRIC_FAMILIES
│   ├── task_performance.py          # P1 EM, P2 F1
│   ├── nli_scorer.py                # P3 Faithfulness (DeBERTa NLI)
│   ├── retrieval_quality.py         # P4 Recall@k, P5 Precision@k
│   ├── robustness.py                # R1 CRA, R2 DRR
│   ├── memory_efficiency.py         # E1-E5
│   ├── memory_quality.py            # Q1-Q4 + temporal pair extraction
│   ├── scaling_behavior.py          # S1 CBPC (global)
│   └── behavioral_dynamics.py       # D3 PRB
│
├── benchmarks/                      # Benchmark loaders
│   ├── base.py                      # BaseBenchmark, BenchmarkSample, PredictionResult
│   ├── locomo.py                    # LoCoMo
│   ├── longmemeval.py               # LongMemEval
│   ├── memory_agent_bench.py        # MemoryAgentBench
│   └── alfworld.py                  # ALFWorld (Plan A proxy, superseded by rollout runner)
│
├── models/                          # LLM backends
│   ├── base.py                      # BaseModel interface
│   ├── qwen.py                      # Local HuggingFace (Qwen3.5-0.8B)
│   └── api.py                       # OpenAI-compatible API (GPT-4o-mini, GPT-5-nano, etc.)
│
├── third_party/
│   └── memit/                       # MEMIT code (git clone https://github.com/kmeng01/memit)
│
├── runners/run_c1c2_test.py                 # QA runner (LoCoMo / LongMemEval / MemoryAgentBench)
├── runners/run_alfworld_rollout.py          # Agentic rollout runner (ALFWorld, 134 valid_unseen)
├── run_alfworld_rollout_vllm.sh     # Slurm script: spins up vLLM then runs rollout
├── scripts/preprocess_alfworld.py   # Builds Plan-A eval split from dslighfdsl corpus
├── config.py                        # ModelConfig, MemoryConfig, BenchmarkConfig
└── .env                             # OPENAI_API_KEY (gitignored)
```

---

## Running Experiments

### Basic Commands

```bash
# C1-C3 (no API needed, fast)
python3 runners/run_c1c2_test.py --memory C1 C2 C3 --max-samples 10 --fast

# C4-C7 (need API for LLM extraction)
python3 runners/run_c1c2_test.py --memory C4 C5 C6 C7 --max-samples 1 --fast --aux-model openai/gpt-4o-mini

# C10-C12 (KV cache, local only)
python3 runners/run_c1c2_test.py --memory C10 C11 C12 --max-samples 1 --fast

# Post-hoc evaluation (P3/P4/P5/R2/D3 from saved predictions)
python3 runners/run_c1c2_test.py --memory C1 C2 C3 --eval-only --run-dir results/<run_dir>

```

### ALFWorld Rollout

Separate runner (see `benchmarks/alfworld.py` note above). Uses a live
TextWorld env and reports **task success rate**.

**One-time setup:**

```bash
# 1. Install alfworld (already installed in $MEMSKILL_ENV)
$MEMSKILL_ENV/bin/pip install alfworld

# 2. Download canonical ALFWorld data (~320 MB: json + pddl + mrcnn weights)
#    Any writable location works; the runners read $ALFWORLD_DATA.
export ALFWORLD_DATA=<path/to/SubMem>/data/alfworld_data
$MEMSKILL_ENV/bin/alfworld-download
```

**Smoke test (2 tasks, OpenAI API, no GPU):**

```bash
export ALFWORLD_DATA=<path/to/SubMem>/data/alfworld_data
export SUBMEM_RESULTS=<path/to/results>   # optional override of the results root
set -a; source .env; set +a

$MEMSKILL_ENV/bin/python3 runners/run_alfworld_rollout.py \
    --memory C1 \
    --model openai/gpt-4o-mini \
    --max-tasks 2 --max-steps 15 --top-k 3
```

Completes in ~30 s. Verifies env loads, action parsing works, memory
writes/reads happen.

**Full run (134 valid_unseen × C1 C2 via vLLM):**

```bash
sbatch run_alfworld_rollout_vllm.sh
```

The slurm script spins up a local vLLM server (Qwen3-8B on port 8771),
points generation at it via `SUBMEM_LOCAL_BASE_URL`, and leaves
`OPENAI_BASE_URL` pointing at the real OpenAI API (for embeddings that
power C1/C2 writes). Both endpoints coexist without conflict.

Tune inside the `.sh`:
- `--max-tasks` (134 = canonical full set)
- `--max-steps` (50 = ReAct/Reflexion standard)
- `--top-k` (5 demos retrieved per step)
- `MODEL=...` (any model with vLLM support)

**Inspect results:**

```python
import json
base = "$SUBMEM_RESULTS/<ts>_alfworld_<model>"
for mid in ("C1", "C2"):
    summary = json.load(open(f"{base}/{mid}/summary.json"))
    print(mid, summary["success_rate"], summary["by_type"])
    # Full per-task trajectories:
    tasks = json.load(open(f"{base}/{mid}/per_task.json"))
    print(f"task 0 ended at step {tasks[0]['steps']}, "
          f"success={tasks[0]['success']}")
```

**Budget notes:** 134 tasks × 50 steps × 2 memory units × ~1 s/step
≈ 4 h of GPU time ≈ $3.60. For a cheap sanity check run 30 tasks × 30
steps ≈ 30 min ≈ $1.80.

**Known limitations:**
- Zero-shot Qwen3-8B baselines are near-0 task success (canonical ReAct
  numbers used 540B+ models with hand-crafted demos). Pipeline validates
  independently of model capability.
- `parse_action` falls back to `filtered_admissible[0]` when the model
  output doesn't match any admissible — check trajectories for signs of
  heavy fallback use before interpreting numbers.
- Cross-episode memory is NOT tested here; each `env.reset()` also does
  `memory.reset()`. For MemSkill-style transfer, add a second persistent
  memory layer on top.

### CLI Flags

| Flag | Description |
|------|-------------|
| `--memory C1 C2 ...` | Which substrates to run |
| `--max-samples N` | Number of LoCoMo conversations (default 3, max 10) |
| `--max-queries N` | Total query limit across all samples |
| `--fast` | Shortcut: `--no-nli --no-p3 --no-q` (only P1/P2/E1-E5) |
| `--no-nli` | Skip NLI model loading |
| `--no-p3` | Skip claim decomposition |
| `--no-q` | Skip Q family (Q1-Q4) |
| `--aux-model openai/gpt-4o-mini` | API model for C4-C8 write-phase LLM calls |
| `--model Qwen/Qwen3.5-0.8B` | Base model for answer generation |
| `--dataset locomo` | Benchmark (locomo / longmemeval) |
| `--eval-only` | Skip inference, evaluate from saved predictions |
| `--run-dir results/...` | Specify run directory for eval-only |

### Model Routing

```
--model Qwen/Qwen3.5-0.8B       → Local HuggingFace model (answer generation)
--aux-model openai/gpt-4o-mini   → API model (C4-C8 write-phase LLM calls)

C1-C3:  write uses no LLM → aux-model irrelevant
C4-C7:  write uses aux-model for extraction/summarization
C8:     write uses aux-model for triple extraction, MEMIT edits local model
C9:     write uses local model for QA generation + LoRA training
C10:    no LLM in write
C11-C12: no LLM in write, uses local model for prefill/attention
```

---

## Output Structure

Each run creates a timestamped directory:

```
results/20260401_234420_locomo_Qwen3.5-0.8B/
├── 01_data_summary.json                    # Dataset statistics
├── C1/
│   ├── 02_writes_sample0.json              # Per-sample write logs
│   ├── 02_writes_all.json                  # All write logs (incremental save)
│   ├── 03_retrievals.json                  # Per-query retrieval results
│   ├── 04_prompts.json                     # Full prompts sent to LLM
│   ├── 05_predictions.json                 # Predictions + gold answers
│   ├── 06_memory_snapshot_sample0.json     # Memory state after sample 0
│   ├── 06_memory_snapshot_sample1.json     # Memory state after sample 1
│   ├── 06_run_meta.json                    # E/Q stats
│   └── 07_metrics.json                     # All computed metrics
├── C2/
│   └── ...
├── examples/                               # Per-query divergence examples
├── locomo_test.json                        # Combined results for all C's
```

### Crash Recovery

All logs are saved incrementally after each sample. If the run crashes at sample 5:
- `05_predictions.json` has samples 0-4
- `06_memory_snapshot_sample0-4.json` all saved
- Can `--eval-only` from saved predictions for P1/P2/P3/P4/P5/R2/D3
- E/Q stats only available if the run completed fully

---

## Key Design Decisions

### Input Granularity (faithful to original papers)

| C | Input Unit | Rationale |
|---|------------|-----------|
| C1/C2 | per-utterance | Standard RAG baseline |
| C3 | per-utterance → FIFO buffer | MemGPT processes messages one at a time |
| C4 | ~500-word pages | ReadAgent: w_min=280, w_max=600 words |
| C5 | per-message-pair | Mem0: `add(messages)` processes pairs |
| C6 | per-utterance episode | Graphiti: `add_episode()` per message |
| C7 | per-utterance | H-MEM: "after each interaction" |

### E3 Write Latency — Amortized for Lazy Indexing

C3/C4/C5/C6/C7 use lazy indexing (`_ensure_indexed()` on first read). Write latency is amortized: total indexing time / number of entries. This gives a fair per-entry write cost comparable to C1's immediate indexing.

### E5 LLM Calls — Write-Only

E5 counts only write-phase LLM calls (extraction, classification, summarization). Read-phase `model.generate()` is the pipeline's responsibility, not the memory unit's.

### P4/P5 dia_id Tracking

Chunk-based methods (C3/C4/C5) use `chunk_by_tokens_with_metadata()` to track which original utterance `dia_id`s are covered by each chunk. C6 attaches `dia_ids` per edge. This allows P4/P5 evidence matching across all C's.

### API Model for Write-Phase LLM Calls

C4-C8's write-phase LLM calls (extraction, summarization, classification) can use an API model (`--aux-model`) while answer generation uses the local model. This separates extraction quality from generation quality.

For GPT-5-nano (reasoning model): `max_completion_tokens = max_new_tokens + 4096` to account for reasoning token overhead.

---

## Dependencies

### Core
```
torch, transformers, faiss-cpu, numpy, scikit-learn
rank_bm25                    # C2 BM25
networkx                     # C6 knowledge graph
sentence-transformers        # C12 EpiCache segment encoding
peft                         # C9 LoRA
tiktoken                     # API model tokenization
openai                       # API model backend
matplotlib                   # MEMIT dependency
```

### Optional
```
third_party/memit/           # C8 MEMIT (git clone https://github.com/kmeng01/memit)
```

### API
```
.env file with OPENAI_API_KEY for --aux-model
```

---

## Known Limitations

1. **Qwen3.5-0.8B is small** — F1 scores are low across all C's (~0.15-0.20). Larger models will show clearer differentiation.
2. **C11/C12 use text selection instead of direct KV cache compression** — hybrid architecture (linear + standard attention) prevents direct KV manipulation. Faithful in selection logic, adapted in execution.
3. **C8 MEMIT requires EasyEdit or direct memit package** — falls back to unedited model if not installed.
4. **P4/P5 = 0 for parametric methods (C8/C9/C11/C12)** — no explicit `dia_id` in retrieved entries since retrieval is internal.
5. **LoCoMo public data is 10 conversations** — paper reports on 50. Our results are on the 10 available.
