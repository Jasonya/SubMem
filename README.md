# SubMem

A holistic evaluation harness for **memory substrates** in LLM memory agents.

A memory agent is usually reported as a single system, so it is hard to tell how
much of its behaviour comes from the agent scaffold and how much from the
substrate underneath it. SubMem holds the scaffold fixed and swaps the substrate,
which makes the substrate itself the unit of comparison. Eleven substrates run
behind one interface, across six benchmarks and three backbones, under the same
metric suite.

## Substrates

| ID | Substrate | Basis |
|----|-----------|-------|
| M1 | Dense Vector | embedding index, cosine retrieval |
| M2 | Sparse Vector | BM25 lexical matching |
| M3 | Summarized Text | ReadAgent-style gist pages |
| M4 | A-Mem | agentic memory, Xu et al. 2025 |
| M5 | LightRAG | dual-level graph RAG, Guo et al. 2024 |
| M6 | Hierarchical | RAPTOR-style recursive abstractive tree |
| M7 | ReasoningBank | reusable reasoning traces, agentic tasks only |
| M8 | Skill Bundles | MemSkill, vendored upstream plus an LLM controller |
| M9 | LoRA | parametric memory written by fine-tuning |
| M10 | Full Context | no memory, the upper-bound control |
| M11 | EpiCache | episodic KV cache selection |

All eleven implement the same interface in [memory/base.py](memory/base.py), so a
benchmark never knows which substrate it is running against.

## Benchmarks

| Benchmark | Kind | Entry point |
|-----------|------|-------------|
| LoCoMo | multi-session dialogue QA | [runners/run_experiment.py](runners/run_experiment.py) |
| LongMemEval-S | long-horizon memory QA | [runners/run_mab.py](runners/run_mab.py) |
| MemoryAgentBench | four memory competencies | [runners/run_mab.py](runners/run_mab.py) |
| ALFWorld | embodied, multi-episode | [runners/run_alfworld_rollout.py](runners/run_alfworld_rollout.py) |
| BigCodeBench | agentic code generation | [runners/run_agentic_bcb.py](runners/run_agentic_bcb.py) |
| MemoryArena | formal reasoning, group travel planning | [benchmarks/memoryarena/](benchmarks/memoryarena/) |

Backbones: Qwen3-8B, Qwen3-32B-AWQ, Gemma-4-26B-A4B-it.

## Metrics

Six families in [metrics/](metrics/): task performance (exact match, token F1,
BLEU, LLM judge, faithfulness, retrieval recall and precision), robustness
(contradiction resolution, adversarial rejection), memory efficiency (storage,
read and write latency, per-stage token and LLM-call costs, wall clock), memory
quality (compression, coverage diversity), scaling behaviour (session retention,
capacity cliff), and behavioural dynamics (recency bias).

## Getting started

```bash
conda env create -f environment.yml && conda activate submem
export OPENAI_API_KEY=...          # auxiliary LLM, embeddings, judge
python -m runners.run_experiment --help
```

Paths, model endpoints, and cache locations are in [config.py](config.py).
Cluster job scripts live in [scripts/](scripts/) and assume SLURM. They read their
locations from environment variables: `SUBMEM_ROOT` (this checkout), `SUBMEM_RESULTS`
(run outputs, default `$SUBMEM_ROOT/results`), `HF_HOME`, and `VLLM_SIF` (the vLLM
Apptainer image used by the serving scripts).

MemoryArena is not vendored. [benchmarks/memoryarena/setup.sh](benchmarks/memoryarena/setup.sh)
clones it, applies the integration patch, and installs the adapter, so a fresh
clone of this repository is enough to run it.

### One prerequisite worth knowing about

M5 (LightRAG) needs an `LD_PRELOAD` shim for a symbol-resolution conflict between
its native dependencies and the serving stack. The other ten substrates import
without it. [GUIDE.md](GUIDE.md) has the exact error and the fix.

## Layout

```
memory/          the eleven substrates behind one interface
benchmarks/      benchmark adapters, including the MemoryArena integration
runners/         entry points, one per benchmark family
metrics/         the six metric families
scripts/         SLURM job scripts and sweeps
tools/           inspection and harvesting utilities
docs/            runbook, prompt inventory, LoCoMo quick start
```

## What is not here

Raw run outputs, figures, and analysis artifacts run to several GB and are not
tracked. Every number in the paper is reproducible from the code here plus the
configurations under `benchmarks/` and `scripts/`.

## Further reading

[GUIDE.md](GUIDE.md) for the operational walkthrough,
[IMPLEMENTATION.md](IMPLEMENTATION.md) for substrate-level design notes,
[docs/RUNBOOK.md](docs/RUNBOOK.md) for cluster procedures.
