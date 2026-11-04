# MemoryArena integration

Everything needed to reproduce the MemoryArena results reported in the paper's
appendix, kept inside this repository so the evaluation does not depend on a
sibling checkout being present at a particular path.

MemoryArena itself is upstream code and is not vendored here. `setup.sh` clones
it, applies our integration patch, and installs the adapter and configurations,
so a fresh clone of this repository is enough to run the evaluation.

## Layout

```
setup.sh                    clone the benchmark, patch it, install the integration
patch/                      our changes to upstream files, as a git patch
adapter/submem_adapter.py   SubMem substrates behind MemoryArena's memory interface
configs/math/               formal reasoning domain, one config per substrate
configs/travel/             group travel planning domain, one config per substrate
slurm/                      job scripts for both domains
```

## Reproducing

```bash
./setup.sh                       # clones, patches, installs, verifies
export OPENAI_API_KEY=...        # auxiliary LLM, embeddings, judge
```

`setup.sh` prints the exact environment variables and the two submit commands
when it finishes. Substrate registration in `memory/server.py`, which exposes
the names `submem_m1` through `submem_m11`, is part of the patch.

## Settings that matter

The agent is Qwen3-8B served by vLLM; the auxiliary LLM, the embedding encoder,
and the judge are the ones the paper uses, so the only thing that changes across
runs is the substrate.

`SUBMEM_KEEP_BASE=1` keeps the opening observation of a session verbatim
alongside whatever the substrate itself stores. The travel domain needs it: the
shared trip specification appears only in that first observation and is never
restated, so a substrate that compresses it away leaves the agent without the
destination. The travel job scripts set it.

`SUBMEM_MEM_CHAR_CAP` bounds the total injected memory, the kept opening
observation included. Travel sets 20000 because the environment's tool results
already occupy most of the serving window; the math domain uses the default.

`MA_MEM_TIMEOUT` raises the memory-server timeout above the config default.
Summarizing substrates exceed 300 s on long papers, and a timeout there discards
the whole paper rather than one query.

## Raw outputs

The per-paper and per-group outputs run to several GB and are not kept in the
repository. Scoring uses MemoryArena's own `combination.combine` followed by
`eval.evaluate`.
