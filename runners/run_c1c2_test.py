"""Test memory substrates (M*) on LoCoMo — saves all intermediates.

Output folder layout:
    results/<timestamp>_<dataset>_<model>/<C1..C13>/
        02_writes_*.json, 03_retrievals.json, 04_prompts.json,
        05_predictions.json, 06_run_meta.json, 07_metrics.json
    results/<timestamp>_<dataset>_<model>/
        01_data_summary.json, locomo_test.json, examples/

Usage:
    # Full run all active substrates:
    python SubMem/run_c1c2_test.py

    # Run specific units:
    python SubMem/run_c1c2_test.py --memory M1 M2 M4

    # Evaluate only from saved predictions (auto-detect latest run):
    python SubMem/run_c1c2_test.py --eval-only

    # Evaluate a specific run:
    python SubMem/run_c1c2_test.py --eval-only --run-dir results/20260326_143000_locomo_Qwen2.5-7B-Instruct
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # parent of the SubMem package

from SubMem.config import ModelConfig, MemoryConfig, BenchmarkConfig, to_modern_id
from SubMem.models import create_model
from SubMem.memory import create_memory_unit
from SubMem.benchmarks import create_benchmark
from SubMem.benchmarks.base import PredictionResult
from SubMem.metrics.task_performance import TaskPerformanceMetrics, compute_em, compute_f1, compute_bleu1
from SubMem.metrics.memory_efficiency import MemoryEfficiencyMetrics
from SubMem.metrics.nli_scorer import (
    NLIScorer, decompose_claims_llm, decompose_claims_rule,
    compute_faithfulness_nli_score,
)
from SubMem.metrics.retrieval_quality import RetrievalQualityMetrics
from SubMem.metrics.memory_quality import MemoryQualityMetrics
from SubMem.cache import compute_cache_key, cache_lookup, cache_store
from SubMem.memory import MEMORY_UNIT_REGISTRY
from SubMem.metrics.llm_judge import LLMJudge, compute_llm_judge_score
from SubMem.metrics.robustness import RobustnessMetrics
from SubMem.metrics.scaling_behavior import ScalingBehaviorMetrics
from SubMem.metrics.behavioral_dynamics import BehavioralDynamicsMetrics
from SubMem.utils.helpers import set_seed, save_json, load_json, get_gpu_memory_usage

RESULTS_BASE = os.environ.get("SUBMEM_RESULTS", os.path.join(os.path.dirname(__file__), "results"))


def _parse_evidence_position(evidence: list) -> int:
    """Parse LoCoMo evidence IDs to an absolute position.

    Evidence format: ["D1:3", "D5:8"] → session_num * 1000 + turn_num.
    Returns the position of the first evidence ID, or 0 if none.
    """
    if not evidence:
        return 0
    first = evidence[0]  # e.g. "D1:3"
    match = re.match(r"D(\d+):(\d+)", str(first))
    if match:
        return int(match.group(1)) * 1000 + int(match.group(2))
    return 0


def _model_short_name(model_name: str) -> str:
    """Qwen/Qwen3.5-9B -> Qwen2.5-7B-Instruct"""
    return model_name.split("/")[-1]


def _make_run_dir(dataset: str, model_name: str) -> str:
    """Create and return a timestamped run directory.

    Layout: results/<YYYYMMDD_HHMMSS>_<dataset>_<model>_pid<PID>/
    PID suffix avoids collisions when sweep submits multiple jobs in the
    same wall-clock second.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = _model_short_name(model_name)
    name = f"{ts}_{dataset}_{short}_pid{os.getpid()}"
    d = os.path.join(RESULTS_BASE, name)
    # If still colliding (extremely unlikely), append microsecond
    if os.path.exists(d):
        name = f"{name}_{datetime.now().strftime('%f')}"
        d = os.path.join(RESULTS_BASE, name)
    os.makedirs(d, exist_ok=True)
    return d


def _find_latest_run_dir() -> str:
    """Find the most recent run directory under RESULTS_BASE."""
    if not os.path.isdir(RESULTS_BASE):
        raise FileNotFoundError(f"No results directory: {RESULTS_BASE}")
    dirs = [
        d for d in os.listdir(RESULTS_BASE)
        if os.path.isdir(os.path.join(RESULTS_BASE, d)) and re.match(r"\d{8}_\d{6}_", d)
    ]
    if not dirs:
        raise FileNotFoundError(f"No run directories found in {RESULTS_BASE}")
    dirs.sort(reverse=True)
    return os.path.join(RESULTS_BASE, dirs[0])


def _memory_dir(run_dir: str, memory_id: str) -> str:
    """Per-memory-unit subdirectory: <run_dir>/<C1>/ etc."""
    d = os.path.join(run_dir, memory_id)
    os.makedirs(d, exist_ok=True)
    return d


# ────────────────────────────────────────────────────────────
#  Stage 1: Load & save benchmark data
# ────────────────────────────────────────────────────────────
def stage_load_data(benchmark_config):
    """Load LoCoMo samples and save raw data snapshot."""
    benchmark = create_benchmark(benchmark_config)
    samples = benchmark.load_data()

    records = []
    for s in samples:
        records.append({
            "sample_id": s.sample_id,
            "num_sessions": len(s.sessions),
            "num_turns": sum(len(sess.get("turns", [])) for sess in s.sessions),
            "num_queries": len(s.queries),
            "query_types": [q.get("type", "") for q in s.queries],
            "metadata": s.metadata,
        })
    return benchmark, samples, records


_REFUSAL_PATTERNS = re.compile(
    r"^("
    r"no evidence|no such|not mentioned|not found|not supported|"
    r"no information|no memory|no relevant|cannot be determined|"
    r"unable to determine|cannot determine|no record|"
    r"there is no|none found|unknown|n/a|unanswerable|"
    r"no \w+ mentioned|no \w+ happened|no \w+ in memor"
    r")",
    re.IGNORECASE,
)


def _normalize_refusal(answer: str) -> str:
    """Convert refusal-style answers to empty string for adversarial scoring.

    When the QA prompt instructs the model to refuse unanswerable questions,
    the model often outputs phrases like 'No evidence in memories' instead of
    an empty string. This normalises those patterns so compute_f1 scores them
    as correct refusals (F1=1.0 when gold is also empty).
    """
    stripped = answer.strip().rstrip(".")
    if _REFUSAL_PATTERNS.match(stripped):
        return ""
    return answer


# ────────────────────────────────────────────────────────────
#  Stage 2-5: Run full pipeline with all intermediates
# ────────────────────────────────────────────────────────────
def _try_cache_load(actual_memory_id, sample, mem_config, mem_model, *,
                    benchmark_name, encoder_model, encoder_dim, aux_model_name,
                    gen_model_name, use_cache):
    """Try load memory unit from cache. Returns (unit_or_none, cache_dir_or_none, cache_key)."""
    if not use_cache:
        return None, None, None
    unit_cls = MEMORY_UNIT_REGISTRY.get(actual_memory_id)
    if unit_cls is None or not unit_cls.__module__:
        return None, None, None
    if not getattr(unit_cls, "save_state", None):
        return None, None, None
    # Substrate must implement save_state (override of base raises)
    try:
        # Invoke supports_caching on a temp instance — but we don't have one;
        # check via class method MRO:
        from SubMem.memory.base import BaseMemoryUnit
        if unit_cls.save_state is BaseMemoryUnit.save_state:
            return None, None, None
    except Exception:
        return None, None, None
    import sys as _sys
    mod_path = _sys.modules.get(unit_cls.__module__).__file__
    key = compute_cache_key(
        benchmark=benchmark_name,
        substrate=actual_memory_id,
        sample_id=sample.sample_id,
        memory_unit_cls=unit_cls,
        encoder=encoder_model or "",
        encoder_dim=int(encoder_dim or 0),
        aux_model=aux_model_name or "",
        gen_model=gen_model_name or "",
        extra=mem_config.extra or {},
        substrate_module_path=mod_path,
    )
    payload = cache_lookup(key)
    if payload is None:
        return None, None, key
    try:
        unit = unit_cls.load_state(str(payload / "memory_state"), mem_config, mem_model)
        print(f"  [cache] HIT {key.hash()} for {actual_memory_id}/{sample.sample_id}", flush=True)
        return unit, payload, key
    except Exception as e:
        print(f"  [cache] HIT but load_state failed ({e}); rebuilding", flush=True)
        return None, None, key


def _try_cache_store(unit, key, write_metrics):
    """Persist a freshly-built memory unit + write_metrics into cache."""
    if key is None or unit is None:
        return
    try:
        path = cache_store(key, unit.save_state, write_metrics)
        print(f"  [cache] STORED {key.hash()} → {path}", flush=True)
    except NotImplementedError:
        pass
    except Exception as e:
        print(f"  [cache] store failed: {e}", flush=True)


def run_inference(memory_id: str, model, benchmark, samples, run_dir: str, seed: int = 42,
                  max_queries: Optional[int] = None,
                  skip_p3: bool = False, skip_q: bool = False,
                  aux_model=None, top_k: Optional[int] = None,
                  encoder_model: str = "text-embedding-3-small",
                  benchmark_name: str = "locomo",
                  use_cache: bool = True,
                  group_words: Optional[int] = None):
    """Run the full pipeline, saving every stage to disk.

    Args:
        model: main model for answer generation
        aux_model: optional API model for memory unit internal LLM calls (external-memory substrates).
                   If None, memory units use the main model.
    """
    set_seed(seed)
    rd = _memory_dir(run_dir, memory_id)

    print(f"\n{'='*60}")
    print(f"  {memory_id} | seed={seed} | INFERENCE")
    print(f"{'='*60}")

    # Parametric memory (C8/C9) must use the local model for weight editing/training
    # External-memory substrates that need an auxiliary LLM use aux_model if available
    # Normalise to the canonical M-id; legacy C-ids still accepted.
    base_id = to_modern_id(memory_id.split("-")[0])  # "M10-3K"/"C10-3K" -> "M10"
    # Internal-memory substrates (parametric / full-context / KV) need the
    # local model; M9 LoRA, M10 Full Context, M11 EpiCache (+ retired
    # C8 MEMIT, C11 SnapKV in old eval runs).
    if base_id in ("M9", "M10", "M11", "C8", "C11"):
        mem_model = model
    else:
        mem_model = aux_model if aux_model is not None else model
    # Support budget-matched M10 variants: "M10-3K" -> M10 with max_tokens=3072
    # (legacy "C10-3K" also accepted).
    extra = {}
    actual_memory_id = memory_id
    _budget_pfx = next((p for p in ("M10-", "C10-") if memory_id.startswith(p)), None)
    if _budget_pfx and memory_id[4:].rstrip("K").isdigit():
        budget_str = memory_id[4:]  # e.g. "3K"
        budget_k = int(budget_str.rstrip("K"))
        extra = {"max_tokens": budget_k * 1024}
        actual_memory_id = "M10"
        print(f"  M10 budget-matched variant: max_tokens={extra['max_tokens']}")
    extra["encoder_model"] = encoder_model
    # C9 LoRA hyperparameter overrides via env vars (for sweep experiments)
    if to_modern_id(actual_memory_id) == "M9":
        for env_key, extra_key, cast in [
            ("M9_LORA_EPOCHS", "lora_epochs", int),
            ("M9_LORA_RANK", "lora_rank", int),
            ("M9_LORA_ALPHA", "lora_alpha", int),
            ("M9_LORA_LR", "lora_lr", float),
            ("M9_LORA_BATCH_SIZE", "lora_batch_size", int),
        ]:
            v = os.environ.get(env_key)
            if v is not None and v != "":
                extra[extra_key] = cast(v)
                print(f"  M9 override: {extra_key}={extra[extra_key]}")
    mem_config = MemoryConfig(memory_id=actual_memory_id, top_k=top_k, extra=extra)
    memory_unit = create_memory_unit(mem_config, mem_model)
    total_queries = sum(len(s.queries) for s in samples)
    if max_queries is not None:
        total_queries = min(total_queries, max_queries)
    done = 0

    all_write_logs = []      # stage 2: what was written
    all_retrieval_logs = []  # stage 3: what was retrieved per query
    all_prompt_logs = []     # stage 4: full prompts
    all_prediction_logs = [] # stage 5: predictions
    predictions = []         # PredictionResult objects for evaluation
    total_original_tokens = 0  # for Q3 compression ratio

    t_start = time.time()
    # E14 amortized: this-run wall + reconstructed write-phase time when cached
    cache_write_wall_recovered_sec = 0.0  # sum of cached write phases reused

    # Cross-sample E5..E11 accumulators. Substrate ``reset()`` between samples
    # zeroes ``token_consumption`` so the unit-side counters only reflect the
    # last sample. We snapshot the unit's counters before each reset and add
    # them here so the final aggregate covers all samples.
    cross_sample_token_counts: Dict[str, int] = {}
    def _accumulate_token_counts(unit):
        tc = (unit.get_stats() or {}).get("token_consumption", {}) or {}
        for k, v in tc.items():
            if isinstance(v, (int, float)):
                cross_sample_token_counts[k] = cross_sample_token_counts.get(k, 0) + v
    # Some substrates (M9 LoRA, M10 FullCtx) do not expose token_consumption
    # in the same shape; the dict-update loop above still works because we
    # only iterate over keys that are present.

    # Resolve cache parameters (substrate-aware)
    encoder_dim = getattr(memory_unit, "_enc", None)
    encoder_dim = encoder_dim.dim if encoder_dim is not None else 0
    aux_model_name = ""
    if aux_model is not None:
        aux_model_name = getattr(aux_model, "model_name", "") or str(aux_model)
    gen_model_name = getattr(model, "model_name", "") or ""

    for si, sample in enumerate(samples):
        if max_queries is not None and done >= max_queries:
            break

        # ── Stage 2: cache lookup → load OR build memory ──
        cached_unit, cache_payload, cache_key = _try_cache_load(
            actual_memory_id, sample, mem_config, mem_model,
            benchmark_name=benchmark_name, encoder_model=encoder_model,
            encoder_dim=encoder_dim, aux_model_name=aux_model_name,
            gen_model_name=gen_model_name, use_cache=use_cache,
        )

        sample_entries = []
        if cached_unit is not None:
            # Use loaded unit; skip write phase
            memory_unit = cached_unit
            # Recover the original write-phase wall time from cached write_metrics
            try:
                import json as _json
                wm_path = cache_payload / "write_metrics.json"
                if wm_path.exists():
                    wm = _json.load(open(wm_path))
                    cache_write_wall_recovered_sec += float(wm.get("wall_sec", 0.0))
            except Exception:
                pass
            # Approximate sample_entries from buffer if available, or from the
            # benchmark's raw sessions so the write log + total_original_tokens
            # are populated even on cache HIT (otherwise Q1 ends up None).
            buffer = getattr(memory_unit, "_buffer", []) or []
            if buffer:
                for ei, text in enumerate(buffer):
                    meta = {}
                    try:
                        meta = (memory_unit._buffer_metadata[ei] if ei < len(memory_unit._buffer_metadata) else {}) or {}
                    except Exception:
                        meta = {}
                    total_original_tokens += len(model.encode(text))
                    sample_entries.append({
                        "entry_idx": ei, "session_id": meta.get("session_id"),
                        "text": text, "metadata": meta,
                        "write_latency_ms": 0.0,
                    })
            else:
                # Substrates that don't keep a _buffer (Graphiti, Letta, LoRA)
                # — recover turn text from the benchmark's session data so we
                # still count original tokens for Q1 and persist a real write
                # log downstream.
                ei = 0
                for session in sample.sessions:
                    session_date = session.get("date", "")
                    for turn in session.get("turns", []):
                        text = turn["content"]
                        if session_date:
                            text = f"[{session_date}] {text}"
                        meta = {"session_id": session.get("session_id"),
                                "session_date": session_date,
                                **turn.get("metadata", {})}
                        total_original_tokens += len(model.encode(text))
                        sample_entries.append({
                            "entry_idx": ei, "session_id": meta.get("session_id"),
                            "text": text, "metadata": meta,
                            "write_latency_ms": 0.0,
                        })
                        ei += 1
        else:
            # Snapshot prior sample's token counters before reset() wipes them.
            # (No-op when si == 0 because counters are still all zero.)
            if si > 0:
                _accumulate_token_counts(memory_unit)
            memory_unit.reset()
            entry_idx = 0
            # Optional utterance grouping (--group-words N): merge consecutive
            # turns within a session into ~N-word blocks before write().
            # Mirrors MAB's single-blob chunking so the chunk-granularity
            # sensitivity sweep has a uniform knob on
            # LoCoMo too. Default (None) preserves per-utterance writes.
            _gw = group_words
            for session in sample.sessions:
                session_date = session.get("date", "")
                _buf, _buf_meta, _buf_words = [], [], 0

                def _flush():
                    nonlocal entry_idx, total_original_tokens, _buf, _buf_meta, _buf_words
                    if not _buf:
                        return
                    text = "\n".join(_buf)
                    md = {"session_id": session.get("session_id"),
                          "session_date": session_date,
                          **(_buf_meta[0] or {}),
                          "dia_ids": [m.get("dia_id") for m in _buf_meta if m.get("dia_id")],
                          "has_answer": any(m.get("has_answer") for m in _buf_meta)}
                    total_original_tokens += len(model.encode(text))
                    t_w0 = time.perf_counter()
                    memory_unit.write(text, metadata=md)
                    t_w1 = time.perf_counter()
                    sample_entries.append({
                        "entry_idx": entry_idx,
                        "session_id": session.get("session_id"),
                        "text": text,
                        "metadata": md,
                        "write_latency_ms": round((t_w1 - t_w0) * 1000, 3),
                    })
                    entry_idx += 1
                    _buf, _buf_meta, _buf_words = [], [], 0

                for turn in session.get("turns", []):
                    text = turn["content"]
                    if session_date:
                        text = f"[{session_date}] {text}"
                    metadata = {"session_id": session.get("session_id"), "session_date": session_date, **turn.get("metadata", {})}
                    if not _gw:
                        total_original_tokens += len(model.encode(text))
                        t_w0 = time.perf_counter()
                        memory_unit.write(text, metadata=metadata)
                        t_w1 = time.perf_counter()
                        sample_entries.append({
                            "entry_idx": entry_idx,
                            "session_id": session.get("session_id"),
                            "text": text,
                            "metadata": metadata,
                            "write_latency_ms": round((t_w1 - t_w0) * 1000, 3),
                        })
                        entry_idx += 1
                    else:
                        _buf.append(text)
                        _buf_meta.append(turn.get("metadata", {}))
                        _buf_words += len(text.split())
                        if _buf_words >= _gw:
                            _flush()
                _flush()

            # ── Stage 2.5: store to cache ──
            if cache_key is not None and use_cache:
                _try_cache_store(memory_unit, cache_key, write_metrics={
                    "n_writes": len(sample_entries),
                    "wall_sec": sum(e["write_latency_ms"] for e in sample_entries) / 1000.0,
                    "per_write_latency_ms": [e["write_latency_ms"] for e in sample_entries],
                    "memory_size_bytes": memory_unit.get_external_storage_bytes() if hasattr(memory_unit, "get_external_storage_bytes") else 0,
                    "token_counts": dict(getattr(memory_unit, "_token_counts", {}) or {}),
                })

        write_log = {
            "sample_id": sample.sample_id,
            "num_entries": len(sample_entries),
            "entries": sample_entries,
            "memory_size_after": memory_unit.size(),
            "cache": {
                "hit": cached_unit is not None,
                "key": cache_key.hash() if cache_key else None,
            },
        }
        all_write_logs.append(write_log)

        # Save write log incrementally (per sample)
        save_json(write_log, os.path.join(rd, f"02_writes_sample{si}.json"))

        # Ensure any lazy indexing (e.g. C14 distillation) is done before
        # the query loop so it doesn't inflate the first query's retrieval latency.
        if hasattr(memory_unit, '_ensure_indexed'):
            memory_unit._ensure_indexed()

        # Some substrates (e.g. M8 MemSkill-faithful) batch-run an upstream
        # pipeline on the first read() and need every query up front. Hand
        # them the query list via stage_queries() before the loop. The
        # --max-queries cap is applied here too so the upstream pass does
        # not process queries the loop below will skip.
        if hasattr(memory_unit, 'stage_queries'):
            staged = list(sample.queries)
            if max_queries is not None:
                staged = staged[:max(max_queries - done, 0)]
            try:
                memory_unit.stage_queries(staged)
            except Exception as _e:
                print(f"  [stage_queries] failed: {_e}", flush=True)

        # ── Stage 3-5: Query loop ──
        for qi, query in enumerate(sample.queries):
            if max_queries is not None and done >= max_queries:
                break
            q_text = query["question"]
            q_id = query.get("id", f"{sample.sample_id}_q{qi}")

            # Stage 3: Retrieve. Only override the substrate's own top_k
            # default if the user explicitly passed --top-k; otherwise
            # each memory_unit uses the paper-faithful default baked into
            # its read() signature (e.g. C14 = 1, C1 = 10, C16 = 5).
            t_r0 = time.perf_counter()
            if top_k is None:
                retrieved = memory_unit.read(q_text)
            else:
                retrieved = memory_unit.read(q_text, top_k=top_k)
            t_r1 = time.perf_counter()

            retrieval_log = {
                "query_id": q_id,
                "sample_id": sample.sample_id,
                "query": q_text,
                "query_type": query.get("type", ""),
                "gold_evidence": query.get("evidence", []),
                "retrieval_latency_ms": round((t_r1 - t_r0) * 1000, 3),
                "num_retrieved": len(retrieved),
                "retrieved": [
                    {
                        "rank": ri,
                        "score": round(float(e.score), 6),
                        "content": e.content,
                        "metadata": e.metadata,
                    }
                    for ri, e in enumerate(retrieved)
                ],
            }
            all_retrieval_logs.append(retrieval_log)

            # Stage 4-5: Build prompt & Generate
            # C8 (parametric memory): read() already returns the answer directly
            # C5 is no longer parametric — its read() now returns raw facts
            # (like C1/C2), and the runner's own generator produces the answer.
            # Keeping C5 in this list made the runner return the raw first fact
            # verbatim as the prediction.
            is_parametric = base_id in ("M9", "M11", "C8", "C11", "C3-Letta")

            if is_parametric and retrieved:
                # Parametric memory: read() output IS the answer
                answer = retrieved[0].content
                prompt = f"[parametric] {q_text}"
                t_g0 = t_r1
                t_g1 = t_r1  # generation happened inside read()
            else:
                # Standard pipeline: retrieve → build prompt → generate
                if benchmark.name == "longmemeval":
                    # Group retrieved entries by session for structured prompt
                    from collections import OrderedDict
                    sess_groups = OrderedDict()
                    for e in retrieved:
                        sid = e.metadata.get("session_id", "unknown")
                        sess_groups.setdefault(sid, []).append(e.content)
                    parts = []
                    for i, (sid, texts) in enumerate(sess_groups.items()):
                        parts.append(f"### Session {i+1}:\nSession Content:\n" + "\n".join(texts))
                    context = "\n\n".join(parts)
                else:
                    context = "\n".join([e.content for e in retrieved])
                prompt = benchmark.build_prompt(query, context)
                t_g0 = time.perf_counter()
                answer = model.generate(prompt)
                t_g1 = time.perf_counter()

            # Post-process: normalize refusal patterns to empty string
            # so adversarial queries (gold="") get F1=1.0 when model refuses
            answer = _normalize_refusal(answer)

            prompt_log = {
                "query_id": q_id,
                "prompt_length_chars": len(prompt),
                "prompt": prompt,
            }
            all_prompt_logs.append(prompt_log)

            gold = query.get("answer", "")

            # Decompose prediction into atomic claims (for P5 faithfulness NLI).
            # Always done (skip_p3 only disables P3 BLEU; claims are cheap
            # regex-based). Without this, P5 sees empty claim list → 0.
            claims = decompose_claims_rule(answer)
            # Also store retrieved content for offline NLI scoring.
            # For parametric memories (C5), .content is the synthesized answer;
            # pull raw memory facts from metadata so NLI has real premises.
            if is_parametric and retrieved:
                md = retrieved[0].metadata or {}
                raw_mems = []
                for m in (md.get("speaker_a_memories") or []) + (md.get("speaker_b_memories") or []):
                    if isinstance(m, dict) and m.get("memory"):
                        raw_mems.append(str(m["memory"]))
                retrieved_contents = raw_mems if raw_mems else [e.content for e in retrieved]
            else:
                # For substrates whose .content is itself an LLM-paraphrased
                # answer rather than raw retrieved text (e.g.\ M5 LightRAG),
                # prefer the raw underlying chunks if the wrapper exposes them
                # under metadata.raw_chunks. Otherwise fall back to .content.
                raw_mems = []
                if retrieved:
                    for e in retrieved:
                        rc = (e.metadata or {}).get("raw_chunks") or []
                        for c in rc:
                            if c:
                                raw_mems.append(str(c))
                retrieved_contents = (
                    raw_mems if raw_mems else [e.content for e in retrieved]
                )

            pred_log = {
                "query_id": q_id,
                "sample_id": sample.sample_id,
                "query": q_text,
                "query_type": query.get("type", ""),
                "gold_answer": gold,
                "prediction": answer,
                "claims": claims,
                "retrieved_contents": retrieved_contents,
                "generation_latency_ms": round((t_g1 - t_g0) * 1000, 3),
                "retrieval_latency_ms": retrieval_log["retrieval_latency_ms"],
                "evidence_position": _parse_evidence_position(query.get("evidence", [])),
            }
            # Preserve adversarial_answer for R2 DRR metric
            if "adversarial_answer" in query:
                pred_log["adversarial_answer"] = query["adversarial_answer"]
            all_prediction_logs.append(pred_log)

            predictions.append(PredictionResult(
                query_id=q_id,
                query=q_text,
                prediction=answer,
                gold_answer=str(gold),
                query_type=query.get("type", ""),
                metadata={"sample_id": sample.sample_id},
            ))

            done += 1
            if done % 20 == 0 or done == total_queries:
                print(f"  [{done}/{total_queries}] queries done", flush=True)

        # ── Per-sample: save all intermediates incrementally ──
        # Memory snapshot (before reset) + cached embeddings for Q2.
        # We reconstruct embeddings from the memory unit's FAISS index if
        # available — those were already computed during write, so this is
        # zero-cost. Saving them as a sibling .npy lets offline Q2
        # computation cluster without re-embedding via OpenAI.
        sample_entries_snapshot = memory_unit.get_all_entries()
        snapshot = {
            "sample_id": sample.sample_id,
            "memory_id": memory_id,
            "num_entries": len(sample_entries_snapshot),
            "entries": [
                {"content": e.content, "metadata": e.metadata, "score": e.score}
                for e in sample_entries_snapshot
            ],
        }
        snap_path = os.path.join(rd, f"06_memory_snapshot_sample{si}.json")
        save_json(snapshot, snap_path)

        # Sidecar embeddings for Q2 (no-cost reuse of write-time vectors).
        idx = getattr(memory_unit, "_index", None)
        n_entries = len(sample_entries_snapshot)
        if idx is not None and getattr(idx, "ntotal", 0) >= n_entries > 0:
            try:
                import numpy as _np
                embs = _np.vstack([idx.reconstruct(i) for i in range(n_entries)])
                emb_path = snap_path.replace(".json", "_embeddings.npy")
                _np.save(emb_path, embs)
            except Exception as _e:
                pass  # embedding cache is best-effort

        # Save all accumulated logs after each sample (overwrite previous)
        # So if the run crashes, we still have results up to the last completed sample
        save_json(all_write_logs, os.path.join(rd, "02_writes_all.json"))
        save_json(all_retrieval_logs, os.path.join(rd, "03_retrievals.json"))
        save_json(all_prompt_logs, os.path.join(rd, "04_prompts.json"))
        save_json(all_prediction_logs, os.path.join(rd, "05_predictions.json"))
        print(f"  [sample {si+1}/{len(samples)}] saved ({len(all_prediction_logs)} predictions so far)", flush=True)

    t_total = time.time() - t_start

    # ── Stage 6: Memory unit stats (E family) ──
    # IMPORTANT: record E stats BEFORE Q diagnostics to avoid polluting latency data
    # Collect per-query inference times (retrieval + generation)
    inference_times_ms = []
    for p in all_prediction_logs:
        ret_ms = p.get("retrieval_latency_ms", 0)
        gen_ms = p.get("generation_latency_ms", 0)
        inference_times_ms.append(ret_ms + gen_ms)

    # Snapshot the LAST sample's token counters before reading mem_stats so
    # the cross-sample accumulator covers every sample, not just sample N-1.
    _accumulate_token_counts(memory_unit)

    mem_stats = memory_unit.get_stats()
    # Replace the unit's per-sample token_consumption with the cross-sample
    # accumulated counts so downstream E5..E13 reflect the whole run.
    if cross_sample_token_counts:
        # Preserve any keys the unit had that we did not accumulate (rare).
        merged = dict((mem_stats.get("token_consumption") or {}))
        for k, v in cross_sample_token_counts.items():
            merged[k] = v
        mem_stats["token_consumption"] = merged

    efficiency = MemoryEfficiencyMetrics.compute(memory_unit, inference_times_ms=inference_times_ms)
    # MemoryEfficiencyMetrics reads from memory_unit.get_stats() internally;
    # patch its E5..E13 with the cross-sample counts (we already merged them
    # into mem_stats above for downstream serialisation, but the live
    # ``memory_unit._token_counts`` only holds the last sample's data).
    if cross_sample_token_counts:
        tc = cross_sample_token_counts
        efficiency["E5_construct_tokens"] = (
            tc.get("write_llm_input", 0) + tc.get("write_llm_output", 0)
        )
        efficiency["E6_construct_calls"] = tc.get("write_llm_calls", 0)
        # E7 (retrieved tokens to gen) prefers `_prompt` if a downstream
        # truncation accounting exists, falls back to pool / context total.
        efficiency["E7_retrieved_tokens"] = (
            tc.get("read_context_tokens_prompt", 0)
            or tc.get("read_context_tokens", 0)
        )
        efficiency["E8_retrieval_tokens"] = (
            tc.get("read_llm_input", 0) + tc.get("read_llm_output", 0)
        )
        efficiency["E9_retrieval_calls"] = tc.get("read_llm_calls", 0)
        efficiency["E10_mgmt_tokens"] = (
            tc.get("mgmt_llm_input", 0) + tc.get("mgmt_llm_output", 0)
        )
        efficiency["E11_mgmt_calls"] = tc.get("mgmt_llm_calls", 0)
        efficiency["E12_total_tokens"] = (
            efficiency["E5_construct_tokens"]
            + efficiency["E7_retrieved_tokens"]
            + efficiency["E8_retrieval_tokens"]
            + efficiency["E10_mgmt_tokens"]
        )
        efficiency["E13_total_calls"] = (
            efficiency["E6_construct_calls"]
            + efficiency["E9_retrieval_calls"]
            + efficiency["E11_mgmt_calls"]
        )

    # Internal memory (parametric / KV compression): E3/E4/E7 are not
    # meaningful — memory ops are fused with model forward pass.
    # Only E2 (total inference time) is valid.
    # Internal memory (M9/M10/M11: parametric / full context / KV compression).
    # E3-E11 are not meaningful — no separate write/read/retrieval phases.
    # Only E1 (storage) and E2 (end-to-end inference time) are valid.
    # E3-E13: N/A for internal memory (M9/M10/M11)
    if base_id in ("M9", "M10", "M11", "C8", "C11"):
        # C10/C11 track read_context_tokens (prompt fed to LLM), so E7 is meaningful.
        skip_prefixes = ("E3_", "E4_", "E5", "E6", "E8", "E9", "E10", "E11", "E12", "E13")
        if base_id not in ("M10", "M11", "C11"):
            skip_prefixes = skip_prefixes + ("E7",)
        for k in list(efficiency.keys()):
            if k.startswith(skip_prefixes):
                efficiency[k] = None
        mem_stats["read_latency_ms"] = {"mean": None, "p50": None, "p90": None}
        mem_stats["write_latency_ms"] = {"mean": None, "p50": None, "p90": None}

    # P4/P5: N/A only for parametric (C8/C9) — no retrieved entries at all.
    # C10/C11/C12 have dia_ids tracking so P4/P5 are meaningful.
    if base_id in ("M9", "C8"):
        mem_stats["_p4p5_na"] = True

    # E14: Total wall time. Two flavors:
    #   actual    = this run's elapsed wall time (cache hits skip write → low)
    #   amortized = read time + reconstructed write phase from cache
    #               (fair cross-run comparison; cached writes count as if rerun)
    #
    # E15 is E14 / n_queries — i.e. the per-query wall time including the
    # amortised write+management cost. This is the headline "cost per
    # answered query" number used in the main tables. It is NOT the same as
    # E2 (which is read-phase only: read() + generate()). Substrates with a
    # heavy write phase (M4 A-Mem evolution, M8 MemSkill clustering) will
    # have E15 ≫ E2/1000; embedding-only substrates (M1/M2) will have
    # E15 ≈ E2/1000. Tables/captions must spell this out so readers don't
    # treat the two as interchangeable.
    n_queries = len(all_prediction_logs) or 1
    e14_actual = round(t_total, 2)

    # Cache-hit amortisation (legacy): adds back wall time recovered from
    # save_state cache hits. Only fires when this run loaded a previously
    # cached write phase. Substrates without a cache hit get +0 here, which
    # historically made cross-substrate E14 comparisons unfair (a substrate
    # that ran twice — second time cached — looked artificially "slower"
    # than one that ran once).
    e14_cache_amortized = round(t_total + cache_write_wall_recovered_sec, 2)

    # Symmetric write-phase imputation: independent of cache hits, charge
    # every substrate (E3_mean × n_writes) for its write phase in the headline
    # E14 number. This is the apples-to-apples per-query cost.
    #   - When this run actually executed the writes, t_total already
    #     contains them, so we replace the contained portion rather than
    #     double-count: t_total already has read+write for live runs, so
    #     `read_wall_only` ≈ t_total - actual_write_wall (≈ E3 × n_writes).
    #   - When the writes were cache-restored (live wall ≈ read only), we
    #     impute write_wall = E3 × n_writes on top of t_total.
    write_lat_ms = (mem_stats.get("write_latency_ms") or {}).get("mean") or 0.0
    n_writes_actual = mem_stats.get("write_call_count") or len(all_write_logs) or 0
    imputed_write_sec = (write_lat_ms / 1000.0) * n_writes_actual
    if cache_write_wall_recovered_sec > 0:
        # Live wall already excludes the cached writes — add imputation back.
        e14_imputed = round(t_total + imputed_write_sec, 2)
    else:
        # Live wall already contains the writes; treat actual as authoritative.
        e14_imputed = e14_actual
    efficiency["E14_total_time_sec"] = e14_imputed
    efficiency["E14_actual_wall_sec"] = e14_actual
    efficiency["E14_cache_amortized_sec"] = e14_cache_amortized  # legacy, kept
    efficiency["E14_cache_recovered_write_sec"] = round(cache_write_wall_recovered_sec, 2)
    efficiency["E14_imputed_write_sec"] = round(imputed_write_sec, 2)
    efficiency["E15_avg_time_per_query_sec"] = round(e14_imputed / n_queries, 4)
    efficiency["E15_actual_per_query_sec"] = round(e14_actual / n_queries, 4)

    # Save E stats immediately (in case Q crashes)
    run_meta = {
        "memory_id": memory_id,
        "seed": seed,
        "elapsed_sec": round(t_total, 2),
        "num_predictions": len(predictions),
        "memory_unit_stats": mem_stats,
        "efficiency": efficiency,
        "quality": {},
    }
    save_json(run_meta, os.path.join(rd, "06_run_meta.json"))

    # ── Stage 6b: Memory quality diagnostics (Q family) ──
    if not skip_q:
        # Q1 (text-token compression ratio) and Q2 (entry-embedding entropy)
        # are only meaningful for substrates that store *textual* entries.
        # Parametric methods (C8/C9 — adapter/edited weights), full-context
        # buffers (C10 — buffer of raw utterances; Q1 trivially 1) and
        # KV-cache compressors (C11/C12 — episode centroids in latent space)
        # have no informative text-compression number, so we report n/a in
        # the cross-substrate tables to avoid placeholder garbage.
        if base_id in ("M9", "M10", "M11", "C8", "C11"):
            quality = {"Q1_compression_ratio": None, "Q2_coverage_diversity": None}
        else:
            quality = MemoryQualityMetrics.compute(
                memory_unit,
                original_token_count=total_original_tokens,
                tokenizer_encode=model.encode,
            )
            # Cross-validate Q1 against the per-sample snapshot (which captures
            # the actual stored bytes including LLM-generated notes/entities/
            # summaries). Substrate `get_all_entries()` implementations
            # historically under-reported the stored size for M4 (A-Mem) and
            # M5 (LightRAG) because they only exposed raw chunks, not the
            # full augmented store; the snapshot does not have this gap.
            try:
                snap_files = sorted(
                    f for f in os.listdir(rd)
                    if f.startswith("06_memory_snapshot_sample") and f.endswith(".json")
                )
                stored_tokens_snap = 0
                input_tokens_snap = 0
                for sf in snap_files:
                    snap = load_json(os.path.join(rd, sf))
                    for e in snap.get("entries", []):
                        c = e.get("content") or ""
                        if c:
                            stored_tokens_snap += len(model.encode(c))
                    # Pair with same-sample writes for input-token count
                    si = sf.replace("06_memory_snapshot_sample", "").replace(".json", "")
                    si = si.split("_")[0]
                    wf = os.path.join(rd, f"02_writes_sample{si}.json")
                    if os.path.isfile(wf):
                        ws = load_json(wf)
                        for e in ws.get("entries", []):
                            t = e.get("text") or ""
                            if t:
                                input_tokens_snap += len(model.encode(t))
                if stored_tokens_snap > 0 and input_tokens_snap > 0:
                    quality["Q1_compression_ratio_snapshot"] = round(
                        input_tokens_snap / stored_tokens_snap, 4
                    )
                    # If the wrapper-based Q1 disagrees by >20% with the
                    # snapshot-based Q1, prefer the snapshot — it sees the
                    # whole augmented store.
                    q1_w = quality.get("Q1_compression_ratio")
                    q1_s = quality["Q1_compression_ratio_snapshot"]
                    if q1_w is None or abs(q1_w - q1_s) / max(q1_s, 1e-6) > 0.20:
                        quality["Q1_compression_ratio"] = q1_s
            except Exception as _e:
                pass
        _q1 = quality.get('Q1_compression_ratio')
        _q2 = quality.get('Q2_coverage_diversity')
        print(f"  Q1 Compression Ratio:   {_q1:.2f}" if _q1 is not None else "  Q1 Compression Ratio:   -")
        print(f"  Q2 Coverage Diversity:      {_q2:.4f}" if _q2 is not None else "  Q2 Coverage Diversity:      -")

        # Update run_meta with Q results
        run_meta["quality"] = quality
        save_json(run_meta, os.path.join(rd, "06_run_meta.json"))

    print(f"  Inference done in {t_total:.1f}s — {len(predictions)} predictions")
    print(f"  All stages saved to {rd}/")
    return run_meta


# ────────────────────────────────────────────────────────────
#  Stage 7: Evaluate from saved predictions
# ────────────────────────────────────────────────────────────
def _compute_q_from_texts(entries_text, original_token_count, model, encoder_model,
                          cached_embeddings=None):
    """Compute Q1/Q2 from raw text entries without reinstantiating memory unit.

    Used for stateful libraries (Graphiti/Letta) where replaying write() is
    prohibitively slow. Q1 = original_tokens / sum(encode(text)); Q2 clusters
    entry embeddings + Shannon entropy.

    Args:
        cached_embeddings: optional pre-computed ndarray of shape [N, dim].
            If provided (and length matches ``entries_text``), Q2 reuses it
            and avoids any OpenAI embedding call.
    """
    import math
    from collections import Counter

    results = {"Q1_compression_ratio": float("nan"), "Q2_coverage_diversity": 0.0}
    if not entries_text:
        return results

    stored_tokens = sum(len(model.encode(t)) for t in entries_text)
    if stored_tokens > 0 and original_token_count > 0:
        results["Q1_compression_ratio"] = original_token_count / stored_tokens

    # Q2: cluster embeddings + Shannon entropy. Prefer write-time cached
    # vectors so we don't pay for embedding twice.
    if len(entries_text) < 2:
        return results
    import numpy as _np
    X = None
    if cached_embeddings is not None:
        try:
            arr = _np.asarray(cached_embeddings)
            if arr.ndim == 2 and arr.shape[0] >= len(entries_text):
                X = arr[: len(entries_text)]
        except Exception:
            X = None
    try:
        if X is None:
            import os as _os
            from openai import OpenAI
            client = OpenAI(
                api_key=_os.environ.get("OPENAI_API_KEY", "dummy"),
                base_url=_os.environ.get("OPENAI_BASE_URL", None),
            )
            vecs = []
            for i in range(0, len(entries_text), 128):
                chunk = entries_text[i:i + 128]
                resp = client.embeddings.create(model=encoder_model, input=chunk)
                vecs.extend([d.embedding for d in resp.data])
            X = _np.array(vecs, dtype=_np.float32)
    except Exception as e:
        print(f"  WARN: direct Q2 embed failed ({e}); using TF-IDF fallback")
        from collections import defaultdict
        vocab = defaultdict(int)
        doc_tokens = []
        for t in entries_text:
            toks = t.lower().split()
            doc_tokens.append(toks)
            for w in set(toks):
                vocab[w] += 1
        vocab_list = sorted(vocab.keys())
        vocab_idx = {w: i for i, w in enumerate(vocab_list)}
        import numpy as _np
        X = _np.zeros((len(entries_text), len(vocab_list)))
        for di, toks in enumerate(doc_tokens):
            tf = Counter(toks)
            for w, c in tf.items():
                if w in vocab_idx:
                    idf = math.log(len(entries_text) / (1 + vocab[w]))
                    X[di, vocab_idx[w]] = c * idf

    k = min(5, len(X))
    if k < 2:
        return results
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=3, random_state=42)
        labels = km.fit_predict(X)
    except Exception:
        import numpy as _np
        labels = _np.random.RandomState(42).randint(0, k, size=len(X))
    counts = Counter(labels.tolist() if hasattr(labels, 'tolist') else labels)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    results["Q2_coverage_diversity"] = ent
    return results


def _rebuild_memory_and_compute_q(memory_id: str, run_dir: str, model,
                                  top_k: Optional[int] = None,
                                  encoder_model: str = "text-embedding-3-small") -> dict:
    """Rebuild memory unit from saved write logs and compute Q1-Q4 offline.

    For Q metrics we only need entries + token counts — no embedding needed.
    We populate _entries directly to avoid API calls (e.g. OpenAI embedding
    for C1) that are unnecessary and can fail.
    """
    from SubMem.memory import create_memory_unit
    from SubMem.memory.base import MemoryEntry

    rd = _memory_dir(run_dir, memory_id)

    # Find all write log files
    write_files = sorted(
        f for f in os.listdir(rd)
        if f.startswith("02_writes_sample") and f.endswith(".json")
    )
    if not write_files:
        print(f"  No write logs found for Q metrics")
        return {}

    # Compute original token count from write logs (raw input tokens).
    total_original_tokens = 0
    for wf in write_files:
        write_log = load_json(os.path.join(rd, wf))
        for entry in write_log.get("entries", []):
            total_original_tokens += len(model.encode(entry["text"]))

    # ── Fast-path for stateful libraries (C3/C6): if snapshot has entries,
    #    compute Q metrics directly from text without instantiating the memory
    #    unit. Avoids double-running Letta/Graphiti add loops which take
    #    hours to reprocess. Also works when snapshot is empty but write logs
    #    exist — we treat written text as stored text (Q1≈1, Q2 over raw).
    snapshot_files_direct = sorted(
        f for f in os.listdir(rd)
        if f.startswith("06_memory_snapshot_") and f.endswith(".json")
    )
    direct_entries_text: List[str] = []
    cached_embs = None
    if snapshot_files_direct:
        snap_path = os.path.join(rd, snapshot_files_direct[-1])
        snap = load_json(snap_path)
        for e in snap.get("entries", []):
            t = e.get("content") or e.get("text") or ""
            if t:
                direct_entries_text.append(t)
        # Sidecar embeddings written at inference time — avoids re-embedding.
        emb_path = snap_path.replace(".json", "_embeddings.npy")
        if os.path.exists(emb_path):
            try:
                import numpy as _np
                cached_embs = _np.load(emb_path)
            except Exception:
                cached_embs = None
    stateful_units = ("C3-Letta", "C6", "C6-Graphiti")
    if memory_id in stateful_units:
        if not direct_entries_text:
            # Snapshot empty — fall back to write-log text so we still produce
            # Q numbers (will reflect raw writes, not compressed store).
            for wf in write_files:
                wl = load_json(os.path.join(rd, wf))
                for e in wl.get("entries", []):
                    t = e.get("text") or ""
                    if t:
                        direct_entries_text.append(t)
        quality = _compute_q_from_texts(
            direct_entries_text, total_original_tokens, model, encoder_model,
            cached_embeddings=cached_embs,
        )
        meta_file = os.path.join(rd, "06_run_meta.json")
        if os.path.exists(meta_file):
            run_meta = load_json(meta_file)
            run_meta["quality"] = quality
            save_json(run_meta, meta_file)
        _q1 = quality.get('Q1_compression_ratio')
        _q2 = quality.get('Q2_coverage_diversity')
        print(f"  Q1 Compression Ratio:   {_q1:.2f}" if _q1 is not None else "  Q1 Compression Ratio:   -")
        print(f"  Q2 Coverage Diversity:      {_q2:.4f}" if _q2 is not None else "  Q2 Coverage Diversity:      -")
        return quality

    # Create memory unit with local model (default replay path)
    mem_config = MemoryConfig(memory_id=memory_id, top_k=top_k, extra={"encoder_model": encoder_model})
    memory_unit = create_memory_unit(mem_config, model)

    # Populate _entries from memory snapshot if available (preserves gist /
    # compressed representations for external-memory substrates).  Fall back to write logs
    # for simple memory units where stored == written (C1/C2).
    snapshot_files = sorted(
        f for f in os.listdir(rd)
        if f.startswith("06_memory_snapshot_") and f.endswith(".json")
    )
    if snapshot_files and hasattr(memory_unit, '_episodes'):
        # KV-cache episodic units (e.g. C12): restore _episodes from snapshot.
        snapshot = load_json(os.path.join(rd, snapshot_files[-1]))
        memory_unit._episodes.clear()
        for entry in snapshot.get("entries", []):
            meta = entry.get("metadata", {})
            ep_id = meta.get("episode_id", len(memory_unit._episodes))
            memory_unit._episodes[ep_id] = {
                "selected_text": entry["content"],
                "dia_ids": meta.get("dia_ids", []),
                "centroid": None,
                "selected_indices": [],
                "medoid": "",
            }
    elif snapshot_files and hasattr(memory_unit, '_entries'):
        # Use the last snapshot (last sample's state)
        snapshot = load_json(os.path.join(rd, snapshot_files[-1]))
        memory_unit._entries.clear()
        for entry in snapshot.get("entries", []):
            memory_unit._entries.append(MemoryEntry(
                content=entry["content"],
                metadata=entry.get("metadata", {}),
                timestamp=len(memory_unit._entries),
            ))
    else:
        # Fallback: replay from write logs (works for C1/C2 where stored == written)
        for wf in write_files:
            write_log = load_json(os.path.join(rd, wf))
            if hasattr(memory_unit, '_entries'):
                memory_unit._entries.clear()
            for entry in write_log.get("entries", []):
                text = entry["text"]
                if hasattr(memory_unit, '_entries'):
                    memory_unit._entries.append(MemoryEntry(
                        content=text,
                        metadata=entry.get("metadata", {}),
                        timestamp=len(memory_unit._entries),
                    ))
                else:
                    # Fallback for memory units without _entries (e.g. C8/C9)
                    memory_unit.write(text, metadata=entry.get("metadata", {}))

    # Extract temporal pairs for Q2 (need sessions from benchmark)
    # Load data summary to reconstruct sessions
    quality = MemoryQualityMetrics.compute(
        memory_unit,
        original_token_count=total_original_tokens,
        tokenizer_encode=model.encode,
    )

    _q1 = quality.get('Q1_compression_ratio')
    _q2 = quality.get('Q2_coverage_diversity')
    print(f"  Q1 Compression Ratio:   {_q1:.2f}" if _q1 is not None else "  Q1 Compression Ratio:   -")
    print(f"  Q2 Coverage Diversity:      {_q2:.4f}" if _q2 is not None else "  Q2 Coverage Diversity:      -")

    # Update run_meta with Q results
    meta_file = os.path.join(rd, "06_run_meta.json")
    if os.path.exists(meta_file):
        run_meta = load_json(meta_file)
        run_meta["quality"] = quality
        save_json(run_meta, meta_file)

    return quality


def evaluate_predictions(memory_id: str, run_dir: str, seed: int = 42,
                         nli_scorer: "NLIScorer" = None,
                         p3_model=None, q_model=None,
                         top_k: Optional[int] = None,
                         encoder_model: str = "text-embedding-3-small",
                         benchmark_name: str = None) -> dict:
    """Evaluate from saved predictions.

    Args:
        nli_scorer: optional loaded NLIScorer for P5 faithfulness.
            If None, P5 is skipped.
        p3_model: accepted for backwards compatibility but ignored.
            Policy: P5 uses rule-based claim decomposition only (no LLM).
        q_model: optional local model for offline Q1-Q2 computation.
            If provided, rebuilds memory from write logs and computes Q metrics.
    """
    rd = _memory_dir(run_dir, memory_id)
    pred_file = os.path.join(rd, "05_predictions.json")
    base_id = to_modern_id(memory_id.split("-")[0])  # "M10-3K"/"C10-3K" -> "M10"

    print(f"\n{'='*60}")
    print(f"  {memory_id} | seed={seed} | EVALUATE")
    print(f"{'='*60}")

    if not os.path.exists(pred_file):
        print(f"  ERROR: {pred_file} not found. Run inference first.")
        return {}

    raw_preds = load_json(pred_file)
    predictions = [
        PredictionResult(
            query_id=r["query_id"],
            query=r["query"],
            prediction=r["prediction"],
            gold_answer=str(r["gold_answer"]),
            query_type=r["query_type"],
            metadata={"sample_id": r.get("sample_id", "")},
        )
        for r in raw_preds
    ]
    print(f"  Loaded {len(predictions)} predictions")

    # Load run meta
    meta_file = os.path.join(rd, "06_run_meta.json")
    run_meta = load_json(meta_file) if os.path.exists(meta_file) else {}

    # ── Detect benchmark from run_dir name if not provided ──
    if benchmark_name is None:
        run_basename = os.path.basename(run_dir)
        if "longmemeval" in run_basename:
            benchmark_name = "longmemeval"
        else:
            benchmark_name = "locomo"

    # ── Benchmark-specific metrics ──
    benchmark = create_benchmark(BenchmarkConfig(name=benchmark_name))
    bench_metrics = benchmark.evaluate(predictions)

    # ── Task Performance (P1, P2) ──
    task_perf = TaskPerformanceMetrics.compute(predictions)

    # ── P4 LLM-as-a-Judge ──
    p4_results = None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            judge_model = os.environ.get("SUBMEM_JUDGE_MODEL", "gpt-4o-mini")
            judge = LLMJudge(model=judge_model, benchmark=benchmark_name)
            p4_results = compute_llm_judge_score(predictions, judge)
            task_perf["P4_llm_judge"] = p4_results["P4_llm_judge"]
            save_json(p4_results, os.path.join(rd, "08_llm_judge.json"))
            print(f"  P4 LLM-Judge:   {p4_results['P4_llm_judge']:.4f} ({p4_results['num_queries']} queries)")
        except Exception as e:
            print(f"  P4 LLM-Judge:   SKIPPED ({e})")

    # ── P5 Faithfulness (NLI) ──
    p5_results = None
    if nli_scorer is not None:
        # Parametric / KV-compression substrates have no meaningful
        # retrieved_contents to act as NLI premise, so P5 is N/A.
        if base_id in ("M9", "M10", "M11", "C11"):
            task_perf["P5_faithfulness"] = None
            print(f"  P5: N/A (parametric/KV substrate)")
        else:
            p5_scores = []
            p5_details = []
            for r in raw_preds:
                claims = r.get("claims")
                sources = r.get("retrieved_contents")
                if not claims:
                    claims = decompose_claims_rule(r["prediction"])
                # Short extractive answers (e.g. "May 7, 2023", "Melanie") lack
                # a proposition → NLI gives "neutral" regardless of premise,
                # making P5 useless for short-form QA. Rebuild by prepending
                # the question as context so the claim becomes a full statement.
                if claims and r.get("query"):
                    rebuilt = []
                    for c in claims:
                        wc = len(c.split())
                        if wc < 5:
                            rebuilt.append(f"{r['query'].rstrip('?.!').strip()}: {c}")
                        else:
                            rebuilt.append(c)
                    claims = rebuilt
                if sources is None:
                    sources = []
                res = compute_faithfulness_nli_score(claims, sources, nli_scorer)
                p5_scores.append(res["p5_score"])
                p5_details.append({
                    "query_id": r["query_id"],
                    **res,
                })
            p5_mean = float(np.mean(p5_scores)) if p5_scores else 0.0
            task_perf["P5_faithfulness"] = round(p5_mean, 4)
            p5_results = {
                "P5_faithfulness": round(p5_mean, 4),
                "num_queries": len(p5_scores),
                "per_query": p5_details,
            }
            save_json(p5_results, os.path.join(rd, "09_faithfulness.json"))
            print(f"  P5 faithfulness: {p5_mean:.4f} ({len(p5_scores)} queries)")

    # ── P6/P7 Retrieval Quality ──
    ret_file = os.path.join(rd, "03_retrievals.json")
    if os.path.exists(ret_file):
        # Parametric / KV-compression substrates have no meaningful retrieval
        # set — either no retrieved entries (C9 LoRA) or the entire compressed
        # context (C10/C11/C12), so P6/P7 over flat dia_id sets is misleading.
        # Mark as N/A.
        if base_id in ("M9", "M10", "M11", "C11"):
            task_perf["P6_recall_at_k"] = None
            task_perf["P7_precision_at_k"] = None
            print(f"  P6/P7: N/A (parametric/KV substrate)")
        else:
            ret_logs = load_json(ret_file)
            evidence_map = {r["query_id"]: r.get("gold_evidence", []) for r in ret_logs}
            rq = RetrievalQualityMetrics.compute(ret_logs, evidence_map)
            task_perf["P6_recall_at_k"] = rq.get("P6_recall_at_k", 0.0)
            task_perf["P7_precision_at_k"] = rq.get("P7_precision_at_k", 0.0)
            print(f"  P6 Recall@k:    {task_perf['P6_recall_at_k']:.4f}")
            print(f"  P7 Precision@k: {task_perf['P7_precision_at_k']:.4f}")

    # ── R family: Robustness ──
    robustness = {}
    distractor_queries = RobustnessMetrics.extract_distractor_queries(raw_preds)
    if distractor_queries:
        robustness = RobustnessMetrics.compute(
            predictions=predictions,
            distractor_queries=distractor_queries,
        )
        if "R2_drr" in robustness:
            print(f"  R2 DRR:         {robustness['R2_drr']:.4f} ({len(distractor_queries)} adversarial queries)")

    # ── D family: Behavioral Dynamics ──
    dynamics = {}

    # D3: Primacy-Recency Bias
    has_position = any(r.get("evidence_position", 0) > 0 for r in raw_preds)
    if has_position:
        # Rebuild predictions with position metadata for D3
        preds_with_pos = [
            PredictionResult(
                query_id=r["query_id"],
                query=r["query"],
                prediction=r["prediction"],
                gold_answer=str(r["gold_answer"]),
                query_type=r["query_type"],
                metadata={"position": r.get("evidence_position", 0)},
            )
            for r in raw_preds
            if r.get("evidence_position", 0) > 0
        ]
        if preds_with_pos:
            dynamics["D1_recency_bias"] = BehavioralDynamicsMetrics.compute_d3_from_predictions(preds_with_pos)
            print(f"  D1 Recency Bias: {dynamics['D1_recency_bias']:+.4f} ({'recency' if dynamics['D1_recency_bias'] > 0 else 'primacy'} bias)")

    # ── S family: Scaling Behavior (needs multi-run data, computed externally) ──
    scaling = run_meta.get("scaling", {})

    # ── Memory Efficiency & Quality (from saved stats) ──
    efficiency = run_meta.get("efficiency", {})
    quality = run_meta.get("quality", {})
    mem_stats = run_meta.get("memory_unit_stats", {})

    # ── Offline Q metrics (rebuild memory from write logs) ──
    # Re-run if quality is empty or contains inf/0 values (likely failed)
    q1 = quality.get("Q1_compression_ratio") if quality else None
    # Only rerun offline if Q1 is truly missing/broken. 1.0 is a valid result
    # for substrates that store raw (C1/C2/C10) — rerunning from snapshot
    # (which is raw text) just reproduces 1.0 or breaks if tokenizer differs.
    q_needs_rerun = (not quality) or q1 in (float("inf"), None)
    if q_model is not None and q_needs_rerun:
        try:
            print(f"  Computing Q metrics offline...")
            quality = _rebuild_memory_and_compute_q(memory_id, run_dir, q_model,
                                                      top_k=top_k, encoder_model=encoder_model)
        except Exception as e:
            print(f"  WARNING: Offline Q computation failed: {e}")
            import traceback; traceback.print_exc()

    # ── Per-query-type breakdown ──
    query_types = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]
    query_type_metrics = {}
    for qtype in query_types:
        typed = [p for p in predictions if p.query_type == qtype]
        if typed:
            em_vals = [compute_em(p.prediction, p.gold_answer) for p in typed]
            f1_vals = [compute_f1(p.prediction, p.gold_answer) for p in typed]
            query_type_metrics[qtype] = {
                "count": len(typed),
                "em": float(np.mean(em_vals)),
                "f1": float(np.mean(f1_vals)),
            }

    results = {
        "memory_id": memory_id,
        "seed": seed,
        "num_predictions": len(predictions),
        "elapsed_sec": run_meta.get("elapsed_sec", 0),
        "benchmark_metrics": bench_metrics,
        "task_performance": task_perf,
        "robustness": robustness,
        "efficiency": efficiency,
        "quality": quality,
        "scaling": scaling,
        "dynamics": dynamics,
        "memory_unit_stats": mem_stats,
        "per_query_type": query_type_metrics,
    }

    save_json(results, os.path.join(rd, "07_metrics.json"))

    # ── 09_raw_data.json: per-query data for cross-run aggregation ──
    try:
        per_query_raw = []
        # Index P5 details by query_id if present
        p5_by_qid = {}
        if p5_results:
            for d in p5_results.get("per_query", []):
                p5_by_qid[d["query_id"]] = d.get("p5_score")
        # Index P4 by qid
        p4_by_qid = {}
        if p4_results:
            for d in p4_results.get("per_query", []):
                p4_by_qid[d.get("query_id")] = d.get("score") if "score" in d else d.get("judge_score", 0.0)
        # Index P6/P7 by qid (recompute per-query)
        ret_logs_by_qid = {r["query_id"]: r for r in (raw_retrievals if 'raw_retrievals' in dir() else [])}
        for r in raw_preds:
            qid = r["query_id"]
            entry = {
                "query_id": qid,
                "query_type": r.get("query_type"),
                "gold_answer": r.get("gold_answer"),
                "prediction": r.get("prediction"),
                "claims": r.get("claims") or [],
                "retrieved_contents": r.get("retrieved_contents") or [],
                "evidence_position": r.get("evidence_position", 0),
                "retrieval_latency_ms": r.get("retrieval_latency_ms", 0.0),
                "generation_latency_ms": r.get("generation_latency_ms", 0.0),
                "per_query_metrics": {
                    "P1_em": float(compute_em(r.get("prediction", ""), r.get("gold_answer", ""))),
                    "P2_f1": float(compute_f1(r.get("prediction", ""), r.get("gold_answer", ""))),
                    "P3_bleu1": float(compute_bleu1(r.get("prediction", ""), r.get("gold_answer", ""))),
                    "P4_judge": p4_by_qid.get(qid),
                    "P5_faith": p5_by_qid.get(qid),
                },
            }
            per_query_raw.append(entry)
        raw_data = {
            "schema_version": "v1",
            "memory_id": memory_id,
            "benchmark": benchmark_name,
            "n_queries": len(per_query_raw),
            "n_with_evidence": sum(1 for q in per_query_raw if q["evidence_position"] > 0 or any(q.get("retrieved_contents", []))),
            "n_adversarial": sum(1 for q in per_query_raw if q["query_type"] == "adversarial"),
            "n_non_adversarial": sum(1 for q in per_query_raw if q["query_type"] != "adversarial"),
            "config": results.get("config", {}),
            "per_query": per_query_raw,
        }
        save_json(raw_data, os.path.join(rd, "09_raw_data.json"))
    except Exception as _e:
        print(f"  WARN: failed to write 09_raw_data.json: {_e}")

    # ── Print ──
    print(f"\n  Overall:  EM={bench_metrics.get('em', 0):.4f}  F1={bench_metrics.get('f1', 0):.4f}")

    def _fmt(label, key):
        # Some metrics are None for substrates they don't apply to
        # (e.g. P5 faithfulness on parametric/KV substrates) — skip those.
        v = task_perf.get(key)
        return f"  {label}={v:.4f}" if isinstance(v, (int, float)) else ""

    p3_str = _fmt("P3_bleu1", "P3_bleu1")
    p4_str = _fmt("P4_judge", "P4_llm_judge")
    p5_str = _fmt("P5_faith", "P5_faithfulness")
    print(f"  P1_em={task_perf.get('P1_em', 0):.4f}  "
          f"P2_f1={task_perf.get('P2_f1', 0):.4f}{p3_str}{p4_str}{p5_str}")
    print(f"  E1={efficiency.get('E1_storage_bytes', 0)} bytes  "
          f"E5_construct_tok={efficiency.get('E5_construct_tokens', 0)}  "
          f"E6_construct_calls={efficiency.get('E6_construct_calls', 0)}  "
          f"E7_retrieved_tok={efficiency.get('E7_retrieved_tokens', 0)}  "
          f"E8_retrieval_tok={efficiency.get('E8_retrieval_tokens', 0)}  "
          f"E9_retrieval_calls={efficiency.get('E9_retrieval_calls', 0)}  "
          f"E10_mgmt_tok={efficiency.get('E10_mgmt_tokens', 0)}  "
          f"E11_mgmt_calls={efficiency.get('E11_mgmt_calls', 0)}")
    e2 = {k: efficiency.get(k, 0) for k in ("E2_inference_ms_mean", "E2_inference_ms_p50", "E2_inference_ms_p90")}
    if e2["E2_inference_ms_mean"]:
        print(f"  E2 Inference Latency: mean={e2['E2_inference_ms_mean']:.2f}  p50={e2['E2_inference_ms_p50']:.2f}  p90={e2['E2_inference_ms_p90']:.2f} ms")
    wl = mem_stats.get("write_latency_ms", {})
    if wl and wl.get("mean") is not None:
        n_writes = mem_stats.get("write_call_count", mem_stats.get("num_entries", 0))
        w_tmean = wl.get("trimmed_mean", wl["mean"])
        total_write_s = w_tmean * n_writes / 1000
        print(f"  E3 Write Latency: trimmed_mean={w_tmean:.2f}  p50={wl['p50']:.2f}  p90={wl['p90']:.2f}  mean={wl['mean']:.2f} ms  (total={total_write_s:.1f}s, n={n_writes})")
    rl = mem_stats.get("read_latency_ms", {})
    if rl and rl.get("mean") is not None:
        n_reads = mem_stats.get("token_consumption", {}).get("read_calls", 0)
        tmean = rl.get("trimmed_mean", rl["mean"])
        total_read_s = tmean * n_reads / 1000
        print(f"  E4 Retrieval Latency: trimmed_mean={tmean:.2f}  p50={rl['p50']:.2f}  p90={rl['p90']:.2f}  mean={rl['mean']:.2f} ms  (total={total_read_s:.1f}s, n={n_reads})")
    e15 = efficiency.get("E15_avg_time_per_query_sec")
    if e15 is not None:
        print(f"  E14 Total Time: {efficiency['E14_total_time_sec']}s  |  E15 Avg per query: {e15:.2f}s")
    print(f"  Entries stored: {mem_stats.get('num_entries', 0)}")
    print(f"\n  Per query-type:")
    for qtype, vals in query_type_metrics.items():
        print(f"    {qtype:20s}  n={vals['count']:4d}  EM={vals['em']:.4f}  F1={vals['f1']:.4f}")

    return results


# Display short-names. Keyed by modern M-id for active substrates; old
# C-ids are kept for retired substrates so legacy results still render.
MEMORY_LABELS = {
    "M1": "DenseVec",
    "M2": "SparseVec",
    "M3": "GistIdx",
    "M4": "EvolveNotes",
    "M5": "DualGraph",
    "M6": "HierTree",
    "M7": "DistilStrat",
    "M8": "SkillBundle",
    "M9": "LoRA",
    "M10": "FullCtx",
    "M11": "EpiKV",
    # retired (memory/archived/) — kept for legacy-result rendering
    "C3": "RawBuf",
    "C5": "AtomFact",
    "C6": "TempKG",
    "C8": "KnowEdit",
    "C11": "QueryKV",
}


def print_comparison(all_results: dict):
    """Print comparison table for all memory units that have results."""
    mids = sorted(all_results.keys())
    if len(mids) < 2:
        return

    col_w = 12
    print(f"\n{'='*80}")
    print(f"  COMPARATIVE SUMMARY")
    print(f"{'='*80}")

    # Header
    header = f"{'Metric':36s}" + "".join(
        f" {MEMORY_LABELS.get(m, m):>{col_w}s}" for m in mids
    )
    print(header)
    print("-" * len(header))

    def _val(mid, *keys):
        """Drill into nested result dict."""
        d = all_results.get(mid, {})
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, 0)
            else:
                return 0
        return d

    # ── LOCAL metrics: from a single benchmark run ──
    local_metric_keys = [
        # ── P: Task Performance ──
        ("P1 Exact Match", "task_performance", "P1_em"),
        ("P2 F1", "task_performance", "P2_f1"),
        ("P3 BLEU-1", "task_performance", "P3_bleu1"),
        ("P4 LLM-as-a-Judge", "task_performance", "P4_llm_judge"),
        ("P5 Faithfulness (NLI)", "task_performance", "P5_faithfulness"),
        ("P6 Recall@k", "task_performance", "P6_recall_at_k"),
        ("P7 Precision@k", "task_performance", "P7_precision_at_k"),
        # ── R: Robustness ──
        ("R1 Contradiction Resolution", "robustness", "R1_cra"),
        ("R2 Distractor Rejection", "robustness", "R2_drr"),
        # ── E: Efficiency ──
        ("E1 Storage (bytes)", "efficiency", "E1_storage_bytes"),
        ("E2 Inference Time", "efficiency", "E2_inference_ms_mean"),
        ("E3 Write Latency", "efficiency", "E3_write_latency_ms_mean"),
        ("E4 Retrieval Latency", "efficiency", "E4_retrieval_latency_ms_mean"),
        ("E2 Read Latency mean (ms)", "memory_unit_stats", "read_latency_ms", "mean"),
        ("E2 Read Latency p50 (ms)", "memory_unit_stats", "read_latency_ms", "p50"),
        ("E2 Read Latency trimmed_mean (ms)", "memory_unit_stats", "read_latency_ms", "trimmed_mean"),
        ("E2 Read Latency p90 (ms)", "memory_unit_stats", "read_latency_ms", "p90"),
        ("E3 Write Latency mean (ms)", "memory_unit_stats", "write_latency_ms", "mean"),
        ("E3 Write Latency trimmed_mean (ms)", "memory_unit_stats", "write_latency_ms", "trimmed_mean"),
        ("E3 Write Latency p50 (ms)", "memory_unit_stats", "write_latency_ms", "p50"),
        ("E3 Write Latency p90 (ms)", "memory_unit_stats", "write_latency_ms", "p90"),
        ("E5 Construct Tokens", "efficiency", "E5_construct_tokens"),
        ("E6 Construct Calls", "efficiency", "E6_construct_calls"),
        ("E7 Retrieved Tokens", "efficiency", "E7_retrieved_tokens"),
        ("E8 Retrieval Tokens", "efficiency", "E8_retrieval_tokens"),
        ("E9 Retrieval Calls", "efficiency", "E9_retrieval_calls"),
        ("E10 Management Tokens", "efficiency", "E10_mgmt_tokens"),
        ("E11 Management Calls", "efficiency", "E11_mgmt_calls"),
        ("E12 Total Tokens", "efficiency", "E12_total_tokens"),
        ("E13 Total Calls", "efficiency", "E13_total_calls"),
        ("E14 Total Time (s)", "efficiency", "E14_total_time_sec"),
        ("E15 Avg Time/Query (s)", "efficiency", "E15_avg_time_per_query_sec"),
        # ── Q: Memory Quality ──
        ("Q1 Compression Ratio", "quality", "Q1_compression_ratio"),
        ("Q2 Coverage Diversity", "quality", "Q2_coverage_diversity"),
        # ── D: Behavioral Dynamics ──
        ("D1 Recency Bias", "dynamics", "D1_recency_bias"),
    ]

    # ── GLOBAL metrics: require multiple runs / benchmarks ──
    global_metric_keys = [
        # ── S: Scaling Behavior ──
        ("S1 Session Retention", "scaling", "S1_session_retention"),
        ("S2 Capacity Cliff", "scaling", "S2_capacity_cliff"),
    ]

    other_keys = [
        ("Entries Stored", "memory_unit_stats", "num_entries"),
        ("Write Calls", "memory_unit_stats", "write_call_count"),
        ("Total Time (s)", "elapsed_sec"),
    ]

    metric_keys = local_metric_keys + global_metric_keys + other_keys

    def _print_section(title, rows):
        has_data = False
        for row_def in rows:
            keys = row_def[1:]
            vals = {m: _val(m, *keys) for m in mids}
            if not all(v == 0 or v is None for v in vals.values()):
                has_data = True
                break
        if not has_data:
            return
        print(f"  {title}")
        for row_def in rows:
            label = row_def[0]
            keys = row_def[1:]
            vals = {m: _val(m, *keys) for m in mids}
            if all(v == 0 or v is None for v in vals.values()):
                continue
            parts = f"{label:36s}"
            for m in mids:
                v = vals[m]
                s = f"{v:.4f}" if isinstance(v, float) else str(v)
                parts += f" {s:>{col_w}s}"
            print(parts)

    _print_section("── LOCAL (single benchmark) ──", local_metric_keys)
    print()
    _print_section("── GLOBAL (multi benchmark/run) ──", global_metric_keys)
    print()
    _print_section("── Other ──", other_keys)

    # Per query-type F1
    for qtype in ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]:
        vals = {}
        for m in mids:
            vals[m] = all_results[m].get("per_query_type", {}).get(qtype, {}).get("f1", "-")
        parts = f"{'F1 (' + qtype + ')':36s}"
        for m in mids:
            v = vals[m]
            s = f"{v:.4f}" if isinstance(v, float) else str(v)
            parts += f" {s:>{col_w}s}"
        print(parts)


def main():
    parser = argparse.ArgumentParser(description="LoCoMo substrate test (M*)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, evaluate from saved predictions")
    parser.add_argument("--memory", type=str, nargs="+", default=None,
                        help="Memory unit(s) to run, e.g. --memory M1 M2 M4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-words", type=int, default=None, dest="group_words",
                        help="Group consecutive utterances into ~N-word blocks "
                             "before write() (chunk-size sensitivity sweep). "
                             "Default: per-utterance writes (paper setting).")
    parser.add_argument("--max-samples", type=int, default=3,
                        help="Number of LoCoMo samples (30%% of 10 = 3)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (saves VRAM)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Device map for model loading (default: auto)")
    parser.add_argument("--dataset", type=str, default="locomo",
                        help="Benchmark dataset name")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Existing run directory for --eval-only (auto-detects latest if omitted)")
    parser.add_argument("--no-nli", action="store_true",
                        help="Skip P3 faithfulness (NLI) evaluation")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit total number of queries across all samples")
    parser.add_argument("--no-p3", action="store_true",
                        help="Skip P3 claim decomposition (saves 1 LLM call per query)")
    parser.add_argument("--no-q", action="store_true",
                        help="Skip Q family diagnostics (Q1-Q4)")
    parser.add_argument("--aux-model", type=str, default=None,
                        help="API model for memory unit LLM calls (e.g. openai/gpt-4o-mini). "
                             "Main model still used for answer generation.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Number of retrieved entries per query. Leave "
                             "unset to use each substrate's paper-faithful "
                             "default (e.g. M7 = 1, M1 = 10, M4 = 5). "
                             "Set explicitly to force a single top-k across "
                             "configs for apples-to-apples comparison.")
    parser.add_argument("--encoder-model", type=str, default="text-embedding-3-small",
                        help="OpenAI embedding model for retrieval (default: text-embedding-3-small). "
                             "Options: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002")
    parser.add_argument("--fast", action="store_true",
                        help="Shortcut: sets --no-p3 --no-q (NLI/P5 still runs — it's a local DeBERTa pass, not expensive)")
    parser.add_argument("--lme-variant", type=str, default="s",
                        choices=["s", "m", "oracle"],
                        help="LongMemEval variant: s (~40 sessions), m (~500 sessions), oracle")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable cache lookup/store. Forces fresh memory build.")
    args = parser.parse_args()

    if args.fast:
        args.no_p3 = True
        args.no_q = True

    from SubMem.config import MEMORY_UNIT_IDS as ALL_MEMORY_IDS
    memory_ids = args.memory if args.memory else ALL_MEMORY_IDS
    seed = args.seed

    os.makedirs(RESULTS_BASE, exist_ok=True)

    if args.eval_only:
        # ── Eval-only: find or use specified run dir ──
        if args.run_dir:
            run_dir = args.run_dir
        else:
            run_dir = _find_latest_run_dir()
        print(f"Evaluating from: {run_dir}")

        # Load aux_model for offline P3 claim decomposition
        aux_model = None
        model = None
        if args.aux_model:
            aux_config = ModelConfig(name=args.aux_model, temperature=0.0, max_new_tokens=256)
            aux_model = create_model(aux_config)
            print(f"Aux model: {args.aux_model}")
            aux_model.load()
            print("Aux model loaded.")

        # Load local model for offline Q metrics (rebuild memory from write logs)
        if not args.no_q and args.model:
            max_new = 512 if args.dataset == "longmemeval" else 64
            model_config = ModelConfig(
                name=args.model, dtype="float16", temperature=0.0, max_new_tokens=max_new,
                device_map=args.device_map,
                load_in_4bit=getattr(args, 'load_in_4bit', False),
                load_in_8bit=getattr(args, 'load_in_8bit', False),
            )
            model = create_model(model_config)
            print(f"Loading model for Q metrics: {args.model}")
            model.load()
            print("Model loaded.")
    else:
        # ── Full run: create new timestamped dir ──
        run_dir = _make_run_dir(args.dataset, args.model)
        print(f"Run directory: {run_dir}")

        # ── Load model once ──
        max_new = 512 if args.dataset == "longmemeval" else 64
        model_config = ModelConfig(
            name=args.model, dtype="float16", temperature=0.0, max_new_tokens=max_new,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
        model = create_model(model_config)
        print(f"Model: {args.model}")
        print("Loading model...")
        model.load()
        print("Model loaded.")

        # ── Optional: aux model for memory unit LLM calls ──
        aux_model = None
        if args.aux_model:
            aux_config = ModelConfig(name=args.aux_model, temperature=0.0, max_new_tokens=256)
            aux_model = create_model(aux_config)
            print(f"Aux model: {args.aux_model}")
            aux_model.load()
            print("Aux model loaded.")

        bench_extra = {}
        if args.dataset == "longmemeval" and args.lme_variant != "s":
            bench_extra["variant"] = args.lme_variant
        benchmark_config = BenchmarkConfig(name=args.dataset, max_samples=args.max_samples, extra=bench_extra)
        benchmark, samples, data_records = stage_load_data(benchmark_config)

        # Save stage 1 (shared)
        save_json(data_records, os.path.join(run_dir, "01_data_summary.json"))
        print(f"  Stage 1: {len(samples)} samples, "
              f"{sum(r['num_turns'] for r in data_records)} turns, "
              f"{sum(r['num_queries'] for r in data_records)} queries")

        for mid in memory_ids:
            try:
                run_inference(mid, model, benchmark, samples, run_dir=run_dir, seed=seed,
                             max_queries=args.max_queries,
                             skip_p3=args.no_p3, skip_q=args.no_q,
                             aux_model=aux_model, top_k=args.top_k,
                             encoder_model=args.encoder_model,
                             benchmark_name=args.dataset,
                             use_cache=not args.no_cache,
                             group_words=args.group_words)
            except Exception as e:
                print(f"  ERROR during {mid} inference: {e}")
                import traceback; traceback.print_exc()

    # ── Load NLI scorer for P5 faithfulness ──
    # Policy: only P4 is allowed to call an LLM. P5 claim decomposition
    # always uses the rule-based decomposer (regex split on sentences +
    # conjunctions). NLI scoring itself is a local DeBERTa-v3 forward pass
    # on GPU — no API calls. --no-nli skips P5 entirely.
    nli_scorer = None
    p3_model = None  # intentionally unused; kept for signature compatibility
    if not args.no_nli:
        try:
            print("Loading NLI model for P5 faithfulness...")
            nli_scorer = NLIScorer()
            nli_scorer.load()
            print("NLI model loaded.")
        except Exception as e:
            print(f"  WARNING: Could not load NLI model, skipping P5: {e}")
            nli_scorer = None

    # ── Evaluate (works with or without --eval-only) ──
    all_results = {}
    for mid in memory_ids:
        try:
            res = evaluate_predictions(mid, run_dir=run_dir, seed=seed,
                                       nli_scorer=nli_scorer,
                                       p3_model=p3_model,
                                       q_model=model,
                                       top_k=args.top_k,
                                       encoder_model=args.encoder_model,
                                       benchmark_name=args.dataset)
            if res:
                all_results[mid] = res
        except Exception as e:
            print(f"  ERROR during {mid} evaluation: {e}")
            import traceback; traceback.print_exc()

    # ── Comparison table ──
    print_comparison(all_results)

    # ── Generate examples (need at least 2 units with results) ──
    units_with_results = [m for m in memory_ids if m in all_results]
    if len(units_with_results) >= 2:
        generate_examples(units_with_results, run_dir=run_dir, seed=seed)

    # ── Save combined ──
    combined_path = os.path.join(run_dir, "locomo_test.json")
    save_json(all_results, combined_path)
    print(f"\nAll results saved to {combined_path}")


# ────────────────────────────────────────────────────────────
#  Example generation: per-query comparison across all units
# ────────────────────────────────────────────────────────────
def generate_examples(memory_ids: List[str], run_dir: str, seed: int = 42, num_examples: int = 10):
    """Generate detailed per-query examples comparing all memory units.

    Picks queries where units diverge most, plus some where all agree.
    Saves to <run_dir>/examples/
    """
    print(f"\n{'='*60}")
    print(f"  GENERATING EXAMPLES ({', '.join(memory_ids)})")
    print(f"{'='*60}")

    data = {}
    for mid in memory_ids:
        rd = _memory_dir(run_dir, mid)
        ret_file = os.path.join(rd, "03_retrievals.json")
        pred_file = os.path.join(rd, "05_predictions.json")
        if not os.path.exists(ret_file) or not os.path.exists(pred_file):
            print(f"  Skipping — {mid} data not found")
            return
        data[f'{mid}_ret'] = load_json(ret_file)
        data[f'{mid}_pred'] = load_json(pred_file)

    n = len(data[f'{memory_ids[0]}_pred'])

    # Compute F1 for each (query, unit)
    query_scores = []  # list of (idx, {mid: f1}, query_type)
    for i in range(n):
        scores = {}
        for mid in memory_ids:
            p = data[f'{mid}_pred'][i]
            scores[mid] = compute_f1(p['prediction'], str(p['gold_answer']))
        qt = data[f'{memory_ids[0]}_pred'][i].get('query_type', '')
        query_scores.append((i, scores, qt))

    # Pick interesting examples: max divergence + all-succeed + all-fail
    def divergence(scores_dict):
        vals = list(scores_dict.values())
        return max(vals) - min(vals) if vals else 0

    by_div = sorted(query_scores, key=lambda x: -divergence(x[1]))
    by_avg = sorted(query_scores, key=lambda x: -np.mean(list(x[1].values())))

    picked_indices = set()
    picks = []

    # Top divergent
    for idx, scores, qt in by_div:
        if idx not in picked_indices and divergence(scores) > 0.3:
            picks.append((idx, scores, qt, "high_divergence"))
            picked_indices.add(idx)
        if len(picks) >= num_examples // 2:
            break

    # All succeed (highest avg F1)
    for idx, scores, qt in by_avg:
        if idx not in picked_indices and min(scores.values()) >= 0.5:
            picks.append((idx, scores, qt, "all_succeed"))
            picked_indices.add(idx)
        if len(picks) >= num_examples * 3 // 4:
            break

    # All fail (lowest avg F1)
    for idx, scores, qt in reversed(by_avg):
        if idx not in picked_indices and max(scores.values()) < 0.1:
            picks.append((idx, scores, qt, "all_fail"))
            picked_indices.add(idx)
        if len(picks) >= num_examples:
            break

    # Build examples
    examples = []
    for idx, scores, qt, case in picks:
        ref_pred = data[f'{memory_ids[0]}_pred'][idx]
        ex = {
            "query_index": idx,
            "case": case,
            "query": ref_pred['query'],
            "query_type": qt,
            "gold_answer": ref_pred['gold_answer'],
            "units": {},
        }
        best_mid = max(scores, key=scores.get)
        worst_mid = min(scores, key=scores.get)
        if case == "high_divergence":
            ex["analysis"] = (
                f"{best_mid} ({MEMORY_LABELS.get(best_mid, best_mid)}) scored highest "
                f"(F1={scores[best_mid]:.2f}), while {worst_mid} "
                f"({MEMORY_LABELS.get(worst_mid, worst_mid)}) scored lowest "
                f"(F1={scores[worst_mid]:.2f})."
            )
        elif case == "all_succeed":
            ex["analysis"] = "All methods retrieved relevant context for this query."
        else:
            ex["analysis"] = "No method retrieved sufficient context."

        for mid in memory_ids:
            p = data[f'{mid}_pred'][idx]
            r = data[f'{mid}_ret'][idx]
            em = compute_em(p['prediction'], str(p['gold_answer']))
            ex["units"][mid] = {
                "label": MEMORY_LABELS.get(mid, mid),
                "prediction": p['prediction'],
                "F1": round(scores[mid], 4),
                "EM": round(em, 4),
                "retrieval_latency_ms": p.get('retrieval_latency_ms'),
                "generation_latency_ms": p.get('generation_latency_ms'),
                "top5_retrieved": [
                    {"rank": ri['rank'], "score": round(ri['score'], 4), "content": ri['content']}
                    for ri in r['retrieved'][:5]
                ]
            }
        examples.append(ex)

    # Save
    ex_dir = os.path.join(run_dir, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    fname = f"{'_'.join(memory_ids)}_examples.json"
    save_json(examples, os.path.join(ex_dir, fname))

    # Print summary
    print(f"  Generated {len(examples)} examples:")
    for ex in examples:
        scores_str = "  ".join(f"{m}={ex['units'][m]['F1']:.2f}" for m in memory_ids)
        print(f"    [{ex['query_index']:3d}] {ex['case']:18s} | {ex['query_type']:12s} "
              f"| {scores_str} | Q: {ex['query'][:45]}")
    print(f"  Saved to {ex_dir}/{fname}")


if __name__ == "__main__":
    main()
