"""C18 — LightRAG (HKUDS, EMNLP 2025) memory substrate.

Directly delegates to the ``lightrag`` package vendored at
``third_party/LightRAG``. No re-implementation — we only wrap the
library behind SubMem's ``BaseMemoryUnit`` interface and instrument
LLM calls for E-family metrics.

Pipeline:
  write(text) — buffer until first read (LightRAG's ainsert is
                chunk-oriented; we send all writes at once on build)
  read(query, top_k):
    - On first read, ``ainsert`` all buffered chunks (extract entities
      + relations via LLM, dedupe, build NetworkX graph, embed entries
      into nano-vectordb).
    - ``aquery(query, param=QueryParam(mode='mix'))`` — dual-level
      retrieval: low-level (entity neighborhoods) + high-level (relation
      edges) + vector-chunk fallback, then LLM synthesis.

E-metric instrumentation: monkey-patches
``openai.resources.chat.completions.Completions.create`` to bucket
build-phase tokens into E5/E6 and query-phase tokens into E8/E9.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit


# Ensure LightRAG is importable
_LRAG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "LightRAG",
)
if _LRAG_ROOT not in sys.path:
    sys.path.insert(0, _LRAG_ROOT)


def _run(coro):
    """Run coroutine on a fresh event loop if none exists."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@register_memory_unit("C18")
class LightRAGMemory(BaseMemoryUnit):
    """LightRAG substrate — dual-level KG+vector retrieval."""

    CACHE_KEY_EXTRA = ["lightrag_llm_model", "lightrag_embed_model"]

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._llm_model = extra.get("lightrag_llm_model", "gpt-4o-mini")
        self._embed_model = extra.get("lightrag_embed_model", "text-embedding-3-small")
        self._top_k = int(extra.get("lightrag_top_k", 40))
        self._workspace = extra.get("lightrag_workspace", "")

        self._chunks: List[str] = []
        self._dia_ids: List[str] = []
        self._rag = None
        self._built = False
        self._working_dir: Optional[str] = None

        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._build_time = 0.0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "mgmt_llm_input": 0, "mgmt_llm_output": 0, "mgmt_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        self._last_retrieved_dia_ids: List[str] = []

    def _load_env(self):
        from pathlib import Path
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    def _instrument_openai(self):
        if getattr(self, "_openai_patched", False):
            return
        outer = self

        def _record(resp):
            try:
                usage = getattr(resp, "usage", None)
                if usage is None:
                    return
                bucket = "read" if outer._built else "write"
                outer._token_counts[f"{bucket}_llm_input"] += getattr(usage, "prompt_tokens", 0) or 0
                outer._token_counts[f"{bucket}_llm_output"] += getattr(usage, "completion_tokens", 0) or 0
                outer._token_counts[f"{bucket}_llm_calls"] += 1
            except Exception:
                pass

        # Patch sync path
        try:
            from openai.resources.chat.completions import Completions
            orig_sync = Completions.create

            def tracked_sync(self, *args, **kwargs):
                resp = orig_sync(self, *args, **kwargs)
                _record(resp)
                return resp

            Completions.create = tracked_sync
        except ImportError:
            pass

        # Patch async path (LightRAG uses this via openai_complete_if_cache)
        try:
            from openai.resources.chat.completions import AsyncCompletions
            orig_async = AsyncCompletions.create

            async def tracked_async(self, *args, **kwargs):
                resp = await orig_async(self, *args, **kwargs)
                _record(resp)
                return resp

            AsyncCompletions.create = tracked_async
        except ImportError:
            pass

        self._openai_patched = True

    def _ensure_rag(self):
        if self._rag is not None:
            return
        self._load_env()
        self._instrument_openai()

        real_key = os.environ.get("SUBMEM_REAL_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY", "")
        real_base = os.environ.get("SUBMEM_REAL_OPENAI_BASE", "https://api.openai.com/v1")
        if real_key:
            os.environ["OPENAI_API_KEY"] = real_key
            os.environ["OPENAI_BASE_URL"] = real_base

        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        from lightrag.kg.shared_storage import initialize_pipeline_status

        # Per-PID workdir so concurrent jobs don't clobber
        wd_base = os.environ.get("LIGHTRAG_WORKING_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "submem", "lightrag"
        )
        wd = os.path.join(wd_base, f"pid{os.getpid()}_{int(time.time()*1000)}")
        os.makedirs(wd, exist_ok=True)
        self._working_dir = wd

        async def llm_fn(prompt, system_prompt=None, history_messages=None,
                          keyword_extraction=False, **kwargs):
            return await openai_complete_if_cache(
                self._llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=real_key,
                base_url=real_base,
                **kwargs,
            )

        embed_dim = 1536 if "small" in self._embed_model else 3072

        async def embed_fn(texts):
            return await openai_embed(
                texts,
                model=self._embed_model,
                api_key=real_key,
                base_url=real_base,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=embed_dim,
            func=embed_fn,
        )

        self._rag = LightRAG(
            working_dir=wd,
            llm_model_func=llm_fn,
            embedding_func=embedding_func,
            top_k=self._top_k,
        )
        # LightRAG needs this before first insert/query
        _run(self._rag.initialize_storages())
        _run(initialize_pipeline_status())

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        self._write_call_count += 1
        self._chunks.append(text)
        dia = (metadata or {}).get("dia_id", "")
        self._dia_ids.append(dia)
        self._write_times.append(time.perf_counter() - t0)

    def _ensure_built(self):
        if self._built or not self._chunks:
            return
        self._ensure_rag()
        t0 = time.perf_counter()
        try:
            # Insert all chunks at once; each chunk keeps its source text
            _run(self._rag.ainsert(self._chunks))
        except Exception as e:
            print(f"  [C18] ainsert failed: {e}")
        self._build_time = time.perf_counter() - t0
        if self._write_call_count > 0:
            per = self._build_time / self._write_call_count
            self._write_times = [t + per for t in self._write_times]
        self._built = True
        print(f"  [C18] LightRAG built in {self._build_time:.1f}s over "
              f"{len(self._chunks)} chunks", flush=True)

    def read(self, query: str, top_k: int = 40) -> List[MemoryEntry]:
        self._ensure_built()
        t0 = time.perf_counter()
        self._last_retrieved_dia_ids = []

        from lightrag import QueryParam
        try:
            answer = _run(self._rag.aquery(
                query,
                param=QueryParam(mode="mix", top_k=top_k, include_references=True),
            ))
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            print(f"  [C18] query failed: {e}")
            answer = ""

        # ── dia_id attribution ──
        # LightRAG returns an LLM-synthesised answer plus a `### References`
        # block. The references are themselves LLM-paraphrased and rarely
        # contain the verbatim chunk text, so naive substring matching
        # under-counts hits. We use a layered strategy:
        #   1. Direct substring match on the first 80 chars of each chunk
        #      (cheapest, catches verbatim quoted utterances).
        #   2. Speaker + timestamp match: extract `[<time>] <speaker>:` style
        #      references from the answer (LightRAG's stock format) and
        #      match against chunk metadata.
        #   3. Token-level Jaccard similarity > 0.5 between any reference
        #      bullet and the chunk text (catches paraphrase cases).
        import re

        if answer and self._chunks:
            seen = set()

            # Strategy 1: verbatim substring
            for i, chunk in enumerate(self._chunks):
                if chunk and chunk[:80] in answer:
                    did = self._dia_ids[i] if i < len(self._dia_ids) else ""
                    if did and did not in seen:
                        self._last_retrieved_dia_ids.append(did)
                        seen.add(did)
                        if len(self._last_retrieved_dia_ids) >= top_k:
                            break

            if len(self._last_retrieved_dia_ids) < top_k:
                # Strategy 2/3: parse `### References` block then match
                ref_block = ""
                if "### References" in answer:
                    ref_block = answer.split("### References", 1)[1]
                ref_lines = [
                    ln.strip().lstrip("-*•").strip()
                    for ln in ref_block.splitlines()
                    if ln.strip() and not ln.strip().startswith("#")
                ]

                # Pattern for `[<time>] <speaker>:` (LightRAG default ref body)
                ts_speaker_re = re.compile(
                    r"\[?\s*([0-9]{1,2}[:\s][0-9 apmAPM:.,/-]+[0-9])\s*\]?\s*([A-Z][A-Za-z' ]{1,30}?)\s*[:：]"
                )

                def _norm_tokens(s: str) -> set:
                    return set(re.findall(r"[A-Za-z0-9]+", s.lower()))

                # Pre-extract chunk speaker / timestamp signatures (cheap)
                chunk_sigs = []
                for i, chunk in enumerate(self._chunks):
                    if not chunk:
                        chunk_sigs.append(None); continue
                    head = chunk[:200]
                    m = ts_speaker_re.search(head)
                    if m:
                        chunk_sigs.append(
                            (m.group(1).strip().lower(),
                             m.group(2).strip().lower(),
                             _norm_tokens(chunk))
                        )
                    else:
                        chunk_sigs.append((None, None, _norm_tokens(chunk)))

                for line in ref_lines:
                    if len(self._last_retrieved_dia_ids) >= top_k:
                        break
                    line_tokens = _norm_tokens(line)
                    if not line_tokens:
                        continue

                    m = ts_speaker_re.search(line)
                    line_ts = m.group(1).strip().lower() if m else None
                    line_speaker = m.group(2).strip().lower() if m else None

                    best_i, best_sim = -1, 0.0
                    for i, sig in enumerate(chunk_sigs):
                        if sig is None:
                            continue
                        ts, sp, ctokens = sig
                        # Strategy 2: speaker + timestamp prefix exact match
                        if line_ts and line_speaker and ts and sp:
                            if (line_speaker == sp
                                    and (line_ts in ts or ts in line_ts)):
                                best_i, best_sim = i, 1.0
                                break
                        # Strategy 3: Jaccard
                        if ctokens:
                            inter = len(line_tokens & ctokens)
                            union = len(line_tokens | ctokens)
                            sim = inter / union if union else 0.0
                            if sim > best_sim:
                                best_i, best_sim = i, sim

                    if best_i >= 0 and best_sim >= 0.5:
                        did = self._dia_ids[best_i] if best_i < len(self._dia_ids) else ""
                        if did and did not in seen:
                            self._last_retrieved_dia_ids.append(did)
                            seen.add(did)

        self._token_counts["read_calls"] += 1
        try:
            self._token_counts["read_context_tokens"] += len(self.model.encode(answer))
        except Exception:
            self._token_counts["read_context_tokens"] += max(len(answer) // 4, 0)

        self._read_times.append(time.perf_counter() - t0)

        # Recover the raw chunk text for each attributed dia_id so that the
        # NLI faithfulness scorer (P5) sees the actual underlying utterances
        # rather than LightRAG's paraphrased summary. Mirrors how M3 (ReadAgent)
        # exposes raw retrieved pages even though its store is gisted.
        raw_mems: List[str] = []
        if self._last_retrieved_dia_ids:
            dia_to_chunk = {
                d: self._chunks[i]
                for i, d in enumerate(self._dia_ids)
                if i < len(self._chunks) and d
            }
            for did in self._last_retrieved_dia_ids:
                ch = dia_to_chunk.get(did)
                if ch:
                    raw_mems.append(ch)

        return [MemoryEntry(
            content=answer,
            metadata={
                "source": "lightrag",
                "dia_ids": self._last_retrieved_dia_ids,
                "raw_chunks": raw_mems,
            },
            score=1.0,
        )]

    def reset(self):
        # Finalize + remove workdir
        try:
            if self._rag is not None:
                _run(self._rag.finalize_storages())
        except Exception:
            pass
        if self._working_dir and os.path.isdir(self._working_dir):
            import shutil
            shutil.rmtree(self._working_dir, ignore_errors=True)
        self._rag = None
        self._built = False
        self._chunks.clear()
        self._dia_ids.clear()
        self._working_dir = None
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._build_time = 0.0
        self._last_retrieved_dia_ids = []
        for k in list(self._token_counts):
            self._token_counts[k] = 0

    def size(self) -> int:
        return len(self._chunks)

    def get_all_entries(self) -> List[MemoryEntry]:
        """Dump entity + relation nodes as the compressed store for Q1."""
        if not self._built:
            self._ensure_built()
        out: List[MemoryEntry] = []
        try:
            # LightRAG stores the graph as NetworkX at working_dir/graph_chunk_entity_relation.graphml
            import networkx as nx
            import glob
            paths = glob.glob(os.path.join(self._working_dir or "", "*graph*entity*relation*.graphml"))
            if paths:
                G = nx.read_graphml(paths[0])
                for node, data in G.nodes(data=True):
                    desc = data.get("description", "") or data.get("entity_name", "") or str(node)
                    if desc:
                        out.append(MemoryEntry(content=str(desc), metadata=dict(data)))
                for u, v, data in G.edges(data=True):
                    desc = data.get("description", "")
                    if desc:
                        out.append(MemoryEntry(content=str(desc), metadata=dict(data)))
        except Exception as e:
            print(f"  [C18] get_all_entries (graphml) failed: {e}")
        if not out:
            return [MemoryEntry(content=c) for c in self._chunks]
        return out

    def get_external_storage_bytes(self) -> int:
        return sum(len(e.content.encode("utf-8")) for e in self.get_all_entries())

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        import numpy as _np
        read_ms = [t * 1000 for t in self._read_times]
        write_ms = [t * 1000 for t in self._write_times]
        stats.update({
            "memory_size_bytes": self.get_external_storage_bytes(),
            "num_entries": len(self._chunks),
            "lightrag_llm_model": self._llm_model,
            "lightrag_embed_model": self._embed_model,
            "lightrag_build_time_sec": round(self._build_time, 2),
            "token_consumption": dict(self._token_counts),
            "write_call_count": self._write_call_count,
            "read_latency_ms": {
                "mean": float(_np.mean(read_ms)) if read_ms else 0.0,
                "p50": float(_np.median(read_ms)) if read_ms else 0.0,
                "p90": float(_np.percentile(read_ms, 90)) if read_ms else 0.0,
            },
            "write_latency_ms": {
                "mean": float(_np.mean(write_ms)) if write_ms else 0.0,
                "p50": float(_np.median(write_ms)) if write_ms else 0.0,
                "p90": float(_np.percentile(write_ms, 90)) if write_ms else 0.0,
            },
        })
        return stats
