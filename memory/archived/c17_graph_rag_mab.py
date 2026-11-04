"""C17 — Graph-RAG via MemoryAgentBench (HUST-AI-HYZ) adapter.

Ported from the ``_handle_graph_rag`` pathway in MAB's agent.py, which
delegates to the ``GraphRAG`` class in ``methods/graph_rag.py``.

Pipeline (Microsoft Graph-RAG style, via MAB's vendored implementation):
  write(text) — buffer chunks
  read(query, top_k):
    - On first read (or after context_id switch), process all buffered
      chunks: build FAISS vector store + KnowledgeGraph (entity + concept
      extraction via LLM) + QueryEngine.
    - Query = graph-aware RAG retrieval: FAISS top-k seeds + graph
      traversal expansion, then LLM synthesis over expanded context.

E-metric instrumentation wraps the LLM client so every write-time LLM
call (entity/concept extraction) lands in E5/E6 and read-time LLM calls
(answer synthesis, graph-assisted checks) land in E8/E9.

Requires third_party/MemoryAgentBench on sys.path.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit


# Make MAB's methods/ importable
_MAB_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "MemoryAgentBench",
)
if _MAB_ROOT not in sys.path:
    sys.path.insert(0, _MAB_ROOT)


@register_memory_unit("C17")
class GraphRAGMemory(BaseMemoryUnit):
    """Graph-RAG substrate ported from HUST-AI-HYZ/MemoryAgentBench.

    Note: upstream GraphRAG builds the whole knowledge graph in ``process_documents``
    which is a one-shot operation; we defer it to the first read() so multiple
    writes just accumulate chunks. That matches MAB's ``_handle_graph_rag``
    behaviour (only processes docs when context_id changes).
    """

    CACHE_KEY_EXTRA = ["graphrag_model", "graphrag_max_tokens"]

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._model_name = extra.get("graphrag_model", "gpt-4o-mini")
        self._temperature = float(extra.get("graphrag_temperature", 0.7))
        self._max_tokens = int(extra.get("graphrag_max_tokens", 512))
        self._retrieve_num = int(extra.get("graphrag_retrieve_num", 5))

        self._chunks: List[str] = []
        self._dia_ids: List[str] = []
        self._graph_rag = None  # Lazy init
        self._processed = False

        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._build_time: float = 0.0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "mgmt_llm_input": 0, "mgmt_llm_output": 0, "mgmt_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        self._last_retrieved_dia_ids: List[str] = []

    def _load_env(self):
        """Load OPENAI_API_KEY from repo .env if not already set."""
        from pathlib import Path
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    def _instrument_openai(self):
        """Monkey-patch openai.resources.chat.completions.Completions.create
        so we can attribute tokens to write vs read phase."""
        if getattr(self, "_openai_patched", False):
            return
        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            return
        orig = Completions.create
        outer = self

        def tracked(self, *args, **kwargs):
            resp = orig(self, *args, **kwargs)
            try:
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    # _processed flag flips to True after build completes.
                    # Before = write (E5/E6); after = read (E8/E9).
                    bucket = "read" if outer._processed else "write"
                    outer._token_counts[f"{bucket}_llm_input"] += getattr(usage, "prompt_tokens", 0) or 0
                    outer._token_counts[f"{bucket}_llm_output"] += getattr(usage, "completion_tokens", 0) or 0
                    outer._token_counts[f"{bucket}_llm_calls"] += 1
            except Exception:
                pass
            return resp

        Completions.create = tracked
        self._openai_patched = True

    def _ensure_client(self):
        if self._graph_rag is not None:
            return
        self._load_env()
        self._instrument_openai()
        # Route to real OpenAI creds if runner is behind vLLM
        real_key = os.environ.get("SUBMEM_REAL_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY", "")
        real_base = os.environ.get("SUBMEM_REAL_OPENAI_BASE", "https://api.openai.com/v1")
        if real_key:
            os.environ["OPENAI_API_KEY"] = real_key
            os.environ["OPENAI_BASE_URL"] = real_base

        from methods.graph_rag import GraphRAG
        self._graph_rag = GraphRAG(
            temperature=self._temperature,
            model_name=self._model_name,
            retrieve_num=self._retrieve_num,
            max_tokens=self._max_tokens,
        )

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        self._write_call_count += 1
        self._chunks.append(text)
        dia_id = (metadata or {}).get("dia_id", "")
        self._dia_ids.append(dia_id)
        self._write_times.append(time.perf_counter() - t0)

    def _build_graph(self):
        """One-shot: pass all accumulated chunks to GraphRAG.process_documents.
        Builds FAISS + KnowledgeGraph + QueryEngine."""
        if self._processed or not self._chunks:
            return
        self._ensure_client()
        from langchain_core.documents import Document

        docs = [
            Document(page_content=t, metadata={"source": "submem", "chunk_id": i,
                                                "dia_id": self._dia_ids[i] if i < len(self._dia_ids) else ""})
            for i, t in enumerate(self._chunks)
        ]
        t0 = time.perf_counter()
        try:
            self._graph_rag.process_documents(docs)
        except Exception as e:
            print(f"  [C17] build graph failed: {e}")
        self._build_time = time.perf_counter() - t0
        # amortise build cost across writes so per-write latency is sensible
        if self._write_call_count > 0:
            per = self._build_time / self._write_call_count
            self._write_times = [t + per for t in self._write_times]
        self._processed = True
        print(f"  [C17] graph built in {self._build_time:.1f}s over {len(self._chunks)} chunks", flush=True)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        self._build_graph()
        t0 = time.perf_counter()
        self._last_retrieved_dia_ids = []

        try:
            # upstream returns (final_answer, expanded_context)
            answer, retrieval_context = self._graph_rag.query(query=query)
            if not isinstance(answer, str):
                answer = getattr(answer, "content", str(answer))
        except Exception as e:
            print(f"  [C17] read error: {e}")
            answer = ""
            retrieval_context = ""

        # Best-effort dia_id attribution: scan retrieval_context for chunk text
        # that matches our buffered chunks (exact substring) and grab their dia_ids
        if retrieval_context and self._chunks:
            for i, chunk in enumerate(self._chunks):
                if chunk and chunk[:80] in retrieval_context:
                    did = self._dia_ids[i] if i < len(self._dia_ids) else ""
                    if did and did not in self._last_retrieved_dia_ids:
                        self._last_retrieved_dia_ids.append(did)
                        if len(self._last_retrieved_dia_ids) >= top_k:
                            break

        self._token_counts["read_calls"] += 1
        if retrieval_context:
            # approximate context tokens (chars/4 if no tokenizer handy)
            try:
                self._token_counts["read_context_tokens"] += len(self.model.encode(retrieval_context))
            except Exception:
                self._token_counts["read_context_tokens"] += max(len(retrieval_context) // 4, 0)

        self._read_times.append(time.perf_counter() - t0)

        # Parametric-style: return the generated answer as the single entry,
        # but include raw retrieval_context in metadata so the runner's
        # downstream P5/P6 computation can use it as NLI premise.
        return [MemoryEntry(
            content=answer,
            metadata={
                "source": "graph_rag_mab",
                "dia_ids": self._last_retrieved_dia_ids,
                "retrieval_context": retrieval_context,
            },
            score=1.0,
        )]

    def reset(self):
        self._chunks.clear()
        self._dia_ids.clear()
        self._graph_rag = None
        self._processed = False
        self._build_time = 0.0
        self._last_retrieved_dia_ids = []
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        for k in list(self._token_counts):
            self._token_counts[k] = 0

    def size(self) -> int:
        return len(self._chunks)

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return the compressed store: entity concepts + edges from the
        knowledge graph, not the raw chunks. Used by Q1 compression.
        """
        if not self._processed:
            self._build_graph()
        out: List[MemoryEntry] = []
        try:
            kg = getattr(self._graph_rag, "knowledge_graph", None)
            if kg is not None and hasattr(kg, "graph"):
                # nodes carry chunk content + extracted concepts
                import networkx as nx
                g = kg.graph
                for node, data in g.nodes(data=True):
                    content = data.get("content", "") or str(node)
                    if content:
                        out.append(MemoryEntry(content=content, metadata=dict(data)))
        except Exception as e:
            print(f"  [C17] get_all_entries failed: {e}")
        if not out:
            # fallback to raw chunks
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
            "graphrag_model": self._model_name,
            "graphrag_build_time_sec": round(self._build_time, 2),
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
