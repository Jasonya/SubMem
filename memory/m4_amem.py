"""M4 — A-Mem (Agentic Memory) — Xu et al. 2025.

Reference:
  "A-MEM: Agentic Memory for LLM Agents" (Xu et al., 2025)
  Upstream paper reference code (first-author): https://github.com/WujiangXu/AgenticMemory
  Library fork (clean re-impl, paper-faithful): https://github.com/agiresearch/A-mem

This substrate **delegates to the official** ``agentic_memory.memory_system``
library from agiresearch/A-mem. We do not re-implement the evolution prompt
or the note schema — all logic lives in the upstream ``AgenticMemorySystem``
class (evolution prompt, content analysis, link management, etc.).

SubMem integration:
  - We instantiate ``AgenticMemorySystem`` with:
      * ``model_name`` = OpenAI ``text-embedding-3-small`` equivalent via
        sentence-transformers default (all-MiniLM-L6-v2) *unless* user sets
        ``extra.amem_encoder_model`` in the config to override.
      * ``llm_backend='openai'`` + ``llm_model='gpt-4o-mini'`` by default,
        so the evolution / content-analysis calls route through OpenAI.
  - write(text) → ``system.add_note(content=text)``
  - read(query, top_k) → ``system.search_agentic(query, k=top_k)``
    (paper-faithful retrieval: ChromaDB nearest neighbors + link expansion
    into evolution-built memory graph; both zero LLM calls)
  - We track read/write latency and token costs by monkey-patching the
    library's ``LLMController.get_completion`` to tap into the OpenAI
    response usage before returning.
  - Answer-time generation is done by the SubMem runner with its own model
    (Qwen3-8B); this substrate only provides retrieved notes.

Notable: the official library uses ChromaDB as the vector store. We set the
collection name per-instance so parallel tests don't collide, and the
ChromaDB persistent client goes to a tmp dir unique to this run.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit


@register_memory_unit("M4")
class AMemMemory(BaseMemoryUnit):
    """Thin wrapper around agiresearch/A-mem's ``AgenticMemorySystem``."""

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = config.extra if hasattr(config, "extra") and config.extra else {}

        self._encoder_name = extra.get("amem_encoder_model", "all-MiniLM-L6-v2")
        self._llm_backend = extra.get("amem_llm_backend", "openai")
        self._llm_model = extra.get("amem_llm_model", "gpt-4o-mini")
        self._evo_threshold = int(extra.get("amem_evo_threshold", 100))
        # Unique Chroma dir so concurrent test runs don't collide.
        self._chroma_dir = tempfile.mkdtemp(prefix=f"amem_c16_{os.getpid()}_")

        # Point ChromaDB persistent client at our unique dir via env.
        os.environ.setdefault("CHROMA_TELEMETRY", "False")

        self._system = None  # Lazy-init to allow early stats queries.

        # Stats (MemoryEntry-compatible)
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._note_dia_id: Dict[str, str] = {}
        self._token_counts: Dict[str, int] = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "mgmt_llm_input": 0, "mgmt_llm_output": 0, "mgmt_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }

    # ── Lazy init so importing this module doesn't force Chroma setup ──
    def _ensure_system(self):
        if self._system is not None:
            return
        import chromadb  # noqa: F401 — used by library
        from agentic_memory.memory_system import AgenticMemorySystem

        # Let ChromaDB persist under our unique dir.
        os.environ["CHROMA_PERSIST_DIR"] = self._chroma_dir

        self._system = AgenticMemorySystem(
            model_name=self._encoder_name,
            llm_backend=self._llm_backend,
            llm_model=self._llm_model,
            evo_threshold=self._evo_threshold,
        )
        self._patch_llm_controller()

    def _patch_llm_controller(self):
        """Monkey-patch A-Mem's inner OpenAI client to count tokens.

        memory_system.py calls ``self.llm_controller.llm.get_completion``
        (the inner OpenAIController), NOT the outer ``LLMController``'s
        delegator. We patch the inner one so every write/evolve call
        lands in our counters.
        """
        lc = self._system.llm_controller
        inner = lc.llm  # OpenAIController
        if getattr(inner, "_submem_patched", False):
            return
        orig = inner.get_completion

        def tracked(prompt: str, response_format: Optional[Dict] = None, temperature: float = 0.7):
            # Strict mgmt definition: only count as mgmt when the LLM
            # response actually emits an `update_neighbor` action — that
            # rewrites existing memories' tags/context (true merge). The
            # `strengthen` action only adds links to the new note itself
            # (still construct). All other write-time calls (analyze_content)
            # are also construct.
            result = orig(prompt, response_format=response_format, temperature=temperature)
            in_tok = max(len(prompt) // 4, 0)
            out_tok = max(len(result or "") // 4, 0)
            is_mgmt = False
            try:
                import json as _json
                rj = _json.loads(result or "{}") if result else {}
                actions = rj.get("actions", [])
                if rj.get("should_evolve") and "update_neighbor" in actions:
                    is_mgmt = True
            except Exception:
                # Fallback heuristic: substring check on raw response
                is_mgmt = '"update_neighbor"' in (result or "")
            if is_mgmt:
                self._token_counts["mgmt_llm_input"] += in_tok
                self._token_counts["mgmt_llm_output"] += out_tok
                self._token_counts["mgmt_llm_calls"] += 1
            else:
                self._token_counts["write_llm_input"] += in_tok
                self._token_counts["write_llm_output"] += out_tok
                self._token_counts["write_llm_calls"] += 1
            return result

        inner.get_completion = tracked
        inner._submem_patched = True
        # Also patch the outer delegator for anyone who calls that form
        lc.get_completion = tracked
        lc._submem_patched = True

    # ── BaseMemoryUnit API ──
    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._ensure_system()
        t0 = time.perf_counter()
        note_id = self._system.add_note(content=text)
        self._write_times.append(time.perf_counter() - t0)
        self._write_call_count += 1
        if metadata and isinstance(metadata, dict):
            dia_id = metadata.get("dia_id", "")
            if dia_id:
                self._note_dia_id[note_id] = dia_id
            # Keep the full turn-level metadata so retrieval-quality metrics
            # (P5/P6/P7) can resolve has_answer / session_id / role.
            if not hasattr(self, "_note_meta"):
                self._note_meta = {}
            self._note_meta[note_id] = dict(metadata)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        self._ensure_system()
        t0 = time.perf_counter()
        # Use search_agentic (paper's recommended entry-point, per A-Mem
        # README) so we actually exercise the evolution-built link graph.
        # The plain `.search()` we used before ignored links, defeating the
        # whole purpose of doing evolution during write.
        hits = self._system.search_agentic(query, k=top_k)
        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1

        results: List[MemoryEntry] = []
        for h in hits or []:
            content = h.get("content", "") if isinstance(h, dict) else ""
            nid = h.get("id", "") if isinstance(h, dict) else ""
            dia_id = self._note_dia_id.get(nid, "")
            orig_meta = (self._note_meta.get(nid, {}) if hasattr(self, "_note_meta") else {})
            meta = {
                "id": nid,
                "dia_id": dia_id,
                "tags": h.get("tags", []) if isinstance(h, dict) else [],
                "context": h.get("context", "") if isinstance(h, dict) else "",
                # Forward turn-level fields needed by P5/P6/P7 substring/match.
                "has_answer": orig_meta.get("has_answer"),
                "session_id": orig_meta.get("session_id"),
                "role": orig_meta.get("role"),
            }
            score = 1.0 - float(h.get("score", 0.0)) if isinstance(h, dict) else 1.0
            if content:
                results.append(MemoryEntry(content=content, metadata=meta, score=score))
                self._token_counts["read_context_tokens"] += max(
                    len(self.model.encode(content)), 0
                )
        return results

    def reset(self):
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._note_dia_id.clear()
        for k in list(self._token_counts):
            self._token_counts[k] = 0
        # Tear down and re-init on next write
        self._system = None
        # New chroma dir so we start fresh
        self._chroma_dir = tempfile.mkdtemp(prefix=f"amem_c16_{os.getpid()}_")

    def size(self) -> int:
        if self._system is None:
            return 0
        return len(getattr(self._system, "memories", {}))

    # ── cache support ─────────────────────────────────────────
    CACHE_KEY_EXTRA = ["amem_model", "amem_encoder_model", "amem_evolution"]

    def save_state(self, dir_path: str) -> None:
        """Cache the ChromaDB dir + dia_id mapping + token counts.
        AgenticMemorySystem itself is not pickled; on load we re-init it
        pointing to the cached chroma dir."""
        import os, json, shutil
        os.makedirs(dir_path, exist_ok=True)
        # Copy chroma dir as-is
        cached_chroma = os.path.join(dir_path, "chroma")
        if os.path.exists(cached_chroma):
            shutil.rmtree(cached_chroma)
        if os.path.isdir(self._chroma_dir):
            shutil.copytree(self._chroma_dir, cached_chroma)
        # Pickle the in-memory `memories` dict via the library's API
        if self._system is not None:
            mems = {}
            for nid, n in getattr(self._system, "memories", {}).items():
                mems[nid] = {
                    "id": n.id, "content": n.content,
                    "keywords": list(getattr(n, "keywords", []) or []),
                    "tags": list(getattr(n, "tags", []) or []),
                    "context": getattr(n, "context", ""),
                    "links": list(getattr(n, "links", []) or []),
                    "timestamp": getattr(n, "timestamp", None),
                    "evolution_history": list(getattr(n, "evolution_history", []) or []),
                }
            with open(os.path.join(dir_path, "memories.json"), "w") as f:
                json.dump(mems, f, default=str)
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            json.dump({
                "note_dia_id": self._note_dia_id,
                "token_counts": dict(self._token_counts),
                "write_call_count": self._write_call_count,
                "write_times": list(self._write_times),
            }, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        import os, json, shutil
        unit = cls(config, model)
        # Replace temp chroma with cached one
        cached_chroma = os.path.join(dir_path, "chroma")
        if os.path.isdir(cached_chroma):
            if os.path.exists(unit._chroma_dir):
                shutil.rmtree(unit._chroma_dir)
            shutil.copytree(cached_chroma, unit._chroma_dir)
        # Init system pointing at restored chroma
        unit._ensure_system()
        # Re-attach memories dict if json present
        mems_path = os.path.join(dir_path, "memories.json")
        if os.path.exists(mems_path):
            try:
                from agentic_memory.memory_system import MemoryNote
                with open(mems_path) as f:
                    mems = json.load(f)
                for nid, m in mems.items():
                    note = MemoryNote(content=m["content"])
                    note.id = m["id"]
                    note.keywords = m.get("keywords", [])
                    note.tags = m.get("tags", [])
                    note.context = m.get("context", "")
                    note.links = m.get("links", [])
                    note.timestamp = m.get("timestamp")
                    note.evolution_history = m.get("evolution_history", [])
                    unit._system.memories[nid] = note
            except Exception as e:
                print(f"  [M4 cache] memories restore partial: {e}")
        with open(os.path.join(dir_path, "meta.json")) as f:
            d = json.load(f)
        unit._note_dia_id = d["note_dia_id"]
        unit._token_counts = d["token_counts"]
        unit._write_call_count = d["write_call_count"]
        unit._write_times = d["write_times"]
        return unit

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_system()
        stats = super().get_stats()
        read_ms = [t * 1000 for t in self._read_times]
        write_ms = [t * 1000 for t in self._write_times]
        n_notes = len(getattr(self._system, "memories", {}))
        # Rough storage size: sum of note content bytes + simple metadata
        size_bytes = 0
        for note in getattr(self._system, "memories", {}).values():
            content = getattr(note, "content", "") or ""
            size_bytes += len(content.encode("utf-8"))
        stats.update({
            "memory_size_bytes": size_bytes,
            "num_entries": n_notes,
            "evolution_count": int(getattr(self._system, "evo_cnt", 0)),
            "evo_threshold": self._evo_threshold,
            "amem_encoder": self._encoder_name,
            "amem_llm_backend": self._llm_backend,
            "amem_llm_model": self._llm_model,
            "token_consumption": dict(self._token_counts),
            "write_call_count": self._write_call_count,
            "read_latency_ms": {
                "mean": float(np.mean(read_ms)) if read_ms else 0.0,
                "p50": float(np.median(read_ms)) if read_ms else 0.0,
                "p90": float(np.percentile(read_ms, 90)) if read_ms else 0.0,
            },
            "write_latency_ms": {
                "mean": float(np.mean(write_ms)) if write_ms else 0.0,
                "p50": float(np.median(write_ms)) if write_ms else 0.0,
                "p90": float(np.percentile(write_ms, 90)) if write_ms else 0.0,
            },
        })
        return stats

    def get_external_storage_bytes(self) -> int:
        self._ensure_system()
        size_bytes = 0
        for note in getattr(self._system, "memories", {}).values():
            content = getattr(note, "content", "") or ""
            size_bytes += len(content.encode("utf-8"))
        return size_bytes

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return the full A-Mem note (content + keywords + tags + context)
        as the stored representation for Q1 compression accounting.
        """
        self._ensure_system()
        out: List[MemoryEntry] = []
        for nid, note in getattr(self._system, "memories", {}).items():
            content = getattr(note, "content", "") or ""
            if not content:
                continue
            keywords = getattr(note, "keywords", []) or []
            tags = getattr(note, "tags", []) or []
            context = getattr(note, "context", "") or ""
            # Full stored representation: note + all evolved metadata.
            parts = [content]
            if context:
                parts.append(f"Context: {context}")
            if keywords:
                parts.append("Keywords: " + ", ".join(keywords))
            if tags:
                parts.append("Tags: " + ", ".join(tags))
            out.append(MemoryEntry(
                content="\n".join(parts),
                metadata={
                    "id": nid,
                    "dia_id": self._note_dia_id.get(nid, ""),
                },
                score=1.0,
            ))
        return out
