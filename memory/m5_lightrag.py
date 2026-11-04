"""M5 — LightRAG (Guo+ '24): dual-level graph-based RAG.

Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation"
       (Guo, Xia, Yu, Ao, Huang — 2410.05779, arXiv 2024)
Upstream: https://github.com/HKUDS/LightRAG (cloned at third_party/LightRAG).

This substrate **delegates to the cloned upstream** package — the full
indexing / dual-level retrieval / incremental-update logic lives in
``third_party/LightRAG/lightrag``. We add ``third_party/LightRAG`` to
``sys.path`` at import time so no pip install is required.

Pipeline (LightRAG native):
  write() → buffer text → ``rag.ainsert([texts...])`` on first read.
            LLM extracts entities + relations; graph + entity/relation
            vector stores are built under working_dir (local NVMe).
  read()  → ``rag.aquery_data(q, QueryParam(mode=...))`` returns
            structured ``{entities, relationships, chunks}`` —
            retrieve-only, no answer LLM call. We flatten each into
            one MemoryEntry so SubMem's runner can do its own gen step
            with the experiment's generator model.

Indexing LLM requirements (upstream CLAUDE.md):
  LightRAG's entity/relation extraction prompt expects strict
  structured output AND ships with long few-shot examples (~15-25K
  input tokens/call). Upstream recommends ≥32B parameters and ≥32K
  context. In practice **gpt-4o-mini** is the only sub-32B model that
  hits the format-compliance bar reliably — small open-source models
  (Qwen3-8B, Llama-3.1-8B) produce malformed extractions that the
  parser drops. Also avoid reasoning models (o1 / DeepSeek-R1 / QwQ)
  during indexing: their <think> tokens break the parser.

Configuration via env (mirroring c6_graphiti):
  LIGHTRAG_LLM_BASE_URL   : OpenAI-compatible chat endpoint (unset → api.openai.com)
  LIGHTRAG_LLM_MODEL      : Chat model name (default: gpt-4o-mini — do NOT
                            swap for a smaller open-source model without
                            verifying extraction quality)
  LIGHTRAG_EMBED_BASE_URL : Embedding endpoint (default: None → OpenAI)
  LIGHTRAG_EMBED_MODEL    : Embedding model (default: text-embedding-3-small)
  LIGHTRAG_EMBED_DIM      : Embedding dimension (default: 1536)
  LIGHTRAG_WORKING_DIR    : Working dir root (default: $TMPDIR/lightrag_*)
  M5_MODE                : Query mode — one of
                            {naive, local, global, hybrid, mix} (default: mix).
                            Paper's recommended default is mix (KG + vector).
  M5_MAX_PARALLEL_INSERT : Insert concurrency (default: 2, upstream cap: 10)
  M5_TOP_K               : top_k override for QueryParam (default: from config)

Notes
-----
- The upstream CLAUDE.md warns that changing embedding model requires
  clearing the vector DB. We key the working_dir on pid+time so every
  run gets a fresh graph; cache save_state/load_state copies the whole
  working_dir directory (entities/relations/chunks JSONs + graphml).
- Explicitly pass ``base_url`` to OpenAI helpers rather than relying on
  env-based resolution (see feedback_openai_sdk_base_url: passing
  ``base_url=None`` falls through to ``OPENAI_BASE_URL`` which breaks
  the dual-endpoint setup of vLLM chat + OpenAI embeddings).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit

# ── Make cloned upstream importable ────────────────────────────
_SUBMEM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIGHTRAG_SRC = os.path.join(_SUBMEM_ROOT, "third_party", "LightRAG")
if _LIGHTRAG_SRC not in sys.path:
    sys.path.insert(0, _LIGHTRAG_SRC)


# ── Persistent background event loop ───────────────────────────
# LightRAG's storage backends (nano-vectordb, networkx graphml) hold
# state tied to the asyncio loop that first touched them. Same design
# constraint as c6_graphiti — use ONE long-lived loop on a daemon
# thread and dispatch coroutines onto it.

_PERSISTENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
_PERSISTENT_LOOP_THREAD: Optional[threading.Thread] = None


def _get_persistent_loop() -> asyncio.AbstractEventLoop:
    global _PERSISTENT_LOOP, _PERSISTENT_LOOP_THREAD
    if _PERSISTENT_LOOP is not None and not _PERSISTENT_LOOP.is_closed():
        return _PERSISTENT_LOOP
    _PERSISTENT_LOOP = asyncio.new_event_loop()

    def _run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _PERSISTENT_LOOP_THREAD = threading.Thread(
        target=_run_loop, args=(_PERSISTENT_LOOP,), daemon=True,
    )
    _PERSISTENT_LOOP_THREAD.start()
    return _PERSISTENT_LOOP


def _run(coro):
    loop = _get_persistent_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


@register_memory_unit("M5")
class LightRAGMemory(BaseMemoryUnit):
    """Dual-level graph-based RAG (LightRAG, Guo+ '24).

    write(): buffers text for lazy batch insertion.
    read():  runs ``aquery_data`` with env-selected mode; returns one
             MemoryEntry per retrieved entity / relationship / chunk.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = getattr(config, "extra", {}) or {}

        # Endpoints — feedback_openai_sdk_base_url: pass explicit URLs.
        self._llm_base_url = (
            extra.get("lightrag_llm_base_url")
            or os.environ.get("LIGHTRAG_LLM_BASE_URL")
        )
        self._llm_model = (
            extra.get("lightrag_llm_model")
            or os.environ.get("LIGHTRAG_LLM_MODEL", "gpt-4o-mini")
        )
        self._embed_base_url = (
            extra.get("lightrag_embed_base_url")
            or os.environ.get("LIGHTRAG_EMBED_BASE_URL")
        )
        self._embed_model = (
            extra.get("lightrag_embed_model")
            or os.environ.get("LIGHTRAG_EMBED_MODEL", "text-embedding-3-small")
        )
        self._embed_dim = int(
            extra.get("lightrag_embed_dim")
            or os.environ.get("LIGHTRAG_EMBED_DIM", "1536")
        )
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._llm_api_key = os.environ.get("LIGHTRAG_LLM_API_KEY", self._api_key)

        self._mode = (extra.get("m5_mode") or os.environ.get("M5_MODE", "mix")).lower()
        if self._mode not in {"naive", "local", "global", "hybrid", "mix"}:
            raise ValueError(
                f"M5_MODE={self._mode!r} invalid; must be one of "
                "naive/local/global/hybrid/mix"
            )
        self._max_parallel_insert = int(
            extra.get("m5_max_parallel_insert")
            or os.environ.get("M5_MAX_PARALLEL_INSERT", "2")
        )

        # working_dir handling:
        #   - If LIGHTRAG_WORKING_DIR points at a dir that's empty OR
        #     already contains a LightRAG graph (graph_chunk_entity_
        #     relation.graphml), use it directly. Empty dir = "stage dir"
        #     prepared by run_c18_vllm_1conv.sh; populated dir = cache HIT.
        #   - Only fall back to a nested pid+ts subdir if the env points
        #     at a generic shared root (e.g. /tmp without a job-specific
        #     subdir) AND that root already has unrelated files in it.
        #   - Same NFS caveat as C6: avoid putting working_dir on NFS for
        #     huge graphs. nano-vectordb uses explicit json.dump (no mmap),
        #     so the C6 page-cache-OOM mechanism does not apply.
        _wd_env = (
            extra.get("lightrag_working_dir")
            or os.environ.get("LIGHTRAG_WORKING_DIR")
            or tempfile.gettempdir()
        )
        os.makedirs(_wd_env, exist_ok=True)
        _existing_marker = os.path.join(_wd_env, "graph_chunk_entity_relation.graphml")
        if os.path.isfile(_existing_marker):
            # Cache HIT — reuse the populated dir as-is.
            self._working_dir = _wd_env
            self._reused_working_dir = True
            print(f"  [LightRAG] Reusing existing working_dir: {self._working_dir}")
        elif not os.listdir(_wd_env):
            # Empty dir handed to us by the caller (typical: STAGE_DIR
            # from run_c18_vllm_1conv.sh) — write directly into it so the
            # archive step finds the artifacts at the documented path.
            self._working_dir = _wd_env
            self._reused_working_dir = False
            print(f"  [LightRAG] Using empty working_dir as-is: {self._working_dir}")
        else:
            # Generic shared root with unrelated content — nest under a
            # unique subdir to avoid colliding with anything already there.
            self._working_dir = os.path.join(
                _wd_env, f"lightrag_{os.getpid()}_{int(time.time())}"
            )
            os.makedirs(self._working_dir, exist_ok=True)
            self._reused_working_dir = False

        # P6/P7 evidence matching: LightRAG entry metadata exposes
        # `source_id = "chunk-<md5>"`, but LoCoMo gold evidence is in
        # `dia_id` form (e.g. "D1:5"). We track the mapping at write
        # time so read() can backfill `metadata['dia_ids']` per entry.
        # Layered:
        #   text → doc_id  ("doc-" + md5(text))   — deterministic from input
        #   chunk_id → doc_id                      — read from kv_store_text_chunks.json
        #   doc_id → dia_id(s)                     — populated in write()
        self._doc_id_to_dia_ids: Dict[str, List[str]] = {}
        self._chunk_to_doc_cache: Optional[Dict[str, str]] = None

        # Latency / token tracking.
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts: Dict[str, int] = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "mgmt_llm_input": 0, "mgmt_llm_output": 0, "mgmt_llm_calls": 0,
            "embed_tokens": 0, "embed_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        self._total_writes = 0
        self._phase = "write"

        self._pending_texts: List[str] = []
        self._indexed = False
        self._rag = None
        self._init_rag()

    # ── LLM / embedding closures ─────────────────────────────

    def _make_llm_func(self):
        """Build the llm_model_func LightRAG expects.

        Signature required by lightrag.operate: ``async (prompt, system_prompt=None,
        history_messages=[], keyword_extraction=False, **kwargs) -> str``.
        """
        from lightrag.llm.openai import openai_complete_if_cache
        base_url = self._llm_base_url
        api_key = self._llm_api_key or "dummy"
        model = self._llm_model
        token_counts = self._token_counts
        get_phase = lambda: self._phase  # noqa: E731 — 1-line closure

        async def _llm(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            history = history_messages or []
            # Strip LightRAG-internal kwargs that tenacity/openai SDK chokes on.
            kwargs.pop("hashing_kv", None)
            kwargs.pop("stream", None)
            resp = await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history,
                base_url=base_url,
                api_key=api_key,
                keyword_extraction=keyword_extraction,
                **kwargs,
            )
            # Rough token estimate — upstream's openai_complete_if_cache
            # doesn't expose usage via its public return value, so we
            # approximate by char/4 (same fallback c6_graphiti uses when
            # a provider omits the usage field).
            phase = get_phase()
            inp_est = (len(prompt) + len(system_prompt or "") + sum(
                len(m.get("content", "") or "") for m in history
            )) // 4
            out_est = len(resp) // 4
            # Tag mgmt vs construct on write path. LightRAG's
            # summarize_entity_descriptions is the de-duplicating merge call
            # (multiple descriptions of the same entity collapsed into one);
            # everything else on the write path is initial entity/relation
            # extraction (construct).
            sys_str = system_prompt or ""
            is_mgmt = (
                "summarize" in sys_str.lower()
                or "summarize_entity" in (prompt or "")
                or "given a list" in sys_str.lower() and "description" in sys_str.lower()
            )
            if phase == "read":
                token_counts["read_llm_input"] += inp_est
                token_counts["read_llm_output"] += out_est
                token_counts["read_llm_calls"] += 1
            elif is_mgmt:
                token_counts["mgmt_llm_input"] += inp_est
                token_counts["mgmt_llm_output"] += out_est
                token_counts["mgmt_llm_calls"] += 1
            else:
                token_counts["write_llm_input"] += inp_est
                token_counts["write_llm_output"] += out_est
                token_counts["write_llm_calls"] += 1
            return resp

        return _llm

    def _make_embed_func(self):
        """Build EmbeddingFunc with correct dim/max_token_size."""
        from lightrag.llm.openai import openai_embed
        from lightrag.utils import EmbeddingFunc

        base_url = self._embed_base_url
        api_key = self._api_key or "dummy"
        model = self._embed_model
        token_counts = self._token_counts

        # Use partial on the UNDERLYING func (.func) to avoid double
        # EmbeddingFunc wrapping — the upstream CLAUDE.md flags this
        # as issue #3.
        async def _embed(texts):
            t_texts = texts if isinstance(texts, list) else list(texts)
            result = await openai_embed.func(
                t_texts,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            token_counts["embed_tokens"] += sum(len(t) for t in t_texts) // 4
            token_counts["embed_calls"] += 1
            return result

        return EmbeddingFunc(
            embedding_dim=self._embed_dim,
            max_token_size=8192,
            func=_embed,
        )

    def _init_rag(self):
        from lightrag import LightRAG

        llm_func = self._make_llm_func()
        embed_func = self._make_embed_func()

        self._rag = LightRAG(
            working_dir=self._working_dir,
            llm_model_func=llm_func,
            embedding_func=embed_func,
            max_parallel_insert=self._max_parallel_insert,
        )
        # Required — CLAUDE.md issue #1.
        _run(self._rag.initialize_storages())
        print(
            f"  [LightRAG] Initialized (llm={self._llm_model}, "
            f"embed={self._embed_model}, mode={self._mode}, "
            f"dir={self._working_dir})"
        )

    # ── BaseMemoryUnit interface ──────────────────────────────

    @staticmethod
    def _doc_id_for(text: str) -> str:
        """Replicate LightRAG's doc-id formula (utils.compute_mdhash_id).
        Used to build the (doc_id → dia_ids) map at write time without
        needing a roundtrip through LightRAG state."""
        import hashlib
        return "doc-" + hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_chunk_to_doc_map(self) -> Dict[str, str]:
        """Lazy-load chunk_id → full_doc_id from kv_store_text_chunks.json.
        Refreshes the cache so newly-indexed docs are visible."""
        p = os.path.join(self._working_dir, "kv_store_text_chunks.json")
        if not os.path.isfile(p):
            return {}
        try:
            with open(p, "r") as f:
                data = json.load(f) or {}
            return {ck: rec.get("full_doc_id", "") for ck, rec in data.items()}
        except Exception:
            return {}

    def _resolve_dia_ids(self, source_id_field: Any) -> List[str]:
        """Map a LightRAG entry's source_id field (chunk-id, possibly
        comma/`<SEP>`-separated for multi-source entities) → list of
        LoCoMo dia_ids via chunk→doc→dia_id."""
        if not source_id_field:
            return []
        # source_id can be a single chunk-id or multiple separated by
        # `<SEP>` or comma (LightRAG storage convention varies).
        if isinstance(source_id_field, str):
            # Common separators in LightRAG: <SEP>, ',', ';'
            tokens = []
            for sep in ("<SEP>", ",", ";"):
                if sep in source_id_field:
                    tokens = source_id_field.split(sep)
                    break
            if not tokens:
                tokens = [source_id_field]
        elif isinstance(source_id_field, (list, tuple)):
            tokens = list(source_id_field)
        else:
            return []
        chunk_to_doc = self._chunk_to_doc_cache or {}
        seen = set()
        out: List[str] = []
        for raw in tokens:
            ck = (raw or "").strip()
            if not ck:
                continue
            doc_id = chunk_to_doc.get(ck, "")
            for dia in self._doc_id_to_dia_ids.get(doc_id, []):
                if dia and dia not in seen:
                    seen.add(dia)
                    out.append(dia)
        return out

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Buffer text for batched insertion on first read."""
        self._write_call_count += 1
        self._pending_texts.append(text)
        # Record the dia_id that this text contributes to, keyed on the
        # deterministic LightRAG doc-id. Multiple writes with the same
        # text (deduplicated by LightRAG) map all dia_ids to one doc.
        dia = (metadata or {}).get("dia_id", "")
        if dia:
            doc_id = self._doc_id_for(text)
            self._doc_id_to_dia_ids.setdefault(doc_id, [])
            if dia not in self._doc_id_to_dia_ids[doc_id]:
                self._doc_id_to_dia_ids[doc_id].append(dia)
        self._indexed = False

    def _existing_doc_count(self) -> int:
        """Count records in kv_store_full_docs.json — used to detect a
        reused/cached working_dir that's already indexed."""
        p = os.path.join(self._working_dir, "kv_store_full_docs.json")
        if not os.path.isfile(p):
            return 0
        try:
            with open(p, "r") as f:
                return len(json.load(f) or {})
        except Exception:
            return 0

    def _ensure_indexed(self):
        if self._indexed:
            return
        # If this is a reused working_dir with all writes already indexed,
        # skip ainsert entirely. We check len(_pending_texts) <= existing
        # doc count rather than equality so partial-buffered scenarios
        # (e.g., user calls read() then writes more) still re-index when
        # the buffer outgrows the cache.
        existing = self._existing_doc_count()
        if self._reused_working_dir and existing >= len(self._pending_texts) > 0:
            print(
                f"  [LightRAG] working_dir already has {existing} docs ≥ "
                f"{len(self._pending_texts)} buffered — SKIP indexing",
                flush=True,
            )
            self._total_writes += len(self._pending_texts)
            self._pending_texts = []
            self._indexed = True
            return
        pending = self._pending_texts
        if not pending:
            self._indexed = True
            return
        self._phase = "write"
        t0 = time.perf_counter()
        print(
            f"    [LightRAG] inserting {len(pending)} docs "
            f"(max_parallel={self._max_parallel_insert})",
            flush=True,
        )
        try:
            _run(self._rag.ainsert(list(pending)))
        except Exception as e:
            print(f"  [LightRAG] ainsert error: {e}")
        dt = time.perf_counter() - t0
        per = dt / max(len(pending), 1)
        self._write_times.extend([per] * len(pending))
        self._total_writes += len(pending)
        self._pending_texts = []
        self._indexed = True
        # Force chunk→doc map reload on next read (text_chunks.json
        # was just rewritten by ainsert).
        self._chunk_to_doc_cache = None
        print(
            f"    [LightRAG] insert done in {dt:.1f}s ({per:.2f}s/doc)",
            flush=True,
        )

    def read(self, query: str, top_k: int = 10) -> List[MemoryEntry]:
        """Dual-level retrieval via aquery_data (no answer LLM call)."""
        from lightrag import QueryParam

        self._ensure_indexed()
        # Refresh chunk→doc map after indexing (or after a reused dir
        # is detected — kv_store_text_chunks.json is now populated).
        if self._chunk_to_doc_cache is None or not self._chunk_to_doc_cache:
            self._chunk_to_doc_cache = self._load_chunk_to_doc_map()
        self._phase = "read"
        t0 = time.perf_counter()
        results: List[MemoryEntry] = []

        try:
            qp = QueryParam(
                mode=self._mode,
                only_need_context=True,
                top_k=top_k,
                enable_rerank=False,  # no rerank model configured in SubMem env
                stream=False,
            )
            data = _run(self._rag.aquery_data(query, qp))
        except Exception as e:
            print(f"  [LightRAG] aquery_data error: {e}")
            self._read_times.append(time.perf_counter() - t0)
            return results

        payload = (data or {}).get("data", {}) if isinstance(data, dict) else {}
        entities = payload.get("entities", []) or []
        relations = payload.get("relationships", []) or []
        chunks = payload.get("chunks", []) or []

        # Rank: entities → relations → chunks, in the order upstream
        # returned them (already sorted by relevance within each list).
        n = max(len(entities) + len(relations) + len(chunks), 1)
        rank = 0
        for ent in entities:
            desc = ent.get("description") or ent.get("entity_name", "")
            if not desc:
                continue
            sid = ent.get("source_id", "")
            results.append(MemoryEntry(
                content=f"[Entity] {ent.get('entity_name', '')}: {desc}",
                metadata={
                    "source": "lightrag_entity",
                    "entity_type": ent.get("entity_type", ""),
                    "source_id": sid,
                    "file_path": ent.get("file_path", ""),
                    "dia_ids": self._resolve_dia_ids(sid),
                },
                score=1.0 - rank / n,
            ))
            rank += 1
        for rel in relations:
            desc = rel.get("description", "")
            if not desc:
                continue
            sid = rel.get("source_id", "")
            results.append(MemoryEntry(
                content=(
                    f"[Relation] {rel.get('src_id', '')} — "
                    f"{rel.get('tgt_id', '')}: {desc}"
                ),
                metadata={
                    "source": "lightrag_relation",
                    "keywords": rel.get("keywords", ""),
                    "weight": rel.get("weight", 0.0),
                    "source_id": sid,
                    "file_path": rel.get("file_path", ""),
                    "dia_ids": self._resolve_dia_ids(sid),
                },
                score=1.0 - rank / n,
            ))
            rank += 1
        for ch in chunks:
            content = ch.get("content", "")
            if not content:
                continue
            chunk_id = ch.get("chunk_id", "")
            results.append(MemoryEntry(
                content=content,
                metadata={
                    "source": "lightrag_chunk",
                    "chunk_id": chunk_id,
                    "file_path": ch.get("file_path", ""),
                    "dia_ids": self._resolve_dia_ids(chunk_id),
                },
                score=1.0 - rank / n,
            ))
            rank += 1

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        try:
            self._token_counts["read_context_tokens"] += sum(
                len(self.model.encode(r.content)) for r in results
            )
        except Exception:
            # model may be a dummy / not have .encode in some benches
            pass
        return results

    def reset(self):
        self._pending_texts = []
        self._indexed = False
        if self._rag is not None:
            try:
                _run(self._rag.finalize_storages())
            except Exception:
                pass
        # Wipe working_dir contents on reset — each LoCoMo sample
        # (conv) is an independent test; we cannot leak entities
        # from a previous sample into the next. EXCEPTION: the very
        # first reset (before any sample has indexed anything) on a
        # cache HIT must KEEP the cached graph so sample 0 can reuse
        # it. After that cache-honoring reset, every subsequent reset
        # wipes (and we demote _reused_working_dir to False).
        first_reset = (self._total_writes == 0 and self._reused_working_dir)
        if not first_reset:
            if os.path.isdir(self._working_dir):
                for _entry in os.listdir(self._working_dir):
                    _p = os.path.join(self._working_dir, _entry)
                    try:
                        if os.path.isdir(_p):
                            shutil.rmtree(_p, ignore_errors=True)
                        else:
                            os.unlink(_p)
                    except OSError:
                        pass
            self._doc_id_to_dia_ids.clear()
            self._chunk_to_doc_cache = None
            # Demote: future resets must wipe (now in non-cache mode).
            self._reused_working_dir = False
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._total_writes = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "mgmt_llm_input": 0, "mgmt_llm_output": 0, "mgmt_llm_calls": 0,
            "embed_tokens": 0, "embed_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        self._init_rag()

    def size(self) -> int:
        return self._total_writes

    # ── Diagnostics ──────────────────────────────────────────

    def get_external_storage_bytes(self) -> int:
        total = 0
        if os.path.isdir(self._working_dir):
            for dp, _, fns in os.walk(self._working_dir):
                for fn in fns:
                    try:
                        total += os.path.getsize(os.path.join(dp, fn))
                    except OSError:
                        pass
        return total

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_ms = [t * 1000 for t in self._read_times]
        write_ms = [t * 1000 for t in self._write_times]
        stats.update({
            "lightrag_llm_model": self._llm_model,
            "lightrag_embed_model": self._embed_model,
            "lightrag_mode": self._mode,
            "total_writes": self._total_writes,
            "memory_size_bytes": self.get_external_storage_bytes(),
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

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return all entity + relation descriptions from the working_dir.

        Reads the nano-vectordb JSONs directly (vdb_entities.json /
        vdb_relationships.json) — avoids an aquery roundtrip for Q1
        compression-ratio metrics.
        """
        entries: List[MemoryEntry] = []
        for fname, tag in [
            ("vdb_entities.json", "lightrag_entity"),
            ("vdb_relationships.json", "lightrag_relation"),
        ]:
            p = os.path.join(self._working_dir, fname)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                for rec in (data.get("data") or []):
                    content = rec.get("content") or rec.get("description") or ""
                    if content:
                        entries.append(MemoryEntry(
                            content=content,
                            metadata={"source": tag},
                        ))
            except Exception:
                continue
        return entries
