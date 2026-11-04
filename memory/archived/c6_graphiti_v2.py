"""C6-Graphiti — Zep/Graphiti native temporal knowledge graph.

Uses the actual graphiti-core library (https://github.com/getzep/graphiti)
with Kuzu embedded graph DB (no Neo4j needed). All in-process, no server.

Pipeline (Graphiti native):
  add_episode(text) → LLM extracts entities + facts → temporal KG
  search(query)     → hybrid BM25 + cosine + reranking → facts/edges

Requirements:
  - pip install graphiti-core[kuzu]
  - An OpenAI-compatible LLM endpoint (vLLM or API)
  - OpenAI API key for embeddings (text-embedding-3-small)
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import typing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_LLM_BASE_URL = os.environ.get("GRAPHITI_LLM_BASE_URL", None)  # e.g. http://localhost:8000/v1
_LLM_MODEL = os.environ.get("GRAPHITI_LLM_MODEL", "gpt-4o-mini")

logger = logging.getLogger(__name__)


_PERSISTENT_LOOP = None
_PERSISTENT_LOOP_THREAD = None


def _get_persistent_loop():
    """Return a single long-lived event loop on a background thread.

    Graphiti's Kuzu driver holds a DB connection bound to whatever loop first
    touched it. Using `asyncio.run()` per call creates a fresh loop each time,
    which closes the driver's connection after the first call. We keep ONE
    loop alive for the process lifetime so the DB connection survives across
    add_episode / search calls.
    """
    global _PERSISTENT_LOOP, _PERSISTENT_LOOP_THREAD
    if _PERSISTENT_LOOP is not None and not _PERSISTENT_LOOP.is_closed():
        return _PERSISTENT_LOOP
    import threading
    _PERSISTENT_LOOP = asyncio.new_event_loop()

    def _run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _PERSISTENT_LOOP_THREAD = threading.Thread(
        target=_run_loop, args=(_PERSISTENT_LOOP,), daemon=True
    )
    _PERSISTENT_LOOP_THREAD.start()
    return _PERSISTENT_LOOP


def _run(coro):
    """Run a coroutine on the persistent background loop, block until done."""
    loop = _get_persistent_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


# ── Instrumented LLM client that tracks token usage ──────────

class _InstrumentedOpenAIClient:
    """Wraps graphiti's OpenAIGenericClient to count tokens and calls.

    We intercept **at the OpenAI SDK boundary** (base_client.client.chat
    .completions.create) rather than reimplementing
    `_generate_response`. This preserves every piece of upstream logic —
    small_model vs model routing by `ModelSize`, retry / error handling,
    prompt cleaning, response_format shaping — and just snapshots the
    usage field on the way back.
    """

    def __init__(self, base_client, token_counts: Dict[str, int]):
        self._base = base_client
        self._counts = token_counts
        self._phase = "write"  # "write" | "read" | "mgmt"

        # Patch the inner OpenAI SDK call. Everything graphiti does
        # around it (schema transformation, model-size routing, retries,
        # error handling) is left intact.
        openai_client = base_client.client
        orig_create = openai_client.chat.completions.create

        async def tracked_create(*args, **kwargs):
            response = await orig_create(*args, **kwargs)

            # Estimate usage — prefer OpenAI-reported; char/4 fallback.
            usage = getattr(response, "usage", None)
            if usage:
                inp = getattr(usage, "prompt_tokens", 0) or 0
                out = getattr(usage, "completion_tokens", 0) or 0
            else:
                msgs = kwargs.get("messages", [])
                total_chars = sum(len(m.get("content", "") or "") for m in msgs)
                inp = total_chars // 4
                try:
                    result_text = response.choices[0].message.content or ""
                except Exception:
                    result_text = ""
                out = len(result_text) // 4

            # Upstream graphiti runs dedupe / entity-resolve prompts inside
            # add_episode() (technically "write phase" from our side but
            # conceptually memory management). Detect those via the system
            # prompt substring and re-bucket.
            phase = self._phase
            if phase == "write":
                sys_content = ""
                for m in kwargs.get("messages", []):
                    if m.get("role") == "system":
                        sys_content = m.get("content", "") or ""
                        break
                if "deduplication assistant" in sys_content:
                    phase = "mgmt"

            if phase == "write":
                self._counts["write_llm_input"] += inp
                self._counts["write_llm_output"] += out
                self._counts["write_llm_calls"] += 1
            elif phase == "read":
                self._counts.setdefault("read_llm_input", 0)
                self._counts.setdefault("read_llm_output", 0)
                self._counts.setdefault("read_llm_calls", 0)
                self._counts["read_llm_input"] += inp
                self._counts["read_llm_output"] += out
                self._counts["read_llm_calls"] += 1
            else:
                self._counts.setdefault("mgmt_llm_input", 0)
                self._counts.setdefault("mgmt_llm_output", 0)
                self._counts.setdefault("mgmt_llm_calls", 0)
                self._counts["mgmt_llm_input"] += inp
                self._counts["mgmt_llm_output"] += out
                self._counts["mgmt_llm_calls"] += 1

            return response

        openai_client.chat.completions.create = tracked_create

    def set_phase(self, phase: str):
        self._phase = phase


@register_memory_unit("C6")
class GraphitiMemory(BaseMemoryUnit):
    """Graphiti native temporal knowledge graph memory.

    write(): Calls graphiti.add_episode() — LLM extracts entities, facts,
             temporal bounds, builds/updates the knowledge graph.
    read():  Calls graphiti.search() — hybrid BM25 + cosine + reranking
             over fact edges in the graph.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }
        self._total_episodes = 0
        # Kuzu DB should live on node-local real disk. Delta compute nodes
        # have /tmp on local NVMe (xfs, 1.5TB) — use that by default. If
        # GRAPHITI_DB_ROOT is set explicitly, honor it. Avoid NFS (home,
        # $SUBMEM_RESULTS) — writes there go through page cache which
        # SLURM counts against cgroup RAM, causing OOM at ~45min for
        # 419-episode LoCoMo despite Python RSS staying below 6GB.
        _scratch = os.environ.get("GRAPHITI_DB_ROOT") or tempfile.gettempdir()
        os.makedirs(_scratch, exist_ok=True)
        self._db_path = os.path.join(_scratch, f"graphiti_{os.getpid()}_{int(time.time())}")
        if os.path.exists(self._db_path):
            shutil.rmtree(self._db_path)
        self._graphiti = None
        self._instrumented_client = None
        # Map episode name → dia_id for P6/P7 evidence matching
        self._episode_dia_map: Dict[str, str] = {}
        # Rolling window of most-recent episode UUIDs. graphiti's add_episode
        # uses `previous_episode_uuids` to cross-reference during entity
        # resolution / deduplication; without it, long conversations keep
        # re-creating the same entities as fresh nodes.
        self._recent_episode_uuids: List[str] = []
        self._prev_episode_window = 1   # SubMem: was 5; reduce dedup cost ~5x
        self._init_graphiti()

    def _init_graphiti(self):
        from graphiti_core import Graphiti
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

        # LLM client — can point to vLLM or OpenAI
        llm_config = LLMConfig(
            api_key=_OPENAI_API_KEY or "dummy",
            model=_LLM_MODEL,
            small_model=_LLM_MODEL,
            base_url=_LLM_BASE_URL,
        )
        llm_client = OpenAIGenericClient(config=llm_config)

        # Wrap with instrumentation for token tracking
        self._instrumented_client = _InstrumentedOpenAIClient(llm_client, self._token_counts)

        # Embedder — OpenAI text-embedding-3-small
        embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key=_OPENAI_API_KEY or "dummy",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
        ))

        # Cross-encoder reranker (instrumented for token tracking)
        cross_encoder = OpenAIRerankerClient(config=llm_config)
        self._instrument_cross_encoder(cross_encoder)

        # Kuzu embedded graph DB (path must NOT exist — Kuzu creates it).
        # CRITICAL: kuzu.Database() default buffer_pool_size=0 → Kuzu allocates
        # ~80% of system RAM for mmap (on 2TB H200/A100 nodes that's 1.6TB
        # virtual). SLURM cgroup accounts mmap virtual pages → OOM despite
        # Python RSS <6GB. Cap at 2GB (plenty for 419-episode LoCoMo graph).
        # KuzuDriver.__init__ internally builds `self.db = kuzu.Database(db)`
        # AND `self.client = kuzu.AsyncConnection(self.db)` bound together;
        # we can't replace db after init without orphaning the client. So
        # monkey-patch kuzu.Database's constructor so driver init picks up
        # the capped variant on first call.
        import kuzu as _kuzu
        _orig_db = _kuzu.Database
        if not getattr(_kuzu, "_submem_bp_patched", False):
            def _capped_db(db_path, *args, **kwargs):
                cap_gb = int(os.environ.get("C6_BUFFER_POOL_GB", "8"))
                kwargs.setdefault("buffer_pool_size", cap_gb * 1024 * 1024 * 1024)
                return _orig_db(db_path, *args, **kwargs)
            _kuzu.Database = _capped_db
            _kuzu._submem_bp_patched = True
        driver = KuzuDriver(db=self._db_path)
        self._driver = driver

        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )
        # Patch: KuzuDriver lacks _database attr that graphiti_core expects (Neo4j-ism)
        if not hasattr(driver, '_database'):
            driver._database = None

        # KuzuDriver's build_indices_and_constraints is a no-op (bug in graphiti 0.28),
        # but search requires FTS indices. Create them manually.
        import kuzu
        conn = kuzu.Connection(driver.db)
        fts_queries = [
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ]
        for q in fts_queries:
            try:
                conn.execute(q)
            except Exception:
                pass  # index may already exist
        conn.close()
        print(f"  [Graphiti] Initialized (model={_LLM_MODEL}, db={self._db_path})")

    def _instrument_cross_encoder(self, cross_encoder):
        """Wrap cross-encoder's OpenAI client to track reranking token costs."""
        original_create = cross_encoder.client.chat.completions.create
        token_counts = self._token_counts
        instrumented = self._instrumented_client

        async def tracked_create(*args, **kwargs):
            response = await original_create(*args, **kwargs)
            usage = getattr(response, 'usage', None)
            phase = instrumented._phase if instrumented else "read"
            inp = getattr(usage, 'prompt_tokens', 0) if usage else 0
            out = getattr(usage, 'completion_tokens', 0) if usage else 0
            prefix = phase if phase != "write" else "write"
            if phase == "read":
                token_counts.setdefault("read_llm_input", 0)
                token_counts.setdefault("read_llm_output", 0)
                token_counts.setdefault("read_llm_calls", 0)
                token_counts["read_llm_input"] += inp
                token_counts["read_llm_output"] += out
                token_counts["read_llm_calls"] += 1
            else:
                token_counts[f"{prefix}_llm_input"] += inp
                token_counts[f"{prefix}_llm_output"] += out
                token_counts[f"{prefix}_llm_calls"] += 1
            return response

        cross_encoder.client.chat.completions.create = tracked_create

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Buffer an episode. Actual add_episode() runs in _ensure_indexed()
        so multiple episodes can be processed concurrently via asyncio.gather
        (5-10x speedup vs serial write-per-turn)."""
        self._write_call_count += 1
        if not hasattr(self, "_pending_episodes"):
            self._pending_episodes = []
            self._indexed = False
        self._pending_episodes.append({
            "text": text,
            "metadata": metadata or {},
            "episode_name": f"msg_{self._write_call_count}",
        })

    def _ensure_indexed(self):
        """Flush buffered episodes via concurrent add_episode calls.
        Called lazily before first read(), or by harness before Stage 3."""
        if getattr(self, "_indexed", False):
            return
        pending = getattr(self, "_pending_episodes", [])
        if not pending:
            self._indexed = True
            return

        import asyncio as _aio
        from graphiti_core.nodes import EpisodeType

        if self._instrumented_client:
            self._instrumented_client.set_phase("write")

        concurrency = int(os.environ.get("C6_CONCURRENCY", "8"))
        enable_dedup = os.environ.get("C6_ENABLE_DEDUP", "0") == "1"
        print(f"    [Graphiti] flushing {len(pending)} episodes (concurrency={concurrency}, dedup={enable_dedup})", flush=True)

        async def _add_one(ep):
            dia_id = ep["metadata"].get("dia_id", "")
            session_date = ep["metadata"].get("session_date", "")
            ref_time = datetime.now(timezone.utc)
            if session_date:
                try:
                    ref_time = datetime.fromisoformat(session_date).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass
            if dia_id:
                self._episode_dia_map[ep["episode_name"]] = dia_id
            return await self._graphiti.add_episode(
                name=ep["episode_name"],
                episode_body=ep["text"],
                source_description=dia_id or "conversation message",
                source=EpisodeType.message,
                reference_time=ref_time,
                group_id="benchmark",
                previous_episode_uuids=None,   # dedup off when batched concurrently
            )

        async def _run_batch(batch):
            return await _aio.gather(*(_add_one(ep) for ep in batch), return_exceptions=True)

        t0 = time.perf_counter()
        n_done = 0
        for i in range(0, len(pending), concurrency):
            batch = pending[i:i + concurrency]
            t_b0 = time.perf_counter()
            results = _run(_run_batch(batch))
            t_b = time.perf_counter() - t_b0
            for ep, res in zip(batch, results):
                if isinstance(res, Exception):
                    print(f"  [Graphiti] write error (turn {ep['episode_name']}): {res}")
                    continue
                self._total_episodes += 1
                # per-episode time = batch_time / batch_size (amortized)
                self._write_times.append(t_b / max(len(batch), 1))
                n_done += 1
            if (i + concurrency) % (concurrency * 2) == 0 or i + concurrency >= len(pending):
                try:
                    import psutil, os as _os
                    rss_mb = psutil.Process(_os.getpid()).memory_info().rss / (1024*1024)
                    print(f"    [Graphiti] {n_done}/{len(pending)} episodes done (RSS={rss_mb:.0f}MB, batch {t_b:.1f}s)", flush=True)
                except ImportError:
                    print(f"    [Graphiti] {n_done}/{len(pending)} episodes done", flush=True)

        total = time.perf_counter() - t0
        print(f"    [Graphiti] flush complete in {total:.1f}s ({total/max(len(pending),1):.2f}s/ep)", flush=True)
        self._indexed = True

    def _resolve_dia_ids(self, edge) -> List[str]:
        """Map edge's episode UUIDs back to dia_ids for P6/P7."""
        episodes = getattr(edge, 'episodes', [])
        if not episodes:
            return []

        dia_ids = []
        # Query Kuzu for episode names matching the UUIDs
        try:
            import kuzu
            conn = kuzu.Connection(self._driver.db)
            for ep_uuid in episodes:
                result = conn.execute(
                    "MATCH (e:Episodic {uuid: $uuid}) RETURN e.source_description AS sd",
                    parameters={"uuid": ep_uuid},
                )
                while result.has_next():
                    row = result.get_next()
                    sd = row[0]
                    if sd and sd != "conversation message":
                        dia_ids.append(sd)
            conn.close()
        except Exception:
            pass
        return dia_ids

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Search the temporal knowledge graph."""
        # Flush buffered episodes on first read
        self._ensure_indexed()
        t0 = time.perf_counter()
        results = []

        # Set phase to "read" for token tracking
        if self._instrumented_client:
            self._instrumented_client.set_phase("read")

        try:
            # Use search_() (not the legacy search()): the former exercises
            # the advanced COMBINED_HYBRID_SEARCH_CROSS_ENCODER recipe which
            # (a) returns both nodes AND edges, (b) actually invokes the
            # cross-encoder reranker we wired in at __init__, and (c) is the
            # Zep paper's evaluation path. The legacy search() only does
            # EDGE_HYBRID_SEARCH_RRF with no reranking.
            from copy import deepcopy
            from graphiti_core.search.search_config_recipes import (
                COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            )
            search_config = deepcopy(COMBINED_HYBRID_SEARCH_CROSS_ENCODER)
            search_config.limit = top_k
            # Cap BFS depth to avoid Kuzu buffer-pool exhaustion.
            # Recipe default depth=3 → RELATES_TO*2..4 cypher explodes on
            # 419-episode graphs. env C6_BFS_DEPTH overrides (default 1).
            bfs_depth = int(os.environ.get("C6_BFS_DEPTH", "1"))
            if search_config.edge_config is not None:
                search_config.edge_config.bfs_max_depth = bfs_depth
            if search_config.node_config is not None:
                search_config.node_config.bfs_max_depth = bfs_depth
            search_results = _run(self._graphiti.search_(
                query=query,
                config=search_config,
                group_ids=["benchmark"],
            ))
            edges = list(search_results.edges or [])
            nodes = list(search_results.nodes or [])

            for edge in edges:
                fact = getattr(edge, "fact", "")
                if not fact:
                    continue
                # Include temporal info if available
                valid_at = getattr(edge, "valid_at", None)
                invalid_at = getattr(edge, "invalid_at", None)
                temporal = ""
                if valid_at:
                    temporal += f" [valid from: {valid_at}]"
                if invalid_at:
                    temporal += f" [invalid after: {invalid_at}]"

                # Resolve dia_ids for P6/P7 evidence matching
                dia_ids = self._resolve_dia_ids(edge)

                results.append(MemoryEntry(
                    content=fact + temporal,
                    metadata={
                        "source": "graphiti_kg_edge",
                        "edge_uuid": getattr(edge, "uuid", ""),
                        "source_node": getattr(edge, "source_node_uuid", ""),
                        "target_node": getattr(edge, "target_node_uuid", ""),
                        "dia_ids": dia_ids,
                    },
                    score=1.0,
                ))

            # Paper also uses node summaries as retrieval context. The advanced
            # search returns them; the legacy search() path didn't, which was
            # part of why C6 numbers were weak.
            for node in nodes:
                summary = getattr(node, "summary", "") or getattr(node, "name", "")
                if not summary:
                    continue
                results.append(MemoryEntry(
                    content=summary,
                    metadata={
                        "source": "graphiti_kg_node",
                        "node_uuid": getattr(node, "uuid", ""),
                        "name": getattr(node, "name", ""),
                        "dia_ids": [],
                    },
                    score=1.0,
                ))
        except Exception as e:
            print(f"  [Graphiti] search error: {e}")

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(self.model.encode(r.content)) for r in results
        )
        return results

    def reset(self):
        """Destroy graph and recreate."""
        self._pending_episodes = []
        self._indexed = False
        if self._graphiti:
            try:
                _run(self._graphiti.close())
            except Exception:
                pass
        # Remove old DB
        if os.path.exists(self._db_path):
            shutil.rmtree(self._db_path, ignore_errors=True)
        # Kuzu DB should live on node-local real disk. Delta compute nodes
        # have /tmp on local NVMe (xfs, 1.5TB) — use that by default. If
        # GRAPHITI_DB_ROOT is set explicitly, honor it. Avoid NFS (home,
        # $SUBMEM_RESULTS) — writes there go through page cache which
        # SLURM counts against cgroup RAM, causing OOM at ~45min for
        # 419-episode LoCoMo despite Python RSS staying below 6GB.
        _scratch = os.environ.get("GRAPHITI_DB_ROOT") or tempfile.gettempdir()
        os.makedirs(_scratch, exist_ok=True)
        self._db_path = os.path.join(_scratch, f"graphiti_{os.getpid()}_{int(time.time())}")
        if os.path.exists(self._db_path):
            shutil.rmtree(self._db_path)
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._total_episodes = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }
        self._episode_dia_map.clear()
        self._recent_episode_uuids.clear()
        self._instrumented_client = None
        self._init_graphiti()

    def size(self) -> int:
        return self._total_episodes

    # ── cache support ─────────────────────────────────────────
    # C6 cache is DISABLED. Kuzu DB live-file copy interacts poorly with
    # graphiti_core's cached connection pool: after save_state the next
    # search_() hits "Database is closed". The root cause is KuzuDriver
    # internally holding connections whose lifetime is tied to the db
    # object, and graphiti stashes those via self._driver.client_pool
    # (undocumented attr that survives our db swap). Wiring through a
    # full graphiti.close()+rebuild is too invasive for marginal cache
    # benefit (C6 writes are ~5min, not the slowest substrate).
    # If you need to re-enable, revive the save_state below and call
    # KuzuDriver._database = None before copy.
    CACHE_KEY_EXTRA = ["prev_episode_window"]

    def _disabled_save_state(self, dir_path: str) -> None:
        """Cache: copy Kuzu DB (and WAL) while keeping connection open.
        Kuzu supports concurrent read of file while open — copying live
        files is safe enough for cache purposes (WAL captures uncommitted)."""
        import os, json, shutil
        os.makedirs(dir_path, exist_ok=True)
        cached_kuzu = os.path.join(dir_path, "kuzu_db")
        if os.path.exists(cached_kuzu):
            shutil.rmtree(cached_kuzu)
        if os.path.isdir(self._db_path):
            shutil.copytree(self._db_path, cached_kuzu)
        elif os.path.isfile(self._db_path):
            shutil.copy(self._db_path, cached_kuzu)
            wal = self._db_path + ".wal"
            if os.path.exists(wal):
                shutil.copy(wal, cached_kuzu + ".wal")
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            json.dump({
                "episode_dia_map": dict(self._episode_dia_map),
                "recent_episode_uuids": list(self._recent_episode_uuids),
                "total_episodes": self._total_episodes,
                "token_counts": dict(self._token_counts),
                "write_call_count": self._write_call_count,
                "write_times": list(self._write_times),
            }, f)
        # No close/reopen: graphiti keeps using original _driver.db unchanged.

    @classmethod
    def _disabled_load_state(cls, dir_path: str, config, model):
        """Restore Kuzu DB to a fresh path; re-init Graphiti."""
        import os, json, shutil, tempfile, time as _time
        unit = cls.__new__(cls)   # bypass __init__ since it creates a new DB
        # Manually set up minimal state expected by __init__
        from .base import BaseMemoryUnit
        BaseMemoryUnit.__init__(unit, config, model)
        unit._write_times = []
        unit._read_times = []
        unit._write_call_count = 0
        unit._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        unit._total_episodes = 0
        unit._episode_dia_map = {}
        unit._recent_episode_uuids = []
        unit._prev_episode_window = (
            (config.extra or {}).get("prev_episode_window", 1)
        )
        unit._graphiti = None
        unit._instrumented_client = None
        # Restore Kuzu DB at fresh path so concurrent loads don't collide
        new_db = os.path.join(tempfile.gettempdir(),
                              f"graphiti_load_{os.getpid()}_{int(_time.time())}")
        cached_kuzu = os.path.join(dir_path, "kuzu_db")
        if os.path.isdir(cached_kuzu):
            shutil.copytree(cached_kuzu, new_db)
        elif os.path.isfile(cached_kuzu):
            shutil.copy(cached_kuzu, new_db)
            wal_src = cached_kuzu + ".wal"
            if os.path.exists(wal_src):
                shutil.copy(wal_src, new_db + ".wal")
        unit._db_path = new_db
        # Restore meta
        with open(os.path.join(dir_path, "meta.json")) as f:
            d = json.load(f)
        unit._episode_dia_map = d["episode_dia_map"]
        unit._recent_episode_uuids = d["recent_episode_uuids"]
        unit._total_episodes = d["total_episodes"]
        unit._token_counts = d["token_counts"]
        unit._write_call_count = d["write_call_count"]
        unit._write_times = d["write_times"]
        # Re-init Graphiti pointing at restored DB
        unit._init_graphiti()
        return unit

    def get_external_storage_bytes(self) -> int:
        """Return Kuzu DB size on disk.

        Kuzu creates multiple files with the db_path as prefix
        (e.g. /tmp/graphiti_123_456, /tmp/graphiti_123_456.wal, etc.)
        and/or a directory. Count them all.
        Falls back to in-memory estimation if disk walk returns 0.
        """
        # Flush Kuzu WAL before measuring
        try:
            if hasattr(self, '_driver') and hasattr(self._driver, 'db'):
                import kuzu
                conn = kuzu.Connection(self._driver.db)
                conn.execute("CHECKPOINT")
                conn.close()
        except Exception:
            pass

        total = 0
        db_dir = os.path.dirname(self._db_path)
        db_base = os.path.basename(self._db_path)
        if os.path.isdir(self._db_path):
            for dirpath, _, filenames in os.walk(self._db_path):
                for f in filenames:
                    total += os.path.getsize(os.path.join(dirpath, f))
        elif db_dir:
            # Kuzu uses db_path as prefix for multiple files
            for f in os.listdir(db_dir):
                if f.startswith(db_base):
                    fp = os.path.join(db_dir, f)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
                    elif os.path.isdir(fp):
                        for dp, _, fns in os.walk(fp):
                            for fn in fns:
                                total += os.path.getsize(os.path.join(dp, fn))

        # Fallback: estimate from in-memory data if disk returns 0
        if total == 0 and self._total_episodes > 0:
            try:
                entries = self.get_all_entries()
                total = sum(len(e.content.encode("utf-8")) for e in entries)
            except Exception:
                pass
        return total

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]
        stats.update({
            "graphiti_model": _LLM_MODEL,
            "total_episodes": self._total_episodes,
            "compression_ratio": 1.0,
            "memory_size_bytes": self.get_external_storage_bytes(),
            "token_consumption": dict(self._token_counts),
            "write_call_count": self._write_call_count,
            "read_latency_ms": {
                "mean": float(np.mean(read_times_ms)) if read_times_ms else 0.0,
                "p50": float(np.median(read_times_ms)) if read_times_ms else 0.0,
                "p90": float(np.percentile(read_times_ms, 90)) if read_times_ms else 0.0,
            },
            "write_latency_ms": {
                "mean": float(np.mean(write_times_ms)) if write_times_ms else 0.0,
                "p50": float(np.median(write_times_ms)) if write_times_ms else 0.0,
                "p90": float(np.percentile(write_times_ms, 90)) if write_times_ms else 0.0,
            },
        })
        return stats

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return all fact edges + entity summaries — the actual compressed
        graph representation. Used by Q1 compression ratio.
        """
        entries: List[MemoryEntry] = []
        try:
            # Direct Cypher — avoids BFS buffer-pool hit of graphiti.search(q="")
            driver = self._driver
            q_edges = """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                WHERE a.group_id = $gid AND b.group_id = $gid
                RETURN r.fact AS fact
            """
            q_nodes = """
                MATCH (n:Entity) WHERE n.group_id = $gid
                RETURN coalesce(n.summary, n.name) AS summary
            """
            async def _collect():
                out = []
                rows, _, _ = await driver.execute_query(q_edges, gid="benchmark")
                for rec in rows:
                    f = rec.get("fact")
                    if f: out.append(("edge", f))
                rows, _, _ = await driver.execute_query(q_nodes, gid="benchmark")
                for rec in rows:
                    s = rec.get("summary")
                    if s: out.append(("node", s))
                return out
            results = _run(_collect())
            for kind, text in results:
                entries.append(MemoryEntry(
                    content=text,
                    metadata={"source": f"graphiti_{kind}"},
                ))
        except Exception as e:
            print(f"  [C6] get_all_entries failed: {e}")
        return entries
