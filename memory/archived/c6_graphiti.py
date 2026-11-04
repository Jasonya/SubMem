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
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_LLM_BASE_URL = os.environ.get("GRAPHITI_LLM_BASE_URL", None)  # e.g. http://localhost:8000/v1
_LLM_MODEL = os.environ.get("GRAPHITI_LLM_MODEL", "gpt-4o-mini")


def _run(coro):
    """Run async coroutine from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


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
        self._db_path = os.path.join(tempfile.gettempdir(), f"graphiti_{os.getpid()}_{int(time.time())}")
        if os.path.exists(self._db_path):
            shutil.rmtree(self._db_path)
        self._graphiti = None
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

        # Embedder — OpenAI text-embedding-3-small
        embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key=_OPENAI_API_KEY or "dummy",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
        ))

        # Cross-encoder reranker
        cross_encoder = OpenAIRerankerClient(config=llm_config)

        # Kuzu embedded graph DB (path must NOT exist — Kuzu creates it)
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

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an episode to the temporal knowledge graph.

        Graphiti's add_episode() triggers LLM entity extraction,
        entity resolution, fact extraction with temporal bounds,
        and conflict resolution — all natively.
        """
        t0 = time.perf_counter()
        self._write_call_count += 1

        session_date = ""
        if metadata:
            session_date = metadata.get("session_date", "")

        ref_time = datetime.now(timezone.utc)
        if session_date:
            try:
                ref_time = datetime.fromisoformat(session_date).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        try:
            from graphiti_core.nodes import EpisodeType
            _run(self._graphiti.add_episode(
                name=f"msg_{self._write_call_count}",
                episode_body=text,
                source_description="conversation message",
                source=EpisodeType.message,
                reference_time=ref_time,
                group_id="benchmark",
            ))
            self._total_episodes += 1
        except Exception as e:
            print(f"  [Graphiti] write error (turn {self._write_call_count}): {e}")

        self._write_times.append(time.perf_counter() - t0)

        if self._write_call_count % 20 == 0:
            print(f"    [Graphiti] {self._write_call_count} episodes added", flush=True)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Search the temporal knowledge graph.

        Uses Graphiti's native hybrid search: BM25 + cosine similarity
        over fact edges, with RRF reranking.
        """
        t0 = time.perf_counter()
        results = []

        try:
            edges = _run(self._graphiti.search(
                query=query,
                group_ids=["benchmark"],
                num_results=top_k,
            ))
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

                results.append(MemoryEntry(
                    content=fact + temporal,
                    metadata={
                        "source": "graphiti_kg",
                        "edge_uuid": getattr(edge, "uuid", ""),
                        "source_node": getattr(edge, "source_node_uuid", ""),
                        "target_node": getattr(edge, "target_node_uuid", ""),
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
        if self._graphiti:
            try:
                _run(self._graphiti.close())
            except Exception:
                pass
        # Remove old DB
        if os.path.exists(self._db_path):
            shutil.rmtree(self._db_path, ignore_errors=True)
        self._db_path = os.path.join(tempfile.gettempdir(), f"graphiti_{os.getpid()}_{int(time.time())}")
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
        self._init_graphiti()

    def size(self) -> int:
        return self._total_episodes

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]
        stats.update({
            "graphiti_model": _LLM_MODEL,
            "total_episodes": self._total_episodes,
            "compression_ratio": 1.0,
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
        """Return all fact edges from the graph."""
        entries = []
        try:
            # Search with a broad query to get all facts
            edges = _run(self._graphiti.search(
                query="",
                group_ids=["benchmark"],
                num_results=1000,
            ))
            for edge in edges:
                fact = getattr(edge, "fact", "")
                if fact:
                    entries.append(MemoryEntry(
                        content=fact,
                        metadata={"source": "graphiti_kg"},
                    ))
        except Exception:
            pass
        return entries
