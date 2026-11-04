"""M1 — Dense Vector Memory (OpenAI embedding + FAISS cosine search).

Per-utterance granularity. write() buffers texts and flushes in a batched
embedding call on the next read()/size() to amortise HTTP round-trips.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import MemoryEntry
from .base import register_memory_unit
from .dense_retrieval import DenseRetrievalMemory


@register_memory_unit("M1")
class DenseVectorMemory(DenseRetrievalMemory):
    """Dense retrieval with per-utterance indexing (buffered write)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending: List[Tuple[str, Dict[str, Any]]] = []

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        # Synchronous encode + index add so E3 reflects real write cost
        # (was lazy-batched, producing E3=0 and inflating first-query E4).
        emb = self._encode([text])
        self._index.add(emb)
        self._entries.append(MemoryEntry(
            content=text,
            metadata=metadata or {},
            timestamp=len(self._entries),
        ))
        self._write_call_count += 1
        self._write_times.append(time.perf_counter() - t0)

    def _flush(self):
        # Legacy path kept as no-op for backward compatibility with any
        # caller that still invokes it. Writes are now synchronous.
        if not self._pending:
            return
        texts = [t for t, _ in self._pending]
        metas = [m for _, m in self._pending]
        embs = self._encode(texts)
        self._index.add(embs)
        for text, meta in zip(texts, metas):
            self._entries.append(MemoryEntry(
                content=text,
                metadata=meta,
                timestamp=len(self._entries),
            ))
        self._pending.clear()

    def read(self, query, top_k: int = 10):
        self._flush()
        return super().read(query, top_k=top_k)

    def size(self) -> int:
        return len(self._entries) + len(self._pending)

    def get_all_entries(self):
        self._flush()
        return super().get_all_entries()

    def get_external_storage_bytes(self) -> int:
        self._flush()
        return super().get_external_storage_bytes()

    def _on_reset(self):
        self._pending.clear()
