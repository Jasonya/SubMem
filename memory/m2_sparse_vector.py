"""M2 — Sparse Vector Memory (BM25 / lexical matching).

Tokenizes each memory entry, builds an inverted index via rank_bm25.BM25Okapi.
Retrieval: BM25 scoring (tf-idf + length normalization) top-k.
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "about", "between", "through", "during", "before", "after",
    "and", "but", "or", "not", "no", "so", "if", "then", "than", "too", "very",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize with punctuation stripping + stopword removal."""
    import re
    # Lowercase + strip punctuation
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = text.split()
    # Remove stopwords
    return [t for t in tokens if t and t not in _STOPWORDS]


@register_memory_unit("M2")
class SparseVectorMemory(BaseMemoryUnit):
    """Sparse retrieval via BM25Okapi inverted index."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self._entries: List[MemoryEntry] = []
        self._corpus_tokens: List[List[str]] = []
        self._bm25 = None
        self._index_dirty = True
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

    def _ensure_index(self):
        """Lazily rebuild BM25 index only when needed (before read).

        Index build time is amortized into write_times so E3 reflects
        total write cost (tokenize + index) per entry.
        """
        if not self._index_dirty:
            return
        from rank_bm25 import BM25Okapi

        t0 = time.perf_counter()
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None
        self._index_dirty = False
        # Amortize index build time into write latency
        if self._entries:
            index_time = time.perf_counter() - t0
            per_entry = index_time / len(self._entries)
            self._write_times = [t + per_entry for t in self._write_times]

    # ── interface ────────────────────────────────────────────

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        tokens = _tokenize(text)
        self._corpus_tokens.append(tokens)
        self._entries.append(MemoryEntry(
            content=text,
            metadata=metadata or {},
            timestamp=len(self._entries),
        ))
        self._index_dirty = True
        self._write_call_count += 1
        self._write_times.append(time.perf_counter() - t0)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        self._ensure_index()
        if self._bm25 is None or len(self._entries) == 0:
            return []
        t0 = time.perf_counter()
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        k = min(top_k, len(self._entries))
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            entry = self._entries[idx]
            results.append(MemoryEntry(
                content=entry.content,
                metadata=entry.metadata,
                score=float(scores[idx]),
                timestamp=entry.timestamp,
            ))
        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(self.model.encode(r.content)) for r in results
        )
        return results

    def reset(self):
        self._entries.clear()
        self._corpus_tokens.clear()
        self._bm25 = None
        self._index_dirty = True
        self._write_times.clear()
        self._read_times.clear()
        # Note: _write_call_count intentionally NOT reset (accumulates across
        # samples, consistent with DenseRetrievalMemory behavior)
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }

    def size(self) -> int:
        return len(self._entries)

    def save_state(self, dir_path: str) -> None:
        import os, pickle
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "state.pkl"), "wb") as f:
            pickle.dump({
                "entries": [(e.content, e.metadata, e.score, e.timestamp) for e in self._entries],
                "corpus_tokens": self._corpus_tokens,
                "token_counts": dict(self._token_counts),
                "write_times": list(self._write_times),
                "write_call_count": self._write_call_count,
            }, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        import os, pickle
        unit = cls(config, model)
        with open(os.path.join(dir_path, "state.pkl"), "rb") as f:
            data = pickle.load(f)
        unit._entries = [MemoryEntry(content=c, metadata=m, score=s, timestamp=t)
                         for (c, m, s, t) in data["entries"]]
        unit._corpus_tokens = data["corpus_tokens"]
        unit._token_counts = data["token_counts"]
        unit._write_times = data["write_times"]
        unit._write_call_count = data["write_call_count"]
        unit._index_dirty = True   # rebuild BM25 lazily on first read
        return unit

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]
        stats.update({
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

    def get_gpu_memory_bytes(self) -> int:
        # BM25 is CPU-only, no GPU memory
        return 0

    def get_external_storage_bytes(self) -> int:
        # Approximate: token lists + raw text
        token_bytes = sum(
            sum(len(t) for t in tokens) * 1  # ~1 byte per char in tokens
            for tokens in self._corpus_tokens
        )
        # BM25 internal arrays (idf, doc_len, avgdl, etc.)
        bm25_overhead = len(self._entries) * 8 * 3 if self._entries else 0  # rough
        text_bytes = sum(len(e.content.encode("utf-8")) for e in self._entries)
        return token_bytes + bm25_overhead + text_bytes

    def get_all_entries(self) -> List[MemoryEntry]:
        return list(self._entries)
