"""M10 — Full Context Baseline (No Memory).

Vanilla Transformer with no external memory. All conversation history is
concatenated into the context window. When it exceeds the model's max length,
it is truncated from the beginning (oldest turns dropped first).

This is the baseline: no write/read mechanism, no retrieval, no editing.
Just raw context stuffing. Measures what the model can do with its native
context window alone.
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit


@register_memory_unit("M10")
class FullContextMemory(BaseMemoryUnit):
    """No-memory baseline — concatenate all text, truncate to context window.

    write(): append to buffer
    read():  return entire buffer as a single entry, truncated to model max length
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict[str, Any]] = []
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
        # Buffer truncation budget. Default is 262144 (256K) as a generous
        # fallback when the model's native window is unknown (e.g. API
        # backends). For local HF models we additionally cap to the model's
        # own context window (config.max_position_embeddings) at read-time,
        # so the prompt never exceeds what the model can actually attend to.
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._max_tokens = extra.get("max_tokens", 262144)
        # Reserve tokens for query prompt + generation
        self._reserve_tokens = 512

    def _model_context_window(self) -> Optional[int]:
        """Best-effort lookup of the model's native context window.

        Returns None if we cannot determine it (e.g. API model backends).
        For HF causal LMs, ``config.max_position_embeddings`` is the
        authoritative value (Qwen3-8B/32B = 40960; Gemma-4-26B varies).
        """
        m = getattr(self, "model", None)
        if m is None:
            return None
        inner = getattr(m, "_model", None)
        if inner is not None:
            cfg = getattr(inner, "config", None)
            mpe = getattr(cfg, "max_position_embeddings", None) if cfg else None
            if isinstance(mpe, int) and mpe > 0:
                return mpe
        tok = getattr(m, "_tokenizer", None)
        if tok is not None:
            mml = getattr(tok, "model_max_length", None)
            # HF sets model_max_length to a sentinel (1e30) when unset; ignore.
            if isinstance(mml, int) and 0 < mml < 10**8:
                return mml
        return None

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._write_call_count += 1
        self._write_times.append(time.perf_counter() - t0)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        t0 = time.perf_counter()

        if not self._buffer:
            self._read_times.append(time.perf_counter() - t0)
            return []

        # Truncate from the beginning if exceeds the effective budget.
        # Effective budget = min(configured buffer budget, model's native
        # context window) — without the second cap, a 256K default sends
        # an over-window prompt to a 40K-window model (Qwen3-8B/32B) and
        # the serving stack silently truncates from the front, which is
        # not the "raw-history retention" baseline this substrate claims.
        model_window = self._model_context_window()
        effective_max = self._max_tokens
        if model_window is not None:
            effective_max = min(effective_max, model_window)
        # Build from the end (most recent) until we hit the token limit
        max_ctx_tokens = effective_max - self._reserve_tokens
        kept_indices = []
        total_tokens = 0
        for i in range(len(self._buffer) - 1, -1, -1):
            utt_tokens = len(self.model.encode(self._buffer[i]))
            if total_tokens + utt_tokens > max_ctx_tokens:
                break
            kept_indices.append(i)
            total_tokens += utt_tokens
        kept_indices.reverse()

        full_context = "\n".join(self._buffer[i] for i in kept_indices)

        # Collect dia_ids from kept utterances
        dia_ids = []
        any_has_answer = False
        kept_session_ids = set()
        for i in kept_indices:
            meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}
            did = meta.get("dia_id", "")
            if did:
                dia_ids.append(did)
            if meta.get("has_answer"):
                any_has_answer = True
            sid = meta.get("session_id")
            if sid:
                kept_session_ids.add(sid)

        result = [MemoryEntry(
            content=full_context,
            metadata={"source": "full_context", "dia_ids": dia_ids,
                       "truncated": len(kept_indices) < len(self._buffer),
                       # Aggregate has_answer across all retained turns so
                       # P5/P6/P7 (which scan retrieved entries for the
                       # has_answer flag) can credit M10 when at least one
                       # gold-evidence turn survived truncation.
                       "has_answer": any_has_answer,
                       "session_id": ";".join(sorted(kept_session_ids)) or None,
                       "role": None},
            score=1.0,
        )]

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += total_tokens
        return result

    def reset(self):
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }

    def size(self) -> int:
        return len(self._buffer)

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]
        total_tokens = sum(len(self.model.encode(t)) for t in self._buffer) if self._buffer else 0
        stats.update({
            "total_context_tokens": total_tokens,
            "max_context_tokens": self._max_tokens,
            "truncated": total_tokens > (self._max_tokens - self._reserve_tokens),
            "num_raw_utterances": len(self._buffer),
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

    def get_external_storage_bytes(self) -> int:
        return sum(len(t.encode("utf-8")) for t in self._buffer)

    def get_all_entries(self) -> List[MemoryEntry]:
        return [MemoryEntry(content=t) for t in self._buffer]

    def save_state(self, dir_path: str) -> None:
        import os, json
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "buffer.json"), "w") as f:
            json.dump({
                "buffer": self._buffer,
                "buffer_metadata": self._buffer_metadata,
                "write_call_count": self._write_call_count,
                "write_times": list(self._write_times),
            }, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        import os, json
        unit = cls(config, model)
        with open(os.path.join(dir_path, "buffer.json")) as f:
            d = json.load(f)
        unit._buffer = d["buffer"]
        unit._buffer_metadata = d["buffer_metadata"]
        unit._write_call_count = d["write_call_count"]
        unit._write_times = d["write_times"]
        return unit
