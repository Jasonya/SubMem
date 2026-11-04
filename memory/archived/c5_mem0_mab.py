"""C5-MAB — Mem0 via MemoryAgentBench (HUST-AI-HYZ) adapter.

Ported from the `AgentWrapper._initialize_mem0_agent` and
`AgentWrapper._handle_mem0_agent` implementations in
https://github.com/HUST-AI-HYZ/MemoryAgentBench (agent.py).

Differences from our existing C5:
- Uses MAB's exact add/search calling convention: per-context ``user_id``,
  ``memorize_template``-wrapped messages, ``memory.search(limit=retrieve_num)``.
- Parametric read() that runs an answer model (gpt-4o-mini by default) over
  retrieved memories + current time, mirroring MAB's prompt.
- Retains our E-metric instrumentation (E5/E6 write, E8/E9 read, E10/E11 mgmt).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit


SYSTEM_MESSAGE = (
    "You are a helpful AI assistant. Memorize the conversation to help "
    "answer the user's question."
)

MEMORIZE_TEMPLATE = (
    "Dialogue between User and Assistant {time_stamp}\n"
    "<User> The following context is the conversation between the user and "
    "the assistant:\n{context}\n<Assistant> I have memorized the "
    "conversation and I will answer the question you ask."
)


@register_memory_unit("C5-MAB")
class Mem0MABMemory(BaseMemoryUnit):
    """Mem0 via MemoryAgentBench adapter (ported from HUST-AI-HYZ/MAB)."""

    REQUIRES_GEN_MODEL_IN_KEY = False
    CACHE_KEY_EXTRA = ["mem0_model", "retrieve_num"]

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._mem0_model = extra.get("mem0_model", "gpt-4o-mini")
        self._retrieve_num = int(extra.get("retrieve_num", 10))
        self._temperature = float(extra.get("mem0_temperature", 0.0))
        self._max_tokens = int(extra.get("mem0_max_tokens", 1024))
        self._context_id = int(extra.get("mem0_context_id", 0))
        self._sub_dataset = extra.get("mem0_sub_dataset", "locomo")

        self._memory = None
        self._oai_client = None
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "mgmt_llm_input": 0, "mgmt_llm_output": 0, "mgmt_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        self._recent_dia_ids_per_write: List[List[str]] = []
        self._last_retrieved_dia_ids: List[str] = []

    def _ensure_client(self):
        if self._memory is not None:
            return
        # Load .env for OpenAI credentials
        from pathlib import Path
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

        # Set MEM0_DIR to a per-PID subdir to isolate concurrent jobs
        base = os.environ.get("MEM0_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "submem", "mem0_mab"
        )
        unique = os.path.join(base, f"pid{os.getpid()}_{int(time.time()*1000)}")
        os.makedirs(unique, exist_ok=True)
        os.environ["MEM0_DIR"] = unique

        # Route real OpenAI creds if runner is pointed at vLLM
        real_key = os.environ.get("SUBMEM_REAL_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY", "")
        real_base = os.environ.get("SUBMEM_REAL_OPENAI_BASE", "https://api.openai.com/v1")
        if real_key:
            try:
                from openai import OpenAI as _RealOpenAI, AsyncOpenAI as _RealAsyncOpenAI
                class _MABOpenAI(_RealOpenAI):
                    def __init__(self, *a, **kw):
                        kw["api_key"] = real_key; kw["base_url"] = real_base
                        super().__init__(*a, **kw)
                class _MABAsyncOpenAI(_RealAsyncOpenAI):
                    def __init__(self, *a, **kw):
                        kw["api_key"] = real_key; kw["base_url"] = real_base
                        super().__init__(*a, **kw)
                for mod_name in ("mem0.embeddings.openai", "mem0.llms.openai",
                                 "mem0.llms.openai_structured"):
                    try:
                        import importlib
                        mod = importlib.import_module(mod_name)
                        if hasattr(mod, "OpenAI"): mod.OpenAI = _MABOpenAI
                        if hasattr(mod, "AsyncOpenAI"): mod.AsyncOpenAI = _MABAsyncOpenAI
                    except Exception: pass
            except Exception: pass

        # Instrument mem0's internal openai chat calls for E5/E10 tracking
        self._patch_openai_client()

        from mem0.memory.main import Memory
        # Default config (mem0's own defaults) — keeps parity with MAB which
        # just calls `Memory()` with no args.
        self._memory = Memory()
        self._oai_client = OpenAI(api_key=real_key, base_url=real_base) if real_key else OpenAI()

    def _patch_openai_client(self):
        if getattr(self, "_openai_patched", False): return
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
                    # mem0 internal calls happen on write (fact extract) and
                    # on the optional update/delete path. We treat both as
                    # write (E5/E6) since MAB doesn't run a separate mgmt
                    # pass. Read-time answer generation is tracked by the
                    # outer `_oai_client` used in read().
                    bucket = "write"
                    outer._token_counts[f"{bucket}_llm_input"] += getattr(usage, "prompt_tokens", 0) or 0
                    outer._token_counts[f"{bucket}_llm_output"] += getattr(usage, "completion_tokens", 0) or 0
                    outer._token_counts[f"{bucket}_llm_calls"] += 1
            except Exception: pass
            return resp

        Completions.create = tracked
        self._openai_patched = True

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._ensure_client()
        t0 = time.perf_counter()
        self._write_call_count += 1

        dia_id = (metadata or {}).get("dia_id", "")
        self._recent_dia_ids_per_write.append([dia_id] if dia_id else [])

        user_id = f"context_{self._context_id}_{self._sub_dataset}"
        formatted = MEMORIZE_TEMPLATE.format(
            time_stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            context=text,
        )
        memory_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": formatted},
            {"role": "assistant", "content": "I'll make sure to add the content into the memory."},
        ]
        try:
            self._memory.add(memory_messages, user_id=user_id)
        except Exception as e:
            print(f"  [C5-MAB] write error: {e}")

        self._write_times.append(time.perf_counter() - t0)

    def read(self, query: str, top_k: Optional[int] = None) -> List[MemoryEntry]:
        self._ensure_client()
        t0 = time.perf_counter()
        k = top_k if top_k is not None else self._retrieve_num
        user_id = f"context_{self._context_id}_{self._sub_dataset}"
        self._last_retrieved_dia_ids = []

        try:
            rel = self._memory.search(query=query, user_id=user_id, limit=k)
            results = rel.get("results", []) if isinstance(rel, dict) else (rel or [])
            memories_str = "\n".join(f"- {m.get('memory','')}" for m in results)

            # Propagate dia_ids from metadata if mem0 stored them
            for m in results:
                md = m.get("metadata") or {}
                for did in (md.get("dia_ids") or []) if isinstance(md.get("dia_ids"), list) else []:
                    if did and did not in self._last_retrieved_dia_ids:
                        self._last_retrieved_dia_ids.append(did)

            # MAB's read prompt
            system_prompt = (
                "You are a helpful AI. Answer the question based on query and memories.\n"
                f"{memories_str}\n"
            )
            llm_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query + "\n\nCurrent Time: " +
                 time.strftime("%Y-%m-%d %H:%M:%S")},
            ]
            response = self._oai_client.chat.completions.create(
                model=self._mem0_model,
                messages=llm_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            answer = response.choices[0].message.content or ""

            usage = getattr(response, "usage", None)
            if usage is not None:
                self._token_counts["read_llm_input"] += getattr(usage, "prompt_tokens", 0) or 0
                self._token_counts["read_llm_output"] += getattr(usage, "completion_tokens", 0) or 0
                self._token_counts["read_llm_calls"] += 1
            self._token_counts["read_calls"] += 1
            self._token_counts["read_context_tokens"] += max(len(memories_str) // 4, 0)

        except Exception as e:
            print(f"  [C5-MAB] read error: {e}")
            answer = ""
            results = []

        self._read_times.append(time.perf_counter() - t0)

        # Single parametric MemoryEntry (like our existing C5 read before we
        # split it). Content = generated answer; metadata carries raw facts
        # and dia_ids for P5/P6/P7 downstream.
        return [MemoryEntry(
            content=answer,
            metadata={
                "source": "mem0_mab",
                "dia_ids": self._last_retrieved_dia_ids,
                "raw_memories": [m.get("memory", "") for m in results] if results else [],
            },
            score=1.0,
        )]

    def reset(self):
        # Delete this context's user_id to avoid cross-conv leakage
        try:
            if self._memory:
                user_id = f"context_{self._context_id}_{self._sub_dataset}"
                try:
                    self._memory.delete_all(user_id=user_id)
                except Exception: pass
        except Exception: pass
        self._recent_dia_ids_per_write.clear()
        self._last_retrieved_dia_ids = []
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        for k in list(self._token_counts):
            self._token_counts[k] = 0

    def size(self) -> int:
        return self._write_call_count

    def get_all_entries(self) -> List[MemoryEntry]:
        self._ensure_client()
        user_id = f"context_{self._context_id}_{self._sub_dataset}"
        out: List[MemoryEntry] = []
        try:
            r = self._memory.get_all(user_id=user_id)
            rlist = r.get("results", []) if isinstance(r, dict) else (r or [])
            for item in rlist:
                mem = item.get("memory", "") if isinstance(item, dict) else str(item)
                if mem:
                    out.append(MemoryEntry(content=mem))
        except Exception: pass
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
            "num_entries": len(self.get_all_entries()),
            "mem0_model": self._mem0_model,
            "retrieve_num": self._retrieve_num,
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
