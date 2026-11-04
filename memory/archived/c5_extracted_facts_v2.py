"""C5 — Extracted Facts (Mem0) — delegates to mem0_eval cloned code.

Thin wrapper implementing BaseMemoryUnit interface.
All add/search logic lives in mem0_eval/src/memzero/{add,search}.py
(cloned from mem0-repo/evaluation/ with minimal patches).

Parametric-style: read() returns the final answer directly.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit

# Import mem0_eval cloned code
from ..mem0_eval.src.memzero.add import MemoryADD, _create_local_memory, custom_instructions
from ..mem0_eval.src.memzero.search import MemorySearch


# Known quirk (not patched): gpt-4o-mini occasionally emits UPDATE/DELETE
# actions referencing memory IDs that aren't in the current batch. Upstream
# mem0 simply drops these, which costs a small amount of recall. We used to
# monkey-patch mem0.memory.main to coerce UPDATE-with-unknown-id into ADD
# and skip DELETE-with-unknown-id, but that (a) forks library behaviour
# away from the paper's eval numbers, and (b) empirically hurt F1 by adding
# noisy duplicates. The patch has been removed — we run stock mem0 now.


@register_memory_unit("C5")
class Mem0NativeMemory(BaseMemoryUnit):
    """Mem0 native — delegates to cloned mem0 eval code."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict] = []
        self._indexed = False

        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._mem0_model = extra.get("mem0_model", "gpt-4o-mini")
        self._mem0_top_k = extra.get("mem0_top_k", 10)
        self._gen_model = extra.get("mem0_gen_model", "gpt-4o-mini")
        self._batch_size = extra.get("mem0_batch_size", 2)

        # Lazy init
        self._mem0 = None
        self._adder: Optional[MemoryADD] = None
        self._searcher: Optional[MemorySearch] = None
        self._speaker_a = ""
        self._speaker_b = ""
        self._speaker_a_id = ""
        self._speaker_b_id = ""

        # Timing / stats
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }

    def _ensure_mem0(self):
        if self._mem0 is not None:
            return
        # Load .env
        from pathlib import Path
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

        # Isolate MEM0_DIR per-process so concurrent sweep jobs don't collide
        # on the Qdrant lock file. Users who want a persistent location can
        # set MEM0_DIR_BASE in env; we derive a unique subdir under it.
        # If load_state already pointed us at a restored dir, respect it —
        # otherwise we'd nest into a fresh empty dir and miss the cached data.
        if getattr(self, "_mem0_dir_effective", None):
            unique = self._mem0_dir_effective
        else:
            base = os.environ.get("MEM0_DIR_BASE") or os.environ.get("MEM0_DIR") or \
                os.path.join(os.path.expanduser("~"), ".cache", "submem", "mem0_db")
            unique = os.path.join(base, f"pid{os.getpid()}_{int(time.time()*1000)}")
            os.makedirs(unique, exist_ok=True)
            self._mem0_dir_effective = unique
        os.environ["MEM0_DIR"] = unique

        # Instrument openai client so mem0's internal LLM calls (extract, update,
        # delete classifier; search reranker) get counted in our write/read
        # buckets. Mem0 v2 routes everything through openai.OpenAI().
        self._patch_openai_client()

        # When the outer driver is pointed at a local vLLM server (OPENAI_API_KEY=
        # dummy, OPENAI_BASE_URL=http://localhost...) mem0's internal OpenAI
        # client inherits those and fails because vLLM lacks gpt-4o-mini /
        # text-embedding-3-small. Temporarily restore the real OpenAI creds
        # (saved by the sbatch wrapper) for the duration of mem0 init +
        # subsequent mem0 calls. We patch openai.OpenAI.__init__ so every
        # client mem0 constructs at call time gets real creds.
        real_key = os.environ.get("SUBMEM_REAL_OPENAI_KEY", "")
        real_base = os.environ.get("SUBMEM_REAL_OPENAI_BASE", "https://api.openai.com/v1")
        if real_key:
            try:
                # Only rebind the `OpenAI` symbol inside the three mem0
                # submodules that actually instantiate it. This leaves our
                # SubMem APIModel (which legitimately talks to vLLM) alone.
                from openai import OpenAI as _RealOpenAI, AsyncOpenAI as _RealAsyncOpenAI

                class _Mem0OpenAI(_RealOpenAI):
                    def __init__(self, *a, **kw):
                        kw["api_key"] = real_key
                        kw["base_url"] = real_base
                        super().__init__(*a, **kw)

                class _Mem0AsyncOpenAI(_RealAsyncOpenAI):
                    def __init__(self, *a, **kw):
                        kw["api_key"] = real_key
                        kw["base_url"] = real_base
                        super().__init__(*a, **kw)

                for mod_name in ("mem0.embeddings.openai",
                                 "mem0.llms.openai",
                                 "mem0.llms.openai_structured"):
                    try:
                        import importlib
                        mod = importlib.import_module(mod_name)
                        if hasattr(mod, "OpenAI"):
                            mod.OpenAI = _Mem0OpenAI
                        if hasattr(mod, "AsyncOpenAI"):
                            mod.AsyncOpenAI = _Mem0AsyncOpenAI
                    except Exception:
                        pass
            except Exception:
                pass

        self._mem0 = _create_local_memory(custom_instructions, model=self._mem0_model)
        self._adder = MemoryADD(mem0=self._mem0, batch_size=self._batch_size)
        self._searcher = MemorySearch(self._mem0, top_k=self._mem0_top_k, gen_model=self._gen_model)

    def _patch_openai_client(self):
        if getattr(self, "_openai_patched", False):
            return
        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            return
        orig_create = Completions.create
        outer = self

        def tracked_create(self_client, *args, **kwargs):
            resp = orig_create(self_client, *args, **kwargs)
            # Only count tokens for mem0's real-OpenAI client; skip runner's
            # vLLM client (127.0.0.1) and any other base_url that isn't api.openai.com.
            # self_client is the Completions resource; its parent client holds base_url.
            try:
                parent = getattr(self_client, "_client", None)
                base = str(getattr(parent, "base_url", ""))
                if "openai.com" not in base:
                    return resp
            except Exception:
                return resp
            try:
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    bucket = "write" if not outer._indexed else "read"
                    outer._token_counts[f"{bucket}_llm_input"] += getattr(usage, "prompt_tokens", 0) or 0
                    outer._token_counts[f"{bucket}_llm_output"] += getattr(usage, "completion_tokens", 0) or 0
                    outer._token_counts[f"{bucket}_llm_calls"] = outer._token_counts.get(f"{bucket}_llm_calls", 0) + 1
            except Exception:
                pass
            return resp

        Completions.create = tracked_create
        self._openai_patched = True

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._write_call_count += 1

    def write_experience(
        self,
        query: str,
        trajectory: str,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Override BaseMemoryUnit fallback to preserve Mem0's speaker schema.

        Mem0's eval pipeline assumes dialog turns tagged with `speaker`. Agentic
        benchmarks (BigCodeBench, SWE-bench) pass (query, trajectory) instead of
        turns, so we synthesise a two-turn dialog:
          user    → the task query
          agent   → the model's solution/trajectory
        Both turns share the same session_id so _reconstruct_conversation keeps
        them together. Status (pass/fail) is recorded in metadata for logging.
        """
        md = dict(metadata or {})
        sid = f"exp_{self._write_call_count // 2}"
        base = {
            **md,
            "session_id": sid,
            "session_date": md.get("session_date", ""),
            "query": query,
            "success": success,
        }
        # Turn 1: user asks the query
        self._buffer.append(query)
        self._buffer_metadata.append({**base, "speaker": "user",
                                      "dia_id": f"{sid}:0"})
        # Turn 2: agent replies with the solution/trajectory
        self._buffer.append(trajectory)
        self._buffer_metadata.append({**base, "speaker": "agent",
                                      "dia_id": f"{sid}:1"})
        self._write_call_count += 2

    def _reconstruct_conversation(self) -> dict:
        """Rebuild LoCoMo conversation dict from buffered writes.

        The runner writes individual utterances like "[2023-05-01] Alice: Hello"
        with metadata {speaker, session_id, session_date, dia_id}.
        We reconstruct the original conversation format expected by MemoryADD.
        """
        # Detect speakers
        speakers = sorted(set(
            m.get("speaker", "") for m in self._buffer_metadata if m.get("speaker")
        ))
        if len(speakers) >= 2:
            self._speaker_a, self._speaker_b = speakers[0], speakers[1]
        elif len(speakers) == 1:
            self._speaker_a = speakers[0]
            self._speaker_b = "other"
        else:
            self._speaker_a, self._speaker_b = "speaker_a", "speaker_b"

        conversation = {
            "speaker_a": self._speaker_a,
            "speaker_b": self._speaker_b,
        }

        # Group by session
        sessions: Dict[str, list] = {}
        session_dates: Dict[str, str] = {}
        for i, text in enumerate(self._buffer):
            meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}
            sid = meta.get("session_id", "default")
            if sid not in sessions:
                sessions[sid] = []
                session_dates[sid] = meta.get("session_date", "")

            speaker = meta.get("speaker", "")
            # Strip the "[date] Speaker: " prefix that the runner adds
            raw_text = text
            if "] " in raw_text:
                raw_text = raw_text.split("] ", 1)[1]
            if ": " in raw_text:
                raw_text = raw_text.split(": ", 1)[1]

            sessions[sid].append({
                "speaker": speaker,
                "text": raw_text,
                "dia_id": meta.get("dia_id", ""),
            })

        for sid, turns in sessions.items():
            conversation[sid] = turns
            conversation[f"{sid}_date_time"] = session_dates.get(sid, "")

        return conversation

    def _ensure_indexed(self):
        """Bulk add via MemoryADD — called once on first read()."""
        if self._indexed:
            return
        if not self._buffer:
            self._indexed = True
            return

        self._ensure_mem0()
        t0 = time.perf_counter()

        conversation = self._reconstruct_conversation()
        item = {"conversation": conversation, "qa": []}

        self._speaker_a_id = f"{self._speaker_a}_0"
        self._speaker_b_id = f"{self._speaker_b}_0"

        n = len(self._buffer)
        print(f"  C5 (Mem0): {self._speaker_a} & {self._speaker_b}, {n} messages", flush=True)

        self._adder.process_conversation(item, idx=0)

        total_time = time.perf_counter() - t0
        per_msg = total_time / max(n, 1)
        self._write_times.extend([per_msg] * n)
        self._indexed = True
        print(f"  C5: indexed in {total_time:.0f}s", flush=True)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieval-only: search both speakers' memory banks and return the
        merged fact list as MemoryEntry objects — one entry per fact. The
        benchmark runner renders its own prompt and calls the answer model,
        matching the treatment of C1/C2. This isolates retrieval quality
        from mem0's answer-prompt engineering.

        (Previously this method also rendered mem0's ANSWER_PROMPT and
        called self.model.generate, returning the final answer as a single
        parametric entry — that conflated retrieval and answer-generation
        quality. See mem0's own docs: Memory.search() returns raw facts;
        answer generation is eval-harness scaffolding.)
        """
        self._ensure_indexed()
        if not self._buffer:
            return []

        t_retr0 = time.perf_counter()

        # Search both speakers (via mem0_eval MemorySearch)
        mem_a, _, _ = self._searcher.search_memory(self._speaker_a_id, query)
        mem_b, _, _ = self._searcher.search_memory(self._speaker_b_id, query)

        # Merge + rank by score (higher = better). Cap at top_k.
        all_mems = []
        for m in mem_a:
            all_mems.append(("user", m))
        for m in mem_b:
            all_mems.append(("assistant", m))
        all_mems.sort(key=lambda x: x[1].get("score", 0.0), reverse=True)
        all_mems = all_mems[:top_k]

        self._read_times.append(time.perf_counter() - t_retr0)

        self._token_counts["read_calls"] += 1
        try:
            self._token_counts["read_context_tokens"] += sum(
                len(self.model.encode(m.get("memory", ""))) for _, m in all_mems
            )
        except Exception:
            self._token_counts["read_context_tokens"] += sum(
                len(m.get("memory", "").encode("utf-8")) // 4 for _, m in all_mems
            )

        entries = []
        for speaker, m in all_mems:
            fact = m.get("memory", "")
            ts = m.get("timestamp", "")
            content = f"[{ts}] {fact}" if ts else fact
            entries.append(MemoryEntry(
                content=content,
                metadata={
                    "source": "mem0_eval",
                    "speaker": speaker,
                    "score": m.get("score", 0.0),
                    "dia_ids": m.get("dia_ids", []),
                },
                score=m.get("score", 0.0),
            ))

        return entries

    # ── Lifecycle ──

    def reset(self):
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._indexed = False
        self._speaker_a = ""
        self._speaker_b = ""
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        if self._mem0 is not None:
            for uid in [self._speaker_a_id, self._speaker_b_id]:
                if uid:
                    try:
                        self._mem0.delete_all(user_id=uid)
                    except Exception:
                        pass

    def size(self) -> int:
        if self._mem0 and self._speaker_a_id:
            try:
                count = 0
                for uid in [self._speaker_a_id, self._speaker_b_id]:
                    r = self._mem0.get_all(user_id=uid)
                    if isinstance(r, dict):
                        count += len(r.get("results", []))
                    elif isinstance(r, list):
                        count += len(r)
                return count
            except Exception:
                pass
        return len(self._buffer)

    # ── cache support ─────────────────────────────────────────
    CACHE_KEY_EXTRA = ["mem0_model", "mem0_top_k", "mem0_batch_size"]

    def save_state(self, dir_path: str) -> None:
        """Cache: copy entire MEM0_DIR (sqlite + chroma) + speaker state."""
        import os, json, shutil
        os.makedirs(dir_path, exist_ok=True)
        # Trigger lazy build first
        self._ensure_indexed()
        # mem0's storage dir lives at $MEM0_DIR (default ~/.mem0)
        mem0_root = os.environ.get("MEM0_DIR") or os.path.join(os.path.expanduser("~"), ".mem0")
        cached_dir = os.path.join(dir_path, "mem0_dir")
        if os.path.exists(cached_dir):
            shutil.rmtree(cached_dir)
        if os.path.isdir(mem0_root):
            shutil.copytree(mem0_root, cached_dir)
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            json.dump({
                "buffer": self._buffer,
                "buffer_metadata": self._buffer_metadata,
                "speaker_a": self._speaker_a, "speaker_b": self._speaker_b,
                "speaker_a_id": self._speaker_a_id, "speaker_b_id": self._speaker_b_id,
                "indexed": self._indexed,
                "token_counts": dict(self._token_counts),
                "write_call_count": self._write_call_count,
                "write_times": list(self._write_times),
                "mem0_dir_origin": mem0_root,
            }, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        """Restore by copying cached mem0 dir to a per-PID MEM0_DIR, init mem0."""
        import os, json, shutil, time as _time
        unit = cls(config, model)
        with open(os.path.join(dir_path, "meta.json")) as f:
            d = json.load(f)
        # Fresh per-PID dir for this run so concurrent loads don't collide
        base = os.environ.get("MEM0_DIR_BASE") or os.environ.get("MEM0_DIR") or \
            os.path.join(os.path.expanduser("~"), ".cache", "submem", "mem0_db")
        mem0_root = os.path.join(base, f"pid{os.getpid()}_{int(_time.time()*1000)}")
        os.makedirs(os.path.dirname(mem0_root), exist_ok=True)
        cached_dir = os.path.join(dir_path, "mem0_dir")
        if os.path.isdir(cached_dir):
            if os.path.exists(mem0_root):
                shutil.rmtree(mem0_root)
            shutil.copytree(cached_dir, mem0_root)
        os.environ["MEM0_DIR"] = mem0_root
        unit._mem0_dir_effective = mem0_root
        unit._buffer = d["buffer"]
        unit._buffer_metadata = d["buffer_metadata"]
        unit._speaker_a = d["speaker_a"]
        unit._speaker_b = d["speaker_b"]
        unit._speaker_a_id = d["speaker_a_id"]
        unit._speaker_b_id = d["speaker_b_id"]
        unit._indexed = d["indexed"]
        unit._token_counts = d["token_counts"]
        unit._write_call_count = d["write_call_count"]
        unit._write_times = d["write_times"]
        # Re-init mem0 — it will pick up the restored dir
        unit._ensure_mem0()
        return unit

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_ms = [t * 1000 for t in self._read_times]
        write_ms = [t * 1000 for t in self._write_times]
        stats.update({
            "memory_size_bytes": self.get_external_storage_bytes(),
            "token_consumption": dict(self._token_counts),
            "write_call_count": self._write_call_count,
            "mem0_model": self._mem0_model,
            "mem0_gen_model": self._gen_model,
            "mem0_top_k": self._mem0_top_k,
            "speaker_a": self._speaker_a,
            "speaker_b": self._speaker_b,
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
        # Prefer mem0-extracted facts (the actual stored representation) over
        # raw buffer. Falls back to raw only if mem0 not yet indexed.
        entries = self.get_all_entries()
        if entries:
            return sum(len(e.content.encode("utf-8")) for e in entries)
        return sum(len(t.encode("utf-8")) for t in self._buffer)

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return mem0-extracted facts (compressed store), flushing buffer if
        needed. Falls back to raw buffer only if indexing not possible.
        """
        # Ensure mem0 index is populated so Q1 sees compressed store, not raw.
        if self._buffer and not self._indexed:
            try:
                self._ensure_indexed()
            except Exception as e:
                print(f"  [C5] get_all_entries: ensure_indexed failed: {e}")

        if self._mem0 and self._speaker_a_id:
            entries: List[MemoryEntry] = []
            try:
                for uid in [self._speaker_a_id, self._speaker_b_id]:
                    r = self._mem0.get_all(user_id=uid)
                    rlist = r.get("results", []) if isinstance(r, dict) else (r or [])
                    for item in rlist:
                        mem = item.get("memory", str(item)) if isinstance(item, dict) else str(item)
                        entries.append(MemoryEntry(content=mem))
                if entries:
                    return entries
            except Exception as e:
                print(f"  [C5] mem0.get_all failed: {e}")
        return [MemoryEntry(content=t) for t in self._buffer]
