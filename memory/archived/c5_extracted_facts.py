"""C5 — Mem0 Native Memory (faithful to Mem0 evaluation code).

Uses Mem0's exact pipeline:
  - Dual speaker: user_id per speaker, role=user/assistant perspectives
  - Custom instructions for rich fact extraction
  - Search: query both speakers separately, format per-speaker memories
  - Answer: Mem0's ANSWER_PROMPT (Jinja2 template with speaker-separated memories)
  - Generation: configurable model (default gpt-4o-mini, same as Mem0 paper)

This is a parametric-style unit: read() returns the final answer directly,
bypassing the runner's build_prompt + generate pipeline, so the full Mem0
pipeline (search → per-speaker prompt → generate) is preserved exactly.

Configurable:
  - config.extra["mem0_model"]: LLM for Mem0 internal calls (default: gpt-4o-mini)
  - config.extra["mem0_top_k"]: search limit per speaker (default: 10)
  - config.extra["mem0_gen_model"]: generation model (default: gpt-4o-mini)
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from jinja2 import Template
from openai import OpenAI

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit

# ── Prompts (exact copy from Mem0 evaluation/prompts.py) ──

ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),
   calculate the actual date based on the memory timestamp. For example, if a memory from
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example,
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Answer:
"""

# Custom instructions (exact copy from Mem0 evaluation/src/memzero/add.py)
CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure
"""


def _format_memories(results):
    """Format Mem0 search results with timestamps."""
    formatted = []
    for r in results:
        if isinstance(r, dict):
            mem = r.get("memory", "")
            meta = r.get("metadata") or {}
        elif hasattr(r, "memory"):
            mem = r.memory or ""
            meta = getattr(r, "metadata", None) or {}
        else:
            continue
        ts = meta.get("timestamp", "") if isinstance(meta, dict) else ""
        formatted.append(f"{ts}: {mem}" if ts else mem)
    return formatted


@register_memory_unit("C5")
class Mem0NativeMemory(BaseMemoryUnit):
    """Mem0 native — exact copy of Mem0 eval pipeline."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict] = []
        self._mem0 = None
        self._openai = None
        self._speaker_a = ""
        self._speaker_b = ""
        self._speaker_a_id = ""
        self._speaker_b_id = ""
        self._indexed = False
        self._batch_size = 2

        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._mem0_model = extra.get("mem0_model", "gpt-4o-mini")
        self._mem0_top_k = extra.get("mem0_top_k", 10)
        self._gen_model = extra.get("mem0_gen_model", "gpt-4o-mini")

        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }

    def _load_env(self):
        from pathlib import Path
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

    def _ensure_mem0(self):
        if self._mem0 is not None:
            return
        self._load_env()

        from mem0 import Memory
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self._mem0_model,
                    "temperature": 0.0,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                }
            },
            "custom_prompt": CUSTOM_INSTRUCTIONS,
            "version": "v1.1",
        }
        self._mem0 = Memory.from_config(config)
        self._openai = OpenAI()

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._write_call_count += 1

    def _ensure_indexed(self):
        """Bulk add — called once on first read()."""
        if self._indexed:
            return
        if not self._buffer:
            self._indexed = True
            return

        self._ensure_mem0()
        t0 = time.perf_counter()

        # Detect speakers
        speakers = sorted(set(
            m.get("speaker", "") for m in self._buffer_metadata if m.get("speaker")
        ))
        if len(speakers) >= 2:
            self._speaker_a, self._speaker_b = speakers[0], speakers[1]
        elif len(speakers) == 1:
            self._speaker_a, self._speaker_b = speakers[0], "other"
        else:
            self._speaker_a, self._speaker_b = "speaker_a", "speaker_b"

        self._speaker_a_id = f"{self._speaker_a}_0"
        self._speaker_b_id = f"{self._speaker_b}_0"

        # Delete existing
        for uid in [self._speaker_a_id, self._speaker_b_id]:
            try:
                self._mem0.delete_all(user_id=uid)
            except Exception:
                pass

        n = len(self._buffer)
        print(f"  C5 (Mem0 native): {self._speaker_a} & {self._speaker_b}, {n} messages", flush=True)

        # Group by session
        sessions = {}
        for i, text in enumerate(self._buffer):
            meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}
            sid = meta.get("session_id", "default")
            if sid not in sessions:
                sessions[sid] = {"messages_a": [], "messages_b": [], "timestamp": meta.get("session_date", "")}
            speaker = meta.get("speaker", "")
            if speaker == self._speaker_a:
                sessions[sid]["messages_a"].append({"role": "user", "content": f"{self._speaker_a}: {text}"})
                sessions[sid]["messages_b"].append({"role": "assistant", "content": f"{self._speaker_a}: {text}"})
            else:
                sessions[sid]["messages_a"].append({"role": "assistant", "content": f"{self._speaker_b}: {text}"})
                sessions[sid]["messages_b"].append({"role": "user", "content": f"{self._speaker_b}: {text}"})

        # Add per session, per speaker, in batches (faithful to Mem0 eval)
        total_calls = 0
        for sid, sess in sessions.items():
            ts = sess["timestamp"]
            for label, msgs, uid in [
                ("A", sess["messages_a"], self._speaker_a_id),
                ("B", sess["messages_b"], self._speaker_b_id),
            ]:
                for i in range(0, len(msgs), self._batch_size):
                    batch = msgs[i:i + self._batch_size]
                    try:
                        self._mem0.add(batch, user_id=uid,
                                       metadata={"timestamp": ts} if ts else {})
                        total_calls += 1
                        self._token_counts["write_llm_calls"] += 1
                    except Exception:
                        pass
            print(f"    {sid}: {len(sess['messages_a'])} turns added", flush=True)

        total_time = time.perf_counter() - t0
        per_msg = total_time / max(n, 1)
        self._write_times.extend([per_msg] * n)
        self._indexed = True
        print(f"  C5: {total_calls} API calls in {total_time:.0f}s", flush=True)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Search both speakers → build Mem0 prompt → generate answer.

        Returns a single MemoryEntry whose content is the final answer.
        The runner treats this as parametric (like C8/C9).
        """
        self._ensure_indexed()
        if not self._buffer:
            return []

        t0 = time.perf_counter()
        search_k = self._mem0_top_k

        # Search both speakers separately (faithful to Mem0 eval)
        results_a, results_b = [], []
        for uid, store in [(self._speaker_a_id, results_a), (self._speaker_b_id, results_b)]:
            try:
                results = self._mem0.search(query=query, user_id=uid, limit=search_k)
                if isinstance(results, dict):
                    store.extend(results.get("results", []))
                elif isinstance(results, list):
                    store.extend(results)
            except Exception as e:
                print(f"    [C5] search error {uid}: {e}")

        # Format per-speaker memories with timestamps
        mem_a = _format_memories(results_a)
        mem_b = _format_memories(results_b)

        # Build Mem0's exact prompt (Jinja2 template, speaker-separated)
        template = Template(ANSWER_PROMPT)
        prompt = template.render(
            speaker_1_user_id=self._speaker_a,
            speaker_2_user_id=self._speaker_b,
            speaker_1_memories=json.dumps(mem_a, indent=4),
            speaker_2_memories=json.dumps(mem_b, indent=4),
            question=query,
        )

        # Generate answer with OpenAI (faithful to Mem0 eval)
        response = self._openai.chat.completions.create(
            model=self._gen_model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(mem.encode("utf-8")) for mem in mem_a + mem_b
        )

        # Return answer as single entry (parametric style)
        return [MemoryEntry(
            content=answer,
            metadata={
                "source": "mem0_native",
                "speaker_a_memories": mem_a,
                "speaker_b_memories": mem_b,
            },
            score=1.0,
        )]

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
            "read_context_tokens": 0, "read_calls": 0,
        }
        if self._mem0 is not None:
            for uid in [self._speaker_a_id, self._speaker_b_id]:
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
        return sum(len(t.encode("utf-8")) for t in self._buffer)

    def get_all_entries(self) -> List[MemoryEntry]:
        if self._mem0 and self._speaker_a_id:
            entries = []
            try:
                for uid in [self._speaker_a_id, self._speaker_b_id]:
                    r = self._mem0.get_all(user_id=uid)
                    rlist = r.get("results", []) if isinstance(r, dict) else (r or [])
                    for item in rlist:
                        mem = item.get("memory", str(item)) if isinstance(item, dict) else str(item)
                        entries.append(MemoryEntry(content=mem))
                return entries
            except Exception:
                pass
        return [MemoryEntry(content=t) for t in self._buffer]
