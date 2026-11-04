"""C3-MAB — Letta via MemoryAgentBench (HUST-AI-HYZ) adapter.

Ported from the `AgentWrapper._initialize_letta_agent` and
`AgentWrapper._handle_letta_agent` implementations in
https://github.com/HUST-AI-HYZ/MemoryAgentBench (agent.py, no license
declared — reference-only port).  Unlike our existing C3 Letta
implementation, this variant uses the Letta server client
(``letta.create_client``) exactly as MAB's eval harness does, so numbers
are directly comparable with MAB's published Letta column.

Exposed via the SubMem ``BaseMemoryUnit`` interface:
- ``write(text, metadata)`` → LetalPassageManager.insert_passage or
  ``client.send_message`` depending on ``letta_mode`` (default ``insert``).
- ``read(query, top_k)`` → ``client.send_message`` and returns the agent
  reply as a single ``MemoryEntry`` (parametric-style, like our other
  agentic substrates).

E-metric instrumentation wraps every Letta LLM call so we still populate
E5/E6 (construct) and E10/E11 (mgmt) buckets.
"""
from __future__ import annotations

import json as _json
import os
import time
from typing import Any, Dict, List, Optional

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit


MEMORIZE_TEMPLATE = (
    "Dialogue between User and Assistant {time_stamp}\n"
    "<User> The following context is the conversation between the user and "
    "the assistant:\n{context}\n<Assistant> I have memorized the "
    "conversation and I will answer the question you ask."
)


@register_memory_unit("C3-MAB")
class LettaMABMemory(BaseMemoryUnit):
    """Letta via MemoryAgentBench adapter (ported from HUST-AI-HYZ/MAB)."""

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._model_name = extra.get("letta_model", "gpt-4o-mini")
        self._text_embedding = extra.get("letta_embedding", "text-embedding-3-small")
        self._chunk_size = int(extra.get("letta_chunk_size", 1000))
        self._letta_mode = extra.get("letta_mode", "insert")
        self._system_prompt = extra.get("letta_system_prompt",
            "You are a helpful assistant that can help memorize details in the conversation.")
        self._context_id = int(extra.get("letta_context_id", 0))

        self._client = None
        self._agent_state = None
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "mgmt_llm_input": 0, "mgmt_llm_output": 0, "mgmt_llm_calls": 0,
            "read_llm_input": 0, "read_llm_output": 0, "read_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }
        self._recent_dia_ids: List[str] = []  # track per-write for P6/P7
        self._last_retrieved_dia_ids: List[str] = []

    def _ensure_client(self):
        if self._client is not None:
            return
        # Lazy import — Letta is heavy
        from letta import create_client, LLMConfig, EmbeddingConfig, BasicBlockMemory

        self._client = create_client()
        self._client.set_default_llm_config(LLMConfig.default_config(self._model_name))

        if self._text_embedding == "text-embedding-3-small":
            self._client.set_default_embedding_config(EmbeddingConfig(
                embedding_model="text-embedding-3-small",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=self._chunk_size * 2,
            ))
        else:
            self._client.set_default_embedding_config(
                EmbeddingConfig.default_config(self._text_embedding)
            )

        human_block = self._client.create_block(
            label="human", value="User is sharing the contents they are reading recently.",
            limit=2_000_000,
        )
        persona_block = self._client.create_block(
            label="persona", value=self._system_prompt, limit=2_000_000,
        )
        memory = BasicBlockMemory(blocks=[human_block, persona_block])
        self._agent_state = self._client.create_agent(
            name=f"submem_mab_agent_{os.getpid()}_{int(time.time())}",
            memory=memory,
            system=self._system_prompt,
        )

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._ensure_client()
        t0 = time.perf_counter()
        self._write_call_count += 1

        dia_id = (metadata or {}).get("dia_id", "")
        if dia_id:
            self._recent_dia_ids.append(dia_id)

        formatted = MEMORIZE_TEMPLATE.format(
            time_stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            context=text,
        )

        try:
            if self._letta_mode == "insert":
                # Passage insert — no LLM call, cheap
                self._client.server.passage_manager.insert_passage(
                    agent_state=self._agent_state,
                    agent_id=self._agent_state.id,
                    text=formatted,
                    actor=self._client.user,
                )
            else:
                # chat mode — full agent roundtrip per write
                self._client.send_message(
                    agent_id=self._agent_state.id,
                    message=formatted,
                    role="user",
                )
                # Approximate token accounting (Letta doesn't always surface usage)
                self._token_counts["write_llm_input"] += max(len(formatted) // 4, 0)
                self._token_counts["write_llm_calls"] += 1
        except Exception as e:
            print(f"  [C3-MAB] write error: {e}")

        self._write_times.append(time.perf_counter() - t0)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        self._ensure_client()
        t0 = time.perf_counter()
        self._last_retrieved_dia_ids = []

        try:
            response = self._client.send_message(
                agent_id=self._agent_state.id,
                message=query,
                role="user",
            )
            # MAB extracts from messages[-3].tool_call.arguments.message — same here
            answer = ""
            try:
                msgs = getattr(response, "messages", [])
                for m in reversed(msgs):
                    tc = getattr(m, "tool_call", None)
                    if tc is not None:
                        args = _json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                        if isinstance(args, dict) and args.get("message"):
                            answer = args["message"]; break
                if not answer and msgs:
                    # fall back to last content
                    last = msgs[-1]
                    answer = getattr(last, "content", str(last))
            except Exception:
                answer = str(response)

            # Token counting — Letta doesn't surface usage cleanly
            self._token_counts["read_llm_calls"] += 1
            self._token_counts["read_llm_input"] += max(len(query) // 4, 0)
            self._token_counts["read_llm_output"] += max(len(answer) // 4, 0)
            self._token_counts["read_calls"] += 1
            self._token_counts["read_context_tokens"] += max(len(answer) // 4, 0)

            # Best-effort dia_id attribution: grab recent writes' dia_ids as
            # candidate evidence. Real Letta archival search would expose
            # passage metadata — we don't have that surface from send_message.
            self._last_retrieved_dia_ids = list(self._recent_dia_ids[-top_k:]) if self._recent_dia_ids else []
        except Exception as e:
            print(f"  [C3-MAB] read error: {e}")
            answer = ""

        self._read_times.append(time.perf_counter() - t0)
        return [MemoryEntry(
            content=answer,
            metadata={
                "source": "letta_mab",
                "dia_ids": self._last_retrieved_dia_ids,
            },
            score=1.0,
        )]

    def reset(self):
        # Dispose agent; next write creates a fresh one
        try:
            if self._client and self._agent_state:
                self._client.delete_agent(agent_id=self._agent_state.id)
        except Exception:
            pass
        self._agent_state = None
        self._recent_dia_ids.clear()
        self._last_retrieved_dia_ids = []
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        for k in list(self._token_counts):
            self._token_counts[k] = 0

    def size(self) -> int:
        return self._write_call_count

    def get_all_entries(self) -> List[MemoryEntry]:
        """Best-effort: dump archival passages for Q1 compression accounting."""
        self._ensure_client()
        out: List[MemoryEntry] = []
        try:
            passages = self._client.server.passage_manager.list_passages(
                agent_id=self._agent_state.id,
                actor=self._client.user,
            )
            for p in passages or []:
                text = getattr(p, "text", "") or ""
                if text:
                    out.append(MemoryEntry(content=text))
        except Exception:
            pass
        return out

    def get_external_storage_bytes(self) -> int:
        return sum(len(e.content.encode("utf-8")) for e in self.get_all_entries())

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_ms = [t * 1000 for t in self._read_times]
        write_ms = [t * 1000 for t in self._write_times]
        import numpy as _np
        stats.update({
            "memory_size_bytes": self.get_external_storage_bytes(),
            "num_entries": self._write_call_count,
            "letta_model": self._model_name,
            "letta_mode": self._letta_mode,
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
