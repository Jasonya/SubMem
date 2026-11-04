"""C3 — MemGPT-style Tiered Memory.

Reference: MemGPT / Letta (Packer et al., 2023)
- "MemGPT: Towards LLMs as Operating Systems"
- https://github.com/letta-ai/letta

Faithful to Letta's memory architecture:

Main Context (always in-context):
  1. Core Memory: structured <human> + <persona> blocks, editable via
     LLM function calls (core_memory_append / core_memory_replace).
     Updated EVERY turn (not just on eviction) — the agent continuously
     decides what to persist.
  2. FIFO Buffer: recent messages queue.

External Context (out-of-context, explicit retrieval):
  3. Recall Storage: complete conversation history (hybrid search).
  4. Archival Storage: evicted messages → embed → FAISS (semantic search).

Write: append to FIFO buffer + LLM updates core memory blocks each turn.
       On overflow → evict oldest 30% → LLM summarises evicted messages →
       summary kept in buffer → evicted messages stored in archival + recall.
Read:  core memory blocks (always) + recent buffer + archival search + recall search.
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import register_memory_unit, MemoryEntry
from .dense_retrieval import DenseRetrievalMemory

_DEFAULT_BUFFER_SIZE = 50
_EVICT_FRACTION = 0.3          # Letta keeps 70%, evicts 30%
_CORE_MEMORY_MAX_CHARS = 5000  # per block (Letta: 20K–100K, we use 5K for benchmark)

# ── Batched core-memory update (called every N turns) ────────

_CORE_MEMORY_UPDATE_PROMPT = """You are managing an agent's core memory — a structured store of key facts.
After reading the new messages, decide what to ADD or REPLACE in core memory.

Current core memory:
<human>
{human_block}
</human>
<persona>
{persona_block}
</persona>

New messages to process:
{new_messages}

Instructions:
- The <human> block stores key facts about the user(s): names, preferences, relationships, important dates, goals.
- The <persona> block stores the agent's role and conversation context.
- Only store important, reusable facts — not transient chat content.
- For each change, output one line in this format:
  APPEND human: <text to add>
  REPLACE human: <old text> -> <new text>
  APPEND persona: <text to add>
  REPLACE persona: <old text> -> <new text>
- If no changes needed, output: NONE

Changes:"""

# ── Eviction summary prompt ─────────────────────────────────

_EVICTION_SUMMARY_PROMPT = """Summarise the following conversation messages into a concise paragraph (≤300 words).
Preserve key facts, dates, names, decisions, and any information the user would want remembered.

Messages:
{messages}

Summary:"""

# ── Search decision prompt ───────────────────────────────────

_SEARCH_DECIDE_PROMPT = """You are a memory retrieval agent. Given a user question and context, decide what to search.

Core memory:
<human>
{human_block}
</human>

Recent messages:
{recent}

User question: {query}

You have two search tools:
1. archival_memory_search: semantic search over stored knowledge (use for facts, events, preferences)
2. conversation_search: hybrid search over full conversation history (use for specific quotes, exact phrases, or recent context)

Respond with ONE line:
- ARCHIVAL: <search query>     (semantic search)
- RECALL: <search query>       (hybrid search for exact phrases)
- BOTH: <search query>         (search both)
- NONE                         (answer is in core memory or recent messages)

Search decision:"""

# ── How often to update core memory (every N writes) ─────────
_CORE_UPDATE_INTERVAL = 5  # batch 5 turns then update


@register_memory_unit("C3")
class RawTextMemory(DenseRetrievalMemory):
    """MemGPT-style tiered memory with structured core memory blocks.

    Core memory (<human> + <persona>) is always in-context and updated
    every N turns via LLM function calls. Archival storage uses FAISS
    semantic search. Recall storage uses hybrid (embedding + text) search.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self._fifo_buffer: List[Dict[str, Any]] = []
        # Core memory: structured blocks (MemGPT human/persona)
        self._core_human: str = ""    # key facts about users
        self._core_persona: str = ""  # agent role/context
        # Recall storage: full conversation history for hybrid search
        self._recall_storage: List[Dict[str, Any]] = []
        self._recall_embeddings: Optional[np.ndarray] = None  # for hybrid search
        self._max_buffer = (
            config.extra.get("max_buffer_size", _DEFAULT_BUFFER_SIZE)
            if hasattr(config, "extra") and config.extra
            else _DEFAULT_BUFFER_SIZE
        )
        self._eviction_count = 0
        self._core_update_times: List[float] = []
        # Pending messages for batched core-memory update
        self._pending_for_core: List[str] = []
        self._write_turn_count = 0

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()

        # Always store in recall (complete conversation history)
        emb = self._encode([text])[0]
        self._recall_storage.append({
            "text": text,
            "metadata": metadata or {},
            "embedding": emb,
        })
        # Update recall embedding matrix
        if self._recall_embeddings is None:
            self._recall_embeddings = emb.reshape(1, -1)
        else:
            self._recall_embeddings = np.vstack([self._recall_embeddings, emb.reshape(1, -1)])

        # Append to FIFO buffer
        self._fifo_buffer.append({
            "text": text,
            "metadata": metadata or {},
        })

        # Accumulate for batched core-memory update
        self._pending_for_core.append(text)
        self._write_turn_count += 1

        # Update core memory every N turns
        if len(self._pending_for_core) >= _CORE_UPDATE_INTERVAL:
            batch_text = "\n".join(self._pending_for_core)
            self._update_core_memory(batch_text)
            self._pending_for_core.clear()

        # Check if buffer overflow → evict + summarise + archive
        if len(self._fifo_buffer) > self._max_buffer:
            self._evict()

        self._write_times.append(time.perf_counter() - t0)

    def _evict(self):
        """Evict oldest 30%: summarise → keep summary in buffer → archive in FAISS."""
        n_evict = int(len(self._fifo_buffer) * _EVICT_FRACTION)
        if n_evict == 0:
            return

        evicted = self._fifo_buffer[:n_evict]
        self._fifo_buffer = self._fifo_buffer[n_evict:]

        # Archive evicted messages in vector store (archival_memory_insert)
        for msg in evicted:
            self._add_to_index(msg["text"], metadata=msg["metadata"])

        # Flush any pending core-memory updates before eviction
        if self._pending_for_core:
            batch_text = "\n".join(self._pending_for_core)
            self._update_core_memory(batch_text)
            self._pending_for_core.clear()

        # Generate summary of evicted messages (Letta-style compaction)
        evicted_text = "\n".join(msg["text"] for msg in evicted)
        summary = self._summarise_evicted(evicted_text)

        # Insert summary as first message in buffer (like Letta's system_alert)
        self._fifo_buffer.insert(0, {
            "text": f"[Summary of prior conversation] {summary}",
            "metadata": {"source": "eviction_summary", "num_summarised": n_evict},
        })

        self._eviction_count += 1

    def _summarise_evicted(self, evicted_text: str) -> str:
        """Generate a summary of evicted messages (Letta compaction)."""
        prompt = _EVICTION_SUMMARY_PROMPT.format(
            messages=evicted_text[:3000],
        )
        t0 = time.perf_counter()
        summary = self.model.generate(prompt, max_new_tokens=256)
        self._core_update_times.append(time.perf_counter() - t0)
        self._track_mgmt_call(prompt, summary)
        return summary.strip()

    def _update_core_memory(self, new_messages: str):
        """LLM-driven core memory update (MemGPT function calls)."""
        prompt = _CORE_MEMORY_UPDATE_PROMPT.format(
            human_block=self._core_human or "(empty)",
            persona_block=self._core_persona or "(empty)",
            new_messages=new_messages[:3000],
        )
        t0 = time.perf_counter()
        response = self.model.generate(prompt, max_new_tokens=256)
        self._core_update_times.append(time.perf_counter() - t0)
        self._track_mgmt_call(prompt, response)

        if "NONE" in response.strip().upper().split("\n")[0]:
            return

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line_upper = line.upper()
            if line_upper.startswith("APPEND HUMAN:"):
                text = line[len("APPEND human:"):].strip()
                if text and len(self._core_human) + len(text) < _CORE_MEMORY_MAX_CHARS:
                    self._core_human += ("\n" + text if self._core_human else text)
            elif line_upper.startswith("APPEND PERSONA:"):
                text = line[len("APPEND persona:"):].strip()
                if text and len(self._core_persona) + len(text) < _CORE_MEMORY_MAX_CHARS:
                    self._core_persona += ("\n" + text if self._core_persona else text)
            elif line_upper.startswith("REPLACE HUMAN:"):
                self._apply_replace(line[len("REPLACE human:"):].strip(), "human")
            elif line_upper.startswith("REPLACE PERSONA:"):
                self._apply_replace(line[len("REPLACE persona:"):].strip(), "persona")

    def _apply_replace(self, replace_str: str, block: str):
        """Apply a REPLACE operation: 'old text -> new text' (exact match, like Letta)."""
        if "->" not in replace_str:
            return
        parts = replace_str.split("->", 1)
        old_text = parts[0].strip()
        new_text = parts[1].strip()
        if not old_text or not new_text:
            return
        # Letta uses exact match str.replace(); we do case-insensitive for robustness
        if block == "human" and old_text.lower() in self._core_human.lower():
            idx = self._core_human.lower().find(old_text.lower())
            if idx >= 0:
                self._core_human = (
                    self._core_human[:idx] + new_text +
                    self._core_human[idx + len(old_text):]
                )
        elif block == "persona" and old_text.lower() in self._core_persona.lower():
            idx = self._core_persona.lower().find(old_text.lower())
            if idx >= 0:
                self._core_persona = (
                    self._core_persona[:idx] + new_text +
                    self._core_persona[idx + len(old_text):]
                )

    def _recall_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Hybrid search over recall storage: embedding similarity + text overlap.

        Like Letta's conversation_search which uses RRF (Reciprocal Rank Fusion)
        to combine vector search and full-text search results.
        """
        if not self._recall_storage:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Semantic scores
        q_emb = self._encode([query])[0]
        if self._recall_embeddings is not None and len(self._recall_embeddings) > 0:
            sem_scores = self._recall_embeddings @ q_emb  # cosine (already L2-normalised)
        else:
            sem_scores = np.zeros(len(self._recall_storage))

        # Text overlap scores (normalised)
        text_scores = np.zeros(len(self._recall_storage))
        for i, msg in enumerate(self._recall_storage):
            text_lower = msg["text"].lower()
            text_words = set(text_lower.split())
            overlap = len(query_words & text_words)
            if query_words:
                text_scores[i] = overlap / len(query_words)

        # RRF-style fusion: rank by semantic, rank by text, combine
        n = len(self._recall_storage)
        k_rrf = 60  # standard RRF constant

        sem_order = np.argsort(-sem_scores)
        text_order = np.argsort(-text_scores)

        sem_rank = np.zeros(n)
        text_rank = np.zeros(n)
        for rank, idx in enumerate(sem_order):
            sem_rank[idx] = rank + 1
        for rank, idx in enumerate(text_order):
            text_rank[idx] = rank + 1

        rrf_scores = 1.0 / (k_rrf + sem_rank) + 1.0 / (k_rrf + text_rank)
        top_indices = np.argsort(-rrf_scores)[:top_k]

        return [
            MemoryEntry(
                content=self._recall_storage[i]["text"],
                metadata={**self._recall_storage[i]["metadata"], "source": "recall_storage"},
                score=float(rrf_scores[i]),
            )
            for i in top_indices
            if rrf_scores[i] > 0
        ]

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """MemGPT-style read: core memory + buffer + LLM-directed search.

        Core memory is ALWAYS included (like Letta's system prompt injection).
        The LLM decides whether to search archival, recall, both, or neither.
        """
        t0 = time.perf_counter()

        # Flush any remaining pending core-memory updates
        if self._pending_for_core:
            batch_text = "\n".join(self._pending_for_core)
            self._update_core_memory(batch_text)
            self._pending_for_core.clear()

        results = []

        # 1. Core memory blocks (always in-context, like Letta system prompt)
        core_parts = []
        if self._core_human:
            core_parts.append(f"<human>\n{self._core_human}\n</human>")
        if self._core_persona:
            core_parts.append(f"<persona>\n{self._core_persona}\n</persona>")
        if core_parts:
            results.append(MemoryEntry(
                content=f"[Core Memory]\n" + "\n".join(core_parts),
                metadata={"source": "core_memory"},
                score=2.0,
            ))

        # 2. Recent buffer (last few messages)
        max_buffer_return = min(3, len(self._fifo_buffer))
        recent_texts = []
        for msg in self._fifo_buffer[-max_buffer_return:]:
            results.append(MemoryEntry(
                content=msg["text"],
                metadata={**msg["metadata"], "source": "fifo_buffer"},
                score=1.5,
            ))
            recent_texts.append(msg["text"])

        # 3. LLM-directed search (MemGPT agentic retrieval)
        search_budget = max(top_k - len(results), 1)
        has_archival = self._index.ntotal > 0
        has_recall = len(self._recall_storage) > 0

        if has_archival or has_recall:
            prompt = _SEARCH_DECIDE_PROMPT.format(
                human_block=self._core_human[:500] if self._core_human else "(empty)",
                recent="\n".join(recent_texts[-3:]) if recent_texts else "(none)",
                query=query,
            )
            search_response = self.model.generate(prompt, max_new_tokens=32)
            decision = search_response.strip().split("\n")[0].strip()

            # Track retrieval LLM cost
            self._token_counts.setdefault("read_llm_input", 0)
            self._token_counts.setdefault("read_llm_output", 0)
            self._token_counts.setdefault("read_llm_calls", 0)
            self._token_counts["read_llm_input"] += len(self.model.encode(prompt))
            self._token_counts["read_llm_output"] += len(self.model.encode(search_response))
            self._token_counts["read_llm_calls"] += 1

            decision_upper = decision.upper()
            search_query = decision.split(":", 1)[1].strip() if ":" in decision else query

            if decision_upper.startswith("ARCHIVAL") and has_archival:
                archival_results = super().read(search_query, top_k=search_budget)
                results.extend(archival_results)
            elif decision_upper.startswith("RECALL") and has_recall:
                recall_results = self._recall_search(search_query, top_k=search_budget)
                results.extend(recall_results)
            elif decision_upper.startswith("BOTH"):
                if has_archival:
                    archival_results = super().read(search_query, top_k=search_budget // 2 + 1)
                    results.extend(archival_results)
                if has_recall:
                    recall_results = self._recall_search(search_query, top_k=search_budget // 2 + 1)
                    results.extend(recall_results)
            elif decision_upper != "NONE":
                # Fallback: search archival with original query
                if has_archival:
                    results.extend(super().read(query, top_k=search_budget))

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(self.model.encode(r.content)) for r in results
        )
        return results

    def _on_reset(self):
        self._fifo_buffer.clear()
        self._core_human = ""
        self._core_persona = ""
        self._recall_storage.clear()
        self._recall_embeddings = None
        self._eviction_count = 0
        self._core_update_times.clear()
        self._pending_for_core.clear()
        self._write_turn_count = 0

    def size(self) -> int:
        return len(self._fifo_buffer) + self._index.ntotal

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "fifo_buffer_size": len(self._fifo_buffer),
            "archival_entries": self._index.ntotal,
            "recall_entries": len(self._recall_storage),
            "core_human_length": len(self._core_human),
            "core_persona_length": len(self._core_persona),
            "eviction_count": self._eviction_count,
            "compression_ratio": 1.0,
            "core_update_interval": _CORE_UPDATE_INTERVAL,
            "total_write_turns": self._write_turn_count,
            "avg_core_update_ms": (
                sum(self._core_update_times) / len(self._core_update_times) * 1000
                if self._core_update_times else 0.0
            ),
        })
        return stats

    def get_all_entries(self) -> List[MemoryEntry]:
        entries = []
        for msg in self._fifo_buffer:
            entries.append(MemoryEntry(
                content=msg["text"],
                metadata={**msg["metadata"], "source": "fifo_buffer"},
            ))
        entries.extend(super().get_all_entries())
        return entries
