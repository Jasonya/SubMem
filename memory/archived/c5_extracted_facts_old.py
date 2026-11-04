"""C5 — Extracted Facts Memory (Mem0-style).

Reference: Mem0 (Chhikara et al., 2025)
- "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"

Key adaptations from the paper:
- 4-operation memory management: ADD / UPDATE / DELETE / NOOP
- Semantic deduplication: new facts checked against existing via retrieval
- Conflict resolution: LLM-based classifier decides operation per fact
- Uses self.model (local Qwen) for classification; swap to GPT-4o-mini by
  replacing _classify_operation's model call

Pipeline: buffer utterances → 256-tok chunks → LLM extract facts →
          retrieve similar → LLM classify operation → apply → FAISS
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import MemoryEntry, register_memory_unit
from .dense_retrieval import DenseRetrievalMemory
# No chunking — Mem0 processes per-message

_DEFAULT_CHUNK_SIZE = 256

# Mem0-style extraction prompt: JSON output with few-shot examples
# Faithful to github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py
_FACT_PROMPT = """You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.

Types of Information to Remember:
1. Store Personal Preferences: likes, dislikes, and specific preferences.
2. Maintain Important Personal Details: names, relationships, and important dates.
3. Track Plans and Intentions: upcoming events, trips, goals, and plans.
4. Remember Activity and Service Preferences: dining, travel, hobbies.
5. Monitor Health and Wellness Preferences: dietary restrictions, fitness routines.
6. Store Professional Details: job titles, work habits, career goals.
7. Miscellaneous Information Management: favorite books, movies, brands, etc.

Here are some few shot examples:

Input: Hi.
Output: {{"facts" : []}}

Input: There are branches in trees.
Output: {{"facts" : []}}

Input: Hi, I am looking for a restaurant in San Francisco.
Output: {{"facts" : ["Looking for a restaurant in San Francisco"]}}

Input: Yesterday, I had a meeting with John at 3pm. We discussed the new project.
Output: {{"facts" : ["Had a meeting with John at 3pm", "Discussed the new project"]}}

Input: Hi, my name is John. I am a software engineer.
Output: {{"facts" : ["Name is John", "Is a Software engineer"]}}

Input: Me favourite movies are Inception and Interstellar.
Output: {{"facts" : ["Favourite movies are Inception and Interstellar"]}}

Return the facts in a json format as shown above.

Remember:
- If you do not find anything relevant, return an empty list for the "facts" key.
- Make sure to return the response in json with a key "facts" and a list of strings.
- Only extract facts from the NEW exchange below; use summary and recent messages only for context.

Conversation summary so far:
{summary}

Recent messages:
{recent}

New exchange:
{chunk}

Output:"""

_SUMMARY_PROMPT = """Summarize the following conversation in 2-3 sentences, focusing on key facts, events, and relationships between participants. If there is an existing summary, update it with new information.

Existing summary:
{existing_summary}

New messages:
{new_messages}

Updated summary:"""

# Mem0-style batch update prompt: one LLM call for all new facts
# Faithful to github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py
_BATCH_UPDATE_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) ADD into the memory, (2) UPDATE the memory, (3) DELETE from the memory, and (4) NONE (no change).

Guidelines:
- ADD: The retrieved fact contains new information not present in existing memories.
- UPDATE: The retrieved fact updates or enriches an existing memory. Keep the same ID, update the text.
- DELETE: The retrieved fact contradicts an existing memory.
- NONE: The retrieved fact is already captured. No action needed.

Below is the current content of my memory:
{existing_memories}

The new retrieved facts are:
{new_facts}

Analyze each new fact and decide ADD, UPDATE, DELETE, or NONE for each. Return a JSON response:
{{"memory": [
  {{"id": "<existing or new ID>", "text": "<memory content>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<old content if UPDATE>"}},
  ...
]}}

Rules:
- For ADD, generate a new integer ID (next available).
- For UPDATE/DELETE, use the existing memory ID.
- For NONE, include existing memories unchanged.
- Return ONLY the JSON, nothing else."""


@register_memory_unit("C5")
class ExtractedFactsMemory(DenseRetrievalMemory):
    """Extracted atomic facts with Mem0-style LLM-based dedup + dense retrieval."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict[str, Any]] = []
        self._indexed = False
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._total_facts_extracted = 0
        self._op_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
        self._llm_call_times: List[float] = []
        self._classify_times: List[float] = []
        self._chunk_size = (
            config.extra.get("chunk_size", _DEFAULT_CHUNK_SIZE)
            if hasattr(config, "extra") and config.extra
            else _DEFAULT_CHUNK_SIZE
        )
        # Number of similar memories to retrieve for classification
        self._classify_top_k = 5
        # Mem0-style context: running summary + recent message buffer
        self._summary = ""
        self._recent_window = 10  # last 10 messages for context

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._indexed = False

    def _update_summary(self, new_messages: str):
        """Maintain a running conversation summary (Mem0-style)."""
        prompt = _SUMMARY_PROMPT.format(
            existing_summary=self._summary or "(none yet)",
            new_messages=new_messages[:2000],  # truncate to avoid overly long prompt
        )
        self._summary = self.model.generate(prompt, max_new_tokens=256)
        self._track_llm_call(prompt, self._summary)

    def _extract_facts(self, chunk: str, summary: str = "", recent: str = "") -> List[str]:
        """Extract atomic facts using Mem0-style JSON output with few-shot examples."""
        import json as _json

        prompt = _FACT_PROMPT.format(
            summary=summary or "(no summary yet)",
            recent=recent or "(no recent messages)",
            chunk=chunk,
        )
        t0 = time.perf_counter()
        response = self.model.generate(prompt, max_new_tokens=256)
        self._llm_call_times.append(time.perf_counter() - t0)
        self._track_llm_call(prompt, response)

        # Parse JSON response {"facts": ["fact1", "fact2", ...]}
        facts = []
        try:
            # Find JSON in response (may have extra text around it)
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(text[start:end])
                raw_facts = parsed.get("facts", [])
                facts = [f.strip() for f in raw_facts if isinstance(f, str) and len(f.strip()) > 5]
        except (_json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: line-based parsing if JSON fails
        if not facts:
            for line in response.strip().split("\n"):
                line = line.strip()
                if not line or len(line) <= 5:
                    continue
                for prefix in ["- ", "* ", "• "]:
                    if line.startswith(prefix):
                        line = line[len(prefix):]
                        break
                line = line.strip()
                if line and len(line) > 5 and "facts" not in line.lower():
                    facts.append(line)
        return facts

    def _get_similar_memories(self, fact_emb: np.ndarray) -> List[tuple]:
        """Retrieve top-k similar existing memories for classification context.

        Returns list of (index, score, content) tuples.
        """
        if self._index.ntotal == 0:
            return []
        k = min(self._classify_top_k, self._index.ntotal)
        scores, indices = self._index.search(fact_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            if entry.content == "[DELETED]":
                continue
            results.append((int(idx), float(score), entry.content))
        return results

    def _batch_classify(self, facts: List[str], all_similar: List[List[tuple]]) -> List[tuple]:
        """Mem0-style batch classification: one LLM call for all facts.

        Returns list of (operation, target_idx) per fact.
        """
        import json as _json

        # Collect all unique existing memories referenced by any fact
        seen_ids = set()
        existing_mems = []
        for similar in all_similar:
            for idx, score, content in similar:
                if idx not in seen_ids:
                    seen_ids.add(idx)
                    existing_mems.append({"id": str(idx), "text": content})

        if not existing_mems and all(not s for s in all_similar):
            # No existing memories at all — everything is ADD
            return [("ADD", None) for _ in facts]

        existing_str = _json.dumps(existing_mems, indent=2) if existing_mems else "[]"
        facts_str = _json.dumps(facts, indent=2)

        prompt = _BATCH_UPDATE_PROMPT.format(
            existing_memories=existing_str,
            new_facts=facts_str,
        )

        t0 = time.perf_counter()
        response = self.model.generate(prompt, max_new_tokens=512)
        self._classify_times.append(time.perf_counter() - t0)
        self._track_mgmt_call(prompt, response)

        return self._parse_batch_response(response, facts, seen_ids)

    def _parse_batch_response(self, response: str, facts: List[str],
                              valid_ids: set) -> List[tuple]:
        """Parse batch JSON response into list of (operation, target_idx)."""
        import json as _json

        results = []
        try:
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(text[start:end])
                memory_list = parsed.get("memory", [])
                for item in memory_list:
                    event = item.get("event", "NONE").upper()
                    if event not in ("ADD", "UPDATE", "DELETE", "NONE"):
                        event = "NONE"
                    target_idx = None
                    if event in ("UPDATE", "DELETE"):
                        try:
                            mid = int(item.get("id", -1))
                            if mid in valid_ids:
                                target_idx = mid
                            else:
                                event = "ADD"  # invalid ID, fall back to ADD
                        except (ValueError, TypeError):
                            event = "ADD"
                    # Map NONE → NOOP for internal consistency
                    if event == "NONE":
                        event = "NOOP"
                    results.append((event, target_idx))
        except (_json.JSONDecodeError, KeyError, TypeError):
            pass

        # If parsing failed or returned wrong count, fall back to all ADD
        if len(results) != len(facts):
            # Try to match as many as we can, ADD the rest
            while len(results) < len(facts):
                results.append(("ADD", None))
            results = results[:len(facts)]

        return results

    def _apply_operation(self, op: str, target_idx: Optional[int],
                         fact_text: str, fact_emb: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None):
        """Apply the classified operation to the memory store."""
        if op == "NOOP":
            return

        if op in ("DELETE", "UPDATE") and target_idx is not None:
            # Soft-delete the old entry
            if 0 <= target_idx < len(self._entries):
                self._entries[target_idx] = MemoryEntry(
                    content="[DELETED]", metadata={"deleted": True}, score=0.0
                )

        # ADD new fact (also after DELETE/UPDATE)
        self._index.add(fact_emb)
        self._entries.append(MemoryEntry(
            content=fact_text,
            metadata=metadata or {},
            timestamp=len(self._entries),
        ))

    def _ensure_indexed(self):
        if self._indexed:
            return
        if not self._buffer:
            self._indexed = True
            return

        t0 = time.perf_counter()
        self._raw_token_count = sum(len(self.model.encode(t)) for t in self._buffer)

        # Group into message pairs (Mem0-style: each add() call = one interaction unit)
        # Consecutive pairs of utterances form one interaction
        pairs = []
        for i in range(0, len(self._buffer), 2):
            pair_texts = [self._buffer[i]]
            pair_dia_ids = []
            meta_i = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}
            if meta_i.get("dia_id"):
                pair_dia_ids.append(meta_i["dia_id"])
            if i + 1 < len(self._buffer):
                pair_texts.append(self._buffer[i + 1])
                meta_j = self._buffer_metadata[i + 1] if i + 1 < len(self._buffer_metadata) else {}
                if meta_j.get("dia_id"):
                    pair_dia_ids.append(meta_j["dia_id"])
            pairs.append(("\n".join(pair_texts), pair_dia_ids))

        n_pairs = len(pairs)
        print(f"  C5: extracting facts from {n_pairs} message pairs (Mem0-style)...", flush=True)

        for pi, (pair_text, pair_dia_ids) in enumerate(pairs):
            # Build recent message context (last 10 messages before current pair)
            msg_start = max(0, pi * 2 - self._recent_window)
            msg_end = pi * 2
            recent_msgs = "\n".join(self._buffer[msg_start:msg_end]) if msg_end > msg_start else ""

            facts = self._extract_facts(pair_text, summary=self._summary, recent=recent_msgs)

            # Update running summary every 10 pairs (amortize cost)
            if (pi + 1) % 10 == 0 or pi == 0:
                self._update_summary(pair_text)
            self._total_facts_extracted += len(facts)
            if (pi + 1) % 20 == 0 or pi + 1 == n_pairs:
                print(f"    [{pi+1}/{n_pairs}] {self._total_facts_extracted} facts, "
                      f"{sum(self._op_counts.values())} classified", flush=True)

            if not facts:
                continue

            # Encode all facts and retrieve similar memories
            fact_embs = [self._encode([f]) for f in facts]
            all_similar = [self._get_similar_memories(emb) for emb in fact_embs]

            # Batch classify: one LLM call for all facts from this pair
            operations = self._batch_classify(facts, all_similar)

            for fi, (fact, fact_emb, (op, target_idx)) in enumerate(
                zip(facts, fact_embs, operations)
            ):
                self._op_counts[op] += 1
                fact_meta = {
                    "source": "extracted_fact",
                    "fact_index": fi,
                    "operation": op,
                    "dia_ids": pair_dia_ids,
                }
                self._apply_operation(op, target_idx, fact, fact_emb, metadata=fact_meta)
                if op != "NOOP":
                    self._stored_token_count += len(self.model.encode(fact))

        total_ms = (time.perf_counter() - t0)
        n_entries = sum(1 for e in self._entries if e.content != "[DELETED]") or 1
        per_entry = total_ms / n_entries
        self._write_times.extend([per_entry] * n_entries)
        self._indexed = True

    def read(self, query: str, top_k: int = 5):
        self._ensure_indexed()
        if self._index.ntotal == 0:
            return []
        t0 = time.perf_counter()
        q_emb = self._encode([query])
        # Request extra to compensate for deleted entries
        n_deleted = self._op_counts["DELETE"] + self._op_counts["UPDATE"]
        k = min(top_k + n_deleted, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            if entry.content == "[DELETED]":
                continue
            results.append(MemoryEntry(
                content=entry.content,
                metadata=entry.metadata,
                score=float(score),
                timestamp=entry.timestamp,
            ))
            if len(results) >= top_k:
                break
        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(self.model.encode(r.content)) for r in results
        )
        return results

    def _on_reset(self):
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._indexed = False
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._total_facts_extracted = 0
        self._op_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
        self._llm_call_times.clear()
        self._classify_times.clear()
        self._summary = ""

    def size(self) -> int:
        return sum(1 for e in self._entries if e.content != "[DELETED]")

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_indexed()
        stats = super().get_stats()
        cr = (self._raw_token_count / self._stored_token_count
              if self._stored_token_count > 0 else 1.0)
        stats["compression_ratio"] = round(cr, 2)
        stats["raw_token_count"] = self._raw_token_count
        stats["stored_token_count"] = self._stored_token_count
        stats["total_facts_extracted"] = self._total_facts_extracted
        stats["dedup_ops"] = dict(self._op_counts)
        stats["num_raw_utterances"] = len(self._buffer)
        stats["avg_extraction_llm_ms"] = (
            sum(self._llm_call_times) / len(self._llm_call_times) * 1000
            if self._llm_call_times else 0.0
        )
        stats["avg_classify_llm_ms"] = (
            sum(self._classify_times) / len(self._classify_times) * 1000
            if self._classify_times else 0.0
        )
        return stats

    def get_all_entries(self) -> List[MemoryEntry]:
        return [e for e in self._entries if e.content != "[DELETED]"]
