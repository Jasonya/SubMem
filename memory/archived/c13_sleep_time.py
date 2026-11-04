"""C13 — Sleep-time Compute Memory (chunked adaptation for long dialog).

Reference: Sleep-time Compute (Luo et al., 2025, Letta + UC Berkeley)
- "Sleep-time Compute: Beyond Inference Scaling at Test-time"
- https://github.com/letta-ai/sleep-time-compute

Design note / deviation from paper:
- The paper's sleep-time agent holds ONE shared ``rethink_memory_block``
  capped at 5000 chars, iteratively refined via a tool-calling loop.
  That design is evaluated on AIME / GSM-Symbolic where the context is
  a few hundred tokens, so a single 5000-char block preserves all
  relevant information.
- LoCoMo conversations are ~419 turns / ~21k tokens. A single 5000-char
  block would discard ~95% of the context → unusable. We therefore use
  a chunked adaptation:
    * Split the transcript into ~500-word chunks (~28 chunks per convo).
    * Produce ONE rethink block per chunk via an iterative rethink
      (round 1 = initial, rounds 2+ = refine-overwrite — faithful to the
      paper's rethink_memory tool semantics, just applied per-chunk).
    * Index each block in FAISS so read() can dense-search the relevant
      pre-computed reasoning at query time.
- This is a faithful ADAPTATION of the paper's method for long dialog,
  NOT a reproduction of the paper's exact architecture. Results on
  LoCoMo should be interpreted accordingly.

Pipeline:
  write: buffer utterances → group into ~500-word chunks
  _ensure_indexed (sleep): per-chunk rounds-of-rethink via batched generate
  read: embed query → FAISS search over rethink blocks → return pre-computed reasoning
"""
import time
from typing import Any, Dict, List, Optional

from .base import MemoryEntry, register_memory_unit
from .dense_retrieval import DenseRetrievalMemory

_DEFAULT_CHUNK_WORDS = 500
# Luo et al. 2025 "Sleep-time Compute" uses up to 10 iterative rethink
# rounds per chunk in their Letta agent config. For LoCoMo wall-time
# we ship 2 here; override via extra.rethink_rounds=10 for paper-faithful.
_DEFAULT_RETHINK_ROUNDS = 2


# Initial rethink prompt (round 1) — faithful to the paper's intent:
# "think about situations and come up with useful quantities, inferences,
#  calculations and reasoning to prepare for future questions."
_RETHINK_INITIAL_PROMPT = """You are an expert reasoning agent. Your task is to think about a conversation passage and pre-compute useful information for answering future questions about it.

Given the passage below, produce a comprehensive "rethink memory block" containing:
1. Key facts and entities mentioned (names, dates, places, numbers)
2. Relationships between people/entities
3. Temporal details and sequence of events
4. Inferences that can be drawn from the content
5. Potential questions someone might ask and pre-computed answers
6. Any calculations, cross-references, or reasoning that would help answer questions quickly

Be thorough and verbose. Include everything that could be useful.

Passage:
{chunk}

Rethink memory block:"""


# Refine prompt (rounds 2+) — "add new calculations, double check previous
# calculations, and remove calculations that are not useful" (paper quote,
# expressed in our own wording so the call OVERWRITES the previous block).
_RETHINK_REFINE_PROMPT = """You are an expert reasoning agent refining a pre-computed memory block.

Below is a conversation passage and your previous rethink memory block for it. Your job:
1. Double-check all facts, inferences, and calculations for correctness
2. Add any missing information, relationships, or potential questions
3. Remove anything unhelpful or redundant
4. Make the block more useful for answering future questions

IMPORTANT: Output the COMPLETE refined memory block (this will fully replace the previous one).

Passage:
{chunk}

Previous rethink memory block:
{previous_block}

Refined rethink memory block:"""


@register_memory_unit("C13")
class SleepTimeMemory(DenseRetrievalMemory):
    """Sleep-time Compute (chunked adaptation): per-chunk iterative rethink + FAISS retrieval.

    Faithful to the paper's rethink_memory semantics (overwrite, not append)
    applied per-chunk. At read time we dense-search over rethink blocks
    rather than concatenating them into a single oversized block.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict[str, Any]] = []
        self._indexed = False
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._num_chunks = 0
        self._num_rethink_calls = 0  # total LLM calls across all chunks
        self._llm_call_times: List[float] = []
        # Map from FAISS index position → source chunk text (kept so read()
        # can expose both the pre-computed block and the original passage).
        self._chunk_lookup: Dict[int, str] = {}
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._chunk_words = int(extra.get("chunk_words", _DEFAULT_CHUNK_WORDS))
        self._rethink_rounds = int(extra.get("rethink_rounds", _DEFAULT_RETHINK_ROUNDS))

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._indexed = False

    def _group_into_chunks(self) -> List[tuple]:
        """Group consecutive utterances into ~``chunk_words``-word chunks."""
        chunks: List[tuple] = []
        current_texts: List[str] = []
        current_dia_ids: List[str] = []
        current_session_ids: List[str] = []
        current_word_count = 0

        for i, text in enumerate(self._buffer):
            word_count = len(text.split())
            meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}

            if current_word_count + word_count > self._chunk_words and current_texts:
                chunk_text = "\n".join(current_texts)
                chunk_meta = {
                    "dia_ids": list(current_dia_ids),
                    "session_ids": list(current_session_ids),
                }
                chunks.append((chunk_text, chunk_meta))
                current_texts = []
                current_dia_ids = []
                current_session_ids = []
                current_word_count = 0

            current_texts.append(text)
            current_word_count += word_count
            dia_id = meta.get("dia_id", "")
            if dia_id:
                current_dia_ids.append(dia_id)
            session_id = meta.get("session_id", "")
            if session_id and session_id not in current_session_ids:
                current_session_ids.append(session_id)

        if current_texts:
            chunk_text = "\n".join(current_texts)
            chunk_meta = {
                "dia_ids": list(current_dia_ids),
                "session_ids": list(current_session_ids),
            }
            chunks.append((chunk_text, chunk_meta))

        return chunks

    def _rethink_chunks_batched(self, chunks: List[str]) -> List[str]:
        """Batched rethink for vLLM / any true-batch backend. Round 1 writes;
        rounds ≥ 2 are mgmt. Per-item latency attribution is token-weighted
        to avoid the degenerate mean=p50=p90 pattern that naive even-split
        produces when batched."""
        n = len(chunks)
        if n == 0:
            return []

        # Round 1: initial rethink
        prompts = [_RETHINK_INITIAL_PROMPT.format(chunk=c) for c in chunks]
        t0 = time.perf_counter()
        blocks = self.model.generate_batch(prompts, max_new_tokens=2000)
        round_elapsed = time.perf_counter() - t0
        # Attribute by output length so chunks that produced more text own
        # more of the wall clock — real distribution, not constant.
        out_lens = [max(len(self.model.encode(b or "")), 1) for b in blocks]
        total_out = sum(out_lens) or 1
        for i, (p, b) in enumerate(zip(prompts, blocks)):
            share = round_elapsed * (out_lens[i] / total_out)
            self._llm_call_times.append(share)
            self._track_llm_call(p, b or "")
            self._num_rethink_calls += 1

        # Rounds 2..rethink_rounds → mgmt tokens
        for _ in range(1, self._rethink_rounds):
            prompts = [
                _RETHINK_REFINE_PROMPT.format(chunk=c, previous_block=(b or "").strip())
                for c, b in zip(chunks, blocks)
            ]
            t0 = time.perf_counter()
            blocks = self.model.generate_batch(prompts, max_new_tokens=2000)
            round_elapsed = time.perf_counter() - t0
            out_lens = [max(len(self.model.encode(b or "")), 1) for b in blocks]
            total_out = sum(out_lens) or 1
            for i, (p, b) in enumerate(zip(prompts, blocks)):
                share = round_elapsed * (out_lens[i] / total_out)
                self._llm_call_times.append(share)
                self._track_mgmt_call(p, b or "")
                self._num_rethink_calls += 1

        return [(b or "").strip() for b in blocks]

    def _ensure_indexed(self):
        if self._indexed:
            return
        if not self._buffer:
            self._indexed = True
            return

        chunks = self._group_into_chunks()
        self._num_chunks = len(chunks)
        self._raw_token_count = sum(len(self.model.encode(c)) for c, _ in chunks)

        print(
            f"  C13: sleep-time rethink for {self._num_chunks} chunks "
            f"(~{self._chunk_words} words each, {self._rethink_rounds} rounds)...",
            flush=True,
        )

        chunk_texts = [c for c, _ in chunks]
        batch_t0 = time.perf_counter()
        rethink_blocks = self._rethink_chunks_batched(chunk_texts)
        batch_elapsed = time.perf_counter() - batch_t0

        # Token-weighted per-chunk write-latency share (avoids mean=p50=p90).
        out_lens = [max(len(self.model.encode(b or "")), 1) for b in rethink_blocks]
        total_out = sum(out_lens) or 1
        per_chunk_sec = [batch_elapsed * (out_lens[i] / total_out) for i in range(self._num_chunks)]

        for i, ((chunk_text, chunk_meta), rethink_block) in enumerate(
            zip(chunks, rethink_blocks)
        ):
            if (i + 1) % 10 == 0 or i + 1 == self._num_chunks:
                print(
                    f"    [{i+1}/{self._num_chunks}] chunks indexed "
                    f"({self._num_rethink_calls} rethink calls)",
                    flush=True,
                )

            if rethink_block:
                self._add_to_index(
                    rethink_block,
                    metadata={
                        "source": "sleep_time_rethink",
                        "chunk_index": i,
                        **chunk_meta,
                    },
                )
                self._chunk_lookup[len(self._entries) - 1] = chunk_text
                self._stored_token_count += len(self.model.encode(rethink_block))

            self._write_times.append(per_chunk_sec[i])

        self._indexed = True

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve most relevant pre-computed rethink blocks for the query."""
        self._ensure_indexed()

        if self._index is None or self._index.ntotal == 0:
            return []

        t0 = time.perf_counter()
        q_emb = self._encode([query])
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k)

        results: List[MemoryEntry] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self._entries[idx]
            chunk_text = self._chunk_lookup.get(int(idx), "")
            content = (
                f"Pre-computed reasoning:\n{entry.content}\n\n"
                f"Source passage:\n{chunk_text}"
            )
            results.append(
                MemoryEntry(
                    content=content,
                    metadata=entry.metadata,
                    score=float(score),
                    timestamp=entry.timestamp,
                )
            )

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(self.model.encode(r.content)) for r in results
        )
        return results

    def _on_reset(self):
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._chunk_lookup.clear()
        self._indexed = False
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._num_chunks = 0
        self._num_rethink_calls = 0
        self._llm_call_times.clear()

    # ── cache support ─────────────────────────────────────────
    REQUIRES_GEN_MODEL_IN_KEY = True   # rethink uses self.model.generate
    CACHE_KEY_EXTRA = ["chunk_words", "rethink_rounds", "max_new_tokens"]

    def save_state(self, dir_path: str) -> None:
        self._ensure_indexed()
        super().save_state(dir_path)
        import os, json
        with open(os.path.join(dir_path, "c13_extras.json"), "w") as f:
            json.dump({
                "buffer": self._buffer,
                "buffer_metadata": self._buffer_metadata,
                "indexed": self._indexed,
                "raw_token_count": self._raw_token_count,
                "stored_token_count": self._stored_token_count,
                "num_chunks": self._num_chunks,
                "num_rethink_calls": self._num_rethink_calls,
                "chunk_lookup": {str(k): v for k, v in self._chunk_lookup.items()},
                "llm_call_times": list(self._llm_call_times),
            }, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        unit = super().load_state(dir_path, config, model)
        import os, json
        with open(os.path.join(dir_path, "c13_extras.json")) as f:
            d = json.load(f)
        unit._buffer = d["buffer"]
        unit._buffer_metadata = d["buffer_metadata"]
        unit._indexed = d["indexed"]
        unit._raw_token_count = d["raw_token_count"]
        unit._stored_token_count = d["stored_token_count"]
        unit._num_chunks = d["num_chunks"]
        unit._num_rethink_calls = d["num_rethink_calls"]
        unit._chunk_lookup = {int(k): v for k, v in d["chunk_lookup"].items()}
        unit._llm_call_times = d["llm_call_times"]
        return unit

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_indexed()
        stats = super().get_stats()
        cr = (
            self._raw_token_count / self._stored_token_count
            if self._stored_token_count > 0
            else 1.0
        )
        stats["compression_ratio"] = round(cr, 2)
        stats["raw_token_count"] = self._raw_token_count
        stats["stored_token_count"] = self._stored_token_count
        stats["num_chunks"] = self._num_chunks
        stats["num_rethink_calls"] = self._num_rethink_calls
        stats["rethink_rounds"] = self._rethink_rounds
        stats["num_raw_utterances"] = len(self._buffer)
        stats["avg_llm_call_ms"] = (
            sum(self._llm_call_times) / len(self._llm_call_times) * 1000
            if self._llm_call_times
            else 0.0
        )
        return stats
