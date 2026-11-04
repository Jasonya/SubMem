"""M3 — Summarized Text Memory (ReadAgent-style).

Reference: ReadAgent (Lee et al., ICLR 2024, Google DeepMind)
- "ReadAgent: A Human-Inspired Reading Agent with Gist and Episodic Memory"

Key adaptations from the paper:
- ReadAgent groups text into semantic "pages" (~280-600 words) at paragraph boundaries
- Each page is compressed into a gist summary tagged with page reference
- FAISS indexes the GISTS for search, but read() returns the ORIGINAL page text
  (faithful to ReadAgent's "look up" mechanism: search gist → retrieve raw page)

Pipeline: group utterances into ~500-word pages → LLM gist per page → embed gist → FAISS
          read: LLM reads all gists in-context → decides which pages to look up → return original page text
          fallback: FAISS search if LLM selection fails
"""
import time
from typing import Any, Dict, List, Optional

from .base import register_memory_unit
from .dense_retrieval import DenseRetrievalMemory

_DEFAULT_PAGE_WORDS = 500  # ReadAgent: w_min=280, w_max=600

# Paper-faithful minimal gist prompt (ReadAgent Colab, Google DeepMind)
_GIST_PROMPT = """Please shorten the following passage.
Just give me a shortened version. DO NOT explain your reason.

Passage:
{chunk}"""

# Adapted from the ReadAgent page-lookup prompt (Lee et al., ICLR 2024)
#   https://read-agent.github.io/assets/read_agent_demo.ipynb
# Verbatim except for (a) "conversation" substituted for "article" since
# LoCoMo is dialog, not prose, and (b) the multiple-choice `{options}`
# slot dropped because LoCoMo uses free-form answers.
_PAGE_SELECT_PROMPT = """The following text is what you remember from reading a conversation and a question related to it.
You may read 1 to 5 page(s) of the conversation again to refresh your memory to prepare yourselve for the question.
Please respond with which page(s) you would like to read again.
For example, if your would like to only read Page 8, respond with "I want to look up Page [8] to ...";
if your would like to read Page 7 and 12, respond with "I want to look up Page [7, 12] to ...";
if your would like to read Page 2, 3, 7, 15 and 18, respond with "I want to look up Page [2, 3, 7, 15, 18] to ...".
DO NOT select more pages if you don't need to.
DO NOT answer the question yet.

Text:
{gist_list}

Question:
{query}

Take a deep breath and tell me: Which page(s) would you like to read again?"""


@register_memory_unit("M3")
class SummarizedTextMemory(DenseRetrievalMemory):
    """Summarized text with utterance-grouped pages + LLM gist + dense retrieval."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict[str, Any]] = []
        self._indexed = False
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._num_pages = 0
        self._llm_call_times: List[float] = []
        # ReadAgent page lookup: page_index → original page text + metadata
        self._page_lookup: Dict[int, Dict[str, Any]] = {}
        self._page_words = (
            config.extra.get("page_words", _DEFAULT_PAGE_WORDS)
            if hasattr(config, "extra") and config.extra
            else _DEFAULT_PAGE_WORDS
        )

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._indexed = False

    def _group_into_pages(self) -> List[tuple]:
        """Group consecutive utterances into ~500-word pages.

        Preserves utterance boundaries (ReadAgent-style pagination).
        Returns list of (page_text, page_metadata) where page_metadata
        contains aggregated dia_ids from all utterances in the page.
        """
        pages = []
        current_texts = []
        current_dia_ids = []
        current_session_ids = []
        current_word_count = 0

        current_extra: Dict[str, Any] = {}
        for i, text in enumerate(self._buffer):
            word_count = len(text.split())
            meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}

            # If adding this utterance would exceed page size, flush current page
            if current_texts and current_word_count + word_count > self._page_words:
                page_text = "\n".join(current_texts)
                page_meta = {
                    "dia_ids": list(current_dia_ids),
                    "session_ids": list(current_session_ids),
                    **current_extra,  # propagate upstream fields like success/goal
                }
                pages.append((page_text, page_meta))
                current_texts = []
                current_dia_ids = []
                current_session_ids = []
                current_word_count = 0
                current_extra = {}

            current_texts.append(text)
            current_word_count += word_count
            dia_id = meta.get("dia_id", "")
            if dia_id:
                current_dia_ids.append(dia_id)
            session_id = meta.get("session_id", "")
            if session_id and session_id not in current_session_ids:
                current_session_ids.append(session_id)
            # Keep upstream fields (success/goal/task_type/source/item_id etc.)
            # Last write wins inside a page — OK since each bank entry = one page.
            for k, v in meta.items():
                if k in ("dia_id", "session_id"):
                    continue
                current_extra[k] = v

        # Flush remaining
        if current_texts:
            page_text = "\n".join(current_texts)
            page_meta = {
                "dia_ids": list(current_dia_ids),
                "session_ids": list(current_session_ids),
                **current_extra,
            }
            pages.append((page_text, page_meta))

        return pages

    def _summarize_chunk(self, chunk: str) -> str:
        """Call the LLM to produce a gist summary of a page.

        ReadAgent's reference notebook doesn't cap gist length (PaLM 2 stops
        when it decides to). We budget enough room for a few-sentence gist
        on a ~500-word page (~300 tokens) without being so open-ended that
        a pathological generation explodes memory. 512 matches the ReadAgent
        demo's default sampling budget.
        """
        prompt = _GIST_PROMPT.format(chunk=chunk)
        t0 = time.perf_counter()
        gist = self.model.generate(prompt, max_new_tokens=512)
        self._llm_call_times.append(time.perf_counter() - t0)
        self._track_llm_call(prompt, gist)
        return gist.strip()

    def _ensure_indexed(self):
        if self._indexed:
            return
        if not self._buffer:
            self._indexed = True
            return

        t0 = time.perf_counter()
        pages = self._group_into_pages()
        self._num_pages = len(pages)
        self._raw_token_count = sum(len(self.model.encode(p)) for p, _ in pages)

        print(f"  M3: summarizing {self._num_pages} pages (~{self._page_words} words each)...", flush=True)
        for i, (page_text, page_meta) in enumerate(pages):
            gist = self._summarize_chunk(page_text)
            if (i + 1) % 10 == 0 or i + 1 == self._num_pages:
                print(f"    [{i+1}/{self._num_pages}] gists done", flush=True)
            if gist:
                tagged_gist = f"<Page {i+1}> {gist}"
                self._add_to_index(tagged_gist, metadata={
                    "source": "gist_summary",
                    "page_index": i,
                    "page_ref": i + 1,
                    **page_meta,
                })
                self._stored_token_count += len(self.model.encode(tagged_gist))
                # Store original page for ReadAgent-style lookup
                self._page_lookup[i] = {
                    "text": page_text,
                    "metadata": page_meta,
                }

        # Use per-page LLM call times as write latencies
        if self._llm_call_times:
            self._write_times.extend(self._llm_call_times)
        else:
            # Fallback: amortize total time
            total_time = time.perf_counter() - t0
            n = self._num_pages or 1
            self._write_times.extend([total_time / n] * n)

        self._indexed = True

    def read(self, query: str, top_k: int = 5):
        """ReadAgent-style read: gists + selected raw pages.

        Faithful to ReadAgent's pipeline:
          1. LLM reads ALL gists in-context
          2. LLM decides which pages (1-6) to look up
          3. Build context = gists for non-selected pages + raw text for selected pages
             (ReadAgent keeps unselected gists in context for cross-page reasoning)
          4. Fallback: FAISS search if LLM selection fails
        """
        self._ensure_indexed()
        from .base import MemoryEntry
        import re as _re

        if not self._entries:
            return []

        t0 = time.perf_counter()

        # Step 1: Build gist list for LLM. Stock ReadAgent dumps ALL gists,
        # which on a 1300-turn LME(S*) sample is ~625 gists × ~50 words =
        # ~30-50K tokens — fine for an 8B model with 40K window, but blows
        # up Gemma-26B / Qwen3-32B. We dense-rank gists by query similarity
        # and keep only the top-N when total gist count exceeds the cap.
        # Default keeps the paper-faithful path for shorter conversations.
        extra = getattr(self.config, "extra", {}) or {}
        max_gists = int(extra.get("max_gists_in_prompt", 200))
        if len(self._entries) > max_gists:
            try:
                from .embeddings import create_encoder
                if not hasattr(self, "_gist_encoder"):
                    self._gist_encoder = create_encoder(self.config)
                    self._gist_embs = self._gist_encoder.encode(
                        [e.content for e in self._entries]
                    )
                q_emb = self._gist_encoder.encode([query])
                # cosine similarity (encoder already L2-normalised)
                import numpy as _np
                sims = (self._gist_embs @ q_emb.T).flatten()
                top_idx = sorted(_np.argsort(sims)[-max_gists:].tolist())
                shortlisted = [self._entries[i] for i in top_idx]
            except Exception:
                shortlisted = self._entries[:max_gists]
        else:
            shortlisted = self._entries
        gist_lines = []
        for entry in shortlisted:
            page_ref = entry.metadata.get("page_ref", "?")
            gist_lines.append(f"<Page {page_ref}> {entry.content}")
        gist_list = "\n".join(gist_lines)

        # Step 2: LLM decides which pages to look up.
        # Note: 32 tokens was insufficient when the gist list grows beyond a few
        # dozen pages — the LLM truncates mid-rationale before emitting any
        # page number, leaving an empty `lookup_set` and forcing the fallback
        # to dump all gists. 128 tokens lets the LLM finish a short rationale
        # AND emit a page list (typical valid response: 25-60 tokens).
        prompt = _PAGE_SELECT_PROMPT.format(gist_list=gist_list, query=query)
        response = self.model.generate(prompt, max_new_tokens=128)

        # Track read-phase LLM consumption (E8/E9)
        self._token_counts.setdefault("read_llm_input", 0)
        self._token_counts.setdefault("read_llm_output", 0)
        self._token_counts.setdefault("read_llm_calls", 0)
        self._token_counts["read_llm_input"] += len(self.model.encode(prompt))
        self._token_counts["read_llm_output"] += len(self.model.encode(response))
        self._token_counts["read_llm_calls"] += 1

        # Parse page numbers from response (e.g., "3, 7, 12" or "Page 3 and Page 7")
        page_nums_raw = [int(n) for n in _re.findall(r'\d+', response)]
        # Cap the number of raw pages that get expanded. ReadAgent's prompt
        # says "read 1 to 5 pages", but the original code let 6 through and
        # relied on the LLM to self-limit. Gemma-4 doesn't self-limit
        # reliably (frequently lists 6-33 pages on LRU-style broad
        # questions), and expanding 6 raw pages at page_words=2000 blows
        # the sdpa attention memory even on H200. Capping at 3 keeps us
        # within the paper's stated range and provides safe headroom.
        _MAX_EXPAND_PAGES = 3
        lookup_order: List[int] = []
        lookup_set = set()
        for pnum in page_nums_raw:
            if (pnum - 1) in self._page_lookup and pnum not in lookup_set:
                lookup_set.add(pnum)
                lookup_order.append(pnum)
            if len(lookup_set) >= _MAX_EXPAND_PAGES:
                break

        # Runaway-list guard. When the LLM pads all the way to the 128-token
        # generation budget while listing pages (Gemma-4 does this on ~10%
        # of LRU-style broad queries — see M3.page_select instrumentation
        # log), even the top-3 pick above can still blow up if the query
        # is extremely broad. Collapse to the model's first pick.
        resp_tokens_len = len(self.model.encode(response))
        _degraded = resp_tokens_len >= 120 and len(lookup_set) >= 2
        if _degraded and lookup_order:
            lookup_set = {lookup_order[0]}

        # Instrumentation (env var SUBMEM_M3_LOG_PAGE_SELECT=1): log page-select
        # outcome for diagnosing model-specific failure modes (e.g. Gemma
        # running out of 128 tokens before emitting a page number, or echoing
        # example numbers from the prompt that fall outside the actual range).
        import os as _os
        if _os.environ.get("SUBMEM_M3_LOG_PAGE_SELECT"):
            resp_head = response.replace("\n", " ")[:200]
            print(f"  M3.page_select: pages={self._num_pages} resp_tokens={resp_tokens_len} "
                  f"raw_nums={page_nums_raw[:10]} valid={sorted(lookup_set)} "
                  f"degraded={_degraded} resp='{resp_head}'", flush=True)

        # Step 3: Build context = gists + expanded pages (ReadAgent-style)
        # Non-selected pages keep their gist; selected pages get raw text.
        # We iterate `shortlisted` (the dense-ranked top-N used for
        # page-select), not `self._entries`, so the answer-generation
        # prompt shrinks alongside `max_gists_in_prompt` when the raw
        # gist volume would otherwise OOM the base model — necessary for
        # ICL corpora (TTL banking77 etc.) where gists don't compress
        # well and each page's gist stays near the 512-token cap. The
        # `lookup_set` pages are unioned in so we always expand pages the
        # LLM asked for, even if they fell outside the shortlist.
        context_parts = []
        retrieved_dia_ids = []  # only from selected (expanded) pages
        _seen_prefs = set()
        _lookup_entries = [e for e in self._entries
                           if e.metadata.get("page_ref", 0) in lookup_set]
        for entry in list(shortlisted) + _lookup_entries:
            page_ref = entry.metadata.get("page_ref", 0)
            if page_ref in _seen_prefs:
                continue
            _seen_prefs.add(page_ref)
            page_idx = page_ref - 1

            if page_ref in lookup_set and page_idx in self._page_lookup:
                # Selected page: use original raw text
                page = self._page_lookup[page_idx]
                context_parts.append(f"<Page {page_ref}>\n{page['text']}")
                retrieved_dia_ids.extend(page["metadata"].get("dia_ids", []))
            else:
                # Non-selected page: keep gist (dia_ids NOT counted as retrieved)
                context_parts.append(f"<Page {page_ref} (gist)> {entry.content}")

        combined_context = "\n".join(context_parts)

        # No FAISS fallback: ReadAgent does not have one. If the LLM returned
        # no page numbers, the agent answers from gists alone — we replicate
        # that behaviour, matching the paper. Previously we inserted a FAISS
        # top-3 backup, which silently turned M3 into a Dense-retrieval
        # variant whenever page selection failed, distorting both P6/P7 and
        # E-metrics.

        # Aggregate turn-level metadata across all expanded pages for
        # retrieval-quality metrics (P5/P6/P7). Setting has_answer=True
        # whenever any expanded page contained gold evidence is the
        # closest analog given M3's "merge multiple pages into one entry"
        # design.
        any_has_answer = False
        kept_session_ids = set()
        for page_ref in lookup_set:
            page_idx = page_ref - 1
            if page_idx in self._page_lookup:
                page_meta = self._page_lookup[page_idx].get("metadata", {})
                if page_meta.get("has_answer"):
                    any_has_answer = True
                sid = page_meta.get("session_id")
                if sid:
                    kept_session_ids.add(sid)

        results = [MemoryEntry(
            content=combined_context,
            metadata={"source": "readagent_gist_plus_pages",
                      "lookup_pages": sorted(lookup_set),
                      "dia_ids": retrieved_dia_ids,
                      "has_answer": any_has_answer,
                      "session_id": ";".join(sorted(kept_session_ids)) or None,
                      "role": None},
            score=1.0,
        )]

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(self.model.encode(r.content)) for r in results
        )
        return results

    def _on_reset(self):
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._page_lookup.clear()
        self._indexed = False
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._num_pages = 0
        self._llm_call_times.clear()

    # ── cache support ─────────────────────────────────────────
    CACHE_KEY_EXTRA = ["page_words", "max_new_tokens"]

    def save_state(self, dir_path: str) -> None:
        # Force lazy build before save so cached state has gist summaries.
        self._ensure_indexed()
        super().save_state(dir_path)
        import os, json
        with open(os.path.join(dir_path, "c4_extras.json"), "w") as f:
            json.dump({
                "buffer": self._buffer,
                "buffer_metadata": self._buffer_metadata,
                "indexed": self._indexed,
                "raw_token_count": self._raw_token_count,
                "stored_token_count": self._stored_token_count,
                "num_pages": self._num_pages,
                "page_lookup": {str(k): v for k, v in self._page_lookup.items()},
                "llm_call_times": list(self._llm_call_times),
            }, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        unit = super().load_state(dir_path, config, model)
        import os, json
        with open(os.path.join(dir_path, "c4_extras.json")) as f:
            d = json.load(f)
        unit._buffer = d["buffer"]
        unit._buffer_metadata = d["buffer_metadata"]
        unit._indexed = d["indexed"]
        unit._raw_token_count = d["raw_token_count"]
        unit._stored_token_count = d["stored_token_count"]
        unit._num_pages = d["num_pages"]
        unit._page_lookup = {int(k): v for k, v in d["page_lookup"].items()}
        unit._llm_call_times = d["llm_call_times"]
        return unit

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_indexed()
        stats = super().get_stats()
        cr = (self._raw_token_count / self._stored_token_count
              if self._stored_token_count > 0 else 1.0)
        stats["compression_ratio"] = round(cr, 2)
        stats["raw_token_count"] = self._raw_token_count
        stats["stored_token_count"] = self._stored_token_count
        stats["num_pages"] = self._num_pages
        stats["num_raw_utterances"] = len(self._buffer)
        stats["avg_llm_call_ms"] = (
            sum(self._llm_call_times) / len(self._llm_call_times) * 1000
            if self._llm_call_times else 0.0
        )
        return stats
