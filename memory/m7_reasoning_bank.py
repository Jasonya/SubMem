"""M7 — ReasoningBank Memory (agentic only).

Reference: ReasoningBank (Ouyang et al., ICLR 2026, Google Research + UIUC)
  "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory"
  https://arxiv.org/abs/2509.25140

Re-implementation. Prompts (judge + distill) vendored verbatim from the
thakshak/ReasoningBank OSS release (https://github.com/thakshak/ReasoningBank,
MIT). Note: thakshak's repo is a third-party sketch, not the paper authors'
official code — it has no benchmark drivers, uses ChromaDB (not FAISS), and
only embeds the query (not distilled items). Our memory unit inherits
DenseRetrievalMemory for FAISS + embeds "title. description. content" so items are
actually retrievable. Paper-level fidelity is thus limited.

Pipeline (per agentic task attempt):
  1. Judge: LLM binary-classifies (query, trajectory) as Success / Failure.
  2. Distill: LLM extracts up to a few structured memory items using one
     of two prompts depending on success/failure.
  3. Store: embed each item's (title + description + content), push into FAISS.
  4. Retrieve: at test time, dense search returns top-k strategy items
     (default k=1, per the paper).

Scope: agentic benchmarks only. Dialog benchmarks (LoCoMo, LongMemEval)
do not have a (query, trajectory) pair at write time, so calling the
dialog ``write()`` path on M7 raises NotImplementedError. Agentic
benchmarks should call ``write_experience(query, trajectory, ...)``.
"""
import json
import time
from typing import Any, Dict, List, Optional

from .base import MemoryEntry, register_memory_unit
from .dense_retrieval import DenseRetrievalMemory


# ── Prompts (vendored verbatim from thakshak/ReasoningBank distill.py) ──
# Prompts verbatim from ReasoningBank (Ouyang et al., ICLR 2026),
# arXiv:2509.25140, Appendix A.1 Figures 9–10 and A.3 Figure 11.
# Upstream release: https://github.com/google-research/reasoning-bank
#
# The original paper targets web-navigation agents; we replace "web
# navigation" with "agent" to stay domain-agnostic (this substrate runs
# on SWE-bench / AlfWorld via write_experience, never on dialog). All
# guidance and the Markdown Title/Description/Content schema are kept
# verbatim.
_JUDGE_PROMPT = """System Instruction
You are an expert in evaluating the performance of an agent. The agent is designed to help a human user complete a task. Given the user's intent, the agent's action history, the final state of the environment, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information. The agent's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and responds with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions.
2. Site/environment navigation: The user wants to reach a specific state. Carefully examine the agent's action history and the final state of the environment to determine whether the agent successfully completes the task. No need to consider the agent's response.
3. Content modification: The user wants to modify content or configuration. Carefully examine the agent's action history and the final state of the environment to determine whether the agent successfully completes the task. No need to consider the agent's response.

*IMPORTANT*
Format your response into two lines as shown below:
Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"

Input Prompt
User Intent: {query}
Trajectory: {trajectory}
"""

_DISTILL_SUCCESS_PROMPT = """System Instruction
You are an expert. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>

Input Prompt
Query: {query}
Trajectory: {trajectory}
"""

_DISTILL_FAILURE_PROMPT = """System Instruction
You are an expert. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>

Input Prompt
Query: {query}
Trajectory: {trajectory}
"""

# MaTTS (memory-aware test-time scaling) — Appendix A.3 Figure 11.
# Given k parallel rollouts for the same query, a single self-contrast
# call emits up to 5 consolidated memory items without invoking the
# judge (contrast replaces per-trajectory labelling).
_MATTS_CONTRAST_PROMPT = """System Instruction
You are an expert. You will be given a user query and {n_rollouts} independent trajectories that the agent produced for the same query. Some of the trajectories may have succeeded and others may have failed.

## Guidelines
Compare the trajectories. Identify the contrasting behaviours between trajectories that succeeded and trajectories that failed, and distill up to 5 consolidated memory items that capture the generalizable insights.

## Important notes
- First, think through why some trajectories succeeded and why others failed.
- You can extract at most 5 memory items across all trajectories.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents; focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned>

Input Prompt
Query: {query}
Trajectories:
{trajectories_block}
"""


def _parse_markdown_items(text: str) -> List[Dict[str, str]]:
    """Parse the paper's Markdown schema into {title, description, content}.

    Paper output format (Appendix A.1, Figure 9):

        # Memory Item 1
        ## Title <title>
        ## Description <one-sentence summary>
        ## Content <1-3 sentences>
        # Memory Item 2
        ...

    We split on ``# Memory Item`` headers, then inside each block pick up
    Title / Description / Content sub-sections.
    """
    import re as _re
    items: List[Dict[str, str]] = []
    # Split on "# Memory Item <n>" headers
    blocks = _re.split(r"^\s*#\s+Memory\s+Item\b.*$", text, flags=_re.M)
    for block in blocks:
        if not block.strip():
            continue
        title = desc = content = ""
        for sect, val in _re.findall(
            r"^##\s+(Title|Description|Content)\s+(.*?)(?=^##\s+|\Z)",
            block, flags=_re.M | _re.S,
        ):
            val = val.strip()
            if sect.lower() == "title":
                title = val
            elif sect.lower() == "description":
                desc = val
            elif sect.lower() == "content":
                content = val
        if title or desc or content:
            items.append({"title": title, "description": desc, "content": content})
    return items


def _parse_json_items(text: str) -> List[Dict[str, str]]:
    """Parse extracted memory items. First try the paper's Markdown schema;
    fall back to JSON list for backwards compat with older prompts."""
    md = _parse_markdown_items(text)
    if md:
        return md
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [d for d in parsed if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass

    lo, hi = text.find("["), text.rfind("]")
    if lo >= 0 and hi > lo:
        try:
            parsed = json.loads(text[lo:hi + 1])
            if isinstance(parsed, list):
                return [d for d in parsed if isinstance(d, dict)]
        except json.JSONDecodeError:
            return []
    return []


@register_memory_unit("M7")
class ReasoningBankMemory(DenseRetrievalMemory):
    """Strategy-level distillation of agent trajectories.

    Input: (query, trajectory) pairs via ``write_experience``.
    Storage: FAISS index over embeddings of ``title + description + content``.
    Retrieval: dense search returns top-k memory items.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        # Pending experiences: flushed lazily on first read()
        self._pending: List[Dict[str, Any]] = []
        self._indexed = False
        self._num_items = 0
        self._item_lookup: Dict[int, Dict[str, str]] = {}

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────
    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Dialog-style write is not supported by ReasoningBank.

        ReasoningBank's unit of memory is a (query, trajectory) experience,
        not a single dialog turn. Call ``write_experience`` instead.
        """
        raise NotImplementedError(
            "M7 ReasoningBank requires (query, trajectory) pairs. "
            "Dialog benchmarks are not supported — use an agentic benchmark "
            "driver that calls memory_unit.write_experience(query, trajectory)."
        )

    def write_experience(
        self,
        query: str,
        trajectory: str,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Buffer a (query, trajectory) experience; flushed at first read()."""
        self._pending.append({
            "query": query,
            "trajectory": trajectory,
            "success": success,
            "metadata": dict(metadata or {}),
        })
        self._indexed = False

    def write_experience_matts(
        self,
        query: str,
        trajectories: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """MaTTS (memory-aware test-time scaling) — Paper §3.3 + App. A.3.

        Given ``k`` parallel rollouts for the same query, run a single
        self-contrast call that emits up to 5 consolidated memory items
        without invoking the per-trajectory judge (contrast replaces
        success labelling). This is the paper's default "Parallel MaTTS"
        variant with default k=5.

        For benchmarks that only produce a single rollout, call the
        per-trajectory ``write_experience`` instead.
        """
        if not trajectories:
            return
        self._pending.append({
            "matts": True,
            "query": query,
            "trajectories": list(trajectories),
            "metadata": dict(metadata or {}),
        })
        self._indexed = False

    def read(self, query: str, top_k: int = 1) -> List[MemoryEntry]:
        """Retrieve top-k reasoning memory items.

        Default ``top_k=1`` follows the paper's default.
        """
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
            item = self._item_lookup.get(int(idx), {})
            entry = self._entries[idx]
            content = (
                f"# Memory Item\n"
                f"## Title {item.get('title', '')}\n"
                f"## Description {item.get('description', '')}\n"
                f"## Content {item.get('content', '')}"
            )
            results.append(MemoryEntry(
                content=content,
                metadata={
                    "source": "reasoning_bank",
                    "kind": item.get("kind", ""),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    **entry.metadata,
                },
                score=float(score),
                timestamp=entry.timestamp,
            ))

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            len(self.model.encode(r.content)) for r in results
        )
        return results

    # ────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────
    def _judge(self, query: str, trajectory: str) -> bool:
        """Judge prompt — paper uses Thoughts/Status two-line format."""
        prompt = _JUDGE_PROMPT.format(query=query, trajectory=trajectory)
        # Paper uses deterministic temperature 0 for the judge
        resp = self.model.generate(prompt, max_new_tokens=256)
        self._track_llm_call(prompt, resp)
        # Parse `Status: "success"` / `Status: "failure"` line; fall back
        # to substring match for non-compliant outputs.
        import re as _re
        m = _re.search(r"Status\s*:\s*\"?(success|failure)\"?", resp, _re.I)
        if m:
            return m.group(1).lower() == "success"
        return "success" in resp.lower() and "failure" not in resp.lower()

    def _distill(self, query: str, trajectory: str, success: bool) -> List[Dict[str, str]]:
        tmpl = _DISTILL_SUCCESS_PROMPT if success else _DISTILL_FAILURE_PROMPT
        prompt = tmpl.format(query=query, trajectory=trajectory)
        # Paper caps at 3 items per trajectory (Appendix A.2), so 768 new
        # tokens is plenty for three Title/Description/Content blocks.
        resp = self.model.generate(prompt, max_new_tokens=768)
        self._track_llm_call(prompt, resp)
        return _parse_json_items(resp)

    def _matts_contrast(self, query: str,
                        trajectories: List[str]) -> List[Dict[str, str]]:
        """MaTTS self-contrast distillation — no judge, up to 5 items."""
        traj_block = "\n\n".join(
            f"Trajectory {i+1}:\n{t}" for i, t in enumerate(trajectories)
        )
        prompt = _MATTS_CONTRAST_PROMPT.format(
            query=query,
            n_rollouts=len(trajectories),
            trajectories_block=traj_block,
        )
        # Up to 5 items; allow more output budget than single-traj distill.
        resp = self.model.generate(prompt, max_new_tokens=1280)
        self._track_llm_call(prompt, resp)
        return _parse_json_items(resp)

    def _ensure_indexed(self):
        if self._indexed:
            return
        if not self._pending:
            self._indexed = True
            return

        print(f"  M7: ReasoningBank judge+distill for {len(self._pending)} experiences...",
              flush=True)

        for exp in self._pending:
            query = exp["query"]
            meta = exp["metadata"]

            t0 = time.perf_counter()
            if exp.get("matts"):
                # Parallel MaTTS: self-contrast, no judge.
                items = self._matts_contrast(query, exp["trajectories"])
                kind = "matts"
            else:
                trajectory = exp["trajectory"]
                success = exp.get("success")
                if success is None:
                    success = self._judge(query, trajectory)
                items = self._distill(query, trajectory, success)
                kind = "success" if success else "failure"
            self._write_times.append(time.perf_counter() - t0)

            for item in items:
                title = str(item.get("title", "")).strip()
                desc = str(item.get("description", "")).strip()
                content = str(item.get("content", "")).strip()
                # Paper indexes the full {title, description, content} triple
                # so the dense retriever can match on the concrete reasoning
                # step, not just the short header. Prior behaviour concat'd
                # only "title. description", which is too lossy for long
                # trajectories.
                index_parts = [p for p in (title, desc, content) if p]
                index_text = ". ".join(index_parts)
                if not index_text.strip():
                    continue

                item_idx = self._num_items
                self._add_to_index(index_text, metadata={
                    "source": "reasoning_bank",
                    "kind": kind,
                    "query": query,
                    **meta,
                })
                self._item_lookup[item_idx] = {
                    "title": title,
                    "description": desc,
                    "content": content,
                    "kind": kind,
                }
                self._num_items += 1

        self._pending.clear()
        self._indexed = True

    def _on_reset(self):
        self._pending.clear()
        self._item_lookup.clear()
        self._indexed = False
        self._num_items = 0
