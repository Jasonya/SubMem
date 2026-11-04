"""C15 — MemSkill Memory.

Reference: MemSkill (Zhang et al., arXiv 2602.02474, 2026)
  "MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents"
  https://arxiv.org/abs/2602.02474
  https://github.com/ViktorAxelsen/MemSkill  (Apache-2.0)

Faithful re-implementation, adapted from the official ViktorAxelsen/MemSkill
release. MemSkill treats *memory operations* (INSERT / UPDATE / DELETE / NOOP)
as learnable, evolvable "memory skills" — structured routines for how to
remember, not the remembered content itself.

Pipeline (per text chunk / per trajectory span):
  1. Controller: select top-K skills for the current chunk (given retrieved
     memories as context).
  2. Executor: LLM applies the selected skills to the chunk+retrieved memories
     and emits a list of INSERT / UPDATE / DELETE action blocks.
  3. Memory bank: apply the actions (UPDATE → DELETE in reverse idx order →
     INSERT), re-embed, FAISS-index.
  4. Designer (periodic): cluster failure cases, two-stage prompt (analysis
     → refinement) proposes refine_existing / add_new skill changes, applied
     to the skill bank. Evolves skills over time.
  5. Retrieve: dense search over memory bank at read time.

Scope: supports both dialog (LoCoMo / LongMemEval via ``write``) and agentic
(ALFWorld / HotpotQA via ``write_experience``) benchmarks. On agentic calls,
the trajectory is chunked and processed span-by-span, matching the paper.

Controller design choice: MemSkill's original controller is a PPO actor-critic
trained with task-level reward. For benchmark comparability (no other unit
requires an RL training step), we use an LLM-zero-shot controller: the LLM
picks top-K relevant skills from the skill bank. The skill evolution path
(Designer) is fully preserved.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import MemoryEntry, register_memory_unit
from .dense_retrieval import DenseRetrievalMemory


# ── Controller LLM client (OpenAI gpt-4o-mini) ───────────────────────────
class _ControllerLLM:
    """Dedicated small LLM for zero-shot skill selection.

    Mirrors the _OpenAILLMAdapter pattern used by c16 (A-Mem): skill selection
    is a cheap classification call, so route it through gpt-4o-mini regardless
    of SubMem's answer model.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", None),
        )

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        # Token-consumption hook: the outer MemSkill memory unit (c15) sets
        # ``_submem_owner`` on this client and exposes a ``_token_counts``
        # dict. We bucket by prompt heuristic:
        #   - controller selection prompt → "read" (per-query pick)
        #   - executor / designer prompt   → "write" (skill creation/update)
        #   - rebalancing / dedup prompt   → "mgmt"
        owner = getattr(self, "_submem_owner", None)
        if owner is not None:
            try:
                tc = owner._token_counts
                in_tok = (resp.usage.prompt_tokens
                          if getattr(resp, "usage", None) else len(prompt) // 4)
                out_tok = (resp.usage.completion_tokens
                           if getattr(resp, "usage", None) else 0)
                head = (prompt or "")[:200].lower()
                if "memory management controller" in head or "select" in head[:60]:
                    bucket = "read"   # per-query controller selection
                elif "rebalance" in head or "deduplicate" in head or "merge" in head[:60]:
                    bucket = "mgmt"
                else:
                    bucket = "write"  # executor / designer / new-skill prompts
                tc[f"{bucket}_llm_input"] = tc.get(f"{bucket}_llm_input", 0) + in_tok
                tc[f"{bucket}_llm_output"] = tc.get(f"{bucket}_llm_output", 0) + out_tok
                tc[f"{bucket}_llm_calls"] = tc.get(f"{bucket}_llm_calls", 0) + 1
            except Exception:
                pass
        return resp.choices[0].message.content or ""


# ── Seed skills (vendored from MemSkill/prompts/operation_templates.py) ──
_SEED_SKILLS: Dict[str, Dict[str, str]] = {
    "insert": {
        "name": "insert",
        "update_type": "insert",
        "description": (
            "Memory management skill for capturing new, durable facts from the current text chunk "
            "that are not already in memory."
        ),
        "instruction_template": (
            "Skill: Insert New Memory\n"
            "Purpose: Capture new, durable facts from the current text chunk that are missing in memory.\n"
            "When to use:\n"
            "- The text chunk introduces new facts, events, plans, or context worth storing.\n"
            "- The information is stable and likely useful later.\n"
            "How to apply:\n"
            "- Compare against retrieved memories to avoid duplicates.\n"
            "- Split distinct facts into separate items.\n"
            "- Keep each item concise and specific.\n"
            "Constraints:\n"
            "- Skip trivial, fleeting, or speculative content.\n"
            "- Do not update or delete existing memories.\n"
            "Action type: INSERT only.\n"
        ),
    },
    "update": {
        "name": "update",
        "update_type": "update",
        "description": (
            "Memory management skill for revising an existing memory item when the text chunk "
            "provides corrections or new details."
        ),
        "instruction_template": (
            "Skill: Update Existing Memory\n"
            "Purpose: Revise a retrieved memory with new or corrected information from the text chunk.\n"
            "When to use:\n"
            "- The text chunk clarifies, corrects, or extends a retrieved memory.\n"
            "How to apply:\n"
            "- Select the best matching memory item.\n"
            "- Merge new details into a single updated item.\n"
            "- Preserve accurate details that still hold.\n"
            "Constraints:\n"
            "- Do not create new memories.\n"
            "- Do not delete items.\n"
            "Action type: UPDATE only.\n"
        ),
    },
    "delete": {
        "name": "delete",
        "update_type": "delete",
        "description": (
            "Memory management skill for removing memory items that are incorrect, outdated, "
            "or superseded."
        ),
        "instruction_template": (
            "Skill: Delete Invalid Memory\n"
            "Purpose: Remove a retrieved memory that is wrong, outdated, or superseded by the text chunk.\n"
            "When to use:\n"
            "- The text chunk clearly contradicts a memory.\n"
            "- A plan or fact is explicitly canceled or replaced.\n"
            "How to apply:\n"
            "- Only delete when evidence is explicit.\n"
            "Constraints:\n"
            "- If uncertain, prefer no action over deletion.\n"
            "Action type: DELETE only.\n"
        ),
    },
}


# ── Controller prompt (LLM zero-shot top-K skill selection) ──────────────
_CONTROLLER_PROMPT = """You are a memory management controller. Pick the {top_k} most relevant memory management skills to apply to the current text chunk.

## Available Skills
{skill_list}

## Current Text Chunk
{chunk}

## Retrieved Memories
{retrieved}

Return ONLY a JSON list of {top_k} skill names (no prose), e.g. ["insert", "update"]. Pick skills that are most likely to be useful for this chunk given the retrieved memories. If fewer than {top_k} skills apply, still return exactly {top_k} names (repeat if needed).
"""


# ── Executor prompt (vendored from MemSkill/src/executor.py) ────────────
def _build_executor_prompt(skills: List[Dict[str, str]], chunk: str,
                           retrieved: List[str]) -> str:
    if retrieved:
        mem_text = "\n".join(f"{i}. {m}" for i, m in enumerate(retrieved))
    else:
        mem_text = "(No existing memories retrieved)"

    blocks = []
    seen = set()
    for op in skills:
        name = op.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        lines = [f"[Skill {len(seen)}] {name}"]
        if op.get("description"):
            lines.append(f"Description: {op['description']}")
        ut = op.get("update_type", "").upper()
        if ut:
            lines.append(f"Allowed action: {ut}")
        if op.get("instruction_template"):
            lines.append("Instructions:")
            lines.append(op["instruction_template"])
        blocks.append("\n".join(lines))
    skills_text = "\n\n".join(blocks) if blocks else "(No skills provided)"

    return (
        "You are a memory management executor. Apply the selected skills to the input text\n"
        "chunk and retrieved memories, then output memory actions.\n\n"
        f"Input Text Chunk:\n{chunk}\n\n"
        f"Retrieved Memories (0-based index):\n{mem_text}\n\n"
        f"Selected Skills:\n{skills_text}\n\n"
        "Guidelines:\n"
        "- Apply any skill as needed; a skill may be used multiple times.\n"
        "- Read the input text chunk carefully line by line and apply any skill as needed.\n"
        "- Only use action types supported by the selected skills.\n"
        "- MEMORY_INDEX is 0-based and must reference the retrieved memories list.\n"
        "- Output only action blocks in the format below.\n"
        "- Do not include explanations or REASONING lines.\n"
        "Output format (repeat as needed). Use ONE block per action and separate blocks with"
        " a blank line:\n\n"
        "INSERT block:\n"
        "ACTION: INSERT\n"
        "MEMORY_ITEM: <concise but complete summary with essential details>\n\n"
        "UPDATE block:\n"
        "ACTION: UPDATE\n"
        "MEMORY_INDEX: <0-based index>\n"
        "UPDATED_MEMORY: <concise but complete merged summary with essential updates>\n\n"
        "DELETE block:\n"
        "ACTION: DELETE\n"
        "MEMORY_INDEX: <0-based index>\n\n"
    )


# ── Designer prompts (condensed from MemSkill/prompts/designer_prompts.py) ──
_DESIGNER_ANALYSIS_PROMPT = """You are an expert analyst for a memory-augmented QA system. Analyze the failure cases below to identify why the system failed and how the memory management skills should change.

## Current Memory Management Skills
{skills_desc}

## Failure Cases ({n} cases)
{cases}

Propose up to {max_changes} skill changes (add a new skill or refine an existing one). Output ONLY JSON:
{{
  "failure_patterns": [{{"pattern_name": "...", "affected_cases": [...], "root_cause": "storage_failure|retrieval_failure|memory_quality_failure", "explanation": "..."}}],
  "recommendations": [{{"action": "add_new_operation|refine_existing_operation|no_change", "target_operation": "<name or null>", "rationale": "..."}}],
  "summary": "..."
}}
"""

_DESIGNER_REFINEMENT_PROMPT = """Based on the analysis below, propose concrete skill changes for the memory management skill bank.

## Analysis
{analysis}

## Full Skill Bank
{skills_full}

Output ONLY JSON with up to {max_changes} changes:
{{
  "changes": [
    {{"action": "add_new",        "new_operation":     {{"name": "...", "description": "...", "instruction_template": "...", "update_type": "insert|update", "reasoning": "..."}}}},
    {{"action": "refine_existing","refined_operation": {{"name": "<existing>", "changes": {{"description": "...", "instruction_template": "..."}}, "reasoning": "..."}}}}
  ],
  "summary": "..."
}}
Instruction templates must NOT contain {{session_text}} or {{retrieved_memories}} placeholders.
"""


# ── Parsing helpers (adapted from MemSkill/src/executor.py) ──────────────
_ACTION_RE = re.compile(
    r'(?<!\w)ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b',
    re.IGNORECASE,
)


@dataclass
class _Action:
    kind: str  # INSERT | UPDATE | DELETE | NOOP
    content: str = ""
    index: int = -1


def _parse_executor_output(text: str, num_retrieved: int) -> List[_Action]:
    """Parse executor LLM output into a list of actions. Best-effort, tolerant."""
    text = text.replace("\r\n", "\n").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if "\n" in inner:
                first, rest = inner.split("\n", 1)
                if first.strip().lower() in ("json", "text"):
                    inner = rest
            text = inner.strip()

    matches = list(_ACTION_RE.finditer(text))
    actions: List[_Action] = []
    if not matches:
        return actions

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        kind = m.group(1).upper()
        if kind == "NOOP":
            actions.append(_Action(kind="NOOP"))
            continue
        if kind == "INSERT":
            mm = re.search(
                r'(?:MEMORY[_ ]ITEM|NEW[_ ]MEMORY|CONTENT)\s*(?::|=|-)?\s*(.+?)(?=\n\s*(?:ACTION|MEMORY[_ ]ITEM|UPDATED[_ ]MEMORY|MEMORY[_ ]INDEX|REASONING)\b|$)',
                block, re.IGNORECASE | re.DOTALL,
            )
            if mm:
                content = mm.group(1).strip()
                if content:
                    actions.append(_Action(kind="INSERT", content=content))
        elif kind == "UPDATE":
            idx_m = re.search(r'MEMORY[_ ]INDEX\s*(?::|=|-)?\s*(\d+)', block, re.IGNORECASE)
            upd_m = re.search(
                r'UPDATED[_ ]MEMORY\s*(?::|=|-)?\s*(.+?)(?=\n\s*(?:ACTION|MEMORY[_ ]ITEM|UPDATED[_ ]MEMORY|MEMORY[_ ]INDEX|REASONING)\b|$)',
                block, re.IGNORECASE | re.DOTALL,
            )
            if idx_m and upd_m:
                idx = int(idx_m.group(1))
                if 0 <= idx < num_retrieved:
                    actions.append(_Action(kind="UPDATE", content=upd_m.group(1).strip(), index=idx))
        elif kind == "DELETE":
            idx_m = re.search(r'MEMORY[_ ]INDEX\s*(?::|=|-)?\s*(\d+)', block, re.IGNORECASE)
            if idx_m:
                idx = int(idx_m.group(1))
                if 0 <= idx < num_retrieved:
                    actions.append(_Action(kind="DELETE", index=idx))
    return actions


def _extract_json_object(text: str) -> Optional[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json|JSON)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    lo = text.find("{")
    hi = text.rfind("}")
    if lo < 0 or hi <= lo:
        return None
    try:
        return json.loads(text[lo:hi + 1])
    except json.JSONDecodeError:
        return None


def _extract_json_list(text: str) -> List[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json|JSON)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    lo = text.find("[")
    hi = text.rfind("]")
    if lo < 0 or hi <= lo:
        return []
    try:
        parsed = json.loads(text[lo:hi + 1])
        return [str(x) for x in parsed if isinstance(x, (str, int))]
    except json.JSONDecodeError:
        return []


# ── Memory / skill bank data structures ─────────────────────────────────
@dataclass
class _MemItem:
    content: str
    embedding: Optional[np.ndarray] = None
    op_history: List[str] = field(default_factory=list)


@dataclass
class _Skill:
    name: str
    description: str
    instruction_template: str
    update_type: str  # insert | update | delete

    def to_prompt_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "instruction_template": self.instruction_template,
            "update_type": self.update_type,
        }


@dataclass
class _FailureCase:
    question: str
    ground_truth: str
    prediction: str
    retrieved: List[str]
    f1_score: float = 0.0


@register_memory_unit("C15")
class MemSkillMemory(DenseRetrievalMemory):
    """Skill-conditioned memory construction with evolvable skill bank.

    Key methods:
      - ``write(text, metadata)`` — process a dialog chunk through the
        controller → executor → memory bank pipeline.
      - ``write_experience(query, trajectory, ...)`` — chunk a trajectory
        and process span-by-span.
      - ``read(query, top_k)`` — dense retrieval over the memory bank.
      - ``record_failure(question, gt, pred, retrieved, f1)`` — queue a
        failure case for the designer.
      - ``evolve_skills()`` — run one designer cycle to refine/add skills.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = getattr(config, "extra", {}) or {}
        self._top_k_skills: int = int(extra.get("memskill_top_k", 2))
        self._chunk_size: int = int(extra.get("memskill_chunk_size", 2000))
        self._retrieve_top_k: int = int(extra.get("memskill_retrieve_top_k", 5))
        self._designer_max_changes: int = int(extra.get("memskill_designer_max_changes", 2))
        self._designer_max_skills: int = int(extra.get("memskill_max_skills", 12))
        self._framework_model: str = str(
            extra.get("memskill_framework_model", "gpt-4o-mini")
        )
        # All framework-internal LLM calls (controller / executor / designer)
        # route through gpt-4o-mini. Only the downstream answer model (passed
        # in as `self.model`) stays as the main SubMem model (e.g. Qwen3.5-9B).
        self._framework_llm = _ControllerLLM(self._framework_model)

        # Initialise / extend token-consumption dict so the controller LLM
        # can bucket its calls into write/read/mgmt phases.
        if not hasattr(self, "_token_counts") or not isinstance(self._token_counts, dict):
            self._token_counts = {}
        for k in (
            "write_llm_input", "write_llm_output", "write_llm_calls",
            "read_llm_input",  "read_llm_output",  "read_llm_calls",
            "mgmt_llm_input",  "mgmt_llm_output",  "mgmt_llm_calls",
            "read_context_tokens", "read_calls",
        ):
            self._token_counts.setdefault(k, 0)
        # Wire the controller LLM to write into our counters.
        self._framework_llm._submem_owner = self

        self._skills: Dict[str, _Skill] = {}
        for sk in _SEED_SKILLS.values():
            self._skills[sk["name"]] = _Skill(**sk)

        self._failure_pool: List[_FailureCase] = []

        # Parallel array: embedding per entry in self._entries.
        # Lets rebuild_index skip re-encoding every row on UPDATE/DELETE.
        self._entry_embeddings: List[np.ndarray] = []

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────
    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Process a text chunk through the controller→executor pipeline."""
        if not text or not text.strip():
            return
        t0 = time.perf_counter()
        # ICL fallback: for test_time_learning the (input, label) text in the
        # chunk *is* the signal the downstream prompt needs. The controller→
        # executor pipeline rewrites chunks into abstract "skill" descriptions
        # which strips numeric labels — fatal for ICL classification. Skip
        # the pipeline and dense-add raw chunks instead so DenseRetrievalMemory.
        # read() returns the original text. See c9_lora._icl_dense_fallback.
        extra = getattr(self.config, "extra", {}) or {}
        if extra.get("benchmark_task_type") == "test_time_learning":
            self._add_to_index_cached(text, metadata)
            self._write_times.append(time.perf_counter() - t0)
            return
        for chunk in self._chunk_text(text):
            self._process_chunk(chunk, metadata or {})
        self._write_times.append(time.perf_counter() - t0)

    def write_experience(self, query: str, trajectory: str,
                         success: Optional[bool] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        md = dict(metadata or {})
        md.setdefault("query", query)
        if success is not None:
            md["success"] = success
        self.write(trajectory, metadata=md)

    def record_failure(self, question: str, ground_truth: str, prediction: str,
                        retrieved: List[str], f1_score: float = 0.0):
        self._failure_pool.append(_FailureCase(
            question=question, ground_truth=ground_truth, prediction=prediction,
            retrieved=list(retrieved), f1_score=float(f1_score),
        ))

    def evolve_skills(self) -> Dict[str, Any]:
        """Run one Designer cycle: analyze failure pool → refine/add skills."""
        if not self._failure_pool:
            return {"action": "no_change", "reason": "empty failure pool"}
        analysis = self._designer_analyze(self._failure_pool)
        changes = self._designer_refine(analysis)
        applied = self._apply_skill_changes(changes)
        # Clear pool after an evolution cycle.
        self._failure_pool.clear()
        return {"action": "evolved", "applied": applied, "analysis": analysis,
                "num_skills": len(self._skills)}

    # ────────────────────────────────────────────────────────────
    # Internals — pipeline
    # ────────────────────────────────────────────────────────────
    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i + self._chunk_size])
            i += self._chunk_size
        return chunks

    def _process_chunk(self, chunk: str, metadata: Dict[str, Any]):
        # Retrieve top-k existing memories as context for controller+executor.
        retrieved_contents, retrieved_indices = self._retrieve_for_write(chunk)

        # Controller: zero-shot LLM top-K skill selection.
        selected = self._controller_select(chunk, retrieved_contents)
        if not selected:
            return

        # Executor: ask LLM to emit action blocks.
        prompt = _build_executor_prompt(
            [s.to_prompt_dict() for s in selected],
            chunk, retrieved_contents,
        )
        resp = self._framework_llm.generate(prompt, max_new_tokens=1024)
        self._track_llm_call(prompt, resp)
        actions = _parse_executor_output(resp, len(retrieved_contents))

        # Apply to memory bank: UPDATEs first, then DELETEs in reverse,
        # then INSERTs (matches MemSkill/src/executor.py).
        op_name = "+".join(s.name for s in selected)
        self._apply_actions(actions, retrieved_indices, op_name, metadata)

    def _controller_select(self, chunk: str, retrieved: List[str]) -> List[_Skill]:
        skills = list(self._skills.values())
        k = min(self._top_k_skills, len(skills))
        if k <= 0 or not skills:
            return []
        if len(skills) <= k:
            return skills

        skill_list = "\n".join(
            f"- {s.name}: {s.description}" for s in skills
        )
        retr = "\n".join(f"{i}. {m}" for i, m in enumerate(retrieved)) or "(none)"
        prompt = _CONTROLLER_PROMPT.format(
            top_k=k, skill_list=skill_list, chunk=chunk, retrieved=retr,
        )
        resp = self._framework_llm.generate(prompt, max_new_tokens=128)
        self._track_mgmt_call(prompt, resp)

        picked = _extract_json_list(resp)
        chosen: List[_Skill] = []
        seen = set()
        for name in picked:
            name = name.strip().strip('"').strip("'")
            if name in self._skills and name not in seen:
                chosen.append(self._skills[name])
                seen.add(name)
            if len(chosen) >= k:
                break
        # Fallback: fill with seed order if parse failed / under-selected.
        if len(chosen) < k:
            for s in skills:
                if s.name not in seen:
                    chosen.append(s)
                    seen.add(s.name)
                if len(chosen) >= k:
                    break
        return chosen[:k]

    # ────────────────────────────────────────────────────────────
    # Memory bank ops
    # ────────────────────────────────────────────────────────────
    def _retrieve_for_write(self, chunk: str) -> Tuple[List[str], List[int]]:
        if self._index is None or self._index.ntotal == 0:
            return [], []
        q_emb = self._encode([chunk])
        k = min(self._retrieve_top_k, self._index.ntotal)
        _, indices = self._index.search(q_emb, k)
        contents, idxs = [], []
        for idx in indices[0]:
            if idx < 0:
                continue
            contents.append(self._entries[idx].content)
            idxs.append(int(idx))
        return contents, idxs

    def _apply_actions(self, actions: List[_Action], retrieved_indices: List[int],
                        op_name: str, metadata: Dict[str, Any]):
        updates = [a for a in actions if a.kind == "UPDATE"]
        deletes = [a for a in actions if a.kind == "DELETE"]
        inserts = [a for a in actions if a.kind == "INSERT"]

        dirty = False

        # 1. UPDATEs (before any DELETEs change indices).
        for a in updates:
            actual = retrieved_indices[a.index]
            if 0 <= actual < len(self._entries):
                new_emb = self._encode([a.content])[0]
                self._entries[actual] = MemoryEntry(
                    content=a.content,
                    metadata={**self._entries[actual].metadata,
                              "source": "memskill",
                              "op": op_name},
                    timestamp=self._entries[actual].timestamp,
                )
                self._entry_embeddings[actual] = new_emb
                dirty = True

        # 2. DELETEs in reverse order of actual index.
        delete_actuals = sorted(
            {retrieved_indices[a.index] for a in deletes},
            reverse=True,
        )
        if delete_actuals:
            for actual in delete_actuals:
                if 0 <= actual < len(self._entries):
                    del self._entries[actual]
                    del self._entry_embeddings[actual]
            dirty = True

        if dirty:
            self._rebuild_index_from_cached_embeddings()

        # 3. INSERTs — batch-embed all inserts in a single API call, then
        # add sequentially to the FAISS index. Reduces per-chunk OpenAI
        # round-trips from N+2 to 3 (controller chat + executor chat + 1
        # batched embed call), roughly halving wall time on LME(S*).
        if inserts:
            texts = [a.content for a in inserts]
            embs = self._encode(texts)   # shape [N, dim]
            for i, a in enumerate(inserts):
                self._write_call_count += 1
                emb = embs[i:i+1]
                self._index.add(emb)
                self._entries.append(MemoryEntry(
                    content=a.content,
                    metadata={**metadata, "source": "memskill", "op": op_name},
                    timestamp=len(self._entries),
                ))
                self._entry_embeddings.append(embs[i])

    def _add_to_index_cached(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Incremental add that also caches embedding for later rebuilds."""
        self._write_call_count += 1
        emb = self._encode([text])
        self._index.add(emb)
        self._entries.append(MemoryEntry(
            content=text,
            metadata=metadata or {},
            timestamp=len(self._entries),
        ))
        self._entry_embeddings.append(emb[0])

    def _rebuild_index_from_cached_embeddings(self):
        """Rebuild FAISS index from cached embeddings — no re-encoding."""
        import faiss
        self._index = faiss.IndexFlatIP(self._enc.dim)
        if not self._entry_embeddings:
            return
        embs = np.vstack(self._entry_embeddings).astype("float32")
        self._index.add(embs)

    # ────────────────────────────────────────────────────────────
    # Designer
    # ────────────────────────────────────────────────────────────
    def _designer_analyze(self, cases: List[_FailureCase]) -> str:
        skills_desc = "\n".join(
            f"- **{s.name}** (type: {s.update_type}): {s.description}"
            for s in self._skills.values()
        )
        case_str_parts = []
        for i, c in enumerate(cases[:20]):  # cap prompt size
            mems = "\n".join(f"  {j + 1}. {m}" for j, m in enumerate(c.retrieved[:10]))
            case_str_parts.append(
                f"### Case {i + 1}\n"
                f"**Question:** {c.question}\n"
                f"**Expected:** {c.ground_truth}\n"
                f"**Predicted:** {c.prediction}\n"
                f"**Retrieved Memories:**\n{mems if mems else '  (none)'}\n"
            )
        prompt = _DESIGNER_ANALYSIS_PROMPT.format(
            skills_desc=skills_desc,
            n=len(cases),
            cases="\n".join(case_str_parts),
            max_changes=self._designer_max_changes,
        )
        resp = self._framework_llm.generate(prompt, max_new_tokens=1024)
        self._track_mgmt_call(prompt, resp)
        return resp

    def _designer_refine(self, analysis: str) -> List[Dict[str, Any]]:
        skills_full = "\n".join(
            f"### {s.name}\n- Type: {s.update_type}\n- Description: {s.description}\n"
            f"- Template:\n```\n{s.instruction_template}\n```"
            for s in self._skills.values()
        )
        prompt = _DESIGNER_REFINEMENT_PROMPT.format(
            analysis=analysis,
            skills_full=skills_full,
            max_changes=self._designer_max_changes,
        )
        resp = self._framework_llm.generate(prompt, max_new_tokens=1536)
        self._track_mgmt_call(prompt, resp)
        obj = _extract_json_object(resp)
        if not isinstance(obj, dict):
            return []
        changes = obj.get("changes")
        if not isinstance(changes, list):
            return []
        return [c for c in changes if isinstance(c, dict)]

    def _apply_skill_changes(self, changes: List[Dict[str, Any]]) -> List[str]:
        applied: List[str] = []
        for change in changes[:self._designer_max_changes]:
            action = str(change.get("action", "")).lower().strip()
            if action == "add_new":
                new_op = change.get("new_operation") or {}
                name = str(new_op.get("name", "")).strip()
                desc = str(new_op.get("description", "")).strip()
                tmpl = str(new_op.get("instruction_template", "")).strip()
                ut = str(new_op.get("update_type", "")).lower().strip()
                if not (name and desc and tmpl and ut in ("insert", "update")):
                    continue
                if "{session_text}" in tmpl or "{retrieved_memories}" in tmpl:
                    continue
                if name in self._skills:
                    continue
                if len(self._skills) >= self._designer_max_skills:
                    # Evict: drop the non-seed skill with the lexicographically
                    # smallest name (deterministic placeholder for usage-based
                    # eviction in the paper — we don't track usage here).
                    evictable = [n for n in self._skills
                                 if n not in _SEED_SKILLS]
                    if not evictable:
                        continue
                    del self._skills[sorted(evictable)[0]]
                self._skills[name] = _Skill(
                    name=name, description=desc,
                    instruction_template=tmpl, update_type=ut,
                )
                applied.append(f"add_new:{name}")
            elif action == "refine_existing":
                ref = change.get("refined_operation") or {}
                name = str(ref.get("name", "")).strip()
                if name not in self._skills:
                    continue
                updates = ref.get("changes") or {}
                if not isinstance(updates, dict):
                    continue
                skill = self._skills[name]
                if "description" in updates and isinstance(updates["description"], str):
                    skill.description = updates["description"].strip()
                if "instruction_template" in updates and isinstance(updates["instruction_template"], str):
                    tmpl = updates["instruction_template"].strip()
                    if tmpl and "{session_text}" not in tmpl and "{retrieved_memories}" not in tmpl:
                        skill.instruction_template = tmpl
                applied.append(f"refine_existing:{name}")
        return applied

    # ────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────
    def _on_reset(self):
        # Reset skills to seeds, clear failure pool and embedding cache.
        self._skills = {
            name: _Skill(**sk) for name, sk in _SEED_SKILLS.items()
        }
        self._failure_pool.clear()
        self._entry_embeddings = []

    # ── cache support ─────────────────────────────────────────
    # NOTE: skill bank state (_skills, _failures, etc.) is NOT cached yet.
    # Cache only the dense_retrieval base state (entries + faiss + tokens).
    # Re-running with same extras will trigger skill bank rebuild from scratch.
    CACHE_KEY_EXTRA = ["memskill_top_k", "memskill_chunk_size",
                       "memskill_designer_max_changes", "memskill_max_skills"]

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "num_skills": len(self._skills),
            "skill_names": sorted(self._skills.keys()),
            "failure_pool_size": len(self._failure_pool),
            "token_consumption": dict(self._token_counts),
        })
        return stats
