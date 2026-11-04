"""Adapter exposing SubMem substrates (M1-M8) as MemoryArena memory systems.

Implements MemoryArena's minimal memory interface — add_chunk(chunk) and
wrap_user_prompt(prompt) — on top of a SubMem memory unit, so the paper's
substrates can run inside MemoryArena's multi-session Memory-Agent-Env loop
without modification.

Internal substrates (M9 LoRA, M10 full-context-of-local-model, M11 EpiCache)
require weight/activation access to the task agent, which MemoryArena's
API-backed agents don't expose; they are out of scope for this adapter,
consistent with the external/internal taxonomy in the paper.

Auxiliary LLM operations inside substrates (M3 gist, M4 evolve, M5 graph
extraction, M6 RAPTOR summaries, M7 distillation) run through SubMem's API
model backend (gpt-4o-mini by default, matching the paper's harness aux).

Usage: set memory_system_name to e.g. "submem_m1" ... "submem_m8" after
registering this module in memory/server.py.
"""
import json
import os
import sys
from pathlib import Path

# The SubMem checkout: SUBMEM_ROOT if set, else derived from this file's place
# in the repository (benchmarks/memoryarena/adapter/). setup.sh copies this file
# into MemoryArena's memory/memory_systems/, where the derived default is wrong,
# so the job scripts export SUBMEM_ROOT and SUBMEM_PARENT explicitly.
SUBMEM_ROOT = Path(os.environ.get("SUBMEM_ROOT", Path(__file__).resolve().parents[3]))
SUBMEM_PARENT = os.environ.get("SUBMEM_PARENT", str(SUBMEM_ROOT.parent))
if SUBMEM_PARENT not in sys.path:
    sys.path.insert(0, SUBMEM_PARENT)


class SubMemMemorySystem:
    """Wrap one SubMem substrate behind MemoryArena's memory interface."""

    def __init__(self, memory_id: str = "M1", user_id: str = None,
                 top_k: int = 5, aux_model: str = "openai/gpt-4o-mini"):
        from SubMem.config import MemoryConfig, ModelConfig
        from SubMem.models.api import APIModel
        from SubMem.memory import create_memory_unit

        self.memory_id = memory_id
        self.top_k = top_k
        self.user_id = user_id

        # M11 (EpiCache) needs the backbone's attention scores (_blockwise_score),
        # which requires a local HF model. When SUBMEM_M11_HF_MODEL is set, give
        # M11 a local HF backbone (same 8B as the agent); other substrates keep
        # the API aux model.
        import os as _os
        _hf = (_os.environ.get("SUBMEM_M11_HF_MODEL") if memory_id == "M11"
               else _os.environ.get("SUBMEM_M9_HF_MODEL") if memory_id == "M9" else None)
        if _hf:
            from SubMem.models import create_model
            # One HF backbone per (name, dtype) shared across all memory-system
            # instances: MemoryArena creates a fresh system per paper, and
            # loading a new 16GB model each time leaks the GPU (papers x 16GB).
            global _HF_MODEL_CACHE
            try:
                _HF_MODEL_CACHE
            except NameError:
                _HF_MODEL_CACHE = {}
            _dt = "bfloat16" if memory_id == "M9" else "float16"
            _key = (_hf, _dt)
            if _key not in _HF_MODEL_CACHE:
                model_cfg = ModelConfig(name=_hf, dtype=_dt)
                m = create_model(model_cfg)
                m.load()
                _HF_MODEL_CACHE[_key] = m
            model = _HF_MODEL_CACHE[_key]
        else:
            model_cfg = ModelConfig(name=aux_model)
            model = APIModel(model_cfg)
            model.load()  # required: initializes API client + tokenizer

        mem_cfg = MemoryConfig(memory_id=memory_id)
        mem_cfg.top_k = top_k
        if memory_id == "M9":
            # keep per-session retrains tractable inside a live agent loop
            mem_cfg.extra = {"lora_epochs": 3}
        self._unit = create_memory_unit(mem_cfg, model)
        self._traj_buf = []  # M7 experience buffering (see add_chunk)

    def add_chunk(self, chunk: str):
        if not (chunk and chunk.strip()):
            return
        # SUBMEM_KEEP_BASE probe: distilling substrates (M7/M8/M11) turn the
        # opening observation into strategy items, which drops the shared
        # trip specification the task never restates in the query. Keep the
        # first chunk verbatim so refinement is additive, as upstream intends.
        if os.environ.get("SUBMEM_KEEP_BASE") == "1" and not getattr(self, "_base_chunk", None):
            self._base_chunk = chunk[:4000]
        # M7 ReasoningBank only accepts (query, trajectory) experiences —
        # plain write() raises NotImplementedError. Buffer chunks (each is
        # an action/obs step from the env loop) and flush them as one
        # trajectory at the next wrap_user_prompt, using the incoming
        # question as the experience query. This approximates a session
        # boundary and matches ReasoningBank's distill-from-trajectory
        # design.
        if self.memory_id == "M7":
            self._traj_buf.append(chunk)
            return
        # Paper-harness ingestion standardization: single-blob inputs are
        # chunked at 500 words before write (see the calibration protocol).
        # This also keeps per-write payloads inside the embedding API's
        # 8192-token per-request limit and keeps M11's utterances at the
        # granularity its blockwise scorer expects.
        words = chunk.split()
        if len(words) <= 500:
            self._unit.write(chunk)
        else:
            for i in range(0, len(words), 500):
                piece = " ".join(words[i:i + 500])
                if piece.strip():
                    self._unit.write(piece)

    def wrap_user_prompt(self, prompt: str) -> str:
        if self.memory_id == "M7" and getattr(self, "_traj_buf", None):
            try:
                self._unit.write_experience(
                    query=prompt[:300],
                    trajectory="\n".join(self._traj_buf),
                )
            except Exception as e:
                print(f"[submem_adapter:M7] write_experience failed: {e}",
                      flush=True)
            self._traj_buf = []
        try:
            entries = self._unit.read(prompt, top_k=self.top_k)
        except Exception as e:
            entries = []
            print(f"[submem_adapter:{self.memory_id}] read failed: {e}",
                  flush=True)
        if not entries:
            mem_text = "None"
        else:
            # M4's structured notes can surface dict content; coerce so the
            # join never 500s the whole wrap call.
            mem_text = "\n".join(
                e.content if isinstance(e.content, str)
                else json.dumps(e.content, ensure_ascii=False, default=str)
                for e in entries
            )
        # Cap injected context so prompt + completion fits the 32k serving
        # window; keep the tail (most recent). Char-based: LaTeX-heavy text
        # tokenizes at 2-3x per word, so a word cap still overflowed. 70k
        # chars is roughly 18k tokens worst case. Bounded substrates never
        # hit this; M10's full-history read does on long math sessions.
        base = getattr(self, "_base_chunk", None)
        if base:
            mem_text = "<session_context>\n" + base + "\n</session_context>\n" + mem_text
        # Cap the total injected memory, base included. Capping only the
        # retrieved part let the prepended session context push the prompt
        # past the serving window for substrates whose read is already large.
        _cap = int(os.environ.get("SUBMEM_MEM_CHAR_CAP", "70000"))
        if len(mem_text) > _cap:
            mem_text = mem_text[-_cap:]
        return "\n".join(["<memory_context>", mem_text, "</memory_context>", "", prompt])


def make_submem_system(memory_id: str):
    """Factory helper: make_submem_system('M5') -> class bound to M5."""
    class _Bound(SubMemMemorySystem):
        def __init__(self, user_id=None, **kw):
            super().__init__(memory_id=memory_id, user_id=user_id, **kw)
    _Bound.__name__ = f"SubMem{memory_id}MemorySystem"
    return _Bound
