"""Base memory unit interface.

13 memory units (C1-C13) will be plugged in here.
Each must implement write() and read() with its own mechanism.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """A single item stored in or retrieved from memory."""
    content: str
    metadata: Dict[str, Any] = None
    score: float = 0.0
    timestamp: int = 0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseMemoryUnit(ABC):
    """Abstract interface for all memory substrates.

    Substrate identifiers are the paper-facing M-numbers (paper Table
    tab:substrate). The code historically used C-numbers; the canonical
    C↔M mapping lives in config.SUBSTRATE_ID_MAP.

    Active substrates (registered & runnable):
      M1  - DenseVectorMemory        (Dense Vector)
      M2  - SparseVectorMemory       (Sparse Vector / BM25)
      M3  - SummarizedTextMemory     (Gist Index, ref: ReadAgent)
      M4  - AMemMemory               (Evolving Notes, ref: A-Mem, Xu+ '25)
      M5  - LightRAGMemory           (Dual-Level Graph, ref: LightRAG, Guo+ '24)
      M6  - HierarchicalMemory       (Hierarchical Tree, ref: RAPTOR / H-MEM)
      M7  - ReasoningBankMemory      (Distilled Strategies, ref: Ouyang+ ICLR'26)
      M8  - MemSkillFaithfulMemory   (Skill Bundles, paper-faithful MemSkill)
      M9  - LoRAMemory               (Adapter Tuning, ref: LoRA)
      M10 - FullContextMemory        (Full Context / baseline)
      M11 - EpiCacheMemory           (Episode-Clustered Re-prefill, ref: EpiCache)

    Retired substrates (memory/archived/, not imported or registered),
    listed by their original C-id:
      C3 RawText, C3-Letta, C5 Mem0, C6 Graphiti, C8 MEMIT,
      C11 SnapKV, C13 Sleep-time, C15 MemSkill (re-impl), C17 Graph-RAG.
    """

    # When True, `read()` returns a ready answer (parametric memory), so
    # benchmark runners MUST NOT re-prompt the generator on top of the
    # retrieved content. Default False = retrieve-then-generate semantics.
    PARAMETRIC_OUTPUT: bool = False

    def __init__(self, config, model):
        self.config = config
        self.model = model

    @abstractmethod
    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Write information into memory.

        Args:
            text: The text/information to store.
            metadata: Optional metadata (timestamp, source, etc.)
        """

    @abstractmethod
    def read(self, query: str, top_k: int = 10) -> List[MemoryEntry]:
        """Read/retrieve from memory given a query.

        Args:
            query: The retrieval query.
            top_k: Number of results to return.

        Returns:
            List of MemoryEntry ranked by relevance.
        """

    def write_experience(
        self,
        query: str,
        trajectory: str,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Write an agentic (query, trajectory) experience into memory.

        Used by agentic benchmarks (WebArena-Lite, SWE-bench, etc.) where the
        unit of storage is a whole task attempt, not a single dialog turn.

        Default: fall back to ``write()`` with the trajectory as text so that
        memory units designed for dialog (C1-C13) still behave sensibly.
        C14 (ReasoningBank) overrides this with judge + distill.

        Args:
            query: the task / question the agent was trying to solve.
            trajectory: the agent's action+observation trace as text.
            success: whether the trajectory succeeded. If None, the memory
                unit may self-judge using its LLM (per the ReasoningBank paper).
            metadata: optional metadata (sample_id, session_id, etc.).
        """
        md = dict(metadata or {})
        md.setdefault("query", query)
        if success is not None:
            md.setdefault("success", success)
        self.write(trajectory, metadata=md)

    @abstractmethod
    def reset(self):
        """Clear all stored memory."""

    @abstractmethod
    def size(self) -> int:
        """Return number of items currently stored."""

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic stats (storage size, GPU usage, etc.)."""
        return {
            "memory_id": self.config.memory_id,
            "num_entries": self.size(),
        }

    @property
    def requires_training(self) -> bool:
        """Whether this memory unit needs a training/editing step."""
        return False

    def train(self, data: List[Dict[str, str]]):
        """Training step for weight-based memory (C7, C8). No-op by default."""
        pass

    def get_gpu_memory_bytes(self) -> int:
        """Return GPU memory consumed by this memory unit (bytes)."""
        return 0

    def get_external_storage_bytes(self) -> int:
        """Return external (disk) storage consumed (bytes)."""
        return 0

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return all stored entries (for diagnostics / metrics)."""
        return []

    # ── Cache interface (write-once, reuse-many) ─────────────────
    #
    # Substrates override these to support caching the post-write memory
    # state on disk so subsequent runs (different gen_model, top_k, prompt)
    # can skip the write phase and just read.
    #
    # Bump CACHE_SCHEMA_VERSION when serialize format changes (auto-invalidates
    # all existing caches for that substrate).
    CACHE_SCHEMA_VERSION = "v1"

    # Whitelist of `config.extra` keys that materially affect write output.
    # Anything not in this list is excluded from the cache key (so e.g.
    # changing verbose=True doesn't invalidate caches).
    CACHE_KEY_EXTRA: List[str] = []

    # Substrates whose write phase consumes the *generator* model (not just
    # encoder/aux). C9/C11/C12/C13. These set REQUIRES_GEN_MODEL_IN_KEY=True
    # so the cache key includes gen_model.
    REQUIRES_GEN_MODEL_IN_KEY: bool = False

    def save_state(self, dir_path: str) -> None:
        """Serialize post-write state to dir. Subclasses must implement
        if they want cache support. No-op by default = uncacheable."""
        raise NotImplementedError(f"{type(self).__name__} does not support cache save_state")

    @classmethod
    def load_state(cls, dir_path: str, config, model) -> "BaseMemoryUnit":
        """Reconstruct unit from saved state. Must restore enough that
        read() works correctly without running write() again."""
        raise NotImplementedError(f"{cls.__name__} does not support cache load_state")

    def supports_caching(self) -> bool:
        """Return True iff save_state/load_state are properly implemented."""
        return type(self).save_state is not BaseMemoryUnit.save_state


# Registry: populated as each memory unit is implemented
MEMORY_UNIT_REGISTRY: Dict[str, type] = {}


def register_memory_unit(unit_id: str):
    """Decorator to register a memory unit implementation."""
    def decorator(cls):
        MEMORY_UNIT_REGISTRY[unit_id] = cls
        return cls
    return decorator
