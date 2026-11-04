from .base import BaseMemoryUnit, MEMORY_UNIT_REGISTRY
from .m1_dense_vector import DenseVectorMemory         # noqa: F401 — registers M1
from .m2_sparse_vector import SparseVectorMemory       # noqa: F401 — registers M2
from .m3_summarized_text import SummarizedTextMemory   # noqa: F401 — registers M3
from .m6_hierarchical import HierarchicalMemory        # noqa: F401 — registers M6
# C8 (MEMIT) removed: upstream kmeng01/memit targets GPT-J/GPT-2; Qwen3's
# bare-tensor layer outputs + n_embd→hidden_size naming + compute_z hook
# shape mismatch make a faithful port non-trivial. Dropped rather than
# ship an adaptation. See history in commit logs if reviving.
from .m9_lora import LoRAMemory                        # noqa: F401 — registers M9
from .m10_full_context import FullContextMemory        # noqa: F401 — registers M10
from .m11_epicache import EpiCacheMemory               # noqa: F401 — registers M11
from .m7_reasoning_bank import ReasoningBankMemory     # noqa: F401 — registers M7
try:
    from .m4_amem import AMemMemory                    # noqa: F401 — registers M4
except ImportError as _e:
    import sys as _sys
    print(f"[memory/__init__] M4 import failed: {_e}", file=_sys.stderr)

try:
    from .m8_memskill_faithful import MemSkillFaithfulMemory  # noqa: F401 — registers M8
except ImportError as _e:
    import sys as _sys
    print(f"[memory/__init__] M8 import failed: {_e}", file=_sys.stderr)
try:
    from .m5_lightrag import LightRAGMemory            # noqa: F401 — registers M5 LightRAG
except ImportError as _e:
    import sys as _sys
    print(f"[memory/__init__] M5 LightRAG import failed: {_e}", file=_sys.stderr)


def create_memory_unit(config, model):
    """Factory: create memory unit from config.

    Accepts either a modern M-id or a legacy C-id as config.memory_id —
    the id is normalised through config.to_modern_id() so old scripts
    and saved configs that still say e.g. "C1" keep working.
    """
    from ..config import to_modern_id
    requested = config.memory_id
    unit_cls = MEMORY_UNIT_REGISTRY.get(to_modern_id(requested))
    if unit_cls is None:
        unit_cls = MEMORY_UNIT_REGISTRY.get(requested)
    if unit_cls is None:
        raise NotImplementedError(
            f"Memory unit {requested} not yet implemented. "
            f"Available: {list(MEMORY_UNIT_REGISTRY.keys())}"
        )
    return unit_cls(config, model)
