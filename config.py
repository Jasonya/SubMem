"""Experiment configuration."""
import os
from dataclasses import dataclass, field
from typing import Optional

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3.5-9B"
    device: str = "cuda"
    dtype: str = "float16"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    device_map: Optional[str] = "auto"       # "auto" for multi-GPU, "cuda:0" for single
    load_in_4bit: bool = False                # bitsandbytes 4-bit quantization
    load_in_8bit: bool = False                # bitsandbytes 8-bit quantization
    attn_implementation: Optional[str] = None  # "flash_attention_2", "sdpa", etc.


@dataclass
class MemoryConfig:
    memory_id: str = "C1"  # C1-C13
    capacity: Optional[int] = None
    # `None` means "use the substrate's method-level default" — each
    # memory_unit.read() has its own paper-faithful top_k (e.g. C14
    # defaults to 1, C1 to 10). When a user passes --top-k, harness
    # code sets this and overrides the substrate's default.
    top_k: Optional[int] = None
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    name: str = "locomo"
    data_dir: str = "data"
    split: str = "test"
    max_samples: Optional[int] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    seed: int = 42
    output_dir: str = "results"
    num_seeds: int = 3
    seeds: list = field(default_factory=lambda: [42, 123, 456])


# ── Substrate ID nomenclature: code-internal C* ↔ paper-facing M* ──
#
# The codebase historically used C<n> identifiers. The paper presents
# substrates as M<n> (see Table tab:substrate). The two numberings are
# NOT identity — e.g. C4 is paper M3, C16 is M4. SUBSTRATE_ID_MAP below
# is the single source of truth; both the registry and the result
# loaders resolve through it.
#
# M1-M11 are the paper's 11-row substrate table. M8 is the upstream
# paper-faithful MemSkill (code C20). M-numbers carry the substrate's
# *.py basename suffix (m3_summarized_text.py registers "M3").
#
# Retired substrates (memory/archived/) keep their C-id; they are recorded
# here only so result loaders scanning old multi-substrate run dirs can
# pass them through instead of crashing.
SUBSTRATE_ID_MAP = {
    # code C-id : paper M-id
    "C1":  "M1",    # Dense Vector
    "C2":  "M2",    # Sparse Vector (BM25)
    "C4":  "M3",    # Gist Index (ReadAgent / summarized text)
    "C16": "M4",    # Evolving Notes (A-Mem)
    "C18": "M5",    # Dual-Level Graph (LightRAG)
    "C7":  "M6",    # Hierarchical Tree
    "C14": "M7",    # Distilled Strategies (ReasoningBank)
    "C20": "M8",    # Skill Bundles (paper-faithful MemSkill, upstream)
    "C9":  "M9",    # Adapter Tuning (LoRA)
    "C10": "M10",   # Full Context
    "C12": "M11",   # Episode-Clustered Re-prefill (EpiCache)
}
SUBSTRATE_ID_MAP_REVERSE = {v: k for k, v in SUBSTRATE_ID_MAP.items()}

# Retired substrate C-ids — not runnable, not in SUBSTRATE_ID_MAP, but
# may appear as subdirectory names inside old results/ run directories.
RETIRED_SUBSTRATES = {
    "C3", "C3-Letta", "C5", "C6", "C8", "C11", "C13", "C15", "C17",
}


def _split_variant(substrate_id):
    """Split a substrate id into (base, variant-suffix).

    Substrate ids may carry a '-' suffix variant: 'C10-3K', 'C3-Letta',
    'M10-3K'. The suffix is preserved verbatim across id translation.
    Returns (base_id, suffix) where suffix includes the leading '-' or
    is '' when absent.
    """
    if not substrate_id:
        return substrate_id, ""
    # 'C3-Letta' is a known whole id, not base 'C3' + variant.
    if substrate_id in RETIRED_SUBSTRATES:
        return substrate_id, ""
    if "-" in substrate_id:
        base, _, rest = substrate_id.partition("-")
        return base, "-" + rest
    return substrate_id, ""


def to_modern_id(substrate_id):
    """Return the paper-facing M-id for any substrate id.

    Accepts an M-id (returned unchanged), a C-id (translated via
    SUBSTRATE_ID_MAP), or a retired / unknown id (passed through).
    Any '-' variant suffix (e.g. '-3K') is preserved.
    """
    base, suffix = _split_variant(substrate_id)
    if base in SUBSTRATE_ID_MAP:
        return SUBSTRATE_ID_MAP[base] + suffix
    # already an M-id, or retired/unknown — pass through
    return substrate_id


def to_legacy_id(substrate_id):
    """Return the code-internal C-id for any substrate id.

    Inverse of to_modern_id: an M-id is translated back to its C-id,
    a C-id / retired / unknown id is passed through. Used to locate
    old results/ directories, which are named with C-ids.
    """
    base, suffix = _split_variant(substrate_id)
    if base in SUBSTRATE_ID_MAP_REVERSE:
        return SUBSTRATE_ID_MAP_REVERSE[base] + suffix
    return substrate_id


def resolve_substrate_dir(run_dir, substrate_id):
    """Locate a substrate's subdirectory inside a results/ run directory.

    New runs write M-id subdirs; old runs wrote C-id subdirs. Given a
    requested substrate id (either form), return the path to whichever
    subdir actually exists, or None. Tries the requested id, then its
    legacy C-id, then its modern M-id.
    """
    for cand in (substrate_id, to_legacy_id(substrate_id),
                 to_modern_id(substrate_id)):
        path = os.path.join(run_dir, cand)
        if os.path.isdir(path):
            return path
    return None


def iter_substrate_dirs(run_dir):
    """Enumerate substrate subdirs of a run dir as (modern_id, path).

    Handles run dirs containing C-id subdirs (old), M-id subdirs (new),
    retired-substrate subdirs, and '-' variant suffixes. The returned
    id is always canonicalised to the modern M-id via to_modern_id().
    """
    out = []
    if not os.path.isdir(run_dir):
        return out
    for name in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, name)
        if not os.path.isdir(full):
            continue
        base, _ = _split_variant(name)
        if base in SUBSTRATE_ID_MAP or base in SUBSTRATE_ID_MAP_REVERSE \
                or name in RETIRED_SUBSTRATES or base in RETIRED_SUBSTRATES:
            out.append((to_modern_id(name), full))
    return out


# Active memory unit IDs (registered & runnable), paper-facing M-ids.
# Derived from SUBSTRATE_ID_MAP so there is one source of truth; ordered
# by M-number. Retired substrates (see RETIRED_SUBSTRATES) are excluded.
MEMORY_UNIT_IDS = sorted(
    SUBSTRATE_ID_MAP.values(),
    key=lambda m: int(m[1:]),
)

BENCHMARK_NAMES = [
    "locomo",
    "longmemeval",
    "memory_agent_bench",
    "terminal_bench",
    "webarena_lite",
    "alfworld",
]

# HuggingFace dataset repo IDs — resolved to local cache automatically
DATASET_REPOS = {
    "locomo": None,  # official source: snap-research/locomo GitHub, stored locally
    "longmemeval": "xiaowu0162/longmemeval-cleaned",
    "memory_agent_bench": "ai-hyz/MemoryAgentBench",
    "terminal_bench": "harborframework/terminal-bench-2.0",
    "webarena_lite": "weizhepei/WebArena-Lite-SFT",
    "alfworld": None,  # preprocessed from dslighfdsl/alfworld_train_all, local only
}

SUBMEM_CACHE = os.path.expanduser("~/.cache/submem/datasets")


def get_dataset_path(benchmark_name: str) -> str:
    """Resolve dataset to local path.

    Priority: ~/.cache/submem/datasets/<name>/ (or symlink) → HF cache via repo ID
    """
    # Check submem cache first (works for locomo, longmemeval, etc.)
    local_path = os.path.join(SUBMEM_CACHE, benchmark_name)
    if os.path.isdir(local_path):
        return local_path

    repo_id = DATASET_REPOS[benchmark_name]
    if repo_id is None:
        raise FileNotFoundError(f"No repo configured for {benchmark_name}")
    cache_dir = os.path.join(HF_CACHE, f"datasets--{repo_id.replace('/', '--')}")
    refs_path = os.path.join(cache_dir, "refs", "main")
    if os.path.exists(refs_path):
        with open(refs_path) as f:
            commit = f.read().strip()
        return os.path.join(cache_dir, "snapshots", commit)
    # Fallback: find first snapshot
    snap_dir = os.path.join(cache_dir, "snapshots")
    if os.path.isdir(snap_dir):
        commits = os.listdir(snap_dir)
        if commits:
            return os.path.join(snap_dir, commits[0])
    raise FileNotFoundError(f"Dataset not found locally: {repo_id}. Run download first.")
