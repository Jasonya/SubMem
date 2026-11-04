"""Cache infrastructure for memory write-state reuse across sweeps.

Goal: write a memory once, read N times with different (gen_model, top_k,
prompt). Substrates implement save_state / load_state on BaseMemoryUnit;
this module provides keying, lookup, and the index.
"""
from .cache_io import (
    compute_cache_key,
    cache_lookup,
    cache_store,
    cache_root,
    CacheKey,
)

__all__ = [
    "compute_cache_key",
    "cache_lookup",
    "cache_store",
    "cache_root",
    "CacheKey",
]
