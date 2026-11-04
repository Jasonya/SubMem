"""Cache key computation, lookup, and JSONL index.

Layout:
    $SUBMEM_CACHE/<benchmark>/by_hash/<hash>/   # payload
    $SUBMEM_CACHE/<benchmark>/CACHE_INDEX.jsonl # append-only index

Cache key (per-substrate, per-sample):
    {benchmark, substrate, sample_id, encoder, encoder_dim, aux_model,
     gen_model_at_write?, substrate_extra, substrate_src_sha,
     cache_schema_version}
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def cache_root() -> Path:
    """Resolve cache root from env, default ~/.cache/submem/cache."""
    p = os.environ.get("SUBMEM_CACHE")
    if p:
        return Path(p)
    return Path(os.path.expanduser("~/.cache/submem/cache"))


def benchmark_root(benchmark: str) -> Path:
    root = cache_root() / benchmark
    root.mkdir(parents=True, exist_ok=True)
    (root / "by_hash").mkdir(exist_ok=True)
    return root


def index_path(benchmark: str) -> Path:
    return benchmark_root(benchmark) / "CACHE_INDEX.jsonl"


# ---------------------------------------------------------------------------
# Source hashing — only the substrate's own .py file
# ---------------------------------------------------------------------------

_SRC_HASH_CACHE: Dict[str, str] = {}


def substrate_src_hash(substrate_module_path: str) -> str:
    """SHA256 of the substrate's source file. Cached in-process."""
    if substrate_module_path in _SRC_HASH_CACHE:
        return _SRC_HASH_CACHE[substrate_module_path]
    try:
        with open(substrate_module_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        h = "unknown"
    _SRC_HASH_CACHE[substrate_module_path] = h
    return h


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CacheKey:
    benchmark: str            # "locomo" / "longmemeval" / ...
    substrate: str            # "C3-Letta", "C1", ...
    sample_id: str            # "conv-26"
    encoder: str              # "text-embedding-3-small"
    encoder_dim: int          # 1536
    aux_model: str            # "openai/gpt-4o-mini" or "" if substrate doesn't use one
    gen_model_at_write: str   # set only for REQUIRES_GEN_MODEL_IN_KEY substrates; "" otherwise
    substrate_extra: Dict[str, Any]   # filtered to CACHE_KEY_EXTRA whitelist
    substrate_src_sha: str    # short sha256 of substrate's .py file
    cache_schema_version: str # CACHE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        # Stable JSON: sorted keys + canonical types
        normalized = {
            **self.to_dict(),
            "substrate_extra": dict(sorted(self.substrate_extra.items())),
        }
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def compute_cache_key(
    *,
    benchmark: str,
    substrate: str,
    sample_id: str,
    memory_unit_cls,           # the BaseMemoryUnit subclass
    encoder: str,
    encoder_dim: int,
    aux_model: str,
    gen_model: str,
    extra: Dict[str, Any],
    substrate_module_path: str,
) -> CacheKey:
    """Build the canonical cache key for this run."""
    # Filter extra by whitelist
    whitelist = set(getattr(memory_unit_cls, "CACHE_KEY_EXTRA", []) or [])
    filtered_extra = {k: extra[k] for k in sorted(whitelist) if k in extra}

    # gen_model only matters for substrates whose write uses it
    gen_at_write = ""
    if getattr(memory_unit_cls, "REQUIRES_GEN_MODEL_IN_KEY", False):
        gen_at_write = gen_model or ""

    return CacheKey(
        benchmark=benchmark,
        substrate=substrate,
        sample_id=sample_id,
        encoder=encoder or "",
        encoder_dim=int(encoder_dim or 0),
        aux_model=aux_model or "",
        gen_model_at_write=gen_at_write,
        substrate_extra=filtered_extra,
        substrate_src_sha=substrate_src_hash(substrate_module_path),
        cache_schema_version=getattr(memory_unit_cls, "CACHE_SCHEMA_VERSION", "v1"),
    )


# ---------------------------------------------------------------------------
# Index (append-only JSONL)
# ---------------------------------------------------------------------------

def _append_index(benchmark: str, record: Dict[str, Any]) -> None:
    """Append one JSON record to the index. Atomic enough on POSIX with
    O_APPEND; concurrent writers may interleave but won't corrupt lines."""
    p = index_path(benchmark)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    # Open with O_APPEND so two writers' lines don't overwrite each other.
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def cache_lookup(key: CacheKey) -> Optional[Path]:
    """Return path to cache payload dir if hit, else None."""
    h = key.hash()
    payload = benchmark_root(key.benchmark) / "by_hash" / h
    if not payload.exists():
        return None
    if not (payload / "VERSION").exists():
        return None
    # Schema mismatch double-check
    try:
        v = (payload / "VERSION").read_text().strip()
        if v != key.cache_schema_version:
            return None
    except OSError:
        return None
    # Record access (best-effort, append a 'hit' record)
    _append_index(key.benchmark, {
        "event": "hit",
        "hash": h,
        "ts": time.time(),
    })
    return payload


def cache_store(key: CacheKey, save_state_fn, write_metrics: Dict[str, Any]) -> Path:
    """Materialize a new cache entry.

    Args:
        key: cache key
        save_state_fn: callable(dir_path) -> None  — substrate.save_state
        write_metrics: dict to dump as write_metrics.json

    Returns:
        path to the new cache payload dir.
    """
    h = key.hash()
    payload = benchmark_root(key.benchmark) / "by_hash" / h
    payload.mkdir(parents=True, exist_ok=True)

    # 1. substrate state
    state_dir = payload / "memory_state"
    state_dir.mkdir(exist_ok=True)
    save_state_fn(str(state_dir))

    # 2. write metrics
    with open(payload / "write_metrics.json", "w") as f:
        json.dump(write_metrics, f, indent=2)

    # 3. version marker
    with open(payload / "VERSION", "w") as f:
        f.write(key.cache_schema_version)

    # 4. key dump (debug)
    with open(payload / "key.json", "w") as f:
        json.dump(key.to_dict(), f, indent=2, sort_keys=True)

    # 5. meta
    with open(payload / "meta.json", "w") as f:
        json.dump({
            "created_at": time.time(),
            "schema_version": key.cache_schema_version,
        }, f, indent=2)

    # 6. index entry
    _append_index(key.benchmark, {
        "event": "create",
        "hash": h,
        "substrate": key.substrate,
        "sample_id": key.sample_id,
        "encoder": key.encoder,
        "aux_model": key.aux_model,
        "gen_model_at_write": key.gen_model_at_write,
        "extra": key.substrate_extra,
        "substrate_src_sha": key.substrate_src_sha,
        "schema": key.cache_schema_version,
        "bytes": _dir_size_bytes(payload),
        "ts": time.time(),
    })

    return payload


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Dir-copy helpers (for substrates whose state is a directory: C5/C6/C9/C16)
# ---------------------------------------------------------------------------

def copy_tree(src: str, dst: str) -> None:
    """Copy a directory recursively. Used by substrates whose state is a
    DB dir (C5 mem0 SQLite, C6 Kuzu, C9 LoRA adapter, C16 ChromaDB)."""
    import shutil
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_tree_back(cache_dir: str, dst: str) -> None:
    """Restore a cached dir to dst."""
    copy_tree(cache_dir, dst)
