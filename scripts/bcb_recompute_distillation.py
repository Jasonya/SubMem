"""Re-compute code-benchmark-specific distillation metrics for BCB runs.

Adds `code_bench_distillation` to 07_metrics.json without touching LoCoMo's
Q1. Two numbers:

  distillation_ratio (DR) = sum(stored_tokens) / sum(trajectory_tokens)
    DR < 1 = compression, DR > 1 = expansion.
    Both numerator and denominator use the same tokenizer (model.encode when
    available, else whitespace-split) so the ratio is in consistent units.
    This is the coding-bench analogue of Q1 — see research note in Phase 8.

  entry_fanout  = |entries| / |experiences|
    Disambiguates "few big entries" (Voyager-style) from "many small facts"
    (Mem0-style) at equal DR.

These only make sense on coding/trajectory benchmarks. Do NOT use for LoCoMo.

Usage:
  python scripts/bcb_recompute_distillation.py
    [--experiences $SUBMEM_RESULTS/experiences_bcb.parquet]
    [--runs-glob  '$SUBMEM_RESULTS/*_bcb_*/*/07_metrics.json']
"""
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

# Repo root: $SUBMEM_ROOT, else derived from this file's location (scripts/..).
SUBMEM_ROOT = Path(os.environ.get("SUBMEM_ROOT", Path(__file__).resolve().parents[1]))
SUBMEM_RESULTS = Path(os.environ.get("SUBMEM_RESULTS", SUBMEM_ROOT / "results"))


_WRAPPER_PREFIXES = [
    # PageSum header "<Page N>"
    re.compile(r"^<Page\s+\d+>\s*\n?"),
    # RAPTOR level tag "[L1] ..."
    re.compile(r"^\[L\d+\]\s*"),
    # C13 Sleep "Pre-computed reasoning:" + "### Rethink Memory Block" etc.
    re.compile(r"^Pre-computed reasoning:\s*\n?"),
    re.compile(r"^###\s+Rethink Memory Block\s*\n?"),
    # Markdown code fences
    re.compile(r"^```(?:python|diff|bash)?\s*\n?"),
    re.compile(r"\n?```\s*$"),
]


def strip_wrappers(text: str) -> str:
    out = text
    for pat in _WRAPPER_PREFIXES:
        out = pat.sub("", out)
    # C14 ReasoningBank "# Memory Item / ## Title / ## Description / ## Content"
    # Keep payload (all content), just drop the structural markers for fair compare
    out = re.sub(r"^#\s*Memory Item\s*\n?", "", out, flags=re.M)
    out = re.sub(r"^##\s+(Title|Description|Content)\s*", "", out, flags=re.M)
    return out.strip()


def _tokenizer():
    """Return a callable tokenizer(text) -> list[tokens]. Falls back to whitespace."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: enc.encode(s, disallowed_special=())
    except Exception:
        return lambda s: s.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiences",
        default=str(SUBMEM_RESULTS / "experiences_bcb.parquet"),
    )
    ap.add_argument(
        "--runs-glob",
        default=str(SUBMEM_RESULTS / "*_bcb_*/*/07_metrics.json"),
    )
    args = ap.parse_args()

    tok = _tokenizer()

    # Experience-side: total trajectory tokens (shared across all runs)
    exp = pd.read_parquet(args.experiences)
    n_exp = len(exp)
    total_traj_tokens = sum(len(tok(t)) for t in exp["trajectory_text"])
    print(f"Experiences: {n_exp} trajectories, {total_traj_tokens} tokens")
    print()

    # Per-run: compute stored payload tokens from 03_retrievals.json + populate data.
    # 03_retrievals.json contains retrieved entries; we don't have get_all_entries()
    # here since the memory unit is gone. Instead approximate stored_size via
    # efficiency.E1_storage_bytes * some density, OR better: re-read entries
    # from memory_unit_stats if present. Simplest accurate path: use the
    # buffer/entry count and average stored-content length visible in retrievals.

    import glob
    from collections import Counter

    rows = []
    for mpath in sorted(glob.glob(args.runs_glob)):
        p = Path(mpath)
        rd = p.parent
        cfg = rd.name
        m = json.loads(p.read_text())
        # Pull all stored entries visible in 03_retrievals.json.
        retr_path = rd / "03_retrievals.json"
        if not retr_path.exists():
            continue
        retrievals = json.loads(retr_path.read_text())
        # Collect unique entry contents across all retrievals (content may repeat
        # across queries; dedupe by exact string)
        seen_contents = {}
        for r in retrievals:
            for e in r.get("retrieved", []):
                c = e.get("content", "")
                if c and c not in seen_contents:
                    seen_contents[c] = True
        n_unique_retrieved = len(seen_contents)
        # Approximate stored_tokens: average payload tokens per retrieved entry
        # times the actual memory unit size (memory_unit_stats.size or efficiency
        # proxy). memory_unit.size() is in summary as
        # num_memory_items_after_distill.
        unit_size = m.get("num_memory_items_after_distill", 0)
        if unit_size == 0 or n_unique_retrieved == 0:
            rows.append({"cfg": cfg, "run": rd.name, "dr": None, "fanout": None,
                         "reason": "no entries observed"})
            continue
        avg_payload_tokens = (
            sum(len(tok(strip_wrappers(c))) for c in seen_contents) / n_unique_retrieved
        )
        stored_tokens = int(avg_payload_tokens * unit_size)
        dr = stored_tokens / total_traj_tokens if total_traj_tokens else None
        fanout = unit_size / n_exp

        code_bench = {
            "distillation_ratio": round(dr, 4) if dr is not None else None,
            "entry_fanout": round(fanout, 4),
            "n_memory_entries": unit_size,
            "n_experiences": n_exp,
            "avg_payload_tokens_per_entry": round(avg_payload_tokens, 1),
            "total_stored_tokens_estimate": stored_tokens,
            "total_trajectory_tokens": total_traj_tokens,
            "notes": "payload = content with markdown wrappers stripped; stored "
                     "extrapolated from retrieval sample",
        }
        # Save back into metrics
        m["code_bench_distillation"] = code_bench
        p.write_text(json.dumps(m, indent=2))
        rows.append({"cfg": cfg, "run": rd.parent.name, "dr": code_bench["distillation_ratio"],
                     "fanout": code_bench["entry_fanout"]})

    # Print table (latest run per cfg)
    latest = {}
    for r in rows:
        latest[r["cfg"]] = r  # sorted glob => last wins
    print(f"{'Cfg':<5} {'DR':>7} {'Fanout':>8}")
    print("-" * 23)
    for cfg in ["C1","C2","C3","C4","C5","C6","C7","C13","C14","C15","C16"]:
        r = latest.get(cfg)
        if not r or r["dr"] is None:
            print(f"{cfg:<5} {'-':>7} {'-':>8}")
        else:
            print(f"{cfg:<5} {r['dr']:>7.2f} {r['fanout']:>8.2f}")


if __name__ == "__main__":
    main()
