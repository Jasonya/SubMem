"""Query the SubMem cache index.

Usage:
    python scripts/cache_query.py list [--benchmark locomo] [--substrate C3-Letta]
    python scripts/cache_query.py show <hash>
    python scripts/cache_query.py stats
    python scripts/cache_query.py rm <hash>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Direct module load (avoid the metrics/ chain)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache.cache_io import cache_root, benchmark_root, index_path  # noqa: E402


def iter_index(benchmark: str):
    p = index_path(benchmark)
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collapse(records):
    """Reduce per-event index records to one snapshot per hash."""
    by_hash = {}
    for r in records:
        h = r.get("hash")
        if not h:
            continue
        if h not in by_hash:
            by_hash[h] = {
                "hash": h, "substrate": r.get("substrate"),
                "sample_id": r.get("sample_id"),
                "encoder": r.get("encoder"), "aux_model": r.get("aux_model"),
                "gen_model_at_write": r.get("gen_model_at_write"),
                "extra": r.get("extra", {}),
                "schema": r.get("schema"),
                "substrate_src_sha": r.get("substrate_src_sha"),
                "bytes": r.get("bytes", 0),
                "created_at": r.get("ts") if r.get("event") == "create" else None,
                "last_used_at": r.get("ts"),
                "uses": 0,
            }
        by_hash[h]["last_used_at"] = max(by_hash[h]["last_used_at"] or 0, r.get("ts", 0))
        if r.get("event") == "hit":
            by_hash[h]["uses"] += 1
        # Keep create-time fields if a create event exists
        for k in ("substrate", "sample_id", "encoder", "aux_model", "gen_model_at_write", "extra", "schema", "substrate_src_sha", "bytes"):
            if r.get(k) is not None and by_hash[h].get(k) in (None, "", {}):
                by_hash[h][k] = r[k]
    return list(by_hash.values())


def cmd_list(args):
    for bench in args.benchmarks:
        records = list(iter_index(bench))
        snapshot = collapse(records)
        if args.substrate:
            snapshot = [r for r in snapshot if r["substrate"] == args.substrate]
        if args.sample:
            snapshot = [r for r in snapshot if r["sample_id"] == args.sample]
        print(f"\n=== {bench} ({len(snapshot)} entries) ===")
        for r in sorted(snapshot, key=lambda x: (x["substrate"] or "", x["sample_id"] or "")):
            existed = (benchmark_root(bench) / "by_hash" / r["hash"]).exists()
            mark = "✓" if existed else "✗"
            print(f"  {mark} {r['hash']}  {r['substrate']:<14} {r['sample_id']:<10} "
                  f"enc={r['encoder']:<25} aux={r['aux_model']:<25} "
                  f"gen={r.get('gen_model_at_write','') or '-':<25} "
                  f"size={r['bytes'] / 1e6:.1f}MB  uses={r['uses']}")


def cmd_show(args):
    for bench in args.benchmarks:
        for r in collapse(list(iter_index(bench))):
            if r["hash"] == args.hash:
                payload = benchmark_root(bench) / "by_hash" / args.hash
                print(json.dumps(r, indent=2))
                if payload.exists():
                    print(f"\nPayload at: {payload}")
                    for f in sorted(payload.rglob("*")):
                        if f.is_file():
                            print(f"  {f.relative_to(payload)}  ({f.stat().st_size} bytes)")
                else:
                    print("(payload missing on disk)")
                return
    print(f"hash {args.hash} not found")
    sys.exit(1)


def cmd_stats(args):
    for bench in args.benchmarks:
        records = list(iter_index(bench))
        snapshot = collapse(records)
        total_bytes = sum(r["bytes"] for r in snapshot)
        by_substrate = defaultdict(lambda: [0, 0])  # [count, bytes]
        for r in snapshot:
            by_substrate[r["substrate"]][0] += 1
            by_substrate[r["substrate"]][1] += r["bytes"]
        print(f"\n=== {bench} ===")
        print(f"  total entries: {len(snapshot)}")
        print(f"  total size:    {total_bytes / 1e9:.2f} GB")
        for s, (n, b) in sorted(by_substrate.items()):
            print(f"    {s:<14} n={n:<4} {b / 1e6:.1f} MB")


def cmd_rm(args):
    for bench in args.benchmarks:
        payload = benchmark_root(bench) / "by_hash" / args.hash
        if payload.exists():
            shutil.rmtree(payload)
            print(f"removed {payload}")
            return
    print(f"hash {args.hash} not found")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=None,
                    help="restrict to these benchmark namespaces")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("list")
    p1.add_argument("--substrate")
    p1.add_argument("--sample")
    p1.set_defaults(func=cmd_list)

    p2 = sub.add_parser("show")
    p2.add_argument("hash")
    p2.set_defaults(func=cmd_show)

    p3 = sub.add_parser("stats")
    p3.set_defaults(func=cmd_stats)

    p4 = sub.add_parser("rm")
    p4.add_argument("hash")
    p4.set_defaults(func=cmd_rm)

    args = ap.parse_args()
    if args.benchmarks is None:
        args.benchmarks = [d.name for d in cache_root().iterdir() if d.is_dir()] if cache_root().exists() else []
    args.func(args)


if __name__ == "__main__":
    main()
