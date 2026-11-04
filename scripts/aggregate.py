"""Aggregate per-conv 09_raw_data.json files into combined summary.

Usage:
    python scripts/aggregate.py <raw_data.json> [<raw_data.json> ...]
    python scripts/aggregate.py --glob '$SUBMEM_RESULTS/*locomo*/M1/09_raw_data.json'
    python scripts/aggregate.py --run-dir <run_dir> [--substrate M1]

Output: combined.json + per_conv.csv + combined_table.md printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make SubMem importable when run as `python scripts/aggregate.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SubMem.config import to_modern_id, to_legacy_id  # noqa: E402


def _safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def aggregate(files: List[Path]) -> Dict[str, Any]:
    """Combine multiple per-conv 09_raw_data.json files."""
    runs = []
    for fp in files:
        try:
            runs.append(json.load(open(fp)))
        except Exception as e:
            print(f"  WARN: skip {fp}: {e}", file=sys.stderr)

    if not runs:
        return {}

    all_q = []
    for r in runs:
        all_q.extend(r.get("per_query", []))

    n_total = len(all_q)
    if n_total == 0:
        return {"n": 0, "warning": "no per-query data"}

    def m(key: str, predicate=lambda q: True) -> Optional[float]:
        vals = [q["per_query_metrics"].get(key) for q in all_q if predicate(q)]
        return _safe_mean(vals)

    # P-family: per-query mean with appropriate filters
    P = {
        "P1_em":      m("P1_em"),
        "P2_f1":      m("P2_f1"),
        "P3_bleu1":   m("P3_bleu1"),
        "P4_judge":   m("P4_judge"),
        "P5_faith":   m("P5_faith", predicate=lambda q: q.get("query_type") != "adversarial"),
    }

    # E latency: concat raw → percentile
    e2_vals = [(q.get("retrieval_latency_ms", 0) + q.get("generation_latency_ms", 0)) for q in all_q]
    e4_vals = [q.get("retrieval_latency_ms", 0) for q in all_q]
    E = {
        "E2_p50_ms": _percentile(e2_vals, 50),
        "E2_p90_ms": _percentile(e2_vals, 90),
        "E2_mean_ms": _safe_mean(e2_vals),
        "E4_p50_ms": _percentile(e4_vals, 50),
        "E4_p90_ms": _percentile(e4_vals, 90),
        "E4_mean_ms": _safe_mean(e4_vals),
    }

    # Counts: aggregate from per-conv 07_metrics.json next to 09_raw_data.json
    e_counters = defaultdict(int)
    e1_per_conv = []
    for fp in files:
        m07 = fp.parent / "07_metrics.json"
        if not m07.exists():
            continue
        try:
            metrics07 = json.load(open(m07))
        except Exception:
            continue
        eff = metrics07.get("efficiency", {}) or {}
        for key in ("E5_construct_tokens", "E6_construct_calls", "E7_retrieved_tokens",
                    "E8_retrieval_tokens", "E9_retrieval_calls", "E10_mgmt_tokens",
                    "E11_mgmt_calls", "E12_total_tokens", "E13_total_calls"):
            v = eff.get(key)
            if v is not None:
                e_counters[key] += int(v) if isinstance(v, (int, float)) else 0
        if eff.get("E1_storage_bytes") is not None:
            e1_per_conv.append(eff["E1_storage_bytes"])
        if eff.get("E14_total_time_sec") is not None:
            e_counters["E14_total_time_sec_sum"] += float(eff["E14_total_time_sec"])

    n_q_sum = sum(r.get("n_queries", 0) for r in runs)
    E.update(dict(e_counters))
    E["E1_storage_mean_bytes"] = _safe_mean(e1_per_conv) if e1_per_conv else None
    E["E1_storage_total_bytes"] = sum(e1_per_conv) if e1_per_conv else None
    E["E14_total_time_sec"] = e_counters.get("E14_total_time_sec_sum")
    e14_sum = e_counters.get("E14_total_time_sec_sum", 0.0) or 0.0
    E["E14_total_time_sec"] = e14_sum  # amortized sum
    E["E15_avg_time_per_query_sec"] = (e14_sum / max(n_q_sum, 1)) if n_q_sum else 0.0
    # Also report actual wall (sum of "this-run" times) for reference
    e14_actual_sum = 0.0
    for fp in files:
        m07 = fp.parent / "07_metrics.json"
        if m07.exists():
            try:
                v = json.load(open(m07)).get("efficiency", {}).get("E14_actual_wall_sec")
                if v is not None:
                    e14_actual_sum += float(v)
            except Exception:
                pass
    E["E14_actual_wall_sec"] = e14_actual_sum
    E["E15_actual_per_query_sec"] = (e14_actual_sum / max(n_q_sum, 1)) if n_q_sum else 0.0

    # Q / R / D from 07_metrics
    q_compr_list, q_div_list, r2_list, d1_list = [], [], [], []
    for fp in files:
        m07 = fp.parent / "07_metrics.json"
        if not m07.exists():
            continue
        try:
            metrics07 = json.load(open(m07))
        except Exception:
            continue
        q = metrics07.get("quality", {}) or {}
        r = metrics07.get("robustness", {}) or {}
        dy = metrics07.get("dynamics", {}) or {}
        if q.get("Q1_compression_ratio") is not None:
            q_compr_list.append(q["Q1_compression_ratio"])
        if q.get("Q2_coverage_diversity") is not None:
            q_div_list.append(q["Q2_coverage_diversity"])
        if r.get("R2_drr") is not None:
            r2_list.append(r["R2_drr"])
        if dy.get("D1_recency_bias") is not None:
            d1_list.append(dy["D1_recency_bias"])

    QRD = {
        "Q1_compression_mean": _safe_mean(q_compr_list),
        "Q2_diversity_mean": _safe_mean(q_div_list),
        "R2_drr_mean": _safe_mean(r2_list),
        "D1_recency_mean": _safe_mean(d1_list),
        "n_convs": len(runs),
    }

    return {
        "n_files": len(files),
        "n_queries_total": n_total,
        "n_adversarial": sum(1 for q in all_q if q.get("query_type") == "adversarial"),
        "n_non_adversarial": sum(1 for q in all_q if q.get("query_type") != "adversarial"),
        "P": P,
        "E": E,
        "QRD": QRD,
    }


def per_conv_table(files: List[Path]) -> List[Dict[str, Any]]:
    """One row per (conv) with its own metrics."""
    rows = []
    for fp in files:
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        per_q = r.get("per_query", [])

        def m(key, pred=lambda q: True):
            vals = [q["per_query_metrics"].get(key) for q in per_q if pred(q)]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        m07 = fp.parent / "07_metrics.json"
        eff = json.load(open(m07)).get("efficiency", {}) if m07.exists() else {}

        rows.append({
            "memory_id": to_modern_id(r.get("memory_id", "?")),
            "sample_id": str(fp.parent.parent.name),  # run dir name
            "n_queries": r.get("n_queries"),
            "P1_em":     m("P1_em"),
            "P2_f1":     m("P2_f1"),
            "P3_bleu1":  m("P3_bleu1"),
            "P4_judge":  m("P4_judge"),
            "P5_faith":  m("P5_faith", pred=lambda q: q.get("query_type") != "adversarial"),
            "E14_sec":   eff.get("E14_total_time_sec"),
            "E1_bytes":  eff.get("E1_storage_bytes"),
        })
    return rows


def emit_md(combined: Dict[str, Any]) -> str:
    p = combined.get("P", {})
    e = combined.get("E", {})
    qrd = combined.get("QRD", {})
    n = combined.get("n_queries_total", 0)
    nc = qrd.get("n_convs", 0)
    out = []
    out.append(f"## Combined ({nc} conv, {n} queries)\n")
    out.append("| Metric | Value |\n|---|---:|")
    for k, v in p.items():
        out.append(f"| {k} | {v:.4f} |" if v is not None else f"| {k} | – |")
    out.append("")
    out.append("| Latency (ms) | mean | p50 | p90 |\n|---|---:|---:|---:|")
    out.append(f"| E2 inference | {e.get('E2_mean_ms',0):.1f} | {e.get('E2_p50_ms',0):.1f} | {e.get('E2_p90_ms',0):.1f} |")
    out.append(f"| E4 retrieval | {e.get('E4_mean_ms',0):.1f} | {e.get('E4_p50_ms',0):.1f} | {e.get('E4_p90_ms',0):.1f} |")
    out.append("")
    out.append("| Cost | Value |\n|---|---:|")
    for k in ("E5_construct_tokens", "E6_construct_calls", "E7_retrieved_tokens",
              "E10_mgmt_tokens", "E11_mgmt_calls", "E12_total_tokens", "E13_total_calls",
              "E14_total_time_sec", "E15_avg_time_per_query_sec"):
        v = e.get(k)
        out.append(f"| {k} | {v:,.0f} |" if isinstance(v, (int, float)) else f"| {k} | – |")
    out.append("")
    out.append("| Q+R+D | mean across conv |\n|---|---:|")
    for k, v in qrd.items():
        if k == "n_convs": continue
        out.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="09_raw_data.json files")
    ap.add_argument("--glob", help="glob pattern for 09_raw_data.json files")
    ap.add_argument("--run-dir", help="run dir; aggregate all configs under it")
    ap.add_argument("--substrate", help="restrict to this substrate (e.g. M1)")
    ap.add_argument("--out", default=None, help="output dir for combined.json + per_conv.csv")
    args = ap.parse_args()

    files: List[Path] = []
    if args.glob:
        files += [Path(p) for p in glob.glob(args.glob)]
    if args.run_dir:
        if args.substrate:
            # Substrate dirs may be named with the modern M-id (new runs)
            # or the legacy C-id (old runs) — try both.
            subs = {args.substrate, to_modern_id(args.substrate),
                    to_legacy_id(args.substrate)}
        else:
            subs = {"*"}
        for sub in subs:
            pattern = f"{args.run_dir}/{sub}/09_raw_data.json"
            files += [Path(p) for p in glob.glob(pattern)]
    files += [Path(p) for p in args.paths]

    if not files:
        print("no input files", file=sys.stderr)
        sys.exit(1)

    files = sorted(set(files))
    print(f"# Aggregating {len(files)} files", file=sys.stderr)

    # Per-substrate group
    grouped = defaultdict(list)
    for f in files:
        try:
            # Canonicalise to M-id so old C-id JSONs group with new M-id ones.
            mid = to_modern_id(json.load(open(f)).get("memory_id", "?"))
        except Exception:
            mid = "?"
        grouped[mid].append(f)

    for mid, fs in sorted(grouped.items()):
        print(f"\n# === {mid} (n={len(fs)}) ===")
        combined = aggregate(fs)
        rows = per_conv_table(fs)

        # Output
        if args.out:
            outdir = Path(args.out) / mid
            outdir.mkdir(parents=True, exist_ok=True)
            json.dump(combined, open(outdir / "combined.json", "w"), indent=2, default=str)
            with open(outdir / "per_conv.csv", "w", newline="") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)
            print(f"  → {outdir}/combined.json, per_conv.csv")
        print(emit_md(combined))


if __name__ == "__main__":
    main()
