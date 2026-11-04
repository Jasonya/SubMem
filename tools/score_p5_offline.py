"""Offline P5 (NLI faithfulness) scorer.

Reads existing 05_predictions.json files, runs NLI-based P5 on each,
and writes 09_faithfulness.json alongside. Does not touch any other
pipeline stage; safe to run anytime after a benchmark run completes.

Usage:
    python3 score_p5_offline.py <run_dir>       # e.g., results/20260419_..._locomo_Qwen3-8B
    python3 score_p5_offline.py --all           # all locomo runs under $SUBMEM_RESULTS
    python3 score_p5_offline.py --latest-per-config  # only most recent per config
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

# Direct module load to avoid metrics/__init__.py chain that pulls
# benchmarks/* and triggers relative-import errors when run as a script.
import importlib.util as _ilu
_here = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("nli_scorer_mod", os.path.join(_here, "metrics", "nli_scorer.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NLIScorer = _mod.NLIScorer
compute_faithfulness_nli_score = _mod.compute_faithfulness_nli_score
decompose_claims_rule = _mod.decompose_claims_rule


def score_run(config_dir: Path, nli_scorer: NLIScorer) -> dict | None:
    """Score one <run_dir>/<config>/ directory. Returns summary or None if skipped."""
    pred_file = config_dir / "05_predictions.json"
    if not pred_file.exists():
        return None

    preds = json.load(open(pred_file))
    if not isinstance(preds, list):
        return None

    p5_scores = []
    p5_details = []
    for r in preds:
        claims = r.get("claims")
        sources = r.get("retrieved_contents") or []
        if not claims:
            claims = decompose_claims_rule(r.get("prediction", ""))
        res = compute_faithfulness_nli_score(claims, sources, nli_scorer)
        p5_scores.append(res["p5_score"])
        p5_details.append({
            "query_id": r.get("query_id"),
            **res,
        })

    p5_mean = float(np.mean(p5_scores)) if p5_scores else 0.0
    out = {
        "P5_faithfulness": round(p5_mean, 4),
        "num_queries": len(p5_scores),
        "per_query": p5_details,
    }
    json.dump(out, open(config_dir / "09_faithfulness.json", "w"), indent=2)

    # Also patch 07_metrics.json in-place if present
    m_file = config_dir / "07_metrics.json"
    if m_file.exists():
        m = json.load(open(m_file))
        m.setdefault("task_performance", {})["P5_faithfulness"] = round(p5_mean, 4)
        json.dump(m, open(m_file, "w"), indent=2)

    return {"config": config_dir.name, "P5": round(p5_mean, 4), "n": len(p5_scores)}


def iter_config_dirs(roots: list[Path]):
    for root in roots:
        for pred in root.rglob("05_predictions.json"):
            yield pred.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="run directory or config directory")
    ap.add_argument("--all", action="store_true",
                    help="process every locomo run under $SUBMEM_RESULTS")
    ap.add_argument("--latest-per-config", action="store_true",
                    help="only most recent run per (benchmark, config)")
    args = ap.parse_args()

    results_root = Path(os.environ.get("SUBMEM_RESULTS",
                                       os.path.join(os.path.dirname(__file__), "results")))

    if args.all or args.latest_per_config:
        roots = list(results_root.glob("*locomo_*")) + list(results_root.glob("c5_mem0_*"))
    elif args.path:
        p = Path(args.path)
        roots = [p] if p.is_dir() else []
    else:
        print("specify a path, --all, or --latest-per-config", file=sys.stderr)
        sys.exit(2)

    targets = list(iter_config_dirs(roots))
    if args.latest_per_config:
        # Group by (benchmark, config) — for SubMem layout, benchmark = run_dir.name
        # and config = config_dir.name. Keep the newest run_dir.name per config.
        latest = {}
        for cd in targets:
            bench = cd.parent.name  # e.g. 20260419_..._locomo_Qwen3-8B
            cfg = cd.name  # substrate dir name: M1, C3-Letta (retired), etc.
            if cfg not in latest or bench > latest[cfg][0]:
                latest[cfg] = (bench, cd)
        targets = [v[1] for v in latest.values()]

    if not targets:
        print("no 05_predictions.json found under", roots, file=sys.stderr)
        sys.exit(1)

    print(f"loading NLI model...")
    scorer = NLIScorer()
    scorer.load()
    print(f"scoring {len(targets)} config dirs...")

    for cd in sorted(targets):
        try:
            res = score_run(cd, scorer)
            if res:
                print(f"  {cd.parent.name}/{res['config']}: P5={res['P5']:.4f} ({res['n']} q)")
        except Exception as e:
            print(f"  {cd}: ERROR {e}")


if __name__ == "__main__":
    main()
