"""Offline P5 rescoring for M5 / formerly C18 (LightRAG).

Re-runs P5 (NLI faithfulness) using the *raw* underlying chunks for
each attributed dia_id, instead of LightRAG's paraphrased synthesis.
This mirrors how M3 (ReadAgent) exposes raw retrieved text and avoids
penalising LightRAG for lossy summary entailment.

Reuses the same dia_id attribution as rescore_c18_p6.py.

Usage:
    python scripts/rescore_c18_p5.py --glob 'results/*locomo*/[CM]*'  # M5 (new) or C18 (old) dirs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

# allow `from rescore_c18_p6 import ...`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rescore_c18_p6 import _attribute  # noqa: E402

# Bring in NLI scorer (use SubMem-as-package import to dodge relative-import
# issues triggered by metrics/__init__.py importing benchmarks/.)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_REPO_ROOT))
from SubMem.metrics.nli_scorer import (  # noqa: E402
    NLIScorer,
    compute_faithfulness_nli_score,
    decompose_claims_rule,
)


def _rescore_dir(rd: str, scorer: NLIScorer, top_k: int = 10) -> dict | None:
    ret_p = os.path.join(rd, "03_retrievals.json")
    pred_p = os.path.join(rd, "05_predictions.json")
    if not (os.path.isfile(ret_p) and os.path.isfile(pred_p)):
        return None

    rets = json.load(open(ret_p))
    preds = json.load(open(pred_p))

    # Per-sample chunk pool
    sid_to_chunks: dict[str, list[str]] = {}
    sid_to_dias: dict[str, list[str]] = {}
    for sf in sorted(os.listdir(rd)):
        if not (sf.startswith("02_writes_sample") and sf.endswith(".json")):
            continue
        try:
            ws = json.load(open(os.path.join(rd, sf)))
        except Exception:
            continue
        sid = ws.get("sample_id", "")
        chunks, dias = [], []
        for e in ws.get("entries", []):
            chunks.append(e.get("text", ""))
            dias.append((e.get("metadata") or {}).get("dia_id", ""))
        sid_to_chunks[sid] = chunks
        sid_to_dias[sid] = dias

    # Index retrievals by query_id
    qid_to_ret = {r.get("query_id"): r for r in rets}

    p5_scores = []
    for r in preds:
        qid = r.get("query_id")
        ret = qid_to_ret.get(qid)
        if ret is None:
            continue
        sid = ret.get("sample_id", "")
        chunks = sid_to_chunks.get(sid, [])
        dias = sid_to_dias.get(sid, [])
        retrieved = ret.get("retrieved", [])
        answer = retrieved[0].get("content", "") if retrieved else ""
        new_dias = _attribute(answer, chunks, dias, top_k=top_k)
        dia_to_chunk = {d: chunks[i] for i, d in enumerate(dias) if d}
        raw_mems = [dia_to_chunk[d] for d in new_dias if d in dia_to_chunk]
        if not raw_mems:
            continue

        claims = r.get("claims") or decompose_claims_rule(r["prediction"])
        if not claims:
            continue
        # Apply the same query-rebuild that the live pipeline uses for short
        # claims, so we don't artificially boost P5 here.
        if r.get("query"):
            rebuilt = []
            for c in claims:
                if len(c.split()) < 5:
                    rebuilt.append(f"{r['query'].rstrip('?.!').strip()}: {c}")
                else:
                    rebuilt.append(c)
            claims = rebuilt

        res = compute_faithfulness_nli_score(claims, raw_mems, scorer)
        p5_scores.append(res["p5_score"])

    if not p5_scores:
        return None
    return {
        "n": len(p5_scores),
        "P5_faithfulness": sum(p5_scores) / len(p5_scores),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    print("loading NLI scorer (deberta-v3-large-nli)...")
    scorer = NLIScorer()
    scorer.load()
    print(f"{'P5':>7} {'n':>5}  dir")
    for d in sorted(glob(args.glob)):
        out = _rescore_dir(d, scorer, top_k=args.top_k)
        if out is None:
            print(f"      -      -  {d}")
            continue
        print(f"  {out['P5_faithfulness']:.4f}  {out['n']:>4}  {d}")


if __name__ == "__main__":
    main()
