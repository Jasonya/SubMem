"""Offline post-hoc scorer for MAB runs: P5 (NLI faithfulness), P6 (Recall@k),
P7 (Precision@k).

Usage:
    python -m SubMem.score_mab_p567 \
        path/to/seed_42_predictions.json \
        [--sub_dataset longmemeval_s*] \
        [--skip_nli]  # disable P5 (if DeBERTa-v3-NLI model not available)

Writes sidecar ``seed_<N>_p567.json`` next to the predictions file with:

    {
      "P5_faithfulness":  float,  # fraction of atomic claims entailed
      "P6_recall_at_k":   float,  # |gold_retrieved| / |gold|
      "P7_precision_at_k":float,  # |gold_retrieved| / |retrieved|
      "num_claims_total": int,
      "num_entailed":     int,
      "num_queries_with_evidence": int,
      "per_query": [ ... ]
    }

Gold-evidence model for MAB LongMemEval-S*
------------------------------------------
MAB's ``haystack_sessions`` is laid out as
``haystack_sessions[q_idx][sub_idx][turn_idx]``. Each turn carries a
``has_answer: bool`` flag naming it as evidence for question ``q_idx`` (not for
other questions in the same sample). Our benchmark loader flattens this into
sessions tagged ``sess_<q_idx>_<sub_idx>`` and propagates ``has_answer`` into
each turn's metadata, which ``benchmarks/base.py`` copies into the predictions
dump under ``retrieved_metadata[i]``.

Recall@k / Precision@k for query ``q_idx`` is therefore:
    hits = [e for e in retrieved if e.session_id startswith f"sess_{q_idx}_"
                                    and e.has_answer]
    gold = count of has_answer turns in haystack_sessions[q_idx]
"""
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import get_dataset_path


# ── Gold-evidence bookkeeping from MAB parquet ─────────────────────────


def load_gold_evidence(sub_dataset: str) -> Dict[str, Dict[str, Any]]:
    """Build a per-query map of evidence positions.

    Returns ``{qid: {"q_idx": int, "gold_count": int, "sample_idx": int}}``
    where ``q_idx`` is the position of this question inside its parent
    sample's ``haystack_sessions`` array and ``gold_count`` is the number of
    ``has_answer=True`` turns across that per-question haystack.
    """
    base = os.path.join(get_dataset_path("memory_agent_bench"), "data")
    df = pd.read_parquet(os.path.join(base, "Accurate_Retrieval-00000-of-00001.parquet"))
    rows = df[df["metadata"].apply(lambda m: m.get("source") == sub_dataset)]
    qid_to_gold: Dict[str, Dict[str, Any]] = {}
    for sample_idx, (_, row) in enumerate(rows.iterrows()):
        metadata = row["metadata"]
        qids = list(metadata.get("qa_pair_ids", []))
        hs = metadata.get("haystack_sessions")
        if hs is None:
            continue
        for q_idx, qid in enumerate(qids):
            if q_idx >= len(hs):
                continue
            session_group = hs[q_idx]
            gold = 0
            for sub in session_group:
                for turn in sub:
                    has = turn.get("has_answer", False) if hasattr(turn, "get") else False
                    if has:
                        gold += 1
            qid_to_gold[qid] = {
                "q_idx": q_idx,
                "gold_count": int(gold),
                "sample_idx": sample_idx,
            }
    return qid_to_gold


# ── P6 / P7 ──────────────────────────────────────────────────────────────

_SESS_RE = re.compile(r"sess_(\d+)_")


def _retrieved_hits_for_query(retrieved_metadata: List[Dict[str, Any]],
                              q_idx: int) -> int:
    """Count retrieved entries that belong to this query's haystack AND
    are marked as evidence (``has_answer=True``)."""
    hits = 0
    for m in retrieved_metadata or []:
        sid = (m or {}).get("session_id", "") or ""
        mo = _SESS_RE.match(sid)
        if mo is None:
            continue
        if int(mo.group(1)) != q_idx:
            continue
        if (m or {}).get("has_answer"):
            hits += 1
    return hits


def compute_p6_p7(predictions: List[Dict[str, Any]],
                  qid_to_gold: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    recalls: List[float] = []
    precisions: List[float] = []
    per_query: List[Dict[str, Any]] = []
    num_covered = 0
    num_uncovered = 0
    missing_metadata = 0

    for p in predictions:
        qid = p.get("query_id", "")
        meta = p.get("metadata", {}) or {}
        retrieved_meta = meta.get("retrieved_metadata")
        retrieved_chunks = meta.get("retrieved_chunks") or []
        if retrieved_meta is None:
            missing_metadata += 1
            continue

        gold_info = qid_to_gold.get(qid)
        if gold_info is None:
            continue  # not a MAB LongMemEval qid we know

        q_idx = gold_info["q_idx"]
        gold_count = gold_info["gold_count"]
        retrieved_count = len(retrieved_chunks)
        hits = _retrieved_hits_for_query(retrieved_meta, q_idx)

        if gold_count == 0:
            recall = 1.0  # nothing to recall
            num_uncovered += 1
        else:
            recall = hits / gold_count
            num_covered += 1

        precision = hits / retrieved_count if retrieved_count else 0.0

        recalls.append(recall)
        precisions.append(precision)
        per_query.append({
            "query_id": qid,
            "q_idx": q_idx,
            "gold_count": gold_count,
            "retrieved": retrieved_count,
            "hits": hits,
            "recall_at_k": round(recall, 4),
            "precision_at_k": round(precision, 4),
        })

    return {
        "P6_recall_at_k": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "P7_precision_at_k": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "num_queries_scored": len(recalls),
        "num_queries_with_evidence": num_covered,
        "num_queries_without_evidence": num_uncovered,
        "num_queries_missing_retrieved_metadata": missing_metadata,
        "per_query": per_query,
    }


# ── P5 ────────────────────────────────────────────────────────────────────


def compute_p5(predictions: List[Dict[str, Any]],
               scorer,
               decomposition: str = "rule") -> Dict[str, Any]:
    """Compute P5 NLI faithfulness over every prediction.

    For each prediction, decompose the answer into atomic claims, then run
    NLI with (premise = retrieved_context, hypothesis = claim). Faithfulness
    is the fraction of claims marked entailment.

    ``decomposition``:
        "rule"  — sentence + conjunction splitting (fast, no LLM call)
        "llm"   — requires a model.generate() callable; not implemented here
                  (would defeat the "offline re-score" purpose).
    """
    from .metrics.nli_scorer import (
        decompose_claims_rule,
        compute_faithfulness_nli_score,
    )

    if decomposition != "rule":
        raise NotImplementedError("Only rule-based decomposition is supported in offline scoring.")

    per_query: List[Dict[str, Any]] = []
    total_claims = 0
    total_entailed = 0

    for p in predictions:
        pred_text = p.get("prediction", "") or ""
        retrieved_chunks = (p.get("metadata", {}) or {}).get("retrieved_chunks") or []
        claims = decompose_claims_rule(pred_text)
        res = compute_faithfulness_nli_score(claims, retrieved_chunks, scorer)
        per_query.append({
            "query_id": p.get("query_id", ""),
            "p5_score": res["p5_score"],
            "num_claims": res["num_claims"],
            "num_entailed": res.get("num_entailed", 0),
        })
        total_claims += res["num_claims"]
        total_entailed += res.get("num_entailed", 0)

    # P5 is the per-prediction average of entailment fraction,
    # NOT the pooled rate — predictions with many claims would otherwise
    # dominate. Report both for transparency.
    scores = [pq["p5_score"] for pq in per_query]
    p5_mean_per_query = float(sum(scores) / len(scores)) if scores else 0.0
    p5_pooled = (total_entailed / total_claims) if total_claims else 0.0

    return {
        "P5_faithfulness": round(p5_mean_per_query, 4),
        "P5_pooled_entailment_rate": round(p5_pooled, 4),
        "num_claims_total": total_claims,
        "num_entailed_total": total_entailed,
        "num_queries": len(per_query),
        "per_query": per_query,
    }


# ── Entry point ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc P5/P6/P7 scoring for MAB runs."
    )
    parser.add_argument("predictions_path", help="Path to seed_<N>_predictions.json")
    parser.add_argument("--sub_dataset", default="longmemeval_s*",
                        help="MAB sub_dataset filter (default: longmemeval_s*)")
    parser.add_argument("--skip_nli", action="store_true",
                        help="Skip P5 (useful if DeBERTa model not downloaded)")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base",
                        help="HuggingFace model id for NLI")
    parser.add_argument("--nli_device", default=None,
                        help="cuda|cpu (default: auto)")
    args = parser.parse_args()

    predictions_path = args.predictions_path
    if not os.path.exists(predictions_path):
        print(f"ERROR: {predictions_path} does not exist", file=sys.stderr)
        sys.exit(2)

    with open(predictions_path) as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {predictions_path}", file=sys.stderr)

    # Gold-evidence map from parquet
    qid_to_gold = load_gold_evidence(args.sub_dataset)
    print(f"Gold-evidence map has {len(qid_to_gold)} qids for sub_dataset={args.sub_dataset}",
          file=sys.stderr)

    # P6 / P7
    p67 = compute_p6_p7(predictions, qid_to_gold)
    print(f"P6 recall@k   = {p67['P6_recall_at_k']:.4f}")
    print(f"P7 precision@k= {p67['P7_precision_at_k']:.4f}")
    print(f"  scored={p67['num_queries_scored']} "
          f"evidence-covered={p67['num_queries_with_evidence']} "
          f"missing-retrieved-meta={p67['num_queries_missing_retrieved_metadata']}")

    out: Dict[str, Any] = {
        "P6_recall_at_k": p67["P6_recall_at_k"],
        "P7_precision_at_k": p67["P7_precision_at_k"],
        "P6_P7_diagnostics": {
            "num_queries_scored": p67["num_queries_scored"],
            "num_queries_with_evidence": p67["num_queries_with_evidence"],
            "num_queries_without_evidence": p67["num_queries_without_evidence"],
            "num_queries_missing_retrieved_metadata": p67["num_queries_missing_retrieved_metadata"],
        },
        "per_query_p6_p7": p67["per_query"],
    }

    # P5
    if args.skip_nli:
        print("P5 skipped (--skip_nli)")
    else:
        from .metrics.nli_scorer import NLIScorer
        print(f"Loading NLI model {args.nli_model} on {args.nli_device or 'auto'} ...",
              file=sys.stderr)
        scorer = NLIScorer(model_name=args.nli_model, device=args.nli_device)
        scorer.load()
        p5 = compute_p5(predictions, scorer)
        print(f"P5 faithfulness     = {p5['P5_faithfulness']:.4f}  (mean-per-query)")
        print(f"P5 pooled entail    = {p5['P5_pooled_entailment_rate']:.4f}  "
              f"({p5['num_entailed_total']}/{p5['num_claims_total']} claims entailed)")
        out.update({
            "P5_faithfulness": p5["P5_faithfulness"],
            "P5_pooled_entailment_rate": p5["P5_pooled_entailment_rate"],
            "P5_diagnostics": {
                "num_claims_total": p5["num_claims_total"],
                "num_entailed_total": p5["num_entailed_total"],
                "num_queries": p5["num_queries"],
            },
            "per_query_p5": p5["per_query"],
        })

    out_path = predictions_path.replace("_predictions.json", "_p567.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
