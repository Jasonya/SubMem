"""Offline LLM-as-judge scoring for saved predictions.

Usage:
    python -m SubMem.score_llm_judge path/to/seed_42_predictions.json \
        [--judge_model gpt-4o-mini] [--sub_dataset longmemeval_s*]

Reconstructs LongMemEval per-question types from the MAB parquet metadata,
then calls the LLM judge with per-type prompts. Writes a sidecar
`seed_<N>_llm_judge.json` next to the predictions file.
"""
import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

from .benchmarks.base import PredictionResult
from .config import get_dataset_path
from .metrics.llm_judge import LLMJudge


def build_qid_to_type_map(sub_dataset: str) -> dict:
    """Map qa_pair_id -> LongMemEval question_type (underscored form).

    Scans every MAB capability parquet so this works for AR (longmemeval_s*),
    SF (factconsolidation_*), TTL, and LRU alike. Sub-datasets without
    per-question types return empty → caller falls back to standard prompt.
    """
    base = os.path.join(get_dataset_path("memory_agent_bench"), "data")
    qid_to_type = {}
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(base, fname))
        rows = df[df["metadata"].apply(lambda m: m.get("source") == sub_dataset)]
        for _, row in rows.iterrows():
            qids = list(row["metadata"]["qa_pair_ids"])
            qt_raw = row["metadata"].get("question_types")
            qtypes = list(qt_raw) if qt_raw is not None else []
            for i, qid in enumerate(qids):
                qt = qtypes[i] if i < len(qtypes) else ""
                qid_to_type[qid] = (qt or "").replace("-", "_")
    return qid_to_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_file")
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--sub_dataset", default="longmemeval_s*")
    args = ap.parse_args()

    with open(args.predictions_file) as f:
        raw_preds = json.load(f)

    qid_to_type = build_qid_to_type_map(args.sub_dataset)
    missing = sum(1 for p in raw_preds if p["query_id"] not in qid_to_type)
    if missing:
        print(f"[warn] {missing}/{len(raw_preds)} predictions lack question_type; falling back to standard prompt.", file=sys.stderr)

    preds = [
        PredictionResult(
            query_id=p["query_id"],
            query=p["query"],
            prediction=p["prediction"],
            gold_answer=p["gold_answer"],
            query_type=qid_to_type.get(p["query_id"], "single_session"),
            metadata=p.get("metadata", {}),
        )
        for p in raw_preds
    ]

    print(f"Scoring {len(preds)} predictions with {args.judge_model} (LongMemEval protocol)...")
    print(f"  query_type distribution: {Counter(p.query_type for p in preds)}")
    judge = LLMJudge(model=args.judge_model, benchmark="longmemeval")
    result = judge.score_batch(preds)

    # Per-type breakdown.
    by_type = {}
    for p, d in zip(preds, result["per_query"]):
        by_type.setdefault(p.query_type, []).append(d["score"])
    type_accuracy = {
        t: {"accuracy": sum(s) / len(s), "n": len(s)}
        for t, s in by_type.items()
    }

    result["per_type_accuracy"] = type_accuracy
    out_path = args.predictions_file.replace("_predictions.json", "_llm_judge.json")
    if out_path == args.predictions_file:
        out_path = args.predictions_file + ".llm_judge.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== P4 LLM-Judge Accuracy: {result['P4_llm_judge']:.4f} "
          f"({result['num_correct']}/{result['num_queries']}) ===")
    print("Per-type:")
    for t, d in sorted(type_accuracy.items()):
        print(f"  {t:<30s} {d['accuracy']:.4f}  (n={d['n']})")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
