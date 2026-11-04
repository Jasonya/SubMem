"""Retrieval Quality metrics — P4, P5.

P6 Recall@k    — fraction of gold evidence passages found in top-k retrieved
P7 Precision@k — fraction of top-k retrieved that are gold evidence

Uses dia_id matching between query evidence annotations and retrieved entry metadata.
No LLM calls needed.
"""
from typing import Any, Dict, List


def compute_recall_at_k(gold_evidence_ids: List[str],
                        retrieved_ids: List[str]) -> float:
    """P4: Recall@k — what fraction of gold evidence was retrieved."""
    if not gold_evidence_ids:
        return 1.0  # no evidence needed
    gold = set(gold_evidence_ids)
    hits = gold & set(retrieved_ids)
    return len(hits) / len(gold)


def compute_precision_at_k(gold_evidence_ids: List[str],
                           retrieved_ids: List[str]) -> float:
    """P5: Precision@k — what fraction of retrieved is actually relevant."""
    if not retrieved_ids:
        return 0.0
    gold = set(gold_evidence_ids)
    hits = gold & set(retrieved_ids)
    return len(hits) / len(retrieved_ids)


class RetrievalQualityMetrics:
    """Compute P4/P5 from saved retrieval logs + query evidence annotations."""

    @staticmethod
    def compute(retrieval_logs: List[Dict[str, Any]],
                query_evidence_map: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute P6 Recall@k and P7 Precision@k across all queries.

        Args:
            retrieval_logs: list of dicts from 03_retrievals.json, each with
                "query_id" and "retrieved" (list of {metadata: {dia_id: ...}})
            query_evidence_map: {query_id: [evidence_id, ...]} from query data

        Returns:
            {"P6_recall_at_k": float, "P7_precision_at_k": float}
        """
        recall_scores = []
        precision_scores = []

        for rlog in retrieval_logs:
            qid = rlog["query_id"]
            gold_ids = query_evidence_map.get(qid, [])
            if not gold_ids:
                continue  # skip queries without evidence annotations

            retrieved_ids = []
            for entry in rlog.get("retrieved", []):
                meta = entry.get("metadata", {})
                # Single dia_id (C1/C2 per-utterance)
                dia_id = meta.get("dia_id", "")
                if dia_id:
                    retrieved_ids.append(dia_id)
                # Multiple dia_ids (C3/C4/C5 chunk-based)
                for did in meta.get("dia_ids", []):
                    if did and did not in retrieved_ids:
                        retrieved_ids.append(did)

            recall_scores.append(compute_recall_at_k(gold_ids, retrieved_ids))
            precision_scores.append(compute_precision_at_k(gold_ids, retrieved_ids))

        return {
            "P6_recall_at_k": float(sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0,
            "P7_precision_at_k": float(sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0,
            "num_queries_with_evidence": len(recall_scores),
        }
