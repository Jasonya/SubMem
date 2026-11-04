"""Family R: Robustness (2 metrics) — R1-R2.  [LOCAL]

R1 Contradiction Resolution Accuracy (CRA)
R2 Distractor Rejection Rate (DRR)
"""
from typing import Any, Dict, List

from ..benchmarks.base import PredictionResult
from .task_performance import _normalize_answer


class RobustnessMetrics:

    @staticmethod
    def compute_r1(predictions: List[PredictionResult],
                   contradiction_queries: List[Dict]) -> float:
        """R1: CRA — fraction of contradiction queries resolved correctly.

        Args:
            predictions: model predictions
            contradiction_queries: list of {query_id, correct_answer (latest version)}
        """
        if not contradiction_queries:
            return 0.0
        correct_map = {q["query_id"]: q["correct_answer"] for q in contradiction_queries}
        correct = 0
        total = 0
        for pred in predictions:
            if pred.query_id in correct_map:
                total += 1
                pred_norm = _normalize_answer(pred.prediction)
                gold_norm = _normalize_answer(correct_map[pred.query_id])
                if gold_norm in pred_norm or pred_norm == gold_norm:
                    correct += 1
        return correct / total if total > 0 else 0.0

    @staticmethod
    def compute_r2(predictions: List[PredictionResult],
                   distractor_queries: List[Dict]) -> float:
        """R2: DRR — fraction of distractor queries where agent rejected the distractor.

        Args:
            distractor_queries: list of {query_id, distractor_answer}
                Optionally includes correct_answer. If correct_answer is missing,
                rejection is measured as NOT matching the distractor.
        """
        if not distractor_queries:
            return 0.0
        query_map = {q["query_id"]: q for q in distractor_queries}
        correct = 0
        total = 0
        for pred in predictions:
            if pred.query_id in query_map:
                total += 1
                q = query_map[pred.query_id]
                pred_norm = _normalize_answer(pred.prediction)
                distractor_norm = _normalize_answer(q["distractor_answer"])

                if "correct_answer" in q and q["correct_answer"]:
                    # Both correct and distractor available — compare overlap
                    correct_norm = _normalize_answer(q["correct_answer"])
                    correct_overlap = len(set(pred_norm.split()) & set(correct_norm.split()))
                    distractor_overlap = len(set(pred_norm.split()) & set(distractor_norm.split()))
                    if correct_overlap > distractor_overlap:
                        correct += 1
                else:
                    # Only distractor available (LoCoMo adversarial: no ground-truth)
                    # Rejection = prediction does NOT match the distractor answer
                    distractor_words = set(distractor_norm.split())
                    pred_words = set(pred_norm.split())
                    if not distractor_words:
                        correct += 1
                        continue
                    overlap = len(pred_words & distractor_words) / len(distractor_words)
                    # If less than 50% of distractor words appear, consider it rejected
                    if overlap < 0.5:
                        correct += 1

        return correct / total if total > 0 else 0.0

    @staticmethod
    def extract_distractor_queries(raw_preds: List[Dict]) -> List[Dict]:
        """Extract adversarial queries with distractor answers from prediction logs.

        Works with LoCoMo format where adversarial queries store
        adversarial_answer in query metadata.
        """
        distractor_queries = []
        for r in raw_preds:
            if r.get("query_type") != "adversarial":
                continue
            adv_answer = r.get("adversarial_answer", "")
            if not adv_answer:
                continue
            entry = {
                "query_id": r["query_id"],
                "distractor_answer": adv_answer,
            }
            # If a ground-truth answer exists, include it
            gold = r.get("gold_answer", "")
            if gold:
                entry["correct_answer"] = gold
            distractor_queries.append(entry)
        return distractor_queries

    @staticmethod
    def compute(predictions=None, contradiction_queries=None,
                distractor_queries=None) -> Dict[str, float]:
        results = {}
        if predictions and contradiction_queries:
            results["R1_cra"] = RobustnessMetrics.compute_r1(predictions, contradiction_queries)
        if predictions and distractor_queries:
            results["R2_drr"] = RobustnessMetrics.compute_r2(predictions, distractor_queries)
        return results
