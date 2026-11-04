"""Family D: Behavioral Dynamics (1 metric) — D1.  [LOCAL]

D1 Recency Bias — EM_recent_10% - EM_early_10% (by evidence session position)
"""
from typing import Dict, List, Tuple

from ..benchmarks.base import PredictionResult
from .task_performance import _normalize_answer


class BehavioralDynamicsMetrics:

    @staticmethod
    def compute_d1(predictions_with_memory: List[PredictionResult],
                   predictions_without_memory: List[PredictionResult],
                   bert_score_model=None) -> float:
        """D1: MUR = 1 - BERTScore(output_with, output_without).

        If bert_score_model is None, uses token overlap as proxy.
        """
        if not predictions_with_memory:
            return 0.0

        similarities = []
        for pw, pwo in zip(predictions_with_memory, predictions_without_memory):
            sim = _token_similarity(pw.prediction, pwo.prediction)
            similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities)
        return 1.0 - avg_sim

    @staticmethod
    def compute_d2(predictions: List[PredictionResult],
                   parametric_probes: List[Dict[str, str]]) -> float:
        """D2: OSR — when memory contradicts parametric knowledge, how often memory wins.

        Args:
            parametric_probes: list of {
                "query_id": ...,
                "memory_answer": ...,      # what's in memory (contradicts parametric)
                "parametric_answer": ...,   # what LLM would say without memory
            }
        """
        if not parametric_probes:
            return 0.0
        probe_map = {p["query_id"]: p for p in parametric_probes}
        memory_wins = 0
        total = 0
        for pred in predictions:
            if pred.query_id in probe_map:
                total += 1
                probe = probe_map[pred.query_id]
                pred_norm = _normalize_answer(pred.prediction)
                mem_norm = _normalize_answer(probe["memory_answer"])
                param_norm = _normalize_answer(probe["parametric_answer"])
                mem_overlap = len(set(pred_norm.split()) & set(mem_norm.split()))
                param_overlap = len(set(pred_norm.split()) & set(param_norm.split()))
                if mem_overlap > param_overlap:
                    memory_wins += 1
        return memory_wins / total if total > 0 else 0.0

    @staticmethod
    def compute_d3(early_perf: float, recent_perf: float) -> float:
        """D3: PRB = Perf_recent_10% - Perf_early_10%.

        Positive = recency bias, negative = primacy bias.
        """
        return recent_perf - early_perf

    @staticmethod
    def compute_d3_from_predictions(predictions: List[PredictionResult],
                                    position_key: str = "position") -> float:
        """D3 computed from predictions with position metadata.

        Splits facts by position into early 10% and recent 10%.
        Uses F1 (token overlap) instead of EM for scoring, since EM is
        too strict for memory systems that produce verbose answers.
        """
        if not predictions:
            return 0.0

        # Sort by position
        sorted_preds = sorted(predictions, key=lambda p: p.metadata.get(position_key, 0))
        n = len(sorted_preds)
        cutoff = max(1, n // 10)

        early = sorted_preds[:cutoff]
        recent = sorted_preds[-cutoff:]

        early_perf = _mean_f1(early) if early else 0.0
        recent_perf = _mean_f1(recent) if recent else 0.0

        return recent_perf - early_perf

    @staticmethod
    def compute(predictions_with=None, predictions_without=None,
                parametric_probes=None, early_perf=None, recent_perf=None) -> Dict[str, float]:
        results = {}
        if predictions_with and predictions_without:
            results["D1_mur"] = BehavioralDynamicsMetrics.compute_d1(
                predictions_with, predictions_without
            )
        if predictions_with and parametric_probes:
            results["D2_osr"] = BehavioralDynamicsMetrics.compute_d2(
                predictions_with, parametric_probes
            )
        if early_perf is not None and recent_perf is not None:
            results["D1_recency_bias"] = BehavioralDynamicsMetrics.compute_d3(early_perf, recent_perf)
        return results


def _mean_f1(predictions: List[PredictionResult]) -> float:
    """Mean token F1 over a set of predictions."""
    scores = []
    for p in predictions:
        pred_tokens = set(_normalize_answer(p.prediction).split())
        gold_tokens = set(_normalize_answer(p.gold_answer).split())
        if not gold_tokens:
            scores.append(1.0 if not pred_tokens else 0.0)
            continue
        if not pred_tokens:
            scores.append(0.0)
            continue
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def _token_similarity(text_a: str, text_b: str) -> float:
    """Simple token-level Jaccard similarity as BERTScore proxy."""
    tokens_a = set(_normalize_answer(text_a).split())
    tokens_b = set(_normalize_answer(text_b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
