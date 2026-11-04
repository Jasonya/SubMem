"""Family P: Task Performance (7 metrics) — P1-P7.  [LOCAL]

P1 Exact Match (EM)                   — task_performance.py
P2 F1 Score                           — task_performance.py
P3 BLEU-1                             — task_performance.py
P4 LLM-as-a-Judge                     — llm_judge.py
P5 Faithfulness (NLI)                 — nli_scorer.py
P6 Recall@k                           — retrieval_quality.py
P7 Precision@k                        — retrieval_quality.py
"""
import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional

from ..benchmarks.base import PredictionResult


# ── Utility functions ──

def _normalize_answer(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation/articles/whitespace."""
    text = str(text).lower()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = " ".join(text.split())
    return text


def compute_em(prediction: str, gold: str) -> float:
    """P1: Exact Match."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(gold) else 0.0


def compute_f1(prediction: str, gold: str) -> float:
    """P2: Token-level F1 (set-based, following Mem0/A-Mem convention)."""
    pred_tokens = set(_normalize_answer(prediction).split())
    gold_tokens = set(_normalize_answer(gold).split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu1(prediction: str, gold: str) -> float:
    """P3: BLEU-1 — unigram precision with brevity penalty."""
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens:
        return 0.0
    if not gold_tokens:
        return 0.0
    # Clipped unigram counts
    gold_counts = Counter(gold_tokens)
    clipped = 0
    pred_counts = Counter(pred_tokens)
    for token, count in pred_counts.items():
        clipped += min(count, gold_counts.get(token, 0))
    p1 = clipped / len(pred_tokens)
    # Brevity penalty
    bp = math.exp(1 - len(gold_tokens) / len(pred_tokens)) if len(pred_tokens) < len(gold_tokens) else 1.0
    return bp * p1


def compute_sub_em(prediction: str, gold: str, sub_answers: Optional[List[str]] = None) -> float:
    """SubEM for MemoryAgentBench: fraction of sub-answers found in prediction."""
    if not sub_answers:
        return compute_em(prediction, gold)
    pred_norm = _normalize_answer(prediction)
    hits = sum(1 for sa in sub_answers if _normalize_answer(sa) in pred_norm)
    return hits / len(sub_answers)


class TaskPerformanceMetrics:
    """Compute all Family 1 metrics from predictions."""

    @staticmethod
    def compute(predictions: List[PredictionResult]) -> Dict[str, float]:
        em_scores = []
        f1_scores = []
        bleu1_scores = []
        for pred in predictions:
            em_scores.append(compute_em(pred.prediction, pred.gold_answer))
            f1_scores.append(compute_f1(pred.prediction, pred.gold_answer))
            bleu1_scores.append(compute_bleu1(pred.prediction, pred.gold_answer))
        return {
            "P1_em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
            "P2_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "P3_bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
        }
