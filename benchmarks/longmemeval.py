"""LongMemEval benchmark — knowledge update, abstention, multi-session reasoning.

Official source: xiaowu0162/longmemeval-cleaned (HuggingFace)
Files: longmemeval_s_cleaned.json (~40 sessions, ~115K tokens),
       longmemeval_m_cleaned.json (~500 sessions),
       longmemeval_oracle.json (evidence sessions only)
500 questions, 5 question types
Primary metric: Accuracy
"""
import json
import os
from typing import Any, Dict, List

from .base import BaseBenchmark, BenchmarkSample, PredictionResult
from ..config import get_dataset_path

QTYPE_MAP = {
    "single-session-user": "single_session",
    "single-session-assistant": "single_session",
    "multi-session": "multi_session_reasoning",
    "knowledge-update": "knowledge_update",
    "temporal-reasoning": "temporal_reasoning",
    "abstention": "abstention",
}

VARIANT_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


class LongMemEvalBenchmark(BaseBenchmark):

    QUERY_TYPES = [
        "single_session",
        "knowledge_update",
        "abstention",
        "multi_session_reasoning",
        "temporal_reasoning",
    ]

    DEFAULT_VARIANT = "s"

    @property
    def name(self) -> str:
        return "longmemeval"

    @property
    def primary_metric(self) -> str:
        return "accuracy"

    def get_query_types(self) -> List[str]:
        return self.QUERY_TYPES

    def load_data(self) -> List[BenchmarkSample]:
        data_path = get_dataset_path("longmemeval")
        variant = self.DEFAULT_VARIANT
        if hasattr(self.config, 'extra') and isinstance(self.config.extra, dict):
            variant = self.config.extra.get("variant", variant)
        filename = VARIANT_FILES.get(variant, VARIANT_FILES["s"])
        json_file = os.path.join(data_path, filename)

        with open(json_file, "r") as f:
            raw = json.load(f)

        samples = []
        for item in raw:
            sessions = []
            haystack_sessions = item.get("haystack_sessions", [])
            haystack_ids = item.get("haystack_session_ids", [])
            for idx, sess_turns in enumerate(haystack_sessions):
                sess_id = haystack_ids[idx] if idx < len(haystack_ids) else f"sess_{idx}"
                turns = []
                for turn in sess_turns:
                    turns.append({
                        "content": turn.get("content", ""),
                        "metadata": {"role": turn.get("role", "")},
                    })
                sessions.append({"session_id": sess_id, "turns": turns})

            raw_qtype = item.get("question_type", "")
            qtype = QTYPE_MAP.get(raw_qtype, raw_qtype)

            queries = [{
                "id": item["question_id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": qtype,
                "question_date": item.get("question_date", ""),
            }]

            samples.append(BenchmarkSample(
                sample_id=item["question_id"],
                sessions=sessions,
                queries=queries,
                metadata={
                    "answer_session_ids": item.get("answer_session_ids", []),
                    "question_date": item.get("question_date", ""),
                },
            ))

            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        self._data = samples
        return samples

    def build_prompt(self, query: Dict[str, Any], retrieved_context: str) -> str:
        question_date = query.get("question_date", "")
        date_line = f"Current Date: {question_date}\n" if question_date else ""
        return (
            "I will give you several history chats between you and a user. "
            "Please answer the question based on the relevant chat history. "
            "Answer the question step by step: first extract all the relevant "
            "information, and then reason over the information to get the answer.\n\n\n"
            "History Chats:\n\n"
            f"{retrieved_context}\n\n"
            f"{date_line}"
            f"Question: {query['question']}\n"
            "Answer (step by step):"
        )

    def evaluate(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        results = {}
        type_results = {t: [] for t in self.QUERY_TYPES}
        all_correct = []

        for pred in predictions:
            correct = self._check_accuracy(pred)
            all_correct.append(correct)
            if pred.query_type in type_results:
                type_results[pred.query_type].append(correct)

        results["accuracy"] = sum(all_correct) / len(all_correct) if all_correct else 0.0

        for qtype, scores in type_results.items():
            if scores:
                results[f"accuracy_{qtype}"] = sum(scores) / len(scores)

        return results

    @staticmethod
    def _check_accuracy(pred: PredictionResult) -> float:
        pred_norm = _normalize(pred.prediction)
        gold_norm = _normalize(pred.gold_answer)
        if pred_norm == gold_norm:
            return 1.0
        if gold_norm in pred_norm:
            return 1.0
        return 0.0


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(".")
