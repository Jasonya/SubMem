"""LoCoMo benchmark — multi-session recall.

Official source: snap-research/locomo (GitHub)
File: locomo10.json — 10 conversations, ~2000 queries total
Schema per item: sample_id, conversation (dict with sessions), qa (list),
                 event_summary, observation, session_summary
QA categories: 1=factual, 2=temporal, 3=inferential, 4=comprehension
Primary metrics: F1, EM
"""
import json
import os
import re
from typing import Any, Dict, List

from .base import BaseBenchmark, BenchmarkSample, PredictionResult
from ..config import get_dataset_path

CATEGORY_MAP = {1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_domain"}


class LoComoBenchmark(BaseBenchmark):

    QUERY_TYPES = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]

    @property
    def name(self) -> str:
        return "locomo"

    @property
    def primary_metric(self) -> str:
        return "f1"

    def get_query_types(self) -> List[str]:
        return self.QUERY_TYPES

    def load_data(self) -> List[BenchmarkSample]:
        data_path = get_dataset_path("locomo")
        json_file = os.path.join(data_path, "locomo10.json")
        with open(json_file, "r") as f:
            raw = json.load(f)

        samples = []
        for item in raw:
            conv = item["conversation"]
            sessions = self._parse_conversation(conv)

            queries = []
            for i, qa in enumerate(item["qa"]):
                cat = qa.get("category", 1)
                qtype = CATEGORY_MAP.get(cat, "single_hop")
                # Category 5 = adversarial (no ground-truth answer, has adversarial_answer)
                if cat == 5:
                    qtype = "adversarial"
                    answer = qa.get("answer", "")  # often empty for adversarial
                else:
                    answer = qa.get("answer", "")
                query_entry = {
                    "id": f"{item['sample_id']}_q{i}",
                    "question": qa["question"],
                    "answer": answer,
                    "type": qtype,
                    "evidence": qa.get("evidence", []),
                }
                # Preserve adversarial_answer for R2 DRR metric
                if "adversarial_answer" in qa:
                    query_entry["adversarial_answer"] = qa["adversarial_answer"]
                queries.append(query_entry)

            samples.append(BenchmarkSample(
                sample_id=str(item["sample_id"]),
                sessions=sessions,
                queries=queries,
                metadata={
                    "speaker_a": conv.get("speaker_a", ""),
                    "speaker_b": conv.get("speaker_b", ""),
                    "num_questions": len(queries),
                },
            ))

            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        self._data = samples
        return samples

    @staticmethod
    def _parse_conversation(conv: Dict) -> List[Dict]:
        """Parse official LoCoMo conversation dict into sessions list."""
        # Find all session keys: session_1, session_2, ...
        session_nums = sorted(set(
            int(m.group(1))
            for k in conv.keys()
            if (m := re.match(r"session_(\d+)(?:_date_time)?$", k))
        ))

        sessions = []
        for num in session_nums:
            session_key = f"session_{num}"
            date_key = f"session_{num}_date_time"
            if session_key not in conv:
                continue

            turns = []
            for utterance in conv[session_key]:
                turns.append({
                    "content": f"{utterance['speaker']}: {utterance['text']}",
                    "metadata": {
                        "speaker": utterance.get("speaker", ""),
                        "dia_id": utterance.get("dia_id", ""),
                    },
                })

            sessions.append({
                "session_id": session_key,
                "turns": turns,
                "date": conv.get(date_key, ""),
            })

        return sessions

    def build_prompt(self, query: Dict[str, Any], retrieved_context: str) -> str:
        return (
            "You are an intelligent memory assistant tasked with retrieving accurate "
            "information from conversation memories.\n\n"
            "# CONTEXT:\n"
            "You have access to memories from speakers in a conversation. These memories "
            "contain timestamped information that may be relevant to answering the question.\n\n"
            "# INSTRUCTIONS:\n"
            "1. Carefully analyze all provided memories\n"
            "2. Pay special attention to the timestamps to determine the answer\n"
            "3. If the question asks about a specific event or fact, look for direct "
            "evidence in the memories\n"
            "4. If the memories contain contradictory information, prioritize the most "
            "recent memory\n"
            "5. If there is a question about time references (like 'last year', "
            "'two months ago', etc.), calculate the actual date based on the memory "
            "timestamp. For example, if a memory from 4 May 2022 mentions 'went to "
            "India last year,' then the trip occurred in 2021. If a memory from "
            "25 May 2023 says 'last Saturday', compute the exact date (e.g. 20 May "
            "2023). NEVER output relative phrases like 'last year' or 'next month' "
            "— always output the concrete date, month, or year.\n"
            "6. Always convert relative time references to specific dates, months, or "
            "years. Ignore the reference while answering the question.\n"
            "7. Focus only on the content of the memories. Do not confuse character "
            "names mentioned in memories with the actual users.\n"
            "8. The answer should be less than 5-6 words.\n"
            "9. IMPORTANT: The question may contain a false premise or refer to an "
            "event that never happened. If the memories contain NO evidence that the "
            "event or fact in the question actually occurred, you MUST answer with an "
            "empty string (i.e., output nothing). Do NOT guess or hallucinate an "
            "answer. Only answer when there is clear supporting evidence.\n\n"
            "# APPROACH (Think step by step):\n"
            "1. Examine all memories that contain information related to the question\n"
            "2. Examine the timestamps and content carefully\n"
            "3. Look for explicit mentions of dates, times, locations, or events\n"
            "4. If the answer requires calculation (e.g., converting relative time "
            "references), show your work and output the final concrete date\n"
            "5. Formulate a precise, concise answer based solely on the evidence\n"
            "6. Double-check that your answer directly addresses the question\n"
            "7. Ensure your final answer is specific and avoids vague time references\n"
            "8. If the question assumes something that is NOT supported by the "
            "memories, output an empty answer\n\n"
            "Memories:\n\n"
            f"{retrieved_context}\n\n"
            f"Question: {query['question']}\n\n"
            "Answer:"
        )

    def evaluate(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        from ..metrics.task_performance import compute_em, compute_f1

        results = {}
        type_results = {t: {"em": [], "f1": []} for t in self.QUERY_TYPES}
        em_scores, f1_scores = [], []

        for pred in predictions:
            em = compute_em(pred.prediction, pred.gold_answer)
            f1 = compute_f1(pred.prediction, pred.gold_answer)
            em_scores.append(em)
            f1_scores.append(f1)
            if pred.query_type in type_results:
                type_results[pred.query_type]["em"].append(em)
                type_results[pred.query_type]["f1"].append(f1)

        results["em"] = sum(em_scores) / len(em_scores) if em_scores else 0.0
        results["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        for qtype, scores in type_results.items():
            if scores["em"]:
                results[f"em_{qtype}"] = sum(scores["em"]) / len(scores["em"])
                results[f"f1_{qtype}"] = sum(scores["f1"]) / len(scores["f1"])

        return results
