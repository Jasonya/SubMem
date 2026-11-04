"""P4: LLM-as-a-Judge — Binary correctness scoring (Mem0/Zep/A-Mem convention).

Sends (query, gold_answer, prediction) to an LLM judge which labels
the answer as CORRECT or WRONG. Returns accuracy (fraction of CORRECT).

Prompt follows Mem0's ACCURACY_PROMPT template used by Mem0, Zep/Graphiti,
and A-Mem for LoCoMo evaluation.
"""
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from ..benchmarks.base import PredictionResult

_JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


# ── LongMemEval per-type judge prompts (following original eval code) ───

_LME_JUDGE_STANDARD = """\
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {gold_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only."""

_LME_JUDGE_TEMPORAL = """\
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {gold_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only."""

_LME_JUDGE_KNOWLEDGE_UPDATE = """\
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {gold_answer}

Model Response: {generated_answer}

Is the model response correct? Answer yes or no only."""

_LME_JUDGE_ABSTENTION = """\
I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {gold_answer}

Model Response: {generated_answer}

Does the model correctly identify the question as unanswerable? Answer yes or no only."""

_LME_JUDGE_MAP = {
    "single_session": _LME_JUDGE_STANDARD,
    "multi_session_reasoning": _LME_JUDGE_STANDARD,
    "temporal_reasoning": _LME_JUDGE_TEMPORAL,
    "knowledge_update": _LME_JUDGE_KNOWLEDGE_UPDATE,
    "abstention": _LME_JUDGE_ABSTENTION,
}


def _parse_label(response: str) -> bool:
    """Extract CORRECT/WRONG from judge response. Returns True if CORRECT."""
    text = response.strip()
    # Try JSON parse first
    try:
        obj = json.loads(text)
        label = obj.get("label", "").strip().upper()
        return label == "CORRECT"
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: search for CORRECT/WRONG in text
    text_upper = text.upper()
    if "CORRECT" in text_upper and "WRONG" not in text_upper:
        return True
    if "WRONG" in text_upper and "CORRECT" not in text_upper:
        return False
    # Ambiguous — default to WRONG
    return False


def _parse_yesno(response: str) -> bool:
    """Extract yes/no from LongMemEval judge response. Returns True if yes."""
    text = response.strip().lower()
    # Check last word first (judge often ends with just "yes" or "no")
    last_word = text.split()[-1].rstrip(".") if text.split() else ""
    if last_word == "yes":
        return True
    if last_word == "no":
        return False
    # Fallback: search in full text
    if "yes" in text and "no" not in text:
        return True
    if "no" in text and "yes" not in text:
        return False
    return False


class LLMJudge:
    """LLM-as-a-Judge scorer using OpenAI-compatible API (Mem0 convention)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 base_url: str = None, max_retries: int = 3,
                 rate_limit_delay: float = 0.1, benchmark: str = "locomo"):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", None)
        self._max_retries = max_retries
        self._rate_limit_delay = rate_limit_delay
        self._benchmark = benchmark
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )

    def score_one(self, question: str, gold_answer: str, generated_answer: str,
                  query_type: str = "") -> bool:
        """Score a single prediction. Returns True if CORRECT, False if WRONG."""
        self._ensure_client()

        if self._benchmark == "longmemeval":
            # LongMemEval: per-type prompts, yes/no output
            tmpl = _LME_JUDGE_MAP.get(query_type, _LME_JUDGE_STANDARD)
            prompt = tmpl.format(
                question=question,
                gold_answer=gold_answer,
                generated_answer=generated_answer,
            )
            kwargs = dict(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            parser = _parse_yesno
        else:
            # Default (LoCoMo / Mem0): CORRECT/WRONG with JSON output
            prompt = _JUDGE_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                generated_answer=generated_answer,
            )
            kwargs = dict(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parser = _parse_label

        # Reasoning models (gpt-5*, o1/o3/o4*) reject `temperature` and
        # `max_tokens`; they use `max_completion_tokens` and need headroom
        # for hidden reasoning tokens. Guarded so the default gpt-4o-mini /
        # gpt-4.1-* judges are untouched.
        _mn = (self._model or "").lower()
        if _mn.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs.pop("temperature", None)
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = 4000

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content
                return parser(text)
            except Exception as e:
                if attempt < self._max_retries - 1:
                    time.sleep(self._rate_limit_delay * (attempt + 1))
                else:
                    print(f"  [P4 Judge] Failed after {self._max_retries} retries: {e}")
                    return False

    def score_batch(self, predictions: List[PredictionResult],
                    raw_preds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Score all predictions. Returns accuracy (fraction CORRECT) and per-query details."""
        scores = []
        details = []
        for i, pred in enumerate(predictions):
            is_correct = self.score_one(pred.query, pred.gold_answer, pred.prediction,
                                        query_type=pred.query_type)
            scores.append(1.0 if is_correct else 0.0)
            details.append({
                "query_id": pred.query_id,
                "label": "CORRECT" if is_correct else "WRONG",
                "score": 1.0 if is_correct else 0.0,
            })
            if (i + 1) % 50 == 0:
                acc = sum(scores) / len(scores)
                print(f"  [P4 Judge] {i+1}/{len(predictions)} scored, "
                      f"running accuracy={acc:.4f}")

        accuracy = sum(scores) / len(scores) if scores else 0.0
        return {
            "P4_llm_judge": round(accuracy, 4),
            "num_queries": len(scores),
            "num_correct": int(sum(scores)),
            "per_query": details,
        }


def compute_llm_judge_score(predictions: List[PredictionResult],
                            judge: LLMJudge) -> Dict[str, Any]:
    """Convenience function to compute P4 from predictions."""
    return judge.score_batch(predictions)
