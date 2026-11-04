"""MemoryAgentBench — accurate retrieval, test-time learning, long-range understanding, conflict resolution.

Dataset: ai-hyz/MemoryAgentBench (parquet per task type)
Schema: context, questions (array), answers (array of arrays), metadata
Task types from filenames: Accurate_Retrieval, Conflict_Resolution,
                           Long_Range_Understanding, Test_Time_Learning
Primary metrics: SubEM, Accuracy, GPT4-F1
"""
import os
from typing import Any, Dict, List

import pandas as pd

from .base import BaseBenchmark, BenchmarkSample, PredictionResult
from ..config import get_dataset_path

TASK_FILES = {
    "accurate_retrieval": "Accurate_Retrieval-00000-of-00001.parquet",
    "conflict_resolution": "Conflict_Resolution-00000-of-00001.parquet",
    "long_range_understanding": "Long_Range_Understanding-00000-of-00001.parquet",
    "test_time_learning": "Test_Time_Learning-00000-of-00001.parquet",
}


class MemoryAgentBenchBenchmark(BaseBenchmark):

    QUERY_TYPES = list(TASK_FILES.keys())

    @property
    def name(self) -> str:
        return "memory_agent_bench"

    @property
    def primary_metric(self) -> str:
        return "sub_em"

    def get_query_types(self) -> List[str]:
        return self.QUERY_TYPES

    def load_data(self) -> List[BenchmarkSample]:
        data_path = get_dataset_path("memory_agent_bench")
        data_dir = os.path.join(data_path, "data")

        extra = getattr(self.config, "extra", {}) or {}
        task_filter = extra.get("task_type")
        sub_filter = extra.get("sub_dataset")
        max_q_per = extra.get("max_queries_per_sample")

        if task_filter and task_filter not in TASK_FILES:
            raise ValueError(
                f"Unknown task_type '{task_filter}'. Choices: {list(TASK_FILES)}"
            )

        samples = []
        for task_type, filename in TASK_FILES.items():
            if task_filter and task_type != task_filter:
                continue
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                continue
            df = pd.read_parquet(filepath)

            if sub_filter:
                df = df[df["metadata"].apply(
                    lambda m: m.get("source") == sub_filter
                )].reset_index(drop=True)
                if len(df) == 0:
                    continue

            for row_idx, row in df.iterrows():
                context = row["context"]
                metadata = row.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}

                # Prefer haystack_sessions (LongMemEval-style turn/session structure)
                # over the monolithic `context` field. Otherwise a single write()
                # call carries a 360K-token blob, collapsing to one vector and
                # defeating retrieval. haystack_sessions is an ndarray of per-
                # question session groups; flatten across questions and emit one
                # session per group so each turn becomes a separate memory entry.
                hs = metadata.get("haystack_sessions")
                sessions = []
                if hs is not None and len(hs) > 0:
                    for group_idx, session_group in enumerate(hs):
                        for sub_idx, turns in enumerate(session_group):
                            turn_list = []
                            for t_idx, turn in enumerate(turns):
                                role = turn.get("role", "") if hasattr(turn, "get") else ""
                                content_text = turn.get("content", "") if hasattr(turn, "get") else str(turn)
                                # Mem0 (C5) expects a `speaker` key on dialog turns;
                                # map LongMemEval's user/assistant roles to mem0's
                                # speaker_a/speaker_b convention.
                                speaker = "user" if role == "user" else ("assistant" if role == "assistant" else role or "unknown")
                                turn_list.append({
                                    "content": f"{role}: {content_text}" if role else content_text,
                                    "metadata": {
                                        "role": role,
                                        "speaker": speaker,
                                        "has_answer": bool(turn.get("has_answer", False)) if hasattr(turn, "get") else False,
                                    },
                                })
                            sessions.append({
                                "session_id": f"sess_{group_idx}_{sub_idx}",
                                "turns": turn_list,
                            })
                else:
                    # No haystack_sessions: MAB provides full context as a
                    # single blob. Two splits depending on task:
                    #   * test_time_learning (banking77 / clinic / TREC ICL):
                    #     split by `\n\nlabel: NN\n\n` boundary so each turn
                    #     is one (input, label) pair — substrates like C9
                    #     LoRA / C15 MemSkill need the boundary to do their
                    #     real work (SFT pairs, skill abstractions w/ labels).
                    #   * else (factconsolidation_*): 500-word chunks.
                    # Default 500-word chunking for all single-blob datasets
                    # (factconsolidation, ICL banking77 etc.). Earlier ICL had
                    # an example-level split, but the table needs Qwen3-8B and
                    # Qwen32B/Gemma to share the same loader for fair compare,
                    # so we standardise on 500-word chunks here.
                    _words = str(context).split()
                    # Sweepable via --chunk_words (chunk-size sensitivity
                    # sweep); default keeps the paper's 500-word setting.
                    _chunk_size = int(extra.get("chunk_words", 500))
                    if len(_words) <= _chunk_size:
                        _chunks = [context]
                    else:
                        _chunks = [" ".join(_words[i:i + _chunk_size])
                                   for i in range(0, len(_words), _chunk_size)]
                    turn_list = [
                        {
                            "content": c,
                            "metadata": {"role": "user", "speaker": "user",
                                          "has_answer": False},
                        }
                        for c in _chunks
                    ]
                    sessions = [{"session_id": "ctx_0", "turns": turn_list}]

                questions = row["questions"]
                answers = row["answers"]
                if max_q_per is not None:
                    questions = questions[:max_q_per]
                    answers = answers[:max_q_per]

                qa_pair_ids = []
                if metadata and "qa_pair_ids" in metadata:
                    qa_pair_ids = list(metadata["qa_pair_ids"])

                queries = []
                for q_idx, question in enumerate(questions):
                    # answers[q_idx] is an array of acceptable answers
                    ans_list = list(answers[q_idx]) if q_idx < len(answers) else []
                    gold = ans_list[0] if ans_list else ""
                    qid = qa_pair_ids[q_idx] if q_idx < len(qa_pair_ids) else f"{task_type}_{row_idx}_q{q_idx}"

                    queries.append({
                        "id": qid,
                        "question": question,
                        "answer": gold,
                        "type": task_type,
                        "sub_answers": ans_list,
                    })

                samples.append(BenchmarkSample(
                    sample_id=f"{task_type}_{row_idx}",
                    sessions=sessions,
                    queries=queries,
                    metadata={"task_type": task_type},
                ))

            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        self._data = samples
        return samples

    def build_prompt(self, query: Dict[str, Any], retrieved_context: str) -> str:
        """Per sub-dataset prompts copied from MemoryAgentBench
        ``utils/templates.py`` (Hu et al., 2025), `rag_agent` variant. Keeping
        the upstream wording verbatim ensures our F1/BLEU scores are
        apples-to-apples with the official MAB leaderboard, which uses these
        exact templates to constrain the model to short, gold-aligned answers.
        """
        extra = getattr(self.config, "extra", {}) or {}
        task_type = extra.get("task_type", "")
        sub_ds = extra.get("sub_dataset", "") or ""
        question = query["question"]

        # ---- TTL: ICL classification (banking77 / clinic / TREC) ----
        # MAB template name: 'in_context_learning'
        if task_type == "test_time_learning" or sub_ds.startswith("icl_"):
            return (
                f"{retrieved_context}\n\n"
                "Use the provided mapping from the context to numerical label "
                "to assign a numerical label to the context. Only output "
                "\"label: {label}\" and nothing else.\n\n"
                f"Question:{question} \n\n label:"
            )

        # ---- LME-S: longmemeval (accurate_retrieval etc.) ----
        # MAB template name: 'longmemeval'
        if "longmemeval" in sub_ds or task_type == "accurate_retrieval":
            return (
                f"{retrieved_context}\n\n"
                "The history chats are between you and a user. Based on the "
                "relevant chat history, answer the question as concisely as "
                "you can, using a single phrase if possible.\n\n "
                f"{question} \n\n Answer:"
            )

        # ---- LRU: detective_qa ----
        # MAB template name: 'detective_qa'
        if "detective" in sub_ds or task_type == "long_range_understanding":
            return (
                f"{retrieved_context}\n\n"
                "Based on the context you memorized, answer the question "
                "below. You are required to answer the question based on the "
                "strict output format.\n\n "
                f"{question} \n\n"
            )

        # ---- CR: factconsolidation ----
        # MAB template name: 'factconsolidation'
        if "factconsolidation" in sub_ds or task_type == "conflict_resolution":
            return (
                f"{retrieved_context}\n\n"
                "Pretend you are a knowledge management system. Each fact in "
                "the knowledge pool is provided with a serial number at the "
                "beginning, and the newer fact has larger serial number. \n "
                "You need to solve the conflicts of facts in the knowledge "
                "pool by finding the newest fact with larger serial number. "
                "You need to answer a question based on this rule. You "
                "should give a very concise answer without saying other "
                "words for the question **only** from the knowledge pool you "
                "have memorized rather than the real facts in real world. "
                "\n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based "
                "on the provided Knowledge Pool, what is the name of the "
                "current president of Russia? \nAnswer: Donald Trump \n\n "
                "Now Answer the Question: Based on the provided Knowledge "
                f"Pool, {question} \nAnswer:"
            )

        # ---- Fallback (other MAB sub-datasets / generic QA) ----
        # Ruler-QA / event-QA / infbench-sum etc. — concise default that
        # matches MAB's `long_context_agent`/`rag_agent` "give answer only"
        # convention.
        return (
            f"{retrieved_context}\n\n"
            "Answer the question based on the memorized documents. Only give "
            "me the answer and do not output any other words.\n\n "
            f"Now Answer the Question: {question}"
        )

    def evaluate(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        from ..metrics.task_performance import compute_em, compute_f1, compute_sub_em

        results = {}
        type_results = {t: {"sub_em": [], "acc": [], "f1": []} for t in self.QUERY_TYPES}
        all_sub_em, all_acc, all_f1 = [], [], []

        for pred in predictions:
            sub_answers = pred.metadata.get("sub_answers", [])
            sub_em = compute_sub_em(pred.prediction, pred.gold_answer, sub_answers)
            em = compute_em(pred.prediction, pred.gold_answer)
            f1 = compute_f1(pred.prediction, pred.gold_answer)

            all_sub_em.append(sub_em)
            all_acc.append(em)
            all_f1.append(f1)

            if pred.query_type in type_results:
                type_results[pred.query_type]["sub_em"].append(sub_em)
                type_results[pred.query_type]["acc"].append(em)
                type_results[pred.query_type]["f1"].append(f1)

        results["sub_em"] = sum(all_sub_em) / len(all_sub_em) if all_sub_em else 0.0
        results["accuracy"] = sum(all_acc) / len(all_acc) if all_acc else 0.0
        results["f1"] = sum(all_f1) / len(all_f1) if all_f1 else 0.0

        for qtype, scores in type_results.items():
            if scores["sub_em"]:
                results[f"sub_em_{qtype}"] = sum(scores["sub_em"]) / len(scores["sub_em"])
                results[f"accuracy_{qtype}"] = sum(scores["acc"]) / len(scores["acc"])
                results[f"f1_{qtype}"] = sum(scores["f1"]) / len(scores["f1"])

        return results
