"""ALFWorld — agentic next-action prediction (Plan A, offline).

Data: dslighfdsl/alfworld_train_all trajectories, preprocessed by
scripts/preprocess_alfworld.py into:
    <submem_cache>/alfworld/eval.json     — 312 held-out tasks
    <submem_cache>/alfworld/demo_pool.json — 2807 demo tasks

Per eval sample:
  - sessions: K demo trajectories (balanced same-type + random) written as
    past experiences. Each trajectory is a session whose turns are
    alternating observation / action lines.
  - query: task goal + initial observation
  - answer: gold next action (first step of expert trajectory)

Primary metric: first-action exact match (after normalization).
"""
import json
import os
import random
from typing import Any, Dict, List

from .base import BaseBenchmark, BenchmarkSample, PredictionResult
from ..config import get_dataset_path


def _normalize_action(a: str) -> str:
    return " ".join(a.strip().lower().rstrip(".").split())


class ALFWorldBenchmark(BaseBenchmark):

    QUERY_TYPES = [
        "pick_and_place_simple",
        "pick_two_obj_and_place",
        "pick_clean_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "look_at_obj_in_light",
    ]

    DEFAULT_DEMOS_PER_SAMPLE = 50
    SAME_TYPE_RATIO = 0.2  # 10/50 demos share the task_type

    @property
    def name(self) -> str:
        return "alfworld"

    @property
    def primary_metric(self) -> str:
        return "first_action_em"

    def get_query_types(self) -> List[str]:
        return self.QUERY_TYPES

    def load_data(self) -> List[BenchmarkSample]:
        data_path = get_dataset_path("alfworld")
        cleaned = os.path.join(data_path, "cleaned")
        eval_tasks = json.load(open(os.path.join(cleaned, "eval.json")))
        demo_pool = json.load(open(os.path.join(cleaned, "demo_pool.json")))

        extra = self.config.extra if hasattr(self.config, "extra") else {}
        extra = extra or {}
        demos_per_sample = int(extra.get("demos_per_sample", self.DEFAULT_DEMOS_PER_SAMPLE))
        seed = int(extra.get("seed", 42))

        by_type: Dict[str, List[dict]] = {}
        for d in demo_pool:
            by_type.setdefault(d["task_type"], []).append(d)
        rng = random.Random(seed)

        samples: List[BenchmarkSample] = []
        n_same = max(1, int(round(demos_per_sample * self.SAME_TYPE_RATIO)))
        n_other = demos_per_sample - n_same

        for t in eval_tasks:
            gold_actions = [s["action"] for s in t["trajectory"]]
            if not gold_actions:
                continue

            same_type = by_type.get(t["task_type"], [])
            other_type = [d for typ, lst in by_type.items() if typ != t["task_type"] for d in lst]
            picked = rng.sample(same_type, min(n_same, len(same_type))) \
                + rng.sample(other_type, min(n_other, len(other_type)))
            rng.shuffle(picked)

            sessions = []
            for demo in picked:
                turns = [
                    {"content": f"GOAL: {demo['goal']}", "metadata": {"kind": "goal"}},
                    {"content": f"OBS: {demo['initial_obs']}", "metadata": {"kind": "obs"}},
                ]
                for step in demo["trajectory"]:
                    turns.append({"content": f"ACTION: {step['action']}",
                                  "metadata": {"kind": "action"}})
                    turns.append({"content": f"OBS: {step['obs']}",
                                  "metadata": {"kind": "obs"}})
                sessions.append({
                    "session_id": demo["task_id"],
                    "turns": turns,
                    "metadata": {"task_type": demo["task_type"], "goal": demo["goal"]},
                })

            queries = [{
                "id": t["task_id"],
                "question": f"Goal: {t['goal']}\nInitial observation: {t['initial_obs']}\n"
                            f"Output only the next single action to take.",
                "answer": gold_actions[0],
                "type": t["task_type"],
                "gold_trajectory": gold_actions,
            }]

            samples.append(BenchmarkSample(
                sample_id=t["task_id"],
                sessions=sessions,
                queries=queries,
                metadata={"task_type": t["task_type"], "goal": t["goal"]},
            ))

            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        self._data = samples
        return samples

    def build_prompt(self, query: Dict[str, Any], retrieved_context: str) -> str:
        return (
            "You are an agent acting in a simulated household. "
            "The available actions are: go to <recep>, take <obj> from <recep>, "
            "put <obj> in/on <recep>, open <recep>, close <recep>, "
            "toggle <obj> <recep>, clean <obj> with <recep>, "
            "heat <obj> with <recep>, cool <obj> with <recep>.\n\n"
            "Similar past experiences:\n"
            f"{retrieved_context}\n\n"
            f"{query['question']}\n"
            "Output the action on a single line starting with `Action:`.\n"
            "Action:"
        )

    def evaluate(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        per_type: Dict[str, List[float]] = {t: [] for t in self.QUERY_TYPES}
        all_scores: List[float] = []
        for p in predictions:
            pred_txt = p.prediction
            if "Action:" in pred_txt:
                pred_txt = pred_txt.split("Action:", 1)[1]
            pred_txt = pred_txt.strip().splitlines()[0] if pred_txt.strip() else ""
            score = 1.0 if _normalize_action(pred_txt) == _normalize_action(p.gold_answer) else 0.0
            all_scores.append(score)
            if p.query_type in per_type:
                per_type[p.query_type].append(score)

        out = {
            "first_action_em": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
            "num_queries": len(all_scores),
        }
        for typ, xs in per_type.items():
            if xs:
                out[f"first_action_em_{typ}"] = round(sum(xs) / len(xs), 4)
        return out

    def _check_accuracy(self, pred: PredictionResult) -> bool:
        pred_txt = pred.prediction
        if "Action:" in pred_txt:
            pred_txt = pred_txt.split("Action:", 1)[1]
        pred_txt = pred_txt.strip().splitlines()[0] if pred_txt.strip() else ""
        return _normalize_action(pred_txt) == _normalize_action(pred.gold_answer)
