"""WebArena-Lite benchmark — web agent tasks.

Dataset: weizhepei/WebArena-Lite-SFT
Schema per item: problem, history (list of rounds), observation, solution, remarks
Primary metric: Pass rate (solution match)
"""
import json
import os
from typing import Any, Dict, List

from .base import BaseBenchmark, BenchmarkSample, PredictionResult
from ..config import get_dataset_path


class WebArenaLiteBenchmark(BaseBenchmark):

    @property
    def name(self) -> str:
        return "webarena_lite"

    @property
    def primary_metric(self) -> str:
        return "pass_rate"

    def load_data(self) -> List[BenchmarkSample]:
        data_path = get_dataset_path("webarena_lite")
        split = self.config.split if self.config.split else "test"
        json_file = os.path.join(data_path, f"{split}.json")
        with open(json_file, "r") as f:
            raw = json.load(f)

        samples = []
        for idx, item in enumerate(raw):
            problem = item.get("problem", "")
            history = item.get("history", [])
            observation = item.get("observation", "")
            solution = item.get("solution", "")
            remarks = item.get("remarks", "")

            # Build sessions from interaction history
            sessions = []
            for round_data in history:
                if isinstance(round_data, dict):
                    obs = round_data.get("observation", "")
                    action = round_data.get("action", "")
                    remark = round_data.get("remarks", "")
                    content = f"Observation: {obs}\nAction: {action}"
                    if remark:
                        content += f"\nRemark: {remark}"
                    sessions.append({
                        "session_id": f"round_{round_data.get('round', idx)}",
                        "turns": [{"content": content, "metadata": {"round": round_data.get("round", "")}}],
                    })

            # Current observation as final session
            if observation:
                sessions.append({
                    "session_id": "current",
                    "turns": [{"content": f"Current observation: {observation}", "metadata": {}}],
                })

            queries = [{
                "id": f"webarena_{idx}",
                "question": problem,
                "answer": solution,
                "type": "web_task",
            }]

            samples.append(BenchmarkSample(
                sample_id=f"webarena_{idx}",
                sessions=sessions,
                queries=queries,
                metadata={"remarks": remarks},
            ))

            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        self._data = samples
        return samples

    def build_prompt(self, query: Dict[str, Any], retrieved_context: str) -> str:
        return (
            "You are a web browsing agent. Based on the interaction history and current observation, "
            "decide the next action to complete the task.\n\n"
            "Memory context (previous interactions):\n"
            f"{retrieved_context}\n\n"
            f"Task: {query['question']}\n\n"
            "Provide the action to take.\n"
            "Action:"
        )

    def evaluate(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """Evaluate pass rate based on solution match."""
        total, passed = 0, 0
        for pred in predictions:
            total += 1
            pred_norm = pred.prediction.strip().lower()
            gold_norm = pred.gold_answer.strip().lower()
            if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
                passed += 1

        return {"pass_rate": passed / total if total > 0 else 0.0}
