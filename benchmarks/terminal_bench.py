"""Terminal-Bench 2.0 — terminal/coding agent tasks.

Dataset: harborframework/terminal-bench-2.0
Structure: each task is a folder with task.toml, instruction.md, solution/, tests/, environment/
Primary metric: Pass rate (task solved or not)
"""
import os
from typing import Any, Dict, List

from .base import BaseBenchmark, BenchmarkSample, PredictionResult
from ..config import get_dataset_path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


class TerminalBenchBenchmark(BaseBenchmark):

    @property
    def name(self) -> str:
        return "terminal_bench"

    @property
    def primary_metric(self) -> str:
        return "pass_rate"

    def get_query_types(self) -> List[str]:
        return ["easy", "medium", "hard"]

    def load_data(self) -> List[BenchmarkSample]:
        data_path = get_dataset_path("terminal_bench")

        samples = []
        for task_dir in sorted(os.listdir(data_path)):
            task_toml = os.path.join(data_path, task_dir, "task.toml")
            instruction_md = os.path.join(data_path, task_dir, "instruction.md")
            if not os.path.isfile(task_toml):
                continue

            # Parse task.toml
            with open(task_toml, "rb") as f:
                task_meta = tomllib.load(f)

            # Read instruction
            instruction = ""
            if os.path.isfile(instruction_md):
                with open(instruction_md, "r") as f:
                    instruction = f.read()

            metadata_section = task_meta.get("metadata", {})
            difficulty = metadata_section.get("difficulty", "medium")
            category = metadata_section.get("category", "")

            # The "session" is the instruction context; the "query" is to solve the task
            sessions = [{"session_id": "instruction", "turns": [{"content": instruction, "metadata": {}}]}]
            queries = [{
                "id": task_dir,
                "question": instruction,
                "answer": "",  # answer is verified by test execution, not string match
                "type": difficulty,
            }]

            samples.append(BenchmarkSample(
                sample_id=task_dir,
                sessions=sessions,
                queries=queries,
                metadata={
                    "difficulty": difficulty,
                    "category": category,
                    "timeout_sec": task_meta.get("agent", {}).get("timeout_sec", 900),
                    "docker_image": task_meta.get("environment", {}).get("docker_image", ""),
                    "task_path": os.path.join(data_path, task_dir),
                },
            ))

            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        self._data = samples
        return samples

    def build_prompt(self, query: Dict[str, Any], retrieved_context: str) -> str:
        return (
            "You are an expert software engineer working in a terminal environment.\n\n"
            "Relevant context from memory:\n"
            f"{retrieved_context}\n\n"
            "Task instruction:\n"
            f"{query['question']}\n\n"
            "Provide the commands or code needed to solve this task.\n"
            "Solution:"
        )

    def evaluate(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """Evaluate pass rate. Actual test execution is out of scope here;
        this computes a placeholder based on non-empty predictions.
        Override with docker-based test runner for real evaluation."""
        results = {}
        type_counts = {}
        total, passed = 0, 0

        for pred in predictions:
            total += 1
            # Placeholder: mark as passed if prediction is non-empty
            is_pass = len(pred.prediction.strip()) > 0
            if is_pass:
                passed += 1
            difficulty = pred.query_type
            type_counts.setdefault(difficulty, {"total": 0, "passed": 0})
            type_counts[difficulty]["total"] += 1
            if is_pass:
                type_counts[difficulty]["passed"] += 1

        results["pass_rate"] = passed / total if total > 0 else 0.0

        for diff, counts in type_counts.items():
            results[f"pass_rate_{diff}"] = counts["passed"] / counts["total"] if counts["total"] > 0 else 0.0

        return results
