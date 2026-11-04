"""Base benchmark interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkSample:
    """A single evaluation sample."""
    sample_id: str
    sessions: List[Dict[str, Any]]  # conversation sessions to write into memory
    queries: List[Dict[str, Any]]   # queries to evaluate
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PredictionResult:
    """Model prediction for a single query."""
    query_id: str
    query: str
    prediction: str
    gold_answer: str
    query_type: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseBenchmark(ABC):
    """Abstract benchmark interface."""

    def __init__(self, config):
        self.config = config
        self._data = None

    @abstractmethod
    def load_data(self) -> List[BenchmarkSample]:
        """Load and parse the benchmark dataset."""

    @abstractmethod
    def build_prompt(self, query: Dict[str, Any], retrieved_context: str) -> str:
        """Build the prompt for the model given query + retrieved memory context."""

    @abstractmethod
    def evaluate(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """Compute benchmark-specific metrics on predictions.

        Returns dict of metric_name -> score.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""

    @property
    @abstractmethod
    def primary_metric(self) -> str:
        """Name of the primary metric for this benchmark."""

    def get_query_types(self) -> List[str]:
        """Return list of query sub-types for fine-grained analysis."""
        return []

    def run(self, model, memory_unit) -> List[PredictionResult]:
        """Full run: load data -> write to memory -> query -> collect predictions.

        Side effects on self:
            _inference_times_ms : List[float]   — model.generate() latency per query
            _agg_write_times    : List[float]   — per-write latency, accumulated across samples
            _agg_read_times     : List[float]   — per-read latency, accumulated across samples
            _original_token_counts : List[int]  — raw context chars per sample (proxy tokens)

        Incremental checkpointing:
            If `self._checkpoint_path` is set (string path), we flush partial
            predictions + phase progress to JSON after every sample and every
            20 queries, so timeouts/OOM still leave usable output.
        """
        import json as _json
        import os as _os
        import time

        samples = self.load_data()
        predictions = []
        total_queries = sum(len(s.queries) for s in samples)
        done = 0

        self._inference_times_ms = []
        self._agg_write_times = []
        self._agg_read_times = []
        self._original_token_counts = []

        top_k = self.config.get("top_k", 10) if isinstance(self.config, dict) else 10

        ckpt_path = getattr(self, "_checkpoint_path", None)

        def _write_ckpt(phase, sample_idx, query_idx):
            if not ckpt_path:
                return
            _os.makedirs(_os.path.dirname(ckpt_path) or ".", exist_ok=True)
            tmp = ckpt_path + ".tmp"
            data = {
                "phase": phase,
                "sample_idx": sample_idx,
                "query_idx": query_idx,
                "total_samples": len(samples),
                "total_queries": total_queries,
                "queries_done": len(predictions),
                "predictions": [
                    {
                        "query_id": p.query_id,
                        "query": p.query,
                        "prediction": p.prediction,
                        "gold_answer": p.gold_answer,
                        "query_type": p.query_type,
                        "metadata": p.metadata,
                    }
                    for p in predictions
                ],
            }
            with open(tmp, "w") as f:
                _json.dump(data, f)
            _os.replace(tmp, ckpt_path)

        for si, sample in enumerate(samples):
            # Phase 1: Write sessions into memory
            # Flush per-sample timings from previous sample before reset clears them.
            if hasattr(memory_unit, "_write_times"):
                self._agg_write_times.extend(memory_unit._write_times)
            if hasattr(memory_unit, "_read_times"):
                self._agg_read_times.extend(memory_unit._read_times)
            memory_unit.reset()

            _write_ckpt("write_start", si, 0)

            sample_chars = 0
            n_turns = sum(len(sess.get("turns", [])) for sess in sample.sessions)
            t_write_start = time.perf_counter()
            written = 0
            for session in sample.sessions:
                session_date = session.get("date", "")
                for turn in session.get("turns", []):
                    content = turn["content"]
                    if session_date:
                        content = f"[{session_date}] {content}"
                    sample_chars += len(content)
                    memory_unit.write(
                        content,
                        metadata={"session_id": session.get("session_id"), **turn.get("metadata", {})}
                    )
                    written += 1
                    if written % 200 == 0:
                        elapsed = time.perf_counter() - t_write_start
                        print(f"  [sample {si+1}/{len(samples)}] write {written}/{n_turns} turns ({elapsed:.1f}s)", flush=True)
            self._original_token_counts.append(sample_chars)
            _write_ckpt("write_done", si, 0)

            # Phase 2: Query
            parametric = getattr(memory_unit, "PARAMETRIC_OUTPUT", False)
            # Optional: substrates that need to see all queries before
            # the first read() (e.g. C20 batch-runs the upstream pipeline)
            # can implement stage_queries(queries).
            if hasattr(memory_unit, "stage_queries"):
                try:
                    memory_unit.stage_queries(list(sample.queries))
                except Exception as _e:
                    print(f"  [stage_queries] failed: {_e}", flush=True)
            for qi, query in enumerate(sample.queries):
                t0 = time.perf_counter()
                retrieved = memory_unit.read(query["question"], top_k=top_k)

                if parametric:
                    # Memory already rendered the answer (e.g. Mem0 C5). Take
                    # the first entry's content verbatim; do not re-prompt the
                    # generator on top of its own output.
                    answer = retrieved[0].content if retrieved else ""
                    self._inference_times_ms.append((time.perf_counter() - t0) * 1000)
                else:
                    context = "\n".join([e.content for e in retrieved])
                    prompt = self.build_prompt(query, context)
                    answer = model.generate(prompt)
                    self._inference_times_ms.append((time.perf_counter() - t0) * 1000)

                predictions.append(PredictionResult(
                    query_id=query.get("id", ""),
                    query=query["question"],
                    prediction=answer,
                    gold_answer=query.get("answer", ""),
                    query_type=query.get("type", ""),
                    metadata={
                        "sample_id": sample.sample_id,
                        "sub_answers": query.get("sub_answers", []),
                        "retrieved_chunks": [e.content for e in retrieved],
                        "retrieved_scores": [float(getattr(e, "score", 0.0)) for e in retrieved],
                        "retrieved_metadata": [
                            {
                                "session_id": (e.metadata or {}).get("session_id"),
                                "has_answer": (e.metadata or {}).get("has_answer"),
                                "role": (e.metadata or {}).get("role"),
                            }
                            for e in retrieved
                        ],
                    },
                ))
                done += 1
                if done % 20 == 0 or done == total_queries:
                    print(f"  [{done}/{total_queries}] queries done", flush=True)
                    _write_ckpt("querying", si, qi)

        # Final flush after last sample.
        if hasattr(memory_unit, "_write_times"):
            self._agg_write_times.extend(memory_unit._write_times)
        if hasattr(memory_unit, "_read_times"):
            self._agg_read_times.extend(memory_unit._read_times)

        return predictions
