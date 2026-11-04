from .base import BaseBenchmark, BenchmarkSample, PredictionResult
from .locomo import LoComoBenchmark
from .longmemeval import LongMemEvalBenchmark
from .memory_agent_bench import MemoryAgentBenchBenchmark
from .terminal_bench import TerminalBenchBenchmark
from .webarena_lite import WebArenaLiteBenchmark
from .alfworld import ALFWorldBenchmark

BENCHMARK_REGISTRY = {
    "locomo": LoComoBenchmark,
    "longmemeval": LongMemEvalBenchmark,
    "memory_agent_bench": MemoryAgentBenchBenchmark,
    "terminal_bench": TerminalBenchBenchmark,
    "webarena_lite": WebArenaLiteBenchmark,
    "alfworld": ALFWorldBenchmark,
}


def create_benchmark(config):
    """Factory: create benchmark from config."""
    cls = BENCHMARK_REGISTRY.get(config.name)
    if cls is None:
        raise ValueError(f"Unknown benchmark: {config.name}. Available: {list(BENCHMARK_REGISTRY.keys())}")
    return cls(config)
