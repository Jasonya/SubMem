"""Main experiment runner.

Usage:
    python -m SubMem.run_experiment --memory_id M1 --benchmark locomo --seed 42
    python -m SubMem.run_experiment --run_all
    python -m SubMem.run_experiment --aggregate
"""
import argparse
import json
import os
import sys
import time

from .config import (
    ExperimentConfig, ModelConfig, MemoryConfig, BenchmarkConfig,
    MEMORY_UNIT_IDS, BENCHMARK_NAMES, to_modern_id,
)
from .models import create_model
from .memory import create_memory_unit
from .benchmarks import create_benchmark
from .metrics.task_performance import TaskPerformanceMetrics
from .metrics.memory_efficiency import MemoryEfficiencyMetrics
from .metrics.memory_quality import MemoryQualityMetrics
from .utils.helpers import set_seed, save_json, get_gpu_memory_usage


def run_single(exp_config: ExperimentConfig) -> dict:
    """Run a single experiment: 1 memory unit x 1 benchmark x 1 seed."""
    set_seed(exp_config.seed)
    print(f"[Run] memory={exp_config.memory.memory_id} "
          f"benchmark={exp_config.benchmark.name} "
          f"seed={exp_config.seed}")

    # 1. Load model
    model = create_model(exp_config.model)
    model.load()
    base_gpu = get_gpu_memory_usage()

    # 2. Create memory unit
    memory_unit = create_memory_unit(exp_config.memory, model)

    # 3. Create benchmark
    benchmark = create_benchmark(exp_config.benchmark)

    # 3b. Tell benchmark where to checkpoint partial predictions. Written
    # atomically after every sample + every 20 queries so partial progress
    # survives timeouts/OOM.
    ckpt_dir = os.path.join(
        exp_config.output_dir,
        exp_config.model.name.replace("/", "_"),
        exp_config.memory.memory_id,
        exp_config.benchmark.name,
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    benchmark._checkpoint_path = os.path.join(ckpt_dir, f"seed_{exp_config.seed}_checkpoint.json")

    # 4. Run benchmark
    t0 = time.time()
    predictions = benchmark.run(model, memory_unit)
    elapsed = time.time() - t0

    # 5. Evaluate (benchmark-specific metrics)
    bench_results = benchmark.evaluate(predictions)

    # 6. Compute cross-family metrics
    all_results = {"benchmark_metrics": bench_results}

    # Family 1: Task Performance
    all_results["task_performance"] = TaskPerformanceMetrics.compute(predictions)

    # Family 2: Memory Efficiency (E1-E14)
    inference_times_ms = getattr(benchmark, "_inference_times_ms", []) or []
    # Re-attach aggregated per-sample timings so MemoryEfficiencyMetrics can
    # see the full write/read history even though reset() was called between samples.
    if hasattr(benchmark, "_agg_write_times"):
        memory_unit._write_times = list(benchmark._agg_write_times)
    if hasattr(benchmark, "_agg_read_times"):
        memory_unit._read_times = list(benchmark._agg_read_times)
    eff = MemoryEfficiencyMetrics.compute(memory_unit, inference_times_ms=inference_times_ms)
    eff["E14_total_time_sec"] = elapsed
    all_results["memory_efficiency"] = eff

    # Family 3: Memory Quality (Q1 compression, Q2 diversity) — uses final memory
    # state, so Q1 reflects only the *last* sample's context. Record anyway;
    # aggregation across samples would require per-sample snapshots.
    original_tokens = (
        sum(getattr(benchmark, "_original_token_counts", []) or [0])
        // 4  # rough chars->tokens heuristic
    )
    try:
        all_results["memory_quality"] = MemoryQualityMetrics.compute(
            memory_unit, original_token_count=original_tokens
        )
    except Exception as e:
        all_results["memory_quality"] = {"error": str(e)}

    # Families 4-7: Retrieval Quality / Robustness / Scaling / Behavioral Dynamics
    # require paired runs (with/without memory), gold evidence IDs, or multiple
    # session-count points. MemoryAgentBench does not ship evidence annotations
    # and single-run smoke tests can't produce paired predictions.
    all_results["retrieval_quality"] = {"status": "N/A — no gold evidence IDs in MAB"}
    all_results["robustness"] = {"status": "N/A — requires contradiction/distractor variants"}
    all_results["scaling_behavior"] = {"status": "N/A — requires multi-session-count sweep"}
    all_results["behavioral_dynamics"] = {"status": "N/A — requires paired with/without-memory runs"}

    # Metadata
    all_results["meta"] = {
        "memory_id": exp_config.memory.memory_id,
        "benchmark": exp_config.benchmark.name,
        "model": exp_config.model.name,
        "seed": exp_config.seed,
        "elapsed_sec": elapsed,
        "num_predictions": len(predictions),
    }

    # Save
    out_dir = os.path.join(
        exp_config.output_dir,
        exp_config.model.name.replace("/", "_"),
        exp_config.memory.memory_id,
        exp_config.benchmark.name,
    )
    out_path = os.path.join(out_dir, f"seed_{exp_config.seed}.json")
    save_json(all_results, out_path)

    # Also dump raw per-query predictions for post-hoc inspection.
    preds_path = os.path.join(out_dir, f"seed_{exp_config.seed}_predictions.json")
    preds_dump = [
        {
            "query_id": p.query_id,
            "query": p.query,
            "prediction": p.prediction,
            "gold_answer": p.gold_answer,
            "query_type": p.query_type,
            "metadata": p.metadata,
        }
        for p in predictions
    ]
    save_json(preds_dump, preds_path)
    print(f"[Done] Saved to {out_path}")

    return all_results


def run_all(base_config: ExperimentConfig):
    """Run all memory x benchmark x seed combinations."""
    results_all = []
    for mid in MEMORY_UNIT_IDS:
        for bname in BENCHMARK_NAMES:
            for seed in base_config.seeds:
                cfg = ExperimentConfig(
                    model=base_config.model,
                    memory=MemoryConfig(memory_id=mid),
                    benchmark=BenchmarkConfig(name=bname),
                    seed=seed,
                    output_dir=base_config.output_dir,
                )
                try:
                    res = run_single(cfg)
                    results_all.append(res)
                except NotImplementedError as e:
                    print(f"[Skip] {mid} not implemented: {e}")
                except Exception as e:
                    print(f"[Error] {mid}/{bname}/seed{seed}: {e}")
    return results_all


def aggregate_results(output_dir: str):
    """Aggregate results across seeds: compute mean +/- std."""
    import glob
    import numpy as np

    pattern = os.path.join(output_dir, "**", "seed_*.json")
    files = glob.glob(pattern, recursive=True)

    groups = {}
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        # Normalise to M-id so old C-id results group with new M-id ones.
        key = (to_modern_id(data["meta"]["memory_id"]), data["meta"]["benchmark"])
        groups.setdefault(key, []).append(data)

    summary = {}
    for (mid, bname), runs in groups.items():
        metric_values = {}
        for run in runs:
            for family_key in ["benchmark_metrics", "task_performance", "memory_efficiency"]:
                if family_key in run:
                    for k, v in run[family_key].items():
                        if isinstance(v, (int, float)):
                            metric_values.setdefault(k, []).append(v)

        agg = {}
        for k, vals in metric_values.items():
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        summary[f"{mid}/{bname}"] = agg

    save_json(summary, os.path.join(output_dir, "aggregated_results.json"))
    print(f"[Aggregate] Saved to {output_dir}/aggregated_results.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SubMem Experiment Runner")
    parser.add_argument("--memory_id", type=str, default="M1")
    parser.add_argument("--benchmark", type=str, default="locomo",
                        choices=BENCHMARK_NAMES)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    exp_config = ExperimentConfig(
        model=ModelConfig(name=args.model),
        memory=MemoryConfig(memory_id=args.memory_id),
        benchmark=BenchmarkConfig(
            name=args.benchmark,
            max_samples=args.max_samples,
        ),
        seed=args.seed,
        output_dir=args.output_dir,
    )

    if args.aggregate:
        aggregate_results(args.output_dir)
    elif args.run_all:
        run_all(exp_config)
    else:
        run_single(exp_config)


if __name__ == "__main__":
    main()
