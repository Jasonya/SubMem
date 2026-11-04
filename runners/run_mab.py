"""MemoryAgentBench runner — pick one (task_type, sub_dataset) bucket per run.

Matches the official MemoryAgentBench design: one YAML = one bucket.
Here we expose it as CLI flags instead.

Examples:
    # AR at 64K context (EventQA 65536 variant)
    python -m SubMem.run_mab --memory_id M1 \\
        --task_type accurate_retrieval --sub_dataset eventqa_65536

    # SF single-hop at 32K
    python -m SubMem.run_mab --memory_id M1 \\
        --task_type conflict_resolution --sub_dataset factconsolidation_sh_32k

    # LongMemEval(S*) subset only
    python -m SubMem.run_mab --memory_id M1 \\
        --task_type accurate_retrieval --sub_dataset longmemeval_s*

    # List all available (task_type, sub_dataset) buckets
    python -m SubMem.run_mab --list
"""
import argparse
import os
import sys

from .config import (
    ExperimentConfig, ModelConfig, MemoryConfig, BenchmarkConfig,
    get_dataset_path,
)
from .run_experiment import run_single


TASK_TYPES = [
    "accurate_retrieval",
    "conflict_resolution",
    "long_range_understanding",
    "test_time_learning",
]


def list_buckets():
    """Enumerate every (task_type, sub_dataset) pair with row/question counts."""
    import pandas as pd
    from .benchmarks.memory_agent_bench import TASK_FILES

    base = os.path.join(get_dataset_path("memory_agent_bench"), "data")
    print(f"{'task_type':<28s} {'sub_dataset':<45s} {'seqs':>5s} {'Qs':>6s}")
    print("-" * 90)
    for task_type, filename in TASK_FILES.items():
        path = os.path.join(base, filename)
        if not os.path.exists(path):
            print(f"{task_type:<28s} (parquet missing: {filename})")
            continue
        df = pd.read_parquet(path)
        by_src = {}
        for _, row in df.iterrows():
            src = row["metadata"].get("source", "<unknown>")
            d = by_src.setdefault(src, {"rows": 0, "qs": 0})
            d["rows"] += 1
            d["qs"] += len(row["questions"])
        for src, d in sorted(by_src.items()):
            print(f"{task_type:<28s} {src:<45s} {d['rows']:>5d} {d['qs']:>6d}")


def main():
    parser = argparse.ArgumentParser(
        description="MemoryAgentBench runner (one bucket per invocation)",
    )
    parser.add_argument("--list", action="store_true",
                        help="List all available buckets and exit")
    parser.add_argument("--memory_id", type=str, default="M1")
    parser.add_argument("--task_type", type=str, choices=TASK_TYPES,
                        help="Which capability category to evaluate")
    parser.add_argument("--sub_dataset", type=str, default=None,
                        help="metadata.source filter (e.g. eventqa_65536). "
                             "If omitted, runs every sub_dataset in task_type.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on number of sequences (parquet rows) loaded")
    parser.add_argument("--max_queries_per_sample", type=int, default=None,
                        help="Cap on queries per sequence (useful for smoke tests)")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="bitsandbytes nf4 quantization (saves ~12GB VRAM on Qwen3-8B)")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--memory_max_tokens", type=int, default=None,
                        help="Override memory unit's max_tokens budget (M10 FullContext uses this to cap prompt length)")
    parser.add_argument("--m8_max_queries", type=int, default=None,
                        help="M8-only: cap upstream query count (smoke testing)")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="e.g. 'flash_attention_2' — cuts attention VRAM for long-context prompts")
    parser.add_argument("--block_size", type=int, default=None,
                        help="M11 EpiCache utterance-block size for "
                             "attention scoring forward (default 512). Lower this "
                             "(e.g. 64) to fit large generators like Gemma-26B that "
                             "would otherwise OOM on a 70K-token block.")
    parser.add_argument("--n_medoid", type=int, default=None,
                        help="M11 EpiCache: medoid segments per episode forming the "
                             "patched prompt (default 8). Reduce (e.g. 2) to shrink "
                             "the patched_prompt forward for large generators.")
    parser.add_argument("--w_embed", type=int, default=None,
                        help="M11 EpiCache: utterances per segment for embedding "
                             "(default 4). Reduce to shrink each medoid segment.")
    parser.add_argument("--generator_prompt_cap", type=int, default=None,
                        help="Cap retrieved-context tokens fed into the generator. "
                             "M3 ReadAgent / M5 LightRAG go through gpt-4o-mini "
                             "summarisation when over the cap; others tail-truncate. "
                             "Pass e.g. 12000 for Gemma-26B safety.")
    parser.add_argument("--chunk_words", type=int, default=None,
                        help="Words per chunk for single-blob MAB datasets "
                             "(factconsolidation etc.); default 500. For the "
                             "chunk-size sensitivity sweep.")
    parser.add_argument("--max_gists_in_prompt", type=int, default=None,
                        help="M3 ReadAgent: cap dense-ranked gists put into the "
                             "page-selection prompt (default 200). Lower for "
                             "large generators that OOM on a 30K-token gist forward.")
    parser.add_argument("--page_words", type=int, default=None,
                        help="M3 ReadAgent: words per page in _group_into_pages "
                             "(default 500). Raise to reduce the number of gists "
                             "(and thus the answer-generation prompt size) for "
                             "large models that OOM on long retrieved context.")
    args = parser.parse_args()

    if args.list:
        list_buckets()
        return

    if not args.task_type:
        parser.error("--task_type is required (or use --list). "
                     f"Choices: {TASK_TYPES}")

    extra = {"task_type": args.task_type}
    if args.sub_dataset:
        extra["sub_dataset"] = args.sub_dataset
    if args.max_queries_per_sample is not None:
        extra["max_queries_per_sample"] = args.max_queries_per_sample
    if args.generator_prompt_cap is not None:
        extra["generator_prompt_cap"] = args.generator_prompt_cap
    if args.chunk_words is not None:
        extra["chunk_words"] = args.chunk_words

    tag = args.sub_dataset or args.task_type
    out_subdir = os.path.join(args.output_dir, f"mab_{tag}")

    memory_cfg = MemoryConfig(memory_id=args.memory_id)
    if args.top_k is not None:
        memory_cfg.top_k = args.top_k
    if args.memory_max_tokens is not None:
        memory_cfg.extra["max_tokens"] = args.memory_max_tokens
    if args.m8_max_queries is not None:
        memory_cfg.extra["m8_max_queries"] = args.m8_max_queries
    if args.block_size is not None:
        memory_cfg.extra["block_size"] = args.block_size
    if args.n_medoid is not None:
        memory_cfg.extra["n_medoid"] = args.n_medoid
    if args.w_embed is not None:
        memory_cfg.extra["w_embed"] = args.w_embed
    if args.max_gists_in_prompt is not None:
        memory_cfg.extra["max_gists_in_prompt"] = args.max_gists_in_prompt
    if args.page_words is not None:
        memory_cfg.extra["page_words"] = args.page_words
    # Forward MAB task_type/sub_dataset hints so substrates that abstract
    # away raw chunks (M9 LoRA, M11 EpiCache, M8 MemSkill) can switch to a
    # raw-passthrough mode for ICL benchmarks where the (input, label) text
    # itself is the signal.
    memory_cfg.extra["benchmark_task_type"] = args.task_type
    if args.sub_dataset:
        memory_cfg.extra["benchmark_sub_dataset"] = args.sub_dataset

    model_cfg = ModelConfig(name=args.model)
    if args.load_in_4bit:
        model_cfg.load_in_4bit = True
    if args.load_in_8bit:
        model_cfg.load_in_8bit = True
    if args.attn_implementation:
        model_cfg.attn_implementation = args.attn_implementation

    exp_config = ExperimentConfig(
        model=model_cfg,
        memory=memory_cfg,
        benchmark=BenchmarkConfig(
            name="memory_agent_bench",
            max_samples=args.max_samples,
            extra=extra,
        ),
        seed=args.seed,
        output_dir=out_subdir,
    )
    run_single(exp_config)


if __name__ == "__main__":
    main()
