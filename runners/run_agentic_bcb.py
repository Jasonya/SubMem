"""Agentic BigCodeBench driver for C1..C16 memory substrates.

Populates memory once from a pool of bigcode/bigcodebench-results solutions
(cross-task experiences), then evaluates on a BCB subset by retrieving memory
per task and generating code. Output is a JSONL compatible with
``bigcodebench.evaluate --execution local`` for real pass rate grading.

BCB task model:
  - Single-turn code completion, not multi-turn repo editing.
  - We build a simple prompt: task description + retrieved memory + "write code".
  - Prediction is the raw generated code string (not a patch diff).
  - Post-eval grading is via bigcodebench.evaluate.

Usage:
    python SubMem/run_agentic_bcb.py --memory C1 --max-queries 100
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # parent of the SubMem package

from SubMem.config import ModelConfig, MemoryConfig
from SubMem.models import create_model
from SubMem.memory import create_memory_unit
from SubMem.utils.helpers import set_seed, save_json
from SubMem.metrics.memory_efficiency import MemoryEfficiencyMetrics
from SubMem.metrics.memory_quality import MemoryQualityMetrics


# Repo root: $SUBMEM_ROOT, else derived from this file's location (runners/..).
SUBMEM_ROOT = Path(os.environ.get("SUBMEM_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_BASE = os.environ.get("SUBMEM_RESULTS", str(SUBMEM_ROOT / "results"))
# Hugging Face cache root: $HF_HOME, else the HF default ~/.cache/huggingface.
HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

EXPERIENCES_PATH = os.path.join(RESULTS_BASE, "experiences_bcb_v2.parquet")
BCB_PATH = os.path.join(
    HF_HOME,
    "hub/datasets--bigcode--bigcodebench"
    "/snapshots/b74c0d0bf70d2c0bc459be537895cca163007f1a/data/train-00000-of-00001.parquet",
)


def _build_prompt(task_prompt: str, retrieved_context: str) -> str:
    """Single-turn code-generation prompt.

    Matches BigCodeBench official instruction prefix so our numbers are
    apples-to-apples vs the leaderboard
    (see bigcodebench/provider/utility.py make_raw_chat_prompt). Retrieved
    memory, if any, is prepended as an auxiliary block above BCB's prefix.
    """
    parts = []
    if retrieved_context:
        parts.append(retrieved_context.strip())
        parts.append("")  # blank line separator
    parts.append(
        "Please provide a self-contained Python script that solves the following "
        "problem in a markdown code block:"
    )
    parts.append(task_prompt.strip())
    return "\n".join(parts)


def _format_retrieved(entries, include_labels: bool = True) -> str:
    """Format top-k retrieved memory entries with explicit PASS/FAIL labels.

    When the memory unit carries a `success` boolean in metadata (our pool
    has this for all experiences), emit the entry as either:
      [SUCCESSFUL reference — imitate this pattern]
      <content>
    or
      [FAILED attempt — do NOT imitate, learn from the mistake]
      <content>

    Without labels the model cannot distinguish good from bad examples in
    the pool and may imitate failures, which in practice costs ~5pp pass@1.
    """
    if not entries:
        return ""
    blocks = []
    for e in entries:
        meta = getattr(e, "metadata", None) or {}
        success = meta.get("success")
        if include_labels and success is True:
            header = "[SUCCESSFUL reference — imitate this pattern]"
        elif include_labels and success is False:
            header = "[FAILED attempt — do NOT imitate, learn from the mistake]"
        else:
            header = "[reference]"
        content = getattr(e, "content", "") or ""
        blocks.append(f"{header}\n{content}")
    return "Reference memory:\n" + "\n\n---\n\n".join(blocks)


_CODE_FENCE_RE = __import__("re").compile(
    r"```(?:python)?\s*\n(.*?)```", __import__("re").DOTALL
)


_THINK_RE = __import__("re").compile(r"<think>.*?</think>\s*", __import__("re").DOTALL)


def extract_code(text: str) -> str:
    """Strip markdown fences + <think> reasoning blocks; return plain code."""
    if not text:
        return ""
    # Qwen3 reasoning mode wraps its chain of thought in <think>...</think>.
    # Strip all such blocks before searching for code.
    text = _THINK_RE.sub("", text)
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    # Fallback: return as-is, trimmed
    return text.strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="C14")
    ap.add_argument("--model-name", default="openai/gpt-4o-mini")
    ap.add_argument("--encoder-model", default="text-embedding-3-small")
    ap.add_argument("--experiences", default=EXPERIENCES_PATH)
    ap.add_argument("--bcb-parquet", default=BCB_PATH)
    ap.add_argument("--max-queries", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--balanced-retrieval", action="store_true",
                    help="Retrieve top_k pass + top_k fail separately "
                         "(requires success metadata on entries). "
                         "Over-fetches 8*top_k from memory and filters by "
                         "metadata.success; falls back to unfiltered if not "
                         "enough labeled items come back.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load base model in 4-bit NF4 (QLoRA for C9)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--subset-ids",
        default=None,
        help="Path to newline-delimited BCB task_id list. If None, use first "
             "--max-queries tasks.",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = args.model_name.split("/")[-1]
    run_dir = os.path.join(RESULTS_BASE, f"{ts}_bcb_{short}")
    mem_dir = os.path.join(run_dir, args.memory)
    os.makedirs(mem_dir, exist_ok=True)
    print(f"Run dir: {run_dir}", flush=True)

    # ── Load data ────────────────────────────────────────
    print(f"Loading experiences: {args.experiences}", flush=True)
    exp_df = pd.read_parquet(args.experiences)
    print(f"  {len(exp_df)} experiences "
          f"(pass={int(exp_df.target.sum())}, "
          f"fail={int((~exp_df.target).sum())})", flush=True)

    print(f"Loading BCB: {args.bcb_parquet}", flush=True)
    bcb = pd.read_parquet(args.bcb_parquet)
    if args.subset_ids:
        ids = [l.strip() for l in open(args.subset_ids) if l.strip()]
        bcb = bcb[bcb["task_id"].isin(ids)].reset_index(drop=True)
    if args.max_queries:
        bcb = bcb.head(args.max_queries).copy()
    print(f"  {len(bcb)} tasks", flush=True)

    # ── Model + memory ───────────────────────────────────
    print(f"Creating model {args.model_name}...", flush=True)
    model = create_model(ModelConfig(
        name=args.model_name,
        load_in_4bit=args.load_in_4bit,
    ))
    model.load()

    # C10 FullCtx uses `max_tokens` as its hard context budget when deciding how
    # much buffer to keep. Without this we inherit the C10 default (262144)
    # which silently exceeds whatever the deployed model actually accepts.
    # Reserve headroom for: instance prompt + model_max_new_tokens + safety.
    # Qwen3-32B-AWQ native max is 40960 — leave 8k for generation/system.
    mem_config = MemoryConfig(
        memory_id=args.memory,
        top_k=args.top_k,
        extra={
            "encoder_model": args.encoder_model,
            "max_tokens": int(os.environ.get("SUBMEM_MAX_CTX_TOKENS", "32768")),
        },
    )
    memory_unit = create_memory_unit(mem_config, model)

    # ── Populate memory via write_experience ─────────────
    print(f"\nPopulating memory ({args.memory}) from {len(exp_df)} experiences...",
          flush=True)
    t_w0 = time.time()
    total_original_tokens = 0
    # Use the same tokenizer as Q1 denominator so DR = stored_tokens /
    # original_tokens is in consistent units. Fall back to whitespace words
    # if the model doesn't expose `encode`.
    _tok = getattr(model, "encode", None)
    def _count(txt: str) -> int:
        if _tok is not None:
            try:
                return len(_tok(txt))
            except Exception:
                pass
        return len(txt.split())
    for i, row in enumerate(exp_df.itertuples(index=False), 1):
        memory_unit.write_experience(
            query=row.query,
            trajectory=row.trajectory_text,
            success=bool(row.target),
            metadata={"task_id": row.task_id, "model": row.model_name},
        )
        total_original_tokens += _count(row.trajectory_text)
        if i % 50 == 0:
            print(f"  buffered {i}/{len(exp_df)}", flush=True)

    if hasattr(memory_unit, "_ensure_indexed"):
        memory_unit._ensure_indexed()
    t_write_total = time.time() - t_w0
    print(f"Memory populated. size={memory_unit.size()}  "
          f"write_total={t_write_total:.1f}s", flush=True)

    # ── Per-task retrieve + generate ─────────────────────
    predictions = []
    retrieval_logs = []
    prompt_logs = []
    print(f"\nRunning {len(bcb)} tasks (top_k={args.top_k})...", flush=True)
    t_qa0 = time.time()
    for qi, row in enumerate(bcb.itertuples(index=False), 1):
        task_id = row.task_id
        task_prompt = row.instruct_prompt

        t_r0 = time.perf_counter()
        if args.balanced_retrieval:
            # Over-fetch, then split by metadata.success and take top_k each.
            over_k = max(args.top_k * 8, 10)
            raw = memory_unit.read(task_prompt, top_k=over_k)
            pass_items, fail_items, unk = [], [], []
            for e in raw:
                s = (getattr(e, "metadata", None) or {}).get("success")
                if s is True:
                    pass_items.append(e)
                elif s is False:
                    fail_items.append(e)
                else:
                    unk.append(e)
            pass_items = pass_items[:args.top_k]
            fail_items = fail_items[:args.top_k]
            retrieved = pass_items + fail_items
            # Fallback: if both buckets empty (e.g. C9 LoRA returns synthetic
            # output with no label), keep the raw top_k.
            if not retrieved:
                retrieved = raw[:args.top_k]
        else:
            retrieved = memory_unit.read(task_prompt, top_k=args.top_k)
        t_r1 = time.perf_counter()
        context = _format_retrieved(retrieved, include_labels=True)

        prompt = _build_prompt(task_prompt, context)
        if args.no_think:
            prompt = prompt + " /no_think"
        t_g0 = time.perf_counter()
        answer = model.generate(prompt, max_new_tokens=args.max_new_tokens)
        t_g1 = time.perf_counter()
        code = extract_code(answer)

        predictions.append({
            "task_id": task_id,
            "instruct_prompt": task_prompt,
            "canonical_solution": getattr(row, "canonical_solution", ""),
            "raw_prediction": answer,
            "solution": code,
            "retrieval_latency_ms": round((t_r1 - t_r0) * 1000, 2),
            "generation_latency_ms": round((t_g1 - t_g0) * 1000, 2),
        })
        retrieval_logs.append({
            "task_id": task_id,
            "retrieved": [
                {"content": e.content[:800], "metadata": e.metadata,
                 "score": e.score}
                for e in retrieved
            ],
        })
        prompt_logs.append({
            "task_id": task_id,
            "prompt_length_chars": len(prompt),
        })

        if qi % 10 == 0 or qi == len(bcb):
            print(f"  {qi}/{len(bcb)} done", flush=True)

    t_qa_total = time.time() - t_qa0
    t_total = t_write_total + t_qa_total
    n = len(predictions)

    # ── E-family: efficiency ─────────────────────────────
    inference_times_ms = [p["retrieval_latency_ms"] + p["generation_latency_ms"]
                          for p in predictions]
    efficiency = MemoryEfficiencyMetrics.compute(
        memory_unit, inference_times_ms=inference_times_ms
    )
    efficiency["E14_total_time_sec"] = round(t_total, 2)
    efficiency["E15_avg_time_per_query_sec"] = round(t_total / n, 4) if n else None

    base_id = args.memory.split("-")[0]
    if base_id in ("C8", "C9", "C10", "C11", "C12"):
        skip_prefixes = ("E3_", "E4_", "E5", "E6", "E8", "E9", "E10", "E11",
                         "E12", "E13")
        if base_id not in ("C10", "C11", "C12"):
            skip_prefixes = skip_prefixes + ("E7",)
        for k in list(efficiency.keys()):
            if k.startswith(skip_prefixes):
                efficiency[k] = None

    # ── Q-family: memory quality ─────────────────────────
    try:
        quality = MemoryQualityMetrics.compute(
            memory_unit,
            original_token_count=total_original_tokens,
            tokenizer_encode=getattr(model, "encode", None),
        )
    except Exception as e:
        print(f"WARN: Q-metrics failed: {e}", flush=True)
        quality = {"Q1_compression_ratio": None, "Q2_coverage_diversity": None}

    summary = {
        "memory": args.memory,
        "model": args.model_name,
        "benchmark": "bigcodebench",
        "num_experiences": int(len(exp_df)),
        "num_memory_items_after_distill": memory_unit.size(),
        "num_tasks": n,
        "top_k": args.top_k,
        "time_write_seconds": round(t_write_total, 1),
        "time_qa_seconds": round(t_qa_total, 1),
        "total_original_tokens": total_original_tokens,
        "efficiency": efficiency,
        "quality": quality,
        "memory_unit_stats": memory_unit.get_stats(),
    }
    print("\n=== Summary ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "memory_unit_stats"}, indent=2, default=str))

    save_json(predictions, os.path.join(mem_dir, "05_predictions.json"))
    save_json(retrieval_logs, os.path.join(mem_dir, "03_retrievals.json"))
    save_json(prompt_logs, os.path.join(mem_dir, "04_prompts.json"))
    save_json(summary, os.path.join(mem_dir, "07_metrics.json"))

    # Emit bigcodebench.evaluate-compatible JSONL.
    bcb_jsonl = os.path.join(mem_dir, "preds_bcb.jsonl")
    n_empty = 0
    with open(bcb_jsonl, "w") as f:
        for p in predictions:
            code = p["solution"]
            if not code.strip():
                n_empty += 1
            f.write(json.dumps({"task_id": p["task_id"], "solution": code}) + "\n")
    print(f"Emitted {bcb_jsonl} (empty={n_empty}/{n})", flush=True)
    print(f"\nAll artifacts saved under {mem_dir}", flush=True)


if __name__ == "__main__":
    main()
