"""Run Mem0 evaluation directly using mem0's own eval code + SubMem metrics.

Uses mem0's exact pipeline from their evaluation repo:
  - MemoryADD (add.py): dual-speaker, batch add, custom instructions
  - MemorySearch (search.py): search both speakers, ANSWER_PROMPT, GPT generate

But replaces MemoryClient (cloud API) with local Memory.from_config(),
and after collecting predictions, computes SubMem's full 27 metrics.

Output: SubMem-compatible results directory with all intermediates + metrics.

Usage:
    python SubMem/run_mem0_direct.py --max-samples 1
    python SubMem/run_mem0_direct.py --max-samples 10 --top-k 10
    python SubMem/run_mem0_direct.py --eval-only --run-dir results/mem0_direct_...
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Auto-load .env ──
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from jinja2 import Template
from openai import OpenAI

from mem0 import Memory

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from SubMem.benchmarks import create_benchmark
from SubMem.benchmarks.base import PredictionResult
from SubMem.config import BenchmarkConfig
from SubMem.metrics.task_performance import (
    TaskPerformanceMetrics,
    compute_em,
    compute_f1,
)
from SubMem.metrics.llm_judge import LLMJudge, compute_llm_judge_score
from SubMem.metrics.nli_scorer import (
    NLIScorer,
    decompose_claims_rule,
    compute_faithfulness_nli_score,
)
from SubMem.metrics.retrieval_quality import RetrievalQualityMetrics
from SubMem.metrics.robustness import RobustnessMetrics
from SubMem.metrics.behavioral_dynamics import BehavioralDynamicsMetrics
from SubMem.utils.helpers import save_json, load_json

# ── Add mem0 eval repo to path for reference ──
MEM0_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mem0-repo", "evaluation")

RESULTS_BASE = os.environ.get(
    "SUBMEM_RESULTS",
    os.path.join(os.path.dirname(__file__), "results"),
)

# ============================================================
# Prompts — exact copy from mem0-repo/evaluation/prompts.py
# ============================================================
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """

# Custom instructions — exact copy from mem0-repo/evaluation/src/memzero/add.py
CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


# ============================================================
# Mem0 local adapter — replaces MemoryClient with local Memory
# Follows exact same interface as mem0 eval code
# ============================================================

def create_mem0(model: str = "gpt-4o-mini") -> Memory:
    """Create local Mem0 Memory instance (same as eval repo but local)."""
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            },
        },
        "custom_prompt": CUSTOM_INSTRUCTIONS,
        "version": "v1.1",
    }
    return Memory.from_config(config)


# ============================================================
# Add memories — faithful to mem0-repo/evaluation/src/memzero/add.py
# ============================================================

def add_memories(mem0: Memory, conversation: dict, idx: int,
                 batch_size: int = 2) -> dict:
    """Add memories for both speakers (faithful to Mem0 eval add.py).

    Returns timing/stats dict for metrics.
    """
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    speaker_a_id = f"{speaker_a}_{idx}"
    speaker_b_id = f"{speaker_b}_{idx}"

    # Delete existing (same as eval repo)
    try:
        mem0.delete_all(user_id=speaker_a_id)
    except Exception:
        pass
    try:
        mem0.delete_all(user_id=speaker_b_id)
    except Exception:
        pass

    t0 = time.time()
    total_calls = 0
    total_messages = 0

    for key in conversation.keys():
        if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
            continue

        date_time_key = key + "_date_time"
        timestamp = conversation.get(date_time_key, "")
        chats = conversation[key]

        # Build messages for both perspectives (exact same as eval repo)
        messages = []
        messages_reverse = []
        for chat in chats:
            if chat["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
            elif chat["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})

        total_messages += len(messages)

        # Add in batches — speaker A (sequential, same as eval repo)
        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]
            try:
                mem0.add(batch, user_id=speaker_a_id, metadata={"timestamp": timestamp})
                total_calls += 1
            except Exception as e:
                print(f"    [add] speaker_a error: {e}")

        # Add in batches — speaker B
        for i in range(0, len(messages_reverse), batch_size):
            batch = messages_reverse[i : i + batch_size]
            try:
                mem0.add(batch, user_id=speaker_b_id, metadata={"timestamp": timestamp})
                total_calls += 1
            except Exception as e:
                print(f"    [add] speaker_b error: {e}")

        print(f"    Session {key}: {len(chats)} turns added for both speakers", flush=True)

    elapsed = time.time() - t0
    print(f"  Add complete: {total_calls} API calls, {total_messages} messages in {elapsed:.0f}s", flush=True)

    return {
        "speaker_a_id": speaker_a_id,
        "speaker_b_id": speaker_b_id,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "add_time_sec": elapsed,
        "add_api_calls": total_calls,
        "total_messages": total_messages,
    }


# ============================================================
# Search & Answer — faithful to mem0-repo/evaluation/src/memzero/search.py
# ============================================================

def search_and_answer(
    mem0: Memory,
    openai_client: OpenAI,
    speaker_a_id: str,
    speaker_b_id: str,
    question: str,
    top_k: int = 10,
    gen_model: str = "gpt-4o-mini",
) -> dict:
    """Search both speakers and generate answer (faithful to Mem0 eval search.py).

    Returns dict with answer, memories, and timing for metrics.
    """
    # Search speaker A
    t_search_start = time.time()
    try:
        results_a = mem0.search(query=question, user_id=speaker_a_id, limit=top_k)
        if isinstance(results_a, dict):
            results_a = results_a.get("results", [])
    except Exception:
        results_a = []

    # Search speaker B
    try:
        results_b = mem0.search(query=question, user_id=speaker_b_id, limit=top_k)
        if isinstance(results_b, dict):
            results_b = results_b.get("results", [])
    except Exception:
        results_b = []
    t_search_end = time.time()

    # Format memories with timestamps (same as eval repo)
    def format_memories(results):
        formatted = []
        raw_memories = []
        for r in results:
            if isinstance(r, dict):
                mem = r.get("memory", "")
                meta = r.get("metadata") or {}
                score = r.get("score", 0.0)
            elif hasattr(r, "memory"):
                mem = r.memory or ""
                meta = getattr(r, "metadata", None) or {}
                score = getattr(r, "score", 0.0)
            else:
                continue
            ts = meta.get("timestamp", "") if isinstance(meta, dict) else ""
            formatted.append(f"{ts}: {mem}" if ts else mem)
            raw_memories.append({
                "memory": mem,
                "timestamp": ts,
                "score": round(float(score), 4) if score else 0.0,
            })
        return formatted, raw_memories

    mem_a_fmt, mem_a_raw = format_memories(results_a)
    mem_b_fmt, mem_b_raw = format_memories(results_b)

    # Build prompt (exact Mem0 template from eval repo)
    template = Template(ANSWER_PROMPT)
    answer_prompt = template.render(
        speaker_1_user_id=speaker_a_id.split("_")[0],
        speaker_2_user_id=speaker_b_id.split("_")[0],
        speaker_1_memories=json.dumps(mem_a_fmt, indent=4),
        speaker_2_memories=json.dumps(mem_b_fmt, indent=4),
        question=question,
    )

    # Generate answer with OpenAI (same as eval repo)
    t_gen_start = time.time()
    response = openai_client.chat.completions.create(
        model=gen_model,
        messages=[{"role": "system", "content": answer_prompt}],
        temperature=0.0,
    )
    t_gen_end = time.time()

    answer = response.choices[0].message.content.strip()

    # Token usage from response
    usage = response.usage
    gen_input_tokens = usage.prompt_tokens if usage else 0
    gen_output_tokens = usage.completion_tokens if usage else 0

    return {
        "answer": answer,
        "speaker_1_memories": mem_a_raw,
        "speaker_2_memories": mem_b_raw,
        "num_speaker_1_memories": len(mem_a_raw),
        "num_speaker_2_memories": len(mem_b_raw),
        "search_time_sec": t_search_end - t_search_start,
        "gen_time_sec": t_gen_end - t_gen_start,
        "gen_input_tokens": gen_input_tokens,
        "gen_output_tokens": gen_output_tokens,
        "prompt": answer_prompt,
        # For retrieval metrics: combine both speakers' memories as context
        "retrieved_contents": mem_a_fmt + mem_b_fmt,
    }


# ============================================================
# Metrics computation — uses SubMem metrics pipeline
# ============================================================

def _parse_evidence_position(evidence: list) -> int:
    """Parse LoCoMo evidence IDs to an absolute position."""
    if not evidence:
        return 0
    first = evidence[0]
    match = re.match(r"D(\d+):(\d+)", str(first))
    if match:
        return int(match.group(1)) * 1000 + int(match.group(2))
    return 0


def _normalize_refusal(text: str) -> str:
    """Normalize refusal patterns to empty string for adversarial scoring."""
    refusal_patterns = [
        r"(?i)^(i )?(don'?t|do not|cannot|can'?t) (know|have|find|recall|remember)",
        r"(?i)^(there is )?no (information|evidence|data|mention|record)",
        r"(?i)^not (mentioned|available|found|provided|specified|stated)",
        r"(?i)^(i'?m )?(not sure|unable|uncertain)",
        r"(?i)^n/?a$",
        r"(?i)^unknown$",
        r"(?i)^unanswerable$",
    ]
    text = text.strip()
    for pat in refusal_patterns:
        if re.match(pat, text):
            return ""
    return text


def compute_all_metrics(
    all_results: List[dict],
    run_dir: str,
    add_stats: List[dict],
    search_stats: dict,
    nli_scorer: Optional["NLIScorer"] = None,
) -> dict:
    """Compute SubMem's full metrics from collected predictions.

    Args:
        all_results: list of per-query result dicts
        run_dir: directory to save outputs
        add_stats: list of per-conversation add stats
        search_stats: aggregated search/gen timing stats
        nli_scorer: optional loaded NLIScorer for P5 faithfulness
    """
    rd = os.path.join(run_dir, "C5_mem0")
    os.makedirs(rd, exist_ok=True)

    # ── Save raw predictions in SubMem format ──
    prediction_logs = []
    for r in all_results:
        pred_log = {
            "query_id": r["query_id"],
            "sample_id": r.get("sample_id", ""),
            "query": r["query"],
            "query_type": r["query_type"],
            "gold_answer": r["gold_answer"],
            "prediction": r["prediction"],
            "claims": [],
            "retrieved_contents": r.get("retrieved_contents", []),
            "generation_latency_ms": r.get("gen_time_ms", 0),
            "retrieval_latency_ms": r.get("search_time_ms", 0),
            "evidence_position": _parse_evidence_position(r.get("evidence", [])),
        }
        if "adversarial_answer" in r:
            pred_log["adversarial_answer"] = r["adversarial_answer"]
        prediction_logs.append(pred_log)

    save_json(prediction_logs, os.path.join(rd, "05_predictions.json"))

    # ── Build PredictionResult list ──
    predictions = [
        PredictionResult(
            query_id=r["query_id"],
            query=r["query"],
            prediction=r["prediction"],
            gold_answer=str(r["gold_answer"]),
            query_type=r["query_type"],
            metadata={"sample_id": r.get("sample_id", "")},
        )
        for r in all_results
    ]

    # ── P1-P3: Task Performance ──
    task_perf = TaskPerformanceMetrics.compute(predictions)

    # ── P4: LLM-as-Judge ──
    if os.environ.get("OPENAI_API_KEY"):
        try:
            judge_model = os.environ.get("SUBMEM_JUDGE_MODEL", "gpt-4o-mini")
            judge = LLMJudge(model=judge_model)
            p4_results = compute_llm_judge_score(predictions, judge)
            task_perf["P4_llm_judge"] = p4_results["P4_llm_judge"]
            save_json(p4_results, os.path.join(rd, "08_llm_judge.json"))
            print(f"  P4 LLM-Judge:   {p4_results['P4_llm_judge']:.4f}")
        except Exception as e:
            print(f"  P4 LLM-Judge:   SKIPPED ({e})")

    # ── P5: Faithfulness (NLI) ──
    if nli_scorer is not None:
        p5_scores = []
        p5_details = []
        for r in prediction_logs:
            claims = decompose_claims_rule(r["prediction"])
            sources = r.get("retrieved_contents", [])
            res = compute_faithfulness_nli_score(claims, sources, nli_scorer)
            p5_scores.append(res["p5_score"])
            p5_details.append({"query_id": r["query_id"], **res})
        p5_mean = float(np.mean(p5_scores)) if p5_scores else 0.0
        task_perf["P5_faithfulness"] = round(p5_mean, 4)
        save_json(
            {"P5_faithfulness": round(p5_mean, 4), "num_queries": len(p5_scores), "per_query": p5_details},
            os.path.join(rd, "09_faithfulness.json"),
        )
        print(f"  P5 Faithfulness: {p5_mean:.4f} ({len(p5_scores)} queries)")

    # ── P6/P7: Retrieval Quality ──
    # Mem0 returns extracted facts (not raw utterances), so dia_id matching
    # is not directly applicable. We build retrieval logs in SubMem format
    # and let RetrievalQualityMetrics compute what it can.
    retrieval_logs = []
    evidence_map = {}
    for r in prediction_logs:
        qid = r["query_id"]
        gold_evidence = r.get("evidence", [])
        if gold_evidence:
            evidence_map[qid] = gold_evidence
        # Build retrieval log entry — mem0 memories don't have dia_id,
        # so P6/P7 will be 0 unless we can match evidence IDs.
        retrieved_entries = [
            {"rank": i, "score": 0.0, "content": c, "metadata": {}}
            for i, c in enumerate(r.get("retrieved_contents", []))
        ]
        retrieval_logs.append({
            "query_id": qid,
            "gold_evidence": gold_evidence,
            "retrieved": retrieved_entries,
        })
    if retrieval_logs and evidence_map:
        rq = RetrievalQualityMetrics.compute(retrieval_logs, evidence_map)
        task_perf["P6_recall_at_k"] = rq.get("P6_recall_at_k", 0.0)
        task_perf["P7_precision_at_k"] = rq.get("P7_precision_at_k", 0.0)
        print(f"  P6 Recall@k:    {task_perf['P6_recall_at_k']:.4f}")
        print(f"  P7 Precision@k: {task_perf['P7_precision_at_k']:.4f}")
    save_json(retrieval_logs, os.path.join(rd, "03_retrievals.json"))

    # ── R: Robustness ──
    robustness = {}
    distractor_queries = RobustnessMetrics.extract_distractor_queries(prediction_logs)
    if distractor_queries:
        robustness = RobustnessMetrics.compute(
            predictions=predictions,
            distractor_queries=distractor_queries,
        )
        if "R2_drr" in robustness:
            print(f"  R2 DRR:         {robustness['R2_drr']:.4f}")

    # ── D: Behavioral Dynamics ──
    dynamics = {}
    has_position = any(r.get("evidence_position", 0) > 0 for r in prediction_logs)
    if has_position:
        preds_with_pos = [
            PredictionResult(
                query_id=r["query_id"],
                query=r["query"],
                prediction=r["prediction"],
                gold_answer=str(r["gold_answer"]),
                query_type=r["query_type"],
                metadata={"position": r.get("evidence_position", 0)},
            )
            for r in prediction_logs
            if r.get("evidence_position", 0) > 0
        ]
        if preds_with_pos:
            dynamics["D1_recency_bias"] = BehavioralDynamicsMetrics.compute_d3_from_predictions(preds_with_pos)
            print(f"  D1 Recency Bias: {dynamics['D1_recency_bias']:+.4f}")

    # ── E: Efficiency (computed from timing stats) ──
    total_add_time = sum(s["add_time_sec"] for s in add_stats)
    total_add_calls = sum(s["add_api_calls"] for s in add_stats)
    total_messages = sum(s["total_messages"] for s in add_stats)

    search_times = search_stats.get("search_times_ms", [])
    gen_times = search_stats.get("gen_times_ms", [])
    inference_times = [s + g for s, g in zip(search_times, gen_times)]

    total_gen_input_tokens = search_stats.get("total_gen_input_tokens", 0)
    total_gen_output_tokens = search_stats.get("total_gen_output_tokens", 0)

    # Memory size: count stored memories
    mem0_instance = search_stats.get("mem0_instance")
    stored_entries = 0
    stored_bytes = 0
    all_entries_text = []
    if mem0_instance:
        for s in add_stats:
            for uid in [s["speaker_a_id"], s["speaker_b_id"]]:
                try:
                    r = mem0_instance.get_all(user_id=uid)
                    rlist = r.get("results", []) if isinstance(r, dict) else (r or [])
                    for item in rlist:
                        mem_text = item.get("memory", str(item)) if isinstance(item, dict) else str(item)
                        stored_entries += 1
                        stored_bytes += len(mem_text.encode("utf-8"))
                        all_entries_text.append(mem_text)
                except Exception:
                    pass

    efficiency = {
        "E1_storage_bytes": stored_bytes,
        "E2_inference_ms_mean": round(float(np.mean(inference_times)), 2) if inference_times else 0,
        "E2_inference_ms_p50": round(float(np.median(inference_times)), 2) if inference_times else 0,
        "E2_inference_ms_p90": round(float(np.percentile(inference_times, 90)), 2) if inference_times else 0,
        "E3_write_latency_ms_mean": round(total_add_time * 1000 / max(total_messages, 1), 2),
        "E4_retrieval_latency_ms_mean": round(float(np.mean(search_times)), 2) if search_times else 0,
        "E5_construct_tokens": 0,  # Mem0 uses its own internal token counting
        "E6_construct_calls": total_add_calls,
        "E7_retrieved_tokens": sum(
            sum(len(m.encode("utf-8")) for m in r.get("retrieved_contents", []))
            for r in all_results
        ),
        "E8_retrieval_tokens": total_gen_input_tokens,
        "E9_retrieval_calls": len(all_results),  # one search pair per query
        "E10_mgmt_tokens": 0,
        "E11_mgmt_calls": 0,
        "E12_total_tokens": total_gen_input_tokens + total_gen_output_tokens,
        "E13_total_calls": total_add_calls + len(all_results),
        "E14_total_time_sec": round(search_stats.get("total_time_sec", 0), 2),
    }

    mem_stats = {
        "memory_id": "C5_mem0",
        "num_entries": stored_entries,
        "memory_size_bytes": stored_bytes,
        "mem0_model": search_stats.get("mem0_model", "gpt-4o-mini"),
        "mem0_gen_model": search_stats.get("gen_model", "gpt-4o-mini"),
        "mem0_top_k": search_stats.get("top_k", 10),
        "read_latency_ms": {
            "mean": round(float(np.mean(search_times)), 2) if search_times else 0,
            "p50": round(float(np.median(search_times)), 2) if search_times else 0,
            "p90": round(float(np.percentile(search_times, 90)), 2) if search_times else 0,
        },
        "write_latency_ms": {
            "mean": round(total_add_time * 1000 / max(total_messages, 1), 2),
            "p50": None,
            "p90": None,
        },
    }

    # ── Per-query-type breakdown ──
    query_types = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]
    query_type_metrics = {}
    for qtype in query_types:
        typed = [p for p in predictions if p.query_type == qtype]
        if typed:
            em_vals = [compute_em(p.prediction, p.gold_answer) for p in typed]
            f1_vals = [compute_f1(p.prediction, p.gold_answer) for p in typed]
            query_type_metrics[qtype] = {
                "count": len(typed),
                "em": float(np.mean(em_vals)),
                "f1": float(np.mean(f1_vals)),
            }

    # ── Assemble final results ──
    benchmark = create_benchmark(BenchmarkConfig(name="locomo"))
    bench_metrics = benchmark.evaluate(predictions)

    results = {
        "memory_id": "C5_mem0",
        "method": "mem0_direct",
        "num_predictions": len(predictions),
        "elapsed_sec": search_stats.get("total_time_sec", 0),
        "benchmark_metrics": bench_metrics,
        "task_performance": task_perf,
        "robustness": robustness,
        "efficiency": efficiency,
        "quality": {},
        "scaling": {},
        "dynamics": dynamics,
        "memory_unit_stats": mem_stats,
        "per_query_type": query_type_metrics,
    }

    # ── Save ──
    save_json(results, os.path.join(rd, "07_metrics.json"))
    run_meta = {
        "memory_id": "C5_mem0",
        "elapsed_sec": search_stats.get("total_time_sec", 0),
        "num_predictions": len(predictions),
        "memory_unit_stats": mem_stats,
        "efficiency": efficiency,
        "quality": {},
    }
    save_json(run_meta, os.path.join(rd, "06_run_meta.json"))

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  Mem0 Direct — Results")
    print(f"{'='*60}")
    print(f"  Overall:  EM={bench_metrics.get('em', 0):.4f}  F1={bench_metrics.get('f1', 0):.4f}")
    p_str = f"  P1_em={task_perf['P1_em']:.4f}  P2_f1={task_perf['P2_f1']:.4f}  P3_bleu1={task_perf['P3_bleu1']:.4f}"
    if "P4_llm_judge" in task_perf:
        p_str += f"  P4_judge={task_perf['P4_llm_judge']:.4f}"
    if "P5_faithfulness" in task_perf:
        p_str += f"  P5_faith={task_perf['P5_faithfulness']:.4f}"
    if "P6_recall_at_k" in task_perf:
        p_str += f"  P6_recall={task_perf['P6_recall_at_k']:.4f}"
    if "P7_precision_at_k" in task_perf:
        p_str += f"  P7_prec={task_perf['P7_precision_at_k']:.4f}"
    print(p_str)
    print(f"  E1={efficiency['E1_storage_bytes']} bytes  "
          f"E6_calls={efficiency['E6_construct_calls']}  "
          f"E14_time={efficiency['E14_total_time_sec']}s")
    print(f"  Entries stored: {stored_entries}")
    print(f"\n  Per query-type:")
    for qtype, vals in query_type_metrics.items():
        print(f"    {qtype:20s}  n={vals['count']:4d}  EM={vals['em']:.4f}  F1={vals['f1']:.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mem0 Direct Evaluation with SubMem Metrics")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of memories to retrieve per speaker (default: 10)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model for Mem0 internal calls (fact extraction)")
    parser.add_argument("--gen-model", type=str, default=None,
                        help="Model for answer generation (default: same as --model)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for mem0.add() calls (default: 2)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, evaluate from saved predictions")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory for --eval-only")
    parser.add_argument("--no-nli", action="store_true",
                        help="Skip P5 faithfulness (NLI) evaluation")
    args = parser.parse_args()

    gen_model = args.gen_model or args.model

    # ── Load NLI scorer for P5 ──
    nli_scorer = None
    if not args.no_nli:
        try:
            nli_scorer = NLIScorer()
            nli_scorer.load()
            print("NLI scorer loaded (P5 faithfulness enabled)")
        except Exception as e:
            print(f"NLI scorer failed to load, P5 will be skipped: {e}")
            nli_scorer = None

    # ── Eval-only mode ──
    if args.eval_only:
        run_dir = args.run_dir
        if not run_dir:
            # Find latest mem0_direct run
            base = args.output_dir or RESULTS_BASE
            dirs = sorted(
                [d for d in os.listdir(base)
                 if d.startswith("mem0_direct_") and os.path.isdir(os.path.join(base, d))],
                reverse=True,
            )
            if not dirs:
                print("ERROR: No mem0_direct run directory found.")
                sys.exit(1)
            run_dir = os.path.join(base, dirs[0])

        print(f"Evaluating from: {run_dir}")
        pred_file = os.path.join(run_dir, "C5_mem0", "05_predictions.json")
        if not os.path.exists(pred_file):
            # Try flat layout
            pred_file = os.path.join(run_dir, "mem0_predictions.json")
        if not os.path.exists(pred_file):
            print(f"ERROR: predictions not found in {run_dir}")
            sys.exit(1)

        raw_preds = load_json(pred_file)
        # Rebuild metrics from saved predictions
        compute_all_metrics(
            all_results=raw_preds,
            run_dir=run_dir,
            add_stats=[],
            search_stats={"search_times_ms": [], "gen_times_ms": [], "total_time_sec": 0},
            nli_scorer=nli_scorer,
        )
        return

    # ── Load data ──
    data_path = os.path.expanduser("~/.cache/submem/datasets/locomo/locomo10.json")
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("  Run: python SubMem/run_c1c2_test.py --max-samples 1 to download it first")
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    if args.max_samples:
        data = data[: args.max_samples]

    # ── Create output directory ──
    output_base = args.output_dir or RESULTS_BASE
    run_dir = os.path.join(output_base, f"mem0_direct_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Create mem0 instance ──
    print(f"Creating Mem0 (model={args.model}, gen_model={gen_model}, top_k={args.top_k})")
    mem0 = create_mem0(model=args.model)
    openai_client = OpenAI()

    # ── Process conversations ──
    all_results = []
    all_add_stats = []
    search_times_ms = []
    gen_times_ms = []
    total_gen_input_tokens = 0
    total_gen_output_tokens = 0

    t_total_start = time.time()

    for idx, item in enumerate(data):
        conversation = item["conversation"]
        qa = item["qa"]
        sample_id = item.get("sample_id", str(idx))

        print(f"\n{'='*60}")
        print(f"  Conversation {idx}: {conversation['speaker_a']} & {conversation['speaker_b']}")
        print(f"  {len(qa)} questions")
        print(f"{'='*60}")

        # ── Add memories (faithful to mem0 eval) ──
        add_stat = add_memories(mem0, conversation, idx, batch_size=args.batch_size)
        all_add_stats.append(add_stat)

        # ── Answer questions ──
        for qi, q in enumerate(qa):
            question = q["question"]
            gold = q.get("answer", "")
            category = q.get("category", 0)

            result = search_and_answer(
                mem0, openai_client,
                add_stat["speaker_a_id"], add_stat["speaker_b_id"],
                question, top_k=args.top_k, gen_model=gen_model,
            )

            prediction = _normalize_refusal(result["answer"])

            query_type = {
                1: "single_hop", 2: "temporal", 3: "multi_hop",
                4: "open_domain", 5: "adversarial",
            }.get(category, "unknown")

            entry = {
                "query_id": f"{sample_id}_q{qi}",
                "sample_id": sample_id,
                "conversation_idx": idx,
                "query": question,
                "gold_answer": str(gold),
                "prediction": prediction,
                "query_type": query_type,
                "category": category,
                "speaker_1_memories": result["speaker_1_memories"],
                "speaker_2_memories": result["speaker_2_memories"],
                "num_speaker_1_memories": result["num_speaker_1_memories"],
                "num_speaker_2_memories": result["num_speaker_2_memories"],
                "retrieved_contents": result["retrieved_contents"],
                "search_time_ms": round(result["search_time_sec"] * 1000, 2),
                "gen_time_ms": round(result["gen_time_sec"] * 1000, 2),
                "gen_input_tokens": result["gen_input_tokens"],
                "gen_output_tokens": result["gen_output_tokens"],
                "adversarial_answer": q.get("adversarial_answer", ""),
                "evidence": q.get("evidence", []),
            }
            all_results.append(entry)

            search_times_ms.append(entry["search_time_ms"])
            gen_times_ms.append(entry["gen_time_ms"])
            total_gen_input_tokens += result["gen_input_tokens"]
            total_gen_output_tokens += result["gen_output_tokens"]

            if (qi + 1) % 20 == 0:
                print(f"    [{qi+1}/{len(qa)}] answered", flush=True)

        print(f"  Done: {len(qa)} questions answered")

        # Save incrementally
        save_json(all_results, os.path.join(run_dir, "mem0_predictions_raw.json"))

    t_total = time.time() - t_total_start
    print(f"\nTotal time: {t_total:.0f}s for {len(all_results)} predictions")

    # ── Save data summary ──
    data_summary = {
        "dataset": "locomo",
        "num_conversations": len(data),
        "num_questions": len(all_results),
        "mem0_model": args.model,
        "gen_model": gen_model,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
    }
    save_json(data_summary, os.path.join(run_dir, "01_data_summary.json"))

    # ── Compute SubMem metrics ──
    search_stats = {
        "search_times_ms": search_times_ms,
        "gen_times_ms": gen_times_ms,
        "total_gen_input_tokens": total_gen_input_tokens,
        "total_gen_output_tokens": total_gen_output_tokens,
        "total_time_sec": t_total,
        "mem0_instance": mem0,
        "mem0_model": args.model,
        "gen_model": gen_model,
        "top_k": args.top_k,
    }

    compute_all_metrics(all_results, run_dir, all_add_stats, search_stats, nli_scorer=nli_scorer)

    print(f"\nResults saved to {run_dir}/")


if __name__ == "__main__":
    main()
