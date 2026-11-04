"""C5 (Mem0) evaluation — uses mem0's cloned eval code + SubMem metrics.

Pipeline:
  1. mem0_eval.src.memzero.add.MemoryADD   — add memories (dual-speaker, threading, retry)
  2. mem0_eval.src.memzero.search.MemorySearch — search + answer (ANSWER_PROMPT, OpenAI gen)
  3. SubMem metrics (P1-P7, R1-R2, E1-E14, D1, per-query-type)

Output: SubMem-compatible results directory.

Usage:
    python SubMem/run_c5.py --max-samples 1
    python SubMem/run_c5.py --max-samples 10 --top-k 10
    python SubMem/run_c5.py --eval-only --run-dir results/c5_mem0_...
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Auto-load .env ──
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # parent of the SubMem package

# ── mem0 eval code (cloned from mem0-repo, minimally patched) ──
from SubMem.mem0_eval.src.memzero.add import MemoryADD, _create_local_memory
from SubMem.mem0_eval.src.memzero.search import MemorySearch

# ── SubMem metrics ──
from SubMem.benchmarks import create_benchmark
from SubMem.benchmarks.base import PredictionResult
from SubMem.config import BenchmarkConfig
from SubMem.metrics.task_performance import TaskPerformanceMetrics, compute_em, compute_f1
from SubMem.metrics.llm_judge import LLMJudge, compute_llm_judge_score
try:
    from SubMem.metrics.nli_scorer import NLIScorer, decompose_claims_rule, compute_faithfulness_nli_score
except ImportError:
    NLIScorer = None
from SubMem.metrics.retrieval_quality import RetrievalQualityMetrics
from SubMem.metrics.robustness import RobustnessMetrics
from SubMem.metrics.behavioral_dynamics import BehavioralDynamicsMetrics
from SubMem.utils.helpers import save_json, load_json

RESULTS_BASE = os.environ.get("SUBMEM_RESULTS", os.path.join(os.path.dirname(__file__), "results"))

QUERY_TYPE_MAP = {1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_domain", 5: "adversarial"}

# Tokenizer for E7/Q1 (token counting without torch)
try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def _count_tokens(text: str) -> int:
        return len(text.split())  # fallback: word count


class TokenTracker:
    """Wraps mem0's internal OpenAI client to track token usage during add phase.

    Splits calls into two phases by system-prompt signature:
      * construct/write → fact-retrieval (extract new memories from conversation)
      * mgmt           → update-memory (ADD/UPDATE/DELETE/NONE decisions over existing entries)
    """

    # mem0 prompt signatures (see mem0/configs/prompts.py)
    # mem0 v2 merged extract + update into a single ADDITIVE_EXTRACTION call,
    # so there's no longer a distinct "smart memory manager" LLM turn. We
    # still classify on the v1 signatures in case a downstream mem0 pins to
    # the old prompts (custom_prompt only overrides extraction).
    _FACT_SIGS = (
        "You are a Personal Information Organizer",
        "ADDITIVE_EXTRACTION",
        "You analyze a conversation and decide which memories to add, update",
    )
    _MGMT_SIG = "You are a smart memory manager"

    def __init__(self, mem0_instance):
        self.prompt_tokens = 0          # extract only
        self.completion_tokens = 0
        self.call_count = 0
        self.mgmt_prompt_tokens = 0
        self.mgmt_completion_tokens = 0
        self.mgmt_call_count = 0
        self._mem0 = mem0_instance
        self._original_create = None

    def start(self):
        llm = self._mem0.llm
        if hasattr(llm, 'client') and hasattr(llm.client, 'chat'):
            self._original_create = llm.client.chat.completions.create
            llm.client.chat.completions.create = self._tracked_create

    def stop(self):
        if self._original_create is not None:
            self._mem0.llm.client.chat.completions.create = self._original_create
            self._original_create = None

    def _classify(self, kwargs) -> str:
        msgs = kwargs.get("messages") or []
        for m in msgs:
            content = m.get("content", "") if isinstance(m, dict) else ""
            if self._MGMT_SIG in content:
                return "mgmt"
            for sig in self._FACT_SIGS:
                if sig in content:
                    return "write"
        # Unknown prompts default to write (conservative — matches prior behavior)
        return "write"

    def _tracked_create(self, *args, **kwargs):
        response = self._original_create(*args, **kwargs)
        phase = self._classify(kwargs)
        try:
            pt = getattr(response.usage, 'prompt_tokens', 0) or 0 if response.usage else 0
            ct = getattr(response.usage, 'completion_tokens', 0) or 0 if response.usage else 0
        except Exception:
            pt, ct = 0, 0
        if phase == "mgmt":
            self.mgmt_prompt_tokens += pt
            self.mgmt_completion_tokens += ct
            self.mgmt_call_count += 1
        else:
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.call_count += 1
        return response


def _parse_evidence_position(evidence: list) -> int:
    if not evidence:
        return 0
    match = re.match(r"D(\d+):(\d+)", str(evidence[0]))
    return int(match.group(1)) * 1000 + int(match.group(2)) if match else 0


def _normalize_refusal(text: str) -> str:
    patterns = [
        r"(?i)^(i )?(don'?t|do not|cannot|can'?t) (know|have|find|recall|remember)",
        r"(?i)^(there is )?no (information|evidence|data|mention|record)",
        r"(?i)^not (mentioned|available|found|provided|specified|stated)",
        r"(?i)^(i'?m )?(not sure|unable|uncertain)",
        r"(?i)^n/?a$", r"(?i)^unknown$", r"(?i)^unanswerable$",
    ]
    text = text.strip()
    for pat in patterns:
        if re.match(pat, text):
            return ""
    return text


# ============================================================
# Run: add + search via mem0_eval, collect results
# ============================================================

def run_inference(data, mem0, gen_model, args) -> tuple:
    """Run mem0 add + search pipeline, return (all_results, add_stats, search_stats).

    Pipeline:
      1. MemoryADD.process_conversation() — add memories (uses gpt-4o-mini internally)
      2. MemorySearch.search_memory() — retrieve memories (embedding search, no LLM)
      3. Render ANSWER_PROMPT + gen_model.generate() — answer with Qwen (same as other C units)
    """
    from jinja2 import Template
    from SubMem.mem0_eval.prompts import ANSWER_PROMPT

    adder = MemoryADD(mem0=mem0, batch_size=args.batch_size)
    # If the user didn't pass --top-k, let MemorySearch keep its built-in
    # default (10 — matches mem0ai/mem0's evaluation/src/memzero/search.py).
    search_kwargs = {}
    if args.top_k is not None:
        search_kwargs["top_k"] = args.top_k
    searcher = MemorySearch(mem0, **search_kwargs)
    # The effective top_k we actually run with — used downstream in
    # stats / prints rather than the raw CLI arg, which may be None.
    args.effective_top_k = searcher.top_k
    answer_template = Template(ANSWER_PROMPT)

    all_results = []
    all_add_stats = []
    search_times_ms, gen_times_ms = [], []

    t_start = time.time()

    for idx, item in enumerate(data):
        conversation = item["conversation"]
        qa = item["qa"]
        sample_id = item.get("sample_id", str(idx))
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        print(f"\n{'='*60}")
        print(f"  Conversation {idx}: {speaker_a} & {speaker_b} ({len(qa)} questions)")
        print(f"{'='*60}")

        # ── Add (via mem0_eval MemoryADD) — track tokens ──
        tracker = TokenTracker(mem0)
        tracker.start()
        t_add = time.time()
        adder.process_conversation(item, idx)
        add_elapsed = time.time() - t_add
        tracker.stop()

        speaker_a_id = f"{speaker_a}_{idx}"
        speaker_b_id = f"{speaker_b}_{idx}"

        # Count messages
        total_messages = sum(
            len(conversation[k]) for k in conversation
            if k not in ("speaker_a", "speaker_b") and "date" not in k and "timestamp" not in k
        )
        add_stat = {
            "speaker_a_id": speaker_a_id, "speaker_b_id": speaker_b_id,
            "speaker_a": speaker_a, "speaker_b": speaker_b,
            "add_time_sec": add_elapsed, "total_messages": total_messages,
            "add_api_calls": tracker.call_count,
            "construct_prompt_tokens": tracker.prompt_tokens,
            "construct_completion_tokens": tracker.completion_tokens,
            "mgmt_api_calls": tracker.mgmt_call_count,
            "mgmt_prompt_tokens": tracker.mgmt_prompt_tokens,
            "mgmt_completion_tokens": tracker.mgmt_completion_tokens,
            "add_call_latencies_ms": list(getattr(adder, "add_call_latencies_ms", [])),
        }
        all_add_stats.append(add_stat)

        # ── Search + Generate ──
        for qi, q in enumerate(qa):
            question = q["question"]
            gold = q.get("answer", "")
            category = q.get("category", 0)

            # Step 1: Search both speakers (via mem0_eval MemorySearch)
            t_search = time.time()
            mem_a, _, _ = searcher.search_memory(speaker_a_id, question)
            mem_b, _, _ = searcher.search_memory(speaker_b_id, question)
            search_elapsed = time.time() - t_search

            # Step 2: Format memories (same as mem0 eval)
            fmt_a = [f"{m['timestamp']}: {m['memory']}" for m in mem_a]
            fmt_b = [f"{m['timestamp']}: {m['memory']}" for m in mem_b]

            # Step 3: Render ANSWER_PROMPT (exact same template as mem0 eval)
            import json as _json
            prompt = answer_template.render(
                speaker_1_user_id=speaker_a,
                speaker_2_user_id=speaker_b,
                speaker_1_memories=_json.dumps(fmt_a, indent=4),
                speaker_2_memories=_json.dumps(fmt_b, indent=4),
                question=question,
            )

            # Step 4: Generate with Qwen model (same model as other C units)
            t_gen = time.time()
            answer = gen_model.generate(prompt)
            gen_elapsed = time.time() - t_gen

            prediction = _normalize_refusal(answer)
            search_ms = round(search_elapsed * 1000, 2)
            gen_ms = round(gen_elapsed * 1000, 2)
            retrieved_contents = fmt_a + fmt_b

            entry = {
                "query_id": f"{sample_id}_q{qi}",
                "sample_id": sample_id,
                "query": question,
                "gold_answer": str(gold),
                "prediction": prediction,
                "query_type": QUERY_TYPE_MAP.get(category, "unknown"),
                "category": category,
                "speaker_1_memories": mem_a,
                "speaker_2_memories": mem_b,
                "retrieved_contents": retrieved_contents,
                "search_time_ms": search_ms,
                "gen_time_ms": gen_ms,
                "adversarial_answer": q.get("adversarial_answer", ""),
                "evidence": q.get("evidence", []),
            }
            all_results.append(entry)
            search_times_ms.append(search_ms)
            gen_times_ms.append(gen_ms)

            if (qi + 1) % 20 == 0:
                print(f"    [{qi+1}/{len(qa)}] answered", flush=True)

        print(f"  Done: {len(qa)} questions answered")

    t_total = time.time() - t_start

    search_stats = {
        "search_times_ms": search_times_ms,
        "gen_times_ms": gen_times_ms,
        "total_gen_input_tokens": 0,  # local model, not tracked via API
        "total_gen_output_tokens": 0,
        "total_time_sec": t_total,
        "mem0_instance": mem0,
        "mem0_model": args.mem0_model,
        "gen_model": args.model,
        "top_k": args.effective_top_k,
    }
    return all_results, all_add_stats, search_stats


# ============================================================
# Compute all SubMem metrics
# ============================================================

def compute_all_metrics(all_results, run_dir, add_stats, search_stats, nli_scorer=None):
    rd = os.path.join(run_dir, "C5")
    os.makedirs(rd, exist_ok=True)

    # ── Prediction logs ──
    prediction_logs = []
    for r in all_results:
        pred_log = {
            "query_id": r["query_id"], "sample_id": r.get("sample_id", ""),
            "query": r["query"], "query_type": r["query_type"],
            "gold_answer": r["gold_answer"], "prediction": r["prediction"],
            "claims": [], "retrieved_contents": r.get("retrieved_contents", []),
            "generation_latency_ms": r.get("gen_time_ms", 0),
            "retrieval_latency_ms": r.get("search_time_ms", 0),
            "evidence_position": _parse_evidence_position(r.get("evidence", [])),
            "evidence": r.get("evidence", []),
        }
        if "adversarial_answer" in r:
            pred_log["adversarial_answer"] = r["adversarial_answer"]
        prediction_logs.append(pred_log)
    save_json(prediction_logs, os.path.join(rd, "05_predictions.json"))

    predictions = [
        PredictionResult(
            query_id=r["query_id"], query=r["query"],
            prediction=r["prediction"], gold_answer=str(r["gold_answer"]),
            query_type=r["query_type"], metadata={"sample_id": r.get("sample_id", "")},
        )
        for r in all_results
    ]

    # ── P1-P3 ──
    task_perf = TaskPerformanceMetrics.compute(predictions)

    # ── P4: LLM Judge ──
    if os.environ.get("OPENAI_API_KEY"):
        try:
            judge = LLMJudge(model=os.environ.get("SUBMEM_JUDGE_MODEL", "gpt-4o-mini"))
            p4 = compute_llm_judge_score(predictions, judge)
            task_perf["P4_llm_judge"] = p4["P4_llm_judge"]
            save_json(p4, os.path.join(rd, "08_llm_judge.json"))
            print(f"  P4 LLM-Judge:   {p4['P4_llm_judge']:.4f}")
        except Exception as e:
            print(f"  P4 LLM-Judge:   SKIPPED ({e})")

    # ── P5: Faithfulness (NLI) ──
    if nli_scorer is not None:
        p5_scores, p5_details = [], []
        for r in prediction_logs:
            claims = decompose_claims_rule(r["prediction"])
            res = compute_faithfulness_nli_score(claims, r.get("retrieved_contents", []), nli_scorer)
            p5_scores.append(res["p5_score"])
            p5_details.append({"query_id": r["query_id"], **res})
        p5_mean = float(np.mean(p5_scores)) if p5_scores else 0.0
        task_perf["P5_faithfulness"] = round(p5_mean, 4)
        save_json({"P5_faithfulness": round(p5_mean, 4), "num_queries": len(p5_scores), "per_query": p5_details},
                  os.path.join(rd, "09_faithfulness.json"))
        print(f"  P5 Faithfulness: {p5_mean:.4f} ({len(p5_scores)} queries)")

    # ── P6/P7: Retrieval Quality ──
    # Build retrieval logs with dia_ids from mem0 search results
    retrieval_logs, evidence_map = [], {}
    for r in all_results:
        qid = r["query_id"]
        gold_ev = r.get("evidence", [])
        if gold_ev:
            evidence_map[qid] = gold_ev
        retrieved_entries = []
        for i, mem in enumerate(r.get("speaker_1_memories", []) + r.get("speaker_2_memories", [])):
            meta = {}
            if isinstance(mem, dict):
                # dia_ids from mem0 metadata (source utterance IDs)
                dia_ids = mem.get("dia_ids", [])
                if dia_ids:
                    meta["dia_ids"] = dia_ids
            retrieved_entries.append({
                "rank": i, "score": mem.get("score", 0.0) if isinstance(mem, dict) else 0.0,
                "content": mem.get("memory", str(mem)) if isinstance(mem, dict) else str(mem),
                "metadata": meta,
            })
        retrieval_logs.append({
            "query_id": qid, "gold_evidence": gold_ev,
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
        robustness = RobustnessMetrics.compute(predictions=predictions, distractor_queries=distractor_queries)
        if "R2_drr" in robustness:
            print(f"  R2 DRR:         {robustness['R2_drr']:.4f}")

    # ── D: Behavioral Dynamics ──
    dynamics = {}
    preds_with_pos = [
        PredictionResult(query_id=r["query_id"], query=r["query"], prediction=r["prediction"],
                         gold_answer=str(r["gold_answer"]), query_type=r["query_type"],
                         metadata={"position": r.get("evidence_position", 0)})
        for r in prediction_logs if r.get("evidence_position", 0) > 0
    ]
    if preds_with_pos:
        dynamics["D1_recency_bias"] = BehavioralDynamicsMetrics.compute_d3_from_predictions(preds_with_pos)
        print(f"  D1 Recency Bias: {dynamics['D1_recency_bias']:+.4f}")

    # ── E: Efficiency ──
    # In eval-only mode (no add_stats/search_stats), preserve existing E from 06_run_meta.json
    existing_meta_path = os.path.join(rd, "06_run_meta.json")
    existing_meta = load_json(existing_meta_path) if os.path.exists(existing_meta_path) else {}
    existing_efficiency = existing_meta.get("efficiency", {})
    existing_mem_stats = existing_meta.get("memory_unit_stats", {})

    has_runtime = bool(add_stats) and bool(search_stats.get("search_times_ms"))

    if has_runtime:
        total_add_time = sum(s.get("add_time_sec", 0) for s in add_stats)
        total_messages = sum(s.get("total_messages", 0) for s in add_stats)
        search_times = search_stats.get("search_times_ms", [])
        gen_times = search_stats.get("gen_times_ms", [])
        inference_times = [s + g for s, g in zip(search_times, gen_times)]

        # Count stored memories
        mem0_inst = search_stats.get("mem0_instance")
        stored_entries, stored_bytes = 0, 0
        if mem0_inst:
            for s in add_stats:
                for uid in [s["speaker_a_id"], s["speaker_b_id"]]:
                    try:
                        # mem0 v2 API: get_all takes filters dict
                        try:
                            r = mem0_inst.get_all(filters={"user_id": uid})
                        except TypeError:
                            r = mem0_inst.get_all(user_id=uid)
                        rlist = r.get("results", []) if isinstance(r, dict) else (r or [])
                        for item in rlist:
                            text = item.get("memory", str(item)) if isinstance(item, dict) else str(item)
                            stored_entries += 1
                            stored_bytes += len(text.encode("utf-8"))
                    except Exception:
                        pass
        # Fallback: if get_all returned nothing, measure the actual on-disk
        # storage of the mem0 SQLite + Chroma directories. Reflects E1 as
        # raw bytes consumed by the substrate, even if API enumeration fails.
        if stored_bytes == 0:
            mem0_dir = os.environ.get("MEM0_DIR", "")
            if mem0_dir and os.path.isdir(mem0_dir):
                for dirpath, _, files in os.walk(mem0_dir):
                    for f in files:
                        try:
                            stored_bytes += os.path.getsize(os.path.join(dirpath, f))
                        except OSError:
                            pass

        total_add_calls = sum(s.get("add_api_calls", 0) for s in add_stats)
        total_mgmt_calls = sum(s.get("mgmt_api_calls", 0) for s in add_stats)
        total_mgmt_tokens = sum(s.get("mgmt_prompt_tokens", 0) + s.get("mgmt_completion_tokens", 0) for s in add_stats)

        efficiency = {
            "E1_storage_bytes": stored_bytes,
            "E2_inference_ms_mean": round(float(np.mean(inference_times)), 2),
            "E2_inference_ms_p50": round(float(np.median(inference_times)), 2),
            "E2_inference_ms_p90": round(float(np.percentile(inference_times, 90)), 2),
            "E3_write_latency_ms_mean": round(total_add_time * 1000 / max(total_messages, 1), 2),
            "E3_write_latency_ms_p50": (round(float(np.median([x for s in add_stats for x in s.get("add_call_latencies_ms", [])])), 2)
                                          if any(s.get("add_call_latencies_ms") for s in add_stats) else None),
            "E3_write_latency_ms_p90": (round(float(np.percentile([x for s in add_stats for x in s.get("add_call_latencies_ms", [])], 90)), 2)
                                          if any(s.get("add_call_latencies_ms") for s in add_stats) else None),
            "E4_retrieval_latency_ms_mean": round(float(np.mean(search_times)), 2),
            "E5_construct_tokens": sum(s.get("construct_prompt_tokens", 0) + s.get("construct_completion_tokens", 0) for s in add_stats),
            "E6_construct_calls": total_add_calls,
            "E7_retrieved_tokens": sum(sum(_count_tokens(m) for m in r.get("retrieved_contents", [])) for r in all_results),
            "E8_retrieval_tokens": 0,  # mem0 search is embedding-based, no LLM calls
            # E9 counts *retrieval LLM* calls. mem0 search is pure embedding
            # lookup (no LLM), so E9 = 0 by definition. The previous value
            # (2 * #queries) conflated FAISS search with LLM calls.
            "E9_retrieval_calls": 0,
            "E10_mgmt_tokens": total_mgmt_tokens,
            "E11_mgmt_calls": total_mgmt_calls,
        }
        # E12 = E5 + E7 + E8 + E10 (total LLM tokens across all phases)
        efficiency["E12_total_tokens"] = (efficiency["E5_construct_tokens"] + efficiency["E7_retrieved_tokens"]
                                          + efficiency["E8_retrieval_tokens"] + efficiency["E10_mgmt_tokens"])
        # E13 = E6 + E9 + E11 (total LLM calls across all phases)
        efficiency["E13_total_calls"] = efficiency["E6_construct_calls"] + efficiency["E9_retrieval_calls"] + efficiency["E11_mgmt_calls"]
        efficiency["E14_total_time_sec"] = round(search_stats.get("total_time_sec", 0), 2)
        # E15: average wall time per query (E14 / #queries)
        n_queries_total = max(len(all_results), 1)
        efficiency["E15_avg_time_per_query_sec"] = round(
            efficiency["E14_total_time_sec"] / n_queries_total, 4
        )
        # Token consumption (consistent with other memory units' format)
        construct_prompt = sum(s.get("construct_prompt_tokens", 0) for s in add_stats)
        construct_completion = sum(s.get("construct_completion_tokens", 0) for s in add_stats)
        retrieved_tokens = efficiency["E7_retrieved_tokens"]
        mgmt_prompt = sum(s.get("mgmt_prompt_tokens", 0) for s in add_stats)
        mgmt_completion = sum(s.get("mgmt_completion_tokens", 0) for s in add_stats)
        token_consumption = {
            "write_llm_input": construct_prompt,
            "write_llm_output": construct_completion,
            "write_llm_calls": total_add_calls,
            "read_context_tokens": retrieved_tokens,
            "read_calls": len(all_results),
            "mgmt_llm_input": mgmt_prompt,
            "mgmt_llm_output": mgmt_completion,
            "mgmt_llm_calls": total_mgmt_calls,
        }
        mem_stats = {
            "memory_id": "C5", "num_entries": stored_entries, "memory_size_bytes": stored_bytes,
            "mem0_model": search_stats.get("mem0_model", "gpt-4o-mini"),
            "mem0_gen_model": search_stats.get("gen_model", "gpt-4o-mini"),
            "mem0_top_k": search_stats.get("top_k", 10),
            "token_consumption": token_consumption,
            "write_call_count": total_messages,
            "read_latency_ms": {
                "mean": round(float(np.mean(search_times)), 2),
                "p50": round(float(np.median(search_times)), 2),
                "p90": round(float(np.percentile(search_times, 90)), 2),
            },
            "write_latency_ms": {"mean": round(total_add_time * 1000 / max(total_messages, 1), 2), "p50": None, "p90": None},
        }
    else:
        # Eval-only: recover E stats from existing meta OR from raw_predictions timing
        if existing_efficiency and existing_efficiency.get("E2_inference_ms_mean", 0) > 0:
            efficiency = dict(existing_efficiency)  # copy so we can update
            # Always recompute E7 (tokens, not bytes) and E12/E13 (correct formulas)
            efficiency["E7_retrieved_tokens"] = sum(sum(_count_tokens(m) for m in r.get("retrieved_contents", [])) for r in all_results)
            efficiency["E8_retrieval_tokens"] = 0  # mem0 search is embedding-based
            efficiency["E12_total_tokens"] = (efficiency.get("E5_construct_tokens", 0) + efficiency["E7_retrieved_tokens"]
                                              + efficiency["E8_retrieval_tokens"] + efficiency.get("E10_mgmt_tokens", 0))
            efficiency["E13_total_calls"] = (efficiency.get("E6_construct_calls", 0) + efficiency.get("E9_retrieval_calls", 0)
                                             + efficiency.get("E11_mgmt_calls", 0))
            # Keep E15 consistent whether we inherit existing or recompute
            if "E14_total_time_sec" in efficiency and efficiency["E14_total_time_sec"]:
                efficiency["E15_avg_time_per_query_sec"] = round(
                    efficiency["E14_total_time_sec"] / max(len(all_results), 1), 4
                )
            mem_stats = existing_mem_stats
            stored_entries = existing_mem_stats.get("num_entries", 0)
        else:
            # Rebuild E2/E3/E4 from per-query timing in raw_predictions
            search_times = [r.get("search_time_ms", 0) for r in all_results if r.get("search_time_ms")]
            gen_times = [r.get("gen_time_ms", 0) for r in all_results if r.get("gen_time_ms")]
            inference_times = [s + g for s, g in zip(search_times, gen_times)] if search_times else []
            efficiency = {
                "E1_storage_bytes": existing_efficiency.get("E1_storage_bytes", 0),
                "E2_inference_ms_mean": round(float(np.mean(inference_times)), 2) if inference_times else 0,
                "E2_inference_ms_p50": round(float(np.median(inference_times)), 2) if inference_times else 0,
                "E2_inference_ms_p90": round(float(np.percentile(inference_times, 90)), 2) if inference_times else 0,
                "E3_write_latency_ms_mean": existing_efficiency.get("E3_write_latency_ms_mean", 0),
                "E4_retrieval_latency_ms_mean": round(float(np.mean(search_times)), 2) if search_times else 0,
                "E5_construct_tokens": 0,
                "E6_construct_calls": existing_efficiency.get("E6_construct_calls", 0),
                "E7_retrieved_tokens": sum(sum(_count_tokens(m) for m in r.get("retrieved_contents", [])) for r in all_results),
                "E8_retrieval_tokens": 0,  # mem0 search is embedding-based
                "E9_retrieval_calls": 0,  # mem0 search is embedding-only
                "E10_mgmt_tokens": 0, "E11_mgmt_calls": 0,
            }
            # E12 = E5 + E7 + E8 + E10
            efficiency["E12_total_tokens"] = (efficiency["E5_construct_tokens"] + efficiency["E7_retrieved_tokens"]
                                              + efficiency["E8_retrieval_tokens"] + efficiency["E10_mgmt_tokens"])
            # E13 = E6 + E9 + E11
            efficiency["E13_total_calls"] = efficiency["E6_construct_calls"] + efficiency["E9_retrieval_calls"] + efficiency["E11_mgmt_calls"]
            efficiency["E14_total_time_sec"] = existing_efficiency.get("E14_total_time_sec", 0)
            efficiency["E15_avg_time_per_query_sec"] = round(
                efficiency["E14_total_time_sec"] / max(len(all_results), 1), 4
            )
            mem_stats = existing_mem_stats if existing_mem_stats else {
                "memory_id": "C5", "num_entries": 0, "memory_size_bytes": 0,
                "read_latency_ms": {
                    "mean": round(float(np.mean(search_times)), 2) if search_times else 0,
                    "p50": round(float(np.median(search_times)), 2) if search_times else 0,
                    "p90": round(float(np.percentile(search_times, 90)), 2) if search_times else 0,
                },
                "write_latency_ms": {"mean": 0, "p50": None, "p90": None},
            }
            stored_entries = mem_stats.get("num_entries", 0)

    # ── Per-query-type ──
    query_type_metrics = {}
    for qtype in ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]:
        typed = [p for p in predictions if p.query_type == qtype]
        if typed:
            query_type_metrics[qtype] = {
                "count": len(typed),
                "em": float(np.mean([compute_em(p.prediction, p.gold_answer) for p in typed])),
                "f1": float(np.mean([compute_f1(p.prediction, p.gold_answer) for p in typed])),
            }

    # ── Q: Quality ──
    quality = {}
    try:
        # Q1: compression ratio = original input tokens / stored memory tokens
        # Original = all conversation utterances written to mem0
        # Stored = extracted fact memories
        original_tokens = 0
        data_path = os.path.expanduser("~/.cache/submem/datasets/locomo/locomo10.json")
        if os.path.exists(data_path):
            with open(data_path) as f:
                locomo_data = json.load(f)
            # Count tokens from all conversations that were processed
            sample_ids = set(r.get("sample_id", "") for r in all_results)
            for item in locomo_data:
                if str(item.get("sample_id", "")) not in sample_ids:
                    continue
                conv = item["conversation"]
                for k in conv:
                    if k in ("speaker_a", "speaker_b") or "date" in k or "timestamp" in k:
                        continue
                    for chat in conv[k]:
                        original_tokens += _count_tokens(f"{chat['speaker']}: {chat['text']}")

        # Stored = unique extracted facts from mem0
        all_stored = set()
        for r in all_results:
            for m in r.get("speaker_1_memories", []) + r.get("speaker_2_memories", []):
                mem_text = m.get("memory", str(m)) if isinstance(m, dict) else str(m)
                all_stored.add(mem_text)
        stored_tokens = sum(_count_tokens(t) for t in all_stored)

        if stored_tokens > 0 and original_tokens > 0:
            quality["Q1_compression_ratio"] = round(original_tokens / stored_tokens, 4)
        else:
            quality["Q1_compression_ratio"] = float("inf")

        # Q2: coverage diversity via sklearn TF-IDF (stopwords removed) + KMeans + Shannon entropy
        if len(all_stored) >= 2:
            import math
            from collections import Counter

            # Deterministic ordering (sets iterate in PYTHONHASHSEED order)
            stored_sorted = sorted(all_stored)
            n_docs = len(stored_sorted)
            k = min(5, n_docs)

            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.cluster import KMeans

                # TF-IDF with English stopwords + bigrams, remove very common/rare terms
                vec = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8,
                    sublinear_tf=True,
                )
                X = vec.fit_transform(stored_sorted)
                if X.shape[1] == 0:  # no features after filtering
                    labels = np.random.RandomState(42).randint(0, k, size=n_docs)
                else:
                    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
            except ImportError:
                labels = np.random.RandomState(42).randint(0, k, size=n_docs)

            counts = Counter(labels)
            total = sum(counts.values())
            entropy = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
            quality["Q2_coverage_diversity"] = round(entropy, 4)
        else:
            quality["Q2_coverage_diversity"] = 0.0

        print(f"  Q1 Compression:  {quality.get('Q1_compression_ratio', 0):.4f}")
        print(f"  Q2 Diversity:    {quality.get('Q2_coverage_diversity', 0):.4f}")
    except Exception as e:
        print(f"  Q1/Q2: SKIPPED ({e})")

    # ── Benchmark metrics ──
    bench_metrics = create_benchmark(BenchmarkConfig(name="locomo")).evaluate(predictions)

    results = {
        "memory_id": "C5", "method": "mem0_eval",
        "num_predictions": len(predictions),
        "elapsed_sec": search_stats.get("total_time_sec", 0),
        "benchmark_metrics": bench_metrics, "task_performance": task_perf,
        "robustness": robustness, "efficiency": efficiency,
        "quality": quality, "scaling": {}, "dynamics": dynamics,
        "memory_unit_stats": mem_stats, "per_query_type": query_type_metrics,
    }

    save_json(results, os.path.join(rd, "07_metrics.json"))
    save_json({"memory_id": "C5", "elapsed_sec": search_stats.get("total_time_sec", 0),
               "num_predictions": len(predictions), "memory_unit_stats": mem_stats,
               "efficiency": efficiency, "quality": quality}, os.path.join(rd, "06_run_meta.json"))

    # ── Print ──
    print(f"\n{'='*60}")
    print(f"  C5 (Mem0) — Results")
    print(f"{'='*60}")
    print(f"  Overall:  EM={bench_metrics.get('em', 0):.4f}  F1={bench_metrics.get('f1', 0):.4f}")
    parts = [f"P1={task_perf['P1_em']:.4f}", f"P2={task_perf['P2_f1']:.4f}", f"P3={task_perf['P3_bleu1']:.4f}"]
    for k, label in [("P4_llm_judge", "P4"), ("P5_faithfulness", "P5"),
                     ("P6_recall_at_k", "P6"), ("P7_precision_at_k", "P7")]:
        if k in task_perf:
            parts.append(f"{label}={task_perf[k]:.4f}")
    print(f"  {'  '.join(parts)}")
    print(f"  E1={efficiency['E1_storage_bytes']}B  E14={efficiency['E14_total_time_sec']}s  entries={stored_entries}")
    for qtype, vals in query_type_metrics.items():
        print(f"    {qtype:20s}  n={vals['count']:4d}  EM={vals['em']:.4f}  F1={vals['f1']:.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="C5 (Mem0) Evaluation with SubMem Metrics")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Retrieval top-k. Unset = use mem0 evaluation "
                             "repo's default (10); set explicitly to force.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B",
                        help="Generation model for answering questions (same as other C units)")
    parser.add_argument("--mem0-model", type=str, default="gpt-4o-mini",
                        help="Mem0 internal LLM for fact extraction (default: gpt-4o-mini)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--no-nli", action="store_true", help="Skip P5")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device-map", type=str, default="auto")
    args = parser.parse_args()

    # ── NLI scorer ──
    nli_scorer = None
    if not args.no_nli:
        try:
            nli_scorer = NLIScorer()
            nli_scorer.load()
            print("NLI scorer loaded (P5 enabled)")
        except Exception as e:
            print(f"NLI scorer unavailable, P5 skipped: {e}")

    # ── Eval-only ──
    if args.eval_only:
        run_dir = args.run_dir
        if not run_dir:
            base = args.output_dir or RESULTS_BASE
            dirs = sorted([d for d in os.listdir(base) if d.startswith("c5_mem0_")], reverse=True)
            if not dirs:
                sys.exit("ERROR: No c5_mem0_ run directory found.")
            run_dir = os.path.join(base, dirs[0])
        print(f"Evaluating from: {run_dir}")
        raw_file = os.path.join(run_dir, "raw_predictions.json")
        pred_file = os.path.join(run_dir, "C5", "05_predictions.json")
        if os.path.exists(raw_file):
            all_results = load_json(raw_file)
        elif os.path.exists(pred_file):
            all_results = load_json(pred_file)
        else:
            sys.exit(f"ERROR: No predictions found in {run_dir}")
        compute_all_metrics(all_results, run_dir, [],
                            {"search_times_ms": [], "gen_times_ms": [], "total_time_sec": 0},
                            nli_scorer=nli_scorer)
        return

    # ── Load generation model (Qwen) ──
    from SubMem.config import ModelConfig
    from SubMem.models import create_model

    model_config = ModelConfig(
        name=args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
    )
    gen_model = create_model(model_config)
    gen_model.load()
    print(f"Generation model loaded: {args.model}")

    # ── Load data ──
    data_path = os.path.expanduser("~/.cache/submem/datasets/locomo/locomo10.json")
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: Dataset not found at {data_path}")
    with open(data_path) as f:
        data = json.load(f)
    if args.max_samples:
        data = data[:args.max_samples]

    # ── Output dir ──
    output_base = args.output_dir or RESULTS_BASE
    short_model = args.model.split("/")[-1]
    run_dir = os.path.join(output_base, f"c5_mem0_{time.strftime('%Y%m%d_%H%M%S')}_{short_model}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Create mem0 (internal LLM = gpt-4o-mini for fact extraction) ──
    tk_str = str(args.top_k) if args.top_k is not None else "(default)"
    print(f"Creating Mem0 (mem0_model={args.mem0_model}, gen_model={args.model}, top_k={tk_str})")
    from SubMem.mem0_eval.src.memzero.add import custom_instructions
    mem0 = _create_local_memory(custom_instructions, model=args.mem0_model)

    # ── Run ──
    all_results, all_add_stats, search_stats = run_inference(data, mem0, gen_model, args)

    save_json(all_results, os.path.join(run_dir, "raw_predictions.json"))
    save_json({"dataset": "locomo", "num_conversations": len(data), "num_questions": len(all_results),
               "mem0_model": args.mem0_model, "gen_model": args.model,
               "top_k": args.effective_top_k, "batch_size": args.batch_size},
              os.path.join(run_dir, "01_data_summary.json"))

    # ── Metrics ──
    compute_all_metrics(all_results, run_dir, all_add_stats, search_stats, nli_scorer=nli_scorer)
    print(f"\nResults saved to {run_dir}/")


if __name__ == "__main__":
    main()
