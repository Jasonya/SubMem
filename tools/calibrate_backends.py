"""Backend calibration: measure HF-transformers vs vLLM throughput ratio k.

Runs a fixed set of prompts on both backends (same model, same decoding params)
and reports:
    k_prefill     = HF_prefill_ms    / vLLM_prefill_ms
    k_decode      = HF_ms_per_token  / vLLM_ms_per_token
    k_end_to_end  = HF_total_ms      / vLLM_total_ms

k_end_to_end is the factor to divide HF-measured E2/E3/E4/E14 by to make them
comparable with vLLM-measured latencies.

Usage (single SLURM GPU job, vLLM must be runnable for the chosen model):
    python3 calibrate_backends.py --model Qwen/Qwen3-8B --vllm-port 8900
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


_PROMPTS = [
    # Representative of memory-benchmark workloads: short→long, Q&A flavor
    "What did the user say about their weekend plans?",
    "Summarize the key events from May 2023 in one paragraph.",
    "List three hobbies the speaker mentioned across these conversations.",
    "Given the previous context, what is the speaker's relationship to Caroline?",
    "Extract all dates mentioned in the following text: On Monday we went hiking, "
    "then on Tuesday we visited the museum, and by Friday we were back home.",
    "The speaker has been working on a novel for three years. The novel is about "
    "a detective in post-war Berlin. The speaker recently had writer's block. "
    "Based on this, what genre is the novel and what challenge is the speaker facing?",
    "Write a 3-sentence summary of the following: The user bought a house in 2020, "
    "renovated it in 2021, and is considering selling it in 2024 due to a job change "
    "that requires relocation to another city. They have mixed feelings about leaving.",
    "Given these memories: [1] User enjoys hiking. [2] User visited Yosemite in 2023. "
    "[3] User plans a trip to Patagonia. Answer: what outdoor activity does the user enjoy?",
]
# Token output budget — matches C13 rethink workload (long generations are the
# pain point where vLLM wins most)
OUTPUT_TOKENS = [256, 512, 1024, 2000]


def _tok_count(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


_HF_CACHE = {"model": None, "tok": None}


def run_hf(model_name: str, prompts, max_new_tokens: int):
    """Run HF transformers backend; return per-prompt metrics.

    Caches the loaded model across repeated invocations within the same process.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if _HF_CACHE["model"] is None:
        print(f"[HF] loading {model_name}...")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        _HF_CACHE["model"] = model
        _HF_CACHE["tok"] = tok
        print("[HF] loaded.")
    model = _HF_CACHE["model"]
    tok = _HF_CACHE["tok"]
    pad_id = tok.pad_token_id or tok.eos_token_id

    results = []
    for i, prompt in enumerate(prompts):
        # Apply chat template for fair comparison (vLLM server does the same)
        if tok.chat_template:
            msgs = [{"role": "user", "content": prompt}]
            text_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text_in = prompt
        inputs = tok(text_in, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]

        # Warmup
        if i == 0:
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=pad_id)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        n_out = out.shape[1] - n_in
        ms_per_tok = elapsed_ms / max(n_out, 1)
        results.append({
            "prompt_idx": i,
            "n_in": n_in,
            "n_out": n_out,
            "total_ms": elapsed_ms,
            "ms_per_token": ms_per_tok,
        })
        print(f"  [HF] prompt {i}: in={n_in} out={n_out} total={elapsed_ms:.0f}ms "
              f"per_tok={ms_per_tok:.1f}ms")

    return results


def release_hf():
    """Free GPU memory held by cached HF model. Call before starting vLLM."""
    import torch
    if _HF_CACHE["model"] is not None:
        del _HF_CACHE["model"]
        _HF_CACHE["model"] = None
        _HF_CACHE["tok"] = None
        torch.cuda.empty_cache()


def run_vllm(model_name: str, port: int, prompts, max_new_tokens: int):
    """Run vLLM OpenAI-compatible server; return per-prompt metrics.

    Assumes the server was already started externally on localhost:port.
    """
    from openai import OpenAI

    client = OpenAI(api_key="dummy", base_url=f"http://localhost:{port}/v1")
    print(f"[vLLM] using server at port {port}")

    # Sanity check server
    for _ in range(60):
        try:
            _ = client.models.list()
            break
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError(f"vLLM server on port {port} not responding")

    results = []
    for i, prompt in enumerate(prompts):
        # Warmup first request
        if i == 0:
            _ = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8, temperature=0.0,
            )
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens, temperature=0.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        n_in = usage.prompt_tokens if usage else -1
        n_out = usage.completion_tokens if usage else -1
        ms_per_tok = elapsed_ms / max(n_out, 1)
        results.append({
            "prompt_idx": i,
            "n_in": n_in,
            "n_out": n_out,
            "total_ms": elapsed_ms,
            "ms_per_token": ms_per_tok,
        })
        print(f"  [vLLM] prompt {i}: in={n_in} out={n_out} total={elapsed_ms:.0f}ms "
              f"per_tok={ms_per_tok:.1f}ms")
    return results


def compute_k(hf_results, vllm_results):
    """Compute k ratios from parallel HF and vLLM runs."""
    # end-to-end total time ratio (weighted by output length to match realistic workloads)
    hf_total = sum(r["total_ms"] for r in hf_results)
    vllm_total = sum(r["total_ms"] for r in vllm_results)
    k_total = hf_total / max(vllm_total, 1e-9)

    # Per-token decode speed ratio (median across prompts — robust)
    hf_per_tok = statistics.median([r["ms_per_token"] for r in hf_results if r["n_out"] > 0])
    vllm_per_tok = statistics.median([r["ms_per_token"] for r in vllm_results if r["n_out"] > 0])
    k_decode = hf_per_tok / max(vllm_per_tok, 1e-9)

    return {
        "k_total": round(k_total, 3),
        "k_decode": round(k_decode, 3),
        "hf_total_ms": round(hf_total, 1),
        "vllm_total_ms": round(vllm_total, 1),
        "hf_median_ms_per_tok": round(hf_per_tok, 2),
        "vllm_median_ms_per_tok": round(vllm_per_tok, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--vllm-port", type=int, default=8900)
    ap.add_argument("--output", default=None)
    ap.add_argument("--backend", choices=["hf", "vllm", "both"], default="both")
    args = ap.parse_args()

    # Generate prompt × max_tokens matrix
    prompts_expanded = []
    for p in _PROMPTS:
        for t in OUTPUT_TOKENS:
            prompts_expanded.append((p, t))
    print(f"Running {len(prompts_expanded)} (prompt, max_tokens) pairs per backend.")

    report = {"model": args.model, "num_trials": len(prompts_expanded)}

    if args.backend in ("vllm", "both"):
        # vLLM first while GPU is free
        vllm_all = []
        for p, t in prompts_expanded:
            r = run_vllm(args.model, args.vllm_port, [p], t)[0]
            r["max_new_tokens"] = t
            vllm_all.append(r)
        report["vllm"] = vllm_all

    if args.backend in ("hf", "both"):
        hf_all = []
        for p, t in prompts_expanded:
            r = run_hf(args.model, [p], t)[0]
            r["max_new_tokens"] = t
            hf_all.append(r)
        report["hf"] = hf_all

    if args.backend == "both":
        report["calibration"] = compute_k(report["hf"], report["vllm"])
        print("\n=== CALIBRATION SUMMARY ===")
        for k, v in report["calibration"].items():
            print(f"  {k}: {v}")

    out = args.output or f"calibration_{args.model.replace('/', '_')}.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
