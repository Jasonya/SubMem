"""Generate Qwen3-32B solutions on ALL BigCodeBench full tasks, in parallel.

Uses asyncio + vLLM OpenAI-compatible endpoint. 32 concurrent requests.
Skips tasks that already have a non-empty solution in the output file.

Usage:
  # on a compute node with vLLM running on port 8765:
  OPENAI_BASE_URL=http://localhost:8765/v1 OPENAI_API_KEY=dummy \
    python scripts/qwen_bcb_generate_parallel.py \
      --output $SUBMEM_RESULTS/qwen_bcb_gen/qwen3-32b-all.jsonl
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

# Hugging Face cache root: $HF_HOME, else the HF default ~/.cache/huggingface.
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def extract_code(text: str) -> str:
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


async def gen_one(client: AsyncOpenAI, model: str, task: dict, max_new: int) -> dict:
    prompt = task["instruct_prompt"] + " /no_think"
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert Python programmer. Write code only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_new,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        return {"task_id": task["task_id"], "solution": "", "error": str(e)[:200]}
    return {"task_id": task["task_id"], "solution": extract_code(raw)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--bcb-parquet",
                    default=str(HF_HOME / "hub/datasets--bigcode--bigcodebench/snapshots/b74c0d0bf70d2c0bc459be537895cca163007f1a/data/train-00000-of-00001.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(args.bcb_parquet)
    print(f"Loaded {len(df)} tasks", flush=True)

    # Resume: keep existing non-empty solutions
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("solution", "").strip():
                        done[d["task_id"]] = d
                except json.JSONDecodeError:
                    continue
        print(f"Resuming — {len(done)} already done", flush=True)

    todo = [row for _, row in df.iterrows() if row["task_id"] not in done]
    print(f"TODO: {len(todo)} tasks", flush=True)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    results = list(done.values())
    lock = asyncio.Lock()

    async def worker(task):
        async with sem:
            r = await gen_one(client, args.model, task, args.max_new_tokens)
        async with lock:
            results.append(r)
            if len(results) % 20 == 0:
                print(f"  {len(results)}/{len(df)} done", flush=True)
            # Append to file immediately so we can resume on failure
            with open(out_path, "a") as f:
                f.write(json.dumps(r) + "\n")

    tasks = [asyncio.create_task(worker(t)) for t in todo]
    await asyncio.gather(*tasks)
    print(f"DONE. total={len(results)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
