"""Build experiences_bcb.parquet: balanced (task, solution, passed) pool.

Inputs:
  - BigCodeBench main dataset (HF bigcode/bigcodebench)  — for query (instruct_prompt)
  - One or more model sample JSONL + matching *_eval_results.json

Output schema (column names align with nebius SWE-bench pool):
  task_id, model_name, target (bool), query, trajectory_text, generated_patch

Here:
  target          = eval status == "pass"
  query           = instruct_prompt (task description)
  trajectory_text = solution (the code)
  generated_patch = same as trajectory_text (kept for schema compatibility)

Usage:
  python scripts/build_bcb_experiences.py \
    --samples <model1.jsonl> <model2.jsonl> ... \
    --n-success 100 --n-failure 100 \
    --output results/experiences_bcb.parquet
"""
import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def _model_from_sample_path(p: str) -> str:
    """Heuristic: pull a short model name out of the sample filename."""
    base = Path(p).stem
    # e.g. 'gpt-4o-mini-2024-07-18--bigcodebench-instruct--openai-0-1-sanitized-calibrated'
    # Take the piece before '--bigcodebench'
    m = re.match(r"(.+?)--bigcodebench", base)
    return m.group(1) if m else base[:40]


def load_samples_with_status(sample_path: str) -> pd.DataFrame:
    """Join a sample JSONL with its *_eval_results.json to get pass/fail per task."""
    sample_path = Path(sample_path)
    # Eval results live next to the sample with _eval_results.json suffix
    # The file extension is .jsonl; strip it and add the suffix.
    results_path = sample_path.with_name(
        sample_path.stem + "_eval_results.json"
    )
    if not results_path.exists():
        # Some filenames use '-sanitized-calibrated' vs '_sanitized_calibrated' — try both
        alt = sample_path.with_name(
            sample_path.stem.replace("-sanitized-calibrated", "-sanitized_calibrated")
            + "_eval_results.json"
        )
        if alt.exists():
            results_path = alt
        else:
            raise FileNotFoundError(f"No eval_results.json for {sample_path}")

    # Load samples (task_id, solution)
    samples = []
    with open(sample_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples_df = pd.DataFrame(samples)

    # Load eval results → flat per-task status
    eval_obj = json.load(open(results_path))
    eval_map = eval_obj.get("eval", {})
    records = []
    for task_id, runs in eval_map.items():
        if isinstance(runs, list) and runs:
            status = runs[0].get("status", "?")
        elif isinstance(runs, dict):
            status = runs.get("status", "?")
        else:
            status = "?"
        records.append({"task_id": task_id, "status": status})
    eval_df = pd.DataFrame(records)

    merged = samples_df.merge(eval_df, on="task_id", how="left")
    merged["model_name"] = _model_from_sample_path(sample_path)
    merged["target"] = merged["status"] == "pass"
    return merged[["task_id", "model_name", "target", "solution", "status"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True,
                    help="One or more model sample JSONL files (with eval_results.json siblings)")
    ap.add_argument("--n-success", type=int, default=100)
    ap.add_argument("--n-failure", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--bcb-parquet",
        default=None,
        help="Path to bigcode/bigcodebench main parquet for instruct_prompt lookup. "
             "If None, auto-resolve via HF cache.",
    )
    args = ap.parse_args()
    random.seed(args.seed)

    # Load BCB main dataset for query text
    if args.bcb_parquet:
        bcb = pd.read_parquet(args.bcb_parquet)
    else:
        from huggingface_hub import snapshot_download
        p = snapshot_download(repo_id="bigcode/bigcodebench", repo_type="dataset",
                              allow_patterns=["*.parquet"])
        pfile = list(Path(p).rglob("*.parquet"))[0]
        bcb = pd.read_parquet(pfile)
    print(f"BCB main: {len(bcb)} tasks, cols={list(bcb.columns)}")

    task2query = dict(zip(bcb["task_id"], bcb["instruct_prompt"]))

    # Load and merge all model samples
    all_df = []
    for s in args.samples:
        df = load_samples_with_status(s)
        print(f"  {Path(s).name}: {len(df)} rows, "
              f"pass={df.target.sum()}, fail={(~df.target).sum()}")
        all_df.append(df)
    merged = pd.concat(all_df, ignore_index=True)
    merged["query"] = merged["task_id"].map(task2query)
    merged = merged.dropna(subset=["query", "solution"])
    # Drop trivially empty solutions
    merged = merged[merged["solution"].str.len() > 20]
    print(f"\nTotal after filter: {len(merged)} "
          f"(pass={merged.target.sum()}, fail={(~merged.target).sum()})")

    # Balanced sample
    succ_pool = merged[merged.target]
    fail_pool = merged[~merged.target]
    n_succ = min(args.n_success, len(succ_pool))
    n_fail = min(args.n_failure, len(fail_pool))
    succ = succ_pool.sample(n=n_succ, random_state=args.seed)
    fail = fail_pool.sample(n=n_fail, random_state=args.seed)
    out = pd.concat([succ, fail], ignore_index=True).sample(
        frac=1, random_state=args.seed
    ).reset_index(drop=True)

    # Align with nebius pool schema for driver compatibility
    out["trajectory_text"] = out["solution"]
    out["generated_patch"] = out["solution"]
    out = out[["task_id", "model_name", "target", "query",
               "trajectory_text", "generated_patch"]]

    out.to_parquet(args.output, index=False)
    print(f"\nWrote {len(out)} experiences → {args.output}")
    print(f"  pass={out.target.sum()}, fail={(~out.target).sum()}")
    print(f"  avg solution len: {out.trajectory_text.str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()
