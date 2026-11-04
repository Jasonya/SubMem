"""Filter nebius/SWE-agent-trajectories into a small experience pool for C14.

Selects 100 success + 100 failure trajectories, balanced across the 3 models
in the dataset, with trajectory length capped so context stays manageable.

Output columns: instance_id, model_name, target (bool), trajectory_text (str),
                query (str = instance_id, to be replaced later with problem_statement).

Usage:
  python scripts/prepare_nebius_experiences.py
"""
import argparse
import glob
import os
import random
import re
from pathlib import Path

import pandas as pd


_ISSUE_RE = re.compile(r"ISSUE:\s*\n(.*?)(?:\n\s*INSTRUCTIONS:|\Z)", re.DOTALL)


def _extract_query(traj) -> str:
    """Pull the ISSUE block from the first user turn as the task query."""
    if traj is None:
        return ""
    for step in traj:
        if step.get("role") != "user":
            continue
        text = step.get("text") or ""
        m = _ISSUE_RE.search(text)
        if m:
            return m.group(1).strip()
        return text.strip()[:2000]
    return ""


# Repo root: $SUBMEM_ROOT, else derived from this file's location (scripts/..).
SUBMEM_ROOT = Path(os.environ.get("SUBMEM_ROOT", Path(__file__).resolve().parents[1]))
# Root of the nebius/SWE-agent-trajectories download: $NEBIUS_DIR, else
# $SUBMEM_ROOT/data/nebius_swe_trajectories. Parquet shards live in its data/
# subdirectory; the balanced experience pool is written next to them.
NEBIUS_ROOT = Path(os.environ.get("NEBIUS_DIR", SUBMEM_ROOT / "data" / "nebius_swe_trajectories"))
NEBIUS_DIR = str(NEBIUS_ROOT / "data")
OUTPUT_PATH = str(NEBIUS_ROOT / "experiences_balanced.parquet")


def _trajectory_to_text(traj, max_steps: int, max_chars_per_step: int) -> str:
    """Flatten a trajectory list into ``[role] text\n`` lines.

    Skips system prompts (role='system') to keep the text focused on
    the agent's reasoning and environment feedback.
    """
    if traj is None:
        return ""
    lines = []
    steps = 0
    for step in traj:
        role = step.get("role", "")
        if role == "system":
            continue
        text = step.get("text") or ""
        if not text:
            continue
        if len(text) > max_chars_per_step:
            text = text[:max_chars_per_step] + "... [truncated]"
        lines.append(f"[{role}] {text}")
        steps += 1
        if steps >= max_steps:
            break
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-success", type=int, default=100)
    ap.add_argument("--n-failure", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-chars-per-step", type=int, default=2000)
    ap.add_argument("--max-total-chars", type=int, default=40000,
                    help="Drop trajectories whose serialized text exceeds this.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=OUTPUT_PATH)
    ap.add_argument("--input-dir", default=NEBIUS_DIR,
                    help="Directory containing nebius train-*.parquet shards.")
    args = ap.parse_args()

    random.seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    print(f"Loading {len(files)} parquet shards...", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df)} rows, {df.instance_id.nunique()} unique instances, "
          f"{df.model_name.nunique()} models", flush=True)

    # Only keep submitted runs (exclude crashes / context blowouts)
    df = df[df.exit_status.astype(str).str.startswith("submitted")]
    print(f"  after exit_status filter: {len(df)} rows", flush=True)

    print("Serializing trajectories and extracting queries...", flush=True)
    df["query"] = df.trajectory.apply(_extract_query)
    df["trajectory_text"] = df.trajectory.apply(
        lambda t: _trajectory_to_text(t, args.max_steps, args.max_chars_per_step)
    )
    df = df[df["query"].str.len() > 20]
    print(f"  after query-extraction filter: {len(df)} rows", flush=True)
    df["traj_len"] = df.trajectory_text.str.len()
    df = df[(df.traj_len > 200) & (df.traj_len < args.max_total_chars)]
    print(f"  after length filter: {len(df)} rows "
          f"(len range {df.traj_len.min()}-{df.traj_len.max()})", flush=True)

    # Balanced sampling per model
    models = sorted(df.model_name.unique().tolist())
    n_per_model_success = args.n_success // len(models)
    n_per_model_failure = args.n_failure // len(models)

    picked = []
    for m in models:
        dm = df[df.model_name == m]
        succ = dm[dm.target].sample(
            n=min(n_per_model_success, (dm.target).sum()),
            random_state=args.seed,
        )
        fail = dm[~dm.target].sample(
            n=min(n_per_model_failure, (~dm.target).sum()),
            random_state=args.seed,
        )
        picked.append(succ)
        picked.append(fail)
        print(f"  {m}: picked {len(succ)} success + {len(fail)} failure", flush=True)

    out = pd.concat(picked, ignore_index=True)
    # Keep only columns we need
    out = out[["instance_id", "model_name", "target", "query", "trajectory_text", "generated_patch"]].copy()
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"\nWrote {len(out)} experiences → {args.output}", flush=True)
    print(f"  success={out.target.sum()}, failure={(~out.target).sum()}", flush=True)
    print(f"  avg trajectory length: {out.trajectory_text.str.len().mean():.0f} chars", flush=True)


if __name__ == "__main__":
    main()
