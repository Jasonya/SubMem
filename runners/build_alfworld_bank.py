"""Build a pre-populated memory bank for ALFWorld agents.

Sources:
  - Success (100): AgentGym/AgentTraj-L, filtered to ALFWorld, balanced across 6 task types
  - Failure (100): Heaplax/ARMAP-RM-ALFWorld preference pairs, taking the rejected side

Filters out any task prefix that appears in ALFWORLD_DATA/json_2.1.1/valid_unseen
to prevent eval-time leakage.

Output:
  results/bank_<Cx>.pkl — pickled memory unit ready to load into run_alfworld_rollout.py

Usage:
  python build_alfworld_bank.py --memory C1 C14 \\
      --alfworld-data $ALFWORLD_DATA --n-success 100 --n-failure 100
"""
import argparse
import json
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # parent of the SubMem package

from SubMem.config import MemoryConfig
from SubMem.memory import create_memory_unit


class _EmbedOnlyModel:
    """Module-level so pickle works. Used at bank-build time; swapped for the
    real model when the bank is loaded by run_alfworld_rollout.py."""

    def encode(self, s):
        return s.split()

    def generate(self, p, **k):
        return ""

ALF_TASK_TYPES = [
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
]


def classify_task_type(item_id: str) -> Optional[str]:
    for t in ALF_TASK_TYPES:
        if item_id.startswith(t):
            return t
    return None


def task_prefix_from_agentgym(item_id: str) -> str:
    """AgentGym id: 'look_at_obj_in_light-AlarmClock-None-DeskLamp-301_trial_T...'
    We want the prefix before '_trial_' which is the ALFWorld task dir name."""
    return item_id.split("_trial_", 1)[0]


def load_valid_unseen_prefixes(alfworld_data: str) -> set:
    vu = Path(alfworld_data) / "json_2.1.1" / "valid_unseen"
    return set(os.listdir(vu))


def extract_goal(conversations: List[Dict[str, str]]) -> str:
    """Goal is in the first environment description (human message)."""
    for m in conversations:
        v = m["value"]
        mm = re.search(r"Your task is to:\s*(.+?)(?:\.|\n|$)", v)
        if mm:
            return mm.group(1).strip()
    return ""


def format_trajectory(conversations: List[Dict[str, str]]) -> str:
    """Render ShareGPT conversation as a readable trajectory.
    Keep full content — length survey showed med ~400-470 tokens, p90 ~1200."""
    lines = []
    for m in conversations:
        role = m["from"]
        val = m["value"].strip()
        if not val:
            continue
        # Skip the long initial system/setup message to save tokens
        if role == "human" and len(val) > 800 and "you see" in val and "Your task" in val:
            # Keep only the room description + task line
            lines.append(val[:800] + "...")
            continue
        lines.append(f"[{role}] {val}")
    return "\n".join(lines)


# ── Sample AgentGym success ──────────────────────────────────────
def sample_agentgym_success(n: int, forbidden_prefixes: set, seed: int = 0) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("AgentGym/AgentTraj-L", split="train", streaming=True)

    # Group by task_type, prefer balanced sampling
    pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # Scan up to 10k rows to find all task types
    SCAN_LIMIT = 10000
    scanned = 0
    skipped_leak = 0
    for r in ds:
        scanned += 1
        if scanned > SCAN_LIMIT:
            break
        item_id = r["item_id"]
        t = classify_task_type(item_id)
        if t is None:
            continue
        prefix = task_prefix_from_agentgym(item_id)
        if prefix in forbidden_prefixes:
            skipped_leak += 1
            continue
        pool[t].append({
            "item_id": item_id,
            "conversations": r["conversations"],
            "task_type": t,
            "source": "agentgym",
            "success": True,
        })
    print(f"  [agentgym] scanned {scanned} rows, skipped {skipped_leak} leak")
    print(f"  [agentgym] pool sizes: { {t: len(pool[t]) for t in pool} }")

    # Balanced sample: n // n_types per type, topping up from largest pools
    rng = random.Random(seed)
    n_types = sum(1 for t in pool if pool[t])
    per_type = n // max(n_types, 1)
    samples = []
    for t, rows in pool.items():
        rng.shuffle(rows)
        samples.extend(rows[:per_type])
    # Top up to n from remaining
    rest = []
    for t, rows in pool.items():
        rest.extend(rows[per_type:])
    rng.shuffle(rest)
    samples.extend(rest[:n - len(samples)])
    return samples[:n]


# ── Sample Heaplax failure (rejected side of each pair) ──────────
def sample_heaplax_failure(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Heaplax/ARMAP-RM-ALFWorld",
                        "alfworld_preference.json", repo_type="dataset")
    data = json.load(open(p))
    rng = random.Random(seed)
    rng.shuffle(data)

    samples = []
    for r in data:
        pref = r["preference"]  # 1 => output_1 chosen; 2 => output_2 chosen
        rejected = r["output_2"] if pref == 1 else r["output_1"]
        conv = r["conversations"] + rejected  # prepend original setup msg
        # Reconstruct a ShareGPT-like trajectory using the original setup + rejected rollout
        item_id = f"heaplax_{r['id']}"
        # task_type: infer from goal text using keyword matching
        goal = extract_goal(conv)
        t = None
        g = goal.lower()
        if "look at" in g or "examine" in g: t = "look_at_obj_in_light"
        elif "two" in g: t = "pick_two_obj_and_place"
        elif "hot" in g or "heat" in g: t = "pick_heat_then_place_in_recep"
        elif "cool" in g: t = "pick_cool_then_place_in_recep"
        elif "clean" in g: t = "pick_clean_then_place_in_recep"
        else: t = "pick_and_place_simple"
        samples.append({
            "item_id": item_id,
            "conversations": conv,
            "task_type": t,
            "source": "heaplax_rejected",
            "success": False,
            "goal": goal,
        })
        if len(samples) >= n:
            break
    return samples


# ── Build ─────────────────────────────────────────────────────────
def build_bank(memory_id: str, samples: List[Dict[str, Any]], model) -> Any:
    cfg = MemoryConfig(memory_id=memory_id,
                       extra={"encoder_model": "text-embedding-3-small"})
    mem = create_memory_unit(cfg, model=model)

    for i, s in enumerate(samples):
        goal = s.get("goal") or extract_goal(s["conversations"])
        traj_text = format_trajectory(s["conversations"])
        summary = (f"Goal: {goal}\n"
                   f"Task type: {s['task_type']}\n"
                   f"Success: {s['success']}\n"
                   f"Trajectory:\n{traj_text}")
        md = {
            "task_type": s["task_type"],
            "source": s["source"],
            "success": s["success"],
            "goal": goal,
            "item_id": s["item_id"],
        }
        if hasattr(mem, "write_experience"):
            try:
                mem.write_experience(goal, summary, success=s["success"], metadata=md)
            except (NotImplementedError, TypeError):
                mem.write(summary, metadata=md)
        else:
            mem.write(summary, metadata=md)
        if (i + 1) % 25 == 0:
            print(f"    wrote {i+1}/{len(samples)} into {memory_id}")

    # Trigger lazy indexing (C14/C15 need this to run distill/controller)
    _ = mem.read("dummy probe", top_k=1)
    return mem


# ── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", nargs="+", default=["C1", "C14"],
                    help="Substrates to build banks for")
    ap.add_argument("--alfworld-data",
                    default=os.environ.get("ALFWORLD_DATA", ""),
                    help="Path to alfworld data (for valid_unseen filter)")
    ap.add_argument("--n-success", type=int, default=100)
    ap.add_argument("--n-failure", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__), "results", "alfworld_banks"))
    args = ap.parse_args()

    if not args.alfworld_data:
        raise SystemExit("Set --alfworld-data or $ALFWORLD_DATA")

    forbidden = load_valid_unseen_prefixes(args.alfworld_data)
    print(f"Filter: {len(forbidden)} valid_unseen prefixes blocked from bank")

    print(f"\n=== Sampling {args.n_success} successes from AgentGym ===")
    success = sample_agentgym_success(args.n_success, forbidden, args.seed)
    print(f"  got {len(success)}: { {t:sum(1 for s in success if s['task_type']==t) for t in ALF_TASK_TYPES} }")

    print(f"\n=== Sampling {args.n_failure} failures from Heaplax (rejected side) ===")
    failure = sample_heaplax_failure(args.n_failure, args.seed)
    print(f"  got {len(failure)}: { {t:sum(1 for s in failure if s['task_type']==t) for t in ALF_TASK_TYPES} }")

    all_samples = success + failure
    random.Random(args.seed).shuffle(all_samples)
    print(f"\nTotal: {len(all_samples)} samples")

    os.makedirs(args.out_dir, exist_ok=True)

    for mid in args.memory:
        print(f"\n=== Building bank for {mid} ===")
        model = _EmbedOnlyModel()
        mem = build_bank(mid, all_samples, model)
        size = mem.size() if hasattr(mem, "size") else None
        out_path = os.path.join(args.out_dir, f"bank_{mid}.pkl")
        _strip_unpicklable(mem)
        with open(out_path, "wb") as f:
            pickle.dump(mem, f)
        print(f"  {mid} bank size: {size}  -> {out_path}")


def _strip_unpicklable(mem) -> None:
    """Null out OpenAI clients / thread locks so the memory unit pickles cleanly.
    They are re-initialised lazily on next use after load."""
    import threading
    # Known attribute names that hold non-picklable resources
    for attr in ("model", "_client", "_openai_client", "_framework_llm",
                 "_controller_llm", "_executor_llm", "_designer_llm",
                 "_llm", "_embedder_client",
                 "_mem0", "_adder", "_searcher",          # C5 Mem0
                 "_graphiti", "_graph", "_driver",         # C6 Graphiti
                 "_amem", "_memsystem",                    # C16 A-Mem
                 ):
        if hasattr(mem, attr):
            try:
                setattr(mem, attr, None)
            except Exception:
                pass
    # Walk one level deeper — embedder / encoder objects often hold _client
    for attr_name in list(vars(mem).keys()):
        obj = getattr(mem, attr_name, None)
        if obj is None:
            continue
        for sub in ("_client", "_openai_client", "_thread", "_lock"):
            if hasattr(obj, sub):
                try:
                    setattr(obj, sub, None)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
