"""Preprocess dslighfdsl/alfworld_train_all into per-task trajectories.

Each row in source JSONL = (task, prefix of trajectory + next action).
Same task appears ~10 times with progressively longer prefixes.
We aggregate: for each file_name, keep the row with most messages —
that row contains the full gold trajectory.

Output:
    <out_dir>/demo_pool.json   (~2800 tasks)
    <out_dir>/eval.json        (~312 tasks, last 10%)

Each task record:
    {
        "task_id": "alfworld_0",
        "task_type": "put",   # inferred from goal phrase
        "goal": "put a toiletpaper in toiletpaperhanger",
        "initial_obs": "You are in the middle of a room. ...",
        "trajectory": [
            {"obs": "...", "action": "go to toiletpaperhanger 1"},
            {"obs": "On the toiletpaperhanger 1, you see nothing.",
             "action": "go to toilet 1"},
            ...
        ]
    }
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

# Repo root: $SUBMEM_ROOT, else derived from this file's location (scripts/..).
SUBMEM_ROOT = Path(os.environ.get("SUBMEM_ROOT", Path(__file__).resolve().parents[1]))
# Hugging Face cache root: $HF_HOME, else the HF default ~/.cache/huggingface.
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
# Root for processed datasets: $SUBMEM_DATA, else $SUBMEM_ROOT/data.
SUBMEM_DATA = Path(os.environ.get("SUBMEM_DATA", SUBMEM_ROOT / "data"))

# dslighfdsl/alfworld_train_all snapshot as downloaded into $HF_HOME.
SRC = str(HF_HOME / "datasets--dslighfdsl--alfworld_train_all/snapshots/d14da2b92125569cf229031e66e9ee98980c39dd/train_all_alfworld.jsonl")

_TASK_RE = re.compile(r"Your task is to:\s*(.*?)(?:\.|\n|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_CoT_RE = re.compile(r"<\|reserved_special_token_0\|>.*?<\|reserved_special_token_1\|>", re.DOTALL)

# ALFWorld task types, keyed by goal phrase prefix.
TASK_TYPES = [
    ("look", "look_at_obj_in_light"),
    ("find two", "pick_two_obj_and_place"),
    ("put two", "pick_two_obj_and_place"),
    ("examine", "look_at_obj_in_light"),
    ("heat", "pick_heat_then_place_in_recep"),
    ("cool", "pick_cool_then_place_in_recep"),
    ("clean", "pick_clean_then_place_in_recep"),
    ("put", "pick_and_place_simple"),
]


def infer_task_type(goal: str) -> str:
    low = goal.lower()
    for key, typ in TASK_TYPES:
        if key in low:
            return typ
    return "other"


def parse_trajectory(msgs):
    """Messages layout: [system, 'OK', obs+goal, action, obs, action, ...].
    Returns (goal, initial_obs, trajectory_steps) or (None, ...) on failure.
    """
    if len(msgs) < 4 or msgs[2]["role"] != "user":
        return None, None, None

    goal_and_obs = msgs[2]["content"].strip()
    m = _TASK_RE.search(goal_and_obs)
    if not m:
        return None, None, None
    goal = m.group(1).strip().rstrip(".")
    initial_obs = goal_and_obs[: m.start()].strip()

    steps = []
    last_obs = initial_obs
    for i in range(3, len(msgs), 2):
        if msgs[i]["role"] != "assistant":
            continue
        content = _CoT_RE.sub("", msgs[i]["content"]).strip()
        am = _ACTION_RE.search(content)
        if not am:
            continue
        action = am.group(1).strip().rstrip(".")
        steps.append({"obs": last_obs, "action": action})
        nxt = i + 1
        if nxt < len(msgs) and msgs[nxt]["role"] == "user":
            obs_text = msgs[nxt]["content"].strip()
            if obs_text.lower().startswith("observation:"):
                obs_text = obs_text[len("observation:"):].strip()
            last_obs = obs_text

    return goal, initial_obs, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-dir", default=str(SUBMEM_DATA / "alfworld" / "cleaned"))
    ap.add_argument("--eval-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # First pass: for each file_name, keep the row with most messages.
    best_by_task = {}
    n_rows = 0
    with open(args.src) as f:
        for line in f:
            r = json.loads(line)
            fn = r["file_name"]
            msgs = r["messages"]
            if fn not in best_by_task or len(msgs) > len(best_by_task[fn]["messages"]):
                best_by_task[fn] = {"file_name": fn, "messages": msgs}
            n_rows += 1

    print(f"Scanned {n_rows} rows, {len(best_by_task)} unique tasks")

    # Second pass: parse
    tasks = []
    skipped = 0
    for fn, r in best_by_task.items():
        goal, init_obs, steps = parse_trajectory(r["messages"])
        if goal is None or not steps:
            skipped += 1
            continue
        tasks.append({
            "task_id": fn,
            "task_type": infer_task_type(goal),
            "goal": goal,
            "initial_obs": init_obs,
            "trajectory": steps,
        })
    tasks.sort(key=lambda x: x["task_id"])
    print(f"Parsed {len(tasks)} tasks (skipped {skipped} malformed)")

    # Report task_type distribution
    from collections import Counter
    c = Counter(t["task_type"] for t in tasks)
    print("task_type distribution:", dict(c))

    # Trajectory length stats
    lens = [len(t["trajectory"]) for t in tasks]
    print(f"trajectory len: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")

    # Stratified split by task_type
    random.seed(args.seed)
    by_type = defaultdict(list)
    for t in tasks:
        by_type[t["task_type"]].append(t)
    eval_tasks, demo_tasks = [], []
    for typ, lst in by_type.items():
        random.shuffle(lst)
        n_eval = max(1, int(round(len(lst) * args.eval_fraction)))
        eval_tasks.extend(lst[:n_eval])
        demo_tasks.extend(lst[n_eval:])
    print(f"Split: eval={len(eval_tasks)}  demo_pool={len(demo_tasks)}")

    json.dump(eval_tasks, open(os.path.join(args.out_dir, "eval.json"), "w"), indent=2)
    json.dump(demo_tasks, open(os.path.join(args.out_dir, "demo_pool.json"), "w"), indent=2)
    print(f"Saved → {args.out_dir}/{{eval.json,demo_pool.json}}")


if __name__ == "__main__":
    main()
