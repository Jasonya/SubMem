"""ALFWorld Reflexion-style rollout runner (isolated from QA benchmarks).

Launches the live TextWorld environment and measures **task success rate** on
134 valid_unseen tasks — the canonical ALFWorld metric.

Memory model (Option 2, within-episode):
    For each task the memory unit is reset, then grows step-by-step as the
    agent interacts. At each step we retrieve relevant past
    (obs, action, result) triples and feed them back as grounding so the
    agent avoids repeating dead ends.

Usage:
    python run_alfworld_rollout.py --memory C1 C2 \\
        --model local/Qwen/Qwen3-8B --max-steps 50 --top-k 5

Results saved under:
    $SUBMEM_RESULTS/<ts>_alfworld_<model>/
        <Cx>/per_task.json    — task_id, success, steps, final_reward, trajectory
        <Cx>/summary.json     — aggregated success/step stats
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # parent of the SubMem package

from SubMem.config import ModelConfig, MemoryConfig
from SubMem.models import create_model
from SubMem.memory import create_memory_unit

RESULTS_BASE = os.environ.get(
    "SUBMEM_RESULTS", os.path.join(os.path.dirname(__file__), "results")
)


# ── env config (single source of truth) ─────────────────────────
def _alfworld_config(data_root: str) -> Dict[str, Any]:
    return {
        "dataset": {
            "data_path": f"{data_root}/json_2.1.1/train",
            "eval_id_data_path": f"{data_root}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{data_root}/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "env": {
            "type": "AlfredTWEnv",
            "regen_game_files": False,
            "domain_randomization": False,
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
            "hybrid": {"start_eps": 100000, "thor_prob": 0.5, "eval_mode": "tw"},
            "task_types": [1, 2, 3, 4, 5, 6],
        },
        "general": {
            "random_seed": 42,
            "training_method": "dqn",
            "hide_init_receptacles": False,
            "training": {"batch_size": 1},
            "evaluate": {"run_eval": True, "batch_size": 1, "env": {"type": "AlfredTWEnv"}},
        },
        "rl": {
            "action_space": "admissible",
            "max_target_length": 20,
            "beam_width": 10,
            "training": {"max_nb_steps_per_episode": 50,
                         "learn_start_from_this_episode": 0},
        },
    }


# ── prompt template ─────────────────────────────────────────────
_TASK_RE = re.compile(r"Your task is to:\s*(.*?)(?:\.|\n|$)", re.DOTALL)


def extract_goal(obs: str) -> str:
    m = _TASK_RE.search(obs)
    return m.group(1).strip() if m else ""


# Canonical procedures per task type — scaffolding the model doesn't learn
# from the admissible list alone (Qwen3-8B tends to place object without
# first applying cool/heat/clean). Keyed by goal keywords.
_TASK_PROCEDURES = {
    "look": (
        "Task type: look-at-object-in-light.\n"
        "Procedure:\n"
        "  1. locate the target object (search likely receptacles: desk, drawer, sidetable, dresser)\n"
        "  2. take <object> from its receptacle\n"
        "  3. go to the receptacle holding the desklamp\n"
        "  4. use desklamp  ← this completes the task\n"
        "  DO NOT turn on the lamp before taking the object.\n"
    ),
    "clean": (
        "Task type: clean-then-place.\n"
        "Procedure:\n"
        "  1. find and take the target object\n"
        "  2. go to sinkbasin\n"
        "  3. clean <object> with sinkbasin  ← must do this before placing\n"
        "  4. go to the target receptacle, open it if closed\n"
        "  5. put <object> in/on <target>\n"
    ),
    "cool": (
        "Task type: cool-then-place.\n"
        "Procedure:\n"
        "  1. find and take the target object\n"
        "  2. go to fridge, open fridge\n"
        "  3. cool <object> with fridge  ← must do this before placing\n"
        "  4. go to the target receptacle\n"
        "  5. put <object> in/on <target>\n"
    ),
    "heat": (
        "Task type: heat-then-place.\n"
        "Procedure:\n"
        "  1. find and take the target object\n"
        "  2. go to microwave, open microwave\n"
        "  3. heat <object> with microwave  ← must do this before placing\n"
        "  4. go to the target receptacle\n"
        "  5. put <object> in/on <target>\n"
    ),
    "two": (
        "Task type: pick-two-and-place (two copies of same object).\n"
        "Procedure:\n"
        "  Repeat twice: find a copy, take it, go to target receptacle, put it in/on target.\n"
        "  DO NOT take objects back out once placed.\n"
        "  Find DIFFERENT instances (e.g. soapbar 1 and soapbar 3, not the same one).\n"
    ),
    "put": (  # pick_and_place_simple — fallback, so keep last in matching order
        "Task type: pick-and-place (simple).\n"
        "Procedure:\n"
        "  1. find and take the target object\n"
        "  2. go to the target receptacle, open it if closed\n"
        "  3. put <object> in/on <target>\n"
    ),
}


def get_procedure(goal: str) -> str:
    """Match goal text to a canonical procedure (ordered keyword priority)."""
    g = goal.lower()
    # Ordering matters: 'clean' / 'cool' / 'heat' checked before 'put' fallback.
    for key in ("look", "two", "clean", "cool", "heat", "put"):
        if key in g:
            return _TASK_PROCEDURES[key]
    return ""


# Goal templates:
#   "look at bowl under the desklamp"  -> target=bowl
#   "put a cool tomato in microwave"    -> target=tomato, dest=microwave
#   "put two soapbar in garbagecan"     -> target=soapbar, dest=garbagecan
#   "put a clean soapbar in countertop" -> target=soapbar, dest=countertop
#   "put a hot potato in dining table"  -> target=potato, dest=dining table
#   "find a X, clean it with Y, put in Z" (Heaplax format) covered by simple noun
_STOPWORDS = {"a", "an", "the", "two", "some", "cool", "hot", "clean", "cold",
              "heated", "cooled", "warm", "cleaned", "at", "on", "in", "into",
              "under", "put", "place", "find", "look", "examine",
              "heat", "cooling", "heating", "cleaning",
              "it", "with", "to", "and", "then"}


def extract_target_object(goal: str) -> str:
    """Pull the first content noun from the goal — heuristic, not exact."""
    # Strip common sentence starters
    g = goal.lower()
    g = re.sub(r"^(look at|put|place|find|examine)\s+", "", g)
    # Tokenize on spaces
    tokens = re.findall(r"[a-z]+", g)
    for tok in tokens:
        if tok not in _STOPWORDS and len(tok) >= 2:
            return tok
    return ""


def extract_destination(goal: str) -> str:
    """Pull the destination receptacle (everything after 'in' / 'on')."""
    m = re.search(r"\b(?:in|on|under|into)\s+(?:the\s+)?([a-z ]+?)(?:\.|$)", goal.lower())
    if m:
        dest = m.group(1).strip()
        # Cut at trailing words that aren't part of receptacle name
        return dest.split(",")[0].strip()
    return ""


def build_prompt(goal: str, current_obs: str, admissible: List[str],
                 success_refs: str, failure_refs: str,
                 recent_history: List[Dict[str, str]],
                 use_hints: bool = True) -> str:
    history_lines = []
    for h in recent_history[-6:]:
        history_lines.append(f"Action: {h['action']}")
        history_lines.append(f"Observation: {h['obs']}")
    history_str = "\n".join(history_lines) if history_lines else "(no prior actions yet)"

    ref_block = ""
    if success_refs.strip() or failure_refs.strip():
        ref_block = "=== PAST EXPERIENCE (contrast carefully) ===\n\n"
        if success_refs.strip():
            ref_block += f"{success_refs}\n\n"
        if failure_refs.strip():
            ref_block += f"{failure_refs}\n\n"
        ref_block += "=== END PAST EXPERIENCE ===\n\n"

    # Hints can be disabled for failure-collection runs (we want natural
    # Qwen3-32B failures, not hint-scaffolded successes).
    if use_hints:
        procedure = get_procedure(goal)
        proc_block = f"{procedure}\n" if procedure else ""
        target = extract_target_object(goal)
        dest = extract_destination(goal)
        ground_block = ""
        if target:
            ground_block = f"Target object: {target} (ignore other objects).\n"
            if dest:
                ground_block += f"Destination: {dest}.\n"
            ground_block += "\n"
    else:
        proc_block = ""
        ground_block = ""

    return (
        "You are an agent in a simulated household. You must complete the task "
        "by issuing one action at a time. Each action must be a valid ALFWorld "
        "command such as `go to <receptacle>`, `take <obj> from <recep>`, "
        "`put <obj> in/on <recep>`, `open <recep>`, `close <recep>`, "
        "`toggle <obj> <recep>`, `clean <obj> with <recep>`, "
        "`heat <obj> with <recep>`, `cool <obj> with <recep>`, "
        "`use <obj>`.\n"
        "If you have not moved to a receptacle yet, your first action must be "
        "`go to <receptacle>`.\n\n"
        f"Task: {goal}\n"
        f"{ground_block}"
        f"{proc_block}"
        f"{ref_block}"
        f"Recent trajectory:\n{history_str}\n\n"
        f"Current observation:\n{current_obs}\n\n"
        f"Admissible actions (choose exactly one, copy verbatim):\n"
        + "\n".join(f"- {a}" for a in admissible[:40])
        + "\n\nOutput the chosen action on one line starting with `Action:`. /no_think\n"
        "Action:"
    )


def retrieve_balanced(memory, goal: str, k_success: int, k_failure: int
                      ) -> Tuple[List[Any], List[Any]]:
    """Retrieve top-K success and top-K failure separately from the bank.

    Heuristic: pull 4x k total, then partition by metadata.success. Substrates
    without success metadata (e.g. distilled reasoning items) return all as
    "success_refs"."""
    pool_size = max(k_success + k_failure, 20)
    pool = memory.read(goal, top_k=pool_size)
    # C4-style substrates merge all retrieved content into a single
    # MemoryEntry, which breaks per-entry balanced retrieval. Fall back to
    # iterating the underlying _entries and scoring with the same embedder.
    if len(pool) == 1 and hasattr(memory, "_entries") and len(memory._entries) > k_success + k_failure:
        md = getattr(pool[0], "metadata", None) or {}
        if "success" not in md:
            return _retrieve_balanced_raw(memory, goal, k_success, k_failure)
    succ, fail = [], []
    for e in pool:
        md = getattr(e, "metadata", None) or {}
        if md.get("success") is True:
            if len(succ) < k_success:
                succ.append(e)
        elif md.get("success") is False:
            if len(fail) < k_failure:
                fail.append(e)
        else:
            if len(succ) < k_success:
                succ.append(e)
    return succ, fail


def _retrieve_balanced_raw(memory, goal: str, k_success: int, k_failure: int
                           ) -> Tuple[List[Any], List[Any]]:
    """Direct FAISS search over memory._entries, bypassing substrate-specific
    read() aggregation. For substrates like C4 that merge retrieved content."""
    import numpy as np
    if not hasattr(memory, "_encode") or not hasattr(memory, "_index"):
        return [], []
    q_emb = memory._encode([goal])
    k = min(k_success + k_failure + 20, memory._index.ntotal)
    if k <= 0:
        return [], []
    scores, indices = memory._index.search(q_emb, k)
    succ, fail = [], []
    for idx in indices[0]:
        if idx < 0 or idx >= len(memory._entries):
            continue
        e = memory._entries[idx]
        md = e.metadata or {}
        if md.get("success") is True and len(succ) < k_success:
            succ.append(e)
        elif md.get("success") is False and len(fail) < k_failure:
            fail.append(e)
        if len(succ) >= k_success and len(fail) >= k_failure:
            break
    return succ, fail


def format_refs(entries: List[Any], label: str = "", max_chars: int = 2000) -> str:
    """Render retrieved entries with an explicit SUCCESS/FAILURE tag on each.

    max_chars: per-entry truncation budget. Pass -1 (or <=0) to disable
    truncation and surface the full entry content.
    """
    out = []
    for e in entries:
        md = getattr(e, "metadata", None) or {}
        goal = md.get("goal", "")
        raw = e.content or ""
        body = raw if max_chars is None or max_chars <= 0 else raw[:max_chars]
        tag = ""
        if label == "success":
            tag = "[SUCCESS EXAMPLE — IMITATE THIS STRATEGY]"
        elif label == "failure":
            tag = "[FAILURE EXAMPLE — DO NOT IMITATE, AVOID THIS PATTERN]"
        header = f"{tag}\nGoal: {goal}" if tag else f"- Goal: {goal}"
        out.append(f"{header}\n{body}")
    return "\n\n".join(out)


# ── action parsing ──────────────────────────────────────────────
def parse_action(raw: str, admissible: List[str]) -> str:
    """Extract an admissible action from free-form model output.

    Strategy:
      1. Grab the line after the first `Action:` marker.
      2. Exact match against admissible — return it.
      3. Lowercase + stripped-of-punct match.
      4. Substring fuzzy match (first admissible contained in text).
      5. Fallback: first admissible (usually `go to cabinet 1`) — not ideal
         but keeps the episode moving.
    """
    text = raw
    if "Action:" in text:
        text = text.split("Action:", 1)[1]
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    line = line.strip("`.\"' ")
    if line in admissible:
        return line
    low = line.lower().rstrip(".")
    for a in admissible:
        if a.lower() == low:
            return a
    # Drop leading "i will " etc.
    for prefix in ("i will ", "let me ", "next i will ", "i should "):
        if low.startswith(prefix):
            low = low[len(prefix):]
    for a in admissible:
        al = a.lower()
        if al in low or low in al:
            return a
    return admissible[0]


# ── rollout one task ────────────────────────────────────────────
def run_task_online(env, memory, model, max_steps: int, top_k: int,
                    max_new_tokens: int = 512, use_hints: bool = True) -> Dict[str, Any]:
    """Within-episode memory: fresh `memory` per task, write each step, retrieve each step.

    No pre-built bank. Caller must pass a *fresh* memory unit per task.
    """
    obs, info = env.reset()
    goal = extract_goal(obs[0])
    history: List[Dict[str, str]] = []

    for step in range(max_steps):
        admissible = info["admissible_commands"][0]

        # Retrieve relevant prior (obs, action) triples from this episode's memory
        query = f"Goal: {goal}\nCurrent observation: {obs[0]}"
        try:
            entries = memory.read(query, top_k=top_k)
        except Exception as _e:
            # Do NOT swallow silently: a substrate whose read() always
            # throws (e.g. M11/EpiCache needing model._model on an API
            # backend) would otherwise run as an invisible no-memory
            # baseline. Surface it so such misconfigurations are caught.
            entries = []
            print(f"    [online read failed: {type(_e).__name__}: {_e}]", flush=True)
        refs_text = ""
        if entries:
            lines = []
            for e in entries[:top_k]:
                # MemoryEntry exposes `.content` (not `.text`); falling back to
                # str(e) would inject the object repr instead of the memory.
                txt = getattr(e, "content", None) or getattr(e, "text", None) or str(e)
                lines.append(f"- {txt}")
            refs_text = ("PRIOR ATTEMPTS IN THIS EPISODE (use to avoid repeating "
                         "dead ends):\n" + "\n".join(lines))
        if os.environ.get("SUBMEM_ALF_DEBUG_READ"):
            print(f"    [read] step obs -> {len(entries)} entries", flush=True)

        prompt = build_prompt(goal, obs[0], admissible,
                              success_refs=refs_text, failure_refs="",
                              recent_history=history, use_hints=use_hints)
        t0 = time.time()
        raw = model.generate(prompt, max_new_tokens=max_new_tokens)
        gen_ms = (time.time() - t0) * 1000
        action = parse_action(raw, admissible)
        obs_new, scores, dones, info = env.step([action])
        next_obs = obs_new[0]

        # Write step into memory so later steps can retrieve it
        mem_text = f"step {step}: obs={obs[0][:300]} | action={action} | result={next_obs[:300]}"
        try:
            memory.write(mem_text, metadata={"step": step, "action": action})
        except Exception as e:
            print(f"    [online write failed at step {step}: {e}]", flush=True)

        history.append({"action": action, "obs": next_obs, "raw": raw, "gen_ms": gen_ms})
        obs = obs_new
        if dones[0]:
            break

    success = bool(scores[0]) and bool(dones[0])
    return {
        "goal": goal,
        "success": success,
        "steps": len(history),
        "final_score": float(scores[0]),
        "trajectory": history,
        "n_success_refs": 0,
        "n_failure_refs": 0,
        "online": True,
    }


def run_task(env, memory, model, max_steps: int, k_success: int, k_failure: int,
             max_new_tokens: int = 512, use_hints: bool = True,
             ref_max_chars: int = 2000) -> Dict[str, Any]:
    # NOTE: memory is pre-populated from an offline bank. Do NOT reset between
    # tasks — the bank is shared across all 134 tasks as static reference.
    obs, info = env.reset()
    goal = extract_goal(obs[0])
    history: List[Dict[str, str]] = []

    # One retrieve at episode start (bank is static, no need to re-query per step)
    succ_entries, fail_entries = retrieve_balanced(memory, goal, k_success, k_failure)
    succ_refs = format_refs(succ_entries, label="success", max_chars=ref_max_chars)
    fail_refs = format_refs(fail_entries, label="failure", max_chars=ref_max_chars)

    # Override the substrate-reported `read_context_tokens` with what actually
    # enters the prompt (top-k after format_refs trimming). The base class
    # accumulates pool-level tokens into `read_context_tokens_pool`, which is
    # preserved for recall analysis; this reassignment makes E7 reflect the
    # real prompt cost per query for ALFWorld. Other benchmarks that feed the
    # full pool should not reach this branch.
    try:
        prompt_refs_tok = len(model.encode(succ_refs)) + len(model.encode(fail_refs))
        if hasattr(memory, "_token_counts"):
            tc = memory._token_counts
            if "read_context_tokens_pool" in tc:
                # Subtract the double-counted pool contribution added by the
                # just-completed read() call, then add prompt-actual tokens.
                # We approximate by overwriting to accumulated pool + per-query
                # delta: because we cannot cheaply know the per-call pool size,
                # we instead track prompt tokens in a separate key and let the
                # runner collation prefer it when present.
                tc.setdefault("read_context_tokens_prompt", 0)
                tc["read_context_tokens_prompt"] += prompt_refs_tok
    except Exception:
        pass

    for step in range(max_steps):
        admissible = info["admissible_commands"][0]

        prompt = build_prompt(goal, obs[0], admissible, succ_refs, fail_refs,
                              history, use_hints=use_hints)
        t0 = time.time()
        raw = model.generate(prompt, max_new_tokens=max_new_tokens)
        gen_ms = (time.time() - t0) * 1000
        action = parse_action(raw, admissible)
        obs_new, scores, dones, info = env.step([action])
        next_obs = obs_new[0]

        history.append({"action": action, "obs": next_obs, "raw": raw, "gen_ms": gen_ms})

        obs = obs_new
        if dones[0]:
            break

    success = bool(scores[0]) and bool(dones[0])
    return {
        "goal": goal,
        "success": success,
        "steps": len(history),
        "final_score": float(scores[0]),
        "trajectory": history,
        "n_success_refs": len(succ_entries),
        "n_failure_refs": len(fail_entries),
    }


# ── main ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", nargs="+", default=["C1"],
                    help="Memory unit IDs, e.g. C1 C2")
    ap.add_argument("--model", default="local/Qwen/Qwen3-8B")
    ap.add_argument("--alfworld-data",
                    default=os.environ.get("ALFWORLD_DATA", ""),
                    help="ALFWorld data dir (else reads $ALFWORLD_DATA)")
    ap.add_argument("--max-tasks", type=int, default=134)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--k-success", type=int, default=3,
                    help="# success reference trajectories retrieved per task")
    ap.add_argument("--k-failure", type=int, default=3,
                    help="# failure reference trajectories retrieved per task")
    ap.add_argument("--split", default="eval_out_of_distribution",
                    choices=["eval_in_distribution", "eval_out_of_distribution",
                             "train"])
    ap.add_argument("--no-hints", action="store_true",
                    help="Disable procedure+grounding hints (for failure collection)")
    ap.add_argument("--encoder", default="text-embedding-3-small")
    ap.add_argument("--bank-dir",
                    default=os.path.join(os.path.dirname(__file__),
                                         "results", "alfworld_banks"),
                    help="Directory containing bank_<Cx>.pkl pre-built banks")
    ap.add_argument("--out-dir", default=None,
                    help="Fixed output dir for crash-safe resume. If unset, "
                         "auto-create timestamped dir (no resume across runs).")
    ap.add_argument("--online", action="store_true",
                    help="Within-episode memory mode: fresh memory unit per task, "
                         "write each step, retrieve each step. No bank needed.")
    ap.add_argument("--top-k", type=int, default=5,
                    help="top_k for retrieval in --online mode")
    ap.add_argument("--ref-max-chars", type=int, default=2000,
                    help="Per-entry truncation for retrieved refs. "
                         "-1 or 0 = unlimited (full entry).")
    args = ap.parse_args()

    if not args.alfworld_data:
        raise SystemExit("Set --alfworld-data or $ALFWORLD_DATA")
    os.environ["ALFWORLD_DATA"] = args.alfworld_data

    if args.out_dir:
        run_dir = args.out_dir
        print(f"Using fixed out-dir (resume-capable): {run_dir}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]
        run_dir = os.path.join(RESULTS_BASE, f"{ts}_alfworld_{model_short}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Model — shared across memory units
    model_cfg = ModelConfig(name=args.model, dtype="float16", temperature=0.0,
                            max_new_tokens=512)
    model = create_model(model_cfg)
    model.load()

    # Env — shared across memory units (cheap to reset per task)
    from alfworld.agents.environment import get_environment
    EnvClass = get_environment("AlfredTWEnv")
    env_config = _alfworld_config(args.alfworld_data)
    env_builder = EnvClass(env_config, train_eval=args.split)
    env = env_builder.init_env(batch_size=1)
    num_tasks = min(args.max_tasks, env_builder.num_games)
    print(f"Evaluating {num_tasks} tasks on {args.split}")

    for mid in args.memory:
        mem_dir = os.path.join(run_dir, mid)
        os.makedirs(mem_dir, exist_ok=True)

        # Load pre-built bank if present; if missing, fall back to an empty
        # fresh memory unit (useful for no-bank baselines like C10).
        state_dir = os.path.join(args.bank_dir, f"bank_{mid}")
        pickle_path = os.path.join(args.bank_dir, f"bank_{mid}.pkl")
        mem_cfg = MemoryConfig(memory_id=mid, extra={"encoder_model": args.encoder})
        if os.path.isdir(state_dir) and hasattr(create_memory_unit(mem_cfg, model=model), "load_state"):
            # Substrate-native save_state()/load_state() path
            print(f"[{mid}] loading bank from {state_dir}/ via load_state")
            tmp = create_memory_unit(mem_cfg, model=model)
            memory = type(tmp).load_state(state_dir, mem_cfg, model)
            print(f"[{mid}] bank size: {memory.size()}")
        elif os.path.exists(pickle_path):
            import pickle
            print(f"[{mid}] loading bank from {pickle_path}")
            with open(pickle_path, "rb") as f:
                memory = pickle.load(f)
            memory.model = model
            print(f"[{mid}] bank size: {memory.size()}")
        else:
            print(f"[{mid}] no bank at {state_dir}/ or {pickle_path} -> empty memory (no-bank baseline)")
            memory = create_memory_unit(mem_cfg, model=model)

        # Resume from existing per_task.json if present (crash-recovery)
        per_task_path = os.path.join(mem_dir, "per_task.json")
        per_task = []
        resume_from = 0
        if os.path.exists(per_task_path):
            try:
                with open(per_task_path) as f:
                    loaded = json.load(f)
                if isinstance(loaded, list) and loaded:
                    # Only resume if every entry has task_idx and they're contiguous from 0
                    max_idx = max(r.get("task_idx", -1) for r in loaded)
                    if max_idx >= 0:
                        per_task = loaded
                        resume_from = max_idx + 1
                        print(f"[{mid}] resuming from task_idx={resume_from} ({len(loaded)} prior results loaded)")
            except Exception as e:
                print(f"[{mid}] could not load prior per_task.json ({e}); starting fresh")

        # Skip env forward to match resume point (env state is sequential)
        for _ in range(resume_from):
            try:
                env.reset()
            except Exception as e:
                print(f"[{mid}] env skip-fwd error at task {_}: {e}")
                break

        n_succ = sum(int(r.get("success", False)) for r in per_task)
        t_start = time.time()
        # In online mode each task gets a fresh memory; we accumulate per-task
        # instrumentation here so final E-metrics reflect all 134 tasks.
        online_write_times: List[float] = []
        online_read_times: List[float] = []
        online_token_counts: Dict[str, int] = {}
        online_last_memory = None
        for ti in range(resume_from, num_tasks):
            try:
                if args.online:
                    # Fresh memory unit per task — online / within-episode mode
                    memory = create_memory_unit(mem_cfg, model=model)
                    result = run_task_online(env, memory, model,
                                             max_steps=args.max_steps,
                                             top_k=args.top_k,
                                             use_hints=not args.no_hints)
                    # Accumulate per-task instrumentation
                    if hasattr(memory, "_write_times"):
                        online_write_times.extend(memory._write_times)
                    if hasattr(memory, "_read_times"):
                        online_read_times.extend(memory._read_times)
                    if hasattr(memory, "_token_counts"):
                        for k, v in memory._token_counts.items():
                            online_token_counts[k] = online_token_counts.get(k, 0) + v
                    online_last_memory = memory
                else:
                    result = run_task(env, memory, model,
                                      max_steps=args.max_steps,
                                      k_success=args.k_success,
                                      k_failure=args.k_failure,
                                      use_hints=not args.no_hints,
                                      ref_max_chars=args.ref_max_chars)
            except Exception as e:
                # Record the failure but keep going — crash on one task should
                # not cost us the run
                print(f"  [{mid} task{ti}] EXCEPTION: {type(e).__name__}: {e}", flush=True)
                result = {"error": str(e), "success": False, "steps": 0,
                          "goal": "", "trajectory": []}
            result["task_idx"] = ti
            per_task.append(result)
            n_succ += int(result.get("success", False))
            if (ti + 1) % 10 == 0 or ti + 1 == num_tasks:
                done = ti + 1
                rate = n_succ / done
                print(f"  [{mid} {done}/{num_tasks}] running success={rate:.4f}  "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)
            # Incremental save every task (crash-safe)
            with open(per_task_path, "w") as f:
                json.dump(per_task, f)

        wall_sec = round(time.time() - t_start, 1)
        summary = {
            "memory_id": mid,
            "num_tasks": num_tasks,
            "num_success": n_succ,
            "success_rate": round(n_succ / num_tasks, 4),
            "avg_steps": round(sum(r["steps"] for r in per_task) / num_tasks, 2),
            "elapsed_sec": wall_sec,
        }

        # E-family efficiency metrics (matches BCB runner convention).
        # inference_times_ms = per-task total generation time (sum of gen_ms
        # across all steps in that task). Online mode also covers per-step
        # reads, which surface in E4 via memory._read_times.
        try:
            from SubMem.metrics.memory_efficiency import MemoryEfficiencyMetrics
            inference_times_ms = []
            for r in per_task:
                task_ms = sum(h.get("gen_ms", 0.0) for h in r.get("trajectory", []))
                if task_ms > 0:
                    inference_times_ms.append(task_ms)
            # In online mode, overwrite memory's per-instance stats with the
            # accumulated cross-task stats so E-metrics cover all 134 tasks.
            # E1 still reflects last instance's footprint (representative).
            if args.online and online_last_memory is not None:
                memory = online_last_memory
                memory._write_times = online_write_times
                memory._read_times = online_read_times
                if hasattr(memory, "_token_counts"):
                    memory._token_counts = dict(online_token_counts)
            eff = MemoryEfficiencyMetrics.compute(memory, inference_times_ms=inference_times_ms)
            eff["E14_total_time_sec"] = wall_sec
            eff["E15_avg_time_per_task_sec"] = round(wall_sec / num_tasks, 4) if num_tasks else None
            summary["efficiency"] = eff
            json.dump(eff, open(os.path.join(mem_dir, "efficiency.json"), "w"), indent=2)
        except Exception as e:
            print(f"[{mid}] efficiency compute failed: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
        # Per task_type (inferred from goal phrase)
        by_type = {}
        for r in per_task:
            g = r["goal"].lower()
            key = "other"
            for k in ("look", "two", "heat", "cool", "clean", "put"):
                if k in g:
                    key = k
                    break
            by_type.setdefault(key, []).append(int(r["success"]))
        summary["by_type"] = {
            k: {"n": len(v), "success_rate": round(sum(v) / len(v), 4)}
            for k, v in sorted(by_type.items())
        }
        json.dump(summary, open(os.path.join(mem_dir, "summary.json"), "w"), indent=2)
        print(f"[{mid}] success {n_succ}/{num_tasks} = {summary['success_rate']}")
        print(f"     by_type: {summary['by_type']}")


if __name__ == "__main__":
    main()
