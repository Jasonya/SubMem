"""HotpotQA agentic rollout runner — ReAct on Wikipedia (Search/Lookup/Finish).

Protocol:
    - ReAct tool loop (Yao+ 2022): search[e], lookup[s], finish[a], plus think[t].
    - Per-question max_steps=7 (ReAct default).
    - Dev split subsample with seed=233 (matches ReAct's 500-question protocol).
    - Cross-episode memory: memory bank NOT reset between questions so substrates
      like C5/C14/C15 can accumulate experience. Each question ends with a
      write(trajectory, success) so failures also contribute (Reflexion-style).

Usage:
    python run_hotpotqa_rollout.py --memory C1 C5 C10 --max-questions 3 \\
        --model Qwen/Qwen3-8B
"""
import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # parent of the SubMem package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "third_party", "react"))

from SubMem.config import ModelConfig, MemoryConfig
from SubMem.models import create_model
from SubMem.memory import create_memory_unit

RESULTS_BASE = os.environ.get(
    "SUBMEM_RESULTS", os.path.join(os.path.dirname(__file__), "results")
)

# ── ReAct prompt (1-shot, trimmed from Yao+ 2022 Appendix) ──────
_REACT_EXAMPLE = """Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question asks who Matt Groening named the Milhouse character after. I need to search Milhouse and find who Matt Groening named Milhouse after.
Action 1: search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: finish[Richard Nixon]
"""


def build_prompt(question: str, history: List[Dict[str, str]],
                 memory_context: str, step: int) -> str:
    hist_lines = []
    for h in history:
        hist_lines.append(f"Thought {h['step']}: {h['thought']}")
        hist_lines.append(f"Action {h['step']}: {h['action']}")
        hist_lines.append(f"Observation {h['step']}: {h['obs']}")
    hist_str = "\n".join(hist_lines)

    mem_block = (
        f"\nRelevant prior experience (optional guidance):\n{memory_context}\n"
        if memory_context.strip() else ""
    )

    return (
        "Solve a question answering task with interleaving Thought, Action, "
        "Observation steps. Action can be one of: search[entity], "
        "lookup[string], finish[answer]. Action names are lowercase.\n\n"
        "Here is an example:\n"
        f"{_REACT_EXAMPLE}\n"
        f"{mem_block}"
        f"Question: {question}\n"
        f"{hist_str}\n"
        f"Thought {step}:"
    )


# ── parse ReAct output ───────────────────────────────────────────
_THOUGHT_RE = re.compile(r"^(.*?)(?:Action\s*\d*\s*:)", re.DOTALL)
_ACTION_RE = re.compile(r"Action\s*\d*\s*:\s*(\w+\[[^\]]*\])", re.IGNORECASE)


def parse_react(raw: str, step: int) -> Dict[str, str]:
    """Extract thought + action from model output. Tolerant of missing markers."""
    # Thought: everything up to "Action N:" or first newline
    thought = ""
    action = ""

    m = _ACTION_RE.search(raw)
    if m:
        action = m.group(1).strip()
        # Thought is everything before the match
        thought = raw[:m.start()].strip()
        # Strip leading "Thought N:" if the model echoed it
        thought = re.sub(r"^Thought\s*\d*\s*:\s*", "", thought).strip()
    else:
        # No explicit Action; take first non-empty line as thought, no action
        first_line = raw.strip().split("\n")[0].strip()
        thought = re.sub(r"^Thought\s*\d*\s*:\s*", "", first_line).strip()
        action = ""  # will be treated as invalid

    # Lowercase action verb to match wikienv (Yao's notebook does this too)
    if action and "[" in action:
        verb, rest = action.split("[", 1)
        action = verb.lower().strip() + "[" + rest

    return {"thought": thought, "action": action}


# ── rollout one question ─────────────────────────────────────────
def run_question(env, memory, model, idx: int, max_steps: int, top_k: int,
                 max_new_tokens: int = 200) -> Dict[str, Any]:
    obs = env.reset(idx=idx)
    question = env.data[idx][0]
    gold = env.data[idx][1]

    # Cross-episode retrieval at question start.
    # Always call read() — lazy-indexed substrates (C14/C15) need this to
    # trigger _ensure_indexed() which flushes _pending into the bank.
    t0 = time.time()
    retrieved = memory.read(question, top_k=top_k)
    read_ms = (time.time() - t0) * 1000
    mem_ctx = "\n".join(f"- {e.content[:200]}" for e in retrieved if e.content)

    history: List[Dict[str, str]] = []
    final_answer = None
    for step in range(1, max_steps + 1):
        prompt = build_prompt(question, history, mem_ctx, step)
        t0 = time.time()
        raw = model.generate(prompt, max_new_tokens=max_new_tokens)
        gen_ms = (time.time() - t0) * 1000
        parsed = parse_react(raw, step)
        action = parsed["action"]

        if not action:
            obs_new = "Invalid action: (empty)"
            done = False
        else:
            obs_new, _, done, info = env.step(action)

        history.append({
            "step": step,
            "thought": parsed["thought"][:500],
            "action": action,
            "obs": obs_new[:500] if isinstance(obs_new, str) else str(obs_new),
            "raw": raw[:800],
            "gen_ms": gen_ms,
        })

        if done or (action and action.startswith("finish[")):
            # Extract answer for reward computation even if env already set it
            info = env._get_info()
            final_answer = info.get("answer")
            break

    # Compute reward (EM) + F1
    info = env._get_info()
    reward = env.get_reward(info) if final_answer is not None else 0
    from wrappers import f1_score, normalize_answer
    f1 = 0.0
    if final_answer is not None:
        try:
            f1_tuple = f1_score(final_answer, gold)
            f1 = f1_tuple[0] if isinstance(f1_tuple, tuple) else f1_tuple
        except Exception:
            f1 = 0.0
    success = bool(reward)

    # Cross-episode write: persist trajectory summary + outcome
    key_steps = [f"Action{h['step']}: {h['action']} -> {h['obs'][:120]}" for h in history]
    traj_summary = (
        f"Q: {question}\nAnswer (pred/gold): {final_answer}/{gold}  success={success}\n"
        + "\n".join(key_steps)
    )
    md = {"question": question, "gold": gold, "pred": final_answer,
          "success": success, "f1": f1}
    # C14/C15 implement write_experience(query, trajectory, success, metadata);
    # other substrates use write(text, metadata).
    if hasattr(memory, "write_experience"):
        try:
            memory.write_experience(question, traj_summary, success=success, metadata=md)
        except NotImplementedError:
            memory.write(traj_summary, metadata=md)
    else:
        memory.write(traj_summary, metadata=md)

    return {
        "idx": idx,
        "question": question,
        "gold": gold,
        "pred": final_answer,
        "em": int(success),
        "f1": float(f1),
        "steps": len(history),
        "read_ms": read_ms,
        "trajectory": history,
    }


# ── main ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", nargs="+", default=["C1"],
                    help="Memory unit IDs, e.g. C1 C5 C10")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=7)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sample-seed", type=int, default=233,
                    help="ReAct convention: seed 233 shuffle of 7405 dev")
    ap.add_argument("--encoder", default="text-embedding-3-small")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    # Working dir — wrappers.py uses relative "data/" path
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "third_party", "react"))

    import wikienv, wrappers

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    run_dir = os.path.join(RESULTS_BASE, f"{ts}_hotpotqa_{model_short}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ReAct 500-question subsample
    idxs = list(range(7405))
    random.Random(args.sample_seed).shuffle(idxs)
    idxs = idxs[:args.max_questions]

    # Model
    model_cfg = ModelConfig(name=args.model, dtype="float16", temperature=0.0,
                            max_new_tokens=args.max_new_tokens)
    model = create_model(model_cfg)
    model.load()

    # Env — one WikiEnv shared across memory units (env is stateless between q's)
    base_env = wikienv.WikiEnv()
    env = wrappers.HotPotQAWrapper(base_env, split="dev")

    for mid in args.memory:
        mem_dir = os.path.join(run_dir, mid)
        os.makedirs(mem_dir, exist_ok=True)
        mem_cfg = MemoryConfig(memory_id=mid, extra={"encoder_model": args.encoder})
        memory = create_memory_unit(mem_cfg, model=model)
        # NOTE: no memory.reset() here — cross-episode accumulation

        per_q = []
        n_em = 0
        f1_total = 0.0
        t_start = time.time()
        for pos, qi in enumerate(idxs):
            try:
                result = run_question(env, memory, model, qi,
                                      args.max_steps, args.top_k,
                                      args.max_new_tokens)
            except Exception as e:
                print(f"  [{mid} q{qi}] EXCEPTION: {e}", flush=True)
                result = {"idx": qi, "error": str(e), "em": 0, "f1": 0.0}
            per_q.append(result)
            n_em += int(result.get("em", 0))
            f1_total += float(result.get("f1", 0.0))
            if (pos + 1) % 5 == 0 or pos + 1 == len(idxs):
                em_rate = n_em / (pos + 1)
                f1_rate = f1_total / (pos + 1)
                print(f"  [{mid} {pos+1}/{len(idxs)}] EM={em_rate:.3f} F1={f1_rate:.3f}  "
                      f"mem_size={memory.size()}  elapsed={time.time()-t_start:.0f}s",
                      flush=True)
            with open(os.path.join(mem_dir, "per_question.json"), "w") as f:
                json.dump(per_q, f)

        summary = {
            "memory_id": mid,
            "n_questions": len(idxs),
            "n_em": n_em,
            "em_rate": round(n_em / len(idxs), 4) if idxs else 0.0,
            "f1_mean": round(f1_total / len(idxs), 4) if idxs else 0.0,
            "final_mem_size": memory.size(),
            "elapsed_sec": round(time.time() - t_start, 1),
        }
        json.dump(summary, open(os.path.join(mem_dir, "summary.json"), "w"), indent=2)
        print(f"[{mid}] EM {n_em}/{len(idxs)} = {summary['em_rate']}  F1={summary['f1_mean']}")


if __name__ == "__main__":
    main()
