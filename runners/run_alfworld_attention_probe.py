"""Attention-probe rollout: run ALFWorld with k=1 (1 success + 1 failure) under
two conditions — (a) no memory and (b) C14 ReasoningBank — on a small set of
hand-picked tasks, and dump per-step segment-level attention mass.

Output (per task, per condition, per step):
  results/<ts>_attn_probe/<cond>/task<idx>_step<i>.json
    {
      "task_idx", "step", "goal", "action", "success_so_far",
      "segments": [{"name", "char_start", "char_end", "tok_start", "tok_end"}, ...],
      "segment_attention": {"<name>": <float, sum over heads, last layer>},
      "total_prompt_tokens": int,
      "n_layers", "n_heads"
    }

Conditions:
  - "nomem" : ref_block forced empty (no retrieval)
  - "c14"   : retrieve from pre-built C14 bank with k_success=1, k_failure=1

Run picks the first --n-tasks tasks from valid_unseen split (deterministic).

Usage:
  python run_alfworld_attention_probe.py \\
      --model Qwen/Qwen3-8B \\
      --bank-dir results/alfworld_banks_c14 \\
      --alfworld-data $ALFWORLD_DATA \\
      --n-tasks 4 --max-steps 20 --probe-steps 5
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # parent of the SubMem package

from SubMem.config import ModelConfig, MemoryConfig
from SubMem.models import create_model
from SubMem.memory import create_memory_unit
from SubMem.run_alfworld_rollout import (
    _alfworld_config,
    extract_goal,
    build_prompt,
    parse_action,
    retrieve_balanced,
    format_refs,
)


# Repo root: $SUBMEM_ROOT, else derived from this file's location (runners/..).
SUBMEM_ROOT = Path(os.environ.get("SUBMEM_ROOT", Path(__file__).resolve().parents[1]))
# Where pre-built memory banks live by default: $SUBMEM_RESULTS, else <repo>/results.
RESULTS_BASE = os.environ.get("SUBMEM_RESULTS", str(SUBMEM_ROOT / "results"))


# ── segment markers (must match build_prompt() output verbatim) ──
# Each marker's *first occurrence* in the rendered prompt defines a segment
# boundary. We compute char offsets, then map to token offsets via the
# tokenizer's offset_mapping.
SEGMENT_MARKERS = [
    ("system_intro", "You are an agent in a simulated household."),
    ("task",         "Task:"),
    ("ref_block",    "=== PAST EXPERIENCE"),
    ("history",      "Recent trajectory:"),
    ("current_obs",  "Current observation:"),
    ("admissible",   "Admissible actions"),
    ("cue",          "Output the chosen action"),
]


def find_segment_char_spans(prompt: str) -> List[Tuple[str, int, int]]:
    """Return [(name, char_start, char_end), ...] for each segment that exists
    in the prompt. Segments that aren't present (e.g. ref_block when no memory)
    are silently skipped."""
    hits = []
    for name, marker in SEGMENT_MARKERS:
        pos = prompt.find(marker)
        if pos >= 0:
            hits.append((name, pos))
    hits.sort(key=lambda x: x[1])
    spans = []
    for i, (name, start) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(prompt)
        spans.append((name, start, end))
    return spans


def char_spans_to_token_spans(tokenizer, prompt: str,
                              char_spans: List[Tuple[str, int, int]]
                              ) -> Tuple[List[Dict[str, Any]], int]:
    """Tokenize prompt with offset_mapping; assign each token to the segment
    whose char range contains its start offset. Returns segment list with
    token spans, plus total prompt token count."""
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()  # [(s, e), ...]
    n_tok = len(offsets)
    seg_of_token = [None] * n_tok
    for tok_idx, (s, _e) in enumerate(offsets):
        for seg_name, cs, ce in char_spans:
            if cs <= s < ce:
                seg_of_token[tok_idx] = seg_name
                break
    # Build segment summaries with first/last token positions
    segs = []
    for seg_name, cs, ce in char_spans:
        idxs = [i for i, n in enumerate(seg_of_token) if n == seg_name]
        if idxs:
            segs.append({
                "name": seg_name,
                "char_start": cs,
                "char_end": ce,
                "tok_start": idxs[0],
                "tok_end": idxs[-1] + 1,
                "n_tokens": len(idxs),
            })
    return segs, n_tok


def attention_mass_per_segment(model, tokenizer, prompt: str,
                               segments: List[Dict[str, Any]],
                               layer_range: str = "upper_half"
                               ) -> Dict[str, float]:
    """Run a single forward pass on prompt + chat template, capture attention
    from the *next-token query position* (i.e. the position that would generate
    the first new token), aggregate over heads & layers, then sum attention
    weight onto each segment's token range.

    Why aggregate over layers instead of just last-layer?
      Lower layers route most mass into chat-template special tokens
      (attention sinks), drowning out content-level signal. We average over
      the upper half of layers where task-semantic routing is concentrated.
      `layer_range` ∈ {"last", "upper_half", "all"}.

    Three views are reported per segment:
      - <name>            : raw attention mass (sums to ~1 with _other)
      - <name>__excl_sink : mass renormalised over the *non-template* tokens
                            only — strips the attention-sink confound.
      - <name>__per_tok   : raw mass / n_tokens in segment (density).

    Returns: {segment_name (+suffixes): float, ...} plus meta keys
    "_n_layers", "_n_heads", "_n_prompt_tokens", "_layer_range".
    """
    # Apply chat template the same way HuggingFaceModel.generate does, so the
    # token offsets we computed on the *raw prompt* are still valid as long as
    # the chat-template prefix doesn't appear inside the prompt itself. We
    # explicitly track the prefix length and shift token indices accordingly.
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        # Find where the raw prompt sits inside the templated string
        prompt_offset_chars = text_input.find(prompt)
        if prompt_offset_chars < 0:
            # Chat template munged the prompt (e.g. escaping); fall back to
            # tokenizing the templated string directly and re-locating segments.
            text_input = prompt
            prompt_offset_chars = 0
    else:
        text_input = prompt
        prompt_offset_chars = 0

    enc = tokenizer(text_input, return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=False).to(model.device)
    offsets = enc["offset_mapping"][0].tolist()
    n_tok = len(offsets)

    # Recompute segment token ranges in the templated tokenization
    char_to_seg = {}  # char_start in templated -> seg_name
    for seg in segments:
        cs = seg["char_start"] + prompt_offset_chars
        ce = seg["char_end"] + prompt_offset_chars
        char_to_seg[(cs, ce)] = seg["name"]

    seg_token_idxs: Dict[str, List[int]] = {seg["name"]: [] for seg in segments}
    for tok_idx, (s, _e) in enumerate(offsets):
        for (cs, ce), name in char_to_seg.items():
            if cs <= s < ce:
                seg_token_idxs[name].append(tok_idx)
                break

    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask")

    # Memory-efficient capture: instead of `output_attentions=True` (which
    # materialises the full (n_layers, n_heads, q_len, k_len) tensor and OOMs
    # on long retrieval prompts), hook each decoder layer's self-attention,
    # capture only the *last query row* of attention weights, immediately
    # detach to CPU. Each layer keeps only k_len floats × n_heads.
    captured: List[torch.Tensor] = []   # list of (n_heads, k_len) on CPU

    def make_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            # Qwen3 eager_attention_forward returns (attn_output, attn_weights)
            # — but the hook receives the layer module's output. Different HF
            # versions wrap differently. Try common shapes.
            attn_w = None
            if isinstance(output, tuple):
                for o in output:
                    if torch.is_tensor(o) and o.dim() == 4 and o.shape[-2] == o.shape[-1]:
                        attn_w = o
                        break
            if attn_w is None:
                # Fallback: attribute set by attention forward
                attn_w = getattr(module, "_last_attn_weights", None)
            if attn_w is not None:
                # (B, n_heads, q_len, k_len) → take last query row
                last_row = attn_w[0, :, -1, :].detach().to("cpu", dtype=torch.float32)
                captured.append((layer_idx, last_row))
        return hook

    # Register hooks on every decoder layer's self_attn
    handles = []
    decoder_layers = model.model.layers if hasattr(model, "model") else model.layers
    for li, dl in enumerate(decoder_layers):
        attn_mod = getattr(dl, "self_attn", None)
        if attn_mod is None:
            continue
        h = attn_mod.register_forward_hook(make_hook(li), with_kwargs=True)
        handles.append(h)

    try:
        with torch.no_grad():
            # NOTE: do NOT pass output_attentions=True here — it forces HF to
            # collect & stack all (n_layers, B, n_heads, q_len, k_len) tensors
            # for return, which is what OOMed us. The eager path still computes
            # attn_weights internally (visible to our hook) without keeping
            # them in the output struct.
            _ = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    if not captured:
        raise RuntimeError(
            "No attention weights captured by hooks. "
            "Check that attn_implementation='eager' is in effect."
        )

    # Sort by layer idx, build (n_layers, n_heads, k_len) on CPU
    captured.sort(key=lambda x: x[0])
    n_layers = len(captured)
    n_heads = captured[0][1].shape[0]
    layers_tensor = torch.stack([c[1] for c in captured], dim=0)  # (n_layers, n_heads, k_len)

    if layer_range == "last":
        layer_idxs = [n_layers - 1]
    elif layer_range == "all":
        layer_idxs = list(range(n_layers))
    else:
        layer_idxs = list(range(n_layers // 2, n_layers))

    sliced = layers_tensor[layer_idxs]                  # (sel_layers, n_heads, k_len)
    head_avg = sliced.mean(dim=1).mean(dim=0)           # (k_len,) — head- then layer-mean
    # Free GPU cache between probes — long retrievals fragment fast
    torch.cuda.empty_cache()

    seg_mass = {}
    for name, idxs in seg_token_idxs.items():
        if not idxs:
            seg_mass[name] = 0.0
        else:
            seg_mass[name] = float(head_avg[idxs].sum().item())

    # "_other" = mass on chat-template / special tokens not in any named segment
    covered = set()
    for idxs in seg_token_idxs.values():
        covered.update(idxs)
    uncovered = [i for i in range(n_tok) if i not in covered]
    other_mass = float(head_avg[uncovered].sum().item()) if uncovered else 0.0
    seg_mass["_other"] = other_mass

    # --- Sink-excluded renormalisation -----------------------------------
    # Treat the chat-template prefix/suffix tokens as attention sinks and
    # renormalise the remaining mass to sum to 1 over the named segments.
    in_seg_total = sum(v for k, v in seg_mass.items()
                       if not k.startswith("_"))
    if in_seg_total > 0:
        for seg in segments:
            name = seg["name"]
            seg_mass[f"{name}__excl_sink"] = seg_mass[name] / in_seg_total

    # --- Per-token density ------------------------------------------------
    # mass / n_tokens within segment — robust to segment-size differences.
    for seg in segments:
        name = seg["name"]
        n = seg["n_tokens"] or 1
        seg_mass[f"{name}__per_tok"] = seg_mass[name] / n

    seg_mass["_n_layers"] = n_layers
    seg_mass["_n_heads"] = n_heads
    seg_mass["_n_prompt_tokens"] = n_tok
    seg_mass["_layer_range"] = layer_range
    seg_mass["_n_selected_layers"] = len(layer_idxs)
    # Token-level distribution for per-token heatmap (panel c).
    seg_mass["_token_attention"] = head_avg.tolist()
    return seg_mass


def make_prompt_for_step(goal, current_obs, admissible, history,
                         memory, condition: str, k_success=1, k_failure=1,
                         use_hints=True, ref_max_chars=2000):
    if condition == "nomem":
        succ_refs = ""
        fail_refs = ""
    elif condition in ("c14", "c1"):
        succ_entries, fail_entries = retrieve_balanced(memory, goal,
                                                       k_success, k_failure)
        succ_refs = format_refs(succ_entries, label="success",
                                max_chars=ref_max_chars)
        fail_refs = format_refs(fail_entries, label="failure",
                                max_chars=ref_max_chars)
    else:
        raise ValueError(f"unknown condition {condition}")
    prompt = build_prompt(goal, current_obs, admissible,
                          succ_refs, fail_refs, history, use_hints=use_hints)
    return prompt


def run_one_task(env, model, hf_tokenizer, hf_model, memory, condition: str,
                 task_idx: int, max_steps: int, probe_steps: int,
                 out_dir: str, k_success: int, k_failure: int):
    """Run a single task; for each of the first `probe_steps` steps, dump
    segment-level attention. After probe_steps, continue rolling out (without
    further attention dumps) up to max_steps to learn final task success."""
    obs, info = env.reset()
    goal = extract_goal(obs[0])
    history: List[Dict[str, Any]] = []
    cond_dir = os.path.join(out_dir, condition)
    os.makedirs(cond_dir, exist_ok=True)

    success = False
    for step in range(max_steps):
        admissible = info["admissible_commands"][0]
        prompt = make_prompt_for_step(goal, obs[0], admissible, history,
                                      memory, condition,
                                      k_success=k_success,
                                      k_failure=k_failure)

        if step < probe_steps:
            char_spans = find_segment_char_spans(prompt)
            segs, n_tok = char_spans_to_token_spans(hf_tokenizer, prompt,
                                                    char_spans)
            seg_mass = attention_mass_per_segment(hf_model, hf_tokenizer,
                                                  prompt, segs)
            dump = {
                "task_idx": task_idx,
                "step": step,
                "condition": condition,
                "goal": goal,
                "n_prompt_tokens": n_tok,
                "segments": segs,
                "segment_attention": seg_mass,
            }
            out_path = os.path.join(cond_dir,
                                    f"task{task_idx:02d}_step{step:02d}.json")
            with open(out_path, "w") as f:
                json.dump(dump, f, indent=2)
            print(f"  [{condition} t{task_idx} s{step}] dumped attention "
                  f"({n_tok} tok, {len(segs)} segs)", flush=True)

        torch.cuda.empty_cache()
        t0 = time.time()
        raw = model.generate(prompt, max_new_tokens=128)
        gen_ms = (time.time() - t0) * 1000
        torch.cuda.empty_cache()
        action = parse_action(raw, admissible)
        obs_new, scores, dones, info = env.step([action])
        next_obs = obs_new[0]
        history.append({"action": action, "obs": next_obs,
                        "raw": raw[:500], "gen_ms": gen_ms})
        obs = obs_new
        if dones[0]:
            success = bool(scores[0])
            break

    summary = {
        "task_idx": task_idx,
        "condition": condition,
        "goal": goal,
        "success": success,
        "steps": len(history),
        "trajectory_actions": [h["action"] for h in history],
    }
    summary_path = os.path.join(cond_dir, f"task{task_idx:02d}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [{condition} t{task_idx}] DONE success={success} "
          f"steps={len(history)}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--alfworld-data",
                    default=os.environ.get("ALFWORLD_DATA", ""))
    ap.add_argument("--memory-id", default="C1",
                    help="Memory unit to probe against (C1, C14, ...). "
                         "The 'with-memory' condition is named after this.")
    ap.add_argument("--bank-path", default=None,
                    help="Full path to bank_<Cx>.pkl. If unset, derived from "
                         "default substrate-specific dirs.")
    ap.add_argument("--n-tasks", type=int, default=4,
                    help="# tasks to probe (taken from start of valid_unseen)")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--probe-steps", type=int, default=5,
                    help="dump attention only for the first N steps per task")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--k-success", type=int, default=1)
    ap.add_argument("--k-failure", type=int, default=1)
    ap.add_argument("--encoder", default="text-embedding-3-small")
    ap.add_argument("--split", default="eval_out_of_distribution")
    args = ap.parse_args()

    if not args.alfworld_data:
        raise SystemExit("Set --alfworld-data or $ALFWORLD_DATA")
    os.environ["ALFWORLD_DATA"] = args.alfworld_data

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(
            os.environ.get("SUBMEM_RESULTS",
                           os.path.join(os.path.dirname(__file__), "results")),
            f"{ts}_attn_probe",
        )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Out dir: {args.out_dir}")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Force HF backend (need attention weights; vLLM doesn't expose them).
    # Many of the run_*.sh scripts use prefix "local/" to mean "load via HF"
    # — strip it here.
    model_name = args.model
    if model_name.startswith("local/"):
        model_name = model_name[len("local/"):]
    model_cfg = ModelConfig(name=model_name, dtype="float16",
                            temperature=0.0, max_new_tokens=512)
    # Force eager attention so output_attentions=True returns real weights
    # (sdpa/flash silently return None).
    if hasattr(model_cfg, "attn_implementation"):
        model_cfg.attn_implementation = "eager"
    else:
        try:
            setattr(model_cfg, "attn_implementation", "eager")
        except Exception:
            pass
    model = create_model(model_cfg)
    model.load()
    hf_tokenizer = model.tokenizer
    hf_model = model.model
    # Belt-and-braces: re-set the underlying config too in case ModelConfig
    # didn't propagate it down.
    try:
        hf_model.config.attn_implementation = "eager"
    except Exception:
        pass

    # Env
    from alfworld.agents.environment import get_environment
    EnvClass = get_environment("AlfredTWEnv")
    env_config = _alfworld_config(args.alfworld_data)
    env_builder = EnvClass(env_config, train_eval=args.split)

    # Load memory bank for the with-memory condition
    mem_id = args.memory_id.upper()
    mem_cond = mem_id.lower()
    mem_cfg = MemoryConfig(memory_id=mem_id,
                           extra={"encoder_model": args.encoder})
    if args.bank_path:
        pickle_path = args.bank_path
    else:
        # Default substrate-specific bank locations under $SUBMEM_RESULTS
        default_dirs = {
            "C1": os.path.join(RESULTS_BASE, "alfworld_banks_v2"),
            "C14": os.path.join(RESULTS_BASE, "alfworld_banks_c14"),
        }
        bank_dir = default_dirs.get(mem_id,
                                    os.path.join(RESULTS_BASE, "alfworld_banks_v2"))
        pickle_path = os.path.join(bank_dir, f"bank_{mem_id}.pkl")
    print(f"Loading {mem_id} bank: {pickle_path}")
    import pickle
    with open(pickle_path, "rb") as f:
        mem_memory = pickle.load(f)
    mem_memory.model = model
    print(f"  bank size: {mem_memory.size()}")

    all_summaries = []
    for cond in ("nomem", mem_cond):
        print(f"\n====== condition: {cond} ======")
        env = env_builder.init_env(batch_size=1)
        memory = mem_memory if cond == mem_cond else None
        for ti in range(args.n_tasks):
            s = run_one_task(env, model, hf_tokenizer, hf_model, memory, cond,
                             task_idx=ti, max_steps=args.max_steps,
                             probe_steps=args.probe_steps,
                             out_dir=args.out_dir,
                             k_success=args.k_success,
                             k_failure=args.k_failure)
            all_summaries.append(s)

    with open(os.path.join(args.out_dir, "all_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nDone. {len(all_summaries)} task-condition runs in {args.out_dir}")


if __name__ == "__main__":
    main()
