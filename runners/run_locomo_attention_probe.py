"""LoCoMo attention probe — mirror of run_alfworld_attention_probe.py.

Loads a pre-built LoCoMo memory bank, retrieves with substrate=Cx at
top_k ∈ {1, 2, 5, 10, 20}, runs HF Qwen3-8B with eager attention, and
dumps segment-level attention mass per (k, query) cell:

  segments = [system_intro, instructions, retrieved_block, question, answer_cue]

Output: results/<ts>_locomo_attn_probe/k<k>/<sub>/q<idx>.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, pickle
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # parent of the SubMem package

from SubMem.config import ModelConfig, MemoryConfig
from SubMem.models import create_model
from SubMem.memory import create_memory_unit
from SubMem.benchmarks.locomo import LoComoBenchmark


SEGMENT_MARKERS = [
    ("system_intro",    "You are an intelligent memory assistant"),
    ("instructions",    "# CONTEXT:"),
    ("retrieved_block", "Memories:"),
    ("question",        "Question:"),
    ("answer_cue",      "Answer:"),
]


def find_segment_char_spans(prompt: str) -> List[Tuple[str, int, int]]:
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


def attention_mass_per_segment(model, tokenizer, prompt: str,
                               char_spans: List[Tuple[str, int, int]]
                               ) -> Dict[str, Any]:
    # Apply chat template to match generate() exactly
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompt_offset = text_input.find(prompt)
        if prompt_offset < 0:
            text_input = prompt; prompt_offset = 0
    else:
        text_input = prompt; prompt_offset = 0

    enc = tokenizer(text_input, return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=False).to(model.device)
    offsets = enc["offset_mapping"][0].tolist()
    n_tok = len(offsets)

    # Map each token to a segment via char-offset containment
    seg_token_idxs: Dict[str, List[int]] = {n: [] for n, _, _ in char_spans}
    for tok_idx, (s, _e) in enumerate(offsets):
        for name, cs, ce in char_spans:
            if (cs + prompt_offset) <= s < (ce + prompt_offset):
                seg_token_idxs[name].append(tok_idx)
                break

    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask")

    captured: List[torch.Tensor] = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            attn_w = None
            if isinstance(output, tuple):
                for o in output:
                    if torch.is_tensor(o) and o.dim() == 4 and o.shape[-2] == o.shape[-1]:
                        attn_w = o; break
            if attn_w is None: return
            # last query row
            last_row = attn_w[0, :, -1, :].detach().to("cpu", torch.float32)
            captured.append(last_row)
        return hook

    # Resolve underlying nn.Module (SubMem wraps HF model in HuggingFaceModel)
    hf_model = model.model if hasattr(model, "model") else model

    layers = []
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        layers = hf_model.model.layers
    elif hasattr(hf_model, "layers"):
        layers = hf_model.layers

    handles = []
    for i, layer in enumerate(layers):
        h = layer.self_attn.register_forward_hook(make_hook(i),
                                                  with_kwargs=True)
        handles.append(h)

    try:
        with torch.no_grad():
            hf_model(input_ids=input_ids, attention_mask=attn_mask,
                     output_attentions=False, use_cache=False)
    finally:
        for h in handles: h.remove()

    if not captured:
        return {"_error": "no attention captured"}

    # Average over upper-half of layers, summed over heads, then renormalize
    all_attn = torch.stack(captured, dim=0)  # (n_layers, n_heads, k_len)
    n_layers = all_attn.shape[0]
    upper = all_attn[n_layers // 2 :].mean(dim=0).sum(dim=0)  # (k_len,)
    total = upper.sum().item()
    if total > 0:
        upper = upper / total

    out = {"_n_layers": n_layers,
           "_n_heads": int(all_attn.shape[1]),
           "_n_prompt_tokens": n_tok}
    for name, idxs in seg_token_idxs.items():
        if idxs:
            mass = float(upper[idxs].sum().item())
            out[name] = mass
            out[f"{name}__per_tok"] = mass / max(len(idxs), 1)
            out[f"{name}__n_tokens"] = len(idxs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--memory-id", default="C1")
    ap.add_argument("--n-queries", type=int, default=6)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(
            os.environ.get("SUBMEM_RESULTS",
                           os.path.join(os.path.dirname(__file__), "results")),
            f"{ts}_locomo_attn_probe",
        )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Out dir: {args.out_dir}", flush=True)

    # Load HF model with eager attention
    model_name = args.model.removeprefix("local/")
    model_cfg = ModelConfig(name=model_name, dtype="float16",
                            temperature=0.0, max_new_tokens=128)
    try: setattr(model_cfg, "attn_implementation", "eager")
    except Exception: pass
    model = create_model(model_cfg)
    model.load()
    try: model.model.config.attn_implementation = "eager"
    except Exception: pass

    # Load LoCoMo benchmark via SubMem's standard plumbing
    from SubMem.config import BenchmarkConfig
    bench = LoComoBenchmark(BenchmarkConfig(name="locomo"))
    samples = bench.load_data()
    # Use first conversation only (LoCoMo's `samples` = 10 conversations, each
    # contains many queries). Build memory inline from its turns.
    sample = samples[0]
    n_turns = sum(len(sess.get("turns", [])) for sess in sample.sessions)
    print(f"Sample 0 has {len(sample.queries)} queries, "
          f"{len(sample.sessions)} sessions, {n_turns} turns total",
          flush=True)

    mem_cfg = MemoryConfig(memory_id=args.memory_id.upper(),
                           extra={"encoder_model": "text-embedding-3-small"})
    mem = create_memory_unit(mem_cfg, model=model)
    print(f"Built fresh {args.memory_id} memory; writing turns...", flush=True)

    n_writes = 0
    for session in sample.sessions:
        for turn in session.get("turns", []):
            line = turn.get("content", "")
            if not line: continue
            meta = dict(turn.get("metadata", {}))
            meta["session_id"] = session.get("session_id", "")
            mem.write(line, metadata=meta)
            n_writes += 1
    print(f"  wrote {n_writes} turns", flush=True)
    # Force any lazy index to build
    _ = mem.read("dummy probe", top_k=1)

    queries = sample.queries[: args.n_queries]
    print(f"Probing {len(queries)} queries", flush=True)

    for k in args.ks:
        kdir = os.path.join(args.out_dir, f"k{k}", args.memory_id.lower())
        os.makedirs(kdir, exist_ok=True)
        print(f"\n=== k={k} ===", flush=True)
        for qi, q in enumerate(queries):
            t0 = time.time()
            retrieved = mem.read(q["question"], top_k=k)
            context = "\n".join([e.content for e in retrieved])
            prompt = bench.build_prompt(q, context)
            spans = find_segment_char_spans(prompt)
            attn = attention_mass_per_segment(model, model.tokenizer, prompt, spans)
            attn["_k"] = k; attn["_query_id"] = q.get("query_id", qi)
            attn["_question"] = q["question"][:200]
            with open(os.path.join(kdir, f"q{qi:02d}.json"), "w") as f:
                json.dump(attn, f, indent=2)
            elapsed = time.time() - t0
            print(f"  q{qi}: prompt_tok={attn.get('_n_prompt_tokens')} "
                  f"retr={attn.get('retrieved_block', 0):.3f} "
                  f"sys={attn.get('system_intro', 0):.3f} "
                  f"({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
