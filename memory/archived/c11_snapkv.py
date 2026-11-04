"""C11 — SnapKV-utterance: adapted attention-guided selection baseline.

Inspired by: SnapKV (Li et al., NeurIPS 2024)
  "SnapKV: LLM Knows What You Are Looking for Before Generation"
  arXiv:2404.14469 — https://github.com/FasterDecoding/SnapKV

**This is NOT a paper-faithful reproduction of SnapKV.** The paper's core
contribution is *per-head KV-token eviction* (Algorithm 1): after prefill,
each (layer, head) independently keeps only its top-k most-important KV
entries, then decode reuses the compressed cache without re-prefilling.

SubMem's substrate instead selects whole utterances using the same
observation-window attention signal, then **re-prefills** the selected
utterances at read time. Concretely:
  1. Block-wise forward of buffer + query (observation window = last W
     tokens of each block).
  2. Attention from obs window → buffer tokens, pooled (kernel=5, avgpool).
  3. Per-utterance mean importance; pick top `budget_ratio` utterances.
  4. Re-prefill selected text + system prompt + query; decode.

Why not the paper version? Two independent blockers:
  (a) SnapKV's reference implementation ships monkey-patches for
      Llama/Mistral/Qwen2.5 attention; Qwen3 is not supported upstream
      and its hybrid (attention + DeltaNet) layers need separate handling.
  (b) The reference code depends on `flash_attn`, which requires a CUDA
      `nvcc` toolchain that isn't present on our compute nodes.

This adaptation is therefore closer in spirit to a *retrieval baseline*
than to KV-cache compression. E2 (inference latency) reflects re-prefill
cost, not SnapKV's advertised decode speedup. Results on LoCoMo should be
interpreted as "attention-guided utterance selection with SnapKV-style
scoring," not as a SnapKV reproduction.
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit
from ..models.hf import get_decoder_layers

# Defaults mirror FasterDecoding/SnapKV's SnapKVCluster:
#   window_size=64, max_capacity_prompt=256+64=320, kernel_size=5, pooling='avgpool'
_DEFAULT_BUDGET_RATIO = 0.2
_DEFAULT_WINDOW_SIZE = 64
_DEFAULT_POOL_KERNEL = 5          # SnapKV paper default (was 7 in our re-impl)
_DEFAULT_POOL_MODE = "avgpool"     # SnapKV paper default; our old code was max_pool1d
_DEFAULT_BLOCK_SIZE = 512


@register_memory_unit("C11")
class SnapKVMemory(BaseMemoryUnit):
    """SnapKV — per-query attention-guided text selection."""

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._budget_ratio = extra.get("budget_ratio", _DEFAULT_BUDGET_RATIO)
        self._window_size = extra.get("window_size", _DEFAULT_WINDOW_SIZE)
        self._pool_kernel = extra.get("pool_kernel", _DEFAULT_POOL_KERNEL)
        self._pool_mode = extra.get("pool_mode", _DEFAULT_POOL_MODE)
        self._block_size = extra.get("block_size", _DEFAULT_BLOCK_SIZE)

        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict] = []

        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._compress_times: List[float] = []
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }

    def save_state(self, dir_path: str) -> None:
        import os, json
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "buffer.json"), "w") as f:
            json.dump({"buffer": self._buffer, "buffer_metadata": self._buffer_metadata,
                       "token_counts": dict(self._token_counts),
                       "write_call_count": self._write_call_count}, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        import os, json
        unit = cls(config, model)
        with open(os.path.join(dir_path, "buffer.json")) as f:
            d = json.load(f)
        unit._buffer = d["buffer"]
        unit._buffer_metadata = d["buffer_metadata"]
        unit._token_counts = d["token_counts"]
        unit._write_call_count = d["write_call_count"]
        return unit

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._write_call_count += 1

    # ── Attention scoring infrastructure ──────────────────────────

    def _get_attn_layer_indices(self, hf_model):
        """Identify standard attention layers (Qwen3.5 hybrid: skip DeltaNet layers)."""
        std_attn_indices = []
        for i, layer in enumerate(get_decoder_layers(hf_model)):
            if hasattr(layer, 'self_attn'):
                std_attn_indices.append(i)
        if not std_attn_indices:
            std_attn_indices = list(range(len(get_decoder_layers(hf_model))))
        return std_attn_indices

    def _score_block(self, hf_model, tokenizer, block_text: str,
                     target_set: set, window: int) -> np.ndarray:
        """Score tokens in a block using observation window attention (SnapKV Algo 1).

        Returns per-token importance array for the prefix (all tokens before window).
        Handles GQA (Qwen3: 32 Q heads, 8 KV heads) by hooking into self_attn
        directly and averaging within KV head groups.
        """
        input_ids = tokenizer(block_text, return_tensors="pt").input_ids.to(hf_model.device)
        seq_len = input_ids.shape[1]
        w = min(window, seq_len - 1)
        past_len = seq_len - w

        if past_len <= 0:
            return np.zeros(seq_len, dtype=np.float32)

        importance_acc = np.zeros(past_len, dtype=np.float32)

        # Get GQA config for proper head grouping
        config = hf_model.config
        num_kv_heads = getattr(config, 'num_key_value_heads',
                               getattr(config, 'num_attention_heads', 32))

        def make_hook(layer_idx):
            def hook_fn(module, args, output):
                if not isinstance(output, tuple) or len(output) < 2:
                    return
                attn_weights = output[1]
                if attn_weights is None:
                    return
                if layer_idx in target_set:
                    num_q_heads = attn_weights.shape[1]
                    group_size = num_q_heads // max(num_kv_heads, 1)
                    if group_size > 1:
                        # GQA: average repeated heads within each KV group
                        attn_grouped = attn_weights.reshape(
                            attn_weights.shape[0], num_kv_heads, group_size,
                            attn_weights.shape[2], attn_weights.shape[3]
                        ).mean(dim=2)  # [batch, kv_heads, seq, seq]
                    else:
                        attn_grouped = attn_weights
                    obs_attn = attn_grouped[:, :, -w:, :past_len]
                    importance_acc[:] += obs_attn.sum(dim=(0, 1, 2)).cpu().numpy()
                modified = list(output)
                modified[1] = None
                return tuple(modified)
            return hook_fn

        # Hook into self_attn modules for reliable attention access
        hooks = []
        for i, layer in enumerate(get_decoder_layers(hf_model)):
            attn_module = getattr(layer, 'self_attn', None)
            if attn_module is not None:
                hooks.append(attn_module.register_forward_hook(make_hook(i)))
            else:
                hooks.append(layer.register_forward_hook(make_hook(i)))

        # SDPA doesn't support output_attentions — temporarily switch to eager
        orig_attn_impls = {}
        for i, layer in enumerate(get_decoder_layers(hf_model)):
            attn = getattr(layer, 'self_attn', None)
            if attn is not None and hasattr(attn, '_attn_implementation'):
                orig_attn_impls[i] = attn._attn_implementation
                attn._attn_implementation = "eager"

        with torch.no_grad():
            hf_model(input_ids, output_attentions=True)

        # Restore original attention implementation
        for i, impl in orig_attn_impls.items():
            get_decoder_layers(hf_model)[i].self_attn._attn_implementation = impl

        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        # 1D pooling (SnapKV Algorithm 1). Paper default: avg_pool, kernel=5.
        imp_tensor = torch.from_numpy(importance_acc).unsqueeze(0).unsqueeze(0)
        pool_fn = (
            torch.nn.functional.avg_pool1d
            if self._pool_mode == "avgpool"
            else torch.nn.functional.max_pool1d
        )
        pooled = pool_fn(
            imp_tensor, kernel_size=self._pool_kernel, stride=1,
            padding=self._pool_kernel // 2
        ).squeeze().numpy()

        return pooled

    # ── Main pipeline ─────────────────────────────────────────────

    def _snapkv_select_and_generate(self, query: str) -> str:
        """SnapKV pipeline: score utterances → select top budget → re-prefill → generate.

        Uses chunked scoring for long conversations to avoid OOM.
        Each chunk is scored independently with its own observation window.
        """
        hf_model = self.model._model
        tokenizer = self.model._tokenizer

        # SnapKV paper scores importance from *all* standard-attention
        # layers (Algorithm 1). Previously we only used the last 4 as a
        # memory-saving shortcut; that's no longer necessary and it was
        # distorting which tokens survived the kernel pooling.
        std_attn_indices = self._get_attn_layer_indices(hf_model)
        target_set = set(std_attn_indices)
        window = self._window_size

        t_compress = time.perf_counter()

        # Score utterances in blocks
        utterance_scores = np.zeros(len(self._buffer), dtype=np.float32)

        for block_start in range(0, len(self._buffer), self._block_size):
            block_end = min(block_start + self._block_size, len(self._buffer))
            block_utts = self._buffer[block_start:block_end]

            # Build block text with query appended (SnapKV: query is part of
            # the observation window so it guides which prefix tokens matter)
            block_text = "\n".join(block_utts) + f"\nQuestion: {query}\nAnswer:"

            # Score this block
            token_importance = self._score_block(
                hf_model, tokenizer, block_text, target_set, window
            )

            # Map token importance back to utterances in this block
            running_pos = 0
            for ui, utt in enumerate(block_utts):
                utt_tok_len = len(tokenizer.encode(utt + "\n", add_special_tokens=False))
                start = running_pos
                end = min(running_pos + utt_tok_len, len(token_importance))
                if start < len(token_importance) and end > start:
                    utterance_scores[block_start + ui] = token_importance[start:end].mean()
                running_pos += utt_tok_len

        # Select top budget% utterances globally
        budget = max(1, int(len(self._buffer) * self._budget_ratio))
        top_indices = np.argsort(utterance_scores)[-budget:]
        top_indices = sorted(top_indices)

        selected_text = "\n".join(self._buffer[i] for i in top_indices)

        # Track selected dia_ids for P4 Recall@k
        self._last_selected_dia_ids = []
        for i in top_indices:
            meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}
            did = meta.get("dia_id", "")
            if did:
                self._last_selected_dia_ids.append(did)

        self._compress_times.append(time.perf_counter() - t_compress)

        # Re-prefill with selected text + query → generate
        prompt = (
            "You are a helpful assistant with access to conversation memory.\n\n"
            f"Retrieved memory context:\n{selected_text}\n\n"
            f"Question: {query}\n\n"
            "Answer the question based on the memory context. "
            "If the information is not in the context, say 'I don't know'.\n"
            "Give a short, concise answer (a few words or a short phrase). "
            "Do not explain your reasoning.\n"
            "Answer:"
        )
        self._last_prompt_tokens = len(self.model.encode(prompt))
        answer = self.model.generate(prompt, max_new_tokens=64)
        return answer

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        if not self._buffer:
            return []

        t0 = time.perf_counter()
        answer = self._snapkv_select_and_generate(query)
        answer = self.model._strip_think_tags(answer)

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += getattr(self, "_last_prompt_tokens", 0)

        return [MemoryEntry(
            content=answer,
            metadata={"source": "snapkv", "budget_ratio": self._budget_ratio,
                       "dia_ids": getattr(self, '_last_selected_dia_ids', [])},
            score=1.0,
        )]

    # ── Lifecycle ─────────────────────────────────────────────────

    def reset(self):
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._write_times.clear()
        self._read_times.clear()
        self._compress_times.clear()
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }

    def size(self) -> int:
        return len(self._buffer)

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        compress_times_ms = [t * 1000 for t in self._compress_times]
        stats.update({
            "budget_ratio": self._budget_ratio,
            "window_size": self._window_size,
            "pool_kernel": self._pool_kernel,
            "block_size": self._block_size,
            "avg_compress_time_ms": float(np.mean(compress_times_ms)) if compress_times_ms else 0.0,
            "num_queries_processed": len(self._compress_times),
            "num_raw_utterances": len(self._buffer),
            "memory_size_bytes": self.get_external_storage_bytes(),
            "token_consumption": dict(self._token_counts),
            "write_call_count": self._write_call_count,
            "read_latency_ms": {
                "mean": float(np.mean(read_times_ms)) if read_times_ms else 0.0,
                "p50": float(np.median(read_times_ms)) if read_times_ms else 0.0,
                "p90": float(np.percentile(read_times_ms, 90)) if read_times_ms else 0.0,
            },
            "write_latency_ms": {"mean": 0.0, "p50": 0.0, "p90": 0.0},
        })
        return stats

    def get_external_storage_bytes(self) -> int:
        return sum(len(t.encode("utf-8")) for t in self._buffer)

    def get_all_entries(self) -> List[MemoryEntry]:
        return [MemoryEntry(content=t) for t in self._buffer]
