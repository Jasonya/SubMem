"""C12 — EpiCache-utterance: adapted episodic selection baseline.

Inspired by: EpiCache (Kim et al., 2025, Apple ML Research)
  "EpiCache: Episodic KV Cache Management for Long Conversational QA"
  arXiv:2509.17396 — https://github.com/apple/ml-epicache

**This is NOT a paper-faithful reproduction of EpiCache.** The paper's
core contribution is per-layer KV-token eviction with a power-based
budget allocation (§3.3): each layer's KV cache is shrunk independently
using `EvictCache`, keeping peak memory at O(M + M_block) without ever
materialising the full cache.

SubMem's substrate preserves EpiCache's *episodic clustering* (A1) and
*attention-guided scoring* (A2), plus the power-based layer weighting
we wired up in Phase A, but selects **whole utterances** to re-prefill
rather than evicting tokens from the KV cache directly.

Pipeline here:
  Offline:
    A1. Segment (w_embed=4 utterances) → encode (MiniLM) → K-means
        (n_episodes=4) → pick n_medoid=8 closest segments → concat as
        patched prompt for each episode.
    A2. Block-wise forward(buffer + patched_prompt); hooks on all
        std-attention layers record per-layer attention to the patched
        prompt. Per-layer scores are normalised, raised to α (power) and
        renormalised, then weighted-summed into a single per-token score.
    A3. Pick top `budget_ratio` utterances per episode; cache text +
        centroid embedding + dia_ids.
  Online:
    B1. Embed query → cosine-match to best centroid.
    B2. Re-prefill selected text + system prompt + query → decode.

Why not the paper version? Two independent blockers:
  (a) apple/ml-epicache ships a `LongConvQAModel` wrapper whose monkey-
      patch targets Llama/Qwen2.5 attention; Qwen3 (our base model) is
      not supported upstream and its hybrid architecture adds DeltaNet
      layers that don't have a KV cache to evict.
  (b) The reference code depends on `flash_attn`, which requires a CUDA
      `nvcc` toolchain that isn't present on our compute nodes.

Results on LoCoMo should therefore be interpreted as "EpiCache-style
episodic attention-guided utterance selection," not as an EpiCache
KV-compression reproduction. In particular, E1 (storage) and E2
(inference) reflect whole-utterance re-prefill, not the paper's
bounded-KV decode.
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit
from ..models.hf import get_decoder_layers

# Defaults mirror Apple's EpiCache scripts/run_epicache_eval_qwen.sh:
#   N_CLUSTER=4, N_MEDOID=8, CONV_WINDOW=4, POWER=1.3
_DEFAULT_N_EPISODES = 4     # N_CLUSTER
_DEFAULT_N_MEDOID = 8       # N_MEDOID — medoids per episode for patched prompt
_DEFAULT_W_EMBED = 4        # CONV_WINDOW — utterances per segment
_DEFAULT_BUDGET_RATIO = 0.25
_DEFAULT_POWER = 1.3        # layer-wise budget sharpness (α); 0 = uniform
_DEFAULT_BLOCK_SIZE = 2


@register_memory_unit("M11")
class EpiCacheMemory(BaseMemoryUnit):
    """EpiCache — Episodic block-wise KV cache compression."""

    def __init__(self, config, model):
        super().__init__(config, model)
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._n_episodes = extra.get("n_episodes", _DEFAULT_N_EPISODES)
        self._n_medoid = extra.get("n_medoid", _DEFAULT_N_MEDOID)
        self._w_embed = extra.get("w_embed", _DEFAULT_W_EMBED)
        self._budget_ratio = extra.get("budget_ratio", _DEFAULT_BUDGET_RATIO)
        self._power = extra.get("power", _DEFAULT_POWER)
        self._block_size = extra.get("block_size", _DEFAULT_BLOCK_SIZE)

        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict] = []
        self._episodes: Dict[int, Dict] = {}
        self._centroids: Optional[np.ndarray] = None
        self._indexed = False
        self._seg_encoder = None

        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }

    def _init_seg_encoder(self):
        if self._seg_encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._seg_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def _segment(self, utterances: List[str]) -> List[str]:
        """Divide utterances into segments of w_embed utterances each."""
        segments = []
        for i in range(0, len(utterances), self._w_embed):
            segments.append("\n".join(utterances[i:i + self._w_embed]))
        return segments

    def save_state(self, dir_path: str) -> None:
        # Cache only the raw buffer. _episodes is lazy-built at read time
        # using base model attention; we want it recomputed if base model
        # or budget_ratio changes.
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
        self._indexed = False
        self._write_call_count += 1

    # ── A1: Episodic Clustering ──────────────────────────────────

    def _cluster_episodes(self, segments, embeddings):
        """K-means clustering → episodes with medoid selection."""
        from sklearn.cluster import KMeans

        n_clusters = min(self._n_episodes, len(segments))
        if n_clusters < 2:
            n_clusters = 1
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        self._centroids = kmeans.cluster_centers_

        episodes_info = {}
        for e in range(n_clusters):
            mask = labels == e
            episode_embs = embeddings[mask]
            episode_seg_indices = [i for i in range(len(segments)) if labels[i] == e]

            # Medoids: top-n_medoid segments closest to centroid (EpiCache paper
            # default N_MEDOID=8). The original code took only 1 medoid which
            # makes the patched prompt too narrow — multiple medoids give a
            # richer anchor for attention-based scoring.
            centroid = self._centroids[e]
            dists = np.linalg.norm(episode_embs - centroid, axis=1)
            k_med = min(self._n_medoid, len(dists))
            top_local = np.argsort(dists)[:k_med]
            medoid_seg_indices = [episode_seg_indices[i] for i in top_local]
            medoid_texts = [segments[j] for j in medoid_seg_indices]

            episodes_info[e] = {
                "centroid": centroid,
                "segment_indices": episode_seg_indices,
                "medoid_seg_idx": medoid_seg_indices[0],  # back-compat
                "medoid_seg_indices": medoid_seg_indices,
                "medoid_text": "\n\n".join(medoid_texts),  # concatenate as patched prompt
            }
        return episodes_info, labels

    # ── A2: Block-wise Prefill with Patched Prompt ───────────────

    def _blockwise_score(self, hf_model, tokenizer, utterances, patched_prompt_text):
        """Block-wise attention scoring with patched prompt (EpiCache Algorithm 1 A3).

        Process utterances in blocks of block_size. After each block, append
        the patched prompt (medoid) and compute attention-based importance.
        This keeps peak memory bounded at O(block_tokens + patched_tokens).

        Layer aggregation follows the paper's power-based scheme (§3.3):
        each standard-attention layer's importance is normalised and raised
        to the α power (`self._power`), then renormalised to sum to 1; the
        final per-token importance is a power-weighted average across all
        std-attention layers. α=0 degenerates to uniform averaging; larger
        α concentrates weight on layers whose attention already focuses on
        the patched prompt.
        """
        # Identify standard attention layers (Qwen3.5 hybrid)
        std_attn_indices = []
        for i, layer in enumerate(get_decoder_layers(hf_model)):
            if hasattr(layer, 'self_attn'):
                std_attn_indices.append(i)
        if not std_attn_indices:
            std_attn_indices = list(range(len(get_decoder_layers(hf_model))))
        # Score every std-attention layer (not just last-4); the power
        # weighting naturally down-weights layers that don't attend strongly
        # to the patched prompt.
        target_set = set(std_attn_indices)

        utterance_scores = np.zeros(len(utterances), dtype=np.float32)

        # Optional cap on the scoring context (long benchmark backgrounds can
        # make the eager-attention matrix quadratically large). Truncating the
        # patched prompt keeps block bookkeeping consistent.
        import os as _os
        _cap = int(_os.environ.get("SUBMEM_M11_SCORE_MAX_TOKENS", "0") or 0)
        if _cap:
            _pp = tokenizer.encode(patched_prompt_text, add_special_tokens=False)
            if len(_pp) > _cap:
                patched_prompt_text = tokenizer.decode(_pp[:_cap])

        # Get GQA config for proper head grouping
        config = hf_model.config
        num_kv_heads = getattr(config, 'num_key_value_heads',
                               getattr(config, 'num_attention_heads', 32))

        # Process in blocks
        for block_start in range(0, len(utterances), self._block_size):
            block_end = min(block_start + self._block_size, len(utterances))
            block_utts = utterances[block_start:block_end]

            # Build block text + patched prompt (medoid)
            block_text = "\n".join(block_utts)
            input_text = block_text + "\n" + patched_prompt_text
            input_ids = tokenizer.encode(input_text, add_special_tokens=True,
                                         return_tensors="pt")
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids])
            input_ids = input_ids.to(hf_model.device)

            seq_len = input_ids.shape[1]
            patch_len = len(tokenizer.encode(patched_prompt_text, add_special_tokens=False))
            block_token_len = seq_len - patch_len

            if block_token_len <= 0:
                continue

            # Forward pass with attention hooks (GQA-aware). We now record
            # per-layer importance separately so we can apply power-based
            # layer-budget allocation after the forward pass instead of a
            # plain sum over a hand-picked layer subset.
            per_layer_scores: Dict[int, np.ndarray] = {}

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
                            attn_grouped = attn_weights.reshape(
                                attn_weights.shape[0], num_kv_heads, group_size,
                                attn_weights.shape[2], attn_weights.shape[3]
                            ).mean(dim=2)
                        else:
                            attn_grouped = attn_weights
                        patch_attn = attn_grouped[:, :, -patch_len:, :block_token_len]
                        per_layer_scores[layer_idx] = patch_attn.sum(
                            dim=(0, 1, 2)
                        ).cpu().numpy()
                    modified = list(output)
                    modified[1] = None
                    return tuple(modified)
                return hook_fn

            hooks = []
            for i, layer in enumerate(get_decoder_layers(hf_model)):
                attn_module = getattr(layer, 'self_attn', None)
                if attn_module is not None:
                    hooks.append(attn_module.register_forward_hook(make_hook(i)))
                else:
                    hooks.append(layer.register_forward_hook(make_hook(i)))

            # SDPA doesn't support output_attentions — temporarily switch to eager.
            # Some HF models (e.g. Gemma4) read attn_implementation from
            # MODEL config rather than per-layer; toggle both.
            orig_attn_impls = {}
            orig_model_attn = None
            try:
                if hasattr(hf_model.config, '_attn_implementation'):
                    orig_model_attn = hf_model.config._attn_implementation
                    hf_model.config._attn_implementation = "eager"
            except Exception:
                pass
            for i, layer in enumerate(get_decoder_layers(hf_model)):
                attn = getattr(layer, 'self_attn', None)
                if attn is not None and hasattr(attn, '_attn_implementation'):
                    orig_attn_impls[i] = attn._attn_implementation
                    attn._attn_implementation = "eager"
                # Also flip the per-layer config if present
                if attn is not None and hasattr(attn, 'config') and hasattr(attn.config, '_attn_implementation'):
                    try:
                        attn.config._attn_implementation = "eager"
                    except Exception:
                        pass

            with torch.no_grad():
                hf_model(input_ids, output_attentions=True)

            # Restore original attention implementation
            for i, impl in orig_attn_impls.items():
                get_decoder_layers(hf_model)[i].self_attn._attn_implementation = impl
            if orig_model_attn is not None:
                try:
                    hf_model.config._attn_implementation = orig_model_attn
                except Exception:
                    pass

            for h in hooks:
                h.remove()
            del input_ids
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Power-based layer-budget aggregation (EpiCache §3.3):
            # 1. Normalise each layer's per-token scores to sum=1.
            # 2. Use each layer's *peak* normalised attention to the patched
            #    prompt as its "score"; raise to the α power to get the
            #    layer-budget share.
            # 3. Weighted sum across layers using those shares.
            # With α=0 this degenerates to a uniform mean, reproducing the
            # old last-4-layers behaviour as a limit case.
            if per_layer_scores:
                layer_ids = list(per_layer_scores.keys())
                norm_scores = {}
                layer_peaks = []
                for lid in layer_ids:
                    s = per_layer_scores[lid].astype(np.float64)
                    tot = s.sum()
                    if tot > 0:
                        norm_scores[lid] = s / tot
                        layer_peaks.append(float(norm_scores[lid].max()))
                    else:
                        norm_scores[lid] = np.zeros_like(s)
                        layer_peaks.append(0.0)
                peaks = np.asarray(layer_peaks, dtype=np.float64)
                alpha = max(float(self._power), 0.0)
                weights = peaks ** alpha if alpha > 0 else np.ones_like(peaks)
                wsum = weights.sum()
                if wsum <= 0 or not np.isfinite(wsum):
                    weights = np.ones_like(weights)
                    wsum = float(len(weights))
                weights = weights / wsum
                importance_acc = np.zeros(block_token_len, dtype=np.float32)
                for w, lid in zip(weights, layer_ids):
                    importance_acc += (w * norm_scores[lid]).astype(np.float32)
            else:
                importance_acc = np.zeros(block_token_len, dtype=np.float32)

            # Map token importance back to utterances in this block
            running_pos = 0
            for ui, utt in enumerate(block_utts):
                utt_tok_len = len(tokenizer.encode(utt + "\n", add_special_tokens=False))
                start = running_pos
                end = min(running_pos + utt_tok_len, block_token_len)
                if start < block_token_len and end > start:
                    utterance_scores[block_start + ui] = importance_acc[start:end].mean()
                running_pos += utt_tok_len

        return utterance_scores

    # ── Build Episodes Pipeline ──────────────────────────────────

    def _build_episodes(self):
        if self._indexed:
            return
        if not self._buffer:
            self._indexed = True
            return

        t0 = time.perf_counter()
        self._init_seg_encoder()

        # A1: Segment → encode → cluster
        segments = self._segment(self._buffer)
        if not segments:
            self._indexed = True
            return

        embeddings = self._seg_encoder.encode(segments, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        episodes_info, labels = self._cluster_episodes(segments, embeddings)
        n_clusters = len(episodes_info)

        hf_model = self.model._model
        tokenizer = self.model._tokenizer

        budget_per_episode = max(1, int(len(self._buffer) * self._budget_ratio))

        print(f"  C12: building {n_clusters} episodes, block_size={self._block_size}, "
              f"budget={budget_per_episode} utts/episode...", flush=True)

        # A2+A3: For each episode, block-wise score with patched prompt → select
        for e in range(n_clusters):
            ep = episodes_info[e]
            patched_prompt = ep["medoid_text"]

            # Block-wise attention scoring with patched prompt
            utt_scores = self._blockwise_score(
                hf_model, tokenizer, self._buffer, patched_prompt
            )

            # Select top utterances by attention score
            top_indices = np.argsort(utt_scores)[-budget_per_episode:]
            top_indices = sorted(top_indices)
            selected_text = "\n".join(self._buffer[i] for i in top_indices)

            selected_dia_ids = []
            for i in top_indices:
                meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}
                did = meta.get("dia_id", "")
                if did:
                    selected_dia_ids.append(did)

            self._episodes[e] = {
                "centroid": ep["centroid"],
                "selected_text": selected_text,
                "selected_indices": list(top_indices),
                "dia_ids": selected_dia_ids,
                "medoid": patched_prompt[:100],
            }
            print(f"    episode {e}: {len(ep['segment_indices'])} segments, "
                  f"{len(top_indices)} utterances selected", flush=True)

        total_time = time.perf_counter() - t0
        n = len(self._episodes) or 1
        self._write_times.extend([total_time / n] * n)
        self._indexed = True

    # ── Online: Query → Match Episode → Generate ─────────────────

    def _icl_dense_fallback(self, query: str, top_k: int) -> List[MemoryEntry]:
        """ICL passthrough — see C9 for rationale."""
        if not self._buffer:
            return []
        if not hasattr(self, "_icl_index") or self._icl_index is None or self._icl_index.ntotal != len(self._buffer):
            from .embeddings import create_encoder
            import faiss
            if not hasattr(self, "_icl_encoder") or self._icl_encoder is None:
                self._icl_encoder = create_encoder(self.config)
            embs = self._icl_encoder.encode(list(self._buffer))
            self._icl_index = faiss.IndexFlatIP(self._icl_encoder.dim)
            self._icl_index.add(embs)
        q_emb = self._icl_encoder.encode([query])
        k = min(top_k, self._icl_index.ntotal)
        scores, indices = self._icl_index.search(q_emb, k)
        results = []
        t0 = time.perf_counter()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._buffer):
                continue
            results.append(MemoryEntry(
                content=self._buffer[idx],
                metadata={"source": "icl_dense_fallback"},
                score=float(score),
                timestamp=int(idx),
            ))
        self._read_times.append(time.perf_counter() - t0)
        return results

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        # ICL fallback: for test_time_learning benchmarks, EpiCache's
        # episode-clustered KV-cache compression discards the (input, label)
        # text the downstream ICL prompt needs. Dense-retrieve raw buffer
        # chunks instead. See c9_lora._icl_dense_fallback for the same idea.
        extra = getattr(self.config, "extra", {}) or {}
        if extra.get("benchmark_task_type") == "test_time_learning" and self._buffer:
            return self._icl_dense_fallback(query, top_k)

        self._build_episodes()
        if not self._episodes:
            return []

        t0 = time.perf_counter()
        self._init_seg_encoder()

        # B1: Match query to closest episode centroid
        q_emb = self._seg_encoder.encode([query], show_progress_bar=False)[0]
        similarities = []
        for e_id, ep in self._episodes.items():
            c = ep["centroid"]
            sim = float(np.dot(q_emb, c) / (np.linalg.norm(q_emb) * np.linalg.norm(c) + 1e-9))
            similarities.append((e_id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_id = similarities[0][0]

        # B2: Re-prefill matched episode's selected text → generate
        ep = self._episodes[best_id]
        prompt = (
            "You are a helpful assistant with access to conversation memory.\n\n"
            f"Retrieved memory context:\n{ep['selected_text']}\n\n"
            f"Question: {query}\n\n"
            "Answer the question based on the memory context. "
            "If the information is not in the context, say 'I don't know'.\n"
            "Give a short, concise answer (a few words or a short phrase). "
            "Do not explain your reasoning.\n"
            "Answer:"
        )
        prompt_tokens = len(self.model.encode(prompt))
        answer = self.model.generate(prompt, max_new_tokens=64)

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += prompt_tokens

        return [MemoryEntry(
            content=answer,
            metadata={
                "source": "epicache",
                "episode_id": best_id,
                "dia_ids": ep.get("dia_ids", []),
            },
            score=similarities[0][1],
        )]

    # ── Lifecycle ─────────────────────────────────────────────────

    def reset(self):
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._episodes.clear()
        self._centroids = None
        self._indexed = False
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0, "write_llm_output": 0, "write_llm_calls": 0,
            "read_context_tokens": 0, "read_calls": 0,
        }

    def size(self) -> int:
        return len(self._episodes)

    def get_stats(self) -> Dict[str, Any]:
        self._build_episodes()
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]
        stats.update({
            "n_episodes": len(self._episodes),
            "budget_ratio": self._budget_ratio,
            "block_size": self._block_size,
            "num_raw_utterances": len(self._buffer),
            "memory_size_bytes": self.get_external_storage_bytes(),
            "token_consumption": dict(self._token_counts),
            "write_call_count": self._write_call_count,
            "read_latency_ms": {
                "mean": float(np.mean(read_times_ms)) if read_times_ms else 0.0,
                "p50": float(np.median(read_times_ms)) if read_times_ms else 0.0,
                "p90": float(np.percentile(read_times_ms, 90)) if read_times_ms else 0.0,
            },
            "write_latency_ms": {
                "mean": float(np.mean(write_times_ms)) if write_times_ms else 0.0,
                "p50": float(np.median(write_times_ms)) if write_times_ms else 0.0,
                "p90": float(np.percentile(write_times_ms, 90)) if write_times_ms else 0.0,
            },
        })
        return stats

    def get_external_storage_bytes(self) -> int:
        return sum(len(ep["selected_text"].encode("utf-8")) for ep in self._episodes.values())

    def get_compressed_token_count(self) -> int:
        """Sum of selected-text token counts across all episodes.

        Previous implementation returned `n_episodes * block_size`, which is
        just an arbitrary product of two config knobs and has no relationship
        to what's actually stored. The correct quantity is the total token
        count of the kept utterances, which is what downstream cost models
        (E7 retrieved tokens) expect.
        """
        if not self._episodes:
            return 0
        encode = getattr(self.model, "encode", None)
        if encode is None:
            # Fallback: char/4 heuristic, same as other substrates use.
            return sum(len(ep["selected_text"]) // 4 for ep in self._episodes.values())
        total = 0
        for ep in self._episodes.values():
            text = ep.get("selected_text", "") or ""
            if not text:
                continue
            try:
                total += len(encode(text))
            except Exception:
                total += len(text) // 4
        return total

    def get_all_entries(self) -> List[MemoryEntry]:
        entries = []
        for e_id, ep in self._episodes.items():
            entries.append(MemoryEntry(
                content=ep["selected_text"],
                metadata={"episode_id": e_id, "dia_ids": ep.get("dia_ids", [])},
            ))
        return entries
