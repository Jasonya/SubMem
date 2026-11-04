"""M6 — Hierarchical Memory with RAPTOR-style Recursive Abstractive Tree.

Reference: RAPTOR (Sarthi et al., ICLR 2024)
- "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
- https://arxiv.org/abs/2401.18059

Core algorithm adapted from: https://github.com/parthsarthi03/raptor

Build (bottom-up):
  1. Chunk conversation utterances → leaf nodes with embeddings
  2. UMAP dimensionality reduction → GMM soft clustering (global + local)
  3. LLM summarises each cluster → new parent node
  4. Recurse on parent nodes until convergence (≤ reduction_dim+1 nodes)

Retrieval (collapsed tree):
  All nodes at ALL levels are placed in a single flat index.
  Query embedding is compared against every node → top-k by cosine.
  This lets the query match high-level summaries OR fine-grained leaf chunks.
"""
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# ── Defaults ──────────────────────────────────────────────────
# Defaults match parthsarthi03/raptor's RetrievalAugmentation:
#   tb_max_tokens=100, tb_summarization_length=100, max_context_tokens=3500
_DEFAULT_CHUNK_SIZE = 100       # max tokens per leaf chunk (RAPTOR default)
_DEFAULT_NUM_LAYERS = 5         # max tree depth
_DEFAULT_REDUCTION_DIM = 10     # UMAP target dimension
_DEFAULT_GMM_THRESHOLD = 0.1    # GMM soft-assignment threshold
_DEFAULT_MAX_CLUSTER_TOKENS = 3500  # max ctx tokens (RAPTOR default)
_DEFAULT_SUMMARY_TOKENS = 100   # summarization_length (RAPTOR default)

# ── RAPTOR clustering (from raptor/cluster_utils.py) ──────────

def _global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    import umap
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def _local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    import umap
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def _get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 224
) -> int:
    from sklearn.mixture import GaussianMixture
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def _gmm_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    from sklearn.mixture import GaussianMixture
    n_clusters = _get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def _perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float
) -> List[np.ndarray]:
    """Global UMAP → GMM, then local UMAP → GMM within each global cluster."""
    reduced_global = _global_cluster_embeddings(
        embeddings, min(dim, len(embeddings) - 2)
    )
    global_clusters, n_global = _gmm_cluster(reduced_global, threshold)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global):
        mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embs = embeddings[mask]

        if len(global_cluster_embs) == 0:
            continue
        if len(global_cluster_embs) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embs]
            n_local = 1
        else:
            reduced_local = _local_cluster_embeddings(global_cluster_embs, dim)
            local_clusters, n_local = _gmm_cluster(reduced_local, threshold)

        # Map each local-cluster embedding back to its original row in
        # `embeddings`. The upstream raptor reference uses float equality
        # (embeddings == local_embs[:, None]).all(-1); that is fragile
        # when UMAP produces near-duplicate rows and also O(N*M*d). Replace
        # with a nearest-neighbour lookup via squared-distance argmin.
        global_indices_in_full = np.where(mask)[0]
        for j in range(n_local):
            local_mask = np.array([j in lc for lc in local_clusters])
            # Indices within global_cluster_embs that belong to local cluster j
            local_indices_in_global = np.where(local_mask)[0]
            for li in local_indices_in_global:
                idx = int(global_indices_in_full[li])
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local

    return all_local_clusters


def _raptor_cluster_nodes(
    nodes: list,
    embeddings: np.ndarray,
    max_length_in_cluster: int = 3500,
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    tokenize_fn=None,
) -> List[List[int]]:
    """RAPTOR clustering: returns list of clusters, each cluster is list of node indices.

    Adapted from raptor/cluster_utils.py RAPTOR_Clustering.perform_clustering.
    """
    clusters = _perform_clustering(
        embeddings, dim=reduction_dimension, threshold=threshold
    )

    node_clusters = []
    for label in np.unique(np.concatenate(clusters)):
        indices = [i for i, cluster in enumerate(clusters) if label in cluster]

        if len(indices) == 1:
            node_clusters.append(indices)
            continue

        # Check total token length
        if tokenize_fn is not None:
            total_length = sum(tokenize_fn(nodes[i]) for i in indices)
        else:
            total_length = sum(len(nodes[i].split()) for i in indices)

        if total_length > max_length_in_cluster:
            # Recluster this oversized cluster
            sub_embeddings = embeddings[indices]
            if len(sub_embeddings) > reduction_dimension + 2:
                sub_clusters = _raptor_cluster_nodes(
                    [nodes[i] for i in indices],
                    sub_embeddings,
                    max_length_in_cluster,
                    reduction_dimension,
                    threshold,
                    tokenize_fn,
                )
                # Map sub-indices back to original indices
                for sc in sub_clusters:
                    node_clusters.append([indices[si] for si in sc])
            else:
                node_clusters.append(indices)
        else:
            node_clusters.append(indices)

    return node_clusters


# ── Text splitting (from raptor/utils.py) ─────────────────────

def _split_text(text: str, max_tokens: int, tokenize_fn) -> List[str]:
    """Split text into chunks of roughly max_tokens size."""
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    n_tokens = [tokenize_fn(s) for s in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        if not sentence.strip():
            continue
        if token_count > max_tokens:
            # Split long sentences on commas/semicolons
            sub_sentences = [s.strip() for s in re.split(r"[,;:]", sentence) if s.strip()]
            sub_chunk = []
            sub_length = 0
            for sub_s in sub_sentences:
                sub_tc = tokenize_fn(sub_s)
                if sub_length + sub_tc > max_tokens and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = []
                    sub_length = 0
                sub_chunk.append(sub_s)
                sub_length += sub_tc
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = token_count
        else:
            current_chunk.append(sentence)
            current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ── Summarisation prompt ──────────────────────────────────────
# Paper-faithful minimal prompt (parthsarthi03/raptor SummarizationModels.py)
_SUMMARIZE_PROMPT = """Write a summary of the following, including as many key details as possible: {context}:"""


# ── Tree node ─────────────────────────────────────────────────

class _RaptorNode:
    """A node in the RAPTOR tree (leaf or summary)."""
    __slots__ = ("index", "text", "embedding", "children", "layer", "metadata")

    def __init__(self, index: int, text: str, embedding: np.ndarray,
                 children: Optional[Set[int]] = None, layer: int = 0,
                 metadata: Optional[Dict] = None):
        self.index = index
        self.text = text
        self.embedding = embedding
        self.children = children or set()
        self.layer = layer
        self.metadata = metadata or {}


# ── M6 Memory Unit ────────────────────────────────────────────

@register_memory_unit("M6")
class HierarchicalMemory(BaseMemoryUnit):
    """RAPTOR-style hierarchical memory with collapsed-tree retrieval.

    Build: bottom-up clustering + recursive summarisation.
    Read: flat cosine search across ALL tree nodes (all layers).
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict] = []
        self._indexed = False

        # Tree storage
        self._nodes: Dict[int, _RaptorNode] = {}  # index → node
        self._layer_to_nodes: Dict[int, List[int]] = {}  # layer → [node indices]
        self._num_layers = 0
        self._next_index = 0

        # Config
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        self._chunk_size = extra.get("chunk_size", _DEFAULT_CHUNK_SIZE)
        self._num_layers_max = extra.get("num_layers", _DEFAULT_NUM_LAYERS)
        self._reduction_dim = extra.get("reduction_dimension", _DEFAULT_REDUCTION_DIM)
        self._gmm_threshold = extra.get("gmm_threshold", _DEFAULT_GMM_THRESHOLD)
        self._max_cluster_tokens = extra.get("max_cluster_tokens", _DEFAULT_MAX_CLUSTER_TOKENS)
        self._summary_tokens = extra.get("summary_tokens", _DEFAULT_SUMMARY_TOKENS)

        # Stats
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._llm_call_times: List[float] = []
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._index_build_time = 0.0  # track tree build time separately
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }

        # Encoder
        self._enc = None
        self._init_encoder()

    def _init_encoder(self):
        from .embeddings import create_encoder
        self._enc = create_encoder(self.config)

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self._enc.encode(texts)

    def _tokenize_len(self, text: str) -> int:
        # RAPTOR's reference implementation (parthsarthi03/raptor) splits
        # text with tiktoken cl100k_base (gpt-3.5/4 tokenizer); using our
        # Qwen tokenizer here shifts every chunk boundary and skews the
        # cluster sizes at layer 0. We prefer tiktoken for parity with
        # the paper; fall back to the Qwen tokenizer only if tiktoken
        # isn't installed.
        enc = getattr(self, "_tiktoken_enc", None)
        if enc is None:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = False  # sentinel: tiktoken unavailable
            self._tiktoken_enc = enc
        if enc:
            return len(enc.encode(text))
        return len(self.model.encode(text))

    def _alloc_index(self) -> int:
        idx = self._next_index
        self._next_index += 1
        return idx

    # ── Write ──────────────────────────────────────────────────

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._indexed = False
        self._write_call_count += 1

    # ── Tree construction (RAPTOR algorithm) ───────────────────

    def _create_node(self, text: str, children: Optional[Set[int]] = None,
                     layer: int = 0, metadata: Optional[Dict] = None) -> _RaptorNode:
        """Create a node, embed it, and register it."""
        idx = self._alloc_index()
        emb = self._encode([text])[0]
        node = _RaptorNode(idx, text, emb, children or set(), layer, metadata or {})
        self._nodes[idx] = node
        self._layer_to_nodes.setdefault(layer, []).append(idx)
        self._stored_token_count += self._tokenize_len(text)
        return node

    def _summarize_cluster(self, texts: List[str], parent_layer: int = 1) -> str:
        """LLM-summarise a cluster of texts.

        parent_layer==1 → first rollup from raw leaves (write/construct).
        parent_layer>=2 → summary-of-summaries (mgmt — compressing existing memory).
        """
        context = "\n\n".join(texts)
        prompt = _SUMMARIZE_PROMPT.format(context=context)
        t0 = time.perf_counter()
        response = self.model.generate(prompt, max_new_tokens=self._summary_tokens)
        self._llm_call_times.append(time.perf_counter() - t0)
        inp = self._tokenize_len(prompt)
        out = self._tokenize_len(response)
        if parent_layer <= 1:
            self._token_counts["write_llm_input"] += inp
            self._token_counts["write_llm_output"] += out
            self._token_counts["write_llm_calls"] += 1
        else:
            self._token_counts.setdefault("mgmt_llm_input", 0)
            self._token_counts.setdefault("mgmt_llm_output", 0)
            self._token_counts.setdefault("mgmt_llm_calls", 0)
            self._token_counts["mgmt_llm_input"] += inp
            self._token_counts["mgmt_llm_output"] += out
            self._token_counts["mgmt_llm_calls"] += 1
        return response

    def _build_tree(self, leaf_nodes: List[_RaptorNode]):
        """RAPTOR: recursively cluster → summarise → repeat.

        Adapted from raptor/cluster_tree_builder.py ClusterTreeBuilder.construct_tree.
        """
        current_level_indices = [n.index for n in leaf_nodes]

        for layer_idx in range(self._num_layers_max):
            current_nodes = [self._nodes[i] for i in current_level_indices]

            if len(current_nodes) <= self._reduction_dim + 1:
                self._num_layers = layer_idx
                logging.info(
                    f"  M6-RAPTOR: stopping at layer {layer_idx} "
                    f"({len(current_nodes)} nodes ≤ {self._reduction_dim + 1})"
                )
                break

            # Cluster
            embeddings = np.array([n.embedding for n in current_nodes])
            texts = [n.text for n in current_nodes]

            try:
                clusters = _raptor_cluster_nodes(
                    texts,
                    embeddings,
                    max_length_in_cluster=self._max_cluster_tokens,
                    reduction_dimension=self._reduction_dim,
                    threshold=self._gmm_threshold,
                    tokenize_fn=self._tokenize_len,
                )
            except Exception as e:
                logging.warning(f"  M6-RAPTOR: clustering failed at layer {layer_idx}: {e}")
                self._num_layers = layer_idx
                break

            logging.info(
                f"  M6-RAPTOR: layer {layer_idx + 1} — "
                f"{len(current_nodes)} nodes → {len(clusters)} clusters"
            )

            # Summarise each cluster → new parent node
            new_level_indices = []
            for cluster_indices in clusters:
                cluster_texts = [texts[i] for i in cluster_indices]
                child_ids = {current_nodes[i].index for i in cluster_indices}

                # Inherit dia_ids from all leaf descendants so retrieved
                # summary nodes can still be scored against gold evidence
                # (P6/P7). Walk transitively to collect leaves.
                leaf_dias: List[str] = []
                seen: Set[int] = set()
                stack = list(child_ids)
                while stack:
                    nid = stack.pop()
                    if nid in seen: continue
                    seen.add(nid)
                    n = self._nodes.get(nid)
                    if n is None: continue
                    if not n.children:
                        for d in (n.metadata.get("dia_ids") or []):
                            if d and d not in leaf_dias:
                                leaf_dias.append(d)
                    else:
                        stack.extend(n.children)

                summary = self._summarize_cluster(cluster_texts, parent_layer=layer_idx + 1)
                parent_meta = {"dia_ids": leaf_dias} if leaf_dias else {}
                parent = self._create_node(
                    summary, children=child_ids, layer=layer_idx + 1,
                    metadata=parent_meta,
                )
                new_level_indices.append(parent.index)

            current_level_indices = new_level_indices
            self._num_layers = layer_idx + 1

    def _ensure_indexed(self):
        if self._indexed:
            return
        if not self._buffer:
            self._indexed = True
            return

        t0 = time.perf_counter()
        self._raw_token_count = sum(self._tokenize_len(t) for t in self._buffer)

        # Step 1: Create leaf nodes from buffer
        n_msgs = len(self._buffer)
        logging.info(f"  M6-RAPTOR: building tree from {n_msgs} messages...")

        leaf_nodes = []
        # Leaf granularity is task-adaptive. The default (RAPTOR-paper)
        # splits each buffered text into ~chunk_size token leaves. For ICL
        # classification (test_time_learning), each buffered turn is a list
        # of atomic (input, label) pairs; sub-sentence splitting separates
        # `input:` from its `label:` into different leaves and breaks the
        # pair the model needs at read time. In ICL mode, treat each turn
        # as one leaf so the (input, label) text stays together.
        _icl_mode = (getattr(self.config, "extra", {}) or {}).get(
            "benchmark_task_type") == "test_time_learning"

        for i, text in enumerate(self._buffer):
            meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}
            dia_id = meta.get("dia_id", "")

            # Split long utterances into chunks (default), or keep the whole
            # turn as one leaf (ICL mode — preserves input/label pairing).
            if _icl_mode:
                chunks = [text]
            else:
                chunks = _split_text(text, self._chunk_size, self._tokenize_len)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                # Propagate the original turn-level metadata (has_answer,
                # session_id, role, dia_id) so retrieval-quality metrics
                # (P5/P6/P7) can downstream-match against gold-evidence
                # flags. Without this the leaf node has only the chunk text.
                node_meta = dict(meta or {})
                if dia_id:
                    node_meta["dia_id"] = dia_id
                    node_meta["dia_ids"] = [dia_id]
                node = self._create_node(chunk, layer=0, metadata=node_meta)
                leaf_nodes.append(node)

        logging.info(f"  M6-RAPTOR: {len(leaf_nodes)} leaf nodes created")

        # Step 2: Build tree (recursive clustering + summarisation)
        if len(leaf_nodes) > self._reduction_dim + 1:
            self._build_tree(leaf_nodes)
        else:
            logging.info("  M6-RAPTOR: too few leaves to build tree, using flat index")

        total_time = time.perf_counter() - t0
        self._index_build_time = total_time
        # Amortize total build time across each write() call: user-facing
        # write cost is build_time / n_writes, not per-summary LLM time
        # (summaries are ~85 slow ops but user wrote ~419 times).
        n_writes = self._write_call_count or len(leaf_nodes) or 1
        self._write_times.clear()
        self._write_times.extend([total_time / n_writes] * n_writes)

        logging.info(
            f"  M6-RAPTOR: tree built — {len(self._nodes)} total nodes, "
            f"{self._num_layers + 1} layers, {total_time:.1f}s"
        )
        self._indexed = True

    # ── Read (collapsed tree retrieval) ────────────────────────

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Collapsed-tree retrieval: cosine search across nodes.

        Default policy (collapsed): consider nodes at ALL layers so the
        query can match a high-level summary or a fine-grained leaf
        (adapted from raptor/tree_retriever.py
        retrieve_information_collapse_tree).

        Leaf-only policy: for benchmarks where the per-chunk text *is*
        the signal the model needs (ICL classification: banking77 /
        clinic / TREC, where each chunk is an `input: ... label: <int>`
        pair), RAPTOR's LLM-summarised parent nodes paraphrase the
        numeric labels away. We still build the full tree but restrict
        retrieval to layer-0 leaves, so the original (input, label)
        text reaches the model.

        Selected by extra["benchmark_task_type"] == "test_time_learning".
        """
        self._ensure_indexed()
        if not self._nodes:
            return []

        t0 = time.perf_counter()
        q_emb = self._encode([query])[0]

        # Layer-aware candidate pool: collapsed (default) or leaf-only (ICL).
        extra = getattr(self.config, "extra", {}) or {}
        if extra.get("benchmark_task_type") == "test_time_learning":
            all_nodes = [n for n in self._nodes.values() if n.layer == 0]
        else:
            all_nodes = list(self._nodes.values())
        if not all_nodes:
            self._read_times.append(time.perf_counter() - t0)
            return []
        embeddings = np.array([n.embedding for n in all_nodes])
        # cosine similarity (embeddings are already L2-normalised)
        similarities = embeddings @ q_emb

        # Top-k by similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            node = all_nodes[idx]
            layer_label = f"L{node.layer}"
            content = f"[{layer_label}] {node.text}"

            results.append(MemoryEntry(
                content=content,
                metadata=node.metadata,
                score=float(similarities[idx]),
            ))

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += sum(
            self._tokenize_len(r.content) for r in results
        )
        return results

    # ── Lifecycle ──────────────────────────────────────────────

    def reset(self):
        self._nodes.clear()
        self._layer_to_nodes.clear()
        self._num_layers = 0
        self._next_index = 0
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._indexed = False
        self._raw_token_count = 0
        self._stored_token_count = 0
        self._llm_call_times.clear()
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._index_build_time = 0.0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }

    def size(self) -> int:
        self._ensure_indexed()
        return len([n for n in self._nodes.values() if n.layer == 0])

    # ── cache support ─────────────────────────────────────────
    CACHE_KEY_EXTRA = ["chunk_size", "num_layers", "reduction_dimension",
                       "gmm_threshold", "max_cluster_tokens", "summary_tokens"]

    def save_state(self, dir_path: str) -> None:
        # Force tree build if not yet done
        self._ensure_indexed()
        import os, pickle
        os.makedirs(dir_path, exist_ok=True)
        # Serialize nodes: convert _RaptorNode → dict
        nodes_data = {idx: {
            "index": n.index, "text": n.text,
            "embedding": n.embedding,
            "children": list(n.children),
            "layer": n.layer, "metadata": n.metadata,
        } for idx, n in self._nodes.items()}
        with open(os.path.join(dir_path, "tree.pkl"), "wb") as f:
            pickle.dump({
                "nodes": nodes_data,
                "layer_to_nodes": dict(self._layer_to_nodes),
                "num_layers": self._num_layers,
                "next_index": self._next_index,
                "buffer": self._buffer,
                "buffer_metadata": self._buffer_metadata,
                "indexed": self._indexed,
                "raw_token_count": self._raw_token_count,
                "stored_token_count": self._stored_token_count,
                "token_counts": dict(self._token_counts),
                "write_call_count": self._write_call_count,
                "write_times": list(self._write_times),
                "llm_call_times": list(self._llm_call_times),
                "index_build_time": self._index_build_time,
            }, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        import os, pickle
        unit = cls(config, model)
        with open(os.path.join(dir_path, "tree.pkl"), "rb") as f:
            d = pickle.load(f)
        unit._nodes = {idx: _RaptorNode(
            index=n["index"], text=n["text"], embedding=n["embedding"],
            children=set(n["children"]), layer=n["layer"], metadata=n["metadata"],
        ) for idx, n in d["nodes"].items()}
        unit._layer_to_nodes = d["layer_to_nodes"]
        unit._num_layers = d["num_layers"]
        unit._next_index = d["next_index"]
        unit._buffer = d["buffer"]
        unit._buffer_metadata = d["buffer_metadata"]
        unit._indexed = d["indexed"]
        unit._raw_token_count = d["raw_token_count"]
        unit._stored_token_count = d["stored_token_count"]
        unit._token_counts = d["token_counts"]
        unit._write_call_count = d["write_call_count"]
        unit._write_times = d["write_times"]
        unit._llm_call_times = d["llm_call_times"]
        unit._index_build_time = d["index_build_time"]
        return unit

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_indexed()
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]
        cr = (self._raw_token_count / self._stored_token_count
              if self._stored_token_count > 0 else 1.0)

        # Per-layer node counts
        layer_counts = {}
        for layer, indices in sorted(self._layer_to_nodes.items()):
            layer_counts[f"L{layer}"] = len(indices)

        stats.update({
            "compression_ratio": round(cr, 2),
            "raw_token_count": self._raw_token_count,
            "stored_token_count": self._stored_token_count,
            "total_nodes": len(self._nodes),
            "num_layers": self._num_layers + 1,
            "layer_counts": layer_counts,
            "num_leaf_nodes": self.size(),
            "num_raw_utterances": len(self._buffer),
            "avg_llm_call_ms": (
                sum(self._llm_call_times) / len(self._llm_call_times) * 1000
                if self._llm_call_times else 0.0
            ),
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
        total = 0
        for node in self._nodes.values():
            total += len(node.text.encode("utf-8"))
            if node.embedding is not None:
                total += self._enc.dim * 4  # float32
        return total

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return ALL nodes across all layers for quality metrics.

        Includes both leaf nodes (L0 = original chunks) and summary
        nodes (L1+), so Q1 compression ratio reflects the actual stored
        representation (which may be larger than the input due to summaries).
        """
        self._ensure_indexed()
        return [
            MemoryEntry(
                content=n.text,
                metadata={**n.metadata, "layer": n.layer},
            )
            for n in self._nodes.values()
        ]
