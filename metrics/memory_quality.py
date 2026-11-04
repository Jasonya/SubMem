"""Family Q: Memory Quality (2 metrics) — Q1-Q2.  [LOCAL]

Q1 Compression Ratio (CR)      — original input tokens / stored representation tokens
Q2 Coverage Diversity (CD)      — Shannon entropy of K-means cluster assignment over stored embeddings
"""
import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


def compute_q1(memory_unit, original_token_count: int, tokenizer_encode=None):
    """Q1: Compression Ratio = original input tokens / stored representation tokens.

    Returns None when not applicable (e.g. parametric substrates with no
    text entries, or empty memory). Convention: print None as "–" downstream.
    """
    if original_token_count <= 0:
        return None

    if hasattr(memory_unit, "get_compressed_token_count"):
        stored_tokens = memory_unit.get_compressed_token_count()
        if stored_tokens and stored_tokens > 0:
            return original_token_count / stored_tokens

    entries = memory_unit.get_all_entries()
    if not entries:
        return None

    if tokenizer_encode:
        stored_tokens = sum(len(tokenizer_encode(e.content)) for e in entries)
    else:
        stored_tokens = sum(len(e.content.split()) for e in entries)

    if stored_tokens <= 0:
        return None
    return original_token_count / stored_tokens


def compute_q2(memory_unit, n_clusters: int = 5):
    """Q2: Coverage Diversity — cluster stored embeddings, compute Shannon entropy.

    Returns None when not applicable (no entries, or fewer than 2 entries).
    Convention: print None as "–" downstream.

    Embedding source preference (no duplicate API calls):
      1. If the memory unit already has a populated FAISS index, reconstruct
         the stored vectors from the index (zero extra API cost — these
         embeddings were computed at write time).
      2. If the unit exposes a pre-cached ``_entry_embeddings`` numpy array,
         use that.
      3. Fallback: call ``memory_unit._encode`` to re-embed (was the old
         default; costs extra API calls).
      4. Last resort: TF-IDF.
    """
    entries = memory_unit.get_all_entries()
    if len(entries) < 2:
        return None

    X = None
    # (1) FAISS-backed dense retrieval: pull embeddings back out of the index
    idx = getattr(memory_unit, "_index", None)
    if idx is not None and getattr(idx, "ntotal", 0) >= len(entries):
        try:
            X = np.vstack([idx.reconstruct(i) for i in range(len(entries))])
        except Exception:
            X = None

    # (2) In-memory cache maintained by the unit itself
    if X is None:
        cached = getattr(memory_unit, "_entry_embeddings", None)
        if cached is not None and len(cached) >= len(entries):
            try:
                X = np.asarray(cached[: len(entries)])
            except Exception:
                X = None

    # (3) Re-embed via the unit's encoder (costs API)
    if X is None and hasattr(memory_unit, '_encode'):
        try:
            texts = [e.content for e in entries]
            batch_size = 64
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                all_embs.append(memory_unit._encode(batch))
            X = np.vstack(all_embs)
        except Exception:
            X = None

    use_tfidf = X is None
    if use_tfidf:
        # Sparse/other — use simple TF-IDF as proxy
        from collections import defaultdict
        vocab = defaultdict(int)
        doc_tokens = []
        for e in entries:
            tokens = e.content.lower().split()
            doc_tokens.append(tokens)
            for t in set(tokens):
                vocab[t] += 1

        vocab_list = sorted(vocab.keys())
        vocab_idx = {t: i for i, t in enumerate(vocab_list)}
        n_docs = len(entries)
        X = np.zeros((n_docs, len(vocab_list)))
        for di, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            for t, count in tf.items():
                if t in vocab_idx:
                    idf = math.log(n_docs / (1 + vocab[t]))
                    X[di, vocab_idx[t]] = count * idf

    # K-means clustering — cap k at n//2 to avoid degenerate single-cluster
    # collapse when N is small (e.g. C12 EpiCache with 4 episodes).
    k = max(2, min(n_clusters, len(X) // 2))
    if len(X) < 2:
        return None

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=3, random_state=42)
        labels = km.fit_predict(X)
    except ImportError:
        rng = np.random.RandomState(42)
        labels = rng.randint(0, k, size=len(X))

    counts = Counter(labels)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


class MemoryQualityMetrics:
    """Compute Q1-Q2 from a memory unit after write phase.

    MUST be called AFTER E-family stats are recorded.
    """

    @staticmethod
    def compute(memory_unit, original_token_count: int = 0,
                tokenizer_encode=None) -> Dict[str, float]:
        results = {}
        results["Q1_compression_ratio"] = compute_q1(
            memory_unit, original_token_count, tokenizer_encode
        )
        results["Q2_coverage_diversity"] = compute_q2(memory_unit)
        return results
