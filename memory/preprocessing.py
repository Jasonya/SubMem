"""Standardized input preprocessing for all memory units.

Provides chunking utilities so C1-C5 receive comparable input.
- C1/C2: per-utterance (no chunking needed, but passes through here for consistency)
- C3/C4/C5: 256-token chunks from accumulated utterances
"""
from typing import Any, Dict, List, Optional, Tuple


def chunk_by_tokens(
    texts: List[str],
    tokenizer_encode,
    tokenizer_decode,
    chunk_size: int = 256,
    overlap: int = 0,
) -> List[str]:
    """Concatenate texts and re-chunk into fixed-size token chunks.

    Args:
        texts: List of raw text segments (e.g., utterances).
        tokenizer_encode: callable(str) -> List[int]
        tokenizer_decode: callable(List[int]) -> str
        chunk_size: Target chunk size in tokens.
        overlap: Number of overlapping tokens between chunks.

    Returns:
        List of chunk strings, each ≤ chunk_size tokens.
    """
    # Concatenate all texts with newline separator
    all_text = "\n".join(texts)
    all_tokens = tokenizer_encode(all_text)

    if not all_tokens:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(all_tokens), step):
        chunk_tokens = all_tokens[i:i + chunk_size]
        chunk_text = tokenizer_decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())

    return chunks


def chunk_by_tokens_with_metadata(
    texts: List[str],
    metadata_list: List[Dict[str, Any]],
    tokenizer_encode,
    tokenizer_decode,
    chunk_size: int = 256,
    overlap: int = 0,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Concatenate texts and re-chunk, tracking which original texts each chunk covers.

    Returns list of (chunk_text, metadata_dict) where metadata_dict contains:
        - dia_ids: list of all dia_id values from the original texts in this chunk
        - session_ids: list of session_id values
        - source_indices: indices into the original texts list
    """
    if not texts:
        return []

    # Compute token boundaries for each text
    sep_tokens = tokenizer_encode("\n")
    sep_len = len(sep_tokens)

    text_boundaries = []  # list of (start_token_idx, end_token_idx, original_idx)
    all_tokens = []
    for idx, text in enumerate(texts):
        start = len(all_tokens)
        tokens = tokenizer_encode(text)
        all_tokens.extend(tokens)
        end = len(all_tokens)
        text_boundaries.append((start, end, idx))
        # Add separator between texts
        if idx < len(texts) - 1:
            all_tokens.extend(sep_tokens)

    if not all_tokens:
        return []

    # Chunk and track metadata
    results = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(all_tokens), step):
        chunk_end = min(i + chunk_size, len(all_tokens))
        chunk_tokens = all_tokens[i:chunk_end]
        chunk_text = tokenizer_decode(chunk_tokens)
        if not chunk_text.strip():
            continue

        # Find which original texts overlap with this chunk
        dia_ids = []
        session_ids = []
        source_indices = []
        for start, end, orig_idx in text_boundaries:
            # Check if this text overlaps with the chunk range [i, chunk_end)
            if start < chunk_end and end > i:
                source_indices.append(orig_idx)
                if orig_idx < len(metadata_list):
                    meta = metadata_list[orig_idx]
                    dia_id = meta.get("dia_id", "")
                    if dia_id:
                        dia_ids.append(dia_id)
                    session_id = meta.get("session_id", "")
                    if session_id and session_id not in session_ids:
                        session_ids.append(session_id)

        chunk_meta = {
            "dia_ids": dia_ids,
            "session_ids": session_ids,
            "source_indices": source_indices,
        }
        results.append((chunk_text.strip(), chunk_meta))

    return results
