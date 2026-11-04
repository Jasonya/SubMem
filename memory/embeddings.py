"""Shared embedding layer — OpenAI API embeddings.

Usage:
    encoder = create_encoder()  # default: text-embedding-3-small (1536d)
    encoder = create_encoder(model_name="text-embedding-3-large")  # 3072d

    embs = encoder.encode(["hello world", "another text"])  # → np.ndarray [N, dim]
    dim = encoder.dim  # 1536 for text-embedding-3-small
"""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np

# Safety margin below OpenAI's 8192-token input limit.
_MAX_INPUT_TOKENS = 7500
# Parallel HTTP workers for embedding calls.
_NUM_WORKERS = 16


# Model name → dimension mapping
_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEncoder:
    """OpenAI API embedding encoder."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._dim = _DIMS.get(model_name, 1536)
        self._client = None
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            import tiktoken
            try:
                self._tokenizer = tiktoken.encoding_for_model(self._model_name)
            except KeyError:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def _split_long_text(self, text: str) -> List[str]:
        """Split text into ≤_MAX_INPUT_TOKENS sub-chunks at token boundary."""
        tok = self._get_tokenizer()
        ids = tok.encode(text, disallowed_special=())
        if len(ids) <= _MAX_INPUT_TOKENS:
            return [text]
        chunks = []
        for i in range(0, len(ids), _MAX_INPUT_TOKENS):
            chunks.append(tok.decode(ids[i:i + _MAX_INPUT_TOKENS]))
        return chunks

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            from pathlib import Path
            env_file = Path(__file__).parent.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip())
            # Embeddings may need to go to a *different* endpoint than chat
            # (e.g. chat → local vLLM which does not serve /v1/embeddings).
            # IMPORTANT: the openai SDK falls back to OPENAI_BASE_URL if we
            # pass base_url=None, so we must pass an explicit URL — either
            # OPENAI_EMBEDDINGS_BASE_URL or the real OpenAI endpoint.
            base_url = (
                os.environ.get("OPENAI_EMBEDDINGS_BASE_URL")
                or "https://api.openai.com/v1"
            )
            api_key = (
                os.environ.get("OPENAI_EMBEDDINGS_API_KEY")
                or os.environ.get("OPENAI_API_KEY", "")
            )
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into L2-normalised embeddings. Returns [N, dim].

        Long texts (>8192 tokens) are split into sub-chunks, embedded
        separately, then token-weighted averaged into a single vector.
        """
        self._ensure_client()

        # Split long texts; track which sub-chunks belong to which original text.
        flat_chunks: List[str] = []
        flat_weights: List[int] = []
        owners: List[int] = []  # parallel to flat_chunks; maps to original idx
        for oi, t in enumerate(texts):
            t = t if t.strip() else " "
            subs = self._split_long_text(t)
            tok = self._get_tokenizer()
            for s in subs:
                flat_chunks.append(s)
                flat_weights.append(max(1, len(tok.encode(s, disallowed_special=()))))
                owners.append(oi)

        # Batch-embed all sub-chunks in parallel. Split by both count and token
        # total so we stay under OpenAI's 300K-tokens-per-request cap.
        max_per_batch = 100
        max_tokens_per_batch = 250_000
        batches: List[List[str]] = []
        cur: List[str] = []
        cur_tok = 0
        for chunk, w in zip(flat_chunks, flat_weights):
            if cur and (len(cur) >= max_per_batch or cur_tok + w > max_tokens_per_batch):
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append(chunk)
            cur_tok += w
        if cur:
            batches.append(cur)

        def _embed_batch(batch):
            resp = self._client.embeddings.create(
                model=self._model_name,
                input=batch,
            )
            return [item.embedding for item in resp.data]

        flat_embs: List[list] = [None] * len(batches)
        if len(batches) == 1:
            flat_embs[0] = _embed_batch(batches[0])
        else:
            with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as ex:
                for idx, embs in zip(
                    range(len(batches)),
                    ex.map(_embed_batch, batches),
                ):
                    flat_embs[idx] = embs
        flat_embs = np.array([e for sub in flat_embs for e in sub], dtype=np.float32)

        # Token-weighted average per original text, then L2 normalise.
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        totals = np.zeros(len(texts), dtype=np.float32)
        for emb, w, oi in zip(flat_embs, flat_weights, owners):
            out[oi] += emb * w
            totals[oi] += w
        totals = np.where(totals == 0, 1.0, totals)
        out /= totals[:, None]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return out / norms


def create_encoder(config=None, model_name: str = None) -> OpenAIEncoder:
    """Create OpenAI embedding encoder.

    Args:
        config: MemoryConfig with config.extra.get("encoder_model")
        model_name: explicit override (takes precedence over config)

    Examples:
        create_encoder()                                          → text-embedding-3-small
        create_encoder(model_name="text-embedding-3-large")       → text-embedding-3-large
    """
    if model_name is None and config is not None:
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        name = extra.get("encoder_model", _DEFAULT_MODEL)
    elif model_name is not None:
        name = model_name
    else:
        name = _DEFAULT_MODEL

    # Strip "openai/" prefix if present
    if name.startswith("openai/"):
        name = name.split("/", 1)[1]

    return OpenAIEncoder(model_name=name)
