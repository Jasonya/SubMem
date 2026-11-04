"""API-based model backend (OpenAI-compatible: GPT, Claude, etc.)

Supports any OpenAI-compatible API endpoint. Used for:
- Memory unit internal LLM calls (C4 gist, C5 fact extraction, C6 entity extraction, etc.)
- Answer generation

Usage:
    export OPENAI_API_KEY=sk-...
    python3 run_c1c2_test.py --model openai/gpt-4o-mini ...
    python3 run_c1c2_test.py --model openai/gpt-4o ...

    # Or with custom base URL (e.g., vLLM, Ollama, etc.)
    export OPENAI_BASE_URL=http://localhost:8000/v1
    python3 run_c1c2_test.py --model local/Qwen3.5-9B ...
"""
import os
from pathlib import Path
from typing import List, Optional

from .base import BaseModel

# Auto-load .env from project root
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


class APIModel(BaseModel):
    """OpenAI-compatible API model backend."""

    def __init__(self, config):
        self.config = config
        self._client = None
        self._tokenizer = None
        # Parse model name: "openai/gpt-4o-mini" → provider="openai", model="gpt-4o-mini"
        parts = config.name.split("/", 1)
        if len(parts) == 2:
            self._provider = parts[0]
            self._api_model = parts[1]
        else:
            self._provider = "openai"
            self._api_model = config.name

    def load(self):
        from openai import OpenAI
        import tiktoken

        # "local/..." provider uses a separate vLLM endpoint so it doesn't
        # fight with OpenAI embeddings/judge that share OPENAI_*.
        if self._provider == "local":
            base_url = os.environ.get("SUBMEM_LOCAL_BASE_URL") \
                or os.environ.get("OPENAI_BASE_URL", None)
            api_key = os.environ.get("SUBMEM_LOCAL_API_KEY", "dummy")
        else:
            base_url = os.environ.get("OPENAI_BASE_URL", None)
            api_key = os.environ.get("OPENAI_API_KEY", "dummy")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Tokenizer for encode/decode (used by metrics)
        try:
            self._tokenizer = tiktoken.encoding_for_model(self._api_model)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        print(f"  API model: {self._api_model} (provider: {self._provider})")

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        import time as _time

        max_tokens = max_new_tokens or self.config.max_new_tokens
        is_new_model = any(x in self._api_model for x in ("gpt-5", "gpt-4.1", "o1", "o3", "o4"))
        kwargs = {
            "model": self._api_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if is_new_model:
            kwargs["max_completion_tokens"] = max_tokens + 4096
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = self.config.temperature
        # Qwen3 family chat template enables thinking by default. With small
        # max_tokens budgets (LoCoMo uses 64), the answer is truncated mid-
        # <think> and the harness sees an unfinished reasoning trace instead
        # of the answer. Disable thinking via vLLM's chat_template_kwargs.
        # (Harmless on non-Qwen3 endpoints — vLLM ignores unknown kwargs.)
        if "Qwen3" in self._api_model:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        # Retry with exponential backoff on rate limit / transient errors
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(k in err_str for k in
                    ("rate_limit", "rate limit", "429", "quota", "timeout",
                     "server_error", "500", "502", "503", "overloaded"))
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                    print(f"  API rate limit/error (attempt {attempt+1}/{max_retries}), "
                          f"retrying in {wait}s: {e}", flush=True)
                    _time.sleep(wait)
                else:
                    print(f"  API error (final): {e}")
                    return ""

    def generate_batch(self, prompts: List[str], max_new_tokens: Optional[int] = None) -> List[str]:
        """Concurrent batch generate — lets a vLLM server's continuous-batching
        scheduler actually co-schedule the requests. Falls back cleanly for
        OpenAI/Anthropic endpoints (concurrent requests are safe within limits).
        """
        if not prompts:
            return []
        # Tune concurrency by env var (default 32 — well under vLLM's typical max).
        max_workers = int(os.environ.get("SUBMEM_API_BATCH_CONCURRENCY", "32"))
        max_workers = max(1, min(max_workers, len(prompts)))
        from concurrent.futures import ThreadPoolExecutor
        results: List[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.generate, p, max_new_tokens): i
                for i, p in enumerate(prompts)
            }
            for fut in future_to_idx:
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"  API batch error at idx={idx}: {e}")
                    results[idx] = ""
        return results

    def encode(self, text: str) -> List[int]:
        try:
            return self._tokenizer.encode(str(text), disallowed_special=())
        except TypeError:
            # Non-tiktoken tokenizers may not accept disallowed_special kwarg.
            return self._tokenizer.encode(str(text))

    def decode(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(token_ids)

    def get_hidden_states(self, text: str):
        raise NotImplementedError("API model does not support hidden states")

    def get_kv_cache(self, text: str):
        raise NotImplementedError("API model does not support KV cache")

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        import re
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    @property
    def device(self):
        return "api"

    @property
    def model_name(self) -> str:
        return self.config.name
