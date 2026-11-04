"""Base model interface — swap-friendly for any HF-compatible model."""
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseModel(ABC):
    """Abstract base for all LLM backends."""

    @abstractmethod
    def load(self):
        """Load model and tokenizer onto device."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate a single completion."""

    @abstractmethod
    def generate_batch(self, prompts: List[str], max_new_tokens: Optional[int] = None) -> List[str]:
        """Generate completions for a batch of prompts."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""

    @abstractmethod
    def get_hidden_states(self, text: str):
        """Return last-layer hidden states (for embedding / probing)."""

    @abstractmethod
    def get_kv_cache(self, text: str):
        """Return KV cache after processing text (for KV-cache memory units)."""

    @property
    @abstractmethod
    def device(self):
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
