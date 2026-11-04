from .base import BaseModel
from .hf import HuggingFaceModel
from .api import APIModel

# Backward compatibility
QwenModel = HuggingFaceModel

# API providers — any model name starting with these uses APIModel
_API_PROVIDERS = ("openai/", "anthropic/", "local/", "api/")


def create_model(config):
    """Factory: create model from config.

    Routing:
        openai/gpt-4o-mini        -> APIModel
        anthropic/claude-3        -> APIModel
        local/Qwen3.5-9B -> APIModel (custom base URL via OPENAI_BASE_URL)
        api/whatever              -> APIModel
        anything else             -> HuggingFaceModel (local HuggingFace)

    Examples:
        --model Qwen/Qwen3.5-9B       (local HF, default)
        --model meta-llama/Llama-3.1-8B-Instruct (local HF)
        --model google/gemma-2-9b-it             (local HF)
        --model openai/gpt-4o-mini               (API)
    """
    if any(config.name.startswith(p) for p in _API_PROVIDERS):
        return APIModel(config)
    return HuggingFaceModel(config)
