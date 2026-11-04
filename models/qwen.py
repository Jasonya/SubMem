"""Backward compatibility — use models.hf.HuggingFaceModel instead."""
from .hf import HuggingFaceModel as QwenModel  # noqa: F401

__all__ = ["QwenModel"]
