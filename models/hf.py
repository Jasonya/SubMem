"""Local HuggingFace model backend.

Supports any AutoModelForCausalLM-compatible model (Qwen, Llama, Mistral, Gemma, etc.).
Handles device_map, quantization (4-bit/8-bit), and flash attention automatically.

Usage:
    --model Qwen/Qwen2.5-7B-Instruct
    --model meta-llama/Llama-3.1-8B-Instruct
    --model google/gemma-2-9b-it
    --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
"""
import re
import torch
from typing import List, Optional
from .base import BaseModel


def get_decoder_layers(hf_model):
    """Locate the transformer decoder layer list for a generic HF model.

    Different families nest layers at different paths:
      Qwen / Llama / Mistral:  hf_model.model.layers
      Gemma 4 multimodal:      hf_model.language_model.layers
      Some wrappers:           hf_model.model.language_model.layers
                                hf_model.language_model.model.layers
      Bare causal LM:          hf_model.layers

    Substrates that hook into attention (C11 SnapKV, C12 EpiCache) need
    this resolver so they don't hardcode `model.model.layers`. Returns
    the ModuleList; raises AttributeError if not found.
    """
    candidates = [
        "model.layers",
        "model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.language_model.layers",
        "layers",
    ]
    for path in candidates:
        obj = hf_model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False; break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise AttributeError(
        f"Cannot locate decoder layers in {type(hf_model).__name__}; "
        f"tried: {candidates}"
    )


class HuggingFaceModel(BaseModel):
    """Generic HuggingFace local model backend."""

    def __init__(self, config):
        self.config = config
        self._model = None
        self._tokenizer = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            trust_remote_code=True,
        )

        # AWQ models: transformers 5.x drops autoawq support (requires gptqmodel,
        # which is unstable on torch 2.10). Route AWQ loads through autoawq
        # directly. Detected by model name suffix "-AWQ" or explicit flag.
        is_awq = (
            str(self.config.name).upper().endswith("-AWQ")
            or getattr(self.config, "use_autoawq", False)
        )
        if is_awq:
            from awq import AutoAWQForCausalLM
            # device="cuda" forces the CUDA kernel; without it, autoawq may
            # try the ipex/CPU kernel and fail (login nodes / no GPU). On
            # compute nodes with H200 this routes through the GPU kernel.
            awq_model = AutoAWQForCausalLM.from_quantized(
                self.config.name,
                trust_remote_code=True,
                fuse_layers=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            # Unwrap: AutoAWQ returns a wrapper with .model being the HF
            # PreTrainedModel that .generate() / forward hooks expect.
            self._model = awq_model.model if hasattr(awq_model, "model") else awq_model
            self._model.eval()
            print(f"  Loaded (autoawq): {self.config.name} (dtype={self._model.dtype})")
            return

        load_kwargs = {
            "trust_remote_code": True,
        }

        # Quantization (mutually exclusive with dtype)
        if getattr(self.config, "load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif getattr(self.config, "load_in_8bit", False):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            load_kwargs["torch_dtype"] = getattr(torch, self.config.dtype)

        # Device mapping
        device_map = getattr(self.config, "device_map", None)
        if device_map:
            load_kwargs["device_map"] = device_map
        else:
            load_kwargs["device_map"] = self.config.device

        # Attention implementation
        attn_impl = getattr(self.config, "attn_implementation", None)
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        load_kwargs["low_cpu_mem_usage"] = True

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            **load_kwargs,
        )
        self._model.eval()
        print(f"  Loaded: {self.config.name} (dtype={self._model.dtype}, "
              f"device_map={device_map or self.config.device})")

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        max_tokens = max_new_tokens or self.config.max_new_tokens

        # Apply chat template if tokenizer supports it (chat/instruct models)
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text_input = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._tokenizer(text_input, return_tensors="pt").to(self._model.device)
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        pad_token_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                top_p=self.config.top_p if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._strip_think_tags(text)

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip <think>...</think> wrapper from reasoning models (Qwen3+, etc.).
        Also handles unclosed <think> tags (truncated output).
        """
        # Closed tags
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        # Unclosed <think> (truncated by max_new_tokens) — remove everything from <think> onward
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
        return text

    def generate_batch(self, prompts: List[str], max_new_tokens: Optional[int] = None) -> List[str]:
        """True batched generation: pad prompts, one forward pass.

        Tunable batch size via env SUBMEM_HF_BATCH_SIZE (default 8). Falls back to
        sequential ``generate`` if the tokenizer/model can't handle padding
        (extremely rare for modern causal LMs)."""
        import os as _os
        if not prompts:
            return []
        bs = int(_os.environ.get("SUBMEM_HF_BATCH_SIZE", "8"))
        bs = max(1, min(bs, len(prompts)))
        if bs == 1:
            return [self.generate(p, max_new_tokens) for p in prompts]

        max_tokens = max_new_tokens or self.config.max_new_tokens
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        # Many causal LMs need left-padding for correct generation indexing
        orig_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        def _render(p: str) -> str:
            if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            return p

        results: List[str] = [""] * len(prompts)
        try:
            for start in range(0, len(prompts), bs):
                chunk = prompts[start:start + bs]
                rendered = [_render(p) for p in chunk]
                enc = self._tokenizer(
                    rendered, return_tensors="pt", padding=True, truncation=False,
                ).to(self._model.device)
                with torch.no_grad():
                    out = self._model.generate(
                        **enc,
                        max_new_tokens=max_tokens,
                        temperature=self.config.temperature if self.config.temperature > 0 else None,
                        top_p=self.config.top_p if self.config.temperature > 0 else None,
                        do_sample=self.config.temperature > 0,
                        pad_token_id=pad_id,
                    )
                input_len = enc["input_ids"].shape[1]
                for j, row in enumerate(out):
                    new_tokens = row[input_len:]
                    text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                    results[start + j] = self._strip_think_tags(text)
        finally:
            self._tokenizer.padding_side = orig_side
        return results

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_hidden_states(self, text: str):
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]

    def get_kv_cache(self, text: str):
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs, use_cache=True)
        return outputs.past_key_values

    @property
    def device(self):
        return self._model.device if self._model else self.config.device

    @property
    def model_name(self) -> str:
        return self.config.name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model
