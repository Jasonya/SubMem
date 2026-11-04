"""M9 — LoRA Fine-Tuning Memory (Parametric Memory).

Reference: LoRA (Hu et al., 2022)
- "LoRA: Low-Rank Adaptation of Large Language Models"
- https://arxiv.org/abs/2106.09685
- Implementation via HuggingFace PEFT

Core idea: accumulate conversation data → generate QA training pairs via LLM →
           LoRA fine-tune a copy of the model → read via direct forward pass.

Write: buffer text (fast, no GPU)
Train: on first read(), LLM generates QA pairs from buffer → LoRA fine-tune
Read:  direct forward pass on model+adapter — no retrieval, no context

Requires: pip install peft
"""
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit

# ── QA generation prompt ───────────────────────────────────────

_QA_GEN_PROMPT = """Generate question-answer pairs that capture verifiable facts from the conversation below.

STRICT RULES:
1. ONLY generate questions about concrete, verifiable facts: names, dates, places, objects, numbers, actions performed, events that occurred.
2. DO NOT generate questions about: feelings, opinions, attitudes, generic life advice, philosophical statements, or emotional summaries (e.g., "What did X say is great?", "How did X feel?", "What makes X who they are?"). Every question must have a concrete noun as its subject — NO questions about what someone "said about" an abstract concept.
3. Each question MUST be self-contained and unambiguous — include the speaker's name AND a date anchor. Bad: "What type of group helped?" Good: "What type of support group did Caroline attend on 2023-05-07?"
4. Convert all relative time references ("yesterday", "last week") to absolute dates using the reference date below.
5. Answers must be short (1-5 words), factual, and directly stated in the conversation.
6. Avoid duplicate or near-duplicate questions.
7. Include 1-2 multi-hop questions ONLY if the conversation contains two related facts that can be combined using dates EXPLICITLY stated in the text. DO NOT compute day differences, durations, or dates unless both reference points are explicitly given. If you are not 100% sure of the answer, skip the question.
8. Include 1-2 negative-sample questions with answer "I don't know". These MUST be about entities of the same type already mentioned but with a fabricated specific detail (e.g., if "lake sunrise" was mentioned, ask "Did Melanie paint a mountain sunset?"). DO NOT ask about topics, people, or events that might appear in other parts of the conversation you haven't seen.

FORMAT: One pair per line, exactly: Q: <question> | A: <answer>

Reference date: {reference_date}

Conversation:
{text}

QA pairs:"""


@register_memory_unit("M9")
class LoRAMemory(BaseMemoryUnit):
    """LoRA Fine-Tuning Memory — learns facts by training an adapter.

    Uses HuggingFace PEFT to attach a LoRA adapter to the base model.
    The adapter is trained on LLM-generated QA pairs from buffered conversations.
    Read = direct generation from model+adapter (no retrieval).
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self._buffer: List[str] = []
        self._buffer_metadata: List[Dict] = []
        self._qa_pairs: List[Dict[str, str]] = []
        self._lora_model = None
        self._trained = False

        # LoRA config
        extra = config.extra if hasattr(config, "extra") and config.extra else {}
        # Defaults tuned for "memory-as-LoRA" footprint comparison: small
        # rank, attention-only adapters. With these defaults Qwen3-8B yields
        # ~3-5M trainable params (~12-20 MB fp32). Override via extra={...}.
        self._lora_rank = extra.get("lora_rank", 8)
        self._lora_alpha = extra.get("lora_alpha", 16)
        self._lora_lr = extra.get("lora_lr", 1e-4)
        self._lora_epochs = extra.get("lora_epochs", 10)
        self._lora_batch_size = extra.get("lora_batch_size", 4)
        # LoRA paper §7.1 GPT-3 experiments adapt only W_q and W_v at
        # rank=8 alpha=16. We mirror that default. Users can pass
        # target_modules=[...] in config.extra to override (e.g., to
        # include k_proj/o_proj or a full "all-linear" sweep).
        self._target_modules = extra.get("target_modules", [
            "q_proj", "v_proj",
        ])

        # Stats
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._llm_call_times: List[float] = []
        self._train_time_ms: float = 0.0
        self._write_call_count = 0
        self._num_qa_pairs = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }

    def _track_llm_call(self, prompt: str, response: str):
        """Track as preprocessing (not memory construction) for parametric memory."""
        self._token_counts.setdefault("preprocess_llm_input", 0)
        self._token_counts.setdefault("preprocess_llm_output", 0)
        self._token_counts.setdefault("preprocess_llm_calls", 0)
        self._token_counts["preprocess_llm_input"] += len(self.model.encode(prompt))
        self._token_counts["preprocess_llm_output"] += len(self.model.encode(response))
        self._token_counts["preprocess_llm_calls"] += 1

    # ── Write ──────────────────────────────────────────────────

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        self._buffer.append(text)
        self._buffer_metadata.append(metadata or {})
        self._trained = False
        self._write_call_count += 1
        self._write_times.append(time.perf_counter() - t0)

    def write_experience(
        self,
        query: str,
        trajectory: str,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Agentic write: skip LLM QA generation and use (query, trajectory)
        directly as a training pair. Only successful trajectories become
        positive training signal; failed ones are discarded.

        This avoids the 40+ slow self-QA-generation calls in _generate_all_qa
        on 32B models — training pairs are now ready at write time.
        """
        t0 = time.perf_counter()
        md = dict(metadata or {})
        md["success"] = bool(success) if success is not None else None
        if success is False:
            # Skip failures — LoRA on failed trajectories would teach bugs.
            self._buffer_metadata.append(md)  # track for stats
            self._write_call_count += 1
            self._write_times.append(time.perf_counter() - t0)
            return
        # Store pre-formed QA pair directly.
        self._qa_pairs.append({"question": query, "answer": trajectory})
        self._buffer.append(trajectory)
        self._buffer_metadata.append(md)
        self._trained = False
        self._write_call_count += 1
        self._write_times.append(time.perf_counter() - t0)

    # ── QA pair generation ─────────────────────────────────────

    def _generate_qa_pairs(self, text: str, reference_date: str = "") -> List[Dict[str, str]]:
        """Use LLM to generate QA training pairs from conversation text."""
        prompt = _QA_GEN_PROMPT.format(
            text=text,
            reference_date=reference_date or "unknown",
        )
        t0 = time.perf_counter()
        response = self.model.generate(prompt, max_new_tokens=1024)
        self._llm_call_times.append(time.perf_counter() - t0)
        self._track_llm_call(prompt, response)

        pairs = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                q_part = parts[0].strip()
                a_part = parts[1].strip()
                # Strip "Q: " and "A: " prefixes
                if q_part.upper().startswith("Q:"):
                    q_part = q_part[2:].strip()
                if a_part.upper().startswith("A:"):
                    a_part = a_part[2:].strip()
                if q_part and a_part:
                    pairs.append({"question": q_part, "answer": a_part})
        return pairs

    def _generate_all_qa(self):
        """Generate QA pairs from all buffered text."""
        self._qa_pairs.clear()
        # Process in chunks to avoid redundant LLM calls
        # Group every 5 utterances together, pass session date for temporal grounding
        chunk_size = 5
        for i in range(0, len(self._buffer), chunk_size):
            chunk = "\n".join(self._buffer[i:i + chunk_size])
            # Get reference date from metadata of first utterance in chunk
            meta = self._buffer_metadata[i] if i < len(self._buffer_metadata) else {}
            ref_date = meta.get("date", "")
            pairs = self._generate_qa_pairs(chunk, reference_date=ref_date)
            self._qa_pairs.extend(pairs)
        self._num_qa_pairs = len(self._qa_pairs)

    # ── LoRA training ──────────────────────────────────────────

    def _train_lora(self):
        """Train LoRA adapter on generated QA pairs."""
        if self._trained:
            return
        if not self._buffer:
            self._trained = True
            return

        # API-backend path: the adapter is already trained on disk and
        # loaded by vLLM (--enable-lora --lora-modules). We do not need
        # a local HF model here; just mark as trained and return. The
        # read() path will route generation via `adapter_name`.
        if not hasattr(self.model, "_model"):
            self._trained = True
            # Probe likely adapter dirs so get_external_storage_bytes()
            # can report a non-zero E1 for vLLM-served runs.
            import os as _os
            candidates = [
                _os.environ.get("SUBMEM_C9_ADAPTER_DIR", "").strip(),
                _os.environ.get("SUBMEM_C9_ADAPTER_CACHE", "").strip(),
            ]
            for cand in candidates:
                if cand and _os.path.isdir(cand) and \
                        _os.path.isfile(_os.path.join(cand, "adapter_config.json")):
                    self._adapter_cache_dir = cand
                    break
            print("    M9: vLLM backend detected; skipping local LoRA train "
                  "(adapter served by vLLM --enable-lora)", flush=True)
            return

        # Adapter cache: if SUBMEM_C9_ADAPTER_CACHE is set AND the dir already
        # contains a trained adapter, reload it and skip QA generation + train.
        # Caller is responsible for cache-key correctness (same base model +
        # same experience pool + same LoRA hyperparams).
        import os as _os
        cache_dir = _os.environ.get("SUBMEM_C9_ADAPTER_CACHE", "").strip()
        if cache_dir and _os.path.isdir(cache_dir) and \
                _os.path.isfile(_os.path.join(cache_dir, "adapter_config.json")):
            try:
                from peft import PeftModel
                print(f"    LoRA cache HIT: loading adapter from {cache_dir}",
                      flush=True)
                hf_model = self.model._model
                self._lora_model = PeftModel.from_pretrained(hf_model, cache_dir)
                self._lora_model.eval()
                self._trained = True
                self._adapter_cache_dir = cache_dir
                return
            except Exception as e:
                print(f"    LoRA cache load failed ({e}); falling back to train",
                      flush=True)

        # Step 1: QA pairs
        # If write_experience pre-populated _qa_pairs (agentic mode), skip
        # the slow LLM-based generation path. Otherwise fall back to the
        # original LoCoMo-style self-QA generation.
        if not self._qa_pairs:
            self._generate_all_qa()
        else:
            self._num_qa_pairs = len(self._qa_pairs)
            print(f"    LoRA QA pairs (pre-built from experience): "
                  f"{self._num_qa_pairs}", flush=True)
        if not self._qa_pairs:
            self._trained = True
            return

        t0 = time.perf_counter()

        try:
            from peft import LoraConfig, get_peft_model, TaskType

            hf_model = self.model._model
            tokenizer = self.model._tokenizer

            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # QLoRA: if base is loaded in 4-bit (bnb), prepare it so gradient
            # flows through the dequant path. prepare_model_for_kbit_training
            # freezes base weights, casts layernorms to fp32, and enables
            # input-require-grads — required for stable 4-bit LoRA.
            is_4bit = bool(getattr(hf_model, "is_loaded_in_4bit", False))
            if is_4bit:
                from peft import prepare_model_for_kbit_training
                hf_model = prepare_model_for_kbit_training(hf_model)

            # Step 2: Create LoRA config and wrap model.
            # Use regex pattern so we match Gemma4's wrapped Linear modules
            # (Gemma4ClippableLinear) which are not bare nn.Linear and so are
            # rejected by PEFT's class-based default matching when targets are
            # plain strings like "q_proj".
            tm = self._target_modules
            if isinstance(tm, list):
                # Build regex: matches any submodule path ending with
                # one of the requested names. We explicitly exclude any path
                # containing "vision" so multimodal bases (e.g. Gemma-4 26B
                # A4B-it) only adapt the language model — the vision tower
                # uses Gemma4ClippableLinear wrappers that PEFT's class-based
                # match rejects, and we don't need vision adaptation anyway.
                names = "|".join(tm)
                tm_regex = rf"^(?!.*vision).*\.(?:{names})$"
            else:
                tm_regex = tm
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self._lora_rank,
                lora_alpha=self._lora_alpha,
                lora_dropout=0.05,
                target_modules=tm_regex,
                bias="none",
            )

            self._lora_model = get_peft_model(hf_model, lora_config)
            # Enable gradient checkpointing so LoRA fits on a single 80GB/141GB
            # card when the base is Qwen3-32B bf16. Small compute cost, big
            # memory saving (no activation cache for base layers).
            try:
                self._lora_model.gradient_checkpointing_enable()
                if hasattr(self._lora_model, "enable_input_require_grads"):
                    self._lora_model.enable_input_require_grads()
            except Exception:
                pass
            self._lora_model.print_trainable_parameters()

            # Step 3: Prepare training data with answer-only loss masking
            train_prompts = []
            train_full = []
            for qa in self._qa_pairs:
                prompt_part = f"Q: {qa['question']}\nA:"
                full_part = f"Q: {qa['question']}\nA: {qa['answer']}"
                train_prompts.append(prompt_part)
                train_full.append(full_part)

            # Tokenize full texts
            encodings = tokenizer(
                train_full,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            input_ids = encodings["input_ids"].to(hf_model.device)
            attention_mask = encodings["attention_mask"].to(hf_model.device)

            # Tokenize prompts to get lengths for masking
            prompt_encodings = tokenizer(
                train_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )

            labels = input_ids.clone()
            # Mask padding AND prompt tokens — only compute loss on answer
            labels[attention_mask == 0] = -100
            for j in range(len(train_prompts)):
                prompt_len = prompt_encodings["attention_mask"][j].sum().item()
                labels[j, :prompt_len] = -100

            # Step 4: Train
            self._lora_model.train()
            optimizer = torch.optim.AdamW(
                self._lora_model.parameters(),
                lr=self._lora_lr,
                weight_decay=0.01,
            )

            n_samples = len(train_full)
            bs = min(self._lora_batch_size, n_samples)

            best_loss = float('inf')
            for epoch in range(self._lora_epochs):
                total_loss = 0.0
                n_batches = 0
                for start in range(0, n_samples, bs):
                    end = min(start + bs, n_samples)
                    batch_ids = input_ids[start:end]
                    batch_mask = attention_mask[start:end]
                    batch_labels = labels[start:end]

                    fwd_kwargs = dict(
                        input_ids=batch_ids,
                        attention_mask=batch_mask,
                        labels=batch_labels,
                    )
                    # Multimodal bases (e.g. Gemma-4 26B-A4B-it) require an
                    # explicit zero `mm_token_type_ids` tensor at training
                    # time even when no images are present. We detect this
                    # via the model's forward signature (most reliable across
                    # transformers versions) and inject zeros — a no-op for
                    # text-only models.
                    import inspect as _inspect
                    try:
                        sig_params = _inspect.signature(
                            hf_model.forward).parameters
                    except Exception:
                        sig_params = {}
                    if "mm_token_type_ids" in sig_params:
                        fwd_kwargs["mm_token_type_ids"] = torch.zeros_like(
                            batch_ids)
                    elif "token_type_ids" in sig_params:
                        # Cover non-Gemma text-only multimodal-style wrappers
                        cfg = getattr(hf_model, "config", None)
                        if cfg is not None and hasattr(cfg, "vision_config"):
                            fwd_kwargs["token_type_ids"] = torch.zeros_like(
                                batch_ids)

                    outputs = self._lora_model(**fwd_kwargs)
                    loss = outputs.loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"    LoRA epoch {epoch+1}: nan/inf loss, stopping early")
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._lora_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    n_batches += 1
                else:
                    avg_loss = total_loss / max(n_batches, 1)
                    best_loss = min(best_loss, avg_loss)
                    print(f"    LoRA epoch {epoch+1}/{self._lora_epochs}: loss={avg_loss:.4f}")
                    continue
                break  # nan/inf detected

            self._lora_model.eval()
            print(f"    LoRA training done: {self._num_qa_pairs} QA pairs, "
                  f"{self._lora_epochs} epochs")

            # Save adapter to cache dir (if set) so future runs can skip
            # QA generation + training entirely.
            if cache_dir:
                try:
                    _os.makedirs(cache_dir, exist_ok=True)
                    self._lora_model.save_pretrained(cache_dir)
                    self._adapter_cache_dir = cache_dir
                    print(f"    LoRA cache SAVED to {cache_dir}", flush=True)
                except Exception as e:
                    print(f"    WARNING: failed to save adapter cache: {e}",
                          flush=True)

        except ImportError:
            print("  WARNING: peft not installed. M9 falling back to unedited model.")
            print("  Install: pip install peft")
            self._lora_model = None
        except Exception as e:
            print(f"  WARNING: LoRA training failed: {e}")
            self._lora_model = None

        self._train_time_ms = (time.perf_counter() - t0) * 1000
        self._trained = True

    # ── Read (direct forward pass) ─────────────────────────────

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Read = direct generation from model+LoRA adapter.

        No retrieval, no context. The adapter weights contain the memory.

        Two backends:
          1. Local HuggingFace model — forward pass through `self._lora_model`
             (requires `self.model._model` / `_tokenizer`).
          2. vLLM-served adapter — runner wires `c9_adapter` via
             `--enable-lora --lora-modules`; we override the OpenAI `model`
             field to hit that adapter. Triggered when `self.model` lacks
             local HF handles (e.g. APIModel).
        """
        self._train_lora()

        t0 = time.perf_counter()

        use_api_backend = not hasattr(self.model, "_model")
        if use_api_backend:
            # vLLM named-LoRA path. Runner must have launched vLLM with
            # `--enable-lora --lora-modules c9_adapter=<adapter_dir>`; we
            # temporarily flip the client target to that adapter name.
            adapter_name = os.environ.get("SUBMEM_C9_VLLM_ADAPTER", "c9_adapter")
            prompt = (
                f"Question: {query}\n\n"
                "Give a short, concise answer (a few words or a short phrase). "
                "Do not explain your reasoning.\n"
                "Answer:"
            )
            saved = getattr(self.model, "_api_model", None)
            try:
                if saved is not None:
                    self.model._api_model = adapter_name
                answer = self.model.generate(prompt, max_new_tokens=64)
            finally:
                if saved is not None:
                    self.model._api_model = saved
            answer = answer.strip()
            self._read_times.append(time.perf_counter() - t0)
            self._token_counts["read_calls"] += 1
            try:
                self._token_counts["read_context_tokens"] += len(self.model.encode(answer))
            except Exception:
                pass
            return [MemoryEntry(
                content=answer,
                metadata={"source": "parametric_memory", "method": "LoRA",
                          "backend": "vllm_named_lora"},
                score=1.0,
            )]

        gen_model = self._lora_model if self._lora_model is not None else self.model._model
        tokenizer = self.model._tokenizer

        # Inference prompt mirrors the SFT format used in _train_lora:
        # ICL paths (test_time_learning) feed (input → label) pairs at write
        # time as "Q: <input>\nA: <answer>", so the inference probe is
        # "Q: <q>\nA:". Other tasks use the longer QA-style prompt that
        # matches the external pipeline (minus retrieved context).
        extra = getattr(self.config, "extra", {}) or {}
        is_icl = extra.get("benchmark_task_type") == "test_time_learning"
        if is_icl:
            prompt = f"Q: {query}\nA:"
        else:
            prompt = (
                f"Question: {query}\n\n"
                "Give a short, concise answer (a few words or a short phrase). "
                "Do not explain your reasoning.\n"
                "Answer:"
            )

        # Apply chat template for consistency with M1-M6/M10
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text_input = prompt

        inputs = tokenizer(text_input, return_tensors="pt").to(gen_model.device)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        answer = self.model._strip_think_tags(answer)

        self._read_times.append(time.perf_counter() - t0)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += len(self.model.encode(answer))

        return [MemoryEntry(
            content=answer,
            metadata={"source": "parametric_memory", "method": "LoRA"},
            score=1.0,
        )]

    # ── Lifecycle ──────────────────────────────────────────────

    def reset(self):
        self._buffer.clear()
        self._buffer_metadata.clear()
        self._qa_pairs.clear()
        self._trained = False
        self._num_qa_pairs = 0
        self._train_time_ms = 0.0
        self._llm_call_times.clear()
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
        }
        # Remove LoRA adapter — **discard** without merging. The previous
        # code called `merge_and_unload()` which permanently folds the LoRA
        # delta into the base model's linear layers, so subsequent runs in
        # the same Python process start from a contaminated base. Using
        # `unload()` throws the adapter away and leaves `self.model._model`
        # bit-for-bit identical to the state we received it in.
        if self._lora_model is not None:
            try:
                base = self._lora_model.unload()
                # peft's unload() returns the original base model; nothing
                # else to do — we hold no separate reference that needs reset.
                del base
            except Exception:
                # Extremely defensive fallback: at least detach our handle.
                pass
            self._lora_model = None

    def size(self) -> int:
        return len(self._qa_pairs)

    # ── cache support ─────────────────────────────────────────
    REQUIRES_GEN_MODEL_IN_KEY = True   # adapter is base-model-specific
    CACHE_KEY_EXTRA = ["lora_epochs", "lora_rank", "lora_target_modules",
                       "lora_lr", "lora_alpha"]

    def save_state(self, dir_path: str) -> None:
        # Trigger lazy training if not yet done
        if not self._trained:
            self._train_lora()
        import os, json
        os.makedirs(dir_path, exist_ok=True)
        # peft save_pretrained → adapter dir
        if self._lora_model is not None:
            adapter_dir = os.path.join(dir_path, "adapter")
            self._lora_model.save_pretrained(adapter_dir)
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            json.dump({
                "buffer": self._buffer,
                "buffer_metadata": self._buffer_metadata,
                "qa_pairs": self._qa_pairs,
                "trained": self._trained,
                "write_call_count": self._write_call_count,
                "write_times": list(self._write_times),
                "train_time_ms": self._train_time_ms,
                "num_qa_pairs": self._num_qa_pairs,
                "lora_rank": self._lora_rank,
                "lora_alpha": self._lora_alpha,
                "lora_epochs": self._lora_epochs,
                "lora_lr": self._lora_lr,
                "token_counts": dict(self._token_counts),
                "llm_call_times": list(self._llm_call_times),
            }, f)

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        import os, json
        unit = cls(config, model)
        with open(os.path.join(dir_path, "meta.json")) as f:
            d = json.load(f)
        unit._buffer = d["buffer"]
        unit._buffer_metadata = d["buffer_metadata"]
        unit._qa_pairs = d["qa_pairs"]
        unit._trained = d["trained"]
        unit._write_call_count = d["write_call_count"]
        unit._write_times = d["write_times"]
        # Restore training-phase stats so downstream eval can attribute
        # build-time cost without re-training.
        unit._train_time_ms = d.get("train_time_ms", 0.0)
        unit._num_qa_pairs = d.get("num_qa_pairs", len(unit._qa_pairs))
        unit._lora_rank = d.get("lora_rank", unit._lora_rank)
        unit._lora_alpha = d.get("lora_alpha", unit._lora_alpha)
        unit._lora_epochs = d.get("lora_epochs", unit._lora_epochs)
        unit._lora_lr = d.get("lora_lr", unit._lora_lr)
        if "token_counts" in d:
            unit._token_counts.update(d["token_counts"])
        if "llm_call_times" in d:
            unit._llm_call_times = list(d["llm_call_times"])
        adapter_dir = os.path.join(dir_path, "adapter")
        if os.path.isdir(adapter_dir) and hasattr(model, "_model"):
            # Local HF backend: attach PEFT adapter in-process.
            from peft import PeftModel
            base = model._model
            unit._lora_model = PeftModel.from_pretrained(base, adapter_dir)
        elif os.path.isdir(adapter_dir):
            # vLLM / APIModel backend: adapter is served remotely via
            # `--enable-lora --lora-modules c9_adapter=<dir>`. Nothing to
            # load in-process; read() will route through the HTTP client.
            unit._lora_model = None
        return unit

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]

        # Count trainable params
        trainable_params = 0
        if self._lora_model is not None:
            trainable_params = sum(
                p.numel() for p in self._lora_model.parameters() if p.requires_grad
            )

        stats.update({
            "num_qa_pairs": self._num_qa_pairs,
            "lora_rank": self._lora_rank,
            "lora_epochs": self._lora_epochs,
            "trainable_params": trainable_params,
            "train_time_ms": round(self._train_time_ms, 2),
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
            # write latency = per-utterance buffer append (microseconds) PLUS
            # the LoRA training time amortised over the buffer. Without this,
            # E3 would report ~0 for M9 even though training takes minutes —
            # users reading E3 alone would massively underestimate the write
            # cost. We attribute the train_time uniformly across all writes;
            # it is mathematically equivalent to charging the same total cost
            # to E14 (which is what the runner already does) but makes the
            # per-call metric honest.
            "write_latency_ms": self._write_latency_with_training(write_times_ms),
        })
        return stats

    def _write_latency_with_training(self, write_times_ms):
        """Distribute LoRA training time across buffered writes for E3.

        Returns the standard mean/p50/p90 dict, with each write augmented by
        train_time_ms / max(num_writes, 1). The amortisation is uniform
        because LoRA training is a single batch over the whole buffer; we
        cannot attribute different shares to individual writes.
        """
        n_writes = max(len(write_times_ms), 1)
        per_write_train_ms = (self._train_time_ms or 0.0) / n_writes
        adj = [t + per_write_train_ms for t in write_times_ms]
        if not adj:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
        return {
            "mean": float(np.mean(adj)),
            "p50": float(np.median(adj)),
            "p90": float(np.percentile(adj, 90)),
        }

    def get_external_storage_bytes(self) -> int:
        """Storage = LoRA adapter (A/B matrices) only, not base model.

        Local PEFT path: walk live LoRA tensors via named_parameters().
        vLLM-served path (`_lora_model` is None because adapter lives in
        the vLLM server): probe the on-disk adapter directory if known,
        otherwise estimate from rank * 2 * d_model * n_target_modules
        using the architecture metadata stashed during _train_lora.
        """
        if self._lora_model is not None:
            total = 0
            for name, p in self._lora_model.named_parameters():
                if p.requires_grad and ('lora_A' in name or 'lora_B' in name):
                    total += p.numel() * p.element_size()
            return total

        # vLLM-served adapter — measure on-disk size if cache_dir known
        adapter_dir = getattr(self, "_adapter_cache_dir", None)
        if adapter_dir and os.path.isdir(adapter_dir):
            total = 0
            for root, _, files in os.walk(adapter_dir):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            if total > 0:
                return total

        # Last-resort estimate: 2 * rank * hidden_dim * 2 (q,v) * n_layers * 2 bytes (bf16)
        # For Qwen3-32B: 64 layers * 5120 dim * 8 (rank) * 2 (A+B) * 2 (q,v) * 2 bytes ~ 41 MB
        # For Gemma-4-26B (active 4B MoE expert): similar scale.
        try:
            cfg = getattr(self.model, "_hf_config", None) or getattr(getattr(self.model, "_model", None), "config", None)
            if cfg is not None:
                hidden = getattr(cfg, "hidden_size", 0)
                n_layers = getattr(cfg, "num_hidden_layers", 0)
                rank = self._lora_rank
                # 2 matrices (A,B) * 2 target modules (q,v) * rank * hidden * 2 bytes (bf16)
                return 2 * 2 * rank * hidden * n_layers * 2
        except Exception:
            pass
        return 0

    def get_all_entries(self) -> List[MemoryEntry]:
        return [
            MemoryEntry(
                content=f"Q: {qa['question']} A: {qa['answer']}",
                metadata=qa,
            )
            for qa in self._qa_pairs
        ]
