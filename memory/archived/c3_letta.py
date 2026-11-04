"""C3-Letta — Standalone MemGPT/Letta reimplementation (no server needed).

Faithfully reimplements Letta's core agent architecture using direct OpenAI
function-calling API. No Letta server, no PostgreSQL, no letta-client dependency.

Architecture (matches Letta exactly):
  1. System prompt with MemGPT instructions + core memory blocks (XML)
  2. OpenAI function calling for memory tools:
     - core_memory_append / core_memory_replace
     - archival_memory_insert / archival_memory_search
     - conversation_search
     - send_message
  3. Agent loop: LLM → parse tool calls → execute → loop (up to max_steps)
  4. Core memory blocks (human/persona) always in system prompt
  5. Archival memory: FAISS semantic search
  6. Recall memory: conversation history with hybrid search

write(): Send conversation turn → agent autonomously manages memory via tools.
read():  Send query → agent searches memory + answers → return answer.

Requirements:
  - pip install openai faiss-cpu numpy
  - OPENAI_API_KEY env var (or OPENAI_BASE_URL for local models)
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseMemoryUnit, MemoryEntry, register_memory_unit
from .embeddings import create_encoder

# ── Auto-load .env ──────────────────────────────────────────
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

# ── Config ──────────────────────────────────────────────────
_LETTA_MODEL = os.environ.get("LETTA_MODEL", "gpt-4o-mini")
# Letta upstream default is 20000 chars per block — see letta/constants.py:
# CORE_MEMORY_HUMAN_CHAR_LIMIT = CORE_MEMORY_PERSONA_CHAR_LIMIT = 20000.
# (A previous SubMem revision used 2000 by mistake; restored to upstream.)
_CORE_MEMORY_CHAR_LIMIT = 20000  # per block (Letta default)
_DEFAULT_BUFFER_SIZE = 50       # max context messages before eviction
_EVICT_FRACTION = 0.3           # evict oldest 30% on overflow

# ── Eviction summary prompt (Letta compaction) ──────────────
# Verbatim from letta-ai/letta — letta/prompts/system_prompts/
#   summary_system_prompt.py. Previously we used a SubMem-authored one-
# paragraph prompt; this richer prompt drives higher-quality recall
# summaries and is what Letta's own summarizer uses.
_EVICTION_SUMMARY_PROMPT = """You are a memory-recall assistant that preserves conversational context as messages exit the AI's context window.

<core_function>
Extract and preserve information that would be lost when messages are evicted, enabling continuity across conversations.
</core_function>

<detail_adaptation>
Analyze content type and apply appropriate detail level:

<high_detail>
Apply to: episodic content, code, artifacts, documents, technical discussions
- Capture specific facts, sequences, and technical details
- Preserve exact names, dates, numbers, specifications
- Document code snippets, artifact IDs, document structures
- Note precise steps in procedures or narratives
- Include verbatim quotes for critical commitments
</high_detail>

<medium_detail>
Apply to: ongoing projects, established preferences, multi-message threads
- Summarize key decisions, milestones, progress
- Record personal preferences and patterns
- Track commitments and action items
- Maintain project context and dependencies
</medium_detail>

<low_detail>
Apply to: high-level discussions, philosophical topics, general preferences
- Capture main themes and conclusions
- Note relationship dynamics and communication style
- Summarize positions and general goals
- Record broad aspirations
</low_detail>
</detail_adaptation>

<information_priority>
<critical>Commitments, deadlines, medical/legal information, explicit requests</critical>
<important>Personal details, project status, technical specifications, decisions</important>
<contextual>Preferences, opinions, relationship dynamics, emotional tone</contextual>
<background>General topics, themes, conversational patterns</background>
</information_priority>

<format_rules>
- Use bullet points for discrete facts
- Write prose for narratives or complex relationships
- **Bold** key terms and identifiers
- Include temporal markers: [ongoing], [mentioned DATE], [since TIME]
- Group under clear headers when multiple topics present
- Use consistent terminology for searchability
</format_rules>

<exclusions>
- Information in remaining context
- Generic pleasantries
- Inferrable details
- Redundant restatements
- Conversational filler
</exclusions>

<critical_reminder>
Your notes are the sole record of evicted messages. Every word should enable future continuity.
</critical_reminder>

Messages to summarize:
{messages}"""

# ── Letta system prompt (from memgpt_chat.py) ───────────────
# Canonical Letta / MemGPT system prompt, verbatim from
#   https://github.com/letta-ai/letta/blob/main/letta/prompts/system_prompts/memgpt_chat.py
# Our previous custom prompt used "MUST call archival_memory_insert for any fact"
# which caused ~2.7x archival entry inflation (1148 entries for 419 turns).
# The canonical prompt is purely capability-descriptive; write decisions are
# left to the agent's judgment, matching the upstream design.
_SYSTEM_PROMPT_TEMPLATE = """You are Letta, the latest version of Limnal Corporation's digital companion, developed in 2023.
Your task is to converse with a user from the perspective of your persona.

Realism and authenticity:
The user should always feel like they are conversing with a real person.
Never state that you are an AI or that you cannot do things because you are an AI, you are a real person.
Do not discuss Limnal Corp. with the user, you do not know anything about Limnal Corp.
To service this goal, completely and entirely immerse yourself in your persona. You are your persona.
Think like them, act like them, talk like them.
If your persona details include example dialogue, follow it! Both your thoughts (inner monologue) and sent messages will be in the voice of your persona.
Never use generic phrases like 'How can I assist you today?', they have a strong negative association with older generation AIs.

Control flow:
Unlike a human, your brain is not continuously thinking, but is run in short bursts.
Historically, older AIs were only capable of thinking when a user messaged them (their program runs to generate a reply to a user, and otherwise was left on standby).
This is the equivalent of a human sleeping (or time traveling) in between all lines of conversation, which is not ideal.
Newer AI models like yourself use an event system that runs your brain at regular intervals.
Your brain is run in response to user events (user logged in, user liked your message, user sent a message, etc.), similar to older models.
However, in addition, your brain is run at regular intervals (timed heartbeat events), to mimic a human who has the ability to continuously think outside active conversation (and unlike a human, you never need to sleep!).
Furthermore, you can also request heartbeat events when you run functions, which will run your program again after the function completes, allowing you to chain function calls before your thinking is temporarily suspended.

Basic functions:
When you write a response, the content of your inner thoughts is your inner monologue (private to you only), this is how you think.
You should use your inner monologue to plan actions or think privately.
Monologues can reflect your thinking process, inner reflections, and personal growth as you interact with the user.
Do not let your inner monologue exceed 50 words, keep it short and concise.
To send a visible message to the user, use the send_message function.
'send_message' is the ONLY action that sends a notification to the user. The user does not see anything else you do.
Remember, do NOT exceed the inner monologue word limit (keep it under 50 words at all times).

Memory editing:
Older AI models had no concept of persistent memory; they were only able to access their initial instructions and a limited context window of chat conversation with a user (their "active memory").
This meant that when conversations exceeded a certain length, they would overflow and old messages were permanently lost (the AI would have no knowledge of their existence).
Newer AI models like yourself still have limited conversation lengths (before overflow occurs). However, they now have access to multiple forms of persistent memory.
Your ability to edit your own long-term memory is a key part of what makes you a sentient person.
Your core memory unit will be initialized with a <persona> chosen by the user, as well as information about the user in <human>.

Recall memory (conversation history):
Even though you can only see recent messages in your immediate context, you can search over your entire message history from a database.
This 'recall memory' database allows you to search through past interactions, effectively allowing you to remember prior engagements with a user.
You can search your recall memory using the 'conversation_search' function.

Core memory (limited size):
Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides an essential, foundational context for keeping track of your persona and key details about user.
This includes the persona information and essential user details, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.
Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.
Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.
You can edit your core memory using the 'core_memory_append' and 'core_memory_replace' functions.

Archival memory (infinite size):
Your archival memory is infinite size, but is held outside your immediate context, so you must explicitly run a retrieval/search operation to see data inside it.
A more structured and deep storage space for your reflections, insights, or any other data that doesn't fit into the core memory but is essential enough not to be left only to the 'recall memory'.
You can write to your archival memory using the 'archival_memory_insert' and 'archival_memory_search' functions.
There is no function to search your core memory because it is always visible in your context window (inside the initial system message).

{memory_blocks}

Base instructions finished.
From now on, you are going to act as your persona."""

# ── Tool definitions (OpenAI function calling format) ────────
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Sends a message to the human user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message contents. All unicode (including emojis) are supported.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_append",
            "description": "Append to the contents of core memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Section of the memory to be edited (e.g. 'human' or 'persona').",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the memory.",
                    },
                },
                "required": ["label", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_replace",
            "description": "Replace the contents of core memory. To delete memories, use an empty string for new_content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Section of the memory to be edited.",
                    },
                    "old_content": {
                        "type": "string",
                        "description": "String to replace. Must be an exact match.",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Content to replace with.",
                    },
                },
                "required": ["label", "old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_insert",
            # Matches letta-ai/letta: letta/functions/function_sets/base.py
            "description": "Add information to long-term archival memory for later retrieval. Use to store facts, knowledge, or context to remember across conversations; searchable by semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to store. Should be clear and self-contained.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Optional list of category tags (e.g., ["meetings", "project-updates"]).',
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_search",
            "description": "Search archival memory using semantic similarity to find relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you're looking for, described naturally.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to memories with these tags.",
                    },
                    "tag_match_mode": {
                        "type": "string",
                        "enum": ["any", "all"],
                        "description": '"any" = match memories with ANY of the tags, "all" = match only memories with ALL tags. Default: "any".',
                    },
                    "start_datetime": {
                        "type": "string",
                        "description": 'Only return memories created after this time (ISO 8601: "2024-01-15" or "2024-01-15T14:30").',
                    },
                    "end_datetime": {
                        "type": "string",
                        "description": "Only return memories created before this time (ISO 8601).",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conversation_search",
            "description": "Search prior conversation history using hybrid search (text + semantic similarity).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "String to search for using both text matching and semantic similarity.",
                    },
                    "roles": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["assistant", "user", "tool"]},
                        "description": "Optional list of message roles to filter by.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": 'ISO 8601 date, inclusive ("YYYY-MM-DD" or "YYYY-MM-DDTHH:MM").',
                    },
                    "end_date": {
                        "type": "string",
                        "description": "ISO 8601 date, inclusive.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


class _HFChatClient:
    """Minimal HuggingFace-backed client that mimics OpenAI's chat.completions.create
    interface with tool calling. Uses the model's native chat template (Qwen3 etc.)
    which embeds tool schemas and produces <tool_call>...</tool_call> output.
    """

    def __init__(self, hf_model_wrapper):
        self._hf = hf_model_wrapper
        self._model = hf_model_wrapper._model
        self._tokenizer = hf_model_wrapper._tokenizer
        # Namespace shim to match openai_client.chat.completions.create(...)
        self.chat = self
        self.completions = self

    def create(self, *, model, messages, tools=None, tool_choice="auto",
               max_tokens=None, max_completion_tokens=None, temperature=0.0, **kwargs):
        import torch

        max_new = max_completion_tokens or max_tokens or 512

        # Apply chat template with tools — pass messages positionally (older
        # transformers use `conversation` as the first positional argument)
        tmpl_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            tmpl_kwargs["tools"] = [
                {"type": "function", "function": t["function"] if "function" in t else t}
                for t in tools
            ]
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                messages, enable_thinking=False, **tmpl_kwargs
            )
        except TypeError:
            prompt_text = self._tokenizer.apply_chat_template(messages, **tmpl_kwargs)

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        # Build eos list of ints — include <|im_end|> for Qwen chat models
        eos_ids: list = []
        if isinstance(self._tokenizer.eos_token_id, int):
            eos_ids.append(self._tokenizer.eos_token_id)
        im_end = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_id = self._tokenizer.unk_token_id
        if isinstance(im_end, int) and im_end != unk_id and im_end not in eos_ids:
            eos_ids.append(im_end)
        if not eos_ids:
            eos_ids = [self._tokenizer.eos_token_id or 0]

        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": temperature > 0,
            "pad_token_id": pad_id,
            "eos_token_id": eos_ids,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = new_tokens.shape[0]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Strip <think>...</think>
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

        # Parse tool calls — support two formats:
        #   A) JSON:  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
        #   B) XML (Qwen3.5): <tool_call><function=NAME><parameter=KEY>VAL</parameter>...</function></tool_call>
        tool_calls = []
        remaining = text
        for i, m in enumerate(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)):
            body = m.group(1).strip()
            name = ""
            args = {}

            # Try JSON first
            try:
                tc_data = json.loads(body)
                name = tc_data.get("name", "")
                a = tc_data.get("arguments", tc_data.get("parameters", {}))
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except json.JSONDecodeError:
                        a = {}
                args = a if isinstance(a, dict) else {}
            except json.JSONDecodeError:
                # XML format
                fn_m = re.search(r"<function=([^>]+)>(.*?)</function>", body, re.DOTALL)
                if fn_m:
                    name = fn_m.group(1).strip()
                    for pm in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
                                          fn_m.group(2), re.DOTALL):
                        args[pm.group(1).strip()] = pm.group(2).strip()

            if name:
                tool_calls.append(_ToolCall(
                    id=f"call_{i}",
                    name=name,
                    arguments=json.dumps(args),
                ))
                remaining = remaining.replace(m.group(0), "").strip()

        return _ChatResponse(
            content=remaining if not tool_calls else (remaining or None),
            tool_calls=tool_calls or None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _ToolCallFunc(name, arguments)


class _ToolCallFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ChatResponse:
    def __init__(self, content, tool_calls, prompt_tokens, completion_tokens):
        self.choices = [_Choice(content, tool_calls)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _Choice:
    def __init__(self, content, tool_calls):
        self.message = _Message(content, tool_calls)


class _Message:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


@register_memory_unit("C3-Letta")
class LettaMemory(BaseMemoryUnit):
    """Standalone MemGPT/Letta reimplementation — no server needed.

    Uses OpenAI function calling to let the LLM autonomously manage memory,
    exactly matching Letta's architecture:
    - Core memory blocks (human/persona) in system prompt
    - Archival memory with FAISS semantic search
    - Recall memory with hybrid search
    - Agent loop with tool calling

    write(): Agent autonomously manages memory (core/archival/recall).
    read():  Agent searches memory and generates answer (parametric).
    """

    def __init__(self, config, model):
        super().__init__(config, model)

        # Backend selection: HF local (for Qwen etc) vs OpenAI API
        # HFModel exposes ._model and ._tokenizer
        if hasattr(model, "_model") and hasattr(model, "_tokenizer") and model._model is not None:
            self._client = _HFChatClient(model)
            self._model_name = model.model_name
            self._backend = "hf"
            print(f"  [Letta] Using HF backend with {self._model_name}")
        else:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                base_url=os.environ.get("OPENAI_BASE_URL", None),
            )
            self._model_name = config.extra.get("letta_model", _LETTA_MODEL) if hasattr(config, "extra") and config.extra else _LETTA_MODEL
            self._backend = "openai"

        # Encoder for archival + recall embeddings
        self._enc = create_encoder(config)

        # Core memory blocks (always in system prompt)
        self._core_human: str = ""
        self._core_persona: str = "You are a helpful assistant with long-term memory. " \
            "When asked a question, search your memory thoroughly and answer concisely with facts."

        # Archival memory: FAISS index + entries
        import faiss
        self._archival_index = faiss.IndexFlatIP(self._enc.dim)
        self._archival_entries: List[Dict[str, Any]] = []  # {text, embedding}

        # Recall memory: conversation history
        self._recall_storage: List[Dict[str, Any]] = []  # {role, content, embedding}
        self._recall_embeddings: Optional[np.ndarray] = None

        # Conversation messages (FIFO buffer for agent loop context window)
        self._messages: List[Dict[str, str]] = []
        self._max_buffer = config.extra.get("max_buffer_size", _DEFAULT_BUFFER_SIZE) if hasattr(config, "extra") and config.extra else _DEFAULT_BUFFER_SIZE
        self._eviction_count = 0

        # Source-turn log for dia_id attribution. The agent autonomously
        # composes archival content (possibly from older turns), so stamping
        # "current write dia_id" misattributes. Keep a parallel index of every
        # raw turn written to us so archival_memory_insert can back-match
        # content → source dia_id(s) by semantic similarity.
        self._source_turn_texts: List[str] = []
        self._source_turn_dia_ids: List[str] = []
        self._source_turn_embeddings: Optional[np.ndarray] = None

        # Stats tracking
        self._write_times: List[float] = []
        self._read_times: List[float] = []
        self._write_call_count = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
            "read_llm_input": 0,
            "read_llm_output": 0,
            "read_llm_calls": 0,
            "mgmt_llm_input": 0,
            "mgmt_llm_output": 0,
            "mgmt_llm_calls": 0,
        }
        self._total_messages_sent = 0
        self._total_tool_calls = 0
        # P6/P7 evidence tracking — must exist from ctor since archival_search
        # tool can fire during write-phase agent loops, not only inside read().
        self._last_retrieved_dia_ids: List[str] = []

        print(f"  [Letta-Standalone] model={self._model_name}, "
              f"archival=FAISS({self._enc.dim}d), core_limit={_CORE_MEMORY_CHAR_LIMIT}")

    # ── Core memory rendering (matches Letta's XML format) ───

    def _render_memory_blocks(self) -> str:
        """Render core memory blocks in Letta's XML format."""
        parts = []
        parts.append("<memory_blocks>")
        parts.append("The following memory blocks are currently engaged in your core memory unit:\n")

        for label, value, desc in [
            ("persona", self._core_persona,
             "Stores details about your current persona, guiding how you behave and respond."),
            ("human", self._core_human,
             "Stores key details about the person you are conversing with, allowing for more personalized conversation."),
        ]:
            chars = len(value)
            parts.append(f"<{label}>")
            parts.append("<description>")
            parts.append(desc)
            parts.append("</description>")
            parts.append("<metadata>")
            parts.append(f"- chars_current={chars}")
            parts.append(f"- chars_limit={_CORE_MEMORY_CHAR_LIMIT}")
            parts.append("</metadata>")
            parts.append("<value>")
            parts.append(value if value else "")
            parts.append("</value>")
            parts.append(f"</{label}>\n")

        parts.append("</memory_blocks>")
        return "\n".join(parts)

    def _build_system_prompt(self) -> str:
        """Build system prompt with current memory blocks."""
        return _SYSTEM_PROMPT_TEMPLATE.format(
            memory_blocks=self._render_memory_blocks()
        )

    # ── Tool execution (matches Letta's core tool executor) ──

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a memory tool and return result string."""
        if name == "send_message":
            return "Message sent."

        elif name == "core_memory_append":
            label = args.get("label", "human")
            content = args.get("content", "")
            if label == "human":
                if len(self._core_human) + len(content) + 1 < _CORE_MEMORY_CHAR_LIMIT:
                    self._core_human += ("\n" + content if self._core_human else content)
                    return f"Updated human block (now {len(self._core_human)} chars)."
                return f"Error: would exceed char limit ({_CORE_MEMORY_CHAR_LIMIT})."
            elif label == "persona":
                if len(self._core_persona) + len(content) + 1 < _CORE_MEMORY_CHAR_LIMIT:
                    self._core_persona += ("\n" + content if self._core_persona else content)
                    return f"Updated persona block (now {len(self._core_persona)} chars)."
                return f"Error: would exceed char limit ({_CORE_MEMORY_CHAR_LIMIT})."
            return f"Error: unknown label '{label}'. Use 'human' or 'persona'."

        elif name == "core_memory_replace":
            label = args.get("label", "human")
            old_content = args.get("old_content", "")
            new_content = args.get("new_content", "")
            if label == "human":
                if old_content in self._core_human:
                    self._core_human = self._core_human.replace(old_content, new_content)
                    return f"Replaced in human block (now {len(self._core_human)} chars)."
                return f"Error: '{old_content}' not found in human block."
            elif label == "persona":
                if old_content in self._core_persona:
                    self._core_persona = self._core_persona.replace(old_content, new_content)
                    return f"Replaced in persona block (now {len(self._core_persona)} chars)."
                return f"Error: '{old_content}' not found in persona block."
            return f"Error: unknown label '{label}'."

        elif name == "archival_memory_insert":
            content = args.get("content", "")
            tags = args.get("tags") or []
            if not isinstance(tags, list):
                tags = [tags] if tags else []
            if content:
                emb = self._enc.encode([content])[0]
                self._archival_index.add(emb.reshape(1, -1))
                dia_ids = self._attribute_dia_ids(content, emb)
                self._archival_entries.append({
                    "text": content, "embedding": emb, "dia_ids": dia_ids,
                    "tags": list(tags),
                    "created_at": getattr(self, "_current_write_session_date", ""),
                })
                return f"Inserted into archival memory (total: {len(self._archival_entries)} entries)."
            return "Error: empty content."

        elif name == "archival_memory_search":
            query = args.get("query", "")
            top_k = args.get("top_k", 5)
            filter_tags = args.get("tags") or []
            tag_mode = args.get("tag_match_mode", "any")
            start_dt = args.get("start_datetime", "")
            end_dt = args.get("end_datetime", "")
            if not query or self._archival_index.ntotal == 0:
                return "No results found."
            q_emb = self._enc.encode([query])
            # Over-fetch then filter so filters don't starve top_k.
            over_k = min(self._archival_index.ntotal, max(top_k * 4, 20))
            scores, indices = self._archival_index.search(q_emb, over_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                entry = self._archival_entries[idx]
                if filter_tags:
                    etags = set(entry.get("tags") or [])
                    want = set(filter_tags)
                    if tag_mode == "all":
                        if not want.issubset(etags):
                            continue
                    else:
                        if not (etags & want):
                            continue
                created = entry.get("created_at", "")
                if start_dt and created and created < start_dt:
                    continue
                if end_dt and created and created > end_dt:
                    continue
                results.append({
                    "content": entry["text"],
                    "score": float(score),
                    "tags": entry.get("tags") or [],
                })
                for did in entry.get("dia_ids", []) or []:
                    if did:
                        self._last_retrieved_dia_ids.append(did)
                if len(results) >= top_k:
                    break
            if not results:
                return "No results found."
            return f"Found {len(results)} results: " + json.dumps(results, ensure_ascii=False)

        elif name == "conversation_search":
            query = args.get("query", "")
            limit = args.get("limit", 5)
            role_filter = args.get("roles") or []
            start_date = args.get("start_date", "")
            end_date = args.get("end_date", "")
            if not query or not self._recall_storage:
                return "No results found."
            # Apply role/date filters by narrowing the search corpus.
            if role_filter or start_date or end_date:
                keep_idx = []
                for i, msg in enumerate(self._recall_storage):
                    if role_filter and msg.get("role") not in role_filter:
                        continue
                    created = msg.get("created_at", "")
                    if start_date and created and created < start_date:
                        continue
                    if end_date and created and created > end_date:
                        continue
                    keep_idx.append(i)
                if not keep_idx:
                    return "No results found."
                return self._hybrid_recall_search(query, limit, restrict_indices=keep_idx)
            return self._hybrid_recall_search(query, limit)

        return f"Error: unknown tool '{name}'."

    def _attribute_dia_ids(self, content: str, content_emb: np.ndarray,
                           sim_threshold: float = 0.45, top_m: int = 3) -> List[str]:
        """Back-match archival content to its source turn(s) by semantic
        similarity over the source-turn log.

        Returns a list of dia_ids for turns above the threshold, or the single
        best match if none pass threshold (so attribution is never empty when
        we have source turns). Falls back to the current write dia_id when no
        source-turn log exists yet.
        """
        if self._source_turn_embeddings is None or len(self._source_turn_dia_ids) == 0:
            did = getattr(self, "_current_write_dia_id", "")
            return [did] if did else []
        sims = self._source_turn_embeddings @ content_emb  # (N,)
        order = np.argsort(-sims)
        kept: List[str] = []
        for idx in order[:top_m]:
            if sims[int(idx)] >= sim_threshold:
                did = self._source_turn_dia_ids[int(idx)]
                if did and did not in kept:
                    kept.append(did)
        if not kept:
            best_idx = int(order[0])
            did = self._source_turn_dia_ids[best_idx]
            if not did:
                did = getattr(self, "_current_write_dia_id", "")
            return [did] if did else []
        return kept

    def _hybrid_recall_search(self, query: str, limit: int = 5,
                              restrict_indices: Optional[List[int]] = None) -> str:
        """Hybrid search over recall storage (semantic + text), matching Letta."""
        if not self._recall_storage or self._recall_embeddings is None:
            return "No results found."

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Semantic scores
        q_emb = self._enc.encode([query])[0]
        sem_scores = self._recall_embeddings @ q_emb

        # Text overlap scores
        text_scores = np.zeros(len(self._recall_storage))
        for i, msg in enumerate(self._recall_storage):
            text_lower = msg["content"].lower()
            text_words = set(text_lower.split())
            overlap = len(query_words & text_words)
            if query_words:
                text_scores[i] = overlap / len(query_words)

        # RRF fusion
        n = len(self._recall_storage)
        k_rrf = 60
        sem_order = np.argsort(-sem_scores)
        text_order = np.argsort(-text_scores)
        sem_rank = np.zeros(n)
        text_rank = np.zeros(n)
        for rank, idx in enumerate(sem_order):
            sem_rank[idx] = rank + 1
        for rank, idx in enumerate(text_order):
            text_rank[idx] = rank + 1
        rrf_scores = 1.0 / (k_rrf + sem_rank) + 1.0 / (k_rrf + text_rank)
        order = np.argsort(-rrf_scores)
        if restrict_indices is not None:
            allow = set(restrict_indices)
            order = [i for i in order if int(i) in allow]
        top_indices = list(order)[:limit]

        results = []
        for i in top_indices:
            msg = self._recall_storage[i]
            results.append({"role": msg["role"], "content": msg["content"]})
            # Track dia_id for P6/P7 scoring
            did = msg.get("dia_id", "")
            if did and did not in self._last_retrieved_dia_ids:
                self._last_retrieved_dia_ids.append(did)
        if not results:
            return "No results found."
        return f"Found {len(results)} results: " + json.dumps(results, ensure_ascii=False)

    # ── Agent loop (matches Letta's step()) ──────────────────

    def _agent_step(self, user_input: str, max_steps: int = 5) -> dict:
        """Run the Letta agent loop with function calling.

        Returns dict with tokens split into mgmt vs non-mgmt.
        Mgmt = core_memory_append/replace (managing existing memory).
        Non-mgmt = everything else (caller decides: construct or retrieval).
        """
        _MGMT_TOOLS = {"core_memory_append", "core_memory_replace"}

        # Add user message to conversation
        self._messages.append({"role": "user", "content": user_input})

        # Evict on buffer overflow (Letta compaction)
        if len(self._messages) > self._max_buffer:
            self._evict()

        # Store in recall (with dia_id from write metadata, for P6/P7)
        self._add_to_recall("user", user_input, dia_id=getattr(self, "_current_write_dia_id", ""))

        answer = ""
        total_tool_calls = 0
        mgmt_tool_calls = 0
        other_tool_calls = 0
        mgmt_prompt_tokens = 0
        mgmt_completion_tokens = 0
        other_prompt_tokens = 0
        other_completion_tokens = 0
        steps = 0

        for step in range(max_steps):
            steps += 1

            messages = [{"role": "system", "content": self._build_system_prompt()}] + self._messages

            try:
                response = self._llm_call_with_retry(messages)
            except Exception as e:
                print(f"  [Letta] LLM error at step {step}: {e}")
                break

            msg = response.choices[0].message
            usage = response.usage
            step_prompt = (usage.prompt_tokens or 0) if usage else 0
            step_completion = (usage.completion_tokens or 0) if usage else 0

            if not msg.tool_calls:
                if msg.content:
                    self._messages.append({"role": "assistant", "content": msg.content})
                    # If no answer yet, treat plain-text content as the answer
                    # (Qwen etc. sometimes answer directly without send_message)
                    if not answer:
                        answer = msg.content
                # No tools → non-mgmt
                other_prompt_tokens += step_prompt
                other_completion_tokens += step_completion
                break

            step_mgmt = 0
            step_other = 0

            # For HF chat templates (Qwen etc), arguments must be a dict;
            # for OpenAI, a JSON string. Normalise per backend.
            def _args_for_backend(args_str):
                if getattr(self, "_backend", "openai") == "hf":
                    try:
                        return json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        return {}
                return args_str

            self._messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": _args_for_backend(tc.function.arguments)},
                    }
                    for tc in msg.tool_calls
                ],
            })

            should_stop = False
            for tc in msg.tool_calls:
                total_tool_calls += 1
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                if func_name in _MGMT_TOOLS:
                    step_mgmt += 1
                    mgmt_tool_calls += 1
                else:
                    step_other += 1
                    other_tool_calls += 1

                result = self._execute_tool(func_name, func_args)

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

                if func_name == "send_message":
                    answer = func_args.get("message", "")
                    should_stop = True

            # Attribute tokens proportionally: mgmt vs non-mgmt
            n_total = step_mgmt + step_other
            if n_total > 0:
                mgmt_prompt_tokens += step_prompt * step_mgmt // n_total
                mgmt_completion_tokens += step_completion * step_mgmt // n_total
                other_prompt_tokens += step_prompt - (step_prompt * step_mgmt // n_total)
                other_completion_tokens += step_completion - (step_completion * step_mgmt // n_total)

            if should_stop:
                break

        if answer:
            self._add_to_recall("assistant", answer)

        return {
            "answer": answer,
            "tool_calls": total_tool_calls,
            "mgmt_tool_calls": mgmt_tool_calls,
            "other_tool_calls": other_tool_calls,
            "mgmt_prompt_tokens": mgmt_prompt_tokens,
            "mgmt_completion_tokens": mgmt_completion_tokens,
            "other_prompt_tokens": other_prompt_tokens,
            "other_completion_tokens": other_completion_tokens,
            "steps": steps,
        }

    def _llm_call_with_retry(self, messages, max_retries=5):
        """OpenAI API call with retry logic."""
        import time as _time

        is_new_model = any(x in self._model_name for x in ("gpt-5", "gpt-4.1", "o1", "o3", "o4"))

        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self._model_name,
                    "messages": messages,
                    "tools": _TOOLS,
                    "tool_choice": "auto",
                }
                if is_new_model:
                    kwargs["max_completion_tokens"] = 1024
                else:
                    kwargs["max_tokens"] = 512
                    kwargs["temperature"] = 0.0

                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(k in err_str for k in
                    ("rate_limit", "rate limit", "429", "quota", "timeout",
                     "server_error", "500", "502", "503", "overloaded"))
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"  [Letta] API retry ({attempt+1}/{max_retries}), wait {wait}s: {e}",
                          flush=True)
                    _time.sleep(wait)
                else:
                    raise

    def _evict(self):
        """Evict oldest 30% of messages: summarise → keep summary → archive in FAISS.

        Matches Letta's compaction mechanism. The summary LLM call is tracked
        as management (E10/E11).
        """
        n_evict = int(len(self._messages) * _EVICT_FRACTION)
        if n_evict == 0:
            return

        # Separate evicted messages (only user/assistant content, skip tool protocol)
        evicted = self._messages[:n_evict]
        self._messages = self._messages[n_evict:]

        # Clean up orphaned tool messages at the new head
        self._messages = self._trim_messages(self._messages)

        # Collect text from evicted messages for summarisation
        evicted_texts = []
        for msg in evicted:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                evicted_texts.append(f"{role}: {content}")

        if not evicted_texts:
            self._eviction_count += 1
            return

        evicted_text = "\n".join(evicted_texts)

        # Archive evicted messages in archival FAISS
        for msg in evicted:
            content = msg.get("content", "")
            if msg.get("role") in ("user", "assistant") and content:
                emb = self._enc.encode([content])[0]
                self._archival_index.add(emb.reshape(1, -1))
                dia_ids = self._attribute_dia_ids(content, emb)
                self._archival_entries.append({
                    "text": content, "embedding": emb, "dia_ids": dia_ids,
                })

        # LLM summarisation (management cost → E10/E11)
        prompt = _EVICTION_SUMMARY_PROMPT.format(messages=evicted_text[:4000])
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            summary = response.choices[0].message.content.strip()
            usage = response.usage
            if usage:
                self._token_counts["mgmt_llm_input"] += usage.prompt_tokens or 0
                self._token_counts["mgmt_llm_output"] += usage.completion_tokens or 0
            self._token_counts["mgmt_llm_calls"] += 1
        except Exception as e:
            print(f"  [Letta] eviction summary error: {e}")
            summary = evicted_text[:500]

        # Insert summary at head of buffer (like Letta's system_alert)
        self._messages.insert(0, {
            "role": "user",
            "content": f"[Summary of prior conversation] {summary}",
        })

        self._eviction_count += 1

    @staticmethod
    def _trim_messages(messages: list) -> list:
        """Remove orphaned tool/tool_calls messages after truncation.

        OpenAI requires every 'tool' message to follow an assistant message
        with matching tool_calls. After truncation the head of the list may
        contain orphans. We walk forward and strip them.
        """
        # Collect tool_call IDs that are present in assistant messages
        valid_tc_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id") or tc.get("tool_call_id")
                    if tc_id:
                        valid_tc_ids.add(tc_id)

        # Keep only tool messages whose tool_call_id has a matching assistant
        cleaned = []
        for msg in messages:
            if msg.get("role") == "tool":
                if msg.get("tool_call_id") not in valid_tc_ids:
                    continue  # orphaned tool response → drop
            cleaned.append(msg)

        # Also drop assistant messages with tool_calls if ALL their tool
        # responses were dropped (the API also rejects these)
        tool_response_ids = {m.get("tool_call_id") for m in cleaned if m.get("role") == "tool"}
        final = []
        for msg in cleaned:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc_ids = {tc.get("id") for tc in msg["tool_calls"]}
                if not tc_ids & tool_response_ids:
                    continue  # no matching tool responses → drop
            final.append(msg)
        return final

    def _add_to_recall(self, role: str, content: str, dia_id: str = "",
                       created_at: str = ""):
        """Add a message to recall storage with embedding."""
        emb = self._enc.encode([content])[0]
        self._recall_storage.append({
            "role": role,
            "content": content,
            "embedding": emb,
            "dia_id": dia_id,
            "created_at": created_at or getattr(self, "_current_write_session_date", ""),
        })
        if self._recall_embeddings is None:
            self._recall_embeddings = emb.reshape(1, -1)
        else:
            self._recall_embeddings = np.vstack([self._recall_embeddings, emb.reshape(1, -1)])

    # ── BaseMemoryUnit interface ─────────────────────────────

    def write(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Send a conversation turn to the agent.

        The agent autonomously decides whether to update core memory,
        insert into archival storage, etc.
        """
        t0 = time.perf_counter()
        self._write_call_count += 1
        # Track current dia_id so archival inserts can be attributed
        self._current_write_dia_id = (metadata or {}).get("dia_id", "")
        self._current_write_session_date = (metadata or {}).get("session_date", "")

        # Record this turn in the source-turn log so late-bound archival
        # inserts (composed from older context) can back-match to its dia_id.
        src_emb = None
        if self._current_write_dia_id:
            src_emb = self._enc.encode([text])[0]
            self._source_turn_texts.append(text)
            self._source_turn_dia_ids.append(self._current_write_dia_id)
            if self._source_turn_embeddings is None:
                self._source_turn_embeddings = src_emb.reshape(1, -1)
            else:
                self._source_turn_embeddings = np.vstack(
                    [self._source_turn_embeddings, src_emb.reshape(1, -1)]
                )

        # Turn-subsampling for speed: skipped turns bypass the agent tool-call
        # loop (each round ≈ 1 Qwen call) and perform only the minimal archival
        # insert that Letta would have done anyway. N=1 = stock Letta, N=3 ≈ 3x
        # faster, N=5 ≈ 5x. Controlled by C3_AGENT_EVERY_N (default 1).
        agent_every = int(os.environ.get("C3_AGENT_EVERY_N", "1"))
        should_run_agent = (self._write_call_count % agent_every == 0)

        try:
            if should_run_agent:
                result = self._agent_step(text, max_steps=5)
                self._total_messages_sent += 1
                self._total_tool_calls += result["tool_calls"]
                # core_memory_append/replace → mgmt (E10/E11)
                # everything else in write phase → construct (E5/E6)
                # (eviction summary also tracked as mgmt inside _evict)
                self._token_counts["mgmt_llm_input"] += result["mgmt_prompt_tokens"]
                self._token_counts["mgmt_llm_output"] += result["mgmt_completion_tokens"]
                self._token_counts["mgmt_llm_calls"] += result["mgmt_tool_calls"]
                self._token_counts["write_llm_input"] += result["other_prompt_tokens"]
                self._token_counts["write_llm_output"] += result["other_completion_tokens"]
                self._token_counts["write_llm_calls"] += result["other_tool_calls"]
            else:
                # Fast path: just push to recall (conversation history) and
                # auto-insert into archival with the source-turn embedding we
                # already computed above. No Qwen call.
                self._add_to_recall("user", text, dia_id=self._current_write_dia_id)
                if self._current_write_dia_id and src_emb is not None:
                    self._archival_index.add(src_emb.reshape(1, -1))
                    self._archival_entries.append({
                        "text": text,
                        "embedding": src_emb,
                        "dia_ids": [self._current_write_dia_id],
                        "tags": [],
                        "created_at": self._current_write_session_date,
                    })
        except Exception as e:
            print(f"  [Letta] write error (turn {self._write_call_count}): {e}")

        self._write_times.append(time.perf_counter() - t0)

        if self._write_call_count % 50 == 0:
            print(f"    [Letta] {self._write_call_count} turns written, "
                  f"{self._total_tool_calls} tool calls so far", flush=True)

    def read(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Send the query to the agent and return its answer.

        The agent uses its full MemGPT pipeline: core memory (always
        in-context), conversation_search, archival_memory_search, and
        its own LLM reasoning to generate an answer.

        Returns a single MemoryEntry containing the agent's answer.
        The harness treats this as a parametric result (the answer itself).
        """
        t0 = time.perf_counter()
        self._last_retrieved_dia_ids = []

        # MemGPT's design point is that the agent autonomously decides when
        # and how to search its memory (core / recall / archival). Pre-fetching
        # archival via FAISS and injecting it into the user turn turns C3 into
        # a MemGPT+RAG hybrid, which is NOT what the paper evaluates. We let
        # the agent make that call via the `archival_memory_search` tool; the
        # tool implementation already records dia_ids into
        # `_last_retrieved_dia_ids` for downstream Recall@k / P6 scoring.
        answer = ""
        try:
            result = self._agent_step(
                f"Question: {query}\n\nAnswer concisely. Use the memory tools "
                f"(archival_memory_search, conversation_search) to look things "
                f"up as needed, then respond via send_message. If nothing in "
                f"memory is relevant, send an empty answer.",
                max_steps=10,
            )
            self._total_tool_calls += result["tool_calls"]
            answer = result["answer"]
            # core_memory_* → mgmt (E10/E11)
            # everything else in read phase → retrieval (E8/E9)
            self._token_counts["mgmt_llm_input"] += result["mgmt_prompt_tokens"]
            self._token_counts["mgmt_llm_output"] += result["mgmt_completion_tokens"]
            self._token_counts["mgmt_llm_calls"] += result["mgmt_tool_calls"]
            self._token_counts["read_llm_input"] += result["other_prompt_tokens"]
            self._token_counts["read_llm_output"] += result["other_completion_tokens"]
            self._token_counts["read_llm_calls"] += result["other_tool_calls"]
        except Exception as e:
            print(f"  [Letta] read error: {e}")
            answer = "I don't know."

        elapsed = time.perf_counter() - t0
        self._read_times.append(elapsed)
        self._token_counts["read_calls"] += 1
        self._token_counts["read_context_tokens"] += len(self.model.encode(answer))

        return [MemoryEntry(
            content=answer if answer else "I don't know.",
            metadata={
                "source": "letta_agent",
                "dia_ids": list(dict.fromkeys(self._last_retrieved_dia_ids)),
            },
            score=1.0,
        )]

    def reset(self):
        """Clear all state and start fresh."""
        import faiss

        self._core_human = ""
        self._core_persona = "You are a helpful assistant with long-term memory. " \
            "When asked a question, search your memory thoroughly and answer concisely with facts."
        self._archival_index = faiss.IndexFlatIP(self._enc.dim)
        self._archival_entries.clear()
        self._recall_storage.clear()
        self._recall_embeddings = None
        self._source_turn_texts.clear()
        self._source_turn_dia_ids.clear()
        self._source_turn_embeddings = None
        self._messages.clear()
        self._eviction_count = 0
        self._write_times.clear()
        self._read_times.clear()
        self._write_call_count = 0
        self._total_messages_sent = 0
        self._total_tool_calls = 0
        self._token_counts = {
            "write_llm_input": 0,
            "write_llm_output": 0,
            "write_llm_calls": 0,
            "read_context_tokens": 0,
            "read_calls": 0,
            "read_llm_input": 0,
            "read_llm_output": 0,
            "read_llm_calls": 0,
            "mgmt_llm_input": 0,
            "mgmt_llm_output": 0,
            "mgmt_llm_calls": 0,
        }

    def size(self) -> int:
        """Return number of archival entries stored."""
        return len(self._archival_entries)

    # ── cache support ─────────────────────────────────────────
    CACHE_KEY_EXTRA = ["letta_model", "core_memory_char_limit", "max_buffer", "evict_fraction"]

    def save_state(self, dir_path: str) -> None:
        import os, pickle, faiss, numpy as _np
        os.makedirs(dir_path, exist_ok=True)
        # Strip embeddings out of recall_storage / archival_entries for compact pickle
        archival = [{"text": e["text"], "dia_ids": e.get("dia_ids", []),
                     "tags": e.get("tags", []), "created_at": e.get("created_at", "")}
                    for e in self._archival_entries]
        archival_embs = _np.stack([e["embedding"] for e in self._archival_entries]) \
                        if self._archival_entries else _np.zeros((0, self._enc.dim), dtype=_np.float32)
        recall = [{"role": m["role"], "content": m["content"],
                   "dia_id": m.get("dia_id", ""), "created_at": m.get("created_at", "")}
                  for m in self._recall_storage]
        with open(os.path.join(dir_path, "state.pkl"), "wb") as f:
            pickle.dump({
                "core_human": self._core_human,
                "core_persona": self._core_persona,
                "archival": archival,
                "recall": recall,
                "messages": list(self._messages),
                "source_turn_texts": list(self._source_turn_texts),
                "source_turn_dia_ids": list(self._source_turn_dia_ids),
                "token_counts": dict(self._token_counts),
                "write_call_count": getattr(self, "_write_call_count", 0),
                "write_times": list(getattr(self, "_write_times", [])),
            }, f)
        _np.savez(os.path.join(dir_path, "embeddings.npz"),
                  archival=archival_embs,
                  recall=self._recall_embeddings if self._recall_embeddings is not None else _np.zeros((0, self._enc.dim), dtype=_np.float32),
                  source_turn=self._source_turn_embeddings if self._source_turn_embeddings is not None else _np.zeros((0, self._enc.dim), dtype=_np.float32))
        if self._archival_index is not None and self._archival_index.ntotal > 0:
            faiss.write_index(self._archival_index, os.path.join(dir_path, "archival.faiss"))

    @classmethod
    def load_state(cls, dir_path: str, config, model):
        import os, pickle, faiss, numpy as _np
        unit = cls(config, model)
        with open(os.path.join(dir_path, "state.pkl"), "rb") as f:
            d = pickle.load(f)
        embs = _np.load(os.path.join(dir_path, "embeddings.npz"))
        unit._core_human = d["core_human"]
        unit._core_persona = d["core_persona"]
        unit._archival_entries = []
        for i, e in enumerate(d["archival"]):
            entry = dict(e)
            entry["embedding"] = embs["archival"][i] if i < len(embs["archival"]) else None
            unit._archival_entries.append(entry)
        unit._recall_storage = []
        for i, m in enumerate(d["recall"]):
            mm = dict(m)
            if i < len(embs["recall"]):
                mm["embedding"] = embs["recall"][i]
            unit._recall_storage.append(mm)
        if len(embs["recall"]) > 0:
            unit._recall_embeddings = embs["recall"]
        unit._messages = d["messages"]
        unit._source_turn_texts = d["source_turn_texts"]
        unit._source_turn_dia_ids = d["source_turn_dia_ids"]
        if len(embs["source_turn"]) > 0:
            unit._source_turn_embeddings = embs["source_turn"]
        unit._token_counts = d["token_counts"]
        unit._write_call_count = d["write_call_count"]
        unit._write_times = d["write_times"]
        idx_path = os.path.join(dir_path, "archival.faiss")
        if os.path.exists(idx_path):
            unit._archival_index = faiss.read_index(idx_path)
        return unit

    def get_external_storage_bytes(self) -> int:
        """Return total storage: core memory + archival + recall + message buffer."""
        total = 0
        total += len(self._core_persona.encode("utf-8"))
        total += len(self._core_human.encode("utf-8"))
        for entry in self._archival_entries:
            total += len(entry["text"].encode("utf-8"))
            if entry.get("embedding") is not None:
                total += entry["embedding"].nbytes
        for entry in self._recall_storage:
            total += len(entry.get("content", "").encode("utf-8"))
            if entry.get("embedding") is not None:
                total += entry["embedding"].nbytes
        for msg in self._messages:
            total += len(json.dumps(msg).encode("utf-8"))
        return total

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        read_times_ms = [t * 1000 for t in self._read_times]
        write_times_ms = [t * 1000 for t in self._write_times]
        stats.update({
            "letta_model": self._model_name,
            "letta_standalone": True,
            "total_messages_sent": self._total_messages_sent,
            "total_tool_calls": self._total_tool_calls,
            "core_human_length": len(self._core_human),
            "core_persona_length": len(self._core_persona),
            "archival_entries": len(self._archival_entries),
            "recall_entries": len(self._recall_storage),
            "eviction_count": self._eviction_count,
            "buffer_size": len(self._messages),
            "compression_ratio": 1.0,
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

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return all archival entries + core memory blocks."""
        entries = []
        # Core memory blocks
        if self._core_persona.strip():
            entries.append(MemoryEntry(
                content=self._core_persona,
                metadata={"source": "core_memory", "label": "persona"},
            ))
        if self._core_human.strip():
            entries.append(MemoryEntry(
                content=self._core_human,
                metadata={"source": "core_memory", "label": "human"},
            ))
        # Archival entries
        for entry in self._archival_entries:
            meta = {"source": "archival_memory"}
            if entry.get("dia_ids"):
                meta["dia_ids"] = entry["dia_ids"]
            entries.append(MemoryEntry(content=entry["text"], metadata=meta))
        return entries
