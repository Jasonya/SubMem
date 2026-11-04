"""Unit tests for mem0_eval — adapted mem0 evaluation code.

Tests the add, search, prompts, and metrics modules with mocked
mem0 Memory and OpenAI calls (no real API calls needed).
"""
import json
import os
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch, call

import pytest

# Ensure SubMem is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ============================================================
# Fixtures
# ============================================================

SAMPLE_CONVERSATION = {
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session1": [
            {"speaker": "Alice", "text": "I just got back from Tokyo."},
            {"speaker": "Bob", "text": "That's amazing! How was it?"},
            {"speaker": "Alice", "text": "It was incredible, I visited Shibuya."},
            {"speaker": "Bob", "text": "I've always wanted to go there."},
        ],
        "session1_date_time": "2023-05-01",
        "session2": [
            {"speaker": "Alice", "text": "I'm thinking of learning Japanese."},
            {"speaker": "Bob", "text": "That would be so useful for your trips."},
        ],
        "session2_date_time": "2023-06-15",
    },
    "qa": [
        {
            "question": "Where did Alice travel recently?",
            "answer": "Tokyo",
            "category": 1,
            "evidence": ["D1:1"],
            "adversarial_answer": "",
        },
        {
            "question": "What does Alice want to learn?",
            "answer": "Japanese",
            "category": 1,
            "evidence": ["D2:1"],
            "adversarial_answer": "",
        },
    ],
    "sample_id": "test_001",
}


def make_mock_mem0():
    """Create a mock Memory instance that tracks add/search/delete calls."""
    mock = MagicMock()
    # Store memories in a simple dict
    _store: Dict[str, List[dict]] = defaultdict(list)

    def mock_add(messages, user_id=None, metadata=None):
        for msg in messages if isinstance(messages, list) else [messages]:
            content = msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)
            _store[user_id].append({
                "memory": content,
                "metadata": metadata or {},
            })
        return {"results": []}

    def mock_search(query=None, user_id=None, limit=10):
        entries = _store.get(user_id, [])
        results = []
        for e in entries[:limit]:
            results.append({
                "memory": e["memory"],
                "metadata": e["metadata"],
                "score": 0.85,
            })
        return {"results": results}

    def mock_delete_all(user_id=None):
        _store.pop(user_id, None)

    def mock_get_all(user_id=None):
        return {"results": _store.get(user_id, [])}

    mock.add = MagicMock(side_effect=mock_add)
    mock.search = MagicMock(side_effect=mock_search)
    mock.delete_all = MagicMock(side_effect=mock_delete_all)
    mock.get_all = MagicMock(side_effect=mock_get_all)
    mock._store = _store
    return mock


def make_mock_openai_client():
    """Create a mock OpenAI client that returns a fixed answer."""
    mock = MagicMock()

    def mock_create(model=None, messages=None, temperature=0.0, response_format=None):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        if response_format and response_format.get("type") == "json_object":
            resp.choices[0].message.content = '{"label": "CORRECT"}'
        else:
            resp.choices[0].message.content = "Tokyo"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 5
        return resp

    mock.chat.completions.create = MagicMock(side_effect=mock_create)
    return mock


# ============================================================
# Tests: Prompts
# ============================================================

class TestPrompts:
    def test_answer_prompt_has_required_placeholders(self):
        from SubMem.mem0_eval.prompts import ANSWER_PROMPT
        assert "{{speaker_1_user_id}}" in ANSWER_PROMPT
        assert "{{speaker_2_user_id}}" in ANSWER_PROMPT
        assert "{{speaker_1_memories}}" in ANSWER_PROMPT
        assert "{{speaker_2_memories}}" in ANSWER_PROMPT
        assert "{{question}}" in ANSWER_PROMPT

    def test_answer_prompt_renders_with_jinja2(self):
        from jinja2 import Template
        from SubMem.mem0_eval.prompts import ANSWER_PROMPT
        template = Template(ANSWER_PROMPT)
        rendered = template.render(
            speaker_1_user_id="Alice",
            speaker_2_user_id="Bob",
            speaker_1_memories='["mem1"]',
            speaker_2_memories='["mem2"]',
            question="What happened?",
        )
        assert "Alice" in rendered
        assert "Bob" in rendered
        assert "mem1" in rendered
        assert "What happened?" in rendered

    def test_answer_prompt_matches_original(self):
        """Verify our prompt matches mem0-repo/evaluation/prompts.py exactly."""
        from SubMem.mem0_eval.prompts import ANSWER_PROMPT
        # Key phrases that must be present (from original)
        assert "less than 5-6 words" in ANSWER_PROMPT
        assert "prioritize the most recent memory" in ANSWER_PROMPT
        assert "went to India last year" in ANSWER_PROMPT

    def test_answer_prompt_zep_exists(self):
        from SubMem.mem0_eval.prompts import ANSWER_PROMPT_ZEP
        assert "{{memories}}" in ANSWER_PROMPT_ZEP

    def test_answer_prompt_graph_exists(self):
        from SubMem.mem0_eval.prompts import ANSWER_PROMPT_GRAPH
        assert "{{speaker_1_graph_memories}}" in ANSWER_PROMPT_GRAPH


# ============================================================
# Tests: Custom Instructions
# ============================================================

class TestCustomInstructions:
    def test_custom_instructions_complete(self):
        """Verify custom_instructions matches original (not truncated)."""
        from SubMem.mem0_eval.src.memzero.add import custom_instructions
        # These were truncated in the old C5 — must be present now
        assert 'charity race for mental health' in custom_instructions
        assert 'captures the person\'s experience, challenges, and aspirations' in custom_instructions

    def test_custom_instructions_has_five_guidelines(self):
        from SubMem.mem0_eval.src.memzero.add import custom_instructions
        for i in range(1, 6):
            assert f"{i}." in custom_instructions


# ============================================================
# Tests: MemoryADD
# ============================================================

class TestMemoryADD:
    def test_init(self):
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock, batch_size=2)
        assert adder.batch_size == 2
        assert adder.mem0_client is mock

    def test_process_conversation_creates_correct_speaker_ids(self):
        """Verify speaker IDs follow '{name}_{idx}' format (checked via delete_all calls)."""
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock, batch_size=2)
        adder.process_conversation(SAMPLE_CONVERSATION, idx=3)
        # Original process_conversation returns None; verify IDs via delete_all calls
        delete_calls = mock.delete_all.call_args_list
        user_ids = [c.kwargs.get("user_id") for c in delete_calls]
        assert "Alice_3" in user_ids
        assert "Bob_3" in user_ids

    def test_process_conversation_deletes_existing(self):
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock, batch_size=2)
        adder.process_conversation(SAMPLE_CONVERSATION, idx=0)
        # delete_all called for both speakers
        delete_calls = mock.delete_all.call_args_list
        user_ids = [c.kwargs.get("user_id") for c in delete_calls]
        assert "Alice_0" in user_ids
        assert "Bob_0" in user_ids

    def test_process_conversation_adds_memories(self):
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock, batch_size=2)
        adder.process_conversation(SAMPLE_CONVERSATION, idx=0)
        # mem0_client.add() should have been called
        assert mock.add.call_count > 0

    def test_message_construction_speaker_a_perspective(self):
        """Speaker A's messages: A=user, B=assistant (same as original)."""
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock, batch_size=100)  # large batch to get all in one call
        adder.process_conversation(SAMPLE_CONVERSATION, idx=0)

        # Find calls for Alice_0
        alice_calls = [c for c in mock.add.call_args_list if c.kwargs.get("user_id") == "Alice_0"]
        assert len(alice_calls) > 0
        first_batch = alice_calls[0].args[0]  # first positional arg = messages
        # First message should be Alice as "user"
        assert first_batch[0]["role"] == "user"
        assert "Alice:" in first_batch[0]["content"]
        # Second message should be Bob as "assistant"
        assert first_batch[1]["role"] == "assistant"
        assert "Bob:" in first_batch[1]["content"]

    def test_message_construction_speaker_b_perspective(self):
        """Speaker B's messages: B=user, A=assistant (reverse, same as original)."""
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock, batch_size=100)
        adder.process_conversation(SAMPLE_CONVERSATION, idx=0)

        bob_calls = [c for c in mock.add.call_args_list if c.kwargs.get("user_id") == "Bob_0"]
        assert len(bob_calls) > 0
        first_batch = bob_calls[0].args[0]
        # First message (from Alice) should be "assistant" in Bob's perspective
        assert first_batch[0]["role"] == "assistant"
        assert "Alice:" in first_batch[0]["content"]
        # Second message (from Bob) should be "user" in Bob's perspective
        assert first_batch[1]["role"] == "user"
        assert "Bob:" in first_batch[1]["content"]

    def test_batch_size_respected(self):
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock, batch_size=2)
        adder.process_conversation(SAMPLE_CONVERSATION, idx=0)
        # Session1 has 4 messages → 2 batches per speaker
        # Session2 has 2 messages → 1 batch per speaker
        # Total: (2+1) * 2 speakers = 6 add calls
        # But threading means both speakers run concurrently per session
        assert mock.add.call_count == 6

    def test_metadata_timestamp_passed(self):
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock, batch_size=100)
        adder.process_conversation(SAMPLE_CONVERSATION, idx=0)
        # Check that metadata contains timestamp
        for c in mock.add.call_args_list:
            meta = c.kwargs.get("metadata", {})
            assert "timestamp" in meta

    def test_threading_used(self):
        """Verify that threading is used for parallel speaker add."""
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()

        thread_names = []
        original_start = threading.Thread.start

        def track_start(self):
            thread_names.append(self.name)
            original_start(self)

        with patch.object(threading.Thread, 'start', track_start):
            adder = MemoryADD(mem0=mock, batch_size=2)
            adder.process_conversation(SAMPLE_CONVERSATION, idx=0)

        # Should have created threads (2 per session × 2 sessions = 4)
        assert len(thread_names) >= 4

    def test_retry_on_failure(self):
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        # Make add fail twice then succeed
        call_count = {"n": 0}
        original_add = mock.add.side_effect

        def flaky_add(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("API error")
            return original_add(*args, **kwargs)

        mock.add.side_effect = flaky_add
        adder = MemoryADD(mem0=mock, batch_size=100)
        # add_memory has retry=3, so the 3rd attempt should succeed
        adder.add_memory("user1", [{"role": "user", "content": "test"}],
                         metadata={"timestamp": "2023-01-01"})
        assert call_count["n"] == 3

    def test_unknown_speaker_raises(self):
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        mock = make_mock_mem0()
        adder = MemoryADD(mem0=mock)
        bad_item = {
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session1": [{"speaker": "Charlie", "text": "Hi"}],
                "session1_date_time": "2023-01-01",
            },
            "qa": [],
        }
        with pytest.raises(ValueError, match="Unknown speaker"):
            adder.process_conversation(bad_item, idx=0)


# ============================================================
# Tests: MemorySearch
# ============================================================

class TestMemorySearch:
    def setup_method(self):
        self.mock_mem0 = make_mock_mem0()
        # Pre-populate some memories
        self.mock_mem0._store["Alice_0"] = [
            {"memory": "Alice visited Tokyo in May 2023", "metadata": {"timestamp": "2023-05-01"}},
            {"memory": "Alice visited Shibuya crossing", "metadata": {"timestamp": "2023-05-01"}},
        ]
        self.mock_mem0._store["Bob_0"] = [
            {"memory": "Bob wants to visit Tokyo someday", "metadata": {"timestamp": "2023-05-01"}},
        ]

    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_search_memory_returns_formatted_results(self, mock_openai_cls):
        from SubMem.mem0_eval.src.memzero.search import MemorySearch
        searcher = MemorySearch(self.mock_mem0, top_k=10)
        memories, graph_memories, search_time = searcher.search_memory("Alice_0", "Where did Alice go?")
        assert len(memories) == 2
        assert memories[0]["memory"] == "Alice visited Tokyo in May 2023"
        assert memories[0]["timestamp"] == "2023-05-01"
        assert memories[0]["score"] == 0.85
        assert graph_memories is None
        assert search_time > 0

    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_answer_question_calls_openai(self, mock_openai_cls):
        from SubMem.mem0_eval.src.memzero.search import MemorySearch
        mock_client = make_mock_openai_client()
        searcher = MemorySearch(self.mock_mem0, top_k=10)
        searcher.openai_client = mock_client

        (answer, mem1, mem2, t1, t2,
         graph1, graph2, resp_time, token_usage) = searcher.answer_question(
            "Alice_0", "Bob_0", "Where did Alice travel?", "", 1,
        )
        assert answer == "Tokyo"
        assert len(mem1) == 2  # Alice's memories
        assert len(mem2) == 1  # Bob's memories
        assert resp_time > 0
        assert "prompt_tokens" in token_usage

        # Verify OpenAI was called with system message containing the prompt
        create_call = mock_client.chat.completions.create.call_args
        messages = create_call.kwargs.get("messages") or create_call[1].get("messages")
        assert messages[0]["role"] == "system"
        assert "Where did Alice travel?" in messages[0]["content"]

    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_answer_prompt_contains_both_speakers(self, mock_openai_cls):
        from SubMem.mem0_eval.src.memzero.search import MemorySearch
        mock_client = make_mock_openai_client()
        searcher = MemorySearch(self.mock_mem0, top_k=10)
        searcher.openai_client = mock_client

        searcher.answer_question("Alice_0", "Bob_0", "test question", "", 1)

        create_call = mock_client.chat.completions.create.call_args
        messages = create_call.kwargs.get("messages") or create_call[1].get("messages")
        prompt = messages[0]["content"]
        assert "Alice" in prompt
        assert "Bob" in prompt
        assert "Tokyo" in prompt  # Alice's memory content

    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_speaker_id_split_in_prompt(self, mock_openai_cls):
        """Speaker ID 'Alice_0' should render as 'Alice' in prompt (split on _)."""
        from SubMem.mem0_eval.src.memzero.search import MemorySearch
        mock_client = make_mock_openai_client()
        searcher = MemorySearch(self.mock_mem0, top_k=10)
        searcher.openai_client = mock_client

        searcher.answer_question("Alice_0", "Bob_0", "test", "", 1)

        create_call = mock_client.chat.completions.create.call_args
        messages = create_call.kwargs.get("messages") or create_call[1].get("messages")
        prompt = messages[0]["content"]
        # Should contain "Alice" not "Alice_0"
        assert "Memories for user Alice:" in prompt or "Memories for user Alice\n" in prompt

    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_process_question_returns_expected_fields(self, mock_openai_cls):
        from SubMem.mem0_eval.src.memzero.search import MemorySearch
        mock_client = make_mock_openai_client()
        searcher = MemorySearch(self.mock_mem0, top_k=10)
        searcher.openai_client = mock_client

        qa_item = SAMPLE_CONVERSATION["qa"][0]
        result = searcher.process_question(qa_item, "Alice_0", "Bob_0")

        expected_keys = [
            "question", "answer", "category", "evidence", "response",
            "adversarial_answer", "speaker_1_memories", "speaker_2_memories",
            "num_speaker_1_memories", "num_speaker_2_memories",
            "speaker_1_memory_time", "speaker_2_memory_time", "response_time",
            "token_usage",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        assert result["question"] == "Where did Alice travel recently?"
        assert result["answer"] == "Tokyo"
        assert result["category"] == 1

    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_search_retry_on_failure(self, mock_openai_cls):
        from SubMem.mem0_eval.src.memzero.search import MemorySearch
        mock_mem0 = MagicMock()
        call_count = {"n": 0}

        def flaky_search(query=None, user_id=None, limit=10):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("Search error")
            return {"results": [{"memory": "test", "metadata": {"timestamp": ""}, "score": 0.5}]}

        mock_mem0.search = MagicMock(side_effect=flaky_search)
        searcher = MemorySearch(mock_mem0, top_k=5)
        memories, _, _ = searcher.search_memory("user1", "query")
        assert call_count["n"] == 3
        assert len(memories) == 1

    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_memory_format_timestamp_colon_memory(self, mock_openai_cls):
        """Formatted memory should be 'timestamp: memory_text' (same as original)."""
        from SubMem.mem0_eval.src.memzero.search import MemorySearch
        mock_client = make_mock_openai_client()
        searcher = MemorySearch(self.mock_mem0, top_k=10)
        searcher.openai_client = mock_client

        searcher.answer_question("Alice_0", "Bob_0", "test", "", 1)

        create_call = mock_client.chat.completions.create.call_args
        messages = create_call.kwargs.get("messages") or create_call[1].get("messages")
        prompt = messages[0]["content"]
        # Should contain "2023-05-01: Alice visited Tokyo"
        assert "2023-05-01: Alice visited Tokyo" in prompt


# ============================================================
# Tests: Metrics
# ============================================================

class TestMetricsUtils:
    def test_calculate_metrics_exact_match(self):
        from SubMem.mem0_eval.metrics.utils import calculate_metrics
        result = calculate_metrics("Tokyo", "Tokyo")
        assert result["exact_match"] == 1
        assert result["f1"] == 1.0

    def test_calculate_metrics_no_match(self):
        from SubMem.mem0_eval.metrics.utils import calculate_metrics
        result = calculate_metrics("Paris", "Tokyo")
        assert result["exact_match"] == 0

    def test_calculate_metrics_partial_match(self):
        from SubMem.mem0_eval.metrics.utils import calculate_metrics
        result = calculate_metrics("I think it was Tokyo", "Tokyo")
        assert result["f1"] > 0
        assert result["f1"] < 1.0

    def test_calculate_metrics_empty_input(self):
        from SubMem.mem0_eval.metrics.utils import calculate_metrics
        result = calculate_metrics("", "Tokyo")
        assert result["f1"] == 0.0
        assert result["exact_match"] == 0

    def test_calculate_bleu_scores(self):
        from SubMem.mem0_eval.metrics.utils import calculate_bleu_scores
        scores = calculate_bleu_scores("Tokyo is great", "Tokyo is great")
        assert scores["bleu1"] > 0.9

    def test_f1_token_overlap(self):
        from SubMem.mem0_eval.metrics.utils import calculate_metrics
        # "shell necklace" vs "A shell necklace from Hawaii"
        result = calculate_metrics("A shell necklace from Hawaii", "shell necklace")
        # Should have some F1 overlap
        assert result["f1"] > 0.3

    def test_aggregate_metrics(self):
        from SubMem.mem0_eval.metrics.utils import aggregate_metrics
        metrics = [
            {"f1": 0.8, "exact_match": 1},
            {"f1": 0.4, "exact_match": 0},
            {"f1": 0.6, "exact_match": 0},
        ]
        categories = [1, 1, 2]
        agg = aggregate_metrics(metrics, categories)
        assert "overall" in agg
        assert "category_1" in agg
        assert "category_2" in agg
        assert agg["overall"]["f1"]["mean"] == pytest.approx(0.6, abs=0.01)


# ============================================================
# Tests: LLM Judge
# ============================================================

class TestLLMJudge:
    @patch("SubMem.mem0_eval.metrics.llm_judge._get_client")
    def test_evaluate_correct(self, mock_get_client):
        from SubMem.mem0_eval.metrics.llm_judge import evaluate_llm_judge
        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"label": "CORRECT"}'
        mock_client.chat.completions.create = MagicMock(return_value=resp)
        mock_get_client.return_value = mock_client
        score = evaluate_llm_judge("Where?", "Tokyo", "Tokyo")
        assert score == 1

    @patch("SubMem.mem0_eval.metrics.llm_judge._get_client")
    def test_evaluate_wrong(self, mock_get_client):
        from SubMem.mem0_eval.metrics.llm_judge import evaluate_llm_judge
        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"label": "WRONG"}'
        mock_client.chat.completions.create = MagicMock(return_value=resp)
        mock_get_client.return_value = mock_client
        score = evaluate_llm_judge("Where?", "Tokyo", "Paris")
        assert score == 0

    def test_accuracy_prompt_format(self):
        from SubMem.mem0_eval.metrics.llm_judge import ACCURACY_PROMPT
        assert "{question}" in ACCURACY_PROMPT
        assert "{gold_answer}" in ACCURACY_PROMPT
        assert "{generated_answer}" in ACCURACY_PROMPT
        assert "CORRECT" in ACCURACY_PROMPT
        assert "WRONG" in ACCURACY_PROMPT


# ============================================================
# Tests: End-to-end pipeline (add → search → answer)
# ============================================================

class TestEndToEnd:
    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_add_then_search(self, mock_openai_cls):
        """Full pipeline: add memories → search → answer question."""
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        from SubMem.mem0_eval.src.memzero.search import MemorySearch

        mock_mem0 = make_mock_mem0()
        mock_client = make_mock_openai_client()

        # Add (process_conversation returns None like original)
        adder = MemoryADD(mem0=mock_mem0, batch_size=2)
        adder.process_conversation(SAMPLE_CONVERSATION, idx=0)
        a_id, b_id = "Alice_0", "Bob_0"

        # Verify memories were stored
        assert len(mock_mem0._store.get(a_id, [])) > 0
        assert len(mock_mem0._store.get(b_id, [])) > 0

        # Search & Answer
        searcher = MemorySearch(mock_mem0, top_k=10)
        searcher.openai_client = mock_client

        result = searcher.process_question(
            SAMPLE_CONVERSATION["qa"][0], a_id, b_id,
        )
        assert result["response"] is not None
        assert result["num_speaker_1_memories"] > 0
        assert result["num_speaker_2_memories"] > 0

    @patch("SubMem.mem0_eval.src.memzero.search.OpenAI")
    def test_result_format_compatible_with_evals(self, mock_openai_cls):
        """Result dict should have all fields expected by evals.py."""
        from SubMem.mem0_eval.src.memzero.add import MemoryADD
        from SubMem.mem0_eval.src.memzero.search import MemorySearch
        from SubMem.mem0_eval.metrics.utils import calculate_metrics

        mock_mem0 = make_mock_mem0()
        mock_client = make_mock_openai_client()

        adder = MemoryADD(mem0=mock_mem0, batch_size=2)
        adder.process_conversation(SAMPLE_CONVERSATION, idx=0)
        a_id, b_id = "Alice_0", "Bob_0"

        searcher = MemorySearch(mock_mem0, top_k=10)
        searcher.openai_client = mock_client

        result = searcher.process_question(
            SAMPLE_CONVERSATION["qa"][0], a_id, b_id,
        )

        # evals.py expects these fields
        assert "question" in result
        assert "answer" in result
        assert "response" in result
        assert "category" in result

        # Should be compatible with calculate_metrics
        metrics = calculate_metrics(str(result["response"]), str(result["answer"]))
        assert "f1" in metrics
        assert "exact_match" in metrics
        assert "bleu1" in metrics
