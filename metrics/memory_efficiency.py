"""Family E: Memory Efficiency (11 metrics) — E1-E11.  [LOCAL]

E1  Storage Footprint (bytes)           — memory unit's own data structures
E2  Inference Time (ms)                 — total per-query time: read + generate (mean / p50 / p90)
E3  Write Latency (ms)                  — mean / p50 / p90
E4  Retrieval Latency (ms)              — memory unit read() latency only (mean / p50 / p90)
E5  Memory Construction Token Consumption      — LLM tokens consumed during memory construction (write phase)
E6  Memory Construction LLM Calls              — LLM calls during memory construction (write phase)
E7  Memory Retrieved Token Output              — tokens output from memory to support answer generation
E8  Memory Retrieval Token Consumption         — LLM tokens consumed during retrieval process (read-phase LLM calls)
E9  Memory Retrieval LLM Calls                 — LLM calls during retrieval process (read phase)
E10 Memory Management Token Consumption        — LLM tokens consumed for memory management (dedup, update, delete, resolution)
E11 Memory Management LLM Calls               — LLM calls for memory management
"""
import numpy as np
from typing import Any, Dict, List, Union


def _latency_stats(times_ms: List[float], prefix: str) -> Dict[str, float]:
    """Compute mean / p50 / p90 from a list of latency values in ms."""
    if not times_ms:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(times_ms)),
        f"{prefix}_p50": float(np.median(times_ms)),
        f"{prefix}_p90": float(np.percentile(times_ms, 90)),
    }


class MemoryEfficiencyMetrics:
    """Compute all Family E metrics from memory unit stats."""

    @staticmethod
    def compute_e1(memory_unit) -> int:
        """E1: Storage Footprint — memory unit's own data structures in bytes."""
        return memory_unit.get_external_storage_bytes()

    @staticmethod
    def compute_e2(inference_times_ms: List[float]) -> Dict[str, float]:
        """E2: Inference Time — total per-query time (read + generate), mean / p50 / p90 in ms."""
        return _latency_stats(inference_times_ms, "E2_inference_ms")

    @staticmethod
    def compute_e3(write_times_ms: List[float]) -> Dict[str, float]:
        """E3: Write Latency — mean / p50 / p90 in ms."""
        return _latency_stats(write_times_ms, "E3_write_latency_ms")

    @staticmethod
    def compute_e4(read_times_ms: List[float]) -> Dict[str, float]:
        """E4: Retrieval Latency — memory unit read() only, mean / p50 / p90 in ms."""
        return _latency_stats(read_times_ms, "E4_retrieval_latency_ms")

    @staticmethod
    def compute_e5(token_counts: Dict[str, int]) -> int:
        """E5: Construction Token Consumption — LLM tokens consumed during write phase."""
        return (
            token_counts.get("write_llm_input", 0)
            + token_counts.get("write_llm_output", 0)
        )

    @staticmethod
    def compute_e6(token_counts: Dict[str, int]) -> int:
        """E6: Construction LLM Calls — LLM calls during write phase."""
        return token_counts.get("write_llm_calls", 0)

    @staticmethod
    def compute_e7(token_counts: Dict[str, int]) -> int:
        """E7: Retrieved Token Output — non-LLM retrieval context size.

        Counts tokens that flow *from memory into the downstream answer
        LLM's context* without an intermediate LLM call. This covers:
          - FAISS / BM25 / KV-cache lookups (C1, C2, C10-C12, C13, C15, C16)
          - Graph-walk returns (C6)
        Parametric agents (C3 Letta, C4 PageSum) consume retrieved context
        *inside* their own LLM call, which is accounted for in E8. Returning
        E7 > 0 for them would double-count.

        Prefers `read_context_tokens_prompt` (tokens that actually enter the
        downstream prompt after per-entry truncation / top-k trimming) when
        the runner has recorded it; falls back to `read_context_tokens`
        (substrate-level pool total) otherwise.
        """
        if token_counts.get("read_context_tokens_prompt", 0) > 0:
            return token_counts["read_context_tokens_prompt"]
        return token_counts.get("read_context_tokens", 0)

    @staticmethod
    def compute_e8(token_counts: Dict[str, int]) -> int:
        """E8: Retrieval Token Consumption — LLM tokens consumed during retrieval process."""
        return (
            token_counts.get("read_llm_input", 0)
            + token_counts.get("read_llm_output", 0)
        )

    @staticmethod
    def compute_e9(token_counts: Dict[str, int]) -> int:
        """E9: Retrieval LLM Calls — LLM calls during retrieval process (read phase)."""
        return token_counts.get("read_llm_calls", 0)

    @staticmethod
    def compute_e10(token_counts: Dict[str, int]) -> int:
        """E10: Management Token Consumption — LLM tokens for memory management."""
        return (
            token_counts.get("mgmt_llm_input", 0)
            + token_counts.get("mgmt_llm_output", 0)
        )

    @staticmethod
    def compute_e11(token_counts: Dict[str, int]) -> int:
        """E11: Management LLM Calls — LLM calls for memory management."""
        return token_counts.get("mgmt_llm_calls", 0)

    @staticmethod
    def compute(memory_unit, inference_times_ms: List[float] = None) -> Dict[str, Union[int, float]]:
        """Compute all E-family metrics from a memory unit's stats."""
        stats = memory_unit.get_stats()
        token_counts = stats.get("token_consumption", {})

        results = {}
        results["E1_storage_bytes"] = MemoryEfficiencyMetrics.compute_e1(memory_unit)

        read_times = [t * 1000 for t in memory_unit._read_times] if hasattr(memory_unit, '_read_times') else []
        write_times = [t * 1000 for t in memory_unit._write_times] if hasattr(memory_unit, '_write_times') else []

        results.update(MemoryEfficiencyMetrics.compute_e2(inference_times_ms or []))
        results.update(MemoryEfficiencyMetrics.compute_e3(write_times))
        results.update(MemoryEfficiencyMetrics.compute_e4(read_times))

        results["E5_construct_tokens"] = MemoryEfficiencyMetrics.compute_e5(token_counts)
        results["E6_construct_calls"] = MemoryEfficiencyMetrics.compute_e6(token_counts)
        results["E7_retrieved_tokens"] = MemoryEfficiencyMetrics.compute_e7(token_counts)
        results["E8_retrieval_tokens"] = MemoryEfficiencyMetrics.compute_e8(token_counts)
        results["E9_retrieval_calls"] = MemoryEfficiencyMetrics.compute_e9(token_counts)
        results["E10_mgmt_tokens"] = MemoryEfficiencyMetrics.compute_e10(token_counts)
        results["E11_mgmt_calls"] = MemoryEfficiencyMetrics.compute_e11(token_counts)

        # E12: Total LLM Token Consumption (E5 + E7 + E8 + E10)
        results["E12_total_tokens"] = (
            results["E5_construct_tokens"]
            + results["E7_retrieved_tokens"]
            + results["E8_retrieval_tokens"]
            + results["E10_mgmt_tokens"]
        )

        # E13: Total LLM Calls (E6 + E9 + E11)
        results["E13_total_calls"] = (
            results["E6_construct_calls"]
            + results["E9_retrieval_calls"]
            + results["E11_mgmt_calls"]
        )

        # E14: Total Wall Time — write + inference (set by runner from elapsed_sec)
        # Placeholder here; runner overwrites with actual wall clock time
        results["E14_total_time_sec"] = None

        return results
