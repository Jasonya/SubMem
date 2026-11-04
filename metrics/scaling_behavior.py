"""Family S: Scaling Behavior (2 metrics) — S1-S2.

GLOBAL metrics — cannot be computed from a single benchmark run.
Require aggregated results from multiple runs or benchmarks.

S1 Session Retention     — needs ≥2 session counts (e.g. 5, 10, 20, 50)
S2 Capacity Cliff        — needs ≥3 utilization levels
"""
import math
from typing import Dict, List, Optional, Tuple


class ScalingBehaviorMetrics:

    @staticmethod
    def compute_s1(perf_by_session_count: Dict[int, float]) -> float:
        """S1: Session Retention = F1_50sess / F1_5sess.

        Args:
            perf_by_session_count: {session_count: performance}
                e.g. {5: 0.8, 10: 0.75, 20: 0.6, 50: 0.4}
        """
        if 5 not in perf_by_session_count or 50 not in perf_by_session_count:
            counts = sorted(perf_by_session_count.keys())
            if len(counts) < 2:
                return 1.0
            return perf_by_session_count[counts[-1]] / perf_by_session_count[counts[0]]
        baseline = perf_by_session_count[5]
        if baseline == 0:
            return 0.0
        return perf_by_session_count[50] / baseline

    @staticmethod
    def compute_s2(utilization_perf_curve: List[Tuple[float, float]],
                   threshold: float = 0.1) -> float:
        """S2: Capacity Cliff — utilization ratio where dPerf/dUtil drops sharply.

        Args:
            utilization_perf_curve: [(utilization_fraction, performance), ...]
                sorted by utilization ascending.
            threshold: slope magnitude threshold for "cliff".

        Returns:
            Utilization fraction at cliff point, or float('inf') for unbounded.
        """
        if len(utilization_perf_curve) < 3:
            return float("inf")

        curve = sorted(utilization_perf_curve, key=lambda x: x[0])
        max_drop = 0.0
        cliff_point = float("inf")

        for i in range(1, len(curve)):
            du = curve[i][0] - curve[i - 1][0]
            dp = curve[i][1] - curve[i - 1][1]
            if du > 0:
                slope = dp / du
                if slope < -threshold and abs(slope) > max_drop:
                    max_drop = abs(slope)
                    cliff_point = curve[i][0]

        return cliff_point

    @staticmethod
    def compute(perf_by_session_count=None,
                utilization_perf_curve=None) -> Dict[str, float]:
        results = {}
        if perf_by_session_count:
            results["S1_session_retention"] = ScalingBehaviorMetrics.compute_s1(perf_by_session_count)
        if utilization_perf_curve:
            results["S2_capacity_cliff"] = ScalingBehaviorMetrics.compute_s2(utilization_perf_curve)
        return results
