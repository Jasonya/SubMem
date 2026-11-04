from .task_performance import TaskPerformanceMetrics
from .memory_efficiency import MemoryEfficiencyMetrics
from .scaling_behavior import ScalingBehaviorMetrics
from .robustness import RobustnessMetrics
from .memory_quality import MemoryQualityMetrics
from .behavioral_dynamics import BehavioralDynamicsMetrics
try:
    from .nli_scorer import NLIScorer, compute_faithfulness_nli_score
except ImportError:
    NLIScorer = None
    compute_faithfulness_nli_score = None
from .retrieval_quality import RetrievalQualityMetrics
from .llm_judge import LLMJudge, compute_llm_judge_score

# ── Local Metrics: computable from a single benchmark run ──
LOCAL_METRIC_FAMILIES = {
    "task_performance": TaskPerformanceMetrics,    # P1-P7
    "robustness": RobustnessMetrics,               # R1-R2
    "memory_efficiency": MemoryEfficiencyMetrics,  # E1-E11
    "memory_quality": MemoryQualityMetrics,        # Q1-Q2
    "behavioral_dynamics": BehavioralDynamicsMetrics,  # D1-D3
}

# ── Global Metrics: require multiple runs / benchmarks ──
GLOBAL_METRIC_FAMILIES = {
    "scaling_behavior": ScalingBehaviorMetrics,    # S1-S3
}

ALL_METRIC_FAMILIES = {**LOCAL_METRIC_FAMILIES, **GLOBAL_METRIC_FAMILIES}
