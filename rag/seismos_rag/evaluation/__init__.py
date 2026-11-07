"""Evaluation metrics: answer quality, faithfulness, evidence proxies, statistics."""

from .evidence_metrics import (
    AVAILABLE_PROXY_METRICS,
    UNAVAILABLE_METRICS,
    availability_report,
    score_evidence,
)
from .faithfulness import FaithfulnessScorer, build_faithfulness_scorer
from .qa_metrics import (
    METRIC_NAMES,
    exact_match_score,
    normalize_answer,
    qa_f1_score,
    score_prediction,
)
from .statistics import compare_methods, run_all_comparisons, summarize

__all__ = [
    "AVAILABLE_PROXY_METRICS",
    "UNAVAILABLE_METRICS",
    "FaithfulnessScorer",
    "METRIC_NAMES",
    "availability_report",
    "build_faithfulness_scorer",
    "compare_methods",
    "exact_match_score",
    "normalize_answer",
    "qa_f1_score",
    "run_all_comparisons",
    "score_evidence",
    "score_prediction",
    "summarize",
]
