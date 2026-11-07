"""LongBench QA F1 / EM (SQuAD normalisation), max over reference answers."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence

__all__ = [
    "normalize_answer",
    "f1_score",
    "qa_f1_score",
    "exact_match_score",
    "score_prediction",
    "METRIC_NAMES",
]

METRIC_NAMES = ("qa_f1", "exact_match")

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace (in that order)."""

    def remove_articles(value: str) -> str:
        return _ARTICLES.sub(" ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in _PUNCTUATION)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(prediction_tokens: Sequence[str], ground_truth_tokens: Sequence[str]) -> float:
    """Token-overlap F1 over already-normalised token lists."""
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """LongBench ``qa_f1_score`` for a single reference."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    return f1_score(prediction_tokens, ground_truth_tokens)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """1.0 iff the normalised strings are identical."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def score_prediction(prediction: str, references: Sequence[str]) -> Dict[str, float]:
    """Max over references. An empty prediction scores 0."""
    if not references:
        raise ValueError("at least one reference answer is required")

    prediction = prediction or ""
    f1_values: List[float] = [qa_f1_score(prediction, ref) for ref in references]
    em_values: List[float] = [exact_match_score(prediction, ref) for ref in references]

    best = int(max(range(len(f1_values)), key=lambda i: f1_values[i]))
    return {
        "qa_f1": float(max(f1_values)),
        "exact_match": float(max(em_values)),
        "best_reference_index": best,
        "n_references": len(references),
        "prediction_empty": not prediction.strip(),
    }
