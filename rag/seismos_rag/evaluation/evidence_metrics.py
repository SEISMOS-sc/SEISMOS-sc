"""Label-free evidence proxies.

NarrativeQA has no gold evidence spans, so evidence precision/recall are not
computed; they are listed in :data:`UNAVAILABLE_METRICS`. The proxies below use
only the reference answer:

``answer_token_recall``      share of answer tokens found in the retrieved context
``answer_present``           whole answer occurs contiguously in the context
``answer_recall_at_k``       ``answer_present`` over the top-1..top-k prefixes
``context_precision_proxy``  share of chunks containing a non-stopword answer token
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .qa_metrics import normalize_answer

# Written to metric_availability.json.
UNAVAILABLE_METRICS: Dict[str, str] = {
    "evidence_completeness": (
        "requires gold supporting-evidence spans; LongBench NarrativeQA "
        "distributes only (input, context, answers)"
    ),
    "evidence_precision": "requires gold supporting-evidence spans",
    "evidence_f1": "requires gold supporting-evidence spans",
    "chunk_level_ndcg": (
        "requires per-chunk relevance judgements; chunk boundaries here are "
        "method-dependent, so no fixed qrel set exists"
    ),
    "chunk_level_recall": "requires per-chunk relevance judgements",
}

AVAILABLE_PROXY_METRICS = (
    "answer_token_recall",
    "answer_present",
    "context_precision_proxy",
    "n_answer_tokens",
)

_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for
    with by from as is are was were be been being it its he she they them his her
    their who whom which what when where why how not no do does did have has had
    i you we us our your me my""".split()
)


@dataclass
class EvidenceScores:
    answer_token_recall: float
    answer_present: float
    context_precision_proxy: float
    n_answer_tokens: int
    answer_recall_at_k: Dict[int, float]
    best_reference_index: int

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "answer_token_recall": round(self.answer_token_recall, 6),
            "answer_present": self.answer_present,
            "context_precision_proxy": round(self.context_precision_proxy, 6),
            "n_answer_tokens": self.n_answer_tokens,
            "best_reference_index": self.best_reference_index,
        }
        for k, value in sorted(self.answer_recall_at_k.items()):
            row[f"answer_present_at_{k}"] = value
        # Unavailable metrics are emitted as explicit nulls so the columns exist
        # in the CSV and their emptiness is visible rather than inferred.
        for name in UNAVAILABLE_METRICS:
            row[name] = None
        return row


def _tokens(text: str) -> List[str]:
    return normalize_answer(text).split()


def answer_token_recall(reference: str, context: str) -> float:
    """Fraction of reference answer tokens present in the context (multiset-aware)."""
    from collections import Counter

    reference_tokens = _tokens(reference)
    if not reference_tokens:
        return 0.0
    context_counts = Counter(_tokens(context))
    reference_counts = Counter(reference_tokens)
    covered = sum(min(count, context_counts[token]) for token, count in reference_counts.items())
    return covered / len(reference_tokens)


def answer_present(reference: str, context: str) -> float:
    """1.0 iff the normalised reference occurs contiguously in the normalised context."""
    needle = normalize_answer(reference)
    if not needle:
        return 0.0
    return float(needle in normalize_answer(context))


def context_precision_proxy(reference: str, chunk_texts: Sequence[str]) -> float:
    """Share of retrieved chunks containing at least one content-word answer token."""
    if not chunk_texts:
        return 0.0
    content_tokens = {t for t in _tokens(reference) if t not in _STOPWORDS}
    if not content_tokens:
        return 0.0
    informative = sum(
        1 for text in chunk_texts if content_tokens & set(_tokens(text))
    )
    return informative / len(chunk_texts)


def score_evidence(
    references: Sequence[str],
    chunk_texts: Sequence[str],
    separator: str = "\n\n",
) -> EvidenceScores:
    """Proxies against the reference with the highest token recall."""
    if not references:
        raise ValueError("at least one reference answer is required")

    context = separator.join(chunk_texts)

    recalls = [answer_token_recall(ref, context) for ref in references]
    best = int(max(range(len(recalls)), key=lambda i: recalls[i]))
    reference = references[best]

    recall_at_k: Dict[int, float] = {}
    for k in range(1, len(chunk_texts) + 1):
        recall_at_k[k] = answer_present(reference, separator.join(chunk_texts[:k]))

    return EvidenceScores(
        answer_token_recall=recalls[best],
        answer_present=max(answer_present(ref, context) for ref in references),
        context_precision_proxy=context_precision_proxy(reference, chunk_texts),
        n_answer_tokens=len(_tokens(reference)),
        answer_recall_at_k=recall_at_k,
        best_reference_index=best,
    )


def availability_report(
    faithfulness_available: bool,
    faithfulness_reason: Optional[str] = None,
    evidence_enabled: bool = True,
) -> Dict[str, Any]:
    """The contents of ``metric_availability.json``."""
    return {
        "dataset": "LongBench NarrativeQA",
        "note": (
            "Metrics are split into those computable from the distributed data "
            "and those that would require annotations this dataset does not "
            "ship. No value is imputed for the latter."
        ),
        "available": {
            "qa_f1": "LongBench official token-F1, max over reference answers",
            "exact_match": "normalised string equality, max over reference answers",
            "faithfulness": (
                "Vectara HHEM over (retrieved context, generated answer)"
                if faithfulness_available
                else "UNAVAILABLE at run time"
            ),
            "context_tokens": "prompt length measured with the generator's tokenizer",
            "avg_chunk_tokens": "encoder-tokenizer token counts per chunk",
            **(
                {name: "label-free proxy computed from the reference answer only"
                 for name in AVAILABLE_PROXY_METRICS}
                if evidence_enabled
                else {}
            ),
        },
        "unavailable": dict(UNAVAILABLE_METRICS),
        "faithfulness_runtime": {
            "available": faithfulness_available,
            "reason": faithfulness_reason,
        },
        "evidence_proxies_enabled": evidence_enabled,
        "caveat": (
            "answer_token_recall and answer_present measure whether the retrieved "
            "context could support the reference answer. They are not evidence "
            "completeness: NarrativeQA answers are frequently paraphrases of the "
            "narrative rather than extractive spans, so absolute values are low "
            "for every method. Compare across methods, not against 1.0."
        ),
    }
