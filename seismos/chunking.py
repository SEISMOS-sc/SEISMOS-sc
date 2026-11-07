"""Sentences + similarity signal -> chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .detectors import MIN_SENTENCES, PAPER_K, laplacian_splits, prominence_splits, zscore_splits
from .segment import SENTENCE_JOIN, split_sentences

Range = Tuple[int, int]  # inclusive sentence indices

METHODS = ("seismos", "zscore", "prominence", "fixed", "whole")

# Documents shorter than this are emitted whole. The paper's runs gated fixed
# the same way as SEISMOS; the z-score and prominence runs had no gate.
DEFAULT_MIN_SENTENCES = {"seismos": MIN_SENTENCES, "fixed": MIN_SENTENCES, "zscore": 0, "prominence": 0}


@dataclass
class DocumentSignal:
    sentences: List[str]
    spans: List[Tuple[int, int]]
    X: Optional[np.ndarray]  # None for fewer than 2 sentences
    token_counts: List[int] = field(default_factory=list)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    first_sentence: int
    last_sentence: int


@dataclass(frozen=True)
class ChunkSpec:
    method: str = "seismos"
    k: float = PAPER_K
    max_tokens: Optional[int] = 480
    alpha: float = 0.05
    epsilon: float = 0.04
    min_sentences: Optional[int] = None
    sign_convention: str = "canonical"

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; expected one of {METHODS}")

    @property
    def label(self) -> str:
        cap = f"@{self.max_tokens}" if self.max_tokens else ""
        if self.method == "seismos":
            suffix = "" if self.sign_convention == "canonical" else f"-{self.sign_convention}"
            return f"seismos{suffix}_k{self.k:g}{cap}"
        if self.method == "zscore":
            return f"zscore_k{self.k:g}{cap}"
        if self.method == "prominence":
            return f"prominence_a{self.alpha:g}{cap}"
        if self.method == "fixed":
            return f"fixed{cap}"
        return "whole"


def compute_signal(text: str, encoder, lang: str = "en") -> DocumentSignal:
    sentences, spans = split_sentences(text, lang)
    X = None
    if len(sentences) >= 2:
        E = encoder.encode_passages(sentences, normalize=True)
        X = np.sum(E[:-1] * E[1:], axis=1).astype(np.float32)
    counts = encoder.token_counts(sentences) if sentences else []
    return DocumentSignal(sentences, spans, X, counts)


def pack_ranges(token_counts: Sequence[int], split_after: Set[int], cap: Optional[int]) -> List[Range]:
    """Break after each boundary and before any sentence that would exceed ``cap``.
    Sentences are never split, so one longer than ``cap`` becomes its own range."""
    n = len(token_counts)
    if n == 0:
        return []
    ranges, start, cur = [], 0, 0
    for i in range(n):
        t = token_counts[i]
        if cap is not None and i > start and cur + t > cap:
            ranges.append((start, i - 1))
            start, cur = i, 0
        cur += t
        if i in split_after and i < n - 1:
            ranges.append((start, i))
            start, cur = i + 1, 0
    ranges.append((start, n - 1))
    return ranges


def boundaries(signal: DocumentSignal, spec: ChunkSpec) -> Set[int]:
    if spec.method == "seismos":
        return laplacian_splits(signal.X, spec.k, spec.sign_convention)
    if spec.method == "zscore":
        return zscore_splits(signal.X, spec.k, spec.epsilon)
    if spec.method == "prominence":
        return prominence_splits(signal.X, spec.alpha)
    return set()


def chunk_ranges(signal: DocumentSignal, spec: ChunkSpec) -> List[Range]:
    n = signal.n_sentences
    if n == 0:
        return []
    min_sents = spec.min_sentences if spec.min_sentences is not None else DEFAULT_MIN_SENTENCES.get(spec.method, 0)
    if spec.method == "whole" or n < min_sents:
        return [(0, n - 1)]
    if spec.max_tokens is not None and not signal.token_counts:
        raise ValueError("a token cap needs DocumentSignal.token_counts")
    counts = signal.token_counts or [0] * n
    return pack_ranges(counts, boundaries(signal, spec), spec.max_tokens)


def make_chunks(signal: DocumentSignal, spec: ChunkSpec, lang: str = "en") -> List[Chunk]:
    join = SENTENCE_JOIN.get(lang, " ")
    return [
        Chunk(
            text=join.join(signal.sentences[i0:i1 + 1]),
            start=signal.spans[i0][0],
            end=signal.spans[i1][1],
            first_sentence=i0,
            last_sentence=i1,
        )
        for i0, i1 in chunk_ranges(signal, spec)
    ]


class SeismosChunker:
    def __init__(self, encoder, k: float = PAPER_K, max_tokens: Optional[int] = 480,
                 lang: str = "en", method: str = "seismos", **spec_kwargs):
        self.encoder = encoder
        self.lang = lang
        self.spec = ChunkSpec(method=method, k=k, max_tokens=max_tokens, **spec_kwargs)

    def signal(self, text: str) -> DocumentSignal:
        return compute_signal(text, self.encoder, self.lang)

    def chunk(self, text: str) -> List[Chunk]:
        return make_chunks(self.signal(text), self.spec, self.lang)

    def chunk_texts(self, text: str) -> List[str]:
        return [c.text for c in self.chunk(text)]


def chunk_corpus(signals: Dict[str, DocumentSignal], spec: ChunkSpec, lang: str = "en"):
    """Returns ``(texts, owner, doc_chunks)``; ``doc_chunks[doc]`` lists ``(start, end, j)``."""
    texts: List[str] = []
    owner: List[str] = []
    doc_chunks: Dict[str, List[Tuple[int, int, int]]] = {}
    for doc_id, sig in signals.items():
        spans = []
        for c in make_chunks(sig, spec, lang):
            spans.append((c.start, c.end, len(texts)))
            texts.append(c.text)
            owner.append(doc_id)
        if spans:
            doc_chunks[doc_id] = spans
    return texts, owner, doc_chunks


def chunk_stats(texts: List[str], owner: List[str], n_docs: int) -> dict:
    lengths = [len(t) for t in texts]
    multi = sum(1 for c in _counts(owner).values() if c > 1)
    return {
        "n_chunks": len(texts),
        "chunks_per_doc": len(texts) / max(1, n_docs),
        "median_chars": float(np.median(lengths)) if lengths else 0.0,
        "pct_docs_split": 100.0 * multi / max(1, n_docs),
    }


def _counts(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
