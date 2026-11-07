"""Shared fixtures, including a fake encoder. Tests marked ``slow`` / ``network``
need real weights or the HF hub and are deselected by default.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # seismos_rag
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # core seismos package

from seismos_rag.chunkers.base import ChunkingContext  # noqa: E402
from seismos_rag.utils.logging import setup_logging  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real model weights")
    config.addinivalue_line("markers", "network: needs network access")
    setup_logging(None)


class FakeTokenizer:
    """Whitespace tokenizer with the HF call surface the pipeline uses."""

    chat_template = None

    def __call__(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, str):
            return {"input_ids": text.split()}
        return {"input_ids": [t.split() for t in text]}

    def decode(self, ids, **kwargs):
        return " ".join(ids)

    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeEncoder:
    """Deterministic hash-based encoder: same text -> same unit vector."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.tokenizer = FakeTokenizer()
        self.batch_size = 8

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        return [len(t.split()) for t in texts]

    def count_tokens_single(self, text: str) -> int:
        return len(text.split())

    def _vector(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.normal(size=self.dim).astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-12)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    encode_passages = _encode
    encode_queries = _encode
    encode_sentences = _encode
    encode_chunks_cached = _encode
    encode_queries_cached = _encode

    def encode_sentences_cached(self, doc_id, sentences):
        return self._encode(sentences)

    @property
    def dimension(self) -> int:
        return self.dim


class FakeSplitter:
    """Sentence splitter that cuts on '. ' and reports exact spans."""

    def split(self, text: str):
        from seismos_rag.chunkers.sentence_splitter import SentenceSplit

        spans = []
        position = 0
        while position < len(text):
            dot = text.find(". ", position)
            end = len(text) if dot == -1 else dot + 1
            start = position
            while start < end and text[start].isspace():
                start += 1
            if end > start:
                spans.append((start, end))
            position = end + 1
        return SentenceSplit([text[a:b] for a, b in spans], spans)

    def cache_stats(self):
        return {"hits": 0, "misses": 0}


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def fake_splitter() -> FakeSplitter:
    return FakeSplitter()


@pytest.fixture
def document() -> str:
    """A 12-sentence document with a clear topic shift in the middle."""
    first = " ".join(
        f"The mariner charted the northern reef in season {i} and logged the tides."
        for i in range(6)
    )
    second = " ".join(
        f"Meanwhile the accountant reconciled ledger {i} against the quarterly filings."
        for i in range(6)
    )
    return first + " " + second


@pytest.fixture
def context(document, fake_encoder, fake_splitter) -> ChunkingContext:
    split = fake_splitter.split(document)
    counts = fake_encoder.count_tokens(split.sentences)
    return ChunkingContext(
        doc_id="doc-0",
        text=document,
        sentences=split.sentences,
        spans=split.spans,
        sentence_token_counts=counts,
        sentence_embeddings=fake_encoder.encode_sentences(split.sentences),
        total_document_tokens=sum(counts),
    )


@pytest.fixture
def sample() -> dict:
    return {
        "id": "narrativeqa-00001",
        "question": "Who charted the northern reef?",
        "context": "The mariner charted the northern reef. He logged the tides.",
        "answers": ["the mariner", "a mariner"],
    }
