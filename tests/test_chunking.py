import numpy as np
import pytest

from seismos.chunking import ChunkSpec, DocumentSignal, chunk_ranges, make_chunks, pack_ranges
from seismos.datasets import join_passages
from seismos.retrieval import chunk_qrels, locate_chunk
from seismos.segment import split_sentences


def reference_chunk_indices(tcounts, split_at, cap):
    """Verbatim copy of ``chunk_indices`` from the ConTEB/DAPR/MIRACL runs."""
    n = len(tcounts)
    if n == 0:
        return []
    ranges, start, cur = [], 0, 0
    for i in range(n):
        t = tcounts[i]
        if i > start and cur + t > cap:
            ranges.append((start, i - 1)); start, cur = i, 0
        cur += t
        if i in split_at and i < n - 1:
            ranges.append((start, i)); start, cur = i + 1, 0
    ranges.append((start, n - 1))
    return ranges


def test_pack_ranges_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 80))
        counts = rng.integers(1, 200, size=n).tolist()
        splits = set(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False).tolist())
        for cap in (64, 256, 480):
            assert pack_ranges(counts, splits, cap) == reference_chunk_indices(counts, splits, cap)


def test_pack_ranges_without_cap_only_uses_boundaries():
    assert pack_ranges([100] * 6, {1, 3}, None) == [(0, 1), (2, 3), (4, 5)]


def test_oversized_sentence_is_kept_whole():
    assert pack_ranges([10, 900, 10], set(), 480) == [(0, 0), (1, 1), (2, 2)]


def _signal(n, dip_at=None):
    X = np.full(n - 1, 0.85, dtype=np.float32)
    if dip_at is not None:
        X[dip_at] = 0.1
    sents = [f"Sentence {i}." for i in range(n)]
    spans, pos = [], 0
    for s in sents:
        spans.append((pos, pos + len(s)))
        pos += len(s) + 1
    return DocumentSignal(sents, spans, X, [3] * n)


def test_short_documents_are_not_split():
    sig = _signal(4, dip_at=1)
    assert chunk_ranges(sig, ChunkSpec("seismos", k=0.1, max_tokens=None)) == [(0, 3)]


def test_seismos_chunks_cover_document_in_order():
    sig = _signal(80, dip_at=40)
    chunks = make_chunks(sig, ChunkSpec("seismos", k=2.5, max_tokens=None))
    assert [c.first_sentence for c in chunks][0] == 0
    assert chunks[-1].last_sentence == 79
    for a, b in zip(chunks, chunks[1:]):
        assert b.first_sentence == a.last_sentence + 1


def test_whole_and_fixed():
    sig = _signal(10)
    assert chunk_ranges(sig, ChunkSpec("whole")) == [(0, 9)]
    assert chunk_ranges(sig, ChunkSpec("fixed", max_tokens=9)) == [(0, 2), (3, 5), (6, 8), (9, 9)]


def test_labels_are_distinct():
    labels = {ChunkSpec("seismos", k=2.5).label, ChunkSpec("seismos", k=2.5, max_tokens=None).label,
              ChunkSpec("seismos", k=2.5, sign_convention="valley").label, ChunkSpec("fixed").label,
              ChunkSpec("zscore", k=2.5).label, ChunkSpec("prominence").label, ChunkSpec("whole").label}
    assert len(labels) == 7


@pytest.mark.parametrize("lang,text", [
    ("ja", "猫が座った。犬が走った！本当？"),
    ("ar", "بلغ عمر الأرض 4.5 مليار سنة. وهذا رقم كبير؟ نعم."),
])
def test_regex_segmentation_spans_are_exact(lang, text):
    sents, spans = split_sentences(text, lang)
    assert len(sents) == 3
    for s, (a, b) in zip(sents, spans):
        assert text[a:b] == s


def test_passage_midpoints_transfer_to_chunks():
    doc, mids = join_passages([("p0", "a" * 10), ("p1", "b" * 10), ("p2", "c" * 10)])
    assert doc[mids["p1"]] == "b"
    doc_chunks = {"d": [(0, 15, 0), (16, 32, 1)]}
    assert locate_chunk(doc_chunks["d"], mids["p0"]) == 0
    qrels = chunk_qrels(doc_chunks, {"q": {"p2": 1}}, {"p0": "d", "p1": "d", "p2": "d"}, mids)
    assert qrels == {"q": {"c1": 1}}
