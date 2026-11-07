"""Sentence spans, long-document blocking and caching, on ``spacy.blank`` + sentencizer."""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")

from seismos_rag.chunkers.sentence_splitter import (  # noqa: E402
    SentenceSplitter,
    _iter_blocks,
    _merge_degenerate,
    validate_spans,
)


@pytest.fixture
def splitter(tmp_path, monkeypatch) -> SentenceSplitter:
    """A SentenceSplitter backed by the rule-based sentencizer."""

    def blank_pipeline(self):
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.max_length = self.max_length
        return nlp

    monkeypatch.setattr(SentenceSplitter, "_load_spacy", blank_pipeline)
    return SentenceSplitter(cache_dir=str(tmp_path / "sentcache"))


class TestSpans:
    def test_spans_reproduce_their_sentences(self, splitter):
        text = "First sentence here. Second one follows! And a third? Yes."
        split = splitter.split(text)
        validate_spans(text, split)

    def test_leading_and_trailing_whitespace_is_trimmed(self, splitter):
        text = "Alpha beta.    Gamma delta.\n\n   Epsilon."
        split = splitter.split(text)
        for sentence in split.sentences:
            assert sentence == sentence.strip()

    def test_spans_are_strictly_increasing(self, splitter):
        text = " ".join(f"Sentence number {i} is here." for i in range(30))
        split = splitter.split(text)
        for previous, current in zip(split.spans, split.spans[1:]):
            assert previous[1] <= current[0]

    def test_multiple_sentences_are_found(self, splitter):
        split = splitter.split("One. Two. Three. Four. Five.")
        assert len(split) == 5

    def test_empty_text_gives_empty_split(self, splitter):
        split = splitter.split("")
        assert len(split) == 0

    def test_whitespace_only_text_gives_no_sentences(self, splitter):
        assert len(splitter.split("   \n\n  ")) == 0

    def test_single_sentence_without_terminator(self, splitter):
        split = splitter.split("no terminating punctuation here")
        assert len(split) == 1
        validate_spans("no terminating punctuation here", split)


class TestCaching:
    def test_second_call_hits_the_cache(self, splitter):
        text = "Alpha beta gamma. Delta epsilon zeta."
        first = splitter.split(text)
        assert splitter.cache_stats()["misses"] == 1

        second = splitter.split(text)
        assert splitter.cache_stats()["hits"] == 1
        assert first.sentences == second.sentences
        assert first.spans == second.spans

    def test_cached_sentences_are_rebuilt_from_spans(self, splitter):
        """Spans are the cached source of truth, so they cannot drift from text."""
        text = "Alpha beta. Gamma delta."
        splitter.split(text)
        restored = splitter.split(text)
        validate_spans(text, restored)

    def test_different_text_does_not_collide(self, splitter):
        a = splitter.split("Alpha beta. Gamma delta.")
        b = splitter.split("Totally different content here. And more of it.")
        assert a.sentences != b.sentences


class TestBlocking:
    """Long documents are processed in blocks; the stitching must be lossless."""

    def test_short_text_is_one_block(self):
        blocks = list(_iter_blocks("short text", 1000))
        assert blocks == [(0, "short text")]

    def test_blocks_cover_the_text_without_loss(self):
        text = " ".join(f"word{i}" for i in range(5000))
        blocks = list(_iter_blocks(text, 1000))
        assert "".join(block for _, block in blocks) == text

    def test_offsets_index_back_correctly(self):
        text = " ".join(f"word{i}" for i in range(5000))
        for offset, block in _iter_blocks(text, 1000):
            assert text[offset : offset + len(block)] == block

    def test_blocks_are_cut_at_whitespace(self):
        text = " ".join(f"word{i}" for i in range(5000))
        blocks = list(_iter_blocks(text, 1000))
        for _, block in blocks[:-1]:
            # A cut inside a token would leave the block ending mid-word.
            assert not block.endswith("wor")

    def test_long_document_splits_and_spans_stay_exact(self, splitter):
        text = " ".join(f"This is sentence number {i} of a long document." for i in range(4000))
        split = splitter.split(text)
        assert len(split) > 3000
        validate_spans(text, split)


class TestDegenerateSpans:
    def test_zero_length_spans_are_dropped(self):
        assert _merge_degenerate([(0, 5), (5, 5), (6, 10)]) == [(0, 5), (6, 10)]

    def test_overlapping_spans_are_merged(self):
        assert _merge_degenerate([(0, 10), (5, 15)]) == [(0, 15)]

    def test_disjoint_spans_pass_through(self):
        assert _merge_degenerate([(0, 5), (6, 10)]) == [(0, 5), (6, 10)]


class TestModelLoading:
    def test_missing_model_gives_an_actionable_error(self):
        splitter = SentenceSplitter(spacy_model="definitely_not_a_model", cache_dir=None)
        with pytest.raises(RuntimeError, match="python -m spacy download"):
            _ = splitter.nlp

    def test_unknown_pipe_is_rejected(self, tmp_path):
        splitter = SentenceSplitter(pipe="nonsense", cache_dir=None)
        with pytest.raises((ValueError, RuntimeError)):
            _ = splitter.nlp


@pytest.mark.slow
class TestRealModel:
    def test_en_core_web_sm_produces_exact_spans(self):
        pytest.importorskip("en_core_web_sm")
        splitter = SentenceSplitter(cache_dir=None)
        text = (
            "Dr. Smith arrived at 3 p.m. on Tuesday. The reef was charted that "
            "evening. Mr. Jones disagreed."
        )
        split = splitter.split(text)
        validate_spans(text, split)
        assert len(split) >= 2
