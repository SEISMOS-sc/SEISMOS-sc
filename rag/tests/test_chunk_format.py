"""Chunk output contract, the packer, span recovery, and the SEISMOS detector."""

from __future__ import annotations

import numpy as np
import pytest

from seismos_rag.chunkers.base import (
    CHUNK_KEYS,
    ChunkFormatError,
    chunk_statistics,
    chunks_from_texts,
    locate_spans,
    pack_sentence_ranges,
    validate_chunks,
)
from seismos_rag.chunkers.fixed_chunker import FixedChunker
from seismos_rag.chunkers.seismos import (
    PAPER_K,
    SeismosChunker,
    build_k_sweep_chunkers,
    laplacian_diagnostics,
    laplacian_splits,
)


class TestContract:
    def test_keys_match_spec(self):
        assert CHUNK_KEYS == (
            "chunk_id",
            "text",
            "token_count",
            "start_position",
            "end_position",
        )

    def test_fixed_chunker_output_conforms(self, context):
        chunks = FixedChunker(target_tokens=40, max_tokens=40).chunk(context)
        validate_chunks(chunks, context.text, chunker="fixed")
        assert all(set(CHUNK_KEYS) <= set(c) for c in chunks)

    def test_seismos_output_conforms(self, context):
        chunks = SeismosChunker(k=1.0, max_tokens=40).chunk(context)
        validate_chunks(chunks, context.text, chunker="seismos")

    def test_spans_reproduce_the_source_text(self, context):
        for chunk in FixedChunker(target_tokens=40, max_tokens=40).chunk(context):
            start, end = chunk["start_position"], chunk["end_position"]
            assert context.text[start:end] == chunk["text"]

    def test_chunk_ids_are_contiguous_from_zero(self, context):
        chunks = FixedChunker(target_tokens=30, max_tokens=30).chunk(context)
        assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))

    def test_spans_are_non_overlapping_and_increasing(self, context):
        chunks = FixedChunker(target_tokens=30, max_tokens=30).chunk(context)
        for previous, current in zip(chunks, chunks[1:]):
            assert previous["end_position"] <= current["start_position"]


class TestValidation:
    def _chunk(self, **overrides):
        base = {
            "chunk_id": 0,
            "text": "hello world",
            "token_count": 2,
            "start_position": 0,
            "end_position": 11,
        }
        base.update(overrides)
        return [base]

    def test_empty_list_raises(self):
        with pytest.raises(ChunkFormatError, match="zero chunks"):
            validate_chunks([], "text")

    def test_missing_key_raises(self):
        chunk = self._chunk()
        del chunk[0]["token_count"]
        with pytest.raises(ChunkFormatError, match="token_count"):
            validate_chunks(chunk, "hello world")

    def test_wrong_chunk_id_raises(self):
        with pytest.raises(ChunkFormatError, match="0-based position"):
            validate_chunks(self._chunk(chunk_id=3), "hello world")

    def test_blank_text_raises(self):
        with pytest.raises(ChunkFormatError, match="non-empty"):
            validate_chunks(self._chunk(text="   "), "hello world")

    def test_negative_token_count_raises(self):
        with pytest.raises(ChunkFormatError, match="non-negative"):
            validate_chunks(self._chunk(token_count=-1), "hello world")

    def test_inverted_span_raises(self):
        with pytest.raises(ChunkFormatError, match="must exceed"):
            validate_chunks(self._chunk(start_position=5, end_position=2), "hello world")

    def test_span_past_document_end_raises(self):
        with pytest.raises(ChunkFormatError, match="exceeds document length"):
            validate_chunks(self._chunk(end_position=999), "hello world")

    def test_half_unresolved_span_raises(self):
        with pytest.raises(ChunkFormatError, match="unresolved spans"):
            validate_chunks(self._chunk(start_position=-1, end_position=11), "hello world")

    def test_fully_unresolved_span_is_allowed(self):
        validate_chunks(self._chunk(start_position=-1, end_position=-1), "hello world")


class TestPacker:
    def test_respects_the_cap(self):
        counts = [10] * 10
        for first, last in pack_sentence_ranges(counts, set(), cap=25):
            assert sum(counts[first : last + 1]) <= 25

    def test_covers_every_sentence_exactly_once(self):
        counts = [7, 3, 9, 4, 6, 2]
        ranges = pack_sentence_ranges(counts, {2}, cap=15)
        covered = [i for first, last in ranges for i in range(first, last + 1)]
        assert covered == list(range(len(counts)))

    def test_semantic_split_forces_a_boundary(self):
        ranges = pack_sentence_ranges([5] * 6, {2}, cap=1000)
        assert ranges[0] == (0, 2)

    def test_oversized_single_sentence_is_not_split(self):
        # The packer never cuts inside a sentence; the chunk legitimately
        # exceeds the cap. This is documented behaviour, so pin it.
        ranges = pack_sentence_ranges([5, 900, 5], set(), cap=100)
        assert (1, 1) in ranges

    def test_empty_input_gives_no_ranges(self):
        assert pack_sentence_ranges([], set(), cap=100) == []

    def test_trailing_split_index_is_ignored(self):
        # A split after the final sentence would create an empty trailing chunk.
        assert pack_sentence_ranges([5, 5, 5], {2}, cap=1000) == [(0, 2)]


class TestSeismosDetector:
    """Pins the exact detector geometry used by every published SEISMOS run."""

    def test_flat_signal_yields_no_boundaries(self):
        assert laplacian_splits(np.full(20, 0.8), k=2.5) == set()

    def test_none_signal_yields_no_boundaries(self):
        assert laplacian_splits(None, k=2.5) == set()

    def test_short_signal_yields_no_boundaries(self):
        assert laplacian_splits(np.array([0.9, 0.8]), k=2.5) == set()

    def test_isolated_dip_fires_at_the_shoulders_not_the_floor(self):
        """The published rule fires beside an isolated dip, not at it."""
        signal = np.full(21, 0.9)
        signal[10] = 0.1

        fired = laplacian_splits(signal, k=1.5)
        assert fired == {9, 11}
        assert 10 not in fired  # the actual minimum never fires

    def test_valley_ablation_fires_at_the_floor(self):
        signal = np.full(21, 0.9)
        signal[10] = 0.1
        assert laplacian_splits(signal, k=1.5, sign_convention="valley") == {10}

    def test_shoulder_statistic_is_sqrt_n_over_6_regardless_of_depth(self):
        """For one isolated dip, t at the shoulders is sqrt(N/6) whatever the depth."""
        for depth in (0.1, 0.4, 0.8):
            signal = np.full(21, 0.9)
            signal[10] = 0.9 - depth
            n = len(signal)
            D = np.array([signal[i + 1] - 2.0 * signal[i] + signal[i - 1] for i in range(1, n - 1)])
            shoulder_t = -D[8] / D.std()
            assert shoulder_t == pytest.approx(np.sqrt(len(D) / 6.0), rel=1e-6)

        # Consequence: too short to fire at k=2.5, long enough at N=60.
        short = np.full(21, 0.9)
        short[10] = 0.1
        assert laplacian_splits(short, k=2.5) == set()

        long_signal = np.full(61, 0.9)
        long_signal[30] = 0.1
        assert laplacian_splits(long_signal, k=2.5) == {29, 31}

    def test_unknown_sign_convention_raises(self):
        with pytest.raises(ValueError, match="sign_convention"):
            laplacian_splits(np.full(20, 0.5), k=2.5, sign_convention="nonsense")

    def test_higher_k_is_strictly_more_conservative(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0.8, 0.05, size=200)
        signal[50] = 0.1
        signal[120] = 0.2
        counts = [len(laplacian_splits(signal, k=k)) for k in (1.5, 2.0, 2.5, 3.0, 3.5)]
        assert counts == sorted(counts, reverse=True)

    def test_boundaries_at_higher_k_are_a_subset_of_lower_k(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(0.75, 0.08, size=150)
        assert laplacian_splits(signal, 3.0) <= laplacian_splits(signal, 2.0)

    def test_statistic_matches_the_published_formula(self):
        """t_i = -(X[i+1] - 2X[i] + X[i-1]) / std(D), boundary iff t_i > k."""
        rng = np.random.default_rng(7)
        X = rng.normal(0.8, 0.1, size=40)
        n = len(X)
        D = np.array([X[i + 1] - 2.0 * X[i] + X[i - 1] for i in range(1, n - 1)])
        expected = {
            i for idx, i in enumerate(range(1, n - 1)) if (-D[idx] / D.std()) > 2.5
        }
        assert laplacian_splits(X, 2.5) == expected

    def test_boundaries_lie_in_the_interior(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(0.8, 0.1, size=30)
        boundaries = laplacian_splits(signal, k=1.0)
        assert all(1 <= b <= len(signal) - 2 for b in boundaries)

    def test_paper_k_constant(self):
        assert PAPER_K == 2.5


class TestSeismosChunker:
    def test_short_document_is_gated_whole(self, fake_encoder, fake_splitter):
        from seismos_rag.chunkers.base import ChunkingContext

        text = "One sentence. Two sentence. Three sentence."
        split = fake_splitter.split(text)
        counts = fake_encoder.count_tokens(split.sentences)
        ctx = ChunkingContext(
            doc_id="short",
            text=text,
            sentences=split.sentences,
            spans=split.spans,
            sentence_token_counts=counts,
            sentence_embeddings=fake_encoder.encode_sentences(split.sentences),
            total_document_tokens=sum(counts),
        )
        chunks = SeismosChunker(k=2.5, min_sentences=5).chunk(ctx)
        assert len(chunks) == 1

    def test_diagnostics_are_recorded(self, context):
        chunker = SeismosChunker(k=2.5)
        chunker.chunk(context)
        assert "seismos" in context.diagnostics
        assert "acf_factor" in context.diagnostics["seismos"]["seismos"]

    def test_k_sweep_names_are_unique(self):
        chunkers = build_k_sweep_chunkers([1.5, 2.0, 2.5, 3.0, 3.5])
        names = [c.name for c in chunkers]
        assert names == [
            "seismos_k1.5",
            "seismos_k2",
            "seismos_k2.5",
            "seismos_k3",
            "seismos_k3.5",
        ]
        assert len(set(names)) == len(names)

    def test_params_record_the_detector(self):
        params = SeismosChunker(k=3.0).params()
        assert params["detector"] == "discrete_laplacian"
        assert params["k"] == 3.0
        assert params["paper_k"] == 2.5
        assert params["sign_convention"] == "canonical"
        assert params["is_published_detector"] is True

    def test_ablation_is_flagged_as_not_the_published_detector(self):
        params = SeismosChunker(k=3.0, sign_convention="valley").params()
        assert params["is_published_detector"] is False
        assert params["statistic"] == "+D_i / std(D)"

    def test_invalid_sign_convention_raises_at_construction(self):
        with pytest.raises(ValueError, match="sign_convention"):
            SeismosChunker(k=2.5, sign_convention="backwards")

    def test_k_sweep_names_disambiguate_the_ablation(self):
        names = [c.name for c in build_k_sweep_chunkers([2.5], sign_convention="valley")]
        assert names == ["seismos_k2.5_valley"]


class TestDiagnostics:
    def test_acf_factor_is_near_theory_for_ar1_signal(self):
        """std(D)/std(X) should approach sqrt(6 - 8*rho + 2*rho^2)."""
        rho = 0.35
        rng = np.random.default_rng(11)
        n = 20000
        signal = np.zeros(n)
        for i in range(1, n):
            signal[i] = rho * signal[i - 1] + rng.normal(0, 1)

        observed = laplacian_diagnostics(signal)["acf_factor"]
        theoretical = float(np.sqrt(6 - 8 * rho + 2 * rho**2))
        assert observed == pytest.approx(theoretical, rel=0.05)

    def test_lag1_autocorrelation_is_recovered(self):
        rho = 0.5
        rng = np.random.default_rng(5)
        n = 50000
        signal = np.zeros(n)
        for i in range(1, n):
            signal[i] = rho * signal[i - 1] + rng.normal(0, 1)
        assert laplacian_diagnostics(signal)["lag1_autocorr"] == pytest.approx(rho, abs=0.02)

    def test_short_signal_returns_nulls(self):
        result = laplacian_diagnostics(np.array([0.5, 0.6]))
        assert result["acf_factor"] is None


class TestSpanRecovery:
    def test_exact_substrings_resolve(self):
        text = "Alpha beta. Gamma delta. Epsilon zeta."
        assert locate_spans(text, ["Alpha beta.", "Gamma delta."]) == [(0, 11), (12, 24)]

    def test_whitespace_collapsed_chunks_still_resolve(self):
        text = "Alpha  beta.\n\nGamma   delta."
        spans = locate_spans(text, ["Alpha beta.", "Gamma delta."])
        assert all(span != (-1, -1) for span in spans)
        assert text[spans[0][0] : spans[0][1]].split() == ["Alpha", "beta."]

    def test_absent_chunk_is_marked_unresolved_not_guessed(self):
        assert locate_spans("Alpha beta.", ["Totally different text"]) == [(-1, -1)]

    def test_repeated_text_advances_the_cursor(self):
        text = "same. same. same."
        spans = locate_spans(text, ["same.", "same.", "same."])
        assert spans[0][0] < spans[1][0] < spans[2][0]

    def test_chunks_from_texts_marks_diagnostics(self, context):
        chunks = chunks_from_texts(context, ["nowhere to be found"], [4])
        assert chunks[0]["start_position"] == -1
        assert context.diagnostics["unresolved_spans"] == 1


class TestChunkStatistics:
    def test_statistics_match_the_chunks(self, context):
        chunks = FixedChunker(target_tokens=40, max_tokens=40).chunk(context)
        stats = chunk_statistics(chunks, context.total_document_tokens)
        counts = [c["token_count"] for c in chunks]
        assert stats["num_chunks"] == len(chunks)
        assert stats["max_chunk_tokens"] == max(counts)
        assert stats["avg_chunk_tokens"] == pytest.approx(sum(counts) / len(counts))
        assert stats["total_document_tokens"] == context.total_document_tokens

    def test_empty_chunks_give_zeroed_statistics(self):
        stats = chunk_statistics([], 100)
        assert stats["num_chunks"] == 0
        assert stats["total_document_tokens"] == 100


class TestChunkerParity:
    """Fixed and SEISMOS must differ only in their boundary set."""

    def test_seismos_with_no_detected_boundaries_equals_fixed(self, context):
        # A flat signal produces no boundaries, so SEISMOS reduces to the packer.
        context.sentence_embeddings = np.tile(
            np.eye(1, 16, 0, dtype=np.float32), (len(context.sentences), 1)
        )
        fixed = FixedChunker(target_tokens=40, max_tokens=40).chunk(context)
        seismos = SeismosChunker(k=2.5, max_tokens=40).chunk(context)
        assert [c["text"] for c in fixed] == [c["text"] for c in seismos]

    def test_lower_k_never_yields_fewer_chunks(self, context):
        counts = [
            len(SeismosChunker(k=k, max_tokens=10_000).chunk(context)) for k in (1.0, 3.5)
        ]
        assert counts[0] >= counts[1]
