"""FAISS indexing, top-k retrieval, and the retrieval record contract."""

from __future__ import annotations

import numpy as np
import pytest

from seismos_rag.retrieval.faiss_index import (
    NORM_TOLERANCE,
    IndexError_,
    build_flat_ip,
    check_normalized,
    search,
)
from seismos_rag.retrieval.retriever import CONTEXT_SEPARATOR, Retriever


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def chunks():
    return [
        {
            "chunk_id": i,
            "text": f"chunk number {i} about topic {i}",
            "token_count": 6,
            "start_position": i * 30,
            "end_position": (i + 1) * 30,
        }
        for i in range(8)
    ]


class TestNormalisation:
    def test_unit_vectors_pass(self):
        check_normalized(unit(np.random.default_rng(0).normal(size=(5, 16))))

    def test_unnormalised_vectors_raise(self):
        with pytest.raises(IndexError_, match="not unit-norm"):
            check_normalized(np.full((3, 4), 2.0, dtype=np.float32))

    def test_fp16_rounding_is_tolerated(self):
        vectors = unit(np.random.default_rng(1).normal(size=(5, 32)))
        check_normalized(vectors.astype(np.float16).astype(np.float32))

    def test_empty_input_is_a_no_op(self):
        check_normalized(np.zeros((0, 8), dtype=np.float32))

    def test_tolerance_is_tight_enough_to_catch_real_errors(self):
        assert NORM_TOLERANCE < 0.01


class TestIndex:
    def test_inner_product_equals_cosine_on_unit_vectors(self):
        rng = np.random.default_rng(2)
        vectors = unit(rng.normal(size=(20, 32))).astype(np.float32)
        index = build_flat_ip(vectors)
        query = vectors[7:8]
        scores, indices = search(index, query, top_k=1)
        assert indices[0][0] == 7
        assert scores[0][0] == pytest.approx(1.0, abs=1e-4)

    def test_scores_are_descending(self):
        rng = np.random.default_rng(3)
        vectors = unit(rng.normal(size=(30, 16))).astype(np.float32)
        index = build_flat_ip(vectors)
        scores, _ = search(index, unit(rng.normal(size=(1, 16))).astype(np.float32), 5)
        assert list(scores[0]) == sorted(scores[0], reverse=True)

    def test_empty_index_raises(self):
        with pytest.raises(IndexError_, match="zero vectors"):
            build_flat_ip(np.zeros((0, 8), dtype=np.float32))

    def test_one_dimensional_input_raises(self):
        with pytest.raises(IndexError_, match="2-D"):
            build_flat_ip(np.zeros(8, dtype=np.float32))

    def test_unnormalised_index_is_rejected(self):
        with pytest.raises(IndexError_):
            build_flat_ip(np.full((4, 8), 3.0, dtype=np.float32))

    def test_k_is_clipped_to_index_size(self):
        vectors = unit(np.random.default_rng(4).normal(size=(3, 8))).astype(np.float32)
        index = build_flat_ip(vectors)
        scores, indices = search(index, vectors[0:1], top_k=10)
        assert indices.shape == (1, 3)
        assert -1 not in indices  # no phantom padding ids

    def test_search_is_deterministic(self):
        rng = np.random.default_rng(5)
        vectors = unit(rng.normal(size=(50, 16))).astype(np.float32)
        query = unit(rng.normal(size=(1, 16))).astype(np.float32)
        index = build_flat_ip(vectors)
        first = search(index, query, 5)
        second = search(index, query, 5)
        assert np.array_equal(first[1], second[1])
        assert np.array_equal(first[0], second[0])


class TestRetriever:
    def test_returns_top_k_hits(self, fake_encoder, chunks):
        result = Retriever(fake_encoder, top_k=5).retrieve(
            "q1", "fixed", "what is topic 3", chunks
        )
        assert len(result.hits) == 5
        assert [h.rank for h in result.hits] == [0, 1, 2, 3, 4]

    def test_short_document_returns_what_exists(self, fake_encoder, chunks):
        result = Retriever(fake_encoder, top_k=5).retrieve(
            "q1", "fixed", "question", chunks[:2]
        )
        assert len(result.hits) == 2
        assert result.diagnostics["truncated_by_document_size"] is True

    def test_hits_carry_the_source_chunk_metadata(self, fake_encoder, chunks):
        result = Retriever(fake_encoder, top_k=3).retrieve("q1", "fixed", "topic", chunks)
        by_id = {c["chunk_id"]: c for c in chunks}
        for hit in result.hits:
            assert hit.text == by_id[hit.chunk_id]["text"]
            assert hit.token_count == by_id[hit.chunk_id]["token_count"]

    def test_scores_are_descending(self, fake_encoder, chunks):
        result = Retriever(fake_encoder, top_k=5).retrieve("q1", "fixed", "topic", chunks)
        assert result.scores == sorted(result.scores, reverse=True)

    def test_empty_chunk_list_raises(self, fake_encoder):
        with pytest.raises(ValueError, match="no chunks"):
            Retriever(fake_encoder, top_k=5).retrieve("q1", "fixed", "question", [])

    def test_embedding_count_mismatch_raises(self, fake_encoder, chunks):
        with pytest.raises(ValueError, match="embeddings for"):
            Retriever(fake_encoder, top_k=5).retrieve(
                "q1",
                "fixed",
                "question",
                chunks,
                chunk_embeddings=fake_encoder.encode_passages(["a", "b"]),
            )

    def test_retrieval_is_deterministic(self, fake_encoder, chunks):
        retriever = Retriever(fake_encoder, top_k=5)
        first = retriever.retrieve("q1", "fixed", "topic 2", chunks)
        second = retriever.retrieve("q1", "fixed", "topic 2", chunks)
        assert first.retrieved_chunk_ids == second.retrieved_chunk_ids
        assert first.scores == pytest.approx(second.scores)


class TestRetrievalRecord:
    def test_context_efficiency_accounting(self, fake_encoder, chunks):
        result = Retriever(fake_encoder, top_k=4).retrieve("q1", "fixed", "topic", chunks)
        assert result.retrieved_chunk_tokens == sum(h.token_count for h in result.hits)
        assert result.avg_retrieved_chunk_tokens == pytest.approx(
            result.retrieved_chunk_tokens / 4
        )

    def test_context_joins_with_the_pipeline_separator(self, fake_encoder, chunks):
        result = Retriever(fake_encoder, top_k=3).retrieve("q1", "fixed", "topic", chunks)
        assert result.context() == CONTEXT_SEPARATOR.join(result.texts)

    def test_record_round_trips_the_required_fields(self, fake_encoder, chunks):
        from seismos_rag.retrieval.retriever import RetrievalResult

        original = Retriever(fake_encoder, top_k=3).retrieve("q1", "seismos", "topic", chunks)
        record = original.to_record()
        for key in ("qid", "chunker", "retrieved_chunk_ids", "scores", "retrieved_text"):
            assert key in record

        restored = RetrievalResult.from_record(record)
        assert restored.retrieved_chunk_ids == original.retrieved_chunk_ids
        assert restored.texts == original.texts
        assert restored.scores == pytest.approx(original.scores)
