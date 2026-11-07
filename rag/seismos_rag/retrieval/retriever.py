"""Per-document dense retrieval: embed one chunker's chunks, index, take top-k."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..utils.logging import get_logger
from .faiss_index import build_flat_ip, search

CONTEXT_SEPARATOR = "\n\n"


@dataclass
class RetrievalHit:
    rank: int
    chunk_id: int
    score: float
    text: str
    token_count: int
    start_position: int
    end_position: int

    def as_row(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "chunk_text": self.text,
            "chunk_tokens": self.token_count,
            "chunk_start": self.start_position,
            "chunk_end": self.end_position,
        }


@dataclass
class RetrievalResult:
    qid: str
    chunker: str
    question: str
    hits: List[RetrievalHit]
    num_chunks_in_document: int
    retrieval_seconds: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def retrieved_chunk_ids(self) -> List[int]:
        return [h.chunk_id for h in self.hits]

    @property
    def scores(self) -> List[float]:
        return [h.score for h in self.hits]

    @property
    def texts(self) -> List[str]:
        return [h.text for h in self.hits]

    def context(self, separator: str = CONTEXT_SEPARATOR) -> str:
        """The retrieved context block handed to the generator."""
        return separator.join(self.texts)

    @property
    def retrieved_chunk_tokens(self) -> int:
        """Encoder-token total of the retrieved chunks."""
        return int(sum(h.token_count for h in self.hits))

    @property
    def avg_retrieved_chunk_tokens(self) -> float:
        return self.retrieved_chunk_tokens / len(self.hits) if self.hits else 0.0

    def to_record(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "chunker": self.chunker,
            "question": self.question,
            "num_chunks_in_document": self.num_chunks_in_document,
            "num_retrieved": len(self.hits),
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "scores": self.scores,
            "retrieved_text": self.texts,
            "retrieved_chunk_tokens": self.retrieved_chunk_tokens,
            "avg_retrieved_chunk_tokens": self.avg_retrieved_chunk_tokens,
            "retrieval_seconds": self.retrieval_seconds,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "RetrievalResult":
        hits = [
            RetrievalHit(
                rank=rank,
                chunk_id=int(cid),
                score=float(score),
                text=text,
                token_count=0,
                start_position=-1,
                end_position=-1,
            )
            for rank, (cid, score, text) in enumerate(
                zip(record["retrieved_chunk_ids"], record["scores"], record["retrieved_text"])
            )
        ]
        result = cls(
            qid=record["qid"],
            chunker=record["chunker"],
            question=record.get("question", ""),
            hits=hits,
            num_chunks_in_document=int(record.get("num_chunks_in_document", 0)),
            retrieval_seconds=float(record.get("retrieval_seconds", 0.0)),
            diagnostics=record.get("diagnostics", {}),
        )
        # Token counts are not recoverable per-hit from the record, so restore
        # the aggregate the caller actually consumes.
        result.diagnostics.setdefault(
            "restored_retrieved_chunk_tokens",
            int(record.get("retrieved_chunk_tokens", 0)),
        )
        return result


class Retriever:
    """Builds a per-document index and retrieves top-k chunks."""

    def __init__(self, encoder, top_k: int = 5) -> None:
        self.encoder = encoder
        self.top_k = int(top_k)

    def retrieve(
        self,
        qid: str,
        chunker_name: str,
        question: str,
        chunks: Sequence[Dict[str, Any]],
        query_embedding: Optional[np.ndarray] = None,
        chunk_embeddings: Optional[np.ndarray] = None,
    ) -> RetrievalResult:
        import time

        if not chunks:
            raise ValueError(f"{qid}/{chunker_name}: no chunks to retrieve from")

        texts = [c["text"] for c in chunks]

        start = time.perf_counter()
        if chunk_embeddings is None:
            chunk_embeddings = self.encoder.encode_chunks_cached(texts)
        if query_embedding is None:
            query_embedding = self.encoder.encode_queries_cached([question])[0]

        if chunk_embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"{qid}/{chunker_name}: {chunk_embeddings.shape[0]} embeddings "
                f"for {len(chunks)} chunks"
            )

        index = build_flat_ip(chunk_embeddings)
        scores, indices = search(index, query_embedding, self.top_k)
        elapsed = time.perf_counter() - start

        hits: List[RetrievalHit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            chunk = chunks[int(idx)]
            hits.append(
                RetrievalHit(
                    rank=rank,
                    chunk_id=int(chunk["chunk_id"]),
                    score=float(score),
                    text=chunk["text"],
                    token_count=int(chunk["token_count"]),
                    start_position=int(chunk["start_position"]),
                    end_position=int(chunk["end_position"]),
                )
            )

        if not hits:
            raise ValueError(f"{qid}/{chunker_name}: retrieval returned no hits")

        return RetrievalResult(
            qid=qid,
            chunker=chunker_name,
            question=question,
            hits=hits,
            num_chunks_in_document=len(chunks),
            retrieval_seconds=elapsed,
            diagnostics={
                "top_k_requested": self.top_k,
                "top_k_returned": len(hits),
                "truncated_by_document_size": len(hits) < self.top_k,
            },
        )


def log_retrieval_summary(results: Sequence[RetrievalResult]) -> None:
    if not results:
        return
    log = get_logger()
    latencies = np.array([r.retrieval_seconds for r in results])
    tokens = np.array([r.retrieved_chunk_tokens for r in results])
    log.info(
        "retrieval: %d queries | latency mean=%.1f ms p95=%.1f ms | "
        "retrieved chunk tokens mean=%.0f",
        len(results),
        float(latencies.mean() * 1000),
        float(np.percentile(latencies, 95) * 1000),
        float(tokens.mean()),
    )
