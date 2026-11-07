"""Exact ``IndexFlatIP`` over unit-norm vectors, one index per document."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..utils.logging import get_logger

#: Vectors whose L2 norm deviates from 1 by more than this are treated as
#: unnormalised. fp16 encoding introduces error around 1e-3.
NORM_TOLERANCE = 5e-3


class IndexError_(RuntimeError):
    pass


def check_normalized(embeddings: np.ndarray, label: str = "embeddings") -> None:
    if embeddings.size == 0:
        return
    norms = np.linalg.norm(embeddings, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > NORM_TOLERANCE:
        raise IndexError_(
            f"{label} are not unit-norm (max |‖v‖-1| = {worst:.4g} > {NORM_TOLERANCE}). "
            "IndexFlatIP would not compute cosine similarity."
        )


def build_flat_ip(embeddings: np.ndarray, verify_normalized: bool = True):
    """Build an ``IndexFlatIP`` over ``embeddings`` (n_vectors x dim, float32)."""
    import faiss

    if embeddings.ndim != 2:
        raise IndexError_(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise IndexError_("cannot build an index over zero vectors")

    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    if verify_normalized:
        check_normalized(vectors, "chunk embeddings")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(
    index, query: np.ndarray, top_k: int, verify_normalized: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """``k`` is clipped to the index size, so no ``-1`` padding is returned."""
    if query.ndim == 1:
        query = query.reshape(1, -1)
    vectors = np.ascontiguousarray(query, dtype=np.float32)
    if verify_normalized:
        check_normalized(vectors, "query embeddings")

    k = int(min(top_k, index.ntotal))
    if k <= 0:
        raise IndexError_("index is empty")
    if k < top_k:
        get_logger().debug(
            "requested top_k=%d but index holds %d vectors; returning %d",
            top_k,
            index.ntotal,
            k,
        )

    scores, indices = index.search(vectors, k)
    return scores, indices


def free_index(index: Optional[object]) -> None:
    """Drop a per-document index. FAISS frees on refcount zero; make it explicit."""
    del index
