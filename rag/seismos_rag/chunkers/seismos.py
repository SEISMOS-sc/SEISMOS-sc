"""SEISMOS chunker. The detector comes from :mod:`seismos.detectors`; this module
adds the harness interface and the shared token cap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from seismos.detectors import PAPER_K, SIGN_CONVENTIONS, laplacian_splits
from seismos.detectors import signal_diagnostics as laplacian_diagnostics

from .base import (
    BaseChunker,
    Chunk,
    ChunkingContext,
    pack_sentence_ranges,
    ranges_to_chunks,
    whole_document_chunk,
)

DEFAULT_K_SWEEP = (1.5, 2.0, 2.5, 3.0, 3.5)

__all__ = ["SeismosChunker", "build_k_sweep_chunkers", "laplacian_splits", "laplacian_diagnostics",
           "PAPER_K", "DEFAULT_K_SWEEP", "SIGN_CONVENTIONS"]


class SeismosChunker(BaseChunker):
    """SEISMOS chunker with the shared token cap."""

    def __init__(
        self,
        k: float = 2.5,
        min_sentences: int = 5,
        max_tokens: int = 480,
        apply_cap: bool = True,
        name: Optional[str] = None,
        sign_convention: str = "canonical",
    ) -> None:
        super().__init__(max_tokens=max_tokens, apply_cap=apply_cap)
        if sign_convention not in SIGN_CONVENTIONS:
            raise ValueError(f"sign_convention must be one of {SIGN_CONVENTIONS}")
        self.k = float(k)
        self.min_sentences = int(min_sentences)
        self.sign_convention = sign_convention
        self.name = name or "seismos"

    def params(self) -> Dict[str, Any]:
        params = super().params()
        params.update(
            {
                "k": self.k,
                "min_sentences": self.min_sentences,
                "detector": "discrete_laplacian",
                "statistic": (
                    "-D_i / std(D)" if self.sign_convention == "canonical" else "+D_i / std(D)"
                ),
                "sign_convention": self.sign_convention,
                "is_published_detector": self.sign_convention == "canonical",
                "paper_k": PAPER_K,
            }
        )
        return params

    def _chunk(self, ctx: ChunkingContext) -> List[Chunk]:
        if ctx.n_sentences == 0:
            return []

        X = ctx.adjacent_similarity()
        ctx.diagnostics.setdefault("seismos", {})[self.name] = laplacian_diagnostics(X)

        # Structural gate: too few sentences to estimate std(D).
        if ctx.n_sentences < self.min_sentences:
            ctx.diagnostics.setdefault("gated_docs", []).append(self.name)
            return whole_document_chunk(ctx)

        split_after = laplacian_splits(X, self.k, self.sign_convention)
        ctx.diagnostics.setdefault("n_semantic_splits", {})[self.name] = len(split_after)

        cap = self.max_tokens if self.apply_cap else 10**9
        ranges = pack_sentence_ranges(ctx.sentence_token_counts, split_after, cap)
        return ranges_to_chunks(ctx, ranges)


def build_k_sweep_chunkers(
    k_values,
    min_sentences: int = 5,
    max_tokens: int = 480,
    apply_cap: bool = True,
    sign_convention: str = "canonical",
) -> List[SeismosChunker]:
    """One chunker per ``k``, each with a distinct method name for the CSVs."""
    suffix = "" if sign_convention == "canonical" else f"_{sign_convention}"
    chunkers = []
    for k in k_values:
        label = f"seismos_k{float(k):g}{suffix}"
        chunkers.append(
            SeismosChunker(
                k=k,
                min_sentences=min_sentences,
                max_tokens=max_tokens,
                apply_cap=apply_cap,
                name=label,
                sign_convention=sign_convention,
            )
        )
    return chunkers
