"""LangChain ``SemanticChunker`` baseline, driven by the shared encoder.

The library's own sentence splitting is kept. Default breakpoint: 95th percentile.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseChunker, Chunk, ChunkingContext, chunks_from_texts, enforce_token_cap
from .recursive_chunker import MissingDependencyError

VALID_THRESHOLD_TYPES = (
    "percentile",
    "standard_deviation",
    "interquartile",
    "gradient",
)


class LangChainSemanticChunker(BaseChunker):
    """``langchain_experimental.text_splitter.SemanticChunker``."""

    name = "langchain_semantic"

    def __init__(
        self,
        encoder,
        sentence_splitter,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95,
        buffer_size: int = 1,
        min_chunk_size: Optional[int] = None,
        max_tokens: int = 480,
        apply_cap: bool = True,
    ) -> None:
        super().__init__(max_tokens=max_tokens, apply_cap=apply_cap)
        if breakpoint_threshold_type not in VALID_THRESHOLD_TYPES:
            raise ValueError(
                f"breakpoint_threshold_type must be one of {VALID_THRESHOLD_TYPES}"
            )
        self.encoder = encoder
        self.sentence_splitter = sentence_splitter
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.buffer_size = int(buffer_size)
        self.min_chunk_size = min_chunk_size
        self._splitter = None

    def params(self) -> Dict[str, Any]:
        params = super().params()
        params.update(
            {
                "breakpoint_threshold_type": self.breakpoint_threshold_type,
                "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
                "buffer_size": self.buffer_size,
                "min_chunk_size": self.min_chunk_size,
                "library": "langchain_experimental.text_splitter.SemanticChunker",
                "sentence_split": "library-internal regex (not the shared spaCy split)",
            }
        )
        return params

    @property
    def splitter(self):
        if self._splitter is None:
            self._splitter = self._build_splitter()
        return self._splitter

    def _build_splitter(self):
        try:
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'langchain_semantic' chunker needs langchain-experimental:\n"
                "  pip install langchain-experimental langchain-core"
            ) from exc

        from ..embeddings.encoder import LangChainEmbeddings

        kwargs: Dict[str, Any] = {
            "embeddings": LangChainEmbeddings(self.encoder),
            "breakpoint_threshold_type": self.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
            "buffer_size": self.buffer_size,
        }
        if self.min_chunk_size is not None:
            kwargs["min_chunk_size"] = self.min_chunk_size
        return SemanticChunker(**kwargs)

    def _chunk(self, ctx: ChunkingContext) -> List[Chunk]:
        texts = [t for t in self.splitter.split_text(ctx.text) if t.strip()]
        if not texts:
            texts = [ctx.text]
        token_counts = self.encoder.count_tokens(texts)
        chunks = chunks_from_texts(ctx, texts, token_counts)

        if self.apply_cap:
            chunks = enforce_token_cap(
                chunks, self.max_tokens, self.sentence_splitter, self.encoder
            )
        return chunks
