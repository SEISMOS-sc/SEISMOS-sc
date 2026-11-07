"""LangChain ``RecursiveCharacterTextSplitter`` baseline, with ``chunk_size``
counted in encoder tokens rather than characters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import BaseChunker, Chunk, ChunkingContext, chunks_from_texts


class MissingDependencyError(RuntimeError):
    """Raised when an optional chunker's library is not installed."""


class RecursiveChunker(BaseChunker):
    """LangChain recursive character splitter, token-budgeted."""

    name = "recursive"

    def __init__(
        self,
        encoder,
        chunk_size: int = 480,
        chunk_overlap: int = 0,
        separators: Optional[Sequence[str]] = None,
        max_tokens: int = 480,
    ) -> None:
        super().__init__(max_tokens=max_tokens, apply_cap=False)
        self.encoder = encoder
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.separators = list(separators) if separators else ["\n\n", "\n", ". ", " ", ""]
        self._splitter = None

    def params(self) -> Dict[str, Any]:
        params = super().params()
        params.update(
            {
                "chunk_size": self.chunk_size,
                "chunk_size_unit": "tokens (encoder tokenizer)",
                "chunk_overlap": self.chunk_overlap,
                "separators": self.separators,
                "library": "langchain_text_splitters.RecursiveCharacterTextSplitter",
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
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            except ImportError as exc:
                raise MissingDependencyError(
                    "The 'recursive' chunker needs langchain-text-splitters:\n"
                    "  pip install langchain-text-splitters"
                ) from exc

        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.encoder.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def _chunk(self, ctx: ChunkingContext) -> List[Chunk]:
        texts = [t for t in self.splitter.split_text(ctx.text) if t.strip()]
        if not texts:
            texts = [ctx.text]
        token_counts = self.encoder.count_tokens(texts)
        return chunks_from_texts(ctx, texts, token_counts)
