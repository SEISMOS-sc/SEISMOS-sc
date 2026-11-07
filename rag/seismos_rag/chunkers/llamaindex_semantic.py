"""LlamaIndex ``SemanticSplitterNodeParser`` baseline, driven by the shared encoder."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseChunker, Chunk, ChunkingContext, chunks_from_texts, enforce_token_cap
from .recursive_chunker import MissingDependencyError


class LlamaIndexSemanticChunker(BaseChunker):
    """``llama_index.core.node_parser.SemanticSplitterNodeParser``."""

    name = "llamaindex_semantic"

    def __init__(
        self,
        encoder,
        sentence_splitter,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        max_tokens: int = 480,
        apply_cap: bool = True,
    ) -> None:
        super().__init__(max_tokens=max_tokens, apply_cap=apply_cap)
        self.encoder = encoder
        self.sentence_splitter = sentence_splitter
        self.buffer_size = int(buffer_size)
        self.breakpoint_percentile_threshold = int(breakpoint_percentile_threshold)
        self._parser = None

    def params(self) -> Dict[str, Any]:
        params = super().params()
        params.update(
            {
                "buffer_size": self.buffer_size,
                "breakpoint_percentile_threshold": self.breakpoint_percentile_threshold,
                "library": "llama_index.core.node_parser.SemanticSplitterNodeParser",
                "sentence_split": "library-internal splitter (not the shared spaCy split)",
            }
        )
        return params

    @property
    def parser(self):
        if self._parser is None:
            self._parser = self._build_parser()
        return self._parser

    def _build_parser(self):
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'llamaindex_semantic' chunker needs llama-index-core:\n"
                "  pip install llama-index-core"
            ) from exc

        from ..embeddings.encoder import LlamaIndexEmbedding

        return SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=LlamaIndexEmbedding(encoder=self.encoder),
        )

    def _chunk(self, ctx: ChunkingContext) -> List[Chunk]:
        from llama_index.core import Document

        document = Document(text=ctx.text, doc_id=ctx.doc_id)
        nodes = self.parser.get_nodes_from_documents([document], show_progress=False)

        texts = [n.get_content() for n in nodes if n.get_content().strip()]
        if not texts:
            texts = [ctx.text]
        token_counts = self.encoder.count_tokens(texts)
        chunks = chunks_from_texts(ctx, texts, token_counts)

        if self.apply_cap:
            chunks = enforce_token_cap(
                chunks, self.max_tokens, self.sentence_splitter, self.encoder
            )
        return chunks
