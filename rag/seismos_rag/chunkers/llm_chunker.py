"""Optional LLM chunker (Meta-Chunking margin sampling). Off by default.

For each adjacent sentence pair the LLM answers a yes/no boundary question,
read from the first answer token's logits. One forward pass per boundary, so
it is slow on long documents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..utils.logging import get_logger
from .base import (
    BaseChunker,
    Chunk,
    ChunkingContext,
    pack_sentence_ranges,
    ranges_to_chunks,
    whole_document_chunk,
)

PROMPT = (
    "You are segmenting a document into topically coherent parts.\n\n"
    "Passage A:\n{a}\n\n"
    "Passage B:\n{b}\n\n"
    "Does Passage B begin a new topic that should be separated from Passage A? "
    "Answer with exactly one word, Yes or No.\n"
    "Answer:"
)


class LLMChunker(BaseChunker):
    """Binary boundary decisions read from first-token logits."""

    name = "llm_chunker"

    def __init__(
        self,
        llm_runner,
        window_sentences: int = 8,
        min_sentences: int = 5,
        max_tokens: int = 480,
        apply_cap: bool = True,
        batch_size: int = 8,
    ) -> None:
        super().__init__(max_tokens=max_tokens, apply_cap=apply_cap)
        self.llm = llm_runner
        self.window_sentences = int(window_sentences)
        self.min_sentences = int(min_sentences)
        self.batch_size = int(batch_size)

    def params(self) -> Dict[str, Any]:
        params = super().params()
        params.update(
            {
                "window_sentences": self.window_sentences,
                "min_sentences": self.min_sentences,
                "decision_rule": "logit(Yes) > logit(No) on first answer token",
                "reference": "Meta-Chunking margin sampling (arXiv 2410.12788)",
                "model": getattr(self.llm, "model_name", None),
            }
        )
        return params

    def _chunk(self, ctx: ChunkingContext) -> List[Chunk]:
        if ctx.n_sentences == 0:
            return []
        if ctx.n_sentences < self.min_sentences:
            return whole_document_chunk(ctx)

        split_after = self._boundaries(ctx)
        ctx.diagnostics.setdefault("n_semantic_splits", {})[self.name] = len(split_after)

        cap = self.max_tokens if self.apply_cap else 10**9
        ranges = pack_sentence_ranges(ctx.sentence_token_counts, split_after, cap)
        return ranges_to_chunks(ctx, ranges)

    def _boundaries(self, ctx: ChunkingContext) -> Set[int]:
        """Ask about every adjacent sentence pair, windowing the context."""
        prompts: List[str] = []
        candidates: List[int] = []

        for i in range(ctx.n_sentences - 1):
            left_start = max(0, i - self.window_sentences + 1)
            passage_a = " ".join(ctx.sentences[left_start : i + 1])
            right_end = min(ctx.n_sentences, i + 1 + self.window_sentences)
            passage_b = " ".join(ctx.sentences[i + 1 : right_end])
            prompts.append(PROMPT.format(a=passage_a, b=passage_b))
            candidates.append(i)

        if not prompts:
            return set()

        decisions = self.llm.binary_choice(
            prompts, positive="Yes", negative="No", batch_size=self.batch_size
        )
        get_logger().debug(
            "[llm_chunker] %s: %d/%d boundaries", ctx.doc_id, sum(decisions), len(decisions)
        )
        return {i for i, decision in zip(candidates, decisions) if decision}
