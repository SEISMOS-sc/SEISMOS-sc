"""Fixed baseline: the SEISMOS sentence packer with no semantic boundaries."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseChunker, Chunk, ChunkingContext, pack_sentence_ranges, ranges_to_chunks


class FixedChunker(BaseChunker):
    """Greedy sentence packing to a target token length."""

    name = "fixed"

    def __init__(self, target_tokens: int = 480, max_tokens: int = 480) -> None:
        # The cap and the target are the same knob for this method: packing stops
        # at the target, which is by definition the cap.
        super().__init__(max_tokens=max_tokens, apply_cap=True)
        self.target_tokens = int(target_tokens)

    def params(self) -> Dict[str, Any]:
        params = super().params()
        params.update(
            {"target_tokens": self.target_tokens, "preserve_sentence_boundaries": True}
        )
        return params

    def _chunk(self, ctx: ChunkingContext) -> List[Chunk]:
        if ctx.n_sentences == 0:
            return []
        ranges = pack_sentence_ranges(
            ctx.sentence_token_counts, split_after=set(), cap=self.target_tokens
        )
        return ranges_to_chunks(ctx, ranges)
