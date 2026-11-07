"""Shared chunker contract, packing and span recovery.

Every chunker returns ``{chunk_id, text, token_count, start_position, end_position}``,
with half-open character offsets into the original document. Chunks whose span
cannot be recovered get ``-1`` offsets.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..utils.logging import get_logger

CHUNK_KEYS: Tuple[str, ...] = (
    "chunk_id",
    "text",
    "token_count",
    "start_position",
    "end_position",
)

Chunk = Dict[str, Any]


# -- context passed to every chunker ------------------------------------------


@dataclass
class ChunkingContext:
    """Per-document inputs shared by all chunkers (one sentence split, one encoder pass)."""

    doc_id: str
    text: str
    sentences: List[str]
    spans: List[Tuple[int, int]]
    sentence_token_counts: List[int]
    sentence_embeddings: Optional[np.ndarray] = None
    total_document_tokens: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    def adjacent_similarity(self) -> Optional[np.ndarray]:
        """``X[i] = e_i . e_{i+1}`` (embeddings are unit-norm); ``None`` below two sentences."""
        embeddings = self.sentence_embeddings
        if embeddings is None or len(embeddings) < 2:
            return None
        return np.sum(embeddings[:-1] * embeddings[1:], axis=1).astype(np.float64)


# -- base class ---------------------------------------------------------------


class BaseChunker(ABC):
    """Chunker interface. Subclasses implement :meth:`_chunk`."""

    #: short stable identifier used in every CSV and checkpoint key
    name: str = "base"

    def __init__(self, max_tokens: int = 480, apply_cap: bool = True) -> None:
        self.max_tokens = max_tokens
        self.apply_cap = apply_cap

    @abstractmethod
    def _chunk(self, ctx: ChunkingContext) -> List[Chunk]:
        """Produce chunks for one document."""

    def params(self) -> Dict[str, Any]:
        """Parameters recorded in ``config_used.json`` and the CSVs."""
        return {"max_tokens": self.max_tokens, "apply_cap": self.apply_cap}

    def chunk(self, ctx: ChunkingContext) -> List[Chunk]:
        chunks = self._chunk(ctx)
        chunks = renumber(chunks)
        validate_chunks(chunks, ctx.text, chunker=self.name)
        return chunks

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, params={self.params()})"


# -- construction helpers -----------------------------------------------------


def pack_sentence_ranges(
    token_counts: Sequence[int], split_after: Set[int], cap: int
) -> List[Tuple[int, int]]:
    """Inclusive sentence ranges; break after each ``split_after`` index or before
    exceeding ``cap``. Sentences are never split.
    """
    n = len(token_counts)
    if n == 0:
        return []

    ranges: List[Tuple[int, int]] = []
    start, current = 0, 0
    for i in range(n):
        tokens = token_counts[i]
        if i > start and current + tokens > cap:  # hard cap fallback
            ranges.append((start, i - 1))
            start, current = i, 0
        current += tokens
        if i in split_after and i < n - 1:  # semantic boundary after sentence i
            ranges.append((start, i))
            start, current = i + 1, 0
    ranges.append((start, n - 1))
    return ranges


def ranges_to_chunks(
    ctx: ChunkingContext, ranges: Sequence[Tuple[int, int]]
) -> List[Chunk]:
    """Chunk text is the document slice, so ``document[start:end] == text``."""
    chunks: List[Chunk] = []
    for chunk_id, (first, last) in enumerate(ranges):
        start = ctx.spans[first][0]
        end = ctx.spans[last][1]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": ctx.text[start:end],
                "token_count": int(sum(ctx.sentence_token_counts[first : last + 1])),
                "start_position": int(start),
                "end_position": int(end),
            }
        )
    return chunks


def whole_document_chunk(ctx: ChunkingContext) -> List[Chunk]:
    """Single chunk covering the document's sentence extent (gate fallback)."""
    if not ctx.spans:
        return []
    start, end = ctx.spans[0][0], ctx.spans[-1][1]
    return [
        {
            "chunk_id": 0,
            "text": ctx.text[start:end],
            "token_count": int(sum(ctx.sentence_token_counts)),
            "start_position": int(start),
            "end_position": int(end),
        }
    ]


def renumber(chunks: List[Chunk]) -> List[Chunk]:
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i
    return chunks


# -- span recovery for library chunkers ---------------------------------------

_WS = re.compile(r"\s+")


class _WhitespaceIndex:
    """Maps positions in a whitespace-stripped view back to original offsets."""

    def __init__(self, text: str) -> None:
        chars: List[str] = []
        positions: List[int] = []
        for i, ch in enumerate(text):
            if not ch.isspace():
                chars.append(ch)
                positions.append(i)
        self.stripped = "".join(chars)
        self.positions = positions

    def to_original(self, start: int, end: int) -> Tuple[int, int]:
        """``[start, end)`` in stripped space -> ``[a, b)`` in original space."""
        if start >= len(self.positions) or end <= start:
            return (-1, -1)
        a = self.positions[start]
        b = self.positions[min(end, len(self.positions)) - 1] + 1
        return (a, b)


def locate_spans(text: str, chunk_texts: Sequence[str]) -> List[Tuple[int, int]]:
    """Spans for third-party chunk strings: exact search, then whitespace-insensitive
    search forward, then from the start. Unresolved chunks get ``(-1, -1)``.
    """
    index = _WhitespaceIndex(text)
    spans: List[Tuple[int, int]] = []
    cursor = 0  # original-space cursor
    stripped_cursor = 0  # stripped-space cursor

    for chunk in chunk_texts:
        if not chunk:
            spans.append((-1, -1))
            continue

        found = text.find(chunk, cursor)
        if found != -1:
            span = (found, found + len(chunk))
            spans.append(span)
            cursor = span[1]
            stripped_cursor = _stripped_offset(index, span[1])
            continue

        needle = _WS.sub("", chunk)
        if not needle:
            spans.append((-1, -1))
            continue

        position = index.stripped.find(needle, stripped_cursor)
        if position == -1:
            position = index.stripped.find(needle)
        if position == -1:
            spans.append((-1, -1))
            continue

        span = index.to_original(position, position + len(needle))
        spans.append(span)
        cursor = max(cursor, span[1])
        stripped_cursor = position + len(needle)

    return spans


def _stripped_offset(index: _WhitespaceIndex, original_offset: int) -> int:
    """Number of non-whitespace chars before ``original_offset``."""
    import bisect

    return bisect.bisect_left(index.positions, original_offset)


def chunks_from_texts(
    ctx: ChunkingContext, chunk_texts: Sequence[str], token_counts: Sequence[int]
) -> List[Chunk]:
    """Build contract chunks from raw strings emitted by a library splitter."""
    spans = locate_spans(ctx.text, chunk_texts)
    unresolved = sum(1 for s in spans if s == (-1, -1))
    if unresolved:
        ctx.diagnostics.setdefault("unresolved_spans", 0)
        ctx.diagnostics["unresolved_spans"] += unresolved

    chunks: List[Chunk] = []
    for chunk_id, (chunk_text, tokens, (start, end)) in enumerate(
        zip(chunk_texts, token_counts, spans)
    ):
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": int(tokens),
                "start_position": int(start),
                "end_position": int(end),
            }
        )
    return chunks


# -- token cap enforcement for library chunkers -------------------------------


def enforce_token_cap(
    chunks: List[Chunk],
    cap: int,
    splitter,
    encoder,
) -> List[Chunk]:
    """Re-split chunks over ``cap`` by sentence and repack greedily."""
    if not chunks:
        return chunks

    out: List[Chunk] = []
    for chunk in chunks:
        if chunk["token_count"] <= cap:
            out.append(dict(chunk))
            continue

        split = splitter.split(chunk["text"])
        if len(split.sentences) < 2:
            out.append(dict(chunk))
            continue

        token_counts = encoder.count_tokens(split.sentences)
        ranges = pack_sentence_ranges(token_counts, set(), cap)
        base = chunk["start_position"]
        for first, last in ranges:
            sub_start = split.spans[first][0]
            sub_end = split.spans[last][1]
            out.append(
                {
                    "chunk_id": -1,  # renumbered by BaseChunker.chunk
                    "text": chunk["text"][sub_start:sub_end],
                    "token_count": int(sum(token_counts[first : last + 1])),
                    "start_position": (base + sub_start) if base >= 0 else -1,
                    "end_position": (base + sub_end) if base >= 0 else -1,
                }
            )
    return out


# -- validation ---------------------------------------------------------------


class ChunkFormatError(ValueError):
    pass


def validate_chunks(
    chunks: Sequence[Chunk], document: Optional[str] = None, chunker: str = "?"
) -> None:
    """Enforce the chunk output contract. Raises :class:`ChunkFormatError`."""
    if not isinstance(chunks, list):
        raise ChunkFormatError(f"[{chunker}] chunks must be a list")
    if not chunks:
        raise ChunkFormatError(f"[{chunker}] chunker produced zero chunks")

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ChunkFormatError(f"[{chunker}] chunk {position} is not a dict")
        missing = set(CHUNK_KEYS) - set(chunk)
        if missing:
            raise ChunkFormatError(
                f"[{chunker}] chunk {position} missing key(s): {sorted(missing)}"
            )
        if not isinstance(chunk["chunk_id"], int) or isinstance(chunk["chunk_id"], bool):
            raise ChunkFormatError(f"[{chunker}] chunk {position}: chunk_id must be int")
        if chunk["chunk_id"] != position:
            raise ChunkFormatError(
                f"[{chunker}] chunk_id must be the 0-based position; "
                f"got {chunk['chunk_id']} at position {position}"
            )
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ChunkFormatError(f"[{chunker}] chunk {position}: text must be non-empty")
        if not isinstance(chunk["token_count"], int) or chunk["token_count"] < 0:
            raise ChunkFormatError(
                f"[{chunker}] chunk {position}: token_count must be a non-negative int"
            )
        start, end = chunk["start_position"], chunk["end_position"]
        if not isinstance(start, int) or not isinstance(end, int):
            raise ChunkFormatError(f"[{chunker}] chunk {position}: positions must be ints")
        if (start == -1) != (end == -1):
            raise ChunkFormatError(
                f"[{chunker}] chunk {position}: unresolved spans must be (-1, -1)"
            )
        if start != -1:
            if end <= start:
                raise ChunkFormatError(
                    f"[{chunker}] chunk {position}: end_position must exceed start_position"
                )
            if document is not None and end > len(document):
                raise ChunkFormatError(
                    f"[{chunker}] chunk {position}: end_position {end} exceeds "
                    f"document length {len(document)}"
                )


def chunk_statistics(
    chunks: Sequence[Chunk], total_document_tokens: int
) -> Dict[str, Any]:
    """Per-document chunk statistics for ``chunk_statistics.csv``."""
    token_counts = [c["token_count"] for c in chunks]
    if not token_counts:
        return {
            "num_chunks": 0,
            "avg_chunk_tokens": 0.0,
            "max_chunk_tokens": 0,
            "min_chunk_tokens": 0,
            "median_chunk_tokens": 0.0,
            "p95_chunk_tokens": 0.0,
            "std_chunk_tokens": 0.0,
            "total_chunk_tokens": 0,
            "total_document_tokens": int(total_document_tokens),
            "unresolved_spans": 0,
        }
    array = np.asarray(token_counts, dtype=np.float64)
    return {
        "num_chunks": len(chunks),
        "avg_chunk_tokens": float(array.mean()),
        "max_chunk_tokens": int(array.max()),
        "min_chunk_tokens": int(array.min()),
        "median_chunk_tokens": float(np.median(array)),
        "p95_chunk_tokens": float(np.percentile(array, 95)),
        "std_chunk_tokens": float(array.std(ddof=0)),
        "total_chunk_tokens": int(array.sum()),
        "total_document_tokens": int(total_document_tokens),
        "unresolved_spans": sum(1 for c in chunks if c["start_position"] == -1),
    }


def log_chunker_summary(name: str, chunks: Sequence[Chunk], doc_id: str) -> None:
    log = get_logger()
    counts = [c["token_count"] for c in chunks]
    log.debug(
        "[%s] %s -> %d chunks (mean %.0f tok, max %d tok)",
        name,
        doc_id,
        len(chunks),
        float(np.mean(counts)) if counts else 0.0,
        int(max(counts)) if counts else 0,
    )
