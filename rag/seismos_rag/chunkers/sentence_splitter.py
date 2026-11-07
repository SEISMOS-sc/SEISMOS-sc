"""spaCy sentence splitting with exact character spans, cached to disk.

Uses the ``senter`` component by default (much faster than ``parser`` on long
documents); set ``sentence_splitter.spacy_pipe: parser`` to change it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..utils.caching import JsonCache, stable_hash
from ..utils.logging import get_logger

# spaCy's max_length is below LongBench document lengths: split long documents
# into blocks at whitespace and re-offset the spans.
_BLOCK_CHARS = 400_000


@dataclass
class SentenceSplit:
    sentences: List[str]
    spans: List[Tuple[int, int]]

    def __post_init__(self) -> None:
        if len(self.sentences) != len(self.spans):
            raise ValueError("sentences and spans must have equal length")

    def __len__(self) -> int:
        return len(self.sentences)


class SentenceSplitter:
    """Cached spaCy sentence splitter producing exact ``(start, end)`` char spans."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        pipe: str = "senter",
        max_length: int = 4_000_000,
        cache_dir: Optional[str] = "cache/sentences",
    ) -> None:
        self.spacy_model = spacy_model
        self.pipe = pipe
        self.max_length = max_length
        self._nlp = None
        self._cache = (
            JsonCache(cache_dir, namespace=f"{spacy_model}-{pipe}") if cache_dir else None
        )

    # -- model ----------------------------------------------------------------

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = self._load_spacy()
        return self._nlp

    def _load_spacy(self):
        import spacy

        log = get_logger()
        try:
            nlp = spacy.load(self.spacy_model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {self.spacy_model!r} is not installed. Install it with:\n"
                f"  python -m spacy download {self.spacy_model}"
            ) from exc

        if self.pipe == "senter":
            # Disable everything that does not contribute sentence boundaries.
            for name in list(nlp.pipe_names):
                if name != "senter":
                    nlp.disable_pipe(name)
            if "senter" in nlp.component_names and "senter" not in nlp.pipe_names:
                nlp.enable_pipe("senter")
            if "senter" not in nlp.pipe_names:
                log.warning(
                    "%s has no 'senter' component; falling back to rule-based sentencizer",
                    self.spacy_model,
                )
                nlp.add_pipe("sentencizer")
        elif self.pipe == "parser":
            for name in list(nlp.pipe_names):
                if name not in ("tok2vec", "parser"):
                    nlp.disable_pipe(name)
            if "parser" not in nlp.pipe_names:
                raise RuntimeError(
                    f"{self.spacy_model} has no parser component; use spacy_pipe: senter"
                )
        else:
            raise ValueError(f"unknown spacy_pipe {self.pipe!r}; use 'senter' or 'parser'")

        nlp.max_length = self.max_length
        log.info(
            "spaCy loaded: %s | active pipes=%s | max_length=%d",
            self.spacy_model,
            nlp.pipe_names,
            nlp.max_length,
        )
        return nlp

    # -- splitting ------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return stable_hash(text, self.spacy_model, self.pipe, "v1", length=24)

    def split(self, text: str) -> SentenceSplit:
        """Segment ``text``; results are cached by (text, model, pipe)."""
        if not text:
            return SentenceSplit([], [])

        key = self._cache_key(text)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                spans = [(int(a), int(b)) for a, b in cached["spans"]]
                # Reconstruct sentence strings from the spans rather than storing
                # them twice: guarantees the cached spans and texts cannot drift.
                sentences = [text[a:b] for a, b in spans]
                return SentenceSplit(sentences, spans)

        split = self._split_uncached(text)
        if self._cache is not None:
            self._cache.put(key, {"spans": [list(s) for s in split.spans]})
        return split

    def _split_uncached(self, text: str) -> SentenceSplit:
        spans: List[Tuple[int, int]] = []
        for block_start, block_text in _iter_blocks(text, _BLOCK_CHARS):
            doc = self.nlp(block_text)
            for sent in doc.sents:
                start = block_start + sent.start_char
                end = block_start + sent.end_char
                # spaCy sentence spans include trailing whitespace; trim it so
                # that spans reported downstream point at the sentence itself.
                while end > start and text[end - 1].isspace():
                    end -= 1
                while start < end and text[start].isspace():
                    start += 1
                if end > start:
                    spans.append((start, end))

        spans = _merge_degenerate(spans)
        sentences = [text[a:b] for a, b in spans]
        return SentenceSplit(sentences, spans)

    def split_many(self, texts: Sequence[str]) -> List[SentenceSplit]:
        return [self.split(t) for t in texts]

    def cache_stats(self) -> Dict[str, int]:
        return self._cache.stats() if self._cache else {"hits": 0, "misses": 0}


def _iter_blocks(text: str, block_chars: int):
    """Yield ``(offset, block)`` pairs cut at whitespace, covering ``text``."""
    if len(text) <= block_chars:
        yield 0, text
        return

    position = 0
    total = len(text)
    while position < total:
        end = min(position + block_chars, total)
        if end < total:
            # Back off to the last whitespace so a sentence is not severed.
            window_start = max(position + 1, end - 4096)
            cut = text.rfind(" ", window_start, end)
            if cut == -1:
                cut = text.rfind("\n", window_start, end)
            if cut != -1:
                end = cut
        yield position, text[position:end]
        position = end


def _merge_degenerate(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Drop zero-length spans and enforce strictly increasing, non-overlapping spans."""
    out: List[Tuple[int, int]] = []
    for start, end in spans:
        if end <= start:
            continue
        if out and start < out[-1][1]:
            # Overlap can only arise from a block boundary landing inside a
            # sentence; extend the previous span instead of emitting both.
            prev_start, prev_end = out[-1]
            out[-1] = (prev_start, max(prev_end, end))
            continue
        out.append((start, end))
    return out


def validate_spans(text: str, split: SentenceSplit) -> None:
    """Assert that every span indexes back to its sentence exactly."""
    for i, ((start, end), sentence) in enumerate(zip(split.spans, split.sentences)):
        if text[start:end] != sentence:
            raise AssertionError(
                f"span {i} ({start},{end}) does not reproduce its sentence"
            )
        if i and split.spans[i - 1][1] > start:
            raise AssertionError(f"span {i} overlaps span {i - 1}")
