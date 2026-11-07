"""Sentence splitting with exact character spans. Punkt for en/de, regexes for ja/ar."""

from __future__ import annotations

import re
from typing import List, Tuple

Spans = List[Tuple[int, int]]

_WS = re.compile(r"\s+")
_NAN = re.compile(r"\b(nan|NaN|NAN)\b")

_RE_JA = re.compile(r"[^。！？!?]*?[。！？!?]+[」』）】\)\]”\"']*|[^。！？!?]+")
_RE_AR = re.compile(r"[^.!?؟؛…\n]*?[.!?؟؛…]+[»”\"'\)\]]*|[^.!?؟؛…\n]+")
_REGEX_LANGS = {"ja": _RE_JA, "ar": _RE_AR}
_PUNKT_LANGS = {"en": "english", "de": "german"}

SENTENCE_JOIN = {"ja": "", "en": " ", "de": " ", "ar": " "}


def clean(text: str, drop_nan: bool = False) -> str:
    """Collapse whitespace; ``drop_nan`` strips literal "nan" tokens (BEIR only)."""
    text = str(text)
    if drop_nan:
        text = _NAN.sub("", text)
    return _WS.sub(" ", text).strip()


def ensure_punkt() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _punkt(text: str, language: str) -> Tuple[List[str], Spans]:
    import nltk

    sents = nltk.tokenize.sent_tokenize(text, language=language)
    spans, pos = [], 0
    for s in sents:
        idx = text.find(s, pos)
        if idx < 0:
            idx = pos
        spans.append((idx, idx + len(s)))
        pos = idx + len(s)
    return sents, spans


def _merge_decimals(sents: List[str], spans: Spans, text: str) -> Tuple[List[str], Spans]:
    out_s: List[str] = []
    out_sp: Spans = []
    for s, (a, b) in zip(sents, spans):
        prev = out_s[-1] if out_s else ""
        if len(prev) >= 2 and prev.endswith(".") and prev[-2].isdigit() and s[:1].isdigit():
            a0 = out_sp[-1][0]
            out_s[-1], out_sp[-1] = text[a0:b], (a0, b)
        else:
            out_s.append(s)
            out_sp.append((a, b))
    return out_s, out_sp


def _regex(text: str, rx: re.Pattern) -> Tuple[List[str], Spans]:
    sents, spans = [], []
    for m in rx.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            sents.append(text[s:e])
            spans.append((s, e))
    return _merge_decimals(sents, spans, text)


def split_sentences(text: str, lang: str = "en") -> Tuple[List[str], Spans]:
    rx = _REGEX_LANGS.get(lang)
    if rx is not None:
        return _regex(text, rx)
    if lang not in _PUNKT_LANGS:
        raise ValueError(f"unsupported language {lang!r}; expected one of "
                         f"{sorted(_PUNKT_LANGS) + sorted(_REGEX_LANGS)}")
    return _punkt(text, _PUNKT_LANGS[lang])
