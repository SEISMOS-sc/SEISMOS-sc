from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

Qrels = Dict[str, Dict[str, int]]


@dataclass
class RetrievalTask:
    """``pid2doc`` / ``pid2mid`` give each judged passage's document and character
    midpoint, so passage qrels can be mapped onto any chunking.
    """

    name: str
    docs: Dict[str, str]
    queries: Dict[str, str]
    qrels_doc: Optional[Qrels] = None
    qrels_passage: Optional[Qrels] = None
    pid2doc: Dict[str, str] = field(default_factory=dict)
    pid2mid: Dict[str, int] = field(default_factory=dict)
    lang: str = "en"

    @property
    def has_passage_qrels(self) -> bool:
        return self.qrels_passage is not None

    def summary(self) -> str:
        return f"{self.name}: {len(self.docs):,} docs, {len(self.queries):,} queries"


def doc_qrels_from_passages(qrels_passage: Qrels, pid2doc: Dict[str, str]) -> Qrels:
    out: Qrels = {}
    for q, golds in qrels_passage.items():
        rel: Dict[str, int] = {}
        for pid, score in golds.items():
            d = pid2doc.get(pid)
            if d is not None:
                rel[d] = max(rel.get(d, 0), int(score))
        out[q] = rel
    return out


def join_passages(passages, join: str = " "):
    """Returns ``(text, {passage_id: midpoint})``."""
    parts, mids, offset = [], {}, 0
    for pid, text in passages:
        mids[pid] = offset + len(text) // 2
        parts.append(text)
        offset += len(text) + len(join)
    return join.join(parts), mids


def load_task(benchmark: str, subset: str, **kwargs) -> RetrievalTask:
    if benchmark == "beir":
        from .beir import load_beir
        return load_beir(subset, **kwargs)
    if benchmark == "conteb":
        from .conteb import load_conteb
        return load_conteb(subset, **kwargs)
    if benchmark == "dapr":
        from .dapr import load_dapr_pooled
        return load_dapr_pooled(subset, **kwargs)
    if benchmark == "miracl":
        from .miracl import load_miracl_pooled
        return load_miracl_pooled(subset, **kwargs)
    raise ValueError(f"unknown benchmark {benchmark!r}")
