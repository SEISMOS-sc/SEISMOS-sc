"""ConTEB. Documents are rebuilt from the released passages (``<doc>_<index>``)."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..segment import clean
from . import RetrievalTask, join_passages

SUBSETS = {
    "narrativeqa": ("illuin-conteb/narrative-qa", "test", "queries", "test"),
    "geography": ("illuin-conteb/geography", "train", "queries-filtered", "train"),
    "esg": ("illuin-conteb/esg-reports", "test", "queries", "test"),
}


def _first(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def load_conteb(subset: str, n_queries: Optional[int] = None, seed: int = 42) -> RetrievalTask:
    from datasets import load_dataset

    if subset not in SUBSETS:
        raise ValueError(f"unknown ConTEB subset {subset!r}; expected one of {sorted(SUBSETS)}")
    repo, doc_split, q_config, q_split = SUBSETS[subset]
    docs_ds = load_dataset(repo, "documents", split=doc_split)
    q_ds = load_dataset(repo, q_config, split=q_split)

    # "og_query" is the original, non-decontextualised question; not used.
    q_key = _first(q_ds.column_names, ("query", "question"))
    g_key = _first(q_ds.column_names, ("chunk_id", "chunk_ids", "gold_chunk_id"))
    if q_key is None or g_key is None:
        raise ValueError(f"unexpected ConTEB query columns: {q_ds.column_names}")

    by_doc: Dict[str, List] = {}
    for r in docs_ds:
        cid = r["chunk_id"]
        try:
            doc, idx = cid.rsplit("_", 1)
            idx = int(idx)
        except ValueError:
            doc, idx = cid, 0
        by_doc.setdefault(doc, []).append((idx, cid, clean(r["chunk"])))

    docs: Dict[str, str] = {}
    pid2doc: Dict[str, str] = {}
    pid2mid: Dict[str, int] = {}
    for doc, parts in by_doc.items():
        parts.sort(key=lambda x: x[0])
        docs[doc], mids = join_passages([(cid, text) for _, cid, text in parts])
        pid2mid.update(mids)
        pid2doc.update({cid: doc for cid in mids})

    def golds(row) -> List[str]:
        v = row.get(g_key)
        if v is None:
            return []
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]

    rows = [r for r in q_ds if clean(r.get(q_key) or "") and any(g in pid2mid for g in golds(r))]
    if n_queries is not None and len(rows) > n_queries:
        random.Random(seed).shuffle(rows)
        rows = rows[:n_queries]

    queries, qrels = {}, {}
    for i, r in enumerate(rows):
        qid = f"q{i}"
        queries[qid] = clean(r[q_key])
        qrels[qid] = {g: 1 for g in golds(r) if g in pid2mid}

    return RetrievalTask(name=f"conteb/{subset}", docs=docs, queries=queries,
                         qrels_passage=qrels, pid2doc=pid2doc, pid2mid=pid2mid)
