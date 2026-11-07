"""Search and scoring.

Document level: chunk hits are folded to documents by max score.
Chunk level: each gold passage is mapped, by its character midpoint, to the
chunk containing it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Results = Dict[str, Dict[str, float]]
Qrels = Dict[str, Dict[str, int]]


def dense_search(doc_embs: np.ndarray, query_embs: np.ndarray, topn: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    import faiss

    doc_embs = np.ascontiguousarray(doc_embs, dtype=np.float32)
    query_embs = np.ascontiguousarray(query_embs, dtype=np.float32)
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs)
    return index.search(query_embs, min(topn, doc_embs.shape[0]))


def bm25_search(texts: Sequence[str], queries: Sequence[str], topn: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([t.lower().split() for t in texts])
    topn = min(topn, len(texts))
    scores, ids = [], []
    for q in queries:
        s = bm25.get_scores(q.lower().split())
        top = np.argsort(-s)[:topn]
        ids.append(top)
        scores.append(s[top])
    return np.array(scores), np.array(ids)


def chunk_results(q_ids: Sequence[str], scores: np.ndarray, ids: np.ndarray) -> Results:
    """Chunk ids are prefixed ("c<j>") so they never collide with numeric query ids."""
    out: Results = {}
    for qi, q in enumerate(q_ids):
        out[q] = {f"c{int(j)}": float(s) for j, s in zip(ids[qi], scores[qi]) if j >= 0}
    return out


def doc_results(q_ids: Sequence[str], scores: np.ndarray, ids: np.ndarray, owner: Sequence[str]) -> Results:
    out: Results = {}
    for qi, q in enumerate(q_ids):
        best: Dict[str, float] = {}
        for j, s in zip(ids[qi], scores[qi]):
            if j < 0:
                continue
            d = owner[int(j)]
            if d not in best or s > best[d]:
                best[d] = float(s)
        out[q] = best
    return out


def locate_chunk(doc_chunks: List[Tuple[int, int, int]], midpoint: int) -> Optional[int]:
    """Chunk containing ``midpoint``, else the last chunk starting before it."""
    best = None
    for start, end, j in doc_chunks:
        if start <= midpoint < end:
            return j
        if start <= midpoint:
            best = j
    if best is None and doc_chunks:
        best = doc_chunks[0][2]
    return best


def chunk_qrels(doc_chunks: Dict[str, List[Tuple[int, int, int]]], qrels_passage: Qrels,
                pid2doc: Dict[str, str], pid2mid: Dict[str, int]) -> Qrels:
    out: Qrels = {}
    for q, golds in qrels_passage.items():
        rel: Dict[str, int] = {}
        for pid, score in golds.items():
            d = pid2doc.get(pid)
            if d is None or d not in doc_chunks:
                continue
            j = locate_chunk(doc_chunks[d], pid2mid[pid])
            if j is not None:
                key = f"c{j}"
                rel[key] = max(rel.get(key, 0), int(score))
        out[q] = rel
    return out


def evaluate(qrels: Qrels, results: Results, k: int = 10, ignore_identical_ids: bool = False) -> Dict[str, float]:
    """nDCG, Recall and MRR at ``k``. ``ignore_identical_ids`` mirrors BEIR's default (matters for ArguAna)."""
    import pytrec_eval

    if ignore_identical_ids:
        results = {q: {d: s for d, s in hits.items() if d != q} for q, hits in results.items()}
    measures = {f"ndcg_cut.{k}", f"recall.{k}"}
    per_query = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(results)
    n = max(1, len(per_query))
    out = {
        f"NDCG@{k}": sum(v[f"ndcg_cut_{k}"] for v in per_query.values()) / n,
        f"Recall@{k}": sum(v[f"recall_{k}"] for v in per_query.values()) / n,
    }
    rr = []
    for q in per_query:
        gold = {d for d, r in qrels.get(q, {}).items() if r > 0}
        ranked = sorted(results.get(q, {}).items(), key=lambda x: -x[1])[:k]
        rr.append(next((1.0 / (i + 1) for i, (d, _) in enumerate(ranked) if d in gold), 0.0))
    out[f"MRR@{k}"] = float(np.mean(rr)) if rr else 0.0
    return out
