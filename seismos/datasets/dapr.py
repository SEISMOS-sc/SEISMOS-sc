"""DAPR, pooled: ``corpus_docs`` documents per dataset, every gold document
included. Not comparable to DAPR's published full-corpus numbers.
"""

from __future__ import annotations

import bisect
import logging
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple

from ..segment import clean
from . import RetrievalTask, doc_qrels_from_passages, join_passages

logger = logging.getLogger(__name__)

HF_REPO = "datasets/UKPLab/dapr"
HF_REV = "refs%2Fconvert%2Fparquet"
DATASETS = ("ConditionalQA", "Genomics", "NaturalQuestions", "MSMARCO")


class _Parquet:
    def __init__(self):
        from huggingface_hub import HfFileSystem

        self.fs = HfFileSystem()

    def shards(self, config: str, split: str = "test") -> List[str]:
        return sorted(self.fs.ls(f"{HF_REPO}@{HF_REV}/{config}/{split}", detail=False))

    def read_all(self, config: str) -> List[dict]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        tables = [pq.read_table(self.fs.open(f, "rb")) for f in self.shards(config)]
        return pa.concat_tables(tables).to_pylist()


def _build_doc_index(pqt: _Parquet, dataset: str, cache_dir: str) -> dict:
    import pyarrow.parquet as pq

    cache = os.path.join(cache_dir, f"{dataset}__docindex.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    shards = pqt.shards(f"{dataset}-corpus")

    def scan(item):
        si, path = item
        with pqt.fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh, pre_buffer=True)
            starts, off = [], 0
            for i in range(pf.metadata.num_row_groups):
                starts.append(off)
                off += pf.metadata.row_group(i).num_rows
            ids = pf.read(columns=["doc_id"])["doc_id"].to_pylist()
        runs, b = [], 0
        for i in range(1, len(ids) + 1):
            if i == len(ids) or ids[i] != ids[b]:
                runs.append((ids[b], si, b, i - 1))
                b = i
        return si, starts, runs

    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = sorted(ex.map(scan, enumerate(shards)), key=lambda p: p[0])

    runs: Dict[str, list] = {}
    rg_starts = []
    for _, starts, rr in parts:
        rg_starts.append(starts)
        for doc, s, a, b in rr:
            runs.setdefault(doc, []).append((s, a, b))
    idx = {"runs": runs, "rg_starts": rg_starts, "shards": shards}
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(idx, fh, protocol=4)
    return idx


def _cells(idx: dict, doc: str) -> Set[Tuple[int, int]]:
    out = set()
    for s, a, b in idx["runs"][doc]:
        starts = idx["rg_starts"][s]
        lo = bisect.bisect_right(starts, a) - 1
        hi = bisect.bisect_right(starts, b) - 1
        out.update((s, r) for r in range(lo, hi + 1))
    return out


def _fetch(pqt: _Parquet, idx: dict, cells: Set[Tuple[int, int]], keep: Set[str]) -> List[dict]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_shard: Dict[int, List[int]] = {}
    for s, r in cells:
        by_shard.setdefault(s, []).append(r)
    rows = []
    for s in sorted(by_shard):
        with pqt.fs.open(idx["shards"][s], "rb") as fh:
            pf = pq.ParquetFile(fh, pre_buffer=True)
            for r in sorted(by_shard[s]):
                t = pf.read_row_group(r, columns=["_id", "doc_id", "paragraph_no", "text", "total_paragraphs"])
                wanted = [i for i, d in enumerate(t.column("doc_id").to_pylist()) if d in keep]
                if wanted:
                    rows.extend(t.take(pa.array(wanted)).to_pylist())
    return rows


def load_dapr_pooled(dataset: str, corpus_docs: int = 200, n_queries: int = 500, gold_frac: float = 0.8,
                     seed: int = 42, cache_dir: str = "data/dapr") -> RetrievalTask:
    if dataset not in DATASETS:
        raise ValueError(f"unknown DAPR dataset {dataset!r}; expected one of {DATASETS}")
    pqt = _Parquet()

    queries_all = {r["_id"]: clean(r["text"]) for r in pqt.read_all(f"{dataset}-queries")}
    qrels_all: Dict[str, Dict[str, int]] = {}
    for r in pqt.read_all(f"{dataset}-qrels"):
        qrels_all.setdefault(r["query_id"], {})[r["corpus_id"]] = int(r["score"])

    # Passage ids are "<doc_id>-<paragraph_no>".
    judged = sorted(q for q in queries_all if q in qrels_all)
    random.Random(seed).shuffle(judged)
    budget = int(corpus_docs * gold_frac)
    chosen, gold_docs = [], set()
    for q in judged[:n_queries]:
        merged = gold_docs | {p.rsplit("-", 1)[0] for p in qrels_all[q]}
        if len(merged) <= budget:
            chosen.append(q)
            gold_docs = merged
    if not chosen:
        raise RuntimeError(f"{dataset}: no query fits a gold budget of {budget} documents")
    qids = sorted(chosen)
    qrels_passage = {q: qrels_all[q] for q in qids}

    idx = _build_doc_index(pqt, dataset, cache_dir)
    missing = gold_docs - idx["runs"].keys()
    if missing:
        raise RuntimeError(f"{dataset}: {len(missing)} gold documents absent from the corpus")

    cells: Set[Tuple[int, int]] = set()
    for d in gold_docs:
        cells |= _cells(idx, d)
    frame = sorted(d for d in idx["runs"] if d not in gold_docs and _cells(idx, d) <= cells)
    n_distractors = corpus_docs - len(gold_docs)
    if len(frame) < n_distractors:
        raise RuntimeError(f"{dataset}: distractor frame has {len(frame)} docs, need {n_distractors}")
    keep = gold_docs | set(random.Random(seed).sample(frame, n_distractors))

    by_doc: Dict[str, list] = {}
    for r in _fetch(pqt, idx, cells, keep):
        by_doc.setdefault(r["doc_id"], []).append(
            (r.get("paragraph_no", 0), r["_id"], clean(r["text"]), r.get("total_paragraphs")))

    docs: Dict[str, str] = {}
    pid2doc: Dict[str, str] = {}
    pid2mid: Dict[str, int] = {}
    for doc, parts in by_doc.items():
        parts.sort(key=lambda x: x[0])
        total = parts[0][3]
        if total is not None and len(parts) != total:
            raise RuntimeError(f"{dataset}: document {doc} truncated ({len(parts)} of {total} passages)")
        docs[doc], mids = join_passages([(pid, text) for _, pid, text, _ in parts])
        pid2mid.update(mids)
        pid2doc.update({pid: doc for pid in mids})

    gold_pids = {p for g in qrels_passage.values() for p in g}
    if len(docs) != corpus_docs or not gold_pids <= pid2mid.keys():
        raise RuntimeError(f"{dataset}: pool is incomplete ({len(docs)} docs; "
                           f"{len(gold_pids - pid2mid.keys())} gold passages missing)")
    logger.info("%s: %d queries, %d gold + %d distractor docs", dataset, len(qids), len(gold_docs), n_distractors)

    return RetrievalTask(name=f"dapr/{dataset}", docs=docs, queries={q: queries_all[q] for q in qids},
                         qrels_doc=doc_qrels_from_passages(qrels_passage, pid2doc),
                         qrels_passage=qrels_passage, pid2doc=pid2doc, pid2mid=pid2mid)
