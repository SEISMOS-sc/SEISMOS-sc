"""Config-driven retrieval experiments (see ``configs/``).

Rows are appended to ``<output_dir>/results.csv`` as they finish; rows already
there are skipped, so a rerun resumes.
"""

from __future__ import annotations

import csv
import itertools
import logging
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Union

import numpy as np
import yaml
from tqdm.auto import tqdm

from .baselines import BASELINES, BaselineSpec, chunk_corpus_baseline
from .chunking import ChunkSpec, DocumentSignal, chunk_corpus, chunk_stats
from .datasets import RetrievalTask, load_task
from .encoders import Encoder, sanity_check
from .retrieval import bm25_search, chunk_qrels, chunk_results, dense_search, doc_results, evaluate
from .segment import ensure_punkt, split_sentences

logger = logging.getLogger(__name__)

KEY_FIELDS = ("benchmark", "subset", "encoder", "method", "level")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for required in ("name", "benchmark", "subsets", "encoders", "methods"):
        if required not in cfg:
            raise ValueError(f"{path}: missing required key {required!r}")
    cfg.setdefault("output_dir", os.path.join("results", cfg["name"]))
    cfg.setdefault("levels", ["doc"])
    cfg.setdefault("dataset_args", {})
    cfg.setdefault("encoder_args", {})
    cfg.setdefault("bm25", False)
    cfg.setdefault("ndcg_k", 10)
    return cfg


def expand_methods(entries: Iterable[dict]) -> List[Union[ChunkSpec, BaselineSpec]]:
    """List-valued parameters expand into one spec per value."""
    specs = []
    for entry in entries:
        cls = BaselineSpec if entry.get("method") in BASELINES else ChunkSpec
        sweep = {key: v for key, v in entry.items() if isinstance(v, list)}
        fixed = {key: v for key, v in entry.items() if key not in sweep}
        for combo in itertools.product(*sweep.values()) if sweep else [()]:
            specs.append(cls(**fixed, **dict(zip(sweep.keys(), combo))))
    return specs


def compute_signals(docs: Dict[str, str], encoder: Encoder, lang: str = "en",
                    block_sentences: int = 20_000, desc: str = "signals") -> Dict[str, DocumentSignal]:
    """Encodes sentences from many documents per batch; embeddings are not kept."""
    signals: Dict[str, DocumentSignal] = {}
    pending: List[tuple] = []
    n_pending = 0

    def flush():
        nonlocal pending, n_pending
        flat = [s for _, sents, _ in pending for s in sents]
        E = encoder.encode_passages(flat, normalize=True)
        counts = encoder.token_counts(flat)
        pos = 0
        for doc_id, sents, spans in pending:
            n = len(sents)
            e = E[pos:pos + n]
            X = np.sum(e[:-1] * e[1:], axis=1).astype(np.float32) if n >= 2 else None
            signals[doc_id] = DocumentSignal(sents, spans, X, counts[pos:pos + n])
            pos += n
        pending, n_pending = [], 0

    for doc_id, text in tqdm(docs.items(), desc=desc, unit="doc"):
        sents, spans = split_sentences(text, lang)
        pending.append((doc_id, sents, spans))
        n_pending += len(sents)
        if n_pending >= block_sentences:
            flush()
    if pending:
        flush()
    return signals


class ResultsFile:
    def __init__(self, path: str):
        self.path = path
        self.rows: List[dict] = []
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                self.rows = list(csv.DictReader(fh))

    def done(self, **key) -> bool:
        return any(all(str(r.get(f)) == str(key[f]) for f in KEY_FIELDS) for r in self.rows)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        fields = list(dict.fromkeys(f for r in self.rows for f in r))
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)


def _evaluate_levels(task: RetrievalTask, levels, q_ids, scores, ids, owner, doc_chunks, k):
    out = {}
    if "doc" in levels:
        if task.qrels_doc is None:
            raise ValueError(f"{task.name} has no document-level qrels")
        out["doc"] = evaluate(task.qrels_doc, doc_results(q_ids, scores, ids, owner), k,
                              ignore_identical_ids=True)
    if "chunk" in levels:
        if task.qrels_passage is None:
            raise ValueError(f"{task.name} has no passage-level qrels")
        qrels = chunk_qrels(doc_chunks, task.qrels_passage, task.pid2doc, task.pid2mid)
        out["chunk"] = evaluate(qrels, chunk_results(q_ids, scores, ids), k)
    return out


def run_experiment(cfg: dict) -> str:
    ensure_punkt()
    specs = expand_methods(cfg["methods"])
    results = ResultsFile(os.path.join(cfg["output_dir"], "results.csv"))
    k = cfg["ndcg_k"]
    levels = cfg["levels"]

    for subset in cfg["subsets"]:
        task = load_task(cfg["benchmark"], subset, **cfg["dataset_args"])
        logger.info(task.summary())
        q_ids = list(task.queries)
        q_texts = [task.queries[q] for q in q_ids]
        base = {"experiment": cfg["name"], "benchmark": cfg["benchmark"], "subset": subset,
                "n_docs": len(task.docs), "n_queries": len(q_ids)}

        if cfg["bm25"] and "doc" in levels and not results.done(**base, encoder="-", method="bm25", level="doc"):
            doc_ids = list(task.docs)
            scores, ids = bm25_search([task.docs[d] for d in doc_ids], q_texts)
            m = evaluate(task.qrels_doc, doc_results(q_ids, scores, ids, doc_ids), k, ignore_identical_ids=True)
            results.add({**base, "encoder": "-", "method": "bm25", "level": "doc", **m})

        for enc_name in cfg["encoders"]:
            todo = [s for s in specs
                    if not all(results.done(**base, encoder=enc_name, method=s.label, level=lv) for lv in levels)]
            if not todo:
                continue
            encoder = Encoder.from_name(enc_name, **cfg["encoder_args"])
            sanity_check(encoder, langs=sorted({"en", task.lang}))
            signals = None
            if any(isinstance(s, ChunkSpec) for s in todo):
                signals = compute_signals(task.docs, encoder, task.lang, desc=f"{subset}|{enc_name}|signals")
            q_embs = encoder.encode_queries(q_texts)

            for spec in todo:
                if isinstance(spec, BaselineSpec):
                    if levels != ["doc"]:
                        raise ValueError(f"{spec.label}: baseline chunkers have no character spans; "
                                         "use levels: [doc]")
                    texts, owner = chunk_corpus_baseline(task.docs, spec, encoder)
                    doc_chunks = {}
                else:
                    texts, owner, doc_chunks = chunk_corpus(signals, spec, task.lang)
                embs = encoder.encode_passages(texts, show_progress=True)
                scores, ids = dense_search(embs, q_embs)
                per_level = _evaluate_levels(task, levels, q_ids, scores, ids, owner, doc_chunks, k)
                stats = chunk_stats(texts, owner, len(task.docs))
                params = {f"param_{key}": v for key, v in asdict(spec).items()}
                for level, metrics in per_level.items():
                    results.add({**base, "encoder": enc_name, "method": spec.label, "level": level,
                                 **metrics, **stats, **params})
                    logger.info("%s | %s | %s | %s  nDCG@%d=%.4f", subset, enc_name, spec.label, level,
                                k, metrics[f"NDCG@{k}"])
            del encoder, signals
    return results.path
