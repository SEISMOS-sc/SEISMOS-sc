"""MIRACL, pooled: every gold article plus distractors from MTEB's depth-250
candidate pool (``pool_source="mteb"``) or a uniform sample (``"random"``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import random
from typing import Dict, Iterable, Optional, Set

from ..segment import SENTENCE_JOIN, clean
from . import RetrievalTask, join_passages

logger = logging.getLogger(__name__)

MIRACL_REPO = "miracl/miracl"
CORPUS_REPO = "miracl/miracl-corpus"
MTEB_POOL_REPO = "mteb/MIRACLRetrieval_{lang}_top_250_only_w_correct-v2"
CORPUS_SHARDS = {"ar": 5, "de": 32, "ja": 14}
N_ARTICLES = {"ar": 656_982, "de": 2_651_352, "ja": 1_133_444}


def _hub(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=path, repo_type="dataset")


def _article(docid: str) -> str:
    return docid.split("#", 1)[0]


def _passage_no(docid: str) -> int:
    try:
        return int(docid.split("#", 1)[1])
    except (IndexError, ValueError):
        return 0


def _hash01(article: str, seed: int) -> float:
    """Stable per-article uniform value, so pools of different sizes are nested."""
    h = hashlib.blake2b(f"{seed}:{article}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2.0 ** 64


def _read_topics(lang: str, split: str) -> Dict[str, str]:
    path = _hub(MIRACL_REPO, f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{split}.tsv")
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                out[parts[0]] = clean(parts[1])
    return out


def _read_qrels(lang: str, split: str) -> Dict[str, Dict[str, int]]:
    path = _hub(MIRACL_REPO, f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv")
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 4:
                qid, docid, rel = p[0], p[2], p[3]
            elif len(p) == 3:
                qid, docid, rel = p
            else:
                continue
            if rel.lstrip("-").isdigit() and int(rel) > 0:
                qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def _mteb_articles(lang: str, cache_dir: str) -> Set[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files

    cache = os.path.join(cache_dir, f"{lang}__mteb_pool_articles.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return set(json.load(fh))
    repo = MTEB_POOL_REPO.format(lang=lang)
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                   if f.startswith("corpus/") and f.endswith(".parquet"))
    articles = set()
    for f in files:
        path = _hub(repo, f)
        cols = pq.ParquetFile(path).schema.names
        id_col = next(c for c in ("_id", "id", "docid", "corpus-id") if c in cols)
        articles.update(_article(str(d)) for d in pq.read_table(path, columns=[id_col])[id_col].to_pylist())
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(sorted(articles), fh)
    return articles


def _stream_corpus(lang: str) -> Iterable[dict]:
    for i in range(CORPUS_SHARDS[lang]):
        with gzip.open(_hub(CORPUS_REPO, f"miracl-corpus-v1.0-{lang}/docs-{i}.jsonl.gz"), "rt",
                       encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def _build_pool(lang: str, gold: Set[str], corpus_docs: int, pool_source: str, seed: int, path: str) -> None:
    n_dist = max(0, corpus_docs - len(gold))
    if pool_source == "mteb":
        candidates = _mteb_articles(lang, os.path.dirname(path)) - gold
        if len(candidates) < n_dist:
            raise RuntimeError(f"{lang}: MTEB pool has {len(candidates)} non-gold articles, need {n_dist}")
        chosen = set(sorted(candidates, key=lambda a: _hash01(a, seed))[:n_dist])
        keep = lambda a: a in chosen  # noqa: E731
    elif pool_source == "random":
        target = max(1.5 * n_dist, n_dist + 8.0 * math.sqrt(max(1, n_dist)))
        rate = min(1.0, target / N_ARTICLES[lang])
        keep = lambda a: _hash01(a, seed) < rate  # noqa: E731
    else:
        raise ValueError(f"pool_source must be 'mteb' or 'random', got {pool_source!r}")

    tmp = path + ".partial"
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for rec in _stream_corpus(lang):
            art = _article(rec.get("docid", ""))
            if art in gold or keep(art):
                out.write(json.dumps({k: rec.get(k, "") for k in ("docid", "title", "text")},
                                     ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_miracl_pooled(lang: str, split: str = "dev", n_queries: int = 200, corpus_docs: int = 30_000,
                       pool_source: str = "mteb", seed: int = 42, cache_dir: str = "data/miracl") -> RetrievalTask:
    if lang not in CORPUS_SHARDS:
        raise ValueError(f"unsupported MIRACL language {lang!r}; expected one of {sorted(CORPUS_SHARDS)}")

    topics, qrels_all = _read_topics(lang, split), _read_qrels(lang, split)
    judged = sorted(q for q in topics if qrels_all.get(q))
    random.Random(seed).shuffle(judged)
    qids = sorted(judged[:n_queries])
    queries = {q: topics[q] for q in qids}
    qrels_passage = {q: dict(qrels_all[q]) for q in qids}
    gold_articles = {_article(p) for g in qrels_passage.values() for p in g}
    if len(gold_articles) > corpus_docs:
        raise RuntimeError(f"{lang}: {len(gold_articles)} gold articles exceed corpus_docs={corpus_docs}")

    os.makedirs(cache_dir, exist_ok=True)
    pool_path = os.path.join(cache_dir, f"{lang}_{split}_{pool_source}{corpus_docs}_q{n_queries}_s{seed}__pool.jsonl.gz")
    if not os.path.exists(pool_path):
        _build_pool(lang, gold_articles, corpus_docs, pool_source, seed, pool_path)

    by_article: Dict[str, list] = {}
    with gzip.open(pool_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            body = clean(f"{r['title']} {r['text']}") if r.get("title") else clean(r.get("text", ""))
            by_article.setdefault(_article(r["docid"]), []).append((_passage_no(r["docid"]), r["docid"], body))

    if len(by_article) > corpus_docs:
        extra = sorted((a for a in by_article if a not in gold_articles), key=lambda a: _hash01(a, seed))
        for a in extra[corpus_docs - len(gold_articles):]:
            del by_article[a]
    if len(by_article) != corpus_docs:
        raise RuntimeError(f"{lang}: pool has {len(by_article)} articles, expected {corpus_docs}")

    join = SENTENCE_JOIN[lang]
    docs: Dict[str, str] = {}
    pid2doc: Dict[str, str] = {}
    pid2mid: Dict[str, int] = {}
    for art, parts in by_article.items():
        parts.sort(key=lambda x: x[0])
        docs[art], mids = join_passages([(pid, text) for _, pid, text in parts], join=join)
        pid2mid.update(mids)
        pid2doc.update({pid: art for pid in mids})

    missing = {p for g in qrels_passage.values() for p in g} - pid2mid.keys()
    if missing:
        raise RuntimeError(f"{lang}: {len(missing)} gold passages missing from the pool")
    return RetrievalTask(name=f"miracl/{lang}", docs=docs, queries=queries, qrels_passage=qrels_passage,
                         pid2doc=pid2doc, pid2mid=pid2mid, lang=lang)
