from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Dict

from ..segment import clean
from . import RetrievalTask

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEV_DATASETS = ("arguana", "fiqa", "nfcorpus", "scifact")
HELD_OUT = ("trec-covid",)


def download(name: str, data_dir: str) -> Path:
    import requests

    target = Path(data_dir) / name
    if target.exists():
        return target
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(data_dir) / f"{name}.zip"
    with requests.get(BEIR_URL.format(name=name), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as fh:
            for block in r.iter_content(1 << 16):
                fh.write(block)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    os.remove(zip_path)
    return target


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_beir(name: str, data_dir: str = "data/beir", split: str = "test") -> RetrievalTask:
    """Document text is ``title + " " + text``; only judged queries are kept."""
    root = download(name, data_dir)

    qrels: Dict[str, Dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)

    docs = {r["_id"]: clean(f"{r.get('title', '')} {r.get('text', '')}", drop_nan=True)
            for r in _read_jsonl(root / "corpus.jsonl")}
    queries = {r["_id"]: clean(r["text"])
               for r in _read_jsonl(root / "queries.jsonl") if r["_id"] in qrels}
    return RetrievalTask(name=f"beir/{name}", docs=docs, queries=queries, qrels_doc=qrels)
