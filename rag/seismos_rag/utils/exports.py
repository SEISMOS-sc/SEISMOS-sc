"""CSV/JSON export. Required columns are asserted and written first."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .caching import atomic_write_text
from .logging import get_logger

REQUIRED_COLUMNS: Dict[str, Sequence[str]] = {
    "summary_metrics.csv": (
        "chunker",
        "generator",
        "embedding_model",
        "num_questions",
        "qa_f1",
        "exact_match",
        "faithfulness",
        "avg_context_tokens",
        "avg_chunk_tokens",
    ),
    "per_question_results.csv": (
        "qid",
        "chunker",
        "question",
        "reference_answer",
        "generated_answer",
        "qa_f1",
        "exact_match",
        "faithfulness",
        "context_tokens",
    ),
    "retrieval_results.csv": (
        "qid",
        "chunker",
        "rank",
        "chunk_id",
        "score",
        "chunk_text",
    ),
    "chunk_statistics.csv": (
        "qid",
        "chunker",
        "num_chunks",
        "avg_chunk_tokens",
        "max_chunk_tokens",
        "total_document_tokens",
    ),
}


class ExportError(RuntimeError):
    pass


def _ordered_fieldnames(rows: Sequence[Dict[str, Any]], filename: str) -> List[str]:
    required = list(REQUIRED_COLUMNS.get(filename, ()))

    present: List[str] = []
    for row in rows:
        for key in row:
            if key not in present:
                present.append(key)

    missing = [c for c in required if c not in present]
    if missing:
        raise ExportError(
            f"{filename}: required column(s) missing from data: {missing}. "
            f"Present: {present}"
        )

    extras = [c for c in present if c not in required]
    return required + extras


def write_csv(
    path: Path | str,
    rows: Sequence[Dict[str, Any]],
    filename: Optional[str] = None,
) -> Path:
    """Write ``rows`` to ``path``, enforcing the required-column contract."""
    path = Path(path)
    name = filename or path.name
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Still emit the header so a downstream reader gets an empty table, not
        # a missing file it has to special-case.
        header = list(REQUIRED_COLUMNS.get(name, ()))
        if header:
            with path.open("w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=header).writeheader()
            get_logger().warning("%s: no rows; wrote header only", name)
        return path

    fieldnames = _ordered_fieldnames(rows, name)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fieldnames})

    get_logger().info("wrote %s (%d rows, %d columns)", name, len(rows), len(fieldnames))
    return path


def _cell(value: Any) -> Any:
    """``None`` -> empty cell (read back as NaN, not 0); lists/dicts -> JSON."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, str):
        # Excel/pandas both mangle embedded CRs; normalise to LF only.
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return value


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    get_logger().info("wrote %s", path.name)
    return path


def write_jsonl(path: Path | str, rows: Iterable[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    get_logger().info("wrote %s (%d records)", path.name, count)
    return path
