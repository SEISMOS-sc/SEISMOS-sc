"""LongBench NarrativeQA loading.

Returns ``[{id, question, context, answers}]``. Texts are never modified, and the
subsample depends only on (dataset, split, seed, num_samples).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils.caching import atomic_write_text, text_fingerprint
from ..utils.logging import get_logger
from ..utils.seeding import sample_indices

# LongBench field names. ``input`` is the question, ``context`` the long
# document, ``answers`` a list of reference strings, ``_id`` the stable sample id.
FIELD_QUESTION = "input"
FIELD_CONTEXT = "context"
FIELD_ANSWERS = "answers"
FIELD_ID = "_id"

REQUIRED_KEYS = ("id", "question", "context", "answers")


@dataclass
class DatasetStats:
    name: str
    split: str
    n_total: int
    n_selected: int
    seed: int
    mean_context_chars: float
    median_context_chars: float
    min_context_chars: int
    max_context_chars: int
    mean_answers_per_question: float
    source_repo: str

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class DatasetLoadError(RuntimeError):
    pass


def _resolve_repo(
    repos: Sequence[str], name: str, split: str, cache_dir: Optional[str]
):
    """LongBench is published as both ``THUDM/LongBench`` and ``zai-org/LongBench``; try each."""
    from datasets import load_dataset

    log = get_logger()
    errors: List[str] = []
    for repo in repos:
        try:
            log.info("loading %s (%s / %s)", repo, name, split)
            ds = load_dataset(repo, name, split=split, cache_dir=cache_dir)
            log.info("loaded %s: %d rows", repo, len(ds))
            return ds, repo
        except Exception as exc:  # noqa: BLE001 - report all failures together
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")
            log.warning("could not load %s: %s", repo, exc)

    raise DatasetLoadError(
        "none of the candidate LongBench repos could be loaded:\n  - "
        + "\n  - ".join(errors)
        + "\n\nIf this machine is offline, pre-download once with:\n"
        f"  python -c \"from datasets import load_dataset; "
        f"load_dataset('{repos[0]}', '{name}', split='{split}')\""
    )


def _coerce_answers(raw: Any) -> List[str]:
    """LongBench stores ``answers`` as a list of strings; be strict about it."""
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(a) for a in raw]
    raise DatasetLoadError(f"unexpected answers field type: {type(raw).__name__}")


def load_longbench_narrativeqa(
    num_samples: Optional[int] = 200,
    seed: int = 42,
    cache_dir: str = "cache/dataset",
    name: str = "narrativeqa",
    split: str = "test",
    hf_repos: Sequence[str] = ("THUDM/LongBench", "zai-org/LongBench"),
    verify_unmodified: bool = True,
) -> List[Dict[str, Any]]:
    """Load and subsample LongBench NarrativeQA.

    Returns a list of ``{"id", "question", "context", "answers"}`` dicts.
    """
    samples, _ = load_longbench_narrativeqa_with_stats(
        num_samples=num_samples,
        seed=seed,
        cache_dir=cache_dir,
        name=name,
        split=split,
        hf_repos=hf_repos,
        verify_unmodified=verify_unmodified,
    )
    return samples


def load_longbench_narrativeqa_with_stats(
    num_samples: Optional[int] = 200,
    seed: int = 42,
    cache_dir: str = "cache/dataset",
    name: str = "narrativeqa",
    split: str = "test",
    hf_repos: Sequence[str] = ("THUDM/LongBench", "zai-org/LongBench"),
    verify_unmodified: bool = True,
) -> tuple[List[Dict[str, Any]], DatasetStats]:
    """Same as :func:`load_longbench_narrativeqa` but also returns corpus stats."""
    import statistics

    log = get_logger()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    dataset, repo = _resolve_repo(hf_repos, name, split, cache_dir)
    n_total = len(dataset)

    columns = set(dataset.column_names)
    missing = {FIELD_QUESTION, FIELD_CONTEXT, FIELD_ANSWERS} - columns
    if missing:
        raise DatasetLoadError(
            f"{repo}/{name} is missing expected column(s) {sorted(missing)}; "
            f"available columns: {sorted(columns)}"
        )
    has_id = FIELD_ID in columns

    indices = sample_indices(n_total, num_samples, seed)
    log.info(
        "selected %d/%d samples (seed=%d, ascending index order)",
        len(indices),
        n_total,
        seed,
    )

    samples: List[Dict[str, Any]] = []
    for position, index in enumerate(indices):
        row = dataset[int(index)]
        qid = str(row[FIELD_ID]) if has_id else f"{name}-{index:05d}"
        sample = {
            # No .strip(), no normalisation: the context is the context.
            "id": qid,
            "question": row[FIELD_QUESTION],
            "context": row[FIELD_CONTEXT],
            "answers": _coerce_answers(row[FIELD_ANSWERS]),
            "source_index": int(index),
            "order": position,
            "context_sha256": text_fingerprint(row[FIELD_CONTEXT]),
        }
        if verify_unmodified:
            assert sample["question"] == row[FIELD_QUESTION]
            assert sample["context"] == row[FIELD_CONTEXT]
        samples.append(sample)

    if not samples:
        raise DatasetLoadError("no samples selected; check num_samples and the split")

    lengths = [len(s["context"]) for s in samples]
    stats = DatasetStats(
        name=name,
        split=split,
        n_total=n_total,
        n_selected=len(samples),
        seed=seed,
        mean_context_chars=float(statistics.mean(lengths)),
        median_context_chars=float(statistics.median(lengths)),
        min_context_chars=int(min(lengths)),
        max_context_chars=int(max(lengths)),
        mean_answers_per_question=float(
            statistics.mean(len(s["answers"]) for s in samples)
        ),
        source_repo=repo,
    )

    log.info(
        "dataset: %s/%s split=%s | %d selected of %d | context chars "
        "mean=%.0f median=%.0f min=%d max=%d | %.2f answers/question",
        repo,
        name,
        split,
        stats.n_selected,
        stats.n_total,
        stats.mean_context_chars,
        stats.median_context_chars,
        stats.min_context_chars,
        stats.max_context_chars,
        stats.mean_answers_per_question,
    )

    _warn_on_duplicate_ids(samples)
    return samples, stats


def _warn_on_duplicate_ids(samples: Sequence[Dict[str, Any]]) -> None:
    seen: Dict[str, int] = {}
    for sample in samples:
        seen[sample["id"]] = seen.get(sample["id"], 0) + 1
    duplicates = {k: v for k, v in seen.items() if v > 1}
    if duplicates:
        # qid is the checkpoint key; duplicates would silently collapse work units.
        raise DatasetLoadError(
            f"duplicate sample ids in selection: {sorted(duplicates)[:5]} "
            f"({len(duplicates)} total). Checkpoint keys would collide."
        )


def validate_sample(sample: Dict[str, Any]) -> None:
    """Raise if a sample does not satisfy the loader contract."""
    for key in REQUIRED_KEYS:
        if key not in sample:
            raise DatasetLoadError(f"sample missing required key {key!r}")
    if not isinstance(sample["id"], str) or not sample["id"]:
        raise DatasetLoadError("sample id must be a non-empty string")
    if not isinstance(sample["question"], str) or not sample["question"].strip():
        raise DatasetLoadError(f"sample {sample['id']}: question must be a non-empty string")
    if not isinstance(sample["context"], str) or not sample["context"].strip():
        raise DatasetLoadError(f"sample {sample['id']}: context must be a non-empty string")
    if not isinstance(sample["answers"], list) or not sample["answers"]:
        raise DatasetLoadError(f"sample {sample['id']}: answers must be a non-empty list")
    if not all(isinstance(a, str) for a in sample["answers"]):
        raise DatasetLoadError(f"sample {sample['id']}: all answers must be strings")


def save_manifest(
    samples: Sequence[Dict[str, Any]], stats: DatasetStats, path: str | Path
) -> None:
    """Write the exact selection to disk so a rerun can be audited or replayed."""
    payload = {
        "stats": stats.as_dict(),
        "samples": [
            {
                "id": s["id"],
                "source_index": s["source_index"],
                "order": s["order"],
                "context_sha256": s["context_sha256"],
                "context_chars": len(s["context"]),
                "n_answers": len(s["answers"]),
                "question": s["question"],
            }
            for s in samples
        ],
    }
    atomic_write_text(Path(path), json.dumps(payload, indent=2, ensure_ascii=False))
