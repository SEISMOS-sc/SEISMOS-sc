"""Run logging and latency/GPU instrumentation."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "datasets",
    "fsspec",
    "sentence_transformers",
    "transformers.configuration_utils",
    "transformers.modeling_utils",
    "transformers.tokenization_utils_base",
)

_LOGGER_NAME = "seismos_rag"


def setup_logging(
    log_path: Optional[os.PathLike | str] = None,
    level: int = logging.INFO,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """Configure and return the pipeline logger (idempotent)."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Re-configuring in the same process (tests, notebooks) must not duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    if quiet_libraries:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def banner(text: str, char: str = "=", width: int = 78) -> None:
    log = get_logger()
    log.info(char * width)
    log.info(text)
    log.info(char * width)


# -- timing -------------------------------------------------------------------


@dataclass
class LatencyRecord:
    stage: str
    label: str
    seconds: float
    n_items: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def seconds_per_item(self) -> Optional[float]:
        if not self.n_items:
            return None
        return self.seconds / self.n_items

    def as_row(self) -> Dict[str, Any]:
        row = {
            "stage": self.stage,
            "label": self.label,
            "seconds": round(self.seconds, 6),
            "n_items": self.n_items,
            "seconds_per_item": (
                round(self.seconds_per_item, 6)
                if self.seconds_per_item is not None
                else None
            ),
        }
        row.update(self.extra)
        return row


class LatencyLog:
    """Collects :class:`LatencyRecord` s for export to ``latency_log.csv``."""

    def __init__(self) -> None:
        self._records: List[LatencyRecord] = []

    def add(self, record: LatencyRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> List[LatencyRecord]:
        return list(self._records)

    def rows(self) -> List[Dict[str, Any]]:
        return [r.as_row() for r in self._records]

    def total(self, stage: Optional[str] = None) -> float:
        return sum(
            r.seconds for r in self._records if stage is None or r.stage == stage
        )


@contextmanager
def timer(
    label: str,
    stage: str = "-",
    n_items: Optional[int] = None,
    sink: Optional[LatencyLog] = None,
    log_level: int = logging.INFO,
    **extra: Any,
) -> Iterator[Dict[str, Any]]:
    """Time a block. Set ``result["n_items"]`` inside it if known only afterwards."""
    result: Dict[str, Any] = {"n_items": n_items}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        count = result.get("n_items")
        record = LatencyRecord(
            stage=stage, label=label, seconds=elapsed, n_items=count, extra=dict(extra)
        )
        if sink is not None:
            sink.add(record)
        msg = f"[{stage}] {label}: {elapsed:.2f}s"
        if count:
            msg += f"  ({count} items, {elapsed / count * 1000:.1f} ms/item)"
        get_logger().log(log_level, msg)


class Timer:
    """Manual start/stop timer for loops where a context manager is awkward."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def reset(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def lap(self) -> float:
        value = self.elapsed()
        self.reset()
        return value
