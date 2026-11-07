"""Content-addressed caches (safe to delete) and append-only checkpoints keyed by
(qid, chunker), which are the resume mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from .logging import get_logger


# -- hashing ------------------------------------------------------------------


def stable_hash(*parts: Any, length: int = 16) -> str:
    """SHA-256 of canonical JSON (sorted keys)."""
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            payload = bytes(part)
        elif isinstance(part, str):
            payload = part.encode("utf-8")
        else:
            payload = json.dumps(
                part, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        hasher.update(len(payload).to_bytes(8, "little"))
        hasher.update(payload)
    return hasher.hexdigest()[:length]


def text_fingerprint(text: str) -> str:
    """Full-strength fingerprint of a document, used to detect content drift."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- atomic write -------------------------------------------------------------


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


# -- caches -------------------------------------------------------------------


class JsonCache:
    """One JSON file per key, sharded two levels deep to keep directories small."""

    def __init__(self, root: os.PathLike | str, namespace: str = "") -> None:
        self.root = Path(root) / namespace if namespace else Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                value = json.load(fh)
            self.hits += 1
            return value
        except (json.JSONDecodeError, OSError) as exc:
            # A truncated cache entry (killed mid-write on a filesystem without
            # atomic replace) must not poison the run.
            get_logger().warning("discarding corrupt cache entry %s: %s", path, exc)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        atomic_write_text(
            self._path(key), json.dumps(value, separators=(",", ":"), default=str)
        )

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.put(key, value)
        return value

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class NpyCache:
    """Content-addressed float array cache (``.npy``) for embeddings."""

    def __init__(self, root: os.PathLike | str, namespace: str = "") -> None:
        self.root = Path(root) / namespace if namespace else Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.npy"

    def get(self, key: str):
        import numpy as np

        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            array = np.load(path, allow_pickle=False)
            self.hits += 1
            return array
        except (ValueError, OSError) as exc:
            get_logger().warning("discarding corrupt embedding cache %s: %s", path, exc)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None

    def put(self, key: str, array) -> None:
        import numpy as np

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npy.tmp")
        os.close(fd)
        try:
            np.save(tmp, array, allow_pickle=False)
            # np.save appends .npy when the name lacks it; mkstemp gave us a
            # name that already ends in .tmp, so it wrote to tmp + ".npy".
            written = Path(tmp + ".npy") if Path(tmp + ".npy").exists() else Path(tmp)
            os.replace(written, path)
        finally:
            Path(tmp).unlink(missing_ok=True)
            Path(tmp + ".npy").unlink(missing_ok=True)

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# -- checkpoints --------------------------------------------------------------


class JsonlCheckpoint:
    """Append-only JSONL with a key index; a truncated last line is dropped on open."""

    def __init__(
        self,
        path: os.PathLike | str,
        key_fields: Tuple[str, ...],
        enabled: bool = True,
    ) -> None:
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        self.enabled = enabled
        self._done: Set[Tuple[Any, ...]] = set()
        self._handle = None
        self.resumed_count = 0

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load_existing()
            self._handle = self.path.open("a", encoding="utf-8")

    # -- lifecycle ------------------------------------------------------------

    def _load_existing(self) -> None:
        if not self.path.is_file():
            return
        good_lines: List[str] = []
        dropped = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                try:
                    self._done.add(self._key_of(record))
                except KeyError:
                    dropped += 1
                    continue
                good_lines.append(line)

        if dropped:
            # Rewrite without the damaged tail so subsequent appends stay parseable.
            get_logger().warning(
                "checkpoint %s: dropped %d unreadable record(s) during resume",
                self.path.name,
                dropped,
            )
            atomic_write_text(self.path, "\n".join(good_lines) + ("\n" if good_lines else ""))

        self.resumed_count = len(self._done)
        if self.resumed_count:
            get_logger().info(
                "checkpoint %s: resuming with %d completed record(s)",
                self.path.name,
                self.resumed_count,
            )

    def _key_of(self, record: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(record[field] for field in self.key_fields)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "JsonlCheckpoint":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- api ------------------------------------------------------------------

    def is_done(self, *key: Any) -> bool:
        if not self.enabled:
            return False
        return tuple(key) in self._done

    def append(self, record: Dict[str, Any]) -> None:
        key = self._key_of(record)
        self._done.add(key)
        if not self.enabled or self._handle is None:
            return
        self._handle.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
        # Per-record flush: a run killed by an OOM or a Colab timeout keeps
        # everything up to the last completed unit.
        self._handle.flush()

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def iter_all(self) -> Iterator[Dict[str, Any]]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def __len__(self) -> int:
        return len(self._done)


def pending_units(
    units: Iterable[Tuple[Any, ...]], checkpoint: JsonlCheckpoint
) -> List[Tuple[Any, ...]]:
    """Filter a work list down to the units this checkpoint has not completed."""
    return [unit for unit in units if not checkpoint.is_done(*unit)]
