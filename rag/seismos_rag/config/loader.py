"""Config loading, CLI overrides and provenance capture."""

from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

# CLI flag -> dotted config path.
CLI_OVERRIDE_MAP: Dict[str, str] = {
    "dataset": "dataset.name",
    "num_samples": "dataset.num_samples",
    "seed": "seed",
    "embedding": "embedding.model",
    "generator": "generation.model",
    "top_k": "retrieval.top_k",
    "max_tokens": "chunking.max_tokens",
    "seismos_k": "chunking.seismos.k",
    "run_k_sweep": "chunking.seismos.run_k_sweep",
    "quantization": "generation.quantization",
    "gen_batch_size": "generation.batch_size",
    "max_new_tokens": "generation.max_new_tokens",
    "max_input_tokens": "generation.max_input_tokens",
    "attn": "generation.attn_implementation",
    "preflight": "generation.preflight",
    "output_dir": "output.dir",
    "device": "runtime.device",
    "faithfulness": "faithfulness.enabled",
    "llm_chunker": "chunking.llm_chunker.enabled",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Nested config with dotted-path access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    # -- access ---------------------------------------------------------------
    def __getitem__(self, path: str) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"config path not found: {path!r}")
            node = node[part]
        return node

    def get(self, path: str, default: Any = None) -> Any:
        try:
            return self[path]
        except KeyError:
            return default

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def section(self, path: str) -> Dict[str, Any]:
        value = self[path]
        if not isinstance(value, dict):
            raise TypeError(f"config path {path!r} is not a section")
        return value

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({json.dumps(self._data, indent=2, default=str)})"


def load_config(
    config_path: Optional[os.PathLike | str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load ``config.yaml`` and apply CLI overrides; ``None`` overrides are ignored."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")

    cfg = Config(data)

    for dest, value in (cli_overrides or {}).items():
        if value is None:
            continue
        if dest not in CLI_OVERRIDE_MAP:
            continue
        cfg.set(CLI_OVERRIDE_MAP[dest], value)

    # --cache_dir moves all three caches.
    cache_root = (cli_overrides or {}).get("cache_dir")
    if cache_root:
        root = Path(cache_root)
        cfg.set("dataset.cache_dir", str(root / "dataset"))
        cfg.set("sentence_splitter.cache_dir", str(root / "sentences"))
        cfg.set("embedding.cache_dir", str(root / "embeddings"))

    cfg.set("_provenance.config_path", str(path.resolve()))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Fail fast on configurations that would produce meaningless results."""
    problems = []

    if cfg["retrieval.top_k"] < 1:
        problems.append("retrieval.top_k must be >= 1")
    if cfg["dataset.num_samples"] is not None and cfg["dataset.num_samples"] < 1:
        problems.append("dataset.num_samples must be >= 1 or null (all)")
    if cfg["chunking.max_tokens"] < 32:
        problems.append("chunking.max_tokens must be >= 32")
    if cfg["chunking.seismos.min_sentences"] < 4:
        # std(D) is estimated from n-3 values; n<4 leaves nothing to estimate from.
        problems.append("chunking.seismos.min_sentences must be >= 4")
    if not cfg["embedding.normalize"]:
        problems.append(
            "embedding.normalize must be true: IndexFlatIP only equals cosine "
            "similarity on unit-norm vectors"
        )
    if cfg["retrieval.index_type"] != "IndexFlatIP":
        problems.append("retrieval.index_type must be IndexFlatIP (exact search)")
    if float(cfg["generation.temperature"]) != 0.0:
        problems.append(
            "generation.temperature must be 0.0; the protocol specifies greedy decoding"
        )
    if cfg["generation.quantization"] not in ("none", "4bit", "8bit"):
        problems.append("generation.quantization must be one of none|4bit|8bit")
    attn = cfg.get("generation.attn_implementation", "auto")
    if attn not in ("auto", "sdpa", "eager", "flash_attention_2"):
        problems.append(
            "generation.attn_implementation must be one of "
            "auto|sdpa|eager|flash_attention_2"
        )
    if cfg["generation.batch_size"] < 1:
        problems.append("generation.batch_size must be >= 1")
    if cfg["generation.max_input_tokens"] < 128:
        problems.append("generation.max_input_tokens must be >= 128")
    if cfg["statistics.confidence_level"] <= 0 or cfg["statistics.confidence_level"] >= 1:
        problems.append("statistics.confidence_level must be in (0, 1)")

    k = cfg["chunking.seismos.k"]
    if not isinstance(k, (int, float)) or k <= 0:
        problems.append("chunking.seismos.k must be a positive number")

    if problems:
        raise ValueError("invalid configuration:\n  - " + "\n  - ".join(problems))


# -- provenance ---------------------------------------------------------------


def _package_versions(packages: Iterable[str]) -> Dict[str, Optional[str]]:
    from importlib import metadata

    out: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def _git_revision(repo_root: Path) -> Optional[str]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if rev.returncode == 0:
            return rev.stdout.strip()
    except Exception:
        pass
    return None


PROVENANCE_PACKAGES = (
    "torch",
    "transformers",
    "sentence-transformers",
    "faiss-cpu",
    "faiss-gpu",
    "datasets",
    "spacy",
    "numpy",
    "scipy",
    "pandas",
    "langchain-experimental",
    "langchain-text-splitters",
    "llama-index-core",
    "accelerate",
    "bitsandbytes",
)


@dataclass
class RunProvenance:
    """Everything needed to know what produced a results directory."""

    config: Dict[str, Any]
    resolved: Dict[str, Any]

    def to_json(self, path: os.PathLike | str) -> None:
        payload = {"config": self.config, "resolved": self.resolved}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str, sort_keys=False)


def build_provenance(cfg: Config, extra: Optional[Dict[str, Any]] = None) -> RunProvenance:
    """Snapshot config + environment for ``config_used.json``."""
    import datetime as _dt

    repo_root = Path(__file__).resolve().parents[2]

    resolved: Dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_revision": _git_revision(repo_root),
        "packages": _package_versions(PROVENANCE_PACKAGES),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
    }

    try:
        import torch

        resolved["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "device_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception:
        resolved["torch"] = None

    if extra:
        resolved.update(extra)

    return RunProvenance(config=cfg.as_dict(), resolved=resolved)
