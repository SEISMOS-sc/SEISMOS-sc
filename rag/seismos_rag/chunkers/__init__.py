"""Chunker registry."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..utils.logging import get_logger
from .base import (
    BaseChunker,
    Chunk,
    ChunkFormatError,
    ChunkingContext,
    CHUNK_KEYS,
    chunk_statistics,
    validate_chunks,
)
from .fixed_chunker import FixedChunker
from .langchain_semantic import LangChainSemanticChunker
from .llamaindex_semantic import LlamaIndexSemanticChunker
from .recursive_chunker import MissingDependencyError, RecursiveChunker
from .seismos import SeismosChunker, build_k_sweep_chunkers, laplacian_splits
from .sentence_splitter import SentenceSplitter

#: Methods included by ``--chunkers all``. The LLM baseline is excluded by
#: design; it only appears when explicitly enabled in config.
MAIN_CHUNKERS = (
    "fixed",
    "recursive",
    "langchain_semantic",
    "llamaindex_semantic",
    "seismos",
)

ALL_CHUNKER_NAMES = MAIN_CHUNKERS + ("llm_chunker",)

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkFormatError",
    "ChunkingContext",
    "CHUNK_KEYS",
    "FixedChunker",
    "LangChainSemanticChunker",
    "LlamaIndexSemanticChunker",
    "MAIN_CHUNKERS",
    "ALL_CHUNKER_NAMES",
    "MissingDependencyError",
    "RecursiveChunker",
    "SeismosChunker",
    "SentenceSplitter",
    "build_chunkers",
    "build_k_sweep_chunkers",
    "chunk_statistics",
    "laplacian_splits",
    "resolve_chunker_names",
    "validate_chunks",
]


def resolve_chunker_names(requested: Sequence[str] | str) -> List[str]:
    """Expand ``all`` and validate names, preserving :data:`MAIN_CHUNKERS` order."""
    if isinstance(requested, str):
        requested = [part.strip() for part in requested.split(",") if part.strip()]

    if not requested or list(requested) == ["all"]:
        return list(MAIN_CHUNKERS)

    unknown = [n for n in requested if n not in ALL_CHUNKER_NAMES and n != "all"]
    if unknown:
        raise ValueError(
            f"unknown chunker(s): {unknown}. Available: {list(ALL_CHUNKER_NAMES)} or 'all'"
        )

    names: List[str] = []
    for name in requested:
        if name == "all":
            names.extend(n for n in MAIN_CHUNKERS if n not in names)
        elif name not in names:
            names.append(name)
    return names


def build_chunkers(
    names: Sequence[str],
    cfg,
    encoder,
    sentence_splitter: SentenceSplitter,
    llm_runner: Optional[Any] = None,
    allow_missing: bool = False,
) -> List[BaseChunker]:
    """Instantiate chunkers from config. A missing optional dependency raises unless allowed."""
    log = get_logger()
    max_tokens = cfg["chunking.max_tokens"]
    cap_all = bool(cfg["chunking.cap_all_chunkers"])

    chunkers: List[BaseChunker] = []
    for name in names:
        try:
            chunkers.extend(
                _build_one(
                    name, cfg, encoder, sentence_splitter, llm_runner, max_tokens, cap_all
                )
            )
        except MissingDependencyError as exc:
            if not allow_missing:
                raise
            log.warning("skipping chunker %r: %s", name, exc)

    if not chunkers:
        raise ValueError("no chunkers could be constructed")

    log.info("chunkers: %s", [c.name for c in chunkers])
    return chunkers


def _build_one(
    name: str,
    cfg,
    encoder,
    sentence_splitter: SentenceSplitter,
    llm_runner,
    max_tokens: int,
    cap_all: bool,
) -> List[BaseChunker]:
    if name == "fixed":
        return [
            FixedChunker(
                target_tokens=cfg["chunking.fixed.target_tokens"], max_tokens=max_tokens
            )
        ]

    if name == "recursive":
        return [
            RecursiveChunker(
                encoder=encoder,
                chunk_size=cfg["chunking.recursive.chunk_size"],
                chunk_overlap=cfg["chunking.recursive.chunk_overlap"],
                separators=cfg["chunking.recursive.separators"],
                max_tokens=max_tokens,
            )
        ]

    if name == "langchain_semantic":
        section = cfg.section("chunking.langchain_semantic")
        return [
            LangChainSemanticChunker(
                encoder=encoder,
                sentence_splitter=sentence_splitter,
                breakpoint_threshold_type=section["breakpoint_threshold_type"],
                breakpoint_threshold_amount=section["breakpoint_threshold_amount"],
                buffer_size=section["buffer_size"],
                min_chunk_size=section.get("min_chunk_size"),
                max_tokens=max_tokens,
                apply_cap=cap_all,
            )
        ]

    if name == "llamaindex_semantic":
        section = cfg.section("chunking.llamaindex_semantic")
        return [
            LlamaIndexSemanticChunker(
                encoder=encoder,
                sentence_splitter=sentence_splitter,
                buffer_size=section["buffer_size"],
                breakpoint_percentile_threshold=section["breakpoint_percentile_threshold"],
                max_tokens=max_tokens,
                apply_cap=cap_all,
            )
        ]

    if name == "seismos":
        section = cfg.section("chunking.seismos")
        conventions = (
            ["canonical", "valley"]
            if section.get("compare_sign_conventions")
            else [section.get("sign_convention", "canonical")]
        )
        if "valley" in conventions:
            get_logger().warning(
                "SEISMOS sign_convention 'valley' is an ablation, not the published "
                "detector; its rows are named seismos*_valley"
            )

        built: List[BaseChunker] = []
        for convention in conventions:
            if section.get("run_k_sweep"):
                built.extend(
                    build_k_sweep_chunkers(
                        section["k_sweep"],
                        min_sentences=section["min_sentences"],
                        max_tokens=max_tokens,
                        apply_cap=cap_all,
                        sign_convention=convention,
                    )
                )
            else:
                suffix = "" if convention == "canonical" else f"_{convention}"
                built.append(
                    SeismosChunker(
                        k=section["k"],
                        min_sentences=section["min_sentences"],
                        max_tokens=max_tokens,
                        apply_cap=cap_all,
                        name=f"seismos{suffix}",
                        sign_convention=convention,
                    )
                )
        return built

    if name == "llm_chunker":
        section = cfg.section("chunking.llm_chunker")
        if not section.get("enabled"):
            raise ValueError(
                "llm_chunker requested but chunking.llm_chunker.enabled is false"
            )
        if llm_runner is None:
            raise ValueError("llm_chunker requires an LLM runner to be supplied")
        from .llm_chunker import LLMChunker

        return [
            LLMChunker(
                llm_runner=llm_runner,
                window_sentences=section["window_sentences"],
                min_sentences=cfg["chunking.seismos.min_sentences"],
                max_tokens=max_tokens,
                apply_cap=cap_all,
            )
        ]

    raise ValueError(f"unknown chunker: {name}")
