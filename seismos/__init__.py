"""SEISMOS: semantic chunking as statistical boundary detection."""

from .chunking import Chunk, ChunkSpec, DocumentSignal, SeismosChunker, compute_signal, make_chunks, pack_ranges
from .detectors import PAPER_K, laplacian_splits, prominence_splits, signal_diagnostics, zscore_splits
from .encoders import ENCODERS, Encoder, sanity_check
from .segment import split_sentences

__version__ = "1.0.0"

__all__ = [
    "Chunk", "ChunkSpec", "DocumentSignal", "SeismosChunker", "compute_signal", "make_chunks", "pack_ranges",
    "PAPER_K", "laplacian_splits", "zscore_splits", "prominence_splits", "signal_diagnostics",
    "ENCODERS", "Encoder", "sanity_check", "split_sentences",
]
