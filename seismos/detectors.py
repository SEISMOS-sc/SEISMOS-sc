"""Boundary detectors over the adjacent-sentence similarity signal.

Each detector takes ``X`` (``X[i] = cos(e_i, e_{i+1})``) and returns the set of
sentence indices ``i`` with a boundary after sentence ``i``.

The published Laplacian rule ``-D_i / std(D) > k`` fires at local maxima of
``X`` (``D_i > 0`` at a minimum). All paper results use it, so it is the
default; ``sign_convention="valley"`` flips the sign for ablations.
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np

PAPER_K = 2.5
MIN_SENTENCES = 5
ZSCORE_EPSILON = 0.04
SIGN_CONVENTIONS = ("canonical", "valley")


def second_difference(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X[2:] - 2.0 * X[1:-1] + X[:-2]


def laplacian_splits(
    X: Optional[np.ndarray],
    k: float = PAPER_K,
    sign_convention: str = "canonical",
) -> Set[int]:
    """SEISMOS: split where the Laplacian of ``X``, normalised by its own std, exceeds ``k``."""
    if sign_convention not in SIGN_CONVENTIONS:
        raise ValueError(f"sign_convention must be one of {SIGN_CONVENTIONS}, got {sign_convention!r}")
    if X is None or len(X) < 3:
        return set()
    D = second_difference(X)
    std_D = float(D.std())
    if std_D == 0.0:
        return set()
    t = (-D if sign_convention == "canonical" else D) / std_D
    return {int(i) + 1 for i in np.flatnonzero(t > k)}


def zscore_splits(
    X: Optional[np.ndarray],
    k: float = PAPER_K,
    epsilon: float = ZSCORE_EPSILON,
) -> Set[int]:
    """Split where both slopes into a valley exceed ``k * std(X)``; no splits if ``std(X) < epsilon``."""
    if X is None or len(X) < 3:
        return set()
    X = np.asarray(X, dtype=np.float64)
    sd = float(X.std())
    if sd < epsilon:
        return set()
    left = (X[:-2] - X[1:-1]) / sd
    right = (X[2:] - X[1:-1]) / sd
    return {int(i) + 1 for i in np.flatnonzero((left > k) & (right > k))}


def prominence_splits(X: Optional[np.ndarray], alpha: float = 0.05) -> Set[int]:
    """Adaptive Prominence: split where ``min(X[i-1] - X[i], X[i+1] - X[i]) > alpha * mean(X)``."""
    if X is None or len(X) < 3:
        return set()
    X = np.asarray(X, dtype=np.float64)
    P = np.minimum(X[:-2] - X[1:-1], X[2:] - X[1:-1])
    return {int(i) + 1 for i in np.flatnonzero(P > alpha * float(X.mean()))}


def signal_diagnostics(X: Optional[np.ndarray]) -> dict:
    """Signal statistics. Under AR(1), ``acf_factor`` should be ``sqrt(6 - 8 rho + 2 rho^2)``."""
    if X is None or len(X) < 3:
        return {"signal_length": 0 if X is None else len(X), "mean_X": None, "std_X": None,
                "std_D": None, "acf_factor": None, "lag1_autocorr": None}
    X = np.asarray(X, dtype=np.float64)
    D = second_difference(X)
    std_X, std_D = float(X.std()), float(D.std())
    c = X - X.mean()
    denom = float(c @ c)
    return {
        "signal_length": int(len(X)),
        "mean_X": float(X.mean()),
        "std_X": std_X,
        "std_D": std_D,
        "acf_factor": std_D / std_X if std_X > 0 else None,
        "lag1_autocorr": float(c[:-1] @ c[1:]) / denom if denom > 0 else None,
    }


DETECTORS = {
    "laplacian": laplacian_splits,
    "zscore": zscore_splits,
    "prominence": prominence_splits,
}
