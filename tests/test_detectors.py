"""Detectors vs. frozen copies of the notebook functions behind the paper's numbers."""

import numpy as np
import pytest

from seismos.detectors import (laplacian_splits, prominence_splits, second_difference,
                               signal_diagnostics, zscore_splits)


def reference_laplacian(X, k):
    if X is None:
        return set()
    n = len(X)
    if n < 3:
        return set()
    D = np.array([X[i + 1] - 2.0 * X[i] + X[i - 1] for i in range(1, n - 1)])
    std_D = float(D.std())
    if std_D == 0.0:
        return set()
    return {i for idx, i in enumerate(range(1, n - 1)) if (-D[idx] / std_D) > k}


def reference_zscore(X, k, epsilon=0.04):
    doc_std = float(X.std())
    if doc_std < epsilon:
        return set()
    split_at = set()
    for i in range(1, len(X) - 1):
        if (X[i - 1] - X[i]) / doc_std > k and (X[i + 1] - X[i]) / doc_std > k:
            split_at.add(i)
    return split_at


def random_signals(seed, count=200):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(3, 300))
        yield rng.normal(0.8, rng.uniform(0.01, 0.25), size=n)


def ar1_signals(seed, count=100, rho=0.35):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(10, 400))
        x = np.zeros(n)
        x[0] = 0.8
        for i in range(1, n):
            x[i] = rho * x[i - 1] + (1 - rho) * 0.8 + rng.normal(0, 0.05)
        yield x


@pytest.mark.parametrize("k", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
def test_laplacian_matches_reference_on_random_signals(k):
    for x in random_signals(20260728):
        assert laplacian_splits(x, k) == reference_laplacian(x, k)


@pytest.mark.parametrize("k", [1.5, 2.5, 3.5])
def test_laplacian_matches_reference_on_ar1_signals(k):
    for x in ar1_signals(7):
        assert laplacian_splits(x, k) == reference_laplacian(x, k)


@pytest.mark.parametrize("x", [None, np.array([]), np.array([0.5]), np.array([0.5, 0.6]),
                               np.array([0.5, 0.6, 0.7]), np.full(50, 0.7), np.zeros(20),
                               np.array([1.0, -1.0] * 25)])
def test_laplacian_degenerate_inputs(x):
    for k in (1.5, 2.5, 3.5):
        assert laplacian_splits(x, k) == reference_laplacian(x, k)


@pytest.mark.parametrize("k", [0.5, 1.0, 2.0, 2.5])
def test_zscore_matches_reference(k):
    for x in random_signals(11):
        assert zscore_splits(x, k) == reference_zscore(x, k)


def test_canonical_rule_fires_beside_an_isolated_dip():
    """The published rule (-D/std(D) > k) fires on the shoulders of a dip, not at it."""
    x = np.full(60, 0.85)
    x[30] = 0.2
    fired = laplacian_splits(x, 2.5)
    assert fired == {29, 31}
    assert laplacian_splits(x, 2.5, sign_convention="valley") == {30}


def test_shoulder_statistic_is_independent_of_dip_depth():
    """For one isolated dip, the statistic at the shoulders is sqrt(N/6) whatever its depth."""
    for depth in (0.05, 0.3, 0.8):
        x = np.full(80, 0.9)
        x[40] -= depth
        D = second_difference(x)
        t = -D / D.std()
        assert t.max() == pytest.approx(np.sqrt(len(D) / 6), rel=1e-6)


def test_valley_convention_is_disjoint_from_canonical():
    x = np.random.default_rng(5).normal(0.8, 0.1, size=120)
    canonical, valley = laplacian_splits(x, 2.0), laplacian_splits(x, 2.0, "valley")
    assert valley and canonical and not (canonical & valley)


def test_unknown_sign_convention_is_rejected():
    with pytest.raises(ValueError):
        laplacian_splits(np.ones(10), 2.5, sign_convention="valleys")


def test_prominence_uses_mean_scaled_threshold():
    x = np.array([0.9, 0.9, 0.5, 0.9, 0.9, 0.88, 0.9])
    # P(2) = 0.4 > 0.05 * mean; P(5) = 0.02 < 0.05 * mean (~0.041)
    assert prominence_splits(x, alpha=0.05) == {2}


def test_acf_factor_close_to_theory_for_ar1():
    rho = 0.35
    rng = np.random.default_rng(1)
    x = np.zeros(20000)
    for i in range(1, len(x)):
        x[i] = rho * x[i - 1] + rng.normal()
    diag = signal_diagnostics(x)
    assert diag["acf_factor"] == pytest.approx(np.sqrt(6 - 8 * rho + 2 * rho ** 2), rel=0.03)
    assert diag["lag1_autocorr"] == pytest.approx(rho, abs=0.02)
