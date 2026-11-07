"""Paired significance tests against a baseline chunker: bootstrap CI on the mean
difference, Wilcoxon signed-rank, and Holm correction across methods per metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger


@dataclass
class ComparisonResult:
    metric: str
    baseline: str
    method: str
    n_paired: int
    baseline_mean: float
    method_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    confidence_level: float
    bootstrap_iterations: int
    wilcoxon_statistic: Optional[float]
    p_value: Optional[float]
    p_value_corrected: Optional[float]
    correction: str
    significant: Optional[bool]
    n_wins: int
    n_losses: int
    n_ties: int
    cohens_dz: Optional[float]
    note: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "method": self.method,
            "n_paired": self.n_paired,
            "baseline_mean": _round(self.baseline_mean),
            "method_mean": _round(self.method_mean),
            "mean_difference": _round(self.mean_difference),
            f"ci{int(self.confidence_level * 100)}_low": _round(self.ci_low),
            f"ci{int(self.confidence_level * 100)}_high": _round(self.ci_high),
            "bootstrap_iterations": self.bootstrap_iterations,
            "wilcoxon_statistic": _round(self.wilcoxon_statistic),
            "p_value": _round(self.p_value, 8),
            "p_value_corrected": _round(self.p_value_corrected, 8),
            "correction": self.correction,
            "significant_at_0.05": self.significant,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "n_ties": self.n_ties,
            "cohens_dz": _round(self.cohens_dz),
            "note": self.note,
        }


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def paired_bootstrap_ci(
    baseline: np.ndarray,
    method: np.ndarray,
    iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for ``mean(method - baseline)``, resampling questions."""
    n = len(baseline)
    if n == 0:
        return (float("nan"), float("nan"))

    differences = method - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(iterations, n))
    means = differences[indices].mean(axis=1)

    alpha = 1.0 - confidence_level
    low = float(np.percentile(means, 100 * alpha / 2))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return low, high


def wilcoxon_test(
    baseline: np.ndarray, method: np.ndarray
) -> Tuple[Optional[float], Optional[float], str]:
    """Two-sided Wilcoxon signed-rank test. Returns ``(statistic, p, note)``."""
    differences = method - baseline
    non_zero = differences[differences != 0]

    if len(non_zero) == 0:
        return None, None, "all paired differences are zero; test not defined"
    if len(non_zero) < 6:
        # Below ~6 non-tied pairs the exact test cannot reach p<0.05 two-sided.
        note = f"only {len(non_zero)} non-tied pairs; p-value is not informative"
    else:
        note = ""

    try:
        from scipy.stats import wilcoxon

        statistic, p_value = wilcoxon(
            baseline, method, zero_method="wilcox", alternative="two-sided"
        )
        return float(statistic), float(p_value), note
    except Exception as exc:  # noqa: BLE001
        get_logger().warning("Wilcoxon test failed: %s", exc)
        return None, None, f"test failed: {exc}"


def holm_correction(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down adjustment. ``None`` entries pass through."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    corrected: List[Optional[float]] = [None] * len(p_values)
    if not indexed:
        return corrected

    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)
    running_max = 0.0
    for rank, (index, p) in enumerate(indexed):
        adjusted = min(1.0, (m - rank) * p)
        running_max = max(running_max, adjusted)  # enforce monotonicity
        corrected[index] = running_max
    return corrected


def cohens_dz(baseline: np.ndarray, method: np.ndarray) -> Optional[float]:
    """Paired effect size: mean difference over the SD of differences."""
    differences = method - baseline
    std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    if std == 0.0:
        return None
    return float(differences.mean() / std)


def compare_methods(
    per_question: Dict[str, Dict[str, float]],
    metric: str,
    baseline: str,
    iterations: int = 10000,
    confidence_level: float = 0.95,
    correction: str = "holm",
    seed: int = 42,
) -> List[ComparisonResult]:
    """Only questions answered by both arms are used; their count is ``n_paired``."""
    log = get_logger()
    if baseline not in per_question:
        raise ValueError(
            f"baseline {baseline!r} not among methods: {sorted(per_question)}"
        )

    baseline_scores = per_question[baseline]
    results: List[ComparisonResult] = []

    for method in sorted(per_question):
        if method == baseline:
            continue
        method_scores = per_question[method]

        shared = sorted(set(baseline_scores) & set(method_scores))
        if not shared:
            log.warning("no shared qids between %s and %s on %s", baseline, method, metric)
            continue

        a = np.array([baseline_scores[q] for q in shared], dtype=np.float64)
        b = np.array([method_scores[q] for q in shared], dtype=np.float64)

        low, high = paired_bootstrap_ci(a, b, iterations, confidence_level, seed)
        statistic, p_value, note = wilcoxon_test(a, b)

        differences = b - a
        results.append(
            ComparisonResult(
                metric=metric,
                baseline=baseline,
                method=method,
                n_paired=len(shared),
                baseline_mean=float(a.mean()),
                method_mean=float(b.mean()),
                mean_difference=float(differences.mean()),
                ci_low=low,
                ci_high=high,
                confidence_level=confidence_level,
                bootstrap_iterations=iterations,
                wilcoxon_statistic=statistic,
                p_value=p_value,
                p_value_corrected=None,
                correction=correction,
                significant=None,
                n_wins=int((differences > 0).sum()),
                n_losses=int((differences < 0).sum()),
                n_ties=int((differences == 0).sum()),
                cohens_dz=cohens_dz(a, b),
                note=note,
            )
        )

    if correction == "holm" and results:
        adjusted = holm_correction([r.p_value for r in results])
        for result, value in zip(results, adjusted):
            result.p_value_corrected = value
            result.significant = None if value is None else bool(value < 0.05)
    else:
        for result in results:
            result.p_value_corrected = result.p_value
            result.significant = (
                None if result.p_value is None else bool(result.p_value < 0.05)
            )

    return results


def run_all_comparisons(
    per_question_by_metric: Dict[str, Dict[str, Dict[str, float]]],
    baseline: str,
    iterations: int = 10000,
    confidence_level: float = 0.95,
    correction: str = "holm",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """One Holm family per metric."""
    rows: List[Dict[str, Any]] = []
    for metric, per_question in per_question_by_metric.items():
        if baseline not in per_question:
            get_logger().warning(
                "skipping significance tests for %s: baseline %r missing", metric, baseline
            )
            continue
        results = compare_methods(
            per_question,
            metric=metric,
            baseline=baseline,
            iterations=iterations,
            confidence_level=confidence_level,
            correction=correction,
            seed=seed,
        )
        rows.extend(r.as_row() for r in results)
    return rows


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Mean/std/median/quartiles, used for the summary CSV."""
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "p25": float("nan"), "p75": float("nan"), "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "n": int(array.size),
    }
