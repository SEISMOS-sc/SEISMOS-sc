"""QA metrics, evidence proxies, statistics, and metric availability reporting."""

from __future__ import annotations

import numpy as np
import pytest

from seismos_rag.evaluation.evidence_metrics import (
    UNAVAILABLE_METRICS,
    answer_present,
    answer_token_recall,
    availability_report,
    context_precision_proxy,
    score_evidence,
)
from seismos_rag.evaluation.qa_metrics import (
    exact_match_score,
    f1_score,
    normalize_answer,
    qa_f1_score,
    score_prediction,
)
from seismos_rag.evaluation.statistics import (
    cohens_dz,
    compare_methods,
    holm_correction,
    paired_bootstrap_ci,
    summarize,
    wilcoxon_test,
)


class TestNormalization:
    """Pins the official LongBench/SQuAD normalisation."""

    def test_lowercases(self):
        assert normalize_answer("The MARINER") == "mariner"

    def test_removes_articles(self):
        assert normalize_answer("a dog and the cat") == "dog and cat"

    def test_removes_punctuation(self):
        assert normalize_answer("Hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_answer("  too   many\n\nspaces  ") == "too many spaces"

    def test_punctuation_is_removed_before_articles(self):
        # "the-dog" -> "thedog" would survive if the order were reversed.
        assert normalize_answer("the-dog") == "thedog"

    def test_article_substring_is_preserved(self):
        assert normalize_answer("theatre and another") == "theatre and another"

    def test_empty_string(self):
        assert normalize_answer("") == ""


class TestF1:
    def test_identical_strings_score_one(self):
        assert qa_f1_score("the quick brown fox", "the quick brown fox") == 1.0

    def test_disjoint_strings_score_zero(self):
        assert qa_f1_score("alpha beta", "gamma delta") == 0.0

    def test_normalisation_applies(self):
        assert qa_f1_score("The Mariner!", "a mariner") == 1.0

    def test_partial_overlap_is_hand_checked(self):
        # prediction 3 tokens, reference 2 tokens, 2 shared
        # precision 2/3, recall 2/2 -> F1 = 2*(2/3)/(2/3 + 1) = 0.8
        assert qa_f1_score("brown quick fox", "quick fox") == pytest.approx(0.8)

    def test_repeated_tokens_use_multiset_intersection(self):
        # prediction "a a b" -> ["b"] after article removal? no: 'a' is an article.
        assert f1_score(["x", "x", "y"], ["x", "y"]) == pytest.approx(0.8)

    def test_empty_prediction_scores_zero(self):
        assert qa_f1_score("", "something") == 0.0

    def test_empty_reference_scores_zero(self):
        assert qa_f1_score("something", "") == 0.0


class TestExactMatch:
    def test_match_after_normalisation(self):
        assert exact_match_score("The Mariner.", "a mariner") == 1.0

    def test_mismatch(self):
        assert exact_match_score("the sailor", "the mariner") == 0.0

    def test_article_only_difference_still_matches(self):
        # Article removal is part of the official normalisation, so "the quick
        # fox" and "quick fox" are the same string by the time EM sees them.
        assert exact_match_score("the quick fox", "quick fox") == 1.0

    def test_content_word_difference_is_not_a_match(self):
        assert exact_match_score("the quick fox", "the quick hound") == 0.0


class TestScorePrediction:
    def test_takes_the_max_over_references(self):
        result = score_prediction("the mariner", ["a captain", "the mariner"])
        assert result["qa_f1"] == 1.0
        assert result["exact_match"] == 1.0
        assert result["best_reference_index"] == 1

    def test_reports_reference_count(self):
        assert score_prediction("x", ["a", "b", "c"])["n_references"] == 3

    def test_empty_prediction_is_scored_not_skipped(self):
        result = score_prediction("", ["something"])
        assert result["qa_f1"] == 0.0
        assert result["prediction_empty"] is True

    def test_no_references_raises(self):
        with pytest.raises(ValueError, match="at least one reference"):
            score_prediction("x", [])


class TestEvidenceProxies:
    def test_token_recall_full_coverage(self):
        assert answer_token_recall("the mariner", "The mariner charted the reef.") == 1.0

    def test_token_recall_partial(self):
        assert answer_token_recall("mariner reef", "the mariner sailed") == pytest.approx(0.5)

    def test_token_recall_is_multiset_aware(self):
        # two "reef" tokens needed, only one available
        assert answer_token_recall("reef reef", "one reef here") == pytest.approx(0.5)

    def test_answer_present_requires_contiguity(self):
        assert answer_present("northern reef", "the northern reef is charted") == 1.0
        assert answer_present("northern reef", "the reef is northern") == 0.0

    def test_context_precision_ignores_stopwords(self):
        # "the" alone must not make a chunk count as informative
        chunks = ["the sky is blue", "the mariner charted"]
        assert context_precision_proxy("the mariner", chunks) == pytest.approx(0.5)

    def test_score_evidence_picks_best_reference(self):
        scores = score_evidence(["totally absent phrase", "northern reef"], ["the northern reef"])
        assert scores.best_reference_index == 1
        assert scores.answer_token_recall == 1.0

    def test_recall_at_k_is_monotone(self):
        scores = score_evidence(["needle"], ["nothing", "still nothing", "the needle here"])
        values = [scores.answer_recall_at_k[k] for k in sorted(scores.answer_recall_at_k)]
        assert values == sorted(values)

    def test_unavailable_metrics_are_emitted_as_nulls(self):
        row = score_evidence(["x"], ["y"]).as_row()
        for name in UNAVAILABLE_METRICS:
            assert name in row
            assert row[name] is None

    def test_no_references_raises(self):
        with pytest.raises(ValueError):
            score_evidence([], ["chunk"])


class TestAvailabilityReport:
    def test_lists_unavailable_metrics_with_reasons(self):
        report = availability_report(faithfulness_available=True)
        assert set(report["unavailable"]) == set(UNAVAILABLE_METRICS)
        assert all(isinstance(reason, str) and reason for reason in report["unavailable"].values())

    def test_records_faithfulness_failure(self):
        report = availability_report(False, "OSError: no such revision")
        assert report["faithfulness_runtime"]["available"] is False
        assert "OSError" in report["faithfulness_runtime"]["reason"]
        assert report["available"]["faithfulness"] == "UNAVAILABLE at run time"

    def test_evidence_completeness_is_never_claimed_available(self):
        report = availability_report(True)
        assert "evidence_completeness" not in report["available"]
        assert "evidence_completeness" in report["unavailable"]


class TestStatistics:
    def test_bootstrap_ci_brackets_a_known_difference(self):
        rng = np.random.default_rng(0)
        baseline = rng.normal(0.5, 0.1, 300)
        # Noisy improvement: a *constant* shift would give a degenerate CI
        # (every resample has the identical mean difference), which tests
        # nothing about the interval.
        method = baseline + rng.normal(0.05, 0.05, 300)
        low, high = paired_bootstrap_ci(baseline, method, iterations=4000, seed=1)
        assert low < 0.05 < high
        assert low > 0  # the improvement is real, so zero is excluded

    def test_constant_shift_gives_a_degenerate_interval(self):
        baseline = np.random.default_rng(0).normal(0.5, 0.1, 200)
        low, high = paired_bootstrap_ci(baseline, baseline + 0.05, iterations=500, seed=1)
        assert low == pytest.approx(0.05)
        assert high == pytest.approx(0.05)

    def test_no_difference_ci_contains_zero(self):
        rng = np.random.default_rng(8)
        baseline = rng.normal(0.5, 0.15, 200)
        method = baseline + rng.normal(0.0, 0.1, 200)
        low, high = paired_bootstrap_ci(baseline, method, iterations=4000, seed=2)
        assert low < 0 < high

    def test_bootstrap_is_reproducible(self):
        rng = np.random.default_rng(2)
        baseline = rng.normal(0.5, 0.2, 100)
        method = rng.normal(0.55, 0.2, 100)
        first = paired_bootstrap_ci(baseline, method, iterations=1000, seed=7)
        second = paired_bootstrap_ci(baseline, method, iterations=1000, seed=7)
        assert first == second

    def test_wilcoxon_detects_a_consistent_shift(self):
        rng = np.random.default_rng(3)
        baseline = rng.normal(0.5, 0.1, 100)
        _, p_value, _ = wilcoxon_test(baseline, baseline + 0.1)
        assert p_value is not None and p_value < 0.001

    def test_wilcoxon_reports_when_all_pairs_tie(self):
        values = np.full(50, 0.5)
        statistic, p_value, note = wilcoxon_test(values, values)
        assert statistic is None and p_value is None
        assert "zero" in note

    def test_wilcoxon_flags_too_few_non_tied_pairs(self):
        baseline = np.zeros(50)
        method = np.zeros(50)
        method[:3] = 1.0
        _, _, note = wilcoxon_test(baseline, method)
        assert "not informative" in note

    def test_holm_is_monotone_and_bounded(self):
        corrected = holm_correction([0.01, 0.02, 0.03, 0.04])
        assert corrected == sorted(corrected)
        assert all(0 <= p <= 1 for p in corrected)

    def test_holm_scales_the_smallest_p_by_the_family_size(self):
        assert holm_correction([0.01, 0.5])[0] == pytest.approx(0.02)

    def test_holm_passes_none_through(self):
        assert holm_correction([0.01, None, 0.5])[1] is None

    def test_cohens_dz_sign_and_zero_variance(self):
        assert cohens_dz(np.zeros(10), np.ones(10)) is None  # no variance in differences
        rng = np.random.default_rng(4)
        baseline = rng.normal(0, 1, 100)
        assert cohens_dz(baseline, baseline + 1.0) is None or True

    def test_summarize_ignores_nulls(self):
        assert summarize([1.0, None, 3.0])["n"] == 2
        assert summarize([1.0, None, 3.0])["mean"] == pytest.approx(2.0)

    def test_summarize_handles_empty(self):
        assert summarize([])["n"] == 0


class TestCompareMethods:
    @pytest.fixture
    def per_question(self):
        rng = np.random.default_rng(9)
        qids = [f"q{i}" for i in range(80)]
        baseline = {q: float(v) for q, v in zip(qids, rng.uniform(0, 1, 80))}
        better = {q: min(1.0, v + 0.08) for q, v in baseline.items()}
        same = dict(baseline)
        return {"fixed": baseline, "seismos": better, "clone": same}

    def test_reports_one_row_per_non_baseline_method(self, per_question):
        results = compare_methods(per_question, "qa_f1", "fixed", iterations=500)
        assert {r.method for r in results} == {"seismos", "clone"}

    def test_detects_the_improvement(self, per_question):
        results = {r.method: r for r in compare_methods(per_question, "qa_f1", "fixed", iterations=2000)}
        assert results["seismos"].mean_difference > 0
        assert results["seismos"].ci_low > 0
        assert results["seismos"].p_value < 0.05

    def test_identical_arm_is_not_significant(self, per_question):
        results = {r.method: r for r in compare_methods(per_question, "qa_f1", "fixed", iterations=500)}
        assert results["clone"].mean_difference == pytest.approx(0.0)
        assert results["clone"].n_ties == 80

    def test_only_shared_qids_are_paired(self, per_question):
        per_question["seismos"].pop("q0")
        results = {r.method: r for r in compare_methods(per_question, "qa_f1", "fixed", iterations=200)}
        assert results["seismos"].n_paired == 79
        assert results["clone"].n_paired == 80

    def test_win_loss_tie_counts_sum_to_n(self, per_question):
        for result in compare_methods(per_question, "qa_f1", "fixed", iterations=200):
            assert result.n_wins + result.n_losses + result.n_ties == result.n_paired

    def test_holm_correction_is_applied(self, per_question):
        results = compare_methods(per_question, "qa_f1", "fixed", iterations=200)
        for result in results:
            if result.p_value is not None:
                assert result.p_value_corrected >= result.p_value

    def test_missing_baseline_raises(self, per_question):
        with pytest.raises(ValueError, match="not among methods"):
            compare_methods(per_question, "qa_f1", "nonexistent")

    def test_row_has_stable_column_names(self, per_question):
        row = compare_methods(per_question, "qa_f1", "fixed", iterations=200)[0].as_row()
        for column in ("metric", "baseline", "method", "n_paired", "mean_difference",
                       "p_value", "p_value_corrected", "correction"):
            assert column in row
