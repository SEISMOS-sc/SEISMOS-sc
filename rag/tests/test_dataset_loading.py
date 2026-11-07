"""Dataset loading: contract, determinism, and non-modification."""

from __future__ import annotations

import pytest

from seismos_rag.data.longbench_loader import (
    DatasetLoadError,
    REQUIRED_KEYS,
    _coerce_answers,
    _warn_on_duplicate_ids,
    validate_sample,
)
from seismos_rag.utils.seeding import sample_indices


class TestSampleContract:
    def test_valid_sample_passes(self, sample):
        validate_sample(sample)

    def test_required_keys_are_the_spec_keys(self):
        assert REQUIRED_KEYS == ("id", "question", "context", "answers")

    @pytest.mark.parametrize("key", ["id", "question", "context", "answers"])
    def test_missing_key_raises(self, sample, key):
        del sample[key]
        with pytest.raises(DatasetLoadError, match=key):
            validate_sample(sample)

    def test_empty_answers_raises(self, sample):
        sample["answers"] = []
        with pytest.raises(DatasetLoadError, match="non-empty list"):
            validate_sample(sample)

    def test_non_string_answer_raises(self, sample):
        sample["answers"] = ["ok", 42]
        with pytest.raises(DatasetLoadError, match="must be strings"):
            validate_sample(sample)

    def test_blank_context_raises(self, sample):
        sample["context"] = "   "
        with pytest.raises(DatasetLoadError, match="context"):
            validate_sample(sample)


class TestAnswerCoercion:
    def test_list_passes_through(self):
        assert _coerce_answers(["a", "b"]) == ["a", "b"]

    def test_bare_string_is_wrapped(self):
        assert _coerce_answers("only") == ["only"]

    def test_unexpected_type_raises(self):
        with pytest.raises(DatasetLoadError):
            _coerce_answers({"a": 1})


class TestDuplicateIds:
    def test_unique_ids_pass(self):
        _warn_on_duplicate_ids([{"id": "a"}, {"id": "b"}])

    def test_duplicates_raise_because_they_collide_checkpoint_keys(self):
        with pytest.raises(DatasetLoadError, match="duplicate"):
            _warn_on_duplicate_ids([{"id": "a"}, {"id": "a"}])


class TestSubsampling:
    def test_same_seed_gives_same_selection(self):
        assert sample_indices(1000, 50, seed=42) == sample_indices(1000, 50, seed=42)

    def test_different_seed_gives_different_selection(self):
        assert sample_indices(1000, 50, seed=42) != sample_indices(1000, 50, seed=7)

    def test_selection_is_ascending(self):
        indices = sample_indices(1000, 50, seed=42)
        assert indices == sorted(indices)

    def test_no_duplicates(self):
        indices = sample_indices(1000, 200, seed=1)
        assert len(set(indices)) == len(indices) == 200

    def test_requesting_more_than_available_returns_everything(self):
        assert sample_indices(10, 50, seed=1) == list(range(10))

    def test_none_returns_everything(self):
        assert sample_indices(10, None, seed=1) == list(range(10))

    def test_indices_are_in_range(self):
        assert all(0 <= i < 200 for i in sample_indices(200, 200, seed=3))


@pytest.mark.network
@pytest.mark.slow
class TestRealDataset:
    """Hits the HF hub. Deselect with -m 'not network'."""

    def test_loads_and_respects_contract(self, tmp_path):
        from seismos_rag.data.longbench_loader import load_longbench_narrativeqa_with_stats

        samples, stats = load_longbench_narrativeqa_with_stats(
            num_samples=4, seed=42, cache_dir=str(tmp_path)
        )
        assert len(samples) == 4
        assert stats.n_selected == 4
        for entry in samples:
            validate_sample(entry)
            assert len(entry["context"]) > 1000  # LongBench contexts are long

    def test_selection_is_reproducible(self, tmp_path):
        from seismos_rag.data.longbench_loader import load_longbench_narrativeqa

        first = load_longbench_narrativeqa(num_samples=4, seed=42, cache_dir=str(tmp_path))
        second = load_longbench_narrativeqa(num_samples=4, seed=42, cache_dir=str(tmp_path))
        assert [s["id"] for s in first] == [s["id"] for s in second]
        assert [s["context_sha256"] for s in first] == [s["context_sha256"] for s in second]
