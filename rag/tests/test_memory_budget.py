"""VRAM preflight: weights, KV cache and attention workspace, checked against both
the summed and the smallest-card free memory."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from seismos_rag.generation.llm_runner import (  # noqa: E402
    ATTN_IMPLEMENTATIONS,
    LLMRunner,
    _is_cuda_oom,
    resolve_attn_implementation,
)
from seismos_rag.utils.gpu import (  # noqa: E402
    KNOWN_MODEL_SHAPES,
    KNOWN_MODEL_WEIGHT_GB,
    MemoryEstimate,
    estimate_generator_memory,
)

QWEN = "Qwen/Qwen2.5-7B-Instruct"


class TestAttentionResolution:
    def test_auto_prefers_sdpa(self):
        assert resolve_attn_implementation("auto") == "sdpa"

    def test_sdpa_is_the_default_for_the_runner(self):
        """REGRESSION: the default used to be eager, which is what OOM'd."""
        assert LLMRunner(model_key="qwen", device="cpu").attn_implementation == "auto"
        assert (
            LLMRunner(model_key="qwen", device="cpu").resolved_attn_implementation == "sdpa"
        )

    def test_explicit_eager_is_honoured(self):
        """It is the reference kernel; debugging with it must stay possible."""
        assert resolve_attn_implementation("eager") == "eager"

    def test_falls_back_to_eager_without_sdpa(self, monkeypatch):
        import seismos_rag.generation.llm_runner as runner_module

        monkeypatch.setattr(runner_module, "supports_sdpa", lambda: False)
        assert resolve_attn_implementation("auto") == "eager"

    def test_flash_attention_2_downgrades_when_unsupported(self, monkeypatch):
        """T4 is sm_75: no FA2 kernels exist. Downgrade to sdpa, not to eager."""
        import seismos_rag.generation.llm_runner as runner_module

        monkeypatch.setattr(runner_module, "supports_flash_attention_2", lambda: False)
        assert resolve_attn_implementation("flash_attention_2") == "sdpa"

    def test_flash_attention_2_is_used_when_available(self, monkeypatch):
        import seismos_rag.generation.llm_runner as runner_module

        monkeypatch.setattr(runner_module, "supports_flash_attention_2", lambda: True)
        assert resolve_attn_implementation("flash_attention_2") == "flash_attention_2"

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="attn_implementation"):
            resolve_attn_implementation("xformers")

    def test_every_accepted_value_resolves(self, monkeypatch):
        import seismos_rag.generation.llm_runner as runner_module

        monkeypatch.setattr(runner_module, "supports_flash_attention_2", lambda: True)
        for requested in ATTN_IMPLEMENTATIONS:
            assert resolve_attn_implementation(requested) in (
                "sdpa",
                "eager",
                "flash_attention_2",
            )

    def test_resolution_is_recorded_in_provenance(self):
        described = LLMRunner(model_key="qwen", device="cpu").describe()
        assert described["attn_implementation"] == "auto"
        assert described["attn_implementation_resolved"] == "sdpa"


class TestMemoryEstimate:
    def _estimate(self, attn: str, batch: int = 4, tokens: int = 3072):
        return estimate_generator_memory(
            QWEN,
            "float16",
            "none",
            batch_size=batch,
            max_input_tokens=tokens,
            max_new_tokens=128,
            attn_implementation=attn,
        )

    def test_unknown_model_has_no_estimate(self):
        assert estimate_generator_memory("some/unknown-model", "float16") is None

    def test_every_weight_entry_has_a_shape(self):
        """A model that can be sized but not shaped would silently skip the check."""
        for name, _ in KNOWN_MODEL_WEIGHT_GB:
            assert name in KNOWN_MODEL_SHAPES, name

    def test_eager_costs_far_more_than_sdpa(self):
        assert self._estimate("eager").activation_gb > 5 * self._estimate("sdpa").activation_gb

    def test_eager_is_quadratic_in_prompt_length(self):
        short = self._estimate("eager", tokens=1024).activation_gb
        long = self._estimate("eager", tokens=2048).activation_gb
        assert long == pytest.approx(4 * short, rel=0.01)

    def test_activations_scale_with_batch(self):
        one = self._estimate("sdpa", batch=1).activation_gb
        four = self._estimate("sdpa", batch=4).activation_gb
        assert four == pytest.approx(4 * one, rel=0.01)

    def test_reproduces_the_failing_configuration(self):
        """The config from the failed run must not fit on a 14.56 GB T4."""
        estimate = self._estimate("eager", batch=4, tokens=3072)
        # 2 cards, so the weights are halved; the activations are not.
        assert estimate.per_device_gb(2) > 14.56

    def test_the_sdpa_replacement_does_fit(self):
        estimate = self._estimate("sdpa", batch=4, tokens=3072)
        assert estimate.per_device_gb(2) < 14.56

    def test_kv_cache_uses_grouped_query_heads(self):
        """Qwen2.5-7B has 28 query heads but 4 KV heads; using 28 overstates by 7x."""
        estimate = self._estimate("sdpa", batch=1, tokens=3072)
        shape = KNOWN_MODEL_SHAPES[QWEN]
        expected = (
            2 * shape["layers"] * shape["kv_heads"] * shape["head_dim"] * (3072 + 128) * 2
        ) / 1024**3
        assert estimate.kv_cache_gb == pytest.approx(expected, rel=1e-6)

    def test_per_device_divides_only_the_sharded_part(self):
        estimate = MemoryEstimate(
            weights_gb=16.0, kv_cache_gb=4.0, activation_gb=3.0, headroom_gb=1.0
        )
        assert estimate.total_gb == 24.0
        # (16 + 4) / 2 + 3 + 1
        assert estimate.per_device_gb(2) == 14.0

    def test_single_device_per_device_equals_total(self):
        estimate = MemoryEstimate(
            weights_gb=8.0, kv_cache_gb=1.0, activation_gb=2.0, headroom_gb=1.5
        )
        assert estimate.per_device_gb(1) == estimate.total_gb


class TestOomDetection:
    def test_torch_oom_error_is_recognised(self):
        error_type = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
        assert _is_cuda_oom(error_type("CUDA out of memory."))

    def test_legacy_runtime_error_is_recognised(self):
        assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1.54 GiB"))

    def test_unrelated_runtime_error_is_not(self):
        assert not _is_cuda_oom(RuntimeError("shape mismatch"))

    def test_unrelated_exception_type_is_not(self):
        assert not _is_cuda_oom(ValueError("out of memory"))


class _OomRunner(LLMRunner):
    """Fails on any batch larger than ``limit``, like a GPU with a fixed budget."""

    def __init__(self, limit: int, **kwargs):
        super().__init__(model_key="qwen", device="cpu", **kwargs)
        self.limit = limit
        self.attempts = []

    def generate_batch(self, prompts):
        self.attempts.append(len(prompts))
        if len(prompts) > self.limit:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.54 GiB")
        return [
            {
                "answer": text,
                "input_tokens": 10,
                "output_tokens": 3,
                "seconds": 0.1,
                "dominant_token_share": 0.1,
                "dominant_token_id": 5,
                "dominant_token_count": 3,
                "degenerate": False,
                "effective_batch_size": len(prompts),
            }
            for text in prompts
        ]

    def _count_tokens(self, text):
        return len(text)


class TestOomBackoff:
    def test_batch_is_halved_until_it_fits(self):
        runner = _OomRunner(limit=1, batch_size=4)
        results = runner._generate_batch_resilient(["a", "b", "c", "d"])

        assert [r["answer"] for r in results] == ["a", "b", "c", "d"]
        assert runner.attempts == [4, 2, 1, 1, 2, 1, 1]
        assert all(r["effective_batch_size"] == 1 for r in results)

    def test_order_is_preserved_across_splits(self):
        runner = _OomRunner(limit=2, batch_size=8)
        prompts = [str(i) for i in range(8)]
        assert [r["answer"] for r in runner._generate_batch_resilient(prompts)] == prompts

    def test_effective_batch_size_records_the_reduction(self):
        runner = _OomRunner(limit=2, batch_size=4)
        results = runner._generate_batch_resilient(["a", "b", "c", "d"])
        assert [r["effective_batch_size"] for r in results] == [2, 2, 2, 2]

    def test_oom_events_are_counted(self):
        runner = _OomRunner(limit=2, batch_size=4)
        runner._generate_batch_resilient(["a", "b", "c", "d"])
        assert runner._oom_events == 1

    def test_a_single_prompt_that_still_ooms_raises_with_remedies(self):
        runner = _OomRunner(limit=0, batch_size=4)
        with pytest.raises(RuntimeError) as excinfo:
            runner._generate_batch_resilient(["a"])

        message = str(excinfo.value)
        assert "out of memory" in message.lower()
        # The remedies must be actionable flags, and the resume note must be
        # there: without it a user re-runs from scratch for no reason.
        assert "--gen_batch_size" in message
        assert "--max_input_tokens" in message
        assert "--quantization 4bit" in message
        assert "generations.jsonl" in message

    def test_backoff_can_be_disabled(self):
        runner = _OomRunner(limit=1, batch_size=4, oom_backoff=False)
        with pytest.raises(RuntimeError, match="out of memory"):
            runner._generate_batch_resilient(["a", "b", "c", "d"])
        assert runner.attempts == [4]

    def test_non_oom_errors_are_not_retried(self):
        class _Broken(_OomRunner):
            def generate_batch(self, prompts):
                self.attempts.append(len(prompts))
                raise RuntimeError("shape mismatch")

        runner = _Broken(limit=1, batch_size=4)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            runner._generate_batch_resilient(["a", "b"])
        assert runner.attempts == [2]

    def test_eager_gets_the_sdpa_remedy_first(self):
        runner = _OomRunner(limit=0, batch_size=1, attn_implementation="eager")
        with pytest.raises(RuntimeError, match="--attn sdpa"):
            runner._generate_batch_resilient(["a"])

    def test_sdpa_does_not_suggest_switching_to_sdpa(self):
        runner = _OomRunner(limit=0, batch_size=1, attn_implementation="sdpa")
        with pytest.raises(RuntimeError) as excinfo:
            runner._generate_batch_resilient(["a"])
        assert "--attn sdpa" not in str(excinfo.value)
