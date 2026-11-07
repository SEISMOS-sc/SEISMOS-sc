"""Generator health checks: dtype kwarg selection, bf16 rejection on non-bf16 GPUs,
and the degenerate-first-batch abort."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from seismos_rag.generation.llm_runner import (  # noqa: E402
    DEGENERATE_MIN_TOKENS,
    DEGENERATE_TOKEN_SHARE,
    LLMRunner,
    _describe_param_dtypes,
    _dominant_token_share,
    _dtype_kwarg_name,
    is_degenerate,
)


class TestDtypeKwarg:
    def test_matches_the_installed_transformers(self):
        from transformers import __version__ as version

        major, minor = (int(p) for p in version.split(".")[:2])
        expected = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        assert _dtype_kwarg_name() == expected

    def test_returns_one_of_the_two_valid_names(self):
        assert _dtype_kwarg_name() in ("dtype", "torch_dtype")


class TestDominantTokenShare:
    def test_all_one_token_is_full_share(self):
        share, token_id, n = _dominant_token_share(torch.zeros(64, dtype=torch.long), None)
        assert share == 1.0
        assert token_id == 0
        assert n == 64

    def test_diverse_output_has_low_share(self):
        share, _, n = _dominant_token_share(torch.arange(64, dtype=torch.long), None)
        assert share == pytest.approx(1 / 64)
        assert n == 64

    def test_padding_is_excluded(self):
        # 8 real tokens (all id 5) followed by 56 pads: share must be 1.0, not 1/8.
        row = torch.cat([torch.full((8,), 5), torch.full((56,), 99)]).long()
        share, token_id, n = _dominant_token_share(row, pad_token_id=99)
        assert share == 1.0
        assert token_id == 5
        assert n == 8

    def test_all_padding_returns_zero(self):
        share, token_id, n = _dominant_token_share(torch.full((16,), 99).long(), 99)
        assert share == 0.0
        assert token_id is None
        assert n == 0


class TestDegeneracyRule:
    """The length floor exists because the guard produced false positives."""

    def test_long_output_of_one_token_is_degenerate(self):
        # The real failure: 128 tokens of id 0.
        share, _, n = _dominant_token_share(torch.zeros(128, dtype=torch.long), None)
        assert is_degenerate(share, n)

    def test_interleaved_garbage_is_degenerate(self):
        """The observed pattern: real tokens with id 0 injected between them."""
        row = torch.tensor([101, 0, 102, 0, 0, 0, 103, 0, 0, 0, 0, 0, 104, 0]).long()
        share, token_id, n = _dominant_token_share(row, pad_token_id=None)
        assert token_id == 0
        assert is_degenerate(share, n)

    def test_single_token_answer_is_not_degenerate(self):
        """REGRESSION: 'Charles' is a valid NarrativeQA answer and 100% dominant."""
        share, _, n = _dominant_token_share(torch.tensor([64084]).long(), None)
        assert share == 1.0
        assert not is_degenerate(share, n)

    def test_smoke_test_reply_is_not_degenerate(self):
        """REGRESSION: 'Hello' + EOS is 50% dominant and perfectly correct."""
        share, _, n = _dominant_token_share(torch.tensor([9707, 151645]).long(), None)
        assert share == 0.5
        assert not is_degenerate(share, n)

    def test_short_repetitive_answer_is_not_flagged(self):
        row = torch.tensor([1, 1, 1, 1, 1, 1]).long()
        share, _, n = _dominant_token_share(row, None)
        assert not is_degenerate(share, n)

    def test_healthy_long_answer_is_not_flagged(self):
        row = torch.arange(40, dtype=torch.long) % 17
        share, _, n = _dominant_token_share(row, None)
        assert not is_degenerate(share, n)

    def test_the_floor_is_the_documented_boundary(self):
        assert not is_degenerate(1.0, DEGENERATE_MIN_TOKENS - 1)
        assert is_degenerate(1.0, DEGENERATE_MIN_TOKENS)

    def test_share_below_threshold_is_never_degenerate(self):
        assert not is_degenerate(DEGENERATE_TOKEN_SHARE - 0.01, 1000)


class TestFirstBatchGuard:
    def _runner(self):
        runner = LLMRunner(model_key="qwen")

        class _Tok:
            def decode(self, ids, **kwargs):
                return "!" if list(ids) == [0] else "?"

        runner._tokenizer = _Tok()
        return runner

    def test_degenerate_first_batch_aborts(self):
        batch = [
            {"degenerate": True, "dominant_token_share": 0.95, "dominant_token_id": 0,
             "dominant_token_count": 128, "answer": "The! manager! has!!!!! an!!! affair!"},
            {"degenerate": True, "dominant_token_share": 1.0, "dominant_token_id": 0,
             "dominant_token_count": 128, "answer": "!!!!!!!!!!!!"},
        ]
        with pytest.raises(RuntimeError) as excinfo:
            self._runner()._check_first_batch(batch)

        message = str(excinfo.value)
        assert "degenerate" in message
        assert "token id 0" in message
        # The remedy must be actionable and must mention clearing the checkpoint,
        # or a resumed run silently reuses the broken answers.
        assert "generations.jsonl" in message
        assert "bfloat16" in message

    def test_healthy_first_batch_passes(self):
        batch = [
            {"degenerate": False, "dominant_token_share": 0.12, "dominant_token_id": 264,
             "dominant_token_count": 40, "answer": "The manager has an affair with Lisa."}
        ]
        self._runner()._check_first_batch(batch)

    def test_batch_of_short_correct_answers_passes(self):
        """REGRESSION: a batch of one-word answers must not abort the run."""
        batch = [
            {"degenerate": False, "dominant_token_share": 1.0, "dominant_token_id": 64084,
             "dominant_token_count": 1, "answer": "Charles"},
            {"degenerate": False, "dominant_token_share": 1.0, "dominant_token_id": 88,
             "dominant_token_count": 1, "answer": "Virginie"},
        ]
        self._runner()._check_first_batch(batch)

    def test_empty_batch_passes(self):
        self._runner()._check_first_batch([])


class TestBf16Guard:
    def test_rejects_bf16_params_when_gpu_lacks_bf16(self, monkeypatch):
        import seismos_rag.generation.llm_runner as runner_module

        monkeypatch.setattr(runner_module, "supports_bfloat16", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        model = torch.nn.Linear(4, 4).to(torch.bfloat16)
        with pytest.raises(RuntimeError, match="bfloat16"):
            runner_module._assert_no_bf16_on_unsupported_gpu(model)

    def test_accepts_fp16_params(self, monkeypatch):
        import seismos_rag.generation.llm_runner as runner_module

        monkeypatch.setattr(runner_module, "supports_bfloat16", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        runner_module._assert_no_bf16_on_unsupported_gpu(
            torch.nn.Linear(4, 4).to(torch.float16)
        )

    def test_allows_bf16_when_the_gpu_supports_it(self, monkeypatch):
        import seismos_rag.generation.llm_runner as runner_module

        monkeypatch.setattr(runner_module, "supports_bfloat16", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        runner_module._assert_no_bf16_on_unsupported_gpu(
            torch.nn.Linear(4, 4).to(torch.bfloat16)
        )

    def test_no_gpu_is_a_no_op(self, monkeypatch):
        import seismos_rag.generation.llm_runner as runner_module

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        runner_module._assert_no_bf16_on_unsupported_gpu(
            torch.nn.Linear(4, 4).to(torch.bfloat16)
        )


class TestDtypeReporting:
    def test_histogram_names_the_dtypes(self):
        described = _describe_param_dtypes(torch.nn.Linear(4, 4).to(torch.float16))
        assert "float16" in described

    def test_mixed_dtypes_are_all_reported(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4).to(torch.float16),
            torch.nn.Linear(4, 4).to(torch.bfloat16),
        )
        described = _describe_param_dtypes(model)
        assert "float16" in described and "bfloat16" in described


class TestDeviceMap:
    def test_multi_gpu_shards_automatically(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        runner = LLMRunner(model_key="qwen", device="cpu", device_map="auto")
        assert runner._resolve_device_map() == "auto"

    def test_single_gpu_pins_to_device_zero(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        runner = LLMRunner(model_key="qwen", device="cpu", device_map="auto")
        assert runner._resolve_device_map() == {"": 0}

    def test_explicit_device_map_is_respected(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        runner = LLMRunner(model_key="qwen", device="cpu", device_map={"": 1})
        assert runner._resolve_device_map() == {"": 1}

    def test_device_map_is_recorded_in_provenance(self):
        assert LLMRunner(model_key="qwen", device="cpu").describe()["device_map"] == "auto"


class TestSmokePrompt:
    def test_falls_back_without_a_chat_template(self):
        from seismos_rag.generation.llm_runner import _render_smoke_prompt
        from tests.conftest import FakeTokenizer

        rendered = _render_smoke_prompt(FakeTokenizer())
        assert rendered.rstrip().endswith("Answer:")

    def test_prompt_elicits_a_multi_token_answer(self):
        """A one-word prompt cannot be distinguished from garbage; don't use one."""
        from seismos_rag.generation.llm_runner import _render_smoke_prompt
        from tests.conftest import FakeTokenizer

        rendered = _render_smoke_prompt(FakeTokenizer()).lower()
        assert "one word" not in rendered
        assert "count" in rendered


class TestQuantizationConsistency:
    def test_compute_dtype_matches_the_model_dtype(self):
        pytest.importorskip("bitsandbytes")
        runner = LLMRunner(model_key="qwen", dtype="float16", quantization="4bit")
        config = runner._quantization_config()
        assert config.bnb_4bit_compute_dtype == runner._resolve_dtype()

    def test_no_quantization_returns_no_config(self):
        assert LLMRunner(model_key="qwen", quantization="none")._quantization_config() is None
