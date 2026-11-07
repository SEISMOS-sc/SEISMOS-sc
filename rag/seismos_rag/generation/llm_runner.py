"""Generator loading and greedy batch decoding.

Greedy decoding is deterministic for a fixed batch composition, so
``generation.batch_size`` is part of the recorded configuration. On T4 (sm_75)
the dtype is fp16 and attention defaults to SDPA, which avoids materialising
the full score matrix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..utils.gpu import (
    log_gpu_memory,
    peak_memory_gb,
    preflight,
    release,
    reset_peak_memory,
    resolve_device,
    supports_bfloat16,
    supports_flash_attention_2,
    supports_sdpa,
)
from ..utils.logging import get_logger
from .prompt import BuiltPrompt, build_prompt_with_budget

#: Accepted values for ``generation.attn_implementation``. ``auto`` resolves to
#: the cheapest backend this GPU can actually run.
ATTN_IMPLEMENTATIONS = ("auto", "sdpa", "eager", "flash_attention_2")


def resolve_attn_implementation(requested: str = "auto") -> str:
    """Map the requested attention backend to one available here. ``auto`` -> ``sdpa``."""
    log = get_logger()
    if requested not in ATTN_IMPLEMENTATIONS:
        raise ValueError(
            f"unknown attn_implementation {requested!r}; expected one of "
            f"{list(ATTN_IMPLEMENTATIONS)}"
        )

    if requested == "flash_attention_2":
        if supports_flash_attention_2():
            return "flash_attention_2"
        log.warning(
            "flash_attention_2 requested but unavailable (needs sm_80+ and the "
            "flash-attn package); falling back to sdpa"
        )
        requested = "auto"

    if requested == "eager":
        log.warning(
            "attn_implementation=eager materialises a [batch, heads, len, len] "
            "score matrix per layer (fp16 scores + an fp32 softmax copy). At "
            "batch 4 and 3k tokens that is several GB of transient. Use 'sdpa' "
            "unless you specifically need the reference kernel."
        )
        return "eager"

    if supports_sdpa():
        return "sdpa"
    log.warning(
        "this torch build has no scaled_dot_product_attention; falling back to eager"
    )
    return "eager"


def _is_cuda_oom(exc: BaseException) -> bool:
    """Matches CUDA OOM across torch versions."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return False

    candidates = tuple(
        cls
        for cls in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(cls, type)
    )
    if candidates and isinstance(exc, candidates):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _dtype_kwarg_name() -> str:
    """``torch_dtype`` before transformers 4.56, ``dtype`` after. ``from_pretrained``
    silently ignores the wrong one, hence detection rather than try/except.
    """
    try:
        from transformers import __version__ as version

        major, minor = (int(part) for part in version.split(".")[:2])
    except Exception:  # pragma: no cover - unparseable dev version
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def _assert_no_bf16_on_unsupported_gpu(model) -> None:
    """Raise if any parameter is bf16 on a GPU without bf16 support."""
    import torch

    if not torch.cuda.is_available() or supports_bfloat16():
        return

    offenders = [
        name for name, param in model.named_parameters() if param.dtype == torch.bfloat16
    ]
    if not offenders:
        return

    raise RuntimeError(
        f"{len(offenders)} parameter tensor(s) loaded as bfloat16 on a GPU without "
        f"bfloat16 support (e.g. {offenders[:3]}). This produces corrupted logits, "
        "not an error: decoding collapses to token id 0. Pass a float16 dtype to "
        "from_pretrained on every load path, including the quantized one."
    )


# Degenerate: one token id is at least this share of an output...
DEGENERATE_TOKEN_SHARE = 0.5

# ...of at least this many tokens (short answers are legitimately one token).
DEGENERATE_MIN_TOKENS = 12


def _dominant_token_share(row, pad_token_id):
    """Return ``(share, token_id, n_tokens)`` for the most frequent output token."""
    import torch

    real = row[row != pad_token_id] if pad_token_id is not None else row
    if real.numel() == 0:
        return 0.0, None, 0
    values, counts = torch.unique(real, return_counts=True)
    best = int(torch.argmax(counts))
    return (
        float(counts[best].item() / real.numel()),
        int(values[best].item()),
        int(real.numel()),
    )


def is_degenerate(share: float, n_tokens: int) -> bool:
    """A long output dominated by a single token id."""
    return n_tokens >= DEGENERATE_MIN_TOKENS and share >= DEGENERATE_TOKEN_SHARE


def _render_smoke_prompt(tokenizer) -> str:
    """A prompt with a multi-token answer, so degeneracy is detectable."""
    body = "Count from one to ten, writing each number as a word."
    if getattr(tokenizer, "chat_template", None) is None:
        return body + "\nAnswer:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True
    )


def _describe_param_dtypes(model) -> str:
    """Histogram of parameter dtypes, logged so a mismatch is visible up front."""
    from collections import Counter

    histogram = Counter(str(p.dtype).replace("torch.", "") for p in model.parameters())
    return ", ".join(f"{name}={count}" for name, count in histogram.most_common())

GENERATOR_REGISTRY: Dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


@dataclass
class Generation:
    qid: str
    chunker: str
    prompt: str
    answer: str
    input_tokens: int
    output_tokens: int
    truncated: bool
    context_tokens: int
    dropped_context_tokens: int
    seconds: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "chunker": self.chunker,
            "prompt": self.prompt,
            "answer": self.answer,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "prompt_truncated": self.truncated,
            "context_tokens": self.context_tokens,
            "dropped_context_tokens": self.dropped_context_tokens,
            "generation_seconds": self.seconds,
            "diagnostics": self.diagnostics,
        }


class LLMRunner:
    """Loads a causal LM and decodes greedily in batches."""

    def __init__(
        self,
        model_key: str = "qwen",
        dtype: str = "float16",
        quantization: str = "none",
        device: str = "auto",
        max_new_tokens: int = 128,
        batch_size: int = 4,
        max_input_tokens: int = 3072,
        attn_implementation: str = "auto",
        trust_remote_code: bool = False,
        preflight_check: bool = True,
        device_map: Any = "auto",
        oom_backoff: bool = True,
    ) -> None:
        self.model_key = model_key
        self.model_name = GENERATOR_REGISTRY.get(model_key, model_key)
        self.dtype = dtype
        self.quantization = quantization
        self.device = resolve_device(device)
        self.device_map = device_map
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.max_input_tokens = int(max_input_tokens)
        self.attn_implementation = attn_implementation
        self.trust_remote_code = bool(trust_remote_code)
        self.preflight_check = preflight_check
        self.oom_backoff = bool(oom_backoff)

        self._model = None
        self._tokenizer = None
        self._resolved_attn: Optional[str] = None
        self._oom_events = 0

    # -- lifecycle ------------------------------------------------------------

    @property
    def resolved_attn_implementation(self) -> str:
        """The backend actually passed to ``from_pretrained``. Cached, so the
        warning that comes with a downgrade is logged once per run."""
        if self._resolved_attn is None:
            self._resolved_attn = resolve_attn_implementation(self.attn_implementation)
        return self._resolved_attn

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        # Decoder-only batching needs left padding.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _resolve_dtype(self):
        import torch

        if self.dtype == "bfloat16" and not supports_bfloat16():
            get_logger().warning(
                "bfloat16 requested but this GPU (sm_75 or older) does not "
                "support it; falling back to float16"
            )
            return torch.float16
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            self.dtype
        ]

    def _quantization_config(self):
        if self.quantization == "none":
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"quantization={self.quantization} needs bitsandbytes:\n"
                "  pip install bitsandbytes accelerate"
            ) from exc

        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # Must match the dtype the rest of the model is loaded in, or
                # activations cross a dtype boundary every layer.
                bnb_4bit_compute_dtype=self._resolve_dtype(),
            )
        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"unknown quantization: {self.quantization}")

    def _resolve_device_map(self):
        """Shard across all visible GPUs when there is more than one."""
        import torch

        if self.device_map != "auto":
            return self.device_map

        count = torch.cuda.device_count()
        if count > 1:
            get_logger().info("sharding generator across %d GPUs (device_map=auto)", count)
            return "auto"
        return {"": 0}

    def _smoke_test(self, model) -> None:
        """Generate a few tokens right after loading and reject degenerate output."""
        import torch

        tokenizer = self.tokenizer
        prompt = _render_smoke_prompt(tokenizer)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=48, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        share, token_id, n_tokens = _dominant_token_share(generated, tokenizer.pad_token_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)

        log = get_logger()
        log.info("generator smoke test (%d tokens): %r", n_tokens, text)

        if not text.strip():
            raise RuntimeError(
                "generator smoke test failed: the model produced no text at all. "
                "Diagnose with: python tools/diagnose_generation.py"
            )

        if not is_degenerate(share, n_tokens):
            return

        rendering = tokenizer.decode([token_id]) if token_id is not None else "?"
        raise RuntimeError(
            f"generator smoke test failed: token id {token_id} ({rendering!r}) is "
            f"{share:.0%} of a {n_tokens}-token output for a trivial prompt "
            f"(got {text!r}).\n\n"
            "The model is numerically broken at load time, before any of this "
            "pipeline's prompts are involved. Diagnose with:\n"
            "    python tools/diagnose_generation.py\n"
            "which tests fp16 and 4-bit separately and reports library versions."
        )

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM

        log = get_logger()
        attn = self.resolved_attn_implementation
        if self.preflight_check and self.device == "cuda":
            preflight(
                self.model_name,
                self.dtype,
                self.quantization,
                batch_size=self.batch_size,
                max_input_tokens=self.max_input_tokens,
                max_new_tokens=self.max_new_tokens,
                attn_implementation=attn,
            )

        described_attn = (
            attn if attn == self.attn_implementation
            else f"{attn} (requested {self.attn_implementation})"
        )
        log.info(
            "loading generator %s (dtype=%s quantization=%s attn=%s) on %s",
            self.model_name,
            self.dtype,
            self.quantization,
            described_attn,
            self.device,
        )
        reset_peak_memory()
        start = time.perf_counter()

        kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "attn_implementation": attn,
        }

        # Set dtype even when quantizing: bitsandbytes leaves non-Linear modules in
        # the checkpoint's bf16, which breaks on GPUs without bf16.
        resolved_dtype = self._resolve_dtype()
        kwargs[_dtype_kwarg_name()] = resolved_dtype

        quant_config = self._quantization_config()
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        if self.device == "cuda":
            kwargs["device_map"] = self._resolve_device_map()

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        _assert_no_bf16_on_unsupported_gpu(model)

        if quant_config is None and self.device == "cpu":
            model = model.to("cpu")
        model.eval()

        # Greedy decoding must not be overridden by the checkpoint's shipped
        # generation_config (Qwen ships temperature/top_p/top_k defaults).
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        log.info("generator param dtypes: %s", _describe_param_dtypes(model))
        self._smoke_test(model)
        log.info(
            "generator ready in %.1fs | peak load memory %.2f GB",
            time.perf_counter() - start,
            peak_memory_gb() or 0.0,
        )
        log_gpu_memory("after generator load:")
        return model

    def unload(self) -> None:
        if self._model is not None:
            release(self._model)
            self._model = None
            self._tokenizer = None
            get_logger().info("generator unloaded")
            log_gpu_memory("after generator unload:")

    # -- prompt building ------------------------------------------------------

    def build(self, context: str, question: str) -> BuiltPrompt:
        return build_prompt_with_budget(
            context=context,
            question=question,
            tokenizer=self.tokenizer,
            max_input_tokens=self.max_input_tokens,
            chat_template=True,
        )

    # -- generation -----------------------------------------------------------

    def generate_batch(self, prompts: Sequence[str]) -> List[Dict[str, Any]]:
        """Greedy-decode a batch. Returns one dict per prompt."""
        import torch

        if not prompts:
            return []

        tokenizer = self.tokenizer
        model = self.model

        encoded = tokenizer(
            list(prompts), return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)

        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy: this is what temperature=0 means
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start

        prompt_length = encoded["input_ids"].shape[1]
        generated = output[:, prompt_length:]

        results: List[Dict[str, Any]] = []
        for row, mask in zip(generated, encoded["attention_mask"]):
            text = tokenizer.decode(row, skip_special_tokens=True).strip()
            output_tokens = int((row != tokenizer.pad_token_id).sum().item())
            share, token_id, n_tokens = _dominant_token_share(row, tokenizer.pad_token_id)
            results.append(
                {
                    "answer": text,
                    "input_tokens": int(mask.sum().item()),
                    "output_tokens": output_tokens,
                    "seconds": elapsed / len(prompts),
                    "dominant_token_share": share,
                    "dominant_token_id": token_id,
                    "dominant_token_count": n_tokens,
                    "degenerate": is_degenerate(share, n_tokens),
                    "effective_batch_size": len(prompts),
                }
            )
        return results

    def _generate_batch_resilient(self, prompts: Sequence[str]) -> List[Dict[str, Any]]:
        """``generate_batch`` that halves the batch on CUDA OOM. Each result records its
        ``effective_batch_size``. Disable with ``generation.oom_backoff: false``.
        """
        prompts = list(prompts)
        try:
            return self.generate_batch(prompts)
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            self._oom_events += 1
            # Drop the failed forward pass's cached blocks before retrying;
            # without this the retry inherits the fragmentation that caused it.
            release()
            if not self.oom_backoff or len(prompts) <= 1:
                raise RuntimeError(self._oom_message(prompts)) from exc

            half = len(prompts) // 2
            get_logger().warning(
                "CUDA OOM at batch size %d; retrying as %d + %d. The affected "
                "generations are recorded with their reduced effective_batch_size.",
                len(prompts),
                half,
                len(prompts) - half,
            )
            return self._generate_batch_resilient(prompts[:half]) + (
                self._generate_batch_resilient(prompts[half:])
            )

    def _oom_message(self, prompts: Sequence[str]) -> str:
        longest = max((self._count_tokens(p) for p in prompts), default=0)
        attn = self.resolved_attn_implementation
        lines = [
            f"CUDA out of memory generating a batch of {len(prompts)} "
            f"(longest prompt {longest} tokens, max_input_tokens="
            f"{self.max_input_tokens}, attn={attn}).",
            "",
            "Completed units are already in results/checkpoints/generations.jsonl, "
            "so a re-run resumes rather than restarting. Try, in order:",
        ]
        if attn == "eager":
            lines.append(
                "  1. --attn sdpa   eager builds a [batch, heads, len, len] score "
                "matrix per layer; SDPA does not, and it works on T4"
            )
        if self.batch_size > 1:
            lines.append(f"  - --gen_batch_size {max(1, self.batch_size // 2)}")
        lines.append(f"  - --max_input_tokens {max(512, self.max_input_tokens // 2)}")
        lines.append("  - --quantization 4bit")
        lines.append("  - --generator qwen2.5-3b")
        return "\n".join(lines)

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:  # pragma: no cover - diagnostics must never mask the OOM
            return -1

    def generate(
        self,
        items: Sequence[Dict[str, Any]],
        on_result=None,
        progress: bool = True,
    ) -> List[Generation]:
        """Generate for ``{qid, chunker, context, question}`` items; ``on_result`` is called per item."""
        log = get_logger()
        outputs: List[Generation] = []

        built = [self.build(item["context"], item["question"]) for item in items]
        truncated = sum(1 for b in built if b.truncated)
        if truncated:
            log.warning(
                "%d/%d prompts exceeded max_input_tokens=%d and had their context "
                "middle-truncated (recorded as prompt_truncated)",
                truncated,
                len(built),
                self.max_input_tokens,
            )

        # Fixed-order (not length-sorted) batches, so reruns batch identically.
        iterator = range(0, len(items), self.batch_size)
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(list(iterator), desc="generating", ncols=80)
            except ImportError:
                pass

        checked_first_batch = False
        degenerate_total = 0

        for offset in iterator:
            window = slice(offset, offset + self.batch_size)
            batch_items = items[window]
            batch_built = built[window]
            batch = self._generate_batch_resilient([b.text for b in batch_built])

            if not checked_first_batch:
                self._check_first_batch(batch)
                checked_first_batch = True
            degenerate_total += sum(1 for r in batch if r["degenerate"])

            for item, prompt, result in zip(batch_items, batch_built, batch):
                generation = Generation(
                    qid=item["qid"],
                    chunker=item["chunker"],
                    prompt=prompt.text,
                    answer=result["answer"],
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    truncated=prompt.truncated,
                    context_tokens=prompt.context_tokens,
                    dropped_context_tokens=prompt.dropped_context_tokens,
                    seconds=result["seconds"],
                    diagnostics={
                        "batch_size": self.batch_size,
                        "effective_batch_size": result.get(
                            "effective_batch_size", self.batch_size
                        ),
                        "max_new_tokens": self.max_new_tokens,
                        "decoding": "greedy",
                        "dominant_token_share": round(result["dominant_token_share"], 4),
                        "degenerate": result["degenerate"],
                    },
                )
                outputs.append(generation)
                if on_result is not None:
                    on_result(generation)

        if self._oom_events:
            log.warning(
                "%d CUDA OOM event(s) were absorbed by halving the batch. The run "
                "completed, but the affected units were decoded at a smaller batch "
                "than generation.batch_size=%d (see effective_batch_size in "
                "raw_generations.csv). Lower --gen_batch_size for a clean re-run.",
                self._oom_events,
                self.batch_size,
            )

        if degenerate_total:
            # After the first batch, warn instead of aborting.
            log.warning(
                "%d/%d generations were degenerate (one token id >= %.0f%% of output). "
                "Inspect raw_generations.csv before trusting the metrics.",
                degenerate_total,
                len(items),
                DEGENERATE_TOKEN_SHARE * 100,
            )

        return outputs

    def _check_first_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Abort if the first batch is degenerate (one token id dominating the output)."""
        degenerate = [r for r in batch if r["degenerate"]]
        if not degenerate:
            return

        worst = max(degenerate, key=lambda r: r["dominant_token_share"])
        token_id = worst["dominant_token_id"]
        rendering = self.tokenizer.decode([token_id]) if token_id is not None else "?"
        raise RuntimeError(
            f"{len(degenerate)}/{len(batch)} generations in the first batch are "
            f"degenerate: token id {token_id} ({rendering!r}) accounts for "
            f"{worst['dominant_token_share']:.0%} of a "
            f"{worst['dominant_token_count']}-token output.\n\n"
            "This is a numerical fault, not a modelling result. Most likely causes:\n"
            "  1. Model parameters loaded in bfloat16 on a GPU without bf16 support "
            "(T4/V100). Check the 'param dtypes' line in the log.\n"
            "  2. bnb_4bit_compute_dtype not matching the model dtype.\n"
            "  3. A bitsandbytes build mismatched to the installed CUDA.\n\n"
            "Sample output:\n"
            f"  {worst.get('answer', '')[:200]!r}\n\n"
            "Fix the cause, then DELETE results/checkpoints/generations.jsonl "
            "before re-running: resume would otherwise reuse the broken answers."
        )

    # -- used by the optional LLM chunker -------------------------------------

    def binary_choice(
        self,
        prompts: Sequence[str],
        positive: str = "Yes",
        negative: str = "No",
        batch_size: Optional[int] = None,
    ) -> List[bool]:
        """Compare first-token logits for two answers. No decoding, no sampling."""
        import torch

        tokenizer = self.tokenizer
        model = self.model
        size = batch_size or self.batch_size

        pos_id = tokenizer.encode(positive, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(negative, add_special_tokens=False)[0]

        decisions: List[bool] = []
        for offset in range(0, len(prompts), size):
            batch = list(prompts[offset : offset + size])
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            with torch.inference_mode():
                logits = model(**encoded).logits[:, -1, :]
            decisions.extend((logits[:, pos_id] > logits[:, neg_id]).tolist())
        return decisions

    def describe(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model": self.model_name,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "max_input_tokens": self.max_input_tokens,
            "attn_implementation": self.attn_implementation,
            "attn_implementation_resolved": self.resolved_attn_implementation,
            "device_map": self.device_map,
            "oom_backoff": self.oom_backoff,
            "decoding": "greedy (do_sample=False, num_beams=1)",
        }


def build_llm_runner(cfg) -> LLMRunner:
    return LLMRunner(
        model_key=cfg["generation.model"],
        dtype=cfg["generation.dtype"],
        quantization=cfg["generation.quantization"],
        device=cfg["runtime.device"],
        max_new_tokens=cfg["generation.max_new_tokens"],
        batch_size=cfg["generation.batch_size"],
        max_input_tokens=cfg["generation.max_input_tokens"],
        attn_implementation=cfg["generation.attn_implementation"],
        trust_remote_code=cfg["generation.trust_remote_code"],
        device_map=cfg.get("generation.device_map", "auto"),
        preflight_check=cfg.get("generation.preflight", True),
        oom_backoff=cfg.get("generation.oom_backoff", True),
    )
