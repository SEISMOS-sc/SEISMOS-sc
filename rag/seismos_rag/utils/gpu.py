"""GPU memory estimates, preflight checks and model unloading."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Optional

from .logging import get_logger

# Approximate weight footprints in GB, for the preflight check only. Derived as
# params * bytes_per_param; KV cache and activations are added separately.
KNOWN_MODEL_WEIGHT_GB = {
    ("Qwen/Qwen2.5-7B-Instruct", "float16"): 15.2,
    ("Qwen/Qwen2.5-7B-Instruct", "4bit"): 5.0,
    ("Qwen/Qwen2.5-7B-Instruct", "8bit"): 8.5,
    ("Qwen/Qwen2.5-3B-Instruct", "float16"): 6.2,
    ("Qwen/Qwen2.5-3B-Instruct", "4bit"): 2.2,
    ("Qwen/Qwen2.5-3B-Instruct", "8bit"): 3.6,
    ("meta-llama/Llama-3.1-8B-Instruct", "float16"): 16.1,
    ("meta-llama/Llama-3.1-8B-Instruct", "4bit"): 5.4,
    ("meta-llama/Llama-3.1-8B-Instruct", "8bit"): 9.0,
}

# Architecture constants for KV-cache and attention-workspace estimates.
KNOWN_MODEL_SHAPES = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "layers": 28, "heads": 28, "kv_heads": 4, "head_dim": 128, "hidden": 3584,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "layers": 36, "heads": 16, "kv_heads": 2, "head_dim": 128, "hidden": 2048,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128, "hidden": 4096,
    },
}

# Transient bytes per attention-score element under eager attention
# (fp16 scores, mask, fp32 softmax copy).
EAGER_ATTN_BYTES_PER_ELEMENT = 10

# SDPA peak, in copies of the [batch, seq, hidden] tensor.
SDPA_ACTIVATION_COPIES = 12


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def cuda_available() -> bool:
    if not torch_available():
        return False
    import torch

    return torch.cuda.is_available()


def resolve_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    if requested == "cuda" and not cuda_available():
        get_logger().warning("device='cuda' requested but CUDA is unavailable; using CPU")
        return "cpu"
    return requested


@dataclass
class GpuMemory:
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float

    def __str__(self) -> str:
        return (
            f"GPU {self.allocated_gb:.2f}/{self.total_gb:.2f} GB allocated "
            f"({self.reserved_gb:.2f} reserved, {self.free_gb:.2f} free)"
        )


def gpu_memory(all_devices: bool = False) -> Optional[GpuMemory]:
    """Memory on device 0, or summed over all GPUs."""
    if not cuda_available():
        return None
    import torch

    gb = 1024**3
    devices = range(torch.cuda.device_count()) if all_devices else [0]

    free_total = allocated = reserved = capacity = 0
    for index in devices:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        free_total += free_bytes
        capacity += total_bytes
        allocated += torch.cuda.memory_allocated(index)
        reserved += torch.cuda.memory_reserved(index)

    return GpuMemory(
        total_gb=capacity / gb,
        allocated_gb=allocated / gb,
        reserved_gb=reserved / gb,
        free_gb=free_total / gb,
    )


def per_device_free_gb() -> list:
    if not cuda_available():
        return []
    import torch

    gb = 1024**3
    return [
        torch.cuda.mem_get_info(index)[0] / gb for index in range(torch.cuda.device_count())
    ]


def gpu_memory_str() -> str:
    mem = gpu_memory()
    return str(mem) if mem else "GPU n/a (cpu)"


def log_gpu_memory(prefix: str = "") -> None:
    get_logger().info("%s%s", f"{prefix} " if prefix else "", gpu_memory_str())


def release(*objects: Any) -> None:
    """Drop the given references, collect, and empty the CUDA cache."""
    for obj in objects:
        if obj is None:
            continue
        # SentenceTransformer / HF models: move off the device first so the
        # allocator can actually reclaim the blocks.
        try:
            if hasattr(obj, "to"):
                obj.to("cpu")
        except Exception:
            pass
        del obj

    gc.collect()
    if cuda_available():
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class PreflightError(RuntimeError):
    """Raised when a stage cannot possibly fit in the available memory."""


@dataclass
class MemoryEstimate:
    """Predicted generator footprint, split into the parts that scale differently."""

    weights_gb: float
    kv_cache_gb: float
    activation_gb: float
    headroom_gb: float

    @property
    def sharded_gb(self) -> float:
        """The part ``device_map`` divides between cards."""
        return self.weights_gb + self.kv_cache_gb

    @property
    def total_gb(self) -> float:
        return self.sharded_gb + self.activation_gb + self.headroom_gb

    def per_device_gb(self, n_devices: int) -> float:
        """Weights and KV cache are sharded; activations are not."""
        return self.sharded_gb / max(1, n_devices) + self.activation_gb + self.headroom_gb

    def __str__(self) -> str:
        return (
            f"weights {self.weights_gb:.1f} + KV {self.kv_cache_gb:.1f} + "
            f"activations {self.activation_gb:.1f} + headroom {self.headroom_gb:.1f}"
        )


def estimate_generator_memory(
    model_name: str,
    dtype: str,
    quantization: str = "none",
    batch_size: int = 1,
    max_input_tokens: int = 3072,
    max_new_tokens: int = 128,
    attn_implementation: str = "sdpa",
    headroom_gb: float = 1.5,
) -> Optional[MemoryEstimate]:
    """Weights + KV cache + attention workspace, or ``None`` for unknown models."""
    key_dtype = quantization if quantization in ("4bit", "8bit") else dtype
    weights = KNOWN_MODEL_WEIGHT_GB.get((model_name, key_dtype))
    shape = KNOWN_MODEL_SHAPES.get(model_name)
    if weights is None or shape is None:
        return None

    gb = 1024**3
    sequence = int(max_input_tokens) + int(max_new_tokens)
    batch = max(1, int(batch_size))

    per_token = 2 * shape["layers"] * shape["kv_heads"] * shape["head_dim"]  # K + V
    kv_bytes = per_token * sequence * batch * 2  # 2 bytes per fp16 element

    if attn_implementation == "eager":
        elements = batch * shape["heads"] * int(max_input_tokens) ** 2
        activation_bytes = elements * EAGER_ATTN_BYTES_PER_ELEMENT
    else:
        activation_bytes = (
            batch * int(max_input_tokens) * shape["hidden"] * 2 * SDPA_ACTIVATION_COPIES
        )

    return MemoryEstimate(
        weights_gb=weights,
        kv_cache_gb=kv_bytes / gb,
        activation_gb=activation_bytes / gb,
        headroom_gb=headroom_gb,
    )


def preflight(
    model_name: str,
    dtype: str,
    quantization: str = "none",
    headroom_gb: float = 1.5,
    raise_on_fail: bool = True,
    batch_size: int = 1,
    max_input_tokens: int = 3072,
    max_new_tokens: int = 128,
    attn_implementation: str = "sdpa",
) -> Optional[str]:
    """Check aggregate free memory (weights, KV) and the smallest card (activations).
    Returns a warning or ``None``; raises :class:`PreflightError` if it cannot fit
    and ``raise_on_fail``.
    """
    log = get_logger()
    # Sum across devices: the generator is sharded with device_map, so the
    # budget is the total, not device 0's share.
    mem = gpu_memory(all_devices=True)
    if mem is None:
        return None

    key_dtype = quantization if quantization in ("4bit", "8bit") else dtype
    estimate = estimate_generator_memory(
        model_name,
        dtype,
        quantization,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        attn_implementation=attn_implementation,
        headroom_gb=headroom_gb,
    )
    if estimate is None:
        log.info(
            "preflight: no size estimate for %s @ %s; skipping check", model_name, key_dtype
        )
        return None

    import torch

    n_devices = max(1, torch.cuda.device_count())
    free_per_device = per_device_free_gb()
    smallest_free = min(free_per_device) if free_per_device else mem.free_gb

    log.info(
        "preflight: %s @ %s, %s attention, batch %d x %d tokens -> ~%.1f GB total "
        "(%s), ~%.1f GB per card across %d; %.1f GB free total, %.1f GB on the "
        "tightest card",
        model_name,
        key_dtype,
        attn_implementation,
        batch_size,
        max_input_tokens,
        estimate.total_gb,
        estimate,
        estimate.per_device_gb(n_devices),
        n_devices,
        mem.free_gb,
        smallest_free,
    )

    problems = []
    if estimate.total_gb > mem.free_gb:
        problems.append(
            f"needs ~{estimate.total_gb:.1f} GB in total but only {mem.free_gb:.1f} GB "
            f"is free across {n_devices} x {_device_name()}"
        )
    per_device = estimate.per_device_gb(n_devices)
    if per_device > smallest_free:
        problems.append(
            f"needs ~{per_device:.1f} GB on a single card (sharded weights+KV "
            f"{estimate.sharded_gb / n_devices:.1f} + activations "
            f"{estimate.activation_gb:.1f} + headroom {estimate.headroom_gb:.1f}) "
            f"but the tightest card has {smallest_free:.1f} GB free"
        )
    if not problems:
        return None

    message = f"{model_name} in {key_dtype}: " + "; and ".join(problems) + ".\n\nRemedies:\n"
    message += "\n".join(_preflight_remedies(model_name, attn_implementation, batch_size))
    if raise_on_fail:
        raise PreflightError(message)
    log.warning(message)
    return message


def _preflight_remedies(model_name: str, attn_implementation: str, batch_size: int) -> list:
    """Ordered by how much they cost the experiment, cheapest first."""
    remedies = []
    if attn_implementation == "eager":
        remedies.append(
            "  1. --attn sdpa  (eager builds a [batch, heads, len, len] score "
            "matrix per layer; SDPA's memory-efficient backend does not, and it "
            "runs on sm_75/T4 even though FlashAttention-2 does not)"
        )
    if batch_size > 1:
        remedies.append(
            f"  {len(remedies) + 1}. --gen_batch_size {max(1, batch_size // 2)}  "
            "(activations scale linearly in the batch)"
        )
    remedies.append(
        f"  {len(remedies) + 1}. --max_input_tokens 2048  (under eager attention "
        "the cost is quadratic in this number)"
    )
    remedies.append(
        f"  {len(remedies) + 1}. --quantization 4bit  (~"
        f"{KNOWN_MODEL_WEIGHT_GB.get((model_name, '4bit'), 5.0):.1f} GB of weights, "
        "needs bitsandbytes + accelerate)"
    )
    remedies.append(
        f"  {len(remedies) + 1}. --generator qwen2.5-3b, or a smaller generator"
    )
    return remedies


def _device_name() -> str:
    if not cuda_available():
        return "cpu"
    import torch

    return torch.cuda.get_device_name(0)


def supports_bfloat16() -> bool:
    """T4 (sm_75) has no bf16 support; Ampere and later do."""
    if not cuda_available():
        return False
    import torch

    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def supports_flash_attention_2() -> bool:
    """Needs sm_80+ and the flash-attn package."""
    if not supports_bfloat16():
        return False
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def supports_sdpa() -> bool:
    """PyTorch's fused attention. Present since torch 2.0 and works on sm_75."""
    try:
        from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
    except Exception:
        return False
    return True


def reset_peak_memory() -> None:
    if cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> Optional[float]:
    if not cuda_available():
        return None
    import torch

    return torch.cuda.max_memory_allocated() / (1024**3)
