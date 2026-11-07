#!/usr/bin/env python
"""Load a generator under each dtype/quantization setting and report which produce
sane output.

    python tools/diagnose_generation.py --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from typing import Any, Dict, List, Optional

PROMPT = "Reply with exactly one word: hello"


def report_environment() -> None:
    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)

    from importlib import metadata

    for package in (
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentence-transformers",
        "datasets",
    ):
        try:
            print(f"  {package:24s} {metadata.version(package)}")
        except Exception:
            print(f"  {package:24s} NOT INSTALLED")

    try:
        import torch

        print(f"  {'torch.version.cuda':24s} {torch.version.cuda}")
        print(f"  {'cuda available':24s} {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                major, minor = torch.cuda.get_device_capability(i)
                free, total = torch.cuda.mem_get_info(i)
                print(
                    f"    GPU {i}: {name} sm_{major}{minor} "
                    f"{free / 1024**3:.1f}/{total / 1024**3:.1f} GB free  "
                    f"bf16={'yes' if major >= 8 else 'NO'}"
                )
    except Exception as exc:
        print(f"  torch inspection failed: {exc}")
    print()


def _free() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def try_config(model_name: str, label: str, **load_kwargs) -> Dict[str, Any]:
    """Load under one configuration, generate, and report."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("-" * 72)
    print(f"CONFIG: {label}")
    print("-" * 72)

    result: Dict[str, Any] = {"label": label, "ok": False, "error": None, "text": None}
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()

        from collections import Counter

        dtypes = Counter(str(p.dtype).replace("torch.", "") for p in model.parameters())
        print(f"  param dtypes: {dict(dtypes)}")
        print(f"  device map:   {getattr(model, 'hf_device_map', 'n/a')}")

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=16, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        values, counts = torch.unique(generated, return_counts=True)
        best = int(torch.argmax(counts))
        share = float(counts[best].item() / generated.numel())

        result["text"] = text
        result["dominant_share"] = share
        result["dominant_token"] = int(values[best].item())
        result["ok"] = share < 0.5

        verdict = "OK" if result["ok"] else "DEGENERATE"
        print(f"  output:  {text!r}")
        print(
            f"  verdict: {verdict}  (token {result['dominant_token']} is "
            f"{share:.0%} of output)"
        )

    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED TO RUN: {result['error']}")
        traceback.print_exc(limit=3)
    finally:
        del model
        _free()
        print()

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--skip-fp16",
        action="store_true",
        help="skip the unquantized test (use when total VRAM is clearly too small)",
    )
    args = parser.parse_args()

    report_environment()

    import torch
    from transformers import __version__ as tv

    major, minor = (int(p) for p in tv.split(".")[:2])
    dtype_kwarg = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    print(f"using '{dtype_kwarg}=' for this transformers version\n")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    results: List[Dict[str, Any]] = []

    if not args.skip_fp16:
        results.append(
            try_config(
                args.model,
                f"fp16, device_map=auto across {n_gpus} GPU(s), NO bitsandbytes",
                **{dtype_kwarg: torch.float16},
                device_map="auto",
                attn_implementation="eager",
            )
        )

    try:
        from transformers import BitsAndBytesConfig

        results.append(
            try_config(
                args.model,
                "4-bit NF4, compute dtype fp16",
                **{dtype_kwarg: torch.float16},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                device_map="auto",
                attn_implementation="eager",
            )
        )
        results.append(
            try_config(
                args.model,
                "4-bit FP4, no double quant (isolates the NF4 kernel)",
                **{dtype_kwarg: torch.float16},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                device_map="auto",
                attn_implementation="eager",
            )
        )
    except ImportError:
        print("bitsandbytes not installed; skipped the quantized configurations\n")

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for result in results:
        status = "OK        " if result["ok"] else ("ERROR     " if result["error"] else "DEGENERATE")
        print(f"  [{status}] {result['label']}")
        if result["error"]:
            print(f"              {result['error']}")
        elif result["text"] is not None:
            print(f"              {result['text']!r}")

    working = [r for r in results if r["ok"]]
    print()
    if not working:
        print("  Nothing worked. Try a smaller model to separate 'model too big' from")
        print("  'library broken':  python diagnose_generation.py --model Qwen/Qwen2.5-1.5B-Instruct")
        return 1

    print(f"  Use this configuration: {working[0]['label']}")
    if "NO bitsandbytes" in working[0]["label"]:
        print("  -> run the pipeline with --quantization none")
    elif "NF4" in working[0]["label"]:
        print("  -> run the pipeline with --quantization 4bit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
