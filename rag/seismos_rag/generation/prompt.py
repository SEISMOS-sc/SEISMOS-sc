"""The fixed prompt, identical for every chunker and generator.

Prompts over ``max_input_tokens`` have the middle of the context removed (never
the instruction or question); truncations are counted as ``prompt_truncated``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence

PROMPT_TEMPLATE = """You are a helpful assistant.
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:"""

#: Marker inserted where the context was cut, so a truncated prompt is visible
#: in the saved raw prompts rather than looking like a short document.
TRUNCATION_MARKER = "\n\n[... context truncated ...]\n\n"


def prompt_hash() -> str:
    """Stable fingerprint of the template, recorded in run provenance."""
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


@dataclass
class BuiltPrompt:
    text: str
    input_tokens: int
    truncated: bool
    context_tokens: int
    dropped_context_tokens: int = 0


def build_prompt_with_budget(
    context: str,
    question: str,
    tokenizer,
    max_input_tokens: int,
    chat_template: bool = True,
    system_message: Optional[str] = None,
) -> BuiltPrompt:
    """``input_tokens`` counts the final string, chat template included."""
    text = _render(build_prompt(context, question), tokenizer, chat_template, system_message)
    total = _count(tokenizer, text)
    context_tokens = _count(tokenizer, context)

    if total <= max_input_tokens:
        return BuiltPrompt(
            text=text,
            input_tokens=total,
            truncated=False,
            context_tokens=context_tokens,
        )

    # Budget for the context = max - (everything else). Measure the overhead by
    # rendering with an empty context rather than estimating it.
    empty = _render(build_prompt("", question), tokenizer, chat_template, system_message)
    overhead = _count(tokenizer, empty)
    marker_tokens = _count(tokenizer, TRUNCATION_MARKER)
    budget = max_input_tokens - overhead - marker_tokens

    if budget <= 0:
        # The question alone exhausts the budget; keep the question intact and
        # send an empty context rather than corrupting the instruction.
        return BuiltPrompt(
            text=empty,
            input_tokens=overhead,
            truncated=True,
            context_tokens=0,
            dropped_context_tokens=context_tokens,
        )

    ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    head = budget // 2
    tail = budget - head
    kept = tokenizer.decode(ids[:head]) + TRUNCATION_MARKER + tokenizer.decode(ids[-tail:])

    text = _render(build_prompt(kept, question), tokenizer, chat_template, system_message)
    return BuiltPrompt(
        text=text,
        input_tokens=_count(tokenizer, text),
        truncated=True,
        context_tokens=_count(tokenizer, kept),
        dropped_context_tokens=max(0, context_tokens - budget),
    )


def _render(
    body: str, tokenizer, chat_template: bool, system_message: Optional[str]
) -> str:
    """Wrap in the model's chat template when it has one."""
    if not chat_template or getattr(tokenizer, "chat_template", None) is None:
        return body

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": body})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def join_context(texts: Sequence[str], separator: str = "\n\n") -> str:
    return separator.join(texts)


def describe() -> dict:
    return {
        "template": PROMPT_TEMPLATE,
        "template_sha256_16": prompt_hash(),
        "truncation_marker": TRUNCATION_MARKER,
        "truncation_policy": "middle-out on the context block only",
    }
