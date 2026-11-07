"""HHEM-2.1-Open faithfulness: P(answer is supported by the retrieved context).

If the model cannot be loaded, every value is ``None`` and the reason is recorded
in ``metric_availability.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..utils.gpu import release, resolve_device
from ..utils.logging import get_logger

DEFAULT_MODEL = "vectara/hallucination_evaluation_model"


@dataclass
class FaithfulnessAvailability:
    available: bool
    model: str
    reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "model": self.model, "reason": self.reason}


class FaithfulnessScorer:
    """HHEM wrapper with explicit unavailability rather than silent fallback."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        revision: str = "main",
        device: str = "auto",
        batch_size: int = 8,
        max_premise_tokens: int = 2048,
        fail_open: bool = True,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.max_premise_tokens = int(max_premise_tokens)
        self.fail_open = bool(fail_open)

        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None
        self._load_attempted = False

    # -- lifecycle ------------------------------------------------------------

    def _load(self) -> bool:
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True

        log = get_logger()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            log.info("loading faithfulness model %s", self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.revision, trust_remote_code=True
            )
            self._model.eval()
            if self.device == "cuda":
                self._model = self._model.to("cuda")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, revision=self.revision, trust_remote_code=True
                )
            except Exception:
                # HHEM-2.1-Open bundles its tokenizer inside the custom class;
                # premise truncation falls back to a character budget below.
                self._tokenizer = None
            log.info("faithfulness model ready on %s", self.device)
            return True
        except Exception as exc:  # noqa: BLE001 - reason is recorded, not swallowed
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._model = None
            message = (
                f"could not load {self.model_name}: {self._load_error}. "
                "Faithfulness will be reported as null. HHEM-2.1-Open needs "
                "trust_remote_code and is known to break on transformers>=5; "
                "pin 'transformers>=4.41,<5' to enable it."
            )
            if self.fail_open:
                log.warning(message)
                return False
            raise RuntimeError(message) from exc

    def availability(self) -> FaithfulnessAvailability:
        loaded = self._load()
        return FaithfulnessAvailability(
            available=loaded, model=self.model_name, reason=self._load_error
        )

    def unload(self) -> None:
        if self._model is not None:
            release(self._model)
            self._model = None
            self._tokenizer = None
            get_logger().info("faithfulness model unloaded")

    # -- scoring --------------------------------------------------------------

    def _truncate_premise(self, premise: str) -> str:
        if self._tokenizer is not None:
            ids = self._tokenizer(premise, add_special_tokens=False)["input_ids"]
            if len(ids) <= self.max_premise_tokens:
                return premise
            return self._tokenizer.decode(ids[: self.max_premise_tokens])
        # ~4 characters per token is the standard English approximation; only
        # used when the checkpoint exposes no tokenizer.
        budget = self.max_premise_tokens * 4
        return premise if len(premise) <= budget else premise[:budget]

    def score_pairs(
        self, pairs: Sequence[Tuple[str, str]], progress: bool = True
    ) -> List[Optional[float]]:
        """Score ``(premise, hypothesis)`` pairs; ``None`` where unavailable."""
        if not pairs:
            return []
        if not self._load():
            return [None] * len(pairs)

        prepared = [(self._truncate_premise(p), h or "") for p, h in pairs]

        iterator = range(0, len(prepared), self.batch_size)
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(list(iterator), desc="faithfulness", ncols=80)
            except ImportError:
                pass

        import torch

        scores: List[Optional[float]] = []
        for offset in iterator:
            batch = prepared[offset : offset + self.batch_size]
            try:
                with torch.inference_mode():
                    values = self._model.predict(batch)
                scores.extend(float(v) for v in _to_list(values))
            except Exception as exc:  # noqa: BLE001
                get_logger().warning(
                    "faithfulness scoring failed for batch at offset %d: %s", offset, exc
                )
                scores.extend([None] * len(batch))

        return scores

    def score(
        self, contexts: Sequence[str], answers: Sequence[str], progress: bool = True
    ) -> List[Optional[float]]:
        if len(contexts) != len(answers):
            raise ValueError("contexts and answers must have equal length")
        return self.score_pairs(list(zip(contexts, answers)), progress=progress)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_premise_tokens": self.max_premise_tokens,
            "premise": "concatenated retrieved chunks",
            "hypothesis": "generated answer",
            "range": "[0, 1], higher = better supported",
        }


def _to_list(values: Any) -> List[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(v) for v in values]


def build_faithfulness_scorer(cfg) -> Optional[FaithfulnessScorer]:
    if not cfg["faithfulness.enabled"]:
        return None
    return FaithfulnessScorer(
        model_name=cfg["faithfulness.model"],
        revision=cfg["faithfulness.revision"],
        device=cfg["runtime.device"],
        batch_size=cfg["faithfulness.batch_size"],
        max_premise_tokens=cfg["faithfulness.max_premise_tokens"],
        fail_open=cfg["faithfulness.fail_open"],
    )
