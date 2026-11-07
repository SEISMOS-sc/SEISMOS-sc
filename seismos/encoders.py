"""Encoders used in the paper and their query/passage prefixes.

``Alibaba-NLP/gte-*-en-v1.5`` is not supported: under current ``transformers``
its remote code leaves position/RoPE buffers uninitialised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EncoderSpec:
    model: str
    query_prefix: str = ""
    passage_prefix: str = ""
    trust_remote_code: bool = False
    fp16: bool = True
    batch_size: int = 64


ENCODERS: Dict[str, EncoderSpec] = {
    "bge": EncoderSpec("BAAI/bge-base-en-v1.5", query_prefix=BGE_QUERY_INSTRUCTION),
    "e5": EncoderSpec("intfloat/e5-base-v2", query_prefix="query: ", passage_prefix="passage: "),
    "gte": EncoderSpec("thenlper/gte-base"),
    "gte-multi": EncoderSpec("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True),
    "me5": EncoderSpec("intfloat/multilingual-e5-base", query_prefix="query: ", passage_prefix="passage: "),
}

# Per language: two paraphrases, then an unrelated sentence.
SANITY_PROBES = {
    "en": ["The cat sat quietly on the mat beside the front door.",
           "A feline rested silently on the rug near the entrance.",
           "Quarterly revenue rose twelve percent on strong subscription growth."],
    "ja": ["猫は玄関の脇のマットの上で静かに座っていた。",
           "ネコが入口付近の敷物の上でしずかに休んでいた。",
           "四半期の売上高は定額サービスの好調により一二パーセント増加した。"],
    "de": ["Die Katze saß ruhig auf der Matte neben der Haustür.",
           "Eine Katze ruhte still auf dem Teppich nahe dem Eingang.",
           "Der Quartalsumsatz stieg um zwölf Prozent dank starkem Abonnementgeschäft."],
    "ar": ["جلست القطة بهدوء على السجادة بجانب الباب الأمامي.",
           "استراح قط بصمت على البساط بالقرب من المدخل.",
           "ارتفعت الإيرادات الفصلية اثني عشر بالمئة بفضل نمو الاشتراكات."],
}
SANITY_MARGIN = 0.10


class Encoder:
    def __init__(self, spec: EncoderSpec, device: Optional[str] = None, max_seq_length: int = 512,
                 query_prefix: Optional[str] = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_prefix = spec.query_prefix if query_prefix is None else query_prefix
        self.passage_prefix = spec.passage_prefix
        self.model = SentenceTransformer(spec.model, device=self.device,
                                         trust_remote_code=spec.trust_remote_code)
        self.model.max_seq_length = max_seq_length
        if spec.trust_remote_code:
            _repair_position_ids(self.model)
        if self.device == "cuda" and spec.fp16:
            self.model = self.model.half()
        self.tokenizer = self.model.tokenizer
        self.batch_size = spec.batch_size

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "Encoder":
        if name not in ENCODERS:
            raise KeyError(f"unknown encoder {name!r}; available: {sorted(ENCODERS)}")
        return cls(ENCODERS[name], **kwargs)

    @property
    def name(self) -> str:
        return self.spec.model

    def _encode(self, texts: Sequence[str], normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        if len(texts) == 0:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        E = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=normalize,
                              convert_to_numpy=True, show_progress_bar=show_progress)
        return np.ascontiguousarray(E, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str], normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts], normalize, show_progress)

    def encode_passages(self, texts: Sequence[str], normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts], normalize, show_progress)

    def token_counts(self, texts: Sequence[str]) -> List[int]:
        if not texts:
            return []
        return [len(ids) for ids in self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]]


def _repair_position_ids(st) -> None:
    """gte-multilingual's remote code can load with garbage ``position_ids``; rebuild them."""
    import torch

    try:
        emb = st[0].auto_model.embeddings
    except (AttributeError, IndexError, TypeError):
        return
    pid = getattr(emb, "position_ids", None)
    if pid is None or pid.numel() < 8:
        return
    head = pid.reshape(-1)[:8].detach().cpu().long()
    if not torch.equal(head, torch.arange(8, dtype=head.dtype)):
        n = int(getattr(st[0].auto_model.config, "max_position_embeddings", pid.numel()))
        emb.position_ids = torch.arange(n, device=pid.device).reshape(1, n)
        logger.warning("%s: position_ids were uninitialised and have been rebuilt", st[0].auto_model.name_or_path)


def sanity_check(encoder: Encoder, langs: Sequence[str] = ("en",), margin: float = SANITY_MARGIN) -> None:
    """Raise unless paraphrases score at least ``margin`` above an unrelated sentence."""
    for lang in langs:
        probe = SANITY_PROBES[lang]
        E = encoder.encode_passages(probe)
        if not np.isfinite(E).all():
            raise RuntimeError(f"{encoder.name} [{lang}]: embeddings contain NaN/Inf")
        para, unrel = float(E[0] @ E[1]), float(E[0] @ E[2])
        if para <= unrel + margin:
            raise RuntimeError(
                f"{encoder.name} [{lang}] is not encoding: paraphrase cos={para:.3f}, "
                f"unrelated cos={unrel:.3f}, required margin {margin}. Do not record its results.")
        logger.info("%s [%s] sanity OK: paraphrase %.3f > unrelated %.3f", encoder.name, lang, para, unrel)
