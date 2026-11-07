"""The shared encoder: sentence embeddings for chunking, chunk embeddings and
query embeddings all come from one instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..utils.caching import NpyCache, stable_hash
from ..utils.gpu import release, resolve_device
from ..utils.logging import get_logger


@dataclass(frozen=True)
class EncoderSpec:
    tag: str
    model: str
    query_prefix: str
    passage_prefix: str
    max_seq_length: int = 512
    dim: int = 768


ENCODER_REGISTRY: Dict[str, EncoderSpec] = {
    "e5": EncoderSpec(
        tag="e5base",
        model="intfloat/e5-base-v2",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "bge": EncoderSpec(
        tag="bgebase",
        model="BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
    ),
    "gte": EncoderSpec(
        tag="gtebase",
        model="thenlper/gte-base",
        query_prefix="",
        passage_prefix="",
    ),
}


class Encoder:
    """SentenceTransformers wrapper with prefixes, fp16 and a disk cache."""

    def __init__(
        self,
        model_key: str = "e5",
        batch_size: int = 64,
        fp16: bool = True,
        normalize: bool = True,
        device: str = "auto",
        cache_dir: Optional[str] = "cache/embeddings",
        cache_sentences: bool = True,
        cache_chunks: bool = True,
    ) -> None:
        if model_key not in ENCODER_REGISTRY:
            raise ValueError(
                f"unknown embedding model {model_key!r}; "
                f"available: {sorted(ENCODER_REGISTRY)}"
            )
        self.spec = ENCODER_REGISTRY[model_key]
        self.model_key = model_key
        self.batch_size = int(batch_size)
        self.fp16 = bool(fp16)
        self.normalize = bool(normalize)
        self.device = resolve_device(device)
        self.cache_sentences = cache_sentences
        self.cache_chunks = cache_chunks

        self._model = None
        self._tokenizer = None
        self._cache = (
            NpyCache(cache_dir, namespace=self.spec.tag) if cache_dir else None
        )

        if not self.normalize:
            raise ValueError(
                "Encoder.normalize must be True: IndexFlatIP equals cosine "
                "similarity only for unit-norm vectors"
            )

    # -- model lifecycle ------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self):
        from sentence_transformers import SentenceTransformer

        log = get_logger()
        log.info(
            "loading encoder %s (%s) on %s fp16=%s",
            self.model_key,
            self.spec.model,
            self.device,
            self.fp16,
        )
        model = SentenceTransformer(self.spec.model, device=self.device)
        model.max_seq_length = self.spec.max_seq_length
        if self.fp16 and self.device == "cuda":
            model = model.half()
        model.eval()
        log.info(
            "encoder ready: dim=%d max_seq_length=%d pooling from modules.json",
            model.get_sentence_embedding_dimension(),
            model.max_seq_length,
        )
        return model

    @property
    def tokenizer(self):
        """The encoder's own tokenizer - the token budget every chunker uses."""
        if self._tokenizer is None:
            self._tokenizer = self.model.tokenizer
        return self._tokenizer

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def unload(self) -> None:
        if self._model is not None:
            release(self._model)
            self._model = None
            self._tokenizer = None
            get_logger().info("encoder unloaded")

    # -- tokenisation ---------------------------------------------------------

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        """Token counts without special tokens (what ``chunking.max_tokens`` budgets)."""
        if not texts:
            return []
        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def count_tokens_single(self, text: str) -> int:
        return self.count_tokens([text])[0]

    # -- encoding -------------------------------------------------------------

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        embeddings = self.model.encode(
            payload,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.spec.passage_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.spec.query_prefix)

    def encode_sentences(self, sentences: Sequence[str]) -> np.ndarray:
        """Sentences get the passage prefix, as they do inside indexed chunks."""
        return self.encode_passages(sentences)

    # -- cached variants ------------------------------------------------------

    def _cache_key(self, kind: str, payload: Any) -> str:
        return stable_hash(
            kind,
            self.spec.model,
            self.spec.passage_prefix,
            self.spec.query_prefix,
            "fp16" if self.fp16 else "fp32",
            payload,
            length=24,
        )

    def encode_sentences_cached(self, doc_id: str, sentences: Sequence[str]) -> np.ndarray:
        """Cached by sentence content, not by document id."""
        if not sentences:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._cache is None or not self.cache_sentences:
            return self.encode_sentences(sentences)

        key = self._cache_key("sentences", list(sentences))
        cached = self._cache.get(key)
        if cached is not None and cached.shape[0] == len(sentences):
            return cached.astype(np.float32, copy=False)

        embeddings = self.encode_sentences(sentences)
        self._cache.put(key, embeddings)
        return embeddings

    def encode_chunks_cached(self, chunk_texts: Sequence[str]) -> np.ndarray:
        if not chunk_texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._cache is None or not self.cache_chunks:
            return self.encode_passages(chunk_texts)

        key = self._cache_key("chunks", list(chunk_texts))
        cached = self._cache.get(key)
        if cached is not None and cached.shape[0] == len(chunk_texts):
            return cached.astype(np.float32, copy=False)

        embeddings = self.encode_passages(chunk_texts)
        self._cache.put(key, embeddings)
        return embeddings

    def encode_queries_cached(self, queries: Sequence[str]) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._cache is None:
            return self.encode_queries(queries)

        key = self._cache_key("queries", list(queries))
        cached = self._cache.get(key)
        if cached is not None and cached.shape[0] == len(queries):
            return cached.astype(np.float32, copy=False)

        embeddings = self.encode_queries(queries)
        self._cache.put(key, embeddings)
        return embeddings

    def cache_stats(self) -> Dict[str, int]:
        return self._cache.stats() if self._cache else {"hits": 0, "misses": 0}

    def describe(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model": self.spec.model,
            "query_prefix": self.spec.query_prefix,
            "passage_prefix": self.spec.passage_prefix,
            "max_seq_length": self.spec.max_seq_length,
            "batch_size": self.batch_size,
            "fp16": self.fp16,
            "normalize": self.normalize,
            "device": self.device,
        }


# LangChain / LlamaIndex adapters onto the shared encoder.


class LangChainEmbeddings:
    """Minimal ``langchain_core.embeddings.Embeddings`` implementation."""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encoder.encode_passages(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encoder.encode_queries([text])[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def _llamaindex_base():
    from llama_index.core.embeddings import BaseEmbedding

    return BaseEmbedding


class _LlamaIndexEmbeddingFactory:
    """Built lazily so llama-index stays optional."""

    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is not None:
            return cls._cls

        BaseEmbedding = _llamaindex_base()
        from typing import Any as _Any

        from pydantic import PrivateAttr

        class _Adapter(BaseEmbedding):
            _encoder: _Any = PrivateAttr()

            def __init__(self, encoder: Encoder, **kwargs: _Any) -> None:
                super().__init__(
                    model_name=encoder.spec.model,
                    embed_batch_size=encoder.batch_size,
                    **kwargs,
                )
                self._encoder = encoder

            def _get_query_embedding(self, query: str) -> List[float]:
                return self._encoder.encode_queries([query])[0].tolist()

            def _get_text_embedding(self, text: str) -> List[float]:
                return self._encoder.encode_passages([text])[0].tolist()

            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return self._encoder.encode_passages(texts).tolist()

            async def _aget_query_embedding(self, query: str) -> List[float]:
                return self._get_query_embedding(query)

            async def _aget_text_embedding(self, text: str) -> List[float]:
                return self._get_text_embedding(text)

        cls._cls = _Adapter
        return cls._cls


def LlamaIndexEmbedding(encoder: Encoder):  # noqa: N802 - factory named like a class
    """Return a ``BaseEmbedding`` bound to ``encoder``."""
    return _LlamaIndexEmbeddingFactory.get()(encoder=encoder)


def build_encoder(cfg) -> Encoder:
    """Construct the shared encoder from config."""
    return Encoder(
        model_key=cfg["embedding.model"],
        batch_size=cfg["embedding.batch_size"],
        fp16=cfg["embedding.fp16"],
        normalize=cfg["embedding.normalize"],
        device=cfg["runtime.device"],
        cache_dir=cfg["embedding.cache_dir"],
        cache_sentences=cfg["embedding.cache_sentence_embeddings"],
        cache_chunks=cfg["embedding.cache_chunk_embeddings"],
    )
