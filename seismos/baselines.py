"""Table 1 baselines. They produce text without character spans, so they are
evaluated at document level only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

BASELINES = ("fixed_chars", "recursive_chars", "langchain", "llamaindex")


@dataclass(frozen=True)
class BaselineSpec:
    method: str
    chunk_size: int = 512  # characters
    overlap: float = 0.10
    threshold_type: str = "percentile"
    threshold_amount: Optional[float] = None
    buffer_size: int = 1
    percentile: int = 95

    def __post_init__(self):
        if self.method not in BASELINES:
            raise ValueError(f"unknown baseline {self.method!r}; expected one of {BASELINES}")

    @property
    def label(self) -> str:
        if self.method in ("fixed_chars", "recursive_chars"):
            return f"{self.method}{self.chunk_size}"
        if self.method == "langchain":
            return f"langchain_{self.threshold_type}"
        return f"llamaindex_p{self.percentile}"


class _LangChainEmbeddings:
    def __init__(self, encoder):
        self.encoder = encoder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encoder.encode_passages(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encoder.encode_queries([text])[0].tolist()


def _llamaindex_embedding(encoder):
    from llama_index.core.embeddings import BaseEmbedding

    class _Embedding(BaseEmbedding):
        def _get_query_embedding(self, query: str) -> List[float]:
            return encoder.encode_queries([query])[0].tolist()

        def _get_text_embedding(self, text: str) -> List[float]:
            return encoder.encode_passages([text])[0].tolist()

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            return encoder.encode_passages(texts).tolist()

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_query_embedding(query)

    return _Embedding(model_name=encoder.name)


def build_splitter(spec: BaselineSpec, encoder=None):
    if spec.method in ("fixed_chars", "recursive_chars"):
        from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

        overlap = int(spec.chunk_size * spec.overlap)
        if spec.method == "fixed_chars":
            splitter = CharacterTextSplitter(separator="", chunk_size=spec.chunk_size, chunk_overlap=overlap)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=spec.chunk_size, chunk_overlap=overlap)
        return splitter.split_text

    if encoder is None:
        raise ValueError(f"{spec.method} needs an encoder")

    if spec.method == "langchain":
        from langchain_experimental.text_splitter import SemanticChunker

        kwargs = {"breakpoint_threshold_type": spec.threshold_type, "buffer_size": spec.buffer_size}
        if spec.threshold_amount is not None:
            kwargs["breakpoint_threshold_amount"] = spec.threshold_amount
        return SemanticChunker(_LangChainEmbeddings(encoder), **kwargs).split_text

    from llama_index.core import Document
    from llama_index.core.node_parser import SemanticSplitterNodeParser

    parser = SemanticSplitterNodeParser(buffer_size=spec.buffer_size,
                                        breakpoint_percentile_threshold=spec.percentile,
                                        embed_model=_llamaindex_embedding(encoder))

    def split(text: str) -> List[str]:
        nodes = parser.get_nodes_from_documents([Document(text=text)], show_progress=False)
        return [n.get_content() for n in nodes if n.get_content().strip()]

    return split


def chunk_corpus_baseline(docs: Dict[str, str], spec: BaselineSpec, encoder=None):
    split = build_splitter(spec, encoder)
    texts: List[str] = []
    owner: List[str] = []
    for doc_id, text in docs.items():
        if not text:
            continue
        pieces = [p for p in split(text) if p.strip()] or [text]
        texts.extend(pieces)
        owner.extend([doc_id] * len(pieces))
    return texts, owner
