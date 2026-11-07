"""End-to-end RAG evaluation of chunking strategies on LongBench NarrativeQA.

Stages: load -> sentence split -> chunk/embed/retrieve -> generate ->
faithfulness -> aggregate. One large model is resident at a time; each stage
checkpoints per (qid, chunker), so reruns resume.

    python run_pipeline.py --chunkers all --generator qwen --num_samples 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow both `python seismos_rag/run_pipeline.py` and `python -m seismos_rag.run_pipeline`.
if __package__ in (None, ""):  # pragma: no cover - entry-point shim
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "seismos_rag"

import numpy as np

from .chunkers import (
    ChunkingContext,
    SentenceSplitter,
    build_chunkers,
    chunk_statistics,
    resolve_chunker_names,
)
from .config.loader import CLI_OVERRIDE_MAP, build_provenance, load_config
from .data.longbench_loader import (
    load_longbench_narrativeqa_with_stats,
    save_manifest,
    validate_sample,
)
from .embeddings.encoder import build_encoder
from .evaluation.evidence_metrics import availability_report, score_evidence
from .evaluation.faithfulness import build_faithfulness_scorer
from .evaluation.qa_metrics import score_prediction
from .evaluation.statistics import run_all_comparisons, summarize
from .generation.llm_runner import build_llm_runner
from .generation.prompt import describe as describe_prompt
from .retrieval.retriever import CONTEXT_SEPARATOR, Retriever
from .utils.caching import JsonlCheckpoint
from .utils.exports import write_csv, write_json
from .utils.gpu import (
    log_gpu_memory,
    peak_memory_gb,
    preflight,
    reset_peak_memory,
    resolve_device,
)
from .utils.logging import LatencyLog, LatencyRecord, banner, get_logger, setup_logging, timer
from .utils.seeding import set_global_seed



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="SEISMOS end-to-end RAG evaluation on LongBench NarrativeQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--dataset", default=None, help="LongBench subset (default: narrativeqa)")
    parser.add_argument(
        "--chunkers",
        default="all",
        help="comma-separated chunkers, or 'all' (fixed,recursive,langchain_semantic,"
        "llamaindex_semantic,seismos)",
    )
    parser.add_argument("--generator", default=None, help="qwen | llama | HF model id")
    parser.add_argument("--embedding", default=None, help="e5 | bge | gte")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None, help="chunk token cap")
    parser.add_argument("--seismos_k", type=float, default=None)
    parser.add_argument(
        "--run_k_sweep",
        action="store_true",
        default=None,
        help="evaluate every k in chunking.seismos.k_sweep as a separate method",
    )
    parser.add_argument("--quantization", default=None, choices=["none", "4bit", "8bit"])
    parser.add_argument("--gen_batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=None,
        help="prompt ceiling; context is middle-truncated above it. Under eager "
        "attention the memory cost is quadratic in this number.",
    )
    parser.add_argument(
        "--attn",
        default=None,
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        help="attention backend (default: auto -> sdpa). 'eager' is the reference "
        "kernel and needs several GB more VRAM per batch.",
    )
    parser.add_argument(
        "--no-preflight",
        dest="preflight",
        action="store_false",
        default=None,
        help="skip the VRAM estimate and let the load attempt fail on its own",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="root for dataset/sentence/embedding caches. Keep this on fast local "
        "disk (e.g. /content/cache) even when --output_dir points at Drive.",
    )
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--faithfulness",
        dest="faithfulness",
        action="store_true",
        default=None,
        help="force faithfulness scoring on",
    )
    parser.add_argument(
        "--no-faithfulness", dest="faithfulness", action="store_false", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--llm_chunker",
        action="store_true",
        default=None,
        help="enable the optional LLM chunker baseline",
    )
    parser.add_argument(
        "--allow-missing-chunkers",
        action="store_true",
        help="skip chunkers whose library is not installed instead of failing",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore existing checkpoints and recompute everything",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="comma-separated subset of: chunk,retrieve,generate,faithfulness,aggregate",
    )
    parser.add_argument("--limit-docs", type=int, default=None, help="debug: cap documents")
    return parser


ALL_STAGES = ("chunk", "retrieve", "generate", "faithfulness", "aggregate")


def resolve_stages(value: str) -> List[str]:
    if value == "all":
        return list(ALL_STAGES)
    stages = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; available: {list(ALL_STAGES)}")
    return stages


def preflight_generator(cfg) -> None:
    """Estimate the generator's VRAM need before any download or loading."""
    if not cfg.get("generation.preflight", True):
        get_logger().info("generator preflight disabled in config")
        return
    if resolve_device(cfg["runtime.device"]) != "cuda":
        return

    from .generation.llm_runner import GENERATOR_REGISTRY, resolve_attn_implementation

    model_key = cfg["generation.model"]
    preflight(
        GENERATOR_REGISTRY.get(model_key, model_key),
        cfg["generation.dtype"],
        cfg["generation.quantization"],
        batch_size=cfg["generation.batch_size"],
        max_input_tokens=cfg["generation.max_input_tokens"],
        max_new_tokens=cfg["generation.max_new_tokens"],
        attn_implementation=resolve_attn_implementation(
            cfg.get("generation.attn_implementation", "auto")
        ),
    )



class Pipeline:
    def __init__(self, cfg, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.log = get_logger()
        self.latency = LatencyLog()

        self.output_dir = Path(cfg["output.dir"])
        # Checkpoints live under output_dir so they move with it.
        configured_checkpoints = cfg.get("output.checkpoint_dir")
        self.checkpoint_dir = (
            Path(configured_checkpoints)
            if configured_checkpoints
            else self.output_dir / "checkpoints"
        )
        cfg.set("output.checkpoint_dir", str(self.checkpoint_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.resume = not args.no_resume
        self.top_k = int(cfg["retrieval.top_k"])
        self.device = resolve_device(cfg["runtime.device"])

        self.chunker_names = resolve_chunker_names(args.chunkers)
        if cfg["chunking.llm_chunker.enabled"] and "llm_chunker" not in self.chunker_names:
            self.chunker_names.append("llm_chunker")

        self.samples: List[Dict[str, Any]] = []
        self.sample_by_id: Dict[str, Dict[str, Any]] = {}
        self.method_names: List[str] = []
        self.dataset_stats = None

    # -- checkpoints ----------------------------------------------------------

    def _checkpoint(self, name: str, key_fields: Tuple[str, ...]) -> JsonlCheckpoint:
        path = self.checkpoint_dir / name
        if not self.resume and path.exists():
            self.log.warning("--no-resume: discarding checkpoint %s", name)
            path.unlink()
        return JsonlCheckpoint(path, key_fields=key_fields, enabled=True)

    # -- stage 1: data --------------------------------------------------------

    def load_data(self) -> None:
        banner("STAGE 1 - dataset")
        with timer("load dataset", stage="data", sink=self.latency) as t:
            samples, stats = load_longbench_narrativeqa_with_stats(
                num_samples=self.cfg["dataset.num_samples"],
                seed=self.cfg["seed"],
                cache_dir=self.cfg["dataset.cache_dir"],
                name=self.cfg["dataset.name"],
                split=self.cfg["dataset.split"],
                hf_repos=self.cfg["dataset.hf_repos"],
                verify_unmodified=self.cfg["dataset.verify_unmodified"],
            )
            t["n_items"] = len(samples)

        if self.args.limit_docs:
            samples = samples[: self.args.limit_docs]
            self.log.warning("--limit-docs: truncated to %d documents", len(samples))

        for sample in samples:
            validate_sample(sample)

        self.samples = samples
        self.sample_by_id = {s["id"]: s for s in samples}
        self.dataset_stats = stats
        save_manifest(samples, stats, self.output_dir / "dataset_manifest.json")

    # -- stages 2+3: chunk, embed, retrieve -----------------------------------

    def chunk_and_retrieve(self, do_chunk: bool, do_retrieve: bool) -> None:
        banner("STAGE 2+3 - sentence split, chunking, embedding, retrieval")

        splitter = SentenceSplitter(
            spacy_model=self.cfg["sentence_splitter.spacy_model"],
            pipe=self.cfg["sentence_splitter.spacy_pipe"],
            max_length=self.cfg["sentence_splitter.max_length"],
            cache_dir=self.cfg["sentence_splitter.cache_dir"],
        )
        encoder = build_encoder(self.cfg)
        chunkers = build_chunkers(
            self.chunker_names,
            self.cfg,
            encoder=encoder,
            sentence_splitter=splitter,
            llm_runner=None,
            allow_missing=self.args.allow_missing_chunkers,
        )
        self.method_names = [c.name for c in chunkers]

        chunk_ckpt = self._checkpoint("chunks.jsonl", ("qid", "chunker"))
        retrieval_ckpt = self._checkpoint("retrieval.jsonl", ("qid", "chunker"))
        signal_ckpt = self._checkpoint("signal_diagnostics.jsonl", ("qid",))

        retriever = Retriever(encoder=encoder, top_k=self.top_k)
        cached_chunks = self._load_cached_chunks(chunk_ckpt)

        reset_peak_memory()
        progress = self._progress(self.samples, "documents")

        total_chunks = 0
        for sample in progress:
            qid = sample["id"]

            outstanding = [
                c
                for c in chunkers
                if (do_chunk and not chunk_ckpt.is_done(qid, c.name))
                or (do_retrieve and not retrieval_ckpt.is_done(qid, c.name))
            ]
            if not outstanding:
                continue

            ctx = self._build_context(sample, splitter, encoder)

            if not signal_ckpt.is_done(qid):
                signal_ckpt.append(self._signal_record(qid, ctx))

            for chunker in outstanding:
                chunks = cached_chunks.get((qid, chunker.name))
                if chunks is None:
                    with timer(
                        f"chunk {chunker.name}",
                        stage="chunking",
                        sink=self.latency,
                        chunker=chunker.name,
                        qid=qid,
                    ):
                        chunks = chunker.chunk(ctx)
                    stats = chunk_statistics(chunks, ctx.total_document_tokens)
                    chunk_ckpt.append(
                        {
                            "qid": qid,
                            "chunker": chunker.name,
                            "chunks": chunks,
                            "stats": stats,
                            "params": chunker.params(),
                        }
                    )
                    cached_chunks[(qid, chunker.name)] = chunks
                total_chunks += len(chunks)

                if do_retrieve and not retrieval_ckpt.is_done(qid, chunker.name):
                    result = retriever.retrieve(
                        qid=qid,
                        chunker_name=chunker.name,
                        question=sample["question"],
                        chunks=chunks,
                    )
                    self.latency.add(
                        LatencyRecord(
                            stage="retrieval",
                            label=f"retrieve {chunker.name}",
                            seconds=result.retrieval_seconds,
                            n_items=1,
                            extra={"qid": qid, "chunker": chunker.name},
                        )
                    )
                    retrieval_ckpt.append(result.to_record())

        chunk_ckpt.close()
        retrieval_ckpt.close()
        signal_ckpt.close()

        self.log.info(
            "chunking+retrieval done | %d chunks produced this run | encoder cache %s | "
            "sentence cache %s | peak GPU %.2f GB",
            total_chunks,
            encoder.cache_stats(),
            splitter.cache_stats(),
            peak_memory_gb() or 0.0,
        )
        log_gpu_memory("before encoder unload:")
        if self.cfg["runtime.unload_models_between_stages"]:
            encoder.unload()
            log_gpu_memory("after encoder unload:")

    def _build_context(self, sample, splitter, encoder) -> ChunkingContext:
        text = sample["context"]
        with timer(
            "sentence split", stage="chunking", sink=self.latency, qid=sample["id"]
        ) as t:
            split = splitter.split(text)
            t["n_items"] = len(split.sentences)

        token_counts = encoder.count_tokens(split.sentences)

        sentence_embeddings = None
        if split.sentences:
            with timer(
                "sentence embeddings",
                stage="embedding",
                sink=self.latency,
                qid=sample["id"],
            ) as t:
                sentence_embeddings = encoder.encode_sentences_cached(
                    sample["id"], split.sentences
                )
                t["n_items"] = len(split.sentences)

        return ChunkingContext(
            doc_id=sample["id"],
            text=text,
            sentences=split.sentences,
            spans=split.spans,
            sentence_token_counts=token_counts,
            sentence_embeddings=sentence_embeddings,
            total_document_tokens=int(sum(token_counts)),
        )

    def _signal_record(self, qid: str, ctx: ChunkingContext) -> Dict[str, Any]:
        """SEISMOS signal diagnostics; validates the AR(1) argument empirically."""
        from .chunkers.seismos import laplacian_diagnostics

        diagnostics = laplacian_diagnostics(ctx.adjacent_similarity())
        return {
            "qid": qid,
            "n_sentences": ctx.n_sentences,
            "total_document_tokens": ctx.total_document_tokens,
            **diagnostics,
        }

    def _load_cached_chunks(self, checkpoint: JsonlCheckpoint) -> Dict[Tuple[str, str], List[Dict]]:
        cache: Dict[Tuple[str, str], List[Dict]] = {}
        for record in checkpoint.iter_all():
            cache[(record["qid"], record["chunker"])] = record["chunks"]
        if cache:
            self.log.info("loaded %d cached chunk sets from checkpoint", len(cache))
        return cache

    # -- stage 4: generation --------------------------------------------------

    def generate(self) -> None:
        banner("STAGE 4 - generation")

        retrieval_ckpt = JsonlCheckpoint(
            self.checkpoint_dir / "retrieval.jsonl", ("qid", "chunker"), enabled=True
        )
        retrieval_records = retrieval_ckpt.read_all()
        retrieval_ckpt.close()
        if not retrieval_records:
            path = self.checkpoint_dir / "retrieval.jsonl"
            raise RuntimeError(
                f"no retrieval records found at {path}.\n\n"
                "Generation consumes the retrieval checkpoint, so the chunk and "
                "retrieve stages must have run into this same checkpoint directory.\n\n"
                "Most likely cause: --stages generate was run in a fresh session. "
                "Hosted notebooks (Kaggle /kaggle/working, Colab /content) wipe the "
                "working directory between sessions, so checkpoints written by an "
                "earlier session are gone.\n\n"
                "Fix, in order of preference:\n"
                "  1. Drop --stages and run the whole pipeline:\n"
                "       python run_pipeline.py --chunkers all --num_samples N\n"
                "  2. Run the prerequisite stages first, in this session:\n"
                "       python run_pipeline.py --stages chunk,retrieve ...\n"
                "  3. Restore checkpoints from a previous session's saved output "
                f"into {self.checkpoint_dir}/ before re-running.\n\n"
                "Splitting stages across separate commands only works while the "
                "checkpoint directory survives between them."
            )

        generation_ckpt = self._checkpoint("generations.jsonl", ("qid", "chunker"))

        # Records from a larger earlier --num_samples.
        orphaned = [r for r in retrieval_records if r["qid"] not in self.sample_by_id]
        if orphaned:
            self.log.warning(
                "%d retrieval record(s) reference qids outside the current selection "
                "(checkpoint from a different --num_samples/--seed?); skipping them",
                len(orphaned),
            )
            retrieval_records = [
                r for r in retrieval_records if r["qid"] in self.sample_by_id
            ]

        pending = [
            record
            for record in retrieval_records
            if not generation_ckpt.is_done(record["qid"], record["chunker"])
        ]
        self.log.info(
            "generation: %d/%d units pending (%d already complete)",
            len(pending),
            len(retrieval_records),
            len(retrieval_records) - len(pending),
        )
        if not pending:
            generation_ckpt.close()
            return

        runner = build_llm_runner(self.cfg)
        items = [
            {
                "qid": record["qid"],
                "chunker": record["chunker"],
                "question": self.sample_by_id[record["qid"]]["question"],
                "context": CONTEXT_SEPARATOR.join(record["retrieved_text"]),
            }
            for record in pending
        ]

        reset_peak_memory()
        with timer("generate answers", stage="generation", sink=self.latency) as t:
            runner.generate(items, on_result=lambda g: generation_ckpt.append(g.to_record()))
            t["n_items"] = len(items)

        self.log.info("generation peak GPU %.2f GB", peak_memory_gb() or 0.0)
        generation_ckpt.close()
        if self.cfg["runtime.unload_models_between_stages"]:
            runner.unload()

    # -- stage 5: faithfulness ------------------------------------------------

    def faithfulness(self) -> Dict[str, Any]:
        banner("STAGE 5 - faithfulness (HHEM)")

        scorer = build_faithfulness_scorer(self.cfg)
        if scorer is None:
            self.log.info("faithfulness disabled in config")
            return {"available": False, "reason": "disabled in config"}

        retrieval = {
            (r["qid"], r["chunker"]): r
            for r in JsonlCheckpoint(
                self.checkpoint_dir / "retrieval.jsonl", ("qid", "chunker")
            ).read_all()
        }
        generations = JsonlCheckpoint(
            self.checkpoint_dir / "generations.jsonl", ("qid", "chunker")
        ).read_all()

        faithfulness_ckpt = self._checkpoint("faithfulness.jsonl", ("qid", "chunker"))
        pending = [
            g for g in generations if not faithfulness_ckpt.is_done(g["qid"], g["chunker"])
        ]
        self.log.info("faithfulness: %d/%d units pending", len(pending), len(generations))

        availability = scorer.availability()
        if not availability.available:
            # Record explicit nulls so every row exists and its emptiness is a
            # recorded fact rather than a missing join.
            for generation in pending:
                faithfulness_ckpt.append(
                    {
                        "qid": generation["qid"],
                        "chunker": generation["chunker"],
                        "faithfulness": None,
                        "reason": availability.reason,
                    }
                )
            faithfulness_ckpt.close()
            return availability.as_dict()

        if pending:
            pairs = [
                (
                    CONTEXT_SEPARATOR.join(
                        retrieval[(g["qid"], g["chunker"])]["retrieved_text"]
                    ),
                    g["answer"],
                )
                for g in pending
            ]
            with timer("faithfulness scoring", stage="faithfulness", sink=self.latency) as t:
                scores = scorer.score_pairs(pairs, progress=self.cfg["runtime.progress_bars"])
                t["n_items"] = len(pairs)

            for generation, score in zip(pending, scores):
                faithfulness_ckpt.append(
                    {
                        "qid": generation["qid"],
                        "chunker": generation["chunker"],
                        "faithfulness": score,
                        "reason": None,
                    }
                )

        faithfulness_ckpt.close()
        if self.cfg["runtime.unload_models_between_stages"]:
            scorer.unload()
        return availability.as_dict()

    # -- stage 6: aggregation -------------------------------------------------

    def aggregate(self, faithfulness_availability: Dict[str, Any]) -> Dict[str, Any]:
        banner("STAGE 6 - metrics and export")

        chunk_records = JsonlCheckpoint(
            self.checkpoint_dir / "chunks.jsonl", ("qid", "chunker")
        ).read_all()
        retrieval_records = JsonlCheckpoint(
            self.checkpoint_dir / "retrieval.jsonl", ("qid", "chunker")
        ).read_all()
        generation_records = JsonlCheckpoint(
            self.checkpoint_dir / "generations.jsonl", ("qid", "chunker")
        ).read_all()
        faithfulness_records = JsonlCheckpoint(
            self.checkpoint_dir / "faithfulness.jsonl", ("qid", "chunker")
        ).read_all()
        signal_records = JsonlCheckpoint(
            self.checkpoint_dir / "signal_diagnostics.jsonl", ("qid",)
        ).read_all()

        retrieval_by_key = {(r["qid"], r["chunker"]): r for r in retrieval_records}
        faithfulness_by_key = {
            (r["qid"], r["chunker"]): r.get("faithfulness") for r in faithfulness_records
        }
        chunk_stats_by_key = {(r["qid"], r["chunker"]): r["stats"] for r in chunk_records}

        methods = sorted({r["chunker"] for r in chunk_records}) or self.method_names
        generator_name = self.cfg["generation.model"]
        embedding_name = self.cfg["embedding.model"]

        # -- per-question ----------------------------------------------------
        per_question_rows: List[Dict[str, Any]] = []
        evidence_rows: List[Dict[str, Any]] = []

        for generation in generation_records:
            qid, chunker = generation["qid"], generation["chunker"]
            sample = self.sample_by_id.get(qid)
            if sample is None:
                continue
            retrieval = retrieval_by_key.get((qid, chunker))
            if retrieval is None:
                self.log.warning("no retrieval record for %s/%s; skipping", qid, chunker)
                continue

            answer = generation["answer"]
            references = sample["answers"]
            quality = score_prediction(answer, references)
            chunk_stats = chunk_stats_by_key.get((qid, chunker), {})

            row = {
                "qid": qid,
                "chunker": chunker,
                "question": sample["question"],
                "reference_answer": " ||| ".join(references),
                "generated_answer": answer,
                "qa_f1": round(quality["qa_f1"], 6),
                "exact_match": quality["exact_match"],
                "faithfulness": faithfulness_by_key.get((qid, chunker)),
                "context_tokens": generation["context_tokens"],
                # -- diagnostics beyond the required columns --
                "generator": generator_name,
                "embedding_model": embedding_name,
                "prompt_input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"],
                "prompt_truncated": generation["prompt_truncated"],
                "dropped_context_tokens": generation["dropped_context_tokens"],
                "num_retrieved": retrieval["num_retrieved"],
                "retrieved_chunk_tokens": retrieval["retrieved_chunk_tokens"],
                "avg_retrieved_chunk_tokens": round(
                    retrieval["avg_retrieved_chunk_tokens"], 4
                ),
                "num_chunks_in_document": retrieval["num_chunks_in_document"],
                "doc_num_chunks": chunk_stats.get("num_chunks"),
                "doc_avg_chunk_tokens": _round(chunk_stats.get("avg_chunk_tokens")),
                "n_references": quality["n_references"],
                "prediction_empty": quality["prediction_empty"],
                "retrieval_seconds": round(retrieval["retrieval_seconds"], 6),
                "generation_seconds": round(generation["generation_seconds"], 6),
            }
            per_question_rows.append(row)

            if self.cfg["evidence.enabled"] and self.cfg["evidence.compute_label_free_proxies"]:
                evidence = score_evidence(
                    references, retrieval["retrieved_text"], CONTEXT_SEPARATOR
                )
                evidence_rows.append({"qid": qid, "chunker": chunker, **evidence.as_row()})

        # -- retrieval -------------------------------------------------------
        truncate = self.cfg["output.retrieval_csv_text_chars"]
        retrieval_rows: List[Dict[str, Any]] = []
        for record in retrieval_records:
            for rank, (chunk_id, score, text) in enumerate(
                zip(record["retrieved_chunk_ids"], record["scores"], record["retrieved_text"])
            ):
                retrieval_rows.append(
                    {
                        "qid": record["qid"],
                        "chunker": record["chunker"],
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "score": round(float(score), 6),
                        "chunk_text": text[:truncate] if truncate else text,
                        "chunk_text_truncated_for_csv": bool(truncate and len(text) > truncate),
                    }
                )

        # -- chunk statistics -------------------------------------------------
        chunk_rows = [
            {
                "qid": record["qid"],
                "chunker": record["chunker"],
                "num_chunks": record["stats"]["num_chunks"],
                "avg_chunk_tokens": _round(record["stats"]["avg_chunk_tokens"]),
                "max_chunk_tokens": record["stats"]["max_chunk_tokens"],
                "total_document_tokens": record["stats"]["total_document_tokens"],
                "min_chunk_tokens": record["stats"]["min_chunk_tokens"],
                "median_chunk_tokens": _round(record["stats"]["median_chunk_tokens"]),
                "p95_chunk_tokens": _round(record["stats"]["p95_chunk_tokens"]),
                "std_chunk_tokens": _round(record["stats"]["std_chunk_tokens"]),
                "total_chunk_tokens": record["stats"]["total_chunk_tokens"],
                "unresolved_spans": record["stats"]["unresolved_spans"],
            }
            for record in chunk_records
        ]

        # -- summary ----------------------------------------------------------
        summary_rows = self._build_summary(
            per_question_rows, chunk_rows, methods, generator_name, embedding_name
        )

        # -- significance ------------------------------------------------------
        significance_rows: List[Dict[str, Any]] = []
        if self.cfg["statistics.enabled"]:
            significance_rows = self._run_statistics(per_question_rows, evidence_rows)

        # -- write -------------------------------------------------------------
        write_csv(self.output_dir / "summary_metrics.csv", summary_rows)
        write_csv(self.output_dir / "per_question_results.csv", per_question_rows)
        write_csv(self.output_dir / "retrieval_results.csv", retrieval_rows)
        write_csv(self.output_dir / "chunk_statistics.csv", chunk_rows)
        if evidence_rows:
            write_csv(self.output_dir / "evidence_metrics.csv", evidence_rows)
        if significance_rows:
            write_csv(self.output_dir / "significance_tests.csv", significance_rows)
        if signal_records:
            write_csv(self.output_dir / "seismos_signal_diagnostics.csv", signal_records)
        write_csv(self.output_dir / "latency_log.csv", self.latency.rows())

        write_json(
            self.output_dir / "metric_availability.json",
            availability_report(
                faithfulness_available=bool(faithfulness_availability.get("available")),
                faithfulness_reason=faithfulness_availability.get("reason"),
                evidence_enabled=bool(self.cfg["evidence.enabled"]),
            ),
        )

        if self.cfg["output.save_raw_generations"]:
            write_csv(
                self.output_dir / "raw_generations.csv",
                [
                    {
                        "qid": g["qid"],
                        "chunker": g["chunker"],
                        "prompt": g["prompt"],
                        "answer": g["answer"],
                        "input_tokens": g["input_tokens"],
                        "output_tokens": g["output_tokens"],
                        "prompt_truncated": g["prompt_truncated"],
                        "degenerate": g.get("diagnostics", {}).get("degenerate"),
                        "effective_batch_size": g.get("diagnostics", {}).get(
                            "effective_batch_size"
                        ),
                    }
                    for g in generation_records
                ],
            )

        return {
            "n_per_question_rows": len(per_question_rows),
            "n_retrieval_rows": len(retrieval_rows),
            "n_chunk_rows": len(chunk_rows),
            "methods": methods,
            "summary": summary_rows,
        }

    def _build_summary(
        self,
        per_question_rows: Sequence[Dict[str, Any]],
        chunk_rows: Sequence[Dict[str, Any]],
        methods: Sequence[str],
        generator_name: str,
        embedding_name: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for method in methods:
            questions = [r for r in per_question_rows if r["chunker"] == method]
            chunks = [r for r in chunk_rows if r["chunker"] == method]
            if not questions:
                self.log.warning("no answered questions for method %s", method)
                continue

            faithfulness_values = [
                r["faithfulness"] for r in questions if r["faithfulness"] is not None
            ]
            f1 = summarize([r["qa_f1"] for r in questions])
            em = summarize([r["exact_match"] for r in questions])
            context = summarize([r["context_tokens"] for r in questions])
            chunk_tokens = summarize([r["avg_chunk_tokens"] for r in chunks])
            faith = summarize(faithfulness_values) if faithfulness_values else None

            rows.append(
                {
                    "chunker": method,
                    "generator": generator_name,
                    "embedding_model": embedding_name,
                    "num_questions": len(questions),
                    "qa_f1": round(f1["mean"], 6),
                    "exact_match": round(em["mean"], 6),
                    "faithfulness": round(faith["mean"], 6) if faith else None,
                    "avg_context_tokens": round(context["mean"], 2),
                    "avg_chunk_tokens": round(chunk_tokens["mean"], 2),
                    # -- diagnostics --
                    "qa_f1_std": round(f1["std"], 6),
                    "qa_f1_median": round(f1["median"], 6),
                    "faithfulness_n": len(faithfulness_values),
                    "faithfulness_std": round(faith["std"], 6) if faith else None,
                    "median_context_tokens": round(context["median"], 2),
                    "p95_context_tokens": round(
                        float(np.percentile([r["context_tokens"] for r in questions], 95)), 2
                    ),
                    "avg_chunks_per_document": round(
                        float(np.mean([r["num_chunks"] for r in chunks])), 3
                    )
                    if chunks
                    else None,
                    "avg_num_retrieved": round(
                        float(np.mean([r["num_retrieved"] for r in questions])), 3
                    ),
                    "avg_retrieved_chunk_tokens": round(
                        float(np.mean([r["retrieved_chunk_tokens"] for r in questions])), 2
                    ),
                    "n_prompts_truncated": sum(1 for r in questions if r["prompt_truncated"]),
                    "n_empty_predictions": sum(1 for r in questions if r["prediction_empty"]),
                    "avg_retrieval_seconds": round(
                        float(np.mean([r["retrieval_seconds"] for r in questions])), 6
                    ),
                    "avg_generation_seconds": round(
                        float(np.mean([r["generation_seconds"] for r in questions])), 6
                    ),
                }
            )
        return rows

    def _run_statistics(
        self,
        per_question_rows: Sequence[Dict[str, Any]],
        evidence_rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        baseline = self.cfg["statistics.baseline_chunker"]
        methods = {r["chunker"] for r in per_question_rows}
        if baseline not in methods:
            self.log.warning(
                "baseline %r not present (methods: %s); skipping significance tests",
                baseline,
                sorted(methods),
            )
            return []

        by_metric: Dict[str, Dict[str, Dict[str, float]]] = {}

        def collect(metric: str, rows: Sequence[Dict[str, Any]], key: str) -> None:
            table: Dict[str, Dict[str, float]] = {}
            for row in rows:
                value = row.get(key)
                if value is None:
                    continue
                table.setdefault(row["chunker"], {})[row["qid"]] = float(value)
            if table:
                by_metric[metric] = table

        collect("qa_f1", per_question_rows, "qa_f1")
        collect("exact_match", per_question_rows, "exact_match")
        collect("faithfulness", per_question_rows, "faithfulness")
        collect("context_tokens", per_question_rows, "context_tokens")
        if evidence_rows:
            collect("answer_token_recall", evidence_rows, "answer_token_recall")
            collect("answer_present", evidence_rows, "answer_present")

        return run_all_comparisons(
            by_metric,
            baseline=baseline,
            iterations=self.cfg["statistics.bootstrap_iterations"],
            confidence_level=self.cfg["statistics.confidence_level"],
            correction=self.cfg["statistics.multiple_comparison_correction"],
            seed=self.cfg["seed"],
        )

    # -- helpers --------------------------------------------------------------

    def _progress(self, items, description: str):
        if not self.cfg["runtime.progress_bars"]:
            return items
        try:
            from tqdm.auto import tqdm

            return tqdm(items, desc=description, ncols=80)
        except ImportError:
            return items


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    overrides = {dest: getattr(args, dest, None) for dest in CLI_OVERRIDE_MAP}
    cfg = load_config(args.config, overrides)

    output_dir = Path(cfg["output.dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir / "run.log")

    banner("SEISMOS end-to-end RAG evaluation")
    set_global_seed(
        cfg["seed"],
        strict_determinism=cfg["determinism.strict_determinism"],
        cudnn_deterministic=cfg["determinism.cudnn_deterministic"],
        cudnn_benchmark=cfg["determinism.cudnn_benchmark"],
    )

    if abs(float(cfg["chunking.seismos.k"]) - 2.5) > 1e-9:
        log.warning(
            "SEISMOS k=%s; the paper reports every result at k=2.5.",
            cfg["chunking.seismos.k"],
        )

    stages = resolve_stages(args.stages)

    if "generate" in stages:
        preflight_generator(cfg)

    pipeline = Pipeline(cfg, args)

    started = time.perf_counter()
    pipeline.load_data()

    if "chunk" in stages or "retrieve" in stages:
        pipeline.chunk_and_retrieve("chunk" in stages, "retrieve" in stages)

    if "generate" in stages:
        pipeline.generate()

    availability = {"available": False, "reason": "faithfulness stage not run"}
    if "faithfulness" in stages:
        availability = pipeline.faithfulness()

    result: Dict[str, Any] = {}
    if "aggregate" in stages:
        result = pipeline.aggregate(availability)

    elapsed = time.perf_counter() - started

    provenance = build_provenance(
        cfg,
        extra={
            "prompt": describe_prompt(),
            "chunkers_requested": pipeline.chunker_names,
            "methods_evaluated": result.get("methods", pipeline.method_names),
            "stages_run": stages,
            "resume_enabled": pipeline.resume,
            "device": pipeline.device,
            "dataset_stats": (
                pipeline.dataset_stats.as_dict() if pipeline.dataset_stats else None
            ),
            "faithfulness_availability": availability,
            "wall_clock_seconds": round(elapsed, 2),
            "peak_gpu_gb": peak_memory_gb(),
        },
    )
    provenance.to_json(output_dir / "config_used.json")

    banner("RUN COMPLETE")
    log.info("wall clock: %.1f s (%.2f h)", elapsed, elapsed / 3600)
    log.info("artifacts written to %s", output_dir.resolve())
    for row in result.get("summary", []):
        log.info(
            "  %-22s qa_f1=%.4f em=%.4f faith=%s ctx_tok=%.0f chunk_tok=%.0f (n=%d)",
            row["chunker"],
            row["qa_f1"],
            row["exact_match"],
            f"{row['faithfulness']:.4f}" if row["faithfulness"] is not None else "n/a",
            row["avg_context_tokens"],
            row["avg_chunk_tokens"],
            row["num_questions"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
