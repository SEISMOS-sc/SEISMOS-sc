"""End-to-end chunk -> retrieve -> aggregate on fake models: wiring, resume, CSV contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seismos_rag.chunkers import build_chunkers, resolve_chunker_names
from seismos_rag.config.loader import load_config, validate_config
from seismos_rag.utils.caching import JsonlCheckpoint, NpyCache, stable_hash
from seismos_rag.utils.exports import ExportError, REQUIRED_COLUMNS, write_csv


class TestConfig:
    def test_reference_config_loads_and_validates(self):
        cfg = load_config()
        assert cfg["dataset.name"] == "narrativeqa"
        assert cfg["retrieval.index_type"] == "IndexFlatIP"
        assert cfg["embedding.normalize"] is True

    def test_cli_overrides_apply(self):
        cfg = load_config(cli_overrides={"num_samples": 7, "top_k": 3, "embedding": "bge"})
        assert cfg["dataset.num_samples"] == 7
        assert cfg["retrieval.top_k"] == 3
        assert cfg["embedding.model"] == "bge"

    def test_none_override_does_not_clobber_the_file(self):
        cfg = load_config(cli_overrides={"num_samples": None})
        assert cfg["dataset.num_samples"] == 200

    def test_nonzero_temperature_is_rejected(self):
        cfg = load_config()
        cfg.set("generation.temperature", 0.7)
        with pytest.raises(ValueError, match="greedy decoding"):
            validate_config(cfg)

    def test_unnormalised_embeddings_are_rejected(self):
        cfg = load_config()
        cfg.set("embedding.normalize", False)
        with pytest.raises(ValueError, match="cosine"):
            validate_config(cfg)

    def test_non_flat_index_is_rejected(self):
        cfg = load_config()
        cfg.set("retrieval.index_type", "IndexIVFFlat")
        with pytest.raises(ValueError, match="IndexFlatIP"):
            validate_config(cfg)

    def test_seismos_default_is_documented_against_the_paper(self):
        cfg = load_config()
        assert 2.5 in cfg["chunking.seismos.k_sweep"]
        assert cfg["chunking.seismos.sign_convention"] == "canonical"


class TestChunkerRegistry:
    def test_all_expands_to_the_five_main_methods(self):
        assert resolve_chunker_names("all") == [
            "fixed",
            "recursive",
            "langchain_semantic",
            "llamaindex_semantic",
            "seismos",
        ]

    def test_llm_chunker_is_excluded_from_all(self):
        assert "llm_chunker" not in resolve_chunker_names("all")

    def test_explicit_selection_is_preserved(self):
        assert resolve_chunker_names("seismos") == ["seismos"]
        assert resolve_chunker_names("seismos,fixed") == ["seismos", "fixed"]

    def test_duplicates_are_collapsed(self):
        assert resolve_chunker_names("fixed,fixed") == ["fixed"]

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown chunker"):
            resolve_chunker_names("nonexistent")

    def test_builds_dependency_free_chunkers(self, fake_encoder, fake_splitter):
        cfg = load_config()
        chunkers = build_chunkers(
            ["fixed", "seismos"], cfg, fake_encoder, fake_splitter, allow_missing=False
        )
        assert [c.name for c in chunkers] == ["fixed", "seismos"]

    def test_sign_convention_comparison_emits_both(self, fake_encoder, fake_splitter):
        cfg = load_config()
        cfg.set("chunking.seismos.compare_sign_conventions", True)
        names = [
            c.name
            for c in build_chunkers(["seismos"], cfg, fake_encoder, fake_splitter)
        ]
        assert names == ["seismos", "seismos_valley"]

    def test_missing_dependency_raises_unless_allowed(self, fake_encoder, fake_splitter):
        from seismos_rag.chunkers.recursive_chunker import MissingDependencyError

        cfg = load_config()
        try:
            build_chunkers(["recursive"], cfg, fake_encoder, fake_splitter, allow_missing=False)
        except MissingDependencyError:
            # Library absent: confirm allow_missing turns it into a skip.
            with pytest.raises(ValueError, match="no chunkers"):
                build_chunkers(
                    ["recursive"], cfg, fake_encoder, fake_splitter, allow_missing=True
                )


class TestCheckpointResume:
    def test_completed_units_are_skipped_on_reopen(self, tmp_path):
        path = tmp_path / "ckpt.jsonl"
        with JsonlCheckpoint(path, ("qid", "chunker")) as ckpt:
            ckpt.append({"qid": "q1", "chunker": "fixed", "value": 1})
            ckpt.append({"qid": "q2", "chunker": "fixed", "value": 2})

        with JsonlCheckpoint(path, ("qid", "chunker")) as reopened:
            assert reopened.resumed_count == 2
            assert reopened.is_done("q1", "fixed")
            assert not reopened.is_done("q3", "fixed")

    def test_appends_accumulate_across_sessions(self, tmp_path):
        path = tmp_path / "ckpt.jsonl"
        with JsonlCheckpoint(path, ("qid",)) as ckpt:
            ckpt.append({"qid": "q1"})
        with JsonlCheckpoint(path, ("qid",)) as ckpt:
            ckpt.append({"qid": "q2"})
        with JsonlCheckpoint(path, ("qid",)) as ckpt:
            assert {r["qid"] for r in ckpt.read_all()} == {"q1", "q2"}

    def test_truncated_tail_is_dropped_not_fatal(self, tmp_path):
        """Simulates a hard kill mid-write."""
        path = tmp_path / "ckpt.jsonl"
        path.write_text(
            json.dumps({"qid": "q1", "v": 1}) + "\n" + '{"qid": "q2", "v"',
            encoding="utf-8",
        )
        with JsonlCheckpoint(path, ("qid",)) as ckpt:
            assert ckpt.is_done("q1")
            assert not ckpt.is_done("q2")
            ckpt.append({"qid": "q2", "v": 2})

        with JsonlCheckpoint(path, ("qid",)) as reopened:
            assert reopened.resumed_count == 2

    def test_disabled_checkpoint_reports_nothing_done(self, tmp_path):
        ckpt = JsonlCheckpoint(tmp_path / "x.jsonl", ("qid",), enabled=False)
        ckpt.append({"qid": "q1"})
        assert not ckpt.is_done("q1")


class TestCaching:
    def test_hash_is_order_independent_for_dicts(self):
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})

    def test_hash_distinguishes_different_content(self):
        assert stable_hash("alpha") != stable_hash("beta")

    def test_hash_is_not_confused_by_concatenation(self):
        # Length-prefixing prevents ("ab","c") colliding with ("a","bc").
        assert stable_hash("ab", "c") != stable_hash("a", "bc")

    def test_npy_cache_round_trip(self, tmp_path):
        import numpy as np

        cache = NpyCache(tmp_path)
        array = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
        cache.put("key123456789abc", array)
        assert np.array_equal(cache.get("key123456789abc"), array)
        assert cache.get("missingkey123456") is None


class TestExports:
    def test_required_columns_lead_the_header(self, tmp_path):
        rows = [
            {
                "extra_diagnostic": 1,
                "qid": "q1",
                "chunker": "fixed",
                "rank": 0,
                "chunk_id": 3,
                "score": 0.9,
                "chunk_text": "text",
            }
        ]
        path = write_csv(tmp_path / "retrieval_results.csv", rows)
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header[:6] == list(REQUIRED_COLUMNS["retrieval_results.csv"])
        assert "extra_diagnostic" in header

    def test_missing_required_column_raises(self, tmp_path):
        with pytest.raises(ExportError, match="required column"):
            write_csv(tmp_path / "summary_metrics.csv", [{"chunker": "fixed"}])

    def test_none_becomes_empty_not_zero(self, tmp_path):
        """An unavailable metric must read back as NaN, never as 0."""
        rows = [
            {
                "qid": "q1",
                "chunker": "fixed",
                "question": "q",
                "reference_answer": "a",
                "generated_answer": "b",
                "qa_f1": 0.5,
                "exact_match": 0,
                "faithfulness": None,
                "context_tokens": 100,
            }
        ]
        path = write_csv(tmp_path / "per_question_results.csv", rows)
        line = path.read_text(encoding="utf-8").splitlines()[1]
        assert ",," in line  # the empty faithfulness cell

        pd = pytest.importorskip("pandas")
        frame = pd.read_csv(path)
        assert frame["faithfulness"].isna().all()
        assert frame["exact_match"].iloc[0] == 0

    def test_empty_rows_still_write_a_header(self, tmp_path):
        path = write_csv(tmp_path / "chunk_statistics.csv", [])
        assert path.read_text(encoding="utf-8").strip().split(",")[:6] == list(
            REQUIRED_COLUMNS["chunk_statistics.csv"]
        )

    def test_lists_survive_a_json_round_trip(self, tmp_path):
        path = write_csv(tmp_path / "misc.csv", [{"ids": [1, 2, 3]}])
        pd = pytest.importorskip("pandas")
        value = pd.read_csv(path)["ids"].iloc[0]
        assert json.loads(value) == [1, 2, 3]


class TestPromptContract:
    def test_template_matches_the_specification(self):
        from seismos_rag.generation.prompt import PROMPT_TEMPLATE

        assert PROMPT_TEMPLATE.startswith("You are a helpful assistant.")
        assert "Answer the question using only the provided context." in PROMPT_TEMPLATE
        assert "{context}" in PROMPT_TEMPLATE and "{question}" in PROMPT_TEMPLATE
        assert PROMPT_TEMPLATE.rstrip().endswith("Answer:")

    def test_hash_is_stable(self):
        from seismos_rag.generation.prompt import prompt_hash

        assert prompt_hash() == prompt_hash()

    def test_within_budget_prompts_are_untouched(self):
        from seismos_rag.generation.prompt import build_prompt_with_budget
        from tests.conftest import FakeTokenizer

        built = build_prompt_with_budget(
            "short context", "a question?", FakeTokenizer(), max_input_tokens=1000
        )
        assert built.truncated is False
        assert "short context" in built.text

    def test_oversized_context_is_truncated_in_the_middle(self):
        from seismos_rag.generation.prompt import TRUNCATION_MARKER, build_prompt_with_budget
        from tests.conftest import FakeTokenizer

        context = " ".join(f"w{i}" for i in range(2000))
        built = build_prompt_with_budget(
            context, "a question?", FakeTokenizer(), max_input_tokens=200
        )
        assert built.truncated is True
        assert TRUNCATION_MARKER.strip() in built.text
        assert built.dropped_context_tokens > 0
        # The question survives intact; only the context is cut.
        assert "a question?" in built.text
        assert "w0" in built.text and "w1999" in built.text


class TestEndToEnd:
    """chunk -> retrieve -> stub generate -> aggregate, on fake models."""

    def test_full_path_produces_the_required_artifacts(
        self, tmp_path, fake_encoder, fake_splitter, monkeypatch
    ):
        import argparse

        from seismos_rag.chunkers.base import ChunkingContext
        from seismos_rag.retrieval.retriever import Retriever
        from seismos_rag.run_pipeline import Pipeline

        cfg = load_config()
        cfg.set("output.dir", str(tmp_path / "results"))
        cfg.set("statistics.bootstrap_iterations", 200)
        cfg.set("runtime.progress_bars", False)

        args = argparse.Namespace(
            chunkers="fixed,seismos",
            no_resume=False,
            allow_missing_chunkers=True,
            limit_docs=None,
        )
        pipeline = Pipeline(cfg, args)

        # -- fake corpus --
        samples = []
        for doc in range(4):
            sentences = [
                f"Document {doc} sentence {i} discusses subject {i // 4} in detail."
                for i in range(20)
            ]
            samples.append(
                {
                    "id": f"q{doc}",
                    "question": f"What does document {doc} discuss?",
                    "context": " ".join(sentences),
                    "answers": [f"subject {doc % 5}", "a subject"],
                }
            )
        pipeline.samples = samples
        pipeline.sample_by_id = {s["id"]: s for s in samples}

        chunkers = build_chunkers(
            ["fixed", "seismos"], cfg, fake_encoder, fake_splitter, allow_missing=True
        )
        pipeline.method_names = [c.name for c in chunkers]
        retriever = Retriever(fake_encoder, top_k=3)

        chunk_ckpt = pipeline._checkpoint("chunks.jsonl", ("qid", "chunker"))
        retrieval_ckpt = pipeline._checkpoint("retrieval.jsonl", ("qid", "chunker"))
        generation_ckpt = pipeline._checkpoint("generations.jsonl", ("qid", "chunker"))
        faith_ckpt = pipeline._checkpoint("faithfulness.jsonl", ("qid", "chunker"))

        from seismos_rag.chunkers import chunk_statistics

        for sample in samples:
            split = fake_splitter.split(sample["context"])
            counts = fake_encoder.count_tokens(split.sentences)
            ctx = ChunkingContext(
                doc_id=sample["id"],
                text=sample["context"],
                sentences=split.sentences,
                spans=split.spans,
                sentence_token_counts=counts,
                sentence_embeddings=fake_encoder.encode_sentences(split.sentences),
                total_document_tokens=sum(counts),
            )
            for chunker in chunkers:
                chunks = chunker.chunk(ctx)
                chunk_ckpt.append(
                    {
                        "qid": sample["id"],
                        "chunker": chunker.name,
                        "chunks": chunks,
                        "stats": chunk_statistics(chunks, ctx.total_document_tokens),
                        "params": chunker.params(),
                    }
                )
                result = retriever.retrieve(
                    sample["id"], chunker.name, sample["question"], chunks
                )
                retrieval_ckpt.append(result.to_record())
                generation_ckpt.append(
                    {
                        "qid": sample["id"],
                        "chunker": chunker.name,
                        "prompt": "PROMPT",
                        "answer": "a subject" if chunker.name == "seismos" else "unrelated",
                        "input_tokens": 300,
                        "output_tokens": 5,
                        "prompt_truncated": False,
                        "context_tokens": 250,
                        "dropped_context_tokens": 0,
                        "generation_seconds": 0.1,
                    }
                )
                faith_ckpt.append(
                    {
                        "qid": sample["id"],
                        "chunker": chunker.name,
                        "faithfulness": 0.8 if chunker.name == "seismos" else 0.4,
                        "reason": None,
                    }
                )

        for ckpt in (chunk_ckpt, retrieval_ckpt, generation_ckpt, faith_ckpt):
            ckpt.close()

        result = pipeline.aggregate({"available": True, "reason": None})

        out = Path(cfg["output.dir"])
        for name in (
            "summary_metrics.csv",
            "per_question_results.csv",
            "retrieval_results.csv",
            "chunk_statistics.csv",
            "evidence_metrics.csv",
            "metric_availability.json",
        ):
            assert (out / name).is_file(), f"{name} was not written"

        pd = pytest.importorskip("pandas")

        summary = pd.read_csv(out / "summary_metrics.csv")
        assert set(summary["chunker"]) == {"fixed", "seismos"}
        assert (summary["num_questions"] == 4).all()
        # The stub gave seismos the correct answer, so it must score higher.
        seismos_f1 = summary.loc[summary.chunker == "seismos", "qa_f1"].iloc[0]
        fixed_f1 = summary.loc[summary.chunker == "fixed", "qa_f1"].iloc[0]
        assert seismos_f1 > fixed_f1

        per_question = pd.read_csv(out / "per_question_results.csv")
        assert len(per_question) == 8
        assert per_question["faithfulness"].notna().all()

        retrieval = pd.read_csv(out / "retrieval_results.csv")
        assert set(retrieval["rank"]) <= {0, 1, 2}
        assert retrieval["chunk_text"].notna().all()

        chunk_stats = pd.read_csv(out / "chunk_statistics.csv")
        assert (chunk_stats["num_chunks"] > 0).all()
        assert (chunk_stats["total_document_tokens"] > 0).all()

        availability = json.loads((out / "metric_availability.json").read_text())
        assert "evidence_completeness" in availability["unavailable"]
        assert result["n_per_question_rows"] == 8

    def test_main_writes_provenance_and_honours_stage_selection(
        self, tmp_path, monkeypatch
    ):
        """Exercises main(): argv -> config -> seeding -> provenance."""
        import seismos_rag.run_pipeline as pipeline_module
        from seismos_rag.data.longbench_loader import DatasetStats

        samples = [
            {
                "id": f"q{i}",
                "question": f"Question {i}?",
                "context": " ".join(f"Sentence {j} of document {i}." for j in range(30)),
                "answers": ["an answer"],
                "source_index": i,
                "order": i,
                "context_sha256": "0" * 64,
            }
            for i in range(3)
        ]
        stats = DatasetStats(
            name="narrativeqa", split="test", n_total=200, n_selected=3, seed=42,
            mean_context_chars=900.0, median_context_chars=900.0,
            min_context_chars=900, max_context_chars=900,
            mean_answers_per_question=1.0, source_repo="fake/LongBench",
        )
        monkeypatch.setattr(
            pipeline_module,
            "load_longbench_narrativeqa_with_stats",
            lambda **kwargs: (samples, stats),
        )

        out = tmp_path / "results"
        exit_code = pipeline_module.main(
            [
                "--chunkers", "fixed",
                "--stages", "aggregate",
                "--output_dir", str(out),
                "--num_samples", "3",
                "--seismos_k", "2.5",
            ]
        )
        assert exit_code == 0

        provenance = json.loads((out / "config_used.json").read_text(encoding="utf-8"))
        assert provenance["config"]["seed"] == 42
        assert provenance["config"]["dataset"]["num_samples"] == 3
        assert provenance["config"]["chunking"]["seismos"]["k"] == 2.5
        assert provenance["resolved"]["stages_run"] == ["aggregate"]
        assert provenance["resolved"]["prompt"]["template_sha256_16"]
        assert provenance["resolved"]["packages"]["torch"]
        assert provenance["resolved"]["dataset_stats"]["n_selected"] == 3
        assert (out / "dataset_manifest.json").is_file()
        assert (out / "run.log").is_file()

        # --output_dir must relocate the checkpoints too, or a second run would
        # silently resume the first one's work into a different directory.
        assert (out / "checkpoints").is_dir()
        assert provenance["config"]["output"]["checkpoint_dir"] == str(out / "checkpoints")
        assert not (Path.cwd() / "results" / "checkpoints").exists() or (
            Path.cwd() / "results"
        ) == out

    def test_invalid_stage_is_rejected(self):
        from seismos_rag.run_pipeline import resolve_stages

        with pytest.raises(ValueError, match="unknown stage"):
            resolve_stages("chunk,teleport")

    def test_generate_without_retrieval_explains_how_to_recover(self, tmp_path):
        """The most common hosted-notebook mistake must be self-diagnosing."""
        import argparse

        from seismos_rag.run_pipeline import Pipeline

        cfg = load_config()
        cfg.set("output.dir", str(tmp_path / "results"))
        args = argparse.Namespace(
            chunkers="fixed", no_resume=False, allow_missing_chunkers=True, limit_docs=None
        )
        pipeline = Pipeline(cfg, args)

        with pytest.raises(RuntimeError) as excinfo:
            pipeline.generate()

        message = str(excinfo.value)
        assert "retrieval.jsonl" in message
        assert "--stages generate" in message
        assert "wipe the working directory" in message
        # Must hand back a runnable command, not just a diagnosis.
        assert "python run_pipeline.py --chunkers all" in message

    def test_second_pass_skips_completed_units(self, tmp_path):
        """Resume must not duplicate work already checkpointed."""
        path = tmp_path / "chunks.jsonl"
        with JsonlCheckpoint(path, ("qid", "chunker")) as ckpt:
            for qid in ("q0", "q1"):
                for chunker in ("fixed", "seismos"):
                    ckpt.append({"qid": qid, "chunker": chunker, "chunks": []})

        with JsonlCheckpoint(path, ("qid", "chunker")) as reopened:
            pending = [
                (qid, chunker)
                for qid in ("q0", "q1", "q2")
                for chunker in ("fixed", "seismos")
                if not reopened.is_done(qid, chunker)
            ]
        assert pending == [("q2", "fixed"), ("q2", "seismos")]
