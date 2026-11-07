# SEISMOS end-to-end RAG evaluation

Does better chunking improve **answers**, not just retrieval scores? This
harness measures it on **LongBench NarrativeQA**. Retrieval and generation stay
fixed across five chunkers, and only the chunk boundaries change:

| Chunker | Implementation |
|---|---|
| `fixed` | sentence-preserving, fixed token budget |
| `recursive` | LangChain `RecursiveCharacterTextSplitter`, token-budgeted |
| `langchain_semantic` | LangChain `SemanticChunker` (95th percentile) |
| `llamaindex_semantic` | LlamaIndex `SemanticSplitterNodeParser` (buffer 1, 95th percentile) |
| `seismos` | SEISMOS discrete-Laplacian detector, `k = 2.5` |

The pipeline:

1. One E5-base-v2 encoder produces the sentence, chunk and query embeddings.
2. FAISS `IndexFlatIP` retrieves the top-5 chunks.
3. Qwen2.5-7B-Instruct answers each question with greedy decoding and one fixed prompt.

All chunkers share a 480-token budget.

The SEISMOS detector is imported from the core library (`seismos.detectors`).
The RAG results and the retrieval experiments therefore run the same code.

---

## Installation

From this directory:

```bash
pip install -r requirements.txt        # also installs the core library from ..
python -m spacy download en_core_web_sm
```

**`transformers < 5`.** The HHEM-2.1-Open faithfulness model uses remote code
that does not import under transformers 5.x. With 5.x, the pipeline still runs,
but every faithfulness value is written as `null`.

Check the install without network access or model weights:

```bash
python -m pytest
```

---

## Running

```bash
python run_pipeline.py --chunkers all --generator qwen --num_samples 200
```

```bash
python run_pipeline.py --chunkers seismos --run_k_sweep    # SEISMOS only, with the k sensitivity sweep
```

### GPU memory

Qwen2.5-7B-Instruct in fp16 needs 15.2 GB for its weights alone. That does not
fit on a single 16 GB T4.

```bash
python run_pipeline.py --chunkers all --generator qwen                     # 2 GPUs: device_map=auto shards the weights
```

```bash
python run_pipeline.py --chunkers all --generator qwen --quantization 4bit # 1 x 16 GB: NF4 weights, ~5 GB
```

Attention defaults to SDPA, which never materialises the full score matrix.
Before downloading anything, a preflight check estimates weights + KV cache +
attention workspace against the free memory. If an OOM happens anyway,
`generation.oom_backoff` halves the batch and retries. Affected rows record their
`effective_batch_size`.

If generation looks degenerate (empty or repeated-token output), run
`python tools/diagnose_generation.py`. It tests fp16 and 4-bit loading in isolation.

### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--chunkers` | `all` | `all` = fixed, recursive, langchain_semantic, llamaindex_semantic, seismos (the optional LLM chunker is never included) |
| `--generator` | `qwen` | `qwen` \| `llama` \| any HF model id |
| `--embedding` | `e5` | `e5` \| `bge` \| `gte`; used for sentence, chunk and query embeddings |
| `--num_samples` | 200 | seeded subsample |
| `--top_k` | 5 | retrieved chunks per question |
| `--seismos_k` | 2.5 | the paper's operating point |
| `--run_k_sweep` | off | adds `seismos_k1.5 … seismos_k3.5` |
| `--quantization` | `none` | `4bit` for a single 16 GB card |
| `--attn` | `auto` | `auto` resolves to `sdpa` |
| `--gen_batch_size` | 4 | activations scale linearly with batch size |
| `--max_input_tokens` | 3072 | prompt ceiling; longer contexts are middle-truncated |
| `--stages` | `all` | any of `chunk,retrieve,generate,faithfulness,aggregate` |
| `--no-resume` | off | discard checkpoints and recompute |
| `--allow-missing-chunkers` | off | skip baselines whose library is missing (this removes them from the comparison) |

A missing baseline library is a hard error by default, so an incomplete
comparison never looks complete. The full configuration is
`seismos_rag/config/config.yaml`.

### Resuming

Each stage keeps an append-only JSONL checkpoint keyed by `(qid, chunker)`. To
resume after an interruption, re-run the same command. A checkpoint truncated by
a hard kill loses only its damaged tail.

---

## Design invariants

- **One encoder for everything.** LangChain and LlamaIndex are driven through
  adapters onto the same encoder instance, so every comparison is chunking
  against chunking, never encoder against encoder.
- **Shared 480-token cap** (`chunking.cap_all_chunkers`). Without it,
  context-efficiency comparisons would measure which method was allowed longer
  chunks.
- **Recursive chunking is token-budgeted** (`from_huggingface_tokenizer`). A
  480-*character* budget would be about 110 tokens.
- **Inputs are used byte-for-byte** as distributed; no cleaning.
- **Greedy decoding** (`do_sample=False`).

---

## Output (`results/`)

| File | Contents |
|---|---|
| `summary_metrics.csv` | one row per chunker: QA F1, exact match, faithfulness, context and chunk tokens, latencies |
| `per_question_results.csv` | one row per (question, chunker); every summary number traces back here |
| `retrieval_results.csv` | one row per retrieved chunk (rank, score, text) |
| `chunk_statistics.csv` | per (question, chunker) chunk counts and token-length distribution |
| `significance_tests.csv` | paired bootstrap CI, Wilcoxon signed-rank and Holm correction against `statistics.baseline_chunker` |
| `evidence_metrics.csv` | label-free evidence proxies |
| `metric_availability.json` | which metrics were computed, and why the others were not |
| `seismos_signal_diagnostics.csv` | per-document `std(X)`, `std(D)`, ACF factor, lag-1 autocorrelation |
| `config_used.json` | resolved config, package versions, git revision, device, prompt hash |
| `raw_generations.csv`, `latency_log.csv`, `dataset_manifest.json`, `run.log` | full prompts and outputs, timings, exact sample selection |

### Reading the numbers

- **QA F1 / EM** use the official LongBench normalisation and take the maximum
  over the reference answers.
- **Faithfulness** is the HHEM probability that the answer is supported by the
  retrieved context. It measures grounding, not correctness.
- **Context tokens** are counted with the generator's tokenizer. Chunk tokens
  are counted with the encoder's tokenizer. Compare each only within its own
  column.
- **Evidence metrics.** NarrativeQA has no gold evidence spans, so evidence
  precision, recall and F1 are written as explicit nulls, never imputed. The
  proxies computed from the reference answer measure whether supporting text is
  *available*, not whether it is correct. Compare them across methods; don't read
  their absolute values.
- **A null is not a zero.** Unavailable metrics are written as empty cells.

### Reproducibility

The following are fixed by construction: greedy decoding, exact flat-index
search, seeded subsampling, and a seeded bootstrap.

Two settings are part of the experiment and are recorded in `config_used.json`:

- **`generation.batch_size`**: padding changes batched matmul numerics.
- **The resolved attention backend**: SDPA and eager agree to within fp16 rounding, not bit for bit.

Chunk token counts are encoder-specific.

---

## Layout

```
seismos_rag/
  config/            config.yaml, loader + provenance capture
  data/              LongBench loader (download, cache, seeded subsample)
  chunkers/          sentence splitter, shared chunk contract, the five chunkers (+ optional LLM chunker)
  embeddings/        shared encoder and LangChain / LlamaIndex adapters
  retrieval/         FAISS index, per-document top-k retriever
  generation/        the fixed prompt, batched greedy decoding, VRAM preflight
  evaluation/        LongBench F1/EM, HHEM faithfulness, evidence proxies, statistics
  utils/             logging, seeding, GPU, caching, exports
  run_pipeline.py    orchestrator
tools/diagnose_generation.py
tests/               offline tests (no weights or network by default)
```

`python -m pytest -m ""` also runs the tests that download models or data.
