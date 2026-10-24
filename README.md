# FastAPI Documentation RAG Assistant

A small, reproducible Retrieval-Augmented Generation (RAG) project for **AAI 655O — Bridging Search and Generative AI with RAG**. It answers questions using a selected set of official FastAPI documentation pages, cites the retrieved passages, and refuses questions when the evidence is insufficient.

## What it does

1. Parses **20 selected English FastAPI tutorial Markdown pages** and expands their referenced Python examples.
2. Splits content by Markdown headings, then into overlapping chunks; enforces the embedding model's token limit.
3. Embeds passages with `sentence-transformers/all-MiniLM-L6-v2` and stores normalized vectors in persistent ChromaDB using cosine distance.
4. Retrieves up to **12 candidates** with cosine similarity >= **0.35**, reranks them with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and passes the **best 4** passages to the generator.
5. Generates an evidence-grounded answer with numbered citations, or returns the exact refusal sentence when evidence is insufficient.

The evaluated index contained **555 chunks**. The generator supports Gemini (used in the evaluation), OpenAI, and local Ollama.

## Requirements

- Python **3.13** (see `.python-version` and `pyproject.toml`).
- [uv](https://docs.astral.sh/uv/) for environment and dependency management.
- Internet access on first use to download the embedding and reranking models.
- A Gemini or OpenAI API key, or a running Ollama server, for answer generation.

## Quick start

```bash
git clone https://github.com/alainlau/fastapi-documentation-rag.git
cd fastapi-documentation-rag
uv sync
```

Copy `.env.example` to `.env` and set your provider's credentials. In Git Bash/macOS/Linux:

```bash
cp .env.example .env
```

In Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

For the evaluated Gemini setup, use `LLM_PROVIDER=gemini`, a valid `GEMINI_API_KEY`, and the model name in `.env.example` (availability depends on your account). `JUDGE_MODEL` is optional and defaults to the value specified in `config.py`.

### Included documentation inputs

The repository includes the **20 selected Markdown pages** in `data/corpus/`, plus the corresponding official FastAPI source documentation under `data/raw/docs/` and referenced Python examples under `data/raw/docs_src/`. These are included to make the indexing process reproducible without requiring a separate upstream checkout.

The included corpus is a **snapshot**, not a live synchronization with FastAPI. For the evaluated results below, use the committed `data/corpus/` files unchanged. `prepare_corpus.py` is only needed when intentionally rebuilding the selected corpus from `data/raw/docs/`.

### Build the index and ask questions

From the project root:

```bash
uv run python ingest.py
uv run python app.py
```

`ingest.py` recreates the Chroma collection. The index is written to `chroma_db/` and is ignored by Git. The included `data/raw/docs_src/` directory supplies the Python examples referenced by the selected Markdown files.

Example questions:

- How do I validate incoming JSON data in FastAPI?
- How can I return a custom error when a request contains invalid data?
- How do I reuse the same database connection logic across several endpoints?
- What will the weather be tomorrow?

The last question is intentionally out of scope. The expected refusal is:

> I cannot answer this question based on the provided documentation.

The CLI displays each selected evidence passage's source, heading and cosine similarity before the answer. Citation markers such as `[1]` refer to those displayed passages. Type `exit` to quit.

## Configuration

The central `CONFIG` dictionary in `config.py` controls ingestion, retrieval, reranking, generation and judging.

| Setting | Evaluated value |
|---|---|
| Corpus | 20 FastAPI English tutorial Markdown files |
| Chunk size / overlap | 256 / 60 (`cl100k_base` splitter) |
| Embedding limit | 240 model tokens, with further splitting as needed |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database / metric | ChromaDB / cosine |
| First-stage retrieval | Similarity, top 12, threshold 0.35 |
| Advanced retrieval | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator context | Top 4 reranked passages |
| Generator / judge in experiment | Gemini, `gemini-3.6-flash` as recorded in the saved evaluation |

`use_reranker` can be set to `False` for the non-reranked pipeline. An alternative MMR retrieval mode remains in the code as a prior experiment; the final evaluated configuration uses similarity followed by cross-encoder reranking.

## Evaluation and results

The fixed gold test set is `evaluation/test_questions.json`. It contains 11 scored retrieval questions, a corpus-wide challenge, and out-of-scope questions. Retrieval metrics score the **first four results** against the gold section labels; out-of-scope and global questions are excluded from those retrieval metrics.

Run the retrieval comparisons after building the index:

```bash
uv run python evaluation/evaluate_retrieval.py
uv run python evaluation/compare_reranking.py
```

| Metric (11 scored questions) | Similarity baseline | Cross-encoder reranking |
|---|---:|---:|
| Hit Rate@4 | 36.4% (4/11) | **63.6% (7/11)** |
| MRR@4 | 0.258 | **0.439** |
| Mean measured retrieval latency | 115 ms | 296 ms |

The same 12-candidate first-stage pool and 0.35 threshold were used in the comparison. Gold-section candidate recall was **8/11 (72.7%)**: reranking cannot recover passages absent from its initial pool. Timings are from a small local experiment, include first-query effects, and are not production benchmarks.

Saved measurements are in `evaluation/baseline_results.json` and `evaluation/reranking_comparison.json`.

### Generation evaluation

```bash
uv run python evaluation/evaluate_generation.py
```

This makes generator **and** Gemini judge API calls. It evaluates nine selected questions using the final pipeline and writes full answers, numbered retrieved passages, and judge rationales to `evaluation/generation_results.json`. The judge scores **context relevance, groundedness and answer relevance** on a 1–3 scale.

In the saved run, average scores were **2.78 / 3.00 / 2.78**, respectively. Seven of nine evaluations received full marks on all three dimensions. The assistant still refused an in-scope question about protecting endpoints for logged-in users because the retrieved context lacked implementation detail. A corpus-wide security-count question was also refused appropriately because four passages cannot establish a global count. These limitations are reported rather than hidden.

Judge verdicts are model-generated estimates, not ground truth. The final report includes manual checks of at least two verdicts.

## Repository guide

| Path | Purpose |
|---|---|
| `app.py` | Interactive CLI and final retrieval → rerank → generate pipeline |
| `config.py` | Central experiment and runtime settings |
| `prepare_corpus.py` | Optional extraction of 20 pages from an upstream Markdown copy |
| `markdown_parser.py` | Expands upstream Python include directives |
| `compare_chunking.py` | Heading-aware chunking and original chunk-size comparison |
| `ingest.py` | Token-limit enforcement, embeddings and Chroma indexing |
| `inspect_corpus.py`, `test_parser.py` | Corpus inspection and parser checks |
| `data/corpus/` | Frozen selection of 20 FastAPI documentation Markdown files |
| `data/raw/docs/`, `data/raw/docs_src/` | Included upstream Markdown documentation and referenced Python examples |
| `evaluation/` | Frozen questions, comparison scripts, judge script, saved results |
| `.env.example` | Provider configuration template (no secrets) |

## Limitations and reproducibility

- Only 20 selected pages are indexed; the assistant does not cover all FastAPI features or track the latest documentation automatically.
- Cross-encoder reranking improves ranking but cannot recover missing first-stage candidates.
- The 0.35 similarity threshold may exclude relevant passages, particularly paraphrased security questions.
- Local model inference adds latency; LLM generation and judging may incur API costs.
- The included upstream documentation is a snapshot; FastAPI and model versions can change. The saved corpus and evaluation JSON preserve the evaluated inputs, but exact future reproduction would also benefit from recording the upstream source commit and model revisions.
- Reindex after documentation, chunking, or embedding-model changes. In a production adaptation, monitor **retrieval hit/citation quality** and **refusal rate for known in-scope questions**, alongside latency and API failures.

Documentation content and examples originate from the [official FastAPI repository](https://github.com/fastapi/fastapi). See its repository for upstream licensing and attribution.

## Updating the FastAPI documentation (optional maintenance)

The project **does not automatically fetch updates**. If FastAPI publishes new documentation, you can refresh the local source and rebuild the index manually:

1. Obtain the latest [official FastAPI source](https://github.com/fastapi/fastapi) (for example, `git clone https://github.com/fastapi/fastapi.git fastapi-upstream` or `git -C fastapi-upstream pull`). Record the upstream commit ID for traceability.
2. Replace `data/raw/docs/` with the upstream **English Markdown documentation** (typically `docs/en/docs/`), preserving the relative paths such as `tutorial/body.md`. Replace `data/raw/docs_src/` with the matching upstream **`docs_src` Python examples**, preserving their subdirectories. Both directories must come from the same upstream revision.
3. From the project root, rebuild the selected 20-page corpus and its vector index:

   ```bash
   uv run python prepare_corpus.py
   uv run python ingest.py
   ```

4. Smoke-test the CLI and rerun the evaluations:

   ```bash
   uv run python app.py
   uv run python evaluation/evaluate_retrieval.py
   uv run python evaluation/compare_reranking.py
   uv run python evaluation/evaluate_generation.py
   ```

**Important:** `prepare_corpus.py` deletes and recreates `data/corpus/`, and `ingest.py` rebuilds the Chroma collection. Make a Git commit or backup first. An updated corpus may produce a different number of chunks, retrieval scores, answers, and judge verdicts; the published Week 3 results above apply only to the original evaluated snapshot. Inspect missing-file warnings and parser/include errors before trusting a refreshed index.
