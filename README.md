# PhilHealth Policy & Benefits RAG Agent

A local, citation-first assistant that answers questions about PhilHealth (Philippine Health Insurance Corporation) circulars and advisories, and refuses when the documents don't support an answer.

> Status: **Phase 1, Stage 2 (upgrade loop)**. Runs fully locally on Ollama. Latest eval: **21 / 22 golden questions pass**.

---

## Background

Insurance policy documents are long, dense and full of exact numbers: rates, floors and ceilings, day limits, deadlines. A wrong figure in an answer is worse than no answer. PhilHealth publishes these rules as PDF circulars and advisories, some with a text layer and some only as scanned images.

This project is a Retrieval-Augmented Generation (RAG) agent built around three requirements that matter in insurance:

- **Verified citations.** Every answer must point to a source document, a page and a quote, and the quote is checked against the retrieved text in code. The model's word isn't taken for it.
- **Compliance-minded refusals.** If the excerpts don't contain the answer, the agent says so and returns no citations, rather than guessing.
- **Measured quality.** Every change is evaluated against a golden set of questions with known answers and pages, so improvements (and regressions) are visible.

It's built **skeleton first**: a thin end-to-end pipeline came first, and pieces are being upgraded one at a time, with an eval run after each change. All model access goes through one config file, so the same code can later run on AWS Bedrock.

**Documents used so far** (public PhilHealth issuances):

| File | What it is | Pages | Text |
|---|---|---|---|
| PA2025-0002.pdf | Advisory: 2025 premium contribution for direct contributors | 1 | Text layer |
| PC2026-0002.pdf | Circular on GAMOT facilities (main body pp. 1–8, Annexes A–D pp. 9–28) | 28 | Text layer, with some OCR errors |
| PC2024-0013.pdf | Circular on the Konsulta benefit package | 32 | Scanned: needs OCR (golden set written, not yet in the index) |

---

## What's built so far

```text
Question -> [Guard] -> [Retrieve] -> [Grade] --not relevant--> [Rewrite] -> back to Retrieve
                                        | relevant
                                        v
                                   [Generate] -> [Check] --unsupported--> [Fallback]
                                   (Pydantic)       | grounded
                                                    v
                                  Answer + citations + per-node timings
```

| Area | What it does now |
|---|---|
| **Pipeline** | LangGraph state machine with every node and edge wired. Retrieve and Generate are real; Guard, Grade, Rewrite, Check and Fallback are still stubs |
| **Ingest** | PyMuPDF text extraction, with a Tesseract OCR fallback for scanned pages, cached per page |
| | Whitespace clean-up: single newline → space, paragraph breaks kept |
| | List-aware chunking (paragraph → numbered item → lettered item → sentence), 700 characters with 100 overlap |
| | Abbreviation expansion from config, e.g. PRO → "PhilHealth Regional Office (PRO)", so both forms are searchable |
| | `doc_part` metadata ("main" / "annex") per page, from the footer or annex heading. Stored, not yet used in retrieval |
| | Deterministic chunk IDs (`<file>-p<page>-c<i>`), so re-ingesting overwrites instead of duplicating |
| **Retrieval** | Hybrid search: vector top 10 + BM25 top 10, fused with Reciprocal Rank Fusion (RRF, k=60), top 4 sent to the LLM. A `mode: vector / hybrid` switch in config |
| **Generation** | One LLM call with structured output (`Answer` = answer + citations of source, page and quote) |
| **Citation check** | Each quote must appear in a retrieved chunk; source and page are taken from the matched chunk, not from the LLM. Matching ignores spacing, quote styles, punctuation, case and common OCR errors (o/0) |
| **Telemetry** | Per-node timings in the graph state, merged by a reducer |
| **Eval** | Pytest harness over a golden set: checks the answer, the cited page (any of a list of valid pages) and refusals on unanswerable questions |
| **UI** | Minimal Streamlit page (question in, answer out) |

---

## Important files

| File | Description |
|---|---|
| `config.yaml` | All settings: provider (Ollama / Bedrock), models, chunk size and overlap, abbreviations, retrieval mode and k values, graph settings |
| `src/config.py` | Loads the config; `get_llm()` and `get_embeddings()` are the only place that knows which provider is used |
| `src/ingest.py` | PDF → text (or OCR) → clean-up → abbreviation expansion → chunks with metadata → Chroma |
| `src/vector_store.py` | One cached Chroma instance, stored under `data/chroma/<provider>/` |
| `src/retriever.py` | `vector_search`, `bm25_search` and `hybrid_search` (RRF fusion) |
| `src/schemas.py` | Pydantic `Citation` / `Answer`, and the LangGraph `QueryState` |
| `src/nodes.py` | Graph nodes: guard, retrieve, grade, generate (with the citation check), rewrite, check, fallback |
| `src/state_machine.py` | Wires the nodes and edges and compiles the graph as `app` |
| `app.py` | Streamlit UI |
| `eval/test_run_eval.py` | Pytest eval over the golden set, with answer normalisation (currency, separators, trailing zeros) |
| `eval/golden_set.jsonl` | 22 questions on PA2025-0002 and PC2026-0002 (3 unanswerable) |
| `eval/golden_set_pc2024_0013.jsonl` | 30 questions on PC2024-0013 (waiting for that document to be ingested) |
| `scripts/check_providers.py`, `scripts/check_schemas.py` | Smoke tests for the model connection and the Pydantic schemas |
| `docs/PHASE1_PLAN.md` | The build plan: stages, upgrades and how each is measured |
| `docs/LEARNING_LOG.md` | Dated journal: what was built, surprises, takeaways |
| `docs/DECISIONS.md` | Architecture decision records (e.g. local Tesseract OCR) |
| `docs/CHUNK_K_COMPARISON.md` | Chunk size and k experiments |

---

## How to run

**Requirements:** macOS or Linux, Python 3.12, [Ollama](https://ollama.com), and [Tesseract](https://github.com/tesseract-ocr/tesseract) for scanned PDFs. Tested on an M1 with 16 GB RAM.

1. **Install Ollama, the models and Tesseract**
   ```bash
   brew install ollama tesseract
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
   Keep Ollama running (the Ollama app, or `ollama serve`).

2. **Create the environment**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Check the setup**
   ```bash
   python -m scripts.check_providers   # a model reply, then 768 (embedding size)
   python -m scripts.check_schemas
   ```

4. **Add the documents.** Download the PDFs listed above from the PhilHealth website into `data/raw/`. PDFs aren't committed to the repo.

5. **Build the index.** Run every command from the project root.
   ```bash
   python -m src.ingest
   ```
   After changing chunk settings, start from an empty index (delete `data/chroma/ollama/`) so old chunks don't linger.

6. **Ask a question**
   ```bash
   python -m src.state_machine        # runs a built-in sample question
   streamlit run app.py               # browser UI
   ```

7. **Run the eval**
   ```bash
   python -m pytest eval/test_run_eval.py -s     # -s shows answers and timings
   python -m pytest eval/test_run_eval.py -s -x  # stop at the first failure
   ```
   A full run of 22 questions takes about 6–7 minutes locally.

---

## Results so far

Eight eval runs on the 22-question golden set. Each row changes one or more settings from the previous row.

| Run | Settings | Main change | Passed | Avg time / question |
|---|---|---|---|---|
| R1 | 200 / 50, vector | Baseline, 3 documents, one expected page per question | 11 / 22 (50%) | 13.3 s |
| R2 | 600 / 100, vector | Bigger chunks, whitespace clean-up, lists of valid pages in the golden set | 16 / 22 (73%) | 22.4 s |
| R3 | 700 / 100, vector | First 700 run, stopped at the first failure | 8 / 9 (partial) | 14.7 s |
| R4 | 700 / 100, vector | List-aware split, abbreviations, `doc_part`, citation prompt and filter | 18 / 22 (82%) | 14.9 s |
| R5 | 700 / 100, hybrid | BM25 + RRF, but on a stale index (re-ingested without a reset) | 19 / 22 (86%) | 18.2 s |
| R6 | 700 / 100, hybrid | Clean re-ingest | 20 / 22 (91%) | 17.7 s |
| R7 | 700 / 100, hybrid | Citation check by quote instead of by filename | 19 / 22 (86%) | 16.9 s |
| R8 | 700 / 100, hybrid | `strip_space()` normalisation in the quote check | **21 / 22 (95%)** | 17.4 s |

**Latest run (R8), in detail:**

| Measure | Value |
|---|---|
| Pass rate | 95% (21 / 22) |
| Expected page in the top 4 chunks (answerable questions) | 95% |
| Expected page ranked 1st by vector search alone | 68% |
| Expected page ranked 1st by BM25 alone | 84% |
| Questions with a dropped citation | 0 |
| Average time per question | 17.4 s, of which generation 17.3 s and retrieval 0.1 s |

**The one remaining failure:** "How are payments to GAMOT facilities made?" fails in every run. The chunks that answer it are never retrieved for that wording, so this is a question-wording problem rather than a chunking or search setting.

**Main lessons:**
- **Chunk boundaries decide whether a fact can be found.** If the subject and the number land in different chunks, neither chunk matches the question.
- **Vector and keyword search fail on different questions.** One question wasn't in the vector top 10 but was ranked 1st by BM25. Combining the two fixed it.
- **Annexes and templates repeat the question's wording** and can outrank the policy text itself.
- **A stale index silently changes results.** Re-ingesting without a reset left old chunks and duplicates.
- **Never trust LLM-written filenames or quotes as-is.** Mistyped filenames, curly quotes and OCR errors ("6o" for "60") all broke correct citations until the check was made forgiving and anchored to the retrieved chunks.
- **A correct answer can still fail on its citation**, so the answer and the citation are checked separately.
- **Latency is almost all generation.** Bigger chunks mean more text for the LLM to read; retrieval takes about 0.1 s.

More detail is in [docs/LEARNING_LOG.md](docs/LEARNING_LOG.md).

---

## What's next

**Finish Phase 1 (local)**
- [ ] Ingest PC2024-0013 (OCR) and run its 30-question golden set
- [ ] Fix the remaining failure: query rewriting, or filtering annex chunks with `doc_part`
- [ ] Eval summary report: hit@k, MRR and refusal accuracy, not just pass / fail
- [ ] Real **grader**: a cross-encoder reranker that sends weak retrievals to Rewrite
- [ ] Real **rewrite**: the LLM reformulates the query, max 2 retries
- [ ] Real **grounding check**: verify the answer against the chunks, otherwise return the fallback message
- [ ] **Guard**: mask PhilHealth ID numbers, refuse off-topic questions
- [ ] Deterministic **premium calculator**: rate × salary, clamped to the floor and ceiling from the circular
- [ ] Reduce latency: shorter output, prompt ordering for caching, fewer or smaller chunks
- [ ] CLI that takes the question as an argument; Streamlit page showing citations and timings
- [ ] Token counts and estimated cost per query

**Phase 2 (cloud)**
- Move the LLM and embeddings to AWS Bedrock by changing config only, then compare quality, latency and cost with the local setup
- Tracing and observability per node
- An MCP server exposing the retriever as a tool
- Serverless deployment with CI running the eval on every push
