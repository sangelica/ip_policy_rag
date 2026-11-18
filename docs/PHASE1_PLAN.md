# PhilHealth Policy & Benefits RAG Agent: Phase 1 Plan (Stages 1–3)

An insurance-domain assistant that answers PhilHealth benefit and policy questions **with verified citations**, and refuses when the documents don't support an answer.

**Approach:** build a thin end-to-end working version first, then upgrade one piece at a time, measuring every change.

| Phase | Stages | Runs on |
|---|---|---|
| **Phase 1 (this plan)** | 1 Walking skeleton → 2 Upgrade loop → 3 Package | Local only (Ollama) |
| Phase 2 (separate plan) | MCP server, observability, Bedrock, deployment + CI/CD | AWS |

**Constraints:** free and local (M1, 16 GB RAM, Python 3.12); all model access goes through `src/config.py`, so moving to Bedrock changes config, not code.

**Rule:** start a stage only when the previous one runs end to end, its eval has been re-run, and it is committed.

---

## 1. Architecture

```
Question → [1 Guard] → [2 Retrieve] → [3 Grade] ──not relevant──→ [Rewrite] → back to 2 (max 2 retries)
                                          │ relevant
                                          ▼
                                    [4 Generate] → [5 Check] ──unsupported──→ "Insufficient policy data to confirm."
                                     (Pydantic)        │ grounded
                                                       ▼
                                     Answer + citations + per-node timings
```

Stage-by-stage diagrams: [stage-diagrams.md](stage-diagrams.md).

| Layer | Phase 1 (local, free) | Phase 2 option |
|---|---|---|
| LLM | Ollama `qwen2.5:7b` | Claude Haiku on Bedrock |
| Embeddings | Ollama `nomic-embed-text` (768-dim) | Titan Embed v2 |
| Vector store | Chroma on disk (`data/chroma/<provider>/`) | OpenSearch Serverless / Bedrock KB |
| Orchestration | LangGraph | same |
| UI | Streamlit | App Runner / EC2 |
| OCR (scanned PDFs) | Tesseract via PyMuPDF | Amazon Textract |

## 2. Project layout

```
insurance_policy_assistant/
  config.yaml            # providers, ingest, retrieval, graph settings
  app.py                 # Streamlit UI (run from the project root)
  src/
    config.py            # load_config, get_llm, get_embeddings (only file that knows the provider)
    vector_store.py      # get_vector_store(): one cached Chroma instance
    ingest.py            # PDF → text (or OCR) → chunks → Chroma
    retriever.py         # search() helper for manual checks
    schemas.py           # Citation, Answer (Pydantic), QueryState (LangGraph)
    nodes.py             # guard, retrieve, grade, rewrite, generate, check, fallback
    state_machine.py     # graph wiring → compiled `app`
  scripts/               # smoke tests: check_providers, check_schemas
  eval/                  # golden_set*.jsonl, test_run_eval.py (pytest)
  data/raw/  data/ocr/  data/chroma/     # all git-ignored
  docs/                  # this plan, DECISIONS, LEARNING_LOG, FEASIBILITY, stage-diagrams
```

Each node is a plain function `node(state) -> dict` that returns only the keys it changes. Upgrading a node means editing one function; the graph stays the same.

---

## 3. Stage 1: Walking skeleton

**Goal:** one question flows through every node and returns a cited answer.
**Learn:** RAG, embeddings, LangGraph state/nodes/edges, structured output.

### What each piece does in the skeleton

| Piece | Skeleton version | Status |
|---|---|---|
| Ingest | PyMuPDF text (OCR fallback for scanned pages) → recursive chunks → Chroma, metadata `{source, page, chunk, ocr}`, deterministic IDs | Built |
| 1 Guard | One regex masking 12-digit PhilHealth IDs | Stub (returns no change) |
| 2 Retrieve | Vector search, `k = top_k` from config | Built |
| 3 Grade | Always "relevant" (edge to Rewrite exists, never taken) | Stub |
| Rewrite | Query unchanged; must increment `retries` | Stub |
| 4 Generate | One LLM call with `with_structured_output(Answer)` | Built |
| 5 Check | Always "grounded" (fallback edge exists) | Stub |
| Fallback | Sets the config fallback message, empty citations | Stub |
| Telemetry | `time.perf_counter()` per node → `state["timings"]` (merged by reducer) | Built |
| Interface | CLI (`python -m src.state_machine "question"`), then Streamlit | Streamlit started; CLI to do |
| Eval | Golden set + pytest checks | Built |

### Build order

Run from the project root with the venv active. Don't start a step until the previous one passes.

| # | Step | Test | Pass when |
|---|---|---|---|
| 1 | Ollama + models | `ollama list`, `ollama run qwen2.5:7b "Reply with OK"` | Both models listed; the model replies |
| 2 | `config.py` + `config.yaml` | `python -m scripts.check_providers` | A reply, then `768` |
| 3 | `ingest.py` | `python -m src.ingest` twice | Same chunk count both times (IDs upsert, no duplicates) |
| 4 | `retriever.py` | `search("premium contribution rate")` | Results carry `source` and `page`; the text is really on that page of the PDF |
| 5 | `schemas.py` | `python -m scripts.check_schemas` | Valid answer accepted; `page="abc"` and `page=0` rejected |
| 6 | `nodes.py` | unit tests (`tests/test_nodes.py`, to do) | Guard masks an ID; rewrite increments `retries`; fallback returns the config message |
| 7 | `state_machine.py` | `python -m src.state_machine "What is the premium rate?"` | Answer, ≥1 citation, a timing for every node |
| 8 | `app.py` | `streamlit run app.py` | Same answer, citations and timings as the CLI |
| 9 | Eval | `python -m pytest eval/test_run_eval.py -s` | Every question runs; failures show what the model said |
| 10 | Wrap up | `git tag v0.1-skeleton`; write `docs/FEASIBILITY.md` | Clean tree, tag listed, read-out filled in |

### Done when
Golden questions run end to end, Streamlit shows cited answers with per-node timings, and the work is committed and tagged `v0.1-skeleton`.

### Feasibility read-out (`docs/FEASIBILITY.md`)

| Measure | If it's bad… |
|---|---|
| Latency per node (Generate dominates: ~11 s of ~11.4 s so far) | Smaller model locally, or Bedrock in Phase 2 |
| Structured-output success rate (valid `Answer`, citations filled) | Clearer prompt, simpler schema, retry, or another model |
| Retrieval hit rate (expected page in chunks) | Chunking / k first (see [CHUNK_K_COMPARISON.md](CHUNK_K_COMPARISON.md)), then hybrid search |

---

## 4. Stage 2: Upgrade loop

For each upgrade: **learn the concept → replace one stub → re-run the eval → commit** with the numbers in the message (e.g. `upgrade-3: hybrid retrieval, hit@5 0.6→0.8`).

| # | Upgrade | What you learn | Measure |
|---|---|---|---|
| 1 | Grow the corpus (1 → 3 → 10–15 PDFs), tune chunking/overlap, add `circular_no`, `year` metadata | chunking trade-offs; how retrieval degrades with size | hit@k per corpus size |
| 2 | Golden set to 25–30+ items incl. 3–5 unanswerable; add `run_eval.py` summary table | RAG evaluation | baseline: hit@5, MRR, refusal accuracy |
| 3 | Hybrid retrieval: BM25 + vector, merged with RRF | why exact terms (circular numbers, ₱ amounts) need keyword search | hit@5 Δ |
| 4 | Real grader: cross-encoder reranker (`bge-reranker-base`) threshold | reranking | hit@5, latency |
| 5 | Real rewrite: LLM reformulates the query, max 2 retries | corrective RAG loop | recall on hard questions |
| 6 | Real check: LLM judge verifies the answer against the chunks, else fallback (optional free hosted judge) | hallucination detection, judge limits | faithfulness %, false refusals |
| 7 | Guard: scope check, more PII patterns, disclaimer | guardrails | refusal accuracy |
| 8 | `premium_calculator`: `clamp(salary, floor, ceiling) × rate` from the circular | deterministic tools vs LLM maths | correct on calculator questions |
| 9 | Tokens + estimated cost per query | LLMOps | shown in UI |

**Priority if time is short:** 1 → 2 → 3 → 6 → 4; the rest are stretch goals. Tag `v0.2` when done.

### Upgrade notes

- **Metrics (Upgrade 2).** *hit@5*: was the right page in the top 5? *MRR*: average of 1/rank (rank 1 → 1.0, rank 3 → 0.33, missed → 0). *Refusal accuracy*: share of unanswerable questions correctly refused.
- **RRF (Upgrade 3).** `score = Σ 1/(60 + rank)` across both rankings. A chunk ranked 1st by BM25 and 3rd by vector (1/61 + 1/63 = 0.032) beats one ranked 1st by vector only (0.016): agreement wins.
- **Reranker (Upgrade 4).** Embeddings encode question and chunk separately (fast, coarse); a cross-encoder reads them together (slower, precise). Fewer than 2 chunks above the threshold → Rewrite. No LLM call.
- **Rewrite loop (Upgrade 5).** "Konsulta stuff?" → 1 relevant chunk → rewrite to "Konsulta package covered services" → 3 relevant → Generate. The retry cap stops infinite loops.
- **Calculator (Upgrade 8).** Illustrative: salary ₱45,000, floor ₱10,000, ceiling ₱100,000, rate 5% → ₱2,250. Real values come from PA2025-0002.

---

## 5. Stage 3: Package

**Goal (checkpoint: working version):** anyone can understand, run and verify the project from the README alone.

README contents:
- The problem and why it matters for insurance (verified citations, compliance-minded refusals, measured quality, cloud-portable design)
- Architecture as a Mermaid diagram (copy from [stage-diagrams.md](stage-diagrams.md))
- How to run it (setup, ingest, CLI, Streamlit, eval)
- **Before/after eval table per upgrade**: the strongest evidence
- A ~20 s demo GIF: one cited answer, one refusal
- Limitations and roadmap (Phase 2)

Then push to GitHub.

---

## 6. Working practices

**Document as you go**
- `docs/LEARNING_LOG.md`: per stage or upgrade: Concept · What I built · Result (before/after) · Surprises/bugs · Key takeaway
- `docs/DECISIONS.md`: short ADRs (context, choice, why, trade-off)
- `docs/FEASIBILITY.md`: the Stage 1 read-out

**Evaluation data** (`eval/golden_set*.jsonl`, one JSON object per line)
```json
{"q": "...", "expected_source": "PC2026-0002.pdf", "expected_pages": [4, 24], "answerable": true,
 "answer": "15", "must_contain_any": ["15", "fifteen"]}
```
- `expected_pages` lists every page stating the fact; the citation check passes if any cited page is in it.
- Page numbers are PDF page indices (1-based), not the printed "Page X of Y".
- Unanswerable questions: `answerable: false`, `expected_pages: []`; pass when citations are empty.
- Values from OCR'd pages must be checked against the page image before going into the golden set.

**Security**
- Never commit credentials: `.env`, credential CSVs and keys stay in `.gitignore`; no keys in code.
- PDFs, OCR cache and Chroma data are git-ignored.

---

## 7. Verification (end of Phase 1)

- `python -m src.state_machine "What is the 2025 premium rate?"` prints a Pydantic answer with citations and per-node timings; `streamlit run app.py` shows the same.
- Each upgrade's eval run is committed with it, and the metrics move in the expected direction.
- An off-topic question is refused at the guard; an unanswerable policy question returns the fallback; "premium on ₱45,000/month" uses the calculator and cites the circular.

---

## Appendix A: Pitfalls met so far

| Symptom | Cause | Fix |
|---|---|---|
| `get_text()` returns nothing | Scanned PDF, no text layer | OCR fallback, cached in `data/ocr/` (ADR-003) |
| Quote check fails on a correct quote | Soft hyphens (`\xad`), OCR errors like `6o` for `60` | Strip `\xad` in `normalize`; accept that source text can be wrong |
| `ImportError` from `src.nodes` | An empty `src/nodes/` folder shadowed `nodes.py` | Folder removed: a package beats a module of the same name |
| `No module named 'src'` | Script run from inside `src/` | Run from the root: `python -m …`, `streamlit run app.py` |
| Prints missing under pytest | pytest captures stdout | `pytest -s` |
| `res["answer"]` is a string | Node returned `ans.model_dump()` unwrapped | Return `{"answer": ans.model_dump()}` |
| Empty citations on answerable questions | Prompt/schema made `[]` the easy default | Rules first, "at least one citation" stated explicitly |
| Duplicate chunks | Re-ingest without fixed IDs | Deterministic IDs `<file>-p<page>-c<i>`; reset when chunk settings change |
| Connection refused | Ollama not running | Start the Ollama app / `ollama serve` |

## Appendix B: Glossary

- **RAG:** retrieve relevant excerpts, put them in the prompt, and make the LLM answer only from them, with citations. An open-book exam.
- **Hallucination:** a confident answer the sources don't support.
- **Embedding:** text → vector (768 floats here); similar meanings sit close together.
- **Chunk:** a piece of a page plus metadata (`source`, `page`); the metadata makes citations possible.
- **Ingest:** the one-off pipeline PDF → text → chunks → embeddings → vector store.
- **Chroma:** the vector database; returns the k chunks closest to a question.
- **PII:** data identifying a person (e.g. a 12-digit PhilHealth ID); must not reach the LLM.
- **Pydantic / structured output:** typed, validated data shapes; `with_structured_output(Answer)` makes the LLM return JSON matching `Answer`.
- **LangGraph:** the pipeline as a state machine. State = shared dict; node = `function(state) -> updates`; edge = transition; conditional edge = branch; reducer = how updates combine (e.g. `merge` for `timings`).
- **Stub:** a placeholder returning a fixed value, replaced later without changing the graph.
- **Golden set:** questions with known answers and pages; the regression suite for answer quality.
- **OCR:** reading text from page images; needed for scanned PDFs, and error-prone.
- **Token:** the unit an LLM reads and writes (~¾ of a word); cloud providers bill per token.
