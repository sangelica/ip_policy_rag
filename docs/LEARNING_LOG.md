# Learning Log

A dated journal of building the PhilHealth Policy RAG Agent. Each day: **What I built · Surprises / bugs · Takeaways**.
Full eval output lives in the result files, not here; this log keeps the conclusions.

---

## 2026-11-16 — Stage 1: ingest skeleton

### What I built
- `config.yaml`: configuration data (LLM model, chunk size/overlap for ingestion, output folders, etc.)
- `config.py`: functions to load the config and get the LLM and embeddings
- `ingest.py`:
  - reads the PDF
  - chunks the text per page (chunk size: 800)
  - wraps each chunk in a `Document` with metadata and adds it to the Chroma vector store
- First retrieval check on PA2025-0002.pdf (one page), with the query *"What is the 2025 premium rate for direct contributors?"*, k=4:
  - chunk 0 (rates table) scored 0.5947
  - chunk 1 (Monthly Basic Salary definition) scored 0.6714

### Surprises / bugs
1. **The whole corpus is only 2 chunks.** PA2025-0002 is a one-page advisory, and at chunk_size 800 it splits into two pieces. Any search with k=4 returns everything, so this run proves the pipeline works, not that retrieval is any good. Add a second, longer PDF soon so retrieval has to choose.
2. **The table lost its structure.** "P10,000.01 to P99,999.99 P500.00 to P5,000.00" is a table flattened into one line. A human can barely read it, and a 7B model may misread it. This is why the plan has a deterministic `premium_calculator`; this advisory gives its inputs (rate 5.0%, floor ₱10,000, ceiling ₱100,000).
3. **Text order follows the PDF layout, not reading order.** The signature block ("(Sgd.) EMMANUEL R. LEDESMA…") comes before the body, and "Pre mium" is split in two.
4. **Headings land in the wrong chunk.** Chunk 1 ends with "Premium Contribution for All Direct Contributors for CY 2025", the heading for the table that is actually in chunk 0.
5. **Scores are distances: lower means closer.** Chunk 0 (the rates) scored 0.59, better than chunk 1 at 0.67, which matches a manual check.

### Takeaways
- A working pipeline is not the same as good retrieval; a tiny corpus can't test retrieval.
- PDF text extraction loses tables, reading order and words split across lines. Expect to clean text before chunking.
- Always note which query a similarity score came from, and remember Chroma scores are distances.

---

## 2026-11-17 — Stage 1: state machine, schemas, first eval

### What I built
- `state_machine.py`: the LangGraph state machine. Creates the graph nodes and connects them with edges, some of them conditional.
- `nodes.py`: the states of the state machine, each one a function.
- `schemas.py`: data models used by the code.
  - `QueryState`: shared data across nodes
  - `Citation` / `Answer`: the schema the LLM must follow, via `get_llm().with_structured_output(Answer)`
- `vector_store.py`: the Chroma vector store as a single cached instance.
- `eval/test_run_eval.py`: runs the 5 questions in `golden_set.jsonl` with pytest. Run `python -m pytest -s` from the project folder.
- Per-node timings (`time.perf_counter()` into `state["timings"]`).

### Results
| Run | Passed | Notes |
|---|---|---|
| chunk 800, k=4 | 4/5 | Konsulta (unanswerable) failed. Generate 7–25 s per question |
| chunk 200, k=4 | 4/5 | Konsulta failed again |
| chunk 200, k=2 | 3/5 | Premium-rate question failed: the 5.0% fell outside both chunks, so the model said "not provided". Konsulta failed |
| chunk 200, overlap 50, k=2 | 4/5 | The overlap brought "5.0%" back into the retrieved chunk. Konsulta failed |
| same, with timings | 4/5 | retrieve ~0.05–0.24 s, generate 4–11 s. Generate is almost all of the latency |

### Surprises / bugs
1. **Unanswerable questions still get a citation.** When the document can't answer, the model still returns a citation with a page and an empty source and quote, even though the prompt says citations must be empty:
   `'answer': 'The provided excerpts do not contain any information about what PhilHealth Konsulta covers.', 'citations': [{'source': '', 'page': 1, 'quote': ''}]`
   The result changes with the quality of the prompt, and the 7B model may be weak at following it. Possible fix: handle it in the fallback.
2. **Instructions in a Pydantic `Field` description are ignored by the Ollama model.** What works is changing the prompt itself in `nodes.py`.
3. **Small chunks can cut a fact in half.** At chunk 200 / k=2 the sentence with the rate was split from "5.0%", and the answer was lost.
4. **The model formats `source` inconsistently:** `[PA2025-0002.pdf p.1]`, `[PA2025-0002.pdf]`, `PA2025-0002.pdf` or empty.

### Takeaways
- Chunk size, overlap and k interact: overlap rescues facts that straddle a chunk boundary.
- With a local 7B model, the prompt is the control lever; don't rely on schema descriptions alone.
- Never trust the model's citation formatting. Normalise or validate it in code.
- Latency is all in the LLM call; retrieval takes milliseconds.

---

## 2026-11-18 — Stage 2: more data, OCR, hybrid retrieval

### What I built
<!-- Check these against what you actually did today -->
- Added PC2026-0002.pdf (28 pages: main body pp. 1–8, Annexes A–D pp. 9–28) and wrote 17 golden questions for it (22 in `golden_set.jsonl` in total).
- `expected_pages` is now a list; the eval passes if any cited page is in it.
- Eval checks: `normalize()` / `contains()` for answer matching (currency, thousands separators, trailing zeros, number boundaries), `must_contain_any`, and empty citations for unanswerable questions.
- OCR fallback in `ingest.py` for scanned pages (Tesseract via PyMuPDF), cached per page in `data/ocr/` (ADR-003). PC2024-0013 (32 scanned pages) has its own golden set of 30 questions, but is not ingested yet.
- List-aware chunking (split at paragraphs, numbered items, lettered items, then sentences), with chunk 600 → 700 and overlap 100.
- Abbreviation expansion at ingest from `config.yaml` (PRO → "PhilHealth Regional Office (PRO)").
- `doc_part` metadata ("main" or "annex") from the page footer or annex heading.
- Hybrid retrieval in `retriever.py`: BM25 + vector search, merged with reciprocal rank fusion (`candidates_k: 10`, `top_k: 4`, `rrf_k: 60`).
- Citation cleanup in `generate`: keep only citations whose source and page were actually retrieved.
- Cross-encoder reranker (`src/reranker.py`, `BAAI/bge-reranker-base`), run in the grade node: reranks the fused top 10 (`fused_k`) and keeps the top 4. `rerank.enabled` switch in `config.yaml`, plus a session fixture in the eval that loads the model once before the tests. Turned off in config (`rerank.enabled: false`) after testing (see Surprises 6 and 8).

### Surprises / bugs
1. **Some PDFs have no text layer.** PC2024-0013 is fully scanned, so `get_text()` returns nothing and OCR is needed.
2. **A text layer can still be wrong.** PC2026-0002 has OCR errors baked in ("sixty (6o)" on p. 3) and soft hyphens (`\xad`).
3. **The main body mentions annexes too**, so `doc_part` has to look only at the footer and the start of the page.
4. **Citations come back empty on correct answers.** Either the model returns none, or the cleanup filter drops them because the source string doesn't match exactly.
5. Generate still takes 7–23 s per question.
6. **The reranker did not fix Q18.** The answer chunk p24-c0 reached the reranker (fused position 5–6) but scored below the 4 chunks kept; p2-c0 (likely the ACPS definition) was cut by `fused_k: 10` before reranking. The top-scored chunk, the annex payment form (p14-c0, 0.72), does name the payment mode, but garbled: the checkboxes in "□ACPS □Check □Other" come out of the text layer as "DACPS DCheck DOther".
7. **Rewording Q18 makes it pass.** "How are GAMOT facilities paid by PhilHealth?" passes; the original "How are payments to GAMOT facilities made?" fails. With the new wording, both searches find the answer chunk p24-c0 (vector rank 5, BM25 rank 5), so RRF gives it two terms (1/65 + 1/65 = 0.0308) and it is fused 2nd. With the original wording, only BM25 finds it (rank 3, 1/63 = 0.0159), and four chunks that both searches found mid-list (≈ 0.03 each) take the top 4. Added the new wording to the golden set as Q22, keeping the original Q18 as the harder test.
8. **The reranker broke Q1**, which passed before. Turning the reranker off in config made it pass again.

### Takeaways
- **BM25** is lexical (keyword) search. It scores a query against a document by:
  - rarity: common words score lower than rare ones
  - frequency: repeating a keyword dozens of times has diminishing returns
  - length normalisation: a longer document doesn't automatically score higher
- **RRF** merges two rankings by rank, not score: `score += 1 / (rrf_k + rank)`. A chunk found by both searches beats one found by only one.
- Exact terms (circular numbers, ₱ amounts, abbreviations) are where keyword search helps vector search.
- **Vector and keyword search fail on different questions, so combining them helps.** (Q16: not in the vector top 10, BM25 rank 1)
- **Chunk boundaries decide whether a fact is retrievable.** If the subject and the number land in different chunks, neither chunk matches the question. (Q8: "PRO" in one chunk, "seven (7) days" in the next)
- **List-aware splitting helps, but can cut elsewhere**, e.g. at a sentence that starts with "It". (Q16 and Q17 broke at 700, until hybrid search recovered them)
- **Annexes and templates repeat the question's wording and crowd out the policy text.** (Q8, Q18: annex chunks outranked the main circular)
- **A stale index silently changes results.** Re-ingesting without a reset leaves old chunks and duplicates. (Hybrid run: p4-c3 and p4-c6 identical; +1 pass after a clean re-ingest)
- **Never trust LLM-written IDs or quotes as-is. Verify them against the source.** (Mistyped filename in Q10, curly quotes in Q16, OCR "6o" vs "60" in Q5)
- **A correct answer can still fail on its citation**, so check the citation separately from the answer. (Q2, Q5, Q10, Q16)
- **Change one thing per run, or results can't be attributed.** (The 600 run changed 4 things; the hybrid run also had a re-ingest)
- **Fix the test when it's wrong**, e.g. a paragraph that spans two pages. (Q13 on pages 6–7, and `expected_pages` lists)
- **Some failures are question-wording problems, not retrieval settings.** (Q18: no chunking or search setting found the ACPS chunks)
- **Bigger chunks cost latency, because the LLM reads more text.** (13 s → 22 s per question at 600)
- Write golden answers only after checking the page image, especially for OCR'd pages.
- **A reranker (cross-encoder) reads the query and the chunk together** and scores how well that chunk answers the question. Vector search embeds them separately and only compares topics. It is more precise, but it can only reorder chunks that retrieval already found.
- **RRF rewards agreement, not being best in one list.** A chunk ranked 5th by both searches (0.0308) beats one ranked 1st by only one (0.0164). So the wording of a question decides whether both searches find the answer, and that decides its fused rank. (Q18 vs Q22)
- **Keep the hard wording in the golden set and add the easy one beside it.** Replacing the failing question would make the eval pass without the system improving.
- **A more advanced component isn't automatically better.** The reranker didn't fix the failure it was added for and broke a question that passed. Measure it on the whole golden set before keeping it.

