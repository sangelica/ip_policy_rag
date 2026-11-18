# Architecture Decisions

Format: **Decision** — context, choice, why, trade-off.

## ADR-001: Local-first with a provider abstraction
- **Context:** minimise cost while learning; target AWS Bedrock later.
- **Choice:** Ollama locally; all model access goes through `src/config.py` (`get_llm()`, `get_embeddings()`), driven by `config.yaml`.
- **Why:** free and offline; swapping to Bedrock changes config, not code.
- **Trade-off:** local 7B models are weaker, especially as judges.

## ADR-002: Walking skeleton before features
- **Context:** many moving parts (guard, retrieve, grade, rewrite, generate, check).
- **Choice:** wire every node end to end with stubs on 1 PDF, then upgrade one node at a time.
- **Why:** prove feasibility early; every upgrade is measured against a working baseline.
- **Trade-off:** early answers are lower quality by design.

## ADR-003: Local Tesseract OCR for scanned PDFs, cached per page
- **Context:** PC2024-0013 is a scanned circular: all 32 pages have no text layer, so `get_text()` returns nothing.
- **Choice:** use the text layer when a page has one; otherwise OCR it with Tesseract via PyMuPDF (`get_textpage_ocr`, 300 dpi). Cache each OCR'd page as `data/ocr/<file>/pNN.txt` and tag chunks with `ocr: true`.
- **Why:** free and local (consistent with ADR-001); the cache makes re-ingesting fast and lets OCR output be inspected and corrected by hand; the flag shows which answers rest on less reliable text.
- **Trade-off:** OCR errors (stamps read as text, "10th" read as "10%", flattened tables). Values from OCR'd pages must be checked against the page image before going into the golden set. A managed service (e.g. Amazon Textract) reads tables better but costs money: revisit in Phase 2.

## ADR-004: Hybrid retrieval (vector + BM25, fused with RRF)
- **Context:** with vector search alone (700 / 100, list-aware chunks, top 4), 18 / 22 golden questions passed. The misses were retrieval misses: the right chunk wasn't in the top 4. Policy questions hinge on exact terms (circular numbers, ₱ amounts, "seven (7) days", "PRO"), and embeddings match meaning, not exact words. Q16's answer chunk wasn't even in the vector top 10.
- **Choice:** run vector search and BM25 keyword search (`rank_bm25`) side by side, take the top 10 from each (`candidates_k: 10`), and merge them with Reciprocal Rank Fusion: `score += 1 / (rrf_k + rank)` with `rrf_k: 60`. The top 4 (`top_k`) go to the LLM. The BM25 index is built once from the chunks already stored in Chroma and cached. `retrieval.mode: vector | hybrid` in `config.yaml` switches between the two for comparison.
- **Why:**
  - The two searches fail on different questions. Expected page ranked 1st: vector 68%, BM25 84% (answerable questions, run R8). Q16 was outside the vector top 10 but ranked 1st by BM25.
  - RRF combines ranks, not raw scores, so the two incompatible scales (cosine distance vs BM25 score) need no tuning, and a chunk that both searches rank well beats one found by only one.
  - Result: Q8, Q16 and Q17 fixed; 18 → 20 / 22 after a clean re-ingest (runs R4 → R6). With hybrid search, the expected page is in the top 4 chunks for 95% of answerable questions. It's free and local (consistent with ADR-001).
- **Trade-off:**
  - The BM25 index lives in memory and is cached per process, so it must be rebuilt (restart) after re-ingesting.
  - Retrieval time went from ~0.05 s to ~0.1 s: negligible next to ~17 s of generation.
  - Annexes and templates repeat the question's wording, so BM25 can rank them above the policy text (Q8, Q18). `doc_part` metadata is stored to filter or down-weight annexes later.
  - The first hybrid run (R5) also included an accidental re-ingest without a reset, so its +1 can't be attributed to hybrid search alone; R6, on a clean index, is the fair comparison.
  - Q18 still fails: no retrieval setting finds its chunks for that wording. Query rewriting or a reranker (Stage 2 upgrades 4–5) is the next step.
