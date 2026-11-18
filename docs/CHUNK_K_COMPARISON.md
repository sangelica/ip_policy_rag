# Chunk size, overlap and k comparison (Stage 1)

Source: `docs/LEARNING_LOG.md`, 11/16–11/17 runs. Document: `PA2025-0002.pdf` (1 page). Model: Ollama `qwen2.5:7b`. Eval: `eval/test_run_eval.py`, 5 questions.

## Configurations

| Config | Chunk size | Overlap | k | Chunks in corpus | Share of corpus sent to LLM |
|---|---|---|---|---|---|
| A | 800 | not logged | 4 | 2 (c0–c1) | All of it (k > corpus size) |
| B | 200 | not logged | 4 | 10 (c0–c9) | 4 of 10 |
| C | 200 | not logged | 2 | 10 (c0–c9) | 2 of 10 |
| D | 200 | 50 | 2 | 11 (c0–c10) | 2 of 11 |

## Results per question

✓ = test passed · ✗ = test failed · timing in seconds (end to end per question)

| Question | A: 800 / k=4 | B: 200 / k=4 | C: 200 / k=2 | D: 200 + 50 overlap / k=2 |
|---|---|---|---|---|
| Q1 2025 premium rate | ✓ 25.10 | ✓ 15.56 | ✗ (no timing logged) | ✓ 9.17 |
| Q2 Income ceiling | ✓ 6.86 | ✓ 12.49 | ✓ 8.36 | ✓ 8.38 |
| Q3 MBS includes overtime? | ✓ 12.71 | ✓ 17.40 | ✓ 9.65 | ✓ 7.54 |
| Q4 Konsulta (not in doc) | ✗ (no timing logged) | ✗ (no timing logged) | ✗ (no timing logged) | ✗ (no timing logged) |
| Q5 Premium at ₱10,000 | ✓ 8.82 | ✓ 11.08 | ✓ 8.44 | ✓ 10.65 |
| **Passed** | **4/5** | **4/5** | **3/5** | **4/5** |

## Timing summary

| Measure | A | B | C | D |
|---|---|---|---|---|
| Average, passed questions | 13.37 s (4 Qs) | 14.13 s (4 Qs) | 8.82 s (3 Qs) | 8.94 s (4 Qs) |
| Average, Q2 + Q3 + Q5 only (passed in all configs) | 9.46 s | 13.66 s | 8.82 s | 8.86 s |
| Slowest | 25.10 s (Q1) | 17.40 s (Q3) | 9.65 s (Q3) | 10.65 s (Q5) |

Q1 in config A (25.10 s) was the first question of the run and may include model load time. This was not measured separately.

## Differences in results

| Question | A: 800 / k=4 | B: 200 / k=4 | C: 200 / k=2 | D: 200 + 50 overlap / k=2 |
|---|---|---|---|---|
| Q1 | Correct (5.0%); long answer restating the advisory and RA 11223 | Correct (5.0%); short answer | **Wrong:** "excerpts do not provide the rate". Retrieved c9 (heading) and c2 (sentence cut before "5.0%"); c3, which holds 5.0%, was ranked 3rd and dropped | Correct (5.0%). Top chunk c3 now holds the whole sentence "premium rate … remains at 5.0%" |
| Q2 | Correct (P100,000.00) | Correct (P100,000) | Correct (P100,000); quote is the shortest and most exact | Correct (P100,000.00); top chunk c4 holds the whole table |
| Q3 | Correct, quotes the MBS definition | Correct, longest answer, repeats the quote inside the answer | Correct, short answer | Correct, short answer; citation `source` is `''` |
| Q4 | Correct refusal text, but citations not empty | Correct refusal text, but citations `[{'source': '', 'page': 1, 'quote': ''}]` | Same as B | Same as B |
| Q5 | Correct (₱500.00) | Correct (₱500.00); quote is only "P500.00" | Correct (₱500.00); quote is only "P500.00" | Correct (₱500.00); quote "Monthly Premium P500.00" is not word for word in the chunk (the table text has other values between those words) |

## Other observations from the runs

| Observation | Configs |
|---|---|
| Citation `source` format is inconsistent: `[PA2025-0002.pdf p.1]`, `PA2025-0002.pdf`, `[PA2025-0002.pdf]`, or `''` | B, C, D |
| 200-char chunks without overlap cut sentences mid-way (e.g. "premium rate effective the applicable" / "period of January 2025 remains at 5.0%") | B, C |
| With 50-char overlap, the rate sentence (c3) and the whole table (c4) each fit in one chunk | D |
| Quotes can be reworded rather than copied word for word (Q5) | D |
| `timings` in the final state is `{}`, so per-node timings were not recorded | B, C, D |
| With 2 chunks and k=4, retrieval never has to choose, so config A does not test retrieval quality | A |

## Takeaways

1. **Overlap fixed the k=2 failure.** With the same chunk size and k, adding 50 characters of overlap (C → D) made Q1 pass again: 3/5 → 4/5. The answer sentence no longer gets split across two chunks.
2. **D matches the best pass rate at the lowest cost.** 4/5, like A and B, with an average of 8.86 s on the shared questions, about the same as C (8.82 s) and faster than B (13.66 s).
3. **Without overlap, smaller chunks need a larger k.** In C, the Q1 answer was split across c2 and c3; k=4 (B) caught both, k=2 kept only c2.
4. **Q4 fails in every config**, so it isn't a chunking or k problem. It's the empty-citation handling (fallback node or prompt).
5. **Quote verification is needed.** The Q5 quote in D isn't word for word in the chunk. The deterministic check in Upgrade 6 (quote must appear in the chunk) would catch this.
6. **Config A's result is not representative.** The whole document fits in k=4, so a longer document is needed before these settings can be compared fairly.
