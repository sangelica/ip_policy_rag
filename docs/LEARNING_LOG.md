# Learning Log

One entry per stage or upgrade. Write the concept in your own words — this becomes your README and project write-up.

---

## Stage 1 — Walking skeleton
**Concept:**
Insurance Policy RAG Agent

**What I built:**
11/16:
config.yaml - configuration data including llm model, chunk size/overlap for ingestion, output folders etc.
config.py - functions to get config data
ingest.py
    - reads pdf
    - chunks results (chunk size: 800) per page
    - adds per chunk info to a "Document", added to a chroma DB vector store.
11/17:
state_machine.py - The langchain state machine flow. Create the graph nodes, connect the flow through "edges" which can be conditionals.
nodes.py - These are the states of the state machine, represented by functions.
run_eval.py - Run validation on 5 questions from golden_set.jsonl using pytest framework. Use "python -m pytest -s" from project folder.
schemas.py - Data models used by the code.
    - QueryState - shared data across nodes
    - Citation/Answer - schema fed to the LLM, that it follows. This is done through "llm = get_llm().with_structured_output(Answer)"
vector_store.py - chroma vector db is made into a single instance

**Result:** 
11/16 (Used PA2025-0002.pdf):
Score: 0.5947 | Content: Philippine Health Insurance Corporation ADVISORY No. 2025 - 0002 (Sgd.) EMMANUEL R. LEDESMA, JR. President and Chief Executive As prescribed in Section 10 of Republic Act No. 11223 also known as the Universal Health Care (UHC) Act and in accordance with PhilHealth Circular No. 2020-0005 on the “Pre mium Contribution Schedule in the National Health Insurance Program (Revision 1),” all Direct Contributors are hereby advised that the premium rate effective the applicable period of January 2025 remains at 5.0%, with an income floor of P10,000 and income ceiling of P100,000, as follows: Year Monthly Basic Salary Premium Rate Monthly Premium 2025 P10,000.00 5.0% P500.00 P10,000.01 to P99,999.99 P500.00 to P5,000.00 P100,000.00 5,000.00
Metadata: {'page': 1, 'chunk': 0, 'source': 'PA2025-0002.pdf'}

Score: 0.6714 | Content: As stated in the same issuance, all Employers are reminded to use the Monthly Basic Salary (MBS) in the computation of monthly contributions of their Employees. The MBS is the fixed basic rate of an employee which shall not include sales commission, overtime pay, allowances, thirteenth month pay, bonuses or other gratuity payments. Further to this, those deductions to the employee’s pay occasioned by his/her undertime, tardiness, leave(s) without pay, absences, or other similar circumstances shall also be excluded in this computation. Further inquiriesmay be referred to the nearest PhilHealth Local Health Insurance Office or to the assigned PhilHealth Account Information Management Specialist (PAIMS). Premium Contribution for All Direct Contributors for CY 2025
Metadata: {'chunk': 1, 'source': 'PA2025-0002.pdf', 'page': 1}

11/17
eval/test_run_eval.py Q:What is the 2025 premium rate for direct contributors?
Ans:The 2025 premium rate for direct contributors is 5.0%. This is stated in Philippine Health Insurance Corporation ADVISORY No. 2025 - 0002, as prescribed in Section 10 of Republic Act No. 11223 also known as the Universal Health Care (UHC) Act and in accordance with PhilHealth Circular No. 2020-0005 on the “Premium Contribution Schedule in the National Health Insurance Program (Revision 1).
Timing:25.10s

.Q:What is the income ceiling?
Ans:The income ceiling is P100,000.00.
Timing:6.86s

.Q:Does the monthly basic salary include overtime pay?
Ans:No, the monthly basic salary does not include overtime pay. According to the excerpt from PA2025-0002.pdf p.1, the Monthly Basic Salary (MBS) is defined as the fixed basic rate of an employee which shall not include sales commission, overtime pay, allowances, thirteenth month pay, bonuses or other gratuity payments.
Timing:12.71s

.FQ:What is the monthly premium for someone earning ₱10,000?
Ans:The monthly premium for someone earning ₱10,000 is ₱500.00.
Timing:8.82s

**Surprises / bugs:**
11/16 update:
Used PA2025-0002.pdf, which is a one page document.

1. The whole corpus is only 2 chunks. PA2025-0002 is a one-page advisory, and at chunk_size: 800 it splits into two pieces. Any search with k=4 returns everything, so this run proves the pipeline works, not that retrieval is any good. That's fine for the skeleton, but add a second, longer PDF soon so retrieval actually has to choose.

2. The table lost its structure. P10,000.01 to P99,999.99 P500.00 to P5,000.00 is a table flattened into one line. A human can barely read it, and a 7B model may misread it. This is the main reason the plan has a deterministic premium_calculator: this advisory gives you its inputs (rate 5.0%, floor ₱10,000, ceiling ₱100,000).

3. Text order follows the PDF's layout, not its reading order. The signature block ("(Sgd.) EMMANUEL R. LEDESMA…") appears before the body text, and "Pre mium" is split in two.

4. The heading problem. Chunk 1 ends with "Premium Contribution for All Direct Contributors for CY 2025", the heading for the table that's actually in chunk 0.

5. Scores are distances, so lower means closer. Chunk 0, the one with the rates, scored 0.59, which is better than chunk 1 at 0.64. Note what query you used.
    - I used "results = VECTOR_STORE.similarity_search_with_score("What is the 2025 premium rate for direct contributors?", k=4)
    - chunk 0 does seem to be closer based on manual check.

11/17 update:
1.
When running the golden test data, the answers to the questions are correct. However, when the document cannot answer the query, a page and source (without a quote) is still provided in citations
even when prompt explicitly says citations need to be empty.

The result changes depending on the quality of the prompt, the llm model used may be weak as well.
This is the question and answer that failed validation.

Question: What does PhilHealth Konsulta cover?
'answer': {'answer': 'The provided excerpts do not contain any information about what PhilHealth Konsulta covers.', 'citations': [{'source': '', 'page': 1, 'quote': ''}]}, 'grounded': True, 'timings': {}}

Try to handle this in fallback handling.
2. For the ollama model, providing instructions to LLM through "Field" description in schema does not work. What works is modifying the prompt itself in nodes.py
**Key takeaway:** 
