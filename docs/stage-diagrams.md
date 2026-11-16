# PhilHealth Policy RAG Agent: architecture by stage (Phase 1)

Each diagram shows the system at the end of that stage.

**Legend:** green = new in this stage · grey dashed = stub (placeholder) · plain = unchanged from earlier stages

## Progression overview

```mermaid
flowchart LR
  S1["1 Skeleton<br/>works end to end"] --> S2["2 Upgrades<br/>accurate, measured"] --> S3["3 Package<br/>visible to others"]
  classDef local fill:#EEEDFE,stroke:#534AB7,color:#26215C
  class S1,S2,S3 local
```

## Stage 1: Walking skeleton

New: every node exists and is wired. Only Guard, Retrieve and Generate do real work.

```mermaid
flowchart LR
  subgraph ingest["Ingest (one-time)"]
    PDF["1 PDF"] --> PARSE["PyMuPDF<br/>extract text"] --> CHUNK["Fixed 800-char chunks<br/>+ source, page"] --> EMB["nomic-embed-text<br/>(Ollama)"] --> DB[("Chroma<br/>on disk")]
  end
  subgraph qg["Query graph (LangGraph)"]
    Q(["Question"]) --> G["1 Guard<br/>PII regex"] --> R["2 Retrieve<br/>vector, k=4"] --> GR{"3 Grade<br/>stub: always yes"}
    GR -->|relevant| GEN["4 Generate<br/>qwen2.5:7b + Pydantic"] --> C{"5 Check<br/>stub: always grounded"}
    GR -.->|not relevant| RW["Rewrite<br/>stub"] -.-> R
    C -->|grounded| A(["Answer + citations<br/>+ per-node timings"])
    C -.->|unsupported| F(["Insufficient policy data"])
  end
  DB --- R
  A --> UI["CLI / Streamlit"]
  EV["Eval: 5 questions<br/>hit rate + latency"] -.-> Q
  classDef new fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  classDef stub fill:#F1EFE8,stroke:#888780,stroke-dasharray:5 5,color:#444441
  class PDF,PARSE,CHUNK,EMB,DB,Q,G,R,GEN,A,UI,EV new
  class GR,RW,C,F stub
```

## Stage 2: Upgrade loop

New: every stub replaced by a real component, plus hybrid search, a calculator tool and an eval harness.

```mermaid
flowchart LR
  subgraph ingest["Ingest"]
    PDF["10-15 PDFs"] --> PARSE["PyMuPDF"] --> CHUNK["Recursive chunks + overlap<br/>+ circular_no, year"] --> EMB["nomic-embed-text"] --> DB[("Chroma")]
    CHUNK --> BM[("BM25 index")]
  end
  subgraph qg["Query graph"]
    Q(["Question"]) --> G{"1 Guard<br/>PII + scope check"}
    G -->|off-topic| REF(["Refuse + disclaimer"])
    G -->|in scope| R["2 Retrieve<br/>hybrid BM25 + vector, RRF"]
    R --> GR{"3 Grade<br/>bge-reranker threshold"}
    GR -->|"2+ relevant"| GEN["4 Generate<br/>Pydantic citations"]
    GR -->|"fewer than 2"| RW["Rewrite<br/>LLM, max 2 retries"] --> R
    GEN <--> CALC["premium_calculator<br/>rate, floor, ceiling"]
    GEN --> C{"5 Check<br/>LLM judge"}
    C -->|grounded| A(["Answer + citations"])
    C -->|unsupported| F(["Insufficient policy data"])
  end
  DB --- R
  BM --- R
  A --> TEL["Telemetry<br/>latency, tokens, est. cost"]
  subgraph ev["Eval harness"]
    GS["Golden set 25-30 Qs<br/>incl. out-of-scope"] --> RE["run_eval.py"] --> M["hit@5, MRR, refusal<br/>accuracy, faithfulness"]
  end
  classDef new fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  class PDF,CHUNK,BM,G,REF,R,GR,RW,CALC,C,TEL,GS,RE,M new
```

## Stage 3: Package

New: the README and the public repo, which pull together the code, results and notes.

```mermaid
flowchart LR
  CODE["Code<br/>src/, app.py"] --> RM
  EVAL["Eval tables<br/>before/after per upgrade"] --> RM
  DOCS["docs/<br/>LEARNING_LOG, DECISIONS,<br/>FEASIBILITY"] --> RM
  GIF["Demo GIF"] --> RM
  RM["README.md<br/>problem, architecture, how to run,<br/>limitations, roadmap, why insurance"] --> GH[("GitHub repo")]
  GH --> RV(["Checkpoint: working version"])
  classDef new fill:#E1F5EE,stroke:#0F6E56,color:#04342C
  class GIF,RM,GH new
```

