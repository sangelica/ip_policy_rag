# Architecture Decisions

Format: **Decision** — context, choice, why, trade-off.

## ADR-001: Local-first with a provider abstraction
- **Context:** minimise cost while learning; target AWS Bedrock later.
- **Choice:** Ollama locally; all model access goes through `src/providers.py`, driven by `config.yaml`.
- **Why:** free and offline; swapping to Bedrock changes config, not code.
- **Trade-off:** local 7B models are weaker, especially as judges.

## ADR-002: Walking skeleton before features
- **Context:** many moving parts (guard, retrieve, grade, rewrite, generate, check).
- **Choice:** wire every node end to end with stubs on 1 PDF, then upgrade one node at a time.
- **Why:** prove feasibility early; every upgrade is measured against a working baseline.
- **Trade-off:** early answers are lower quality by design.
