"""Data shapes shared across the system.

- Citation / Answer (Pydantic): the contract the LLM must fill via
  with_structured_output(Answer). Validated at runtime, because LLM output is untrusted.
  Field descriptions are sent to the LLM as part of the JSON schema, so they are
  written as instructions to the model.
- State (TypedDict): the LangGraph shared dictionary. Type hints only, no runtime checks.
"""
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


# Structure for Citation
class Citation(BaseModel):
    source: str = Field(description="Filename of the excerpt, exactly as shown in its label.")
    page: int = Field(description="Page number shown in the excerpt label.")
    quote: str = Field(description="Sentence copied word for word from the excerpt that supports the answer.")


# Structure for Answer
class Answer(BaseModel):
    answer: str = Field(
        description=(
            "Answer to the question using only the excerpts. "
            "If the excerpts don't contain the answer, say so."
        )
    )
    citations: list[Citation] = Field(
        description="Excerpts supporting the answer. Empty if the excerpts don't answer the question."
    )


def merge(a: dict, b: dict) -> dict:
    """LangGraph reducer: combine a node's update with the existing value instead of overwriting."""
    return {**a, **b}


class QueryState(TypedDict, total=False):
    query: str                         # the question (masked by Guard, changed by Rewrite)
    chunks: list[Document]             # retrieved Documents
    relevant: bool                     # Grade's verdict
    retries: int                       # Rewrite counter
    answer: dict                       # Answer.model_dump()
    grounded: bool                     # Check's verdict
    timings: Annotated[dict, merge]    # {node_name: seconds}, merged rather than overwritten
