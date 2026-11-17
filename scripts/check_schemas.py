"""Smoke test for src/schemas.py.

Run from the project root:
    python -m scripts.check_schemas
"""
from pydantic import ValidationError

from src.schemas import Answer, State, merge

VALID = {
    "answer": "The 2025 premium rate is 5.0%.",
    "citations": [{"source": "PA2025-0002.pdf", "page": 1, "quote": "remains at 5.0%"}],
}


def expect_invalid(label: str, page) -> None:
    bad = {**VALID, "citations": [{**VALID["citations"][0], "page": page}]}
    try:
        Answer.model_validate(bad)
        print(f"FAIL {label}: page={page!r} was accepted")
    except ValidationError:
        print(f"OK   {label}: page={page!r} rejected")


def main() -> None:
    print("OK   valid answer:", Answer.model_validate(VALID))
    print("OK   empty citations allowed:", Answer.model_validate({"answer": "Not found.", "citations": []}))
    expect_invalid("non-numeric page", "abc")
    expect_invalid("page zero", 0)

    merged = merge({"guard": 0.0001}, {"retrieve": 0.21})
    print("OK   merge keeps both timings:" if merged == {"guard": 0.0001, "retrieve": 0.21} else "FAIL merge:", merged)

    print("OK   State keys:", list(State.__annotations__))


if __name__ == "__main__":
    main()
