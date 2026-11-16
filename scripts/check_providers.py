"""Smoke test for src/config.py.

Run from the project root (with Ollama running):
    python -m scripts.check_providers
"""
import time

from src.config import CONFIG, get_embeddings, get_llm


def main() -> None:
    print(f"provider: {CONFIG['provider']}")

    t = time.perf_counter()
    vector = get_embeddings().embed_query("hi")
    print(f"embeddings OK: {len(vector)} dims ({time.perf_counter() - t:.2f}s)")

    t = time.perf_counter()
    reply = get_llm().invoke("Say hi in five words.").content
    print(f"llm OK: {reply!r} ({time.perf_counter() - t:.2f}s)")


if __name__ == "__main__":
    main()
