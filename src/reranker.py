from functools import lru_cache
from sentence_transformers import CrossEncoder
from src.config import CONFIG

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the model once; later calls reuse it."""
    return CrossEncoder(CONFIG["rerank"]["model"])

def rerank(query: str, docs: list, top_k: int) -> list[tuple]:
    """Score each (query, chunk) pair together; return the best top_k as (doc, score)."""
    pairs = [(query, d.page_content) for d in docs]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# used to call reranker once
if __name__ == "__main__":
    get_reranker()