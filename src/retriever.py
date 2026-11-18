"""Retrieval: vector search, BM25 keyword search, and hybrid search fusing both with RRF.

The graph's retrieve node calls hybrid_search() or vector_search(), depending on
retrieval.mode in config.yaml.
"""
import re
from functools import lru_cache
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.config import CONFIG
from src.vector_store import get_vector_store


# Words too common to help keyword search; dropped from both queries and chunks.
STOP_WORDS = set("a an the of to in on for and or is are be by with what which how many much "
                 "does do must its it this that within at as from".split())


def tokenize(text: str) -> list[str]:
    """Split text into normalised keywords for BM25.

    BM25 matches exact words, so the query and the chunks must be tokenised the same
    way: lowercase, numbers without thousands separators, punctuation dropped, a crude
    plural stem, and stop words removed. "Payments of P20,000 within seven (7) days"
    becomes ["payment", "p20000", "seven", "7", "day"].

    Args:
        text: A query or a chunk's text.

    Returns:
        The keywords, in order.
    """
    text = text.lower().replace(",", "")                 # "P20,000" -> "p20000"
    words = re.findall(r"[a-z0-9]+", text)               # drops ( ) . so "(7)" -> "7"
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
             for w in words]                              # crude plural stem: payments -> payment
    return [w for w in words if w not in STOP_WORDS]


@lru_cache(maxsize=1)
def bm25_index():
    """Load every chunk from Chroma once and build the BM25 index in memory.

    Chroma stores the chunks but has no keyword search, so BM25 needs its own index.
    It is cached for the life of the process: restart after re-ingesting, or the index
    will still hold the old chunks.

    Returns:
        A (BM25Okapi, docs) pair. Score i from the index belongs to docs[i].
    """
    data = get_vector_store().get()  # ids, documents, metadatas

    # .get() returns parallel lists, not Documents; rebuild Documents keeping the id,
    # which hybrid_search() uses to recognise the same chunk from both searches.
    docs = [Document(id=i, page_content=t, metadata=m)
            for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])]

    return BM25Okapi([tokenize(d.page_content) for d in docs]), docs


def vector_search(query: str, k: int) -> list[Document]:
    """Return the k chunks closest in meaning to the query (embedding similarity).

    Good at paraphrases ("paid late" vs "payment delay"), weaker on exact terms such
    as circular numbers, peso amounts and abbreviations.

    Args:
        query: The user's question.
        k: How many chunks to return.

    Returns:
        Up to k Documents, closest first.
    """
    return get_vector_store().similarity_search(query, k=k)



def bm25_search(query: str, k: int) -> list[Document]:
    """Return the k chunks that best match the query's keywords (BM25).

    Good at exact terms that vector search can miss. Chunks sharing no keyword with
    the query (score 0) are left out, so fewer than k may be returned.

    Args:
        query: The user's question.
        k: The most chunks to return.

    Returns:
        Up to k Documents, best match first.
    """
    bm25, docs = bm25_index()
    scores = bm25.get_scores(tokenize(query))             # one score per chunk

    # Chunk positions sorted by score, highest first; keep the top k.
    top_positions = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_positions if scores[i] > 0]


def hybrid_search(query: str, ret_k: int) -> list[Document]:
    """Combine vector and BM25 results with Reciprocal Rank Fusion (RRF).

    Each search returns its top candidates_k chunks. A chunk scores 1 / (rrf_k + rank)
    for every list it appears in, and the scores are summed, so a chunk that both
    searches rank well beats one found by only one. RRF uses ranks, not raw scores,
    so the two searches' different score scales don't need calibrating.

    Settings come from config.yaml: retrieval.candidates_k, rrf_k and top_k.

    Args:
        query: The user's question.

    Returns:
        The top_k Documents by fused score, best first.
    """

    cfg = CONFIG["retrieval"]

    candidates_k, rrf_k = cfg["candidates_k"], cfg["rrf_k"]
    rrf_scores, docs_by_id = {}, {}   # chunk id -> fused score; chunk id -> Document

    # Go through each search's ranked list in turn: vector first, then BM25.
    for ranking in (vector_search(query, candidates_k), bm25_search(query, candidates_k)):
        for rank, doc in enumerate(ranking, start=1):
            print(f" rank {rank:>2}  {doc.id}")
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1 / (rrf_k + rank)
            docs_by_id[doc.id] = doc

    order = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [docs_by_id[i] for i in order[:ret_k]]


def search(query, k=4):
    """Print the top k vector-search chunks with their distance scores (manual check).

    Lower score = closer match (Chroma returns distances).

    Args:
        query: The question to test.
        k: How many chunks to show.
    """
    results = get_vector_store().similarity_search_with_score(query, k)

    for doc, score in results:
        preview = " ".join(doc.page_content.split())[:800]
        print(f"Score: {score:.4f} | Content: {preview}")
        print(f"Metadata: {doc.metadata}\n")

    return 

#search("What is the premium rate of direct contributors with salary 15,000 pesos?")
if __name__ == "__main__":
    search("What is the 2025 premium rate for direct contributors?")
    search("What does PhilHealth Konsulta cover?")