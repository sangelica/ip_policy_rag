import re
from functools import lru_cache
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.config import CONFIG
from src.vector_store import get_vector_store



STOP = set("a an the of to in on for and or is are be by with what which how many much "
           "does do must its it this that within at as from".split())

# TODO: State purpose of tokenizing here
def tokenize(text: str) -> list[str]:
    text = text.lower().replace(",", "")                 # "P20,000" -> "p20000"
    words = re.findall(r"[a-z0-9]+", text)               # drops ( ) . so "(7)" -> "7"
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
             for w in words]                              # crude plural stem: payments -> payment
    return [w for w in words if w not in STOP]

# TODO: State purpose of funtion here
@lru_cache(maxsize=1)
def bm25_index():
    """Load every chunk from Chroma once and build the BM25 index in memory."""
    data = get_vector_store().get()                       # ids, documents, metadatas
    docs = [Document(id=i, page_content=t, metadata=m)
            for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])]
    return BM25Okapi([tokenize(d.page_content) for d in docs]), docs

# TODO: State purpose of funtion here
def vector_search(query: str, k: int) -> list[Document]:
    return get_vector_store().similarity_search(query, k=k)

# TODO: State purpose of funtion here
def bm25_search(query: str, k: int) -> list[Document]:
    bm25, docs = bm25_index()
    scores = bm25.get_scores(tokenize(query))
    best = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in best if scores[i] > 0]

# TODO: State purpose of funtion here
# returns list of documetn objects
def hybrid_search(query: str) -> list[Document]:
    #TODO: add step by step comments here
    cfg = CONFIG["retrieval"]
    candidates_k, rrk_const = cfg["candidates_k"], cfg["rrf_k"]
    fused, by_id = {}, {}

    # run loop 2x, first 
    for ranked in (vector_search(query, candidates_k), bm25_search(query, candidates_k)):
        for rank, doc in enumerate(ranked, start=1):

            print(rank, doc.id)
            # 1 docA
            # 2 docB
            # 3 docC
     
            fused[doc.id] = fused.get(doc.id, 0) + 1 / (rrk_const + rank)
            by_id[doc.id] = doc

    print(fused)
    order = sorted(fused, key=fused.get, reverse=True)
    return [by_id[i] for i in order[:cfg["top_k"]]]

##############
def search(query, k=4):
    results = get_vector_store().similarity_search_with_score(query, k)

    for doc, score in results:
        preview = " ".join(doc.page_content.split())[:800]
        print(f"Score: {score:.4f} | Content: {preview}")
        print(f"Metadata: {doc.metadata}\n")

    return 

#search("What is the premium rate of direct contributors with salary 15,000 pesos?")
search("What is the 2025 premium rate for direct contributors?")
search("What does PhilHealth Konsulta cover?")