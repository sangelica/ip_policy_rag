import re, time
from src.vector_store import get_vector_store
from src.schemas import QueryState, Answer
from src.config import get_llm,CONFIG
from src.retriever import hybrid_search, vector_search
from src.reranker import rerank


def guard(state: QueryState) -> dict:
    start = time.perf_counter()
    # best practice, this should return elements that changed
    return {"timings": {"guard": time.perf_counter() - start}} # no change yet, return query as is

def retrieve(state: QueryState) -> dict:
    start = time.perf_counter()
    cfg = CONFIG["retrieval"]
    if cfg.get("mode") == "hybrid":
        if CONFIG['rerank']['enabled']:
            ret_k = CONFIG["retrieval"]["fused_k"]
        else:
            ret_k = CONFIG["retrieval"]["top_k"]
        docs = hybrid_search(state["query"], ret_k)
    else:
        docs = vector_search(state["query"], CONFIG["retrieval"]["top_k"])

    return {"chunks": docs, "timings": {"retrieve": time.perf_counter() - start}}


def grade(state: QueryState) -> dict:
    start = time.perf_counter()

    # if reranker turned on, rerank top 'fused_k' elements
    if CONFIG["rerank"]["enabled"]:
        top_k_chunks = rerank(state["query"], state["chunks"], CONFIG["retrieval"]["top_k"])

        for doc, score in top_k_chunks:                       # log while you calibrate
            print(f"RERANK {score:.4f} {doc.id}")

        return {"chunks": [d for d, _ in top_k_chunks],
                #"rerank_scores": [float(s) for _, s in top_k_chunks],
                "relevant": True,                       # keep True until rewrite exists
                "timings": {"grade": time.perf_counter() - start}}
    else:
        # just return true if no need for reranker
        return {"relevant": True, "timings": {"grade": time.perf_counter() - start}}  # stub


def strip_space(s):
    s = s.lower()
    s = re.sub(r"[‘’“”'\"□]", "", s)               # 1. remove quote marks and the □ checkbox
    s = re.sub(r"(?<=\d)o|o(?=\d)", "0", s)        # 2. OCR fix: letter o next to a digit -> 0 ("6o" -> "60")
    s = re.sub(r"[^\w\s%.]", " ", s)               # 3. other punctuation ; : ( ) [ ] , - -> space
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)      # 4. full stops -> space, but keep decimals like 5.0
    return re.sub(r"\s+", " ", s).strip()          # 5. collapse spaces

def generate(state: QueryState) -> dict:
    start = time.perf_counter()
    llm = get_llm().with_structured_output(Answer)

    context = "\n\n".join(
        f"[{doc.metadata['source']} p.{doc.metadata['page']}]\n{doc.page_content}"
        for doc in state["chunks"])

    prompt = f"""You answer questions using ONLY the excerpts below.

                Excerpts: {context}
                Question: {state['query']}

                Rules:
                1. If the excerpts contain the answer to the question, write the answer and add at least one citation
                which includes source and page exactly as in the excerpt label, and a quote copied word for word.
                2. If the excerpts do not provide any answer to the question, say so in the answer, and return citations as an empty list.
                """

    #print(context)
    #print(prompt)

    ans = llm.invoke(prompt).model_dump()     # Answer object -> plain dict

    # make output dor citations deterministic
    retrieved = {(d.metadata["source"], d.metadata["page"]) for d in state["chunks"]}

    # compose the citations list
    citations_list = []

    print("RAW", ans["citations"])
    for citation in ans["citations"]:
        quote = strip_space(citation["quote"])

        if not quote:
            continue

        match = None
        for chunk in state["chunks"]:
            if quote in strip_space(chunk.page_content):
                match = chunk
                break          # stop at the first match

        if match:
            citations_list.append({"source": match.metadata["source"],
                                    "page": match.metadata["page"],
                                    "quote": citation["quote"]})
        else:
            print("dropped", citation)

    ans["citations"] = citations_list

    return {"answer": ans, "timings": {"generate": time.perf_counter() - start}}

def rewrite(state: QueryState) -> dict:
    start = time.perf_counter()
    return {"timings": {"rewrite": time.perf_counter() - start}} # no change yet, return query as is

def check(state: QueryState) -> dict:
    start = time.perf_counter()
    return {"grounded": True, "timings": {"check": time.perf_counter() - start}}  # stub

def fallback(state: QueryState) -> dict:
    start = time.perf_counter()
    return {"timings": {"fallback": time.perf_counter() - start}}  # stub
