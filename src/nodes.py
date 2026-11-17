import re, time
from src.vector_store import get_vector_store
from src.schemas import QueryState, Answer
from src.config import get_llm


def guard(state: QueryState) -> dict:
    # best practice, this should return elements that changed
    return {} # no change yet, return query as is

def retrieve(state: QueryState) -> dict:
    docs = get_vector_store().similarity_search(state["query"], k = 4)
    return {"chunks": docs}

def grade(state: QueryState) -> dict:
    return {"relevant": True}                    # stub

def generate(state: QueryState) -> dict:
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
                2. If the excerpts do not provide any answer to the question, say so in the answer, and leave all fields in citations empty
                including the source, page and quote.
                """

    #print(context)
    #print(prompt)

    ans = llm.invoke(prompt)          # returns an Answer object, not text
    
    return {"answer": ans.model_dump()}

def rewrite(state: QueryState) -> dict:
    return {} # no change yet, return query as is

def check(state: QueryState) -> dict:
    return {"grounded": True}                    # stub

def fallback(state: QueryState) -> dict:
    return {}                    # stub