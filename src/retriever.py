from src.vector_store import get_vector_store

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