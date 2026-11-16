import pymupdf
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CONFIG, PROJECT_ROOT, get_embeddings

FILE = PROJECT_ROOT / CONFIG["ingest"]["raw_dir"] / "PA2025-0002.pdf"

""" Ingest pdf file"""
def ingest(file=FILE):
    with pymupdf.open(file) as pdf_file:

        # Built once per file, not per page; sizes come from config.yaml so evals record them.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["ingest"]["chunk_size"],
            chunk_overlap=CONFIG["ingest"]["chunk_overlap"],
        )

        for page_num in range(len(pdf_file)):
            text = pdf_file[page_num].get_text("text")
            page = page_num + 1  # PyMuPDF is 0-based; citations use human page numbers

            if not text.strip():  # blank or scanned (image-only) page: nothing to embed
                print(f"skipped page {page}: no text layer")
                continue

            chunks = splitter.split_text(text)

            docs = []
            ids = []
            for i, chunk in enumerate(chunks):
                metadata = {}
                metadata["source"] = Path(file).name
                metadata["page"] = page
                metadata["chunk"] = i
                doc = Document(page_content = chunk, metadata=metadata)
                ids.append(f"{Path(file).stem}-p{page}-c{i}")
                docs.append(doc)
            VECTOR_STORE.add_documents(docs, ids=ids)

    #pdf_file.close()


def add_to_chroma():
    pass

"""
1. open the file
2. ingest the file by chunks
3. add to chroma db by chunks
"""

# One folder per provider: Ollama (768-dim) and Bedrock Titan vectors are not
# comparable, so they must never share a collection.
CHROMA_DIR = PROJECT_ROOT / CONFIG["ingest"]["chroma_dir"] / CONFIG["provider"]

VECTOR_STORE = Chroma(collection_name = CONFIG["ingest"]["collection"],
                      embedding_function = get_embeddings(),
                      persist_directory = str(CHROMA_DIR))


# Runs only with `python -m src.ingest`, not when another module imports this file.
if __name__ == "__main__":
    VECTOR_STORE.reset_collection()
    ingest(FILE)

    print(len(VECTOR_STORE.get()["ids"]))
    results = VECTOR_STORE.similarity_search_with_score("What is the 2025 premium rate for direct contributors?", k=4)

    for doc, score in results:
        preview = " ".join(doc.page_content.split())[:800]
        print(f"Score: {score:.4f} | Content: {preview}")
        print(f"Metadata: {doc.metadata}\n")
