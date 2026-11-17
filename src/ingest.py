import pymupdf
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CONFIG, PROJECT_ROOT, get_embeddings
from src.vector_store import get_vector_store

FILE = PROJECT_ROOT / CONFIG["ingest"]["raw_dir"] / "PA2025-0002.pdf"

""" Ingest pdf file"""
def ingest(vector_store: Chroma, file=FILE):
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
            vector_store.add_documents(docs, ids=ids)

    #pdf_file.close()


# Runs only with `python -m src.ingest`, not when another module imports this file.
if __name__ == "__main__":
    vector_store = get_vector_store()
    #vector_store.reset_collection()
    ingest(vector_store, FILE)

    print(len(vector_store.get()["ids"]))
