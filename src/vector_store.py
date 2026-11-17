from langchain_chroma import Chroma
from functools import lru_cache

from src.config import CONFIG, PROJECT_ROOT, get_embeddings

# One folder per provider: Ollama (768-dim) and Bedrock Titan vectors are not
# comparable, so they must never share a collection.
CHROMA_DIR = PROJECT_ROOT / CONFIG["ingest"]["chroma_dir"] / CONFIG["provider"]

# Only one instance for all modules
@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(collection_name = CONFIG["ingest"]["collection"],
                      embedding_function = get_embeddings(),
                      persist_directory = str(CHROMA_DIR))

