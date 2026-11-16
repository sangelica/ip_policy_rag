"""Provider switch: the only module that knows which model service is in use.

Other modules import get_llm() / get_embeddings() and receive standard LangChain
interfaces, so swapping Ollama -> Bedrock is a change to config.yaml only.
"""
from pathlib import Path

import yaml

__all__ = ["PROJECT_ROOT", "CONFIG", "load_config", "get_llm", "get_embeddings"]

# Project root, found from this file's location, so paths work whatever folder you run from.
# Relative paths in config.yaml (e.g. data/chroma) are resolved against this.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml (located relative to this file, not the working directory)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Loaded once, at first import. Python caches modules, so every importer shares
# this dict. Edits to config.yaml need a process restart to take effect.
CONFIG = load_config()


def get_llm():
    """Chat model used for generation (and, in the skeleton, grading/checking)."""
    provider = CONFIG["provider"]
    if provider == "ollama":
        # Imports live inside branches so unused providers need not be installed.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=CONFIG["ollama"]["llm_model"],
            base_url=CONFIG["ollama"]["base_url"],
            temperature=0,  # repeatable answers for policy Q&A and evals
        )
    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model=CONFIG["bedrock"]["llm_model"],
            region_name=CONFIG["bedrock"]["region"],
            temperature=0,
        )
    raise ValueError(f"Unknown provider in config.yaml: {provider!r}")


def get_embeddings():
    """Embedding model used by ingestion and retrieval (must be the same for both)."""
    provider = CONFIG["provider"]
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=CONFIG["ollama"]["embed_model"],
            base_url=CONFIG["ollama"]["base_url"],
        )
    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(
            model_id=CONFIG["bedrock"]["embed_model"],
            region_name=CONFIG["bedrock"]["region"],
        )
    raise ValueError(f"Unknown provider in config.yaml: {provider!r}")
