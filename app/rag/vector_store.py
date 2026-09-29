import os
import chromadb
from chromadb.utils import embedding_functions
from app.config import settings

def get_chroma_client() -> chromadb.PersistentClient:
    """Returns the persistent ChromaDB client."""
    # Ensure the directory exists
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)

def get_embedding_function():
    """Returns the configured sentence-transformers embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model_name
    )

def get_collection():
    """Returns the default collection for research documents."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="research_documents",
        embedding_function=get_embedding_function()
    )
