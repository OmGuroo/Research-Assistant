import logging
from typing import List
from app.rag.vector_store import get_collection
from app.schemas.research import SearchResult
from app.logger import logger

def retrieve_documents(query: str, k: int = 5, start_idx: int = 1) -> List[SearchResult]:
    """Retrieves semantically relevant document chunks from ChromaDB."""
    try:
        collection = get_collection()
        
        # Check if collection is empty
        if collection.count() == 0:
            return []
            
        # Chroma DB may raise an exception if k is larger than the collection count
        actual_k = min(k, collection.count())
            
        results = collection.query(
            query_texts=[query],
            n_results=actual_k
        )
        
        docs = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return []
            
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [None] * len(documents)
        
        for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            # We repurpose the URL field for page info to keep it simple, and title for filename
            docs.append(SearchResult(
                id=f"D{start_idx + idx}",
                title=meta.get("filename", "Unknown Document"),
                url=f"Page {meta.get('page', 'Unknown')}",
                content=doc,
                score=dist,
                domain="local_document",
                source_type=meta.get("source_type", "document")
            ))
            
        return docs
    except Exception as e:
        logger.error(f"Failed to retrieve documents from ChromaDB for query '{query}': {e}")
        return []
