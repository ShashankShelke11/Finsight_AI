"""
FinSight AI Retrieval Package
Dense embeddings, vector store adapter, and baseline vector search engine.
"""

from retrieval.embeddings import EmbeddingService, embedding_service
from retrieval.search_engine import RetrievalResponse, VectorSearchEngine, search_engine
from retrieval.vector_store import SearchResult, VectorStore

__all__ = [
    "EmbeddingService",
    "embedding_service",
    "SearchResult",
    "VectorStore",
    "RetrievalResponse",
    "VectorSearchEngine",
    "search_engine",
]
