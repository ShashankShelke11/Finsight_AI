"""
FinSight AI Retrieval Package
Dense embeddings, BM25 keyword search, Reciprocal Rank Fusion,
Cross-Encoder reranking, and the Unified Hybrid Search Engine.
"""

from retrieval.embeddings import EmbeddingService, embedding_service
from retrieval.hybrid import HybridRetriever, HybridSearchResult, hybrid_retriever
from retrieval.keyword_search import BM25Scorer, KeywordSearchEngine
from retrieval.reranker import CrossEncoderReranker, RerankedResult, cross_encoder_reranker
from retrieval.search_engine import (
    RetrievalResponse,
    UnifiedSearchEngine,
    VectorSearchEngine,
    search_engine,
)
from retrieval.vector_store import SearchResult, VectorStore

__all__ = [
    "EmbeddingService",
    "embedding_service",
    "SearchResult",
    "VectorStore",
    "BM25Scorer",
    "KeywordSearchEngine",
    "HybridSearchResult",
    "HybridRetriever",
    "hybrid_retriever",
    "RerankedResult",
    "CrossEncoderReranker",
    "cross_encoder_reranker",
    "RetrievalResponse",
    "UnifiedSearchEngine",
    "VectorSearchEngine",
    "search_engine",
]
