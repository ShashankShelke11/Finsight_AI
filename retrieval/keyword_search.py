"""
BM25 Lexical and Keyword Search Engine for FinSight AI
Provides term-frequency saturation, document length normalization,
exact phrase boosting, and metadata filtering over financial disclosures.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from core.database import db
from core.logging import logger
from retrieval.vector_store import SearchResult


class BM25Scorer:
    """Computes Okapi BM25 scores with phrase proximity enhancements."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenizes text into lowercase alphanumeric keywords."""
        if not text:
            return []
        return re.findall(r"\b[a-z0-9_\$%\.]+\b", text.lower())

    def compute_scores(
        self,
        query: str,
        corpus: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Scores each document in corpus against query using BM25 with exact phrase boosting.
        Returns list of (doc_dict, bm25_score) tuples.
        """
        if not corpus:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return [(d, 0.0) for d in corpus]

        N = len(corpus)
        # Tokenize all documents and track document frequencies n(q)
        doc_token_lists: List[List[str]] = [self.tokenize(d.get("clean_text") or d.get("content") or "") for d in corpus]
        doc_lengths = [len(tokens) for tokens in doc_token_lists]
        avgdl = sum(doc_lengths) / N if N > 0 else 1.0

        # Document frequency n(t) for each query term
        doc_freqs: Dict[str, int] = Counter()
        for tokens in doc_token_lists:
            unique_terms = set(tokens)
            for t in query_tokens:
                if t in unique_terms:
                    doc_freqs[t] += 1

        # Calculate IDF for each query term: ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
        idf: Dict[str, float] = {}
        for t in set(query_tokens):
            n_t = doc_freqs.get(t, 0)
            idf[t] = math.log(((N - n_t + 0.5) / (n_t + 0.5)) + 1.0)

        # Score documents
        clean_query = query.lower().strip()
        scored_docs: List[Tuple[Dict[str, Any], float]] = []

        for doc, tokens, dl in zip(corpus, doc_token_lists, doc_lengths):
            tf_counter = Counter(tokens)
            doc_score = 0.0

            # 1. BM25 term scores
            for t in query_tokens:
                f_td = tf_counter.get(t, 0)
                if f_td > 0:
                    numerator = f_td * (self.k1 + 1.0)
                    denominator = f_td + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                    doc_score += idf[t] * (numerator / denominator)

            # 2. Exact phrase bonus if the exact query or bigrams appear in text
            doc_text = (doc.get("clean_text") or doc.get("content") or "").lower()
            if len(clean_query.split()) > 1 and clean_query in doc_text:
                doc_score += 2.5 * math.log(len(clean_query.split()) + 1)

            # Heading/Section relevance bonus
            sec_name = (doc.get("section_name") or "").lower()
            if any(t in sec_name for t in query_tokens):
                doc_score += 1.5

            scored_docs.append((doc, doc_score))

        return scored_docs


class KeywordSearchEngine:
    """Executes lexical BM25 search over document chunks with metadata filtering."""

    scorer = BM25Scorer(k1=1.5, b=0.75)

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        reporting_period: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Executes BM25 keyword search over stored document chunks in the database.
        Applies metadata filters and returns ranked SearchResult items.
        """
        if not query or not query.strip():
            return []

        query_sql = """
            SELECT 
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                d.fiscal_year,
                d.reporting_period,
                comp.ticker AS company_ticker,
                c.page_number,
                c.section_name,
                c.content,
                c.clean_text,
                c.is_table
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            JOIN companies comp ON d.company_id = comp.id
            WHERE 1=1
        """
        params: List[Any] = []
        if ticker:
            query_sql += " AND comp.ticker = ?" if db.is_sqlite else " AND comp.ticker = %s"
            params.append(ticker.upper())
        if fiscal_year:
            query_sql += " AND d.fiscal_year = ?" if db.is_sqlite else " AND d.fiscal_year = %s"
            params.append(fiscal_year)
        if reporting_period:
            query_sql += " AND d.reporting_period = ?" if db.is_sqlite else " AND d.reporting_period = %s"
            params.append(reporting_period)

        candidate_chunks = db.fetch_all(query_sql, tuple(params))
        if not candidate_chunks:
            return []

        # Score candidates with BM25
        scored_pairs = cls.scorer.compute_scores(query, candidate_chunks)

        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        # Normalize scores to 0.0 - 1.0 range for comparability
        max_score = scored_pairs[0][1] if scored_pairs and scored_pairs[0][1] > 0 else 1.0

        results: List[SearchResult] = []
        for doc, raw_score in scored_pairs[:top_k]:
            normalized_score = raw_score / max_score if max_score > 0 else 0.0
            ref = f"{doc['company_ticker']} {doc['reporting_period']}, Page {doc['page_number']} ({doc['section_name']})"
            results.append(
                SearchResult(
                    chunk_id=doc["chunk_id"],
                    document_id=doc["document_id"],
                    company_ticker=doc["company_ticker"],
                    document_title=doc["document_title"],
                    fiscal_year=doc["fiscal_year"],
                    reporting_period=doc["reporting_period"],
                    page_number=doc["page_number"],
                    section_name=doc["section_name"] or "Overview",
                    content=doc["content"],
                    clean_text=doc["clean_text"],
                    score=round(normalized_score, 4),
                    is_table=bool(doc["is_table"]),
                    source_reference=ref,
                )
            )

        logger.debug(
            f"Keyword BM25 search for '{query[:30]}...' retrieved {len(results)} chunks "
            f"(top_score={results[0].score if results else 0.0})"
        )
        return results
