"""
Retrieval Evaluation Engine and Experiment A Runner for FinSight AI
Evaluates Recall@K, MRR, NDCG@K, and Latency across:
- Experiment A1: Vector-Only RAG
- Experiment A2: Hybrid Retrieval (Vector + BM25 RRF)
- Experiment A3: Hybrid Retrieval + Cross-Encoder Reranking
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.config import settings
from core.database import db
from core.logging import logger
from retrieval.search_engine import UnifiedSearchEngine


@dataclass
class MetricSummary:
    """Aggregate retrieval benchmark metrics for a single experiment configuration."""
    mode: str
    query_count: int
    hit_rate_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float
    mean_ndcg_at_5: float
    mean_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "query_count": self.query_count,
            "hit_rate_at_1": round(self.hit_rate_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mean_reciprocal_rank, 4),
            "ndcg_at_5": round(self.mean_ndcg_at_5, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


class RetrievalEvaluator:
    """Evaluates retrieval quality against curated golden query benchmarks."""

    def __init__(self, golden_dataset_path: Optional[Path] = None):
        self.dataset_path = (
            golden_dataset_path
            or settings.BASE_DIR / "evaluation" / "dataset" / "retrieval_golden_queries.json"
        )
        self.queries = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Loads golden queries."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def is_chunk_relevant(cls, chunk_content: str, chunk_section: str, chunk_ticker: str, golden: Dict[str, Any]) -> bool:
        """Determines if a retrieved chunk matches ground truth company, section, and keywords."""
        if chunk_ticker.upper() != golden["company_ticker"].upper():
            return False

        content_lower = chunk_content.lower()
        sec_lower = chunk_section.lower()

        # Check expected keywords (at least 50% must be present in the passage)
        expected = [k.lower() for k in golden.get("expected_keywords", [])]
        if expected:
            matched = sum(1 for kw in expected if kw in content_lower or kw in sec_lower)
            if (matched / len(expected)) >= 0.4:
                return True

        # Check target section matching
        target_sec = golden.get("target_section", "").lower()
        if target_sec and target_sec in sec_lower:
            return True

        return False

    def evaluate_mode(self, mode: str, top_k: int = 5) -> MetricSummary:
        """Evaluates a retrieval mode over all golden queries."""
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        reciprocal_ranks: List[float] = []
        ndcg_scores: List[float] = []
        latencies: List[float] = []

        for item in self.queries:
            query = item["query"]
            ticker = item.get("company_ticker")

            t0 = time.perf_counter()
            response = UnifiedSearchEngine.search(
                query=query,
                mode=mode,
                top_k=top_k,
                ticker=ticker,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

            # Evaluate relevance of ranked results
            first_relevant_rank: Optional[int] = None
            relevances: List[int] = []

            for rank, r in enumerate(response.results, start=1):
                is_rel = self.is_chunk_relevant(r.content, r.section_name, r.company_ticker, item)
                relevances.append(1 if is_rel else 0)
                if is_rel and first_relevant_rank is None:
                    first_relevant_rank = rank

            # Compute Recall@K
            if first_relevant_rank == 1:
                hits_at_1 += 1
            if first_relevant_rank is not None and first_relevant_rank <= 3:
                hits_at_3 += 1
            if first_relevant_rank is not None and first_relevant_rank <= 5:
                hits_at_5 += 1

            # Reciprocal Rank
            if first_relevant_rank is not None:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)

            # Compute NDCG@5: DCG / IDCG
            dcg = sum((rel / math.log2(idx + 1)) for idx, rel in enumerate(relevances, start=1))
            ideal_relevances = sorted(relevances, reverse=True)
            idcg = sum((rel / math.log2(idx + 1)) for idx, rel in enumerate(ideal_relevances, start=1))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        N = len(self.queries)
        sorted_latencies = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_latencies))
        p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]

        return MetricSummary(
            mode=mode,
            query_count=N,
            hit_rate_at_1=(hits_at_1 / N) if N else 0.0,
            recall_at_3=(hits_at_3 / N) if N else 0.0,
            recall_at_5=(hits_at_5 / N) if N else 0.0,
            mean_reciprocal_rank=statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            mean_ndcg_at_5=statistics.mean(ndcg_scores) if ndcg_scores else 0.0,
            mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            p95_latency_ms=p95,
        )

    def run_experiment_a(self) -> Dict[str, MetricSummary]:
        """Runs comparative Experiment A: Vector-only vs Hybrid vs Hybrid+Rerank."""
        logger.info("Starting Experiment A: Comparative Retrieval Evaluation...")
        modes = ["vector", "hybrid", "hybrid_rerank"]
        results: Dict[str, MetricSummary] = {}

        for m in modes:
            logger.info(f"Running evaluation benchmark for mode: '{m}'...")
            res = self.evaluate_mode(mode=m, top_k=5)
            results[m] = res
            logger.info(
                f"Mode '{m}' -> Recall@5: {res.recall_at_5*100:.1f}%, "
                f"MRR: {res.mean_reciprocal_rank:.4f}, NDCG@5: {res.mean_ndcg_at_5:.4f}, "
                f"Mean Latency: {res.mean_latency_ms:.2f}ms"
            )

        return results

    def format_markdown_report(self, results: Dict[str, MetricSummary]) -> str:
        """Formats comparative benchmark results into a clean markdown table."""
        lines = [
            "### Experiment A: Retrieval Architecture Comparison",
            "",
            "| Architecture Mode | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 | Mean Latency (ms) | P95 Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        labels = {
            "vector": "1. Vector-Only Baseline",
            "hybrid": "2. Hybrid (Vector + BM25 RRF)",
            "hybrid_rerank": "3. Hybrid + Cross-Encoder Rerank",
        }
        for mode, summary in results.items():
            name = labels.get(mode, mode)
            lines.append(
                f"| **{name}** | {summary.hit_rate_at_1*100:.1f}% | {summary.recall_at_3*100:.1f}% | "
                f"{summary.recall_at_5*100:.1f}% | {summary.mean_reciprocal_rank:.4f} | "
                f"{summary.mean_ndcg_at_5:.4f} | {summary.mean_latency_ms:.2f}ms | {summary.p95_latency_ms:.2f}ms |"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    # Ensure project root in sys.path when run directly
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    evaluator = RetrievalEvaluator()
    exp_results = evaluator.run_experiment_a()
    report = evaluator.format_markdown_report(exp_results)
    print("\n" + report + "\n")
