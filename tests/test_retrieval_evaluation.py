"""
Integration and Benchmark Tests for Retrieval Evaluation (Experiment A)
"""

from __future__ import annotations

import unittest
from core.database import db
from evaluation.retrieval_eval import RetrievalEvaluator


class TestRetrievalEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        from scripts.seed_sample_data import seed_database
        seed_database()
        cls.evaluator = RetrievalEvaluator()

    def test_run_experiment_a_benchmarks(self):
        """Runs comparative Experiment A over golden queries and asserts performance targets."""
        results = self.evaluator.run_experiment_a()

        self.assertIn("vector", results)
        self.assertIn("hybrid", results)
        self.assertIn("hybrid_rerank", results)

        vector_res = results["vector"]
        hybrid_res = results["hybrid"]
        rerank_res = results["hybrid_rerank"]

        # 1. Assert minimum Recall@5 performance
        self.assertGreaterEqual(vector_res.recall_at_5, 0.70)
        self.assertGreaterEqual(hybrid_res.recall_at_5, 0.80)
        self.assertGreaterEqual(rerank_res.recall_at_5, 0.80)

        # 2. Assert MRR is strictly positive
        self.assertGreater(vector_res.mean_reciprocal_rank, 0.50)
        self.assertGreater(hybrid_res.mean_reciprocal_rank, 0.60)
        self.assertGreater(rerank_res.mean_reciprocal_rank, 0.60)

        # 3. Assert NDCG@5 is positive
        self.assertGreater(rerank_res.mean_ndcg_at_5, 0.60)

        # 4. Latency must be reasonably fast (< 200ms per query in local environment)
        self.assertLess(rerank_res.mean_latency_ms, 200.0)

        # 5. Format and verify report output
        report = self.evaluator.format_markdown_report(results)
        self.assertIn("Experiment A: Retrieval Architecture Comparison", report)
        self.assertIn("Hybrid + Cross-Encoder Rerank", report)
        print("\n" + report + "\n")


if __name__ == "__main__":
    unittest.main()
