import math
import os
import unittest
from unittest.mock import patch

from config import get_reranker_settings
from domain.models import Document, RetrievedDocument
from services.reranker_candidates import candidate_slice, union_candidates
from services.reranker_service import RerankerError, RerankerService


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def compute_score(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores[: len(pairs)]


def candidate(document_id, score=0.5, **fields):
    return RetrievedDocument(
        Document(document_id, f"content {document_id}", {"type": "lecturer"}),
        score=score,
        **fields,
    )


class RerankerServiceTests(unittest.TestCase):
    def test_empty_and_single_candidate_do_not_load_model(self):
        service = RerankerService(backend_factory=lambda: self.fail("must not load"))
        self.assertEqual(service.rerank("query", [], top_k=5), [])
        result = service.rerank("query", [candidate("a")], top_k=5)
        self.assertEqual(result[0].document.id, "a")
        self.assertIsNone(result[0].reranker_score)
        self.assertEqual(service.model_load_count, 0)

    def test_sort_top_k_metadata_and_original_rank_are_preserved(self):
        backend = FakeReranker([0.2, 0.9, 0.5])
        service = RerankerService(backend_factory=lambda: backend, batch_size=1)
        source = [candidate("a", 0.7), candidate("b", 0.6), candidate("c", 0.5)]
        result = service.rerank("query", source, top_k=2)
        self.assertEqual([item.document.id for item in result], ["b", "c"])
        self.assertEqual([item.original_rank for item in result], [2, 3])
        self.assertEqual([item.final_rank for item in result], [1, 2])
        self.assertEqual(result[0].original_score, 0.6)
        self.assertEqual(result[0].document.metadata["type"], "lecturer")
        self.assertEqual(service.model_load_count, 1)
        self.assertEqual(service.score_call_count, 1)

    def test_tie_is_stable_by_original_rank(self):
        backend = FakeReranker([0.5, 0.5, 0.5])
        service = RerankerService(backend_factory=lambda: backend)
        result = service.rerank(
            "query", [candidate("c"), candidate("a"), candidate("b")], top_k=3
        )
        self.assertEqual([item.document.id for item in result], ["c", "a", "b"])

    def test_non_finite_score_is_rejected(self):
        service = RerankerService(backend_factory=lambda: FakeReranker([math.nan, 0.2]))
        with self.assertRaises(RerankerError):
            service.rerank("query", [candidate("a"), candidate("b")])

    def test_union_deduplicates_and_preserves_branch_metadata(self):
        dense = [candidate("a", 0.9), candidate("b", 0.8), candidate("c", 0.7)]
        sparse = [candidate("c", 0.6), candidate("d", 0.5), candidate("e", 0.4)]
        result = union_candidates(dense, sparse)
        self.assertEqual([item.document.id for item in result], list("abcde"))
        self.assertEqual(result[2].dense_rank, 3)
        self.assertEqual(result[2].sparse_rank, 1)
        self.assertEqual(candidate_slice("union", dense, sparse, [], 3), result)

    def test_candidate_strategies_keep_existing_filtered_documents(self):
        dense = [candidate("d")]
        sparse = [candidate("s")]
        hybrid = [candidate("h")]
        self.assertEqual(candidate_slice("dense", dense, sparse, hybrid, 1), dense)
        self.assertEqual(candidate_slice("hybrid", dense, sparse, hybrid, 1), hybrid)
        self.assertEqual(
            [item.document.metadata["type"] for item in candidate_slice("union", dense, sparse, hybrid, 1)],
            ["lecturer", "lecturer"],
        )

    def test_reranker_defaults_are_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_reranker_settings()
        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["model_name"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(settings["device"], "cpu")
        self.assertEqual(settings["batch_size"], 1)
        self.assertEqual(settings["candidate_source"], "union")


@unittest.skipUnless(
    os.getenv("RUN_RERANKER_INTEGRATION") == "1",
    "Set RUN_RERANKER_INTEGRATION=1 untuk actual reranker test.",
)
class ActualRerankerIntegrationTests(unittest.TestCase):
    def test_indonesian_machine_learning_relevance(self):
        service = RerankerService(batch_size=1, max_length=256)
        scores = service.score_pairs(
            [
                ("dosen yang fokus machine learning", "Nama: Ratih. Research Interest: Machine learning"),
                ("dosen yang fokus machine learning", "Research Interest: optimization"),
                ("dosen yang fokus machine learning", "Research Interest: artificial intelligence"),
            ]
        )
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertEqual(max(range(len(scores)), key=scores.__getitem__), 0)
        self.assertEqual(service.model_load_count, 1)


if __name__ == "__main__":
    unittest.main()
