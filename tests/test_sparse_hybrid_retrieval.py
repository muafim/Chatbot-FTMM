import unittest

import numpy as np

from domain.models import DocumentChunk, QueryIntent, RetrievalPlan, SparseVector
from indexes.in_memory_sparse_index import InMemorySparseIndex
from indexes.in_memory_vector_index import InMemoryVectorIndex
from retrievers.dense_retriever import DenseRetriever
from services.rank_fusion import reciprocal_rank_fusion


def sparse(**values):
    return SparseVector.from_mapping(values)


class SparseIndexTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            DocumentChunk("a", "a", "machine learning", 0, {"type": "lecturer", "chunk_role": "research_interest"}),
            DocumentChunk("b", "b", "data mining", 0, {"type": "lecturer", "chunk_role": "research_interest"}),
            DocumentChunk("c", "c", "machine learning course", 0, {"type": "course"}),
        ]
        self.index = InMemorySparseIndex()
        self.index.build(
            self.documents,
            [sparse(machine=1.0, learning=0.8), sparse(data=1.0), sparse(machine=0.7)],
            embedding_model="bge",
        )

    def test_search_exact_term_top_k_and_metadata_filter(self):
        results = self.index.search(sparse(machine=1.0), top_k=2, embedding_model="bge")
        self.assertEqual([item.document.id for item in results], ["a", "c"])
        filtered = self.index.search(
            sparse(machine=1.0), top_k=5,
            metadata_filter={"type": "lecturer", "chunk_role": "research_interest"},
        )
        self.assertEqual([item.document.id for item in filtered], ["a"])

    def test_no_matching_token_returns_empty(self):
        self.assertEqual(self.index.search(sparse(unknown=1.0), top_k=5), [])

    def test_tie_break_is_deterministic_by_document_id(self):
        first = self.index.search(sparse(machine=1.0), top_k=5)
        second = self.index.search(sparse(machine=1.0), top_k=5)
        self.assertEqual([item.document.id for item in first], [item.document.id for item in second])


class RRFFusionTests(unittest.TestCase):
    def setUp(self):
        from domain.models import Document, RetrievedDocument

        docs = {key: Document(key, key, {}) for key in "ABCD"}
        self.dense = [RetrievedDocument(docs[key], score=score) for key, score in zip("ABC", (0.9, 0.8, 0.7))]
        self.sparse = [RetrievedDocument(docs[key], score=score) for key, score in zip("CAD", (3.0, 2.0, 1.0))]

    def test_equal_weight_rrf_merges_duplicates_and_scores(self):
        fused = reciprocal_rank_fusion(self.dense, self.sparse, rrf_k=60)
        self.assertEqual(len(fused), 4)
        by_id = {item.document.id: item for item in fused}
        self.assertAlmostEqual(by_id["A"].fusion_score, 1 / 61 + 1 / 62)
        self.assertEqual(by_id["A"].dense_rank, 1)
        self.assertEqual(by_id["A"].sparse_rank, 2)
        self.assertIsNone(by_id["B"].sparse_rank)

    def test_weight_and_k_are_configurable_and_ties_deterministic(self):
        first = reciprocal_rank_fusion(self.dense, self.sparse, rrf_k=10, dense_weight=2.0, sparse_weight=1.0)
        second = reciprocal_rank_fusion(self.dense, self.sparse, rrf_k=10, dense_weight=2.0, sparse_weight=1.0)
        self.assertEqual([item.document.id for item in first], [item.document.id for item in second])
        self.assertGreater(first[0].fusion_score, 0)


class FakeHybridService:
    model_name = "bge"

    def encode_query_hybrid(self, query):
        from domain.models import HybridEmbedding

        return HybridEmbedding(np.asarray([1.0, 0.0]), sparse(machine=1.0))


class FakeColdHybridService(FakeHybridService):
    model_name = "bge"
    cache_configuration = {"backend": "bge"}

    def __init__(self):
        self.hybrid_document_calls = 0
        self.dense_document_calls = 0
        self.sparse_document_calls = 0

    def encode_documents_hybrid(self, texts):
        from domain.models import HybridEmbeddingBatch

        items = list(texts)
        self.hybrid_document_calls += 1
        return HybridEmbeddingBatch(
            np.asarray([[1.0, 0.0] for _ in items]),
            tuple(sparse(machine=1.0) for _ in items),
        )

    def encode_documents(self, texts):
        self.dense_document_calls += 1
        raise AssertionError("Dense corpus pass terpisah tidak boleh dipanggil.")

    def encode_documents_sparse(self, texts):
        self.sparse_document_calls += 1
        raise AssertionError("Sparse corpus pass terpisah tidak boleh dipanggil.")

    def encode_query_sparse(self, query):
        return sparse(machine=1.0)


class HybridFilterTests(unittest.TestCase):
    def test_cold_hybrid_initialization_encodes_corpus_once(self):
        class Repository:
            def get_documents(self):
                from domain.models import Document

                return [Document("one", "machine learning", {"type": "lecturer"})]

        service = FakeColdHybridService()
        retriever = DenseRetriever(
            Repository(), service, InMemoryVectorIndex(),
            sparse_index=InMemorySparseIndex(), retrieval_mode="hybrid",
            min_score=None,
        )
        retriever.initialize()
        self.assertEqual(service.hybrid_document_calls, 1)
        self.assertEqual(service.dense_document_calls, 0)
        self.assertEqual(service.sparse_document_calls, 0)
        self.assertTrue(retriever.hybrid_embedding_one_pass)
        self.assertEqual(retriever.vector_index.document_count, 1)
        self.assertEqual(retriever.sparse_index.document_count, 1)

    def test_same_multifield_filter_applies_to_both_branches(self):
        documents = [
            DocumentChunk("lecturer", "lecturer", "machine learning", 0, {"type": "lecturer", "chunk_role": "research_interest"}),
            DocumentChunk("course", "course", "machine learning", 0, {"type": "course"}),
        ]
        dense_index = InMemoryVectorIndex()
        dense_index.build(documents, np.asarray([[0.8, 0.2], [1.0, 0.0]]), embedding_model="bge")
        sparse_index = InMemorySparseIndex()
        sparse_index.build(documents, [sparse(machine=0.8), sparse(machine=1.0)], embedding_model="bge")
        retriever = DenseRetriever(
            repository=None,
            embedding_service=FakeHybridService(),
            vector_index=dense_index,
            sparse_index=sparse_index,
            retrieval_mode="hybrid",
            min_score=None,
        )
        plan = RetrievalPlan(
            "dosen machine learning",
            QueryIntent.LECTURER,
            {"type": "lecturer", "chunk_role": "research_interest"},
        )
        results = retriever.retrieve_plan(plan, top_k=5)
        self.assertEqual([item.document.id for item in results], ["lecturer"])
        self.assertEqual([item.document.id for item in retriever.last_dense_candidates], ["lecturer"])
        self.assertEqual([item.document.id for item in retriever.last_sparse_candidates], ["lecturer"])


if __name__ == "__main__":
    unittest.main()
