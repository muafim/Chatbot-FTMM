import unittest
from pathlib import Path

import numpy as np

from domain.models import Document, DocumentChunk, QueryIntent, RetrievedDocument, RetrievalPlan
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.knowledge_repository import KnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from services.chat_service import ChatService
from services.context_builder import ContextBuilder
from services.embedding_service import EmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingModel:
    def encode(self, texts):
        return np.asarray([[len(text), text.lower().count("a")] for text in texts])


class FakeEmbeddingService:
    def __init__(self):
        self.document_encode_calls = 0

    def encode_documents(self, texts):
        self.document_encode_calls += 1
        return np.asarray([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]])

    def encode_query(self, query):
        return np.asarray([1.0, 0.0])


class FakeRepository:
    def __init__(self, documents):
        self.documents = documents

    def get_documents(self):
        return list(self.documents)


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=None):
        return self.results


class CapturingLLMService:
    def __init__(self):
        self.context = None

    def generate_answer(self, question, context, conversation_history=None):
        self.context = context
        return "jawaban-test"


class RetrievalComponentTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                id="course-a",
                content="Nama Mata Kuliah: A\nDeskripsi: machine learning",
                metadata={"type": "course", "name": "A", "source": "test.csv"},
            ),
            Document(
                id="lecturer-b",
                content="Nama: B\nResearch Interest: natural language processing",
                metadata={"type": "lecturer", "name": "B", "source": "test.csv"},
            ),
            Document(
                id="staff-c",
                content="Jabatan: Dekan\nNama: C",
                metadata={"type": "staff", "name": "C", "source": "test.csv"},
            ),
        ]

    def test_document_has_id_content_and_metadata(self):
        document = self.documents[0]
        self.assertEqual(document.id, "course-a")
        self.assertIn("machine learning", document.content)
        self.assertEqual(document.metadata["type"], "course")

    def test_knowledge_repository_loads_documents_with_unique_ids(self):
        repository = KnowledgeRepository(PROJECT_ROOT / "data")
        documents = repository.get_documents()
        self.assertGreater(len(documents), 0)
        self.assertEqual(len(documents), len({document.id for document in documents}))
        self.assertTrue(all(document.metadata.get("type") for document in documents))

    def test_document_representation_exposes_semantic_type_and_staff_unit(self):
        repository = KnowledgeRepository(PROJECT_ROOT / "data")
        documents = repository.get_documents()
        staff = next(document for document in documents if document.id.startswith("staff-1-dekan"))
        course = next(document for document in documents if document.metadata["type"] == "course")
        lecturer = next(document for document in documents if document.metadata["type"] == "lecturer")
        academic = next(document for document in documents if document.metadata["type"] == "academic")
        ftmm = next(document for document in documents if document.metadata["type"] == "ftmm")
        self.assertIn("Jenis: Pejabat/Staf FTMM", staff.content)
        self.assertIn("Jabatan: Dekan FTMM", staff.content)
        self.assertIn("Fakultas Teknologi Maju dan Multidisiplin", staff.content)
        self.assertIn("Jenis: Mata Kuliah FTMM", course.content)
        self.assertIn("Jenis: Dosen FTMM", lecturer.content)
        self.assertIn("Jenis: Informasi Akademik FTMM", academic.content)
        self.assertIn("Jenis: Informasi Fakultas FTMM", ftmm.content)

    def test_embedding_service_encodes_query_and_documents(self):
        service = EmbeddingService("fake-model")
        service._model = FakeEmbeddingModel()
        document_vectors = service.encode_documents(["alpha", "beta"])
        query_vector = service.encode_query("alpha")
        self.assertEqual(document_vectors.ndim, 2)
        self.assertEqual(query_vector.ndim, 1)
        self.assertEqual(document_vectors.shape[1], query_vector.shape[0])

    def test_vector_index_supports_top_k(self):
        index = InMemoryVectorIndex()
        index.build(
            self.documents,
            np.asarray([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]]),
        )
        results = index.search(np.asarray([1.0, 0.0]), top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].document.id, "course-a")
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_vector_index_rejects_dimension_and_model_mismatch(self):
        index = InMemoryVectorIndex()
        index.build(
            self.documents,
            np.asarray([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]]),
            embedding_model="baseline-model",
        )
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            index.search(np.asarray([1.0, 0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            index.search(
                np.asarray([1.0, 0.0]), embedding_model="different-model"
            )

    def test_retriever_returns_ranked_results_and_indexes_once(self):
        embedding_service = FakeEmbeddingService()
        retriever = DenseRetriever(
            repository=FakeRepository(self.documents),
            embedding_service=embedding_service,
            vector_index=InMemoryVectorIndex(),
            top_k=3,
            min_score=None,
        )
        first_results = retriever.retrieve("machine learning", top_k=2)
        second_results = retriever.retrieve("machine learning", top_k=2)
        self.assertTrue(all(isinstance(item, RetrievedDocument) for item in first_results))
        self.assertTrue(all(item.score is not None for item in first_results))
        self.assertEqual(
            [item.score for item in first_results],
            sorted((item.score for item in first_results), reverse=True),
        )
        self.assertEqual(embedding_service.document_encode_calls, 1)
        self.assertEqual(len(second_results), 2)

    def test_context_builder_preserves_content_and_source_identifier(self):
        retrieved = [RetrievedDocument(self.documents[1], score=0.9)]
        context = ContextBuilder().build_context(retrieved)
        self.assertIn("[SOURCE 1]", context)
        self.assertIn("Chunk ID: lecturer-b", context)
        self.assertIn("Parent ID: lecturer-b", context)
        self.assertIn("Research Interest: natural language processing", context)

    def test_chat_service_orchestrates_retrieval_context_and_llm(self):
        retrieved = [RetrievedDocument(self.documents[0], score=0.95)]
        llm_service = CapturingLLMService()
        chat_service = ChatService(
            retriever=FakeRetriever(retrieved),
            context_builder=ContextBuilder(),
            llm_service=llm_service,
        )
        result = chat_service.answer("Apa mata kuliahnya?")
        self.assertEqual(result.answer, "jawaban-test")
        self.assertEqual(result.retrieved_documents, retrieved)
        self.assertIn("[SOURCE 1]", llm_service.context)

    def test_vector_index_metadata_filter_limits_document_type(self):
        index = InMemoryVectorIndex()
        index.build(
            self.documents,
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        )
        lecturer = index.search(
            np.asarray([1.0, 0.0]), top_k=5, metadata_filter={"type": "lecturer"}
        )
        course = index.search(
            np.asarray([1.0, 0.0]), top_k=5, metadata_filter={"type": "course"}
        )
        no_filter = index.search(np.asarray([1.0, 0.0]), top_k=5)
        self.assertTrue(lecturer and all(item.document.metadata["type"] == "lecturer" for item in lecturer))
        self.assertTrue(course and all(item.document.metadata["type"] == "course" for item in course))
        self.assertEqual({item.document.metadata["type"] for item in no_filter}, {"course", "lecturer", "staff"})

    def test_vector_index_metadata_filter_supports_and_across_fields(self):
        chunks = [
            DocumentChunk("l::profile", "l", "profile", 0, {"type": "lecturer", "chunk_role": "profile"}),
            DocumentChunk("l::research", "l", "machine learning", 1, {"type": "lecturer", "chunk_role": "research_interest"}),
            DocumentChunk("c::research", "c", "machine learning", 0, {"type": "course", "chunk_role": "research_interest"}),
        ]
        index = InMemoryVectorIndex()
        index.build(chunks, np.asarray([[0.5, 0.5], [1.0, 0.0], [0.9, 0.1]]))
        results = index.search(
            np.asarray([1.0, 0.0]), top_k=5,
            metadata_filter={"type": "lecturer", "chunk_role": "research_interest"},
        )
        self.assertEqual([item.document.id for item in results], ["l::research"])

    def test_retrieval_plan_falls_back_only_when_strict_filter_has_no_candidates(self):
        index = InMemoryVectorIndex()
        index.build(self.documents, np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]))
        retriever = DenseRetriever(
            FakeRepository([]), FakeEmbeddingService(), index, min_score=None
        )
        plan = RetrievalPlan(
            query="dosen bidang quantum",
            intent=QueryIntent.LECTURER,
            metadata_filter={"type": "lecturer", "chunk_role": "research_interest"},
            preferred_chunk_role="research_interest",
        )
        results = retriever.retrieve_plan(plan, top_k=3)
        self.assertTrue(results)
        self.assertTrue(all(item.metadata["type"] == "lecturer" for item in results))
        self.assertEqual(retriever.last_fallback_used, "type_only")

    def test_intent_feature_flag_false_preserves_unfiltered_behavior(self):
        class ExplodingRouter:
            def update_corpus_metadata(self, documents=None):
                pass

            def route(self, query):
                raise AssertionError("Router tidak boleh dipanggil.")

        index = InMemoryVectorIndex()
        index.build(self.documents, np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]))
        retriever = DenseRetriever(
            FakeRepository([]), FakeEmbeddingService(), index, min_score=None,
            intent_router=ExplodingRouter(), intent_routing_enabled=False,
        )
        results = retriever.retrieve("dosen machine learning", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(retriever.last_retrieval_plan.intent, QueryIntent.GENERAL)

    def test_retriever_diversifies_parent_chunks(self):
        chunks = [
            DocumentChunk("a::chunk-0000", "a", "alpha satu", 0, {"type": "course"}),
            DocumentChunk("a::chunk-0001", "a", "alpha dua", 1, {"type": "course"}),
            DocumentChunk("b::chunk-0000", "b", "beta", 0, {"type": "lecturer"}),
        ]
        index = InMemoryVectorIndex()
        index.build(chunks, np.asarray([[1.0, 0.0], [0.99, 0.01], [0.9, 0.1]]))
        retriever = DenseRetriever(
            repository=FakeRepository([]),
            embedding_service=FakeEmbeddingService(),
            vector_index=index,
            top_k=2,
            min_score=None,
            max_chunks_per_parent=1,
            candidate_multiplier=3,
        )
        results = retriever.retrieve("alpha", top_k=2)
        self.assertEqual([item.parent_document_id for item in results], ["a", "b"])
        self.assertTrue(all(item.chunk_id for item in results))

    def test_chunking_disabled_indexes_parent_documents(self):
        class DisabledChunker:
            enabled = False
            cache_configuration = {"chunking_enabled": False}

            def chunk_documents(self, documents):
                raise AssertionError("Chunker tidak boleh dipanggil ketika disabled.")

        embedding_service = FakeEmbeddingService()
        retriever = DenseRetriever(
            repository=FakeRepository(self.documents),
            embedding_service=embedding_service,
            vector_index=InMemoryVectorIndex(),
            min_score=None,
            document_chunker=DisabledChunker(),
        )
        retriever.initialize()
        self.assertEqual(retriever.vector_index.document_count, len(self.documents))


if __name__ == "__main__":
    unittest.main()
