import unittest
from pathlib import Path

import numpy as np

from domain.models import Document, RetrievedDocument
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
        self.assertIn("Document ID: lecturer-b", context)
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


if __name__ == "__main__":
    unittest.main()
