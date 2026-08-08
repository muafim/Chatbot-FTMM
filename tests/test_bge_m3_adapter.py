import os
import unittest
from pathlib import Path

import numpy as np

from services.bge_m3_embedding_service import BGEM3EmbeddingService


class FakeBGEM3Model:
    def encode_queries(self, queries, **kwargs):
        return {"dense_vecs": np.ones((len(queries), 8), dtype=np.float32)}

    def encode_corpus(self, documents, **kwargs):
        return {"dense_vecs": np.ones((len(documents), 8), dtype=np.float32)}


class BGEM3AdapterUnitTests(unittest.TestCase):
    def setUp(self):
        self.service = BGEM3EmbeddingService(
            model_name="fake-bge-m3",
            device="cpu",
            batch_size=2,
            max_length=32,
        )
        self.service._model = FakeBGEM3Model()

    def test_service_initialization_is_dense_cpu_fp32(self):
        self.assertEqual(self.service.device, "cpu")
        self.assertEqual(self.service.precision, "fp32")
        self.assertFalse(self.service.use_fp16)
        self.assertFalse(self.service.use_bf16)
        self.assertTrue(self.service.is_loaded)

    def test_auto_device_falls_back_to_cpu_without_cuda(self):
        service = BGEM3EmbeddingService(
            model_name="fake-bge-m3", device="auto", precision="auto"
        )
        self.assertEqual(service.device, "cpu")
        self.assertEqual(service.precision, "fp32")

    def test_query_and_document_interfaces_return_valid_vectors(self):
        query_vector = self.service.encode_query("pertanyaan")
        document_vectors = self.service.encode_documents(["dokumen satu", "dokumen dua"])
        self.assertEqual(query_vector.shape, (8,))
        self.assertEqual(document_vectors.shape, (2, 8))
        self.assertTrue(np.isfinite(query_vector).all())
        self.assertTrue(np.isfinite(document_vectors).all())
        self.assertEqual(self.service.embedding_dimension, 8)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.encode_query(" ")
        with self.assertRaises(ValueError):
            self.service.encode_documents([])


@unittest.skipUnless(
    os.getenv("RUN_BGE_M3_INTEGRATION", "false").lower() == "true",
    "Set RUN_BGE_M3_INTEGRATION=true untuk menjalankan model BGE-M3 penuh.",
)
class BGEM3AdapterIntegrationTests(unittest.TestCase):
    def test_real_model_dense_vectors_are_1024_and_finite(self):
        from config import get_bge_m3_settings

        service = BGEM3EmbeddingService(**get_bge_m3_settings())
        query_vector = service.encode_query("siapa dekan FTMM")
        document_vectors = service.encode_documents(
            ["Dekan FTMM memimpin Fakultas Teknologi Maju dan Multidisiplin."]
        )
        self.assertEqual(query_vector.shape, (1024,))
        self.assertEqual(document_vectors.shape, (1, 1024))
        self.assertTrue(np.isfinite(query_vector).all())
        self.assertTrue(np.isfinite(document_vectors).all())
        self.assertFalse(np.allclose(query_vector, 0))
        self.assertFalse(np.allclose(document_vectors, 0))


@unittest.skipUnless(
    os.getenv("RUN_BGE_M3_FULL_CORPUS", "false").lower() == "true",
    "Set RUN_BGE_M3_FULL_CORPUS=true untuk integration test full corpus.",
)
class BGEM3FullCorpusIntegrationTests(unittest.TestCase):
    def test_full_corpus_cache_and_retrieval_are_1024_dimensions(self):
        from config import get_bge_m3_settings
        from indexes.in_memory_vector_index import InMemoryVectorIndex
        from repositories.knowledge_repository import KnowledgeRepository
        from retrievers.dense_retriever import DenseRetriever
        from services.embedding_cache import DocumentEmbeddingCache

        project_root = Path(__file__).resolve().parents[1]
        repository = KnowledgeRepository(project_root / "data")
        retriever = DenseRetriever(
            repository=repository,
            embedding_service=BGEM3EmbeddingService(**get_bge_m3_settings()),
            vector_index=InMemoryVectorIndex(),
            top_k=5,
            min_score=None,
            embedding_cache=DocumentEmbeddingCache(
                project_root / "cache" / "embeddings", "bge_m3", enabled=True
            ),
        )
        results = retriever.retrieve("Siapa Dekan FTMM?", top_k=5)
        self.assertEqual(len(repository.get_documents()), 391)
        self.assertEqual(retriever.vector_index.embedding_dimension, 1024)
        self.assertEqual(
            results[0].document.id,
            "staff-1-dekan-prof-dr-dwi-setyawan-s-si-m-si-apt",
        )
