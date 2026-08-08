import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from domain.models import Document
from services.bge_m3_embedding_service import BGEM3EmbeddingService
from services.embedding_cache import DocumentEmbeddingCache, corpus_fingerprint
from services.embedding_factory import create_embedding_service
from services.embedding_service import EmbeddingService
from config import get_embedding_backend


BASELINE_SETTINGS = {"model_name": "baseline-model", "device": "cpu"}
BGE_SETTINGS = {
    "model_name": "fake-bge",
    "device": "cpu",
    "precision": "bf16",
    "batch_size": 1,
    "max_length": 32,
}


class EmbeddingFactoryTests(unittest.TestCase):
    def test_application_default_backend_is_bge_m3(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(get_embedding_backend(), "bge-m3")

    def test_baseline_selection_does_not_load_model(self):
        service = create_embedding_service(
            "baseline", BASELINE_SETTINGS, BGE_SETTINGS
        )
        self.assertIsInstance(service, EmbeddingService)
        self.assertEqual(service.backend_name, "baseline")
        self.assertFalse(service.is_loaded)

    def test_bge_selection_does_not_load_baseline(self):
        service = create_embedding_service("bge-m3", BASELINE_SETTINGS, BGE_SETTINGS)
        self.assertIsInstance(service, BGEM3EmbeddingService)
        self.assertEqual(service.backend_name, "bge-m3")
        self.assertFalse(service.is_loaded)

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "baseline atau bge-m3"):
            create_embedding_service("unknown", BASELINE_SETTINGS, BGE_SETTINGS)


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cache = DocumentEmbeddingCache(
            self.temporary_directory.name, "test", enabled=True
        )
        self.documents = [
            Document("doc-1", "Jenis: Test\nNama: Satu", {"type": "test"}),
            Document("doc-2", "Jenis: Test\nNama: Dua", {"type": "test"}),
        ]
        self.vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.configuration = {"backend": "test", "max_length": 32}

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_no_cache_then_save_and_reuse(self):
        self.assertIsNone(
            self.cache.load(self.documents, "test-model", self.configuration)
        )
        self.cache.save(
            self.documents, self.vectors, "test-model", self.configuration
        )
        loaded = self.cache.load(
            self.documents, "test-model", self.configuration
        )
        np.testing.assert_array_equal(loaded, self.vectors)
        self.assertEqual(self.cache.last_status, "hit")

    def test_fingerprint_is_deterministic_and_content_sensitive(self):
        first = corpus_fingerprint(self.documents)
        self.assertEqual(first, corpus_fingerprint(list(self.documents)))
        changed = [self.documents[0], Document("doc-2", "berubah", {"type": "test"})]
        self.assertNotEqual(first, corpus_fingerprint(changed))

    def test_document_change_invalidates_cache(self):
        self.cache.save(
            self.documents, self.vectors, "test-model", self.configuration
        )
        changed = [self.documents[0], Document("doc-2", "berubah", {"type": "test"})]
        self.assertIsNone(self.cache.load(changed, "test-model", self.configuration))
        self.assertEqual(self.cache.last_status, "invalid_corpus_fingerprint")

    def test_model_change_invalidates_cache(self):
        self.cache.save(
            self.documents, self.vectors, "test-model", self.configuration
        )
        self.assertIsNone(
            self.cache.load(self.documents, "other-model", self.configuration)
        )
        self.assertEqual(self.cache.last_status, "invalid_embedding_model")

    def test_dimension_mismatch_is_rejected(self):
        self.cache.save(
            self.documents, self.vectors, "test-model", self.configuration
        )
        metadata = json.loads(self.cache.metadata_path.read_text(encoding="utf-8"))
        metadata["embedding_dimension"] = 3
        self.cache.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertIsNone(
            self.cache.load(self.documents, "test-model", self.configuration)
        )
        self.assertEqual(self.cache.last_status, "invalid_vectors")

    def test_corrupt_cache_is_recoverable(self):
        self.cache.directory.mkdir(parents=True, exist_ok=True)
        self.cache.vector_path.write_bytes(b"not-an-npz")
        self.cache.metadata_path.write_text("{invalid", encoding="utf-8")
        self.assertIsNone(
            self.cache.load(self.documents, "test-model", self.configuration)
        )
        self.assertEqual(self.cache.last_status, "corrupt")
        self.cache.save(
            self.documents, self.vectors, "test-model", self.configuration
        )
        self.assertIsNotNone(
            self.cache.load(self.documents, "test-model", self.configuration)
        )


if __name__ == "__main__":
    unittest.main()
