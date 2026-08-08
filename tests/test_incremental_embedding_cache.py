import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from domain.models import Document
from indexes.in_memory_vector_index import InMemoryVectorIndex
from services.embedding_cache import DocumentEmbeddingCache, INCREMENTAL_DENSE_FORMAT


class CountingEmbedder:
    model_name = "deterministic-model"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def encode_documents(self, texts):
        values = list(texts)
        self.calls.append(values)
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        rows = []
        for text in values:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([digest[0] / 255, digest[1] / 255, digest[2] / 255])
        return np.asarray(rows, dtype=np.float32)


def documents(count, prefix="doc"):
    return [Document(f"{prefix}-{i}", f"content-{i}", {"type": "test"}) for i in range(count)]


class IncrementalDenseCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = DocumentEmbeddingCache(Path(self.temp.name), "test")
        self.config = {"precision": "bf16", "max_length": 256, "chunking_version": "v1"}

    def tearDown(self):
        self.temp.cleanup()

    def sync(self, docs, embedder=None, config=None, rebuild=False):
        return self.cache.sync_dense(docs, embedder or CountingEmbedder(), config or self.config, force_rebuild=rebuild)

    def test_add_only_embeds_new_chunks(self):
        base = documents(10)
        self.sync(base)
        embedder = CountingEmbedder()
        result = self.sync(base + documents(2, "new"), embedder)
        self.assertEqual(result.mode, "incremental")
        self.assertEqual(result.reused_vectors, 10)
        self.assertEqual(result.embedded_vectors, 2)
        self.assertEqual(len(embedder.calls[0]), 2)

    def test_modify_one_delete_two_and_rename_reuse(self):
        base = documents(10)
        self.sync(base)
        changed = list(base)
        changed[5] = Document("doc-5", "changed-content", {"type": "test"})
        modified = self.sync(changed)
        self.assertEqual((modified.reused_vectors, modified.embedded_vectors, modified.removed_vectors), (9, 1, 1))

        deleted = self.sync(changed[:8], CountingEmbedder())
        self.assertEqual((deleted.reused_vectors, deleted.embedded_vectors, deleted.removed_vectors), (8, 0, 2))

        renamed = list(changed[:8])
        renamed[0] = Document("renamed-id", renamed[0].content, {"type": "test"})
        rename_result = self.sync(renamed, CountingEmbedder())
        self.assertEqual(rename_result.embedded_vectors, 0)
        self.assertEqual(rename_result.reused_vectors, 8)

    def test_deleted_vectors_remain_reusable_when_source_is_restored(self):
        base = documents(5)
        self.sync(base)
        deleted = self.sync(base[:2], CountingEmbedder())
        self.assertEqual((deleted.embedded_vectors, deleted.removed_vectors), (0, 3))
        embedder = CountingEmbedder(fail=True)
        restored = self.sync(base, embedder)
        self.assertEqual(restored.embedded_vectors, 0)
        self.assertEqual(restored.reused_vectors, 5)
        self.assertEqual(embedder.calls, [])

    def test_model_or_chunk_config_change_forces_full_rebuild(self):
        base = documents(4)
        self.sync(base)
        different_model = CountingEmbedder()
        different_model.model_name = "other-model"
        model_result = self.sync(base, different_model)
        self.assertEqual(model_result.embedded_vectors, 4)
        config_result = self.sync(base, CountingEmbedder(), {**self.config, "chunking_version": "v2"})
        self.assertEqual(config_result.embedded_vectors, 4)

    def test_corrupt_cache_falls_back_to_full_rebuild(self):
        base = documents(3)
        self.sync(base)
        self.cache.vector_path.write_bytes(b"broken")
        result = self.sync(base)
        self.assertEqual(result.mode, "full_rebuild")
        self.assertEqual(result.embedded_vectors, 3)

    def test_embedding_failure_preserves_old_cache_and_registry_independent(self):
        base = documents(3)
        self.sync(base)
        vector_before = self.cache.vector_path.read_bytes()
        metadata_before = self.cache.metadata_path.read_bytes()
        with self.assertRaises(RuntimeError):
            self.sync(base + [Document("new", "new", {})], CountingEmbedder(fail=True))
        self.assertEqual(self.cache.vector_path.read_bytes(), vector_before)
        self.assertEqual(self.cache.metadata_path.read_bytes(), metadata_before)

    def test_second_atomic_replace_failure_rolls_back_cache_pair(self):
        base = documents(3)
        self.sync(base)
        vector_before = self.cache.vector_path.read_bytes()
        metadata_before = self.cache.metadata_path.read_bytes()
        real_replace = __import__("os").replace
        calls = 0

        def fail_second_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated metadata commit failure")
            return real_replace(source, destination)

        with patch("services.embedding_cache.os.replace", side_effect=fail_second_replace):
            with self.assertRaises(OSError):
                self.sync(base + [Document("new", "new", {})])
        self.assertEqual(self.cache.vector_path.read_bytes(), vector_before)
        self.assertEqual(self.cache.metadata_path.read_bytes(), metadata_before)

    def test_dry_plan_does_not_mutate_or_load_embedder(self):
        base = documents(3)
        self.sync(base)
        before = (self.cache.vector_path.read_bytes(), self.cache.metadata_path.read_bytes())
        plan = self.cache.plan_dense_update(base + [Document("new", "new", {})], "deterministic-model", self.config)
        self.assertEqual(plan["chunks_to_embed"], 1)
        self.assertEqual((self.cache.vector_path.read_bytes(), self.cache.metadata_path.read_bytes()), before)

    def test_incremental_and_full_vectors_and_search_are_equivalent(self):
        base = documents(4)
        self.sync(base)
        final_docs = base + [Document("new", "unique-new-content", {"type": "test"})]
        incremental = self.sync(final_docs).vectors
        with tempfile.TemporaryDirectory() as other:
            fresh_cache = DocumentEmbeddingCache(Path(other), "fresh")
            full = fresh_cache.sync_dense(final_docs, CountingEmbedder(), self.config, force_rebuild=True).vectors
        np.testing.assert_array_equal(incremental, full)
        first, second = InMemoryVectorIndex(), InMemoryVectorIndex()
        first.build(final_docs, incremental)
        second.build(final_docs, full)
        query = CountingEmbedder().encode_documents(["unique-new-content"])[0]
        self.assertEqual(
            [item.document.id for item in first.search(query, top_k=5)],
            [item.document.id for item in second.search(query, top_k=5)],
        )

    def test_legacy_exact_cache_migrates_without_embedding(self):
        base = documents(3)
        vectors = CountingEmbedder().encode_documents(item.content for item in base)
        self.cache.save(base, vectors, "deterministic-model", self.config)
        embedder = CountingEmbedder(fail=True)
        result = self.sync(base, embedder)
        self.assertTrue(result.migrated_legacy)
        self.assertEqual(result.embedded_vectors, 0)
        self.assertEqual(embedder.calls, [])
        import json
        metadata = json.loads(self.cache.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["cache_format_version"], INCREMENTAL_DENSE_FORMAT)

    def test_openai_is_never_used(self):
        with patch("openai.OpenAI", side_effect=AssertionError("OpenAI tidak boleh dipanggil")):
            self.sync(documents(2))


if __name__ == "__main__":
    unittest.main()
