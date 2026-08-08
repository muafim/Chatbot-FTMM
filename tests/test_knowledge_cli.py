import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import manage_knowledge_index


class KnowledgeCLITests(unittest.TestCase):
    def test_status_is_read_only_and_does_not_load_bge_or_openai(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data" / "pdfs").mkdir(parents=True)
            registry = root / "cache" / "knowledge" / "registry.json"
            cache_dir = root / "cache" / "embeddings"
            env = {
                "KNOWLEDGE_REGISTRY_PATH": str(registry),
                "PDF_SOURCE_DIRECTORY": str(root / "data" / "pdfs"),
                "PDF_MANIFEST_PATH": str(root / "cache" / "knowledge" / "manifest.json"),
                "EMBEDDING_CACHE_DIRECTORY": str(cache_dir),
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("services.bge_m3_embedding_service.BGEM3EmbeddingService.load_model", side_effect=AssertionError("BGE tidak boleh load")):
                    with patch("openai.OpenAI", side_effect=AssertionError("OpenAI tidak boleh load")):
                        report = manage_knowledge_index.status_report()
            self.assertEqual(report["mode"], "status")
            self.assertFalse(report["embedding_model_loaded"])
            self.assertFalse(report["mutated"])
            self.assertFalse(registry.exists())
            self.assertFalse(cache_dir.exists())


if __name__ == "__main__":
    unittest.main()
