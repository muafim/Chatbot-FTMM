import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.knowledge_registry import (
    KnowledgeChangeDetector,
    KnowledgeRegistry,
    SourceSnapshot,
    SourceStatus,
)


def snapshot(name="pedoman.pdf", digest="a" * 64, status=SourceStatus.ACTIVE):
    stem = Path(name).stem
    return SourceSnapshot(
        logical_source_id=f"pdf-{stem}",
        source_document_id=f"pdf-{stem}-{digest[:12]}",
        source_type="pdf",
        source_file=name,
        title=stem,
        content_hash=digest,
        version_id=f"sha256-{digest}",
        page_count=3,
        parent_ids=("p1", "p2", "p3"),
        chunk_ids=("c1", "c2", "c3"),
        status=status,
    )


class KnowledgeRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "registry.json"
        self.registry = KnowledgeRegistry(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_empty_and_corrupt_registry_are_rebuildable(self):
        self.assertEqual(self.registry.status()["source_versions"], 0)
        self.path.write_text("{broken", encoding="utf-8")
        status = self.registry.status()
        self.assertTrue(status["corrupt"])
        self.assertEqual(status["source_versions"], 0)
        self.registry.commit((snapshot(),))
        self.assertFalse(self.registry.status()["corrupt"])

    def test_added_unchanged_modified_deleted_and_restored_history(self):
        first = snapshot()
        plan = self.registry.plan((first,))
        self.assertEqual(plan.counts["added"], 1)
        self.registry.commit((first,), plan)

        self.assertEqual(self.registry.plan((first,)).counts["unchanged"], 1)
        changed = snapshot(digest="b" * 64)
        modified = self.registry.plan((changed,))
        self.assertEqual(modified.counts["modified"], 1)
        self.registry.commit((changed,), modified)
        records = self.registry.load()["sources"]
        self.assertEqual({item["status"] for item in records}, {"ACTIVE", "SUPERSEDED"})

        removed = self.registry.plan(())
        self.assertEqual(removed.counts["removed"], 1)
        self.registry.commit((), removed)
        self.assertEqual(self.registry.status()["status_counts"]["REMOVED"], 1)

        restored = self.registry.plan((changed,))
        self.assertEqual(restored.counts["restored"], 1)
        self.registry.commit((changed,), restored)
        active = [item for item in self.registry.load()["sources"] if item["status"] == "ACTIVE"]
        self.assertEqual(len(active), 1)

    def test_rename_and_duplicate_content_do_not_create_duplicate_version(self):
        original = snapshot("pedoman.pdf")
        self.registry.commit((original,))
        renamed = snapshot("nama-baru.pdf")
        plan = self.registry.plan((renamed,))
        self.assertEqual(plan.counts["renamed"], 1)
        self.assertEqual(plan.counts["added"], 0)
        self.registry.commit((renamed,), plan)
        self.assertEqual(len(self.registry.load()["sources"]), 1)

        duplicate = snapshot("copy.pdf")
        duplicate_plan = KnowledgeChangeDetector().detect([], (original, duplicate))
        self.assertEqual(duplicate_plan.counts["added"], 1)
        self.assertEqual(duplicate_plan.counts["duplicate_aliases"], 1)

    def test_superseded_version_is_classified_as_restored(self):
        original = snapshot(digest="a" * 64)
        changed = snapshot(digest="b" * 64)
        self.registry.commit((original,))
        self.registry.commit((changed,), self.registry.plan((changed,)))
        self.registry.commit((), self.registry.plan(()))

        restored = self.registry.plan((original,))
        self.assertEqual(restored.counts["restored"], 1)
        self.assertEqual(restored.counts["added"], 0)

    def test_invalid_source_is_recorded_without_removing_healthy_source(self):
        healthy = snapshot()
        self.registry.commit((healthy,))
        invalid = snapshot("rusak.pdf", "c" * 64, SourceStatus.INVALID)
        plan = self.registry.plan((healthy, invalid))
        self.assertEqual(plan.counts["invalid"], 1)
        self.assertEqual(plan.counts["unchanged"], 1)
        self.registry.commit((healthy, invalid), plan)
        status = self.registry.status()["status_counts"]
        self.assertEqual(status["ACTIVE"], 1)
        self.assertEqual(status["INVALID"], 1)

    def test_atomic_failure_preserves_old_registry(self):
        self.registry.commit((snapshot(),))
        before = self.path.read_bytes()
        with patch("services.knowledge_registry.os.replace", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                self.registry.commit((snapshot(digest="d" * 64),))
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
