import unittest

from domain.models import Document, DocumentChunk
from services.document_chunker import DocumentChunker


class WhitespaceTokenizer:
    def encode(self, text, **kwargs):
        return str(text).split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


class DocumentChunkerTests(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker(
            max_tokens=30,
            overlap_tokens=5,
            tokenizer=WhitespaceTokenizer(),
        )

    def test_short_document_stays_one_chunk_with_stable_identity(self):
        document = Document(
            "staff-1",
            "Jenis: Pejabat/Staf FTMM\nJabatan: Dekan FTMM\nNama: Test",
            {"type": "staff", "source": "staff.csv", "name": "Test", "role": "Dekan"},
        )
        first = self.chunker.chunk_document(document)
        second = self.chunker.chunk_document(document)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertIsInstance(first[0], DocumentChunk)
        self.assertEqual(first[0].id, "staff-1::chunk-0000")
        self.assertEqual(first[0].parent_document_id, "staff-1")
        self.assertEqual(first[0].chunk_index, 0)
        self.assertEqual(first[0].metadata["position"], "Dekan")

    def test_long_content_respects_max_tokens_and_overlap(self):
        text = " ".join(f"token-{index}" for index in range(70))
        document = Document("generic-1", text, {"type": "generic", "source": "x"})
        chunks = self.chunker.chunk_document(document)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["token_count"] <= 30 for chunk in chunks))
        first_tokens = chunks[0].content.split()
        second_tokens = chunks[1].content.split()
        self.assertEqual(first_tokens[-5:], second_tokens[:5])
        self.assertTrue(all(chunk.content.strip() for chunk in chunks))

    def test_long_course_repeats_course_identity(self):
        description = " ".join(["machine-learning"] * 80)
        content = (
            "Jenis: Mata Kuliah FTMM\nNama Mata Kuliah: Machine Learning\nKode: SIC306\n"
            "Program Studi: TSD\nSemester: 6\nSKS: 3\nPrasyarat: Tidak Ada\n"
            f"Capaian Pembelajaran: {description}\nDeskripsi: {description}"
        )
        document = Document(
            "course-sic306", content, {"type": "course", "source": "course.csv", "name": "Machine Learning", "code": "SIC306", "credits": 3}
        )
        chunks = self.chunker.chunk_document(document)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("Nama Mata Kuliah: Machine Learning" in chunk.content for chunk in chunks))
        self.assertTrue(all(chunk.metadata["course_code"] == "SIC306" for chunk in chunks))

    def test_long_lecturer_keeps_identity_with_research_chunks(self):
        interests = " ".join(["machine-learning"] * 60)
        content = (
            "Jenis: Dosen FTMM\nNama: Dr. Data\nProgram Studi: TSD\n"
            f"Research Interest: {interests}\nPendidikan: S3\nDeskripsi: Peneliti"
        )
        document = Document(
            "lecturer-data", content, {"type": "lecturer", "source": "lecturer.csv", "name": "Dr. Data", "program": "TSD", "research_interest": interests}
        )
        chunks = self.chunker.chunk_document(document)
        research_chunks = [
            chunk for chunk in chunks
            if chunk.metadata["chunk_role"] == "research_interest"
        ]
        self.assertEqual(len(research_chunks), 1)
        self.assertEqual(research_chunks[0].id, "lecturer-data::research-interest")
        self.assertIn("Nama: Dr. Data", research_chunks[0].content)
        self.assertIn("Research Interest", research_chunks[0].content)
        self.assertTrue(all(chunk.metadata["lecturer_name"] == "Dr. Data" for chunk in chunks))

    def test_description_research_interest_is_preserved_without_invention(self):
        document = Document(
            "lecturer-mary",
            "Jenis: Dosen FTMM\nNama: Mary\nProgram Studi: TSD\n"
            "Research Interest: Tidak Ada\nPendidikan: S3\n"
            "Deskripsi: Her research interests are in NLP and information retrieval.\n"
            "Email: mary@example.test\nPortofolio: Tidak Ada",
            {
                "type": "lecturer", "source": "lecturer.csv", "name": "Mary",
                "program": "TSD", "research_interest": "Tidak Ada",
            },
        )
        chunks = self.chunker.chunk_document(document)
        research = [item for item in chunks if item.metadata["chunk_role"] == "research_interest"]
        self.assertEqual(len(research), 1)
        self.assertIn("NLP and information retrieval", research[0].content)
        self.assertEqual(research[0].metadata["research_interest_source"], "Description")

    def test_missing_research_interest_does_not_create_empty_chunk(self):
        document = Document(
            "lecturer-none",
            "Jenis: Dosen FTMM\nNama: No RI\nProgram Studi: TSD\n"
            "Research Interest: Tidak Ada\nPendidikan: S2\n"
            "Deskripsi: Lecturer profile only.\nEmail: x@example.test\nPortofolio: -",
            {"type": "lecturer", "source": "lecturer.csv", "name": "No RI", "research_interest": "Tidak Ada"},
        )
        chunks = self.chunker.chunk_document(document)
        self.assertFalse(any(item.metadata["chunk_role"] == "research_interest" for item in chunks))

    def test_ftmm_splits_real_sections_and_preserves_parent(self):
        document = Document(
            "ftmm-overview-1",
            "Jenis: Informasi Fakultas FTMM\nNama Fakultas: FTMM\nVisi: Menjadi unggul\nMisi: Pendidikan dan penelitian\nSejarah Fakultas: Dibentuk pada tahun tertentu",
            {"type": "ftmm", "source": "ftmm.csv", "name": "FTMM"},
        )
        chunks = self.chunker.chunk_document(document)
        self.assertEqual([chunk.metadata["section"] for chunk in chunks], ["Visi", "Misi", "Sejarah Fakultas"])
        self.assertTrue(all(chunk.parent_document_id == document.id for chunk in chunks))
        self.assertTrue(all("Topik:" in chunk.content for chunk in chunks))

    def test_exact_duplicate_chunks_are_removed(self):
        documents = [
            Document("a", "konten identik", {"type": "generic"}),
            Document("b", "konten identik", {"type": "generic"}),
        ]
        chunks = self.chunker.chunk_documents(documents)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(self.chunker.duplicate_chunks_removed, 1)


if __name__ == "__main__":
    unittest.main()
