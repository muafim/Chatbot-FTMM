import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from repositories.knowledge_repository import KnowledgeRepository
from repositories.pdf_knowledge_repository import PDFKnowledgeRepository
from services.context_builder import ContextBuilder
from services.document_chunker import DocumentChunker
from services.embedding_cache import corpus_fingerprint
from services.embedding_cache import DocumentEmbeddingCache
from services.pdf_document_loader import PDFDocumentLoader, PDFLoadError
from domain.models import RetrievedDocument
from indexes.in_memory_vector_index import InMemoryVectorIndex
from retrievers.dense_retriever import DenseRetriever


class WhitespaceTokenizer:
    def encode(self, text, **kwargs):
        return str(text).split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


def write_text_pdf(path, page_texts, title=None):
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_reference = writer._add_object(font)
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})
        })
        escaped = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if title:
        writer.add_metadata({"/Title": title})
    with Path(path).open("wb") as handle:
        writer.write(handle)


class PDFDocumentLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.loader = PDFDocumentLoader(max_file_size_mb=5)

    def tearDown(self):
        self.temp.cleanup()

    def test_extracts_pages_with_metadata_and_skips_blank_page(self):
        path = self.root / "Panduan_Akademik.pdf"
        write_text_pdf(path, ["  Registrasi   dilakukan daring.  ", "", "Kontak: akademik FTMM"], "Panduan Akademik")
        result = self.loader.load(path, allowed_root=self.root)
        self.assertEqual([page.page_number for page in result.pages], [1, 3])
        self.assertEqual(result.pages[0].text, "Registrasi dilakukan daring.")
        self.assertEqual(result.document_title, "Panduan Akademik")
        documents = result.to_documents()
        self.assertEqual(documents[0].metadata["page_start"], 1)
        self.assertEqual(documents[1].metadata["page_end"], 3)
        self.assertEqual(documents[0].metadata["type"], "pdf")
        self.assertIn(result.content_hash[:12], result.source_document_id)
        self.assertTrue(result.warnings)

    def test_controlled_failures(self):
        with self.assertRaisesRegex(PDFLoadError, "tidak ditemukan"):
            self.loader.load(self.root / "missing.pdf")
        text_path = self.root / "notes.txt"
        text_path.write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(PDFLoadError, "berekstensi"):
            self.loader.load(text_path)
        corrupt = self.root / "corrupt.pdf"
        corrupt.write_bytes(b"not a pdf")
        with self.assertRaises(PDFLoadError) as raised:
            self.loader.load(corrupt)
        self.assertEqual(raised.exception.status, "corrupt")

    def test_textless_and_encrypted_are_controlled(self):
        blank = self.root / "scan.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with blank.open("wb") as handle:
            writer.write(handle)
        with self.assertRaises(PDFLoadError) as raised:
            self.loader.load(blank)
        self.assertEqual(raised.exception.status, "no_extractable_text")

        plain = self.root / "plain.pdf"
        write_text_pdf(plain, ["rahasia"])
        reader = PdfReader(plain)
        encrypted_writer = PdfWriter()
        encrypted_writer.append_pages_from_reader(reader)
        encrypted_writer.encrypt("secret")
        encrypted = self.root / "encrypted.pdf"
        with encrypted.open("wb") as handle:
            encrypted_writer.write(handle)
        with self.assertRaises(PDFLoadError) as raised:
            self.loader.load(encrypted)
        self.assertEqual(raised.exception.status, "encrypted")


class PDFKnowledgeRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir()
        self.manifest = self.root / "cache" / "manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def repository(self):
        return PDFKnowledgeRepository(self.pdf_dir, self.manifest, max_file_size_mb=5)

    def test_duplicate_manifest_warm_cache_change_and_delete(self):
        original = self.pdf_dir / "a.pdf"
        duplicate = self.pdf_dir / "b.pdf"
        write_text_pdf(original, ["Fakta unik FTMM tahap delapan"])
        duplicate.write_bytes(original.read_bytes())

        first_repo = self.repository()
        first_docs = first_repo.load_documents()
        first_fingerprint = corpus_fingerprint(first_docs)
        self.assertEqual(len(first_docs), 1)
        self.assertEqual(first_repo.report.duplicate_files, 1)
        self.assertTrue(self.manifest.exists())
        first_repo.record_chunk_counts(
            DocumentChunker(max_tokens=100, overlap_tokens=10, tokenizer=WhitespaceTokenizer()).chunk_documents(first_docs)
        )
        manifest_payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        loaded_entry = next(item for item in manifest_payload["files"] if item["status"] == "loaded")
        self.assertEqual(loaded_entry["chunk_count"], 1)

        warm_repo = self.repository()
        warm_docs = warm_repo.load_documents()
        self.assertEqual(warm_repo.report.extraction_cache_hits, 1)
        self.assertEqual(corpus_fingerprint(warm_docs), first_fingerprint)

        duplicate.unlink()
        write_text_pdf(original, ["Fakta telah berubah pada nama file yang sama"])
        changed_repo = self.repository()
        changed_docs = changed_repo.load_documents()
        self.assertNotEqual(corpus_fingerprint(changed_docs), first_fingerprint)
        self.assertNotEqual(changed_docs[0].id, first_docs[0].id)

        original.unlink()
        deleted_repo = self.repository()
        self.assertEqual(deleted_repo.load_documents(), [])
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["files"], [])

    def test_corrupt_manifest_rebuilds_and_missing_directory_is_safe(self):
        write_text_pdf(self.pdf_dir / "source.pdf", ["Teks sumber"])
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text("{broken", encoding="utf-8")
        self.assertEqual(len(self.repository().load_documents()), 1)

        missing = PDFKnowledgeRepository(self.root / "none", self.manifest)
        self.assertEqual(missing.load_documents(), [])
        self.assertTrue(missing.warnings)

    def test_dry_run_does_not_write_manifest(self):
        write_text_pdf(self.pdf_dir / "source.pdf", ["Teks sumber"])
        self.assertEqual(len(self.repository().load_documents(dry_run=True)), 1)
        self.assertFalse(self.manifest.exists())

    def test_ingestion_needs_no_openai_key_or_client(self):
        write_text_pdf(self.pdf_dir / "offline.pdf", ["Ingestion sepenuhnya lokal"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with patch("openai.OpenAI", side_effect=AssertionError("OpenAI tidak boleh dipanggil")):
                documents = self.repository().load_documents(dry_run=True)
        self.assertEqual(len(documents), 1)


class PDFPipelineTests(unittest.TestCase):
    def test_pdf_chunk_and_context_keep_page_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "Pedoman.pdf"
            write_text_pdf(path, ["Ketentuan registrasi mahasiswa FTMM berlaku daring."], "Pedoman Registrasi")
            document = PDFDocumentLoader().load(path, allowed_root=root).to_documents()[0]
            chunker = DocumentChunker(max_tokens=12, overlap_tokens=2, tokenizer=WhitespaceTokenizer())
            chunks = chunker.chunk_document(document)
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(item.parent_document_id == document.id for item in chunks))
            self.assertTrue(all(item.metadata["page_start"] == 1 for item in chunks))
            self.assertTrue(all(item.metadata["section"] == "Halaman 1" for item in chunks))
            context = ContextBuilder().build_grounded_context([RetrievedDocument(chunks[0], 0.9)])
            citation = context.sources["S1"]
            self.assertEqual(citation.title, "Pedoman Registrasi")
            self.assertEqual(citation.metadata["page_start"], 1)
            self.assertIn("Page start: 1", context.text)

    def test_csv_baseline_count_unchanged_without_pdf(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            pdf_repo = PDFKnowledgeRepository(Path(temporary) / "missing", Path(temporary) / "manifest.json")
            repository = KnowledgeRepository(project_root / "data", pdf_repository=pdf_repo)
            self.assertEqual(len(repository.get_documents()), 391)

    def test_pdf_uses_same_retriever_cache_and_strict_filter_excludes_it(self):
        class Repository:
            def __init__(self, documents):
                self.documents = documents

            def get_documents(self):
                return list(self.documents)

        class KeywordEmbedding:
            model_name = "test-keyword-embedding"
            cache_configuration = {"test": True}

            @staticmethod
            def vector(text):
                lowered = str(text).casefold()
                values = np.asarray([
                    float("akselerasi" in lowered),
                    float("algoritma" in lowered),
                    float("dekan" in lowered),
                ])
                return values if values.any() else np.asarray([0.01, 0.01, 0.01])

            def encode_documents(self, texts):
                return np.vstack([self.vector(text) for text in texts])

            def encode_query(self, query):
                return self.vector(query)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "Kebijakan.pdf"
            write_text_pdf(pdf_path, ["Program akselerasi akademik khusus mahasiswa FTMM."], "Kebijakan Akademik")
            pdf_document = PDFDocumentLoader().load(pdf_path, allowed_root=root).to_documents()[0]
            from domain.models import Document
            csv_documents = [
                Document("course-x", "Nama Mata Kuliah: Algoritma", {"type": "course", "name": "Algoritma"}),
                Document("staff-x", "Jabatan: Dekan FTMM", {"type": "staff", "name": "Dekan"}),
            ]
            cache = DocumentEmbeddingCache(root / "embeddings", "combined", enabled=True)
            chunker = DocumentChunker(max_tokens=100, overlap_tokens=10, tokenizer=WhitespaceTokenizer())
            retriever = DenseRetriever(
                Repository(csv_documents + [pdf_document]), KeywordEmbedding(),
                InMemoryVectorIndex(), document_chunker=chunker,
                embedding_cache=cache, min_score=None, top_k=3,
            )
            general = retriever.retrieve("program akselerasi", top_k=1)
            self.assertEqual(general[0].metadata["type"], "pdf")
            self.assertEqual(retriever.vector_index.document_count, 3)

            warm = DenseRetriever(
                Repository(csv_documents + [pdf_document]), KeywordEmbedding(),
                InMemoryVectorIndex(), document_chunker=chunker,
                embedding_cache=cache, min_score=None, top_k=3,
            )
            strict = warm.retrieve("algoritma", top_k=3, metadata_filter={"type": "course"})
            self.assertTrue(warm.cache_hit)
            self.assertTrue(strict)
            self.assertTrue(all(item.metadata["type"] == "course" for item in strict))


if __name__ == "__main__":
    unittest.main()
