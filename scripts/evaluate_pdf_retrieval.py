import argparse
import json
import tempfile
import time
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from config import get_bge_m3_settings, get_chunking_settings
from domain.models import Document
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.pdf_knowledge_repository import PDFKnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from services.bge_m3_embedding_service import BGEM3EmbeddingService
from services.document_chunker import DocumentChunker
from services.embedding_cache import DocumentEmbeddingCache


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StaticRepository:
    def __init__(self, documents):
        self.documents = list(documents)

    def get_documents(self):
        return list(self.documents)


def _write_fixture(path, pages, title):
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_reference = writer._add_object(font)
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})
        })
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 10 Tf 54 720 Td ({escaped}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": title})
    with path.open("wb") as handle:
        writer.write(handle)


def _csv_competitors():
    return [
        Document(
            "eval-course-algoritma", "Jenis: Mata Kuliah FTMM\nNama Mata Kuliah: Algoritma\nKode: EVAL101\nProgram Studi: TSD\nSemester: 1\nSKS: 3\nPrasyarat: Tidak Ada\nCapaian Pembelajaran: Dasar algoritma\nDeskripsi: Struktur algoritma",
            {"type": "course", "source": "synthetic.csv", "name": "Algoritma", "code": "EVAL101", "credits": 3},
        ),
        Document(
            "eval-staff-dekan", "Jenis: Pejabat/Staf FTMM\nJabatan: Dekan FTMM\nNama: Prof. Evaluasi\nUnit: FTMM Universitas Airlangga",
            {"type": "staff", "source": "synthetic.csv", "name": "Prof. Evaluasi", "role": "Dekan"},
        ),
        Document(
            "eval-academic-cuti", "Jenis: Informasi Akademik FTMM\nJudul: Pengajuan Cuti\nLink: https://example.test/cuti\nProsedur pengajuan cuti dilakukan melalui portal akademik.",
            {"type": "academic", "source": "synthetic.csv", "name": "Pengajuan Cuti", "link": "https://example.test/cuti"},
        ),
    ]


def evaluate(fixture_path):
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pdf_dir = root / "pdfs"
        pdf_dir.mkdir()
        _write_fixture(pdf_dir / "pedoman-evaluasi.pdf", fixture["pages"], fixture["fixture_title"])
        pdf_repository = PDFKnowledgeRepository(pdf_dir, root / "manifest.json")
        pdf_documents = pdf_repository.load_documents()
        documents = _csv_competitors() + pdf_documents
        embedding = BGEM3EmbeddingService(**get_bge_m3_settings())
        chunker = DocumentChunker(**get_chunking_settings())
        chunk_started = time.perf_counter()
        preview_chunks = chunker.chunk_documents(documents)
        chunk_seconds = time.perf_counter() - chunk_started
        cache = DocumentEmbeddingCache(root / "embeddings", "pdf-eval")
        retriever = DenseRetriever(
            StaticRepository(documents), embedding, InMemoryVectorIndex(),
            document_chunker=chunker,
            embedding_cache=cache,
            top_k=5, min_score=None, max_chunks_per_parent=2, candidate_multiplier=5,
        )
        rows = []
        latencies = []
        reciprocal_ranks = []
        recall = {1: 0, 3: 0, 5: 0}
        for case in fixture["queries"]:
            started = time.perf_counter()
            results = retriever.retrieve(case["query"], top_k=5)
            latencies.append(time.perf_counter() - started)
            rank = None
            for index, item in enumerate(results, 1):
                metadata = item.metadata
                type_match = metadata.get("type") == case["expected_type"]
                page_match = case.get("expected_page") is None or metadata.get("page_start") == case["expected_page"]
                if type_match and page_match:
                    rank = index
                    break
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            for k in recall:
                recall[k] += int(rank is not None and rank <= k)
            rows.append({
                "query": case["query"], "expected_type": case["expected_type"],
                "expected_page": case.get("expected_page"), "rank": rank,
                "top_results": [{"id": item.document.id, "type": item.metadata.get("type"), "page": item.metadata.get("page_start"), "score": round(float(item.score), 6)} for item in results],
            })
        total = len(rows)
        warm_retriever = DenseRetriever(
            StaticRepository(documents), embedding, InMemoryVectorIndex(),
            document_chunker=chunker, embedding_cache=cache,
            top_k=5, min_score=None, max_chunks_per_parent=2, candidate_multiplier=5,
        )
        warm_started = time.perf_counter()
        warm_retriever.initialize()
        warm_seconds = time.perf_counter() - warm_started
        return {
            "model": embedding.model_name,
            "document_count": len(documents),
            "retrieval_chunk_count": retriever.vector_index.document_count,
            "preview_chunk_count": len(preview_chunks),
            "embedding_dimension": retriever.vector_index.embedding_dimension,
            "pdf_extraction_seconds": pdf_repository.report.elapsed_seconds,
            "chunking_seconds": chunk_seconds,
            "indexing_seconds": retriever.indexing_seconds,
            "document_embedding_seconds": retriever.embedding_seconds,
            "warm_initialization_seconds": warm_seconds,
            "warm_embedding_cache_hit": warm_retriever.cache_hit,
            "average_query_seconds": sum(latencies) / len(latencies),
            "recall_at_1": recall[1] / total,
            "recall_at_3": recall[3] / total,
            "recall_at_5": recall[5] / total,
            "mrr": sum(reciprocal_ranks) / total,
            "cases": rows,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluasi retrieval PDF sintetis tanpa OpenAI.")
    parser.add_argument("--fixture", default=str(PROJECT_ROOT / "data" / "pdf_retrieval_evaluation.json"))
    parser.add_argument("--output")
    args = parser.parse_args()
    report = evaluate(args.fixture)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
