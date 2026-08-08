import argparse
import json
from pathlib import Path

from config import get_chunking_settings, get_pdf_knowledge_settings
from repositories.knowledge_repository import KnowledgeRepository
from repositories.pdf_knowledge_repository import PDFKnowledgeRepository
from services.document_chunker import DocumentChunker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_report(dry_run=False):
    settings = get_pdf_knowledge_settings(PROJECT_ROOT)
    pdf_repository = PDFKnowledgeRepository(**settings)
    pdf_documents = pdf_repository.load_documents(dry_run=dry_run)
    csv_repository = KnowledgeRepository(PROJECT_ROOT / "data")
    csv_documents = csv_repository.get_documents()
    chunker = DocumentChunker(**get_chunking_settings())
    chunk_error = None
    try:
        pdf_chunks = chunker.chunk_documents(pdf_documents)
        combined_chunks = chunker.chunk_documents(csv_documents + pdf_documents)
    except Exception as exc:
        pdf_chunks = combined_chunks = None
        chunk_error = (
            f"Chunk audit tidak tersedia ({type(exc).__name__}); "
            "set BGE_M3_LOCAL_PATH ke snapshot tokenizer lokal untuk mode offline."
        )
    report = pdf_repository.report
    return {
        "mode": "dry-run" if dry_run else "manifest-update",
        "source_directory": str(settings["directory"]),
        "manifest_path": str(settings["manifest_path"]),
        "pdf_enabled": settings["enabled"],
        "discovered_pdf_files": report.discovered_files,
        "loaded_pdf_files": report.loaded_files,
        "duplicate_pdf_files": report.duplicate_files,
        "failed_pdf_files": report.failed_files,
        "extracted_pdf_pages": report.extracted_pages,
        "pdf_parent_documents": len(pdf_documents),
        "pdf_chunks": len(pdf_chunks) if pdf_chunks is not None else None,
        "csv_parent_documents": len(csv_documents),
        "combined_parent_documents": len(csv_documents) + len(pdf_documents),
        "combined_chunks": len(combined_chunks) if combined_chunks is not None else None,
        "extraction_cache_hits": report.extraction_cache_hits,
        "ingestion_seconds": round(report.elapsed_seconds, 4),
        "warnings": pdf_repository.warnings,
        "errors": csv_repository.errors + pdf_repository.errors + ([chunk_error] if chunk_error else []),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit ingestion PDF lokal tanpa OpenAI.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pindai dan ekstrak tanpa menulis manifest.",
    )
    args = parser.parse_args()
    print(json.dumps(build_report(dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
