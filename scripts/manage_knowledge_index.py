import argparse
import json
import time
from pathlib import Path

from config import (
    get_bge_m3_settings,
    get_chunking_settings,
    get_embedding_allow_fallback,
    get_embedding_backend,
    get_embedding_cache_settings,
    get_embedding_settings,
    get_knowledge_index_settings,
    get_pdf_knowledge_settings,
    get_retrieval_settings,
)
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.knowledge_repository import KnowledgeRepository
from repositories.pdf_knowledge_repository import PDFKnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from services.document_chunker import DocumentChunker
from services.embedding_cache import DocumentEmbeddingCache
from services.embedding_factory import create_embedding_service
from services.knowledge_registry import KnowledgeRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _components(dry_run=False, force_rebuild=False):
    pdf_repository = PDFKnowledgeRepository(**get_pdf_knowledge_settings(PROJECT_ROOT))
    registry_settings = get_knowledge_index_settings(PROJECT_ROOT)
    registry = KnowledgeRegistry(registry_settings["registry_path"])
    chunker = DocumentChunker(**get_chunking_settings())
    if dry_run:
        started_at = time.perf_counter()
        csv_repository = KnowledgeRepository(PROJECT_ROOT / "data")
        repository_started_at = time.perf_counter()
        parents = csv_repository.get_documents() + pdf_repository.load_documents(dry_run=True)
        repository_load_seconds = time.perf_counter() - repository_started_at
        chunking_started_at = time.perf_counter()
        chunks = chunker.chunk_documents(parents)
        chunking_seconds = time.perf_counter() - chunking_started_at
        snapshots = registry.build_snapshots(
            parents, chunks, invalid_entries=pdf_repository.invalid_source_snapshots()
        )
        plan = registry.plan(snapshots)
        backend = get_embedding_backend()
        cache_settings = get_embedding_cache_settings(PROJECT_ROOT)
        cache = DocumentEmbeddingCache(cache_settings["directory"], backend.replace("-", "_"), cache_settings["enabled"])
        embedding_settings = get_bge_m3_settings() if backend == "bge-m3" else get_embedding_settings()
        model_name = embedding_settings["model_name"]
        configuration = {
            "backend": backend,
            "max_length": embedding_settings.get("max_length"),
            "precision": embedding_settings.get("precision"),
            "normalized": True if backend == "bge-m3" else None,
            **chunker.cache_configuration,
        }
        cache_plan = cache.plan_dense_update(chunks, model_name, configuration, force_rebuild=force_rebuild)
        return {
            "mode": "dry-run",
            "registry_plan": plan.counts,
            **cache_plan,
            "parents_active": len(parents),
            "pdf_report": pdf_repository.report.__dict__,
            "repository_load_seconds": repository_load_seconds,
            "extraction_seconds": pdf_repository.report.elapsed_seconds,
            "chunking_seconds": chunking_seconds,
            "total_seconds": time.perf_counter() - started_at,
            "mutated": False,
        }

    repository = KnowledgeRepository(
        PROJECT_ROOT / "data", pdf_repository=pdf_repository, knowledge_registry=registry
    )
    backend = get_embedding_backend()
    embedding = create_embedding_service(
        backend, get_embedding_settings(), get_bge_m3_settings(), get_embedding_allow_fallback()
    )
    cache_settings = get_embedding_cache_settings(PROJECT_ROOT)
    cache = DocumentEmbeddingCache(cache_settings["directory"], backend.replace("-", "_"), cache_settings["enabled"])
    retrieval = get_retrieval_settings()
    if retrieval["retrieval_mode"] != "dense":
        raise RuntimeError("Knowledge sync Tahap 8B hanya mendukung production dense mode.")
    retriever = DenseRetriever(
        repository, embedding, InMemoryVectorIndex(),
        embedding_cache=cache, document_chunker=chunker,
        incremental_index_enabled=True, force_full_rebuild=force_rebuild,
        **retrieval,
    )
    retriever.initialize()
    source_counts = retriever.registry_update_plan.counts if retriever.registry_update_plan else {}
    registry_status = registry.status()
    return {
        "mode": "rebuild" if force_rebuild else "sync",
        "cache_mode": retriever.cache_mode,
        "sources": source_counts,
        "sources_total": registry_status["source_versions"],
        "sources_active": registry_status["status_counts"].get("ACTIVE", 0),
        "chunks_active": len(retriever._documents),
        "chunks_reused": retriever.reused_vectors,
        "chunks_embedded": retriever.embedded_vectors,
        "chunks_removed": retriever.removed_vectors,
        "reuse_rate": retriever.vector_reuse_rate,
        "embedding_seconds": retriever.embedding_seconds or 0.0,
        "repository_load_seconds": retriever.repository_load_seconds,
        "extraction_seconds": retriever.extraction_seconds,
        "chunking_seconds": retriever.chunking_seconds,
        "index_build_seconds": retriever.index_build_seconds,
        "total_seconds": retriever.indexing_seconds,
        "model_load_count": getattr(embedding, "model_load_count", 0),
        "registry": registry_status,
    }


def status_report():
    registry_settings = get_knowledge_index_settings(PROJECT_ROOT)
    registry = KnowledgeRegistry(registry_settings["registry_path"])
    pdf_settings = get_pdf_knowledge_settings(PROJECT_ROOT)
    pdf_files = PDFKnowledgeRepository(
        **pdf_settings
    ).loader.discover(pdf_settings["directory"])
    cache_settings = get_embedding_cache_settings(PROJECT_ROOT)
    backend = get_embedding_backend()
    metadata_path = cache_settings["directory"] / f"{backend.replace('-', '_')}_dense_metadata.json"
    cache_metadata = {}
    if metadata_path.exists():
        try:
            cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache_metadata = {"status": "corrupt"}
    return {
        "mode": "status",
        "registry": registry.status(),
        "pdf_files_discovered": len(pdf_files),
        "cache_format": cache_metadata.get("cache_format_version", "legacy-or-missing"),
        "cached_chunks": cache_metadata.get("document_count", 0),
        "embedding_model_loaded": False,
        "mutated": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Kelola registry dan dense knowledge index secara offline.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--sync", action="store_true")
    modes.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.status:
        report = status_report()
    elif args.dry_run:
        report = _components(dry_run=True)
    else:
        report = _components(force_rebuild=args.rebuild)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
