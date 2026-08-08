import json
import shutil
import tempfile
from pathlib import Path

from config import get_bge_m3_settings, get_chunking_settings
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.knowledge_repository import KnowledgeRepository
from repositories.pdf_knowledge_repository import PDFKnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from scripts.evaluate_pdf_retrieval import _write_fixture
from services.bge_m3_embedding_service import BGEM3EmbeddingService
from services.document_chunker import DocumentChunker
from services.embedding_cache import DocumentEmbeddingCache
from services.knowledge_registry import KnowledgeRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_sync(data_dir, pdf_dir, manifest, registry_path, cache_dir, embedding, rebuild=False):
    pdf_repository = PDFKnowledgeRepository(pdf_dir, manifest)
    registry = KnowledgeRegistry(registry_path)
    repository = KnowledgeRepository(data_dir, pdf_repository, registry)
    retriever = DenseRetriever(
        repository, embedding, InMemoryVectorIndex(),
        embedding_cache=DocumentEmbeddingCache(cache_dir, "bge_m3"),
        document_chunker=DocumentChunker(**get_chunking_settings()),
        min_score=None, top_k=5, candidate_multiplier=5,
        incremental_index_enabled=True, force_full_rebuild=rebuild,
    )
    retriever.initialize()
    return retriever, {
        "cache_mode": retriever.cache_mode,
        "sources": retriever.registry_update_plan.counts,
        "active": len(retriever._documents),
        "reused": retriever.reused_vectors,
        "embedded": retriever.embedded_vectors,
        "removed": retriever.removed_vectors,
        "reuse_rate": retriever.vector_reuse_rate,
        "embedding_seconds": retriever.embedding_seconds or 0.0,
        "index_seconds": retriever.index_build_seconds,
        "total_seconds": retriever.indexing_seconds,
    }


def evaluate():
    fixture = json.loads(
        (PROJECT_ROOT / "data" / "pdf_retrieval_evaluation.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pdf_dir = root / "pdfs"
        cache_dir = root / "cache" / "embeddings"
        full_cache_dir = root / "full" / "embeddings"
        pdf_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        for suffix in ("dense.npz", "dense_metadata.json"):
            source = PROJECT_ROOT / "cache" / "embeddings" / f"bge_m3_{suffix}"
            shutil.copy2(source, cache_dir / source.name)
        manifest = root / "cache" / "knowledge" / "manifest.json"
        registry = root / "cache" / "knowledge" / "registry.json"
        embedding = BGEM3EmbeddingService(**get_bge_m3_settings())

        _, baseline = _run_sync(PROJECT_ROOT / "data", pdf_dir, manifest, registry, cache_dir, embedding)
        pdf_path = pdf_dir / "pedoman-evaluasi.pdf"
        _write_fixture(pdf_path, fixture["pages"], fixture["fixture_title"])
        added_retriever, added = _run_sync(PROJECT_ROOT / "data", pdf_dir, manifest, registry, cache_dir, embedding)
        queries = [case["query"] for case in fixture["queries"][:3]]
        incremental_results = [
            [item.document.id for item in added_retriever.retrieve(query, top_k=5)]
            for query in queries
        ]

        modified_pages = list(fixture["pages"])
        modified_pages[1] += " Perubahan sintetis hanya pada halaman kedua."
        _write_fixture(pdf_path, modified_pages, fixture["fixture_title"])
        _, modified = _run_sync(PROJECT_ROOT / "data", pdf_dir, manifest, registry, cache_dir, embedding)

        pdf_path.unlink()
        _, deleted = _run_sync(PROJECT_ROOT / "data", pdf_dir, manifest, registry, cache_dir, embedding)

        _write_fixture(pdf_path, fixture["pages"], fixture["fixture_title"])
        _, restored = _run_sync(PROJECT_ROOT / "data", pdf_dir, manifest, registry, cache_dir, embedding)

        full_retriever, full = _run_sync(
            PROJECT_ROOT / "data", pdf_dir, root / "full" / "manifest.json",
            root / "full" / "registry.json", full_cache_dir, embedding, rebuild=True,
        )
        full_results = [
            [item.document.id for item in full_retriever.retrieve(query, top_k=5)]
            for query in queries
        ]
        return {
            "baseline": baseline,
            "incremental_add": added,
            "incremental_modify_page_2": modified,
            "incremental_delete": deleted,
            "incremental_restore": restored,
            "full_rebuild": full,
            "top_k_equivalent": incremental_results == full_results,
            "compared_queries": queries,
            "model_load_count": embedding.model_load_count,
            "openai_calls": 0,
        }


def main():
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
