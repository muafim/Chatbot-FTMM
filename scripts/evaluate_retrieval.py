"""Full-corpus dense retrieval evaluation; tidak menggunakan Flask atau LLM."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import psutil

from config import (
    get_bge_m3_settings,
    get_chunking_settings,
    get_embedding_settings,
    get_retrieval_settings,
)
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.knowledge_repository import KnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from services.embedding_cache import DocumentEmbeddingCache
from services.embedding_factory import create_embedding_service
from services.document_chunker import DocumentChunker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_ROOT / "data" / "retrieval_evaluation.json"
DEFAULT_CACHE_DIRECTORY = PROJECT_ROOT / "cache" / "embeddings"

MAIN_QUERIES = [
    "Apa itu FTMM?",
    "Siapa Dekan FTMM?",
    "Siapa Wakil Dekan I FTMM?",
    "Dosen yang meneliti natural language processing",
    "Dosen yang fokus machine learning",
    "Mata kuliah yang berkaitan dengan machine learning",
    "Mata kuliah semester 1",
    "Bagaimana cara mengajukan surat mahasiswa aktif?",
    "Bagaimana cara mengajukan surat rekomendasi?",
    "Apa visi FTMM?",
]


def evaluate(backend, cache_directory, cache_enabled=True):
    process = psutil.Process()
    rss_start = process.memory_info().rss
    repository = KnowledgeRepository(PROJECT_ROOT / "data")
    documents = repository.get_documents()
    rss_after_repository = process.memory_info().rss
    service = create_embedding_service(
        backend,
        baseline_settings=get_embedding_settings(),
        bge_m3_settings=get_bge_m3_settings(),
    )
    chunker = DocumentChunker(**get_chunking_settings())
    retrieval_settings = get_retrieval_settings()
    cache = DocumentEmbeddingCache(
        cache_directory,
        backend.replace("-", "_"),
        enabled=cache_enabled,
    )
    retriever = DenseRetriever(
        repository,
        service,
        InMemoryVectorIndex(),
        top_k=5,
        min_score=None,
        max_chunks_per_parent=retrieval_settings["max_chunks_per_parent"],
        candidate_multiplier=retrieval_settings["candidate_multiplier"],
        embedding_cache=cache,
        document_chunker=chunker,
    )

    initialize_started = time.perf_counter()
    retriever.initialize()
    initialize_seconds = time.perf_counter() - initialize_started
    rss_after_index = process.memory_info().rss

    evaluation_cases = [
        case
        for case in json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
        if case.get("relevant_document_ids")
    ]
    reciprocal_ranks = []
    recall_hits = {1: 0, 3: 0, 5: 0}
    cases = []
    query_latencies = []
    query_embedding_latencies = []
    vector_search_latencies = []
    post_filter_latencies = []
    rss_samples = []
    for case in evaluation_cases:
        started_at = time.perf_counter()
        results = retriever.retrieve(case["query"], top_k=5)
        query_latencies.append(time.perf_counter() - started_at)
        query_embedding_latencies.append(retriever.query_embedding_seconds[-1])
        vector_search_latencies.append(retriever.vector_search_seconds[-1])
        post_filter_latencies.append(retriever.post_filter_seconds[-1])
        rss_samples.append(process.memory_info().rss)
        relevant = set(case["relevant_document_ids"])
        relevant_sections = set(case.get("relevant_sections", []))
        rank = next(
            (
                position
                for position, item in enumerate(results, 1)
                if getattr(item.document, "parent_document_id", item.document.id)
                in relevant
                and (
                    not relevant_sections
                    or not hasattr(item.document, "parent_document_id")
                    or item.document.metadata.get("section") in relevant_sections
                )
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in recall_hits:
            recall_hits[k] += int(rank is not None and rank <= k)
        cases.append(
            {
                **case,
                "expected_rank": rank,
                "top_5": [serialize_result(item, position) for position, item in enumerate(results, 1)],
            }
        )

    main_results = {}
    for query in MAIN_QUERIES:
        results = retriever.retrieve(query, top_k=5)
        main_results[query] = [
            serialize_result(item, position) for position, item in enumerate(results, 1)
        ]

    filtered_analysis = {}
    filter_probes = (
        ("Dosen yang meneliti natural language processing", "lecturer"),
        ("Dosen yang fokus machine learning", "lecturer"),
        ("Mata kuliah yang berkaitan dengan machine learning", "course"),
        ("Siapa Dekan FTMM?", "staff"),
    )
    for query, document_type in filter_probes:
        filtered_analysis[f"{document_type}:{query}"] = [
            serialize_result(item, position)
            for position, item in enumerate(
                retriever.retrieve(
                    query, top_k=5, metadata_filter={"type": document_type}
                ),
                1,
            )
        ]

    case_count = len(evaluation_cases)
    category_metrics = calculate_category_metrics(cases)
    legacy_metrics = calculate_metrics(cases[:20])
    chunks = retriever.vector_index.documents
    return {
        "backend": backend,
        "model": service.model_name,
        "document_count": len(documents),
        "retrieval_unit_count": retriever.vector_index.document_count,
        "type_distribution": dict(Counter(document.metadata["type"] for document in documents)),
        "embedding_dimension": retriever.vector_index.embedding_dimension,
        "chunking": {
            "enabled": chunker.enabled,
            "configuration": chunker.cache_configuration,
            "statistics": chunker.statistics(chunks) if chunker.enabled else None,
            "type_distribution": dict(Counter(item.metadata["type"] for item in chunks)),
        },
        "cache": {
            "status": cache.last_status,
            "hit": retriever.cache_hit,
            "vector_path": str(cache.vector_path),
            "metadata_path": str(cache.metadata_path),
            "vector_size_bytes": cache.vector_path.stat().st_size if cache.vector_path.exists() else None,
        },
        "performance_seconds": {
            "repository_load": repository.load_seconds,
            "initialize_total": initialize_seconds,
            "cache_load": retriever.cache_load_seconds,
            "document_embedding": retriever.embedding_seconds,
            "index_build": retriever.index_build_seconds,
            "model_load": service.model_load_seconds,
            "first_query_total": query_latencies[0],
            "warm_query_average": sum(query_latencies[1:]) / len(query_latencies[1:]),
            "query_average": sum(query_latencies) / len(query_latencies),
            "first_query_embedding": query_embedding_latencies[0],
            "warm_query_embedding_average": sum(query_embedding_latencies[1:]) / len(query_embedding_latencies[1:]),
            "vector_search_average": sum(vector_search_latencies) / len(vector_search_latencies),
            "post_filter_average": sum(post_filter_latencies) / len(post_filter_latencies),
            "query_min": min(query_latencies),
            "query_max": max(query_latencies),
        },
        "memory_bytes": {
            "start": rss_start,
            "after_repository": rss_after_repository,
            "after_index": rss_after_index,
            "after_queries": process.memory_info().rss,
            "query_peak_observed": max(rss_samples),
        },
        "metrics": {
            "recall_at_1": recall_hits[1] / case_count,
            "recall_at_3": recall_hits[3] / case_count,
            "recall_at_5": recall_hits[5] / case_count,
            "mrr": sum(reciprocal_ranks) / case_count,
            "hit_rate_at_5": recall_hits[5] / case_count,
        },
        "legacy_20_metrics": legacy_metrics,
        "category_metrics": category_metrics,
        "sequential_query_count": len(evaluation_cases),
        "model_load_count": service.model_load_count,
        "cases": cases,
        "main_queries": main_results,
        "filtered_analysis": filtered_analysis,
    }


def serialize_result(item, rank):
    return {
        "rank": rank,
        "chunk_id": item.document.id,
        "parent_document_id": getattr(
            item.document, "parent_document_id", item.document.id
        ),
        "type": item.document.metadata.get("type"),
        "name": item.document.metadata.get("name"),
        "section": item.document.metadata.get("section"),
        "chunk_role": item.document.metadata.get("chunk_role"),
        "score": item.score,
    }


def calculate_metrics(cases):
    if not cases:
        return {}
    ranks = [case["expected_rank"] for case in cases]
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank if rank else 0.0 for rank in ranks) / len(ranks),
    }


def calculate_category_metrics(cases):
    categories = {}
    for category in sorted({case["category"] for case in cases}):
        categories[category] = calculate_metrics(
            [case for case in cases if case["category"] == category]
        )
    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["baseline", "bge-m3"], required=True)
    parser.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.backend, args.cache_directory, not args.no_cache)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
