"""Bandingkan dense retrieval tanpa routing dan intent-aware routing dalam satu proses."""

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
from domain.models import QueryIntent, RetrievalPlan
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.knowledge_repository import KnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from services.document_chunker import DocumentChunker
from services.embedding_cache import DocumentEmbeddingCache
from services.embedding_factory import create_embedding_service
from services.query_intent_router import QueryIntentRouter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_ROOT / "data" / "retrieval_evaluation.json"
DEFAULT_CACHE_DIRECTORY = PROJECT_ROOT / "cache" / "embeddings"
ML_LECTURERS = {
    "Ratih Ardiati Ningrum": "lecturer-199501262020013201",
    "Rizki Putra Prastio": "lecturer-199106272020073101",
    "Asif Ali Zamzami": "lecturer-199207222022103101",
    "Rezi Delfianti": "lecturer-199609262023103201",
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


def relevant_rank(case, results):
    relevant = set(case.get("relevant_document_ids", []))
    relevant_sections = set(case.get("relevant_sections", []))
    if not relevant:
        return None
    return next(
        (
            rank
            for rank, item in enumerate(results, 1)
            if getattr(item.document, "parent_document_id", item.document.id)
            in relevant
            and (
                not relevant_sections
                or item.document.metadata.get("section") in relevant_sections
            )
        ),
        None,
    )


def metrics(cases):
    if not cases:
        return {}
    ranks = [case["rank"] for case in cases]
    return {
        "query_count": len(cases),
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank if rank else 0.0 for rank in ranks) / len(ranks),
    }


def run_mode(retriever, cases, routing_enabled, process):
    retriever.intent_routing_enabled = routing_enabled
    offsets = {
        "query": len(retriever.query_seconds),
        "embedding": len(retriever.query_embedding_seconds),
        "search": len(retriever.vector_search_seconds),
        "router": len(retriever.router_seconds),
    }
    evaluated = []
    routing_records = []
    memory_samples = []
    failures = []
    for case in cases:
        try:
            results = retriever.retrieve(case["query"], top_k=5)
            plan = retriever.last_retrieval_plan
            rank = relevant_rank(case, results)
            routing_records.append(
                {
                    "query": case["query"],
                    "expected": case["expected_intent"],
                    "actual": plan.intent.value,
                    "confidence": plan.confidence.value,
                    "metadata_filter": plan.metadata_filter,
                    "preferred_chunk_role": plan.preferred_chunk_role,
                    "reason": plan.reason,
                    "fallback": retriever.last_fallback_used,
                }
            )
            if case.get("relevant_document_ids"):
                evaluated.append(
                    {
                        "category": case["category"],
                        "query": case["query"],
                        "rank": rank,
                        "top_5": [
                            serialize_result(item, position)
                            for position, item in enumerate(results, 1)
                        ],
                    }
                )
            memory_samples.append(process.memory_info().rss)
        except Exception as exc:
            failures.append({"query": case["query"], "error": type(exc).__name__})

    query_times = retriever.query_seconds[offsets["query"] :]
    embedding_times = retriever.query_embedding_seconds[offsets["embedding"] :]
    search_times = retriever.vector_search_seconds[offsets["search"] :]
    router_times = retriever.router_seconds[offsets["router"] :]
    by_category = {
        category: metrics([item for item in evaluated if item["category"] == category])
        for category in sorted({item["category"] for item in evaluated})
    }
    correct = [item for item in routing_records if item["expected"] == item["actual"]]
    per_intent = {}
    for intent in sorted({item["expected"] for item in routing_records}):
        selected = [item for item in routing_records if item["expected"] == intent]
        per_intent[intent] = {
            "correct": sum(item["actual"] == intent for item in selected),
            "total": len(selected),
            "accuracy": sum(item["actual"] == intent for item in selected) / len(selected),
        }
    confusion = Counter(
        (item["expected"], item["actual"])
        for item in routing_records
        if item["expected"] != item["actual"]
    )
    return {
        "routing_enabled": routing_enabled,
        "metrics": metrics(evaluated),
        "category_metrics": by_category,
        "intent_accuracy": {
            "correct": len(correct),
            "total": len(routing_records),
            "accuracy": len(correct) / len(routing_records),
            "per_intent": per_intent,
            "confusion": [
                {"expected": key[0], "actual": key[1], "count": value}
                for key, value in sorted(confusion.items())
            ],
        },
        "performance_seconds": {
            "router_average": sum(router_times) / len(router_times),
            "embedding_average": sum(embedding_times) / len(embedding_times),
            "vector_search_average": sum(search_times) / len(search_times),
            "retrieval_average": sum(query_times) / len(query_times),
            "retrieval_min": min(query_times),
            "retrieval_max": max(query_times),
        },
        "sequential": {
            "queries": len(cases),
            "success": len(cases) - len(failures),
            "failure": len(failures),
            "failures": failures,
            "memory_min_bytes": min(memory_samples),
            "memory_max_bytes": max(memory_samples),
        },
        "cases": evaluated,
        "routing_records": routing_records,
    }


def probe(retriever, query, routing_enabled, top_k=5):
    retriever.intent_routing_enabled = routing_enabled
    results = retriever.retrieve(query, top_k=top_k)
    return {
        "plan": {
            "intent": retriever.last_retrieval_plan.intent.value,
            "confidence": retriever.last_retrieval_plan.confidence.value,
            "metadata_filter": retriever.last_retrieval_plan.metadata_filter,
            "preferred_chunk_role": retriever.last_retrieval_plan.preferred_chunk_role,
            "reason": retriever.last_retrieval_plan.reason,
        },
        "fallback": retriever.last_fallback_used,
        "top": [serialize_result(item, rank) for rank, item in enumerate(results, 1)],
    }


def evaluate(cache_directory, cache_enabled=True):
    process = psutil.Process()
    memory_start = process.memory_info().rss
    repository = KnowledgeRepository(PROJECT_ROOT / "data")
    documents = repository.get_documents()
    service = create_embedding_service(
        "bge-m3",
        baseline_settings=get_embedding_settings(),
        bge_m3_settings=get_bge_m3_settings(),
    )
    chunker = DocumentChunker(**get_chunking_settings())
    settings = get_retrieval_settings()
    router = QueryIntentRouter()
    retriever = DenseRetriever(
        repository,
        service,
        InMemoryVectorIndex(),
        top_k=5,
        min_score=None,
        max_chunks_per_parent=settings["max_chunks_per_parent"],
        candidate_multiplier=settings["candidate_multiplier"],
        embedding_cache=DocumentEmbeddingCache(
            cache_directory, "bge_m3", enabled=cache_enabled
        ),
        document_chunker=chunker,
        intent_router=router,
        intent_routing_enabled=False,
    )
    initialized_at = time.perf_counter()
    retriever.initialize()
    initialize_seconds = time.perf_counter() - initialized_at
    memory_after_index = process.memory_info().rss
    cases = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))

    stage3 = run_mode(retriever, cases, False, process)
    stage4 = run_mode(retriever, cases, True, process)

    probes = {}
    for key, query, top_k in (
        ("lecturer_nlp", "Dosen yang meneliti natural language processing", 5),
        ("lecturer_machine_learning", "Dosen yang fokus machine learning", 10),
        ("course_nlp", "Mata kuliah yang mempelajari natural language processing", 5),
        ("profile_maryamah", "profil Maryamah", 5),
        ("ftmm_visi", "Apa visi FTMM?", 3),
        ("ftmm_misi", "Apa misi FTMM?", 3),
    ):
        probes[key] = {
            "stage3": probe(retriever, query, False, top_k),
            "stage4": probe(retriever, query, True, top_k),
        }

    strict_missing_role = RetrievalPlan(
        query="dosen dengan bidang quantum",
        intent=QueryIntent.LECTURER,
        metadata_filter={"type": "lecturer", "chunk_role": "not_available"},
        preferred_chunk_role="not_available",
    )
    fallback_results = retriever.retrieve_plan(strict_missing_role, top_k=5)
    chunks = retriever.vector_index.documents
    lecturer_chunks = [item for item in chunks if item.metadata.get("type") == "lecturer"]
    research_chunks = [
        item for item in lecturer_chunks
        if item.metadata.get("chunk_role") == "research_interest"
    ]
    ml_probe = probes["lecturer_machine_learning"]
    for mode in ("stage3", "stage4"):
        rankings = {
            item["parent_document_id"]: item["rank"]
            for item in ml_probe[mode]["top"]
        }
        ml_probe[mode]["ground_truth_ranks"] = {
            name: rankings.get(parent_id) for name, parent_id in ML_LECTURERS.items()
        }

    return {
        "architecture": "query -> deterministic intent -> retrieval plan -> metadata filter -> dense BGE-M3",
        "parent_document_count": len(documents),
        "retrieval_unit_count": len(chunks),
        "embedding_matrix_shape": [len(chunks), retriever.vector_index.embedding_dimension],
        "chunking": {
            "configuration": chunker.cache_configuration,
            "type_distribution": dict(Counter(item.metadata.get("type") for item in chunks)),
            "lecturer_chunk_count": len(lecturer_chunks),
            "lecturer_role_distribution": dict(Counter(item.metadata.get("chunk_role") for item in lecturer_chunks)),
            "research_interest_chunks": len(research_chunks),
            "research_interest_sources": dict(Counter(item.metadata.get("research_interest_source") for item in research_chunks)),
        },
        "cache": {
            "hit": retriever.cache_hit,
            "status": retriever.embedding_cache.last_status,
            "vector_size_bytes": retriever.embedding_cache.vector_path.stat().st_size,
        },
        "indexing": {
            "initialize_seconds": initialize_seconds,
            "document_embedding_seconds": retriever.embedding_seconds,
            "cache_load_seconds": retriever.cache_load_seconds,
            "model_load_seconds": service.model_load_seconds,
            "model_load_count": service.model_load_count,
        },
        "memory_bytes": {
            "start": memory_start,
            "after_index": memory_after_index,
            "after_all_queries": process.memory_info().rss,
        },
        "stage3_no_routing_on_stage4_corpus": stage3,
        "stage4_routed": stage4,
        "probes": probes,
        "fallback_probe": {
            "requested_filter": strict_missing_role.metadata_filter,
            "fallback_used": retriever.last_fallback_used,
            "result_types": [item.metadata.get("type") for item in fallback_results],
            "count": len(fallback_results),
        },
        "final_model_load_count": service.model_load_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.cache_directory, not args.no_cache)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
