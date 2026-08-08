"""Evaluasi dense, native BGE-M3 sparse, dan RRF hybrid tanpa LLM."""

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
from indexes.in_memory_sparse_index import InMemorySparseIndex
from indexes.in_memory_vector_index import InMemoryVectorIndex
from repositories.knowledge_repository import KnowledgeRepository
from retrievers.dense_retriever import DenseRetriever
from services.document_chunker import DocumentChunker
from services.embedding_cache import DocumentEmbeddingCache, corpus_fingerprint
from services.embedding_factory import create_embedding_service
from services.query_intent_router import QueryIntentRouter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_EVALUATION = PROJECT_ROOT / "data" / "retrieval_evaluation.json"
EXACT_EVALUATION = PROJECT_ROOT / "data" / "lecturer_exact_term_evaluation.json"
DEFAULT_CACHE_DIRECTORY = PROJECT_ROOT / "cache" / "embeddings"
MODES = ("dense", "sparse", "hybrid")
ML_LECTURERS = {
    "Ratih Ardiati Ningrum": "lecturer-199501262020013201",
    "Rizki Putra Prastio": "lecturer-199106272020073101",
    "Asif Ali Zamzami": "lecturer-199207222022103101",
    "Rezi Delfianti": "lecturer-199609262023103201",
}
CONTROL_QUERIES = {
    "lecturer_nlp": "Dosen yang meneliti natural language processing",
    "course_nlp": "Mata kuliah yang mempelajari natural language processing",
    "course_machine_learning": "Mata kuliah tentang machine learning",
    "ftmm_visi": "Apa visi FTMM?",
    "ftmm_misi": "Apa misi FTMM?",
    "ftmm_sejarah": "Bagaimana sejarah FTMM?",
    "ftmm_maskot": "Apa maskot FTMM?",
    "staff_dekan": "Siapa Dekan FTMM?",
    "staff_wakil_dekan": "Siapa Wakil Dekan I?",
    "academic_skma": "Bagaimana mengajukan surat mahasiswa aktif?",
    "academic_rekomendasi": "Bagaimana mengajukan surat rekomendasi?",
}


def serialize(item, rank):
    return {
        "rank": rank,
        "chunk_id": item.document.id,
        "parent_document_id": getattr(item.document, "parent_document_id", item.document.id),
        "type": item.document.metadata.get("type"),
        "name": item.document.metadata.get("name"),
        "section": item.document.metadata.get("section"),
        "chunk_role": item.document.metadata.get("chunk_role"),
        "score": item.score,
        "dense_rank": item.dense_rank,
        "dense_score": item.dense_score,
        "sparse_rank": item.sparse_rank,
        "sparse_score": item.sparse_score,
        "fusion_score": item.fusion_score,
        "final_rank": item.final_rank,
    }


def relevant_rank(case, results):
    relevant = set(case.get("relevant_document_ids", []))
    sections = set(case.get("relevant_sections", []))
    if not relevant:
        return None
    return next(
        (
            rank
            for rank, item in enumerate(results, 1)
            if getattr(item.document, "parent_document_id", item.document.id) in relevant
            and (not sections or item.document.metadata.get("section") in sections)
        ),
        None,
    )


def calculate_metrics(records):
    if not records:
        return {}
    ranks = [record["rank"] for record in records]
    return {
        "query_count": len(records),
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank if rank else 0 for rank in ranks) / len(ranks),
    }


def run_fixture(retriever, cases, mode, process):
    retriever.retrieval_mode = mode
    offsets = {
        "total": len(retriever.query_seconds),
        "encode": len(retriever.query_embedding_seconds),
        "dense": len(retriever.vector_search_seconds),
        "sparse": len(retriever.sparse_search_seconds),
        "fusion": len(retriever.fusion_seconds),
        "diversify": len(retriever.post_filter_seconds),
        "router": len(retriever.router_seconds),
    }
    records = []
    intent_records = []
    failures = []
    memory_samples = []
    for case in cases:
        try:
            results = retriever.retrieve(case["query"], top_k=5)
            plan = retriever.last_retrieval_plan
            intent_records.append(
                {
                    "query": case["query"],
                    "expected": case.get("expected_intent"),
                    "actual": plan.intent.value,
                }
            )
            if case.get("relevant_document_ids"):
                records.append(
                    {
                        "category": case.get("category", "lecturer"),
                        "query": case["query"],
                        "rank": relevant_rank(case, results),
                        "top_5": [serialize(item, rank) for rank, item in enumerate(results, 1)],
                    }
                )
            memory_samples.append(process.memory_info().rss)
        except Exception as exc:
            failures.append({"query": case["query"], "error": type(exc).__name__})

    timings = {
        key: getattr(retriever, attribute)[offsets[key] :]
        for key, attribute in (
            ("total", "query_seconds"),
            ("encode", "query_embedding_seconds"),
            ("dense", "vector_search_seconds"),
            ("sparse", "sparse_search_seconds"),
            ("fusion", "fusion_seconds"),
            ("diversify", "post_filter_seconds"),
            ("router", "router_seconds"),
        )
    }
    intent_labeled = [item for item in intent_records if item["expected"]]
    intent_correct = sum(item["expected"] == item["actual"] for item in intent_labeled)
    return {
        "mode": mode,
        "metrics": calculate_metrics(records),
        "common_39_metrics": calculate_metrics(records[:39]),
        "category_metrics": {
            category: calculate_metrics(
                [record for record in records if record["category"] == category]
            )
            for category in sorted({record["category"] for record in records})
        },
        "intent_accuracy": {
            "correct": intent_correct,
            "total": len(intent_labeled),
            "accuracy": intent_correct / len(intent_labeled) if intent_labeled else None,
            "errors": [item for item in intent_labeled if item["expected"] != item["actual"]],
        },
        "performance_seconds": {
            key: sum(values) / len(values) if values else 0.0
            for key, values in timings.items()
        },
        "success": len(cases) - len(failures),
        "failure": len(failures),
        "failures": failures,
        "memory_min_bytes": min(memory_samples) if memory_samples else None,
        "memory_max_bytes": max(memory_samples) if memory_samples else None,
        "records": records,
    }


def probe(retriever, mode, query, top_k=5):
    retriever.retrieval_mode = mode
    results = retriever.retrieve(query, top_k=top_k)
    return {
        "intent": retriever.last_retrieval_plan.intent.value,
        "metadata_filter": retriever.last_used_filter,
        "fallback": retriever.last_fallback_used,
        "query_sparse_nonzero": (
            retriever.last_query_sparse_nonzero
        ),
        "dense_candidate_count": retriever.last_dense_candidate_count,
        "sparse_candidate_count": retriever.last_sparse_candidate_count,
        "top": [serialize(item, rank) for rank, item in enumerate(results, 1)],
    }


def sequential_hybrid_test(retriever, cases, process):
    retriever.retrieval_mode = "hybrid"
    queries = [case["query"] for case in cases]
    queries = (queries + queries[:8])[:60]
    failures = []
    memory = []
    intent_pairs = []
    for index, query in enumerate(queries):
        try:
            results = retriever.retrieve(query, top_k=5)
            if not results:
                failures.append({"index": index, "query": query, "error": "empty"})
            intent_pairs.append((query, retriever.last_retrieval_plan.intent.value))
            memory.append(process.memory_info().rss)
        except Exception as exc:
            failures.append({"index": index, "query": query, "error": type(exc).__name__})
    return {
        "queries": len(queries),
        "success": len(queries) - len(failures),
        "failure": len(failures),
        "failures": failures,
        "deterministic_repeated_intents": all(
            first_intent == next(
                intent for repeated_query, intent in reversed(intent_pairs)
                if repeated_query == query
            )
            for query, first_intent in intent_pairs
        ),
        "memory_min_bytes": min(memory),
        "memory_max_bytes": max(memory),
    }


def evaluate(cache_directory, selected_mode="all", cache_enabled=True):
    process = psutil.Process()
    memory_start = process.memory_info().rss
    main_cases = json.loads(MAIN_EVALUATION.read_text(encoding="utf-8"))
    exact_cases = json.loads(EXACT_EVALUATION.read_text(encoding="utf-8"))
    modes = MODES if selected_mode == "all" else (selected_mode,)
    initialization_mode = "hybrid" if selected_mode == "all" else selected_mode

    repository = KnowledgeRepository(PROJECT_ROOT / "data")
    parents = repository.get_documents()
    service = create_embedding_service(
        "bge-m3",
        baseline_settings=get_embedding_settings(),
        bge_m3_settings=get_bge_m3_settings(),
    )
    chunker = DocumentChunker(**get_chunking_settings())
    settings = get_retrieval_settings()
    cache_key = "bge_m3" if initialization_mode == "dense" else "bge_m3_hybrid"
    cache = DocumentEmbeddingCache(cache_directory, cache_key, enabled=cache_enabled)
    retriever = DenseRetriever(
        repository,
        service,
        InMemoryVectorIndex(),
        sparse_index=InMemorySparseIndex(),
        retrieval_mode=initialization_mode,
        top_k=5,
        min_score=None,
        max_chunks_per_parent=settings["max_chunks_per_parent"],
        candidate_multiplier=settings["candidate_multiplier"],
        hybrid_rrf_k=settings["hybrid_rrf_k"],
        hybrid_candidate_k=settings["hybrid_candidate_k"],
        hybrid_dense_weight=settings["hybrid_dense_weight"],
        hybrid_sparse_weight=settings["hybrid_sparse_weight"],
        embedding_cache=cache,
        document_chunker=chunker,
        intent_router=QueryIntentRouter(),
        intent_routing_enabled=True,
    )
    initialized_at = time.perf_counter()
    retriever.initialize()
    initialize_seconds = time.perf_counter() - initialized_at
    memory_after_index = process.memory_info().rss

    overall = {mode: run_fixture(retriever, main_cases, mode, process) for mode in modes}
    exact = {mode: run_fixture(retriever, exact_cases, mode, process) for mode in modes}

    machine_learning = {}
    for mode in modes:
        result = probe(retriever, mode, "Dosen yang fokus machine learning", top_k=42)
        parent_ranks = {item["parent_document_id"]: item["rank"] for item in result["top"]}
        result["ground_truth_ranks"] = {
            name: parent_ranks.get(parent_id) for name, parent_id in ML_LECTURERS.items()
        }
        machine_learning[mode] = result

    controls = {
        key: {mode: probe(retriever, mode, query, top_k=5) for mode in modes}
        for key, query in CONTROL_QUERIES.items()
    }
    ambiguous = {
        query: {mode: probe(retriever, mode, query, top_k=5) for mode in modes}
        for query in ("machine learning", "NLP", "AI")
    }
    sparse_smoke = {}
    if "sparse" in modes:
        for term in (
            "machine learning",
            "natural language processing",
            "Applied Machine Learning",
            "Machine Learning and Estimation Theory",
            "image processing",
            "data mining",
        ):
            sparse_smoke[term] = probe(retriever, "sparse", term, top_k=5)

    sequential = (
        sequential_hybrid_test(retriever, main_cases, process)
        if "hybrid" in modes else None
    )
    chunks = retriever._documents
    metadata = (
        json.loads(cache.metadata_path.read_text(encoding="utf-8"))
        if cache.metadata_path.exists() else {}
    )
    return {
        "modes": list(modes),
        "model": service.model_name,
        "parent_count": len(parents),
        "chunk_count": len(chunks),
        "corpus_fingerprint": corpus_fingerprint(chunks),
        "embedding_dimension": retriever.vector_index.embedding_dimension,
        "sparse_index": {
            "document_count": retriever.sparse_index.document_count,
            "unique_token_count": retriever.sparse_index.unique_token_count,
            "posting_count": retriever.sparse_index.posting_count,
            "average_nonzero_terms": (
                retriever.sparse_index.posting_count / retriever.sparse_index.document_count
                if retriever.sparse_index.document_count else None
            ),
        },
        "cache": {
            "key": cache_key,
            "dense_hit": retriever.dense_cache_hit,
            "sparse_hit": retriever.sparse_cache_hit,
            "dense_status": cache.dense_status,
            "sparse_status": cache.sparse_status,
            "dense_size_bytes": cache.vector_path.stat().st_size if cache.vector_path.exists() else None,
            "sparse_size_bytes": cache.sparse_path.stat().st_size if cache.sparse_path.exists() else None,
            "metadata": metadata,
        },
        "initialization": {
            "total_seconds": initialize_seconds,
            "model_load_seconds": service.model_load_seconds,
            "embedding_seconds": retriever.embedding_seconds,
            "hybrid_one_pass": retriever.hybrid_embedding_one_pass,
            "dense_index_build_seconds": retriever.dense_index_build_seconds,
            "sparse_index_build_seconds": retriever.sparse_index_build_seconds,
            "cache_load_seconds": retriever.cache_load_seconds,
            "model_load_count": service.model_load_count,
            "hybrid_document_encode_count": service.hybrid_document_encode_count,
        },
        "rrf": {
            "k": retriever.hybrid_rrf_k,
            "candidate_k": retriever.hybrid_candidate_k,
            "dense_weight": retriever.hybrid_dense_weight,
            "sparse_weight": retriever.hybrid_sparse_weight,
        },
        "overall": overall,
        "lecturer_exact_term": exact,
        "machine_learning": machine_learning,
        "controls": controls,
        "ambiguous": ambiguous,
        "sparse_smoke": sparse_smoke,
        "sequential_hybrid": sequential,
        "memory_bytes": {
            "start": memory_start,
            "after_index": memory_after_index,
            "after_all": process.memory_info().rss,
        },
        "final_model_load_count": service.model_load_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.cache_directory, args.mode, not args.no_cache)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
