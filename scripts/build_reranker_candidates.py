"""Bangun fixture kandidat retrieval; reranker tidak dimuat oleh script ini."""

import argparse
import json
from pathlib import Path

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
from services.reranker_candidates import union_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "evaluations" / "reranker_candidates.json"


def serialize(item, rank):
    return {
        "chunk_id": item.document.id,
        "parent_document_id": getattr(item.document, "parent_document_id", item.document.id),
        "content": item.document.content,
        "metadata": item.document.metadata,
        "rank": rank,
        "retrieval_score": item.score,
        "dense_rank": item.dense_rank,
        "dense_score": item.dense_score,
        "sparse_rank": item.sparse_rank,
        "sparse_score": item.sparse_score,
        "fusion_score": item.fusion_score,
    }


def load_cases():
    cases = []
    seen = set()
    for filename, fixture_name in (
        ("retrieval_evaluation.json", "overall"),
        ("lecturer_exact_term_evaluation.json", "lecturer_exact_term"),
    ):
        source = json.loads((PROJECT_ROOT / "data" / filename).read_text(encoding="utf-8"))
        for position, case in enumerate(source):
            key = (fixture_name, case["query"])
            if key in seen:
                continue
            seen.add(key)
            cases.append({**case, "fixture": fixture_name, "fixture_position": position})
    return cases


def build(depth=40):
    repository = KnowledgeRepository(PROJECT_ROOT / "data")
    service = create_embedding_service(
        "bge-m3",
        baseline_settings=get_embedding_settings(),
        bge_m3_settings=get_bge_m3_settings(),
    )
    settings = get_retrieval_settings()
    retriever = DenseRetriever(
        repository,
        service,
        InMemoryVectorIndex(),
        sparse_index=InMemorySparseIndex(),
        retrieval_mode="hybrid",
        top_k=5,
        min_score=None,
        max_chunks_per_parent=settings["max_chunks_per_parent"],
        candidate_multiplier=settings["candidate_multiplier"],
        hybrid_rrf_k=settings["hybrid_rrf_k"],
        hybrid_candidate_k=max(settings["hybrid_candidate_k"], depth),
        hybrid_dense_weight=settings["hybrid_dense_weight"],
        hybrid_sparse_weight=settings["hybrid_sparse_weight"],
        embedding_cache=DocumentEmbeddingCache(
            PROJECT_ROOT / "cache" / "embeddings", "bge_m3_hybrid", enabled=True
        ),
        document_chunker=DocumentChunker(**get_chunking_settings()),
        intent_router=QueryIntentRouter(),
        intent_routing_enabled=True,
    )
    retriever.initialize()
    records = []
    for case in load_cases():
        hybrid = retriever.retrieve(case["query"], top_k=depth)
        dense = retriever.last_dense_candidates[:depth]
        sparse = retriever.last_sparse_candidates[:depth]
        union = union_candidates(dense, sparse)
        plan = retriever.last_retrieval_plan
        records.append(
            {
                "fixture": case["fixture"],
                "fixture_position": case["fixture_position"],
                "query": case["query"],
                "category": case.get("category", "lecturer"),
                "expected_intent": case.get("expected_intent"),
                "relevant_document_ids": case.get("relevant_document_ids", []),
                "relevant_sections": case.get("relevant_sections", []),
                "retrieval_trait": case.get("retrieval_trait"),
                "intent": plan.intent.value,
                "metadata_filter": retriever.last_used_filter,
                "fallback": retriever.last_fallback_used,
                "dense": [serialize(item, rank) for rank, item in enumerate(dense, 1)],
                "sparse": [serialize(item, rank) for rank, item in enumerate(sparse, 1)],
                "hybrid": [serialize(item, rank) for rank, item in enumerate(hybrid, 1)],
                "union": [serialize(item, rank) for rank, item in enumerate(union, 1)],
            }
        )
    return {
        "format_version": 1,
        "model": service.model_name,
        "candidate_depth": depth,
        "chunk_count": len(retriever._documents),
        "corpus_fingerprint": corpus_fingerprint(retriever._documents),
        "model_load_count": service.model_load_count,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.depth <= 0:
        parser.error("--depth harus lebih besar dari nol")
    result = build(args.depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(result['records'])} queries to {args.output}")


if __name__ == "__main__":
    main()
