"""Evaluasi reranker dari candidate fixture tanpa memuat BGE-M3 embedder."""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import psutil

from config import get_reranker_settings
from services.reranker_service import RerankerService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "cache" / "evaluations" / "reranker_candidates.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "evaluations" / "stage6a_reranker.json"
STRATEGIES = ("dense", "hybrid", "union")
DEPTHS = (10, 20, 30)


def relevant(candidate, record):
    return (
        candidate["parent_document_id"] in set(record["relevant_document_ids"])
        and (
            not record.get("relevant_sections")
            or candidate["metadata"].get("section") in set(record["relevant_sections"])
        )
    )


def first_relevant_rank(candidates, record):
    return next((rank for rank, item in enumerate(candidates, 1) if relevant(item, record)), None)


def metrics(rows):
    if not rows:
        return {}
    ranks = [row["rank"] for row in rows]
    return {
        "query_count": len(rows),
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(rows),
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(rows),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(rows),
        "mrr": sum(1 / rank if rank else 0 for rank in ranks) / len(rows),
    }


def reranked(candidates, score_map, top_k=5):
    return sorted(
        candidates,
        key=lambda item: (-score_map[item["chunk_id"]], item["rank"], item["chunk_id"]),
    )[:top_k]


def strategy_candidates(record, strategy, depth):
    if strategy != "union":
        return list(record[strategy][:depth])
    merged = {}
    order = []
    for source in ("dense", "sparse"):
        for item in record[source][:depth]:
            chunk_id = item["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = dict(item)
                order.append(chunk_id)
            else:
                current = merged[chunk_id]
                for key in ("dense_rank", "dense_score", "sparse_rank", "sparse_score"):
                    if current.get(key) is None and item.get(key) is not None:
                        current[key] = item[key]
    return [dict(merged[chunk_id], rank=rank) for rank, chunk_id in enumerate(order, 1)]


def candidate_recall(records, strategy, depth):
    hits = []
    coverage = []
    for record in records:
        candidates = strategy_candidates(record, strategy, depth)
        relevant_parents = set(record["relevant_document_ids"])
        found = {item["parent_document_id"] for item in candidates if relevant(item, record)}
        hits.append(bool(found))
        coverage.append(len(found) / len(relevant_parents) if relevant_parents else 0.0)
    return {
        "query_hit_recall": sum(hits) / len(hits) if hits else None,
        "relevant_parent_coverage": sum(coverage) / len(coverage) if coverage else None,
    }


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def evaluate(input_path, candidate_depth=30, top_k=5):
    fixture = json.loads(input_path.read_text(encoding="utf-8"))
    records = fixture["records"]
    settings = get_reranker_settings()
    process = psutil.Process()
    memory_start = process.memory_info().rss
    service = RerankerService(
        model_name=settings["model_name"],
        device=settings["device"],
        batch_size=settings["batch_size"],
        max_length=settings["max_length"],
    )
    load_started = time.perf_counter()
    backend = service.backend
    load_wall_seconds = time.perf_counter() - load_started
    memory_after_model = process.memory_info().rss

    all_rows = defaultdict(list)
    detailed = []
    pair_lengths = []
    first_score_memory = None
    latencies = []
    for record in records:
        unique = {}
        for strategy in STRATEGIES:
            for item in strategy_candidates(record, strategy, candidate_depth):
                unique.setdefault(item["chunk_id"], item)
        candidates = list(unique.values())
        pairs = [(record["query"], item["content"]) for item in candidates]
        encoded = backend.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            truncation=False,
            add_special_tokens=True,
        )
        pair_lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
        started = time.perf_counter()
        scores = service.score_pairs(pairs)
        latencies.append({"pairs": len(pairs), "seconds": time.perf_counter() - started})
        if first_score_memory is None:
            first_score_memory = process.memory_info().rss
        score_map = dict(zip(unique, scores))
        record_result = {"query": record["query"], "fixture": record["fixture"], "scores": score_map}
        for strategy in STRATEGIES:
            source = strategy_candidates(record, strategy, candidate_depth)
            raw = source[:top_k]
            ranked = reranked(source, score_map, top_k)
            record_result[strategy] = {
                "raw_rank": first_relevant_rank(raw, record),
                "reranked_rank": first_relevant_rank(ranked, record),
                "reranked_top": [
                    {
                        "rank": rank,
                        "chunk_id": item["chunk_id"],
                        "parent_document_id": item["parent_document_id"],
                        "name": item["metadata"].get("name"),
                        "section": item["metadata"].get("section"),
                        "reranker_score": score_map[item["chunk_id"]],
                        "original_rank": item["rank"],
                    }
                    for rank, item in enumerate(ranked, 1)
                ],
            }
            all_rows[f"{strategy}_raw"].append(
                {"rank": first_relevant_rank(raw, record), "record": record}
            )
            all_rows[f"{strategy}_rerank"].append(
                {"rank": first_relevant_rank(ranked, record), "record": record}
            )
        detailed.append(record_result)

    overall_records = [record for record in records if record["fixture"] == "overall" and record["relevant_document_ids"]]
    exact_records = [record for record in records if record["fixture"] == "lecturer_exact_term"]
    output_metrics = {}
    category_metrics = {}
    for name, rows in all_rows.items():
        overall_rows = [row for row in rows if row["record"] in overall_records]
        exact_rows = [row for row in rows if row["record"] in exact_records]
        output_metrics[name] = {
            "overall_48": metrics(overall_rows),
            "common_39": metrics(overall_rows[:39]),
            "lecturer_exact_term": metrics(exact_rows),
        }
        category_metrics[name] = {
            category: metrics([row for row in overall_rows if row["record"].get("category") == category])
            for category in ("academic", "course", "ftmm", "lecturer", "staff")
        }

    return {
        "model": settings["model_name"],
        "device": settings["device"],
        "precision": "fp32 (use_fp16=False)",
        "batch_size": settings["batch_size"],
        "max_length": settings["max_length"],
        "candidate_depth": candidate_depth,
        "top_k": top_k,
        "candidate_fixture": str(input_path),
        "embedding_model_loaded": False,
        "candidate_recall_exact_term": {
            strategy: {
                str(depth): candidate_recall(exact_records, strategy, depth)
                for depth in DEPTHS
            }
            for strategy in STRATEGIES
        },
        "metrics": output_metrics,
        "category_metrics": category_metrics,
        "token_lengths": {
            "pairs": len(pair_lengths),
            "min": min(pair_lengths),
            "median": statistics.median(pair_lengths),
            "p95": percentile(pair_lengths, 0.95),
            "max": max(pair_lengths),
            "truncated": sum(length > settings["max_length"] for length in pair_lengths),
        },
        "performance": {
            "load_wall_seconds": load_wall_seconds,
            "service_load_seconds": service.model_load_seconds,
            "score_calls": service.score_call_count,
            "total_pairs": sum(item["pairs"] for item in latencies),
            "total_score_seconds": sum(item["seconds"] for item in latencies),
            "calls": latencies,
        },
        "memory_bytes": {
            "start": memory_start,
            "after_model": memory_after_model,
            "after_first_scoring": first_score_memory,
            "after_all": process.memory_info().rss,
        },
        "model_load_count": service.model_load_count,
        "details": detailed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-depth", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    result = evaluate(args.input, args.candidate_depth, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("model", "token_lengths", "performance", "memory_bytes")}, indent=2))


if __name__ == "__main__":
    main()
