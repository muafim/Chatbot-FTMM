"""Guarded co-residency probe; tidak terhubung ke Flask atau production pipeline."""

import argparse
import json
import time
from pathlib import Path

import psutil

from config import get_bge_m3_settings, get_embedding_settings, get_reranker_settings
from services.embedding_factory import create_embedding_service
from services.reranker_service import RerankerService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "evaluations" / "stage6a_coresidency.json"
MIN_START_AVAILABLE = 2 * 1024**3
MIN_JOINT_INFERENCE_AVAILABLE = 2 * 1024**3


def sample(process):
    return {
        "process_rss": process.memory_info().rss,
        "system_available": psutil.virtual_memory().available,
    }


def benchmark():
    process = psutil.Process()
    result = {"start": sample(process), "safe": False, "sequential": None}
    if result["start"]["system_available"] < MIN_START_AVAILABLE:
        result["status"] = "skipped_low_start_memory"
        return result

    embedding = create_embedding_service(
        "bge-m3",
        baseline_settings=get_embedding_settings(),
        bge_m3_settings=get_bge_m3_settings(),
    )
    started = time.perf_counter()
    embedding.encode_query_hybrid("Dosen yang fokus machine learning")
    result["after_embedding"] = sample(process)
    result["embedding_load_and_query_seconds"] = time.perf_counter() - started

    settings = get_reranker_settings()
    reranker = RerankerService(
        model_name=settings["model_name"],
        device=settings["device"],
        batch_size=settings["batch_size"],
        max_length=settings["max_length"],
    )
    started = time.perf_counter()
    reranker.backend
    result["after_reranker"] = sample(process)
    result["reranker_load_seconds"] = time.perf_counter() - started
    result["embedding_model_load_count"] = embedding.model_load_count
    result["reranker_model_load_count"] = reranker.model_load_count

    if result["after_reranker"]["system_available"] < MIN_JOINT_INFERENCE_AVAILABLE:
        result["status"] = "models_loaded_but_joint_inference_blocked_by_memory_guard"
        return result

    started = time.perf_counter()
    fixture = json.loads(
        (PROJECT_ROOT / "cache" / "evaluations" / "reranker_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    ratih_content = next(
        item["content"]
        for record in fixture["records"]
        for item in record["union"]
        if item["parent_document_id"] == "lecturer-199501262020013201"
        and item["metadata"].get("chunk_role") == "research_interest"
    )
    scores = reranker.score_pairs(
        [
            (
                "Dosen yang fokus machine learning",
                ratih_content,
            )
        ]
    )
    result["after_joint_query"] = sample(process)
    result["joint_query_seconds"] = time.perf_counter() - started
    result["joint_score"] = scores[0]
    result["safe"] = True
    result["status"] = "joint_inference_completed"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
