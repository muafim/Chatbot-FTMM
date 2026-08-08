"""Benchmark resource dan smoke test reranker tanpa BGE-M3 embedder."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import psutil

from config import get_reranker_settings
from services.reranker_service import RerankerService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = PROJECT_ROOT / "cache" / "evaluations" / "reranker_candidates.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "evaluations" / "stage6a_reranker_isolated.json"
PARENTS = {
    "ratih": "lecturer-199501262020013201",
    "rizki": "lecturer-199106272020073101",
    "asif": "lecturer-199207222022103101",
    "rezi": "lecturer-199609262023103201",
    "maryamah": "lecturer-199507012022103201",
    "fakhruzzaman": "lecturer-199308222020013101",
    "hci": "lecturer-199308222020013101",
}


def available_bytes():
    return psutil.virtual_memory().available


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def find_record(records, query):
    return next(record for record in records if record["query"].lower() == query.lower())


def content_map(records):
    result = {}
    for record in records:
        for source in ("dense", "sparse", "hybrid", "union"):
            for item in record[source]:
                parent_id = item["parent_document_id"]
                if (
                    parent_id not in result
                    or item["metadata"].get("chunk_role") == "research_interest"
                ):
                    result[parent_id] = item["content"]
    return result


def snapshot_size(model_name):
    cache = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + model_name.replace("/", "--")
    )
    unique_files = {}
    for path in cache.rglob("*"):
        if path.is_file():
            try:
                unique_files[str(path.resolve())] = path.stat().st_size
            except OSError:
                continue
    return {"path": str(cache), "bytes": sum(unique_files.values()), "files": len(unique_files)}


def benchmark(fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    records = fixture["records"]
    contents = content_map(records)
    settings = get_reranker_settings()
    process = psutil.Process()
    rss_start = process.memory_info().rss
    available_start = available_bytes()
    service = RerankerService(
        model_name=settings["model_name"],
        device=settings["device"],
        batch_size=settings["batch_size"],
        max_length=settings["max_length"],
    )
    loaded_at = time.perf_counter()
    backend = service.backend
    load_wall = time.perf_counter() - loaded_at
    rss_after_model = process.memory_info().rss
    available_after_model = available_bytes()

    ml_query = "dosen yang fokus machine learning"
    ml_pairs = [
        (ml_query, contents[PARENTS["ratih"]]),
        (ml_query, contents["lecturer-199208252024023201"]),
        (ml_query, contents["lecturer-197107122008122001"]),
    ]
    first_at = time.perf_counter()
    ml_raw = service.score_pairs(ml_pairs)
    first_seconds = time.perf_counter() - first_at
    rss_after_first = process.memory_info().rss
    ml_normalized = service.score_pairs(ml_pairs, normalize=True)
    ml_repeated = service.score_pairs(ml_pairs)
    ml_single = [service.score_pairs([pair])[0] for pair in ml_pairs]

    nlp_query = "dosen yang meneliti natural language processing"
    nlp_pairs = [
        (nlp_query, contents[PARENTS["maryamah"]]),
        (nlp_query, contents[PARENTS["fakhruzzaman"]]),
        (nlp_query, contents["lecturer-199208252024023201"]),
    ]
    nlp_scores = service.score_pairs(nlp_pairs)
    hci_query = "dosen yang meneliti interaksi manusia dan komputer"
    hci_pairs = [
        (hci_query, contents[PARENTS["hci"]]),
        (hci_query, contents["lecturer-199208252024023201"]),
    ]
    hci_scores = service.score_pairs(hci_pairs)

    ml_record = find_record(records, "Dosen yang fokus machine learning")
    latency = {}
    for count in (1, 5, 10, 20, 30):
        pairs = [(ml_record["query"], item["content"]) for item in ml_record["union"][:count]]
        started = time.perf_counter()
        scores = service.score_pairs(pairs)
        latency[str(count)] = {
            "seconds": time.perf_counter() - started,
            "finite": all(math.isfinite(score) for score in scores),
        }
    rss_after_batch = process.memory_info().rss

    token_lengths = []
    for record in records:
        candidates = record["union"][:30]
        encoded = backend.tokenizer(
            [record["query"]] * len(candidates),
            [item["content"] for item in candidates],
            truncation=False,
            add_special_tokens=True,
        )
        token_lengths.extend(len(ids) for ids in encoded["input_ids"])
    memory_info = process.memory_info()
    peak = getattr(memory_info, "peak_wset", None)
    return {
        "model": settings["model_name"],
        "library": "FlagEmbedding 1.4.0",
        "device": settings["device"],
        "precision": "fp32 (use_fp16=False)",
        "model_cache": snapshot_size(settings["model_name"]),
        "load_seconds": load_wall,
        "service_load_seconds": service.model_load_seconds,
        "memory_bytes": {
            "process_start": rss_start,
            "after_model": rss_after_model,
            "after_first_scoring": rss_after_first,
            "after_batch_scoring": rss_after_batch,
            "peak_working_set": peak,
            "system_available_start": available_start,
            "system_available_after_model": available_after_model,
        },
        "score_validation": {
            "machine_learning_raw": ml_raw,
            "machine_learning_normalized": ml_normalized,
            "machine_learning_repeated": ml_repeated,
            "machine_learning_single_pair": ml_single,
            "raw_normalized_same_order": sorted(range(3), key=lambda i: -ml_raw[i]) == sorted(range(3), key=lambda i: -ml_normalized[i]),
            "repeated_max_abs_difference": max(abs(a - b) for a, b in zip(ml_raw, ml_repeated)),
            "single_batch_max_abs_difference": max(abs(a - b) for a, b in zip(ml_raw, ml_single)),
            "all_finite": all(math.isfinite(score) for score in ml_raw + ml_normalized + nlp_scores + hci_scores),
            "machine_learning_relevant_first": max(range(3), key=ml_raw.__getitem__) == 0,
            "nlp_scores_maryamah_fakhruzzaman_unrelated": nlp_scores,
            "nlp_relevant_top_two": set(sorted(range(3), key=lambda i: -nlp_scores[i])[:2]) == {0, 1},
            "hci_scores_relevant_unrelated": hci_scores,
            "hci_relevant_first": hci_scores[0] > hci_scores[1],
        },
        "first_scoring_seconds": first_seconds,
        "latency_seconds": latency,
        "token_lengths": {
            "pairs": len(token_lengths),
            "min": min(token_lengths),
            "median": statistics.median(token_lengths),
            "p95": percentile(token_lengths, 0.95),
            "max": max(token_lengths),
            "max_length": settings["max_length"],
            "truncated": sum(length > settings["max_length"] for length in token_lengths),
        },
        "model_load_count": service.model_load_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = benchmark(args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
