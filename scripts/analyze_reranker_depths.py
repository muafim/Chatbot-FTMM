"""Analisis candidate depth dari score fixture; tidak memuat model apa pun."""

import argparse
import json
from pathlib import Path

from scripts.evaluate_reranker import (
    first_relevant_rank,
    metrics,
    reranked,
    strategy_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyze(candidate_path, result_path):
    fixture = json.loads(candidate_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    score_by_key = {
        (item["fixture"], item["query"]): item["scores"] for item in result["details"]
    }
    output = {}
    for depth in (10, 20, 30):
        output[str(depth)] = {}
        for strategy in ("dense", "hybrid", "union"):
            overall = []
            exact = []
            categories = {}
            for record in fixture["records"]:
                scores = score_by_key[(record["fixture"], record["query"])]
                candidates = strategy_candidates(record, strategy, depth)
                ranked = reranked(candidates, scores, 5)
                row = {"rank": first_relevant_rank(ranked, record), "record": record}
                if record["fixture"] == "overall" and record["relevant_document_ids"]:
                    overall.append(row)
                elif record["fixture"] == "lecturer_exact_term":
                    exact.append(row)
            for category in ("academic", "course", "ftmm", "lecturer", "staff"):
                categories[category] = metrics(
                    [row for row in overall if row["record"].get("category") == category]
                )
            output[str(depth)][strategy] = {
                "common_39": metrics(overall[:39]),
                "overall_48": metrics(overall),
                "lecturer_exact_term": metrics(exact),
                "categories": categories,
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "cache" / "evaluations" / "reranker_candidates.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "cache" / "evaluations" / "stage6a_reranker.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "cache" / "evaluations" / "stage6a_reranker_depths.json",
    )
    args = parser.parse_args()
    result = analyze(args.candidates, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
