"""Bandingkan ranking dua output evaluate_retrieval tanpa membandingkan skala skor."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("bge_m3", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    bge = json.loads(args.bge_m3.read_text(encoding="utf-8"))
    print("Metric | Baseline | BGE-M3")
    for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr"):
        print(f"{metric} | {baseline['metrics'][metric]:.4f} | {bge['metrics'][metric]:.4f}")
    print("\nQuery | Expected | Baseline rank | BGE-M3 rank | BGE Top-3")
    bge_cases = {case["query"]: case for case in bge["cases"]}
    for baseline_case in baseline["cases"]:
        bge_case = bge_cases[baseline_case["query"]]
        top_three = ", ".join(
            item.get("parent_document_id", item.get("document_id"))
            for item in bge_case["top_5"][:3]
        )
        expected = ",".join(baseline_case["relevant_document_ids"])
        print(
            f"{baseline_case['query']} | {expected} | "
            f"{baseline_case['expected_rank'] or '-'} | "
            f"{bge_case['expected_rank'] or '-'} | {top_three}"
        )


if __name__ == "__main__":
    main()
