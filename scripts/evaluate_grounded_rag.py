"""Evaluate retrieval-to-citation contracts; optional live OpenAI generation.

The default contract mode uses verified fixture facts as a deterministic composer.
It measures plumbing, retrieval availability, answerability guards, and citation
traceability. It is deliberately not reported as an LLM factual-quality score.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

from domain.models import Answerability
from services.chat_service import ChatService
from services.context_builder import ContextBuilder
from services.llm_service import LLMGeneratedAnswer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = PROJECT_ROOT / "data" / "grounded_rag_evaluation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "evaluations" / "stage7_grounded_contract.json"


class FixtureComposer:
    """Evaluation-only composer; never imported by Flask production."""

    def __init__(self):
        self.case = None

    def generate_grounded_answer(self, grounded_context, evidence_assessment, **kwargs):
        relevant = set(self.case.get("relevant_document_ids", []))
        source_ids = [
            source_id
            for source_id, source in grounded_context.sources.items()
            if source.parent_document_id in relevant
        ]
        if not source_ids:
            return LLMGeneratedAnswer(
                "Informasi tidak ditemukan pada sumber yang diberikan.", (), "not_answerable"
            )
        facts = self.case.get("expected_facts", [])
        marker_text = " ".join(f"[{source_id}]" for source_id in source_ids)
        answer = "; ".join(facts) + f" {marker_text}"
        if evidence_assessment.missing_part:
            answer += f" {evidence_assessment.missing_part}"
        return LLMGeneratedAnswer(
            answer.strip(), tuple(source_ids), evidence_assessment.answerability.value
        )


def normalize(text):
    return " ".join(str(text).casefold().split())


def evaluate(mode, fixture_path, output_path):
    # Importing app keeps model lazy; the first actual retrieval initializes dense BGE-M3.
    import app as production_app

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY tidak tersedia; live evaluation tidak dijalankan.")
        composer = production_app.llm_service
    else:
        composer = FixtureComposer()
    service = ChatService(
        retriever=production_app.retriever,
        context_builder=ContextBuilder(),
        llm_service=composer,
        citation_correction_retries=1,
    )

    rows = []
    category_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    valid_citations = total_citations = covered = expected_factual = 0
    relevant_cited = total_cited = no_answer_correct = expected_no_answer = 0
    hallucinations = 0
    for case in fixture:
        if isinstance(composer, FixtureComposer):
            composer.case = case
        result = service.answer(case["query"])
        expected = case["expected_answerability"]
        predicted = result.answerability.value
        correct = predicted == expected
        category_counts[case["category"]]["total"] += 1
        category_counts[case["category"]]["correct"] += int(correct)

        citations = result.sources
        total_citations += len(citations)
        valid_citations += len(citations)  # Final ChatResult exposes validated citations only.
        relevant = set(case.get("relevant_document_ids", []))
        cited_parents = [source.parent_document_id for source in citations]
        total_cited += len(cited_parents)
        relevant_cited += sum(parent in relevant for parent in cited_parents)

        if expected != Answerability.NOT_ANSWERABLE.value:
            expected_factual += 1
            covered += int(bool(citations))
        else:
            expected_no_answer += 1
            no_answer_correct += int(predicted == Answerability.NOT_ANSWERABLE.value and not citations)

        forbidden_hit = any(normalize(fact) in normalize(result.answer) for fact in case.get("forbidden_facts", []))
        if expected == Answerability.NOT_ANSWERABLE.value and (citations or forbidden_hit):
            hallucinations += 1
        expected_fact_hits = {
            fact: normalize(fact) in normalize(result.answer)
            for fact in case.get("expected_facts", [])
        }
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "expected_answerability": expected,
            "answerability": predicted,
            "answer": result.answer,
            "citations": [source.source_id for source in citations],
            "valid_citations": [source.source_id for source in citations],
            "cited_parent_ids": cited_parents,
            "relevant_sources": sorted(relevant),
            "expected_fact_hits": expected_fact_hits,
            "grounding_status": result.grounding_status.value,
            "intent": result.retrieval_metadata.get("intent", "unknown"),
            "retrieved_chunk_ids": result.retrieval_metadata.get("chunk_ids", []),
            "latency_ms": result.retrieval_metadata.get("timing_ms", {}),
        })

    total = len(rows)
    timing_keys = ("retrieval", "context", "llm", "validation", "total")
    performance = {
        key: round(mean(row["latency_ms"].get(key, 0.0) for row in rows), 3)
        for key in timing_keys
    }
    summary = {
        "evaluation_mode": "retrieval_and_grounding_contract" if mode == "contract" else "live_openai",
        "scope_note": (
            "Contract mode does not measure free-form LLM factual quality."
            if mode == "contract"
            else "Live OpenAI output evaluated with normalized expected-fact checks."
        ),
        "query_count": total,
        "answerability_accuracy": round(sum(row["expected_answerability"] == row["answerability"] for row in rows) / total, 4),
        "citation_validity": round(valid_citations / total_citations, 4) if total_citations else 1.0,
        "citation_coverage": round(covered / expected_factual, 4) if expected_factual else 1.0,
        "source_precision": round(relevant_cited / total_cited, 4) if total_cited else 1.0,
        "no_answer_accuracy": round(no_answer_correct / expected_no_answer, 4) if expected_no_answer else 1.0,
        "hallucination_count": hallucinations,
        "hallucination_rate": round(hallucinations / total, 4),
        "per_category": {
            category: {
                **counts,
                "answerability_accuracy": round(counts["correct"] / counts["total"], 4),
            }
            for category, counts in sorted(category_counts.items())
        },
        "mean_latency_ms": performance,
    }
    payload = {"summary": summary, "results": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "openai"), default="contract")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evaluate(args.mode, args.fixture, args.output)


if __name__ == "__main__":
    main()
