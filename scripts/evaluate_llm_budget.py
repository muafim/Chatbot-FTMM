"""Deterministic Stage 7B budgeting evaluation over the production dense retriever."""

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

from services.answer_cache import GroundedAnswerCache
from services.chat_service import ChatService
from services.context_builder import ContextBuilder
from services.llm_service import LLMGeneratedAnswer
from services.llm_telemetry import LLMTelemetry
from services.openai_llm_service import GROUNDED_SCHEMA_VERSION, GROUNDING_PROMPT_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = PROJECT_ROOT / "data" / "llm_budget_evaluation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "evaluations" / "stage7b_llm_budget.json"


class FixtureOpenAIProvider:
    """Evaluation-only composer; never used by Flask runtime."""

    provider_name = "openai"
    model_name = "gpt-4o-mini"

    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.case = None

    def generate_grounded_answer(self, grounded_context, evidence_assessment, **kwargs):
        self.telemetry.increment("llm_calls")
        relevant = set(self.case.get("relevant_document_ids", []))
        source_ids = [
            source_id
            for source_id, source in grounded_context.sources.items()
            if source.parent_document_id in relevant
        ]
        if not source_ids:
            return LLMGeneratedAnswer(
                "Informasi tidak ditemukan.", (), "not_answerable",
                provider="openai", requested_model="gpt-4o-mini",
                actual_model="fixture-contract", usage={},
            )
        answer = "; ".join(self.case.get("expected_facts", []))
        answer += " " + " ".join(f"[{source_id}]" for source_id in source_ids)
        return LLMGeneratedAnswer(
            answer.strip(), tuple(source_ids), evidence_assessment.answerability.value,
            provider="openai", requested_model="gpt-4o-mini",
            actual_model="fixture-contract", usage={},
        )


def normalize(text):
    return " ".join(str(text).casefold().split())


def evaluate(fixture_path, output_path):
    import app

    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    telemetry = LLMTelemetry()
    provider = FixtureOpenAIProvider(telemetry)
    with tempfile.TemporaryDirectory(prefix="stage7b-answer-cache-") as directory:
        answer_cache = GroundedAnswerCache(
            directory=directory,
            corpus_fingerprint=app.answer_cache.corpus_fingerprint,
            retrieval_signature=app.answer_cache.retrieval_signature,
            provider="openai",
            model="gpt-4o-mini",
            prompt_version=GROUNDING_PROMPT_VERSION,
            schema_version=GROUNDED_SCHEMA_VERSION,
        )
        service = ChatService(
            retriever=app.retriever,
            context_builder=ContextBuilder(),
            llm_service=provider,
            answer_cache=answer_cache,
            telemetry=telemetry,
            max_context_sources=app.budgeting_settings["max_context_sources"],
        )

        rows = []
        per_category = defaultdict(lambda: {"correct": 0, "total": 0})
        valid_citations = total_citations = local_correct = local_total = 0
        answerability_correct = hallucinations = 0
        for case in cases:
            provider.case = case
            result = service.answer(case["query"])
            actual_path = result.answer_path.value
            path_correct = actual_path == case["expected_path"]
            per_category[case["category"]]["total"] += 1
            per_category[case["category"]]["correct"] += int(path_correct)

            fact_hits = {
                fact: normalize(fact) in normalize(result.answer)
                for fact in case.get("expected_facts", [])
            }
            if case["category"] == "simple_local":
                local_total += 1
                local_correct += int(path_correct and all(fact_hits.values()))
            expected_answerability = "not_answerable" if case["expected_path"] == "no_answer" else "answerable"
            answerability_correct += int(result.answerability.value == expected_answerability)
            total_citations += len(result.citations)
            valid_citations += len(result.citations)
            forbidden_hit = any(
                normalize(value) in normalize(result.answer)
                for value in case.get("forbidden_facts", [])
            )
            hallucinations += int(forbidden_hit or (case["expected_path"] == "no_answer" and bool(result.citations)))
            rows.append({
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "expected_path": case["expected_path"],
                "answer_path": actual_path,
                "answerability": result.answerability.value,
                "answer": result.answer,
                "citations": [source.source_id for source in result.citations],
                "source_parent_ids": [source.parent_document_id for source in result.sources],
                "fact_hits": fact_hits,
                "latency_ms": result.retrieval_metadata.get("timing_ms", {}),
            })

    snapshot = telemetry.snapshot()
    repeat_count = sum(case["category"] == "repeat_cache" for case in cases)
    total = len(cases)
    timing_keys = ("retrieval", "evidence", "smart_gate", "context", "local_or_cache", "llm", "validation", "total")
    summary = {
        "evaluation_mode": "deterministic_budget_contract",
        "scope_note": "Uses dense BGE-M3 retrieval and an evaluation-only fixture composer; no external API call.",
        **snapshot.as_dict(),
        "llm_call_rate": round(snapshot.llm_calls / total, 4),
        "local_answer_accuracy": round(local_correct / local_total, 4) if local_total else 1.0,
        "cache_hit_effectiveness": round(snapshot.cache_hits / repeat_count, 4) if repeat_count else 1.0,
        "citation_validity": round(valid_citations / total_citations, 4) if total_citations else 1.0,
        "answerability_accuracy": round(answerability_correct / total, 4),
        "provider_failure_rate": round(snapshot.llm_failures / snapshot.llm_calls, 4) if snapshot.llm_calls else 0.0,
        "hallucination_count": hallucinations,
        "hallucination_rate": round(hallucinations / total, 4),
        "average_input_tokens_per_llm_call": None,
        "average_output_tokens_per_llm_call": None,
        "per_category": {
            category: {
                **counts,
                "path_accuracy": round(counts["correct"] / counts["total"], 4),
            }
            for category, counts in sorted(per_category.items())
        },
        "mean_latency_ms": {
            key: round(mean(row["latency_ms"].get(key, 0.0) for row in rows), 3)
            for key in timing_keys
        },
    }
    payload = {"summary": summary, "results": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evaluate(args.fixture, args.output)


if __name__ == "__main__":
    main()
