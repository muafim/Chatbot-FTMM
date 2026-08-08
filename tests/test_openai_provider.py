import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from domain.models import DocumentChunk, RetrievedDocument
from services.context_builder import ContextBuilder
from services.evidence_evaluator import EvidenceEvaluator
from services.errors import (
    LLMAuthenticationError,
    LLMGenerationError,
    LLMProviderError,
    LLMRateLimitError,
)
from services.llm_budget import DailyLLMBudgetGuard
from services.llm_telemetry import LLMTelemetry
from services.openai_llm_service import GroundedResponseSchema, OpenAILLMProvider


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class FakeTimeout(Exception):
    pass


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


def response(payload, model="gpt-4o-mini", usage=True):
    usage_object = (
        SimpleNamespace(input_tokens=120, output_tokens=30, total_tokens=150)
        if usage else None
    )
    return SimpleNamespace(output_parsed=payload, model=model, usage=usage_object)


def valid_payload():
    return GroundedResponseSchema(
        answer="Dekan FTMM adalah Prof. Test. [S1]",
        citations=["S1"],
        answerability="answerable",
    )


def grounded_input():
    item = RetrievedDocument(
        DocumentChunk(
            "staff::chunk-0000",
            "staff",
            "Jabatan: Dekan FTMM\nNama: Prof. Test",
            0,
            {"type": "staff", "name": "Prof. Test"},
        ),
        score=0.9,
    )
    context = ContextBuilder().build_grounded_context([item])
    assessment = EvidenceEvaluator().evaluate("Siapa Dekan FTMM?", [item])
    return context, assessment


class OpenAIProviderTests(unittest.TestCase):
    def test_production_runtime_is_openai_direct_without_openrouter(self):
        root = Path(__file__).resolve().parents[1]
        runtime_files = [
            root / "app.py",
            root / "config.py",
            root / "services" / "llm_service.py",
            root / "services" / "openai_llm_service.py",
            root / "services" / "chat_service.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
        self.assertIn("OPENAI_API_KEY", combined)
        self.assertIn("responses.parse", combined)
        self.assertNotIn("OPENROUTER", combined.upper())
        self.assertNotIn("openrouter.ai", combined.lower())

    def test_valid_structured_response_usage_model_and_request(self):
        client = FakeClient([response(valid_payload())])
        telemetry = LLMTelemetry()
        provider = OpenAILLMProvider(
            api_key="server-secret",
            client=client,
            telemetry=telemetry,
        )
        context, assessment = grounded_input()
        result = provider.generate_grounded_answer("Siapa Dekan?", context, assessment)
        self.assertEqual(result.actual_model, "gpt-4o-mini")
        self.assertEqual(result.usage["total_tokens"], 150)
        request = client.responses.calls[0]
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertIs(request["text_format"], GroundedResponseSchema)
        self.assertEqual(request["max_output_tokens"], 300)
        self.assertNotIn("base_url", request)
        self.assertNotIn("server-secret", repr(request))
        self.assertEqual(telemetry.snapshot().llm_calls, 1)

    def test_malformed_and_empty_response_retry_once_then_fail(self):
        for outcomes in (
            [response({"answer": "missing fields"}), response({"answer": "still bad"})],
            [response(None), response(None)],
        ):
            with self.subTest():
                client = FakeClient(outcomes)
                provider = OpenAILLMProvider(api_key="x", client=client, max_retries=1)
                context, assessment = grounded_input()
                with self.assertRaises(LLMGenerationError):
                    provider.generate_grounded_answer("q", context, assessment)
                self.assertEqual(len(client.responses.calls), 2)

    def test_authentication_and_rate_limit_are_not_retried(self):
        for status, expected in ((401, LLMAuthenticationError), (429, LLMRateLimitError)):
            client = FakeClient([FakeHTTPError(status)])
            provider = OpenAILLMProvider(api_key="x", client=client, max_retries=1)
            context, assessment = grounded_input()
            with self.assertRaises(expected):
                provider.generate_grounded_answer("q", context, assessment)
            self.assertEqual(len(client.responses.calls), 1)

    def test_server_error_and_timeout_retry_once(self):
        for error in (FakeHTTPError(500), FakeTimeout("slow")):
            client = FakeClient([error, error])
            provider = OpenAILLMProvider(api_key="x", client=client, max_retries=1)
            context, assessment = grounded_input()
            with self.assertRaises(LLMProviderError):
                provider.generate_grounded_answer("q", context, assessment)
            self.assertEqual(len(client.responses.calls), 2)

    def test_every_retry_consumes_daily_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = DailyLLMBudgetGuard(Path(directory) / "budget.json", daily_call_limit=1)
            client = FakeClient([response(None)])
            provider = OpenAILLMProvider(
                api_key="x", client=client, max_retries=1, budget_guard=budget
            )
            context, assessment = grounded_input()
            with self.assertRaisesRegex(Exception, "batas|Batas"):
                provider.generate_grounded_answer("q", context, assessment)
            self.assertEqual(len(client.responses.calls), 1)


if __name__ == "__main__":
    unittest.main()
