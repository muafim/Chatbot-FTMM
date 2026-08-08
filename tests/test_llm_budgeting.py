import tempfile
import unittest
from pathlib import Path

from domain.models import (
    AnswerPath,
    Answerability,
    DocumentChunk,
    GroundedAnswer,
    GroundingStatus,
    QueryIntent,
    RetrievedDocument,
    RetrievalPlan,
)
from services.answer_cache import GroundedAnswerCache, normalize_cache_query
from services.chat_service import ChatService
from services.context_builder import ContextBuilder
from services.errors import LLMBudgetExceeded
from services.llm_budget import DailyLLMBudgetGuard
from services.llm_service import LLMGeneratedAnswer
from services.llm_telemetry import LLMTelemetry


class PlannedRetriever:
    def __init__(self, intent, results):
        self.last_retrieval_plan = RetrievalPlan(query="fixture", intent=intent)
        self.results = results

    def retrieve(self, query):
        self.last_retrieval_plan = RetrievalPlan(query=query, intent=self.last_retrieval_plan.intent)
        return list(self.results)


class CountingLLM:
    def __init__(self, output=None, error=None):
        self.calls = 0
        self.output = output or LLMGeneratedAnswer(
            "Ringkasan berdasarkan sumber. [S1]", ("S1",), "answerable",
            provider="openai", requested_model="gpt-4o-mini", actual_model="gpt-4o-mini",
        )
        self.error = error

    def generate_grounded_answer(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output


def item(chunk_id, parent_id, content, metadata):
    return RetrievedDocument(DocumentChunk(chunk_id, parent_id, content, 0, metadata), score=0.9)


STAFF = item(
    "staff-dean::chunk-0000", "staff-dean",
    "Jabatan: Dekan FTMM\nNama: Prof. Dr. Dwi Setyawan",
    {"type": "staff", "name": "Prof. Dr. Dwi Setyawan", "role": "Dekan"},
)
VISION = item(
    "ftmm::visi", "ftmm",
    "Informasi FTMM | Visi\nVisi: Fakultas teknik unggul dan bermartabat.",
    {"type": "ftmm", "name": "FTMM", "section": "Visi"},
)
CONTACT = item(
    "ftmm::contact", "ftmm",
    "Informasi FTMM | Kontak Fakultas\nHelpdesk: +62 881-0360-00830\nEmail: info@ftmm.unair.ac.id",
    {"type": "ftmm", "name": "FTMM", "section": "Kontak Fakultas"},
)
COURSE = item(
    "course-sic306::overview", "course-sic306",
    "Nama Mata Kuliah: Machine Learning\nKode: SIC306\nSKS: 3 sks\nSemester: 6",
    {"type": "course", "name": "Machine Learning", "code": "SIC306", "credits": "3 sks", "semester": "6"},
)
LECTURER = item(
    "lecturer-mary::research", "lecturer-mary",
    "Nama: Dr. Maryamah\nResearch Interest: Natural Language Processing",
    {"type": "lecturer", "name": "Dr. Maryamah", "research_interest": "Natural Language Processing"},
)
HISTORY = item(
    "ftmm::history", "ftmm",
    "Informasi FTMM | Sejarah Fakultas\nFTMM didirikan melalui keputusan universitas.",
    {"type": "ftmm", "name": "FTMM", "section": "Sejarah Fakultas"},
)


def cache(directory, corpus="corpus-a", model="gpt-4o-mini", prompt="prompt-v1"):
    return GroundedAnswerCache(
        directory=directory,
        corpus_fingerprint=corpus,
        retrieval_signature={"mode": "dense", "top_k": 10},
        provider="openai",
        model=model,
        prompt_version=prompt,
        schema_version="schema-v1",
    )


class SmartBudgetingTests(unittest.TestCase):
    def test_simple_staff_ftmm_course_and_lecturer_are_local(self):
        cases = (
            (QueryIntent.STAFF, STAFF, "Siapa Dekan FTMM?"),
            (QueryIntent.FTMM, VISION, "Apa visi FTMM?"),
            (QueryIntent.FTMM, CONTACT, "Apa kontak FTMM?"),
            (QueryIntent.COURSE, COURSE, "Berapa SKS Machine Learning?"),
            (QueryIntent.LECTURER, LECTURER, "Apa research interest Dr. Maryamah?"),
        )
        for intent, evidence, query in cases:
            with self.subTest(query=query):
                llm = CountingLLM()
                result = ChatService(PlannedRetriever(intent, [evidence]), ContextBuilder(), llm).answer(query)
                self.assertEqual(result.answer_path, AnswerPath.LOCAL)
                self.assertEqual(llm.calls, 0)
                self.assertEqual([source.source_id for source in result.citations], ["S1"])
                if query == "Apa kontak FTMM?":
                    self.assertIn("info@ftmm.unair.ac.id", result.answer)

    def test_complex_shapes_use_llm(self):
        for query in (
            "Jelaskan sejarah FTMM secara ringkas.",
            "Jelaskan visi dan misi FTMM.",
            "Bandingkan dua mata kuliah.",
            "Ringkas prosedur SKMA.",
        ):
            llm = CountingLLM()
            result = ChatService(PlannedRetriever(QueryIntent.FTMM, [HISTORY]), ContextBuilder(), llm).answer(query)
            self.assertEqual(result.answer_path, AnswerPath.LLM)
            self.assertEqual(llm.calls, 1)

    def test_no_answer_paths_make_zero_llm_calls(self):
        for query in ("Berapa harga Bitcoin?", "Siapa Wakil Dekan IV?", "Program Kedokteran FTMM"):
            llm = CountingLLM()
            result = ChatService(PlannedRetriever(QueryIntent.GENERAL, [STAFF]), ContextBuilder(), llm).answer(query)
            self.assertEqual(result.answer_path, AnswerPath.NO_ANSWER)
            self.assertEqual(llm.calls, 0)

    def test_complex_repeat_uses_cache_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            llm = CountingLLM()
            service = ChatService(
                PlannedRetriever(QueryIntent.FTMM, [HISTORY]), ContextBuilder(), llm,
                answer_cache=cache(directory),
            )
            first = service.answer("Jelaskan sejarah FTMM secara ringkas?")
            second = service.answer("  jelaskan sejarah ftmm secara ringkas  ")
            self.assertEqual(first.answer_path, AnswerPath.LLM)
            self.assertEqual(second.answer_path, AnswerPath.CACHE)
            self.assertEqual(llm.calls, 1)
            self.assertEqual(normalize_cache_query("Q?"), normalize_cache_query(" q "))

    def test_cache_invalidates_on_corpus_prompt_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ContextBuilder().build_grounded_context([HISTORY]).sources["S1"]
            answer = GroundedAnswer(
                "Fakta [S1]", [source], [source], True, Answerability.ANSWERABLE,
                retrieval_metadata={"provider": "openai", "requested_model": "gpt-4o-mini"},
                grounding_status=GroundingStatus.GROUNDED, answer_path=AnswerPath.LLM,
            )
            original = cache(directory)
            self.assertTrue(original.save("Jelaskan sejarah", answer))
            self.assertIsNotNone(original.load("Jelaskan sejarah"))
            self.assertIsNone(cache(directory, corpus="changed").load("Jelaskan sejarah"))
            self.assertIsNone(cache(directory, prompt="prompt-v2").load("Jelaskan sejarah"))
            self.assertIsNone(cache(directory, model="specific/free").load("Jelaskan sejarah"))

    def test_daily_budget_resets_by_date_and_blocks_at_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = DailyLLMBudgetGuard(Path(directory) / "counter.json", daily_call_limit=1)
            self.assertEqual(guard.reserve_call(), 1)
            with self.assertRaises(LLMBudgetExceeded):
                guard.reserve_call()
            self.assertEqual(guard.current()["llm_calls"], 1)

    def test_limit_message_keeps_sources_without_fake_citation(self):
        llm = CountingLLM(error=LLMBudgetExceeded("limit"))
        result = ChatService(PlannedRetriever(QueryIntent.FTMM, [HISTORY]), ContextBuilder(), llm).answer(
            "Jelaskan sejarah FTMM secara ringkas."
        )
        self.assertEqual(result.grounding_status, GroundingStatus.BUDGET_LIMITED)
        self.assertEqual(result.citations, [])
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(llm.calls, 1)

    def test_telemetry_reports_bypass_rate(self):
        telemetry = LLMTelemetry()
        local = ChatService(PlannedRetriever(QueryIntent.STAFF, [STAFF]), ContextBuilder(), CountingLLM(), telemetry=telemetry)
        local.answer("Siapa Dekan FTMM?")
        local.answer("Berapa harga Bitcoin?")
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot.total_questions, 2)
        self.assertEqual(snapshot.local_answers, 1)
        self.assertEqual(snapshot.no_answers, 1)
        self.assertEqual(snapshot.bypass_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
