import os
import unittest

from domain.models import AnswerPath


@unittest.skipUnless(
    os.getenv("RUN_OPENAI_INTEGRATION", "").lower() == "true"
    and os.getenv("OPENAI_API_KEY"),
    "Set RUN_OPENAI_INTEGRATION=true dan OPENAI_API_KEY untuk test aktual.",
)
class OpenAIGroundedIntegrationTests(unittest.TestCase):
    def test_complex_grounded_queries_capture_model_usage_and_latency(self):
        import app

        cases = (
            "Jelaskan sejarah FTMM secara ringkas.",
            "Apa visi FTMM?",
            "Siapa dosen yang meneliti NLP?",
            "Bagaimana mengajukan SKMA?",
        )
        for query in cases:
            with self.subTest(query=query):
                result = app.chat_service.answer(query)
                self.assertTrue(result.answer.strip())
                if result.answer_path == AnswerPath.LLM:
                    self.assertTrue(result.sources)
                    self.assertEqual(result.retrieval_metadata.get("provider"), "openai")
                    self.assertTrue(result.retrieval_metadata.get("requested_model"))
                    self.assertTrue(result.retrieval_metadata.get("actual_model"))
                    self.assertGreaterEqual(result.retrieval_metadata["timing_ms"]["llm"], 0)


if __name__ == "__main__":
    unittest.main()
