import unittest

from domain.models import DocumentChunk, QueryIntent, RetrievedDocument, RetrievalPlan


class SmokeRetriever:
    def __init__(self):
        self.calls = 0
        self.last_retrieval_plan = RetrievalPlan(
            query="fixture", intent=QueryIntent.FTMM
        )

    def retrieve(self, query):
        self.calls += 1
        self.last_retrieval_plan = RetrievalPlan(query=query, intent=QueryIntent.FTMM)
        return [
            RetrievedDocument(
                DocumentChunk(
                    "ftmm::history",
                    "ftmm",
                    "Informasi FTMM | Sejarah Fakultas\nFTMM didirikan berdasarkan keputusan universitas.",
                    0,
                    {"type": "ftmm", "name": "FTMM", "section": "Sejarah Fakultas"},
                ),
                score=0.9,
            )
        ]


class OpenAIMissingKeyFlaskSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app

        cls.production_app = app

    def test_get_and_complex_post_are_http_200_without_key(self):
        provider = self.production_app.llm_service.provider
        original_key = provider.api_key
        original_retriever = self.production_app.chat_service.retriever
        smoke_retriever = SmokeRetriever()
        provider.api_key = None
        self.production_app.chat_service.retriever = smoke_retriever
        try:
            client = self.production_app.app.test_client()
            self.assertEqual(client.get("/").status_code, 200)
            response = client.post(
                "/",
                data={"query": "Jelaskan sejarah FTMM secara ringkas."},
                follow_redirects=True,
            )
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn("layanan penyusunan jawaban AI belum tersedia", html)
            self.assertIn("Sumber", html)
            self.assertEqual(smoke_retriever.calls, 1)
        finally:
            provider.api_key = original_key
            self.production_app.chat_service.retriever = original_retriever


if __name__ == "__main__":
    unittest.main()
