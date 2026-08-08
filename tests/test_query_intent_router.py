import unittest

from domain.models import IntentConfidence, QueryIntent, RetrievalPlan
from services.query_intent_router import QueryIntentRouter


class QueryIntentRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = QueryIntentRouter(
            available_sections=["Visi", "Misi", "Sejarah Fakultas", "Maskot Cirion"],
            lecturer_names=["Dr. Maryamah, S.Kom."],
        )

    def assert_intent(self, query, expected):
        plan = self.router.route(query)
        self.assertIsInstance(plan, RetrievalPlan)
        self.assertEqual(plan.intent, expected)
        return plan

    def test_explicit_domain_intents(self):
        for query in ("dosen yang fokus NLP", "pengajar yang meneliti AI"):
            plan = self.assert_intent(query, QueryIntent.LECTURER)
            self.assertEqual(plan.metadata_filter["chunk_role"], "research_interest")
        for query in ("mata kuliah tentang NLP", "berapa SKS machine learning"):
            self.assert_intent(query, QueryIntent.COURSE)
        self.assert_intent("siapa dekan FTMM", QueryIntent.STAFF)
        self.assert_intent("cara mengajukan surat aktif", QueryIntent.ACADEMIC)

    def test_ftmm_section_is_matched_from_available_metadata(self):
        visi = self.assert_intent("apa visi FTMM", QueryIntent.FTMM)
        misi = self.assert_intent("apa misi FTMM", QueryIntent.FTMM)
        self.assertEqual(visi.metadata_filter, {"type": "ftmm", "section": "Visi"})
        self.assertEqual(misi.metadata_filter, {"type": "ftmm", "section": "Misi"})
        maskot = self.assert_intent("apa nama maskot FTMM", QueryIntent.FTMM)
        self.assertEqual(maskot.metadata_filter["section"], "Maskot Cirion")

    def test_topic_only_queries_are_general(self):
        for query in ("NLP", "machine learning", "AI", "robotika", "data science"):
            plan = self.assert_intent(query, QueryIntent.GENERAL)
            self.assertIsNone(plan.metadata_filter)
            self.assertEqual(plan.confidence, IntentConfidence.LOW)

    def test_profile_name_uses_lecturer_type_without_research_role(self):
        plan = self.assert_intent("profil Maryamah", QueryIntent.LECTURER)
        self.assertEqual(plan.metadata_filter, {"type": "lecturer"})
        self.assertIsNone(plan.preferred_chunk_role)

    def test_mixed_lecturer_course_request_uses_documented_precedence(self):
        lecturer = self.assert_intent(
            "dosen pengampu mata kuliah machine learning", QueryIntent.LECTURER
        )
        self.assertEqual(lecturer.metadata_filter, {"type": "lecturer"})
        ambiguous = self.assert_intent(
            "dosen atau mata kuliah machine learning", QueryIntent.GENERAL
        )
        self.assertIsNone(ambiguous.metadata_filter)


if __name__ == "__main__":
    unittest.main()
