import unittest

from flask import Flask, render_template


class GroundedUITests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__, template_folder="../templates", static_folder="../static")
        self.app.add_url_rule("/evaluate", "evaluate", lambda: "ok")
        self.app.add_url_rule("/reset", "reset", lambda: "ok", methods=["POST"])

        @self.app.route("/")
        def page():
            return render_template(
                "index.html",
                current_year=2026,
                dataset_errors=[],
                form_error=None,
                chat_history=[{
                    "user": "<script>alert(1)</script>",
                    "bot": "Dekan FTMM [S1] <img src=x onerror=alert(2)>",
                    "answerability": "answerable",
                    "grounding_status": "grounded",
                    "warning": None,
                    "sources": [{
                        "source_id": "S1",
                        "chunk_id": "staff::chunk-0000",
                        "title": "Dekan <script>alert(3)</script>",
                        "source_type": "staff",
                        "section": "Profil",
                        "url": None,
                        "supporting_excerpt": "Teks <script>alert(4)</script>",
                    }],
                }],
            )

    def test_answer_markers_and_source_cards_render_without_xss(self):
        html = self.app.test_client().get("/").get_data(as_text=True)
        self.assertIn("Dekan FTMM [S1]", html)
        self.assertIn("Sumber", html)
        self.assertIn("staff::chunk-0000", html)
        self.assertNotIn("<script>alert", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
