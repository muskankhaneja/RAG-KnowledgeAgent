import os
import unittest
os.environ.setdefault("MOCK_LLM", "1")
os.environ.setdefault("HF_ACCESS_TOKEN", "mock")

from fastapi.testclient import TestClient
from src.agent.app import app


class TestRAGAgentApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # ── UI / static ────────────────────────────────────────────────────────────

    def test_root_returns_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("<html", response.text.lower())

    def test_static_js_served(self):
        response = self.client.get("/static/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("javascript", response.headers["content-type"])

    def test_static_config_js_served(self):
        response = self.client.get("/static/config.js")
        self.assertEqual(response.status_code, 200)

    def test_favicon_no_error(self):
        response = self.client.get("/favicon.ico")
        self.assertIn(response.status_code, [200, 204, 404])

    # ── /chat ──────────────────────────────────────────────────────────────────

    def test_chat_full_context(self):
        """Full-context mode: chunks sent directly, mock LLM returns answer."""
        response = self.client.post("/chat", json={
            "query": "What does the add function do?",
            "chunks": [
                {"id": "c1", "text": "def add(a, b): return a + b", "source": "main.py"}
            ],
            "history": []
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)

    def test_chat_bm25_candidates(self):
        """BM25 mode: candidates field instead of chunks."""
        response = self.client.post("/chat", json={
            "query": "Explain the algorithm",
            "candidates": [
                {"id": "c2", "text": "BM25 is a ranking function.", "source": "doc.txt"}
            ],
            "history": []
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", response.json())

    def test_chat_empty_context(self):
        """No chunks or candidates — server should still return an answer."""
        response = self.client.post("/chat", json={
            "query": "Hello?",
            "chunks": [],
            "history": []
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", response.json())

    def test_chat_with_history(self):
        """Chat history is accepted without error."""
        response = self.client.post("/chat", json={
            "query": "Follow-up question",
            "chunks": [{"id": "c3", "text": "Some context.", "source": "file.md"}],
            "history": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"}
            ]
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", response.json())

    # ── /ingest/questions ──────────────────────────────────────────────────────

    def test_ingest_questions_returns_list(self):
        """Mock LLM path: each chunk gets a questions list."""
        response = self.client.post("/ingest/questions", json={
            "chunks": [
                {"id": "q1", "text": "Python is a programming language.", "source": "intro.md"}
            ]
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertIsInstance(data["results"], list)
        self.assertEqual(len(data["results"]), 1)
        self.assertIn("questions", data["results"][0])

    def test_ingest_questions_multiple_chunks(self):
        response = self.client.post("/ingest/questions", json={
            "chunks": [
                {"id": "a", "text": "Chunk A text.", "source": "a.md"},
                {"id": "b", "text": "Chunk B text.", "source": "b.md"},
            ]
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 2)

    # ── /fetch/github ──────────────────────────────────────────────────────────

    def test_fetch_github_invalid_url_rejected(self):
        """Bare non-git URLs (no scheme) should be rejected with 400."""
        response = self.client.post("/fetch/github", json={
            "url": "notaurl"
        })
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
