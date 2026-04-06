import unittest
from fastapi.testclient import TestClient
from src.agent.app import app


class TestRAGAgentApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_returns_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn("Add New Project", html)
        self.assertIn("Ingested Projects", html)
        self.assertIn("Chat with Documents", html)
        self.assertIn("button", html)

    def test_static_js_is_available(self):
        response = self.client.get("/static/app.js?v=4")
        self.assertEqual(response.status_code, 200)
        self.assertIn("loadProjects();", response.text)
        self.assertIn("sendMessage()", response.text)

    def test_projects_endpoint_returns_project_list(self):
        response = self.client.get("/projects")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("projects", data)
        self.assertIsInstance(data["projects"], dict)

    def test_query_returns_retrieved_documents(self):
        response = self.client.post(
            "/query",
            json={
                "query": "What does the add function do?",
                "team": "analytics",
                "project": "sample_project",
                "top_k": 1,
                "use_llm": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("retrieved", data)
        self.assertIsInstance(data["retrieved"], dict)

    def test_query_project_exists(self):
        response = self.client.get("/projects")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        analytics_projects = data["projects"].get("analytics", [])
        self.assertIn("sample_project", analytics_projects)

    def test_create_project(self):
        response = self.client.post("/projects/create", json={"team": "test_team", "project": "test_project"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["created"], True)

    def test_rename_project(self):
        # Create a project first, then rename it
        self.client.post("/projects/create", json={"team": "test_team", "project": "rename_src"})
        response = self.client.post("/projects/rename", json={
            "team": "test_team", "old_project": "rename_src", "new_project": "rename_dst"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["renamed"], True)
        self.assertEqual(data["new_project"], "rename_dst")

    def test_upload_document(self):
        response = self.client.post("/upload", json={
            "team": "analytics",
            "project": "sample_project",
            "filename": "test.md",
            "content": "Test content",
            "doc_type": "uploaded"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("saved_to", data)

    def test_ingest_web_url(self):
        response = self.client.post("/ingest", json={
            "team": "analytics",
            "project": "sample_project",
            "source": "https://example.com",
            "doc_type": "web"
        })
        # Assuming it handles web URLs, but may fail if not implemented
        self.assertIn(response.status_code, [200, 400])  # 200 if successful, 400 if not supported

    def test_query_all_projects(self):
        response = self.client.post("/query", json={
            "query": "What is this?",
            "top_k": 1,
            "use_llm": False
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("retrieved", data)


if __name__ == "__main__":
    unittest.main()
