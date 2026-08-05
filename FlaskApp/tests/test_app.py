import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module


def test_home_page():
    client = app_module.app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_faq_answer():
    client = app_module.app.test_client()
    response = client.post("/ask", json={"question": "What is Flask?"})
    assert response.status_code == 200
    data = response.get_json()
    assert "Flask" in data["answer"]


def test_local_ollama_answer(monkeypatch):
    calls = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "Hello from local Ollama"}

    def fake_post(url, headers=None, json=None, timeout=20):
        calls["url"] = url
        calls["json"] = json
        return DummyResponse()

    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:270m")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(app_module.requests, "post", fake_post)

    result = app_module.get_ai_answer("Tell me about Flask")

    assert result == "Hello from local Ollama"
    assert calls["url"] == "http://127.0.0.1:11434/api/generate"
    assert calls["json"]["model"] == "gemma3:270m"
