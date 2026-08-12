from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lestrade.main import app
from lestrade.ratelimit import _store


@pytest.fixture(autouse=True)
def clear_rate_limit():
    _store.clear()
    yield
    _store.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_models(client):
    with patch("lestrade.llm.list_all_models") as mock_list:
        mock_list.return_value = ["ollama/qwen2", "gpt-4o"]
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "ollama/qwen2" in ids


def test_chat_config(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    assert "model" in resp.json()


class TestChatCompletionsJSON:
    def test_chat_json_success(self, client):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value="This is the answer.")

        with patch("lestrade.router.chat.get_llm") as mock_get_llm:
            mock_get_llm.return_value = (mock_llm, "test-model")

            with patch("lestrade.router.chat.engine.search") as mock_search:
                mock_search.return_value = []

                resp = client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "What is this?"}],
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "This is the answer."

    def test_chat_json_with_context(self, client):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value="Answer with context.")

        with patch("lestrade.router.chat.get_llm") as mock_get_llm:
            mock_get_llm.return_value = (mock_llm, "test-model")

            with patch("lestrade.router.chat.engine.search") as mock_search:
                mock_search.return_value = [
                    ("Relevant content here.", "en/doc.md", 0.1),
                    ("More relevant info.", "en/doc2.md", 0.15),
                ]

                resp = client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Tell me about X."}],
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Answer with context."

    def test_chat_json_llm_error(self, client):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("API error"))

        with patch("lestrade.router.chat.get_llm") as mock_get_llm:
            mock_get_llm.return_value = (mock_llm, "test-model")

            with patch("lestrade.router.chat.engine.search") as mock_search:
                mock_search.return_value = []

                resp = client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                })

        assert resp.status_code == 502

    def test_chat_json_fallback_when_no_context(self, client, monkeypatch):
        monkeypatch.setattr("lestrade.config.RATE_LIMIT_MAX_REQUESTS", 100)

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value="I don't know.")

        with patch("lestrade.router.chat.get_llm") as mock_get_llm:
            mock_get_llm.return_value = (mock_llm, "test-model")

            with patch("lestrade.router.chat.engine.search") as mock_search:
                mock_search.return_value = []

                resp = client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Unknown topic?"}],
                })

        assert resp.status_code == 200

    def test_chat_json_search_error_handled(self, client, monkeypatch):
        monkeypatch.setattr("lestrade.config.RATE_LIMIT_MAX_REQUESTS", 100)

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value="Answer anyway.")

        with patch("lestrade.router.chat.get_llm") as mock_get_llm:
            mock_get_llm.return_value = (mock_llm, "test-model")

            with patch("lestrade.router.chat.engine.search") as mock_search:
                mock_search.side_effect = Exception("Search failed")

                resp = client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                })

        assert resp.status_code == 200


class TestChatCompletionsStream:
    def test_chat_stream(self, client, monkeypatch):
        monkeypatch.setattr("lestrade.config.RATE_LIMIT_MAX_REQUESTS", 100)

        async def mock_stream(*args, **kwargs):
            yield "Hello"
            yield " world"

        mock_llm = MagicMock()
        mock_llm.chat_stream = mock_stream

        with patch("lestrade.router.chat.get_llm") as mock_get_llm:
            mock_get_llm.return_value = (mock_llm, "test-model")

            with patch("lestrade.router.chat.engine.search") as mock_search:
                mock_search.return_value = []

                resp = client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


class TestHTMLFormat:
    def test_format_html_bullets(self):
        from lestrade.plugins.response import DefaultResponse
        plugin = DefaultResponse()
        result = plugin.format_response("- Item 1\n- Item 2")
        assert "<ul>" in result
        assert "<li>Item 1</li>" in result
        assert "<li>Item 2</li>" in result

    def test_format_html_disabled(self):
        from lestrade.plugins.response import DefaultResponse, set_html_format
        set_html_format(False)
        plugin = DefaultResponse()
        result = plugin.format_response("- Item 1\n- Item 2")
        assert result == "- Item 1\n- Item 2"

    def test_format_html_no_list(self):
        from lestrade.plugins.response import DefaultResponse
        plugin = DefaultResponse()
        result = plugin.format_response("Plain text without lists.")
        assert result == "Plain text without lists."
