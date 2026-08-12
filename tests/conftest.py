import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr("lestrade.config.FAISS_INDEX_PATH",
                       os.path.join(tempfile.mkdtemp(), "test_index.faiss"))
    monkeypatch.setattr("lestrade.config.KB_DIRS", "")
    monkeypatch.setattr("lestrade.config.OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr("lestrade.config.LLM_API_KEY", "")
    monkeypatch.setattr("lestrade.config.LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr("lestrade.config.EMBEDDING_MODEL", "bge-m3")
    monkeypatch.setattr("lestrade.config.CHAT_MODEL", "test-model")
    monkeypatch.setattr("lestrade.config.EXTERNAL_MODELS", ["test-model"])
    monkeypatch.setattr("lestrade.config.RATE_LIMIT_WINDOW", 60)
    monkeypatch.setattr("lestrade.config.RATE_LIMIT_MAX_REQUESTS", 3)
    monkeypatch.setattr("lestrade.config.FALLBACK_MESSAGE_EN", "No info found.")
    monkeypatch.setattr("lestrade.config.FALLBACK_MESSAGE_ZH", "未找到信息。")
    monkeypatch.setattr("lestrade.config.HTML_FORMAT", True)
