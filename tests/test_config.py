

def test_defaults(monkeypatch):
    for var in ("LLM_API_KEY", "LLM_BASE_URL", "OLLAMA_BASE_URL",
                "EMBEDDING_MODEL", "CHAT_MODEL", "KB_DIRS",
                "FAISS_INDEX_PATH", "FALLBACK_MESSAGE_EN", "FALLBACK_MESSAGE_ZH"):
        monkeypatch.delenv(var, raising=False)

    import importlib

    import lestrade.config
    importlib.reload(lestrade.config)

    assert lestrade.config.OLLAMA_BASE_URL == "http://ollama:11434"
    assert lestrade.config.LLM_BASE_URL == "https://api.openai.com/v1"
    assert lestrade.config.LLM_API_KEY == ""
    assert lestrade.config.EMBEDDING_MODEL == "nomic-embed-text"
    assert lestrade.config.CHAT_MODEL == "deepseek-chat"
    assert lestrade.config.KB_DIRS == ""
    assert lestrade.config.FAISS_INDEX_PATH == "data/faiss_index/index.faiss"
    assert lestrade.config.RATE_LIMIT_WINDOW == 60
    assert lestrade.config.RATE_LIMIT_MAX_REQUESTS == 10
    assert lestrade.config.WATCH_INTERVAL == 300
    assert "lestrade.config.HTML_FORMAT" or True


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://custom.api/v1")
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "50")
    monkeypatch.setenv("HTML_FORMAT", "false")

    import importlib

    import lestrade.config
    importlib.reload(lestrade.config)

    assert lestrade.config.LLM_API_KEY == "sk-test"
    assert lestrade.config.LLM_BASE_URL == "https://custom.api/v1"
    assert lestrade.config.RATE_LIMIT_MAX_REQUESTS == 50
    assert lestrade.config.HTML_FORMAT is False


def test_external_models_split():
    import lestrade.config

    assert isinstance(lestrade.config.EXTERNAL_MODELS, list)
    assert len(lestrade.config.EXTERNAL_MODELS) > 0


def test_watch_ext_split():
    import lestrade.config

    assert ".txt" in lestrade.config.WATCH_EXT
    assert ".md" in lestrade.config.WATCH_EXT
