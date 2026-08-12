from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lestrade.llm import get_llm
from lestrade.llm.base import BaseLLM, ChatMessage
from lestrade.llm.ollama import OllamaLLM
from lestrade.llm.openai_api import OpenAILLM


class TestBaseLLM:
    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_base_llm_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLLM()


class TestLLMRouting:
    def test_get_llm_ollama_prefix(self):
        llm, model = get_llm("ollama/qwen2:1.5b")
        assert isinstance(llm, OllamaLLM)
        assert model == "qwen2:1.5b"

    def test_get_llm_local_prefix(self):
        llm, model = get_llm("local/llama3")
        assert isinstance(llm, OllamaLLM)
        assert model == "llama3"

    def test_get_llm_openai_default(self):
        llm, model = get_llm("gpt-4o")
        assert isinstance(llm, OpenAILLM)
        assert model == "gpt-4o"

    def test_get_llm_deepseek(self):
        llm, model = get_llm("deepseek-chat")
        assert isinstance(llm, OpenAILLM)
        assert model == "deepseek-chat"


class TestOpenAILLM:
    @pytest.mark.asyncio
    async def test_chat_success(self):
        llm = OpenAILLM()
        messages = [ChatMessage(role="user", content="Hi")]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await llm.chat("gpt-4o", messages)
            assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_list_models(self, monkeypatch):
        monkeypatch.setattr("lestrade.config.EXTERNAL_MODELS", ["gpt-4o", "gpt-4o-mini"])
        llm = OpenAILLM()
        models = await llm.list_models()
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models


class TestOllamaLLM:
    @pytest.mark.asyncio
    async def test_chat_success(self):
        llm = OllamaLLM()
        messages = [ChatMessage(role="user", content="Hi")]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Ollama!"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await llm.chat("qwen2", messages)
            assert result == "Hello from Ollama!"

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        llm = OllamaLLM()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "qwen2:1.5b"}, {"name": "bge-m3"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            models = await llm.list_models()
            assert "qwen2:1.5b" in models
            assert "bge-m3" in models

    @pytest.mark.asyncio
    async def test_list_models_failure_returns_empty(self):
        llm = OllamaLLM()
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=Exception("Down"))
            models = await llm.list_models()
            assert models == []
