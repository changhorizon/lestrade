from typing import List

import httpx

from .. import config
from .base import BaseLLM, ChatMessage


class OpenAILLM(BaseLLM):
    def __init__(self):
        self.base_url = config.LLM_BASE_URL.rstrip("/")
        self.api_key = config.LLM_API_KEY

    async def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def list_models(self) -> List[str]:
        return config.EXTERNAL_MODELS
