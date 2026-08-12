from abc import ABC, abstractmethod
from typing import AsyncIterator, List


class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class BaseLLM(ABC):
    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        pass

    async def chat_stream(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        content = await self.chat(model, messages, temperature, max_tokens)
        yield content

    @abstractmethod
    async def list_models(self) -> List[str]:
        pass
