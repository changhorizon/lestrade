from typing import List, Optional

from pydantic import BaseModel


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageModel]
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 2048


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "system"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class IngestRequest(BaseModel):
    text: str
    source: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
