from lestrade.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageModel,
    IngestRequest,
    ModelInfo,
    ModelList,
)


class TestSchemas:
    def test_chat_message(self):
        msg = ChatMessageModel(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_completion_request_defaults(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessageModel(role="user", content="Hi")],
        )
        assert req.temperature == 0.0
        assert req.stream is False
        assert req.max_tokens == 2048

    def test_chat_completion_request_stream(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessageModel(role="user", content="Hi")],
            stream=True,
            temperature=0.5,
        )
        assert req.stream is True
        assert req.temperature == 0.5

    def test_model_info(self):
        mi = ModelInfo(id="gpt-4o")
        assert mi.id == "gpt-4o"
        assert mi.object == "model"
        assert mi.owned_by == "system"

    def test_model_list(self):
        ml = ModelList(data=[ModelInfo(id="a"), ModelInfo(id="b")])
        assert ml.object == "list"
        assert len(ml.data) == 2

    def test_ingest_request(self):
        req = IngestRequest(text="Doc content", source="doc.md")
        assert req.text == "Doc content"
        assert req.source == "doc.md"

    def test_chat_completion_response(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
