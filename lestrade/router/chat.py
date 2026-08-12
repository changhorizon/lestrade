import json
import re
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import config
from ..llm import get_llm
from ..models.schemas import ChatCompletionRequest, ChatCompletionResponse
from ..plugins import get_response, set_html_format
from ..rag.engine import engine

_RE_ZH = re.compile(r'[\u4e00-\u9fff]')

_response = get_response()
set_html_format(config.HTML_FORMAT)


def _detect_lang(req) -> str:
    msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    return 'zh' if _RE_ZH.search(msg) else 'en'


router = APIRouter()


@router.get("/config")
async def chat_config():
    return {"model": config.CHAT_MODEL}


_K = 5


def _sse_chunk(chunk_id, created, model, delta, finish_reason=None):
    choice = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return _handle_stream(req)
    return await _handle_json(req)


def _handle_stream(req):
    llm, resolved_model = get_llm(req.model)
    lang = _detect_lang(req)

    async def generate():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        try:
            contexts = await engine.search(user_message, k=_K, lang=lang)
        except Exception:
            contexts = []
        msgs = _response.build_messages(contexts, user_message, lang)

        yield _sse_chunk(chunk_id, created, req.model, {"role": "assistant"})

        tokens = []
        try:
            async for token in llm.chat_stream(
                model=resolved_model,
                messages=msgs,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                tokens.append(token)
                yield _sse_chunk(chunk_id, created, req.model, {"content": token})
        except Exception:
            yield _sse_chunk(chunk_id, created, req.model, {}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        raw = "".join(tokens)
        formatted = _response.format_response(raw)
        if formatted != raw:
            yield _sse_chunk(chunk_id, created, req.model, {"content": formatted})

        yield _sse_chunk(chunk_id, created, req.model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _handle_json(req):
    llm, resolved_model = get_llm(req.model)
    lang = _detect_lang(req)

    user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    try:
        contexts = await engine.search(user_message, k=_K, lang=lang)
    except Exception:
        contexts = []
    msgs = _response.build_messages(contexts, user_message, lang)

    try:
        content = await llm.chat(
            model=resolved_model,
            messages=msgs,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(e)}")
    content = _response.format_response(content)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


@router.get("/v1/models")
async def list_models():
    from ..llm import list_all_models
    models = await list_all_models()
    from ..models.schemas import ModelInfo, ModelList
    return ModelList(data=[ModelInfo(id=m) for m in models])
