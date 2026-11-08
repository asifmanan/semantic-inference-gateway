# routes.py
# OpenAI-compatible /v1/chat/completions endpoint with tiered cache resolution (Redis L1 -> pgvector L2 -> upstream LLM).
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from cache.embeddings import async_encode_prompt
from cache.redis_client import get_exact_match
from cache.semantic import query_semantic_cache
from gateway.upstream import UpstreamInferenceError, UpstreamTimeoutError

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    stream: bool | None = False
    similarity_threshold: float | None = Field(default=0.90, ge=0.0, le=1.0)


def extract_prompt(messages: list[ChatMessage]) -> str:
    """The cache key is the content of the last user message."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    raise HTTPException(status_code=422, detail="messages must contain at least one 'user' message")


def build_cached_completion(hit: dict, requested_model: str | None) -> dict:
    """Wrap a cached response in an OpenAI-style chat.completion body."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": hit.get("model_name") or requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": hit["response"]},
                "finish_reason": "stop",
            }
        ],
    }


def _sse(data: dict | str) -> str:
    return f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"


def _chunk(completion_id: str, model: str, delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _sse_stream(
    tokens: AsyncGenerator[str, None], first_token: str | None, model: str
) -> AsyncGenerator[str, None]:
    """Re-emit upstream tokens as OpenAI chat.completion.chunk SSE events."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    try:
        yield _sse(_chunk(completion_id, model, {"role": "assistant", "content": ""}))
        if first_token is not None:
            yield _sse(_chunk(completion_id, model, {"content": first_token}))
            async for token in tokens:
                yield _sse(_chunk(completion_id, model, {"content": token}))
        yield _sse(_chunk(completion_id, model, {}, finish_reason="stop"))
    except UpstreamInferenceError as exc:
        # Headers are already sent, so the status code can't change; surface the failure in-band.
        logger.error("Upstream failed mid-stream: %s", exc)
        yield _sse({"error": {"message": str(exc), "type": type(exc).__name__}})
    finally:
        await tokens.aclose()
    yield _sse("[DONE]")


def _upstream_http_error(exc: UpstreamInferenceError) -> HTTPException:
    status = 504 if isinstance(exc, UpstreamTimeoutError) else 502
    return HTTPException(status_code=status, detail=str(exc))


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    state = request.app.state
    prompt = extract_prompt(body.messages)

    # Tier 1: Redis exact match
    hit = await get_exact_match(prompt)
    if hit is not None:
        return JSONResponse(
            build_cached_completion(hit, body.model), headers={"X-Cache-Status": "HIT-L1"}
        )

    # Tier 2: pgvector semantic match
    vector = await async_encode_prompt(prompt)
    hit = await query_semantic_cache(prompt, vector, state.pool, body.similarity_threshold)
    if hit is not None:
        return JSONResponse(
            build_cached_completion(hit, body.model), headers={"X-Cache-Status": "HIT-L2"}
        )

    # Tier 3: upstream LLM
    messages = [m.model_dump() for m in body.messages]
    kwargs = {} if body.temperature is None else {"temperature": body.temperature}
    headers = {"X-Cache-Status": "MISS"}

    if not body.stream:
        try:
            completion = await state.upstream.generate_completion(messages, body.model, **kwargs)
        except UpstreamInferenceError as exc:
            raise _upstream_http_error(exc) from exc
        return JSONResponse(completion, headers=headers)

    tokens = state.upstream.stream_completion(messages, body.model, **kwargs)
    try:
        # Pull the first token before responding so connect/HTTP failures still map to 502/504.
        first_token = await anext(tokens, None)
    except UpstreamInferenceError as exc:
        raise _upstream_http_error(exc) from exc
    model = body.model or state.upstream.model
    return StreamingResponse(
        _sse_stream(tokens, first_token, model), media_type="text/event-stream", headers=headers
    )
