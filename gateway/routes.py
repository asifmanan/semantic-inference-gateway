# routes.py
# OpenAI-compatible /v1/chat/completions endpoint with tiered cache resolution (Redis L1 -> pgvector L2 -> upstream LLM).
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable

import asyncpg
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from cache.embeddings import async_encode_prompt
from cache.redis_client import get_exact_match, set_exact_match
from cache.semantic import insert_semantic_cache, query_semantic_cache
from core.config import get_settings
from gateway.upstream import UpstreamInferenceError, UpstreamTimeoutError

logger = logging.getLogger(__name__)

router = APIRouter()

# Failures that mean "dependency unreachable or unhealthy"; anything else is a bug and should surface.
REDIS_ERRORS = (RedisError, OSError, TimeoutError)
POSTGRES_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError)


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


async def ingest_cache_miss(
    prompt: str,
    response_text: str,
    model_name: str,
    pool: asyncpg.Pool,
    ttl: int = 3600,
) -> None:
    """Write an upstream completion into L1 and L2. Fail-open: never raises."""
    if not response_text or not response_text.strip():
        return

    try:
        await set_exact_match(prompt, response_text, model_name, ttl=ttl)
    except REDIS_ERRORS as exc:
        logger.warning("Cache ingestion: Redis L1 write failed: %r", exc)

    try:
        vector = await async_encode_prompt(prompt)
        await insert_semantic_cache(prompt, vector, response_text, model_name, pool)
    except (*POSTGRES_ERRORS, ValueError, RuntimeError) as exc:
        logger.warning("Cache ingestion: pgvector L2 write failed: %r", exc)


def _extract_content(completion: dict) -> str:
    try:
        return completion["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


async def _sse_stream(
    tokens: AsyncGenerator[str, None],
    first_token: str | None,
    model: str,
    on_complete: Callable[[str], None],
) -> AsyncGenerator[str, None]:
    """Re-emit upstream tokens as OpenAI chat.completion.chunk SSE events.

    on_complete receives the full text only if the upstream stream finished cleanly.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    parts: list[str] = []
    try:
        yield _sse(_chunk(completion_id, model, {"role": "assistant", "content": ""}))
        if first_token is not None:
            parts.append(first_token)
            yield _sse(_chunk(completion_id, model, {"content": first_token}))
            async for token in tokens:
                parts.append(token)
                yield _sse(_chunk(completion_id, model, {"content": token}))
        yield _sse(_chunk(completion_id, model, {}, finish_reason="stop"))
        on_complete("".join(parts))
    except UpstreamInferenceError as exc:
        # Headers are already sent, so the status code can't change; surface the failure in-band.
        # Partial output is deliberately not passed to on_complete.
        logger.error("Upstream failed mid-stream: %s", exc)
        yield _sse({"error": {"message": str(exc), "type": type(exc).__name__}})
    finally:
        await tokens.aclose()
    yield _sse("[DONE]")


def _upstream_http_error(exc: UpstreamInferenceError) -> HTTPException:
    status = 504 if isinstance(exc, UpstreamTimeoutError) else 502
    return HTTPException(status_code=status, detail=str(exc))


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest, request: Request, background_tasks: BackgroundTasks
):
    state = request.app.state
    settings = get_settings()
    prompt = extract_prompt(body.messages)

    # Tier 1: Redis exact match (fail-open)
    try:
        hit = await get_exact_match(prompt)
    except REDIS_ERRORS as exc:
        logger.warning("Redis L1 lookup failed, falling through to L2: %r", exc)
        hit = None
    if hit is not None:
        return JSONResponse(
            build_cached_completion(hit, body.model), headers={"X-Cache-Status": "HIT-L1"}
        )

    # Tier 2: pgvector semantic match (fail-open)
    try:
        vector = await async_encode_prompt(prompt)
        hit = await query_semantic_cache(prompt, vector, state.pool, body.similarity_threshold)
    except POSTGRES_ERRORS as exc:
        logger.warning("pgvector L2 lookup failed, falling through to upstream: %r", exc)
        hit = None
    if hit is not None:
        return JSONResponse(
            build_cached_completion(hit, body.model), headers={"X-Cache-Status": "HIT-L2"}
        )

    # Tier 3: upstream LLM, then write the result back to both cache tiers in the background
    messages = [m.model_dump() for m in body.messages]
    kwargs = {} if body.temperature is None else {"temperature": body.temperature}
    headers = {"X-Cache-Status": "MISS"}
    resolved_model = body.model or state.upstream.model

    def schedule_ingestion(response_text: str) -> None:
        background_tasks.add_task(
            ingest_cache_miss,
            prompt,
            response_text,
            resolved_model,
            state.pool,
            settings.cache_ttl_seconds,
        )

    if not body.stream:
        try:
            completion = await state.upstream.generate_completion(messages, body.model, **kwargs)
        except UpstreamInferenceError as exc:
            raise _upstream_http_error(exc) from exc
        schedule_ingestion(_extract_content(completion))
        return JSONResponse(completion, headers=headers, background=background_tasks)

    tokens = state.upstream.stream_completion(messages, body.model, **kwargs)
    try:
        # Pull the first token before responding so connect/HTTP failures still map to 502/504.
        first_token = await anext(tokens, None)
    except UpstreamInferenceError as exc:
        raise _upstream_http_error(exc) from exc
    return StreamingResponse(
        _sse_stream(tokens, first_token, resolved_model, schedule_ingestion),
        media_type="text/event-stream",
        headers=headers,
        background=background_tasks,
    )
