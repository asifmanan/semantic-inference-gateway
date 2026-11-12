# tests/test_gateway_api.py
# Tests for the FastAPI gateway skeleton: lifespan wiring and the /health endpoint.
import json
import logging
from contextlib import ExitStack
from unittest.mock import patch

import asyncpg
import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError

from cache.embeddings import MODEL_NAME, async_encode_prompt
from cache.redis_client import (
    get_exact_match,
    get_redis_client,
    hash_prompt,
    set_exact_match,
)
from cache.semantic import insert_semantic_cache
from gateway.app import app
from gateway.upstream import UpstreamLLMClient

TEST_MODEL = "gateway-api-test"


@pytest_asyncio.fixture
async def client():
    # ASGITransport does not send lifespan events, so run the lifespan explicitly.
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "database": "connected",
        "redis": "connected",
        "embedding_model": MODEL_NAME,
    }


@pytest.mark.asyncio
async def test_app_state_populated_and_operational(client):
    assert await app.state.redis.ping() is True
    async with app.state.pool.acquire() as conn:
        assert await conn.fetchval("SELECT 1;") == 1


@pytest.mark.asyncio
async def test_health_returns_503_when_dependency_fails(client, monkeypatch):
    async def boom():
        raise ConnectionError("redis down")

    monkeypatch.setattr(app.state.redis, "ping", boom)
    response = await client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["redis"] == "unavailable"
    assert body["database"] == "connected"


@pytest.mark.asyncio
async def test_startup_failure_releases_acquired_resources(monkeypatch):
    import gateway.app as gw

    closed = {"pool": False, "redis": False}

    class FakePool:
        async def close(self):
            closed["pool"] = True

    async def fake_init_db():
        return FakePool()

    async def failing_redis():
        raise ConnectionError("redis down")

    async def fake_close_redis():
        closed["redis"] = True

    monkeypatch.setattr(gw, "init_db", fake_init_db)
    monkeypatch.setattr(gw, "get_redis_client", failing_redis)
    monkeypatch.setattr(gw, "close_redis", fake_close_redis)

    with pytest.raises(ConnectionError):
        async with gw.lifespan(gw.app):
            pass
    assert closed == {"pool": True, "redis": True}


@pytest.mark.asyncio
async def test_pool_init_failure_does_not_mask_error(monkeypatch):
    import gateway.app as gw

    async def failing_init_db():
        raise OSError("postgres down")

    async def fake_close_redis():
        pass

    monkeypatch.setattr(gw, "init_db", failing_init_db)
    monkeypatch.setattr(gw, "close_redis", fake_close_redis)

    with pytest.raises(OSError, match="postgres down"):
        async with gw.lifespan(gw.app):
            pass


@pytest.mark.asyncio
async def test_health_returns_503_when_postgres_fails(client, monkeypatch):
    import asyncpg

    def broken_acquire(*args, **kwargs):
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(type(app.state.pool), "acquire", broken_acquire)
    response = await client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["database"] == "unavailable"
    assert body["redis"] == "connected"


@pytest.mark.asyncio
async def test_health_does_not_swallow_unexpected_errors(client, monkeypatch):
    async def bug():
        raise ValueError("programming error")

    monkeypatch.setattr(app.state.redis, "ping", bug)
    with pytest.raises(ValueError):
        await client.get("/health")


# --- /v1/chat/completions -------------------------------------------------------------------


def upstream_completion(content: str) -> dict:
    return {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


def sse_body(*tokens: str) -> bytes:
    events = [
        f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': t}}]})}\n\n" for t in tokens
    ]
    return ("".join(events) + "data: [DONE]\n\n").encode()


class FakeUpstream:
    """Configurable httpx.MockTransport handler that records the requests it receives."""

    def __init__(self):
        self.requests: list[dict] = []
        self.response = lambda request: httpx.Response(200, json=upstream_completion("upstream answer"))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return self.response(request)


@pytest_asyncio.fixture
async def chat(client):
    """Swap the lifespan's upstream client for one backed by a mock transport; clean test data around it."""
    fake = FakeUpstream()
    await app.state.upstream.aclose()
    app.state.upstream = UpstreamLLMClient(
        base_url="http://upstream.test/v1", model="mock-model", transport=httpx.MockTransport(fake)
    )
    await app.state.upstream.start()

    redis = await get_redis_client()
    seeded_prompts: list[str] = []

    async def cleanup():
        async with app.state.pool.acquire() as conn:
            await conn.execute("DELETE FROM semantic_cache WHERE model_name = $1", TEST_MODEL)
        if seeded_prompts:
            await redis.delete(*(hash_prompt(p) for p in seeded_prompts))

    await cleanup()

    class Ctx:
        upstream = fake
        track = seeded_prompts.append

        @staticmethod
        async def post(prompt: str, **extra):
            # Requests carry TEST_MODEL so rows ingested in the background are tagged and cleaned up.
            payload = {"model": TEST_MODEL, "messages": [{"role": "user", "content": prompt}], **extra}
            return await client.post("/v1/chat/completions", json=payload)

        @staticmethod
        async def db_rows() -> int:
            async with app.state.pool.acquire() as conn:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM semantic_cache WHERE model_name = $1", TEST_MODEL
                )

    try:
        yield Ctx
    finally:
        await cleanup()


def answer(response: httpx.Response) -> str:
    return response.json()["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_cold_query_misses_and_returns_upstream_response(chat):
    prompt = "Describe the migratory patterns of the Arctic tern."
    chat.track(prompt)
    response = await chat.post(prompt)

    assert response.status_code == 200
    assert response.headers["X-Cache-Status"] == "MISS"
    assert answer(response) == "upstream answer"
    assert len(chat.upstream.requests) == 1
    forwarded = chat.upstream.requests[0]
    assert forwarded["messages"] == [{"role": "user", "content": prompt}]
    assert forwarded["stream"] is False
    assert forwarded["temperature"] == 0.7


@pytest.mark.asyncio
async def test_seeded_redis_key_hits_l1_without_db_or_upstream(chat):
    prompt = "What is the capital of France?"
    chat.track(prompt)
    await set_exact_match(prompt, "Paris (from L1)", TEST_MODEL, ttl=60)

    with (
        patch("gateway.routes.async_encode_prompt") as mock_encode,
        patch("gateway.routes.query_semantic_cache") as mock_query,
    ):
        response = await chat.post("  WHAT is the   capital of france?\n")

    assert response.status_code == 200
    assert response.headers["X-Cache-Status"] == "HIT-L1"
    assert answer(response) == "Paris (from L1)"
    mock_encode.assert_not_called()
    mock_query.assert_not_called()
    assert chat.upstream.requests == []


@pytest.mark.asyncio
async def test_paraphrase_hits_l2_semantic_cache(chat):
    stored = "Tell me the capital of France"
    query = "What is the capital of France?"
    chat.track(stored)
    chat.track(query)
    await insert_semantic_cache(
        stored, await async_encode_prompt(stored), "Paris (from L2)", TEST_MODEL, app.state.pool
    )

    response = await chat.post(query, similarity_threshold=0.85)

    assert response.status_code == 200
    assert response.headers["X-Cache-Status"] == "HIT-L2"
    assert answer(response) == "Paris (from L2)"
    assert chat.upstream.requests == []


@pytest.mark.asyncio
async def test_conflicting_numbers_bypass_cache_and_fall_back_to_upstream(chat):
    stored = "Who was president of the United States in 2020?"
    query = "Who was president of the United States in 2021?"
    chat.track(stored)
    chat.track(query)
    await insert_semantic_cache(
        stored, await async_encode_prompt(stored), "cached 2020 answer", TEST_MODEL, app.state.pool
    )

    response = await chat.post(query, similarity_threshold=0.85)

    assert response.status_code == 200
    assert response.headers["X-Cache-Status"] == "MISS"
    assert answer(response) == "upstream answer"
    assert len(chat.upstream.requests) == 1


@pytest.mark.asyncio
async def test_streaming_request_yields_sse_chunks(chat):
    prompt = "Stream me a short greeting for the Arctic tern."
    chat.track(prompt)
    chat.upstream.response = lambda request: httpx.Response(
        200, content=sse_body("Hel", "lo", "!"), headers={"content-type": "text/event-stream"}
    )

    response = await chat.post(prompt, stream=True)

    assert response.status_code == 200
    assert response.headers["X-Cache-Status"] == "MISS"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert chat.upstream.requests[0]["stream"] is True

    events = [line[len("data: "):] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello!"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("upstream_response", "expected_status"),
    [
        (lambda request: httpx.Response(500, text="model crashed"), 502),
        (lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=request)), 504),
    ],
    ids=["upstream-500-is-502", "upstream-timeout-is-504"],
)
async def test_upstream_failures_map_to_gateway_errors(chat, stream, upstream_response, expected_status):
    prompt = "Explain the lifecycle of a deep sea anglerfish."
    chat.track(prompt)
    chat.upstream.response = upstream_response

    response = await chat.post(prompt, stream=stream)

    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_request_without_user_message_is_rejected(client):
    response = await client.post(
        "/v1/chat/completions", json={"messages": [{"role": "system", "content": "be nice"}]}
    )
    assert response.status_code == 422


# --- background cache ingestion & fail-open behaviour ---------------------------------------


class MidStreamFailure(httpx.AsyncByteStream):
    """SSE body that delivers some tokens, then drops the connection."""

    async def __aiter__(self):
        yield sse_body("Hel", "lo")[: -len(b"data: [DONE]\n\n")]
        raise httpx.ReadError("connection reset")


@pytest.mark.asyncio
async def test_non_streaming_miss_is_ingested_and_served_from_l1(chat):
    prompt = "What is the tallest mountain in the solar system?"
    chat.track(prompt)

    first = await chat.post(prompt)
    assert first.headers["X-Cache-Status"] == "MISS"
    assert len(chat.upstream.requests) == 1

    second = await chat.post(prompt)
    assert second.status_code == 200
    assert second.headers["X-Cache-Status"] == "HIT-L1"
    assert answer(second) == "upstream answer"
    assert len(chat.upstream.requests) == 1  # upstream untouched
    assert await chat.db_rows() == 1  # L2 was populated too


@pytest.mark.asyncio
async def test_streaming_miss_is_ingested_and_served_from_l2(chat):
    stored = "Tell me the capital of France"
    query = "What is the capital of France?"
    chat.track(stored)
    chat.track(query)
    chat.upstream.response = lambda request: httpx.Response(
        200, content=sse_body("Par", "is"), headers={"content-type": "text/event-stream"}
    )

    first = await chat.post(stored, stream=True)
    assert first.headers["X-Cache-Status"] == "MISS"
    assert "[DONE]" in first.text

    second = await chat.post(query, similarity_threshold=0.85)
    assert second.status_code == 200
    assert second.headers["X-Cache-Status"] == "HIT-L2"
    assert answer(second) == "Paris"
    assert len(chat.upstream.requests) == 1


@pytest.mark.asyncio
async def test_mid_stream_error_does_not_persist_partial_output(chat):
    prompt = "Recite a very long poem about the ocean."
    chat.track(prompt)
    chat.upstream.response = lambda request: httpx.Response(
        200, stream=MidStreamFailure(), headers={"content-type": "text/event-stream"}
    )

    response = await chat.post(prompt, stream=True)

    assert response.status_code == 200
    assert '"error"' in response.text
    assert "Hel" in response.text  # the client did see the partial tokens
    assert await get_exact_match(prompt) is None
    assert await chat.db_rows() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["redis", "postgres", "both"])
async def test_background_ingestion_failures_do_not_affect_client(chat, caplog, failing):
    prompt = f"Name a famous lighthouse ({failing})."
    chat.track(prompt)

    with caplog.at_level(logging.WARNING, logger="gateway.routes"), ExitStack() as stack:
        if failing in ("redis", "both"):
            stack.enter_context(
                patch("gateway.routes.set_exact_match", side_effect=RedisConnectionError("redis down"))
            )
        if failing in ("postgres", "both"):
            stack.enter_context(
                patch(
                    "gateway.routes.insert_semantic_cache",
                    side_effect=asyncpg.InterfaceError("pool closing"),
                )
            )
        response = await chat.post(prompt)

    assert response.status_code == 200
    assert response.headers["X-Cache-Status"] == "MISS"
    assert answer(response) == "upstream answer"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert ("Redis L1 write failed" in messages) == (failing in ("redis", "both"))
    assert ("pgvector L2 write failed" in messages) == (failing in ("postgres", "both"))
    # The tier that did not fail is still populated independently.
    assert (await get_exact_match(prompt) is not None) == (failing == "postgres")
    assert (await chat.db_rows() == 1) == (failing == "redis")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("gateway.routes.get_exact_match", RedisConnectionError("redis down")),
        ("gateway.routes.query_semantic_cache", asyncpg.InterfaceError("pool closing")),
    ],
    ids=["redis-lookup-down", "postgres-lookup-down"],
)
async def test_lookup_failures_fail_open_to_upstream(chat, caplog, target, error):
    prompt = "How tall is the Eiffel Tower?"
    chat.track(prompt)

    with caplog.at_level(logging.WARNING, logger="gateway.routes"), patch(target, side_effect=error):
        response = await chat.post(prompt)

    assert response.status_code == 200
    assert response.headers["X-Cache-Status"] == "MISS"
    assert answer(response) == "upstream answer"
    assert len(chat.upstream.requests) == 1
    assert any(r.levelno == logging.WARNING and "lookup failed" in r.getMessage() for r in caplog.records)
