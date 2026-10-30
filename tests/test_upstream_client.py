# tests/test_upstream_client.py
# Tests for the upstream LLM proxy client using httpx.MockTransport (no external model server needed).
import json

import httpx
import pytest

from gateway.upstream import (
    UpstreamInferenceError,
    UpstreamLLMClient,
    UpstreamTimeoutError,
)

MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


def make_client(handler) -> UpstreamLLMClient:
    return UpstreamLLMClient(
        base_url="http://upstream.test/v1",
        model="test-model",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )


def sse(*chunks: str | dict) -> bytes:
    lines = []
    for chunk in chunks:
        data = chunk if isinstance(chunk, str) else json.dumps(chunk)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode()


def delta(content: str | None, role: str | None = None) -> dict:
    d = {}
    if role:
        d["role"] = role
    if content is not None:
        d["content"] = content
    return {"choices": [{"index": 0, "delta": d}]}


@pytest.mark.asyncio
async def test_generate_completion_parses_openai_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Paris"},
                     "finish_reason": "stop"}
                ],
            },
        )

    async with make_client(handler) as client:
        result = await client.generate_completion(MESSAGES, temperature=0.2)

    assert result["choices"][0]["message"]["content"] == "Paris"
    assert seen["url"] == "http://upstream.test/v1/chat/completions"
    assert seen["body"] == {
        "model": "test-model",
        "messages": MESSAGES,
        "stream": False,
        "temperature": 0.2,
    }


@pytest.mark.asyncio
async def test_stream_completion_yields_tokens_and_stops_at_done():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        body = sse(
            delta("", role="assistant"),
            delta("The"),
            delta(" capital"),
            delta(" is Paris"),
            delta(None),
            "[DONE]",
            delta("SHOULD NOT APPEAR"),
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with make_client(handler) as client:
        tokens = [t async for t in client.stream_completion(MESSAGES)]

    assert tokens == ["The", " capital", " is Paris"]


@pytest.mark.asyncio
async def test_http_500_raises_upstream_inference_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model crashed")

    async with make_client(handler) as client:
        with pytest.raises(UpstreamInferenceError, match="500") as non_stream:
            await client.generate_completion(MESSAGES)
        assert not isinstance(non_stream.value, UpstreamTimeoutError)

        with pytest.raises(UpstreamInferenceError, match="500"):
            async for _ in client.stream_completion(MESSAGES):
                pass


@pytest.mark.asyncio
async def test_timeout_raises_upstream_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    async with make_client(handler) as client:
        with pytest.raises(UpstreamTimeoutError):
            await client.generate_completion(MESSAGES)
        with pytest.raises(UpstreamTimeoutError):
            async for _ in client.stream_completion(MESSAGES):
                pass


@pytest.mark.asyncio
async def test_connection_error_raises_upstream_inference_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with make_client(handler) as client:
        with pytest.raises(UpstreamInferenceError) as exc_info:
            await client.generate_completion(MESSAGES)
    assert not isinstance(exc_info.value, UpstreamTimeoutError)


@pytest.mark.asyncio
async def test_client_lifecycle():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = make_client(handler)
    assert client._client is None
    with pytest.raises(UpstreamInferenceError, match="not started"):
        await client.generate_completion(MESSAGES)

    await client.start()
    await client.start()  # idempotent
    inner = client._client
    assert inner is not None and not inner.is_closed
    await client.generate_completion(MESSAGES)

    await client.aclose()
    assert inner.is_closed
    assert client._client is None
    await client.aclose()  # safe to call twice

    async with make_client(handler) as ctx_client:
        ctx_inner = ctx_client._client
        assert ctx_inner is not None and not ctx_inner.is_closed
    assert ctx_inner.is_closed


def test_defaults_come_from_settings():
    client = UpstreamLLMClient()
    assert client.base_url == "http://localhost:11434/v1"
    assert client.model == "llama3.2:1b"
    assert client.timeout == 30.0
