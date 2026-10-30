# upstream.py
# Asynchronous client for an OpenAI-compatible upstream LLM (Ollama, llama.cpp, vLLM) with streaming support.
import json
from collections.abc import AsyncGenerator
from typing import Self

import httpx

from core.config import get_settings


class UpstreamInferenceError(Exception):
    """The upstream LLM failed or returned an unusable response."""


class UpstreamTimeoutError(UpstreamInferenceError):
    """The upstream LLM did not respond in time."""


class UpstreamLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.upstream_base_url
        self.model = model or settings.upstream_model
        self.timeout = timeout if timeout is not None else settings.upstream_timeout_seconds
        self._limits = httpx.Limits(
            max_connections=settings.upstream_max_connections,
            max_keepalive_connections=settings.upstream_max_keepalive_connections,
        )
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=self._limits,
                transport=self._transport,
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise UpstreamInferenceError("UpstreamLLMClient is not started; call start() first")
        return self._client

    def _payload(self, messages: list[dict], model: str | None, stream: bool, **kwargs) -> dict:
        return {"model": model or self.model, "messages": messages, "stream": stream, **kwargs}

    async def generate_completion(
        self, messages: list[dict], model: str | None = None, **kwargs
    ) -> dict:
        client = self._require_client()
        payload = self._payload(messages, model, False, **kwargs)
        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"Upstream timed out after {self.timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            raise UpstreamInferenceError(
                f"Upstream returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamInferenceError(f"Upstream request failed: {exc!r}") from exc
        except json.JSONDecodeError as exc:
            raise UpstreamInferenceError("Upstream returned invalid JSON") from exc

    async def stream_completion(
        self, messages: list[dict], model: str | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        client = self._require_client()
        payload = self._payload(messages, model, True, **kwargs)
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")[:200]
                    raise UpstreamInferenceError(
                        f"Upstream returned HTTP {response.status_code}: {body}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue  # blank keep-alives, comments, other SSE fields
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices") or [{}]
                        token = choices[0].get("delta", {}).get("content") or ""
                    except (json.JSONDecodeError, AttributeError, IndexError) as exc:
                        raise UpstreamInferenceError(
                            f"Malformed SSE chunk from upstream: {data[:200]}"
                        ) from exc
                    if token:
                        yield token
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"Upstream timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise UpstreamInferenceError(f"Upstream request failed: {exc!r}") from exc
