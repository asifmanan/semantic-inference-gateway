# redis_client.py
# Provides an asynchronous Redis client for caching exact matches of prompts and their corresponding responses.
import hashlib
import json
import re
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis

from core.config import get_settings

_redis_client: Redis | None = None


def normalize_prompt(prompt: str) -> str:
    cleaned = prompt.strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def hash_prompt(prompt: str) -> str:
    normalized = normalize_prompt(prompt)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"prompt:exact:{digest}"


async def get_redis_client() -> Redis:
    """Acquire or initialize the global asynchronous Redis client pool."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_l1_dsn,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )
    return _redis_client


async def close_redis() -> None:
    """Close the Redis client pool connection cleanly."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


async def get_exact_match(prompt: str) -> dict[str, Any]:
    client = await get_redis_client()
    key = hash_prompt(prompt)
    data = await client.get(key)
    if data is None:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


async def set_exact_match(
    prompt: str,
    response: str,
    model_name: str,
    ttl: int = 3600,
) -> bool:
    client = await get_redis_client()
    key = hash_prompt(prompt)
    payload = {
        "prompt": prompt,
        "response": response,
        "model_name": model_name,
    }
    return await client.set(key, json.dumps(payload), ex=ttl)