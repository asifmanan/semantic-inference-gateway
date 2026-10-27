# tests/test_redis_cache.py
# Tests for the Redis caching layer, focusing on exact match retrieval and storage of prompts and responses.
import asyncio

import pytest

from cache.redis_client import (
    close_redis,
    get_exact_match,
    get_redis_client,
    hash_prompt,
    normalize_prompt,
    set_exact_match,
)


def test_prompt_normalization_and_hashing():
    """
    Test the normalization and hashing of prompts.
    """
    raw_prompt_1 = "  What is the capital   of France? \n"
    raw_prompt_2 = "what is the capital of france?"

    # Verify whitespace collapsing and casing
    assert normalize_prompt(raw_prompt_1) == "what is the capital of france?"
    # Verify deterministic SHA-256 hash output
    assert hash_prompt(raw_prompt_1) == hash_prompt(raw_prompt_2)


@pytest.mark.asyncio
async def test_redis_connection_ping():
    """
    Test the connection to the Redis instance.
    """
    client = await get_redis_client()
    try:
        pong = await client.ping()
        assert pong is True
    finally:
        # cleanup: ensure the client is closed
        await close_redis()


@pytest.mark.asyncio
async def test_exact_match_hit_and_miss():
    """
    Test exact match retrieval and storage of prompts and responses.
    """
    client = await get_redis_client()
    test_prompt = "Explain quantum entanglement in simple terms."
    test_response = "Quantum entanglement is a phenomenon where particles become interconnected..."
    model = "llama-3.2-1b"
    key = hash_prompt(test_prompt)

    try:
        # 1. Cold miss: verify key does not return stale data
        await client.delete(key)
        miss_result = await get_exact_match(test_prompt)
        assert miss_result is None

        # 2. Store payload in L1 cache
        stored = await set_exact_match(
            prompt=test_prompt,
            response=test_response,
            model_name=model,
            ttl=60,
        )
        assert stored is True

        # 3. Exact hit with altered whitespace and uppercase formatting
        formatted_prompt = "  EXPLAIN quantum   entanglement in simple terms.  \n"
        hit_result = await get_exact_match(formatted_prompt)

        assert hit_result is not None
        assert hit_result["response"] == test_response
        assert hit_result["model_name"] == model
        assert hit_result["prompt"] == test_prompt

    finally:
        # cleanup: ensure the key is deleted to avoid test pollution
        await client.delete(key)
        await close_redis()


@pytest.mark.asyncio
async def test_ttl_expiration():
    """
    Test the expiration of cached entries based on their TTL.
    """
    client = await get_redis_client()
    test_prompt = "Temporary prompt for TTL verification."
    test_response = "Temporary response."
    model = "llama-3.2-1b"
    key = hash_prompt(test_prompt)

    try:
        # Set with a 1-second TTL
        await set_exact_match(
            prompt=test_prompt,
            response=test_response,
            model_name=model,
            ttl=1,
        )

        # Immediate check should hit
        hit = await get_exact_match(test_prompt)
        assert hit is not None

        # Wait for TTL expiration (1.2s gives adequate safety margin)
        await asyncio.sleep(1.2)

        # Post-expiration check should return None
        expired = await get_exact_match(test_prompt)
        assert expired is None
        assert await client.exists(key) == 0

    finally:
        # cleanup
        await client.delete(key)
        await close_redis()