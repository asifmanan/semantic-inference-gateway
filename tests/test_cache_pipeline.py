# tests/test_cache_pipeline.py
# Tests for hierarchical dual-tier cache resolution: Redis L1 exact match -> pgvector L2 semantic search.
import time
from unittest.mock import patch

import pytest
import pytest_asyncio

from cache.database import init_db
from cache.embeddings import async_encode_prompt
from cache.redis_client import (
    close_redis,
    get_exact_match,
    get_redis_client,
    hash_prompt,
    set_exact_match,
)
from cache.semantic import insert_semantic_cache, query_semantic_cache

MODEL = "pipeline-test"

pytestmark = pytest.mark.asyncio


async def resolve_cached_query(
    prompt: str, pool, similarity_threshold: float = 0.85
) -> tuple[dict | None, str]:
    """Resolve a prompt through L1 (Redis exact) then L2 (pgvector semantic)."""
    hit = await get_exact_match(prompt)
    if hit is not None:
        return hit, "HIT_L1"

    vector = await async_encode_prompt(prompt)
    hit = await query_semantic_cache(prompt, vector, pool, similarity_threshold)
    if hit is not None:
        return hit, "HIT_L2"

    return None, "MISS"


async def populate_both_tiers(pool, prompt: str, response: str) -> int:
    """Simulate an LLM completion being written to L1 and L2."""
    vector = await async_encode_prompt(prompt)
    await set_exact_match(prompt, response, MODEL, ttl=60)
    return await insert_semantic_cache(prompt, vector, response, MODEL, pool)


async def _clean_pg(pool):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM semantic_cache WHERE model_name = $1", MODEL)


@pytest_asyncio.fixture
async def pool():
    pool = await init_db()
    await _clean_pg(pool)
    try:
        yield pool
    finally:
        await _clean_pg(pool)
        await pool.close()


@pytest_asyncio.fixture
async def redis_keys():
    """Collects prompts used in a test; deletes their L1 keys on setup and teardown."""
    prompts: list[str] = []
    client = await get_redis_client()
    try:
        yield prompts
    finally:
        if prompts:
            await client.delete(*(hash_prompt(p) for p in prompts))
        await close_redis()


async def test_cold_miss_and_storage(pool, redis_keys):
    prompt = "What is the capital of France?"
    redis_keys.append(prompt)
    client = await get_redis_client()
    await client.delete(hash_prompt(prompt))

    result, status = await resolve_cached_query(prompt, pool)
    assert (result, status) == (None, "MISS")

    row_id = await populate_both_tiers(pool, prompt, "Paris")
    assert isinstance(row_id, int)

    # Both tiers now serve the entry.
    l1 = await get_exact_match(prompt)
    assert l1 is not None and l1["response"] == "Paris"
    vector = await async_encode_prompt(prompt)
    l2 = await query_semantic_cache(prompt, vector, pool)
    assert l2 is not None and l2["id"] == row_id


async def test_l1_exact_hit_bypasses_vector_engine(pool, redis_keys):
    prompt = "What is the capital of France?"
    redis_keys.append(prompt)
    await populate_both_tiers(pool, prompt, "Paris")

    altered = "  WHAT is the   capital of FRANCE?\n"
    with (
        patch(f"{__name__}.async_encode_prompt") as mock_encode,
        patch(f"{__name__}.query_semantic_cache") as mock_query,
    ):
        await resolve_cached_query(altered, pool)  # warm the connection
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            result, status = await resolve_cached_query(altered, pool)
            timings.append(time.perf_counter() - start)

        mock_encode.assert_not_called()
        mock_query.assert_not_called()

    assert status == "HIT_L1"
    assert result["response"] == "Paris"
    assert min(timings) < 0.001, f"L1 hit took {min(timings) * 1000:.3f}ms"


async def test_l2_semantic_hit_paraphrase(pool, redis_keys):
    stored = "What is the capital of France?"
    query = "Tell me the French capital"
    redis_keys.extend([stored, query])
    await populate_both_tiers(pool, stored, "Paris")

    assert await get_exact_match(query) is None  # L1 must miss

    result, status = await resolve_cached_query(query, pool)
    assert status == "HIT_L2"
    assert result["response"] == "Paris"
    assert result["similarity"] >= 0.85


async def test_guardrail_rejection_causes_cache_bypass(pool, redis_keys):
    stored = "Who was president of the United States in 2020?"
    query = "Who was president of the United States in 2021?"
    redis_keys.extend([stored, query])
    await populate_both_tiers(pool, stored, "Donald Trump")

    # Without the guardrail the vector search alone would hit.
    vector = await async_encode_prompt(query)
    async with pool.acquire() as conn:
        distance = await conn.fetchval(
            "SELECT MIN(prompt_vector <=> $1) FROM semantic_cache WHERE model_name = $2",
            vector,
            MODEL,
        )
    assert 1.0 - distance >= 0.85

    result, status = await resolve_cached_query(query, pool)
    assert (result, status) == (None, "MISS")
