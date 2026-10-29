# tests/test_semantic_cache.py
# Tests for L2 semantic search: guardrails, insertion, exact/paraphrase hits, misses, and guardrail rejection.
import numpy as np
import pytest
import pytest_asyncio

from cache.database import init_db
from cache.embeddings import async_encode_prompt
from cache.semantic import (
    insert_semantic_cache,
    query_semantic_cache,
    verify_numerical_guardrails,
)

MODEL = "test-model"


def test_guardrails_matching_numbers():
    assert verify_numerical_guardrails("Events of 2020 in 3 cities", "In 3 cities, the 2020 events")
    assert verify_numerical_guardrails("No numbers here", "Still none")
    assert verify_numerical_guardrails("Budget is 1,000", "Budget is 1000")


def test_guardrails_mismatching_numbers():
    assert not verify_numerical_guardrails("Who won in 2020?", "Who won in 2024?")
    assert not verify_numerical_guardrails("Top 5 movies", "Top 10 movies")
    assert not verify_numerical_guardrails("Who won in 2020?", "Who won?")
    assert not verify_numerical_guardrails("Pi is 3.14", "Pi is 3.15")


@pytest_asyncio.fixture
async def pool():
    pool = await init_db()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM semantic_cache WHERE model_name = $1", MODEL)
    try:
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM semantic_cache WHERE model_name = $1", MODEL)
        await pool.close()


async def _store(pool, prompt, response):
    vec = await async_encode_prompt(prompt)
    return await insert_semantic_cache(prompt, vec, response, MODEL, pool)


@pytest.mark.asyncio
async def test_insert_and_exact_retrieval(pool):
    prompt = "What is the capital of France?"
    row_id = await _store(pool, prompt, "Paris")
    assert isinstance(row_id, int)

    vec = await async_encode_prompt(prompt)
    hit = await query_semantic_cache(prompt, vec, pool)
    assert hit is not None
    assert hit["id"] == row_id
    assert hit["response"] == "Paris"
    assert hit["model_name"] == MODEL
    assert hit["similarity"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.asyncio
async def test_paraphrase_hit(pool):
    await _store(pool, "Tell me the capital of France", "Paris")
    query = "What is the capital of France?"
    vec = await async_encode_prompt(query)
    hit = await query_semantic_cache(query, vec, pool, similarity_threshold=0.85)
    assert hit is not None
    assert hit["response"] == "Paris"
    assert hit["similarity"] >= 0.85


@pytest.mark.asyncio
async def test_unrelated_query_misses(pool):
    await _store(pool, "What is the capital of France?", "Paris")
    query = "Explain photosynthesis"
    vec = await async_encode_prompt(query)
    assert await query_semantic_cache(query, vec, pool) is None


@pytest.mark.asyncio
async def test_guardrail_forces_miss(pool):
    await _store(pool, "Who won the World Cup in 2018?", "France")
    query = "Who won the World Cup in 2022?"
    vec = await async_encode_prompt(query)
    # Semantically near-identical, so the vector search alone would hit.
    assert await query_semantic_cache(query, vec, pool, similarity_threshold=0.80) is None


@pytest.mark.asyncio
async def test_invalid_dimensions_rejected(pool):
    bad = np.zeros(128, dtype=np.float32)
    with pytest.raises(ValueError):
        await query_semantic_cache("x", bad, pool)
    with pytest.raises(ValueError):
        await insert_semantic_cache("x", bad, "y", MODEL, pool)
