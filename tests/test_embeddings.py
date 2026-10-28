# tests/test_embeddings.py
# Tests for the local embedding pipeline: shape, dtype, normalization, semantic similarity, async behavior.
import asyncio
import time

import numpy as np
import pytest

from cache.embeddings import (
    EMBEDDING_DIM,
    async_encode_prompt,
    encode_prompt,
    get_embedding_model,
)


def test_model_is_singleton():
    assert get_embedding_model() is get_embedding_model()


def test_output_shape_and_dtype():
    vector = encode_prompt("Hello world")
    assert vector.shape == (EMBEDDING_DIM,) == (384,)
    assert vector.dtype == np.float32


def test_l2_normalized():
    vector = encode_prompt("Explain quantum computing in simple terms.")
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_whitespace_normalization():
    a = encode_prompt("  What   is\n the capital of France? ")
    b = encode_prompt("What is the capital of France?")
    assert np.allclose(a, b, atol=1e-6)


def test_semantic_similarity():
    a = encode_prompt("What is the capital of France?")
    b = encode_prompt("Tell me the French capital")
    assert float(np.dot(a, b)) > 0.80


def test_unrelated_prompts_less_similar():
    a = encode_prompt("What is the capital of France?")
    b = encode_prompt("How do I bake sourdough bread?")
    assert float(np.dot(a, b)) < 0.5


@pytest.mark.asyncio
async def test_async_encode_matches_sync():
    prompt = "What is the capital of France?"
    vector = await async_encode_prompt(prompt)
    assert vector.shape == (384,)
    assert vector.dtype == np.float32
    assert np.allclose(vector, encode_prompt(prompt), atol=1e-5)


@pytest.mark.asyncio
async def test_async_encode_does_not_block_event_loop():
    get_embedding_model()  # warm up so load time isn't measured
    ticks = 0
    done = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(0.001)
            ticks += 1

    task = asyncio.create_task(ticker())
    prompts = [f"Benchmark prompt number {i} about machine learning systems" for i in range(64)]
    start = time.perf_counter()
    await asyncio.gather(*(async_encode_prompt(p) for p in prompts))
    elapsed = time.perf_counter() - start
    done.set()
    await task

    # If the loop were blocked, the ticker would barely run during encoding.
    assert ticks > 5, f"event loop starved: {ticks} ticks in {elapsed:.2f}s"
