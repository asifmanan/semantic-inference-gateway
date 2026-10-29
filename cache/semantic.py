# semantic.py
# L2 semantic cache: pgvector cosine search with similarity thresholding and numerical guardrails.
import re

import asyncpg
import numpy as np

from cache.embeddings import EMBEDDING_DIM

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _extract_numbers(text: str) -> set[str]:
    # Strip thousands separators so "1,000" == "1000"
    return {m.replace(",", "") for m in _NUMBER_RE.findall(text)}


def verify_numerical_guardrails(prompt_a: str, prompt_b: str) -> bool:
    """True only if both prompts contain exactly the same set of numbers."""
    return _extract_numbers(prompt_a) == _extract_numbers(prompt_b)


def _validate_vector(prompt_vector: np.ndarray) -> None:
    if prompt_vector.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Expected vector of shape ({EMBEDDING_DIM},), got {prompt_vector.shape}"
        )


async def query_semantic_cache(
    prompt: str,
    prompt_vector: np.ndarray,
    pool: asyncpg.Pool,
    similarity_threshold: float = 0.90,
) -> dict | None:
    """Return the nearest cached entry if it passes the threshold and guardrails."""
    _validate_vector(prompt_vector)
    max_distance = 1.0 - similarity_threshold

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, prompt, response, model_name, (prompt_vector <=> $1) AS distance
            FROM semantic_cache
            ORDER BY prompt_vector <=> $1
            LIMIT 1;
            """,
            prompt_vector,
        )

    if row is None or row["distance"] > max_distance:
        return None
    if not verify_numerical_guardrails(prompt, row["prompt"]):
        return None

    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "response": row["response"],
        "model_name": row["model_name"],
        "similarity": 1.0 - row["distance"],
    }


async def insert_semantic_cache(
    prompt: str,
    prompt_vector: np.ndarray,
    response: str,
    model_name: str,
    pool: asyncpg.Pool,
) -> int:
    """Insert a cache entry and return its new row id."""
    _validate_vector(prompt_vector)
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO semantic_cache (prompt, prompt_vector, response, model_name)
            VALUES ($1, $2, $3, $4)
            RETURNING id;
            """,
            prompt,
            prompt_vector,
            response,
            model_name,
        )
