import numpy as np
import pytest

from cache.database import init_db


@pytest.mark.asyncio
async def test_database_connection_and_vector_ops():
    pool = await init_db()
    try:
        dummy_vector = np.random.rand(384).astype(np.float32)

        async with pool.acquire() as conn:
            # 1. Test insertion
            row = await conn.fetchrow("""
                INSERT INTO semantic_cache (prompt, prompt_vector, response, model_name)
                VALUES ($1, $2, $3, $4)
                RETURNING id, prompt;
            """, "What is the capital of France?", dummy_vector, "Paris", "llama-3.2-1b")
            
            assert row is not None
            assert row["prompt"] == "What is the capital of France?"

            # 2. Test vector cosine distance query (<=> operator)
            result = await conn.fetchrow("""
                SELECT id, prompt, response, (prompt_vector <=> $1) AS cosine_distance
                FROM semantic_cache
                ORDER BY prompt_vector <=> $1
                LIMIT 1;
            """, dummy_vector)
            
            assert result is not None
            assert result["cosine_distance"] < 1e-4

    finally:
        await pool.close()