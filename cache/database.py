import asyncpg
from pgvector.asyncpg import register_vector

from core.config import get_settings

settings = get_settings()

def get_dsn() -> str:
    """
    Returns the DSN (Data Source Name) for connecting to the PostgreSQL database.
    """
    return settings.asyncpg_dsn

async def get_db_pool():
    dsn = get_dsn()

    return await asyncpg.create_pool(
        dsn=dsn,
        min_size=5,
        max_size=20,
        init=init_connection
    )

async def init_connection(conn):
    # Register pgvector type handler with asyncpg
    await register_vector(conn)

async def init_db():
    dsn = get_dsn()
    
    # 1. Open a temporary bootstrap connection to create the extension if needed
    bootstrap_conn = await asyncpg.connect(dsn)
    try:
        await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    finally:
        await bootstrap_conn.close()

    # 2. Now that the vector type definitely exists, create the regular pool
    pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=5,
        max_size=20,
        init=init_connection
    )

    # 3. Create tables and HNSW index using the pool
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id BIGSERIAL PRIMARY KEY,
                prompt TEXT NOT NULL,
                prompt_vector vector(384) NOT NULL,
                response TEXT NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_cache_vector_hnsw 
            ON semantic_cache 
            USING hnsw (prompt_vector vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

    return pool