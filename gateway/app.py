# app.py
# FastAPI application skeleton: connection lifecycle, embedding model preloading, and health check.
import logging
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from cache.database import init_db
from cache.embeddings import MODEL_NAME, get_embedding_model
from cache.redis_client import close_redis, get_redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = None
    app.state.redis = None
    try:
        app.state.pool = await init_db()
        app.state.redis = await get_redis_client()
        # Pre-warm so PyTorch weights are in RAM before the first request.
        get_embedding_model()
        yield
    finally:
        # Each resource is released independently so one failure can't skip the other.
        try:
            if app.state.pool is not None:
                await app.state.pool.close()
        finally:
            await close_redis()


logger = logging.getLogger(__name__)

# Failures that mean "dependency unreachable or unhealthy"; anything else is a bug and should surface.
REDIS_ERRORS = (RedisError, OSError, TimeoutError)
POSTGRES_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError)

app = FastAPI(title="MLSys Semantic Inference Gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    status = {"database": "connected", "redis": "connected"}
    try:
        await app.state.redis.ping()
    except REDIS_ERRORS as exc:
        logger.warning("Health check: Redis unavailable: %r", exc)
        status["redis"] = "unavailable"
    try:
        async with app.state.pool.acquire() as conn:
            await conn.execute("SELECT 1;")
    except POSTGRES_ERRORS as exc:
        logger.warning("Health check: PostgreSQL unavailable: %r", exc)
        status["database"] = "unavailable"

    if "unavailable" in status.values():
        return JSONResponse(status_code=503, content={"status": "unhealthy", **status})
    return {"status": "healthy", **status, "embedding_model": MODEL_NAME}
