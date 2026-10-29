# tests/test_gateway_api.py
# Tests for the FastAPI gateway skeleton: lifespan wiring and the /health endpoint.
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from cache.embeddings import MODEL_NAME
from gateway.app import app


@pytest_asyncio.fixture
async def client():
    # ASGITransport does not send lifespan events, so run the lifespan explicitly.
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "database": "connected",
        "redis": "connected",
        "embedding_model": MODEL_NAME,
    }


@pytest.mark.asyncio
async def test_app_state_populated_and_operational(client):
    assert await app.state.redis.ping() is True
    async with app.state.pool.acquire() as conn:
        assert await conn.fetchval("SELECT 1;") == 1


@pytest.mark.asyncio
async def test_health_returns_503_when_dependency_fails(client, monkeypatch):
    async def boom():
        raise ConnectionError("redis down")

    monkeypatch.setattr(app.state.redis, "ping", boom)
    response = await client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["redis"] == "unavailable"
    assert body["database"] == "connected"


@pytest.mark.asyncio
async def test_startup_failure_releases_acquired_resources(monkeypatch):
    import gateway.app as gw

    closed = {"pool": False, "redis": False}

    class FakePool:
        async def close(self):
            closed["pool"] = True

    async def fake_init_db():
        return FakePool()

    async def failing_redis():
        raise ConnectionError("redis down")

    async def fake_close_redis():
        closed["redis"] = True

    monkeypatch.setattr(gw, "init_db", fake_init_db)
    monkeypatch.setattr(gw, "get_redis_client", failing_redis)
    monkeypatch.setattr(gw, "close_redis", fake_close_redis)

    with pytest.raises(ConnectionError):
        async with gw.lifespan(gw.app):
            pass
    assert closed == {"pool": True, "redis": True}


@pytest.mark.asyncio
async def test_pool_init_failure_does_not_mask_error(monkeypatch):
    import gateway.app as gw

    async def failing_init_db():
        raise OSError("postgres down")

    async def fake_close_redis():
        pass

    monkeypatch.setattr(gw, "init_db", failing_init_db)
    monkeypatch.setattr(gw, "close_redis", fake_close_redis)

    with pytest.raises(OSError, match="postgres down"):
        async with gw.lifespan(gw.app):
            pass


@pytest.mark.asyncio
async def test_health_returns_503_when_postgres_fails(client, monkeypatch):
    import asyncpg

    def broken_acquire(*args, **kwargs):
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(type(app.state.pool), "acquire", broken_acquire)
    response = await client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["database"] == "unavailable"
    assert body["redis"] == "connected"


@pytest.mark.asyncio
async def test_health_does_not_swallow_unexpected_errors(client, monkeypatch):
    async def bug():
        raise ValueError("programming error")

    monkeypatch.setattr(app.state.redis, "ping", bug)
    with pytest.raises(ValueError):
        await client.get("/health")
