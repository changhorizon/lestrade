
import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from lestrade.ratelimit import RateLimitMiddleware, _store


@pytest.fixture(autouse=True)
def clear_store():
    _store.clear()
    yield
    _store.clear()


async def chat_endpoint(request):
    return JSONResponse({"ok": True})


async def health_endpoint(request):
    return JSONResponse({"status": "ok"})


def create_app():
    routes = [
        Route("/v1/chat/completions", chat_endpoint, methods=["POST"]),
        Route("/health", health_endpoint, methods=["GET"]),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(RateLimitMiddleware)])
    return app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_normal_request_passes(client):
    resp = client.post("/v1/chat/completions", json={"model": "test"})
    assert resp.status_code == 200


def test_health_not_rate_limited(client):
    for _ in range(20):
        resp = client.get("/health")
        assert resp.status_code == 200


def test_rate_limit_triggers(client):
    for i in range(3):
        resp = client.post("/v1/chat/completions", json={"model": "test"})
        assert resp.status_code == 200

    resp = client.post("/v1/chat/completions", json={"model": "test"})
    assert resp.status_code == 429
    data = resp.json()
    assert "error" in data
    assert "retry_after" in data
    assert "Retry-After" in resp.headers


def test_rate_limit_resets_after_window(client, monkeypatch):
    monkeypatch.setattr("lestrade.config.RATE_LIMIT_WINDOW", 0)
    monkeypatch.setattr("lestrade.config.RATE_LIMIT_MAX_REQUESTS", 2)

    for i in range(2):
        resp = client.post("/v1/chat/completions", json={"model": "test"})
        assert resp.status_code == 200

    resp = client.post("/v1/chat/completions", json={"model": "test"})
    assert resp.status_code == 200
