from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from lestrade.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestIngest:
    def test_ingest_text(self, client):
        mock_add = AsyncMock(return_value=3)
        with patch("lestrade.router.ingest.engine.add_text", mock_add):
            resp = client.post("/api/ingest", data={
                "text": "Sample document content for testing.",
                "source": "test.md",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chunks"] == 3
        assert data["source"] == "test.md"

    def test_ingest_file(self, client):
        mock_add = AsyncMock(return_value=2)
        with patch("lestrade.router.ingest.engine.add_text", mock_add):
            resp = client.post("/api/ingest/file", files={
                "file": ("doc.txt", b"File content here.", "text/plain"),
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chunks"] == 2
        assert data["source"] == "doc.txt"

    def test_ingest_file_non_utf8(self, client):
        mock_add = AsyncMock(return_value=1)
        with patch("lestrade.router.ingest.engine.add_text", mock_add):
            resp = client.post("/api/ingest/file", files={
                "file": ("doc.bin", b"\xff\xfe\x00\x01", "application/octet-stream"),
            })

        assert resp.status_code == 400

    def test_ingest_stats(self, client):
        resp = client.get("/api/ingest/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_vectors" in data
        assert "index_path" in data

    def test_ingest_engine_error(self, client):
        mock_add = AsyncMock(side_effect=Exception("DB error"))
        with patch("lestrade.router.ingest.engine.add_text", mock_add):
            resp = client.post("/api/ingest", data={
                "text": "content",
                "source": "err.md",
            })

        assert resp.status_code == 500
