from fastapi.testclient import TestClient

from app import app


def test_website_and_health_endpoint():
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Basira ID" in page.text
        assert "streamlit" not in page.text.lower()

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.headers["cache-control"] == "no-store"


def test_analysis_rejects_missing_and_unsupported_uploads():
    with TestClient(app) as client:
        missing = client.post("/api/analyze")
        assert missing.status_code == 422

        bad = client.post(
            "/api/analyze",
            files={"files": ("not-an-image.txt", b"not an image", "text/plain")},
        )
        assert bad.status_code == 415
