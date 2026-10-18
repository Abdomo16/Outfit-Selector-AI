"""
tests/test_phase2.py

API contract for wardrobe uploads. Uploads are queued as background jobs,
so the job runner is replaced and no model or database is touched.
Run from the project root:
    python -m pytest tests/test_phase2.py -v
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.routes import wardrobe


@pytest.fixture
def client(monkeypatch):
    submitted = []
    monkeypatch.setattr(wardrobe._EXECUTOR, "submit", lambda *args: submitted.append(args))
    app.state.pipeline = object()
    # No context manager, so the lifespan hook does not load the models.
    test_client = TestClient(app)
    test_client.submitted = submitted
    return test_client


def _make_image_bytes(color_rgb: tuple, fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (200, 200), color=color_rgb)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_upload_queues_a_job(client):
    resp = client.post(
        "/wardrobe/upload",
        files={"file": ("test.jpg", _make_image_bytes((200, 50, 50)), "image/jpeg")},
        data={"user_id": "user-1"},
    )

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "processing"
    assert len(client.submitted) == 1
    assert client.submitted[0][1] == body["job_id"]
    assert client.submitted[0][3] == "user-1"


def test_new_job_reports_processing(client):
    job_id = client.post(
        "/wardrobe/upload",
        files={"file": ("grey.jpg", _make_image_bytes((100, 100, 100)), "image/jpeg")},
    ).json()["job_id"]

    resp = client.get(f"/wardrobe/jobs/{job_id}")

    assert resp.status_code == 200
    assert resp.json() == {
        "job_id": job_id, "status": "processing", "items": [], "pending": [], "error": None,
    }


def test_unknown_job_is_404(client):
    assert client.get("/wardrobe/jobs/does-not-exist").status_code == 404


def test_upload_invalid_file(client):
    """Non-image upload must return 400."""
    resp = client.post(
        "/wardrobe/upload",
        files={"file": ("data.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
