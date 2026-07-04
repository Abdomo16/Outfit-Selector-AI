"""
tests/test_phase2.py

Integration tests for Phase 2: Classifier + Color Detection pipeline.
Run from the project root:
    python -m pytest tests/test_phase2.py -v
"""
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_image_bytes(color_rgb: tuple, fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (200, 200), color=color_rgb)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def test_upload_response_shape(client):
    """Response must have 'items' list with correct keys."""
    image_bytes = _make_image_bytes((200, 50, 50))  # reddish
    resp = client.post(
        "/wardrobe/upload",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "items" in data, "Response missing 'items' key"
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert set(item.keys()) >= {"type", "confidence", "color", "hex"}


def test_upload_confidence_range(client):
    """Confidence must be a valid probability (0.0 – 1.0)."""
    image_bytes = _make_image_bytes((100, 100, 100))
    resp = client.post(
        "/wardrobe/upload",
        files={"file": ("grey.jpg", image_bytes, "image/jpeg")},
    )
    item = resp.json()["items"][0]
    assert 0.0 <= item["confidence"] <= 1.0


def test_upload_hex_format(client):
    """Hex color must be a valid #rrggbb string."""
    image_bytes = _make_image_bytes((0, 0, 200))  # blue
    resp = client.post(
        "/wardrobe/upload",
        files={"file": ("blue.jpg", image_bytes, "image/jpeg")},
    )
    hex_val = resp.json()["items"][0]["hex"]
    assert hex_val.startswith("#"), f"Expected hex to start with #, got: {hex_val}"
    assert len(hex_val) == 7, f"Expected #rrggbb format, got: {hex_val}"


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
