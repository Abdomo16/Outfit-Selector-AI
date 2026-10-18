from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.routes import wardrobe


class FakeSupabase:
    def __init__(self, tokens):
        self.rows = []
        db = self

        class Auth:
            def get_user(self, token):
                if token not in tokens:
                    raise ValueError("bad token")
                return SimpleNamespace(user=SimpleNamespace(id=tokens[token]))

        class Insert:
            def __init__(self, payload):
                self.payload = payload

            def execute(self):
                db.rows.append(self.payload)
                return SimpleNamespace(data=[dict(self.payload)])

        class Table:
            def insert(self, payload):
                return Insert(payload)

        self.auth = Auth()
        self._table = Table()

    def table(self, name):
        assert name == "wardrobe_items"
        return self._table


class FakePipeline:
    def process_upload(self, image):
        crop = Image.new("RGB", (40, 60), (20, 30, 40))
        return {"items": [
            {"crop": crop, "partial": None, "type": "shirt", "category": "Tops",
             "confidence": 0.9, "color": "Olive", "hex": "#556b2f"},
            {"crop": crop, "partial": "cut_off", "type": "trousers", "category": "Bottoms",
             "confidence": 0.6, "color": "Navy", "hex": "#1f2a44"},
        ]}


@pytest.fixture
def fake(monkeypatch):
    db = FakeSupabase(tokens={"good-token": "me", "other-token": "someone-else"})
    monkeypatch.setattr(wardrobe, "get_supabase", lambda: db)
    monkeypatch.setattr(wardrobe, "_upload_crop", lambda crop, item_id: f"https://img/{item_id}.jpg")
    return db


@pytest.fixture
def client():
    # No context manager, so the lifespan hook does not load the models.
    return TestClient(app)


def _photo_bytes():
    import io
    buffer = io.BytesIO()
    Image.new("RGB", (80, 120), (200, 200, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def _run_job(user_id="me"):
    job_id = "job-1"
    wardrobe._set_job(job_id, status="processing", items=[], pending=[], held={},
                      user_id=user_id, error=None)
    wardrobe._process_upload_job(job_id, _photo_bytes(), user_id, FakePipeline())
    return job_id


def test_partial_item_is_held_not_saved(client, fake):
    job_id = _run_job()

    body = client.get(f"/wardrobe/jobs/{job_id}").json()
    assert body["status"] == "done"
    assert [i["type"] for i in body["items"]] == ["shirt"]
    assert [r["type"] for r in fake.rows] == ["shirt"]

    [pending] = body["pending"]
    assert pending["reason"] == "cut_off"
    assert pending["type"] == "trousers"
    assert pending["preview_base64"]
    assert "embedding" not in pending and "crop" not in pending


def test_keep_saves_the_pending_item(client, fake):
    job_id = _run_job()
    pending_id = client.get(f"/wardrobe/jobs/{job_id}").json()["pending"][0]["id"]

    resp = client.post(
        f"/wardrobe/jobs/{job_id}/keep", json={"ids": [pending_id]},
        headers={"Authorization": "Bearer good-token"},
    )

    assert resp.status_code == 200
    assert [i["type"] for i in resp.json()["items"]] == ["trousers"]
    assert [r["type"] for r in fake.rows] == ["shirt", "trousers"]
    assert fake.rows[1]["user_id"] == "me"
    assert client.get(f"/wardrobe/jobs/{job_id}").json()["pending"] == []


def test_skip_drops_the_pending_item(client, fake):
    job_id = _run_job()

    resp = client.post(
        f"/wardrobe/jobs/{job_id}/keep", json={"ids": []},
        headers={"Authorization": "Bearer good-token"},
    )

    assert resp.json() == {"items": []}
    assert [r["type"] for r in fake.rows] == ["shirt"]
    assert client.get(f"/wardrobe/jobs/{job_id}").json()["pending"] == []


def test_keep_needs_the_uploaders_token(client, fake):
    job_id = _run_job()
    pending_id = client.get(f"/wardrobe/jobs/{job_id}").json()["pending"][0]["id"]

    assert client.post(f"/wardrobe/jobs/{job_id}/keep", json={"ids": [pending_id]}).status_code == 401
    resp = client.post(
        f"/wardrobe/jobs/{job_id}/keep", json={"ids": [pending_id]},
        headers={"Authorization": "Bearer other-token"},
    )
    assert resp.status_code == 403
    assert [r["type"] for r in fake.rows] == ["shirt"]
    assert len(client.get(f"/wardrobe/jobs/{job_id}").json()["pending"]) == 1


def test_keep_on_unknown_job_is_404(client, fake):
    resp = client.post(
        "/wardrobe/jobs/nope/keep", json={"ids": []},
        headers={"Authorization": "Bearer good-token"},
    )
    assert resp.status_code == 404
