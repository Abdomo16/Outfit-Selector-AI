from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import wardrobe

PUBLIC = "https://x.supabase.co/storage/v1/object/public/wardrobe-images"


class FakeQuery:
    def __init__(self, db, action):
        self.db = db
        self.action = action
        self.filters = []

    def in_(self, column, values):
        self.filters.append(lambda r: str(r[column]) in {str(v) for v in values})
        return self

    def eq(self, column, value):
        self.filters.append(lambda r: r[column] == value)
        return self

    def execute(self):
        matched = [r for r in self.db.rows if all(f(r) for f in self.filters)]
        if self.action == "delete":
            self.db.rows = [r for r in self.db.rows if r not in matched]
        return SimpleNamespace(data=[dict(r) for r in matched])


class FakeSupabase:
    def __init__(self, rows, tokens):
        self.rows = rows
        self.removed = []
        fake = self

        class Auth:
            def get_user(self, token):
                if token not in tokens:
                    raise ValueError("bad token")
                return SimpleNamespace(user=SimpleNamespace(id=tokens[token]))

        class Bucket:
            def remove(self, paths):
                fake.removed.extend(paths)

        class Storage:
            def from_(self, name):
                assert name == wardrobe.WARDROBE_BUCKET
                return Bucket()

        self.auth = Auth()
        self.storage = Storage()

    def table(self, name):
        assert name == "wardrobe_items"
        db = self

        class Table:
            def select(self, columns):
                return FakeQuery(db, "select")

            def delete(self):
                return FakeQuery(db, "delete")

        return Table()


@pytest.fixture
def fake(monkeypatch):
    db = FakeSupabase(
        rows=[
            {"id": "a", "user_id": "me", "image_path": f"{PUBLIC}/a/1.jpg?"},
            {"id": "b", "user_id": "me", "image_path": f"{PUBLIC}/b/2.jpg"},
            {"id": "c", "user_id": "someone-else", "image_path": f"{PUBLIC}/c/3.jpg"},
        ],
        tokens={"good-token": "me"},
    )
    monkeypatch.setattr(wardrobe, "get_supabase", lambda: db)
    return db


@pytest.fixture
def client():
    # No context manager, so the lifespan hook does not load the models.
    return TestClient(app)


def test_missing_or_bad_token_is_rejected(client, fake):
    assert client.post("/wardrobe/items/delete", json={"ids": ["a"]}).status_code == 401
    resp = client.post(
        "/wardrobe/items/delete", json={"ids": ["a"]},
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401
    assert len(fake.rows) == 3


def test_deletes_only_the_callers_rows_and_photos(client, fake):
    resp = client.post(
        "/wardrobe/items/delete", json={"ids": ["a", "b", "c"]},
        headers={"Authorization": "Bearer good-token"},
    )

    assert resp.status_code == 200
    assert sorted(resp.json()["deleted"]) == ["a", "b"]
    assert sorted(fake.removed) == ["a/1.jpg", "b/2.jpg"]
    assert [r["id"] for r in fake.rows] == ["c"]


def test_empty_selection_deletes_nothing(client, fake):
    resp = client.post(
        "/wardrobe/items/delete", json={"ids": []},
        headers={"Authorization": "Bearer good-token"},
    )
    assert resp.json() == {"deleted": []}
    assert len(fake.rows) == 3


def test_storage_path_parsing():
    assert wardrobe._storage_path(f"{PUBLIC}/item/photo.jpg?") == "item/photo.jpg"
    assert wardrobe._storage_path(f"{PUBLIC}/item/photo.jpg?t=1") == "item/photo.jpg"
    assert wardrobe._storage_path("https://elsewhere.com/x.jpg") is None
    assert wardrobe._storage_path(None) is None
