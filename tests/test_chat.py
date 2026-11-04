import requests
from fastapi.testclient import TestClient

from api.main import app
from api.routes import chat as chat_route
from models.assistant import intent as intents
from models.assistant import openrouter_client
from models.assistant.openrouter_client import AssistantError, complete_json

client = TestClient(app)

WARDROBE = [
    {"id": "t1", "type": "t-shirt", "confidence": 1.0, "color": "white", "hex": "#fff"},
    {"id": "s1", "type": "shirt", "confidence": 1.0, "color": "white", "hex": "#fff"},
    {"id": "b1", "type": "jeans", "confidence": 1.0, "color": "navy", "hex": "#00f"},
    {"id": "sh1", "type": "sneakers", "confidence": 1.0, "color": "white", "hex": "#fff"},
    {"id": "sh2", "type": "boots", "confidence": 1.0, "color": "black", "hex": "#000"},
]


def _post(**body):
    payload = {"message": "hi", "wardrobe": WARDROBE, **body}
    return client.post("/chat/", json=payload)


def _model_returns(monkeypatch, raw):
    def fake(messages, timeout=20.0):
        if isinstance(raw, Exception):
            raise raw
        return raw
    monkeypatch.setattr(chat_route, "complete_json", fake)


def test_model_intent_drives_the_outfit(monkeypatch):
    _model_returns(monkeypatch, {
        "wants_outfit": True,
        "occasion": "Date Night",
        "style": None,
        "reply": "Here is a sharp date look.",
    })
    response = _post(message="dinner date tonight")
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Here is a sharp date look."
    assert data["used_fallback"] is False
    assert data["outfit"]["occasion"] == "date night"
    ids = {item["id"] for item in data["outfit"]["outfit"]}
    assert "t1" not in ids


def test_invented_ids_and_values_are_dropped():
    parsed = intents.validate(
        {
            "occasion": "beach rave",
            "style": "Elegant",
            "weather": "snowstorm",
            "change_zones": ["shoes", "hats", "shoes"],
            "lock_ids": ["b1", "ghost"],
            "exclude_ids": [42],
        },
        ["t1", "b1"],
    )
    assert parsed.occasion is None
    assert parsed.style == "elegant"
    assert parsed.weather is None
    assert parsed.change_zones == ["shoes"]
    assert parsed.lock_ids == ["b1"]
    assert parsed.exclude_ids == []


def test_change_zone_swaps_only_that_part(monkeypatch):
    _model_returns(monkeypatch, {"wants_outfit": True, "change_zones": ["shoes"], "reply": "New shoes."})
    response = _post(message="swap the shoes", current_outfit_ids=["t1", "b1", "sh1"])
    ids = {item["id"] for item in response.json()["outfit"]["outfit"]}
    assert {"t1", "b1", "sh2"} <= ids
    assert "sh1" not in ids


def test_question_without_outfit(monkeypatch):
    _model_returns(monkeypatch, {"wants_outfit": False, "reply": "Navy goes with almost anything."})
    data = _post(message="what goes with navy?").json()
    assert data["outfit"] is None
    assert data["reply"] == "Navy goes with almost anything."
    assert data["no_match"] is False


def test_fallback_when_model_fails(monkeypatch):
    _model_returns(monkeypatch, AssistantError("timeout"))
    data = _post(message="something for the gym please").json()
    assert data["used_fallback"] is True
    assert data["reply"] == ""
    assert data["outfit"]["occasion"] == "gym"


def test_no_match_is_reported(monkeypatch):
    _model_returns(monkeypatch, {"wants_outfit": True, "occasion": "work", "reply": "Office look."})
    wardrobe = [{"id": "x", "type": "hoodie", "confidence": 1.0, "color": "black", "hex": "#000", "style": "Sport"}]
    data = client.post("/chat/", json={"message": "work", "wardrobe": wardrobe}).json()
    assert data["no_match"] is True
    assert data["outfit"] is None


def test_keyword_intent_english_and_arabic():
    english = intents.keyword_intent("Make it warmer and swap the shoes")
    assert english.weather == "cold"
    assert english.change_zones == ["shoes"]

    arabic = intents.keyword_intent("عايز لبس للجيم")
    assert arabic.occasion == "gym"

    whole = intents.keyword_intent("show me a different color")
    assert whole.change_zones == ["all"]


def test_client_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        complete_json([{"role": "user", "content": "hi"}])
    except AssistantError:
        pass
    else:
        raise AssertionError("expected AssistantError")


class _Response:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_client_parses_fenced_json_and_rejects_bad_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        openrouter_client.requests, "post",
        lambda *args, **kwargs: _Response('```json\n{"wants_outfit": false}\n```'),
    )
    assert complete_json([]) == {"wants_outfit": False}

    monkeypatch.setattr(openrouter_client.requests, "post", lambda *args, **kwargs: _Response("not json"))
    try:
        complete_json([])
    except AssistantError:
        pass
    else:
        raise AssertionError("expected AssistantError")

    def timeout(*args, **kwargs):
        raise requests.Timeout("slow")
    monkeypatch.setattr(openrouter_client.requests, "post", timeout)
    try:
        complete_json([])
    except AssistantError:
        pass
    else:
        raise AssertionError("expected AssistantError")


def test_prompt_lists_wardrobe_and_current_outfit():
    from api.schemas import WardrobeItemFull
    wardrobe = [WardrobeItemFull(**item) for item in WARDROBE]
    messages = intents.build_messages(
        "hi", [{"role": "user", "content": "earlier"}, {"role": "tool", "content": "x"}],
        wardrobe, ["t1"], "work", "mild", "spring", "ar",
    )
    assert "Arabic" in messages[0]["content"]
    assert "t1 | top | t-shirt | white" in messages[1]["content"]
    assert "Current outfit:" in messages[1]["content"]
    assert [m["role"] for m in messages[2:]] == ["user", "user"]
