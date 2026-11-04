from fastapi.testclient import TestClient

from api.main import app
from api.routes import chat as chat_route
from models.assistant.openrouter_client import AssistantError
from models.assistant.routine import keyword_routine, validate

client = TestClient(app)

MON, TUE, WED, THU, FRI, SAT, SUN = 1, 2, 3, 4, 5, 6, 7


def test_english_days_after_occasions():
    days = keyword_routine("University on Sunday and Monday, gym on Tuesday").days
    assert days == {SUN: ["university"], MON: ["university"], TUE: ["gym"]}


def test_egyptian_arabic_with_joined_and():
    days = keyword_routine("عندي جامعة الحد والاتنين وجيم التلات").days
    assert days == {SUN: ["university"], MON: ["university"], TUE: ["gym"]}


def test_arabic_range_wraps_the_egyptian_week():
    days = keyword_routine("شغل من السبت للأربع").days
    assert set(days) == {SAT, SUN, MON, TUE, WED}
    assert all(value == ["work"] for value in days.values())


def test_english_range_and_days_first():
    days = keyword_routine("Sunday to Thursday I have work").days
    assert set(days) == {SUN, MON, TUE, WED, THU}


def test_friday_prayer_is_family_on_friday():
    assert keyword_routine("صلاة الجمعة").days == {FRI: ["family"]}


def test_two_activities_on_one_day_are_kept_in_order():
    days = keyword_routine("university and gym on Wednesday").days
    assert days == {WED: ["university", "gym"]}


def test_nothing_recognised_gives_an_empty_week():
    assert keyword_routine("hello there").days == {}


def test_validate_drops_unknown_days_and_occasions():
    result = validate({
        "days": {"sunday": ["university", "juggling", "gym", "party"], "funday": ["gym"]},
        "reply": " Got it ",
    })
    assert result.days == {SUN: ["university", "gym"]}
    assert result.reply == "Got it"


def test_endpoint_uses_the_model(monkeypatch):
    monkeypatch.setattr(
        chat_route, "complete_json",
        lambda messages, timeout=20.0: {"days": {"saturday": ["work"]}, "reply": "Done"},
    )
    response = client.post("/chat/routine", json={"message": "work saturdays", "locale": "en"})
    assert response.status_code == 200
    assert response.json() == {"days": {"6": ["work"]}, "reply": "Done", "used_fallback": False}


def test_endpoint_falls_back_to_keywords(monkeypatch):
    def fail(messages, timeout=20.0):
        raise AssistantError("offline")
    monkeypatch.setattr(chat_route, "complete_json", fail)
    response = client.post("/chat/routine", json={"message": "gym on Monday"})
    assert response.json()["days"] == {"1": ["gym"]}
    assert response.json()["used_fallback"] is True
