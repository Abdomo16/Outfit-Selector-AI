from fastapi import APIRouter

from api.schemas import (
    ChatRequest,
    ChatResponse,
    RecommendResponse,
    RoutineRequest,
    RoutineResponse,
)
from models.assistant import intent as intents
from models.assistant import routine as routines
from models.assistant.openrouter_client import AssistantError, complete_json
from models.recommender.recommender import Recommender, _zone

router = APIRouter()
recommender = Recommender()


def _resolve_intent(request: ChatRequest, current_ids: list) -> tuple:
    messages = intents.build_messages(
        request.message,
        [turn.model_dump() for turn in request.history],
        request.wardrobe,
        current_ids,
        request.occasion,
        request.weather,
        request.season,
        request.locale,
    )
    owned = [str(item.id) for item in request.wardrobe if item.id is not None]
    try:
        return intents.validate(complete_json(messages), owned), False
    except AssistantError:
        return intents.keyword_intent(request.message), True


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    current_ids = [str(item_id) for item_id in request.current_outfit_ids]
    intent, used_fallback = _resolve_intent(request, current_ids)
    if not intent.wants_outfit:
        return ChatResponse(reply=intent.reply, used_fallback=used_fallback)

    locks = list(intent.lock_ids)
    excludes = [str(item_id) for item_id in request.exclude_ids] + intent.exclude_ids
    if current_ids and intent.change_zones:
        by_id = {str(item.id): item for item in request.wardrobe}
        change_all = "all" in intent.change_zones
        for item_id in current_ids:
            item = by_id.get(item_id)
            if item is None:
                continue
            if change_all or _zone(item) in intent.change_zones:
                excludes.append(item_id)
            elif item_id not in locks:
                locks.append(item_id)

    occasion = intent.occasion or request.occasion or "everyday"
    result = recommender.recommend(
        wardrobe=request.wardrobe,
        occasion=occasion,
        season=request.season,
        weather=intent.weather or request.weather,
        style=intent.style,
        lock_ids=locks,
        exclude_ids=[item_id for item_id in excludes if item_id not in locks],
    )
    if not result.outfit:
        return ChatResponse(reply=intent.reply, used_fallback=used_fallback, no_match=True)

    return ChatResponse(
        reply=intent.reply,
        used_fallback=used_fallback,
        outfit=RecommendResponse(
            outfit=result.outfit,
            score=result.score,
            color_score=result.color_score,
            reasons=result.reasons,
            reason_codes=result.reason_codes,
            occasion=occasion,
        ),
    )


@router.post("/routine", response_model=RoutineResponse)
def routine(request: RoutineRequest):
    """The user's usual week, from a sentence such as "university Sunday to
    Wednesday and gym on Tuesday"."""
    result, used_fallback = routines.parse(request.message, request.locale, complete_json)
    return RoutineResponse(
        days=routines.to_json(result.days),
        reply=result.reply,
        used_fallback=used_fallback,
    )
