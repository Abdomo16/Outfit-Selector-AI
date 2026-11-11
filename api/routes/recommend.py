from fastapi import APIRouter, HTTPException
from api.schemas import RecommendRequest, RecommendResponse
from api.services.recommendation import recommender, to_response

router = APIRouter()

@router.post("/", response_model=RecommendResponse)
def recommend_outfits(request: RecommendRequest):
    result = recommender.recommend(
        wardrobe=request.wardrobe,
        occasion=request.occasion,
        season=request.season,
        weather=request.weather,
        style=request.style,
        lock_ids=request.lock_ids,
        swap_id=request.swap_id,
        exclude_ids=request.exclude_ids,
        layer=bool(request.layer),
        temp_high=request.temp_high,
        temp_low=request.temp_low,
        last_worn=request.last_worn,
        previous_ids=request.previous_ids,
    )
    if not result.outfit:
        raise HTTPException(status_code=404, detail="No valid combinations found for the given rules.")

    return to_response(result, request.occasion)
