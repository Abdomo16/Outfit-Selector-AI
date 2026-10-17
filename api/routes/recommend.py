from fastapi import APIRouter, HTTPException
from api.schemas import RecommendRequest, RecommendResponse
from models.recommender.recommender import Recommender

router = APIRouter()
recommender = Recommender()

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
    )
    if not result.outfit:
        raise HTTPException(status_code=404, detail="No valid combinations found for the given rules.")

    return RecommendResponse(
        outfit=result.outfit,
        score=result.score,
        color_score=result.color_score,
        reasons=result.reasons,
        reason_codes=result.reason_codes,
        occasion=request.occasion,
    )
