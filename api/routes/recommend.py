from fastapi import APIRouter, HTTPException
from api.schemas import RecommendRequest, RecommendResponse
from models.recommender.recommender import Recommender

router = APIRouter()
recommender = Recommender()

@router.post("/", response_model=RecommendResponse)
def recommend_outfits(request: RecommendRequest):
    outfit, score = recommender.recommend(
        wardrobe=request.wardrobe,
        occasion=request.occasion,
        season=request.season
    )
    if not outfit:
        raise HTTPException(status_code=404, detail="No valid combinations found for the given rules.")
        
    return RecommendResponse(outfit=outfit, score=score, occasion=request.occasion)
