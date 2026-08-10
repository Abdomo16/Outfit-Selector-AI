from fastapi import APIRouter, HTTPException
from api.schemas import RecommendRequest, RecommendResponse
from models.recommender.recommender import Recommender

router = APIRouter()
recommender = Recommender()

@router.post("/", response_model=RecommendResponse)
def recommend_outfits(request: RecommendRequest):
    # TEMP DEBUG - remove after diagnosing the sport-tee leak
    print("[DEBUG request]")
    for it in request.wardrobe:
        print(f"  [DEBUG item] type={it.type!r} style={it.style!r} occasion={it.occasion!r}")
    print(f"  [DEBUG occasion] {request.occasion!r} season={request.season!r}")
    outfit, score = recommender.recommend(
        wardrobe=request.wardrobe,
        occasion=request.occasion,
        season=request.season
    )
    if not outfit:
        raise HTTPException(status_code=404, detail="No valid combinations found for the given rules.")
        
    return RecommendResponse(outfit=outfit, score=score, occasion=request.occasion)
