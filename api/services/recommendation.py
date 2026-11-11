"""The outfit recommender shared by the recommend and chat routes."""
from api.schemas import RecommendResponse
from models.recommender.recommender import Recommender, RecommendResult

recommender = Recommender()


def to_response(result: RecommendResult, occasion: str) -> RecommendResponse:
    return RecommendResponse(
        outfit=result.outfit,
        score=result.score,
        color_score=result.color_score,
        reasons=result.reasons,
        reason_codes=result.reason_codes,
        occasion=occasion,
        missing=result.missing,
        relaxed=result.relaxed,
    )
