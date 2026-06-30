from pydantic import BaseModel
from typing import Optional, List

class UploadItemResponse(BaseModel):
    item_id: str
    category: str
    color: str

class RecommendRequest(BaseModel):
    occasion: Optional[str] = None
    weather: Optional[str] = None

class OutfitRecommendation(BaseModel):
    top_id: str
    bottom_id: str
    shoes_id: Optional[str] = None
    score: float

class RecommendResponse(BaseModel):
    recommendations: List[OutfitRecommendation]
