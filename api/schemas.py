from pydantic import BaseModel
from typing import List, Optional


#  Phase 2: Wardrobe Upload 

class WardrobeItemBasic(BaseModel):
    """Returned by /wardrobe/upload in Phase 2 (no DB, no embeddings yet)."""
    type:       str
    confidence: float
    color:      str
    hex:        str


class UploadResponse(BaseModel):
    items: List[WardrobeItemBasic]


# Phase 4+: Full Wardrobe Item (with DB + embeddings)

class WardrobeItemFull(BaseModel):
    id:          Optional[int] = None
    type:        str
    confidence:  float
    color:       str
    hex:         str
    pattern:     Optional[str] = None
    style:       Optional[str] = None
    season:      Optional[str] = None
    embedding:   Optional[List[float]] = None
    image_path:  Optional[str] = None


#  Phase 5: Recommend 

class RecommendRequest(BaseModel):
    wardrobe:  List[WardrobeItemFull]
    occasion:  str
    weather:   Optional[str] = None
    season:    Optional[str] = None


class OutfitItem(BaseModel):
    id:    Optional[int]
    type:  str
    color: str


class RecommendResponse(BaseModel):
    outfit:  List[OutfitItem]
    score:   float
    occasion: str
