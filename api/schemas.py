from pydantic import BaseModel
from typing import List, Optional, Union


#  Phase 2: Wardrobe Upload 

class WardrobeItemBasic(BaseModel):
    """Returned by /wardrobe/upload in Phase 2 (no DB, no embeddings yet)."""
    type:        str
    confidence:  float
    color:       str
    hex:         str
    crop_base64: Optional[str] = None


# Phase 4+: Full Wardrobe Item 

class WardrobeItemFull(BaseModel):
    id:          Optional[Union[int, str]] = None
    category:    Optional[str] = None   # wardrobe tab, e.g. "Outerwear"
    type:        str
    confidence:  float
    color:       str
    hex:         str
    secondary_color: Optional[str] = None  # second fabric color of patterned items
    secondary_hex:   Optional[str] = None
    pattern:     Optional[str] = None
    style:       Optional[str] = None
    occasion:    Optional[str] = None   # CLIP-predicted item occasion, e.g. "Gym", "Work"
    season:      Optional[str] = None
    embedding:   Optional[List[float]] = None
    image_path:  Optional[str] = None


class UploadResponse(BaseModel):
    items: List[WardrobeItemFull]


class DeleteItemsRequest(BaseModel):
    ids: List[Union[int, str]]


class PendingItem(BaseModel):
    """An item held back from the wardrobe until the user keeps it."""
    id:             str
    reason:         str             # "cut_off" or "hidden"
    type:           str
    category:       Optional[str] = None
    color:          str
    preview_base64: Optional[str] = None


class KeepItemsRequest(BaseModel):
    ids: List[str]


#  Phase 5: Recommend 

class RecommendRequest(BaseModel):
    wardrobe:  List[WardrobeItemFull]
    occasion:  str
    weather:   Optional[str] = None
    season:    Optional[str] = None
    style:     Optional[str] = None
    lock_ids:  Optional[List[Union[int, str]]] = None
    swap_id:   Optional[Union[int, str]] = None
    exclude_ids: Optional[List[Union[int, str]]] = None


class ReasonCode(BaseModel):
    code: str
    colors: List[str] = []


class OutfitItem(BaseModel):
    id:         Union[int, str, None] = None
    category:   str                     # the item's detailed type, e.g. "jeans"
    wardrobeCategory: Optional[str] = None  # its wardrobe tab, e.g. "Bottoms"
    color:      str
    confidence: Optional[float] = None
    style:      Optional[str] = None
    pattern:    Optional[str] = None
    season:     Optional[str] = None
    imageUrl:   Optional[str] = None


class RecommendResponse(BaseModel):
    outfit:  List[OutfitItem]
    score:   float
    color_score: float
    reasons: List[str] = []
    reason_codes: List[ReasonCode] = []
    occasion: str
