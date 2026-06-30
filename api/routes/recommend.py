from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/")
def recommend_outfits():
    raise HTTPException(status_code=501, detail="Not Implemented")
