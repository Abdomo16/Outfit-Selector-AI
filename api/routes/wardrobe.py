from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/upload")
def upload_wardrobe_item():
    raise HTTPException(status_code=501, detail="Not Implemented")
