import io

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from PIL import Image

router = APIRouter()


@router.post("/upload")
def upload_wardrobe_item(request: Request, file: UploadFile = File(...)):
    """
    Accept an image of a clothing item.
    Returns { items: [ { type, confidence, color, hex } ] }
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    contents = file.file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    pipeline = request.app.state.pipeline
    try:
        result = pipeline.process_upload(image)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process image.")
    return result
