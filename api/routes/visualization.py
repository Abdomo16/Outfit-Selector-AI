from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List
from models.visualization.vton import VirtualTryOn
from PIL import Image
import io
import json

router = APIRouter()
vton_engine = VirtualTryOn()

@router.post("/try-on")
async def try_on(
    base_image: UploadFile = File(...),
    wardrobe_items: str = Form(..., description="JSON string of selected apparel items")
):
    """
    Virtual Try-On Endpoint.
    Accepts a base image of a person/mannequin and a list of clothing items to map on.
    """
    if not base_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Base image must be a valid image file.")

    try:
        items = json.loads(wardrobe_items)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="wardrobe_items must be a valid JSON string array.")

    contents = await base_image.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode base image file.")
   
    result_image = vton_engine.generate(image, items)
    
    return {
        "status": "success", 
        "message": "Virtual Try-On generated successfully.",
        "items_used": len(items),
        "result_url": "https://example.com/generated_tryon_image.png" # Stub URL
    }
