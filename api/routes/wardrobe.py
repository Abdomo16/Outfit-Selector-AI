import base64
import io
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from api.schemas import WardrobeItemFull
from api.supabase_client import get_supabase

router = APIRouter()

WARDROBE_BUCKET = "wardrobe-images"


def _ensure_bucket() -> None:
    """Create the wardrobe storage bucket if it does not exist."""
    supabase = get_supabase()
    buckets = supabase.storage.list_buckets()
    if not any(bucket.name == WARDROBE_BUCKET for bucket in buckets):
        supabase.storage.create_bucket(WARDROBE_BUCKET, options={"public": True})


def _encode_crop(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _upload_crop(image: Image.Image, item_id: int) -> str:
    """Upload a crop to Supabase Storage and return its public URL."""
    _ensure_bucket()
    supabase = get_supabase()

    path = f"{item_id}/{uuid.uuid4()}.png"
    data = _encode_crop(image)
    supabase.storage.from_(WARDROBE_BUCKET).upload(path, data, file_options={"content-type": "image/png"})

    public_url = supabase.storage.from_(WARDROBE_BUCKET).get_public_url(path)
    return public_url


@router.post("/upload")
def upload_wardrobe_item(
    request: Request,
    file: UploadFile = File(...),
    user_id: str | None = Form(None),
):
    """
    Accept an image of a clothing item.
    Segments the image, persists each detected item to Supabase, uploads the
    cropped image to Supabase Storage, and returns the persisted metadata
    including a public image URL.
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

    supabase = get_supabase()
    stored_items = []

    for item in result.get("items", []):
        crop = item.pop("crop", None)

        payload = {
            "user_id": user_id,
            "type": item.get("type"),
            "confidence": float(item.get("confidence", 0)),
            "color": item.get("color"),
            "hex": item.get("hex"),
            "pattern": item.get("pattern"),
            "style": item.get("style"),
            "occasion": item.get("occasion"),
            "season": item.get("season"),
            "embedding": item.get("embedding"),
        }

        response = supabase.table("wardrobe_items").insert(payload).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to save wardrobe item to database.")

        record = response.data[0]
        item_id = record["id"]

        image_url = None
        if crop is not None:
            try:
                image_url = _upload_crop(crop, item_id)
                supabase.table("wardrobe_items").update({"image_path": image_url}).eq("id", item_id).execute()
            except Exception:
                raise HTTPException(status_code=500, detail=f"Failed to upload crop for item {item_id}.")

        stored_items.append(WardrobeItemFull(
            id=item_id,
            type=record["type"],
            confidence=record["confidence"],
            color=record["color"],
            hex=record["hex"],
            pattern=record.get("pattern"),
            style=record.get("style"),
            occasion=record.get("occasion"),
            season=record.get("season"),
            embedding=record.get("embedding"),
            image_path=image_url,
        ))

    return {"items": stored_items}
