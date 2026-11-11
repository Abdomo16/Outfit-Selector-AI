"""Saving classified wardrobe items: photos to storage, rows to the database."""
import uuid

from PIL import Image
from pydantic import BaseModel

from api.schemas import WardrobeItemFull
from api.services import wardrobe_storage as storage


def as_dict(model: BaseModel) -> dict:
    """A pydantic model as a plain dict, on pydantic 1 or 2."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def insert_item(supabase, payload: dict):
    """Insert a wardrobe row. The embedding is optional if the column rejects it."""
    try:
        return supabase.table("wardrobe_items").insert(payload).execute()
    except Exception:
        if payload.get("embedding") is None:
            raise
        return supabase.table("wardrobe_items").insert({**payload, "embedding": None}).execute()


def store_item(supabase, item: dict, crop: Image.Image | None, user_id: str | None, pipeline) -> dict:
    """Upload the crop and its cutout, insert the wardrobe row, and return
    the stored item."""
    item_id = str(uuid.uuid4())

    # Upload crop BEFORE insert so we never persist a row without image_path.
    image_url = None
    cutout_url = None
    if crop is not None:
        try:
            image_url = storage.upload_crop(crop, item_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to upload crop for item {item_id}: {exc}") from exc
        cutout = storage.make_cutout(crop, pipeline)
        if cutout is not None:
            cutout_url = storage.upload_cutout(cutout, item_id)

    payload = {
        "id": item_id,
        "user_id": user_id,
        "category": item.get("category"),
        "type": item.get("type"),
        "confidence": float(item.get("confidence", 0)),
        "color": item.get("color"),
        "hex": item.get("hex"),
        "secondary_color": item.get("secondary_color"),
        "secondary_hex": item.get("secondary_hex"),
        "pattern": item.get("pattern"),
        "style": item.get("style"),
        "season": item.get("season"),
        "embedding": item.get("embedding"),
        "image_path": image_url,
        "cutout_path": cutout_url,
    }

    response = insert_item(supabase, payload)
    if not response.data:
        raise RuntimeError("Failed to save wardrobe item to database.")

    record = response.data[0]
    return as_dict(WardrobeItemFull(
        id=record["id"],
        category=record.get("category"),
        type=record["type"],
        confidence=record["confidence"],
        color=record["color"],
        hex=record["hex"],
        secondary_color=record.get("secondary_color"),
        secondary_hex=record.get("secondary_hex"),
        pattern=record.get("pattern"),
        style=record.get("style"),
        occasion=item.get("occasion"),
        season=record.get("season"),
        embedding=record.get("embedding"),
        image_path=record.get("image_path") or image_url,
        cutout_path=record.get("cutout_path") or cutout_url,
    ))
