"""Saving classified wardrobe items: photos to local storage, rows to PostgreSQL."""
import uuid

from PIL import Image
from pydantic import BaseModel

from api.models import WardrobeItem
from api.schemas import WardrobeItemFull
from api.services import wardrobe_storage as storage


def as_dict(model: BaseModel) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def item_to_full(record: WardrobeItem, occasion: str | None = None) -> WardrobeItemFull:
    return WardrobeItemFull(
        id=str(record.id),
        category=record.category,
        type=record.type or "",
        confidence=record.confidence or 0.0,
        color=record.color or "",
        hex=record.hex or "",
        secondary_color=record.secondary_color,
        secondary_hex=record.secondary_hex,
        pattern=record.pattern,
        style=record.style,
        occasion=occasion,
        season=record.season,
        embedding=record.embedding,
        image_path=record.image_path,
        cutout_path=record.cutout_path,
    )


def store_item(session, item: dict, crop: Image.Image | None, user_id: str | None, pipeline) -> dict:
    """Upload the crop and its cutout, insert the wardrobe row, and return the stored item."""
    item_id = str(uuid.uuid4())

    # Upload crop BEFORE insert so we never persist a row without image_path.
    image_url = None
    cutout_url = None
    if crop is not None:
        try:
            image_url = storage.upload_crop(crop, item_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to save crop for item {item_id}: {exc}") from exc
        cutout = storage.make_cutout(crop, pipeline)
        if cutout is not None:
            cutout_url = storage.upload_cutout(cutout, item_id)

    record = WardrobeItem(
        id=item_id,
        user_id=user_id,
        category=item.get("category"),
        type=item.get("type"),
        confidence=float(item.get("confidence", 0)),
        color=item.get("color"),
        hex=item.get("hex"),
        secondary_color=item.get("secondary_color"),
        secondary_hex=item.get("secondary_hex"),
        pattern=item.get("pattern"),
        style=item.get("style"),
        season=item.get("season"),
        embedding=item.get("embedding"),
        image_path=image_url,
        cutout_path=cutout_url,
    )
    session.add(record)
    session.flush()

    return as_dict(item_to_full(record, occasion=item.get("occasion")))
