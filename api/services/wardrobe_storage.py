"""Wardrobe photos: resizing, encoding and the Supabase Storage bucket."""
import base64
import io
import uuid

from PIL import Image

from api import supabase_client

WARDROBE_BUCKET = "wardrobe-images"
MAX_IMAGE_SIDE = 1536
PREVIEW_SIDE = 320


def ensure_bucket() -> None:
    """Create the wardrobe storage bucket if it does not exist."""
    supabase = supabase_client.get_supabase()
    buckets = supabase.storage.list_buckets()
    if not any(bucket.name == WARDROBE_BUCKET for bucket in buckets):
        supabase.storage.create_bucket(WARDROBE_BUCKET, options={"public": True})


def downscale(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Shrink longest side to max_side (no upscale)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def encode_crop(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Encode a PIL crop as JPEG (default) or PNG bytes."""
    buffer = io.BytesIO()
    rgb = image.convert("RGB")
    if fmt.upper() == "PNG":
        rgb.save(buffer, format="PNG", optimize=True)
    else:
        rgb.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def preview_base64(crop: Image.Image | None) -> str | None:
    """A small JPEG of the crop for the keep-or-skip sheet."""
    if crop is None:
        return None
    return base64.b64encode(encode_crop(downscale(crop, PREVIEW_SIDE))).decode("ascii")


def public_url(path: str) -> str:
    """Normalize Supabase public URL (strip trailing ? some clients add)."""
    url = supabase_client.get_supabase().storage.from_(WARDROBE_BUCKET).get_public_url(path)
    return url.rstrip("?")


def _upload(path: str, data: bytes, mime: str) -> None:
    supabase_client.get_supabase().storage.from_(WARDROBE_BUCKET).upload(
        path,
        data,
        file_options={"content-type": mime, "contentType": mime, "upsert": "true"},
    )


def upload_crop(image: Image.Image, item_id: str) -> str:
    """Upload a crop to Supabase Storage and return its public URL.

    JPEG is tried first because it is typically 5–10× smaller than PNG for
    photos. PNG is the fallback for the rare bucket configurations that
    reject JPEG uploads.
    """
    ensure_bucket()
    attempts = (
        ("jpg", "JPEG", "image/jpeg"),
        ("png", "PNG", "image/png"),
    )
    last_error: Exception | None = None

    for ext, fmt, mime in attempts:
        path = f"{item_id}/{uuid.uuid4()}.{ext}"
        try:
            _upload(path, encode_crop(image, fmt=fmt), mime)
            return public_url(path)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Storage upload failed: {last_error}")


def upload_cutout(image: Image.Image, item_id: str) -> str | None:
    """Upload a transparent-background RGBA PNG and return its public URL.

    PNG is required because JPEG has no alpha channel; saving as JPEG
    would composite the garment against white and lose the transparency
    the mannequin view depends on.
    """
    ensure_bucket()
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format="PNG", optimize=True)
    path = f"{item_id}/cutout.png"
    try:
        _upload(path, buffer.getvalue(), "image/png")
        return public_url(path)
    except Exception as exc:
        print(f"[wardrobe] WARNING: cutout upload failed for {item_id}: {exc}")
        return None


def make_cutout(crop: Image.Image, pipeline) -> Image.Image | None:
    """Run rembg on a crop and return an RGBA image with the background removed.

    The pipeline's already-loaded rembg session is reused here so a second
    model instance is never allocated. The color_detector holds the session;
    if it was disabled at startup (DISABLE_BACKGROUND_REMOVAL=1) or failed
    to load, background_remover is None and this returns None gracefully.
    """
    try:
        bg = getattr(getattr(pipeline, "color_detector", None), "background_remover", None)
        if bg is None:
            return None
        from rembg import remove as rembg_remove
        return rembg_remove(crop.convert("RGB"), session=bg.session)
    except Exception as exc:
        print(f"[wardrobe] WARNING: cutout generation failed: {exc}")
        return None


def storage_path(image_url: str | None) -> str | None:
    """The object path inside the wardrobe bucket for a public photo URL."""
    if not image_url:
        return None
    marker = f"/{WARDROBE_BUCKET}/"
    url = image_url.split("?", 1)[0]
    if marker not in url:
        return None
    return url.split(marker, 1)[1] or None


def remove_photos(paths: list[str]) -> None:
    """Delete photos from the bucket; a failure is logged, not raised."""
    if not paths:
        return
    try:
        supabase_client.get_supabase().storage.from_(WARDROBE_BUCKET).remove(paths)
    except Exception as exc:
        print(f"[wardrobe] WARNING: could not remove photos {paths}: {exc}")
