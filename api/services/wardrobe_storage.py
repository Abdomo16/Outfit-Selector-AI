"""Wardrobe photos: resizing, encoding, and local-filesystem storage.

Photos are stored under STORAGE_ROOT/wardrobe-images/{item_id}/ and served
by FastAPI's StaticFiles mount at /static/.  Set BASE_URL in the environment
so the public URL resolves correctly when the server is reachable from other
hosts (e.g. a device hitting the backend on the LAN).
"""
import base64
import io
import os
import uuid
from pathlib import Path

from PIL import Image

MAX_IMAGE_SIDE = 1536
PREVIEW_SIDE = 320

_STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "storage"))
_WARDROBE_DIR = _STORAGE_ROOT / "wardrobe-images"


def _base_url() -> str:
    return os.getenv("BASE_URL", "http://localhost:8002").rstrip("/")


def _item_dir(item_id: str) -> Path:
    d = _WARDROBE_DIR / item_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def downscale(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Shrink longest side to max_side (no upscale)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def encode_crop(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Encode a PIL image as JPEG (default) or PNG bytes."""
    buf = io.BytesIO()
    if fmt.upper() == "PNG":
        image.save(buf, format="PNG", optimize=True)
    else:
        image.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def preview_base64(crop: Image.Image | None) -> str | None:
    """Small JPEG thumbnail for the keep-or-skip sheet, as base64."""
    if crop is None:
        return None
    return base64.b64encode(encode_crop(downscale(crop, PREVIEW_SIDE))).decode("ascii")


def _public_url(rel_path: Path) -> str:
    """Build a public URL from a path relative to STORAGE_ROOT."""
    parts = rel_path.parts
    return f"{_base_url()}/static/{'/'.join(parts)}"


def upload_crop(image: Image.Image, item_id: str) -> str:
    """Save a JPEG crop to local storage and return its public URL."""
    dest = _item_dir(item_id) / f"{uuid.uuid4()}.jpg"
    dest.write_bytes(encode_crop(image, fmt="JPEG"))
    return _public_url(dest.relative_to(_STORAGE_ROOT))


def upload_cutout(image: Image.Image, item_id: str) -> str | None:
    """Save a transparent-background RGBA PNG and return its public URL.

    PNG is required because JPEG has no alpha channel; saving as JPEG
    would composite the garment against white and lose the transparency
    the outfit preview depends on.
    """
    dest = _item_dir(item_id) / "cutout.png"
    try:
        buf = io.BytesIO()
        image.convert("RGBA").save(buf, format="PNG", optimize=True)
        dest.write_bytes(buf.getvalue())
        return _public_url(dest.relative_to(_STORAGE_ROOT))
    except Exception as exc:
        print(f"[wardrobe] WARNING: cutout save failed for {item_id}: {exc}")
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


def storage_path(image_url: str | None) -> Path | None:
    """The local Path for a photo public URL, or None if the file is not from local storage."""
    if not image_url:
        return None
    marker = "/static/"
    if marker not in image_url:
        return None
    rel = image_url.split(marker, 1)[1]
    path = _STORAGE_ROOT / rel
    return path if path.exists() else None


def remove_photos(paths: list[Path | None]) -> None:
    """Delete local photo files; failures are logged, not raised."""
    for path in paths:
        if path and path.exists():
            try:
                path.unlink()
            except Exception as exc:
                print(f"[wardrobe] WARNING: could not remove {path}: {exc}")
