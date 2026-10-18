import base64
import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from api.schemas import DeleteItemsRequest, KeepItemsRequest, PendingItem, WardrobeItemFull
from api.supabase_client import get_supabase

router = APIRouter()

WARDROBE_BUCKET = "wardrobe-images"
MAX_IMAGE_SIDE = 1536
PREVIEW_SIDE = 320
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _ensure_bucket() -> None:
    """Create the wardrobe storage bucket if it does not exist."""
    supabase = get_supabase()
    buckets = supabase.storage.list_buckets()
    if not any(bucket.name == WARDROBE_BUCKET for bucket in buckets):
        supabase.storage.create_bucket(WARDROBE_BUCKET, options={"public": True})


def _downscale(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Shrink longest side to max_side (no upscale)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _encode_crop(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Encode a PIL crop as JPEG (default) or PNG bytes."""
    buffer = io.BytesIO()
    rgb = image.convert("RGB")
    if fmt.upper() == "PNG":
        rgb.save(buffer, format="PNG", optimize=True)
    else:
        rgb.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def _public_url(path: str) -> str:
    """Normalize Supabase public URL (strip trailing ? some clients add)."""
    url = get_supabase().storage.from_(WARDROBE_BUCKET).get_public_url(path)
    return url.rstrip("?")


def _upload_crop(image: Image.Image, item_id: str) -> str:
    """Upload a crop to Supabase Storage and return its public URL."""
    _ensure_bucket()
    supabase = get_supabase()

    attempts = (
        ("jpg", "JPEG", "image/jpeg"),
        ("png", "PNG", "image/png"),
    )
    last_error: Exception | None = None

    for ext, fmt, mime in attempts:
        path = f"{item_id}/{uuid.uuid4()}.{ext}"
        data = _encode_crop(image, fmt=fmt)
        try:
            supabase.storage.from_(WARDROBE_BUCKET).upload(
                path,
                data,
                file_options={
                    "content-type": mime,
                    "contentType": mime,
                    "upsert": "true",
                },
            )
            return _public_url(path)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Storage upload failed: {last_error}")


def _insert_item(supabase, payload: dict):
    """Insert a wardrobe row. The embedding is optional if the column rejects it."""
    try:
        return supabase.table("wardrobe_items").insert(payload).execute()
    except Exception:
        if payload.get("embedding") is None:
            raise
        return supabase.table("wardrobe_items").insert({**payload, "embedding": None}).execute()


def _set_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id, {})
        job.update(fields)
        _JOBS[job_id] = job


def _get_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _store_item(supabase, item: dict, crop: Image.Image | None, user_id: str | None) -> dict:
    """Upload the crop, insert the wardrobe row and return the stored item."""
    item_id = str(uuid.uuid4())

    # Upload crop BEFORE insert so we never persist a row without image_path.
    image_url = None
    if crop is not None:
        try:
            image_url = _upload_crop(crop, item_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to upload crop for item {item_id}: {exc}") from exc

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
    }

    response = _insert_item(supabase, payload)
    if not response.data:
        raise RuntimeError("Failed to save wardrobe item to database.")

    record = response.data[0]

    full = WardrobeItemFull(
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
    )
    return full.model_dump() if hasattr(full, "model_dump") else full.dict()


def _preview_base64(crop: Image.Image | None) -> str | None:
    """A small JPEG of the crop for the keep-or-skip sheet."""
    if crop is None:
        return None
    return base64.b64encode(_encode_crop(_downscale(crop, PREVIEW_SIDE))).decode("ascii")


def _process_upload_job(job_id: str, contents: bytes, user_id: str | None, pipeline) -> None:
    """
    Run segment/classify/store off the request thread.

    Items that are only partly in the photo are not stored. They wait in the
    job as pending until the user keeps or skips them.
    """
    try:
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            _set_job(job_id, status="failed", error="Could not decode image file.")
            return

        image = _downscale(image)

        try:
            result = pipeline.process_upload(image)
        except Exception:
            _set_job(job_id, status="failed", error="Failed to process image.")
            return

        supabase = get_supabase()
        stored_items = []
        pending = []
        held = {}

        for item in result.get("items", []):
            crop = item.pop("crop", None)
            reason = item.pop("partial", None)
            if reason:
                pending_id = str(uuid.uuid4())
                held[pending_id] = (item, crop)
                entry = PendingItem(
                    id=pending_id,
                    reason=reason,
                    type=item.get("type") or "Unknown",
                    category=item.get("category"),
                    color=item.get("color") or "Unknown",
                    preview_base64=_preview_base64(crop),
                )
                pending.append(entry.model_dump() if hasattr(entry, "model_dump") else entry.dict())
                continue

            try:
                stored_items.append(_store_item(supabase, item, crop, user_id))
            except Exception as exc:
                _set_job(job_id, status="failed", error=str(exc))
                return

        _set_job(job_id, status="done", items=stored_items, pending=pending,
                 held=held, error=None)
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc))


@router.post("/upload")
async def upload_wardrobe_item(
    request: Request,
    file: UploadFile = File(...),
    user_id: str | None = Form(None),
):
    """
    Accept an image, enqueue AI processing, return 202 + job_id immediately.
    Poll GET /wardrobe/jobs/{job_id} for status.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file.")

    job_id = str(uuid.uuid4())
    _set_job(job_id, status="processing", items=[], pending=[], held={},
             user_id=user_id, error=None)

    pipeline = request.app.state.pipeline
    _EXECUTOR.submit(_process_upload_job, job_id, contents, user_id, pipeline)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "processing"},
    )


@router.get("/jobs/{job_id}")
def get_upload_job(job_id: str):
    """Return processing | done | failed for an upload job."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": job_id,
        "status": job.get("status", "processing"),
        "items": job.get("items") or [],
        "pending": job.get("pending") or [],
        "error": job.get("error"),
    }


def _storage_path(image_url: str | None) -> str | None:
    """The object path inside the wardrobe bucket for a public photo URL."""
    if not image_url:
        return None
    marker = f"/{WARDROBE_BUCKET}/"
    url = image_url.split("?", 1)[0]
    if marker not in url:
        return None
    return url.split(marker, 1)[1] or None


def _user_id_from_token(authorization: str | None) -> str:
    """The Supabase user behind a Bearer access token, or 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing access token.")
    token = authorization[len("bearer "):].strip()
    try:
        response = get_supabase().auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid access token.")
    user = getattr(response, "user", None)
    if user is None or not getattr(user, "id", None):
        raise HTTPException(status_code=401, detail="Invalid access token.")
    return str(user.id)


@router.post("/items/delete")
def delete_wardrobe_items(body: DeleteItemsRequest, authorization: str | None = Header(None)):
    """
    Delete the caller's wardrobe items and their photos.

    Photos are uploaded with the service key, so the app's own storage
    access cannot remove them. Ids owned by other users are ignored.
    """
    user_id = _user_id_from_token(authorization)
    ids = [str(item_id) for item_id in body.ids]
    if not ids:
        return {"deleted": []}

    supabase = get_supabase()
    rows = (
        supabase.table("wardrobe_items").select("id,image_path")
        .in_("id", ids).eq("user_id", user_id).execute().data
        or []
    )
    if not rows:
        return {"deleted": []}

    paths = [p for p in (_storage_path(r.get("image_path")) for r in rows) if p]
    if paths:
        try:
            supabase.storage.from_(WARDROBE_BUCKET).remove(paths)
        except Exception as exc:
            print(f"[wardrobe] WARNING: could not remove photos {paths}: {exc}")

    owned = [str(r["id"]) for r in rows]
    supabase.table("wardrobe_items").delete().in_("id", owned).eq("user_id", user_id).execute()
    return {"deleted": owned}


@router.post("/jobs/{job_id}/keep")
def keep_pending_items(job_id: str, body: KeepItemsRequest, authorization: str | None = Header(None)):
    """
    Save the pending items the user kept and drop the rest.

    An empty id list skips every pending item of the job.
    """
    user_id = _user_id_from_token(authorization)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Not your upload.")
        held = job.get("held") or {}
        job["held"] = {}
        job["pending"] = []

    keep = {str(item_id) for item_id in body.ids}
    supabase = get_supabase()
    stored = []
    for pending_id, (item, crop) in held.items():
        if pending_id not in keep:
            continue
        try:
            stored.append(_store_item(supabase, item, crop, user_id))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"items": stored}
