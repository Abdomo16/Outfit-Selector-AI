import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from api.schemas import WardrobeItemFull
from api.supabase_client import get_supabase

router = APIRouter()

WARDROBE_BUCKET = "wardrobe-images"
MAX_IMAGE_SIDE = 1536
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


def _set_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id, {})
        job.update(fields)
        _JOBS[job_id] = job


def _get_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _process_upload_job(job_id: str, contents: bytes, user_id: str | None, pipeline) -> None:
    """Run segment/classify/store off the request thread."""
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

        for item in result.get("items", []):
            crop = item.pop("crop", None)
            item_id = str(uuid.uuid4())

            # Upload crop BEFORE insert so we never persist a row without image_path.
            image_url = None
            if crop is not None:
                try:
                    image_url = _upload_crop(crop, item_id)
                except Exception as exc:
                    _set_job(
                        job_id,
                        status="failed",
                        error=f"Failed to upload crop for item {item_id}: {exc}",
                    )
                    return

            payload = {
                "id": item_id,
                "user_id": user_id,
                "type": item.get("type"),
                "confidence": float(item.get("confidence", 0)),
                "color": item.get("color"),
                "hex": item.get("hex"),
                "pattern": item.get("pattern"),
                "style": item.get("style"),
                "season": item.get("season"),
                "embedding": item.get("embedding"),
                "image_path": image_url,
            }

            response = supabase.table("wardrobe_items").insert(payload).execute()
            if not response.data:
                _set_job(job_id, status="failed", error="Failed to save wardrobe item to database.")
                return

            record = response.data[0]

            full = WardrobeItemFull(
                id=record["id"],
                type=record["type"],
                confidence=record["confidence"],
                color=record["color"],
                hex=record["hex"],
                pattern=record.get("pattern"),
                style=record.get("style"),
                occasion=item.get("occasion"),
                season=record.get("season"),
                embedding=record.get("embedding"),
                image_path=record.get("image_path") or image_url,
            )
            stored_items.append(
                full.model_dump() if hasattr(full, "model_dump") else full.dict()
            )

        _set_job(job_id, status="done", items=stored_items, error=None)
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
    _set_job(job_id, status="processing", items=[], error=None)

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
        "error": job.get("error"),
    }
