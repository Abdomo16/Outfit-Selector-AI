import io
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from api import supabase_client
from api.auth import user_id_from_token
from api.schemas import DeleteItemsRequest, KeepItemsRequest, PendingItem
from api.services import upload_jobs
from api.services import wardrobe_storage as storage
from api.services.wardrobe_items import as_dict, store_item

router = APIRouter()

_EXECUTOR = ThreadPoolExecutor(max_workers=1)


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
            upload_jobs.set_job(job_id, status="failed", error="Could not decode image file.")
            return

        image = storage.downscale(image)

        try:
            result = pipeline.process_upload(image)
        except Exception:
            upload_jobs.set_job(job_id, status="failed", error="Failed to process image.")
            return

        supabase = supabase_client.get_supabase()
        stored_items = []
        pending = []
        held = {}

        for item in result.get("items", []):
            crop = item.pop("crop", None)
            reason = item.pop("partial", None)
            if reason:
                pending_id = str(uuid.uuid4())
                held[pending_id] = (item, crop)
                pending.append(as_dict(PendingItem(
                    id=pending_id,
                    reason=reason,
                    type=item.get("type") or "Unknown",
                    category=item.get("category"),
                    color=item.get("color") or "Unknown",
                    preview_base64=storage.preview_base64(crop),
                )))
                continue

            try:
                stored_items.append(store_item(supabase, item, crop, user_id, pipeline))
            except Exception as exc:
                upload_jobs.set_job(job_id, status="failed", error=str(exc))
                return

        upload_jobs.set_job(job_id, status="done", items=stored_items, pending=pending,
                            held=held, error=None)
    except Exception as exc:
        upload_jobs.set_job(job_id, status="failed", error=str(exc))


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
    upload_jobs.set_job(job_id, status="processing", items=[], pending=[], held={},
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
    job = upload_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": job_id,
        "status": job.get("status", "processing"),
        "items": job.get("items") or [],
        "pending": job.get("pending") or [],
        "error": job.get("error"),
    }


@router.post("/items/delete")
def delete_wardrobe_items(body: DeleteItemsRequest, authorization: str | None = Header(None)):
    """
    Delete the caller's wardrobe items and their photos.

    Photos are uploaded with the service key, so the app's own storage
    access cannot remove them. Ids owned by other users are ignored.
    """
    user_id = user_id_from_token(authorization)
    ids = [str(item_id) for item_id in body.ids]
    if not ids:
        return {"deleted": []}

    supabase = supabase_client.get_supabase()
    rows = (
        supabase.table("wardrobe_items").select("id,image_path,cutout_path")
        .in_("id", ids).eq("user_id", user_id).execute().data
        or []
    )
    if not rows:
        return {"deleted": []}

    storage.remove_photos([
        path for row in rows
        for raw in (row.get("image_path"), row.get("cutout_path"))
        for path in [storage.storage_path(raw)] if path
    ])

    owned = [str(row["id"]) for row in rows]
    supabase.table("wardrobe_items").delete().in_("id", owned).eq("user_id", user_id).execute()
    return {"deleted": owned}


@router.post("/jobs/{job_id}/keep")
def keep_pending_items(job_id: str, body: KeepItemsRequest, request: Request, authorization: str | None = Header(None)):
    """
    Save the pending items the user kept and drop the rest.

    An empty id list skips every pending item of the job.
    """
    user_id = user_id_from_token(authorization)
    try:
        held = upload_jobs.take_held(job_id, user_id)
    except upload_jobs.JobNotFound:
        raise HTTPException(status_code=404, detail="Job not found.")
    except upload_jobs.NotYourJob:
        raise HTTPException(status_code=403, detail="Not your upload.")

    keep = {str(item_id) for item_id in body.ids}
    supabase = supabase_client.get_supabase()
    pipeline = request.app.state.pipeline
    stored = []
    for pending_id, (item, crop) in held.items():
        if pending_id not in keep:
            continue
        try:
            stored.append(store_item(supabase, item, crop, user_id, pipeline))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"items": stored}
