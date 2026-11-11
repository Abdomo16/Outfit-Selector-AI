import io
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from api import db
from api.auth import user_id_from_token
from api.models import WardrobeItem
from api.schemas import DeleteItemsRequest, KeepItemsRequest, PendingItem, WardrobeItemFull
from api.services import upload_jobs
from api.services import wardrobe_storage as storage
from api.services.wardrobe_items import as_dict, item_to_full, store_item

router = APIRouter()

_EXECUTOR = ThreadPoolExecutor(max_workers=1)


class UpdateItemRequest(BaseModel):
    category: str | None = None
    type: str | None = None
    color: str | None = None
    hex: str | None = None
    pattern: str | None = None
    style: str | None = None
    season: str | None = None


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
                with db.get_session() as session:
                    stored_items.append(store_item(session, item, crop, user_id, pipeline))
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


@router.get("/items", response_model=List[WardrobeItemFull])
def list_wardrobe_items(authorization: str | None = Header(None)):
    """Return all wardrobe items belonging to the authenticated user."""
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        items = session.query(WardrobeItem).filter(WardrobeItem.user_id == user_id).all()
        return [as_dict(item_to_full(item)) for item in items]


@router.get("/items/{item_id}", response_model=WardrobeItemFull)
def get_wardrobe_item(item_id: str, authorization: str | None = Header(None)):
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        item = (
            session.query(WardrobeItem)
            .filter(WardrobeItem.id == item_id, WardrobeItem.user_id == user_id)
            .first()
        )
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found.")
        return as_dict(item_to_full(item))


@router.put("/items/{item_id}", response_model=WardrobeItemFull)
def update_wardrobe_item(
    item_id: str,
    body: UpdateItemRequest,
    authorization: str | None = Header(None),
):
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        item = (
            session.query(WardrobeItem)
            .filter(WardrobeItem.id == item_id, WardrobeItem.user_id == user_id)
            .first()
        )
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found.")
        for field, value in body.model_dump(exclude_none=True).items():
            setattr(item, field, value)
        session.flush()
        return as_dict(item_to_full(item))


@router.post("/items/delete")
def delete_wardrobe_items(body: DeleteItemsRequest, authorization: str | None = Header(None)):
    """Delete the caller's wardrobe items and their local photos."""
    user_id = user_id_from_token(authorization)
    ids = [str(item_id) for item_id in body.ids]
    if not ids:
        return {"deleted": []}

    with db.get_session() as session:
        items = (
            session.query(WardrobeItem)
            .filter(WardrobeItem.id.in_(ids), WardrobeItem.user_id == user_id)
            .all()
        )
        if not items:
            return {"deleted": []}

        storage.remove_photos([
            path
            for item in items
            for raw in (item.image_path, item.cutout_path)
            for path in [storage.storage_path(raw)]
            if path
        ])

        owned = [str(item.id) for item in items]
        for item in items:
            session.delete(item)

    return {"deleted": owned}


@router.post("/jobs/{job_id}/keep")
def keep_pending_items(
    job_id: str,
    body: KeepItemsRequest,
    request: Request,
    authorization: str | None = Header(None),
):
    """Save the pending items the user kept and drop the rest."""
    user_id = user_id_from_token(authorization)
    try:
        held = upload_jobs.take_held(job_id, user_id)
    except upload_jobs.JobNotFound:
        raise HTTPException(status_code=404, detail="Job not found.")
    except upload_jobs.NotYourJob:
        raise HTTPException(status_code=403, detail="Not your upload.")

    keep = {str(item_id) for item_id in body.ids}
    pipeline = request.app.state.pipeline
    stored = []
    for pending_id, (item, crop) in held.items():
        if pending_id not in keep:
            continue
        try:
            with db.get_session() as session:
                stored.append(store_item(session, item, crop, user_id, pipeline))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"items": stored}
