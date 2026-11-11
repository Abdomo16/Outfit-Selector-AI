from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from api import db
from api.auth import user_id_from_token
from api.models import SavedOutfit

router = APIRouter()


class SaveOutfitRequest(BaseModel):
    outfit_data: dict


@router.get("/")
def list_saved_outfits(authorization: str | None = Header(None)):
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        rows = (
            session.query(SavedOutfit)
            .filter(SavedOutfit.user_id == user_id)
            .order_by(SavedOutfit.created_at.desc())
            .all()
        )
        return [
            {"id": str(r.id), "outfit_data": r.outfit_data, "created_at": r.created_at.isoformat()}
            for r in rows
        ]


@router.post("/")
def create_saved_outfit(body: SaveOutfitRequest, authorization: str | None = Header(None)):
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        outfit = SavedOutfit(user_id=user_id, outfit_data=body.outfit_data)
        session.add(outfit)
        session.flush()
        return {
            "id": str(outfit.id),
            "outfit_data": outfit.outfit_data,
            "created_at": outfit.created_at.isoformat(),
        }


@router.delete("/{outfit_id}")
def delete_saved_outfit(outfit_id: str, authorization: str | None = Header(None)):
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        outfit = (
            session.query(SavedOutfit)
            .filter(SavedOutfit.id == outfit_id, SavedOutfit.user_id == user_id)
            .first()
        )
        if outfit:
            session.delete(outfit)
    return {"deleted": outfit_id}
