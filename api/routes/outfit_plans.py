from datetime import date
from typing import List

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from api import db
from api.auth import user_id_from_token
from api.models import OutfitPlan

router = APIRouter()


class PlanRequest(BaseModel):
    kind: str
    title: str = ""
    starts_on: date
    days: List[dict]


class UpdateDaysRequest(BaseModel):
    days: List[dict]


@router.post("/")
def create_plan(body: PlanRequest, authorization: str | None = Header(None)):
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        plan = OutfitPlan(
            user_id=user_id,
            kind=body.kind,
            title=body.title,
            starts_on=body.starts_on,
            days=body.days,
        )
        session.add(plan)
        session.flush()
        return {"id": str(plan.id)}


@router.put("/{plan_id}/days")
def update_plan_days(plan_id: str, body: UpdateDaysRequest, authorization: str | None = Header(None)):
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        plan = (
            session.query(OutfitPlan)
            .filter(OutfitPlan.id == plan_id, OutfitPlan.user_id == user_id)
            .first()
        )
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found.")
        plan.days = body.days
    return {"ok": True}


@router.get("/latest")
def latest_plan(kind: str, authorization: str | None = Header(None)):
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        plan = (
            session.query(OutfitPlan)
            .filter(OutfitPlan.user_id == user_id, OutfitPlan.kind == kind)
            .order_by(OutfitPlan.created_at.desc())
            .first()
        )
        if not plan:
            return None
        return {
            "id": str(plan.id),
            "kind": plan.kind,
            "title": plan.title,
            "starts_on": str(plan.starts_on),
            "days": plan.days,
        }
