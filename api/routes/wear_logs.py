from datetime import date, timedelta

from fastapi import APIRouter, Header
from pydantic import BaseModel

from api import db
from api.auth import user_id_from_token
from api.models import WearLog

router = APIRouter()


class WornRequest(BaseModel):
    worn_on: date
    outfit_data: dict


@router.post("/")
def save_worn(body: WornRequest, authorization: str | None = Header(None)):
    """Record or overwrite what was worn on a given day (upsert by user+date)."""
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        existing = (
            session.query(WearLog)
            .filter(WearLog.user_id == user_id, WearLog.worn_on == body.worn_on)
            .first()
        )
        if existing:
            existing.outfit_data = body.outfit_data
        else:
            session.add(WearLog(user_id=user_id, worn_on=body.worn_on, outfit_data=body.outfit_data))
    return {"ok": True}


@router.get("/recent-ids")
def recent_item_ids(days: int = 3, authorization: str | None = Header(None)):
    user_id = user_id_from_token(authorization)
    start = date.today() - timedelta(days=days)
    with db.get_session() as session:
        rows = (
            session.query(WearLog.outfit_data)
            .filter(WearLog.user_id == user_id, WearLog.worn_on >= start)
            .all()
        )
    ids: set[str] = set()
    for (data,) in rows:
        if not isinstance(data, dict):
            continue
        for item in data.get("items", []):
            if isinstance(item, dict):
                item_id = str(item.get("id", ""))
                if item_id:
                    ids.add(item_id)
    return {"ids": list(ids)}


@router.get("/last-worn")
def last_worn_dates(days: int = 60, authorization: str | None = Header(None)):
    """Return a map of item_id → ISO date for pieces worn in the past [days] days."""
    user_id = user_id_from_token(authorization)
    start = date.today() - timedelta(days=days)
    with db.get_session() as session:
        rows = (
            session.query(WearLog.worn_on, WearLog.outfit_data)
            .filter(WearLog.user_id == user_id, WearLog.worn_on >= start)
            .all()
        )
    dates: dict[str, str] = {}
    for worn_on, data in rows:
        if not isinstance(data, dict):
            continue
        for item in data.get("items", []):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            worn_str = str(worn_on)
            if item_id not in dates or dates[item_id] < worn_str:
                dates[item_id] = worn_str
    return dates
