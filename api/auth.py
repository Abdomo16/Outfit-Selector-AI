from fastapi import HTTPException

from api import supabase_client


def user_id_from_token(authorization: str | None) -> str:
    """The Supabase user behind a Bearer access token, or 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing access token.")
    token = authorization[len("bearer "):].strip()
    try:
        response = supabase_client.get_supabase().auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid access token.")
    user = getattr(response, "user", None)
    if user is None or not getattr(user, "id", None):
        raise HTTPException(status_code=401, detail="Invalid access token.")
    return str(user.id)
