"""JWT-based authentication helpers (replaces Supabase auth)."""
import os
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt
from fastapi import HTTPException
from jose import JWTError, jwt

_ALGORITHM = "HS256"
_EXPIRE_DAYS = 30


def _secret() -> str:
    key = os.getenv("JWT_SECRET_KEY")
    if not key:
        raise RuntimeError("JWT_SECRET_KEY must be set in the environment")
    return key


def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode()[:72], _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode()[:72], hashed.encode())


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_EXPIRE_DAYS)
    return jwt.encode({"sub": user_id, "exp": expire}, _secret(), algorithm=_ALGORITHM)


def user_id_from_token(authorization: str | None) -> str:
    """Extract the user id from a Bearer JWT, or raise 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing access token.")
    token = authorization[len("bearer "):].strip()
    try:
        payload = jwt.decode(token, _secret(), algorithms=[_ALGORITHM])
        user_id: str | None = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid access token.")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid access token.")
    return user_id
