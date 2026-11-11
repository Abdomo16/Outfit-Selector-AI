from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from api import db
from api.auth import create_access_token, hash_password, user_id_from_token, verify_password
from api.models import User

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = "User"


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    user_id: str
    email: str
    name: str


class UserResponse(BaseModel):
    user_id: str
    email: str
    name: str


@router.post("/register", response_model=TokenResponse)
def register(body: RegisterRequest):
    with db.get_session() as session:
        if session.query(User).filter(User.email == body.email).first():
            raise HTTPException(status_code=409, detail="Email already registered.")
        user = User(
            email=body.email,
            password_hash=hash_password(body.password),
            name=body.name,
        )
        session.add(user)
        session.flush()
        return TokenResponse(
            access_token=create_access_token(str(user.id)),
            user_id=str(user.id),
            email=user.email,
            name=user.name,
        )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    with db.get_session() as session:
        user = session.query(User).filter(User.email == body.email).first()
        if not user or not verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        return TokenResponse(
            access_token=create_access_token(str(user.id)),
            user_id=str(user.id),
            email=user.email,
            name=user.name,
        )


@router.get("/me", response_model=UserResponse)
def me(authorization: str | None = Header(None)):
    """Verify the token and return the current user's profile."""
    user_id = user_id_from_token(authorization)
    with db.get_session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        return UserResponse(user_id=str(user.id), email=user.email, name=user.name)
