"""SQLAlchemy ORM models for the standalone PostgreSQL database."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Date, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMPTZ, UUID

from api.db import Base


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email         = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    name          = Column(String, nullable=False, default="User")
    avatar_url    = Column(String)
    is_premium    = Column(Boolean, default=False)
    created_at    = Column(TIMESTAMPTZ, default=_now)
    updated_at    = Column(TIMESTAMPTZ, default=_now, onupdate=_now)


class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id         = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    category        = Column(String)
    type            = Column(String)
    color           = Column(String)
    hex             = Column(String)
    pattern         = Column(String)
    style           = Column(String)
    season          = Column(String)
    confidence      = Column(Float)
    image_path      = Column(Text)
    cutout_path     = Column(Text)
    embedding       = Column(ARRAY(Float))
    secondary_color = Column(String)
    secondary_hex   = Column(String)
    created_at      = Column(TIMESTAMPTZ, default=_now)


class SavedOutfit(Base):
    __tablename__ = "saved_outfits"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    outfit_data = Column(JSONB, nullable=False)
    created_at  = Column(TIMESTAMPTZ, default=_now)


class WearLog(Base):
    __tablename__ = "wear_logs"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    worn_on     = Column(Date, nullable=False)
    outfit_data = Column(JSONB, nullable=False)
    created_at  = Column(TIMESTAMPTZ, default=_now)


class OutfitPlan(Base):
    __tablename__ = "outfit_plans"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind       = Column(String, nullable=False)
    title      = Column(String)
    starts_on  = Column(Date, nullable=False)
    days       = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMPTZ, default=_now)
