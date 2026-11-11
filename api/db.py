"""SQLAlchemy engine and session factory."""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

_engine = None
_SessionFactory = None


def _get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL must be set in the environment")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def _get_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=_get_engine(), autocommit=False, autoflush=False
        )
    return _SessionFactory


@contextmanager
def get_session():
    session = _get_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables that do not yet exist (useful for local dev without init.sql)."""
    from api import models  # noqa: F401 — registers tables with Base
    Base.metadata.create_all(bind=_get_engine())
