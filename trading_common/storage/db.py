# Ported from D:\chanakya\options_advisor\storage\db.py
"""Async SQLAlchemy engine, session factory, and database initialisation.

This module ports the reusable async engine/session pattern only. It defines
its own minimal declarative `Base` for use by the generic repository pattern
in trading_common.storage.repositories.base; the top-level daytrader/storage/
layer is expected to define its own domain models (orders, fills, positions,
etc.) against its own Base (or reuse this one if that proves convenient) and
its own alembic/env.py wiring — that is a separate task, not part of this
package.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from trading_common.config.settings import settings


class Base(DeclarativeBase):
    """Shared declarative base for trading_common ORM models."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _normalise_db_url(url: str) -> str:
    """Ensure PostgreSQL URLs use the asyncpg driver prefix required by SQLAlchemy."""
    if url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    return url


def _get_engine() -> AsyncEngine:
    """Return (and lazily create) the shared async engine.

    Uses ``settings.async_database_url`` when set (PostgreSQL in production),
    otherwise falls back to ``settings.database_url`` (SQLite in development).
    PostgreSQL connections do not need ``check_same_thread``; WAL-mode pragma is
    SQLite-only and is skipped for PostgreSQL.
    """
    global _engine
    if _engine is None:
        db_url = _normalise_db_url(
            settings.async_database_url if settings.async_database_url else settings.database_url
        )
        connect_args: dict[str, Any] = {}
        is_sqlite = db_url.startswith("sqlite")
        if is_sqlite:
            connect_args = {"check_same_thread": False}
        _engine = create_async_engine(
            db_url,
            connect_args=connect_args,
            echo=(settings.app_env == "development"),
        )
        if is_sqlite:
            # WAL mode must be issued per connection; SQLAlchemy's connect event
            # fires on each new raw DBAPI connection before it enters the pool.
            @event.listens_for(_engine.sync_engine, "connect")
            def _set_wal(dbapi_conn: Any, _connection_record: Any) -> None:
                dbapi_conn.execute("PRAGMA journal_mode=WAL")

    return _engine


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return (and lazily create) the session factory bound to the current engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _session_factory


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for use with FastAPI Depends()."""
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create all tables if they do not exist (development / test convenience).

    In production, use `alembic upgrade head` instead.

    Note: unlike the source (options_advisor.storage.db.init_db), this does not
    run any Postgres/SQLite idempotent column migrations — those were specific
    to options-advisor tables (chat_sessions, analysis_history, trade_log) that
    do not exist in trading_common. The top-level daytrader/storage/ layer
    should add its own equivalent post-create_all migrations if it needs them.
    """
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
