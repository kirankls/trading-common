"""Unit tests for trading_common.storage.db — async engine/session pattern.

Exercises the engine/session-factory helpers against an in-memory SQLite
(aiosqlite) database. The WAL-mode pragma and check_same_thread connect_args
are SQLite-specific paths in _get_engine() and are exercised implicitly here
since settings.database_url defaults to a sqlite+aiosqlite URL. Postgres-only
behavior (asyncpg driver normalisation, no connect_args) is not exercised by
these tests since no live Postgres instance is available; test_normalise_db_url
covers the URL-rewrite logic directly without needing a real connection.
"""
from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession

import trading_common.storage.db as db_module
from trading_common.storage.db import Base, _normalise_db_url, get_db, init_db


@pytest.fixture(autouse=True)
def reset_engine_singletons(monkeypatch, tmp_path):
    """Ensure each test gets a fresh in-memory engine/session factory."""
    db_module._engine = None
    db_module._session_factory = None

    from trading_common.config.settings import settings

    monkeypatch.setattr(settings, "async_database_url", "")
    monkeypatch.setattr(settings, "database_url", "sqlite+aiosqlite:///:memory:")
    yield
    db_module._engine = None
    db_module._session_factory = None


class _Widget(Base):
    __tablename__ = "widgets_test"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)


def test_normalise_db_url_postgres_scheme():
    assert _normalise_db_url("postgres://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"


def test_normalise_db_url_postgresql_scheme():
    assert _normalise_db_url("postgresql://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"


def test_normalise_db_url_leaves_sqlite_unchanged():
    url = "sqlite+aiosqlite:///./data/x.db"
    assert _normalise_db_url(url) == url


async def test_init_db_creates_tables_and_get_db_yields_working_session():
    await init_db()

    async for session in get_db():
        assert isinstance(session, AsyncSession)
        session.add(_Widget(id=1, name="gizmo"))
        await session.flush()
        from sqlalchemy import select
        result = await session.execute(select(_Widget).where(_Widget.id == 1))
        row = result.scalar_one()
        assert row.name == "gizmo"


async def test_get_db_rolls_back_on_exception():
    await init_db()

    with pytest.raises(ValueError):
        async for session in get_db():
            session.add(_Widget(id=2, name="broken"))
            await session.flush()
            raise ValueError("boom")

    # A fresh session should not see the row from the rolled-back transaction.
    async for session in get_db():
        from sqlalchemy import select
        result = await session.execute(select(_Widget).where(_Widget.id == 2))
        assert result.scalar_one_or_none() is None
