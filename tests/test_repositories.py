"""Unit tests for trading_common.storage.repositories.base.BaseRepository.

Tests the generic repository pattern against an in-memory SQLite (aiosqlite)
engine with a concrete throwaway model/repository. Nothing Postgres-specific
is exercised by BaseRepository itself (no LISTEN/NOTIFY, no dialect-specific
SQL) — the base class is DB-agnostic, so SQLite fully covers it.
"""
from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trading_common.storage.db import Base
from trading_common.storage.repositories.base import BaseRepository


class _Item(Base):
    __tablename__ = "items_test"
    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String, nullable=False)


class ItemRepository(BaseRepository[_Item]):
    model = _Item

    async def get_by_id(self, record_id):
        return await self.session.get(_Item, record_id)

    async def create(self, **kwargs):
        obj = _Item(**kwargs)
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def update(self, record_id, **kwargs):
        obj = await self.get_by_id(record_id)
        if obj is None:
            return None
        for key, value in kwargs.items():
            setattr(obj, key, value)
        await self.session.flush()
        return obj

    async def delete(self, record_id):
        obj = await self.get_by_id(record_id)
        if obj is None:
            return False
        await self.session.delete(obj)
        await self.session.flush()
        return True


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def test_base_repository_requires_model_attribute():
    with pytest.raises(TypeError, match="must define a 'model' class attribute"):

        class BadRepository(BaseRepository):
            async def get_by_id(self, record_id):
                return None

            async def create(self, **kwargs):
                return None

            async def update(self, record_id, **kwargs):
                return None

            async def delete(self, record_id):
                return False


def test_base_repository_cannot_be_instantiated_directly(session):
    with pytest.raises(TypeError):
        BaseRepository(session)


async def test_create_and_get_by_id(session):
    repo = ItemRepository(session)
    created = await repo.create(label="widget")
    assert created.id is not None

    fetched = await repo.get_by_id(created.id)
    assert fetched is not None
    assert fetched.label == "widget"


async def test_get_by_id_missing_returns_none(session):
    repo = ItemRepository(session)
    assert await repo.get_by_id(9999) is None


async def test_update_existing_record(session):
    repo = ItemRepository(session)
    created = await repo.create(label="original")

    updated = await repo.update(created.id, label="changed")
    assert updated is not None
    assert updated.label == "changed"

    refetched = await repo.get_by_id(created.id)
    assert refetched.label == "changed"


async def test_update_missing_record_returns_none(session):
    repo = ItemRepository(session)
    assert await repo.update(9999, label="nope") is None


async def test_delete_existing_record(session):
    repo = ItemRepository(session)
    created = await repo.create(label="to-delete")

    deleted = await repo.delete(created.id)
    assert deleted is True
    assert await repo.get_by_id(created.id) is None


async def test_delete_missing_record_returns_false(session):
    repo = ItemRepository(session)
    assert await repo.delete(9999) is False
