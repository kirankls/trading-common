# Ported from D:\chanakya\options_advisor\storage\repositories\base.py
"""Base repository providing generic CRUD helpers for all domain repositories."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from trading_common.storage.db import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(ABC, Generic[ModelT]):
    """Generic async repository base class.

    Subclasses must set the ``model`` class attribute to the SQLAlchemy
    ORM model they manage.
    """

    model: type[ModelT]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "model") or isinstance(cls.__dict__.get("model"), type(None)):
            # Allow abstract intermediaries; only concrete subclasses must define model.
            if not getattr(cls, "__abstractmethods__", None):
                raise TypeError(f"{cls.__name__} must define a 'model' class attribute")

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @abstractmethod
    async def get_by_id(self, record_id: Any) -> ModelT | None:
        """Return a single record by its primary key, or None if not found."""

    @abstractmethod
    async def create(self, **kwargs: Any) -> ModelT:
        """Instantiate and persist a new record, returning the saved instance."""

    @abstractmethod
    async def update(self, record_id: Any, **kwargs: Any) -> ModelT | None:
        """Update fields on an existing record and return the updated instance."""

    @abstractmethod
    async def delete(self, record_id: Any) -> bool:
        """Delete a record by its primary key. Returns True if a row was deleted."""
