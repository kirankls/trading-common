# Ported from D:\chanakya\options_advisor\data_clients\_cache.py
"""Lightweight in-process TTL cache for external API calls.

Dict operations are GIL-atomic in CPython. With FastAPI's single-threaded
asyncio event loop, there are no concurrent coroutine writes — no lock needed.
"""
from __future__ import annotations

import time
from typing import Any

_store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)


def get(key: str) -> Any | None:
    entry = _store.get(key)
    if entry and time.monotonic() < entry[1]:
        return entry[0]
    if entry:
        del _store[key]
    return None


def set(key: str, value: Any, ttl_seconds: int) -> None:
    _store[key] = (value, time.monotonic() + ttl_seconds)


def clear() -> None:
    _store.clear()
