"""Unit tests for trading_common.data_clients._cache.

Covers set/get roundtrip, TTL expiry (via monkeypatched time.monotonic),
and clear().
"""
from __future__ import annotations

import pytest

from trading_common.data_clients import _cache


@pytest.fixture(autouse=True)
def _clear_cache_around_test():
    _cache.clear()
    yield
    _cache.clear()


def test_set_get_roundtrip():
    _cache.set("key1", {"value": 42}, ttl_seconds=60)
    assert _cache.get("key1") == {"value": 42}


def test_get_missing_key_returns_none():
    assert _cache.get("does-not-exist") is None


def test_ttl_expiry_via_monkeypatched_monotonic(monkeypatch):
    fake_time = [1000.0]
    monkeypatch.setattr(_cache.time, "monotonic", lambda: fake_time[0])

    _cache.set("key1", "value1", ttl_seconds=10)
    # Still within TTL
    fake_time[0] = 1005.0
    assert _cache.get("key1") == "value1"

    # Past TTL
    fake_time[0] = 1011.0
    assert _cache.get("key1") is None

    # Entry should have been evicted from the store
    assert "key1" not in _cache._store


def test_clear_removes_all_entries():
    _cache.set("key1", "a", ttl_seconds=60)
    _cache.set("key2", "b", ttl_seconds=60)
    _cache.clear()
    assert _cache.get("key1") is None
    assert _cache.get("key2") is None
    assert _cache._store == {}
