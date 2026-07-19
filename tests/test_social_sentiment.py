"""Unit tests for trading_common.data_clients.social_sentiment.
fetch_wsb_mentions -- ported from the chanakya repo alongside the module
itself (no dedicated test existed there; this is new coverage written for
the extracted package).

Rules: no real network calls -- httpx is always mocked. The in-process
cache is cleared before each test so results don't leak across tests.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_common.data_clients import _cache
from trading_common.data_clients.social_sentiment import fetch_wsb_mentions


@pytest.fixture(autouse=True)
def _clear_cache():
    _cache.clear()
    yield
    _cache.clear()


def _mock_client(status_code: int, payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    client = AsyncMock()
    client.get.return_value = response
    cm = MagicMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm


class TestFetchWsbMentions:
    def test_ticker_found_parses_mention_and_rank_data(self):
        payload = {
            "results": [
                {"ticker": "GME", "mentions": 500, "mentions_24h_ago": 50, "rank": 3, "rank_24h_ago": 10}
            ]
        }
        with patch("trading_common.data_clients.social_sentiment.httpx.AsyncClient", return_value=_mock_client(200, payload)):
            result = asyncio.run(fetch_wsb_mentions("GME"))

        assert result["mentions"] == 500
        assert result["rank"] == 3
        assert result["rank_change"] == 7  # was 10, now 3 -> rising
        assert result["mention_velocity"] == 10.0
        assert result["squeeze_watch"] is True  # velocity (10.0) > 5.0

    def test_ticker_not_found_returns_zeroed_result(self):
        payload = {"results": [{"ticker": "OTHER", "mentions": 10, "mentions_24h_ago": 5}]}
        with patch("trading_common.data_clients.social_sentiment.httpx.AsyncClient", return_value=_mock_client(200, payload)):
            result = asyncio.run(fetch_wsb_mentions("GME"))

        assert result == {
            "mentions": 0, "rank": None, "rank_24h_ago": None,
            "rank_change": None, "mention_velocity": 0.0, "squeeze_watch": False,
        }

    def test_non_200_status_returns_empty_dict(self):
        with patch("trading_common.data_clients.social_sentiment.httpx.AsyncClient", return_value=_mock_client(500)):
            result = asyncio.run(fetch_wsb_mentions("GME"))

        assert result == {}

    def test_second_call_is_served_from_cache(self):
        payload = {"results": [{"ticker": "GME", "mentions": 500, "mentions_24h_ago": 100, "rank": 3, "rank_24h_ago": 10}]}
        with patch("trading_common.data_clients.social_sentiment.httpx.AsyncClient", return_value=_mock_client(200, payload)) as mock_ac:
            asyncio.run(fetch_wsb_mentions("GME"))
            asyncio.run(fetch_wsb_mentions("GME"))

        assert mock_ac.call_count == 1

    def test_exception_returns_empty_dict(self):
        with patch("trading_common.data_clients.social_sentiment.httpx.AsyncClient", side_effect=RuntimeError("boom")):
            result = asyncio.run(fetch_wsb_mentions("GME"))

        assert result == {}
