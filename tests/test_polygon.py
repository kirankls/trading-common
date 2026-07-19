"""Unit tests for trading_common.data_clients.polygon.

Mocks the httpx layer with respx. Unlike finnhub.py, polygon.py calls
r.raise_for_status() on most endpoints, so 4xx/5xx responses raise
httpx.HTTPStatusError which is caught by the broad except Exception in each
fetch function and converted to {} / [] / None — matching the original
chanakya client's behaviour (no custom FetchErrorType mapping exists here).
"""
from __future__ import annotations

import httpx
import pytest
import respx

from trading_common.data_clients import _cache, polygon


@pytest.fixture(autouse=True)
def _clear_cache_and_key(monkeypatch):
    _cache.clear()
    monkeypatch.setattr(polygon, "_key", lambda: "test-polygon-key")
    yield
    _cache.clear()


@respx.mock
async def test_fetch_ticker_news_success():
    respx.get(f"{polygon._BASE}/v2/reference/news").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Company announces buyback",
                        "published_utc": "2026-07-01T12:00:00Z",
                        "article_url": "https://example.com/a1",
                        "publisher": {"name": "Polygon News"},
                        "description": "Some description",
                        "tickers": ["AAPL"],
                        "insights": [{"sentiment": "positive", "sentiment_reasoning": "buyback"}],
                    }
                ]
            },
        )
    )
    result = await polygon.fetch_ticker_news("AAPL")
    assert len(result) == 1
    assert result[0]["title"] == "Company announces buyback"
    assert result[0]["sentiment"] == "positive"
    assert result[0]["source"] == "Polygon News"


@respx.mock
async def test_fetch_ticker_news_auth_failure_returns_empty_list():
    # 401 is not retried (only 429/5xx trigger retry) so raise_for_status()
    # raises immediately, caught by except Exception -> [].
    respx.get(f"{polygon._BASE}/v2/reference/news").mock(return_value=httpx.Response(401, json={"error": "unauthorized"}))
    result = await polygon.fetch_ticker_news("AAPL")
    assert result == []


@respx.mock
async def test_fetch_ticker_news_rate_limited_eventually_returns_empty_list():
    # Always 429 -> retried up to _max_retries times, then raise_for_status()
    # raises on the final (still 429) response -> [].
    route = respx.get(f"{polygon._BASE}/v2/reference/news").mock(return_value=httpx.Response(429))
    result = await polygon.fetch_ticker_news("AAPL")
    assert result == []
    assert route.call_count == 3  # _max_retries in fetch_ticker_news


@respx.mock
async def test_fetch_ticker_news_timeout_returns_empty_list():
    respx.get(f"{polygon._BASE}/v2/reference/news").mock(side_effect=httpx.TimeoutException("timed out"))
    result = await polygon.fetch_ticker_news("AAPL")
    assert result == []


async def test_fetch_ticker_news_no_key_returns_empty_list(monkeypatch):
    monkeypatch.setattr(polygon, "_key", lambda: "")
    result = await polygon.fetch_ticker_news("AAPL")
    assert result == []


@respx.mock
async def test_fetch_earnings_success():
    respx.get(f"{polygon._BASE}/vX/reference/financials").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period_of_report_date": "2026-03-31",
                        "fiscal_period": "Q1",
                        "fiscal_year": "2026",
                        "financials": {
                            "income_statement": {
                                "basic_earnings_per_share": {"value": 1.5},
                                "revenues": {"value": 1_000_000},
                            }
                        },
                    },
                    {
                        "fiscal_period": "Q4",
                        "fiscal_year": "2025",
                        "financials": {
                            "income_statement": {
                                "basic_earnings_per_share": {"value": 1.2},
                                "revenues": {"value": 900_000},
                            }
                        },
                    },
                ]
            },
        )
    )
    result = await polygon.fetch_earnings("AAPL")
    assert result["eps_actual"] == 1.5
    assert result["revenue_actual"] == 1_000_000
    assert result["eps_trend"] == "improving"
    assert result["revenue_trend"] == "improving"


@respx.mock
async def test_fetch_earnings_no_results_returns_empty_dict():
    respx.get(f"{polygon._BASE}/vX/reference/financials").mock(return_value=httpx.Response(200, json={"results": []}))
    result = await polygon.fetch_earnings("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_earnings_server_error_returns_empty_dict():
    respx.get(f"{polygon._BASE}/vX/reference/financials").mock(return_value=httpx.Response(500))
    result = await polygon.fetch_earnings("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_rsi_success():
    respx.get(f"{polygon._BASE}/v1/indicators/rsi/AAPL").mock(
        return_value=httpx.Response(200, json={"results": {"values": [{"value": 55.4321}]}})
    )
    result = await polygon.fetch_rsi("AAPL")
    assert result == 55.43


@respx.mock
async def test_fetch_rsi_no_values_returns_none():
    respx.get(f"{polygon._BASE}/v1/indicators/rsi/AAPL").mock(
        return_value=httpx.Response(200, json={"results": {"values": []}})
    )
    result = await polygon.fetch_rsi("AAPL")
    assert result is None


@respx.mock
async def test_fetch_rsi_error_returns_none():
    respx.get(f"{polygon._BASE}/v1/indicators/rsi/AAPL").mock(return_value=httpx.Response(403))
    result = await polygon.fetch_rsi("AAPL")
    assert result is None


@respx.mock
async def test_fetch_polygon_news_sentiment_aggregates_correctly():
    respx.get(f"{polygon._BASE}/v2/reference/news").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Bullish news",
                        "published_utc": "t1",
                        "article_url": "u1",
                        "publisher": {"name": "src1"},
                        "insights": [{"sentiment": "positive"}],
                    },
                    {
                        "title": "Bearish news",
                        "published_utc": "t2",
                        "article_url": "u2",
                        "publisher": {"name": "src2"},
                        "insights": [{"sentiment": "negative"}],
                    },
                ]
            },
        )
    )
    result = await polygon.fetch_polygon_news_sentiment("AAPL")
    assert result["bullish_count"] == 1
    assert result["bearish_count"] == 1
    assert result["source"] == "polygon"


def test_trend_label_improving():
    assert polygon._trend_label([1.5, 1.2]) == "improving"


def test_trend_label_declining():
    assert polygon._trend_label([1.0, 1.5]) == "declining"


def test_trend_label_stable():
    assert polygon._trend_label([1.0, 1.0]) == "stable"


def test_trend_label_insufficient_data():
    assert polygon._trend_label([1.0]) == "insufficient_data"
    assert polygon._trend_label([]) == "insufficient_data"
