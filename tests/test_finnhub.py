"""Unit tests for trading_common.data_clients.finnhub.

Mocks the httpx layer with respx and asserts parsed results on 200 responses
and graceful empty-result fallback on error responses, matching the original
chanakya client's behaviour (it never raises on expected failure modes — it
returns {} / [] and records the failure via the SourceTracker).
"""
from __future__ import annotations

import httpx
import pytest
import respx

from trading_common.data_clients import _cache, finnhub


@pytest.fixture(autouse=True)
def _clear_cache_and_settings(monkeypatch):
    _cache.clear()
    # Provide a deterministic API key so _resolve_token() doesn't fall through
    # to real settings/env state.
    monkeypatch.setattr(finnhub, "_token", lambda: "test-finnhub-token")
    yield
    _cache.clear()


@respx.mock
async def test_fetch_news_sentiment_success():
    route = respx.get(f"{finnhub._BASE}/company-news").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"headline": "Company beats earnings, shares surges", "source": "Reuters", "url": "u1", "datetime": 1},
                {"headline": "Analysts warn of weak guidance", "source": "Bloomberg", "url": "u2", "datetime": 2},
            ],
        )
    )
    result = await finnhub.fetch_news_sentiment("AAPL")
    assert route.called
    assert result["bullish_count"] == 1
    assert result["bearish_count"] == 1
    assert result["source"] == "finnhub"
    assert len(result["recent_headlines"]) == 2


@respx.mock
async def test_fetch_news_sentiment_empty_articles():
    respx.get(f"{finnhub._BASE}/company-news").mock(return_value=httpx.Response(200, json=[]))
    result = await finnhub.fetch_news_sentiment("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_news_sentiment_auth_failure_returns_empty_dict():
    # Original code does not branch on status code beyond "!= 200" -> {} for all
    # non-200 responses (401 included). No custom FetchErrorType mapping exists
    # in the source; it just returns {} and logs via the tracker.
    respx.get(f"{finnhub._BASE}/company-news").mock(return_value=httpx.Response(401, json={"error": "bad token"}))
    result = await finnhub.fetch_news_sentiment("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_news_sentiment_rate_limited_returns_empty_dict():
    # 429 triggers the retry loop inside _async_get_with_retry; after retries
    # are exhausted the final (still 429) response is returned and the caller
    # sees status_code != 200 -> {}.
    respx.get(f"{finnhub._BASE}/company-news").mock(return_value=httpx.Response(429))
    result = await finnhub.fetch_news_sentiment("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_news_sentiment_timeout_returns_empty_dict():
    respx.get(f"{finnhub._BASE}/company-news").mock(side_effect=httpx.TimeoutException("timed out"))
    result = await finnhub.fetch_news_sentiment("AAPL")
    assert result == {}


async def test_fetch_news_sentiment_no_token_returns_empty_dict(monkeypatch):
    monkeypatch.setattr(finnhub, "_token", lambda: None)
    result = await finnhub.fetch_news_sentiment("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_analyst_rating_success():
    respx.get(f"{finnhub._BASE}/stock/recommendation").mock(
        return_value=httpx.Response(
            200,
            json=[{"period": "2026-06-01", "strongBuy": 10, "buy": 5, "hold": 2, "sell": 1, "strongSell": 0}],
        )
    )
    result = await finnhub.fetch_analyst_rating("AAPL")
    assert result["signal"] == "buy"
    assert result["strong_buy"] == 10


@respx.mock
async def test_fetch_analyst_rating_bad_status_returns_empty_dict():
    respx.get(f"{finnhub._BASE}/stock/recommendation").mock(return_value=httpx.Response(500))
    result = await finnhub.fetch_analyst_rating("AAPL")
    assert result == {}


@respx.mock
def test_fetch_earnings_history_success():
    respx.get(f"{finnhub._BASE}/stock/earnings").mock(
        return_value=httpx.Response(
            200,
            json=[{"period": "2026-Q1", "actual": 1.10, "estimate": 1.00}],
        )
    )
    result = finnhub.fetch_earnings_history("AAPL")
    assert result == [{"date": "2026-Q1", "actual": 1.10, "estimate": 1.00, "surprise_pct": 10.0}]


@respx.mock
def test_fetch_earnings_history_non_200_returns_empty_list():
    respx.get(f"{finnhub._BASE}/stock/earnings").mock(return_value=httpx.Response(403))
    result = finnhub.fetch_earnings_history("AAPL")
    assert result == []


@respx.mock
async def test_fetch_company_profile_success():
    route = respx.get(f"{finnhub._BASE}/stock/profile2").mock(
        return_value=httpx.Response(
            200,
            json={
                "symbol": "AAPL",
                "shareOutstanding": 15334.1,
                "marketCapitalization": 2900000.0,
                "name": "Apple Inc",
            },
        )
    )
    result = await finnhub.fetch_company_profile("AAPL")
    assert route.called
    assert result["shares_outstanding_millions"] == 15334.1
    assert result["market_cap_millions"] == 2900000.0


@respx.mock
async def test_fetch_company_profile_bad_status_returns_empty_dict():
    respx.get(f"{finnhub._BASE}/stock/profile2").mock(return_value=httpx.Response(500))
    result = await finnhub.fetch_company_profile("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_company_profile_empty_body_returns_empty_dict():
    # Finnhub returns {} (200 OK) for tickers it doesn't cover.
    respx.get(f"{finnhub._BASE}/stock/profile2").mock(return_value=httpx.Response(200, json={}))
    result = await finnhub.fetch_company_profile("UNKNOWNTICKER")
    assert result == {}


async def test_fetch_company_profile_no_token_returns_empty_dict(monkeypatch):
    monkeypatch.setattr(finnhub, "_token", lambda: None)
    result = await finnhub.fetch_company_profile("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_company_profile_timeout_returns_empty_dict():
    respx.get(f"{finnhub._BASE}/stock/profile2").mock(side_effect=httpx.TimeoutException("timed out"))
    result = await finnhub.fetch_company_profile("AAPL")
    assert result == {}


@respx.mock
async def test_fetch_company_profile_is_cached_within_ttl():
    route = respx.get(f"{finnhub._BASE}/stock/profile2").mock(
        return_value=httpx.Response(200, json={"shareOutstanding": 100.0, "marketCapitalization": 5000.0})
    )
    result1 = await finnhub.fetch_company_profile("AAPL")
    result2 = await finnhub.fetch_company_profile("AAPL")
    assert route.call_count == 1
    assert result1 == result2 == {"shares_outstanding_millions": 100.0, "market_cap_millions": 5000.0}


@respx.mock
async def test_fetch_economic_calendar_filters_high_medium_us_events():
    respx.get(f"{finnhub._BASE}/calendar/economic").mock(
        return_value=httpx.Response(
            200,
            json={
                "economicCalendar": [
                    {"event": "FOMC Rate Decision", "time": "2026-07-15 14:00:00", "impact": "high", "country": "US"},
                    {"event": "Low impact event", "time": "2026-07-16 08:00:00", "impact": "low", "country": "US"},
                    {"event": "Foreign event", "time": "2026-07-17 08:00:00", "impact": "high", "country": "DE"},
                ]
            },
        )
    )
    result = await finnhub.fetch_economic_calendar()
    assert len(result) == 1
    assert result[0]["event"] == "FOMC Rate Decision"
    assert result[0]["date"] == "2026-07-15"
