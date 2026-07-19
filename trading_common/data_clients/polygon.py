# Ported from D:\chanakya\options_advisor\data_clients\polygon.py
"""Polygon.io data client — news, earnings/financials, and stock aggregates."""
from __future__ import annotations

import asyncio
import time as _time
from typing import Any

import httpx

from trading_common.config.settings import settings
from trading_common.data_clients._cache import get as _cache_get
from trading_common.data_clients._cache import set as _cache_set
from trading_common.monitoring.tracker import get_tracker

_BASE = "https://api.polygon.io"
_TIMEOUT = 10.0


def _key() -> str:
    return settings.polygon_api_key.get_secret_value()


def _resolve_key(api_key: str | None) -> str:
    """Return user-supplied key if provided, otherwise fall back to platform settings."""
    return api_key if api_key else _key()


def _clean(s: str, maxlen: int = 200) -> str:
    """Sanitise a headline/description string before it reaches the synthesis prompt."""
    return s.replace("\n", " ").replace("\r", " ").strip()[:maxlen]


async def fetch_ticker_news(
    ticker: str, limit: int = 10, api_key: str | None = None
) -> list[dict[str, Any]]:
    """Fetch recent news articles for ticker from Polygon.

    Returns list of dicts with: title, published_utc, article_url, sentiment (if available), description, tickers.
    Returns [] on any error or if key not configured.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = f"polygon:news:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    key = _resolve_key(api_key)
    if not key:
        return []
    _t0 = _time.perf_counter()
    try:
        params = {
            "ticker": ticker,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc",
            "apiKey": key,
        }
        _max_retries = 3
        _delay = 1.0
        for _attempt in range(_max_retries):
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(f"{_BASE}/v2/reference/news", params=params)
            if r.status_code == 429 or r.status_code >= 500:
                if _attempt < _max_retries - 1:
                    await asyncio.sleep(_delay)
                    _delay *= 2
                    continue
            r.raise_for_status()
            break
        data = r.json()
        articles = data.get("results", [])
        result = [
            {
                "title": _clean(a.get("title", "")),
                "published_utc": a.get("published_utc", ""),
                "url": a.get("article_url", ""),
                "source": a.get("publisher", {}).get("name", ""),
                "description": _clean((a.get("description") or "")[:300]),
                "tickers": a.get("tickers", []),
                "sentiment": a.get("insights", [{}])[0].get("sentiment") if a.get("insights") else None,
                "sentiment_reasoning": a.get("insights", [{}])[0].get("sentiment_reasoning", "") if a.get("insights") else "",
            }
            for a in articles
        ]
        _cache_set(cache_key, result, 900)
        get_tracker("polygon").record(True, (_time.perf_counter() - _t0) * 1000)
        return result
    except Exception as _exc:
        get_tracker("polygon").record(False, (_time.perf_counter() - _t0) * 1000, error=str(_exc))
        return []


async def fetch_earnings(ticker: str, api_key: str | None = None) -> dict[str, Any]:
    """Fetch latest earnings data from Polygon financials endpoint.

    Returns dict with: eps_actual, eps_estimate, eps_surprise_pct, revenue_actual,
    revenue_estimate, period_of_report, fiscal_year, quarter, surprise_direction.
    Returns {} on any error.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = f"polygon:earnings:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    key = _resolve_key(api_key)
    if not key:
        return {}
    try:
        params = {
            "ticker": ticker,
            "limit": 4,  # Last 4 quarters
            "order": "desc",
            "sort": "filing_date",
            "apiKey": key,
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(f"{_BASE}/vX/reference/financials", params=params)
            r.raise_for_status()
            data = r.json()
        results = data.get("results", [])
        if not results:
            return {}

        latest = results[0]
        financials = latest.get("financials", {})
        income = financials.get("income_statement", {})

        # Extract EPS and revenue
        eps = income.get("basic_earnings_per_share", {})
        revenue = income.get("revenues", {})

        eps_actual = eps.get("value")
        revenue_actual = revenue.get("value")

        # Build quarter-over-quarter list for trend
        quarters = []
        for q in results[:4]:
            q_income = q.get("financials", {}).get("income_statement", {})
            q_eps = q_income.get("basic_earnings_per_share", {}).get("value")
            q_rev = q_income.get("revenues", {}).get("value")
            quarters.append({
                "period": q.get("fiscal_period", ""),
                "year": q.get("fiscal_year", ""),
                "eps": q_eps,
                "revenue": q_rev,
            })

        result = {
            "eps_actual": eps_actual,
            "revenue_actual": revenue_actual,
            "period_of_report": latest.get("period_of_report_date", ""),
            "fiscal_period": latest.get("fiscal_period", ""),
            "fiscal_year": latest.get("fiscal_year", ""),
            "recent_quarters": quarters,
            "eps_trend": _trend_label([q["eps"] for q in quarters if q["eps"] is not None]),
            "revenue_trend": _trend_label([q["revenue"] for q in quarters if q["revenue"] is not None]),
        }
        _cache_set(cache_key, result, 3600)
        return result
    except Exception:
        return {}


async def fetch_rsi(
    ticker: str, window: int = 14, api_key: str | None = None
) -> float | None:
    """Fetch latest RSI from Polygon's built-in indicator endpoint.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = f"polygon:rsi:{ticker.upper()}:{window}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    key = _resolve_key(api_key)
    if not key:
        return None
    try:
        params = {
            "timespan": "day",
            "adjusted": "true",
            "window": window,
            "series_type": "close",
            "limit": 1,
            "apiKey": key,
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(f"{_BASE}/v1/indicators/rsi/{ticker}", params=params)
            r.raise_for_status()
            data = r.json()
        results = data.get("results", {}).get("values", [])
        if results:
            result = round(float(results[0]["value"]), 2)
            _cache_set(cache_key, result, 300)
            return result
        return None
    except Exception:
        return None


async def fetch_polygon_news_sentiment(
    ticker: str, api_key: str | None = None
) -> dict[str, Any]:
    """Aggregate news sentiment from Polygon articles.

    Returns: {bullish_count, bearish_count, neutral_count, sentiment_score (-1 to 1),
               sentiment_label, recent_headlines (list of 5)}

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    articles = await fetch_ticker_news(ticker, limit=20, api_key=api_key)
    if not articles:
        return {}

    # Polygon returns "positive"/"negative"/"neutral" — normalise to bullish/bearish/neutral
    sentiment_map_ = {"positive": "bullish", "negative": "bearish", "bullish": "bullish", "bearish": "bearish"}
    counts: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
    headlines: list[dict[str, Any]] = []
    for a in articles:
        raw = (a.get("sentiment") or "neutral").lower()
        s = sentiment_map_.get(raw, "neutral")
        counts[s] += 1
        if len(headlines) < 5:
            headlines.append({"title": a["title"], "source": a["source"], "sentiment": s, "url": a["url"]})

    total = sum(counts.values()) or 1
    score = (counts["bullish"] - counts["bearish"]) / total

    if score > 0.2:
        label = "bullish"
    elif score < -0.2:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "bullish_count": counts["bullish"],
        "bearish_count": counts["bearish"],
        "neutral_count": counts["neutral"],
        "sentiment_score": round(score, 3),
        "sentiment_label": label,
        "recent_headlines": headlines,
        "source": "polygon",
    }


def _trend_label(values: list) -> str:
    """Return 'improving', 'declining', or 'stable' based on last 3 values."""
    if len(values) < 2:
        return "insufficient_data"
    recent = values[:2]  # Most recent first
    if recent[0] > recent[1] * 1.05:
        return "improving"
    elif recent[0] < recent[1] * 0.95:
        return "declining"
    return "stable"
