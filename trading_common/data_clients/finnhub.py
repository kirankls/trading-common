# Ported from D:\chanakya\options_advisor\data_clients\finnhub.py
"""Finnhub free-tier API client — news sentiment, analyst ratings, economic calendar, company profile (shares outstanding/market cap)."""
from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Any

import httpx

from trading_common.data_clients._cache import get as _cache_get
from trading_common.data_clients._cache import set as _cache_set
from trading_common.monitoring.tracker import get_tracker

_BASE = "https://finnhub.io/api/v1"
_log = logging.getLogger(__name__)


def _http_get_with_retry(
    url: str,
    *,
    params=None,
    headers=None,
    timeout: float = 15,
    max_retries: int = 3,
) -> httpx.Response:
    """GET with exponential backoff on 429 and 5xx. Raises on final failure."""
    delay = 1.0
    r: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < max_retries:
                    _log.warning(
                        "_http_get_with_retry: status %s, retrying in %.1fs (attempt %d/%d)",
                        r.status_code, delay, attempt + 1, max_retries,
                    )
                    _time.sleep(delay)
                    delay *= 2
                    continue
            return r
        except httpx.TimeoutException:
            if attempt < max_retries:
                _log.warning(
                    "_http_get_with_retry: timeout, retrying in %.1fs (attempt %d/%d)",
                    delay, attempt + 1, max_retries,
                )
                _time.sleep(delay)
                delay *= 2
                continue
            raise
    # r is always assigned (loop runs >= 1 time for max_retries >= 0); mypy
    # can't prove that statically since r starts as Response | None.
    return r  # type: ignore[return-value]


async def _async_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params=None,
    max_retries: int = 3,
) -> httpx.Response:
    """Async GET with exponential backoff on 429 and 5xx. Raises on final failure."""
    delay = 1.0
    r: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            r = await client.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < max_retries:
                    _log.warning(
                        "_async_get_with_retry: status %s, retrying in %.1fs (attempt %d/%d)",
                        r.status_code, delay, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
            return r
        except httpx.TimeoutException:
            if attempt < max_retries:
                _log.warning(
                    "_async_get_with_retry: timeout, retrying in %.1fs (attempt %d/%d)",
                    delay, attempt + 1, max_retries,
                )
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise
    # r is always assigned (loop runs >= 1 time for max_retries >= 0); mypy
    # can't prove that statically since r starts as Response | None.
    return r  # type: ignore[return-value]


def _token() -> str | None:
    from trading_common.config.settings import settings
    val = settings.finnhub_api_key.get_secret_value()
    return val if val else None


def _resolve_token(api_key: str | None) -> str | None:
    """Return user-supplied key if provided, otherwise fall back to platform settings."""
    return api_key if api_key else _token()


def _clean_headline(s: str, maxlen: int = 200) -> str:
    """Sanitise a headline string before it reaches the synthesis prompt."""
    return s.replace("\n", " ").replace("\r", " ").strip()[:maxlen]


async def fetch_news_sentiment(
    ticker: str, api_key: str | None = None
) -> dict[str, Any]:
    """Return recent news headlines and derived sentiment for a ticker.

    Uses /company-news (free tier). Derives bullish/bearish/neutral label by
    counting keyword signals in headlines since /news-sentiment requires a paid plan.
    Returns empty dict if API key is not set or request fails.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = f"finnhub:news:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    token = _resolve_token(api_key)
    if not token:
        return {}
    _t0 = _time.perf_counter()
    try:
        from datetime import date, timedelta
        today = date.today().isoformat()
        week_ago = (date.today() - timedelta(days=7)).isoformat()
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await _async_get_with_retry(
                client,
                f"{_BASE}/company-news",
                params={"symbol": ticker.upper(), "from": week_ago, "to": today, "token": token},
            )
            if r.status_code != 200:
                get_tracker("finnhub").record(False, (_time.perf_counter() - _t0) * 1000,
                                              error=f"HTTP {r.status_code}", status_code=r.status_code)
                return {}
            articles = r.json()[:20]  # cap at 20 most recent
        if not articles:
            get_tracker("finnhub").record(True, (_time.perf_counter() - _t0) * 1000)
            return {}

        bullish_words = {"beat", "surges", "gains", "upgrade", "buy", "outperform", "record", "rally", "rises", "strong"}
        bearish_words = {"miss", "falls", "drops", "downgrade", "sell", "underperform", "cut", "warning", "weak", "loss"}

        bullish = bearish = 0
        headlines = []
        for a in articles[:15]:
            text = (a.get("headline") or a.get("summary") or "").lower()
            words = set(text.split())
            if words & bullish_words:
                bullish += 1
                sentiment = "bullish"
            elif words & bearish_words:
                bearish += 1
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            if len(headlines) < 5:
                headlines.append({
                    "title": _clean_headline(a.get("headline", "")),
                    "source": a.get("source", ""),
                    "sentiment": sentiment,
                    "url": a.get("url", ""),
                    "datetime": a.get("datetime", ""),
                })

        total = len(articles[:15]) or 1
        score = (bullish - bearish) / total
        label = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"

        result = {
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": total - bullish - bearish,
            "sentiment_score": round(score, 3),
            "sentiment_label": label,
            "recent_headlines": headlines,
            "source": "finnhub",
        }
        _cache_set(cache_key, result, 3600)
        get_tracker("finnhub").record(True, (_time.perf_counter() - _t0) * 1000)
        return result
    except Exception as _exc:
        get_tracker("finnhub").record(False, (_time.perf_counter() - _t0) * 1000, error=str(_exc))
        return {}


async def fetch_analyst_rating(
    ticker: str, api_key: str | None = None
) -> dict[str, Any]:
    """Return latest analyst consensus and most recent rating change.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = f"finnhub:analyst:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    token = _resolve_token(api_key)
    if not token:
        return {}
    _t0 = _time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await _async_get_with_retry(
                client,
                f"{_BASE}/stock/recommendation",
                params={"symbol": ticker.upper(), "token": token},
            )
            if r.status_code != 200 or not r.json():
                get_tracker("finnhub").record(False, (_time.perf_counter() - _t0) * 1000,
                                              error=f"HTTP {r.status_code}", status_code=r.status_code)
                return {}
            latest = r.json()[0]
            strong_buy = latest.get("strongBuy") or 0
            buy = latest.get("buy") or 0
            sell = latest.get("sell") or 0
            strong_sell = latest.get("strongSell") or 0
            bullish_count = strong_buy + buy
            bearish_count = strong_sell + sell
            if bullish_count > bearish_count * 1.5:
                signal = "buy"
            elif bearish_count > bullish_count * 1.5:
                signal = "sell"
            else:
                signal = "hold"
            result = {
                "period": latest.get("period"),
                "strong_buy": latest.get("strongBuy"),
                "buy": latest.get("buy"),
                "hold": latest.get("hold"),
                "sell": latest.get("sell"),
                "strong_sell": latest.get("strongSell"),
                "signal": signal,
            }
            _cache_set(cache_key, result, 21600)
            get_tracker("finnhub").record(True, (_time.perf_counter() - _t0) * 1000)
            return result
    except Exception as _exc:
        get_tracker("finnhub").record(False, (_time.perf_counter() - _t0) * 1000, error=str(_exc))
        return {}


async def fetch_company_profile(
    ticker: str, api_key: str | None = None
) -> dict[str, Any]:
    """Return company profile fundamentals, notably shares outstanding.

    Uses /stock/profile2 (Finnhub's documented "free version of the
    Company Profile endpoint"). Returns:
        {"shares_outstanding_millions": float | None,
         "market_cap_millions": float | None}

    ``shares_outstanding_millions`` comes straight from Finnhub's
    ``shareOutstanding`` field (already expressed in millions of shares by
    Finnhub's own convention). ``market_cap_millions`` comes from
    ``marketCapitalization`` (also in millions) -- captured because it's
    free in the same response and may be independently useful, even though
    the shares-outstanding figure is what callers of this function actually
    need it for (see `engine.premarket_scanner`'s S4 float-proxy filter).

    Returns {} if the API key is not set or the request fails -- same
    never-raise contract as `fetch_analyst_rating`/`fetch_news_sentiment`.

    Cache TTL is 24h (much longer than `fetch_analyst_rating`'s 6h):
    shares outstanding / market cap are slow-moving fundamentals that only
    change on corporate actions (buybacks, secondary offerings, splits) --
    unlike a live quote or even an analyst rating, there's no meaningful
    intraday signal lost by caching this for a full day.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = f"finnhub:profile:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    token = _resolve_token(api_key)
    if not token:
        return {}
    _t0 = _time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await _async_get_with_retry(
                client,
                f"{_BASE}/stock/profile2",
                params={"symbol": ticker.upper(), "token": token},
            )
            if r.status_code != 200 or not r.json():
                get_tracker("finnhub").record(False, (_time.perf_counter() - _t0) * 1000,
                                              error=f"HTTP {r.status_code}", status_code=r.status_code)
                return {}
            data = r.json()
            result = {
                "shares_outstanding_millions": data.get("shareOutstanding"),
                "market_cap_millions": data.get("marketCapitalization"),
            }
            _cache_set(cache_key, result, 86400)
            get_tracker("finnhub").record(True, (_time.perf_counter() - _t0) * 1000)
            return result
    except Exception as _exc:
        get_tracker("finnhub").record(False, (_time.perf_counter() - _t0) * 1000, error=str(_exc))
        return {}


def fetch_earnings_history(ticker: str, api_key: str | None = None) -> list[dict[str, Any]]:
    """Return last 8 quarters of earnings surprise data for a ticker.

    Calls Finnhub GET /stock/earnings?symbol={ticker}&limit=8.
    Returns a list of dicts:
        {"date": str, "actual": float | None, "estimate": float | None, "surprise_pct": float | None}
    Returns [] if the API key is not set or the request fails.

    This is a synchronous function intended to be called via run_in_executor.
    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = f"finnhub:earnings_hist:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    token = _resolve_token(api_key)
    if not token:
        return []
    try:
        r = _http_get_with_retry(
            f"{_BASE}/stock/earnings",
            params={"symbol": ticker.upper(), "limit": 8, "token": token},
            timeout=8.0,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []
        result: list[dict[str, Any]] = []
        for item in data:
            actual = item.get("actual")
            estimate = item.get("estimate")
            surprise_pct: float | None = None
            if actual is not None and estimate is not None and estimate != 0:
                surprise_pct = round((actual - estimate) / abs(estimate) * 100, 2)
            result.append({
                "date": item.get("period", ""),
                "actual": actual,
                "estimate": estimate,
                "surprise_pct": surprise_pct,
            })
        _cache_set(cache_key, result, 3600)
        return result
    except Exception:
        return []


async def fetch_economic_calendar(api_key: str | None = None) -> list[dict[str, Any]]:
    """Return upcoming high-impact economic events (FOMC, CPI, NFP, etc.).

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = "finnhub:calendar"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    token = _resolve_token(api_key)
    if not token:
        return []
    try:
        from datetime import date, timedelta
        today = date.today().isoformat()
        end = (date.today() + timedelta(days=30)).isoformat()
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await _async_get_with_retry(
                client,
                f"{_BASE}/calendar/economic",
                params={"from": today, "to": end, "token": token},
            )
            if r.status_code != 200:
                return []
            events = r.json().get("economicCalendar", [])
            # Filter to high-impact US events only
            result = [
                {"event": e.get("event"), "date": e.get("time", "")[:10], "impact": e.get("impact")}
                for e in events
                if e.get("country") == "US" and e.get("impact") in ("high", "medium")
            ][:10]
            _cache_set(cache_key, result, 1800)
            return result
    except Exception:
        return []
