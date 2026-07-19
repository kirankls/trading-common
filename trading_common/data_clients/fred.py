"""FRED (Federal Reserve Economic Data) client — free REST API.

Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html.
Set the FRED_API_KEY environment variable. If the key is absent the helpers
return empty dicts gracefully rather than raising.
"""
from __future__ import annotations

import asyncio
import logging
import time as _time
from datetime import UTC, datetime
from typing import Any

import httpx

from trading_common.data_clients._cache import get as _cache_get
from trading_common.data_clients._cache import set as _cache_set
from trading_common.monitoring.tracker import get_tracker

_log = logging.getLogger(__name__)


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
    return r  # type: ignore[return-value]  # last response (may be an error status; unreachable with max_retries>=0)

_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _api_key() -> str:
    from trading_common.config.settings import settings
    return settings.fred_api_key.get_secret_value()


def _resolve_api_key(api_key: str | None) -> str:
    """Return user-supplied key if provided, otherwise fall back to platform settings."""
    return api_key if api_key else _api_key()


async def _fetch_series(
    series_id: str, limit: int = 5, api_key: str | None = None
) -> list[dict[str, Any]]:
    """Fetch the most recent *limit* observations for a FRED series.

    Returns an empty list when the key is absent, the request fails, or the
    response cannot be parsed.
    """
    key = _resolve_api_key(api_key)
    if not key:
        return []
    _t0 = _time.perf_counter()
    try:
        params: dict[str, Any] = {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": limit,
            "file_type": "json",
            "api_key": key,
        }
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await _async_get_with_retry(client, _BASE, params=params)
            if r.status_code != 200:
                get_tracker("fred").record(False, (_time.perf_counter() - _t0) * 1000,
                                           error=f"HTTP {r.status_code} for {series_id}",
                                           status_code=r.status_code)
                return []
            obs = r.json().get("observations", [])
            # Filter out missing-value sentinels FRED uses ("." or blank)
            result = [o for o in obs if o.get("value") not in (".", None, "")]
            get_tracker("fred").record(True, (_time.perf_counter() - _t0) * 1000)
            return result
    except Exception as _exc:
        get_tracker("fred").record(False, (_time.perf_counter() - _t0) * 1000, error=str(_exc))
        return []


async def fetch_hy_credit_spread(api_key: str | None = None) -> dict[str, Any]:
    """Return the current High Yield OAS credit spread (BAMLH0A0HYM2).

    HY spread > 500 bp = stressed market (risk-off signal for options sellers).
    HY spread < 300 bp = tight / benign credit conditions.

    Returns an empty dict when the API key is absent or the request fails.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = "fred:BAMLH0A0HYM2"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        obs = await _fetch_series("BAMLH0A0HYM2", limit=10, api_key=api_key)
        if not obs:
            return {}
        latest = obs[0]
        # Series is in percentage points; multiply by 100 to get basis points
        spread_bp = float(latest["value"]) * 100

        if spread_bp < 300:
            regime = "tight"
        elif spread_bp < 500:
            regime = "normal"
        else:
            regime = "stressed"

        # 5-observation (≈ 20-business-day) change
        change_bp: float | None = None
        if len(obs) >= 5:
            try:
                older = float(obs[4]["value"]) * 100
                change_bp = round(spread_bp - older, 1)
            except Exception:
                pass

        result = {
            "hy_spread_bp": round(spread_bp, 1),
            "hy_spread_regime": regime,
            "hy_spread_change_bp": change_bp,
            "as_of": latest.get("date"),
        }
        _cache_set(cache_key, result, 86400)
        return result
    except Exception:
        return {}


async def fetch_breakeven_inflation(api_key: str | None = None) -> dict[str, Any]:
    """Return the 10-year breakeven inflation rate (T10YIE).

    This is the market-implied CPI expectation derived from TIPS spreads.
    Returns an empty dict when the API key is absent or the request fails.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = "fred:T10YIE"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        obs = await _fetch_series("T10YIE", limit=3, api_key=api_key)
        if not obs:
            return {}
        latest = obs[0]
        result = {
            "breakeven_inflation_10y": round(float(latest["value"]), 2),
            "as_of": latest.get("date"),
        }
        _cache_set(cache_key, result, 86400)
        return result
    except Exception:
        return {}


async def fetch_fed_funds_rate(api_key: str | None = None) -> dict[str, Any]:
    """Fetch the effective federal funds rate (FEDFUNDS series).

    Returns dict with key ``fed_funds_rate_pct`` or empty dict on failure.
    Results are cached for 24 hours since the FOMC only meets ~8 times per year.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = "fred:FEDFUNDS"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        obs = await _fetch_series("FEDFUNDS", limit=1, api_key=api_key)
        if not obs:
            return {}
        latest = obs[0]
        rate = float(latest["value"])
        result: dict[str, Any] = {
            "fed_funds_rate_pct": round(rate, 2),
            "as_of": latest.get("date"),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _cache_set(cache_key, result, 86400)
        return result
    except Exception:
        return {}


async def fetch_cpi(api_key: str | None = None) -> dict[str, Any]:
    """Fetch CPI YoY change from CPIAUCSL series (13 months for year-over-year calc).

    Returns dict with key ``cpi_yoy_pct`` or empty dict on failure.
    Results are cached for 24 hours since CPI is released monthly.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    cache_key = "fred:CPIAUCSL"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        obs = await _fetch_series("CPIAUCSL", limit=13, api_key=api_key)
        if not obs:
            return {}
        latest = obs[0]
        yoy: float | None = None
        if len(obs) >= 13:
            try:
                current_val = float(obs[0]["value"])
                year_ago_val = float(obs[12]["value"])
                if year_ago_val != 0:
                    yoy = round((current_val - year_ago_val) / year_ago_val * 100, 2)
            except Exception:
                yoy = None
        result: dict[str, Any] = {
            "cpi_yoy_pct": yoy,
            "as_of": latest.get("date"),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _cache_set(cache_key, result, 86400)
        return result
    except Exception:
        return {}


async def get_risk_free_rate(tenor_days: int, api_key: str | None = None) -> float:
    """Return the risk-free rate (decimal, e.g. 0.045 for 4.5%) for Black-Scholes pricing.

    Uses FRED series ``DGS3MO`` — the 3-month Treasury Constant Maturity Rate.
    Constant-maturity (as opposed to ``DTB3``'s discount-basis quoting) is the
    standard choice for Black-Scholes: it is already a bond-equivalent yield,
    which is the convention `r` assumes, whereas a discount-basis T-bill rate
    would need a day-count conversion before it's usable as-is.

    ``tenor_days`` is accepted for forward-compatibility with the pricing
    layer (Phase 2) but is not yet used to select/interpolate a curve — DTE
    horizons in this codebase rarely exceed a year, so a single representative
    short-tenor (3-month) rate is a reasonable proxy across that whole range.
    A future revision could blend with ``DGS1`` (1-year CMT) for tenor_days
    beyond ~180, but that has been intentionally deferred to keep this simple
    until the pricing core (Phase 2) actually needs the extra precision.

    Unlike the other ``fetch_*`` helpers in this module (which return ``{}``
    on failure), this function ALWAYS returns a usable float and NEVER raises:
    on a missing API key, a request failure, or a parse failure it falls back
    to ``0.05`` and logs a warning — callers (pricing code) must never have to
    handle a missing risk-free rate.

    Cached for 24 hours (3-month T-bill yields don't move meaningfully
    intraday). Pass ``api_key`` to use a per-user key instead of the platform
    default.
    """
    cache_key = "fred:risk_free_rate"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        obs = await _fetch_series("DGS3MO", limit=5, api_key=api_key)
        if not obs:
            _log.warning(
                "get_risk_free_rate: no observations returned for DGS3MO "
                "(missing API key or request failure) — falling back to 0.05"
            )
            return 0.05
        latest = obs[0]
        # DGS3MO is quoted in percentage points (e.g. 4.53 == 4.53%); convert
        # to a decimal fraction to match how risk_free_rate is consumed today
        # (strike_select.py's default is the decimal 0.05, not 5.0).
        rate = round(float(latest["value"]) / 100.0, 5)
        _cache_set(cache_key, rate, 86400)
        return rate
    except Exception as exc:
        _log.warning("get_risk_free_rate: failed to fetch/parse DGS3MO (%s) — falling back to 0.05", exc)
        return 0.05


async def fetch_macro_fred(api_key: str | None = None) -> dict[str, Any]:
    """Fetch all FRED macro indicators concurrently and return a combined dict.

    Keys present depend on which requests succeed. Always returns a dict (never
    raises). Empty dict means either the API key is absent or all requests failed.

    Pass ``api_key`` to use a per-user key instead of the platform default.
    """
    _results = await asyncio.gather(
        fetch_hy_credit_spread(api_key=api_key),
        fetch_breakeven_inflation(api_key=api_key),
        fetch_fed_funds_rate(api_key=api_key),
        fetch_cpi(api_key=api_key),
        return_exceptions=True,
    )
    hy, inflation, fed_funds, cpi = [
        r if not isinstance(r, BaseException) else {}
        for r in _results
    ]
    result: dict[str, Any] = {}
    result.update(hy)
    result.update(inflation)
    result.update(fed_funds)
    result.update(cpi)
    return result
