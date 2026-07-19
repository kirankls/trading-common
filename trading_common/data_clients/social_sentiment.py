"""ApeWisdom Reddit/WSB mention sentiment — no API key required."""
from __future__ import annotations

import time as _time
from typing import Any

import httpx

from trading_common.data_clients._cache import get as _cache_get
from trading_common.data_clients._cache import set as _cache_set
from trading_common.monitoring.tracker import get_tracker


async def fetch_wsb_mentions(ticker: str) -> dict[str, Any]:
    """Return WallStreetBets mention count and rank change for a ticker.

    Uses the free apewisdom.io API (no key needed).
    Returns empty dict on failure.
    """
    cache_key = f"social:wsb:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    _t0 = _time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(
                "https://apewisdom.io/api/v1.0/filter/wallstreetbets",
                headers={"User-Agent": "options-advisor/1.0"},
            )
            if r.status_code != 200:
                get_tracker("wsb").record(False, (_time.perf_counter() - _t0) * 1000,
                                          error=f"HTTP {r.status_code}", status_code=r.status_code)
                return {}
            results = r.json().get("results", [])
            for item in results:
                if (item.get("ticker") or "").upper() == ticker.upper():
                    mentions = item.get("mentions", 0)
                    mentions_24h_ago = item.get("mentions_24h_ago") or 1
                    velocity = mentions / max(mentions_24h_ago, 1)
                    rank = item.get("rank")
                    rank_24h_ago = item.get("rank_24h_ago")
                    # rank_change = rank_24h_ago - current_rank
                    # Positive value = stock is rising in popularity (rank number got smaller = higher up the list)
                    # e.g. was rank 50 yesterday, now rank 20 → rank_change = +30 (rising)
                    rank_change = None
                    if rank is not None and rank_24h_ago is not None:
                        rank_change = rank_24h_ago - rank
                    result = {
                        "mentions": mentions,
                        "rank": rank,
                        "rank_24h_ago": rank_24h_ago,
                        "rank_change": rank_change,
                        "mention_velocity": round(velocity, 2),
                        "squeeze_watch": velocity > 5.0,
                    }
                    _cache_set(cache_key, result, 900)
                    get_tracker("wsb").record(True, (_time.perf_counter() - _t0) * 1000)
                    return result
            # Ticker not found in WSB rankings — cache the empty result too
            not_found: dict[str, Any] = {"mentions": 0, "rank": None, "rank_24h_ago": None, "rank_change": None, "mention_velocity": 0.0, "squeeze_watch": False}
            _cache_set(cache_key, not_found, 900)
            get_tracker("wsb").record(True, (_time.perf_counter() - _t0) * 1000)
            return not_found
    except Exception as _exc:
        get_tracker("wsb").record(False, (_time.perf_counter() - _t0) * 1000, error=str(_exc))
        return {}
