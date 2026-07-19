"""Short interest data via yfinance (free, no API key required).

yfinance exposes ``shortPercentOfFloat``, ``shortRatio``, and ``sharesShort``
from Yahoo Finance's info endpoint. Data is updated approximately weekly by
Yahoo and originates from FINRA regulatory short interest filings.

Note: FINRA also publishes bi-monthly short interest CSV files directly at
https://cdn.finra.org/equity/regsho/biweekly/CNMSshvol{DATE}.txt but parsing
those files requires knowing the exact release dates. The yfinance approach is
simpler and sufficient for advisory use.
"""
from __future__ import annotations

from typing import Any


async def fetch_short_interest(ticker: str) -> dict[str, Any]:
    """Return short interest metrics for *ticker*.

    Uses the yfinance ``Ticker.info`` dict. The call is blocking internally;
    it is run in the default asyncio executor so it does not block the event
    loop.

    Returns an empty dict when:
    - The ticker maps to an index symbol (e.g. ^GSPC) — indices have no float.
    - yfinance returns no ``info`` data.
    - Any exception occurs.

    Fields returned (all optional — only present when Yahoo reports them):
    - ``short_float_pct``: short interest as percentage of float (e.g. 5.2 = 5.2 %)
    - ``high_short_interest``: True when short_float_pct > 20 %
    - ``days_to_cover``: shares short / average daily volume (short ratio)
    - ``squeeze_risk``: True when days_to_cover > 5
    - ``shares_short``: absolute number of shares sold short
    """
    try:
        import asyncio

        import yfinance as yf

        from trading_common.data_clients.base import map_price_ticker

        # Index symbols (e.g. ^GSPC) do not have float / short interest data
        mapped = map_price_ticker(ticker)
        if mapped.startswith("^"):
            return {}

        # yfinance.Ticker.info is synchronous — run in executor to stay async
        loop = asyncio.get_running_loop()
        info: dict[str, Any] = await loop.run_in_executor(
            None, lambda: yf.Ticker(ticker.upper()).info or {}
        )
        if not info:
            return {}

        short_pct = info.get("shortPercentOfFloat")
        short_ratio = info.get("shortRatio")   # days to cover
        shares_short = info.get("sharesShort")

        result: dict[str, Any] = {}

        if short_pct is not None:
            pct_value = float(short_pct) * 100  # Yahoo returns a decimal fraction
            result["short_float_pct"] = round(pct_value, 2)
            result["high_short_interest"] = pct_value > 20.0  # > 20 % of float

        if short_ratio is not None:
            ratio = float(short_ratio)
            result["days_to_cover"] = round(ratio, 1)
            result["squeeze_risk"] = ratio > 5.0  # > 5 days = elevated squeeze risk

        if shares_short is not None:
            result["shares_short"] = int(shares_short)

        return result
    except Exception:
        return {}
