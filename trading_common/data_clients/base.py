# Ported from D:\chanakya\options_advisor\data_clients\base.py
"""Base data client protocol and Result type."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

# Options-class symbols that Yahoo Finance doesn't recognise — map to the
# yfinance-compatible ticker for their underlying index price/OHLCV lookups.
# The original symbol is still used for options chain fetches (e.g. Tradier).
_YAHOO_PRICE_TICKER: dict[str, str] = {
    "SPXW": "^GSPC",   # S&P 500 weekly options → SPX index
    "SPX": "^GSPC",    # S&P 500 index
    "XSP": "^GSPC",    # Mini-SPX options
    "NDX": "^NDX",     # Nasdaq-100 index
    "NDXP": "^NDX",    # Nasdaq-100 puts
    "RUT": "^RUT",     # Russell 2000 index
    "RUTW": "^RUT",    # Russell 2000 weeklies
    "VIX": "^VIX",     # CBOE Volatility Index
}


def map_price_ticker(ticker: str) -> str:
    """Return the yfinance-compatible ticker for price/OHLCV lookups.

    Index-derivative symbols like SPXW are not valid yfinance tickers; this
    maps them to their underlying (e.g. ^GSPC) so price data resolves correctly.
    """
    return _YAHOO_PRICE_TICKER.get(ticker.upper(), ticker)


# Multi-share-class tickers where the scanner universe stores the bare/hyphen
# form (yfinance convention, e.g. "BRK-B" or "BRKB") but Schwab's
# /marketdata/v1/pricehistory endpoint only recognises the period-delimited
# form. Confirmed in production (chanakya): Schwab returned HTTP 200
# {"empty": true} for "BRKB" on every scan (schwab_ohlcv_ok stuck at 0 partly
# because of this, though most tickers never need the Schwab fallback at all
# since yfinance succeeds for them directly). Ported from chanakya's
# data_clients/base.py -- this fix was missing from the initial extraction.
_SCHWAB_SHARE_CLASS_TICKER: dict[str, str] = {
    "BRKB": "BRK.B",
    "BRK-B": "BRK.B",
    "BFA": "BF.A",
    "BF-A": "BF.A",
    "BFB": "BF.B",
    "BF-B": "BF.B",
}


def map_schwab_ticker(ticker: str) -> str:
    """Return the Schwab-compatible ticker for price/OHLCV lookups."""
    return _SCHWAB_SHARE_CLASS_TICKER.get(ticker.upper(), ticker)

T = TypeVar("T")


class FetchErrorType(Enum):
    """Categorised failure modes for data client fetches."""

    TIMEOUT = "timeout"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    PARSE = "parse"
    UNAVAILABLE = "unavailable"


@dataclass
class FetchError(Exception):
    """Describes a data-fetch failure; also an Exception so it can be raised."""

    source: str
    error_type: FetchErrorType
    message: str

    def __str__(self) -> str:
        return f"[{self.source}] {self.error_type.value}: {self.message}"


@dataclass
class Result(Generic[T]):
    """A discriminated union of success (ok=True, data set) or failure (ok=False, error set).

    All data client methods return Result[T] so callers can handle partial
    failures without exception propagation.
    """

    ok: bool
    data: T | None = None
    error: FetchError | None = None

    @classmethod
    def success(cls, data: T) -> Result[T]:
        """Construct a successful result carrying data."""
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, error: FetchError) -> Result[T]:
        """Construct a failed result carrying an error description."""
        return cls(ok=False, error=error)
