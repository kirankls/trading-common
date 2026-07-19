"""Shared external-data client package.

Ported from D:\\chanakya\\options_advisor\\data_clients\\ — Result/FetchError
discriminated-union pattern, in-process TTL cache, and the Finnhub/Polygon
HTTP clients used for earnings/economic-calendar and historical bar data.
"""
from __future__ import annotations

from trading_common.data_clients import finnhub, polygon
from trading_common.data_clients.base import (
    FetchError,
    FetchErrorType,
    Result,
    map_price_ticker,
)

__all__ = [
    "FetchError",
    "FetchErrorType",
    "Result",
    "map_price_ticker",
    "finnhub",
    "polygon",
]
