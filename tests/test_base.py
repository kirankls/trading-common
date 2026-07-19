"""Unit tests for trading_common.data_clients.base.

Covers Result.success/failure construction, FetchError.__str__ formatting,
and the map_price_ticker mapping table.
"""
from __future__ import annotations

import pytest

from trading_common.data_clients.base import (
    FetchError,
    FetchErrorType,
    Result,
    map_price_ticker,
)


def test_result_success_construction():
    r = Result.success({"a": 1})
    assert r.ok is True
    assert r.data == {"a": 1}
    assert r.error is None


def test_result_failure_construction():
    err = FetchError(source="finnhub", error_type=FetchErrorType.AUTH, message="bad token")
    r = Result.failure(err)
    assert r.ok is False
    assert r.data is None
    assert r.error is err


def test_fetch_error_str_formatting():
    err = FetchError(source="polygon", error_type=FetchErrorType.RATE_LIMIT, message="too many requests")
    assert str(err) == "[polygon] rate_limit: too many requests"


@pytest.mark.parametrize(
    "error_type,expected_value",
    [
        (FetchErrorType.TIMEOUT, "timeout"),
        (FetchErrorType.AUTH, "auth"),
        (FetchErrorType.RATE_LIMIT, "rate_limit"),
        (FetchErrorType.PARSE, "parse"),
        (FetchErrorType.UNAVAILABLE, "unavailable"),
    ],
)
def test_fetch_error_type_values(error_type, expected_value):
    assert error_type.value == expected_value


def test_fetch_error_is_exception():
    err = FetchError(source="finnhub", error_type=FetchErrorType.TIMEOUT, message="timed out")
    assert isinstance(err, Exception)
    with pytest.raises(FetchError):
        raise err


@pytest.mark.parametrize(
    "ticker,expected",
    [
        ("SPXW", "^GSPC"),
        ("SPX", "^GSPC"),
        ("XSP", "^GSPC"),
        ("NDX", "^NDX"),
        ("NDXP", "^NDX"),
        ("RUT", "^RUT"),
        ("RUTW", "^RUT"),
        ("VIX", "^VIX"),
        ("spx", "^GSPC"),  # case-insensitive
        ("AAPL", "AAPL"),  # unmapped ticker passes through unchanged
    ],
)
def test_map_price_ticker(ticker, expected):
    assert map_price_ticker(ticker) == expected
