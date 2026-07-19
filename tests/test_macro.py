"""Unit tests for trading_common.data_clients.macro.MacroClient -- ported
from the chanakya repo alongside the module itself (no dedicated test
existed there; this is new coverage written for the extracted package).

Rules: no real network calls -- yfinance and requests are always mocked.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from trading_common.data_clients.macro import MacroClient


def _mock_vix(price: float = 15.5) -> MagicMock:
    ticker = MagicMock()
    ticker.info = {"regularMarketPrice": price}
    return ticker


class TestFetchContext:
    def test_no_fred_key_leaves_fred_fields_none_but_still_fetches_vix_and_events(self):
        with (
            patch("trading_common.data_clients.macro.settings.fred_api_key.get_secret_value", return_value=""),
            patch("trading_common.data_clients.macro.yf.Ticker", return_value=_mock_vix()),
            patch("requests.get", side_effect=RuntimeError("no network in tests")),
        ):
            result = MacroClient().fetch_context()

        assert result.ok is True
        assert result.data.fed_funds_rate is None
        assert result.data.cpi_yoy is None
        assert result.data.ten_year_yield is None
        assert result.data.vix == 15.5
        assert len(result.data.upcoming_events) > 0  # static FOMC/CPI/NFP fallback still populates

    def test_fred_key_present_fetches_fred_series(self):
        fake_fred = MagicMock()
        fake_series = MagicMock()
        fake_series.empty = False
        fake_series.iloc = [5.25]
        fake_series.__len__ = lambda self: 13
        fake_series.pct_change.return_value.iloc = [3.2 / 100]
        fake_fred.get_series.return_value = fake_series

        with (
            patch("trading_common.data_clients.macro.settings.fred_api_key.get_secret_value", return_value="fake-key"),
            patch("trading_common.data_clients.macro.yf.Ticker", return_value=_mock_vix()),
            patch("requests.get", side_effect=RuntimeError("no network in tests")),
            patch.dict("sys.modules", {"fredapi": MagicMock(Fred=lambda api_key: fake_fred)}),
        ):
            result = MacroClient().fetch_context()

        assert result.ok is True
        assert result.data.fed_funds_rate == 5.25

    def test_vix_fetch_failure_leaves_vix_none_but_does_not_fail_the_whole_result(self):
        with (
            patch("trading_common.data_clients.macro.settings.fred_api_key.get_secret_value", return_value=""),
            patch("trading_common.data_clients.macro.yf.Ticker", side_effect=RuntimeError("yfinance down")),
            patch("requests.get", side_effect=RuntimeError("no network in tests")),
        ):
            result = MacroClient().fetch_context()

        assert result.ok is True
        assert result.data.vix is None

    def test_fetch_async_wrapper_delegates_to_fetch_context(self):
        with (
            patch("trading_common.data_clients.macro.settings.fred_api_key.get_secret_value", return_value=""),
            patch("trading_common.data_clients.macro.yf.Ticker", return_value=_mock_vix()),
            patch("requests.get", side_effect=RuntimeError("no network in tests")),
        ):
            result = asyncio.run(MacroClient().fetch())

        assert result.ok is True
        assert result.data.vix == 15.5
