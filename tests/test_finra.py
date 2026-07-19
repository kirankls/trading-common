"""Unit tests for trading_common.data_clients.finra.fetch_short_interest --
ported from the chanakya repo alongside the module itself (no dedicated
test existed there; this is new coverage written for the extracted
package).

Rules: no real network calls -- yfinance is always mocked.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from trading_common.data_clients.finra import fetch_short_interest


class TestFetchShortInterest:
    def test_index_symbol_returns_empty_dict_without_calling_yfinance(self):
        with patch("yfinance.Ticker") as mock_ticker:
            result = asyncio.run(fetch_short_interest("SPX"))  # maps to ^GSPC

        assert result == {}
        mock_ticker.assert_not_called()

    def test_parses_short_interest_fields(self):
        ticker = MagicMock()
        ticker.info = {
            "shortPercentOfFloat": 0.25,  # 25%
            "shortRatio": 6.2,
            "sharesShort": 1_000_000,
        }
        with patch("yfinance.Ticker", return_value=ticker):
            result = asyncio.run(fetch_short_interest("GME"))

        assert result["short_float_pct"] == 25.0
        assert result["high_short_interest"] is True
        assert result["days_to_cover"] == 6.2
        assert result["squeeze_risk"] is True
        assert result["shares_short"] == 1_000_000

    def test_low_short_interest_does_not_flag_risk(self):
        ticker = MagicMock()
        ticker.info = {"shortPercentOfFloat": 0.02, "shortRatio": 1.0, "sharesShort": 5000}
        with patch("yfinance.Ticker", return_value=ticker):
            result = asyncio.run(fetch_short_interest("AAPL"))

        assert result["high_short_interest"] is False
        assert result["squeeze_risk"] is False

    def test_no_info_returns_empty_dict(self):
        ticker = MagicMock()
        ticker.info = None
        with patch("yfinance.Ticker", return_value=ticker):
            result = asyncio.run(fetch_short_interest("ZZZZ"))

        assert result == {}

    def test_exception_returns_empty_dict(self):
        with patch("yfinance.Ticker", side_effect=RuntimeError("boom")):
            result = asyncio.run(fetch_short_interest("AAPL"))

        assert result == {}
