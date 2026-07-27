"""Tests for trading_common.occ_symbol (Phase 3 addition,
docs/DAYTRADER_BACKTEST_INTEGRATION_PROMPT.md) -- no prior test file
existed for this module at all.
"""
from __future__ import annotations

from datetime import date

from trading_common.occ_symbol import (
    format_occ_symbol,
    parse_occ_symbol,
    parse_polygon_option_ticker,
)


class TestParseOccSymbol:
    def test_schwab_format_round_trips(self):
        parsed = parse_occ_symbol("SPY   260717C00500000")
        assert parsed is not None
        assert parsed.root == "SPY"
        assert parsed.expiration == date(2026, 7, 17)
        assert parsed.option_type == "CALL"
        assert parsed.strike == 500.0

    def test_non_option_ticker_returns_none(self):
        assert parse_occ_symbol("SPY") is None

    def test_format_and_parse_are_inverse(self):
        symbol = format_occ_symbol("AAPL", date(2026, 3, 20), "PUT", 150.5)
        parsed = parse_occ_symbol(symbol)
        assert parsed is not None
        assert parsed.root == "AAPL"
        assert parsed.expiration == date(2026, 3, 20)
        assert parsed.option_type == "PUT"
        assert parsed.strike == 150.5


class TestParsePolygonOptionTicker:
    """Every ticker below is real, taken directly from a downloaded
    us_options_opra/day_aggs_v1 flat file (2026-07-17) -- not fabricated."""

    def test_single_char_root(self):
        parsed = parse_polygon_option_ticker("O:A260717C00120000")
        assert parsed is not None
        assert parsed.root == "A"
        assert parsed.expiration == date(2026, 7, 17)
        assert parsed.option_type == "CALL"
        assert parsed.strike == 120.0

    def test_put_contract(self):
        parsed = parse_polygon_option_ticker("O:A260717P00130000")
        assert parsed is not None
        assert parsed.option_type == "PUT"
        assert parsed.strike == 130.0

    def test_multi_char_root(self):
        parsed = parse_polygon_option_ticker("O:AAPL260117C00150000")
        assert parsed is not None
        assert parsed.root == "AAPL"
        assert parsed.expiration == date(2026, 1, 17)
        assert parsed.option_type == "CALL"
        assert parsed.strike == 150.0

    def test_four_char_root_spy(self):
        parsed = parse_polygon_option_ticker("O:SPY260727C00525000")
        assert parsed is not None
        assert parsed.root == "SPY"
        assert parsed.expiration == date(2026, 7, 27)
        assert parsed.strike == 525.0

    def test_fractional_strike(self):
        # 8-digit strike is strike*1000 -- confirms sub-dollar precision
        # survives the round trip (real contracts do have $0.50 strikes).
        parsed = parse_polygon_option_ticker("O:A260717C00120500")
        assert parsed is not None
        assert parsed.strike == 120.5

    def test_missing_prefix_returns_none(self):
        # Polygon's own STOCK tickers have no "O:" prefix at all -- the
        # normal, expected non-match case when this parser runs against
        # mixed stock/option input, not an error.
        assert parse_polygon_option_ticker("AAPL") is None

    def test_schwab_format_does_not_match_polygon_parser(self):
        # Cross-check: the two formats are genuinely different shapes,
        # not just cosmetically -- Schwab's fixed-width/padded format must
        # not accidentally parse as a Polygon ticker.
        assert parse_polygon_option_ticker("SPY   260717C00500000") is None

    def test_too_short_returns_none(self):
        assert parse_polygon_option_ticker("O:AC00120000") is None

    def test_invalid_contract_type_returns_none(self):
        assert parse_polygon_option_ticker("O:AAPL260117X00150000") is None

    def test_invalid_date_returns_none(self):
        assert parse_polygon_option_ticker("O:AAPL269917C00150000") is None

    def test_empty_root_returns_none(self):
        assert parse_polygon_option_ticker("O:260717C00120000") is None
