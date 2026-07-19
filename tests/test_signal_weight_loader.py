"""Tests for trading_common.features.signal_weight_loader.

`enrich_with_snapshots` imports `trading_common.storage.repositories.AnalysisHistory`
inside a try/except — that model is options-advisor-specific and intentionally
NOT ported (DayTrader has no per-analysis snapshot table), so the expected
behavior is a graceful, non-fatal fall-through returning the trade dicts
unenriched (matches the original chanakya behavior on any DB/import failure).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading_common.features.signal_weight_loader import (
    build_trade_dicts_for_weighting,
    enrich_with_snapshots,
)


class TestBuildTradeDictsForWeighting:
    def test_empty_list_returns_empty(self):
        assert build_trade_dicts_for_weighting([]) == []

    def test_extracts_expected_fields(self):
        trade = SimpleNamespace(pnl_dollar=123.45, strategy="Iron Condor", analysis_id="abc-1")
        rows = build_trade_dicts_for_weighting([trade])
        assert len(rows) == 1
        row = rows[0]
        assert row["pnl_dollar"] == 123.45
        assert row["strategy"] == "Iron Condor"
        assert row["analysis_id"] == "abc-1"
        # Signal-context fields default to None until enriched.
        assert row["entry_iv_rank"] is None
        assert row["entry_adx"] is None
        assert row["entry_vix"] is None
        assert row["entry_trend_bullish"] is None
        assert row["had_earnings_proximity"] is None

    def test_missing_attributes_default_gracefully(self):
        trade = SimpleNamespace()  # no attributes at all
        rows = build_trade_dicts_for_weighting([trade])
        assert rows[0]["pnl_dollar"] is None
        assert rows[0]["strategy"] is None

    def test_non_numeric_pnl_becomes_none(self):
        trade = SimpleNamespace(pnl_dollar="not-a-number")
        rows = build_trade_dicts_for_weighting([trade])
        assert rows[0]["pnl_dollar"] is None


class TestEnrichWithSnapshots:
    @pytest.mark.asyncio
    async def test_empty_trade_dicts_returns_empty(self):
        result = await enrich_with_snapshots([], db=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_analysis_ids_returns_unmodified(self):
        trade_dicts = [{"pnl_dollar": 1.0, "analysis_id": None}]
        result = await enrich_with_snapshots(trade_dicts, db=None)
        assert result == trade_dicts

    @pytest.mark.asyncio
    async def test_missing_analysis_history_model_falls_back_gracefully(self):
        """AnalysisHistory doesn't exist in trading_common — the import inside
        enrich_with_snapshots must fail into the except branch and return the
        dicts unenriched rather than raising."""
        trade = SimpleNamespace(pnl_dollar=1.0, analysis_id="some-id")
        trade_dicts = build_trade_dicts_for_weighting([trade])
        result = await enrich_with_snapshots(trade_dicts, db=object())
        assert result == trade_dicts
        assert result[0]["entry_iv_rank"] is None
