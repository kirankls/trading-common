"""Tests for trading_common.features.volatility.

This module was rewritten during the M0 extraction (see the DEVIATION note
in the module docstring): chanakya's original `volatility.py` computed
almost everything from a live options chain, which is out of scope for this
project phase. Only the OHLCV-only 30-day rolling realized-vol calc was
portable; `expected_move_pct()` is new code built on top of it for
strategy S3's gate (DAY_TRADER_STRATEGY.md §5).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading_common.features.volatility import (
    VolatilitySnapshot,
    compute,
    expected_move_pct,
)


def _make_ohlcv(n_bars: int, seed: int = 7, annualized_vol: float = 0.25) -> pd.DataFrame:
    """Deterministic daily closes with an approximately known realized vol."""
    rng = np.random.default_rng(seed)
    daily_std = annualized_vol / math.sqrt(252)
    daily_returns = rng.normal(loc=0.0, scale=daily_std, size=n_bars)
    close = 100.0 * np.cumprod(1 + daily_returns)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)
    return pd.DataFrame({"close": close}, index=dates)


class TestCompute:
    def test_returns_none_with_too_little_history(self):
        df = _make_ohlcv(10)
        assert compute("TEST", df) is None

    def test_returns_none_with_no_dataframe(self):
        assert compute("TEST", None) is None

    def test_returns_snapshot_with_enough_history(self):
        df = _make_ohlcv(120)
        snap = compute("TEST", df)
        assert isinstance(snap, VolatilitySnapshot)
        assert snap.ticker == "TEST"
        assert snap.current_annualized_hv is not None
        assert snap.current_annualized_hv > 0

    def test_rank_and_percentile_within_bounds(self):
        df = _make_ohlcv(150)
        snap = compute("TEST", df)
        if snap.hv_rank_52wk is not None:
            assert 0.0 <= snap.hv_rank_52wk <= 100.0
        if snap.hv_percentile_52wk is not None:
            assert 0.0 <= snap.hv_percentile_52wk <= 100.0

    def test_accepts_capitalized_close_column(self):
        df = _make_ohlcv(120).rename(columns={"close": "Close"})
        snap = compute("TEST", df)
        assert snap is not None


class TestExpectedMovePct:
    def test_none_with_too_little_history(self):
        df = _make_ohlcv(10)
        assert expected_move_pct(df) is None

    def test_scales_with_sqrt_time(self):
        df = _make_ohlcv(150, annualized_vol=0.32)
        one_day = expected_move_pct(df, days=1.0)
        four_days = expected_move_pct(df, days=4.0)
        assert one_day is not None and four_days is not None
        assert one_day > 0
        # sqrt(4) = 2x the one-day move, exactly (same underlying HV figure).
        assert four_days == pytest.approx(one_day * 2.0, rel=1e-9)

    def test_roughly_matches_known_annualized_vol(self):
        df = _make_ohlcv(250, annualized_vol=0.30, seed=11)
        move = expected_move_pct(df, days=1.0)
        assert move is not None
        implied_annualized = move / math.sqrt(1.0 / 252)
        # Realized vol from a finite random sample won't exactly match the
        # generating parameter — allow a wide tolerance band.
        assert 0.10 < implied_annualized < 0.60
