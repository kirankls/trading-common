"""Tests for trading_common.features.technical (ported from chanakya options_advisor).

Verifies the pandas-ta computation path produces sane indicator values on a
synthetic OHLCV DataFrame, and that the banned tradingview-ta network-call
path has been fully stripped from the ported source.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_common.data_clients.base import Result
from trading_common.data_clients.market_data import OHLCVData
from trading_common.features.technical import (
    TechnicalSnapshot,
    compute,
    compute_weekly_confirmation,
)

_TECHNICAL_SRC_PATH = (
    Path(__file__).resolve().parents[1] / "trading_common" / "features" / "technical.py"
)


def _make_synthetic_ohlcv(n_bars: int = 260, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic n-bar daily OHLCV DataFrame with a mild uptrend + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)

    # Deterministic drift + noise random walk, floor to keep prices positive.
    daily_returns = rng.normal(loc=0.0006, scale=0.012, size=n_bars)
    close = 100.0 * np.cumprod(1 + daily_returns)

    # Derive high/low/open around close with small deterministic offsets.
    high = close * (1 + np.abs(rng.normal(0.004, 0.002, size=n_bars)))
    low = close * (1 - np.abs(rng.normal(0.004, 0.002, size=n_bars)))
    open_ = low + (high - low) * rng.uniform(0.3, 0.7, size=n_bars)
    volume = rng.integers(500_000, 2_000_000, size=n_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


@pytest.fixture
def ohlcv_60() -> pd.DataFrame:
    """60-bar synthetic OHLCV frame (minimum span called out in the task)."""
    return _make_synthetic_ohlcv(n_bars=60)


@pytest.fixture
def ohlcv_260() -> pd.DataFrame:
    """260-bar synthetic OHLCV frame — long enough to exercise 52wk/EMA200 fields."""
    return _make_synthetic_ohlcv(n_bars=260)


def _result_for(df: pd.DataFrame, ticker: str = "TEST") -> Result[OHLCVData]:
    return Result.success(OHLCVData(ticker=ticker, period="1y", df=df))


class TestComputeBasic:
    def test_returns_snapshot_for_valid_data(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        assert snap is not None
        assert isinstance(snap, TechnicalSnapshot)
        assert snap.ticker == "TEST"

    def test_returns_none_on_failed_result(self):
        bad_result: Result[OHLCVData] = Result.failure(
            error=None  # type: ignore[arg-type]
        )
        assert compute(bad_result) is None

    def test_returns_none_on_empty_dataframe(self):
        empty_df = pd.DataFrame()
        assert compute(_result_for(empty_df)) is None

    def test_tv_signal_field_always_none(self, ohlcv_60):
        """tv_signal is retained for schema compatibility but tradingview-ta is gone."""
        snap = compute(_result_for(ohlcv_60))
        assert snap.tv_signal is None


class TestIndicatorSanity:
    def test_rsi_within_bounds(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        assert snap.rsi_14 is not None
        assert 0.0 <= snap.rsi_14 <= 100.0

    def test_stoch_rsi_within_bounds(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        if snap.stoch_rsi_k is not None:
            assert 0.0 <= snap.stoch_rsi_k <= 100.0
        if snap.stoch_rsi_d is not None:
            assert 0.0 <= snap.stoch_rsi_d <= 100.0

    def test_williams_r_within_bounds(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        assert snap.williams_r is not None
        assert -100.0 <= snap.williams_r <= 0.0

    def test_mfi_within_bounds(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        assert snap.mfi_14 is not None
        assert 0.0 <= snap.mfi_14 <= 100.0

    def test_adx_within_bounds(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        if snap.adx_14 is not None:
            assert 0.0 <= snap.adx_14 <= 100.0

    def test_bb_pct_b_reasonable(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        assert snap.bb_pct_b is not None
        # bb_pct_b can occasionally go slightly outside [0, 1] when price breaks
        # the bands, but should stay within a sane extended range.
        assert -1.0 <= snap.bb_pct_b <= 2.0

    def test_bollinger_band_ordering(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        assert snap.bb_upper is not None
        assert snap.bb_middle is not None
        assert snap.bb_lower is not None
        assert snap.bb_lower <= snap.bb_middle <= snap.bb_upper

    def test_vwap_like_bb_middle_within_price_range(self, ohlcv_60):
        """No true VWAP field on TechnicalSnapshot; bb_middle (SMA20) is the
        closest local-average price metric, so sanity-check it sits within
        the overall bar range as a proxy for 'no NaN explosion / garbage value'."""
        snap = compute(_result_for(ohlcv_60))
        lo = float(ohlcv_60["low"].min())
        hi = float(ohlcv_60["high"].max())
        assert snap.bb_middle is not None
        assert lo * 0.5 <= snap.bb_middle <= hi * 1.5

    def test_atr_positive(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        assert snap.atr_14 is not None
        assert snap.atr_14 > 0

    def test_no_nan_explosion(self, ohlcv_60):
        """None of the returned float fields should be NaN — compute() guards
        every value through _safe_float, which converts NaN -> None."""
        snap = compute(_result_for(ohlcv_60))
        for field_name, value in vars(snap).items():
            if isinstance(value, float):
                assert not math.isnan(value), f"{field_name} is NaN"

    def test_ema_ordering_present(self, ohlcv_60):
        snap = compute(_result_for(ohlcv_60))
        assert snap.ema_20 is not None
        assert snap.ema_50 is not None
        # ema_200 needs >= 200 bars; not guaranteed with 60 bars via pandas-ta append
        # (pandas-ta computes even on short series, producing leading NaN-then-values,
        # so ema_200 may or may not be None here) — just check type when present.
        if snap.ema_200 is not None:
            assert snap.ema_200 > 0


class TestLongerSeriesFields:
    """Fields that require >= 220-260 bars (52wk window, ema_200 slope, Minervini, stage)."""

    def test_dist_from_52wk_high_and_low_present(self, ohlcv_260):
        snap = compute(_result_for(ohlcv_260))
        assert snap.dist_from_52wk_high_pct is not None
        assert snap.pct_above_52wk_low is not None
        assert snap.dist_from_52wk_high_pct >= 0
        assert snap.pct_above_52wk_low >= 0

    def test_stage_is_valid_when_present(self, ohlcv_260):
        snap = compute(_result_for(ohlcv_260))
        if snap.stage is not None:
            assert snap.stage in (1, 2, 3, 4)

    def test_fib_levels_bracket_swing(self, ohlcv_260):
        snap = compute(_result_for(ohlcv_260))
        assert snap.fib_swing_high is not None
        assert snap.fib_swing_low is not None
        assert snap.fib_swing_high > snap.fib_swing_low
        assert snap.fib_levels is not None
        for ratio, price in snap.fib_levels.items():
            # Retracement levels between swing low/high; extensions (1.272, 1.618)
            # can fall outside the swing range by construction.
            if ratio <= 1.0:
                assert snap.fib_swing_low <= price <= snap.fib_swing_high


class TestWeeklyConfirmation:
    def test_empty_on_insufficient_bars(self):
        short_df = _make_synthetic_ohlcv(n_bars=5)
        result = compute_weekly_confirmation(short_df)
        assert result == {}

    def test_populates_expected_keys(self):
        weekly_df = _make_synthetic_ohlcv(n_bars=60)
        result = compute_weekly_confirmation(weekly_df)
        assert "weekly_rsi_14" in result
        assert 0.0 <= result["weekly_rsi_14"] <= 100.0
        if "weekly_trend" in result:
            assert result["weekly_trend"] in ("uptrend", "downtrend", "neutral")
        if "weekly_macd_signal" in result:
            assert result["weekly_macd_signal"] in ("bullish", "bearish", "neutral")


class TestTradingViewTaStripped:
    """CRITICAL: tradingview-ta must be fully removed — banned for this project
    (1-3s network round trip is unusable at intraday speed)."""

    def test_tradingview_ta_not_in_source(self):
        """No executable reference to the tradingview-ta package remains.

        A documentation comment explaining *why* the network-call path was
        stripped is fine and expected (see the module docstring's DEVIATION
        note) — only the import/usage is banned.
        """
        source = _TECHNICAL_SRC_PATH.read_text(encoding="utf-8")
        assert "import tradingview_ta" not in source
        assert "from tradingview_ta" not in source
        assert "TA_Handler(" not in source

    def test_tradingview_ta_not_imported_after_compute(self, ohlcv_60):
        """Running compute() must not cause tradingview_ta to be imported."""
        import sys

        # Ensure a clean slate for this check (best effort — module may already
        # be absent in a fresh test process, which is the expected state).
        sys.modules.pop("tradingview_ta", None)
        compute(_result_for(ohlcv_60))
        assert "tradingview_ta" not in sys.modules

    def test_attribution_comment_present(self):
        source = _TECHNICAL_SRC_PATH.read_text(encoding="utf-8")
        assert "D:\\chanakya\\options_advisor\\features\\technical.py" in source
