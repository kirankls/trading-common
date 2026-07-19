# Ported from D:\chanakya\options_advisor\features\technical.py
# DEVIATION FROM SOURCE: the tradingview-ta network-call code path has been
# removed entirely (banned for this project — 1-3s network round trip is
# unusable at intraday speed). Only the pandas-ta / pure-pandas computation
# path remains. See the module docstring below and the porting report for
# exactly what was deleted.
"""Technical indicator feature computation via pandas-ta (local, no network calls)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_common.data_clients.base import Result
from trading_common.data_clients.market_data import OHLCVData


@dataclass
class TechnicalSnapshot:
    """Computed technical indicator values for a ticker."""

    ticker: str
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_histogram: float | None
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    bb_pct_b: float | None
    atr_14: float | None
    volume_zscore: float | None
    ema_20: float | None
    ema_50: float | None
    ema_200: float | None
    tv_signal: str | None  # retained for schema compatibility; always None (tradingview-ta removed)
    extra: dict[str, Any] = field(default_factory=dict)

    # Momentum oscillators
    stoch_rsi_k: float | None = None      # Stochastic RSI %K (0–100)
    stoch_rsi_d: float | None = None      # Stochastic RSI %D signal line
    williams_r: float | None = None       # Williams %R (-100 to 0)
    roc_12: float | None = None           # 12-bar rate of change %
    mfi_14: float | None = None           # Money Flow Index (0–100)

    # Trend strength
    adx_14: float | None = None           # Average Directional Index (0–100; >25 = trending)
    adx_plus_di: float | None = None      # +DI directional indicator
    adx_minus_di: float | None = None     # -DI directional indicator

    # Volume & flow
    obv: float | None = None              # On-Balance Volume (cumulative)
    obv_ema_20: float | None = None       # 20-period EMA of OBV (trend direction)
    cmf_20: float | None = None           # Chaikin Money Flow (-1 to +1)

    # Volatility expansion
    bb_width_pct: float | None = None     # Bollinger Band Width % (squeeze indicator)
    hv_10: float | None = None            # 10-day historical volatility (annualized)
    hv_30: float | None = None            # 30-day historical volatility (annualized) — standard ATM IV comparison period
    hv_60: float | None = None            # 60-day historical volatility (annualized)

    # Fibonacci levels (Elliott Wave proxy)
    fib_swing_high: float | None = None   # 60-bar rolling high
    fib_swing_low: float | None = None    # 60-bar rolling low
    fib_levels: dict | None = None        # {0.236: price, 0.382: ..., 0.5: ..., 0.618: ..., 0.786: ..., 1.272: ..., 1.618: ...}
    fib_current_zone: str | None = None   # e.g. "between_0.382_and_0.5"

    # Weekly timeframe confirmation
    weekly_rsi_14: float | None = None
    weekly_macd_signal: str | None = None   # "bullish" | "bearish" | "neutral"
    weekly_ema_20: float | None = None
    weekly_ema_50: float | None = None
    weekly_trend: str | None = None         # "uptrend" | "downtrend" | "neutral"

    # ------------------------------------------------------------------
    # Momentum / breakout stack (Minervini Trend Template + O'Neil)
    # ------------------------------------------------------------------
    ema_150: float | None = None              # 150-period EMA (Minervini TT)
    ema_200_slope_pct: float | None = None    # 20-bar slope of ema_200 in %
    dist_from_52wk_high_pct: float | None = None  # (52wk_high - close) / 52wk_high * 100
    pct_above_52wk_low: float | None = None       # (close - 52wk_low) / 52wk_low * 100
    vol_vs_50day_adv: float | None = None         # today's volume / 50-day ADV
    up_down_vol_ratio_50d: float | None = None    # sum up-day vol / sum down-day vol (50 sess)
    is_pocket_pivot_today: bool | None = None     # O'Neil pocket-pivot definition
    ants_signal: str | None = None                # "green" | "blue" | "yellow" | "gray" | None
    minervini_template_met: bool | None = None    # all 8 Minervini conditions
    stage: int | None = None                      # Weinstein stage 1-4


def _safe_float(val: Any) -> float | None:
    """Return float(val) or None when val is missing or non-numeric."""
    if val is None:
        return None
    try:
        result = float(val)
        # NaN/Inf values are not useful indicator readings
        if result != result or result in (float("inf"), float("-inf")):
            return None
        return result
    except (TypeError, ValueError):
        return None


def compute(ohlcv_result: Result[OHLCVData]) -> TechnicalSnapshot | None:
    """Compute technical indicators from an OHLCV result.

    Returns None if the result is a failure or the DataFrame is empty.
    Uses pandas-ta for RSI, MACD, Bollinger Bands, ATR, and EMAs (local
    computation only — no network calls).
    """
    if not ohlcv_result.ok or ohlcv_result.data is None:
        return None

    ohlcv_data: OHLCVData = ohlcv_result.data
    df: pd.DataFrame = ohlcv_data.df

    if df is None or df.empty:
        return None

    ticker = ohlcv_data.ticker

    # Initialise all indicator fields to None
    rsi_14 = macd = macd_signal = macd_histogram = None
    bb_upper = bb_middle = bb_lower = bb_pct_b = None
    atr_14 = volume_zscore = None
    ema_20 = ema_50 = ema_200 = None
    tv_signal: str | None = None
    extra: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # pandas-ta (local computation, no network calls)
    # ------------------------------------------------------------------
    try:

        df = df.copy()

        # RSI
        if rsi_14 is None:
            try:
                df.ta.rsi(length=14, append=True)
                col = next((c for c in df.columns if c.startswith("RSI_14")), None)
                if col:
                    rsi_14 = _safe_float(df[col].iloc[-1])
            except Exception:
                pass

        # MACD
        if macd is None:
            try:
                df.ta.macd(append=True)
                macd_col = next((c for c in df.columns if c.startswith("MACD_12_26_9")), None)
                sig_col = next((c for c in df.columns if c.startswith("MACDs_12_26_9")), None)
                hist_col = next((c for c in df.columns if c.startswith("MACDh_12_26_9")), None)
                if macd_col:
                    macd = _safe_float(df[macd_col].iloc[-1])
                if sig_col:
                    macd_signal = _safe_float(df[sig_col].iloc[-1])
                if hist_col:
                    macd_histogram = _safe_float(df[hist_col].iloc[-1])
            except Exception:
                pass

        # Bollinger Bands
        if bb_upper is None:
            try:
                df.ta.bbands(length=20, append=True)
                upper_col = next((c for c in df.columns if c.startswith("BBU_20")), None)
                mid_col = next((c for c in df.columns if c.startswith("BBM_20")), None)
                lower_col = next((c for c in df.columns if c.startswith("BBL_20")), None)
                if upper_col:
                    bb_upper = _safe_float(df[upper_col].iloc[-1])
                if mid_col:
                    bb_middle = _safe_float(df[mid_col].iloc[-1])
                if lower_col:
                    bb_lower = _safe_float(df[lower_col].iloc[-1])
            except Exception:
                pass

        # ATR
        if atr_14 is None:
            try:
                df.ta.atr(length=14, append=True)
                atr_col = next((c for c in df.columns if c.startswith("ATRr_14")), None)
                if atr_col:
                    atr_14 = _safe_float(df[atr_col].iloc[-1])
            except Exception:
                pass

        # EMAs
        for length, attr_name in ((20, "ema_20"), (50, "ema_50"), (200, "ema_200")):
            try:
                df.ta.ema(length=length, append=True)
                col = next(
                    (c for c in df.columns if c.upper() == f"EMA_{length}"), None
                )
                if col and locals()[attr_name] is None:
                    locals()[attr_name]  # reference to avoid unused-var lint
                    val = _safe_float(df[col].iloc[-1])
                    if attr_name == "ema_20":
                        ema_20 = val
                    elif attr_name == "ema_50":
                        ema_50 = val
                    elif attr_name == "ema_200":
                        ema_200 = val
            except Exception:
                pass

    except Exception as pta_exc:
        extra["pandas_ta_error"] = str(pta_exc)

    # ------------------------------------------------------------------
    # Pure-pandas fallback — runs when pandas-ta is unavailable or failed
    # to fill a value, using only numpy/pandas (always present via yfinance).
    # ------------------------------------------------------------------
    try:
        closes = df["close"].astype(float)
        highs = df["high"].astype(float) if "high" in df.columns else closes
        lows = df["low"].astype(float) if "low" in df.columns else closes

        # RSI-14 (outer guard already ensures >= 20 weekly bars)
        if rsi_14 is None:
            delta = closes.diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi_14 = _safe_float(100 - 100 / (1 + rs.iloc[-1]))

        # MACD (12, 26, 9)
        if macd is None and len(closes) >= 35:
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd = _safe_float(macd_line.iloc[-1])
            macd_signal = _safe_float(signal_line.iloc[-1])
            macd_histogram = _safe_float((macd_line - signal_line).iloc[-1])

        # EMAs
        for span, target in ((20, "ema_20"), (50, "ema_50"), (200, "ema_200")):
            if locals()[target] is None and len(closes) >= span:
                val = _safe_float(closes.ewm(span=span, adjust=False).mean().iloc[-1])
                if target == "ema_20":
                    ema_20 = val
                elif target == "ema_50":
                    ema_50 = val
                elif target == "ema_200":
                    ema_200 = val

        # Bollinger Bands (20, 2σ)
        if bb_upper is None and len(closes) >= 20:
            sma20 = closes.rolling(20).mean()
            std20 = closes.rolling(20).std()
            bb_middle = _safe_float(sma20.iloc[-1])
            bb_upper = _safe_float((sma20 + 2 * std20).iloc[-1])
            bb_lower = _safe_float((sma20 - 2 * std20).iloc[-1])

        # ATR-14
        if atr_14 is None and len(closes) >= 15:
            tr = pd.concat([
                highs - lows,
                (highs - closes.shift()).abs(),
                (lows - closes.shift()).abs(),
            ], axis=1).max(axis=1)
            atr_14 = _safe_float(tr.ewm(com=13, adjust=False).mean().iloc[-1])

    except Exception:
        pass

    # ------------------------------------------------------------------
    # Extended indicators — pure pandas (no extra deps)
    # Always computed from raw OHLCV.
    # ------------------------------------------------------------------
    try:
        closes = df["close"].astype(float)
        highs = df["high"].astype(float) if "high" in df.columns else closes
        lows = df["low"].astype(float) if "low" in df.columns else closes
    except Exception:
        closes = highs = lows = None

    # Stochastic RSI (14,14,3,3)
    stoch_rsi_k: float | None = None
    stoch_rsi_d: float | None = None
    try:
        if stoch_rsi_k is None and closes is not None and len(closes) >= 28:
            delta = closes.diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            rs = gain / loss.clip(lower=1e-10)
            rsi_series = 100 - 100 / (1 + rs)
            rsi_min = rsi_series.rolling(14).min()
            rsi_max = rsi_series.rolling(14).max()
            rsi_range = rsi_max - rsi_min
            stoch_k_raw = ((rsi_series - rsi_min) / rsi_range.replace(0, float("nan"))) * 100
            stoch_k_smooth = stoch_k_raw.rolling(3).mean()
            stoch_d_smooth = stoch_k_smooth.rolling(3).mean()
            stoch_rsi_k = _safe_float(stoch_k_smooth.iloc[-1])
            stoch_rsi_d = _safe_float(stoch_d_smooth.iloc[-1])
    except Exception:
        pass

    # Williams %R (14)
    williams_r: float | None = None
    try:
        if williams_r is None and closes is not None and len(closes) >= 14:
            high14 = highs.rolling(14).max()
            low14 = lows.rolling(14).min()
            williams_r = _safe_float(
                -100 * (high14.iloc[-1] - closes.iloc[-1]) / max(high14.iloc[-1] - low14.iloc[-1], 1e-9)
            )
    except Exception:
        pass

    # Rate of Change (12-bar)
    roc_12 = None
    try:
        if closes is not None and len(closes) >= 13:
            prev = closes.iloc[-13]
            if prev != 0:
                roc_12 = _safe_float((closes.iloc[-1] - prev) / prev * 100)
    except Exception:
        pass

    # Money Flow Index (14)
    mfi_14 = None
    try:
        if closes is not None and "volume" in df.columns and len(closes) >= 15:
            typical_price = (highs + lows + closes) / 3
            raw_mf = typical_price * df["volume"].astype(float)
            tp_diff = typical_price.diff()
            pos_mf = raw_mf.where(tp_diff > 0, 0).rolling(14).sum()
            neg_mf = raw_mf.where(tp_diff < 0, 0).rolling(14).sum()
            mfr = pos_mf / neg_mf.replace(0, float("nan"))
            mfi_series = 100 - 100 / (1 + mfr)
            mfi_14 = _safe_float(mfi_series.iloc[-1])
    except Exception:
        pass

    # ADX (14) with +DI and -DI
    adx_14: float | None = None
    adx_plus_di: float | None = None
    adx_minus_di: float | None = None
    try:
        if adx_14 is None and closes is not None and len(closes) >= 28:
            prev_high = highs.shift(1)
            prev_low = lows.shift(1)
            plus_dm_raw = (highs - prev_high).clip(lower=0)
            minus_dm_raw = (prev_low - lows).clip(lower=0)
            # Apply Wilder's rule: zero the smaller DM; when equal, zero both.
            # plus_dm is kept only when strictly greater; minus_dm only when strictly greater.
            mask = plus_dm_raw > minus_dm_raw
            plus_dm = plus_dm_raw.where(mask, 0.0)
            minus_dm = minus_dm_raw.where(minus_dm_raw > plus_dm_raw, 0.0)
            tr = pd.concat([
                highs - lows,
                (highs - closes.shift()).abs(),
                (lows - closes.shift()).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, float("nan"))
            minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, float("nan"))
            di_sum = plus_di + minus_di
            dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, float("nan"))
            adx_series = dx.ewm(alpha=1/14, adjust=False).mean()
            adx_14 = _safe_float(adx_series.iloc[-1])
            adx_plus_di = _safe_float(plus_di.iloc[-1])
            adx_minus_di = _safe_float(minus_di.iloc[-1])
    except Exception:
        pass

    # On-Balance Volume (OBV) + EMA-20 of OBV
    obv = obv_ema_20 = None
    try:
        if closes is not None and "volume" in df.columns and len(closes) >= 2:
            direction = closes.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv_series = (direction * df["volume"].astype(float)).cumsum()
            obv = _safe_float(obv_series.iloc[-1])
            if len(obv_series) >= 20:
                obv_ema_20 = _safe_float(obv_series.ewm(span=20, adjust=False).mean().iloc[-1])
    except Exception:
        pass

    # Chaikin Money Flow (20)
    cmf_20 = None
    try:
        if closes is not None and "volume" in df.columns and len(closes) >= 20:
            hl_range = (highs - lows).replace(0, float("nan"))
            mfv = ((2 * closes - highs - lows) / hl_range) * df["volume"].astype(float)
            vol_sum = df["volume"].astype(float).rolling(20).sum()
            cmf_series = mfv.rolling(20).sum() / vol_sum.replace(0, float("nan"))
            cmf_20 = _safe_float(cmf_series.iloc[-1])
    except Exception:
        pass

    # BB Width %
    bb_width_pct = None
    try:
        if bb_upper is not None and bb_lower is not None and bb_middle is not None and bb_middle != 0:
            bb_width_pct = _safe_float((bb_upper - bb_lower) / bb_middle * 100)
    except Exception:
        pass

    # HV-10, HV-30, and HV-60 (annualized log-return volatility)
    hv_10: float | None = None
    hv_30: float | None = None
    hv_60: float | None = None
    try:
        import math as _math
        if closes is not None:
            log_returns = closes.pct_change().apply(lambda r: _math.log(1 + r) if (r == r and r > -1) else float("nan"))
            try:
                if len(log_returns) >= 11:
                    hv_10 = float(log_returns.rolling(10).std().iloc[-1] * (252 ** 0.5))
            except Exception:
                pass
            try:
                if len(log_returns) >= 31:
                    hv_30 = float(log_returns.rolling(30).std().iloc[-1] * (252 ** 0.5))
            except Exception:
                pass
            try:
                if len(log_returns) >= 61:
                    hv_60 = float(log_returns.rolling(60).std().iloc[-1] * (252 ** 0.5))
            except Exception:
                pass
    except Exception:
        pass

    # Fibonacci retracement levels (60-bar swing)
    fib_swing_high = fib_swing_low = fib_levels = fib_current_zone = None
    try:
        if closes is not None:
            high_col = "high" if "high" in df.columns else "High"
            low_col = "low" if "low" in df.columns else "Low"
            fib_highs = df[high_col].dropna().astype(float)
            fib_lows = df[low_col].dropna().astype(float)
            window = min(60, len(closes))
            if window >= 10 and len(fib_highs) >= window and len(fib_lows) >= window:
                sw_high = float(fib_highs.iloc[-window:].max())
                sw_low = float(fib_lows.iloc[-window:].min())
                if sw_high != sw_low:
                    fib_swing_high = sw_high
                    fib_swing_low = sw_low
                    price_range = sw_high - sw_low
                    fib_levels = {
                        0.236: round(sw_high - 0.236 * price_range, 4),
                        0.382: round(sw_high - 0.382 * price_range, 4),
                        0.5:   round(sw_high - 0.5   * price_range, 4),
                        0.618: round(sw_high - 0.618 * price_range, 4),
                        0.786: round(sw_high - 0.786 * price_range, 4),
                        # Extensions: key is Fib ratio; formula sw_low - (ratio-1)*range
                        # is equivalent to sw_high - ratio*range (same price level).
                        1.272: round(sw_low  - 0.272 * price_range, 4),
                        1.618: round(sw_low  - 0.618 * price_range, 4),
                    }
                    current = float(closes.iloc[-1])
                    fibs_sorted = sorted(fib_levels.items(), key=lambda x: x[1], reverse=True)
                    zone = None
                    for i in range(len(fibs_sorted) - 1):
                        upper_ratio, upper_price = fibs_sorted[i]
                        lower_ratio, lower_price = fibs_sorted[i + 1]
                        if lower_price <= current <= upper_price:
                            r1, r2 = sorted([lower_ratio, upper_ratio])
                            zone = f"between_{r1}_and_{r2}"
                            break
                    if zone is None:
                        if current > fibs_sorted[0][1]:
                            zone = f"above_{fibs_sorted[0][0]}"
                        else:
                            # Price is below the lowest Fibonacci level (the 1.618 extension);
                            # use the actual ratio of that level rather than a hardcoded label.
                            zone = f"below_{fibs_sorted[-1][0]}"
                    fib_current_zone = zone
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Derived metrics (computed from whatever indicator values we have)
    # ------------------------------------------------------------------

    # bb_pct_b: position within Bollinger Bands
    try:
        if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
            close_val = _safe_float(df["close"].iloc[-1])
            if close_val is not None:
                bb_pct_b = (close_val - bb_lower) / (bb_upper - bb_lower)
    except Exception:
        pass

    # volume_zscore: compare current bar to the prior 20 bars (exclude current to avoid look-ahead)
    try:
        if "volume" in df.columns and len(df) >= 21:
            vol_window = df["volume"].iloc[-21:-1]
            vol_mean = float(vol_window.mean())
            vol_std = float(vol_window.std())
            last_vol = float(df["volume"].iloc[-1])
            if vol_std > 0:
                volume_zscore = (last_vol - vol_mean) / vol_std
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Momentum / breakout stack (Minervini, O'Neil, Weinstein, ANTS)
    # ------------------------------------------------------------------
    ema_150: float | None = None
    ema_200_slope_pct: float | None = None
    dist_from_52wk_high_pct: float | None = None
    pct_above_52wk_low: float | None = None
    vol_vs_50day_adv: float | None = None
    up_down_vol_ratio_50d: float | None = None
    is_pocket_pivot_today: bool | None = None
    ants_signal: str | None = None
    minervini_template_met: bool | None = None
    stage: int | None = None

    try:
        if closes is not None and len(closes) >= 1:
            close_now = float(closes.iloc[-1])

            # EMA-150
            if len(closes) >= 150:
                try:
                    ema_150_series = closes.ewm(span=150, adjust=False).mean()
                    ema_150 = _safe_float(ema_150_series.iloc[-1])
                except Exception:
                    pass

            # EMA-200 slope over last 20 sessions (%)
            try:
                if len(closes) >= 220:
                    ema_200_series = closes.ewm(span=200, adjust=False).mean()
                    cur = float(ema_200_series.iloc[-1])
                    prior = float(ema_200_series.iloc[-21])  # 20 sessions ago
                    if prior != 0:
                        ema_200_slope_pct = _safe_float((cur - prior) / prior * 100)
            except Exception:
                pass

            # 52-week (252-session) high/low distances
            try:
                window_52w = min(252, len(closes))
                if window_52w >= 20:
                    high_col_name = "high" if "high" in df.columns else ("High" if "High" in df.columns else None)
                    low_col_name = "low" if "low" in df.columns else ("Low" if "Low" in df.columns else None)
                    hs = df[high_col_name].astype(float).iloc[-window_52w:] if high_col_name else closes.iloc[-window_52w:]
                    ls = df[low_col_name].astype(float).iloc[-window_52w:] if low_col_name else closes.iloc[-window_52w:]
                    high_52w = float(hs.max())
                    low_52w = float(ls.min())
                    if high_52w > 0:
                        dist_from_52wk_high_pct = _safe_float((high_52w - close_now) / high_52w * 100)
                    if low_52w > 0:
                        pct_above_52wk_low = _safe_float((close_now - low_52w) / low_52w * 100)
            except Exception:
                pass

            # Volume vs 50-day ADV
            try:
                if "volume" in df.columns and len(df) >= 51:
                    vols = df["volume"].astype(float)
                    adv_50 = float(vols.iloc[-51:-1].mean())
                    cur_vol = float(vols.iloc[-1])
                    if adv_50 > 0:
                        vol_vs_50day_adv = _safe_float(cur_vol / adv_50)
            except Exception:
                pass

            # Up/Down volume ratio over last 50 sessions
            try:
                if "volume" in df.columns and len(closes) >= 51:
                    last_50_closes = closes.iloc[-51:]   # need one prior for first diff
                    last_50_vols = df["volume"].astype(float).iloc[-51:]
                    diffs = last_50_closes.diff().iloc[1:]  # 50 diffs
                    vols_50 = last_50_vols.iloc[1:]
                    up_vol = float(vols_50[diffs > 0].sum())
                    down_vol = float(vols_50[diffs < 0].sum())
                    if down_vol > 0:
                        up_down_vol_ratio_50d = _safe_float(up_vol / down_vol)
            except Exception:
                pass

            # Pocket pivot today (O'Neil / Morales)
            #   - close > prior close
            #   - close > SMA(10)
            #   - today's volume > max(volume on any down-day in prior 10 sessions)
            try:
                if "volume" in df.columns and len(closes) >= 12:
                    sma_10 = float(closes.iloc[-10:].mean())
                    prev_close = float(closes.iloc[-2])
                    cur_vol = float(df["volume"].astype(float).iloc[-1])
                    prior_closes = closes.iloc[-12:-1]  # 11 bars ending yesterday
                    prior_vols = df["volume"].astype(float).iloc[-12:-1]
                    prior_diffs = prior_closes.diff().iloc[1:]  # 10 diffs (last 10 sessions)
                    prior_vols_aligned = prior_vols.iloc[1:]
                    down_day_vols = prior_vols_aligned[prior_diffs < 0]
                    max_down_vol = float(down_day_vols.max()) if len(down_day_vols) > 0 else 0.0
                    is_pocket_pivot_today = bool(
                        (close_now > prev_close)
                        and (close_now > sma_10)
                        and (cur_vol > max_down_vol)
                    )
            except Exception:
                pass

            # ANTS signal — 15-session momentum/price/volume burst
            try:
                if "volume" in df.columns and len(closes) >= 16:
                    last_16_closes = closes.iloc[-16:]
                    diffs_15 = last_16_closes.diff().iloc[1:]  # 15 diffs
                    up_days = int((diffs_15 > 0).sum())
                    momentum_ok = up_days >= 12

                    close_15_ago = float(closes.iloc[-16])
                    price_change = (close_now - close_15_ago) / close_15_ago if close_15_ago != 0 else 0.0
                    price_ok = price_change >= 0.20

                    vols = df["volume"].astype(float)
                    vol_now = float(vols.iloc[-1])
                    vol_15_ago = float(vols.iloc[-16])
                    vol_change = (vol_now - vol_15_ago) / vol_15_ago if vol_15_ago != 0 else 0.0
                    volume_ok = vol_change >= 0.20

                    if momentum_ok and price_ok and volume_ok:
                        ants_signal = "green"
                    elif momentum_ok and price_ok:
                        ants_signal = "blue"
                    elif momentum_ok and volume_ok:
                        ants_signal = "yellow"
                    elif momentum_ok:
                        ants_signal = "gray"
                    else:
                        ants_signal = None
            except Exception:
                pass

            # Minervini Trend Template — 8 conditions
            # NOTE: spec allows ema_150/ema_200 as proxies for sma_150/sma_200.
            try:
                if (
                    ema_150 is not None
                    and ema_200 is not None
                    and ema_50 is not None
                    and ema_200_slope_pct is not None
                    and dist_from_52wk_high_pct is not None
                    and pct_above_52wk_low is not None
                ):
                    high_52w_val = close_now / (1 - dist_from_52wk_high_pct / 100) if dist_from_52wk_high_pct < 100 else None
                    low_52w_val = close_now / (1 + pct_above_52wk_low / 100) if pct_above_52wk_low > -100 else None

                    cond1 = close_now > ema_150                                # close > sma_150 proxy
                    cond2 = close_now > ema_200                                # close > sma_200 proxy
                    cond3 = ema_150 > ema_200                                  # sma_150 > sma_200 proxy
                    cond4 = ema_200_slope_pct > 0                              # rising 20 sessions
                    cond5 = (ema_50 > ema_150) and (ema_50 > ema_200)
                    cond6 = close_now > ema_50
                    cond7 = (low_52w_val is not None) and (close_now >= 1.25 * low_52w_val)
                    cond8 = (high_52w_val is not None) and (close_now >= 0.75 * high_52w_val)

                    minervini_template_met = bool(
                        cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8
                    )
            except Exception:
                pass

            # Weinstein stage classification (1-4)
            try:
                if (
                    ema_150 is not None
                    and ema_200 is not None
                    and ema_200_slope_pct is not None
                ):
                    if (
                        close_now > ema_150
                        and ema_150 > ema_200
                        and ema_200_slope_pct > 0
                    ):
                        stage = 2
                    elif (
                        close_now < ema_150
                        and ema_150 < ema_200
                        and ema_200_slope_pct < 0
                    ):
                        stage = 4
                    elif (
                        abs(close_now - ema_150) / ema_150 < 0.05
                        and -1 < ema_200_slope_pct < 1
                    ):
                        stage = 1
                    else:
                        stage = 3
            except Exception:
                pass
    except Exception:
        pass

    return TechnicalSnapshot(
        ticker=ticker,
        rsi_14=rsi_14,
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_pct_b=bb_pct_b,
        atr_14=atr_14,
        volume_zscore=volume_zscore,
        ema_20=ema_20,
        ema_50=ema_50,
        ema_200=ema_200,
        tv_signal=tv_signal,
        extra=extra,
        stoch_rsi_k=stoch_rsi_k,
        stoch_rsi_d=stoch_rsi_d,
        williams_r=williams_r,
        roc_12=roc_12,
        mfi_14=mfi_14,
        adx_14=adx_14,
        adx_plus_di=adx_plus_di,
        adx_minus_di=adx_minus_di,
        obv=obv,
        obv_ema_20=obv_ema_20,
        cmf_20=cmf_20,
        bb_width_pct=bb_width_pct,
        hv_10=hv_10,
        hv_30=hv_30,
        hv_60=hv_60,
        fib_swing_high=fib_swing_high,
        fib_swing_low=fib_swing_low,
        fib_levels=fib_levels,
        fib_current_zone=fib_current_zone,
        ema_150=ema_150,
        ema_200_slope_pct=ema_200_slope_pct,
        dist_from_52wk_high_pct=dist_from_52wk_high_pct,
        pct_above_52wk_low=pct_above_52wk_low,
        vol_vs_50day_adv=vol_vs_50day_adv,
        up_down_vol_ratio_50d=up_down_vol_ratio_50d,
        is_pocket_pivot_today=is_pocket_pivot_today,
        ants_signal=ants_signal,
        minervini_template_met=minervini_template_met,
        stage=stage,
    )


def compute_weekly_confirmation(weekly_df: pd.DataFrame) -> dict:
    """Compute weekly-bar confirmation signals from a weekly OHLCV DataFrame.

    Returns dict of weekly_* fields. Returns empty dict on any failure.
    Requires at least 20 weekly bars (roughly 5 months of data).
    """
    result: dict = {}
    try:
        if weekly_df is None or weekly_df.empty or len(weekly_df) < 20:
            return result

        closes = (
            weekly_df["close"].astype(float)
            if "close" in weekly_df.columns
            else weekly_df["Close"].astype(float)
        )

        # Weekly RSI-14 (clip to tiny floor avoids NaN when all bars are gains/losses)
        if len(closes) >= 15:
            delta = closes.diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            rs = gain / loss.clip(lower=1e-10)
            rsi_val = _safe_float(100 - 100 / (1 + rs.iloc[-1]))
            if rsi_val is not None:
                result["weekly_rsi_14"] = rsi_val

        # Weekly MACD histogram signal
        if len(closes) >= 35:
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            hist_val = (macd_line - signal_line).iloc[-1]  # NaN evaluates as False in comparisons — resolves to "neutral"
            result["weekly_macd_signal"] = (
                "bullish" if hist_val > 0 else ("bearish" if hist_val < 0 else "neutral")
            )

        # Weekly EMAs
        if len(closes) >= 20:
            result["weekly_ema_20"] = _safe_float(
                closes.ewm(span=20, adjust=False).mean().iloc[-1]
            )
        if len(closes) >= 50:
            result["weekly_ema_50"] = _safe_float(
                closes.ewm(span=50, adjust=False).mean().iloc[-1]
            )

        # Weekly trend — price vs. EMA-20 and EMA-20 vs. EMA-50
        price = closes.iloc[-1]
        w20 = result.get("weekly_ema_20")
        w50 = result.get("weekly_ema_50")
        if w20 is not None and w50 is not None:
            if price > w20 and w20 > w50:
                result["weekly_trend"] = "uptrend"
            elif price < w20 and w20 < w50:
                result["weekly_trend"] = "downtrend"
            else:
                result["weekly_trend"] = "neutral"

    except Exception:
        pass

    return result
