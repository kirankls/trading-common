# Ported from D:\chanakya\options_advisor\features\volatility.py
"""Expected-move / realized-volatility feature computation from OHLCV data.

DEVIATION FROM SOURCE: chanakya's `volatility.py` computes a
`VolatilitySnapshot` almost entirely from a live *options chain*
(`OptionsChain`/`OptionContract` — ATM IV, put/call skew, gamma exposure,
term structure). Options are explicitly out of scope for this project phase
(see CLAUDE_CODE_PROMPT.md "Explicitly out of scope") and DayTrader does not
fetch options chains at all in Phase 0/1 — so none of that options-chain
machinery is portable here.

The only genuinely reusable piece — because it depends on OHLCV closes
alone, not the options chain — is the 30-day rolling realized-volatility
calc chanakya used as an HV-rank *proxy* for IV rank when Tradier's IV
history was unavailable (see source lines computing `hv_rank_52wk` /
`hv_percentile_52wk`). That calc is ported verbatim below as `_rolling_hv`.

On top of it, this module adds `expected_move_pct()`, a thin wrapper scaling
the current annualized realized vol down to a chosen horizon
(`sqrt(days / 252)`) — this is what strategy S3 (DAY_TRADER_STRATEGY.md §5)
needs: the ">= 0.35x expected-move fraction" gate for the 9:30-10:00 return.
It is a realized-vol proxy for expected move, not the
`ATM straddle price x 0.85` options-implied definition chanakya's LLM
orchestrator prompt references (that formula needs live option mid-prices,
which this phase doesn't have) — flagged here rather than guessed silently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class VolatilitySnapshot:
    """Realized-volatility metrics derived from OHLCV closes only."""

    ticker: str
    hv_rank_52wk: float | None          # 30-day realised vol rank (0-100)
    hv_percentile_52wk: float | None    # 30-day realised vol percentile (0-100)
    current_annualized_hv: float | None = None  # current 30-day rolling annualized vol


def _rolling_hv(ohlcv_df: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    """30-day rolling annualized realized vol -> (rank, percentile, current).

    Ported verbatim from chanakya's `compute()` HV-rank block.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return None, None, None

    close_col = "close" if "close" in ohlcv_df.columns else "Close"
    closes = ohlcv_df[close_col].dropna()
    if len(closes) < 32:
        return None, None, None

    log_returns = closes.pct_change().apply(lambda r: math.log(1 + r) if r > -1 else 0.0)
    rolling_hv = (log_returns.rolling(30).std() * math.sqrt(252)).dropna()
    if len(rolling_hv) < 2:
        return None, None, None

    current_hv = float(rolling_hv.iloc[-1])
    min_hv = float(rolling_hv.min())
    max_hv = float(rolling_hv.max())

    rank = None
    percentile = None
    if max_hv > min_hv:
        rank = round((current_hv - min_hv) / (max_hv - min_hv) * 100, 1)
        percentile = round((rolling_hv < current_hv).sum() / len(rolling_hv) * 100, 1)

    return rank, percentile, current_hv


def compute(ticker: str, ohlcv_df: pd.DataFrame | None) -> VolatilitySnapshot | None:
    """Compute realized-vol metrics for `ticker` from a daily OHLCV frame.

    Returns None if there isn't enough history (needs >= 32 bars).
    """
    rank, percentile, current_hv = _rolling_hv(ohlcv_df) if ohlcv_df is not None else (None, None, None)
    if current_hv is None:
        return None
    return VolatilitySnapshot(
        ticker=ticker,
        hv_rank_52wk=rank,
        hv_percentile_52wk=percentile,
        current_annualized_hv=current_hv,
    )


def expected_move_pct(ohlcv_df: pd.DataFrame, days: float = 1.0) -> float | None:
    """Expected fractional move over `days` trading days, as a realized-vol proxy.

    `current_annualized_hv * sqrt(days / 252)`. Used by strategy S3's
    ">= 0.35x expected-move fraction" gate on the opening-30-minute return
    (DAY_TRADER_STRATEGY.md §5). Returns None if there isn't enough history.
    """
    _, _, current_hv = _rolling_hv(ohlcv_df)
    if current_hv is None:
        return None
    return current_hv * math.sqrt(days / 252)
