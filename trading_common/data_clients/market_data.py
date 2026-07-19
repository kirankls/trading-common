"""OHLCV data container used by `trading_common.features.technical`.

Ported from `D:\\chanakya\\options_advisor\\data_clients\\market_data.py` —
only the `OHLCVData` dataclass is extracted here. The full yfinance-backed
market data client is options-advisor-specific and out of scope for
DayTrader (market data comes from the Schwab stream + REST history / Polygon
backfill instead — see `engine/market_data.py`, a later milestone).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OHLCVData:
    """OHLCV price data for a ticker (daily or intraday, per `period`)."""

    ticker: str
    period: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
