"""Feature-engineering package: technical indicators, volatility, and signal weighting.

Ported from D:\\chanakya\\options_advisor\\features\\ (extract-don't-rewrite port).
The tradingview-ta network-call path was stripped from technical.py — pandas-ta /
pure-pandas computation only.
"""
from __future__ import annotations

from trading_common.features.signal_weight_loader import (
    build_trade_dicts_for_weighting,
    enrich_with_snapshots,
)
from trading_common.features.signal_weights import (
    EnsembleConfidence,
    SignalWeights,
    compute_ensemble_confidence,
    compute_signal_weights,
)
from trading_common.features.technical import (
    TechnicalSnapshot,
    compute_weekly_confirmation,
)
from trading_common.features.technical import (
    compute as compute_technical,
)
from trading_common.features.volatility import (
    VolatilitySnapshot,
)
from trading_common.features.volatility import (
    compute as compute_volatility,
)

__all__ = [
    "TechnicalSnapshot",
    "compute_technical",
    "compute_weekly_confirmation",
    "VolatilitySnapshot",
    "compute_volatility",
    "SignalWeights",
    "EnsembleConfidence",
    "compute_signal_weights",
    "compute_ensemble_confidence",
    "build_trade_dicts_for_weighting",
    "enrich_with_snapshots",
]
