# Ported from D:\chanakya\options_advisor\features\signal_weights.py
"""
Signal weight engine — derives evidence-based weights from closed trade history.

For each signal dimension, we look at whether the signal was 'correct' at trade entry
(i.e., the trade was profitable) and compute a Bayesian-style weight.

Weights are used in two ways:
  1. Injected into signal_summary so Claude's synthesis can reference them.
  2. Used to compute an ensemble_confidence score (0-1) for the full analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

# Default weight when no history exists for a signal
_DEFAULT_WEIGHT = 0.50

# Minimum trades required before a signal weight is trusted
_MIN_TRADES_FOR_WEIGHT = 5

# Keywords that identify premium-selling / credit strategies.
# Used in ADX and IV-rank accuracy gates so all relevant strategies are counted,
# not just iron condors.
_PREMIUM_SELLING_KEYWORDS: frozenset[str] = frozenset({
    "condor", "spread", "strangle", "straddle",
    "put", "call", "wheel", "lizard", "covered", "collar",
})

_DIRECTIONAL_STRATEGIES: frozenset[str] = frozenset({
    "Bull Put Spread", "Bear Call Spread", "Bull Call Spread", "Bear Put Spread",
    "Long Call", "Long Put", "Debit Spread",
})


@dataclass
class SignalWeights:
    """Per-signal predictive weights derived from closed trade history."""

    iv_rank_accuracy: float = _DEFAULT_WEIGHT          # P(profit | iv_rank gate was correct)
    adx_accuracy: float = _DEFAULT_WEIGHT              # P(profit | adx regime matched strategy)
    vix_regime_accuracy: float = _DEFAULT_WEIGHT       # P(profit | vix tier was correct)
    earnings_proximity_accuracy: float = _DEFAULT_WEIGHT
    trend_direction_accuracy: float = _DEFAULT_WEIGHT  # RSI + MACD agreement
    overall_accuracy: float = _DEFAULT_WEIGHT          # Baseline win rate
    sample_size: int = 0
    weights_source: str = "default"  # "default" | "user_history" | "blended"


@dataclass
class EnsembleConfidence:
    """Weighted ensemble confidence for a single analysis."""

    score: float                           # 0.0 – 1.0
    contributing_signals: list[str]        # Which signals contributed positively
    limiting_signals: list[str]            # Which signals are weak/absent
    narrative: str                         # Human-readable summary for Claude
    signal_weights: SignalWeights = field(default_factory=SignalWeights)


def compute_signal_weights(closed_trades: list[dict]) -> SignalWeights:
    """
    Derive signal accuracy weights from the user's closed trade history.

    Each closed trade is expected to have:
      - pnl_dollar (float): realised P&L (positive = profit)
      - entry_iv_rank (float | None): IV rank at entry, from signal_snapshot_json
      - entry_adx (float | None): ADX at entry
      - entry_vix (float | None): VIX at entry
      - strategy (str): strategy name
      - had_earnings_proximity (bool | None): whether earnings were in DTE window
      - entry_trend_bullish (bool | None): RSI > 50 and MACD > 0 at entry

    These fields may not all be populated (older trades have no snapshot).
    Use only trades where the field is non-None.
    """
    if not closed_trades:
        return SignalWeights(weights_source="default")

    def _accuracy(trades_with_signal: list[dict], signal_key: str, threshold_fn) -> float | None:
        """
        P(profit | signal condition met). Returns None if sample too small.
        threshold_fn: callable(trade) -> bool — True if signal condition was 'correct'
        """
        relevant = [t for t in trades_with_signal if t.get(signal_key) is not None]
        if len(relevant) < _MIN_TRADES_FOR_WEIGHT:
            return None
        correct_total = sum(1 for t in relevant if threshold_fn(t))
        if correct_total == 0:
            # Known limitation (F-04): the condition was never met in the user's trade history
            # (e.g., they never entered a position when IV rank >= 40).  We cannot distinguish
            # "signal never triggered" from "signal triggered but always lost", so we return
            # None and fall back to the uninformative prior (0.50).  This is semantically
            # correct Bayesian behaviour for an unseen event.  A future improvement could
            # track a separate `never_observed` state and apply a mild penalty, but the risk
            # of penalising users who simply haven't traded in certain regimes outweighs the
            # benefit at this sample size.
            return None  # condition never observed — use prior
        correct_and_profitable = sum(
            1 for t in relevant
            if threshold_fn(t) and (t.get("pnl_dollar") or 0) > 0
        )
        return correct_and_profitable / correct_total

    # Baseline win rate across all closed trades
    wins = sum(1 for t in closed_trades if (t.get("pnl_dollar") or 0) > 0)
    baseline = wins / len(closed_trades) if closed_trades else _DEFAULT_WEIGHT

    # IV rank accuracy: credit strategies work best when IV rank >= 40
    iv_acc = _accuracy(
        closed_trades,
        "entry_iv_rank",
        lambda t: (
            (t["entry_iv_rank"] or 0) >= 40
            and any(
                kw in (t.get("strategy") or "").lower()
                for kw in _PREMIUM_SELLING_KEYWORDS
            )
        ),
    )

    # ADX accuracy: premium-selling in choppy market (ADX < 25)
    adx_acc = _accuracy(
        closed_trades,
        "entry_adx",
        lambda t: (
            (t["entry_adx"] if t["entry_adx"] is not None else 999) < 25
            and any(
                kw in (t.get("strategy") or "").lower()
                for kw in _PREMIUM_SELLING_KEYWORDS
            )
        ),
    )

    # VIX regime: credit strategies entered when VIX 15-25
    vix_acc = _accuracy(
        closed_trades,
        "entry_vix",
        lambda t: 15 <= (t["entry_vix"] or 0) <= 25,
    )

    # Trend direction: RSI > 50 and MACD > 0 at entry (stored as entry_trend_bullish bool)
    trend_acc = _accuracy(
        closed_trades,
        "entry_trend_bullish",
        lambda t: t["entry_trend_bullish"] is True,
    )

    # Blend observed accuracy with prior using a credibility-weighted formula (F-25).
    # Prior weight shrinks as the number of qualifying trades (n) grows, so small samples
    # stay anchored to the uninformative prior while large samples trust the data.
    # K=10: at 10 qualifying trades the blend is 50/50; at n→∞ the prior weight → 0%.
    def _blend(computed: float | None, n: int = 0) -> float:
        """Credibility-weighted blend of observed accuracy and uninformative prior.

        Args:
            computed: observed P(profit | condition met), or None if unseen / insufficient data.
            n: number of trades in which the signal condition was met (qualifying trades),
               NOT the total closed-trade count.
        """
        if computed is None:
            return _DEFAULT_WEIGHT
        k = 10
        w_data = n / (n + k)
        w_prior = 1.0 - w_data
        return w_data * computed + w_prior * _DEFAULT_WEIGHT

    # Count qualifying trades per signal so _blend receives the right n.
    def _count_qualifying(trades: list[dict], signal_key: str, threshold_fn) -> int:
        relevant = [t for t in trades if t.get(signal_key) is not None]
        return sum(1 for t in relevant if threshold_fn(t))

    iv_threshold = lambda t: (  # noqa: E731
        (t["entry_iv_rank"] or 0) >= 40
        and any(kw in (t.get("strategy") or "").lower() for kw in _PREMIUM_SELLING_KEYWORDS)
    )
    adx_threshold = lambda t: (  # noqa: E731
        (t["entry_adx"] if t["entry_adx"] is not None else 999) < 25
        and any(kw in (t.get("strategy") or "").lower() for kw in _PREMIUM_SELLING_KEYWORDS)
    )
    vix_threshold = lambda t: 15 <= (t["entry_vix"] or 0) <= 25  # noqa: E731
    trend_threshold = lambda t: t["entry_trend_bullish"] is True  # noqa: E731

    n_iv = _count_qualifying(closed_trades, "entry_iv_rank", iv_threshold)
    n_adx = _count_qualifying(closed_trades, "entry_adx", adx_threshold)
    n_vix = _count_qualifying(closed_trades, "entry_vix", vix_threshold)
    n_trend = _count_qualifying(closed_trades, "entry_trend_bullish", trend_threshold)

    n = len(closed_trades)
    return SignalWeights(
        iv_rank_accuracy=_blend(iv_acc, n_iv),
        adx_accuracy=_blend(adx_acc, n_adx),
        vix_regime_accuracy=_blend(vix_acc, n_vix),
        earnings_proximity_accuracy=_DEFAULT_WEIGHT,   # future: track when snapshot has earnings data
        trend_direction_accuracy=_blend(trend_acc, n_trend),
        overall_accuracy=baseline,
        sample_size=n,
        weights_source="user_history" if n >= _MIN_TRADES_FOR_WEIGHT else "default",
    )


def compute_ensemble_confidence(
    signal_summary: dict[str, Any],
    weights: SignalWeights,
) -> EnsembleConfidence:
    """
    Compute a weighted ensemble confidence score for the current analysis.

    Each signal is scored 0-1 based on how well current market conditions
    match the user's historically profitable setups, weighted by signal accuracy.
    """
    vol = signal_summary.get("volatility") or {}
    tech = signal_summary.get("technical") or {}
    macro = signal_summary.get("macro") or {}

    contributing: list[str] = []
    limiting: list[str] = []
    weighted_scores: list[tuple[float, float]] = []   # (score, weight)

    # --- IV rank signal ---
    iv_rank = vol.get("iv_rank_52wk")
    if iv_rank is not None:
        # Score: 0 at IV rank <=20, linearly rises to 1.0 at IV rank >=80
        iv_score = min(1.0, max(0.0, (iv_rank - 20) / 60))
        weighted_scores.append((iv_score, weights.iv_rank_accuracy))
        label = f"IV rank {iv_rank:.0f} ({'favourable' if iv_score > 0.5 else 'low — avoid premium-selling'})"
        (contributing if iv_score > 0.5 else limiting).append(label)
    else:
        limiting.append("IV rank unavailable")

    # --- ADX regime signal ---
    adx = tech.get("adx_14")
    if adx is not None:
        strategy = signal_summary.get("strategy_constraint", "") or ""
        is_directional = any(s.lower() in strategy.lower() for s in _DIRECTIONAL_STRATEGIES)
        if is_directional:
            adx_score = min(1.0, adx / 35)
        else:
            if adx < 15:
                adx_score = 1.0
            else:
                adx_score = max(0.0, 1.0 - (adx - 15) / 30)
        adx_score = min(1.0, adx_score)
        weighted_scores.append((adx_score, weights.adx_accuracy))
        label = (
            f"ADX {adx:.1f} ({'range-bound' if adx < 25 else 'trending — directional spreads preferred'})"
        )
        (contributing if adx_score > 0.5 else limiting).append(label)
    else:
        limiting.append("ADX unavailable")

    # --- VIX regime signal ---
    vix = macro.get("vix")
    if vix is not None:
        if 15 <= vix <= 25:
            vix_score = 0.9
            contributing.append(f"VIX {vix:.1f} (core premium-selling zone)")
        elif vix < 15:
            vix_score = 0.4
            limiting.append(f"VIX {vix:.1f} (low — premium compressed)")
        elif 25 < vix <= 40:
            vix_score = 0.5
            contributing.append(f"VIX {vix:.1f} (elevated — reduce size)")
        else:
            vix_score = 0.2
            limiting.append(f"VIX {vix:.1f} (extreme — close undefined risk)")
        weighted_scores.append((vix_score, weights.vix_regime_accuracy))
    else:
        limiting.append("VIX unavailable")

    # --- Earnings proximity signal ---
    earnings = signal_summary.get("earnings") or {}
    days_away = earnings.get("days_away")
    if days_away is not None and days_away < 0:
        weighted_scores.append((0.9, weights.earnings_proximity_accuracy))
        contributing.append("Earnings already passed — safe entry window")
    elif days_away == 0:
        weighted_scores.append((0.20, weights.earnings_proximity_accuracy))
        limiting.append("Earnings today — high risk, avoid new positions")
    elif days_away is not None and days_away > 0:
        if days_away <= 7:
            weighted_scores.append((0.3, weights.earnings_proximity_accuracy))
            limiting.append(f"Earnings in {days_away}d — high risk for undefined strategies")
        elif days_away <= 21:
            weighted_scores.append((0.6, weights.earnings_proximity_accuracy))
            limiting.append(f"Earnings in {days_away}d — monitor closely")
        else:
            weighted_scores.append((0.9, weights.earnings_proximity_accuracy))
            contributing.append(f"Earnings {days_away}d away — safe entry window")
    else:
        # F-15: earnings date unknown — apply a mild caution discount rather than ignoring
        # the risk entirely.  Omitting this entry would silently inflate the confidence
        # score for tickers whose earnings data is unavailable.
        weighted_scores.append((0.35, weights.earnings_proximity_accuracy))
        limiting.append("Earnings date unknown — mild caution applied")

    # --- Compute weighted ensemble score ---
    if weighted_scores:
        total_weight = sum(w for _, w in weighted_scores)
        if total_weight > 0:
            raw_score = sum(s * w for s, w in weighted_scores) / total_weight
        else:
            raw_score = _DEFAULT_WEIGHT
        # Blend 75% signal ensemble + 25% historical baseline win rate
        score = 0.75 * raw_score + 0.25 * weights.overall_accuracy
    else:
        score = weights.overall_accuracy

    score = round(min(1.0, max(0.0, score)), 3)

    # --- Build narrative for Claude ---
    if weights.sample_size >= _MIN_TRADES_FOR_WEIGHT:
        source_note = f"(based on {weights.sample_size} closed trades)"
    else:
        source_note = "(insufficient trade history — using market defaults)"

    contrib_str = "; ".join(contributing) if contributing else "no strong positive signals"
    limit_str = "; ".join(limiting) if limiting else "none"

    narrative = (
        f"Ensemble confidence: {score * 100:.0f}% {source_note}. "
        f"Positive signals: {contrib_str}. "
        f"Limiting factors: {limit_str}."
    )

    return EnsembleConfidence(
        score=score,
        contributing_signals=contributing,
        limiting_signals=limiting,
        narrative=narrative,
        signal_weights=weights,
    )


# ---------------------------------------------------------------------------
# NOTE on original chanakya orchestrator.py integration:
#
# The original file included an "INTEGRATION NOTE" block describing how to
# wire compute_signal_weights()/compute_ensemble_confidence() into chanakya's
# orchestrator.py and its Claude synthesis system prompt. That integration is
# specific to the options-advisor app (Claude-orchestrated ticker analysis)
# and is NOT ported here — DayTrader's strategy engine wiring for these
# functions happens in a later milestone (see task description). This module
# only needs to compile and be correct for now.
#
# DayTrader ensemble thresholds (see DAY_TRADER_STRATEGY.md §5 "Ensemble"):
#   score < 0.40        -> skip the trade
#   0.40 <= score <= 0.55 -> half size
#   score > 0.55        -> full size
#   minimum 20 trades before deviating from the uninformative prior
# These thresholds are NOT encoded in this module (the source file didn't
# parametrize them either) — the strategy engine that consumes
# EnsembleConfidence.score is responsible for applying this banding.
# ---------------------------------------------------------------------------
