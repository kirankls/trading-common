"""Tests for trading_common.features.signal_weights (ported verbatim from
chanakya's per-signal, options-oriented Bayesian-style weight engine).

Per the module's trailing NOTE, DayTrader's own Beta(wins+1, losses+1)
per-STRATEGY ensemble (DAY_TRADER_STRATEGY.md §5) is a separate, later-
milestone reimplementation that consumes different inputs (strategy tag +
win/loss, not IV rank / ADX / VIX signal conditions) — these tests only
verify the ported module itself still behaves as chanakya's source did.
"""
from __future__ import annotations

from trading_common.features.signal_weights import (
    EnsembleConfidence,
    SignalWeights,
    compute_ensemble_confidence,
    compute_signal_weights,
)


class TestComputeSignalWeights:
    def test_empty_history_returns_defaults(self):
        weights = compute_signal_weights([])
        assert weights.weights_source == "default"
        assert weights.sample_size == 0
        assert weights.overall_accuracy == 0.50

    def test_below_min_trades_uses_default_source(self):
        trades = [{"pnl_dollar": 100.0}, {"pnl_dollar": -50.0}]
        weights = compute_signal_weights(trades)
        assert weights.weights_source == "default"
        assert weights.sample_size == 2

    def test_baseline_win_rate_reflects_closed_trades(self):
        trades = [{"pnl_dollar": p} for p in [100, 100, 100, -50, -50]]
        weights = compute_signal_weights(trades)
        assert weights.overall_accuracy == 3 / 5

    def test_at_min_trades_source_becomes_user_history(self):
        trades = [{"pnl_dollar": 100.0} for _ in range(5)]
        weights = compute_signal_weights(trades)
        assert weights.weights_source == "user_history"

    def test_iv_rank_accuracy_blends_toward_prior_with_few_qualifying_trades(self):
        # Only 1 trade qualifies the IV-rank/premium-selling condition — heavily
        # credibility-weighted toward the 0.50 prior (K=10 blending formula).
        trades = [
            {
                "pnl_dollar": 500.0,
                "strategy": "Iron Condor",
                "entry_iv_rank": 60,
            }
        ] * 5
        weights = compute_signal_weights(trades)
        # n=5 qualifying, all profitable -> computed=1.0; blend = 5/15*1.0 + 10/15*0.5 = 0.667
        assert 0.5 < weights.iv_rank_accuracy < 1.0

    def test_condition_never_observed_falls_back_to_prior(self):
        trades = [
            {"pnl_dollar": 100.0, "entry_vix": 5.0}  # outside the 15-25 "correct" band
            for _ in range(6)
        ]
        weights = compute_signal_weights(trades)
        assert weights.vix_regime_accuracy == 0.50


class TestComputeEnsembleConfidence:
    def test_empty_summary_blends_earnings_unknown_with_overall_accuracy(self):
        """IV/ADX/VIX are all 'unavailable' with an empty summary, but the
        earnings block always contributes a score (0.35, mild-caution 'unknown'
        case) — so the result is the documented 75/25 blend of that score with
        overall_accuracy, not overall_accuracy alone."""
        weights = SignalWeights(overall_accuracy=0.6, sample_size=20, weights_source="user_history")
        conf = compute_ensemble_confidence({}, weights)
        assert isinstance(conf, EnsembleConfidence)
        expected_raw = 0.35  # earnings-unknown score, sole entry in weighted_scores
        expected = round(0.75 * expected_raw + 0.25 * weights.overall_accuracy, 3)
        assert conf.score == expected

    def test_score_bounded_between_0_and_1(self):
        weights = SignalWeights()
        summary = {
            "volatility": {"iv_rank_52wk": 95},
            "technical": {"adx_14": 40},
            "macro": {"vix": 12},
            "earnings": {"days_away": 0},
        }
        conf = compute_ensemble_confidence(summary, weights)
        assert 0.0 <= conf.score <= 1.0

    def test_favourable_iv_rank_is_contributing_signal(self):
        weights = SignalWeights()
        summary = {"volatility": {"iv_rank_52wk": 85}}
        conf = compute_ensemble_confidence(summary, weights)
        assert any("favourable" in s for s in conf.contributing_signals)

    def test_missing_signals_are_limiting(self):
        weights = SignalWeights()
        conf = compute_ensemble_confidence({}, weights)
        assert "IV rank unavailable" in conf.limiting_signals
        assert "ADX unavailable" in conf.limiting_signals
        assert "VIX unavailable" in conf.limiting_signals

    def test_narrative_mentions_sample_size_when_sufficient(self):
        weights = SignalWeights(sample_size=25, weights_source="user_history")
        conf = compute_ensemble_confidence({}, weights)
        assert "25 closed trades" in conf.narrative
