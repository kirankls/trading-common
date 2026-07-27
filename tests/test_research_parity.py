"""Tests for trading_common.research.parity
(docs/DAYTRADER_BACKTEST_INTEGRATION_PROMPT.md Phase 2).

Ported alongside the module itself from Chanakya's
tests/unit/test_learning_promotion.py::TestBootstrap/TestRegimeWorse — the
same assertions, generalized field names (bucketed_materially_worse takes
a_key/b_key/bucket_key explicitly instead of promotion.py's hardcoded
production_realized/shadow_realized/vix_bucket).
"""
from __future__ import annotations

from trading_common.research.parity import (
    bucketed_materially_worse,
    paired_bootstrap_p_value,
)


class TestPairedBootstrapPValue:
    def test_clear_outperformance_low_p(self):
        diffs = [30.0] * 80
        assert paired_bootstrap_p_value(diffs) < 0.1

    def test_no_outperformance_high_p(self):
        diffs = [-5.0] * 80
        assert paired_bootstrap_p_value(diffs) == 1.0  # non-positive mean

    def test_empty(self):
        assert paired_bootstrap_p_value([]) == 1.0

    def test_deterministic_given_same_seed(self):
        diffs = [10.0, -5.0, 8.0, 12.0, -2.0] * 10
        p1 = paired_bootstrap_p_value(diffs, seed=42)
        p2 = paired_bootstrap_p_value(diffs, seed=42)
        assert p1 == p2

    def test_different_seed_can_differ(self):
        diffs = [10.0, -5.0, 8.0, 12.0, -2.0] * 3
        p1 = paired_bootstrap_p_value(diffs, seed=1, iters=200)
        p2 = paired_bootstrap_p_value(diffs, seed=2, iters=200)
        # Not asserting inequality (could coincidentally match) -- just
        # that both are valid probabilities and the seed param is honored
        # (see test_deterministic_given_same_seed for the honoring proof).
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0


class TestBucketedMateriallyWorse:
    def test_detects_worse_bucket(self):
        decisions = [
            {"a": -10.0, "b": 20.0, "regime": "panic"}
            for _ in range(15)
        ]
        worse = bucketed_materially_worse(
            decisions, bucket_key="regime", a_key="a", b_key="b", margin=5.0, min_samples=5,
        )
        assert "panic" in worse

    def test_thin_bucket_ignored(self):
        decisions = [
            {"a": -10.0, "b": 20.0, "regime": "panic"}
            for _ in range(3)  # < min_samples
        ]
        worse = bucketed_materially_worse(
            decisions, bucket_key="regime", a_key="a", b_key="b", margin=5.0, min_samples=5,
        )
        assert worse == []

    def test_missing_values_skipped_not_crashed(self):
        decisions = [
            {"a": None, "b": 20.0, "regime": "panic"},
            {"a": -10.0, "b": None, "regime": "panic"},
        ]
        worse = bucketed_materially_worse(
            decisions, bucket_key="regime", a_key="a", b_key="b", margin=5.0, min_samples=1,
        )
        assert worse == []  # both rows skipped -- no data to judge

    def test_missing_bucket_key_defaults_to_unknown(self):
        decisions = [{"a": -10.0, "b": 20.0} for _ in range(5)]
        worse = bucketed_materially_worse(
            decisions, bucket_key="regime", a_key="a", b_key="b", margin=5.0, min_samples=5,
        )
        assert "unknown" in worse

    def test_not_worse_when_within_margin(self):
        decisions = [{"a": 18.0, "b": 20.0, "regime": "normal"} for _ in range(10)]
        worse = bucketed_materially_worse(
            decisions, bucket_key="regime", a_key="a", b_key="b", margin=5.0, min_samples=5,
        )
        assert worse == []
