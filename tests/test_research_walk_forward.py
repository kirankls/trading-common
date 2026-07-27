"""Tests for trading_common.research.walk_forward
(docs/DAYTRADER_BACKTEST_INTEGRATION_PROMPT.md Phase 2).
"""
from __future__ import annotations

import datetime as dt

import pytest

from trading_common.research.walk_forward import (
    assert_anchored_split,
    best_by_rank,
    rank_key,
)


class TestAssertAnchoredSplit:
    def test_train_before_test_passes(self):
        assert_anchored_split(dt.date(2026, 1, 1), dt.date(2026, 1, 2))

    def test_train_equal_test_passes(self):
        # <=, not < -- the last train bar's timestamp being exactly the
        # first test bar's timestamp is the documented boundary case.
        assert_anchored_split(dt.date(2026, 1, 1), dt.date(2026, 1, 1))

    def test_train_after_test_raises(self):
        with pytest.raises(AssertionError, match="anchored walk-forward split"):
            assert_anchored_split(dt.date(2026, 1, 2), dt.date(2026, 1, 1))

    def test_works_with_datetimes_too(self):
        assert_anchored_split(
            dt.datetime(2026, 1, 1, 12, 0), dt.datetime(2026, 1, 1, 12, 0, 1)
        )

    def test_custom_label_appears_in_message(self):
        with pytest.raises(AssertionError, match="my custom window"):
            assert_anchored_split(
                dt.date(2026, 1, 2), dt.date(2026, 1, 1), label="my custom window"
            )


class TestRankKey:
    def test_higher_expectancy_wins(self):
        a = rank_key(expectancy=10.0, profit_factor=1.0, max_drawdown=100.0)
        b = rank_key(expectancy=20.0, profit_factor=1.0, max_drawdown=100.0)
        assert b > a

    def test_tiebreak_by_profit_factor(self):
        a = rank_key(expectancy=10.0, profit_factor=1.0, max_drawdown=100.0)
        b = rank_key(expectancy=10.0, profit_factor=2.0, max_drawdown=100.0)
        assert b > a

    def test_final_tiebreak_by_lower_drawdown(self):
        a = rank_key(expectancy=10.0, profit_factor=1.0, max_drawdown=200.0)
        b = rank_key(expectancy=10.0, profit_factor=1.0, max_drawdown=100.0)
        assert b > a  # lower drawdown wins

    def test_win_rate_is_not_a_parameter(self):
        # No win_rate argument exists at all -- this is a structural
        # guarantee, not just a behavioral one. Calling with an extra
        # kwarg must fail loudly.
        with pytest.raises(TypeError):
            rank_key(expectancy=10.0, profit_factor=1.0, max_drawdown=100.0, win_rate=0.9)  # type: ignore[call-arg]


class TestBestByRank:
    def test_picks_highest_expectancy(self):
        candidates = [
            {"name": "low", "expectancy": 5.0, "pf": 1.0, "dd": 50.0},
            {"name": "high", "expectancy": 15.0, "pf": 1.0, "dd": 50.0},
        ]
        best = best_by_rank(candidates, key_fn=lambda c: (c["expectancy"], c["pf"], c["dd"]))
        assert best["name"] == "high"

    def test_never_ranks_by_win_rate_even_if_present_in_the_candidate(self):
        # A candidate dict may carry a win_rate field for other purposes,
        # but best_by_rank's key_fn contract never receives it -- a
        # low-expectancy, high-win-rate candidate must still lose.
        candidates = [
            {"name": "lucky_but_thin", "expectancy": 2.0, "pf": 1.0, "dd": 50.0, "win_rate": 0.95},
            {"name": "real_edge", "expectancy": 20.0, "pf": 1.2, "dd": 80.0, "win_rate": 0.40},
        ]
        best = best_by_rank(candidates, key_fn=lambda c: (c["expectancy"], c["pf"], c["dd"]))
        assert best["name"] == "real_edge"

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            best_by_rank([], key_fn=lambda c: (0.0, 0.0, 0.0))
