"""Anchored walk-forward validation invariants, shared across DayTrader and
Chanakya research/backtest tooling
(docs/DAYTRADER_BACKTEST_INTEGRATION_PROMPT.md Phase 2, ported from
DayTrader's research/walk_forward.py).

**Scope note.** DayTrader's own ``walk_forward_refit``/``parameter_grid``
machinery is tightly coupled to its own live-trading domain model
(``engine.strategy.Strategy``, ``engine.events.BarEvent``,
``engine.risk.RiskLimits``, ``research.backtest.run_backtest``). Porting
that whole apparatus into this shared, domain-agnostic package would either
drag DayTrader-only types into ``trading_common`` (breaking the "never
import one repo's code from the other" rule both projects follow) or force
Chanakya's own walk-forward
(``options_advisor.backtest.engine_backtest.run_walk_forward_validation`` —
a fixed-config date-rolling stability REPORT, not a parameter-grid search)
into a shape it doesn't need. That parameter-search machinery stays in
DayTrader's own repo; it migrates to a shared port later, on its own
schedule, if and when it's actually needed outside DayTrader.

What IS genuinely shared, and is what lives here, are the two
domain-independent invariants every walk-forward validation must uphold
regardless of what's being tested:

1. **The anchored split invariant** — train must chronologically precede
   test. A leaked/reversed split lets a parameter or weight selection "see
   the future" it's being validated against, silently invalidating the
   whole exercise (DAY_TRADER_STRATEGY.md Sec.6: "Anchored walk-forward
   12m train / 3m test rolled quarterly; report concatenated
   out-of-sample only").
2. **The ranking rule** — candidates are judged on
   ``(expectancy desc, profit_factor desc, max_drawdown asc)``.
   ``win_rate`` is NEVER part of the comparison, at any tiebreak level
   (DayTrader's program-wide invariant #6: "Expectancy over win rate:
   strategies are judged on OOS expectancy, drawdown, profit factor —
   never win percentage").
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Callable, TypeVar

__all__ = ["assert_anchored_split", "rank_key", "best_by_rank"]


def assert_anchored_split(
    train_end: date | datetime,
    test_start: date | datetime,
    *,
    label: str = "walk-forward window",
) -> None:
    """Assert *train_end* chronologically precedes (or equals) *test_start*
    — the anchored walk-forward split invariant. Raises ``AssertionError``
    with a clear message on violation rather than silently allowing an
    overlapping/reversed split.
    """
    assert train_end <= test_start, (
        f"{label}: train must chronologically precede test (anchored walk-forward split) — "
        f"train_end {train_end} is after test_start {test_start}"
    )


def rank_key(expectancy: float, profit_factor: float, max_drawdown: float) -> tuple[float, float, float]:
    """``(expectancy desc, profit_factor desc, max_drawdown asc)`` sort key.

    ``max_drawdown`` should be passed as a POSITIVE magnitude (dollars or a
    percentage — whichever unit both candidates being compared share); this
    function negates it internally so "lower drawdown wins" sorts correctly
    within the same ascending tuple as the other two (higher-is-better)
    terms. Suitable for ``max(candidates, key=lambda c: rank_key(...))`` or
    ``sorted(candidates, key=..., reverse=True)``.

    ``win_rate`` is deliberately not a parameter to this function — there is
    no way to accidentally let it influence the ranking.
    """
    return (expectancy, profit_factor, -max_drawdown)


T = TypeVar("T")


def best_by_rank(candidates: list[T], key_fn: Callable[[T], tuple[float, float, float]]) -> T:
    """Return the single best candidate per ``rank_key``, applied via
    *key_fn* (``candidate -> (expectancy, profit_factor, max_drawdown)``).

    Raises ``ValueError`` on an empty list — callers must handle "no
    candidates" explicitly rather than risk forgetting to check a ``None``
    return.
    """
    if not candidates:
        raise ValueError("best_by_rank: candidates is empty")
    return max(candidates, key=lambda c: rank_key(*key_fn(c)))
