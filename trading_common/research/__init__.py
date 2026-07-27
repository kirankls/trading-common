"""Shared research/validation primitives for DayTrader and Chanakya's
backtest tooling (docs/DAYTRADER_BACKTEST_INTEGRATION_PROMPT.md Phase 2).

See walk_forward.py and parity.py for what's shared and why each is scoped
to domain-independent statistics/invariants rather than either project's
own backtest-replay machinery.
"""
