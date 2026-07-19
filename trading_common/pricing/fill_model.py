# Ported from D:\chanakya\options_advisor\backtest\fill_model.py
"""Realistic fill-price model for options entry/exit simulation.

Pure, no I/O. Documents (and enforces in code, not just prose) the single
most common backtest/paper-fill realism bug: pricing entries/exits at the
raw mid overstates edge, because no real order fills at mid on every leg
every time.

Formula (matches the chanakya original)
----------------------------------------
Per leg:
    half_spread = (ask - bid) / 2
    slippage    = slippage_fraction * half_spread      (default 25%)
    BUY  fill = mid + slippage   (you pay more than mid to buy)
    SELL fill = mid - slippage   (you receive less than mid to sell)

This is symmetric for entry and exit — closing a position (whichever side
you're now on) uses the exact same per-leg formula, just with the leg's
current bid/ask/mid at the exit date instead of entry date.

Commissions: a flat per-contract dollar amount (default $0.65, config),
charged once per leg per fill (entry AND exit each incur it).

--------------------------------------------------------------------------
Scoping decision (Phase 4, day-trading PaperBroker options-fill simulation)
--------------------------------------------------------------------------
Ported: ``FillModelConfig``, ``leg_fill_price``, ``structure_fill``, and
``StructureFillResult`` — these are exactly what a day-trading PaperBroker
needs to simulate one entry fill and one exit fill for a multi-leg
options structure (e.g. a defined-risk vertical spread).

Deliberately NOT ported from the chanakya original:
  - ``FillModelConfig.from_settings()`` / ``default_fill_model_config()``,
    which read ``options_advisor.config.settings.settings`` (a config
    surface that doesn't exist in this repo). ``FillModelConfig()``'s
    dataclass defaults (0.25 / 0.65) are used directly instead; if this
    project later wants these tunable via ``strategy_config``/env, that
    plumbing can be added the same way ``config.settings.DayTraderSettings``
    already handles other tunables, without touching this module's core
    fill math.
  - The full multi-day backtest payoff/breakeven/margin machinery in
    ``pricing/payoff.py`` (``Leg``/``Structure`` classes, T+0 mark-to-model
    reprice via ``bs.price``, breakeven root-finding via ``scipy.optimize.
    brentq``, Reg-T margin estimation). That machinery answers "what does
    this structure's P&L look like over its whole life", which is a
    swing-trade/backtest research question. A day-trading PaperBroker only
    needs entry/exit fill-price simulation — it marks positions to the
    live/replayed chain each tick and never needs a payoff curve or a
    margin estimate to execute a fill — so none of ``payoff.py`` is
    required as a supporting type here. ``structure_fill``'s leg input is
    a plain ``(action, bid, ask, mid, quantity)`` tuple (matching the
    chanakya original exactly), not a ``payoff.Leg`` instance, so it works
    standalone with no import of ``payoff.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FillModelConfig",
    "leg_fill_price",
    "structure_fill",
    "StructureFillResult",
]


@dataclass(frozen=True)
class FillModelConfig:
    """Fill-model parameters."""

    slippage_fraction: float = 0.25       # fraction of half-spread charged as slippage per leg
    commission_per_contract: float = 0.65  # USD, charged per leg per fill (entry or exit)


def leg_fill_price(action: str, bid: float, ask: float, mid: float, config: FillModelConfig | None = None) -> float:
    """Return the realistic fill price for ONE leg's ONE fill (entry or
    exit) — never the raw mid.

    ``action``: "BUY" or "SELL" for THIS fill (a position's exit fill has
    the opposite action from its entry fill — e.g. entering a short call
    is a SELL fill; closing it later is a BUY fill. Callers pass the
    action of the fill being priced, not the position's original entry
    action).

    Falls back to *mid* untouched (no slippage adjustment, since there's no
    spread to derive it from) when bid/ask don't form a usable spread
    (e.g. an imported/thin quote with bid==ask==0) — still charges
    commission separately in ``structure_fill``.
    """
    cfg = config or FillModelConfig()
    half_spread = max(0.0, (ask - bid) / 2.0)
    slippage = cfg.slippage_fraction * half_spread
    if action.upper() == "BUY":
        return mid + slippage
    return mid - slippage


@dataclass
class StructureFillResult:
    """Aggregate fill result for a whole multi-leg structure's one fill
    event (entry or exit)."""

    net_credit_debit: float   # positive = net credit received, negative = net debit paid,
                               # AFTER slippage, BEFORE commissions
    total_commission: float   # sum of commission_per_contract * quantity across all legs
    net_cash_flow: float      # net_credit_debit adjusted for commission (commission always
                               # reduces cash flow, whichever direction): for a credit this is
                               # net_credit_debit - total_commission; for a debit this is
                               # net_credit_debit - total_commission (more negative = costs more)
    leg_fills: list[float]    # per-leg fill price, same order as the input legs


def structure_fill(
    legs: list[tuple[str, float, float, float, int]],
    config: FillModelConfig | None = None,
) -> StructureFillResult:
    """Fill an entire multi-leg structure through the slippage + commission
    model in one call.

    *legs*: list of ``(action, bid, ask, mid, quantity)`` tuples, one per
    leg (BUY/SELL, quantity always positive — a short leg is expressed via
    action="SELL", not a negative quantity). Multiplier of 100 (standard
    equity option contract) is applied internally; pass per-share
    bid/ask/mid, not per-contract.

    Returns a ``StructureFillResult`` — ``net_credit_debit`` is
    positive = credit, negative = debit (structure-level convention).
    """
    cfg = config or FillModelConfig()
    total_cash_flow = 0.0
    total_commission = 0.0
    leg_fills: list[float] = []

    for action, bid, ask, mid, quantity in legs:
        fill = leg_fill_price(action, bid, ask, mid, cfg)
        leg_fills.append(fill)
        sign = 1.0 if action.upper() == "SELL" else -1.0  # SELL receives (+), BUY pays (-)
        total_cash_flow += sign * fill * 100.0 * quantity
        total_commission += cfg.commission_per_contract * quantity

    net_credit_debit = total_cash_flow
    net_cash_flow = net_credit_debit - total_commission

    return StructureFillResult(
        net_credit_debit=round(net_credit_debit, 4),
        total_commission=round(total_commission, 4),
        net_cash_flow=round(net_cash_flow, 4),
        leg_fills=[round(f, 4) for f in leg_fills],
    )
