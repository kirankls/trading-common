"""Unit tests for trading_common.pricing.bs and trading_common.pricing.fill_model.

bs.price/bs.greeks are checked against a well-known textbook Black-Scholes
reference (Hull, "Options, Futures, and Other Derivatives": S=42, K=40,
T=0.5, r=0.10, sigma=0.20 -> call ~= 4.76, put ~= 0.81 via put-call parity)
rather than merely asserting a float comes back.

leg_fill_price/structure_fill are checked against hand-computed fills for a
simple 2-leg vertical spread with known bid/ask/mid.
"""
from __future__ import annotations

import math

import pytest

from trading_common.pricing.bs import greeks, price
from trading_common.pricing.fill_model import FillModelConfig, leg_fill_price, structure_fill

# ---------------------------------------------------------------------------
# bs.price / bs.greeks
# ---------------------------------------------------------------------------

# Hull textbook example (10th ed., ch. 15/19): S=42, K=40, T=0.5, r=0.10,
# sigma=0.20, q=0 -> call price = 4.76 (to 2dp).
_S, _K, _T, _R, _SIGMA = 42.0, 40.0, 0.5, 0.10, 0.20


def test_price_call_matches_textbook_value():
    call_price = price(_S, _K, _T, _R, _SIGMA, "call")
    assert isinstance(call_price, float)
    assert call_price == pytest.approx(4.76, abs=0.01)


def test_price_put_matches_put_call_parity():
    call_price = price(_S, _K, _T, _R, _SIGMA, "call")
    put_price = price(_S, _K, _T, _R, _SIGMA, "put")
    # Put-call parity: C - P = S - K*e^(-rT)
    lhs = call_price - put_price
    rhs = _S - _K * math.exp(-_R * _T)
    assert lhs == pytest.approx(rhs, abs=1e-6)
    assert put_price == pytest.approx(0.81, abs=0.01)


def test_price_atm_call_known_value():
    # S=K=100, T=1, r=0.05, sigma=0.2, q=0 -> a widely-cited reference value
    # for this exact parameterisation is ~10.4506.
    call_price = price(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    assert call_price == pytest.approx(10.4506, abs=0.01)


def test_price_at_expiry_returns_intrinsic_value():
    # T=0 should short-circuit to intrinsic value, not blow up py_vollib.
    itm_call = price(110.0, 100.0, 0.0, 0.05, 0.2, "call")
    otm_call = price(90.0, 100.0, 0.0, 0.05, 0.2, "call")
    assert itm_call == pytest.approx(10.0)
    assert otm_call == pytest.approx(0.0)


def test_greeks_call_delta_matches_textbook_value():
    g = greeks(_S, _K, _T, _R, _SIGMA, "call")
    assert set(g.keys()) >= {"delta", "gamma", "theta", "vega"}
    # Hull's worked example gives delta ~= 0.7794 for this call.
    assert g["delta"] == pytest.approx(0.7791, abs=0.01)
    assert g["gamma"] > 0
    assert g["vega"] > 0


def test_greeks_put_delta_is_call_delta_minus_one():
    call_delta = greeks(_S, _K, _T, _R, _SIGMA, "call")["delta"]
    put_delta = greeks(_S, _K, _T, _R, _SIGMA, "put")["delta"]
    assert put_delta == pytest.approx(call_delta - 1.0, abs=1e-6)


def test_price_rejects_invalid_option_type():
    with pytest.raises(ValueError):
        price(_S, _K, _T, _R, _SIGMA, "straddle")


# ---------------------------------------------------------------------------
# leg_fill_price / structure_fill
# ---------------------------------------------------------------------------


def test_leg_fill_price_buy_pays_above_mid():
    # bid=1.00, ask=1.20 -> mid=1.10, half_spread=0.10
    # slippage = 0.25 * 0.10 = 0.025 -> BUY fill = 1.10 + 0.025 = 1.125
    fill = leg_fill_price("BUY", bid=1.00, ask=1.20, mid=1.10, config=FillModelConfig())
    assert fill == pytest.approx(1.125)


def test_leg_fill_price_sell_receives_below_mid():
    # SELL fill = 1.10 - 0.025 = 1.075
    fill = leg_fill_price("SELL", bid=1.00, ask=1.20, mid=1.10, config=FillModelConfig())
    assert fill == pytest.approx(1.075)


def test_leg_fill_price_zero_spread_falls_back_to_mid():
    fill = leg_fill_price("BUY", bid=0.0, ask=0.0, mid=0.50, config=FillModelConfig())
    assert fill == pytest.approx(0.50)


def test_leg_fill_price_custom_slippage_fraction():
    cfg = FillModelConfig(slippage_fraction=0.5)
    # half_spread=0.10, slippage=0.5*0.10=0.05 -> BUY fill=1.15
    fill = leg_fill_price("BUY", bid=1.00, ask=1.20, mid=1.10, config=cfg)
    assert fill == pytest.approx(1.15)


def test_structure_fill_vertical_spread_hand_computed():
    """A 2-leg bull call vertical: BUY 1x 100C (bid=2.00,ask=2.20,mid=2.10),
    SELL 1x 105C (bid=0.90,ask=1.10,mid=1.00), 1 contract each.

    Hand computation with default config (slippage_fraction=0.25,
    commission_per_contract=0.65):
      leg 1 (BUY 100C): half_spread=0.10, slippage=0.025, fill=2.125
      leg 2 (SELL 105C): half_spread=0.10, slippage=0.025, fill=0.975

      cash flow leg1 (BUY, pays) = -2.125 * 100 * 1 = -212.5
      cash flow leg2 (SELL, receives) = +0.975 * 100 * 1 = +97.5
      net_credit_debit = -212.5 + 97.5 = -115.0  (net debit)

      total_commission = 0.65 * 1 + 0.65 * 1 = 1.30
      net_cash_flow = -115.0 - 1.30 = -116.30
    """
    legs = [
        ("BUY", 2.00, 2.20, 2.10, 1),
        ("SELL", 0.90, 1.10, 1.00, 1),
    ]
    result = structure_fill(legs, FillModelConfig())

    assert result.leg_fills == [pytest.approx(2.125), pytest.approx(0.975)]
    assert result.net_credit_debit == pytest.approx(-115.0)
    assert result.total_commission == pytest.approx(1.30)
    assert result.net_cash_flow == pytest.approx(-116.30)


def test_structure_fill_multi_contract_scales_commission_and_cash_flow():
    # Same vertical as above but 3 contracts each.
    legs = [
        ("BUY", 2.00, 2.20, 2.10, 3),
        ("SELL", 0.90, 1.10, 1.00, 3),
    ]
    result = structure_fill(legs, FillModelConfig())

    # cash flow leg1 = -2.125 * 100 * 3 = -637.5
    # cash flow leg2 = +0.975 * 100 * 3 = +292.5
    # net_credit_debit = -637.5 + 292.5 = -345.0
    assert result.net_credit_debit == pytest.approx(-345.0)
    # commission = 0.65*3 + 0.65*3 = 3.90
    assert result.total_commission == pytest.approx(3.90)
    assert result.net_cash_flow == pytest.approx(-348.90)


def test_structure_fill_defaults_to_default_config_when_none_passed():
    legs = [("BUY", 1.00, 1.20, 1.10, 1)]
    explicit = structure_fill(legs, FillModelConfig())
    implicit = structure_fill(legs)
    assert implicit == explicit
