"""Unit tests for trading_common.data_clients.options_chain.OptionsChainClient.

Covers the Result[T, FetchError] success/failure paths for the primary
(Schwab) source. Follows the same mocking conventions as
tests/brokers/test_schwab.py (schwab.auth.client_from_token_file and
SchwabTokenManager.plaintext_context are monkeypatched -- no real network
call, no real token file, no real encryption key) and
tests/trading_common/test_finnhub.py (respx for the Tradier HTTP fallback).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from trading_common.data_clients import _cache
from trading_common.data_clients.base import FetchErrorType, Result
from trading_common.data_clients.options_chain import (
    OptionsChain,
    OptionsChainClient,
    _resolve_option_root,
)

_SCHWAB_CHAIN_RESPONSE = {
    "underlyingPrice": 450.0,
    "callExpDateMap": {
        "2026-08-21:48": {
            "450.0": [
                {
                    "symbol": "SPY_082126C450",
                    "bid": 5.0,
                    "ask": 5.20,
                    "mark": 5.10,
                    "last": 5.10,
                    "totalVolume": 1200,
                    "openInterest": 4500,
                    "volatility": 18.5,
                    "delta": 0.52,
                    "gamma": 0.02,
                    "theta": -0.05,
                    "vega": 0.12,
                    "rho": 0.03,
                    "inTheMoney": True,
                }
            ]
        }
    },
    "putExpDateMap": {
        "2026-08-21:48": {
            "450.0": [
                {
                    "symbol": "SPY_082126P450",
                    "bid": 4.80,
                    "ask": 5.00,
                    "mark": 4.90,
                    "last": 4.90,
                    "totalVolume": 900,
                    "openInterest": 3000,
                    "volatility": 19.0,
                    "delta": -0.48,
                    "gamma": 0.02,
                    "theta": -0.04,
                    "vega": 0.11,
                    "rho": -0.03,
                    "inTheMoney": False,
                }
            ]
        }
    },
}


# A chain fetched for underlying "SPX" mixes a true monthly (3rd-Friday,
# AM-settled, OCC root "SPX") contract and a weekly (PM-settled, OCC root
# "SPXW") contract for a DIFFERENT expiry, together in the same response --
# exactly how Schwab's real API behaves (one fetch, both kinds together).
# 2026-07-17 is a real 3rd Friday (verified: date(2026,7,17).weekday()==4,
# 15 <= 17 <= 21); 2026-07-10 is a Friday but NOT the 3rd Friday (a weekly).
_SPX_CHAIN_RESPONSE = {
    "underlyingPrice": 5050.0,
    "callExpDateMap": {
        "2026-07-17:5": {
            "5000.0": [
                {
                    "symbol": "SPX_071726C5000",
                    "bid": 60.0,
                    "ask": 61.0,
                    "mark": 60.5,
                    "last": 60.5,
                    "totalVolume": 100,
                    "openInterest": 500,
                    "volatility": 15.0,
                    "delta": 0.55,
                    "gamma": 0.01,
                    "theta": -0.10,
                    "vega": 0.20,
                    "rho": 0.05,
                    "inTheMoney": True,
                    "expirationType": "S",
                }
            ]
        },
        "2026-07-10:2": {
            "5040.0": [
                {
                    "symbol": "SPXW_071026C5040",
                    "bid": 20.0,
                    "ask": 21.0,
                    "mark": 20.5,
                    "last": 20.5,
                    "totalVolume": 80,
                    "openInterest": 300,
                    "volatility": 16.0,
                    "delta": 0.50,
                    "gamma": 0.01,
                    "theta": -0.12,
                    "vega": 0.18,
                    "rho": 0.04,
                    "inTheMoney": False,
                    "expirationType": "W",
                }
            ]
        },
    },
    "putExpDateMap": {},
}


class FakeResponse:
    """Stand-in for schwab-py's response object: .status_code, .json(), .raise_for_status()."""

    def __init__(self, *, status_code: int = 200, json_data: Any = None) -> None:
        self.status_code = status_code
        self._json_data = json_data

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=MagicMock(), response=MagicMock(status_code=self.status_code))


@contextmanager
def _no_op_plaintext_context(_path):
    yield "irrelevant-fake-path"


@pytest.fixture(autouse=True)
def _clear_cache():
    _cache.clear()
    yield
    _cache.clear()


@pytest.fixture(autouse=True)
def _patch_plaintext_context(monkeypatch):
    """Never touch a real (encrypted) token file or the real encryption
    master key -- the token manager's context manager is replaced with a
    no-op that yields a placeholder path."""
    monkeypatch.setattr(
        "trading_common.services.schwab_token.SchwabTokenManager.plaintext_context",
        lambda self, path: _no_op_plaintext_context(path),
    )


@pytest.fixture(autouse=True)
def _configured_settings(monkeypatch, tmp_path):
    """Point trading_common.config.settings.settings at fake, non-empty
    Schwab creds and a token path that exists (Path.exists() gate in
    _fetch_from_schwab), and clear Tradier so the fallback chain doesn't
    accidentally engage in Schwab-focused tests."""
    from pydantic import SecretStr

    from trading_common.config.settings import settings

    token_file = tmp_path / "schwab_token.enc.json"
    token_file.write_bytes(b"irrelevant-ciphertext")

    monkeypatch.setattr(settings, "schwab_trading_app_key", SecretStr("fake-app-key"))
    monkeypatch.setattr(settings, "schwab_trading_app_secret", SecretStr("fake-app-secret"))
    monkeypatch.setattr(settings, "schwab_token_path", str(token_file))
    monkeypatch.setattr(settings, "tradier_api_key", SecretStr(""))
    return settings


def _patch_schwab_client(monkeypatch, fake_client: MagicMock) -> None:
    monkeypatch.setattr("schwab.auth.client_from_token_file", lambda *args, **kwargs: fake_client)


def _block_yfinance_fallback(monkeypatch) -> None:
    """Prevent the yfinance tertiary fallback from making a real network
    call in tests that intentionally exercise a Schwab failure path --
    those tests want to observe the Schwab-specific Result.failure(), not
    whatever yfinance's fallback happens to return."""
    from trading_common.data_clients.base import FetchError, FetchErrorType, Result

    monkeypatch.setattr(
        OptionsChainClient,
        "_fetch_from_yfinance",
        lambda self, ticker, expiries: Result.failure(
            FetchError("yfinance", FetchErrorType.UNAVAILABLE, "blocked in test")
        ),
    )


class TestFetchChainSchwabPrimary:
    def test_success_returns_normalised_chain(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(
            return_value=FakeResponse(status_code=200, json_data=_SCHWAB_CHAIN_RESPONSE)
        )
        _patch_schwab_client(monkeypatch, fake_client)

        result = OptionsChainClient().fetch_chain("SPY")

        assert result.ok is True
        assert result.error is None
        chain = result.data
        assert chain is not None
        assert chain.source == OptionsChainClient.SOURCE_SCHWAB
        assert chain.ticker == "SPY"
        assert chain.underlying_price == 450.0
        assert chain.expiries == ["2026-08-21"]
        assert len(chain.contracts) == 2

        call = next(c for c in chain.contracts if c.option_type == "call")
        assert call.strike == 450.0
        assert call.bid == 5.0
        assert call.ask == 5.20
        assert call.delta == 0.52
        assert call.implied_volatility == pytest.approx(0.185)
        assert call.in_the_money is True
        # Regression guard: ordinary equity underlyings must still resolve
        # option_root == the plain ticker (SPY has only one OCC root).
        assert call.option_root == "SPY"

        put = next(c for c in chain.contracts if c.option_type == "put")
        assert put.strike == 450.0
        assert put.delta == -0.48
        assert put.option_root == "SPY"

    def test_success_filters_to_requested_expiries(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(
            return_value=FakeResponse(status_code=200, json_data=_SCHWAB_CHAIN_RESPONSE)
        )
        _patch_schwab_client(monkeypatch, fake_client)

        result = OptionsChainClient().fetch_chain("SPY", expiries=["1999-01-01"])

        assert result.ok is True
        assert result.data.contracts == []
        assert result.data.expiries == []

    def test_401_returns_auth_failure(self, monkeypatch):
        # Exercises _fetch_from_schwab directly: fetch_chain() would fall
        # through to Tradier/yfinance on any Schwab failure (by design --
        # matches the chanakya original's fallback-chain behaviour), so the
        # Schwab-specific Result.failure() is only observable at this layer.
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(return_value=FakeResponse(status_code=401))
        _patch_schwab_client(monkeypatch, fake_client)

        result = OptionsChainClient()._fetch_from_schwab("SPY", None, "fake-key", "fake-secret")

        assert result.ok is False
        assert result.error.source == OptionsChainClient.SOURCE_SCHWAB
        assert result.error.error_type == FetchErrorType.AUTH

    def test_429_returns_rate_limit_failure(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(return_value=FakeResponse(status_code=429))
        _patch_schwab_client(monkeypatch, fake_client)

        result = OptionsChainClient()._fetch_from_schwab("SPY", None, "fake-key", "fake-secret")

        assert result.ok is False
        assert result.error.error_type == FetchErrorType.RATE_LIMIT

    def test_missing_token_file_returns_auth_failure_without_touching_schwab_py(self, monkeypatch, _configured_settings):
        # No get_option_chain patch needed -- the missing-token-file branch
        # must short-circuit before any schwab-py client is constructed.
        monkeypatch.setattr(_configured_settings, "schwab_token_path", "does-not-exist.json")

        result = OptionsChainClient()._fetch_from_schwab("SPY", None, "fake-key", "fake-secret")

        assert result.ok is False
        assert result.error.source == OptionsChainClient.SOURCE_SCHWAB
        assert result.error.error_type == FetchErrorType.AUTH

    def test_unexpected_exception_returns_unavailable_failure(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(side_effect=RuntimeError("boom"))
        _patch_schwab_client(monkeypatch, fake_client)

        result = OptionsChainClient()._fetch_from_schwab("SPY", None, "fake-key", "fake-secret")

        assert result.ok is False
        assert result.error.error_type == FetchErrorType.UNAVAILABLE

    def test_fetch_chain_falls_through_to_yfinance_when_schwab_fails(self, monkeypatch):
        """End-to-end: a Schwab 401 does not surface as the final Result --
        fetch_chain() continues down the fallback chain (Tradier not
        configured in this fixture -> yfinance), matching chanakya's
        original "only fail when every provider is unavailable" contract."""
        _block_yfinance_fallback(monkeypatch)
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(return_value=FakeResponse(status_code=401))
        _patch_schwab_client(monkeypatch, fake_client)

        result = OptionsChainClient().fetch_chain("SPY")

        assert result.ok is False
        assert result.error.source == "yfinance"  # last-attempted source's failure surfaces

    def test_result_is_cached_for_subsequent_calls(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(
            return_value=FakeResponse(status_code=200, json_data=_SCHWAB_CHAIN_RESPONSE)
        )
        _patch_schwab_client(monkeypatch, fake_client)

        client = OptionsChainClient()
        first = client.fetch_chain("SPY")
        second = client.fetch_chain("SPY")

        assert first.ok is True and second.ok is True
        assert fake_client.get_option_chain.call_count == 1  # second call served from cache


class TestFetchChainSchwabNotConfiguredFallsThroughToTradier:
    @respx.mock
    def test_no_schwab_creds_uses_tradier(self, monkeypatch, _configured_settings):
        from pydantic import SecretStr

        monkeypatch.setattr(_configured_settings, "schwab_trading_app_key", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "schwab_trading_app_secret", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "tradier_api_key", SecretStr("fake-tradier-key"))

        respx.get("https://api.tradier.com/v1/markets/options/expirations").mock(
            return_value=httpx.Response(200, json={"expirations": {"date": ["2026-08-21"]}})
        )
        respx.get("https://api.tradier.com/v1/markets/options/chains").mock(
            return_value=httpx.Response(
                200,
                json={
                    "options": {
                        "option": [
                            {
                                "symbol": "SPY260821C00450000",
                                "strike": 450.0,
                                "option_type": "call",
                                "bid": 5.0,
                                "ask": 5.2,
                                "last": 5.1,
                                "volume": 100,
                                "open_interest": 200,
                                "greeks": {"mid_iv": 0.19, "delta": 0.5},
                                "in_the_money": True,
                            }
                        ]
                    }
                },
            )
        )

        result = OptionsChainClient().fetch_chain("SPY")

        assert result.ok is True
        assert result.data.source == OptionsChainClient.SOURCE_TRADIER
        assert len(result.data.contracts) == 1

    def test_no_schwab_and_no_tradier_creds_falls_through_to_yfinance(self, monkeypatch, _configured_settings):
        from pydantic import SecretStr

        monkeypatch.setattr(_configured_settings, "schwab_trading_app_key", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "schwab_trading_app_secret", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "tradier_api_key", SecretStr(""))

        client = OptionsChainClient()
        monkeypatch.setattr(
            client,
            "_fetch_from_yfinance",
            lambda ticker, expiries: __import__("trading_common.data_clients.base", fromlist=["Result"]).Result.failure(
                __import__("trading_common.data_clients.base", fromlist=["FetchError"]).FetchError(
                    "yfinance", FetchErrorType.UNAVAILABLE, "no network in test"
                )
            ),
        )

        result = client.fetch_chain("SPY")

        assert result.ok is False
        assert result.error.source == "yfinance"


class TestFetchChainSpxRootResolution:
    """SPX splits its OCC root by contract type (monthly "SPX" vs weekly
    "SPXW") -- confirmed from schwab-py's own OptionSymbol docstring (see
    trading_common/data_clients/options_chain.py's OptionContract.
    option_root docstring). One chain fetch for "SPX" returns BOTH kinds
    of contract together, so root resolution must happen per-contract."""

    def test_spx_chain_resolves_monthly_and_weekly_roots_correctly(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(
            return_value=FakeResponse(status_code=200, json_data=_SPX_CHAIN_RESPONSE)
        )
        _patch_schwab_client(monkeypatch, fake_client)

        result = OptionsChainClient().fetch_chain("SPX")

        assert result.ok is True
        chain = result.data
        assert chain is not None
        assert len(chain.contracts) == 2

        monthly = next(c for c in chain.contracts if c.expiry == "2026-07-17")
        assert monthly.option_root == "SPX"

        weekly = next(c for c in chain.contracts if c.expiry == "2026-07-10")
        assert weekly.option_root == "SPXW"


class TestSpxRequiresSchwabFailClosed:
    """M4 (SPX OPTIONS AUDIT): SPX/SPXW must never receive a Tradier- or
    yfinance-sourced chain -- both fallbacks default every contract's
    `option_root` to the plain ticker (see `_fetch_from_tradier` /
    `_fetch_from_yfinance`), which is wrong for the monthly/weekly OCC-root
    split real SPX/SPXW chains have. An order built from a wrong
    `option_root` would target the wrong contract at the broker, so
    `fetch_chain` must fail closed (return `Result.failure()`, `data is
    None`) for these two underlyings rather than silently falling back."""

    @respx.mock
    def test_spx_without_schwab_does_not_fall_back_to_tradier(self, monkeypatch, _configured_settings):
        from pydantic import SecretStr

        monkeypatch.setattr(_configured_settings, "schwab_trading_app_key", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "schwab_trading_app_secret", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "tradier_api_key", SecretStr("fake-tradier-key"))

        # Tradier's endpoints are mocked to succeed -- if the SPX/SPXW
        # guard were missing, fetch_chain would happily return this
        # (wrong-root) chain. Asserting the route was never even hit
        # proves the guard short-circuits before any Tradier call.
        expirations_route = respx.get("https://api.tradier.com/v1/markets/options/expirations").mock(
            return_value=httpx.Response(200, json={"expirations": {"date": ["2026-08-21"]}})
        )
        chains_route = respx.get("https://api.tradier.com/v1/markets/options/chains").mock(
            return_value=httpx.Response(
                200,
                json={"options": {"option": [{"symbol": "SPX260821C05000000", "strike": 5000.0}]}},
            )
        )

        result = OptionsChainClient().fetch_chain("SPX")

        assert result.ok is False
        assert result.data is None
        assert result.error is not None
        assert result.error.source == OptionsChainClient.SOURCE_SCHWAB
        assert not expirations_route.called
        assert not chains_route.called

    def test_spxw_without_schwab_or_tradier_does_not_fall_back_to_yfinance(self, monkeypatch, _configured_settings):
        from pydantic import SecretStr

        monkeypatch.setattr(_configured_settings, "schwab_trading_app_key", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "schwab_trading_app_secret", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "tradier_api_key", SecretStr(""))

        # yfinance mocked to succeed -- again, proving the guard fires
        # before this fallback would even be attempted.
        yfinance_mock = MagicMock(
            return_value=Result.success(
                OptionsChain(
                    ticker="SPXW",
                    underlying_price=5000.0,
                    expiries=["2026-07-10"],
                    contracts=[],
                    source=OptionsChainClient.SOURCE_YFINANCE,
                )
            )
        )
        monkeypatch.setattr(OptionsChainClient, "_fetch_from_yfinance", yfinance_mock)

        result = OptionsChainClient().fetch_chain("SPXW")

        assert result.ok is False
        assert result.data is None
        yfinance_mock.assert_not_called()

    @respx.mock
    def test_spy_is_unaffected_and_still_falls_back_to_tradier(self, monkeypatch, _configured_settings):
        """Regression guard: the SPX/SPXW-only guard must not change
        behaviour for ordinary equity underlyings -- SPY must still reach
        (and succeed via) the Tradier fallback exactly as before."""
        from pydantic import SecretStr

        monkeypatch.setattr(_configured_settings, "schwab_trading_app_key", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "schwab_trading_app_secret", SecretStr(""))
        monkeypatch.setattr(_configured_settings, "tradier_api_key", SecretStr("fake-tradier-key"))

        respx.get("https://api.tradier.com/v1/markets/options/expirations").mock(
            return_value=httpx.Response(200, json={"expirations": {"date": ["2026-08-21"]}})
        )
        respx.get("https://api.tradier.com/v1/markets/options/chains").mock(
            return_value=httpx.Response(
                200,
                json={
                    "options": {
                        "option": [
                            {
                                "symbol": "SPY260821C00450000",
                                "strike": 450.0,
                                "option_type": "call",
                                "bid": 5.0,
                                "ask": 5.2,
                                "last": 5.1,
                                "volume": 100,
                                "open_interest": 200,
                                "greeks": {"mid_iv": 0.19, "delta": 0.5},
                                "in_the_money": True,
                            }
                        ]
                    }
                },
            )
        )

        result = OptionsChainClient().fetch_chain("SPY")

        assert result.ok is True
        assert result.data is not None
        assert result.data.source == OptionsChainClient.SOURCE_TRADIER

    def test_spx_with_schwab_chain_available_still_succeeds(self, monkeypatch):
        """The guard only refuses Tradier/yfinance -- a genuine Schwab
        chain for SPX/SPXW must still be returned normally."""
        fake_client = MagicMock()
        fake_client.get_option_chain = MagicMock(
            return_value=FakeResponse(status_code=200, json_data=_SPX_CHAIN_RESPONSE)
        )
        _patch_schwab_client(monkeypatch, fake_client)

        result = OptionsChainClient().fetch_chain("SPX")

        assert result.ok is True
        assert result.error is None
        chain = result.data
        assert chain is not None
        assert chain.source == OptionsChainClient.SOURCE_SCHWAB
        assert len(chain.contracts) == 2


class TestResolveOptionRoot:
    """Unit tests for `_resolve_option_root`'s resolution order: Schwab's
    own `optionRoot` field, then the root parsed from the contract's own
    `symbol`, then (SPX only) `expirationType`/3rd-Friday date-math, then
    the plain ticker."""

    def test_option_root_field_wins_when_present(self):
        # Even a symbol that would parse to something else must lose to an
        # explicit optionRoot field -- it's Schwab's own strongest signal.
        contract = {"optionRoot": "SPXW", "symbol": "SPX_071726C5000", "expirationType": "S"}
        assert _resolve_option_root("SPX", contract, "2026-07-17") == "SPXW"

    def test_parses_root_from_symbol_when_no_option_root_field(self):
        contract = {"symbol": "SPXW_071026C5040"}
        assert _resolve_option_root("SPX", contract, "2026-07-10") == "SPXW"

    def test_parses_root_from_symbol_generalizes_to_other_underlyings(self):
        # No SPX-specific branch is involved here at all -- this is the
        # same generic symbol-parsing path used for SPY/QQQ, just applied
        # to a hypothetical other index underlying's own weekly root
        # spelling (NDX/NDXP), to confirm nothing about this path is
        # actually hardcoded to "SPX"/"SPXW" specifically.
        contract = {"symbol": "NDXP_071026C18000"}
        assert _resolve_option_root("NDX", contract, "2026-07-10") == "NDXP"

    def test_expiration_type_fallback_for_spx_weekly(self):
        contract = {"expirationType": "W"}  # no optionRoot, no parseable symbol
        assert _resolve_option_root("SPX", contract, "2026-07-10") == "SPXW"

    def test_expiration_type_fallback_for_spx_monthly(self):
        contract = {"expirationType": "S"}
        assert _resolve_option_root("SPX", contract, "2026-07-17") == "SPX"

    def test_date_math_fallback_for_spx_third_friday(self):
        # No optionRoot, no parseable symbol, no expirationType either --
        # 2026-07-17 is a real 3rd Friday.
        contract: dict = {}
        assert _resolve_option_root("SPX", contract, "2026-07-17") == "SPX"

    def test_date_math_fallback_for_spx_non_third_friday(self):
        contract: dict = {}
        assert _resolve_option_root("SPX", contract, "2026-07-10") == "SPXW"

    def test_defaults_to_ticker_for_non_spx_with_no_signals(self):
        contract: dict = {}
        assert _resolve_option_root("SPY", contract, "2026-07-17") == "SPY"
