# Ported from D:\chanakya\options_advisor\data_clients\options_chain.py
"""Options chain client: Schwab primary, Tradier fallback, yfinance tertiary.

Phase 4 ("Defined-risk options day trades (verticals) reusing chanakya
options_chain.py, Greeks, and the 26-check validator pattern" per
DAY_TRADER_STRATEGY.md) — this module ports the chanakya
``OptionsChainClient`` fetch chain (Schwab via schwab-py's
``client.get_option_chain(...)``, Tradier HTTP fallback, yfinance tertiary
fallback with no Greeks) onto this project's own infra:

  - ``Result[T]`` / ``FetchError`` from ``trading_common.data_clients.base``
    (this repo's already-extracted partial-failure pattern) instead of
    chanakya's ``options_advisor.data_clients.base``.
  - The process-local TTL cache in ``trading_common.data_clients._cache``.
  - Schwab auth goes through this project's own
    ``trading_common.services.schwab_token.SchwabTokenManager`` +
    ``config.settings.settings`` (``schwab_trading_app_key``/``schwab_trading_app_secret``/
    ``schwab_token_path``) — the same decrypt-into-temp-file /
    re-encrypt-on-exit pattern ``brokers/schwab.py``'s ``_get_client`` and
    ``replay/cli.py``'s ``_fetch_via_schwab`` already use — instead of
    chanakya's per-user ``token_json``/``token_output`` DB-persistence
    callback plumbing (there is no multi-tenant per-user token store in
    this codebase; there is exactly one Schwab account, one on-disk
    encrypted token file, matching ``SchwabBroker``).

Dropped from the chanakya original (options-advisor user/portfolio
concepts that don't exist in this codebase):
  - GEX-by-strike / put-call-ratio / IV-term-structure "derived features"
    attached to the chain post-fetch, and the ``ChainSnapshot``/
    ``chain_to_snapshots`` Parquet-archival helpers built for options-
    advisor's IV-rank history feature. None of that is needed for a
    day-trading vertical-spread entry/exit pipeline; it can be re-added
    later if a strategy needs it.
  - Per-user ``tradier_key``/``schwab_trading_app_key``/``schwab_trading_app_secret``/
    ``token_json`` overrides on ``fetch_chain`` (options-advisor is
    multi-tenant; this bot is single-account) — keys are always read from
    ``config.settings.settings``.
  - ``fetch_iv_history`` (Tradier-history-based realized-vol IV proxy) —
    out of scope for this slice; ``trading_common.features.volatility``
    already covers expected-move calculations per the strategy doc.
"""
from __future__ import annotations

import logging
import re
import time as _time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import yfinance as yf

from trading_common.data_clients._cache import get as _cache_get
from trading_common.data_clients._cache import set as _cache_set
from trading_common.data_clients.base import FetchError, FetchErrorType, Result, map_price_ticker
from trading_common.monitoring.tracker import get_tracker
from trading_common.services.schwab_token import SchwabTokenManager

_log = logging.getLogger(__name__)

_TRADIER_BASE_URL = "https://api.tradier.com/v1/markets/options"

# SPX-style index underlyings whose Tradier and yfinance fallback fetches
# (`_fetch_from_tradier` / `_fetch_from_yfinance`) cannot resolve a correct
# per-contract `option_root` -- both paths always default `option_root` to
# the plain ticker (see the comments at their `OptionContract(...)` calls),
# which is wrong for roughly half of any SPX/SPXW chain (the monthly/weekly
# OCC-root split; see `OptionContract.option_root`'s and
# `_resolve_option_root`'s docstrings). Only Schwab resolves this correctly.
# An order built from a wrong `option_root` would target the wrong contract
# at the broker -- a structural correctness bug, not a data-quality
# preference -- so `fetch_chain` below refuses to let these two underlyings
# fall through to Tradier/yfinance at all; see M4 (SPX OPTIONS AUDIT).
_SPX_STYLE_UNDERLYINGS = frozenset({"SPX", "SPXW"})


def _http_get_with_retry(
    url: str,
    *,
    params: Any = None,
    headers: Any = None,
    timeout: float = 15,
    max_retries: int = 3,
) -> httpx.Response:
    """GET with exponential backoff on 429 and 5xx. Raises on final failure."""
    delay = 1.0
    r: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < max_retries:
                    _log.warning(
                        "_http_get_with_retry: status %s, retrying in %.1fs (attempt %d/%d)",
                        r.status_code, delay, attempt + 1, max_retries,
                    )
                    _time.sleep(delay)
                    delay *= 2
                    continue
            return r
        except httpx.TimeoutException:
            if attempt < max_retries:
                _log.warning(
                    "_http_get_with_retry: timeout, retrying in %.1fs (attempt %d/%d)",
                    delay, attempt + 1, max_retries,
                )
                _time.sleep(delay)
                delay *= 2
                continue
            raise
    # r is always assigned (loop runs >= 1 time for max_retries >= 0); mypy
    # can't prove that statically since r starts as Response | None.
    return r  # type: ignore[return-value]


@dataclass
class OptionContract:
    """A single option contract with standardised fields."""

    symbol: str
    strike: float
    expiry: str
    option_type: str  # "call" | "put"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    rho: float | None
    in_the_money: bool
    # Mark / mid price. Critical after-hours and for illiquid strikes where
    # bid/ask collapse to 0 — Schwab carries a usable `mark` (and
    # `closePrice`) even when bid/ask are 0.
    mark: float = 0.0
    # OCC-format root symbol for THIS contract -- the string that must be
    # passed as `OptionSymbol(underlying_symbol=...)` (schwab-py) to build a
    # valid order symbol. For ordinary equity underlyings this always
    # equals `ticker` (e.g. "SPY"), but index underlyings that split their
    # chain across more than one OCC root per contract type -- confirmed
    # for SPX, where true monthly (3rd-Friday, AM-settled) contracts use
    # root "SPX" but weekly/daily (PM-settled) contracts sharing the same
    # underlying index use root "SPXW" (see schwab-py's own
    # `schwab.orders.options.OptionSymbol` docstring) -- need this resolved
    # PER CONTRACT, since one chain fetch for "SPX" returns both kinds
    # together. Defaults to "" (meaning "not resolved / not applicable") so
    # existing callers that construct an `OptionContract` directly (tests,
    # the Tradier/yfinance fetch paths) are unaffected; `_normalise_schwab`
    # always fills this in via `_resolve_option_root`.
    option_root: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def best_mid(c: OptionContract) -> float:
    """Return the most reliable per-contract price.

    Preference: bid/ask midpoint when BOTH sides are quoting (>0), else the
    mark, else the last trade. After hours bid/ask are 0, so a naive
    (bid+ask)/2 yields 0 — this falls through to mark/last instead.
    """
    bid = c.bid or 0.0
    ask = c.ask or 0.0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if (c.mark or 0.0) > 0:
        return c.mark
    return c.last or 0.0


@dataclass
class OptionsChain:
    """Normalised options chain for a ticker across one or more expiries."""

    ticker: str
    underlying_price: float
    expiries: list[str]
    contracts: list[OptionContract] = field(default_factory=list)
    source: str = ""


@dataclass
class ChainSnapshot:
    """Normalised, point-in-time snapshot of a single option contract.

    Built from an OptionContract (Schwab- or yfinance-sourced — both paths
    already funnel through that common shape) via ``chain_to_snapshots()``.
    """

    strike: float
    expiry: str
    type: str  # "call" | "put"
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    iv: float
    underlying_price: float
    timestamp: str
    spread_pct: float | None = None
    is_liquid: bool = False


class OptionsChainClient:
    """Fetches options chain data: Schwab primary, Tradier fallback,
    yfinance tertiary fallback (free, ~15-min delayed, no Greeks).

    Exception: SPX/SPXW never use the Tradier/yfinance fallbacks (see
    ``_SPX_STYLE_UNDERLYINGS`` and ``fetch_chain``'s docstring) -- only a
    Schwab-sourced chain is acceptable for those two underlyings."""

    SOURCE_SCHWAB = "schwab"
    SOURCE_TRADIER = "tradier"
    SOURCE_YFINANCE = "yfinance"

    def fetch_chain(
        self,
        ticker: str,
        expiries: list[str] | None = None,
    ) -> Result[OptionsChain]:
        """Fetch the options chain for ticker, optionally filtered to expiries.

        Priority: Schwab → Tradier → yfinance (free, 15-min delayed).
        Returns Result.failure() only when all providers are unavailable.

        SPX/SPXW are the exception to that fallback chain: Tradier and
        yfinance can't resolve a correct per-contract ``option_root`` for
        these index underlyings (see ``_SPX_STYLE_UNDERLYINGS``), so this
        never falls through to them for SPX/SPXW -- only a Schwab-sourced
        chain is acceptable. When Schwab isn't configured or its fetch
        fails for one of these tickers, this returns ``Result.failure()``
        (fail closed: no chain, no trade) instead of a wrong-root chain.

        Keys are always read from ``trading_common.config.settings.settings``
        (single-account bot — no per-user key overrides).

        Results are cached for 300 seconds (5 minutes).
        """
        expiry_filter_key = ",".join(sorted(expiries)) if expiries else "all"
        cache_key = f"options_chain:{ticker.upper()}:{expiry_filter_key}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        from trading_common.config.settings import settings

        schwab_key = settings.schwab_trading_app_key.get_secret_value()
        schwab_secret = settings.schwab_trading_app_secret.get_secret_value()
        tradier_key = settings.tradier_api_key.get_secret_value()
        is_spx_style = ticker.upper() in _SPX_STYLE_UNDERLYINGS

        # Schwab (real-time) — skip immediately when not configured
        if schwab_key and schwab_secret:
            result = self._fetch_from_schwab(ticker, expiries, schwab_key, schwab_secret)
            if result.ok:
                _cache_set(cache_key, result, 300)
                return result

        if is_spx_style:
            # M4 (SPX OPTIONS AUDIT): refuse to fall back to Tradier/
            # yfinance for SPX/SPXW -- see `_SPX_STYLE_UNDERLYINGS`'s
            # docstring. Fail closed rather than silently handing back a
            # chain with a structurally wrong `option_root`.
            return Result.failure(
                FetchError(
                    self.SOURCE_SCHWAB,
                    FetchErrorType.UNAVAILABLE,
                    f"{ticker}: SPX/SPXW requires a Schwab-sourced options "
                    "chain -- Tradier and yfinance report the wrong "
                    "option_root for SPX-style index underlyings, so this "
                    "refuses to fall back to them (no chain, no trade)",
                )
            )

        # Tradier (15-min delayed on free tier, real-time on paid)
        if tradier_key:
            result = self._fetch_from_tradier(ticker, expiries, tradier_key)
            if result.ok:
                _cache_set(cache_key, result, 300)
                return result

        # yfinance fallback — always available, 15-min delayed, no Greeks
        result = self._fetch_from_yfinance(ticker, expiries)
        if result.ok:
            _cache_set(cache_key, result, 300)
        return result

    def _fetch_from_schwab(
        self,
        ticker: str,
        expiries: list[str] | None,
        schwab_key: str,
        schwab_secret: str,
    ) -> Result[OptionsChain]:
        """Attempt to fetch the options chain via the Schwab API.

        Auth follows the same self-authenticating pattern used elsewhere in
        this codebase (e.g. a consuming app's own broker client): the
        on-disk encrypted token (``trading_common.config.settings.settings.
        schwab_token_path``) is decrypted into a short-lived temp file via
        ``SchwabTokenManager.plaintext_context``, used to build a
        schwab-py client, then re-encrypted (capturing any token refresh
        schwab-py performed) on context exit.
        """
        _t0 = _time.perf_counter()
        try:
            import schwab.auth as schwab_auth

            from trading_common.config.settings import settings

            token_path = Path(settings.schwab_token_path)
            if not token_path.exists():
                return Result.failure(
                    FetchError(
                        self.SOURCE_SCHWAB,
                        FetchErrorType.AUTH,
                        f"No Schwab token found at {token_path} — run the token bootstrap script",
                    )
                )

            today = date.today()
            with SchwabTokenManager().plaintext_context(token_path) as tmp_path:
                client = schwab_auth.client_from_token_file(tmp_path, schwab_key, schwab_secret)
                response = client.get_option_chain(
                    ticker,
                    strike_count=40,
                    from_date=today,
                    to_date=today + timedelta(days=90),
                )
                if response.status_code == 401:
                    return Result.failure(
                        FetchError(
                            self.SOURCE_SCHWAB,
                            FetchErrorType.AUTH,
                            "Schwab token rejected (401) — re-authenticate via the token bootstrap script",
                        )
                    )
                if response.status_code == 429:
                    return Result.failure(
                        FetchError(
                            self.SOURCE_SCHWAB,
                            FetchErrorType.RATE_LIMIT,
                            "Schwab API rate limit exceeded (429) — retry in 60 seconds",
                        )
                    )
                response.raise_for_status()
                raw = response.json()

            chain = self._normalise_schwab(raw, ticker)

            if expiries:
                expiry_set = set(expiries)
                chain.contracts = [c for c in chain.contracts if c.expiry in expiry_set]
                chain.expiries = [e for e in chain.expiries if e in expiry_set]

            get_tracker("schwab").record(True, (_time.perf_counter() - _t0) * 1000)
            return Result.success(chain)
        except httpx.TimeoutException as e:
            get_tracker("schwab").record(False, (_time.perf_counter() - _t0) * 1000, error=str(e))
            return Result.failure(FetchError(self.SOURCE_SCHWAB, FetchErrorType.TIMEOUT, str(e)))
        except Exception as e:
            get_tracker("schwab").record(False, (_time.perf_counter() - _t0) * 1000, error=str(e))
            return Result.failure(FetchError(self.SOURCE_SCHWAB, FetchErrorType.UNAVAILABLE, str(e)))

    def _fetch_from_tradier(
        self,
        ticker: str,
        expiries: list[str] | None,
        tradier_key: str,
    ) -> Result[OptionsChain]:
        """Attempt to fetch the options chain via the Tradier API."""
        _t0 = _time.perf_counter()
        try:
            headers = {
                "Authorization": f"Bearer {tradier_key}",
                "Accept": "application/json",
            }

            if not expiries:
                exp_resp = _http_get_with_retry(
                    f"{_TRADIER_BASE_URL}/expirations",
                    params={"symbol": ticker},
                    headers=headers,
                    timeout=15.0,
                )
                exp_resp.raise_for_status()
                exp_data = exp_resp.json()
                expiry_dates: list[str] = exp_data.get("expirations", {}).get("date", []) or []
                # Use the nearest 4 expiries to keep response size manageable
                expiries_to_fetch = expiry_dates[:4]
            else:
                expiries_to_fetch = expiries

            if not expiries_to_fetch:
                return Result.failure(
                    FetchError(
                        self.SOURCE_TRADIER,
                        FetchErrorType.UNAVAILABLE,
                        f"No expiration dates available for {ticker}",
                    )
                )

            all_contracts: list[OptionContract] = []
            seen_expiries: list[str] = []

            for exp_date in expiries_to_fetch:
                resp = _http_get_with_retry(
                    f"{_TRADIER_BASE_URL}/chains",
                    params={"symbol": ticker, "expiration": exp_date, "greeks": "true"},
                    headers=headers,
                    timeout=15.0,
                )
                if resp.status_code == 429:
                    raise FetchError(self.SOURCE_TRADIER, FetchErrorType.RATE_LIMIT, "Tradier rate limit exceeded")
                resp.raise_for_status()
                raw = resp.json()

                contracts_raw = (raw.get("options") or {}).get("option") or []
                if not isinstance(contracts_raw, list):
                    contracts_raw = [contracts_raw]

                got_any = False
                for c in contracts_raw:
                    if not isinstance(c, dict):
                        continue
                    greeks_raw = c.get("greeks") or {}
                    all_contracts.append(OptionContract(
                        symbol=c.get("symbol", ""),
                        strike=float(c.get("strike", 0.0)),
                        expiry=exp_date,
                        option_type=c.get("option_type", "").lower(),
                        bid=float(c.get("bid", 0.0) or 0.0),
                        ask=float(c.get("ask", 0.0) or 0.0),
                        last=float(c.get("last", 0.0) or 0.0),
                        volume=int(c.get("volume", 0) or 0),
                        open_interest=int(c.get("open_interest", 0) or 0),
                        implied_volatility=float(greeks_raw.get("mid_iv", 0.0) or 0.0),
                        delta=_float_or_none(greeks_raw.get("delta")),
                        gamma=_float_or_none(greeks_raw.get("gamma")),
                        theta=_float_or_none(greeks_raw.get("theta")),
                        vega=_float_or_none(greeks_raw.get("vega")),
                        rho=_float_or_none(greeks_raw.get("rho")),
                        in_the_money=bool(c.get("in_the_money", False)),
                        # Tradier's OCC-ish `symbol` field already embeds
                        # whichever root Tradier itself resolved (e.g.
                        # "SPXW" for SPX weeklies) as its first several
                        # characters, but this fallback path has no
                        # confirmed field/format to parse that out of
                        # reliably (unlike the Schwab path -- see
                        # `_resolve_option_root`), so it defaults to the
                        # plain ticker. Threading Tradier's true per-
                        # contract root through is a natural follow-up if
                        # this fallback path is ever exercised for SPX.
                        option_root=ticker.upper(),
                    ))
                    got_any = True
                if got_any:
                    seen_expiries.append(exp_date)

            try:
                underlying_price = float(yf.Ticker(map_price_ticker(ticker)).fast_info.last_price or 0.0)
            except Exception:
                underlying_price = 0.0

            chain = OptionsChain(
                ticker=ticker,
                underlying_price=underlying_price,
                expiries=seen_expiries,
                contracts=all_contracts,
                source=self.SOURCE_TRADIER,
            )
            get_tracker("tradier").record(True, (_time.perf_counter() - _t0) * 1000)
            return Result.success(chain)

        except FetchError as e:
            get_tracker("tradier").record(False, (_time.perf_counter() - _t0) * 1000, error=e.message)
            return Result.failure(e)
        except httpx.TimeoutException as e:
            get_tracker("tradier").record(False, (_time.perf_counter() - _t0) * 1000, error=str(e))
            return Result.failure(FetchError(self.SOURCE_TRADIER, FetchErrorType.TIMEOUT, str(e)))
        except Exception as e:
            get_tracker("tradier").record(False, (_time.perf_counter() - _t0) * 1000, error=str(e))
            return Result.failure(FetchError(self.SOURCE_TRADIER, FetchErrorType.UNAVAILABLE, str(e)))

    def _fetch_from_yfinance(
        self,
        ticker: str,
        expiries: list[str] | None,
    ) -> Result[OptionsChain]:
        """Fetch options chain via yfinance — free, ~15-min delayed, no Greeks."""
        _t0 = _time.perf_counter()
        try:
            t = yf.Ticker(ticker)
            available = list(t.options or [])
            if not available:
                return Result.failure(
                    FetchError(self.SOURCE_YFINANCE, FetchErrorType.UNAVAILABLE, f"No options expiries available for {ticker}")
                )

            if expiries:
                to_fetch = [e for e in expiries if e in available] or available[:4]
            else:
                to_fetch = available[:4]

            try:
                pt = yf.Ticker(map_price_ticker(ticker))
                underlying_price = float(getattr(pt.fast_info, "last_price", None) or 0.0)
                if underlying_price <= 0:
                    underlying_price = float(getattr(pt.fast_info, "regular_market_price", None) or 0.0)
            except Exception:
                underlying_price = 0.0

            all_contracts: list[OptionContract] = []
            seen_expiries: list[str] = []

            for exp_date in to_fetch:
                try:
                    opt_chain = t.option_chain(exp_date)
                except Exception:
                    continue

                for opt_type, df in (("call", opt_chain.calls), ("put", opt_chain.puts)):
                    for _, row in df.iterrows():
                        all_contracts.append(
                            OptionContract(
                                symbol=str(row.get("contractSymbol", "")),
                                strike=_safe_float(row.get("strike")),
                                expiry=exp_date,
                                option_type=opt_type,
                                bid=_safe_float(row.get("bid")),
                                ask=_safe_float(row.get("ask")),
                                last=_safe_float(row.get("lastPrice")),
                                volume=_safe_int(row.get("volume")),
                                open_interest=_safe_int(row.get("openInterest")),
                                implied_volatility=_safe_float(row.get("impliedVolatility")),
                                delta=None,
                                gamma=None,
                                theta=None,
                                vega=None,
                                rho=None,
                                in_the_money=bool(row.get("inTheMoney", False)),
                                # See the matching comment in
                                # `_fetch_from_tradier` -- no confirmed
                                # per-contract root field/format for this
                                # fallback source either, so default to the
                                # plain ticker.
                                option_root=ticker.upper(),
                            )
                        )
                seen_expiries.append(exp_date)

            if not all_contracts:
                _err = f"Empty chain for {ticker}"
                get_tracker("yfinance").record(False, (_time.perf_counter() - _t0) * 1000, error=_err)
                return Result.failure(FetchError(self.SOURCE_YFINANCE, FetchErrorType.UNAVAILABLE, _err))

            get_tracker("yfinance").record(True, (_time.perf_counter() - _t0) * 1000)
            return Result.success(
                OptionsChain(
                    ticker=ticker,
                    underlying_price=underlying_price,
                    expiries=seen_expiries,
                    contracts=all_contracts,
                    source=self.SOURCE_YFINANCE,
                )
            )
        except Exception as e:
            get_tracker("yfinance").record(False, (_time.perf_counter() - _t0) * 1000, error=str(e))
            return Result.failure(FetchError(self.SOURCE_YFINANCE, FetchErrorType.UNAVAILABLE, str(e)))

    def _normalise_schwab(self, raw: Any, ticker: str) -> OptionsChain:
        """Convert Schwab API response into a normalised OptionsChain."""
        underlying_price = float(raw.get("underlyingPrice", 0.0) or 0.0)
        contracts: list[OptionContract] = []
        expiry_set: set[str] = set()

        for option_type_label, map_key in (("call", "callExpDateMap"), ("put", "putExpDateMap")):
            exp_map: dict = raw.get(map_key, {}) or {}
            for exp_key, strikes_dict in exp_map.items():
                # exp_key looks like "2026-01-17:30" — take just the date part
                expiry_date = exp_key.split(":")[0]
                expiry_set.add(expiry_date)
                for strike_str, contract_list in (strikes_dict or {}).items():
                    if not isinstance(contract_list, list):
                        contract_list = [contract_list]
                    for c in contract_list:
                        if not isinstance(c, dict):
                            continue
                        _bid = float(c.get("bid", 0.0) or 0.0)
                        _ask = float(c.get("ask", 0.0) or 0.0)
                        _mark = float(c.get("mark", 0.0) or 0.0)
                        _last = float(c.get("last") or c.get("mark") or c.get("closePrice") or 0.0)
                        contract = OptionContract(
                            symbol=c.get("symbol", ""),
                            strike=float(strike_str),
                            expiry=expiry_date,
                            option_type=option_type_label,
                            bid=_bid,
                            ask=_ask,
                            last=_last,
                            volume=int(c.get("totalVolume", 0) or 0),
                            open_interest=int(c.get("openInterest", 0) or 0),
                            implied_volatility=float(c.get("volatility", 0.0) or 0.0) / 100.0,
                            delta=_float_or_none(c.get("delta")),
                            gamma=_float_or_none(c.get("gamma")),
                            theta=_float_or_none(c.get("theta")),
                            vega=_float_or_none(c.get("vega")),
                            rho=_float_or_none(c.get("rho")),
                            in_the_money=bool(c.get("inTheMoney", False)),
                            mark=_mark if _mark > 0 else ((_bid + _ask) / 2.0 if _bid > 0 and _ask > 0 else 0.0),
                            option_root=_resolve_option_root(ticker, c, expiry_date),
                        )
                        contracts.append(contract)

        return OptionsChain(
            ticker=ticker,
            underlying_price=underlying_price,
            expiries=sorted(expiry_set),
            contracts=contracts,
            source=self.SOURCE_SCHWAB,
        )


# OCC root parsed from the leading letters of Schwab's own per-contract
# `symbol` field, which this codebase's Schwab fixtures (and Schwab's real
# chain response) format as `{root}_{MMDDYY}{C|P}{strike}` (e.g.
# "SPY_082126C450", "SPXW_042024C5040") -- also tolerates a bare space
# instead of an underscore, or no separator at all, since none of that
# affects where the root's letters end and the digits begin.
_ROOT_FROM_SYMBOL_RE = re.compile(r"^([A-Za-z]+)[_ ]?\d{6}[CP]")


def _is_third_friday(d: date) -> bool:
    """True when `d` is the 3rd Friday of its month -- the standard monthly
    equity/index option expiration date. Used only as the last-resort
    fallback in `_resolve_option_root` (see its docstring): every other
    signal (Schwab's own `optionRoot` field, or the root parsed from the
    contract's own `symbol`) is preferred over this date-math heuristic."""
    if d.weekday() != 4:  # Friday
        return False
    return 15 <= d.day <= 21


def _resolve_option_root(ticker: str, raw_contract: dict[str, Any], expiry_date: str) -> str:
    """Resolve the correct per-contract OCC root symbol from one Schwab
    option-chain contract dict.

    Most underlyings (SPY, QQQ, ...) have exactly one OCC root, always
    equal to the ticker itself. SPX-style index options are the documented
    exception this function exists for: confirmed directly from the
    installed schwab-py source (`schwab.orders.options.OptionSymbol`'s own
    docstring example, `"SPXW  240420C05040000"` = "SPX Weekly Apr 20, 2024
    5040 Call"), SPX splits its chain across TWO OCC roots depending on
    contract type -- "SPX" for true monthly (3rd-Friday, AM-settled)
    contracts, "SPXW" for weekly/daily (PM-settled) contracts sharing the
    same underlying index. A single chain fetch for underlying "SPX"
    returns both kinds of contract together, so this must resolve PER
    CONTRACT, not once per ticker/chain.

    Deliberately designed as "does this contract need root resolution"
    rather than an SPX-only special case -- steps 1-2 below use only data
    Schwab itself already sends on the contract, so they resolve correctly
    for any other index underlying that splits its OCC root the same way
    (e.g. NDX/NDXP, RUT/RUTW) without this code needing to hardcode that
    underlying's specific weekly-root spelling. Only the last-resort
    fallback (step 3) is SPX-specific, and only ever runs if Schwab's
    response is missing BOTH of the stronger signals below (not expected
    in production -- kept so a future change to Schwab's response shape
    degrades to a still-correct-for-SPX answer rather than a silently
    wrong one).

    Resolution order (first that yields an answer wins):
      1. Schwab's own `optionRoot` field on the contract, when present and
         non-empty -- the strongest signal, since it's Schwab's own
         resolution rather than one derived here.
      2. Parsed directly from the contract's own `symbol` field (see
         `_ROOT_FROM_SYMBOL_RE`) -- also Schwab's own data, already
         correctly resolved per-contract.
      3. SPX-only fallback: Schwab's `expirationType` field ("W" => weekly
         => "SPXW"; "M"/"S"/"Q" => monthly-family => "SPX"), or -- if even
         that field is absent -- a 3rd-Friday date-math heuristic on
         `expiry_date` (3rd Friday => "SPX", else => "SPXW").
      4. The plain ticker itself -- correct for ordinary equity
         underlyings, and the final fallback for any other underlying this
         function hasn't been taught a weekly-root convention for.
    """
    raw_root = raw_contract.get("optionRoot")
    if isinstance(raw_root, str) and raw_root.strip():
        return raw_root.strip()

    symbol = raw_contract.get("symbol")
    if isinstance(symbol, str):
        match = _ROOT_FROM_SYMBOL_RE.match(symbol)
        if match:
            return match.group(1)

    if ticker.upper() == "SPX":
        expiration_type = str(raw_contract.get("expirationType") or "").upper()
        if expiration_type == "W":
            return "SPXW"
        if expiration_type in ("M", "S", "Q"):
            return "SPX"
        try:
            parsed_expiry = date.fromisoformat(expiry_date)
        except ValueError:
            return "SPXW"  # can't date-math it either -- weekly is the safer default (no OCC collision risk)
        return "SPX" if _is_third_friday(parsed_expiry) else "SPXW"

    return ticker


def _float_or_none(val: Any) -> float | None:
    """Return float(val) or None when val is missing/non-numeric/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


def _safe_float(val: Any) -> float:
    """Return float(val), defaulting to 0.0 for None/NaN/non-numeric."""
    try:
        f = float(val)
        return f if f == f else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_int(val: Any) -> int:
    """Return int(val), defaulting to 0 for None/NaN/non-numeric."""
    try:
        f = float(val)
        return int(f) if f == f else 0
    except (TypeError, ValueError):
        return 0
