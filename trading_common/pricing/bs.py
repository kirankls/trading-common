# Ported from D:\chanakya\options_advisor\pricing\bs.py
"""Vectorized Black-Scholes-Merton pricing, Greeks, and implied vol.

Thin wrapper over ``py_vollib_vectorized``. All functions accept scalars or
numpy arrays (broadcast per numpy rules) and return the same shape back
(scalar in -> python float out; array in -> numpy array out).

``option_type`` convention: lowercase ``"call"`` / ``"put"`` strings,
matching the rest of the codebase (``OptionContract.option_type``).

--------------------------------------------------------------------------
Why this module sets ``NUMBA_DISABLE_JIT=1``
--------------------------------------------------------------------------
The installed ``py_vollib_vectorized==0.1.1`` numba-JITs a self-recursive
helper (``py_vollib_vectorized._model_calls.black``, which calls itself
with a negated argument to map ITM<->OTM). Under the numba version pinned
transitively in this environment (numba 0.66 / numpy 2.4), numba's
nopython type inference cannot resolve the recursive call
("cannot type infer runaway recursion") and every vectorized price/greek
call raises ``numba.core.errors.TypingError`` — 100% reproducible, not an
input-shape issue (verified with numpy arrays, python lists, single- and
multi-element inputs, with and without dividend yield).

The library exposes a module-level ``use_jit`` flag
(``py_vollib_vectorized.util.jit_helper.use_jit``) intended to disable
JIT, but flipping it after import is a no-op: Python always executes a
package's ``__init__.py`` before any of its submodules, and that
``__init__.py`` eagerly imports and JIT-decorates every internal function
at import time — by the time ``jit_helper`` is reachable to patch, the
damage is done. The only reliable switch is the environment variable
``NUMBA_DISABLE_JIT=1``, which numba itself reads at import time and turns
every ``@jit`` decorator into a no-op (plain Python). This has been
verified to fix all price/greeks/IV calls and to match an independent
scalar oracle (``py_vollib.black_scholes_merton`` directly) to machine
precision across randomized (S, K, T, r, sigma, q, call/put) combinations.

Performance note: without numba JIT, these calls run in pure
Python/numpy — still vectorized (no Python-level loop over strikes), just
without the JIT speedup. This is acceptable; if profiling later shows it's
a bottleneck, revisit (e.g. pin a numba version known compatible with this
py_vollib_vectorized release, or replace with a hand-vectorized numpy BS
implementation, which is only ~20 lines).

We set the env var here, at import time of this module, *before*
importing py_vollib_vectorized, and only if it isn't already set (so a
caller/deployment can override with NUMBA_DISABLE_JIT=0 if they've pinned
a compatible numba and want the speed).
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import warnings
from typing import Any

import numpy as np

with warnings.catch_warnings():
    # py_vollib emits a harmless DeprecationWarning on import ("use vollib
    # instead") — py_vollib_vectorized depends on the py_vollib.* API and
    # still works correctly; this is expected noise, not a real issue.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import py_vollib_vectorized as _pvv  # noqa: E402

__all__ = ["price", "greeks", "implied_vol"]

_FLAG_MAP = {"call": "c", "put": "p"}


def _as_array(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64))


def _flags_array(option_type: str | np.ndarray, n: int) -> np.ndarray:
    """Build an array of 'c'/'p' flags of length n from a scalar or array
    of "call"/"put" strings."""
    if isinstance(option_type, str):
        try:
            flag = _FLAG_MAP[option_type.lower()]
        except KeyError:
            raise ValueError(
                f"option_type must be 'call' or 'put', got {option_type!r}"
            ) from None
        return np.full(n, flag)
    arr = np.asarray(option_type)
    flat = arr.reshape(-1)
    try:
        mapped = [_FLAG_MAP[str(v).lower()] for v in flat]
    except KeyError as exc:
        raise ValueError(f"option_type must be 'call' or 'put', got {exc}") from None
    out = np.array(mapped)
    if out.size == 1 and n > 1:
        out = np.full(n, out[0])
    return out


def _broadcast_shape(*arrays: np.ndarray) -> tuple[int, ...]:
    return np.broadcast_shapes(*(a.shape for a in arrays))


def _maybe_scalar(arr: np.ndarray, was_scalar: bool) -> np.ndarray | float:
    if was_scalar:
        return float(arr.reshape(-1)[0])
    return arr


def _inputs_are_scalar(*raw_inputs: Any) -> bool:
    return all(np.ndim(x) == 0 for x in raw_inputs)


def price(
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    sigma: Any,
    option_type: str | np.ndarray,
    q: Any = 0.0,
) -> np.ndarray | float:
    """Black-Scholes-Merton price (continuous dividend yield q).

    S, K, T, r, sigma, q: scalar or numpy array (broadcastable).
    option_type: "call"/"put" scalar string, or an array of them matching
        the broadcast shape (for pricing a mixed call/put strike ladder in
        one vectorized call).
    Returns a python float if all numeric inputs were scalar, else a numpy
    array.
    """
    was_scalar = _inputs_are_scalar(S, K, T, r, sigma, q) and isinstance(option_type, str)

    Sa, Ka, Ta, ra, siga, qa = (_as_array(x) for x in (S, K, T, r, sigma, q))
    shape = _broadcast_shape(Sa, Ka, Ta, ra, siga, qa)
    n = int(np.prod(shape)) if shape else 1

    Sa, Ka, Ta, ra, siga, qa = (
        np.broadcast_to(a, shape).reshape(-1).astype(np.float64)
        for a in (Sa, Ka, Ta, ra, siga, qa)
    )
    flags = _flags_array(option_type, n)

    # Intrinsic value for T<=0 (py_vollib_vectorized expects T>0); handle
    # expired/at-expiry legs ourselves rather than feeding it degenerate T.
    out = np.empty(n, dtype=np.float64)
    live = Ta > 0
    if np.any(live):
        out[live] = _pvv.vectorized_black_scholes_merton(
            flags[live], Sa[live], Ka[live], Ta[live], ra[live], siga[live], qa[live],
            return_as="numpy",
        )
    if np.any(~live):
        is_call = flags[~live] == "c"
        intrinsic = np.where(is_call, np.maximum(Sa[~live] - Ka[~live], 0.0),
                              np.maximum(Ka[~live] - Sa[~live], 0.0))
        out[~live] = intrinsic

    out = out.reshape(shape) if shape else out  # type: ignore[assignment]
    return _maybe_scalar(np.atleast_1d(out), was_scalar)


def _rho_bsm(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray,
             sigma: np.ndarray, q: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Closed-form Black-Scholes-Merton rho, computed directly.

    NOTE: ``py_vollib_vectorized.vectorized_rho(..., model='black_scholes_merton')``
    was found to return incorrect values (wrong magnitude and sign) when a
    non-zero dividend yield q is supplied — verified against the scalar
    ``py_vollib.black_scholes_merton.greeks.analytical.rho`` oracle across
    randomized inputs (vectorized gave e.g. -0.0315 where the correct,
    scalar-oracle-matching value is +0.2507 for a representative ATM call).
    Delta/gamma/theta/vega from the vectorized merton path all matched the
    scalar oracle to ~1e-4, so only rho needed a local replacement. This is
    the standard closed-form rho for BSM, scaled per 1% rate move (divide
    by 100) to match py_vollib's convention for the other Greeks.
    """
    from scipy.stats import norm

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    is_call = flags == "c"
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
    return np.where(is_call, call_rho, put_rho)


def greeks(
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    sigma: Any,
    option_type: str | np.ndarray,
    q: Any = 0.0,
) -> dict[str, np.ndarray | float]:
    """Black-Scholes-Merton Greeks: delta, gamma, theta (per day), vega
    (per 1 vol point, i.e. per 0.01 sigma), rho (per 1% rate move).

    Returns a dict of the same scalar-or-array convention as ``price``.
    Keys: delta, gamma, theta, vega, rho.
    """
    was_scalar = _inputs_are_scalar(S, K, T, r, sigma, q) and isinstance(option_type, str)

    Sa, Ka, Ta, ra, siga, qa = (_as_array(x) for x in (S, K, T, r, sigma, q))
    shape = _broadcast_shape(Sa, Ka, Ta, ra, siga, qa)
    n = int(np.prod(shape)) if shape else 1

    Sa, Ka, Ta, ra, siga, qa = (
        np.broadcast_to(a, shape).reshape(-1).astype(np.float64)
        for a in (Sa, Ka, Ta, ra, siga, qa)
    )
    flags = _flags_array(option_type, n)

    out = {name: np.zeros(n, dtype=np.float64) for name in
           ("delta", "gamma", "theta", "vega", "rho")}

    live = Ta > 0
    if np.any(live):
        g = _pvv.get_all_greeks(
            flags[live], Sa[live], Ka[live], Ta[live], ra[live], siga[live], qa[live],
            model="black_scholes_merton", return_as="numpy",
        )
        out["delta"][live] = np.asarray(g["delta"], dtype=np.float64)
        out["gamma"][live] = np.asarray(g["gamma"], dtype=np.float64)
        out["theta"][live] = np.asarray(g["theta"], dtype=np.float64)
        out["vega"][live] = np.asarray(g["vega"], dtype=np.float64)
        out["rho"][live] = _rho_bsm(Sa[live], Ka[live], Ta[live], ra[live],
                                     siga[live], qa[live], flags[live])
    if np.any(~live):
        # At/after expiry: delta is 0/±1 (ignoring the knife-edge at K),
        # all other Greeks are 0.
        is_call = flags[~live] == "c"
        itm = np.where(is_call, Sa[~live] > Ka[~live], Sa[~live] < Ka[~live])
        out["delta"][~live] = np.where(itm, np.where(is_call, 1.0, -1.0), 0.0)

    result: dict[str, np.ndarray | float] = {}
    for key, arr in out.items():
        arr = arr.reshape(shape) if shape else arr  # type: ignore[assignment]
        result[key] = _maybe_scalar(np.atleast_1d(arr), was_scalar)
    return result


def implied_vol(
    price_: Any,
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    option_type: str | np.ndarray,
    q: Any = 0.0,
) -> np.ndarray | float:
    """Invert the BSM price to recover implied volatility.

    Returns NaN for inputs where inversion fails (e.g. price outside
    no-arbitrage bounds) rather than raising — callers should filter NaNs.
    """
    was_scalar = _inputs_are_scalar(price_, S, K, T, r, q) and isinstance(option_type, str)

    Pa, Sa, Ka, Ta, ra, qa = (_as_array(x) for x in (price_, S, K, T, r, q))
    shape = _broadcast_shape(Pa, Sa, Ka, Ta, ra, qa)
    n = int(np.prod(shape)) if shape else 1

    Pa, Sa, Ka, Ta, ra, qa = (
        np.broadcast_to(a, shape).reshape(-1).astype(np.float64)
        for a in (Pa, Sa, Ka, Ta, ra, qa)
    )
    flags = _flags_array(option_type, n)

    out = np.full(n, np.nan, dtype=np.float64)
    live = Ta > 0
    if np.any(live):
        try:
            iv = _pvv.vectorized_implied_volatility(
                Pa[live], Sa[live], Ka[live], Ta[live], ra[live], flags[live], qa[live],
                model="black_scholes_merton", return_as="numpy", on_error="ignore",
            )
            out[live] = np.asarray(iv, dtype=np.float64)
        except Exception:
            pass

    out = out.reshape(shape) if shape else out  # type: ignore[assignment]
    return _maybe_scalar(np.atleast_1d(out), was_scalar)
