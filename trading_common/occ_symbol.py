"""OCC option-symbol round-trip parsing (SPX audit C3: "fetch Schwab option
positions, round-trip OCC symbols, diff against DB spread positions").

`brokers.schwab._option_symbol` already BUILDS an OCC symbol from an
`OptionLeg` via schwab-py's `OptionSymbol` builder; this module is the
missing INVERSE direction -- parsing a raw OCC symbol string (as returned
by Schwab's own positions/orders REST responses, e.g.
`"SPY   260717C00500000"`) back into its component fields, needed to
compare a live broker position against a persisted `OptionSpreadPosition`
row's `legs` JSONB without re-deriving/guessing anything.

Standard OCC symbol format (21 characters, no separators other than the
root's own trailing space-padding): 6-char space-padded root symbol +
6-digit expiration (YYMMDD) + 1-char C/P + 8-digit strike price
(strike * 1000, zero-padded). This is a plain, dependency-free string
format -- deliberately NOT built on schwab-py's own `OptionSymbol` class
(which only builds, has no parse-back direction), and deliberately
living here (not `brokers/schwab.py` or `brokers/paper.py`) so BOTH
brokers can share one parser without either depending on the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

__all__ = [
    "OccSymbolParts",
    "format_occ_symbol",
    "parse_occ_symbol",
    "parse_polygon_option_ticker",
]


@dataclass(frozen=True)
class OccSymbolParts:
    root: str
    expiration: date
    option_type: str  # "CALL" | "PUT"
    strike: float


def format_occ_symbol(root: str, expiration: date, option_type: str, strike: float) -> str:
    """The encode direction, mirroring `brokers.schwab._option_symbol`'s
    schwab-py-backed builder but dependency-free (no schwab-py import) --
    used by `PaperBroker.positions()` so its simulated option positions
    come back in the SAME OCC-symbol shape a real Schwab account's
    positions response uses, letting reconciliation code treat both
    brokers identically."""
    date_part = expiration.strftime("%y%m%d")
    contract_type = "C" if option_type.upper() == "CALL" else "P"
    strike_part = f"{round(strike * 1000):08d}"
    return f"{root:<6}{date_part}{contract_type}{strike_part}"


def parse_occ_symbol(symbol: str) -> OccSymbolParts | None:
    """Parse a 21-character OCC option symbol. Returns `None` (never
    raises) for anything that doesn't match the expected shape -- a plain
    equity ticker (e.g. "SPY") is the normal, expected non-match case
    (Schwab's account positions/orders responses mix equity and option
    positions in the same list), not an error."""
    if len(symbol) != 21:
        return None

    root = symbol[:6].strip()
    date_part = symbol[6:12]
    contract_type = symbol[12]
    strike_part = symbol[13:21]

    if not root:
        return None
    if contract_type not in ("C", "P"):
        return None
    if not (date_part.isdigit() and strike_part.isdigit()):
        return None

    try:
        expiration = date(2000 + int(date_part[0:2]), int(date_part[2:4]), int(date_part[4:6]))
    except ValueError:
        return None

    strike = int(strike_part) / 1000.0
    option_type = "CALL" if contract_type == "C" else "PUT"
    return OccSymbolParts(root=root, expiration=expiration, option_type=option_type, strike=strike)


def parse_polygon_option_ticker(ticker: str) -> OccSymbolParts | None:
    """Parse a Polygon/Massive-format option ticker (e.g.
    ``"O:A260717C00120000"``) -- verified against a real options
    day-aggregates flat file before writing this parser (docs/
    DAYTRADER_BACKTEST_INTEGRATION_PROMPT.md Phase 3): the file's own
    ``ticker`` column carries the full contract identity (no separate
    underlying/expiry/strike/type columns), and Polygon's convention is
    NOT the same shape ``parse_occ_symbol`` above handles -- there is no
    fixed 6-char space-padded root; the root is variable-length with no
    padding at all, immediately followed by the 6-digit date.

    Format: ``"O:"`` prefix + variable-length root (no padding) + 6-digit
    expiration (YYMMDD) + 1-char C/P + 8-digit strike (strike * 1000,
    zero-padded). Parsed from the END of the string (strike is always the
    last 8 characters, then C/P, then the 6-digit date, then whatever
    remains is the root) -- the only reliable way to handle a
    non-padded, variable-width root without a lookup table of valid
    tickers.

    Returns ``None`` (never raises) for anything that doesn't match --
    Polygon's stock-aggregates files use the bare ticker with no prefix at
    all, which is the normal, expected non-match case when this parser is
    run against mixed input, not an error.
    """
    if not ticker.startswith("O:"):
        return None
    body = ticker[2:]
    if len(body) < 16:  # 1+ char root + 6 date + 1 type + 8 strike, minimum
        return None

    strike_part = body[-8:]
    contract_type = body[-9]
    date_part = body[-15:-9]
    root = body[:-15]

    if not root:
        return None
    if contract_type not in ("C", "P"):
        return None
    if not (date_part.isdigit() and strike_part.isdigit()):
        return None

    try:
        expiration = date(2000 + int(date_part[0:2]), int(date_part[2:4]), int(date_part[4:6]))
    except ValueError:
        return None

    strike = int(strike_part) / 1000.0
    option_type = "CALL" if contract_type == "C" else "PUT"
    return OccSymbolParts(root=root, expiration=expiration, option_type=option_type, strike=strike)
