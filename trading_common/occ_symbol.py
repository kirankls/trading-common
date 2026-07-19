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

__all__ = ["OccSymbolParts", "format_occ_symbol", "parse_occ_symbol"]


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
