"""Pure computation of a Schwab OAuth token's health from its decrypted
JSON -- no DB/file access, so both `GET /api/schwab/status` (api/routes/
schwab.py) and the proactive expiry alert (worker/main.py) share the exact
same "is this token about to become useless" logic rather than two
implementations that could silently drift apart.

Schwab's refresh token has a hard, non-negotiable 7-day lifetime from the
moment it was issued -- unlike many OAuth providers, continued use does
NOT extend or rotate it; re-authenticating (a real, fresh interactive
login) is the only way to get a new one. schwab-py's own token format
tracks exactly this: a `creation_timestamp` set once, at issuance, and
preserved unchanged across every subsequent access-token refresh (see
`schwab.auth.TokenMetadata` -- `wrap_token_in_metadata` always reuses
`self.creation_timestamp`, never `time.time()`). That field is this
module's only input.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

# A real, documented Schwab API platform constraint, not a value this
# project chose -- see this module's docstring.
REFRESH_TOKEN_LIFETIME = timedelta(days=7)

# How far ahead of the hard 7-day cliff to CRITICAL-alert, so there's time
# to react (run scripts/bootstrap_schwab_token.py or use the web reauth
# flow) before the token actually stops working and bot-worker degrades to
# "no live market data."
EXPIRY_WARNING_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class SchwabTokenStatus:
    configured: bool
    valid: bool
    expired: bool
    expires_at: datetime | None
    reason: str


def compute_token_status(token_dict: dict | None, *, now: datetime | None = None) -> SchwabTokenStatus:
    """`token_dict` is the decrypted JSON exactly as schwab-py/authlib
    persist it: `{"creation_timestamp": <unix seconds>, "token": {...}}`.
    `None` means no token has ever been stored (a fresh deployment, or one
    that's never completed the reauth flow)."""
    now = now if now is not None else datetime.now(UTC)

    if token_dict is None:
        return SchwabTokenStatus(
            configured=False, valid=False, expired=False, expires_at=None,
            reason="No Schwab token has been configured yet -- use Settings to connect.",
        )

    creation_timestamp = token_dict.get("creation_timestamp")
    if creation_timestamp is None:
        return SchwabTokenStatus(
            configured=True, valid=False, expired=True, expires_at=None,
            reason="Stored token is missing metadata (legacy/corrupted format) -- reauthorize.",
        )

    expires_at = datetime.fromtimestamp(creation_timestamp, tz=UTC) + REFRESH_TOKEN_LIFETIME
    expired = now >= expires_at
    return SchwabTokenStatus(
        configured=True,
        valid=not expired,
        expired=expired,
        expires_at=expires_at,
        reason=(
            "Refresh token has expired -- reauthorize via Settings."
            if expired
            else f"Valid until {expires_at.isoformat()}."
        ),
    )


def is_within_expiry_warning_window(status: SchwabTokenStatus, *, now: datetime | None = None) -> bool:
    """True iff the token is still valid but will expire within
    `EXPIRY_WARNING_WINDOW` -- the proactive-alert trigger condition.
    Never true for an already-expired or never-configured token (those
    have their own, already-firing failure signals -- the crash-loop fix
    and `worker.market_data_unavailable`/`token_refresh_failure` alert --
    this is specifically the "heads up, act before it breaks" case)."""
    if not status.configured or status.expired or status.expires_at is None:
        return False
    now = now if now is not None else datetime.now(UTC)
    return status.expires_at - now <= EXPIRY_WARNING_WINDOW
