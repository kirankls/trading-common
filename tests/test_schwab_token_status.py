"""Unit tests for trading_common.services.schwab_token_status."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trading_common.services.schwab_token_status import (
    REFRESH_TOKEN_LIFETIME,
    compute_token_status,
    is_within_expiry_warning_window,
)

_NOW = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


class TestComputeTokenStatusNotConfigured:
    def test_none_token_is_not_configured(self):
        status = compute_token_status(None, now=_NOW)
        assert status.configured is False
        assert status.valid is False
        assert status.expired is False
        assert status.expires_at is None


class TestComputeTokenStatusMissingMetadata:
    def test_missing_creation_timestamp_is_expired(self):
        status = compute_token_status({"token": {"access_token": "x"}}, now=_NOW)
        assert status.configured is True
        assert status.valid is False
        assert status.expired is True
        assert status.expires_at is None


class TestComputeTokenStatusFreshToken:
    def test_freshly_issued_token_is_valid(self):
        token = {"creation_timestamp": int(_NOW.timestamp()), "token": {"access_token": "x"}}
        status = compute_token_status(token, now=_NOW)
        assert status.configured is True
        assert status.valid is True
        assert status.expired is False
        assert status.expires_at == _NOW + REFRESH_TOKEN_LIFETIME


class TestComputeTokenStatusExpiry:
    def test_token_exactly_at_the_7_day_cliff_is_expired(self):
        issued_at = _NOW - REFRESH_TOKEN_LIFETIME
        token = {"creation_timestamp": int(issued_at.timestamp()), "token": {}}
        status = compute_token_status(token, now=_NOW)
        assert status.expired is True
        assert status.valid is False

    def test_token_one_second_before_the_cliff_is_still_valid(self):
        issued_at = _NOW - REFRESH_TOKEN_LIFETIME + timedelta(seconds=1)
        token = {"creation_timestamp": int(issued_at.timestamp()), "token": {}}
        status = compute_token_status(token, now=_NOW)
        assert status.expired is False
        assert status.valid is True

    def test_a_reauth_resets_the_clock_from_the_new_creation_timestamp(self):
        """Found live in production: reauthorizing gets a genuinely NEW
        refresh token from Schwab, not a rotation of the old one --
        `creation_timestamp` must be set to the moment of the successful
        callback, not preserved from any prior token, or the 7-day clock
        would never actually reset."""
        old_token = {"creation_timestamp": int((_NOW - REFRESH_TOKEN_LIFETIME).timestamp()), "token": {}}
        assert compute_token_status(old_token, now=_NOW).expired is True

        reauthed_token = {"creation_timestamp": int(_NOW.timestamp()), "token": {}}
        assert compute_token_status(reauthed_token, now=_NOW).expired is False


class TestExpiryWarningWindow:
    def test_a_token_expiring_within_24h_triggers_the_warning(self):
        issued_at = _NOW - REFRESH_TOKEN_LIFETIME + timedelta(hours=12)  # 12h left
        status = compute_token_status({"creation_timestamp": int(issued_at.timestamp()), "token": {}}, now=_NOW)
        assert is_within_expiry_warning_window(status, now=_NOW) is True

    def test_a_token_with_plenty_of_time_left_does_not_trigger(self):
        issued_at = _NOW  # 7 days left
        status = compute_token_status({"creation_timestamp": int(issued_at.timestamp()), "token": {}}, now=_NOW)
        assert is_within_expiry_warning_window(status, now=_NOW) is False

    def test_an_already_expired_token_does_not_trigger_the_warning(self):
        """The warning is specifically "act before it breaks" -- an
        already-expired token has its own distinct failure signals (the
        crash-loop fix's graceful degradation + token_refresh_failure
        alert), not this one."""
        issued_at = _NOW - REFRESH_TOKEN_LIFETIME - timedelta(hours=1)
        status = compute_token_status({"creation_timestamp": int(issued_at.timestamp()), "token": {}}, now=_NOW)
        assert is_within_expiry_warning_window(status, now=_NOW) is False

    def test_a_never_configured_token_does_not_trigger_the_warning(self):
        status = compute_token_status(None, now=_NOW)
        assert is_within_expiry_warning_window(status, now=_NOW) is False
