"""Tests for trading_common.auth: password hashing (argon2) and JWT
access-token helpers."""
from __future__ import annotations

from trading_common.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

_SECRET = "test-secret-key"


class TestPasswordHashing:
    def test_verify_accepts_correct_password(self):
        hashed = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", hashed) is True

    def test_verify_rejects_wrong_password(self):
        hashed = hash_password("correct horse battery staple")
        assert verify_password("wrong password", hashed) is False

    def test_hash_is_not_the_plaintext(self):
        hashed = hash_password("secret123")
        assert hashed != "secret123"


class TestAccessToken:
    def test_round_trip(self):
        token = create_access_token(subject="user-123", secret_key=_SECRET)
        assert decode_access_token(token, secret_key=_SECRET) == "user-123"

    def test_wrong_secret_fails_to_decode(self):
        token = create_access_token(subject="user-123", secret_key=_SECRET)
        assert decode_access_token(token, secret_key="a-different-secret") is None

    def test_garbage_token_fails_to_decode(self):
        assert decode_access_token("not-a-real-jwt", secret_key=_SECRET) is None

    def test_expired_token_fails_to_decode(self):
        token = create_access_token(subject="user-123", secret_key=_SECRET, expires_minutes=-1)
        assert decode_access_token(token, secret_key=_SECRET) is None

    def test_non_string_subject_claim_is_rejected(self):
        # Craft a token whose 'sub' claim is a non-string (decode_access_token
        # must not hand back a non-str subject even if some other issuer's
        # token has a numeric sub).
        from datetime import UTC, datetime, timedelta

        from jose import jwt

        payload = {"sub": 12345, "exp": datetime.now(UTC) + timedelta(minutes=5)}
        token = jwt.encode(payload, _SECRET, algorithm="HS256")
        assert decode_access_token(token, secret_key=_SECRET) is None

    def test_custom_expiry_and_algorithm_round_trip(self):
        token = create_access_token(subject="user-456", secret_key=_SECRET, algorithm="HS512", expires_minutes=5)
        assert decode_access_token(token, secret_key=_SECRET, algorithm="HS512") == "user-456"
