"""Password hashing + JWT access-token helpers shared by the API service.

Generic and storage-agnostic (no `User` model coupling here -- that lives in
`daytrader/storage/models.py`, since the user table is DayTrader-specific).
Pattern ported from `D:\\chanakya\\options_advisor\\api\\routes\\user_auth.py`:
argon2 for password hashing (memory-hard, no separate bcrypt 72-byte-input
footgun), `python-jose` for JWT encode/decode.

Scope note: this is a single-operator control panel (one seeded dev/admin
user per CLAUDE_CODE_PROMPT.md's "Seed a dev user... for local login and
e2e tests"), not a multi-tenant SaaS -- unlike chanakya's user_auth.py, there
is deliberately no registration endpoint, email verification, password
reset, or refresh-token-family theft detection here. A single short-lived
Bearer access token is enough for an operator dashboard; adding that
machinery now would be speculative complexity with no second user to
justify it.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from jose import JWTError, jwt

_hasher = PasswordHasher()


def hash_password(plain_password: str) -> str:
    return _hasher.hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, plain_password)
    except VerifyMismatchError:
        return False


def create_access_token(
    *, subject: str, secret_key: str, algorithm: str = "HS256", expires_minutes: int = 60
) -> str:
    """`subject` is the JWT `sub` claim -- the user id, as a string."""
    expire = datetime.now(UTC) + timedelta(minutes=expires_minutes)
    return jwt.encode({"sub": subject, "exp": expire}, secret_key, algorithm=algorithm)


def decode_access_token(token: str, *, secret_key: str, algorithm: str = "HS256") -> str | None:
    """Return the `sub` claim if `token` is a valid, unexpired JWT signed
    with `secret_key`; None if invalid/expired/malformed (never raises --
    callers turn None into an HTTP 401)."""
    try:
        payload: dict[str, Any] = jwt.decode(token, secret_key, algorithms=[algorithm])
    except JWTError:
        return None
    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None
