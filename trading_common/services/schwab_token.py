# Ported from D:\chanakya\options_advisor\services\schwab_token.py
"""Schwab OAuth token encryption/decryption using PyNaCl SecretBox.

The token file on disk is encrypted so that a plaintext access/refresh token
is never stored at rest. The ENCRYPTION_MASTER_KEY (same key used for user API
key encryption) is the symmetric secret.

Usage:
    # After writing a plaintext token file:
    SchwabTokenManager().encrypt_file(path)

    # Before passing the path to schwab-py:
    with SchwabTokenManager().plaintext_context(path) as tmp_path:
        client = schwab.auth.client_from_token_file(tmp_path, ...)

SECURITY (fixed after a real leak was found in production use): schwab-py
writes/rewrites the token file in place whenever it refreshes an access
token. `plaintext_context()` used to hand schwab-py a temp file created via
`tempfile.mkstemp(dir=path.parent, ...)` -- i.e. a PLAINTEXT token file
sitting directly inside `secrets/`, next to the encrypted one, for the
duration of the `with` block. If the process was killed, crashed, or the
re-encryption step's silently-swallowed `except Exception: pass` failed for
any reason, that plaintext file was orphaned there permanently -- exactly
what happened (`secrets/tmp0sqf9hpz.json`, `secrets/tmp8wsuccm3.json`, both
unencrypted, both containing a real access/refresh token, discovered
sitting next to `schwab_token.enc.json`).

Fix: the plaintext working copy now lives in a private OS temp directory
(`tempfile.mkdtemp()`, never `secrets/`) that is *shredded* (overwritten
with random bytes, then unlinked, then the whole directory removed) in a
`finally` block that always runs. Re-encryption failures are logged, not
swallowed -- a lost token refresh must be visible, never silent. On every
call, any stray plaintext file already sitting in `secrets/` (a leftover
from a prior crash, e.g. the exact bug this fixes) is purged first, so the
directory self-heals rather than accumulating leaks indefinitely.

SECOND BUG (found running a real, long-lived worker session against the
live market): `plaintext_context()`'s temp file only exists for the
duration of its own `with` block. That is correct for every *short-lived*
caller here (`replay.cli._fetch_via_schwab`, `options_chain.py`'s
`_fetch_chain_schwab` -- construct a client, make one call, discard it, all
inside the same `with` block) but wrong for `brokers.schwab.SchwabBroker`,
which caches its client (`self._client`) for the life of the broker
instance and may reuse it for hours. schwab-py's OAuth session holds a
closure over the temp path and calls back into it to persist a refreshed
access/refresh token on ANY subsequent API call -- including ones long
after `plaintext_context()`'s `with` block (and its temp file) is already
gone, which raised a bare `FileNotFoundError` straight out of authlib and
silently dropped the refreshed token. `token_read_write_funcs()` below
fixes this for that one caller: it hands schwab-py a pair of callables
(`client_from_access_functions`, not `client_from_token_file`) that read/
write the ENCRYPTED file directly, in memory, on every call -- there is no
window, ever, not even a temp file, where the token sits in plaintext on
disk. Strictly stronger than the temp-file approach, and it works
correctly for a client of any lifetime.

THIRD BUG (security review FIX 2 -- closing the "migration window"):
`decrypt_bytes()` used to silently accept a plaintext (non-`_MARKER`)
token file forever, returning its raw bytes with no warning and no
attempt to fix it -- a comment literally called this a "migration
window," but nothing ever closed it. A token file that was ever written
in plaintext (e.g. by an out-of-band bootstrap script, or a pre-fix
version of this code) would stay silently unencrypted-at-rest
indefinitely, with every single read reinforcing the illusion that
encryption-at-rest was actually in effect. Fixed: `decrypt_bytes()` now
re-encrypts a recognized plaintext token in place (via
`_atomic_write_bytes`) the FIRST time it's ever read, logging a WARNING
so the migration is visible in logs rather than invisible -- there is
now at most one plaintext read per token file, ever, not an open-ended
window. Content that is neither valid ciphertext nor a recognizable
plaintext token now raises `ValueError` instead of being handed back to
the caller as if it were usable.
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_master_key() -> bytes:
    from trading_common.config.settings import settings

    raw = settings.encryption_master_key.get_secret_value()
    if not raw:
        raise RuntimeError("ENCRYPTION_MASTER_KEY not set")
    padding = 4 - len(raw) % 4
    if padding != 4:
        raw = raw + "=" * padding
    return base64.urlsafe_b64decode(raw)


def _shred_file(path: Path) -> None:
    """Best-effort secure delete: overwrite the file's contents with random
    bytes (and fsync) before unlinking, so a plaintext token doesn't just
    sit recoverable in freed disk blocks. Not a guarantee against every
    possible recovery technique (SSD wear-leveling in particular can retain
    old blocks despite an in-place overwrite) -- but strictly better than a
    bare unlink, and costs nothing for a file this small."""
    try:
        size = path.stat().st_size
        with open(path, "r+b") as f:
            f.write(os.urandom(size))
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass  # best-effort -- still attempt the unlink below regardless
    try:
        path.unlink()
    except OSError:
        pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` via a same-directory temp file + `os.replace()`
    -- atomic on both POSIX and Windows (a single filesystem rename/
    replace, not a truncate-then-write), so a crash mid-write can never
    leave `path` half-written, truncated, or missing. Used by
    `decrypt_bytes`'s in-place plaintext-migration re-encryption (security
    review FIX 2) -- the one write path where "the file briefly doesn't
    exist, or exists half-written" would be worse than the plaintext
    window this is closing."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _shred_directory(dir_path: Path) -> None:
    """Shred every file in `dir_path` (non-recursive -- this is only ever
    used for the single-file private tmpdir `plaintext_context` creates),
    then remove the directory itself. Never raises -- this runs in a
    `finally` block and must not mask whatever the `with` block itself
    raised."""
    try:
        for child in dir_path.iterdir():
            if child.is_file():
                _shred_file(child)
    except OSError:
        pass
    try:
        shutil.rmtree(dir_path, ignore_errors=True)
    except OSError:
        pass


class SchwabTokenManager:
    """Encrypts/decrypts the Schwab token JSON file at rest."""

    _MARKER = b"SCHWAB_ENC_V1:"

    def encrypt_plaintext(self, plaintext: bytes) -> bytes:
        """Pure core: `plaintext` -> `_MARKER + base64(ciphertext)`. Both
        `encrypt_file` (below) and a consuming app's own Postgres-backed
        token store (e.g. daytrader's `services.schwab_token_store`, used
        by its web-based Schwab reauth flow) build on this SAME primitive,
        so a token can move between file and DB storage without two
        diverging encryption implementations."""
        import nacl.secret

        box = nacl.secret.SecretBox(_get_master_key())
        ciphertext = box.encrypt(plaintext)
        return self._MARKER + base64.b64encode(ciphertext)

    def decrypt_marked(self, raw: bytes) -> bytes:
        """Pure core: the inverse of `encrypt_plaintext`. Raises
        `ValueError` if `raw` doesn't start with `_MARKER` -- callers that
        need to handle a legacy-plaintext migration (`decrypt_bytes`,
        below) check for that themselves first; this is the strict
        no-fallback primitive."""
        import nacl.secret

        if not raw.startswith(self._MARKER):
            raise ValueError("content is not in the expected encrypted format")
        encoded = raw[len(self._MARKER):]
        box = nacl.secret.SecretBox(_get_master_key())
        return box.decrypt(base64.b64decode(encoded))

    def encrypt_file(self, path: Path) -> None:
        """Encrypt a plaintext JSON token file in-place."""
        plaintext = path.read_bytes()
        if plaintext.startswith(self._MARKER):
            return  # already encrypted
        path.write_bytes(self.encrypt_plaintext(plaintext))

    def decrypt_bytes(self, path: Path) -> bytes:
        """Return the decrypted token JSON bytes from `path`.

        SECURITY (security review FIX 2 -- closes the "migration window"):
        a plaintext (unencrypted) token file is no longer accepted
        silently, forever. If `path`'s content isn't in our encrypted
        format but DOES look like a genuine plaintext token (see
        `_looks_like_plaintext_token`), this logs a WARNING and
        immediately re-encrypts it in place (`_atomic_write_bytes` --
        same-directory temp file + `os.replace()`, so a crash mid-write
        can never leave the file half-written or missing) before
        returning the decrypted bytes -- there is now at most one read of
        a plaintext file, ever, per token file, not an indefinitely-long
        window where the encryption-at-rest guarantee simply doesn't
        apply. A file that is neither valid ciphertext NOR a recognizable
        plaintext token (corrupted, truncated, or garbage) raises
        `ValueError` rather than silently falling through and handing the
        caller undecryptable nonsense as if it were a real token."""
        raw = path.read_bytes()
        if raw.startswith(self._MARKER):
            return self.decrypt_marked(raw)

        if not self._looks_like_plaintext_token(raw):
            raise ValueError(
                f"{path}: content is neither valid ciphertext nor a recognizable "
                "plaintext token -- refusing to proceed"
            )

        logger.warning(
            "schwab_token.decrypt_bytes: %s is an unencrypted plaintext token file -- "
            "re-encrypting it in place now (never accepted silently, see module docstring)",
            path,
        )
        _atomic_write_bytes(path, self.encrypt_plaintext(raw))
        return raw

    _PLAINTEXT_TOKEN_FIELDS = ("access_token", "refresh_token", "token")

    def _looks_like_plaintext_token(self, content: bytes) -> bool:
        """True iff `content` is not our encrypted format AND parses as a
        JSON object containing a token-shaped field. Deliberately does NOT
        treat "any non-marker file" as a leak -- an empty file, a
        `.gitkeep`, or some unrelated file dropped in the same directory
        must never be touched, only something that actually looks like a
        leaked credential."""
        if content.startswith(self._MARKER):
            return False
        import json

        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return False
        if not isinstance(parsed, dict):
            return False
        return any(key in parsed for key in self._PLAINTEXT_TOKEN_FIELDS)

    def purge_stray_plaintext_files(self, path: Path) -> list[str]:
        """Defense in depth: scan `path.parent` for any file that is not
        `path` itself and looks like a leaked plaintext token (see
        `_looks_like_plaintext_token`) -- most likely a leftover from a
        crash during a pre-fix `plaintext_context` run -- and shred+delete
        it. Returns the names of any files it removed, purely for
        logging/test visibility -- callers don't need to act on the return
        value.

        Deliberately conservative: only ever touches files directly inside
        `path.parent` (never recurses), never touches `path` itself
        regardless of its content, and never touches a file that doesn't
        actually parse as token-shaped JSON (so unrelated files like
        `.gitkeep` are left alone)."""
        removed: list[str] = []
        if not path.parent.exists():
            return removed
        for candidate in path.parent.iterdir():
            if not candidate.is_file() or candidate == path:
                continue
            try:
                content = candidate.read_bytes()
            except OSError:
                continue
            if not self._looks_like_plaintext_token(content):
                continue
            logger.warning(
                "schwab_token.purge_stray_plaintext_file: removing unencrypted "
                "file %s found next to the token store (see module docstring)",
                candidate.name,
            )
            _shred_file(candidate)
            removed.append(candidate.name)
        return removed

    @contextmanager
    def plaintext_context(self, path: Path):
        """Yield a temporary file path containing the decrypted token JSON.

        The temp file lives in a private OS temp directory (never
        `path.parent`/`secrets/`) for the entire duration of the `with`
        block. On exit, that directory -- which schwab-py may have
        rewritten with a refreshed token -- is re-encrypted back to `path`,
        then unconditionally shredded and removed, whether the `with` block
        succeeded, raised, or the re-encryption itself failed. See the
        module docstring for why this matters: this is fixing a real
        plaintext-token leak, not a theoretical one.
        """
        self.purge_stray_plaintext_files(path)

        if not path.exists():
            yield str(path)
            return

        plaintext = self.decrypt_bytes(path)
        tmpdir = Path(tempfile.mkdtemp(prefix="schwab_token_"))
        tmp_path = tmpdir / "token.json"
        try:
            tmp_path.write_bytes(plaintext)
            yield str(tmp_path)
        finally:
            # Re-encrypt whatever schwab-py left behind (it may have
            # rewritten the file with a refreshed access/refresh token) --
            # a failure here is logged, never silently swallowed, since a
            # swallowed failure would silently drop a real token refresh.
            try:
                if tmp_path.exists():
                    updated = tmp_path.read_bytes()
                    import nacl.secret

                    box = nacl.secret.SecretBox(_get_master_key())
                    path.write_bytes(self._MARKER + base64.b64encode(box.encrypt(updated)))
            except Exception:
                logger.exception(
                    "schwab_token.plaintext_context: failed to re-encrypt refreshed "
                    "token back to %s -- the previous encrypted token on disk is "
                    "unchanged, but any refresh that happened during this session "
                    "was NOT persisted",
                    path,
                )
            # Always shred + remove the private tmpdir, regardless of
            # whether re-encryption above succeeded -- this must never be
            # skipped, it is the entire point of this fix.
            _shred_directory(tmpdir)

    def token_read_write_funcs(self, path: Path) -> tuple[Callable[[], dict], Callable[..., None]]:
        """A `(token_read_func, token_write_func)` pair for schwab-py's
        `schwab.auth.client_from_access_functions` -- for long-lived
        clients (see "SECOND BUG" in the module docstring) instead of
        `plaintext_context()` + `client_from_token_file`. Both callables
        operate on the encrypted file directly: the token exists in
        plaintext only as an in-memory `dict`, for the duration of a single
        read or write call, and never touches disk unencrypted at all --
        not even in a temp file.

        `token_write_func` matches schwab-py's own `__make_update_token_func`
        signature (`def update_token(t, *args, **kwargs)` -- schwab-py's
        `TokenMetadata` wrapper calls it with extra positional/keyword
        metadata this implementation doesn't need, hence `*args, **kwargs`
        being accepted and ignored)."""
        self.purge_stray_plaintext_files(path)

        def read_token() -> dict:
            import json

            return json.loads(self.decrypt_bytes(path))

        def write_token(token: dict, *args: Any, **kwargs: Any) -> None:
            import json

            import nacl.secret

            plaintext = json.dumps(token).encode()
            box = nacl.secret.SecretBox(_get_master_key())
            path.write_bytes(self._MARKER + base64.b64encode(box.encrypt(plaintext)))

        return read_token, write_token
