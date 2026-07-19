"""Unit tests for trading_common.services.schwab_token.SchwabTokenManager.

Round-trips a plaintext JSON token file through encrypt_file()/decrypt_bytes(),
verifies the ciphertext file does not contain the plaintext substring, and
exercises plaintext_context() re-encryption on exit.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from trading_common.services.schwab_token import SchwabTokenManager


@pytest.fixture(autouse=True)
def fake_master_key(monkeypatch):
    """Monkeypatch a 32-byte base64url-encoded master key into settings."""
    raw_key = os.urandom(32)
    encoded = base64.urlsafe_b64encode(raw_key).decode().rstrip("=")

    from trading_common.config.settings import settings

    monkeypatch.setattr(
        settings,
        "encryption_master_key",
        __import__("pydantic").SecretStr(encoded),
    )
    return encoded


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "token.json"
    payload = {"access_token": "abc123-super-secret", "refresh_token": "refresh-xyz"}
    path.write_bytes(json.dumps(payload).encode())
    return path, payload


def test_encrypt_then_decrypt_round_trip(token_file):
    path, payload = token_file
    mgr = SchwabTokenManager()

    mgr.encrypt_file(path)
    decrypted = mgr.decrypt_bytes(path)

    assert json.loads(decrypted) == payload


def test_ciphertext_does_not_contain_plaintext_secret(token_file):
    path, payload = token_file
    mgr = SchwabTokenManager()

    mgr.encrypt_file(path)
    ciphertext = path.read_bytes()

    assert b"abc123-super-secret" not in ciphertext
    assert b"refresh-xyz" not in ciphertext
    assert ciphertext.startswith(mgr._MARKER)


def test_encrypt_file_is_idempotent(token_file):
    path, payload = token_file
    mgr = SchwabTokenManager()

    mgr.encrypt_file(path)
    once = path.read_bytes()
    mgr.encrypt_file(path)  # should detect marker and no-op
    twice = path.read_bytes()

    assert once == twice


def test_decrypt_bytes_reencrypts_a_plaintext_file_in_place(tmp_path, caplog):
    """SECURITY (FIX 2 -- closes the migration window): a plaintext token
    file still decrypts correctly on first read, but decrypt_bytes must
    never accept it silently -- it logs a WARNING and re-encrypts the
    file in place before returning."""
    import logging

    path = tmp_path / "plain.json"
    payload = {"access_token": "still-plaintext"}
    path.write_bytes(json.dumps(payload).encode())

    mgr = SchwabTokenManager()
    with caplog.at_level(logging.WARNING):
        result = mgr.decrypt_bytes(path)

    assert json.loads(result) == payload
    assert any("plaintext" in record.message.lower() for record in caplog.records)


def test_no_plaintext_remains_on_disk_after_first_read(tmp_path):
    """After decrypt_bytes has read a plaintext file once, the file on
    disk must be the encrypted form -- not a lingering plaintext copy,
    and correctly decryptable from then on."""
    path = tmp_path / "plain.json"
    payload = {"access_token": "still-plaintext", "refresh_token": "refresh-xyz"}
    path.write_bytes(json.dumps(payload).encode())

    mgr = SchwabTokenManager()
    mgr.decrypt_bytes(path)  # triggers the in-place re-encryption

    on_disk = path.read_bytes()
    assert on_disk.startswith(mgr._MARKER)
    assert b"still-plaintext" not in on_disk
    assert b"refresh-xyz" not in on_disk

    # And it's still correctly readable as real ciphertext going forward.
    assert json.loads(mgr.decrypt_bytes(path)) == payload


def test_decrypt_bytes_malformed_file_raises(tmp_path):
    """Content that is neither valid ciphertext (no marker) nor a
    recognizable plaintext token (not JSON, or JSON without a
    token-shaped field) must raise, not be silently treated as either
    format."""
    path = tmp_path / "garbage.json"
    path.write_bytes(b"this is not json and not our ciphertext marker")

    mgr = SchwabTokenManager()
    with pytest.raises(ValueError, match="neither valid ciphertext"):
        mgr.decrypt_bytes(path)


def test_decrypt_bytes_json_without_token_fields_raises(tmp_path):
    """Valid JSON that simply isn't token-shaped (e.g. some unrelated
    config file someone pointed this at by mistake) must also raise, not
    be treated as a legitimate plaintext token to migrate."""
    path = tmp_path / "unrelated.json"
    path.write_bytes(json.dumps({"unrelated_field": "value"}).encode())

    mgr = SchwabTokenManager()
    with pytest.raises(ValueError, match="neither valid ciphertext"):
        mgr.decrypt_bytes(path)


def test_reencryption_failure_cleans_up_its_temp_file_and_leaves_original_untouched(tmp_path, monkeypatch):
    """If the atomic re-encrypt write fails partway (os.replace raises),
    the original plaintext file must be left exactly as it was -- never
    corrupted, truncated, or deleted -- and no stray temp file should be
    left behind either."""
    from trading_common.services import schwab_token as schwab_token_module

    path = tmp_path / "plain.json"
    payload = {"access_token": "still-plaintext"}
    original_bytes = json.dumps(payload).encode()
    path.write_bytes(original_bytes)

    def _boom(*args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(schwab_token_module.os, "replace", _boom)

    mgr = SchwabTokenManager()
    with pytest.raises(OSError, match="disk full"):
        mgr.decrypt_bytes(path)

    assert path.read_bytes() == original_bytes  # untouched
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []


def test_plaintext_context_yields_decrypted_temp_file(token_file):
    path, payload = token_file
    mgr = SchwabTokenManager()
    mgr.encrypt_file(path)

    with mgr.plaintext_context(path) as tmp_path:
        assert os.path.exists(tmp_path)
        with open(tmp_path, "rb") as f:
            assert json.loads(f.read()) == payload

    # Temp file cleaned up after context exit
    assert not os.path.exists(tmp_path)


def test_plaintext_context_reencrypts_on_exit_with_updates(token_file):
    path, payload = token_file
    mgr = SchwabTokenManager()
    mgr.encrypt_file(path)

    updated_payload = {"access_token": "refreshed-token", "refresh_token": "new-refresh"}
    with mgr.plaintext_context(path) as tmp_path:
        with open(tmp_path, "wb") as f:
            f.write(json.dumps(updated_payload).encode())

    # File on disk should now be re-encrypted with the updated content
    ciphertext = path.read_bytes()
    assert ciphertext.startswith(mgr._MARKER)
    assert b"refreshed-token" not in ciphertext

    decrypted = mgr.decrypt_bytes(path)
    assert json.loads(decrypted) == updated_payload


def test_plaintext_context_missing_file_yields_path_unchanged(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    mgr = SchwabTokenManager()

    with mgr.plaintext_context(missing) as yielded:
        assert yielded == str(missing)

    assert not missing.exists()


def test_missing_master_key_raises(monkeypatch, token_file):
    path, _ = token_file
    from pydantic import SecretStr

    from trading_common.config.settings import settings

    monkeypatch.setattr(settings, "encryption_master_key", SecretStr(""))

    mgr = SchwabTokenManager()
    with pytest.raises(RuntimeError, match="ENCRYPTION_MASTER_KEY not set"):
        mgr.encrypt_file(path)


def _is_plaintext_token(raw: bytes, marker: bytes) -> bool:
    """True iff `raw` is NOT marked as our encrypted format AND parses as
    JSON containing something that looks like a real token field. This is
    deliberately permissive about what counts as "looks like a token" --
    the regression this guards is a real credential leak, so a false
    positive (flagging a non-token JSON file) is far cheaper than a false
    negative (missing an actual leaked token)."""
    if raw.startswith(marker):
        return False
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    return any(key in parsed for key in ("access_token", "refresh_token", "token"))


class TestNoPlaintextTokenLeak:
    """Regression test for a real leak: `plaintext_context()` used to write
    schwab-py's working copy via `tempfile.mkstemp(dir=path.parent, ...)`,
    i.e. directly inside the same directory as the encrypted token file.
    A crash mid-refresh (or the old code's silently-swallowed re-encrypt
    failure) left plaintext token files sitting there permanently --
    exactly what `secrets/tmp0sqf9hpz.json` / `secrets/tmp8wsuccm3.json`
    were when this was found. This asserts the token directory contains
    ONLY encrypted files after every exercised code path, including a
    simulated crash mid-refresh."""

    def test_secrets_dir_contains_only_encrypted_files_after_simulated_refresh(self, token_file):
        path, payload = token_file
        secrets_dir = path.parent
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        # Simulate schwab-py refreshing the token: it rewrites the
        # yielded temp file path with a new access/refresh token.
        refreshed_payload = {"access_token": "refreshed-abc", "refresh_token": "refreshed-xyz"}
        with mgr.plaintext_context(path) as tmp_path:
            with open(tmp_path, "wb") as f:
                f.write(json.dumps(refreshed_payload).encode())

        for candidate in secrets_dir.iterdir():
            if not candidate.is_file():
                continue
            raw = candidate.read_bytes()
            assert not _is_plaintext_token(raw, mgr._MARKER), (
                f"{candidate} contains a parseable plaintext token after "
                f"plaintext_context() exited -- this is the exact leak this "
                f"test guards against"
            )

        # The refresh must have actually landed, encrypted, at `path`.
        assert json.loads(mgr.decrypt_bytes(path)) == refreshed_payload

    def test_plaintext_context_never_creates_a_file_inside_secrets_dir(self, token_file, monkeypatch):
        """Even transiently, mid-`with`-block: the working copy must live
        in a private OS temp directory, never `path.parent`."""
        path, _ = token_file
        secrets_dir = path.parent
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        files_before = set(secrets_dir.iterdir())

        with mgr.plaintext_context(path) as tmp_path:
            # The yielded path itself must not be inside secrets_dir.
            assert Path(tmp_path).parent != secrets_dir
            # And no NEW file should have appeared inside secrets_dir
            # while the context is open.
            files_during = set(secrets_dir.iterdir())
            assert files_during == files_before

    def test_a_crash_mid_refresh_leaves_no_plaintext_behind(self, token_file):
        """Simulates the process dying partway through a refresh (an
        exception propagates out of the `with` block after schwab-py has
        already rewritten the temp file) -- the private tmpdir must still
        be shredded and removed, and the encrypted file on disk must still
        be the pre-refresh version (the crash means the refresh is lost,
        which is correct: we must never persist a half-written token)."""
        path, payload = token_file
        secrets_dir = path.parent
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        captured_tmp_path: list[str] = []
        with pytest.raises(RuntimeError, match="simulated crash"):
            with mgr.plaintext_context(path) as tmp_path:
                captured_tmp_path.append(tmp_path)
                with open(tmp_path, "wb") as f:
                    f.write(json.dumps({"access_token": "mid-refresh"}).encode())
                raise RuntimeError("simulated crash")

        # The private tmpdir (not secrets_dir) must be gone.
        assert not os.path.exists(captured_tmp_path[0])
        assert not os.path.exists(os.path.dirname(captured_tmp_path[0]))

        # secrets_dir must contain no plaintext leak.
        for candidate in secrets_dir.iterdir():
            if candidate.is_file():
                raw = candidate.read_bytes()
                assert not _is_plaintext_token(raw, mgr._MARKER)

    def test_purges_a_preexisting_stray_plaintext_file(self, token_file):
        """Defense in depth: a stray unencrypted file already sitting next
        to the token store (e.g. a leftover from a crash before this fix
        existed) is purged the next time plaintext_context() runs, rather
        than accumulating forever."""
        path, payload = token_file
        secrets_dir = path.parent
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        stray = secrets_dir / "tmpSTRAY123.json"
        stray.write_bytes(json.dumps({"access_token": "leaked-from-a-prior-crash"}).encode())
        assert stray.exists()

        with mgr.plaintext_context(path):
            pass

        assert not stray.exists()

    def test_purge_stray_plaintext_files_reports_what_it_removed(self, token_file):
        path, payload = token_file
        secrets_dir = path.parent
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        stray = secrets_dir / "tmpSTRAY456.json"
        stray.write_bytes(json.dumps({"refresh_token": "also-leaked"}).encode())

        removed = mgr.purge_stray_plaintext_files(path)

        assert removed == ["tmpSTRAY456.json"]
        assert not stray.exists()

    def test_purge_ignores_non_json_and_already_encrypted_files(self, token_file):
        """Must not misfire on unrelated files (e.g. a .gitkeep) or on the
        legitimately-encrypted token file itself."""
        path, payload = token_file
        secrets_dir = path.parent
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        unrelated = secrets_dir / ".gitkeep"
        unrelated.write_bytes(b"")

        removed = mgr.purge_stray_plaintext_files(path)

        assert removed == []
        assert unrelated.exists()
        assert path.exists()


class TestTokenReadWriteFuncs:
    """Regression coverage for the second real bug this module's docstring
    describes: a long-lived client (SchwabBroker, cached across the whole
    worker process) needs to persist a refreshed token on ANY later call,
    not just at construction time -- `plaintext_context()`'s temp file is
    long gone by then. `token_read_write_funcs()` fixes this by never
    touching disk in plaintext at all, for a client of any lifetime."""

    def test_read_returns_the_decrypted_token_as_a_dict(self, token_file):
        path, payload = token_file
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        read_token, _write_token = mgr.token_read_write_funcs(path)

        assert read_token() == payload

    def test_write_then_read_round_trips_a_refreshed_token(self, token_file):
        """The exact scenario that broke: read once at construction, then
        write again much later (simulating a token refresh long after any
        short-lived temp file would have been shredded), then read again
        -- must reflect the refreshed token, still encrypted at rest."""
        path, _payload = token_file
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        read_token, write_token = mgr.token_read_write_funcs(path)
        assert read_token() is not None  # simulates SchwabBroker's initial client construction

        refreshed = {"access_token": "refreshed-abc", "refresh_token": "refreshed-xyz"}
        write_token(refreshed)  # simulates a refresh happening long after construction

        ciphertext = path.read_bytes()
        assert ciphertext.startswith(mgr._MARKER)
        assert b"refreshed-abc" not in ciphertext
        assert read_token() == refreshed

    def test_write_accepts_and_ignores_extra_positional_and_keyword_args(self, token_file):
        """Matches schwab-py's own `update_token(t, *args, **kwargs)`
        signature -- its TokenMetadata wrapper calls the write function
        with extra metadata this implementation doesn't need."""
        path, _payload = token_file
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        _read_token, write_token = mgr.token_read_write_funcs(path)
        refreshed = {"access_token": "still-works", "refresh_token": "still-works-too"}

        write_token(refreshed, "some_extra_positional_arg", some_kwarg="ignored")

        assert json.loads(mgr.decrypt_bytes(path)) == refreshed

    def test_never_creates_any_file_in_secrets_dir_across_many_read_write_cycles(self, token_file):
        """The core guarantee: unlike plaintext_context(), there is no
        window -- not even a temp file -- where a plaintext token sits on
        disk, across any number of read/write cycles."""
        path, _payload = token_file
        secrets_dir = path.parent
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        files_before = set(secrets_dir.iterdir())
        read_token, write_token = mgr.token_read_write_funcs(path)

        for i in range(3):
            read_token()
            write_token({"access_token": f"cycle-{i}", "refresh_token": f"cycle-{i}-refresh"})

        assert set(secrets_dir.iterdir()) == files_before
        for candidate in secrets_dir.iterdir():
            if candidate.is_file():
                assert not _is_plaintext_token(candidate.read_bytes(), mgr._MARKER)

    def test_purges_a_preexisting_stray_plaintext_file_on_construction(self, token_file):
        """Same defense-in-depth purge as plaintext_context() -- a stray
        leaked file next to the token store is cleaned up when the
        read/write functions are constructed."""
        path, _payload = token_file
        secrets_dir = path.parent
        mgr = SchwabTokenManager()
        mgr.encrypt_file(path)

        stray = secrets_dir / "tmpSTRAY789.json"
        stray.write_bytes(json.dumps({"access_token": "leaked-from-a-prior-crash"}).encode())

        mgr.token_read_write_funcs(path)

        assert not stray.exists()
