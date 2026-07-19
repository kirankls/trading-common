"""Unit tests for trading_common.config.settings.Settings.

Covers env-var loading and that SecretStr fields never leak their raw value
via repr()/str().
"""
from __future__ import annotations

import warnings

from trading_common.config.settings import Settings


def test_settings_loads_fields_from_env(monkeypatch):
    monkeypatch.setenv("SCHWAB_TRADING_APP_KEY", "app-key-123")
    monkeypatch.setenv("SCHWAB_TRADING_APP_SECRET", "app-secret-456")
    monkeypatch.setenv("SCHWAB_CALLBACK_URL", "https://example.com/callback")
    monkeypatch.setenv("FINNHUB_API_KEY", "finnhub-789")
    monkeypatch.setenv("POLYGON_API_KEY", "polygon-abc")
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "master-key-def")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./data/test.db")
    monkeypatch.setenv("ASYNC_DATABASE_URL", "postgresql+asyncpg://u:p@host/db")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Settings(_env_file=None)

    assert s.schwab_trading_app_key.get_secret_value() == "app-key-123"
    assert s.schwab_trading_app_secret.get_secret_value() == "app-secret-456"
    assert s.schwab_callback_url == "https://example.com/callback"
    assert s.finnhub_api_key.get_secret_value() == "finnhub-789"
    assert s.polygon_api_key.get_secret_value() == "polygon-abc"
    assert s.encryption_master_key.get_secret_value() == "master-key-def"
    assert s.database_url == "sqlite+aiosqlite:///./data/test.db"
    assert s.async_database_url == "postgresql+asyncpg://u:p@host/db"
    assert s.app_env == "production"
    assert s.log_level == "DEBUG"


def test_settings_defaults(monkeypatch):
    for key in (
        "SCHWAB_TRADING_APP_KEY", "SCHWAB_TRADING_APP_SECRET", "SCHWAB_CALLBACK_URL",
        "FINNHUB_API_KEY", "POLYGON_API_KEY", "ENCRYPTION_MASTER_KEY",
        "DATABASE_URL", "ASYNC_DATABASE_URL", "APP_ENV", "LOG_LEVEL",
    ):
        monkeypatch.delenv(key, raising=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Settings(_env_file=None)

    assert s.schwab_callback_url == "https://127.0.0.1"
    assert s.database_url == "sqlite+aiosqlite:///./data/daytrader.db"
    assert s.async_database_url == ""
    assert s.app_env == "development"
    assert s.log_level == "INFO"


def test_secretstr_fields_do_not_leak_in_repr_or_str(monkeypatch):
    monkeypatch.setenv("SCHWAB_TRADING_APP_KEY", "super-secret-value")
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "another-secret")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Settings(_env_file=None)

    dump = repr(s) + str(s)
    assert "super-secret-value" not in dump
    assert "another-secret" not in dump
    assert "**********" in repr(s.schwab_trading_app_key) or str(s.schwab_trading_app_key) == "**********"


def test_warns_on_missing_critical_key(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_MASTER_KEY", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Settings(_env_file=None)
    messages = [str(w.message) for w in caught]
    assert any("encryption_master_key" in m for m in messages)


def test_no_warning_when_critical_key_set(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "set-value")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Settings(_env_file=None)
    messages = [str(w.message) for w in caught]
    assert not any("encryption_master_key" in m for m in messages)
