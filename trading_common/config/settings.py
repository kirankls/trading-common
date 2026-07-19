"""Shared application settings loaded from environment variables.

This is the trading_common package's OWN settings singleton -- independent
of (but reading the same process environment as) any consuming app's own
extended settings (e.g. daytrader's `config.settings.DayTraderSettings`,
which subclasses this `Settings` for its own application-level config:
risk limits, paper-trading equity, JWT auth, etc.). Every field here is
something a `trading_common` module itself reads directly (`data_clients.
options_chain`'s self-authenticating Schwab fetch, `services.alerts`'s
email/Telegram channels) -- no LLM/Claude fields and no app-specific
fields live here.
"""
from __future__ import annotations

import warnings

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_CRITICAL_KEYS = ("encryption_master_key",)
_OPTIONAL_KEYS = (
    "schwab_trading_app_key", "schwab_trading_app_secret", "finnhub_api_key", "polygon_api_key",
    "fred_api_key", "newsapi_key",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Schwab
    schwab_trading_app_key: SecretStr = SecretStr("")
    schwab_trading_app_secret: SecretStr = SecretStr("")
    schwab_callback_url: str = "https://127.0.0.1"
    # data_clients/options_chain.py's self-authenticating Schwab REST fetch
    # (and any consuming app's own equivalent) decrypts the on-disk token at
    # this path -- a consuming app may override it via SCHWAB_TOKEN_PATH.
    schwab_token_path: str = "secrets/schwab_token.enc.json"

    # Polygon.io
    polygon_api_key: SecretStr = SecretStr("")

    # Finnhub
    finnhub_api_key: SecretStr = SecretStr("")

    # Tradier (options chain fallback, ported with data_clients/options_chain.py)
    tradier_api_key: SecretStr = SecretStr("")

    # FRED (data_clients/fred.py, data_clients/macro.py) -- free key from
    # https://fred.stlouisfed.org/docs/api/api_key.html
    fred_api_key: SecretStr = SecretStr("")

    # NewsAPI (data_clients/news.py) -- free key from https://newsapi.org
    newsapi_key: SecretStr = SecretStr("")
    news_lookback_days: int = 7

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/daytrader.db"

    # Async database URL (PostgreSQL for production; falls back to database_url)
    async_database_url: str = ""   # e.g. postgresql+asyncpg://user:pass@host/db

    # Encryption (for API keys / tokens at rest)
    encryption_master_key: SecretStr = SecretStr("")   # 32-byte base64url; see .env.example

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Alerting (services/alerts.py's send_alert -- email/Telegram channels).
    # Entirely opt-in: every field defaults to empty/None and send_alert
    # no-ops per-channel when that channel's config is incomplete.
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    smtp_sender: str = ""
    smtp_password: SecretStr = SecretStr("")
    alert_recipient_email: str = ""
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    @model_validator(mode="after")
    def _warn_missing_keys(self) -> Settings:
        for field in _CRITICAL_KEYS:
            val = getattr(self, field)
            if not val.get_secret_value():
                warnings.warn(
                    f"Required API key '{field}' is not set — core features will fail.",
                    stacklevel=2,
                )
        for field in _OPTIONAL_KEYS:
            val = getattr(self, field)
            if not val.get_secret_value():
                warnings.warn(
                    f"Optional API key '{field}' is not set — related data source will be skipped.",
                    stacklevel=2,
                )
        return self


settings = Settings()
