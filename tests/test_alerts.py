"""Unit tests for trading_common.services.alerts.send_alert and its two
channel senders (_send_email_alert / _send_telegram_alert).

Never touches the network or a real SMTP server: `_send_email` (the sync
core, ported from chanakya's email.py) is monkeypatched directly rather
than mocking smtplib, and httpx.AsyncClient is monkeypatched with a fake
async context manager for the Telegram path.
"""
from __future__ import annotations

import pydantic
import pytest

from trading_common.services import alerts as alerts_module
from trading_common.services.alerts import send_alert


def _set_email_configured(monkeypatch, *, configured: bool) -> None:
    from trading_common.config.settings import settings

    if configured:
        monkeypatch.setattr(settings, "smtp_sender", "bot@example.com")
        monkeypatch.setattr(settings, "smtp_password", pydantic.SecretStr("app-password"))
        monkeypatch.setattr(settings, "alert_recipient_email", "owner@example.com")
        monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    else:
        monkeypatch.setattr(settings, "smtp_sender", "")
        monkeypatch.setattr(settings, "smtp_password", pydantic.SecretStr(""))
        monkeypatch.setattr(settings, "alert_recipient_email", "")


def _set_telegram_configured(monkeypatch, *, configured: bool) -> None:
    from trading_common.config.settings import settings

    if configured:
        monkeypatch.setattr(settings, "telegram_bot_token", pydantic.SecretStr("123:ABC"))
        monkeypatch.setattr(settings, "telegram_chat_id", "42")
    else:
        monkeypatch.setattr(settings, "telegram_bot_token", None)
        monkeypatch.setattr(settings, "telegram_chat_id", None)


class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class _FakeAsyncClient:
    """Fake httpx.AsyncClient supporting `async with ... as client` and a
    single `.post()` call, recording calls on a shared list so tests can
    assert on what was sent."""

    instances: list[_FakeAsyncClient] = []

    def __init__(self, *args, **kwargs) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.response = _FakeResponse(200)
        self.raise_on_post: Exception | None = None
        _FakeAsyncClient.instances.append(self)

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None

    async def post(self, url: str, json: dict):
        self.posts.append((url, json))
        if self.raise_on_post is not None:
            raise self.raise_on_post
        return self.response


@pytest.fixture(autouse=True)
def _reset_fake_client():
    _FakeAsyncClient.instances = []
    yield
    _FakeAsyncClient.instances = []


class TestNoChannelsConfigured:
    async def test_no_op_no_error(self, monkeypatch):
        _set_email_configured(monkeypatch, configured=False)
        _set_telegram_configured(monkeypatch, configured=False)
        monkeypatch.setattr(alerts_module.httpx, "AsyncClient", _FakeAsyncClient)

        email_calls = []
        monkeypatch.setattr(alerts_module, "_send_email", lambda *a, **k: (email_calls.append(a), False)[1])

        delivered = await send_alert("kill_switch", "test message")

        assert email_calls == []
        assert _FakeAsyncClient.instances == []
        assert delivered is False


class TestEmailOnlyConfigured:
    async def test_only_email_attempted(self, monkeypatch):
        _set_email_configured(monkeypatch, configured=True)
        _set_telegram_configured(monkeypatch, configured=False)
        monkeypatch.setattr(alerts_module.httpx, "AsyncClient", _FakeAsyncClient)

        sent = {}

        def fake_send_email(subject, body_html, body_text=""):
            sent["subject"] = subject
            sent["body_html"] = body_html
            sent["body_text"] = body_text
            return True

        monkeypatch.setattr(alerts_module, "_send_email", fake_send_email)

        delivered = await send_alert("kill_switch", "human pressed KILL")

        assert sent["subject"] == "[DayTrader ALERT] kill_switch"
        assert "kill_switch" in sent["body_html"]
        assert "human pressed KILL" in sent["body_html"]
        assert _FakeAsyncClient.instances == []  # telegram never attempted
        assert delivered is True


class TestBothConfigured:
    async def test_both_attempted(self, monkeypatch):
        _set_email_configured(monkeypatch, configured=True)
        _set_telegram_configured(monkeypatch, configured=True)
        monkeypatch.setattr(alerts_module.httpx, "AsyncClient", _FakeAsyncClient)

        email_calls = []
        monkeypatch.setattr(
            alerts_module, "_send_email", lambda *a, **k: (email_calls.append((a, k)), True)[1]
        )

        delivered = await send_alert("circuit_breaker", "3 consecutive losses")

        assert len(email_calls) == 1
        assert len(_FakeAsyncClient.instances) == 1
        client = _FakeAsyncClient.instances[0]
        assert len(client.posts) == 1
        url, payload = client.posts[0]
        assert url == "https://api.telegram.org/bot123:ABC/sendMessage"
        assert payload["chat_id"] == "42"
        assert "circuit_breaker" in payload["text"]
        assert "3 consecutive losses" in payload["text"]
        assert delivered is True


class TestChannelFailureIsolation:
    async def test_email_exception_does_not_prevent_telegram(self, monkeypatch):
        _set_email_configured(monkeypatch, configured=True)
        _set_telegram_configured(monkeypatch, configured=True)
        monkeypatch.setattr(alerts_module.httpx, "AsyncClient", _FakeAsyncClient)

        def raising_send_email(*a, **k):
            raise RuntimeError("smtp connection refused")

        monkeypatch.setattr(alerts_module, "_send_email", raising_send_email)

        # Must not raise.
        delivered = await send_alert("reconciliation_mismatch", "order X missing from broker")

        assert len(_FakeAsyncClient.instances) == 1
        assert len(_FakeAsyncClient.instances[0].posts) == 1
        assert delivered is True  # telegram still got through

    async def test_telegram_exception_does_not_prevent_email(self, monkeypatch):
        _set_email_configured(monkeypatch, configured=True)
        _set_telegram_configured(monkeypatch, configured=True)

        class _RaisingClient(_FakeAsyncClient):
            async def post(self, url: str, json: dict):
                await super().post(url, json)
                raise RuntimeError("network unreachable")

        monkeypatch.setattr(alerts_module.httpx, "AsyncClient", _RaisingClient)

        email_calls = []
        monkeypatch.setattr(alerts_module, "_send_email", lambda *a, **k: (email_calls.append(a), True)[1])

        # Must not raise.
        delivered = await send_alert("heartbeat_loss", "worker heartbeat stale 90s")

        assert len(email_calls) == 1
        assert delivered is True  # email still got through

    async def test_neither_channel_raising_propagates_out_of_send_alert(self, monkeypatch):
        """Even a totally broken config (e.g. settings import blows up
        inside a sender) must not escape send_alert."""
        _set_email_configured(monkeypatch, configured=True)
        _set_telegram_configured(monkeypatch, configured=True)

        def raising_send_email(*a, **k):
            raise RuntimeError("boom")

        class _RaisingClient(_FakeAsyncClient):
            async def post(self, url: str, json: dict):
                raise RuntimeError("boom too")

        monkeypatch.setattr(alerts_module, "_send_email", raising_send_email)
        monkeypatch.setattr(alerts_module.httpx, "AsyncClient", _RaisingClient)

        delivered = await send_alert("drawdown", "intraday drawdown breached")  # must not raise
        assert delivered is False


class TestMessageFormatting:
    async def test_email_subject_and_body_include_event_type(self, monkeypatch):
        _set_email_configured(monkeypatch, configured=True)
        _set_telegram_configured(monkeypatch, configured=False)

        captured = {}

        def fake_send_email(subject, body_html, body_text=""):
            captured["subject"] = subject
            captured["body_html"] = body_html
            captured["body_text"] = body_text
            return True

        monkeypatch.setattr(alerts_module, "_send_email", fake_send_email)

        delivered = await send_alert("stale_data", "no tick for 15s during RTH", channels=("email",))

        assert captured["subject"] == "[DayTrader ALERT] stale_data"
        assert "stale_data" in captured["body_html"]
        assert "no tick for 15s during RTH" in captured["body_html"]
        assert captured["body_text"] == "no tick for 15s during RTH"
        assert delivered is True

    async def test_telegram_text_includes_event_type_and_message(self, monkeypatch):
        _set_email_configured(monkeypatch, configured=False)
        _set_telegram_configured(monkeypatch, configured=True)
        monkeypatch.setattr(alerts_module.httpx, "AsyncClient", _FakeAsyncClient)

        delivered = await send_alert("token_refresh_failure", "Schwab auth failed: InvalidGrantError", channels=("telegram",))

        client = _FakeAsyncClient.instances[0]
        _, payload = client.posts[0]
        assert "token_refresh_failure" in payload["text"]
        assert "Schwab auth failed: InvalidGrantError" in payload["text"]
        assert delivered is True


class TestChannelSelection:
    async def test_channels_param_restricts_to_requested_subset(self, monkeypatch):
        _set_email_configured(monkeypatch, configured=True)
        _set_telegram_configured(monkeypatch, configured=True)
        monkeypatch.setattr(alerts_module.httpx, "AsyncClient", _FakeAsyncClient)

        email_calls = []
        monkeypatch.setattr(alerts_module, "_send_email", lambda *a, **k: (email_calls.append(a), True)[1])

        delivered = await send_alert("kill_switch", "manual kill", channels=("email",))

        assert len(email_calls) == 1
        assert _FakeAsyncClient.instances == []
        assert delivered is True
