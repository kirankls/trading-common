"""Best-effort alert fan-out: email (SMTP) + Telegram.

CLAUDE_CODE_PROMPT.md Phase 2: "Alerting live: email/Telegram on kill
switch, circuit breaker, reconciliation mismatch, heartbeat loss,
token-refresh failure." DAY_TRADER_STRATEGY.md 8b.7: "alert on: heartbeat
loss, reconciliation mismatch, circuit breaker, token refresh failure, 3x
stream reconnects."

Email-sending logic is ported (not rewritten) from
`D:\\chanakya\\options_advisor\\services\\email.py`'s `send_email`/
`send_email_async` (Gmail SMTP via `smtplib.SMTP_SSL`, run off the event
loop via `asyncio.to_thread`) -- generalized to read `config.settings.
settings`'s `smtp_*` fields instead of chanakya's `gmail_*` names, since
this project doesn't hardcode Gmail. Telegram has no chanakya precedent;
it's a small, fresh `httpx.AsyncClient` POST to the Bot API's `sendMessage`
method.

Design (per this module's callers -- engine/runner.py's `_enter_halted`,
worker/persistence.py's reconciliation-mismatch handling, api/main.py's
`/health/deep`, brokers/schwab.py's auth-failure path): alerting is
opt-in per deployment (a channel with incomplete config is silently
skipped, never an error -- the same "absence of config means the safe/
no-op default, never more risk" discipline used throughout this
codebase's `_persist_*`/`_load_*` hooks) and must never itself become a
new failure mode. `send_alert` therefore never raises: every channel's
send is wrapped so one channel's exception can't prevent the other
channel's attempt, and a total alerting outage never crashes the caller
-- whether that caller is the worker's own halt/circuit-breaker path or
the API's health check, alerting failing is strictly less bad than the
thing it was trying to alert about going unreported.
"""
from __future__ import annotations

import asyncio as _asyncio
import smtplib
import ssl
from collections.abc import Sequence
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx
import structlog

logger = structlog.get_logger(__name__)

_TELEGRAM_API_BASE = "https://api.telegram.org"


def _send_email(subject: str, body_html: str, body_text: str = "") -> bool:
    """Send an email via SMTP. Returns True on success, False on failure
    (including "not configured"). Ported from chanakya's `send_email`
    (same Gmail-SMTP-over-SSL approach), reading `trading_common.config.
    settings.settings` for host/port/sender/password/recipient instead of
    chanakya's `gmail_sender`/`gmail_app_password`/`alert_recipient_email`
    names.
    """
    from trading_common.config.settings import settings

    sender = settings.smtp_sender
    password = settings.smtp_password.get_secret_value()
    recipient = settings.alert_recipient_email
    host = settings.smtp_host
    port = settings.smtp_port
    if not sender or not password or not recipient or not host:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        return True
    except Exception:
        return False


async def send_email_async(subject: str, body_html: str, body_text: str = "") -> bool:
    """Non-blocking wrapper -- runs `_send_email` in a thread pool (ported
    verbatim from chanakya's `send_email_async`, `smtplib` is synchronous
    I/O and must never block the event loop)."""
    return await _asyncio.to_thread(_send_email, subject, body_html, body_text)


async def _send_email_alert(event_type: str, message: str) -> bool:
    """No-ops immediately if SMTP isn't fully configured. Never raises --
    `send_email_async`'s own try/except already reduces failures to a
    `False` return; this wrapper additionally guards against any
    unexpected exception escaping (e.g. `config.settings` import errors)
    so a broken email channel can never take down `send_alert`. Returns
    whether the send actually succeeded (False for "not configured", a
    send failure, or a caught exception alike -- all three mean this
    channel did not deliver)."""
    from trading_common.config.settings import settings

    if not (settings.smtp_sender and settings.smtp_password.get_secret_value() and settings.alert_recipient_email):
        return False
    subject = f"[DayTrader ALERT] {event_type}"
    body_html = f"<h3>{event_type}</h3><p>{message}</p>"
    try:
        ok = await send_email_async(subject, body_html, message)
        if not ok:
            logger.warning("alert_email_send_failed", event_type=event_type)
        return ok
    except Exception:
        logger.exception("alert_email_send_raised", event_type=event_type)
        return False


async def _send_telegram_alert(event_type: str, message: str) -> bool:
    """No-ops immediately if the Telegram bot token/chat id aren't both
    configured. Never raises -- any `httpx` error (network, timeout, 4xx/
    5xx from Telegram) is caught and logged, matching `_send_email_alert`'s
    contract. Returns whether the send actually succeeded."""
    from trading_common.config.settings import settings

    if settings.telegram_bot_token is None or settings.telegram_chat_id is None:
        return False
    token = settings.telegram_bot_token.get_secret_value()
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        return False
    text = f"[DayTrader ALERT] {event_type}\n{message}"
    url = f"{_TELEGRAM_API_BASE}/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={"chat_id": chat_id, "text": text})
            if resp.status_code >= 400:
                logger.warning(
                    "alert_telegram_send_failed", event_type=event_type, status_code=resp.status_code
                )
                return False
            return True
    except Exception:
        logger.exception("alert_telegram_send_raised", event_type=event_type)
        return False


async def send_alert(event_type: str, message: str, *, channels: Sequence[str] = ("email", "telegram")) -> bool:
    """Best-effort fan-out to every requested channel that has credentials
    configured. A channel with no credentials configured is silently
    skipped (not an error -- alerting is opt-in per deployment). Each
    channel's send failure is logged but never raised -- a broken alert
    channel must never crash the caller (the worker's own halt/circuit-
    breaker path, or the API's health check).

    Channels run concurrently (`asyncio.gather(..., return_exceptions=True)`)
    so one channel's exception can never prevent another's attempt; any
    exception that somehow still escapes an individual `_send_*_alert`
    (both of which already catch broadly) is caught and logged here too,
    as a last line of defense -- this function's contract is "never
    raises," full stop.

    Returns True if at least one requested, configured channel actually
    delivered -- a consuming app's own AlertService (e.g. daytrader's
    `services.alert_service.AlertService`) uses this to decide its
    `alerts` table row's `delivered` flag and whether to log its own
    CRITICAL delivery-failure fallback. False covers "nothing was
    configured" and "every attempted channel failed" alike; callers that
    only care about fire-and-forget behavior can simply ignore it.
    """
    senders = {
        "email": _send_email_alert,
        "telegram": _send_telegram_alert,
    }
    requested = [channel for channel in channels if channel in senders]
    if not requested:
        return False
    tasks = [senders[channel](event_type, message) for channel in requested]
    results = await _asyncio.gather(*tasks, return_exceptions=True)
    delivered = False
    for channel, result in zip(requested, results, strict=True):
        if isinstance(result, Exception):
            logger.exception("alert_channel_raised_unexpectedly", event_type=event_type, channel=channel, exc_info=result)
        elif result:
            delivered = True
    return delivered
