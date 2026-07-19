"""Unit tests for trading_common.data_clients.news.NewsClient -- ported
from the chanakya repo alongside the module itself (no dedicated test
existed there; this is new coverage written for the extracted package).

Rules: no real network calls -- httpx is always mocked.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from trading_common.data_clients.base import FetchErrorType
from trading_common.data_clients.news import NewsClient


def _mock_response(status_code: int, payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    if status_code == 200:
        response.raise_for_status.return_value = None
    return response


class TestFetchHeadlines:
    def test_no_api_key_returns_auth_failure(self):
        with patch("trading_common.data_clients.news.settings.newsapi_key.get_secret_value", return_value=""):
            result = NewsClient().fetch_headlines("AAPL")

        assert result.ok is False
        assert result.error.error_type is FetchErrorType.AUTH

    def test_successful_response_parses_articles(self):
        payload = {
            "articles": [
                {
                    "title": "Apple hits new high",
                    "url": "https://example.com/a",
                    "source": {"name": "Example News"},
                    "publishedAt": "2026-07-01T12:00:00Z",
                    "description": "Apple stock rallies.",
                }
            ]
        }
        with (
            patch("trading_common.data_clients.news.settings.newsapi_key.get_secret_value", return_value="fake-key"),
            patch("trading_common.data_clients.news.httpx.get", return_value=_mock_response(200, payload)),
        ):
            result = NewsClient().fetch_headlines("AAPL")

        assert result.ok is True
        [article] = result.data
        assert article.title == "Apple hits new high"
        assert article.source == "Example News"
        assert article.url_hash  # sha256 hex digest, non-empty

    def test_401_returns_auth_failure(self):
        with (
            patch("trading_common.data_clients.news.settings.newsapi_key.get_secret_value", return_value="fake-key"),
            patch("trading_common.data_clients.news.httpx.get", return_value=_mock_response(401)),
        ):
            result = NewsClient().fetch_headlines("AAPL")

        assert result.ok is False
        assert result.error.error_type is FetchErrorType.AUTH

    def test_429_returns_rate_limit_failure(self):
        with (
            patch("trading_common.data_clients.news.settings.newsapi_key.get_secret_value", return_value="fake-key"),
            patch("trading_common.data_clients.news.httpx.get", return_value=_mock_response(429)),
        ):
            result = NewsClient().fetch_headlines("AAPL")

        assert result.ok is False
        assert result.error.error_type is FetchErrorType.RATE_LIMIT

    def test_timeout_returns_timeout_failure(self):
        import httpx

        with (
            patch("trading_common.data_clients.news.settings.newsapi_key.get_secret_value", return_value="fake-key"),
            patch("trading_common.data_clients.news.httpx.get", side_effect=httpx.TimeoutException("timed out")),
        ):
            result = NewsClient().fetch_headlines("AAPL")

        assert result.ok is False
        assert result.error.error_type is FetchErrorType.TIMEOUT

    def test_fetch_async_wrapper_delegates_to_fetch_headlines(self):
        with (
            patch("trading_common.data_clients.news.settings.newsapi_key.get_secret_value", return_value=""),
        ):
            result = asyncio.run(NewsClient().fetch("AAPL"))

        assert result.ok is False
        assert result.error.error_type is FetchErrorType.AUTH
