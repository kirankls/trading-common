"""Unit tests for trading_common.data_clients.fred.get_risk_free_rate.

Rules:
  - No network calls — httpx is always mocked.
  - No API key required to run these tests.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_common.data_clients.fred import get_risk_free_rate


class TestGetRiskFreeRate:
    def test_returns_default_when_no_api_key(self):
        """No FRED API key configured → returns the 0.05 fallback, never raises."""
        with (
            patch("trading_common.data_clients.fred._api_key", return_value=""),
            patch("trading_common.data_clients.fred._cache_get", return_value=None),
            patch("trading_common.data_clients.fred._cache_set"),
        ):
            result = asyncio.run(get_risk_free_rate(45))
        assert result == 0.05

    def test_parses_successful_response_and_caches(self):
        """A successful FRED response is parsed to a decimal fraction and cached.

        DGS3MO is quoted in percentage points (e.g. 4.53 == 4.53%), so the
        returned float must be that value divided by 100.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"date": "2026-07-02", "value": "4.53"},
                {"date": "2026-07-01", "value": "4.52"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_cm = MagicMock()
        mock_client_cm.__aenter__.return_value = mock_client
        mock_client_cm.__aexit__.return_value = None

        cache_store: dict = {}

        def _fake_cache_get(key):
            return cache_store.get(key)

        def _fake_cache_set(key, value, ttl):
            cache_store[key] = value

        with (
            patch("trading_common.data_clients.fred._api_key", return_value="fake-key"),
            patch("trading_common.data_clients.fred._cache_get", side_effect=_fake_cache_get),
            patch("trading_common.data_clients.fred._cache_set", side_effect=_fake_cache_set),
            patch("trading_common.data_clients.fred.httpx.AsyncClient", return_value=mock_client_cm) as mock_ac,
        ):
            result1 = asyncio.run(get_risk_free_rate(45))
            result2 = asyncio.run(get_risk_free_rate(45))

        assert result1 == pytest.approx(0.0453)
        assert result2 == pytest.approx(0.0453)
        # Second call must be served from cache — httpx.AsyncClient constructed once.
        assert mock_ac.call_count == 1

    def test_returns_default_and_does_not_raise_on_timeout(self):
        """An HTTP timeout/exception is swallowed; falls back to 0.05."""
        with (
            patch("trading_common.data_clients.fred._api_key", return_value="fake-key"),
            patch("trading_common.data_clients.fred._cache_get", return_value=None),
            patch("trading_common.data_clients.fred._cache_set"),
            patch(
                "trading_common.data_clients.fred.httpx.AsyncClient",
                side_effect=RuntimeError("connection refused"),
            ),
        ):
            result = asyncio.run(get_risk_free_rate(45))
        assert result == 0.05

    def test_accepts_per_user_api_key_override(self):
        """A per-user api_key overrides the platform default without touching settings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [{"date": "2026-07-02", "value": "5.00"}]
        }
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_cm = MagicMock()
        mock_client_cm.__aenter__.return_value = mock_client
        mock_client_cm.__aexit__.return_value = None

        with (
            patch("trading_common.data_clients.fred._cache_get", return_value=None),
            patch("trading_common.data_clients.fred._cache_set"),
            patch("trading_common.data_clients.fred.httpx.AsyncClient", return_value=mock_client_cm),
            patch("trading_common.data_clients.fred._api_key") as mock_platform_key,
        ):
            result = asyncio.run(get_risk_free_rate(45, api_key="user-supplied-key"))

        mock_platform_key.assert_not_called()
        assert result == pytest.approx(0.05)
