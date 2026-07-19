"""Macro data client wrapping fredapi for Fed funds rate, CPI, and event calendar."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import yfinance as yf

from trading_common.config.settings import settings
from trading_common.data_clients.base import FetchError, FetchErrorType, Result


@dataclass
class MacroEvent:
    """An upcoming macro event (FOMC, CPI, NFP, etc.)."""

    name: str
    date: str
    description: str | None


@dataclass
class MacroContext:
    """Aggregated macro-economic context for inclusion in analysis prompts."""

    fed_funds_rate: float | None
    cpi_yoy: float | None
    ten_year_yield: float | None
    vix: float | None
    upcoming_events: list[MacroEvent] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# Known FOMC meeting dates for 2025 and 2026 (published annually by the Fed)
_FOMC_DATES_2025 = [
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
]

_FOMC_DATES_2026 = [
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
]

_FOMC_DATES_2027 = [
    "2027-01-27",
    "2027-03-17",
    "2027-04-28",
    "2027-06-16",
    "2027-07-28",
    "2027-09-15",
    "2027-10-27",
    "2027-12-15",
]


def _next_nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence (1-based) of weekday (0=Mon … 6=Sun) in year/month."""
    first = date(year, month, 1)
    # day-of-week of the first of the month
    first_wd = first.weekday()
    delta = (weekday - first_wd) % 7
    first_occurrence = first + timedelta(days=delta)
    return first_occurrence + timedelta(weeks=n - 1)


def _upcoming_cpi_dates(count: int = 3) -> list[date]:
    """CPI is typically released on the second or third Tuesday each month.

    The BLS releases CPI around the 10th–15th of the month; empirically this
    falls on the second or third Tuesday.  We approximate with the second
    Tuesday of each month — close enough for advisory purposes.
    """
    results: list[date] = []
    today = date.today()
    year, month = today.year, today.month

    while len(results) < count:
        candidate = _next_nth_weekday_of_month(year, month, 1, 2)  # 2nd Tuesday
        if candidate >= today:
            results.append(candidate)
        # Advance month
        month += 1
        if month > 12:
            month = 1
            year += 1

    return results


def _upcoming_nfp_dates(count: int = 3) -> list[date]:
    """NFP (Non-Farm Payrolls) is released on the first Friday of each month."""
    results: list[date] = []
    today = date.today()
    year, month = today.year, today.month

    while len(results) < count:
        candidate = _next_nth_weekday_of_month(year, month, 4, 1)  # 1st Friday
        if candidate >= today:
            results.append(candidate)
        month += 1
        if month > 12:
            month = 1
            year += 1

    return results


class MacroClient:
    """Fetches macro-economic context from FRED and other free sources."""

    SOURCE = "fred"

    def fetch_context(self) -> Result[MacroContext]:
        """Fetch current Fed funds rate, CPI, 10-year yield, VIX, and upcoming events.

        VIX and upcoming events are always fetched. FRED-dependent fields
        (fed_funds_rate, cpi_yoy, ten_year_yield) are fetched only when the FRED
        API key is configured — missing key yields None for those fields, not failure.
        """
        try:
            fred_key = settings.fred_api_key.get_secret_value()

            if fred_key:
                fed_funds_rate = self._fetch_fred_series("FEDFUNDS")
                cpi_yoy = self._fetch_cpi_yoy(fred_key)
                ten_year_yield = self._fetch_fred_series("GS10")
            else:
                fed_funds_rate = None
                cpi_yoy = None
                ten_year_yield = None

            vix = self._fetch_vix()
            upcoming_events = self._fetch_upcoming_events()

            context = MacroContext(
                fed_funds_rate=fed_funds_rate,
                cpi_yoy=cpi_yoy,
                ten_year_yield=ten_year_yield,
                vix=vix,
                upcoming_events=upcoming_events,
            )
            return Result.success(context)

        except Exception as e:
            return Result.failure(
                FetchError(self.SOURCE, FetchErrorType.UNAVAILABLE, str(e))
            )

    def _fetch_fred_series(self, series_id: str) -> float | None:
        """Return the most recent value for a FRED series, or None on error."""
        fred_key = settings.fred_api_key.get_secret_value()
        if not fred_key:
            return None
        try:
            from fredapi import Fred  # type: ignore[import]

            fred = Fred(api_key=fred_key)
            series = fred.get_series(series_id)
            if series is None or series.empty:
                return None
            val = series.iloc[-1]
            return float(val) if val is not None else None
        except Exception:
            return None

    def _fetch_cpi_yoy(self, fred_key: str) -> float | None:
        """Fetch CPI YoY percentage change from FRED."""
        try:
            from fredapi import Fred  # type: ignore[import]

            fred = Fred(api_key=fred_key)
            series = fred.get_series("CPIAUCSL")
            if series is None or len(series) < 13:
                return None
            yoy = series.pct_change(12).iloc[-1] * 100
            return float(yoy) if yoy is not None else None
        except Exception:
            return None

    def _fetch_vix(self) -> float | None:
        """Fetch current VIX level via yfinance."""
        try:
            info = yf.Ticker("^VIX").info
            val = info.get("regularMarketPrice")
            return float(val) if val is not None else None
        except Exception:
            return None

    def _fetch_fomc_dates_live(self) -> list[str] | None:
        try:
            import requests
            response = requests.get(
                "https://www.federalreserve.gov/monetarypolicy/fomccalendars.json",
                timeout=5,
            )
            raw = response.text
            dates = sorted(set(re.findall(r"\d{4}-\d{2}-\d{2}", raw)))
            return dates if dates else None
        except Exception:
            return None

    def _fetch_upcoming_events(self) -> list[MacroEvent]:
        """Return upcoming FOMC, CPI, and NFP event dates."""
        today_str = date.today().isoformat()
        events: list[MacroEvent] = []

        live_fomc = self._fetch_fomc_dates_live()
        fomc_dates = live_fomc if live_fomc else (_FOMC_DATES_2025 + _FOMC_DATES_2026 + _FOMC_DATES_2027)
        for d in [d for d in fomc_dates if d >= today_str]:
            events.append(
                MacroEvent(
                    name="FOMC Meeting",
                    date=d,
                    description="Federal Open Market Committee interest rate decision",
                )
            )

        # Next 3 CPI release dates
        for cpi_date in _upcoming_cpi_dates(3):
            events.append(
                MacroEvent(
                    name="CPI Release",
                    date=cpi_date.isoformat(),
                    description="Bureau of Labor Statistics Consumer Price Index release",
                )
            )

        # Next 3 NFP release dates
        for nfp_date in _upcoming_nfp_dates(3):
            events.append(
                MacroEvent(
                    name="Non-Farm Payrolls",
                    date=nfp_date.isoformat(),
                    description="Bureau of Labor Statistics Non-Farm Payrolls jobs report",
                )
            )

        # Sort all events chronologically
        events.sort(key=lambda e: e.date)
        return events

    async def fetch(self) -> Result[MacroContext]:
        """Async entry point: run fetch_context in the executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.fetch_context)
