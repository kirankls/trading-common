"""News client wrapping NewsAPI for ticker-specific headlines."""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx

from trading_common.config.settings import settings
from trading_common.data_clients.base import FetchError, FetchErrorType, Result

_NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"


@dataclass
class NewsArticle:
    """A single news article with optional NLP-scored sentiment."""

    title: str
    url: str
    url_hash: str
    source: str
    published_at: str | None
    snippet: str | None
    sentiment_score: float | None = None
    sentiment_themes: list[str] = field(default_factory=list)


class NewsClient:
    """Fetches recent news headlines for a ticker via NewsAPI."""

    SOURCE = "newsapi"

    def fetch_headlines(
        self,
        ticker: str,
        days: int | None = None,
    ) -> Result[list[NewsArticle]]:
        """Fetch up to 100 recent news articles mentioning ticker.

        days defaults to settings.news_lookback_days when not provided.
        Returns Result.failure() if the NewsAPI key is missing or the request fails.
        """
        api_key = settings.newsapi_key.get_secret_value()
        if not api_key:
            return Result.failure(
                FetchError(
                    self.SOURCE,
                    FetchErrorType.AUTH,
                    "API key not configured",
                )
            )

        lookback_days = days if days is not None else settings.news_lookback_days
        from_date = (datetime.now(UTC) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        params = {
            "q": self._build_query(ticker),
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": api_key,
        }

        try:
            response = httpx.get(_NEWSAPI_BASE_URL, params=params, timeout=15.0)

            if response.status_code == 401:
                return Result.failure(
                    FetchError(self.SOURCE, FetchErrorType.AUTH, "Invalid NewsAPI key")
                )
            if response.status_code == 429:
                return Result.failure(
                    FetchError(
                        self.SOURCE,
                        FetchErrorType.RATE_LIMIT,
                        "NewsAPI rate limit exceeded",
                    )
                )
            response.raise_for_status()

            data = response.json()
            raw_articles = data.get("articles") or []

            articles: list[NewsArticle] = []
            for a in raw_articles:
                if not isinstance(a, dict):
                    continue
                url = a.get("url") or ""
                source_name = ""
                source_obj = a.get("source")
                if isinstance(source_obj, dict):
                    source_name = source_obj.get("name") or ""

                article = NewsArticle(
                    title=a.get("title") or "",
                    url=url,
                    url_hash=self._hash_url(url),
                    source=source_name,
                    published_at=a.get("publishedAt"),
                    snippet=a.get("description"),
                )
                articles.append(article)

            return Result.success(articles)

        except httpx.TimeoutException as e:
            return Result.failure(
                FetchError(self.SOURCE, FetchErrorType.TIMEOUT, str(e))
            )
        except Exception as e:
            return Result.failure(
                FetchError(self.SOURCE, FetchErrorType.UNAVAILABLE, str(e))
            )

    def _build_query(self, ticker: str) -> str:
        """Build a NewsAPI q-string for the given ticker symbol."""
        return f'"{ticker}" stock OR options'

    def _hash_url(self, url: str) -> str:
        """Return the SHA-256 hex digest of a URL for deduplication."""
        return hashlib.sha256(url.encode()).hexdigest()

    async def fetch(self, ticker: str) -> Result[list[NewsArticle]]:
        """Async entry point: run fetch_headlines in the executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.fetch_headlines, ticker)
