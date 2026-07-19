# trading-common

Shared trading data-client/services library, extracted (D2, `INTEGRATION_PLAN.md`
Phase 1) from `daytrader`'s in-repo `trading_common/` package -- consumed by
`daytrader` today, and (in a separate migration session) `chanakya`.

## Contents

- `trading_common/data_clients/` -- market data, options chain (SPX-aware),
  Finnhub, Polygon, FRED, NewsAPI, FINRA short-interest, WSB/Reddit
  sentiment. All data-client methods return `Result[T]` (`data_clients.base`)
  so callers handle partial failures without exception propagation.
- `trading_common/features/` -- technical indicators, volatility, signal
  weights.
- `trading_common/pricing/` -- Black-Scholes pricing, fill-price modeling.
- `trading_common/services/` -- Schwab token encryption/refresh, generic
  multi-channel alert senders (email/Telegram).
- `trading_common/storage/` -- async SQLAlchemy engine/session factory and
  a generic repository pattern (its own minimal `Base` -- a consuming app
  defines its own domain models against its own `Base`/alembic wiring).
- `trading_common/monitoring/` -- per-source fetch tracking (success rate,
  latency, last error) for the `/health` surface a consuming app builds.
- `trading_common/config/settings.py` -- this package's OWN settings
  singleton, independent of (but reading the same process environment as)
  any consuming app's own extended settings.
- `trading_common/auth.py` -- password hashing (argon2) + JWT helpers.

**Not included** (daytrader-specific, coupled to its own `storage.models`/
`engine.*` and kept in daytrader's own repo instead):
`alert_service.py` (→ daytrader's `services/alert_service.py`),
`schwab_token_store.py` (→ daytrader's `services/schwab_token_store.py`),
`premarket_bars.py` (→ daytrader's `engine/premarket_bars.py`) -- each of
these persists against or reads daytrader's own ORM models/engine event
types, which don't exist in this package's namespace.

## Installing

**Local dev (editable, sibling checkout):**

```bash
pip install -e ../trading-common
# or, with uv, from the consuming app's own project:
uv pip install -e ../trading-common
```

**Pinned (Railway / CI), once this repo has a real git remote:**

```
trading-common @ git+https://github.com/<org>/trading-common.git@v1.0.0
```

Upgrades are deliberate version bumps (new git tag), tested per consuming
app -- never a silent floating dependency.

## Running the test suite standalone

```bash
pip install -e ".[dev]"
pytest
```

The suite never touches a real network (every `httpx`/`yfinance`/`requests`
call is mocked) and never requires a real Postgres instance (SQLite-backed
tests only, or pure unit tests with no DB at all).

## Versioning

Semantic version tags (`v1.0.0`, ...). A breaking change to any public
function/class signature is a major version bump.
