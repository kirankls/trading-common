# Ported from D:\chanakya\options_advisor\features\signal_weight_loader.py
"""Load closed trade history with signal context for weight computation.

Design notes
------------
TradeRepository is async-only and is expected to be called by the strategy
engine to load closed trades for the signal-weighting block. To avoid a
second DB round-trip we expose two helpers:

  build_trade_dicts_for_weighting(trades)
      Pure function.  Accepts an already-loaded list of trade-log ORM objects
      (or any objects exposing the same attributes) and returns list[dict]
      using only columns stored on the trade log itself.  No
      signal_snapshot_json is available without a joined analysis-history
      row, so IV-rank / ADX / VIX fields will be None for trades that lack a
      snapshot (older history).  The weight engine handles missing fields
      gracefully via its _MIN_TRADES_FOR_WEIGHT guard.

  enrich_with_snapshots(trade_dicts, db)  [async]
      Optional enrichment pass.  Given the flat dicts produced above plus an
      AsyncSession, fetches signal_snapshot_json from analysis history for
      any trade that carries an analysis_id and backfills entry_iv_rank,
      entry_adx, entry_vix, entry_trend_bullish, and had_earnings_proximity.
      Call this when you want fully-personalised weights; skip it for speed.

Typical usage
-------------
  from trading_common.features.signal_weight_loader import (
      build_trade_dicts_for_weighting,
      enrich_with_snapshots,
  )

  # _closed_trades already loaded by the strategy-performance block
  _weight_dicts = build_trade_dicts_for_weighting(_closed_trades)

  # Optional: enrich with snapshot data (adds one SELECT per batch of trades)
  async with _get_session_factory()() as _sw_db:
      _weight_dicts = await enrich_with_snapshots(_weight_dicts, _sw_db)

  _signal_weights = compute_signal_weights(_weight_dicts)

DayTrader adaptation note
--------------------------
The original chanakya module imported ``options_advisor.storage.models``
(SQLAlchemy AnalysisHistory ORM model) directly inside enrich_with_snapshots().
That import is rewritten below to `trading_common.storage.repositories` —
a sibling agent is porting the repository pattern there in parallel. This
module expects that package to eventually expose an analysis-history model
readable via a SQLAlchemy `select()` with at minimum the columns
`analysis_id` and `signal_snapshot_json` (see the try/except import inside
enrich_with_snapshots for the exact expected shape). Until that lands, any
call to enrich_with_snapshots() will hit the except-branch and return the
trade_dicts unenriched (non-fatal, matches original behaviour on DB failure).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_log = logging.getLogger(__name__)


def build_trade_dicts_for_weighting(trades: list[Any]) -> list[dict]:
    """
    Convert a list of trade-log ORM objects into flat dicts for weight computation.

    Uses only the fields present on the trade log itself:
      - pnl_dollar
      - strategy
      - analysis_id  (carried through so enrich_with_snapshots can JOIN)

    All signal-context fields (entry_iv_rank, entry_adx, entry_vix,
    entry_trend_bullish, had_earnings_proximity) default to None and are
    populated by a subsequent call to enrich_with_snapshots() when available.

    Parameters
    ----------
    trades:
        list of trade-log ORM objects (or any objects with the same attributes).
        May be empty — returns an empty list.

    Returns
    -------
    list[dict] ready for compute_signal_weights().
    """
    result: list[dict] = []
    for trade in trades:
        row: dict[str, Any] = {
            "pnl_dollar": _safe_float(getattr(trade, "pnl_dollar", None)),
            "strategy": getattr(trade, "strategy", None),
            "analysis_id": getattr(trade, "analysis_id", None),
            # Signal-context fields — populated by enrich_with_snapshots if available
            "entry_iv_rank": None,
            "entry_adx": None,
            "entry_vix": None,
            "entry_trend_bullish": None,
            "had_earnings_proximity": None,
        }
        result.append(row)
    return result


async def enrich_with_snapshots(
    trade_dicts: list[dict],
    db: AsyncSession,
    analysis_history_model: Any = None,
) -> list[dict]:
    """
    Back-fill signal-context fields from analysis_history.signal_snapshot_json.

    For each trade_dict that carries a non-None ``analysis_id``, fetches the
    corresponding analysis-history row and extracts:
      - entry_iv_rank   from snapshot["volatility"]["iv_rank_52wk"]
      - entry_adx       from snapshot["technical"]["adx_14"]
      - entry_vix       from snapshot["macro"]["vix"]
      - entry_trend_bullish  True when RSI > 50 and MACD > 0
      - had_earnings_proximity  True when earnings.days_away <= 21 at entry

    Rows with no analysis_id, missing snapshot, or malformed JSON are left as
    None — the weight engine treats them as missing and skips them.

    Parameters
    ----------
    trade_dicts:
        Output of build_trade_dicts_for_weighting().
    db:
        An open AsyncSession.  The caller owns the session lifetime.
    analysis_history_model:
        Optional SQLAlchemy model exposing ``analysis_id`` and
        ``signal_snapshot_json`` columns. Pass this explicitly when your own
        app already has an analysis-history model (e.g. chanakya's
        ``options_advisor.storage.models.AnalysisHistory``) so this function
        doesn't have to guess an import path. When omitted, falls back to
        attempting ``trading_common.storage.repositories.AnalysisHistory``
        (daytrader's own future model, not yet ported as of this writing).

    Returns
    -------
    The same list, mutated in-place (and returned for convenience).
    """
    if not trade_dicts:
        return trade_dicts

    # Collect unique analysis_ids that are non-None
    ids_needed = list(
        {d["analysis_id"] for d in trade_dicts if d.get("analysis_id") is not None}
    )
    if not ids_needed:
        return trade_dicts

    # Batch-fetch signal_snapshot_json for all relevant analysis rows
    snapshot_map: dict[str, dict] = {}
    # AnalysisHistory is options-advisor-specific and intentionally not ported
    # (see module docstring) — callers that already have their own model
    # (chanakya) should pass it via analysis_history_model. The default
    # import path below is daytrader's own future model and is expected to
    # fail at runtime until/unless a later milestone adds it; the except
    # branch handles that as a non-fatal fallback either way.
    try:
        from sqlalchemy import select

        if analysis_history_model is not None:
            AnalysisHistory = analysis_history_model
        else:
            from trading_common.storage.repositories import (  # type: ignore[attr-defined]
                AnalysisHistory,
            )

        stmt = (
            select(AnalysisHistory.analysis_id, AnalysisHistory.signal_snapshot_json)
            .where(AnalysisHistory.analysis_id.in_(ids_needed))
        )
        result = await db.execute(stmt)
        for analysis_id, snap_json in result:
            if not snap_json:
                continue
            try:
                parsed = json.loads(snap_json) if isinstance(snap_json, str) else snap_json
                snapshot_map[analysis_id] = parsed if isinstance(parsed, dict) else {}
            except Exception:
                pass  # malformed JSON — skip
    except Exception as exc:
        _log.warning("enrich_with_snapshots: DB fetch failed (%s) — returning unenriched dicts", exc)
        return trade_dicts

    # Back-fill each trade_dict from its snapshot
    for row in trade_dicts:
        aid = row.get("analysis_id")
        if aid is None or aid not in snapshot_map:
            continue
        snap = snapshot_map[aid]
        try:
            vol = snap.get("volatility") or {}
            tech = snap.get("technical") or {}
            macro = snap.get("macro") or {}
            earnings = snap.get("earnings") or {}

            row["entry_iv_rank"] = _safe_float(vol.get("iv_rank_52wk"))
            row["entry_adx"] = _safe_float(tech.get("adx_14"))
            row["entry_vix"] = _safe_float(macro.get("vix"))

            # Trend bullish: RSI > 50 AND MACD > 0
            rsi = _safe_float(tech.get("rsi_14"))
            macd = _safe_float(tech.get("macd"))
            if rsi is not None and macd is not None:
                row["entry_trend_bullish"] = rsi > 50 and macd > 0

            # Earnings proximity flag
            days_away = earnings.get("days_away")
            if days_away is not None:
                try:
                    row["had_earnings_proximity"] = int(days_away) <= 21
                except (TypeError, ValueError):
                    pass
        except Exception as exc:
            _log.debug("enrich_with_snapshots: failed to parse snapshot for %s: %s", aid, exc)

    return trade_dicts


def _safe_float(value: Any) -> float | None:
    """Return float(value) or None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
