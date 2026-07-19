# Ported from D:\chanakya\options_advisor\monitoring\tracker.py
"""Process-local ring-buffer tracker for data source health monitoring."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class SourceEvent:
    timestamp: float
    success: bool
    latency_ms: float
    error: str | None = None
    status_code: int | None = None


class SourceTracker:
    def __init__(self, name: str, max_events: int = 200):
        self.name = name
        self._events: deque[SourceEvent] = deque(maxlen=max_events)

    def record(self, success: bool, latency_ms: float,
               error: str | None = None, status_code: int | None = None) -> None:
        self._events.append(SourceEvent(
            timestamp=time.time(),
            success=success,
            latency_ms=latency_ms,
            error=error[:300] if error else None,
            status_code=status_code,
        ))

    def status_dict(self) -> dict:
        events = list(self._events)
        now = time.time()
        recent = [e for e in events if now - e.timestamp < 3600]
        if not events:
            return {
                "status": "unknown", "last_success": None, "avg_latency_ms": None,
                "error_rate_pct": None, "total_calls_1h": 0,
                "recent_errors": [], "last_error": None,
            }
        error_recent   = [e for e in recent if not e.success]
        error_rate     = len(error_recent) / len(recent) if recent else 0

        # Classify status
        if not recent:
            status = "stale"
        elif error_rate >= 0.8:
            status = "down"
        elif error_rate >= 0.3:
            status = "degraded"
        elif any("429" in (e.error or "") or "rate" in (e.error or "").lower() for e in error_recent[-5:]):
            status = "rate_limited"
        elif any("timeout" in (e.error or "").lower() or "timed out" in (e.error or "").lower() for e in error_recent[-5:]):
            status = "timeout"
        else:
            status = "healthy"

        last_success = next((e.timestamp for e in reversed(events) if e.success), None)
        avg_lat = sum(e.latency_ms for e in recent) / len(recent) if recent else None

        return {
            "status": status,
            "last_success": last_success,
            "avg_latency_ms": round(avg_lat, 1) if avg_lat else None,
            "error_rate_pct": round(error_rate * 100, 1),
            "total_calls_1h": len(recent),
            "recent_errors": [
                {
                    "time": e.timestamp,
                    "error": e.error,
                    "status_code": e.status_code,
                    "latency_ms": round(e.latency_ms, 1),
                }
                for e in reversed(list(self._events)[-50:])
                if not e.success
            ][:10],
            "last_error": error_recent[-1].error if error_recent else None,
        }


# Global tracker registry
_registry: dict[str, SourceTracker] = {}


def get_tracker(name: str) -> SourceTracker:
    if name not in _registry:
        _registry[name] = SourceTracker(name)
    return _registry[name]


def all_statuses() -> dict[str, dict]:
    return {name: t.status_dict() for name, t in _registry.items()}
