"""Unit tests for trading_common.monitoring.tracker.SourceTracker.

Exercises the healthy -> degraded -> down status cascade and the
rate_limited / timeout / stale / unknown classifications, matching the
thresholds actually implemented in SourceTracker.status_dict():
  - error_rate >= 0.8 (of events within the last hour) -> "down"
  - error_rate >= 0.3                                   -> "degraded"
  - else if last 5 errors mention 429/rate               -> "rate_limited"
  - else if last 5 errors mention timeout                -> "timeout"
  - else                                                  -> "healthy"
  - no events at all -> "unknown"
  - events exist but none within the last hour -> "stale"
"""
from __future__ import annotations

from trading_common.monitoring.tracker import SourceTracker, all_statuses, get_tracker


def test_unknown_status_with_no_events():
    t = SourceTracker("test-src")
    status = t.status_dict()
    assert status["status"] == "unknown"
    assert status["last_success"] is None
    assert status["total_calls_1h"] == 0


def test_healthy_status_all_successes():
    t = SourceTracker("test-src")
    for _ in range(10):
        t.record(success=True, latency_ms=50.0)
    status = t.status_dict()
    assert status["status"] == "healthy"
    assert status["error_rate_pct"] == 0.0
    assert status["total_calls_1h"] == 10
    assert status["last_success"] is not None


def test_degraded_status_at_30_percent_errors():
    t = SourceTracker("test-src")
    # 3 errors out of 10 = 30% -> degraded (generic errors, not rate/timeout)
    for _ in range(7):
        t.record(success=True, latency_ms=50.0)
    for _ in range(3):
        t.record(success=False, latency_ms=100.0, error="server error 500")
    status = t.status_dict()
    assert status["status"] == "degraded"
    assert status["error_rate_pct"] == 30.0


def test_down_status_at_80_percent_errors():
    t = SourceTracker("test-src")
    for _ in range(2):
        t.record(success=True, latency_ms=50.0)
    for _ in range(8):
        t.record(success=False, latency_ms=100.0, error="server error 500")
    status = t.status_dict()
    assert status["status"] == "down"
    assert status["error_rate_pct"] == 80.0


def test_rate_limited_classification_below_degraded_threshold():
    t = SourceTracker("test-src")
    # Keep error rate < 30% so it doesn't get classified as degraded/down first,
    # but the most recent errors mention 429 -> rate_limited
    for _ in range(9):
        t.record(success=True, latency_ms=50.0)
    t.record(success=False, latency_ms=100.0, error="429 Too Many Requests", status_code=429)
    status = t.status_dict()
    assert status["status"] == "rate_limited"


def test_timeout_classification_below_degraded_threshold():
    t = SourceTracker("test-src")
    for _ in range(9):
        t.record(success=True, latency_ms=50.0)
    t.record(success=False, latency_ms=5000.0, error="Request timed out")
    status = t.status_dict()
    assert status["status"] == "timeout"


def test_stale_status_when_events_exist_but_none_recent():
    t = SourceTracker("test-src")
    # Manually inject an old event (older than the 3600s recency window)
    import time

    from trading_common.monitoring.tracker import SourceEvent

    old_event = SourceEvent(timestamp=time.time() - 7200, success=True, latency_ms=10.0)
    t._events.append(old_event)
    status = t.status_dict()
    assert status["status"] == "stale"
    assert status["total_calls_1h"] == 0


def test_recovery_from_down_to_healthy():
    t = SourceTracker("test-src")
    for _ in range(10):
        t.record(success=False, latency_ms=100.0, error="server error 500")
    assert t.status_dict()["status"] == "down"

    # Recovery: enough new successes pushes the window's error rate back down.
    for _ in range(40):
        t.record(success=True, latency_ms=50.0)
    status = t.status_dict()
    assert status["status"] == "healthy"
    assert status["error_rate_pct"] < 30.0


def test_max_events_ring_buffer_eviction():
    t = SourceTracker("test-src", max_events=5)
    for i in range(10):
        t.record(success=True, latency_ms=float(i))
    assert len(t._events) == 5
    # Only the last 5 records remain (latencies 5..9)
    assert [e.latency_ms for e in t._events] == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_error_truncated_to_300_chars():
    t = SourceTracker("test-src")
    long_error = "x" * 500
    t.record(success=False, latency_ms=10.0, error=long_error)
    assert len(t._events[-1].error) == 300


def test_get_tracker_registry_singleton():
    t1 = get_tracker("shared-source")
    t2 = get_tracker("shared-source")
    assert t1 is t2


def test_all_statuses_reports_registered_trackers():
    get_tracker("another-source").record(success=True, latency_ms=1.0)
    statuses = all_statuses()
    assert "another-source" in statuses
    assert statuses["another-source"]["status"] == "healthy"


def test_recent_errors_capped_at_ten_and_last_error_set():
    t = SourceTracker("test-src")
    for i in range(15):
        t.record(success=False, latency_ms=1.0, error=f"error-{i}")
    status = t.status_dict()
    assert len(status["recent_errors"]) == 10
    assert status["last_error"] == "error-14"
