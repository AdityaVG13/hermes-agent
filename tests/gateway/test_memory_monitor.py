"""Tests for gateway.memory_monitor — periodic process memory logging.

Ported from cline/cline#10343.  The module logs a structured
``[MEMORY] rss=...MB ...`` line periodically so long-running gateway
leaks show up as a time series in agent.log / gateway.log.
"""

from __future__ import annotations


import pytest

from gateway import memory_monitor as mm


@pytest.fixture(autouse=True)
def _ensure_monitor_stopped():
    """Every test starts from a clean state and leaves one behind."""
    mm.stop_memory_monitoring(timeout=1.0)
    yield
    mm.stop_memory_monitoring(timeout=1.0)






def test_double_start_is_noop():
    assert mm.start_memory_monitoring(interval_seconds=3600.0) is True
    assert mm.start_memory_monitoring(interval_seconds=3600.0) is False
    assert mm.is_running() is True


def test_stop_ends_monitoring():
    mm.start_memory_monitoring(interval_seconds=3600.0)
    mm.stop_memory_monitoring(timeout=1.0)
    assert mm.is_running() is False


def test_rss_is_current_not_the_high_water_mark(monkeypatch):
    """ru_maxrss only ever rises, so reporting it as ``rss`` hid every release."""
    import resource

    class _StandingPeak:
        ru_maxrss = 64 * 1024 * 1024 * 1024  # 64 GiB peak already reached

    monkeypatch.setattr(resource, "getrusage", lambda *_a, **_k: _StandingPeak())
    rss = mm._get_rss_mb()
    if rss is None:
        pytest.skip("no current-RSS source on this host")
    assert isinstance(rss, int) and rss < 1024 * 1024, (
        "rss must be the current resident size, not a high-water mark")


def test_log_line_separates_current_rss_from_peak(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="gateway.memory_monitor"):
        mm.log_memory_usage(prefix="baseline")
    message = next(r.getMessage() for r in caplog.records if "[MEMORY]" in r.getMessage())
    assert "rss=" in message and "peak=" in message, message


