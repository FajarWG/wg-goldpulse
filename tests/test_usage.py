from datetime import datetime, timezone

from forex.usage import UsageTracker


def test_usage_is_grouped_by_utc_day_and_history_is_retained(tmp_path):
    tracker = UsageTracker(str(tmp_path / "usage.sqlite3"), daily_limit=800)
    tracker.record(
        endpoint="/time_series",
        symbol="XAU/USD",
        interval="5min",
        status_code=200,
        headers={"api-credits-used": "1", "api-credits-left": "7"},
        success=True,
        now=datetime(2026, 8, 31, 23, 59, tzinfo=timezone.utc),
    )
    tracker.record(
        endpoint="/time_series",
        symbol="XAU/USD",
        interval="1h",
        status_code=429,
        headers={},
        success=False,
        now=datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc),
    )

    first = tracker.summary("2026-08-31")
    second = tracker.summary("2026-09-01")
    assert first.estimated_credits == 1
    assert first.latest_header_left == 7
    assert second.estimated_credits == 1
    assert second.requests == 1
    assert second.failures == 1
    assert second.remaining_estimate == 799


def test_empty_day_starts_at_zero(tmp_path):
    tracker = UsageTracker(str(tmp_path / "usage.sqlite3"), daily_limit=800)
    summary = tracker.summary("2030-01-01")
    assert summary.requests == 0
    assert summary.estimated_credits == 0
    assert summary.remaining_estimate == 800


def test_official_quota_snapshot_is_exposed(tmp_path):
    tracker = UsageTracker(str(tmp_path / "usage.sqlite3"), daily_limit=800)
    now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    tracker.record_snapshot(
        daily_usage=123,
        daily_limit=800,
        minute_usage=4,
        minute_limit=8,
        plan_category="basic",
        now=now,
    )
    summary = tracker.summary("2026-08-31")
    assert summary.official_daily_usage == 123
    assert summary.official_daily_limit == 800
    assert summary.official_minute_usage == 4
    assert summary.official_minute_limit == 8
