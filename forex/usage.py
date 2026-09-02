"""Persistent Twelve Data API usage accounting.

The Basic plan resets at midnight UTC.  We retain prior days in SQLite and
compute the current counter by UTC date, so a new day starts at zero without
destroying history.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class UsageSummary:
    day_utc: str
    requests: int
    estimated_credits: int
    daily_limit: int
    remaining_estimate: int
    latest_header_used: Optional[int]
    latest_header_left: Optional[int]
    failures: int
    official_daily_usage: Optional[int]
    official_daily_limit: Optional[int]
    official_minute_usage: Optional[int]
    official_minute_limit: Optional[int]
    official_snapshot_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class UsageTracker:
    def __init__(self, path: str, daily_limit: int = 800):
        self.path = str(path)
        self.daily_limit = max(int(daily_limit), 1)

    def _connect(self) -> sqlite3.Connection:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS api_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                day_utc TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                status_code INTEGER,
                estimated_credits INTEGER NOT NULL,
                header_used INTEGER,
                header_left INTEGER,
                success INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS quota_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                day_utc TEXT NOT NULL,
                daily_usage INTEGER,
                daily_limit INTEGER,
                minute_usage INTEGER,
                minute_limit INTEGER,
                plan_category TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_api_requests_day ON api_requests(day_utc)"
        )
        return connection

    def record(
        self,
        *,
        endpoint: str,
        symbol: str,
        interval: str,
        status_code: Optional[int],
        headers: Optional[Dict[str, Any]] = None,
        success: bool,
        estimated_credits: int = 1,
        now: Optional[datetime] = None,
    ) -> None:
        timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        headers = headers or {}
        header_used = _as_int(headers.get("api-credits-used"))
        header_left = _as_int(headers.get("api-credits-left"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO api_requests (
                    ts_utc, day_utc, endpoint, symbol, interval, status_code,
                    estimated_credits, header_used, header_left, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp.isoformat(timespec="seconds"),
                    timestamp.date().isoformat(),
                    endpoint,
                    symbol,
                    interval,
                    status_code,
                    max(int(estimated_credits), 0),
                    header_used,
                    header_left,
                    1 if success else 0,
                ),
            )

    def record_snapshot(
        self,
        *,
        daily_usage: Optional[int],
        daily_limit: Optional[int],
        minute_usage: Optional[int],
        minute_limit: Optional[int],
        plan_category: str = "",
        now: Optional[datetime] = None,
    ) -> None:
        timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO quota_snapshots (
                    ts_utc, day_utc, daily_usage, daily_limit,
                    minute_usage, minute_limit, plan_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp.isoformat(timespec="seconds"),
                    timestamp.date().isoformat(),
                    _as_int(daily_usage),
                    _as_int(daily_limit),
                    _as_int(minute_usage),
                    _as_int(minute_limit),
                    str(plan_category or ""),
                ),
            )

    def summary(self, day_utc: Optional[str] = None) -> UsageSummary:
        target = day_utc or datetime.now(timezone.utc).date().isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(estimated_credits), 0),
                       COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0)
                FROM api_requests WHERE day_utc = ?
                """,
                (target,),
            ).fetchone()
            latest = connection.execute(
                """
                SELECT header_used, header_left FROM api_requests
                WHERE day_utc = ? AND (header_used IS NOT NULL OR header_left IS NOT NULL)
                ORDER BY id DESC LIMIT 1
                """,
                (target,),
            ).fetchone()
            snapshot = connection.execute(
                """
                SELECT ts_utc, daily_usage, daily_limit, minute_usage, minute_limit
                FROM quota_snapshots WHERE day_utc = ? ORDER BY id DESC LIMIT 1
                """,
                (target,),
            ).fetchone()

        requests, credits, failures = (int(value or 0) for value in row)
        used = latest[0] if latest else None
        left = latest[1] if latest else None
        return UsageSummary(
            day_utc=target,
            requests=requests,
            estimated_credits=credits,
            daily_limit=self.daily_limit,
            remaining_estimate=max(self.daily_limit - credits, 0),
            latest_header_used=used,
            latest_header_left=left,
            failures=failures,
            official_daily_usage=snapshot[1] if snapshot else None,
            official_daily_limit=snapshot[2] if snapshot else None,
            official_minute_usage=snapshot[3] if snapshot else None,
            official_minute_limit=snapshot[4] if snapshot else None,
            official_snapshot_at=snapshot[0] if snapshot else None,
        )

    def history(self, days: int = 7) -> Iterable[UsageSummary]:
        today = datetime.now(timezone.utc).date()
        with self._connect() as connection:
            values = connection.execute(
                "SELECT DISTINCT day_utc FROM api_requests ORDER BY day_utc DESC LIMIT ?",
                (max(int(days), 1),),
            ).fetchall()
        dates = [row[0] for row in values]
        if today.isoformat() not in dates:
            dates.insert(0, today.isoformat())
        return [self.summary(value) for value in dates[:days]]


def tracker_from_env() -> Optional[UsageTracker]:
    path = os.getenv("TWELVEDATA_USAGE_DB", "").strip()
    if not path:
        return None
    try:
        limit = int(os.getenv("TWELVEDATA_DAILY_LIMIT", "800"))
    except ValueError:
        limit = 800
    return UsageTracker(path, daily_limit=limit)
