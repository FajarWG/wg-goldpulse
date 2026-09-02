#!/usr/bin/env python3
"""Print today's Twelve Data usage and recent UTC-day history."""

from __future__ import annotations

import os

from forex.usage import UsageTracker


def main() -> int:
    path = os.getenv("TWELVEDATA_USAGE_DB", "/var/lib/xauusd-analysis/usage.sqlite3")
    limit = int(os.getenv("TWELVEDATA_DAILY_LIMIT", "800"))
    tracker = UsageTracker(path, daily_limit=limit)
    print("day_utc    account     bot     requests   failures   bot_remaining")
    for item in tracker.history(7):
        account = (
            f"{item.official_daily_usage}/{item.official_daily_limit}"
            if item.official_daily_usage is not None
            else "n/a"
        )
        print(
            f"{item.day_utc}  {account:>9}   {item.estimated_credits:>3}/{item.daily_limit:<3}"
            f"   {item.requests:>8}   {item.failures:>8}   {item.remaining_estimate:>13}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
