#!/usr/bin/env python3
"""Run or read the WG GoldPulse historical backtest."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from forex.backtest import (
    format_backtest,
    load_backtest_data,
    run_backtest,
    save_backtest,
)
from forex.config import Config
from forex.notify import send_telegram
from forex.product import CURRENT_STRATEGY_VERSION


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refresh cached candles from Twelve Data")
    parser.add_argument("--notify", action="store_true", help="send the result to Telegram")
    parser.add_argument(
        "--strategy",
        choices=("v1", "v2", "v3", "v3.1", "v4", CURRENT_STRATEGY_VERSION, "all"),
        default=CURRENT_STRATEGY_VERSION,
    )
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--output-label", default="")
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="refresh/cache candles without replaying a strategy",
    )
    args = parser.parse_args()
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    backtest_dir = state_dir / "backtest"
    frames, fetched = load_backtest_data(
        backtest_dir / "cache",
        refresh=args.refresh,
        lookback_days=args.lookback_days or int(os.getenv("BACKTEST_LOOKBACK_DAYS", "90")),
    )
    if args.fetch_only:
        counts = ", ".join(f"{timeframe}={len(frame):,}" for timeframe, frame in frames.items())
        print(f"Cached candles: {counts}")
        print(f"Data fetched: {', '.join(fetched) if fetched else 'none (cache)'}")
        return 0
    versions = (CURRENT_STRATEGY_VERSION,) if args.strategy == "all" else (args.strategy,)
    results = {}
    for version in versions:
        summary, trades = run_backtest(
            frames,
            strategy_version=version,
            timeout_minutes=int(os.getenv("SIGNAL_TIMEOUT_MINUTES", "240")),
            max_per_day=int(os.getenv("SIGNAL_MAX_PER_DAY", "3")),
            cooldown_minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "45")),
        )
        label = args.output_label if len(versions) == 1 and args.output_label else version
        save_backtest(backtest_dir / label, summary, trades)
        if version == CURRENT_STRATEGY_VERSION:
            save_backtest(backtest_dir, summary, trades)
        results[version] = summary
    text = format_backtest(results[CURRENT_STRATEGY_VERSION if args.strategy == "all" else args.strategy])
    print(text)
    print(f"Data fetched: {', '.join(fetched) if fetched else 'none (cache)'}")
    if args.notify and config.telegram.enabled:
        if not send_telegram(text, config.telegram):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
