#!/usr/bin/env python3
"""Sweep momentum reward:risk to find the value that maximises risk-adjusted edge.

Loads the cached M5/H1/H4/D1 frames once, then replays the momentum strategy for
several reward:risk values without touching the network.
"""

from __future__ import annotations

import os
from pathlib import Path

from forex.backtest import load_backtest_data, run_momentum_backtest


def main() -> int:
    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    cache_dir = state_dir / "backtest" / "cache"
    frames, _ = load_backtest_data(cache_dir, refresh=False, lookback_days=90)

    print(f"{'RR':>5}  {'Signals':>7}  {'WR':>6}  {'Total R':>8}  {'PF':>6}  {'DD':>6}")
    for reward_r in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        summary, trades = run_momentum_backtest(
            frames,
            timeout_minutes=240,
            max_per_day=5,
            cooldown_minutes=30,
            max_active=3,
            reward_r=reward_r,
        )
        wr = f"{summary.win_rate:.1f}%" if summary.win_rate is not None else "—"
        pf = "—" if summary.profit_factor is None else f"{summary.profit_factor:.2f}"
        print(
            f"{reward_r:>5}  {summary.signals:>7}  {wr:>6}  "
            f"{summary.total_r:>+8.1f}  {pf:>6}  {summary.max_drawdown_r:>6.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
