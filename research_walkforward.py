#!/usr/bin/env python3
"""Walk-forward validation for the momentum strategy.

The strategy was tuned on the recent 90-day window. To check whether its edge is
real rather than an artifact of that window, we split history into an in-sample
("train") period and an untouched out-of-sample ("holdout") period, replay both,
and print the metrics side by side.

Usage:
    python research_walkforward.py [--holdout-start YYYY-MM-DD] [--lookback-days N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from forex.backtest import load_backtest_data, run_momentum_backtest


def _load(lookback_days: int) -> dict:
    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    cache_dir = state_dir / "backtest" / "cache"
    frames, _ = load_backtest_data(
        cache_dir,
        refresh=False,
        lookback_days=lookback_days,
    )
    return frames


def _run(frames: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    sliced = {key: value for key, value in frames.items()}
    m5 = sliced["M5"].loc[(sliced["M5"].index >= start) & (sliced["M5"].index <= end)]
    if m5.empty:
        return {"signals": 0, "wr": None, "total_r": 0.0, "pf": None, "dd": 0.0}
    sliced["M5"] = m5
    summary, trades = run_momentum_backtest(
        sliced,
        timeout_minutes=240,
        max_per_day=5,
        cooldown_minutes=30,
        max_active=3,
        reward_r=1.0,
    )
    wins = sum(t.result == "win" for t in trades)
    losses = sum(t.result == "loss" for t in trades)
    completed = wins + losses
    total_r = sum(t.result_r for t in trades)
    profit = sum(t.result_r for t in trades if t.result_r > 0)
    loss_abs = abs(sum(t.result_r for t in trades if t.result_r < 0))
    pf = profit / loss_abs if loss_abs else None
    return {
        "signals": len(trades),
        "wr": (wins / completed * 100) if completed else None,
        "total_r": total_r,
        "pf": pf,
        "dd": summary.max_drawdown_r,
    }


def _fmt(name: str, m: dict) -> str:
    wr = "—" if m["wr"] is None else f"{m['wr']:.1f}%"
    pf = "—" if m["pf"] is None else f"{m['pf']:.2f}"
    return (
        f"{name:<22} n={m['signals']:>4}  WR={wr:>6}  "
        f"R={m['total_r']:+7.1f}  PF={pf:>6}  DD={m['dd']:5.1f}R"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--holdout-start", default=None, help="YYYY-MM-DD; default splits lookback in half")
    args = parser.parse_args()

    frames = _load(args.lookback_days)
    m5 = frames["M5"]
    if m5.empty:
        print("Tidak ada data M5 untuk periode tersebut.", file=sys.stderr)
        return 2

    end = m5.index.max()
    if args.holdout_start:
        split = pd.Timestamp(args.holdout_start, tz="UTC")
    else:
        split = m5.index.min() + (end - m5.index.min()) / 2

    train = _run(frames, m5.index.min(), split)
    holdout = _run(frames, split, end)

    print(f"Periode cache: {m5.index.min().date()} → {end.date()}")
    print(f"Split holdout: {split.date()}")
    print()
    print(_fmt("In-sample (train)", train))
    print(_fmt("Out-of-sample (holdout)", holdout))
    print()
    print("Baca: jika holdout tetap positif dan PF > 1, edge momentum lebih")
    print("meyakinkan. Jika holdout negatif, hasil 90 hari terakhir kemungkinan")
    print("overfit dan filter perlu dievaluasi ulang.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
