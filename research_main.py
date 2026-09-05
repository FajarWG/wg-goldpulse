#!/usr/bin/env python3
"""Analyse backtest trades to find filters that improve edge without overfitting.

Reads ``latest_trades.csv`` from a backtest output directory and prints, for
every candidate filter, the resulting sample size, win rate, total R, profit
factor and drawdown. This is a research aid: a filter only becomes part of the
strategy after it survives a separate holdout period.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["result"].isin(("win", "loss"))].copy()


def _metrics(df: pd.DataFrame) -> dict:
    completed = _completed(df)
    wins = int((completed["result"] == "win").sum())
    losses = int((completed["result"] == "loss").sum())
    total_r = float(df["result_r"].sum())
    profit = float(df.loc[df["result_r"] > 0, "result_r"].sum())
    loss_abs = float(df.loc[df["result_r"] < 0, "result_r"].abs().sum())
    cumulative = df["result_r"].cumsum()
    peak = cumulative.cummax()
    drawdown = float((peak - cumulative).max())
    return {
        "n": len(df),
        "completed": wins + losses,
        "wr": (wins / (wins + losses) * 100) if wins + losses else 0.0,
        "total_r": total_r,
        "pf": (profit / loss_abs) if loss_abs else float("inf"),
        "dd": drawdown,
    }


def _show(name: str, df: pd.DataFrame) -> None:
    m = _metrics(df)
    pf = "∞" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    print(
        f"{name:<40} n={m['n']:>4}  WR={m['wr']:5.1f}%  "
        f"R={m['total_r']:+7.1f}  PF={pf:>6}  DD={m['dd']:5.1f}R"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trades_csv",
        nargs="?",
        help="path to latest_trades.csv (defaults to the momentum backtest)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="only consider trades with score >= this value",
    )
    args = parser.parse_args()

    path = Path(args.trades_csv) if args.trades_csv else Path(
        "/var/lib/xauusd-analysis/backtest/momentum_v1/latest_trades.csv"
    )
    if not path.exists():
        print(f"File tidak ditemukan: {path}", file=sys.stderr)
        return 2

    df = pd.read_csv(path)
    if args.min_score:
        df = df[df["score"] >= args.min_score]

    print(f"File: {path}")
    print(f"Total baris: {len(df)}")
    print()

    _show("Semua signal", df)
    _show("Win/Loss saja (tanpa expired)", _completed(df))
    print()

    if "score" in df.columns:
        print("--- Berdasarkan skor momentum ---")
        for lo, hi in ((0, 49), (50, 64), (65, 79), (80, 100)):
            sub = df[df["score"].between(lo, hi)]
            if sub.empty:
                continue
            _show(f"Score {lo}-{hi}", sub)
        print()

    if "regime" in df.columns:
        print("--- Berdasarkan regime ---")
        for regime in sorted(df["regime"].dropna().unique()):
            _show(f"Regime {regime}", df[df["regime"] == regime])
        print()

    print("--- Berdasarkan alignment (momentum) ---")
    for column in ("ema_aligned", "m15_aligned", "volume_aligned"):
        if column not in df.columns:
            continue
        for value in (True, False):
            sub = df[df[column] == value]
            if sub.empty:
                continue
            _show(f"{column}={value}", sub)
        print()

    print("--- Berdasarkan arah ---")
    for direction in sorted(df["direction"].dropna().unique()):
        _show(f"Arah {direction}", df[df["direction"] == direction])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
