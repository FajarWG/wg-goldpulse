"""Persistent paper-signal tracking and objective result statistics."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .product import CURRENT_STRATEGY_VERSION


@dataclass(frozen=True)
class SignalStats:
    total: int
    completed: int
    wins: int
    losses: int
    active: int
    expired: int
    total_r: float

    @property
    def win_rate(self) -> Optional[float]:
        if self.completed == 0:
            return None
        return self.wins / self.completed * 100


@dataclass(frozen=True)
class SignalTypeStats:
    """Per-type statistics breakdown (full analysis vs momentum candle)."""
    signal_type: str
    total: int
    completed: int
    wins: int
    losses: int
    active: int
    expired: int
    total_r: float

    @property
    def win_rate(self) -> Optional[float]:
        if self.completed == 0:
            return None
        return self.wins / self.completed * 100


def _utc(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


class SignalTracker:
    """Store paper signals and resolve them against subsequent closed candles."""

    def __init__(
        self,
        path: Path | str,
        strategy_version: str = CURRENT_STRATEGY_VERSION,
    ) -> None:
        self.path = Path(path)
        self.strategy_version = strategy_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    @classmethod
    def from_env(cls) -> Optional["SignalTracker"]:
        path = os.getenv("SIGNAL_TRACKING_DB", "").strip()
        if not path:
            return None
        version = os.getenv("SIGNAL_STRATEGY_VERSION", CURRENT_STRATEGY_VERSION).strip()
        return cls(path, version or CURRENT_STRATEGY_VERSION)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_signals (
                    id TEXT PRIMARY KEY,
                    strategy_version TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    signal_candle_at_utc TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
                    entry REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    score INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active', 'win', 'loss', 'expired')),
                    last_checked_candle_utc TEXT,
                    closed_at_utc TEXT,
                    exit_price REAL,
                    result_r REAL,
                    ambiguous INTEGER NOT NULL DEFAULT 0,
                    context_json TEXT NOT NULL DEFAULT '{}',
                    regime TEXT,
                    bias_source TEXT,
                    hysteresis TEXT,
                    signal_type TEXT NOT NULL DEFAULT 'full'
                )
                """
            )
            # Migration for databases created before newer columns existed.
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(paper_signals)").fetchall()
            }
            for name in ("regime", "bias_source", "hysteresis"):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE paper_signals ADD COLUMN {name} TEXT"
                    )
            if "signal_type" not in columns:
                connection.execute(
                    "ALTER TABLE paper_signals ADD COLUMN signal_type TEXT NOT NULL DEFAULT 'full'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_paper_signals_status ON paper_signals(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_paper_signals_created ON paper_signals(created_at_utc)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_decisions (
                    signal_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL CHECK(decision IN ('take', 'skip')),
                    decided_at_utc TEXT NOT NULL,
                    chat_id TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def can_create(
        self,
        now: Optional[datetime] = None,
        max_per_day: int = 5,
        cooldown_minutes: int = 30,
        signal_type: str = "full",
    ) -> bool:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        with self._connect() as connection:
            if connection.execute(
                """
                SELECT 1 FROM paper_signals
                WHERE status = 'active' AND strategy_version = ? AND signal_type = ? LIMIT 1
                """,
                (self.strategy_version, signal_type),
            ).fetchone():
                return False
            count = connection.execute(
                """
                SELECT COUNT(*) FROM paper_signals
                WHERE created_at_utc >= ? AND strategy_version = ? AND signal_type = ?
                """,
                (_iso(day_start), self.strategy_version, signal_type),
            ).fetchone()[0]
            if count >= max_per_day:
                return False
            latest = connection.execute(
                """
                SELECT created_at_utc FROM paper_signals
                WHERE strategy_version = ? AND signal_type = ?
                ORDER BY created_at_utc DESC LIMIT 1
                """,
                (self.strategy_version, signal_type),
            ).fetchone()
        if latest is None:
            return True
        return now - datetime.fromisoformat(latest[0]) >= timedelta(minutes=cooldown_minutes)

    def create_signal(
        self,
        reading: Any,
        candle_at: Any,
        created_at: Optional[datetime] = None,
        signal_type: str = "full",
    ) -> Optional[str]:
        if reading.action not in ("LONG", "SHORT"):
            return None
        if reading.entry is None or reading.stop_loss is None or reading.take_profit is None:
            return None
        created_at = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        candle = _utc(candle_at)
        signal_id = f"{self.strategy_version}:{signal_type}:{reading.action}:{_iso(candle)}"
        context: Dict[str, Any] = reading.to_dict()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO paper_signals (
                    id, strategy_version, created_at_utc, signal_candle_at_utc,
                    direction, entry, stop_loss, take_profit, score, status, context_json,
                    regime, bias_source, hysteresis, signal_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    self.strategy_version,
                    _iso(created_at),
                    _iso(candle),
                    reading.action,
                    float(reading.entry),
                    float(reading.stop_loss),
                    float(reading.take_profit),
                    int(reading.score if signal_type == "momentum" else reading.confluence_score),
                    json.dumps(context, ensure_ascii=False),
                    getattr(reading, "regime", None),
                    getattr(reading, "bias_source", None),
                    getattr(reading, "hysteresis", None),
                    signal_type,
                ),
            )
        return signal_id if cursor.rowcount == 1 else None

    def evaluate(
        self,
        frame: pd.DataFrame,
        timeout_minutes: int = 240,
        now: Optional[datetime] = None,
    ) -> int:
        """Resolve active signals. If TP and SL touch together, count a loss."""
        if frame.empty:
            return 0
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        resolved = 0
        with self._connect() as connection:
            active = connection.execute(
                """
                SELECT * FROM paper_signals
                WHERE status = 'active' AND strategy_version = ?
                ORDER BY created_at_utc
                """,
                (self.strategy_version,),
            ).fetchall()
            for signal in active:
                after = signal["last_checked_candle_utc"] or signal["signal_candle_at_utc"]
                after_dt = datetime.fromisoformat(after)
                created = datetime.fromisoformat(signal["created_at_utc"])
                deadline = created + timedelta(minutes=timeout_minutes)
                candidate_rows = []
                for timestamp, candle in frame.iterrows():
                    candle_at = _utc(timestamp)
                    if after_dt < candle_at <= deadline:
                        candidate_rows.append((candle_at, candle))

                outcome = None
                ambiguous = 0
                closed_at = None
                exit_price = None
                result_r = None
                last_checked = None
                for candle_at, candle in candidate_rows:
                    last_checked = candle_at
                    high, low = float(candle["high"]), float(candle["low"])
                    if signal["direction"] == "LONG":
                        hit_target = high >= signal["take_profit"]
                        hit_stop = low <= signal["stop_loss"]
                    else:
                        hit_target = low <= signal["take_profit"]
                        hit_stop = high >= signal["stop_loss"]
                    if hit_target and hit_stop:
                        outcome, ambiguous = "loss", 1
                        exit_price, result_r = signal["stop_loss"], -1.0
                    elif hit_stop:
                        outcome = "loss"
                        exit_price, result_r = signal["stop_loss"], -1.0
                    elif hit_target:
                        outcome = "win"
                        exit_price, result_r = signal["take_profit"], 2.0
                    if outcome:
                        closed_at = candle_at
                        break

                if outcome is None and now - created >= timedelta(minutes=timeout_minutes):
                    outcome = "expired"
                    closed_at = now
                    result_r = 0.0

                if outcome is not None:
                    connection.execute(
                        """
                        UPDATE paper_signals
                        SET status = ?, last_checked_candle_utc = ?, closed_at_utc = ?,
                            exit_price = ?, result_r = ?, ambiguous = ?
                        WHERE id = ?
                        """,
                        (
                            outcome,
                            _iso(last_checked) if last_checked else signal["last_checked_candle_utc"],
                            _iso(closed_at),
                            exit_price,
                            result_r,
                            ambiguous,
                            signal["id"],
                        ),
                    )
                    resolved += 1
                elif last_checked is not None:
                    connection.execute(
                        "UPDATE paper_signals SET last_checked_candle_utc = ? WHERE id = ?",
                        (_iso(last_checked), signal["id"]),
                    )
        return resolved

    def stats(self) -> SignalStats:
        with self._connect() as connection:
            rows = dict(
                connection.execute(
                    """
                    SELECT status, COUNT(*) AS count FROM paper_signals
                    WHERE strategy_version = ? GROUP BY status
                    """,
                    (self.strategy_version,),
                ).fetchall()
            )
            total_r = connection.execute(
                """
                SELECT COALESCE(SUM(result_r), 0) FROM paper_signals
                WHERE strategy_version = ?
                """,
                (self.strategy_version,),
            ).fetchone()[0]
        wins = int(rows.get("win", 0))
        losses = int(rows.get("loss", 0))
        active = int(rows.get("active", 0))
        expired = int(rows.get("expired", 0))
        return SignalStats(
            total=wins + losses + active + expired,
            completed=wins + losses,
            wins=wins,
            losses=losses,
            active=active,
            expired=expired,
            total_r=float(total_r),
        )

    def stats_by_type(self, signal_type: str) -> SignalTypeStats:
        """Compute statistics filtered by signal_type ('full' or 'momentum')."""
        with self._connect() as connection:
            rows = dict(
                connection.execute(
                    """
                    SELECT status, COUNT(*) AS count FROM paper_signals
                    WHERE strategy_version = ? AND signal_type = ? GROUP BY status
                    """,
                    (self.strategy_version, signal_type),
                ).fetchall()
            )
            total_r = connection.execute(
                """
                SELECT COALESCE(SUM(result_r), 0) FROM paper_signals
                WHERE strategy_version = ? AND signal_type = ?
                """,
                (self.strategy_version, signal_type),
            ).fetchone()[0]
        wins = int(rows.get("win", 0))
        losses = int(rows.get("loss", 0))
        active = int(rows.get("active", 0))
        expired = int(rows.get("expired", 0))
        return SignalTypeStats(
            signal_type=signal_type,
            total=wins + losses + active + expired,
            completed=wins + losses,
            wins=wins,
            losses=losses,
            active=active,
            expired=expired,
            total_r=float(total_r),
        )

    def record_decision(self, signal_id: str, decision: str, chat_id: str) -> str:
        """Record one immutable manual take/skip choice for an active signal."""
        if decision not in ("take", "skip"):
            return "invalid"
        with self._connect() as connection:
            signal = connection.execute(
                "SELECT status FROM paper_signals WHERE id = ?", (signal_id,)
            ).fetchone()
            if signal is None:
                return "not_found"
            existing = connection.execute(
                "SELECT decision FROM signal_decisions WHERE signal_id = ?", (signal_id,)
            ).fetchone()
            if existing is not None:
                return f"already:{existing['decision']}"
            if signal["status"] != "active":
                return "closed"
            connection.execute(
                """
                INSERT INTO signal_decisions(signal_id, decision, decided_at_utc, chat_id)
                VALUES (?, ?, ?, ?)
                """,
                (signal_id, decision, _iso(datetime.now(timezone.utc)), str(chat_id)),
            )
        return "saved"

    def get_state(self, key: str) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM telegram_state WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO telegram_state(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )


def format_stats_footer(stats: SignalStats) -> str:
    """Render stats with per-type breakdown (full analysis vs momentum)."""
    lines = [
        "📊 STATISTIK FORWARD TEST",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Signal tercatat: {stats.total}",
        f"Sudah selesai: {stats.completed}",
        "",
        "HASIL SELESAI",
        f"✅ Benar: {stats.wins}",
        f"❌ Salah: {stats.losses}",
    ]
    if stats.completed > 0:
        lines.append(f"🎯 Win rate: {stats.win_rate:.1f}%")
    else:
        lines.append("🎯 Win rate: Belum tersedia")
    lines.extend([
        "",
        "STATUS LAIN",
        f"⏳ Masih aktif: {stats.active}",
        f"⌛ Kedaluwarsa: {stats.expired}",
        f"📈 Akumulasi hasil: {stats.total_r:+.1f}R",
        "━━━━━━━━━━━━━━━━━━━━",
    ])

    # Per-type breakdown
    tracker = SignalTracker.from_env()
    if tracker is not None:
        for stype, label in (("full", "Analisis Full"), ("momentum", "Momentum Candle")):
            ts = tracker.stats_by_type(stype)
            if ts.total == 0:
                continue
            wr = f"{ts.win_rate:.1f}%" if ts.completed > 0 else "—"
            lines.extend([
                "",
                f"📋 {label}",
                f"Signal: {ts.total} · WR: {wr} · {ts.total_r:+.1f}R",
            ])

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("Win rate hanya menghitung signal yang sudah menyentuh TP atau SL.")
    return "\n".join(lines)


def append_stats_footer(text: str) -> str:
    marker = "📊 STATISTIK FORWARD TEST"
    if marker in text:
        return text
    tracker = SignalTracker.from_env()
    if tracker is None:
        return text
    footer = format_stats_footer(tracker.stats())
    limit = 4096
    available = limit - len(footer) - 2
    body = text if len(text) <= available else text[: max(0, available - 3)].rstrip() + "..."
    return f"{body}\n\n{footer}"
