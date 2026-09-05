#!/usr/bin/env python3
"""One-shot XAUUSD signal confirmation run for a five-minute systemd timer.

Emits two independent signal types:
  - Full analysis (SMC confluence, max 5/day)
  - Momentum candle (candle-action only, max 5/day)
Each type has its own cooldown and daily cap.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from forex.config import Config
from forex.instruments import parse_symbol
from forex.notify import send_telegram, signal_keyboard
from forex.product import bot_name
from forex.providers import TwelveDataProvider, resample
from forex.smc import MomentumReading, SMCReading, evaluate, momentum_candle
from forex.tracking import SignalTracker


logger = logging.getLogger("xauusd.signal")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_macro(path: Path, max_age_minutes: int = 90) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(payload["generated_at"])
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - generated > timedelta(minutes=max_age_minutes):
            return "stale"
        for pair in payload.get("pairs", []):
            if pair.get("symbol") == "XAUUSD" and not pair.get("error"):
                return (pair.get("alignment") or {}).get("verdict", "unavailable")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass
    return "unavailable"


def _drop_open_candle(frame, minutes: int):
    if frame.empty:
        return frame
    now = datetime.now(timezone.utc)
    index = frame.index
    if index.tz is None:
        last_open = index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    else:
        last_open = index[-1].to_pydatetime().astimezone(timezone.utc)
    if last_open + timedelta(minutes=minutes) > now:
        return frame.iloc[:-1]
    return frame


def _append_event(path: Path, event: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _telegram_text(reading: SMCReading, generated_at: Optional[datetime] = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    action_label = {"LONG": "BUY", "SHORT": "SELL"}.get(reading.action, reading.action)
    macro_label = {
        "up": "naik",
        "down": "turun",
        "sideways": "sideways",
        "conflicted": "berlawanan",
    }.get(reading.macro_bias, reading.macro_bias)
    structure_label = {
        "bullish": "bullish",
        "bearish": "bearish",
        "range": "sideways",
        "unknown": "belum jelas",
    }
    regime_label = {
        "normal": "normal",
        "high_vol": "volatilitas tinggi",
        "unknown": "belum diketahui",
    }.get(reading.regime, reading.regime)
    volume_label = {
        True: "searah harga",
        False: "berlawanan dengan harga",
        None: "tidak tersedia",
    }[reading.volume_confirm]
    lines = [
        f"🥇 {bot_name().upper()} — SIGNAL SIAP {action_label}",
        "📋 Tipe: Analisis Full",
        f"Waktu: {generated_at:%Y-%m-%d %H:%M} UTC",
        "━━━━━━━━━━━━━━━━━━━━",
        "",
        "📍 RENCANA HARGA",
    ]
    if reading.action in ("LONG", "SHORT"):
        lines.extend([
            f"Entry referensi: {reading.entry:.2f}",
            f"Stop loss: {reading.stop_loss:.2f}",
            f"Take profit: {reading.take_profit:.2f}",
            f"Risk/reward: 1:{reading.risk_reward:.1f}",
        ])
    lines.extend([
        "",
        "🔎 KONFIRMASI",
        f"Skor: {reading.confluence_score}/100",
        f"Arah H1/H4/D1: {macro_label}",
        f"Struktur M15: {structure_label.get(reading.m15_structure, reading.m15_structure)}",
        f"Struktur M5: {structure_label.get(reading.m5_structure, reading.m5_structure)}",
        f"RSI M5: {reading.rsi:.1f}",
        f"Kondisi pasar: {regime_label}",
        f"Volume: {volume_label}",
        "",
        "Skor adalah kekuatan konfirmasi, bukan persentase peluang menang.",
    ])
    if reading.reasons:
        lines.extend(["", "✅ ALASAN SIGNAL", *[f"• {reason}" for reason in reading.reasons]])
    if reading.cautions:
        lines.extend(["", "⚠️ PERHATIAN", *[f"• {item}" for item in reading.cautions]])
    lines.extend([
        "",
        "Bot mencatat hasil sampai TP, SL, atau kedaluwarsa. Tidak ada transaksi otomatis.",
    ])
    return "\n".join(lines)


def _momentum_text(reading: MomentumReading, generated_at: Optional[datetime] = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    action_label = {"LONG": "BUY", "SHORT": "SELL"}.get(reading.action, reading.action)
    regime_label = {
        "normal": "normal",
        "high_vol": "volatilitas tinggi",
        "unknown": "belum diketahui",
    }.get(reading.regime, reading.regime)
    pattern_label = None
    if reading.candle_pattern:
        pattern_label = reading.candle_pattern[0] if isinstance(reading.candle_pattern, tuple) else reading.candle_pattern
        pattern_label = str(pattern_label).replace("_", " ").title() if pattern_label else None
    lines = [
        f"🥇 {bot_name().upper()} — MOMENTUM {action_label}",
        "📋 Tipe: Momentum Candle",
        f"Waktu: {generated_at:%Y-%m-%d %H:%M} UTC",
        "━━━━━━━━━━━━━━━━━━━━",
        "",
        "📍 RENCANA HARGA",
    ]
    if reading.action in ("LONG", "SHORT"):
        lines.extend([
            f"Entry referensi: {reading.entry:.2f}",
            f"Stop loss: {reading.stop_loss:.2f}",
            f"Take profit: {reading.take_profit:.2f}",
            f"Risk/reward: 1:{reading.risk_reward:.1f}",
        ])
    lines.extend([
        "",
        "🔎 KONFIRMASI",
        f"Skor momentum: {reading.score}/100",
        f"RSI M5: {reading.rsi:.1f}",
        f"Body candle: {reading.m5_body_ratio:.0%}",
        f"Kondisi pasar: {regime_label}",
    ])
    if pattern_label:
        lines.append(f"Pola candle: {pattern_label}")
    lines.extend([
        f"EMA searah: {'ya' if reading.ema_aligned else 'tidak'}",
        f"M15 konfirmasi: {'ya' if reading.m15_aligned else 'tidak'}",
        f"Volume searah: {'ya' if reading.volume_aligned else 'tidak'}",
        "",
        "Skor momentum = kekuatan aksi harga, bukan peluang menang.",
    ])
    if reading.reasons:
        lines.extend(["", "✅ ALASAN SIGNAL", *[f"• {r}" for r in reading.reasons]])
    if reading.cautions:
        lines.extend(["", "⚠️ PERHATIAN", *[f"• {c}" for c in reading.cautions]])
    lines.extend([
        "",
        "Bot mencatat hasil sampai TP, SL, atau kedaluwarsa. Tidak ada transaksi otomatis.",
    ])
    return "\n".join(lines)


def main() -> int:
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    provider = TwelveDataProvider()
    if not provider.is_available():
        logger.error("TWELVEDATA_API_KEY is not configured")
        return 2

    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    tracker = SignalTracker.from_env() or SignalTracker(state_dir / "signals.sqlite3")
    macro_path = Path(os.getenv("FOREX_STATE_PATH", str(state_dir / "latest.json")))
    instrument = parse_symbol("XAUUSD")
    m5 = _drop_open_candle(provider.fetch(instrument, "M5", 500), 5)
    m15 = resample(m5, "15min")

    # Resolve previous signals
    stats_before = tracker.stats()
    resolved = tracker.evaluate(
        m5,
        timeout_minutes=int(os.getenv("SIGNAL_TIMEOUT_MINUTES", "240")),
    )
    stats_after_evaluation = tracker.stats()

    now = datetime.now(timezone.utc)
    event_path = state_dir / "signals" / f"{now.date().isoformat()}.jsonl"
    max_full = int(os.getenv("SIGNAL_MAX_FULL_PER_DAY", "5"))
    max_momentum = int(os.getenv("SIGNAL_MAX_MOMENTUM_PER_DAY", "5"))
    max_active_momentum = int(os.getenv("SIGNAL_MAX_ACTIVE_MOMENTUM", "3"))
    cooldown = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "30"))

    # --- Full analysis signal ---
    macro_bias = _load_macro(macro_path)
    full_reading = evaluate(m5, m15, macro_bias)
    full_created = False
    full_signal_id = None
    full_notification = False
    if full_reading.action in ("LONG", "SHORT") and tracker.can_create(
        now=now,
        max_per_day=max_full,
        cooldown_minutes=cooldown,
        signal_type="full",
    ):
        full_signal_id = tracker.create_signal(full_reading, m5.index[-1], created_at=now, signal_type="full")
        full_created = full_signal_id is not None
    if full_created and config.telegram.enabled:
        full_notification = send_telegram(
            _telegram_text(full_reading, now),
            config.telegram,
            reply_markup=signal_keyboard(),
        )

    # --- Momentum candle signal ---
    momentum_reading = momentum_candle(m5, m15)
    momentum_created = False
    momentum_signal_id = None
    momentum_notification = False
    if momentum_reading.action in ("LONG", "SHORT") and tracker.can_create(
        now=now,
        max_per_day=max_momentum,
        cooldown_minutes=cooldown,
        signal_type="momentum",
        max_active=max_active_momentum,
    ):
        momentum_signal_id = tracker.create_signal(momentum_reading, m5.index[-1], created_at=now, signal_type="momentum")
        momentum_created = momentum_signal_id is not None
    if momentum_created and config.telegram.enabled:
        momentum_notification = send_telegram(
            _momentum_text(momentum_reading, now),
            config.telegram,
            reply_markup=signal_keyboard(),
        )

    # --- Result notification ---
    result_notification_sent = False
    if resolved and not full_created and not momentum_created and config.telegram.enabled:
        changes = []
        won = stats_after_evaluation.wins - stats_before.wins
        lost = stats_after_evaluation.losses - stats_before.losses
        expired = stats_after_evaluation.expired - stats_before.expired
        if won:
            changes.append(f"✅ Benar: +{won}")
        if lost:
            changes.append(f"❌ Salah: +{lost}")
        if expired:
            changes.append(f"⌛ Kedaluwarsa: +{expired}")
        result_notification_sent = send_telegram(
            "\n".join([
                "🔄 HASIL SIGNAL SELESAI DINILAI",
                "━━━━━━━━━━━━━━━━━━━━",
                *changes,
                "",
                "Tekan Statistik untuk melihat rekap lengkap.",
            ]),
            config.telegram,
        )

    # --- Event log ---
    event = {
        "generated_at": now.isoformat(timespec="seconds"),
        "full_analysis": {
            **full_reading.to_dict(),
            "signal_id": full_signal_id,
            "created": full_created,
            "notification_sent": full_notification,
        },
        "momentum_candle": {
            **momentum_reading.to_dict(),
            "signal_id": momentum_signal_id,
            "created": momentum_created,
            "notification_sent": momentum_notification,
        },
        "resolved_previous_signals": resolved,
        "tracking_stats": tracker.stats().__dict__,
        "result_notification_sent": result_notification_sent,
    }
    _append_event(event_path, event)
    _atomic_json(state_dir / "latest_signal.json", event)

    logger.info(
        "full=%s(%s) momentum=%s(%s) resolved=%d",
        full_reading.action,
        full_reading.confluence_score,
        momentum_reading.action,
        momentum_reading.score,
        resolved,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
