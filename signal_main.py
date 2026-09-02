#!/usr/bin/env python3
"""One-shot XAUUSD signal confirmation run for a five-minute systemd timer."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from forex.config import Config
from forex.instruments import parse_symbol
from forex.notify import send_telegram, signal_keyboard
from forex.product import bot_name
from forex.providers import TwelveDataProvider, resample
from forex.smc import SMCReading, evaluate
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


def _telegram_text(reading: SMCReading) -> str:
    lines = [
        f"🥇 {bot_name().upper()} — SIGNAL {reading.action}",
        "Mode evaluasi saja, tidak membuka transaksi otomatis.",
        "",
        f"Harga saat signal: {reading.price:.2f}",
        f"Skor konfirmasi: {reading.confluence_score}/100 (bukan peluang menang)",
        f"Arah besar: {reading.macro_bias} | Struktur M15: {reading.m15_structure} | M5: {reading.m5_structure}",
    ]
    if reading.action in ("LONG", "SHORT"):
        lines.extend([
            "",
            f"Referensi entry: {reading.entry:.2f}",
            f"Batas salah / SL: {reading.stop_loss:.2f}",
            f"Target / TP (2R): {reading.take_profit:.2f}",
        ])
    if reading.reasons:
        lines.extend(["", "Alasan signal:", *[f"• {reason}" for reason in reading.reasons]])
    if reading.cautions:
        lines.extend(["", "Perhatian:", *[f"• {item}" for item in reading.cautions]])
    lines.extend(["", "Gunakan sebagai bahan analisis; cocokkan harga dengan broker."])
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
    stats_before = tracker.stats()
    resolved = tracker.evaluate(
        m5,
        timeout_minutes=int(os.getenv("SIGNAL_TIMEOUT_MINUTES", "240")),
    )
    stats_after_evaluation = tracker.stats()
    m15 = resample(m5, "15min")
    reading = evaluate(m5, m15, _load_macro(macro_path))

    now = datetime.now(timezone.utc)
    event_path = state_dir / "signals" / f"{now.date().isoformat()}.jsonl"
    send = False
    tracked_signal_id = None
    if reading.action in ("LONG", "SHORT") and tracker.can_create(
        now=now,
        max_per_day=int(os.getenv("SIGNAL_MAX_PER_DAY", "3")),
        cooldown_minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "45")),
    ):
        tracked_signal_id = tracker.create_signal(reading, m5.index[-1], created_at=now)
        send = tracked_signal_id is not None
    notification_sent = False
    if send and config.telegram.enabled:
        notification_sent = send_telegram(
            _telegram_text(reading),
            config.telegram,
            reply_markup=signal_keyboard(tracked_signal_id),
        )
    result_notification_sent = False
    if resolved and not send and config.telegram.enabled:
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
            "\n".join(["🔄 HASIL SIGNAL DIPERBARUI", *changes]),
            config.telegram,
        )

    event = {
        "generated_at": now.isoformat(timespec="seconds"),
        **reading.to_dict(),
        "tracked_signal_id": tracked_signal_id,
        "resolved_previous_signals": resolved,
        "tracking_stats": tracker.stats().__dict__,
        "notification_requested": send,
        "notification_sent": notification_sent,
        "result_notification_sent": result_notification_sent,
    }
    _append_event(event_path, event)
    _atomic_json(state_dir / "latest_signal.json", event)

    logger.info(
        "action=%s confluence=%s macro=%s m15=%s m5=%s resolved=%s telegram=%s",
        reading.action,
        reading.confluence_score,
        reading.macro_bias,
        reading.m15_structure,
        reading.m5_structure,
        resolved,
        "sent" if notification_sent else ("not-triggered" if not send else "failed"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
