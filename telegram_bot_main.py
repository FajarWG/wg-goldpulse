#!/usr/bin/env python3
"""Long-poll Telegram callback buttons for the XAUUSD analysis bot."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from forex.config import Config, TelegramConfig
from forex.notify import send_telegram, stats_keyboard
from forex.backtest import format_backtest, load_latest_summary
from forex.llm import generate_commentary
from forex.product import (
    BOT_DESCRIPTION,
    BOT_SHORT_DESCRIPTION,
    CURRENT_STRATEGY_VERSION,
    bot_name,
)
from forex.tracking import SignalTracker, format_stats_footer
from forex.usage import tracker_from_env


logger = logging.getLogger("xauusd.telegram")
_TELEGRAM_API = "https://api.telegram.org"


def _post(config: TelegramConfig, method: str, payload: Dict[str, Any], timeout: int = 20):
    response = requests.post(
        f"{_TELEGRAM_API}/bot{config.bot_token}/{method}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _answer(config: TelegramConfig, callback_id: str, text: str, alert: bool = False) -> None:
    _post(
        config,
        "answerCallbackQuery",
        {"callback_query_id": callback_id, "text": text, "show_alert": alert},
    )


def _help_text() -> str:
    return "\n".join(
        [
            f"🥇 {bot_name().upper()}",
            "Asisten analisis XAUUSD—tanpa auto-trading.",
            "",
            "/status — rekap signal forward test",
            "/backtest — hasil historical backtest terakhir",
            "/usage — pemakaian Twelve Data",
            "/ai — minta analisis AI untuk hasil terakhir (manual)",
            "/help — bantuan ini",
            "",
            "AI hanya dipicu manual lewat /ai atau tombol 🤖 Analisis AI — "
            "tidak pernah otomatis di notifikasi harian.",
            "",
            "Semua signal siap dicatat dan dinilai otomatis sampai TP, SL, atau kedaluwarsa. "
            "Bot tidak mengeksekusi transaksi.",
        ]
    )


def _backtest_text() -> str:
    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    root = state_dir / "backtest"
    current = load_latest_summary(root / CURRENT_STRATEGY_VERSION / "latest.json")
    fallback = load_latest_summary(root / "latest.json")
    if current is None and fallback is not None and fallback.strategy_version == CURRENT_STRATEGY_VERSION:
        current = fallback
    momentum = load_latest_summary(root / "momentum_v1" / "latest.json")
    sections = []
    for label, summary in ((f"Analisis Full · {CURRENT_STRATEGY_VERSION}", current),
                           ("Momentum Candle · momentum_v1", momentum)):
        sections.append(label + "\n" + (format_backtest(summary) if summary else
                        "Backtest versi ini belum tersedia. Jadwal otomatis: Sabtu setelah pasar tutup."))
    return "\n\n".join(sections)


def _usage_text() -> str:
    tracker = tracker_from_env()
    if tracker is None:
        return "Pemantauan Twelve Data belum dikonfigurasi."
    usage = tracker.summary()
    return "\n".join(
        [
            "📡 PEMAKAIAN TWELVE DATA",
            "━━━━━━━━━━━━━━━━",
            f"Tanggal UTC: {usage.day_utc}",
            f"Request bot: {usage.requests}",
            f"Estimasi kredit: {usage.estimated_credits}/{usage.daily_limit}",
            f"Sisa konservatif: {usage.remaining_estimate}",
            f"Gagal: {usage.failures}",
            "━━━━━━━━━━━━━━━━",
            "Dua API key dipakai bergantian dan sebagai failover.",
        ]
    )


def _load_latest_payload() -> Optional[Dict[str, Any]]:
    """Read the latest computed analysis state written by the pipeline."""
    path = os.getenv("FOREX_STATE_PATH", "").strip()
    if not path:
        state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
        path = str(state_dir / "latest.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _ai_text() -> str:
    """Generate AI commentary over the latest computed payload (on demand)."""
    config = Config.from_env()
    if not config.llm.enabled:
        return (
            "🤖 AI belum dikonfigurasi (tidak ada API key). "
            "Analisis deterministik tetap berjalan; AI hanya on-demand."
        )
    payload = _load_latest_payload()
    if not payload or not payload.get("pairs"):
        return (
            "Belum ada hasil analisis untuk dijelaskan. "
            "Jalankan `python main.py` dulu (analisis terjadwal otomatis), "
            "lalu minta AI lagi."
        )
    try:
        metadata: Dict[str, Any] = {}
        commentary = generate_commentary(payload, config.llm, metadata=metadata)
    except Exception as exc:
        logger.warning("AI commentary failed (%s)", type(exc).__name__)
        return "🤖 Gagal menghasilkan analisis AI. Coba lagi nanti."
    if not commentary:
        return "🤖 Penyedia AI tidak membalas. Coba lagi nanti."
    provider = ""
    if metadata.get("provider"):
        provider = f"\nSumber: {metadata['provider']} · {metadata.get('model', 'model default')}"
    return f"🤖 ANALISIS AI (XAUUSD & pasangan){provider}\n\n{commentary}"


def _send_command(command: str, config: TelegramConfig, tracker: SignalTracker) -> None:
    if command in ("/start", "/help"):
        send_telegram(_help_text(), config)
    elif command == "/status":
        send_telegram(format_stats_footer(tracker.stats()), config)
    elif command == "/backtest":
        send_telegram(_backtest_text(), config)
    elif command == "/usage":
        send_telegram(_usage_text(), config)
    elif command == "/ai":
        send_telegram(_ai_text(), config)


def handle_message(message: Dict[str, Any], config: TelegramConfig, tracker: SignalTracker) -> None:
    chat_id = str((message.get("chat") or {}).get("id", ""))
    if chat_id != str(config.chat_id):
        return
    command = str(message.get("text", "")).strip().split(maxsplit=1)[0].split("@", 1)[0].lower()
    if command.startswith("/"):
        _send_command(command, config, tracker)


def handle_callback(
    query: Dict[str, Any],
    config: TelegramConfig,
    tracker: SignalTracker,
) -> None:
    callback_id = str(query.get("id", ""))
    message = query.get("message") or {}
    chat_id = str(message.get("chat", {}).get("id", ""))
    if chat_id != str(config.chat_id):
        _answer(config, callback_id, "Chat ini tidak diizinkan.", alert=True)
        return

    data = str(query.get("data", ""))
    if data == "ai":
        _answer(config, callback_id, "Menghasilkan analisis AI…")
        send_telegram(_ai_text(), config)
        return
    if data == "stats":
        _answer(config, callback_id, "Statistik diperbarui")
        send_telegram(format_stats_footer(tracker.stats()), config)
        return
    if data == "backtest":
        _answer(config, callback_id, "Membuka backtest terakhir")
        send_telegram(_backtest_text(), config)
        return
    if data == "help":
        _answer(config, callback_id, "Membuka bantuan")
        send_telegram(_help_text(), config)
        return
    _answer(config, callback_id, "Tombol tidak dikenali.", alert=True)


def main() -> int:
    config = Config.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if not config.telegram.enabled:
        logger.error("Telegram bot token or chat id is not configured")
        return 2
    tracker = SignalTracker.from_env()
    if tracker is None:
        logger.error("SIGNAL_TRACKING_DB is not configured")
        return 2

    offset_value = tracker.get_state("telegram_update_offset")
    offset = int(offset_value) if offset_value else None
    try:
        _post(config.telegram, "setMyName", {"name": bot_name()})
        _post(config.telegram, "setMyDescription", {"description": BOT_DESCRIPTION})
        _post(config.telegram, "setMyShortDescription", {"short_description": BOT_SHORT_DESCRIPTION})
        _post(
            config.telegram,
            "setMyCommands",
            {
                "commands": [
                    {"command": "status", "description": "Rekap signal dan win rate"},
                    {"command": "backtest", "description": "Historical backtest terakhir"},
                    {"command": "usage", "description": "Pemakaian Twelve Data"},
                    {"command": "ai", "description": "Minta analisis AI hasil terakhir (manual)"},
                    {"command": "help", "description": "Cara memakai bot"},
                ]
            },
        )
    except Exception as exc:
        logger.warning("Unable to update Telegram profile (%s)", type(exc).__name__)
    logger.info("Telegram command and button listener started")
    while True:
        payload: Dict[str, Any] = {
            "timeout": 45,
            "allowed_updates": ["callback_query", "message"],
        }
        if offset is not None:
            payload["offset"] = offset
        try:
            response = _post(config.telegram, "getUpdates", payload, timeout=55)
            for update in response.get("result", []):
                callback = update.get("callback_query")
                if callback:
                    try:
                        handle_callback(callback, config.telegram, tracker)
                    except Exception as exc:
                        logger.warning("Callback handling failed (%s)", type(exc).__name__)
                message = update.get("message")
                if message:
                    try:
                        handle_message(message, config.telegram, tracker)
                    except Exception as exc:
                        logger.warning("Message handling failed (%s)", type(exc).__name__)
                offset = int(update["update_id"]) + 1
                tracker.set_state("telegram_update_offset", str(offset))
        except Exception as exc:
            logger.warning("Telegram polling failed (%s)", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
