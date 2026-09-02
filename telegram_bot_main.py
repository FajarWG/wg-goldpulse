#!/usr/bin/env python3
"""Long-poll Telegram callback buttons for the XAUUSD analysis bot."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import requests

from forex.config import Config, TelegramConfig
from forex.notify import send_telegram, stats_keyboard
from forex.backtest import format_backtest, format_comparison, load_latest_summary
from forex.product import BOT_DESCRIPTION, BOT_SHORT_DESCRIPTION, bot_name
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


def _mark_decision(config: TelegramConfig, query: Dict[str, Any], decision: str) -> None:
    message = query.get("message") or {}
    label = "✅ Dipilih: Ambil" if decision == "take" else "⏭ Dipilih: Lewati"
    _post(
        config,
        "editMessageReplyMarkup",
        {
            "chat_id": message.get("chat", {}).get("id"),
            "message_id": message.get("message_id"),
            "reply_markup": {
                "inline_keyboard": [
                    [{"text": label, "callback_data": "noop"}],
                    stats_keyboard()["inline_keyboard"][0],
                ]
            },
        },
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
            "/ai — status AI dan urutan fallback",
            "/help — bantuan ini",
            "",
            "Saat signal READY, gunakan tombol Ambil atau Lewati. Hasil signal tetap dinilai otomatis.",
        ]
    )


def _backtest_text() -> str:
    state_dir = Path(os.getenv("FOREX_STATE_DIR", "/var/lib/xauusd-analysis"))
    root = state_dir / "backtest"
    v1 = load_latest_summary(root / "v1" / "latest.json") or load_latest_summary(root / "latest.json")
    v2 = load_latest_summary(root / "v2" / "latest.json")
    if v1 is not None and v2 is not None:
        return format_comparison(v1, v2)
    if v1 is None:
        return "🧪 Backtest belum tersedia. Backtest otomatis dijalankan setiap Sabtu setelah pasar tutup."
    return format_backtest(v1)


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


def _ai_text() -> str:
    candidates = Config.from_env().llm.candidates_from_env()
    if not candidates:
        return "🤖 AI belum dikonfigurasi. Analisis deterministik tetap berjalan."
    lines = ["🤖 AI EXPLANATION", "━━━━━━━━━━━━━━━━", "Urutan fallback:"]
    lines.extend(f"{index}. {item.provider.title()} · {item.model}" for index, item in enumerate(candidates, 1))
    lines.extend(["━━━━━━━━━━━━━━━━", "AI hanya menjelaskan hasil; tidak mengubah signal, skor, SL, atau TP."])
    return "\n".join(lines)


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
    if data == "noop":
        _answer(config, callback_id, "Keputusan sudah tersimpan.")
        return
    if ":" not in data:
        _answer(config, callback_id, "Tombol tidak dikenali.", alert=True)
        return

    decision, signal_id = data.split(":", 1)
    result = tracker.record_decision(signal_id, decision, chat_id)
    if result == "saved":
        word = "diambil" if decision == "take" else "dilewati"
        _answer(config, callback_id, f"Signal berhasil {word}.")
        _mark_decision(config, query, decision)
    elif result.startswith("already:"):
        _answer(config, callback_id, "Keputusan signal ini sudah tersimpan.", alert=True)
    elif result == "closed":
        _answer(config, callback_id, "Signal sudah selesai dinilai.", alert=True)
    else:
        _answer(config, callback_id, "Signal tidak ditemukan.", alert=True)


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
                    {"command": "ai", "description": "Status AI explanation"},
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
