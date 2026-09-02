"""Optional notification channels.

Every sender returns a boolean and never raises: a failed push must not discard a
report that was already computed and written to disk.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .config import TelegramConfig

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


def stats_keyboard() -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "📊 Statistik", "callback_data": "stats"},
                {"text": "🧪 Backtest", "callback_data": "backtest"},
            ],
            [{"text": "ℹ️ Bantuan", "callback_data": "help"}],
        ]
    }


def signal_keyboard(signal_id: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Ambil", "callback_data": f"take:{signal_id}"},
                {"text": "⏭ Lewati", "callback_data": f"skip:{signal_id}"},
            ],
            *stats_keyboard()["inline_keyboard"],
        ]
    }


def send_telegram(
    text: str,
    config: TelegramConfig,
    timeout: int = 20,
    reply_markup: Optional[Dict[str, Any]] = None,
) -> bool:
    """Send a plain-text message. Returns True on success."""
    if not config.enabled:
        logger.info("Telegram push skipped: bot token or chat id missing")
        return False

    import requests

    url = f"{_TELEGRAM_API}/bot{config.bot_token}/sendMessage"
    payload = {
        "chat_id": config.chat_id,
        "text": text[:4096],
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    elif os.getenv("SIGNAL_TRACKING_DB", "").strip():
        payload["reply_markup"] = stats_keyboard()

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except Exception as exc:
        # Request exception text may contain the URL, which embeds the bot token.
        logger.warning("Telegram push failed (%s)", type(exc).__name__)
        return False

    if response.status_code >= 400:
        # Never log the token; the URL contains it.
        logger.warning("Telegram push rejected (%s): %s", response.status_code, response.text[:200])
        return False

    return True


def notify(text: str, telegram: Optional[TelegramConfig] = None) -> dict:
    """Fan out to every configured channel, reporting per-channel outcome."""
    results = {}
    if telegram is not None:
        results["telegram"] = send_telegram(text, telegram)
    return results
