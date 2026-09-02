"""Product identity shared by reports, Telegram, and documentation."""

from __future__ import annotations

import os


def bot_name() -> str:
    return os.getenv("BOT_DISPLAY_NAME", "WG GoldPulse").strip() or "WG GoldPulse"


BOT_DESCRIPTION = (
    "Asisten analisis XAUUSD multi-timeframe dengan signal SMC, validasi hasil, "
    "historical backtest, AI explanation, dan pemantauan kuota data. "
    "Tidak melakukan transaksi otomatis."
)

BOT_SHORT_DESCRIPTION = "Analisis dan validasi signal XAUUSD—tanpa auto-trading."
