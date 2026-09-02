# WG GoldPulse

WG GoldPulse adalah bot analisis XAU/USD untuk Ubuntu/VPS. Sistem mengambil data pasar, menghitung analisis multi-timeframe dan signal deterministik, mencatat hasil signal, menjalankan historical backtest, menambahkan penjelasan AI opsional, lalu mengirim laporan ke Telegram.

> **Analysis only.** Proyek ini tidak terhubung ke akun broker dan tidak membuka atau menutup transaksi. Hasil backtest bukan jaminan performa berikutnya.

## Status strategi

| Versi | Status | Hasil utama |
| --- | --- | --- |
| V1 | Tidak direkomendasikan | 90 hari: −19R, WR 27,8% |
| V2 | Kandidat/shadow | Locked holdout: +1R, PF 1,05 |
| V3/V3.1 | Riset gagal | Tidak mencapai promotion gate |
| V4 | Riset gagal | Development 75,3% WR, tetapi locked holdout hanya 54,4% dan −10,8R setelah cost |

Tidak ada versi yang saat ini boleh dianggap profitable. Signal Telegram harus diperlakukan sebagai eksperimen manual, bukan instruksi trading.

## Fitur

- Analisis H1/H4/D1 dengan EMA, RSI, MACD, ATR, Bollinger, Donchian, volatility regime, support, dan resistance.
- Konfirmasi M5/M15 berbasis market structure, BOS, liquidity sweep, FVG, candle pattern, dan confluence score.
- Signal LONG/SHORT/WAIT dengan entry, batas salah/SL, dan target referensi.
- Telegram commands dan inline buttons untuk statistik, backtest, bantuan, Ambil, dan Lewati.
- Forward validation otomatis: menang, kalah, expired, win rate, dan akumulasi R.
- Historical replay V1–V4 dengan cache data UTC, versioned output, drawdown, profit factor, p-value Monte Carlo, dan breakdown regime.
- Dual Twelve Data API key, quota accounting, failover, throttling, dan retry.
- AI explanation **on-demand** (fallback Groq → Gemini → DeepSeek): tidak pernah otomatis. Dipicu manual via `python main.py --ai`, tombol "AI Analisis" di desktop, atau `/ai`/tombol 🤖 di Telegram.
- systemd services/timers untuk market analysis, signal checks, Telegram listener, dan weekly backtest.

## Quick start lokal

```bash
git clone https://github.com/FajarWG/wg-goldpulse.git
cd wg-goldpulse
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Isi `.env` secara lokal. Jangan commit API key atau token.

```bash
python main.py --symbols XAUUSD --timeframes H1,H4,D1 --stdout
python main.py --symbols XAUUSD --ai --stdout   # tambahkan penjelasan AI (manual)
python signal_main.py
python backtest_main.py --strategy all
python telegram_bot_main.py
```

## Konfigurasi minimum

```dotenv
TWELVEDATA_API_KEY=
TWELVEDATA_API_KEY2=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

AI bersifat opsional:

```dotenv
GROQ_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
AI_PROVIDER_ORDER=groq,gemini,deepseek
```

Gunakan `.env.example` sebagai referensi lengkap. File runtime VPS disimpan di `/etc/xauusd-analysis.env` dengan permission `0600`, bukan di Git.

## Instalasi Ubuntu VPS

```bash
sudo mkdir -p /opt/xauusd-analysis /var/lib/xauusd-analysis
sudo chown ubuntu:ubuntu /opt/xauusd-analysis /var/lib/xauusd-analysis

git clone https://github.com/FajarWG/wg-goldpulse.git /opt/xauusd-analysis
cd /opt/xauusd-analysis
python3 -m venv .venv
.venv/bin/pip install -e .

sudo install -o root -g root -m 0600 \
  deploy/xauusd-analysis.env.example /etc/xauusd-analysis.env
sudoedit /etc/xauusd-analysis.env

sudo install -m 0644 deploy/*.service deploy/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now \
  xauusd-analysis.timer \
  xauusd-signal.timer \
  xauusd-backtest.timer \
  xauusd-telegram.service
```

Status dan log:

```bash
systemctl status xauusd-analysis.timer xauusd-signal.timer xauusd-backtest.timer
systemctl status xauusd-telegram.service
journalctl -u xauusd-signal.service -n 50 --no-pager
journalctl -u xauusd-telegram.service -n 50 --no-pager
```

## Update VPS dari GitHub

```bash
cd /opt/xauusd-analysis
git pull --ff-only
.venv/bin/pip install -e .
sudo systemctl restart xauusd-telegram.service
sudo systemctl daemon-reload
```

Periksa diff dan release notes sebelum menjalankan update pada VPS produksi.

## Testing

```bash
python -m pytest -q
git diff --check
```

Test tidak memerlukan kredensial live dan tidak melakukan request jaringan.

## Struktur repository

```text
forex/                    indicator, provider, SMC, tracking, backtest, AI
tests/                    unit tests tanpa network
deploy/                   template systemd dan environment VPS
docs/GUIDE.md             flow dan panduan operasional lengkap
docs/STRATEGY-RESEARCH.md hasil validasi V3/V4
main.py                   macro analysis H1/H4/D1
signal_main.py            evaluasi M5/M15
telegram_bot_main.py      Telegram listener dan commands
backtest_main.py          historical replay
usage_main.py             laporan pemakaian Twelve Data
```

## Keamanan

- `.env`, private key, database SQLite, market cache, dan reports tidak boleh di-commit.
- Jangan menaruh credential pada command line, issue, log, atau Telegram chat.
- Bila token pernah terekspos, rotasi melalui provider terkait.
- HTTP error Twelve Data disanitasi agar URL berisi API key tidak masuk ke log aplikasi.

## Asal proyek

WG GoldPulse dikembangkan dari [0xgetz/daily_forex_analysis](https://github.com/0xgetz/daily_forex_analysis). Modul analisis umum, lisensi MIT, dan atribusi upstream tetap dipertahankan.

Dokumentasi lebih rinci tersedia di [docs/GUIDE.md](docs/GUIDE.md) dan [docs/STRATEGY-RESEARCH.md](docs/STRATEGY-RESEARCH.md).
