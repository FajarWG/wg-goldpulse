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
| V5 | Eksperimen/forward-test | 27 Apr–1 Sep 2026: 31,9% WR, −6R, PF 0,94; belum lolos promotion gate |
| V6 | Aktif/forward-test | Dual signal: Analisis Full + Momentum Candle; hasil masih dikumpulkan |

Tidak ada versi yang saat ini boleh dianggap profitable. Signal Telegram harus diperlakukan sebagai eksperimen manual, bukan instruksi trading.

## Fitur

- Analisis H1/H4/D1 dengan EMA, RSI, MACD, ATR, Bollinger, Donchian, volatility regime, support, dan resistance.
- Konfirmasi M5/M15 berbasis market structure, BOS, liquidity sweep, FVG, candle pattern, dan confluence score.
- Signal LONG/SHORT/WAIT dengan entry, batas salah/SL, dan target referensi.
- Dua tipe signal independen: **Analisis Full** (SMC confluence) dan **Momentum Candle** (candle-action). Masing-masing punya cooldown, daily cap, dan statistik terpisah.
- Telegram commands dan inline buttons untuk analisis AI, statistik, backtest, dan bantuan. Tidak ada tombol Ambil/Lewati.
- Forward validation otomatis: menang, kalah, expired, win rate, dan akumulasi R.
- Historical replay strategi aktif V6 plus strategi momentum dengan cache data UTC, versioned output, drawdown, profit factor, p-value sign-randomisation, dan breakdown regime. V1–V5 tetap terdokumentasi sebagai riset lama.
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
python backtest_main.py --strategy v6
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
sudo systemctl daemon-reload
sudo systemctl restart xauusd-signal.timer xauusd-backtest.timer
sudo systemctl restart xauusd-telegram.service
```

`xauusd-signal` dan `xauusd-backtest` adalah oneshot yang dijalankan timer, jadi restart timernya (bukan hanya service-nya) supaya reload unit terbaru. Tambahkan `SIGNAL_MAX_ACTIVE_MOMENTUM=3` ke `/etc/xauusd-analysis.env` bila ingin mengatur jumlah momentum aktif bersamaan (default 3 di kode).

Periksa diff dan release notes sebelum menjalankan update pada VPS produksi.

## Backtest

Backtest memakai cache M5/H1/H4/D1 di `state_dir/backtest/cache`. Refresh cache kalau datanya kurang (dipakai ulang pada run berikutnya):

```bash
python backtest_main.py --fetch-only
python backtest_main.py --refresh --fetch-only
```

Jalankan replay (strategi aktif `v6` default, atau `--strategy all` untuk semua versi):

```bash
# Analisis Full saja
python backtest_main.py --strategy v6

# Semua versi + momentum candle
python backtest_main.py --strategy all
```

Output di terminal menampilkan rekap Analisis Full lalu rekap Momentum Candle. Hasil tersimpan di `state_dir/backtest/<label>/latest.json` (`momentum_v1` untuk momentum). Kirim ke Telegram:

```bash
python backtest_main.py --strategy all --notify
```

Di VPS, backtest lengkap (refresh data + kirim ke Telegram) bisa dipanggil lewat service mingguan:

```bash
sudo systemctl start xauusd-backtest.service
```

Parameter sinyal dibaca dari env: `SIGNAL_MAX_FULL_PER_DAY`, `SIGNAL_MAX_MOMENTUM_PER_DAY`, `SIGNAL_MAX_ACTIVE_MOMENTUM`, `SIGNAL_COOLDOWN_MINUTES`, `SIGNAL_TIMEOUT_MINUTES`.

### Riset filter signal

`research_main.py` membaca `latest_trades.csv` dan membandingkan win rate, total R, profit factor, dan drawdown per kelompok (skor, regime, alignment, arah). Pakai untuk mencari filter yang menyaring signal jelek:

```bash
# Momentum candle (path default)
python research_main.py

# Analisis Full
python research_main.py /var/lib/xauusd-analysis/backtest/v6/latest_trades.csv

# Hanya trade dengan skor >= 65
python research_main.py --min-score 65
```

Hasil riset ini adalah alat bantu, bukan dasar langsung untuk mengubah strategi. Filter baru harus lolos periode out-of-sample/holdout sebelum dipakai live.

### Sweep reward:risk momentum

`research_rr.py` memuat cache sekali lalu mencoba beberapa nilai reward:risk untuk strategi momentum, supaya nilai TP/SL dipilih dari data (bukan tebak):

```bash
python research_rr.py
```

Output membandingkan jumlah signal, win rate, total R, profit factor, dan drawdown untuk setiap reward:risk. Nilai momentum reward:risk bisa di-override lewat `--momentum-rr` atau `SIGNAL_MOMENTUM_REWARD_R`.

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
research_main.py          analisis filter signal dari latest_trades.csv
research_rr.py            sweep reward:risk strategi momentum
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
