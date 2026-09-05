# WG GoldPulse

## Identitas

**Nama:** WG GoldPulse

**Tagline:** Analisis dan validasi signal XAUUSD—tanpa auto-trading.
**Deskripsi:** Asisten analisis XAUUSD multi-timeframe dengan signal Smart Money Concepts, validasi hasil otomatis, historical backtest, penjelasan AI, dan pemantauan kuota data.

WG GoldPulse tidak terhubung ke akun broker dan tidak membuka transaksi. Entry, SL, dan TP adalah referensi analisis yang tetap harus dicocokkan dengan harga broker.

## Alur utama

```text
Twelve Data key 1 + key 2
          │
          ├── Setiap jam :00 JST
          │     H1 + H4 + D1 → tren besar → Telegram market update
          │
          ├── Setiap 5 menit mulai :02 JST
          │     M5 → M15 → SMC/confluence → WAIT atau LONG/SHORT
          │                                      │
          │                                      ├── Telegram
          │                                      └── SQLite forward validation
          │
          └── Sabtu 07:30 JST
                Refresh incremental → replay V6 + momentum → Telegram backtest report
```

Urutan `:00` lalu `:02` memastikan analisis besar diperbarui sebelum pemeriksaan signal pertama pada jam tersebut.

## Fitur

### 1. Analisis multi-timeframe

- H1, H4, dan D1 untuk arah besar.
- M15 untuk konfirmasi struktur.
- M5 untuk timing signal.
- EMA, RSI, MACD, ATR, Bollinger, Donchian, range, dan volatility regime.

### 2. Smart Money Concepts

- Market structure.
- BOS dan arah swing.
- Liquidity sweep.
- Fair Value Gap.
- Candlestick confirmation.
- Confluence score.
- Entry, batas salah/SL, dan target 2R.

### 3. Forward validation otomatis

- Semua signal READY disimpan dan dinilai otomatis.
- TP lebih dulu: menang `+2R` pada strategi eksperimen V6.
- SL lebih dulu: kalah `-1R`.
- TP dan SL pada candle M5 yang sama: dihitung kalah secara konservatif.
- Tidak selesai dalam empat jam: kedaluwarsa dan tidak masuk pembagi win rate.
- Analisis Full: maksimal 5 signal per hari, cooldown 30 menit, satu signal aktif.
- Momentum Candle: maksimal 5 signal per hari, cooldown 30 menit, hingga 3 signal aktif bersamaan (dapat diatur lewat `SIGNAL_MAX_ACTIVE_MOMENTUM`).

### 4. Telegram interaktif

- Tidak ada tombol Ambil/Lewati; keputusan transaksi selalu di luar bot.
- `🤖 Analisis AI` menjelaskan hasil teknikal terakhir secara manual/on-demand.
- `📊 Statistik` membuka rekap forward validation.
- `🧪 Backtest` membuka historical backtest terakhir.
- `ℹ️ Bantuan` menampilkan cara memakai bot.
- Hasil signal objektif tidak dipengaruhi tindakan pengguna di Telegram.

### 5. AI explanation

Urutan fallback:

1. Groq — gratis/utama.
2. Gemini — gratis/cadangan.
3. DeepSeek — berbayar/cadangan terakhir.

AI menerima indikator yang sudah dihitung. AI hanya menyusun penjelasan Bahasa Indonesia dan tidak boleh mengubah arah signal, skor, entry, SL, atau TP. Jika semua AI gagal, analisis deterministik dan signal tetap berjalan.

Default endpoint menggunakan antarmuka OpenAI-compatible resmi masing-masing provider. Model dapat diganti lewat `GROQ_MODEL`, `GEMINI_MODEL`, dan `DEEPSEEK_MODEL` tanpa mengubah kode.

Referensi konfigurasi resmi: [Groq OpenAI compatibility](https://console.groq.com/docs/openai), [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai), dan [DeepSeek API](https://api-docs.deepseek.com/).

### 6. Dua API key Twelve Data

- `TWELVEDATA_API_KEY` dan `TWELVEDATA_API_KEY2` dipakai bergantian.
- Jika key pertama gagal atau terkena rate limit, request mencoba key lainnya.
- Semua pemakaian bot tetap dicatat dalam SQLite per tanggal UTC.
- Perhitungan batas tetap konservatif terhadap kuota 800 agar aman bila kedua key berbagi akun/quota.

## Historical backtest

Backtest menyimpan versi strategi secara terpisah agar perubahan aturan tidak menghapus pembanding. Operasional saat ini memakai dua strategi aktif:

- **V6 / Analisis Full (eksperimen/forward-test):** macro H1/H4/D1, struktur M15/M5, composite vote, regime volatilitas, volume direction, liquidity sweep, FVG, candle pattern, dan target 2R.
- **Momentum Candle (`momentum_v1`):** sinyal berbasis aksi harga candle M5 terakhir tanpa ketergantungan macro bias; cocok untuk kondisi momentum, dengan target 2R.
- V1–V5 adalah arsip riset dan tidak dijalankan oleh service mingguan.

Data yang diambil dan disimpan lokal:

- M5: sekitar 90 hari, diambil per potongan 14 hari lalu digabungkan.
- H1: 3.000 candle.
- H4: 1.000 candle.
- D1: 400 candle.

Twelve Data membatasi satu respons historical time series hingga 5.000 data points. Karena itu periode M5 dibagi menjadi potongan 14 hari, disimpan di cache, lalu replay strategi dilakukan sepenuhnya di VPS. Refresh mingguan hanya memperbarui bagian data terbaru. Referensi: [Twelve Data — Getting historical data](https://support.twelvedata.com/en/articles/5214728-getting-historical-data).

### Otomatis

Backtest mingguan berjalan setiap Sabtu pukul 07:30 JST, setelah sesi mingguan XAUUSD ditutup. Cache lama dipertahankan; hanya bagian data terbaru yang diambil. Replay default memakai 90 hari terakhir dan hasil JSON/CSV (Analisis Full V6 + Momentum Candle) dikirim ke Telegram.

### Manual

```bash
# Refresh data dan jalankan backtest lengkap melalui service
sudo systemctl start xauusd-backtest.service

# Lihat hasil dari Telegram
/backtest
```

Menjalankan tanpa mengambil data baru:

```bash
cd /opt/xauusd-analysis
# Analisis Full V6 saja
sudo -u ubuntu /opt/xauusd-analysis/.venv/bin/python backtest_main.py --strategy v6

# Semua versi + momentum candle
sudo -u ubuntu /opt/xauusd-analysis/.venv/bin/python backtest_main.py --strategy all
```

Hasil utama:

- Jumlah evaluasi dan signal.
- Menang, kalah, kedaluwarsa.
- Win rate.
- Total R.
- Profit factor.
- Maximum drawdown.
- Skor kandidat tertinggi.
- CSV setiap transaksi simulasi.

Backtest historis dan forward validation harus dibaca terpisah. Backtest membantu menyaring strategi; forward validation mengukur perilaku sistem pada data baru yang belum pernah dilihat.

## Hasil validasi strategi aktif V5 (arsip)

V5 sudah digantikan V6. Hasil di bawah disimpan sebagai arsip riset.

Validasi setelah koreksi matematika pada 2 September 2026:

```text
Periode          : 2026-04-27 → 2026-09-01
Data M5          : 36.496 candle
Evaluasi         : 23.526
Signal           : 210
Menang           : 46
Kalah            : 98
Kedaluwarsa      : 66
Win rate         : 31,9%
Total            : -6R
Profit factor    : 0,94
Maximum drawdown : 20R
P-value          : 0,687 (sign-randomisation)
```

Kesimpulan: **V5 belum terbukti memiliki edge dan hanya dijalankan sebagai eksperimen/forward-test.** Skor konfirmasi bukan probabilitas menang. AI tidak boleh mengubah keputusan mesin.

## Arsip hasil strategi v1

Backtest pertama yang lengkap dijalankan pada 1 September 2026:

```text
Periode          : 2026-06-03 → 2026-09-01
Data M5          : 25.921 candle
Evaluasi         : 15.548
Signal           : 165
Menang           : 32
Kalah            : 83
Kedaluwarsa      : 50
Win rate         : 27,8%
Total            : -19R
Profit factor    : 0,77
Maximum drawdown : 19R
```

Kesimpulan: **strategi v1 belum layak dianggap profitable dan tidak boleh dibuat lebih agresif berdasarkan hasil ini**. Live bot tetap analysis-only dan diberi label eksperimental.

Temuan kalibrasi setelah seluruh candle dinormalisasi ke UTC:

- Skor 65 menghasilkan `−1R`, skor 70 `−5R`, dan skor 75 `−9R`.
- Bullish FVG hanya mencapai impas `0R`; bearish FVG menghasilkan `−14R`.
- Skor tinggi belum berarti probabilitas menang lebih tinggi.
- Belum ada subset sederhana yang cukup kuat untuk langsung dijadikan strategi v2.
- Semua perubahan rule berikutnya harus dibuat sebagai `strategy_version=v2`, lalu diuji pada periode out-of-sample sebelum dipakai live.

Artefak rinci tersedia dalam `latest.json` dan `latest_trades.csv`; CSV menyimpan arah, skor, struktur M15/M5, RSI, liquidity event, FVG, candle pattern, waktu buka/tutup, serta hasil R setiap signal.

## Hasil backtest strategi v2

V2 memakai aturan yang dibekukan sebelum holdout: keselarasan H1/H4/D1, M15 dan M5; sesi London 06:00–11:59 UTC; RSI 35–65; target 1,25R; batas salah 1R; dan timeout empat jam.

### Periode pengembangan 90 hari

```text
Periode          : 2026-06-03 → 2026-09-01
Signal           : 61
Menang           : 24
Kalah            : 23
Kedaluwarsa      : 14
Win rate         : 51,1%
Total            : +7R
Profit factor    : 1,30
Maximum drawdown : 4,8R
```

### Holdout yang belum dipakai menyusun V2

```text
Periode          : 2026-03-05 → 2026-06-02
Signal           : 43
Menang           : 16
Kalah            : 19
Kedaluwarsa      : 8
Win rate         : 45,7%
Total            : +1R
Profit factor    : 1,05
Maximum drawdown : 8,75R
```

Hasil gabungan 180 hari adalah 106 signal, 41 menang, 43 kalah, 22 kedaluwarsa, win rate 48,8%, total `+8,25R`, profit factor 1,19, dan maximum drawdown 10,75R.

**Keputusan:** V2 jauh lebih baik daripada V1, tetapi belum lolos untuk menggantikan strategi live. Keunggulan holdout hanya `+1R` dengan profit factor 1,05 dan drawdown 8,75R. V2 sebaiknya dijalankan sebagai shadow/forward test sampai memperoleh sampel baru yang memadai. Aturan holdout tidak boleh disetel ulang hanya untuk mempercantik hasil historis.

## Cara menjalankan dan mengecek

Service utama:

```bash
systemctl status xauusd-analysis.timer
systemctl status xauusd-signal.timer
systemctl status xauusd-telegram.service
systemctl status xauusd-backtest.timer
```

Log:

```bash
sudo journalctl -u xauusd-analysis.service -n 50 --no-pager
sudo journalctl -u xauusd-signal.service -n 50 --no-pager
sudo journalctl -u xauusd-telegram.service -n 50 --no-pager
sudo journalctl -u xauusd-backtest.service -n 100 --no-pager
```

Pemakaian data:

```bash
xauusd-usage
```

## Perintah Telegram

```text
/status    Rekap signal forward test
/backtest  Historical backtest terakhir
/usage     Pemakaian Twelve Data
/ai        AI aktif dan urutan fallback
/help      Cara memakai bot
```

## Contoh pesan Telegram

### Market update

```text
🥇 WG GOLDPULSE — MARKET UPDATE
━━━━━━━━━━━━━━━━
XAU/USD · 4453.10
Arah pasar: KONFLIK
Kekuatan: LOW

Timeframe:
• H1: SIDEWAYS · RSI 55.2
• H4: TURUN · RSI 38.0
• D1: NAIK · RSI 52.1

Belum ada signal arah; tunggu konfirmasi M5/M15.

[🤖 Analisis AI] [📊 Statistik] [🧪 Backtest]
[ℹ️ Bantuan]
```

### Signal READY

```text
🥇 WG GOLDPULSE — SIGNAL SIAP BUY
Waktu: 2026-09-02 08:15 UTC
━━━━━━━━━━━━━━━━━━━━

📍 RENCANA HARGA
Entry referensi: 4450.20
Stop loss: 4443.70
Take profit: 4463.20
Risk/reward: 1:2.0

🔎 KONFIRMASI
Skor: 75/100
Arah H1/H4/D1: naik
Struktur M15: bullish
Struktur M5: bullish
RSI M5: 52.4
Kondisi pasar: normal
Volume: searah harga

Skor adalah kekuatan konfirmasi, bukan persentase peluang menang.

✅ ALASAN SIGNAL
• H1/H4/D1 macro bias bullish
• M15 bullish structure
• M5 structure bullish

Bot mencatat hasil sampai TP, SL, atau kedaluwarsa. Tidak ada transaksi otomatis.

[🤖 Analisis AI] [📊 Statistik] [🧪 Backtest]
[ℹ️ Bantuan]
```

### Statistik

```text
📊 STATISTIK FORWARD TEST
━━━━━━━━━━━━━━━━━━━━
Signal tercatat: 15
Sudah selesai: 12

HASIL SELESAI
✅ Benar: 7
❌ Salah: 5
🎯 Win rate: 58.3%

STATUS LAIN
⏳ Masih aktif: 1
⌛ Kedaluwarsa: 2
📈 Akumulasi hasil: +9.0R
━━━━━━━━━━━━━━━━━━━━
```

### Backtest

```text
🧪 WG GOLDPULSE — HISTORICAL BACKTEST
━━━━━━━━━━━━━━━━
Periode: 2026-04-27 → 2026-09-01
Data M5: 36,496 candle
Evaluasi: 23,526

Signal: 210
✅ Menang: 46
❌ Kalah: 98
⌛ Kedaluwarsa: 66
🎯 Win rate: 31.9%

📈 Total: -6.0R
📉 Max drawdown: 20.0R
⚖️ Profit factor: 0.94
🎲 P-value (sign-randomisation): 0.687
━━━━━━━━━━━━━━━━
```

## Batas tanggung jawab setiap komponen

| Komponen | Tugas | Tidak boleh melakukan |
| --- | --- | --- |
| Twelve Data | Memberikan OHLC | Menentukan signal |
| Indicator engine | Menghitung indikator | Menulis narasi AI |
| SMC engine | Menentukan WAIT/LONG/SHORT | Membuka order broker |
| AI router | Menjelaskan hasil | Mengubah angka atau signal |
| Forward tracker | Menilai signal baru | Mengubah hasil berdasarkan tindakan user |
| Backtest engine | Replay data historis | Menjamin performa masa depan |
| Telegram | Menampilkan analisis dan statistik | Menjalankan auto-trading |

## Arah improvement berikutnya

1. Tambahkan spread dan slippage agar backtest lebih realistis.
2. Pisahkan laporan per versi strategi sebelum threshold diubah.
3. Tambahkan walk-forward test dan periode out-of-sample.
4. Bandingkan performa berdasarkan sesi Tokyo, London, dan New York.
5. Analisis statistik setup BOS, sweep, FVG, dan candle pattern secara terpisah.
6. Ubah threshold hanya berdasarkan backtest + forward test, bukan berdasarkan sedikitnya signal semata.

## Riset target win rate 70–90% (V3/V4)

Riset lanjutan memakai 243.335 candle M5 selama tiga tahun dan aturan yang dibekukan sebelum setiap holdout. V4 sempat mencapai 75,3% pada development dua tahun, tetapi hanya 54,4% pada locked holdout tahun ketiga dan menghasilkan `−10,8R` setelah stress cost `0,05R` per signal. Gabungan tiga tahun adalah 68,5% tetapi `−3,5R` setelah cost.

Karena itu V3, V3.1, dan V4 **tidak dipromosikan ke live**. Target 70–90% belum terbukti out-of-sample. Laporan metodologi dan hasil lengkap tersedia di `WG-GoldPulse-Strategy-Research-V4.md`.
