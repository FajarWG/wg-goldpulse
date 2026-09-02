# WG GoldPulse — Strategy Research V3/V4

Tanggal pengujian: 1 September 2026

Instrumen: XAU/USD

Data: 243.335 candle M5, September 2023–September 2026
Mode: historical analysis only; tidak membuka transaksi broker

## Tujuan

Menguji apakah strategi dapat mencapai win rate 70–90% secara out-of-sample, tanpa mengubah aturan setelah melihat holdout dan tanpa mengecilkan target sampai expectancy tidak bermakna.

## Metodologi

- H1, H4, dan D1 diturunkan dari sumber M5 UTC yang sama agar seluruh periode memiliki macro history konsisten.
- TP dan SL pada candle yang sama dihitung kalah secara konservatif.
- Timeout empat jam.
- Maksimal tiga signal per hari, cooldown 45 menit, dan satu signal simulasi aktif.
- Setiap kandidat dibekukan sebelum periode holdout berikutnya dibuka.
- Stress cost `0,05R` dikenakan pada setiap signal untuk menguji spread/slippage secara konservatif.

## V3 — target 1,25R

Aturan: semua timeframe searah, struktur M5 searah, RSI 35–65, struktur murni skor 65, jam 07:00–10:59 UTC.

| Segmen | Signal | W/L/E | WR | Total | PF | DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 75 | 32/30/13 | 51,6% | +10R | 1,33 | 5R |
| Unseen | 85 | 33/39/13 | 45,8% | +2,25R | 1,06 | 11,75R |

Keputusan: gagal target akurasi dan gagal promotion gate.

## V3.1 — target 1,25R

Aturan: skor 65; RSI 35–44,99; LONG hanya jam 07 UTC; SHORT jam 07, 09, dan 10 UTC.

| Segmen | Signal | W/L/E | WR | Raw total | Setelah cost | Net PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 33 | 19/10/4 | 65,5% | +13,75R | +12,10R | 2,13 |
| Locked unseen | 19 | 8/8/3 | 50,0% | +2R | +1,05R | 1,12 |

Keputusan: gagal minimum sampel, win rate, dan profit-factor gate.

## V4 — high-accuracy target 0,5R

Aturan dibekukan sebelum holdout tahun ketiga: struktur murni skor 65, jam 07 UTC, RSI 35–65, target 0,5R.

| Segmen | Signal | W/L/E | WR | Raw total | Setelah cost | Net PF | Net DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development 2 tahun | 104 | 73/24/7 | **75,3%** | +12,5R | +7,3R | 1,29 | 3,75R |
| Locked holdout tahun ke-3 | 46 | 25/21/0 | **54,4%** | −8,5R | −10,8R | 0,51 | 13,35R |
| Gabungan 3 tahun | 150 | 98/45/7 | 68,5% | +4R | −3,5R | 0,93 | 13,35R |

Promotion gate V4:

- Minimum 30 signal unseen: lolos.
- Win rate unseen ≥70%: gagal.
- Net profit factor ≥1,2: gagal.
- Net total R positif: gagal.
- Net drawdown ≤10R: gagal.

## Kesimpulan

Target 70–90% **belum tercapai secara valid**. V4 mencapai 75,3% pada development, tetapi runtuh menjadi 54,4% pada holdout yang belum pernah dilihat. Memasang V4 live atau menyetel ulang cutoff berdasarkan holdout ini akan menjadi overfitting.

V1, V2, V3, V3.1, dan V4 tetap tersimpan sebagai versi terpisah. Strategi live tidak diganti. Langkah yang layak berikutnya bukan grid threshold tambahan, melainkan menambah fitur yang dapat menjelaskan perubahan regime:

1. Spread broker aktual dan slippage berdasarkan jam.
2. Kalender berita berdampak tinggi (CPI, NFP, FOMC, keputusan suku bunga).
3. Regime classifier berbasis volatilitas/trend sebelum memilih sub-strategy.
4. Bid/ask atau tick data, bukan OHLC midpoint saja.
5. Walk-forward forward test minimal 50–100 signal baru sebelum promosi.

AI/LLM tidak digunakan untuk memilih hasil backtest atau mengubah label menang/kalah. AI tetap hanya boleh menjelaskan output deterministik.
