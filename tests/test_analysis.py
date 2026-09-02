"""Indicator maths and structural reads.

Indicators are verified against hand-computable cases and known-shape synthetic
series, not against whatever the code currently returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forex.analysis import (
    InsufficientDataError,
    adx,
    align_timeframes,
    analyse_timeframe,
    atr,
    bollinger,
    candle_patterns,
    composite_vote,
    donchian,
    ema,
    hysteresis,
    macd,
    rsi,
    sma,
    support_resistance_clusters,
    true_range,
    volatility_regime,
)
from forex.instruments import parse_symbol

from tests.helpers import make_candles

EURUSD = parse_symbol("EURUSD")
USDJPY = parse_symbol("USDJPY")


class TestMovingAverages:
    def test_sma_matches_manual_mean(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert sma(series, 3).iloc[-1] == pytest.approx(4.0)  # (3+4+5)/3

    def test_sma_is_nan_before_period_filled(self):
        series = pd.Series([1.0, 2.0, 3.0])
        assert np.isnan(sma(series, 3).iloc[1])
        assert not np.isnan(sma(series, 3).iloc[2])

    def test_ema_of_constant_series_equals_constant(self):
        series = pd.Series([2.5] * 30)
        assert ema(series, 10).iloc[-1] == pytest.approx(2.5)

    def test_ema_reacts_faster_than_sma(self):
        """On a step change the EMA must move further than the SMA."""
        series = pd.Series([1.0] * 20 + [2.0] * 5)
        assert ema(series, 10).iloc[-1] > sma(series, 10).iloc[-1]


class TestRSI:
    def test_monotonic_rise_gives_100(self):
        series = pd.Series(np.arange(1.0, 40.0))
        assert rsi(series, 14).iloc[-1] == pytest.approx(100.0)

    def test_monotonic_fall_gives_0(self):
        series = pd.Series(np.arange(40.0, 1.0, -1.0))
        assert rsi(series, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_bounded_between_0_and_100(self):
        values = rsi(make_candles(n=300, seed=3)["close"], 14).dropna()
        assert values.between(0.0, 100.0).all()

    def test_uptrend_reads_higher_than_downtrend(self):
        up = rsi(make_candles(n=200, drift=0.0005, noise=0.0001)["close"]).iloc[-1]
        down = rsi(make_candles(n=200, drift=-0.0005, noise=0.0001)["close"]).iloc[-1]
        assert up > down


class TestTrueRangeAndATR:
    def test_true_range_uses_widest_of_three_measures(self):
        df = pd.DataFrame(
            {
                "open": [10.0, 10.0],
                "high": [11.0, 12.0],
                "low": [9.0, 11.5],
                "close": [10.5, 11.8],
            }
        )
        # Second candle gaps up: high-low is 0.5 but high-prev_close is 1.5.
        assert true_range(df).iloc[-1] == pytest.approx(1.5)

    def test_atr_positive_and_finite(self):
        value = atr(make_candles(n=100), 14).dropna().iloc[-1]
        assert value > 0 and np.isfinite(value)

    def test_atr_larger_for_noisier_series(self):
        calm = atr(make_candles(n=200, noise=0.00005), 14).dropna().iloc[-1]
        wild = atr(make_candles(n=200, noise=0.0020), 14).dropna().iloc[-1]
        assert wild > calm


class TestMACDAndBands:
    def test_macd_histogram_is_line_minus_signal(self):
        frame = macd(make_candles(n=120)["close"]).dropna()
        assert frame["histogram"].iloc[-1] == pytest.approx(
            frame["macd"].iloc[-1] - frame["signal"].iloc[-1]
        )

    def test_macd_positive_in_uptrend(self):
        frame = macd(make_candles(n=200, drift=0.0006, noise=0.0001)["close"]).dropna()
        assert frame["macd"].iloc[-1] > 0

    def test_bollinger_bands_ordered(self):
        frame = bollinger(make_candles(n=100)["close"]).dropna()
        row = frame.iloc[-1]
        assert row["lower"] < row["middle"] < row["upper"]

    def test_bollinger_width_zero_on_constant_series(self):
        frame = bollinger(pd.Series([1.5] * 40)).dropna()
        assert frame["upper"].iloc[-1] == pytest.approx(frame["lower"].iloc[-1])

    def test_donchian_channel_contains_price(self):
        df = make_candles(n=100)
        channel = donchian(df, 20).dropna()
        last = channel.iloc[-1]
        assert last["lower"] <= df["close"].iloc[-1] <= last["upper"]


class TestAnalyseTimeframe:
    def test_uptrend_detected(self, uptrend):
        read = analyse_timeframe(uptrend, EURUSD, "H1")
        assert read.trend.direction == "up"
        assert read.candles == len(uptrend)

    def test_downtrend_detected(self, downtrend):
        assert analyse_timeframe(downtrend, EURUSD, "H1").trend.direction == "down"

    def test_flat_series_is_sideways(self, flat):
        """MAs inside noise must not be reported as a trend."""
        assert analyse_timeframe(flat, EURUSD, "H1").trend.direction == "sideways"

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            analyse_timeframe(pd.DataFrame({"close": [1.0, 2.0]}), EURUSD, "H1")

    def test_single_candle_raises(self):
        with pytest.raises(InsufficientDataError):
            analyse_timeframe(make_candles(n=1), EURUSD, "H1")

    def test_short_history_degrades_with_warning(self):
        """Fewer candles than the slow MA yields a warning, not an exception."""
        read = analyse_timeframe(make_candles(n=25), EURUSD, "H1")
        assert read.warnings
        assert "degraded" in read.warnings[0]

    def test_pip_values_scale_with_instrument(self):
        """The same candles read on a JPY pair give 100x smaller pip counts.

        Compared on the raw ATR rather than the rounded report field, since
        ``atr_pips`` is rounded to one decimal for display and a sub-pip JPY
        value would lose the ratio to rounding.
        """
        df = make_candles(n=200, noise=0.0005)
        raw_atr = atr(df, 14).dropna().iloc[-1]
        assert EURUSD.pips(raw_atr) == pytest.approx(USDJPY.pips(raw_atr) * 100)

        # The rendered fields still differ by two orders of magnitude.
        as_eur = analyse_timeframe(df, EURUSD, "H1").volatility.atr_pips
        as_jpy = analyse_timeframe(df, USDJPY, "H1").volatility.atr_pips
        assert as_eur > as_jpy * 50

    def test_range_position_bounded(self):
        position = analyse_timeframe(make_candles(n=100), EURUSD, "H1").levels.position_in_range
        assert 0.0 <= position <= 1.0

    def test_to_dict_is_json_serialisable(self):
        import json

        payload = analyse_timeframe(make_candles(n=100), EURUSD, "H1").to_dict()
        assert json.loads(json.dumps(payload))["timeframe"] == "H1"

    def test_unsorted_index_is_handled(self):
        """Providers sometimes return newest-first; analysis must sort."""
        df = make_candles(n=100)
        reversed_df = df.iloc[::-1]
        assert analyse_timeframe(reversed_df, EURUSD, "H1").last_close == pytest.approx(
            analyse_timeframe(df, EURUSD, "H1").last_close
        )


class TestAlignTimeframes:
    def _read(self, df, tf):
        return analyse_timeframe(df, EURUSD, tf)

    def test_all_up_is_high_confidence(self, uptrend):
        reads = {tf: self._read(uptrend, tf) for tf in ("H1", "H4", "D1")}
        result = align_timeframes(reads)
        assert result["verdict"] == "up"
        assert result["confidence"] == "high"

    def test_conflict_is_low_confidence(self, uptrend, downtrend):
        reads = {"H1": self._read(uptrend, "H1"), "D1": self._read(downtrend, "D1")}
        result = align_timeframes(reads)
        assert result["verdict"] == "conflicted"
        assert result["confidence"] == "low"

    def test_all_sideways(self, flat):
        reads = {tf: self._read(flat, tf) for tf in ("H1", "D1")}
        assert align_timeframes(reads)["verdict"] == "sideways"

    def test_partial_agreement_is_medium(self, uptrend, flat):
        """One trending plus one flat is agreement, but weaker."""
        reads = {"H1": self._read(uptrend, "H1"), "D1": self._read(flat, "D1")}
        result = align_timeframes(reads)
        assert result["verdict"] == "up"
        assert result["confidence"] == "medium"

    def test_note_always_present(self, uptrend):
        assert align_timeframes({"H1": self._read(uptrend, "H1")})["note"]


def _rising_volume_candles(n=200, drift=0.0005, start=1.1):
    """Synthetic OHLC with monotonically rising volume (for volume tests)."""
    df = make_candles(n=n, drift=drift, start=start, noise=0.0002)
    df["volume"] = np.linspace(500.0, 5000.0, n)
    return df


class TestADX:
    def test_adx_bounded_and_finite(self):
        value = adx(make_candles(n=300, drift=0.0005, noise=0.0001)).dropna().iloc[-1]
        assert 0.0 <= value <= 100.0

    def test_adx_higher_in_strong_trend(self):
        trending = adx(make_candles(n=300, drift=0.001, noise=0.00005)).dropna().iloc[-1]
        choppy = adx(make_candles(n=300, drift=0.0, noise=0.0005)).dropna().iloc[-1]
        assert trending > choppy

    def test_trend_read_carries_adx(self, uptrend):
        read = analyse_timeframe(uptrend, EURUSD, "H1")
        assert read.trend.adx is not None


class TestVolatilityRegime:
    def test_short_history_unknown(self):
        assert volatility_regime(make_candles(n=5)["close"]) == "unknown"

    def test_normal_series_is_normal(self):
        assert volatility_regime(make_candles(n=400, noise=0.0005)["close"]) == "normal"

    def test_volatility_shock_is_high_vol(self):
        rng = np.random.default_rng(0)
        close = make_candles(n=300, noise=0.0001)["close"]
        shocked_tail = close.iloc[-40:].values * (1 + rng.normal(0, 0.004, 40))
        tail = pd.Series(shocked_tail, index=close.index[-40:])
        shocked = pd.concat([close.iloc[:-40], tail])
        assert volatility_regime(shocked) == "high_vol"


class TestHysteresis:
    def test_enters_above_threshold_and_stays(self):
        series = pd.Series([0.2, 0.3, 0.7, 0.6, 0.5, 0.5])
        # Once above 0.65 it latches on; 0.5 is still above the 0.45 exit.
        assert hysteresis(series, enter=0.65, exit=0.45) == "on"

    def test_turns_off_below_exit(self):
        series = pd.Series([0.8, 0.5, 0.4])
        assert hysteresis(series, enter=0.65, exit=0.45) == "off"

    def test_dead_band_prevents_flapping(self):
        noisy = pd.Series([0.5, 0.55, 0.5, 0.6, 0.5, 0.58])
        assert hysteresis(noisy, enter=0.65, exit=0.45) == "off"


class TestCandlePatterns:
    def test_engulfing_detected(self):
        df = pd.DataFrame(
            {
                "open": [1.10, 1.09],
                "high": [1.11, 1.13],
                "low": [1.09, 1.085],
                "close": [1.095, 1.125],
            }
        )
        names = {p["pattern"] for p in candle_patterns(df)}
        assert "bullish_engulfing" in names

    def test_doji_detected(self):
        df = pd.DataFrame(
            {
                "open": [1.10, 1.10],
                "high": [1.11, 1.105],
                "low": [1.09, 1.095],
                "close": [1.095, 1.1005],
            }
        )
        names = {p["pattern"] for p in candle_patterns(df)}
        assert "doji" in names

    def test_empty_on_single_candle(self):
        assert candle_patterns(make_candles(n=1)) == []


class TestSupportResistanceClusters:
    def test_returns_list_of_levels(self):
        df = make_candles(n=300, noise=0.0008)
        levels = support_resistance_clusters(df)
        assert isinstance(levels, list)
        assert all("level" in item and "strength" in item for item in levels)

    def test_near_flat_series_gives_levels_inside_price_range(self):
        df = make_candles(n=100, drift=0.0, noise=0.000001)
        levels = support_resistance_clusters(df)
        assert isinstance(levels, list)
        if levels:
            low, high = float(df["low"].min()), float(df["high"].max())
            assert all(low <= item["level"] <= high for item in levels)


class TestCompositeVote:
    def test_uptrend_votes_long(self, uptrend):
        vote = composite_vote(uptrend)
        assert vote["trend"] == "long"

    def test_flat_volume_never_confirms(self, flat):
        """Constant volume must not fabricate a volume confirmation."""
        vote = composite_vote(flat)
        assert vote["volume_confirm"] is None

    def test_rising_volume_in_uptrend_confirms(self):
        df = _rising_volume_candles()
        vote = composite_vote(df)
        assert vote["volume_confirm"] is True

    def test_composite_included_in_read(self, uptrend):
        read = analyse_timeframe(uptrend, EURUSD, "H1")
        assert read.composite.get("vote") in {"long", "short", "neutral"}
