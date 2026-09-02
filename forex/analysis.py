"""Technical indicators and structure detection for FX candles.

Everything here is pure: a DataFrame of OHLC goes in, numbers come out. No
network, no configuration, no LLM. That keeps the analysis layer fully testable
against synthetic data, which is how the test suite verifies indicator maths
without depending on a live provider.

Results are expressed in pips wherever a price distance is involved, because a
50-point move means nothing in FX unless you know the pair's pip size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .instruments import Instrument

# Column names expected on every candle frame.
REQUIRED_COLUMNS = ("open", "high", "low", "close")


class InsufficientDataError(ValueError):
    """Raised when a frame has too few candles for the requested calculation."""


def _validate(df: pd.DataFrame, minimum: int) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"candle frame missing columns: {missing}")
    if len(df) < minimum:
        raise InsufficientDataError(
            f"need at least {minimum} candles, got {len(df)}"
        )


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    Uses an exponential mean with ``alpha = 1/period``, which is Wilder's
    smoothing, rather than a simple mean of gains and losses.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # All-gain windows have zero average loss -> RSI is 100 by definition.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, out.where(avg_loss == 0.0, 0.0))
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": line, "signal": signal_line, "histogram": line - signal_line}
    )


def bollinger(series: pd.Series, period: int = 20, stddev: float = 2.0) -> pd.DataFrame:
    middle = sma(series, period)
    # Population std matches the conventional Bollinger definition.
    sigma = series.rolling(window=period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + stddev * sigma,
            "lower": middle - stddev * sigma,
        }
    )


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Rolling high/low channel — the cleanest read on FX range breakouts."""
    return pd.DataFrame(
        {
            "upper": df["high"].rolling(window=period, min_periods=period).max(),
            "lower": df["low"].rolling(window=period, min_periods=period).min(),
        }
    )


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ADX (directional strength), 0-100.

    ``>= 25`` conventionally marks a trending market; below that the market is
    ranging and directional reads carry less weight.
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    tr = true_range(df)
    atr_smooth = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume. Flat when volume is constant or absent."""
    if "volume" not in df.columns:
        return pd.Series(0.0, index=df.index)
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"].fillna(0.0)).cumsum()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> Optional[float]:
    """Latest volume vs its trailing mean; None when volume is unusable."""
    if "volume" not in df.columns:
        return None
    volume = df["volume"].astype(float)
    baseline = volume.rolling(window=period, min_periods=period).mean()
    ratio = (volume / baseline).dropna()
    if ratio.empty:
        return None
    return float(ratio.iloc[-1])


def volatility_regime(close: pd.Series, short: int = 20, long: int = 252) -> str:
    """Classify recent volatility vs the longer baseline.

    Mirrors the Vibe-Trading bull/bear/high-vol classifier on the volatility
    axis alone: short-window realised vol above 1.5x the long-window mean marks
    a high-volatility regime. Returns ``"high_vol"``, ``"normal"`` or
    ``"unknown"`` when there is not enough history.
    """
    if len(close) < long + 1:
        if len(close) < short + 2:
            return "unknown"
        long = max(short + 1, len(close) // 2)
    returns = close.pct_change(fill_method=None).dropna()
    if len(returns) < long + 1:
        return "unknown"
    short_vol = float(returns.tail(short).std(ddof=0))
    long_vol = float(returns.tail(long).std(ddof=0))
    if not np.isfinite(short_vol) or not np.isfinite(long_vol) or long_vol <= 0:
        return "unknown"
    return "high_vol" if short_vol > 1.5 * long_vol else "normal"


def hysteresis(values: pd.Series, enter: float = 0.65, exit: float = 0.45) -> str:
    """Schmitt-trigger state over a bounded scalar series.

    The state flips to ``"on"`` only above ``enter`` and back to ``"off"`` only
    below ``exit``. The dead band between the two thresholds suppresses the
    whipsaw a single threshold would cause around a noisy boundary.
    """
    state = False
    for value in values.dropna().tolist():
        if state:
            if value < exit:
                state = False
        elif value > enter:
            state = True
    return "on" if state else "off"


def candle_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect common candlestick patterns on the last candle.

    Returns a list of ``{"pattern", "direction"}`` dicts where direction is +1
    for bullish patterns, -1 for bearish and 0 for neutral. Mirrors the
    vectorised pattern family in Vibe-Trading: engulfing, harami, piercing
    line, dark cloud cover, doji, hammer, inverted hammer, shooting star,
    spinning top.
    """
    if len(df) < 2:
        return []
    current = df.iloc[-1]
    previous = df.iloc[-2]
    open_, high, low, close = (float(current[c]) for c in ("open", "high", "low", "close"))
    p_open, p_high, p_low, p_close = (float(previous[c]) for c in ("open", "high", "low", "close"))

    body = abs(close - open_)
    total_range = high - low
    if body <= 0 or total_range <= 0:
        return []

    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    real_body = body / total_range
    range_fraction = body / (p_high - p_low) if p_high > p_low else 0.0
    previous_bullish = p_close > p_open
    current_bullish = close > open_

    patterns: List[Dict[str, Any]] = []

    def _add(name: str, direction: int) -> None:
        patterns.append({"pattern": name, "direction": direction})

    if current_bullish and not previous_bullish and close >= p_open and open_ <= p_close:
        _add("bullish_engulfing", +1)
    elif not current_bullish and previous_bullish and close <= p_open and open_ >= p_close:
        _add("bearish_engulfing", -1)

    previous_body_low = min(p_open, p_close)
    previous_body_high = max(p_open, p_close)
    current_inside_previous = (
        min(open_, close) >= previous_body_low
        and max(open_, close) <= previous_body_high
    )
    if previous_bullish and not current_bullish and current_inside_previous:
        _add("bearish_harami", -1)
    elif not previous_bullish and current_bullish and current_inside_previous:
        _add("bullish_harami", +1)

    if current_bullish and not previous_bullish and open_ < p_close and close > (p_open + p_close) / 2:
        _add("bullish_piercing", +1)
    if not current_bullish and previous_bullish and open_ > p_close and close < (p_open + p_close) / 2:
        _add("bearish_dark_cloud", -1)

    if real_body < 0.10:
        _add("doji", 0)
    elif lower_wick >= body * 2 and upper_wick <= body:
        _add("hammer", +1)
    elif lower_wick <= body and upper_wick >= body * 2:
        _add("shooting_star", -1)
    elif lower_wick >= body * 2 and upper_wick >= body * 2:
        _add("spinning_top", 0)
    elif body > 0 and range_fraction >= 0.6:
        _add("marubozu", +1 if current_bullish else -1)

    return patterns


def support_resistance_clusters(
    df: pd.DataFrame,
    window: int = 5,
    tolerance_frac: float = 0.01,
    max_levels: int = 6,
) -> List[Dict[str, Any]]:
    """Cluster swing highs/lows into support and resistance levels.

    Swing points are found with a centred rolling window (Vibe-Trading's
    ``window=5`` geometry default). Points within ``tolerance_frac`` of each
    other merge into a cluster whose strength is the number of touches; the
    strongest clusters become the reported levels.
    """
    if len(df) < window * 2 + 1:
        return []
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    span = float(df["high"].tail(200).max() - df["low"].tail(200).min())
    if not np.isfinite(span) or span <= 0:
        return []
    tolerance = max(span * tolerance_frac, float(df["close"].iloc[-1]) * 0.0001)

    def _extrema(values: pd.Series, is_high: bool) -> List[float]:
        flat = values.reset_index(drop=True)
        rolling = flat.rolling(window * 2 + 1, center=True)
        extreme = rolling.max() if is_high else rolling.min()
        matches = flat.eq(extreme) & extreme.notna()
        return [float(flat.iloc[i]) for i in flat.index[matches]]

    points = _extrema(highs, True) + _extrema(lows, False)
    clusters: List[List[float]] = []
    for point in points:
        for cluster in clusters:
            if abs(point - cluster[0]) <= tolerance:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    ranked = sorted(
        ({"level": float(np.mean(c)), "strength": len(c)} for c in clusters),
        key=lambda item: item["strength"],
        reverse=True,
    )
    return ranked[:max_levels]


def _last_float(series: pd.Series) -> Optional[float]:
    """Last non-NaN value as a plain float, or None if the series is empty/NaN."""
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


@dataclass
class TrendRead:
    """Directional read from moving-average structure."""

    direction: str  # "up" | "down" | "sideways"
    fast_ma: Optional[float]
    slow_ma: Optional[float]
    separation_pips: Optional[float]
    note: str
    adx: Optional[float] = None
    regime: str = "unknown"  # "high_vol" | "normal" | "unknown"


@dataclass
class MomentumRead:
    rsi: Optional[float]
    macd_histogram: Optional[float]
    state: str  # "overbought" | "oversold" | "neutral"
    note: str
    bollinger_position: Optional[float] = None  # 0.0 at lower band, 1.0 at upper
    volume_ratio: Optional[float] = None
    volume_confirm: Optional[bool] = None  # volume agrees with close direction
    candle: Optional[Dict[str, Any]] = None  # {"pattern", "direction"}


@dataclass
class VolatilityRead:
    atr_pips: Optional[float]
    atr_percentile: Optional[float]
    regime: str  # "expanding" | "contracting" | "normal" | "unknown"
    note: str
    hysteresis: str = "off"  # Schmitt state of ATR percentile: "on" | "off"


@dataclass
class LevelsRead:
    recent_high: Optional[float]
    recent_low: Optional[float]
    range_pips: Optional[float]
    position_in_range: Optional[float]  # 0.0 at the low, 1.0 at the high
    note: str
    clusters: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TimeframeAnalysis:
    """Complete read of one timeframe for one instrument."""

    timeframe: str
    candles: int
    last_close: Optional[float]
    change_pips: Optional[float]
    trend: TrendRead
    momentum: MomentumRead
    volatility: VolatilityRead
    levels: LevelsRead
    warnings: List[str] = field(default_factory=list)
    composite: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Flatten for JSON output and LLM prompting."""
        candle = None
        if self.momentum.candle is not None:
            candle = {
                "pattern": self.momentum.candle["pattern"],
                "direction": self.momentum.candle["direction"],
            }
        return {
            "timeframe": self.timeframe,
            "candles": self.candles,
            "last_close": self.last_close,
            "change_pips": self.change_pips,
            "trend": {
                "direction": self.trend.direction,
                "fast_ma": self.trend.fast_ma,
                "slow_ma": self.trend.slow_ma,
                "separation_pips": self.trend.separation_pips,
                "adx": self.trend.adx,
                "regime": self.trend.regime,
                "note": self.trend.note,
            },
            "momentum": {
                "rsi": self.momentum.rsi,
                "macd_histogram": self.momentum.macd_histogram,
                "state": self.momentum.state,
                "bollinger_position": self.momentum.bollinger_position,
                "volume_ratio": self.momentum.volume_ratio,
                "volume_confirm": self.momentum.volume_confirm,
                "candle": candle,
                "note": self.momentum.note,
            },
            "volatility": {
                "atr_pips": self.volatility.atr_pips,
                "atr_percentile": self.volatility.atr_percentile,
                "regime": self.volatility.regime,
                "hysteresis": self.volatility.hysteresis,
                "note": self.volatility.note,
            },
            "levels": {
                "recent_high": self.levels.recent_high,
                "recent_low": self.levels.recent_low,
                "range_pips": self.levels.range_pips,
                "position_in_range": self.levels.position_in_range,
                "clusters": self.levels.clusters,
                "note": self.levels.note,
            },
            "composite": self.composite,
            "warnings": list(self.warnings),
        }


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(value, digits)


def _analyse_trend(
    df: pd.DataFrame, instrument: Instrument, fast: int, slow: int
) -> TrendRead:
    if len(df) < slow:
        return TrendRead(
            "sideways", None, None, None,
            f"insufficient history for {slow}-period MA",
        )

    fast_ma = _last_float(ema(df["close"], fast))
    slow_ma = _last_float(ema(df["close"], slow))
    if fast_ma is None or slow_ma is None:
        return TrendRead("sideways", fast_ma, slow_ma, None, "moving averages unavailable")

    separation = instrument.pips(fast_ma - slow_ma)
    atr_series = atr(df)
    atr_value = _last_float(atr_series)

    # "Sideways" means the MAs sit closer together than one average candle range,
    # so their ordering is indistinguishable from noise. Empirically the ratio is
    # well under 1 for directionless series and several multiples above it for
    # trending ones, which makes one ATR a clean separator.
    noise_floor_pips = instrument.pips(atr_value) if atr_value else 0.0

    if abs(separation) <= noise_floor_pips:
        direction = "sideways"
        note = "moving averages compressed within noise; no clear trend"
    elif separation > 0:
        direction = "up"
        note = f"fast EMA above slow EMA by {abs(separation):.1f} pips"
    else:
        direction = "down"
        note = f"fast EMA below slow EMA by {abs(separation):.1f} pips"

    adx_value = _last_float(adx(df))
    regime = volatility_regime(df["close"])
    if adx_value is not None:
        note += f"; ADX {adx_value:.1f}"
        if adx_value >= 25:
            note += " (trending)"
        else:
            note += " (ranging)"

    return TrendRead(
        direction,
        _round(fast_ma, 6),
        _round(slow_ma, 6),
        _round(separation, 1),
        note,
        adx=_round(adx_value, 1),
        regime=regime,
    )


def composite_vote(df: pd.DataFrame) -> Dict[str, Any]:
    """Three-dimensional voting read adapted from Vibe-Trading.

    Dimensions:
    - trend: EMA 12/26 ordering, kept only when ADX(14) >= 25
    - mean_reversion: Bollinger(20,2) position and RSI(14) extremes
    - volume: OBV slope over 20 bars, kept only when volume data is usable

    The vote is ``"long"``, ``"short"`` or ``"neutral"``; volume confirmation
    is ``None`` when trend or volume direction is unavailable. Constant or flat
    volume therefore cannot fabricate a confirmation.
    """
    if len(df) < 2:
        return {"trend": "neutral", "mean_reversion": "neutral", "volume": "neutral", "vote": "neutral"}
    close = df["close"]
    fast = ema(close, 12)
    slow = ema(close, 26)
    last_fast = _last_float(fast)
    last_slow = _last_float(slow)
    strength = _last_float(adx(df, 14))

    trend = "neutral"
    if last_fast is not None and last_slow is not None and strength is not None:
        if strength >= 25 and abs(last_fast - last_slow) > 0:
            trend = "long" if last_fast > last_slow else "short"

    bands = bollinger(close, 20, 2.0).dropna()
    rsi_value = _last_float(rsi(close, 14))
    mean_reversion = "neutral"
    if not bands.empty and rsi_value is not None:
        row = bands.iloc[-1]
        width = row["upper"] - row["lower"]
        if width > 0 and row["middle"] == row["middle"]:
            position = float((close.iloc[-1] - row["lower"]) / width)
            if rsi_value >= 65 or position >= 0.8:
                mean_reversion = "short"
            elif rsi_value <= 35 or position <= 0.2:
                mean_reversion = "long"

    volume = "neutral"
    volume_confirm = None
    volume_ratio_value = None
    if "volume" in df.columns and df["volume"].astype(float).nunique() > 1:
        volume_ratio_value = volume_ratio(df, 20)
        obv_series = obv(df)
        if len(obv_series.dropna()) >= 25:
            slope = float(obv_series.iloc[-1]) - float(obv_series.iloc[-25])
            if slope > 0:
                volume = "long"
            elif slope < 0:
                volume = "short"
            if volume != "neutral":
                volume_confirm = volume == trend if trend != "neutral" else None

    sides = [d for d in (trend, mean_reversion, volume) if d != "neutral"]
    vote = "neutral"
    if sides:
        vote = max(set(sides), key=sides.count)
        if sides.count(vote) == 1:
            vote = "neutral"
    return {
        "trend": trend,
        "mean_reversion": mean_reversion,
        "volume": volume,
        "vote": vote,
        "volume_confirm": volume_confirm,
        "volume_ratio": round(volume_ratio_value, 2) if volume_ratio_value is not None else None,
    }


def _analyse_momentum(df: pd.DataFrame, rsi_period: int) -> MomentumRead:
    rsi_value = _last_float(rsi(df["close"], rsi_period)) if len(df) > rsi_period else None
    hist = _last_float(macd(df["close"])["histogram"]) if len(df) >= 35 else None

    if rsi_value is None:
        state, note = "neutral", "insufficient history for RSI"
    elif rsi_value >= 70:
        state, note = "overbought", f"RSI {rsi_value:.1f} in overbought territory"
    elif rsi_value <= 30:
        state, note = "oversold", f"RSI {rsi_value:.1f} in oversold territory"
    else:
        state, note = "neutral", f"RSI {rsi_value:.1f} mid-range"

    if hist is not None:
        note += f"; MACD histogram {'positive' if hist > 0 else 'negative'}"

    bands = bollinger(df["close"], 20, 2.0).dropna()
    bollinger_position = None
    if not bands.empty:
        row = bands.iloc[-1]
        width = row["upper"] - row["lower"]
        if width > 0:
            bollinger_position = _round(
                float((df["close"].iloc[-1] - row["lower"]) / width), 3
            )
            if bollinger_position is not None:
                note += (
                    f"; price at {bollinger_position:.0%} of the Bollinger band"
                )

    volume_ratio_value = volume_ratio(df, 20)
    volume_confirm = None
    if volume_ratio_value is not None and "volume" in df.columns:
        volume_series = df["volume"].astype(float)
        if volume_series.nunique() > 1 and len(df) >= 25:
            obv_series = obv(df)
            slope = float(obv_series.iloc[-1]) - float(obv_series.iloc[-25])
            price_change = float(df["close"].iloc[-1]) - float(df["close"].iloc[-25])
            if slope != 0 and price_change != 0:
                volume_confirm = (slope > 0) == (price_change > 0)
                note += "; volume confirms price direction" if volume_confirm else "; volume diverges from price"

    candle = None
    detected = candle_patterns(df)
    if detected:
        candle = detected[0]
        pattern_label = candle["pattern"].replace("_", " ")
        if candle["direction"] > 0:
            note += f"; {pattern_label} formed on the last candle"
        elif candle["direction"] < 0:
            note += f"; {pattern_label} formed on the last candle"

    return MomentumRead(
        _round(rsi_value, 1),
        _round(hist, 6),
        state,
        note,
        bollinger_position=bollinger_position,
        volume_ratio=_round(volume_ratio_value, 2),
        volume_confirm=volume_confirm,
        candle=candle,
    )


def _analyse_volatility(
    df: pd.DataFrame, instrument: Instrument, period: int
) -> VolatilityRead:
    if len(df) < period + 1:
        return VolatilityRead(None, None, "unknown", "insufficient history for ATR")

    atr_series = atr(df, period).dropna()
    if atr_series.empty:
        return VolatilityRead(None, None, "unknown", "ATR unavailable")

    current = float(atr_series.iloc[-1])
    atr_pips = instrument.pips(current)

    percentile: Optional[float] = None
    regime = "normal"
    if len(atr_series) >= 20:
        percentile = float((atr_series <= current).mean() * 100.0)
        if percentile >= 75:
            regime = "expanding"
        elif percentile <= 25:
            regime = "contracting"

    note = f"ATR {atr_pips:.1f} pips"
    if percentile is not None:
        # Avoid English ordinal suffixes ("21th"); a plain rank reads correctly.
        note += f" (percentile {percentile:.0f} of recent history)"
    if regime == "contracting":
        note += "; compressed ranges often precede breakouts"
    elif regime == "expanding":
        note += "; wide ranges imply wider stops are needed"

    hysteresis_state = "off"
    if percentile is not None and len(atr_series) >= 60:
        # Rolling 20-bar percentile of ATR, then a Schmitt trigger on it.
        percentile_series = atr_series.rolling(20, min_periods=20).apply(
            lambda window: float((window <= window.iloc[-1]).mean()), raw=False
        )
        hysteresis_state = hysteresis(percentile_series, enter=0.75, exit=0.25)

    return VolatilityRead(
        _round(atr_pips, 1),
        _round(percentile, 0),
        regime,
        note,
        hysteresis=hysteresis_state,
    )


def _analyse_levels(
    df: pd.DataFrame, instrument: Instrument, period: int
) -> LevelsRead:
    window = df.tail(period)
    if window.empty:
        return LevelsRead(None, None, None, None, "no candles available")

    high = float(window["high"].max())
    low = float(window["low"].min())
    close = float(window["close"].iloc[-1])
    span = high - low

    if span <= 0:
        return LevelsRead(
            _round(high, 6), _round(low, 6), 0.0, None,
            "flat range; levels carry no information",
        )

    position = (close - low) / span
    range_pips = instrument.pips(span)

    if position >= 0.8:
        note = f"price in upper fifth of the {period}-period range ({range_pips:.0f} pips wide)"
    elif position <= 0.2:
        note = f"price in lower fifth of the {period}-period range ({range_pips:.0f} pips wide)"
    else:
        note = f"price mid-range ({range_pips:.0f} pips wide)"

    clusters: List[Dict[str, Any]] = []
    if len(df) >= 30:
        clusters = support_resistance_clusters(df)

    return LevelsRead(
        _round(high, 6), _round(low, 6), _round(range_pips, 1), _round(position, 3), note,
        clusters=clusters,
    )


def analyse_timeframe(
    df: pd.DataFrame,
    instrument: Instrument,
    timeframe: str,
    fast_ma: int = 20,
    slow_ma: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
    level_period: int = 20,
) -> TimeframeAnalysis:
    """Run the full indicator set over one timeframe.

    Degrades gracefully: a short frame yields ``None`` fields and a warning
    rather than raising, so a provider returning limited history still produces a
    usable (if thinner) report.
    """
    _validate(df, minimum=2)

    df = df.sort_index()
    warnings: List[str] = []

    if len(df) < slow_ma:
        warnings.append(
            f"only {len(df)} candles on {timeframe}; "
            f"trend read needs {slow_ma} and was degraded"
        )

    last_close = float(df["close"].iloc[-1])
    change_pips = instrument.pips(last_close - float(df["close"].iloc[-2]))

    composite: Dict[str, Any] = {}
    if len(df) >= 30:
        composite = composite_vote(df)

    return TimeframeAnalysis(
        timeframe=timeframe,
        candles=len(df),
        last_close=round(last_close, 6),
        change_pips=_round(change_pips, 1),
        trend=_analyse_trend(df, instrument, fast_ma, slow_ma),
        momentum=_analyse_momentum(df, rsi_period),
        volatility=_analyse_volatility(df, instrument, atr_period),
        levels=_analyse_levels(df, instrument, level_period),
        warnings=warnings,
        composite=composite,
    )


def align_timeframes(reads: Dict[str, TimeframeAnalysis]) -> Dict[str, object]:
    """Summarise agreement across timeframes.

    Multi-timeframe confluence is the single most useful structural signal in
    discretionary FX analysis, so it is computed explicitly instead of being
    left for the model to infer.
    """
    directions = {tf: read.trend.direction for tf, read in reads.items()}
    values = [d for d in directions.values() if d != "sideways"]

    if not values:
        verdict, confidence = "sideways", "low"
    elif all(d == "up" for d in values):
        verdict = "up"
        confidence = "high" if len(values) == len(directions) else "medium"
    elif all(d == "down" for d in values):
        verdict = "down"
        confidence = "high" if len(values) == len(directions) else "medium"
    else:
        verdict, confidence = "conflicted", "low"

    return {
        "per_timeframe": directions,
        "verdict": verdict,
        "confidence": confidence,
        "note": {
            "up": "timeframes agree on an uptrend",
            "down": "timeframes agree on a downtrend",
            "sideways": "no timeframe shows a clear trend",
            "conflicted": "timeframes disagree; treat directional bias as weak",
        }[verdict],
    }
