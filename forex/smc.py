"""Pure, analysis-only Smart Money Concepts confirmation engine.

The score is a deterministic confluence score, not a probability of profit.
No broker, account, position sizing, or order execution code belongs here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .analysis import (
    candle_patterns,
    composite_vote,
    hysteresis,
    support_resistance_clusters,
    volatility_regime,
)
from .analysis import rsi as rsi_series


@dataclass(frozen=True)
class Structure:
    direction: str
    break_type: Optional[str]
    last_swing_high: Optional[float]
    last_swing_low: Optional[float]


@dataclass(frozen=True)
class SMCReading:
    action: str
    confluence_score: int
    price: float
    macro_bias: str
    m15_structure: str
    m5_structure: str
    entry: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward: Optional[float]
    rsi: float
    atr: float
    liquidity_event: Optional[str]
    fvg: Optional[str]
    candle_pattern: Optional[str]
    reasons: List[str]
    cautions: List[str]
    regime: str = "normal"  # "high_vol" | "normal" | "unknown"
    bias_source: str = "alignment"  # "alignment" | "voting" | "unavailable"
    hysteresis: str = "off"  # Schmitt state over recent M5 scores
    volume_confirm: Optional[bool] = None
    composite: Dict[str, Any] = field(default_factory=dict)
    sr_levels: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _swings(frame: pd.DataFrame, lookback: int = 3) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    window = lookback * 2 + 1
    high_values = frame["high"].astype(float).reset_index(drop=True)
    low_values = frame["low"].astype(float).reset_index(drop=True)
    high_mask = high_values.eq(high_values.rolling(window, center=True).max()).fillna(False)
    low_mask = low_values.eq(low_values.rolling(window, center=True).min()).fillna(False)
    highs = [(int(index), float(high_values.iloc[index])) for index in high_values.index[high_mask]]
    lows = [(int(index), float(low_values.iloc[index])) for index in low_values.index[low_mask]]
    return highs, lows


def market_structure(frame: pd.DataFrame) -> Structure:
    if len(frame) < 35:
        return Structure("neutral", None, None, None)
    recent = frame.tail(100).reset_index(drop=True)
    highs, lows = _swings(recent)
    if len(highs) < 2 or len(lows) < 2:
        return Structure("neutral", None, None, None)

    previous_high, last_high = highs[-2][1], highs[-1][1]
    previous_low, last_low = lows[-2][1], lows[-1][1]
    close = float(recent.iloc[-1]["close"])
    ema_fast = float(recent["close"].ewm(span=10, adjust=False).mean().iloc[-1])
    ema_slow = float(recent["close"].ewm(span=30, adjust=False).mean().iloc[-1])

    bullish_sequence = last_high > previous_high and last_low > previous_low
    bearish_sequence = last_high < previous_high and last_low < previous_low
    bullish_break = close > last_high
    bearish_break = close < last_low

    if bullish_break and ema_fast > ema_slow:
        return Structure("bullish", "BOS", last_high, last_low)
    if bearish_break and ema_fast < ema_slow:
        return Structure("bearish", "BOS", last_high, last_low)
    if bullish_sequence and ema_fast > ema_slow:
        return Structure("bullish", None, last_high, last_low)
    if bearish_sequence and ema_fast < ema_slow:
        return Structure("bearish", None, last_high, last_low)
    return Structure("neutral", None, last_high, last_low)


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = true_range.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(value)


def _rsi(frame: pd.DataFrame, period: int = 14) -> float:
    delta = frame["close"].diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = float(losses.iloc[-1])
    if loss == 0:
        return 100.0
    value = 100 - 100 / (1 + float(gains.iloc[-1]) / loss)
    return float(value)


def liquidity_sweep(frame: pd.DataFrame, atr: float) -> Optional[str]:
    if len(frame) < 12:
        return None
    completed = frame.tail(12)
    signal = completed.iloc[-1]
    reference = completed.iloc[:-1]
    tolerance = max(atr * 0.03, float(signal["close"]) * 0.00005)
    prior_high = float(reference["high"].max())
    prior_low = float(reference["low"].min())
    if float(signal["high"]) > prior_high + tolerance and float(signal["close"]) < prior_high:
        return "bearish_sweep"
    if float(signal["low"]) < prior_low - tolerance and float(signal["close"]) > prior_low:
        return "bullish_sweep"
    return None


def recent_fvg(frame: pd.DataFrame, atr: float) -> Optional[str]:
    if len(frame) < 3:
        return None
    threshold = max(atr * 0.08, float(frame.iloc[-1]["close"]) * 0.00005)
    current = float(frame.iloc[-1]["close"])
    for index in range(len(frame) - 3, max(len(frame) - 12, -1), -1):
        first = frame.iloc[index]
        third = frame.iloc[index + 2]
        bullish_gap = float(third["low"]) - float(first["high"])
        bearish_gap = float(first["low"]) - float(third["high"])
        if bullish_gap >= threshold and current > float(first["high"]):
            return "bullish_fvg"
        if bearish_gap >= threshold and current < float(first["low"]):
            return "bearish_fvg"
    return None


def candle_pattern(frame: pd.DataFrame) -> Optional[str]:
    if len(frame) < 2:
        return None
    previous, current = frame.iloc[-2], frame.iloc[-1]
    previous_bullish = float(previous["close"]) > float(previous["open"])
    current_bullish = float(current["close"]) > float(current["open"])
    if not previous_bullish and current_bullish and float(current["open"]) <= float(previous["close"]) and float(current["close"]) >= float(previous["open"]):
        return "bullish_engulfing"
    if previous_bullish and not current_bullish and float(current["open"]) >= float(previous["close"]) and float(current["close"]) <= float(previous["open"]):
        return "bearish_engulfing"
    body = abs(float(current["close"]) - float(current["open"]))
    if body == 0:
        return None
    upper_wick = float(current["high"]) - max(float(current["open"]), float(current["close"]))
    lower_wick = min(float(current["open"]), float(current["close"])) - float(current["low"])
    if lower_wick >= body * 2 and upper_wick <= body:
        return "hammer"
    if upper_wick >= body * 2 and lower_wick <= body:
        return "shooting_star"
    return None


def evaluate(m5: pd.DataFrame, m15: pd.DataFrame, macro_bias: str) -> SMCReading:
    if len(m5) < 35 or len(m15) < 35:
        raise ValueError("at least 35 closed candles are required on M5 and M15")

    m5 = m5.copy().dropna(subset=["open", "high", "low", "close"])
    m15 = m15.copy().dropna(subset=["open", "high", "low", "close"])
    price = float(m5.iloc[-1]["close"])
    atr = _atr(m5)
    rsi = _rsi(m5)
    m5_structure = market_structure(m5)
    m15_structure = market_structure(m15)
    sweep = liquidity_sweep(m5, atr)
    fvg = recent_fvg(m5, atr)
    pattern = candle_pattern(m5)

    # Vibe-Trading style overlay: volatility regime, composite vote and a
    # hysteresis dead-band over the M5 RSI series.
    regime = volatility_regime(m5["close"])
    composite = composite_vote(m5)
    m5_rsi_series = rsi_series(m5["close"], 14)
    if len(m5_rsi_series.dropna()) >= 2:
        hysteresis_state = hysteresis(m5_rsi_series, enter=70.0, exit=30.0)
    else:
        hysteresis_state = "off"

    sr_levels: List[Dict[str, Any]] = []
    if len(m5) >= 30:
        sr_levels = support_resistance_clusters(m5, window=5, tolerance_frac=0.002)

    long_score = 0
    short_score = 0
    long_reasons: List[str] = []
    short_reasons: List[str] = []
    cautions: List[str] = []

    bias_source = "unavailable"
    if macro_bias == "up":
        long_score += 25
        long_reasons.append("H1/H4/D1 macro bias bullish")
        bias_source = "alignment"
    elif macro_bias == "down":
        short_score += 25
        short_reasons.append("H1/H4/D1 macro bias bearish")
        bias_source = "alignment"
    else:
        cautions.append(f"macro bias {macro_bias or 'unavailable'}; directional signal blocked")

    composite_direction = composite.get("vote", "neutral")
    if composite_direction in ("long", "short"):
        if composite_direction == "long":
            long_score += 15
            long_reasons.append("M5 composite vote long (trend/momentum/volume)")
        else:
            short_score += 15
            short_reasons.append("M5 composite vote short (trend/momentum/volume)")
        bias_source = "voting"

    if m15_structure.direction == "bullish":
        long_score += 25
        long_reasons.append(f"M15 bullish structure{f' {m15_structure.break_type}' if m15_structure.break_type else ''}")
    elif m15_structure.direction == "bearish":
        short_score += 25
        short_reasons.append(f"M15 bearish structure{f' {m15_structure.break_type}' if m15_structure.break_type else ''}")

    if m5_structure.direction == "bullish":
        long_score += 15
        long_reasons.append("M5 structure bullish")
    elif m5_structure.direction == "bearish":
        short_score += 15
        short_reasons.append("M5 structure bearish")

    if sweep == "bullish_sweep":
        long_score += 15
        long_reasons.append("M5 downside liquidity sweep")
    elif sweep == "bearish_sweep":
        short_score += 15
        short_reasons.append("M5 upside liquidity sweep")

    if fvg == "bullish_fvg":
        long_score += 10
        long_reasons.append("recent bullish fair-value gap")
    elif fvg == "bearish_fvg":
        short_score += 10
        short_reasons.append("recent bearish fair-value gap")

    if pattern in ("bullish_engulfing", "hammer"):
        long_score += 10
        long_reasons.append(f"M5 {pattern.replace('_', ' ')}")
    elif pattern in ("bearish_engulfing", "shooting_star"):
        short_score += 10
        short_reasons.append(f"M5 {pattern.replace('_', ' ')}")

    if rsi <= 35:
        long_score += 5
        long_reasons.append(f"M5 RSI oversold ({rsi:.1f})")
    elif rsi >= 65:
        short_score += 5
        short_reasons.append(f"M5 RSI overbought ({rsi:.1f})")

    direction = "long" if long_score >= short_score else "short"
    volume_confirm = composite.get("volume_confirm")
    if volume_confirm is True and direction == "long":
        long_score += 5
        long_reasons.append("M5 volume confirms upside")
    elif volume_confirm is False and direction == "short":
        short_score += 5
        short_reasons.append("M5 volume confirms downside")

    score = max(long_score, short_score)
    if regime == "high_vol":
        cautions.append("high-volatility regime; new directional signals are blocked")

    action = "WAIT"
    reasons: List[str] = []
    macro_allows = (direction == "long" and macro_bias == "up") or (direction == "short" and macro_bias == "down")
    structure_allows = m15_structure.direction == ("bullish" if direction == "long" else "bearish")
    if (
        score >= 65
        and macro_allows
        and structure_allows
        and long_score != short_score
        and regime != "high_vol"
    ):
        action = direction.upper()
        reasons = long_reasons if direction == "long" else short_reasons
    else:
        reasons = long_reasons if long_score >= short_score else short_reasons
        if score < 65:
            cautions.append(f"confluence {score}/100 below 65 threshold")
        if not structure_allows:
            cautions.append("M15 structure does not confirm the candidate direction")

    entry = stop = target = None
    risk_reward = None
    if action == "LONG":
        entry = price
        structural = m5_structure.last_swing_low or float(m5.tail(20)["low"].min())
        stop = min(structural - atr * 0.2, price - atr)
        target = price + (price - stop) * 2
        risk_reward = 2.0
    elif action == "SHORT":
        entry = price
        structural = m5_structure.last_swing_high or float(m5.tail(20)["high"].max())
        stop = max(structural + atr * 0.2, price + atr)
        target = price - (stop - price) * 2
        risk_reward = 2.0

    return SMCReading(
        action=action,
        confluence_score=min(score, 100),
        price=price,
        macro_bias=macro_bias,
        m15_structure=m15_structure.direction,
        m5_structure=m5_structure.direction,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        risk_reward=risk_reward,
        rsi=rsi,
        atr=atr,
        liquidity_event=sweep,
        fvg=fvg,
        candle_pattern=pattern,
        reasons=reasons,
        cautions=cautions,
        regime=regime,
        bias_source=bias_source,
        hysteresis=hysteresis_state,
        volume_confirm=volume_confirm,
        composite=composite,
        sr_levels=sr_levels,
    )
