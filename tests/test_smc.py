import numpy as np
import pandas as pd

from forex.providers import resample
from forex.smc import evaluate, market_structure


def _trend_frame(direction=1, rows=240, frequency="5min"):
    index = pd.date_range("2026-01-01", periods=rows, freq=frequency, tz="UTC")
    sequence = np.arange(rows, dtype=float)
    close = 2000 + direction * sequence * 0.25 + np.sin(sequence / 3) * 1.5
    open_ = close - direction * 0.08
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.35,
            "low": np.minimum(open_, close) - 0.35,
            "close": close,
        },
        index=index,
    )


def test_bullish_macro_and_structures_can_confirm_long():
    m5 = _trend_frame(direction=1)
    m15 = resample(m5, "15min")
    assert market_structure(m5).direction == "bullish"
    assert market_structure(m15).direction == "bullish"
    reading = evaluate(m5, m15, "up")
    assert reading.action == "LONG"
    assert reading.confluence_score >= 65
    assert reading.stop_loss < reading.entry < reading.take_profit


def test_conflicted_macro_blocks_directional_signal():
    m5 = _trend_frame(direction=1)
    reading = evaluate(m5, resample(m5, "15min"), "conflicted")
    assert reading.action == "WAIT"
    assert any("blocked" in caution for caution in reading.cautions)


def test_bearish_macro_and_structures_can_confirm_short():
    m5 = _trend_frame(direction=-1)
    reading = evaluate(m5, resample(m5, "15min"), "down")
    assert reading.action == "SHORT"
    assert reading.stop_loss > reading.entry > reading.take_profit
