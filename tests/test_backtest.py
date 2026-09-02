from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forex.backtest import (
    BacktestSummary,
    BacktestTrade,
    _resolve_trade,
    _strategy_allows,
    _strategy_time_allows,
    format_backtest,
    load_latest_summary,
    monte_carlo_pvalue,
    save_backtest,
)
from forex.smc import SMCReading


def _future(highs, lows):
    return pd.DataFrame(
        {"open": [100.0] * len(highs), "high": highs, "low": lows, "close": [100.0] * len(highs)},
        index=pd.date_range("2026-01-01", periods=len(highs), freq="5min", tz="UTC"),
    )


def test_resolve_trade_uses_first_subsequent_touch():
    frame = _future([101.0, 102.1], [99.5, 99.5])
    result, result_r, closed, ambiguous = _resolve_trade(
        frame, "LONG", 99.0, 102.0, frame.index[-1]
    )
    assert (result, result_r, ambiguous) == ("win", 2.0, False)
    assert closed == frame.index[-1]


def test_resolve_trade_counts_same_candle_as_loss():
    frame = _future([102.1], [98.9])
    result, result_r, _, ambiguous = _resolve_trade(
        frame, "LONG", 99.0, 102.0, frame.index[-1]
    )
    assert (result, result_r, ambiguous) == ("loss", -1.0, True)


def test_resolve_trade_uses_configured_reward_r():
    frame = _future([101.3], [99.5])
    result, result_r, _, _ = _resolve_trade(
        frame, "LONG", 99.0, 101.25, frame.index[-1], reward_r=1.25
    )
    assert (result, result_r) == ("win", 1.25)


def test_backtest_summary_round_trip_and_format(tmp_path):
    summary = BacktestSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        strategy_version="v1",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-01-31T00:00:00+00:00",
        m5_candles=5000,
        evaluations=4500,
        signals=3,
        wins=2,
        losses=1,
        expired=0,
        win_rate=66.666,
        total_r=3.0,
        profit_factor=4.0,
        max_drawdown_r=1.0,
        max_candidate_score=80,
    )
    path, _ = save_backtest(Path(tmp_path), summary, [])
    loaded = load_latest_summary(path)
    assert loaded == summary
    text = format_backtest(loaded)
    assert "66.7%" in text
    assert "+3.0R" in text


def _reading(action="LONG", m5="bullish", rsi=50.0, score=65):
    return SMCReading(
        action=action,
        confluence_score=score,
        price=100.0,
        macro_bias="up" if action == "LONG" else "down",
        m15_structure="bullish" if action == "LONG" else "bearish",
        m5_structure=m5,
        entry=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        risk_reward=2.0,
        rsi=rsi,
        atr=1.0,
        liquidity_event=None,
        fvg=None,
        candle_pattern=None,
        reasons=[],
        cautions=[],
    )


def test_v2_requires_london_m5_alignment_and_neutral_rsi():
    london = pd.Timestamp("2026-01-01T08:00:00Z")
    assert _strategy_allows(_reading(), london, "v2") is True
    assert _strategy_allows(_reading(m5="bearish"), london, "v2") is False
    assert _strategy_allows(_reading(rsi=70), london, "v2") is False
    assert _strategy_allows(_reading(), pd.Timestamp("2026-01-01T15:00:00Z"), "v2") is False


def test_v3_requires_clean_structure_score_and_core_london_hours():
    assert _strategy_allows(_reading(), pd.Timestamp("2026-01-01T07:00:00Z"), "v3") is True
    assert _strategy_allows(_reading(), pd.Timestamp("2026-01-01T10:55:00Z"), "v3") is True
    assert _strategy_allows(_reading(), pd.Timestamp("2026-01-01T06:55:00Z"), "v3") is False
    assert _strategy_allows(_reading(), pd.Timestamp("2026-01-01T11:00:00Z"), "v3") is False
    assert _strategy_allows(_reading(score=75), pd.Timestamp("2026-01-01T08:00:00Z"), "v3") is False


def test_strategy_time_prefilter_matches_version_windows():
    assert _strategy_time_allows(pd.Timestamp("2026-01-01T05:00:00Z"), "v1") is True
    assert _strategy_time_allows(pd.Timestamp("2026-01-01T06:00:00Z"), "v2") is True
    assert _strategy_time_allows(pd.Timestamp("2026-01-01T11:59:00Z"), "v2") is True
    assert _strategy_time_allows(pd.Timestamp("2026-01-01T06:59:00Z"), "v3") is False
    assert _strategy_time_allows(pd.Timestamp("2026-01-01T10:59:00Z"), "v3") is True


def test_v31_uses_directional_hours_and_pullback_rsi_band():
    at_7 = pd.Timestamp("2026-01-01T07:00:00Z")
    at_9 = pd.Timestamp("2026-01-01T09:00:00Z")
    assert _strategy_allows(_reading(action="LONG", rsi=40), at_7, "v3.1") is True
    assert _strategy_allows(_reading(action="LONG", rsi=40), at_9, "v3.1") is False
    assert _strategy_allows(_reading(action="SHORT", m5="bearish", rsi=40), at_9, "v3.1") is True
    assert _strategy_allows(_reading(action="SHORT", m5="bearish", rsi=45), at_9, "v3.1") is False
    assert _strategy_allows(_reading(action="SHORT", m5="bearish", rsi=40, score=75), at_9, "v3.1") is False
    assert _strategy_time_allows(pd.Timestamp("2026-01-01T08:00:00Z"), "v3.1") is False


def test_v4_uses_only_07utc_clean_structure_setups():
    at_7 = pd.Timestamp("2026-01-01T07:00:00Z")
    assert _strategy_allows(_reading(rsi=35), at_7, "v4") is True
    assert _strategy_allows(_reading(rsi=65), at_7, "v4") is True
    assert _strategy_allows(_reading(score=75), at_7, "v4") is False
    assert _strategy_allows(_reading(), pd.Timestamp("2026-01-01T08:00:00Z"), "v4") is False
    assert _strategy_time_allows(at_7, "v4") is True


def test_backtest_summary_format_shows_new_fields(tmp_path):
    summary = BacktestSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        strategy_version="v1",
        period_start="2026-01-01T00:00:00+00:00",
        period_end="2026-01-31T00:00:00+00:00",
        m5_candles=5000,
        evaluations=4500,
        signals=10,
        wins=6,
        losses=4,
        expired=0,
        win_rate=60.0,
        total_r=8.0,
        profit_factor=2.0,
        max_drawdown_r=1.5,
        max_candidate_score=80,
        warmup_bars=500,
        monte_carlo_p_value=0.02,
        regime_breakdown={"normal": 8, "high_vol": 2},
    )
    text = format_backtest(summary)
    assert "P-value (Monte Carlo): 0.020" in text
    assert "Regime: high_vol: 2, normal: 8" in text


def test_monte_carlo_pvalue_deterministic_and_bounded():
    trades = [
        BacktestTrade(
            opened_at="2026-01-01T00:00:00+00:00",
            closed_at="2026-01-01T01:00:00+00:00",
            direction="LONG",
            entry=100.0,
            stop_loss=99.0,
            take_profit=102.0,
            score=70,
            macro_bias="up",
            m15_structure="bullish",
            m5_structure="bullish",
            rsi=50.0,
            liquidity_event=None,
            fvg=None,
            candle_pattern=None,
            result="win" if i % 2 == 0 else "loss",
            result_r=2.0 if i % 2 == 0 else -1.0,
            ambiguous=False,
        )
        for i in range(12)
    ]
    first = monte_carlo_pvalue(trades, iterations=100)
    second = monte_carlo_pvalue(trades, iterations=100)
    assert first == second  # seeded -> deterministic
    assert 0.0 <= first <= 1.0


def test_monte_carlo_pvalue_none_for_few_trades():
    assert monte_carlo_pvalue([]) is None
