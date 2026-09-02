from datetime import datetime, timedelta, timezone

import pandas as pd

from forex.smc import SMCReading
from forex.tracking import SignalTracker, append_stats_footer, format_stats_footer


def _reading(direction="LONG"):
    return SMCReading(
        action=direction,
        confluence_score=75,
        price=100.0,
        macro_bias="up" if direction == "LONG" else "down",
        m15_structure="bullish" if direction == "LONG" else "bearish",
        m5_structure="bullish" if direction == "LONG" else "bearish",
        entry=100.0,
        stop_loss=99.0 if direction == "LONG" else 101.0,
        take_profit=102.0 if direction == "LONG" else 98.0,
        risk_reward=2.0,
        rsi=50.0,
        atr=1.0,
        liquidity_event=None,
        fvg=None,
        candle_pattern=None,
        reasons=["test"],
        cautions=[],
    )


def _frame(timestamp, high, low, close=100.0):
    return pd.DataFrame(
        [{"open": 100.0, "high": high, "low": low, "close": close}],
        index=pd.DatetimeIndex([timestamp]),
    )


def test_long_target_is_a_win(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert tracker.create_signal(_reading(), candle, candle) is not None
    assert tracker.evaluate(_frame(candle + timedelta(minutes=5), 102.1, 99.5), now=candle + timedelta(minutes=5)) == 1
    stats = tracker.stats()
    assert (stats.completed, stats.wins, stats.losses, stats.win_rate, stats.total_r) == (1, 1, 0, 100.0, 2.0)


def test_short_stop_is_a_loss(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading("SHORT"), candle, candle)
    tracker.evaluate(_frame(candle + timedelta(minutes=5), 101.1, 99.0), now=candle + timedelta(minutes=5))
    stats = tracker.stats()
    assert (stats.completed, stats.wins, stats.losses, stats.total_r) == (1, 0, 1, -1.0)


def test_same_candle_target_and_stop_is_conservative_loss(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading(), candle, candle)
    tracker.evaluate(_frame(candle + timedelta(minutes=5), 102.1, 98.9), now=candle + timedelta(minutes=5))
    assert tracker.stats().losses == 1
    with tracker._connect() as connection:
        assert connection.execute("SELECT ambiguous FROM paper_signals").fetchone()[0] == 1


def test_expired_is_excluded_from_win_rate(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading(), candle, candle)
    tracker.evaluate(
        _frame(candle + timedelta(minutes=5), 100.5, 99.5),
        timeout_minutes=240,
        now=candle + timedelta(minutes=241),
    )
    stats = tracker.stats()
    assert stats.expired == 1
    assert stats.completed == 0
    assert stats.win_rate is None
    assert "Belum tersedia" in format_stats_footer(stats)


def test_target_after_timeout_does_not_turn_expired_signal_into_win(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading(), candle, candle)
    tracker.evaluate(
        _frame(candle + timedelta(minutes=245), 102.1, 99.5),
        timeout_minutes=240,
        now=candle + timedelta(minutes=245),
    )
    stats = tracker.stats()
    assert stats.expired == 1
    assert stats.wins == 0


def test_active_signal_blocks_another(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tracker.create_signal(_reading(), candle, candle)
    assert tracker.can_create(now=candle + timedelta(hours=1)) is False


def test_strategy_versions_have_isolated_limits_and_statistics(tmp_path):
    path = tmp_path / "signals.db"
    old_tracker = SignalTracker(path, strategy_version="v1")
    current_tracker = SignalTracker(path, strategy_version="v5")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    old_tracker.create_signal(_reading(), candle, candle)

    assert old_tracker.stats().total == 1
    assert current_tracker.stats().total == 0
    assert current_tracker.can_create(now=candle + timedelta(minutes=5)) is True


def test_footer_is_appended_when_database_is_configured(tmp_path, monkeypatch):
    path = tmp_path / "signals.db"
    SignalTracker(path)
    monkeypatch.setenv("SIGNAL_TRACKING_DB", str(path))
    result = append_stats_footer("Laporan")
    assert "Laporan" in result
    assert "✅ Benar: 0" in result
    assert "❌ Salah: 0" in result
    assert append_stats_footer(result) == result


def test_manual_decision_is_saved_once_for_active_signal(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    candle = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    signal_id = tracker.create_signal(_reading(), candle, candle)
    assert tracker.record_decision(signal_id, "take", "123") == "saved"
    assert tracker.record_decision(signal_id, "skip", "123") == "already:take"


def test_state_round_trip(tmp_path):
    tracker = SignalTracker(tmp_path / "signals.db")
    assert tracker.get_state("offset") is None
    tracker.set_state("offset", "42")
    assert tracker.get_state("offset") == "42"
