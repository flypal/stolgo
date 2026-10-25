"""Rule checks for examples/pa/channel_breakout.py on hand-built hourly bars."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from stolgo import Backtest

_PATH = Path(__file__).resolve().parents[2] / "examples" / "pa" / "channel_breakout.py"
_spec = importlib.util.spec_from_file_location("channel_breakout_example", _PATH)
example = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = example
_spec.loader.exec_module(example)

# Day 1 sits at 100, so day 2's pivot is ~100. Day 2 holds a 104.0-104.4 range (~0.38% wide)
# for 22 bars, then a bullish bar closes above it; the next bar trips the 105.0 trigger.
DAY1 = [(100, 100.2, 99.8, 100)] * 24
RANGE = [(104.2, 104.4, 104.0, 104.2), (104.2, 104.3, 104.0, 104.1)] * 11
BREAKOUT = (104.2, 104.9, 104.2, 104.8)
ENTRY = (104.9, 105.2, 104.8, 105.1)
TARGET = (105.2, 106.3, 105.1, 106.2)


def _run(rows, *, mirror: bool = False, range_pct: float = 0.005):
    if mirror:
        rows = [(200 - o, 200 - lo, 200 - h, 200 - c) for o, h, lo, c in rows]
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)
    cfg = example.ChannelBreakoutConfig(range_pct=range_pct, cash=10_000, tick=0.1)
    return Backtest(example.ChannelBreakout(cfg), df, cash=10_000, commission=0.0).run()


@pytest.mark.parametrize("mirror", [False, True], ids=["long", "short"])
def test_breakout_of_tight_range_trades_one_r(mirror: bool) -> None:
    trades = _run([*DAY1, *RANGE, BREAKOUT, ENTRY, TARGET], mirror=mirror).trades
    assert len(trades) == 1
    row = trades.iloc[0]
    flip = (lambda p: 200 - p) if mirror else (lambda p: p)
    assert row["side"] == ("short" if mirror else "long")
    assert row["entry_price"] == pytest.approx(flip(105.0))
    assert row["stop_loss"] == pytest.approx(flip(104.0))
    assert row["exit_price"] == pytest.approx(flip(106.0))
    assert row["qty"] == pytest.approx(100.0)
    assert row["r_multiple"] == pytest.approx(1.0)


def test_range_wider_than_limit_is_ignored() -> None:
    assert _run([*DAY1, *RANGE, BREAKOUT, ENTRY, TARGET], range_pct=0.003).events == []


def test_breakout_against_daily_pivot_is_ignored() -> None:
    above_day = [(110, 110.2, 109.8, 110)] * 24
    assert _run([*above_day, *RANGE, BREAKOUT, ENTRY, TARGET]).events == []


def test_bearish_breakout_bar_is_ignored() -> None:
    weak = (104.9, 104.9, 104.2, 104.8)
    assert _run([*DAY1, *RANGE, weak, ENTRY, TARGET]).events == []
