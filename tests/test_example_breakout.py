"""Rule checks for examples/breakout_1r1_backtest.py on a hand-built channel breakout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from stolgo import Backtest

_PATH = Path(__file__).resolve().parents[1] / "examples" / "breakout_1r1_backtest.py"
_spec = importlib.util.spec_from_file_location("breakout_1r1_example", _PATH)
breakout = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = breakout
_spec.loader.exec_module(breakout)

# Bars 2-6 form the channel for bar 7: H = 104, lowest low 99 closes at 101, so x = 3.
# Bar 7 closes above H (trigger 106), bar 8 fills the entry, bar 9 reaches every target.
RANGE = [
    (102, 104, 100, 102),
    (102, 103, 101, 102),
    (102, 103, 99, 101),
    (101, 103, 100, 102),
    (102, 104, 101, 103),
    (103, 103, 101, 102),
    (102, 103, 101, 102),
]
BREAKOUT = (102, 105, 102, 105)
ENTRY = (105, 106.5, 104, 106)
TARGET = (106, 109.5, 105.5, 109)


def _df(rows, *, mirror: bool = False) -> pd.DataFrame:
    if mirror:
        rows = [(200 - o, 200 - lo, 200 - h, 200 - c) for o, h, lo, c in rows]
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    return df.assign(volume=1.0)


def _run(rows, anchor: str, *, mirror: bool = False):
    cfg = breakout.BreakoutConfig(
        lookback=5, ema_period=3, ema_slope_bars=1, anchor=anchor, cash=10_000, tick=1.0
    )
    strategy = breakout.DonchianBreakout1R1(cfg)
    return Backtest(strategy, _df(rows, mirror=mirror), cash=10_000, commission=0.0).run()


def _trades(rows, anchor: str, *, mirror: bool = False) -> pd.DataFrame:
    return _run(rows, anchor, mirror=mirror).trades


@pytest.mark.parametrize("mirror", [False, True], ids=["long", "short"])
@pytest.mark.parametrize(
    ("anchor", "entry", "exit", "stop", "r"),
    [
        ("edge", 106, 107, 101, 0.2),
        ("entry", 106, 109, 103, 1.0),
        ("pullback", 104, 107, 101, 1.0),
    ],
)
def test_anchor_sets_entry_and_bracket(anchor, entry, exit, stop, r, mirror) -> None:
    trades = _trades([*RANGE, BREAKOUT, ENTRY, TARGET], anchor, mirror=mirror)
    assert len(trades) == 1
    row = trades.iloc[0]
    flip = (lambda p: 200 - p) if mirror else (lambda p: p)
    assert row["side"] == ("short" if mirror else "long")
    assert row["entry_price"] == pytest.approx(flip(entry))
    assert row["exit_price"] == pytest.approx(flip(exit))
    assert row["stop_loss"] == pytest.approx(flip(stop))
    assert row["initial_risk"] == pytest.approx(100.0)
    assert row["r_multiple"] == pytest.approx(r)


def test_entry_expires_when_next_bar_misses_trigger() -> None:
    miss = (105, 105.5, 104.5, 105)
    flat = (105, 106.5, 104.5, 105)
    assert _run([*RANGE, BREAKOUT, miss, flat], "edge").events == []


def test_edge_skips_breakout_whose_trigger_is_past_target() -> None:
    wide = (102, 107, 102, 105)
    through = (105, 108.5, 104, 106)
    assert _run([*RANGE, wide, through], "edge").events == []
    assert _run([*RANGE, wide, through], "entry").events
