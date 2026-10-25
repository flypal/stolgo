"""Tests for stolgo.report.trades."""

import pandas as pd
import pytest

from stolgo.core.events import FillEvent
from stolgo.core.types import Fill, Side
from stolgo.report.trades import build_trades_from_fills


def test_round_trip_trade():
    buy = FillEvent(
        fill=Fill("f1", "o1", "X", Side.BUY, 10.0, 100.0, 0.1, 1),
        order_id="o1",
        index=0,
    )
    sell = FillEvent(
        fill=Fill("f2", "o2", "X", Side.SELL, 10.0, 110.0, 0.1, 2),
        order_id="o2",
        index=1,
    )
    trades = build_trades_from_fills([buy, sell])
    assert len(trades) == 1
    assert trades.iloc[0]["gross_pnl"] == 100.0
    assert trades.iloc[0]["net_pnl"] == pytest.approx(99.8, rel=1e-3)


def _fe(fid, side, qty, price, ts, stop_loss=None):
    return FillEvent(
        fill=Fill(fid, fid, "X", side, qty, price, 0.0, ts),
        order_id=fid,
        index=ts,
        stop_loss=stop_loss,
    )


def test_r_multiple_uses_risk_to_bracket_stop():
    trades = build_trades_from_fills(
        [_fe("o1", Side.SELL, 2.0, 100.0, 0, stop_loss=104.0), _fe("o2", Side.BUY, 2.0, 106.0, 1)],
        multiplier=10,
    )
    row = trades.iloc[0]
    assert row["stop_loss"] == 104.0
    assert row["initial_risk"] == 80.0
    assert row["r_multiple"] == pytest.approx(-1.5)


def test_r_multiple_is_nan_without_stop():
    trades = build_trades_from_fills([_fe("o1", Side.BUY, 1.0, 100.0, 0), _fe("o2", Side.SELL, 1.0, 110.0, 1)])
    assert pd.isna(trades.iloc[0]["initial_risk"])
    assert pd.isna(trades.iloc[0]["r_multiple"])
