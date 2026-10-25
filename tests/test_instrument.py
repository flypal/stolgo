"""Tests for stolgo.instrument and instrument-aware accounting."""

from __future__ import annotations

import pandas as pd
import pytest
from stolgo import Backtest, Future, Spot, Strategy
from stolgo.core.events import FillEvent
from stolgo.core.types import Bar, Fill, Side
from stolgo.oms.commission import PerLotCommission
from stolgo.portfolio.portfolio import Portfolio
from stolgo.report.trades import build_trades_from_fills
from stolgo.strategy.context import Context


def _fill(side: Side, qty: float, price: float, commission: float = 0.0, ts: int = 1) -> Fill:
    return Fill("f", "o", "X", side, qty, price, commission, ts)


def _bar(close: float) -> Bar:
    return Bar(ts=1, open=close, high=close, low=close, close=close, volume=1.0, symbol="X")


def test_future_round_qty_and_max_position() -> None:
    fut = Future(symbol="KQ.i@SHFE.rb", multiplier=10, margin_rate=0.1)
    assert fut.round_qty(2.9) == 2.0
    assert fut.max_position(equity=100_000, price=4_000) == 25.0
    assert fut.max_position(equity=-1, price=4_000) == 0.0


def test_future_rejects_invalid_spec() -> None:
    with pytest.raises(ValueError):
        Future(symbol="X", multiplier=0)
    with pytest.raises(ValueError):
        Future(symbol="X", multiplier=10, margin_rate=0)


def test_spot_lot_rounding() -> None:
    assert Spot(lot_size=100).round_qty(250) == 200
    assert Spot().round_qty(0.37) == 0.37


def test_future_portfolio_only_moves_cash_by_pnl() -> None:
    pf = Portfolio(100_000, "RB", Future(symbol="RB", multiplier=10))
    pf.apply_fill(_fill(Side.BUY, 2, 4_000, commission=5))
    assert pf.cash == pytest.approx(100_000 - 5)
    assert pf.mark_to_market(_bar(4_050)) == pytest.approx(100_000 - 5 + 50 * 2 * 10)

    pf.apply_fill(_fill(Side.SELL, 2, 4_050, commission=5))
    assert pf.position.flat
    assert pf.cash == pytest.approx(100_000 - 10 + 1_000)


def test_future_short_profit_and_flip() -> None:
    pf = Portfolio(100_000, "RB", Future(symbol="RB", multiplier=10))
    pf.apply_fill(_fill(Side.SELL, 1, 4_000))
    assert pf.position.qty == -1
    assert pf.mark_to_market(_bar(3_900)) == pytest.approx(101_000)

    pf.apply_fill(_fill(Side.BUY, 3, 3_900))
    assert pf.position.qty == 2
    assert pf.position.avg_entry_price == 3_900
    assert pf.cash == pytest.approx(101_000)


def test_clamp_order_respects_margin() -> None:
    pf = Portfolio(100_000, "RB", Future(symbol="RB", multiplier=10, margin_rate=0.1))
    assert pf.clamp_order_qty(Side.BUY, 1_000, price=4_000) == 25
    pf.apply_fill(_fill(Side.BUY, 25, 4_000))
    assert pf.clamp_order_qty(Side.BUY, 5, price=4_000) == 0
    assert pf.clamp_order_qty(Side.SELL, 25, price=4_000) == 25


def test_clamp_counts_pending_orders() -> None:
    pf = Portfolio(100_000, "RB", Future(symbol="RB", multiplier=10, margin_rate=0.1))
    pf.apply_fill(_fill(Side.SELL, 20, 4_000))
    assert pf.clamp_order_qty(Side.BUY, 30, price=4_000, pending=20) == 25


def test_spot_clamp_is_unchecked() -> None:
    pf = Portfolio(1_000, "X")
    assert pf.clamp_order_qty(Side.BUY, 1e6, price=100) == 1e6


def test_short_round_trip_trade_uses_multiplier() -> None:
    sell = FillEvent(fill=_fill(Side.SELL, 2, 4_000, commission=4, ts=1), order_id="o1", index=0)
    buy = FillEvent(fill=_fill(Side.BUY, 2, 3_950, commission=4, ts=2), order_id="o2", index=1)
    trades = build_trades_from_fills([sell, buy], multiplier=10)
    row = trades.iloc[0]
    assert row["side"] == "short"
    assert row["gross_pnl"] == pytest.approx(50 * 2 * 10)
    assert row["net_pnl"] == pytest.approx(1_000 - 8)


def test_per_lot_commission() -> None:
    assert PerLotCommission(3.5).fee(Side.BUY, 40_000, 4) == 14.0


class _BuyThenClose(Strategy):
    def __init__(self, qty: float | None = None, size_pct: float | None = None) -> None:
        self._qty = qty
        self._size_pct = size_pct

    def on_bar(self, ctx: Context) -> None:
        if ctx.i == 5 and ctx.position.flat:
            ctx.buy(qty=self._qty, size_pct=self._size_pct)
        if ctx.i == 20 and not ctx.position.flat:
            ctx.close()


def test_future_lots_match_spot_units(synthetic_100bars_df: pd.DataFrame) -> None:
    fut = Backtest(
        _BuyThenClose(qty=2),
        synthetic_100bars_df,
        instrument=Future(symbol="X", multiplier=10, margin_rate=0.1),
    ).run()
    spot = Backtest(_BuyThenClose(qty=20), synthetic_100bars_df).run()
    assert fut.equity.to_numpy() == pytest.approx(spot.equity.to_numpy())
    assert fut.trades.iloc[0]["qty"] == 2
    assert fut.trades.iloc[0]["net_pnl"] == pytest.approx(spot.trades.iloc[0]["net_pnl"])
    assert fut.params["instrument"]["multiplier"] == 10


def test_future_size_pct_is_notional_exposure(synthetic_100bars_df: pd.DataFrame) -> None:
    result = Backtest(
        _BuyThenClose(size_pct=1.0),
        synthetic_100bars_df,
        instrument=Future(symbol="X", multiplier=10, margin_rate=0.1),
        cash=100_000,
    ).run()
    lots = result.trades.iloc[0]["qty"]
    close_at_signal = float(synthetic_100bars_df["close"].iloc[5])
    assert lots == int(100_000 / (close_at_signal * 10))


def test_future_explicit_qty_clamped_by_margin(synthetic_100bars_df: pd.DataFrame) -> None:
    result = Backtest(
        _BuyThenClose(qty=10_000),
        synthetic_100bars_df,
        instrument=Future(symbol="X", multiplier=10, margin_rate=0.5),
        cash=100_000,
    ).run()
    close_at_signal = float(synthetic_100bars_df["close"].iloc[5])
    assert result.trades.iloc[0]["qty"] == int(100_000 / (close_at_signal * 10 * 0.5))


def test_future_bps_commission_on_contract_notional(synthetic_100bars_df: pd.DataFrame) -> None:
    result = Backtest(
        _BuyThenClose(qty=1),
        synthetic_100bars_df,
        instrument=Future(symbol="X", multiplier=10),
        commission=0.001,
    ).run()
    row = result.trades.iloc[0]
    expected = (row["entry_price"] + row["exit_price"]) * 10 * 0.001
    assert row["commission"] == pytest.approx(expected)


def test_instrument_commission_overrides_config(synthetic_100bars_df: pd.DataFrame) -> None:
    result = Backtest(
        _BuyThenClose(qty=3),
        synthetic_100bars_df,
        instrument=Future(symbol="X", multiplier=10, commission=PerLotCommission(2.0)),
        commission=0.5,
    ).run()
    assert result.trades.iloc[0]["commission"] == pytest.approx(3 * 2.0 * 2)


def test_short_strategy_on_future(synthetic_100bars_df: pd.DataFrame) -> None:
    class ShortThenCover(Strategy):
        def on_bar(self, ctx: Context) -> None:
            if ctx.i == 5 and ctx.position.flat:
                ctx.sell(qty=1)
            if ctx.i == 20 and not ctx.position.flat:
                ctx.close()

    result = Backtest(
        ShortThenCover(), synthetic_100bars_df, instrument=Future(symbol="X", multiplier=10)
    ).run()
    row = result.trades.iloc[0]
    assert row["side"] == "short"
    assert row["gross_pnl"] == pytest.approx((row["entry_price"] - row["exit_price"]) * 10)
    assert result.equity.iloc[-1] == pytest.approx(100_000 + row["net_pnl"])


def test_reversal_in_one_bar_respects_margin(synthetic_100bars_df: pd.DataFrame) -> None:
    class Reverse(Strategy):
        def on_bar(self, ctx: Context) -> None:
            if ctx.i == 5:
                ctx.sell(qty=10_000)
            if ctx.i == 10:
                ctx.close()
                ctx.buy(qty=10_000)

    result = Backtest(
        Reverse(),
        synthetic_100bars_df,
        instrument=Future(symbol="X", multiplier=10, margin_rate=0.5),
    ).run()
    positions = result.positions["qty"]
    close = synthetic_100bars_df["close"]
    cap_short = int(100_000 / (close.iloc[5] * 10 * 0.5))
    assert positions.min() == -cap_short
    assert 0 < positions.max() <= int(result.equity.iloc[10] / (close.iloc[10] * 10 * 0.5))


def test_bracket_sizing_uses_multiplier() -> None:
    import numpy as np
    from stolgo import trade
    from stolgo.core.types import Position
    from stolgo.strategy.context import BarDataView

    arr = np.array([4_000.0])
    ctx = Context(
        i=0,
        data=BarDataView(
            _open=arr, _high=np.array([4_010.0]), _low=np.array([3_980.0]),
            _close=np.array([4_000.0]), _volume=arr, _limit=0,
        ),
        position=Position(symbol="RB"),
        instrument=Future(symbol="RB", multiplier=10),
    )
    bracket = trade.long(ctx, stop="candle_low", size_risk_pct=0.01, cash=100_000)
    assert bracket is not None
    assert bracket.qty == 5.0
