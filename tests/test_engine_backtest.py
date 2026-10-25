"""Integration tests for stolgo.core.engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stolgo import Backtest, RunConfig, Strategy
from stolgo.core.exceptions import ModeNotSupportedError
from stolgo.core.types import OrderType, Side
from stolgo.strategy.context import Context


class NoOpStrategy(Strategy):
    pass


class BuyOnceStrategy(Strategy):
    def on_bar(self, ctx: Context) -> None:
        if ctx.i == 5 and ctx.position.flat:
            ctx.buy(qty=10.0)


def test_synthetic_100bars_golden(synthetic_100bars_df: pd.DataFrame) -> None:
    import json
    from pathlib import Path

    result = Backtest(NoOpStrategy(), synthetic_100bars_df, cash=100_000, seed=42).run()
    golden = json.loads(
        (Path(__file__).parent / "fixtures" / "golden_synthetic_100bars.json").read_text()
    )
    assert len(result.equity) == 100
    assert result.equity.iloc[-1] == pytest.approx(golden["final_equity"])
    assert result.metrics["num_trades"] == golden["trade_count"]


def test_buy_once_produces_trades(synthetic_100bars_df: pd.DataFrame) -> None:
    class BuySellStrategy(Strategy):
        def on_bar(self, ctx: Context) -> None:
            if ctx.i == 5 and ctx.position.flat:
                ctx.buy(qty=10.0)
            if ctx.i == 20 and not ctx.position.flat:
                ctx.close()

    result = Backtest(BuySellStrategy(), synthetic_100bars_df, cash=100_000, seed=42).run()
    assert len(result.trades) >= 1
    assert result.metrics["num_trades"] >= 1


def test_engine_requires_backtest_mode() -> None:
    from stolgo.core.engine import Engine

    eng = Engine(RunConfig())
    assert eng._config.mode == "backtest"


def test_vector_lift_parity(synthetic_100bars_df: pd.DataFrame) -> None:
    lookback = 5
    thresh = 0.01
    entry_qty = 10.0

    class EventMomentum(Strategy):
        def on_bar(self, ctx: Context) -> None:
            if ctx.i < lookback:
                return
            mom = ctx.data.close[-1] / ctx.data.close[-(lookback + 1)] - 1
            if mom > thresh and ctx.position.flat:
                ctx.buy(qty=entry_qty)
            elif mom < 0 and not ctx.position.flat:
                ctx.close()

    class VectorMomentum(Strategy):
        vector_entry_qty = entry_qty

        def on_start(self, ctx: Context) -> None:
            close = ctx.data.close
            mom = np.full(len(close), np.nan)
            mom[lookback:] = close[lookback:] / close[:-lookback] - 1
            self.entries = mom > thresh
            self.exits = mom < 0

    event = Backtest(EventMomentum(), synthetic_100bars_df, cash=100_000, seed=42).run()
    vector = Backtest(VectorMomentum(), synthetic_100bars_df, cash=100_000, seed=42).run()
    assert len(event.trades) == len(vector.trades)
    assert event.equity.iloc[-1] == pytest.approx(vector.equity.iloc[-1], rel=1e-9)


def test_drawdown_halt_still_allows_closing() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    close = np.where(np.arange(30) < 5, 100.0, 30.0)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
        index=idx,
    )

    class CrashStrategy(Strategy):
        def on_bar(self, ctx: Context) -> None:
            if ctx.i == 1 and ctx.position.flat:
                ctx.buy(qty=900.0)
            if ctx.i == 20 and not ctx.position.flat:
                ctx.close()
            if ctx.i == 25 and ctx.position.flat:
                ctx.buy(qty=10.0)

    result = Backtest(CrashStrategy(), df, cash=100_000, commission=0.0).run()
    assert result.positions["qty"].iloc[-1] == 0.0
    assert len(result.trades) == 1


def _bars(highs: list[float], lows: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(highs), freq="D", tz="UTC")
    mid = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {"open": mid, "high": highs, "low": lows, "close": mid, "volume": 1.0}, index=idx
    )


class StopEntryForOneBar(Strategy):
    def on_bar(self, ctx: Context) -> None:
        if ctx.i == 1:
            ctx.buy(qty=1.0, stop_price=11.0)
        elif ctx.i == 2:
            ctx.cancel_all()


def test_stop_entry_fills_at_trigger_next_bar() -> None:
    df = _bars([10, 10, 12, 12], [9, 9, 10, 10])
    result = Backtest(StopEntryForOneBar(), df, cash=1_000, commission=0.0).run()
    fills = [fe.fill for fe in result.events]
    assert [(f.side, f.price) for f in fills] == [(Side.BUY, 11.0)]


def test_cancelled_stop_entry_never_fills() -> None:
    df = _bars([10, 10, 10.5, 12], [9, 9, 9.5, 10])
    result = Backtest(StopEntryForOneBar(), df, cash=1_000, commission=0.0).run()
    assert result.events == []
    assert result.positions["qty"].iloc[-1] == 0.0


def test_limit_entry_fills_at_limit_price() -> None:
    class LimitEntry(Strategy):
        def on_bar(self, ctx: Context) -> None:
            if ctx.i == 1:
                ctx.buy(qty=1.0, limit_price=9.5)

    df = _bars([10, 10, 10, 10], [9, 9, 9, 9])
    result = Backtest(LimitEntry(), df, cash=1_000, commission=0.0).run()
    assert [(fe.fill.side, fe.fill.price) for fe in result.events] == [(Side.BUY, 9.5)]
