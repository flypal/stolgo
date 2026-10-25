"""Donchian channel breakout with a 1:1 bracket measured from the broken range edge.

Rules
-----
1. Range: the prior ``lookback`` bars (breakout bar excluded); H = highest high,
   L = lowest low.
2. Signal: close above the channel high with EMA(``ema_period``) rising -> long;
   close below the channel low with EMA falling -> short.
   Slope = EMA[t] - EMA[t - ``ema_slope_bars``]; a one-bar slope always agrees with a
   channel breakout, so it filters nothing.
3. Range height:
   long:  x = H - close of the lowest-low bar
   short: x = close of the highest-high bar - L
4. Entry and bracket depend on ``anchor``; every entry order is valid for the bar after
   the breakout bar only, then expires and the strategy waits for the next breakout.
   edge:     stop order at breakout high + 1 tick (low - 1 tick for shorts);
             target = H + x / L - x, stop = H - x / L + x. Skipped if the trigger
             is already beyond the target.
   entry:    same stop order; target and stop are x away from the actual fill.
   pullback: limit order back at the broken edge (H / L); same bracket as ``edge``.
5. Exit: stop and target rest as one-cancels-other orders from the fill bar on and fill
   at their price, or at the open if it gaps through. If one bar touches both, the stop
   is assumed to hit first.
6. Size: lose ``risk_pct`` of starting cash if filled at the order price and stopped out.

Run:
    PYTHONPATH=lib .venv/bin/python examples/breakout_1r1_backtest.py --symbol SHFE.rb --interval 5m
    PYTHONPATH=lib .venv/bin/python examples/breakout_1r1_backtest.py --source bandl --symbol BTCUSDT --interval 1d --long-only --tick 0.01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from stolgo import Backtest, Bandl, Context, Strategy, trade
from stolgo.report.exporters import export_all
from stolgo.signals import ema


@dataclass
class BreakoutConfig:
    lookback: int = 20
    ema_period: int = 20
    ema_slope_bars: int = 5
    anchor: Literal["edge", "entry", "pullback"] = "edge"
    risk_pct: float = 0.01
    cash: float = 500_000.0
    allow_short: bool = True
    tick: float | None = None


class DonchianBreakout1R1(Strategy):
    def __init__(self, cfg: BreakoutConfig) -> None:
        self.cfg = cfg

    def on_start(self, ctx: Context) -> None:
        self._ema = ema(ctx.data.close, self.cfg.ema_period)
        self._tick = self.cfg.tick or getattr(ctx.instrument, "price_tick", 0.0)
        if not self._tick or self._tick <= 0:
            raise ValueError("price tick unknown: set BreakoutConfig.tick (--tick)")

    def on_bar(self, ctx: Context) -> None:
        n = self.cfg.lookback
        k = self.cfg.ema_slope_bars
        if not ctx.position.flat or ctx.i < max(n, self.cfg.ema_period + k):
            return

        high, low, close = ctx.data.high, ctx.data.low, ctx.data.close
        w_high, w_low, w_close = high[-(n + 1) : -1], low[-(n + 1) : -1], close[-(n + 1) : -1]
        slope = self._ema[ctx.i] - self._ema[ctx.i - k]
        if close[-1] > w_high.max() and slope > 0:
            direction, edge = 1, w_high.max()
            x = edge - w_close[np.argmin(w_low)]
            trigger = high[-1] + self._tick
        elif self.cfg.allow_short and close[-1] < w_low.min() and slope < 0:
            direction, edge = -1, w_low.min()
            x = w_close[np.argmax(w_high)] - edge
            trigger = low[-1] - self._tick
        else:
            return

        if x <= 0:
            return
        pullback = self.cfg.anchor == "pullback"
        price = edge if pullback else trigger
        center = trigger if self.cfg.anchor == "entry" else edge
        place = trade.buy_bracket if direction > 0 else trade.sell_bracket
        place(
            ctx,
            stop_loss=center - direction * x,
            take_profit=center + direction * x,
            limit_price=price if pullback else None,
            stop_price=None if pullback else price,
            valid_bars=1,
            exits_from="fill" if self.cfg.anchor == "entry" else "order",
            size_risk_pct=self.cfg.risk_pct,
            cash=self.cfg.cash,
            tag="breakout_long" if direction > 0 else "breakout_short",
        )


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, object | None]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    if args.source == "bandl":
        df = Bandl().history(args.symbol, args.interval, start, end)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df.sort_index(), None

    from stolgo import TqSdk
    from stolgo.oms.commission import PerLotCommission

    with TqSdk() as tq:
        df = tq.history(args.symbol, args.interval, start, end)
        instrument = tq.instrument(
            args.symbol,
            margin_rate=args.margin_rate,
            commission=PerLotCommission(args.fee_per_lot),
        )
    return df, instrument


def print_r_stats(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No completed trades.")
        return
    r = trades["r_multiple"]
    wins = int((trades["net_pnl"] > 0).sum())
    print(f"trades {len(trades)}  wins {wins}  losses {len(trades) - wins}  win_rate {wins / len(trades):.1%}")
    print(f"realized R  mean {r.mean():+.3f}  median win {r[r > 0].median():+.3f}  median loss {r[r <= 0].median():+.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Donchian breakout with 1:1 reward:risk.")
    parser.add_argument("--source", choices=["tqsdk", "bandl"], default="tqsdk")
    parser.add_argument("--symbol", default="SHFE.rb")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=3 * 365)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ema-period", type=int, default=20)
    parser.add_argument("--ema-slope-bars", type=int, default=5)
    parser.add_argument("--anchor", choices=["edge", "entry", "pullback"], default="edge")
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--cash", type=float, default=500_000.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--tick", type=float, default=None, help="defaults to the instrument price_tick")
    parser.add_argument("--margin-rate", type=float, default=0.15)
    parser.add_argument("--fee-per-lot", type=float, default=3.0)
    parser.add_argument("--commission", type=float, default=0.001, help="bps-style rate for bandl spot")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = BreakoutConfig(
        lookback=args.lookback,
        ema_period=args.ema_period,
        ema_slope_bars=args.ema_slope_bars,
        anchor=args.anchor,
        risk_pct=args.risk_pct,
        cash=args.cash,
        allow_short=not args.long_only,
        tick=args.tick,
    )
    df, instrument = load_data(args)
    symbol = getattr(instrument, "symbol", args.symbol)
    print(f"{symbol}: {len(df):,} bars {df.index.min()} -> {df.index.max()}")

    strategy = DonchianBreakout1R1(cfg)
    result = Backtest(
        strategy,
        df,
        instrument=instrument,
        symbol=symbol,
        interval=args.interval,
        cash=cfg.cash,
        commission=0.0 if instrument is not None else args.commission,
        slippage_bps=1.0,
    ).run()

    print(result.summary())
    print_r_stats(result.trades)

    slug = f"{args.symbol.lower().replace('.', '-')}-{args.interval}-{cfg.anchor}"
    output = args.output or Path("runs") / f"breakout-1r1-{slug}"
    side = "long/short" if cfg.allow_short else "long-only"
    name = f"{symbol} Donchian{cfg.lookback} EMA{cfg.ema_period} breakout 1:1 {cfg.anchor} ({side})"
    export_all(result, output, strategy_name=name)
    print(f"Exported to {output.resolve()}")


if __name__ == "__main__":
    main()
