"""Long/short SMA trend on a TqSdk weighted futures index, exported for the Stolgo UI.

Needs a Shinny (快期) account in TQ_USER / TQ_PASSWORD. Free accounts load the
latest 10000 bars; pro accounts load the full requested range.

Run:
    PYTHONPATH=lib .venv/bin/python examples/tqsdk_futures_sma_backtest.py --symbol SHFE.rb
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stolgo import Backtest, Context, Strategy, TqSdk
from stolgo.oms.commission import PerLotCommission
from stolgo.report.exporters import export_all
from stolgo.signals import sma


class FuturesSmaTrend(Strategy):
    fast_period = 10
    slow_period = 30
    size_pct = 1.0

    def on_start(self, ctx: Context) -> None:
        close = ctx.data.close
        self._fast = sma(close, self.fast_period)
        self._slow = sma(close, self.slow_period)

    def on_bar(self, ctx: Context) -> None:
        if ctx.i < self.slow_period:
            return

        long_bias = self._fast[ctx.i] > self._slow[ctx.i]
        qty = ctx.position.qty
        if long_bias and qty <= 0:
            ctx.close(tag="flip_long")
            ctx.buy(size_pct=self.size_pct, tag="long")
        elif not long_bias and qty >= 0:
            ctx.close(tag="flip_short")
            ctx.sell(size_pct=self.size_pct, tag="short")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TqSdk weighted-index futures SMA backtest.")
    parser.add_argument("--symbol", default="SHFE.rb")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--days", type=int, default=3 * 365)
    parser.add_argument("--margin-rate", type=float, default=0.15)
    parser.add_argument("--fee-per-lot", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("runs/tqsdk-futures-sma"))
    args = parser.parse_args(argv)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    with TqSdk() as tq:
        df = tq.history(args.symbol, args.interval, start, end)
        instrument = tq.instrument(
            args.symbol,
            margin_rate=args.margin_rate,
            commission=PerLotCommission(args.fee_per_lot),
        )
    print(f"{instrument.symbol}: {len(df):,} bars {df.index.min()} -> {df.index.max()}")
    print(f"multiplier={instrument.multiplier} margin_rate={instrument.margin_rate}")

    result = Backtest(
        FuturesSmaTrend(),
        df,
        instrument=instrument,
        symbol=instrument.symbol,
        interval=args.interval,
        cash=500_000,
        slippage_bps=1.0,
    ).run()

    print(result.summary())
    export_all(result, args.output, strategy_name=f"{instrument.symbol} SMA long/short")
    print(f"Exported to {args.output.resolve()}")


if __name__ == "__main__":
    main()
