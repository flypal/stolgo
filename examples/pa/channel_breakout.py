"""Channel breakout expressed with stolgo.pa: tight range, first close through it, daily bias.

Rules
-----
1. Setup:   ``pa.consolidation(lookback, range_pct)``: the prior ``lookback`` bars span at
            most ``range_pct`` of price.
2. Trigger: ``pa.crosses_above(pa.range_high(lookback))`` on a bullish bar that closes above
            the daily pivot (``pa.pivot_point``); mirrored for shorts.
3. Entry:   stop order 1 tick beyond the breakout bar, valid for the next bar only.
4. Exit:    stop on the far side of the range (``pa.range_low`` / ``pa.range_high``), target
            ``rr`` times that risk from the trigger; both rest as one-cancels-other orders.
5. Size:    lose ``risk_pct`` of starting cash if filled at the trigger and stopped out.

Run:
    PYTHONPATH=lib .venv/bin/python examples/pa/channel_breakout.py --symbol SHFE.rb --interval 5m
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import stolgo.pa as pa
from stolgo import Backtest, Context, Strategy, TqSdk, trade
from stolgo.oms.commission import PerLotCommission
from stolgo.report.exporters import export_all


@dataclass
class ChannelBreakoutConfig:
    lookback: int = 20
    range_pct: float = 0.004
    rr: float = 1.0
    risk_pct: float = 0.01
    cash: float = 500_000.0
    allow_short: bool = True
    tick: float | None = None


class ChannelBreakout(Strategy):
    def __init__(self, cfg: ChannelBreakoutConfig) -> None:
        self.cfg = cfg

    def on_start(self, ctx: Context) -> None:
        n = self.cfg.lookback
        top, bottom = pa.range_high(n), pa.range_low(n)
        tight = pa.consolidation(n, self.cfg.range_pct)
        pivot = pa.pivot_point()
        long_entry = tight & pa.crosses_above(top) & pa.bullish() & pa.above(pivot)
        short_entry = tight & pa.crosses_below(bottom) & pa.bearish() & pa.below(pivot)

        # pa levels and rules only read completed bars (tests/pa/test_no_lookahead.py),
        # so evaluating them once over the whole history matches evaluating them per bar.
        df = ctx.data.frame()
        self._long = long_entry.series(df).to_numpy()
        self._short = short_entry.series(df).to_numpy()
        self._top = top.values(df).to_numpy()
        self._bottom = bottom.values(df).to_numpy()

        self._tick = self.cfg.tick or getattr(ctx.instrument, "price_tick", 0.0)
        if not self._tick or self._tick <= 0:
            raise ValueError("price tick unknown: set ChannelBreakoutConfig.tick (--tick)")

    def on_bar(self, ctx: Context) -> None:
        i = ctx.i
        if not ctx.position.flat:
            return
        if self._long[i]:
            trigger, stop, place, tag = ctx.data.high[-1] + self._tick, self._bottom[i], trade.buy_bracket, "long"
        elif self.cfg.allow_short and self._short[i]:
            trigger, stop, place, tag = ctx.data.low[-1] - self._tick, self._top[i], trade.sell_bracket, "short"
        else:
            return
        place(
            ctx,
            stop_price=trigger,
            stop_loss=stop,
            take_profit=trigger + self.cfg.rr * (trigger - stop),
            valid_bars=1,
            size_risk_pct=self.cfg.risk_pct,
            cash=self.cfg.cash,
            tag=f"channel_{tag}",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="pa channel breakout on TqSdk futures.")
    parser.add_argument("--symbol", default="SHFE.rb")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=3 * 365)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--range-pct", type=float, default=0.004)
    parser.add_argument("--rr", type=float, default=1.0)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--cash", type=float, default=500_000.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--margin-rate", type=float, default=0.15)
    parser.add_argument("--fee-per-lot", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = ChannelBreakoutConfig(
        lookback=args.lookback,
        range_pct=args.range_pct,
        rr=args.rr,
        risk_pct=args.risk_pct,
        cash=args.cash,
        allow_short=not args.long_only,
    )
    end = datetime.now(timezone.utc)
    with TqSdk() as tq:
        df = tq.history(args.symbol, args.interval, end - timedelta(days=args.days), end)
        instrument = tq.instrument(
            args.symbol, margin_rate=args.margin_rate, commission=PerLotCommission(args.fee_per_lot)
        )

    result = Backtest(
        ChannelBreakout(cfg),
        df,
        instrument=instrument,
        symbol=instrument.symbol,
        interval=args.interval,
        cash=cfg.cash,
        slippage_bps=1.0,
    ).run()
    print(result.summary())
    r = result.trades["r_multiple"]
    if len(r):
        print(f"trades {len(r)}  win_rate {(r > 0).mean():.1%}  mean R {r.mean():+.3f}")

    slug = f"{args.symbol.lower().replace('.', '-')}-{args.interval}"
    output = args.output or Path("runs") / f"pa-channel-breakout-{slug}"
    name = f"{instrument.symbol} pa channel breakout {cfg.lookback} bars <= {cfg.range_pct:.2%} rr {cfg.rr:g}"
    export_all(result, output, strategy_name=name)
    print(f"Exported to {output.resolve()}")


if __name__ == "__main__":
    main()
