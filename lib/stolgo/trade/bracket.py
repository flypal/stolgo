# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

"""Risk/reward bracket helpers for strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from stolgo.core.types import OrderIntent, Side
from stolgo.strategy.context import Context

StopKind = Literal["candle_low", "candle_high", "entry"]


@dataclass
class Bracket:
    side: Side
    entry_price: float
    stop_price: float
    target_price: float
    qty: float
    tag: str | None = None


def _stop_price(ctx: Context, stop: StopKind, entry: float, side: Side) -> float:
    low = float(ctx.data.low[-1])
    high = float(ctx.data.high[-1])
    if stop == "candle_low":
        return low
    if stop == "candle_high":
        return high
    return entry


def _target(entry: float, stop: float, rr: tuple[int, int], side: Side) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return entry
    reward = risk * (rr[1] / rr[0])
    if side == Side.BUY:
        return entry + reward
    return entry - reward


def _size(
    ctx: Context,
    entry: float,
    stop: float,
    size_risk_pct: float,
    qty: float | None,
    cash: float,
) -> float:
    if qty is not None:
        return qty
    risk_per_unit = abs(entry - stop) * ctx.instrument.multiplier
    if risk_per_unit <= 0:
        return 0.0
    dollar_risk = cash * size_risk_pct
    return ctx.instrument.round_qty(dollar_risk / risk_per_unit)


def long(
    ctx: Context,
    *,
    rr: tuple[int, int] = (1, 2),
    stop: StopKind = "candle_low",
    size_risk_pct: float = 0.01,
    qty: float | None = None,
    tag: str | None = "long",
    cash: float = 100_000.0,
) -> Bracket | None:
    entry = float(ctx.data.close[-1])
    stop_px = _stop_price(ctx, stop, entry, Side.BUY)
    if entry <= stop_px:
        return None
    q = _size(ctx, entry, stop_px, size_risk_pct, qty, cash)
    if q <= 0:
        return None
    target = _target(entry, stop_px, rr, Side.BUY)
    ctx.buy(qty=q, tag=tag)
    return Bracket(Side.BUY, entry, stop_px, target, q, tag)


def short(
    ctx: Context,
    *,
    rr: tuple[int, int] = (1, 2),
    stop: StopKind = "candle_high",
    size_risk_pct: float = 0.01,
    qty: float | None = None,
    tag: str | None = "short",
    cash: float = 100_000.0,
) -> Bracket | None:
    entry = float(ctx.data.close[-1])
    stop_px = _stop_price(ctx, stop, entry, Side.SELL)
    if entry >= stop_px:
        return None
    q = _size(ctx, entry, stop_px, size_risk_pct, qty, cash)
    if q <= 0:
        return None
    target = _target(entry, stop_px, rr, Side.SELL)
    ctx.sell(qty=q, tag=tag)
    return Bracket(Side.SELL, entry, stop_px, target, q, tag)


def _bracket_order(
    ctx: Context,
    side: Side,
    *,
    stop_loss: float,
    take_profit: float,
    stop_price: float | None,
    limit_price: float | None,
    valid_bars: int | None,
    exits_from: Literal["order", "fill"],
    size_risk_pct: float,
    qty: float | None,
    cash: float,
    tag: str | None,
) -> OrderIntent | None:
    if stop_price is not None:
        entry = stop_price
    elif limit_price is not None:
        entry = limit_price
    else:
        entry = float(ctx.data.close[-1])
    direction = 1.0 if side == Side.BUY else -1.0
    if (entry - stop_loss) * direction <= 0 or (take_profit - entry) * direction <= 0:
        return None
    q = _size(ctx, entry, stop_loss, size_risk_pct, qty, cash)
    if q <= 0:
        return None
    place = ctx.buy if side == Side.BUY else ctx.sell
    return place(
        qty=q,
        stop_price=stop_price,
        limit_price=limit_price,
        valid_bars=valid_bars,
        stop_loss=stop_loss,
        take_profit=take_profit,
        exits_follow_fill=exits_from == "fill",
        tag=tag,
    )


def buy_bracket(
    ctx: Context,
    *,
    stop_loss: float,
    take_profit: float,
    stop_price: float | None = None,
    limit_price: float | None = None,
    valid_bars: int | None = None,
    exits_from: Literal["order", "fill"] = "order",
    size_risk_pct: float = 0.01,
    qty: float | None = None,
    cash: float = 100_000.0,
    tag: str | None = "long",
) -> OrderIntent | None:
    """Long entry with exchange-managed stop-loss and take-profit.

    Entry is a stop order at ``stop_price``, a limit at ``limit_price``, or market.
    Unfilled entries are cancelled after ``valid_bars``. On fill, the stop and target go
    live as one-cancels-other orders; ``exits_from="fill"`` keeps their distance from the
    actual fill instead of the order price. Size risks ``size_risk_pct`` of ``cash`` between
    the order price and ``stop_loss``. Returns None if the levels are on the wrong side.
    """
    return _bracket_order(
        ctx, Side.BUY, stop_loss=stop_loss, take_profit=take_profit,
        stop_price=stop_price, limit_price=limit_price, valid_bars=valid_bars,
        exits_from=exits_from, size_risk_pct=size_risk_pct, qty=qty, cash=cash, tag=tag,
    )


def sell_bracket(
    ctx: Context,
    *,
    stop_loss: float,
    take_profit: float,
    stop_price: float | None = None,
    limit_price: float | None = None,
    valid_bars: int | None = None,
    exits_from: Literal["order", "fill"] = "order",
    size_risk_pct: float = 0.01,
    qty: float | None = None,
    cash: float = 100_000.0,
    tag: str | None = "short",
) -> OrderIntent | None:
    """Short mirror of :func:`buy_bracket`."""
    return _bracket_order(
        ctx, Side.SELL, stop_loss=stop_loss, take_profit=take_profit,
        stop_price=stop_price, limit_price=limit_price, valid_bars=valid_bars,
        exits_from=exits_from, size_risk_pct=size_risk_pct, qty=qty, cash=cash, tag=tag,
    )


def close(ctx: Context, *, tag: str | None = None) -> None:
    ctx.close(tag=tag)


def bracket_hit(ctx: Context, bracket: Bracket | None) -> str | None:
    """Return ``stop``, ``target``, or None if bracket not hit on this bar."""
    if bracket is None:
        return None
    low = float(ctx.data.low[-1])
    high = float(ctx.data.high[-1])
    if bracket.side == Side.BUY:
        if low <= bracket.stop_price:
            return "stop"
        if high >= bracket.target_price:
            return "target"
    else:
        if high >= bracket.stop_price:
            return "stop"
        if low <= bracket.target_price:
            return "target"
    return None
