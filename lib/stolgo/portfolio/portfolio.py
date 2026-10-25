# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

from __future__ import annotations

import math

from stolgo.core.types import Bar, Fill, Position, Side
from stolgo.instrument.base import Instrument, Spot


class Portfolio:
    def __init__(self, cash: float, symbol: str, instrument: Instrument | None = None) -> None:
        self._cash = cash
        self._position = Position(symbol=symbol)
        self._instrument = instrument or Spot(symbol=symbol)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def position(self) -> Position:
        return self._position

    @property
    def instrument(self) -> Instrument:
        return self._instrument

    def apply_fill(self, fill: Fill) -> None:
        pos = self._position
        signed = fill.qty if fill.side == Side.BUY else -fill.qty
        old = pos.qty
        realized = 0.0
        if old * signed < 0:
            closed = min(abs(signed), abs(old))
            realized = (
                (fill.price - pos.avg_entry_price)
                * closed
                * math.copysign(1.0, old)
                * self._instrument.multiplier
            )

        new = old + signed
        if abs(new) < 1e-12:
            pos.qty = 0.0
            pos.avg_entry_price = 0.0
        else:
            if old * signed >= 0:
                pos.avg_entry_price = (
                    pos.avg_entry_price * abs(old) + fill.price * abs(signed)
                ) / abs(new)
            elif old * new < 0:
                pos.avg_entry_price = fill.price
            pos.qty = new

        self._cash += self._instrument.cash_delta(signed, fill.price, realized) - fill.commission

    def equity(self, price: float) -> float:
        pos = self._position
        return self._cash + self._instrument.market_value(pos.qty, price, pos.avg_entry_price)

    def mark_to_market(self, bar: Bar) -> float:
        return self.equity(bar.close)

    def clamp_order_qty(
        self, side: Side, qty: float, price: float, *, pending: float = 0.0
    ) -> float:
        """Shrink orders that would push the position past the instrument's limit.

        ``pending`` is the signed qty of orders submitted but not yet filled.
        """
        cap = self._instrument.max_position(self.equity(price), price)
        if math.isinf(cap):
            return qty
        cur = self._position.qty + pending
        signed = qty if side == Side.BUY else -qty
        target = cur + signed
        if abs(target) <= cap or abs(target) <= abs(cur):
            return qty
        delta = math.copysign(cap, target) - cur
        if delta * signed <= 0:
            return 0.0
        return abs(delta)
