# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

from __future__ import annotations

from dataclasses import replace
from typing import Literal
from uuid import uuid4

from stolgo.core.events import FillEvent
from stolgo.core.types import Bar, Fill, Order, OrderType, Side
from stolgo.oms.commission import CommissionModel
from stolgo.oms.fill_model import FillModel
from stolgo.oms.order_book import OrderBook, touch_price
from stolgo.oms.slippage import SlippageModel

_EMPTY_FILLS: tuple[FillEvent, ...] = ()


class SimBroker:
    def __init__(
        self,
        fill_model: FillModel,
        slippage: SlippageModel,
        commission: CommissionModel,
        *,
        fill_on: Literal["next_open", "close"] = "next_open",
        multiplier: float = 1.0,
    ) -> None:
        self._fill_model = fill_model
        self._slippage = slippage
        self._commission = commission
        self._fill_on = fill_on
        self._multiplier = multiplier
        self._pending: list[Order] = []
        self._book = OrderBook()
        self._seq = 0
        self._net = 0.0

    def submit(self, order: Order) -> str:
        self._pending.append(order)
        return order.order_id

    def cancel(self, order_id: str) -> None:
        self._pending = [o for o in self._pending if o.order_id != order_id]
        self._book.cancel(order_id)

    def open_orders(self) -> list[Order]:
        return list(self._pending) + list(self._book._resting)

    def _next_id(self) -> str:
        self._seq += 1
        return f"ord-{self._seq}"

    def match(self, bar: Bar, *, bar_index: int) -> list[FillEvent]:
        events: list[FillEvent] = []
        still_pending: list[Order] = []
        for order in self._pending:
            if order.order_type != OrderType.MARKET:
                self._book.add(order)
                continue
            price = self._fill_model.fill_price(order, bar, bar_index=bar_index)
            if price is None:
                still_pending.append(order)
                continue
            # A close fill is the bar's last print, so its exit legs start next bar.
            legs_start = None if self._fill_on == "close" else price
            events.extend(self._fill(order, bar, price, bar_index, legs_start))
        self._pending = still_pending

        for order, price in self._book.match(bar):
            if order.oco_id is not None and self._net == 0.0:
                continue
            events.extend(self._fill(order, bar, price, bar_index, price))

        self._pending = [
            o for o in self._pending if o.expire_index is None or o.expire_index > bar_index
        ]
        self._book.expire(bar_index)
        return events if events else list(_EMPTY_FILLS)

    def _fill(
        self,
        order: Order,
        bar: Bar,
        price: float,
        bar_index: int,
        legs_start: float | None,
    ) -> list[FillEvent]:
        fe = self._make_fill(order, bar, price, bar_index)
        signed = fe.fill.qty if fe.fill.side == Side.BUY else -fe.fill.qty
        self._net += signed
        if abs(self._net) < 1e-12:
            self._net = 0.0
            self._book.drop_exit_legs()

        legs = self._exit_legs(order, fe.fill)
        stops = [leg.stop_price for leg in legs if leg.order_type == OrderType.STOP]
        if stops:
            fe = replace(fe, stop_loss=stops[0])
        events = [fe]
        if legs_start is not None:
            for leg in legs:
                leg_price = touch_price(leg, bar, legs_start)
                if leg_price is not None:
                    events.extend(self._fill(leg, bar, leg_price, bar_index, None))
                    return events
        for leg in legs:
            self._book.add(leg)
        return events

    def _exit_legs(self, parent: Order, fill: Fill) -> list[Order]:
        if parent.stop_loss is None and parent.take_profit is None:
            return []
        shift = fill.price - parent.exits_ref if parent.exits_ref is not None else 0.0
        side = Side.SELL if fill.side == Side.BUY else Side.BUY
        legs: list[Order] = []
        if parent.stop_loss is not None:
            legs.append(
                Order(
                    order_id=self._next_id(),
                    symbol=parent.symbol,
                    side=side,
                    order_type=OrderType.STOP,
                    qty=fill.qty,
                    stop_price=parent.stop_loss + shift,
                    tag="stop",
                    oco_id=parent.order_id,
                )
            )
        if parent.take_profit is not None:
            legs.append(
                Order(
                    order_id=self._next_id(),
                    symbol=parent.symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    qty=fill.qty,
                    limit_price=parent.take_profit + shift,
                    tag="target",
                    oco_id=parent.order_id,
                )
            )
        return legs

    def _make_fill(self, order: Order, bar: Bar, price: float, bar_index: int) -> FillEvent:
        px = self._slippage.adjust(order.side, price, order.qty)
        fee = self._commission.fee(order.side, px * self._multiplier, order.qty)
        fill = Fill(
            fill_id=self._next_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=px,
            commission=fee,
            ts=bar.ts,
        )
        return FillEvent(fill=fill, order_id=order.order_id, index=bar_index)

    def create_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        order_type: OrderType,
        *,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tag: str | None = None,
        expire_index: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        exits_ref: float | None = None,
    ) -> Order:
        return Order(
            order_id=self._next_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            tag=tag,
            expire_index=expire_index,
            stop_loss=stop_loss,
            take_profit=take_profit,
            exits_ref=exits_ref,
        )
