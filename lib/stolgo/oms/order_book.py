# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

from __future__ import annotations

from stolgo.core.types import Bar, Order, OrderType, Side


def touch_price(order: Order, bar: Bar, start: float) -> float | None:
    """Fill price if ``bar`` reaches the order, else None.

    ``start`` is the first price the order could trade at: the bar open, or the entry
    fill price for exit legs that become live mid-bar.
    """
    if order.order_type == OrderType.LIMIT and order.limit_price is not None:
        if order.side == Side.BUY and bar.low <= order.limit_price:
            return min(start, order.limit_price)
        if order.side == Side.SELL and bar.high >= order.limit_price:
            return max(start, order.limit_price)
    elif order.order_type == OrderType.STOP and order.stop_price is not None:
        if order.side == Side.BUY and bar.high >= order.stop_price:
            return max(start, order.stop_price)
        if order.side == Side.SELL and bar.low <= order.stop_price:
            return min(start, order.stop_price)
    return None


def _precedence(order: Order, price: float, start: float) -> int:
    # Bar path is unknown: a leg tradable at ``start`` went first, otherwise assume the stop.
    if price == start:
        return 0
    return 1 if order.order_type == OrderType.STOP else 2


class OrderBook:
    def __init__(self) -> None:
        self._resting: list[Order] = []

    def add(self, order: Order) -> None:
        if order.order_type in (OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT):
            self._resting.append(order)

    def cancel(self, order_id: str) -> None:
        self._resting = [o for o in self._resting if o.order_id != order_id]

    def drop_exit_legs(self) -> None:
        self._resting = [o for o in self._resting if o.oco_id is None]

    def expire(self, bar_index: int) -> None:
        self._resting = [
            o for o in self._resting if o.expire_index is None or o.expire_index > bar_index
        ]

    def match(self, bar: Bar) -> list[tuple[Order, float]]:
        hits: list[tuple[Order, float]] = []
        for order in self._resting:
            price = touch_price(order, bar, bar.open)
            if price is not None:
                hits.append((order, price))

        winners: dict[str, tuple[Order, float]] = {}
        for order, price in hits:
            if order.oco_id is None:
                continue
            best = winners.get(order.oco_id)
            if best is None or _precedence(order, price, bar.open) < _precedence(*best, bar.open):
                winners[order.oco_id] = (order, price)

        filled = [
            (o, p) for o, p in hits if o.oco_id is None or winners[o.oco_id][0] is o
        ]
        hit_ids = {o.order_id for o, _ in hits}
        self._resting = [
            o for o in self._resting if o.order_id not in hit_ids and o.oco_id not in winners
        ]
        return filled
