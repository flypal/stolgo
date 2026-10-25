# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

"""Strategy context (HLD §4.4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stolgo.core.exceptions import LookaheadError
from stolgo.core.types import OrderIntent, OrderType, Position, Side
from stolgo.instrument.base import Instrument, Spot


@dataclass
class BarDataView:
    _open: np.ndarray
    _high: np.ndarray
    _low: np.ndarray
    _close: np.ndarray
    _volume: np.ndarray
    _limit: int
    _index: pd.DatetimeIndex | None = None

    @property
    def open(self) -> np.ndarray:
        return self._open[: self._limit + 1]

    @property
    def high(self) -> np.ndarray:
        return self._high[: self._limit + 1]

    @property
    def low(self) -> np.ndarray:
        return self._low[: self._limit + 1]

    @property
    def close(self) -> np.ndarray:
        return self._close[: self._limit + 1]

    @property
    def volume(self) -> np.ndarray:
        return self._volume[: self._limit + 1]

    def _check(self, idx: int) -> None:
        if idx > self._limit:
            raise LookaheadError(f"cannot access index {idx} at bar {self._limit}")

    def __len__(self) -> int:
        return self._limit + 1

    @property
    def index(self) -> pd.DatetimeIndex:
        if self._index is not None:
            return self._index[: self._limit + 1]
        return pd.DatetimeIndex(range(self._limit + 1), tz="UTC")


@dataclass
class Context:
    i: int
    data: BarDataView
    position: Position
    entries: np.ndarray | None = None
    exits: np.ndarray | None = None
    instrument: Instrument = field(default_factory=Spot)
    _intents: list[OrderIntent] = field(default_factory=list)
    _cancel_all: bool = False

    def _order(
        self,
        side: Side,
        qty: float | None,
        size_pct: float | None,
        stop_price: float | None,
        limit_price: float | None,
        tag: str | None,
        valid_bars: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        exits_follow_fill: bool = False,
    ) -> OrderIntent:
        if stop_price is not None and limit_price is not None:
            raise ValueError("stop-limit orders are not supported; pass stop_price or limit_price")
        if stop_price is not None:
            order_type = OrderType.STOP
        elif limit_price is not None:
            order_type = OrderType.LIMIT
        else:
            order_type = OrderType.MARKET
        intent = OrderIntent(
            symbol=self.position.symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            size_pct=size_pct,
            limit_price=limit_price,
            stop_price=stop_price,
            tag=tag,
            valid_bars=valid_bars,
            stop_loss=stop_loss,
            take_profit=take_profit,
            exits_follow_fill=exits_follow_fill,
        )
        self._intents.append(intent)
        return intent

    def buy(
        self,
        *,
        qty: float | None = None,
        size_pct: float | None = None,
        stop_price: float | None = None,
        limit_price: float | None = None,
        tag: str | None = None,
        valid_bars: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        exits_follow_fill: bool = False,
    ) -> OrderIntent:
        """Queue a buy. ``valid_bars`` auto-cancels it if unfilled after that many bars.

        ``stop_loss`` / ``take_profit`` attach OCO exit orders that go live on fill; with
        ``exits_follow_fill`` they keep their distance from the actual fill price.
        """
        return self._order(
            Side.BUY, qty, size_pct, stop_price, limit_price, tag,
            valid_bars, stop_loss, take_profit, exits_follow_fill,
        )

    def sell(
        self,
        *,
        qty: float | None = None,
        size_pct: float | None = None,
        stop_price: float | None = None,
        limit_price: float | None = None,
        tag: str | None = None,
        valid_bars: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        exits_follow_fill: bool = False,
    ) -> OrderIntent:
        """Queue a sell; options as in :meth:`buy`."""
        return self._order(
            Side.SELL, qty, size_pct, stop_price, limit_price, tag,
            valid_bars, stop_loss, take_profit, exits_follow_fill,
        )

    def cancel_all(self) -> None:
        """Cancel every unfilled order before this bar's new orders are submitted."""
        self._cancel_all = True

    def consume_cancel_all(self) -> bool:
        out = self._cancel_all
        self._cancel_all = False
        return out

    def close(self, *, tag: str | None = None) -> OrderIntent | None:
        if self.position.flat:
            return None
        qty = abs(self.position.qty)
        if self.position.qty > 0:
            return self.sell(qty=qty, tag=tag)
        return self.buy(qty=qty, tag=tag)

    def consume_intents(self) -> list[OrderIntent]:
        out = list(self._intents)
        self._intents.clear()
        return out
