# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D
# [ ] no pandas in oms/portfolio hot path

"""Instrument accounting strategies.

Strategies trade native units (shares / lots) and size with ``size_pct`` of
capital. Contract multiplier, lot rounding, margin and cash settlement are
delegated to an :class:`Instrument`, so strategy code is identical for stocks,
crypto and futures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from stolgo.oms.commission import CommissionModel

_EPS = 1e-9


class Instrument(Protocol):
    symbol: str
    multiplier: float
    commission: CommissionModel | None

    def round_qty(self, qty: float) -> float:
        """Round an order size down to a tradable quantity."""
        ...

    def cash_delta(self, signed_qty: float, price: float, realized_pnl: float) -> float:
        """Cash change for a fill (before commission); ``signed_qty`` < 0 for sells."""
        ...

    def market_value(self, qty: float, price: float, avg_price: float) -> float:
        """Value of the open position that is not already in cash."""
        ...

    def sizing_capital(self, cash: float, equity: float) -> float:
        """Capital that ``size_pct`` is applied to."""
        ...

    def max_position(self, equity: float, price: float) -> float:
        """Largest absolute position the account can carry (``inf`` = unchecked)."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary stored in ``RunResult.params``."""
        ...


def _commission_name(commission: CommissionModel | None) -> str | None:
    return type(commission).__name__ if commission is not None else None


@dataclass(frozen=True)
class Spot:
    """Fully cash-settled asset (stock, crypto spot): buying pays the whole notional."""

    symbol: str = "UNKNOWN"
    lot_size: float = 0.0
    commission: CommissionModel | None = None
    multiplier: float = field(default=1.0, init=False)

    def round_qty(self, qty: float) -> float:
        if self.lot_size <= 0:
            return qty
        return math.floor(qty / self.lot_size + _EPS) * self.lot_size

    def cash_delta(self, signed_qty: float, price: float, realized_pnl: float) -> float:
        return -signed_qty * price

    def market_value(self, qty: float, price: float, avg_price: float) -> float:
        return qty * price

    def sizing_capital(self, cash: float, equity: float) -> float:
        return cash

    def max_position(self, equity: float, price: float) -> float:
        return math.inf

    def describe(self) -> dict[str, Any]:
        return {
            "type": "spot",
            "symbol": self.symbol,
            "lot_size": self.lot_size,
            "commission": _commission_name(self.commission),
        }


@dataclass(frozen=True)
class Future:
    """Margin-traded contract: only PnL moves cash, notional = price × multiplier."""

    symbol: str
    multiplier: float
    margin_rate: float = 0.15
    price_tick: float = 0.0
    commission: CommissionModel | None = None

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError("Future.multiplier must be positive")
        if not 0 < self.margin_rate <= 1:
            raise ValueError("Future.margin_rate must be in (0, 1]")

    def round_qty(self, qty: float) -> float:
        return float(math.floor(qty + _EPS))

    def cash_delta(self, signed_qty: float, price: float, realized_pnl: float) -> float:
        return realized_pnl

    def market_value(self, qty: float, price: float, avg_price: float) -> float:
        return (price - avg_price) * qty * self.multiplier

    def sizing_capital(self, cash: float, equity: float) -> float:
        return equity

    def max_position(self, equity: float, price: float) -> float:
        per_lot = price * self.multiplier * self.margin_rate
        if equity <= 0 or per_lot <= 0:
            return 0.0
        return float(math.floor(equity / per_lot + _EPS))

    def describe(self) -> dict[str, Any]:
        return {
            "type": "future",
            "symbol": self.symbol,
            "multiplier": self.multiplier,
            "margin_rate": self.margin_rate,
            "price_tick": self.price_tick,
            "commission": _commission_name(self.commission),
        }
