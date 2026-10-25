# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

from __future__ import annotations

from typing import Protocol

from stolgo.core.types import Side


class CommissionModel(Protocol):
    def fee(self, side: Side, price: float, qty: float) -> float: ...


class BpsCommission:
    """Fee as a fraction of notional; ``price`` is per-unit contract value."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def fee(self, side: Side, price: float, qty: float) -> float:
        return price * qty * self._rate


class PerLotCommission:
    """Fixed fee per lot/share, e.g. 3 元/手 for CN futures."""

    def __init__(self, amount: float) -> None:
        self._amount = amount

    def fee(self, side: Side, price: float, qty: float) -> float:
        return qty * self._amount


class FlatCommission:
    def __init__(self, amount: float) -> None:
        self._amount = amount

    def fee(self, side: Side, price: float, qty: float) -> float:
        return self._amount
