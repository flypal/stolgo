# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

from __future__ import annotations

from stolgo.core.types import OrderIntent
from stolgo.portfolio.portfolio import Portfolio


def resolve_qty(
    intent: OrderIntent,
    portfolio: Portfolio,
    price: float,
    cash: float,
    *,
    multiplier: float = 1.0,
) -> float:
    """``size_pct`` targets notional exposure: ``cash * size_pct = qty * price * multiplier``."""
    if intent.qty is not None:
        return intent.qty
    if intent.size_pct is not None:
        return (cash * intent.size_pct) / (price * multiplier)
    return 0.0
