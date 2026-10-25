# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

from __future__ import annotations

import numpy as np

from stolgo.core.config import RunConfig
from stolgo.core.types import OrderIntent, Side
from stolgo.portfolio.portfolio import Portfolio


def _reduces_position(intent: OrderIntent, portfolio: Portfolio) -> bool:
    pos = portfolio.position.qty
    if pos == 0.0 or intent.qty is None:
        return False
    opposite = (intent.side == Side.SELL) == (pos > 0)
    return opposite and intent.qty <= abs(pos) + 1e-12


def apply_risk(
    intent: OrderIntent | None,
    portfolio: Portfolio,
    equity_curve: list[float],
    config: RunConfig,
) -> OrderIntent | None:
    if intent is None:
        return None
    if len(equity_curve) < 2:
        return intent
    peak = max(equity_curve)
    current = equity_curve[-1]
    if peak > 0 and (peak - current) / peak > 0.5 and not _reduces_position(intent, portfolio):
        return None
    return intent
