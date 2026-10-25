# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

"""Build round-trip trade log from fill events (cold path)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from stolgo.core.events import FillEvent
from stolgo.core.types import Side

TRADE_COLUMNS: tuple[str, ...] = (
    "entry_ts",
    "exit_ts",
    "side",
    "entry_price",
    "exit_price",
    "stop_loss",
    "qty",
    "multiplier",
    "gross_pnl",
    "net_pnl",
    "commission",
    "initial_risk",
    "r_multiple",
    "tag",
)


@dataclass
class _Lot:
    entry_ts: pd.Timestamp
    entry_price: float
    qty: float
    entry_commission: float
    direction: float
    stop_loss: float | None


def build_trades_from_fills(fills: list[FillEvent], *, multiplier: float = 1.0) -> pd.DataFrame:
    """FIFO-match opposing fills into long and short round trips.

    ``r_multiple`` is net PnL over the initial risk between entry and the bracket stop;
    trades opened without a stop have NaN risk and R.
    """
    lots: list[_Lot] = []
    rows: list[dict] = []

    for fe in fills:
        f = fe.fill
        ts = pd.Timestamp(f.ts, unit="ns", tz="UTC")
        direction = 1.0 if f.side == Side.BUY else -1.0
        remaining = f.qty
        commission_per_unit = f.commission / f.qty if f.qty else 0.0

        while remaining > 1e-12 and lots and lots[0].direction != direction:
            lot = lots[0]
            match_qty = min(remaining, lot.qty)
            entry_comm_part = lot.entry_commission * (match_qty / lot.qty) if lot.qty else 0.0
            exit_comm_part = commission_per_unit * match_qty
            gross = (f.price - lot.entry_price) * match_qty * lot.direction * multiplier
            net = gross - entry_comm_part - exit_comm_part
            risk = math.nan
            if lot.stop_loss is not None:
                risk = abs(lot.entry_price - lot.stop_loss) * match_qty * multiplier
            r_multiple = net / risk if risk > 0 else math.nan

            rows.append(
                {
                    "entry_ts": lot.entry_ts,
                    "exit_ts": ts,
                    "side": "long" if lot.direction > 0 else "short",
                    "entry_price": lot.entry_price,
                    "exit_price": f.price,
                    "stop_loss": math.nan if lot.stop_loss is None else lot.stop_loss,
                    "qty": match_qty,
                    "multiplier": multiplier,
                    "gross_pnl": gross,
                    "net_pnl": net,
                    "commission": entry_comm_part + exit_comm_part,
                    "initial_risk": risk,
                    "r_multiple": r_multiple,
                    "tag": f.order_id,
                }
            )

            lot.qty -= match_qty
            lot.entry_commission -= entry_comm_part
            remaining -= match_qty
            if lot.qty <= 1e-12:
                lots.pop(0)

        if remaining > 1e-12:
            lots.append(
                _Lot(
                    entry_ts=ts,
                    entry_price=f.price,
                    qty=remaining,
                    entry_commission=commission_per_unit * remaining,
                    direction=direction,
                    stop_loss=fe.stop_loss,
                )
            )

    if not rows:
        return pd.DataFrame(columns=list(TRADE_COLUMNS))
    return pd.DataFrame(rows, columns=list(TRADE_COLUMNS))
