"""Every pa level and rule must only read completed bars.

Strategies may evaluate pa once over the whole history in ``on_start``; that is only
valid if the value at bar t equals the value computed from bars ``[0, t]`` alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stolgo.pa as pa


def _intraday_random_walk() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=8 * 48, freq="30min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.3, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.05, 0.4, len(idx))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(100, 1000, len(idx)),
        },
        index=idx,
    )


OHLCV = _intraday_random_walk()
REF = pa.resistance(20)

LEVELS = {
    "resistance": pa.resistance(20),
    "support": pa.support(20),
    "range_high": pa.range_high(20),
    "range_low": pa.range_low(20),
    "resistance_1d": pa.resistance(3, tf="1d"),
    "range_low_1d": pa.range_low(3, tf="1d"),
    "swing_high": pa.swing_high(5),
    "swing_low": pa.swing_low(5),
    "swing_high_1d": pa.swing_high(3, tf="1d"),
    "swing_low_1d": pa.swing_low(3, tf="1d"),
    "prev_day_high": pa.prev_day_high(),
    "prev_day_low": pa.prev_day_low(),
    "pivot_point": pa.pivot_point(),
    "vwap": pa.vwap(),
    "cluster": pa.cluster(pa.resistance(20), pa.prev_day_high()),
}

RULES = {
    "above": pa.above(REF),
    "crosses_above": pa.crosses_above(REF),
    "crosses_below": pa.crosses_below(pa.support(20)),
    "rejected_at": pa.rejected_at(REF),
    "recovered_at": pa.recovered_at(pa.support(20)),
    "near": pa.near(REF),
    "touched": pa.touched(REF),
    "consolidation": pa.consolidation(20, 0.03),
    "consolidation_1d": pa.consolidation(3, 0.05, tf="1d"),
    "breakout_up": pa.breakout_up(20, 0.03),
    "breakout_down": pa.breakout_down(20, 0.03),
    "doji": pa.doji(),
    "hammer": pa.hammer(),
    "bullish_engulfing": pa.bullish_engulfing(),
    "streak": pa.streak.green(3),
    "first_red_day": pa.first_red_day(),
    "run_up": pa.run_up(),
    "parabolic_up": pa.parabolic_up(),
    "giant_uptrend": pa.giant_uptrend(),
    "then": pa.green().then(pa.red()),
}

BARS = range(60, len(OHLCV), 5)


def _same(a, b) -> bool:
    return (pd.isna(a) and pd.isna(b)) or bool(np.isclose(a, b))


@pytest.mark.parametrize("name", LEVELS)
def test_level_has_no_lookahead(name: str) -> None:
    level = LEVELS[name]
    full = level.values(OHLCV).to_numpy()
    leaks = [t for t in BARS if not _same(full[t], level.values(OHLCV.iloc[: t + 1]).iloc[-1])]
    assert leaks == []


@pytest.mark.parametrize("name", RULES)
def test_rule_has_no_lookahead(name: str) -> None:
    rule = RULES[name]
    full = rule.series(OHLCV).to_numpy()
    leaks = [t for t in BARS if full[t] != rule.series(OHLCV.iloc[: t + 1]).iloc[-1]]
    assert leaks == []
