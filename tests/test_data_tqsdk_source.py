"""Tests for stolgo.data.tqsdk_source (offline TqApi double)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from stolgo import Future, Spot
from stolgo.core.exceptions import DataError
from stolgo.data.cache import ParquetCache
from stolgo.data.tqsdk_source import (
    RangeKlineFetcher,
    RecentKlineFetcher,
    TqSdkDataSource,
    interval_seconds,
    resolve_symbol,
)

TRADING_DAYS = pd.bdate_range("2024-01-02", periods=10)


def _daily_klines(pad: int = 0) -> pd.DataFrame:
    """TqSdk-shaped daily klines stamped 00:00 Beijing, NaN-padded like get_kline_serial."""
    ns = [pd.Timestamp(d, tz="Asia/Shanghai").value for d in TRADING_DAYS]
    close = np.linspace(3_500, 3_590, len(ns))
    df = pd.DataFrame(
        {
            "id": np.arange(len(ns)),
            "datetime": ns,
            "open": close - 5,
            "high": close + 10,
            "low": close - 10,
            "close": close,
            "volume": 1_000.0,
            "open_oi": 5e5,
            "close_oi": 5e5,
        }
    )
    if pad:
        nan_rows = pd.DataFrame(np.nan, index=range(pad), columns=df.columns)
        df = pd.concat([nan_rows, df], ignore_index=True)
    return df


class _Auth:
    def __init__(self, pro: bool) -> None:
        self._pro = pro

    def _has_feature(self, name: str) -> bool:
        return self._pro and name == "tq_dl"


class _Quote:
    volume_multiple = 10
    price_tick = 1.0


class FakeTqApi:
    def __init__(self, *, pro: bool = False, pad: int = 3) -> None:
        self._auth = _Auth(pro)
        self._pad = pad
        self.calls: list[tuple] = []
        self.closed = False

    def get_kline_serial(self, symbol, duration, data_length=200, adj_type=None):
        self.calls.append(("serial", symbol, duration, data_length, adj_type))
        return _daily_klines(pad=self._pad)

    def get_kline_data_series(self, symbol, duration, start_dt, end_dt, adj_type=None):
        self.calls.append(("range", symbol, duration, start_dt, end_dt, adj_type))
        return _daily_klines()

    def get_quote(self, symbol):
        self.calls.append(("quote", symbol))
        return _Quote()

    def close(self) -> None:
        self.closed = True


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 12, 31, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("symbol", "query"),
    [
        ("SHFE.rb", "KQ.i@SHFE.rb"),
        ("SHFE.rb2501", "KQ.i@SHFE.rb"),
        ("KQ.m@SHFE.rb", "KQ.i@SHFE.rb"),
        ("KQ.i@SHFE.rb", "KQ.i@SHFE.rb"),
        ("CZCE.MA501", "KQ.i@CZCE.MA"),
        ("CFFEX.IF2412", "KQ.i@CFFEX.IF"),
    ],
)
def test_futures_resolve_to_weighted_index(symbol: str, query: str) -> None:
    resolved = resolve_symbol(symbol)
    assert resolved.query == query
    assert resolved.kind == "future"


def test_stock_symbol_kept() -> None:
    resolved = resolve_symbol("SSE.600000")
    assert (resolved.query, resolved.kind) == ("SSE.600000", "stock")


@pytest.mark.parametrize("symbol", ["SHFE.cu2501C70000", "BTCUSDT", "NYSE.AAPL"])
def test_unsupported_symbols_raise(symbol: str) -> None:
    with pytest.raises(DataError):
        resolve_symbol(symbol)


def test_interval_seconds() -> None:
    assert interval_seconds("1m") == 60
    assert interval_seconds("15m") == 900
    assert interval_seconds("1h") == 3_600
    assert interval_seconds("1d") == 86_400
    with pytest.raises(DataError):
        interval_seconds("1M")


def test_free_tier_uses_recent_serial(tmp_path) -> None:
    api = FakeTqApi(pro=False)
    src = TqSdkDataSource(api, cache=ParquetCache(tmp_path))
    df = src.history("SHFE.rb2501", "1d", START, END)

    assert isinstance(src.fetcher, RecentKlineFetcher)
    assert api.calls[0][:3] == ("serial", "KQ.i@SHFE.rb", 86_400)
    assert len(df) == len(TRADING_DAYS)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.attrs["symbol"] == "KQ.i@SHFE.rb"


def test_daily_bars_keep_trading_date(tmp_path) -> None:
    src = TqSdkDataSource(FakeTqApi(), cache=ParquetCache(tmp_path))
    df = src.history("SHFE.rb", "1d", START, END)
    expected = pd.DatetimeIndex(TRADING_DAYS).tz_localize("UTC")
    assert (df.index == expected).all()


def test_pro_tier_uses_range(tmp_path) -> None:
    api = FakeTqApi(pro=True)
    src = TqSdkDataSource(api, cache=ParquetCache(tmp_path))
    src.history("DCE.i", "1d", START, END)

    assert isinstance(src.fetcher, RangeKlineFetcher)
    kind, symbol, _, start, end, _ = api.calls[0]
    assert (kind, symbol) == ("range", "KQ.i@DCE.i")
    assert start == START and end == END


def test_explicit_fetcher_overrides_tier(tmp_path) -> None:
    api = FakeTqApi(pro=True)
    src = TqSdkDataSource(api, fetcher="recent", cache=ParquetCache(tmp_path))
    src.history("SHFE.rb", "1d", START, END)
    assert api.calls[0][0] == "serial"


def test_range_filter_is_inclusive(tmp_path) -> None:
    src = TqSdkDataSource(FakeTqApi(), cache=ParquetCache(tmp_path))
    start = datetime(2024, 1, 3, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)
    df = src.history("SHFE.rb", "1d", start, end)
    assert [d.day for d in df.index] == [3, 4, 5]


def test_truncated_free_history_warns(tmp_path) -> None:
    api = FakeTqApi(pad=0)
    src = TqSdkDataSource(
        api, fetcher=RecentKlineFetcher(max_bars=len(TRADING_DAYS)), cache=ParquetCache(tmp_path)
    )
    with pytest.warns(UserWarning, match="at most"):
        src.history("SHFE.rb", "1d", datetime(2020, 1, 1, tzinfo=timezone.utc), END)


def test_cache_hit_skips_fetch(tmp_path) -> None:
    api = FakeTqApi()
    src = TqSdkDataSource(api, cache=ParquetCache(tmp_path))
    src.history("SHFE.rb", "1d", START, END)
    src.history("SHFE.rb", "1d", START, END)
    assert sum(1 for c in api.calls if c[0] == "serial") == 1


def test_stock_requests_forward_adjusted(tmp_path) -> None:
    api = FakeTqApi()
    src = TqSdkDataSource(api, cache=ParquetCache(tmp_path))
    src.history("SSE.600000", "1d", START, END)
    assert api.calls[0][1] == "SSE.600000"
    assert api.calls[0][4] == "F"


def test_empty_range_raises(tmp_path) -> None:
    src = TqSdkDataSource(FakeTqApi(), cache=ParquetCache(tmp_path))
    with pytest.raises(DataError):
        src.history(
            "SHFE.rb",
            "1d",
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 2, 1, tzinfo=timezone.utc),
        )


def test_future_instrument_from_quote() -> None:
    api = FakeTqApi()
    inst = TqSdkDataSource(api, margin_rate=0.12).instrument("SHFE.rb2501")
    assert isinstance(inst, Future)
    assert inst.symbol == "KQ.i@SHFE.rb"
    assert (inst.multiplier, inst.price_tick, inst.margin_rate) == (10.0, 1.0, 0.12)
    assert api.calls == [("quote", "KQ.i@SHFE.rb")]


def test_multiplier_override_skips_quote() -> None:
    api = FakeTqApi()
    inst = TqSdkDataSource(api).instrument("SHFE.rb", multiplier=5, margin_rate=0.2)
    assert (inst.multiplier, inst.margin_rate) == (5, 0.2)
    assert api.calls == []


def test_stock_instrument_is_spot_with_board_lot() -> None:
    inst = TqSdkDataSource(FakeTqApi()).instrument("SZSE.000001")
    assert isinstance(inst, Spot)
    assert inst.lot_size == 100


def test_injected_api_is_not_closed() -> None:
    api = FakeTqApi()
    with TqSdkDataSource(api):
        pass
    assert api.closed is False


def test_missing_credentials_raise(monkeypatch) -> None:
    monkeypatch.delenv("TQ_USER", raising=False)
    monkeypatch.delenv("TQ_PASSWORD", raising=False)
    src = TqSdkDataSource()
    with pytest.raises(DataError, match="TQ_USER"):
        _ = src.api
