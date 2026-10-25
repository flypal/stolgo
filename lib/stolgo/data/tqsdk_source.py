# stolgo agent mistake checklist — docs/IMPLEMENTATION_PLAN_BACKTEST.md §D

"""TqSdk (天勤) market data adapter.

Import as ``from stolgo import TqSdk``. Futures are always served from the
open-interest weighted index ``KQ.i@<EXCHANGE>.<product>`` so backtests have no
contract-roll gaps; stocks are forward-adjusted. How klines are fetched is a
pluggable :class:`KlineFetcher`: free accounts get the most recent bars via
``get_kline_serial``, pro accounts fetch arbitrary ranges via
``get_kline_data_series``.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

import pandas as pd

from stolgo.core.exceptions import DataError
from stolgo.data.cache import ParquetCache
from stolgo.data.normalize import normalize_ohlcv
from stolgo.instrument.base import Future, Instrument, Spot
from stolgo.oms.commission import CommissionModel

FUTURE_EXCHANGES = frozenset({"SHFE", "DCE", "CZCE", "CFFEX", "INE", "GFEX"})
STOCK_EXCHANGES = frozenset({"SSE", "SZSE"})

_DAY_SECONDS = 86_400
_INTERVAL_UNITS = {"s": 1, "m": 60, "h": 3_600, "d": _DAY_SECONDS, "w": 7 * _DAY_SECONDS}
_FUTURE_SYMBOL = re.compile(r"^(?:KQ\.[im]@)?(?P<exchange>[A-Z]+)\.(?P<product>[A-Za-z]+)\d*$")
_PRICE_COLUMNS = ["open", "high", "low", "close"]


def interval_seconds(interval: str) -> int:
    """``"1m"`` → 60, ``"1h"`` → 3600, ``"1d"`` → 86400."""
    match = re.fullmatch(r"(\d+)([smhdw])", interval.strip())
    if match is None:
        raise DataError(f"unsupported interval {interval!r}; use e.g. 1m, 15m, 1h, 1d")
    seconds = int(match.group(1)) * _INTERVAL_UNITS[match.group(2)]
    if seconds <= 0 or (seconds > _DAY_SECONDS and seconds % _DAY_SECONDS):
        raise DataError(f"TqSdk intervals above 1d must be whole days, got {interval!r}")
    return seconds


@dataclass(frozen=True)
class TqSymbol:
    query: str
    kind: Literal["future", "stock"]


def resolve_symbol(symbol: str) -> TqSymbol:
    """Map a user symbol to the series used for backtests.

    ``SHFE.rb``, ``SHFE.rb2501`` and ``KQ.m@SHFE.rb`` all resolve to the weighted
    index ``KQ.i@SHFE.rb``; ``SSE.600000`` is kept as-is.
    """
    match = _FUTURE_SYMBOL.match(symbol)
    if match is not None and match["exchange"] in FUTURE_EXCHANGES:
        return TqSymbol(f"KQ.i@{match['exchange']}.{match['product']}", "future")
    if symbol.split(".", 1)[0] in STOCK_EXCHANGES:
        return TqSymbol(symbol, "stock")
    raise DataError(
        f"unsupported TqSdk symbol {symbol!r}; use a futures product like SHFE.rb "
        "or a stock like SSE.600000"
    )


class KlineFetcher(Protocol):
    name: str

    def fetch(
        self,
        api: Any,
        symbol: str,
        duration: int,
        start: datetime,
        end: datetime,
        adj_type: str | None,
    ) -> pd.DataFrame:
        """Return raw TqSdk klines (``datetime`` in ns plus OHLCV columns)."""
        ...


class RecentKlineFetcher:
    """Free tier: the latest ``max_bars`` bars via ``get_kline_serial``."""

    name = "recent"

    def __init__(self, max_bars: int = 10_000) -> None:
        self._max_bars = max_bars

    def fetch(
        self,
        api: Any,
        symbol: str,
        duration: int,
        start: datetime,
        end: datetime,
        adj_type: str | None,
    ) -> pd.DataFrame:
        raw = api.get_kline_serial(
            symbol, duration, data_length=self._max_bars, adj_type=adj_type
        ).copy()
        raw = raw.dropna(subset=_PRICE_COLUMNS)
        if len(raw) >= self._max_bars and int(raw["datetime"].iloc[0]) > pd.Timestamp(start).value:
            first = pd.Timestamp(int(raw["datetime"].iloc[0]), unit="ns", tz="UTC")
            warnings.warn(
                f"get_kline_serial returns at most {self._max_bars} bars; {symbol} history "
                f"starts at {first}, earlier part of the requested range is missing. "
                "A TqSdk pro account (fetcher='range') can load the full range.",
                stacklevel=3,
            )
        return raw


class RangeKlineFetcher:
    """Pro tier: any time range via ``get_kline_data_series``."""

    name = "range"

    def fetch(
        self,
        api: Any,
        symbol: str,
        duration: int,
        start: datetime,
        end: datetime,
        adj_type: str | None,
    ) -> pd.DataFrame:
        return api.get_kline_data_series(symbol, duration, start, end, adj_type=adj_type)


def default_fetcher(api: Any) -> KlineFetcher:
    # TqSdk has no public tier check; ``tq_dl`` is the flag gating get_kline_data_series.
    has_feature = getattr(getattr(api, "_auth", None), "_has_feature", None)
    if callable(has_feature) and has_feature("tq_dl"):
        return RangeKlineFetcher()
    return RecentKlineFetcher()


_FETCHERS = {"recent": RecentKlineFetcher, "range": RangeKlineFetcher}


def _connect(auth: Any) -> Any:
    if auth is None:
        user = os.environ.get("TQ_USER")
        password = os.environ.get("TQ_PASSWORD")
        if not (user and password):
            raise DataError(
                "TqSdk needs a Shinny (快期) account: pass auth=(user, password) "
                "or set TQ_USER / TQ_PASSWORD"
            )
        auth = (user, password)
    try:
        from tqsdk import TqApi, TqAuth
    except ImportError as exc:
        raise DataError("TqSdk is not installed; pip install 'stolgo[tqsdk]'") from exc
    if isinstance(auth, tuple):
        auth = TqAuth(*auth)
    return TqApi(auth=auth)


def _as_utc(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _to_ohlcv(raw: pd.DataFrame, duration: int) -> pd.DataFrame:
    raw = raw.dropna(subset=_PRICE_COLUMNS)
    # get_kline_serial pads with NaN, so ``datetime`` may arrive as float64 and lose
    # sub-microsecond precision; bars are second-aligned, so round it back.
    ts = pd.to_datetime(raw["datetime"].astype("int64"), unit="ns", utc=True).dt.round("s")
    if duration >= _DAY_SECONDS:
        # Daily+ bars are stamped 00:00 Beijing on the trading day; keep that date
        # rather than letting UTC shift it to the previous calendar day.
        ts = ts.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
        ts = ts.dt.tz_localize("UTC")
    out = raw[_PRICE_COLUMNS].copy()
    out["volume"] = raw["volume"].fillna(0.0)
    out.insert(0, "timestamp", ts)
    return out


class TqSdkDataSource:
    """Historical OHLCV and instrument specs via TqSdk.

    Public alias: :class:`TqSdk`. ``fetcher`` is ``"auto"`` (pick by account
    tier), ``"recent"``, ``"range"`` or a custom :class:`KlineFetcher`.
    """

    def __init__(
        self,
        api: Any = None,
        *,
        auth: Any = None,
        fetcher: KlineFetcher | Literal["auto", "recent", "range"] = "auto",
        cache: ParquetCache | None = None,
        margin_rate: float = 0.15,
        stock_adj_type: Literal["F", "B"] | None = "F",
        stock_lot_size: float = 100,
    ) -> None:
        if isinstance(fetcher, str) and fetcher != "auto" and fetcher not in _FETCHERS:
            raise DataError(f"unknown fetcher {fetcher!r}; use auto, recent or range")
        self._api = api
        self._owns_api = api is None
        self._auth = auth
        self._fetcher_option = fetcher
        self._fetcher: KlineFetcher | None = None
        self._cache = cache or ParquetCache()
        self._margin_rate = margin_rate
        self._stock_adj_type = stock_adj_type
        self._stock_lot_size = stock_lot_size

    @property
    def api(self) -> Any:
        if self._api is None:
            self._api = _connect(self._auth)
        return self._api

    @property
    def fetcher(self) -> KlineFetcher:
        if self._fetcher is None:
            option = self._fetcher_option
            if option == "auto":
                self._fetcher = default_fetcher(self.api)
            elif isinstance(option, str):
                self._fetcher = _FETCHERS[option]()
            else:
                self._fetcher = option
        return self._fetcher

    def history(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        start_utc, end_utc = _as_utc(start), _as_utc(end)
        if start_utc >= end_utc:
            raise DataError("start must be before end")

        resolved = resolve_symbol(symbol)
        duration = interval_seconds(interval)
        adj_type = self._stock_adj_type if resolved.kind == "stock" else None
        fetcher = self.fetcher

        provider_key = f"tqsdk-{fetcher.name}" + (f"-{adj_type}" if adj_type else "")
        key = self._cache.make_key(provider_key, resolved.query, interval, start_utc, end_utc)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        raw = fetcher.fetch(self.api, resolved.query, duration, start_utc, end_utc, adj_type)
        frame = _to_ohlcv(raw, duration)
        frame = frame[(frame["timestamp"] >= start_utc) & (frame["timestamp"] <= end_utc)]
        if frame.empty:
            raise DataError(f"no {resolved.query} bars between {start_utc} and {end_utc}")

        df = normalize_ohlcv(frame, symbol=resolved.query)
        self._cache.put(key, df)
        return df

    def instrument(
        self,
        symbol: str,
        *,
        margin_rate: float | None = None,
        commission: CommissionModel | None = None,
        multiplier: float | None = None,
    ) -> Instrument:
        """Contract spec for :class:`stolgo.Backtest` (multiplier/tick from the quote)."""
        resolved = resolve_symbol(symbol)
        if resolved.kind == "stock":
            return Spot(symbol=resolved.query, lot_size=self._stock_lot_size, commission=commission)

        price_tick = 0.0
        if multiplier is None:
            quote = self.api.get_quote(resolved.query)
            multiplier = float(quote.volume_multiple or 0)
            tick = float(quote.price_tick)
            price_tick = 0.0 if math.isnan(tick) else tick
        if multiplier <= 0:
            raise DataError(
                f"TqSdk returned no contract multiplier for {resolved.query}; pass multiplier="
            )
        return Future(
            symbol=resolved.query,
            multiplier=multiplier,
            margin_rate=margin_rate if margin_rate is not None else self._margin_rate,
            price_tick=price_tick,
            commission=commission,
        )

    def close(self) -> None:
        if self._owns_api and self._api is not None:
            self._api.close()
            self._api = None

    def __enter__(self) -> TqSdkDataSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# Preferred public name (``from stolgo import TqSdk``).
TqSdk = TqSdkDataSource
