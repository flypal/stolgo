from stolgo.core.types import Bar, OrderType, Side
from stolgo.oms.commission import BpsCommission
from stolgo.oms.fill_model import NextOpenFill
from stolgo.oms.sim_broker import SimBroker
from stolgo.oms.slippage import NoSlippage


def test_market_fill_next_open():
    broker = SimBroker(NextOpenFill(), NoSlippage(), BpsCommission(0.0))
    order = broker.create_order("X", Side.BUY, 1.0, OrderType.MARKET)
    broker.submit(order)
    bar = Bar(ts=1, open=10.0, high=11, low=9, close=10.5, volume=1, symbol="X")
    fills = broker.match(bar, bar_index=1)
    assert len(fills) == 1
    assert fills[0].fill.price == 10.0


def _stop_fill_price(side, stop, bar):
    broker = SimBroker(NextOpenFill(), NoSlippage(), BpsCommission(0.0))
    broker.submit(broker.create_order("X", side, 1.0, OrderType.STOP, stop_price=stop))
    fills = broker.match(bar, bar_index=1)
    return fills[0].fill.price if fills else None


def test_stop_fills_at_trigger_or_gap_open():
    bar = Bar(ts=1, open=10.0, high=12, low=9, close=11, volume=1, symbol="X")
    assert _stop_fill_price(Side.BUY, 11.0, bar) == 11.0
    assert _stop_fill_price(Side.SELL, 9.5, bar) == 9.5
    assert _stop_fill_price(Side.BUY, 13.0, bar) is None
    gap_up = Bar(ts=1, open=12.0, high=13, low=11.5, close=12.5, volume=1, symbol="X")
    assert _stop_fill_price(Side.BUY, 11.0, gap_up) == 12.0
